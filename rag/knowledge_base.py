# -*- coding: utf-8 -*-
"""RAG 知识库封装（7.4）：生产知识目录 → 索引 → 检索 → 上下文。

设计要点：
- 构造时**不加载模型、不读取文件**（仅保存参数与状态）；
- 首次 search 时一次性完成索引：扫描文档 → 加载 → 切分 → 编码 → 入库；
  索引与 Embedding 缓存在实例内（重复检索不重复加载/切分/编码）；
- `embedding_factory` 可注入 stub —— 测试可完全不加载真实模型；
- `search()` 返回统一结构，失败可区分阶段、可诊断：
    {"ok": True,  "query", "count", "results":[{source,chunk_id,score,chunk}], "context"}
    {"ok": False, "stage": "init"|"search", "error", "query"}
  空命中是**正常检索结果**（ok=True, count=0），不是异常；
- 初始化失败会缓存错误（后续调用直接返回失败，不反复尝试加载）；
- 不修改任何既有 RAG 模块的接口。
"""

from pathlib import Path
import hashlib
import logging
import os
import stat
import threading

from doc_formats import GROUP_ORDER, is_supported_suffix, suffix_of

from rag.context_builder import ContextBuilder
from rag.loader import get_loader
from rag.retriever import Retriever
from rag.splitter import TextSplitter
from rag.vector_store import VectorStore


logger = logging.getLogger(__name__)


# document_id 的十六进制长度。必须与上传层
# ``knowledge_store.DOCUMENT_ID_HEX_LENGTH`` 保持一致（跨层一致性由集成测试
# 逐字节比对锁定，不在代码里互相 import）。
_DOCUMENT_ID_HEX_LENGTH = 32


# 检索结果需要透传的来源字段（B-2）。
#
# 语义：**文档上有就透传，没有就不出现**。PDF 的 page/page_count、DOCX 的
# section/section_index/segment_index/tables 的 table_index 只在对应格式上
# 存在；缺失时不能补 None，否则调用方无法区分「该格式没有这个概念」与
# 「取到了空值」。document_id / chunk_seq 由本层在索引期写入，四种格式都有。
_SOURCE_METADATA_KEYS = (
    "document_id",
    "chunk_seq",
    "page",
    "page_count",
    "section",
    "section_index",
    "segment_index",
    "table_index",
)


def _is_reparse_point(st):
    """Windows 重解析点判定（覆盖 symlink / junction / 其它 tag）。

    非 Windows 平台没有 ``st_file_attributes``，此处恒为 False；
    符号链接由 ``entry.is_file(follow_symlinks=False)`` 与 ``S_ISREG`` 兜底。
    """
    attribute = getattr(st, "st_file_attributes", 0)
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(flag) and bool(attribute & flag)


def _iter_knowledge_files(doc_dir):
    """枚举顶层受支持后缀的普通文件，返回 ``([(Path, (dev, ino))], reparse, unstable)``。

    受支持后缀的唯一来源是 ``doc_formats.SUPPORTED_SUFFIXES``（顺序见 ``GROUP_ORDER``）。
    对 ``.md`` / ``.txt`` 而言，与原先 ``glob("*.md") + glob("*.txt")`` 的口径等价
    （仅顶层、大小写不敏感、仍匹配点开头的文件名、md 组在前 txt 组在后且各自排序），
    但额外：
    - 不跟随符号链接 / junction / 其它重解析点；
    - 只接受普通文件（目录、设备、管道一律排除）；
    - 记录 ``(st_dev, st_ino)``，供打开前复核身份使用。

    身份**必须**取自 ``os.lstat``（与 L2 同源），不得使用
    ``DirEntry.stat(follow_symlinks=False)``：Windows 上后者复用目录扫描缓存
    （``WIN32_FIND_DATA``），该结构不含文件索引与卷序列号，``st_dev`` /
    ``st_ino`` 恒为 0，与 L2 的 ``os.lstat`` 真实值比较必然不等，
    会导致每个正常文件都被误判为「身份已变化」而全部跳过。
    """
    groups = {group_suffix: [] for group_suffix in GROUP_ORDER}
    skipped_reparse = 0
    skipped_unstable = 0

    with os.scandir(doc_dir) as entries:
        for entry in entries:
            suffix = suffix_of(entry.name)
            if not is_supported_suffix(suffix):
                continue
            try:
                # 身份来源与 L2 的复核同源（均为 os.lstat，不跟随重解析点）。
                # 不能用 entry.stat(follow_symlinks=False)：Windows 上它读的是
                # 目录扫描缓存，st_dev/st_ino 恒为 (0, 0)，与 L2 比较必然不等。
                st = os.lstat(entry.path)
            except OSError:
                skipped_unstable += 1
                continue
            # 先判重解析点、再判类型：避免符号链接被同时归入「非普通文件」，
            # 导致统计口径混乱。
            if _is_reparse_point(st):
                skipped_reparse += 1
                logger.warning(
                    "知识文档是重解析点，已跳过（不读取目标）：%s", entry.name
                )
                continue
            if not stat.S_ISREG(st.st_mode):
                skipped_unstable += 1
                continue
            item = (Path(entry.path), (st.st_dev, st.st_ino))
            groups[suffix].append(item)

    # 分组顺序由 GROUP_ORDER 决定（MD 在 TXT 前），与阶段一逐字节一致；
    # 新增格式只会追加在末尾，不会改变既有 md/txt 的相对顺序。
    ordered = []
    for group_suffix in GROUP_ORDER:
        ordered.extend(sorted(groups[group_suffix]))

    return ordered, skipped_reparse, skipped_unstable


class KnowledgeBase:

    def __init__(self, doc_dir, embedding_factory=None):
        self.doc_dir = Path(doc_dir)
        self._embedding_factory = embedding_factory or _default_embedding_factory

        self._embedding = None
        self._retriever = None
        self._ready = False
        self._init_error = None
        # 保护首次索引初始化：同一实例的并发首次检索只允许一个
        # 初始化流程实际执行（锁仅覆盖初始化，不覆盖查询编码）。
        self._init_lock = threading.Lock()

        # 索引统计（供测试与诊断使用）
        self.file_count = 0
        self.chunk_count = 0
        # B1-04：解析成功但没有任何有效 chunk 的文件数（空文件 / 空 DOCX /
        # 全空白文本）。与 skipped_count（加载或编码失败）**互斥**：
        # 同一文件只会落入 loaded / skipped / empty 三者之一。
        self.empty_file_count = 0

        # 索引完整性诊断（N1：供检索状态判定使用，不改变 search() 返回结构）
        # skipped_count：索引期加载/切分/编码失败而被跳过的文件数。
        # init_error_kind：初始化失败类别——
        #   "dir_missing"     知识目录不存在或不是目录
        #   "dir_no_files"    目录存在，但没有 .md/.txt 可索引
        #   "all_unparsable"  有文件，但全部无法加载或解析
        #   "init_error"      其它初始化异常（如 Embedding 加载失败）
        # 初始化成功后保持 None。这些属性在首次初始化时一次性写入，
        # 之后不再变化，因此被多个请求共享也不会互相污染。
        self.skipped_count = 0
        # 阶段一新增：把「被安全过滤排除」与「加载/切分/编码失败」分开统计。
        # skipped_reparse_count：因重解析点被排除（枚举期或打开前复核）。
        # skipped_unstable_count：因非普通文件、身份变化或 stat 失败被排除。
        # skipped_count 的既有语义保持不变（仅统计加载/切分/编码失败）。
        self.skipped_reparse_count = 0
        self.skipped_unstable_count = 0
        self.init_error_kind = None

    # ------------------------------------------------------------
    # 单文件加载与暂存（B1-01 ~ B1-05）
    # ------------------------------------------------------------

    @staticmethod
    def _compute_document_id(filename):
        """由文件名派生文档标识：``sha256(filename)[:32]``。

        与上传层 ``knowledge_store.compute_document_id`` 算法**完全一致**。
        这里刻意不 import 该模块：``knowledge_store`` 属于 HTTP/上传层，
        ``rag`` 层反向依赖会造成层次倒置。两层一致性不靠注释保证，而是由
        ``test_knowledge_base_loader_integration.py`` 逐字节比对锁定。

        标识只由**文件名**（不含目录）派生，因此与上传层的口径一致：
        同一物理文件只有一个 id，改名即换 id。
        """
        if not isinstance(filename, str):
            raise TypeError("filename 必须是字符串")
        return hashlib.sha256(
            filename.encode("utf-8")
        ).hexdigest()[:_DOCUMENT_ID_HEX_LENGTH]

    def _stage_file(self, file_path, splitter, embedding):
        """加载单个文件，返回**尚未提交**的 ``[(chunk, vector), ...]``。

        为什么必须暂存而不是边切边写：``VectorStore`` 只有 append 语义
        （``vector_store.py:10``），既没有 remove 也没有事务，一旦写进去
        就撤不回来。若逐块写入，第 N 个 Document 编码失败时，前 N-1 个
        Document 的 chunk 会永久留在索引里，形成「半截文件」——既查得到
        残片、又统计不出异常（B1-05）。

        因此本方法把「解析 → 切块 → 过滤 → 编号 → 编码」全部在**本地
        列表**完成，任何一步抛出异常都直接向上传播；调用方拿到异常时，
        该文件的零个 chunk 进入过全局库。

        返回空列表表示「解析成功但没有任何有效 chunk」（如空 DOCX、
        空 TXT、全空白文本），调用方据此计 ``empty_file_count`` 而**不是**
        ``loaded_count``（B1-04 / N4）。

        编号规则（B1-03）：
        - ``chunk_id``：Splitter 生成的 Document 内序号，原样保留；
        - ``chunk_seq``：本文档内的全局连续序号，跨所有 Loader Document
          递增，每个文件从 0 重新开始；
        - 空白 chunk 先被过滤，**再**分配 ``chunk_seq``，保证序号无空洞。
        """
        # 与后缀白名单同源：Loaders 由 doc_formats.SUPPORTED_SUFFIXES 驱动，
        # 此处不重复维护任何后缀字面量（B1-01）。
        loader = get_loader(str(file_path))
        documents = loader.load()

        document_id = self._compute_document_id(file_path.name)
        staged = []
        chunk_seq = 0

        for document in documents:
            for chunk in splitter.split(document):
                # 空白 chunk 不进索引：既避免零向量污染检索，
                # 也让 chunk_seq 保持连续。
                if not chunk.page_content.strip():
                    continue

                # source 维持本层既有的 basename 语义（B1-06 要求不变）；
                # 其余 Loader 原始键（page/page_count/section/segment_index/
                # table_index/...）全部原样保留，不做任何白名单过滤。
                chunk.metadata["source"] = file_path.name
                chunk.metadata["document_id"] = document_id
                chunk.metadata["chunk_seq"] = chunk_seq

                vector = embedding.encode(chunk.page_content)
                staged.append((chunk, vector))
                chunk_seq += 1

        return staged

    # ------------------------------------------------------------
    # 索引构建（惰性、一次性）
    # ------------------------------------------------------------

    def _ensure_index(self):
        if self._ready:
            return

        if self._init_error is not None:
            # 初始化已失败过：不再反复尝试加载，直接复现失败信息
            raise RuntimeError(self._init_error)

        with self._init_lock:
            # 双重检查：等待锁期间，初始化可能已被其他线程完成或已失败
            if self._ready:
                return

            if self._init_error is not None:
                raise RuntimeError(self._init_error)

            try:
                # 「目录不存在」与「目录存在但没有文件」是两种不同状态：
                # 前者属于部署/配置异常（上层判定为 degraded），
                # 后者才是知识库本身为空（上层判定为 kb_empty）。
                # 两者的错误文案都保留原文「知识目录缺失或为空」，
                # 以免破坏既有测试对该消息的断言，仅在括号内补充细节。
                #
                # dir_untrusted 是唯一**不缓存** _init_error 的类别：
                # 目录被修复后应允许重新尝试初始化，因此重入时先清零类别。
                # 其余类别一旦失败就会写入 _init_error，根本不会重入到这里。
                if self.init_error_kind == "dir_untrusted":
                    self.init_error_kind = None

                # 用 lstat（不跟随）判定叶子目录：exists()/is_dir() 会跟随
                # 重解析点，悬空链接还会被误判为不存在而绕过信任检查。
                try:
                    dir_stat = os.lstat(self.doc_dir)
                except FileNotFoundError:
                    self.init_error_kind = "dir_missing"
                    raise FileNotFoundError(
                        f"知识目录缺失或为空：{self.doc_dir}"
                        "（目录不存在或不是目录）"
                    ) from None
                except NotADirectoryError:
                    # 祖先路径是文件：与旧 exists()==False 的行为保持一致。
                    self.init_error_kind = "dir_missing"
                    raise FileNotFoundError(
                        f"知识目录缺失或为空：{self.doc_dir}"
                        "（目录不存在或不是目录）"
                    ) from None
                except PermissionError:
                    # 存在但不可访问：不再谎报为「缺失或为空」。
                    self.init_error_kind = "init_error"
                    raise RuntimeError(
                        f"知识目录无法访问：{self.doc_dir}（PermissionError）"
                    ) from None
                except OSError as exc:
                    self.init_error_kind = "init_error"
                    raise RuntimeError(
                        f"知识目录无法访问：{self.doc_dir}"
                        f"（{type(exc).__name__}）"
                    ) from exc

                if _is_reparse_point(dir_stat):
                    self.init_error_kind = "dir_untrusted"
                    raise RuntimeError(
                        "知识目录是重解析点（junction/符号链接），"
                        f"拒绝构建索引：{self.doc_dir}"
                    )

                if not stat.S_ISDIR(dir_stat.st_mode):
                    self.init_error_kind = "dir_missing"
                    raise FileNotFoundError(
                        f"知识目录缺失或为空：{self.doc_dir}"
                        "（目录不存在或不是目录）"
                    )

                files, skipped_reparse, skipped_unstable = _iter_knowledge_files(
                    self.doc_dir
                )

                if not files:
                    self.init_error_kind = "dir_no_files"
                    raise FileNotFoundError(
                        f"知识目录缺失或为空：{self.doc_dir}"
                        "（目录内没有 .md/.txt 文件）"
                    )

                embedding = self._embedding_factory()
                store = VectorStore()
                splitter = TextSplitter()

                chunk_total = 0
                loaded_count = 0
                empty_file_count = 0
                # 每次索引尝试从零统计（初始化只会成功一次或缓存失败，
                # 此处重置可避免极端重入下重复累加）。
                self.skipped_count = 0
                self.skipped_reparse_count = skipped_reparse
                self.skipped_unstable_count = skipped_unstable
                for file_path, identity in files:
                    # L2：打开前复核「仍是普通文件 / 不是重解析点 / 身份未变」。
                    # 任一条件不满足即跳过，绝不调用 Loader 打开文件。
                    # 注意：本复核与 Loader 内部的 open() 之间仍存在
                    # 未量化的 TOCTOU 窗口，此处不声称消除竞态。
                    try:
                        current_stat = os.lstat(file_path)
                    except OSError:
                        self.skipped_unstable_count += 1
                        logger.warning(
                            "知识文档在枚举后无法 stat，已跳过：%s", file_path.name
                        )
                        continue
                    if _is_reparse_point(current_stat):
                        self.skipped_reparse_count += 1
                        logger.warning(
                            "知识文档在枚举后变成重解析点，已跳过：%s", file_path.name
                        )
                        continue
                    if not stat.S_ISREG(current_stat.st_mode):
                        self.skipped_unstable_count += 1
                        logger.warning(
                            "知识文档在枚举后不再是普通文件，已跳过：%s", file_path.name
                        )
                        continue
                    if (current_stat.st_dev, current_stat.st_ino) != identity:
                        self.skipped_unstable_count += 1
                        logger.warning(
                            "知识文档在枚举后身份已变化，已跳过：%s", file_path.name
                        )
                        continue

                    try:
                        # 文件级暂存：解析 / 切块 / 过滤 / 编号 / 编码全部
                        # 在本地列表完成，本调用返回前不会有任何 chunk
                        # 进入 store（B1-05）。
                        staged = self._stage_file(
                            file_path=file_path,
                            splitter=splitter,
                            embedding=embedding,
                        )
                    except Exception as exc:
                        # 单个文档加载/切分/编码失败不拖垮整个索引：
                        # 记录并跳过该文件，保留其余可解析文档。
                        # 此时该文件的 chunk **一个都没有**提交过。
                        self.skipped_count += 1
                        logger.warning(
                            "知识文档加载失败，已跳过：%s（%s: %s）",
                            file_path.name,
                            type(exc).__name__,
                            exc,
                        )
                        continue

                    if not staged:
                        # 解析成功但没有任何有效 chunk（空文件 / 空 DOCX /
                        # 全空白文本）：既不算加载成功，也不算加载失败。
                        # loaded / skipped / empty 三者互斥，同一文件
                        # 只会落入其中一个计数器（B1-04 / N4）。
                        empty_file_count += 1
                        logger.info(
                            "知识文档无有效文本，未计入索引：%s", file_path.name
                        )
                        continue

                    # 该文件的全部 chunk 与向量都已就绪才统一提交：
                    # VectorStore 无删除/回滚能力，提前写入就会留下
                    # 无法清理的半截索引。
                    for chunk, vector in staged:
                        store.add(chunk, vector)
                    chunk_total += len(staged)
                    loaded_count += 1

                self.empty_file_count = empty_file_count

                if loaded_count == 0:
                    self.init_error_kind = "all_unparsable"
                    raise RuntimeError("所有知识文档均无法加载或解析")

                self._embedding = embedding
                self._retriever = Retriever(store)
                self.file_count = loaded_count
                self.chunk_count = chunk_total
                self._ready = True

            except Exception as exc:
                # 兜底类别：仅当更具体的类别（目录缺失 / 无文件 / 全部不可解析）
                # 尚未写入时，才标记为通用初始化异常（如 Embedding 加载失败）。
                if self.init_error_kind is None:
                    self.init_error_kind = "init_error"
                # dir_untrusted 不写 _init_error：否则目录修好后会被快速失败
                # 路径永久复用同一错误，无法重新尝试初始化。
                # 其余类别保持原有「失败只加载一次」的缓存语义不变。
                if self.init_error_kind != "dir_untrusted":
                    self._init_error = f"{type(exc).__name__}: {exc}"
                raise

    # ------------------------------------------------------------
    # 索引状态（只读，绝不触发构建）
    # ------------------------------------------------------------

    @property
    def is_indexed(self):
        """索引是否已在本实例构建成功；只读属性，不触发任何构建动作。"""
        return self._ready

    def indexed_chunk_counts(self):
        """返回 ``{source: chunk_count}``；索引未构建时返回 None。

        None 与 ``{}`` 语义不同：
        - None：索引不存在 → 调用方应报 ``unknown``（不得假装已索引）；
        - {}：索引存在但没有任何记录 → 才可能报 ``not_indexed``。
        """
        if not self._ready or self._retriever is None:
            return None

        counts = {}
        store_items = getattr(self._retriever.vector_store, "items", None)
        if not isinstance(store_items, list):
            return None
        for item in store_items:
            document = item.get("document") if isinstance(item, dict) else None
            metadata = getattr(document, "metadata", None) or {}
            source = metadata.get("source")
            if source is None:
                continue
            counts[source] = counts.get(source, 0) + 1
        return counts

    # ------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------

    def search(self, query, top_k=3, threshold=None):
        """检索知识库。

        threshold 语义：
        - 默认 None：过滤相似度恰好为 0.0 的结果（不计为有效命中），
          但不引入 0.3 / 0.5 等未经验证的默认语义阈值；
        - 显式传入（含 0.0）：完全沿用 Retriever 的 >= 比较语义。
        """

        if not isinstance(query, str) or not query.strip():
            return {
                "ok": False,
                "stage": "input",
                "error": "知识检索问题不能为空",
                "query": query,
            }
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 10:
            return {
                "ok": False,
                "stage": "input",
                "error": "top_k 必须是 1 到 10 之间的整数",
                "query": query,
            }

        try:
            self._ensure_index()
        except Exception as exc:
            return {
                "ok": False,
                "stage": "init",
                "error": f"{type(exc).__name__}: {exc}",
                "query": query,
            }

        try:
            query_vector = self._embedding.encode(query)
            raw = self._retriever.retrieve(
                query_vector,
                top_k=top_k,
                threshold=threshold,
            )

            # 默认（threshold=None）时过滤掉零相似度结果：只有
            # 检索成功但没有有效结果，才属于空命中（ok=True, count=0）。
            if threshold is None:
                raw = [item for item in raw if item["score"] > 0.0]

            results = []
            for item in raw:
                metadata = item["document"].metadata or {}
                record = {
                    "source": metadata.get("source"),
                    "chunk_id": metadata.get("chunk_id"),
                    "score": float(item["score"]),
                    "chunk": item["document"].page_content,
                }
                # 来源元数据透传（B-2）：只搬运真实存在的键，不做白名单裁剪、
                # 不重新构造仅含旧四字段的对象、也不为缺失字段补默认值。
                for key in _SOURCE_METADATA_KEYS:
                    if key in metadata:
                        record[key] = metadata[key]
                results.append(record)

            return {
                "ok": True,
                "query": query,
                "count": len(results),
                "results": results,
                "context": ContextBuilder().build(raw),
            }

        except Exception as exc:
            return {
                "ok": False,
                "stage": "search",
                "error": f"{type(exc).__name__}: {exc}",
                "query": query,
            }


def _default_embedding_factory():
    """默认工厂：真正需要构建索引时才导入并加载真实 Embedding。

    延迟 import 使构造 KnowledgeBase（以及 import 本模块）完全不触发
    sentence-transformers 的模型加载。
    """
    from rag.embedding import Embedding
    return Embedding()
