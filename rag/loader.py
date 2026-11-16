import logging

from rag.document import Document


logger = logging.getLogger(__name__)


class TextLoader:
    """
    文本文件加载器
    """

    def __init__(self, file_path: str):
        self.file_path = file_path


    def load(self):

        with open(
            self.file_path,
            "r",
            encoding="utf-8"
        ) as f:
            text = f.read()


        document = Document(
            page_content=text,
            metadata={
                "source": self.file_path
            }
        )


        return [document]


# ============================================================
# 阶段二 P0-2 批次 A：多格式 Loader
#
# 设计约束（与任务书一致）：
# - ``TextLoader`` 一字未改，既有 20+ 处 ``from rag.loader import TextLoader``
#   的调用方（含全部评估脚本）行为完全不变；
# - ``pypdf`` / ``python-docx`` 均为**延迟导入**（放在 load() 内），
#   使 ``import rag.loader`` 本身不依赖这两个三方包 —— 与
#   ``knowledge_base._default_embedding_factory`` 的延迟导入风格一致；
# - 异常按「加密 / 损坏 / 无文本层」分列，不合并为同一类；
# - DOCX 不产生页码字段，绝不伪造。
# ============================================================


# ------------------------------------------------------------
# 异常类型
# ------------------------------------------------------------
class LoaderError(Exception):
    """Loader 层异常基类。"""


class UnsupportedSuffixError(LoaderError):
    """后缀不在 ``doc_formats.SUPPORTED_SUFFIXES`` 内，无法选择 Loader。"""

    def __init__(self, suffix, file_path=None):
        self.suffix = suffix
        self.file_path = file_path
        super().__init__(
            f"不支持的文件后缀：{suffix!r}（文件：{file_path}）"
        )


class PdfLoadError(LoaderError):
    """PDF 加载异常基类。"""


class EncryptedPdfError(PdfLoadError):
    """PDF 已加密，且空密码无法解密。"""


class CorruptPdfError(PdfLoadError):
    """PDF 损坏、为空或结构不可解析。"""


def _page_diagnostics(failed_records):
    """把 ``[(page, exc), ...]`` 转成稳定的结构化诊断列表。

    每个元素固定包含三个键，使调用方**不必解析异常文案**即可区分
    「纯空白」与「混合故障」：

        {"page": int, "error_type": str, "message": str}
    """
    return [
        {
            "page": page,
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
        for page, exc in failed_records
    ]


class NoTextLayerError(PdfLoadError):
    """PDF 没有任何可提取的文本。

    除消息外，本异常暴露两个**结构化诊断属性**（稳定、可直接读取）：

    - ``empty_pages``：提取成功但内容为空的页码列表（``list[int]``，1 基）；
    - ``failed_pages``：提取失败的页面诊断列表（``list[dict]``，元素含
      ``page`` / ``error_type`` / ``message``）。

    区分规则：纯空白 PDF 的 ``failed_pages`` 恒为空列表；混合故障
    （部分页为空、部分页提取异常）的 ``failed_pages`` 非空。因此调用方
    仅凭属性即可判定，不再依赖是否有人去读异常文案。
    """

    def __init__(self, message, empty_pages=None, failed_pages=None):
        super().__init__(message)
        self.empty_pages = list(empty_pages or [])
        self.failed_pages = list(failed_pages or [])


class PdfPageExtractionError(PdfLoadError):
    """PDF 全部页面的文本提取均抛出异常。

    与 ``NoTextLayerError`` 严格区分：本类是**提取失败**（内容流损坏、
    字体资源缺失等），不是「本来就没有文本」。若混为一类，运维会把
    解析故障误判为扫描件，从而放弃本可修复的文件。

    同样暴露 ``empty_pages`` / ``failed_pages`` 结构化诊断；首个原始页面
    异常由抛出点的 ``raise ... from`` 保留在 ``__cause__`` 上，可直接取回
    原始异常对象（不是被转成字符串的副本）。
    """

    def __init__(self, message, empty_pages=None, failed_pages=None):
        super().__init__(message)
        self.empty_pages = list(empty_pages or [])
        self.failed_pages = list(failed_pages or [])


class DocxLoadError(LoaderError):
    """DOCX 损坏、非 OOXML 包或结构不可解析。"""


# ------------------------------------------------------------
# PDF
# ------------------------------------------------------------
class PdfLoader:
    """逐页加载 PDF，每页产出一个独立 ``Document``。

    metadata 固定包含 ``source`` / ``page``（1 基）/ ``page_count``（总页数）。
    部分页面为空或提取失败时**跳过该页但不失败**，只在整份文档拿不到任何
    有效文本时才抛异常，且两类原因分属**互不继承**的异常：

    - 全部页面**文本提取均失败**（内容流损坏、字体资源缺失等解析故障）
      → ``PdfPageExtractionError``，首例原始异常挂在 ``__cause__`` 上；
    - 无有效文本但**不属于**全页提取失败（纯空白扫描件，或空白页与故障页
      混合且无成功页）→ ``NoTextLayerError``，不建立 ``__cause__``。

    两者同属 ``PdfLoadError``，但互相之间没有继承关系：把解析故障误判为
    扫描件会让本可修复的文件被直接放弃，因此不允许 ``except`` 互相兜住。
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self):
        # 延迟导入：未安装 pypdf 时只有真正使用 PDF 才会报错。
        try:
            from pypdf import PdfReader
            from pypdf.errors import PdfReadError
        except ImportError as exc:  # pragma: no cover - 环境缺依赖
            raise PdfLoadError(
                "缺少 pypdf 依赖，无法解析 PDF"
            ) from exc

        try:
            reader = PdfReader(self.file_path, strict=False)
        except FileNotFoundError:
            # 与 TextLoader 的 open() 行为保持一致：文件不存在直接透传，
            # 不包装成「损坏」——否则运维无法区分「文件被删」与「文件坏了」。
            raise
        except Exception as exc:
            # 损坏 / 空文件 / 结构非法：统一归为 CorruptPdfError，
            # 但保留原始异常链以便诊断。
            raise CorruptPdfError(
                f"PDF 无法解析（{type(exc).__name__}）：{self.file_path}"
            ) from exc

        if getattr(reader, "is_encrypted", False):
            # 仅尝试空密码：不猜测、不暴力破解。
            try:
                decrypted = reader.decrypt("")
            except Exception as exc:
                raise EncryptedPdfError(
                    f"PDF 已加密且无法用空密码解密：{self.file_path}"
                ) from exc
            if not decrypted:
                raise EncryptedPdfError(
                    f"PDF 已加密且无法用空密码解密：{self.file_path}"
                )

        try:
            pages = list(reader.pages)
        except Exception as exc:
            raise CorruptPdfError(
                f"PDF 页面结构不可读（{type(exc).__name__}）：{self.file_path}"
            ) from exc

        page_count = len(pages)
        if page_count == 0:
            raise CorruptPdfError(f"PDF 不含任何页面：{self.file_path}")

        documents = []
        # 空白页与异常页**必须分开统计**：前者是内容事实（该页确实没字），
        # 后者是解析故障。混在一个列表里会让日志无法区分两者。
        empty_pages = []      # 提取成功，但文本为空
        failed_records = []   # 提取抛异常，元素为 (page, 原始异常对象)

        for index, page in enumerate(pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as exc:
                # 单页提取失败不拖垮整份文档。此处保存的是**原始异常对象**
                # 而不是它的字符串副本，供 PdfPageExtractionError 通过
                # ``raise ... from`` 建立异常链（__cause__）。
                failed_records.append((index, exc))
                logger.warning(
                    "PDF 第 %d 页文本提取异常，已跳过：%s（%s: %s）",
                    index,
                    self.file_path,
                    type(exc).__name__,
                    exc,
                )
                continue

            if not text.strip():
                empty_pages.append(index)
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": self.file_path,
                        "page": index,
                        "page_count": page_count,
                    },
                )
            )

        failed_count = len(failed_records)
        failed_diagnostics = _page_diagnostics(failed_records)

        if not documents:
            # 规则 2：全部页都抛异常 → 解析故障，绝不能说成「扫描件」。
            if failed_count == page_count:
                first_page, first_exc = failed_records[0]
                raise PdfPageExtractionError(
                    f"PDF 全部 {page_count} 页文本提取均失败"
                    f"（首例：第 {first_page} 页 "
                    f"{type(first_exc).__name__}: {first_exc}）"
                    f"：{self.file_path}",
                    empty_pages=empty_pages,
                    failed_pages=failed_diagnostics,
                ) from first_exc

            # 规则 4：部分页为空、部分页异常，且无任何有效文本。
            # 按「无有效文本」归类，但诊断必须结构化可读（不能只写在
            # 消息里）：empty_pages / failed_pages 两个属性承载全部信息，
            # 调用方据此即可区分「纯空白」与「混合故障」。
            #
            # 此处不建立 __cause__：归类依据是「没有可用文本」而不是
            # 「被某个页异常触发」，挂上异常链会误导调用方。
            if failed_count:
                failed_numbers = ",".join(
                    str(record[0]) for record in failed_records
                )
                raise NoTextLayerError(
                    f"PDF 共 {page_count} 页，无任何有效文本"
                    f"（空页 {len(empty_pages)} 页；"
                    f"提取异常 {failed_count} 页，页码 {failed_numbers}）"
                    f"：{self.file_path}",
                    empty_pages=empty_pages,
                    failed_pages=failed_diagnostics,
                )

            # 规则 1：全部页提取成功，但文本均为空。
            # failed_pages 显式传空列表：纯空白 PDF 必须能仅凭属性识别。
            raise NoTextLayerError(
                f"PDF 全部 {page_count} 页均无可提取文本"
                f"（疑似扫描件，本阶段不做 OCR）：{self.file_path}",
                empty_pages=empty_pages,
                failed_pages=[],
            )

        # 规则 3：存在有效文本 → 保留成功页，异常页仅记录诊断。
        if empty_pages:
            logger.info(
                "PDF 共 %d 页，空白页 %d 页（页码 %s）：%s",
                page_count,
                len(empty_pages),
                ",".join(str(number) for number in empty_pages),
                self.file_path,
            )
        if failed_records:
            logger.warning(
                "PDF 共 %d 页，提取异常 %d 页（页码 %s）：%s",
                page_count,
                failed_count,
                ",".join(str(record[0]) for record in failed_records),
                self.file_path,
            )

        return documents


# ------------------------------------------------------------
# DOCX
# ------------------------------------------------------------
class DocxLoader:
    """按 OOXML ``w:body`` 子元素原始顺序解析 DOCX。

    产出的 Document 分两类：

    - **节级 Document**：以标题为界切节，metadata 含
      ``source`` / ``section`` / ``section_index`` / ``heading_level``
      / ``start_paragraph_index`` / ``end_paragraph_index``；
    - **表格 Document**：每个 ``w:tbl`` 独立一个，metadata 含
      ``table_index`` / ``table_rows`` / ``table_cols`` / ``table_ragged``
      / ``body_position`` / ``after_paragraph_index``。

    **不产生 page / page_count**：DOCX 没有页概念，编号即为伪造。
    """

    # 英文内建样式 "Heading 1"，中文版 Word 保存的样式名亦可能是 "标题 1"。
    _HEADING_RE = None

    @classmethod
    def _heading_level(cls, paragraph):
        """返回标题级别 1-9；正文返回 0。解析失败一律按正文处理，不猜测。"""
        import re

        style = getattr(paragraph, "style", None)
        name = getattr(style, "name", "") or ""
        name = name.strip()

        if cls._HEADING_RE is None:
            cls._HEADING_RE = re.compile(
                r"^(?:heading|标题)\s*([1-9])$", re.IGNORECASE
            )
        matched = cls._HEADING_RE.match(name)
        if matched:
            return int(matched.group(1))

        # 兜底：部分文档用 outlineLvl 表达大纲级别（0 基）。
        try:
            from docx.oxml.ns import qn

            p_pr = paragraph._p.pPr
            if p_pr is not None:
                outline = p_pr.find(qn("w:outlineLvl"))
                if outline is not None:
                    value = outline.get(qn("w:val"))
                    if value is not None and value.isdigit():
                        level = int(value)
                        if 0 <= level <= 8:
                            return level + 1
        except Exception:
            pass

        return 0

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self):
        # 延迟导入，理由同 PdfLoader。
        try:
            import docx
            from docx.oxml.ns import qn
            from docx.table import Table
            from docx.text.paragraph import Paragraph
        except ImportError as exc:  # pragma: no cover - 环境缺依赖
            raise DocxLoadError(
                "缺少 python-docx 依赖，无法解析 DOCX"
            ) from exc

        try:
            document = docx.Document(self.file_path)
        except FileNotFoundError:
            # 同 PdfLoader：文件不存在透传，不伪装成结构错误。
            raise
        except Exception as exc:
            # python-docx 对不存在的路径抛 PackageNotFoundError（它用
            # is_zipfile() 判断，不区分「不存在」与「不是 zip」）。
            # 若不还原，调用方就无法区分「文件被删」与「文件损坏」。
            # 这里按文件系统事实还原为 FileNotFoundError，与 TextLoader
            # 的 open() 语义保持一致。
            import os as _os

            if not _os.path.exists(self.file_path):
                raise FileNotFoundError(
                    f"文件不存在：{self.file_path}"
                ) from exc
            raise DocxLoadError(
                f"DOCX 无法解析（{type(exc).__name__}）：{self.file_path}"
            ) from exc

        try:
            body = document.element.body
        except Exception as exc:
            raise DocxLoadError(
                f"DOCX 缺少可读的正文节点：{self.file_path}"
            ) from exc

        # ---- 第一遍：按 body 子元素原始顺序，收集段落与表格 ----
        # 段落索引只数 w:p（不含表格），表格用 body_position 与
        # after_paragraph_index 锚定，两套索引互不污染。
        items = []
        paragraph_index = 0
        table_index = 0
        body_position = 0

        for child in body.iterchildren():
            if child.tag == qn("w:p"):
                try:
                    paragraph = Paragraph(child, document)
                    text = paragraph.text
                    level = self._heading_level(paragraph)
                except Exception as exc:
                    logger.warning(
                        "DOCX 段落解析失败，按空段落处理：%s（%s: %s）",
                        self.file_path,
                        type(exc).__name__,
                        exc,
                    )
                    text = ""
                    level = 0

                items.append(
                    {
                        "kind": "paragraph",
                        "text": text,
                        "level": level,
                        "paragraph_index": paragraph_index,
                        "body_position": body_position,
                    }
                )
                paragraph_index += 1

            elif child.tag == qn("w:tbl"):
                # table_index 表示「这是第几个 w:tbl」（0 基），与 body 顺序
                # 严格对应：即使该表格解析失败或被判定为空，也必须占用编号，
                # 否则后续表格的编号会与真实位置错位。
                this_table_index = table_index
                table_index += 1

                try:
                    table = Table(child, document)
                    rows = [list(row.cells) for row in table.rows]
                    cells = [
                        [cell.text for cell in row]
                        for row in rows
                    ]
                    row_count = len(cells)
                    col_count = max((len(row) for row in cells), default=0)
                    ragged = len({len(row) for row in cells}) > 1
                except Exception as exc:
                    logger.warning(
                        "DOCX 表格解析失败，已跳过：%s（%s: %s）",
                        self.file_path,
                        type(exc).__name__,
                        exc,
                    )
                    body_position += 1
                    continue

                if row_count == 0 or col_count == 0:
                    # 空表格不产出 Document，避免污染向量库。
                    logger.info(
                        "DOCX 空表格已跳过（table_index=%d）：%s",
                        this_table_index,
                        self.file_path,
                    )
                    body_position += 1
                    continue

                items.append(
                    {
                        "kind": "table",
                        "cells": cells,
                        "row_count": row_count,
                        "col_count": col_count,
                        "ragged": ragged,
                        "table_index": this_table_index,
                        "body_position": body_position,
                        "after_paragraph_index": paragraph_index,
                    }
                )

            body_position += 1

        # ---- 第二遍：按原始 XML 顺序生成「有序内容流」 ----
        # 关键：不采用「事后再按 body_position 排序」的做法。段落与表格的
        # 交错位置必须在生成阶段固化，否则同一节的段落会被整体挪到表格
        # 之后，调用方拿到的列表顺序与文档真实顺序不一致。
        #
        # 表格会**打断当前片段**（flush），但**不切换节**：被打断后的段落
        # 仍属同一节，仅 segment_index 递增。
        ordered = []          # 元素为 segment / table，顺序即 body 顺序
        current = None
        section_uid_counter = 0

        def _flush_segment():
            """把当前累积的段落片段追加为 ordered 中的一个 segment。"""
            if current is None or not current["paragraphs"]:
                return
            ordered.append(
                {
                    "kind": "segment",
                    "section_uid": current["uid"],
                    "section": current["section"],
                    "heading_level": current["heading_level"],
                    "paragraphs": current["paragraphs"],
                }
            )
            # 清空片段但保留节身份（uid/section/heading_level 不变）。
            current["paragraphs"] = []

        for item in items:
            if item["kind"] == "paragraph" and item["level"] > 0:
                # 新标题：结束上一节的最后片段，切换到新节。
                _flush_segment()
                current = {
                    "uid": section_uid_counter,
                    "section": item["text"].strip(),
                    "heading_level": item["level"],
                    "paragraphs": [],
                }
                section_uid_counter += 1
                current["paragraphs"].append(item)
                continue

            if item["kind"] == "paragraph":
                if current is None:
                    # 首个标题之前的正文：preamble 节（section 为空）。
                    current = {
                        "uid": section_uid_counter,
                        "section": "",
                        "heading_level": 0,
                        "paragraphs": [],
                    }
                    section_uid_counter += 1
                current["paragraphs"].append(item)
                continue

            # 表格：先 flush 当前片段，再在原位置输出表格；
            # 之后的同节段落继续累积成新片段。
            _flush_segment()
            ordered.append(
                {
                    "kind": "table",
                    "cells": item["cells"],
                    "row_count": item["row_count"],
                    "col_count": item["col_count"],
                    "ragged": item["ragged"],
                    "table_index": item["table_index"],
                    "body_position": item["body_position"],
                    "after_paragraph_index": item["after_paragraph_index"],
                }
            )

        _flush_segment()

        # ---- 第三遍：按 ordered 顺序生成 Document，并分配两个索引 ----
        # section_index：仅对「真正产出了片段」的节从 0 递增，避免空节占号；
        # segment_index：同一节内从 0 连续递增，空片段不占号。
        section_index_by_uid = {}
        segment_counter_by_uid = {}
        next_section_index = 0

        documents = []
        for entry in ordered:
            if entry["kind"] == "table":
                lines = [" | ".join(row) for row in entry["cells"]]
                documents.append(
                    Document(
                        page_content="\n".join(lines),
                        metadata={
                            "source": self.file_path,
                            "table_index": entry["table_index"],
                            "table_rows": entry["row_count"],
                            "table_cols": entry["col_count"],
                            "table_ragged": entry["ragged"],
                            "body_position": entry["body_position"],
                            "after_paragraph_index": entry[
                                "after_paragraph_index"
                            ],
                        },
                    )
                )
                continue

            content = "\n".join(
                paragraph["text"] for paragraph in entry["paragraphs"]
            ).strip()
            # 全空片段不产出（否则会产生空向量），也不占用 segment_index。
            if not content:
                continue

            uid = entry["section_uid"]
            if uid not in section_index_by_uid:
                section_index_by_uid[uid] = next_section_index
                next_section_index += 1
            segment_index = segment_counter_by_uid.get(uid, 0)
            segment_counter_by_uid[uid] = segment_index + 1

            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": self.file_path,
                        "section": entry["section"],
                        "section_index": section_index_by_uid[uid],
                        "segment_index": segment_index,
                        "heading_level": entry["heading_level"],
                        "start_paragraph_index": entry["paragraphs"][0][
                            "paragraph_index"
                        ],
                        "end_paragraph_index": entry["paragraphs"][-1][
                            "paragraph_index"
                        ],
                    },
                )
            )

        return documents


# ------------------------------------------------------------
# 工厂
# ------------------------------------------------------------
# 键集合必须与 doc_formats.SUPPORTED_SUFFIXES 完全一致，
# 由 RAG测试/test_loader_factory.py 双向锁定。
SUFFIX_LOADERS = {
    ".md": TextLoader,
    ".txt": TextLoader,
    ".pdf": PdfLoader,
    ".docx": DocxLoader,
}


def get_loader(file_path):
    """按后缀返回 Loader 实例。

    后缀来源唯一：``doc_formats.suffix_of``（与上传校验、索引枚举同源）。
    不支持的后缀抛 ``UnsupportedSuffixError``，由调用方决定跳过还是失败。
    """
    from doc_formats import suffix_of

    suffix = suffix_of(file_path)
    loader_class = SUFFIX_LOADERS.get(suffix)
    if loader_class is None:
        raise UnsupportedSuffixError(suffix, file_path)
    return loader_class(str(file_path))