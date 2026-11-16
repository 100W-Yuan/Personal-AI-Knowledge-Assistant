# -*- coding: utf-8 -*-
"""RAG 知识检索质量 · 离线专项测试（第 3 轮）。

覆盖 rag 模块的离线单元行为：
    A. TextLoader       —— 正常 / 空文件 / 缺失 / 非法 UTF-8
    B. TextSplitter     —— 空 / 空白 / 短 / 边界 / 长 / 重叠 / 元数据
    C. VectorStore      —— 空库 / 排序 / Top-K / 重复 / 零向量 / 维度不一致
    D. Retriever        —— 无结果 / 阈值 / Top-K 后不补位 / top_k=0 / 负数
    E. KnowledgeBase    —— 惰性初始化 / Stub 检索 / 失败处理 / 缓存 / 空命中 / 元数据

安全约束：
    - 仅使用临时目录（tmp_path）与虚构内容；
    - 不实例化真实 Embedding()，不下载模型，不联网；
    - 危险/潜在缺陷行为仅做静态或受控断言，绝不调用可能不终止的生产方法。

运行约定（零删除、零缓存、零字节码）：
    pytest RAG测试/test_rag_quality_offline.py \
        --basetemp=<唯一新目录> \
        -o tmp_path_retention_policy=all \
        -p no:cacheprovider
    并设置环境变量 PYTHONDONTWRITEBYTECODE=1。
"""

import os

import pytest

from rag.document import Document
from rag.loader import TextLoader
from rag.splitter import TextSplitter
from rag.vector_store import VectorStore
from rag.retriever import Retriever
from rag.context_builder import ContextBuilder
from rag.knowledge_base import KnowledgeBase


# ---------------------------------------------------------------------------
# 离线环境加固（导入路径层面）
# ---------------------------------------------------------------------------
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


# ---------------------------------------------------------------------------
# 测试辅助：Stub Embedding 工厂
# ---------------------------------------------------------------------------
class _StubEmbedding:
    """确定性 stub：把文本映射为固定维度向量，绝不加载真实模型。"""

    def __init__(self, dim=8):
        self.dim = dim
        self.encode_calls = []

    def encode(self, text):
        self.encode_calls.append(text)
        # 用文本长度 + 简单哈希构造确定性非零向量，保证不同文本可分。
        h = sum(ord(ch) for ch in text) % 1000
        vec = [0.0] * self.dim
        vec[0] = float(len(text) + 1)
        vec[1] = float(h + 1)
        # 归一化，保证余弦相似度语义稳定。
        norm = (vec[0] ** 2 + vec[1] ** 2) ** 0.5
        vec[0] /= norm
        vec[1] /= norm
        return vec


def _stub_factory(dim=8):
    inst = _StubEmbedding(dim=dim)
    return lambda: inst


# ---------------------------------------------------------------------------
# A. TextLoader
# ---------------------------------------------------------------------------
class TestTextLoader:

    def test_load_utf8_returns_single_document(self, tmp_path):
        p = tmp_path / "a.txt"
        p.write_text("你好，世界", encoding="utf-8")
        docs = TextLoader(str(p)).load()
        assert isinstance(docs, list)
        assert len(docs) == 1
        assert docs[0].page_content == "你好，世界"
        assert docs[0].metadata["source"] == str(p)

    def test_load_empty_file_returns_one_doc_with_empty_content(self, tmp_path):
        p = tmp_path / "empty.txt"
        p.write_text("", encoding="utf-8")
        docs = TextLoader(str(p)).load()
        assert len(docs) == 1
        assert docs[0].page_content == ""

    def test_missing_file_raises_file_not_found(self, tmp_path):
        missing = tmp_path / "does_not_exist.txt"
        with pytest.raises(FileNotFoundError):
            TextLoader(str(missing)).load()

    def test_invalid_utf8_raises_unicode_decode_error(self, tmp_path):
        p = tmp_path / "bad.txt"
        p.write_bytes(b"\xff\xfe\x00\x41\x00")  # 非法 UTF-8 序列
        with pytest.raises(UnicodeDecodeError):
            TextLoader(str(p)).load()


# ---------------------------------------------------------------------------
# B. TextSplitter
# ---------------------------------------------------------------------------
class TestTextSplitter:

    def test_split_empty_text_returns_empty_list(self):
        doc = Document(page_content="")
        assert TextSplitter().split(doc) == []

    def test_split_whitespace_text_returns_one_chunk(self):
        doc = Document(page_content="   ")
        chunks = TextSplitter().split(doc)
        assert len(chunks) == 1
        assert chunks[0].page_content == "   "

    def test_split_short_text_returns_one_chunk_with_chunk_id_0(self):
        doc = Document(page_content="short", metadata={"source": "s.txt"})
        chunks = TextSplitter().split(doc)
        assert len(chunks) == 1
        assert chunks[0].metadata["chunk_id"] == 0
        assert chunks[0].metadata["source"] == "s.txt"

    def test_split_exact_chunk_size_returns_single_chunk(self):
        # 契约：len == chunk_size 时只生成一个块，内容完整且 chunk_id=0。
        text = "a" * 100
        doc = Document(page_content=text)
        chunks = TextSplitter(chunk_size=100, overlap=20).split(doc)
        assert len(chunks) == 1
        assert chunks[0].page_content == text
        assert chunks[0].metadata["chunk_id"] == 0

    def test_split_long_text_chunk_ids_are_sequential(self):
        # 300 字符，chunk_size=100, overlap=20 -> 步长 80
        text = "x" * 300
        doc = Document(page_content=text)
        chunks = TextSplitter(chunk_size=100, overlap=20).split(doc)
        assert len(chunks) == 4  # 0..99, 80..179, 160..259, 240..299
        assert [c.metadata["chunk_id"] for c in chunks] == [0, 1, 2, 3]

    def test_split_reconstructs_original_text_by_overlap_rule(self):
        # 契约：按 chunk_size=30, overlap=10 拼接去重叠后能完整还原原文，
        # 且顺序正确（逐块起始偏移递增，无字符丢失、无重复）。
        text = "0123456789" * 10  # 100 字符
        doc = Document(page_content=text)
        chunks = TextSplitter(chunk_size=30, overlap=10).split(doc)

        # 顺序：每块 chunk_id 连续递增。
        assert [c.metadata["chunk_id"] for c in chunks] == list(range(len(chunks)))

        # 覆盖：去重叠拼接还原原文（无字符丢失、无重复）。
        rebuilt = chunks[0].page_content
        for c in chunks[1:]:
            rebuilt += c.page_content[10:]  # 每块去掉前 overlap 字符
        assert rebuilt == text

    def test_split_overlap_repeats_tail_of_previous_chunk(self):
        text = "0123456789" * 10  # 100 字符
        doc = Document(page_content=text)
        chunks = TextSplitter(chunk_size=30, overlap=10).split(doc)
        assert len(chunks) >= 2
        # 相邻 chunk：前一块末尾 10 字符 == 后一块开头 10 字符
        tail = chunks[0].page_content[-10:]
        head = chunks[1].page_content[:10]
        assert tail == head

    def test_split_overlap_length_is_exact(self):
        # 契约：相邻块重叠长度恰好 == overlap。
        text = "abcdefghijklmnopqrstuvwxyz" * 4  # 104 字符
        doc = Document(page_content=text)
        chunks = TextSplitter(chunk_size=20, overlap=5).split(doc)
        assert len(chunks) >= 2
        for i in range(len(chunks) - 1):
            tail = chunks[i].page_content[-5:]
            head = chunks[i + 1].page_content[:5]
            assert tail == head

    def test_split_preserves_original_metadata(self):
        doc = Document(page_content="abc", metadata={"source": "s.txt", "k": "v"})
        chunks = TextSplitter().split(doc)
        assert chunks[0].metadata["source"] == "s.txt"
        assert chunks[0].metadata["k"] == "v"
        assert "chunk_id" in chunks[0].metadata


class TestTextSplitterParamValidation:

    def test_chunk_size_zero_raises_value_error(self):
        with pytest.raises(ValueError):
            TextSplitter(chunk_size=0)

    def test_chunk_size_negative_raises_value_error(self):
        with pytest.raises(ValueError):
            TextSplitter(chunk_size=-5)

    def test_overlap_negative_raises_value_error(self):
        with pytest.raises(ValueError):
            TextSplitter(chunk_size=100, overlap=-1)

    def test_overlap_ge_chunk_size_raises_value_error(self):
        with pytest.raises(ValueError):
            TextSplitter(chunk_size=100, overlap=100)


# ---------------------------------------------------------------------------
# C. VectorStore
# ---------------------------------------------------------------------------
class TestVectorStore:

    def test_search_empty_store_returns_empty_list(self):
        assert VectorStore().search([1.0, 0.0]) == []

    def test_search_sorts_by_similarity_descending(self):
        store = VectorStore()
        # 用单位向量构造，便于断言相似度。
        store.add(Document(page_content="same", metadata={"i": 0}), [1.0, 0.0])
        store.add(Document(page_content="same", metadata={"i": 1}), [0.0, 1.0])
        store.add(Document(page_content="same", metadata={"i": 2}), [1.0, 0.0])
        results = store.search([1.0, 0.0], top_k=3)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)
        assert scores[0] == pytest.approx(1.0)

    def test_search_top_k_truncation(self):
        store = VectorStore()
        for i in range(5):
            store.add(Document(page_content="x", metadata={"i": i}), [float(i + 1), 0.0])
        results = store.search([1.0, 0.0], top_k=2)
        assert len(results) == 2

    def test_duplicate_add_keeps_all_items(self):
        store = VectorStore()
        for _ in range(3):
            store.add(Document(page_content="dup"), [1.0, 0.0])
        assert len(store.items) == 3
        assert len(store.search([1.0, 0.0], top_k=10)) == 3

    def test_similarity_correct_for_same_dimension(self):
        store = VectorStore()
        # 两个同维单位向量，相似度应为精确余弦值。
        assert store.similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
        assert store.similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_dimension_mismatch_raises_value_error(self):
        """契约：维度不一致时 similarity 抛出 ValueError，而非静默截断。"""
        store = VectorStore()
        with pytest.raises(ValueError):
            store.similarity([1.0], [1.0, 0.0, 0.0])

    def test_search_does_not_silently_truncate_dimension(self):
        """契约：通过 search() 检索时同样不能静默截断维度。"""
        store = VectorStore()
        store.add(Document(page_content="long"), [1.0, 0.0, 0.0])
        with pytest.raises(ValueError):
            store.search([1.0, 0.0], top_k=1)

    def test_zero_vector_similarity_is_zero_not_division_error(self):
        store = VectorStore()
        store.add(Document(page_content="z"), [0.0, 0.0])
        result = store.search([1.0, 0.0], top_k=1)[0]
        assert result["score"] == 0.0

    def test_normal_results_still_sorted_descending(self):
        store = VectorStore()
        store.add(Document(page_content="a", metadata={"i": 0}), [1.0, 0.0])
        store.add(Document(page_content="b", metadata={"i": 1}), [0.0, 1.0])
        store.add(Document(page_content="c", metadata={"i": 2}), [0.6, 0.8])
        results = store.search([1.0, 0.0], top_k=3)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# D. Retriever
# ---------------------------------------------------------------------------
class TestRetriever:

    def _store_with(self, vectors):
        store = VectorStore()
        for i, v in enumerate(vectors):
            store.add(Document(page_content=f"doc{i}", metadata={"i": i}), v)
        return store

    def test_retrieve_no_results_returns_empty(self):
        ret = Retriever(VectorStore())
        assert ret.retrieve([1.0, 0.0]) == []

    def test_retrieve_default_no_threshold_returns_top_k(self):
        ret = Retriever(self._store_with([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]]))
        results = ret.retrieve([1.0, 0.0], top_k=2)
        assert len(results) == 2

    def test_retrieve_threshold_filters(self):
        ret = Retriever(self._store_with([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]]))
        results = ret.retrieve([1.0, 0.0], top_k=3, threshold=0.7)
        # 0.0 向量的相似度为 0.0，0.6/0.8 向量的相似度约 0.6，均 < 0.7
        assert all(r["score"] >= 0.7 for r in results)
        # 明确断言：低于阈值的候选确实被移除（不应出现在结果中）。
        kept = {r["document"].metadata["i"] for r in results}
        assert 0 in kept          # [1.0, 0.0] 相似度 1.0，保留
        assert 1 not in kept      # [0.0, 1.0] 相似度 0.0，移除
        assert 2 not in kept      # [0.6, 0.8] 相似度约 0.6，移除

    def test_retrieve_threshold_keeps_equal_boundary(self):
        ret = Retriever(self._store_with([[1.0, 0.0]]))
        results = ret.retrieve([1.0, 0.0], top_k=1, threshold=1.0)
        assert len(results) == 1  # score == 1.0，等于阈值，必须保留

    def test_retrieve_filter_does_not_refill(self):
        # 3 个结果，阈值过滤后仅剩 1 个；不应补足到 top_k。
        ret = Retriever(self._store_with([[1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]))
        results = ret.retrieve([1.0, 0.0], top_k=3, threshold=0.9)
        assert len(results) == 1

    def test_retrieve_top_k_zero_returns_empty(self):
        ret = Retriever(self._store_with([[1.0, 0.0], [0.0, 1.0]]))
        assert ret.retrieve([1.0, 0.0], top_k=0) == []

    def test_retrieve_negative_top_k_raises_value_error(self):
        """契约：负数 top_k 抛出 ValueError，而非静默切片截尾。"""
        ret = Retriever(self._store_with([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]]))
        with pytest.raises(ValueError):
            ret.retrieve([1.0, 0.0], top_k=-1)

    def test_retrieve_non_integer_top_k_raises_value_error(self):
        """契约：浮点数等非整数 top_k 抛出 ValueError。"""
        ret = Retriever(self._store_with([[1.0, 0.0], [0.0, 1.0]]))
        with pytest.raises(ValueError):
            ret.retrieve([1.0, 0.0], top_k=1.5)

    def test_retrieve_bool_top_k_raises_value_error(self):
        """契约：True/False 不应被当作合法数量参数（bool 是 int 子类）。"""
        ret = Retriever(self._store_with([[1.0, 0.0], [0.0, 1.0]]))
        with pytest.raises(ValueError):
            ret.retrieve([1.0, 0.0], top_k=True)
        with pytest.raises(ValueError):
            ret.retrieve([1.0, 0.0], top_k=False)


# ---------------------------------------------------------------------------
# E. KnowledgeBase
# ---------------------------------------------------------------------------
class TestKnowledgeBase:

    def _make_kb(self, tmp_path, dim=8):
        stub = _StubEmbedding(dim=dim)
        kb = KnowledgeBase(str(tmp_path), embedding_factory=lambda: stub)
        return kb, stub

    def test_constructor_is_lazy(self, tmp_path):
        kb, stub = self._make_kb(tmp_path)
        assert kb.file_count == 0
        assert kb.chunk_count == 0
        assert kb._embedding is None
        assert stub.encode_calls == []  # 构造时不触发任何 encode

    def test_first_search_returns_ok_structure(self, tmp_path):
        (tmp_path / "a.md").write_text("Python 是一种编程语言。", encoding="utf-8")
        kb, stub = self._make_kb(tmp_path)
        result = kb.search("Python")
        assert result["ok"] is True
        assert "query" in result and "count" in result
        assert "results" in result and "context" in result
        assert result["count"] >= 1
        assert stub.encode_calls  # 首次检索触发了编码

    def test_source_metadata_is_filename_not_full_path(self, tmp_path):
        (tmp_path / "data.md").write_text("内容" * 50, encoding="utf-8")
        kb, _ = self._make_kb(tmp_path)
        result = kb.search("内容")
        for r in result["results"]:
            assert r["source"] == "data.md"
            assert "\\" not in r["source"] and "/" not in r["source"]

    def test_missing_or_empty_dir_returns_init_failure(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        kb, _ = self._make_kb(empty_dir)
        result = kb.search("query")
        assert result["ok"] is False
        assert result["stage"] == "init"
        # 契约：错误信息应同时保留异常类型名与有意义的消息。
        assert "FileNotFoundError" in result["error"]
        assert "知识目录缺失或为空" in result["error"]

    def test_search_stage_failure_keeps_type_and_message(self, tmp_path):
        """契约：搜索阶段失败时 stage="search"，且保留异常类型与消息。

        构造「索引阶段成功、查询编码阶段抛异常」的 stub，确保失败发生在
        search 阶段（而非 _ensure_index 的 init 阶段）。
        """
        (tmp_path / "a.md").write_text("Alpha Beta Gamma", encoding="utf-8")

        class _BoomOnQueryEmbedding:
            def __init__(self):
                self.calls = 0

            def encode(self, text):
                self.calls += 1
                # 第一次调用发生在索引构建（对文档 chunk 编码），应成功；
                # 第二次调用发生在查询编码，抛异常。
                if self.calls == 1:
                    return [1.0, 0.0]
                raise RuntimeError("查询编码阶段故障")

        kb = KnowledgeBase(str(tmp_path), embedding_factory=lambda: _BoomOnQueryEmbedding())
        result = kb.search("Alpha")
        assert result["ok"] is False
        assert result["stage"] == "search"
        assert "RuntimeError" in result["error"]
        assert "查询编码阶段故障" in result["error"]

    def test_init_failure_is_cached(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        kb, stub = self._make_kb(empty_dir)
        r1 = kb.search("q")
        calls_before = len(stub.encode_calls)
        r2 = kb.search("q")
        assert r1["ok"] is False and r2["ok"] is False
        # 失败被缓存，第二次不触发额外 encode。
        assert len(stub.encode_calls) == calls_before

    def test_empty_hit_returns_ok_count_zero(self, tmp_path):
        (tmp_path / "a.md").write_text("Alpha Beta Gamma", encoding="utf-8")
        kb, _ = self._make_kb(tmp_path)
        # 用高阈值使所有候选被过滤，验证空命中是正常结果（ok=True）。
        result = kb.search("Alpha", threshold=1.0)
        assert result["ok"] is True
        assert result["count"] == 0
        assert result["results"] == []
        assert result["context"] == ""

    def test_threshold_is_forwarded(self, tmp_path):
        (tmp_path / "a.md").write_text("Alpha Beta Gamma" * 20, encoding="utf-8")
        kb, _ = self._make_kb(tmp_path)
        all_result = kb.search("Alpha")
        filtered = kb.search("Alpha", threshold=0.999999)
        # 阈值过滤实际生效：过滤后结果数严格少于不过滤时。
        assert filtered["count"] < all_result["count"]
        assert all_result["count"] >= 1


# ---------------------------------------------------------------------------
# F. ContextBuilder
# ---------------------------------------------------------------------------
class TestContextBuilder:

    def test_build_empty_returns_empty_string(self):
        assert ContextBuilder().build([]) == ""

    def test_build_includes_score_and_content(self):
        doc = Document(page_content="hello", metadata={"source": "s"})
        results = [{"document": doc, "score": 0.9}]
        ctx = ContextBuilder().build(results)
        assert "hello" in ctx
        assert "0.9" in ctx


# ---------------------------------------------------------------------------
# 参数校验测试（取代原先的纯静态注记，实际调用构造器断言拒绝行为）
# ---------------------------------------------------------------------------
# TextSplitter 的参数校验见 TestTextSplitterParamValidation（上方）。
# VectorStore 维度校验、Retriever top_k 校验见各自测试类。

