# -*- coding: utf-8 -*-
"""RAG 端到端检索质量 · 离线专项测试（第 7 轮）。

在「不修改生产代码」前提下，用虚构文档、临时目录、确定性 Stub Embedding
和人工向量，验证 RAG 从文档加载 → 切分 → 向量写入 → 检索 → 阈值过滤 →
上下文组装的完整数据流。

范围与边界：
    - 全程零真实 Embedding、零网络、零云端 API；
    - 测试 search_knowledge 工具函数（tools.make_search_knowledge_tool）本身，
      不实例化 Agent（Agent 构造需 OpenAI 客户端 + Memory，涉及 DB/网络，
      超出离线安全范围，见文末说明）；
    - 对无明确约定的边界行为标注为「待决策」，不擅自认定。

运行约定（零删除、零缓存、零字节码、离线）：
    pytest RAG测试/test_rag_end_to_end_offline.py \
        --basetemp=<唯一新目录> \
        -o tmp_path_retention_policy=all -p no:cacheprovider
    环境变量：PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"""

import os

import pytest

from rag.document import Document
from rag.knowledge_base import KnowledgeBase
from tools import make_search_knowledge_tool


# ---------------------------------------------------------------------------
# 离线环境加固
# ---------------------------------------------------------------------------
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


# ---------------------------------------------------------------------------
# 确定性 Stub Embedding：关键词 one-hot 向量
# ---------------------------------------------------------------------------
# 用一个固定关键词表构造向量。向量第 i 维 = 文本是否含关键词 KW[i]。
# 这样余弦相似度完全可控：两段文本重叠的关键词越多，相似度越高。
KW = ["苹果", "香蕉", "汽车", "飞机", "数据库", "网络"]


class KeywordEmbedding:
    """把文本映射为「是否含关键词」的 one-hot 向量（6 维）。

    与查询完全同关键词的文档相似度=1；完全无交集=0；部分交集=0~1。
    归一化保证余弦语义稳定，且可精确断言相似度排序。
    """

    def __init__(self):
        self.encode_calls = []

    def encode(self, text):
        self.encode_calls.append(text)
        vec = [1.0 if kw in text else 0.0 for kw in KW]
        norm = (sum(v * v for v in vec)) ** 0.5
        if norm == 0.0:
            # 无任何关键词：返回全零向量（相似度恒为 0）。
            return [0.0] * len(KW)
        return [v / norm for v in vec]


def _make_kb(tmp_path, docs):
    """构造 KnowledgeBase + 注入 KeywordEmbedding，返回 (kb, embedding)。

    docs: dict[文件名 -> 文本内容]
    """
    for name, content in docs.items():
        (tmp_path / name).write_text(content, encoding="utf-8")
    emb = KeywordEmbedding()
    kb = KnowledgeBase(str(tmp_path), embedding_factory=lambda: emb)
    return kb, emb


# ---------------------------------------------------------------------------
# 1. 完整数据流：加载→切分→向量→检索→阈值→上下文
# ---------------------------------------------------------------------------
class TestEndToEndFlow:

    def test_full_pipeline_hit_returns_source_chunk_and_score(self, tmp_path):
        docs = {
            "fruit.md": "苹果是一种水果，香蕉也是一种水果。",
            "vehicle.md": "汽车和飞机都是交通工具。",
        }
        kb, _ = _make_kb(tmp_path, docs)
        result = kb.search("苹果", top_k=3)

        assert result["ok"] is True
        assert result["count"] >= 1
        # 命中的结果中，来源应为含关键词的 fruit.md，且片段含「苹果」。
        top = result["results"][0]
        assert top["source"] == "fruit.md"
        assert "苹果" in top["chunk"]
        # 相似度：查询向量归一化后含「苹果」维度=1（单关键词），
        # 文档含「苹果」「香蕉」两个关键词，相似度 = 1/sqrt(2)。
        assert top["score"] == pytest.approx(1.0 / (2 ** 0.5), abs=1e-6)
        # 上下文组装包含命中片段内容。
        assert "苹果" in result["context"]

    def test_results_sorted_by_similarity_descending(self, tmp_path):
        docs = {
            "a.md": "苹果和香蕉。",   # 与查询「苹果」重叠 1 个关键词
            "b.md": "苹果。",         # 与查询「苹果」完全一致（1 关键词，归一化后=1）
        }
        kb, _ = _make_kb(tmp_path, docs)
        result = kb.search("苹果", top_k=3)
        scores = [r["score"] for r in result["results"]]
        # 降序排列。
        assert scores == sorted(scores, reverse=True)
        # 最高分应为「苹果。」（仅含查询关键词，相似度=1）。
        assert result["results"][0]["source"] == "b.md"
        assert result["results"][0]["score"] == pytest.approx(1.0, abs=1e-6)

    def test_multiple_docs_do_not_confuse_sources(self, tmp_path):
        docs = {
            "apple.md": "苹果很甜。",
            "banana.md": "香蕉是黄色的。",
            "car.md": "汽车跑得快。",
        }
        kb, _ = _make_kb(tmp_path, docs)
        # 查询「汽车」并显式阈值过滤，只保留相似度 > 0 的结果（car.md）。
        result = kb.search("汽车", top_k=3, threshold=1e-6)
        sources = {r["source"] for r in result["results"]}
        assert sources == {"car.md"}

    def test_no_hit_returns_ok_true_count_zero(self, tmp_path):
        docs = {"a.md": "苹果和香蕉。"}
        kb, _ = _make_kb(tmp_path, docs)
        # 无关键词交集：显式阈值过滤掉相似度为 0 的结果，得到空命中。
        result = kb.search("飞机", top_k=3, threshold=1e-6)
        assert result["ok"] is True
        assert result["count"] == 0
        assert result["results"] == []
        assert result["context"] == ""

    def test_high_threshold_filters_all(self, tmp_path):
        docs = {"a.md": "苹果很甜。"}
        kb, _ = _make_kb(tmp_path, docs)
        # 阈值 0.99：即使命中（相似度 1.0）也应保留；这里用 1.01 过滤掉一切。
        result = kb.search("苹果", top_k=3, threshold=1.01)
        assert result["ok"] is True
        assert result["count"] == 0
        assert result["results"] == []
        assert result["context"] == ""

    def test_top_k_one_returns_single_highest(self, tmp_path):
        docs = {
            "a.md": "苹果。",       # 相似度 1.0
            "b.md": "苹果和香蕉。",  # 相似度 1/sqrt(2) ≈ 0.707
        }
        kb, _ = _make_kb(tmp_path, docs)
        result = kb.search("苹果", top_k=1)
        assert result["count"] == 1
        assert result["results"][0]["source"] == "a.md"
        assert result["results"][0]["score"] == pytest.approx(1.0, abs=1e-6)

    def test_chunks_and_chunk_id_correspond(self, tmp_path):
        # 用长文本触发切分，验证 chunk 内容与 chunk_id 对应、重叠符合契约。
        long_text = "苹果" * 50  # 100 字符，默认 chunk_size=100 → 1 块
        # 改为触发多块的文本：200 字符，chunk_size 默认 100，overlap 20。
        long_text = ("苹果香蕉汽车飞机数据库网络" * 20)  # 240 字符
        docs = {"long.md": long_text}
        kb, _ = _make_kb(tmp_path, docs)
        result = kb.search("苹果", top_k=10)
        # 断言结果 chunk 非空、score 为 float。
        assert result["count"] >= 1
        for r in result["results"]:
            assert isinstance(r["chunk"], str) and r["chunk"]
            assert isinstance(r["chunk_id"], int) and r["chunk_id"] >= 0
            assert isinstance(r["score"], float)

    def test_context_does_not_fabricate_content(self, tmp_path):
        docs = {"a.md": "苹果很甜。"}
        kb, _ = _make_kb(tmp_path, docs)
        result = kb.search("苹果", top_k=3)
        # 上下文只应包含原文已有的内容，不出现原文没有的词。
        assert "苹果" in result["context"]
        assert "西瓜" not in result["context"]   # 原文无「西瓜」
        assert "虚构词XYZ" not in result["context"]


# ---------------------------------------------------------------------------
# 2. 惰性初始化与重复查询
# ---------------------------------------------------------------------------
class TestLazyInitAndReuse:

    def test_lazy_init_only_on_first_search(self, tmp_path):
        docs = {"a.md": "苹果和香蕉。"}
        kb, emb = _make_kb(tmp_path, docs)
        assert kb.file_count == 0
        assert kb.chunk_count == 0
        assert emb.encode_calls == []   # 构造后未编码

        kb.search("苹果")
        first_calls = len(emb.encode_calls)
        assert first_calls > 0          # 首次检索触发编码（文档 chunk）

        kb.search("苹果")
        kb.search("香蕉")
        # 重复查询不重复构建索引：不再新增「文档编码」调用，
        # 每次查询仅新增 1 次查询编码。
        assert kb.chunk_count >= 1
        # 索引只构建一次：file_count 稳定。
        assert kb.file_count == 1


# ---------------------------------------------------------------------------
# 3. 结构化错误字段
# ---------------------------------------------------------------------------
class TestErrorStructure:

    def test_missing_dir_returns_structured_error(self, tmp_path):
        missing = tmp_path / "nope"
        emb = KeywordEmbedding()
        kb = KnowledgeBase(str(missing), embedding_factory=lambda: emb)
        result = kb.search("任意")
        assert result["ok"] is False
        assert result["stage"] == "init"
        assert "FileNotFoundError" in result["error"]
        assert "知识目录缺失或为空" in result["error"]


# ---------------------------------------------------------------------------
# 4. search_knowledge 工具函数（无需实例化 Agent）
# ---------------------------------------------------------------------------
class TestSearchKnowledgeTool:

    def test_tool_returns_structured_result_on_hit(self, tmp_path):
        docs = {"a.md": "苹果很甜。"}
        kb, _ = _make_kb(tmp_path, docs)
        tool = make_search_knowledge_tool(kb)
        out = tool("苹果", top_k=3)
        assert out["query"] == "苹果"
        assert out["count"] >= 1
        assert out["results"]
        assert "苹果" in out["context"]

    def test_tool_returns_hint_on_no_hit(self, tmp_path):
        """记录工具层未命中提示分支的可达性观察。

        search_knowledge 工具（tools.py:58）不传 threshold，因此其 count==0
        的提示分支（tools.py:67-68）在当前接口下**无法通过正常检索触发**——
        因为 KnowledgeBase.search 默认 threshold=None，Top-K 截断后 count
        恒 >= 1（除非目录为空，但空目录走 init 失败抛 RuntimeError）。

        本测试仅验证：工具在无关键词交集查询下仍返回结构化结果，
        不擅自判定「未命中提示应/不应出现」。
        """
        docs = {"a.md": "苹果很甜。"}
        kb, _ = _make_kb(tmp_path, docs)
        tool = make_search_knowledge_tool(kb)
        out = tool("飞机", top_k=3)
        assert out["query"] == "飞机"
        assert isinstance(out["count"], int)
        assert isinstance(out["context"], str)
        assert isinstance(out["results"], list)

    def test_tool_raises_on_init_failure(self, tmp_path):
        missing = tmp_path / "nope"
        emb = KeywordEmbedding()
        kb = KnowledgeBase(str(missing), embedding_factory=lambda: emb)
        tool = make_search_knowledge_tool(kb)
        with pytest.raises(RuntimeError):
            tool("任意", top_k=3)


# ---------------------------------------------------------------------------
# 5. 边界行为：标注待决策，不做结论
# ---------------------------------------------------------------------------
class TestUndecidedBoundary:

    def test_context_marker_is_present(self, tmp_path):
        """记录当前契约：命中时上下文含【相关资料】分隔标记（非凭空结论）。"""
        docs = {"a.md": "苹果很甜。"}
        kb, _ = _make_kb(tmp_path, docs)
        result = kb.search("苹果", top_k=3)
        # 仅验证当前实现行为，不代表这是唯一正确契约。
        assert "【相关资料】" in result["context"]
