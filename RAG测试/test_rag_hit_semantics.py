# -*- coding: utf-8 -*-
"""RAG 命中语义与空命中工具链 · 离线专项测试（第 8 轮）。

验证第七轮发现的两个问题的修复契约：
    1. 默认检索将相似度恰好为 0.0 的结果计为命中 → 应过滤，不计为有效命中。
    2. search_knowledge 工具的空命中提示分支难以触发 → 默认过滤零相似度后，
       count==0 可正常触发，输出「未找到相关内容」提示。

契约要点（本轮确定，非擅自下结论）：
    - KnowledgeBase.search 默认（threshold=None）过滤相似度 == 0.0 的结果；
    - 显式传入 threshold 时，完全沿用原有 >= 比较语义（含 threshold=0 时
      0 >= 0 保留）；
    - 不引入 0.3 / 0.5 等未经验证的默认语义阈值；
    - 只有「检索成功但无有效结果」才返回 ok=True 的空命中结构；
    - 初始化失败 / 搜索异常仍返回错误状态，不伪装成空命中。

全程零真实 Embedding、零网络、零云端 API。
"""

import os

import pytest

from rag.knowledge_base import KnowledgeBase
from tools import make_search_knowledge_tool


os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


# ---------------------------------------------------------------------------
# 确定性 Stub Embedding：关键词 one-hot 向量
# ---------------------------------------------------------------------------
KW = ["苹果", "香蕉", "汽车", "飞机"]


class KeywordEmbedding:
    """文本 → 「是否含关键词」的 one-hot 向量（归一化）。

    与查询完全同关键词 → 相似度 1；无交集 → 相似度 0（零向量）。
    """

    def __init__(self):
        self.encode_calls = []

    def encode(self, text):
        self.encode_calls.append(text)
        vec = [1.0 if kw in text else 0.0 for kw in KW]
        norm = (sum(v * v for v in vec)) ** 0.5
        if norm == 0.0:
            return [0.0] * len(KW)
        return [v / norm for v in vec]


def _make_kb(tmp_path, docs):
    for name, content in docs.items():
        (tmp_path / name).write_text(content, encoding="utf-8")
    emb = KeywordEmbedding()
    kb = KnowledgeBase(str(tmp_path), embedding_factory=lambda: emb)
    return kb, emb


# ---------------------------------------------------------------------------
# 1. 默认过滤零相似度
# ---------------------------------------------------------------------------
class TestDefaultZeroFilter:

    def test_zero_similarity_not_counted_as_hit(self, tmp_path):
        docs = {"a.md": "苹果很甜。"}
        kb, _ = _make_kb(tmp_path, docs)
        # 查询「飞机」与「苹果很甜。」无关键词交集 → 相似度 0。
        # 默认（threshold=None）应过滤掉该零相似度结果。
        result = kb.search("飞机", top_k=3)
        assert result["ok"] is True
        assert result["count"] == 0
        assert result["results"] == []
        assert result["context"] == ""

    def test_positive_similarity_kept_without_explicit_threshold(self, tmp_path):
        docs = {"a.md": "苹果很甜。", "b.md": "飞机很快。"}
        kb, _ = _make_kb(tmp_path, docs)
        # 查询「苹果」：a.md 相似度 1，b.md 相似度 0。
        # 默认过滤零相似度后，只保留 a.md。
        result = kb.search("苹果", top_k=3)
        assert result["count"] == 1
        assert result["results"][0]["source"] == "a.md"

    def test_explicit_threshold_still_applies_original_semantics(self, tmp_path):
        docs = {"a.md": "苹果很甜。", "b.md": "飞机很快。"}
        kb, _ = _make_kb(tmp_path, docs)
        # 显式 threshold=0.0：沿用 >= 语义，相似度 0 的结果也会保留（0 >= 0）。
        result = kb.search("苹果", top_k=3, threshold=0.0)
        # 此时 0.0 相似度的 b.md 也满足 0 >= 0，故 count 可能 >= 1（含零相似度项）。
        assert result["ok"] is True
        assert result["count"] >= 1

    def test_all_filtered_returns_empty_hit(self, tmp_path):
        docs = {"a.md": "苹果很甜。"}
        kb, _ = _make_kb(tmp_path, docs)
        result = kb.search("飞机", top_k=3)
        assert result["ok"] is True
        assert result["count"] == 0
        assert result["results"] == []
        assert result["context"] == ""


# ---------------------------------------------------------------------------
# 2. search_knowledge 工具的空命中提示
# ---------------------------------------------------------------------------
class TestSearchKnowledgeToolNoHit:

    def test_tool_returns_not_found_hint_on_zero_hit(self, tmp_path):
        docs = {"a.md": "苹果很甜。"}
        kb, _ = _make_kb(tmp_path, docs)
        tool = make_search_knowledge_tool(kb)
        out = tool("飞机", top_k=3)
        assert out["count"] == 0
        assert out["results"] == []
        assert out["context"] == "（知识库中未找到相关内容）"

    def test_tool_returns_real_context_on_hit(self, tmp_path):
        docs = {"a.md": "苹果很甜。"}
        kb, _ = _make_kb(tmp_path, docs)
        tool = make_search_knowledge_tool(kb)
        out = tool("苹果", top_k=3)
        assert out["count"] >= 1
        assert out["results"]
        assert "苹果" in out["context"]

    def test_tool_raises_on_init_failure_not_empty_hit(self, tmp_path):
        missing = tmp_path / "nope"
        emb = KeywordEmbedding()
        kb = KnowledgeBase(str(missing), embedding_factory=lambda: emb)
        tool = make_search_knowledge_tool(kb)
        with pytest.raises(RuntimeError):
            tool("任意", top_k=3)


# ---------------------------------------------------------------------------
# 3. 来源与排序正确性（回归）
# ---------------------------------------------------------------------------
class TestSourceAndSorting:

    def test_sources_not_confused_and_sorted(self, tmp_path):
        docs = {
            "apple.md": "苹果。",          # 相似度 1
            "both.md": "苹果和香蕉。",      # 相似度 1/sqrt(2)
            "car.md": "汽车跑得快。",       # 相似度 0 → 默认过滤
        }
        kb, _ = _make_kb(tmp_path, docs)
        result = kb.search("苹果", top_k=10)
        sources = [r["source"] for r in result["results"]]
        # 默认过滤零相似度：car.md 不出现在结果中。
        assert "car.md" not in sources
        # 降序：apple.md(1.0) 在 both.md(0.707) 之前。
        assert sources[0] == "apple.md"
        scores = [r["score"] for r in result["results"]]
        assert scores == sorted(scores, reverse=True)
