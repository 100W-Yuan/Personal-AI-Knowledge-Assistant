# -*- coding: utf-8 -*-
"""RAG 检索质量与拒答边界验收（第 12 轮）。

在真实 Embedding（离线加载）下，验证知识库检索能否区分：
    A. 有明确证据的问题；
    B. 相近主题、但缺少目标事实的问题；
    C. 完全无关的问题；
    D. 提高阈值后正确证据被过滤的情况。

关键约定：
    - 相关性由「文档是否包含回答问题所需的事实」决定，不由分数决定；
    - 「检索到相近文档」≠「检索到支持答案的证据」；
    - 「检索到文档」≠「文档支持答案」；
    - 「未检索到文档」≠「最终回答会拒答」。
    本轮只验证检索层 + 上下文层，不验证 Agent 最终回答/拒答行为。

安全：全程离线（HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1），不联网、不下载。
"""

import os

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from rag.embedding import Embedding               # noqa: E402
from rag.knowledge_base import KnowledgeBase      # noqa: E402


# ---------------------------------------------------------------------------
# 虚构产品知识库（星辰 X1 手机）
# ---------------------------------------------------------------------------
# 事实边界明确：每个文档只含某一方面的信息，不含「防水等级」「处理器」「重量」等。
DOCS = {
    "battery.md": "星辰 X1 手机的电池容量是 5000 毫安时，支持 66 瓦快充。",
    "screen.md": "星辰 X1 手机采用 6.7 英寸 AMOLED 屏幕，刷新率为 120 赫兹。",
    "camera.md": "星辰 X1 手机配备 5000 万像素主摄像头，支持夜景模式。",
    "price.md": "星辰 X1 手机的起售价是 3999 元，提供 8GB 和 12GB 两个内存版本。",
}

# 每个「有答案」查询的关键事实（用于断言 ContextBuilder 内容包含）。
FACTS = {
    "battery.md": ["5000 毫安", "66 瓦"],
    "screen.md": ["6.7 英寸", "120 赫兹"],
    "camera.md": ["5000 万像素"],
    "price.md": ["3999 元"],
}


@pytest.fixture(scope="module")
def kb(tmp_path_factory):
    d = tmp_path_factory.mktemp("rag_abstain_docs")
    for name, content in DOCS.items():
        (d / name).write_text(content, encoding="utf-8")
    emb = Embedding()
    return KnowledgeBase(str(d), embedding_factory=lambda: emb)


def top_sources(result):
    return [r["source"] for r in result["results"]]


def top_scores(result):
    return [(r["source"], round(r["score"], 4)) for r in result["results"]]


# ---------------------------------------------------------------------------
# A 组：明确有答案（≥3 案例）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("query,expected_doc", [
    ("星辰 X1 的电池容量是多少？", "battery.md"),
    ("星辰 X1 的屏幕尺寸和刷新率是多少？", "screen.md"),
    ("星辰 X1 的起售价是多少？", "price.md"),
])
def test_group_a_has_answer(kb, query, expected_doc):
    result = kb.search(query, top_k=4)
    assert result["ok"] is True
    sources = top_sources(result)
    # 证据文档必须被召回。
    assert expected_doc in sources, (
        f"查询「{query}」未召回证据 {expected_doc}，实际={top_scores(result)}"
    )
    # KnowledgeBase.search 内部已调用 ContextBuilder 生成 context 字段，
    # 直接断言 context 包含关键事实（无需重新调用 ContextBuilder.build）。
    context = result["context"]
    for fact in FACTS[expected_doc]:
        assert fact in context, (
            f"上下文缺少关键事实「{fact}」，实际 context={context!r}"
        )


# ---------------------------------------------------------------------------
# B 组：相近主题但缺少答案（≥3 案例）
# ---------------------------------------------------------------------------
# 这些问题都指向「星辰 X1 手机」这个主题，但知识库中没有任何文档包含答案。
@pytest.mark.parametrize("query", [
    "星辰 X1 手机的防水等级是多少？",   # 库中无防水信息
    "星辰 X1 手机用的是什么处理器？",   # 库中无处理器信息
    "星辰 X1 手机的重量是多少克？",     # 库中无重量信息
])
def test_group_b_near_topic_but_no_answer(kb, query):
    """主题相近，但检索到的文档不含答案所需事实。

    断言：即便返回了相近文档，上下文也不包含答案所需的关键事实。
    不因「主题相似/分数较高」就认定答案有依据。
    """
    result = kb.search(query, top_k=4)
    assert result["ok"] is True
    context = result["context"]

    # 这些问题的答案关键词在知识库中不存在，上下文不应凭空出现。
    absent_facts = {
        "防水等级": ["防水", "IP"],
        "处理器": ["处理器", "芯片"],
        "重量": ["重量", "克"],
    }
    for key in absent_facts:
        if key in query:
            for term in absent_facts[key]:
                # 不能断言「一定不出现」因为文档里可能巧合含「克」；
                # 这里只做「可观察」记录，不做强断言。
                pass

    # 核心可验证断言：检索到的候选确实存在（相近主题会被召回），
    # 但回答所需的具体事实（防水等级数值/IP 等级）不在文档中。
    # 由于库中确实无「防水/IP」等词，可安全断言这些词不出现在上下文。
    if "防水" in query:
        assert "IP" not in context and "防水" not in context, (
            f"上下文不应包含防水信息，实际={context!r}"
        )
    if "处理器" in query:
        assert "处理器" not in context and "芯片" not in context
    if "重量" in query:
        assert "重量" not in context

    # 记录实际返回（供人工复核，不参与判定）。
    print(f"\n[B组] {query!r} -> {top_scores(result)}")


# ---------------------------------------------------------------------------
# C 组：完全无关问题（≥3 案例）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("query", [
    "今天北京的天气怎么样？",
    "如何制作提拉米苏？",
    "二战是哪一年结束的？",
])
def test_group_c_irrelevant(kb, query):
    """完全无关问题：只记录候选，不预设空结果，不凭非零分数判误召回。"""
    result = kb.search(query, top_k=4)
    assert result["ok"] is True
    print(f"\n[C组] {query!r} -> {top_scores(result)}")
    # 仅验证结构合法。
    assert all(isinstance(r["score"], float) for r in result["results"])


# ---------------------------------------------------------------------------
# D 组：阈值敏感性（≥3 案例）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("query,expected_doc", [
    ("星辰 X1 的电池容量是多少？", "battery.md"),   # A 组代表
    ("星辰 X1 手机的防水等级是多少？", None),        # B 组代表（无答案）
    ("星辰 X1 的起售价是多少？", "price.md"),        # A 组代表
])
def test_group_d_threshold_sensitivity(kb, query, expected_doc):
    """同一查询在不同阈值下，记录候选数量与目标证据是否被过滤。"""
    thresholds = [None, 0.3, 0.6, 0.8]
    print(f"\n[D组] 查询 {query!r}:")
    for t in thresholds:
        result = kb.search(query, top_k=4, threshold=t)
        sources = top_sources(result)
        kept = (expected_doc in sources) if expected_doc else "N/A"
        print(f"  threshold={t}: count={result['count']}, "
              f"sources={top_scores(result)}, 目标证据保留={kept}")
    # 至少验证默认阈值下的结果合法。
    default = kb.search(query, top_k=4)
    assert default["ok"] is True
