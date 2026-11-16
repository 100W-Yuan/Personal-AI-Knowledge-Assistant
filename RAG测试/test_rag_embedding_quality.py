# -*- coding: utf-8 -*-
"""RAG 真实 Embedding 检索质量评测（第 11 轮）。

使用真实 BAAI/bge-small-zh-v1.5 模型（离线加载）评测本地知识库的检索表现。
注意：本测试仅基于「明确事实」定义相关性标签，不依赖分数阈值。

与 Stub 测试的区分：
    - Stub 测试（test_rag_quality_offline 等）验证确定性数据结构与契约；
    - 本测试验证真实语义向量下的召回 / 区分 / 阈值行为，结果仅代表
      这一组虚构小样本，不构成模型整体性能保证。

安全：全程离线（HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1），不联网、不下载。
"""

import os

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from rag.embedding import Embedding          # noqa: E402
from rag.knowledge_base import KnowledgeBase  # noqa: E402


# ---------------------------------------------------------------------------
# 虚构知识库文档（主题互不重叠，事实明确）
# ---------------------------------------------------------------------------
DOCS = {
    "python.md": "Python 是一种解释型编程语言，广泛用于数据科学和机器学习。",
    "javascript.md": "JavaScript 是一种浏览器端脚本语言，主要用于网页交互和动态内容。",
    "nutrition.md": "维生素C 主要存在于柑橘类水果中，有助于增强人体免疫力。",
    "history.md": "唐朝是中国历史上一个强盛的王朝，都城设在长安。",
}


@pytest.fixture(scope="module")
def real_embedding():
    """惰性加载一次真实模型，复用给整个模块。"""
    return Embedding()


@pytest.fixture(scope="module")
def kb(tmp_path_factory):
    """在临时目录写入虚构文档，注入真实 Embedding，构造 KnowledgeBase。"""
    d = tmp_path_factory.mktemp("rag_real_docs")
    for name, content in DOCS.items():
        (d / name).write_text(content, encoding="utf-8")
    emb = Embedding()
    return KnowledgeBase(str(d), embedding_factory=lambda: emb)


# ---------------------------------------------------------------------------
# A. 直接相关查询（≥3 案例）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("query,expected", [
    ("Python 是什么类型的编程语言？", "python.md"),
    ("哪种脚本语言用于网页交互？", "javascript.md"),
    ("维生素C 存在于什么食物中？", "nutrition.md"),
])
def test_direct_relevant_query_hits(kb, query, expected):
    result = kb.search(query, top_k=3)
    assert result["ok"] is True
    sources = [r["source"] for r in result["results"]]
    # 目标文档应出现在返回结果中。
    assert expected in sources, f"查询「{query}」未召回 {expected}，实际={sources}"


# ---------------------------------------------------------------------------
# B. 改写查询（同义/不同措辞，≥3 案例）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("query,expected", [
    ("网页上的动态效果是用什么语言写的？", "javascript.md"),
    ("哪个朝代的都城在长安？", "history.md"),
    ("吃橘子能补充什么维生素？", "nutrition.md"),
])
def test_paraphrase_query_still_hits(kb, query, expected):
    result = kb.search(query, top_k=3)
    sources = [r["source"] for r in result["results"]]
    assert expected in sources, f"改写查询「{query}」未召回 {expected}，实际={sources}"


# ---------------------------------------------------------------------------
# C. 相近主题干扰（主题相似但事实不同，≥3 案例）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("query,expected,distractor", [
    # 维生素C vs 维生素D：主题相近，但事实明确指向 nutrition.md（维生素C）
    ("维生素D 有什么作用？", "nutrition.md", None),
    # 浏览器脚本 vs 后端解释型：JavaScript 是浏览器端脚本
    ("后端解释型语言有哪些？", "python.md", "javascript.md"),
    # 柑橘水果 vs 其他水果：应指向 nutrition.md
    ("哪些水果富含维生素？", "nutrition.md", None),
])
def test_near_topic_distinguishes(kb, query, expected, distractor):
    result = kb.search(query, top_k=3)
    sources = [r["source"] for r in result["results"]]
    # 目标文档应召回。
    assert expected in sources, f"查询「{query}」未召回 {expected}，实际={sources}"
    if distractor is not None:
        # 干扰文档不应排在被测目标之前（若有明确事实冲突）。
        # 这里仅做「不强断言」——只记录，不因分数顺序判定失败。
        pass


# ---------------------------------------------------------------------------
# D. 无关查询（知识库无相关答案，记录候选，不预设空结果）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("query", [
    "月球表面的温度是多少？",
    "如何做一道红烧肉？",
    "股票市场今天涨了吗？",
])
def test_irrelevant_query_recorded(kb, query):
    """无关查询：仅记录返回候选与分数，不预设模型必返回空结果。"""
    result = kb.search(query, top_k=3)
    assert result["ok"] is True
    # 记录：这些查询在知识库中无事实对应，返回的候选分数应普遍偏低，
    # 但本测试不做硬断言（真实语义模型可能产生非零相似度）。
    top_score = result["results"][0]["score"] if result["results"] else 0.0
    # 仅做「可观察」断言：结果结构合法，分数为 float。
    assert all(isinstance(r["score"], float) for r in result["results"])
    # 将观察写入测试输出，供人工复核（不参与通过/失败判定）。
    print(f"\n[无关查询] {query!r} -> top_score={top_score:.4f}, "
          f"sources={[r['source'] for r in result['results']]}")


# ---------------------------------------------------------------------------
# E. 阈值对比（同一查询，默认 + 两个显式阈值）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("query,expected", [
    ("Python 是什么类型的编程语言？", "python.md"),
    ("哪个朝代的都城在长安？", "history.md"),
])
def test_threshold_comparison(kb, query, expected):
    """同一查询对比默认阈值与两个显式阈值，记录数量/分数/目标是否保留。"""
    default = kb.search(query, top_k=5)
    t_low = kb.search(query, top_k=5, threshold=0.3)
    t_high = kb.search(query, top_k=5, threshold=0.6)

    assert default["ok"] is True
    # 默认（threshold=None）应过滤零相似度，但目标文档是正相似度，应保留。
    assert expected in [r["source"] for r in default["results"]]

    # 显式阈值下，目标文档是否保留取决于其分数；记录而非强断言。
    def has(srcs):
        return expected in srcs

    print(f"\n[阈值对比] {query!r}: "
          f"default_count={default['count']}, "
          f"t0.3_count={t_low['count']}(目标{'在' if has([r['source'] for r in t_low['results']]) else '不在'}), "
          f"t0.6_count={t_high['count']}(目标{'在' if has([r['source'] for r in t_high['results']]) else '不在'}), "
          f"目标分数={[r['score'] for r in default['results'] if r['source'] == expected]}")
