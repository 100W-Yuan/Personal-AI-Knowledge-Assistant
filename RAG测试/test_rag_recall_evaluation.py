# -*- coding: utf-8 -*-
"""RAG 检索效果评估（Recall@K）· 默认离线、可重复。

口径说明（务必与代码一致，不得编造准确率）：
- 数据集 DATASET 为**固定**的中文文档（文件名 -> 正文），全部写入 pytest 临时目录。
- 每条 QUERY 有显式相关来源集合（ground truth，人工标注）。
- 检索使用**确定性 Stub Embedding**（关键词 one-hot，见 KeywordEmbedding），
  零真实模型、零网络；相似度由 VectorStore 的余弦计算。
- Recall@K = |检索 Top-K 来源 ∩ 相关来源| / |相关来源|。
- 阈值沿用 KnowledgeBase 默认（过滤相似度为 0 的结果）。

本测试同时输出一份 Recall@K 汇总表，便于重复运行对比。
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.knowledge_base import KnowledgeBase  # noqa: E402

DATASET = {
    "退货政策.md": "退货政策：商品支持七天无理由退货，退货需要保持商品完好。",
    "物流配送.md": "配送说明：默认使用顺丰快递，配送时效为一到三天。",
    "支付方式.md": "支付说明：支持微信支付和支付宝，不支持货到付款。",
    "售后保修.md": "保修说明：电子产品提供一年质保，人为损坏不在保修范围。",
}

# (查询, 相关来源集合, 达到全召回所需的 K)
QUERIES = [
    ("如何申请退货", {"退货政策.md"}, 1),
    ("配送要多久", {"物流配送.md"}, 1),
    ("支持哪些支付方式", {"支付方式.md"}, 1),
    ("保修期是多久", {"售后保修.md"}, 1),
    ("退货和保修分别怎么处理", {"退货政策.md", "售后保修.md"}, 2),
]


class KeywordEmbedding:
    """确定性 one-hot 编码（关键词维度），归一化保证余弦稳定。"""

    KW = ["退货", "配送", "支付", "保修"]

    def encode(self, text):
        vec = [1.0 if kw in text else 0.0 for kw in self.KW]
        norm = (sum(v * v for v in vec)) ** 0.5
        return [0.0] * len(vec) if norm == 0.0 else [v / norm for v in vec]


def build_kb(tmp_path):
    for name, content in DATASET.items():
        (tmp_path / name).write_text(content, encoding="utf-8")
    return KnowledgeBase(str(tmp_path), embedding_factory=lambda: KeywordEmbedding())


def recall_at_k(sources, relevant, k):
    return len(set(sources[:k]) & set(relevant)) / len(relevant)


def evaluate(tmp_path, top_k=5):
    kb = build_kb(tmp_path)
    rows = []
    for query, relevant, expected_k in QUERIES:
        result = kb.search(query, top_k=top_k)
        sources = [item["source"] for item in result["results"]]
        recall = recall_at_k(sources, relevant, expected_k)
        rows.append((query, sorted(relevant), sources, recall, expected_k))
    return rows


def test_retrieval_recalls_expected_source(tmp_path):
    rows = evaluate(tmp_path)
    for query, relevant, sources, recall, expected_k in rows:
        assert recall == 1.0, (
            f"Recall@{expected_k} 未命中：query={query!r} sources={sources}"
        )


def test_multi_relevant_query_recall_at_k2(tmp_path):
    kb = build_kb(tmp_path)
    result = kb.search("退货和保修分别怎么处理", top_k=5)
    sources = [item["source"] for item in result["results"]]
    # 两个相关来源都应在 Top-2 内召回。
    assert recall_at_k(sources, {"退货政策.md", "售后保修.md"}, 2) == 1.0


def test_irrelevant_query_returns_no_result(tmp_path):
    kb = build_kb(tmp_path)
    result = kb.search("今天天气怎么样", top_k=5)
    assert result["ok"] is True
    assert result["count"] == 0
    assert result["results"] == []


def test_empty_knowledge_dir_is_init_error(tmp_path):
    kb = KnowledgeBase(
        str(tmp_path / "empty_dir"),
        embedding_factory=lambda: KeywordEmbedding(),
    )
    result = kb.search("如何申请退货")
    assert result["ok"] is False
    assert result["stage"] == "init"


def test_recall_report_is_reproducible(tmp_path):
    """同一数据集多次运行结果应完全一致（确定性 stub）。"""
    first = evaluate(tmp_path)
    second = evaluate(tmp_path)
    assert first == second
    # 打印汇总表，便于人工核对（非断言，仅供重复运行对比）。
    print("\n=== Recall@K 评估汇总（Stub Embedding，top_k=5）===")
    for query, relevant, sources, recall, expected_k in first:
        print(
            f"  query={query!r:24} relevant={relevant!r:34} "
            f"recall@{expected_k}={recall}"
        )
    print("====================================================")
