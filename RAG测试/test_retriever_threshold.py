# -*- coding: utf-8 -*-
"""RAG 6.14：Retriever 可选 similarity threshold 专项测试。

实现语义（与被测代码一致）：
  VectorStore 内部：算分 → 降序排序 → Top-K 截断；
  Retriever：search 之后、仅当显式传入 threshold 时过滤
             （保留 score >= threshold；等于阈值必须保留），
             因此过滤后结果数 <= top_k。
  不传 threshold 时行为与旧版逐字节一致（原样返回 search 结果）。

本测试全程使用人工构造向量（二维），不需要加载 Embedding 模型。

运行方式（均支持）：
  python -m RAG测试.test_retriever_threshold
  python -m pytest -q "RAG测试/test_retriever_threshold.py"
"""

import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.context_builder import ContextBuilder   # noqa: E402
from rag.document import Document                 # noqa: E402
from rag.retriever import Retriever               # noqa: E402
from rag.vector_store import VectorStore          # noqa: E402

# 人工向量设计（query=[1,0]）：
#   A=[1,0] → cos=1.0     B=[1,1] → 0.7071      C=[1,2] → 0.4472   D=[0,1] → 0.0
VEC = {"A": [1, 0], "B": [1, 1], "C": [1, 2], "D": [0, 1]}
QUERY = [1, 0]


def build_store():
    store = VectorStore()
    for name in ("A", "B", "C", "D"):
        store.add(Document(f"文档{name}", {"source": name}), VEC[name])
    return store


def sources(results):
    return [r["document"].metadata["source"] for r in results]


def scores(results):
    return [float(r["score"]) for r in results]


# ----------------------------------------------------------------------
# pytest 模式（全部为真实断言）
# ----------------------------------------------------------------------

def test_old_interface_compatible():
    r = Retriever(build_store())
    # 旧调用形式：默认 top_k=3
    out_default = r.retrieve(QUERY)
    assert sources(out_default) == ["A", "B", "C"]
    # 旧调用形式：显式 top_k
    out_k3 = r.retrieve(QUERY, top_k=3)
    assert sources(out_k3) == ["A", "B", "C"]
    assert scores(out_k3) == sorted(scores(out_k3), reverse=True)


def test_no_threshold_same_as_old_behavior():
    store = build_store()
    r = Retriever(store)
    out = r.retrieve(QUERY, top_k=3)
    baseline = store.search(QUERY, top_k=3)
    # 不传 threshold：内容与直接调用 search 完全一致（旧行为）
    assert [(x["document"].metadata["source"], float(x["score"])) for x in out] == \
           [(x["document"].metadata["source"], float(x["score"])) for x in baseline]


def test_threshold_filters_low_scores():
    r = Retriever(build_store())
    out = r.retrieve(QUERY, top_k=3, threshold=0.5)
    assert sources(out) == ["A", "B"]          # C(0.447) / D(0.0) 被过滤


def test_score_equal_to_threshold_kept():
    r = Retriever(build_store())
    base = r.retrieve(QUERY, top_k=3)
    b_score = float(base[1]["score"])          # B 的实际分数
    out = r.retrieve(QUERY, top_k=3, threshold=b_score)
    assert "B" in sources(out)                 # 恰好等于阈值 → 保留
    assert sources(out) == ["A", "B"]


def test_threshold_with_top_k_combination():
    r = Retriever(build_store())
    # 先 Top-K 截断再过滤：top_k=2 → [A,B]，threshold=0.9 → 只剩 A
    out = r.retrieve(QUERY, top_k=2, threshold=0.9)
    assert sources(out) == ["A"]
    assert len(out) <= 2
    # 组合：top_k=3 + threshold=0.7 → [A,B]，数量与顺序正确
    out2 = r.retrieve(QUERY, top_k=3, threshold=0.7)
    assert sources(out2) == ["A", "B"]
    assert len(out2) <= 3


def test_all_below_threshold_returns_empty():
    r = Retriever(build_store())
    assert r.retrieve(QUERY, top_k=3, threshold=1.5) == []
    # threshold=1.0 时仅 A 恰好等于阈值 → 保留 A
    out = r.retrieve(QUERY, top_k=3, threshold=1.0)
    assert sources(out) == ["A"]


def test_empty_store_returns_empty():
    r = Retriever(VectorStore())
    assert r.retrieve(QUERY) == []
    assert r.retrieve(QUERY, threshold=0.5) == []


def test_zero_vector_boundary():
    store = build_store()
    r = Retriever(store)
    # 零向量查询：全部 score=0.0
    out_old = r.retrieve([0, 0], top_k=4)
    assert len(out_old) == 4 and all(float(x["score"]) == 0.0 for x in out_old)
    # 阈值 0.5 → 全被过滤；阈值 0 → 全保留（0 >= 0）
    assert r.retrieve([0, 0], top_k=4, threshold=0.5) == []
    assert len(r.retrieve([0, 0], top_k=4, threshold=0)) == 4


def test_numpy_vector_compat():
    """numpy 向量输入时阈值比较不崩溃（真实 Embedding 场景）。"""
    store = VectorStore()
    store.add(Document("文档N1", {"source": "N1"}), np.array([1.0, 0.0]))
    store.add(Document("文档N2", {"source": "N2"}), np.array([0.0, 1.0]))
    r = Retriever(store)
    out = r.retrieve(np.array([1.0, 0.0]), top_k=2, threshold=0.5)
    assert sources(out) == ["N1"]


def test_filtered_docs_not_in_context():
    r = Retriever(build_store())
    out = r.retrieve(QUERY, top_k=3, threshold=0.5)   # A、B
    context = ContextBuilder().build(out)
    assert "文档A" in context and "文档B" in context
    assert "文档C" not in context and "文档D" not in context   # 被滤文档不得进入 Context


def test_order_preserved_after_filter():
    r = Retriever(build_store())
    out = r.retrieve(QUERY, top_k=4, threshold=0.4)   # A,B,C
    assert sources(out) == ["A", "B", "C"]
    assert scores(out) == sorted(scores(out), reverse=True)


# ----------------------------------------------------------------------
# 脚本模式
# ----------------------------------------------------------------------

CHECKS = []


def record(name, fn):
    try:
        fn()
        CHECKS.append((name, True, ""))
    except AssertionError as exc:
        CHECKS.append((name, False, str(exc) or "断言失败"))
    except Exception as exc:  # noqa: BLE001
        CHECKS.append((name, False, f"{type(exc).__name__}: {exc}"))


def main():
    r = Retriever(build_store())

    print("=" * 64)
    print("RAG 6.14 Retriever threshold 专项验证（人工向量）")
    print("=" * 64)
    print("原始（不传 threshold, top_k=3）:",
          [(s, round(sc, 6)) for s, sc in zip(sources(r.retrieve(QUERY, top_k=3)),
                                             scores(r.retrieve(QUERY, top_k=3)))])
    for t in (0.0, 0.4, 0.5, 0.7, 1.0, 1.5):
        out = r.retrieve(QUERY, top_k=3, threshold=t)
        print(f"threshold={t:<4} → {[(s, round(sc, 6)) for s, sc in zip(sources(out), scores(out))]}")

    record("旧接口兼容（默认/显式 top_k）", test_old_interface_compatible)
    record("不传 threshold 与旧行为一致", test_no_threshold_same_as_old_behavior)
    record("显式 threshold 过滤低分结果", test_threshold_filters_low_scores)
    record("score == threshold 时保留", test_score_equal_to_threshold_kept)
    record("threshold × top_k 组合正确", test_threshold_with_top_k_combination)
    record("全部低于阈值 → []", test_all_below_threshold_returns_empty)
    record("空向量库 → []", test_empty_store_returns_empty)
    record("零向量边界", test_zero_vector_boundary)
    record("numpy 向量兼容", test_numpy_vector_compat)
    record("被滤文档不进入 Context", test_filtered_docs_not_in_context)
    record("过滤后顺序保持降序", test_order_preserved_after_filter)

    print("-" * 64)
    for name, ok, detail in CHECKS:
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {name}" + (f"  （{detail}）" if detail else ""))
    total = len(CHECKS)
    passed = sum(1 for _, ok, _ in CHECKS if ok)
    print("-" * 64)
    print(f"统计：总 {total} | PASS {passed} | FAIL {total - passed}")
    print("=" * 64)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
