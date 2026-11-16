# -*- coding: utf-8 -*-
"""RAG 6.6：VectorStore 相似度排序与 Top-K 独立验证。

验证目标（不修改生产代码，仅验证）：
  查询向量 → cosine similarity → score 从高到低排序 → 返回 Top-K。
  结果字段统一为 document / vector / score（无旧字段 distance）。

测试数据（按 6.6 任务指定）：
  query = [1, 2, 3]
  A = [1, 2, 3]      （与 query 同方向，score 应为 1.0）
  B = [1, 2, 4]
  C = [10, 20, 40]

  注意（数据观察，如实记录）：C = 10 × B，二者方向完全相同，因此
  B 与 C 的 cosine 在数学上精确相等（仅区分出 A 与其余两者）；
  本文件不写死 B/C 的小数值，只验证与 A 的相对关系及排序/Top-K 行为。

运行方式（均支持）：
  python "RAG测试/test_vector_store_topk_verify.py"     （脚本模式 + 退出码）
  pytest "RAG测试/test_vector_store_topk_verify.py"     （测试模式）
"""

import sys
from pathlib import Path

# 保证从任意工作目录/任意运行方式都能 import 项目根下的 rag 包
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.document import Document          # noqa: E402
from rag.vector_store import VectorStore   # noqa: E402


def build_store():
    store = VectorStore()
    doc_a = Document("文档A", {"source": "A.txt"})
    doc_b = Document("文档B", {"source": "B.txt"})
    doc_c = Document("文档C", {"source": "C.txt"})
    store.add(doc_a, [1, 2, 3])
    store.add(doc_b, [1, 2, 4])
    store.add(doc_c, [10, 20, 40])
    return store, doc_a, doc_b, doc_c


QUERY = [1, 2, 3]


def run_checks():
    """执行全部验证，返回 (结果记录 dict, 是否全部通过)。"""
    store, doc_a, doc_b, doc_c = build_store()
    checks = []

    def record(name, passed, detail=""):
        checks.append((name, bool(passed), detail))

    # ---- 1. 相似度计算（相对关系，不写死 B/C 小数） ----
    results3 = store.search(QUERY, top_k=3)
    by_doc = {}
    for r in results3:
        if r["document"] is doc_a:
            by_doc["A"] = r["score"]
        elif r["document"] is doc_b:
            by_doc["B"] = r["score"]
        elif r["document"] is doc_c:
            by_doc["C"] = r["score"]

    score_a, score_b, score_c = by_doc["A"], by_doc["B"], by_doc["C"]

    record("A 与 query 同方向 → score ≈ 1.0", abs(score_a - 1.0) < 1e-9,
           f"A={score_a:.10f}")
    record("A score > B score", score_a > score_b, f"{score_a:.6f} > {score_b:.6f}")
    record("A score > C score", score_a > score_c, f"{score_a:.6f} > {score_c:.6f}")

    # ---- 2. 排序 ----
    scores3 = [r["score"] for r in results3]
    record("结果按 score 从高到低排序",
           scores3 == sorted(scores3, reverse=True),
           f"顺序={[round(s, 6) for s in scores3]}")
    record("最相似文档 A 排在第一位",
           results3[0]["document"] is doc_a,
           f"第一位={results3[0]['document'].metadata.get('source')}")

    # ---- 3. Top-K ----
    top2 = store.search(QUERY, top_k=2)
    record("top_k=2 → 返回 2 条", len(top2) == 2, f"len={len(top2)}")
    record("top_k=2 → 包含 A 且 [0] 为 A", top2[0]["document"] is doc_a)
    record("top_k=2 → 第二条来自并列的 B 或 C",
           top2[1]["document"] in (doc_b, doc_c),
           f"第二条={top2[1]['document'].metadata.get('source')}")
    record("top_k=3 → 覆盖全部 3 条文档",
           len(store.search(QUERY, top_k=3)) == 3)

    # ---- 4. 结果字段（统一 score；不得再有旧字段 distance） ----
    field_ok = all(
        ("document" in r and "vector" in r and "score" in r)
        and ("distance" not in r)
        for r in results3
    )
    record("结果字段包含 document/vector/score 且无 distance", field_ok)

    # ---- 5. 边界情况（快速补充，非核心重点） ----
    top1 = store.search(QUERY, top_k=1)
    record("top_k=1 → 仅返回 A", len(top1) == 1 and top1[0]["document"] is doc_a)

    top10 = store.search(QUERY, top_k=10)
    record("top_k 大于文档数 → 返回全部 3 条", len(top10) == 3)

    empty = VectorStore().search(QUERY, top_k=3)
    record("空 VectorStore → 返回空列表", empty == [])

    zero = store.search([0, 0, 0], top_k=3)
    record("零向量查询 → 全部 score 为 0.0 且不崩溃",
           len(zero) == 3 and all(r["score"] == 0.0 for r in zero))

    all_passed = all(ok for _, ok, _ in checks)
    return {
        "scores": {"A": score_a, "B": score_b, "C": score_c},
        "sorted_sources": [r["document"].metadata.get("source") for r in results3],
        "checks": checks,
        "all_passed": all_passed,
    }


# ----------------------------------------------------------------------
# 测试模式（pytest）
# ----------------------------------------------------------------------

def test_similarity_relative_order():
    r = run_checks()
    s = r["scores"]
    assert abs(s["A"] - 1.0) < 1e-9
    assert s["A"] > s["B"]
    assert s["A"] > s["C"]


def test_sorting_and_topk():
    store, doc_a, doc_b, doc_c = build_store()

    results3 = store.search(QUERY, top_k=3)
    scores3 = [r["score"] for r in results3]
    assert scores3 == sorted(scores3, reverse=True)
    assert results3[0]["document"] is doc_a

    top2 = store.search(QUERY, top_k=2)
    assert len(top2) == 2
    assert top2[0]["document"] is doc_a
    assert top2[1]["document"] in (doc_b, doc_c)

    assert len(store.search(QUERY, top_k=1)) == 1
    assert len(store.search(QUERY, top_k=10)) == 3


def test_result_fields():
    store, doc_a, doc_b, doc_c = build_store()
    for r in store.search(QUERY, top_k=3):
        assert "document" in r and "vector" in r and "score" in r
        assert "distance" not in r


def test_edge_cases():
    store, doc_a, doc_b, doc_c = build_store()
    assert VectorStore().search(QUERY, top_k=3) == []
    zero = store.search([0, 0, 0], top_k=3)
    assert len(zero) == 3 and all(r["score"] == 0.0 for r in zero)


# ----------------------------------------------------------------------
# 脚本模式（直接运行：打印全部结果并返回退出码）
# ----------------------------------------------------------------------

def main():
    r = run_checks()
    s = r["scores"]

    print("=" * 60)
    print("RAG 6.6 VectorStore 相似度排序与 Top-K 验证")
    print("=" * 60)
    print(f"query = {QUERY}")
    print(f"A = [1, 2, 3]   B = [1, 2, 4]   C = [10, 20, 40]")
    print("-" * 60)
    print(f"A score = {s['A']:.6f}")
    print(f"B score = {s['B']:.6f}")
    print(f"C score = {s['C']:.6f}")
    print(f"最终排序（top_k=3）= {r['sorted_sources']}")
    print("-" * 60)
    for name, ok, detail in r["checks"]:
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {name}" + (f"  （{detail}）" if detail else ""))
    print("=" * 60)
    print("全部验证通过" if r["all_passed"] else "存在失败项，请查看上方 FAIL 行")
    return 0 if r["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
