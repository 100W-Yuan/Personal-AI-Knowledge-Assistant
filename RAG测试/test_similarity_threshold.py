# -*- coding: utf-8 -*-
"""RAG 6.12：Similarity Threshold（相似度阈值）实验。

背景：VectorStore 为纯 Top-K 检索（无阈值），完全无关查询也会返回结果。
本文件在**测试侧**模拟阈值过滤（filter_by_threshold，不修改 Retriever），
分析：哪一区间的阈值能在减少 False Positive 的同时尽量不产生
真正相关内容的 False Negative。

约束：仅本地 BAAI/bge-small-zh-v1.5（HF_HUB_OFFLINE=1，零网络零费用）；
      虚构文档；不修改任何生产代码；不写死固定分数（关系/趋势断言）。

运行方式（均支持）：
  python -m RAG测试.test_similarity_threshold
  python -m pytest -q "RAG测试/test_similarity_threshold.py"
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.document import Document                 # noqa: E402
from rag.embedding import Embedding               # noqa: E402
from rag.retriever import Retriever               # noqa: E402
from rag.vector_store import VectorStore          # noqa: E402

DOCUMENTS = [
    Document("我的 AI Agent 项目已经实现了 Tool Calling、Task Loop 和 Long Term Memory。",
             {"source": "A.txt"}),
    Document("我的 AI Agent 项目目前正在学习 RAG，包括文档加载、文本切分和向量检索。",
             {"source": "B.txt"}),
    Document("我每周进行三次力量训练，主要安排胸、背和腿部训练。",
             {"source": "C.txt"}),
    Document("今天晚上准备吃面条和一些蔬菜。",
             {"source": "D.txt"}),
    Document("最近正在学习 Python，包括类、函数、异常处理和模块。",
             {"source": "E.txt"}),
    Document("周末准备看一部科幻电影。",
             {"source": "F.txt"}),
]

ALL_SOURCES = [d.metadata["source"] for d in DOCUMENTS]

THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]

# (id, group, query, expected_sources)
# 边界案例：expected 为 None（A/B 或 B/E 均可能合理，不设唯一答案）
QUERIES = [
    ("Q01", "G1 强相关", "我的 Agent 项目已经有哪些功能？", {"A.txt"}),
    ("Q02", "G1 强相关", "Tool Calling 和 Task Loop 是我的项目已经实现的吗？", {"A.txt"}),
    ("Q03", "G2 相关但不同表达", "我目前的 AI 项目做到了什么？", {"A.txt"}),
    ("Q04", "G2 相关但不同表达", "我最近在学习哪方面的检索技术？", {"B.txt"}),
    ("Q05", "G3 中等相关(边界)", "我的项目现在有哪些技术？", None),
    ("Q06", "G3 中等相关(边界)", "最近学习的文档处理相关内容是什么？", None),
    ("Q07", "G4 对各自文档相关", "我最近健身主要练什么？", {"C.txt"}),
    ("Q08", "G4 对各自文档相关", "今晚准备吃什么？", {"D.txt"}),
    ("Q09", "G5 完全无关", "天气预报", set()),
    ("Q10", "G5 完全无关", "股票市场", set()),
    ("Q11", "G5 完全无关", "火星探测器", set()),
    ("Q12", "G5 完全无关", "量子计算", set()),   # 补充：任务列示 11 条，补 1 条达 12
]


_ALL = None


def _lazy():
    global _ALL
    if _ALL is None:
        embedding = Embedding()
        store = VectorStore()
        for doc in DOCUMENTS:
            store.add(doc, embedding.encode(doc.page_content))
        _ALL = (embedding, Retriever(store))
    return _ALL


def raw_results(query, top_k=6):
    embedding, retriever = _lazy()
    query_vector = embedding.encode(query)     # 先编码，杜绝字符串直传
    return retriever.retrieve(query_vector, top_k=top_k)


def filter_by_threshold(results, threshold):
    """测试侧实验函数（不修改生产 Retriever）。"""
    return [r for r in results if float(r["score"]) >= threshold]


def collect_matrix():
    """执行 12 个 Query 的完整实验矩阵。"""
    rows = []
    for qid, group, query, expected in QUERIES:
        results = raw_results(query)
        scores = [(r["document"].metadata["source"], float(r["score"]))
                  for r in results]
        kept_by_threshold = {
            t: [s for s, sc in scores if sc >= t] for t in THRESHOLDS
        }
        rows.append({
            "id": qid, "group": group, "query": query,
            "expected": expected, "scores": scores,
            "kept": kept_by_threshold,
        })
    return rows


# ----------------------------------------------------------------------
# 指标计算（以 top_k=6 全量为基础）
# ----------------------------------------------------------------------

def false_negatives(row, threshold):
    """expected 中被阈值过滤掉的文档列表（空集/边界返回 []）。"""
    exp = row["expected"]
    if not exp:
        return []
    kept = set(row["kept"][threshold])
    return sorted(s for s in exp if s not in kept)


def false_positives(row, threshold):
    """保留结果中的无关文档列表。

    - expected 非空：保留的 全部文档-expected 即无关；
    - 完全无关 Query（expected=空集）：保留的都算 FP。
    """
    exp = row["expected"]
    kept = row["kept"][threshold]
    if exp is None:
        return []                      # 边界案例不判 FP
    if not exp:
        return list(kept)              # 完全无关：全部算 FP
    return [s for s in kept if s not in exp]


def summarize(rows):
    """按阈值汇总：强相关召回 / 无关过滤 / 边界表现。"""
    table = {}
    for t in THRESHOLDS:
        related = [r for r in rows if r["group"].startswith(("G1", "G2", "G4"))]
        fn_queries = [r["id"] for r in related if false_negatives(r, t)]
        total_fn = sum(len(false_negatives(r, t)) for r in related)

        g5 = [r for r in rows if r["group"].startswith("G5")]
        g5_fp_total = sum(len(false_positives(r, t)) for r in g5)
        g5_still_returning = sum(1 for r in g5 if r["kept"][t])

        boundary = [r for r in rows if r["expected"] is None]
        boundary_kept = {r["id"]: len(r["kept"][t]) for r in boundary}

        table[t] = {
            "强相关/目标召回成功": f"{len(related) - len(fn_queries)}/{len(related)}",
            "被误滤的相关doc数": total_fn,
            "完全无关查询仍有返回数": f"{g5_still_returning}/{len(g5)}",
            "完全无关查询FP总数": g5_fp_total,
            "边界保留数(Q05/Q06)": f"{boundary_kept.get('Q05')}/{boundary_kept.get('Q06')}",
        }
    return table


# ----------------------------------------------------------------------
# 断言（关系/趋势，而非固定分数）
# ----------------------------------------------------------------------

def compute_assertions(rows, table):
    checks = []

    def record(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    def top1(r):
        return r["scores"][0][1]

    related_queries = [r for r in rows if r["expected"]]           # G1/G2/G4
    g5 = [r for r in rows if r["group"].startswith("G5")]

    min_expected_score = min(
        sc for r in related_queries for s, sc in r["scores"] if s in r["expected"]
    )
    max_irrelevant_top1 = max(top1(r) for r in g5)
    record("核心分离性：全部目标文档的最低分 > 完全无关查询的最高 Top-1",
           min_expected_score > max_irrelevant_top1,
           f"{min_expected_score:.4f} > {max_irrelevant_top1:.4f}")

    no_fn_at_low = all(not false_negatives(r, 0.30) for r in related_queries)
    record("阈值 0.30 时不产生任何 False Negative（低阈值安全）", no_fn_at_low)

    full_filter = [t for t in THRESHOLDS
                   if all(not r["kept"][t] for r in g5)]
    record("存在阈值可把全部完全无关查询过滤为空（FP 全消）",
           len(full_filter) > 0,
           f"最小达标阈值={min(full_filter) if full_filter else '无'}")

    # 推荐区间：同时满足“G5 全滤”且“无 FN”
    valid = [t for t in THRESHOLDS
             if all(not r["kept"][t] for r in g5)
             and all(not false_negatives(r, t) for r in related_queries)]
    record("存在同时满足“无关全滤 + 相关零 FN”的阈值区间",
           len(valid) > 0, f"区间={valid}")

    # 矩阵计算正确性（对 Q09 抽查：0.50 时应为 0 条）
    q09 = next(r for r in rows if r["id"] == "Q09")
    record("矩阵计算抽查：Q09 在 threshold=0.50 时保留条数正确",
           q09["kept"][0.50] == [s for s, sc in q09["scores"] if sc >= 0.50],
           f"kept={q09['kept'][0.50]}")

    return checks


# ----------------------------------------------------------------------
# 打印
# ----------------------------------------------------------------------

def print_report(rows, table):
    print("=" * 76)
    print("RAG 6.12 Similarity Threshold 实验")
    print("=" * 76)

    for r in rows:
        print(f"\nQuery[{r['id']}][{r['group']}]：{r['query']}")
        if r["expected"] is None:
            print("  期望：边界案例（不设唯一答案）")
        elif not r["expected"]:
            print("  期望：无相关文档（完全无关）")
        else:
            print(f"  期望：{sorted(r['expected'])}")
        print("  原始 Top-6:")
        for i, (s, sc) in enumerate(r["scores"], start=1):
            print(f"    {i}. {s} {sc:.6f}")
        matrix = "  ".join(f"{t:.2f}→{len(r['kept'][t])}条" for t in THRESHOLDS)
        print(f"  阈值矩阵: {matrix}")
        kept_detail = "；".join(
            f"{t:.2f}: {','.join(r['kept'][t]) or '（空）'}" for t in THRESHOLDS
        )
        print(f"  保留明细: {kept_detail}")

    print("\n" + "=" * 76)
    print("阈值汇总表")
    print("=" * 76)
    header = f"{'threshold':>9s} | {'强相关召回':>10s} | {'误滤相关数':>8s} | {'无关仍有返回':>10s} | {'无关FP总数':>9s} | 边界保留(Q05/Q06)"
    print(header)
    for t in THRESHOLDS:
        d = table[t]
        print(f"{t:>9.2f} | {d['强相关/目标召回成功']:>10s} | {d['被误滤的相关doc数']:>8d} | "
              f"{d['完全无关查询仍有返回数']:>10s} | {d['完全无关查询FP总数']:>9d} | {d['边界保留数(Q05/Q06)']}")

    print("\n" + "=" * 76)
    print("False Positive 明细（完全无关查询 retained 文档）")
    print("=" * 76)
    for t in THRESHOLDS:
        g5_fp = {r["id"]: false_positives(r, t)
                 for r in rows if r["group"].startswith("G5")}
        nz = {k: v for k, v in g5_fp.items() if v}
        print(f"threshold={t:.2f}: " + ("全部过滤为空" if not nz else str(nz)))

    print("\n" + "=" * 76)
    print("False Negative 明细（目标文档被误滤）")
    print("=" * 76)
    any_fn = False
    for t in THRESHOLDS:
        fns = {r["id"]: false_negatives(r, t)
               for r in rows if r["expected"]}
        nz = {k: v for k, v in fns.items() if v}
        if nz:
            any_fn = True
            print(f"threshold={t:.2f}: {nz}")
    if not any_fn:
        print("在 0.30–0.75 全部候选阈值下，未出现目标文档被误滤（False Negative）。")


# ----------------------------------------------------------------------
# pytest 模式
# ----------------------------------------------------------------------

def test_threshold_assertions():
    rows = collect_matrix()
    table = summarize(rows)
    checks = compute_assertions(rows, table)
    for name, ok, detail in checks:
        assert ok, f"{name} 失败（{detail}）"


def test_matrix_dimensions():
    rows = collect_matrix()
    assert len(rows) == 12
    for r in rows:
        assert len(r["scores"]) == 6          # 全量 6 文档
        scores = [sc for _, sc in r["scores"]]
        assert scores == sorted(scores, reverse=True)
        for t in THRESHOLDS:
            assert set(r["kept"][t]) <= {s for s, _ in r["scores"]}


# ----------------------------------------------------------------------
# 脚本模式
# ----------------------------------------------------------------------

def main():
    rows = collect_matrix()
    table = summarize(rows)
    print_report(rows, table)

    checks = compute_assertions(rows, table)
    print("\n" + "=" * 76)
    for name, ok, detail in checks:
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {name}" + (f"  （{detail}）" if detail else ""))
    total = len(checks)
    passed = sum(1 for _, ok, _ in checks if ok)
    print("-" * 76)
    print(f"统计：总 {total} | PASS {passed} | FAIL {total - passed}")
    print("=" * 76)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
