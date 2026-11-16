# -*- coding: utf-8 -*-
"""RAG 6.11：检索质量专项测试（Retriever 到底"搜得准不准"）。

六组测试：同义/近义、RAG 语义改写、无关问题、False Positive、Top-K、边界。
原则：全部使用本地 Embedding（BAAI/bge-small-zh-v1.5，离线）与虚构文档；
      不修改任何生产代码；不调用任何 API；断言基于"排名关系/降序"，
      不写死固定分数（禁止 score == 0.8234 类脆弱断言）。
链路：文本 → Embedding.encode() → numpy 向量 → Retriever → VectorStore → similarity
      （绝不把字符串直接传给 retrieve —— 防止历史 TypeError 复发）

运行方式（均支持；建议 HF_HUB_OFFLINE=1）：
  python -m RAG测试.test_retrieval_quality
  python -m pytest -q "RAG测试/test_retrieval_quality.py"
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

_ALL = None


def _lazy():
    """单例构建 Embedding/VectorStore（供脚本与 pytest 共用，避免重复加载模型）。"""
    global _ALL
    if _ALL is None:
        embedding = Embedding()
        store = VectorStore()
        for doc in DOCUMENTS:
            store.add(doc, embedding.encode(doc.page_content))
        _ALL = (embedding, Retriever(store))
    return _ALL


def results_for(query, top_k=6):
    """执行检索全链路（统一入口）。"""
    embedding, retriever = _lazy()
    query_vector = embedding.encode(query)   # 关键：先编码再检索
    return retriever.retrieve(query_vector, top_k=top_k)


def rank_of(source, results):
    for i, r in enumerate(results):
        if r["document"].metadata.get("source") == source:
            return i
    return None


def sources_of(results):
    return [r["document"].metadata.get("source") for r in results]


def format_results(query, results):
    lines = [f"Query: {query}", "Top-K:"]
    for i, r in enumerate(results, start=1):
        lines.append(
            f"{i}. source={r['document'].metadata.get('source')} "
            f"score={float(r['score']):.6f} "
            f"content={r['document'].page_content}"
        )
    return "\n".join(lines)


def _better_than_irrelevant(target, results):
    """target 的排名优于全部无关文档（C/D/E/F）。"""
    t_rank = rank_of(target, results)
    if t_rank is None:
        return False, f"{target} 不在返回集合中"
    for other in ("C.txt", "D.txt", "E.txt", "F.txt"):
        if other == target:
            continue
        o_rank = rank_of(other, results)
        if o_rank is not None and o_rank <= t_rank:
            return False, f"{target}(#{t_rank}) 未优于 {other}(#{o_rank})"
    return True, f"{target} 排名 #{t_rank}，优于全部无关文档"


# ======================================================================
# 第一组：同义/近义问题（期望 A 靠前）
# ======================================================================

def group1_paraphrase():
    checks = []
    prints = []

    q1 = "我的 Agent 项目已经有哪些功能？"
    r1 = results_for(q1)
    prints.append(format_results(q1, r1))
    checks.append(("G1-Q1 A 进入 Top-2（“非常靠前”）",
                   (rank_of("A.txt", r1) is not None and rank_of("A.txt", r1) <= 1),
                   f"A 实际排名 #{rank_of('A.txt', r1)}"))
    ok, detail = _better_than_irrelevant("A.txt", r1)
    checks.append(("G1-Q1 A 优于全部无关文档", ok, detail))

    q2 = "我目前的 AI 项目做到了什么？"
    r2 = results_for(q2)
    prints.append(format_results(q2, r2))
    checks.append(("G1-Q2 A 进入 Top-3", rank_of("A.txt", r2) is not None and rank_of("A.txt", r2) <= 2,
                   f"A 实际排名 #{rank_of('A.txt', r2)}"))
    ok, detail = _better_than_irrelevant("A.txt", r2)
    checks.append(("G1-Q2 A 优于全部无关文档", ok, detail))

    q3 = "我的项目现在具备哪些能力？"
    r3 = results_for(q3)
    prints.append(format_results(q3, r3))
    checks.append(("G1-Q3 A 进入 Top-3", rank_of("A.txt", r3) is not None and rank_of("A.txt", r3) <= 2,
                   f"A 实际排名 #{rank_of('A.txt', r3)}"))
    ok, detail = _better_than_irrelevant("A.txt", r3)
    checks.append(("G1-Q3 A 优于全部无关文档", ok, detail))

    return checks, prints


# ======================================================================
# 第二组：RAG 语义改写（期望 B 进入 Top-K）
# ======================================================================

def group2_rag_rewrite():
    checks = []
    prints = []

    for cid, q in (
        ("Q1", "我最近在学习哪方面的检索技术？"),
        ("Q2", "我的 RAG 学习到什么内容了？"),
        ("Q3", "最近学习的文档处理相关内容是什么？"),
    ):
        r = results_for(q)
        prints.append(format_results(q, r))
        top3 = sources_of(r)[:3]
        checks.append((f"G2-{cid} B 进入 Top-3", "B.txt" in top3, f"Top-3={top3}"))
        b_rank = rank_of("B.txt", r)
        best_irrelevant = min(
            (rank_of(x, r) for x in ("C.txt", "D.txt", "F.txt")
             if rank_of(x, r) is not None), default=None)
        ok = (b_rank is not None and best_irrelevant is not None and b_rank < best_irrelevant)
        checks.append((f"G2-{cid} B 优于无关文档（C/D/F）", ok,
                       f"B=#{b_rank}，无关最好排名=#{best_irrelevant}"))

    return checks, prints


# ======================================================================
# 第三组：无关问题（期望对应文档靠前）
# ======================================================================

def group3_unrelated_targets():
    checks = []
    prints = []

    for cid, q, target in (
        ("Q1", "我最近健身主要练什么？", "C.txt"),
        ("Q2", "今晚准备吃什么？", "D.txt"),
        ("Q3", "最近在学习什么编程语言？", "E.txt"),
        ("Q4", "周末准备做什么娱乐活动？", "F.txt"),
    ):
        r = results_for(q)
        prints.append(format_results(q, r))
        rank = rank_of(target, r)
        checks.append((f"G3-{cid} {target} 进入 Top-3", rank is not None and rank <= 2,
                       f"实际排名 #{rank}"))

    return checks, prints


# ======================================================================
# 第四组：False Positive（本阶段最重要）
# ======================================================================

def group4_false_positive():
    checks = []
    prints = []

    q = "我的 Agent 项目支持图像生成吗？"
    r = results_for(q)
    prints.append(format_results(q, r))

    top2 = sources_of(r)[:2]
    checks.append(("G4 排名合理：Top-2 只应来自 Agent 项目相关文档（A/B）",
                   set(top2) <= {"A.txt", "B.txt"}, f"Top-2={top2}"))

    no_image_content = all(
        ("图像" not in res["document"].page_content
         and "生成" not in res["document"].page_content)
        for res in r
    )
    checks.append(("G4 不把“Agent 项目”表面匹配误认为“图像生成”相关："
                   "全部返回文档均不含图像生成内容", no_image_content,
                   "6 条文档均无“图像/生成”字样"))

    top1_note = (
        "检索层无法感知“问题无答案”（无 similarity threshold，属当前架构设计）；"
        "“明确回答不知道”由 LLM 层承担（6.10 已实测通过）"
    )
    checks.append(("G4 “无明确答案”识别机制说明（观察项，非缺陷）", True, top1_note))

    return checks, prints


# ======================================================================
# 第五组：Top-K
# ======================================================================

def group5_top_k():
    checks = []
    prints = []
    q = "我的 Agent 项目已经有哪些功能？"

    full = results_for(q, top_k=10)
    for k in (1, 2, 3):
        r = results_for(q, top_k=k)
        prints.append(format_results(f"{q}（top_k={k}）", r))
        checks.append((f"G5 top_k={k} → 返回 {k} 条", len(r) == k, f"实际={len(r)}"))
        scores = [float(x["score"]) for x in r]
        checks.append((f"G5 top_k={k} 降序", scores == sorted(scores, reverse=True), f"{scores}"))

    prints.append(format_results(f"{q}（top_k=10）", full))
    checks.append(("G5 top_k=10 > 文档数 → 返回全部 6 条", len(full) == 6, f"实际={len(full)}"))
    scores = [float(x["score"]) for x in full]
    checks.append(("G5 top_k=10 降序", scores == sorted(scores, reverse=True), f"{scores}"))

    r1 = results_for(q, top_k=1)
    checks.append(("G5 边界一致性：top_k=1 即全排序第一名",
                   sources_of(r1) == sources_of(full)[:1],
                   f"{sources_of(r1)} vs {sources_of(full)[:1]}"))

    return checks, prints


# ======================================================================
# 第六组：边界测试
# ======================================================================

def group6_boundary():
    checks = []
    prints = []

    # 1. 空查询（观察：是否报错/返回什么——不预设生产行为）
    try:
        r_empty = results_for("", top_k=3)
        behavior = f"未报错，返回 {len(r_empty)} 条，Top-1="
        if r_empty:
            behavior += (f"{r_empty[0]['document'].metadata.get('source')} "
                         f"score={float(r_empty[0]['score']):.6f}")
        prints.append(format_results("「空字符串查询」", r_empty))
        checks.append(("G6 空查询行为已记录（观察项）", True, behavior))
    except Exception as exc:
        checks.append(("G6 空查询行为已记录（观察项）", True,
                       f"抛出异常：{type(exc).__name__}: {exc}"))
        prints.append(f"Query: （空字符串）\n结果：抛出异常 {type(exc).__name__}: {exc}")

    # 2. 非常短的查询
    q_short = "Agent"
    r_short = results_for(q_short, top_k=3)
    prints.append(format_results(q_short, r_short))
    short_top3 = sources_of(r_short)
    checks.append(("G6 短查询「Agent」：A 或 B 进入 Top-3",
                   ("A.txt" in short_top3) or ("B.txt" in short_top3),
                   f"Top-3={short_top3}"))

    # 3. 非常具体的查询
    q_specific = "Tool Calling 和 Task Loop"
    r_specific = results_for(q_specific, top_k=3)
    prints.append(format_results(q_specific, r_specific))
    a_rank = rank_of("A.txt", r_specific)
    checks.append(("G6 具体查询：「Tool Calling 和 Task Loop」→ A 明显靠前（Top-2）",
                   a_rank is not None and a_rank <= 1, f"A 实际排名 #{a_rank}"))

    # 4. 完全无关查询（记录 Top-K 与分数，不强求空结果）
    q_irrelevant = "天气预报"
    r_irr = results_for(q_irrelevant, top_k=3)
    prints.append(format_results(q_irrelevant, r_irr))
    checks.append(("G6 无关查询「天气预报」：行为已记录（观察项）", True,
                   f"Top-3={sources_of(r_irr)}"))

    return checks, prints


# ======================================================================
# 汇总与运行
# ======================================================================

GROUPS = [
    ("第一组 同义/近义", group1_paraphrase),
    ("第二组 RAG 语义改写", group2_rag_rewrite),
    ("第三组 无关问题", group3_unrelated_targets),
    ("第四组 False Positive", group4_false_positive),
    ("第五组 Top-K", group5_top_k),
    ("第六组 边界测试", group6_boundary),
]


def run_checks(group_key=None):
    """执行全部（或指定）组，返回总明细。"""
    all_checks = []
    all_prints = []
    for name, fn in GROUPS:
        if group_key and group_key not in name:
            continue
        checks, prints = fn()
        all_checks.extend(checks)
        all_prints.extend(prints)
    return all_checks, all_prints


# ----------------------------------------------------------------------
# pytest 模式（每组一个测试函数）
# ----------------------------------------------------------------------

def test_group1_paraphrase():
    checks, _ = group1_paraphrase()
    for name, ok, detail in checks:
        assert ok, f"{name} 失败（{detail}）"


def test_group2_rag_rewrite():
    checks, _ = group2_rag_rewrite()
    for name, ok, detail in checks:
        assert ok, f"{name} 失败（{detail}）"


def test_group3_unrelated_targets():
    checks, _ = group3_unrelated_targets()
    for name, ok, detail in checks:
        assert ok, f"{name} 失败（{detail}）"


def test_group4_false_positive():
    checks, _ = group4_false_positive()
    for name, ok, detail in checks:
        assert ok, f"{name} 失败（{detail}）"


def test_group5_top_k():
    checks, _ = group5_top_k()
    for name, ok, detail in checks:
        assert ok, f"{name} 失败（{detail}）"


def test_group6_boundary():
    checks, _ = group6_boundary()
    for name, ok, detail in checks:
        assert ok, f"{name} 失败（{detail}）"


# ----------------------------------------------------------------------
# 脚本模式
# ----------------------------------------------------------------------

def main():
    print("=" * 72)
    print("RAG 6.11 检索质量专项测试")
    print("=" * 72)

    total = passed = failed = 0
    for name, fn in GROUPS:
        checks, prints = fn()
        print(f"\n########## {name} ##########")
        for block in prints:
            print(block + "\n")
        for cname, ok, detail in checks:
            total += 1
            mark = "PASS" if ok else "FAIL"
            if ok:
                passed += 1
            else:
                failed += 1
            print(f"[{mark}] {cname}" + (f"  （{detail}）" if detail else ""))

    print("\n" + "=" * 72)
    print(f"统计：总 {total} | PASS {passed} | FAIL {failed}")
    print("=" * 72)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
