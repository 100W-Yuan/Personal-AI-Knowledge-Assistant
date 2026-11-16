# -*- coding: utf-8 -*-
"""RAG 6.9：完整 RAG Pipeline 串联验证（数据链路正确性）。

链路：用户问题 → Embedding → VectorStore → Retriever → Top-K → ContextBuilder → Context
本阶段不接入 Agent、不调用 LLM；使用真实本地模型 BAAI/bge-small-zh-v1.5
（离线缓存），虚构数据，不读正式长期记忆。

验证原则：只验证数据链路正确性（相关内容进入 Top-K、数量、降序、Context 构建），
不设置脆弱的固定分数阈值（真实模型相似度不写死小数）。

运行方式（均支持）：
  python -m RAG测试.test_rag_pipeline_full          （脚本模式 + 退出码）
  python -m pytest -q "RAG测试/test_rag_pipeline_full.py"（测试模式）

建议设置 HF_HUB_OFFLINE=1 以强制使用本地缓存（不产生任何云端请求）。
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.context_builder import ContextBuilder   # noqa: E402
from rag.document import Document                 # noqa: E402
from rag.embedding import Embedding               # noqa: E402
from rag.retriever import Retriever               # noqa: E402
from rag.vector_store import VectorStore          # noqa: E402

QUERY = "我的 Agent 项目目前实现了什么？"

DOCUMENTS = [
    Document(
        "我的 AI Agent 项目已经实现了 Tool Calling、Task Loop 和 Long Term Memory。",
        {"source": "A.txt"},
    ),
    Document(
        "我的 AI Agent 项目目前正在学习 RAG，包括文档加载、文本切分和向量检索。",
        {"source": "B.txt"},
    ),
    Document(
        "我每周进行三次力量训练。",
        {"source": "C.txt"},
    ),
    Document(
        "今天晚上准备吃面条。",
        {"source": "D.txt"},
    ),
]

TOP_K = 2


def run_pipeline():
    """执行完整链路，返回所有中间结果与验证明细。"""
    checks = []

    def record(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    # ---- Step 1：真实 Embedding 编码 4 个 Document ----
    embedding = Embedding()
    doc_vectors = [
        embedding.encode(doc.page_content) for doc in DOCUMENTS
    ]
    record("[链路] 完整执行未抛出异常", True)

    # ---- Step 2：Document + vector 放入 VectorStore ----
    store = VectorStore()
    for doc, vec in zip(DOCUMENTS, doc_vectors):
        store.add(doc, vec)
    record("[步骤2] VectorStore 保存 4 个文档",
           len(store.items) == 4, f"实际={len(store.items)}")

    # ---- Step 3：对用户问题进行真实 Embedding（不得直接传字符串） ----
    query_vector = embedding.encode(QUERY)
    q_dim = len(query_vector)
    record("[步骤3] 用户问题经 Embedding 得到向量（非字符串）",
           not isinstance(query_vector, str) and q_dim > 1,
           f"维度={q_dim}")

    retriever = Retriever(store)
    results = retriever.retrieve(query_vector, top_k=TOP_K)

    # ---- Step 4/5：Retriever → ContextBuilder → Context ----
    context = ContextBuilder().build(results)

    # ---- 验证 2：问题经过 Embedding（维度与文档向量一致） ----
    record("[验证2] 查询向量维度与文档向量一致",
           q_dim == len(doc_vectors[0]), f"{q_dim} vs {len(doc_vectors[0])}")

    # ---- 验证 3/4：Retriever 返回结果；top_k=2 最多 2 个 ----
    record("[验证3] Retriever 返回结果非空", len(results) > 0,
           f"返回 {len(results)} 条")
    record("[验证4] top_k=2 → 最多返回 2 条", len(results) <= TOP_K,
           f"实际 {len(results)} 条")

    # ---- 验证 5/6：字段 document / score ----
    fields_ok = all(("document" in r and "score" in r) for r in results)
    record("[验证5/6] 每条结果包含 document 和 score", fields_ok)

    # ---- 验证 7：score 降序 ----
    scores = [r["score"] for r in results]
    record("[验证7] score 按降序排列",
           scores == sorted(scores, reverse=True),
           f"顺序={[round(float(s), 6) for s in scores]}")

    # ---- 验证 8/9：相关内容进入 Top-K；健身/晚餐不同时占满 ----
    top_sources = [r["document"].metadata.get("source") for r in results]
    related_in_top = any(s in ("A.txt", "B.txt") for s in top_sources)
    record("[验证8] Agent 项目相关内容（A/B）进入 Top-K",
           related_in_top, f"Top-K 来源={top_sources}")
    record("[验证9] 无关内容（C 健身 / D 晚餐）未同时占据 Top-K",
           not ({"C.txt", "D.txt"} <= set(top_sources)),
           f"Top-K 来源={top_sources}")

    # ---- 验证 10：ContextBuilder 正常接收 ----
    record("[验证10] ContextBuilder 成功接收结果（返回字符串）",
           isinstance(context, str), f"类型={type(context).__name__}")

    # ---- 验证 11：Context 包含 Retriever 返回的 Chunk ----
    chunks_ok = all(r["document"].page_content in context for r in results)
    record("[验证11] 最终 Context 包含全部返回 Chunk 内容", chunks_ok,
           f"覆盖 {len(results)} 条")

    # ---- 验证 12：Context 保留对应 score ----
    scores_in_ctx = all(f"相似度：{r['score']}" in context for r in results)
    record("[验证12] 最终 Context 保留对应 score", scores_in_ctx)

    # ---- 验证 13：Context 非空 ----
    record("[验证13] 最终 Context 非空", len(context) > 0,
           f"长度={len(context)}")

    return {
        "query": QUERY,
        "query_dim": q_dim,
        "results": results,
        "scores": [float(s) for s in scores],
        "top_sources": top_sources,
        "context": context,
        "checks": checks,
        "all_passed": all(ok for _, ok, _ in checks),
    }


# ----------------------------------------------------------------------
# 测试模式（pytest）：逐项断言全部检查结果（不弱化任何一项）
# ----------------------------------------------------------------------

def test_full_rag_pipeline():
    r = run_pipeline()
    for name, ok, detail in r["checks"]:
        assert ok, f"{name} 失败" + (f"（{detail}）" if detail else "")


# ----------------------------------------------------------------------
# 脚本模式（直接运行：打印逐步过程并返回退出码）
# ----------------------------------------------------------------------

def main():
    r = run_pipeline()

    print("=" * 64)
    print("RAG 6.9 完整 Pipeline 串联验证")
    print("=" * 64)
    print(f"Query：{r['query']}")
    print(f"[Embedding] 查询向量维度 = {r['query_dim']}")
    print(f"[Retriever] Top-K（top_k={TOP_K}）：")
    for i, res in enumerate(r["results"], start=1):
        print(f"  {i}. [{res['document'].metadata.get('source')}] "
              f"score={float(res['score']):.6f} | "
              f"{res['document'].page_content}")
    print(f"[ContextBuilder] Top-K 来源顺序 = {r['top_sources']}")
    print("---- 最终 Context 预览 ----")
    print(r["context"].strip())
    print("-" * 64)
    for name, ok, detail in r["checks"]:
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {name}" + (f"  （{detail}）" if detail else ""))
    print("=" * 64)
    print("全部验证通过" if r["all_passed"] else "存在失败项，请查看上方 FAIL 行")
    return 0 if r["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
