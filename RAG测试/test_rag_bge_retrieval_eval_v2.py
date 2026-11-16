# -*- coding: utf-8 -*-
"""真实 BGE 离线检索指标评估（阶段 7B-4B）。

对 `RAG测试/rag_offline_eval_dataset_v2.json` 的 60 条查询，用**生产检索链路**
（TextLoader → TextSplitter → Embedding.encode → VectorStore 余弦 → Retriever Top-K
→ KnowledgeBase 默认过滤）做离线检索评估，输出逐查询结果、分组指标、无答案误召回
与阈值扫描权衡。

相关性映射规则（三级标签**不**压成等价二值）：
- 二值排名指标（Recall@K / MRR / nDCG）中「relevant」= `expected_evidence` 锚点解析出的
  (source, chunk_id)。完全相关查询的 R=全部必要证据；部分相关查询的 R=其「已支持事实」
  证据 chunk（召回它只表示找到部分证据，不代表完整回答）。三级标签通过分组指标保留：
  完全相关 / 部分相关 / 多chunk联合 各自独立报告。
- 无答案（不相关）查询 expected_evidence 为空，单独报告误召回与 Top-1 分数分布。

阈值扫描（仅分析，不选定/应用生产阈值）：τ ∈ [0.00, 1.00] 步长 0.05，复现生产
「先 Top-K 截断再 score>=τ 过滤」语义：
  - 有答案误拒答率(τ) = 相关证据 chunk 最高分 < τ 的有答案查询占比（答案证据被过滤）；
  - 无答案误召回率(τ) = Top-1 分数 >= τ 的无答案查询占比（仍返回结果）。

安全约束：仅从本地缓存加载模型；不 import openai/requests/main；不读 .env；不调用
真实 LLM/视觉/付费 API；不修改任何文件、不修改标签与评分规则。

运行方式（显式授权）：
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 RUN_BGE_EVAL=1
    python -B -m pytest RAG测试/test_rag_bge_retrieval_eval_v2.py -v -s \
        --basetemp=<唯一新目录> -o tmp_path_retention_policy=all -p no:cacheprovider
"""

import json
import math
import os
import statistics
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.knowledge_base import KnowledgeBase  # noqa: E402
from rag.loader import TextLoader  # noqa: E402
from rag.splitter import TextSplitter  # noqa: E402

MODEL_ID = "BAAI/bge-small-zh-v1.5"
DATASET_PATH = PROJECT_ROOT / "RAG测试" / "rag_offline_eval_dataset_v2.json"

RUN_BGE_EVAL = os.environ.get("RUN_BGE_EVAL") == "1"
THRESHOLDS = [round(i * 0.05, 2) for i in range(0, 21)]  # 0.00 .. 1.00


def _check_offline_env():
    hf = os.environ.get("HF_HUB_OFFLINE")
    tr = os.environ.get("TRANSFORMERS_OFFLINE")
    if hf != "1" or tr != "1":
        pytest.fail(
            "离线环境变量未正确设置：HF_HUB_OFFLINE=%r, TRANSFORMERS_OFFLINE=%r "
            "（必须均为 '1'）。停止。" % (hf, tr), pytrace=False)


def _huggingface_cache_root():
    hf_home = os.environ.get("HF_HOME")
    return Path(hf_home) / "hub" if hf_home else Path.home() / ".cache" / "huggingface" / "hub"


def _check_model_cache():
    model_dir = _huggingface_cache_root() / ("models--" + MODEL_ID.replace("/", "--"))
    refs_main = model_dir / "refs" / "main"
    if not refs_main.exists():
        pytest.fail("模型缓存缺失：%s。停止。" % refs_main, pytrace=False)
    try:
        rev = refs_main.read_text(encoding="utf-8").strip()
    except OSError as exc:
        pytest.fail("无法读取缓存 refs/main：%s。停止。" % exc, pytrace=False)
    if not (model_dir / "snapshots" / rev / "model.safetensors").exists():
        pytest.fail("模型权重缺失。停止（未联网兜底）。", pytrace=False)
    return rev, model_dir


def enumerate_chunks(doc_dir, filenames):
    chunks = []
    splitter = TextSplitter()
    for name in sorted(filenames):
        path = Path(doc_dir) / name
        for document in TextLoader(str(path)).load():
            for chunk in splitter.split(document):
                chunks.append((name, chunk.metadata.get("chunk_id"), chunk.page_content))
    return chunks


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _median(vals):
    return statistics.median(vals) if vals else None


def recall_at_k(ranked, relevant, k):
    if not relevant:
        return None
    return len(set(ranked[:k]) & relevant) / len(relevant)


def mrr_at_k(ranked, relevant, k):
    if not relevant:
        return None
    for i, item in enumerate(ranked[:k], start=1):
        if item in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked, relevant, k):
    if not relevant:
        return None
    dcg = sum(1.0 / math.log2(i + 1) for i, item in enumerate(ranked[:k], start=1) if item in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(k, len(relevant)) + 1))
    return dcg / ideal if ideal else 0.0


@pytest.mark.skipif(not RUN_BGE_EVAL, reason="需显式设置 RUN_BGE_EVAL=1 才加载真实模型")
def test_rag_bge_retrieval_eval_v2(tmp_path):
    # ---- 离线 + 模型 ----
    _check_offline_env()
    rev, model_dir = _check_model_cache()
    from rag.embedding import Embedding
    try:
        embedding = Embedding()
        dim = len(list(embedding.encode("离线检索评估探针")))
    except Exception as exc:
        pytest.fail("真实 BGE 模型加载失败（缓存不可用）：%r。停止。" % exc, pytrace=False)

    # ---- 数据集 ----
    data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    documents = data["documents"]
    queries = data["queries"]
    id_to_filename = {d["doc_id"]: d["filename"] for d in documents}

    # ---- 真实分块 + 锚点解析（复用 7B-4A 已校验通过的数据）----
    for d in documents:
        (tmp_path / d["filename"]).write_text(d["content"], encoding="utf-8")
    chunks = enumerate_chunks(str(tmp_path), list(id_to_filename.values()))

    def resolve(doc_id, anchor):
        did = id_to_filename.get(doc_id)
        hits = [(s, c) for (s, c, t) in chunks if anchor in t and s == did]
        if not hits:
            pytest.fail("锚点未命中：%s / %r" % (doc_id, anchor), pytrace=False)
        if len(hits) > 1:
            pytest.fail("锚点歧义（多 chunk）：%s / %r -> %s" % (doc_id, anchor, hits), pytrace=False)
        return hits[0]

    # ---- 生产检索链路 ----
    kb = KnowledgeBase(str(tmp_path), embedding_factory=lambda: embedding)

    answerable_rows, noanswer_rows = [], []
    for q in queries:
        grade = q["relevance_grade"]
        category = q["category"]
        ev = q.get("expected_evidence") or []
        relevant = set()
        for e in ev:
            relevant.add(resolve(e["doc_id"], e["anchor"]))

        result = kb.search(q["text"], top_k=10)
        assert result["ok"] is True, "检索失败：%r %r" % (q["text"], result)
        ranked = [(r["source"], r["chunk_id"]) for r in result["results"]]
        scores = [float(r["score"]) for r in result["results"]]
        top_items = [(r["source"], r["chunk_id"], float(r["score"])) for r in result["results"]]

        row = {
            "query_id": q["query_id"], "text": q["text"], "category": category,
            "grade": grade, "relevant": relevant, "ranked": ranked, "scores": scores,
            "top_items": top_items, "subtype": q.get("no_answer_subtype"),
        }
        if relevant:
            answerable_rows.append(row)
        else:
            noanswer_rows.append(row)

    # ---- 逐查询结果打印 ----
    line = "=" * 78
    print(line)
    print("阶段 7B-4B 真实 BGE 离线检索指标评估")
    print(line)
    print("[离线证据]")
    print("  HF_HUB_OFFLINE=%r TRANSFORMERS_OFFLINE=%r RUN_BGE_EVAL=%r" % (
        os.environ.get("HF_HUB_OFFLINE"), os.environ.get("TRANSFORMERS_OFFLINE"),
        os.environ.get("RUN_BGE_EVAL")))
    print("  缓存 revision=%s  目录=%s" % (rev, model_dir))
    print("  模型加载=OK（维度=%d）  文档=%d  查询=%d  总chunk=%d" % (
        dim, len(documents), len(queries), len(chunks)))

    print("\n[逐查询结果（有答案 %d 条）]" % len(answerable_rows))
    print("  query_id  grade/category                          R(top3命中)  R@1  R@3  MRR3  nD3")
    for r in answerable_rows:
        top3 = ", ".join("%s#%s:%.3f" % (s, c, sc) for s, c, sc in r["top_items"][:3])
        rel = "{%s}" % ",".join("%s#%s" % (s, c) for s, c in sorted(r["relevant"]))
        print("  %-9s %-38s %-16s %.2f %.2f %.2f %.2f" % (
            r["query_id"], "[" + r["grade"] + "/" + r["category"] + "]", rel,
            recall_at_k(r["ranked"], r["relevant"], 1),
            recall_at_k(r["ranked"], r["relevant"], 3),
            mrr_at_k(r["ranked"], r["relevant"], 3),
            ndcg_at_k(r["ranked"], r["relevant"], 3)))
        print("         top3=%s" % top3)

    print("\n[逐查询结果（无答案 %d 条）]" % len(noanswer_rows))
    print("  query_id  subtype                 count  top1_score  top3")
    for r in noanswer_rows:
        top1 = r["top_items"][0] if r["top_items"] else None
        top3 = ", ".join("%s#%s:%.3f" % (s, c, sc) for s, c, sc in r["top_items"][:3])
        print("  %-9s %-22s %-5d %-10.4f %s" % (
            r["query_id"], "[" + (r["subtype"] or "") + "]", len(r["ranked"]),
            (r["scores"][0] if r["scores"] else float("nan")), top3))

    # ---- 分组指标 ----
    def group_metrics(rows):
        n = len(rows)
        if n == 0:
            return None
        return {
            "n": n,
            "R@1": _mean([recall_at_k(r["ranked"], r["relevant"], 1) for r in rows]),
            "R@3": _mean([recall_at_k(r["ranked"], r["relevant"], 3) for r in rows]),
            "R@10": _mean([recall_at_k(r["ranked"], r["relevant"], 10) for r in rows]),
            "MRR@3": _mean([mrr_at_k(r["ranked"], r["relevant"], 3) for r in rows]),
            "nDCG@3": _mean([ndcg_at_k(r["ranked"], r["relevant"], 3) for r in rows]),
        }

    all_ans = group_metrics(answerable_rows)
    full = group_metrics([r for r in answerable_rows if r["grade"] == "完全相关"])
    partial = group_metrics([r for r in answerable_rows if r["grade"] == "部分相关"])
    multi = group_metrics([r for r in answerable_rows if r["category"] == "需要多个 chunk 联合回答"])

    print("\n[分组指标（宏观平均）]")
    for name, g in [("有答案合计(30)", all_ans), ("完全相关(23)", full),
                    ("部分相关(7)", partial), ("多chunk联合(7)", multi)]:
        if g:
            print("  %-16s n=%-2d R@1=%.4f R@3=%.4f R@10=%.4f MRR@3=%.4f nDCG@3=%.4f" % (
                name, g["n"], g["R@1"], g["R@3"], g["R@10"], g["MRR@3"], g["nDCG@3"]))
    print("  注：部分相关组的 R@ 仅度量「已支持事实」证据 chunk 被召回，不代表完整回答（缺失事实不在语料）。")

    # ---- 多 chunk 必要证据完整覆盖率 ----
    multi_rows = [r for r in answerable_rows if r["category"] == "需要多个 chunk 联合回答"]
    if multi_rows:
        full_cov3 = sum(1 for r in multi_rows if recall_at_k(r["ranked"], r["relevant"], 3) == 1.0)
        full_cov10 = sum(1 for r in multi_rows if recall_at_k(r["ranked"], r["relevant"], 10) == 1.0)
        print("\n[多chunk 必要证据完整覆盖率]")
        print("  完整召回@3 = %d/%d  完整召回@10 = %d/%d" % (
            full_cov3, len(multi_rows), full_cov10, len(multi_rows)))

    # ---- 无答案误召回 + 子类型分布 ----
    print("\n[无答案误召回（默认 score>0 过滤，top_k=10）]")
    mis = sum(1 for r in noanswer_rows if len(r["ranked"]) > 0)
    print("  总体误召回 = %d / %d" % (mis, len(noanswer_rows)))
    subtypes = {}
    for r in noanswer_rows:
        subtypes.setdefault(r["subtype"], []).append(r)
    for st, rows in sorted(subtypes.items()):
        st_mis = sum(1 for r in rows if len(r["ranked"]) > 0)
        top1 = [r["scores"][0] for r in rows if r["scores"]]
        print("    [%s] n=%d 误召回=%d/%d  Top1分数 min=%.4f med=%.4f mean=%.4f max=%.4f" % (
            st, len(rows), st_mis, len(rows),
            min(top1), _median(top1), _mean(top1), max(top1)))
        print("        Top1明细=%s" % [round(v, 4) for v in top1])

    # ---- 分数分布（诊断）----
    ans_top1_rel = []  # 有答案：相关证据 chunk 的最高分
    for r in answerable_rows:
        rel_scores = [sc for (s, c, sc) in r["top_items"] if (s, c) in r["relevant"]]
        ans_top1_rel.append(max(rel_scores) if rel_scores else 0.0)
    na_top1 = [r["scores"][0] for r in noanswer_rows if r["scores"]]
    print("\n[分数分布（诊断，非置信度）]")
    print("  有答案 相关证据最高分：min=%.4f med=%.4f mean=%.4f max=%.4f" % (
        min(ans_top1_rel), _median(ans_top1_rel), _mean(ans_top1_rel), max(ans_top1_rel)))
    print("  无答案 Top-1 分数：min=%.4f med=%.4f mean=%.4f max=%.4f" % (
        min(na_top1), _median(na_top1), _mean(na_top1), max(na_top1)))

    # ---- 阈值扫描 ----
    print("\n[阈值扫描（仅分析，不选定生产阈值）]")
    print("  %-8s %-14s %-14s" % ("阈值τ", "有答案误拒答率", "无答案误召回率"))
    for tau in THRESHOLDS:
        refused = sum(1 for r in answerable_rows if max(
            [sc for (s, c, sc) in r["top_items"] if (s, c) in r["relevant"]] or [0.0]) < tau)
        recalled = sum(1 for r in noanswer_rows if (r["scores"][0] if r["scores"] else 0.0) >= tau)
        print("  %-8.2f %-14s %-14s" % (
            tau, "%d/%d=%.3f" % (refused, len(answerable_rows), refused / len(answerable_rows)),
            "%d/%d=%.3f" % (recalled, len(noanswer_rows), recalled / len(noanswer_rows))))

    print("\n[样本规模与适用范围]")
    print("  60 条（有答案30 + 无答案30）/ 12 文档 / 14 chunk；自编客服语料，不代表正式数学题知识库。")
    print("  本阶段只做评估，不改生产行为、不选定阈值。")
    print(line)
