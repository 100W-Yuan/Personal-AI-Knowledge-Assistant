#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生产语料 RAG 检索评估（N2-B2）。

定位
----
只读评估脚本，服务于 `RAG测试/rag_production_eval_dataset_v1.json`。
语料直接取自 `rag_knowledge/`，**不使用** JSON 内嵌文档副本。

两种模式
--------
1. 默认（校验模式，不加载任何模型）：
       py -3.12 -B "RAG测试/rag_production_eval.py"
   校验数据集结构、语料哈希、切分参数、锚点唯一性、类别计数一致性。
   有错则退出码 1，全通过则退出码 0。

2. 检索模式（需显式授权 + 双重开关）：
       export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 RUN_PRODUCTION_RAG_EVAL=1
       py -3.12 -B "RAG测试/rag_production_eval.py" --retrieval
   加载本地 BGE 模型，输出 Hit@1/3/10、MRR@3、Precision@3、
   无答案误召回率、证据完整率、nDCG@3（辅助）。

安全约束
--------
- 模块导入期不加载 Embedding；`rag.embedding` 只在 `run_retrieval()` 内部导入。
- 检索模式下若 `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` 不为 "1"，直接退出（码 2），
  **不做任何联网兜底**。
- 语料先复制到 `tempfile.TemporaryDirectory()`，再在临时目录建索引，
  保证对 `rag_knowledge/` 零写入；不连接任何数据库；不 import main / agent / openai。
- 不修改任何生产 RAG 代码，只消费 `TextLoader` / `TextSplitter` / `KnowledgeBase`。

指标口径（重要）
----------------
- 本数据集 A 类均为单证据，故 `Hit@k` 即 `Recall@k` 在该场景下的退化形式，
  两者数值相同；脚本统一输出 Hit@k，避免把"命中率"误读成"召回率"。
- `Precision@3 = |Top3 ∩ R| / 3`，分母固定为 3（即使返回不足 3 条）。
- `证据完整率@3`：Top3 中存在"单块含完整证据单元"的 chunk 即记 1。
  跨块题目的 `evidence_complete_chunk_ids` 为空数组，该类查询恒记 0——
  这是语料切分的真实上限，不是脚本缺陷。
- `nDCG@3` 为辅助指标：单证据二值增益下它与 `MRR@3` 同序，信息重复，只作对照。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 只导入轻量组件：两者都不依赖 torch / sentence-transformers。
from rag.loader import TextLoader  # noqa: E402
from rag.splitter import TextSplitter  # noqa: E402

DATASET_PATH = PROJECT_ROOT / "RAG测试" / "rag_production_eval_dataset_v1.json"
CORPUS_DIR = PROJECT_ROOT / "rag_knowledge"
MODEL_ID = "BAAI/bge-small-zh-v1.5"
RUN_ENV = "RUN_PRODUCTION_RAG_EVAL"

CATEGORIES = (
    "有答案-事实型",
    "主题相似但缺少关键答案",
    "题面存在但答案键缺失",
    "完全无关",
)

GRADE_BY_CAT = {
    "有答案-事实型": "完全相关",
    "主题相似但缺少关键答案": "不相关",
    "题面存在但答案键缺失": "不相关",
    "完全无关": "不相关",
}

SUBTYPE_BY_CAT = {
    "主题相似但缺少关键答案": "主题相似但缺少答案",
    "题面存在但答案键缺失": "题面存在但答案键缺失",
    "完全无关": "完全无关",
}

REQUIRED_QUERY_FIELDS = (
    "query_id",
    "text",
    "category",
    "query_style",
    "relevance_grade",
    "expected_evidence",
    "question_chunk_span",
    "evidence_complete_chunk_ids",
    "supported_facts",
    "missing_facts",
    "no_answer_subtype",
    "expected_behavior",
    "rationale",
)


# ======================================================================
# 基础工具
# ======================================================================


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def load_dataset(path: Path = DATASET_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def enumerate_chunks(doc_dir: Path, filenames) -> list:
    """用生产 TextLoader + TextSplitter 实际分块。

    返回 [(filename, chunk_id, text), ...]；顺序与生产索引一致
    （文件名排序 → 文件内 chunk 序）。
    """
    chunks = []
    splitter = TextSplitter()
    for name in sorted(filenames):
        for document in TextLoader(str(Path(doc_dir) / name)).load():
            for chunk in splitter.split(document):
                chunks.append(
                    (name, chunk.metadata.get("chunk_id"), chunk.page_content)
                )
    return chunks


def resolve_anchor(anchor: str, chunks, filename: str) -> list:
    """返回能在本文件内**完整包含**该锚点的 chunk_id 列表（去重保序）。"""
    hits = []
    for (name, chunk_id, text) in chunks:
        if name == filename and anchor in text and chunk_id not in hits:
            hits.append(chunk_id)
    return hits


# ======================================================================
# 离线保护（全部在加载模型之前执行）
# ======================================================================


def check_offline_env() -> str:
    """返回错误描述；空字符串表示通过。不修改任何环境变量。"""
    problems = []
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        problems.append(
            "HF_HUB_OFFLINE=%r（必须为 '1'）" % os.environ.get("HF_HUB_OFFLINE")
        )
    if os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        problems.append(
            "TRANSFORMERS_OFFLINE=%r（必须为 '1'）"
            % os.environ.get("TRANSFORMERS_OFFLINE")
        )
    return "；".join(problems)


def check_model_cache() -> tuple:
    """按 eval_v2 的同一规则检查本地缓存；缺失即返回错误，不联网兜底。"""
    hf_home = os.environ.get("HF_HOME")
    root = Path(hf_home) / "hub" if hf_home else Path.home() / ".cache" / "huggingface" / "hub"
    model_dir = root / ("models--" + MODEL_ID.replace("/", "--"))
    refs_main = model_dir / "refs" / "main"
    if not refs_main.exists():
        return "", "模型缓存缺失：%s" % refs_main
    try:
        revision = refs_main.read_text(encoding="utf-8").strip()
    except OSError as exc:
        return "", "无法读取缓存 refs/main：%s" % exc
    weights = model_dir / "snapshots" / revision / "model.safetensors"
    if not weights.exists():
        return "", "模型权重缺失：%s" % weights
    return revision, ""


# ======================================================================
# 静态校验（不加载模型）
# ======================================================================


def validate_dataset(data: dict) -> tuple:
    """返回 (errors, chunks, stats)。errors 为空表示通过。"""
    errors = []
    meta = data.get("meta") or {}
    queries = data.get("queries") or []

    # ---- 1. 语料文件：存在性 + 哈希 + 尺寸 ----
    declared_files = meta.get("files") or []
    if not declared_files:
        errors.append("meta.files 为空：数据集必须声明语料文件与哈希")
    filenames = []
    for item in declared_files:
        name = item.get("filename")
        path = CORPUS_DIR / str(name)
        filenames.append(name)
        if not path.is_file():
            errors.append("语料缺失：%s" % path)
            continue
        actual_hash = sha256_file(path)
        if actual_hash != item.get("sha256"):
            errors.append(
                "语料哈希不一致：%s 实际=%s 声明=%s（语料已变更，需重新标注）"
                % (name, actual_hash, item.get("sha256"))
            )
        raw = path.read_bytes()
        # 字符数/行数按**文本模式**统计：TextLoader 以通用换行读入，
        # 磁盘上的 CRLF 会被折叠为 \n，切分偏移也基于折叠后的文本。
        text = path.read_text(encoding="utf-8")
        if item.get("bytes") is not None and len(raw) != item.get("bytes"):
            errors.append("语料字节数不一致：%s 实际=%d" % (name, len(raw)))
        if item.get("chars") is not None and len(text) != item.get("chars"):
            errors.append(
                "语料字符数不一致（文本模式口径）：%s 实际=%d" % (name, len(text))
            )
        if item.get("lines") is not None and len(text.split("\n")) != item.get("lines"):
            errors.append("语料行数不一致：%s 实际=%d" % (name, len(text.split("\n"))))

    if errors:
        return errors, [], {}

    # ---- 2. 切分参数与 chunk 数 ----
    chunks = enumerate_chunks(CORPUS_DIR, filenames)
    chunking = meta.get("chunking") or {}
    splitter = TextSplitter()
    if chunking.get("chunk_size") != splitter.chunk_size:
        errors.append(
            "chunk_size 与生产不一致：声明=%r 生产=%r"
            % (chunking.get("chunk_size"), splitter.chunk_size)
        )
    if chunking.get("overlap") != splitter.overlap:
        errors.append(
            "overlap 与生产不一致：声明=%r 生产=%r"
            % (chunking.get("overlap"), splitter.overlap)
        )
    if chunking.get("chunk_count") != len(chunks):
        errors.append(
            "chunk_count 与生产切分不一致：声明=%r 实际=%d"
            % (chunking.get("chunk_count"), len(chunks))
        )

    # ---- 3. 查询结构 ----
    query_ids = [q.get("query_id") for q in queries]
    duplicated_ids = sorted({qid for qid in query_ids if query_ids.count(qid) > 1})
    if duplicated_ids:
        errors.append("query_id 重复：%s" % duplicated_ids)
    texts = [q.get("text") for q in queries]
    duplicated_texts = sorted({t for t in texts if texts.count(t) > 1})
    if duplicated_texts:
        errors.append("查询文本重复：%s" % duplicated_texts)

    for q in queries:
        qid = q.get("query_id", "?")
        for field in REQUIRED_QUERY_FIELDS:
            if field not in q:
                errors.append("%s 缺字段 %s" % (qid, field))
        if q.get("category") not in CATEGORIES:
            errors.append("%s 非法 category=%r" % (qid, q.get("category")))
            continue
        if q.get("relevance_grade") != GRADE_BY_CAT[q["category"]]:
            errors.append(
                "%s 类别=%s 但 grade=%r（应为 %r）"
                % (qid, q["category"], q.get("relevance_grade"), GRADE_BY_CAT[q["category"]])
            )
        expected_subtype = SUBTYPE_BY_CAT.get(q["category"])
        if q.get("no_answer_subtype") != expected_subtype:
            errors.append(
                "%s 类别=%s 但 no_answer_subtype=%r（应为 %r）"
                % (qid, q["category"], q.get("no_answer_subtype"), expected_subtype)
            )

    # ---- 4. 类别计数必须与 meta 完全一致 ----
    actual_counts = Counter(q.get("category") for q in queries)
    declared_counts = meta.get("count_targets") or {}
    for category in CATEGORIES:
        actual = actual_counts.get(category, 0)
        declared = declared_counts.get(category)
        if declared != actual:
            errors.append(
                "类别计数不一致：%s 实际=%d 声明=%r" % (category, actual, declared)
            )
    unknown_categories = sorted(set(declared_counts) - set(CATEGORIES))
    if unknown_categories:
        errors.append("meta.count_targets 含未知类别：%s" % unknown_categories)
    if meta.get("total_queries") != len(queries):
        errors.append(
            "total_queries 不一致：声明=%r 实际=%d"
            % (meta.get("total_queries"), len(queries))
        )

    # ---- 5. 证据与锚点 ----
    for q in queries:
        qid = q.get("query_id", "?")
        category = q.get("category")
        evidence = q.get("expected_evidence") or []
        grade = q.get("relevance_grade")

        if grade == "不相关":
            if evidence:
                errors.append("%s 不相关却给出了 expected_evidence" % qid)
            if q.get("supported_facts") not in (None, ""):
                errors.append("%s 不相关却填了 supported_facts" % qid)
            if q.get("missing_facts") in (None, ""):
                errors.append("%s 不相关但 missing_facts 为空（必须写明缺什么）" % qid)
            continue

        if not evidence:
            errors.append("%s 有答案但 expected_evidence 为空" % qid)
            continue
        if q.get("supported_facts") in (None, ""):
            errors.append("%s 有答案但 supported_facts 为空" % qid)

        anchors = set()
        for item in evidence:
            name = item.get("filename")
            anchor = item.get("anchor")
            declared_chunk = item.get("chunk_id")
            if name not in filenames:
                errors.append("%s 证据 filename=%r 不在语料清单内" % (qid, name))
                continue
            if not isinstance(anchor, str) or not anchor:
                errors.append("%s 锚点为空" % qid)
                continue
            if anchor in anchors:
                errors.append("%s 锚点重复：%r" % (qid, anchor))
                continue
            anchors.add(anchor)

            hits = resolve_anchor(anchor, chunks, name)
            if not hits:
                errors.append("%s 锚点未命中任何 chunk：%r" % (qid, anchor))
            elif len(hits) > 1:
                errors.append(
                    "%s 锚点歧义（overlap 跨 chunk）：%r -> %s"
                    % (qid, anchor, hits)
                )
            elif declared_chunk != hits[0]:
                errors.append(
                    "%s 锚点 %r 实际落在 chunk %r，数据集声明 %r"
                    % (qid, anchor, hits[0], declared_chunk)
                )

        span = q.get("question_chunk_span") or []
        complete = q.get("evidence_complete_chunk_ids") or []
        if not span:
            errors.append("%s 有答案但 question_chunk_span 为空" % qid)
        for chunk_id in complete:
            if chunk_id not in span:
                errors.append(
                    "%s evidence_complete_chunk_ids=%r 不在 span=%r 内"
                    % (qid, chunk_id, span)
                )
        # 单块题目必须给出完整证据块；跨块题目必须留空（不伪造 chunk_id）。
        if len(span) == 1 and complete != list(span):
            errors.append(
                "%s 单块题目（span=%r）的 complete 应为 %r，实际 %r"
                % (qid, span, list(span), complete)
            )
        if len(span) > 1 and complete:
            errors.append(
                "%s 跨块题目（span=%r）不得声明完整证据块，实际 %r"
                % (qid, span, complete)
            )

    # ---- 6. 类别专属规则 ----
    for q in queries:
        qid = q.get("query_id", "?")
        if q.get("category") == "题面存在但答案键缺失":
            if not (q.get("topic_chunk_ids") or []):
                errors.append("%s 应给出 topic_chunk_ids（题面所在块）" % qid)
            if "答案" not in str(q.get("missing_facts") or ""):
                errors.append("%s missing_facts 需明确写出“答案键缺失”" % qid)
        if q.get("category") == "完全无关" and (q.get("topic_chunk_ids") or []):
            errors.append("%s 完全无关却给出了 topic_chunk_ids" % qid)

    stats = {
        "files": len(declared_files),
        "chunks": len(chunks),
        "queries": len(queries),
        "counts": dict(actual_counts),
        "answerable": sum(1 for q in queries if q.get("relevance_grade") == "完全相关"),
        "no_answer": sum(1 for q in queries if q.get("relevance_grade") == "不相关"),
        "complete_evidence": sum(
            1 for q in queries if q.get("evidence_complete_chunk_ids")
        ),
    }
    return errors, chunks, stats


# ======================================================================
# 指标
# ======================================================================


def hit_at_k(ranked, relevant, k) -> float:
    return 1.0 if set(ranked[:k]) & relevant else 0.0


def mrr_at_k(ranked, relevant, k) -> float:
    for index, item in enumerate(ranked[:k], start=1):
        if item in relevant:
            return 1.0 / index
    return 0.0


def precision_at_k(ranked, relevant, k) -> float:
    return len(set(ranked[:k]) & relevant) / float(k)


def ndcg_at_k(ranked, relevant, k) -> float:
    dcg = sum(
        1.0 / math.log2(index + 1)
        for index, item in enumerate(ranked[:k], start=1)
        if item in relevant
    )
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(k, len(relevant)) + 1))
    return dcg / ideal if ideal else 0.0


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


# ======================================================================
# 检索模式（唯一会加载模型的分支）
# ======================================================================


def run_retrieval(data: dict) -> int:
    problem = check_offline_env()
    if problem:
        print("[拒绝执行] 离线环境保护未满足：%s" % problem)
        print("           请先导出 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1，本脚本不做联网兜底。")
        return 2

    revision, problem = check_model_cache()
    if problem:
        print("[拒绝执行] %s" % problem)
        return 2

    if os.environ.get(RUN_ENV) != "1":
        print("[未启用] 真实模型评估需要显式设置 %s=1。" % RUN_ENV)
        print("         本次只运行静态校验，不加载模型。")
        return 3

    meta = data["meta"]
    queries = data["queries"]
    filenames = [item["filename"] for item in meta["files"]]

    print("=" * 78)
    print("生产语料 RAG 检索评估（N2-B2）")
    print("=" * 78)
    print("[离线证据] HF_HUB_OFFLINE=%r TRANSFORMERS_OFFLINE=%r %s=%r"
          % (os.environ.get("HF_HUB_OFFLINE"), os.environ.get("TRANSFORMERS_OFFLINE"),
             RUN_ENV, os.environ.get(RUN_ENV)))
    print("           模型缓存 revision=%s" % revision)

    # 语料复制到临时目录后建索引：对 rag_knowledge/ 零写入。
    with tempfile.TemporaryDirectory(prefix="rag_prod_eval_") as temp_dir:
        for name in filenames:
            shutil.copy2(CORPUS_DIR / name, Path(temp_dir) / name)

        # 延迟导入：模块导入期不会加载 Embedding。
        from rag.knowledge_base import KnowledgeBase
        from rag.embedding import Embedding

        try:
            embedding = Embedding()
        except Exception as exc:  # pragma: no cover - 仅在缓存损坏时触发
            print("[失败] 本地模型加载失败：%r" % exc)
            return 2

        kb = KnowledgeBase(temp_dir, embedding_factory=lambda: embedding)

        rows = []
        for q in queries:
            result = kb.search(q["text"], top_k=10)
            if not result.get("ok"):
                print("[失败] 检索返回错误：%s -> %r" % (q["query_id"], result))
                return 1
            ranked = [(item["source"], item["chunk_id"]) for item in result["results"]]
            relevant = {
                (item["filename"], item["chunk_id"])
                for item in (q.get("expected_evidence") or [])
            }
            complete = {
                (item["filename"], chunk_id)
                for item in (q.get("expected_evidence") or [])
                for chunk_id in (q.get("evidence_complete_chunk_ids") or [])
            }
            rows.append(
                {
                    "q": q,
                    "ranked": ranked,
                    "scores": [float(item["score"]) for item in result["results"]],
                    "relevant": relevant,
                    "complete": complete,
                }
            )

    answerable = [r for r in rows if r["relevant"]]
    no_answer = [r for r in rows if not r["relevant"]]

    print("\n[有答案 %d 条]" % len(answerable))
    print("  id     Hit@1 Hit@3 Hit@10 MRR@3 P@3   完整证据@3  top1")
    for row in answerable:
        qid = row["q"]["query_id"]
        complete_hit = 1.0 if (set(row["ranked"][:3]) & row["complete"]) else 0.0
        top1 = row["scores"][0] if row["scores"] else float("nan")
        print(
            "  %-6s %.2f  %.2f  %.2f   %.2f  %.2f  %.2f       %.4f"
            % (
                qid,
                hit_at_k(row["ranked"], row["relevant"], 1),
                hit_at_k(row["ranked"], row["relevant"], 3),
                hit_at_k(row["ranked"], row["relevant"], 10),
                mrr_at_k(row["ranked"], row["relevant"], 3),
                precision_at_k(row["ranked"], row["relevant"], 3),
                complete_hit,
                top1,
            )
        )

    print("\n[汇总 · 有答案]")
    print("  Hit@1  = %.4f" % _mean(hit_at_k(r["ranked"], r["relevant"], 1) for r in answerable))
    print("  Hit@3  = %.4f" % _mean(hit_at_k(r["ranked"], r["relevant"], 3) for r in answerable))
    print("  Hit@10 = %.4f" % _mean(hit_at_k(r["ranked"], r["relevant"], 10) for r in answerable))
    print("  MRR@3  = %.4f" % _mean(mrr_at_k(r["ranked"], r["relevant"], 3) for r in answerable))
    print("  P@3    = %.4f" % _mean(precision_at_k(r["ranked"], r["relevant"], 3) for r in answerable))
    print("  nDCG@3 = %.4f（辅助；单证据下与 MRR@3 同序，信息重复）"
          % _mean(ndcg_at_k(r["ranked"], r["relevant"], 3) for r in answerable))
    print("  证据完整率@3 = %.4f（跨块题目的完整证据块为空，恒记 0）"
          % _mean(
              1.0 if (set(r["ranked"][:3]) & r["complete"]) else 0.0
              for r in answerable
          ))

    print("\n[无答案 %d 条]" % len(no_answer))
    by_category = {}
    for row in no_answer:
        by_category.setdefault(row["q"]["category"], []).append(row)
    for category, rows_in_cat in by_category.items():
        false_recall = sum(1 for r in rows_in_cat if r["ranked"])
        top1 = [r["scores"][0] for r in rows_in_cat if r["scores"]]
        print(
            "  %-14s n=%-2d 误召回=%d/%d  Top1 min=%.4f med=%.4f max=%.4f"
            % (
                category,
                len(rows_in_cat),
                false_recall,
                len(rows_in_cat),
                min(top1) if top1 else float("nan"),
                sorted(top1)[len(top1) // 2] if top1 else float("nan"),
                max(top1) if top1 else float("nan"),
            )
        )
    total_false = sum(1 for r in no_answer if r["ranked"])
    print("  总体误召回率 = %d/%d = %.4f"
          % (total_false, len(no_answer), total_false / len(no_answer) if no_answer else float("nan")))

    print("\n[说明] 本输出只反映当前语料（%d chunk）与当前切分下的检索表现，"
          "不代表阈值结论，也未改动任何生产参数。" % len(enumerate_chunks(CORPUS_DIR, filenames)))
    return 0


# ======================================================================
# 入口
# ======================================================================


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="生产语料 RAG 检索评估（默认只做静态校验）")
    parser.add_argument("--retrieval", action="store_true", help="启用检索评估（需 RUN_PRODUCTION_RAG_EVAL=1 且离线变量就绪）")
    args = parser.parse_args(argv)

    if not DATASET_PATH.is_file():
        print("[失败] 数据集缺失：%s" % DATASET_PATH)
        return 1

    data = load_dataset()
    errors, chunks, stats = validate_dataset(data)

    print("=" * 70)
    print("数据集静态校验：%s" % DATASET_PATH.name)
    print("=" * 70)
    if stats:
        print("  语料文件=%d  chunk=%d  查询=%d  有答案=%d  无答案=%d"
              % (stats["files"], stats["chunks"], stats["queries"],
                 stats["answerable"], stats["no_answer"]))
        for category in CATEGORIES:
            print("    %-14s %d" % (category, stats["counts"].get(category, 0)))
        print("  单块完整证据的查询=%d（跨块题目不伪造完整证据块）"
              % stats["complete_evidence"])

    if errors:
        print("\n[校验未通过] 共 %d 项：" % len(errors))
        for item in errors:
            print("  - %s" % item)
        return 1

    print("\n[校验通过] 结构、哈希、切分参数、锚点唯一性、类别计数均一致。")

    if args.retrieval:
        return run_retrieval(data)

    print("[静态模式] 未指定 --retrieval，未加载任何模型。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
