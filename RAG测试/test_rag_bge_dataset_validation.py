# -*- coding: utf-8 -*-
"""真实 BGE 离线数据集校验（阶段 7B-4A）。

对 `RAG测试/rag_offline_eval_dataset_v2.json` 执行三类校验：
  A. 数据结构：ID 唯一、必需字段齐全、分类计数符合声明、相关性等级/无答案子类型一致；
  B. 真实分块与锚点：使用生产 `TextLoader` + `TextSplitter` 实际分块，把每条有答案查询的
     锚点映射到真实 (source, chunk_id)；锚点不存在 / 跨文档 / 同文档 overlap 多 chunk 命中
     一律报告并失败，不自行选择或修改标签；部分相关与多 chunk 联合标签单独校验；
  C. 离线安全：模型仅从本地缓存加载；`HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` 在初始化前
     校验；不 import openai/requests/main，不读 .env，不调用任何真实 LLM/视觉/付费 API。

运行方式（显式授权）：
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 RUN_BGE_EVAL=1
    python -B -m pytest RAG测试/test_rag_bge_dataset_validation.py -v -s \
        --basetemp=<唯一新目录> -o tmp_path_retention_policy=all -p no:cacheprovider
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 生产链路组件（轻量，不含 torch / 网络；真实 Embedding 仅测试内惰性加载）
from rag.loader import TextLoader  # noqa: E402
from rag.splitter import TextSplitter  # noqa: E402

MODEL_ID = "BAAI/bge-small-zh-v1.5"
DATASET_PATH = PROJECT_ROOT / "RAG测试" / "rag_offline_eval_dataset_v2.json"

RUN_BGE_EVAL = os.environ.get("RUN_BGE_EVAL") == "1"

CATEGORIES = [
    "明确有答案", "语义改写但有答案", "部分相关证据",
    "主题相似但缺少关键答案", "完全无关",
    "需要多个 chunk 联合回答", "超出知识库范围",
]
GRADES = {"完全相关", "部分相关", "不相关"}
SUBTYPES = {"主题相似但缺少答案", "完全无关", "超出知识库范围"}

GRADE_BY_CAT = {
    "明确有答案": "完全相关",
    "语义改写但有答案": "完全相关",
    "需要多个 chunk 联合回答": "完全相关",
    "部分相关证据": "部分相关",
    "主题相似但缺少关键答案": "不相关",
    "完全无关": "不相关",
    "超出知识库范围": "不相关",
}
SUBTYPE_BY_CAT = {
    "主题相似但缺少关键答案": "主题相似但缺少答案",
    "完全无关": "完全无关",
    "超出知识库范围": "超出知识库范围",
}
REQUIRED_FIELDS = [
    "query_id", "text", "category", "relevance_grade", "expected_evidence",
    "supported_facts", "missing_facts", "no_answer_subtype", "expected_behavior", "rationale",
]


def _check_offline_env():
    """模型初始化前校验离线变量实际值；任一不符立即失败，不初始化模型。"""
    hf = os.environ.get("HF_HUB_OFFLINE")
    tr = os.environ.get("TRANSFORMERS_OFFLINE")
    if hf != "1" or tr != "1":
        pytest.fail(
            "离线环境变量未正确设置：HF_HUB_OFFLINE=%r, TRANSFORMERS_OFFLINE=%r "
            "（必须均为 '1'，且须在启动 pytest 前显式导出）。停止，不初始化模型。" % (hf, tr),
            pytrace=False,
        )


def _huggingface_cache_root():
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _check_model_cache():
    """模型初始化前校验本地缓存快照存在；缺失即失败，不联网兜底。"""
    model_dir = _huggingface_cache_root() / ("models--" + MODEL_ID.replace("/", "--"))
    refs_main = model_dir / "refs" / "main"
    if not refs_main.exists():
        pytest.fail("模型缓存缺失：%s 不存在。停止（未联网兜底）。" % refs_main, pytrace=False)
    try:
        rev = refs_main.read_text(encoding="utf-8").strip()
    except OSError as exc:
        pytest.fail("无法读取模型缓存 refs/main：%s。停止。" % exc, pytrace=False)
    safetensors = model_dir / "snapshots" / rev / "model.safetensors"
    if not safetensors.exists():
        pytest.fail("模型权重缺失：%s。停止（未联网兜底）。" % safetensors, pytrace=False)
    return rev, model_dir


def enumerate_chunks(doc_dir, filenames):
    """用生产 TextLoader + TextSplitter 实际分块，返回 [(filename, chunk_id, text), ...]。"""
    chunks = []
    splitter = TextSplitter()
    for name in sorted(filenames):
        path = Path(doc_dir) / name
        for document in TextLoader(str(path)).load():
            for chunk in splitter.split(document):
                chunks.append((name, chunk.metadata.get("chunk_id"), chunk.page_content))
    return chunks


@pytest.mark.skipif(not RUN_BGE_EVAL, reason="需显式设置 RUN_BGE_EVAL=1 才加载真实模型")
def test_bge_dataset_validation(tmp_path):
    errors = []

    # ---- C. 离线校验（模型初始化前）----
    _check_offline_env()
    rev, model_dir = _check_model_cache()

    from rag.embedding import Embedding
    try:
        embedding = Embedding()
        probe = list(embedding.encode("离线数据集校验探针"))
        dim = len(probe)
    except Exception as exc:
        pytest.fail(
            "真实 BGE 模型加载失败（缓存不可用）：%r。停止，未联网兜底。" % exc,
            pytrace=False,
        )

    # ---- 加载数据集 ----
    if not DATASET_PATH.exists():
        pytest.fail("数据集缺失：%s" % DATASET_PATH, pytrace=False)
    data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    documents = data["documents"]
    queries = data["queries"]
    meta = data["meta"]

    docs_by_id = {d["doc_id"]: d for d in documents}
    id_to_filename = {d["doc_id"]: d["filename"] for d in documents}

    # ---- A. 数据结构校验 ----
    if len(docs_by_id) != len(documents):
        errors.append("doc_id 重复")
    qids = [q["query_id"] for q in queries]
    dup = sorted({q for q in qids if qids.count(q) > 1})
    if dup:
        errors.append("query_id 重复: %s" % dup)
    texts = [q["text"] for q in queries]
    dup_t = sorted({t for t in texts if texts.count(t) > 1})
    if dup_t:
        errors.append("重复查询文本: %s" % dup_t)

    for q in queries:
        for f in REQUIRED_FIELDS:
            if f not in q:
                errors.append("%s 缺字段 %s" % (q.get("query_id", "?"), f))

    catc = Counter(q.get("category") for q in queries)
    for q in queries:
        if q.get("category") not in CATEGORIES:
            errors.append("%s 非法 category=%r" % (q.get("query_id"), q.get("category")))
        if q.get("relevance_grade") not in GRADES:
            errors.append("%s 非法 relevance_grade=%r" % (q.get("query_id"), q.get("relevance_grade")))
        st = q.get("no_answer_subtype")
        if st is not None and st not in SUBTYPES:
            errors.append("%s 非法 no_answer_subtype=%r" % (q.get("query_id"), st))
    for cat, target in meta.get("count_targets", {}).items():
        if catc.get(cat, 0) != target:
            errors.append("类别计数不符: %s 实际=%d 声明=%d" % (cat, catc.get(cat, 0), target))

    for q in queries:
        cat, grade, st = q.get("category"), q.get("relevance_grade"), q.get("no_answer_subtype")
        if cat in GRADE_BY_CAT and grade != GRADE_BY_CAT[cat]:
            errors.append("%s 类别=%s 但 grade=%s" % (q.get("query_id"), cat, grade))
        if cat in SUBTYPE_BY_CAT:
            if st != SUBTYPE_BY_CAT[cat]:
                errors.append("%s 类别=%s 但 no_answer_subtype=%s" % (q.get("query_id"), cat, st))
        elif st is not None:
            errors.append("%s 有答案类别却带 no_answer_subtype=%s" % (q.get("query_id"), st))
        ev = q.get("expected_evidence") or []
        if grade == "不相关":
            if ev:
                errors.append("%s 不相关却给了证据" % q.get("query_id"))
        else:
            if not ev:
                errors.append("%s 有答案但 expected_evidence 为空" % q.get("query_id"))
            if q.get("supported_facts") in (None, ""):
                errors.append("%s 有答案但 supported_facts 为空" % q.get("query_id"))

    # ---- B. 真实分块 + 锚点映射 ----
    for d in documents:
        (tmp_path / d["filename"]).write_text(d["content"], encoding="utf-8")
    chunks = enumerate_chunks(str(tmp_path), list(id_to_filename.values()))

    answerable = [q for q in queries if q.get("relevance_grade") != "不相关"]
    for q in answerable:
        resolved = set()
        for e in q.get("expected_evidence") or []:
            did, anchor = e.get("doc_id"), e.get("anchor")
            if did not in id_to_filename:
                errors.append("%s 证据 doc_id=%s 不存在" % (q["query_id"], did))
                continue
            hits = [(s, c, t) for (s, c, t) in chunks if anchor in t]
            if not hits:
                errors.append("%s 锚点 %r 未命中任何 chunk" % (q["query_id"], anchor))
                continue
            distinct_docs = {s for (s, _, _) in hits}
            if len(distinct_docs) > 1:
                errors.append(
                    "%s 锚点 %r 跨文档命中：%s"
                    % (q["query_id"], anchor, [(s, c) for s, c, _ in hits])
                )
                continue
            if len(hits) > 1:
                errors.append(
                    "%s 锚点 %r 同文档跨 chunk（overlap）命中：%s"
                    % (q["query_id"], anchor, [(s, c) for s, c, _ in hits])
                )
                continue
            s, c, _ = hits[0]
            if s != id_to_filename[did]:
                errors.append(
                    "%s 锚点 %r 命中文档 %s 但期望 %s"
                    % (q["query_id"], anchor, s, id_to_filename[did])
                )
                continue
            resolved.add((s, c))
        if q.get("category") == "需要多个 chunk 联合回答" and len(resolved) < 2:
            errors.append(
                "%s 多chunk联合但实际只落到 %d 个 chunk：%s"
                % (q["query_id"], len(resolved), sorted(resolved))
            )
        if q.get("category") == "部分相关证据" and q.get("missing_facts") in (None, ""):
            errors.append("%s 部分相关但缺失事实未标注" % q["query_id"])

    # ---- 报告 ----
    line = "=" * 70
    print(line)
    print("阶段 7B-4A 真实 BGE 离线数据集校验")
    print(line)
    print("[离线证据]")
    print("  HF_HUB_OFFLINE        = %r" % os.environ.get("HF_HUB_OFFLINE"))
    print("  TRANSFORMERS_OFFLINE  = %r" % os.environ.get("TRANSFORMERS_OFFLINE"))
    print("  RUN_BGE_EVAL          = %r" % os.environ.get("RUN_BGE_EVAL"))
    print("  缓存 revision         = %s" % rev)
    print("  缓存目录              = %s" % model_dir)
    print("  模型加载              = OK（embedding 维度=%d，本地缓存命中）" % dim)
    print("[数据结构]")
    print("  文档数=%d  查询数=%d" % (len(documents), len(queries)))
    for cat in CATEGORIES:
        print("    %-18s %d" % (cat, catc.get(cat, 0)))
    print("  相关等级：完全相关=%d 部分相关=%d 不相关=%d" % (
        sum(1 for q in queries if q.get("relevance_grade") == "完全相关"),
        sum(1 for q in queries if q.get("relevance_grade") == "部分相关"),
        sum(1 for q in queries if q.get("relevance_grade") == "不相关"),
    ))
    print("[真实分块]")
    print("  总 chunk 数=%d" % len(chunks))
    for fname in sorted(id_to_filename.values()):
        ids = [c for (s, c, _) in chunks if s == fname]
        print("    %-14s chunk_ids=%s" % (fname, ids))
    print("[锚点解析（有答案 %d 条）]" % len(answerable))
    for q in answerable:
        rs = set()
        for e in q.get("expected_evidence") or []:
            did, anchor = e.get("doc_id"), e.get("anchor")
            rs.update((s, c) for (s, c, t) in chunks if anchor in t and s == id_to_filename.get(did))
        print("    %-6s %-30s -> %s" % (q["query_id"], q["text"], sorted(rs) if rs else "（解析失败）"))

    if errors:
        print("\n[校验失败 %d 项]" % len(errors))
        for e in errors:
            print("  -", e)
        print(line)
        pytest.fail(
            "数据集校验失败：%d 项问题（详见上方输出）。不修改标签，保留失败现场。" % len(errors),
            pytrace=False,
        )
    else:
        print("\n[校验通过] 结构/标签/锚点/多chunk 全部通过，未发现歧义。")
        print(line)
