# -*- coding: utf-8 -*-
"""真实 BGE 离线检索质量评测基线（阶段 7B-1）。

设计目标
--------
- 用**真实本地语义向量**（BAAI/bge-small-zh-v1.5）而非 keyword one-hot，
  对生产检索链路（TextLoader → TextSplitter → Embedding.encode →
  VectorStore 余弦 → Retriever Top-K → KnowledgeBase 默认过滤）做离线基线评测。
- 语料为**独立自编**的 8 份文档 + 14 条查询，写入 pytest 临时目录，
  **绝不触碰正式 rag_knowledge/**。
- 相关性标注用**文本锚点**（答案原文子串），由**实际生产分块结果**在运行时
  解析为 (source, chunk_id)；锚点未命中或命中多个 chunk 立即失败并输出
  chunk ID 与文本，不自行选择/修改/放宽标注。

安全约束（阶段 7B-1 第四部分）
------------------------------
- 必须在启动 pytest 前显式导出 HF_HUB_OFFLINE=1、TRANSFORMERS_OFFLINE=1、
  RUN_BGE_EVAL=1；本模块在**模型初始化前**校验实际值，不符即 pytest.fail。
- 模型加载失败（缓存缺失等）立即 pytest.fail，**不联网兜底、不静默跳过**。
- 不 import openai/requests/main；不读取 .env；不调用真实 LLM/视觉/付费 API。
- 本评测只输出指标与诊断，不预设阈值、不承诺通过率。

运行方式（显式授权）：
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 RUN_BGE_EVAL=1
    python -B -m pytest 自动测试脚本/test_rag_bge_retrieval_baseline.py -v -s \
        --basetemp=<唯一新目录> -o tmp_path_retention_policy=all -p no:cacheprovider
"""

import math
import os
import statistics
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 生产链路组件（均为轻量模块，不含 torch / 网络；真实 Embedding 仅测试内惰性加载）
from rag.knowledge_base import KnowledgeBase  # noqa: E402
from rag.loader import TextLoader  # noqa: E402
from rag.splitter import TextSplitter  # noqa: E402

MODEL_ID = "BAAI/bge-small-zh-v1.5"

# 门控：仅显式授权时运行；未授权则跳过（不加载模型）。
RUN_BGE_EVAL = os.environ.get("RUN_BGE_EVAL") == "1"


# ---------------------------------------------------------------------------
# 数据集（独立自编语料，写入 tmp_path，绝不触碰正式 rag_knowledge/）
# ---------------------------------------------------------------------------

DATASET = {
    "退货政策.md": "退货政策：商品支持七天无理由退货，退货时需保持商品完好，不影响二次销售。",
    "退货运费.md": "退货运费说明：七天无理由退货时运费由买家承担；商品存在质量问题时运费由卖家承担。",
    "物流配送.md": "配送说明：默认使用顺丰快递，配送时效为一到三天，偏远地区会适当延长。",
    "支付方式.md": "支付说明：支持微信支付和支付宝，不支持货到付款，付款后订单即时生效。",
    "售后保修.md": "保修说明：电子产品提供一年质保，人为损坏不在保修范围内，需凭购买凭证申请。",
    "会员积分.md": "会员积分规则：购物每满一百元积一分，积分可在下次购物时抵扣，一百积分可抵扣一元。",
    "门店服务.md": "门店服务：全国门店营业时间为每天九点到晚上九点，节假日正常营业，欢迎到店咨询。",
    "产品说明书.md": (
        "智能保温杯使用说明书。产品概述：本杯采用不锈钢内胆，可保温十二小时、保冷二十四小时，容量五百毫升。"
        "日常清洁保养：首次使用前请用清水冲洗，日常可用软布擦拭，请勿使用钢丝球或强酸强碱清洁剂，避免划伤内壁。"
        "常见故障排查：若保温效果变差，请先检查杯盖密封圈是否完好；若杯盖漏水，请检查密封圈是否安装到位；"
        "若出现异味，请用温水加小苏打浸泡后冲洗。保修条款：本产品自购买之日起保修一年。"
    ),
}

# (类别, 查询, 相关锚点列表)
# 锚点 = 答案原文子串，必须在全语料中命中且仅命中 1 个 chunk。
# 空列表 = 无关 / 无答案（|R|=0，不进入宏平均，单列报告）。
QUERIES = [
    ("措辞一致", "商品支持七天无理由退货吗", ["商品支持七天无理由退货"]),
    ("措辞一致", "配送时效是多久", ["配送时效为一到三天"]),
    ("语义相同·措辞不同", "买了东西能退吗，需要什么条件", ["保持商品完好"]),
    ("语义相同·措辞不同", "快递寄到大概要几天", ["配送时效为一到三天"]),
    ("多相关", "退货和保修分别怎么处理", ["不影响二次销售", "需凭购买凭证申请"]),
    ("多相关", "支持哪些支付方式，退货时运费谁出", ["支持微信支付和支付宝", "运费由买家承担"]),
    ("相近事实不同", "退货时运费由谁承担", ["运费由买家承担"]),
    ("相近事实不同", "保修期内人为弄坏了还能换吗", ["人为损坏不在保修范围内"]),
    ("无关", "今天天气怎么样", []),
    ("无答案", "门店的具体地址在哪里", []),
    ("无答案", "会员积分能不能兑换现金", []),
    ("长文本多chunk", "说明书里怎么清洁保养保温杯", ["日常清洁保养"]),
    ("长文本多chunk", "说明书里保温效果变差先检查什么", ["若保温效果变差"]),
    ("无答案", "积分能不能直接抵运费", []),
]


# ---------------------------------------------------------------------------
# 离线校验（在模型初始化之前执行）
# ---------------------------------------------------------------------------

def _check_offline_env():
    """模型初始化前校验离线变量实际值；任一不符立即失败，不初始化模型。"""
    hf = os.environ.get("HF_HUB_OFFLINE")
    tr = os.environ.get("TRANSFORMERS_OFFLINE")
    if hf != "1" or tr != "1":
        pytest.fail(
            "离线环境变量未正确设置：HF_HUB_OFFLINE=%r, TRANSFORMERS_OFFLINE=%r "
            "（必须均为 '1'，且须在启动 pytest 前显式导出）。停止，不初始化模型。"
            % (hf, tr),
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
        pytest.fail(
            "模型缓存缺失：%s 不存在。停止（未联网兜底）。" % refs_main,
            pytrace=False,
        )
    try:
        rev = refs_main.read_text(encoding="utf-8").strip()
    except OSError as exc:
        pytest.fail("无法读取模型缓存 refs/main：%s。停止。" % exc, pytrace=False)
    safetensors = model_dir / "snapshots" / rev / "model.safetensors"
    if not safetensors.exists():
        pytest.fail(
            "模型权重缺失：%s。停止（未联网兜底）。" % safetensors,
            pytrace=False,
        )
    return rev


# ---------------------------------------------------------------------------
# 指标（chunk 级，二值相关；|R|=0 返回 None，不进入宏平均）
# ---------------------------------------------------------------------------

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
    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, item in enumerate(ranked[:k], start=1)
        if item in relevant
    )
    ideal = sum(
        1.0 / math.log2(i + 1)
        for i in range(1, min(k, len(relevant)) + 1)
    )
    return dcg / ideal if ideal else 0.0


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _median(vals):
    return statistics.median(vals) if vals else None


# ---------------------------------------------------------------------------
# 分块枚举与锚点解析
# ---------------------------------------------------------------------------

def enumerate_chunks(doc_dir):
    """复用生产分块，返回 [(source, chunk_id, text), ...]（按文件名排序）。"""
    chunks = []
    splitter = TextSplitter()
    for name in sorted(p.name for p in Path(doc_dir).iterdir()
                       if p.suffix in (".md", ".txt")):
        path = Path(doc_dir) / name
        for document in TextLoader(str(path)).load():
            for chunk in splitter.split(document):
                chunks.append(
                    (name, chunk.metadata.get("chunk_id"), chunk.page_content)
                )
    return chunks


def _fmt_chunks(chunks):
    return "; ".join("%s#%s" % (src, cid) for src, cid, _ in chunks)


def resolve_anchors(chunks, anchors):
    """锚点 → 相关 (source, chunk_id) 集合；命中 0 或 >1 个 chunk 立即失败。"""
    relevant = set()
    for anchor in anchors:
        hits = [(s, c, t) for (s, c, t) in chunks if anchor in t]
        if len(hits) == 0:
            pytest.fail(
                "锚点未命中任何 chunk：%r\n"
                "全部 chunk（id + 文本）：\n%s"
                % (anchor, "\n".join("%s#%s => %r" % (s, c, t) for s, c, t in chunks)),
                pytrace=False,
            )
        if len(hits) > 1:
            pytest.fail(
                "锚点命中多个 chunk（标注歧义）：%r\n"
                "命中（id + 文本）：\n%s"
                % (anchor, "\n".join("%s#%s => %r" % (s, c, t) for s, c, t in hits)),
                pytrace=False,
            )
        relevant.add((hits[0][0], hits[0][1]))
    return relevant


# ---------------------------------------------------------------------------
# 评测主体
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not RUN_BGE_EVAL,
    reason="需显式设置 RUN_BGE_EVAL=1 才加载真实（本地缓存）BGE 模型",
)
def test_rag_bge_retrieval_baseline(tmp_path):
    # 1. 模型初始化前：校验离线变量实际值
    _check_offline_env()
    # 2. 模型初始化前：校验本地缓存
    rev = _check_model_cache()

    # 3. 写入独立语料到临时目录（不触碰正式 rag_knowledge/）
    for name, content in DATASET.items():
        (tmp_path / name).write_text(content, encoding="utf-8")

    # 4. 加载真实模型（缓存缺失/加载失败 → pytest.fail，不联网兜底）
    from rag.embedding import Embedding
    try:
        embedding = Embedding()
    except Exception as exc:
        pytest.fail(
            "真实 BGE 模型加载失败（缓存不可用）：%r。停止，未联网兜底。" % exc,
            pytrace=False,
        )

    # 5. 构建知识库（生产检索链路）
    kb = KnowledgeBase(str(tmp_path), embedding_factory=lambda: embedding)

    # 6. 枚举实际生产分块结果
    chunks = enumerate_chunks(str(tmp_path))

    # 7. 逐条查询评测
    evaluable_rows = []
    noanswer_rows = []
    resolved = []
    for category, query, anchors in QUERIES:
        relevant = resolve_anchors(chunks, anchors) if anchors else set()
        resolved.append((query, relevant))
        result = kb.search(query, top_k=10)
        assert result["ok"] is True, "检索失败：query=%r result=%r" % (query, result)
        ranked = [(r["source"], r["chunk_id"]) for r in result["results"]]
        scores = [float(r["score"]) for r in result["results"]]
        if relevant:
            evaluable_rows.append({
                "category": category,
                "query": query,
                "relevant": relevant,
                "ranked": ranked,
                "scores": scores,
            })
        else:
            noanswer_rows.append({
                "category": category,
                "query": query,
                "ranked": ranked,
                "scores": scores,
                "results": result["results"],
            })

    # 8. 汇总并打印报告（仅控制台，不落盘）
    _print_report(rev, embedding, kb, chunks, resolved, evaluable_rows, noanswer_rows)


# ---------------------------------------------------------------------------
# 报告输出
# ---------------------------------------------------------------------------

def _print_report(rev, embedding, kb, chunks, resolved, evaluable_rows, noanswer_rows):
    line = "=" * 70
    print(line)
    print("阶段 7B-1 真实 BGE 离线检索质量评测基线")
    print(line)

    print("\n[离线证据]")
    print("  HF_HUB_OFFLINE      = %r" % os.environ.get("HF_HUB_OFFLINE"))
    print("  TRANSFORMERS_OFFLINE = %r" % os.environ.get("TRANSFORMERS_OFFLINE"))
    print("  RUN_BGE_EVAL        = %r" % os.environ.get("RUN_BGE_EVAL"))
    print("  模型缓存 revision   = %s" % rev)
    try:
        probe = list(embedding.encode("离线评测探针"))
        print("  模型加载            = OK（embedding 维度=%d）" % len(probe))
    except Exception as exc:
        print("  模型加载            = 失败 %r" % exc)

    print("\n[语料与分块]")
    print("  文档数=%d  chunk 总数=%d" % (kb.file_count, kb.chunk_count))
    for src in sorted(DATASET):
        ids = [c for (s, c, _) in chunks if s == src]
        print("    %s -> chunk_ids=%s" % (src, ids))

    print("\n[锚点解析（实际生产分块 → 相关 chunk）]")
    for query, relevant in resolved:
        rel = sorted("%s#%s" % (s, c) for s, c in relevant) if relevant else ["∅"]
        print("    %-28s => %s" % (query, rel))

    print("\n[可评估查询逐条指标（|R|>0，二值相关）]")
    print("    %-4s %-24s %-6s %-6s %-6s %-6s %-6s %-6s" % (
        "类别", "查询", "R@1", "R@3", "MRR3", "MRR10", "nD3", "nD10"))
    for row in evaluable_rows:
        ranked, rel = row["ranked"], row["relevant"]
        print("    %-4s %-24s %-6.2f %-6.2f %-6.2f %-6.2f %-6.2f %-6.2f" % (
            row["category"], row["query"],
            recall_at_k(ranked, rel, 1), recall_at_k(ranked, rel, 3),
            mrr_at_k(ranked, rel, 3), mrr_at_k(ranked, rel, 10),
            ndcg_at_k(ranked, rel, 3), ndcg_at_k(ranked, rel, 10),
        ))

    n_eval = len(evaluable_rows)
    print("\n[宏平均（仅 |R|>0 可评估查询，N=%d）]" % n_eval)
    if n_eval:
        for k in (1, 3):
            print("    Recall@%d  宏平均 = %.4f" % (
                k, _mean([recall_at_k(r["ranked"], r["relevant"], k) for r in evaluable_rows])))
        for k in (3, 10):
            print("    MRR@%d  宏平均 = %.4f" % (
                k, _mean([mrr_at_k(r["ranked"], r["relevant"], k) for r in evaluable_rows])))
            print("    nDCG@%d 宏平均 = %.4f" % (
                k, _mean([ndcg_at_k(r["ranked"], r["relevant"], k) for r in evaluable_rows])))
        print("    注：语料仅 %d 个 chunk，top_k=10 时 Recall@10 恒为 1.0（无区分度），故以 Recall@1/@3 为有效召回口径。"
              % kb.chunk_count)

    print("\n[无答案/无关独立报告（|R|=0，误召回）]")
    mis_count = 0
    for row in noanswer_rows:
        cnt = len(row["ranked"])
        top1 = row["results"][0] if row["results"] else None
        if cnt > 0:
            mis_count += 1
        if top1 is not None:
            hit = "%s#%s => %r" % (
                top1["source"], top1["chunk_id"], top1["chunk"][:30])
        else:
            hit = "（无命中）"
        print("    [%s] %-24s count=%d  top1_score=%.4f  top1=%s" % (
            row["category"], row["query"], cnt,
            row["scores"][0] if row["scores"] else float("nan"), hit))
        print("           top5_scores=%s" % [round(s, 4) for s in row["scores"][:5]])
    print("    误召回条数（count>0 的无答案/无关查询）= %d / %d" % (mis_count, len(noanswer_rows)))

    print("\n[分数分布（仅诊断，非答案置信度）]")
    eval_top1 = [r["scores"][0] for r in evaluable_rows if r["scores"]]
    na_top1 = [r["scores"][0] for r in noanswer_rows if r["scores"]]
    print("    可评估查询 Top-1 分数：min=%.4f median=%.4f mean=%.4f max=%.4f" % (
        min(eval_top1), _median(eval_top1), _mean(eval_top1), max(eval_top1)))
    print("      明细=%s" % [round(v, 4) for v in eval_top1])
    print("    无答案查询 Top-1 分数：min=%.4f median=%.4f mean=%.4f max=%.4f" % (
        min(na_top1), _median(na_top1), _mean(na_top1), max(na_top1)))
    print("      明细=%s" % [round(v, 4) for v in na_top1])

    print("\n[样本规模与适用范围]")
    print("    样本：%d 份自编文档 / %d 条查询（可评估 %d + 无答案/无关 %d）/ %d 个 chunk。"
          % (len(DATASET), len(QUERIES), n_eval, len(noanswer_rows), kb.chunk_count))
    print("    本评测仅反映「该自编客服语料 + bge-small-zh-v1.5 + 生产检索链路」的检索排序与")
    print("    无答案区分能力，**不能直接证明正式数学题知识库（rag_knowledge/数学题.txt）的")
    print("    实际召回率**；正式数据本轮只读、未参与、未修改。")
    print("    本阶段只建立基线，不设阈值、不承诺通过率。")
    print(line)
