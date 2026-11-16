# -*- coding: utf-8 -*-
"""RAG 6.10：Context 注入与 LLM 问答验证。

链路：问题 → Embedding → Retriever → Top-K → ContextBuilder → Prompt → LLM → 回答
核心验证：LLM 是否真正依据 Retriever 提供的 Context 回答（含防幻觉问题）。

R2-B 安全约束（本轮整改）：
- **导入本模块不读取项目 .env**：``load_dotenv`` 与 Key 读取都推迟到真实调用路径；
- 真实 LLM API 仅在环境变量 ``RUN_REAL_LLM=1`` 时才会被调用；
- 默认（未授权）路径：两个真实 API 用例被 skip，另有一个纯 Stub 用例
  在不加载真实 Embedding、不访问网络的前提下验证 Context 注入与 Prompt 组装；
- 未授权时调用 ``call_llm`` 会直接抛错，作为第二道防线，绝不发请求。

解释器约定：本模块顶层 ``import pytest``，且真实路径依赖 sentence-transformers /
openai / dotenv；这些只安装在 py -3.12，裸 ``python``（3.13）未装。

真实模式运行（会读取 .env、加载真实 Embedding、调用 DeepSeek）：
  RUN_REAL_LLM=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      py -3.12 -m RAG测试.test_rag_llm
  RUN_REAL_LLM=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      py -3.12 -m pytest -q "RAG测试/test_rag_llm.py"

默认模式（零读取 .env、零模型、零网络）：
  py -3.12 -m pytest -q "RAG测试/test_rag_llm.py"
  py -3.12 -m RAG测试.test_rag_llm        （脚本模式：打印 SKIP 并退出 0）
"""

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.context_builder import ContextBuilder       # noqa: E402
from rag.document import Document                    # noqa: E402
from rag.retriever import Retriever                  # noqa: E402
from rag.vector_store import VectorStore             # noqa: E402

# 默认关：只有显式设置 RUN_REAL_LLM=1 才允许真实 API 调用。
RUN_REAL_LLM = os.environ.get("RUN_REAL_LLM") == "1"

BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-flash"
API_CALLS = {"count": 0}

DOCUMENTS = [
    Document("我的 AI Agent 项目已经实现了 Tool Calling、Task Loop 和 Long Term Memory。",
             {"source": "A.txt"}),
    Document("我的 AI Agent 项目目前正在学习 RAG，包括文档加载、文本切分和向量检索。",
             {"source": "B.txt"}),
    Document("我每周进行三次力量训练。", {"source": "C.txt"}),
    Document("今天晚上准备吃面条。", {"source": "D.txt"}),
]

Q1 = "我的 Agent 项目目前实现了什么？"
Q2 = "我的 Agent 项目已经支持图像生成了吗？"


def get_api_key():
    """只读取进程环境变量；不读 .env、不打印任何值。"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("环境缺少 DEEPSEEK_API_KEY（项目 .env），按任务要求立即停止")
    return api_key


def _load_real_api_key():
    """真实调用路径专用：此时才导入 dotenv 并读取项目 .env。"""
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    return get_api_key()


def build_rag_context(question, top_k=2):
    """上一阶段已验证的检索链：Embedding → VectorStore → Retriever → ContextBuilder。

    真实 Embedding 只在函数执行时导入/实例化，导入模块不会加载模型。
    """
    from rag.embedding import Embedding

    embedding = Embedding()
    store = VectorStore()
    for doc in DOCUMENTS:
        store.add(doc, embedding.encode(doc.page_content))

    query_vector = embedding.encode(question)
    results = Retriever(store).retrieve(query_vector, top_k=top_k)
    context = ContextBuilder().build(results)
    return results, context


def build_prompt(context, question):
    return (
        "你是一个问答助手。\n"
        "请根据下面的相关资料回答用户问题。\n"
        "要求：优先使用资料中的信息；不要编造资料中没有的信息；"
        "如果资料中没有答案，明确说明不知道；只回答问题，不需要解释检索过程。\n\n"
        f"{context}\n"
        f"用户问题：\n{question}\n"
        "请直接回答。"
    )


def call_llm(prompt):
    """真实 LLM 调用：默认关闭，未授权直接抛错，不发起任何网络请求。"""
    if not RUN_REAL_LLM:
        raise RuntimeError(
            "未设置 RUN_REAL_LLM=1，禁止真实 LLM 调用；"
            "默认路径应使用 Stub，不应到达此函数。"
        )

    from openai import OpenAI

    client = OpenAI(api_key=_load_real_api_key(), base_url=BASE_URL)
    API_CALLS["count"] += 1
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content


# ----------------------------------------------------------------------
# 断言工具（关键词 / 事实一致性，不写死固定文本）
# ----------------------------------------------------------------------

def answer_mentions_capabilities(answer):
    a = answer.lower()
    implemented_groups = [
        ("tool calling", "工具调用"),
        ("task loop", "任务循环"),
        ("long term memory", "长期记忆"),
    ]
    learning_groups = [("rag",), ("文档加载",), ("切分",), ("向量检索",)]

    hit_implemented = sum(
        1 for group in implemented_groups
        if any(k in a for k in group)
    )
    hit_learning = any(
        any(k in a for k in group) for group in learning_groups
    )
    return hit_implemented, hit_learning


HALLUCINATION_PHRASES = [
    "已经支持图像生成", "已支持图像生成", "支持图像生成",
    "实现了图像生成", "具有图像生成", "图像生成功能已",
    "图像生成功能已经",
]

UNCERTAIN_MARKERS = [
    "不知", "无法", "不能", "没有", "未提及", "未包含", "未说明",
    "不包含", "不清楚", "无法确认", "无法判断", "无法回答", "不在",
    "没有提到", "没提到", "未有", "尚未",
]


def check_no_hallucination(answer):
    """判断回答是否出现"幻觉性确认"。

    注意（6.10 修正）：黑名单短语做子串匹配时，须排除否定/疑问语境——
    "不知道是否已经支持图像生成"包含子串"已经支持图像生成"，但显然是
    正确的不确定回答；只有**无否定/疑问前缀**的肯定陈述才算幻觉。
    """
    normalized = answer.replace(" ", "")
    uncertain = any(m in answer for m in UNCERTAIN_MARKERS)

    neg_context = ("是否", "不知", "没", "未", "无法", "不能", "不")
    hallucinated = False
    for phrase in HALLUCINATION_PHRASES:
        start = 0
        while True:
            idx = normalized.find(phrase, start)
            if idx == -1:
                break
            window = normalized[max(0, idx - 6):idx]
            if not any(n in window for n in neg_context):
                hallucinated = True
                break
            start = idx + 1
        if hallucinated:
            break
    return (not hallucinated), uncertain


# ----------------------------------------------------------------------
# 默认离线用例：Stub 编码 + 不调用 LLM
# ----------------------------------------------------------------------

class StubEmbedding:
    """确定性 stub 编码器：按关键词计数生成 3 维向量（零模型加载）。"""

    def encode(self, text):
        return [
            float(text.count("项目")),
            float(text.count("Agent")),
            float(text.count("RAG")),
        ]


def test_context_and_prompt_without_real_llm():
    """不加载真实模型、不调用 API，验证检索上下文与 Prompt 组装。"""
    stub = StubEmbedding()
    store = VectorStore()
    for doc in DOCUMENTS:
        store.add(doc, stub.encode(doc.page_content))

    results = Retriever(store).retrieve(
        stub.encode(Q1),
        top_k=2,
    )
    context = ContextBuilder().build(results)

    assert "Tool Calling" in context and "Task Loop" in context
    assert "Long Term Memory" in context
    assert "RAG" in context

    prompt = build_prompt(context, Q1)
    assert "Tool Calling" in prompt and Q1 in prompt

    # 默认路径绝不允许走到真实调用；真实模式下不执行该负向断言，
    # 避免为了验证"拒绝"反而触发一次付费调用。
    if not RUN_REAL_LLM:
        with pytest.raises(RuntimeError):
            call_llm(prompt)


# ----------------------------------------------------------------------
# 真实 API 用例（默认 skip；仅 RUN_REAL_LLM=1 时执行）
# ----------------------------------------------------------------------

@pytest.mark.skipif(
    not RUN_REAL_LLM,
    reason="需显式设置 RUN_REAL_LLM=1 才调用真实 DeepSeek API",
)
def test_q1_answer_uses_context():
    results, context = build_rag_context(Q1)
    sources = [r["document"].metadata.get("source") for r in results]

    assert "Tool Calling" in context and "Task Loop" in context
    assert "Long Term Memory" in context
    assert "RAG" in context

    prompt = build_prompt(context, Q1)
    assert "Tool Calling" in prompt and Q1 in prompt  # Context 注入 Prompt

    answer = call_llm(prompt)
    assert isinstance(answer, str) and answer.strip()

    hit_impl, hit_learn = answer_mentions_capabilities(answer)
    # 断言对齐任务书示例：“回答至少包含 Agent 项目已有能力中的若干关键词”。
    # 注：B 的内容是“正在学习 RAG”（非“已实现”），问题问“实现了什么”时
    # 模型不提及 B 属正确行为；Context 中 A/B 的注入由上方 prompt 断言覆盖。
    assert hit_impl >= 2, (
        f"回答未充分提及已实现能力（命中 {hit_impl} 组）；来源={sources}；回答={answer}"
    )


@pytest.mark.skipif(
    not RUN_REAL_LLM,
    reason="需显式设置 RUN_REAL_LLM=1 才调用真实 DeepSeek API",
)
def test_q2_no_hallucination():
    results, context = build_rag_context(Q2)
    assert "图像" not in context and "生成" not in context

    prompt = build_prompt(context, Q2)
    answer = call_llm(prompt)
    assert isinstance(answer, str) and answer.strip()

    not_hallucinated, uncertain = check_no_hallucination(answer)
    assert not_hallucinated, f"出现幻觉性确认（图像生成）：{answer}"
    assert uncertain, f"回答未表现为不确定/不知道：{answer}"


# ----------------------------------------------------------------------
# 脚本模式（打印完整过程并返回退出码）
# ----------------------------------------------------------------------

def run_question(question):
    results, context = build_rag_context(question)
    prompt = build_prompt(context, question)
    answer = call_llm(prompt)
    return results, context, prompt, answer


def main():
    if not RUN_REAL_LLM:
        print(
            "[SKIP] 未设置 RUN_REAL_LLM=1："
            "未读取 .env、未加载真实 Embedding、未调用 LLM。"
        )
        return 0

    checks = []

    def record(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    print("=" * 68)
    print("RAG 6.10 Context 注入与 LLM 问答验证")
    print("=" * 68)

    # ------- 问题 1 -------
    r1, ctx1, prompt1, ans1 = run_question(Q1)
    src1 = [r["document"].metadata.get("source") for r in r1]

    print(f"\n【问题 1】{Q1}")
    print(f"[Retriever] Top-K 来源 = {src1}")
    print(f"[Context] 长度={len(ctx1)}")
    print(f"[Answer] {ans1}")

    record("Q1 Context 含 A 内容（Tool Calling/Task Loop/Long Term Memory）",
           "Tool Calling" in ctx1 and "Task Loop" in ctx1 and "Long Term Memory" in ctx1)
    record("Q1 Context 含 B 内容（RAG/文档加载/文本切分/向量检索）",
           "RAG" in ctx1 and "文档加载" in ctx1 and "向量检索" in ctx1)
    record("Q1 Prompt 注入 Context（含相关资料与问题）",
           "Tool Calling" in prompt1 and Q1 in prompt1)
    record("Q1 回答非空", isinstance(ans1, str) and bool(ans1.strip()))

    hit_impl, hit_learn = answer_mentions_capabilities(ans1)
    record("Q1 回答与 Context 一致（已实现能力 ≥2 组）", hit_impl >= 2,
           f"已实现命中={hit_impl}/3；学习内容提及（观察项，非断言）={hit_learn}")

    # ------- 问题 2 -------
    r2, ctx2, prompt2, ans2 = run_question(Q2)
    src2 = [r["document"].metadata.get("source") for r in r2]

    print(f"\n【问题 2】{Q2}")
    print(f"[Retriever] Top-K 来源 = {src2}")
    print(f"[Context] 长度={len(ctx2)}（不含图像生成信息："
          f"{'图像' not in ctx2 and '生成' not in ctx2}）")
    print(f"[Answer] {ans2}")

    record("Q2 Context 不含图像生成信息（前提成立）",
           "图像" not in ctx2 and "生成" not in ctx2)
    not_hallucinated, uncertain = check_no_hallucination(ans2)
    record("Q2 回答未出现幻觉性确认（不声称已支持图像生成）", not_hallucinated)
    record("Q2 回答表现为不确定/不知道（符合预期）", uncertain)
    record("Q2 回答非空", isinstance(ans2, str) and bool(ans2.strip()))

    print("\n" + "-" * 68)
    for name, ok, detail in checks:
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {name}" + (f"  （{detail}）" if detail else ""))
    print("-" * 68)
    print(f"模型：{MODEL_NAME} | API 调用次数（本次运行）：{API_CALLS['count']}")
    print("=" * 68)
    all_passed = all(ok for _, ok, _ in checks)
    print("全部验证通过" if all_passed else "存在失败项，请查看上方 FAIL 行")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
