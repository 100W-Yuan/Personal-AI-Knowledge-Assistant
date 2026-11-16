# -*- coding: utf-8 -*-
"""阶段 6B：知识检索意图门控离线测试。

覆盖三块：
1. 分类器：`Agent._classify_knowledge_intent` 返回 recall / deny / neutral；
2. 执行路径：recall 确定性检索、去重、deny 拒绝、neutral 保留 LLM 决策、
   空结果与失败区分、来源不被拒绝污染；
3. 契约兼容：Agent 成功/失败返回契约、sources 字段形状不变。

零真实 LLM / Embedding / 数据库：
- ScriptedClient 脚本化假客户端（脚本耗尽即报错，防误判）；
- StubKB 替身知识库（记录调用次数，可配置命中/空/失败）；
- monkeypatch.chdir(tmp_path) 隔离 memory.db，不触碰正式数据。

本文件位于 自动测试脚本/ 目录，继承 conftest.py 的离线隔离
（假 dotenv / 假 Key / 空壳 OpenAI / 回环网络守卫）。
"""

import contextlib
import io
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent import Agent                      # noqa: E402
from memory import Memory                    # noqa: E402
from task import AgentState                  # noqa: E402

DUMMY = dict(api_key="dummy", base_url="https://example.invalid", model="dummy")

SOURCE_ITEM = {
    "source": "faq.md",
    "chunk_id": 0,
    "score": 0.9,
    "chunk": "依据片段",
}

HIT_RESULT = {
    "ok": True,
    "count": 1,
    "results": [SOURCE_ITEM],
    "context": "依据片段",
}

EMPTY_RESULT = {
    "ok": True,
    "count": 0,
    "results": [],
    "context": "",
}

FAIL_RESULT = {
    "ok": False,
    "stage": "search",
    "error": "stub 检索异常",
}


class StubKB:
    """替身知识库：记录调用次数，可配置命中 / 空 / 失败三种结果。"""

    def __init__(self, result=None):
        self.result = result
        self.calls = 0
        self.queries = []

    def search(self, query, top_k=3, threshold=None):
        self.calls += 1
        self.queries.append((query, top_k, threshold))
        if self.result is None:
            return {"ok": False, "stage": "search", "error": "stub 检索异常"}
        return dict(self.result, query=query)


class ScriptedClient:
    """脚本化假客户端：预置响应序列；耗尽即报错，避免误判脚本长度。"""

    def __init__(self, script):
        self._script = list(script)
        self.calls = []

    def _complete(self, messages):
        self.calls.append(messages)
        if not self._script:
            raise AssertionError(
                "ScriptedClient 脚本耗尽：LLM 调用次数超出预期，"
                f"已发生 {len(self.calls)} 次调用"
            )
        return self._script.pop(0)

    @property
    def chat(self):
        outer = self

        class _C:
            class completions:
                @staticmethod
                def create(model=None, messages=None, **kwargs):
                    content = outer._complete(messages)

                    class _M:
                        pass
                    m = _M()
                    m.content = content

                    class _Ch:
                        message = m

                    class _R:
                        choices = [_Ch()]
                    return _R()
        return _C()


def build_agent(tmp_path, monkeypatch, knowledge_base=None):
    """构造 Agent，并把 memory 隔离到 tmp_path（不触碰正式 memory.db）。"""
    monkeypatch.chdir(tmp_path)
    agent = Agent(**DUMMY, knowledge_base=knowledge_base)
    agent.memory.close()
    agent.memory = Memory(db_path=str(tmp_path / "memory.db"))
    return agent


def run_agent(agent, user_message, plan, tool_params=None):
    """按 plan 构造脚本并运行一次 Agent，返回 run() 的结果字典。

    脚本顺序固定：extract -> create_plan -> 每个步骤（tool 参数生成 /
    reasoning 文本）-> 最终 reasoning。
    """
    script = [
        json.dumps({"should_remember": False, "memories": []}),
        json.dumps({"plan": plan}),
    ]
    for step in plan:
        if step.get("type") == "tool":
            params = tool_params
            if params is None:
                params = {"query": user_message, "top_k": 3}
            script.append(json.dumps(params))
        else:
            script.append("中间推理")
    script.append("最终答案。")

    agent.client = ScriptedClient(script)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = agent.run(user_message, user_id="test-user")
    return result


def make_reasoning_plan():
    return [{"description": "回答", "type": "reasoning"}]


def make_tool_plan():
    return [
        {"description": "检索项目知识", "type": "tool",
         "tool_name": "search_knowledge"},
        {"description": "回答", "type": "reasoning"},
    ]


# ======================================================================
# 一、分类器测试
# ======================================================================


def test_classify_recall_explicit_knowledge_query():
    classify = Agent._classify_knowledge_intent
    for text in (
        "查询我导入的项目文档，开发计划有哪几个阶段？",
        "根据知识库中的资料，项目使用了什么技术栈？",
        "查一下已导入资料中关于这个问题的说明。",
        "查询知识库中的项目文档",
    ):
        assert classify(text) == "recall", text


def test_classify_recall_not_false_denied_by_bie():
    # "别"不应命中"特别"等复合词导致误判为 deny。
    assert Agent._classify_knowledge_intent("特别查阅资料") == "recall"


def test_classify_deny_explicit_no_search():
    classify = Agent._classify_knowledge_intent
    for text in (
        "不要查知识库，直接回答。",
        "不要搜索我导入的文档。",
        "别查文档",
        "请别查资料",
    ):
        assert classify(text) == "deny", text


def test_classify_deny_greeting():
    classify = Agent._classify_knowledge_intent
    for text in ("你好", "在吗", "早上好"):
        assert classify(text) == "deny", text


def test_classify_personal_memory_is_deny_not_recall():
    classify = Agent._classify_knowledge_intent
    for text in ("我叫什么？", "我的专业是什么？", "我说过什么？"):
        assert classify(text) == "deny", text


def test_classify_neutral_ambiguous():
    classify = Agent._classify_knowledge_intent
    for text in ("这个项目怎么弄？", "给我讲讲原理。"):
        assert classify(text) == "neutral", text


def test_classify_mixed_not_overly_deny():
    classify = Agent._classify_knowledge_intent
    for text in (
        "帮我写代码实现这个功能",
        "结合我的偏好和知识库资料讲讲项目",
    ):
        assert classify(text) in ("neutral", "recall"), text


# ======================================================================
# 二、执行路径测试（execute_tool 直调）
# ======================================================================


def test_deny_blocks_search_knowledge_execution(tmp_path, monkeypatch):
    kb = StubKB(result=HIT_RESULT)
    agent = build_agent(tmp_path, monkeypatch, knowledge_base=kb)

    state = AgentState("不要查知识库，直接回答。")
    result = agent.execute_tool(
        state, "search_knowledge", {"query": "不要查知识库", "top_k": 3}
    )

    assert result["success"] is True
    assert result["result"]["refused"] is True
    assert kb.calls == 0
    # 拒绝记录不污染来源。
    assert Agent._knowledge_sources(state) == []


def test_recall_dedup_in_execute_tool(tmp_path, monkeypatch):
    kb = StubKB(result=HIT_RESULT)
    agent = build_agent(tmp_path, monkeypatch, knowledge_base=kb)

    state = AgentState("查询知识库中的项目文档")

    # 第一次：确定性预检索真正执行。
    first = agent.execute_tool(
        state, "search_knowledge", {"query": "查询知识库中的项目文档", "top_k": 3}
    )
    assert first["success"] is True
    assert kb.calls == 1
    state.knowledge_prefetched_result = first

    # 第二次：Planner 又安排，去重复用缓存，不重复检索。
    second = agent.execute_tool(
        state, "search_knowledge", {"query": "查询知识库中的项目文档", "top_k": 3}
    )
    assert second["success"] is True
    assert kb.calls == 1
    assert second is first


def test_empty_vs_failure_distinguished(tmp_path, monkeypatch):
    # 正常无命中：success=True, count=0。
    kb_empty = StubKB(result=EMPTY_RESULT)
    agent_empty = build_agent(tmp_path, monkeypatch, knowledge_base=kb_empty)
    state_empty = AgentState("查询知识库资料")
    r_empty = agent_empty.execute_tool(
        state_empty, "search_knowledge", {"query": "查询知识库资料", "top_k": 3}
    )
    assert r_empty["success"] is True
    assert r_empty["result"]["count"] == 0

    # 检索异常：success=False，携带 error，不得伪装成无结果。
    kb_fail = StubKB(result=FAIL_RESULT)
    agent_fail = build_agent(tmp_path, monkeypatch, knowledge_base=kb_fail)
    state_fail = AgentState("查询知识库资料")
    r_fail = agent_fail.execute_tool(
        state_fail, "search_knowledge", {"query": "查询知识库资料", "top_k": 3}
    )
    assert r_fail["success"] is False
    assert "error" in r_fail
    assert "stub 检索异常" in r_fail["error"]


def test_other_tools_unaffected(tmp_path, monkeypatch):
    agent = build_agent(tmp_path, monkeypatch, knowledge_base=StubKB(HIT_RESULT))
    state = AgentState("帮我算 1+2")
    result = agent.execute_tool(state, "calculator", {"expression": "1+2"})
    assert result["success"] is True
    assert result["result"] == "3"


# ======================================================================
# 三、run() 集成测试
# ======================================================================


def test_recall_runs_deterministic_retrieval_and_sources_flow(tmp_path, monkeypatch):
    """recall：即使 Planner 不安排工具，也确定性执行一次检索，来源可用。"""
    kb = StubKB(result=HIT_RESULT)
    agent = build_agent(tmp_path, monkeypatch, knowledge_base=kb)

    result = run_agent(
        agent,
        "查询知识库中的项目文档，开发计划有哪几个阶段？",
        make_reasoning_plan(),
    )

    assert kb.calls == 1
    assert result["success"] is True
    assert result["answer"] == "最终答案。"
    assert result["sources"] == [SOURCE_ITEM]


def test_recall_no_double_retrieval_when_planner_schedules(tmp_path, monkeypatch):
    """recall：Planner 也安排 search_knowledge 时，去重不重复检索。"""
    kb = StubKB(result=HIT_RESULT)
    agent = build_agent(tmp_path, monkeypatch, knowledge_base=kb)

    result = run_agent(
        agent,
        "查询知识库中的项目文档",
        make_tool_plan(),
        tool_params={"query": "查询知识库中的项目文档", "top_k": 3},
    )

    assert kb.calls == 1
    assert result["success"] is True
    assert result["sources"] == [SOURCE_ITEM]


def test_deny_blocks_even_when_planner_schedules(tmp_path, monkeypatch):
    """deny：即使 Planner 仍安排 search_knowledge，也不执行检索、不污染来源。"""
    kb = StubKB(result=HIT_RESULT)
    agent = build_agent(tmp_path, monkeypatch, knowledge_base=kb)

    result = run_agent(
        agent,
        "不要查知识库，直接回答。",
        make_tool_plan(),
        tool_params={"query": "不要查知识库", "top_k": 3},
    )

    assert kb.calls == 0
    assert result["success"] is True
    assert result["sources"] == []


def test_neutral_does_not_force_retrieval_when_planner_declines(tmp_path, monkeypatch):
    """neutral：Planner 不安排工具时，不强制检索。"""
    kb = StubKB(result=HIT_RESULT)
    agent = build_agent(tmp_path, monkeypatch, knowledge_base=kb)

    result = run_agent(
        agent,
        "这个项目怎么弄？",
        make_reasoning_plan(),
    )

    assert kb.calls == 0
    assert result["success"] is True
    assert result["sources"] == []


def test_neutral_allows_llm_decision_when_planner_schedules(tmp_path, monkeypatch):
    """neutral：保留 LLM 工具决策——Planner 安排即正常执行。"""
    kb = StubKB(result=HIT_RESULT)
    agent = build_agent(tmp_path, monkeypatch, knowledge_base=kb)

    result = run_agent(
        agent,
        "这个项目怎么弄？",
        make_tool_plan(),
        tool_params={"query": "这个项目怎么弄", "top_k": 3},
    )

    assert kb.calls == 1
    assert result["success"] is True
    assert result["sources"] == [SOURCE_ITEM]


def test_recall_empty_result_is_success_not_failure(tmp_path, monkeypatch):
    """recall 且无命中：success=True、count=0，来源为空，不伪装成失败。"""
    kb = StubKB(result=EMPTY_RESULT)
    agent = build_agent(tmp_path, monkeypatch, knowledge_base=kb)

    result = run_agent(
        agent,
        "查询知识库中的资料",
        make_reasoning_plan(),
    )

    assert kb.calls == 1
    assert result["success"] is True
    assert result["sources"] == []


# ======================================================================
# 四、契约兼容测试
# ======================================================================


def test_run_returns_contract_shape(tmp_path, monkeypatch):
    """成功路径返回 {success, answer, sources}，来源条目字段符合契约。"""
    kb = StubKB(result=HIT_RESULT)
    agent = build_agent(tmp_path, monkeypatch, knowledge_base=kb)

    result = run_agent(
        agent,
        "查询知识库中的项目文档",
        make_reasoning_plan(),
    )

    assert set(result.keys()) >= {"success", "answer", "sources"}
    assert result["success"] is True
    assert isinstance(result["answer"], str)
    assert isinstance(result["sources"], list)
    assert set(result["sources"][0]) == {"source", "chunk_id", "score", "chunk"}
