# -*- coding: utf-8 -*-
"""Agent 层知识证据测试 · 离线补齐（第 14 轮）。

为第十三轮只读审查发现的 Agent 层缺口补充离线测试，覆盖：
    A. 工具结果进入 execution_history；
    B. 空结果（count=0）与工具异常的区分；
    C. Reasoning 请求确实携带 execution_history（含空结果提示/失败信息）。

安全边界：
    - 全程零真实模型、零网络、零云端 API、零正式 DB；
    - Agent 用 fake api_key / base_url 构造（openai SDK 构造不联网）；
    - 用 ScriptedClient 替换 agent.client，不发起真实请求；
    - 不调用 Agent.run()，只直接调用 execute_tool / reasoning，
      避免触发 Memory.save_message 等 DB 写入；
    - 知识库用带 search() 方法的假对象（Stub），不实例化真实 Embedding。

重要说明（防止过度解读）：
    - 本测试只验证「程序实际传递了什么内容」到 reasoning 请求，
      不声称「模型一定会遵守证据约束 / 一定会拒答」。
"""

import json
import os

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from agent import Agent                     # noqa: E402
from task import AgentState                 # noqa: E402


@pytest.fixture(autouse=True)
def isolated_agent_memory(tmp_path, monkeypatch):
    """把 Agent 内部的 Memory() 锁定到 tmp_path（阶段 1-R2）。

    Agent.__init__ 会构造 Memory()，默认路径是项目根目录的正式
    memory.db；本文件只验证请求内容，不需要任何真实历史数据。
    """
    import agent as agent_module
    from memory import Memory

    instances = []

    def _factory(*args, **kwargs):
        instance = Memory(db_path=str(tmp_path / "agent_memory.db"))
        instances.append(instance)
        return instance

    monkeypatch.setattr(agent_module, "Memory", _factory)

    yield

    for instance in instances:
        instance.close()


# ---------------------------------------------------------------------------
# 测试辅助
# ---------------------------------------------------------------------------
class ScriptedClient:
    """预设响应的模拟 OpenAI 客户端：记录每次发送的 messages，返回预设内容。"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, model, messages, **kwargs):
        self.calls.append({"model": model, "messages": messages})
        if not self.responses:
            raise AssertionError("ScriptedClient 没有更多预设响应")
        return self.responses.pop(0)


class _Response:
    def __init__(self, content):
        self.choices = [_Choice(content)]


class _Choice:
    def __init__(self, content):
        self.message = _Message(content)


class _Message:
    def __init__(self, content):
        self.content = content


class FakeKnowledgeBase:
    """假知识库：search() 返回预设结果或抛异常，绝不加载真实模型。"""

    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error
        self.search_calls = []

    def search(self, query, top_k=3, threshold=None):
        self.search_calls.append({"query": query, "top_k": top_k})
        if self._error is not None:
            raise self._error
        return self._result


def make_agent(knowledge_base=None):
    """构造 Agent：fake 凭据，openai 客户端构造不联网；可注入假 KB。"""
    return Agent(
        api_key="fake-key",
        base_url="http://127.0.0.1:1",  # 本机端口，构造时不请求
        model="fake-model",
        knowledge_base=knowledge_base,
    )


# ---------------------------------------------------------------------------
# A 组：工具结果进入 execution_history
# ---------------------------------------------------------------------------
class TestToolResultInHistory:

    def test_search_knowledge_success_written_to_history(self):
        fake_kb = FakeKnowledgeBase(result={
            "ok": True, "query": "q", "count": 1,
            "results": [{"source": "a.md", "score": 0.9, "chunk": "内容"}],
            "context": "【相关资料】\n内容：xxx\n",
        })
        agent = make_agent(knowledge_base=fake_kb)
        state = AgentState("测试问题")

        result = agent.execute_tool(state, "search_knowledge", {"query": "q"})

        assert result.get("success") is True
        # 执行历史应包含一条 tool 记录。
        assert len(state.execution_history) == 1
        entry = state.execution_history[0]
        assert entry["type"] == "tool"
        assert entry["tool_name"] == "search_knowledge"
        assert entry["arguments"] == {"query": "q"}
        # 结果结构正确（成功 + result 字段）。
        assert entry["result"]["success"] is True
        assert entry["result"]["result"]["count"] == 1

    def test_tool_arguments_recorded_correctly(self):
        fake_kb = FakeKnowledgeBase(result={
            "ok": True, "query": "q", "count": 0, "results": [], "context": ""
        })
        agent = make_agent(knowledge_base=fake_kb)
        state = AgentState("测试问题")

        agent.execute_tool(state, "search_knowledge", {"query": "电池容量", "top_k": 5})

        entry = state.execution_history[0]
        assert entry["arguments"] == {"query": "电池容量", "top_k": 5}


# ---------------------------------------------------------------------------
# B 组：空结果与工具异常的区分
# ---------------------------------------------------------------------------
class TestEmptyVsError:

    def test_empty_result_is_success_not_error(self):
        # search_knowledge 工具在 count==0 时返回成功结构（含提示 context），
        # 不抛异常 → 应记录为 success=True。
        fake_kb = FakeKnowledgeBase(result={
            "ok": True, "query": "q", "count": 0, "results": [], "context": ""
        })
        agent = make_agent(knowledge_base=fake_kb)
        state = AgentState("测试问题")

        result = agent.execute_tool(state, "search_knowledge", {"query": "q"})

        assert result.get("success") is True
        entry = state.execution_history[0]
        assert entry["result"]["success"] is True
        assert entry["result"]["result"]["count"] == 0

    def test_tool_error_is_failure_not_empty(self):
        # search_knowledge 工具在 ok=False 时抛 RuntimeError → tool_registry
        # 捕获为 success=False + error，不得伪装成正常空结果。
        fake_kb = FakeKnowledgeBase(error=RuntimeError("知识检索失败"))
        agent = make_agent(knowledge_base=fake_kb)
        state = AgentState("测试问题")

        result = agent.execute_tool(state, "search_knowledge", {"query": "q"})

        assert result.get("success") is False
        entry = state.execution_history[0]
        assert entry["result"]["success"] is False
        assert "error" in entry["result"]

    def test_empty_and_error_distinguishable_in_history(self):
        # 对比两条记录：空结果是 success=True + count==0，
        # 异常是 success=False + error。二者不可混淆。
        kb_empty = FakeKnowledgeBase(result={
            "ok": True, "query": "q", "count": 0, "results": [], "context": ""
        })
        kb_err = FakeKnowledgeBase(error=RuntimeError("boom"))

        agent_empty = make_agent(knowledge_base=kb_empty)
        agent_err = make_agent(knowledge_base=kb_err)
        st_empty = AgentState("a")
        st_err = AgentState("b")

        r_empty = agent_empty.execute_tool(st_empty, "search_knowledge", {"query": "q"})
        r_err = agent_err.execute_tool(st_err, "search_knowledge", {"query": "q"})

        assert r_empty.get("success") is True
        assert r_err.get("success") is False
        assert st_empty.execution_history[0]["result"]["success"] is True
        assert st_err.execution_history[0]["result"]["success"] is False


# ---------------------------------------------------------------------------
# C 组：推理上下文传递
# ---------------------------------------------------------------------------
class TestReasoningContextPropagation:

    def _make_agent_with_reasoning_client(self, response_content="最终答案"):
        agent = make_agent()
        # 替换 client 为 ScriptedClient，捕获 reasoning 请求。
        agent.client = ScriptedClient([_Response(response_content)])
        return agent

    def test_execution_history_enters_reasoning_request(self):
        agent = self._make_agent_with_reasoning_client()
        state = AgentState("测试问题")
        # 预置一条工具结果到执行历史。
        state.execution_history.append({
            "type": "tool",
            "tool_name": "search_knowledge",
            "arguments": {"query": "q"},
            "result": {"success": True, "result": {"count": 1, "context": "证据内容"}},
        })

        agent.reasoning(state)

        # 捕获的 messages 中，system content 应包含执行历史（json 序列化后）。
        assert agent.client.calls, "reasoning 未发起请求"
        system_content = agent.client.calls[0]["messages"][0]["content"]
        assert "search_knowledge" in system_content
        assert "证据内容" in system_content
        # 执行历史确以 JSON 形式传入。
        assert "execution_history" in system_content or "执行历史" in system_content

    def test_empty_hint_enters_reasoning_request(self):
        agent = self._make_agent_with_reasoning_client()
        state = AgentState("测试问题")
        state.execution_history.append({
            "type": "tool",
            "tool_name": "search_knowledge",
            "arguments": {"query": "q"},
            "result": {
                "success": True,
                "result": {"count": 0, "context": "（知识库中未找到相关内容）"},
            },
        })

        agent.reasoning(state)

        system_content = agent.client.calls[0]["messages"][0]["content"]
        assert "（知识库中未找到相关内容）" in system_content

    def test_failure_distinct_from_normal_in_reasoning_request(self):
        agent = self._make_agent_with_reasoning_client()
        state = AgentState("测试问题")
        state.execution_history.append({
            "type": "tool",
            "tool_name": "search_knowledge",
            "arguments": {"query": "q"},
            "result": {"success": False, "error": "知识检索失败（init）：..."},
        })

        agent.reasoning(state)

        system_content = agent.client.calls[0]["messages"][0]["content"]
        # 失败信息进入请求，且与「未找到相关内容」提示是不同内容。
        assert "知识检索失败" in system_content
        assert "（知识库中未找到相关内容）" not in system_content
