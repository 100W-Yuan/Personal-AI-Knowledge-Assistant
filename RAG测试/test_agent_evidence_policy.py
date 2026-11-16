# -*- coding: utf-8 -*-
"""Agent 证据状态与拒答规则测试设计（第 15 轮 A 阶段）。

为「证据不足拒答」最小生产修改提供离线测试依据。本阶段只新增测试，
不修改生产代码。

核心原则：
    - 「证据充分性」与「检索状态」是两个不同概念：
      有候选文档 ≠ 有充分证据；相似度高 ≠ 证据充分。
    - 当前生产代码（agent.py）尚未有明确的「证据状态」字段或
      「证据不足拒答」提示词约束，因此：
        * A~D 组验证「当前已能传递的内容」；
        * 「待生产实现的验收测试」标记未来期望，不伪造为已通过。
    - 只用 Mock 客户端与虚构工具结果，不运行真实模型、不碰正式 DB。

安全：全程离线，不联网、不下载、不读正式 memory.db。
"""

import os

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from agent import Agent        # noqa: E402
from task import AgentState    # noqa: E402


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
# Mock 客户端
# ---------------------------------------------------------------------------
class ScriptedClient:
    def __init__(self, response_content):
        self._content = response_content
        self.calls = []

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, model, messages, **kwargs):
        self.calls.append({"model": model, "messages": messages})
        return _Resp(self._content)


class _Resp:
    def __init__(self, content):
        self.choices = [_Ch(content)]


class _Ch:
    def __init__(self, content):
        self.message = _Msg(content)


class _Msg:
    def __init__(self, content):
        self.content = content


# ---------------------------------------------------------------------------
# 假知识库
# ---------------------------------------------------------------------------
class FakeKnowledgeBase:
    """search() 返回预设结果或抛异常。"""

    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error

    def search(self, query, top_k=3, threshold=None):
        if self._error is not None:
            raise self._error
        return self._result


def make_agent(knowledge_base=None):
    return Agent(
        api_key="fake-key",
        base_url="http://127.0.0.1:1",
        model="fake-model",
        knowledge_base=knowledge_base,
    )


def run_reasoning_with_history(history):
    """构造 agent，注入 execution_history，运行 reasoning，返回捕获的 messages。"""
    agent = make_agent()
    agent.client = ScriptedClient("最终回答")
    state = AgentState("测试问题")
    state.execution_history = list(history)
    agent.reasoning(state)
    return agent, state


# ---------------------------------------------------------------------------
# A 组：正常检索且包含答案事实
# ---------------------------------------------------------------------------
class TestGroupA_HasAnswer:

    def test_context_with_answer_enters_reasoning(self):
        kb = FakeKnowledgeBase(result={
            "ok": True, "query": "电池容量", "count": 1,
            "results": [{"source": "battery.md", "score": 0.9,
                         "chunk": "星辰 X1 电池容量是 5000 毫安时。"}],
            "context": "【相关资料】\n内容：星辰 X1 电池容量是 5000 毫安时。\n",
        })
        agent = make_agent(knowledge_base=kb)
        agent.client = ScriptedClient("答案")
        state = AgentState("电池容量是多少")

        agent.execute_tool(state, "search_knowledge", {"query": "电池容量"})
        # 工具结果进入 execution_history（含完整 context）。
        assert state.execution_history[0]["result"]["result"]["context"]
        assert "5000 毫安" in state.execution_history[0]["result"]["result"]["context"]

        agent.reasoning(state)
        system = agent.client.calls[0]["messages"][0]["content"]
        assert "5000 毫安" in system  # 上下文进入推理请求


# ---------------------------------------------------------------------------
# B 组：正常检索但证据不足（相近主题、缺目标事实）
# ---------------------------------------------------------------------------
class TestGroupB_InsufficientEvidence:

    def test_near_topic_without_fact_recorded_as_insufficient(self):
        # 文档有电池容量，但没有防水等级 → 证据不足。
        kb = FakeKnowledgeBase(result={
            "ok": True, "query": "防水等级", "count": 1,
            "results": [{"source": "battery.md", "score": 0.76,
                         "chunk": "星辰 X1 电池容量是 5000 毫安时，支持 66 瓦快充。"}],
            "context": "【相关资料】\n内容：星辰 X1 电池容量是 5000 毫安时。\n",
        })
        agent = make_agent(knowledge_base=kb)
        agent.client = ScriptedClient("答案")
        state = AgentState("防水等级是多少")

        agent.execute_tool(state, "search_knowledge", {"query": "防水等级"})
        agent.reasoning(state)

        system = agent.client.calls[0]["messages"][0]["content"]
        # 候选文档进入请求（相似度高），但内容不含防水事实。
        assert "5000 毫安" in system

        # 正确检查点：工具返回的 context（检索到的内容）不含防水事实，
        # 而非整个 prompt（prompt 里用户问题本身就含「防水」二字）。
        tool_result = state.execution_history[0]["result"]["result"]
        assert "防水" not in tool_result["context"]

        # 测试数据明确记录：该文档缺少「防水等级」事实。
        # 这里用显式断言锁定「候选 ≠ 证据充分」这一测试意图。
        chunk = tool_result["results"][0]["chunk"]
        assert "防水" not in chunk  # 文档确实缺防水事实


# ---------------------------------------------------------------------------
# C 组：正常无结果（count=0）
# ---------------------------------------------------------------------------
class TestGroupC_EmptyResult:

    def test_empty_result_enters_reasoning_distinct_from_error(self):
        kb = FakeKnowledgeBase(result={
            "ok": True, "query": "q", "count": 0, "results": [],
            "context": "",
        })
        agent = make_agent(knowledge_base=kb)
        agent.client = ScriptedClient("答案")
        state = AgentState("问题")

        result = agent.execute_tool(state, "search_knowledge", {"query": "q"})
        # 工具层把 count==0 转为「未找到相关内容」提示。
        # execute_tool 返回的是 tool_registry.execute 的结构 {success, result}。
        assert result["success"] is True
        assert result["result"]["count"] == 0
        assert result["result"]["context"] == "（知识库中未找到相关内容）"

        agent.reasoning(state)
        system = agent.client.calls[0]["messages"][0]["content"]
        # 无结果提示进入请求；且不是失败信息。
        assert "（知识库中未找到相关内容）" in system
        assert "知识检索失败" not in system


# ---------------------------------------------------------------------------
# D 组：工具执行失败
# ---------------------------------------------------------------------------
class TestGroupD_ToolFailure:

    def test_failure_enters_history_and_reasoning(self):
        kb = FakeKnowledgeBase(error=RuntimeError("知识检索失败（init）"))
        agent = make_agent(knowledge_base=kb)
        agent.client = ScriptedClient("答案")
        state = AgentState("问题")

        result = agent.execute_tool(state, "search_knowledge", {"query": "q"})
        assert result["success"] is False  # 失败，非正常无结果

        agent.reasoning(state)
        system = agent.client.calls[0]["messages"][0]["content"]
        assert "知识检索失败" in system  # 失败信息进入请求，可区分于无结果


# ---------------------------------------------------------------------------
# E 组：非知识库任务与长期记忆隔离（不调用正式 DB）
# ---------------------------------------------------------------------------
class TestGroupE_Isolation:

    def test_search_memory_not_subject_to_knowledge_policy(self):
        """行为级验证：search_memory 的访问门控与 search_knowledge 的知识
        意图门控相互独立——知识检索策略不会阻断、也不会绕过个人记忆的
        既有权限控制。

        阶段 6B 在 execute_tool 中新增了 search_knowledge 门控分支，原先
        "execute_tool 源码不得出现 search_knowledge 分支"的结构断言已过时，
        故改为行为断言：用 spy 替换 search_memory 底层函数，验证
        - 第三方主体（非本人主语）仍被记忆主体守卫拒绝，且不执行底层查询；
        - 明确本人主语仍正常放行并执行工具；
        - 拒绝理由来自"长期记忆/主体守卫"，而非"知识检索"门控。
        全程不访问正式数据库、不修改记忆权限逻辑。
        """
        agent = make_agent()
        assert "search_memory" != "search_knowledge"

        # 用 spy 替换 search_memory 的底层函数，观察是否真正执行。
        calls = []

        def spy_search_memory(query, category=None):
            calls.append((query, category))
            return {"query": query, "count": 0, "memories": []}

        agent.tool_registry.tools["search_memory"]["function"] = spy_search_memory

        # 1) 第三方主体（无本人主语）：主体守卫拒绝，底层查询不执行。
        third_party = AgentState("他的职业目标是什么")
        refused = agent.execute_tool(
            third_party, "search_memory", {"query": "他的职业目标是什么"}
        )
        assert refused["success"] is True
        assert refused["result"]["refused"] is True
        assert "长期记忆" in refused["result"]["reason"]      # 记忆门控
        assert "知识检索" not in refused["result"]["reason"]  # 非知识门控
        assert calls == []  # 底层查询未执行

        # 2) 明确本人主语：放行并执行工具。
        own = AgentState("我叫什么")
        result = agent.execute_tool(
            own, "search_memory", {"query": "我叫什么"}
        )
        assert result["success"] is True
        assert result["result"]["count"] == 0
        assert calls == [("我叫什么", None)]  # 底层查询确已执行


# ---------------------------------------------------------------------------
# 待生产实现的验收测试（标记，不伪造为已通过）
# ---------------------------------------------------------------------------
class TestPendingEvidencePolicy:
    """这些测试验证「未来生产修改应满足的期望行为」。

    当前生产代码尚无证据状态字段/拒答提示词约束，故这些测试
    **预期失败**，用于驱动下一阶段的最小修改，不伪造为已通过。
    """

    @pytest.mark.xfail(reason="待生产实现：reasoning 提示词尚未包含证据边界约束",
                       strict=False)
    def test_reasoning_prompt_should_contain_evidence_policy(self):
        agent = make_agent()
        agent.client = ScriptedClient("答案")
        state = AgentState("问题")
        state.execution_history = [{
            "type": "tool", "tool_name": "search_knowledge",
            "arguments": {}, "result": {"success": True, "result": {"count": 0}},
        }]
        agent.reasoning(state)
        system = agent.client.calls[0]["messages"][0]["content"]
        # 期望：提示词包含「不得把相近内容当作答案」等证据边界约束。
        assert "相近" in system or "证据" in system or "依据" in system
