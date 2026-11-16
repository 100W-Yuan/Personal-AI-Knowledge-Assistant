# -*- coding: utf-8 -*-
"""RAG 7.9.15：execution_history 执行历史专项测试（直接观测）。

方法：monkeypatch 替换 agent 模块级名字 `AgentState` 为不覆盖任何行为的
子类，记录 run() 创建的实例；每个测试使用**独立捕获列表**（子类在测试内
定义，闭包持有列表），run 前记录基线长度、run 后仅检查新增实例。
同时以 registry.execute 的实例级包装做工具调用计数，与历史条目交叉核对。

原则（依据 RAG 7.9.14 源码审查）：
- execution_history 真实条目只有三类：tool（含 success）、error、reasoning；
- 源码不记录 Retry/Replan 事件——本文件不虚构这两类条目；
- Replan 后执行的新步骤会以新的 tool/reasoning 条目出现，序列即证据。

全程 FakeClient：零真实 API / 零模型 / 零正式数据库。
"""

import io
import contextlib
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import agent as agent_module          # noqa: E402  （用于 monkeypatch 模块级名字）
from agent import Agent               # noqa: E402
from memory import Memory             # noqa: E402

DUMMY = dict(api_key="dummy", base_url="https://example.invalid", model="dummy")

EXTRACT_OK = json.dumps({"should_remember": False, "memories": []})


class ScriptedClient:
    def __init__(self, script):
        self._script = list(script)
        self.calls = []

    def _complete(self, messages):
        self.calls.append(messages)
        if self._script:
            return self._script.pop(0)
        return json.dumps({"plan": [{"description": "回答", "type": "reasoning"}]})

    @property
    def chat(self):
        outer = self

        class _C:
            class completions:
                @staticmethod
                def create(model=None, messages=None, **kw):
                    content = outer._complete(messages)

                    class _M:
                        pass
                    m = _M(); m.content = content

                    class _Ch:
                        message = m

                    class _R:
                        choices = [_Ch()]
                    return _R()
        return _C()


def build_agent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    a = Agent(**DUMMY)
    a.memory.close()
    a.memory = Memory(db_path=str(tmp_path / "memory.db"))
    return a


def plan_with_tool(tool_name):
    return json.dumps({"plan": [
        {"description": "执行工具", "type": "tool", "tool_name": tool_name},
        {"description": "回答", "type": "reasoning"},
    ]})


def finish_plan():
    return json.dumps({"plan": [{"description": "回答", "type": "reasoning"}]})


def install_state_capture(monkeypatch):
    """安装独立捕获列表：子类不改行为，仅记录实例。"""
    captured = []

    class SpyState(agent_module.AgentState):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            captured.append(self)

    monkeypatch.setattr(agent_module, "AgentState", SpyState)
    return captured


def install_tool_counter(agent):
    """工具调用计数（同 7.9.12 思路）：包装实例方法，记录 (tool, success)。"""
    log = []

    real_execute = agent.tool_registry.execute

    def spy(name, arguments):
        result = real_execute(name, arguments)
        log.append({"tool": name, "success": result.get("success")})
        return result

    agent.tool_registry.execute = spy
    return log


def run_and_capture(agent, captured, question, script):
    baseline = len(captured)
    agent.client = ScriptedClient(script)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = agent.run(question, user_id="test-user")
    new_states = captured[baseline:]
    return result, buf.getvalue(), new_states


def tool_history_entries(state):
    return [e for e in state.execution_history if e.get("type") == "tool"]


# ======================================================================
# 测试 A：失败 → Replan → 后续成功
# ======================================================================

def test_a_failure_replan_then_success_history(tmp_path, monkeypatch):
    agent = build_agent(tmp_path, monkeypatch)
    captured = install_state_capture(monkeypatch)
    tool_log = install_tool_counter(agent)

    script = [EXTRACT_OK, plan_with_tool("test_failure"),
              "{}", "{}",                        # 两次失败（重试上限）
              finish_plan(), "中间", "最终答案。"]
    result, output, states = run_and_capture(
        agent, captured, "故障后换策略", script)

    # 捕获断言：必须恰好捕获 1 个状态对象（不允许静默失败）
    assert len(states) == 1
    state = states[0]

    # 失败工具记录 ×2 + Reasoning 条目（源码确认 reasoning 步骤会追加）
    assert len(state.execution_history) == 3
    first, second, reasoning = state.execution_history[0], \
        state.execution_history[1], state.execution_history[2]

    for entry in (first, second):
        assert entry["type"] == "tool"
        assert entry["tool_name"] == "test_failure"
        assert entry["result"]["success"] is False
    assert reasoning["type"] == "reasoning"
    assert reasoning["result"] == "中间"

    # 历史与工具调用计数交叉核对
    assert len(tool_log) == 2
    assert [c["success"] for c in tool_log] == [False, False]
    assert len(tool_history_entries(state)) == len(tool_log)

    # 既有控制流契约：replan 后成功收尾
    assert result["success"] is True


# ======================================================================
# 测试 B：同签名 Replan 后再次失败并终止
# ======================================================================

def test_b_same_signature_terminates_history(tmp_path, monkeypatch):
    agent = build_agent(tmp_path, monkeypatch)
    captured = install_state_capture(monkeypatch)
    tool_log = install_tool_counter(agent)

    script = [EXTRACT_OK, plan_with_tool("test_failure"),
              "{}", "{}",                           # 两次失败 → replan
              plan_with_tool("test_failure"),       # replan 后同签名策略
              "{}",                                 # 第三次失败 → 终止
              finish_plan(), "中间", "最终答案。"]
    result, output, states = run_and_capture(
        agent, captured, "不可恢复", script)

    assert len(states) == 1
    state = states[0]

    # 三条失败工具记录，无 reasoning（终止后未继续），无重复/无伪造条目
    tool_entries = tool_history_entries(state)
    assert len(tool_entries) == 3
    assert all(e["tool_name"] == "test_failure" for e in tool_entries)
    assert all(e["result"]["success"] is False for e in tool_entries)
    assert len(state.execution_history) == 3          # 不存在额外条目
    assert "🧠 执行 Reasoning" not in output           # 终止后未继续步骤
    assert result["success"] is False

    # 历史与调用计数一致
    assert len(tool_log) == 3
    assert [c["success"] for c in tool_log] == [False, False, False]


# ======================================================================
# 测试 C：同一 Agent 的两次独立任务
# ======================================================================

def test_c_independent_tasks_isolated_history(tmp_path, monkeypatch):
    agent = build_agent(tmp_path, monkeypatch)
    captured = install_state_capture(monkeypatch)
    tool_log = install_tool_counter(agent)

    def one_run(question):
        script = [EXTRACT_OK, plan_with_tool("test_failure"),
                  "{}", "{}", finish_plan(), "中间", "最终答案。"]
        return run_and_capture(agent, captured, question, script)

    r1, _, states1 = one_run("任务一")
    calls_after_first = len(tool_log)
    r2, _, states2 = one_run("任务二")
    calls_after_second = len(tool_log)

    assert r1["success"] is True and r2["success"] is True
    assert len(states1) == 1 and len(states2) == 1
    s1, s2 = states1[0], states2[0]

    # 不同任务对象：state / history 列表 / control 均相互独立
    assert s1 is not s2
    assert s1.execution_history is not s2.execution_history
    assert s1.control is not s2.control

    # 第二次任务的历史不含第一次记录（各自 3 条——2 失败 + 1 reasoning）
    assert len(s1.execution_history) == 3
    assert len(s2.execution_history) == 3
    assert all(
        e["type"] in ("tool", "reasoning") for e in s2.execution_history
    )

    # 工具调用计数按任务切分：各 2 次
    assert calls_after_first == 2
    assert calls_after_second - calls_after_first == 2
