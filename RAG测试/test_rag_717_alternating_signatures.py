# -*- coding: utf-8 -*-
"""RAG 7.9.17：A→B→A 交替失败签名专项测试。

构造（RAG 7.9.16 审查确认的通道）：
- 签名 = f"{tool_name}:{error}"（agent.py:1741-1745）；
- 同一工具（calculator）通过不同参数产生不同错误：
  A = 缺少必填参数（参数 {}）；B = 执行期异常（参数 {"expression": "abc"}，
  eval 抛 NameError）；再次 {} 回到 A（文本与 A 逐字符相同）。

测试 1：A→B→A 三连——验证异签名重置连续计数、第三次 A 不被误终止、
        进入 Replan；签名关系从真实 execution_history 复算。
测试 2：Replan 后 A 再次出现——A 已登记进 replanned_signatures，
        再次失败应按终止条件结束（重复失败且已用 replan 机会）。

全程 ScriptedClient + AgentState 捕获 + 工具执行 spy；零真实网络/模型。
"""

import io
import contextlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import agent as agent_module          # noqa: E402
from agent import Agent               # noqa: E402
from memory import Memory             # noqa: E402

DUMMY = dict(api_key="dummy", base_url="https://example.invalid", model="dummy")

EXTRACT_OK = json.dumps({"should_remember": False, "memories": []})
FAIL_MARK = "❌ 工具执行失败"
BAD_ARGS = json.dumps({"expression": "abc"})   # 执行期异常通道


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
    captured = []

    class SpyState(agent_module.AgentState):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            captured.append(self)

    monkeypatch.setattr(agent_module, "AgentState", SpyState)
    return captured


def install_tool_counter(agent):
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
    return result, buf.getvalue(), captured[baseline:]


def signature_of(entry):
    """按生产公式复算签名：tool_name + ':' + error。"""
    return f"{entry['tool_name']}:{entry['result'].get('error')}"


# ======================================================================
# 测试 1：A → B → A（异签名重置；第三次 A 进入 Replan，不提前终止）
# ======================================================================

def test_abc_alternating_signatures_reset_and_replan(tmp_path, monkeypatch):
    agent = build_agent(tmp_path, monkeypatch)
    captured = install_state_capture(monkeypatch)
    tool_log = install_tool_counter(agent)

    # 调用链消耗：extract → plan → 参数×3 → replan → reasoning → final
    script = [EXTRACT_OK, plan_with_tool("calculator"),
              "{}",            # A：缺少必填参数 → 失败 1 → Retry 1
              BAD_ARGS,        # B：执行期异常 → 失败 2 → Retry 2
              "{}",            # A：再次缺参 → 失败 3 → Replan（不是终止）
              finish_plan(),   # Replan 响应
              "中间", "最终答案。"]
    result, output, states = run_and_capture(
        agent, captured, "交替失败", script)

    # 捕获断言：恰 1 个状态对象
    assert len(states) == 1
    state = states[0]

    # 恰 3 条工具失败记录，顺序 A/B/A；另有 Replan 成功后收尾的 1 条
    # reasoning 条目（源码确认 reasoning 步骤会追加），共 4 条、无其他条目。
    tool_entries = [
        e for e in state.execution_history if e.get("type") == "tool"
    ]
    assert len(tool_entries) == 3
    assert all(e["result"].get("success") is False for e in tool_entries)
    assert all(e["tool_name"] == "calculator" for e in tool_entries)
    assert len(state.execution_history) == 4
    reasoning_entries = [
        e for e in state.execution_history if e.get("type") == "reasoning"
    ]
    assert len(reasoning_entries) == 1
    assert reasoning_entries[0]["result"] == "中间"

    # 签名关系（关系式断言，不写死异常文本）：A == A、A != B
    sig_a1 = signature_of(tool_entries[0])
    sig_b = signature_of(tool_entries[1])
    sig_a2 = signature_of(tool_entries[2])
    assert sig_a1 == sig_a2
    assert sig_a1 != sig_b

    # 工具执行计数与历史一致。
    # M3 本地预检会拦截缺少必填参数的调用（本序列中两次 {} 均被
    # 拦截），只有实际通过预检的 BAD_ARGS 进入 registry.execute
    # 并被 spy 记录，因此计数为 1（第四轮实测确认）。
    assert len(tool_log) == 1
    assert all(not c["success"] for c in tool_log)

    # Retry 与 Replan 的实际次数（源码契约：Retry 1/2 各一次；Replan 一次）
    assert output.count("🔁 Retry 1/") == 1
    assert output.count("🔁 Retry 2/") == 1
    assert output.count("🔄 Replan 1/") == 1

    # 第三次 A 不被误判为连续重复失败：无提前终止；Replan 后正常收尾
    assert "检测到连续重复失败" not in output
    assert result["success"] is True


# ======================================================================
# 测试 2：Replan 后 A 再次出现（已登记 → 按终止条件结束）
# ======================================================================

def test_a_after_replan_terminates_when_repeated(tmp_path, monkeypatch):
    agent = build_agent(tmp_path, monkeypatch)
    captured = install_state_capture(monkeypatch)
    tool_log = install_tool_counter(agent)

    # 失败 3（A）→ Replan（A 入集合）→ 新计划仍用 calculator →
    # 参数 {} 再次失败（repeated 达阈值且 A 已用过 replan）→ 终止
    script = [EXTRACT_OK, plan_with_tool("calculator"),
              "{}",            # A（失败 1）→ Retry 1
              BAD_ARGS,        # B（失败 2）→ Retry 2
              "{}",            # A（失败 3）→ Replan（登记 A）
              plan_with_tool("calculator"),   # Replan 后同工具策略
              "{}",            # A（失败 4）→ 应终止
              finish_plan(), "中间", "最终答案。"]
    result, output, states = run_and_capture(
        agent, captured, "交替后重现", script)

    assert len(states) == 1
    state = states[0]

    tool_entries = [
        e for e in state.execution_history if e.get("type") == "tool"
    ]
    assert len(tool_entries) == 4                      # 四次失败全部记录
    assert all(e["result"].get("success") is False for e in tool_entries)
    assert len(state.execution_history) == 4           # 无 reasoning/额外条目

    # 签名链：A,B,A,A（第 4 次签名与第 3 次相同 → 同签名重复且已 replan）
    assert signature_of(tool_entries[0]) == signature_of(tool_entries[2])
    assert signature_of(tool_entries[2]) == signature_of(tool_entries[3])

    # 实际行为：Replan 一次；无第三次 Retry；终止恰一次；任务失败
    assert output.count("🔄 Replan 1/") == 1
    assert "🔁 Retry 3/" not in output
    assert output.count("检测到连续重复失败") == 1
    assert result["success"] is False

    # 调用计数与历史一致。
    # M3 本地预检会拦截缺少必填参数的调用（本序列中三次 {} 均被
    # 拦截），只有实际通过预检的 BAD_ARGS 进入 registry.execute
    # 并被 spy 记录，因此计数为 1（第四轮实测确认）。
    assert len(tool_log) == 1
    assert all(not c["success"] for c in tool_log)
