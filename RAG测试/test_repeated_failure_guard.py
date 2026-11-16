# -*- coding: utf-8 -*-
"""RAG 7.9.4：重复失败提前终止机制专项测试（红测导向）。

背景：RAG 7.9.3 审计确认"重复失败兜底"位于 retry/replan 的 continue 之后，
"同签名≥2 提前停"的意图在还能 retry/replan 时结构性不可达。

本文件按**修复目标行为**断言（不把当前错误行为写成正确预期）：
- A：同签名第 2 次失败应提前停止（不再 retry、兜底直接触发）；
- C：首签名失败仍允许一次 retry（重试能力不被禁用）——防护断言；
- B：不同签名不得被合并为同一错误提前停——防护断言；
- D：未知工具继续走预检路径（7.9.2 不回归）；
- E：失败历史次数 / retry / replan / 最终状态之间的关系。

预期红测（供下轮修复依据）：A 与 E 的"提前停"相关断言。全部使用 FakeClient，
零真实 API / 零模型 / 零正式数据库。
"""

import io
import contextlib
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent import Agent      # noqa: E402
from memory import Memory    # noqa: E402

DUMMY = dict(api_key="dummy", base_url="https://example.invalid", model="dummy")

ARG_PROMPT_MARK = "请生成这个工具需要的 JSON 参数"
EXTRACT_OK = json.dumps({"should_remember": False, "memories": []})


class ScriptedClient:
    """预置响应序列；耗尽后兜底返回收尾计划。"""

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


class ContentAwareClient:
    """按请求内容应答的无限供给 fake（用于长链场景，无需精确响应序列）。"""

    def __init__(self, tool_name):
        self.tool_name = tool_name
        self.calls = []

    def _complete(self, messages):
        self.calls.append(messages)
        text = json.dumps(messages, ensure_ascii=False)
        if "长期记忆提取器" in text:
            return EXTRACT_OK
        if ARG_PROMPT_MARK in text:
            return "{}"
        if "Replanner" in text:
            return json.dumps({"plan": [
                {"description": "再试工具", "type": "tool",
                 "tool_name": self.tool_name}]})
        if "Planner" in text:
            return json.dumps({"plan": [
                {"description": "执行工具", "type": "tool",
                 "tool_name": self.tool_name}]})
        return "最终答案。"

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
    agent = Agent(**DUMMY)
    agent.memory.close()
    agent.memory = Memory(db_path=str(tmp_path / "memory.db"))
    return agent


def run_agent(agent, question, client):
    agent.client = client
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = agent.run(question, user_id="test-user")
    return result, buf.getvalue()


def plan_with_tool(tool_name):
    return json.dumps({"plan": [
        {"description": "执行工具", "type": "tool", "tool_name": tool_name},
        {"description": "回答", "type": "reasoning"},
    ]})


def finish_plan():
    return json.dumps({"plan": [{"description": "回答", "type": "reasoning"}]})


def arg_prompt_count(client):
    return sum(
        1 for msgs in client.calls
        if ARG_PROMPT_MARK in json.dumps(msgs, ensure_ascii=False)
    )


def instrument_tool_calls(agent):
    """直接调用计数（RAG 7.9.12）：包装实例 registry.execute，
    以真实调用记录替代日志文本推断失败次数。不修改生产代码。"""
    if not hasattr(agent, "_tool_call_log"):
        agent._tool_call_log = []
        real_execute = agent.tool_registry.execute

        def spy(name, arguments):
            result = real_execute(name, arguments)
            agent._tool_call_log.append(
                {"tool": name, "success": result.get("success")}
            )
            return result

        agent.tool_registry.execute = spy
    return agent._tool_call_log


def failed_call_count(agent):
    return sum(1 for c in agent._tool_call_log if not c["success"])


# ----------------------------------------------------------------------
# A + E：同签名第 2 次失败应提前停止（不再 retry、兜底立即触发）
# ----------------------------------------------------------------------

def test_same_signature_second_failure_stops_early(tmp_path, monkeypatch):
    """A + E（RAG 7.9.10 契约更新）：同签名第 2 次失败后不再 Retry，
    获得一次 Replan；终止仅在 replan 后同签名再败时发生。

    说明：7.9.5 版本用例断言"第 2 次失败即终止"；7.9.10 将"达上限"
    改为"先获一次 replan 机会"，故断言更新为 replan 分支证据（非删除）。
    """
    agent = build_agent(tmp_path, monkeypatch)
    log = instrument_tool_calls(agent)
    # test_failure 的 error 文本固定 → 多次尝试产生【相同签名】
    script = [EXTRACT_OK, plan_with_tool("test_failure"),
              "{}", "{}",                # 2 次尝试预算（新语义仅消耗 2 次）
              finish_plan(), "中间", "最终答案。"]
    result, output = run_agent(agent, "触发失败", ScriptedClient(script))

    assert "🔁 Retry 1/" in output, "首签名失败应允许一次重试"
    assert "🔁 Retry 2/" not in output, "同签名第 2 次失败不应继续普通重试"
    assert output.count("❌ 工具执行失败") == 2, (
        f"失败执行应恰为 2 次（当前={output.count('❌ 工具执行失败')}）"
    )
    assert "🔄 Replan 1/" in output, "达上限后应获得一次 replan 机会"
    assert "检测到连续重复失败" not in output
    assert result["success"] is True, "replan 后的新计划应完成收尾"
    assert len(log) == 2 and failed_call_count(agent) == 2   # [直接计数]


# ----------------------------------------------------------------------
# C：首签名失败仍允许一次重试（防止修复时无条件禁用重试）
# ----------------------------------------------------------------------

def test_first_failure_still_retries(tmp_path, monkeypatch):
    agent = build_agent(tmp_path, monkeypatch)
    script = [EXTRACT_OK, plan_with_tool("calculator"),
              "{}",                                  # 首次：缺必填参数 → 失败
              json.dumps({"expression": "6*7"}),     # 重试：修正参数 → 成功
              finish_plan(), "中间", "最终答案。"]
    result, output = run_agent(agent, "算一下 6*7", ScriptedClient(script))

    assert "🔁 Retry 1/" in output          # 有限重试能力保留
    assert "检测到连续重复失败" not in output
    assert result["success"] is True
    assert "42" in output


# ----------------------------------------------------------------------
# B：不同签名不被合并为同一错误（不误杀）
# ----------------------------------------------------------------------

def test_different_signature_not_merged(tmp_path, monkeypatch):
    agent = build_agent(tmp_path, monkeypatch)
    # 未知工具（预检路径，不进签名段）→ replan → calculator 校验失败（新签名）
    # → retry → 成功。全程不得出现"连续重复失败"提前停。
    script = [EXTRACT_OK,
              plan_with_tool("not_a_tool"),
              plan_with_tool("calculator"),
              "{}",
              json.dumps({"expression": "5+5"}),
              finish_plan(), "中间", "最终答案。"]
    result, output = run_agent(agent, "混合场景", ScriptedClient(script))

    assert "检测到连续重复失败" not in output, "异类失败不得被合并提前停"
    assert result["success"] is True


# ----------------------------------------------------------------------
# D：未知工具预检不回归（参数生成 0 次、无 retry）
# ----------------------------------------------------------------------

def test_unknown_tool_precheck_not_regressed(tmp_path, monkeypatch):
    agent = build_agent(tmp_path, monkeypatch)
    script = [EXTRACT_OK, plan_with_tool("not_a_tool"), finish_plan(),
              "中间", "最终答案。"]
    result, output = run_agent(agent, "触发未知工具", ScriptedClient(script))

    assert arg_prompt_count(agent.client) == 0
    assert "🔁" not in output
    assert result["success"] is True


# ----------------------------------------------------------------------
# E（长链）：兜底触发时机——修复目标：应在 replan 耗尽之前提前停止
# ----------------------------------------------------------------------

def test_guard_reachability_timing(tmp_path, monkeypatch):
    agent = build_agent(tmp_path, monkeypatch)
    log = instrument_tool_calls(agent)
    client = ContentAwareClient(tool_name="test_failure")   # 无限供给同签名失败
    result, output = run_agent(agent, "持续失败", client)

    replan_numbers = [int(m) for m in re.findall(r"🔄 Replan (\d+)/", output)]
    max_replan = max(replan_numbers) if replan_numbers else 0
    failed_execs = output.count("❌ 工具执行失败")

    assert "检测到连续重复失败" in output, (
        "兜底最终应可达（记录：当前需耗尽 retry/replan 才可达）"
    )
    assert max_replan < 5, (
        "同签名提前停应在 replan 耗尽（5）之前发生；"
        f"当前记录 max_replan={max_replan}、失败执行={failed_execs}"
    )
    # [直接计数] 失败 2 次 → replan → 第 3 次失败终止（共 3 次执行，全部失败）
    assert len(log) == 3 and failed_call_count(agent) == 3
