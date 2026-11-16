# -*- coding: utf-8 -*-
"""RAG 7.9.7：重复失败提前终止后的控制流边界审计测试。

基于当前源码（agent.py @ e8b95bd8）推演出的预期行为：
1. 未知工具（预检）→ replan → 新计划中的新签名失败只计 1 次，
   不会被跨计划累计误停（"不同失败不合并"）。
2. 未知工具在连续多次 replan 中每次都在参数生成之前被拦截，
   参数生成调用恒为 0，最终在 replan 上限（5）后正确 failed。
3. 同签名第 2 次失败触发提前终止，**先于** Replan 判定——
   即失败类工具步骤不再进入 replan（修复后的行为事实，锁定）。
4. 终止分支只执行一次：无重复终止打印、无重复失败记录副作用。

说明（审计缺口，见报告）：A,B,A 签名交替在真实控制流中不可构造——
失败后的 should_retry 必然重入同一步骤（同一工具、同一错误文本），
使"同签名连续"成为必然；异签名只能跨 replan 出现，而该类失败在第
2 次即被提前终止拦截。因此交替签名仅作源码级逻辑保证，不设构造性
测试（不以宽泛断言冒充覆盖）。
"""

import io
import contextlib
import json
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


class AlwaysUnknownClient:
    """始终返回"未知工具"计划：用于验证每次 replan 都过预检。"""

    def __init__(self):
        self.calls = []

    def _complete(self, messages):
        self.calls.append(messages)
        text = json.dumps(messages, ensure_ascii=False)
        if "长期记忆提取器" in text:
            return EXTRACT_OK
        if ARG_PROMPT_MARK in text:
            raise AssertionError("未知工具不应触发参数生成请求")
        return json.dumps({"plan": [
            {"description": "执行未知工具", "type": "tool",
             "tool_name": "not_a_tool"}]})

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


# ----------------------------------------------------------------------
# 1. 未知工具 → replan → 新计划新签名失败：不误停（跨计划不累计误判）
# ----------------------------------------------------------------------

def test_replan_switch_new_signature_not_misstopped(tmp_path, monkeypatch):
    agent = build_agent(tmp_path, monkeypatch)
    script = [EXTRACT_OK,
              plan_with_tool("not_a_tool"),          # 预检拦截（不计签名）
              plan_with_tool("calculator"),          # replan 后新计划
              "{}",                                  # 校验失败（新签名，第 1 次）
              json.dumps({"expression": "8*8"}),     # Retry 1 → 成功
              finish_plan(), "中间", "最终答案。"]
    result, output = run_agent(agent, "混合场景", ScriptedClient(script))

    assert "🔁 Retry 1/" in output                 # 首次失败允许重试
    assert "检测到连续重复失败" not in output      # 不因跨计划相邻而误停
    assert result["success"] is True
    assert "64" in output                          # 重试成功（8*8）


# ----------------------------------------------------------------------
# 2. 未知工具在每次 replan 中都被预检拦截；参数生成恒为 0；上限后 failed
# ----------------------------------------------------------------------

def test_unknown_tool_intercepted_on_every_replan(tmp_path, monkeypatch):
    agent = build_agent(tmp_path, monkeypatch)
    client = AlwaysUnknownClient()
    result, output = run_agent(agent, "持续未知工具", client)

    intercepted = output.count("❌ 未知工具：not_a_tool")
    replans = output.count("🔄 准备 Replan")
    arg_prompts = sum(
        1 for msgs in client.calls
        if ARG_PROMPT_MARK in json.dumps(msgs, ensure_ascii=False)
    )

    assert intercepted == 6                         # 初始 1 次 + 5 次 replan 后各 1 次
    assert replans == 5                             # replan 上限恰好用满
    assert arg_prompts == 0                         # 任何一次都未触发参数生成
    assert "🔁" not in output                       # 未进入普通 Retry
    assert result["success"] is False
    assert "未知工具" in result["answer"]           # 上限后正确 failed 收尾


# ----------------------------------------------------------------------
# 3. 同签名第 2 次失败：提前终止先于 Replan（锁定修复后的行为事实）
# ----------------------------------------------------------------------

def test_same_signature_stops_before_replan(tmp_path, monkeypatch):
    """RAG 7.9.10 契约更新：同签名第 2 次失败不再提前终止，而是跳过
    Retry 进入一次 Replan（原 7.9.7 版本断言"无 replan"记录旧行为）。
    """
    agent = build_agent(tmp_path, monkeypatch)
    script = [EXTRACT_OK, plan_with_tool("test_failure"),
              "{}", "{}",
              finish_plan(), "中间", "最终答案。"]
    result, output = run_agent(agent, "持续失败", ScriptedClient(script))

    assert "🔁 Retry 1/" in output
    assert "🔁 Retry 2/" not in output
    assert output.count("❌ 工具执行失败") == 2      # 恰 2 次失败执行
    assert "🔄 Replan 1/" in output                 # 达上限 → 进入 replan
    assert "检测到连续重复失败" not in output
    assert result["success"] is True                # 新计划收尾成功


# ----------------------------------------------------------------------
# 4. 终止分支唯一执行：无双重终止、无重复失败记录副作用
# ----------------------------------------------------------------------

def test_single_termination_no_double_fail(tmp_path, monkeypatch):
    """RAG 7.9.10 契约更新：终止场景改为"replan 后同签名再次失败"；
    终止仍唯一执行、失败计数与执行一致。"""
    agent = build_agent(tmp_path, monkeypatch)
    script = [EXTRACT_OK, plan_with_tool("test_failure"),
              "{}", "{}",
              plan_with_tool("test_failure"),       # replan 返回同策略
              "{}",                                 # 第三次失败 → 终止
              finish_plan(), "中间", "最终答案。"]
    result, output = run_agent(agent, "不可恢复", ScriptedClient(script))

    assert output.count("❌ 检测到连续重复失败") == 1   # 终止仅一次
    assert output.count("🛑 停止 Agent，避免无限循环") == 1
    assert output.count("❌ 工具执行失败") == 3          # 与终止前的执行一致
    assert result["success"] is False
