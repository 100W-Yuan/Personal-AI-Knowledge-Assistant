# -*- coding: utf-8 -*-
"""RAG 7.9.2：未知工具无效重试修复专项测试。

验证目标（FakeClient，零真实模型/API/正式数据库）：
- 未知工具不触发参数生成模型调用（Planner 与 Replanner 两条来源）。
- 未知工具不进入普通 retry 分支。
- 未知工具错误写入 execution_history（含新增 retryable=False 字段）。
- replan 上限耗尽后正常终止（不无限循环）。
- 已注册工具照常执行；工具内部异常仍保留 retry；参数校验失败行为未变。
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

from agent import Agent      # noqa: E402
from memory import Memory    # noqa: E402

DUMMY = dict(api_key="dummy", base_url="https://example.invalid", model="dummy")

ARG_PROMPT_MARK = "请生成这个工具需要的 JSON 参数"
REPLANNER_MARK = "Replanner"


class ScriptedClient:
    """预置响应序列；耗尽后兜底返回收尾计划（防崩溃）。"""

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
    agent = Agent(**DUMMY)
    agent.memory.close()
    agent.memory = Memory(db_path=str(tmp_path / "memory.db"))
    return agent


def run_agent(agent, question, script):
    agent.client = ScriptedClient(script)
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


EXTRACT_OK = json.dumps({"should_remember": False, "memories": []})


def arg_prompt_count(client):
    return sum(
        1 for msgs in client.calls
        if ARG_PROMPT_MARK in json.dumps(msgs, ensure_ascii=False)
    )


def replanner_count(client):
    return sum(
        1 for msgs in client.calls
        if REPLANNER_MARK in json.dumps(msgs, ensure_ascii=False)
    )


# ----------------------------------------------------------------------
# 1. Planner 生成未知工具：参数生成调用 = 0
# ----------------------------------------------------------------------

def test_planner_unknown_tool_no_argument_generation(tmp_path, monkeypatch):
    agent = build_agent(tmp_path, monkeypatch)
    script = [EXTRACT_OK, plan_with_tool("not_a_tool"), finish_plan(),
              "中间", "最终答案。"]
    result, output = run_agent(agent, "帮我做件事", script)

    assert result["success"] is True
    assert arg_prompt_count(agent.client) == 0          # 核心：零参数生成
    assert replanner_count(agent.client) == 1           # 直接进入 replan
    assert "❌ 未知工具：not_a_tool" in output
    assert "🔁" not in output                           # 未进入 retry


# ----------------------------------------------------------------------
# 2. Replanner 生成未知工具：同样不触发参数生成
# ----------------------------------------------------------------------

def test_replanner_unknown_tool_no_argument_generation(tmp_path, monkeypatch):
    agent = build_agent(tmp_path, monkeypatch)
    script = [EXTRACT_OK,
              plan_with_tool("not_a_tool_1"),        # 初始计划未知工具
              plan_with_tool("not_a_tool_2"),        # replan 又是未知工具
              finish_plan(),
              "中间", "最终答案。"]
    result, output = run_agent(agent, "帮我做件事", script)

    assert result["success"] is True
    assert arg_prompt_count(agent.client) == 0          # 两条来源均零参数生成
    assert replanner_count(agent.client) == 2           # 两次受控 replan
    assert "not_a_tool_1" in output and "not_a_tool_2" in output


# ----------------------------------------------------------------------
# 3+4. 不进入普通 retry；错误写入 execution_history（进 Reasoning 提示词）
# ----------------------------------------------------------------------

def test_unknown_tool_error_recorded_and_no_retry(tmp_path, monkeypatch):
    agent = build_agent(tmp_path, monkeypatch)
    script = [EXTRACT_OK, plan_with_tool("not_a_tool"), finish_plan(),
              "中间", "最终答案。"]
    result, output = run_agent(agent, "帮我做件事", script)

    assert "🔁 Retry" not in output                     # 未走重试
    # execution_history 注入 → Reasoning 请求（system prompt 原文）可见
    # 错误与新增 retryable 字段（直接取 content，避免外层 json.dumps 的转义层）
    reasoning_content = agent.client.calls[3][0]["content"]
    assert "未知工具：not_a_tool" in reasoning_content
    assert '"retryable": false' in reasoning_content


# ----------------------------------------------------------------------
# 5. replan 上限耗尽 → 正常终止（不无限循环）
# ----------------------------------------------------------------------

def test_replan_limit_terminates(tmp_path, monkeypatch):
    agent = build_agent(tmp_path, monkeypatch)
    unknown = plan_with_tool("not_a_tool")
    script = [EXTRACT_OK, unknown] + [unknown] * 5      # 初始 + 5 次 replan 均未知
    result, output = run_agent(agent, "帮我做件事", script)

    assert result["success"] is False                   # 达到上限后终止
    assert "未知工具：not_a_tool" in result["answer"]
    assert arg_prompt_count(agent.client) == 0          # 全程零参数生成
    assert replanner_count(agent.client) == 5           # replan 恰好 5 次（上限）


# ----------------------------------------------------------------------
# 6. 已注册工具仍能成功执行（含参数生成 1 次）
# ----------------------------------------------------------------------

def test_registered_tool_still_works(tmp_path, monkeypatch):
    agent = build_agent(tmp_path, monkeypatch)
    script = [EXTRACT_OK, plan_with_tool("calculator"),
              json.dumps({"expression": "1+1"}), finish_plan(),
              "中间", "最终答案。"]
    result, output = run_agent(agent, "算一下 1+1", script)

    assert result["success"] is True
    assert arg_prompt_count(agent.client) == 1          # 正常工具照常参数生成
    assert "2" in output                                # 工具结果（1+1=2）进入执行链


# ----------------------------------------------------------------------
# 7. 工具内部异常仍保留 retry（test_failure 内置抛错工具）
# ----------------------------------------------------------------------

def test_internal_exception_still_retries(tmp_path, monkeypatch):
    """内部异常：首失败仍允许有限重试；同签名达上限后获得一次 Replan
    （RAG 7.9.10 契约：不再直接终止）。

    契约更新说明（RAG 7.9.10）：7.9.6 版断言"达上限即终止"，已在
    7.9.10 被"达上限 → 一次 replan 机会"取代；现按新语义断言：
    首失败重试保留 + 第二次不重试 + 进入 replan + 新计划收尾成功 +
    精确执行计数，而非删除断言。
    """
    agent = build_agent(tmp_path, monkeypatch)
    script = [EXTRACT_OK, plan_with_tool("test_failure"),
              "{}", "{}",                                # 参数预算（仅消耗 2 次）
              finish_plan(), "中间", "最终答案。"]
    result, output = run_agent(agent, "触发失败", script)

    # 1) 首次失败后允许既有策略规定的有限重试
    assert "🔁 Retry 1/" in output
    # 2) 同签名第二次失败后不再进行普通重试
    assert "🔁 Retry 2/" not in output
    # 3) 达上限后的处理：获得一次 replan（通用分支标记）
    assert "🔄 Replan 1/" in output
    assert "检测到连续重复失败" not in output
    # 4) 任务最终状态：新计划收尾成功
    assert result["success"] is True
    # 5) 精确执行计数：恰 2 次失败执行（无重复执行/无第三次尝试）
    assert output.count("❌ 工具执行失败") == 2
    # 6) 工具确实被调用（保留原证据断言）
    assert "test_failure" in output


# ----------------------------------------------------------------------
# 8. 参数校验失败行为未变（calculator 缺必填参数 → 仍走 retry）
# ----------------------------------------------------------------------

def test_invalid_arguments_still_retries(tmp_path, monkeypatch):
    agent = build_agent(tmp_path, monkeypatch)
    script = [EXTRACT_OK, plan_with_tool("calculator"),
              "{}",                                      # 校验失败（缺 expression）
              json.dumps({"expression": "2+2"}),         # retry 后重新生成合法参数
              finish_plan(), "中间", "最终答案。"]
    result, output = run_agent(agent, "算一下 2+2", script)

    assert "🔁 Retry 1/" in output                      # 参数失败仍可重试（未变）
    assert "工具参数校验失败" in output
    assert result["success"] is True
