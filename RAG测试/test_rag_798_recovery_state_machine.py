# -*- coding: utf-8 -*-
"""RAG 7.9.8：失败恢复状态机专项验收测试（方案 B——区分重试与重新规划）。

说明：本文件按**目标契约**断言（任务四），不把当前实现的行为直接写成正确
预期。预期存在红测（B1：重试耗尽后应有机会重新规划——当前被提前终止抢先）。

可观测手段（每个断言的验证类型已在注释中标注）：
- [行为] 真实可观测：工具实际执行次数（"❌ 工具执行失败"计数）、Retry/Replan
  打印、run() 返回的 success/answer；
- [mock] 替身调用计数：模型请求中按 prompt 特征分类计数（参数生成/Planner/
  Replanner）；
- [间接] 内部状态（execution_history/计数器）无法从 run() 直接读出，
  通过执行与调用计数间接推断——已在文中标明，"间接推断"不作为直接验证宣称。

已知不可构造场景（不计入通过数，列为未验证）：
- B3 同签名"重新规划最多一次"：失败类 replan 当前不可达（见 B1 红测），
  无执行通道可构造；未知工具 replan 不进入失败签名段。
- D5 不同错误交替：7.9.7 已证真实流程不可构造（retry 强制同签名连续）。
"""

import io
import contextlib
import json
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent import Agent      # noqa: E402
from memory import Memory    # noqa: E402

DUMMY = dict(api_key="dummy", base_url="https://example.invalid", model="dummy")

ARG_PROMPT_MARK = "请生成这个工具需要的 JSON 参数"
PLANNER_MARK = "的 Planner"
REPLANNER_MARK = "Replanner"
EXTRACT_OK = json.dumps({"should_remember": False, "memories": []})

FAIL_MARK = "❌ 工具执行失败"


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
    """始终返回未知工具计划（用于 replan 上限场景）。"""

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


def plan_steps(steps):
    return json.dumps({"plan": steps})


def plan_with_tool(tool_name):
    return plan_steps([
        {"description": "执行工具", "type": "tool", "tool_name": tool_name},
        {"description": "回答", "type": "reasoning"},
    ])


def finish_plan():
    return plan_steps([{"description": "回答", "type": "reasoning"}])


def arg_prompt_calls(client):
    return sum(
        1 for msgs in client.calls
        if ARG_PROMPT_MARK in json.dumps(msgs, ensure_ascii=False)
    )


def instrument_tool_calls(agent):
    """直接调用计数（RAG 7.9.12）：包装实例的 registry.execute，
    记录每次工具执行的 (tool, success)——用于以真实调用证据替代
    仅凭日志文本推断失败次数。不修改生产代码（运行时实例替换）。"""
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
    return sum(
        1 for c in agent._tool_call_log if not c["success"]
    )


# ======================================================================
# A 组：重试上限与成功恢复
# ======================================================================

def test_a1_retry_success_completes_task(tmp_path, monkeypatch):
    """A1：第一次失败、重试成功 → 任务成功结束。[行为]"""
    agent = build_agent(tmp_path, monkeypatch)
    script = [EXTRACT_OK, plan_with_tool("calculator"),
              "{}",                                  # 校验失败
              json.dumps({"expression": "9*9"}),     # 重试成功
              finish_plan(), "中间", "最终答案。"]
    result, output = run_agent(agent, "算 9*9", ScriptedClient(script))

    assert output.count(FAIL_MARK) == 1              # [行为] 恰一次失败执行
    assert "🔁 Retry 1/" in output                   # [行为] 允许一次重试
    assert result["success"] is True
    assert "81" in output
    assert arg_prompt_calls(agent.client) == 2       # [mock] 两次参数生成


def test_a2_same_strategy_fails_at_most_twice(tmp_path, monkeypatch):
    """A2/A3：同策略连续失败最多执行两次；达到上限后获得一次 replan
    （RAG 7.9.10 契约：不再直接终止）。[行为]"""
    agent = build_agent(tmp_path, monkeypatch)
    log = instrument_tool_calls(agent)
    script = [EXTRACT_OK, plan_with_tool("test_failure"),
              "{}", "{}", finish_plan(), "中间", "最终答案。"]
    result, output = run_agent(agent, "持续失败", ScriptedClient(script))

    assert output.count(FAIL_MARK) == 2              # [行为] 恰两次执行（上限）
    assert "🔁 Retry 1/" in output
    assert "🔁 Retry 2/" not in output               # 第二次失败后不再重试
    assert "🔄 Replan 1/" in output                  # 达到上限后进入 replan
    assert result["success"] is True                 # replan 后按新计划收尾
    assert len(log) == 2 and failed_call_count(agent) == 2  # [直接计数]


def test_a4_previous_success_resets_failure_count(tmp_path, monkeypatch):
    """A4：前一步成功后，不应污染后续独立步骤的失败计数。[行为+间接]"""
    agent = build_agent(tmp_path, monkeypatch)
    two_step_plan = plan_steps([
        {"description": "第一步：计算", "type": "tool", "tool_name": "calculator"},
        {"description": "第二步：故障工具", "type": "tool",
         "tool_name": "test_failure"},
        {"description": "回答", "type": "reasoning"},
    ])
    script = [EXTRACT_OK, two_step_plan,
              "{}",                                  # step1 失败
              json.dumps({"expression": "3+3"}),     # step1 重试成功（应重置计数）
              "{}", "{}"]                            # step2 失败两次 → replan 接管
    result, output = run_agent(agent, "两步任务", ScriptedClient(script))

    # [行为] step1 失败 1 次 + step2 失败 2 次 = 3；若计数被污染，step2 会
    # 首败即停 → 总失败数将只有 2、Retry 1 只会出现 1 次。
    assert output.count(FAIL_MARK) == 3
    assert output.count("🔁 Retry 1/") == 2           # 两个步骤各自获得重试
    assert "🔁 Retry 2/" not in output
    assert "🔄 Replan 1/" in output                  # 达上限后获得一次 replan
    assert result["success"] is True


# ======================================================================
# B 组：重新规划上限与失败签名
# ======================================================================

def test_b1_replan_opportunity_after_retry_exhausted(tmp_path, monkeypatch):
    """B1【已达成（RAG 7.9.10 起）】：同签名失败达到重试上限后获得一次
    Replan 机会（而非被提前终止阻断），新计划完成后任务成功。

    验证方式（RAG 7.9.12 补强）：直接调用计数（registry.execute 包装）
    证明失败执行恰 2 次、均为 test_failure；结合 Replan 打印与最终结果。
    """
    agent = build_agent(tmp_path, monkeypatch)
    log = instrument_tool_calls(agent)
    script = [EXTRACT_OK, plan_with_tool("test_failure"),
              "{}", "{}",                           # 两次失败（重试上限）
              finish_plan(), "中间", "最终答案。"]   # 重新规划后的收尾计划
    result, output = run_agent(agent, "故障后换策略", ScriptedClient(script))

    assert output.count(FAIL_MARK) == 2              # [行为] 仍只执行两次
    assert "🔄 Replan 1/" in output                  # 目标：获得 replan 机会（通用分支标记）
    assert result["success"] is True                 # 目标：换策略后成功
    # [直接计数] 恰 2 次工具执行、全部失败、均为 test_failure
    assert len(log) == 2
    assert failed_call_count(agent) == 2
    assert {c["tool"] for c in log} == {"test_failure"}


def test_b2_replan_new_strategy_completes_task(tmp_path, monkeypatch):
    """B2：重新规划生成不同工具后，新策略能成功完成任务。[行为+mock]"""
    agent = build_agent(tmp_path, monkeypatch)
    script = [EXTRACT_OK,
              plan_with_tool("not_a_tool"),          # 未知工具 → 预检 replan
              plan_with_tool("calculator"),          # 新策略
              json.dumps({"expression": "7*7"}),
              finish_plan(), "中间", "最终答案。"]
    result, output = run_agent(agent, "换策略", ScriptedClient(script))

    assert "❌ 未知工具：not_a_tool" in output
    assert result["success"] is True
    assert "49" in output
    assert arg_prompt_calls(agent.client) == 1       # [mock] 仅新策略生成参数


def test_b56_replan_limit_stops_replanner(tmp_path, monkeypatch):
    """B5/B6：全局 replan 上限 5 次；达到后不得继续调用（Replanner 请求恰 5）。"""
    agent = build_agent(tmp_path, monkeypatch)
    log = instrument_tool_calls(agent)
    client = AlwaysUnknownClient()
    result, output = run_agent(agent, "持续未知工具", client)

    text_calls = [json.dumps(c, ensure_ascii=False) for c in client.calls]
    planner_calls = sum(1 for t in text_calls if PLANNER_MARK in t)
    replanner_calls = sum(1 for t in text_calls if REPLANNER_MARK in t)

    assert output.count("❌ 未知工具：not_a_tool") == 6     # [行为] 初始+5
    assert output.count("🔄 准备 Replan") == 5
    assert planner_calls == 1                              # [mock] 初始规划仅一次
    assert replanner_calls == 5                            # [mock] 上限后不再调用
    assert result["success"] is False
    assert len(log) == 0                                   # [直接计数] 预检拦截，零工具执行


# ======================================================================
# C 组：失败终止与状态一致性
# ======================================================================

def test_c1_c2_c3_termination_single_and_consistent(tmp_path, monkeypatch):
    """C1-C3（RAG 7.9.10 契约）：同签名 replan 后再次失败 → 终止；
    不可恢复任务以 failed 结束；终止只执行一次；失败记录与执行一致。"""
    agent = build_agent(tmp_path, monkeypatch)
    log = instrument_tool_calls(agent)
    script = [EXTRACT_OK, plan_with_tool("test_failure"),
              "{}", "{}",                           # 两次失败 → replan
              plan_with_tool("test_failure"),       # replan 后同签名策略
              "{}",                                 # 第三次失败 → 应终止
              finish_plan(), "中间", "最终答案。"]
    result, output = run_agent(agent, "不可恢复", ScriptedClient(script))

    assert result["success"] is False                       # C1 [行为]
    assert output.count("🔄 Replan 1/") == 1                 # replan 已使用一次
    assert output.count(FAIL_MARK) == 3                      # 失败 3 次后终止
    assert output.count("🔁 Retry 1/") == 1                  # replan 后未再 retry
    assert output.count("检测到连续重复失败") == 1            # C2 [行为] 终止唯一
    assert output.count("🛑 停止 Agent，避免无限循环") == 1
    assert "🧠 执行 Reasoning" not in output                 # 终止后未继续
    assert len(log) == 3 and failed_call_count(agent) == 3   # [直接计数]


# ======================================================================
# 独立任务隔离（RAG 7.9.10 新契约：replanned 集合不得跨任务共享）
# ======================================================================

def test_independent_tasks_do_not_share_replanned_signatures(
        tmp_path, monkeypatch):
    """两个独立 run 复用同一 Agent：第二个任务同签名仍应获得 replan
    （集合挂在 state.control 上，每任务新建）。[行为]"""
    agent = build_agent(tmp_path, monkeypatch)
    log = instrument_tool_calls(agent)

    def one_run(question):
        script = [EXTRACT_OK, plan_with_tool("test_failure"),
                  "{}", "{}", finish_plan(), "中间", "最终答案。"]
        return run_agent(agent, question, ScriptedClient(script))

    r1, o1 = one_run("任务一")
    after_first = len(log)
    r2, o2 = one_run("任务二")
    after_second = len(log)

    for r, o in ((r1, o1), (r2, o2)):
        assert r["success"] is True
        assert "🔄 Replan 1/" in o          # 每个任务各自获得一次 replan
        assert "检测到连续重复失败" not in o
    # [直接计数] 每个任务各 2 次失败执行（互不继承计数/集合）
    assert after_first == 2
    assert after_second - after_first == 2


# ======================================================================
# D 组：不同失败类型的一致性（路径区分）
# ======================================================================

def test_d_failure_paths_are_distinct(tmp_path, monkeypatch):
    """D1-D3：未注册工具在参数生成前被拒（零参数生成、零 retry）；
    参数校验失败与内部异常保留有限重试。[行为+mock]"""

    # 每个子场景使用独立临时目录
    def fresh_agent():
        td = Path(tempfile.mkdtemp(prefix="d798_"))
        os.chdir(td)
        a = Agent(**DUMMY)
        a.memory.close()
        a.memory = Memory(db_path=str(td / "memory.db"))
        return a

    a1 = fresh_agent()
    log1 = instrument_tool_calls(a1)
    s1 = [EXTRACT_OK, plan_with_tool("not_a_tool"), finish_plan(), "中间", "最终。"]
    r1, o1 = run_agent(a1, "未知工具", ScriptedClient(s1))
    assert arg_prompt_calls(a1.client) == 0          # [mock] D1 零参数生成
    assert len(log1) == 0                            # [直接计数] 预检拦截，零执行
    assert "🔁" not in o1                            # [行为] D1 零 retry
    assert r1["success"] is True

    # D2 参数校验失败 → 允许重试
    a2 = fresh_agent()
    s2 = [EXTRACT_OK, plan_with_tool("calculator"),
          "{}", json.dumps({"expression": "2*2"}),
          finish_plan(), "中间", "最终。"]
    r2, o2 = run_agent(a2, "参数失败", ScriptedClient(s2))
    assert "🔁 Retry 1/" in o2                       # [行为] D2 重试保留
    assert "工具参数校验失败" in o2
    assert r2["success"] is True

    # D3 内部异常 → 允许首失败重试；第二次失败后获得一次 replan（7.9.10 契约）
    a3 = fresh_agent()
    log3 = instrument_tool_calls(a3)
    s3 = [EXTRACT_OK, plan_with_tool("test_failure"),
          "{}", "{}", finish_plan(), "中间", "最终。"]
    r3, o3 = run_agent(a3, "内部异常", ScriptedClient(s3))
    assert "🔁 Retry 1/" in o3                       # [行为] D3 重试保留
    assert o3.count(FAIL_MARK) == 2                  # [行为] 执行上限
    assert "🔄 Replan 1/" in o3                      # [行为] 上限后进入 replan
    assert r3["success"] is True                     # 新计划收尾成功
    assert len(log3) == 2 and failed_call_count(a3) == 2  # [直接计数]
