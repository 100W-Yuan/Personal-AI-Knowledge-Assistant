# -*- coding: utf-8 -*-
"""RAG 7.1：Agent 参数生成契约专项测试（决策 A + B，红测导向）。

背景（7.1 只读复查确认的生产现状，agent.py @ be2fd4e5）：
- 参数生成提示词（run() argument_prompt）不含工具参数 schema、
  不含上一次失败原因；
- 非法 JSON 被静默转换为 {}，伪装成"缺少必填参数"错误；
- 非对象 JSON 根类型（list/str/None）会原样进入 execute_tool →
  tool_registry.execute，由 registry 校验拒绝；
- 缺少必填参数时仍会调用 tool_registry.execute()。

本文件按"修复目标行为"断言（红测导向，不把当前错误行为写成正确预期）：
- T01/T02/T03/T04/T07/T08/T09 对应当前未实现的契约，预期失败；
- T05/T06/T10/T11/T12 锁定当前已具备、且预期修复后仍须保持的行为
  （T11/T12 的精确计数为"当前实现快照"，修复后需随契约更新，
  详见各测试内注释）。

全部用例使用 FakeClient / 脚本化响应 / 临时数据库 / 临时目录：
零真实模型、零云端 API、零正式 memory.db、零网络。
"""

import contextlib
import io
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

# 参数生成提示词的标记句：测试与生产契约的共同锚点。
# 任何提示词改动都必须保留该句（现有 4 个控制流测试依赖它分类调用）。
ARG_PROMPT_MARK = "请生成这个工具需要的 JSON 参数"
PLANNER_MARK = "的 Planner"
REPLANNER_MARK = "Replanner"
EXTRACT_MARK = "长期记忆提取器"
EXTRACT_OK = json.dumps({"should_remember": False, "memories": []})

ILLEGAL_JSON = "我无法生成参数"
NON_OBJECT_PAYLOADS = ["[1,2]", '"text"', "null"]

STOP_MARK = "🛑 停止 Agent，避免无限循环"
REPEATED_FAIL_MARK = "❌ 检测到连续重复失败"


# ----------------------------------------------------------------------
# Fake clients（沿用项目现有 ScriptedClient / ContentAwareClient 模式）
# ----------------------------------------------------------------------

class ScriptedClient:
    """线性预置响应序列；耗尽后兜底返回收尾计划（防崩溃）。"""

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
                    m = _M()
                    m.content = content

                    class _Ch:
                        message = m

                    class _R:
                        choices = [_Ch()]
                    return _R()
        return _C()


class ContentAwareClient:
    """按请求内容应答的无限供给 fake：参数生成恒返回 arg_payload，
    Planner/Replanner 恒返回 calculator 计划（用于长链/停止场景）。"""

    def __init__(self, arg_payload):
        self.arg_payload = arg_payload
        self.calls = []

    def _complete(self, messages):
        self.calls.append(messages)
        text = json.dumps(messages, ensure_ascii=False)
        if EXTRACT_MARK in text:
            return EXTRACT_OK
        if ARG_PROMPT_MARK in text:
            return self.arg_payload
        if REPLANNER_MARK in text:
            return plan_with_tool("calculator")
        if PLANNER_MARK in text:
            return plan_with_tool("calculator")
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
                    m = _M()
                    m.content = content

                    class _Ch:
                        message = m

                    class _R:
                        choices = [_Ch()]
                    return _R()
        return _C()


# ----------------------------------------------------------------------
# 构造与 instrumentation（沿用现有测试方式，零真实副作用）
# ----------------------------------------------------------------------

def plan_with_tool(tool_name):
    return json.dumps({"plan": [
        {"description": "执行工具", "type": "tool", "tool_name": tool_name},
        {"description": "回答", "type": "reasoning"},
    ]})


def finish_plan():
    return json.dumps({"plan": [{"description": "回答", "type": "reasoning"}]})


def build_agent(tmp_path, monkeypatch):
    """先切到临时目录再构造 Agent，杜绝触碰正式 memory.db。"""
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


def install_state_capture(monkeypatch):
    """捕获 run() 内部构造的 AgentState（不修改生产代码）。"""
    captured = []

    class SpyState(agent_module.AgentState):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            captured.append(self)

    monkeypatch.setattr(agent_module, "AgentState", SpyState)
    return captured


def instrument_tool_calls(agent):
    """spy 包装实例的 registry.execute，记录真实调用（tool, arguments, success）。"""
    log = []
    real_execute = agent.tool_registry.execute

    def spy(name, arguments):
        result = real_execute(name, arguments)
        log.append({"tool": name, "arguments": arguments,
                    "success": result.get("success")})
        return result

    agent.tool_registry.execute = spy
    return log


def arg_prompt_texts(client):
    """取出全部参数生成提示词（按标记句分类，与生产契约同锚点）。"""
    texts = []
    for msgs in client.calls:
        if not msgs:
            continue
        first = msgs[0]
        content = first.get("content", "") if isinstance(first, dict) else ""
        if ARG_PROMPT_MARK in content:
            texts.append(content)
    return texts


def count_mark(client, mark):
    return sum(
        1 for msgs in client.calls
        if mark in json.dumps(msgs, ensure_ascii=False)
    )


def first_tool_error(state):
    for entry in state.execution_history:
        if entry.get("type") == "tool":
            return entry["result"].get("error", "")
    return None


# ======================================================================
# A 组：参数 schema 与失败反馈
# ======================================================================

def test_t01_arg_prompt_contains_tool_schema(tmp_path, monkeypatch):
    """T01：参数生成提示词必须包含工具参数 schema（expression/required），
    且标记句必须原样保留。当前实现无 schema 注入——预期红测。"""
    agent = build_agent(tmp_path, monkeypatch)
    script = [EXTRACT_OK, plan_with_tool("calculator"),
              json.dumps({"expression": "2+2"}), finish_plan(),
              "中间", "最终答案。"]
    result, output = run_agent(agent, "算一下 2+2", ScriptedClient(script))

    assert result["success"] is True
    prompts = arg_prompt_texts(agent.client)
    assert len(prompts) == 1

    prompt = prompts[0]
    # 绿：标记句未被删除或改写（现有 4 个控制流测试的共同锚点）
    assert ARG_PROMPT_MARK in prompt, "参数生成标记句被删除或改写"
    # 红：schema 注入契约（当前实现不满足）
    assert "expression" in prompt, (
        "参数生成提示词未包含 calculator 的 expression 参数定义"
    )
    assert "required" in prompt, (
        "参数生成提示词未包含必填约束（required）"
    )


def test_t02_retry_prompt_contains_last_failure_reason(tmp_path, monkeypatch):
    """T02：Retry 后的参数生成提示词必须包含上一次失败原因。
    当前实现重试时复用同一提示词——预期红测。"""
    agent = build_agent(tmp_path, monkeypatch)
    script = [EXTRACT_OK, plan_with_tool("calculator"),
              "{}", json.dumps({"expression": "2+2"}), finish_plan(),
              "中间", "最终答案。"]
    result, output = run_agent(agent, "算一下 2+2", ScriptedClient(script))

    assert result["success"] is True
    assert "🔁 Retry 1/" in output

    prompts = arg_prompt_texts(agent.client)
    assert len(prompts) == 2, f"预期两次参数生成，实际 {len(prompts)}"

    retry_prompt = prompts[1]
    # 绿：标记句保留
    assert ARG_PROMPT_MARK in retry_prompt
    # 红：失败原因反馈契约（当前实现不满足）
    assert "缺少必填参数" in retry_prompt, (
        "Retry 后的参数生成提示词未包含上一次失败原因"
        "（缺少必填参数：expression）"
    )
    assert "expression" in retry_prompt, (
        "Retry 后的参数生成提示词未包含 expression 参数信息"
    )


def test_t03_replan_prompt_contains_schema_and_failure_reason(
        tmp_path, monkeypatch):
    """T03：Replan 后再次规划到 calculator 时，后续参数生成提示词
    仍必须包含 schema 与最近一次失败原因。当前实现两者皆无——预期红测。

    场景构造（线性脚本，确定性）：
    extract → plan(calc) → {}失败 → {}失败 → replan → plan(calc)
    → 合法参数成功。第三个参数提示词即 replan 之后的首次生成。
    """
    agent = build_agent(tmp_path, monkeypatch)
    script = [EXTRACT_OK, plan_with_tool("calculator"),
              "{}", "{}",
              plan_with_tool("calculator"),
              json.dumps({"expression": "2+2"}), finish_plan(),
              "中间", "最终答案。"]
    result, output = run_agent(agent, "算一下 2+2", ScriptedClient(script))

    assert result["success"] is True
    assert "🔄 Replan 1/" in output

    prompts = arg_prompt_texts(agent.client)
    assert len(prompts) == 3, f"预期三次参数生成，实际 {len(prompts)}"

    post_replan_prompt = prompts[2]
    # 红：schema + 失败原因双契约（当前实现均不满足）
    assert "expression" in post_replan_prompt and "required" in post_replan_prompt, (
        "Replan 后的参数生成提示词未包含工具 schema"
    )
    assert "缺少必填参数" in post_replan_prompt, (
        "Replan 后的参数生成提示词未包含最近一次失败原因"
    )


# ======================================================================
# B 组：本地必填参数预检
# ======================================================================

def test_t04_missing_required_param_blocks_registry_execute(
        tmp_path, monkeypatch):
    """T04：缺少必填参数时不得调用 tool_registry.execute()。
    当前实现会调用（由 registry 内部校验拒绝）——预期红测。

    同时锁定必须保持的行为：有界停止、execution_history 记录、
    last_tool_result 一致性。
    """
    captured = install_state_capture(monkeypatch)
    agent = build_agent(tmp_path, monkeypatch)
    spy = instrument_tool_calls(agent)
    client = ContentAwareClient(arg_payload="{}")
    result, output = run_agent(agent, "算一下", client)

    # 绿：有界停止（现有保护必须保留）
    assert result["success"] is False
    assert REPEATED_FAIL_MARK in output
    assert output.count(STOP_MARK) == 1

    state = captured[0]
    tool_entries = [e for e in state.execution_history
                    if e.get("type") == "tool"]
    # 绿：失败被完整记录（工具、原因、状态）
    assert len(tool_entries) >= 1
    assert all(e["tool_name"] == "calculator" for e in tool_entries)
    assert all(e["result"]["success"] is False for e in tool_entries)
    assert all("缺少必填参数：expression" in e["result"]["error"]
               for e in tool_entries)
    # 绿：last_tool_result 与历史记录一致
    assert state.last_tool_result == state.execution_history[-1]["result"]

    # 红：预检契约（当前实现不满足，缺参仍会到达 registry.execute）
    assert len(spy) == 0, (
        f"缺少必填参数时不得调用 registry.execute；实际调用 {len(spy)} 次"
    )


def test_t05_valid_arguments_still_execute(tmp_path, monkeypatch):
    """T05：合法参数仍走注册表执行路径并成功。"""
    agent = build_agent(tmp_path, monkeypatch)
    spy = instrument_tool_calls(agent)
    script = [EXTRACT_OK, plan_with_tool("calculator"),
              json.dumps({"expression": "2+2"}), finish_plan(),
              "中间", "最终答案。"]
    result, output = run_agent(agent, "算一下 2+2", ScriptedClient(script))

    assert result["success"] is True
    assert len(spy) == 1, f"合法参数应恰一次进入 registry.execute；实际 {len(spy)}"
    assert spy[0]["tool"] == "calculator"
    assert spy[0]["success"] is True
    assert "4" in output


def test_t06_no_required_param_tool_not_intercepted(tmp_path, monkeypatch):
    """T06：无必填参数的工具（test_failure）传 {} 时必须进入执行路径，
    不得被误判为参数缺失；其执行期异常路径保持可用。"""
    agent = build_agent(tmp_path, monkeypatch)
    spy = instrument_tool_calls(agent)
    script = [EXTRACT_OK, plan_with_tool("test_failure"),
              "{}", "{}", finish_plan(),
              "中间", "最终答案。"]
    result, output = run_agent(agent, "触发失败", ScriptedClient(script))

    assert result["success"] is True
    # 两次尝试都进入执行路径（预检不得误拦截合法空参数）
    assert len(spy) == 2, f"无必填参数工具传空对象应进入执行；实际 execute {len(spy)} 次"
    assert all(c["tool"] == "test_failure" for c in spy)
    # 失败来自执行期异常，而非参数校验
    assert "缺少必填参数" not in output
    assert "这是一次故意制造的工具错误" in output
    assert "🔁 Retry 1/" in output
    assert "🔄 Replan 1/" in output


# ======================================================================
# C 组：JSON 格式分类
# ======================================================================

def test_t07_legal_empty_object_vs_illegal_json_distinguished(
        tmp_path, monkeypatch):
    """T07：合法空对象 {} 与非法 JSON 必须得到不同错误类别/文本。
    当前实现把非法 JSON 静默转 {} 后伪装成缺参错误——预期红测。"""
    captured = install_state_capture(monkeypatch)

    agent_a = build_agent(tmp_path, monkeypatch)
    result_a, out_a = run_agent(
        agent_a, "算一下", ContentAwareClient(arg_payload="{}"))

    agent_b = build_agent(tmp_path, monkeypatch)
    result_b, out_b = run_agent(
        agent_b, "算一下", ContentAwareClient(arg_payload=ILLEGAL_JSON))

    # 两者都有界停止（保护机制与错误分类无关）
    assert result_a["success"] is False
    assert result_b["success"] is False

    err_empty = first_tool_error(captured[0])
    err_illegal = first_tool_error(captured[1])

    # 绿：{} 对 calculator 属于参数校验失败
    assert "缺少必填参数" in err_empty, f"实际：{err_empty}"
    # 红：非法 JSON 不得伪装成同一错误（当前实现两者文本完全相同）
    assert err_illegal != err_empty, (
        f"非法 JSON 与合法空对象得到了完全相同的错误文本：{err_illegal!r}"
    )
    # 红：非法 JSON 的错误必须可识别为 JSON/格式问题
    assert "JSON" in err_illegal, (
        f"非法 JSON 的错误文本未标明格式问题：{err_illegal!r}"
    )


def test_t08_non_object_json_root_not_passed_to_execute(tmp_path, monkeypatch):
    """T08：非对象 JSON 根类型（数组/字符串/null）不得作为工具参数
    进入 execute_tool / registry.execute。当前实现会原样传入——
    对三种根类型均预期红测。"""
    for payload in NON_OBJECT_PAYLOADS:
        agent = build_agent(tmp_path, monkeypatch)
        spy = instrument_tool_calls(agent)
        client = ContentAwareClient(arg_payload=payload)
        result, output = run_agent(agent, "算一下", client)

        # 绿：有界停止
        assert result["success"] is False, f"payload={payload} 未停止"
        assert REPEATED_FAIL_MARK in output, f"payload={payload} 无停止保护"
        # 红：非对象根不得到达 registry.execute
        assert len(spy) == 0, (
            f"payload={payload}：非对象 JSON 根不得进入 registry.execute；"
            f"实际 {len(spy)} 次"
        )
        assert all(isinstance(c["arguments"], dict) for c in spy), (
            f"payload={payload}：非对象参数被传入执行层"
        )


def test_t09_format_error_records_diagnostic_raw_text(tmp_path, monkeypatch):
    """T09：非法 JSON 的原文（或有界截断）必须进入失败记录供诊断。
    当前实现静默丢弃原文——预期红测。"""
    captured = install_state_capture(monkeypatch)
    agent = build_agent(tmp_path, monkeypatch)
    client = ContentAwareClient(arg_payload=ILLEGAL_JSON)
    result, output = run_agent(agent, "算一下", client)

    assert result["success"] is False

    state = captured[0]
    history_json = json.dumps(state.execution_history, ensure_ascii=False)
    # 红：原文或明确的格式类别必须可诊断（当前实现两者都没有）
    assert (ILLEGAL_JSON in history_json) or ("非法 JSON" in history_json), (
        "非法 JSON 的原文/格式类别未进入 execution_history，无法诊断"
    )


# ======================================================================
# D 组：恢复和停止保护
# ======================================================================

def test_t10_recovery_with_corrected_arguments(tmp_path, monkeypatch):
    """T10：缺参失败后返回正确参数能够恢复并执行成功。"""
    agent = build_agent(tmp_path, monkeypatch)
    script = [EXTRACT_OK, plan_with_tool("calculator"),
              "{}", json.dumps({"expression": "2+2"}), finish_plan(),
              "中间", "最终答案。"]
    result, output = run_agent(agent, "算一下 2+2", ScriptedClient(script))

    assert result["success"] is True
    assert "🔁 Retry 1/" in output
    assert "工具参数校验失败" in output
    assert "4" in output


def test_t11_persistent_illegal_json_bounded_stop(tmp_path, monkeypatch):
    """T11：fake 持续返回同一段非法 JSON 时，Agent 必须有界停止。

    当前实现快照（非法 JSON 被转 {} 后以缺参签名走完保护时序）：
    参数生成 3 次、Retry 1 次、Replan 1 次、registry.execute 3 次。
    实施 M2 后 execute 应变为 0（格式错误不进执行层），届时需随契约
    更新本条的 execute 计数；其余计数在稳定签名下保持不变。
    本条只验证现有代码的实际行为，不预先假定稳定签名已实现。
    """
    agent = build_agent(tmp_path, monkeypatch)
    spy = instrument_tool_calls(agent)
    client = ContentAwareClient(arg_payload=ILLEGAL_JSON)
    result, output = run_agent(agent, "算一下", client)

    n_arg = len(arg_prompt_texts(client))
    n_replan = count_mark(client, REPLANNER_MARK)
    n_retry = output.count("🔁 Retry ")
    n_stop = output.count(STOP_MARK)
    n_exec = len(spy)

    assert result["success"] is False
    assert REPEATED_FAIL_MARK in output
    assert n_stop == 1, f"停止标记应恰 1 次，实际 {n_stop}"

    # 当前实现快照（修复后仅 n_exec 预期变化）
    assert n_arg == 3, f"参数生成次数记录：实际 {n_arg}"
    assert n_replan == 1, f"Replan 次数记录：实际 {n_replan}"
    assert n_retry == 1, f"Retry 次数记录：实际 {n_retry}"
    # M2 契约：非法 JSON 在参数解析阶段被拦截，不进入执行层，
    # 因此 registry.execute 调用数为 0（第四轮实测确认）。
    assert n_exec == 0, (
        f"非法 JSON 应在解析阶段被拦截，不进入 registry.execute；实际 {n_exec}"
    )


def test_t12_persistent_missing_required_bounded_stop(tmp_path, monkeypatch):
    """T12：fake 持续返回 {} 时，Agent 必须有界停止。

    当前实现快照：参数生成 3 次、Retry 1 次、Replan 1 次、
    registry.execute 3 次（缺参由 registry 内部校验拒绝）。
    实施 M3 预检后 execute 应变为 0，届时需随契约更新本计数。
    """
    agent = build_agent(tmp_path, monkeypatch)
    spy = instrument_tool_calls(agent)
    client = ContentAwareClient(arg_payload="{}")
    result, output = run_agent(agent, "算一下", client)

    n_arg = len(arg_prompt_texts(client))
    n_replan = count_mark(client, REPLANNER_MARK)
    n_retry = output.count("🔁 Retry ")
    n_stop = output.count(STOP_MARK)
    n_exec = len(spy)

    assert result["success"] is False
    assert REPEATED_FAIL_MARK in output
    assert n_stop == 1, f"停止标记应恰 1 次，实际 {n_stop}"

    # 当前实现快照（修复后仅 n_exec 预期变化）
    assert n_arg == 3, f"参数生成次数记录：实际 {n_arg}"
    assert n_replan == 1, f"Replan 次数记录：实际 {n_replan}"
    assert n_retry == 1, f"Retry 次数记录：实际 {n_retry}"
    # M3 契约：缺少必填参数在 execute_tool() 的本地预检阶段被
    # 拦截，不进入 registry.execute，因此调用数为 0（第四轮实测确认）。
    assert n_exec == 0, (
        f"缺参应在 execute_tool 预检阶段被拦截，不进入 registry.execute；实际 {n_exec}"
    )
