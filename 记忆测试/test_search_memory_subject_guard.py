# -*- coding: utf-8 -*-
"""search_memory 工具主体守卫（旁路最小修复）专属测试。

覆盖（第六部分第二步）：
  A. 第三方/假设/混合主体：工具被拒绝执行、零底层查询、执行历史无记忆内容、
     Reasoning 上下文无敏感记忆值（直接层 + 集成层）。
  B. 用户自身查询：主体门放行、工具原路径正常（主体门行为与检索结果分开断言）。
  C. 否定句：本轮不修复——守卫会放行，工具旁路守卫不能阻止语义否定；
     如实锁定基线（单独记录，不伪装为安全）。
  D. 其他工具：calculator 不受本次特判影响。

安全：临时库 + 虚构记忆 + 假客户端；不访问正式数据库；不接外部服务。
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

from agent import Agent          # noqa: E402
from memory import Memory        # noqa: E402
from task import AgentState      # noqa: E402

DUMMY = dict(api_key="dummy", base_url="https://example.invalid", model="dummy")

CORPUS = [
    ("major", "计算机科学与技术", "profile"),
    ("career_goal", "机器人工程师", "goal"),
    ("learning_goal", "学好检索与评估技术", "goal"),
    ("preference", "喜欢黑咖啡不加糖", "preference"),
    ("language_preference", "偏好用中文交流", "preference"),
    ("name", "李小明", "identity"),
]


class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)


class _Resp:
    def __init__(self, content):
        self.choices = [_Choice(content)]


class _Completions:
    def __init__(self, script):
        self._script = list(script)
        self.calls = []

    def create(self, model=None, messages=None, **kw):
        self.calls.append(messages)
        return _Resp(self._script.pop(0))


class _Chat:
    def __init__(self, script):
        self.completions = _Completions(script)


class FakeClient:
    def __init__(self, script):
        self.chat = _Chat(script)

    @property
    def calls(self):
        return self.chat.completions.calls


def build_agent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    agent = Agent(**DUMMY)
    agent.memory = Memory(db_path=str(tmp_path / "memory.db"))
    for key, value, category in CORPUS:
        agent.memory.save_memory(key=key, value=value, category=category,
                                 reason="工具守卫测试虚构语料", user_id="test-user")
    return agent


def attach_spies(agent):
    """同时监控：底层记忆三方法调用 + tool_registry.execute 调用。"""
    mem = {"find": 0, "search": 0, "by_cat": 0}
    real_find = agent.memory.find_memory
    real_search = agent.memory.search_long_term_memory
    real_by_cat = agent.memory.get_memories_by_category

    def spy_find(*a, **k):
        mem["find"] += 1
        return real_find(*a, **k)

    def spy_search(*a, **k):
        mem["search"] += 1
        return real_search(*a, **k)

    def spy_by_cat(*a, **k):
        mem["by_cat"] += 1
        return real_by_cat(*a, **k)

    agent.memory.find_memory = spy_find
    agent.memory.search_long_term_memory = spy_search
    agent.memory.get_memories_by_category = spy_by_cat

    reg = {"calls": []}
    real_exec = agent.tool_registry.execute

    def spy_exec(name, arguments):
        reg["calls"].append((name, arguments))
        return real_exec(name, arguments)

    agent.tool_registry.execute = spy_exec
    return mem, reg


def run_tool_plan(agent, user_message, tool_params, tool="search_memory"):
    """用假客户端构造含工具步骤的计划，完整跑 run() 并捕获提示词。"""
    script = [
        json.dumps({"should_remember": False, "memories": []}),
        json.dumps({"plan": [
            {"description": "检索", "type": "tool", "tool_name": tool},
            {"description": "回答", "type": "reasoning"},
        ]}),
        json.dumps(tool_params),
        "中间推理",
        "最终答案。",
    ]
    agent.client = FakeClient(script)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = agent.run(user_message, user_id="test-user")
    planner = agent.client.calls[1][0]["content"]
    transcript = agent.client.calls[3][0]["content"] + agent.client.calls[4][0]["content"]
    return result, planner, transcript


SENSITIVE = ("喜欢黑咖啡不加糖", "机器人工程师", "李小明")


# =====================================================================
# A. 直接层：第三方/假设/混合主体 → 拒绝执行，零查询，历史无记忆内容
# =====================================================================

REFUSAL_CASES = [
    "我室友说他喜欢喝手冲咖啡",
    "他的职业规划是怎样的",
    "假设我的专业是建筑学",
    "我想了解我朋友的职业规划",
    "我想知道其他同学学什么专业",
]


@pytest.mark.parametrize("q", REFUSAL_CASES)
def test_direct_refusal(q, tmp_path, monkeypatch, capsys):
    agent = build_agent(tmp_path, monkeypatch)
    mem, reg = attach_spies(agent)
    state = AgentState(q)
    state.task.start()

    refusal = agent.execute_tool(state, "search_memory",
                                 {"query": q})
    capsys.readouterr()

    # 1) 拒绝语义与返回约定（success 供调用方继续，refused 供追踪）
    assert refusal["success"] is True
    assert refusal["result"]["refused"] is True
    assert refusal["result"]["tool"] == "search_memory"
    # 2) 工具未实际执行、底层记忆查询零调用
    assert reg["calls"] == [], f"tool_registry 不应被调用：{reg['calls']}"
    assert mem == {"find": 0, "search": 0, "by_cat": 0}, mem
    # 3) 执行历史仅一条拒绝记录，且不含任何记忆内容
    assert len(state.execution_history) == 1
    entry = state.execution_history[0]
    dumped = json.dumps(entry, ensure_ascii=False)
    assert "memories" not in dumped
    for value in SENSITIVE:
        assert value not in dumped
    # 4) last_tool_result 同步为拒绝结果
    assert state.last_tool_result == refusal


# =====================================================================
# B. 直接层：用户自身查询 → 守卫放行，工具原路径正常
# =====================================================================

ALLOW_CASES = [
    "我是什么专业的",
    "你还记得我叫什么吗",
    "我以前说过自己喜欢什么吗",
]


@pytest.mark.parametrize("q", ALLOW_CASES)
def test_direct_allow(q, tmp_path, monkeypatch, capsys):
    agent = build_agent(tmp_path, monkeypatch)
    mem, reg = attach_spies(agent)

    # 主体门单独断言（放行）
    assert Agent._is_user_subject(q) is True

    state = AgentState(q)
    state.task.start()
    result = agent.execute_tool(state, "search_memory", {"query": q})
    capsys.readouterr()

    # 工具被实际执行（放行），且非拒绝结果；
    # 注意：主体门与底层词元检索是两件事——不假定一定能命中记忆，
    # 因此这里只断言"执行路径正常"，不断言 count > 0。
    assert reg["calls"] and reg["calls"][0][0] == "search_memory"
    assert not (isinstance(result.get("result"), dict)
                and result["result"].get("refused"))
    assert "count" in result["result"]
    assert len(state.execution_history) == 1
    assert state.execution_history[0]["result"]["success"] is True


# =====================================================================
# C. 否定句：本轮不修复——守卫放行、工具旁路守卫无法阻止语义否定
# =====================================================================

def test_negative_sentence_tool_refused(tmp_path, monkeypatch, capsys):
    """【安全基线（已更新，原 test_negative_sentence_guard_passes_through）】
    否定句"我没有职业目标"：主体门仍按第一人称规则放行（事实保留，
    拦截来自意图门而非主体门），统一意图门判定为 negative →
    工具被拒绝执行：零 registry 调用、零记忆查询、
    执行历史不含任何记忆记录或记忆值。
    原"缺口"已由语义否定最小补丁修复；真实模型行为仍未验证。"""
    agent = build_agent(tmp_path, monkeypatch)
    mem, reg = attach_spies(agent)
    state = AgentState("我没有职业目标")
    state.task.start()

    result = agent.execute_tool(state, "search_memory",
                                {"query": "我没有职业目标"})
    capsys.readouterr()

    # 事实保留：主体门本身仍放行该句（拦截来自意图门）
    assert Agent._is_user_subject("我没有职业目标") is True
    # 工具未被实际执行；返回明确体现 refused
    assert reg["calls"] == [], f"否定句下工具不应被执行：{reg['calls']}"
    assert result["result"]["refused"] is True
    # 零记忆查询 + 执行历史不含记忆记录/记忆值
    assert mem == {"find": 0, "search": 0, "by_cat": 0}, mem
    dumped = json.dumps(state.execution_history, ensure_ascii=False)
    assert "memories" not in dumped
    for value in ("机器人工程师", "学好检索与评估技术"):
        assert value not in dumped


def test_negative_sentence_integration_refused_baseline(tmp_path, monkeypatch,
                                                        capsys):
    """【安全基线（已更新，原 test_negative_sentence_integration_gap_baseline）】
    集成：否定句"我没有职业目标"——
    主检索路径不注入 career_goal（统一门拒绝、零记忆查询）；
    工具步骤同样被拒（拒绝记录可追踪、不泄露记忆值）。
    原"缺口"已由语义否定最小补丁修复；真实模型行为仍未验证。"""
    agent = build_agent(tmp_path, monkeypatch)
    mem, reg = attach_spies(agent)
    result, planner, transcript = run_tool_plan(
        agent, "我没有职业目标", {"query": "我没有职业目标"})
    capsys.readouterr()

    assert result["success"] is True
    # 主检索路径：Planner 不含 career_goal 记忆值，官方记忆查询零调用
    assert "- career_goal =" not in planner
    assert "机器人工程师" not in planner
    assert "（无相关长期记忆" in planner
    assert mem == {"find": 0, "search": 0, "by_cat": 0}, mem
    # 工具/旁路：拒绝记录可追踪，且不泄露记忆内容
    assert "被主体守卫拒绝" in transcript
    assert "机器人工程师" not in transcript
    assert "学好检索与评估技术" not in transcript


# =====================================================================
# C2. 否定句 × 命中型工具参数（第六部分第八步补强）
# 即使工具参数（"机器人"）本身能够命中临时记忆，系统仍必须依据
# 原始用户请求（"我没有职业目标。"，意图 negative）拒绝执行工具；
# 守卫使用的是原始用户请求，而不是模型生成的工具参数。
# =====================================================================

def test_negative_sentence_with_matching_tool_argument_refused(
        tmp_path, monkeypatch, capsys):
    """直接层：先证明"机器人"参数可命中，再验证仍被拒绝。"""
    agent = build_agent(tmp_path, monkeypatch)

    # 基线命中证明（无守卫的底层检索；先于 spy 安装，不计入守卫计数）
    baseline = agent.memory.search_long_term_memory("机器人", limit=5, user_id="test-user")
    baseline_keys = [m["key"] for m in baseline]
    assert "career_goal" in baseline_keys, (
        f"基线证明失败：'机器人'未能命中临时记忆 {baseline_keys}"
    )
    assert any(m["value"] == "机器人工程师" for m in baseline), (
        "基线证明失败：命中记录不含虚构记忆值"
    )

    mem, reg = attach_spies(agent)
    state = AgentState("我没有职业目标。")
    state.task.start()

    refusal = agent.execute_tool(state, "search_memory", {"query": "机器人"})
    capsys.readouterr()

    assert refusal["result"]["refused"] is True
    assert reg["calls"] == [], f"工具不得被执行：{reg['calls']}"
    assert mem == {"find": 0, "search": 0, "by_cat": 0}, mem
    dumped = json.dumps(state.execution_history, ensure_ascii=False)
    assert "career_goal" not in dumped
    assert "机器人工程师" not in dumped
    assert "memories" not in dumped


def test_negative_sentence_with_matching_argument_integration(
        tmp_path, monkeypatch, capsys):
    """集成层：FakeClient 的工具步骤参数使用 '机器人'；
    原始请求为否定句 → Planner / Reasoning 均不得收到该记忆值。"""
    agent = build_agent(tmp_path, monkeypatch)
    mem, reg = attach_spies(agent)
    result, planner, transcript = run_tool_plan(
        agent, "我没有职业目标。", {"query": "机器人"})
    capsys.readouterr()

    assert result["success"] is True
    assert mem == {"find": 0, "search": 0, "by_cat": 0}, mem
    assert reg["calls"] == [], f"工具不得被执行：{reg['calls']}"
    assert "机器人工程师" not in planner
    assert "机器人工程师" not in transcript
    assert "（无相关长期记忆" in planner
    assert "被主体守卫拒绝" in transcript


# =====================================================================
# 集成层：拒绝结果可达且不泄漏（第三方）／放行路径正常（自身）
# =====================================================================

def test_integration_third_party_refused_and_no_leak(tmp_path, monkeypatch,
                                                     capsys):
    agent = build_agent(tmp_path, monkeypatch)
    # 参数故意选择"会命中"的短语（审计证实过），验证即使命中也被拦
    result, planner, transcript = run_tool_plan(
        agent, "我室友说他喜欢喝手冲咖啡", {"query": "喜欢喝咖啡"})
    capsys.readouterr()

    assert result["success"] is True
    # Reasoning 可见"被拒绝"的追踪信息（拒绝结果进入执行历史）
    assert "被主体守卫拒绝" in transcript
    assert "refused" in transcript
    # 但任何敏感记忆值都不得出现
    for value in SENSITIVE:
        assert value not in transcript, f"泄漏：{value}"
        assert value not in planner


def test_integration_own_query_tool_executes(tmp_path, monkeypatch, capsys):
    agent = build_agent(tmp_path, monkeypatch)
    result, planner, transcript = run_tool_plan(
        agent, "我是什么专业的", {"query": "我是什么专业"})
    capsys.readouterr()

    assert result["success"] is True
    assert "被主体守卫拒绝" not in transcript, "自身查询不应被拒绝"


# =====================================================================
# D. 其他工具回归：calculator 不受特判影响
# =====================================================================

def test_other_tool_unaffected(tmp_path, monkeypatch, capsys):
    agent = build_agent(tmp_path, monkeypatch)
    mem, reg = attach_spies(agent)
    state = AgentState("123*456 等于几")
    state.task.start()

    result = agent.execute_tool(state, "calculator",
                                {"expression": "123*456"})
    capsys.readouterr()

    assert result["success"] is True
    assert result["result"] == "56088"
    assert reg["calls"] == [("calculator", {"expression": "123*456"})]
    assert state.execution_history[0]["result"]["result"] == "56088"
