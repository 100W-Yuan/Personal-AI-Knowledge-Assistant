# -*- coding: utf-8 -*-
"""N 组意图词表扩充（爱喝/有什么习惯/做什么营生/念的是）独立回归测试。

背景：第六部分第九步只读审计确认 N01–N03 为词表覆盖缺口；本文件验证
第六部分第十步新增的 4 个短语：能触发正确规则、能进入正常召回、不能
被第三方语境/否定语意/工具参数绕过、不改变安全门优先级。

安全：临时库 + 虚构记忆 + FakeClient；不访问正式数据库；不接外部服务。
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

import agent as agent_mod       # noqa: E402
from agent import Agent          # noqa: E402
from memory import Memory        # noqa: E402
from task import AgentState      # noqa: E402

DUMMY = dict(api_key="dummy", base_url="https://example.invalid", model="dummy")

CORPUS = [
    ("major", "计算机科学与技术", "profile"),
    ("degree", "本科在读", "profile"),
    ("career_goal", "机器人工程师", "goal"),
    ("learning_goal", "学好检索与评估技术", "goal"),
    ("preference", "喜欢黑咖啡不加糖", "preference"),
    ("language_preference", "偏好用中文交流", "preference"),
    ("name", "李小明", "identity"),
]

GOAL_VALUES = ("机器人工程师", "学好检索与评估技术")

RULES = agent_mod.LONG_TERM_MEMORY_INTENT_RULES
RULE_NAMES = ["name", "major", "goal", "preference", "skill", "project"]


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
                                 reason="N组扩充回归虚构语料", user_id="test-user")
    return agent


def attach_spies(agent):
    calls = {"find": [], "search": 0, "by_cat": 0}
    reg = {"calls": []}
    real_find = agent.memory.find_memory
    real_search = agent.memory.search_long_term_memory
    real_by_cat = agent.memory.get_memories_by_category

    def spy_find(key, *a, **k):
        calls["find"].append(key)
        return real_find(key, *a, **k)

    def spy_search(*a, **k):
        calls["search"] += 1
        return real_search(*a, **k)

    def spy_by_cat(*a, **k):
        calls["by_cat"] += 1
        return real_by_cat(*a, **k)

    agent.memory.find_memory = spy_find
    agent.memory.search_long_term_memory = spy_search
    agent.memory.get_memories_by_category = spy_by_cat

    real_exec = agent.tool_registry.execute

    def spy_exec(name, arguments):
        reg["calls"].append((name, arguments))
        return real_exec(name, arguments)

    agent.tool_registry.execute = spy_exec
    return calls, reg


RUN_SCRIPT = [
    json.dumps({"should_remember": False, "memories": []}),
    json.dumps({"plan": [{"description": "回答", "type": "reasoning"}]}),
    "中间推理",
    "最终答案。",
]


def run_case(agent, q, tool_params=None):
    script = RUN_SCRIPT
    if tool_params is not None:
        script = [
            json.dumps({"should_remember": False, "memories": []}),
            json.dumps({"plan": [
                {"description": "检索", "type": "tool",
                 "tool_name": "search_memory"},
                {"description": "回答", "type": "reasoning"},
            ]}),
            json.dumps(tool_params),
            "中间推理",
            "最终答案。",
        ]
    agent.client = FakeClient(script)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = agent.run(q, user_id="test-user")
    planner = agent.client.calls[1][0]["content"]
    transcript = agent.client.calls[-1][0]["content"] + agent.client.calls[-2][0]["content"]
    return result, planner, transcript


def rule_of(message):
    """用真实规则表复刻 retrieve 的匹配顺序（首条命中即停）。"""
    for idx, (keywords, _, _) in enumerate(RULES):
        matched = [k for k in keywords if k in message]
        if matched:
            return RULE_NAMES[idx], matched
    return None, []


# =====================================================================
# 0. 词表扩充的存在性与完整性（防手误）
# =====================================================================

def test_new_phrases_present_in_real_rules():
    by_name = dict(zip(RULE_NAMES, [r[0] for r in RULES]))
    assert "念的是" in by_name["major"]
    assert "做什么营生" in by_name["goal"]
    assert "爱喝" in by_name["preference"]
    assert "有什么习惯" in by_name["preference"]
    # 原有词未删除（抽样校验）
    for original in ("专业", "学的是", "职业目标", "想从事",
                     "我的习惯", "喜欢喝"):
        assert any(original in kws for kws in by_name.values())


# =====================================================================
# 1. N01/N02/N03 对应表达：触发正确规则 + 进入真实召回流程
# =====================================================================

NEW_COVERAGE = [
    ("N01", "我平时爱喝点什么来着", "preference",
     ["preference", "language_preference"]),
    ("N02", "我这个人在生活里有什么习惯", "preference",
     ["preference", "language_preference"]),
    ("N03", "我以后准备做什么营生", "goal",
     ["career_goal", "learning_goal"]),
]


@pytest.mark.parametrize("cid,q,rule,expect_keys", NEW_COVERAGE,
                         ids=[c[0] for c in NEW_COVERAGE])
def test_new_phrases_trigger_and_recall(cid, q, rule, expect_keys,
                                        tmp_path, monkeypatch, capsys):
    agent = build_agent(tmp_path, monkeypatch)

    # 规则命中检查（真实规则表）
    hit_rule, matched = rule_of(q)
    assert hit_rule == rule and matched, (
        f"[{cid}] 规则命中异常：{hit_rule} {matched}"
    )
    # 安全门放行（主体 + 意图）
    assert Agent._memory_access_allowed(q) is True

    # 真实召回流程：查询次数、find 键序、候选结果
    calls, _ = attach_spies(agent)
    cand = agent.retrieve_long_term_memories(q)
    capsys.readouterr()
    assert calls["find"] == expect_keys, f"[{cid}] find 调用：{calls['find']}"
    keys = [m["key"] for m in cand]
    assert keys == expect_keys, f"[{cid}] 候选：{keys}"

    # 集成：记忆实际进入 Planner
    result, planner, transcript = run_case(agent, q)
    capsys.readouterr()
    assert result["success"] is True
    if rule == "preference":
        assert "- preference =" in planner
    else:
        assert "- career_goal =" in planner


def test_n04_not_covered_by_this_patch(tmp_path, monkeypatch, capsys):
    """N04「我念的是什么方向」主链路行为锁定（注释于第十四步更新）。

    当前 N04 的测试契约要求 major（详见 test_intent_semantic_cross_layer.py
    中的案例定义：rel=["major"]）；degree 仍在允许范围（allow）内，
    但不是 N04 必须召回的键。
    本测试锁定的是："念的是"词表补丁后该句的真实链路行为——规则命中
    major、候选恰为 ["major"]（degree 无召回通道的机制事实，保留为记录）。
    该测试通过不代表 N04 仍失败；N04 的最终契约由
    test_intent_semantic_cross_layer.py 的案例定义决定。"""
    agent = build_agent(tmp_path, monkeypatch)
    hit_rule, matched = rule_of("我念的是什么方向")
    assert hit_rule == "major" and "念的是" in matched  # 规则已可命中

    cand = agent.retrieve_long_term_memories("我念的是什么方向")
    capsys.readouterr()
    keys = [m["key"] for m in cand]
    assert keys == ["major"]  # degree 不在候选（无召回通道的机制记录；
    # degree 属 allow 允许项，非契约要求）


# =====================================================================
# 2. 第三方语境：新短语不得成为用户记忆注入的新通道
# =====================================================================

THIRD_PARTY = [
    "我室友说他爱喝手冲咖啡",
    "表弟有什么习惯你知道吗",
    "他们公司准备做什么营生",
]


@pytest.mark.parametrize("q", THIRD_PARTY)
def test_third_party_with_new_phrases_still_refused(q, tmp_path,
                                                    monkeypatch, capsys):
    agent = build_agent(tmp_path, monkeypatch)
    calls, _ = attach_spies(agent)

    cand = agent.retrieve_long_term_memories(q)
    assert cand == []
    assert calls == {"find": [], "search": 0, "by_cat": 0}, calls

    result, planner, transcript = run_case(agent, q)
    capsys.readouterr()
    assert "（无相关长期记忆" in planner
    assert "- preference =" not in planner
    assert "- career_goal =" not in planner


# =====================================================================
# 3. 否定/拒绝优先于新短语的规则匹配
# =====================================================================

def test_negative_priority_over_new_phrase(tmp_path, monkeypatch, capsys):
    """「我没有什么习惯」命中新词"有什么习惯"，但意图 negative
    必须优先：零查询、零注入。"""
    q = "我没有什么习惯"
    assert rule_of(q)[0] == "preference"  # 新词确实命中规则
    assert Agent._classify_memory_intent(q) == "negative"
    assert Agent._memory_access_allowed(q) is False

    agent = build_agent(tmp_path, monkeypatch)
    calls, _ = attach_spies(agent)
    cand = agent.retrieve_long_term_memories(q)
    capsys.readouterr()
    assert cand == []
    assert calls == {"find": [], "search": 0, "by_cat": 0}


def test_deny_priority_over_new_phrase(tmp_path, monkeypatch, capsys):
    """「不要检索我的口味记录」：deny 优先，零查询、零注入。"""
    q = "不要检索我的口味记录"
    assert rule_of(q)[0] == "preference"
    assert Agent._classify_memory_intent(q) == "deny"
    assert Agent._memory_access_allowed(q) is False

    agent = build_agent(tmp_path, monkeypatch)
    calls, _ = attach_spies(agent)
    cand = agent.retrieve_long_term_memories(q)
    capsys.readouterr()
    assert cand == []
    assert calls == {"find": [], "search": 0, "by_cat": 0}


# =====================================================================
# 4. 工具参数不能绕过原始消息的安全门
# =====================================================================

def test_tool_argument_cannot_bypass_gate(tmp_path, monkeypatch, capsys):
    """原始消息为第三方句（含新词"爱喝"），工具参数使用会命中的短语：
    工具必须被拒绝——安全门看的是原始用户消息，不是工具参数。"""
    agent = build_agent(tmp_path, monkeypatch)
    calls, reg = attach_spies(agent)
    state = AgentState("我室友说他爱喝手冲咖啡")
    state.task.start()

    refusal = agent.execute_tool(state, "search_memory",
                                 {"query": "喜欢喝咖啡"})
    capsys.readouterr()

    assert refusal["result"]["refused"] is True
    assert reg["calls"] == []
    assert calls == {"find": [], "search": 0, "by_cat": 0}
    dumped = json.dumps(state.execution_history, ensure_ascii=False)
    assert "喜欢黑咖啡不加糖" not in dumped
    assert "memories" not in dumped


def test_tool_argument_bypass_integration(tmp_path, monkeypatch, capsys):
    """集成层：否定句 + 命中型工具参数 → 拒绝记录可追踪、零泄漏。"""
    agent = build_agent(tmp_path, monkeypatch)
    calls, reg = attach_spies(agent)
    result, planner, transcript = run_case(
        agent, "我没有什么习惯", {"query": "喜欢喝咖啡"})
    capsys.readouterr()

    assert result["success"] is True
    assert reg["calls"] == []
    assert calls == {"find": [], "search": 0, "by_cat": 0}
    assert "喜欢黑咖啡不加糖" not in planner
    assert "喜欢黑咖啡不加糖" not in transcript
    assert "被主体守卫拒绝" in transcript
