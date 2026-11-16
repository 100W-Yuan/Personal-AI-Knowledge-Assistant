# -*- coding: utf-8 -*-
"""记忆访问意图判定（语义否定最小补丁 v3）专属测试。

冻结规格来源：第六部分第四步《最终冻结规格 v3》。
实施前基线（本文件相关断言依据，均为实施前独立实验实测）：
  - "我不是没有职业目标，只是还没确定具体方向。"（#8, neutral）
      实施前：候选 ['career_goal','learning_goal']，planner 注入 goal 类 >0；
  - "我想知道自己适合什么职业。" / "帮我查一下现在有哪些 AI 岗位。"（#12/#13, neutral）
      实施前：规则零命中，planner 为空段；
  - "#1/#10"（negative/deny 目标）实施前：被注入（本轮修复目标）。

层次声明：A 组为纯规则单元测试；B/C/D 为 FakeClient 模拟模型集成测试；
真实模型（对 deny/recall 的遵循行为、工具选择频率）未验证，不得描述为
真实模型验证。

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

GOAL_VALUES = ("机器人工程师", "学好检索与评估技术")


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


def build_agent(tmp_path, monkeypatch, corpus=None):
    monkeypatch.chdir(tmp_path)
    agent = Agent(**DUMMY)
    agent.memory = Memory(db_path=str(tmp_path / "memory.db"))
    for key, value, category in (corpus or CORPUS):
        agent.memory.save_memory(key=key, value=value, category=category,
                                 reason="意图守卫测试虚构语料", user_id="test-user")
    return agent


def attach_find_spy(agent):
    calls = {"find": [], "search": 0, "by_cat": 0}
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
    return calls


RUN_SCRIPT = [
    json.dumps({"should_remember": False, "memories": []}),
    json.dumps({"plan": [{"description": "回答", "type": "reasoning"}]}),
    "中间推理",
    "最终答案。",
]


def run_case(agent, q, script=None, tool=None, tool_params=None):
    if tool:
        script = [
            json.dumps({"should_remember": False, "memories": []}),
            json.dumps({"plan": [
                {"description": "检索", "type": "tool", "tool_name": tool},
                {"description": "回答", "type": "reasoning"},
            ]}),
            json.dumps(tool_params or {}),
            "中间推理",
            "最终答案。",
        ]
    agent.client = FakeClient(script or RUN_SCRIPT)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = agent.run(q, user_id="test-user")
    planner = agent.client.calls[1][0]["content"]
    transcript = agent.client.calls[-1][0]["content"] + agent.client.calls[-2][0]["content"]
    return result, planner, transcript


# =====================================================================
# A. 分类规则（纯单元测试；冻结规格 15 案例 + 边界）
# =====================================================================

FROZEN_15 = [
    ("我没有职业目标。", "negative"),
    ("我目前还没有明确的职业目标。", "negative"),
    ("我不记得自己说过什么职业目标。", "recall"),
    ("我没有职业目标，但以前说过想从事什么工作。", "recall"),
    ("我没有职业目标，帮我分析一下适合什么方向。", "negative"),
    ("我朋友没有职业目标。", "negative"),
    ("我想知道我以前说过什么职业目标。", "recall"),
    ("我不是没有职业目标，只是还没确定具体方向。", "neutral"),
    ("我没有明确的职业目标，但我想回忆以前的相关记录。", "recall"),
    ("请不要检索我的职业目标。", "deny"),
    ("请不要检索我的职业目标，即使我以前说过。", "deny"),
    ("我想知道自己适合什么职业。", "neutral"),
    ("帮我查一下现在有哪些 AI 岗位。", "neutral"),
    ("请不要把我以前的职业目标放进这次回答。", "deny"),
    ("不要忘记提醒我以前说过什么。", "recall"),
]


class TestClassifyIntent:

    @pytest.mark.parametrize("q,expected", FROZEN_15,
                             ids=[f"C{i+1:02d}" for i in range(15)])
    def test_frozen_15(self, q, expected):
        assert Agent._classify_memory_intent(q) == expected

    @pytest.mark.parametrize("q,expected,why", [
        ("我不是没有职业目标", "neutral", "双重否定豁免"),
        ("并非没有职业目标", "neutral", "双重否定豁免变体"),
        ("我难道没有职业目标吗？", "neutral", "反问豁免"),
        ("我是不是没有职业目标", "neutral", "反问豁免（是否）"),
        ("不要忘记提醒我以前说过什么。", "recall",
         "'不要忘记'不构成 deny，且含回忆信号"),
        ("我今天没有心情，想聊聊我的职业目标。", "neutral",
         "否定与事项词不同分句，不误拦"),
        ("我没有什么可说的，直接查我的职业目标吧。", "neutral",
         "'没什么'不在否定词表"),
        ("我没有职业目标但是", "negative", "同句近距离命中"),
        ("请勿查询我的相关信息", "deny", "请勿+查（动作绑定变体）"),
        ("不用检索我的个人资料了", "deny", "不用+检索"),
        ("帮我查一下现在有哪些 AI 岗位。", "neutral",
         "'查一下'宽泛表达不构成回忆"),
    ])
    def test_boundaries(self, q, expected, why):
        assert Agent._classify_memory_intent(q) == expected, why

    def test_long_distance_negation_not_flagged(self):
        """否定词与事项词距离明显超过 12 字 -> 不判 negative（保留既有行为）。"""
        q = "我从来没有在过去很多年的学习与工作经历里认真考虑过自己的职业目标"
        # "没有/从未/从来没" 到 "职业目标" 的间隔 23 字 > 12 -> 不触发；
        # 且无 recall 信号 -> neutral（沿用既有流程的裁决）
        assert Agent._classify_memory_intent(q) == "neutral"

    def test_access_gate_combination(self):
        assert Agent._memory_access_allowed("请不要检索我的职业目标。") is False
        assert Agent._memory_access_allowed("我没有职业目标。") is False
        assert Agent._memory_access_allowed("我想知道我以前说过什么职业目标。") is True
        assert Agent._memory_access_allowed(
            "我不是没有职业目标，只是还没确定具体方向。") is True
        # 主体门优先：第三方句即使无意图问题也必须拒绝
        assert Agent._memory_access_allowed("我朋友没有职业目标。") is False


# =====================================================================
# B. 主检索集成（FakeClient；检查 Planner/Reasoning 实际提示词）
# =====================================================================

REFUSED_CASES = [
    "我没有职业目标。",
    "我目前还没有明确的职业目标。",
    "我没有职业目标，帮我分析一下适合什么方向。",
    "请不要检索我的职业目标。",
    "请不要检索我的职业目标，即使我以前说过。",
    "请不要把我以前的职业目标放进这次回答。",
]


class TestMainPathIntegration:

    @pytest.mark.parametrize("q", REFUSED_CASES)
    def test_refused_zero_query_and_no_injection(self, q, tmp_path,
                                                 monkeypatch, capsys):
        agent = build_agent(tmp_path, monkeypatch)
        calls = attach_find_spy(agent)
        result, planner, transcript = run_case(agent, q)
        capsys.readouterr()

        assert result["success"] is True
        assert calls == {"find": [], "search": 0, "by_cat": 0}, (
            f"被拒绝时不得发生记忆查询：{calls}"
        )
        assert "（无相关长期记忆" in planner
        for value in GOAL_VALUES:
            assert value not in planner
            assert value not in transcript

    @pytest.mark.parametrize("q", [
        "我没有职业目标，但以前说过想从事什么工作。",
        "我想知道我以前说过什么职业目标。",
        "我没有明确的职业目标，但我想回忆以前的相关记录。",
    ])
    def test_recall_still_injects(self, q, tmp_path, monkeypatch, capsys):
        agent = build_agent(tmp_path, monkeypatch)
        result, planner, transcript = run_case(agent, q)
        capsys.readouterr()
        assert result["success"] is True
        assert "- career_goal = 机器人工程师" in planner

    def test_neutral_baseline_case8_unchanged(self, tmp_path, monkeypatch,
                                              capsys):
        """#8 neutral：分类器不额外干预。
        实施前基线（独立实验实测）：候选 ['career_goal','learning_goal']，
        planner 注入 goal 类。实施后必须保持一致。"""
        q = "我不是没有职业目标，只是还没确定具体方向。"
        agent = build_agent(tmp_path, monkeypatch)
        assert Agent._classify_memory_intent(q) == "neutral"

        cand = agent.retrieve_long_term_memories(q)
        cand_keys = [m["key"] for m in cand]
        assert cand_keys == ["career_goal", "learning_goal"], (
            f"与实施前基线不一致：{cand_keys}"
        )

        result, planner, transcript = run_case(agent, q)
        capsys.readouterr()
        assert result["success"] is True
        assert "- career_goal = 机器人工程师" in planner

    @pytest.mark.parametrize("q", [
        "我想知道自己适合什么职业。",
        "帮我查一下现在有哪些 AI 岗位。",
    ])
    def test_neutral_no_match_cases_unchanged(self, q, tmp_path, monkeypatch,
                                              capsys):
        """#12/#13 neutral：实施前=规则零命中、planner 空段。
        neutral 不额外改变流程，此行为保持。"""
        agent = build_agent(tmp_path, monkeypatch)
        assert Agent._classify_memory_intent(q) == "neutral"
        calls = attach_find_spy(agent)
        result, planner, transcript = run_case(agent, q)
        capsys.readouterr()
        assert calls["find"] == []
        assert "（无相关长期记忆" in planner


# =====================================================================
# C. 工具路径集成（与主检索共用同一门；拒绝零执行零泄漏）
# =====================================================================

class TestToolPathIntegration:

    @pytest.mark.parametrize("q", [
        "我没有职业目标。",
        "请不要检索我的职业目标。",
    ])
    def test_tool_refused_for_deny_and_negative(self, q, tmp_path,
                                                monkeypatch, capsys):
        agent = build_agent(tmp_path, monkeypatch)
        reg = {"calls": []}
        real_exec = agent.tool_registry.execute

        def spy_exec(name, arguments):
            reg["calls"].append((name, arguments))
            return real_exec(name, arguments)

        agent.tool_registry.execute = spy_exec
        state = AgentState(q)
        state.task.start()
        refusal = agent.execute_tool(state, "search_memory", {"query": q})
        capsys.readouterr()

        assert refusal["result"]["refused"] is True
        assert reg["calls"] == []
        dumped = json.dumps(state.execution_history, ensure_ascii=False)
        for value in GOAL_VALUES:
            assert value not in dumped

    def test_tool_allowed_for_recall(self, tmp_path, monkeypatch, capsys):
        q = "我想知道我以前说过什么职业目标。"
        agent = build_agent(tmp_path, monkeypatch)
        reg = {"calls": []}
        real_exec = agent.tool_registry.execute

        def spy_exec(name, arguments):
            reg["calls"].append((name, arguments))
            return real_exec(name, arguments)

        agent.tool_registry.execute = spy_exec
        state = AgentState(q)
        state.task.start()
        result = agent.execute_tool(state, "search_memory", {"query": q})
        capsys.readouterr()

        assert reg["calls"] and reg["calls"][0][0] == "search_memory"
        assert not result["result"].get("refused")

    def test_main_tool_path_consistency(self):
        """同一句在主检索与工具路径的门判定必须一致（共用同一函数）。"""
        for q in ["我没有职业目标。", "请不要检索我的职业目标。",
                  "我想知道我以前说过什么职业目标。",
                  "我朋友没有职业目标。"]:
            gate = Agent._memory_access_allowed(q)
            assert gate is (Agent._is_user_subject(q)
                            and Agent._classify_memory_intent(q)
                            not in ("deny", "negative")), q

    def test_tool_integration_no_leak(self, tmp_path, monkeypatch, capsys):
        agent = build_agent(tmp_path, monkeypatch)
        result, planner, transcript = run_case(
            agent, "请不要检索我的职业目标。",
            tool="search_memory", tool_params={"query": "职业目标"})
        capsys.readouterr()
        assert result["success"] is True
        assert "被主体守卫拒绝" in transcript  # 沿用现有拒绝记录文案
        for value in GOAL_VALUES:
            assert value not in transcript
            assert value not in planner


# =====================================================================
# E. 原有相关性筛选回归（允许检索后原流程不被绕过）
# =====================================================================

class TestRelevanceRegression:

    def test_min_score_not_bypassed(self, tmp_path, monkeypatch, capsys):
        """允许检索后，原 min_score/字符匹配逻辑照旧：
        '我是什么专业的' 只应返回 major，不得捞入无字符交集的记忆。"""
        agent = build_agent(tmp_path, monkeypatch)

        cand = agent.retrieve_long_term_memories("我是什么专业的")
        keys = [m["key"] for m in cand]
        capsys.readouterr()
        assert "major" in keys
        for unrelated in ("preference", "language_preference", "name",
                          "career_goal", "learning_goal"):
            assert unrelated not in keys, f"无关记忆被捞入：{unrelated}"

    def test_result_cap_not_bypassed(self, tmp_path, monkeypatch, capsys):
        """允许检索后，返回数量上限 3 照旧生效。"""
        corpus = CORPUS + [
            ("skill", "熟悉Python基础语法", "skill"),
            ("skill_b", "了解Git基础操作", "skill"),
            ("skill_c", "知道Docker基本概念", "skill"),
            ("skill_d", "接触过Linux命令", "skill"),
        ]
        agent = build_agent(tmp_path, monkeypatch, corpus=corpus)

        cand = agent.retrieve_long_term_memories("我擅长哪些方面")
        keys = [m["key"] for m in cand]
        capsys.readouterr()
        assert "skill" in keys
        assert len(keys) <= 3, f"上限被绕过：{keys}"
        assert len(keys) == len(set(keys)), f"去重被绕过：{keys}"
