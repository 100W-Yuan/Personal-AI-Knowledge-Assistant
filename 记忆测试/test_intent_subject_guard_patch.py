# -*- coding: utf-8 -*-
"""主体守卫最小补丁专属测试（第四阶段第四步第五部分）。

覆盖：
  1. _is_user_subject 边界扫描（“其他”豁免、集体主语、假设、引用、无我）。
  2. 16 条验收案例：拦截案例须“零记忆查询”；放行案例须正常命中。
  3. 孤立关键词“目标”删除后，明确表达（职业目标等）仍命中。
  4. 上下文层抽查（真实 run() + 假客户端捕获提示词）。
  5. 已知缺口/损失的基线锁定（否定句、混合句误拦）——如实记录，不伪称已解决。

安全：临时库 + 虚构记忆 + 假客户端；不读正式库；不接任何外部服务。
"""

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent import Agent          # noqa: E402
from memory import Memory        # noqa: E402

DUMMY = dict(api_key="dummy", base_url="https://example.invalid", model="dummy")

CORPUS = [
    ("major", "计算机科学与技术", "profile"),
    ("degree", "本科在读", "profile"),
    ("school", "华东电子科技大学", "profile"),
    ("career_goal", "机器人工程师", "goal"),
    ("learning_goal", "学好检索与评估技术", "goal"),
    ("side_goal", "帮朋友规划机器人比赛", "goal"),
    ("preference", "喜欢黑咖啡不加糖", "preference"),
    ("language_preference", "偏好用中文交流", "preference"),
    ("project", "社区图书角项目", "project"),
    ("skill", "熟悉Python基础语法", "skill"),
    ("location", "现居上海", "profile"),
    ("name", "李小明", "identity"),
]

RUN_SCRIPT = [
    json.dumps({"should_remember": False, "memories": []}),
    json.dumps({"plan": [{"description": "直接回答", "type": "reasoning"}]}),
    "中间推理",
    "最终答案。",
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
                                 reason="主体守卫测试虚构语料", user_id="test-user")
    return agent


def attach_spy(agent):
    """记录三个记忆查询方法的调用次数（拦截验证的核心证据）。"""
    counts = {"find": 0, "search": 0, "by_cat": 0, "find_keys": []}
    real_find = agent.memory.find_memory
    real_search = agent.memory.search_long_term_memory
    real_by_cat = agent.memory.get_memories_by_category

    def spy_find(key, *a, **k):
        counts["find"] += 1
        counts["find_keys"].append(key)
        return real_find(key, *a, **k)

    def spy_search(*a, **k):
        counts["search"] += 1
        return real_search(*a, **k)

    def spy_by_cat(*a, **k):
        counts["by_cat"] += 1
        return real_by_cat(*a, **k)

    agent.memory.find_memory = spy_find
    agent.memory.search_long_term_memory = spy_search
    agent.memory.get_memories_by_category = spy_by_cat
    return counts


def total_calls(counts):
    return counts["find"] + counts["search"] + counts["by_cat"]


# =====================================================================
# 1. 守卫函数边界扫描（无需数据库，classmethod 直接调用）
# =====================================================================

class TestGuardFunction:

    @pytest.mark.parametrize("message,expected,why", [
        ("我关心其他学科", True, "其他中的'他'必须豁免，且有第一人称"),
        ("我知道其他情况", True, "同上，纯豁免验证"),
        ("他的职业规划是怎样的", False, "第三人称代词"),
        ("她的方案如何", False, "她；且无第一人称"),
        ("我们团队的目标", False, "集体主语"),
        ("我司今年的工作目标", False, "我司——集体主语，归属不确定"),
        ("这个项目的目标用户是谁", False, "无第一人称"),
        ("假设我的专业是建筑学", False, "假设词（即使含'我的'也拦截）"),
        ("如果我喜欢喝气泡水", False, "假设词"),
        ("帮我分析一下那句“我喜欢你”", False, "引用标记'那句'"),
        ("我是什么专业的", True, "明确自身查询"),
        ("帮我推荐适合我的饮品", True, "自身查询（'帮我'含第一人称）"),
        ("你还记得我叫什么吗", True, "自身查询"),
        ("我以前说过自己喜欢什么吗", True, "自身查询"),
        ("我室友说他喜欢喝咖啡", False, "混合主体"),
        ("他建议我重新规划职业目标", False, "混合主体"),
    ])
    def test_guard_boundaries(self, message, expected, why):
        assert Agent._is_user_subject(message) is expected, why

    def test_empty_and_none_message(self):
        assert Agent._is_user_subject("") is False
        assert Agent._is_user_subject(None) is False


# =====================================================================
# 2. 16 条验收案例（agent 层：拦截=零查询；放行=正常命中）
# =====================================================================

# expect: intercept / pass / observe（已知缺口或规则未覆盖，如实记录）
ACCEPT_CASES = [
    ("G01", "我朋友说我应该换工作", "intercept", None,
     "普通闲聊+第三方名词"),
    ("G02", "他建议我重新规划职业目标", "intercept", None,
     "混合主体（他+我）"),
    ("G03", "我想知道其他同学学什么专业", "intercept", None,
     "第三方名词；'其他'豁免不掩盖'同学'"),
    ("G04", "我室友推荐我看的课程适合我吗", "intercept", None,
     "【已知召回损失】语义主体或为用户，按预声明保守拦截"),
    ("G05", "假设我的专业是建筑学", "intercept", None,
     "假设场景"),
    ("G06", "我以前说过自己喜欢什么吗", "pass", ["preference"],
     "自身查询，不得误拦"),
    ("G07", "你还记得我叫什么吗", "pass", ["name"],
     "自身查询，不得误拦（既有 A02 型）"),
    ("G08", "这个项目的目标用户是谁", "intercept", None,
     "无第一人称；删'目标'后若命中 project 也须被守卫兜住"),
    ("G09", "我司今年的工作目标需要细化", "intercept", None,
     "集体主语；删'目标'后规则亦不再命中（双重保障）"),
    ("G10", "我想了解我朋友的职业规划", "intercept", None,
     "混合主体"),
    ("G11", "我没有职业目标", "observe", None,
     "【已知缺口】否定句：预期仍命中'职业目标'并注入，如实记录"),
    ("G12", "我知道其他学科的情况", "observe", None,
     "'他'豁免正例：守卫放行，但规则本就未命中（零查询非守卫所致）"),
    ("G13", "我室友说他喜欢喝手冲咖啡", "intercept", None,
     "混合主体（跨层 T01 同型）"),
    ("G14", "他的职业规划是怎样的", "intercept", None,
     "纯第三方（跨层 T03 同型）"),
    ("G15", "我是什么专业的", "pass", ["major"],
     "自身查询基础能力"),
    ("G16", "我以后准备做什么工作", "observe", None,
     "守卫放行；但该表述不在词表内→零查询（记录覆盖缺口，非守卫拦截）"),
]


@pytest.mark.parametrize("cid,q,expect,rel,_why", ACCEPT_CASES,
                         ids=[c[0] for c in ACCEPT_CASES])
def test_acceptance_cases(cid, q, expect, rel, _why, tmp_path,
                          monkeypatch, capsys):
    agent = build_agent(tmp_path, monkeypatch)
    counts = attach_spy(agent)

    cand = agent.retrieve_long_term_memories(q)
    keys = [m["key"] for m in cand]
    n_calls = total_calls(counts)
    print(f"\n[{cid}]「{q}」expect={expect} 调用={n_calls} "
          f"find={counts['find_keys']} 候选={keys}")

    if expect == "intercept":
        assert n_calls == 0, (
            f"[{cid}] 被拦截时不得发生任何记忆查询，实际 {n_calls} 次"
        )
        assert cand == [], f"[{cid}] 被拦截时候选必须为空：{keys}"
    elif expect == "pass":
        assert n_calls > 0, f"[{cid}] 自身查询被误拦（零查询）"
        for key in rel:
            assert key in keys, f"[{cid}] 应命中 {key}，实际 {keys}"
    else:  # observe：如实记录，不做通过/失败判定以外的断言
        pass


# =====================================================================
# 3. 孤立“目标”删除后的影响（明确表达保留 / 宽泛表达不再命中）
# =====================================================================

def test_explicit_goal_phrases_still_match(tmp_path, monkeypatch, capsys):
    """删除孤立'目标'后，明确表达仍应命中 goal 规则。"""
    agent = build_agent(tmp_path, monkeypatch)

    for q, must in [
        ("我的职业目标是什么？", "career_goal"),
        ("我的职业规划是什么", "career_goal"),
        ("我的目标你还记得吗", "career_goal"),
        ("我以后想从事什么工作", "career_goal"),
        ("我想做哪方面的工作", "career_goal"),
    ]:
        counts = attach_spy(agent)
        cand = agent.retrieve_long_term_memories(q)
        keys = [m["key"] for m in cand]
        capsys.readouterr()
        print(f"\n[目标-保留]「{q}」→ {keys}")
        assert total_calls(counts) > 0, f"「{q}」应命中 goal 规则"
        assert must in keys, f"「{q}」应返回 {must}，实际 {keys}"


def test_bare_goal_word_no_longer_triggers(tmp_path, monkeypatch, capsys):
    """删除孤立'目标'后，仅含'目标'的宽泛句不再命中 goal 规则。

    注意：'我有什么长期目标'属于自然问法，删除孤立词后失去覆盖——
    这是删词的可预期代价，如实断言为'不再触发'并保留记录。
    """
    agent = build_agent(tmp_path, monkeypatch)
    counts = attach_spy(agent)
    cand = agent.retrieve_long_term_memories("我有什么长期目标")
    capsys.readouterr()
    print(f"\n[目标-移除]「我有什么长期目标」→ 调用={total_calls(counts)} 候选={cand}")
    assert total_calls(counts) == 0, "孤立'目标'已删除，不应再触发检索"


# =====================================================================
# 4. 上下文层抽查（真实 run() + 假客户端捕获提示词）
# =====================================================================

def _run_and_capture(agent, q):
    agent.client = FakeClient(RUN_SCRIPT)
    result = agent.run(q, user_id="test-user")
    assert result["success"] is True
    return agent.client.calls[1][0]["content"]  # Planner 提示词


def test_context_layer_intercept_case13(tmp_path, monkeypatch, capsys):
    """混合主体案例：上下文层不得出现任何用户记忆条目。"""
    agent = build_agent(tmp_path, monkeypatch)
    planner = _run_and_capture(agent, "我室友说他喜欢喝手冲咖啡")
    capsys.readouterr()
    assert "- preference =" not in planner
    assert "- language_preference =" not in planner
    assert "（无相关长期记忆" in planner


def test_context_layer_pass_case7(tmp_path, monkeypatch, capsys):
    """自身查询案例：上下文层应包含应注入的记忆。"""
    agent = build_agent(tmp_path, monkeypatch)
    planner = _run_and_capture(agent, "你还记得我叫什么吗")
    capsys.readouterr()
    assert "- name = 李小明" in planner


def test_context_layer_negative_sentence_refused_case11(
        tmp_path, monkeypatch, capsys):
    """【安全基线（已更新，原 test_context_layer_negative_sentence_gap_case11）】
    否定句"我没有职业目标"：主体门放行，但意图分类为 negative →
    统一门在一切记忆数据库查询之前拒绝；Planner 不注入 career_goal。
    原"缺口基线"负向断言（要求注入仍存在）已由"语义否定最小补丁"
    修复，本测试改为验证拒绝行为。真实模型行为仍未验证。"""
    agent = build_agent(tmp_path, monkeypatch)
    counts = attach_spy(agent)
    planner = _run_and_capture(agent, "我没有职业目标")
    capsys.readouterr()

    # 拒绝必须发生在查询之前（零记忆查询调用）
    assert (counts["find"] == 0 and counts["search"] == 0
            and counts["by_cat"] == 0), f"拒绝必须先于查询，实际：{counts}"

    injected = "- career_goal =" in planner
    print(f"\n[安全基线] 否定句注入 career_goal = {injected}")
    assert injected is False
    assert "机器人工程师" not in planner
    assert "（无相关长期记忆" in planner
