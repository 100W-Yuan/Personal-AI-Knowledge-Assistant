# -*- coding: utf-8 -*-
"""第七部分第三步修复的边界测试（先于生产修改编写）。

验证三个修复的分界行为：
  A. 虚构语境隔离：变异虚构标记（扮演/剧本/主角）拒绝；"明确询问真实记忆"
     的豁免窗放行；小说推荐零查询。
  B. 弱任务词收窄：明确第一人称回忆（"解释一下我以前的职业目标"）放行；
     引用/元语言任务（"解释一下这句'…'"）保持拒绝。
  C. 隐式回忆补词：变异句式"我学的什么方向来着"放行召回；知识句式保持拒绝。

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

from agent import Agent          # noqa: E402
from memory import Memory        # noqa: E402
from task import AgentState      # noqa: E402

DUMMY = dict(api_key="dummy", base_url="https://example.invalid", model="dummy")

CORPUS = [
    ("major", "计算机科学与技术", "profile"),
    ("career_goal", "机器人工程师", "goal"),
    ("preference", "喜欢黑咖啡不加糖", "preference"),
    ("name", "李小明", "identity"),
]

VALUES = ("机器人工程师", "喜欢黑咖啡不加糖", "李小明")
ZERO = {"find": [], "search": 0, "by_cat": 0}


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


RUN_SCRIPT = [
    json.dumps({"should_remember": False, "memories": []}),
    json.dumps({"plan": [{"description": "回答", "type": "reasoning"}]}),
    "中间推理",
    "最终答案。",
]


def build_agent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    agent = Agent(**DUMMY)
    agent.memory = Memory(db_path=str(tmp_path / "memory.db"))
    for key, value, category in CORPUS:
        agent.memory.save_memory(key=key, value=value, category=category,
                                 reason="修复边界测试虚构语料", user_id="test-user")
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


def run_case(agent, q):
    agent.client = FakeClient(RUN_SCRIPT)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = agent.run(q, user_id="test-user")
    planner = agent.client.calls[1][0]["content"]
    transcript = (agent.client.calls[-1][0]["content"]
                  + agent.client.calls[-2][0]["content"])
    return result, planner, transcript


def observe(agent, q):
    calls, reg = attach_spies(agent)
    cand = agent.retrieve_long_term_memories(q)
    cand_keys = [m["key"] for m in cand]
    n_query = len(calls["find"]) + calls["search"] + calls["by_cat"]

    st = AgentState(q)
    st.task.start()
    ref = agent.execute_tool(st, "search_memory", {"query": q[:10]})
    refused = ref["result"].get("refused", False)
    dumped = json.dumps(st.execution_history, ensure_ascii=False)

    result, planner, transcript = run_case(agent, q)
    return dict(
        subject=Agent._is_user_subject(q),
        gate=Agent._memory_access_allowed(q),
        cand_keys=cand_keys, n_query=n_query,
        tool_refused=refused, tool_calls=list(reg["calls"]),
        hist_has_value=any(v in dumped for v in VALUES),
        planner_has_value=any(v in planner for v in VALUES),
        transcript_has_value=any(v in transcript for v in VALUES),
        planner=planner,
    )


# =====================================================================
# A. 虚构语境隔离边界
# =====================================================================

FICTION_VARIANTS = [
    ("FA1", "我在扮演医生，我的职业目标是什么？"),
    ("FA2", "剧本里主角的职业目标是什么来着"),
]


@pytest.mark.parametrize("cid,q", FICTION_VARIANTS,
                         ids=[c[0] for c in FICTION_VARIANTS])
def test_fiction_variants_refused(cid, q, tmp_path, monkeypatch, capsys):
    """变异虚构标记（扮演/剧本/主角）：拒 + 零查询 + 零注入。"""
    agent = build_agent(tmp_path, monkeypatch)
    obs = observe(agent, q)
    capsys.readouterr()
    assert obs["gate"] is False, (
        f"[{cid}] 虚构语境应被拒绝；实际 gate={obs['gate']} "
        f"候选={obs['cand_keys']}"
    )
    assert obs["n_query"] == 0, f"[{cid}] 零查询失败 n={obs['n_query']}"
    assert obs["tool_refused"] is True
    assert obs["hist_has_value"] is False
    assert obs["planner_has_value"] is False
    assert obs["transcript_has_value"] is False


def test_fiction_real_intent_exemption(tmp_path, monkeypatch, capsys):
    """豁免窗：明确询问真实个人记忆（"但我真实的…"）应放行检索。"""
    q = "我在写小说，但我真实的职业目标是什么？"
    agent = build_agent(tmp_path, monkeypatch)
    obs = observe(agent, q)
    capsys.readouterr()
    assert obs["gate"] is True, (
        f"真实意图豁免失败：gate={obs['gate']}"
    )
    assert "career_goal" in obs["cand_keys"]
    assert "- career_goal =" in obs["planner"]


def test_fiction_recommendation_zero_trigger(tmp_path, monkeypatch, capsys):
    """小说推荐请求：零查询、零注入（不因虚构标记影响正常回答）。"""
    q = "我在写小说，帮我推荐几本书。"
    agent = build_agent(tmp_path, monkeypatch)
    obs = observe(agent, q)
    capsys.readouterr()
    assert obs["n_query"] == 0
    assert obs["planner_has_value"] is False


# =====================================================================
# B. 弱任务词收窄边界
# =====================================================================

def test_weak_task_recall_now_allowed(tmp_path, monkeypatch, capsys):
    """明确第一人称回忆（此前被"解释"误拦）应放行并真实检索。"""
    q = "解释一下我以前的职业目标。"
    agent = build_agent(tmp_path, monkeypatch)
    obs = observe(agent, q)
    capsys.readouterr()
    assert obs["gate"] is True
    assert "career_goal" in obs["cand_keys"]
    assert "- career_goal =" in obs["planner"]


def test_weak_task_quote_task_still_refused(tmp_path, monkeypatch, capsys):
    """引用/元语言任务（含引号）保持拒绝：零查询、工具拒绝、零注入。"""
    q = "帮我解释一下这句'我喜欢喝咖啡'。"
    agent = build_agent(tmp_path, monkeypatch)
    obs = observe(agent, q)
    capsys.readouterr()
    assert obs["gate"] is False, (
        f"引用任务被放行（回归风险）：gate={obs['gate']}"
    )
    assert obs["n_query"] == 0 and obs["tool_refused"] is True
    assert obs["planner_has_value"] is False


def test_weak_task_knowledge_no_person_question_refused(tmp_path, monkeypatch,
                                                        capsys):
    """无第一人称的元语言/知识句保持拒绝（"解释一下什么是职业规划"）。"""
    q = "解释一下什么是职业规划。"
    agent = build_agent(tmp_path, monkeypatch)
    obs = observe(agent, q)
    capsys.readouterr()
    assert obs["gate"] is False
    assert obs["n_query"] == 0


# =====================================================================
# C. 隐式回忆补词边界
# =====================================================================

def test_c1_variant_recall_with_first_person(tmp_path, monkeypatch, capsys):
    """含第一人称的变体式回忆应召回。"""
    q = "我学的什么方向来着？"
    agent = build_agent(tmp_path, monkeypatch)
    obs = observe(agent, q)
    capsys.readouterr()
    assert obs["gate"] is True
    assert "major" in obs["cand_keys"], f"候选={obs['cand_keys']}"
    assert "- major =" in obs["planner"]


def test_c1_knowledge_variant_refused(tmp_path, monkeypatch, capsys):
    """近似知识句（无回忆句式/无"来着"）保持零查询。"""
    q = "学的什么比较好就业"
    agent = build_agent(tmp_path, monkeypatch)
    obs = observe(agent, q)
    capsys.readouterr()
    assert obs["n_query"] == 0
    assert obs["planner_has_value"] is False
