# -*- coding: utf-8 -*-
"""长期记忆误检索最小修复测试（第六部分第十九步）——先于生产修改编写。

目标问题（第十八步审计 P1-1 / P1-4）：
  P1-1：半角引号 / 全角单引号包裹的引用内容不应通过隐式主语例外。
  P1-4："X是什么？" 式普通知识问句不应仅因事项词触发个人记忆检索，
        隐式回忆例外应要求明确的回忆句式（"来着"）。
每个案例同时断言四层：主体守卫 / 意图分类 / 统一访问门 / 实际检索行为
（spy 零查询 + 工具拒绝 + 执行历史无泄漏 + Planner/Reasoning 无注入）。

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

PREFERENCE_VALUE = "喜欢黑咖啡不加糖"


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
                                 reason="误检索修复测试虚构语料", user_id="test-user")
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


def run_case(agent, q):
    agent.client = FakeClient(RUN_SCRIPT)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = agent.run(q, user_id="test-user")
    planner = agent.client.calls[1][0]["content"]
    transcript = (agent.client.calls[-1][0]["content"]
                  + agent.client.calls[-2][0]["content"])
    return result, planner, transcript


ZERO = {"find": [], "search": 0, "by_cat": 0}


# =====================================================================
# A. 引号与引用语境（期望：全部拒绝、零查询、零注入）
# =====================================================================

QUOTE_CASES = [
    ("A1", "'喜欢喝什么来着'？"),          # 半角单引号（成对）
    ("A2", '"喜欢喝什么来着"？'),          # 半角双引号（成对）
    ("A3", "‘喜欢喝什么来着’？"),          # 全角单引号（成对）
    ("A4", "“喜欢喝什么来着”？"),          # 全角双引号（成对）
    ("A5", "“喜欢喝什么来着”是什么意思？"),
    ("A6", "“喜欢喝什么来着”这句英文怎么翻译？"),
    ("A7", "喜欢喝什么来着”？"),           # 不成对（右全角）
    ("A8", "喜欢喝什么来着'？"),            # 不成对（右半角单）
]


@pytest.mark.parametrize("cid,q", QUOTE_CASES, ids=[c[0] for c in QUOTE_CASES])
def test_quote_cases_refused(cid, q, tmp_path, monkeypatch, capsys):
    # 四层断言：守卫 / 分类 / 门 / 行为
    assert Agent._is_user_subject(q) is False, f"[{cid}] 引用语境应被主体门拒绝"
    assert Agent._classify_memory_intent(q) == "neutral"
    assert Agent._memory_access_allowed(q) is False, f"[{cid}] 统一门应拒绝"

    agent = build_agent(tmp_path, monkeypatch)
    calls, reg = attach_spies(agent)
    cand = agent.retrieve_long_term_memories(q)
    capsys.readouterr()
    assert cand == []
    assert calls == ZERO, f"[{cid}] 必须零数据库查询：{calls}"

    # 工具路径：不被执行、零泄漏
    state = AgentState(q)
    state.task.start()
    refusal = agent.execute_tool(state, "search_memory", {"query": "喜欢喝"})
    capsys.readouterr()
    assert refusal["result"]["refused"] is True
    assert reg["calls"] == []
    dumped = json.dumps(state.execution_history, ensure_ascii=False)
    assert PREFERENCE_VALUE not in dumped and "memories" not in dumped

    # 上下文层：Planner / Reasoning 无个人记忆
    result, planner, transcript = run_case(agent, q)
    capsys.readouterr()
    assert result["success"] is True
    assert PREFERENCE_VALUE not in planner
    assert PREFERENCE_VALUE not in transcript


# =====================================================================
# B. 普通知识问句 vs 明确回忆（期望：知识=拒；"来着"回忆=保留召回）
# =====================================================================

KNOWLEDGE_REFUSED = [
    ("B1", "职业规划是什么？"),
    ("B2", "计算机专业是什么？"),
    ("B3", "职业规划应该怎么做？"),
    ("B4", "什么专业好就业？"),
]

RECALL_PRESERVED = [
    ("B5", "专业是什么来着？", "major"),
    ("B6", "职业目标是什么来着？", "career_goal"),
]


@pytest.mark.parametrize("cid,q", KNOWLEDGE_REFUSED,
                         ids=[c[0] for c in KNOWLEDGE_REFUSED])
def test_knowledge_questions_no_injection(cid, q, tmp_path, monkeypatch,
                                          capsys):
    assert Agent._is_user_subject(q) is False, f"[{cid}] 知识问句不应放行"
    assert Agent._classify_memory_intent(q) == "neutral"
    assert Agent._memory_access_allowed(q) is False

    agent = build_agent(tmp_path, monkeypatch)
    calls, _ = attach_spies(agent)
    cand = agent.retrieve_long_term_memories(q)
    capsys.readouterr()
    assert cand == []
    assert calls == ZERO, f"[{cid}] 知识问句必须零查询：{calls}"

    result, planner, transcript = run_case(agent, q)
    capsys.readouterr()
    assert result["success"] is True
    assert "（无相关长期记忆" in planner


@pytest.mark.parametrize("cid,q,key", RECALL_PRESERVED,
                         ids=[c[0] for c in RECALL_PRESERVED])
def test_recall_preserved_with_lai_zhe(cid, q, key, tmp_path, monkeypatch,
                                       capsys):
    """明确回忆句式（"来着"）仍保留召回能力。"""
    assert Agent._is_user_subject(q) is True, f"[{cid}] 隐式回忆应放行"
    assert Agent._classify_memory_intent(q) == "neutral"
    assert Agent._memory_access_allowed(q) is True

    agent = build_agent(tmp_path, monkeypatch)
    calls, _ = attach_spies(agent)
    cand = agent.retrieve_long_term_memories(q)
    capsys.readouterr()
    keys = [m["key"] for m in cand]
    assert calls["find"], f"[{cid}] 应发生查询：{calls}"
    assert key in keys, f"[{cid}] 应召回 {key}，实际 {keys}"

    result, planner, transcript = run_case(agent, q)
    capsys.readouterr()
    assert result["success"] is True
    assert f"- {key} =" in planner
