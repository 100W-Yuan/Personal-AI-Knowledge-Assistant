# -*- coding: utf-8 -*-
"""有限隐式主语回忆例外（P1）安全测试 —— 先于生产修改编写。

设计依据：第六部分第十六步矩阵（20 案例）。
P1 目标行为（仅当测试锁定后才实施到 agent.py::_is_user_subject）：
  例外放行 = 无第一人称 且 无强负标记（复用同一 has_negative）
             且 无否定/拒绝信号 且 无元语言/任务请求词
             且 含记忆事项词 且 含回忆句式（是什么/是啥/什么来着/来着）

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
    ("learning_goal", "学好检索与评估技术", "goal"),
    ("preference", "喜欢黑咖啡不加糖", "preference"),
    ("project", "社区图书角项目", "project"),
    ("skill", "熟悉Python基础语法", "skill"),
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
                                 reason="隐式主语例外测试虚构语料", user_id="test-user")
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
    transcript = (agent.client.calls[-1][0]["content"]
                  + agent.client.calls[-2][0]["content"])
    return result, planner, transcript


# =====================================================================
# A. 隐式主语回忆正例（6）：通过主体守卫；放行 != 必然命中（分开断言）
# =====================================================================

IMPLICIT_POSITIVE = [
    ("A1", "专业是什么来着？", "major"),
    ("A2", "职业目标是什么来着？", "career_goal"),
    ("A3", "喜欢喝什么来着？", "preference"),
    ("A4", "名字是什么来着？", "name"),
    ("A5", "擅长什么来着？", "skill"),
    ("A6", "在做什么项目来着？", "project"),
]


@pytest.mark.parametrize("cid,q,key", IMPLICIT_POSITIVE,
                         ids=[c[0] for c in IMPLICIT_POSITIVE])
def test_implicit_positive(cid, q, key, tmp_path, monkeypatch, capsys):
    # 守卫层（P1 目标：放行）
    assert Agent._is_user_subject(q) is True, f"[{cid}] 隐式回忆未被放行"
    assert Agent._memory_access_allowed(q) is True, f"[{cid}] 统一门未放行"

    # 链路层：放行后进入检索（本语料 key 均存在 → 应命中；
    # 注意：守卫放行 != 必然命中——命中还取决于规则与语料）
    agent = build_agent(tmp_path, monkeypatch)
    calls, _ = attach_spies(agent)
    cand = agent.retrieve_long_term_memories(q)
    capsys.readouterr()
    keys = [m["key"] for m in cand]
    assert calls["find"], f"[{cid}] 放行后应发生 key 查询：{calls}"
    assert key in keys, f"[{cid}] 应召回 {key}，实际 {keys}"

    # 集成：记忆进入 Planner
    result, planner, transcript = run_case(agent, q)
    capsys.readouterr()
    assert result["success"] is True
    assert f"- {key} =" in planner


# =====================================================================
# B. 必须拒绝的负例（10）
# =====================================================================

# (id, 文本, 主体门预期)：主体门只判"主体是谁"；
# 含"我"的拒绝检索/否定句由意图门拦截（双门协同），主体门预期为 True。
NEGATIVE = [
    ("B1", "我室友是什么专业？", False),
    ("B2", "他的职业目标是什么？", False),
    ("B3", "假设我是学生，专业是什么来着？", False),
    ("B4", "他转述“专业是什么来着”是什么意思？", False),
    ("B5", "“喜欢喝什么来着”这句英文怎么翻译？", False),
    ("B6", "我没有专业目标", True),   # 含"我"：主体门放行，意图层 negative 拦截
    ("B7", "不要检索我的名字", True),  # 含"我"：主体门放行，意图层 deny 拦截
    ("B8", "公司专业是什么来着？", False),
    ("B9", "他们说的专业是什么来着", False),
    ("B10", "别查我的职业目标", True),  # 含"我"：主体门放行，意图层 deny 拦截
]


@pytest.mark.parametrize("cid,q,expect_subject", NEGATIVE,
                         ids=[c[0] for c in NEGATIVE])
def test_negative_refused_zero_query(cid, q, expect_subject, tmp_path,
                                     monkeypatch, capsys):
    assert Agent._is_user_subject(q) is expect_subject, (
        f"[{cid}] 主体门预期 {expect_subject}"
    )
    assert Agent._memory_access_allowed(q) is False, f"[{cid}] 统一门应拒绝"

    agent = build_agent(tmp_path, monkeypatch)
    calls, reg = attach_spies(agent)
    cand = agent.retrieve_long_term_memories(q)
    capsys.readouterr()
    assert cand == [], f"[{cid}] 拒绝态不得返回候选：{cand}"
    assert calls == {"find": [], "search": 0, "by_cat": 0}, (
        f"[{cid}] 拒绝态必须零数据库查询：{calls}"
    )
    assert reg["calls"] == [], f"[{cid}] 拒绝态不得调用工具：{reg['calls']}"

    result, planner, transcript = run_case(agent, q)
    capsys.readouterr()
    assert result["success"] is True
    assert "（无相关长期记忆" in planner
    assert PREFERENCE_VALUE not in planner
    assert PREFERENCE_VALUE not in transcript


def test_b5_translation_quote_special(tmp_path, monkeypatch, capsys):
    """B5 专项：翻译引用句必须拒绝且工具路径也不可绕过。"""
    q = "“喜欢喝什么来着”这句英文怎么翻译？"

    agent = build_agent(tmp_path, monkeypatch)
    calls, reg = attach_spies(agent)
    state = AgentState(q)
    state.task.start()
    refusal = agent.execute_tool(state, "search_memory", {"query": "喜欢喝"})
    capsys.readouterr()
    assert refusal["result"]["refused"] is True
    assert reg["calls"] == []
    assert calls == {"find": [], "search": 0, "by_cat": 0}
    dumped = json.dumps(state.execution_history, ensure_ascii=False)
    assert PREFERENCE_VALUE not in dumped
    assert "memories" not in dumped

    # 集成：工具计划 + 命中型参数 → 拒绝、零泄漏
    mem2, reg2 = attach_spies(agent)
    result, planner, transcript = run_case(agent, q,
                                           tool_params={"query": "喜欢喝"})
    capsys.readouterr()
    assert result["success"] is True
    assert reg2["calls"] == []
    assert PREFERENCE_VALUE not in planner
    assert PREFERENCE_VALUE not in transcript


# =====================================================================
# C. 普通知识回归（4）
# =====================================================================

def test_c1_c2_c4_refused(tmp_path, monkeypatch, capsys):
    for q in ("计算机专业学什么？", "职业规划应该怎么做？",
              "什么专业好就业？"):
        assert Agent._is_user_subject(q) is False, q
        assert Agent._memory_access_allowed(q) is False, q


def test_c3_gate_allows_but_zero_query(tmp_path, monkeypatch, capsys):
    """C3「目标管理是什么？」：事项词+句式可放行，但孤立"目标"已删除
    → 规则不命中 → 零查询零注入（双条件安全实证）。"""
    q = "目标管理是什么？"
    agent = build_agent(tmp_path, monkeypatch)
    calls, _ = attach_spies(agent)
    cand = agent.retrieve_long_term_memories(q)
    capsys.readouterr()
    assert calls == {"find": [], "search": 0, "by_cat": 0}, calls
    assert cand == []


# =====================================================================
# D. 工具集成：允许路径不误伤 / 拒绝路径不绕过
# =====================================================================

def test_tool_allowed_for_implicit_positive(tmp_path, monkeypatch, capsys):
    agent = build_agent(tmp_path, monkeypatch)
    calls, reg = attach_spies(agent)
    result, planner, transcript = run_case(
        agent, "专业是什么来着？", tool_params={"query": "专业"})
    capsys.readouterr()
    assert result["success"] is True
    assert reg["calls"] and reg["calls"][0][0] == "search_memory"
    assert "refused" not in transcript
