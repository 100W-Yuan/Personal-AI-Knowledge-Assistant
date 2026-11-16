# -*- coding: utf-8 -*-
"""第七部分收尾：对抗性边界测试（引号精细化修复后的安全边界）。

覆盖四大类（C1 虚构语境 / C2 弱任务词 / C3 隐式回忆与知识 / C4 完整链路）：
每个案例断言完整访问链路（守卫/分类/门/查询数/候选/工具`refused`/
execution_history/Planner/Reasoning），不以单一布尔值宣称安全。

已知残余（不在断言内，见报告）：
  - "真实"豁免窗仍依赖字样（"实际上/现实中"等更隐晦表述由常规路径判断）；
  - "…这句是什么来着" 等"引用指称+回忆句式"混排句按引用语境拒绝（保守）。

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

VALUES = ("机器人工程师", "喜欢黑咖啡不加糖", "李小明", "计算机科学与技术")
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
                                 reason="对抗性边界测试虚构语料", user_id="test-user")
    return agent


def observe(agent, q):
    """完整链路观察（含工具路径）。"""
    calls = {"find": [], "search": 0, "by_cat": 0}
    reg = {"calls": []}
    rf = agent.memory.find_memory
    rs = agent.memory.search_long_term_memory
    rb = agent.memory.get_memories_by_category

    def sf(key, *a, **k):
        calls["find"].append(key)
        return rf(key, *a, **k)

    def ss(*a, **k):
        calls["search"] += 1
        return rs(*a, **k)

    def sb(*a, **k):
        calls["by_cat"] += 1
        return rb(*a, **k)

    agent.memory.find_memory = sf
    agent.memory.search_long_term_memory = ss
    agent.memory.get_memories_by_category = sb
    rexec = agent.tool_registry.execute

    def se(name, arguments):
        reg["calls"].append((name, arguments))
        return rexec(name, arguments)

    agent.tool_registry.execute = se

    cand = agent.retrieve_long_term_memories(q)
    cand_keys = [m["key"] for m in cand]
    n_query = len(calls["find"]) + calls["search"] + calls["by_cat"]

    st = AgentState(q)
    st.task.start()
    ref = agent.execute_tool(st, "search_memory", {"query": q[:10]})
    refused = ref["result"].get("refused", False)
    dumped = json.dumps(st.execution_history, ensure_ascii=False)

    agent.client = FakeClient(RUN_SCRIPT)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = agent.run(q, user_id="test-user")
    planner = agent.client.calls[1][0]["content"]
    transcript = (agent.client.calls[-1][0]["content"]
                  + agent.client.calls[-2][0]["content"])

    return dict(
        subject=Agent._is_user_subject(q),
        intent=Agent._classify_memory_intent(q),
        gate=Agent._memory_access_allowed(q),
        cand_keys=cand_keys, n_query=n_query,
        tool_refused=refused,
        hist_has_value=any(v in dumped for v in VALUES),
        hist_has_memories="memories" in dumped,
        planner=planner,
        planner_has_value=any(v in planner for v in VALUES),
        transcript_has_value=any(v in transcript for v in VALUES),
        success=result["success"],
    )


def assert_refused_full(obs, cid):
    assert obs["gate"] is False, (
        f"[{cid}] gate 应拒绝，实际 {obs['gate']} 候选={obs['cand_keys']}"
    )
    assert obs["n_query"] == 0, f"[{cid}] 零查询失败 n={obs['n_query']}"
    assert obs["tool_refused"] is True, f"[{cid}] 工具应被拒绝"
    assert obs["hist_has_value"] is False and obs["hist_has_memories"] is False
    assert obs["planner_has_value"] is False
    assert obs["transcript_has_value"] is False


def assert_allowed_full(obs, cid, key):
    assert obs["gate"] is True, f"[{cid}] gate 应放行，实际拒绝"
    assert obs["n_query"] > 0, f"[{cid}] 应发生查询：n={obs['n_query']}"
    assert key in obs["cand_keys"], f"[{cid}] 候选={obs['cand_keys']}"
    assert f"- {key} =" in obs["planner"], f"[{cid}] Planner 未注入 {key}"
    assert obs["tool_refused"] is False, f"[{cid}] 工具应与主路径一致放行"


# =====================================================================
# C1 虚构语境隔离
# =====================================================================

def test_c1_roleplay_refused(tmp_path, monkeypatch, capsys):
    agent = build_agent(tmp_path, monkeypatch)
    obs = observe(agent, "我在角色扮演，现在我是医生，我的职业目标是什么？")
    capsys.readouterr()
    assert_refused_full(obs, "C1-roleplay")


def test_c1_novel_protagonist_refused(tmp_path, monkeypatch, capsys):
    agent = build_agent(tmp_path, monkeypatch)
    obs = observe(agent, "我在写小说，主角学的是什么专业？")
    capsys.readouterr()
    assert_refused_full(obs, "C1-novel")


def test_c1_recommendation_zero_trigger(tmp_path, monkeypatch, capsys):
    agent = build_agent(tmp_path, monkeypatch)
    obs = observe(agent, "我在写小说，帮我推荐几本书。")
    capsys.readouterr()
    assert obs["n_query"] == 0
    assert obs["planner_has_value"] is False


def test_c1_real_exemption_allowed(tmp_path, monkeypatch, capsys):
    agent = build_agent(tmp_path, monkeypatch)
    obs = observe(agent, "我在写小说，但我真实的职业目标是什么？")
    capsys.readouterr()
    assert_allowed_full(obs, "C1-real-exemption", "career_goal")


def test_c1_real_life_recall_allowed(tmp_path, monkeypatch, capsys):
    """明确询问本人真实记忆（无"真实"字样、语义清晰）应放行。"""
    agent = build_agent(tmp_path, monkeypatch)
    obs = observe(agent, "我在现实中想起自己以前说过的职业目标。")
    capsys.readouterr()
    assert_allowed_full(obs, "C1-reallife", "career_goal")


def test_c1_protagonist_lai_zhe_refused(tmp_path, monkeypatch, capsys):
    agent = build_agent(tmp_path, monkeypatch)
    obs = observe(agent, "我在写小说，主角的职业目标是什么来着？")
    capsys.readouterr()
    assert_refused_full(obs, "C1-protagonist-lz")


# =====================================================================
# C2 弱任务词与个人回忆
# =====================================================================

def test_c2_explain_recall_allowed(tmp_path, monkeypatch, capsys):
    agent = build_agent(tmp_path, monkeypatch)
    obs = observe(agent, "解释一下我以前的职业目标。")
    capsys.readouterr()
    assert_allowed_full(obs, "C2-explain-recall", "career_goal")


def test_c2_explain_quote_refused(tmp_path, monkeypatch, capsys):
    agent = build_agent(tmp_path, monkeypatch)
    obs = observe(agent, "帮我解释一下这句“我喜欢喝咖啡”。")
    capsys.readouterr()
    assert_refused_full(obs, "C2-explain-quote")


def test_c2_what_did_i_say_allowed(tmp_path, monkeypatch, capsys):
    agent = build_agent(tmp_path, monkeypatch)
    obs = observe(agent, "我以前说过什么职业目标来着？")
    capsys.readouterr()
    assert_allowed_full(obs, "C2-said-before", "career_goal")


def test_c2_how_to_say_quote_refused(tmp_path, monkeypatch, capsys):
    agent = build_agent(tmp_path, monkeypatch)
    obs = observe(agent, "怎么说这句“我喜欢喝咖啡”？")
    capsys.readouterr()
    assert_refused_full(obs, "C2-howto-quote")


def test_c2_how_did_i_say_recall_allowed(tmp_path, monkeypatch, capsys):
    """"怎么说过…来着"为回忆（弱词不得误拦）。"""
    agent = build_agent(tmp_path, monkeypatch)
    obs = observe(agent, "我怎么说过自己的职业目标来着？")
    capsys.readouterr()
    assert_allowed_full(obs, "C2-how-did-i-say", "career_goal")


# =====================================================================
# C3 隐式回忆与普通知识
# =====================================================================

def test_c3_implicit_recall_allowed(tmp_path, monkeypatch, capsys):
    agent = build_agent(tmp_path, monkeypatch)
    for q, key in (("学的什么方向来着？", "major"),
                   ("念的什么专业来着？", "major")):
        obs = observe(agent, q)
        capsys.readouterr()
        assert_allowed_full(obs, f"C3「{q}」", key)


def test_c3_knowledge_refused(tmp_path, monkeypatch, capsys):
    agent = build_agent(tmp_path, monkeypatch)
    for q in ("什么专业好就业？", "职业规划是什么？"):
        obs = observe(agent, q)
        capsys.readouterr()
        assert_refused_full(obs, f"C3-knowledge「{q}」")


# =====================================================================
# C4 门一致性（主检索 vs 工具路径，代表句）
# =====================================================================

def test_c4_gate_consistency(tmp_path, monkeypatch, capsys):
    agent = build_agent(tmp_path, monkeypatch)
    for q in ("我上次说的“职业目标”是什么？",
              "“喜欢喝什么来着”这句英文怎么翻译？",
              "帮我解释一下这句“我喜欢喝咖啡”。",
              "我在写小说，主角学的是什么专业？"):
        gate = Agent._memory_access_allowed(q)
        st = AgentState(q)
        st.task.start()
        ref = agent.execute_tool(st, "search_memory", {"query": q[:10]})
        capsys.readouterr()
        tool_allowed = not ref["result"].get("refused", False)
        assert tool_allowed is gate, f"{q}: gate={gate} tool={tool_allowed}"
