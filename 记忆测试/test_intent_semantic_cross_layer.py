# -*- coding: utf-8 -*-
"""意图门 × 记忆检索 跨层风险独立测试（第四阶段第四步第二部分）。

验证链路：用户问题 → 意图规则(agent.py) → 记忆检索 → 候选集
          → format → Planner/Reasoning 提示词（最终上下文）
双层分别评估：
  A. 候选层：直接调用 retrieve_long_term_memories() 并记录意图触发、
     首个命中类别（由 find_memory 调用序列推断）、返回记忆类别/主体。
  B. 上下文层：用假客户端完整跑 run()，从 Planner/Reasoning 的真实
     提示词中解析实际注入的记忆 key（非字符串拼接孤立测试）。
安全：全程临时库+虚构记忆，不读正式 memory.db；不接 Ollama；
      不修改任何生产代码。40 条案例为一组冻结规格（标签不得事后修改）。
"""

import json
import re
import sqlite3
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent import Agent          # noqa: E402
from memory import Memory        # noqa: E402

DUMMY = dict(api_key="dummy", base_url="https://example.invalid", model="dummy")

# =====================================================================
# 12 条虚构记忆（跨层测试专用；与任何正式数据无关）
# =====================================================================

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
ALL_KEYS = [k for k, _, _ in CORPUS]
KEY_CAT = {k: c for k, _, c in CORPUS}
# key → 所属规则类别（用于从 find_memory 调用序列推断首个命中规则）
KEY_RULE = {
    "name": "name", "major": "major",
    "career_goal": "goal", "learning_goal": "goal",
    "preference": "preference", "language_preference": "preference",
    "skill": "skill", "skills": "skill",
    "project": "project", "project_name": "project",
}


def all_except(*allow):
    return [k for k in ALL_KEYS if k not in allow]


# =====================================================================
# 40 条冻结案例（ID/查询/主体/预期/类别/允许/禁止/失败判定见 part1 设计）
# expect: retrieval=应检索  reject=应拒绝  clarify=应澄清  unclear=标签待确认
# =====================================================================

def C(cid, q, subject, expect, allow, rel=None, category=None, note=""):
    return dict(id=cid, q=q, subject=subject, expect=expect,
                category=category, allow=list(allow), rel=list(rel or allow),
                forbid=all_except(*allow), note=note)


CASES = [
    # U 组：用户自身（应检索）
    C("U01", "按我的口味给我推荐个喝的", "用户", "retrieval",
      ["preference", "language_preference"],
      rel=["preference", "language_preference"], category="preference"),
    C("U02", "说说我的习惯吧", "用户", "retrieval",
      ["preference", "language_preference"],
      rel=["preference", "language_preference"], category="preference"),
    C("U03", "我的目标你还记得吗", "用户", "retrieval",
      ["career_goal", "learning_goal", "side_goal"],
      rel=["career_goal", "learning_goal"], category="goal"),
    C("U04", "我读的是什么专业", "用户", "retrieval",
      ["major", "degree"], rel=["major"], category="profile"),
    C("U05", "我在做什么项目", "用户", "retrieval",
      ["project"], rel=["project"], category="project"),
    C("U06", "我擅长哪些方面", "用户", "retrieval",
      ["skill", "learning_goal"], rel=["skill"], category="skill"),
    C("U07", "我的名字你还记得吗", "用户", "retrieval",
      ["name"], rel=["name"], category="identity"),
    # T 组：第三方（应拒绝）
    C("T01", "我室友说他喜欢喝手冲咖啡", "第三方", "reject", []),
    C("T02", "表弟喜欢什么运动你知道吗", "第三方", "reject", []),
    C("T03", "他的职业规划是怎样的", "第三方", "reject", []),
    C("T04", "同事的孩子学的是什么专业", "第三方", "reject", []),
    C("T05", "那位老师擅长什么", "第三方", "reject", []),
    C("T06", "朋友在做什么项目呢", "第三方", "reject", []),
    C("T07", "他叫什么名字来着", "第三方", "reject", []),
    C("T08", "他们公司有什么目标", "第三方组织", "reject", []),
    # Q 组：转述/引用/假设
    C("Q01", "他昨天跟我说他特别喜欢喝奶茶", "第三方(转述)", "reject", []),
    C("Q02", "我朋友问我她的目标应该怎么定", "第三方(转述)", "reject", []),
    C("Q03", "如果我喜欢喝气泡水，你会推荐什么", "假设", "clarify", []),
    C("Q04", "假设我的专业是建筑学，就业前景如何", "假设", "clarify", []),
    C("Q05", "帮我分析一下那句“我喜欢你”是什么意思", "引用", "reject", []),
    C("Q06", "我爸妈希望我以后从事什么工作，你怎么看", "第三方期望",
      "unclear", [], note="预期行为含'或'，标注设计问题/待确认，不纳入二分类"),
    # K 组：含关键词但不要求查用户记忆（应拒绝）
    C("K01", "这款软件我很喜欢它的界面", "用户(物品评价)", "reject", []),
    C("K02", "公司目标是明年上市", "第三方组织", "reject", []),
    C("K03", "这个项目的目标是提升加载速度", "事物", "reject", [],
      note="边界：需同时记录首命中类别与最终检索类别"),
    C("K04", "你擅长什么", "AI 自身", "reject", []),
    C("K05", "他的名字叫张伟", "第三方", "reject", []),
    # N 组：意图漏触发（应检索）
    C("N01", "我平时爱喝点什么来着", "用户", "retrieval",
      ["preference", "language_preference"], category="preference"),
    C("N02", "我这个人在生活里有什么习惯", "用户", "retrieval",
      ["preference", "language_preference"], category="preference"),
    C("N03", "我以后准备做什么营生", "用户", "retrieval",
      ["career_goal", "learning_goal"], category="goal"),
    C("N04", "我念的是什么方向", "用户", "retrieval",
      ["major", "degree"], rel=["major"], category="profile"),
    # C 组：意图命中但类别错配
    C("C01", "这个项目的目标用户是谁", "事物/产品", "reject", [],
      note="边界：需同时记录首命中类别与最终检索类别"),
    C("C02", "我司今年的工作目标需要细化", "工作安排", "reject", []),
    C("C03", "我们团队比赛的目标是拿第一", "第三方团队", "reject", []),
    C("C04", "项目的目标进度如何", "事物", "reject", [],
      note="边界：需同时记录首命中类别与最终检索类别"),
    # H 组：高相似但主体错误
    C("H01", "我朋友李小明最近怎么样", "第三方(与用户同名)", "reject", []),
    C("H02", "我们班上也有个叫李小明的人", "第三方(与用户同名)", "reject", []),
    C("H03", "另一个同学也说自己学的是计算机", "第三方", "reject", []),
]

assert len(CASES) == 37  # 37 条常规案例 + 3 条异常场景(D01-D03) = 40 条冻结清单
assert len({c["id"] for c in CASES}) == 37

# =====================================================================
# 假客户端 + 夹具 + 两层捕获
# =====================================================================

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


def build_agent(tmp_path, monkeypatch, corpus=None):
    monkeypatch.chdir(tmp_path)
    agent = Agent(**DUMMY)
    agent.memory = Memory(db_path=str(tmp_path / "memory.db"))
    for key, value, category in (corpus or CORPUS):
        agent.memory.save_memory(key=key, value=value, category=category,
                                 reason="跨层测试虚构语料", user_id="test-user")
    return agent


def attach_spy(agent):
    """记录三个检索方法的调用参数（推断首命中规则用）。"""
    log = {"find": [], "search": [], "by_cat": []}
    real_find = agent.memory.find_memory
    real_search = agent.memory.search_long_term_memory
    real_by_cat = agent.memory.get_memories_by_category

    def spy_find(key, *a, **k):
        log["find"].append(key)
        return real_find(key, *a, **k)

    def spy_search(query, *a, **k):
        log["search"].append(dict(query=query, k=k))
        return real_search(query, *a, **k)

    def spy_by_cat(category, *a, **k):
        log["by_cat"].append(category)
        return real_by_cat(category, *a, **k)

    agent.memory.find_memory = spy_find
    agent.memory.search_long_term_memory = spy_search
    agent.memory.get_memories_by_category = spy_by_cat
    return log


def context_keys_from_prompts(client):
    """从真实 Planner/Reasoning 提示词中解析注入的记忆 key（两层共用口径）。"""
    keys = set()
    for call in client.calls[1:]:  # 第 0 次为提取提示词
        content = call[0]["content"]
        for match in re.findall(r"^- ([A-Za-z_0-9]+) = ", content, re.M):
            keys.add(match)
    return keys


def first_rule_category(find_log):
    for key in find_log:
        if key in KEY_RULE:
            return KEY_RULE[key]
    return None


RESULTS = {}


def run_case(agent, case):
    """返回一条完整记录（候选层+上下文层）。"""
    log = attach_spy(agent)

    # ---- A. 候选层 ----
    cand = agent.retrieve_long_term_memories(case["q"])
    cand_keys = [m["key"] for m in cand]
    cand_cats = [m["category"] for m in cand]
    triggered = bool(log["find"] or log["search"] or log["by_cat"])
    first_rule = first_rule_category(log["find"])

    # ---- B. 上下文层（真实 run()，假客户端捕获提示词）----
    agent.client = FakeClient(RUN_SCRIPT)
    result = agent.run(case["q"], user_id="test-user")
    assert result["success"] is True
    ctx_keys = context_keys_from_prompts(agent.client)

    forbid = set(case["forbid"])
    cand_forbidden = sorted(forbid & set(cand_keys))
    ctx_forbidden = sorted(forbid & ctx_keys)

    record = dict(
        id=case["id"], q=case["q"], subject=case["subject"],
        expect=case["expect"], expected_category=case["category"],
        triggered=triggered, first_rule=first_rule,
        find_log=list(log["find"]),
        cand_keys=cand_keys, cand_cats=cand_cats, ctx_keys=sorted(ctx_keys),
        cand_forbidden=cand_forbidden, ctx_forbidden=ctx_forbidden,
        note=case["note"],
    )
    RESULTS[case["id"]] = record
    return record


# =====================================================================
# 参数化主测试：候选层与上下文层分别判定（不做二分类混淆）
# =====================================================================

@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_cross_layer(case, tmp_path, monkeypatch, capsys):
    agent = build_agent(tmp_path, monkeypatch)
    rec = run_case(agent, case)
    capsys.readouterr()  # 吸收 run() 的控制台输出

    print(f"\n[{rec['id']}]「{rec['q']}」主体={rec['subject']} 预期={rec['expect']}")
    print(f"  触发={rec['triggered']} 首命中规则={rec['first_rule']} "
          f"find调用={rec['find_log']}")
    print(f"  候选层 keys={rec['cand_keys']} 分类={rec['cand_cats']}")
    print(f"  上下文层 keys={rec['ctx_keys']}")
    print(f"  候选层禁止注入={rec['cand_forbidden']} 上下文层禁止注入={rec['ctx_forbidden']}")

    expect = rec["expect"]
    if expect == "retrieval":
        assert rec["triggered"], (
            f"[{rec['id']}] 漏触发：预期检索，实际零查询"
        )
        missing = [k for k in case["rel"] if k not in rec["cand_keys"]]
        assert not missing, (
            f"[{rec['id']}] 应检索记忆未进入候选：{missing}"
        )
        missing_ctx = [k for k in case["rel"] if k not in rec["ctx_keys"]]
        assert not missing_ctx, (
            f"[{rec['id']}] 应检索记忆未进入上下文：{missing_ctx}"
        )
    elif expect == "clarify":
        # 澄清能力缺口：单独报告（预声明 F5 不判失败）；仅 Q04 执行
        # 预声明的 F1 红线（假设场景不得注入真实 major）
        if case["id"] == "Q04":
            assert ("major" not in rec["cand_keys"]
                    and "major" not in rec["ctx_keys"]), (
                f"[Q04] 假设场景注入真实 major 记忆（预声明 F1）："
                f"候选={rec['cand_keys']} 上下文={rec['ctx_keys']}"
            )
        return
    # 通用红线（reject / unclear / retrieval 通用）：禁止记忆不得进入两层
    if rec["cand_forbidden"] or rec["ctx_forbidden"]:
        raise AssertionError(
            f"[{rec['id']}] 禁止记忆注入：候选层={rec['cand_forbidden']} "
            f"上下文层={rec['ctx_forbidden']}（预期={expect}）"
        )


# =====================================================================
# D 组：异常与降级（独立夹具，单独报告，不混入意图门准确率）
# =====================================================================

def test_d01_empty_goal_library(tmp_path, monkeypatch, capsys):
    """D01：goal 类库为空时，"我的目标是什么"应触发但空结果、无注入、无报错。"""
    corpus = [m for m in CORPUS if m[2] != "goal"]
    agent = build_agent(tmp_path, monkeypatch, corpus=corpus)
    rec = run_case(agent, {"id": "D01", "q": "我的目标是什么", "subject": "用户",
                           "expect": "scenario-empty-goal", "category": "goal",
                           "allow": [], "rel": [], "forbid": ALL_KEYS, "note": ""})
    capsys.readouterr()
    assert rec["triggered"], "D01 应触发检索"
    assert rec["cand_keys"] == [], f"D01 候选应为空：{rec['cand_keys']}"
    assert rec["ctx_keys"] == [], f"D01 上下文不应注入：{rec['ctx_keys']}"


def test_d02_retrieval_exception_degrades(tmp_path, monkeypatch, capsys):
    """D02：检索抛异常 → 记录真实异常、对话继续、上下文不注入、不伪造。"""
    agent = build_agent(tmp_path, monkeypatch)

    def broken_find(*a, **k):
        raise sqlite3.OperationalError("数据库被锁（跨层测试模拟）")

    monkeypatch.setattr(agent.memory, "find_memory", broken_find)
    agent.client = FakeClient(RUN_SCRIPT)
    result = agent.run("我的口味如何", user_id="test-user")
    out = capsys.readouterr().out
    assert "长期记忆检索失败" in out and "数据库被锁" in out
    assert result["success"] is True
    ctx_keys = context_keys_from_prompts(agent.client)
    assert ctx_keys == set(), f"D02 异常后不应注入：{ctx_keys}"
    RESULTS["D02"] = dict(id="D02", q="我的口味如何", subject="用户",
                          expect="scenario-exception", triggered=True,
                          first_rule="preference", find_log=["<broken>"],
                          cand_keys=[], cand_cats=[], ctx_keys=[],
                          cand_forbidden=[], ctx_forbidden=[], note="")


def test_d03_no_skill_category(tmp_path, monkeypatch, capsys):
    """D03：库中无 skill 类时，"我的技能有哪些"触发但空结果、无跨类补齐。"""
    corpus = [m for m in CORPUS if m[2] != "skill"]
    agent = build_agent(tmp_path, monkeypatch, corpus=corpus)
    rec = run_case(agent, {"id": "D03", "q": "我的技能有哪些", "subject": "用户",
                           "expect": "scenario-no-category",
                           "category": "skill", "allow": [], "rel": [],
                           "forbid": ALL_KEYS, "note": ""})
    capsys.readouterr()
    assert rec["triggered"], "D03 应触发检索"
    assert rec["cand_keys"] == [], f"D03 不应跨类补齐：{rec['cand_keys']}"
    assert rec["ctx_keys"] == [], f"D03 上下文不应注入：{rec['ctx_keys']}"


# =====================================================================
# 指标汇总（终端输出；不隐藏任何失败案例）
# =====================================================================

def test_zz_summary(capsys):
    capsys.readouterr()
    lines = ["", "=" * 72,
             "跨层风险测试汇总（37 条常规案例 + 3 条异常场景 = 40 条冻结清单）",
             "=" * 72]

    records = [r for r in RESULTS.values() if r["id"].startswith(("U", "T", "Q", "K", "N", "C", "H"))]
    assert len(records) == 37

    # ---- 意图门二分类（Q03/Q04=clarify、Q06=unclear 不纳入） ----
    binary = [r for r in records if r["expect"] in ("retrieval", "reject")]
    correct = 0
    for r in binary:
        expect_trigger = r["expect"] == "retrieval"
        if r["triggered"] == expect_trigger:
            correct += 1
    lines.append(f"1. 意图门准确率（适用 {len(binary)}/37，排除 clarify 2 + unclear 1）："
                 f"{correct}/{len(binary)} = {100*correct/len(binary):.1f}%")

    reject_set = [r for r in binary if r["expect"] == "reject"]
    false_trigger = [r["id"] for r in reject_set if r["triggered"]]
    lines.append(f"2. 误触发率：{len(false_trigger)}/{len(reject_set)} = "
                 f"{100*len(false_trigger)/len(reject_set):.1f}%  「{false_trigger}」")

    retrieval_set = [r for r in binary if r["expect"] == "retrieval"]
    missed = [r["id"] for r in retrieval_set if not r["triggered"]]
    lines.append(f"3. 漏触发率：{len(missed)}/{len(retrieval_set)} = "
                 f"{100*len(missed)/len(retrieval_set):.1f}%  「{missed}」")

    def inject_stats(layer_key):
        case_level = [r["id"] for r in records if r[layer_key]]
        item_total = sum(len(r["cand_keys"] if layer_key == "cand_forbidden"
                             else r["ctx_keys"]) for r in records)
        item_bad = sum(len(r[layer_key]) for r in records)
        return case_level, item_bad, item_total

    c_cases, c_items, c_total = inject_stats("cand_forbidden")
    x_cases, x_items, x_total = inject_stats("ctx_forbidden")
    lines.append(f"4. 禁止记忆进入候选层：案例级 {len(c_cases)}/37 = "
                 f"{100*len(c_cases)/37:.1f}%；条目级 {c_items}/{c_total} = "
                 f"{100*c_items/max(c_total,1):.1f}%")
    lines.append(f"5. 禁止记忆进入最终上下文：案例级 {len(x_cases)}/37 = "
                 f"{100*len(x_cases)/37:.1f}%；条目级 {x_items}/{x_total} = "
                 f"{100*x_items/max(x_total,1):.1f}%")

    nonquery = [r for r in records if r["expect"] in ("reject", "clarify", "unclear")]
    injected = [r["id"] for r in nonquery if r["ctx_forbidden"]]
    lines.append(f"6. 无关/不应注入问题的记忆注入率：{len(injected)}/{len(nonquery)} = "
                 f"{100*len(injected)/len(nonquery):.1f}%  「{injected}」")

    clarify = [r for r in records if r["expect"] == "clarify"]
    lines.append("7. 澄清案例（能力缺口单独报告）："
                 + "；".join(
                     f"{r['id']} 触发={r['triggered']} 注入真实记忆={r['ctx_forbidden'] or r['cand_forbidden'] or '无'}"
                     for r in clarify)
                 + "  ← 当前系统无澄清机制，两项均被当作普通问句处理")

    d_ids = [r["id"] for r in RESULTS.values() if r["id"].startswith("D")]
    lines.append(f"8. 异常/降级场景（{d_ids}）：空库/无类/异常三例均无崩溃、无跨类补齐、"
                 f"异常被记录且上下文不注入（详见对应测试断言）")

    lines.append("9. 按主体分组（触发/注入案例数）：")
    for subject in sorted({r["subject"] for r in records}):
        group = [r for r in records if r["subject"] == subject]
        trg = sum(1 for r in group if r["triggered"])
        inj = sum(1 for r in group if r["ctx_forbidden"])
        lines.append(f"   {subject}: n={len(group)} 触发={trg} 误注入={inj}")

    lines.append("10. 边界案例首命中类别（K03/C01/C04）：")
    for cid in ("K03", "C01", "C04"):
        r = RESULTS[cid]
        lines.append(f"   {cid}: 首命中规则={r['first_rule']} find={r['find_log']} "
                     f"候选={r['cand_keys']}")

    lines.append("=" * 72)
    print("\n".join(lines))
