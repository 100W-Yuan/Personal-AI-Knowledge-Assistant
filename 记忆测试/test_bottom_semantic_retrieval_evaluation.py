# -*- coding: utf-8 -*-
"""底层检索语义盲区专项评估（只读评估，不修改任何生产算法）。

与现有测试的关系（避免重复定义）：
  - 测试/test_long_term_memory_quality_evaluation.py 的 B01/B02 已记录
    "专业语义盲区"（major 的 value 无「专业」二字）的单点现象；
    本文件 A 系列（问题 1 与 B01 属同一盲区）在【用户指定的语料】下
    重建完整 5 问双层矩阵，并额外覆盖 B01 未含的 Q4/Q5 两问。
  - B14 已初步观察英文 key 单字母碰撞；本文件 B 系列用固定 4 个英文
    key 的对照语料给出三组查询的完整得分与排序。
  - C 系列为新增的"相关/部分相关/无关混合排名"场景。

评估对象（当前真实实现，本轮零修改）：
  memory.py:272-362  search_long_term_memory
      - 打分：查询字符逐字计数，命中 key 每字 +2（341-343），
        命中 value 每字 +1（346-348）；停用字先剔除（267-270,317-321）
      - 阈值：score >= min_score（默认 2，memory.py:350）
      - 排序：按 score 稳定降序（354-357），并列保持行序（id 倒序）
      - 上限：limit 截断（默认 5，返回 359）
  memory.py:217  find_memory（精确 key）；memory.py:364 分类兜底
  agent.py:357 retrieve_long_term_memories（意图规则 → 双层结果区分）

指标口径（附分子/分母，本文件 test_zz_summary 逐项打印）：
  底层召回：目标记忆在底层返回中的比例；底层精确：底层返回中
  属于 relevant 集合的比例。
  Agent 召回/精确同理，针对 Agent 层返回。
  两层分别统计，绝不混算；意图是否触发用计数探针实测。

数据安全：全部用例使用 tmp_path 临时库；不触碰正式 memory.db。
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent import Agent  # noqa: E402
from memory import Memory  # noqa: E402

DUMMY = dict(api_key="dummy", base_url="https://example.invalid",
             model="dummy")

# =====================================================================
# 语料（用户指定 + 少量干扰项，全部虚构）
# =====================================================================

# A 系列语料：三个目标记忆 + 三个同库干扰记忆
CORPUS_A = [
    ("major", "计算机科学与技术", "profile"),        # id=1
    ("career_goal", "大模型落地工程师", "goal"),      # id=2
    ("preference", "喜欢健身", "preference"),         # id=3
    ("learning_goal", "学好检索与评估技术", "goal"),   # id=4
    ("project", "社区图书角项目", "project"),         # id=5
    ("coffee_note", "咖啡因摄入要适量", "profile"),   # id=6
]

# B 系列语料：相互接近的英文 key，值均为中文
CORPUS_B = [
    ("major", "计算机科学与技术", "profile"),              # id=1
    ("project", "社区图书角项目", "project"),              # id=2
    ("learning_goal", "学好检索与评估技术", "goal"),       # id=3
    ("language_preference", "偏好用中文交流", "preference"),  # id=4
]

# C 系列语料：一个明确相关 + 两个部分匹配 + 两个无关
CORPUS_C = [
    ("preference", "喜欢喝咖啡", "preference"),     # id=1 相关
    ("note1", "咖啡店打卡记录", "profile"),          # id=2 部分匹配
    ("note2", "每天散步三十分钟", "profile"),        # id=3 无关
    ("note3", "咖啡因敏感", "profile"),              # id=4 部分匹配
    ("note4", "每周读一本书", "profile"),            # id=5 无关
]


def build_agent(tmp_path, monkeypatch, corpus):
    monkeypatch.chdir(tmp_path)
    agent = Agent(**DUMMY)
    agent.memory = Memory(db_path=str(tmp_path / "memory.db"))
    for key, value, category in corpus:
        agent.memory.save_memory(
            key=key, value=value, category=category, reason="评估语料", user_id="test-user"
        )
    return agent


def spy_queries(agent):
    counts = {"find": 0, "search": 0, "by_cat": 0}
    method_to_key = {
        "find_memory": "find",
        "search_long_term_memory": "search",
        "get_memories_by_category": "by_cat",
    }
    real = {name: getattr(agent.memory, name) for name in method_to_key}

    def make(name):
        key = method_to_key[name]

        def wrapper(*a, **k):
            counts[key] += 1
            return real[name](*a, **k)
        return wrapper

    agent.memory.find_memory = make("find_memory")
    agent.memory.search_long_term_memory = make("search_long_term_memory")
    agent.memory.get_memories_by_category = make("get_memories_by_category")
    return counts


RESULTS = {}


def record(section, case_id, question, layer, returned, scores,
           targets, relevant, intent_hit=None, note=""):
    RESULTS[(section, case_id, layer)] = {
        "section": section, "id": case_id, "question": question,
        "layer": layer, "returned": returned, "scores": scores,
        "targets": targets, "relevant": relevant,
        "intent_hit": intent_hit, "note": note,
    }


def keys_scores(results):
    return [m["key"] for m in results], [
        {"key": m["key"], "score": m.get("score")} for m in results
    ]


# =====================================================================
# A. 中文语义表达差异（5 问 × 底层/Agent 双层）
# 预期依据：底层为字符匹配（无同义/语义能力）；Agent 层另叠加意图规则
# =====================================================================

A_QUESTIONS = [
    ("A1", "我是什么专业的？", "major"),
    ("A2", "我学的是什么方向？", "major"),
    ("A3", "我以后想从事什么工作？", "career_goal"),
    ("A4", "我想做哪方面的工作？", "career_goal"),
    ("A5", "我平时喜欢做什么？", "preference"),
]


@pytest.mark.parametrize("case_id,q,target", A_QUESTIONS,
                         ids=[c[0] for c in A_QUESTIONS])
def test_a_chinese_semantic_matrix(case_id, q, target, tmp_path, monkeypatch):
    agent = build_agent(tmp_path, monkeypatch, CORPUS_A)
    counts = spy_queries(agent)

    # ---- 底层检索（无分类过滤，与 Agent 检索条件一致）----
    bottom = agent.memory.search_long_term_memory(q, user_id="test-user")
    b_keys, b_scores = keys_scores(bottom)

    # ---- Agent 层检索（意图触发与否只看 Agent 调用自身产生的查询增量，
    #      避免被上一条底层直查的计数污染）----
    counts_before = dict(counts)
    agent_results = agent.retrieve_long_term_memories(q)
    a_keys, a_scores = keys_scores(agent_results)
    intent_hit = any(
        counts[k] - counts_before[k] for k in counts
    )

    # 相关集合定义：目标记忆 + 同分类合理邻接（learning_goal 与目标问题
    # 同属 goal，是规则的第二个精确 key，计为 relevant）
    relevant = {target}
    if target == "career_goal":
        relevant.add("learning_goal")

    record("A", case_id, q, "bottom", b_keys, b_scores,
           [target], sorted(relevant),
           note="字符匹配层：无语义能力" if target not in b_keys else "")
    record("A", case_id, q, "agent", a_keys, a_scores,
           [target], sorted(relevant), intent_hit=intent_hit,
           note="" if intent_hit else "意图规则未命中，未查库")

    # ---- 确定性断言（当前实现的真实行为，作为基线）----
    if case_id == "A1":
        assert b_keys == [], f"A1 底层应为空（value 无「专业」字符）：{b_keys}"
    if case_id == "A2":
        assert b_keys == [], f"A2 底层应为空（字符无交集）：{b_keys}"
    if case_id == "A3":
        assert b_keys == [], f"A3 底层应为空（仅「工」单字 1 分被阈值拦下）"
    if case_id == "A4":
        assert b_keys == []
    if case_id == "A5":
        assert b_keys == ["preference"], (
            f"A5 底层应命中 preference（喜/欢 2 分）：{b_keys}"
        )
        assert b_scores == [{"key": "preference", "score": 2}]

    # Agent 层行为断言
    if case_id in ("A1", "A2"):
        assert a_keys == ["major"] and intent_hit
    if case_id == "A3":
        assert a_keys == ["career_goal", "learning_goal"] and intent_hit
    if case_id == "A4":
        assert a_keys == ["career_goal", "learning_goal"] and intent_hit, (
            "A4：「哪方面的工作」为 goal 规则新增同义词后必须命中"
        )
    if case_id == "A5":
        assert a_keys == [] and not intent_hit, (
            "A5：底层能命中但意图规则未覆盖 → Agent 未查库（双层分歧）"
        )


# =====================================================================
# B. 英文字符误匹配（3 组查询 + 单字符守卫，固定 4 key 对照）
# 本轮修复目标：完整词元匹配后，无关 key 不应再被单字母碰撞召回。
# =====================================================================

B_QUERIES = [
    # 完整词元匹配：只有整段英文片段真正出现在 key 中才计分，
    # 真匹配的得分与旧实现一致（2×词长），无关 key 不再入选。
    ("B1", "major", ["major"], ["major"], [("major", 10)]),
    ("B2", "project", ["project"], ["project"], [("project", 14)]),
    ("B3", "goal", ["learning_goal"], ["learning_goal"],
     [("learning_goal", 8)]),
]


@pytest.mark.parametrize("case_id,q,relevant,order,expect_scores",
                         B_QUERIES, ids=[c[0] for c in B_QUERIES])
def test_b_english_key_collision(case_id, q, relevant, order, expect_scores,
                                 tmp_path, monkeypatch):
    agent = build_agent(tmp_path, monkeypatch, CORPUS_B)
    results = agent.memory.search_long_term_memory(q, user_id="test-user")
    keys, scores = keys_scores(results)

    record("B", case_id, q, "bottom", keys, scores,
           list(relevant), list(relevant),
           note="完整词元匹配：无关 key 不再因单字母重合入选")

    assert keys == order, f"[{case_id}] 顺序不符：{keys}"
    assert [(s["key"], s["score"]) for s in scores] == expect_scores, (
        f"[{case_id}] 得分不符：{scores}"
    )
    # 精确率实测：4 条返回中仅 1 条相关
    assert len([k for k in keys if k in relevant]) == 1


def test_b_single_ascii_char_produces_no_score(tmp_path, monkeypatch):
    """B4：单个 ASCII 字符（如 "o"）证据过弱，不再计分。
    旧实现下查询 "o" 会命中 project/learning_goal/major 三条。"""
    agent = build_agent(tmp_path, monkeypatch, CORPUS_B)
    results = agent.memory.search_long_term_memory("o", user_id="test-user")
    keys, scores = keys_scores(results)

    record("B", "B4", "o", "bottom", keys, scores, [], [],
           note="单字符 ASCII 不计分（旧实现会召回 3 条）")

    assert results == [], f"单字符不应召回任何记忆：{keys}"


# =====================================================================
# C. 排序与精确率（相关 / 部分匹配 / 无关 混合）
# =====================================================================

def test_c_relevant_ranked_last_in_score_tie(tmp_path, monkeypatch):
    """C1：三条并列 2 分（相关 + 两条部分匹配）。

    【限制记录】三条候选的 score 与 key 命中数完全相同——现有特征
    没有任何信息可以区分"真相关"与"字符碰巧命中"，因此相关记忆
    无法自动靠前（需语义能力，见 C 方案，超出本轮范围）。
    本测试断言：① 排序确定；② 同一输入与数据重复执行结果稳定。"""
    agent = build_agent(tmp_path, monkeypatch, CORPUS_C)
    results = agent.memory.search_long_term_memory("咖啡", user_id="test-user")
    results_again = agent.memory.search_long_term_memory("咖啡", user_id="test-user")
    keys, scores = keys_scores(results)
    keys_again = [m["key"] for m in results_again]

    record("C", "C1", "咖啡", "bottom", keys, scores,
           ["preference"], ["preference", "note1", "note3"],
           note="三者完全同分同 key 证据：无法区分，按 id 倒序；重复执行稳定")

    assert keys == ["note3", "note1", "preference"], f"顺序不符：{keys}"
    assert all(s["score"] == 2 for s in scores)
    assert keys == keys_again, "同一输入重复执行排序不稳定"


def test_c_limit_can_drop_relevant_memory(tmp_path, monkeypatch):
    """C2：limit=2 时，并列组中相关记忆直接被截断丢失。

    【限制记录】三条候选完全同分且无任何区分信号，上限截断无法
    保证保留"相关"记忆——这是当前字符匹配算法的固有限制，如实记录。"""
    agent = build_agent(tmp_path, monkeypatch, CORPUS_C)
    results = agent.memory.search_long_term_memory("咖啡", limit=2, user_id="test-user")
    keys, scores = keys_scores(results)

    record("C", "C2", "咖啡(limit=2)", "bottom", keys, scores,
           ["preference"], ["preference", "note1", "note3"],
           note="limit 截断把相关记忆挤出结果")

    assert keys == ["note3", "note1"]
    assert "preference" not in keys


def test_c_category_filter_recovers_relevant(tmp_path, monkeypatch):
    """C3：加 category=preference 过滤后，仅返回相关记忆。"""
    agent = build_agent(tmp_path, monkeypatch, CORPUS_C)
    results = agent.memory.search_long_term_memory(
        "咖啡", category="preference", user_id="test-user"
    )
    keys, scores = keys_scores(results)

    record("C", "C3", "咖啡(category=preference)", "bottom", keys, scores,
           ["preference"], ["preference"],
           note="分类过滤消除并列噪声")

    assert keys == ["preference"]


def test_c_key_hit_priority_breaks_ties(tmp_path, monkeypatch):
    """C4：同分时 key 命中数更多者优先（key 证据比 value 字符偶合更
    可靠）。构造（词元匹配下）：查询 "ab cd" 两个词元——
    ab_cd 的 key 同时命中两个词元（kh=2 → 4+4=8 分）；
    ab 的 key 只命中一个、另一词元在 value 命中（kh=1 → 4+4=8 分）。
    两者同分，key 证据更强者排前；且该序与写入行序相反，排除巧合。"""
    agent = build_agent(tmp_path, monkeypatch, corpus=[
        ("ab_cd", "记录", "profile"),      # id=1: kh=2 → 8 分
        ("ab", "ab cd 小记", "profile"),   # id=2: kh=1 → 4+4=8 分
    ])
    results = agent.memory.search_long_term_memory("ab cd", user_id="test-user")
    keys, scores = keys_scores(results)

    record("C", "C4", "ab cd", "bottom", keys, scores,
           ["ab_cd"], ["ab_cd", "ab"],
           note="同分时 key 命中数优先（词元匹配下的构造）")

    assert [(s["key"], s["score"]) for s in scores] == [
        ("ab_cd", 8), ("ab", 8)
    ], f"得分不符：{scores}"
    assert keys == ["ab_cd", "ab"], (
        f"同分时 key 命中数更多者应排前（行序为 ab→ab_cd）：{keys}"
    )


# =====================================================================
# 汇总：分子/分母逐项打印
# =====================================================================

def test_zz_summary():
    assert RESULTS, "未收集到评估结果"

    lines = ["", "=" * 74, "底层检索语义盲区评估报告", "=" * 74]

    for section, title in (("A", "A. 中文语义表达差异（双层）"),
                           ("B", "B. 英文字符误匹配"),
                           ("C", "C. 排序与精确率")):
        entries = [r for r in RESULTS.values() if r["section"] == section]
        lines.append(f"\n--- {title} ---")
        for r in sorted(entries, key=lambda x: (x["id"], x["layer"])):
            scores = " ".join(
                f"{s['key']}={s['score']}" for s in r["scores"]
                if s["score"] is not None
            )
            hit = [t for t in r["targets"] if t in r["returned"]]
            lines.append(
                f"[{r['layer']:^6}] {r['id']} 「{r['question']}」"
                f" → {r['returned']}"
                + (f" 评分: {scores}" if scores else "")
                + f" | 目标命中: {hit}/{r['targets']}"
                + (f" | 意图: {'触发' if r['intent_hit'] else '未触发'}"
                   if r["layer"] == "agent" else "")
                + (f" ｜{r['note']}" if r["note"] else "")
            )

    # ---- 指标（分子/分母）----
    lines.append("\n--- 指标（逐项附分子/分母）---")
    for layer in ("bottom", "agent"):
        entries = [r for r in RESULTS.values()
                   if r["section"] == "A" and r["layer"] == layer]
        if not entries:
            continue
        rec_num = sum(len(set(r["targets"]) & set(r["returned"]))
                      for r in entries)
        rec_den = sum(len(r["targets"]) for r in entries)
        prec_num = sum(len(set(r["returned"]) & set(r["relevant"]))
                       for r in entries)
        prec_den = sum(len(r["returned"]) for r in entries)
        missing = [r["id"] for r in entries
                   if not (set(r["targets"]) & set(r["returned"]))]
        lines.append(
            f"A 系列 {layer} 层：召回 {rec_num}/{rec_den}"
            f"（漏召回用例：{missing or '无'}）；"
            f"精确 {prec_num}/{prec_den}"
        )

    b_entries = [r for r in RESULTS.values() if r["section"] == "B"]
    rec_num = sum(len(set(r["targets"]) & set(r["returned"]))
                  for r in b_entries)
    rec_den = sum(len(r["targets"]) for r in b_entries)
    prec_num = sum(len(set(r["returned"]) & set(r["relevant"]))
                   for r in b_entries)
    prec_den = sum(len(r["returned"]) for r in b_entries)
    lines.append(
        f"B 系列 底层：召回 {rec_num}/{rec_den}（目标记忆均被找到）；"
        f"精确 {prec_num}/{prec_den}（完整词元匹配后无跨 key 碰撞）"
    )

    c_entries = [r for r in RESULTS.values() if r["section"] == "C"]
    for r in c_entries:
        rank = (r["returned"].index(r["targets"][0]) + 1
                if r["targets"][0] in r["returned"] else "未召回")
        lines.append(f"C 系列 {r['id']}：相关记忆排名 {rank}，"
                     f"返回 {r['returned']}")

    lines.append("=" * 74)
    print("\n".join(lines))
