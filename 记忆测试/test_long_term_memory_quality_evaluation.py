# -*- coding: utf-8 -*-
"""长期记忆检索质量评估测试集（基线测量，不修改任何生产代码）。

目的
====
在不改变检索算法的前提下，对当前实现建立可重复运行的检索质量基线：

  A. 精确事实召回（姓名/专业/职业目标/偏好/项目/技能）
  B. 中文表达变化（同一条记忆的不同问法）
  C. 多条记忆与排序（单命中/多命中/相似关键词不同事实/顺序与评分）
  D. 分类过滤与 key 查询（category 排除性 / 精确 key / 不存在 key / 去重）
  E. 无关问题与边界（数学/常识/天气/空串/停用词/无匹配）

分层原则（重要）
================
底层检索（memory.search_long_term_memory / find_memory /
get_memories_by_category）与 Agent 意图判断
（agent.retrieve_long_term_memories，含意图规则与数据库是否被查询）
分别测试、分别统计，绝不混为一谈。

预期口径
========
每条用例的预期按【当前实现的真实能力】设定（字符级打分、
意图关键词规则、min_score>=2、稳定排序），不假定未实现的语义能力：
  - tier="quality"  strict=True ：必须通过的质量断言。
  - tier="gap"      strict=False：已知能力缺口（如语义变体漏召回），
                    只记录实际行为并做安全断言（禁止召回仍生效），
                    不计入失败——缺口在总结中单独披露。
  - tier="accepted_trigger"      ：设计上接受的宽触发（如"我喜欢"），
                    记录实际召回，不计入失败。
  - tier="negative"              ：必须零查询、零召回。

指标口径（在 test_zz_quality_summary 中计算并打印）
====================================================
  召回率 Recall（微平均）= Σ|expected ∩ returned| / Σ|expected|，
      仅统计 expected 非空的用例；tier=gap 用例按实际召回如实计入。
  精确率 Precision（微平均）= Σ|returned ∩ relevant| / Σ|returned|，
      relevant = expected ∪ allowed；returned 为空的用例不进入分母；
      负向用例 relevant 为空，一旦返回即拉低精确率。
  错误分类：fail_miss（漏召回）/ fail_false_positive（误召回）/
      fail_order（排序错误）/ gap_miss（能力缺口-漏召回）/
      trigger_observed（已接受宽触发）。
  意图判断：Agent 层用例均以计数探针记录数据库是否被查询
      （intent_hit），负向用例必须全零。

数据安全
========
全部用例使用 pytest tmp_path 临时数据库 + 固定虚构语料
（不复制正式 memory.db 中的任何真实个人信息）。
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
# 固定虚构语料（按写入顺序产生 id 1..12，id 决定并列时的排序）
# =====================================================================

CORPUS = [
    ("name", "李小明", "identity"),
    ("major", "计算机科学与技术", "profile"),
    ("major_history", "本科转专业过一次", "profile"),
    ("career_goal", "机器人工程师", "goal"),
    ("sibling_goal", "帮朋友规划机器人比赛", "goal"),
    ("learning_goal", "学好检索与评估技术", "goal"),
    ("preference", "喜欢喝黑咖啡不加糖", "preference"),
    ("language_preference", "偏好用中文交流", "preference"),
    ("coffee_note", "咖啡因摄入要适量", "profile"),
    ("project", "社区图书角项目", "project"),
    ("skill", "熟悉Python基础语法", "skill"),
    ("skills", "会用Git管理代码版本", "skill"),
]

CORPUS_KEY_NAMES = [key for key, _, _ in CORPUS]


def others(*include):
    """语料中除 include 之外的所有 key（构造跨分类禁止召回列表）。"""
    return [k for k in CORPUS_KEY_NAMES if k not in include]


CROSS_CATEGORY_FORBIDDEN = [
    "name", "major", "major_history", "career_goal", "sibling_goal",
    "learning_goal", "project", "skill", "skills", "coffee_note",
]


def build_eval_agent(tmp_path, monkeypatch, corpus=None):
    """临时目录 + 临时库 + 语料（默认固定 CORPUS）；返回 agent。"""
    monkeypatch.chdir(tmp_path)
    agent = Agent(**DUMMY)
    agent.memory = Memory(db_path=str(tmp_path / "memory.db"))
    for key, value, category in (CORPUS if corpus is None else corpus):
        agent.memory.save_memory(
            key=key, value=value, category=category, reason="评估语料", user_id="test-user"
        )
    return agent


def attach_query_spy(agent):
    """给三个检索方法挂计数探针（种子写入完成后再挂，避免污染计数）。"""
    counts = {"find": 0, "search": 0, "by_cat": 0}
    real_find = agent.memory.find_memory
    real_search = agent.memory.search_long_term_memory
    real_by_cat = agent.memory.get_memories_by_category

    def spy_find(*a, **k):
        counts["find"] += 1
        return real_find(*a, **k)

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


# =====================================================================
# 结果登记（summary 用例汇总；key 为 (layer, case_id)，幂等可重跑）
# =====================================================================

RESULTS = {}


def record(layer, case, returned_keys, returned_scores, intent_hit, status,
           note=""):
    RESULTS[(layer, case["id"])] = {
        "layer": layer,
        "id": case["id"],
        "question": case.get("question", case.get("query", "")),
        "expected": case.get("expected", []),
        "allowed": case.get("allowed", []),
        "forbidden": case.get("forbidden", []),
        "returned": returned_keys,
        "scores": returned_scores,
        "intent_hit": intent_hit,
        "status": status,
        "tier": case.get("tier", "quality"),
        "note": note or case.get("reason", ""),
    }


# =====================================================================
# Agent 层评估矩阵
# 预期推导依据：
#   意图规则 agent.py:31-62（顺序 name→major→goal→preference→skill→project，
#   首条命中即停，agent.py:371-379）；规则命中后依次：
#   find_memory(规则 keys) → category 文本检索(min_score=2) →
#   category 兜底(get_memories_by_category limit=3)，按 id 去重，上限 3。
# =====================================================================

AGENT_CASES = [
    # ---- A. 精确事实召回 ----
    dict(id="A01", question="我叫什么名字？", expected=["name"],
         forbidden=others("name"), order=["name"],
         reason="name 规则精确 key 命中"),
    dict(id="A02", question="你还记得我叫什么吗？", expected=["name"],
         forbidden=others("name"),
         reason="包含关键词「我叫什么」"),
    dict(id="A03", question="我的名字是什么？", expected=["name"],
         forbidden=others("name"), reason="包含「名字」"),
    dict(id="A04", question="我是谁？", expected=["name"],
         forbidden=others("name"), reason="包含「我是谁」"),
    dict(id="A05", question="我的专业是什么？", expected=["major"],
         allowed=["major_history"],
         forbidden=others("major", "major_history"),
         order=["major", "major_history"],
         reason="major 精确 key 在前；文本检索含「专业」二字的同分类记忆在后"),
    dict(id="A07", question="你还记得我学什么的吗？", expected=["major"],
         forbidden=others("major"),
         reason="包含关键词「学什么」"),
    dict(id="A08", question="我的职业目标是什么？",
         expected=["career_goal"],
         allowed=["learning_goal"],
         forbidden=others("career_goal", "learning_goal"),
         order=["career_goal", "learning_goal"],
         reason="两个规则 key 精确命中即止；分类兜底不触发"
                "（agent.py:411 仅在两路结果为空时兜底）"),
    dict(id="A09", question="我未来想做什么工作？",
         expected=["career_goal"],
         allowed=["learning_goal", "sibling_goal"],
         forbidden=others("career_goal", "learning_goal",
                          "sibling_goal"),
         reason="包含「想做什么工作」"),
    dict(id="A11", question="我喜欢什么？", expected=["preference"],
         allowed=["language_preference"],
         forbidden=[k for k in CROSS_CATEGORY_FORBIDDEN
                    if k != "language_preference"],
         order=["preference", "language_preference"],
         reason="两个规则 key 精确命中，顺序即规则 key 顺序"),
    dict(id="A13", question="帮我推荐一些适合我的饮品", expected=["preference"],
         allowed=["language_preference"],
         forbidden=[k for k in CROSS_CATEGORY_FORBIDDEN
                    if k != "language_preference"],
         reason="包含「适合我」（第三阶段新增关键词）"),
    dict(id="A15", question="我会什么技能？", expected=["skill", "skills"],
         forbidden=[k for k in CROSS_CATEGORY_FORBIDDEN
                    if k not in ("skill", "skills")],
         order=["skill", "skills"], reason="两个规则 key 精确命中"),
    dict(id="A16", question="我擅长什么？", expected=["skill", "skills"],
         forbidden=[k for k in CROSS_CATEGORY_FORBIDDEN
                    if k not in ("skill", "skills")],
         reason="包含「擅长」"),
    dict(id="A17", question="我最近在忙什么项目？", expected=["project"],
         forbidden=others("project"), reason="包含「项目」"),
    dict(id="A18", question="我在做什么项目？", expected=["project"],
         forbidden=others("project"), reason="包含「在做什么项目」"),
    dict(id="A19", question="我想做哪方面的工作？",
         expected=["career_goal"],
         allowed=["learning_goal"],
         forbidden=others("career_goal", "learning_goal"),
         order=["career_goal", "learning_goal"],
         reason="「哪方面的工作」为 goal 规则新增同义词后必须命中"),
    dict(id="A20", question="毕业后我想做什么？", expected=["career_goal"],
         allowed=["learning_goal"],
         forbidden=others("career_goal", "learning_goal"),
         order=["career_goal", "learning_goal"],
         reason="「我想做什么」为 goal 规则新增同义词后必须命中"),
    # ---- B. 中文表达变化（意图同义词修复后必须召回）----
    dict(id="G01", question="我学的是什么方向？", expected=["major"],
         forbidden=others("major"),
         reason="「学的是」为 major 规则新增同义词后必须命中"),
    dict(id="G02", question="我以后想从事什么工作？",
         expected=["career_goal"],
         allowed=["learning_goal"],
         forbidden=others("career_goal", "learning_goal"),
         reason="「想从事」为 goal 规则新增同义词后必须命中"),
    dict(id="G03", question="我平时喜欢喝点什么？", expected=["preference"],
         allowed=["language_preference"],
         forbidden=others("preference", "language_preference"),
         reason="「喜欢喝」为 preference 规则新增同义词后必须命中"),
    # ---- E. 已接受宽触发（第三阶段设计决定，记录不判负）----
    dict(id="T01", question="我喜欢这部电影吗？", expected=[],
         allowed=["preference", "language_preference"],
         forbidden=[k for k in CROSS_CATEGORY_FORBIDDEN
                    if k not in ("preference", "language_preference")],
         tier="accepted_trigger",
         reason="含「我喜欢」→ 意图命中；返回真实偏好属已接受行为"),
    # ---- E. 无关问题：必须零查询、零召回 ----
    dict(id="N01", question="123乘以456等于多少", expected=[],
         forbidden=list(CORPUS_KEY_NAMES), tier="negative",
         reason="数学问题不得触发检索"),
    dict(id="N02", question="地球绕太阳一圈需要多久", expected=[],
         forbidden=list(CORPUS_KEY_NAMES), tier="negative"),
    dict(id="N03", question="明天会下雨吗", expected=[],
         forbidden=list(CORPUS_KEY_NAMES), tier="negative"),
    dict(id="N04", question="", expected=[],
         forbidden=list(CORPUS_KEY_NAMES), tier="negative"),
    dict(id="N05", question="   ", expected=[],
         forbidden=list(CORPUS_KEY_NAMES), tier="negative"),
    dict(id="N06", question="今天心情不错", expected=[],
         forbidden=list(CORPUS_KEY_NAMES), tier="negative"),
    dict(id="N07", question="你好", expected=[],
         forbidden=list(CORPUS_KEY_NAMES), tier="negative"),
    # ---- 语义化负向：专业 / 职业目标 / 偏好 各 2 ----
    dict(id="M-N1", question="这两门课应该怎么选？", expected=[],
         forbidden=list(CORPUS_KEY_NAMES), tier="negative",
         reason="谈选课但不涉及专业身份，任何规则都不应命中"),
    dict(id="M-N2", question="我喜欢这台电脑的配置", expected=[],
         allowed=["preference", "language_preference"],
         forbidden=[k for k in CORPUS_KEY_NAMES
                    if k not in ("preference", "language_preference")],
         tier="accepted_trigger",
         reason="「我喜欢」触发偏好检索属已接受行为；"
                "断言不得引入 major 等无关类别（类别隔离）"),
    dict(id="J-N1", question="公司今年的业绩怎么样？", expected=[],
         forbidden=list(CORPUS_KEY_NAMES), tier="negative",
         reason="谈业绩，与个人职业目标无关，不应命中 goal 规则"),
    dict(id="J-N2", question="我的职业是程序员，你的呢？", expected=[],
         forbidden=list(CORPUS_KEY_NAMES), tier="negative",
         reason="提到「职业」但既非「职业目标」也非「职业规划」，"
                "且不含「想做什么工作」——不应命中 goal 规则"),
    dict(id="P-N1", question="这家店的服务态度很好", expected=[],
         forbidden=list(CORPUS_KEY_NAMES), tier="negative",
         reason="评价服务，与个人偏好无关"),
    dict(id="P-N2", question="他喜欢的书是哪本？", expected=[],
         forbidden=list(CORPUS_KEY_NAMES), tier="negative",
         reason="「他喜欢」不是「我喜欢」，谈论第三方的喜好，"
               "不应命中 preference 规则"),
    dict(id="J-N3", question="今天工作好忙", expected=[],
         forbidden=list(CORPUS_KEY_NAMES), tier="negative",
         reason="普通工作话题，不含任何 goal 同义词"),
    dict(id="J-N4", question="他做什么工作的？", expected=[],
         forbidden=list(CORPUS_KEY_NAMES), tier="negative",
         reason="询问他人职业：「他做什么工作」不匹配带第一人称的规则词"),
    dict(id="J-N5", question="最近在追一部剧", expected=[],
         forbidden=list(CORPUS_KEY_NAMES), tier="negative",
         reason="一般闲聊，任何规则都不应命中"),
]


# =====================================================================
# Agent 层参数化评估
# =====================================================================

@pytest.mark.parametrize("case", AGENT_CASES, ids=lambda c: c["id"])
def test_agent_retrieval_quality(case, tmp_path, monkeypatch):
    agent = build_eval_agent(tmp_path, monkeypatch)
    counts = attach_query_spy(agent)

    result = agent.retrieve_long_term_memories(case["question"])

    returned_keys = [m["key"] for m in result]
    returned_scores = [
        {"key": m["key"], "score": m.get("score")} for m in result
    ]
    intent_hit = any(counts.values())

    expected = case.get("expected", [])
    allowed = case.get("allowed", [])
    forbidden = case.get("forbidden", [])
    strict = case.get("tier", "quality") == "quality"
    tier = case.get("tier", "quality")

    missing = [k for k in expected if k not in returned_keys]
    fp = [k for k in returned_keys if k in forbidden]

    # ---- 安全断言（所有层级都必须满足）：禁止召回绝不出现 ----
    assert not fp, (
        f"[{case['id']}] 误召回：{fp} 出现在结果中，实际返回 {returned_keys}"
    )

    # ---- 负向用例：零查询、零召回 ----
    if tier == "negative":
        record("agent", case, returned_keys, returned_scores, intent_hit,
               "pass" if (not returned_keys and not intent_hit)
               else "fail_negative_violation")
        assert not returned_keys, (
            f"[{case['id']}] 无关问题返回了记忆：{returned_keys}"
        )
        assert not intent_hit, (
            f"[{case['id']}] 无关问题触发了数据库查询：{counts}"
        )
        return

    # ---- 已接受宽触发：只记录 ----
    if tier == "accepted_trigger":
        status = "trigger_observed" if returned_keys else "no_trigger"
        record("agent", case, returned_keys, returned_scores, intent_hit,
               status)
        return

    # ---- 已知缺口：记录实际行为，断言只保安全 ----
    if tier == "gap":
        status = "gap_ok" if not missing else "gap_miss"
        record("agent", case, returned_keys, returned_scores, intent_hit,
               status)
        return

    # ---- 质量断言（strict）----
    order_expected = case.get("order")
    order_ok = (
        order_expected is None or returned_keys == order_expected
    )

    if missing:
        status = "fail_miss"
    elif not order_ok:
        status = "fail_order"
    else:
        status = "pass"

    record("agent", case, returned_keys, returned_scores, intent_hit, status)

    assert not missing, (
        f"[{case['id']}] 漏召回：{missing} 未返回，实际 {returned_keys}"
    )
    if order_expected is not None:
        assert returned_keys == order_expected, (
            f"[{case['id']}] 排序不符：期望 {order_expected}，"
            f"实际 {returned_keys}"
        )


# =====================================================================
# 底层检索评估矩阵（直接调用 Memory 方法，不经意图规则）
# 预期推导依据：memory.py 打分 key+2（340-343）、value+1（346-348）、
# min_score>=2（350）、稳定降序（354-357，并列保持 id 倒序行序）、
# limit 截断（359）、停用字表（267-270）。
# =====================================================================

BOTTOM_CASES = [
    dict(id="B01", query="我是什么专业的", method="search", category=None,
         expected=["major"], allowed=["major_history"], tier="gap",
         reason="已知缺口：major 的 value 无「专业」二字，字符匹配只能"
                "捞到含「专业」的 major_history（语义目标漏召回）"),
    dict(id="B02", query="我是什么专业的", method="search", category="profile",
         expected=["major"], allowed=["major_history"], tier="gap",
         reason="同 B01，限定 profile 分类"),
    dict(id="B03", query="我是什么专业的", method="search", category="goal",
         expected=[], forbidden=list(CORPUS_KEY_NAMES),
         reason="分类过滤：goal 分类无匹配"),
    dict(id="B04", query="我是什么专业的", method="search",
         category="preference", expected=[],
         forbidden=list(CORPUS_KEY_NAMES), reason="分类过滤：preference 无匹配"),
    dict(id="B05", query="喜欢咖啡", method="search", category=None,
         expected=["preference"], allowed=["coffee_note"],
         order=["preference", "coffee_note"],
         reason="评分区分：preference 得 4 分（喜欢咖啡全中）,"
                "coffee_note 得 2 分（咖啡）"),
    dict(id="B06", query="咖啡", method="search", category=None,
         expected=["preference", "coffee_note"],
         order=["coffee_note", "preference"],
         reason="并列 2 分按 id 倒序（coffee_note 写入更晚）——"
                "换一种问法排序会翻转（对照 B05），记录该基线行为"),
    dict(id="B07", query="喜欢咖啡", method="search", category=None,
         limit=1, expected=["preference"],
         reason="limit 截断取最高分"),
    dict(id="B08", query="的是了吗呢吧", method="search", category=None,
         expected=[], forbidden=list(CORPUS_KEY_NAMES),
         reason="纯停用字查询不得命中任何记忆"),
    dict(id="B09", query="", method="search", category=None,
         expected=[], forbidden=list(CORPUS_KEY_NAMES),
         reason="空字符串查询返回空列表（另在用例内断言纯空白）"),
    dict(id="B10", query="量子纠缠的原理", method="search", category=None,
         expected=[], forbidden=list(CORPUS_KEY_NAMES),
         reason="无任何匹配记忆的问题"),
    dict(id="B11", query="黑咖啡", method="search", category=None,
         expected=["preference"], allowed=["coffee_note"],
         order=["preference", "coffee_note"],
         reason="preference 3 分（黑咖啡）高于 coffee_note 2 分（咖啡）"),
    dict(id="B12", query="机器人", method="search", category=None,
         expected=["career_goal", "sibling_goal"],
         order=["sibling_goal", "career_goal"],
         reason="相似事实并列 3 分，按 id 倒序——记录基线排序"),
    dict(id="B13", query="机器人工程师", method="search", category=None,
         expected=["career_goal"], allowed=["sibling_goal"],
         order=["career_goal", "sibling_goal"],
         reason="更具体的问法拉开分差（6 分 vs 3 分），排序正确"),
    dict(id="B14", query="major", method="search", category=None,
         expected=["major", "major_history"],
         allowed=[],
         forbidden=others("major", "major_history"),
         order=["major_history", "major"],
         reason="完整词元匹配：仅真含 major 片段的 key 得分（各 10 分），"
                "project/learning_goal 等不再被单字母碰撞召回；"
                "同分按 id 倒序"),
    dict(id="B15", query="咖啡因", method="search", category=None,
         expected=["coffee_note"], allowed=["preference"],
         order=["coffee_note", "preference"],
         reason="无分类过滤时 profile 类的 coffee_note 得 3 分居首"),
    dict(id="B16", query="咖啡因", method="search", category="preference",
         expected=["preference"], forbidden=list(
             set(CORPUS_KEY_NAMES) - {"preference"}),
         reason="分类过滤把跨分类相似记忆（coffee_note）排除在外"),
    dict(id="B17", query="喜欢喝咖啡", method="search", category=None,
         min_score=1, expected=["preference"], allowed=["coffee_note"],
         reason="显式 min_score=1 恢复宽松行为（单字命中也可入选）"),
]


@pytest.mark.parametrize("case", BOTTOM_CASES, ids=lambda c: c["id"])
def test_bottom_layer_retrieval_quality(case, tmp_path, monkeypatch):
    agent = build_eval_agent(tmp_path, monkeypatch)
    memory = agent.memory

    method = case["method"]
    assert method == "search", "当前矩阵仅覆盖 search；find/by_cat 见下方专测"

    kwargs = {"limit": case.get("limit", 5),
              "category": case.get("category"),
              "min_score": case.get("min_score", 2),
              "user_id": "test-user"}
    results = memory.search_long_term_memory(case["query"], **kwargs)

    returned_keys = [m["key"] for m in results]
    returned_scores = [{"key": m["key"], "score": m["score"]}
                       for m in results]

    expected = case.get("expected", [])
    allowed = case.get("allowed", [])
    forbidden = case.get("forbidden", [])
    strict = case.get("tier", "quality") == "quality"
    tier = case.get("tier", "quality")

    missing = [k for k in expected if k not in returned_keys]
    fp = [k for k in returned_keys if k in forbidden]

    assert not fp, (
        f"[{case['id']}] 误召回：{fp} 出现在结果中，实际返回 {returned_keys}"
    )

    if tier == "gap":
        record("bottom", case, returned_keys, returned_scores, True,
               "gap_ok" if not missing else "gap_miss")
        return

    order_expected = case.get("order")
    order_ok = (
        order_expected is None or returned_keys == order_expected
    )

    if missing:
        status = "fail_miss"
    elif not order_ok:
        status = "fail_order"
    else:
        status = "pass"

    record("bottom", case, returned_keys, returned_scores, True, status)

    assert not missing, (
        f"[{case['id']}] 漏召回：{missing} 未返回，实际 {returned_keys}"
    )
    if order_expected is not None:
        assert returned_keys == order_expected, (
            f"[{case['id']}] 排序不符：期望 {order_expected}，"
            f"实际 {returned_keys}"
        )


# =====================================================================
# 底层专测：精确 key 查询 / 分类兜底 / 去重
# =====================================================================

def test_bottom_find_memory_exact_and_missing(tmp_path, monkeypatch):
    agent = build_eval_agent(tmp_path, monkeypatch)
    memory = agent.memory

    hit = memory.find_memory("major", user_id="test-user")
    assert hit is not None and hit["key"] == "major"
    assert hit["value"] == "计算机科学与技术"
    assert hit["category"] == "profile"

    assert memory.find_memory("no_such_key", user_id="test-user") is None
    assert memory.find_memory("", user_id="test-user") is None
    assert memory.find_memory("name", user_id="test-user")["value"] == "李小明"

    record("bottom", {"id": "F01", "question": "find_memory 精确/缺失",
                      "allowed": ["major", "name"]},
           ["major", "name"], [], True, "pass",
           note="精确 key 命中；不存在的 key 与空 key 返回 None")


def test_bottom_category_fallback_order_and_limit(tmp_path, monkeypatch):
    agent = build_eval_agent(tmp_path, monkeypatch)
    memory = agent.memory

    rows = memory.get_memories_by_category("goal", limit=2, user_id="test-user")
    assert [r["key"] for r in rows] == ["learning_goal", "sibling_goal"]

    rows_all = memory.get_memories_by_category("goal", limit=10, user_id="test-user")
    assert [r["key"] for r in rows_all] == [
        "learning_goal", "sibling_goal", "career_goal",
    ]

    assert memory.get_memories_by_category("no_such_category", user_id="test-user") == []

    record("bottom", {"id": "F02", "question": "get_memories_by_category",
                      "allowed": ["learning_goal", "sibling_goal",
                                  "career_goal"]},
           [r["key"] for r in rows_all], [], True, "pass",
           note="分类兜底按 id 倒序、受 limit 约束、未知分类返回空")


def test_bottom_search_no_duplicate_keys_for_same_memory(tmp_path,
                                                         monkeypatch):
    """同一条记忆不会在单次 search 结果中重复出现（结果来自同一游标
    遍历，每行至多产生一条记录——此为结构保证，用实际结果复核）。"""
    agent = build_eval_agent(tmp_path, monkeypatch)
    results = agent.memory.search_long_term_memory("喜欢咖啡", limit=100, user_id="test-user")
    keys = [m["key"] for m in results]
    assert len(keys) == len(set(keys))

    record("bottom", {"id": "F03", "question": "search 单次结果去重",
                      "allowed": ["preference", "coffee_note"]},
           keys, [{"key": m["key"], "score": m["score"]} for m in results],
           True, "pass", note="单次检索结果内 key 无重复")


def test_agent_level_dedup_between_paths(tmp_path, monkeypatch):
    """Agent 层去重：find_memory 与 category 文本检索命中同一 id 时只保留一条。"""
    agent = build_eval_agent(tmp_path, monkeypatch)

    results = agent.retrieve_long_term_memories("我的专业是什么")

    keys = [m["key"] for m in results]
    ids = [m["id"] for m in results]
    assert len(ids) == len(set(ids)), f"按 id 去重失败：{results}"
    assert keys.count("major") == 1

    record("agent", {"id": "D01", "question": "我的专业是什么",
                     "allowed": ["major", "major_history"]},
           keys, [{"key": m["key"], "score": m.get("score")}
                  for m in results],
           True, "pass", note="多路径命中同一记忆按 id 去重")


def test_agent_level_result_cap_three(tmp_path, monkeypatch):
    """Agent 层注入上限 3 条（goal 分类兜底最多 3 条）。"""
    agent = build_eval_agent(tmp_path, monkeypatch)

    results = agent.retrieve_long_term_memories("我的职业目标")

    assert len(results) <= 3
    record("agent", {"id": "D02", "question": "我的职业目标",
                     "allowed": ["career_goal", "learning_goal"]},
           [m["key"] for m in results],
           [{"key": m["key"], "score": m.get("score")} for m in results],
           True, "pass", note="返回数量上限 3 生效")


# =====================================================================
# C 系列：分类兜底补齐（先以测试定义预期行为，再改生产代码）
# 设计语义：规则 key 命中不足（len(results) < len(keys)，即确有预期
# 记忆未命中）时，用【同分类】最近记忆补足，按 id 去重、追加在已有
# 结果之后，总数仍以 3 为上限；key 全命中时不补齐（避免同分类噪声）。
# =====================================================================

def test_c01_complement_fills_when_key_missing(tmp_path, monkeypatch):
    """C01 少量命中时补齐：goal 两个 key 只命中一个，兜底补入同分类
    的 sibling_goal；已有结果在前。"""
    agent = build_eval_agent(
        tmp_path, monkeypatch,
        corpus=[
            ("career_goal", "机器人工程师", "goal"),
            ("sibling_goal", "帮朋友规划机器人比赛", "goal"),
        ],
    )
    results = agent.retrieve_long_term_memories("我的职业目标是什么？")
    keys = [m["key"] for m in results]
    ids = [m["id"] for m in results]

    assert keys == ["career_goal", "sibling_goal"], (
        f"补齐未生效或顺序错误：{keys}"
    )
    assert len(ids) == len(set(ids)), "补齐引入重复记忆"

    record("agent", {"id": "C01", "question": "我的职业目标是什么？",
                     "expected": ["career_goal"],
                     "allowed": ["sibling_goal"],
                     "forbidden": []},
           keys, [{"key": m["key"], "score": m.get("score")}
                  for m in results], True, "pass",
           note="key 缺失时补齐同分类记忆，已有结果在前")


def test_c02_complement_respects_cap_and_category(tmp_path, monkeypatch):
    """C02 结果上限与类别隔离（补齐路径）：goal 两 key 均未命中时，
    只补 goal 分类最新 3 条，绝不带入其他分类。"""
    corpus = [(f"goal_{i}", f"条目{i}", "goal") for i in range(5)]
    corpus += [("preference", "喜欢喝黑咖啡", "preference"),
               ("name", "李小明", "identity")]
    agent = build_eval_agent(tmp_path, monkeypatch, corpus=corpus)

    results = agent.retrieve_long_term_memories("我的职业目标是什么？")
    keys = [m["key"] for m in results]

    assert len(keys) == 3, f"补齐后应恰好达到上限 3，实际 {len(keys)}"
    assert keys == ["goal_4", "goal_3", "goal_2"], f"顺序不符：{keys}"
    assert all(m["category"] == "goal" for m in results), "引入无关分类"

    record("agent", {"id": "C02", "question": "我的职业目标是什么？",
                     "expected": ["goal_4", "goal_3", "goal_2"],
                     "allowed": [],
                     "forbidden": ["preference", "name"]},
           keys, [{"key": m["key"], "score": m.get("score")}
                  for m in results], True, "pass",
           note="补齐受上限 3 约束，只补 goal 分类")


def test_c03_complement_isolates_category(tmp_path, monkeypatch):
    """C03 类别隔离（补齐路径）：skill 规则缺一个 key 时只补 skill
    分类，不得把 preference/name 等无关分类带入。"""
    agent = build_eval_agent(
        tmp_path, monkeypatch,
        corpus=[
            ("skill", "熟悉Python基础语法", "skill"),
            ("git_skill", "会用Git管理代码版本", "skill"),
            ("preference", "喜欢喝黑咖啡", "preference"),
            ("name", "李小明", "identity"),
        ],
    )
    results = agent.retrieve_long_term_memories("我会什么技能？")
    keys = [m["key"] for m in results]

    assert keys == ["skill", "git_skill"], f"补齐/隔离错误：{keys}"
    assert all(m["category"] == "skill" for m in results), "引入无关分类"

    record("agent", {"id": "C03", "question": "我会什么技能？",
                     "expected": ["skill"], "allowed": ["git_skill"],
                     "forbidden": ["preference", "name"]},
           keys, [{"key": m["key"], "score": m.get("score")}
                  for m in results], True, "pass",
           note="补齐只发生在同分类内")


def test_c04_complement_without_effective_source(tmp_path, monkeypatch):
    """C04 无有效补充：兜底来源只有已在结果中的记忆时，
    正常返回已有结果，不报错、不重复。"""
    agent = build_eval_agent(
        tmp_path, monkeypatch,
        corpus=[("career_goal", "机器人工程师", "goal")],
    )
    results = agent.retrieve_long_term_memories("我的职业目标是什么？")
    keys = [m["key"] for m in results]

    assert keys == ["career_goal"], f"无有效补充时应原样返回：{keys}"

    record("agent", {"id": "C04", "question": "我的职业目标是什么？",
                     "expected": ["career_goal"], "allowed": [],
                     "forbidden": []},
           keys, [{"key": m["key"], "score": m.get("score")}
                  for m in results], True, "pass",
           note="兜底无新记忆时原样返回，不报错")


def test_c05_complement_preserves_existing_order(tmp_path, monkeypatch):
    """C05 排序稳定性：补齐只追加，不重排已有结果。"""
    agent = build_eval_agent(
        tmp_path, monkeypatch,
        corpus=[
            ("preference", "喜欢喝黑咖啡不加糖", "preference"),
            ("coffee_note", "咖啡因摄入要适量", "preference"),
        ],
    )
    results = agent.retrieve_long_term_memories("我喜欢什么？")
    keys = [m["key"] for m in results]

    assert keys == ["preference", "coffee_note"], (
        f"已有结果顺序被改变或补齐失败：{keys}"
    )

    record("agent", {"id": "C05", "question": "我喜欢什么？",
                     "expected": ["preference"],
                     "allowed": ["coffee_note"], "forbidden": []},
           keys, [{"key": m["key"], "score": m.get("score")}
                  for m in results], True, "pass",
           note="补齐追加在已有结果之后")


# =====================================================================
# 汇总评估报告（放在文件末尾，pytest 按定义顺序执行）
# =====================================================================

def test_zz_quality_summary():
    assert RESULTS, "未收集到任何评估结果——参数化用例未运行？"

    lines = ["", "=" * 72, "长期记忆检索质量评估基线报告", "=" * 72]

    for layer in ("agent", "bottom"):
        entries = [r for r in RESULTS.values() if r["layer"] == layer]
        if not entries:
            continue
        lines.append(f"\n--- {layer} 层 ---")
        for r in sorted(entries, key=lambda x: x["id"]):
            scores = " ".join(
                f"{s['key']}={s['score']}" for s in r["scores"]
                if s["score"] is not None
            )
            lines.append(
                f"[{r['status']:^20}] {r['id']} 「{r['question']}」"
                f" → {r['returned']}"
                + (f" 评分: {scores}" if scores else "")
                + (f" ｜{r['note']}" if r["note"] else "")
            )

        # ---- 指标计算（口径见模块 docstring）----
        recall_num = recall_den = 0
        prec_num = prec_den = 0
        for r in entries:
            returned = set(r["returned"])
            expected = set(r["expected"])
            if expected:
                recall_num += len(returned & expected)
                recall_den += len(expected)
            if returned:
                relevant = set(r["expected"]) | set(r["allowed"])
                prec_num += len(returned & relevant)
                prec_den += len(returned)

        recall = recall_num / recall_den if recall_den else float("nan")
        precision = prec_num / prec_den if prec_den else float("nan")

        statuses = [r["status"] for r in entries]
        lines.append(f"\n用例总数: {len(entries)}")
        lines.append(f"通过(pass): {statuses.count('pass')}")
        lines.append(f"失败(fail_*): "
                     f"{sum(1 for s in statuses if s.startswith('fail'))}")
        lines.append(f"漏召回 fail_miss: {statuses.count('fail_miss')}")
        lines.append(
            f"误召回 fail_false_positive: "
            f"{statuses.count('fail_false_positive')}"
        )
        lines.append(f"排序错误 fail_order: {statuses.count('fail_order')}")
        lines.append(
            f"能力缺口 gap_miss: {statuses.count('gap_miss')} "
            f"（已知设计限制，不计入失败）"
        )
        if layer == "agent":
            lines.append(
                f"已接受宽触发 trigger_observed: "
                f"{statuses.count('trigger_observed')}"
            )
            intent_miss = sum(
                1 for r in entries
                if r["expected"] and not r["intent_hit"]
            )
            lines.append(f"意图未命中(有预期但零查询): {intent_miss}")
        lines.append(
            f"召回率 Recall（微平均, Σ|E∩R|/Σ|E|, 含缺口用例实际值）: "
            f"{recall_num}/{recall_den} = {recall:.2%}"
        )
        lines.append(
            f"精确率 Precision（微平均, Σ|R∩rel|/Σ|R|, "
            f"relevant=expected∪allowed, 仅 returned 非空用例进分母）: "
            f"{prec_num}/{prec_den} = {precision:.2%}"
        )

    lines.append("=" * 72)
    report = "\n".join(lines)
    print(report)

    # ---- 总断言：不允许任何严格失败 ----
    failures = [r for r in RESULTS.values()
                if r["status"].startswith("fail")]
    assert not failures, (
        f"存在 {len(failures)} 项质量断言失败：\n"
        + "\n".join(
            f"  [{r['layer']}/{r['id']}] {r['status']} 期望={r['expected']} "
            f"实际={r['returned']}"
            for r in failures
        )
    )
