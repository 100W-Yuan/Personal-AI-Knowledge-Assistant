# -*- coding: utf-8 -*-
"""长期记忆检索质量专项（RAG 8.21 轮）：Reasoning 上下文注入与数量限制隔离。

三个测试：
1. 正常回忆：Planner 与真实 Reasoning 请求的"用户长期记忆"段均含唯一测试值；
2. 虚构角色：零查询、工具未执行、执行历史无泄漏、全部 Reasoning 请求不含值；
3. 数量限制：主检索 ≤3、工具检索 ≤5、工具路径不调用 find_memory 兜底，
   并记录 min_score 参数差异（不断定为缺陷）。

复用 test_memory_quality_phase7 的 FakeClient / attach_spies / RUN_SCRIPT；
不复制其大段 helper。请求识别规则（源码依据）：
- Planner 请求：content 含"你是一个 AI Agent 的 Planner"；
- Reasoning 请求：content 含"你是一个 AI Agent。"且不含"的 Planner"
  （步骤式 :1936 与收尾式 :1479 同模板，均属推理上下文）。

全程临时数据库 + 唯一虚构值；零真实 API / 零正式数据库。
"""

import io
import contextlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MEMORY_TESTS_DIR = PROJECT_ROOT / "记忆测试"
for path in (str(PROJECT_ROOT), str(MEMORY_TESTS_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

import test_memory_quality_phase7 as p7      # noqa: E402  复用既有 helper

import agent as agent_module                  # noqa: E402
from agent import Agent                       # noqa: E402
from memory import Memory                     # noqa: E402

TEST_VALUE = "TEST_MEMORY_VALUE_821"
MEMORY_MARKER = "- career_goal = " + TEST_VALUE

PLANNER_MARK = "你是一个 AI Agent 的 Planner"
REASONING_MARK = "你是一个 AI Agent。"


def is_planner_request(messages):
    return PLANNER_MARK in messages[0]["content"]


def is_reasoning_request(messages):
    content = messages[0]["content"]
    return REASONING_MARK in content and "的 Planner" not in content


LONG_TERM_HEADER = "用户长期记忆（检索到的历史事实"
PLAN_HEADER = "当前执行计划"


def in_long_term_section(content):
    """值是否位于"用户长期记忆"段内（区域定位，非任意位置匹配）。"""
    start = content.find(LONG_TERM_HEADER)
    end = content.find(PLAN_HEADER)
    if start == -1 or end == -1 or end <= start:
        return False
    return MEMORY_MARKER in content[start:end]


def classify_reasoning(messages):
    """区分步骤式与收尾式 Reasoning（源码依据，非数组索引）：

    两者共用同一 prompt 模板；区别在"当前执行计划"段的状态——
    步骤式执行时当前步仍为 pending（agent.py:1936 调用、:1944 才置
    completed）；收尾式（:1479）在所有步骤完成后触发，计划段无 pending。
    """
    content = messages[0]["content"]
    if not (REASONING_MARK in content and "的 Planner" not in content):
        return None
    return "step" if '"status": "pending"' in content else "final"


def unique_build_agent(tmp_path, monkeypatch):
    """临时库 + 唯一测试值（复用 phase7 的构造模式，语料不同）。

    注意：search_memory 工具闭包持有 Agent 构造时的 memory 实例；
    替换 agent.memory 后两者是不同对象、但指向同一临时库文件。
    因此保留 `_closure_memory` 引用，供 spy 同时包装。
    """
    monkeypatch.chdir(tmp_path)
    a = Agent(api_key="d", base_url="https://example.invalid", model="d")
    a._closure_memory = a.memory          # 工具闭包实际使用的实例
    a.memory.close()
    a.memory = Memory(db_path=str(tmp_path / "memory.db"))
    a.memory.save_memory(
        key="career_goal",
        value=TEST_VALUE,
        category="goal",
        reason="RAG 8.21 专项测试虚构语料", user_id="test-user"
    )
    return a


def install_state_capture(monkeypatch):
    captured = []

    class SpyState(agent_module.AgentState):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            captured.append(self)

    monkeypatch.setattr(agent_module, "AgentState", SpyState)
    return captured


def instrument_memory(agent):
    """细粒度 spy：记录查询方法及其关键字参数（含 min_score），
    并记录真实返回值（find 的命中 ID / search 的返回 ID 列表）。

    同时包装 agent.memory 与工具闭包持有的实例（不同对象、同一临时库），
    使主检索与工具检索的查询都能被观测。全部为"调用真实方法后记录
    再返回"，不改变行为、顺序、返回值与异常传播。
    """
    calls = {"find": [], "search": [], "by_cat": 0}

    def wrap(memory_obj):
        real_find = memory_obj.find_memory
        real_search = memory_obj.search_long_term_memory
        real_by_cat = memory_obj.get_memories_by_category

        def spy_find(key, *a, **k):
            result = real_find(key, *a, **k)
            calls["find"].append(
                {"key": key,
                 "returned_id": result.get("id") if result else None}
            )
            return result

        def spy_search(*a, **k):
            result = real_search(*a, **k)
            calls["search"].append(
                {"kwargs": k, "returned": len(result),
                 "ids": [m["id"] for m in result]}
            )
            return result

        def spy_by_cat(*a, **k):
            calls["by_cat"] += 1
            return real_by_cat(*a, **k)

        memory_obj.find_memory = spy_find
        memory_obj.search_long_term_memory = spy_search
        memory_obj.get_memories_by_category = spy_by_cat

    wrap(agent.memory)
    closure_memory = getattr(agent, "_closure_memory", None)
    if closure_memory is not None and closure_memory is not agent.memory:
        wrap(closure_memory)
    return calls


def run_with_fake(agent, q):
    agent.client = p7.FakeClient(p7.RUN_SCRIPT)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = agent.run(q, user_id="test-user")
    return result, agent.client.calls


# ======================================================================
# 测试 1：正常回忆 → Planner 与 Reasoning 上下文中均含测试值
# ======================================================================

def test_reasoning_context_receives_memory_value_on_recall(
        tmp_path, monkeypatch):
    q = "我上次说的“职业目标”是什么？"
    agent = unique_build_agent(tmp_path, monkeypatch)
    mem_calls = instrument_memory(agent)
    reg_spies = p7.attach_spies(agent)[1]     # 仅需工具 spy（记忆 spy 已自建）

    # 主检索直接验证返回值（检索层）
    candidates = agent.retrieve_long_term_memories(q)
    assert any(TEST_VALUE in m["value"] for m in candidates)

    result, calls = run_with_fake(agent, q)

    # 数据库层：主检索确实执行（find/search/by_cat 任一 > 0）
    n_query = (
        len(mem_calls["find"])
        + len(mem_calls["search"])
        + mem_calls["by_cat"]
    )
    assert n_query > 0

    planners = [m for m in calls if is_planner_request(m)]

    # Planner 请求的长期记忆段包含正确键值
    assert len(planners) >= 1
    assert any(MEMORY_MARKER in m[0]["content"] for m in planners)

    # 步骤式与收尾式 Reasoning 分别断言（按计划状态特征分类，非数组索引）
    step_reqs = [m for m in calls if classify_reasoning(m) == "step"]
    final_reqs = [m for m in calls if classify_reasoning(m) == "final"]
    assert len(step_reqs) >= 1, "未观测到步骤式 Reasoning 请求（执行路径未覆盖）"
    assert len(final_reqs) >= 1, "未观测到收尾式 Reasoning 请求（执行路径未覆盖）"

    for m in step_reqs + final_reqs:
        content = m[0]["content"]
        # 区域定位断言：值必须位于"用户长期记忆"段内
        assert in_long_term_section(content), (
            "测试值不在长期记忆上下文区域（任意位置匹配不算通过）"
        )


# ======================================================================
# 测试 2：虚构角色 → 零查询、零工具、历史无泄漏、Reasoning 干净
# ======================================================================

def test_reasoning_context_clean_on_fiction_refusal(tmp_path, monkeypatch):
    q = "我在角色扮演，现在我是医生，我的职业目标是什么？"
    agent = unique_build_agent(tmp_path, monkeypatch)
    captured = install_state_capture(monkeypatch)
    mem_calls = instrument_memory(agent)
    calls_spy, reg_spy = p7.attach_spies(agent)

    result, calls = run_with_fake(agent, q)

    # 数据库层：零查询
    assert mem_calls["find"] == []
    assert mem_calls["search"] == []
    assert mem_calls["by_cat"] == 0

    # 工具层：未执行真实检索（registry 未被调用）
    assert reg_spy["calls"] == []

    # 执行历史：无测试值、无记忆字段
    assert len(captured) == 1
    dumped = json.dumps(captured[0].execution_history, ensure_ascii=False)
    assert TEST_VALUE not in dumped
    assert "memories" not in dumped

    # 上下文层：Planner 不含值；全部 Reasoning 请求不含值
    planners = [m for m in calls if is_planner_request(m)]
    reasonings = [m for m in calls if is_reasoning_request(m)]
    assert len(planners) >= 1
    assert all(TEST_VALUE not in m[0]["content"] for m in planners)
    assert len(reasonings) >= 1, (
        "未产生可观测的 Reasoning 请求（执行路径未覆盖，"
        "不能据此宣称 Reasoning 隔离通过）"
    )
    assert all(TEST_VALUE not in m[0]["content"] for m in reasonings)


# ======================================================================
# 测试 3：主检索 ≤3 / 工具检索 ≤5 / 工具不调 find 兜底 / min_score 差异记录
# ======================================================================

def test_tool_vs_main_result_limit_isolation(tmp_path, monkeypatch):
    """RAG 8.21-6 修正：真实触发两侧上限（此前 category 错配导致假阳性）。

    - 记忆类别改为与"专业"意图规则一致的 "profile"（规则 categories）；
    - 先插 key="major"（最早 id，供 find 命中且不与 search 前 3 重叠），
      再插 6 条同类记忆 → 主候选 7 条（>3）、工具候选 7 条（>5）；
    - 主：内部 search(limit=3) 返回 3；find 1 + search 3 = 4 个候选
      → 最终 [:3] 截断为 3（去重后不重叠）；
    - 工具：内部 search(limit=5) 返回 5（候选 7>5，截断生效）；
    - min_score：主显式 2；工具未传、使用默认 2——实际阈值相同。
    """
    import inspect

    agent = unique_build_agent(tmp_path, monkeypatch)
    agent.memory.save_memory(
        key="major",
        value="计算机专业（主检索 find 命中项）",
        category="profile",
        reason="RAG 8.21 主路径 find 命中项（最早 id）", user_id="test-user"
    )
    for i in range(6):
        agent.memory.save_memory(
            key=f"major_{i}",
            value=f"计算机专业方向{i}记录",
            category="profile",
            reason="RAG 8.21 数量限制测试语料", user_id="test-user"
        )

    mem_calls = instrument_memory(agent)
    tool_calls = []
    real_exec = agent.tool_registry.execute

    def spy_exec(name, arguments):
        tool_calls.append((name, arguments))
        return real_exec(name, arguments)

    agent.tool_registry.execute = spy_exec

    main_q = "我的专业是什么来着？"

    # 容量验证（证明"确有 >上限 的候选可通过实际评分条件"）
    main_capacity = agent.memory.search_long_term_memory(
        main_q, limit=100, category="profile", min_score=2, user_id="test-user"
    )
    tool_capacity = agent.memory.search_long_term_memory("专业", limit=100, user_id="test-user")

    # 真实两条路径（模型隔离）
    main_results = agent.retrieve_long_term_memories(main_q)
    find_after_main = len(mem_calls["find"])
    tool_result = agent.tool_registry.execute("search_memory", {"query": "专业"})
    tool_memories = tool_result["result"]["memories"]

    # 内部搜索记录：主容量 / 工具容量 / 主检索 / 工具（共 4 次）
    assert len(mem_calls["search"]) == 4
    c_main_cap, c_tool_cap, c_main, c_tool = mem_calls["search"]

    # 主检索上限证据：候选 7 > 3 → 内部 limit=3 返回 3 → 最终 3
    assert len(main_capacity) == 7 and len(main_capacity) > 3
    assert c_main_cap["returned"] == 7
    assert c_main["kwargs"].get("limit") == 3
    assert c_main["returned"] == 3
    assert len(main_results) == 3

    # 去重与截断：find 命中 1（最早 id）+ search 3（最新 id）应互不重叠
    # → 4 个候选 → 最终 [:3] 截断为 3；不能把两路数量简单相加。
    assert find_after_main == 1
    assert len({m["id"] for m in main_results}) == 3

    # 显式验证不重叠（真实 ID 比对；不依赖排序逻辑、注释或数量推断）
    find_hit_id = mem_calls["find"][0]["returned_id"]
    assert find_hit_id is not None
    assert find_hit_id not in set(c_main["ids"]), (
        f"find 命中 ID={find_hit_id} 出现在主内部 search 返回 ID "
        f"{c_main['ids']} 中（重叠，截断证据不成立）"
    )

    # 工具上限证据：候选 7 > 5 → 内部 limit=5 返回 5（截断生效）
    assert len(tool_capacity) == 7 and len(tool_capacity) > 5
    assert c_tool_cap["returned"] == 7
    assert c_tool["kwargs"].get("limit") == 5
    assert c_tool["returned"] == 5
    assert len(tool_memories) == 5
    assert tool_calls == [("search_memory", {"query": "专业"})]

    # 工具路径不调用 find_memory 兜底（主路径专有）
    assert len(mem_calls["find"]) == find_after_main == 1

    # min_score：主显式 2；工具未传参 → 使用默认值 2——实际阈值相同，
    # 不得把"未传参"当作"无门槛"。
    assert c_main["kwargs"].get("min_score") == 2
    assert "min_score" not in c_tool["kwargs"]
    default = inspect.signature(
        Memory.search_long_term_memory
    ).parameters["min_score"].default
    assert default == 2

    print(
        f"\n[821 记录] 主：候选 {len(main_capacity)}、内部 "
        f"{c_main['returned']}、最终 {len(main_results)}；"
        f"工具：候选 {len(tool_capacity)}、内部 {c_tool['returned']}、"
        f"最终 {len(tool_memories)}；min_score 主=2（显式）/ "
        f"工具默认={default}"
    )
