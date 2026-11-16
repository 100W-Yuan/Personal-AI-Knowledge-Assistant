# -*- coding: utf-8 -*-
"""管理员模式订单查询工具（query_orders）隔离测试。

覆盖 R3-AGENT-ORDER-02B 要求的 15 项用例，并额外验证「数据库异常与空结果分离」。

数据隔离（硬要求）：
1. 业务数据库是绝对路径（business/db.py 用 __file__ 推导），
   仅 monkeypatch.chdir() 无效，必须替换 DATA_DIR / DATABASE_PATH。
2. Agent 的记忆数据库是相对路径（memory.py 默认 "memory.db"），
   构造 Agent 前 chdir 到 tmp_path，确保 memory.db 落在临时目录。
3. 模块级 OpenAI 被替换为拒绝网络的替身，任何模型调用立即失败。
4. 不读取 .env、不使用真实 API Key、不连接正式业务数据库。
"""

import json
import sqlite3
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from business import db as business_db                # noqa: E402
from business import queries as business_queries      # noqa: E402
from business.validators import ORDER_STATUSES        # noqa: E402
from tools import (                                   # noqa: E402
    ADMIN_ORDER_MAX_PAGE_SIZE,
    ADMIN_ORDER_SCOPE,
    QUERY_ORDERS_FAILURE_PREFIX,
    build_query_orders_tool_definition,
    make_query_orders_tool,
)
from tool_registry import ToolRegistry                # noqa: E402

# 虚构订单 / 客户编号；base_url 使用 RFC 2606 保留域名，永不解析
DUMMY = dict(
    api_key="dummy-key",
    base_url="https://example.invalid",
    model="dummy-model",
)

CUSTOMER_A = "CUST-TEST-A"
CUSTOMER_B = "CUST-TEST-B"

DEFAULT_FIELDS = {"order_id", "status", "created_at", "amount"}


class _BlockedOpenAI:
    """拒绝网络的模型客户端替身。

    构造可以成功（Agent.__init__ 需要），但一旦访问 .chat
    立即抛出 AssertionError，使任何真实调用变成测试失败。
    """

    def __init__(self, *args, **kwargs):
        self.init_args = args
        self.init_kwargs = kwargs

    @property
    def chat(self):
        raise AssertionError("测试期间禁止调用真实模型 API")


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """把业务数据库锁定在 tmp_path 内，绝不触达正式 data/business.db。"""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    db_path = data_dir / "business_order_tool_test.db"

    monkeypatch.setattr(business_db, "DATA_DIR", data_dir)
    monkeypatch.setattr(business_db, "DATABASE_PATH", db_path)
    business_db.init_database()

    assert business_db.DATABASE_PATH.is_relative_to(tmp_path)
    assert db_path.exists()

    return db_path


@pytest.fixture
def tool():
    """工具原始函数（未经 Tool Registry 包装）。"""
    return make_query_orders_tool()


@pytest.fixture
def registry():
    """按 Agent 注册时的同一份定义构建注册表。"""
    reg = ToolRegistry()
    reg.register(
        "query_orders",
        make_query_orders_tool(),
        build_query_orders_tool_definition(),
    )
    return reg


@pytest.fixture
def agent_env(tmp_path, monkeypatch, temp_db):
    """构造真实 Agent（记忆库隔离到 tmp_path，模型客户端被阻断）。"""
    import agent as agent_module

    monkeypatch.setattr(agent_module, "OpenAI", _BlockedOpenAI)
    monkeypatch.chdir(tmp_path)

    agent_obj = agent_module.Agent(**DUMMY)

    # 记忆库必须落在临时目录，而不是项目根
    assert (tmp_path / "memory.db").exists()

    return agent_obj


def insert_order(
    db_path,
    order_id,
    customer_id=CUSTOMER_A,
    status="已支付",
    created_at="2026-11-01T10:00:00",
    amount=100.0,
    paid_at=None,
):
    """直接写入临时业务库；字段与 orders 表结构一致。"""
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            "INSERT INTO orders (order_id, customer_id, status, created_at,"
            " amount, paid_at) VALUES (?, ?, ?, ?, ?, ?)",
            (order_id, customer_id, status, created_at, amount, paid_at),
        )
        connection.commit()
    finally:
        connection.close()


# ========================================================
# 1. 最新订单排序正确
# ========================================================

def test_01_latest_order_first(temp_db, tool):
    insert_order(temp_db, "ORDER-001", created_at="2026-11-01T10:00:00")
    insert_order(temp_db, "ORDER-002", created_at="2026-11-02T10:00:00")
    insert_order(temp_db, "ORDER-003", created_at="2026-11-03T10:00:00")

    result = tool(page_size=3)

    assert result["scope"] == ADMIN_ORDER_SCOPE
    assert result["count"] == 3
    assert result["total"] == 3
    assert result["items"][0]["order_id"] == "ORDER-003"
    assert [item["order_id"] for item in result["items"]] == [
        "ORDER-003",
        "ORDER-002",
        "ORDER-001",
    ]


# ========================================================
# 2. 相同创建时间的稳定排序（created_at DESC, order_id ASC）
# ========================================================

def test_02_stable_order_on_same_created_at(temp_db, tool):
    same_time = "2026-11-02T10:00:00"
    insert_order(temp_db, "ORDER-002", created_at=same_time)
    insert_order(temp_db, "ORDER-001", created_at=same_time)

    result = tool()

    assert [item["order_id"] for item in result["items"]] == [
        "ORDER-001",
        "ORDER-002",
    ]


# ========================================================
# 3. 指定客户编号筛选（筛选生效，但默认不回显客户编号）
# ========================================================

def test_03_filter_by_customer_id(temp_db, tool):
    insert_order(temp_db, "ORDER-001", customer_id=CUSTOMER_A)
    insert_order(temp_db, "ORDER-002", customer_id=CUSTOMER_A)
    insert_order(temp_db, "ORDER-003", customer_id=CUSTOMER_B)

    result = tool(customer_id=CUSTOMER_A)

    assert result["count"] == 2
    assert result["total"] == 2
    assert {item["order_id"] for item in result["items"]} == {
        "ORDER-001",
        "ORDER-002",
    }
    # 按客户筛选 != 可以显示客户编号
    for item in result["items"]:
        assert "customer_id" not in item
    assert result["filters"]["customer_id"] == CUSTOMER_A


# ========================================================
# 4. 中文订单状态筛选
# ========================================================

def test_04_filter_by_chinese_status(temp_db, tool):
    insert_order(temp_db, "ORDER-001", status="已支付")
    insert_order(temp_db, "ORDER-002", status="已支付")
    insert_order(temp_db, "ORDER-003", status="已完成")

    result = tool(status="已支付")

    assert result["count"] == 2
    assert {item["order_id"] for item in result["items"]} == {
        "ORDER-001",
        "ORDER-002",
    }
    for item in result["items"]:
        assert item["status"] == "已支付"


# ========================================================
# 5. 空数据库正常返回空列表（不是异常）
# ========================================================

def test_05_empty_database_returns_empty_list(temp_db, tool):
    result = tool()

    assert result["count"] == 0
    assert result["total"] == 0
    assert result["items"] == []
    assert result["has_more"] is False


# ========================================================
# 6. 不合法状态被拒绝
# ========================================================

def test_06_invalid_status_rejected(temp_db, tool, registry):
    insert_order(temp_db, "ORDER-001")

    # 工具原始函数：抛 ValueError，消息带统一前缀
    with pytest.raises(ValueError) as excinfo:
        tool(status="PAID")
    assert str(excinfo.value).startswith(QUERY_ORDERS_FAILURE_PREFIX)
    assert "PAID" in str(excinfo.value)

    # 经 Registry：包装为 success=False，不向外抛异常
    outcome = registry.execute("query_orders", {"status": "PAID"})
    assert outcome["success"] is False
    assert outcome["error"].startswith(QUERY_ORDERS_FAILURE_PREFIX)

    # 合法状态仍然可用
    assert tool(status="已支付")["count"] == 1
    assert ORDER_STATUSES  # 枚举确实来自 business.validators


# ========================================================
# 7. 不存在的订单详情：found=false，不是数据库异常
# ========================================================

def test_07_missing_order_detail_found_false(temp_db, tool, registry):
    result = tool(order_id="ORDER-NOT-EXIST")

    assert result["found"] is False
    assert result["count"] == 0
    assert result["items"] == []
    assert result["scope"] == ADMIN_ORDER_SCOPE

    outcome = registry.execute(
        "query_orders", {"order_id": "ORDER-NOT-EXIST"}
    )
    assert outcome["success"] is True
    assert outcome["result"]["found"] is False


# ========================================================
# 8. 默认只返回四个字段
# ========================================================

def test_08_default_fields_only(temp_db, tool):
    insert_order(temp_db, "ORDER-001", paid_at="2026-11-01T12:00:00")

    item = tool()["items"][0]

    assert set(item.keys()) == DEFAULT_FIELDS
    assert "customer_id" not in item
    assert "paid_at" not in item


# ========================================================
# 9. include_customer_id=true 才返回客户编号
# ========================================================

def test_09_include_customer_id_opt_in(temp_db, tool):
    insert_order(temp_db, "ORDER-001", customer_id=CUSTOMER_A)

    without = tool()["items"][0]
    with_flag = tool(include_customer_id=True)["items"][0]

    assert "customer_id" not in without
    assert with_flag["customer_id"] == CUSTOMER_A
    assert set(with_flag.keys()) == DEFAULT_FIELDS | {"customer_id"}


# ========================================================
# 10. include_paid_at=true 才返回付款时间
# ========================================================

def test_10_include_paid_at_opt_in(temp_db, tool):
    insert_order(temp_db, "ORDER-001", paid_at="2026-11-01T12:00:00")

    without = tool()["items"][0]
    with_flag = tool(include_paid_at=True)["items"][0]

    assert "paid_at" not in without
    assert with_flag["paid_at"] == "2026-11-01T12:00:00"
    assert set(with_flag.keys()) == DEFAULT_FIELDS | {"paid_at"}


# ========================================================
# 11. 默认与最大分页均不超过 20
# ========================================================

def test_11_pagination_cap(temp_db, tool):
    for index in range(25):
        insert_order(
            temp_db,
            "ORDER-%03d" % index,
            created_at="2026-11-%02dT10:00:00" % (index % 28 + 1),
        )

    default_result = tool()
    assert default_result["page"] == 1
    assert default_result["page_size"] == ADMIN_ORDER_MAX_PAGE_SIZE
    assert default_result["count"] == ADMIN_ORDER_MAX_PAGE_SIZE
    assert default_result["total"] == 25
    assert default_result["has_more"] is True

    oversized = tool(page_size=100)
    assert oversized["page_size"] == ADMIN_ORDER_MAX_PAGE_SIZE
    assert oversized["count"] == ADMIN_ORDER_MAX_PAGE_SIZE

    second_page = tool(page=2, page_size=20)
    assert second_page["page"] == 2
    assert second_page["count"] == 5
    assert second_page["has_more"] is False


# ========================================================
# 12. page=0、page_size=0、bool 参数校验
# ========================================================

@pytest.mark.parametrize(
    "kwargs",
    [
        {"page": 0},
        {"page": -1},
        {"page_size": 0},
        {"page_size": -5},
        {"page": True},
        {"page_size": True},
    ],
)
def test_12_pagination_bounds_rejected(temp_db, tool, kwargs):
    with pytest.raises(ValueError) as excinfo:
        tool(**kwargs)
    assert str(excinfo.value).startswith(QUERY_ORDERS_FAILURE_PREFIX)


def test_12b_pagination_minimum_accepted(temp_db, tool):
    insert_order(temp_db, "ORDER-001")

    assert tool(page=1, page_size=1)["count"] == 1


# ========================================================
# 13. 未知参数被 Tool Registry 拒绝
# ========================================================

def test_13_unknown_argument_rejected(registry):
    validation = registry.validate_arguments(
        "query_orders", {"sql": "DROP TABLE orders"}
    )
    assert validation["valid"] is False
    assert validation["error"] == "未知参数：sql"

    outcome = registry.execute("query_orders", {"sql": "DROP TABLE orders"})
    assert outcome["success"] is False
    assert "未知参数" in outcome["error"]


# ========================================================
# 14. Agent 正确注册 query_orders（不重复）
# ========================================================

def test_14_agent_registers_query_orders(agent_env):
    registry_obj = agent_env.tool_registry

    assert "query_orders" in registry_obj.tools

    # 注册一次且仅一次
    names = [
        tool_item["function"]["name"]
        for tool_item in registry_obj.get_openai_tools()
    ]
    assert names.count("query_orders") == 1

    # 未注入 knowledge_base 时：4 个既有工具 + query_orders
    assert len(registry_obj.tools) == 5

    definition = registry_obj.tools["query_orders"]["definition"]
    assert definition["name"] == "query_orders"
    assert set(definition["parameters"]["properties"]) == {
        "order_id",
        "customer_id",
        "status",
        "page",
        "page_size",
        "include_customer_id",
        "include_paid_at",
    }
    assert definition["parameters"]["required"] == []


# ========================================================
# 15. Tool Registry 正确分发，真实结果进入执行历史
# ========================================================

def test_15_dispatch_records_execution_history(agent_env, temp_db):
    from task import AgentState

    insert_order(temp_db, "ORDER-001", created_at="2026-11-01T10:00:00")
    insert_order(temp_db, "ORDER-002", created_at="2026-11-02T10:00:00")

    state = AgentState("最近的订单")

    outcome = agent_env.execute_tool(
        state, "query_orders", {"page_size": 2}
    )

    # Agent.execute_tool 直接返回 Registry 的包装结构
    assert outcome["success"] is True
    assert outcome["result"]["scope"] == ADMIN_ORDER_SCOPE
    assert outcome["result"]["count"] == 2
    assert outcome["result"]["items"][0]["order_id"] == "ORDER-002"

    # execution_history 记录的是同一份包装结构
    assert len(state.execution_history) == 1
    entry = state.execution_history[0]
    assert entry["type"] == "tool"
    assert entry["tool_name"] == "query_orders"
    assert entry["arguments"] == {"page_size": 2}
    assert entry["result"]["success"] is True
    assert entry["result"]["result"]["count"] == 2

    # reasoning 会把执行历史整体序列化进 prompt，必须可序列化
    json.dumps(state.execution_history, ensure_ascii=False)


# ========================================================
# 16. 数据库异常与空结果分离，且不泄露内部细节
# ========================================================

def test_16_database_error_is_safe_and_distinct(
    temp_db, registry, monkeypatch
):
    secret_detail = "unable to open database file: C:/secret/path/business.db"

    def boom(*args, **kwargs):
        raise sqlite3.OperationalError(secret_detail)

    monkeypatch.setattr(business_queries, "list_orders", boom)

    outcome = registry.execute("query_orders", {"page_size": 1})

    assert outcome["success"] is False
    assert outcome["error"] == (
        QUERY_ORDERS_FAILURE_PREFIX + "暂时无法读取订单数据，请稍后重试。"
    )
    assert "secret" not in outcome["error"]
    assert "path" not in outcome["error"]
    assert "Traceback" not in outcome["error"]


# ========================================================
# 17-20. 真实 Agent Loop 集成测试（Mock OpenAI）
#
# 只替换模块级 OpenAI 客户端；以下全部使用真实实现：
#   Agent.run / create_plan / execute_tool / ToolRegistry.execute / query_orders
# 工具与数据库均为真实：隔离 SQLite + 真实只读查询函数。
# 模型响应由脚本队列驱动，脚本耗尽立即 AssertionError，杜绝意外外呼。
# ========================================================

ORDER_LOG_ID = "TEST-ORDER-LOG-001"
CUSTOMER_SECRET = "TEST-CUSTOMER-SECRET-001"
ORDER_AMOUNT = 987654.32
ORDER_CREATED_AT = "2026-11-05T08:00:00"
PAID_AT = "2026-11-05T09:30:00"

ORDER_TOOL_PROPERTIES = (
    "order_id",
    "customer_id",
    "status",
    "page",
    "page_size",
    "include_customer_id",
    "include_paid_at",
)


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _ScriptedCompletions:
    """按脚本顺序返回响应，并记录每次请求。"""

    def __init__(self, script, calls):
        self._script = list(script)
        self._calls = calls

    def create(self, model=None, messages=None, **kwargs):
        self._calls.append(
            {"model": model, "messages": messages, "kwargs": kwargs}
        )
        if not self._script:
            raise AssertionError("Mock 模型收到脚本之外的额外调用")
        return _FakeResponse(self._script.pop(0))


class _ScriptedChat:
    def __init__(self, script, calls):
        self.completions = _ScriptedCompletions(script, calls)


def _install_scripted_openai(monkeypatch, script):
    """把模块级 OpenAI 换成脚本化替身，返回调用记录列表。"""
    import agent as agent_module

    calls = []

    class _ScriptedOpenAI:
        def __init__(self, *args, **kwargs):
            self.init_kwargs = kwargs
            self.chat = _ScriptedChat(script, calls)

    monkeypatch.setattr(agent_module, "OpenAI", _ScriptedOpenAI)
    return calls


def _call_text(call):
    """拼接一次请求里的文本消息，用于断言模型实际收到的上下文。"""
    parts = []
    for message in call.get("messages") or []:
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
    return "\n".join(parts)


def _agent_loop_script(tool_arguments, final_answer):
    """驱动一次完整 Agent Loop 的响应队列。

    顺序：记忆提取 → Planner → 工具参数生成 → reasoning 步骤 → 最终 reasoning
    """
    return [
        json.dumps({"should_remember": False, "memories": []}),
        json.dumps(
            {
                "plan": [
                    {
                        "description": "查询最近订单",
                        "type": "tool",
                        "tool_name": "query_orders",
                    },
                    {"description": "整理回答", "type": "reasoning"},
                ]
            }
        ),
        json.dumps(tool_arguments),
        "已取得订单数据，准备整理回答。",
        final_answer,
    ]


def _run_agent_loop(monkeypatch, tmp_path, script, user_message):
    """在隔离环境中执行完整 Agent.run()，返回 (agent, calls, result)。"""
    calls = _install_scripted_openai(monkeypatch, script)

    import agent as agent_module

    monkeypatch.chdir(tmp_path)

    agent_obj = agent_module.Agent(**DUMMY)

    # 记忆库必须落在临时目录；业务库必须指向 tmp_path
    assert (tmp_path / "memory.db").exists()
    assert business_db.DATABASE_PATH.is_relative_to(tmp_path)

    result = agent_obj.run(user_message, user_id="test-user")

    return agent_obj, calls, result


# ========================================================
# 17. 完整 Agent Loop：工具可见 + 计划 + 真实执行 + 回答
# ========================================================

def test_17_agent_loop_executes_query_orders(tmp_path, monkeypatch, temp_db):
    insert_order(
        temp_db,
        ORDER_LOG_ID,
        customer_id=CUSTOMER_SECRET,
        status="已支付",
        created_at=ORDER_CREATED_AT,
        amount=ORDER_AMOUNT,
        paid_at=PAID_AT,
    )

    final_answer = (
        f"按管理员全局范围查询到 1 条订单：{ORDER_LOG_ID}，"
        f"状态已支付，创建时间 {ORDER_CREATED_AT}，金额 {ORDER_AMOUNT}。"
    )
    script = _agent_loop_script({"page_size": 5}, final_answer)

    agent_obj, calls, result = _run_agent_loop(
        monkeypatch, tmp_path, script, "查一下最近的订单"
    )

    assert result["success"] is True
    assert result["answer"] == final_answer

    # 模型调用次数：记忆提取 / Planner / 参数生成 / reasoning 步骤 / 最终 reasoning
    assert len(calls) == 5

    texts = [_call_text(call) for call in calls]

    # A. 工具可见：Planner 的 system prompt 内嵌工具定义
    planner_text = texts[1]
    assert "query_orders" in planner_text
    for name in ORDER_TOOL_PROPERTIES:
        assert name in planner_text

    # 未知参数仍无法通过注册表校验
    assert agent_obj.tool_registry.validate_arguments(
        "query_orders", {"not_allowed": 1}
    )["valid"] is False

    # B. 工具真实执行：结果进入 reasoning 上下文（execution_history 序列化）
    reasoning_text = texts[-1]
    assert '"tool_name": "query_orders"' in reasoning_text
    assert '"success": true' in reasoning_text
    assert ORDER_LOG_ID in reasoning_text
    assert '"count": 1' in reasoning_text

    # C. 回答中的字段全部来自隔离数据库
    assert ORDER_LOG_ID in result["answer"]
    assert "已支付" in result["answer"]
    assert ORDER_CREATED_AT in result["answer"]
    assert str(ORDER_AMOUNT) in result["answer"]

    # 持久化：最终回答按原有机制写入临时 memory.db（正式库不受影响）
    connection = sqlite3.connect(tmp_path / "memory.db")
    try:
        rows = connection.execute(
            "SELECT role, content FROM messages ORDER BY id DESC LIMIT 1"
        ).fetchall()
    finally:
        connection.close()

    assert rows
    assert rows[0][0] == "assistant"
    assert rows[0][1] == final_answer


# ========================================================
# 18. 控制台日志脱敏（capsys）
# ========================================================

def test_18_console_log_redacted(
    tmp_path, monkeypatch, temp_db, capsys
):
    insert_order(
        temp_db,
        ORDER_LOG_ID,
        customer_id=CUSTOMER_SECRET,
        status="已支付",
        created_at=ORDER_CREATED_AT,
        amount=ORDER_AMOUNT,
        paid_at=PAID_AT,
    )

    # 回答本身保持通用措辞，使 stdout 中不应出现任何订单明细
    script = _agent_loop_script(
        {"page_size": 1}, "已按管理员全局范围完成查询，共 1 条订单。"
    )

    _run_agent_loop(monkeypatch, tmp_path, script, "查一下最近的订单")

    out = capsys.readouterr().out

    # 脱敏后的诊断信息存在
    assert "📤 工具结果：query_orders" in out
    assert "状态：成功" in out
    assert "返回数量：1" in out
    assert "总记录数：1" in out
    assert "当前页：1" in out
    assert "每页数量：1" in out

    # 订单明细与内部实现细节均不得出现
    assert ORDER_LOG_ID not in out
    assert CUSTOMER_SECRET not in out
    assert str(ORDER_AMOUNT) not in out
    assert PAID_AT not in out
    assert '"items"' not in out
    assert "business_order_tool_test.db" not in out


# ========================================================
# 19. 默认最小字段：customer_id / paid_at 不进入模型上下文
# ========================================================

def test_19_default_fields_not_sent_to_model(
    tmp_path, monkeypatch, temp_db
):
    insert_order(
        temp_db,
        ORDER_LOG_ID,
        customer_id=CUSTOMER_SECRET,
        status="已支付",
        created_at=ORDER_CREATED_AT,
        amount=ORDER_AMOUNT,
        paid_at=PAID_AT,
    )

    script = _agent_loop_script({"page_size": 5}, "已查询。")

    _, calls, _ = _run_agent_loop(
        monkeypatch, tmp_path, script, "查一下最近的订单"
    )

    reasoning_text = _call_text(calls[-1])

    # 默认字段确实进入了上下文
    assert ORDER_LOG_ID in reasoning_text
    assert ORDER_CREATED_AT in reasoning_text
    assert str(ORDER_AMOUNT) in reasoning_text

    # 可选字段值不得出现
    assert CUSTOMER_SECRET not in reasoning_text
    assert PAID_AT not in reasoning_text

    # filters 会回显筛选状态（未指定时为 customer_id: null），
    # 这里约束的是「不得出现任何客户编号值」，而不是禁止 null 占位。
    assert '"customer_id": "' not in reasoning_text
    assert '"paid_at"' not in reasoning_text


# ========================================================
# 20. 显式开启后，可选字段才进入模型上下文
# ========================================================

def test_20_optional_fields_opt_in_reach_model(
    tmp_path, monkeypatch, temp_db
):
    insert_order(
        temp_db,
        ORDER_LOG_ID,
        customer_id=CUSTOMER_SECRET,
        status="已支付",
        created_at=ORDER_CREATED_AT,
        amount=ORDER_AMOUNT,
        paid_at=PAID_AT,
    )

    final_answer = (
        f"客户 {CUSTOMER_SECRET} 的订单 {ORDER_LOG_ID}，"
        f"付款时间 {PAID_AT}。"
    )
    script = _agent_loop_script(
        {
            "page_size": 5,
            "include_customer_id": True,
            "include_paid_at": True,
        },
        final_answer,
    )

    _, calls, result = _run_agent_loop(
        monkeypatch, tmp_path, script, "查订单，并显示客户编号和付款时间"
    )

    reasoning_text = _call_text(calls[-1])

    assert '"customer_id": "TEST-CUSTOMER-SECRET-001"' in reasoning_text
    assert '"paid_at": "2026-11-05T09:30:00"' in reasoning_text
    assert CUSTOMER_SECRET in result["answer"]
    assert PAID_AT in result["answer"]


# ========================================================
# R3-AGENT-ORDER-02C1：参数日志与最终回答日志脱敏
# ========================================================

C1_ORDER_ID = "TEST-C1-ORDER-001"
C1_CUSTOMER_ID = "TEST-C1-CUSTOMER-001"
C1_AMOUNT = 987654.32


# ========================================================
# 测试 A：订单参数日志脱敏
# ========================================================

def test_A_order_arguments_log_redacted(
    tmp_path, monkeypatch, temp_db, capsys
):
    insert_order(
        temp_db,
        C1_ORDER_ID,
        customer_id=C1_CUSTOMER_ID,
        status="已支付",
        created_at="2026-11-05T10:00:00",
        amount=C1_AMOUNT,
        paid_at="2026-11-05T11:00:00",
    )

    # 故意把订单编号与客户编号都放进参数，验证日志不泄露实际值
    script = _agent_loop_script(
        {
            "order_id": C1_ORDER_ID,
            "customer_id": C1_CUSTOMER_ID,
            "page_size": 3,
        },
        "已按管理员全局范围完成查询。",
    )

    _, calls, result = _run_agent_loop(
        monkeypatch, tmp_path, script, "查询指定订单"
    )

    out = capsys.readouterr().out

    assert result["success"] is True

    # 脱敏后的参数日志存在，且只暴露字段名
    assert "📦 工具参数：query_orders" in out
    assert "参数字段：order_id, customer_id, page_size" in out

    # 实际值一律不出现
    assert C1_ORDER_ID not in out
    assert C1_CUSTOMER_ID not in out
    assert str(C1_AMOUNT) not in out

    # 不出现原始参数结构（旧格式 dict dump / JSON 回退）
    assert "📦 参数：{" not in out
    assert '"order_id":' not in out

    # 传入注册表的参数未被脱敏逻辑修改
    reasoning_text = _call_text(calls[-1])
    assert '"arguments": {' in reasoning_text
    assert f'"customer_id": "{C1_CUSTOMER_ID}"' in reasoning_text
    assert '"page_size": 3' in reasoning_text


# ========================================================
# 测试 B：订单最终回答日志脱敏
# ========================================================

def test_B_order_final_answer_log_redacted(
    tmp_path, monkeypatch, temp_db, capsys
):
    insert_order(
        temp_db,
        C1_ORDER_ID,
        customer_id=C1_CUSTOMER_ID,
        status="已支付",
        created_at="2026-11-05T10:00:00",
        amount=C1_AMOUNT,
        paid_at="2026-11-05T11:00:00",
    )

    final_answer = (
        f"订单 {C1_ORDER_ID}（客户 {C1_CUSTOMER_ID}）"
        f"状态已支付，金额 {C1_AMOUNT}。"
    )
    script = _agent_loop_script(
        {
            "page_size": 5,
            "include_customer_id": True,
            "include_paid_at": True,
        },
        final_answer,
    )

    _, calls, result = _run_agent_loop(
        monkeypatch, tmp_path, script, "查询订单并显示客户编号"
    )

    out = capsys.readouterr().out

    # 本次确实执行过 query_orders
    assert any(
        '"tool_name": "query_orders"' in _call_text(call)
        for call in calls
    )

    # 返回值完整
    assert result["success"] is True
    assert result["answer"] == final_answer

    # 控制台只出现安全摘要
    assert "🤖 最终答案：已生成" in out
    assert "来源：query_orders" in out
    assert "输出长度：" in out
    assert "订单内容：已隐藏" in out

    # 订单明细与回答原文均不出现
    assert C1_ORDER_ID not in out
    assert C1_CUSTOMER_ID not in out
    assert str(C1_AMOUNT) not in out
    assert "状态已支付" not in out

    # 原有持久化机制仍写入临时 memory.db
    connection = sqlite3.connect(tmp_path / "memory.db")
    try:
        rows = connection.execute(
            "SELECT role, content FROM messages ORDER BY id DESC LIMIT 1"
        ).fetchall()
    finally:
        connection.close()

    assert rows
    assert rows[0][0] == "assistant"
    assert rows[0][1] == final_answer


# ========================================================
# 测试 C：非订单工具日志兼容
# ========================================================

def test_C_non_order_tool_log_unchanged(
    tmp_path, monkeypatch, temp_db, capsys
):
    script = _agent_loop_script(
        {"expression": "1+2"},
        "计算结果是 3。",
    )
    # 把计划里的工具换成 calculator，其余流程不变
    script[1] = json.dumps(
        {
            "plan": [
                {
                    "description": "计算表达式",
                    "type": "tool",
                    "tool_name": "calculator",
                },
                {"description": "整理回答", "type": "reasoning"},
            ]
        }
    )

    _, _, result = _run_agent_loop(
        monkeypatch, tmp_path, script, "帮我算 1+2"
    )

    out = capsys.readouterr().out

    assert result["success"] is True

    # 原有参数日志保留（完整 dict repr）
    assert "📦 参数：{'expression': '1+2'}" in out

    # 原有工具结果日志保留
    assert "📤 工具结果：{'success': True, 'result': '3'}" in out

    # 原有最终回答日志保留（逐字输出）
    assert "🤖 最终答案：计算结果是 3。" in out


# ========================================================
# 测试 D：订单查询失败日志
# ========================================================

def test_D_order_failure_log_safe(
    tmp_path, monkeypatch, temp_db, capsys
):
    import agent as agent_module

    _install_scripted_openai(monkeypatch, [])
    monkeypatch.chdir(tmp_path)

    agent_obj = agent_module.Agent(**DUMMY)
    state = agent_module.AgentState("查询订单")

    # 读取一次构造期的工具定义输出，后续断言只看失败段
    capsys.readouterr()

    # 场景 1：数据库读取异常
    def boom(*args, **kwargs):
        raise sqlite3.OperationalError(
            "unable to open database file: C:/secret/r3ao02c1.db"
        )

    monkeypatch.setattr(business_queries, "list_orders", boom)

    outcome = agent_obj.execute_tool(
        state, "query_orders", {"page_size": 1}
    )
    assert outcome["success"] is False

    out = capsys.readouterr().out

    assert "📤 工具结果：query_orders" in out
    assert "状态：失败" in out
    assert "错误类型：ValueError" in out
    assert "secret" not in out
    assert "r3ao02c1.db" not in out
    assert "SELECT" not in out
    assert "Traceback" not in out
    assert C1_ORDER_ID not in out

    # 场景 2：参数校验失败（状态值不得出现在控制台）
    capsys.readouterr()

    outcome2 = agent_obj.execute_tool(
        state, "query_orders", {"status": "PAID"}
    )
    assert outcome2["success"] is False

    out2 = capsys.readouterr().out

    assert "📦 工具参数：query_orders" in out2
    assert "参数字段：status" in out2
    assert "PAID" not in out2
    assert "订单查询失败" not in out2


# ========================================================
# R3-AGENT-ORDER-02C2：剩余三处日志出口脱敏
# ========================================================

PLAN_SECRET_ORDER = "TEST-C2-ORDER-PLAN-SECRET"
PLAN_SECRET_CUSTOMER = "TEST-C2-CUSTOMER-PLAN-SECRET"

REASON_SECRET_ORDER = "TEST-C2-REASON-ORDER-SECRET"
REASON_SECRET_CUSTOMER = "TEST-C2-REASON-CUSTOMER-SECRET"
REASON_AMOUNT_TEXT = "987654.32"

PRECHECK_KEY = "customer_phone_secret"
PRECHECK_VALUE = "TEST-C2-PRECHECK-VALUE"

FLOW_ORDER_ID = "TEST-C2-FLOW-ORDER-SECRET"
FLOW_CUSTOMER_ID = "TEST-C2-FLOW-CUSTOMER-SECRET"
FLOW_AMOUNT = 987654.32


# ========================================================
# 测试 E：Planner 执行计划日志脱敏
# ========================================================

def test_E_order_plan_log_redacted(tmp_path, monkeypatch, temp_db, capsys):
    """订单语境的执行计划只输出摘要，不输出步骤描述。

    直接调用 create_plan()，把 Planner 日志与 run() 循环里的
    「📝 描述」输出隔离开，确保本用例只度量 Planner 这一处出口。
    """
    import agent as agent_module

    _install_scripted_openai(
        monkeypatch,
        [
            json.dumps(
                {
                    "plan": [
                        {
                            "description": f"查询订单 {PLAN_SECRET_ORDER}",
                            "type": "tool",
                            "tool_name": "query_orders",
                        },
                        {
                            "description": (
                                f"向客户 {PLAN_SECRET_CUSTOMER} 说明结果"
                            ),
                            "type": "reasoning",
                        },
                    ]
                }
            ),
            json.dumps(
                {
                    "plan": [
                        {
                            "description": (
                                f"重试订单 {PLAN_SECRET_ORDER}"
                            ),
                            "type": "tool",
                            "tool_name": "query_orders",
                        },
                        {
                            "description": (
                                f"重试后向 {PLAN_SECRET_CUSTOMER} 说明"
                            ),
                            "type": "reasoning",
                        },
                    ]
                }
            ),
        ],
    )
    monkeypatch.chdir(tmp_path)

    agent_obj = agent_module.Agent(**DUMMY)
    state = agent_module.AgentState("查询订单")

    # 丢弃构造期「当前注册工具」输出
    capsys.readouterr()

    agent_obj.create_plan(state)

    out = capsys.readouterr().out

    assert len(state.plan) == 2

    # 摘要格式
    assert "📋 执行计划：已生成" in out
    assert "任务类型：订单查询" in out
    assert "步骤数量：2" in out
    assert "订单数据：已隐藏" in out

    # 步骤描述与实际值一律不出现
    assert PLAN_SECRET_ORDER not in out
    assert PLAN_SECRET_CUSTOMER not in out
    assert "  1. " not in out
    assert "[tool=query_orders]" not in out

    # 不出现原始计划 JSON / 步骤结构
    assert '"plan"' not in out
    assert '"tool_name": "query_orders"' not in out
    assert '"description"' not in out

    # 脱敏只作用于控制台：计划本体保持完整
    assert PLAN_SECRET_ORDER in state.plan[0]["description"]
    assert state.plan[0]["tool_name"] == "query_orders"

    # ----------------------------------------
    # Replanner 输出的是同类「执行计划日志」，
    # 按 4.1 / 4.2 同一规则处理，这里一并验证。
    # ----------------------------------------
    capsys.readouterr()

    state.execution_history.append(
        {
            "type": "tool",
            "tool_name": "query_orders",
            "result": {"success": False, "error": "模拟失败"},
        }
    )

    agent_obj.replan(state)

    replan_out = capsys.readouterr().out

    assert "🔄 Replan 后的新计划：已生成" in replan_out
    assert "任务类型：订单查询" in replan_out
    assert "步骤数量：2" in replan_out
    assert "订单数据：已隐藏" in replan_out

    assert PLAN_SECRET_ORDER not in replan_out
    assert PLAN_SECRET_CUSTOMER not in replan_out
    assert "  1. " not in replan_out
    assert '"tool_name": "query_orders"' not in replan_out

    # 新计划本体同样未被脱敏逻辑改写
    assert PLAN_SECRET_ORDER in state.plan[0]["description"]


# ========================================================
# 测试 F：Reasoning 模型输出日志脱敏
# ========================================================

def test_F_order_reasoning_log_redacted(
    tmp_path, monkeypatch, temp_db, capsys
):
    """订单语境的 Reasoning 正文不打印，只打印安全摘要。"""
    reasoning_text = (
        f"已取得订单 {REASON_SECRET_ORDER}，"
        f"客户 {REASON_SECRET_CUSTOMER}，"
        f"金额 {REASON_AMOUNT_TEXT}，准备整理回答。"
    )

    script = _agent_loop_script(
        {"page_size": 5},
        "已按管理员全局范围完成查询。",
    )
    script[3] = reasoning_text

    _, calls, result = _run_agent_loop(
        monkeypatch, tmp_path, script, "查询最近订单"
    )

    out = capsys.readouterr().out

    assert result["success"] is True

    # 最终回答保持完整（脱敏只作用于 stdout）
    assert result["answer"] == "已按管理员全局范围完成查询。"

    # 摘要格式
    assert "💭 Reasoning：已处理订单查询结果" in out
    assert "状态：完成" in out
    assert "输出长度：" in out

    # 正文与明细一律不出现
    assert REASON_SECRET_ORDER not in out
    assert REASON_SECRET_CUSTOMER not in out
    assert REASON_AMOUNT_TEXT not in out
    assert "💭 Reasoning：已取得" not in out

    # 推理正文未被修改：最后一次模型请求仍含完整文本
    assert REASON_SECRET_ORDER in _call_text(calls[-1])


# ========================================================
# 测试 G：query_orders 参数预检错误日志脱敏
# ========================================================

def test_G_order_precheck_log_redacted(
    tmp_path, monkeypatch, temp_db, capsys
):
    """参数预检失败的日志改为固定分类摘要，错误诊断本身不变。"""
    import agent as agent_module

    _install_scripted_openai(monkeypatch, [])
    monkeypatch.chdir(tmp_path)

    agent_obj = agent_module.Agent(**DUMMY)
    state = agent_module.AgentState("查询订单")

    capsys.readouterr()

    outcome = agent_obj.execute_tool(
        state,
        "query_orders",
        {PRECHECK_KEY: PRECHECK_VALUE},
    )

    out = capsys.readouterr().out

    assert outcome["success"] is False

    # 摘要格式
    assert "⛔ 参数预检未通过：query_orders" in out
    assert "错误类型：参数校验失败" in out
    assert "订单参数值：已隐藏" in out

    # 原始校验文本（含模型编造的键名）不再打印
    assert "未知参数" not in out
    assert "工具参数校验失败" not in out
    assert PRECHECK_KEY not in out
    assert PRECHECK_VALUE not in out

    # 只改日志：返回给循环的错误诊断保持完整
    assert "未知参数" in outcome["error"]
    assert PRECHECK_KEY in outcome["error"]
    assert state.last_tool_result is outcome
    assert (
        state.execution_history[-1]["result"]["error"]
        == outcome["error"]
    )

    # 场景 2：订单编号 / 客户编号以数值形式误传，触发类型校验失败
    capsys.readouterr()

    outcome2 = agent_obj.execute_tool(
        state,
        "query_orders",
        {"order_id": 12345, "customer_id": 67890},
    )

    out2 = capsys.readouterr().out

    assert outcome2["success"] is False
    assert "⛔ 参数预检未通过：query_orders" in out2
    assert "错误类型：参数校验失败" in out2
    assert "订单参数值：已隐藏" in out2

    # 非法值、订单编号、客户编号、完整错误描述均不出现
    assert "12345" not in out2
    assert "67890" not in out2
    assert "类型错误" not in out2
    assert "参数 order_id 类型错误" not in out2
    assert "参数 customer_id 类型错误" not in out2

    # 诊断信息仍完整保留在返回值与执行历史中
    assert "参数 order_id 类型错误" in outcome2["error"]
    assert (
        state.execution_history[-1]["result"]["error"]
        == outcome2["error"]
    )


# ========================================================
# 测试 H：普通任务日志兼容（不得被订单脱敏波及）
# ========================================================

def test_H_non_order_logs_unchanged(
    tmp_path, monkeypatch, temp_db, capsys
):
    """非订单任务的计划、推理、参数、结果、回答、预检全部保持原文。"""
    import agent as agent_module

    script = _agent_loop_script(
        {"expression": "1+2"},
        "计算结果是 3。",
    )
    script[1] = json.dumps(
        {
            "plan": [
                {
                    "description": "计算表达式",
                    "type": "tool",
                    "tool_name": "calculator",
                },
                {"description": "整理回答", "type": "reasoning"},
            ]
        }
    )
    script[3] = "计算完成，准备给出结果。"

    agent_obj, _, result = _run_agent_loop(
        monkeypatch, tmp_path, script, "帮我算 1+2"
    )

    out = capsys.readouterr().out

    assert result["success"] is True

    # Planner 保持旧格式（逐步骤 + 类型 + 工具名）
    assert "📋 执行计划：" in out
    assert "  1. 计算表达式 [tool] [tool=calculator]" in out
    assert "  2. 整理回答 [reasoning] [tool=None]" in out
    assert "📋 执行计划：已生成" not in out

    # Reasoning 保持原文
    assert "💭 Reasoning：计算完成，准备给出结果。" in out
    assert "💭 Reasoning：已处理订单查询结果" not in out

    # 参数、工具结果、最终回答保持 02C1 既有行为
    assert "📦 参数：{'expression': '1+2'}" in out
    assert "📤 工具结果：{'success': True, 'result': '3'}" in out
    assert "🤖 最终答案：计算结果是 3。" in out

    # 非订单工具的参数预检失败仍是原文
    state = agent_module.AgentState("帮我算 1+2")
    capsys.readouterr()

    failure = agent_obj.execute_tool(
        state,
        "calculator",
        {"expression": 123},
    )

    out2 = capsys.readouterr().out

    assert failure["success"] is False
    assert "⛔ 参数预检未通过：工具参数校验失败：" in out2
    assert "参数 expression 类型错误" in out2


# ========================================================
# 测试 I：完整流程 stdout 全量扫描
# ========================================================

def test_I_full_flow_stdout_scan(
    tmp_path, monkeypatch, temp_db, capsys
):
    """一次真实订单查询的 stdout 内不得出现任何订单明细。"""
    insert_order(
        temp_db,
        FLOW_ORDER_ID,
        customer_id=FLOW_CUSTOMER_ID,
        status="已支付",
        created_at="2026-11-05T12:00:00",
        amount=FLOW_AMOUNT,
        paid_at="2026-11-05T13:00:00",
    )

    reasoning_text = (
        f"已读取订单 {FLOW_ORDER_ID}，"
        f"客户 {FLOW_CUSTOMER_ID}，"
        f"金额 {FLOW_AMOUNT}。"
    )
    final_answer = (
        f"最近一笔订单是 {FLOW_ORDER_ID}，"
        f"客户 {FLOW_CUSTOMER_ID}，"
        f"金额 {FLOW_AMOUNT} 元，状态已支付。"
    )

    script = _agent_loop_script(
        {"page_size": 5, "include_customer_id": True},
        final_answer,
    )
    script[3] = reasoning_text

    _, calls, result = _run_agent_loop(
        monkeypatch, tmp_path, script, "查询最近订单"
    )

    out = capsys.readouterr().out

    assert result["success"] is True
    assert result["answer"] == final_answer

    # 五个控制台出口全部是脱敏格式
    assert "📋 执行计划：已生成" in out
    assert "📦 工具参数：query_orders" in out
    assert "📤 工具结果：query_orders" in out
    assert "💭 Reasoning：已处理订单查询结果" in out
    assert "🤖 最终答案：已生成" in out

    # 全量扫描：订单编号 / 客户编号 / 金额 / 状态明细均不出现
    for secret in (
        FLOW_ORDER_ID,
        FLOW_CUSTOMER_ID,
        str(FLOW_AMOUNT),
        REASON_SECRET_ORDER,
    ):
        assert secret not in out

    assert "状态已支付" not in out
    assert "SELECT" not in out
    assert "Traceback" not in out

    # ----------------------------------------
    # 模型侧仍拿到真实隔离订单结果
    # 最后一次请求内嵌 state.execution_history，
    # 因此这里同时证明「执行历史未被改动」。
    # ----------------------------------------
    reasoning_context = _call_text(calls[-1])

    assert FLOW_ORDER_ID in reasoning_context
    assert FLOW_CUSTOMER_ID in reasoning_context
    assert str(FLOW_AMOUNT) in reasoning_context
    assert '"tool_name": "query_orders"' in reasoning_context

    # ----------------------------------------
    # 对话历史仍按原有机制写入 tmp_path/memory.db
    # ----------------------------------------
    connection = sqlite3.connect(tmp_path / "memory.db")
    try:
        rows = connection.execute(
            "SELECT role, content FROM messages ORDER BY id DESC LIMIT 2"
        ).fetchall()
    finally:
        connection.close()

    assert len(rows) == 2
    assert rows[0][0] == "assistant"
    assert rows[0][1] == final_answer
    assert rows[1][0] == "user"
    assert rows[1][1] == "查询最近订单"


# ========================================================
# R3-AGENT-ORDER-02C3-REMEDIATION：四处日志出口整改验证
# 说明：既有 test_A ~ test_D 属 02C1 轮次，本轮用例统一加 C3 前缀。
# ========================================================

C3_MEMORY_SECRET = "TEST-C3-MEMORY-ORDER-SECRET"
C3_MEMORY_CUSTOMER = "TEST-C3-MEMORY-CUSTOMER-SECRET"
C3_MEMORY_AMOUNT = "987654.32"

C3_HTTP_SECRET = "TEST-C3-HTTP-ORDER-SECRET"
C3_HTTP_CUSTOMER = "TEST-C3-HTTP-CUSTOMER-SECRET"

C3_PLAN_SECRET = "TEST-C3-PLAN-ORDER-SECRET"

C3_STATUS_VALUE = "TEST-C3-ILLEGAL-STATUS"
C3_JSON_SECRET = "TEST-C3-RAW-JSON-SECRET"
C3_DB_SECRET = "TEST-C3-DB-PATH-SECRET"


def _c3_no_tool_script(step_text, final_answer):
    """无工具任务的 4 次模型调用：记忆提取 → 计划 → 步骤推理 → 最终推理。"""
    return [
        json.dumps({"should_remember": False, "memories": []}),
        json.dumps(
            {
                "plan": [
                    {"description": "回答问候", "type": "reasoning"},
                ]
            }
        ),
        step_text,
        final_answer,
    ]


def _c3_failing_tool_script(plan_description, arguments_payload):
    """工具首次失败即停（Retry / Replan 在用例内被关闭）。"""
    return [
        json.dumps({"should_remember": False, "memories": []}),
        json.dumps(
            {
                "plan": [
                    {
                        "description": plan_description,
                        "type": "tool",
                        "tool_name": "query_orders",
                    },
                    {"description": "整理回答", "type": "reasoning"},
                ]
            }
        ),
        arguments_payload,
    ]


def _c3_patch_stop_retry(monkeypatch):
    """关闭 Retry / Replan，使一次失败即进入最终失败分支。"""
    import agent as agent_module

    monkeypatch.setattr(
        agent_module.Agent,
        "should_retry",
        lambda self, state, tool_name, result: False,
    )
    monkeypatch.setattr(
        agent_module.Agent,
        "should_replan",
        lambda self, state, result: False,
    )


# ========================================================
# 测试 C3-A：跨轮历史回放不得打印正文
# ========================================================

def test_C3_A_cross_turn_history_log_redacted(
    tmp_path, monkeypatch, temp_db, capsys
):
    """第一轮订单回答入库后，第二轮普通对话不得把它打印到控制台。"""
    import agent as agent_module

    first_round_answer = (
        f"订单 {C3_MEMORY_SECRET}，客户 {C3_MEMORY_CUSTOMER}，"
        f"金额 {C3_MEMORY_AMOUNT} 元，状态已支付。"
    )

    script = _c3_no_tool_script(
        "这一轮只是打个招呼。",
        "你好，有什么可以帮你的？",
    )
    calls = _install_scripted_openai(monkeypatch, script)
    monkeypatch.chdir(tmp_path)

    agent_obj = agent_module.Agent(**DUMMY)

    assert (tmp_path / "memory.db").exists()
    assert business_db.DATABASE_PATH.is_relative_to(tmp_path)

    # 模拟第一轮：订单回答已按原机制持久化
    agent_obj.memory.save_message("assistant", first_round_answer, user_id="test-user")

    capsys.readouterr()

    # 第二轮：普通对话，问句字符不命中历史内容 → 走 get_recent_messages 回退
    result = agent_obj.run("在吗", user_id="test-user")

    out = capsys.readouterr().out

    assert result["success"] is True
    assert result["answer"] == "你好，有什么可以帮你的？"

    # 控制台只剩元数据
    assert "[长度=" in out
    assert "[score=" in out

    # 正文与明细一律不出现
    assert C3_MEMORY_SECRET not in out
    assert C3_MEMORY_CUSTOMER not in out
    assert C3_MEMORY_AMOUNT not in out
    assert "状态已支付" not in out

    # 历史上下文仍完整提供给模型（只改日志，不改上下文）
    assert C3_MEMORY_SECRET in _call_text(calls[-1])
    assert C3_MEMORY_CUSTOMER in _call_text(calls[-1])

    # 历史数据未被删除或截断
    connection = sqlite3.connect(tmp_path / "memory.db")
    try:
        total = connection.execute(
            "SELECT COUNT(*) FROM messages"
        ).fetchone()[0]
        stored = connection.execute(
            "SELECT content FROM messages WHERE role = 'assistant'"
            " ORDER BY id ASC LIMIT 1"
        ).fetchone()[0]
    finally:
        connection.close()

    assert total >= 2
    assert stored == first_round_answer


# ========================================================
# 测试 C3-B：/chat HTTP 调试日志不得打印 answer
# ========================================================

def test_C3_B_chat_http_log_redacted(
    tmp_path, monkeypatch, temp_db, capsys
):
    """/chat 的调试输出只允许安全元数据，返回结构不变。"""
    import main as main_module
    from fastapi.testclient import TestClient

    http_answer = (
        f"订单 {C3_HTTP_SECRET}，客户 {C3_HTTP_CUSTOMER}，"
        f"金额 {C3_MEMORY_AMOUNT} 元。"
    )

    class _FakeAgent:
        def run(self, message, conversation_id=None, *, user_id=None):
            return {"success": True, "answer": http_answer}

    # /chat 现在会先落会话记录：必须把 Memory 锁定到 tmp_path，
    # 否则 get_memory() 会打开工程根目录的正式 memory.db。
    from memory import Memory

    monkeypatch.chdir(tmp_path)

    isolated_memory = Memory(db_path=str(tmp_path / "c3b_memory.db"))

    monkeypatch.setattr(
        main_module, "get_memory", lambda: isolated_memory
    )

    monkeypatch.setattr(main_module, "get_agent", lambda: _FakeAgent())

    # PAI-AUTH-MVP：/chat 要求登录。认证库与 memory 库同文件，
    # 这样 /chat 写入的归属就是本次登录用户的 id。
    from auth.service import AuthService, _reset_failures_for_tests

    service = AuthService(db_path=str(tmp_path / "c3b_memory.db"))
    service.ensure_schema()
    service.create_user("chat_tester", "Chat-Tester-2026", role="admin")
    _reset_failures_for_tests()

    capsys.readouterr()

    client = TestClient(main_module.app)
    logged_in = client.post(
        "/auth/login",
        json={"username": "chat_tester", "password": "Chat-Tester-2026"},
        headers={"Origin": "http://testserver"},
    )
    assert logged_in.status_code == 200, logged_in.text

    response = client.post("/chat", json={"message": "查询最近订单"})

    out = capsys.readouterr().out

    assert response.status_code == 200

    payload = response.json()

    # 返回结构保持不变
    assert payload["success"] is True
    assert payload["answer"] == http_answer
    assert payload["sources"] == []

    # 控制台只剩元数据
    assert "answer 长度：" in out
    assert "success：" in out
    assert "========== Agent 返回结果 ==========" in out

    # 旧调试输出与正文一律不出现
    assert "结果内容" not in out
    assert C3_HTTP_SECRET not in out
    assert C3_HTTP_CUSTOMER not in out
    assert C3_MEMORY_AMOUNT not in out


# ========================================================
# 测试 C3-C：订单工具失败日志不得打印原始 error
# ========================================================

def test_C3_C_order_tool_failure_log_redacted(
    tmp_path, monkeypatch, temp_db, capsys
):
    """非法 status / 非法 JSON / 工具异常三种失败，日志均只留分类。"""
    _c3_patch_stop_retry(monkeypatch)

    # ---------------- 场景 1：非法 status ----------------
    script = _c3_failing_tool_script(
        "查询订单",
        json.dumps({"status": C3_STATUS_VALUE}),
    )
    _, _, result = _run_agent_loop(
        monkeypatch, tmp_path, script, "查询订单"
    )
    out = capsys.readouterr().out

    assert result["success"] is False
    assert "❌ 工具执行失败：query_orders" in out
    assert "错误类型：订单查询失败" in out

    assert C3_STATUS_VALUE not in out
    assert "不在允许范围内" not in out

    # 错误仍完整传递给 Agent（只改日志，不改错误传播）
    assert C3_STATUS_VALUE in result["answer"]

    # ---------------- 场景 2：非法 JSON ----------------
    capsys.readouterr()

    bad_json = '{"order_id": "' + C3_JSON_SECRET + '"'

    script2 = _c3_failing_tool_script("查询订单", bad_json)
    _, _, result2 = _run_agent_loop(
        monkeypatch, tmp_path, script2, "查询订单"
    )
    out2 = capsys.readouterr().out

    assert result2["success"] is False
    assert "❌ 工具执行失败：query_orders" in out2
    assert "错误类型：参数生成格式错误（invalid_json）" in out2

    assert C3_JSON_SECRET not in out2
    assert "参数生成返回非法 JSON" not in out2

    # 原始文本仍保留在返回值中，供 Agent 纠错
    assert "参数生成返回非法 JSON" in result2["answer"]
    assert C3_JSON_SECRET in result2["answer"]

    # ---------------- 场景 3：工具执行异常 ----------------
    capsys.readouterr()

    def _boom(*args, **kwargs):
        raise sqlite3.OperationalError(
            f"unable to open database file: C:/{C3_DB_SECRET}/x.db"
        )

    monkeypatch.setattr(business_queries, "list_orders", _boom)

    script3 = _c3_failing_tool_script(
        "查询订单",
        json.dumps({"page_size": 1}),
    )
    _, _, result3 = _run_agent_loop(
        monkeypatch, tmp_path, script3, "查询订单"
    )
    out3 = capsys.readouterr().out

    assert result3["success"] is False
    assert "❌ 工具执行失败：query_orders" in out3
    assert "错误类型：订单查询失败" in out3

    assert C3_DB_SECRET not in out3
    assert "unable to open database file" not in out3
    assert "Traceback" not in out3


# ========================================================
# 测试 C3-D：计划描述在订单语境下隐藏，普通任务保留
# ========================================================

def test_C3_D_plan_description_log_redacted(
    tmp_path, monkeypatch, temp_db, capsys
):
    """订单任务隐藏 📝 描述 正文；普通任务保持原文。"""
    order_script = [
        json.dumps({"should_remember": False, "memories": []}),
        json.dumps(
            {
                "plan": [
                    {
                        "description": f"查询订单 {C3_PLAN_SECRET}",
                        "type": "reasoning",
                    },
                ]
            }
        ),
        "已收到订单任务。",
        "已按管理员全局范围完成查询。",
    ]

    _, _, result = _run_agent_loop(
        monkeypatch, tmp_path, order_script, "查询订单"
    )
    out = capsys.readouterr().out

    assert result["success"] is True
    assert "📝 描述：已隐藏" in out
    assert C3_PLAN_SECRET not in out

    # ---------------- 普通任务：描述原样保留 ----------------
    capsys.readouterr()

    plain_script = [
        json.dumps({"should_remember": False, "memories": []}),
        json.dumps(
            {
                "plan": [
                    {"description": "计算表达式", "type": "reasoning"},
                ]
            }
        ),
        "计算完成。",
        "计算结果是 3。",
    ]

    _, _, plain = _run_agent_loop(
        monkeypatch, tmp_path, plain_script, "帮我算 1+2"
    )
    out2 = capsys.readouterr().out

    assert plain["success"] is True
    assert "📝 描述：计算表达式" in out2
    assert "📝 描述：已隐藏" not in out2
