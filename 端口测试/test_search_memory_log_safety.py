# -*- coding: utf-8 -*-
"""search_memory 日志安全整改（R3-AGENT-ORDER-03B-02）隔离测试。

覆盖 T00–T08 共九项用例：

- T00  环境自证：网络守卫生效、两库隔离、模型替身不可调用
- T01  search_memory 虚构 query 标记不出现在 stdout
- T02  虚构记忆正文标记不出现在 stdout
- T03  虚构 reason 标记不出现在 stdout
- T04  长期记忆命中数量仍然正确输出（完整 Agent.run() 路径）
- T05  工具名称 / 成功状态 / 返回数量仍然正确输出
- T06  模型侧仍能获得完整记忆工具结果
- T07  query_orders 原有脱敏行为回归
- T08  calculator / get_current_time 等其他工具行为回归

安全约束（硬要求）：

1. 网络守卫在构造 Agent / OpenAI 对象之前生效：autouse fixture 先
   替换 socket.socket.connect / connect_ex 与 socket.create_connection，
   任何真实出站连接立即抛 AssertionError。T00 主动验证守卫有效。
2. 业务库是绝对路径（business/db.py 用 __file__ 推导），仅 chdir
   无效，必须替换 DATA_DIR / DATABASE_PATH；记忆库是相对路径
   （memory.py 默认 "memory.db"），chdir 到 tmp_path 即隔离。
3. 不读取 .env、不使用真实 API Key、不连接正式 data/business.db
   与项目根 memory.db。
4. 本文件只做验证，不修改任何业务源码。
"""

import json
import re
import socket
import sqlite3
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from business import db as business_db                # noqa: E402

# 虚构模型配置；base_url 使用 RFC 2606 保留域名，永不解析
DUMMY = dict(
    api_key="dummy-key",
    base_url="https://example.invalid",
    model="dummy-model",
)

# 虚构标记：任何一个出现在 stdout 都说明脱敏失效
QUERY_MARK = "标记Q7X2查询词"
VALUE_MARK = "标记V8K3记忆正文"
VALUE_MARK_B = "标记W4P6第二正文"
REASON_MARK = "标记R9M5保存原因"
REASON_MARK_B = "标记Y2N8第二原因"
ERROR_MARK = "标记E5T1异常文本"

# 03B-02-R1 专用标记：用于真实 Agent.run() 路径与模型请求 messages 断言
R1_SECRET_ERROR_MARKER = "R1_SECRET_ERROR_MARKER_9F3A7C"
R1_MEMORY_VALUE_MARKER = "R1_MEMORY_VALUE_MARKER_2B6D1E"
R1_MEMORY_REASON_MARKER = "R1_MEMORY_REASON_MARKER_7C4A5D"
R1_LONG_HEAD = "R1_LONG_HEAD_"
R1_LONG_TAIL = "R1_LONG_TAIL_MARKER_8E2F"

SECRET_NAME = "张三丰虚构"
SECRET_MAJOR = "计算机科学与技术虚构专业"

ORDER_MARK = "ORDER-LOG-SECRET-9527"
CUSTOMER_MARK = "CUST-LOG-SECRET-7411"

# 含第一人称 + 命中 LONG_TERM_MEMORY_INTENT_RULES 的「名字」规则，
# 同时通过主体守卫（无第三人称 / 第三方名词 / 假设词 / 集体主语）。
MEMORY_QUESTION = "我上次说的名字是什么"


# ========================================================
# 网络守卫（autouse，先于任何 Agent / OpenAI 构造）
# ========================================================

@pytest.fixture(autouse=True)
def network_guard(monkeypatch):
    """阻断一切真实出站连接，并记录被拦截的目标地址。"""

    attempts = []

    def _deny(address):
        attempts.append(address)
        raise AssertionError(
            f"测试期间禁止真实网络连接：{address!r}"
        )

    def _connect(self, address, *args, **kwargs):
        _deny(address)

    def _connect_ex(self, address, *args, **kwargs):
        _deny(address)

    def _create_connection(address, *args, **kwargs):
        _deny(address)

    monkeypatch.setattr(
        socket.socket, "connect", _connect, raising=True
    )
    monkeypatch.setattr(
        socket.socket, "connect_ex", _connect_ex, raising=True
    )
    monkeypatch.setattr(
        socket, "create_connection", _create_connection, raising=True
    )

    return attempts


# ========================================================
# 数据库隔离
# ========================================================

@pytest.fixture
def isolated(tmp_path, monkeypatch, network_guard):
    """业务库与记忆库双双锁定在 tmp_path 内。"""

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    db_path = data_dir / "business_log_safety_test.db"

    monkeypatch.setattr(business_db, "DATA_DIR", data_dir)
    monkeypatch.setattr(business_db, "DATABASE_PATH", db_path)
    business_db.init_database()

    # 记忆库走相对路径，chdir 即隔离
    monkeypatch.chdir(tmp_path)

    assert business_db.DATABASE_PATH.is_relative_to(tmp_path)
    assert db_path.exists()

    return db_path


# ========================================================
# 模型替身
# ========================================================

def _install_blocked_openai(monkeypatch):
    """构造可成功、任何模型调用立即失败的替身。"""

    import agent as agent_module

    class _BlockedCompletions:
        def create(self, *args, **kwargs):
            raise AssertionError("测试期间禁止调用真实模型 API")

    class _BlockedChat:
        def __init__(self):
            self.completions = _BlockedCompletions()

    class _BlockedOpenAI:
        def __init__(self, *args, **kwargs):
            self.init_kwargs = kwargs
            self.chat = _BlockedChat()

    monkeypatch.setattr(agent_module, "OpenAI", _BlockedOpenAI)


def _install_scripted_openai(monkeypatch, script):
    """按脚本顺序返回响应的替身，返回请求记录列表。"""

    import agent as agent_module

    queue = list(script)
    calls = []

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
        def create(self, model=None, messages=None, **kwargs):
            calls.append(
                {
                    "model": model,
                    "messages": messages,
                    "kwargs": kwargs,
                }
            )
            if not queue:
                raise AssertionError("Mock 模型收到脚本之外的额外调用")
            return _FakeResponse(queue.pop(0))

    class _ScriptedChat:
        def __init__(self):
            self.completions = _ScriptedCompletions()

    class _ScriptedOpenAI:
        def __init__(self, *args, **kwargs):
            self.init_kwargs = kwargs
            self.chat = _ScriptedChat()

    monkeypatch.setattr(agent_module, "OpenAI", _ScriptedOpenAI)

    return calls


def _install_adaptive_openai(
    monkeypatch,
    plan,
    params,
    replan_plan=None,
    extract=None,
    reasoning_text="已完成本轮判断，准备给出回答。",
):
    """按 system prompt 特征选择响应的自适应替身（03B-02-R1）。

    与 _install_scripted_openai 的按序出队不同，本替身根据请求内容
    判断当前阶段，因此不受 Agent Loop 内 Retry / Replan 次数变化的
    影响。完整记录每次请求的 messages，供模型侧上下文断言使用。
    """

    import agent as agent_module

    calls = []

    def _default_replan_plan():
        return {
            "plan": [
                {"description": "直接回答", "type": "reasoning"}
            ]
        }

    def _pick(system_text):

        if "请重新制定一个可执行计划" in system_text:
            return json.dumps(replan_plan or _default_replan_plan())

        if "工具参数 schema（必须严格遵守）" in system_text:
            return json.dumps(params)

        if "请根据当前执行情况判断下一步应该做什么" in system_text:
            return reasoning_text

        if '"should_remember"' in system_text:
            return json.dumps(
                extract
                if extract is not None
                else {"should_remember": False, "memories": []}
            )

        return json.dumps(plan)

    class _FakeMessage:
        def __init__(self, content):
            self.content = content

    class _FakeChoice:
        def __init__(self, content):
            self.message = _FakeMessage(content)

    class _FakeResponse:
        def __init__(self, content):
            self.choices = [_FakeChoice(content)]

    class _AdaptiveCompletions:
        def create(self, model=None, messages=None, **kwargs):
            calls.append(
                {
                    "model": model,
                    "messages": messages,
                    "kwargs": kwargs,
                }
            )

            system_text = "\n".join(
                (message.get("content") or "")
                for message in (messages or [])
                if message.get("role") == "system"
            )

            return _FakeResponse(_pick(system_text))

    class _AdaptiveChat:
        def __init__(self):
            self.completions = _AdaptiveCompletions()

    class _AdaptiveOpenAI:
        def __init__(self, *args, **kwargs):
            self.init_kwargs = kwargs
            self.chat = _AdaptiveChat()

    monkeypatch.setattr(agent_module, "OpenAI", _AdaptiveOpenAI)

    return calls


def _system_texts(calls):
    """取每次请求中所有 system message 的拼接文本。"""

    out = []

    for call in calls:
        out.append(
            "\n".join(
                (message.get("content") or "")
                for message in (call.get("messages") or [])
                if message.get("role") == "system"
            )
        )

    return out


@pytest.fixture
def agent_env(isolated, monkeypatch):
    """真实 Agent：记忆库落 tmp_path，模型客户端不可调用。

    PAI-AUTH-MVP：`search_memory` 工具每次调用都要解析当前用户身份，
    而本文件大量直接调用 execute_tool（不经过 Agent.run）。这里显式
    设置 ContextVar，模拟「处于某个用户请求内」的运行环境。
    """

    import agent as agent_module

    _install_blocked_openai(monkeypatch)

    agent_obj = agent_module.Agent(**DUMMY)

    assert (Path.cwd() / "memory.db").exists()

    token = agent_module._ACTIVE_USER_ID.set("test-user")

    yield agent_obj

    agent_module._ACTIVE_USER_ID.reset(token)


# ========================================================
# 数据准备
# ========================================================

def _seed_memory(agent_obj, key, value, category, reason):
    """写入一条虚构长期记忆；返回保存结果，断言确为新增。"""

    result = agent_obj.memory.save_memory(
        key=key,
        value=value,
        source="user",
        confidence=1.0,
        category=category,
        reason=reason, user_id="test-user"
    )

    assert result == "add"

    return result


def _insert_order(
    db_path,
    order_id,
    customer_id,
    status,
    created_at,
    amount,
    paid_at=None,
):
    """直接写入临时业务库，字段与 orders 表结构一致。"""

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
# T00 环境自证
# ========================================================

def test_T00_environment_guard(
    isolated, agent_env, network_guard
):
    """网络守卫、两库隔离与模型替身三项前提同时成立。"""

    # 1. 主动发起真实连接，必须被守卫拦截
    with pytest.raises(AssertionError):
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            probe.connect(("127.0.0.1", 9))
        finally:
            probe.close()

    assert network_guard, "网络守卫未记录任何连接尝试"

    # 2. 两库均在 tmp_path 内
    tmp_path = isolated.parent.parent

    assert business_db.DATABASE_PATH.is_relative_to(tmp_path)
    assert (tmp_path / "memory.db").exists()

    # 3. 模型替身构造成功，但调用即失败
    with pytest.raises(AssertionError):
        agent_env.client.chat.completions.create(
            model="dummy-model", messages=[]
        )


# ========================================================
# T01 查询文本不进入 stdout
# ========================================================

def test_T01_query_text_not_printed(agent_env, capsys):
    """虚构 query 标记不得出现在控制台。"""

    import agent as agent_module

    state = agent_module.AgentState(MEMORY_QUESTION)
    capsys.readouterr()

    outcome = agent_env.execute_tool(
        state, "search_memory", {"query": QUERY_MARK}
    )

    out = capsys.readouterr().out

    # 真实值未被篡改
    assert outcome["success"] is True
    assert outcome["result"]["query"] == QUERY_MARK

    # 脱敏后的参数摘要
    assert "📦 工具参数：search_memory" in out
    assert "参数字段：query" in out
    assert f"查询文本长度：{len(QUERY_MARK)}" in out
    assert "分类：未指定" in out

    # 原文不得出现
    assert QUERY_MARK not in out


# ========================================================
# T02 记忆正文不进入 stdout
# ========================================================

def test_T02_memory_value_not_printed(agent_env, capsys):
    """虚构记忆正文标记不得出现在控制台。"""

    import agent as agent_module

    _seed_memory(
        agent_env,
        key="name",
        value=VALUE_MARK + "，我的名字是" + SECRET_NAME,
        category="identity",
        reason=REASON_MARK,
    )

    state = agent_module.AgentState(MEMORY_QUESTION)
    capsys.readouterr()

    outcome = agent_env.execute_tool(
        state, "search_memory", {"query": "名字"}
    )

    out = capsys.readouterr().out

    assert outcome["success"] is True
    assert outcome["result"]["count"] == 1
    assert VALUE_MARK in outcome["result"]["memories"][0]["value"]

    assert "📤 工具结果：search_memory" in out
    assert "状态：成功" in out
    assert "返回数量：1" in out

    assert VALUE_MARK not in out
    assert SECRET_NAME not in out


# ========================================================
# T03 保存原因不进入 stdout
# ========================================================

def test_T03_reason_not_printed(agent_env, capsys):
    """虚构 reason 标记不得出现在控制台。"""

    import agent as agent_module

    _seed_memory(
        agent_env,
        key="major",
        value="我的专业是" + SECRET_MAJOR,
        category="profile",
        reason=REASON_MARK,
    )

    state = agent_module.AgentState(MEMORY_QUESTION)
    capsys.readouterr()

    outcome = agent_env.execute_tool(
        state, "search_memory", {"query": "专业"}
    )

    out = capsys.readouterr().out

    assert outcome["success"] is True
    assert outcome["result"]["count"] == 1
    assert outcome["result"]["memories"][0]["reason"] == REASON_MARK

    assert REASON_MARK not in out
    assert SECRET_MAJOR not in out


# ========================================================
# T04 长期记忆命中数量仍然正确输出（完整 run 路径）
# ========================================================

def test_T04_long_term_memory_hit_count_printed(
    isolated, monkeypatch, capsys
):
    """命中数量、分类标签、分数标签保留；key / value / reason 不外泄。"""

    import agent as agent_module

    script = [
        json.dumps({"should_remember": False, "memories": []}),
        json.dumps(
            {
                "plan": [
                    {
                        "description": "回忆名字",
                        "type": "tool",
                        "tool_name": "search_memory",
                    },
                    {"description": "整理回答", "type": "reasoning"},
                ]
            }
        ),
        json.dumps({"query": "名字"}),
        "已取得记忆结果，准备回答。",
        "已经查到你的名字了。",
    ]

    calls = _install_scripted_openai(monkeypatch, script)

    agent_obj = agent_module.Agent(**DUMMY)

    _seed_memory(
        agent_obj,
        key="name",
        value=VALUE_MARK + "，我的名字是" + SECRET_NAME,
        category="identity",
        reason=REASON_MARK,
    )
    _seed_memory(
        agent_obj,
        key="nickname",
        value=VALUE_MARK_B + "，名字缩写 ZS",
        category="identity",
        reason=REASON_MARK_B,
    )

    capsys.readouterr()

    result = agent_obj.run(MEMORY_QUESTION, user_id="test-user")

    out = capsys.readouterr().out

    assert result["success"] is True
    assert len(calls) == 5

    # 数量仍然正确输出
    assert "🧠 命中长期记忆数量：2" in out
    assert out.count("[key长度=") == 2

    # 受控分类标签与分数标签保留
    assert "[category=identity]" in out
    assert "key长度=" in out
    assert "value长度=" in out

    # key / value / reason 原文全部不出现
    for mark in (
        VALUE_MARK,
        VALUE_MARK_B,
        REASON_MARK,
        REASON_MARK_B,
        SECRET_NAME,
        "nickname",
    ):
        assert mark not in out


# ========================================================
# T05 工具名称 / 成功状态 / 返回数量仍然正确输出
# ========================================================

def test_T05_tool_status_and_count_printed(
    agent_env, capsys, monkeypatch
):
    """成功与失败两条分支的摘要字段都保留，异常文本不外泄。"""

    import agent as agent_module

    _seed_memory(
        agent_env,
        key="name",
        value=VALUE_MARK + "，我的名字是" + SECRET_NAME,
        category="identity",
        reason=REASON_MARK,
    )

    state = agent_module.AgentState(MEMORY_QUESTION)
    capsys.readouterr()

    # 成功分支
    agent_env.execute_tool(
        state, "search_memory", {"query": "名字"}
    )

    out = capsys.readouterr().out

    assert "🔧 执行工具：search_memory" in out
    assert "📤 工具结果：search_memory" in out
    assert "状态：成功" in out
    assert "返回数量：1" in out

    # 失败分支：工具内部异常
    def _boom(*args, **kwargs):
        raise RuntimeError("memory-backend-failure-" + ERROR_MARK)

    monkeypatch.setattr(
        agent_env.memory, "search_long_term_memory", _boom
    )

    capsys.readouterr()

    failed = agent_env.execute_tool(
        state, "search_memory", {"query": "名字"}
    )

    out2 = capsys.readouterr().out

    assert failed["success"] is False
    assert "📤 工具结果：search_memory" in out2
    assert "状态：失败" in out2
    assert "错误类型：未分类执行错误" in out2

    # 异常原文不得外泄
    assert ERROR_MARK not in out2
    assert "RuntimeError" not in out2


# ========================================================
# T06 模型侧仍能获得完整记忆工具结果
# ========================================================

def test_T06_full_result_reaches_model_side(
    agent_env, capsys, monkeypatch
):
    """脱敏只作用于日志；返回值、执行历史与序列化结果保持完整。

    范围声明（03B-02-R1 复审修正）：本用例走 execute_tool 直调，
    不产生任何模型请求，因此只验证到 execution_history 及其 JSON
    序列化层，**不构成对模型请求 messages 的验证**。模型请求消息
    完整性由 test_T06_R1_model_request_messages_contain_memory 覆盖。
    """

    import agent as agent_module

    _seed_memory(
        agent_env,
        key="name",
        value=VALUE_MARK + "，我的名字是" + SECRET_NAME,
        category="identity",
        reason=REASON_MARK,
    )

    spy = Mock(wraps=agent_env.memory.search_long_term_memory)
    monkeypatch.setattr(
        agent_env.memory, "search_long_term_memory", spy
    )

    state = agent_module.AgentState(MEMORY_QUESTION)
    capsys.readouterr()

    outcome = agent_env.execute_tool(
        state, "search_memory", {"query": "名字"}
    )

    # 入参未被日志函数修改
    assert spy.call_count == 1
    assert spy.call_args.args[0] == "名字"
    assert spy.call_args.kwargs["limit"] == 5

    # 返回值完整
    payload = outcome["result"]

    assert payload["query"] == "名字"
    assert payload["count"] == 1

    memory = payload["memories"][0]

    assert memory["key"] == "name"
    assert memory["value"] == VALUE_MARK + "，我的名字是" + SECRET_NAME
    assert memory["reason"] == REASON_MARK
    assert memory["category"] == "identity"

    for field in (
        "id",
        "source",
        "confidence",
        "created_at",
        "updated_at",
        "score",
    ):
        assert field in memory

    # 执行历史保存同一份完整结构（reasoning 上下文来源）
    assert len(state.execution_history) == 1

    entry = state.execution_history[0]

    assert entry["type"] == "tool"
    assert entry["tool_name"] == "search_memory"
    assert entry["arguments"] == {"query": "名字"}
    assert entry["result"] is outcome

    # 序列化后仍含正文：模型侧 prompt 直接使用这段 JSON
    serialized = json.dumps(
        state.execution_history, ensure_ascii=False
    )

    assert VALUE_MARK in serialized
    assert REASON_MARK in serialized
    assert SECRET_NAME in serialized

    # 而 stdout 仍然不含
    out = capsys.readouterr().out

    assert VALUE_MARK not in out
    assert REASON_MARK not in out
    assert SECRET_NAME not in out


# ========================================================
# T07 query_orders 原有脱敏行为回归
# ========================================================

def test_T07_query_orders_logging_regression(
    isolated, agent_env, capsys
):
    """订单工具的日志脱敏不受本次改动影响。"""

    import agent as agent_module

    _insert_order(
        isolated,
        ORDER_MARK,
        customer_id=CUSTOMER_MARK,
        status="已支付",
        created_at="2026-11-05T10:00:00",
        amount=1234.56,
        paid_at="2026-11-05T11:00:00",
    )

    state = agent_module.AgentState("最近的订单")
    capsys.readouterr()

    outcome = agent_env.execute_tool(
        state, "query_orders", {"page_size": 2}
    )

    out = capsys.readouterr().out

    assert outcome["success"] is True
    assert outcome["result"]["count"] == 1

    # 参数：白名单字段名 + 分页数值
    assert "📦 工具参数：query_orders" in out
    assert "参数字段：page_size" in out
    assert "每页数量：2" in out

    # 结果：状态 + 计数，不含明细
    assert "📤 工具结果：query_orders" in out
    assert "状态：成功" in out
    assert "返回数量：1" in out
    assert "总记录数：1" in out

    # 订单 / 客户编号与金额不出现在 stdout
    assert ORDER_MARK not in out
    assert CUSTOMER_MARK not in out
    assert "1234.56" not in out

    # 明细仍在返回值里
    assert outcome["result"]["items"][0]["order_id"] == ORDER_MARK

    # 业务校验失败分支：非法状态原值不得回显
    capsys.readouterr()

    bad = agent_env.execute_tool(
        state, "query_orders", {"status": "PAID"}
    )

    out2 = capsys.readouterr().out

    assert bad["success"] is False
    assert "📦 工具参数：query_orders" in out2
    assert "参数字段：status" in out2
    assert "📤 工具结果：query_orders" in out2
    assert "状态：失败" in out2
    assert "PAID" not in out2


# ========================================================
# T08 其他工具行为回归
# ========================================================

def test_T08_other_tools_logging_unchanged(agent_env, capsys):
    """calculator / get_current_time 仍走通用原样输出分支。"""

    import agent as agent_module

    state = agent_module.AgentState("帮我算 1+2")
    capsys.readouterr()

    calc = agent_env.execute_tool(
        state, "calculator", {"expression": "1+2"}
    )

    out = capsys.readouterr().out

    assert calc["success"] is True
    assert calc["result"] == "3"

    # 非 search_memory / query_orders 工具保持原有原文输出
    assert "📦 参数：{'expression': '1+2'}" in out
    assert "📤 工具结果：{'success': True, 'result': '3'}" in out

    assert "📦 工具参数：search_memory" not in out
    assert "📦 工具参数：query_orders" not in out

    # get_current_time：空参数与时间字符串原样输出
    state2 = agent_module.AgentState("现在几点")
    capsys.readouterr()

    now = agent_env.execute_tool(state2, "get_current_time", {})

    out2 = capsys.readouterr().out

    assert now["success"] is True
    assert re.fullmatch(
        r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", now["result"]
    )

    assert "📦 参数：{}" in out2
    assert (
        "📤 工具结果：{'success': True, 'result': '"
        + now["result"]
        + "'}"
    ) in out2

    # 其他工具的失败分支仍打印原始校验错误
    state3 = agent_module.AgentState("帮我算 1+2")
    capsys.readouterr()

    failure = agent_env.execute_tool(
        state3, "calculator", {"expression": 123}
    )

    out3 = capsys.readouterr().out

    assert failure["success"] is False
    assert "⛔ 参数预检未通过：工具参数校验失败：" in out3
    assert "参数 expression 类型错误" in out3


# ========================================================
# 03B-02-R1：真实 Agent.run() 路径与模型请求 messages 证据
# ========================================================

def _r1_plan_with_search_memory():
    """含一个 search_memory 工具步骤的计划。"""

    return {
        "plan": [
            {
                "description": "回忆名字",
                "type": "tool",
                "tool_name": "search_memory",
            },
            {"description": "整理回答", "type": "reasoning"},
        ]
    }


def test_T05_R1_agent_run_tool_failure_log_redacted(
    isolated, monkeypatch, capsys
):
    """工具内部异常：真正经过 Agent.run() 与计划步骤，两个失败出口均脱敏。"""

    import agent as agent_module

    calls = _install_adaptive_openai(
        monkeypatch,
        plan=_r1_plan_with_search_memory(),
        params={"query": "名字"},
    )

    agent_obj = agent_module.Agent(**DUMMY)

    original = agent_obj.memory.search_long_term_memory

    def _boom(*args, **kwargs):
        # 只让 search_memory 工具路径（limit=5）失败；
        # retrieve_long_term_memories（limit=3）保持正常，
        # 避免误触本轮授权范围之外的长期记忆检索异常出口。
        if kwargs.get("limit") == 5:
            raise RuntimeError(R1_SECRET_ERROR_MARKER)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        agent_obj.memory, "search_long_term_memory", _boom
    )

    capsys.readouterr()

    agent_obj.run(MEMORY_QUESTION, user_id="test-user")

    out = capsys.readouterr().out

    # 1. 唯一异常标记不得出现在 stdout
    assert R1_SECRET_ERROR_MARKER not in out

    # 2. 两个失败出口都只输出固定分类
    assert "📤 工具结果：search_memory" in out
    assert "❌ 工具执行失败：search_memory" in out
    assert "错误类型：未分类执行错误" in out

    # 3. 确认经过真实 Agent Loop 的失败处理机制
    assert "🔁 Retry" in out
    assert "🔄 Replan" in out
    assert len(calls) >= 5

    # 4. 模型仍能拿到原始失败结果（进入后续请求 messages）
    texts = _system_texts(calls)
    assert any(R1_SECRET_ERROR_MARKER in text for text in texts)


def test_T05_R1_agent_run_validation_failure_log_redacted(
    isolated, monkeypatch, capsys
):
    """参数预检失败：经 Agent.run()，预检与失败两个出口均脱敏。"""

    import agent as agent_module

    calls = _install_adaptive_openai(
        monkeypatch,
        plan=_r1_plan_with_search_memory(),
        params={"query": 123},
    )

    agent_obj = agent_module.Agent(**DUMMY)

    capsys.readouterr()

    agent_obj.run(MEMORY_QUESTION, user_id="test-user")

    out = capsys.readouterr().out

    # 预检出口：固定分类
    assert "⛔ 参数预检未通过：search_memory" in out
    assert "错误类型：参数校验失败" in out
    assert "记忆参数值：已隐藏" in out

    # 失败出口：同样是固定分类
    assert "❌ 工具执行失败：search_memory" in out

    # 原始校验文本不得出现
    assert "参数 query 类型错误" not in out
    assert "需要字符串" not in out
    assert "工具参数校验失败：" not in out

    # 确认经过真实 Agent Loop
    assert len(calls) >= 5

    # 模型仍能拿到原始校验错误
    texts = _system_texts(calls)
    assert any("参数 query 类型错误" in text for text in texts)


def test_T06_R1_model_request_messages_contain_memory(
    isolated, monkeypatch, capsys
):
    """直接断言 Fake 模型实际收到的 system message 含完整记忆内容。"""

    import agent as agent_module

    calls = _install_adaptive_openai(
        monkeypatch,
        plan=_r1_plan_with_search_memory(),
        params={"query": "名字"},
    )

    agent_obj = agent_module.Agent(**DUMMY)

    _seed_memory(
        agent_obj,
        key="name",
        value=R1_MEMORY_VALUE_MARKER + "，我的名字是" + SECRET_NAME,
        category="identity",
        reason=R1_MEMORY_REASON_MARKER,
    )

    capsys.readouterr()

    agent_obj.run(MEMORY_QUESTION, user_id="test-user")

    out = capsys.readouterr().out

    texts = _system_texts(calls)

    # 找到工具执行之后、携带执行历史的 reasoning 请求
    with_history = [
        text
        for text in texts
        if "执行历史：" in text and "search_memory" in text
    ]

    assert with_history, "未找到携带 search_memory 执行历史的模型请求"

    target = with_history[-1]

    # 直接断言模型请求消息内容（不是 execution_history，不是中间变量）
    assert R1_MEMORY_VALUE_MARKER in target
    assert R1_MEMORY_REASON_MARKER in target
    assert SECRET_NAME in target
    assert '"tool_name": "search_memory"' in target

    # 同时 stdout 仍不含
    for mark in (
        R1_MEMORY_VALUE_MARKER,
        R1_MEMORY_REASON_MARKER,
        SECRET_NAME,
    ):
        assert mark not in out


def test_T09_long_term_memory_truncation_boundary(
    isolated, monkeypatch, capsys
):
    """长期记忆 prompt 路径按现值截断 100 字；工具历史路径保留完整正文。"""

    import agent as agent_module

    # 构造 >100 字的长值：前 100 字含「名字」词元（保证工具检索命中），
    # 尾部标记落在第 100 字之后（只会出现在未截断的路径里）。
    head = R1_LONG_HEAD + "我的名字是"
    long_value = head + ("X" * (100 - len(head))) + R1_LONG_TAIL

    assert len(long_value) > 100
    assert long_value[:100] == head + ("X" * (100 - len(head)))
    assert R1_LONG_TAIL not in long_value[:100]

    calls = _install_adaptive_openai(
        monkeypatch,
        plan=_r1_plan_with_search_memory(),
        params={"query": "名字"},
    )

    agent_obj = agent_module.Agent(**DUMMY)

    _seed_memory(
        agent_obj,
        key="name",
        value=long_value,
        category="identity",
        reason="长值边界用例",
    )

    capsys.readouterr()

    agent_obj.run(MEMORY_QUESTION, user_id="test-user")

    out = capsys.readouterr().out

    texts = _system_texts(calls)

    target = None
    for text in texts:
        if (
            "执行历史：" in text
            and '"tool_name": "search_memory"' in text
            and R1_LONG_HEAD in text
        ):
            target = text
            break

    assert target is not None, "未找到同时含长期记忆与工具执行历史的模型请求"

    # 路径一：长期记忆 prompt 注入，按现有规则截断 100 字 + 省略号
    match = re.search(r"- name = (.*?)（分类：", target, re.S)

    assert match is not None, "未找到长期记忆注入段落"

    injected = match.group(1)

    assert injected == long_value[:100] + "…"
    assert len(injected) == 101
    assert R1_LONG_TAIL not in injected

    # 路径二：工具执行历史，保留完整正文（不因日志脱敏而截断）
    assert R1_LONG_TAIL in target
    assert long_value in target

    # 两条路径都不影响 stdout 脱敏
    assert R1_LONG_HEAD not in out
    assert R1_LONG_TAIL not in out
