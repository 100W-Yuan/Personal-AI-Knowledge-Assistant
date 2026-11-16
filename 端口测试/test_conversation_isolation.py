# -*- coding: utf-8 -*-
"""阶段 1：会话基础设施与历史消息隔离专项测试（离线）。

覆盖 T01–T10：
- 会话创建、消息绑定、历史隔离、Agent 调用隔离；
- 历史读取接口、前端请求契约、数据库迁移、并发隔离；
- 长期记忆兼容、旧 API 兼容。

安全约束：
- 所有数据库都建在 pytest 的 tmp_path 内，构造任何 Memory / Agent 前先
  chdir 到 tmp_path，正式 memory.db 全程不被打开；
- autouse 网络守卫拦截 socket 出站，并做主动自检；
- 模型客户端一律替换为按 prompt 特征应答的假客户端，真实 API 调用为 0。
"""

import json
import socket
import sqlite3
import sys
import threading
import uuid
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 虚构模型配置；base_url 使用 RFC 2606 保留域名，永不解析
DUMMY = dict(
    api_key="dummy-key",
    base_url="https://example.invalid",
    model="dummy-model",
)

CONV_A_SECRET_MARKER = "CONV_A_SECRET_MARKER_4B7D"
CONV_B_SECRET_MARKER = "CONV_B_SECRET_MARKER_9K2F"

#: PAI-AUTH-MVP：本文件所有 Memory 调用统一归属到该用户。
#: http_env 夹具会把登录账号的 id 改写成同一个值，使 HTTP 层与
#: 直连 Memory 写入的数据指向同一个所有者。
TEST_USER_ID = "test-user"
TEST_USERNAME = "isolation_tester"
TEST_PASSWORD = "Isolation-Pass-2026"

REASONING_ONLY_PLAN = {
    "plan": [
        {"description": "直接回答", "type": "reasoning"}
    ]
}

ANSWER_TEXT = "这是测试用回答。"


# ========================================================
# 网络守卫（autouse，先于任何 Agent / OpenAI 构造）
# ========================================================

@pytest.fixture(autouse=True)
def network_guard(monkeypatch):
    """阻断一切真实出站连接，并记录被拦截的目标地址。

    FastAPI 的 TestClient 走的是进程内回环传输，因此只放行 127.0.0.1 /
    ::1 这类回环地址；任何外部地址一律拦截并记录。真实模型域名
    （example.invalid）与公网目标都会被拒绝。
    """

    attempts = []

    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_create_connection = socket.create_connection

    loopback_hosts = {"127.0.0.1", "::1", "localhost"}

    def _is_loopback(address):
        if isinstance(address, tuple) and address:
            return address[0] in loopback_hosts
        return False

    def _deny(address):
        attempts.append(address)
        raise AssertionError(
            f"测试期间禁止真实网络连接：{address!r}"
        )

    def _connect(self, address, *args, **kwargs):
        if _is_loopback(address):
            return original_connect(self, address, *args, **kwargs)
        _deny(address)

    def _connect_ex(self, address, *args, **kwargs):
        if _is_loopback(address):
            return original_connect_ex(self, address, *args, **kwargs)
        _deny(address)

    def _create_connection(address, *args, **kwargs):
        if _is_loopback(address):
            return original_create_connection(address, *args, **kwargs)
        _deny(address)

    monkeypatch.setattr(socket.socket, "connect", _connect, raising=True)
    monkeypatch.setattr(socket.socket, "connect_ex", _connect_ex, raising=True)
    monkeypatch.setattr(
        socket, "create_connection", _create_connection, raising=True
    )

    return attempts


@pytest.fixture
def isolated(tmp_path, monkeypatch, network_guard):
    """把工作目录切到 tmp_path，记忆库因此落在临时目录。"""

    monkeypatch.chdir(tmp_path)

    assert Path.cwd() == tmp_path

    return tmp_path


# ========================================================
# 模型替身
# ========================================================

def _install_adaptive_openai(monkeypatch, answer=ANSWER_TEXT):
    """按 system prompt 特征应答的假客户端，线程安全地记录全部请求。"""

    import agent as agent_module

    calls = []
    lock = threading.Lock()

    def _pick(system_text):

        if "请重新制定一个可执行计划" in system_text:
            return json.dumps(REASONING_ONLY_PLAN)

        if "工具参数 schema（必须严格遵守）" in system_text:
            return json.dumps({})

        if "请根据当前执行情况判断下一步应该做什么" in system_text:
            return answer

        if '"should_remember"' in system_text:
            return json.dumps({"should_remember": False, "memories": []})

        return json.dumps(REASONING_ONLY_PLAN)

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
            with lock:
                calls.append(
                    {
                        "model": model,
                        "messages": messages,
                        "thread": threading.get_ident(),
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
            self.chat = _AdaptiveChat()

    monkeypatch.setattr(agent_module, "OpenAI", _AdaptiveOpenAI)

    return calls


def _system_texts(calls):
    """取每次请求中所有 system message 的拼接文本。"""

    return [
        "\n".join(
            (message.get("content") or "")
            for message in (call.get("messages") or [])
            if message.get("role") == "system"
        )
        for call in calls
    ]


# ========================================================
# T00：环境守卫自检
# ========================================================

def test_T00_environment_guard(isolated, network_guard):
    """确认临时目录隔离生效，且网络守卫确实拦截出站连接。"""

    assert Path.cwd() == isolated
    assert (isolated / "memory.db").exists() is False

    # 外部地址必须被拦截（回环仅供 TestClient 的进程内传输使用）
    with pytest.raises(AssertionError):
        socket.create_connection(("example.invalid", 443), timeout=0.01)

    assert network_guard, "网络守卫未记录被拦截的连接"
    network_guard.clear()


# ========================================================
# T01：新会话创建
# ========================================================

def test_T01_create_and_read_conversation(isolated):
    """能生成合法 UUID、保存会话、重复读取到同一会话。"""

    from memory import Memory, validate_conversation_id

    memory = Memory(db_path=str(isolated / "t01.db"))

    conversation_id = str(uuid.uuid4())

    assert validate_conversation_id(conversation_id) == conversation_id

    created = memory.create_conversation(conversation_id, title="第一段", user_id="test-user")

    assert created["conversation_id"] == conversation_id
    assert created["title"] == "第一段"
    assert created["created_at"]
    assert created["updated_at"]

    again = memory.get_conversation(conversation_id, user_id="test-user")

    assert again["conversation_id"] == conversation_id
    assert again["title"] == "第一段"

    # 重复创建不覆盖已有标题与时间
    memory.create_conversation(conversation_id, title="改名应被忽略", user_id="test-user")

    assert memory.get_conversation(conversation_id, user_id="test-user")["title"] == "第一段"

    listed = memory.list_conversations(user_id="test-user")

    assert [item["conversation_id"] for item in listed] == [conversation_id]

    # 自动生成 UUID 的路径
    auto = memory.create_conversation(user_id="test-user")

    assert validate_conversation_id(auto["conversation_id"])
    assert auto["title"] == "新对话"


def test_T01_delete_conversation_protects_legacy(isolated):
    """删除会话会连带删除消息，但 legacy 会话禁止删除。"""

    from memory import LEGACY_CONVERSATION_ID, Memory

    memory = Memory(db_path=str(isolated / "t01b.db"))

    conversation_id = str(uuid.uuid4())
    memory.save_message("user", "待删除", conversation_id=conversation_id, user_id="test-user")
    memory.save_message("assistant", "待删除回复", conversation_id=conversation_id, user_id="test-user")

    result = memory.delete_conversation(conversation_id, user_id="test-user")

    assert result["deleted_messages"] == 2
    assert memory.get_conversation(conversation_id, user_id="test-user") is None
    assert memory.get_messages(conversation_id, user_id="test-user")[1] == 0

    memory.save_message("user", "历史消息", user_id="test-user")

    with pytest.raises(ValueError):
        memory.delete_conversation(LEGACY_CONVERSATION_ID, user_id="test-user")

    assert memory.get_messages(LEGACY_CONVERSATION_ID, user_id="test-user")[1] == 1


# ========================================================
# T02：消息绑定
# ========================================================

def test_T02_messages_bound_to_conversation(isolated):
    """user / assistant 消息都属于指定会话，且刷新 updated_at。"""

    from memory import Memory

    memory = Memory(db_path=str(isolated / "t02.db"))

    conversation_id = str(uuid.uuid4())

    memory.save_message("user", "用户问题", conversation_id=conversation_id, user_id="test-user")
    memory.save_message("assistant", "助手回答", conversation_id=conversation_id, user_id="test-user")

    messages, total = memory.get_messages(conversation_id, user_id="test-user")

    assert total == 2
    assert [item["role"] for item in messages] == ["user", "assistant"]
    assert [item["content"] for item in messages] == ["用户问题", "助手回答"]

    # 按 id 正序返回，且带 message_id
    assert messages[0]["id"] < messages[1]["id"]

    conversation = memory.get_conversation(conversation_id, user_id="test-user")

    assert conversation["updated_at"] >= conversation["created_at"]

    # 保存消息前会自动建立会话（明确的 get-or-create）
    fresh_id = str(uuid.uuid4())

    assert memory.get_conversation(fresh_id, user_id="test-user") is None

    memory.save_message("user", "自动建会话", conversation_id=fresh_id, user_id="test-user")

    assert memory.get_conversation(fresh_id, user_id="test-user")["conversation_id"] == fresh_id

    listed = {
        item["conversation_id"]: item
        for item in memory.list_conversations(user_id="test-user")
    }

    assert listed[conversation_id]["message_count"] == 2
    assert listed[fresh_id]["message_count"] == 1


# ========================================================
# T03：历史隔离
# ========================================================

def test_T03_history_and_search_are_isolated(isolated):
    """A 只能读到 A，B 只能读到 B，双向都不串。"""

    from memory import Memory

    memory = Memory(db_path=str(isolated / "t03.db"))

    conv_a = str(uuid.uuid4())
    conv_b = str(uuid.uuid4())

    memory.save_message(
        "user", f"A 的问题 {CONV_A_SECRET_MARKER}", conversation_id=conv_a, user_id="test-user"
    )
    memory.save_message("assistant", "A 的回答", conversation_id=conv_a, user_id="test-user")

    memory.save_message(
        "user", f"B 的问题 {CONV_B_SECRET_MARKER}", conversation_id=conv_b, user_id="test-user"
    )
    memory.save_message("assistant", "B 的回答", conversation_id=conv_b, user_id="test-user")

    a_messages, a_total = memory.get_messages(conv_a, user_id="test-user")
    b_messages, b_total = memory.get_messages(conv_b, user_id="test-user")

    assert a_total == 2 and b_total == 2

    a_text = "\n".join(item["content"] for item in a_messages)
    b_text = "\n".join(item["content"] for item in b_messages)

    assert CONV_A_SECRET_MARKER in a_text
    assert CONV_B_SECRET_MARKER not in a_text

    assert CONV_B_SECRET_MARKER in b_text
    assert CONV_A_SECRET_MARKER not in b_text

    # get_recent_messages 也必须按会话过滤
    a_recent = memory.get_recent_messages(limit=10, conversation_id=conv_a, user_id="test-user")

    assert all(
        CONV_B_SECRET_MARKER not in item["content"] for item in a_recent
    )

    # search_memories 不得跨会话命中
    # 字符打分对共享字母是宽容的，因此按「返回内容是否含对方标记」判定，
    # 而不是要求结果集为空。
    a_hits = memory.search_memories(
        CONV_B_SECRET_MARKER, limit=5, conversation_id=conv_a, user_id="test-user"
    )
    b_hits = memory.search_memories(
        CONV_A_SECRET_MARKER, limit=5, conversation_id=conv_b, user_id="test-user"
    )

    assert all(
        CONV_B_SECRET_MARKER not in item["content"] for item in a_hits
    )
    assert all(
        CONV_A_SECRET_MARKER not in item["content"] for item in b_hits
    )

    own_hit = memory.search_memories(
        CONV_A_SECRET_MARKER, limit=5, conversation_id=conv_a, user_id="test-user"
    )

    assert any(
        CONV_A_SECRET_MARKER in item["content"] for item in own_hit
    )


# ========================================================
# T04：Agent 调用隔离
# ========================================================

def test_T04_agent_model_requests_do_not_mix_conversations(
    isolated, monkeypatch
):
    """Fake 模型实际收到的 messages 不包含对方会话的历史。"""

    import agent as agent_module

    calls = _install_adaptive_openai(monkeypatch)

    agent_obj = agent_module.Agent(**DUMMY)

    conv_a = str(uuid.uuid4())
    conv_b = str(uuid.uuid4())

    first = agent_obj.run(
        f"请记住这条 A 会话内容 {CONV_A_SECRET_MARKER}", conv_a, user_id="test-user"
    )

    assert first["success"] is True

    calls_after_a1 = len(calls)

    # 第二轮 A：此时 A 的历史必须进入提示词（正向对照）
    second = agent_obj.run("继续", conv_a, user_id="test-user")

    assert second["success"] is True

    calls_a2 = calls[calls_after_a1:]

    assert calls_a2, "A 的第二轮没有产生任何模型请求"

    texts_a2 = _system_texts(calls_a2)

    assert any(
        CONV_A_SECRET_MARKER in text for text in texts_a2
    ), "A 自己的历史没有进入模型请求"

    assert all(
        CONV_B_SECRET_MARKER not in text for text in texts_a2
    )

    calls_before_b = len(calls)

    third = agent_obj.run(
        f"请记住这条 B 会话内容 {CONV_B_SECRET_MARKER}", conv_b, user_id="test-user"
    )

    assert third["success"] is True

    texts_b = _system_texts(calls[calls_before_b:])

    assert texts_b

    assert all(
        CONV_A_SECRET_MARKER not in text for text in texts_b
    ), "B 会话的模型请求混入了 A 会话历史"

    # 数据库层面同样隔离
    a_messages, a_total = agent_obj.memory.get_messages(conv_a, user_id="test-user")
    b_messages, b_total = agent_obj.memory.get_messages(conv_b, user_id="test-user")

    assert a_total == 4 and b_total == 2
    assert all(
        CONV_B_SECRET_MARKER not in item["content"] for item in a_messages
    )
    assert all(
        CONV_A_SECRET_MARKER not in item["content"] for item in b_messages
    )


# ========================================================
# T05：刷新恢复接口
# ========================================================

class _FakeAgent:
    """记录收到的 message / conversation_id / user_id。"""

    def __init__(self):
        self.calls = []

    def run(self, message, conversation_id="legacy", *, user_id=None):
        self.calls.append((message, conversation_id, user_id))
        return {"success": True, "answer": "回答", "sources": []}


@pytest.fixture
def http_env(isolated, monkeypatch):
    """把 /chat 与 /conversations 依赖替换为临时库 + 假 Agent。

    PAI-AUTH-MVP：这些接口现在都要求登录，因此夹具额外建一个
    id 恰为 ``test-user`` 的账号并完成登录——与本文件其余用例写入的
    user_id 保持一致，否则读到的永远是空结果。
    """

    from fastapi.testclient import TestClient

    import main
    from auth.service import AuthService, _reset_failures_for_tests
    from memory import Memory

    memory = Memory(db_path=str(isolated / "http.db"))
    fake_agent = _FakeAgent()

    monkeypatch.setattr(main, "get_memory", lambda: memory)
    monkeypatch.setattr(main, "get_agent", lambda: fake_agent)

    service = AuthService(db_path=str(isolated / "http.db"))
    service.ensure_schema()
    service.create_user(TEST_USERNAME, TEST_PASSWORD, role="admin")

    connection = sqlite3.connect(service.db_path)
    try:
        connection.execute(
            "UPDATE users SET id = ? WHERE username = ?",
            (TEST_USER_ID, TEST_USERNAME),
        )
        connection.commit()
    finally:
        connection.close()
    _reset_failures_for_tests()

    client = TestClient(main.app, follow_redirects=False)
    response = client.post(
        "/auth/login",
        json={"username": TEST_USERNAME, "password": TEST_PASSWORD},
        headers={"Origin": "http://testserver"},
    )
    assert response.status_code == 200, response.text

    return client, memory, fake_agent


def test_T05_messages_endpoint_returns_only_own_conversation(http_env):
    """GET 只返回指定会话消息；空会话返回空数组；非法 ID 返回 4xx。"""

    client, memory, _ = http_env

    conv_a = str(uuid.uuid4())
    conv_b = str(uuid.uuid4())

    memory.save_message("user", f"A {CONV_A_SECRET_MARKER}", conversation_id=conv_a, user_id="test-user")
    memory.save_message("assistant", "A 回复", conversation_id=conv_a, user_id="test-user")
    memory.save_message("user", f"B {CONV_B_SECRET_MARKER}", conversation_id=conv_b, user_id="test-user")

    response = client.get(f"/conversations/{conv_a}/messages")

    assert response.status_code == 200

    body = response.json()

    assert body["success"] is True
    assert body["conversation_id"] == conv_a
    assert body["total"] == 2
    assert body["has_more"] is False
    assert [item["role"] for item in body["messages"]] == ["user", "assistant"]
    assert set(body["messages"][0]) == {"id", "role", "content", "created_at"}
    assert all(
        CONV_B_SECRET_MARKER not in item["content"]
        for item in body["messages"]
    )

    # 空会话
    empty_id = str(uuid.uuid4())
    empty = client.get(f"/conversations/{empty_id}/messages")

    assert empty.status_code == 200
    assert empty.json()["messages"] == []
    assert empty.json()["total"] == 0

    # 非法 ID
    for bad in ["not-a-uuid", "x" * 100, "12345"]:
        invalid = client.get(f"/conversations/{bad}/messages")
        assert invalid.status_code == 400, bad
        assert invalid.json()["detail"]["type"] == "invalid_conversation_id"

    # limit 非法
    too_big = client.get(
        f"/conversations/{conv_a}/messages", params={"limit": 9999}
    )
    assert too_big.status_code == 400


def test_T05_messages_endpoint_reports_truncation(http_env):
    """limit 截断时必须如实返回 has_more，不能静默丢历史。"""

    client, memory, _ = http_env

    conversation_id = str(uuid.uuid4())

    for index in range(5):
        memory.save_message(
            "user", f"第 {index} 条", conversation_id=conversation_id, user_id="test-user"
        )

    response = client.get(
        f"/conversations/{conversation_id}/messages", params={"limit": 2}
    )

    body = response.json()

    assert response.status_code == 200
    assert body["total"] == 5
    assert len(body["messages"]) == 2
    assert body["has_more"] is True
    # 返回的是最近 2 条，且按时间正序
    assert [item["content"] for item in body["messages"]] == ["第 3 条", "第 4 条"]


def test_T05_conversations_list_hides_message_bodies(http_env):
    """会话列表只返回元数据，不含消息正文。"""

    client, memory, _ = http_env

    conv_a = str(uuid.uuid4())
    memory.save_message("user", f"机密 {CONV_A_SECRET_MARKER}", conversation_id=conv_a, user_id="test-user")

    response = client.get("/conversations")

    assert response.status_code == 200

    body = response.json()

    assert body["success"] is True
    assert body["total"] == 1
    assert body["conversations"][0]["conversation_id"] == conv_a
    assert body["conversations"][0]["message_count"] == 1
    assert CONV_A_SECRET_MARKER not in json.dumps(body, ensure_ascii=False)


# ========================================================
# T06：前端请求契约（静态检查）
# ========================================================

def test_T06_frontend_sends_and_persists_conversation_id(isolated):
    """前端必须用 crypto.randomUUID + localStorage，并在 /chat 带上 ID。"""

    app_js = (PROJECT_ROOT / "app_ui" / "static" / "app.js").read_text(
        encoding="utf-8"
    )
    index_html = (PROJECT_ROOT / "app_ui" / "index.html").read_text(
        encoding="utf-8"
    )

    assert "crypto.randomUUID()" in app_js
    assert "my_ai_current_conversation_id" in app_js
    assert 'fetch("/chat"' in app_js
    assert "conversation_id: conversationId" in app_js
    assert '"/conversations/" +' in app_js
    assert "startNewConversation" in app_js
    assert "restoreConversation" in app_js

    # 禁止用随机整数 / 时间戳冒充 UUID
    assert "Math.random" not in app_js
    assert "Date.now()" not in app_js

    assert 'id="new-conversation-button"' in index_html


# ========================================================
# T07：数据库迁移
# ========================================================

def _create_legacy_database(path):
    """构造升级前的老库：messages 无 conversation_id、memories 少两列。"""

    connection = sqlite3.connect(str(path))

    try:
        connection.execute(
            """
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                source TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

        connection.executemany(
            "INSERT INTO messages (role, content, created_at) VALUES (?, ?, ?)",
            [
                ("user", "旧消息一", "2026-01-01 10:00:00"),
                ("assistant", "旧消息二", "2026-01-01 10:00:01"),
                ("user", "旧消息三", "2026-01-01 10:00:02"),
            ],
        )
        connection.execute(
            """
            INSERT INTO memories (key, value, source, confidence, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("name", "旧记忆", "user", 1.0, "2026-01-01 10:00:00", "2026-01-01 10:00:00"),
        )

        connection.commit()

    finally:
        connection.close()


def test_T07_migration_keeps_legacy_messages_and_is_idempotent(isolated):
    """旧消息全部保留并归入 legacy；重复迁移不产生重复或破坏。"""

    from memory import LEGACY_CONVERSATION_ID, Memory

    db_path = isolated / "legacy.db"

    _create_legacy_database(db_path)

    memory = Memory(db_path=str(db_path))

    # 1) 迁移只补结构，绝不擅自把历史数据归给某个用户
    connection = sqlite3.connect(str(db_path))

    try:
        owner = connection.execute(
            "SELECT user_id FROM conversations WHERE conversation_id = ?",
            (LEGACY_CONVERSATION_ID,),
        ).fetchone()[0]
        raw_messages = connection.execute(
            "SELECT COUNT(*) FROM messages"
        ).fetchone()[0]
        memory_columns = [
            row[1] for row in connection.execute("PRAGMA table_info(memories)")
        ]
    finally:
        connection.close()

    assert owner is None, "迁移不得擅自为历史数据指定归属"
    assert raw_messages == 3
    assert memory_columns == [
        "id", "key", "value", "source", "confidence",
        "created_at", "updated_at", "category", "reason", "user_id",
    ]

    # 2) 未归属期间：任何用户都读不到（安全默认，不是遗漏）
    assert memory.get_messages(LEGACY_CONVERSATION_ID, user_id="test-user") == ([], 0)
    assert memory.list_conversations(user_id="test-user") == []
    assert memory.get_memories(user_id="test-user") == []

    # 3) 经 claim_legacy_data 归属后，才对所有者可见
    from auth.service import AuthService

    service = AuthService(db_path=str(db_path))
    service.ensure_schema()
    claimed = service.claim_legacy_data("test-user")

    assert claimed["conversations"] == 1
    assert claimed["memories"] == 1

    legacy_messages, legacy_total = memory.get_messages(
        LEGACY_CONVERSATION_ID, user_id="test-user"
    )

    assert legacy_total == 3
    assert [item["content"] for item in legacy_messages] == [
        "旧消息一",
        "旧消息二",
        "旧消息三",
    ]

    # legacy 会话已登记
    legacy_conversation = memory.get_conversation(LEGACY_CONVERSATION_ID, user_id="test-user")

    assert legacy_conversation["conversation_id"] == LEGACY_CONVERSATION_ID
    assert legacy_conversation["title"] == "历史对话"

    # 索引存在
    connection = sqlite3.connect(str(db_path))

    try:
        index_names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            )
        }
        columns = [
            row[1]
            for row in connection.execute("PRAGMA table_info(messages)")
        ]
        memory_columns = [
            row[1]
            for row in connection.execute("PRAGMA table_info(memories)")
        ]
    finally:
        connection.close()

    assert "idx_messages_conversation_id_id" in index_names
    assert "conversation_id" in columns

    # 长期记忆表只补列，内容不变（PAI-AUTH-MVP 追加 user_id）
    assert memory_columns == [
        "id", "key", "value", "source", "confidence",
        "created_at", "updated_at", "category", "reason", "user_id",
    ]

    # 新会话不读取 legacy
    new_id = str(uuid.uuid4())

    memory.save_message("user", "新会话消息", conversation_id=new_id, user_id="test-user")

    assert memory.get_messages(new_id, user_id="test-user")[1] == 1

    new_recent = memory.get_recent_messages(limit=10, conversation_id=new_id, user_id="test-user")

    assert [item["content"] for item in new_recent] == ["新会话消息"]

    assert memory.get_messages(LEGACY_CONVERSATION_ID, user_id="test-user")[1] == 3

    # 重复执行迁移：数量与内容都不变
    Memory(db_path=str(db_path))

    assert memory.get_messages(LEGACY_CONVERSATION_ID, user_id="test-user")[1] == 3
    assert memory.get_messages(new_id, user_id="test-user")[1] == 1

    conversations = memory.list_conversations(user_id="test-user")

    legacy_rows = [
        item for item in conversations
        if item["conversation_id"] == LEGACY_CONVERSATION_ID
    ]

    assert len(legacy_rows) == 1
    assert legacy_rows[0]["message_count"] == 3

    # 旧记忆仍在
    assert len(memory.get_memories(user_id="test-user")) == 1


def test_T07_migration_on_fresh_database(isolated):
    """首次安装：新库直接带 conversation_id 与索引，不需要 ALTER。"""

    from memory import Memory

    db_path = isolated / "fresh.db"

    Memory(db_path=str(db_path))

    connection = sqlite3.connect(str(db_path))

    try:
        columns = [
            row[1]
            for row in connection.execute("PRAGMA table_info(messages)")
        ]
        index_names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            )
        }
        conversations_sql = connection.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'conversations'"
        ).fetchone()[0]
    finally:
        connection.close()

    assert columns == [
        "id", "conversation_id", "role", "content", "created_at"
    ]
    assert "idx_messages_conversation_id_id" in index_names
    assert "conversation_id TEXT PRIMARY KEY" in conversations_sql


# ========================================================
# T08：并发隔离
# ========================================================

def test_T08_concurrent_conversations_do_not_mix(isolated, monkeypatch):
    """两个线程并行跑不同会话：状态独立、历史不混、数据绑定正确。"""

    import agent as agent_module

    calls = _install_adaptive_openai(monkeypatch)

    agent_obj = agent_module.Agent(**DUMMY)

    conv_a = str(uuid.uuid4())
    conv_b = str(uuid.uuid4())

    results = {}
    errors = []

    def _run(name, message, conversation_id):
        try:
            results[name] = agent_obj.run(message, conversation_id, user_id="test-user")
        except Exception as exc:  # pragma: no cover - 失败时用于定位
            errors.append((name, repr(exc)))

    barrier = threading.Barrier(2)
    threads = []

    def _worker(name, message, conversation_id):
        barrier.wait()
        _run(name, message, conversation_id)

    thread_a = threading.Thread(
        target=_worker,
        args=("A", f"并发 A {CONV_A_SECRET_MARKER}", conv_a),
    )
    thread_b = threading.Thread(
        target=_worker,
        args=("B", f"并发 B {CONV_B_SECRET_MARKER}", conv_b),
    )

    threads.extend([thread_a, thread_b])

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join(timeout=30)

    assert not errors, errors
    assert results["A"]["success"] is True
    assert results["B"]["success"] is True

    # 按线程分组检查模型请求
    calls_a = [call for call in calls if call["thread"] == thread_a.ident]
    calls_b = [call for call in calls if call["thread"] == thread_b.ident]

    assert calls_a and calls_b

    texts_a = _system_texts(calls_a)
    texts_b = _system_texts(calls_b)

    assert all(CONV_B_SECRET_MARKER not in text for text in texts_a)
    assert all(CONV_A_SECRET_MARKER not in text for text in texts_b)

    # 数据库绑定正确
    a_messages, a_total = agent_obj.memory.get_messages(conv_a, user_id="test-user")
    b_messages, b_total = agent_obj.memory.get_messages(conv_b, user_id="test-user")

    assert a_total == 2 and b_total == 2
    assert all(
        CONV_B_SECRET_MARKER not in item["content"] for item in a_messages
    )
    assert all(
        CONV_A_SECRET_MARKER not in item["content"] for item in b_messages
    )

    # 两个会话各有一条 user + 一条 assistant
    assert [item["role"] for item in a_messages] == ["user", "assistant"]
    assert [item["role"] for item in b_messages] == ["user", "assistant"]


# ========================================================
# T09：长期记忆兼容
# ========================================================

def test_T09_long_term_memory_is_cross_conversation_but_user_scoped(isolated):
    """长期记忆跨会话可读，但**不跨用户**。

    PAI-AUTH-MVP 之前的语义是「全局可读」；接入用户体系后收窄为
    「同一用户的任意会话可读，其他用户不可读」。这里同时锁定两点：
    1. 它不接收 conversation_id（不受会话边界限制）；
    2. 它必须接收 user_id（受用户边界限制）。
    """
    import inspect

    from memory import Memory

    memory = Memory(db_path=str(isolated / "t09.db"))

    conversation_id = str(uuid.uuid4())

    memory.save_message("user", "本会话消息", conversation_id=conversation_id, user_id="test-user")

    memory.save_memory(
        key="name",
        value="跨会话可见的长期记忆",
        category="identity",
        reason="测试", user_id="test-user"
    )

    signature = inspect.signature(memory.search_long_term_memory)

    assert "conversation_id" not in signature.parameters
    assert "user_id" in signature.parameters
    assert "conversation_id" not in inspect.signature(
        memory.get_memories
    ).parameters
    assert "user_id" in inspect.signature(memory.get_memories).parameters
    assert "conversation_id" not in inspect.signature(
        memory.save_memory
    ).parameters
    assert "user_id" in inspect.signature(memory.save_memory).parameters

    # 无论当前处于哪个会话，同一用户的长期记忆都能被检索到
    hits = memory.search_long_term_memory("长期记忆", limit=5, user_id="test-user")

    assert [item["key"] for item in hits] == ["name"]

    # 消息隔离不影响长期记忆条数
    other_id = str(uuid.uuid4())
    memory.save_message("user", "另一个会话", conversation_id=other_id, user_id="test-user")

    assert len(memory.get_memories(user_id="test-user")) == 1

    # 用户边界：其他用户既检索不到，也列不出来
    assert memory.search_long_term_memory("长期记忆", limit=5, user_id="another-user") == []
    assert memory.get_memories(user_id="another-user") == []

    connection = sqlite3.connect(str(isolated / "t09.db"))

    try:
        columns = [
            row[1]
            for row in connection.execute("PRAGMA table_info(memories)")
        ]
    finally:
        connection.close()

    assert columns == [
        "id", "key", "value", "source", "confidence",
        "created_at", "updated_at", "category", "reason", "user_id",
    ]


# ========================================================
# T10：旧 API 兼容
# ========================================================

def test_T10_legacy_chat_request_still_works(http_env):
    """只发 {"message": ...} 仍然可用，并返回合法 conversation_id。"""

    client, memory, fake_agent = http_env

    response = client.post("/chat", json={"message": "旧调用方式"})

    assert response.status_code == 200

    body = response.json()

    assert body["success"] is True
    assert body["answer"] == "回答"
    assert body["sources"] == []

    returned_id = body["conversation_id"]

    assert isinstance(returned_id, str)

    # 必须是合法 UUID（服务端生成）
    assert str(uuid.UUID(returned_id)) == returned_id

    assert memory.get_conversation(returned_id, user_id="test-user") is not None
    assert fake_agent.calls == [("旧调用方式", returned_id, "test-user")]


def test_T10_explicit_conversation_id_is_reused(http_env):
    """显式传 ID 时必须复用该 ID，而不是另生成一个。"""

    client, memory, fake_agent = http_env

    conversation_id = str(uuid.uuid4())

    response = client.post(
        "/chat",
        json={"message": "指定会话", "conversation_id": conversation_id},
    )

    body = response.json()

    assert response.status_code == 200
    assert body["conversation_id"] == conversation_id
    assert memory.get_conversation(conversation_id, user_id="test-user") is not None
    assert fake_agent.calls[-1] == ("指定会话", conversation_id, "test-user")


def test_T10_invalid_conversation_id_is_rejected(http_env):
    """非法 conversation_id 必须 400，不能静默换成新会话。"""

    client, memory, fake_agent = http_env

    before = len(memory.list_conversations(user_id="test-user"))

    for bad in ["", "   ", "not-a-uuid", "x" * 200, "12345678"]:
        response = client.post(
            "/chat",
            json={"message": "问题", "conversation_id": bad},
        )

        assert response.status_code == 400, bad
        assert response.json()["detail"]["type"] == "invalid_conversation_id"

    assert len(memory.list_conversations(user_id="test-user")) == before
    assert fake_agent.calls == []


def test_T10_memory_default_argument_stays_legacy(isolated):
    """Memory 层旧调用（不传 conversation_id）仍落在 legacy。"""

    from memory import LEGACY_CONVERSATION_ID, Memory

    memory = Memory(db_path=str(isolated / "t10.db"))

    memory.save_message("user", "旧调用", user_id="test-user")

    assert memory.get_messages(LEGACY_CONVERSATION_ID, user_id="test-user")[1] == 1

    legacy_recent = memory.get_recent_messages(limit=10, user_id="test-user")

    assert [item["content"] for item in legacy_recent] == ["旧调用"]
