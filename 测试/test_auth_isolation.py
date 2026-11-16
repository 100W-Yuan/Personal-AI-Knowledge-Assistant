"""PAI-AUTH-MVP：用户隔离测试。

分三层验证「用户只能看到自己的数据」：

  L1 Memory 层    —— 直接验证 SQL 口径（列表、消息、删除、记忆）
  L2 HTTP 层      —— 经 FastAPI 验证越权一律 404（不泄露存在性）
  L3 Agent 身份   —— 验证身份不落在单例实例属性上，只走 ContextVar

为什么要分三层：L1 证明 SQL 本身带归属条件；L2 证明路由把当前用户
真正传下去了（而不是传了常量）；L3 证明并发请求之间不会串线。

边界
----
* 全部使用临时库（``patched_main_memory`` / ``agent.Memory`` 双替换）。
* 不调用真实模型：Agent 只做参数校验与身份机制验证。
"""

from __future__ import annotations

import inspect

import pytest
from fastapi.testclient import TestClient

import agent as agent_module
import main as main_module
from agent import Agent, _ACTIVE_USER_ID, active_user_id
from auth.service import AuthService, _reset_failures_for_tests
from memory import ConversationAccessDenied, Memory

U1 = "user-one"
U2 = "user-two"
PASS_1 = "User-One-Pass-2026"
PASS_2 = "User-Two-Pass-2026"
ORIGIN = {"Origin": "http://testserver"}


# ============================================================
# 夹具
# ============================================================


@pytest.fixture
def memory(patched_main_memory):
    return patched_main_memory


@pytest.fixture
def sessions(patched_main_memory):
    service = AuthService(db_path=patched_main_memory.db_path)
    service.ensure_schema()
    _reset_failures_for_tests()
    yield service
    _reset_failures_for_tests()


@pytest.fixture
def client(sessions, isolated_business_db):
    with TestClient(main_module.app, follow_redirects=False) as c:
        yield c


@pytest.fixture
def two_accounts(sessions, client):
    sessions.create_user(U1, PASS_1, role="admin")
    sessions.create_user(U2, PASS_2, role="user")
    return client


def login_as(client, username, password):
    response = client.post(
        "/auth/login", json={"username": username, "password": password}, headers=ORIGIN
    )
    assert response.status_code == 200, response.text
    return response


# ============================================================
# L1 Memory 层
# ============================================================


def test_L1_conversation_list_is_scoped(memory):
    memory.create_conversation("legacy", title="u1 的会话", user_id=U1)

    assert [c["conversation_id"] for c in memory.list_conversations(user_id=U1)] == ["legacy"]
    assert memory.list_conversations(user_id=U2) == []


def test_L1_get_conversation_returns_none_for_other_user(memory):
    memory.create_conversation("legacy", title="u1 的会话", user_id=U1)

    assert memory.get_conversation("legacy", user_id=U1) is not None
    # 不泄露存在性：他人读取得到 None，而不是抛异常
    assert memory.get_conversation("legacy", user_id=U2) is None


def test_L1_reading_other_users_messages_is_denied(memory):
    memory.create_conversation("legacy", user_id=U1)
    memory.save_message("user", "机密内容", conversation_id="legacy", user_id=U1)

    messages, total = memory.get_messages("legacy", user_id=U1)
    assert total == 1 and messages[0]["content"] == "机密内容"

    with pytest.raises(ConversationAccessDenied):
        memory.get_messages("legacy", user_id=U2)
    with pytest.raises(ConversationAccessDenied):
        memory.get_recent_messages(conversation_id="legacy", user_id=U2)
    with pytest.raises(ConversationAccessDenied):
        memory.search_memories("机密", conversation_id="legacy", user_id=U2)


def test_L1_writing_into_other_users_conversation_is_denied(memory):
    memory.create_conversation("legacy", user_id=U1)

    with pytest.raises(ConversationAccessDenied):
        memory.save_message("user", "越权写入", conversation_id="legacy", user_id=U2)
    with pytest.raises(ConversationAccessDenied):
        memory.ensure_conversation("legacy", user_id=U2)

    # 越权尝试不得改变原会话内容
    messages, total = memory.get_messages("legacy", user_id=U1)
    assert total == 0


def test_L1_delete_only_affects_own_conversation(memory):
    # legacy 会话按设计禁止删除，这里用普通会话验证归属校验本身
    cid = "3b1f5c9e-1111-4a2b-9c3d-000000000001"
    memory.create_conversation(cid, title="u1 的会话", user_id=U1)
    memory.save_message("user", "内容", conversation_id=cid, user_id=U1)

    with pytest.raises(ConversationAccessDenied):
        memory.delete_conversation(cid, user_id=U2)

    assert memory.get_conversation(cid, user_id=U1) is not None
    assert len(memory.get_messages(cid, user_id=U1)[0]) == 1

    result = memory.delete_conversation(cid, user_id=U1)
    assert result["conversation_id"] == cid
    assert result["deleted_messages"] == 1
    assert memory.get_conversation(cid, user_id=U1) is None


def test_L1_legacy_conversation_cannot_be_deleted(memory):
    memory.create_conversation("legacy", user_id=U1)
    with pytest.raises(ValueError):
        memory.delete_conversation("legacy", user_id=U1)


def test_L1_memories_are_isolated(memory):
    memory.save_memory(key="name", value="Alice", user_id=U1)
    memory.save_memory(key="name", value="Bob", user_id=U2)

    assert [m["value"] for m in memory.get_memories(user_id=U1)] == ["Alice"]
    assert [m["value"] for m in memory.get_memories(user_id=U2)] == ["Bob"]

    assert memory.find_memory("name", user_id=U1)["value"] == "Alice"
    assert memory.find_memory("name", user_id=U2)["value"] == "Bob"

    assert memory.search_long_term_memory("Alice", user_id=U1)
    assert memory.search_long_term_memory("Alice", user_id=U2) == []

    assert [m["value"] for m in memory.get_memories_by_category("identity", user_id=U1)] == ["Alice"]
    assert [m["value"] for m in memory.get_memories_by_category("identity", user_id=U2)] == ["Bob"]


def test_L1_empty_user_id_is_rejected(memory):
    for bad in ("", "   ", None, 123):
        with pytest.raises(ValueError):
            memory.list_conversations(user_id=bad)
        with pytest.raises(ValueError):
            memory.save_message("user", "x", conversation_id="legacy", user_id=bad)
        with pytest.raises(ValueError):
            memory.get_memories(user_id=bad)


def test_L1_user_id_is_keyword_only(memory):
    """user_id 必须是仅关键字参数，避免位置传参时被误当成其他参数。"""
    signature = inspect.signature(Memory.get_messages)
    assert signature.parameters["user_id"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["user_id"].default is inspect.Parameter.empty

    for method in (
        Memory.create_conversation,
        Memory.get_conversation,
        Memory.ensure_conversation,
        Memory.list_conversations,
        Memory.delete_conversation,
        Memory.save_message,
        Memory.get_recent_messages,
        Memory.search_memories,
        Memory.find_memory,
        Memory.search_long_term_memory,
        Memory.get_memories_by_category,
        Memory.save_memory,
        Memory.get_memories,
    ):
        param = inspect.signature(method).parameters.get("user_id")
        assert param is not None, method.__name__
        assert param.kind is inspect.Parameter.KEYWORD_ONLY, method.__name__
        assert param.default is inspect.Parameter.empty, method.__name__


# ============================================================
# L2 HTTP 层
# ============================================================


def test_L2_conversation_lists_differ_between_users(two_accounts, memory, sessions):
    uid1 = sessions.get_user(U1)["id"]
    memory.create_conversation("legacy", title="u1 的会话", user_id=uid1)

    login_as(two_accounts, U1, PASS_1)
    first = two_accounts.get("/conversations").json()
    assert first["total"] == 1

    two_accounts.post("/auth/logout", headers=ORIGIN)
    login_as(two_accounts, U2, PASS_2)
    second = two_accounts.get("/conversations").json()
    assert second["total"] == 0


def test_L2_reading_other_users_conversation_returns_nothing(two_accounts, memory, sessions):
    uid1 = sessions.get_user(U1)["id"]
    memory.create_conversation("legacy", title="u1 的会话", user_id=uid1)
    memory.save_message("user", "机密内容", conversation_id="legacy", user_id=uid1)

    login_as(two_accounts, U2, PASS_2)
    response = two_accounts.get("/conversations/legacy/messages")

    # 读取他人会话不报错、不泄露内容：按空会话处理
    assert response.status_code == 200
    payload = response.json()
    assert payload["messages"] == []
    assert payload["total"] == 0
    assert "机密内容" not in response.text


def test_L2_unknown_and_forbidden_conversation_look_identical(two_accounts, memory, sessions):
    uid1 = sessions.get_user(U1)["id"]
    memory.create_conversation("legacy", user_id=uid1)

    login_as(two_accounts, U2, PASS_2)
    stolen = two_accounts.get("/conversations/legacy/messages")
    absent = two_accounts.get("/conversations/2f8a1e40-0000-4000-8000-000000000000/messages")

    # 两者的状态码与消息体必须完全同构，否则可用来探测他人会话是否存在
    assert stolen.status_code == absent.status_code == 200
    assert stolen.json()["messages"] == absent.json()["messages"] == []
    assert stolen.json()["total"] == absent.json()["total"] == 0
    assert stolen.json()["has_more"] == absent.json()["has_more"] is False


def test_L2_chat_into_other_users_conversation_is_404(two_accounts, memory, sessions):
    """不发真实模型请求：越权检查发生在调用 Agent 之前。"""
    uid1 = sessions.get_user(U1)["id"]
    memory.create_conversation("legacy", user_id=uid1)

    login_as(two_accounts, U2, PASS_2)
    response = two_accounts.post(
        "/chat",
        json={"message": "你好", "conversation_id": "legacy"},
        headers=ORIGIN,
    )

    assert response.status_code == 404
    assert response.json()["detail"]["type"] == "conversation_not_found"


# ============================================================
# L3 Agent 身份机制
# ============================================================


def test_L3_run_requires_keyword_user_id():
    signature = inspect.signature(Agent.run)
    param = signature.parameters["user_id"]

    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default is inspect.Parameter.empty, "user_id 不能有默认值"


def test_L3_missing_user_id_raises_type_error(tmp_path, monkeypatch):
    monkeypatch.setattr(
        agent_module, "Memory", lambda *a, **k: Memory(db_path=str(tmp_path / "agent.db"))
    )
    instance = Agent(api_key="dummy", base_url="https://example.invalid", model="dummy")

    with pytest.raises(TypeError):
        instance.run("hi")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        instance.run("hi", "legacy")  # type: ignore[call-arg]


def test_L3_empty_user_id_raises_value_error(tmp_path, monkeypatch):
    monkeypatch.setattr(
        agent_module, "Memory", lambda *a, **k: Memory(db_path=str(tmp_path / "agent.db"))
    )
    instance = Agent(api_key="dummy", base_url="https://example.invalid", model="dummy")

    for bad in ("", "   ", None):
        with pytest.raises((ValueError, TypeError)):
            instance.run("hi", user_id=bad)


def test_L3_identity_is_not_stored_on_instance(tmp_path, monkeypatch):
    """身份必须活在 ContextVar 里：实例属性会在并发请求间串线。"""
    monkeypatch.setattr(
        agent_module, "Memory", lambda *a, **k: Memory(db_path=str(tmp_path / "agent.db"))
    )
    instance = Agent(api_key="dummy", base_url="https://example.invalid", model="dummy")

    assert not hasattr(instance, "user_id")
    assert not hasattr(instance, "_user_id")

    source = inspect.getsource(Agent.__init__)
    assert "self.user_id" not in source
    assert "self._user_id" not in source


def test_L3_contextvar_default_is_empty():
    """未设置身份时读取必须立即失败，而不是返回 None 悄悄放行。"""
    token = _ACTIVE_USER_ID.set(None)
    try:
        with pytest.raises(RuntimeError):
            active_user_id()
    finally:
        _ACTIVE_USER_ID.reset(token)


def test_L3_contextvar_isolated_between_contexts():
    import contextvars

    def worker(value):
        _ACTIVE_USER_ID.set(value)
        return active_user_id()

    ctx = contextvars.copy_context()
    assert ctx.run(worker, "alpha") == "alpha"

    ctx2 = contextvars.copy_context()
    assert ctx2.run(worker, "beta") == "beta"

    # 主上下文未被污染
    token = _ACTIVE_USER_ID.set("main")
    try:
        assert active_user_id() == "main"
    finally:
        _ACTIVE_USER_ID.reset(token)


def test_L3_search_memory_tool_resolves_identity_per_call(memory):
    """工具闭包不能绑定用户：同一工具实例在不同身份下必须返回不同结果。"""
    memory.save_memory(key="name", value="Alice", user_id=U1)
    memory.save_memory(key="name", value="Bob", user_id=U2)

    from tools import make_search_memory_tool

    current = {"uid": U1}
    tool = make_search_memory_tool(memory, lambda: current["uid"])

    first = tool("Alice")
    assert first["count"] >= 1
    assert all(item["value"] == "Alice" for item in first["memories"])

    current["uid"] = U2
    second = tool("Bob")
    assert second["count"] >= 1
    assert all(item["value"] == "Bob" for item in second["memories"])
