"""PAI-WORKBENCH-R1-1-FIX-01：首次改密强制访问控制 测试。

覆盖任务书第五节 26 项要求。编号对应关系：
  G01-G07  普通用户：正常访问 + 强制改密用户被业务 API 拦截（403）
  G08-G09  页面：强制改密用户访问受保护页面 302 到 /change-password
  G10-G12  白名单：/auth/me、/auth/logout、/auth/change-password 可用
  G13      改密成功后恢复访问（含 /auth/me 返回 false）
  G14      改密失败后仍被限制
  G15-G18  管理员：被重置密码后同规则拦截，改密成功后恢复管理员权限
  G19-G20  兼容：未登录仍 401、普通用户访问管理员接口仍 403
  G21-G22  拦截响应：正确错误码 + Cache-Control: no-store
  G23      拦截发生在业务逻辑执行之前（Agent/内存/知识库/视觉客户端零调用）
  G24      白名单与静态资源不被误拦截
  G25      ROUTE_POLICY 已登记新增路由（T18/T18b 做完整性校验）
  G26      旧用户 must_change_password=0 行为不变

边界
----
* 全部使用临时库（``patched_main_memory`` + ``isolated_business_db``），
  不触碰 memory.db / data/business.db。
* 不调用真实模型：/chat 用 Fake Agent；/analyze-image 守卫在视觉客户端
  构造之前拦截；/knowledge/search 守卫在知识库初始化之前拦截。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import main as main_module
from auth.service import AuthService, _reset_failures_for_tests

ORIGIN = {"Origin": "http://testserver"}

ADMIN_NAME = "root_admin"
ADMIN_PASS = "Admin-Pass-2026"
HR_NAME = "hr_user"
HR_PASS = "Hr-Pass-2026-ok"
FORCED_NAME = "forced_user"
FORCED_PASS = "Temp-Pass-2026"
FORCED_NEW = "New-Pass-2026-ok"
ADMIN2_NAME = "second_admin"
ADMIN2_PASS = "Second-Pass-2026"
TEMP_PASS = "Reset-Temp-2026"


class _FakeAgent:
    """离线替身 Agent：仅证明 /chat 走到了业务层。"""

    def __init__(self):
        self.calls = 0

    def run(self, message, conversation_id, user_id=None):
        self.calls += 1
        return {"success": True, "answer": "ok"}


def _assert_change_required(response):
    """断言拦截语义：403 + password_change_required + no-store。"""
    assert response.status_code == 403, response.text
    body = response.json()
    assert body["detail"]["type"] == "password_change_required"
    assert response.headers.get("cache-control") == "no-store"


@pytest.fixture
def svc(patched_main_memory):
    service = AuthService(db_path=patched_main_memory.db_path)
    service.ensure_schema()
    _reset_failures_for_tests()
    yield service
    _reset_failures_for_tests()


@pytest.fixture
def client(svc, isolated_business_db):
    svc.create_user(ADMIN_NAME, ADMIN_PASS, role="admin")
    svc.create_user(HR_NAME, HR_PASS, role="user")
    with TestClient(main_module.app, follow_redirects=False) as c:
        yield c


@pytest.fixture
def forced_client(client, svc):
    """一个 must_change_password=1 的普通用户已登录的客户端。"""
    svc.create_user(FORCED_NAME, FORCED_PASS, role="user", must_change_password=True)
    response = client.post(
        "/auth/login",
        json={"username": FORCED_NAME, "password": FORCED_PASS},
        headers=ORIGIN,
    )
    assert response.status_code == 200
    return client


def login(client, username=ADMIN_NAME, password=ADMIN_PASS):
    return client.post(
        "/auth/login",
        json={"username": username, "password": password},
        headers=ORIGIN,
    )


# ============================================================
# G01-G07 普通用户：正常访问 / 强制改密拦截
# ============================================================


def test_G01_normal_user_can_chat(client, monkeypatch):
    """未强制改密用户可以正常访问 /chat（Fake Agent，不触模型）。"""
    agent = _FakeAgent()
    monkeypatch.setattr(main_module, "get_agent", lambda: agent)
    login(client, HR_NAME, HR_PASS)

    response = client.post("/chat", json={"message": "你好"}, headers=ORIGIN)
    assert response.status_code == 200, response.text
    assert response.json()["success"] is True
    assert agent.calls == 1


def test_G02_forced_user_chat_403(client, svc, monkeypatch):
    """强制改密用户访问 /chat → 403，且 Agent 零调用。"""
    svc.create_user(FORCED_NAME, FORCED_PASS, role="user", must_change_password=True)
    agent = _FakeAgent()
    monkeypatch.setattr(main_module, "get_agent", lambda: agent)
    login_response = login(client, FORCED_NAME, FORCED_PASS)
    assert login_response.status_code == 200, login_response.text

    response = client.post("/chat", json={"message": "你好"}, headers=ORIGIN)
    _assert_change_required(response)
    assert agent.calls == 0


def test_G03_forced_user_conversations_403(forced_client):
    _assert_change_required(forced_client.get("/conversations"))
    _assert_change_required(
        forced_client.get("/conversations/00000000-0000-0000-0000-000000000000/messages")
    )


def test_G04_forced_user_analyze_image_403(forced_client, monkeypatch):
    """强制改密用户上传图片 → 403，视觉客户端不构造、不消耗额度。"""

    def _boom():
        raise AssertionError("get_vision_client 不应被调用")

    monkeypatch.setattr(main_module, "get_vision_client", _boom)
    response = forced_client.post(
        "/analyze-image",
        files={"file": ("a.png", b"not-a-real-image", "image/png")},
        data={"question": "测试"},
        headers=ORIGIN,
    )
    _assert_change_required(response)


def test_G05_forced_user_business_403(forced_client):
    """router 级守卫：/business/* 读与写全部 403。"""
    _assert_change_required(forced_client.get("/business/orders"))
    _assert_change_required(
        forced_client.post("/business/orders", json={}, headers=ORIGIN)
    )


def test_G06_forced_user_knowledge_search_403(forced_client, monkeypatch):
    """强制改密用户检索知识库 → 403，且不初始化知识库/Embedding。"""

    def _boom():
        raise AssertionError("get_knowledge_base 不应被调用")

    monkeypatch.setattr(main_module, "get_knowledge_base", _boom)
    _assert_change_required(forced_client.get("/knowledge/search?query=hello"))


def test_G07_forced_user_knowledge_documents_403(forced_client):
    _assert_change_required(forced_client.get("/knowledge/documents"))


# ============================================================
# G08-G09 页面跳转
# ============================================================


def test_G08_forced_user_workbench_redirects(forced_client):
    response = forced_client.get("/workbench")
    assert response.status_code == 302
    assert response.headers["location"] == "/change-password"


def test_G09_forced_user_business_ui_and_index_redirects(forced_client):
    for path in ("/business-ui", "/"):
        response = forced_client.get(path)
        assert response.status_code == 302, path
        assert response.headers["location"] == "/change-password"


# ============================================================
# G10-G12 白名单接口
# ============================================================


def test_G10_forced_user_me_ok(forced_client):
    response = forced_client.get("/auth/me")
    assert response.status_code == 200
    assert response.json()["user"]["must_change_password"] is True


def test_G11_forced_user_logout_ok(forced_client):
    response = forced_client.post("/auth/logout", headers=ORIGIN)
    assert response.status_code == 200
    assert response.json()["ok"] is True
    # 登出后会话失效
    assert forced_client.get("/auth/me").status_code == 401


def test_G12_forced_user_change_password_ok(forced_client):
    response = forced_client.post(
        "/auth/change-password",
        json={"current_password": FORCED_PASS, "new_password": FORCED_NEW},
        headers=ORIGIN,
    )
    assert response.status_code == 200, response.text
    assert response.json()["must_change_password"] is False


# ============================================================
# G13-G14 改密成功 / 失败
# ============================================================


def test_G13_after_change_access_restored(client, svc):
    """改密成功：Session 保留、业务 API 与页面恢复正常、/auth/me 为 false。"""
    svc.create_user(FORCED_NAME, FORCED_PASS, role="user", must_change_password=True)
    login(client, FORCED_NAME, FORCED_PASS)

    _assert_change_required(client.get("/conversations"))
    assert client.get("/workbench").status_code == 302

    response = client.post(
        "/auth/change-password",
        json={"current_password": FORCED_PASS, "new_password": FORCED_NEW},
        headers=ORIGIN,
    )
    assert response.status_code == 200

    me = client.get("/auth/me")
    assert me.status_code == 200
    assert me.json()["user"]["must_change_password"] is False

    # 业务 API 恢复（当前会话被保留）
    assert client.get("/conversations").status_code == 200
    assert client.get("/business/orders").status_code == 200
    assert client.get("/knowledge/documents").status_code == 200
    # 页面恢复
    assert client.get("/workbench").status_code == 200


def test_G14_failed_change_still_restricted(client, svc):
    """改密失败（旧密码错误）后仍被限制，标记不清零。"""
    svc.create_user(FORCED_NAME, FORCED_PASS, role="user", must_change_password=True)
    login(client, FORCED_NAME, FORCED_PASS)

    response = client.post(
        "/auth/change-password",
        json={"current_password": "Wrong-Pass-000", "new_password": FORCED_NEW},
        headers=ORIGIN,
    )
    assert response.status_code == 401
    assert response.json()["error"] == "invalid_credentials"

    assert client.get("/auth/me").json()["user"]["must_change_password"] is True
    _assert_change_required(client.get("/conversations"))
    _assert_change_required(client.post("/chat", json={"message": "x"}, headers=ORIGIN))


# ============================================================
# G15-G18 管理员场景
# ============================================================


@pytest.fixture
def reset_admin_client(client, svc):
    """管理员被重置密码（must_change=1）后重新登录的客户端。"""
    svc.create_user(ADMIN2_NAME, ADMIN2_PASS, role="admin")
    admin = svc.get_user(ADMIN_NAME)
    svc.reset_password(
        ADMIN2_NAME, TEMP_PASS,
        actor_user_id=admin["id"], actor_type="http_admin",
    )
    response = login(client, ADMIN2_NAME, TEMP_PASS)
    assert response.status_code == 200
    return client


def test_G15_reset_admin_cannot_admin_api(reset_admin_client):
    response = reset_admin_client.get("/admin/users")
    _assert_change_required(response)
    _assert_change_required(
        reset_admin_client.post("/admin/users", json={"username": "x"}, headers=ORIGIN)
    )


def test_G16_reset_admin_cannot_knowledge_write(reset_admin_client):
    """管理员被重置后不能写知识库（role=admin 不绕过强制改密）。"""
    files = {"file": ("doc.md", b"# hi", "text/markdown")}
    _assert_change_required(
        reset_admin_client.post("/knowledge/import", files=files, headers=ORIGIN)
    )
    _assert_change_required(
        reset_admin_client.post(
            "/knowledge/documents", files=files, headers=ORIGIN
        )
    )
    _assert_change_required(
        reset_admin_client.delete(
            "/knowledge/documents/no-such-id", headers=ORIGIN
        )
    )


def test_G17_reset_admin_can_change_password(reset_admin_client):
    assert reset_admin_client.get("/auth/me").json()["user"]["must_change_password"] is True
    response = reset_admin_client.post(
        "/auth/change-password",
        json={"current_password": TEMP_PASS, "new_password": FORCED_NEW},
        headers=ORIGIN,
    )
    assert response.status_code == 200


def test_G18_admin_restored_after_change(reset_admin_client):
    reset_admin_client.post(
        "/auth/change-password",
        json={"current_password": TEMP_PASS, "new_password": FORCED_NEW},
        headers=ORIGIN,
    )
    response = reset_admin_client.get("/admin/users")
    assert response.status_code == 200
    usernames = {item["username"] for item in response.json()["users"]}
    assert ADMIN2_NAME in usernames


# ============================================================
# G19-G22 兼容语义 / 响应契约
# ============================================================


def test_G19_anonymous_still_401(client):
    for method, path in (
        ("post", "/chat"),
        ("get", "/conversations"),
        ("get", "/business/orders"),
        ("get", "/admin/users"),
        ("get", "/knowledge/search?query=x"),
    ):
        response = getattr(client, method)(path)
        assert response.status_code == 401, path


def test_G20_regular_user_admin_api_still_403(client):
    """must_change=0 的普通用户访问管理员接口仍是角色 403，不是改密 403。"""
    login(client, HR_NAME, HR_PASS)
    response = client.get("/admin/users")
    assert response.status_code == 403
    assert response.json()["detail"]["type"] == "forbidden"


def test_G21_error_code_on_every_guarded_api(forced_client):
    """普通用户场景下所有 USER 级业务 API 拦截错误码一致。

    注意：/admin/* 不在此列——普通用户访问管理员接口先命中角色 403
    （forbidden，声明顺序在前），管理员被强制改密的拦截由 G15 覆盖。
    """
    probes = (
        ("post", "/chat", {"json": {"message": "x"}, "headers": ORIGIN}),
        ("get", "/conversations", {}),
        ("get", "/knowledge/search?query=x", {}),
        ("get", "/knowledge/documents", {}),
        ("get", "/business/orders", {}),
    )
    for method, path, kwargs in probes:
        response = getattr(forced_client, method)(path, **kwargs)
        assert response.status_code == 403, path
        assert response.json()["detail"]["type"] == "password_change_required", path


def test_G22_intercepted_responses_no_store(forced_client):
    """password_change_required 拦截响应必须带 Cache-Control: no-store。

    只探测本守卫返回的响应；require_admin 的角色 403 无缓存头属于
    修复前既有行为，不在本次范围内。
    """
    for path in ("/conversations", "/business/orders", "/knowledge/documents"):
        response = forced_client.get(path)
        assert response.headers.get("cache-control") == "no-store", path


# ============================================================
# G23-G26 拦截时机 / 白名单 / ROUTE_POLICY / 旧用户
# ============================================================


def test_G23_interception_before_any_business_logic(client, svc, monkeypatch):
    """拦截发生在业务逻辑之前：Agent / 知识库 / 视觉客户端零调用。

    三个桩只被业务处理器调用：若守卫失效、处理器执行到桩，会得到 500 /
    AssertionError 而不是 403 password_change_required。403 即证明拦截
    发生在依赖解析阶段，先于任何业务逻辑。
    注：main.get_memory 还被 deps._service 的库路径解析合法复用，不可打桩。
    """
    svc.create_user(FORCED_NAME, FORCED_PASS, role="user", must_change_password=True)
    login(client, FORCED_NAME, FORCED_PASS)

    def _boom(name):
        def _raise():
            raise AssertionError("%s 不应被调用" % name)
        return _raise

    monkeypatch.setattr(main_module, "get_agent", _boom("get_agent"))
    monkeypatch.setattr(main_module, "get_knowledge_base", _boom("get_knowledge_base"))
    monkeypatch.setattr(main_module, "get_vision_client", _boom("get_vision_client"))

    r_chat = client.post("/chat", json={"message": "你好"}, headers=ORIGIN)
    _assert_change_required(r_chat)

    r_kb = client.get("/knowledge/search?query=hello")
    _assert_change_required(r_kb)

    r_img = client.post(
        "/analyze-image",
        files={"file": ("a.png", b"x", "image/png")},
        headers=ORIGIN,
    )
    _assert_change_required(r_img)


def test_G24_whitelist_and_statics_not_blocked(forced_client):
    """白名单接口与改密页静态资源不被误拦截。"""
    assert forced_client.get("/auth/me").status_code == 200
    assert forced_client.post("/auth/logout", headers=ORIGIN).status_code == 200

    # 登出后重登，验证改密页面本体与静态资源
    login(forced_client, FORCED_NAME, FORCED_PASS)
    page = forced_client.get("/change-password")
    assert page.status_code == 200
    assert "修改密码" in page.text

    assert forced_client.get("/auth/static/change_password.js").status_code == 200
    assert forced_client.get("/auth/static/style.css").status_code == 200
    # 既有匿名静态资源不受影响
    assert forced_client.get("/workbench/static/style.css").status_code == 200
    assert forced_client.get("/app-ui/static/app.js").status_code == 200


def test_G25_route_policy_contains_new_routes():
    """新增路由已在 ROUTE_POLICY 登记（完整性由 test_auth_mvp T18/T18b 校验）。"""
    from test_auth_mvp import ROUTE_POLICY

    assert ("GET", "/change-password") in ROUTE_POLICY
    assert ("GET", "/auth/static/change_password.js") in ROUTE_POLICY
    assert ROUTE_POLICY[("GET", "/change-password")] == "page"


def test_G26_legacy_users_unaffected(client):
    """旧用户（must_change_password=0）的行为与修复前一致。"""
    login(client, HR_NAME, HR_PASS)

    me = client.get("/auth/me")
    assert me.json()["user"]["must_change_password"] is False

    assert client.get("/conversations").status_code == 200
    assert client.get("/business/orders").status_code == 200
    assert client.get("/knowledge/documents").status_code == 200
    # 页面直接 200，不再被重定向
    assert client.get("/workbench").status_code == 200

    # 匿名语义保持：改密页未登录 → 302 /login
    client.cookies.clear()
    response = client.get("/change-password")
    assert response.status_code == 302
    assert response.headers["location"].startswith("/login")
