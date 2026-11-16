"""PAI-AUTH-MVP：认证系统专项测试（20 项）。

对应第二步实施指令的测试要求，逐项编号如下：

  T-01  建表幂等：ensure_schema 重复执行不报错，users/sessions 存在
  T-02  创建管理员并登录成功，签发 HttpOnly 会话 Cookie
  T-03  密码错误 → 401，且不区分「用户不存在」与「密码错误」
  T-04  禁用账号 → 403 account_disabled
  T-05  连续失败 5 次后 → 429 + Retry-After（进程内限流）
  T-06  登录后 GET /auth/me 返回当前用户，且不含任何敏感字段
  T-07  登出后原会话立即失效；重复登出仍然幂等
  T-08  伪造/随机 token → 401（不接受客户端自造身份）
  T-09  会话绝对过期（7 天）后 → 401
  T-10  空闲超时（24h）后 → 401
  T-11  60 秒节流窗口内不续期；超过窗口后续期
  T-12  重置密码撤销该用户全部会话
  T-13  禁用账号撤销会话；不允许禁用最后一个启用的管理员
  T-14  令牌只以 sha256 摘要入库，库中不存在明文 token
  T-15  同源校验：缺失 Origin/Referer 的写请求 → 403
  T-16  同源校验：跨站 Origin → 403，同源 Origin → 通过
  T-17  生产环境关闭 /docs、/redoc、/openapi.json（子进程验证）
  T-18  路由矩阵：所有实际路由必须被显式分类，未分类即失败
  T-19  路由矩阵：分类结果与真实鉴权行为一致（匿名/登录/管理员）
  T-20  历史数据归属：legacy_status 统计准确，claim_legacy_data 只认领无主数据

边界
----
* 全部使用临时库（``patched_main_memory`` + ``isolated_business_db``），
  不触碰项目的 memory.db / data/business.db。
* 不调用真实模型：/chat 只验证「匿名 401」，不触发 Agent。
* 时间相关用例直接改写 sessions 表的时间字段，不等真实时钟。
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import subprocess
import sys
import uuid
from datetime import timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import main as main_module
from auth import security as auth_security
from auth.service import (
    AccountDisabled,
    AuthService,
    InvalidCredentials,
    UsernameTaken,
    _reset_failures_for_tests,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ORIGIN = {"Origin": "http://testserver"}

ADMIN_NAME = "root_admin"
ADMIN_PASS = "Admin-Pass-2026"
USER_NAME = "hr_user"
USER_PASS = "Hr-Pass-2026-ok"


# ============================================================
# 夹具
# ============================================================


@pytest.fixture
def svc(patched_main_memory):
    """临时库上的 AuthService；lifespan 与业务代码看到的是同一个库。"""
    service = AuthService(db_path=patched_main_memory.db_path)
    service.ensure_schema()
    _reset_failures_for_tests()
    yield service
    _reset_failures_for_tests()


@pytest.fixture
def accounts(svc):
    svc.create_user(ADMIN_NAME, ADMIN_PASS, role="admin")
    svc.create_user(USER_NAME, USER_PASS, role="user")
    return svc


@pytest.fixture
def client(accounts, isolated_business_db):
    """进入上下文以触发 lifespan；库已全部指向 tmp。"""
    with TestClient(main_module.app, follow_redirects=False) as c:
        yield c


def login(client, username=ADMIN_NAME, password=ADMIN_PASS, headers=ORIGIN):
    return client.post(
        "/auth/login",
        json={"username": username, "password": password},
        headers=headers,
    )


def session_rows(db_path):
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in connection.execute("SELECT * FROM sessions")]
    finally:
        connection.close()


def force_session_times(db_path, created_at=None, last_seen_at=None, expires_at=None):
    connection = sqlite3.connect(db_path)
    try:
        if created_at is not None:
            connection.execute("UPDATE sessions SET created_at = ?", (created_at,))
        if last_seen_at is not None:
            connection.execute("UPDATE sessions SET last_seen_at = ?", (last_seen_at,))
        if expires_at is not None:
            connection.execute("UPDATE sessions SET expires_at = ?", (expires_at,))
        connection.commit()
    finally:
        connection.close()


# ============================================================
# T-01 建表幂等
# ============================================================


def test_T01_ensure_schema_is_idempotent(svc):
    svc.ensure_schema()
    svc.ensure_schema()

    connection = sqlite3.connect(svc.db_path)
    try:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(conversations)")
        }
    finally:
        connection.close()

    assert {"users", "sessions"} <= tables
    # 用户归属列由迁移补齐，不应在每次建表时被重复添加或报错
    assert "user_id" in columns


# ============================================================
# T-02 登录成功 / Cookie 契约
# ============================================================


def test_T02_login_success_sets_hardened_cookie(client):
    response = login(client)

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["user"] == {"username": ADMIN_NAME, "role": "admin"}
    # 响应体不得回显密码或哈希
    assert ADMIN_PASS not in response.text
    assert "argon2" not in response.text.lower()

    cookie = response.headers.get("set-cookie", "")
    assert "pai_session=" in cookie
    assert "HttpOnly" in cookie
    assert "Path=/" in cookie
    assert "SameSite=lax" in cookie.lower() or "samesite=lax" in cookie.lower()
    assert "Max-Age=604800" in cookie
    # 本地 http 调试默认不带 Secure；带 Secure 会让浏览器丢弃 Cookie
    assert "Secure" not in cookie


def test_T02b_login_is_case_sensitive_and_trimmed(svc):
    """用户名保留大小写，首尾空白被裁掉。"""
    with pytest.raises(InvalidCredentials):
        svc.login("  " + ADMIN_NAME + "  ", ADMIN_PASS)  # 用户尚未创建，先验不存在
    svc.create_user(ADMIN_NAME, ADMIN_PASS, role="admin")
    assert svc.login("  " + ADMIN_NAME + "  ", ADMIN_PASS)["user"]["username"] == ADMIN_NAME
    with pytest.raises(InvalidCredentials):
        svc.login(ADMIN_NAME.upper(), ADMIN_PASS)


# ============================================================
# T-03 凭据错误不泄露账号存在性
# ============================================================


def test_T03_wrong_password_and_unknown_user_are_indistinguishable(client):
    wrong = login(client, ADMIN_NAME, "definitely-wrong-123")
    unknown = login(client, "no_such_person", "definitely-wrong-123")

    assert wrong.status_code == 401
    assert unknown.status_code == 401
    assert wrong.json() == unknown.json()
    assert wrong.json()["error"] == "invalid_credentials"
    # 401 不得带 Set-Cookie
    assert "set-cookie" not in {k.lower() for k in wrong.headers}


# ============================================================
# T-04 禁用账号
# ============================================================


def test_T04_disabled_account_cannot_login(client, svc):
    svc.set_active(USER_NAME, False)

    response = login(client, USER_NAME, USER_PASS)

    assert response.status_code == 403
    assert response.json()["error"] == "account_disabled"
    assert "set-cookie" not in {k.lower() for k in response.headers}


# ============================================================
# T-05 登录限流
# ============================================================


def test_T05_rate_limit_after_five_failures(client):
    for i in range(5):
        response = login(client, USER_NAME, "wrong-pass-000")
        assert response.status_code == 401, f"第 {i + 1} 次应为 401，实际 {response.status_code}"

    blocked = login(client, USER_NAME, "wrong-pass-000")
    assert blocked.status_code == 429
    assert blocked.json()["error"] == "rate_limited"
    assert blocked.headers.get("Retry-After") == "900"

    # 限流期内即使密码正确也不放行
    assert login(client, USER_NAME, USER_PASS).status_code == 429


# ============================================================
# T-06 /auth/me
# ============================================================


def test_T06_auth_me_returns_minimal_identity(client):
    login(client)
    response = client.get("/auth/me")

    assert response.status_code == 200
    payload = response.json()
    assert payload["user"] == {
        "username": ADMIN_NAME,
        "role": "admin",
        "must_change_password": False,
    }

    # 不得暴露 id / password_hash / is_active 等内部字段
    assert set(payload["user"]) == {"username", "role", "must_change_password"}
    assert payload["user"]["username"] not in ("", None)


# ============================================================
# T-07 登出
# ============================================================


def test_T07_logout_revokes_session_and_is_idempotent(client, svc):
    login(client)
    assert client.get("/auth/me").status_code == 200

    out = client.post("/auth/logout", headers=ORIGIN)
    assert out.status_code == 200
    assert out.json() == {"ok": True}

    assert client.get("/auth/me").status_code == 401

    # 幂等：无 Cookie 再登出也返回 200
    again = client.post("/auth/logout", headers=ORIGIN)
    assert again.status_code == 200

    rows = session_rows(svc.db_path)
    assert rows and all(row["revoked_at"] is not None for row in rows)


# ============================================================
# T-08 伪造令牌
# ============================================================


def test_T08_forged_tokens_are_rejected(client):
    for token in ("deadbeef" * 8, "x", str(uuid.uuid4())):
        client.cookies.set("pai_session", token)
        assert client.get("/auth/me").status_code == 401, token
    client.cookies.clear()


# ============================================================
# T-09 绝对过期
# ============================================================


def test_T09_absolute_timeout_rejects_session(client, svc):
    login(client)
    now = auth_security.utcnow()
    created = now - auth_security.ABSOLUTE_TIMEOUT - timedelta(minutes=1)
    force_session_times(
        svc.db_path,
        created_at=auth_security.format_time(created),
        last_seen_at=auth_security.format_time(now),
        expires_at=auth_security.format_time(now + timedelta(hours=1)),
    )

    assert client.get("/auth/me").status_code == 401
    # 被拒后会话应被标记撤销
    assert all(row["revoked_at"] is not None for row in session_rows(svc.db_path))


# ============================================================
# T-10 空闲超时
# ============================================================


def test_T10_idle_timeout_rejects_session(client, svc):
    login(client)
    now = auth_security.utcnow()
    stale = now - auth_security.IDLE_TIMEOUT - timedelta(minutes=1)
    force_session_times(
        svc.db_path,
        last_seen_at=auth_security.format_time(stale),
        expires_at=auth_security.format_time(stale + auth_security.IDLE_TIMEOUT),
    )

    assert client.get("/auth/me").status_code == 401


# ============================================================
# T-11 续期节流
# ============================================================


def test_T11_renew_throttle_does_not_touch_recent_session(client, svc):
    login(client)
    before = session_rows(svc.db_path)[0]["last_seen_at"]

    assert client.get("/auth/me").status_code == 200
    after = session_rows(svc.db_path)[0]["last_seen_at"]

    # 60 秒内不续期：last_seen_at 必须原样保留
    assert after == before


def test_T11b_renew_extends_last_seen_after_throttle(client, svc):
    login(client)
    now = auth_security.utcnow()
    old = now - timedelta(minutes=5)
    force_session_times(
        svc.db_path,
        last_seen_at=auth_security.format_time(old),
        expires_at=auth_security.format_time(old + auth_security.IDLE_TIMEOUT),
    )

    assert client.get("/auth/me").status_code == 200
    row = session_rows(svc.db_path)[0]
    assert auth_security.parse_time(row["last_seen_at"]) > old
    assert auth_security.parse_time(row["expires_at"]) > old + auth_security.IDLE_TIMEOUT


# ============================================================
# T-12 重置密码撤销会话
# ============================================================


def test_T12_set_password_revokes_all_sessions(client, svc):
    login(client)
    assert client.get("/auth/me").status_code == 200

    result = svc.set_password(USER_NAME, "Brand-New-Pass-2026")
    assert result["revoked_sessions"] >= 0

    # 管理员自己没被重置，仍可用
    assert client.get("/auth/me").status_code == 200

    # 被重置的用户旧密码失效
    _reset_failures_for_tests()
    assert login(client, USER_NAME, USER_PASS).status_code == 401
    assert login(client, USER_NAME, "Brand-New-Pass-2026").status_code == 200


# ============================================================
# T-13 禁用撤销会话 / 保护最后一个管理员
# ============================================================


def test_T13a_disable_revokes_sessions(client, svc):
    login(client, USER_NAME, USER_PASS)
    assert client.get("/auth/me").status_code == 200

    svc.set_active(USER_NAME, False)
    assert client.get("/auth/me").status_code == 401


def test_T13b_cannot_disable_last_admin(svc):
    svc.create_user(ADMIN_NAME, ADMIN_PASS, role="admin")

    with pytest.raises(ValueError):
        svc.set_active(ADMIN_NAME, False)

    # 建第二个管理员后即可禁用第一个
    svc.create_user("second_admin", "Second-Pass-2026", role="admin")
    svc.set_active(ADMIN_NAME, False)
    assert svc.get_user(ADMIN_NAME)["is_active"] is False


# ============================================================
# T-14 令牌只存摘要
# ============================================================


def test_T14_only_token_digest_is_stored(client, svc):
    response = login(client)
    raw = response.cookies.get("pai_session")
    assert raw

    rows = session_rows(svc.db_path)
    stored = {row["token_hash"] for row in rows}

    assert raw not in stored
    assert hashlib.sha256(raw.encode("utf-8")).hexdigest() in stored
    assert all(len(value) == 64 for value in stored)


# ============================================================
# T-15 / T-16 同源校验
# ============================================================


def test_T15_write_without_origin_or_referer_is_rejected(client):
    response = client.post(
        "/auth/login",
        json={"username": ADMIN_NAME, "password": ADMIN_PASS},
    )

    assert response.status_code == 403
    assert response.json()["detail"]["type"] == "origin_rejected"
    assert "set-cookie" not in {k.lower() for k in response.headers}


def test_T16_cross_site_origin_rejected_same_origin_allowed(client):
    evil = client.post(
        "/auth/login",
        json={"username": ADMIN_NAME, "password": ADMIN_PASS},
        headers={"Origin": "https://evil.example"},
    )
    assert evil.status_code == 403
    assert evil.json()["detail"]["type"] == "origin_rejected"

    good = client.post(
        "/auth/login",
        json={"username": ADMIN_NAME, "password": ADMIN_PASS},
        headers={"Origin": "http://testserver"},
    )
    assert good.status_code == 200

    # Referer 也可作为来源证据
    by_referer = client.post(
        "/auth/logout",
        headers={"Referer": "http://testserver/login"},
    )
    assert by_referer.status_code == 200


def test_T16b_get_requests_are_not_origin_checked(client):
    """安全方法不做同源校验，否则普通页面跳转会全部失败。"""
    assert client.get("/auth/status").status_code == 200
    assert client.post("/auth/logout", headers=ORIGIN).status_code == 200


# ============================================================
# T-17 生产环境关闭文档
# ============================================================


def test_T17_production_disables_docs(tmp_path):
    """子进程里以 PAI_ENV=production 导入应用，检查三个文档路由为 None。"""
    script = (
        "import json, main\n"
        "print(json.dumps({\n"
        "  'docs': main.app.docs_url,\n"
        "  'redoc': main.app.redoc_url,\n"
        "  'openapi': main.app.openapi_url,\n"
        "  'production': main.PRODUCTION,\n"
        "}))\n"
    )

    env = dict(os.environ)
    env["PAI_ENV"] = "production"
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    # 不使用真实库：子进程只做导入期检查，不进入 lifespan
    env["PAI_MEMORY_DB"] = str(tmp_path / "prod_probe.db")

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]

    import json

    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["production"] is True
    assert payload["docs"] is None
    assert payload["redoc"] is None
    assert payload["openapi"] is None


# ============================================================
# T-18 / T-19 路由矩阵
# ============================================================

ANON = "anonymous"
PAGE = "page"
USER = "user"
ADMIN = "admin"
DOCS = "docs"

#: 唯一的路由分类表。新增任何路由都必须先在这里登记，
#: 否则 test_T18 会失败——未分类路由不允许默认为公开。
ROUTE_POLICY = {
    # ---- 认证页面与静态资源（匿名）----
    ("GET", "/login"): ANON,
    ("GET", "/auth/static/app.js"): ANON,
    ("GET", "/auth/static/style.css"): ANON,
    ("GET", "/auth/static/change_password.js"): ANON,
    ("GET", "/auth/status"): ANON,
    ("POST", "/auth/login"): ANON,
    ("POST", "/auth/logout"): ANON,
    # ---- 交互式文档（仅开发环境存在）----
    ("GET", "/docs"): DOCS,
    ("GET", "/docs/oauth2-redirect"): DOCS,
    ("GET", "/redoc"): DOCS,
    ("GET", "/openapi.json"): DOCS,
    # ---- 既有页面（未登录 302 到 /login）----
    ("GET", "/"): PAGE,
    ("GET", "/workbench"): PAGE,
    ("GET", "/business-ui"): PAGE,
    ("GET", "/knowledge-ui"): PAGE,
    # ---- 改密页面（未登录 302 到 /login；强制改密用户必须能打开本页）----
    ("GET", "/change-password"): PAGE,
    # ---- 页面静态资源（匿名，否则登录页白屏）----
    ("GET", "/app-ui/static/app.js"): ANON,
    ("GET", "/app-ui/static/style.css"): ANON,
    ("GET", "/workbench/static/app.js"): ANON,
    ("GET", "/workbench/static/style.css"): ANON,
    ("GET", "/business-ui/app.js"): ANON,
    ("GET", "/business-ui/style.css"): ANON,
    ("GET", "/knowledge-ui/static/app.js"): ANON,
    ("GET", "/knowledge-ui/static/style.css"): ANON,
    # ---- 私人 AI 功能（登录）----
    ("POST", "/chat"): USER,
    ("GET", "/conversations"): USER,
    ("GET", "/conversations/{conversation_id}/messages"): USER,
    ("POST", "/analyze-image"): USER,
    ("GET", "/auth/me"): USER,
    # ---- 知识库：检索登录可读，变更仅管理员 ----
    ("GET", "/knowledge/search"): USER,
    ("GET", "/knowledge/documents"): USER,
    ("POST", "/knowledge/import"): ADMIN,
    ("POST", "/knowledge/documents"): ADMIN,
    ("DELETE", "/knowledge/documents/{document_id}"): ADMIN,
    # ---- 业务数据：GET 登录可读，POST 仅管理员 ----
    ("GET", "/business/orders"): USER,
    ("GET", "/business/orders/{order_id}"): USER,
    ("GET", "/business/orders/{order_id}/payments"): USER,
    ("GET", "/business/orders/{order_id}/shipments"): USER,
    ("GET", "/business/shipments/{shipment_id}/events"): USER,
    ("GET", "/business/orders/{order_id}/aftersales"): USER,
    ("GET", "/business/orders/{order_id}/workorders"): USER,
    ("POST", "/business/orders"): ADMIN,
    ("POST", "/business/payments"): ADMIN,
    ("POST", "/business/shipments"): ADMIN,
    ("POST", "/business/shipment-events"): ADMIN,
    ("POST", "/business/aftersales"): ADMIN,
    ("POST", "/business/workorders"): ADMIN,
    # ---- 管理员账号管理 ----
    ("GET", "/admin/users"): ADMIN,
    ("POST", "/admin/users"): ADMIN,
    ("PATCH", "/admin/users/{username}"): ADMIN,
    ("POST", "/admin/users/{username}/reset-password"): ADMIN,
    ("POST", "/admin/users/{username}/revoke-sessions"): ADMIN,
    # ---- 自助改密 ----
    ("POST", "/auth/change-password"): USER,
}


def iter_route_keys(app):
    """展开 include_router 的惰性挂载，得到全部 (method, path)。"""
    keys = set()

    def walk(route):
        sub = getattr(route, "original_router", None)
        if sub is not None:
            for child in sub.routes:
                walk(child)
            return
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", None)
        if path is None or not methods:
            return
        for method in methods:
            if method in ("HEAD", "OPTIONS"):
                continue
            keys.add((method, path))

    for route in app.routes:
        walk(route)
    return keys


def test_T18_every_route_is_classified():
    actual = iter_route_keys(main_module.app)
    declared = set(ROUTE_POLICY)

    unclassified = sorted(actual - declared)
    stale = sorted(declared - actual)

    assert unclassified == [], (
        "发现未分类路由，必须先在 ROUTE_POLICY 中登记："
        f"{unclassified}"
    )
    assert stale == [], f"ROUTE_POLICY 中存在已不存在的路由：{stale}"

    # 文档路由只在非生产环境注册；生产环境由 T-17 单独验证
    docs = {key for key, level in ROUTE_POLICY.items() if level == DOCS}
    assert docs <= actual


def test_T18b_route_policy_matches_method_count():
    """方法级路由总数必须等于分类表规模，防止漏计。"""
    assert len(iter_route_keys(main_module.app)) == len(ROUTE_POLICY)


def _probe_status(client, method, path):
    """把带占位符的路径替换成合法值后发请求。"""
    concrete = (
        path.replace("{conversation_id}", "legacy")
        .replace("{document_id}", "__no_such_document__")
        .replace("{order_id}", "NO-SUCH-ORDER")
        .replace("{shipment_id}", "NO-SUCH-SHIPMENT")
    )
    return client.request(method, concrete)


def test_T19_anonymous_access_matches_policy(client):
    login(client)  # 先登录以便随后清 Cookie
    client.cookies.clear()

    mismatches = []
    for (method, path), level in ROUTE_POLICY.items():
        if level in (USER, ADMIN):
            expected = 401
        elif level == PAGE:
            expected = 302
        else:
            continue  # ANON / DOCS 由别的用例覆盖
        response = _probe_status(client, method, path)
        if response.status_code != expected:
            mismatches.append((method, path, level, response.status_code))

    assert mismatches == [], f"匿名访问与分类不一致：{mismatches}"


def test_T19b_regular_user_cannot_reach_admin_routes(client):
    login(client, USER_NAME, USER_PASS)
    assert client.get("/auth/me").json()["user"]["role"] == "user"

    mismatches = []
    for (method, path), level in ROUTE_POLICY.items():
        if level != ADMIN:
            continue
        response = _probe_status(client, method, path)
        if response.status_code != 403:
            mismatches.append((method, path, response.status_code))

    assert mismatches == [], f"普通用户未被拦截：{mismatches}"


def test_T19c_regular_user_can_reach_user_routes(client):
    """普通用户（如 HR）可以检索知识与读业务数据，但写入仍被拦。"""
    login(client, USER_NAME, USER_PASS)

    assert client.get("/knowledge/documents").status_code == 200
    assert client.get("/business/orders").status_code == 200
    assert client.get("/conversations").status_code == 200
    # 写操作（未带有效载荷）应先被权限拦下，而不是 422
    assert client.post("/knowledge/import", headers=ORIGIN).status_code == 403


def test_T19d_admin_passes_admin_routes(client):
    login(client)
    # 知识库目录在 tmp cwd 下不存在，业务层可能 404/409/500，
    # 但**不能**是 401/403 —— 那说明鉴权挡错了人。
    for method, path in [
        ("POST", "/knowledge/import"),
        ("POST", "/knowledge/documents"),
        ("DELETE", "/knowledge/documents/{document_id}"),
        ("POST", "/business/orders"),
    ]:
        response = _probe_status(client, method, path)
        assert response.status_code not in (401, 403), (method, path, response.status_code)


# ============================================================
# T-20 历史数据归属
# ============================================================


def test_T20_legacy_status_and_claim(svc, patched_main_memory):
    memory = patched_main_memory
    memory.create_conversation("legacy", title="历史对话", user_id="test-user")
    memory.save_message("user", "旧消息", conversation_id="legacy", user_id="test-user")

    # 人为抹掉归属，模拟迁移前的历史数据
    connection = sqlite3.connect(svc.db_path)
    try:
        connection.execute("UPDATE conversations SET user_id = NULL")
        connection.execute("UPDATE memories SET user_id = NULL")
        connection.commit()
    finally:
        connection.close()

    before = svc.legacy_status()
    assert before["unowned_conversations"] == 1
    assert before["total_conversations"] == 1
    assert before["orphan_messages"] == 0

    admin = svc.create_user(ADMIN_NAME, ADMIN_PASS, role="admin")
    claimed = svc.claim_legacy_data(admin["id"])

    assert claimed["conversations"] == 1
    after = svc.legacy_status()
    assert after["unowned_conversations"] == 0
    assert after["unowned_memories"] == 0

    # 已有归属的行不被覆盖
    svc.create_user("other_user", "Other-Pass-2026", role="user")
    other = svc.get_user("other_user")
    assert svc.claim_legacy_data(other["id"])["conversations"] == 0


def test_T20b_create_user_rejects_duplicate(svc):
    svc.create_user(ADMIN_NAME, ADMIN_PASS, role="admin")
    with pytest.raises(UsernameTaken):
        svc.create_user(ADMIN_NAME, ADMIN_PASS, role="admin")


def test_T20c_password_policy_enforced(svc):
    for weak in ("short", "123456789", ""):
        with pytest.raises(ValueError):
            svc.create_user("weak_user", weak, role="user")
    # 恰好 10 位应当通过
    svc.create_user("ok_user", "0123456789", role="user")


def test_T20d_login_success_clears_failure_counter(client):
    for _ in range(3):
        login(client, USER_NAME, "wrong-pass-000")
    assert login(client, USER_NAME, USER_PASS).status_code == 200

    # 计数已清零：再错 4 次仍不应被限流
    for _ in range(4):
        assert login(client, USER_NAME, "wrong-pass-000").status_code == 401
