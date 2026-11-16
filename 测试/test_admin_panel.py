"""PAI-WORKBENCH-R1-1：管理员账号管理 + 首次改密 + 审计 测试。

边界
----
* 全部使用临时库（``patched_main_memory``），不触碰 memory.db / data/business.db。
* 不调用真实模型、不触发 BGE 下载（本文件只测认证与管理，不涉及 Embedding）。
* 审计与密码改密相关断言直接读临时库，不读真实数据。
"""

from __future__ import annotations

import sqlite3

import pytest
from fastapi.testclient import TestClient

import main as main_module
from auth.service import (
    ACTOR_TYPE_HTTP_ADMIN,
    AuthService,
    UsernameTaken,
    _reset_failures_for_tests,
)

ORIGIN = {"Origin": "http://testserver"}

ADMIN_NAME = "root_admin"
ADMIN_PASS = "Admin-Pass-2026"
HR_NAME = "hr_user"
HR_PASS = "Hr-Pass-2026-ok"
NEW_PASS = "New-Pass-2026-ok"


def _audit_rows(db_path):
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        return [
            dict(row)
            for row in connection.execute("SELECT * FROM audit_logs ORDER BY created_at")
        ]
    finally:
        connection.close()


def _user_password_hash(db_path, username):
    connection = sqlite3.connect(db_path)
    try:
        row = connection.execute(
            "SELECT password_hash FROM users WHERE username = ?", (username,)
        ).fetchone()
        return row[0] if row else None
    finally:
        connection.close()


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
    with TestClient(main_module.app, follow_redirects=False) as c:
        yield c


def login(client, username=ADMIN_NAME, password=ADMIN_PASS):
    return client.post(
        "/auth/login",
        json={"username": username, "password": password},
        headers=ORIGIN,
    )


# ============================================================
# CLI 兼容性
# ============================================================


def test_cli_create_user_default_no_force_change(svc):
    """旧 CLI create_user 默认不强制改密（兼容）。"""
    user = svc.create_user(HR_NAME, HR_PASS, role="user")
    assert user["must_change_password"] is False
    assert svc.get_user(HR_NAME)["must_change_password"] is False


def test_cli_set_password_behavior_unchanged(svc):
    """旧 CLI set_password 不强制改密、不写审计。"""
    svc.create_user(HR_NAME, HR_PASS, role="user")
    result = svc.set_password(HR_NAME, NEW_PASS)
    assert result["revoked_sessions"] >= 0
    assert "must_change_password" not in result
    assert svc.get_user(HR_NAME)["must_change_password"] is False
    assert _audit_rows(svc.db_path) == []


# ============================================================
# 创建用户（HTTP 语义）
# ============================================================


def test_create_user_must_change_flag(svc):
    user = svc.create_user(HR_NAME, HR_PASS, role="user", must_change_password=True)
    assert user["must_change_password"] is True
    assert svc.get_user(HR_NAME)["must_change_password"] is True


def test_http_create_user_is_regular_only(client, svc):
    """HTTP 创建用户只能创建普通用户，且临时密码仅回显一次、不落库明文。"""
    login(client)
    response = client.post("/admin/users", json={"username": HR_NAME}, headers=ORIGIN)

    assert response.status_code == 201
    body = response.json()
    assert body["user"]["role"] == "user"
    assert body["user"]["must_change_password"] is True
    temporary = body["temporary_password"]
    assert temporary

    stored_hash = _user_password_hash(svc.db_path, HR_NAME)
    assert stored_hash.startswith("$argon2id$")
    assert temporary not in stored_hash
    # 列表接口不再出现临时密码
    assert temporary not in client.get("/admin/users").text


# ============================================================
# 重置密码
# ============================================================


def test_reset_password_sets_flag_and_revokes_all_sessions(svc):
    admin = svc.create_user(ADMIN_NAME, ADMIN_PASS, role="admin")
    svc.create_user(HR_NAME, HR_PASS, role="user")
    session = svc.login(HR_NAME, HR_PASS)

    result = svc.reset_password(
        HR_NAME, NEW_PASS,
        actor_user_id=admin["id"], actor_type=ACTOR_TYPE_HTTP_ADMIN,
    )

    assert result["must_change_password"] is True
    assert svc.get_user(HR_NAME)["must_change_password"] is True
    # 旧会话全部失效
    assert svc.validate_session(session["token"]) is None
    # 新密码可登录
    assert svc.login(HR_NAME, NEW_PASS)["user"]["username"] == HR_NAME


def test_http_reset_password_returns_temp_once(client, svc):
    svc.create_user(HR_NAME, HR_PASS, role="user")
    login(client)
    response = client.post(
        "/admin/users/%s/reset-password" % HR_NAME, headers=ORIGIN
    )
    assert response.status_code == 200
    body = response.json()
    assert body["must_change_password"] is True
    temporary = body["temporary_password"]
    assert temporary
    assert temporary not in _user_password_hash(svc.db_path, HR_NAME)


# ============================================================
# 首次改密 / 自助改密
# ============================================================


def test_change_password_revokes_other_sessions_keeps_current(svc):
    svc.create_user(HR_NAME, HR_PASS, role="user")
    first = svc.login(HR_NAME, HR_PASS)
    second = svc.login(HR_NAME, HR_PASS)
    user_id = first["user"]["id"]

    svc.change_password(user_id, HR_PASS, NEW_PASS, first["token"])

    # 当前会话保留
    assert svc.validate_session(first["token"]) is not None
    # 其它会话失效
    assert svc.validate_session(second["token"]) is None
    # 改密标记清 0
    assert svc.get_user(HR_NAME)["must_change_password"] is False
    # 旧密码失效、新密码生效
    assert svc.login(HR_NAME, NEW_PASS)["user"]["username"] == HR_NAME


def test_change_password_wrong_current_rejected(svc):
    svc.create_user(HR_NAME, HR_PASS, role="user")
    session = svc.login(HR_NAME, HR_PASS)
    from auth.service import InvalidCredentials

    with pytest.raises(InvalidCredentials):
        svc.change_password(session["user"]["id"], "wrong-pass-000", NEW_PASS, session["token"])

    # 未变
    assert svc.login(HR_NAME, HR_PASS)["user"]["username"] == HR_NAME


def test_http_change_password_flow(client, svc):
    svc.create_user(HR_NAME, HR_PASS, role="user", must_change_password=True)
    login(client, HR_NAME, HR_PASS)

    assert client.get("/auth/me").json()["user"]["must_change_password"] is True

    response = client.post(
        "/auth/change-password",
        json={"current_password": HR_PASS, "new_password": NEW_PASS},
        headers=ORIGIN,
    )
    assert response.status_code == 200
    assert client.get("/auth/me").json()["user"]["must_change_password"] is False


# ============================================================
# 启用 / 禁用
# ============================================================


def test_set_active_cannot_disable_self(svc):
    admin = svc.create_user(ADMIN_NAME, ADMIN_PASS, role="admin")
    with pytest.raises(ValueError):
        svc.set_active(
            ADMIN_NAME, False,
            actor_user_id=admin["id"], actor_type=ACTOR_TYPE_HTTP_ADMIN,
        )


def test_set_active_cannot_disable_last_admin(svc):
    svc.create_user(ADMIN_NAME, ADMIN_PASS, role="admin")
    with pytest.raises(ValueError):
        svc.set_active(ADMIN_NAME, False)


def test_http_admin_cannot_disable_self(client):
    login(client)
    response = client.patch(
        "/admin/users/%s" % ADMIN_NAME,
        json={"is_active": False},
        headers=ORIGIN,
    )
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_operation"


# ============================================================
# 权限
# ============================================================


def test_http_regular_user_cannot_call_admin_api(client, svc):
    svc.create_user(HR_NAME, HR_PASS, role="user")
    login(client, HR_NAME, HR_PASS)

    assert client.get("/admin/users").status_code == 403
    assert client.post("/admin/users", json={"username": "x"}, headers=ORIGIN).status_code == 403
    assert client.patch(
        "/admin/users/%s" % ADMIN_NAME, json={"is_active": True}, headers=ORIGIN
    ).status_code == 403


def test_http_anonymous_cannot_call_admin_api(client):
    assert client.get("/admin/users").status_code == 401
    assert client.post("/admin/users", json={"username": "x"}, headers=ORIGIN).status_code == 401


# ============================================================
# 审计
# ============================================================


def test_audit_success_recorded_atomically(svc):
    admin = svc.create_user(ADMIN_NAME, ADMIN_PASS, role="admin")
    assert _audit_rows(svc.db_path) == []

    svc.create_user(
        HR_NAME, HR_PASS, role="user",
        actor_user_id=admin["id"], actor_type=ACTOR_TYPE_HTTP_ADMIN,
    )

    rows = _audit_rows(svc.db_path)
    assert len(rows) == 1
    assert rows[0]["action"] == "create_user"
    assert rows[0]["result"] == "success"
    assert rows[0]["actor_user_id"] == admin["id"]
    assert rows[0]["actor_type"] == "http_admin"
    assert rows[0]["target_user_id"] == svc.get_user(HR_NAME)["id"]


def test_audit_failure_rolls_back_business(svc, monkeypatch):
    admin = svc.create_user(ADMIN_NAME, ADMIN_PASS, role="admin")

    def boom(*args, **kwargs):
        raise RuntimeError("audit write failed")

    monkeypatch.setattr(AuthService, "_audit_insert", staticmethod(boom))

    with pytest.raises(RuntimeError):
        svc.create_user(
            HR_NAME, HR_PASS, role="user",
            actor_user_id=admin["id"], actor_type=ACTOR_TYPE_HTTP_ADMIN,
        )
    # 业务变更随审计失败一起回滚
    assert svc.get_user(HR_NAME) is None


def test_failure_audit_failure_preserves_original_exception(svc, monkeypatch):
    admin = svc.create_user(ADMIN_NAME, ADMIN_PASS, role="admin")

    def boom(*args, **kwargs):
        raise RuntimeError("audit write failed")

    monkeypatch.setattr(AuthService, "_audit_insert", staticmethod(boom))

    # 业务失败（用户名重复），同时失败审计也写不进去：仍抛原始 UsernameTaken
    with pytest.raises(UsernameTaken):
        svc.create_user(
            ADMIN_NAME, ADMIN_PASS, role="user",
            actor_user_id=admin["id"], actor_type=ACTOR_TYPE_HTTP_ADMIN,
        )


def test_audit_failure_recorded(svc):
    admin = svc.create_user(ADMIN_NAME, ADMIN_PASS, role="admin")
    with pytest.raises(UsernameTaken):
        svc.create_user(
            ADMIN_NAME, ADMIN_PASS, role="user",
            actor_user_id=admin["id"], actor_type=ACTOR_TYPE_HTTP_ADMIN,
        )
    rows = _audit_rows(svc.db_path)
    assert any(row["result"] == "failure" and row["action"] == "create_user" for row in rows)


# ============================================================
# 敏感字段不泄露
# ============================================================


def test_list_users_excludes_sensitive(svc):
    svc.create_user(ADMIN_NAME, ADMIN_PASS, role="admin")
    svc.create_user(HR_NAME, HR_PASS, role="user")
    for user in svc.list_users():
        assert "password_hash" not in user
        assert "token" not in user
        assert "token_hash" not in user


def test_http_responses_do_not_leak_secrets(client):
    login(client)
    for response in (
        client.get("/admin/users"),
        client.post("/admin/users", json={"username": "no_leak_user"}, headers=ORIGIN),
        client.get("/auth/me"),
    ):
        text = response.text.lower()
        assert "argon2" not in text
        assert "password_hash" not in text
        assert "token_hash" not in text
