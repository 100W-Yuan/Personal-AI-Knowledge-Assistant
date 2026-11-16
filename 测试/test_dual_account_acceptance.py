# -*- coding: utf-8 -*-
"""PAI-AUTH-MVP：双账号隔离验收（补充任务第 3 节）。

在**隔离测试库**上验证 7 项验收点，绝不触碰正式 memory.db / data/business.db：

  P1  管理员可登录
  P2  HR（普通用户）可登录
  P3  管理员无法读取 HR 的私人数据
  P4  HR 无法读取管理员的私人数据
  P5  HR 无法调用管理员写接口（业务 6 个 POST + 知识库写）→ 403 forbidden
  P6  HR 可以检索知识库 → 200
  P7  管理员可以修改知识库 → 200（非 401/403）

隔离手段：
- 数据库：``patched_main_memory``（临时 memory.db）+ ``isolated_business_db``；
- 知识库：``main.KNOWLEDGE_DIR`` 重定向到 tmp，检索用轻量替身（不加载真实
  Embedding 模型）；上传走真实落盘路径。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import main as main_module
from auth.service import AuthService, _reset_failures_for_tests

ORIGIN = {"Origin": "http://testserver"}

ADMIN_NAME = "accept_admin"
ADMIN_PASS = "Accept-Admin-2026"
HR_NAME = "accept_hr"
HR_PASS = "Accept-Hr-2026-ok"

# HTTP 层 normalize_conversation_id 要求合法 UUID，Memory 层要求 [0-9a-zA-Z_-]，
# 因此用同时满足两者的 UUID 字符串。
ADMIN_CID = "11111111-1111-4111-8111-111111111111"
HR_CID = "22222222-2222-4222-8222-222222222222"


class _StubKnowledgeBase:
    """仅用于验证路由鉴权的轻量知识库替身，不加载真实模型。"""

    def search(self, query, top_k=3, threshold=None):
        return {"ok": True, "query": query, "results": [], "count": 0, "context": ""}


@pytest.fixture
def service(patched_main_memory):
    svc = AuthService(db_path=patched_main_memory.db_path)
    svc.ensure_schema()
    _reset_failures_for_tests()
    yield svc
    _reset_failures_for_tests()


@pytest.fixture
def accounts(service):
    service.create_user(ADMIN_NAME, ADMIN_PASS, role="admin")
    service.create_user(HR_NAME, HR_PASS, role="user")
    return service


@pytest.fixture
def private_data(accounts, patched_main_memory):
    """给管理员与 HR 各建一条私人会话 + 一条私人记忆。"""
    memory = patched_main_memory
    admin_id = accounts.get_user(ADMIN_NAME)["id"]
    hr_id = accounts.get_user(HR_NAME)["id"]

    memory.create_conversation(ADMIN_CID, title="管理员的私人会话", user_id=admin_id)
    memory.save_message("user", "管理员机密内容", conversation_id=ADMIN_CID, user_id=admin_id)

    memory.create_conversation(HR_CID, title="HR 的私人会话", user_id=hr_id)
    memory.save_message("user", "HR 机密内容", conversation_id=HR_CID, user_id=hr_id)

    memory.save_memory(key="admin_secret", value="仅管理员可见", user_id=admin_id)
    memory.save_memory(key="hr_secret", value="仅 HR 可见", user_id=hr_id)

    return {"admin_id": admin_id, "hr_id": hr_id}


@pytest.fixture
def client(accounts, isolated_business_db):
    with TestClient(main_module.app, follow_redirects=False) as c:
        yield c


def _login(client, username, password):
    client.cookies.clear()
    response = client.post(
        "/auth/login",
        json={"username": username, "password": password},
        headers=ORIGIN,
    )
    assert response.status_code == 200, response.text
    return response


def _logout(client):
    client.post("/auth/logout", headers=ORIGIN)
    client.cookies.clear()


def test_dual_account_acceptance(client, private_data, tmp_path, monkeypatch):
    # ---- 知识库隔离：目录重定向 + 检索替身 ----
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    monkeypatch.setattr(main_module, "KNOWLEDGE_DIR", str(kb_dir))
    monkeypatch.setattr(main_module, "get_knowledge_base", lambda: _StubKnowledgeBase())

    results = {}

    # P1 管理员登录
    _login(client, ADMIN_NAME, ADMIN_PASS)
    results["P1_admin_login"] = client.get("/auth/me").json()["user"]["role"] == "admin"
    _logout(client)

    # P2 HR 登录
    _login(client, HR_NAME, HR_PASS)
    results["P2_hr_login"] = client.get("/auth/me").json()["user"]["role"] == "user"
    _logout(client)

    # P3 管理员无法读取 HR 的私人数据
    _login(client, ADMIN_NAME, ADMIN_PASS)
    response = client.get(f"/conversations/{HR_CID}/messages")
    body = response.json()
    results["P3_admin_cannot_read_hr"] = (
        response.status_code == 200
        and body["messages"] == []
        and body["total"] == 0
        and "HR 机密内容" not in response.text
    )
    _logout(client)

    # P4 HR 无法读取管理员的私人数据
    _login(client, HR_NAME, HR_PASS)
    response = client.get(f"/conversations/{ADMIN_CID}/messages")
    body = response.json()
    results["P4_hr_cannot_read_admin"] = (
        response.status_code == 200
        and body["messages"] == []
        and body["total"] == 0
        and "管理员机密内容" not in response.text
    )

    # P5 HR 无法调用管理员写接口（业务 6 个 POST + 知识库写）
    violations = []
    for method, path in [
        ("POST", "/business/orders"),
        ("POST", "/business/payments"),
        ("POST", "/business/shipments"),
        ("POST", "/business/shipment-events"),
        ("POST", "/business/aftersales"),
        ("POST", "/business/workorders"),
        ("POST", "/knowledge/import"),
        ("POST", "/knowledge/documents"),
        ("DELETE", "/knowledge/documents/no-such-doc"),
    ]:
        response = client.request(method, path, headers=ORIGIN)
        is_forbidden = (
            response.status_code == 403
            and response.json().get("detail", {}).get("type") == "forbidden"
        )
        if not is_forbidden:
            violations.append((method, path, response.status_code, response.text[:120]))
    results["P5_hr_blocked_from_admin_write"] = len(violations) == 0

    # P6 HR 可以检索知识库
    response = client.get("/knowledge/search", params={"query": "验收"})
    results["P6_hr_can_search_kb"] = response.status_code == 200 and response.json()["ok"] is True
    _logout(client)

    # P7 管理员可以修改知识库（真实落盘）
    _login(client, ADMIN_NAME, ADMIN_PASS)
    response = client.post(
        "/knowledge/import",
        files={"file": ("accept.txt", "双账号验收文档内容".encode("utf-8"), "text/plain")},
        headers=ORIGIN,
    )
    results["P7_admin_can_modify_kb"] = (
        response.status_code == 200 and response.json().get("ok") is True
    )

    # 逐项报告失败，不吞细节
    failed = {key: value for key, value in results.items() if value is not True}
    if violations:
        failed["P5_violation_detail"] = violations
    assert failed == {}, f"双账号隔离验收未通过：{failed}"
