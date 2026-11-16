"""PAI-AUTH-MVP：登录页与匿名访问边界的针对性测试。

本文件在 PAI-UI-LOGIN-R1 的同名测试基础上改写，原因：认证后端已真实接入，
原断言「POST /auth/login 固定返回 501」「存在 /register 页面」等已与实现相反，
继续保留会变成对错误行为的锁定。

覆盖范围
--------
A. 登录页可加载、必需区块齐全、静态资源匿名可取；
B. 表单与脚本的结构契约（输入框属性、reveal 切换、提示区显隐）；
C. 匿名访问边界：三个页面 302 到 /login，数据接口 401；
D. 不存在注册入口（路由与页面链接双双缺失）；
E. 前端不硬编码凭据、不自行保存登录态。

边界说明
--------
* A/C/D 是**真实 HTTP 往返**（Starlette TestClient 直接驱动 main.app）。
* B/E 是**结构契约断言**：校验 HTML 元素契约与 JS 绑定代码一致。
  它证明「页面结构与脚本按契约接好了」，不等价于「在真实浏览器里点过」。
* 夹具刻意**不使用** ``with TestClient(...)``：进入上下文会触发 lifespan，
  从而在真实 memory.db 上建表。库表变更只允许由迁移流程执行。
* 全程不触发模型 API：根 conftest 已装 SDK / HTTP / 非回环 socket 拦截。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import main as main_module

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_UI_DIR = PROJECT_ROOT / "app_ui"
LOGIN_HTML = APP_UI_DIR / "login.html"
AUTH_CSS = APP_UI_DIR / "static" / "auth.css"
AUTH_JS = APP_UI_DIR / "static" / "auth.js"

INDEX_HTML = APP_UI_DIR / "index.html"
WORKBENCH_HTML = APP_UI_DIR / "workbench.html"
BUSINESS_HTML = PROJECT_ROOT / "business_ui" / "index.html"


# ============================================================
# 夹具
# ============================================================


@pytest.fixture(scope="module")
def client():
    """不进入上下文：不触发 lifespan，不碰真实业务库。"""
    return TestClient(main_module.app, follow_redirects=False)


@pytest.fixture(scope="module")
def login_html(client):
    response = client.get("/login")
    assert response.status_code == 200
    return response.text


def _read(path: Path) -> str:
    assert path.is_file(), f"缺少文件：{path}"
    return path.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def auth_js():
    return _read(AUTH_JS)


@pytest.fixture(scope="module")
def auth_css():
    return _read(AUTH_CSS)


# ============================================================
# A. 登录页与静态资源
# ============================================================


def test_login_page_loads(client):
    response = client.get("/login")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "charset=utf-8" in response.headers["content-type"].lower()

    body = response.text
    assert "<!DOCTYPE html>" in body
    assert 'lang="zh-CN"' in body
    assert "<title>登录 · Personal AI</title>" in body
    assert len(body) > 4000


def test_login_page_contains_required_blocks(login_html):
    """登录页的九个组成部分逐条落地。"""
    required = {
        "品牌区": 'class="brand"',
        "登录标题": 'id="login-title"',
        "状态区": 'id="auth-status"',
        "状态区标题": 'id="auth-status-title"',
        "状态区正文": 'id="auth-status-text"',
        "登录表单": 'id="login-form"',
        "用户名输入": 'id="login-username"',
        "密码输入": 'id="login-password"',
        "错误提示区": 'id="login-alert"',
        "提交按钮": 'id="login-submit"',
    }
    missing = [name for name, needle in required.items() if needle not in login_html]
    assert missing == [], f"登录页缺少：{missing}"


def test_login_page_has_no_register_entry(login_html, auth_js):
    """公开注册已关闭：页面不得留有注册链接或注册文案。"""
    assert "/register" not in login_html
    assert "邀请码" not in login_html
    assert "还没有账号" not in login_html
    # 脚本里也不得残留注册提交逻辑
    assert "register-form" not in auth_js
    assert "registerBlocked" not in auth_js


def test_login_page_does_not_claim_backend_missing(login_html, auth_js):
    """页面不得再声称「认证后端未接入」——那已是事实错误。"""
    for text in (login_html, auth_js):
        assert "认证后端未接入" not in text
        assert "UI 已完成" not in text
        assert "固定返回 501" not in text


def test_auth_static_assets_load(client):
    css = client.get("/auth/static/style.css")
    js = client.get("/auth/static/app.js")

    assert css.status_code == 200
    assert css.headers["content-type"].startswith("text/css")
    assert js.status_code == 200
    assert "javascript" in js.headers["content-type"]

    # 静态资源不得携带任何 Cookie
    assert "set-cookie" not in {k.lower() for k in css.headers}
    assert "set-cookie" not in {k.lower() for k in js.headers}


def test_brand_link_reaches_workbench(login_html):
    assert 'href="/workbench"' in login_html


# ============================================================
# B. 表单与脚本的结构契约
# ============================================================


def _inputs_of(html: str):
    return re.findall(r"<input\b[^>]*>", html)


def _attr(tag: str, name: str):
    match = re.search(r'\b%s\s*=\s*"([^"]*)"' % re.escape(name), tag)
    return match.group(1) if match else None


def test_login_inputs_have_full_form_contract(login_html):
    inputs = {_attr(tag, "id"): tag for tag in _inputs_of(login_html)}

    assert "login-username" in inputs
    assert "login-password" in inputs

    user = inputs["login-username"]
    assert _attr(user, "name") == "username"
    assert _attr(user, "autocomplete") == "username"
    assert _attr(user, "required") is not None or " required" in user
    assert _attr(user, "maxlength") == "32"

    pwd = inputs["login-password"]
    assert _attr(pwd, "name") == "password"
    assert _attr(pwd, "type") == "password"
    assert _attr(pwd, "autocomplete") == "current-password"
    # 与后端 PASSWORD_MIN_LENGTH = 10 对齐
    assert _attr(pwd, "minlength") == "10"
    assert _attr(pwd, "maxlength") == "128"


def test_js_password_limit_matches_html(login_html, auth_js):
    """脚本里的密码下限必须与 HTML、后端三处一致。"""
    assert re.search(r"password:\s*\{\s*min:\s*10\b", auth_js), "auth.js 密码下限不是 10"
    assert 'minlength="10"' in login_html


def test_js_posts_to_real_login_endpoint(auth_js):
    """脚本必须真实提交到 POST /auth/login，且只在 200 时跳转。"""
    assert '"/auth/login"' in auth_js or "'/auth/login'" in auth_js
    assert 'method: "POST"' in auth_js
    assert 'credentials: "same-origin"' in auth_js
    # 只有 200 才跳转，避免把 4xx 当成功
    assert "result.status === 200" in auth_js
    assert "window.location.replace" in auth_js


def test_js_binds_validation_on_blur_and_input(auth_js):
    assert 'addEventListener("blur"' in auth_js
    assert 'addEventListener("input"' in auth_js


def test_js_shows_and_hides_alert(auth_js):
    assert "box.hidden = false" in auth_js
    assert "box.hidden = true" in auth_js


def test_reveal_toggle_implemented_in_js_and_css(auth_js, auth_css):
    assert "data-reveal" in auth_js
    assert 'setAttribute("type"' in auth_js
    assert "aria-pressed" in auth_js
    assert ".reveal" in auth_css
    assert '.reveal[aria-pressed="true"]' in auth_css


def test_status_region_is_polite_and_reachable(login_html):
    assert 'role="status"' in login_html
    assert 'aria-live="polite"' in login_html
    assert 'aria-labelledby="login-title"' in login_html


# ============================================================
# C. 匿名访问边界
# ============================================================


@pytest.mark.parametrize(
    "path,expected_location",
    [
        ("/", "/login"),
        ("/workbench", "/login?next=/workbench"),
        ("/business-ui", "/login?next=/business-ui"),
    ],
)
def test_pages_redirect_to_login_when_anonymous(client, path, expected_location):
    response = client.get(path)

    assert response.status_code == 302
    assert response.headers["location"] == expected_location
    # 不得返回页面正文，避免绕过跳转直接读到内容
    assert response.content in (b"", None) or b"<!DOCTYPE html>" not in response.content


@pytest.mark.parametrize(
    "method,path",
    [
        ("POST", "/chat"),
        ("GET", "/conversations"),
        ("GET", "/conversations/legacy/messages"),
        ("GET", "/knowledge/search?query=x"),
        ("GET", "/knowledge/documents"),
        ("POST", "/knowledge/import"),
        ("POST", "/knowledge/documents"),
        ("DELETE", "/knowledge/documents/abc"),
        ("POST", "/analyze-image"),
        ("GET", "/auth/me"),
    ],
)
def test_data_endpoints_require_login(client, method, path):
    response = client.request(method, path)

    assert response.status_code == 401
    assert response.json()["detail"]["type"] == "unauthenticated"


def test_register_routes_are_gone(client):
    """公开注册彻底关闭：两条旧路由都必须 404。"""
    assert client.get("/register").status_code == 404
    assert client.post("/auth/register", json={}).status_code == 404


def test_auth_status_is_public_and_truthful(client):
    response = client.get("/auth/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is True
    assert payload["authenticated"] is False
    assert payload["user"] is None
    assert payload["capabilities"]["public_register"] is False
    assert payload["capabilities"]["session_cookie"] is True
    assert payload["capabilities"]["user_isolation"] is True
    # 不得再自报 ui_only
    assert payload["stage"] != "ui_only"


def test_static_assets_remain_anonymous(client):
    """页面要求登录，但页面所需的静态资源必须可匿名取到，否则登录页白屏。"""
    for path in (
        "/app-ui/static/style.css",
        "/app-ui/static/app.js",
        "/workbench/static/style.css",
        "/workbench/static/app.js",
        "/business-ui/app.js",
        "/business-ui/style.css",
        "/auth/static/style.css",
        "/auth/static/app.js",
    ):
        assert client.get(path).status_code == 200, path


# ============================================================
# D. 路由基线
# ============================================================


def test_auth_routes_registered_and_register_absent():
    paths = {getattr(route, "path", None) for route in main_module.app.routes}

    must_exist = {
        "/login",
        "/auth/static/style.css",
        "/auth/static/app.js",
        "/auth/status",
        "/auth/me",
        "/auth/login",
        "/auth/logout",
    }
    assert must_exist <= paths, f"缺少认证路由：{sorted(must_exist - paths)}"

    assert "/register" not in paths
    assert "/auth/register" not in paths


def test_original_business_routes_still_registered():
    paths = {getattr(route, "path", None) for route in main_module.app.routes}

    must_exist = {
        "/",
        "/workbench",
        "/business-ui",
        "/chat",
        "/conversations",
        "/conversations/{conversation_id}/messages",
        "/knowledge/search",
        "/knowledge/import",
        "/knowledge/documents",
        "/knowledge/documents/{document_id}",
        "/analyze-image",
    }
    assert must_exist <= paths, f"原有路由丢失：{sorted(must_exist - paths)}"


# ============================================================
# E. 无硬编码凭据 / 不自行保存登录态
# ============================================================

_SECRET_PATTERNS = (
    (r"sk-[A-Za-z0-9_\-]{16,}", "OpenAI 风格 API Key"),
    (r"AKIA[0-9A-Z]{16}", "AWS Access Key"),
    (r"-----BEGIN [A-Z ]*PRIVATE KEY-----", "私钥"),
    (r"(?i)\bapi[_-]?key\b\s*[:=]\s*['\"][^'\"]{6,}['\"]", "api_key 字面量"),
    (r"(?i)\bsecret\b\s*[:=]\s*['\"][^'\"]{6,}['\"]", "secret 字面量"),
    (r"(?i)\btoken\b\s*[:=]\s*['\"][A-Za-z0-9_\-\.]{8,}['\"]", "token 字面量"),
    (r"(?i)\b(passwd|password|pwd)\b\s*[:=]\s*['\"][^'\"]{3,}['\"]", "密码字面量"),
)


@pytest.mark.parametrize("target", ["login_html", "auth_js", "auth_css"])
def test_no_hardcoded_credential(target, request):
    text = request.getfixturevalue(target)
    name = {"login_html": "login.html", "auth_js": "auth.js", "auth_css": "auth.css"}[target]

    hits = []
    for pattern, label in _SECRET_PATTERNS:
        for match in re.finditer(pattern, text):
            hits.append((label, match.group(0)[:40]))

    assert hits == [], f"{name} 中出现疑似硬编码凭据：{hits}"


@pytest.mark.parametrize("target", ["login_html", "auth_js"])
def test_no_client_side_login_state(target, request):
    """前端不写登录态：不碰 document.cookie、不碰 localStorage/sessionStorage。"""
    text = request.getfixturevalue(target)

    for forbidden in (
        "document.cookie",
        "localStorage",
        "sessionStorage",
        "indexedDB",
    ):
        assert forbidden not in text, f"{target} 中出现 {forbidden}"


def test_no_password_in_url_or_storage(auth_js):
    """密码只能出现在请求体里：不得拼进 query string。"""
    assert "password=" not in auth_js
    assert "?username=" not in auth_js
    # 请求体形态必须是 JSON.stringify({username, password})
    assert "JSON.stringify" in auth_js
    assert '"Content-Type": "application/json"' in auth_js or "'Content-Type': 'application/json'" in auth_js


def test_next_redirect_cannot_leave_site(auth_js):
    """?next= 只接受站内绝对路径，阻断开放重定向。"""
    assert "nextTarget" in auth_js
    assert 'charAt(0) === "/"' in auth_js
    assert 'charAt(1) !== "/"' in auth_js
