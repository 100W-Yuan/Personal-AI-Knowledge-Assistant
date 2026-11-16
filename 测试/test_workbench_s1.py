"""PAI-WORKBENCH-R1-2-S1：统一工作台基础整合测试。

覆盖任务书第五节 12 项要求，编号 S1-01 ~ S1-12：

  S1-01/02/03  登录落点：默认 /workbench；合法 next 保留；外部 next 拒绝
  S1-04/05     admin 可见管理员入口；user 永远不可见
  S1-06        /auth/me 401 → 顶层跳登录页
  S1-07        must_change_password=true → 顶层跳改密页
  S1-08        退出登录真实调用服务端，成功才跳转、失败明确报错
  S1-09/10     AI / Business iframe 创建后复用，切换回来不重新加载
  S1-11        切换模块只隐藏不销毁
  S1-12        未知角色 fail-closed，不获得管理员入口

测试风格与 test_auth_ui_login.py 一致：
* HTML/JS/CSS 为**结构契约断言**（元素契约与脚本绑定一致）；
* 认证/跳转语义为**真实 HTTP 往返**（TestClient 驱动 main.app）；
* 不触发模型 API、不触碰正式库（patched_main_memory + isolated_business_db）。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import main as main_module
from auth.service import AuthService, _reset_failures_for_tests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKBENCH_HTML = PROJECT_ROOT / "app_ui" / "workbench.html"
WORKBENCH_JS = PROJECT_ROOT / "app_ui" / "static" / "workbench.js"
WORKBENCH_CSS = PROJECT_ROOT / "app_ui" / "static" / "workbench.css"
AUTH_JS = PROJECT_ROOT / "app_ui" / "static" / "auth.js"

ORIGIN = {"Origin": "http://testserver"}
ADMIN_NAME = "root_admin"
ADMIN_PASS = "Admin-Pass-2026"
USER_NAME = "hr_user"
USER_PASS = "Hr-Pass-2026-ok"
FORCED_NAME = "forced_user"
FORCED_PASS = "Temp-Pass-2026"


def _read(path: Path) -> str:
    assert path.is_file(), f"缺少文件：{path}"
    return path.read_text(encoding="utf-8")


def _between(src: str, start_marker: str, end_marker: str) -> str:
    """取两个标记之间的源码片段（含 start，不含 end）。

    end 从 start 之后开始查找，避免 end 标记是 start 的前缀时返回空串。
    """
    start = src.index(start_marker)
    end = src.index(end_marker, start + len(start_marker))
    return src[start:end]


@pytest.fixture(scope="module")
def workbench_html():
    return _read(WORKBENCH_HTML)


@pytest.fixture(scope="module")
def workbench_js():
    return _read(WORKBENCH_JS)


@pytest.fixture(scope="module")
def workbench_css():
    return _read(WORKBENCH_CSS)


@pytest.fixture(scope="module")
def auth_js():
    return _read(AUTH_JS)


# ============================================================
# HTTP 夹具（与 test_admin_panel 同构：真实 lifespan + 临时库）
# ============================================================


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
    svc.create_user(USER_NAME, USER_PASS, role="user")
    with TestClient(main_module.app, follow_redirects=False) as c:
        yield c


def login(client, username, password):
    return client.post(
        "/auth/login",
        json={"username": username, "password": password},
        headers=ORIGIN,
    )


# ============================================================
# S1-01 ~ S1-03 登录跳转（auth.js 结构契约）
# ============================================================


def test_S1_01_default_landing_is_workbench(auth_js):
    """登录成功默认进入 /workbench：旧默认 'return \"/\";' 已被替换。"""
    assert "function nextTarget()" in auth_js
    assert 'return "/workbench";' in auth_js
    assert 'return "/";' not in auth_js, "登录默认落点仍是 /，未切换到 /workbench"


def test_S1_02_valid_next_still_honored(auth_js):
    """合法站内 next 仍优先：读取 next 参数 + 接受以单斜杠开头的站内路径。"""
    body = _between(auth_js, "function nextTarget()", "function ")
    assert 'get("next")' in body
    assert 'charAt(0) === "/"' in body


def test_S1_03_external_next_rejected(auth_js):
    """外部 next 仍被拒绝：拒绝 // 、反斜杠；非法时回退 /workbench。"""
    body = _between(auth_js, "function nextTarget()", "function ")
    assert 'charAt(1) !== "/"' in body
    assert r'indexOf("\\") === -1' in body


# ============================================================
# S1-04 / S1-05 / S1-12 角色导航（HTML + JS 契约）
# ============================================================


def test_S1_04_admin_nav_entry_exists_hidden(workbench_html):
    """管理员入口存在但初始 hidden：展示由脚本控制，访问由服务端控制。"""
    anchor = re.search(
        r'<a class="nav-item"[^>]*data-route="admin"[^>]*>', workbench_html
    )
    assert anchor is not None, "缺少管理员导航入口"
    tag = anchor.group(0)
    assert 'data-requires-role="admin"' in tag
    assert "hidden" in tag, "管理员入口必须初始 hidden（fail-closed）"


def test_S1_05_admin_only_reveal(workbench_js):
    """revealAdminNav 全文件只允许出现一次调用，且位于 admin 分支内。"""
    assert workbench_js.count("revealAdminNav()") == 2, (
        "revealAdminNav 必须是「1 处定义 + 1 处调用」"
    )
    # 调用点带分号，与定义（后跟 " {"）区分。
    call_index = workbench_js.index("revealAdminNav();")
    admin_branch = workbench_js.index('user.role === "admin"')
    user_branch = workbench_js.index('user.role === "user"')
    assert admin_branch < call_index, "管理员入口的显示调用不在 admin 分支内"
    assert user_branch > call_index, "user 分支出现在 reveal 之后，顺序异常"


def test_S1_12_unknown_role_fail_closed(workbench_js):
    """未知角色：明确报错、不默认提权、管理员入口保持隐藏。"""
    assert "未知角色，权限未确认" in workbench_js
    # 错误分支之后不得再出现任何 reveal 调用（S1-05 已锁定全文件仅一处）。
    error_index = workbench_js.index("未知角色，权限未确认")
    reveal_call = workbench_js.index("revealAdminNav()")
    assert error_index > reveal_call
    # 身份确认失败同样 fail-closed。
    assert "身份确认失败" in workbench_js


# ============================================================
# S1-06 / S1-07 / S1-08 /auth/me 与退出（JS 契约 + HTTP 往返）
# ============================================================


def test_S1_06_me_401_redirects_to_login(workbench_js, client):
    """JS：401 分支顶层跳登录页；HTTP：未登录 GET /auth/me 真实 401。"""
    assert 'IDENTITY_ENDPOINT = "/auth/me"' in workbench_js
    idx_401 = workbench_js.index("response.status === 401")
    idx_replace = workbench_js.index("window.location.replace(LOGIN_PAGE)")
    assert idx_401 < idx_replace, "401 分支必须先于登录页跳转出现"

    response = client.get("/auth/me")
    assert response.status_code == 401


def test_S1_07_must_change_redirects_top_level(workbench_js, client, svc):
    """JS：must_change=true 顶层 replace /change-password；
    HTTP：服务端 strict 守卫同样 302（双保险）。"""
    assert 'CHANGE_PASSWORD_PAGE = "/change-password"' in workbench_js
    idx_flag = workbench_js.index("must_change_password === true")
    idx_replace = workbench_js.index(
        "window.location.replace(CHANGE_PASSWORD_PAGE)"
    )
    assert idx_flag < idx_replace, "顶层改密跳转必须由 must_change 分支触发"

    svc.create_user(FORCED_NAME, FORCED_PASS, role="user", must_change_password=True)
    login(client, FORCED_NAME, FORCED_PASS)
    response = client.get("/workbench")
    assert response.status_code == 302
    assert response.headers["location"] == "/change-password"


def test_S1_08_logout_calls_server(workbench_js, client):
    """退出登录：真实 POST /auth/logout；仅成功才跳登录页，失败报错不跳转。"""
    assert 'LOGOUT_ENDPOINT = "/auth/logout"' in workbench_js
    body = _between(workbench_js, "function doLogout()", "/* ---------- 初始化")
    assert '"POST"' in body or "'POST'" in body
    assert "window.location.replace(LOGIN_PAGE)" in body
    assert "退出失败" in body
    # 失败分支不得包含跳转：body 中 replace 只能出现一次（成功分支）。
    assert body.count("window.location.replace") == 1

    # HTTP 往返：登录 → 登出 → 会话确实失效（服务端撤销，非仅前端清状态）。
    login(client, USER_NAME, USER_PASS)
    assert client.get("/auth/me").status_code == 200
    response = client.post("/auth/logout", headers=ORIGIN)
    assert response.status_code == 200
    assert client.get("/auth/me").status_code == 401


# ============================================================
# S1-09 / S1-10 / S1-11 iframe 复用机制（workbench.js 契约）
# ============================================================


def test_S1_09_and_10_frame_urls_and_reuse(workbench_js):
    """AI/Business 的 src 保持不变；ensureFrame 对已存在节点绝不重设 src。"""
    assert 'ai: { pane: "frame", title: "AI 助手", src: "/" }' in workbench_js
    assert (
        'business: { pane: "frame", title: "业务管理", src: "/business-ui" }'
        in workbench_js
    )

    body = _between(workbench_js, "function ensureFrame", "function activateFrame")
    assert body.count('setAttribute("src"') == 1, (
        "ensureFrame 中 setAttribute('src') 只能出现在创建分支，"
        "否则复用时会触发重新加载"
    )
    assert body.index("if (el)") < body.index("createElement"), (
        "复用分支必须先于创建分支"
    )
    # 复用分支只允许同步 title（不做任何 src/加载类操作）。
    reuse_branch = body[body.index("if (el)"):body.index("createElement")]
    assert 'setAttribute("src"' not in reuse_branch
    assert 'setAttribute("title"' in reuse_branch


def test_S1_11_switching_hides_not_destroys(workbench_js):
    """切换模块只切可见性：hideFrames 仅移除 is-active，全程无 DOM 销毁调用。"""
    body = _between(workbench_js, "function hideFrames", "/* ---------- 渲染")
    assert 'classList.remove("is-active")' in body
    assert "removeChild" not in workbench_js
    assert "innerHTML" not in workbench_js
    assert ".remove()" not in _between(
        workbench_js, "function ensureFrame", "function activateFrame"
    )


def test_S1_admin_route_registered(workbench_js, workbench_html):
    """admin 占位路由已注册且 pane 存在（不开发真实管理页）。"""
    assert 'admin: { pane: "admin", title: "账户管理" }' in workbench_js
    assert 'data-pane="admin"' in workbench_html


# ============================================================
# HTTP 往返：页面守卫与身份区
# ============================================================


def test_S1_workbench_anonymous_still_302(client):
    """外壳页面守卫不变：未登录 302 /login（FIX-01 strict 守卫保持）。"""
    response = client.get("/workbench")
    assert response.status_code == 302
    assert response.headers["location"].startswith("/login")


def test_S1_admin_sees_workbench_with_identity_markup(client):
    """admin 登录后可打开工作台，页面含身份区与管理员入口结构。"""
    login(client, ADMIN_NAME, ADMIN_PASS)
    response = client.get("/workbench")
    assert response.status_code == 200
    body = response.text
    for needle in (
        'id="identity"',
        'id="identity-name"',
        'id="identity-role"',
        'id="logout-btn"',
        'data-route="admin"',
    ):
        assert needle in body, f"工作台缺少 {needle}"


def test_S1_css_identity_styles(workbench_css):
    """身份区样式已追加（不重写既有规则）。"""
    assert ".identity {" in workbench_css
    assert '.identity__meta em[data-role="admin"]' in workbench_css
    assert '.identity__meta em[data-role="error"]' in workbench_css
