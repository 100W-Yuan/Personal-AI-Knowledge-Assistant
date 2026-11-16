"""PAI-WORKBENCH-R1-2-S2-K1：知识库页面入口与访问守卫测试。

覆盖 K1 任务书 11 项验收，编号 K1-01 ~ K1-11：

  K1-01        匿名 GET /knowledge-ui → 302 /login
  K1-02        普通用户 GET /knowledge-ui → 200（骨架内容正确）
  K1-03        管理员 GET /knowledge-ui → 200
  K1-04        must_change 用户 GET /knowledge-ui → 302 /change-password
  K1-05        固定静态资源匿名 200 且 content-type 正确
  K1-06        白名单外路径 404（含 %2e%2e 编码穿越）
  K1-07        ROUTE_POLICY 与实际路由一致（页面 PAGE、静态 ANON）
  K1-08        工作台入口存在且全角色可见（无 data-requires-role / 无 hidden）
  K1-09        iframe 复用机制未被破坏（setAttribute("src") 全文件仅 1 次）
  K1-10        AI / Business 既有路由未被破坏
  K1-11        app.js 骨架契约：无任何知识库业务请求，仅身份占位

测试风格与 test_workbench_s1.py / test_auth_ui_login.py 一致：
* HTML/JS/CSS 为**结构契约断言**（元素契约与脚本绑定一致）；
* 访问守卫语义为**真实 HTTP 往返**（TestClient 驱动 main.app）；
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
KNOWLEDGE_HTML = PROJECT_ROOT / "knowledge_ui" / "index.html"
KNOWLEDGE_JS = PROJECT_ROOT / "knowledge_ui" / "static" / "app.js"
KNOWLEDGE_CSS = PROJECT_ROOT / "knowledge_ui" / "static" / "style.css"
WORKBENCH_HTML = PROJECT_ROOT / "app_ui" / "workbench.html"
WORKBENCH_JS = PROJECT_ROOT / "app_ui" / "static" / "workbench.js"

#: K2 严格白名单（任务书第六节 E）：页面唯一允许的数据接口。
ALLOWED_KNOWLEDGE_ENDPOINTS = {"/knowledge/documents", "/knowledge/search"}

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


@pytest.fixture(scope="module")
def knowledge_html():
    return _read(KNOWLEDGE_HTML)


@pytest.fixture(scope="module")
def knowledge_js():
    return _read(KNOWLEDGE_JS)


@pytest.fixture(scope="module")
def knowledge_css():
    return _read(KNOWLEDGE_CSS)


@pytest.fixture(scope="module")
def workbench_html():
    return _read(WORKBENCH_HTML)


@pytest.fixture(scope="module")
def workbench_js():
    return _read(WORKBENCH_JS)


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


def login(client, username, password):
    return client.post(
        "/auth/login",
        json={"username": username, "password": password},
        headers=ORIGIN,
    )


# ============================================================
# 访问守卫：真实 HTTP 往返（K1-01 ~ K1-06）
# ============================================================


def test_K01_anonymous_page_redirects_to_login(client):
    """K1-01：未登录访问 /knowledge-ui 必须 302 到 /login。"""
    response = client.get("/knowledge-ui")
    assert response.status_code == 302, response.text
    # 与 /business-ui 同一守卫：302 带 next 参数（登录后回跳本页）。
    assert response.headers["location"].split("?")[0] == "/login"


def test_K02_normal_user_page_200(client):
    """K1-02：普通用户访问 /knowledge-ui 得到 200（K2：真实只读视图）。"""
    assert login(client, USER_NAME, USER_PASS).status_code == 200
    response = client.get("/knowledge-ui")
    assert response.status_code == 200, response.text
    assert "text/html" in response.headers["content-type"]
    body = response.text
    assert "知识库" in body
    assert 'id="identity-status"' in body
    # K2：占位区已替换为真实的列表 / 搜索功能结构。
    assert 'id="doc-state"' in body
    assert 'id="doc-table-wrap"' in body
    assert 'id="doc-rows"' in body
    assert 'id="search-form"' in body
    assert 'id="search-state"' in body
    assert 'id="search-results"' in body


def test_K03_admin_page_200(client):
    """K1-03：管理员访问 /knowledge-ui 同样 200（页面全角色可见）。"""
    assert login(client, ADMIN_NAME, ADMIN_PASS).status_code == 200
    response = client.get("/knowledge-ui")
    assert response.status_code == 200, response.text


def test_K04_must_change_user_redirects_to_change_password(forced_client):
    """K1-04：强制改密用户访问 /knowledge-ui 必须 302 到 /change-password。"""
    response = forced_client.get("/knowledge-ui")
    assert response.status_code == 302, response.text
    assert response.headers["location"] == "/change-password"


@pytest.mark.parametrize(
    "path,media",
    [
        ("/knowledge-ui/static/app.js", "application/javascript"),
        ("/knowledge-ui/static/style.css", "text/css"),
    ],
)
def test_K05_static_assets_anonymous_200(client, path, media):
    """K1-05：两个固定静态资源匿名可访问（页面样式否则白屏）。"""
    response = client.get(path)
    assert response.status_code == 200, response.text
    assert media in response.headers["content-type"]


@pytest.mark.parametrize(
    "path",
    [
        "/knowledge-ui/secret.js",
        # K1-FIX-01：旧静态路径已下线，必须 404，不得与新路径并存。
        "/knowledge-ui/app.js",
        "/knowledge-ui/style.css",
        "/knowledge-ui/%2e%2e/main.py",  # 编码穿越尝试
        "/knowledge-ui/%2e%2e/auth/service.py",
    ],
)
def test_K06_paths_outside_whitelist_404(client, path):
    """K1-06：白名单外任何路径（含旧静态路径与穿越尝试）一律 404。"""
    response = client.get(path)
    assert response.status_code == 404, f"{path} 应 404，实际 {response.status_code}"


# ============================================================
# ROUTE_POLICY 一致性（K1-07）
# ============================================================


def test_K07_route_policy_registers_knowledge_ui():
    """K1-07：/knowledge-ui 页面必须登记为 PAGE，静态资源登记为 ANON。"""
    # 复用 test_auth_mvp 的唯一分类表与展开器，避免两份清单漂移。
    from test_auth_mvp import ANON, PAGE, ROUTE_POLICY, iter_route_keys

    assert ROUTE_POLICY[("GET", "/knowledge-ui")] == PAGE
    assert ROUTE_POLICY[("GET", "/knowledge-ui/static/app.js")] == ANON
    assert ROUTE_POLICY[("GET", "/knowledge-ui/static/style.css")] == ANON
    # K1-FIX-01：旧路径不得残留在分类表中。
    assert ("GET", "/knowledge-ui/app.js") not in ROUTE_POLICY
    assert ("GET", "/knowledge-ui/style.css") not in ROUTE_POLICY

    actual = iter_route_keys(main_module.app)
    assert ("GET", "/knowledge-ui") in actual
    assert ("GET", "/knowledge-ui/static/app.js") in actual
    assert ("GET", "/knowledge-ui/static/style.css") in actual
    # K1-FIX-01：旧路径不得再被注册。
    assert ("GET", "/knowledge-ui/app.js") not in actual
    assert ("GET", "/knowledge-ui/style.css") not in actual


# ============================================================
# 工作台入口与 iframe 机制（结构契约，K1-08 ~ K1-10）
# ============================================================


def _nav_knowledge_item(workbench_html: str) -> str:
    """取出主菜单里 knowledge 导航项的完整 <a> 元素文本。"""
    start = workbench_html.index('<a class="nav-item" href="#/knowledge"')
    end = workbench_html.index("</a>", start) + len("</a>")
    return workbench_html[start:end]


def _quick_knowledge_item(workbench_html: str) -> str:
    """取出快捷卡里知识库入口的完整元素文本。"""
    start = workbench_html.index('href="#/knowledge" data-route="knowledge"')
    start = workbench_html.rindex("<a", 0, start)
    end = workbench_html.index("</a>", start) + len("</a>")
    return workbench_html[start:end]


def test_K08_workbench_entry_visible_to_all_roles(workbench_html):
    """K1-08：工作台主菜单与快捷卡均有知识库入口，且全角色可见。"""
    nav_item = _nav_knowledge_item(workbench_html)
    assert 'href="#/knowledge"' in nav_item
    assert 'data-route="knowledge"' in nav_item
    # 全角色可见：不得有角色限制，也不得初始隐藏。
    # 注意不能直接搜 "hidden" 子串——SVG 的 aria-hidden 会误伤；
    # 真正的 hidden 属性前面必然是空格（" hidden>" 或 " hidden\n"）。
    assert "data-requires-role" not in nav_item
    assert " hidden" not in nav_item
    assert "知识库" in nav_item

    quick_item = _quick_knowledge_item(workbench_html)
    assert quick_item.startswith("<a"), "快捷卡知识库入口必须是可点击 <a>"
    assert 'href="#/knowledge"' in quick_item
    # 「未开放」chip 必须已从知识库入口移除。
    assert "chip" not in quick_item
    assert "未开放" not in quick_item


def test_K09_iframe_reuse_contract_unchanged(workbench_js):
    """K1-09：iframe 复用机制未被破坏。

    ensureFrame 的核心契约：setAttribute("src") 全文件只出现 1 次
    （每模块只创建一次 iframe、绝不重设 src）；知识库条目以 pane=frame
    接入同一机制，而不是另起一套加载逻辑。
    """
    assert workbench_js.count('setAttribute("src"') == 1
    assert 'knowledge: { pane: "frame", title: "知识库", src: "/knowledge-ui" }' in workbench_js
    # 知识库没有专属的加载/销毁逻辑（复用统一 iframe 通道）。
    assert "knowledgeFrame" not in workbench_js
    assert "loadKnowledge" not in workbench_js


def test_K10_existing_ai_business_routes_unchanged(workbench_js):
    """K1-10：AI 助手与业务管理的既有路由条目未被本次改动破坏。"""
    assert 'ai: { pane: "frame", title: "AI 助手", src: "/" }' in workbench_js
    assert 'business: { pane: "frame", title: "业务管理", src: "/business-ui" }' in workbench_js
    # 路由声明顺序：知识库在 business 之后、admin 占位之前。
    assert (
        workbench_js.index("knowledge: { pane")
        < workbench_js.index("admin: { pane")
    )
    assert workbench_js.index("business: { pane") < workbench_js.index("knowledge: { pane")


# ============================================================
# 页面骨架契约（K1-11）
# ============================================================


def test_K11_app_js_knowledge_requests_are_strictly_whitelisted(knowledge_js, knowledge_html, knowledge_css):
    """K1-11（K3A 修订）：知识库请求严格白名单。

    随阶段演进：
      * K1 骨架要求「零业务请求」；
      * K2 允许两个只读接口；
      * K3A 允许一个写接口——POST /knowledge/documents（仅管理员上传）。
    始终锁定：端点集合不扩大、只允许 1 处 POST、PUT/PATCH/DELETE 为零、
    无删除控件、搜索只由提交触发。
    """
    # 1) 所有 /knowledge/... 端点字面量必须恰好是白名单两项（上传复用同一路径）。
    found = set(re.findall(r'"(/knowledge/[^"]*)"', knowledge_js))
    assert found == ALLOWED_KNOWLEDGE_ENDPOINTS, (
        f"出现未批准的知识库接口：{found - ALLOWED_KNOWLEDGE_ENDPOINTS}；"
        f"缺少必需接口：{ALLOWED_KNOWLEDGE_ENDPOINTS - found}"
    )

    # 2) 写方法：仅允许上传这一处 POST；其余写动词一律禁止。
    assert knowledge_js.count('method: "POST"') == 1
    assert 'UPLOAD_ENDPOINT = "/knowledge/documents"' in knowledge_js
    for verb in ('"PUT"', '"PATCH"', '"DELETE"'):
        assert verb not in knowledge_js, f"不得出现写方法 {verb}"

    # 3) 上传逻辑（K3A）：允许 FormData 与文件选择，但仅限管理员区，且无删除控件。
    assert "FormData" in knowledge_js
    assert 'type="file"' in knowledge_html
    assert 'id="upload-section"' in knowledge_html
    assert "Content-Type" not in knowledge_js  # 不手动设置 multipart 头
    assert re.search(r"<button[^>]*>\s*(删除|批量删除)", knowledge_html) is None

    # 4) 搜索只由用户提交触发，页面加载不自动检索。
    assert knowledge_js.count("submitSearch(") == 2  # 1 处定义 + 1 处调用
    assert 'addEventListener("submit"' in knowledge_js
    assert "preventDefault()" in knowledge_js
    # 列表只在初始化与用户点击时加载，且固定 with_index=false（含上传后刷新）。
    assert knowledge_js.count("loadDocuments(false);") == 2
    assert knowledge_js.count("loadDocuments(true)") == 1  # 仅「查看索引状态」

    # 5) 身份契约沿用 K1。
    assert 'IDENTITY_ENDPOINT = "/auth/me"' in knowledge_js
    assert 'LOGIN_PAGE = "/login"' in knowledge_js
    assert 'CHANGE_PASSWORD_PAGE = "/change-password"' in knowledge_js

    # 6) 页面只引用本页固定静态资源（K1-FIX-01：/knowledge-ui/static/ 前缀）。
    assert 'href="/knowledge-ui/static/style.css"' in knowledge_html
    assert 'src="/knowledge-ui/static/app.js"' in knowledge_html
    assert 'href="/knowledge-ui/style.css"' not in knowledge_html
    assert 'src="/knowledge-ui/app.js"' not in knowledge_html

    # 7) 样式文件存在且覆盖状态与占位类。
    assert ".placeholder" in knowledge_css
    assert ".topbar" in knowledge_css
