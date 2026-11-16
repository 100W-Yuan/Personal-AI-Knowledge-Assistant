"""PAI-WORKBENCH-R1-2-S2-K3A：管理员上传文档测试。

覆盖任务书第九节 A/B/C/D 四组：

  A. 角色与控件   K3A-A01 ~ K3A-A05
  B. 上传请求     K3A-B01 ~ K3A-B06
  C. 成功与错误   K3A-C01 ~ K3A-C09
  D. 安全         K3A-D01 ~ K3A-D05

策略：
* 前端逻辑用**结构契约断言**（读取 app.js / index.html）；
* 后端权限与错误分支用**真实 HTTP 往返**，文件落盘到 tmp_path
  （monkeypatch main.KNOWLEDGE_DIR），绝不写正式知识库目录；
* 不加载 Embedding / 模型：上传链路本身不构建索引。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import main as main_module
import knowledge_store
from auth.service import AuthService, _reset_failures_for_tests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_HTML = PROJECT_ROOT / "knowledge_ui" / "index.html"
KNOWLEDGE_JS = PROJECT_ROOT / "knowledge_ui" / "static" / "app.js"

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
def knowledge_js():
    return _read(KNOWLEDGE_JS)


@pytest.fixture(scope="module")
def knowledge_html():
    return _read(KNOWLEDGE_HTML)


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
def kb_dir(tmp_path, monkeypatch):
    """把知识库目录指向临时目录：绝不写正式 rag_knowledge/。"""
    target = tmp_path / "rag_knowledge"
    target.mkdir()
    monkeypatch.setattr(main_module, "KNOWLEDGE_DIR", str(target))
    return target


def login(client, username=ADMIN_NAME, password=ADMIN_PASS):
    response = client.post(
        "/auth/login",
        json={"username": username, "password": password},
        headers=ORIGIN,
    )
    assert response.status_code == 200
    return response


def upload(client, filename="制度.md", content=b"# hello world\n", content_type="text/markdown"):
    return client.post(
        "/knowledge/documents",
        files={"file": (filename, content, content_type)},
        headers=ORIGIN,
    )


# ============================================================
# A. 角色与控件
# ============================================================


def test_A01_admin_sees_upload_section(client, kb_dir, knowledge_js, knowledge_html):
    """K3A-A01：管理员显示上传区，且后端接受其上传。"""
    # 结构：上传区由 applyRole 按 role=admin 显示。
    assert 'isAdmin = role === "admin";' in knowledge_js
    assert "uploadSection.hidden = !isAdmin;" in knowledge_js
    assert 'id="upload-section"' in knowledge_html

    # 行为：管理员上传成功（后端最终边界允许）。
    login(client)
    response = upload(client)
    assert response.status_code == 200, response.text
    assert response.json()["ok"] is True


def test_A02_user_has_no_upload_controls(client, kb_dir, knowledge_js):
    """K3A-A02：普通用户看不到上传区，且后端拒绝其上传。"""
    # 结构：只有 admin 才显示（其余一律 hidden）。
    assert 'uploadSection.hidden = !isAdmin;' in knowledge_js
    # 行为：普通用户直接请求 → 403 forbidden。
    login(client, USER_NAME, USER_PASS)
    response = upload(client)
    assert response.status_code == 403, response.text
    assert response.json()["detail"]["type"] == "forbidden"


def test_A03_unknown_role_fail_closed(client, kb_dir, knowledge_js):
    """K3A-A03：未知角色 fail-closed，不显示上传区。"""
    assert "未知角色（权限未确认）" in knowledge_js
    # applyRole 只认 admin，任何其他取值都走 hidden 分支。
    assert 'isAdmin = role === "admin";' in knowledge_js
    assert "applyRole(null);" in knowledge_js


def test_A04_identity_failure_hides_upload(knowledge_js):
    """K3A-A04：身份检查失败时不显示上传区。"""
    # 身份请求失败（非 401）与网络异常都显式回落到 applyRole(null)。
    failure_branch = knowledge_js.index('setIdentity("身份确认失败", "is-error")')
    assert "applyRole(null);" in knowledge_js[failure_branch : failure_branch + 200]
    assert knowledge_js.count("applyRole(null);") >= 2


def test_A05_upload_section_hidden_by_default(knowledge_html):
    """K3A-A05：上传区 HTML 初始 hidden（JS 未执行也不暴露）。"""
    section = knowledge_html[
        knowledge_html.index('id="upload-section"') - 60 : knowledge_html.index('id="upload-section"') + 200
    ]
    assert "hidden" in section
    # 没有按角色直出的（服务端渲染）分支。
    assert "data-requires-role" not in knowledge_html


# ============================================================
# B. 上传请求
# ============================================================


def test_B01_uses_formdata_with_file_field(knowledge_js):
    """K3A-B01：使用 FormData，字段名 file。"""
    assert "var form = new FormData();" in knowledge_js
    assert 'form.append("file", file, file.name);' in knowledge_js
    assert "body: form," in knowledge_js


def test_B02_no_manual_content_type(knowledge_js):
    """K3A-B02：不手动设置 multipart 的 Content-Type。"""
    assert "Content-Type" not in knowledge_js
    # 上传 fetch 不带 headers 选项（只有 method/body/credentials）。
    start = knowledge_js.index("return fetch(UPLOAD_ENDPOINT, {")
    block = knowledge_js[start : start + 220]
    assert "headers" not in block
    assert 'method: "POST"' in block
    assert "credentials: \"same-origin\"" in block


def test_B03_upload_only_on_user_click(knowledge_js):
    """K3A-B03：仅用户主动点击时上传，页面加载不自动提交。"""
    # 定义 1 处 + 事件绑定（绑定传函数引用，不带括号）。
    assert knowledge_js.count("function submitUpload()") == 1
    assert knowledge_js.count("submitUpload") == 2
    assert 'uploadSubmit.addEventListener("click", submitUpload);' in knowledge_js
    # init 内只有事件绑定，没有任何 submitUpload() 调用。
    init_block = knowledge_js[knowledge_js.index("function init()") :]
    assert "submitUpload();" not in init_block
    assert 'uploadSubmit.addEventListener("click", submitUpload);' in init_block


def test_B04_duplicate_submit_blocked(knowledge_js):
    """K3A-B04：上传过程中禁止重复提交。"""
    assert "if (uploading) return;" in knowledge_js
    assert "uploading = true;" in knowledge_js
    assert "uploadSubmit.disabled = true;" in knowledge_js
    # 失败与成功后都恢复按钮可用。
    assert knowledge_js.count("uploadSubmit.disabled = false;") >= 2


def test_B05_refreshes_list_with_index_false_after_success(client, kb_dir, knowledge_js):
    """K3A-B05：成功后刷新文档列表，使用默认 with_index=false。"""
    # 结构：成功分支里先渲染结果再刷新列表。
    success_at = knowledge_js.index("renderUploadSuccess(data);")
    refresh_at = knowledge_js.index("loadDocuments(false);", success_at)
    assert success_at < refresh_at
    # 不触发搜索、不请求索引。
    tail = knowledge_js[success_at:refresh_at]
    assert "submitSearch" not in tail
    assert "loadDocuments(true)" not in tail

    # 行为：上传成功后列表接口能立即看到新文件。
    login(client)
    assert upload(client, "可见性.md", b"unique-content-1").status_code == 200
    listing = client.get("/knowledge/documents?with_index=false").json()
    assert "可见性.md" in [d["filename"] for d in listing["documents"]]


def test_B06_success_notice_exact_text(knowledge_js):
    """K3A-B06：必须展示「上传成功 ≠ 已完成解析和索引」的提示语。"""
    assert (
        "文件已保存。上传成功不代表文档已经完成解析和索引，实际状态请查看索引状态。"
        in knowledge_js
    )
    assert "notice.textContent = UPLOAD_NOTICE;" in knowledge_js


# ============================================================
# C. 成功与错误（真实 HTTP）
# ============================================================


def test_C01_success_response_contract(client, kb_dir):
    """K3A-C01：成功响应字段与落盘事实。"""
    login(client)
    response = upload(client, "手册.md", b"# title\ncontent")
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["ok"] is True
    assert body["filename"] == "手册.md"
    assert body["bytes"] == len(b"# title\ncontent")
    assert body["storage_type"] == "persistent"
    assert re.fullmatch(r"[0-9a-f]{32}", body["document_id"])

    # 文件确实落在临时知识库目录（不是正式目录）。
    assert (kb_dir / "手册.md").is_file()


def test_C02_empty_file_400(client, kb_dir):
    """K3A-C02：空文件 → 400 empty_file。"""
    login(client)
    response = upload(client, "空.md", b"")
    assert response.status_code == 400, response.text
    assert response.json()["detail"]["type"] == "empty_file"

    # 纯空白同样视为空。
    response = upload(client, "空白.md", b"   \n\t ")
    assert response.status_code == 400, response.text
    assert response.json()["detail"]["type"] == "empty_file"


def test_C03_unsupported_format_400(client, kb_dir):
    """K3A-C03：不支持的后缀 → 400 unsupported_format。"""
    login(client)
    response = upload(client, "恶意.exe", b"MZ...")
    assert response.status_code == 400, response.text
    detail = response.json()["detail"]
    assert detail["type"] == "unsupported_format"
    assert ".md" in detail["message"]


def test_C04_invalid_filename_400(client, kb_dir):
    """K3A-C04：非法文件名（Windows 保留名）→ 400 invalid_filename。"""
    login(client)
    response = upload(client, "CON.md", b"content")
    assert response.status_code == 400, response.text
    assert response.json()["detail"]["type"] == "invalid_filename"


def test_C05_already_exists_409(client, kb_dir):
    """K3A-C05：同名文件已存在 → 409 already_exists（不覆盖）。"""
    login(client)
    assert upload(client, "同名.md", b"first-content").status_code == 200
    response = upload(client, "同名.md", b"second-content")
    assert response.status_code == 409, response.text
    detail = response.json()["detail"]
    assert detail["type"] == "already_exists"
    # 原文件内容未被覆盖。
    assert (kb_dir / "同名.md").read_bytes() == b"first-content"


def test_C06_duplicate_content_409(client, kb_dir):
    """K3A-C06：内容重复 → 409 duplicate_content（带 existing_filename）。"""
    login(client)
    assert upload(client, "原始.md", b"same-bytes-123").status_code == 200
    response = upload(client, "副本.md", b"same-bytes-123")
    assert response.status_code == 409, response.text
    detail = response.json()["detail"]
    assert detail["type"] == "duplicate_content"
    assert detail["existing_filename"] == "原始.md"


def test_C07_file_too_large_413(client, kb_dir, monkeypatch):
    """K3A-C07：超过大小上限 → 413 file_too_large。"""
    # 真实上限契约未变。
    assert main_module.MAX_KNOWLEDGE_FILE_BYTES == 10 * 1024 * 1024
    # 临时缩小上限，避免构造 10 MiB 负载。
    monkeypatch.setattr(main_module, "MAX_KNOWLEDGE_FILE_BYTES", 128)

    login(client)
    response = upload(client, "大.md", b"a" * 200)
    assert response.status_code == 413, response.text
    assert response.json()["detail"]["type"] == "file_too_large"

    # 上限以内仍可上传。
    assert upload(client, "小.md", b"a" * 100).status_code == 200


def test_C08_write_failed_500(client, kb_dir, monkeypatch):
    """K3A-C08：存储层写失败 → 500 write_failed。"""
    def boom(directory, filename, content):
        raise knowledge_store.WriteFailed("模拟写入失败")

    monkeypatch.setattr(knowledge_store, "import_document", boom)

    login(client)
    response = upload(client, "失败.md", b"content")
    assert response.status_code == 500, response.text
    assert response.json()["detail"]["type"] == "write_failed"


def test_C09_frontend_error_mapping(knowledge_js):
    """K3A-C09：前端按状态码 + detail.type 映射错误文案。"""
    # 400
    assert 'if (type === "empty_file") return "文件内容为空，无法上传";' in knowledge_js
    assert 'return "文件格式不支持（仅 .md / .txt / .pdf / .docx）";' in knowledge_js
    assert '"文件名不合法"' in knowledge_js
    # 403
    assert '"权限不足：只有管理员可以上传文档"' in knowledge_js
    assert '"需要先修改初始密码后才能上传"' in knowledge_js
    # 409
    assert 'if (type === "already_exists")' in knowledge_js
    assert '"同名文件已存在，未覆盖"' in knowledge_js
    assert 'if (type === "duplicate_content")' in knowledge_js
    assert '"内容重复：知识库中已有相同内容的文件"' in knowledge_js
    assert "existing_filename" in knowledge_js
    # 413 / 500
    assert 'return "文件过大：单文件上限 10 MiB";' in knowledge_js
    assert 'return "服务端写入失败"' in knowledge_js
    # 网络异常
    assert '"网络请求失败：无法连接服务端，请重试。"' in knowledge_js
    # 403 绝不能显示成成功：错误分支在上传成功渲染之前。
    submit_at = knowledge_js.index("function submitUpload()")
    error_at = knowledge_js.index("if (!res.ok) {", submit_at)
    render_at = knowledge_js.index("renderUploadSuccess(data);", submit_at)
    assert error_at < render_at


# ============================================================
# D. 安全
# ============================================================


def test_D01_upload_content_rendered_as_text(knowledge_js):
    """K3A-D01：上传相关的所有后端内容都以 textContent 渲染。"""
    assert "innerHTML" not in knowledge_js
    assert "insertAdjacentHTML" not in knowledge_js
    # 文件名/大小/文档标识/提示语全部 textContent。
    for needle in (
        "uploadFileInfo.textContent =",
        'dt.textContent = label;',
        "dd.textContent = value;",
        "notice.textContent = UPLOAD_NOTICE;",
    ):
        assert needle in knowledge_js


def test_D02_no_delete_or_batch_controls(knowledge_js, knowledge_html):
    """K3A-D02：不出现删除控件或 DELETE 请求。"""
    assert '"DELETE"' not in knowledge_js
    assert "deleteDocument" not in knowledge_js
    assert re.search(r"<button[^>]*>\s*(删除|批量)", knowledge_html) is None
    assert "method: \"DELETE\"" not in knowledge_js


def test_D03_backend_is_final_authority(client, kb_dir, knowledge_js):
    """K3A-D03：后端仍是权限最终判定方（前端隐藏不是安全边界）。"""
    # 普通用户即便绕过前端直接发请求，也必须 403。
    login(client, USER_NAME, USER_PASS)
    response = upload(client, "绕过.md", b"bypass-attempt")
    assert response.status_code == 403, response.text
    # 未登录一律 401。
    client.post("/auth/logout", headers=ORIGIN)
    response = upload(client, "匿名.md", b"anonymous")
    assert response.status_code == 401, response.text
    # 前端仍然保留 isAdmin 早退（纵深防御，不作为唯一屏障）。
    assert "if (!isAdmin) {" in knowledge_js


def test_D04_must_change_user_cannot_upload(client, svc, kb_dir):
    """K3A-D04：强制改密用户不能上传（403 password_change_required）。"""
    svc.create_user(FORCED_NAME, FORCED_PASS, role="admin", must_change_password=True)
    response = client.post(
        "/auth/login",
        json={"username": FORCED_NAME, "password": FORCED_PASS},
        headers=ORIGIN,
    )
    assert response.status_code == 200

    response = upload(client, "改密前.md", b"content")
    assert response.status_code == 403, response.text
    assert response.json()["detail"]["type"] == "password_change_required"


def test_D05_readonly_features_unchanged(client, kb_dir, knowledge_js):
    """K3A-D05：既有只读功能未被破坏。"""
    # 只读函数仍然存在且未被改写为写操作。
    for fn in (
        "function loadDocuments(withIndex)",
        "function renderDocuments(data, withIndex)",
        "function submitSearch()",
        "function loadIdentity()",
    ):
        assert fn in knowledge_js

    # 只读接口仍可用（上传不影响列表）。
    login(client)
    assert client.get("/knowledge/documents").status_code == 200
    # 检索接口的权限层仍在（匿名 401）；不实际检索，避免触发模型加载。
    client.post("/auth/logout", headers=ORIGIN)
    assert client.get("/knowledge/search", params={"query": "x"}).status_code == 401
