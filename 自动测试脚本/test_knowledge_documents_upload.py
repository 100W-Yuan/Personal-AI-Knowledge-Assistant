# -*- coding: utf-8 -*-
"""P2-2B：``POST /knowledge/documents`` 四格式统一上传接口测试（离线）。

被测对象：``main.knowledge_upload_document``（P2-2A 新增，路由行见 ``main.py``）。

锁定的是 P2-2A 已落地的契约，而不是本次测试的期望：

1. 路径 ``POST /knowledge/documents``，单文件字段名 ``file``；
2. 四道 HTTP 层闸门，顺序固定为
   **文件名合法 → 后缀白名单 → 大小上限 → 非空**；
3. 内容以**原始 bytes** 直通 ``knowledge_store.import_document``，
   上传层**不解码、不嗅探文件头、不判内容结构**；
4. 成功响应五键：``ok / filename / bytes / document_id / storage_type``，
   其中 ``storage_type`` 恒为 ``persistent``，且 ``ok=True`` **只表示已落盘**
   （未解析、未入索引、不保证可检索）；
5. 存储层异常经 ``main._knowledge_http_error`` 映射为
   ``{"type": error_type, "message": message, **extra}``；
6. ``_invalidate_knowledge_caches()`` 只在写入成功后调用，顺序
   ``知识库 → Agent``，且只调用 ``cache_clear()``，不构造任何对象；
7. ``finally`` 统一关闭上传文件，**有原始异常时不覆盖原始异常**。

分组（共 53 项）：

- A（8）成功路径与响应契约
- B（5）后缀白名单闸门
- C（6）文件名合法性
- D（11）路径安全与边界文件名
- E（9）存储层异常 → HTTP 映射
- F（6）大小 / 空内容 / 文件关闭
- G（4）缓存失效与副作用边界
- I（4）编码与跨接口生命周期补齐（P2-2B-1-R 增补，见该组抬头注释）
- H 隔离标准（模块级约定，见下，不产生用例）

------------------------------------------------------------------
H. 隔离标准（模块级约定）
------------------------------------------------------------------

H1 知识库目录：每个用例用 ``monkeypatch`` 把 ``main.KNOWLEDGE_DIR`` 指向
   ``tmp_path``（或它的子目录），**从不**指向仓库内的 ``rag_knowledge/``。
H2 不写仓库：所有落盘断言的目标都在 ``tmp_path`` 之下；越界断言只做
   「父目录/仓库外没有新文件」的**只读**检查，不创建、不清理仓库文件。
H3 不联网：端点不发起任何网络请求；``TestClient`` 走进程内 ASGI 传输，
   根 ``conftest.py`` 的回环感知网络守卫与 HTTP 守卫对其不生效即视为通过。
H4 不加载模型、不建索引：本文件不构造 ``KnowledgeBase`` / ``Embedding`` /
   ``Agent``；``get_knowledge_base`` / ``get_agent`` 一律用替身替换，
   只允许被 ``cache_clear()``，不允许被调用。
H5 不碰数据库：不读写 ``memory.db`` / ``data/business.db``；
   根 ``conftest.py`` 的 DB 指纹哨兵应保持不变。
H6 不读 ``.env``：``自动测试脚本/conftest.py`` 已把 dotenv 换成假实现。
H7 四格式样本全部现场构造：PDF/TXT/MD 由本文件手工拼字节，DOCX 由
   ``python-docx`` 在内存生成；不依赖仓库内任何既有样本文件。
H8 结论口径：断言全部为**固定期望值**，不使用「满足 A 或满足 B」的条件
   断言；若实测与预期不符，应判为契约变更并回溯 P2-2A，而不是放宽断言。

已知局限（P2-2B 已登记）：

- R2 ``UploadFile.close()`` 的「被调用」断言本身不具判别力（框架清理也会
  关闭），真正的判别力在 F06（关闭失败不得覆盖原始异常）；
- R3 若 ``get_knowledge_base`` 替身缺少 ``cache_clear``，``_invalidate_
  knowledge_caches`` 会静默跳过，因此 G 组替身必须同时可调用、可
  ``cache_clear``。
"""

import hashlib
import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import knowledge_store
import main


client = TestClient(main.app)

_ENDPOINT = "/knowledge/documents"

_MIME_BY_SUFFIX = {
    ".md": "text/markdown",
    ".txt": "text/plain",
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument"
    ".wordprocessingml.document",
}


# ============================================================
# 通用 helper
# ============================================================


def _doc_id(name):
    """独立复算 document_id：sha256(basename) 前 32 位小写十六进制。"""
    return hashlib.sha256(name.encode("utf-8")).hexdigest()[:32]


def _clear(factory):
    clear = getattr(factory, "cache_clear", None)
    if callable(clear):
        clear()


@pytest.fixture(autouse=True)
def _reset_caches():
    _clear(main.get_knowledge_base)
    _clear(main.get_agent)
    yield
    _clear(main.get_knowledge_base)
    _clear(main.get_agent)


class _SpyCache:
    """可调用、也可被 ``cache_clear`` 的替身。

    R3：必须同时具备 ``__call__`` 与 ``cache_clear``，
    否则 ``_invalidate_knowledge_caches`` 的属性探测会静默跳过，
    G 组会出现「全组假绿」。
    """

    def __init__(self, label, sink):
        self.label = label
        self.sink = sink
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return None

    def cache_clear(self):
        self.sink.append(self.label)


def _upload(name, content, mime=None):
    if mime is None:
        mime = _MIME_BY_SUFFIX.get(Path(name).suffix.lower(), "application/octet-stream")
    return client.post(_ENDPOINT, files={"file": (name, content, mime)})


def _reset_dir(monkeypatch, tmp_path, name="kb"):
    target = tmp_path / name
    monkeypatch.setattr(main, "KNOWLEDGE_DIR", str(target))
    return target


def _all_files(root):
    """返回 root 下所有普通文件的相对路径（排序），用于越界检查。"""
    root_path = Path(root)
    return sorted(
        str(path.relative_to(root_path))
        for path in root_path.rglob("*")
        if path.is_file()
    )


# ------------------------------------------------------------
# 四格式样本构造（H7：不依赖仓库既有样本）
# ------------------------------------------------------------
def _escape_pdf_text(text):
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def _build_pdf(page_texts):
    """构造带文本层的最小 PDF。

    与 ``RAG测试/test_knowledge_base_loader_integration.py`` 的 ``build_pdf``
    同法（本文件独立实现，避免跨目录 import）。字体 Helvetica 走 latin-1，
    因此页文本只能是 ASCII。
    """
    page_count = len(page_texts)
    font_num = 3 + 2 * page_count
    page_nums = [3 + 2 * i for i in range(page_count)]

    objects = {}
    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    kids = " ".join("%d 0 R" % num for num in page_nums)
    objects[2] = (
        "<< /Type /Pages /Kids [%s] /Count %d >>" % (kids, page_count)
    ).encode("latin-1")

    for index, text in enumerate(page_texts):
        page_num = 3 + 2 * index
        content_num = page_num + 1
        objects[page_num] = (
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            "/Contents %d 0 R /Resources << /Font << /F1 %d 0 R >> >> >>"
            % (content_num, font_num)
        ).encode("latin-1")
        stream = (
            "BT /F1 12 Tf 72 720 Td (%s) Tj ET" % _escape_pdf_text(text)
        ).encode("latin-1") if text else b""
        objects[content_num] = (
            ("<< /Length %d >>\nstream\n" % len(stream)).encode("latin-1")
            + stream
            + b"\nendstream"
        )

    objects[font_num] = (
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    )

    out = bytearray(b"%PDF-1.4\n")
    offsets = {}
    for num in sorted(objects):
        offsets[num] = len(out)
        out += ("%d 0 obj\n" % num).encode("latin-1")
        out += objects[num]
        out += b"\nendobj\n"

    xref_offset = len(out)
    total = font_num + 1
    out += ("xref\n0 %d\n" % total).encode("latin-1")
    out += b"0000000000 65535 f\r\n"
    for num in range(1, total):
        out += ("%010d 00000 n\r\n" % offsets[num]).encode("latin-1")
    out += (
        "trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n"
        % (total, xref_offset)
    ).encode("latin-1")

    return bytes(out)


def _build_docx_bytes():
    """用 python-docx 在内存生成 DOCX（不落临时文件）。"""
    import docx

    document = docx.Document()
    document.add_paragraph("upload contract fixture")
    document.add_paragraph("second paragraph")
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


# ============================================================
# A 组：成功路径与响应契约（8）
# ============================================================


def test_A01_route_registered_as_post():
    """契约 1：``POST /knowledge/documents`` 必须已注册（不是 405）。"""
    paths = main.app.openapi()["paths"]
    assert _ENDPOINT in paths
    assert "post" in paths[_ENDPOINT]
    assert "get" in paths[_ENDPOINT]

    response = client.post(_ENDPOINT)

    # 缺必填 file 字段 → 422（若路由缺失则为 405）
    assert response.status_code == 422


def test_A02_markdown_upload_response_contract(monkeypatch, tmp_path):
    """契约 4：成功响应五键，且 ok=True 只表示已落盘。"""
    target = _reset_dir(monkeypatch, tmp_path)
    payload = "# 标题\n\n正文内容\n".encode("utf-8")

    response = _upload("note.md", payload)

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "ok",
        "filename",
        "bytes",
        "document_id",
        "storage_type",
    }
    assert body["ok"] is True
    assert body["filename"] == "note.md"
    assert body["bytes"] == len(payload)
    assert body["document_id"] == _doc_id("note.md")
    assert body["storage_type"] == "persistent"
    # 响应中不得出现索引状态字段：索引状态查询属 GET 接口职责
    assert "chunk_count" not in body
    assert "indexed" not in body
    assert (target / "note.md").read_bytes() == payload


def test_A03_txt_upload_success(monkeypatch, tmp_path):
    target = _reset_dir(monkeypatch, tmp_path)
    payload = "纯文本中文内容\n".encode("utf-8")

    response = _upload("说明.txt", payload)

    assert response.status_code == 200
    assert response.json()["filename"] == "说明.txt"
    assert response.json()["bytes"] == len(payload)
    assert (target / "说明.txt").read_bytes() == payload


def test_A04_pdf_upload_success(monkeypatch, tmp_path):
    """真实 PDF（二进制容器）必须整段成功，落盘字节逐字节一致。"""
    target = _reset_dir(monkeypatch, tmp_path)
    payload = _build_pdf(["page one text", "page two text"])
    assert payload.startswith(b"%PDF-1.4")

    response = _upload("report.pdf", payload)

    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "report.pdf"
    assert body["bytes"] == len(payload)
    assert body["document_id"] == _doc_id("report.pdf")
    assert (target / "report.pdf").read_bytes() == payload


def test_A05_docx_upload_success(monkeypatch, tmp_path):
    """真实 DOCX（zip 容器）必须整段成功，落盘字节逐字节一致。"""
    target = _reset_dir(monkeypatch, tmp_path)
    payload = _build_docx_bytes()
    assert payload.startswith(b"PK")

    response = _upload("章节.docx", payload)

    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "章节.docx"
    assert body["bytes"] == len(payload)
    assert (target / "章节.docx").read_bytes() == payload


def test_A06_upload_layer_does_not_decode_or_sniff(monkeypatch, tmp_path):
    """契约 3：上传层只判「空 / 非空」，不解码、不嗅探文件头。

    样本故意是「不可 UTF-8 解码」的二进制：先证明它确实不可解码（前置
    断言，保证本用例有判别力），再断言接口仍返回 200。若上传层加回
    ``content.decode("utf-8")`` 或文件头校验，本用例必然失败。
    """
    target = _reset_dir(monkeypatch, tmp_path)
    payload = b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n" + bytes(range(256))

    with pytest.raises(UnicodeDecodeError):
        payload.decode("utf-8")

    response = _upload("binary.pdf", payload)

    assert response.status_code == 200
    assert response.json()["bytes"] == len(payload)
    assert (target / "binary.pdf").read_bytes() == payload


def test_A07_document_id_consistent_across_layers(monkeypatch, tmp_path):
    """``document_id`` 三方一致：HTTP 响应 / 独立复算 / 存储层实现。"""
    target = _reset_dir(monkeypatch, tmp_path)
    name = "知识库身份校验.md"

    response = _upload(name, "内容".encode("utf-8"))

    assert response.status_code == 200
    expected = _doc_id(name)
    assert response.json()["document_id"] == expected
    assert knowledge_store.compute_document_id(name) == expected
    assert (target / name).exists()


def test_A08_uppercase_suffix_accepted(monkeypatch, tmp_path):
    """后缀判定大小写不敏感，落盘文件名保留原始大小写。"""
    target = _reset_dir(monkeypatch, tmp_path)
    payload = _build_pdf(["upper case suffix"])

    response = _upload("Upper.PDF", payload)

    assert response.status_code == 200
    assert response.json()["filename"] == "Upper.PDF"
    assert response.json()["document_id"] == _doc_id("Upper.PDF")
    assert (target / "Upper.PDF").exists()


# ============================================================
# B 组：后缀白名单闸门（5）
# ============================================================


def test_B01_rejects_unsupported_suffix(monkeypatch, tmp_path):
    target = _reset_dir(monkeypatch, tmp_path)

    response = _upload("script.py", b"print(1)")

    assert response.status_code == 400
    assert response.json()["detail"]["type"] == "unsupported_format"
    assert _all_files(target) == []


def test_B02_rejects_filename_without_suffix(monkeypatch, tmp_path):
    _reset_dir(monkeypatch, tmp_path)

    response = _upload("README", b"content")

    assert response.status_code == 400
    assert response.json()["detail"]["type"] == "unsupported_format"


def test_B03_unsupported_message_derived_from_whitelist(monkeypatch, tmp_path):
    """文案由唯一白名单派生，不得在路由内另写一份格式字面量。"""
    _reset_dir(monkeypatch, tmp_path)

    response = _upload("script.py", b"print(1)")

    expected = "仅支持 " + " / ".join(main.ALLOWED_KNOWLEDGE_SUFFIXES) + " 文件"
    assert response.json()["detail"]["message"] == expected
    assert main.ALLOWED_KNOWLEDGE_SUFFIXES == (".md", ".txt", ".pdf", ".docx")


def test_B04_rejects_pseudo_suffix(monkeypatch, tmp_path):
    """后缀只看最后一段：``report.pdf.bak`` 不是 PDF。"""
    _reset_dir(monkeypatch, tmp_path)

    response = _upload("report.pdf.bak", b"backup")

    assert response.status_code == 400
    assert response.json()["detail"]["type"] == "unsupported_format"


def test_B05_suffix_gate_runs_before_size_gate(monkeypatch, tmp_path):
    """契约 2：闸门顺序固定 —— 后缀判定先于大小判定。

    若顺序被调换，本用例会得到 413 而不是 400。
    """
    _reset_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(main, "MAX_KNOWLEDGE_FILE_BYTES", 16)

    response = _upload("oversized.py", b"x" * 64)

    assert response.status_code == 400
    assert response.json()["detail"]["type"] == "unsupported_format"


# ============================================================
# C 组：文件名合法性（6）
# ============================================================


@pytest.mark.parametrize(
    "name",
    ["CON.md", "bad*name.md", "name.md.", "nul.txt"],
)
def test_C01_to_C04_rejects_invalid_filename(monkeypatch, tmp_path, name):
    """保留设备名 / 非法字符 / 尾随点 / 保留名 → 400 invalid_filename。

    四例都带**合法后缀**（.md / .txt），因此能区分「文件名闸门」与
    「后缀闸门」：若文件名校验被删，它们会退化成 200。
    """
    target = _reset_dir(monkeypatch, tmp_path)

    response = _upload(name, b"content")

    assert response.status_code == 400
    assert response.json()["detail"]["type"] == "invalid_filename"
    assert _all_files(target) == []


def test_C05_directory_prefix_is_stripped(monkeypatch, tmp_path):
    """``docs/sub.md`` 只保留 basename，且不会创建 ``docs`` 子目录。"""
    target = _reset_dir(monkeypatch, tmp_path)

    response = _upload("docs/sub.md", b"content")

    assert response.status_code == 200
    assert response.json()["filename"] == "sub.md"
    assert (target / "sub.md").exists()
    assert not (target / "docs").exists()


def test_C06_filename_validator_rejects_control_and_trailing_space():
    """校验函数层对照：控制字符与尾随空格直接拒绝。

    HTTP 层会把 ``\\t`` 等控制字符百分号转义成字面量（实测 ``x.md\\t``
    解析为 ``x.md%09``），因此这两类样本只能在校验函数层面锁定。
    """
    assert knowledge_store.validate_knowledge_filename("bad\x01name.md") == (
        "invalid_filename"
    )
    assert knowledge_store.validate_knowledge_filename("x.md ") == "invalid_filename"
    assert knowledge_store.validate_knowledge_filename("good.md") is None


# ============================================================
# D 组：路径安全与边界文件名（11）
# ============================================================


def test_D01_empty_filename_falls_to_422(monkeypatch, tmp_path):
    """固定预期：``filename=""`` → 422，不是 400。

    依据（只读核对 Starlette 1.6.0 + httpx 0.28.1 源码）：

    - httpx ``FileField.render_headers`` 仅在 ``if self.filename:`` 为真时才
      写 ``filename="..."`` 参数，空串 → header 中**没有** filename；
    - Starlette ``MultiPartParser.on_headers_finished`` 以
      ``if b"filename" in options:`` 决定是否构造 ``UploadFile``，缺失时该
      part 退化为普通表单字段（字符串）；
    - FastAPI 对 ``file: UploadFile = File(...)`` 收到字符串 → 请求体校验失败
      → **422**，永远到不了路由内的 ``empty_file`` 分支。

    这是 multipart 解析层的既定行为，不是生产缺陷，故写死 422。
    """
    target = _reset_dir(monkeypatch, tmp_path)

    response = _upload("", b"content")

    assert response.status_code == 422
    assert isinstance(response.json()["detail"], list)
    assert _all_files(target) == []


def test_D02_parent_traversal_posix_separator(monkeypatch, tmp_path):
    target = _reset_dir(monkeypatch, tmp_path)

    response = _upload("../../evil.md", b"escape")

    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "evil.md"
    # 身份基于归约后的 basename，而不是原始穿越串
    assert body["document_id"] == _doc_id("evil.md")
    assert (target / "evil.md").exists()
    # 只有 KNOWLEDGE_DIR 内那一份，父目录不被写入
    assert not (tmp_path / "evil.md").exists()
    assert not (tmp_path.parent / "evil.md").exists()
    assert _all_files(target) == ["evil.md"]


def test_D03_parent_traversal_windows_separator(monkeypatch, tmp_path):
    """``..\\evil.md``：反斜杠形态，并用列表接口做跨接口复核。

    两层防线各自独立：
    ``python_multipart.parse_options_header`` 只对 UNC / 盘符开头的值按
    ``\\`` 截断，``..\\evil.md`` 会**原样**进入路由；真正拦住它的是
    ``Path(original).name``（Windows 语义下 ``\\`` 也是分隔符）。
    """
    target = _reset_dir(monkeypatch, tmp_path)

    response = _upload("..\\evil.md", b"escape")

    assert response.status_code == 200
    assert response.json()["filename"] == "evil.md"
    assert (target / "evil.md").exists()

    listing = client.get(_ENDPOINT).json()
    assert listing["count"] == 1
    assert [item["filename"] for item in listing["documents"]] == ["evil.md"]


def test_D04_traversal_never_writes_outside(monkeypatch, tmp_path):
    """路径穿越（两种分隔符）：最终只会落在隔离的 KNOWLEDGE_DIR 中。

    断言对象是**落盘位置**而非错误码：穿越名被合法归约为 basename，请求
    本身成功；关键是父目录（``tmp_path`` 与 ``tmp_path.parent``）不得出现
    任何新文件。若 ``Path(original).name`` 被删除，两种形态都会在父目录
    留下 ``evil.md``。

    两种形态各自使用独立的 KNOWLEDGE_DIR，以免第二次上传命中同名判定，
    保证两条路径都被真实走通。
    """
    names = ["../../evil.md", "..\\evil.md"]

    for index, name in enumerate(names, start=1):
        target = _reset_dir(monkeypatch, tmp_path, name="kb%d" % index)

        response = _upload(name, b"escape")

        assert response.status_code == 200
        assert response.json()["filename"] == "evil.md"
        assert _all_files(target) == ["evil.md"]

    assert not (tmp_path / "evil.md").exists()
    assert not (tmp_path.parent / "evil.md").exists()


def test_D05_absolute_posix_path_is_reduced(monkeypatch, tmp_path):
    target = _reset_dir(monkeypatch, tmp_path)

    response = _upload("/etc/evil.md", b"escape")

    assert response.status_code == 200
    assert response.json()["filename"] == "evil.md"
    assert _all_files(target) == ["evil.md"]


def test_D06_absolute_windows_path_is_reduced(monkeypatch, tmp_path):
    """盘符绝对路径：``parse_options_header`` 的盘符特判是第一道，
    ``Path(original).name`` 是第二道，二者都要在。"""
    target = _reset_dir(monkeypatch, tmp_path)

    response = _upload("C:\\Windows\\evil.md", b"escape")

    assert response.status_code == 200
    assert response.json()["filename"] == "evil.md"
    assert (target / "evil.md").exists()
    assert not Path("C:\\Windows\\evil.md").exists()


def test_D07_dotdot_filename_rejected(monkeypatch, tmp_path):
    """``..`` 经 ``Path.name`` 后仍是 ``..``，由校验函数拒绝。"""
    target = _reset_dir(monkeypatch, tmp_path)

    response = _upload("..", b"escape")

    assert response.status_code == 400
    assert response.json()["detail"]["type"] == "invalid_filename"
    assert _all_files(target) == []


def test_D08_dot_filename_rejected(monkeypatch, tmp_path):
    """``.`` 经 ``Path.name`` 后是空串，由校验函数拒绝。"""
    target = _reset_dir(monkeypatch, tmp_path)

    response = _upload(".", b"escape")

    assert response.status_code == 400
    assert response.json()["detail"]["type"] == "invalid_filename"
    assert _all_files(target) == []


def test_D09_trailing_space_rejected(monkeypatch, tmp_path):
    """固定预期：``"x.md "`` → 400 invalid_filename（**不是** 200）。

    依据（只读核对 python_multipart ``parse_options_header``）：
    引号内的尾随空格**不会被** ``strip()`` 剥离（引号是值的最后一个字符），
    实测解析结果就是 ``b'x.md '``。进入路由后 ``Path("x.md ").name`` 仍带
    空格，``validate_knowledge_filename`` 的 ``rstrip(" .")`` 规则命中 →
    invalid_filename。此处的顺序也说明：文件名闸门先于后缀闸门
    （否则会得到 ``unsupported_format``，因为 ``.md `` 不在白名单）。
    """
    target = _reset_dir(monkeypatch, tmp_path)

    response = _upload("x.md ", b"content")

    assert response.status_code == 400
    assert response.json()["detail"]["type"] == "invalid_filename"
    assert _all_files(target) == []


def test_D10_double_dot_prefix_stays_inside(monkeypatch, tmp_path):
    """``..hidden.md`` 是普通文件名，不是穿越。"""
    target = _reset_dir(monkeypatch, tmp_path)

    response = _upload("..hidden.md", b"content")

    assert response.status_code == 200
    assert response.json()["filename"] == "..hidden.md"
    assert _all_files(target) == ["..hidden.md"]


def test_D11_no_file_escapes_knowledge_dir(monkeypatch, tmp_path):
    """组合越界检查：五种穿越形态全部落在 KNOWLEDGE_DIR 内。

    与 D02–D06 的区别：这里检查的是**整棵目录树**（``tmp_path`` 下除
    ``kb/`` 外不得出现任何新文件），而不只是某一个候选路径。
    """
    target = _reset_dir(monkeypatch, tmp_path)
    names = [
        "../../x1.md",
        "..\\x2.md",
        "C:\\Windows\\x3.md",
        "/etc/x4.md",
        "a/b/../../../x5.md",
    ]

    for index, name in enumerate(names, start=1):
        # 内容各不相同，避免命中「内容去重」而不是路径逻辑
        response = _upload(name, ("payload-%d" % index).encode("utf-8"))
        assert response.status_code == 200
        assert response.json()["filename"] == "x%d.md" % index

    assert _all_files(target) == ["x1.md", "x2.md", "x3.md", "x4.md", "x5.md"]
    # tmp_path 下除 kb/ 之外不得有任何普通文件（用 Path.parts 比较，避免依赖分隔符）
    outside = [
        rel for rel in _all_files(tmp_path) if Path(rel).parts[0] != "kb"
    ]
    assert outside == []


# ============================================================
# E 组：存储层异常 → HTTP 映射（9）
# ============================================================
#
# E01 / E02 / E03 / E04 / E09 走**真实生产路径**抛出真实异常；
# E05 / E06 / E07 / E08 的对应异常在 import 路径上不可自然触发，
# 因此用替身 ``import_document`` 抛出**真实异常类**，只验证映射表本身。
# 注释已标明，避免把「替身注入」误读为端到端覆盖。


def test_E01_already_exists_maps_to_409(monkeypatch, tmp_path):
    target = _reset_dir(monkeypatch, tmp_path)
    assert _upload("dup.md", b"first").status_code == 200

    # 内容不同 → 不命中内容去重，只命中同名判定
    response = _upload("dup.md", b"second")

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["type"] == "already_exists"
    assert detail["filename"] == "dup.md"
    assert (target / "dup.md").read_bytes() == b"first"


def test_E02_duplicate_content_maps_to_409(monkeypatch, tmp_path):
    target = _reset_dir(monkeypatch, tmp_path)
    assert _upload("first.md", b"same body").status_code == 200

    response = _upload("second.md", b"same body")

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["type"] == "duplicate_content"
    assert detail["existing_filename"] == "first.md"
    assert _all_files(target) == ["first.md"]


def test_E03_untrusted_directory_maps_to_503(monkeypatch, tmp_path):
    _reset_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(
        knowledge_store, "inspect_knowledge_directory", lambda _path: "untrusted"
    )

    response = _upload("a.md", b"content")

    assert response.status_code == 503
    assert response.json()["detail"]["type"] == "knowledge_dir_untrusted"


def test_E04_invalid_directory_maps_to_500(monkeypatch, tmp_path):
    """目录位置是普通文件 → not_directory → knowledge_dir_invalid。"""
    blocked = tmp_path / "kb"
    blocked.write_text("not a directory", encoding="utf-8")
    monkeypatch.setattr(main, "KNOWLEDGE_DIR", str(blocked))

    response = _upload("a.md", b"content")

    assert response.status_code == 500
    assert response.json()["detail"]["type"] == "knowledge_dir_invalid"


def test_E05_concurrent_modification_maps_to_409(monkeypatch, tmp_path):
    """映射表用例（替身抛真实异常类）。"""
    _reset_dir(monkeypatch, tmp_path)

    def _boom(_directory, filename, _content):
        raise knowledge_store.ConcurrentModification("文件已被并发修改", filename=filename)

    monkeypatch.setattr(knowledge_store, "import_document", _boom)

    response = _upload("a.md", b"content")

    assert response.status_code == 409
    assert response.json()["detail"]["type"] == "concurrent_modification"


def test_E06_file_busy_maps_to_409(monkeypatch, tmp_path):
    """映射表用例（替身抛真实异常类）。"""
    _reset_dir(monkeypatch, tmp_path)

    def _boom(_directory, filename, _content):
        raise knowledge_store.FileBusy("文件被占用", filename=filename)

    monkeypatch.setattr(knowledge_store, "import_document", _boom)

    response = _upload("a.md", b"content")

    assert response.status_code == 409
    assert response.json()["detail"]["type"] == "file_busy"


def test_E07_reparse_point_maps_to_403(monkeypatch, tmp_path):
    """映射表用例（替身抛真实异常类）。"""
    _reset_dir(monkeypatch, tmp_path)

    def _boom(_directory, filename, _content):
        raise knowledge_store.ReparsePointRejected("目标是重解析点", filename=filename)

    monkeypatch.setattr(knowledge_store, "import_document", _boom)

    response = _upload("a.md", b"content")

    assert response.status_code == 403
    assert response.json()["detail"]["type"] == "reparse_point_rejected"


def test_E08_not_regular_file_maps_to_403(monkeypatch, tmp_path):
    """映射表用例（替身抛真实异常类）。"""
    _reset_dir(monkeypatch, tmp_path)

    def _boom(_directory, filename, _content):
        raise knowledge_store.NotRegularFile("目标不是普通文件", filename=filename)

    monkeypatch.setattr(knowledge_store, "import_document", _boom)

    response = _upload("a.md", b"content")

    assert response.status_code == 403
    assert response.json()["detail"]["type"] == "not_regular_file"


def test_E09_write_failed_maps_to_500(monkeypatch, tmp_path):
    """**真实** WriteFailed：让底层写入抛 OSError，由存储层自己转换。

    不使用「替身直接抛 WriteFailed」，避免只验证到异常类的字面量；
    这里断言的是 ``create_document`` 的 OSError → WriteFailed 转换是否生效。
    """
    target = _reset_dir(monkeypatch, tmp_path)

    def _boom_write(_fd, _content):
        raise OSError("disk full")

    monkeypatch.setattr(knowledge_store, "_write_all", _boom_write)

    response = _upload("a.md", b"content")

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert detail["type"] == "write_failed"
    assert detail["reason"] == "OSError"
    assert detail["filename"] == "a.md"
    # 回滚成功时不得留下残留文件
    assert _all_files(target) == []


# ============================================================
# F 组：大小 / 空内容 / 文件关闭（6）
# ============================================================


def test_F01_oversized_file_rejected_with_413(monkeypatch, tmp_path):
    """用真实上限验证常量本身生效（读入上限 + 1 字节即可判定）。"""
    target = _reset_dir(monkeypatch, tmp_path)
    payload = b"a" * (main.MAX_KNOWLEDGE_FILE_BYTES + 1)

    response = _upload("huge.md", payload)

    assert response.status_code == 413
    assert response.json()["detail"]["type"] == "file_too_large"
    assert _all_files(target) == []


def test_F02_size_equal_to_limit_accepted(monkeypatch, tmp_path):
    """边界含等号：``len(content) == limit`` 必须通过（把上限调小以便测）。"""
    target = _reset_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(main, "MAX_KNOWLEDGE_FILE_BYTES", 16)
    payload = b"0123456789abcdef"

    response = _upload("edge.md", payload)

    assert len(payload) == main.MAX_KNOWLEDGE_FILE_BYTES
    assert response.status_code == 200
    assert (target / "edge.md").read_bytes() == payload


def test_F03_size_one_over_limit_rejected(monkeypatch, tmp_path):
    target = _reset_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(main, "MAX_KNOWLEDGE_FILE_BYTES", 16)

    response = _upload("edge.md", b"0123456789abcdefg")

    assert response.status_code == 413
    assert response.json()["detail"]["type"] == "file_too_large"
    assert _all_files(target) == []


def test_F04_empty_file_rejected(monkeypatch, tmp_path):
    target = _reset_dir(monkeypatch, tmp_path)

    response = _upload("empty.md", b"")

    assert response.status_code == 400
    assert response.json()["detail"]["type"] == "empty_file"
    assert _all_files(target) == []


def test_F05_blank_content_rejected(monkeypatch, tmp_path):
    """只含空白的文件按「空」处理（``content.strip()`` 判据）。"""
    target = _reset_dir(monkeypatch, tmp_path)

    response = _upload("blank.md", b"  \r\n\t \x0b\x0c")

    assert response.status_code == 400
    assert response.json()["detail"]["type"] == "empty_file"
    assert _all_files(target) == []


def test_F06_close_failure_does_not_mask_original_error(monkeypatch, tmp_path):
    """契约 7：关闭失败时，若已有原始异常，则**保留原始异常**。

    这是 F 组真正有判别力的用例：若端点的 ``finally`` 写成
    ``await file.close()``（不判断 ``sys.exc_info()``），或把 close 提到
    try 之外，客户端会看到 500 而不是 400。

    替身只在第一次调用时抛错（端点 ``finally``），后续调用转回原实现，
    以免干扰框架自身的清理（R2：单纯「close 被调用」的断言不具判别力）。
    """
    target = _reset_dir(monkeypatch, tmp_path)
    import starlette.datastructures as datastructures

    original_close = datastructures.UploadFile.close
    state = {"calls": 0}

    async def _close(self):
        state["calls"] += 1
        if state["calls"] == 1:
            raise RuntimeError("close 失败（测试注入）")
        return await original_close(self)

    monkeypatch.setattr(datastructures.UploadFile, "close", _close)

    response = _upload("empty.md", b"")

    assert state["calls"] >= 1
    assert response.status_code == 400
    assert response.json()["detail"]["type"] == "empty_file"
    assert _all_files(target) == []


# ============================================================
# G 组：缓存失效与副作用边界（4）
# ============================================================


def test_G01_success_invalidates_caches_in_order(monkeypatch, tmp_path):
    """契约 6：成功后才清理，顺序固定为 知识库 → Agent。"""
    _reset_dir(monkeypatch, tmp_path)
    order = []
    monkeypatch.setattr(main, "get_knowledge_base", _SpyCache("kb", order))
    monkeypatch.setattr(main, "get_agent", _SpyCache("agent", order))

    assert _upload("a.md", b"content").status_code == 200

    assert order == ["kb", "agent"]


def test_G02_format_rejection_keeps_caches(monkeypatch, tmp_path):
    _reset_dir(monkeypatch, tmp_path)
    order = []
    monkeypatch.setattr(main, "get_knowledge_base", _SpyCache("kb", order))
    monkeypatch.setattr(main, "get_agent", _SpyCache("agent", order))

    assert _upload("a.py", b"content").status_code == 400

    assert order == []


def test_G03_store_failure_keeps_caches(monkeypatch, tmp_path):
    _reset_dir(monkeypatch, tmp_path)
    order = []
    monkeypatch.setattr(main, "get_knowledge_base", _SpyCache("kb", order))
    monkeypatch.setattr(main, "get_agent", _SpyCache("agent", order))
    assert _upload("dup.md", b"first").status_code == 200
    order.clear()

    assert _upload("dup.md", b"second").status_code == 409

    assert order == []


def test_G04_upload_never_constructs_kb_or_agent(monkeypatch, tmp_path):
    """``_invalidate_knowledge_caches`` 只调 ``cache_clear()``，不构造对象。

    同时锁定副作用边界：上传不建索引，目录内只有上传的那个文件。
    """
    target = _reset_dir(monkeypatch, tmp_path)
    sink = []
    kb_spy = _SpyCache("kb", sink)
    agent_spy = _SpyCache("agent", sink)
    monkeypatch.setattr(main, "get_knowledge_base", kb_spy)
    monkeypatch.setattr(main, "get_agent", agent_spy)

    assert _upload("a.md", b"content").status_code == 200

    assert kb_spy.calls == 0
    assert agent_spy.calls == 0
    assert sink == ["kb", "agent"]
    assert _all_files(target) == ["a.md"]


# ============================================================
# I 组：编码与跨接口生命周期补齐（4，P2-2B-1-R 增补）
# ============================================================
#
# 只补 P2-2B-1 交付报告中「未明确覆盖」的四项，不重复已有断言：
#
# - I01 非 UTF-8 的 **.txt**：A06 用 ``.pdf`` 后缀，只能证明「二进制容器
#       不解码」；``.txt`` 是历史上被专门施加 UTF-8 校验的格式，必须单独锁定。
# - I02 CRLF + NUL：此前全部样本都是 LF 且无 NUL 字节，换行归一与截断
#       都无从暴露。
# - I03 新旧接口的 UTF-8 差分契约：旧 ``POST /knowledge/import`` 至今仍做
#       ``content.decode("utf-8")``，新接口上线不得顺带改动它。
# - I04 上传 → GET → DELETE 的身份闭环：此前只借 D03 顺带读过 GET，
#       DELETE 完全未覆盖。
#
# 四项均沿用 A/B/E 组的隔离方式：``main.KNOWLEDGE_DIR`` 指向 ``tmp_path``
# 子目录，不触碰仓库内的 ``rag_knowledge/`` 与任何数据库。


def test_I01_non_utf8_txt_upload_preserves_bytes(monkeypatch, tmp_path):
    """非 UTF-8 的 .txt 必须能上传，且落盘字节与上传字节完全一致。

    样本用 GBK 编码的中文：先断言它确实不能被 UTF-8 解码（前置条件，保证
    本用例有判别力），再断言接口 200 且落盘逐字节一致。若上传层对 ``.txt``
    恢复 UTF-8 校验（旧接口的做法），本用例会退化成 400 invalid_encoding。
    """
    target = _reset_dir(monkeypatch, tmp_path)
    payload = "中文内容".encode("gbk")

    with pytest.raises(UnicodeDecodeError):
        payload.decode("utf-8")

    response = _upload("gbk.txt", payload)

    assert response.status_code == 200
    assert response.json()["bytes"] == len(payload)
    assert (target / "gbk.txt").read_bytes() == payload


def test_I02_crlf_and_nul_bytes_are_not_normalized(monkeypatch, tmp_path):
    """CRLF 与 NUL 字节必须原样保留：不解码、不换行归一、不按 NUL 截断。

    判别方式：先传 CRLF+NUL 版本，再传 LF+NUL 版本。两者原始字节不同，
    因此都不应命中内容去重；两份文件的落盘字节必须各自与上传字节一致，
    且**互不相等**。一旦上传层做过 ``decode`` 或 ``splitlines`` 之类处理，
    后一次上传会变成 409 duplicate_content，或落盘字节被改写。
    """
    target = _reset_dir(monkeypatch, tmp_path)
    crlf = b"a\r\nb\x00c\r\n"
    lf = b"a\nb\x00c\n"

    first = _upload("crlf.txt", crlf)
    second = _upload("lf.txt", lf)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["bytes"] == len(crlf)
    assert second.json()["bytes"] == len(lf)
    assert (target / "crlf.txt").read_bytes() == crlf
    assert (target / "lf.txt").read_bytes() == lf
    # 换行归一一发生，两份就会相等；此处显式锁定差异（含 NUL）
    assert (target / "crlf.txt").read_bytes() != (target / "lf.txt").read_bytes()
    assert b"\x00" in (target / "crlf.txt").read_bytes()


def test_I03_legacy_import_still_rejects_invalid_utf8(monkeypatch, tmp_path):
    """差分契约：同一份非 UTF-8 内容，旧接口拒绝、新接口接受。

    锁定的是「两者并存时行为仍分叉」：旧 ``POST /knowledge/import`` 在
    ``main.py`` 中以 ``content.decode("utf-8")`` 拦下非 UTF-8 内容并返回
    ``invalid_encoding``，新接口按契约不解码。同目录
    ``test_knowledge_documents_api.py::test_A38_import_keeps_legacy_error_types``
    只孤立验证旧接口，未验证这一对新老分叉，故此处不重复而是互补。
    """
    target = _reset_dir(monkeypatch, tmp_path)
    payload = b"\xff\xfe\x00garbage"

    legacy = client.post(
        "/knowledge/import",
        files={"file": ("legacy.md", payload, "text/markdown")},
    )

    assert legacy.status_code == 400
    assert legacy.json()["detail"]["type"] == "invalid_encoding"
    # 旧接口的失败路径不落盘
    assert _all_files(target) == []

    modern = _upload("modern.md", payload)

    assert modern.status_code == 200
    assert modern.json()["bytes"] == len(payload)
    assert _all_files(target) == ["modern.md"]
    assert (target / "modern.md").read_bytes() == payload


def test_I04_uploaded_document_is_listed_and_deletable(monkeypatch, tmp_path):
    """生命周期闭环：上传 → GET 可见 → DELETE 可删 → 再 GET 为空。

    锁定「上传返回的 document_id 就是管理接口使用的那把钥匙」，而不只是
    响应字段格式正确：GET 列表里的 ``document_id`` 必须与上传返回值一致，
    DELETE 也用它且报告 ``freed_bytes`` 等于实际字节数。
    """
    target = _reset_dir(monkeypatch, tmp_path)
    payload = _build_pdf(["lifecycle page"])

    created = _upload("报告.pdf", payload)

    assert created.status_code == 200
    document_id = created.json()["document_id"]
    assert document_id == _doc_id("报告.pdf")

    listing = client.get(_ENDPOINT).json()
    assert listing["count"] == 1
    assert listing["documents"][0]["filename"] == "报告.pdf"
    assert listing["documents"][0]["document_id"] == document_id
    assert listing["documents"][0]["file_size"] == len(payload)

    deleted = client.delete(_ENDPOINT + "/" + document_id)

    assert deleted.status_code == 200
    body = deleted.json()
    assert body["ok"] is True
    assert body["document_id"] == document_id
    assert body["filename"] == "报告.pdf"
    assert body["freed_bytes"] == len(payload)
    assert not (target / "报告.pdf").exists()

    after = client.get(_ENDPOINT).json()
    assert after["count"] == 0
    assert after["documents"] == []
