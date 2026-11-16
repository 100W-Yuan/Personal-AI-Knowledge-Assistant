# -*- coding: utf-8 -*-
"""P0-2 批次 A：PdfLoader 测试。

样本全部在本文件内**手工构造**（项目无 reportlab/fpdf，且本批次禁止安装依赖），
用 Helvetica 内建字体写入 ASCII 文本，确保 pypdf 的 extract_text() 能取到文本层。

覆盖任务书要求的 PDF 场景：
正常多页 / 单页 / 部分空白页 / 全部空白（无文本层）/ 加密 / 损坏 / 不存在 /
页码 1 基 / page_count 为总页数 / 页面级失败不拖垮整文件。
"""

import pytest

from rag.document import Document
from rag.loader import (
    CorruptPdfError,
    EncryptedPdfError,
    NoTextLayerError,
    PdfLoadError,
    PdfLoader,
    PdfPageExtractionError,
)


# ------------------------------------------------------------
# 最小 PDF 构造器
# ------------------------------------------------------------
def _escape_pdf_text(text):
    """转义 PDF 字符串字面量中的 \\ ( ) 三个特殊字符。"""
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def build_pdf(page_texts):
    """构造一个带文本层的最小 PDF。

    ``page_texts`` 为空串的页会被写成空内容流 —— 这正是「无文本层」页面的
    真实形态，pypdf 对其 extract_text() 返回空串。
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

        if text:
            stream = (
                "BT /F1 12 Tf 72 720 Td (%s) Tj ET" % _escape_pdf_text(text)
            ).encode("latin-1")
        else:
            stream = b""

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


def _write(path, data):
    path.write_bytes(data)
    return str(path)


# ------------------------------------------------------------
# 正常路径：多页 / 单页
# ------------------------------------------------------------
def test_multi_page_yields_one_document_per_page(tmp_path):
    path = _write(tmp_path / "multi.pdf", build_pdf(["Page one", "Page two", "Page three"]))
    documents = PdfLoader(path).load()

    assert len(documents) == 3
    assert all(isinstance(item, Document) for item in documents)
    assert [item.metadata["page"] for item in documents] == [1, 2, 3]
    assert [item.metadata["page_count"] for item in documents] == [3, 3, 3]
    assert documents[0].page_content.strip() == "Page one"
    assert documents[2].page_content.strip() == "Page three"


def test_single_page(tmp_path):
    path = _write(tmp_path / "single.pdf", build_pdf(["Only page"]))
    documents = PdfLoader(path).load()

    assert len(documents) == 1
    assert documents[0].metadata["page"] == 1
    assert documents[0].metadata["page_count"] == 1


def test_source_is_file_path(tmp_path):
    path = _write(tmp_path / "src.pdf", build_pdf(["Hello"]))
    documents = PdfLoader(path).load()
    assert documents[0].metadata["source"] == path


def test_page_numbering_is_one_based(tmp_path):
    """页码从 1 开始，绝不出现 0。"""
    path = _write(tmp_path / "onebased.pdf", build_pdf(["A", "B"]))
    pages = [item.metadata["page"] for item in PdfLoader(path).load()]
    assert min(pages) == 1
    assert 0 not in pages


# ------------------------------------------------------------
# 部分空白页：跳过但不失败
# ------------------------------------------------------------
def test_partial_blank_pages_are_skipped_not_fatal(tmp_path):
    """中间页空白时保留其余页，且不因少量文本判定为扫描件。"""
    path = _write(
        tmp_path / "partial.pdf",
        build_pdf(["First page text", "", "Third page text"]),
    )
    documents = PdfLoader(path).load()

    assert len(documents) == 2
    assert [item.metadata["page"] for item in documents] == [1, 3]
    # page_count 仍是总页数，不是成功页数。
    assert {item.metadata["page_count"] for item in documents} == {3}


def test_leading_blank_page_is_skipped(tmp_path):
    path = _write(tmp_path / "lead.pdf", build_pdf(["", "Body"]))
    documents = PdfLoader(path).load()
    assert [item.metadata["page"] for item in documents] == [2]
    assert documents[0].metadata["page_count"] == 2


def test_trailing_blank_page_is_skipped(tmp_path):
    path = _write(tmp_path / "tail.pdf", build_pdf(["Body", ""]))
    documents = PdfLoader(path).load()
    assert [item.metadata["page"] for item in documents] == [1]


# ------------------------------------------------------------
# 全部无文本层
# ------------------------------------------------------------
def test_all_blank_pages_raise_no_text_layer(tmp_path):
    path = _write(tmp_path / "scanned.pdf", build_pdf(["", "", ""]))
    with pytest.raises(NoTextLayerError):
        PdfLoader(path).load()


def test_no_text_layer_message_mentions_page_count_and_path(tmp_path):
    path = _write(tmp_path / "scanned2.pdf", build_pdf(["", ""]))
    with pytest.raises(NoTextLayerError) as excinfo:
        PdfLoader(path).load()
    message = str(excinfo.value)
    assert "2" in message
    assert "scanned2.pdf" in message


def test_no_text_layer_is_pdf_load_error(tmp_path):
    path = _write(tmp_path / "scanned3.pdf", build_pdf([""]))
    with pytest.raises(PdfLoadError):
        PdfLoader(path).load()


# ------------------------------------------------------------
# 加密
# ------------------------------------------------------------
def test_encrypted_pdf_raises_encrypted_error(tmp_path):
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.encrypt("secret-password", algorithm="RC4-128")

    path = tmp_path / "locked.pdf"
    with open(str(path), "wb") as handle:
        writer.write(handle)

    with pytest.raises(EncryptedPdfError):
        PdfLoader(str(path)).load()


def test_encrypted_error_is_not_confused_with_corrupt(tmp_path):
    """加密与损坏必须可区分（任务书决策 8）。"""
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.encrypt("pw", algorithm="RC4-128")
    path = tmp_path / "locked2.pdf"
    with open(str(path), "wb") as handle:
        writer.write(handle)

    with pytest.raises(EncryptedPdfError) as excinfo:
        PdfLoader(str(path)).load()
    assert not isinstance(excinfo.value, CorruptPdfError)


# ------------------------------------------------------------
# 损坏 / 空文件 / 不存在
# ------------------------------------------------------------
def test_garbage_bytes_raise_corrupt(tmp_path):
    path = _write(tmp_path / "broken.pdf", b"this is definitely not a pdf")
    with pytest.raises(CorruptPdfError):
        PdfLoader(path).load()


def test_empty_file_raises_corrupt(tmp_path):
    path = _write(tmp_path / "empty.pdf", b"")
    with pytest.raises(CorruptPdfError):
        PdfLoader(path).load()


def test_truncated_pdf_raises_corrupt(tmp_path):
    data = build_pdf(["Some text"])[:40]
    path = _write(tmp_path / "truncated.pdf", data)
    with pytest.raises(CorruptPdfError):
        PdfLoader(path).load()


def test_missing_file_propagates_file_not_found(tmp_path):
    """文件不存在应透传 FileNotFoundError，与 TextLoader 的 open() 一致。"""
    with pytest.raises(FileNotFoundError):
        PdfLoader(str(tmp_path / "nope.pdf")).load()


def test_corrupt_error_keeps_exception_chain(tmp_path):
    path = _write(tmp_path / "broken2.pdf", b"not a pdf")
    with pytest.raises(CorruptPdfError) as excinfo:
        PdfLoader(path).load()
    assert excinfo.value.__cause__ is not None


# ------------------------------------------------------------
# metadata 形状
# ------------------------------------------------------------
def test_metadata_keys_are_exactly_source_page_page_count(tmp_path):
    path = _write(tmp_path / "keys.pdf", build_pdf(["X"]))
    metadata = PdfLoader(path).load()[0].metadata
    assert set(metadata) == {"source", "page", "page_count"}


def test_page_and_page_count_are_ints(tmp_path):
    path = _write(tmp_path / "ints.pdf", build_pdf(["X", "Y"]))
    for document in PdfLoader(path).load():
        assert isinstance(document.metadata["page"], int)
        assert isinstance(document.metadata["page_count"], int)
        assert not isinstance(document.metadata["page"], bool)


# ------------------------------------------------------------
# 页级提取异常（A-R1 / B1）
#
# 合法 PDF 无法构造出「单页 extract_text() 抛异常而其它页正常」的样本，
# 因此这一组用假页面对象注入，不依赖损坏的真实 PDF。
# ------------------------------------------------------------
class _FakePage:
    """可控页面：要么返回固定文本，要么在 extract_text() 时抛异常。"""

    def __init__(self, text=None, error=None):
        self._text = text
        self._error = error

    def extract_text(self):
        if self._error is not None:
            raise self._error
        return self._text


def _install_fake_reader(monkeypatch, pages):
    """替换 pypdf.PdfReader，使 PdfLoader 读到指定的假页面序列。"""
    import pypdf

    class _FakeReader:
        def __init__(self, path, strict=False):
            self.pages = pages
            self.is_encrypted = False

    monkeypatch.setattr(pypdf, "PdfReader", _FakeReader)


def test_all_pages_extraction_failure_raises_page_extraction_error(
    tmp_path, monkeypatch
):
    """规则 2：全部页提取均异常 → 必须是 PdfPageExtractionError。"""
    pages = [
        _FakePage(error=RuntimeError("page one boom")),
        _FakePage(error=ValueError("page two boom")),
    ]
    _install_fake_reader(monkeypatch, pages)
    path = _write(tmp_path / "allfail.pdf", b"placeholder")

    with pytest.raises(PdfPageExtractionError):
        PdfLoader(path).load()


def test_all_pages_extraction_failure_is_not_mistaken_for_scan(
    tmp_path, monkeypatch
):
    """解析故障绝不能被报成「疑似扫描件」。"""
    pages = [_FakePage(error=RuntimeError("boom")) for _ in range(3)]
    _install_fake_reader(monkeypatch, pages)
    path = _write(tmp_path / "allfail2.pdf", b"placeholder")

    with pytest.raises(PdfPageExtractionError) as excinfo:
        PdfLoader(path).load()

    assert not isinstance(excinfo.value, NoTextLayerError)
    # 两者同属 PdfLoadError，但必须互不继承。
    assert not issubclass(PdfPageExtractionError, NoTextLayerError)
    assert not issubclass(NoTextLayerError, PdfPageExtractionError)
    assert isinstance(excinfo.value, PdfLoadError)


def test_page_extraction_error_message_carries_page_and_cause(
    tmp_path, monkeypatch
):
    original = OSError("broken stream")
    pages = [_FakePage(error=original)]
    _install_fake_reader(monkeypatch, pages)
    path = _write(tmp_path / "allfail3.pdf", b"placeholder")

    with pytest.raises(PdfPageExtractionError) as excinfo:
        PdfLoader(path).load()

    message = str(excinfo.value)
    # 明确断言「页码」而非任意数字：单页 PDF 的总页数同样是 1，
    # 若只断言 "1" in message，无论页码正确与否都会通过。
    assert "第 1 页" in message
    assert "broken stream" in message
    assert "OSError" in message
    assert "allfail3.pdf" in message
    # 原始异常链必须指向真正的异常对象（而不是字符串副本）
    assert excinfo.value.__cause__ is original


def test_partial_pages_failure_keeps_successful_pages(tmp_path, monkeypatch):
    """规则 3：部分异常 + 部分有文本 → 保留成功页。"""
    pages = [
        _FakePage(text="Page one text"),
        _FakePage(error=RuntimeError("cannot extract")),
        _FakePage(text="Page three text"),
    ]
    _install_fake_reader(monkeypatch, pages)
    path = _write(tmp_path / "partial.pdf", b"placeholder")

    documents = PdfLoader(path).load()

    assert len(documents) == 2
    assert [item.metadata["page"] for item in documents] == [1, 3]
    assert documents[0].page_content.strip() == "Page one text"
    assert documents[1].page_content.strip() == "Page three text"


def test_partial_failure_does_not_change_page_count(tmp_path, monkeypatch):
    """页码与总页数不受异常页影响。"""
    pages = [
        _FakePage(text="A"),
        _FakePage(error=RuntimeError("x")),
        _FakePage(text="C"),
        _FakePage(error=RuntimeError("y")),
    ]
    _install_fake_reader(monkeypatch, pages)
    path = _write(tmp_path / "count.pdf", b"placeholder")

    documents = PdfLoader(path).load()

    assert [item.metadata["page"] for item in documents] == [1, 3]
    assert {item.metadata["page_count"] for item in documents} == {4}


def test_empty_and_failed_pages_do_not_hide_failure(tmp_path, monkeypatch):
    """规则 4：部分空 + 部分异常且无有效文本 → 按无文本处理但不隐瞒异常。"""
    pages = [
        _FakePage(text=""),
        _FakePage(error=RuntimeError("boom")),
        _FakePage(text="   \n  "),
    ]
    _install_fake_reader(monkeypatch, pages)
    path = _write(tmp_path / "mixed.pdf", b"placeholder")

    with pytest.raises(NoTextLayerError) as excinfo:
        PdfLoader(path).load()

    message = str(excinfo.value)
    # 必须披露异常页数量与页码，不能只说「无可提取文本」。
    assert "提取异常 1 页" in message
    assert "页码 2" in message
    assert "空页 2 页" in message


def test_rule4_is_not_reported_as_page_extraction_error(tmp_path, monkeypatch):
    """规则 4 不算「全部页失败」，因此不抛 PdfPageExtractionError。"""
    pages = [
        _FakePage(text=""),
        _FakePage(error=RuntimeError("boom")),
    ]
    _install_fake_reader(monkeypatch, pages)
    path = _write(tmp_path / "mixed2.pdf", b"placeholder")

    with pytest.raises(NoTextLayerError) as excinfo:
        PdfLoader(path).load()
    assert not isinstance(excinfo.value, PdfPageExtractionError)


def test_all_success_but_empty_still_raises_no_text_layer(tmp_path, monkeypatch):
    """规则 1 不受影响：全部提取成功且均为空 → NoTextLayerError。"""
    pages = [_FakePage(text=""), _FakePage(text="")]
    _install_fake_reader(monkeypatch, pages)
    path = _write(tmp_path / "empty2.pdf", b"placeholder")

    with pytest.raises(NoTextLayerError) as excinfo:
        PdfLoader(path).load()
    # 规则 1 的消息不应出现「提取异常」字样
    assert "提取异常" not in str(excinfo.value)


# ------------------------------------------------------------
# A-R2 / N1：结构化页面诊断属性
# ------------------------------------------------------------
def test_rule1_pure_blank_exposes_structured_diagnostics(tmp_path, monkeypatch):
    """纯空白 PDF：empty_pages 准确，failed_pages 必须为空列表。"""
    pages = [
        _FakePage(text=""),
        _FakePage(text="   "),
        _FakePage(text="\n\t"),
    ]
    _install_fake_reader(monkeypatch, pages)
    path = _write(tmp_path / "pureblank.pdf", b"placeholder")

    with pytest.raises(NoTextLayerError) as excinfo:
        PdfLoader(path).load()

    error = excinfo.value
    assert error.empty_pages == [1, 2, 3]
    assert error.failed_pages == []
    assert isinstance(error.empty_pages, list)
    assert isinstance(error.failed_pages, list)


def test_rule4_mixed_failure_exposes_structured_diagnostics(
    tmp_path, monkeypatch
):
    """混合故障：empty_pages 与 failed_pages 各自准确，无需解析消息。"""
    pages = [
        _FakePage(text=""),
        _FakePage(error=RuntimeError("boom-2")),
        _FakePage(text="  "),
        _FakePage(error=ValueError("boom-4")),
    ]
    _install_fake_reader(monkeypatch, pages)
    path = _write(tmp_path / "mixed3.pdf", b"placeholder")

    with pytest.raises(NoTextLayerError) as excinfo:
        PdfLoader(path).load()

    error = excinfo.value
    assert error.empty_pages == [1, 3]
    assert [item["page"] for item in error.failed_pages] == [2, 4]
    assert [item["error_type"] for item in error.failed_pages] == [
        "RuntimeError",
        "ValueError",
    ]
    assert [item["message"] for item in error.failed_pages] == [
        "boom-2",
        "boom-4",
    ]
    # 每个诊断项固定三键，便于机器消费
    for item in error.failed_pages:
        assert set(item) == {"page", "error_type", "message"}


def test_pure_blank_and_mixed_distinguishable_by_attributes(
    tmp_path, monkeypatch
):
    """仅凭属性即可区分纯空白与混合故障，异常类型相同也不影响判断。"""
    _install_fake_reader(monkeypatch, [_FakePage(text=""), _FakePage(text="")])
    with pytest.raises(NoTextLayerError) as pure:
        PdfLoader(_write(tmp_path / "pure2.pdf", b"placeholder")).load()

    _install_fake_reader(
        monkeypatch,
        [_FakePage(text=""), _FakePage(error=RuntimeError("x"))],
    )
    with pytest.raises(NoTextLayerError) as mixed:
        PdfLoader(_write(tmp_path / "mix2.pdf", b"placeholder")).load()

    assert type(pure.value) is type(mixed.value)   # 同一个异常类
    assert pure.value.failed_pages == []
    assert mixed.value.failed_pages != []
    assert pure.value.empty_pages == [1, 2]
    assert mixed.value.empty_pages == [1]


# ------------------------------------------------------------
# A-R2 / N2：原始异常链
# ------------------------------------------------------------
def test_all_failure_preserves_cause_and_all_diagnostics(
    tmp_path, monkeypatch
):
    """全页异常：__cause__ 是首个原始异常对象，且保留全部页诊断。"""
    first = RuntimeError("page one boom")
    second = ValueError("page two boom")
    pages = [_FakePage(error=first), _FakePage(error=second)]
    _install_fake_reader(monkeypatch, pages)
    path = _write(tmp_path / "allfail4.pdf", b"placeholder")

    with pytest.raises(PdfPageExtractionError) as excinfo:
        PdfLoader(path).load()

    error = excinfo.value
    # 异常链：同一对象，不是字符串副本
    assert error.__cause__ is first
    assert type(error.__cause__) is RuntimeError
    assert str(error.__cause__) == "page one boom"
    # 多页诊断逐项核验
    assert [item["page"] for item in error.failed_pages] == [1, 2]
    assert [item["error_type"] for item in error.failed_pages] == [
        "RuntimeError",
        "ValueError",
    ]
    assert [item["message"] for item in error.failed_pages] == [
        "page one boom",
        "page two boom",
    ]
    assert error.empty_pages == []

    # N8：多页全失败时，消息里的「首例页码」必须锚定第 1 页。
    # 单页样本里 page_count 同样是 1，无法区分 first_page 与 page_count，
    # 因此这条断言只能放在多页用例里才有判别力。
    message = str(error)
    assert "第 1 页" in message
    assert "第 2 页" not in message


def test_all_failure_cause_is_not_a_string_copy(tmp_path, monkeypatch):
    """确认 __cause__ 是异常实例，可用 isinstance 判定原始类型。"""
    original = KeyError("missing-key")
    _install_fake_reader(monkeypatch, [_FakePage(error=original)])
    path = _write(tmp_path / "allfail5.pdf", b"placeholder")

    with pytest.raises(PdfPageExtractionError) as excinfo:
        PdfLoader(path).load()

    cause = excinfo.value.__cause__
    assert isinstance(cause, KeyError)
    assert isinstance(cause, Exception)
    assert not isinstance(cause, str)


def test_mixed_failure_keeps_diagnostics_without_cause(
    tmp_path, monkeypatch
):
    """规则 4 归类依据是「无有效文本」，不建 __cause__，但诊断必须完整。"""
    pages = [_FakePage(text=""), _FakePage(error=RuntimeError("boom"))]
    _install_fake_reader(monkeypatch, pages)
    path = _write(tmp_path / "mixed4.pdf", b"placeholder")

    with pytest.raises(NoTextLayerError) as excinfo:
        PdfLoader(path).load()

    error = excinfo.value
    assert error.__cause__ is None
    assert error.empty_pages == [1]
    assert len(error.failed_pages) == 1
    assert error.failed_pages[0]["page"] == 2
    assert error.failed_pages[0]["error_type"] == "RuntimeError"


def test_encrypted_and_corrupt_errors_unchanged_by_diagnostics(
    tmp_path, monkeypatch
):
    """N1/N2 不得改变加密与损坏的既有分类语义。"""
    path = _write(tmp_path / "garbage2.pdf", b"definitely not a pdf")
    with pytest.raises(CorruptPdfError) as excinfo:
        PdfLoader(path).load()
    assert not isinstance(excinfo.value, NoTextLayerError)
    assert not isinstance(excinfo.value, PdfPageExtractionError)


def test_page_and_page_count_survive_mixed_rule4(tmp_path, monkeypatch):
    """规则 4 被触发时，page_count 仍等于总页数（不因异常页缩水）。"""
    pages = [
        _FakePage(text=""),
        _FakePage(error=RuntimeError("x")),
        _FakePage(text=""),
    ]
    _install_fake_reader(monkeypatch, pages)
    path = _write(tmp_path / "rule4count.pdf", b"placeholder")

    with pytest.raises(NoTextLayerError) as excinfo:
        PdfLoader(path).load()

    assert "3 页" in str(excinfo.value)
    assert excinfo.value.empty_pages == [1, 3]
