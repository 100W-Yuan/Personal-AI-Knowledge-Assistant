# -*- coding: utf-8 -*-
"""P2-3B1：真实文档解析与 RAG 核心链路离线 E2E 验收。

本文件只验证**真实生产代码**的数据流与索引结构，不修改任何生产代码：

    Loader → TextSplitter → KnowledgeBase → Embedding 接口
          → VectorStore → Retriever → KnowledgeBase.search

边界声明（不得据此扩大结论）
------------------------------
1. 使用确定性 Stub Embedding（``_StubEmbedding``），**不导入** sentence_transformers /
   torch / transformers，不加载模型、不发起任何网络请求。
2. 因此本文件验证的是：解析、切块、编号、metadata 透传、检索链路、异常分类、
   索引结构。**不验证**：真实语义相似度、真实召回率、真实答案质量。
3. 所有夹具写在 ``tmp_path``；索引只存在于内存；不触碰正式 ``rag_knowledge/``。

夹具构造说明
------------
- PDF 不依赖任何 PDF 生成库（未安装 reportlab/fpdf），使用本文件内手写的
  最小 PDF 字节流：Type0 / Identity-H 字体 + ToUnicode CMap，使 pypdf 能把
  2 字节 CID 还原成中文原文。第 :func:`test_B01_pdf_fixture_is_readable_by_pypdf`
  先单独证明夹具本身可读，再让后续用例依赖它。
- DOCX 用已安装的 python-docx 生成，含两个标题、两段正文、一个 2x2 表格与
  一个 0 行空表格。
"""

import hashlib
import io
import math
import sys

import pytest

from rag.context_builder import ContextBuilder
from rag.document import Document
from rag.knowledge_base import KnowledgeBase
from rag.loader import (
    CorruptPdfError,
    DocxLoadError,
    NoTextLayerError,
    PdfPageExtractionError,
    TextLoader,
    get_loader,
)
from rag.splitter import TextSplitter
from rag.vector_store import VectorStore


# ============================================================
# F 组基础设施：确定性 Stub Embedding（完全离线）
# ============================================================
class _StubEmbedding:
    """确定性、离线、可计数的 Embedding 替身。

    向量构造：
    - ``vec[0]`` 固定 baseline 1.0，保证任意两段非空文本的余弦相似度恒 ``> 0``
      （生产 ``KnowledgeBase.search`` 在 ``threshold=None`` 时会过滤 ``score == 0.0``
      的结果，全零相似度会让"另一页的内容"直接从结果里消失）；
    - 其余维度由字符 bigram 的 sha256 稳定散列累加，使不同文本得到不同向量，
      从而可以断言"定向检索命中预期页"。

    返回值是普通的 ``list[float]``（生产 VectorStore 只用到 ``len`` / ``zip``）。
    """

    def __init__(self, dim=8):
        self.dim = dim
        self.calls = []

    def encode(self, text):
        self.calls.append(text)
        vec = [0.0] * self.dim
        vec[0] += 1.0  # baseline：保证余弦恒为正
        for index in range(len(text) - 1):
            digest = hashlib.sha256(text[index:index + 2].encode("utf-8")).digest()
            slot = 1 + digest[0] % (self.dim - 1)
            vec[slot] += 1.0
        if not any(vec):
            vec[0] = 1.0
        norm = math.sqrt(sum(value * value for value in vec))
        return [value / norm for value in vec]

    @property
    def chunk_calls(self):
        """索引期调用次数（构建索引后、检索前读取）。"""
        return len(self.calls)


# ============================================================
# 最小 PDF 构造器（不依赖第三方生成库）
# ============================================================
def _pdf_stream(data: bytes) -> bytes:
    return (
        b"<< /Length " + str(len(data)).encode("ascii") + b" >>\nstream\n"
        + data + b"\nendstream"
    )


def _build_pdf(pages):
    """构造最小多页 PDF。

    ``pages`` 每项：
    - ``str``   ：该页写入这段文字（中文经 Identity-H + ToUnicode 还原）；
    - ``None``  ：该页内容流为空（真正的空白页）；
    - ``bytes`` ：原样作为该页内容流（用于"有图形无文字"的无文本层场景）。
    """
    cid_map = {}
    for page in pages:
        if isinstance(page, str):
            for char in page:
                cid_map.setdefault(char, len(cid_map) + 1)

    if cid_map:
        bf_entries = "\n".join(
            "<%04X> <%s>" % (cid, char.encode("utf-16-be").hex().upper())
            for char, cid in cid_map.items()
        )
    else:
        # 无字符时也要给出合法的 bfchar 段（0 条会让部分解析器行为未定义）。
        bf_entries = "<0000> <0020>"

    tounicode = (
        "/CIDInit /ProcSet findresource begin\n"
        "12 dict begin\n"
        "begincmap\n"
        "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n"
        "/CMapName /Adobe-Identity-UCS def\n"
        "/CMapType 2 def\n"
        "1 begincodespacerange\n"
        "<0000> <FFFF>\n"
        "endcodespacerange\n"
        "%d beginbfchar\n"
        "%s\n"
        "endbfchar\n"
        "endcmap\n"
        "CMapName currentdict /CMap defineresource pop\n"
        "end\n"
        "end\n"
    ) % (max(len(cid_map), 1), bf_entries)

    objects = {}
    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objects[3] = (
        b"<< /Type /Font /Subtype /Type0 /BaseFont /SimSun /Encoding /Identity-H "
        b"/DescendantFonts [4 0 R] /ToUnicode 5 0 R >>"
    )
    objects[4] = (
        b"<< /Type /Font /Subtype /CIDFontType0 /BaseFont /SimSun "
        b"/CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> "
        b"/FontDescriptor 6 0 R /DW 1000 >>"
    )
    objects[5] = _pdf_stream(tounicode.encode("ascii"))
    objects[6] = (
        b"<< /Type /FontDescriptor /FontName /SimSun /Flags 4 "
        b"/FontBBox [0 0 1000 1000] /ItalicAngle 0 /Ascent 1000 /Descent 0 "
        b"/CapHeight 1000 /StemV 80 >>"
    )

    next_number = 7
    kids = []
    for page in pages:
        if isinstance(page, str):
            hex_text = "".join("%04X" % cid_map[char] for char in page)
            content = ("BT /F1 12 Tf 72 720 Td <%s> Tj ET" % hex_text).encode("ascii")
        elif isinstance(page, bytes):
            content = page
        else:
            content = b""

        content_number = next_number
        page_number = next_number + 1
        next_number += 2

        objects[content_number] = _pdf_stream(content)
        objects[page_number] = (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
            b"/Resources << /Font << /F1 3 0 R >> >> "
            b"/Contents %d 0 R >>" % content_number
        )
        kids.append(page_number)

    objects[2] = (
        b"<< /Type /Pages /Kids ["
        + b" ".join(b"%d 0 R" % number for number in kids)
        + b"] /Count %d >>" % len(kids)
    )

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = {}
    for number in sorted(objects):
        offsets[number] = len(out)
        out += b"%d 0 obj\n" % number + objects[number] + b"\nendobj\n"

    max_number = max(objects)
    xref_offset = len(out)
    out += b"xref\n0 %d\n" % (max_number + 1)
    out += b"0000000000 65535 f \n"
    for number in range(1, max_number + 1):
        out += b"%010d 00000 n \n" % offsets[number]
    out += (
        b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n"
        % (max_number + 1, xref_offset)
    )
    return bytes(out)


# ============================================================
# 夹具构造
# ============================================================
TXT_TEXT = "第一段：系统支持离线检索测试。\n\n第二段：索引构建不依赖网络。"
MD_TEXT = "# 测试标题\n\n这是 Markdown 正文，用于验证纯文本解析链路。"

PDF_PAGE_ONE = "苹果文档测试内容"
PDF_PAGE_THREE = "香蕉文档测试内容"

DOCX_SECTION_ONE = "第一章"
DOCX_PARAGRAPH_A = "正文段落A"
DOCX_PARAGRAPH_B = "正文段落B"
DOCX_SECTION_TWO = "第二章"
DOCX_PARAGRAPH_C = "正文段落C"


def _write_txt(path, text=TXT_TEXT):
    path.write_text(text, encoding="utf-8")
    return path


def _write_md(path, text=MD_TEXT):
    path.write_text(text, encoding="utf-8")
    return path


def _write_pdf(path, pages=(PDF_PAGE_ONE, None, PDF_PAGE_THREE)):
    path.write_bytes(_build_pdf(list(pages)))
    return path


def _write_docx(path):
    """按任务书 C 组结构生成：标题1 → 段落A → 2x2 表格 → 段落B → 空表 → 标题2 → 段落C。"""
    import docx

    document = docx.Document()
    document.add_heading(DOCX_SECTION_ONE, level=1)
    document.add_paragraph(DOCX_PARAGRAPH_A)

    table = document.add_table(rows=2, cols=2)
    for row in range(2):
        for col in range(2):
            table.cell(row, col).text = "R%dC%d" % (row, col)

    document.add_paragraph(DOCX_PARAGRAPH_B)
    document.add_table(rows=0, cols=0)
    document.add_heading(DOCX_SECTION_TWO, level=1)
    document.add_paragraph(DOCX_PARAGRAPH_C)
    document.save(str(path))
    return path


def _build_kb(directory, stub):
    return KnowledgeBase(str(directory), embedding_factory=lambda: stub)


def _search_all(kb, query, top_k=10):
    result = kb.search(query, top_k=top_k)
    assert result["ok"] is True, result
    return result


def _records_by_source(result):
    grouped = {}
    for record in result["results"]:
        grouped.setdefault(record["source"], []).append(record)
    return grouped


@pytest.fixture
def stub():
    return _StubEmbedding()


# ============================================================
# A 组：四格式核心链路
# ============================================================
def test_A01_four_formats_are_recognised_and_counted(tmp_path, stub):
    """四种格式全部被枚举、加载、切块；file_count/chunk_count 与实际一致。"""
    kbdir = tmp_path / "kb"
    kbdir.mkdir()
    _write_txt(kbdir / "notes.txt")
    _write_md(kbdir / "guide.md")
    _write_pdf(kbdir / "paper.pdf")
    _write_docx(kbdir / "manual.docx")

    kb = _build_kb(kbdir, stub)
    result = _search_all(kb, "测试")

    # 4 个文件全部成功加载：txt(1) + md(1) + pdf(2 页) + docx(4 个 Document) = 8 chunk
    assert kb.file_count == 4
    assert kb.chunk_count == 8
    assert kb.empty_file_count == 0
    assert kb.skipped_count == 0

    sources = {record["source"] for record in result["results"]}
    assert sources == {"notes.txt", "guide.md", "paper.pdf", "manual.docx"}

    # 每种格式的 loader 由 doc_formats 的后缀清单驱动，不在这里另写映射
    assert type(get_loader(str(kbdir / "notes.txt"))).__name__ == "TextLoader"
    assert type(get_loader(str(kbdir / "guide.md"))).__name__ == "TextLoader"
    assert type(get_loader(str(kbdir / "paper.pdf"))).__name__ == "PdfLoader"
    assert type(get_loader(str(kbdir / "manual.docx"))).__name__ == "DocxLoader"


def test_A02_document_id_matches_sha256_prefix_of_filename(tmp_path, stub):
    """document_id 必须等于 sha256(basename)[:32]，四种格式同算法。"""
    kbdir = tmp_path / "kb"
    kbdir.mkdir()
    names = ["notes.txt", "guide.md", "paper.pdf", "manual.docx"]
    _write_txt(kbdir / names[0])
    _write_md(kbdir / names[1])
    _write_pdf(kbdir / names[2])
    _write_docx(kbdir / names[3])

    kb = _build_kb(kbdir, stub)
    result = _search_all(kb, "测试")

    for name in names:
        expected = hashlib.sha256(name.encode("utf-8")).hexdigest()[:32]
        records = _records_by_source(result)[name]
        assert records, name
        assert all(record["document_id"] == expected for record in records)


def test_A03_chunk_seq_starts_at_zero_and_is_contiguous_per_file(tmp_path, stub):
    """chunk_seq 是文件级序号：每个文件从 0 开始、连续、不跨文件累加。"""
    kbdir = tmp_path / "kb"
    kbdir.mkdir()
    _write_txt(kbdir / "notes.txt")
    _write_md(kbdir / "guide.md")
    _write_pdf(kbdir / "paper.pdf")
    _write_docx(kbdir / "manual.docx")

    kb = _build_kb(kbdir, stub)
    result = _search_all(kb, "测试")
    grouped = _records_by_source(result)

    assert set(grouped) == {"notes.txt", "guide.md", "paper.pdf", "manual.docx"}
    for name, records in grouped.items():
        sequences = sorted(record["chunk_seq"] for record in records)
        assert sequences == list(range(len(sequences))), (name, sequences)

    # 总块数等于各文件块数之和，证明确实是"文件级"而非"全局"序号
    assert sum(len(records) for records in grouped.values()) == kb.chunk_count == 8


def test_A04_source_is_basename_after_indexing(tmp_path, stub):
    """Loader 写的是全路径，_stage_file 会覆盖为 basename。"""
    kbdir = tmp_path / "kb"
    kbdir.mkdir()
    pdf_path = _write_pdf(kbdir / "paper.pdf")

    # Loader 层证据：source 是完整路径
    raw_document = get_loader(str(pdf_path)).load()[0]
    assert raw_document.metadata["source"] == str(pdf_path)

    kb = _build_kb(kbdir, stub)
    result = _search_all(kb, "苹果")
    assert all(record["source"] == "paper.pdf" for record in result["results"])


def test_A05_search_results_carry_location_metadata(tmp_path, stub):
    """检索结果按格式携带定位字段：PDF 有页码，DOCX 有章节或表格编号。"""
    kbdir = tmp_path / "kb"
    kbdir.mkdir()
    _write_pdf(kbdir / "paper.pdf")
    _write_docx(kbdir / "manual.docx")

    kb = _build_kb(kbdir, stub)
    result = _search_all(kb, "测试")
    grouped = _records_by_source(result)

    for record in grouped["paper.pdf"]:
        assert record["page"] in (1, 3)
        assert record["page_count"] == 3

    docx_records = grouped["manual.docx"]
    section_records = [r for r in docx_records if "section" in r]
    table_records = [r for r in docx_records if "table_index" in r]
    assert section_records and table_records
    for record in section_records:
        assert record["section"] in (DOCX_SECTION_ONE, DOCX_SECTION_TWO)
        assert record["section_index"] in (0, 1)
        assert "table_index" not in record
    for record in table_records:
        assert record["table_index"] == 0
        assert "section" not in record


# ============================================================
# B 组：PDF 多页与页码定位
# ============================================================
def test_B01_pdf_fixture_is_readable_by_pypdf(tmp_path):
    """先证明手写 PDF 夹具本身可被 pypdf 读取，避免后续断言建立在坏夹具上。"""
    from pypdf import PdfReader

    pdf_path = _write_pdf(tmp_path / "fixture.pdf")
    reader = PdfReader(str(pdf_path))

    assert len(reader.pages) == 3
    assert reader.pages[0].extract_text().strip() == PDF_PAGE_ONE
    assert reader.pages[1].extract_text().strip() == ""
    assert reader.pages[2].extract_text().strip() == PDF_PAGE_THREE


def test_B02_pdf_page_count_and_blank_page_skipped(tmp_path):
    """page_count 是物理总页数（3），空白第 2 页不产出 Document。"""
    from rag.loader import PdfLoader

    pdf_path = _write_pdf(tmp_path / "three_pages.pdf")
    documents = PdfLoader(str(pdf_path)).load()

    assert len(documents) == 2  # 第 2 页为空，被跳过
    assert [d.metadata["page"] for d in documents] == [1, 3]
    assert all(d.metadata["page_count"] == 3 for d in documents)


def test_B03_page_numbers_not_renumbered_after_blank_page(tmp_path, stub):
    """空白页被跳过后，页码仍是原始物理页码，不重新编号。"""
    kbdir = tmp_path / "kb"
    kbdir.mkdir()
    _write_pdf(kbdir / "three_pages.pdf")

    kb = _build_kb(kbdir, stub)
    result = _search_all(kb, "文档测试")
    grouped = _records_by_source(result)
    pages = sorted(record["page"] for record in grouped["three_pages.pdf"])

    assert pages == [1, 3]          # 不是 [1, 2]
    assert 2 not in pages
    assert kb.chunk_count == 2


def test_B04_duplicate_chunk_id_across_pages_is_not_merged(tmp_path, stub):
    """两页的 chunk_id 都是 0，但 chunk_seq 不同，检索时两条都必须保留。"""
    kbdir = tmp_path / "kb"
    kbdir.mkdir()
    _write_pdf(kbdir / "three_pages.pdf")

    kb = _build_kb(kbdir, stub)
    result = _search_all(kb, "文档测试")
    records = _records_by_source(result)["three_pages.pdf"]

    assert len(records) == 2
    assert {record["chunk_id"] for record in records} == {0}   # 相同的 chunk_id
    assert {record["chunk_seq"] for record in records} == {0, 1}  # 不同的身份
    assert {record["page"] for record in records} == {1, 3}

    # 同一次检索里两条都在（没有被"身份去重"错误合并掉）
    assert len({record["document_id"] for record in records}) == 1

    # 定向检索：query 只与第 3 页有 bigram 重叠，第 1 页仍应保留在 top_k 内
    focused = _search_all(kb, PDF_PAGE_THREE)
    focused_pages = {r["page"] for r in _records_by_source(focused)["three_pages.pdf"]}
    assert focused["results"][0]["page"] == 3
    assert focused_pages == {1, 3}


# ============================================================
# C 组：DOCX 原始顺序与表格
# ============================================================
def test_C01_docx_documents_follow_original_body_order(tmp_path):
    """Loader 输出顺序 == w:body 子元素原始顺序（段落/表格交错）。"""
    from rag.loader import DocxLoader

    documents = DocxLoader(str(_write_docx(tmp_path / "manual.docx"))).load()

    assert len(documents) == 4
    kinds = [
        "table" if "table_index" in document.metadata else "segment"
        for document in documents
    ]
    assert kinds == ["segment", "table", "segment", "segment"]

    assert documents[0].page_content.startswith(DOCX_SECTION_ONE)
    assert DOCX_PARAGRAPH_A in documents[0].page_content
    assert DOCX_PARAGRAPH_B in documents[2].page_content
    assert documents[3].page_content.startswith(DOCX_SECTION_TWO)


def test_C02_empty_table_occupies_index_without_document(tmp_path):
    """空表格占用 table_index，但不产出 Document。"""
    from rag.loader import DocxLoader

    documents = DocxLoader(str(_write_docx(tmp_path / "manual.docx"))).load()
    table_documents = [d for d in documents if "table_index" in d.metadata]

    assert len(table_documents) == 1          # 只有一个表格产出 Document
    assert table_documents[0].metadata["table_index"] == 0
    assert table_documents[0].metadata["table_rows"] == 2
    assert table_documents[0].metadata["table_cols"] == 2
    # 空表格（0 行 0 列）没有产出任何 Document，也没有任何 document 用 table_index=1
    assert 1 not in {d.metadata.get("table_index") for d in documents}


def test_C03_section_index_and_paragraph_ownership(tmp_path):
    """两个章节的 section_index 递增；表格前后的段落仍归属正确章节。"""
    from rag.loader import DocxLoader

    documents = DocxLoader(str(_write_docx(tmp_path / "manual.docx"))).load()
    segments = [d for d in documents if "section" in d.metadata]

    assert [d.metadata["section"] for d in segments] == [
        DOCX_SECTION_ONE, DOCX_SECTION_ONE, DOCX_SECTION_TWO
    ]
    assert [d.metadata["section_index"] for d in segments] == [0, 0, 1]
    # 表格打断片段但不切换章节 → 同节第二段 segment_index 递增
    assert [d.metadata["segment_index"] for d in segments] == [0, 1, 0]


def test_C04_paragraph_indexes_match_real_paragraph_numbers(tmp_path):
    """start/end_paragraph_index 指向 w:p 的真实序号（含标题段）。"""
    from rag.loader import DocxLoader

    documents = DocxLoader(str(_write_docx(tmp_path / "manual.docx"))).load()
    segments = [d for d in documents if "section" in d.metadata]

    assert [d.metadata["start_paragraph_index"] for d in segments] == [0, 2, 3]
    assert [d.metadata["end_paragraph_index"] for d in segments] == [1, 2, 4]
    assert segments[0].metadata["heading_level"] == 1
    assert segments[2].metadata["heading_level"] == 1


def test_C05_chunk_seq_continuous_across_sections_and_tables(tmp_path, stub):
    """chunk_seq 是文件级序号：不因章节切换或表格出现而重置。"""
    kbdir = tmp_path / "kb"
    kbdir.mkdir()
    _write_docx(kbdir / "manual.docx")

    kb = _build_kb(kbdir, stub)
    result = _search_all(kb, "测试")
    records = _records_by_source(result)["manual.docx"]

    assert len(records) == 4
    assert sorted(record["chunk_seq"] for record in records) == [0, 1, 2, 3]
    # 4 个 chunk 分属 4 个不同 Document，每个的 chunk_id 都是 0
    assert {record["chunk_id"] for record in records} == {0}


# ============================================================
# D 组：metadata 逐层完整性
# ============================================================
def test_D01_metadata_survives_every_layer_for_pdf(tmp_path, stub):
    """Loader → Splitter → _stage_file → VectorStore → Retriever → search 全链不丢页码。"""
    kbdir = tmp_path / "kb"
    kbdir.mkdir()
    pdf_path = _write_pdf(kbdir / "paper.pdf")

    # 第 1 层：Loader
    documents = get_loader(str(pdf_path)).load()
    assert [d.metadata["page"] for d in documents] == [1, 3]

    # 第 2 层：Splitter（metadata.copy()，只新增 chunk_id）
    chunks = []
    for document in documents:
        chunks.extend(TextSplitter().split(document))
    assert [c.metadata["page"] for c in chunks] == [1, 3]
    assert [c.metadata["chunk_id"] for c in chunks] == [0, 0]
    assert all("chunk_seq" not in c.metadata for c in chunks)

    # 第 3/4 层：_stage_file + VectorStore
    kb = _build_kb(kbdir, stub)
    _search_all(kb, "文档测试")
    store = kb._retriever.vector_store
    assert isinstance(store, VectorStore)
    stored = [item["document"].metadata for item in store.items]
    assert [m["page"] for m in stored] == [1, 3]
    assert [m["chunk_seq"] for m in stored] == [0, 1]
    assert all(m["source"] == "paper.pdf" for m in stored)
    assert len({m["document_id"] for m in stored}) == 1

    # 第 5/6 层：Retriever + search
    records = _records_by_source(_search_all(kb, "文档测试"))["paper.pdf"]
    assert sorted(r["page"] for r in records) == [1, 3]
    assert [r["page_count"] for r in records] == [3, 3]


def test_D02_metadata_survives_every_layer_for_docx(tmp_path, stub):
    """DOCX 的 section / section_index / segment_index / table_index 全链不丢。"""
    kbdir = tmp_path / "kb"
    kbdir.mkdir()
    docx_path = _write_docx(kbdir / "manual.docx")

    documents = get_loader(str(docx_path)).load()
    assert [d.metadata.get("section_index") for d in documents] == [0, None, 0, 1]

    kb = _build_kb(kbdir, stub)
    _search_all(kb, "测试")
    stored = [item["document"].metadata for item in kb._retriever.vector_store.items]
    assert [m.get("section") for m in stored] == [
        DOCX_SECTION_ONE, None, DOCX_SECTION_ONE, DOCX_SECTION_TWO
    ]
    assert [m.get("table_index") for m in stored] == [None, 0, None, None]

    records = _search_all(kb, "测试")["results"]
    section_records = [r for r in records if "section" in r]
    table_records = [r for r in records if "table_index" in r]
    assert sorted(r["section_index"] for r in section_records) == [0, 0, 1]
    assert sorted(r["segment_index"] for r in section_records) == [0, 0, 1]
    assert [r["table_index"] for r in table_records] == [0]


def test_D03_missing_fields_are_not_fabricated(tmp_path, stub):
    """该格式没有的概念不出现在结果里（不是补 None，也不是补 0）。"""
    kbdir = tmp_path / "kb"
    kbdir.mkdir()
    _write_txt(kbdir / "notes.txt")
    _write_pdf(kbdir / "paper.pdf")

    kb = _build_kb(kbdir, stub)
    result = _search_all(kb, "测试")
    grouped = _records_by_source(result)

    for record in grouped["notes.txt"]:
        for absent in ("page", "page_count", "section", "section_index",
                       "segment_index", "table_index"):
            assert absent not in record, (absent, record)
        # 四种格式都有 document_id / chunk_seq
        assert isinstance(record["document_id"], str) and record["document_id"]
        assert isinstance(record["chunk_seq"], int)

    for record in grouped["paper.pdf"]:
        # PDF 没有章节概念
        assert "section" not in record
        assert "section_index" not in record
        assert "table_index" not in record


def test_D04_identity_fields_always_present_and_typed(tmp_path, stub):
    """身份字段在 RAG 层恒存在且类型正确：document_id 为非空 str，chunk_seq 为 >=0 int。"""
    kbdir = tmp_path / "kb"
    kbdir.mkdir()
    _write_txt(kbdir / "notes.txt")
    _write_pdf(kbdir / "paper.pdf")
    _write_docx(kbdir / "manual.docx")

    kb = _build_kb(kbdir, stub)
    result = _search_all(kb, "测试")

    assert result["results"]
    for record in result["results"]:
        assert isinstance(record["document_id"], str) and record["document_id"].strip()
        assert isinstance(record["chunk_seq"], int) and not isinstance(record["chunk_seq"], bool)
        assert record["chunk_seq"] >= 0
        # 身份唯一：(document_id, chunk_seq) 在单次结果内不重复
    identities = [(r["document_id"], r["chunk_seq"]) for r in result["results"]]
    assert len(identities) == len(set(identities))


def test_D05_context_builder_never_invents_location(tmp_path):
    """ContextBuilder 只渲染真实存在的定位字段；TXT 不出现页码/章节/表格字样。"""
    builder = ContextBuilder()

    txt_context = builder.build([
        {"document": Document("纯文本内容", {"source": "notes.txt", "chunk_id": 0}), "score": 0.5}
    ])
    assert "来源：notes.txt（片段 0）" in txt_context
    assert "页" not in txt_context
    assert "节" not in txt_context
    assert "表格" not in txt_context

    pdf_context = builder.build([
        {"document": Document("第2页内容", {"source": "p.pdf", "chunk_id": 0,
                                          "page": 2, "page_count": 3}), "score": 0.6}
    ])
    assert "第 2 页（共 3 页）" in pdf_context

    # page_count 缺失时不能补造总页数
    partial = builder.build([
        {"document": Document("内容", {"source": "p.pdf", "chunk_id": 0, "page": 2}), "score": 0.6}
    ])
    assert "第 2 页" in partial
    assert "共" not in partial

    # DOCX 空标题不得渲染成"第 1 节："
    empty_section = builder.build([
        {"document": Document("内容", {"source": "d.docx", "chunk_id": 0,
                                      "section": "", "section_index": 0}), "score": 0.6}
    ])
    assert "节" not in empty_section

    # DOCX 表格：table_index 0 基 → 展示 1 基
    table_context = builder.build([
        {"document": Document("R0C0 | R0C1", {"source": "d.docx", "chunk_id": 0,
                                             "table_index": 0}), "score": 0.6}
    ])
    assert "表格 1" in table_context


# ============================================================
# E 组：空文件与异常文件
# ============================================================
def test_E01_empty_files_are_counted_as_empty_not_loaded(tmp_path, stub):
    """空 TXT / 全空白 TXT / 空 DOCX 计入 empty_file_count，不计 loaded。"""
    import docx

    kbdir = tmp_path / "kb"
    kbdir.mkdir()
    _write_md(kbdir / "good.md")                 # 唯一有效文件
    (kbdir / "empty.txt").write_text("", encoding="utf-8")
    (kbdir / "blank.txt").write_text("   \n\t\n  ", encoding="utf-8")
    docx.Document().save(str(kbdir / "empty.docx"))

    kb = _build_kb(kbdir, stub)
    result = _search_all(kb, "测试")

    assert kb.file_count == 1
    assert kb.empty_file_count == 3
    assert kb.skipped_count == 0
    assert {record["source"] for record in result["results"]} == {"good.md"}


def test_E02_blank_pdf_raises_no_text_layer_error(tmp_path):
    """全空白页 PDF → NoTextLayerError，empty_pages 记录全部页码，failed_pages 为空。"""
    from rag.loader import PdfLoader

    pdf_path = _write_pdf(tmp_path / "blank.pdf", pages=(None, None, None))
    with pytest.raises(NoTextLayerError) as excinfo:
        PdfLoader(str(pdf_path)).load()

    assert excinfo.value.empty_pages == [1, 2, 3]
    assert excinfo.value.failed_pages == []


def test_E03_corrupt_pdf_raises_corrupt_error(tmp_path):
    """非 PDF 字节 / 零页 PDF → CorruptPdfError（不是 NoTextLayerError）。"""
    from rag.loader import PdfLoader

    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"%PDF-1.4\nthis is not a real pdf body")
    with pytest.raises(CorruptPdfError):
        PdfLoader(str(broken)).load()

    zero_page = tmp_path / "zero.pdf"
    zero_page.write_bytes(_build_pdf([]))
    with pytest.raises(CorruptPdfError):
        PdfLoader(str(zero_page)).load()

    # 两类异常互不继承：损坏不能被当成"无文本层"吞掉
    assert not issubclass(CorruptPdfError, NoTextLayerError)
    assert not issubclass(NoTextLayerError, CorruptPdfError)


def test_E04_pdf_without_text_layer_raises_no_text_layer_error(tmp_path):
    """有内容流但无任何文字（只有图形）→ NoTextLayerError。"""
    from rag.loader import PdfLoader

    graphics_only = b"0 0 100 100 re f\n0 0 200 200 re S"
    pdf_path = _write_pdf(tmp_path / "graphics.pdf", pages=(graphics_only,))
    with pytest.raises(NoTextLayerError) as excinfo:
        PdfLoader(str(pdf_path)).load()

    assert excinfo.value.empty_pages == [1]


def test_E05_failed_file_leaves_no_partial_index(tmp_path, stub):
    """加载失败的文件不得在 VectorStore 中留下任何 chunk。"""
    kbdir = tmp_path / "kb"
    kbdir.mkdir()
    _write_md(kbdir / "good.md")
    (kbdir / "broken.pdf").write_bytes(b"%PDF-1.4\nbroken body")

    kb = _build_kb(kbdir, stub)
    result = _search_all(kb, "测试")

    assert kb.file_count == 1
    assert kb.skipped_count == 1
    assert kb.chunk_count == 1
    assert {record["source"] for record in result["results"]} == {"good.md"}
    stored_sources = {
        item["document"].metadata["source"] for item in kb._retriever.vector_store.items
    }
    assert stored_sources == {"good.md"}

    # Stub 也没有为失败文件的任何文本编码过
    assert len(stub.calls) == kb.chunk_count + 1  # +1 是 query 自身的编码


def test_E06_partial_page_failure_counts_as_loaded_known_risk(tmp_path, stub, monkeypatch):
    """已知风险复现：单页提取失败时不抛异常，文件仍计入 loaded_count。

    生产行为（``rag/loader.py:304-311`` + ``rag/knowledge_base.py:410-413``）：
    失败页只写日志，文件整体仍算"加载成功"，缺页既不进 skipped_count 也不进
    empty_file_count。本用例把该行为固定下来，作为已知风险证据，不修改代码。
    """
    import pypdf._page

    original = pypdf._page.PageObject.extract_text
    state = {"calls": 0}

    def flaky(self, *args, **kwargs):
        state["calls"] += 1
        if state["calls"] == 2:
            raise RuntimeError("模拟第 2 页提取失败")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pypdf._page.PageObject, "extract_text", flaky)

    kbdir = tmp_path / "kb"
    kbdir.mkdir()
    _write_pdf(kbdir / "three_pages.pdf")

    kb = _build_kb(kbdir, stub)
    result = _search_all(kb, "文档测试")
    records = _records_by_source(result)["three_pages.pdf"]

    # 第 2 页原本就是空白页，此处再让它抛异常：失败页被跳过，文件仍算成功
    assert state["calls"] == 3
    assert kb.file_count == 1
    assert kb.skipped_count == 0        # ← 已知风险：缺页不体现在任何计数里
    assert kb.empty_file_count == 0
    assert sorted(record["page"] for record in records) == [1, 3]
    assert kb.chunk_count == 2


# ============================================================
# F 组：Stub Embedding 约束
# ============================================================
@pytest.mark.fresh_process
def test_F01_no_model_runtime_is_imported(tmp_path, stub):
    """索引与检索全程不得导入 sentence_transformers / torch / transformers。"""
    kbdir = tmp_path / "kb"
    kbdir.mkdir()
    _write_txt(kbdir / "notes.txt")
    _write_pdf(kbdir / "paper.pdf")

    kb = _build_kb(kbdir, stub)
    _search_all(kb, "测试")

    for module in ("sentence_transformers", "torch", "transformers", "huggingface_hub"):
        assert module not in sys.modules, module
    # 默认工厂（会 import rag.embedding）也未被触发
    assert "rag.embedding" not in sys.modules


def test_F02_vectors_are_fixed_dim_finite_and_non_zero(stub):
    """Stub 向量：维度固定、全部有限、非零。"""
    first = stub.encode("苹果文档测试内容")
    second = stub.encode("完全不同的另一段文本")

    assert len(first) == len(second) == stub.dim == 8
    for vector in (first, second):
        assert all(isinstance(value, float) for value in vector)
        assert all(math.isfinite(value) for value in vector)
        assert any(value != 0.0 for value in vector)

    # 确定性：同一文本两次编码完全一致
    assert stub.encode("苹果文档测试内容") == first
    # 不同文本得到不同向量（可区分，非恒等）
    assert first != second


def test_F03_encode_called_once_per_chunk_plus_query(tmp_path, stub):
    """encode 调用次数 == 有效 chunk 数 + 查询次数；每个 chunk 都有独立向量。"""
    kbdir = tmp_path / "kb"
    kbdir.mkdir()
    _write_txt(kbdir / "notes.txt")
    _write_pdf(kbdir / "paper.pdf")
    _write_docx(kbdir / "manual.docx")

    kb = _build_kb(kbdir, stub)
    _search_all(kb, "测试")

    assert kb.chunk_count == 7          # txt(1) + pdf(2) + docx(4)
    assert len(stub.calls) == kb.chunk_count + 1
    assert len(kb._retriever.vector_store.items) == kb.chunk_count

    _search_all(kb, "第二次查询")
    assert len(stub.calls) == kb.chunk_count + 2   # 索引已缓存，不重复编码


def test_F04_every_stored_item_has_vector_and_document(tmp_path, stub):
    """VectorStore 中每一条都必须同时具备 Document 与同维向量。"""
    kbdir = tmp_path / "kb"
    kbdir.mkdir()
    _write_pdf(kbdir / "paper.pdf")
    _write_md(kbdir / "guide.md")

    kb = _build_kb(kbdir, stub)
    _search_all(kb, "测试")

    items = kb._retriever.vector_store.items
    assert len(items) == 3
    for item in items:
        assert set(item) == {"document", "vector"}
        assert isinstance(item["document"], Document)
        assert item["document"].page_content.strip()
        assert len(item["vector"]) == stub.dim
        assert any(value != 0.0 for value in item["vector"])
