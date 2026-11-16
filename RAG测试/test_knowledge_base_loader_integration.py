# -*- coding: utf-8 -*-
"""P0-2 批次 B-1：KnowledgeBase 多格式 Loader 集成测试。

约束（与任务书一致）：
- 全部样本写入 pytest 临时目录，不读取 ``rag_knowledge/`` 生产目录；
- 使用确定性 stub embedding，不加载任何模型、不访问网络、不调用真实 API；
- 每个用例独立的 KnowledgeBase 实例与内存 VectorStore；
- PDF 样本本文件内手工构造（与 test_loader_pdf.py 同法），
  DOCX 样本用 python-docx 现场生成。

覆盖任务书第十一节的 22 项：四种格式入库、metadata 完整保留、
document_id 跨层一致、chunk_seq 文件内连续、空文件计数、
空白块过滤、单文件失败隔离、混合 PDF 故障、.md/.txt 向后兼容。
"""

import hashlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 仅用于「跨层 document_id 一致性」比对：生产代码里 rag 层不依赖该模块，
# 测试里必须引入它才能证明两层算法逐字节一致。
import knowledge_store  # noqa: E402
from rag.knowledge_base import KnowledgeBase  # noqa: E402


# ------------------------------------------------------------
# 各格式的最终 metadata 键集合（B1-06 契约）
# ------------------------------------------------------------
BASE_KEYS = {"source", "chunk_id", "document_id", "chunk_seq"}
PDF_KEYS = BASE_KEYS | {"page", "page_count"}
DOCX_SECTION_KEYS = BASE_KEYS | {
    "section",
    "section_index",
    "segment_index",
    "heading_level",
    "start_paragraph_index",
    "end_paragraph_index",
}
DOCX_TABLE_KEYS = BASE_KEYS | {
    "table_index",
    "table_rows",
    "table_cols",
    "table_ragged",
    "body_position",
    "after_paragraph_index",
}


# ------------------------------------------------------------
# 最小 PDF 构造器（与 test_loader_pdf.build_pdf 同法）
# ------------------------------------------------------------
def _escape_pdf_text(text):
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def build_pdf(page_texts):
    """构造带文本层的最小 PDF；空串页写成空内容流。"""
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


def write_docx(path, build):
    """用 python-docx 现场生成 DOCX；``build`` 接收 Document 对象。"""
    import docx

    document = docx.Document()
    build(document)
    document.save(str(path))
    return str(path)


# ------------------------------------------------------------
# stub 编码器与 KnowledgeBase 构造
# ------------------------------------------------------------
class StubEmbedding:
    """确定性 stub：按字符码点分桶计数，不加载任何模型。"""

    DIM = 8

    def __init__(self, fail_on_call=None):
        self.calls = 0
        self.fail_on_call = fail_on_call

    def encode(self, text):
        self.calls += 1
        if self.fail_on_call is not None and self.calls == self.fail_on_call:
            raise RuntimeError("stub 编码失败（测试注入）")
        vector = [0.0] * self.DIM
        for char in text:
            vector[ord(char) % self.DIM] += 1.0
        return vector


def make_kb(tmp_path, fail_on_call=None):
    return KnowledgeBase(
        str(tmp_path),
        embedding_factory=lambda: StubEmbedding(fail_on_call),
    )


def indexed_documents(kb):
    """触发惰性索引并返回入库的 Document 列表（顺序即入库顺序）。"""
    kb.search("索引")
    return [item["document"] for item in kb._retriever.vector_store.items]


def by_source(documents, name):
    return [d for d in documents if d.metadata.get("source") == name]


# ============================================================
# 1 / 2. TXT、MD 正常加载
# ============================================================
def test_txt_is_indexed(tmp_path):
    (tmp_path / "note.txt").write_text(
        "智能体项目包含工具调用与长期记忆。", encoding="utf-8"
    )
    kb = make_kb(tmp_path)
    documents = indexed_documents(kb)

    assert len(documents) == 1
    assert kb.file_count == 1
    assert kb.chunk_count == 1
    assert kb.skipped_count == 0
    assert kb.empty_file_count == 0
    assert set(documents[0].metadata) == BASE_KEYS
    assert documents[0].metadata["source"] == "note.txt"


def test_md_is_indexed(tmp_path):
    (tmp_path / "plan.md").write_text(
        "开发计划包含三个阶段。", encoding="utf-8"
    )
    kb = make_kb(tmp_path)
    documents = indexed_documents(kb)

    assert len(documents) == 1
    assert set(documents[0].metadata) == BASE_KEYS
    assert documents[0].metadata["source"] == "plan.md"
    # .md / .txt 不得因新增 Loader 而多出 page / section 等字段
    assert "page" not in documents[0].metadata


# ============================================================
# 3 / 6. PDF 多页 + page/page_count 进入 chunk metadata
# ============================================================
def test_pdf_multi_page_is_indexed_with_page_metadata(tmp_path):
    (tmp_path / "manual.pdf").write_bytes(
        build_pdf(["Page one text", "Page two text", "Page three text"])
    )
    kb = make_kb(tmp_path)
    documents = indexed_documents(kb)

    assert [d.metadata["page"] for d in documents] == [1, 2, 3]
    assert {d.metadata["page_count"] for d in documents} == {3}
    for document in documents:
        assert set(document.metadata) == PDF_KEYS
    assert kb.file_count == 1
    assert kb.chunk_count == 3
    assert kb.skipped_count == 0


# ============================================================
# 4 / 7. DOCX 多章节 + section_index / segment_index
# ============================================================
def test_docx_multi_section_is_indexed(tmp_path):
    def build(document):
        document.add_heading("第一章", level=1)
        document.add_paragraph("第一章正文内容。")
        document.add_heading("第二章", level=1)
        document.add_paragraph("第二章正文内容。")

    write_docx(tmp_path / "manual.docx", build)
    kb = make_kb(tmp_path)
    documents = indexed_documents(kb)

    assert [d.metadata["section"] for d in documents] == ["第一章", "第二章"]
    assert [d.metadata["section_index"] for d in documents] == [0, 1]
    assert [d.metadata["segment_index"] for d in documents] == [0, 0]
    for document in documents:
        assert set(document.metadata) == DOCX_SECTION_KEYS
    # DOCX 没有页概念：不得伪造 page / page_count
    assert all("page" not in d.metadata for d in documents)
    assert kb.file_count == 1


def test_docx_segment_index_advances_within_one_section(tmp_path):
    def build(document):
        document.add_heading("第一章", level=1)
        document.add_paragraph("段落一。")
        table = document.add_table(rows=1, cols=1)
        table.cell(0, 0).text = "单元格"
        document.add_paragraph("段落二。")

    write_docx(tmp_path / "seg.docx", build)
    kb = make_kb(tmp_path)
    documents = indexed_documents(kb)

    sections = [d for d in documents if "section" in d.metadata]
    assert len(sections) == 2
    # 同一节被表格打断后，section_index 不变、segment_index 递增
    assert [d.metadata["section_index"] for d in sections] == [0, 0]
    assert [d.metadata["segment_index"] for d in sections] == [0, 1]


# ============================================================
# 5 / 8. DOCX 表格独立 Document + table_index
# ============================================================
def test_docx_table_becomes_independent_document(tmp_path):
    def build(document):
        document.add_heading("数据", level=1)
        document.add_paragraph("下表为数据概览。")
        table = document.add_table(rows=2, cols=2)
        table.cell(0, 0).text = "指标"
        table.cell(0, 1).text = "数值"
        table.cell(1, 0).text = "召回率"
        table.cell(1, 1).text = "0.9"

    write_docx(tmp_path / "table.docx", build)
    kb = make_kb(tmp_path)
    documents = indexed_documents(kb)

    tables = [d for d in documents if "table_index" in d.metadata]
    assert len(tables) == 1
    assert set(tables[0].metadata) == DOCX_TABLE_KEYS
    assert tables[0].metadata["table_index"] == 0
    assert tables[0].metadata["table_rows"] == 2
    assert tables[0].metadata["table_cols"] == 2
    assert "召回率" in tables[0].page_content
    # 表格 Document 与节级 Document 是两类，键集合不得混用
    assert "section" not in tables[0].metadata


# ============================================================
# 9. 同一文件不同 Document 的 document_id 相同
# ============================================================
def test_document_id_is_identical_across_pages(tmp_path):
    (tmp_path / "manual.pdf").write_bytes(build_pdf(["alpha", "beta", "gamma"]))
    kb = make_kb(tmp_path)
    documents = indexed_documents(kb)

    ids = {d.metadata["document_id"] for d in documents}
    assert len(ids) == 1
    assert ids.pop() == hashlib.sha256(
        "manual.pdf".encode("utf-8")
    ).hexdigest()[:32]


# ============================================================
# 10. document_id 与上传层算法逐字节一致
# ============================================================
def test_document_id_matches_upload_layer(tmp_path):
    # 非 ASCII 文件名：把「编码方式必须同为 utf-8」也纳入验证
    name = "季度报告.md"
    (tmp_path / name).write_text("季度报告正文。", encoding="utf-8")
    kb = make_kb(tmp_path)
    documents = indexed_documents(kb)

    actual = documents[0].metadata["document_id"]
    assert actual == knowledge_store.compute_document_id(name)
    assert actual == hashlib.sha256(name.encode("utf-8")).hexdigest()[:32]
    assert len(actual) == 32
    assert set(actual) <= set("0123456789abcdef")


# ============================================================
# 11. PDF 多页 chunk_seq 全局连续
# ============================================================
def test_pdf_chunk_seq_is_globally_continuous(tmp_path):
    (tmp_path / "manual.pdf").write_bytes(
        build_pdf(["AAAAAAAAAA", "BBBBBBBBBB", "CCCCCCCCCC"])
    )
    kb = make_kb(tmp_path)
    documents = indexed_documents(kb)

    # 每页各 1 个 chunk：chunk_id 全部为 0，chunk_seq 必须跨页连续
    assert [d.metadata["page"] for d in documents] == [1, 2, 3]
    assert [d.metadata["chunk_id"] for d in documents] == [0, 0, 0]
    assert [d.metadata["chunk_seq"] for d in documents] == [0, 1, 2]


# ============================================================
# 12. DOCX 多片段 chunk_seq 全局连续
# ============================================================
def test_docx_chunk_seq_is_globally_continuous(tmp_path):
    def build(document):
        document.add_heading("第一章", level=1)
        document.add_paragraph("第一章" + "内容" * 80)   # 超长 → 2 个 chunk
        document.add_heading("第二章", level=1)
        document.add_paragraph("第二章正文。")

    write_docx(tmp_path / "long.docx", build)
    kb = make_kb(tmp_path)
    documents = indexed_documents(kb)

    assert [d.metadata["section_index"] for d in documents] == [0, 0, 1]
    # 节内 chunk_id 归零，但 chunk_seq 跨节连续
    assert [d.metadata["chunk_id"] for d in documents] == [0, 1, 0]
    assert [d.metadata["chunk_seq"] for d in documents] == [0, 1, 2]


# ============================================================
# 13. TXT/MD 单 Document：chunk_seq 与 chunk_id 数值一致
# ============================================================
def test_single_document_chunk_seq_equals_chunk_id(tmp_path):
    (tmp_path / "long.md").write_text("知识库" * 100, encoding="utf-8")  # 300 字
    kb = make_kb(tmp_path)
    documents = indexed_documents(kb)

    assert [d.metadata["chunk_id"] for d in documents] == [0, 1, 2, 3]
    assert [d.metadata["chunk_seq"] for d in documents] == [0, 1, 2, 3]


# ============================================================
# 14. 新文件 chunk_seq 重新从 0 开始
# ============================================================
def test_chunk_seq_restarts_for_each_file(tmp_path):
    (tmp_path / "a.md").write_text("甲" * 300, encoding="utf-8")
    (tmp_path / "b.txt").write_text("乙" * 300, encoding="utf-8")
    kb = make_kb(tmp_path)
    documents = indexed_documents(kb)

    first = by_source(documents, "a.md")
    second = by_source(documents, "b.txt")
    assert [d.metadata["chunk_seq"] for d in first] == [0, 1, 2, 3]
    assert [d.metadata["chunk_seq"] for d in second] == [0, 1, 2, 3]
    assert kb.file_count == 2


# ============================================================
# 15. 空 DOCX 不计入 loaded_count
# ============================================================
def test_empty_docx_not_counted_as_loaded(tmp_path):
    write_docx(tmp_path / "empty.docx", lambda document: None)
    (tmp_path / "good.txt").write_text("有效内容。", encoding="utf-8")

    kb = make_kb(tmp_path)
    documents = indexed_documents(kb)

    assert kb.file_count == 1
    assert kb.empty_file_count == 1
    assert kb.skipped_count == 0
    assert [d.metadata["source"] for d in documents] == ["good.txt"]


# ============================================================
# 16. 空 TXT 不计入 loaded_count
# ============================================================
def test_empty_txt_not_counted_as_loaded(tmp_path):
    (tmp_path / "empty.txt").write_text("", encoding="utf-8")
    (tmp_path / "good.md").write_text("有效内容。", encoding="utf-8")

    kb = make_kb(tmp_path)
    documents = indexed_documents(kb)

    assert kb.file_count == 1
    assert kb.empty_file_count == 1
    assert kb.skipped_count == 0
    assert [d.metadata["source"] for d in documents] == ["good.md"]


# ============================================================
# 17. 纯空白 chunk 不进入索引（且 chunk_seq 无空洞）
# ============================================================
def test_whitespace_only_chunk_is_not_indexed(tmp_path):
    # 前 75 个字符有效，其后 200 个空格：切块后会产出纯空白的块
    (tmp_path / "spaces.txt").write_text(
        "B" * 75 + " " * 200, encoding="utf-8"
    )
    kb = make_kb(tmp_path)
    documents = indexed_documents(kb)

    assert len(documents) == 1
    assert kb.chunk_count == 1
    assert all(d.page_content.strip() for d in documents)
    # 空白块被过滤后才分配序号，因此不存在空洞
    assert [d.metadata["chunk_id"] for d in documents] == [0]
    assert [d.metadata["chunk_seq"] for d in documents] == [0]


# ============================================================
# 18. 单文件 Embedding 中途失败：不留半截 chunks
# ============================================================
def test_embedding_failure_leaves_no_partial_chunks(tmp_path):
    (tmp_path / "a.md").write_text("有效内容。", encoding="utf-8")          # 1 chunk
    (tmp_path / "bad.pdf").write_bytes(build_pdf(["p1", "p2", "p3"]))      # 3 chunks
    # 调用顺序：a.md(1) → bad.pdf 第 1/2/3 页(2/3/4)，第 4 次失败
    kb = make_kb(tmp_path, fail_on_call=4)
    kb.search("检索")

    documents = [item["document"] for item in kb._retriever.vector_store.items]
    assert [d.metadata["source"] for d in documents] == ["a.md"]
    assert kb.skipped_count == 1
    assert kb.file_count == 1
    assert kb.chunk_count == 1
    # 关键：失败文件的前两页 chunk 一个都不能残留
    assert not by_source(documents, "bad.pdf")


# ============================================================
# 19. 单文件失败不影响其他文件
# ============================================================
def test_one_file_failure_does_not_block_others(tmp_path):
    (tmp_path / "good.md").write_text("有效内容。", encoding="utf-8")
    (tmp_path / "bad.md").write_bytes(b"\xff\xfe\x00garbage")

    kb = make_kb(tmp_path)
    documents = indexed_documents(kb)

    assert [d.metadata["source"] for d in documents] == ["good.md"]
    assert kb.skipped_count == 1
    assert kb.file_count == 1
    assert kb.chunk_count == 1


# ============================================================
# 20. 全部文件无有效 chunk：沿用既有初始化失败语义
# ============================================================
def test_all_files_without_valid_chunks_keep_init_failure_semantics(tmp_path):
    (tmp_path / "empty.txt").write_text("", encoding="utf-8")
    write_docx(tmp_path / "empty.docx", lambda document: None)

    kb = make_kb(tmp_path)
    result = kb.search("任意问题")

    assert result["ok"] is False
    assert result["stage"] == "init"
    assert "所有知识文档均无法加载或解析" in result["error"]
    assert kb.file_count == 0
    assert kb.empty_file_count == 2
    assert kb.skipped_count == 0


# ============================================================
# 21. 混合 PDF 故障：成功页面仍可索引
# ============================================================
def test_mixed_pdf_failure_still_indexes_good_pages(tmp_path, monkeypatch):
    import pypdf

    class _FakePage:
        def __init__(self, text=None, error=None):
            self._text = text
            self._error = error

        def extract_text(self):
            if self._error is not None:
                raise self._error
            return self._text

    class _FakeReader:
        def __init__(self, path, strict=False):
            self.pages = [
                _FakePage(text="第一页正文"),
                _FakePage(error=RuntimeError("坏页")),
                _FakePage(text="第三页正文"),
            ]
            self.is_encrypted = False

    (tmp_path / "broken.pdf").write_bytes(b"%PDF-1.4 placeholder")
    monkeypatch.setattr(pypdf, "PdfReader", _FakeReader)

    kb = make_kb(tmp_path)
    documents = indexed_documents(kb)

    assert [d.metadata["page"] for d in documents] == [1, 3]
    assert {d.metadata["page_count"] for d in documents} == {3}
    # 部分页故障属于「跳过该页但不失败」，不是整文件失败
    assert kb.skipped_count == 0
    assert kb.file_count == 1
    assert kb.chunk_count == 2


# ============================================================
# 22. .md / .txt 计数与 chunk 行为向后兼容
# ============================================================
def test_md_txt_counts_stay_backward_compatible(tmp_path):
    (tmp_path / "one.md").write_text("第一个文档。", encoding="utf-8")
    (tmp_path / "two.txt").write_text("第二个文档。", encoding="utf-8")

    kb = make_kb(tmp_path)
    documents = indexed_documents(kb)

    assert kb.file_count == 2
    assert kb.chunk_count == 2
    assert [d.metadata["source"] for d in documents] == ["one.md", "two.txt"]
    assert [d.metadata["chunk_id"] for d in documents] == [0, 0]
    assert [d.metadata["chunk_seq"] for d in documents] == [0, 0]


# ============================================================
# 23. B-1-R / G1：DOCX 章节正文与独立表格共用文件级 chunk_seq
# ============================================================
def test_docx_table_chunks_share_file_level_chunk_seq(tmp_path):
    """同一 DOCX 内「章节片段 + 表格 + 后续章节」共用同一个 chunk_seq 计数器。

    B-1 的实现里 ``chunk_seq`` 跨越同一文件的全部 Loader Document 递增，
    但此前的用例只覆盖了「多章节」与「单表格」两种形态，没有任何断言
    证明表格块也参与同一计数器。本用例把三者在同一个文件里串起来，
    并让表格文本超过 chunk_size，从而产生**多个**表格 chunk —— 逐个验证。

    全链路为真实实现：DocxLoader → KnowledgeBase._stage_file →
    TextSplitter → VectorStore；只有 embedding 用 stub（生产代码本就
    为此保留 ``embedding_factory`` 接缝），不 mock 任何提交逻辑。
    """

    def build(document):
        document.add_heading("第一章", level=1)
        document.add_paragraph("第一章正文。")
        # 5 行 × 3 列，跨列拼接后总长约 114 字 → 必被切成 2 个 chunk
        rows = [
            ("指标名称", "数值", "备注说明"),
            ("召回率指标", "0.93", "本季度显著提升"),
            ("准确率指标", "0.91", "环比小幅上升"),
            ("响应时长值", "120", "毫秒为单位统计"),
            ("覆盖率指标", "0.88", "覆盖全部主要场景"),
        ]
        table = document.add_table(rows=len(rows), cols=3)
        for row_index, row in enumerate(rows):
            for col_index, text in enumerate(row):
                table.cell(row_index, col_index).text = text
        document.add_heading("第二章", level=1)
        document.add_paragraph("第二章正文。")

    write_docx(tmp_path / "mixed.docx", build)
    kb = make_kb(tmp_path)
    documents = indexed_documents(kb)

    section_chunks = [d for d in documents if "section" in d.metadata]
    table_chunks = [d for d in documents if "table_index" in d.metadata]

    # 1. 章节正文 chunk 存在（第一章、第二章各一个）
    assert len(section_chunks) == 2
    assert [d.metadata["section"] for d in section_chunks] == ["第一章", "第二章"]

    # 2. 表格 chunk 存在，且被切成了多个块（覆盖多 chunk 表格场景）
    assert len(table_chunks) >= 2

    # 3. 每个表格 chunk 都保留 table_index，且指向同一个表格
    for chunk in table_chunks:
        assert chunk.metadata["table_index"] == 0
        assert chunk.metadata["table_rows"] == 5
        assert chunk.metadata["table_cols"] == 3
        assert "section" not in chunk.metadata

    # 4. 所有入库 chunk 都带 chunk_seq
    assert all(isinstance(d.metadata.get("chunk_seq"), int) for d in documents)

    # 5. 文件级连续：恰好是 0..n-1，无重复、无跳号
    assert [d.metadata["chunk_seq"] for d in documents] == list(
        range(len(documents))
    )

    # 6. 章节与表格共用同一个 document_id
    document_ids = {d.metadata["document_id"] for d in documents}
    assert len(document_ids) == 1
    assert document_ids.pop() == hashlib.sha256(
        "mixed.docx".encode("utf-8")
    ).hexdigest()[:32]

    # 7. chunk_id 仍是 Splitter 的「Document 内序号」：每个 Document 从 0 起
    assert [d.metadata["chunk_id"] for d in section_chunks] == [0, 0]
    assert [d.metadata["chunk_id"] for d in table_chunks] == list(
        range(len(table_chunks))
    )

    # 8. 顺序未被重排：表格块连续地夹在两个章节块之间
    kinds = [
        "table" if "table_index" in d.metadata else "section"
        for d in documents
    ]
    assert kinds[0] == "section"
    assert kinds[-1] == "section"
    first_table = kinds.index("table")
    last_table = len(kinds) - 1 - kinds[::-1].index("table")
    assert all(
        kind == "table" for kind in kinds[first_table:last_table + 1]
    )

