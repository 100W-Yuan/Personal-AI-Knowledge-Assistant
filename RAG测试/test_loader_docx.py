# -*- coding: utf-8 -*-
"""P0-2 批次 A：DocxLoader 测试。

样本用 python-docx 在本文件内即时生成，全部写入 pytest 的临时目录。

覆盖任务书要求的 DOCX 场景：
普通段落 / 多级标题 / 标题之间段落 / 表格 / 段落与表格交错 / 空段落 /
空表格 / 多个节 / 中文与特殊字符 / 段落范围索引 / 表格行列与位置 /
以及「不得伪造页码」这一硬约束。
"""

import docx
import pytest
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

from rag.document import Document
from rag.loader import DocxLoader, DocxLoadError


# ------------------------------------------------------------
# 构造工具
# ------------------------------------------------------------
def _save(path, build):
    document = docx.Document()
    build(document)
    document.save(str(path))
    return str(path)


def _append_empty_table(document):
    """在 body 中插入一个没有 tblPr/grid 的空表格。"""
    table_element = OxmlElement("w:tbl")
    body = document.element.body
    sect_pr = body.find(qn("w:sectPr"))
    if sect_pr is not None:
        sect_pr.addprevious(table_element)
    else:
        body.append(table_element)
    return table_element


def _sections(documents):
    return [item for item in documents if "section" in item.metadata]


def _tables(documents):
    return [item for item in documents if "table_index" in item.metadata]


# ------------------------------------------------------------
# 段落与节
# ------------------------------------------------------------
def test_plain_paragraphs_form_single_preamble_section(tmp_path):
    path = _save(
        tmp_path / "plain.docx",
        lambda d: [d.add_paragraph("第一段"), d.add_paragraph("第二段")],
    )
    documents = DocxLoader(path).load()
    sections = _sections(documents)

    assert len(sections) == 1
    metadata = sections[0].metadata
    assert metadata["section"] == ""
    assert metadata["heading_level"] == 0
    assert metadata["section_index"] == 0
    assert metadata["start_paragraph_index"] == 0
    assert metadata["end_paragraph_index"] == 1
    assert "第一段" in sections[0].page_content
    assert "第二段" in sections[0].page_content


def test_multi_level_headings_create_separate_sections(tmp_path):
    def build(document):
        document.add_heading("第一章", level=1)
        document.add_paragraph("第一章正文")
        document.add_heading("第一节", level=2)
        document.add_paragraph("第一节正文")

    path = _save(tmp_path / "headings.docx", build)
    sections = _sections(DocxLoader(path).load())

    assert len(sections) == 2
    assert [item.metadata["heading_level"] for item in sections] == [1, 2]
    assert [item.metadata["section"] for item in sections] == ["第一章", "第一节"]
    assert [item.metadata["section_index"] for item in sections] == [0, 1]


def test_paragraphs_between_headings_stay_in_preceding_section(tmp_path):
    def build(document):
        document.add_heading("甲", level=1)
        document.add_paragraph("甲下的第一段")
        document.add_paragraph("甲下的第二段")
        document.add_heading("乙", level=1)
        document.add_paragraph("乙下的段落")

    path = _save(tmp_path / "between.docx", build)
    sections = _sections(DocxLoader(path).load())

    assert len(sections) == 2
    # 段落索引：甲=0，甲段1=1，甲段2=2，乙=3，乙段=4
    assert sections[0].metadata["start_paragraph_index"] == 0
    assert sections[0].metadata["end_paragraph_index"] == 2
    assert sections[1].metadata["start_paragraph_index"] == 3
    assert sections[1].metadata["end_paragraph_index"] == 4


def test_leading_paragraphs_before_first_heading_become_preamble(tmp_path):
    def build(document):
        document.add_paragraph("封面说明")
        document.add_heading("正文标题", level=1)
        document.add_paragraph("正文内容")

    path = _save(tmp_path / "preamble.docx", build)
    sections = _sections(DocxLoader(path).load())

    assert len(sections) == 2
    assert sections[0].metadata["section"] == ""
    assert sections[0].metadata["heading_level"] == 0
    assert sections[1].metadata["section"] == "正文标题"
    assert sections[1].metadata["section_index"] == 1


def test_empty_paragraphs_do_not_create_empty_sections(tmp_path):
    def build(document):
        document.add_paragraph("")
        document.add_paragraph("")
        document.add_heading("标题", level=1)
        document.add_paragraph("内容")

    path = _save(tmp_path / "emptypara.docx", build)
    sections = _sections(DocxLoader(path).load())

    # 全空的 preamble 节不产出；只剩标题节。
    assert len(sections) == 1
    assert sections[0].metadata["section"] == "标题"
    assert sections[0].page_content.strip()


def test_section_content_excludes_empty_leading_lines(tmp_path):
    def build(document):
        document.add_heading("标题", level=1)
        document.add_paragraph("")
        document.add_paragraph("有效内容")

    path = _save(tmp_path / "trim.docx", build)
    sections = _sections(DocxLoader(path).load())
    assert sections[0].page_content == "标题\n\n有效内容"


def test_chinese_and_special_characters_are_preserved(tmp_path):
    text = "中文段落，含标点。！@#￥%……&*（）——【】《》"
    path = _save(
        tmp_path / "cn.docx",
        lambda d: [d.add_heading("中文标题", level=1), d.add_paragraph(text)],
    )
    documents = DocxLoader(path).load()
    combined = "\n".join(item.page_content for item in documents)

    assert "中文标题" in combined
    assert text in combined


# ------------------------------------------------------------
# 表格
# ------------------------------------------------------------
def test_table_produces_document_with_shape(tmp_path):
    def build(document):
        table = document.add_table(rows=2, cols=3)
        for row_index in range(2):
            for col_index in range(3):
                table.cell(row_index, col_index).text = f"r{row_index}c{col_index}"

    path = _save(tmp_path / "table.docx", build)
    tables = _tables(DocxLoader(path).load())

    assert len(tables) == 1
    metadata = tables[0].metadata
    assert metadata["table_index"] == 0
    assert metadata["table_rows"] == 2
    assert metadata["table_cols"] == 3
    assert isinstance(metadata["table_ragged"], bool)
    assert "r0c0" in tables[0].page_content
    assert "r1c2" in tables[0].page_content


def test_table_rows_are_joined_with_separator(tmp_path):
    def build(document):
        table = document.add_table(rows=1, cols=2)
        table.cell(0, 0).text = "左"
        table.cell(0, 1).text = "右"

    path = _save(tmp_path / "sep.docx", build)
    tables = _tables(DocxLoader(path).load())
    assert tables[0].page_content == "左 | 右"


def test_paragraph_and_table_order_is_preserved(tmp_path):
    """核心：段落与表格交错时，位置索引必须反映原始 XML 顺序。"""

    def build(document):
        document.add_paragraph("p0")
        table0 = document.add_table(rows=1, cols=1)
        table0.cell(0, 0).text = "t0"
        document.add_paragraph("p1")
        table1 = document.add_table(rows=1, cols=1)
        table1.cell(0, 0).text = "t1"

    path = _save(tmp_path / "interleave.docx", build)
    documents = DocxLoader(path).load()
    tables = _tables(documents)

    # 直接断言返回列表的实际顺序（不靠辅助 metadata）：
    # 文档 XML 顺序是 p0 → t0 → p1 → t1，输出必须一致。
    assert [item.page_content for item in documents] == ["p0", "t0", "p1", "t1"]

    assert len(tables) == 2
    assert [item.metadata["table_index"] for item in tables] == [0, 1]
    # p0 / table0 / p1 / table1  → body_position 应为 0,1,2,3
    assert [item.metadata["body_position"] for item in tables] == [1, 3]
    # 表格之前的段落数
    assert [item.metadata["after_paragraph_index"] for item in tables] == [1, 2]


def test_table_does_not_break_paragraph_index_sequence(tmp_path):
    """表格把节切成两个片段，但段落索引连续、不重叠、不跳号。"""

    def build(document):
        document.add_heading("标题", level=1)
        document.add_paragraph("段一")
        table = document.add_table(rows=1, cols=1)
        table.cell(0, 0).text = "表格"
        document.add_paragraph("段二")

    path = _save(tmp_path / "seq.docx", build)
    segments = _sections(DocxLoader(path).load())

    # 段落索引只数 w:p：标题=0，段一=1，段二=2（表格不占段号）
    # 表格打断后分为两个片段：[标题,段一] 与 [段二]
    assert [
        (item.metadata["start_paragraph_index"],
         item.metadata["end_paragraph_index"])
        for item in segments
    ] == [(0, 1), (2, 2)]
    # 同一节：section_index 不变，segment_index 连续
    assert [item.metadata["section_index"] for item in segments] == [0, 0]
    assert [item.metadata["segment_index"] for item in segments] == [0, 1]


def test_empty_table_is_skipped(tmp_path):
    def build(document):
        document.add_paragraph("正文")
        _append_empty_table(document)

    path = _save(tmp_path / "emptytable.docx", build)
    documents = DocxLoader(path).load()

    assert _tables(documents) == []
    assert len(_sections(documents)) == 1


def test_multiple_tables_are_numbered_in_order(tmp_path):
    def build(document):
        for index in range(3):
            table = document.add_table(rows=1, cols=1)
            table.cell(0, 0).text = f"table-{index}"

    path = _save(tmp_path / "threetables.docx", build)
    tables = _tables(DocxLoader(path).load())

    assert [item.metadata["table_index"] for item in tables] == [0, 1, 2]
    assert [item.metadata["body_position"] for item in tables] == [0, 1, 2]


# ------------------------------------------------------------
# 多节
# ------------------------------------------------------------
def test_multiple_sections_are_indexed_sequentially(tmp_path):
    def build(document):
        for index in range(1, 4):
            document.add_heading(f"标题{index}", level=1)
            document.add_paragraph(f"内容{index}")

    path = _save(tmp_path / "multisection.docx", build)
    sections = _sections(DocxLoader(path).load())

    assert len(sections) == 3
    assert [item.metadata["section_index"] for item in sections] == [0, 1, 2]
    assert [item.metadata["section"] for item in sections] == ["标题1", "标题2", "标题3"]


# ------------------------------------------------------------
# 硬约束：不得伪造页码
# ------------------------------------------------------------
def test_no_page_metadata_anywhere(tmp_path):
    """DOCX 没有页概念，任何 Document 都不得出现 page / page_count。"""

    def build(document):
        document.add_heading("标题", level=1)
        document.add_paragraph("正文")
        table = document.add_table(rows=1, cols=1)
        table.cell(0, 0).text = "单元格"

    path = _save(tmp_path / "nopage.docx", build)
    documents = DocxLoader(path).load()

    assert documents
    for item in documents:
        assert "page" not in item.metadata
        assert "page_count" not in item.metadata


def test_section_metadata_keys(tmp_path):
    path = _save(
        tmp_path / "keys.docx",
        lambda d: [d.add_heading("标题", level=1), d.add_paragraph("正文")],
    )
    sections = _sections(DocxLoader(path).load())

    assert set(sections[0].metadata) == {
        "source",
        "section",
        "section_index",
        "segment_index",
        "heading_level",
        "start_paragraph_index",
        "end_paragraph_index",
    }


def test_table_metadata_keys(tmp_path):
    path = _save(
        tmp_path / "tkeys.docx",
        lambda d: d.add_table(rows=1, cols=1),
    )
    tables = _tables(DocxLoader(path).load())

    assert set(tables[0].metadata) == {
        "source",
        "table_index",
        "table_rows",
        "table_cols",
        "table_ragged",
        "body_position",
        "after_paragraph_index",
    }


def test_source_is_file_path(tmp_path):
    path = _save(tmp_path / "src.docx", lambda d: d.add_paragraph("x"))
    documents = DocxLoader(path).load()
    assert all(item.metadata["source"] == path for item in documents)


def test_documents_are_document_instances(tmp_path):
    path = _save(tmp_path / "inst.docx", lambda d: d.add_paragraph("x"))
    assert all(isinstance(item, Document) for item in DocxLoader(path).load())


# ------------------------------------------------------------
# 异常
# ------------------------------------------------------------
def test_missing_file_propagates_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        DocxLoader(str(tmp_path / "nope.docx")).load()


def test_invalid_docx_raises_docx_load_error(tmp_path):
    path = tmp_path / "broken.docx"
    path.write_bytes(b"this is not a docx package")
    with pytest.raises(DocxLoadError):
        DocxLoader(str(path)).load()


def test_empty_file_raises_docx_load_error(tmp_path):
    path = tmp_path / "empty.docx"
    path.write_bytes(b"")
    with pytest.raises(DocxLoadError):
        DocxLoader(str(path)).load()


# ------------------------------------------------------------
# A-R1 / B2：有序内容流（段落—表格交错）
# ------------------------------------------------------------
def test_five_element_interleaved_order(tmp_path):
    """段落—表格—段落—表格—段落：直接断言输出顺序与内容。"""

    def build(document):
        document.add_paragraph("p0")
        table0 = document.add_table(rows=1, cols=1)
        table0.cell(0, 0).text = "t0"
        document.add_paragraph("p1")
        table1 = document.add_table(rows=1, cols=1)
        table1.cell(0, 0).text = "t1"
        document.add_paragraph("p2")

    path = _save(tmp_path / "five.docx", build)
    documents = DocxLoader(path).load()

    # 顺序断言（不借助 body_position 等辅助 metadata）
    assert [item.page_content for item in documents] == [
        "p0", "t0", "p1", "t1", "p2",
    ]

    # 类型断言：段-表-段-表-段
    kinds = [
        "table" if "table_index" in item.metadata else "segment"
        for item in documents
    ]
    assert kinds == ["segment", "table", "segment", "table", "segment"]


def test_interleaved_section_identity_is_unchanged(tmp_path):
    """同一节被两个表格打断三次，section_index 必须始终为 0。"""

    def build(document):
        document.add_heading("章节", level=1)
        document.add_paragraph("a1")
        table0 = document.add_table(rows=1, cols=1)
        table0.cell(0, 0).text = "x"
        document.add_paragraph("a2")
        table1 = document.add_table(rows=1, cols=1)
        table1.cell(0, 0).text = "y"
        document.add_paragraph("a3")

    path = _save(tmp_path / "identity.docx", build)
    segments = _sections(DocxLoader(path).load())

    assert len(segments) == 3
    assert [item.metadata["section"] for item in segments] == ["章节"] * 3
    assert [item.metadata["section_index"] for item in segments] == [0, 0, 0]
    assert [item.metadata["heading_level"] for item in segments] == [1, 1, 1]


def test_segment_index_is_continuous_within_section(tmp_path):
    """segment_index 在同一节内从 0 连续递增。"""

    def build(document):
        document.add_paragraph("s0")
        for index in range(2):
            table = document.add_table(rows=1, cols=1)
            table.cell(0, 0).text = f"t{index}"
            document.add_paragraph(f"s{index + 1}")

    path = _save(tmp_path / "segindex.docx", build)
    segments = _sections(DocxLoader(path).load())

    assert [item.metadata["segment_index"] for item in segments] == [0, 1, 2]
    assert [item.metadata["section_index"] for item in segments] == [0, 0, 0]


def test_segment_index_restarts_in_new_section(tmp_path):
    """换节后 segment_index 重新从 0 开始。"""

    def build(document):
        document.add_heading("甲", level=1)
        document.add_paragraph("a")
        table = document.add_table(rows=1, cols=1)
        table.cell(0, 0).text = "t"
        document.add_paragraph("b")
        document.add_heading("乙", level=1)
        document.add_paragraph("c")

    path = _save(tmp_path / "restart.docx", build)
    segments = _sections(DocxLoader(path).load())

    # 甲: 2 个片段(0,1)；乙: 1 个片段(0)
    assert [item.metadata["section"] for item in segments] == ["甲", "甲", "乙"]
    assert [item.metadata["section_index"] for item in segments] == [0, 0, 1]
    assert [item.metadata["segment_index"] for item in segments] == [0, 1, 0]


def test_interleaved_paragraph_ranges_are_accurate(tmp_path):
    """每个片段的段落范围必须精确对应其包含的段落。"""

    def build(document):
        document.add_paragraph("p0")
        document.add_paragraph("p1")
        table = document.add_table(rows=1, cols=1)
        table.cell(0, 0).text = "t"
        document.add_paragraph("p2")

    path = _save(tmp_path / "ranges.docx", build)
    segments = _sections(DocxLoader(path).load())

    assert len(segments) == 2
    assert segments[0].metadata["start_paragraph_index"] == 0
    assert segments[0].metadata["end_paragraph_index"] == 1
    assert segments[0].page_content == "p0\np1"
    assert segments[1].metadata["start_paragraph_index"] == 2
    assert segments[1].metadata["end_paragraph_index"] == 2
    assert segments[1].page_content == "p2"


def test_paragraphs_around_table_are_both_kept(tmp_path):
    """表格前后段落不得丢失。"""

    def build(document):
        document.add_heading("标题", level=1)
        document.add_paragraph("表格前段落")
        table = document.add_table(rows=1, cols=1)
        table.cell(0, 0).text = "表内容"
        document.add_paragraph("表格后段落")

    path = _save(tmp_path / "kept.docx", build)
    documents = DocxLoader(path).load()
    combined = "\n".join(item.page_content for item in documents)

    assert "表格前段落" in combined
    assert "表格后段落" in combined
    assert "表内容" in combined


def test_table_content_not_lost_in_interleaved_document(tmp_path):
    """交错结构中每个表格的单元格内容都必须完整保留。"""

    def build(document):
        for index in range(2):
            table = document.add_table(rows=2, cols=2)
            for row in range(2):
                for col in range(2):
                    table.cell(row, col).text = f"r{row}c{col}t{index}"
            document.add_paragraph(f"p{index}")

    path = _save(tmp_path / "tablecontent.docx", build)
    documents = DocxLoader(path).load()
    combined = "\n".join(item.page_content for item in documents)

    for index in range(2):
        for row in range(2):
            for col in range(2):
                assert f"r{row}c{col}t{index}" in combined


def test_middle_empty_table_consumes_table_index(tmp_path):
    """中间的空表格必须占用 table_index，后面的表格编号不得前移。"""

    def build(document):
        first = document.add_table(rows=1, cols=1)
        first.cell(0, 0).text = "first"
        _append_empty_table(document)
        third = document.add_table(rows=1, cols=1)
        third.cell(0, 0).text = "third"

    path = _save(tmp_path / "middleempty.docx", build)
    tables = _tables(DocxLoader(path).load())

    assert len(tables) == 2
    # 空表格占 1 号，因此第二个真实表格是 2 号
    assert [item.metadata["table_index"] for item in tables] == [0, 2]
    assert [item.page_content for item in tables] == ["first", "third"]


def test_interleaved_document_still_has_no_page_metadata(tmp_path):
    """交错结构下依然不得出现 page 字段。"""

    def build(document):
        document.add_paragraph("p0")
        table = document.add_table(rows=1, cols=1)
        table.cell(0, 0).text = "t"
        document.add_paragraph("p1")

    path = _save(tmp_path / "nopage2.docx", build)
    for item in DocxLoader(path).load():
        assert "page" not in item.metadata
        assert "page_count" not in item.metadata
