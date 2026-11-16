# -*- coding: utf-8 -*-
"""RAG 6.8：ContextBuilder 独立测试。

验证目标（使用虚构 Retriever 结果，不加载模型、不调用云端 API）：

  A. 基础结构：每条结果仍渲染「标记 / 来源 / 相似度 / 内容」四部分，
     顺序与输入一致，空结果返回空字符串，TXT·MD 的来源行保持原样；
  B. PDF 页码：page 是 1 基，原样展示；page_count 存在时附带总页数，
     缺失或异常（小于当前页）时只报当前页；
  C. DOCX 章节：section_index 是 0 基，展示时换成 1 基人类序号，
     标题原文一并展示；无序号时只显示标题，空白标题退化为文件名；
  D. DOCX 表格：table_index 是 0 基（空表格也占号），展示时 +1；
     章节块与表格块各读自己的 metadata，互不串用；
  E. 缺失与异常：没有定位字段时不产生分隔符，不用 None / unknown / 0
     占位；bool、负数、字符串类型的伪序号一律不当作序号；
  F. 当前没有预算与截断：build() 不做排序、不丢弃、不截断，输出总长度
     等于全部结果的渲染长度之和；定位标签只增加来源行的字符，不改变
     内容行与相似度行；
  G. 端到端：真实 PDF / DOCX 经 Loader 索引后，Context 里的页号与章节
     标题来自解析结果（而非测试手工拼装）。

运行方式（均支持）：
  python -m RAG测试.test_context_builder        （脚本模式 + 退出码）
  pytest -q "RAG测试/test_context_builder.py"    （测试模式）
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.context_builder import ContextBuilder   # noqa: E402
from rag.document import Document                 # noqa: E402


def make_results():
    """构造虚构的 Retriever 结果（结构与 VectorStore.search 返回值一致）。"""
    return [
        {
            "document": Document("我的Agent项目使用RAG检索方案",
                                 {"source": "A.txt"}),
            "vector": [0.1, 0.2, 0.3],
            "score": 0.98,
        },
        {
            "document": Document("Hello，世界！这是一个 test 用例（含标点），"
                                 "混合了中英文 content。",
                                 {"source": "B.txt"}),
            "vector": [0.4, 0.5, 0.6],
            "score": 0.72,
        },
        {
            "document": Document("健身计划：每周三次力量训练",
                                 {"source": "C.txt"}),
            "vector": [0.7, 0.8, 0.9],
            "score": 0.35,
        },
    ]


# ----------------------------------------------------------------------
# 测试辅助：单条结果渲染 + 取出「来源：」行
# ----------------------------------------------------------------------

DEFAULT_CONTENT = "片段正文"


def build_one(metadata, content=DEFAULT_CONTENT, score=0.5):
    """用给定 metadata 渲染单条结果的 Context。"""
    return ContextBuilder().build([{
        "document": Document(content, metadata),
        "vector": [1.0],
        "score": score,
    }])


def source_line(context):
    """取出 Context 里的第一条「来源：」行，便于逐字符断言标签格式。"""
    for line in context.splitlines():
        if line.startswith("来源："):
            return line
    raise AssertionError(f"Context 中没有来源行：{context!r}")


# ----------------------------------------------------------------------
# A. 基础结构
# ----------------------------------------------------------------------

def test_multiple_results_all_included_in_order():
    results = make_results()
    context = ContextBuilder().build(results)

    # 每条 Chunk 的内容与 score 都出现；顺序与输入一致；不丢失内容
    last_pos = -1
    for r in results:
        content = r["document"].page_content
        assert content in context, f"内容丢失：{content}"
        assert f"相似度：{r['score']}" in context, f"score 丢失：{r['score']}"
        pos = context.find(content)
        assert pos > last_pos, "Chunk 顺序与输入不一致"
        last_pos = pos

    # 三条结果 → 三个分隔标记
    assert context.count("【相关资料】") == 3


def test_empty_results_returns_empty_string():
    context = ContextBuilder().build([])
    assert context == ""


def test_special_text_preserved():
    chunk = "Hello，世界！这是一个 test 用例（含标点）。"
    results = [{
        "document": Document(chunk, {"source": "S.txt"}),
        "vector": [1.0],
        "score": 0.5,
    }]
    context = ContextBuilder().build(results)
    assert chunk in context


def test_block_keeps_four_part_structure():
    """无定位字段时，整块必须逐字节等于引入定位标签之前的格式。"""
    context = build_one({"source": "A.txt", "chunk_id": 0},
                        content="正文内容", score=0.9)
    assert context == (
        "【相关资料】\n"
        "来源：A.txt（片段 0）\n"
        "相似度：0.9\n"
        "内容：正文内容\n\n"
    )


def test_text_and_markdown_source_line_unchanged():
    """TXT / Markdown 没有定位概念，来源行保持旧格式（业务契约测试依赖它）。"""
    assert source_line(build_one({"source": "faq.md", "chunk_id": 2})) == \
        "来源：faq.md（片段 2）"
    assert source_line(build_one({"source": "A.txt", "chunk_id": 0})) == \
        "来源：A.txt（片段 0）"


def test_missing_chunk_id_omits_fragment_clause():
    """chunk_id 缺失时不渲染「片段 None / 片段 unknown」这种伪编号。"""
    assert source_line(build_one({"source": "A.txt"})) == "来源：A.txt"


def test_missing_source_falls_back_to_unknown():
    """source 缺失沿用既有占位（保持旧行为，不扩大改动面）。"""
    assert source_line(build_one({"chunk_id": 0})) == "来源：unknown（片段 0）"


# ----------------------------------------------------------------------
# B. PDF 页码
# ----------------------------------------------------------------------

def test_pdf_label_shows_page_and_page_count():
    metadata = {
        "source": "/data/技术报告.pdf",
        "chunk_id": 0,
        "page": 2,
        "page_count": 5,
        "document_id": "0f1e2d3c4b5a",
        "chunk_seq": 6,
    }
    assert source_line(build_one(metadata)) == \
        "来源：/data/技术报告.pdf（片段 0）｜第 2 页（共 5 页）"


def test_pdf_first_page_is_not_shifted():
    """page 由 PdfLoader 写为 1 基，展示时不得再 +1。"""
    metadata = {"source": "x.pdf", "chunk_id": 0, "page": 1, "page_count": 5}
    assert source_line(build_one(metadata)) == \
        "来源：x.pdf（片段 0）｜第 1 页（共 5 页）"


def test_pdf_without_page_count_shows_page_only():
    """page_count 缺失时只报当前页，不编造总页数。"""
    metadata = {"source": "x.pdf", "chunk_id": 0, "page": 3}
    assert source_line(build_one(metadata)) == "来源：x.pdf（片段 0）｜第 3 页"


def test_pdf_page_count_smaller_than_page_shows_page_only():
    """异常数据（共 2 页却报第 3 页）不得渲染成「第 3 页（共 2 页）」。"""
    metadata = {"source": "x.pdf", "chunk_id": 0, "page": 3, "page_count": 2}
    assert source_line(build_one(metadata)) == "来源：x.pdf（片段 0）｜第 3 页"


def test_pdf_zero_page_is_not_displayed():
    """page 是 1 基，0 不是合法页码，不能当「第 0 页」展示。"""
    metadata = {"source": "x.pdf", "chunk_id": 0, "page": 0, "page_count": 5}
    assert source_line(build_one(metadata)) == "来源：x.pdf（片段 0）"


# ----------------------------------------------------------------------
# C. DOCX 章节
# ----------------------------------------------------------------------

def test_docx_section_index_is_displayed_one_based():
    metadata = {
        "source": "需求文档.docx",
        "chunk_id": 0,
        "section": "第一章 概述",
        "section_index": 0,
        "segment_index": 0,
        "heading_level": 1,
    }
    assert source_line(build_one(metadata)) == \
        "来源：需求文档.docx（片段 0）｜第 1 节：第一章 概述"


def test_docx_section_index_two_renders_third_section():
    metadata = {
        "source": "y.docx",
        "chunk_id": 0,
        "section": "第三章 风险",
        "section_index": 2,
    }
    assert source_line(build_one(metadata)) == \
        "来源：y.docx（片段 0）｜第 3 节：第三章 风险"


def test_docx_section_without_index_shows_title_only():
    """有序号按序号渲染，没序号就只报标题 —— 不猜它是第 1 节。"""
    metadata = {"source": "y.docx", "chunk_id": 0, "section": "附录 A"}
    assert source_line(build_one(metadata)) == \
        "来源：y.docx（片段 0）｜章节：附录 A"


def test_docx_blank_section_falls_back_to_file_name():
    """空白标题不能渲染成「第 3 节：」这样的半截标签。"""
    metadata = {
        "source": "y.docx",
        "chunk_id": 0,
        "section": "   ",
        "section_index": 2,
    }
    assert source_line(build_one(metadata)) == "来源：y.docx（片段 0）"


def test_docx_section_title_kept_verbatim_after_strip():
    metadata = {"source": "y.docx", "chunk_id": 0,
                "section": "  第二章 设计  ", "section_index": 1}
    assert source_line(build_one(metadata)) == \
        "来源：y.docx（片段 0）｜第 2 节：第二章 设计"


# ----------------------------------------------------------------------
# D. DOCX 表格
# ----------------------------------------------------------------------

def test_docx_table_index_is_displayed_one_based():
    metadata = {
        "source": "需求文档.docx",
        "chunk_id": 0,
        "table_index": 0,
        "table_rows": 5,
        "table_cols": 3,
        "table_ragged": False,
    }
    assert source_line(build_one(metadata)) == \
        "来源：需求文档.docx（片段 0）｜表格 1"


def test_docx_table_index_two_renders_third_table():
    """table_index 是「文档内第几个表格」，空表格也占号，故 +1 即真实次序。"""
    metadata = {"source": "y.docx", "chunk_id": 1, "table_index": 2}
    assert source_line(build_one(metadata)) == "来源：y.docx（片段 1）｜表格 3"


def test_docx_section_and_table_read_their_own_metadata():
    """同一文件里章节块与表格块并存时，各自只显示自己的定位。"""
    results = [
        {"document": Document("章节正文", {
            "source": "y.docx", "chunk_id": 0, "chunk_seq": 0,
            "section": "第一章", "section_index": 0, "segment_index": 0,
        }), "vector": [1.0], "score": 0.9},
        {"document": Document("表格正文", {
            "source": "y.docx", "chunk_id": 0, "chunk_seq": 1,
            "table_index": 0, "table_rows": 2, "table_cols": 2,
        }), "vector": [1.0], "score": 0.8},
    ]
    context = ContextBuilder().build(results)
    lines = [line for line in context.splitlines() if line.startswith("来源：")]
    assert lines == [
        "来源：y.docx（片段 0）｜第 1 节：第一章",
        "来源：y.docx（片段 0）｜表格 1",
    ]


# ----------------------------------------------------------------------
# E. 缺失字段与异常类型
# ----------------------------------------------------------------------

def test_no_location_fields_adds_no_separator():
    """没有任何定位字段时不得出现分隔符或空标签。"""
    context = build_one({"source": "A.txt", "chunk_id": 0,
                         "document_id": "abc", "chunk_seq": 7})
    assert "｜" not in context
    assert source_line(context) == "来源：A.txt（片段 0）"


def test_bool_is_not_treated_as_index():
    """True / False 在 Python 里是 int 子类，但不能当页码或序号。"""
    metadata = {"source": "x.pdf", "chunk_id": 0, "page": True,
                "page_count": True, "table_index": False}
    assert source_line(build_one(metadata)) == "来源：x.pdf（片段 0）"


def test_negative_index_falls_back_to_less_specific_label():
    """section_index = -1 不可信 → 退化为只显示标题，不渲染「第 0 节」。"""
    metadata = {"source": "y.docx", "chunk_id": 0,
                "section": "第一章", "section_index": -1}
    assert source_line(build_one(metadata)) == \
        "来源：y.docx（片段 0）｜章节：第一章"


def test_negative_table_index_is_not_displayed():
    metadata = {"source": "y.docx", "chunk_id": 0, "table_index": -1}
    assert source_line(build_one(metadata)) == "来源：y.docx（片段 0）"


def test_string_index_is_not_coerced():
    """字符串 "0" 不能被当成序号 0（否则会渲染出凭空的「第 1 节」）。"""
    metadata = {"source": "y.docx", "chunk_id": 0,
                "section": "第一章", "section_index": "0"}
    assert source_line(build_one(metadata)) == \
        "来源：y.docx（片段 0）｜章节：第一章"


def test_none_metadata_values_are_ignored():
    metadata = {"source": "x.pdf", "chunk_id": 0,
                "page": None, "page_count": None,
                "section": None, "table_index": None}
    assert source_line(build_one(metadata)) == "来源：x.pdf（片段 0）"


def test_mixed_sources_keep_order_and_marker_count():
    """混合格式：顺序不变、每条恰好一个标记、标签互不串用。"""
    results = [
        {"document": Document("纯文本内容", {
            "source": "a.txt", "chunk_id": 0,
        }), "vector": [1.0], "score": 0.95},
        {"document": Document("PDF 第二页内容", {
            "source": "b.pdf", "chunk_id": 0, "page": 2, "page_count": 4,
        }), "vector": [1.0], "score": 0.9},
        {"document": Document("DOCX 章节内容", {
            "source": "c.docx", "chunk_id": 0,
            "section": "第一章", "section_index": 0,
        }), "vector": [1.0], "score": 0.85},
        {"document": Document("DOCX 表格内容", {
            "source": "c.docx", "chunk_id": 0, "table_index": 1,
        }), "vector": [1.0], "score": 0.8},
    ]
    context = ContextBuilder().build(results)

    assert context.count("【相关资料】") == 4
    lines = [line for line in context.splitlines() if line.startswith("来源：")]
    assert lines == [
        "来源：a.txt（片段 0）",
        "来源：b.pdf（片段 0）｜第 2 页（共 4 页）",
        "来源：c.docx（片段 0）｜第 1 节：第一章",
        "来源：c.docx（片段 0）｜表格 2",
    ]
    positions = [context.find(r["document"].page_content) for r in results]
    assert positions == sorted(positions) and -1 not in positions


# ----------------------------------------------------------------------
# F. 预算与截断：锁定「当前不存在」这一事实
# ----------------------------------------------------------------------
# 说明：ContextBuilder 目前没有长度预算、没有截断、也不做排序。以下用例
# 不是「验证预算仍然生效」，而是锁定该现状——将来若有人加入截断或排序，
# 这些断言会立刻失败，从而必须显式修改契约，而不是悄悄丢内容。

def test_build_does_not_truncate_large_context():
    """40 条 × 200 字全部保留，标记数 = 40，总长度不小于内容长度之和。"""
    content_unit = "长内容片段。" * 40            # 200 字
    results = [
        {"document": Document(f"{content_unit}#{i}", {
            "source": f"doc{i}.txt", "chunk_id": 0,
        }), "vector": [1.0], "score": 0.99 - i * 0.01}
        for i in range(40)
    ]
    context = ContextBuilder().build(results)

    assert context.count("【相关资料】") == 40
    for r in results:
        assert r["document"].page_content in context
    assert len(context) >= 40 * len(content_unit)


def test_build_does_not_reorder_by_score():
    """输入顺序即输出顺序：本类不排序，排序在上游 VectorStore.search。"""
    results = [
        {"document": Document(f"内容{name}", {"source": f"{name}.txt",
                                            "chunk_id": 0}),
         "vector": [1.0], "score": score}
        for name, score in (("低分", 0.11), ("高分", 0.98), ("中分", 0.55))
    ]
    context = ContextBuilder().build(results)
    positions = [context.find(r["document"].page_content) for r in results]
    assert positions == sorted(positions)


def test_location_label_only_adds_characters_to_source_line():
    """定位标签只加在来源行；内容行、相似度行、行数都不变。"""
    content = "同一段正文"
    score = 0.66
    bare = build_one({"source": "x.pdf", "chunk_id": 0},
                     content=content, score=score)
    labeled = build_one({"source": "x.pdf", "chunk_id": 0,
                         "page": 2, "page_count": 5},
                        content=content, score=score)

    assert len(labeled) > len(bare)
    assert labeled.splitlines()[2] == bare.splitlines()[2]      # 相似度行
    assert labeled.splitlines()[3] == bare.splitlines()[3]      # 内容行
    assert len(labeled.splitlines()) == len(bare.splitlines())
    assert labeled.replace("｜第 2 页（共 5 页）", "") == bare


# ----------------------------------------------------------------------
# G. 端到端：真实 Loader → KnowledgeBase → Context
# ----------------------------------------------------------------------
# 上面的用例都是手工构造 metadata，只证明「给定这些键就能正确渲染」。
# 下面两条走真实文件：PDF / DOCX 经 Loader 解析、切块、入索引，再检索取回
# Context，用来证明页号与章节标题确实来自 Loader 的解析结果，而不是测试
# 自己拼出来的字符串。

class _LengthEmbedding:
    """只依赖文本长度的确定性编码器。

    PDF 样本由 Helvetica 内建字体 + latin-1 生成，只能是 ASCII；靠中文
    关键词计数的编码器会返回零向量并被过滤掉，因此这里用长度构造向量。
    """

    def encode(self, text):
        return [float(len(text)), 1.0, 1.0]


def _source_lines(context):
    return [line for line in context.splitlines() if line.startswith("来源：")]


def test_end_to_end_pdf_context_shows_loader_page_numbers(tmp_path):
    # 惰性导入：保持本文件在脚本模式下不强制依赖 pypdf / python-docx。
    from rag.knowledge_base import KnowledgeBase
    from test_knowledge_base_loader_integration import build_pdf

    (tmp_path / "manual.pdf").write_bytes(build_pdf([
        "Page one about the project",
        "Page two about the project",
    ]))
    kb = KnowledgeBase(str(tmp_path),
                       embedding_factory=lambda: _LengthEmbedding())
    result = kb.search("anything", top_k=2)

    assert result["ok"] is True and result["count"] == 2

    lines = _source_lines(result["context"])
    assert set(lines) == {
        "来源：manual.pdf（片段 0）｜第 1 页（共 2 页）",
        "来源：manual.pdf（片段 0）｜第 2 页（共 2 页）",
    }
    # 页码顺序与文档物理顺序一致
    assert result["context"].index("第 1 页") < result["context"].index("第 2 页")


def test_end_to_end_docx_context_shows_loader_section_titles(tmp_path):
    from rag.knowledge_base import KnowledgeBase
    from test_knowledge_base_loader_integration import write_docx

    def build(document):
        document.add_heading("第一章 概述", level=1)
        document.add_paragraph("概述正文，描述项目背景。")
        document.add_heading("第二章 设计", level=1)
        document.add_paragraph("设计正文，描述模块划分。")

    write_docx(tmp_path / "manual.docx", build)
    kb = KnowledgeBase(str(tmp_path),
                       embedding_factory=lambda: _LengthEmbedding())
    result = kb.search("anything", top_k=2)

    assert result["ok"] is True and result["count"] == 2

    lines = _source_lines(result["context"])
    assert set(lines) == {
        "来源：manual.docx（片段 0）｜第 1 节：第一章 概述",
        "来源：manual.docx（片段 0）｜第 2 节：第二章 设计",
    }


# ----------------------------------------------------------------------
# 脚本模式（直接运行：打印结果并返回退出码）
# ----------------------------------------------------------------------

def run_checks():
    checks = []

    def record(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    # 1. 多个结果
    results = make_results()
    context = ContextBuilder().build(results)

    all_content_in = all(r["document"].page_content in context for r in results)
    all_score_in = all(f"相似度：{r['score']}" in context for r in results)
    positions = [context.find(r["document"].page_content) for r in results]
    in_order = positions == sorted(positions) and -1 not in positions
    markers = context.count("【相关资料】")

    record("3 个 Chunk 的内容全部进入 Context", all_content_in)
    record("3 个 score 全部进入 Context", all_score_in)
    record("顺序与输入结果一致", in_order, f"位置={positions}")
    record("分隔标记数量 = 3", markers == 3, f"实际={markers}")

    # 2. 空结果
    try:
        empty = ContextBuilder().build([])
        record("空结果返回空字符串且不报错", empty == "", repr(empty))
    except Exception as exc:  # noqa: BLE001 —— 验证“不报错”，异常即失败
        record("空结果返回空字符串且不报错", False, f"异常：{exc!r}")

    # 3. 特殊文本
    special = "Hello，世界！这是一个 test 用例（含标点）。"
    sp_context = ContextBuilder().build([{
        "document": Document(special, {"source": "S.txt"}),
        "vector": [1.0],
        "score": 0.5,
    }])
    record("中文+英文+标点的 Chunk 原样保留", special in sp_context)

    # 4. 各格式来源标签
    label_cases = [
        ("TXT/MD 不含定位标签",
         {"source": "faq.md", "chunk_id": 2},
         "来源：faq.md（片段 2）"),
        ("PDF 显示页码与总页数",
         {"source": "x.pdf", "chunk_id": 0, "page": 2, "page_count": 5},
         "来源：x.pdf（片段 0）｜第 2 页（共 5 页）"),
        ("PDF 缺总页数时只显示页码",
         {"source": "x.pdf", "chunk_id": 0, "page": 3},
         "来源：x.pdf（片段 0）｜第 3 页"),
        ("DOCX 章节序号按 1 基展示",
         {"source": "y.docx", "chunk_id": 0,
          "section": "第一章 概述", "section_index": 0},
         "来源：y.docx（片段 0）｜第 1 节：第一章 概述"),
        ("DOCX 表格序号按 1 基展示",
         {"source": "y.docx", "chunk_id": 0, "table_index": 0},
         "来源：y.docx（片段 0）｜表格 1"),
        ("缺 chunk_id 时不伪造片段号",
         {"source": "A.txt"},
         "来源：A.txt"),
    ]
    for name, metadata, expected in label_cases:
        actual = source_line(build_one(metadata))
        record(name, actual == expected, f"实际={actual!r}")

    # 5. 缺失与异常不伪造定位
    bad_cases = [
        ("page=True 不当作页码",
         {"source": "x.pdf", "chunk_id": 0, "page": True, "page_count": True},
         "来源：x.pdf（片段 0）"),
        ("page=0 不当作页码",
         {"source": "x.pdf", "chunk_id": 0, "page": 0},
         "来源：x.pdf（片段 0）"),
        ("负 table_index 不显示",
         {"source": "y.docx", "chunk_id": 0, "table_index": -1},
         "来源：y.docx（片段 0）"),
        ("字符串序号不强转（退化为只显示标题）",
         {"source": "y.docx", "chunk_id": 0,
          "section": "第一章", "section_index": "0"},
         "来源：y.docx（片段 0）｜章节：第一章"),
    ]
    for name, metadata, expected in bad_cases:
        actual = source_line(build_one(metadata))
        record(name, actual == expected, f"实际={actual!r}")

    # 6. 无预算 / 无截断 / 无排序
    bulk = [
        {"document": Document(f"内容{i}" + "填充" * 100, {
            "source": f"d{i}.txt", "chunk_id": 0,
        }), "vector": [1.0], "score": 0.5}
        for i in range(40)
    ]
    bulk_context = ContextBuilder().build(bulk)
    record("40 条结果一条不丢", bulk_context.count("【相关资料】") == 40)
    record("大 Context 未被截断",
           all(r["document"].page_content in bulk_context for r in bulk))

    all_passed = all(ok for _, ok, _ in checks)

    print("=" * 60)
    print("RAG 6.8 ContextBuilder 验证")
    print("=" * 60)
    print("---- 多结果 Context 预览 ----")
    print(context.replace("\n\n", "\n").strip())
    print("-" * 60)
    print("---- 混合来源标签预览 ----")
    print(ContextBuilder().build([
        {"document": Document("纯文本内容", {"source": "a.txt", "chunk_id": 0}),
         "vector": [1.0], "score": 0.95},
        {"document": Document("PDF 第二页内容", {
            "source": "b.pdf", "chunk_id": 0, "page": 2, "page_count": 4,
        }), "vector": [1.0], "score": 0.90},
        {"document": Document("DOCX 章节内容", {
            "source": "c.docx", "chunk_id": 0,
            "section": "第一章", "section_index": 0,
        }), "vector": [1.0], "score": 0.85},
        {"document": Document("DOCX 表格内容", {
            "source": "c.docx", "chunk_id": 0, "table_index": 1,
        }), "vector": [1.0], "score": 0.80},
    ]).replace("\n\n", "\n").strip())
    print("-" * 60)
    for name, ok, detail in checks:
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {name}" + (f"  （{detail}）" if detail else ""))
    print("=" * 60)
    print("全部验证通过" if all_passed else "存在失败项，请查看上方 FAIL 行")
    return all_passed


def main():
    return 0 if run_checks() else 1


if __name__ == "__main__":
    sys.exit(main())
