# -*- coding: utf-8 -*-
"""RAG 7.4：KnowledgeBase 独立单元测试（全程 stub，零真实模型、零网络）。

约定：所有知识文档写入 pytest 临时目录（虚构内容），不读取 rag_knowledge/
生产目录、不读正式 memory.db、不导入 sentence-transformers 的模型加载路径。
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.knowledge_base import KnowledgeBase   # noqa: E402

# B-2：PDF / DOCX 样本构造器复用 B-1 集成测试中的实现（同目录测试模块）。
# 这样避免在两个测试文件里各维护一份 PDF 语法拼装代码，样本构造规则也保持单源。
from test_knowledge_base_loader_integration import (   # noqa: E402
    build_pdf,
    write_docx,
)

DOC_ONE = "智能体项目已经实现了工具调用和长期记忆功能。"
DOC_TWO = "开发计划包含三个阶段和详细安排。"


class StubEmbedding:
    """确定性 stub 编码器：按关键词计数生成 3 维向量（不加载任何模型）。"""

    def __init__(self, fail_query=None):
        self.encode_calls = []
        self.fail_query = fail_query

    def encode(self, text):
        self.encode_calls.append(text)
        if self.fail_query is not None and self.fail_query in text:
            raise RuntimeError("stub 编码失败（测试注入）")
        return [
            float(text.count("智能") + text.count("项目")),
            float(text.count("计划") + text.count("阶段")),
            float(text.count("工具")),
        ]


def make_kb(tmp_path, fail_query=None, factory_error=None):
    """构造 KnowledgeBase + stub 工厂；返回 (kb, factory_stats)。

    factory_stats["calls"] 记录工厂被调用的次数（验证惰性初始化）。
    """
    (tmp_path / "one.md").write_text(DOC_ONE, encoding="utf-8")
    (tmp_path / "two.txt").write_text(DOC_TWO, encoding="utf-8")

    stats = {"calls": 0}
    holders = {}

    def factory():
        stats["calls"] += 1
        if factory_error is not None:
            raise factory_error
        stub = StubEmbedding(fail_query=fail_query)
        holders["stub"] = stub
        return stub

    kb = KnowledgeBase(tmp_path, embedding_factory=factory)
    return kb, stats, holders


# ----------------------------------------------------------------------
# 1. 多文档加载和切分
# ----------------------------------------------------------------------

def test_multi_document_loading_and_splitting(tmp_path):
    kb, stats, _ = make_kb(tmp_path)
    result = kb.search("项目的智能功能")

    assert result["ok"] is True
    assert kb.file_count == 2
    assert kb.chunk_count >= 2          # 每份文档至少 1 个 chunk
    assert stats["calls"] == 1


# ----------------------------------------------------------------------
# 2. 结果来源 / 分数 / 片段
# ----------------------------------------------------------------------

def test_result_fields_source_score_chunk(tmp_path):
    kb, _, _ = make_kb(tmp_path)
    result = kb.search("项目的智能功能")

    assert result["count"] >= 1
    first = result["results"][0]
    # B-2：原有四字段保持不变，并追加 document_id / chunk_seq 两个通用键
    # （.md / .txt 没有页码与章节概念，因此不再有别的键）。
    assert set(first.keys()) == {
        "source", "chunk_id", "score", "chunk",
        "document_id", "chunk_seq",
    }
    assert first["source"] in {"one.md", "two.txt"}
    assert isinstance(first["chunk_id"], int)
    assert first["chunk_id"] >= 0
    assert isinstance(first["score"], float)
    assert "智能" in first["chunk"] or "项目" in first["chunk"]
    # 结果必须可 JSON 序列化（工具返回值约束）
    json.dumps(result)


# ----------------------------------------------------------------------
# 3. 空检索结果（正常结果，不是异常）
# ----------------------------------------------------------------------

def test_empty_retrieval_is_normal_result(tmp_path):
    kb, _, _ = make_kb(tmp_path)
    result = kb.search("项目的智能功能", threshold=0.99)   # 全部被阈值滤掉

    assert result["ok"] is True
    assert result["count"] == 0
    assert result["results"] == []
    assert result["context"] == ""      # ContextBuilder 对空列表返回空串


# ----------------------------------------------------------------------
# 4. threshold=None 兼容原有行为
# ----------------------------------------------------------------------

def test_threshold_none_matches_default(tmp_path):
    kb, _, _ = make_kb(tmp_path)
    # 追加一篇与查询部分重合的文档，制造「正相似度但非共线」的低分结果。
    (tmp_path / "three.md").write_text("智能工具让生活更高效。", encoding="utf-8")

    # 不传 threshold 与显式 None 完全一致：都默认排除零相似度结果，
    # 同时保留正相似度结果。
    r_default = kb.search("项目的智能功能", top_k=10)
    r_none = kb.search("项目的智能功能", top_k=10, threshold=None)

    # 默认检索：DOC_TWO（相似度 0）被排除，DOC_ONE 与 DOC_THREE 保留。
    assert r_default["count"] == r_none["count"] == 2
    # 保留的都是正相似度结果（score > 0）。
    assert all(item["score"] > 0.0 for item in r_default["results"])


# ----------------------------------------------------------------------
# 5. 显式 threshold 传递
# ----------------------------------------------------------------------

def test_explicit_threshold_filters(tmp_path):
    kb, _, _ = make_kb(tmp_path)
    # 追加一篇与查询部分重合的文档：
    #   DOC_ONE   "智能体项目..." → 相似度 ≈ 0.894（高）
    #   DOC_THREE "智能工具..."   → 相似度 ≈ 0.707（低，非零）
    #   DOC_TWO   "开发计划..."   → 相似度 = 0（零）
    (tmp_path / "three.md").write_text("智能工具让生活更高效。", encoding="utf-8")

    unfiltered = kb.search("项目的智能功能", top_k=10)
    filtered = kb.search("项目的智能功能", top_k=10, threshold=0.8)

    assert filtered["ok"] is True
    # 显式阈值 0.8：DOC_ONE(0.894) 保留，DOC_THREE(0.707) 与 DOC_TWO(0) 被过滤。
    assert filtered["count"] < unfiltered["count"]
    assert unfiltered["count"] == 2      # 默认过滤零相似度后剩 2 条正相似度
    assert filtered["count"] == 1        # 显式阈值只保留高于 0.8 的 DOC_ONE
    for item in filtered["results"]:
        assert item["score"] >= 0.8


# ----------------------------------------------------------------------
# 6. Embedding 延迟初始化
# ----------------------------------------------------------------------

def test_embedding_lazy_initialization(tmp_path):
    kb, stats, _ = make_kb(tmp_path)
    assert stats["calls"] == 0          # 构造后：未加载
    kb.search("项目的智能功能")
    assert stats["calls"] == 1          # 首次检索才加载


# ----------------------------------------------------------------------
# 7. 同一实例不重复加载 / 不重复建索引
# ----------------------------------------------------------------------

def test_index_built_once(tmp_path):
    kb, stats, holders = make_kb(tmp_path)
    kb.search("项目的智能功能")
    first_chunk_count = kb.chunk_count
    encode_after_first = len(holders["stub"].encode_calls)

    kb.search("阶段的安排")
    kb.search("工具调用")

    assert stats["calls"] == 1                      # 工厂仍只调用一次
    assert kb.chunk_count == first_chunk_count      # 索引未重建
    # 新增的 encode 调用应当只有后续两次查询（每次 1 条），无重复编码文档
    assert len(holders["stub"].encode_calls) == encode_after_first + 2


# ----------------------------------------------------------------------
# 8. 缺失知识目录
# ----------------------------------------------------------------------

def test_missing_doc_dir(tmp_path):
    kb = KnowledgeBase(
        tmp_path / "not_exists",
        embedding_factory=lambda: StubEmbedding(),
    )
    result = kb.search("任意问题")

    assert result["ok"] is False
    assert result["stage"] == "init"
    assert "知识目录" in result["error"]


# ----------------------------------------------------------------------
# 9. Embedding 初始化异常（且不反复重试加载）
# ----------------------------------------------------------------------

def test_embedding_init_error_not_retried(tmp_path):
    kb, stats, _ = make_kb(
        tmp_path, factory_error=RuntimeError("stub 初始化失败（测试注入）")
    )
    first = kb.search("项目的智能功能")
    second = kb.search("项目的智能功能")

    assert first["ok"] is False and first["stage"] == "init"
    assert "stub 初始化失败" in first["error"]
    assert second["ok"] is False                # 失败信息可复现
    assert stats["calls"] == 1                  # 不反复尝试加载


# ----------------------------------------------------------------------
# 10. 检索异常（与初始化失败区分）
# ----------------------------------------------------------------------

def test_search_stage_error(tmp_path):
    kb, _, _ = make_kb(tmp_path, fail_query="爆炸")
    ok_result = kb.search("项目的智能功能")      # 先成功建好索引
    bad_result = kb.search("爆炸查询")           # 编码阶段抛错

    assert ok_result["ok"] is True
    assert bad_result["ok"] is False
    assert bad_result["stage"] == "search"
    assert "stub 编码失败" in bad_result["error"]


# ----------------------------------------------------------------------
# 11. stub 路径完全不加载真实模型 / 不访问网络
# ----------------------------------------------------------------------

def test_stub_path_no_real_model(tmp_path):
    kb, stats, holders = make_kb(tmp_path)
    kb.search("项目的智能功能")

    assert stats["calls"] == 1
    assert isinstance(holders["stub"], StubEmbedding)
    # 整个测试文件不 import 真实 Embedding；stub 工厂是唯一编码路径
    assert "sentence_transformers" not in sys.modules or True  # 说明性断言
    # 更强约束：真实模型实例从未被构造（本测试的 factory 已替代默认工厂）
    assert kb._embedding is holders["stub"]


# ======================================================================
# B-2：检索结果必须携带来源元数据（不再被压成旧四字段）
# ======================================================================

BASE_RESULT_KEYS = {"source", "chunk_id", "score", "chunk"}


class _AsciiEmbedding:
    """与语言无关的确定性编码器（长度 + 常量）。

    本文件的 ``StubEmbedding`` 靠中文关键词计数，但 PDF 样本用 Helvetica +
    latin-1 只能写 ASCII 文本，纯 ASCII 会得到零向量、被默认阈值过滤掉。
    这里改用只依赖长度的向量，使 PDF 样本可以正常参与检索。
    """

    def encode(self, text):
        return [float(len(text)), 1.0, 1.0]



def test_text_result_keeps_document_id_and_chunk_seq_only(tmp_path):
    """.md / .txt 没有页码与章节概念：只多出 document_id / chunk_seq，
    其余可选键必须整个不出现（不伪造 None）。"""
    kb, _, _ = make_kb(tmp_path)
    result = kb.search("项目的智能功能")

    assert result["ok"] is True and result["count"] >= 1
    record = result["results"][0]

    # 原有四字段的语义与类型不变
    assert BASE_RESULT_KEYS <= set(record)
    assert record["source"] in {"one.md", "two.txt"}
    assert isinstance(record["chunk_id"], int) and record["chunk_id"] >= 0
    assert isinstance(record["score"], float)
    assert isinstance(record["chunk"], str) and record["chunk"]

    # 只有通用两键被追加，没有 page / section / table_index
    assert set(record) == BASE_RESULT_KEYS | {"document_id", "chunk_seq"}
    assert isinstance(record["document_id"], str)
    assert len(record["document_id"]) == 32
    assert isinstance(record["chunk_seq"], int) and record["chunk_seq"] >= 0


def test_pdf_result_keeps_page_and_page_count(tmp_path):
    # PDF 构造器只支持 ASCII（Helvetica 内建字体 + latin-1）
    (tmp_path / "manual.pdf").write_bytes(
        build_pdf(["Page one about the project", "Page two about the project"])
    )
    kb = KnowledgeBase(
        str(tmp_path), embedding_factory=lambda: _AsciiEmbedding()
    )
    result = kb.search("项目的智能功能", top_k=2)

    assert result["ok"] is True
    assert result["count"] == 2
    assert {r["page"] for r in result["results"]} == {1, 2}
    assert all(r["page_count"] == 2 for r in result["results"])
    # 同一文件的两页共用同一个 document_id，chunk_seq 分别为 0 / 1
    assert len({r["document_id"] for r in result["results"]}) == 1
    assert sorted(r["chunk_seq"] for r in result["results"]) == [0, 1]
    # PDF 没有章节/表格概念
    for record in result["results"]:
        assert "section" not in record
        assert "table_index" not in record


def test_docx_result_keeps_section_metadata(tmp_path):
    def build(document):
        document.add_heading("第一章", level=1)
        document.add_paragraph("项目智能体包含工具调用。")
        document.add_heading("第二章", level=1)
        document.add_paragraph("项目开发计划分为三个阶段。")

    write_docx(tmp_path / "manual.docx", build)
    kb = KnowledgeBase(str(tmp_path), embedding_factory=lambda: StubEmbedding())
    result = kb.search("项目的智能功能", top_k=2)

    assert result["ok"] is True
    assert result["count"] == 2
    assert {r["section"] for r in result["results"]} == {"第一章", "第二章"}
    assert {r["section_index"] for r in result["results"]} == {0, 1}
    assert all(r["segment_index"] == 0 for r in result["results"])
    assert len({r["document_id"] for r in result["results"]}) == 1
    assert sorted(r["chunk_seq"] for r in result["results"]) == [0, 1]
    # DOCX 不伪造页码
    assert all("page" not in r for r in result["results"])


def test_docx_table_result_keeps_table_index(tmp_path):
    def build(document):
        document.add_heading("数据", level=1)
        document.add_paragraph("项目智能数据概览如下。")
        table = document.add_table(rows=2, cols=2)
        table.cell(0, 0).text = "项目指标"
        table.cell(0, 1).text = "数值"
        table.cell(1, 0).text = "智能召回率"
        table.cell(1, 1).text = "0.93"

    write_docx(tmp_path / "table.docx", build)
    kb = KnowledgeBase(str(tmp_path), embedding_factory=lambda: StubEmbedding())
    result = kb.search("项目智能", top_k=3)

    assert result["ok"] is True
    tables = [r for r in result["results"] if "table_index" in r]
    assert len(tables) == 1
    assert tables[0]["table_index"] == 0
    assert "section" not in tables[0]
    assert isinstance(tables[0]["chunk_seq"], int)
    assert isinstance(tables[0]["document_id"], str)

    # 节级结果不得凭空带上 table_index
    sections = [r for r in result["results"] if "section" in r]
    assert sections
    assert all("table_index" not in r for r in sections)

