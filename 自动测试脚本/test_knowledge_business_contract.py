# -*- coding: utf-8 -*-
"""知识库业务契约离线测试；不加载真实模型、不联网、不触碰正式数据库。"""

import hashlib
import importlib
import sys
import types
from types import SimpleNamespace

import pytest

from agent import Agent
from rag.context_builder import ContextBuilder
from rag.document import Document
from rag.knowledge_base import KnowledgeBase
from tools import make_search_knowledge_tool


class StubEmbedding:
    def encode(self, text):
        return [1.0, 0.0] if "命中" in text else [0.0, 1.0]


def make_kb(tmp_path):
    (tmp_path / "faq.md").write_text("命中答案", encoding="utf-8")
    return KnowledgeBase(str(tmp_path), embedding_factory=StubEmbedding)


class EmptyKnowledgeBase:
    def search(self, query, top_k=3):
        return {
            "ok": True,
            "query": query,
            "count": 0,
            "results": [],
            "context": "",
        }


def test_context_includes_source_and_chunk_id():
    document = Document(
        "答案片段",
        {"source": "faq.md", "chunk_id": 2},
    )
    context = ContextBuilder().build([{"document": document, "score": 0.9}])
    assert "来源：faq.md（片段 2）" in context
    assert "答案片段" in context


def test_knowledge_search_returns_source_and_rejects_invalid_input(tmp_path):
    tool = make_search_knowledge_tool(make_kb(tmp_path))
    result = tool("命中", top_k=1)
    assert result["results"][0]["source"] == "faq.md"
    assert "chunk" in result["results"][0]

    with pytest.raises(ValueError, match="不能为空"):
        tool("  ")
    with pytest.raises(ValueError, match="1 到 10"):
        tool("命中", top_k=11)


def test_knowledge_search_result_contains_real_chunk_id_and_matching_text(tmp_path):
    prefix = "前置内容" * 25
    (tmp_path / "long.md").write_text(prefix + "命中目标片段", encoding="utf-8")
    kb = KnowledgeBase(str(tmp_path), embedding_factory=StubEmbedding)

    result = kb.search("命中", top_k=10)

    assert result["ok"] is True
    assert result["count"] == 1
    item = result["results"][0]

    # B-2 契约：四个基础字段 + KnowledgeBase 在索引期写入的两个文件级身份字段。
    # 本文档是 .md（TextLoader）→ metadata 只有 source，因此**没有** PDF 的
    # page/page_count、也没有 DOCX 的 section/section_index/segment_index/
    # table_index。这里用等值断言而非 issubset：多出任何键都说明透传越界或
    # 有人给缺失字段补了 None。
    assert set(item) == {
        "source",
        "chunk_id",
        "score",
        "chunk",
        "document_id",
        "chunk_seq",
    }

    assert item["source"] == "long.md"
    assert item["chunk_id"] == 1

    # document_id = sha256(文件名)[:32]，与上传层 compute_document_id 同算法；
    # 这里独立复算，避免只断言「存在」而失去结构约束。
    assert item["document_id"] == hashlib.sha256(b"long.md").hexdigest()[:32]

    # 该文件只产出 1 个 Loader Document（2 个 chunk），所以 chunk_seq 与
    # chunk_id 在这条数据上恰好同步为 1；两者语义不同，断言分开写。
    assert item["chunk_seq"] == 1

    assert item["chunk"] in result["context"]
    assert "命中目标片段" in item["chunk"]


def test_chunk_id_is_document_local_and_source_keeps_chunks_distinct(tmp_path):
    (tmp_path / "a.md").write_text("命中 A", encoding="utf-8")
    (tmp_path / "b.md").write_text("命中 B", encoding="utf-8")
    kb = KnowledgeBase(str(tmp_path), embedding_factory=StubEmbedding)

    result = kb.search("命中", top_k=10)

    pairs = {(item["source"], item["chunk_id"]) for item in result["results"]}
    assert pairs == {("a.md", 0), ("b.md", 0)}


def test_empty_retrieval_returns_no_sources(tmp_path):
    (tmp_path / "faq.md").write_text("没有目标词", encoding="utf-8")
    kb = KnowledgeBase(str(tmp_path), embedding_factory=StubEmbedding)

    result = kb.search("命中", top_k=1)

    assert result["ok"] is True
    assert result["count"] == 0
    assert result["results"] == []


@pytest.mark.parametrize(
    "top_k",
    [True, False, 1.0, "1", 0, -1, 11],
)
def test_search_knowledge_rejects_invalid_top_k(top_k):
    tool = make_search_knowledge_tool(EmptyKnowledgeBase())
    with pytest.raises(ValueError):
        tool("问题", top_k=top_k)


@pytest.mark.parametrize("top_k", [1, 10])
def test_search_knowledge_accepts_top_k_boundaries(top_k):
    tool = make_search_knowledge_tool(EmptyKnowledgeBase())
    result = tool("问题", top_k=top_k)
    assert result["count"] == 0
    assert result["results"] == []


def _history_with_sources(records):
    return SimpleNamespace(
        execution_history=[
            {"tool_name": "search_memory", "result": {"result": {}}},
            "malformed execution record",
            {
                "tool_name": "search_knowledge",
                "result": {
                    "success": True,
                    "result": {"results": records},
                },
            },
        ]
    )


def test_agent_source_extraction_validates_fields_and_keeps_highest_score():
    """场景 A：身份相同（document_id + chunk_seq 都相同）→ 合并，保留高分。

    同时守住原用例的其余意图：非法记录被丢弃、未知字段不输出、
    结果顺序按首次出现位置（合并后的记录停在原位，不被挪到末尾）。
    """
    records = [
        {
            "source": "faq.md",
            "chunk_id": 0,
            "score": 0.4,
            "chunk": "旧分数",
            "document_id": "doc-faq",
            "chunk_seq": 0,
            "ignored": "字段不应输出",
        },
        {
            "source": "faq.md",
            "chunk_id": 0,
            "score": 0.9,
            "chunk": "高分片段",
            "document_id": "doc-faq",
            "chunk_seq": 0,
        },
        {
            "source": "faq.md",
            "chunk_id": 1,
            "score": 0.8,
            "chunk": "另一片段",
            "document_id": "doc-faq",
            "chunk_seq": 1,
        },
        {"source": "missing-id", "score": 0.8, "chunk": "无编号"},
        {"source": "bool-score", "chunk_id": 2, "score": True, "chunk": "无效"},
        {"source": "bad-score", "chunk_id": 3, "score": "0.8", "chunk": "无效"},
        {"source": "bad-chunk", "chunk_id": 4, "score": 0.8, "chunk": 5},
    ]

    sources = Agent._knowledge_sources(_history_with_sources(records))

    # 前两条同身份 → 合并成 1 条且取 0.9；第三条身份不同 → 保留。
    # 列表等值断言同时锁住内容与顺序（第 0 位即首次出现的位置）。
    assert sources == [
        {
            "source": "faq.md",
            "chunk_id": 0,
            "score": 0.9,
            "chunk": "高分片段",
            "document_id": "doc-faq",
            "chunk_seq": 0,
        },
        {
            "source": "faq.md",
            "chunk_id": 1,
            "score": 0.8,
            "chunk": "另一片段",
            "document_id": "doc-faq",
            "chunk_seq": 1,
        },
    ]
    # 六键结构：既不允许丢掉身份字段，也不允许把 "ignored" 透传出来。
    assert all(
        set(item) == {
            "source",
            "chunk_id",
            "score",
            "chunk",
            "document_id",
            "chunk_seq",
        }
        for item in sources
    )


def test_agent_source_extraction_keeps_same_chunk_seq_across_documents():
    """场景 B：不同文件可以有相同的 chunk_seq（各自从 0 起），必须各留一条。"""
    records = [
        {
            "source": "a.md",
            "chunk_id": 0,
            "score": 0.9,
            "chunk": "文件A内容",
            "document_id": "doc-a",
            "chunk_seq": 0,
        },
        {
            "source": "b.md",
            "chunk_id": 0,
            "score": 0.8,
            "chunk": "文件B内容",
            "document_id": "doc-b",
            "chunk_seq": 0,
        },
    ]

    sources = Agent._knowledge_sources(_history_with_sources(records))

    assert [item["document_id"] for item in sources] == ["doc-a", "doc-b"]
    assert [item["chunk_seq"] for item in sources] == [0, 0]
    assert [item["chunk"] for item in sources] == ["文件A内容", "文件B内容"]


def test_agent_source_extraction_keeps_same_document_different_chunk_seq():
    """场景 C：同一 PDF 的相邻两页 source 与 chunk_id 都是 0，只有 chunk_seq 不同。

    这正是旧身份 (source, chunk_id) 会把第二页吞掉的场景，新身份必须两条都留。
    """
    records = [
        {
            "source": "report.pdf",
            "chunk_id": 0,
            "score": 0.9,
            "chunk": "第一页内容",
            "document_id": "doc-pdf",
            "chunk_seq": 0,
            "page": 1,
            "page_count": 2,
        },
        {
            "source": "report.pdf",
            "chunk_id": 0,
            "score": 0.7,
            "chunk": "第二页内容",
            "document_id": "doc-pdf",
            "chunk_seq": 1,
            "page": 2,
            "page_count": 2,
        },
    ]

    sources = Agent._knowledge_sources(_history_with_sources(records))

    assert [item["chunk_seq"] for item in sources] == [0, 1]
    assert [item["page"] for item in sources] == [1, 2]
    assert [item["chunk"] for item in sources] == ["第一页内容", "第二页内容"]


def test_agent_source_extraction_missing_identity_is_not_deduped():
    """场景 D：身份不完整时保留结果，不得退回 (source, chunk_id) 误删。

    三条记录的 source 与 chunk_id 完全相同，只要实现偷偷退回旧键，就会
    被合并成 1 条。缺失的字段也不许补 None（键必须整个不存在）。
    """
    records = [
        {
            "source": "faq.md",
            "chunk_id": 0,
            "score": 0.9,
            "chunk": "缺文档标识",
            "chunk_seq": 0,
        },
        {
            "source": "faq.md",
            "chunk_id": 0,
            "score": 0.8,
            "chunk": "缺块序号",
            "document_id": "doc-faq",
        },
        {
            "source": "faq.md",
            "chunk_id": 0,
            "score": 0.7,
            "chunk": "身份全缺",
        },
    ]

    sources = Agent._knowledge_sources(_history_with_sources(records))

    assert len(sources) == 3
    assert [item["chunk"] for item in sources] == [
        "缺文档标识",
        "缺块序号",
        "身份全缺",
    ]
    assert "document_id" not in sources[0]
    assert "chunk_seq" not in sources[1]
    assert "document_id" not in sources[2]
    assert "chunk_seq" not in sources[2]


def test_agent_source_extraction_handles_no_call_and_malformed_history():
    assert Agent._knowledge_sources(SimpleNamespace(execution_history=[])) == []
    assert Agent._knowledge_sources(SimpleNamespace(execution_history=[1, None, {}])) == []
    assert Agent._knowledge_sources(SimpleNamespace(execution_history="bad")) == []


#: PAI-AUTH-MVP：这些用例直接调用视图函数 main.chat(...)，绕过 FastAPI
#: 的依赖注入，因此必须显式提供当前用户，否则默认值是一个 Depends 对象。
STUB_USER = {"id": "test-user", "username": "test-admin", "role": "admin"}


def _import_main_without_loading_dotenv(monkeypatch):
    """在不加载真实 .env 的前提下取得一份全新的 main 模块。

    只允许「临时」替换 sys.modules["main"]：重新导入的模块仅通过返回值交给
    当前用例使用，函数返回前必须把全局注册还原为原对象。否则后续用例里
    字符串形式的 patch("main.xxx") 会解析到这个临时模块，而 TestClient /
    端点仍绑定原模块，造成 Mock 静默失效（R6-TEST-ISOLATION-FIX-001）。
    """
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    original = sys.modules.pop("main", None)
    try:
        return importlib.import_module("main")
    finally:
        # 导入成功、导入抛异常、留下半初始化模块，三种情况都恢复原注册状态；
        # 原本不存在 main 时，结束后也必须保持不存在。
        if original is not None:
            sys.modules["main"] = original
        else:
            sys.modules.pop("main", None)


def test_import_helper_restores_previous_registration(monkeypatch):
    """R6-3：存在旧注册时，辅助函数必须把该对象原样放回 sys.modules。

    用占位模块作为已知的"原对象"，使断言不依赖本次运行是否已导入 main。
    """
    real = sys.modules.get("main")
    marker = types.ModuleType("main")
    sys.modules["main"] = marker

    try:
        temp = _import_main_without_loading_dotenv(monkeypatch)

        assert temp is not marker, "辅助函数应返回一份重新导入的独立模块"
        assert sys.modules.get("main") is marker, "全局注册必须还原为原对象"

        # 临时模块仍需可被局部测试正常使用。
        assert hasattr(temp, "chat")
        assert hasattr(temp, "ChatRequest")
        assert temp.ChatRequest(message="问题").message == "问题"
    finally:
        if real is not None:
            sys.modules["main"] = real
        else:
            sys.modules.pop("main", None)


def _isolate_chat_memory(main_module, tmp_path, monkeypatch):
    """/chat 会先写会话记录：把该 main 实例的 Memory 锁进 tmp_path。

    不隔离的话 get_memory() 会打开工程根目录的正式 memory.db。
    """
    from memory import Memory

    memory = Memory(db_path=str(tmp_path / "memory.db"))

    monkeypatch.setattr(main_module, "get_memory", lambda: memory)

    return memory


def test_import_helper_keeps_main_absent_when_it_was_absent(monkeypatch):
    """R6-3：原本没有 main 时，辅助函数结束后也不得遗留 main 注册。"""
    real = sys.modules.pop("main", None)

    try:
        temp = _import_main_without_loading_dotenv(monkeypatch)

        assert hasattr(temp, "chat")
        assert "main" not in sys.modules, "原本不存在 main 时不得遗留注册"
    finally:
        if real is not None:
            sys.modules["main"] = real
        else:
            sys.modules.pop("main", None)


def test_chat_preserves_answer_and_returns_sources(monkeypatch, tmp_path):
    main = _import_main_without_loading_dotenv(monkeypatch)

    _isolate_chat_memory(main, tmp_path, monkeypatch)

    class FakeAgent:
        def run(self, message, conversation_id=None, *, user_id=None):
            assert message == "问题"
            return {
                "success": True,
                "answer": "答案",
                "sources": [
                    {
                        "source": "faq.md",
                        "chunk_id": 0,
                        "score": 0.9,
                        "chunk": "依据",
                    }
                ],
            }

    monkeypatch.setattr(main, "get_agent", lambda: FakeAgent())
    response = main.chat(main.ChatRequest(message="问题"), user=STUB_USER)

    assert response["success"] is True
    assert response["answer"] == "答案"
    assert response["sources"][0]["chunk_id"] == 0


@pytest.mark.parametrize(
    "agent_result",
    [
        {"success": True, "answer": "无来源"},
        "字符串答案",
        123,
    ],
)
def test_chat_returns_empty_sources_when_legacy_result_has_no_sources(
    monkeypatch, agent_result, tmp_path
):
    main = _import_main_without_loading_dotenv(monkeypatch)

    _isolate_chat_memory(main, tmp_path, monkeypatch)

    class FakeAgent:
        def run(self, message, conversation_id=None, *, user_id=None):
            return agent_result

    monkeypatch.setattr(main, "get_agent", lambda: FakeAgent())
    response = main.chat(main.ChatRequest(message="问题"), user=STUB_USER)

    assert response["sources"] == []


def test_chat_failure_keeps_legacy_fields_and_adds_empty_sources(
    monkeypatch, tmp_path
):
    main = _import_main_without_loading_dotenv(monkeypatch)

    _isolate_chat_memory(main, tmp_path, monkeypatch)

    class FakeAgent:
        def run(self, message, conversation_id=None, *, user_id=None):
            return {"success": False, "answer": "失败", "error": "工具失败"}

    monkeypatch.setattr(main, "get_agent", lambda: FakeAgent())
    response = main.chat(main.ChatRequest(message="问题"), user=STUB_USER)

    assert response["success"] is False
    assert response["answer"] == "执行失败：工具失败"
    assert response["error"] == "工具失败"
    assert response["sources"] == []


def test_knowledge_base_rejects_invalid_query_without_initializing(tmp_path):
    kb = make_kb(tmp_path)
    result = kb.search(" ")
    assert result == {
        "ok": False,
        "stage": "input",
        "error": "知识检索问题不能为空",
        "query": " ",
    }
    assert kb._embedding is None
    assert kb.file_count == 0


def test_chat_internal_error_does_not_leak(monkeypatch, tmp_path):
    """验证 /chat 内部异常响应已脱敏，不泄露异常原文（可能含密钥/路径）。"""
    from fastapi import HTTPException

    main = _import_main_without_loading_dotenv(monkeypatch)

    _isolate_chat_memory(main, tmp_path, monkeypatch)

    class BoomAgent:
        def run(self, message, conversation_id=None, *, user_id=None):
            raise RuntimeError("敏感 sk-secret-123 / C:/Users/Doro/secret")

    monkeypatch.setattr(main, "get_agent", lambda: BoomAgent())

    with pytest.raises(HTTPException) as exc_info:
        main.chat(main.ChatRequest(message="问题"), user=STUB_USER)

    assert exc_info.value.status_code == 502
    detail = exc_info.value.detail
    assert "sk-secret-123" not in str(detail)
    assert "C:/Users/Doro/secret" not in str(detail)
    assert "BoomAgent" not in str(detail)

