# -*- coding: utf-8 -*-
"""RAG 7.4：Agent × RAG 集成测试（FakeClient + stub，零真实模型、零网络）。

所有用例通过真实工具注册与执行路径（Agent → ToolRegistry → tools 工厂 →
KnowledgeBase/StubKB）验证集成；不调用云端 API、不加载真实 Embedding、
不读取正式 memory.db。
"""

import io
import contextlib
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent import Agent                      # noqa: E402
from memory import Memory                    # noqa: E402
from rag.knowledge_base import KnowledgeBase  # noqa: E402
from task import AgentState                  # noqa: E402

# B-2：PDF 样本构造器复用 B-1 集成测试中的实现（同目录测试模块），
# 保证「手工拼 PDF」的规则只有一份。
from test_knowledge_base_loader_integration import build_pdf  # noqa: E402

DUMMY = dict(api_key="dummy", base_url="https://example.invalid", model="dummy")

DOC_ONE = "智能体项目已经实现了工具调用和长期记忆功能。"


class StubEmbedding:
    """确定性 stub 编码器（零模型加载）。"""

    def encode(self, text):
        return [
            float(text.count("智能") + text.count("项目")),
            float(text.count("计划")),
            float(text.count("工具")),
        ]


class ScriptedClient:
    """脚本化假客户端：预置响应序列，耗尽后给出兜底计划响应（防崩溃）。"""

    def __init__(self, script):
        self._script = list(script)
        self.calls = []

    def _complete(self, messages):
        self.calls.append(messages)
        if self._script:
            return self._script.pop(0)
        return json.dumps({"plan": [{"description": "回答", "type": "reasoning"}]})

    @property
    def chat(self):
        outer = self

        class _C:
            class completions:
                @staticmethod
                def create(model=None, messages=None, **kw):
                    content = outer._complete(messages)

                    class _M:
                        pass
                    m = _M()
                    m.content = content

                    class _Ch:
                        message = m

                    class _R:
                        choices = [_Ch()]
                    return _R()
        return _C()


def build_agent_with_kb(tmp_path, monkeypatch, knowledge_base):
    monkeypatch.chdir(tmp_path)
    agent = Agent(**DUMMY, knowledge_base=knowledge_base)
    agent.memory.close()
    agent.memory = Memory(db_path=str(tmp_path / "memory.db"))
    return agent


def make_real_kb(tmp_path, factory_stats=None):
    """真 KnowledgeBase + stub 编码（不加载真实模型）。"""
    (tmp_path / "one.md").write_text(DOC_ONE, encoding="utf-8")

    def factory():
        if factory_stats is not None:
            factory_stats["calls"] = factory_stats.get("calls", 0) + 1
        return StubEmbedding()

    return KnowledgeBase(tmp_path, embedding_factory=factory)


class StubKB:
    """仅实现 search 的替身：用于空结果与失败路径。"""

    def __init__(self, result):
        self.result = result
        self.calls = 0

    def search(self, query, top_k=3, threshold=None):
        self.calls += 1
        return dict(self.result, query=query)


def run_tool_plan(agent, user_message, tool_params):
    script = [
        json.dumps({"should_remember": False, "memories": []}),
        json.dumps({"plan": [
            {"description": "检索项目知识", "type": "tool",
             "tool_name": "search_knowledge"},
            {"description": "回答", "type": "reasoning"},
        ]}),
        json.dumps(tool_params),
        "中间推理",
        "最终答案。",
    ]
    agent.client = ScriptedClient(script)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = agent.run(user_message, user_id="test-user")
    return result, buf.getvalue()


# ----------------------------------------------------------------------
# 1/2/3. 工具成功执行 / execution_history / Reasoning 收到片段
# ----------------------------------------------------------------------

def test_tool_success_history_and_reasoning(tmp_path, monkeypatch):
    kb = make_real_kb(tmp_path)
    agent = build_agent_with_kb(tmp_path, monkeypatch, kb)

    result, output = run_tool_plan(
        agent, "项目实现了什么", {"query": "项目的智能功能"})

    assert result["success"] is True
    assert result["sources"]
    assert result["sources"][0]["source"] == "one.md"
    assert isinstance(result["sources"][0]["chunk_id"], int)
    # B-2：归一化不再丢弃来源元数据；.md 只多出通用两键（无页码/章节）
    assert set(result["sources"][0]) == {
        "source", "chunk_id", "score", "chunk",
        "document_id", "chunk_seq",
    }
    assert "page" not in result["sources"][0]

    # 输出（含 execution_history 打印）中出现结果字段
    assert "search_knowledge" in output
    assert "one.md" in output          # 来源进入执行历史
    # Reasoning 阶段实际收到的提示词包含知识片段与来源
    reasoning_msg = agent.client.calls[3][0]["content"]
    assert "one.md" in reasoning_msg
    assert "智能" in reasoning_msg


# ----------------------------------------------------------------------
# 4. 空检索结果：正常结果、不触发工具失败重试
# ----------------------------------------------------------------------

def test_empty_result_is_not_failure(tmp_path, monkeypatch):
    kb = StubKB({"ok": True, "count": 0, "results": [], "context": ""})
    agent = build_agent_with_kb(tmp_path, monkeypatch, kb)

    result, output = run_tool_plan(
        agent, "问个知识库没有的问题", {"query": "不存在的内容"})

    assert result["success"] is True
    assert kb.calls == 1                # 未重试
    assert "❌" not in output           # 未进入失败分支
    assert "知识库中未找到相关内容" in output  # 空结果的明确提示进入历史


# ----------------------------------------------------------------------
# 5+10. 执行失败遵循既有错误处理；其他功能不受影响
# ----------------------------------------------------------------------

def test_tool_failure_follows_registry_and_agent_stays_usable(
        tmp_path, monkeypatch):
    kb = StubKB({"ok": False, "stage": "init", "error": "知识目录缺失或为空"})
    agent = build_agent_with_kb(tmp_path, monkeypatch, kb)

    # 直接层：工具执行失败 → registry 既有约定（success=False + error）
    state = AgentState("检索")
    state.task.start()
    with contextlib.redirect_stdout(io.StringIO()):
        failure = agent.execute_tool(
            state, "search_knowledge", {"query": "任何问题"})
    assert failure["success"] is False
    assert "知识检索失败" in failure["error"]
    assert "init" in failure["error"]

    # 其他功能不受影响：普通对话（无工具）正常完成
    agent.client = ScriptedClient([
        json.dumps({"should_remember": False, "memories": []}),
        json.dumps({"plan": [{"description": "回答", "type": "reasoning"}]}),
        "中间",
        "正常回答。",
    ])
    with contextlib.redirect_stdout(io.StringIO()):
        result = agent.run("你好", user_id="test-user")
    assert result["success"] is True


# ----------------------------------------------------------------------
# 6. 闲聊不初始化 Embedding
# ----------------------------------------------------------------------

def test_chitchat_does_not_init_embedding(tmp_path, monkeypatch):
    stats = {}
    kb = make_real_kb(tmp_path, factory_stats=stats)
    agent = build_agent_with_kb(tmp_path, monkeypatch, kb)

    agent.client = ScriptedClient([
        json.dumps({"should_remember": False, "memories": []}),
        json.dumps({"plan": [{"description": "回答", "type": "reasoning"}]}),
        "中间",
        "你好呀。",
    ])
    with contextlib.redirect_stdout(io.StringIO()):
        result = agent.run("今天天气不错", user_id="test-user")
    assert result["success"] is True
    assert stats.get("calls", 0) == 0    # 工厂未被调用
    assert kb._ready is False            # 索引未构建


# ----------------------------------------------------------------------
# 7. knowledge_base=None：不注册知识工具
# ----------------------------------------------------------------------

def test_none_knowledge_base_no_tool(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with contextlib.redirect_stdout(io.StringIO()):
        agent = Agent(**DUMMY)
    assert "search_knowledge" not in agent.tool_registry.tools
    assert "query_orders" in agent.tool_registry.tools
    assert len(agent.tool_registry.tools) == 5
    agent.memory.close()


# ----------------------------------------------------------------------
# 8. 个人记忆工具仍正常注册与执行
# ----------------------------------------------------------------------

def test_memory_tool_still_works(tmp_path, monkeypatch):
    kb = make_real_kb(tmp_path)
    agent = build_agent_with_kb(tmp_path, monkeypatch, kb)

    assert "search_memory" in agent.tool_registry.tools
    # 与既有测试相同模式：同一物理库（构造 cwd 与替换路径一致）
    agent.memory.save_memory(key="preference", value="喜欢黑咖啡不加糖",
                             category="preference", reason="集成测试语料", user_id="test-user")
    state = AgentState("我的偏好是什么")
    state.task.start()
    with contextlib.redirect_stdout(io.StringIO()):
        r = agent.execute_tool(state, "search_memory", {"query": "喜欢喝"})
    assert r["success"] is True
    assert r["result"]["count"] >= 1


# ----------------------------------------------------------------------
# 9. 知识结果不写入个人记忆库
# ----------------------------------------------------------------------

def test_knowledge_does_not_write_memory_db(tmp_path, monkeypatch):
    kb = make_real_kb(tmp_path)
    agent = build_agent_with_kb(tmp_path, monkeypatch, kb)

    run_tool_plan(agent, "项目实现了什么", {"query": "项目的智能功能"})

    fresh = Memory(db_path=str(tmp_path / "memory.db"))
    try:
        assert fresh.get_memories(user_id="test-user") == []   # 知识检索未写入任何记忆
    finally:
        fresh.close()


# ----------------------------------------------------------------------
# 11. 全程 stub：不加载真实模型
# ----------------------------------------------------------------------

def test_stub_only_no_real_model(tmp_path, monkeypatch):
    stats = {}
    kb = make_real_kb(tmp_path, factory_stats=stats)
    agent = build_agent_with_kb(tmp_path, monkeypatch, kb)
    assert isinstance(agent.knowledge_base, KnowledgeBase)

    run_tool_plan(agent, "项目实现了什么", {"query": "项目的智能功能"})
    assert stats.get("calls", 0) == 1       # 只经由 stub 工厂
    assert isinstance(kb._embedding, StubEmbedding)


# ----------------------------------------------------------------------
# B-2：来源元数据透传 + 文件级去重身份 (document_id, chunk_seq)
# ----------------------------------------------------------------------

class _HistoryState:
    """只含一条 search_knowledge 记录的最小 state 替身（直接喂 _knowledge_sources）。"""

    def __init__(self, records):
        self.execution_history = [
            {
                "tool_name": "search_knowledge",
                "result": {"result": {"results": records}},
            }
        ]


class _AsciiEmbedding:
    """与语言无关的确定性编码器（长度 + 常量）。

    本文件的 ``StubEmbedding`` 靠中文关键词计数，但 PDF 样本只能写 ASCII
    （Helvetica + latin-1），纯 ASCII 会得到零向量并被默认阈值过滤。
    """

    def encode(self, text):
        return [float(len(text)), 1.0, 1.0]


def _source_record(**overrides):
    record = {
        "source": "manual.pdf",
        "chunk_id": 0,
        "score": 0.5,
        "chunk": "片段内容",
        "document_id": "0" * 32,
        "chunk_seq": 0,
    }
    record.update(overrides)
    return record


def test_sources_keep_optional_metadata_fields():
    record = _source_record(page=2, page_count=3)
    sources = Agent._knowledge_sources(_HistoryState([record]))

    assert len(sources) == 1
    assert sources[0]["document_id"] == "0" * 32
    assert sources[0]["chunk_seq"] == 0
    assert sources[0]["page"] == 2
    assert sources[0]["page_count"] == 3
    # 未出现的可选键不得被补 None
    assert "section" not in sources[0]


def test_dedupe_same_identity_keeps_higher_score():
    """同一 (document_id, chunk_seq) 视为同一个块，保留高分那条。"""
    records = [
        _source_record(score=0.4, chunk="低分"),
        _source_record(score=0.9, chunk="高分"),
    ]
    sources = Agent._knowledge_sources(_HistoryState(records))

    assert len(sources) == 1
    assert sources[0]["chunk"] == "高分"
    assert sources[0]["score"] == 0.9


def test_same_document_different_chunk_seq_keeps_all():
    records = [
        _source_record(chunk_seq=0, chunk="第一块"),
        _source_record(chunk_seq=1, chunk="第二块"),
    ]
    sources = Agent._knowledge_sources(_HistoryState(records))

    assert [s["chunk"] for s in sources] == ["第一块", "第二块"]
    assert [s["chunk_seq"] for s in sources] == [0, 1]


def test_different_document_same_chunk_seq_keeps_all():
    records = [
        _source_record(document_id="a" * 32, chunk="文件 A"),
        _source_record(document_id="b" * 32, chunk="文件 B"),
    ]
    sources = Agent._knowledge_sources(_HistoryState(records))

    assert [s["chunk"] for s in sources] == ["文件 A", "文件 B"]


def test_same_source_and_chunk_id_different_chunk_seq_keeps_all():
    """旧键 (source, chunk_id) 会误删的关键场景：PDF 不同页的 chunk_id 同为 0。"""
    records = [
        _source_record(chunk_id=0, chunk_seq=0, page=1, chunk="第一页"),
        _source_record(chunk_id=0, chunk_seq=1, page=2, chunk="第二页"),
    ]
    sources = Agent._knowledge_sources(_HistoryState(records))

    assert len(sources) == 2
    assert [s["page"] for s in sources] == [1, 2]
    assert [s["chunk_id"] for s in sources] == [0, 0]


def test_missing_document_id_is_not_deduped():
    """缺 document_id 时不得退回旧键：两条即使 source/chunk_id 相同也都要保留。"""
    first = _source_record(chunk="第一条")
    second = _source_record(chunk="第二条")
    del first["document_id"]
    del second["document_id"]

    sources = Agent._knowledge_sources(_HistoryState([first, second]))

    assert [s["chunk"] for s in sources] == ["第一条", "第二条"]


def test_missing_chunk_seq_is_not_deduped():
    first = _source_record(chunk="第一条")
    second = _source_record(chunk="第二条")
    del first["chunk_seq"]
    del second["chunk_seq"]

    sources = Agent._knowledge_sources(_HistoryState([first, second]))

    assert [s["chunk"] for s in sources] == ["第一条", "第二条"]


def test_pdf_pages_with_same_chunk_id_are_all_kept():
    """PDF 多页模拟：source 相同、chunk_id 全为 0、page 与 chunk_seq 不同。"""
    records = [
        _source_record(page=1, chunk_id=0, chunk_seq=0, chunk="第一页正文"),
        _source_record(page=2, chunk_id=0, chunk_seq=1, chunk="第二页正文"),
        _source_record(page=3, chunk_id=0, chunk_seq=2, chunk="第三页正文"),
    ]
    sources = Agent._knowledge_sources(_HistoryState(records))

    assert len(sources) == 3
    assert [s["page"] for s in sources] == [1, 2, 3]
    assert [s["chunk_seq"] for s in sources] == [0, 1, 2]
    assert {s["chunk_id"] for s in sources} == {0}


def test_pdf_pages_survive_end_to_end_without_dedupe(tmp_path, monkeypatch):
    """端到端：KnowledgeBase.search() → Agent 归一化 → 最终 sources。

    这是 B-2 的核心回归——同一 PDF 的两页 chunk_id 都是 0，旧去重键会把
    它们合并成一条；改用 (document_id, chunk_seq) 后两页都必须保留。
    """
    (tmp_path / "manual.pdf").write_bytes(
        build_pdf(["Page one about the project", "Page two about the project"])
    )
    kb = KnowledgeBase(tmp_path, embedding_factory=lambda: _AsciiEmbedding())
    agent = build_agent_with_kb(tmp_path, monkeypatch, kb)

    result, _ = run_tool_plan(
        agent, "项目实现了什么", {"query": "项目的智能功能"})

    assert result["success"] is True
    sources = result["sources"]
    assert len(sources) == 2
    assert {s["page"] for s in sources} == {1, 2}
    assert all(s["page_count"] == 2 for s in sources)
    assert {s["chunk_id"] for s in sources} == {0}
    assert {s["chunk_seq"] for s in sources} == {0, 1}
    assert len({s["document_id"] for s in sources}) == 1

