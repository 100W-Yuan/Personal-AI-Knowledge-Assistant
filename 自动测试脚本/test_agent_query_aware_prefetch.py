# -*- coding: utf-8 -*-
"""P2-3D-C1-R2：查询感知的预检索复用（query-aware prefetch）离线测试。

覆盖这次修复的全部契约：

1. 相同 query + 相同 top_k          → 复用预检索，不重复检索
2. 不同 query                       → 进入真实 search
3. 相同 query + 不同 top_k          → 不复用不足量缓存，重新检索
4. 两个不同子查询                  → 都进入 execution_history
5. 相同子查询重复请求              → 不无限检索
6. 多次检索后 PDF page 元数据      → 保留
7. 多次检索后 DOCX section 元数据  → 保留
8. 多次检索后 DOCX table_index     → 保留
9. 相同 (document_id, chunk_seq)   → 不产生重复来源
10. 同身份高分更新                 → 不改变首次出现位置

组件真实性：
- PDF / DOCX 由 P2-3D-C1 验收脚本的构造函数真实生成（pypdf / python-docx）；
- 索引与检索走真实 ``DocxLoader`` / ``PdfLoader`` / ``TextSplitter`` /
  ``KnowledgeBase`` 与本地 ``BAAI/bge-small-zh-v1.5``（离线模式）；
- 不调用任何 DeepSeek / 网络 API；所有数据落在 pytest 独立临时目录。

本文件位于 自动测试脚本/ 目录，继承该目录 conftest.py 的离线隔离。
"""

import os
import sys
from pathlib import Path

# 离线环境必须在导入任何 rag / sentence_transformers 之前设置。
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pytest  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for _path in (str(PROJECT_ROOT), str(SCRIPT_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from agent import Agent                       # noqa: E402
from memory import Memory                     # noqa: E402
from task import AgentState                   # noqa: E402
from rag.knowledge_base import KnowledgeBase  # noqa: E402

# 复用 C1 验收脚本的真实文档构造器与事实常量，避免两份构造逻辑漂移。
import run_real_agent_rag_citation as citation  # noqa: E402

DUMMY = dict(api_key="dummy", base_url="https://example.invalid", model="dummy")

PDF_FILENAME = citation.PDF_FILENAME
DOCX_FILENAME = citation.DOCX_FILENAME
PDF_PAGES = citation.PDF_PAGES
QUESTION = citation.QUESTION          # 整句综合问题（生产预检索用的就是它）
DOCX_SECTION_EXPECTED = citation.DOCX_SECTION_EXPECTED
DOCX_TABLE_INDEX_EXPECTED = citation.DOCX_TABLE_INDEX_EXPECTED
PDF_PAGE_WITH_FACT = citation.PDF_PAGE_WITH_FACT

# 子查询：与 P2-3D-C1-R1 对照实验使用的短查询逐字一致。
Q_REPLACE = "滤芯更换周期是多少天"
Q_INSPECT = "滤芯巡检间隔是多少天"
Q_METHOD = "检测方式是什么"


# ======================================================================
# 真实组件夹具
# ======================================================================


class CountingKB:
    """包装真实 KnowledgeBase：代理全部属性，只对 search 计数。"""

    def __init__(self, kb):
        self._kb = kb
        self.calls = 0
        self.queries = []

    def search(self, query, top_k=3, threshold=None):
        self.calls += 1
        self.queries.append((query, top_k, threshold))
        return self._kb.search(query, top_k=top_k, threshold=threshold)

    def __getattr__(self, name):
        return getattr(self._kb, name)


@pytest.fixture(scope="module")
def real_kb(tmp_path_factory):
    """真实文档 + 真实索引 + 真实本地 Embedding（模块内只加载一次）。"""
    kb_dir = tmp_path_factory.mktemp("c1r2_kb")
    pdf_path = kb_dir / PDF_FILENAME
    docx_path = kb_dir / DOCX_FILENAME
    citation._build_pdf(pdf_path, PDF_PAGES)
    citation._build_docx(docx_path)

    kb = KnowledgeBase(str(kb_dir))

    # KnowledgeBase 是惰性索引：__init__ 只做字段初始化，首次检索才建索引。
    # 先预热一次，之后所有断言/检索都在已就绪的索引上进行。
    warmup = kb.search(Q_REPLACE, top_k=1)
    assert warmup["ok"] is True, f"预热检索失败：{warmup}"

    diag = {
        name: getattr(kb, name, "<missing>")
        for name in (
            "is_indexed",
            "file_count",
            "chunk_count",
            "empty_file_count",
            "skipped_count",
            "init_error_kind",
        )
    }
    print("\n[KB_DIAG]", diag)

    assert kb.is_indexed is True, f"真实知识库初始化失败：{diag}"
    assert kb.file_count == 2, f"应索引 2 个文件：{diag}"
    assert kb.chunk_count >= 4, f"chunk 数异常：{diag}"
    return kb


@pytest.fixture()
def counting_kb(real_kb):
    return CountingKB(real_kb)


@pytest.fixture()
def agent_factory(tmp_path, monkeypatch, counting_kb):
    """构造收敛到 search_knowledge 的 Agent，memory 隔离到 tmp_path。"""

    def _make(user_message=QUESTION):
        monkeypatch.chdir(tmp_path)
        agent = Agent(**DUMMY, knowledge_base=counting_kb)
        agent.memory.close()
        agent.memory = Memory(db_path=str(tmp_path / "memory.db"))

        # 只保留 search_knowledge：避免任何其他工具触达业务库。
        tools = agent.tool_registry.tools
        for name in list(tools):
            if name != "search_knowledge":
                del tools[name]
        assert list(tools) == ["search_knowledge"]

        return agent, AgentState(user_message)

    return _make


def _search(agent, state, query, top_k=None):
    args = {"query": query}
    if top_k is not None:
        args["top_k"] = top_k
    return agent.execute_tool(state, "search_knowledge", args)


def _history_searches(state):
    return [
        entry
        for entry in state.execution_history
        if entry.get("tool_name") == "search_knowledge"
    ]


def _identities(sources):
    return [(s.get("document_id"), s.get("chunk_seq")) for s in sources]


# ======================================================================
# 1. 相同 query + 相同 top_k → 复用
# ======================================================================


def test_same_query_same_top_k_reuses_prefetch(agent_factory):
    agent, state = agent_factory()

    first = _search(agent, state, Q_REPLACE, 3)
    assert first["success"] is True
    assert agent.knowledge_base.calls == 1
    assert len(_history_searches(state)) == 1

    second = _search(agent, state, Q_REPLACE, 3)
    assert second["success"] is True
    # 真实检索没有被再次调用，且返回的是历史中的同一对象。
    assert agent.knowledge_base.calls == 1
    assert second is first
    # 复用不追加历史。
    assert len(_history_searches(state)) == 1


def test_whitespace_variant_is_same_key(agent_factory):
    """只差空白字符的参数视为同一检索（规范化生效）。"""
    agent, state = agent_factory()

    _search(agent, state, Q_REPLACE, 3)
    assert agent.knowledge_base.calls == 1

    _search(agent, state, f"  {Q_REPLACE}  ", 3)
    assert agent.knowledge_base.calls == 1


def test_omitted_top_k_equals_default(agent_factory):
    """省略 top_k 与显式 top_k=3 等价（都按工具默认值 3 归一）。"""
    agent, state = agent_factory()

    _search(agent, state, Q_REPLACE, None)
    assert agent.knowledge_base.calls == 1

    _search(agent, state, Q_REPLACE, 3)
    assert agent.knowledge_base.calls == 1


# ======================================================================
# 2. 不同 query → 真实执行
# ======================================================================


def test_different_query_runs_real_search(agent_factory):
    agent, state = agent_factory()

    first = _search(agent, state, Q_REPLACE, 3)
    assert agent.knowledge_base.calls == 1

    second = _search(agent, state, Q_METHOD, 3)
    assert agent.knowledge_base.calls == 2
    assert second is not first
    assert len(_history_searches(state)) == 2

    # 两次检索的参数被分别记录，没有被合并成一条。
    recorded = [tuple(e["arguments"].items()) for e in _history_searches(state)]
    assert len(recorded) == 2
    assert recorded[0] != recorded[1]


# ======================================================================
# 3. 相同 query + 不同 top_k → 不复用
# ======================================================================


def test_same_query_different_top_k_not_reused(agent_factory):
    agent, state = agent_factory()

    small = _search(agent, state, QUESTION, 3)
    assert small["success"] is True
    assert agent.knowledge_base.calls == 1
    small_count = small["result"]["count"]

    large = _search(agent, state, QUESTION, 5)
    # top_k 不同 → 必须真实重检，不得拿 3 条冒充 5 条。
    assert agent.knowledge_base.calls == 2
    assert large is not small
    assert large["result"]["count"] > small_count
    assert len(_history_searches(state)) == 2


# ======================================================================
# 4/5. 历史累积与去重
# ======================================================================


def test_two_subqueries_both_enter_history(agent_factory):
    agent, state = agent_factory()

    # 模拟生产顺序：预检索（整句）→ 两个子查询。
    _search(agent, state, QUESTION, 3)
    _search(agent, state, Q_REPLACE, 3)
    _search(agent, state, Q_METHOD, 3)

    searches = _history_searches(state)
    assert len(searches) == 3
    assert agent.knowledge_base.calls == 3

    queries = [e["arguments"]["query"] for e in searches]
    assert queries == [QUESTION, Q_REPLACE, Q_METHOD]
    assert all(e["result"]["success"] is True for e in searches)


def test_repeated_subquery_does_not_loop(agent_factory):
    agent, state = agent_factory()

    for _ in range(5):
        result = _search(agent, state, Q_INSPECT, 3)
        assert result["success"] is True

    assert agent.knowledge_base.calls == 1
    assert len(_history_searches(state)) == 1


# ======================================================================
# 6/7/8 + 核心验收：三类来源在多次检索后并存
# ======================================================================


def test_core_acceptance_three_sources_merged(agent_factory):
    """P2-3D-C1-R2 第 7 节核心验收：三类来源都进入最终 sources。

    只用默认 top_k=3 的子查询，不靠放大 top_k 勉强通过。
    """
    agent, state = agent_factory()

    _search(agent, state, QUESTION, 3)      # 预检索（整句）
    _search(agent, state, Q_REPLACE, 3)     # PDF 第 2 页
    _search(agent, state, Q_INSPECT, 3)     # DOCX 章节
    _search(agent, state, Q_METHOD, 3)      # DOCX 表格

    sources = Agent._knowledge_sources(state)

    pdf_pages = [
        s.get("page")
        for s in sources
        if s.get("source") == PDF_FILENAME and isinstance(s.get("page"), int)
    ]
    assert PDF_PAGE_WITH_FACT in pdf_pages, f"缺少 PDF 第 2 页来源：{pdf_pages}"

    sections = [
        s.get("section")
        for s in sources
        if s.get("source") == DOCX_FILENAME and isinstance(s.get("section"), str)
    ]
    assert DOCX_SECTION_EXPECTED in sections, f"缺少 DOCX 章节来源：{sections}"

    tables = [
        s.get("table_index")
        for s in sources
        if s.get("source") == DOCX_FILENAME and isinstance(s.get("table_index"), int)
    ]
    assert DOCX_TABLE_INDEX_EXPECTED in tables, f"缺少 DOCX 表格来源：{tables}"

    # 四种检索都真实发生过。
    assert agent.knowledge_base.calls == 4
    assert len(_history_searches(state)) == 4


def test_pdf_page_metadata_survives_multiple_searches(agent_factory):
    agent, state = agent_factory()

    _search(agent, state, QUESTION, 3)
    _search(agent, state, Q_INSPECT, 3)
    _search(agent, state, Q_REPLACE, 3)

    sources = Agent._knowledge_sources(state)
    pdf_items = [s for s in sources if s.get("source") == PDF_FILENAME]
    assert pdf_items, "多次检索后 PDF 来源丢失"

    page_item = next(
        (s for s in pdf_items if s.get("page") == PDF_PAGE_WITH_FACT), None
    )
    assert page_item is not None
    assert page_item.get("page_count") == citation.PDF_PAGE_COUNT_EXPECTED


def test_docx_section_metadata_survives_multiple_searches(agent_factory):
    agent, state = agent_factory()

    _search(agent, state, QUESTION, 3)
    _search(agent, state, Q_METHOD, 3)
    _search(agent, state, Q_INSPECT, 3)

    sources = Agent._knowledge_sources(state)
    section_item = next(
        (
            s
            for s in sources
            if s.get("source") == DOCX_FILENAME
            and s.get("section") == DOCX_SECTION_EXPECTED
        ),
        None,
    )
    assert section_item is not None
    assert section_item.get("section_index") == 0
    assert section_item.get("segment_index") == 0


def test_docx_table_index_survives_multiple_searches(agent_factory):
    agent, state = agent_factory()

    _search(agent, state, Q_REPLACE, 3)
    _search(agent, state, Q_METHOD, 3)

    sources = Agent._knowledge_sources(state)
    table_item = next(
        (
            s
            for s in sources
            if s.get("source") == DOCX_FILENAME
            and s.get("table_index") == DOCX_TABLE_INDEX_EXPECTED
        ),
        None,
    )
    assert table_item is not None
    # DOCX 表格节本来就没有 section，不得伪造。
    assert "section" not in table_item, "表格来源被伪造了 section 字段"


# ======================================================================
# 9. 同身份不产生重复来源
# ======================================================================


def test_same_identity_not_duplicated(agent_factory):
    agent, state = agent_factory()

    # 两条语义相近的子查询会命中重叠 chunk。
    _search(agent, state, Q_INSPECT, 3)
    _search(agent, state, "滤芯巡检间隔多少天", 3)

    sources = Agent._knowledge_sources(state)
    identities = [i for i in _identities(sources) if i[0] is not None]
    assert identities, "没有产生可去重的身份，测试无意义"
    assert len(identities) == len(set(identities)), f"来源出现重复身份：{identities}"


# ======================================================================
# 10. 同身份高分更新不改变首次出现位置（纯聚合单元）
# ======================================================================


def _record(chunk_seq, score, **extra):
    base = {
        "source": DOCX_FILENAME,
        "chunk_id": 0,
        "score": score,
        "chunk": f"片段-{chunk_seq}",
        "document_id": "doc-1",
        "chunk_seq": chunk_seq,
    }
    base.update(extra)
    return base


def _tool_entry(records):
    return {
        "type": "tool",
        "tool_name": "search_knowledge",
        "arguments": {"query": "任意", "top_k": 3},
        "result": {
            "success": True,
            "result": {
                "query": "任意",
                "count": len(records),
                "results": records,
                "context": "",
                "refused": False,
            },
        },
    }


def test_high_score_update_keeps_first_position():
    state = AgentState(QUESTION)

    # 第一条历史：低分 0.30；第二条历史：同身份高分 0.90。
    state.execution_history.append(_tool_entry([_record(0, 0.30)]))
    state.execution_history.append(_tool_entry([_record(0, 0.90)]))

    sources = Agent._knowledge_sources(state)
    assert len(sources) == 1, "同身份没有被去重"
    assert sources[0]["score"] == 0.90, "高分未替换低分"
    assert sources[0]["chunk"] == "片段-0"


def test_first_position_is_preserved_across_updates():
    state = AgentState(QUESTION)

    # 首次顺序：A(seq=0) 在前，B(seq=1) 在后。
    state.execution_history.append(
        _tool_entry([_record(0, 0.40), _record(1, 0.20)])
    )
    # 第二次：A 变成高分，B 也更新；A 的位置必须仍在索引 0。
    state.execution_history.append(
        _tool_entry([_record(1, 0.60), _record(0, 0.95)])
    )

    sources = Agent._knowledge_sources(state)
    assert len(sources) == 2
    assert sources[0]["chunk_seq"] == 0, "首次出现位置被改变"
    assert sources[0]["score"] == 0.95
    assert sources[1]["chunk_seq"] == 1
    assert sources[1]["score"] == 0.60


# ======================================================================
# 附加：run() 层的多子查询端到端（离线脚本化 LLM，不调 API）
# ======================================================================


class _ScriptedClient:
    """脚本化假客户端：预置响应序列，耗尽即报错（防脚本长度误判）。"""

    def __init__(self, script):
        self._script = list(script)
        self.calls = 0

    def _next(self, messages):
        self.calls += 1
        if not self._script:
            raise AssertionError(
                f"脚本耗尽：LLM 调用次数超出预期（已 {self.calls} 次）"
            )
        return self._script.pop(0)

    @property
    def chat(self):
        outer = self

        class _Completions:
            @staticmethod
            def create(model=None, messages=None, **kwargs):
                content = outer._next(messages)

                class _Message:
                    pass

                message = _Message()
                message.content = content

                class _Choice:
                    pass

                choice = _Choice()
                choice.message = message

                class _Response:
                    pass

                response = _Response()
                response.choices = [choice]
                return response

        class _Chat:
            completions = _Completions

        return _Chat()


def test_run_executes_multiple_distinct_subqueries(real_kb, tmp_path, monkeypatch):
    """run() 内：预检索 + 三个不同子查询都真实执行，三类来源并存。"""
    import contextlib
    import io
    import json

    monkeypatch.chdir(tmp_path)
    counting = CountingKB(real_kb)
    agent = Agent(**DUMMY, knowledge_base=counting)
    agent.memory.close()
    agent.memory = Memory(db_path=str(tmp_path / "memory.db"))

    tools = agent.tool_registry.tools
    for name in list(tools):
        if name != "search_knowledge":
            del tools[name]

    plan = [
        {"description": "查更换周期", "type": "tool",
         "tool_name": "search_knowledge"},
        {"description": "查巡检间隔", "type": "tool",
         "tool_name": "search_knowledge"},
        {"description": "查检测方式", "type": "tool",
         "tool_name": "search_knowledge"},
    ]
    script = [
        json.dumps({"should_remember": False, "memories": []}),
        json.dumps({"plan": plan}),
        json.dumps({"query": Q_REPLACE, "top_k": 3}),
        json.dumps({"query": Q_INSPECT, "top_k": 3}),
        json.dumps({"query": Q_METHOD, "top_k": 3}),
        "最终答案。",
    ]
    agent.client = _ScriptedClient(script)

    with contextlib.redirect_stdout(io.StringIO()):
        result = agent.run(QUESTION, user_id="test-user")

    assert result["success"] is True
    assert result["answer"] == "最终答案。"
    # 预检索 1 次 + 三个不同子查询 3 次 = 4 次真实检索。
    assert counting.calls == 4, f"真实检索次数异常：{counting.calls}"

    sources = result["sources"]
    assert any(
        s.get("source") == PDF_FILENAME and s.get("page") == PDF_PAGE_WITH_FACT
        for s in sources
    ), f"缺少 PDF 第 2 页：{sources}"
    assert any(
        s.get("source") == DOCX_FILENAME
        and s.get("section") == DOCX_SECTION_EXPECTED
        for s in sources
    ), f"缺少 DOCX 章节：{sources}"
    assert any(
        s.get("source") == DOCX_FILENAME
        and s.get("table_index") == DOCX_TABLE_INDEX_EXPECTED
        for s in sources
    ), f"缺少 DOCX 表格：{sources}"
    assert result["retrieval"]["status"] == "hit"
