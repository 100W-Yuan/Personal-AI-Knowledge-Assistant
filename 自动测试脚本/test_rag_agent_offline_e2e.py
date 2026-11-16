# -*- coding: utf-8 -*-
"""P2-3B2-B：真实 Agent 执行链离线 E2E 补测（G-1 / G-2 / G-3）。

覆盖 P2-3B2-A / P2-3B2-A2 审查确认、既有测试为零覆盖的三处 Agent 层缺口：

    G-1  DOCX 的 section / section_index / segment_index / table_index
         经真实 Loader → TextSplitter → KnowledgeBase → search_knowledge 工具
         → Agent 归一化，最终出现在 run() 返回的 sources 里。
    G-2  ContextBuilder 实际渲染的定位文本（``第 N 页（共 M 页）`` /
         ``第 N 节：标题`` / ``表格 N``）进入**最终 reasoning 请求**的 prompt。
    G-3  单轮内两次 search_knowledge 调用时，来源跨调用合并：同身份只保留
         分数严格更高的记录，且替换发生时保留首次出现的位置。

边界声明（不得据此扩大结论）
------------------------------
1. 全程零真实模型、零真实 Embedding、零网络：模型侧用 ``ScriptedClient``
   脚本化响应（脚本耗尽即报错，绝不静默返回兜底响应），向量侧用
   ``_StubEmbedding``（G-1/G-2）与 ``_MarkerEmbedding``（G-3）。
   不导入 sentence-transformers / torch / transformers。
2. 只经 ``Agent.run()`` 主流程验证；**不**直接调用 ``_knowledge_sources``
   构造被测输出，也**不**用 ``agent.execute_tool`` 替代主循环。
3. 意图文本固定为既有测试已钉死的 neutral 文案，确保不触发 recall 确定性
   预检索与缓存复用分支——否则「工具真的执行了两次」将不成立。
4. 夹具全部落在 ``tmp_path``；不读 ``.env``，不触碰正式 ``memory.db``、
   ``data/business.db``、``rag_knowledge/``。每个用例在 finally 中关闭
   本用例自建的 Memory。
5. 手写 PDF 字节流与 G-1/G-2 的确定性向量复用 ``test_rag_offline_e2e.py``
   （同目录），保证这两样规则只有一份实现。

关于 ``count`` 断言的隐含前提（L-6）
-----------------------------------
G-1 的 ``count == 4``、G-2 的 ``count == 6``、G-3 的 ``count == 4`` 之所以
成立，是因为 ``KnowledgeBase.search`` 在 ``threshold is None`` 时会丢弃
``score <= 0.0`` 的结果（``rag/knowledge_base.py:519-520``）。本文件使用的
两个编码器都保证：

    - 向量首维恒为 1.0 → 任何向量都不可能是零向量；
    - 两个非零向量的余弦 = (1·1 + …) / (‖a‖·‖b‖) ≥ 1/(‖a‖·‖b‖) > 0。

因此所有参与检索的 chunk 都会通过该过滤。**若替换为其它 stub，必须重新
证明这两条**，否则 count 断言会因零相似度被静默过滤而失效。
"""

import contextlib
import hashlib
import io
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
for _path in (str(PROJECT_ROOT), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from agent import Agent                        # noqa: E402
from memory import Memory                      # noqa: E402
from rag.knowledge_base import KnowledgeBase   # noqa: E402

# 同目录复用：手写最小 PDF 构造器 + 确定性 Stub Embedding（G-1/G-2 用）
from test_rag_offline_e2e import (             # noqa: E402
    _StubEmbedding,
    _build_pdf,
)

DUMMY = dict(api_key="dummy", base_url="https://example.invalid", model="dummy")

# neutral：既有 test_knowledge_intent_guard.py:213-216 已固定该文案为 neutral，
# 不会触发 recall 预检索（recall 才会在 Planner 之前先执行一次工具）。
NEUTRAL_MESSAGE = "给我讲讲原理。"

# G-1 / G-2 的 DOCX 夹具结构（顺序不可调换）：
# 标题1 → 段落A → 2x2 表格 → 段落B → 空表格 → 标题2 → 段落C
DOCX_SECTION_ONE = "第一章"
DOCX_PARAGRAPH_A = "正文段落A"
DOCX_PARAGRAPH_B = "正文段落B"
DOCX_SECTION_TWO = "第二章"
DOCX_PARAGRAPH_C = "正文段落C"

PDF_PAGE_ONE = "第一页正文内容"
PDF_PAGE_TWO = "第二页正文内容"

# G-2 需要进入最终 Prompt 的五段定位文本（ContextBuilder 的真实渲染结果）
LOCATION_TEXTS = (
    "第 1 页（共 2 页）",
    "第 2 页（共 2 页）",
    "第 1 节：第一章",
    "第 2 节：第二章",
    "表格 1",
)

# ----------------------------------------------------------------------------
# G-3：四文档 + 五维确定性编码器（不依赖 SHA-256 哈希槽位，也不依赖浮点舍入）
#
# 维度定义（见 _MarkerEmbedding）：
#   v[0] = 1.0
#   v[1] = count("甲")   v[2] = count("丙")
#   v[3] = count("寅")   v[4] = count("辰")
#
# 文档向量、模长平方与模长：
#   alpha → "甲"×5         → [1, 5, 0, 0,  0]   ‖·‖² = 26    ‖·‖ = √26
#   gamma → "丙"×2         → [1, 0, 2, 0,  0]   ‖·‖² = 5     ‖·‖ = √5
#   beta  → "寅"×3         → [1, 0, 0, 3,  0]   ‖·‖² = 10    ‖·‖ = √10
#   delta → "甲" + "辰"×20 → [1, 1, 0, 0, 20]   ‖·‖² = 402   ‖·‖ = √402
#
# 查询文本与目标文档文本完全相同，故查询向量 == 该文档向量。
# 余弦 score = a·b / (‖a‖·‖b‖)：
#   cos(α,α) = 1              cos(γ,γ) = 1
#   cos(β,β) = 1              cos(δ,δ) = 1
#   cos(α,γ) = 1/(√26·√5)  = 1/√130   ≈ 0.087706
#   cos(α,β) = 1/(√26·√10) = 1/√260   ≈ 0.062017
#   cos(α,δ) = 6/(√26·√402)= 6/√10452 ≈ 0.058688   （分子 1·1+5·1 = 6）
#   cos(γ,β) = 1/(√5·√10)  = 1/√50    ≈ 0.141421
#   cos(γ,δ) = 1/(√5·√402) = 1/√2010  ≈ 0.022305   （分子 1·1 = 1）
#
# 第一轮 query=α：α 1.0 > γ 0.087706 > β 0.062017 > δ 0.058688
#                 → [alpha, gamma, beta, delta]
# 第二轮 query=γ：γ 1.0 > β 0.141421 > α 0.087706 > δ 0.022305
#                 → [gamma, beta, alpha, delta]
#
# 每轮内部严格递减、无并列，且所有分数 > 0（不会被 `score > 0.0` 过滤）。
# 注意：cos(α,γ) 与 cos(γ,α) 必然相等（余弦对称），所以**不能**断言
# "跨两轮的相似度全部互不相等"，只能断言"每轮内部"互不相等。
#
# 合并（原地替换）：
#   α 保持 1.0（第二轮 0.087706 更低）；γ 0.087706 → 1.0；
#   β 0.062017 → 0.141421；δ 保持 0.058688（第二轮 0.022305 更低）
#   → [alpha(1.0), gamma(1.0), beta(0.141421), delta(0.058688)]
#   四个来源的首次出现索引 0/1/2/3 全部保持不变。
# ----------------------------------------------------------------------------
ALPHA_TEXT = "甲甲甲甲甲"
GAMMA_TEXT = "丙丙"
BETA_TEXT = "寅寅寅"
# 一个"甲" + 20 个"辰"：v[1] 与 alpha 共享 1（保证两轮相似度均非零），
# v[4]=20 把模长拉到 √402，使 delta 两轮都排在末位且第二轮分数更低。
DELTA_TEXT = "甲" + "辰" * 20

# 关键词 → 向量维度的固定映射（v[1..4] 分别统计这四个字符的出现次数）
_MARKER_CHARS = ("甲", "丙", "寅", "辰")


class _MarkerEmbedding:
    """G-3 专用的五维确定性编码器（零模型、零随机、零网络）。

    维度定义：
        v[0] = 1.0                  —— 基线；保证向量非零，且任意两向量余弦 > 0
        v[1] = text.count("甲")
        v[2] = text.count("丙")
        v[3] = text.count("寅")
        v[4] = text.count("辰")

    四篇文档的取值：α 只在 v[1] 取 5；γ 只在 v[2] 取 2；β 只在 v[3] 取 3；
    δ 在 v[1] 取 1、v[4] 取 20。任意两篇文档的向量既不相等也不成比例，
    因此"不同文档之间余弦并列"在结构上不可能发生——不依赖任何哈希槽位是否
    碰撞，也不依赖浮点舍入（同一文本必然得到同一向量；同一对文本的分数
    逐位相同，余弦对称性因此也是精确成立的）。
    """

    def __init__(self):
        self.calls = []

    def encode(self, text):
        self.calls.append(text)
        return [1.0] + [float(text.count(marker)) for marker in _MARKER_CHARS]


# ============================================================
# 模型侧替身：脚本化客户端
# ============================================================
class ScriptedClient:
    """预置响应序列的假客户端；脚本耗尽直接报错，避免静默兜底。

    与 ``RAG测试/test_agent_rag_integration.py`` 的差别有两处：
    1. 穷尽时抛 ``AssertionError`` 而不是返回兜底计划，使「模型调用次数
       超出预期」立刻暴露，而不是让用例带着错位的脚本继续跑下去；
    2. 异常信息带 ``label``（场景名），使失败能直接定位到 G-1/G-2/G-3。
    """

    def __init__(self, script, label=""):
        self._script = list(script)
        self._declared = len(script)
        self.label = label
        self.calls = []

    def _complete(self, messages):
        self.calls.append(messages)
        if not self._script:
            raise AssertionError(
                f"[{self.label}] ScriptedClient 脚本耗尽："
                f"模型调用次数超出预期，已发生 {len(self.calls)} 次调用，"
                f"脚本长度 {self._declared}"
            )
        return self._script.pop(0)

    @property
    def chat(self):
        outer = self

        class _Completions:
            @staticmethod
            def create(model=None, messages=None, **kwargs):
                content = outer._complete(messages)

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


# ============================================================
# 夹具与执行辅助
# ============================================================
def _write_docx(path):
    """按 G-1 结构生成真实 DOCX（真实 python-docx 写盘，非内存伪造）。"""
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


def _build_kb(kbdir, encoder):
    """真 KnowledgeBase + 注入的确定性向量（绝不加载真实模型）。"""
    return KnowledgeBase(str(kbdir), embedding_factory=lambda: encoder)


def _build_agent(tmp_path, monkeypatch, knowledge_base):
    """先 chdir 再构造 Agent：Memory() 必须落在 tmp_path。

    ``Agent.__init__`` 内部会执行 ``Memory()``，其默认路径跟随当前工作目录。
    """
    monkeypatch.chdir(tmp_path)
    agent = Agent(**DUMMY, knowledge_base=knowledge_base)
    agent.memory.close()
    agent.memory = Memory(db_path=str(tmp_path / "memory.db"))
    return agent


def _spy_search(monkeypatch, knowledge_base):
    """观察 KnowledgeBase.search 的真实调用（委托原实现，不做替换语义）。

    这是唯一可观测「工具是否真的检索了、检索了几次、每次拿到什么」的位置：
    ``run()`` 不返回 state，测试又不得直接调用 ``_knowledge_sources``。
    """
    calls = []
    real_search = knowledge_base.search

    def spy(query, top_k=3, threshold=None):
        # 形参默认值与 rag/knowledge_base.py:475 的 search 签名保持一致；
        # 三个用例都显式传 top_k，因此默认值不会影响断言。
        result = real_search(query, top_k=top_k, threshold=threshold)
        calls.append({"query": query, "top_k": top_k, "result": result})
        return result

    monkeypatch.setattr(knowledge_base, "search", spy)
    return calls


def _run(agent, user_message, plan, tool_params, label):
    """按 plan 构造脚本并跑一次 ``Agent.run``。

    脚本顺序即 Agent 的模型调用顺序：
        extract_memories → create_plan → 每个步骤（tool 参数生成 / reasoning 文本）
        → 全部步骤完成后的最终 reasoning
    即调用次数恒为 ``len(plan) + 3``。
    """
    pending_params = list(tool_params)
    script = [
        json.dumps({"should_remember": False, "memories": []}),
        json.dumps({"plan": plan}),
    ]
    for step in plan:
        if step.get("type") == "tool":
            script.append(json.dumps(pending_params.pop(0)))
        else:
            script.append("中间推理")
    script.append("最终答案。")

    expected_calls = len(plan) + 3
    assert len(script) == expected_calls, (
        f"[{label}] 脚本长度 {len(script)} != 预期模型调用次数 {expected_calls}"
    )
    assert not pending_params, f"[{label}] 存在未被任何步骤消费的工具参数"

    client = ScriptedClient(script, label=label)
    agent.client = client

    with contextlib.redirect_stdout(io.StringIO()):
        result = agent.run(user_message, user_id="test-user")

    # 脚本必须被完整消费：既不能耗尽（会抛错），也不能有剩余（说明实际
    # 调用次数少于预期，后续断言将基于错位的脚本）。
    assert len(client._script) == 0, (
        f"[{label}] 脚本未被完整消费，剩余 {len(client._script)} 条；"
        f"实际调用 {len(client.calls)} 次"
    )
    assert len(client.calls) == expected_calls, (
        f"[{label}] 实际模型调用 {len(client.calls)} 次 != 预期 {expected_calls} 次"
    )

    return result, client


def _tool_plan(*descriptions):
    return [
        {
            "description": description,
            "type": "tool",
            "tool_name": "search_knowledge",
        }
        for description in descriptions
    ]


# ============================================================
# G-1：DOCX 定位信息进入 Agent sources
# ============================================================
def test_g1_docx_location_metadata_reaches_agent_sources(tmp_path, monkeypatch):
    """真实 DOCX → 真实索引 → 真实工具 → Agent sources，定位字段全程不丢。"""
    kbdir = tmp_path / "kb"
    kbdir.mkdir()
    _write_docx(kbdir / "manual.docx")

    stub = _StubEmbedding()
    kb = _build_kb(kbdir, stub)
    searches = _spy_search(monkeypatch, kb)
    agent = _build_agent(tmp_path, monkeypatch, kb)

    try:
        result, client = _run(
            agent,
            NEUTRAL_MESSAGE,
            _tool_plan("检索项目知识") + [{"description": "回答", "type": "reasoning"}],
            [{"query": DOCX_PARAGRAPH_A, "top_k": 4}],
            label="G-1",
        )

        assert result["success"] is True
        # 5 次调用：extract / create_plan / 工具参数 / 步骤 reasoning / 最终 reasoning
        assert len(client.calls) == 5
        # neutral 意图不预检索：真正打到 KnowledgeBase 的只有计划里的那一次
        assert len(searches) == 1
        assert searches[0]["result"]["count"] == 4
        assert result["retrieval"]["status"] == "hit"

        sources = result["sources"]
        assert len(sources) == 4

        section_sources = [s for s in sources if "section" in s]
        table_sources = [s for s in sources if "table_index" in s]

        # 3 个章节片段（标题1+段落A、段落B、标题2+段落C）与 1 个表格
        assert len(section_sources) == 3
        assert len(table_sources) == 1

        for source in section_sources:
            assert source["section"] in (DOCX_SECTION_ONE, DOCX_SECTION_TWO)
            for key in ("section_index", "segment_index"):
                assert key in source, (key, source)
                value = source[key]
                assert isinstance(value, int) and not isinstance(value, bool)
                assert value >= 0
            # 章节片段不得凭空空降表格/页码概念
            assert "table_index" not in source
            assert "page" not in source
            assert "page_count" not in source

        # M-1：成对断言 (section_index, segment_index)。
        # 真实 Loader 产出（rag/loader.py:611-632）：
        #   节0 片段0 → (0, 0)；节0 片段1 → (0, 1)；节1 片段0 → (1, 0)
        # 两个字段的多重集恰好都是 {0, 0, 1}，若分别断言 "sorted(x) == [0,0,1]"
        # 则"把 section_index 复制给 segment_index"这种串用 bug 仍会通过；
        # 成对断言把两者的对应关系一起锁住，串用后必然得到 [(0,0),(1,1),(1,0)]
        # 之类的错误组合而失败。
        assert sorted(
            (s["section_index"], s["segment_index"]) for s in section_sources
        ) == [(0, 0), (0, 1), (1, 0)]

        table_source = table_sources[0]
        assert table_source["table_index"] == 0
        # 表格来源没有凭空生成 section 身份
        assert "section" not in table_source
        assert "section_index" not in table_source
        assert "segment_index" not in table_source

        # 身份字段保留且类型正确（跨调用去重的前提）
        expected_document_id = hashlib.sha256(b"manual.docx").hexdigest()[:32]
        assert {s["document_id"] for s in sources} == {expected_document_id}
        assert sorted(s["chunk_seq"] for s in sources) == [0, 1, 2, 3]
        assert {s["chunk_id"] for s in sources} == {0}
    finally:
        agent.memory.close()


# ============================================================
# G-2：ContextBuilder 定位文本进入最终 Prompt
# ============================================================
def test_g2_context_builder_location_text_reaches_final_reasoning_prompt(
        tmp_path, monkeypatch):
    """PDF 页码 + DOCX 章节 + DOCX 表格三种定位文本都进入最终 reasoning 请求。"""
    kbdir = tmp_path / "kb"
    kbdir.mkdir()
    _write_docx(kbdir / "manual.docx")
    # 2 页有效文本：页码 1、2，page_count 为物理总页数 2
    (kbdir / "paper.pdf").write_bytes(_build_pdf([PDF_PAGE_ONE, PDF_PAGE_TWO]))

    stub = _StubEmbedding()
    kb = _build_kb(kbdir, stub)
    searches = _spy_search(monkeypatch, kb)
    agent = _build_agent(tmp_path, monkeypatch, kb)

    try:
        result, client = _run(
            agent,
            NEUTRAL_MESSAGE,
            _tool_plan("检索项目知识") + [{"description": "回答", "type": "reasoning"}],
            [{"query": PDF_PAGE_ONE, "top_k": 6}],
            label="G-2",
        )

        assert result["success"] is True
        assert len(client.calls) == 5
        assert len(searches) == 1
        # L-1：实际检索调用的 top_k 必须是 6（否则 6 个 chunk 无法全部进入上下文）
        assert searches[0]["top_k"] == 6
        # docx 4 块 + pdf 2 页 = 6 块，top_k=6 全部返回
        assert searches[0]["result"]["count"] == 6
        assert result["retrieval"]["status"] == "hit"

        # 真正的最终 reasoning 请求：第 5 次调用（calls[-1]）的 system 消息
        final_prompt = client.calls[-1][0]["content"]
        # 更早的规划请求（第 2 次调用 = create_plan）不得包含任何定位文本
        plan_prompt = client.calls[1][0]["content"]

        # L-4：五段定位文本全部进入最终 Prompt，且全部不在规划 Prompt 中
        for location in LOCATION_TEXTS:
            assert location in final_prompt, location
            assert location not in plan_prompt, location

        # PDF：ContextBuilder 的原文是「第 N 页（共 M 页）」。
        # 两页的 chunk_id 都是 0，必须靠 page 区分——这里刻意按页码分别断言，
        # 并用完整的来源行证明文本确实由 ContextBuilder 渲染（含「｜」分隔符）。
        assert "来源：paper.pdf（片段 0）｜第 1 页（共 2 页）" in final_prompt
        assert "来源：paper.pdf（片段 0）｜第 2 页（共 2 页）" in final_prompt

        # DOCX：章节序号 1 基 + 标题原文
        assert "｜第 1 节：第一章" in final_prompt
        assert "｜第 2 节：第二章" in final_prompt

        # DOCX：表格序号 1 基
        assert "来源：manual.docx（片段 0）｜表格 1" in final_prompt

        # 答案与来源各自独立：answer 来自模型，sources 来自本轮工具历史
        assert result["answer"] == "最终答案。"
        pdf_sources = [s for s in result["sources"] if s["source"] == "paper.pdf"]
        assert sorted(s["page"] for s in pdf_sources) == [1, 2]
        assert {s["page_count"] for s in pdf_sources} == {2}
        assert {s["chunk_id"] for s in pdf_sources} == {0}
    finally:
        agent.memory.close()


# ============================================================
# G-3：多次工具调用的来源跨调用合并
# ============================================================
def test_g3_multiple_tool_calls_merge_sources_across_calls(tmp_path, monkeypatch):
    """两轮检索：同身份取严格更高分、保留首次出现位置、异身份并存。

    四篇文档的理论分数见文件头 _MarkerEmbedding 段落的推导：
        第一轮 query=α → α 1.0 > γ 1/√130 > β 1/√260 > δ 6/√10452
                         → [α, γ, β, δ]
        第二轮 query=γ → γ 1.0 > β 1/√50 > α 1/√130 > δ 1/√2010
                         → [γ, β, α, δ]
    合并后仍为首次出现顺序 [α, γ, β, δ]，索引 0/1/2/3 全部守恒：
    γ 与 β 被"原地替换"为更高分，α 与 δ 保持第一轮分数。
    """
    kbdir = tmp_path / "kb"
    kbdir.mkdir()
    (kbdir / "alpha.md").write_text(ALPHA_TEXT, encoding="utf-8")
    (kbdir / "gamma.md").write_text(GAMMA_TEXT, encoding="utf-8")
    (kbdir / "beta.md").write_text(BETA_TEXT, encoding="utf-8")
    (kbdir / "delta.md").write_text(DELTA_TEXT, encoding="utf-8")

    encoder = _MarkerEmbedding()
    kb = _build_kb(kbdir, encoder)
    searches = _spy_search(monkeypatch, kb)
    agent = _build_agent(tmp_path, monkeypatch, kb)

    try:
        result, client = _run(
            agent,
            NEUTRAL_MESSAGE,
            _tool_plan("第一轮检索", "第二轮检索"),
            [
                {"query": ALPHA_TEXT, "top_k": 4},
                {"query": GAMMA_TEXT, "top_k": 4},
            ],
            label="G-3",
        )

        assert result["success"] is True
        # 5 次调用：extract / create_plan / 参数1 / 参数2 / 最终 reasoning
        # （无 reasoning 步骤时，两轮工具执行完直接进入最终 reasoning）
        assert len(client.calls) == 5
        assert result["retrieval"]["status"] == "hit"

        # 两次工具调用都真的打到了 KnowledgeBase：查询文本与 top_k 都被真实
        # 传递（不是缓存复用、也不是参数被吞掉）
        assert [call["query"] for call in searches] == [ALPHA_TEXT, GAMMA_TEXT]
        assert [call["top_k"] for call in searches] == [4, 4]

        # 四个文档各产出 1 个 chunk（文本远短于 chunk_size=100），top_k=4 时
        # 每轮都返回全部四条——这是"四文档场景"成立的前提，必须显式锁住。
        first, second = searches[0]["result"], searches[1]["result"]
        assert first["count"] == 4
        assert second["count"] == 4

        # 两轮排序（严格递减、无并列）——由 _MarkerEmbedding 的维度互斥保证
        assert [r["source"] for r in first["results"]] == [
            "alpha.md", "gamma.md", "beta.md", "delta.md"
        ]
        assert [r["source"] for r in second["results"]] == [
            "gamma.md", "beta.md", "alpha.md", "delta.md"
        ]

        alpha_round1, gamma_round1, beta_round1, delta_round1 = [
            r["score"] for r in first["results"]
        ]
        gamma_round2, beta_round2, alpha_round2, delta_round2 = [
            r["score"] for r in second["results"]
        ]

        # 所有相似度严格大于零。这条同时支撑上面 count == 4 的成立：
        # kb.search 在 threshold is None 时会丢弃 score <= 0.0 的结果。
        for score in (alpha_round1, gamma_round1, beta_round1, delta_round1,
                      gamma_round2, beta_round2, alpha_round2, delta_round2):
            assert score > 0.0

        # 同一身份两轮分数不同，且各自方向符合推导
        assert gamma_round2 > gamma_round1      # γ：1/√130 → 1.0
        assert beta_round2 > beta_round1        # β：1/√260 → 1/√50
        assert alpha_round2 < alpha_round1      # α：1.0 → 1/√130（本轮不改进）
        assert delta_round2 < delta_round1      # δ：6/√10452 → 1/√2010（不改进）

        # 每轮内部严格递减、无并列（若并列，下面的"最高分"断言将失去判别力）
        # 注意：cos(α,γ) 与 cos(γ,α) 由余弦对称性必然相等，因此这里只能断言
        # "每轮内部"互不相等，**不能**断言"跨两轮八个分数全部互不相等"。
        for round_result in (first, second):
            round_scores = [r["score"] for r in round_result["results"]]
            assert len(round_scores) == 4
            assert all(
                left > right for left, right in zip(round_scores, round_scores[1:])
            )

        # 跨调用合并：顺序 == 首次出现顺序（四个文档都在第一轮出现）。
        # 这一条同时就是「索引守恒」断言：source_indexes 记录的是首次出现时的
        # 列表下标，替换必须写在原位（sources[i] = normalized），不能追加。
        sources = result["sources"]
        assert len(sources) == 4
        assert [s["source"] for s in sources] == [
            "alpha.md", "gamma.md", "beta.md", "delta.md"
        ]

        alpha_src, gamma_src, beta_src, delta_src = sources

        # 每个来源的最终分数 == 两轮中该身份的最高分
        assert alpha_src["score"] == max(alpha_round1, alpha_round2)
        assert gamma_src["score"] == max(gamma_round1, gamma_round2)
        assert beta_src["score"] == max(beta_round1, beta_round2)
        assert delta_src["score"] == max(delta_round1, delta_round2)

        # 更具体的替换方向（同时说明两条 execution_history 都被读取并参与合并）：
        assert alpha_src["score"] == alpha_round1     # 未被替换（第二轮更低）
        assert gamma_src["score"] == gamma_round2     # 被第二轮高分替换
        assert beta_src["score"] == beta_round2       # 被第二轮高分替换
        assert delta_src["score"] == delta_round1     # 未被替换（第二轮更低）

        # M-3（位置反证）：精确顺序断言必须能区分 `pop(index) + append(item)`
        # 这种"分数对、位置错"的实现。假设该错误实现每次都能正确重建身份
        # 索引，则四文档场景的演化是：
        #
        #   第一轮结束        [α, γ, β, δ]   α0 γ1 β2 δ3
        #   处理 γ（第二轮）  1.0 > 0.0877 → pop(1) → [α, β, δ]，append γ
        #                    [α, β, δ, γ]   α0 β1 δ2 γ3
        #   处理 β（第二轮）  0.1414 > 0.0620 → pop(1) → [α, δ, γ]，append β
        #                    [α, δ, γ, β]   α0 δ1 γ2 β3
        #   处理 α（第二轮）  0.0877 < 1.0   → 不替换
        #   处理 δ（第二轮）  0.0223 < 0.0587 → 不替换
        #   最终              [α, δ, γ, β] ≠ [α, γ, β, δ]
        #
        # 关键在第四个文档：早先被否决的三元素场景里，γ 与 β 各被移到末尾一次，
        # 两次旋转的净效果恰好是恒等置换，该错误实现与"原地替换"不可区分；
        # 加入 δ 后置换变为 [α, δ, γ, β]，上面的顺序断言必然失败。
        #
        # 另一类错误实现是"只 append 不删除"：同一身份会重复出现，
        # len(sources) == 4 与下面的唯一性断言同时失败。
        assert len({(s["document_id"], s["chunk_seq"]) for s in sources}) == 4
        expected_document_ids = {
            hashlib.sha256(name.encode("utf-8")).hexdigest()[:32]
            for name in ("alpha.md", "gamma.md", "beta.md", "delta.md")
        }
        assert {s["document_id"] for s in sources} == expected_document_ids

        # 身份字段完好：四个不同文件、四个不同 document_id、chunk_seq 均为 int
        assert {s["source"] for s in sources} == {
            "alpha.md", "gamma.md", "beta.md", "delta.md"
        }
        assert all(isinstance(s["chunk_seq"], int) for s in sources)
    finally:
        agent.memory.close()
