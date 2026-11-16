"""PAI-WORKBENCH-R1-2-S2-K2：知识库只读功能测试。

覆盖任务书第六节 A/B/C/D 四组：

  A. 文档列表     K2-A01 ~ K2-A08
  B. 索引状态     K2-B01 ~ K2-B04
  C. 知识搜索     K2-C01 ~ K2-C11
  D. 安全         K2-D01 ~ K2-D06

测试策略（与 K1 一致，遵守任务书第七节）：
* 前端逻辑用**结构契约断言**（读取 app.js / index.html 源码，锁定调用形态）；
* 后端契约用**真实 HTTP 往返**（TestClient），全部后端依赖用 monkeypatch 替身，
  绝不加载 Embedding / 模型，绝不访问真实知识库目录或正式数据库；
* 复用根 conftest 的 patched_main_memory + isolated_business_db。
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import main as main_module
import knowledge_store
from auth.service import AuthService, _reset_failures_for_tests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_HTML = PROJECT_ROOT / "knowledge_ui" / "index.html"
KNOWLEDGE_JS = PROJECT_ROOT / "knowledge_ui" / "static" / "app.js"

ORIGIN = {"Origin": "http://testserver"}
ADMIN_NAME = "root_admin"
ADMIN_PASS = "Admin-Pass-2026"
USER_NAME = "hr_user"
USER_PASS = "Hr-Pass-2026-ok"
FORCED_NAME = "forced_user"
FORCED_PASS = "Temp-Pass-2026"


def _read(path: Path) -> str:
    assert path.is_file(), f"缺少文件：{path}"
    return path.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def knowledge_js():
    return _read(KNOWLEDGE_JS)


@pytest.fixture(scope="module")
def knowledge_html():
    return _read(KNOWLEDGE_HTML)


@pytest.fixture
def svc(patched_main_memory):
    service = AuthService(db_path=patched_main_memory.db_path)
    service.ensure_schema()
    _reset_failures_for_tests()
    yield service
    _reset_failures_for_tests()


@pytest.fixture
def client(svc, isolated_business_db):
    svc.create_user(ADMIN_NAME, ADMIN_PASS, role="admin")
    svc.create_user(USER_NAME, USER_PASS, role="user")
    with TestClient(main_module.app, follow_redirects=False) as c:
        yield c


def login(client, username=USER_NAME, password=USER_PASS):
    return client.post(
        "/auth/login",
        json={"username": username, "password": password},
        headers=ORIGIN,
    )


# ============================================================
# 测试替身
# ============================================================


def _doc(filename="产品手册.md", **overrides):
    """构造一份与 knowledge_store._build_metadata 同构的文档元数据。"""
    suffix = filename.rsplit(".", 1)[-1] if "." in filename else ""
    record = {
        "document_id": hashlib.sha256(filename.encode("utf-8")).hexdigest()[:32],
        "filename": filename,
        "file_type": suffix,
        "file_size": 2048,
        "created_at": "2026-11-01T10:00:00",
        "created_at_source": "ctime_fallback",
        "updated_at": "2026-11-02T11:30:00",
        "storage_type": "persistent",
        "parse_status": "unknown",
        "chunk_count": None,
    }
    record.update(overrides)
    return record


def _patch_listing(monkeypatch, documents, directory_exists=True):
    def fake_list_documents(directory):
        return {"directory_exists": directory_exists, "documents": documents}

    monkeypatch.setattr(knowledge_store, "list_documents", fake_list_documents)


class _FakeKB:
    """知识库替身：只记录被调用的方法，永不加载索引或模型。

    search() 复制真实 KnowledgeBase.search 的 input 校验
    （空查询 / top_k 范围），使「HTTP 200 + ok:false + stage=input」
    分支得到真实覆盖；已显式指定业务失败 payload 时优先返回该 payload。
    """

    def __init__(self, search_payload=None, counts=None, skipped_reparse=None):
        self.search_payload = search_payload
        self.counts = counts
        self.skipped_reparse_count = skipped_reparse
        self.calls = []

    def search(self, query, top_k=3, threshold=None):
        self.calls.append(("search", query, top_k, threshold))
        if self.search_payload is not None and self.search_payload.get("ok") is False:
            return self.search_payload
        if not isinstance(query, str) or not query.strip():
            return {
                "ok": False,
                "stage": "input",
                "error": "知识检索问题不能为空",
                "query": query,
            }
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 10:
            return {
                "ok": False,
                "stage": "input",
                "error": "top_k 必须是 1 到 10 之间的整数",
                "query": query,
            }
        return self.search_payload

    def indexed_chunk_counts(self):
        self.calls.append(("indexed_chunk_counts",))
        return self.counts


def _patch_kb(monkeypatch, kb):
    monkeypatch.setattr(main_module, "get_knowledge_base", lambda: kb)
    return kb


# ============================================================
# A. 文档列表
# ============================================================


def test_A01_default_request_uses_with_index_false(client, knowledge_js):
    """K2-A01：列表默认使用 with_index=false；初始化只加载一次。"""
    # 前端：固定拼出 false，初始化调用点唯一。
    assert '"?with_index=" + (withIndex ? "true" : "false")' in knowledge_js
    # K3A 起有两处调用，且都是 with_index=false：初始化 + 上传成功后刷新。
    assert knowledge_js.count("loadDocuments(false);") == 2

    # 后端：缺省与显式 false 等价，且不读取索引。
    assert login(client).status_code == 200
    response = client.get("/knowledge/documents")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["ok"] is True
    assert body["with_index"] is False
    assert body["count"] == len(body["documents"])


def test_A02_success_list_render(client, monkeypatch, knowledge_js):
    """K2-A02：成功列表渲染（字段与渲染入口结构一致）。"""
    _patch_listing(monkeypatch, [_doc(), _doc("报价单.xlsx")])

    assert login(client).status_code == 200
    response = client.get("/knowledge/documents?with_index=false")
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["count"] == 2
    names = [d["filename"] for d in body["documents"]]
    assert names == ["产品手册.md", "报价单.xlsx"]
    for item in body["documents"]:
        # 渲染所需字段必须存在（与 app.js buildRow 的读取一一对应）。
        for key in ("document_id", "filename", "file_type", "file_size", "updated_at"):
            assert key in item

    # 前端：行渲染使用 td.textContent，不使用 HTML 注入。
    assert "td.textContent = text;" in knowledge_js
    assert '"共 " + count + " 个文档"' in knowledge_js


def test_A03_empty_directory_state(client, monkeypatch, knowledge_js):
    """K2-A03：空目录渲染为空状态而非错误。"""
    _patch_listing(monkeypatch, [])

    assert login(client).status_code == 200
    body = client.get("/knowledge/documents").json()
    assert body["count"] == 0
    assert body["documents"] == []
    assert body["directory_exists"] is True

    assert "知识库暂无文档（0 个）。" in knowledge_js


def test_A04_missing_directory_state(client, monkeypatch, knowledge_js):
    """K2-A04：目录不存在时前端给出专门的空状态文案。"""
    _patch_listing(monkeypatch, [], directory_exists=False)

    assert login(client).status_code == 200
    body = client.get("/knowledge/documents").json()
    assert body["directory_exists"] is False
    assert body["count"] == 0

    assert "知识库目录不存在或尚未创建" in knowledge_js


def test_A05_optional_fields_degrade_safely(client, monkeypatch, knowledge_js):
    """K2-A05：可选字段缺失时安全降级，不抛错、不臆造值。"""
    # created_at 为 None（created_at_source=unavailable）、其余可选字段缺失。
    partial = {
        "document_id": "abc123",
        "filename": "扫描件.pdf",
        "file_type": "pdf",
        "file_size": None,
        "updated_at": None,
    }
    _patch_listing(monkeypatch, [partial])

    assert login(client).status_code == 200
    response = client.get("/knowledge/documents")
    assert response.status_code == 200, response.text
    item = response.json()["documents"][0]
    assert item["filename"] == "扫描件.pdf"

    # 前端降级：空值 → 未知；非数字尺寸 → 未知；非数字 chunk_count → 未知。
    assert "function textOrUnknown(value)" in knowledge_js
    assert 'if (value === null || value === undefined || value === "") return "未知";' in knowledge_js
    assert 'if (!isFinite(n) || n < 0) return "未知";' in knowledge_js
    assert "function chunkCountText(value)" in knowledge_js
    assert 'return "未知";' in knowledge_js


def test_A06_list_loading_state(knowledge_js):
    """K2-A06：列表加载中状态存在，且与"请求中"时序一致。"""
    assert '"正在加载文档列表…"' in knowledge_js
    assert "setDocState(" in knowledge_js
    # loading 文案在请求之前设置。
    assert knowledge_js.index('"正在加载文档列表…"') < knowledge_js.index(
        "return requestJSON(url)"
    )


def test_A07_refresh_list(knowledge_js):
    """K2-A07：刷新按钮重新加载（且保持索引列可见性）。"""
    assert 'refreshBtn.addEventListener("click"' in knowledge_js
    assert "loadDocuments(indexVisible)" in knowledge_js


def test_A08_failure_is_not_reported_as_empty(client, monkeypatch, knowledge_js):
    """K2-A08：请求失败显示具体错误，绝不伪装成空目录。"""
    def boom(directory):
        raise knowledge_store.KnowledgeDirInvalid("知识库目录无法读取")

    monkeypatch.setattr(knowledge_store, "list_documents", boom)

    assert login(client).status_code == 200
    response = client.get("/knowledge/documents")
    assert response.status_code == 500, response.text
    detail = response.json()["detail"]
    assert detail["type"] == "knowledge_dir_invalid"

    # 前端：错误分支先于渲染，且文案明确标注"加载失败"。
    assert '"文档列表加载失败：" + result.error' in knowledge_js
    assert knowledge_js.index('"文档列表加载失败："') < knowledge_js.index(
        "renderDocuments(data, withIndex);"
    )


# ============================================================
# B. 索引状态
# ============================================================


def test_B01_index_not_requested_by_default(knowledge_js):
    """K2-B01：默认不请求 with_index=true。"""
    # 仅「查看索引状态」按钮触发 true（唯一一处调用）。
    assert knowledge_js.count("loadDocuments(true)") == 1
    assert "checkIndexBtn.addEventListener(\"click\"" in knowledge_js
    # 初始化走 false：init 内先识别身份、再加载列表。
    init_block = knowledge_js[knowledge_js.index("function init()") :]
    assert "loadIdentity();" in init_block
    assert init_block.index("loadIdentity();") < init_block.index("loadDocuments(false);")


def test_B02_index_state_requires_user_action(client, monkeypatch):
    """K2-B02：with_index=true 只读取已有索引状态（用户主动请求时）。"""
    _patch_listing(monkeypatch, [_doc("a.md"), _doc("b.md")])
    kb = _patch_kb(
        monkeypatch, _FakeKB(counts={"a.md": 7}, skipped_reparse=0)
    )

    assert login(client).status_code == 200
    body = client.get("/knowledge/documents?with_index=true").json()

    assert body["with_index"] is True
    by_name = {d["filename"]: d for d in body["documents"]}
    assert by_name["a.md"]["parse_status"] == "indexed"
    assert by_name["a.md"]["chunk_count"] == 7
    assert by_name["b.md"]["parse_status"] == "not_indexed"
    assert by_name["b.md"]["chunk_count"] is None
    assert body["index_skipped_reparse"] == 0
    # 只读取索引统计，不触发检索/构建。
    assert kb.calls == [("indexed_chunk_counts",)]


def test_B03_unknown_index_state(client, monkeypatch, knowledge_js):
    """K2-B03：索引未构建时显示 unknown，不推断为已索引。"""
    _patch_listing(monkeypatch, [_doc("a.md")])
    _patch_kb(monkeypatch, _FakeKB(counts=None))

    assert login(client).status_code == 200
    body = client.get("/knowledge/documents?with_index=true").json()
    item = body["documents"][0]
    assert item["parse_status"] == "unknown"
    assert item["chunk_count"] is None

    # 前端映射：indexed/not_indexed 之外一律「未知」。
    assert 'if (status === "indexed") return { text: "已索引"' in knowledge_js
    assert 'if (status === "not_indexed") return { text: "未索引"' in knowledge_js
    assert 'return { text: "未知", cls: "status-pill is-unknown" };' in knowledge_js
    assert "不推断为已索引" in knowledge_js


def test_B04_no_index_build_or_model_load(client, monkeypatch):
    """K2-B04：索引状态读取不触发构建或模型调用。"""
    _patch_listing(monkeypatch, [_doc("only.md")])
    kb = _patch_kb(monkeypatch, _FakeKB(counts=None))

    assert login(client).status_code == 200
    response = client.get("/knowledge/documents?with_index=true")
    assert response.status_code == 200, response.text
    # 替身只暴露 index 统计；若接口尝试 build/search 会立刻失败。
    assert kb.calls == [("indexed_chunk_counts",)]


# ============================================================
# C. 知识搜索
# ============================================================


def test_C01_search_only_by_user_submit(knowledge_js):
    """K2-C01：搜索只由表单提交触发，页面加载不自动检索。"""
    assert knowledge_js.count("submitSearch(") == 2  # 定义 + 提交回调内调用
    assert 'addEventListener("submit"' in knowledge_js
    assert "preventDefault()" in knowledge_js
    # 唯一的 search 端点拼接发生在 submitSearch 内部。
    search_url_at = knowledge_js.index("SEARCH_ENDPOINT +")
    submit_fn_at = knowledge_js.index("function submitSearch()")
    init_at = knowledge_js.index("function init()")
    assert submit_fn_at < search_url_at < init_at


def test_C02_empty_input_sends_no_request(knowledge_js):
    """K2-C02：空输入在前端拦截，不发送请求。"""
    assert "var query = raw.trim();" in knowledge_js
    guard_at = knowledge_js.index("if (!query) {")
    url_at = knowledge_js.index("var url =", guard_at)
    assert guard_at < url_at, "空输入分支必须早于请求拼接"
    assert "空输入不会发送请求" in knowledge_js


def test_C03_default_top_k_is_three(client, monkeypatch, knowledge_js):
    """K2-C03：top_k 默认 3（前端常量 + 后端缺省一致）。"""
    assert "var DEFAULT_TOP_K = 3;" in knowledge_js

    payload = {
        "ok": True,
        "query": "报销流程",
        "count": 1,
        "results": [
            {
                "source": "制度.md",
                "chunk_id": "c1",
                "score": 0.8123,
                "chunk": "报销需在 30 日内提交。",
            }
        ],
        "context": "【相关资料】\n来源：制度.md\n",
    }
    kb = _patch_kb(monkeypatch, _FakeKB(search_payload=payload))

    assert login(client).status_code == 200
    body = client.get("/knowledge/search", params={"query": "报销流程", "top_k": 3}).json()
    assert body["ok"] is True
    assert kb.calls[0][2] == 3


def test_C04_top_k_range_enforced(client, monkeypatch, knowledge_js):
    """K2-C04：top_k 限定 1–10（前端 clamp + 后端 input 分支）。"""
    assert "var MIN_TOP_K = 1;" in knowledge_js
    assert "var MAX_TOP_K = 10;" in knowledge_js
    assert "function clampTopK(value)" in knowledge_js
    assert 'if (n < MIN_TOP_K) return MIN_TOP_K;' in knowledge_js
    assert 'if (n > MAX_TOP_K) return MAX_TOP_K;' in knowledge_js

    kb = _patch_kb(monkeypatch, _FakeKB(search_payload={"ok": True, "count": 0, "results": [], "context": ""}))
    assert login(client).status_code == 200
    for bad in (0, 11):
        response = client.get("/knowledge/search", params={"query": "x", "top_k": bad})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["ok"] is False
        assert body["stage"] == "input"
    # 接口层只如实转发；拒绝发生在检索层（与真实实现一致）。
    assert [c[2] for c in kb.calls] == [0, 11]


def test_C05_results_render(client, monkeypatch, knowledge_js):
    """K2-C05：搜索结果按实际条数渲染，字段与后端契约一致。"""
    payload = {
        "ok": True,
        "query": "年假",
        "count": 2,
        "results": [
            {"source": "员工手册.md", "chunk_id": "c1", "score": 0.9, "chunk": "年假 5 天。"},
            {"source": "制度.md", "chunk_id": "c2", "score": 0.4, "chunk": "需提前申请。"},
        ],
        "context": "【相关资料】",
    }
    _patch_kb(monkeypatch, _FakeKB(search_payload=payload))

    assert login(client).status_code == 200
    body = client.get("/knowledge/search", params={"query": "年假"}).json()
    assert body["count"] == 2
    assert [r["chunk_id"] for r in body["results"]] == ["c1", "c2"]

    # 前端：结果卡片读取 source / chunk_id / score / chunk，且全部 textContent。
    for needle in (
        'source.textContent = textOrUnknown(item && item.source);',
        'chunkId.textContent = "片段 " + textOrUnknown(item && item.chunk_id);',
        'score.textContent = "相似度 " + formatScore(item ? item.score : null);',
        "body.textContent =",
    ):
        assert needle in knowledge_js


def test_C06_empty_results_state(client, monkeypatch, knowledge_js):
    """K2-C06：count=0 时显示明确空状态。"""
    _patch_kb(
        monkeypatch,
        _FakeKB(search_payload={"ok": True, "query": "不存在", "count": 0, "results": [], "context": ""}),
    )

    assert login(client).status_code == 200
    body = client.get("/knowledge/search", params={"query": "不存在"}).json()
    assert body["ok"] is True
    assert body["count"] == 0

    assert '"未检索到相关内容（0 条命中）。"' in knowledge_js


def test_C07_context_collapsed_by_default(client, monkeypatch, knowledge_js):
    """K2-C07：context 用默认折叠的 details 展示。"""
    payload = {
        "ok": True,
        "query": "q",
        "count": 1,
        "results": [{"source": "s.md", "chunk_id": "c", "score": 0.5, "chunk": "x"}],
        "context": "【相关资料】\n来源：s.md\n",
    }
    _patch_kb(monkeypatch, _FakeKB(search_payload=payload))

    assert login(client).status_code == 200
    body = client.get("/knowledge/search", params={"query": "q"}).json()
    assert body["context"]

    # details 未设置 open（默认折叠）。
    assert 'document.createElement("details")' in knowledge_js
    assert "details.open = true" not in knowledge_js
    assert "details.setAttribute(\"open\"" not in knowledge_js
    assert "默认折叠" in knowledge_js


def test_C08_search_loading_state(knowledge_js):
    """K2-C08：搜索过程显示 loading 状态。"""
    assert '"正在检索…", "loading"' in knowledge_js
    # 限定在 submitSearch 函数体内比较（loadDocuments 也有同名调用）。
    submit_at = knowledge_js.index("function submitSearch()")
    loading_at = knowledge_js.index('"正在检索…"', submit_at)
    request_at = knowledge_js.index("return requestJSON(url)", submit_at)
    assert loading_at < request_at


def test_C09_http_200_ok_false_three_stages(client, monkeypatch, knowledge_js):
    """K2-C09：HTTP 200 + ok:false 的三个 stage 分支。"""
    for stage in ("input", "init", "search"):
        payload = {
            "ok": False,
            "stage": stage,
            "error": f"{stage} 阶段失败",
            "query": "q",
        }
        _patch_kb(monkeypatch, _FakeKB(search_payload=payload))
        assert login(client).status_code == 200
        response = client.get("/knowledge/search", params={"query": "q"})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["ok"] is False
        assert body["stage"] == stage

    # 前端对三个 stage 分别给出文案。
    assert 'if (stage === "input")' in knowledge_js
    assert 'prefix = "搜索输入有误，请检查搜索问题";' in knowledge_js
    assert 'prefix = "知识库初始化失败";' in knowledge_js
    assert 'prefix = "搜索失败";' in knowledge_js
    # ok:false 分支必须在渲染之前。
    assert knowledge_js.index("if (data.ok !== true)") < knowledge_js.index(
        "renderSearchResults(data);"
    )


def test_C10_http_error_branches(client, monkeypatch, knowledge_js):
    """K2-C10：503 / 400 / 500 错误分支。"""
    # 503：知识库不可用。
    _patch_kb(monkeypatch, None)
    assert login(client).status_code == 200
    response = client.get("/knowledge/search", params={"query": "q"})
    assert response.status_code == 503
    assert response.json()["detail"]["type"] == "knowledge_unavailable"

    # 400：参数错误。
    response = client.get("/knowledge/documents", params={"with_index": "abc"})
    assert response.status_code == 400
    assert response.json()["detail"]["type"] == "invalid_parameter"

    # 500：存储层异常。
    def boom(directory):
        raise knowledge_store.KnowledgeStoreError("无法读取")

    monkeypatch.setattr(knowledge_store, "list_documents", boom)
    response = client.get("/knowledge/documents")
    assert response.status_code == 500

    # 前端三类 HTTP 错误文案。
    assert 'if (status === 503 && type === "knowledge_unavailable")' in knowledge_js
    assert '"知识库服务暂不可用"' in knowledge_js
    assert 'if (status === 400 || status === 422)' in knowledge_js
    assert '"请求参数有误"' in knowledge_js
    assert 'if (status === 500)' in knowledge_js
    assert '"服务端错误"' in knowledge_js


def test_C11_score_not_converted_to_percentage(client, monkeypatch, knowledge_js):
    """K2-C11：score 显示原始数值的小数格式，不换算百分比。"""
    payload = {
        "ok": True,
        "query": "q",
        "count": 1,
        "results": [{"source": "s", "chunk_id": "c", "score": 0.87654321, "chunk": "x"}],
        "context": "",
    }
    _patch_kb(monkeypatch, _FakeKB(search_payload=payload))

    assert login(client).status_code == 200
    body = client.get("/knowledge/search", params={"query": "q"}).json()
    assert body["results"][0]["score"] == 0.87654321

    assert "return n.toFixed(4);" in knowledge_js
    assert "* 100" not in knowledge_js
    assert "%" not in knowledge_js


# ============================================================
# D. 安全
# ============================================================


def test_D01_backend_html_rendered_as_text(client, monkeypatch, knowledge_js):
    """K2-D01：后端返回的 HTML 字符串原样传输，前端只能作为纯文本渲染。"""
    evil = '<img src=x onerror="alert(1)">.md'
    _patch_listing(
        monkeypatch,
        [_doc(evil, file_type="md", chunk_count=None)],
    )

    assert login(client).status_code == 200
    body = client.get("/knowledge/documents").json()
    # 接口不改写内容（原样透传），安全责任在前端的纯文本渲染。
    assert body["documents"][0]["filename"] == evil

    # 前端：不出现 innerHTML；渲染一律走 textContent。
    assert "innerHTML" not in knowledge_js
    assert "insertAdjacentHTML" not in knowledge_js
    assert "document.write" not in knowledge_js
    assert knowledge_js.count("textContent") >= 10


def test_D02_no_write_methods(knowledge_js):
    """K2-D02（K3A 修订）：写请求只允许管理员上传这一处 POST。

    K3A 引入上传后，前端唯一的写请求是 POST /knowledge/documents；
    PUT / PATCH / DELETE 仍必须为零（删除属于后续阶段，本阶段严禁）。
    """
    assert knowledge_js.count('method: "POST"') == 1
    assert 'UPLOAD_ENDPOINT = "/knowledge/documents"' in knowledge_js
    for verb in ('"PUT"', '"PATCH"', '"DELETE"'):
        assert verb not in knowledge_js, f"不得出现写方法 {verb}"

    # 三处 fetch：身份接口 + requestJSON 封装（列表/检索）+ 上传。
    assert knowledge_js.count("fetch(") == 3
    assert "requestJSON(" in knowledge_js


def test_D03_upload_restricted_to_admin(knowledge_js, knowledge_html):
    """K2-D03（K3A 修订）：上传逻辑仅存在于管理员区，且无删除入口。"""
    # 上传必须使用 FormData 与 file 字段（K3A 目标）。
    assert "FormData" in knowledge_js
    assert 'form.append("file", file, file.name);' in knowledge_js
    assert 'type="file"' in knowledge_html

    # 上传区初始 hidden，只有 admin 由脚本显示（fail-closed）。
    assert 'id="upload-section"' in knowledge_html
    assert 'aria-labelledby="upload-title" hidden' in knowledge_html

    # 不手动设置 multipart 的 Content-Type（交给浏览器生成 boundary）。
    assert "Content-Type" not in knowledge_js

    # 仍然没有任何删除 / 批量操作控件。
    assert re.search(r"<button[^>]*>\s*(删除|批量删除)", knowledge_html) is None


def test_D04_no_admin_write_controls(knowledge_html):
    """K2-D04（K3A 修订）：页面不提供删除 / 批量删除等管理写操作入口。

    K3A 允许管理员上传（上传按钮合法存在），但仍严禁删除类入口。
    """
    assert re.search(r"<button[^>]*>\s*(删除|批量)", knowledge_html) is None
    assert 'data-action="delete"' not in knowledge_html
    assert "admin" not in knowledge_html.lower()


def test_D05_unknown_role_fail_closed(knowledge_js, knowledge_html):
    """K2-D05：未知角色 fail-closed，不出现管理员入口。"""
    assert "未知角色（权限未确认）" in knowledge_js
    assert '"is-error"' in knowledge_js
    # 页面本身没有任何按角色显示的管理入口。
    assert "data-requires-role" not in knowledge_html


def test_D06_k1_identity_and_page_guard_preserved(client, knowledge_js):
    """K2-D06：不破坏 K1 身份识别与页面守卫契约。"""
    # 页面守卫：未登录 302 /login（带 next）。
    anonymous = client.get("/knowledge-ui")
    assert anonymous.status_code == 302
    assert anonymous.headers["location"].split("?")[0] == "/login"

    # 身份识别契约保留。
    assert 'IDENTITY_ENDPOINT = "/auth/me"' in knowledge_js
    assert "window.location.replace(LOGIN_PAGE)" in knowledge_js
    assert "window.location.replace(CHANGE_PASSWORD_PAGE)" in knowledge_js
    assert "must_change_password === true" in knowledge_js

    # 业务接口对匿名请求一律 401（不泄露数据）。
    assert client.get("/knowledge/documents").status_code == 401
    assert client.get("/knowledge/search", params={"query": "x"}).status_code == 401
