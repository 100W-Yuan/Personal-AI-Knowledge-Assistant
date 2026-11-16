# -*- coding: utf-8 -*-
"""P2-3C-B：真实 HTTP + 本地 Embedding + RAG 端到端测试。

本文件是本轮**唯一**新增的文件，用来验证 P2-3C-A 结论中「零改码即可安全
跑真实链路」是否成立。链路全部为生产实现，无任何替身：

    Uvicorn(真实 TCP) → FastAPI(main.app) → 上传/列表/删除/检索路由
        → knowledge_store（落盘） → KnowledgeBase（惰性索引）
        → rag.loader（真实 pypdf / python-docx / 文本读取）
        → TextSplitter → rag.embedding.Embedding(SentenceTransformer)
        → VectorStore → Retriever → ContextBuilder

边界声明（不得据此扩大结论）
------------------------------
1. **不使用** TestClient / ASGITransport / httpx.MockTransport：客户端是真实
   ``httpx.Client``，服务端是真实 Uvicorn TCP socket，走完整 HTTP 栈。
2. **不使用** subprocess 启动服务：Uvicorn 在本 pytest 进程的非主线程中运行，
   因此 conftest 的 socket/DNS/UDP/HTTP/SDK 守卫与数据库指纹哨兵**对服务端
   同样生效**（另起进程会让这些守卫全部失效）。
3. 全程离线：``HF_HUB_OFFLINE`` / ``TRANSFORMERS_OFFLINE`` /
   ``HF_HUB_DISABLE_TELEMETRY`` 在**任何** sentence_transformers 导入之前设置。
   模型只从 ``~/.cache/huggingface`` 读取；一旦尝试联网，conftest 的 DNS/连接
   守卫会立刻抛错，用例即失败。
4. ``lifespan="off"``：跳过 ``main.lifespan``，因此 ``init_database()`` 不会
   执行、正式 ``data/business.db`` 不被打开。另有 fail-fast 断言：把
   ``main.init_database`` 换成「一被调用就失败」的函数，测试结束断言零调用。
5. 知识库目录通过 ``main.KNOWLEDGE_DIR`` 重定向到本用例的 ``tmp_path``，并清空
   ``get_knowledge_base`` / ``get_agent`` 两级 LRU 缓存，正式 ``rag_knowledge/``
   全程不被读取。
6. 夹具严格分离两个目录：``tmp_path/src`` 只存放**待上传的源文件**，
   ``tmp_path/kb`` 是唯一知识库目录且**初始为空**——文件只能经 HTTP 上传进入
   ``kb/``。P2-3C-B 首次执行失败正是因为把源文件预写进了 ``kb/``，随后上传
   同名文件被 ``import_document`` 的 ``lexists`` 判重拦成 409 ``already_exists``。
7. 不调用 ``/chat``（会构造真实 Agent 并写 ``memory.db``）。本轮只用四条
   知识库路由。

证据输出
--------
测试把关键证据（实际端口、模型信息、各阶段响应、索引统计）写入
``tmp_path/e2e_evidence.json``。该文件位于 pytest 外部临时目录内，属于任务书
允许写入的运行数据。
"""

import contextlib
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# 1. 离线开关：必须在任何 sentence_transformers / huggingface_hub 导入之前设置。
#    本文件因此在模块顶部就写环境变量，且不在此处 import 任何 rag / main 模块。
# ---------------------------------------------------------------------------
OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}
for _key, _value in OFFLINE_ENV.items():
    os.environ[_key] = _value

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
for _path in (str(PROJECT_ROOT), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import httpx                                    # noqa: E402
import uvicorn                                  # noqa: E402

# 复用 P2-3B1 已验证过的真实 PDF 构造器（Type0/Identity-H + ToUnicode CMap，
# pypdf 能还原中文原文；该 helper 的「可被 pypdf 读取」由
# test_rag_offline_e2e.py::test_B01_pdf_fixture_is_readable_by_pypdf 单独锁定）。
from test_rag_offline_e2e import _build_pdf     # noqa: E402

MODEL_NAME = "BAAI/bge-small-zh-v1.5"
MODEL_DIM = 512
QUERY = "知识库检索测试"

# 四份文档共用 QUERY 作为公共检索词（保证首次检索能覆盖全部文档），
# 各自再带一个独特词（用于证明「命中的确实是这一份」）。
TXT_NAME = "e2e-txt.txt"
TXT_TEXT = "知识库检索测试。独特词：蓝鲸。"

MD_NAME = "e2e-md.md"
MD_TEXT = "# 知识库检索测试\n\n独特词：赤狐。"

DOCX_NAME = "e2e-docx.docx"
DOCX_HEADING = "知识库说明"
DOCX_BODY = "知识库检索测试。独特词：雪豹。"

PDF_NAME = "e2e-pdf.pdf"
PDF_TEXT = "知识库检索测试。独特词：金雕。"

EXTRA_NAME = "e2e-extra.txt"
EXTRA_TEXT = "知识库检索测试。独特词：白鹭。"


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _document_id(filename):
    """与 knowledge_store / KnowledgeBase 同算法：sha256(filename)[:32]。"""
    return hashlib.sha256(filename.encode("utf-8")).hexdigest()[:32]


def _dump(path, payload):
    """把证据写到临时目录；失败时保留已完成的部分，便于事后诊断。"""
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


@contextlib.contextmanager
def _running_uvicorn(app):
    """在非主线程启动真实 Uvicorn，产出 (port, server)。退出时确保停止并 join。"""
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=0,                 # 由内核分配空闲端口
        lifespan="off",         # 不执行 main.lifespan（不初始化 business.db）
        access_log=False,
        log_level="error",
    )
    server = uvicorn.Server(config)

    # uvicorn 0.52.4 的 signal 安装位于 Server.capture_signals()，该函数在
    # 非主线程直接 yield（见 uvicorn/server.py），因此线程内启动不会安装
    # signal handler。下面这行是兼容旧版的防御，旧版把安装写在 run() 里。
    if hasattr(server, "install_signal_handlers"):
        server.install_signal_handlers = lambda: None

    thread = threading.Thread(
        target=server.run, name="p2-3c-b-uvicorn", daemon=True
    )
    thread.start()

    try:
        deadline = time.monotonic() + 30
        while not server.started:
            if not thread.is_alive():
                raise AssertionError("Uvicorn 线程已退出，服务未能启动")
            if time.monotonic() > deadline:
                raise AssertionError("等待 server.started 超时（30s）")
            time.sleep(0.05)

        sockets = server.servers[0].sockets
        assert sockets, "Uvicorn 未创建监听 socket"
        host, port = sockets[0].getsockname()[:2]
        assert host == "127.0.0.1", host
        yield int(port), server
    finally:
        server.should_exit = True
        thread.join(timeout=30)
        assert not thread.is_alive(), "Uvicorn 线程未能在 30s 内退出"


def _write_docx(path):
    """用 python-docx 生成真实 DOCX：1 个 Heading 1 + 1 段正文。"""
    import docx

    document = docx.Document()
    document.add_heading(DOCX_HEADING, level=1)
    document.add_paragraph(DOCX_BODY)
    document.save(str(path))


# ---------------------------------------------------------------------------
# 主用例
# ---------------------------------------------------------------------------
@pytest.mark.fresh_process
def test_real_http_rag_end_to_end(tmp_path, monkeypatch):
    """真实 HTTP + 真实 Loader + 真实 Embedding 的四阶段端到端验收。"""
    evidence = {"phases": {}, "offline_env": dict(OFFLINE_ENV)}
    evidence_path = tmp_path / "e2e_evidence.json"

    # =====================================================================
    # 阶段 0：导入前的隔离前置
    # =====================================================================
    fake_dotenv = sys.modules.get("dotenv")
    assert fake_dotenv is not None, "conftest 未注入假 dotenv 模块"
    assert getattr(fake_dotenv.load_dotenv, "_wb_fake_dotenv", False) is True, (
        "sys.modules['dotenv'].load_dotenv 不是 conftest 的假实现"
    )
    assert "main" not in sys.modules, (
        "main 在用例开始前已被导入：假 dotenv 无法保证生效，必须中止"
    )
    assert "sentence_transformers" not in sys.modules, (
        "sentence_transformers 在离线变量设置之前已被导入"
    )
    for key, value in OFFLINE_ENV.items():
        assert os.environ.get(key) == value, (key, os.environ.get(key))

    # 夹具：src/ 只放「待上传的源文件」，kb/ 是唯一知识库目录且必须保持为空。
    # P2-3C-B 首次执行失败的原因正是把源文件写进了 kb/，随后 HTTP 上传同名
    # 文件被 knowledge_store.import_document 的 lexists 判重拦成 409。
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    kbdir = tmp_path / "kb"
    kbdir.mkdir()

    txt_path = src_dir / TXT_NAME
    txt_path.write_text(TXT_TEXT, encoding="utf-8")
    md_path = src_dir / MD_NAME
    md_path.write_text(MD_TEXT, encoding="utf-8")
    docx_path = src_dir / DOCX_NAME
    _write_docx(docx_path)
    pdf_path = src_dir / PDF_NAME
    pdf_path.write_bytes(_build_pdf([PDF_TEXT]))

    payloads = {
        TXT_NAME: txt_path.read_bytes(),
        MD_NAME: md_path.read_bytes(),
        DOCX_NAME: docx_path.read_bytes(),
        PDF_NAME: pdf_path.read_bytes(),
    }

    # ---- 首次 HTTP 上传前的结构断言（缺一不可，否则会重演 409）----
    assert kbdir.is_dir() and not kbdir.is_symlink()
    assert src_dir.is_dir() and not src_dir.is_symlink()
    assert list(kbdir.iterdir()) == [], "kb/ 必须初始为空"
    for name in payloads:
        assert not (kbdir / name).exists(), (name, "源文件不得预先写入 kb/")
    assert src_dir.resolve() != kbdir.resolve()
    assert not src_dir.resolve().is_relative_to(kbdir.resolve())
    assert not kbdir.resolve().is_relative_to(src_dir.resolve())
    # 四份内容互不相同：否则 import_document 的内容去重会返回 409 duplicate_content
    assert len({hashlib.sha256(data).hexdigest() for data in payloads.values()}) == 4

    # =====================================================================
    # 阶段 0.5：import main（此时假 dotenv 已在位）并重定向知识库
    # =====================================================================
    import main

    assert getattr(main.load_dotenv, "_wb_fake_dotenv", False) is True, (
        "main.load_dotenv 不是假实现：.env 可能已被真实读取"
    )

    # fail-fast：lifespan=off 时 init_database 不应被调用。
    init_calls = []

    def _forbid_init_database(*args, **kwargs):
        init_calls.append((args, kwargs))
        raise AssertionError(
            "main.init_database 被调用：lifespan 未关闭，正式 data/business.db 可能被写入"
        )

    monkeypatch.setattr(main, "init_database", _forbid_init_database)

    monkeypatch.setattr(main, "KNOWLEDGE_DIR", str(kbdir))
    main.get_knowledge_base.cache_clear()
    main.get_agent.cache_clear()
    assert main.KNOWLEDGE_DIR == str(kbdir)
    assert main.get_knowledge_base.cache_info().currsize == 0
    assert main.get_agent.cache_info().currsize == 0
    assert main.app.state.business_db_available is False

    # PAI-AUTH-MVP：知识库的变更接口现在要求管理员。本用例验证的是
    # 上传/列表/删除/检索的真实 HTTP 链路的业务语义，不是鉴权本身
    # （鉴权由 测试/test_auth_mvp.py 覆盖），因此这里替换身份依赖。
    # 必须在 _running_uvicorn 之前设置：override 挂在 app 对象上，
    # 启动后再改也能生效，但提前设置更符合「请求发生前就绪」。
    import auth.deps as auth_deps

    _e2e_admin = {
        "id": "e2e-admin",
        "username": "e2e-admin",
        "role": "admin",
        "is_active": True,
        "created_at": "2026-01-01T00:00:00Z",
    }
    main.app.dependency_overrides[auth_deps.get_current_user] = lambda: _e2e_admin
    main.app.dependency_overrides[auth_deps.get_optional_user] = lambda: _e2e_admin
    main.app.dependency_overrides[auth_deps.get_page_user] = lambda: _e2e_admin
    main.app.dependency_overrides[auth_deps.require_admin] = lambda: _e2e_admin

    evidence["phases"]["phase0_isolation"] = {
        "fake_dotenv": True,
        "src_dir": str(src_dir),
        "knowledge_dir": str(kbdir),
        "kb_initial_entries": 0,
        "kb_cache_size": 0,
        "agent_cache_size": 0,
        "business_db_available": False,
        "sentence_transformers_preloaded": False,
    }
    _dump(evidence_path, evidence)

    with _running_uvicorn(main.app) as (port, _server):
        base_url = "http://127.0.0.1:%d" % port
        evidence["server"] = {"host": "127.0.0.1", "port": port, "lifespan": "off",
                              "base_url": base_url}
        _dump(evidence_path, evidence)

        with httpx.Client(base_url=base_url, timeout=180.0, trust_env=False) as client:

            # =============================================================
            # 阶段 1：四种真实文件上传 + 列表 + 未索引状态
            # =============================================================
            uploads = {}
            for name in (TXT_NAME, MD_NAME, DOCX_NAME, PDF_NAME):
                data = payloads[name]
                response = client.post(
                    "/knowledge/documents",
                    files={"file": (name, data, "application/octet-stream")},
                )
                body = response.json()
                assert response.status_code == 200, (name, response.status_code, body)
                assert body["ok"] is True, (name, body)
                assert body["filename"] == name, (name, body)
                assert body["bytes"] == len(data), (name, body, len(data))
                assert body["document_id"] == _document_id(name), (name, body)
                assert body["storage_type"] == "persistent", (name, body)
                uploads[name] = body

            # 上传成功后，四份文件必须真实落在知识库目录里（且字节完全一致）
            landed = {path.name for path in kbdir.iterdir()}
            assert landed == set(payloads), landed
            for name, data in payloads.items():
                assert (kbdir / name).read_bytes() == data, name
            assert len(list(kbdir.iterdir())) == 4, len(list(kbdir.iterdir()))

            listed = client.get("/knowledge/documents")
            listed_body = listed.json()
            assert listed.status_code == 200, listed_body
            assert listed_body["ok"] is True
            assert listed_body["directory"] == str(kbdir), listed_body["directory"]
            assert listed_body["directory_exists"] is True
            assert listed_body["count"] == 4, listed_body["count"]
            assert {d["filename"] for d in listed_body["documents"]} == set(payloads)
            assert {d["document_id"] for d in listed_body["documents"]} == {
                _document_id(name) for name in payloads
            }
            assert listed_body["with_index"] is False

            # 首次 search 之前：with_index=true 不得触发索引
            pre_index = client.get(
                "/knowledge/documents", params={"with_index": "true"}
            )
            pre_body = pre_index.json()
            assert pre_index.status_code == 200, pre_body
            assert pre_body["with_index"] is True
            assert pre_body["count"] == 4
            assert all(d["parse_status"] == "unknown" for d in pre_body["documents"])
            assert all(d["chunk_count"] is None for d in pre_body["documents"])
            # index_skipped_reparse 的取值时机（main.py:785-792）：只有
            # indexed_chunk_counts() 返回非 None（索引已构建）时才读取
            # kb.skipped_reparse_count；否则保持局部变量初值 None。
            assert pre_body["index_skipped_reparse"] is None, (
                pre_body["index_skipped_reparse"]
            )
            # 关键证据：该 GET 只读索引状态，没有加载模型
            assert "sentence_transformers" not in sys.modules, (
                "with_index=true 不应加载 Embedding 模型"
            )
            assert main.get_knowledge_base().is_indexed is False, (
                "with_index=true 不应触发索引构建"
            )

            evidence["phases"]["phase1_upload"] = {
                "uploads": uploads,
                "list_count": listed_body["count"],
                "list_directory": listed_body["directory"],
                "pre_index_status": sorted(
                    {d["parse_status"] for d in pre_body["documents"]}
                ),
                "sentence_transformers_loaded_before_search": False,
                "kb_is_indexed_before_search": False,
            }
            _dump(evidence_path, evidence)

            # =============================================================
            # 阶段 2：真实 Embedding + 真实检索
            # =============================================================
            first = client.get(
                "/knowledge/search",
                params={"query": QUERY, "top_k": 10, "threshold": 0.0},
            )
            first_body = first.json()
            assert first.status_code == 200, (first.status_code, first_body)
            assert first_body["ok"] is True, first_body
            assert first_body["query"] == QUERY
            assert first_body["count"] == len(first_body["results"]), first_body["count"]
            assert first_body["count"] == 4, first_body["count"]

            by_source = {}
            for record in first_body["results"]:
                by_source.setdefault(record["source"], []).append(record)
            assert set(by_source) == set(payloads), sorted(by_source)

            # 四种文档的 document_id 全部出现
            found_ids = {record["document_id"] for record in first_body["results"]}
            assert found_ids == {_document_id(name) for name in payloads}, found_ids

            # 命中内容确实来自对应文件（独特词逐份核对）
            assert "蓝鲸" in by_source[TXT_NAME][0]["chunk"]
            assert "赤狐" in by_source[MD_NAME][0]["chunk"]
            assert "雪豹" in by_source[DOCX_NAME][0]["chunk"]
            assert "金雕" in by_source[PDF_NAME][0]["chunk"]

            # 来源字段不得伪造：TXT / MD 没有 page、section、table 概念
            for name in (TXT_NAME, MD_NAME):
                record = by_source[name][0]
                for key in ("page", "page_count", "section", "section_index",
                            "segment_index", "table_index"):
                    assert key not in record, (name, key, record)

            # PDF 保留页码，且没有 section 概念
            pdf_record = by_source[PDF_NAME][0]
            assert pdf_record["page"] == 1, pdf_record
            assert pdf_record["page_count"] == 1, pdf_record
            for key in ("section", "section_index", "segment_index", "table_index"):
                assert key not in pdf_record, (PDF_NAME, key, pdf_record)

            # DOCX 保留章节定位，且没有 page 概念
            docx_record = by_source[DOCX_NAME][0]
            assert docx_record["section"] == DOCX_HEADING, docx_record
            assert docx_record["section_index"] == 0, docx_record
            assert docx_record["segment_index"] == 0, docx_record
            for key in ("page", "page_count", "table_index"):
                assert key not in docx_record, (DOCX_NAME, key, docx_record)

            assert isinstance(first_body["context"], str) and first_body["context"].strip()

            # ---- 真实模型证据（不能只看"导入了类名"）----
            assert "sentence_transformers" in sys.modules, "检索后仍未见模型库"
            import sentence_transformers

            kb = main.get_knowledge_base()
            assert kb is not None and kb.is_indexed is True
            assert kb.file_count == 4, kb.file_count
            assert kb.chunk_count == 4, kb.chunk_count
            assert kb.init_error_kind is None, kb.init_error_kind

            embedding = kb._embedding
            assert type(embedding).__module__ == "rag.embedding", type(embedding)
            assert type(embedding).__name__ == "Embedding", type(embedding)
            assert not hasattr(embedding, "_wb_offline_shell")
            assert not hasattr(embedding, "calls"), "疑似测试替身（带 calls 计数器）"

            model = embedding.model
            assert isinstance(model, sentence_transformers.SentenceTransformer), type(model)
            assert model.get_sentence_embedding_dimension() == MODEL_DIM, (
                model.get_sentence_embedding_dimension()
            )
            auto_model = model[0].auto_model
            assert MODEL_NAME.split("/")[-1] in str(auto_model.name_or_path), (
                auto_model.name_or_path
            )

            # 向量确实由真实模型编码：512 维 float32、L2 归一化、互不相同
            store_items = kb._retriever.vector_store.items
            assert len(store_items) == 4, len(store_items)
            vectors = []
            for item in store_items:
                vector = item["vector"]
                assert type(vector).__module__.startswith("numpy"), type(vector)
                assert tuple(vector.shape) == (MODEL_DIM,), vector.shape
                assert str(vector.dtype) == "float32", vector.dtype
                norm = float((vector.astype("float64") ** 2).sum() ** 0.5)
                assert abs(norm - 1.0) < 1e-4, norm
                vectors.append(vector.tobytes())
            assert len(set(vectors)) == 4, "四份文档得到了同一向量（疑似替身）"

            # 索引状态：四份文件均为 indexed，chunk_count == 1
            post_index = client.get(
                "/knowledge/documents", params={"with_index": "true"}
            )
            post_body = post_index.json()
            assert post_index.status_code == 200, post_body
            assert post_body["index_skipped_reparse"] == 0, post_body["index_skipped_reparse"]
            assert all(d["parse_status"] == "indexed" for d in post_body["documents"])

            # chunk_count == 1 是**推导值**，不是经验值：
            # - DOCX 只有 1 个 Heading 1 + 1 段正文；DocxLoader 的 _flush_segment
            #   把「标题 + 其后的段落」并入同一个 segment（loader.py:530-541、
            #   603-605），因此只产出 1 个 Document，正文 21 字符；
            # - TextSplitter(chunk_size=100) 对 len(text) <= 100 的文本只产出 1 个
            #   chunk，并在 end >= len(text) 时立即 break（splitter.py:28-51）；
            # - TXT / MD 各只有 1 个短 Document，PDF 为单页（page_count=1）。
            chunk_counts = {
                d["filename"]: d["chunk_count"] for d in post_body["documents"]
            }
            assert chunk_counts == {name: 1 for name in payloads}, chunk_counts

            evidence["phases"]["phase2_real_retrieval"] = {
                "query": QUERY,
                "count": first_body["count"],
                "sources": sorted(by_source),
                "scores": {r["source"]: round(r["score"], 6) for r in first_body["results"]},
                "context_length": len(first_body["context"]),
                "model": {
                    "name_or_path": str(auto_model.name_or_path),
                    "class": type(model).__name__,
                    "dim": model.get_sentence_embedding_dimension(),
                    "file_count": kb.file_count,
                    "chunk_count": kb.chunk_count,
                    "vector_dtype": "float32",
                    "vector_norm": 1.0,
                },
                "index_status": {d["filename"]: [d["parse_status"], d["chunk_count"]]
                                 for d in post_body["documents"]},
            }
            _dump(evidence_path, evidence)

            # 释放强引用：后续阶段只从 LRU 缓存取 KB 实例
            del kb
            del embedding

            # =============================================================
            # 阶段 3：上传后缓存失效与真实重建
            # =============================================================
            assert main.get_knowledge_base.cache_info().currsize == 1
            assert main.get_agent.cache_info().currsize == 0

            extra_src = src_dir / EXTRA_NAME
            extra_src.write_text(EXTRA_TEXT, encoding="utf-8")
            extra_bytes = extra_src.read_bytes()
            # 与四份初始文件同一纪律：不得预先写进知识库目录
            assert not (kbdir / EXTRA_NAME).exists()

            extra = client.post(
                "/knowledge/documents",
                files={"file": (EXTRA_NAME, extra_bytes, "application/octet-stream")},
            )
            extra_body = extra.json()
            assert extra.status_code == 200, extra_body
            assert extra_body["ok"] is True
            assert extra_body["document_id"] == _document_id(EXTRA_NAME)
            assert (kbdir / EXTRA_NAME).read_bytes() == extra_bytes
            assert len(list(kbdir.iterdir())) == 5, len(list(kbdir.iterdir()))

            assert main.get_knowledge_base.cache_info().currsize == 0, (
                "上传成功后 KnowledgeBase 缓存未失效"
            )
            assert main.get_agent.cache_info().currsize == 0, (
                "上传成功后 Agent 缓存未失效"
            )

            second = client.get(
                "/knowledge/search",
                params={"query": QUERY, "top_k": 10, "threshold": 0.0},
            )
            second_body = second.json()
            assert second.status_code == 200, second_body
            assert second_body["ok"] is True, second_body
            assert second_body["count"] == 5, second_body["count"]
            second_sources = {r["source"] for r in second_body["results"]}
            assert second_sources == set(payloads) | {EXTRA_NAME}, sorted(second_sources)
            assert _document_id(EXTRA_NAME) in {
                r["document_id"] for r in second_body["results"]
            }
            assert "白鹭" in next(
                r["chunk"] for r in second_body["results"] if r["source"] == EXTRA_NAME
            )

            rebuilt_kb = main.get_knowledge_base()
            assert rebuilt_kb.is_indexed is True
            assert rebuilt_kb.file_count == 5, rebuilt_kb.file_count
            assert rebuilt_kb.chunk_count == 5, rebuilt_kb.chunk_count
            assert type(rebuilt_kb._embedding).__module__ == "rag.embedding"
            assert isinstance(
                rebuilt_kb._embedding.model, sentence_transformers.SentenceTransformer
            ), type(rebuilt_kb._embedding.model)
            assert rebuilt_kb._embedding.model.get_sentence_embedding_dimension() == MODEL_DIM
            rebuilt_items = rebuilt_kb._retriever.vector_store.items
            assert len(rebuilt_items) == 5, len(rebuilt_items)

            evidence["phases"]["phase3_upload_rebuild"] = {
                "cache_before_upload": 1,
                "cache_after_upload": 0,
                "agent_cache_after_upload": 0,
                "count_after_rebuild": second_body["count"],
                "sources_after_rebuild": sorted(second_sources),
                "file_count": rebuilt_kb.file_count,
                "chunk_count": rebuilt_kb.chunk_count,
                "embedding_class": type(rebuilt_kb._embedding).__module__
                                   + "." + type(rebuilt_kb._embedding).__name__,
                "model_dim": rebuilt_kb._embedding.model.get_sentence_embedding_dimension(),
            }
            _dump(evidence_path, evidence)

            del rebuilt_kb

            # =============================================================
            # 阶段 4：删除后缓存失效与真实重建
            # =============================================================
            pdf_id = _document_id(PDF_NAME)
            deleted = client.delete("/knowledge/documents/%s" % pdf_id)
            delete_body = deleted.json()
            assert deleted.status_code == 200, delete_body
            assert delete_body["ok"] is True
            assert delete_body["document_id"] == pdf_id
            assert delete_body["filename"] == PDF_NAME
            assert delete_body["freed_bytes"] == len(payloads[PDF_NAME])

            assert main.get_knowledge_base.cache_info().currsize == 0, (
                "删除成功后 KnowledgeBase 缓存未失效"
            )
            assert main.get_agent.cache_info().currsize == 0, (
                "删除成功后 Agent 缓存未失效"
            )

            third = client.get(
                "/knowledge/search",
                params={"query": QUERY, "top_k": 10, "threshold": 0.0},
            )
            third_body = third.json()
            assert third.status_code == 200, third_body
            assert third_body["ok"] is True, third_body
            assert third_body["count"] == 4, third_body["count"]
            third_sources = {r["source"] for r in third_body["results"]}
            assert PDF_NAME not in third_sources, sorted(third_sources)
            assert third_sources == {TXT_NAME, MD_NAME, DOCX_NAME, EXTRA_NAME}
            assert pdf_id not in {r["document_id"] for r in third_body["results"]}

            after_delete_kb = main.get_knowledge_base()
            assert after_delete_kb.is_indexed is True
            assert after_delete_kb.file_count == 4, after_delete_kb.file_count
            assert after_delete_kb.chunk_count == 4, after_delete_kb.chunk_count
            assert type(after_delete_kb._embedding).__module__ == "rag.embedding"
            assert isinstance(
                after_delete_kb._embedding.model, sentence_transformers.SentenceTransformer
            )

            # 删除后的目录列表也只剩四份
            final_listing = client.get("/knowledge/documents").json()
            assert final_listing["count"] == 4, final_listing["count"]
            assert PDF_NAME not in {d["filename"] for d in final_listing["documents"]}
            # 文件系统侧同样确认 PDF 已从知识库目录移除
            assert not (kbdir / PDF_NAME).exists()
            assert len(list(kbdir.iterdir())) == 4, len(list(kbdir.iterdir()))

            evidence["phases"]["phase4_delete_rebuild"] = {
                "deleted": delete_body,
                "cache_after_delete": 0,
                "count_after_rebuild": third_body["count"],
                "sources_after_rebuild": sorted(third_sources),
                "file_count": after_delete_kb.file_count,
                "chunk_count": after_delete_kb.chunk_count,
            }
            _dump(evidence_path, evidence)

            del after_delete_kb

            # =============================================================
            # 收尾：全局安全断言
            # =============================================================
            assert init_calls == [], "lifespan 被触发，init_database 执行过"
            assert main.app.state.business_db_available is False
            assert main.KNOWLEDGE_DIR == str(kbdir)
            # 知识库目录仍在本用例临时目录内，正式 rag_knowledge/ 未被写入
            assert Path(main.KNOWLEDGE_DIR).resolve().is_relative_to(tmp_path.resolve())

    evidence["result"] = "passed"
    _dump(evidence_path, evidence)
