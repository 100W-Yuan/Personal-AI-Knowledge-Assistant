# -*- coding: utf-8 -*-
"""知识库业务接口（/knowledge/search、/knowledge/import）离线测试。

- /knowledge/search：通过注入 stub KB 验证检索透传与 503 降级，不加载真实模型。
- /knowledge/import：通过 monkeypatch KNOWLEDGE_DIR 到临时目录验证导入、
  格式/空文件/重复/编码/路径穿越约束，不写入正式 rag_knowledge。
"""

from fastapi.testclient import TestClient

import main


client = TestClient(main.app)


class StubKB:
    """仅实现 search 的替身，返回固定结果或注入失败。"""

    def __init__(self, result):
        self.result = result
        self.calls = 0

    def search(self, query, top_k=3, threshold=None):
        self.calls += 1
        return dict(self.result, query=query)


def test_knowledge_search_returns_results(monkeypatch):
    monkeypatch.setattr(
        main,
        "get_knowledge_base",
        lambda: StubKB(
            {
                "ok": True,
                "count": 1,
                "results": [
                    {
                        "source": "a.md",
                        "chunk_id": 0,
                        "score": 0.9,
                        "chunk": "内容",
                    }
                ],
                "context": "【相关资料】内容",
            }
        ),
    )

    response = client.get("/knowledge/search", params={"query": "问题"})

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["results"][0]["source"] == "a.md"


def test_knowledge_search_business_error_passthrough(monkeypatch):
    monkeypatch.setattr(
        main,
        "get_knowledge_base",
        lambda: StubKB(
            {"ok": False, "stage": "input", "error": "知识检索问题不能为空"}
        ),
    )

    response = client.get("/knowledge/search", params={"query": "  "})

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is False
    assert body["stage"] == "input"


def test_knowledge_search_unavailable(monkeypatch):
    monkeypatch.setattr(main, "get_knowledge_base", lambda: None)

    response = client.get("/knowledge/search", params={"query": "问题"})

    assert response.status_code == 503
    assert response.json()["detail"]["type"] == "knowledge_unavailable"


def test_knowledge_import_md(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "KNOWLEDGE_DIR", str(tmp_path))
    main.get_knowledge_base.cache_clear()
    main.get_agent.cache_clear()

    response = client.post(
        "/knowledge/import",
        files={"file": ("doc.md", "正文内容".encode("utf-8"), "text/markdown")},
    )

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert (tmp_path / "doc.md").read_text(encoding="utf-8") == "正文内容"


def test_knowledge_import_rejects_unsupported(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "KNOWLEDGE_DIR", str(tmp_path))

    response = client.post(
        "/knowledge/import",
        files={"file": ("a.py", b"print(1)", "text/x-python")},
    )

    assert response.status_code == 400
    assert response.json()["detail"]["type"] == "unsupported_format"
    assert not (tmp_path / "a.py").exists()


def test_knowledge_import_rejects_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "KNOWLEDGE_DIR", str(tmp_path))

    response = client.post(
        "/knowledge/import",
        files={"file": ("empty.md", b"", "text/markdown")},
    )

    assert response.status_code == 400
    assert response.json()["detail"]["type"] == "empty_file"
    assert not (tmp_path / "empty.md").exists()


def test_knowledge_import_rejects_duplicate(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "KNOWLEDGE_DIR", str(tmp_path))

    payload = ("dup.md", "内容".encode("utf-8"), "text/markdown")
    first = client.post("/knowledge/import", files={"file": payload})
    second = client.post("/knowledge/import", files={"file": payload})

    assert first.status_code == 200
    assert second.status_code == 409
    assert second.json()["detail"]["type"] == "already_exists"


def test_knowledge_import_rejects_non_utf8(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "KNOWLEDGE_DIR", str(tmp_path))

    response = client.post(
        "/knowledge/import",
        files={"file": ("bad.md", b"\xff\xfe\x00garbage", "text/markdown")},
    )

    assert response.status_code == 400
    assert response.json()["detail"]["type"] == "invalid_encoding"
    assert not (tmp_path / "bad.md").exists()


def test_knowledge_import_sanitizes_path_traversal(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "KNOWLEDGE_DIR", str(tmp_path))

    response = client.post(
        "/knowledge/import",
        files={"file": ("../../evil.md", "内容".encode("utf-8"), "text/markdown")},
    )

    assert response.status_code == 200
    assert response.json()["filename"] == "evil.md"
    assert (tmp_path / "evil.md").exists()
    assert not (tmp_path / ".." / "evil.md").exists()


def test_knowledge_import_invalidates_index_caches(monkeypatch, tmp_path):
    """导入成功后必须失效知识库与 Agent 缓存（各一次），使下次检索纳入新文档。"""
    monkeypatch.setattr(main, "KNOWLEDGE_DIR", str(tmp_path))

    class _SpyCache:
        def __init__(self):
            self.clear_calls = 0

        def cache_clear(self):
            self.clear_calls += 1

        def __call__(self, *args, **kwargs):
            return None

    kb_spy = _SpyCache()
    agent_spy = _SpyCache()
    monkeypatch.setattr(main, "get_knowledge_base", kb_spy)
    monkeypatch.setattr(main, "get_agent", agent_spy)

    response = client.post(
        "/knowledge/import",
        files={"file": ("new.md", "内容".encode("utf-8"), "text/markdown")},
    )

    assert response.status_code == 200
    assert kb_spy.clear_calls == 1
    assert agent_spy.clear_calls == 1
