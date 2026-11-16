# -*- coding: utf-8 -*-
"""阶段一：知识库文档管理接口（离线）。

覆盖《阶段一实施前设计审查 V1.6》：

- GET  /knowledge/documents：字段、排序、过滤、with_index 取值与副作用边界；
- DELETE /knowledge/documents/{document_id}：错误码矩阵、路径安全、身份复核；
- POST /knowledge/import：文件名合法性、同名冲突、内容去重、目录信任、回滚失败；
- 缓存一致性：成功才清理、顺序固定、lambda 替身不抛 AttributeError。

不加载真实模型、不联网、不写正式 rag_knowledge 目录。
重解析点相关用例通过 stub 的 scandir / stat 注入，属**模拟测试**。
"""

import hashlib
import os
import stat
import types

import pytest
from fastapi.testclient import TestClient

import knowledge_store
import main
import rag.knowledge_base as kb_module


client = TestClient(main.app)


def _doc_id(name):
    return hashlib.sha256(name.encode("utf-8")).hexdigest()[:32]


def _clear(factory):
    clear = getattr(factory, "cache_clear", None)
    if callable(clear):
        clear()


@pytest.fixture(autouse=True)
def _reset_caches():
    _clear(main.get_knowledge_base)
    _clear(main.get_agent)
    yield
    _clear(main.get_knowledge_base)
    _clear(main.get_agent)


class _FakeEntry:
    def __init__(self, path, kind="regular", stat_result=None):
        self.path = str(path)
        self.name = os.path.basename(self.path)
        self._kind = kind
        self._stat = stat_result

    def is_symlink(self):
        return self._kind in ("reparse", "symlink")

    def is_file(self, follow_symlinks=True):
        return self._kind == "regular"

    def is_dir(self, follow_symlinks=True):
        return self._kind == "dir"

    def stat(self, follow_symlinks=True):
        if self._stat is not None:
            return self._stat
        return os.stat(self.path, follow_symlinks=follow_symlinks)


class _FakeScandir:
    def __init__(self, entries):
        self._entries = list(entries)

    def __enter__(self):
        return iter(self._entries)

    def __exit__(self, *exc_info):
        return False


def _patch_scandir(monkeypatch, entries):
    monkeypatch.setattr(
        knowledge_store.os, "scandir", lambda _path: _FakeScandir(entries)
    )


def _reparse_stat(base_stat):
    return types.SimpleNamespace(
        st_mode=base_stat.st_mode,
        st_dev=base_stat.st_dev,
        st_ino=base_stat.st_ino,
        st_size=base_stat.st_size,
        st_mtime=base_stat.st_mtime,
        st_ctime=base_stat.st_ctime,
        st_file_attributes=stat.FILE_ATTRIBUTE_REPARSE_POINT,
    )


class _SpyCache:
    """记录 cache_clear 调用顺序的替身（可调用，也可被 cache_clear）。"""

    def __init__(self, label, sink):
        self.label = label
        self.sink = sink

    def __call__(self, *args, **kwargs):
        return None

    def cache_clear(self):
        self.sink.append(self.label)


# ============================================================
# GET /knowledge/documents
# ============================================================


def test_A01_empty_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "KNOWLEDGE_DIR", str(tmp_path))

    response = client.get("/knowledge/documents")

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["directory_exists"] is True
    assert body["count"] == 0
    assert body["documents"] == []
    assert body["with_index"] is False
    assert body["index_skipped_reparse"] is None


def test_A02_missing_directory_returns_200_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "KNOWLEDGE_DIR", str(tmp_path / "nope"))

    response = client.get("/knowledge/documents")

    assert response.status_code == 200
    body = response.json()
    assert body["directory_exists"] is False
    assert body["documents"] == []


def test_A03_directory_is_plain_file(monkeypatch, tmp_path):
    target = tmp_path / "kb"
    target.write_text("not a dir", encoding="utf-8")
    monkeypatch.setattr(main, "KNOWLEDGE_DIR", str(target))

    response = client.get("/knowledge/documents")

    assert response.status_code == 500
    assert response.json()["detail"]["type"] == "knowledge_dir_invalid"


def test_A04_directory_is_reparse_point(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "KNOWLEDGE_DIR", str(tmp_path))
    monkeypatch.setattr(
        knowledge_store,
        "inspect_knowledge_directory",
        lambda _path: "untrusted",
    )

    response = client.get("/knowledge/documents")

    assert response.status_code == 503
    assert response.json()["detail"]["type"] == "knowledge_dir_untrusted"


def test_A05_document_metadata_fields(monkeypatch, tmp_path):
    (tmp_path / "a.txt").write_text("内容\n" * 5, encoding="utf-8")
    monkeypatch.setattr(main, "KNOWLEDGE_DIR", str(tmp_path))

    body = client.get("/knowledge/documents").json()
    document = body["documents"][0]

    assert document["document_id"] == _doc_id("a.txt")
    assert len(document["document_id"]) == 32
    assert set(document["document_id"]) <= set("0123456789abcdef")
    assert document["filename"] == "a.txt"
    assert document["file_type"] == "txt"
    assert document["file_size"] == (tmp_path / "a.txt").stat().st_size
    assert document["storage_type"] == "persistent"
    assert document["parse_status"] == "unknown"
    assert document["chunk_count"] is None
    assert document["created_at_source"] in (
        "birthtime",
        "ctime_fallback",
        "unavailable",
    )
    if document["created_at"] is not None:
        from datetime import datetime

        datetime.fromisoformat(document["created_at"])


def test_A06_ignores_non_whitelisted_suffix(monkeypatch, tmp_path):
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    (tmp_path / "b.py").write_text("print(1)", encoding="utf-8")
    (tmp_path / "c.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(main, "KNOWLEDGE_DIR", str(tmp_path))

    names = [d["filename"] for d in client.get("/knowledge/documents").json()["documents"]]

    assert names == ["a.txt"]


def test_A07_ignores_subdirectories(monkeypatch, tmp_path):
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "b.txt").write_text("y", encoding="utf-8")
    monkeypatch.setattr(main, "KNOWLEDGE_DIR", str(tmp_path))

    names = [d["filename"] for d in client.get("/knowledge/documents").json()["documents"]]

    assert names == ["a.txt"]


def test_A08_ignores_symlink_entries(monkeypatch, tmp_path):
    real = tmp_path / "a.txt"
    real.write_text("x", encoding="utf-8")
    link = tmp_path / "link.txt"
    entries = [
        _FakeEntry(real, kind="regular"),
        _FakeEntry(link, kind="symlink", stat_result=_reparse_stat(os.stat(real))),
    ]
    monkeypatch.setattr(main, "KNOWLEDGE_DIR", str(tmp_path))
    _patch_scandir(monkeypatch, entries)

    names = [d["filename"] for d in client.get("/knowledge/documents").json()["documents"]]

    assert names == ["a.txt"]


def test_A09_sorted_by_filename(monkeypatch, tmp_path):
    for name in ("b.txt", "a.md", "c.txt"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    monkeypatch.setattr(main, "KNOWLEDGE_DIR", str(tmp_path))

    names = [d["filename"] for d in client.get("/knowledge/documents").json()["documents"]]

    assert names == ["a.md", "b.txt", "c.txt"]


def test_A10_document_id_stable_across_cache_clear(monkeypatch, tmp_path):
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    monkeypatch.setattr(main, "KNOWLEDGE_DIR", str(tmp_path))

    first = client.get("/knowledge/documents").json()["documents"][0]["document_id"]
    _clear(main.get_knowledge_base)
    second = client.get("/knowledge/documents").json()["documents"][0]["document_id"]

    assert first == second


def test_A11_default_does_not_report_index_state(monkeypatch, tmp_path):
    (tmp_path / "a.txt").write_text("x" * 400, encoding="utf-8")
    monkeypatch.setattr(main, "KNOWLEDGE_DIR", str(tmp_path))

    document = client.get("/knowledge/documents").json()["documents"][0]

    assert document["parse_status"] == "unknown"
    assert document["chunk_count"] is None


def test_A12_with_index_never_builds_index(monkeypatch, tmp_path):
    (tmp_path / "a.txt").write_text("x" * 400, encoding="utf-8")
    monkeypatch.setattr(main, "KNOWLEDGE_DIR", str(tmp_path))

    calls = {"n": 0}

    def _forbidden(self):
        calls["n"] += 1
        raise AssertionError("with_index 不得触发索引构建")

    monkeypatch.setattr(kb_module.KnowledgeBase, "_ensure_index", _forbidden)

    response = client.get("/knowledge/documents", params={"with_index": "true"})

    assert response.status_code == 200
    document = response.json()["documents"][0]
    assert document["parse_status"] == "unknown"
    assert document["chunk_count"] is None
    assert calls["n"] == 0


def test_A13_with_index_reads_existing_index_only(monkeypatch, tmp_path):
    (tmp_path / "a.txt").write_text("x" * 400, encoding="utf-8")
    (tmp_path / "b.txt").write_text("y" * 400, encoding="utf-8")
    monkeypatch.setattr(main, "KNOWLEDGE_DIR", str(tmp_path))

    class _StubKB:
        skipped_reparse_count = 2

        def indexed_chunk_counts(self):
            return {"a.txt": 3}

    monkeypatch.setattr(main, "get_knowledge_base", lambda: _StubKB())

    body = client.get("/knowledge/documents", params={"with_index": "true"}).json()
    by_name = {d["filename"]: d for d in body["documents"]}

    assert by_name["a.txt"]["parse_status"] == "indexed"
    assert by_name["a.txt"]["chunk_count"] == 3
    assert by_name["b.txt"]["parse_status"] == "not_indexed"
    assert by_name["b.txt"]["chunk_count"] is None
    assert body["index_skipped_reparse"] == 2


@pytest.mark.parametrize("value", ["true", "TRUE", "1", "false", "0"])
def test_A14_with_index_accepts_enum_values(monkeypatch, tmp_path, value):
    monkeypatch.setattr(main, "KNOWLEDGE_DIR", str(tmp_path))

    response = client.get("/knowledge/documents", params={"with_index": value})

    assert response.status_code == 200
    assert response.json()["with_index"] is (value.lower() in ("true", "1"))


@pytest.mark.parametrize("value", ["maybe", "yes", "", "2", " true", "true "])
def test_A15_with_index_rejects_invalid_values(monkeypatch, tmp_path, value):
    monkeypatch.setattr(main, "KNOWLEDGE_DIR", str(tmp_path))

    response = client.get("/knowledge/documents", params={"with_index": value})

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert isinstance(detail, dict)
    assert detail["type"] == "invalid_parameter"
    assert detail["message"]


def test_A16_with_index_rejects_repeated_parameter(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "KNOWLEDGE_DIR", str(tmp_path))

    response = client.get(
        "/knowledge/documents?with_index=true&with_index=false"
    )

    assert response.status_code == 400
    assert response.json()["detail"]["type"] == "invalid_parameter"


# ============================================================
# DELETE /knowledge/documents/{document_id}
# ============================================================


def test_A17_delete_success(monkeypatch, tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("0123456789", encoding="utf-8")
    monkeypatch.setattr(main, "KNOWLEDGE_DIR", str(tmp_path))

    response = client.delete("/knowledge/documents/%s" % _doc_id("a.txt"))

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["filename"] == "a.txt"
    assert body["freed_bytes"] == 10
    assert not target.exists()


def test_A18_delete_invalidates_both_caches_in_order(monkeypatch, tmp_path):
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    monkeypatch.setattr(main, "KNOWLEDGE_DIR", str(tmp_path))

    order = []
    monkeypatch.setattr(main, "get_knowledge_base", _SpyCache("kb", order))
    monkeypatch.setattr(main, "get_agent", _SpyCache("agent", order))

    client.delete("/knowledge/documents/%s" % _doc_id("a.txt"))

    assert order == ["kb", "agent"]


def test_A19_delete_rejects_bad_id_without_scanning(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "KNOWLEDGE_DIR", str(tmp_path))

    calls = {"n": 0}

    def _spy(directory):
        calls["n"] += 1
        return {}

    monkeypatch.setattr(knowledge_store, "_scan_delete_index", _spy)

    response = client.delete("/knowledge/documents/not-a-valid-id")

    assert response.status_code == 400
    assert response.json()["detail"]["type"] == "invalid_document_id"
    assert calls["n"] == 0


def test_A20_delete_unknown_id_returns_404(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "KNOWLEDGE_DIR", str(tmp_path))

    response = client.delete("/knowledge/documents/%s" % ("0" * 32))

    assert response.status_code == 404
    assert response.json()["detail"]["type"] == "document_not_found"


def test_A21_delete_missing_directory_returns_404(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "KNOWLEDGE_DIR", str(tmp_path / "nope"))

    response = client.delete("/knowledge/documents/%s" % ("a" * 32))

    assert response.status_code == 404
    assert response.json()["detail"]["type"] == "document_not_found"


def test_A22_delete_uppercase_id_is_invalid(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "KNOWLEDGE_DIR", str(tmp_path))

    response = client.delete("/knowledge/documents/%s" % (_doc_id("a.txt").upper()))

    assert response.status_code == 400
    assert response.json()["detail"]["type"] == "invalid_document_id"


def test_A23_delete_case_variant_id_is_not_found(monkeypatch, tmp_path):
    (tmp_path / "Math.md").write_text("x", encoding="utf-8")
    monkeypatch.setattr(main, "KNOWLEDGE_DIR", str(tmp_path))

    response = client.delete("/knowledge/documents/%s" % _doc_id("MATH.MD"))

    assert response.status_code == 404
    assert response.json()["detail"]["type"] == "document_not_found"
    assert (tmp_path / "Math.md").exists()


def test_A24_delete_reparse_point_is_rejected(monkeypatch, tmp_path):
    outside = tmp_path / "outside.md"
    outside.write_text("库外", encoding="utf-8")
    entries = [
        _FakeEntry(
            tmp_path / "link.md",
            kind="reparse",
            stat_result=_reparse_stat(os.stat(outside)),
        )
    ]
    monkeypatch.setattr(main, "KNOWLEDGE_DIR", str(tmp_path))
    _patch_scandir(monkeypatch, entries)

    response = client.delete("/knowledge/documents/%s" % _doc_id("link.md"))

    assert response.status_code == 403
    assert response.json()["detail"]["type"] == "reparse_point_rejected"
    assert outside.exists()


def test_A25_delete_directory_entry_is_not_regular_file(monkeypatch, tmp_path):
    (tmp_path / "x.md").mkdir()
    monkeypatch.setattr(main, "KNOWLEDGE_DIR", str(tmp_path))

    response = client.delete("/knowledge/documents/%s" % _doc_id("x.md"))

    assert response.status_code == 403
    assert response.json()["detail"]["type"] == "not_regular_file"


def test_A26_delete_detects_concurrent_replacement(monkeypatch, tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("内容", encoding="utf-8")
    monkeypatch.setattr(main, "KNOWLEDGE_DIR", str(tmp_path))

    real_lstat = os.lstat
    counter = {"n": 0}

    def fake_lstat(path, *args, **kwargs):
        result = real_lstat(path, *args, **kwargs)
        if os.path.basename(str(path)) == "a.txt":
            counter["n"] += 1
            if counter["n"] >= 2:
                return types.SimpleNamespace(
                    st_mode=result.st_mode,
                    st_dev=result.st_dev,
                    st_ino=result.st_ino + 1,
                    st_size=result.st_size,
                    st_file_attributes=0,
                )
        return result

    monkeypatch.setattr(knowledge_store.os, "lstat", fake_lstat)

    response = client.delete("/knowledge/documents/%s" % _doc_id("a.txt"))

    assert response.status_code == 409
    assert response.json()["detail"]["type"] == "concurrent_modification"
    assert target.exists()


def test_A27_delete_failure_keeps_caches(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "KNOWLEDGE_DIR", str(tmp_path))

    order = []
    monkeypatch.setattr(main, "get_knowledge_base", _SpyCache("kb", order))
    monkeypatch.setattr(main, "get_agent", _SpyCache("agent", order))

    response = client.delete("/knowledge/documents/%s" % ("0" * 32))

    assert response.status_code == 404
    assert order == []


# ============================================================
# POST /knowledge/import
# ============================================================


def _upload(name, content, mime="text/markdown"):
    return client.post(
        "/knowledge/import",
        files={"file": (name, content, mime)},
    )


def test_A28_import_success_shape(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "KNOWLEDGE_DIR", str(tmp_path))

    response = _upload("new.md", "正文内容".encode("utf-8"))

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["filename"] == "new.md"
    assert body["bytes"] == len("正文内容".encode("utf-8"))
    assert body["document_id"] == _doc_id("new.md")
    assert body["storage_type"] == "persistent"
    assert (tmp_path / "new.md").read_text(encoding="utf-8") == "正文内容"


def test_A29_import_rejects_duplicate_name(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "KNOWLEDGE_DIR", str(tmp_path))

    assert _upload("dup.md", "内容".encode("utf-8")).status_code == 200
    second = _upload("dup.md", "内容".encode("utf-8"))

    assert second.status_code == 409
    assert second.json()["detail"]["type"] == "already_exists"


def test_A30_import_rejects_duplicate_content(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "KNOWLEDGE_DIR", str(tmp_path))

    assert _upload("first.md", "同样内容".encode("utf-8")).status_code == 200
    second = _upload("second.md", "同样内容".encode("utf-8"))

    assert second.status_code == 409
    detail = second.json()["detail"]
    assert detail["type"] == "duplicate_content"
    assert detail["existing_filename"] == "first.md"
    assert not (tmp_path / "second.md").exists()


def test_A31_import_does_not_normalize_newlines(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "KNOWLEDGE_DIR", str(tmp_path))

    assert _upload("lf.md", b"a\nb\n").status_code == 200
    second = _upload("crlf.md", b"a\r\nb\r\n")

    assert second.status_code == 200          # 原始字节不同 → 不判重
    assert (tmp_path / "crlf.md").read_bytes() == b"a\r\nb\r\n"


@pytest.mark.parametrize("name", ["CON.md", "bad*name.md"])
def test_A32_import_rejects_invalid_filename(monkeypatch, tmp_path, name):
    monkeypatch.setattr(main, "KNOWLEDGE_DIR", str(tmp_path))

    response = _upload(name, "内容".encode("utf-8"))

    assert response.status_code == 400
    assert response.json()["detail"]["type"] == "invalid_filename"


@pytest.mark.parametrize("name", ["x.md ", "a:b.md", "", "..", "nul.txt"])
def test_A32b_validate_filename_unit(name):
    """带尾空格 / 非法字符 / 保留名在**校验函数**层面必须被拒绝。

    走接口层会被 multipart 编码与 Path.name 归一化影响（例如
    ``Path("a:b.md").name == "b.md"``），因此单测直接对准校验函数。
    """
    assert knowledge_store.validate_knowledge_filename(name) == "invalid_filename"


def test_A33_import_rejects_untrusted_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "KNOWLEDGE_DIR", str(tmp_path))
    monkeypatch.setattr(
        knowledge_store,
        "inspect_knowledge_directory",
        lambda _path: "untrusted",
    )

    response = _upload("a.md", "内容".encode("utf-8"))

    assert response.status_code == 503
    assert response.json()["detail"]["type"] == "knowledge_dir_untrusted"
    assert not (tmp_path / "a.md").exists()


def test_A34_import_creates_missing_directory(monkeypatch, tmp_path):
    target_dir = tmp_path / "nested" / "kb"
    monkeypatch.setattr(main, "KNOWLEDGE_DIR", str(target_dir))

    response = _upload("a.md", "内容".encode("utf-8"))

    assert response.status_code == 200
    assert (target_dir / "a.md").exists()


def test_A35_import_rollback_failure_reports_residual(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "KNOWLEDGE_DIR", str(tmp_path))

    def _boom_write(fd, content):
        raise OSError("disk full")

    monkeypatch.setattr(knowledge_store, "_write_all", _boom_write)
    monkeypatch.setattr(knowledge_store, "_rollback_created", lambda path, st: True)

    order = []
    monkeypatch.setattr(main, "get_knowledge_base", _SpyCache("kb", order))
    monkeypatch.setattr(main, "get_agent", _SpyCache("agent", order))

    response = _upload("a.md", "内容".encode("utf-8"))

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert detail["type"] == "rollback_failed"
    assert detail["residual"] is True
    assert detail["filename"] == "a.md"
    assert order == []                        # 回滚失败同样不清缓存


def test_A36_import_dangling_symlink_name_conflicts(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "KNOWLEDGE_DIR", str(tmp_path))
    monkeypatch.setattr(
        knowledge_store.os.path, "lexists", lambda _path: True
    )

    response = _upload("dangling.md", "内容".encode("utf-8"))

    assert response.status_code == 409
    assert response.json()["detail"]["type"] == "already_exists"


# ============================================================
# 回归：既有契约不变
# ============================================================


def test_A37_search_unavailable_still_503(monkeypatch):
    monkeypatch.setattr(main, "get_knowledge_base", lambda: None)

    response = client.get("/knowledge/search", params={"query": "问题"})

    assert response.status_code == 503
    assert response.json()["detail"]["type"] == "knowledge_unavailable"


def test_A38_import_keeps_legacy_error_types(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "KNOWLEDGE_DIR", str(tmp_path))

    assert _upload("a.py", b"print(1)", "text/x-python").json()["detail"]["type"] == (
        "unsupported_format"
    )
    assert _upload("empty.md", b"").json()["detail"]["type"] == "empty_file"
    assert _upload("bad.md", b"\xff\xfe\x00garbage").json()["detail"]["type"] == (
        "invalid_encoding"
    )
