# -*- coding: utf-8 -*-
"""P2-2C：旧接口 ``POST /knowledge/import`` 的文本导入契约（离线）。

锁定 F1–F6 之后旧接口的边界，防止「四格式统一上传」上线后旧接口被顺手放宽：

1. 只接受 ``.md`` / ``.txt`` —— 白名单是 ``main.LEGACY_IMPORT_SUFFIXES``，
   且必须是 ``doc_formats.SUPPORTED_SUFFIXES`` 的**真子集**；
2. 内容必须是有效 UTF-8（``content.decode("utf-8")`` 仍在，失败 → 400
   ``invalid_encoding``）；
3. PDF / DOCX 在**后缀闸门**就被拒绝（400 ``unsupported_format``），
   不再落到 UTF-8 解码阶段，也不再用「不是有效的 UTF-8 文本」误导调用方；
4. 同内容异名、缓存失效等既有契约保持不变。

隔离标准：``main.KNOWLEDGE_DIR`` 一律指向 ``tmp_path``，不触碰仓库内的
``rag_knowledge/``；不读数据库、不联网、不构造 ``KnowledgeBase`` / ``Agent``、
不调用任何 Loader —— **PDF / DOCX 样本只用于后缀闸门，从不参与解析**。

与既有测试的分工：``test_knowledge_endpoints.py`` 覆盖 MD 成功、``.py`` 拒绝、
空内容、同名冲突、非 UTF-8 MD、路径穿越、成功清缓存；本文件只补它未覆盖的
TXT 成功、TXT 非 UTF-8、PDF/DOCX 拒绝、同内容异名、失败不清缓存与白名单结构锁。
"""

import hashlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import knowledge_store
import main


client = TestClient(main.app)

_ENDPOINT = "/knowledge/import"

_MIME_BY_SUFFIX = {
    ".md": "text/markdown",
    ".txt": "text/plain",
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument"
    ".wordprocessingml.document",
    ".py": "text/x-python",
}


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


class _SpyCache:
    """可调用、也可被 ``cache_clear`` 的替身（缺任一即会静默跳过清理）。"""

    def __init__(self, label, sink):
        self.label = label
        self.sink = sink
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return None

    def cache_clear(self):
        self.sink.append(self.label)


def _import(name, content, mime=None):
    if mime is None:
        mime = _MIME_BY_SUFFIX.get(Path(name).suffix.lower(), "application/octet-stream")
    return client.post(_ENDPOINT, files={"file": (name, content, mime)})


def _files(root):
    root_path = Path(root)
    return sorted(
        str(path.relative_to(root_path))
        for path in root_path.rglob("*")
        if path.is_file()
    )


def _kb_dir(monkeypatch, tmp_path):
    target = tmp_path / "kb"
    monkeypatch.setattr(main, "KNOWLEDGE_DIR", str(target))
    return target


# ============================================================
# A0：白名单结构锁（F6）
# ============================================================


def test_A0_legacy_whitelist_is_text_subset_of_supported():
    """旧接口专用白名单必须是四格式统一清单的真子集。

    这条锁的是 F6 的结构关系而非某个字符串：若将来有人把 ``.pdf`` 加进
    ``LEGACY_IMPORT_SUFFIXES``，或用它替换掉统一清单，本用例会失败。
    """
    assert main.LEGACY_IMPORT_SUFFIXES == (".md", ".txt")
    assert main.ALLOWED_KNOWLEDGE_SUFFIXES == (".md", ".txt", ".pdf", ".docx")

    legacy = set(main.LEGACY_IMPORT_SUFFIXES)
    supported = set(main.ALLOWED_KNOWLEDGE_SUFFIXES)

    assert legacy < supported                       # 真子集，不是相等
    assert supported - legacy == {".pdf", ".docx"}  # 被排除的只有二进制格式


# ============================================================
# A1 / A3：TXT 的成功与编码边界
# ============================================================


def test_A1_txt_utf8_upload_succeeds(monkeypatch, tmp_path):
    """TXT + 合法 UTF-8 → 200，且落盘字节与上传字节完全一致。

    既有测试只覆盖了 ``.md`` 成功；``.txt`` 是旧接口的另一条合法分支，
    必须单独锁定（F6 收窄白名单后它仍须留在白名单内）。
    """
    target = _kb_dir(monkeypatch, tmp_path)
    payload = "纯文本内容\n第二行\n".encode("utf-8")

    response = _import("note.txt", payload)

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["filename"] == "note.txt"
    assert body["bytes"] == len(payload)
    assert body["document_id"] == hashlib.sha256(b"note.txt").hexdigest()[:32]
    assert body["storage_type"] == "persistent"
    assert body["document_id"] == knowledge_store.compute_document_id("note.txt")
    assert (target / "note.txt").read_bytes() == payload


def test_A3_non_utf8_txt_rejected_and_not_written(monkeypatch, tmp_path):
    """TXT + 非 UTF-8 → 400 ``invalid_encoding``，且不落盘。

    这是旧接口存在的**唯一理由**：它替调用方做 UTF-8 把关。若这道校验被
    删掉（例如为了迁就二进制格式），本用例会退化成 200。
    """
    target = _kb_dir(monkeypatch, tmp_path)
    payload = "中文内容".encode("gbk")

    with pytest.raises(UnicodeDecodeError):
        payload.decode("utf-8")

    response = _import("gbk.txt", payload)

    assert response.status_code == 400
    assert response.json()["detail"]["type"] == "invalid_encoding"
    assert _files(target) == []


# ============================================================
# A5 / A6：PDF / DOCX 由后缀闸门拒绝（F6 的核心新契约）
# ============================================================


def _assert_rejected_by_suffix_gate(name, payload, target):
    """断言 400 unsupported_format（而非 invalid_encoding），且未落盘。

    样本只是各自格式的「头部字节」，因为**闸门在读内容之前就返回**，
    内容根本不参与判定，也不调用任何 Loader。``type != invalid_encoding``
    是本断言的判别核心：F6 之前，PDF/DOCX 会穿过后缀闸门、在
    ``content.decode("utf-8")`` 处失败并报「不是有效的 UTF-8 文本」。
    """
    response = _import(name, payload)

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["type"] == "unsupported_format"
    assert detail["type"] != "invalid_encoding"
    # 文案必须指向正确入口，且由 LEGACY 白名单派生
    assert "/knowledge/documents" in detail["message"]
    for suffix in main.LEGACY_IMPORT_SUFFIXES:
        assert suffix in detail["message"]
    assert _files(target) == []


def test_A5_pdf_rejected_by_suffix_gate(monkeypatch, tmp_path):
    """PDF → 400 ``unsupported_format``（F6 的核心新契约），不落盘。"""
    target = _kb_dir(monkeypatch, tmp_path)

    _assert_rejected_by_suffix_gate(
        "doc.pdf",
        b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<< /Type /Catalog >>\nendobj\n",
        target,
    )


def test_A6_docx_rejected_by_suffix_gate(monkeypatch, tmp_path):
    """DOCX → 400 ``unsupported_format``（F6 的核心新契约），不落盘。"""
    target = _kb_dir(monkeypatch, tmp_path)

    _assert_rejected_by_suffix_gate("doc.docx", b"PK\x03\x04" + b"\x00" * 12, target)


# ============================================================
# A8：同内容异名
# ============================================================


def test_A8_duplicate_content_rejected_without_second_file(monkeypatch, tmp_path):
    """同内容、不同文件名 → 409 ``duplicate_content``，不产生第二份文件。

    既有测试只覆盖了「同名」冲突（``already_exists``），内容去重这条分支
    在旧接口上没有覆盖。（新接口的 E02 覆盖的是自己的路径，不可替代。）
    """
    target = _kb_dir(monkeypatch, tmp_path)
    payload = "same body".encode("utf-8")

    first = _import("first.md", payload)
    second = _import("second.md", payload)

    assert first.status_code == 200
    assert second.status_code == 409

    detail = second.json()["detail"]
    assert detail["type"] == "duplicate_content"
    assert detail["existing_filename"] == "first.md"
    assert _files(target) == ["first.md"]


# ============================================================
# A10：失败路径不得清理缓存
# ============================================================


def test_A10_rejection_keeps_caches(monkeypatch, tmp_path):
    """后缀被拒时不得触发 KB / Agent 缓存失效。

    覆盖两条拒绝路径：F6 新增的二进制格式（``.pdf``）与通用非法后缀
    （``.py``）。同时断言两个替身**从未被调用** —— 清理只允许走
    ``cache_clear()``，不得为了清缓存而构造 KnowledgeBase / Agent。
    """
    _kb_dir(monkeypatch, tmp_path)
    sink = []
    kb_spy = _SpyCache("kb", sink)
    agent_spy = _SpyCache("agent", sink)
    monkeypatch.setattr(main, "get_knowledge_base", kb_spy)
    monkeypatch.setattr(main, "get_agent", agent_spy)

    pdf = _import("doc.pdf", b"%PDF-1.4\n")
    py = _import("script.py", b"print(1)")

    assert pdf.status_code == 400
    assert pdf.json()["detail"]["type"] == "unsupported_format"
    assert py.status_code == 400
    assert py.json()["detail"]["type"] == "unsupported_format"

    assert sink == []
    assert kb_spy.calls == 0
    assert agent_spy.calls == 0
