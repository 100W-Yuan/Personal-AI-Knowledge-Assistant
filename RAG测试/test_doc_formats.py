# -*- coding: utf-8 -*-
"""P0-1：统一文档格式清单（离线）。

覆盖本轮验收点：

- ``SUPPORTED_SUFFIXES`` 的内容与顺序（MD 必须排在 TXT 前）；
- ``GROUP_ORDER`` 与 ``SUPPORTED_SUFFIXES`` 同源；
- 后缀大小写不敏感、未知后缀不受支持、无后缀文件名不受支持；
- 三个模块的白名单**引用同一个对象**（而非各自复制一份元组）；
- 原有常量名 ``ALLOWED_KNOWLEDGE_SUFFIXES`` / ``ALLOWED_SUFFIXES`` 仍可用；
- ``_iter_knowledge_files`` 的分组顺序确实由统一清单驱动。

不加载任何真实 Embedding、不联网、只使用 ``tmp_path``，
不接触正式 ``rag_knowledge/``。
"""

import pytest

import doc_formats
import knowledge_store
import main
import rag.knowledge_base as kb_module


EXPECTED_SUFFIXES = (".md", ".txt", ".pdf", ".docx")


# ============================================================
# 清单内容与顺序
# ============================================================


def test_supported_suffixes_content_and_order():
    assert doc_formats.SUPPORTED_SUFFIXES == EXPECTED_SUFFIXES
    assert len(set(doc_formats.SUPPORTED_SUFFIXES)) == len(EXPECTED_SUFFIXES)


def test_md_precedes_txt():
    order = doc_formats.SUPPORTED_SUFFIXES
    assert order.index(".md") < order.index(".txt")


def test_group_order_is_identical_to_supported_suffixes():
    # 同一个对象，避免出现「第二份顺序」的可能。
    assert doc_formats.GROUP_ORDER is doc_formats.SUPPORTED_SUFFIXES
    assert tuple(doc_formats.GROUP_ORDER) == tuple(EXPECTED_SUFFIXES)


# ============================================================
# 后缀判定
# ============================================================


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("a.md", ".md"),
        ("b.txt", ".txt"),
        ("c.pdf", ".pdf"),
        ("d.docx", ".docx"),
        ("A.MD", ".md"),
        ("B.TxT", ".txt"),
        ("C.Pdf", ".pdf"),
        ("D.DOCX", ".docx"),
    ],
)
def test_suffix_of_is_case_insensitive(filename, expected):
    assert doc_formats.suffix_of(filename) == expected


@pytest.mark.parametrize(
    "suffix,expected",
    [
        (".md", True),
        (".MD", True),
        (".pdf", True),
        (".PdF", True),
        (".docx", True),
        (".txt", True),
    ],
)
def test_is_supported_suffix_accepts_known(suffix, expected):
    assert doc_formats.is_supported_suffix(suffix) is expected


@pytest.mark.parametrize(
    "suffix",
    [".py", ".doc", ".PDFX", ".mdx", ".", "", "md", None, 123],
)
def test_is_supported_suffix_rejects_unknown(suffix):
    assert doc_formats.is_supported_suffix(suffix) is False


@pytest.mark.parametrize("filename", ["README", "no_ext", ".gitignore", ""])
def test_filename_without_supported_suffix(filename):
    assert doc_formats.suffix_of(filename) == ""
    assert doc_formats.is_supported_suffix(doc_formats.suffix_of(filename)) is False


def test_trailing_dot_is_not_a_supported_suffix():
    # splitext("a.") == ("a", ".")，后缀是孤立的点，不是空串。
    # 这类名字在上传侧会被 validate_knowledge_filename 以「结尾是点」拒绝，
    # 此处确认格式清单本身也不会把它当作受支持后缀。
    assert doc_formats.suffix_of("a.") == "."
    assert doc_formats.is_supported_suffix(".") is False


def test_suffix_of_tolerates_non_string():
    assert doc_formats.suffix_of(None) == ""
    assert doc_formats.suffix_of(123) == ""


# ============================================================
# 三模块白名单一致性（同一对象，不是等值副本）
# ============================================================


def test_main_whitelist_references_shared_source():
    assert main.ALLOWED_KNOWLEDGE_SUFFIXES is doc_formats.SUPPORTED_SUFFIXES


def test_knowledge_store_whitelist_references_shared_source():
    assert knowledge_store.ALLOWED_SUFFIXES is doc_formats.SUPPORTED_SUFFIXES


def test_three_whitelists_are_equal_and_identical():
    trio = (
        main.ALLOWED_KNOWLEDGE_SUFFIXES,
        knowledge_store.ALLOWED_SUFFIXES,
        doc_formats.SUPPORTED_SUFFIXES,
    )
    assert trio[0] == trio[1] == trio[2] == EXPECTED_SUFFIXES
    assert trio[0] is trio[1] is trio[2]


def test_legacy_constant_names_still_available():
    assert hasattr(main, "ALLOWED_KNOWLEDGE_SUFFIXES")
    assert hasattr(knowledge_store, "ALLOWED_SUFFIXES")


# ============================================================
# 文件枚举确实由统一清单驱动
# ============================================================


def test_iter_knowledge_files_uses_shared_suffix_list(tmp_path):
    """四种受支持后缀都进入枚举，且分组顺序为 GROUP_ORDER；其它后缀被排除。"""
    for name in ("a.md", "b.txt", "c.pdf", "d.docx", "e.py", "f.doc"):
        (tmp_path / name).write_text("内容\n", encoding="utf-8")

    files, skipped_reparse, skipped_unstable = kb_module._iter_knowledge_files(
        str(tmp_path)
    )

    assert [path.name for path, _identity in files] == [
        "a.md",
        "b.txt",
        "c.pdf",
        "d.docx",
    ]
    assert skipped_reparse == 0
    assert skipped_unstable == 0
