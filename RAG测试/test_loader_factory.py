# -*- coding: utf-8 -*-
"""P0-2 批次 A：Loader 工厂与格式清单一致性测试。

覆盖：
- ``SUFFIX_LOADERS`` 与 ``doc_formats.SUPPORTED_SUFFIXES`` 双向相等
  （这比单侧包含更强：既防止漏注册，也防止注册白名单外的后缀）；
- 四个后缀各自映射到正确的 Loader 类；
- ``get_loader`` 的大小写、无后缀、尾点、Path 对象与未知后缀行为；
- ``TextLoader`` 对 .md/.txt 仍是生效实现（P0-1 兼容性回归）。
"""

import pathlib

import pytest

import doc_formats
from rag.loader import (
    DocxLoader,
    LoaderError,
    PdfLoader,
    SUFFIX_LOADERS,
    TextLoader,
    UnsupportedSuffixError,
    get_loader,
)


# ------------------------------------------------------------
# 注册表与白名单的一致性
# ------------------------------------------------------------
def test_registry_keys_equal_supported_suffixes():
    """双向相等：无漏注册、无越界注册。"""
    assert set(SUFFIX_LOADERS) == set(doc_formats.SUPPORTED_SUFFIXES)


def test_registry_has_no_suffix_outside_whitelist():
    extra = set(SUFFIX_LOADERS) - set(doc_formats.SUPPORTED_SUFFIXES)
    assert extra == set(), f"注册了白名单外的后缀：{extra}"


def test_whitelist_has_no_suffix_without_loader():
    missing = set(doc_formats.SUPPORTED_SUFFIXES) - set(SUFFIX_LOADERS)
    assert missing == set(), f"白名单后缀缺少 Loader：{missing}"


def test_registry_covers_group_order():
    """枚举分组顺序里的每个后缀都必须有 Loader，否则索引会静默跳过。"""
    for suffix in doc_formats.GROUP_ORDER:
        assert suffix in SUFFIX_LOADERS


# ------------------------------------------------------------
# 后缀到 Loader 的映射
# ------------------------------------------------------------
@pytest.mark.parametrize("suffix", [".md", ".txt"])
def test_markdown_and_text_use_text_loader(suffix):
    assert SUFFIX_LOADERS[suffix] is TextLoader


def test_pdf_uses_pdf_loader():
    assert SUFFIX_LOADERS[".pdf"] is PdfLoader


def test_docx_uses_docx_loader():
    assert SUFFIX_LOADERS[".docx"] is DocxLoader


# ------------------------------------------------------------
# get_loader 行为
# ------------------------------------------------------------
@pytest.mark.parametrize(
    "name,expected",
    [
        ("a.md", TextLoader),
        ("a.txt", TextLoader),
        ("a.pdf", PdfLoader),
        ("a.docx", DocxLoader),
    ],
)
def test_get_loader_returns_registry_class(name, expected):
    loader = get_loader(name)
    assert type(loader) is expected


def test_get_loader_is_case_insensitive():
    assert type(get_loader("A.PDF")) is PdfLoader
    assert type(get_loader("A.DoCX")) is DocxLoader
    assert type(get_loader("A.MD")) is TextLoader


def test_get_loader_accepts_pathlib_path():
    loader = get_loader(pathlib.Path("dir") / "doc.pdf")
    assert type(loader) is PdfLoader


def test_get_loader_does_not_require_file_to_exist():
    """构造 Loader 不得触碰文件系统；存在性由调用方与 Loader.load() 负责。"""
    loader = get_loader("this-file-does-not-exist-anywhere.pdf")
    assert type(loader) is PdfLoader


@pytest.mark.parametrize("name", ["a.xyz", "a.exe", "a", "a.", "a.tar.gz"])
def test_get_loader_rejects_unsupported(name):
    with pytest.raises(UnsupportedSuffixError) as excinfo:
        get_loader(name)
    assert isinstance(excinfo.value, LoaderError)


def test_unsupported_error_carries_suffix_and_path():
    with pytest.raises(UnsupportedSuffixError) as excinfo:
        get_loader("report.xlsx")
    error = excinfo.value
    assert error.suffix == ".xlsx"
    assert "report.xlsx" in str(error)


def test_unsupported_error_is_loader_error():
    """调用方可以只捕获 LoaderError 统一处理，不必枚举具体子类。"""
    with pytest.raises(LoaderError):
        get_loader("a.unknown")


# ------------------------------------------------------------
# TextLoader 兼容性（P0-1 回归）
# ------------------------------------------------------------
def test_text_loader_constructor_signature_unchanged(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("内容", encoding="utf-8")
    loader = TextLoader(str(path))
    assert loader.file_path == str(path)


def test_text_loader_still_reads_plain_text(tmp_path):
    path = tmp_path / "a.md"
    path.write_text("正文", encoding="utf-8")
    documents = TextLoader(str(path)).load()
    assert len(documents) == 1
    assert documents[0].page_content == "正文"
    assert documents[0].metadata["source"] == str(path)


def test_text_loader_metadata_has_no_format_specific_keys(tmp_path):
    """回归：不得因新增 Loader 而给 TextLoader 偷偷塞 page/section 等字段。"""
    path = tmp_path / "a.txt"
    path.write_text("正文", encoding="utf-8")
    metadata = TextLoader(str(path)).load()[0].metadata
    assert set(metadata) == {"source"}
