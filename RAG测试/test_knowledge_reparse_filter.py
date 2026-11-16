# -*- coding: utf-8 -*-
"""阶段一：索引层安全过滤与目录异常状态机（离线）。

覆盖《阶段一实施前设计审查 V1.6》第 1/2 节：

- L1 枚举：仅顶层 .md/.txt 普通文件；跳过符号链接 / junction / 其它重解析点；
- L2 打开前复核：普通文件类型、无重解析点、``(st_dev, st_ino)`` 身份未变；
- 计数器分离：skipped_reparse_count / skipped_unstable_count / skipped_count；
- 目录异常映射：ENOENT、ENOTDIR、普通文件、EACCES、其它 OSError；
- dir_untrusted 状态机：不缓存 _init_error、可重试、恢复后能重建。

**证据等级声明**：本机未创建真实 junction / 符号链接（受执行边界限制），
重解析点相关用例全部通过 stub 的 ``os.scandir`` / ``os.lstat`` 注入，
属**模拟测试**，不代表已在真实重解析点上验证。
不加载任何真实 Embedding、不联网。
"""

import os
import shutil
import stat
import tempfile
import types
import unittest.mock as mock

import rag.knowledge_base as kb_module
from rag.knowledge_base import KnowledgeBase


# ============================================================
# 桩：Embedding 与 DirEntry
# ============================================================


class _StubEmbedding:
    """固定二维向量：只为让索引走通，不表达任何语义。"""

    def encode(self, text):
        return [1.0, float(len(text) % 7)]


def _make_kb(doc_dir, factory=None, stats=None):
    def factory_wrapper():
        if stats is not None:
            stats["calls"] = stats.get("calls", 0) + 1
        if factory is not None:
            return factory()
        return _StubEmbedding()

    return KnowledgeBase(str(doc_dir), embedding_factory=factory_wrapper)


class _FakeEntry:
    """最小 DirEntry 替身：name / path / is_symlink / is_file / stat。"""

    def __init__(self, path, kind="regular", stat_result=None):
        self.path = str(path)
        self.name = os.path.basename(self.path)
        self._kind = kind
        self._stat = stat_result

    def is_symlink(self):
        return self._kind in ("reparse", "symlink")

    def is_file(self, follow_symlinks=True):
        if self._kind == "regular":
            return True if not follow_symlinks else os.path.isfile(self.path)
        return False

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


def _reparse_stat(base_stat):
    """把真实 stat 改造成「带重解析点属性」的替身。"""
    return types.SimpleNamespace(
        st_mode=base_stat.st_mode,
        st_dev=base_stat.st_dev,
        st_ino=base_stat.st_ino,
        st_size=base_stat.st_size,
        st_mtime=base_stat.st_mtime,
        st_ctime=base_stat.st_ctime,
        st_file_attributes=stat.FILE_ATTRIBUTE_REPARSE_POINT,
    )


def _symlink_stat(base_stat):
    """模拟 lstat 对符号链接返回的自身属性：S_IFLNK + 重解析点属性位。"""
    return types.SimpleNamespace(
        st_mode=stat.S_IFLNK | 0o777,
        st_dev=base_stat.st_dev,
        st_ino=base_stat.st_ino,
        st_size=0,
        st_mtime=base_stat.st_mtime,
        st_ctime=base_stat.st_ctime,
        st_file_attributes=stat.FILE_ATTRIBUTE_REPARSE_POINT,
    )


# ============================================================
# S1 / S2：普通 TXT、MD 正常加载
# ============================================================


def test_S1_plain_txt_is_indexed(tmp_path):
    (tmp_path / "a.txt").write_text("第一段内容。\n" * 20, encoding="utf-8")

    kb = _make_kb(tmp_path)
    result = kb.search("内容")

    assert result["ok"] is True
    assert kb.is_indexed is True
    assert kb.file_count == 1
    assert kb.chunk_count >= 1
    assert kb.skipped_reparse_count == 0
    assert kb.skipped_unstable_count == 0


def test_S2_plain_md_is_indexed(tmp_path):
    (tmp_path / "b.md").write_text("# 标题\n\n正文段落。\n" * 15, encoding="utf-8")

    kb = _make_kb(tmp_path)
    result = kb.search("标题")

    assert result["ok"] is True
    assert kb.file_count == 1
    assert kb.chunk_count >= 1


# ============================================================
# T5：真实 Windows 文件系统上的身份来源回归守卫（R2 修复专用）
#
# 本用例不使用 _FakeEntry、不 patch os.scandir / os.lstat，
# 全部走真实文件系统；直接对照「L1 记录下来的身份」与「同一文件的
# os.lstat」，因此能在不构建索引、不加载 Embedding 的情况下
# 暴露「L1 与 L2 身份来源不一致」这类缺陷。
# ============================================================


def test_T5_l1_identity_source_matches_lstat():
    """L1 记录的身份必须等于同一文件 os.lstat 的结果。

    回归背景：L1 曾用 ``DirEntry.stat(follow_symlinks=False)`` 取
    ``(st_dev, st_ino)``，而 Windows 上该调用返回 ``(0, 0)``
    （目录扫描缓存 WIN32_FIND_DATA 不含文件索引与卷序列号），
    L2 用 ``os.lstat`` 复核时必然不等，导致所有正常文件都被判为
    「身份已变化」而跳过，索引恒为空。

    这里只断言「两者一致」，不写死具体数值，因此跨平台成立；
    若 L1 退回 ``DirEntry.stat``，在 Windows 上会立即失败。
    """

    work_dir = tempfile.mkdtemp(prefix="kb_identity_guard_")
    try:
        target = os.path.join(work_dir, "real.txt")
        with open(target, "w", encoding="utf-8") as handle:
            handle.write("真实文件内容\n" * 10)

        files, skipped_reparse, skipped_unstable = (
            kb_module._iter_knowledge_files(work_dir)
        )

        assert skipped_reparse == 0
        assert skipped_unstable == 0
        assert len(files) == 1

        path, identity = files[0]
        assert path.name == "real.txt"

        expected = os.lstat(target)
        assert identity == (expected.st_dev, expected.st_ino)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# ============================================================
# S3 / S4：符号链接与重解析点在 L1 被跳过
# ============================================================


def test_S3_symlink_entry_is_skipped(tmp_path):
    real = tmp_path / "a.txt"
    real.write_text("正常内容\n" * 20, encoding="utf-8")
    link = tmp_path / "link.txt"
    link_stat = _symlink_stat(os.lstat(real))
    entries = [
        _FakeEntry(real, kind="regular"),
        _FakeEntry(link, kind="symlink", stat_result=link_stat),
    ]

    # link.txt 在磁盘上并不存在（本机不能创建真实符号链接）。若只伪造
    # _FakeEntry，L1 改用 os.lstat 后会对 link.txt 抛 FileNotFoundError，
    # 结果是计入 skipped_unstable_count 而不是 skipped_reparse_count，
    # 于是永远覆盖不到 S_ISLNK / 重解析点分支。
    # 这里按路径注入「符号链接自身属性」，a.txt 仍走真实 lstat。
    # 证据等级：**模拟**，不代表已在真实 Windows 符号链接上验证。
    fake_lstat = _lstat_with_overrides({link: link_stat})

    with mock.patch.object(kb_module.os, "scandir", lambda _p: _FakeScandir(entries)), \
            mock.patch.object(kb_module.os, "lstat", fake_lstat):
        kb = _make_kb(tmp_path)
        result = kb.search("正常")

    assert result["ok"] is True
    assert kb.file_count == 1
    assert kb.skipped_reparse_count == 1         # 符号链接按重解析点归类
    assert kb.skipped_unstable_count == 0


def test_S4_reparse_entry_is_skipped(tmp_path):
    real = tmp_path / "a.txt"
    real.write_text("正常内容\n" * 20, encoding="utf-8")
    outside = tmp_path / "outside.md"
    outside.write_text("库外内容\n" * 5, encoding="utf-8")

    outside_stat = _reparse_stat(os.lstat(outside))
    entries = [
        _FakeEntry(real, kind="regular"),
        _FakeEntry(outside, kind="reparse", stat_result=outside_stat),
    ]

    # outside.md 在磁盘上是**真实普通文件**，真实 lstat 不含重解析点属性位；
    # 只伪造 _FakeEntry 会被 L1 的 os.lstat 绕过，文件被正常载入（file_count=2）。
    # 这里只对 outside.md 这一个路径注入重解析点属性，a.txt 走真实 lstat。
    fake_lstat = _lstat_with_overrides({outside: outside_stat})

    with mock.patch.object(kb_module.os, "scandir", lambda _p: _FakeScandir(entries)), \
            mock.patch.object(kb_module.os, "lstat", fake_lstat):
        kb = _make_kb(tmp_path)
        result = kb.search("正常")

    assert result["ok"] is True
    assert kb.file_count == 1
    assert kb.skipped_reparse_count == 1
    assert kb.skipped_unstable_count == 0


# ============================================================
# S5 / S6 / S7：L2 打开前复核（身份 / 类型 / 重解析点）
#
# L1 与 L2 同源（都是 os.lstat），因此这三个用例统一用
# 「第 1 次 lstat 正常、第 2 次 lstat 异常」的注入方式，
# 确保命中的是 _ensure_index 循环里的 L2 复核分支，
# 而不是在 L1 枚举阶段就被拦下。
# ============================================================


def _patch_lstat_second_call(monkeypatch, target_name, mutate):
    """只对 target_name 的**第 2 次** ``os.lstat`` 施加 ``mutate``。

    为什么必须按调用次序注入
    ------------------------
    L1 记录身份与 L2 复核身份现在同源（都是 ``os.lstat``）。若像以前那样
    「每次调用都返回改造后的值」，L1 与 L2 会看到同一个伪造值、比较相等，
    用例就再也覆盖不到 L2 的对应分支。

    因此这里按文件分桶计数：第 1 次（L1 枚举并记录身份）返回真实值，
    第 2 次（L2 打开前复核）才返回 ``mutate(真实值)``，从而精确命中
    L2 的三条 ``continue`` 分支之一。

    ``kb_module.os`` 就是进程级 ``os`` 模块，这里替换的是全局属性；
    ``monkeypatch`` 会在用例结束时自动还原。
    """
    real_lstat = os.lstat
    seen = {}

    def fake_lstat(path, *args, **kwargs):
        result = real_lstat(path, *args, **kwargs)
        if os.path.basename(str(path)) == target_name:
            seen[target_name] = seen.get(target_name, 0) + 1
            if seen[target_name] >= 2:
                return mutate(result)
        return result

    monkeypatch.setattr(kb_module.os, "lstat", fake_lstat)


def _lstat_with_overrides(overrides):
    """按路径覆盖 ``os.lstat`` 的返回值；未命中的路径一律走真实 ``os.lstat``。

    为什么需要它
    ------------
    在 Windows 上无法可靠地创建真实符号链接 / junction（且本套用例的执行
    边界不允许创建），而 L1 现在改用 ``os.lstat`` 判定文件类型与重解析点。
    若只在 ``os.scandir`` 返回的 ``_FakeEntry`` 上伪造 ``stat``，L1 会改用
    真实 ``os.lstat`` 重新取一次，伪造值被绕过、用例失效。

    因此改为在 ``os.lstat`` 这一层按**路径**注入：
    - 命中 ``overrides`` 的路径返回给定 stat；
    - 其余路径（包括用例正依赖其真实属性的普通文件）调用真实 lstat。

    键统一经 ``os.path.normcase(os.path.abspath(...))`` 归一化，
    避免大小写 / 相对路径差异导致漏命中。

    注意：``kb_module.os`` 与测试模块里的 ``os`` 是**同一个模块对象**，
    所以真实实现必须在替换**之前**取到本地变量，否则会无限递归。
    """

    real_lstat = os.lstat
    normalized = {
        os.path.normcase(os.path.abspath(str(key))): value
        for key, value in overrides.items()
    }

    def fake_lstat(path, *args, **kwargs):
        key = os.path.normcase(os.path.abspath(str(path)))
        if key in normalized:
            return normalized[key]
        return real_lstat(path, *args, **kwargs)

    return fake_lstat


def test_S5_l2_identity_change_is_skipped(tmp_path, monkeypatch):
    (tmp_path / "a.txt").write_text("甲\n" * 20, encoding="utf-8")
    (tmp_path / "b.txt").write_text("乙\n" * 20, encoding="utf-8")

    def mutate(st):
        return types.SimpleNamespace(
            st_mode=st.st_mode,
            st_dev=st.st_dev,
            st_ino=st.st_ino + 1,
            st_size=st.st_size,
            st_file_attributes=0,
        )

    # L1 看到真实身份 → 进入 files；L2 看到 ino+1 → 命中身份比较分支。
    _patch_lstat_second_call(monkeypatch, "b.txt", mutate)

    kb = _make_kb(tmp_path)
    result = kb.search("甲")

    assert result["ok"] is True
    assert kb.file_count == 1
    assert kb.skipped_unstable_count == 1
    assert kb.skipped_reparse_count == 0


def test_S6_l2_type_change_is_skipped(tmp_path, monkeypatch):
    (tmp_path / "a.txt").write_text("甲\n" * 20, encoding="utf-8")
    (tmp_path / "b.txt").write_text("乙\n" * 20, encoding="utf-8")

    def mutate(st):
        return types.SimpleNamespace(
            st_mode=stat.S_IFDIR,
            st_dev=st.st_dev,
            st_ino=st.st_ino,
            st_size=st.st_size,
            st_file_attributes=0,
        )

    # L1 看到普通文件 → 进入 files；L2 看到 S_IFDIR → 命中类型检查分支。
    _patch_lstat_second_call(monkeypatch, "b.txt", mutate)

    kb = _make_kb(tmp_path)
    kb.search("甲")

    assert kb.file_count == 1
    assert kb.skipped_unstable_count == 1


def test_S7_l2_reparse_is_skipped(tmp_path, monkeypatch):
    (tmp_path / "a.txt").write_text("甲\n" * 20, encoding="utf-8")
    (tmp_path / "b.txt").write_text("乙\n" * 20, encoding="utf-8")

    # L1 看到普通文件 → 进入 files；L2 看到重解析点属性 → 命中 reparse 分支。
    _patch_lstat_second_call(monkeypatch, "b.txt", _reparse_stat)

    kb = _make_kb(tmp_path)
    kb.search("甲")

    assert kb.file_count == 1
    assert kb.skipped_reparse_count == 1
    assert kb.skipped_unstable_count == 0


# ============================================================
# S8：三计数器互不串味
# ============================================================


def test_S8_counters_are_independent(tmp_path):
    (tmp_path / "ok.txt").write_text("正常\n" * 20, encoding="utf-8")
    (tmp_path / "bad.txt").write_bytes(b"\xff\xfe\x00garbage")
    outside = tmp_path / "outside.md"
    outside.write_text("库外\n" * 5, encoding="utf-8")
    outside_stat = _reparse_stat(os.lstat(outside))

    entries = [
        _FakeEntry(tmp_path / "ok.txt", kind="regular"),
        _FakeEntry(tmp_path / "bad.txt", kind="regular"),
        _FakeEntry(outside, kind="reparse", stat_result=outside_stat),
    ]

    # 只对 outside.md 注入重解析点属性；ok.txt / bad.txt 走真实 lstat，
    # 分别负责「正常入索引」与「无效 UTF-8 走 skipped_count」。
    fake_lstat = _lstat_with_overrides({outside: outside_stat})

    with mock.patch.object(kb_module.os, "scandir", lambda _p: _FakeScandir(entries)), \
            mock.patch.object(kb_module.os, "lstat", fake_lstat):
        kb = _make_kb(tmp_path)
        result = kb.search("正常")

    assert result["ok"] is True
    assert kb.file_count == 1
    assert kb.skipped_reparse_count == 1
    assert kb.skipped_count == 1          # 无效 UTF-8 走既有加载失败路径
    assert kb.skipped_unstable_count == 0


# ============================================================
# S9：被跳过的重解析点内容不会进入 VectorStore
# ============================================================


def test_S9_outside_content_never_reaches_vector_store(tmp_path):
    sentinel = "SENTINEL-OUTSIDE-CONTENT-9F3A"
    (tmp_path / "ok.txt").write_text("库内内容\n" * 20, encoding="utf-8")
    outside = tmp_path / "outside.md"
    outside.write_text(sentinel, encoding="utf-8")
    outside_stat = _reparse_stat(os.lstat(outside))

    entries = [
        _FakeEntry(tmp_path / "ok.txt", kind="regular"),
        _FakeEntry(outside, kind="reparse", stat_result=outside_stat),
    ]

    # 同样只对 outside.md 注入重解析点：若 L1 未拦下，sentinel 会随该文件
    # 一起被切块写入 VectorStore，下面的断言就会失败。
    fake_lstat = _lstat_with_overrides({outside: outside_stat})

    with mock.patch.object(kb_module.os, "scandir", lambda _p: _FakeScandir(entries)), \
            mock.patch.object(kb_module.os, "lstat", fake_lstat):
        kb = _make_kb(tmp_path)
        kb.search("库内")

    # 先确认拦截确实发生在 L1 重解析点分支，而不是靠别的原因「碰巧」没进去。
    assert kb.skipped_reparse_count == 1

    items = kb._retriever.vector_store.items
    assert items, "库内正常文件应已入索引"
    assert all(sentinel not in item["document"].page_content for item in items)


# ============================================================
# S10：普通检索回归
# ============================================================


def test_S10_normal_retrieval_regression(tmp_path):
    (tmp_path / "数学题.txt").write_text("三角形的内角和是 180 度。\n" * 10, encoding="utf-8")

    kb = _make_kb(tmp_path)
    result = kb.search("三角形")

    assert result["ok"] is True
    assert result["count"] >= 1
    assert result["results"][0]["source"] == "数学题.txt"
    counts = kb.indexed_chunk_counts()
    assert counts is not None and counts["数学题.txt"] >= 1


# ============================================================
# S11：名为 x.md 的目录计入 unstable，而不是 skipped_count
# ============================================================


def test_S11_directory_named_like_document_is_unstable(tmp_path):
    (tmp_path / "a.txt").write_text("正常\n" * 20, encoding="utf-8")
    (tmp_path / "x.md").mkdir()

    kb = _make_kb(tmp_path)
    kb.search("正常")

    assert kb.file_count == 1
    assert kb.skipped_unstable_count == 1
    assert kb.skipped_count == 0


# ============================================================
# D1 - D5：目录异常映射
# ============================================================


def test_D1_missing_directory_is_dir_missing(tmp_path):
    kb = _make_kb(tmp_path / "nope")
    result = kb.search("任意")

    assert result["ok"] is False and result["stage"] == "init"
    assert "FileNotFoundError" in result["error"]
    assert "知识目录缺失或为空" in result["error"]
    assert kb.init_error_kind == "dir_missing"
    assert kb._init_error is not None          # 与既有缓存语义一致


def test_D2_plain_file_path_is_dir_missing(tmp_path):
    target = tmp_path / "kb"
    target.write_text("not a directory", encoding="utf-8")

    kb = _make_kb(target)
    result = kb.search("任意")

    assert result["ok"] is False
    assert "知识目录缺失或为空" in result["error"]
    assert kb.init_error_kind == "dir_missing"


def test_D3_ancestor_is_file_is_dir_missing(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("file", encoding="utf-8")

    kb = _make_kb(blocker / "kb")
    result = kb.search("任意")

    assert result["ok"] is False
    assert "知识目录缺失或为空" in result["error"]
    assert kb.init_error_kind == "dir_missing"


def test_D4_permission_error_maps_to_init_error(tmp_path, monkeypatch):
    calls = {"n": 0}
    doc_dir = str(tmp_path)
    real_lstat = os.lstat

    def fake_lstat(path, *args, **kwargs):
        if str(path) == doc_dir:
            calls["n"] += 1
            raise PermissionError("denied")
        return real_lstat(path, *args, **kwargs)

    monkeypatch.setattr(kb_module.os, "lstat", fake_lstat)

    kb = _make_kb(tmp_path)
    first = kb.search("任意")
    second = kb.search("任意")

    assert first["ok"] is False and first["stage"] == "init"
    assert "无法访问" in first["error"]
    assert kb.init_error_kind == "init_error"
    assert calls["n"] == 1                     # 已缓存，不再 lstat
    assert second["ok"] is False


def test_D5_other_oserror_maps_to_init_error(tmp_path, monkeypatch):
    doc_dir = str(tmp_path)
    real_lstat = os.lstat

    def fake_lstat(path, *args, **kwargs):
        if str(path) == doc_dir:
            raise OSError("boom")
        return real_lstat(path, *args, **kwargs)

    monkeypatch.setattr(kb_module.os, "lstat", fake_lstat)

    kb = _make_kb(tmp_path)
    result = kb.search("任意")

    assert result["ok"] is False
    assert "无法访问" in result["error"]
    assert kb.init_error_kind == "init_error"
    assert kb._init_error is not None


# ============================================================
# D6 - D10：dir_untrusted 状态机
# ============================================================


def _patch_dir_as_reparse(monkeypatch, doc_dir, counter):
    real_lstat = os.lstat

    def fake_lstat(path, *args, **kwargs):
        if str(path) == str(doc_dir):
            counter["n"] += 1
            base = real_lstat(path, *args, **kwargs)
            return _reparse_stat(base)
        return real_lstat(path, *args, **kwargs)

    monkeypatch.setattr(kb_module.os, "lstat", fake_lstat)


def test_D6_reparse_directory_is_untrusted(tmp_path, monkeypatch):
    counter = {"n": 0}
    _patch_dir_as_reparse(monkeypatch, tmp_path, counter)

    kb = _make_kb(tmp_path)
    result = kb.search("任意")

    assert result["ok"] is False and result["stage"] == "init"
    assert "重解析点" in result["error"]
    assert kb.init_error_kind == "dir_untrusted"
    assert kb._init_error is None              # 关键：不缓存永久失败
    assert counter["n"] == 1


def test_D7_two_searches_each_lstat_the_directory(tmp_path, monkeypatch):
    counter = {"n": 0}
    _patch_dir_as_reparse(monkeypatch, tmp_path, counter)

    kb = _make_kb(tmp_path)
    kb.search("任意")
    kb.search("任意")

    assert counter["n"] == 2                   # 未走快速失败路径
    assert kb._init_error is None


def test_D8_reparse_then_deleted_becomes_dir_missing(tmp_path, monkeypatch):
    state = {"mode": "reparse"}
    doc_dir = str(tmp_path)
    real_lstat = os.lstat

    def fake_lstat(path, *args, **kwargs):
        if str(path) == doc_dir:
            if state["mode"] == "reparse":
                return _reparse_stat(real_lstat(path, *args, **kwargs))
            raise FileNotFoundError(2, "not found", doc_dir)
        return real_lstat(path, *args, **kwargs)

    monkeypatch.setattr(kb_module.os, "lstat", fake_lstat)

    kb = _make_kb(tmp_path)
    first = kb.search("任意")
    state["mode"] = "missing"
    second = kb.search("任意")

    assert "重解析点" in first["error"]
    assert kb.init_error_kind == "dir_missing"
    assert kb._init_error is not None          # 转为可缓存类别
    assert "知识目录缺失或为空" in second["error"]


def test_D9_cached_failure_survives_directory_restore(tmp_path, monkeypatch):
    """reparse → 删除：落入失败缓存后，重建目录也不会自愈（需 cache_clear）。"""
    state = {"mode": "reparse"}
    doc_dir = str(tmp_path)
    real_lstat = os.lstat

    def fake_lstat(path, *args, **kwargs):
        # 三态：reparse → 注入重解析点；missing → 抛 FileNotFoundError；
        # 其它（含 "ok" 与未识别状态）→ 退回真实 lstat。
        # 早先这里只有两条出口，"ok" 会一并落到 raise 分支，
        # 于是 `state["mode"] = "ok"` 之后目录仍被当成缺失，
        # 末尾那条 `fresh.search(...)["ok"] is True` 在该测试代码下不可能成立。
        if str(path) == doc_dir:
            mode = state["mode"]
            if mode == "reparse":
                return _reparse_stat(real_lstat(path, *args, **kwargs))
            if mode == "missing":
                raise FileNotFoundError(2, "not found", doc_dir)
        return real_lstat(path, *args, **kwargs)

    monkeypatch.setattr(kb_module.os, "lstat", fake_lstat)

    (tmp_path / "a.txt").write_text("恢复后的内容\n" * 20, encoding="utf-8")

    kb = _make_kb(tmp_path)
    kb.search("任意")                          # reparse
    state["mode"] = "missing"
    kb.search("任意")                          # → dir_missing 且缓存

    state["mode"] = "ok"                       # 目录「恢复」
    still_failed = kb.search("任意")
    assert still_failed["ok"] is False          # 快速失败复用缓存

    fresh = _make_kb(tmp_path)                  # 等价于 cache_clear 后新建实例
    assert fresh.search("恢复")["ok"] is True
    assert fresh.is_indexed is True


def test_D10_reparse_then_restored_rebuilds(tmp_path, monkeypatch):
    state = {"mode": "reparse"}
    doc_dir = str(tmp_path)
    real_lstat = os.lstat

    def fake_lstat(path, *args, **kwargs):
        if str(path) == doc_dir and state["mode"] == "reparse":
            return _reparse_stat(real_lstat(path, *args, **kwargs))
        return real_lstat(path, *args, **kwargs)

    monkeypatch.setattr(kb_module.os, "lstat", fake_lstat)

    (tmp_path / "a.txt").write_text("恢复后的内容\n" * 20, encoding="utf-8")

    kb = _make_kb(tmp_path)
    first = kb.search("任意")
    assert first["ok"] is False

    state["mode"] = "ok"
    second = kb.search("恢复")

    assert second["ok"] is True
    assert kb.is_indexed is True
    assert kb.init_error_kind is None


def test_D11_plain_init_error_is_still_cached(tmp_path):
    stats = {"calls": 0}

    def boom():
        raise RuntimeError("stub 初始化失败（测试注入）")

    kb = _make_kb(tmp_path, factory=boom, stats=stats)
    (tmp_path / "a.txt").write_text("内容\n" * 20, encoding="utf-8")

    first = kb.search("内容")
    second = kb.search("内容")

    assert first["ok"] is False and first["stage"] == "init"
    assert "stub 初始化失败" in first["error"]
    assert second["ok"] is False
    assert stats["calls"] == 1                  # 既有语义：不反复重试


# ============================================================
# 附加：indexed_chunk_counts 语义
# ============================================================


def test_indexed_chunk_counts_none_before_build(tmp_path):
    (tmp_path / "a.txt").write_text("内容\n" * 10, encoding="utf-8")

    kb = _make_kb(tmp_path)

    assert kb.is_indexed is False
    assert kb.indexed_chunk_counts() is None    # None ≠ {}
