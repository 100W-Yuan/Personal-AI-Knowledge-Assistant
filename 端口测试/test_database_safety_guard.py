# -*- coding: utf-8 -*-
"""阶段 1-R2：测试数据库保护机制自证（T01–T10）。

本文件只验证守卫自身的行为，**不连接、不创建、不读取正式数据库**。
所有"应当被拒绝"的用例都断言两件事：

1. 抛出 ``ProtectedDatabaseAccessError``（而不是 SQLite 的 OperationalError）；
2. 守卫没有把手上的连接委托给真正的 SQLite 实现（用 spy 证明拦截发生在
   打开文件之前）。

守卫由项目根目录 ``conftest.py`` 在 pytest 加载阶段注册到
``sys.modules["_wb_db_guard"]``。若该模块不存在，说明根级保护没有生效，
本文件在导入阶段即报错，绝不静默跳过。
"""

import os
import re
import sqlite3
import sys
from pathlib import Path

import pytest


def _load_guard():
    module = sys.modules.get("_wb_db_guard")

    if module is None:
        raise RuntimeError(
            "根级 conftest.py 未加载：数据库守卫不可用。"
            "请从项目根目录运行 pytest，且不要使用 --noconftest / --confcutdir 绕过。"
        )

    return module


guard = _load_guard()

PROJECT_ROOT = Path(guard.PROJECT_ROOT)
MEMORY_DB = PROJECT_ROOT / "memory.db"
BUSINESS_DB = PROJECT_ROOT / "data" / "business.db"


# ========================================================
# 公共工具
# ========================================================


@pytest.fixture
def connect_spy(monkeypatch):
    """把真正的 SQLite connect 换成探测器。

    守卫若把连接委托出去，探测器会被调用并让用例立刻失败——
    这就是"拦截发生在 SQLite 实际打开文件之前"的直接证据。
    """

    calls = []

    def _spy(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError(
            "守卫把连接委托给了真正的 SQLite 实现，拦截失效"
        )

    monkeypatch.setattr(guard, "_ORIGINAL_CONNECT", _spy, raising=True)

    return calls


def _snapshot_template(**overrides):
    """构造一份与 snapshot_databases() 同形状的虚构快照。"""
    entry = {
        "exists": True,
        "size": 98304,
        "mtime_ns": 1794048003071817100,
        "sha256": "d" * 64,
        "wal": False,
        "shm": False,
        "journal": False,
    }
    entry.update(overrides)
    return entry


# ========================================================
# T01 绝对路径拦截
# ========================================================


def test_T01_absolute_path_is_blocked_before_sqlite(connect_spy):
    """正式 memory.db 的绝对路径必须被拒绝，且不触达真实 SQLite。"""

    with pytest.raises(guard.ProtectedDatabaseAccessError) as excinfo:
        sqlite3.connect(str(MEMORY_DB))

    assert excinfo.value.category == "memory"
    assert excinfo.value.path_type == "filesystem-path"
    assert connect_spy == [], "拦截必须发生在委托给 SQLite 之前"

    # 异常文本只描述边界，不含数据库内容
    message = str(excinfo.value)
    assert "测试数据库保护" in message
    assert "memory" in message


def test_T01_path_object_is_blocked(connect_spy):
    """pathlib.Path 参数同样必须被识别。"""

    with pytest.raises(guard.ProtectedDatabaseAccessError) as excinfo:
        sqlite3.connect(MEMORY_DB)

    assert excinfo.value.category == "memory"
    assert connect_spy == []


# ========================================================
# T02 相对路径拦截
# ========================================================


def test_T02_relative_path_from_project_root_is_blocked(
    connect_spy, monkeypatch
):
    """CWD 为项目根时，裸 "memory.db" 解析出的就是正式库，必须拒绝。"""

    monkeypatch.chdir(PROJECT_ROOT)

    with pytest.raises(guard.ProtectedDatabaseAccessError) as excinfo:
        sqlite3.connect("memory.db")

    assert excinfo.value.category == "memory"
    assert connect_spy == []

    # 不得创建任何文件
    assert not (PROJECT_ROOT / "memory.db.new").exists()


def test_T02_same_name_in_tmp_is_allowed(tmp_path):
    """同名文件在 tmp_path 下必须放行：判定依据是路径，不是文件名。"""

    target = tmp_path / "memory.db"

    connection = sqlite3.connect(target)

    try:
        connection.execute("CREATE TABLE t (v TEXT)")
        connection.execute("INSERT INTO t VALUES ('虚构数据')")
        connection.commit()
    finally:
        connection.close()

    assert target.exists()


# ========================================================
# T03 业务数据库拦截
# ========================================================


def test_T03_business_database_is_blocked(connect_spy):
    """data/business.db 必须与 memory.db 同样受保护。"""

    with pytest.raises(guard.ProtectedDatabaseAccessError) as excinfo:
        sqlite3.connect(str(BUSINESS_DB))

    assert excinfo.value.category == "business"
    assert excinfo.value.path_type == "filesystem-path"
    assert connect_spy == []


def test_T03_business_relative_path_is_blocked(connect_spy, monkeypatch):
    monkeypatch.chdir(PROJECT_ROOT)

    with pytest.raises(guard.ProtectedDatabaseAccessError) as excinfo:
        sqlite3.connect("data/business.db")

    assert excinfo.value.category == "business"
    assert connect_spy == []


# ========================================================
# T04 SQLite URI 拦截
# ========================================================


@pytest.mark.parametrize(
    "build_uri",
    [
        lambda path: f"file:{path}?mode=ro",
        lambda path: f"file:{path}",
        lambda path: f"file:{path}?mode=ro&immutable=1",
        lambda path: "file:" + str(path).replace("\\", "/"),
    ],
    ids=["mode=ro", "plain", "ro+immutable", "forward-slash"],
)
def test_T04_sqlite_uri_is_blocked(connect_spy, build_uri):
    """只读 URI 也拒绝：测试默认不应直接依赖正式数据。"""

    uri = build_uri(MEMORY_DB)

    with pytest.raises(guard.ProtectedDatabaseAccessError) as excinfo:
        sqlite3.connect(uri, uri=True)

    assert excinfo.value.category == "memory"
    assert excinfo.value.path_type == "sqlite-uri"

    # 委托探测：真正的 connect 不允许被调用（connect_spy 会记录）
    assert connect_spy == [], "URI 拦截同样必须发生在打开文件之前"


def test_T04_uri_memory_is_allowed():
    """file::memory: 属于内存库，必须放行。"""

    connection = sqlite3.connect("file::memory:?cache=shared", uri=True)

    try:
        connection.execute("CREATE TABLE t (v TEXT)")
    finally:
        connection.close()


# ========================================================
# T05 / T06 临时库与内存库放行
# ========================================================


def test_T05_tmp_path_database_works(tmp_path):
    """tmp_path 下的独立库必须能正常建表、插入、查询。"""

    target = tmp_path / "sandbox.db"

    connection = sqlite3.connect(str(target))

    try:
        connection.execute("CREATE TABLE demo (id INTEGER PRIMARY KEY, v TEXT)")
        connection.execute("INSERT INTO demo (v) VALUES ('虚构')")
        connection.commit()

        rows = connection.execute("SELECT v FROM demo").fetchall()
    finally:
        connection.close()

    assert rows == [("虚构",)]


def test_T06_memory_database_works():
    """:memory: 必须能正常使用。"""

    connection = sqlite3.connect(":memory:")

    try:
        connection.execute("CREATE TABLE t (v TEXT)")
        connection.execute("INSERT INTO t VALUES ('内存库')")
        connection.commit()

        assert connection.execute("SELECT v FROM t").fetchone() == ("内存库",)
    finally:
        connection.close()


# ========================================================
# T07 默认 Memory 路径阻断
# ========================================================


def test_T07_default_memory_path_is_blocked_at_connect(
    connect_spy, monkeypatch
):
    """在项目根目录触发 main.get_memory()：必须在打开正式库前被拦截。

    不依赖指纹哨兵做事后发现。
    """

    monkeypatch.chdir(PROJECT_ROOT)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "offline-test-key")

    import main as main_module

    # 清除可能存在的缓存，确保本次真的构造 Memory()
    main_module.get_memory.cache_clear()

    with pytest.raises(guard.ProtectedDatabaseAccessError) as excinfo:
        main_module.get_memory()

    assert excinfo.value.category == "memory"
    assert connect_spy == [], "Memory() 构造阶段就应被拦截"

    main_module.get_memory.cache_clear()


# ========================================================
# T08 连接别名覆盖
# ========================================================


def test_T08_both_sqlite_connect_entrypoints_are_patched(connect_spy):
    """sqlite3.connect 与 sqlite3.dbapi2.connect 必须同时被替换。"""

    assert getattr(sqlite3.connect, "_wb_db_guard", False) is True
    assert getattr(sqlite3.dbapi2.connect, "_wb_db_guard", False) is True

    with pytest.raises(guard.ProtectedDatabaseAccessError):
        sqlite3.dbapi2.connect(str(MEMORY_DB))

    assert connect_spy == []


def test_T08_no_connect_alias_in_project():
    """静态扫描：项目内不存在 from sqlite3 import connect 形式的别名。

    若存在，这类模块会在守卫安装前就绑定原始函数，从而绕过保护。
    """

    alias_pattern = re.compile(
        r"^\s*from\s+sqlite3\s+import\s+[^\n#]*\bconnect\b", re.MULTILINE
    )
    dbapi_pattern = re.compile(r"^\s*import\s+sqlite3\.dbapi2", re.MULTILINE)

    offenders = []
    scanned = 0

    for path in sorted(PROJECT_ROOT.rglob("*.py")):
        if ".git" in path.parts or "__pycache__" in path.parts:
            continue
        if ".backup_r6_5" in path.parts:
            continue

        scanned += 1

        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:  # pragma: no cover
            continue

        if alias_pattern.search(text) or dbapi_pattern.search(text):
            offenders.append(str(path.relative_to(PROJECT_ROOT)))

    assert scanned > 50, "扫描文件数异常，静态检查可能未真正执行"
    assert offenders == [], f"发现可绕过守卫的 connect 别名：{offenders}"


# ========================================================
# T09 哨兵变化检测（纯函数，虚构快照）
# ========================================================


def test_T09_identical_snapshots_report_no_change():
    before = {"memory": _snapshot_template(), "business": _snapshot_template()}
    after = {"memory": _snapshot_template(), "business": _snapshot_template()}

    assert guard.compare_snapshots(before, after) == {}


@pytest.mark.parametrize(
    "field,new_value",
    [
        ("size", 98305),
        ("mtime_ns", 1794048003071817101),
        ("sha256", "e" * 64),
        ("exists", False),
        ("wal", True),
        ("shm", True),
        ("journal", True),
    ],
)
def test_T09_each_field_change_is_detected(field, new_value):
    before = {"memory": _snapshot_template()}
    after = {"memory": _snapshot_template(**{field: new_value})}

    differences = guard.compare_snapshots(before, after)

    assert "memory" in differences
    assert field in differences["memory"]
    assert differences["memory"][field]["before"] != differences["memory"][field]["after"]

    # 可读化输出必须包含字段名与前后值
    text = guard.format_differences(differences)
    assert field in text


def test_T09_sidecar_only_change_is_detected():
    """主库不变、仅出现 -wal 也必须判定为变化。"""

    before = {"business": _snapshot_template(wal=False)}
    after = {"business": _snapshot_template(wal=True)}

    differences = guard.compare_snapshots(before, after)

    assert differences == {"business": {"wal": {"before": False, "after": True}}}


# ========================================================
# T10 失败关闭
# ========================================================


def test_T10_unresolvable_path_fails_closed(connect_spy):
    """路径无法解析时不得默认放行。"""

    class _BrokenPath:
        def __fspath__(self):
            raise RuntimeError("故意制造的路径解析异常")

    with pytest.raises(guard.ProtectedDatabaseAccessError) as excinfo:
        sqlite3.connect(_BrokenPath())

    assert excinfo.value.category == "unknown"
    assert excinfo.value.path_type == "unresolved"
    assert connect_spy == []


def test_T10_unsupported_type_fails_closed(connect_spy):
    """非路径类型同样 fail-closed，而不是伪装成 SQLite 的 TypeError。"""

    with pytest.raises(guard.ProtectedDatabaseAccessError) as excinfo:
        sqlite3.connect(12345)

    assert excinfo.value.category == "unknown"
    assert excinfo.value.path_type == "unsupported-type"
    assert connect_spy == []


def test_T10_legitimate_targets_are_not_rejected(tmp_path):
    """失败关闭不能误伤正常目标。"""

    for target in (":memory:", "", str(tmp_path / "ok.db")):
        connection = sqlite3.connect(target)
        connection.close()


# ========================================================
# 守卫元信息
# ========================================================


def test_guard_protects_exactly_two_databases():
    assert guard.PROTECTED_DATABASES == {
        "memory": PROJECT_ROOT / "memory.db",
        "business": PROJECT_ROOT / "data" / "business.db",
    }


def test_guard_is_installed_even_without_fixture():
    """守卫在导入阶段即安装，而不是等到 fixture 执行。"""

    assert getattr(sqlite3.connect, "_wb_db_guard", False) is True
    assert os.path.isabs(str(PROJECT_ROOT))


# ========================================================
# R2-02 统一隔离 fixture 的实际效果
# ========================================================


def test_R2_02_isolated_memory_lives_in_tmp_path(isolated_memory, tmp_path):
    """统一 fixture 产出的 Memory 必须落在 tmp_path 内，并且真的可用。"""

    db_path = Path(isolated_memory.db_path).resolve()

    assert db_path.is_relative_to(Path(tmp_path).resolve())
    assert db_path != MEMORY_DB.resolve()

    isolated_memory.save_message("user", "临时库消息", conversation_id="conv-a", user_id="test-user")

    messages, total = isolated_memory.get_messages("conv-a", user_id="test-user")

    assert total == 1
    assert messages[0]["content"] == "临时库消息"


def test_R2_02_two_temporary_databases_do_not_pollute(tmp_path):
    """两个独立临时库互不污染：A 看不到 B 的消息。"""

    from memory import Memory

    first = Memory(db_path=str(tmp_path / "a.db"))
    second = Memory(db_path=str(tmp_path / "b.db"))

    try:
        first.save_message("user", "A 的数据", conversation_id="c1", user_id="test-user")
        second.save_message("user", "B 的数据", conversation_id="c2", user_id="test-user")

        assert [m["content"] for m in first.get_messages("c1", user_id="test-user")[0]] == ["A 的数据"]
        assert first.get_messages("c2", user_id="test-user")[1] == 0

        assert [m["content"] for m in second.get_messages("c2", user_id="test-user")[0]] == ["B 的数据"]
        assert second.get_messages("c1", user_id="test-user")[1] == 0
    finally:
        first.close()
        second.close()


def test_R2_02_main_get_memory_never_returns_production(
    patched_main_memory, tmp_path
):
    """main.get_memory() 走真实代码路径，也只可能产出 tmp 实例。

    同时验证 lru_cache 的两种状态：
    - 已缓存时返回同一个 tmp 实例；
    - cache_clear() 后重新构造，仍然是 tmp 实例（不会回落到正式库）。
    """

    import main as main_module

    assert main_module.get_memory() is patched_main_memory
    assert main_module.get_memory.cache_info().currsize == 1

    fresh_path = Path(patched_main_memory.db_path).resolve()
    assert fresh_path.is_relative_to(Path(tmp_path).resolve())
    assert fresh_path != MEMORY_DB.resolve()

    # 清空缓存后重新取：仍必须是 tmp 实例
    main_module.get_memory.cache_clear()

    rebuilt = main_module.get_memory()

    assert Path(rebuilt.db_path).resolve().is_relative_to(Path(tmp_path).resolve())
    assert Path(rebuilt.db_path).resolve() != MEMORY_DB.resolve()
    assert main_module.get_memory.cache_info().currsize == 1


def test_R2_02_isolated_business_db_lives_in_tmp_path(
    isolated_business_db, tmp_path
):
    """业务库隔离 fixture 必须指向 tmp_path，且表结构已就绪。"""

    db_path = Path(isolated_business_db).resolve()

    assert db_path.is_relative_to(Path(tmp_path).resolve())
    assert db_path != BUSINESS_DB.resolve()

    connection = sqlite3.connect(str(db_path))

    try:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    finally:
        connection.close()

    assert tables, "tmp 业务库应当已被 init_database 建表"


def test_R2_01_official_databases_match_session_baseline():
    """会话开始时的正式库快照必须与当前状态一致（含 sidecar）。"""

    baseline = guard._SENTINEL["baseline"]

    assert baseline is not None
    assert guard.compare_snapshots(
        baseline, guard.snapshot_databases()
    ) == {}, "正式数据库或 sidecar 在本次会话期间发生了变化"
