"""项目级测试数据库保护（阶段 1-R2）。

目的
----
建立统一、默认生效、fail-closed 的测试数据库保护，禁止任何测试意外
打开或迁移正式数据库：

- 正式记忆库：``<项目根>/memory.db``
- 正式业务库：``<项目根>/data/business.db``

三层结构
--------
1. **SQLite 连接守卫**：包装 ``sqlite3.connect`` 与 ``sqlite3.dbapi2.connect``，
   在 SQLite 真正打开文件之前拒绝对正式库的任何连接，包含 ``mode=ro``
   只读 URI 与 ``query_only`` 用法。
2. **指纹哨兵**：会话级与用例级各做一次快照比对（exists / size /
   mtime_ns / SHA256 / 三个 sidecar），发现变化立即失败，且不恢复、
   不覆盖、不清理。
3. **违规计数**：只记录被拒绝的数据库类别与路径类型，不记录任何
   数据库内容、消息正文或凭据。

安装时机
--------
本文件位于项目根目录。pytest 在导入任何测试模块之前加载 conftest，
因此守卫先于 ``import main`` / ``import memory`` / ``import business.db``
等导入副作用生效。安装是幂等的：通过 ``sqlite3.connect`` 上的
``_wb_db_guard`` 标记判断，重复加载不会重复包装。

允许的目标
----------
``tmp_path`` 下的数据库、测试自建数据库、``:memory:``、``""``（SQLite
临时库）一律放行。判定基准是「解析后的绝对规范化路径是否等于正式库
路径」，而不是文件名。
"""

from __future__ import annotations

import hashlib
import ipaddress
import os
import socket
import sqlite3
import sys
import time
import types
from pathlib import Path
from urllib.parse import unquote

import pytest

# ============================================================
# 1. 受保护的正式数据库
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

PROTECTED_DATABASES = {
    "memory": PROJECT_ROOT / "memory.db",
    "business": PROJECT_ROOT / "data" / "business.db",
}

SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")

SNAPSHOT_FIELDS = (
    "exists",
    "size",
    "mtime_ns",
    "sha256",
    "wal",
    "shm",
    "journal",
)


class ProtectedDatabaseAccessError(RuntimeError):
    """测试期间尝试连接正式数据库。

    异常内容只说明被拒绝的数据库类别、路径类型和它属于测试保护边界，
    不包含数据库内容、消息正文或任何凭据。
    """

    def __init__(self, category: str, path_type: str):
        self.category = category
        self.path_type = path_type
        super().__init__(
            "[测试数据库保护] 已拒绝连接正式数据库："
            f"类别={category}；路径类型={path_type}。"
            "测试必须使用 tmp_path 或 :memory:；"
            "该拦截发生在 SQLite 真正打开文件之前。"
        )


# ============================================================
# 2. 路径解析与判定
# ============================================================


def _normalize_path(path_like) -> str:
    """把路径转换为可比较的绝对规范化形式。

    处理：``pathlib.Path`` / ``os.PathLike`` / ``bytes``、相对路径（按当前
    工作目录解析）、大小写差异、分隔符差异、符号链接（realpath）。
    不展开环境变量，避免把含 ``$``/``%`` 的临时路径改写。
    """
    raw = os.fspath(path_like)

    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "surrogateescape")

    expanded = os.path.expanduser(raw)
    absolute = os.path.abspath(expanded)

    try:
        real = os.path.realpath(absolute)
    except OSError:  # pragma: no cover - 极端文件系统异常
        real = absolute

    return os.path.normcase(real)


def _path_from_sqlite_uri(uri: str) -> str:
    """从 ``file:`` URI 中取出文件路径部分（丢弃 query 与 fragment）。

    支持 ``file:/abs/path``、``file:///C:/path``、``file:relative.db``
    以及百分号编码；``mode=ro`` 等参数不改变判定结果。
    """
    body = uri[len("file:"):]
    body = body.split("#", 1)[0]
    body = body.split("?", 1)[0]
    body = unquote(body)

    if body.startswith("//"):
        body = body[2:]
        if body.startswith("/"):
            body = body.lstrip("/")

    # Windows 上 /C:/dir/db 需要去掉前导斜杠才能被 abspath 正确解析
    if os.name == "nt" and len(body) >= 3 and body[0] == "/" and body[2] == ":":
        body = body[1:]

    return body


def _resolve_candidate(database):
    """解析 ``connect(database)`` 的目标，返回 ``(path_type, normalized)``。

    ``normalized`` 为 ``None`` 表示 SQLite 内存库或临时库，直接放行。
    无法解析时 fail-closed：抛 ``ProtectedDatabaseAccessError``。
    """
    try:
        raw = database

        if isinstance(raw, os.PathLike):
            raw = os.fspath(raw)

        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", "surrogateescape")

        if not isinstance(raw, str):
            raise ProtectedDatabaseAccessError("unknown", "unsupported-type")

        if raw in ("", ":memory:"):
            return "in-memory-or-temp", None

        if raw[:5].lower() == "file:":
            extracted = _path_from_sqlite_uri(raw)
            if extracted in ("", ":memory:"):
                return "uri-memory", None
            return "sqlite-uri", _normalize_path(extracted)

        return "filesystem-path", _normalize_path(raw)

    except ProtectedDatabaseAccessError:
        raise
    except Exception as exc:  # 路径不可解析时不默认放行
        raise ProtectedDatabaseAccessError("unknown", "unresolved") from exc


def _same_file_identity(candidate: Path):
    """按 (st_dev, st_ino) 判断是否与某个正式库是同一文件。

    覆盖硬链接、8.3 短名、符号链接残留等仅靠路径字符串无法识别的别名。
    """
    if not candidate.exists():
        return None

    try:
        cand_stat = candidate.stat()
    except OSError:
        return None

    for category, protected in PROTECTED_DATABASES.items():
        try:
            protected_stat = protected.stat()
        except OSError:
            continue
        if (cand_stat.st_dev, cand_stat.st_ino) == (
            protected_stat.st_dev,
            protected_stat.st_ino,
        ):
            return category

    return None


def classify_database_path(database):
    """判定连接目标。返回 ``(category, path_type)``。

    ``category`` 为 ``None`` 表示允许连接；否则为被保护的数据库类别。
    """
    path_type, normalized = _resolve_candidate(database)

    if normalized is None:
        return None, path_type

    for category, protected in PROTECTED_DATABASES.items():
        if normalized == _normalize_path(protected):
            return category, path_type

    identity_category = _same_file_identity(Path(normalized))

    if identity_category is not None:
        return identity_category, path_type

    return None, path_type


# ============================================================
# 3. 连接守卫
# ============================================================

BLOCKED_ATTEMPTS: list[dict] = []

_ORIGINAL_CONNECT = None
_ORIGINAL_DBAPI2_CONNECT = None


def _guarded_connect(*args, **kwargs):
    """替换 ``sqlite3.connect``：先判定，后委托。"""
    if args:
        database = args[0]
    else:
        database = kwargs.get("database", "")

    try:
        category, path_type = classify_database_path(database)
    except ProtectedDatabaseAccessError as exc:
        # 路径不可解析等 fail-closed 情况同样计入拦截统计
        BLOCKED_ATTEMPTS.append(
            {"category": exc.category, "path_type": exc.path_type}
        )
        raise

    if category is not None:
        BLOCKED_ATTEMPTS.append(
            {"category": category, "path_type": path_type}
        )
        raise ProtectedDatabaseAccessError(category, path_type)

    return _ORIGINAL_CONNECT(*args, **kwargs)


_guarded_connect._wb_db_guard = True


def install_guard():
    """安装守卫；幂等。返回是否本次真正安装。"""
    global _ORIGINAL_CONNECT, _ORIGINAL_DBAPI2_CONNECT

    if getattr(sqlite3.connect, "_wb_db_guard", False):
        return False

    _ORIGINAL_CONNECT = sqlite3.connect
    _ORIGINAL_DBAPI2_CONNECT = getattr(sqlite3.dbapi2, "connect", None)

    _guarded_connect._wb_original = _ORIGINAL_CONNECT

    sqlite3.connect = _guarded_connect
    sqlite3.dbapi2.connect = _guarded_connect

    return True


def uninstall_guard():
    """还原守卫；幂等。用于 pytest 结束时不污染调用进程。"""
    if not getattr(sqlite3.connect, "_wb_db_guard", False):
        return False

    if _ORIGINAL_CONNECT is not None:
        sqlite3.connect = _ORIGINAL_CONNECT
    if _ORIGINAL_DBAPI2_CONNECT is not None:
        sqlite3.dbapi2.connect = _ORIGINAL_DBAPI2_CONNECT

    return True


# ============================================================
# 4. 指纹快照与比较
# ============================================================


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_databases() -> dict:
    """采集正式数据库指纹：存在性、size、mtime_ns、SHA256、三个 sidecar。"""
    snapshot = {}

    for category, path in PROTECTED_DATABASES.items():
        entry = {
            "exists": False,
            "size": None,
            "mtime_ns": None,
            "sha256": None,
        }

        if path.exists():
            stat = path.stat()
            entry["exists"] = True
            entry["size"] = stat.st_size
            entry["mtime_ns"] = stat.st_mtime_ns
            entry["sha256"] = _sha256_file(path)

        for suffix in SIDECAR_SUFFIXES:
            entry[suffix.lstrip("-")] = Path(str(path) + suffix).exists()

        snapshot[category] = entry

    return snapshot


def compare_snapshots(before: dict, after: dict) -> dict:
    """比较两组快照，返回 ``{category: {field: {before, after}}}``。

    纯函数：不读文件、不依赖全局状态，便于用虚构快照单测。
    """
    differences = {}

    for category in sorted(set(before) | set(after)):
        old = before.get(category) or {}
        new = after.get(category) or {}

        changed = {}
        for field in SNAPSHOT_FIELDS:
            if old.get(field) != new.get(field):
                changed[field] = {
                    "before": old.get(field),
                    "after": new.get(field),
                }

        if changed:
            differences[category] = changed

    return differences


def format_differences(differences: dict) -> str:
    """把差异字典格式化为可读文本（只含指纹，不含数据库内容）。"""
    lines = []

    for category, fields in differences.items():
        parts = ", ".join(
            f"{name}: {delta['before']!r} -> {delta['after']!r}"
            for name, delta in fields.items()
        )
        lines.append(f"- {category}: {parts}")

    return "\n".join(lines)


# ============================================================
# 5. 哨兵状态与 fixture
# ============================================================

_SENTINEL = {
    "baseline": None,
    "failed": False,
    "failure": None,
    "checked_tests": 0,
}


@pytest.fixture(scope="session", autouse=True)
def wb_db_fingerprint_session():
    """会话级哨兵：开始时快照，结束时比对。"""
    _SENTINEL["baseline"] = snapshot_databases()

    yield

    final = snapshot_databases()
    differences = compare_snapshots(_SENTINEL["baseline"], final)

    if differences:
        _SENTINEL["failed"] = True
        _SENTINEL["failure"] = format_differences(differences)
        raise AssertionError(
            "[测试数据库保护] 正式数据库在本次 pytest 会话期间发生变化：\n"
            f"{_SENTINEL['failure']}\n"
            f"检测时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            "说明：不恢复、不覆盖、不清理 sidecar。若服务进程（PID 2744）"
            "同时持有连接，变化来源可能不是 pytest，需按「来源不明」处理。"
        )


@pytest.fixture(autouse=True)
def wb_db_fingerprint_guard():
    """用例级哨兵：每个用例前后各比对一次。"""
    if _SENTINEL["failed"]:
        pytest.fail(
            "[测试数据库保护] 已有用例检测到正式数据库变化，"
            "后续用例全部中止，避免继续触及正式库。\n"
            f"{_SENTINEL['failure'] or ''}",
            pytrace=False,
        )

    before = snapshot_databases()

    yield

    after = snapshot_databases()
    differences = compare_snapshots(before, after)
    _SENTINEL["checked_tests"] += 1

    if differences:
        _SENTINEL["failed"] = True
        _SENTINEL["failure"] = format_differences(differences)
        pytest.fail(
            "[测试数据库保护] 正式数据库在本用例执行期间发生变化：\n"
            f"{_SENTINEL['failure']}\n"
            f"检测时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            "说明：不恢复、不覆盖、不清理 sidecar。若服务进程（PID 2744）"
            "同时持有连接，变化来源可能不是 pytest，需按「来源不明」处理。",
            pytrace=False,
        )


# ============================================================
# 5b. R2-02：统一的测试数据库隔离 fixture
#
# 这些 fixture 是「新建测试唯一应当使用的数据库入口」：
# 只要测试需要 Memory / main.get_memory / business.db，就应直接请求
# 对应 fixture，而不是自己拼路径或依赖默认值。
#
# 关于 @lru_cache(maxsize=1)：main.get_memory() 缓存的是 Memory 实例，
# 而 Memory.db_path 可能是相对路径 "memory.db"，其解析依赖「调用时的
# CWD」。若某个测试在 tmp 目录下调用过一次，后续测试即使换了 CWD 也会
# 复用同一实例，从而打开正式库。因此 patched_main_memory 在替换前后都
# 显式 cache_clear()，绝不复用任何既有的缓存实例。
# ============================================================


@pytest.fixture
def isolated_memory(tmp_path):
    """指向 tmp_path 的 Memory 实例；结束时关闭连接。"""
    from memory import Memory

    instance = Memory(db_path=str(tmp_path / "memory.db"))

    yield instance

    instance.close()


@pytest.fixture
def patched_main_memory(tmp_path, monkeypatch):
    """统一替身入口：让 main.get_memory() 只产出 tmp 实例。

    做法是把 ``main.Memory`` 这个构造工厂换掉，而不是替换 ``get_memory``
    本身——这样 ``@lru_cache(maxsize=1)`` 的真实行为（含 cache_info /
    cache_clear）仍然可被测到，同时保证缓存里只可能是 tmp 实例。

    进入与退出都执行 cache_clear()：绝不复用上一个测试留下的实例，
    因为 ``Memory.db_path`` 若是相对路径，其解析依赖调用时的 CWD。
    """
    import main as main_module
    from memory import Memory

    created = []

    def _factory(*args, **kwargs):
        instance = Memory(db_path=str(tmp_path / "memory.db"))
        created.append(instance)
        return instance

    main_module.get_memory.cache_clear()
    monkeypatch.setattr(main_module, "Memory", _factory)

    memory = main_module.get_memory()

    yield memory

    main_module.get_memory.cache_clear()
    for instance in created:
        instance.close()


@pytest.fixture
def isolated_business_db(tmp_path, monkeypatch):
    """把 business.db 锁定到 tmp_path 并初始化表结构。"""
    import business.db as business_db

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    db_path = data_dir / "business.db"

    monkeypatch.setattr(business_db, "DATA_DIR", data_dir)
    monkeypatch.setattr(business_db, "DATABASE_PATH", db_path)

    # 隔离必须真实生效，否则立刻失败而不是继续使用正式库
    assert business_db.DATABASE_PATH == db_path
    assert business_db.DATABASE_PATH.is_relative_to(tmp_path)

    business_db.init_database()

    return db_path


# ============================================================
# 6. pytest 生命周期
# ============================================================


@pytest.fixture(autouse=True)
def default_active_user():
    """PAI-AUTH-MVP：测试环境默认「当前请求用户」= test-user。

    身份通过 ``agent._ACTIVE_USER_ID``（ContextVar）传递，正常路径由
    ``Agent.run(..., user_id=...)`` 设置。但大量既有测试直接调用
    ``Agent.save_extracted_memories`` / ``execute_tool`` 等内部方法，
    没有经过 run，取身份时会抛 RuntimeError。

    这个夹具只为「测试环境」提供一个确定身份，不放宽生产约束：
    ``Agent.run`` 的 user_id 仍是必填关键字参数，缺失即 TypeError。

    用 autouse 而不是逐文件声明，是为了让四个测试目录共用同一约定，
    避免各目录自己写一份、行为不一致。
    """
    try:
        import agent as agent_module
    except Exception:  # pragma: no cover - 导入失败时不影响不相关测试
        yield None
        return

    token = agent_module._ACTIVE_USER_ID.set("test-user")
    try:
        yield "test-user"
    finally:
        agent_module._ACTIVE_USER_ID.reset(token)


def pytest_configure(config):
    """配置阶段再次确认全部守卫已安装（均幂等）。"""
    install_guard()
    install_network_guard()
    install_http_guard()
    install_sdk_guard()
    install_env_guard()


def pytest_unconfigure(config):
    """会话结束还原 patch，避免污染同进程的后续调用者。

    网络守卫只还原"本模块亲自安装的入口"，子目录 conftest 自带的更强守卫
    不会被错误覆盖。HTTP 传输层守卫按"后装先卸"在 socket 守卫之前还原。
    """
    uninstall_http_guard()
    uninstall_network_guard()
    uninstall_sdk_guard()
    uninstall_env_guard()
    uninstall_guard()


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """汇总拦截次数与哨兵结果；只报计数与类别，不报任何内容。"""
    categories = {}

    for item in BLOCKED_ATTEMPTS:
        key = f"{item['category']}/{item['path_type']}"
        categories[key] = categories.get(key, 0) + 1

    terminalreporter.write_sep("-", "测试数据库保护（conftest.py · 阶段 1-R2）")
    terminalreporter.write_line(
        f"blocked_attempts={len(BLOCKED_ATTEMPTS)} detail={categories or '{}'}"
    )
    terminalreporter.write_line(
        "protected: "
        + ", ".join(f"{k}={v}" for k, v in PROTECTED_DATABASES.items())
    )
    terminalreporter.write_line(
        f"fingerprint_checked_tests={_SENTINEL['checked_tests']} "
        f"fingerprint_failed={_SENTINEL['failed']}"
    )

    report = guards_report()
    terminalreporter.write_sep("-", "测试网络/环境隔离（conftest.py · 阶段 1-R2.1）")
    terminalreporter.write_line(
        f"network_installed={report['network_installed']} "
        f"network_owner={report['network_owner']} "
        f"network_blocked={report['network_blocked']} "
        f"http_installed={report['http_installed']} "
        f"http_blocked={report['http_blocked']} "
        f"sdk_installed={report['sdk_installed']} "
        f"sdk_blocked={report['sdk_blocked']} "
        f"dotenv_is_fake={report['dotenv_is_fake']} "
        f"api_key_is_fake={report['api_key_is_fake']}"
    )
    terminalreporter.write_line(
        f"detail: non_loopback={len(NETWORK_STATE['non_loopback'])} "
        f"dns={len(NETWORK_STATE['dns_blocked'])} "
        f"udp={len(NETWORK_STATE['udp_blocked'])} "
        f"http={len(NETWORK_STATE['http_blocked'])} "
        f"sdk={len(NETWORK_STATE['sdk_blocked'])}"
    )


# ============================================================
# 7. 全仓统一网络守卫（阶段 1-R2.1）
#
# 目标：pytest 进程默认离线。非回环连接、DNS 查询、UDP 外网发送
# 必须在真实连接建立之前被拒绝；回环（127.0.0.0/8、::1、localhost）
# 放行，以保证本机 FastAPI / socket 相关测试不受影响。
#
# 不设置 `_wb_guard` 标记：`自动测试脚本/conftest.py` 用该标记判断自己的
# 守卫是否已安装，两者必须能各自独立安装（详见该文件第 8 节）。
# ============================================================

NETWORK_STATE = {
    "non_loopback": [],
    "dns_blocked": [],
    "udp_blocked": [],
    "sdk_blocked": [],
    # 阶段 1-R2.2：HTTP 传输层按「真实目标」拦截的计数（见 7.5 节）
    "http_blocked": [],
}

_NETWORK_ORIGINALS: dict = {}
_NETWORK_INSTALLED = False


class BlockedNetworkAccessError(RuntimeError):
    """测试期间尝试建立非回环网络连接。

    只携带类别与目标地址，不携带任何请求内容、请求体或凭据。
    """

    def __init__(self, kind: str, target: str):
        self.kind = kind
        self.target = target
        super().__init__(
            f"[测试网络保护] 已拒绝非回环网络访问：类别={kind}；目标={target}。"
            "该拦截发生在真实连接建立之前；"
            "测试仅允许访问 127.0.0.0/8、::1、localhost。"
        )


class BlockedSDKRequestError(RuntimeError):
    """测试期间尝试发起真实模型 SDK 请求。

    该拦截发生在任何 HTTP 请求发出之前，早于建连。
    """

    def __init__(self, target: str):
        self.target = target
        super().__init__(
            f"[测试 SDK 保护] 已拒绝真实模型 SDK 请求：目标={target}。"
            "该拦截发生在 HTTP 请求发出之前；测试必须使用 Fake 客户端。"
        )


def _host_of(address):
    """从 socket 地址参数中取出主机部分；无法判断时返回 None。"""
    if isinstance(address, tuple) and address:
        return address[0]
    if isinstance(address, (str, bytes)):
        return address
    return None


def _is_loopback_host(host) -> bool:
    """判定主机是否为回环：127.0.0.0/8、::1、localhost。

    仅放行这三类，不做任何域名解析（解析会触发真实 DNS）。
    """
    if host is None:
        return False

    if isinstance(host, bytes):
        host = host.decode("utf-8", "replace")

    if not isinstance(host, str):
        return False

    name = host.split("%", 1)[0].strip()
    if name.startswith("[") and name.endswith("]"):
        name = name[1:-1]

    if not name:
        return False

    if name.lower() == "localhost":
        return True

    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


def _resolve_network_original(kind: str):
    """取当前保存的原始实现；未安装时返回 None。"""
    return _NETWORK_ORIGINALS.get(kind)


def _guarded_socket_connect(self, address, *args, **kwargs):
    if _is_loopback_host(_host_of(address)):
        return _resolve_network_original("connect")(self, address, *args, **kwargs)
    NETWORK_STATE["non_loopback"].append(("connect", repr(address)))
    raise BlockedNetworkAccessError("socket.connect", repr(address))


def _guarded_socket_connect_ex(self, address, *args, **kwargs):
    if _is_loopback_host(_host_of(address)):
        return _resolve_network_original("connect_ex")(self, address, *args, **kwargs)
    NETWORK_STATE["non_loopback"].append(("connect_ex", repr(address)))
    raise BlockedNetworkAccessError("socket.connect_ex", repr(address))


def _guarded_create_connection(address, *args, **kwargs):
    if _is_loopback_host(_host_of(address)):
        return _resolve_network_original("create_connection")(address, *args, **kwargs)
    NETWORK_STATE["non_loopback"].append(("create_connection", repr(address)))
    raise BlockedNetworkAccessError("socket.create_connection", repr(address))


def _guarded_getaddrinfo(host, port, *args, **kwargs):
    # host 为 None / "" 表示本机通配解析（bind 场景），不产生外部 DNS 查询
    if host is None or host == "" or _is_loopback_host(host):
        return _resolve_network_original("getaddrinfo")(host, port, *args, **kwargs)
    NETWORK_STATE["dns_blocked"].append((repr(host), port))
    raise BlockedNetworkAccessError("socket.getaddrinfo", f"{host!r}:{port}")


def _guarded_socket_sendto(self, data, *args, **kwargs):
    address = kwargs.get("address")
    if address is None and args:
        address = args[-1]
    # address 为 None 表示已连接套接字（connect 已被守卫拦截）
    if address is None or _is_loopback_host(_host_of(address)):
        return _resolve_network_original("sendto")(self, data, *args, **kwargs)
    NETWORK_STATE["udp_blocked"].append(repr(address))
    raise BlockedNetworkAccessError("socket.sendto", repr(address))


def _guarded_socket_sendmsg(self, buffers, *args, **kwargs):
    address = kwargs.get("address")
    if address is None and len(args) >= 3:
        address = args[2]
    if address is None or _is_loopback_host(_host_of(address)):
        return _resolve_network_original("sendmsg")(self, buffers, *args, **kwargs)
    NETWORK_STATE["udp_blocked"].append(repr(address))
    raise BlockedNetworkAccessError("socket.sendmsg", repr(address))


for _net_guard in (
    _guarded_socket_connect,
    _guarded_socket_connect_ex,
    _guarded_create_connection,
    _guarded_getaddrinfo,
    _guarded_socket_sendto,
    _guarded_socket_sendmsg,
):
    _net_guard._wb_net_guard = True


def install_network_guard() -> bool:
    """安装统一网络守卫；已安装时返回 False（幂等）。

    两种"已安装"都算数：
    - 本模块的 ``_wb_net_guard``；
    - 子目录 conftest 自带的 ``_wb_guard``（例如 `自动测试脚本/conftest.py`
      的状态统计型守卫）。
    后者绝不能被本模块再包一层：那会把"原始实现"记成对方的守卫，
    双方互相委托形成无限递归。
    """
    global _NETWORK_INSTALLED

    current = socket.socket.connect
    if getattr(current, "_wb_net_guard", False):
        _NETWORK_INSTALLED = True
        return False
    if getattr(current, "_wb_guard", False):
        # 已有更强的专用守卫在位，本模块让位，不覆盖、不叠加
        _NETWORK_INSTALLED = False
        return False

    _NETWORK_ORIGINALS.update(
        {
            "connect": socket.socket.connect,
            "connect_ex": socket.socket.connect_ex,
            "create_connection": socket.create_connection,
            "getaddrinfo": socket.getaddrinfo,
            "sendto": socket.socket.sendto,
        }
    )

    socket.socket.connect = _guarded_socket_connect
    socket.socket.connect_ex = _guarded_socket_connect_ex
    socket.create_connection = _guarded_create_connection
    socket.getaddrinfo = _guarded_getaddrinfo
    socket.socket.sendto = _guarded_socket_sendto

    original_sendmsg = getattr(socket.socket, "sendmsg", None)
    if original_sendmsg is not None:
        _NETWORK_ORIGINALS["sendmsg"] = original_sendmsg
        socket.socket.sendmsg = _guarded_socket_sendmsg

    _NETWORK_INSTALLED = True
    return True


def uninstall_network_guard() -> bool:
    """还原网络守卫；只还原本模块亲自安装的入口（用 is 判定）。"""
    global _NETWORK_INSTALLED

    restored = False

    if socket.socket.connect is _guarded_socket_connect:
        socket.socket.connect = _NETWORK_ORIGINALS["connect"]
        restored = True
    if socket.socket.connect_ex is _guarded_socket_connect_ex:
        socket.socket.connect_ex = _NETWORK_ORIGINALS["connect_ex"]
        restored = True
    if socket.create_connection is _guarded_create_connection:
        socket.create_connection = _NETWORK_ORIGINALS["create_connection"]
        restored = True
    if socket.getaddrinfo is _guarded_getaddrinfo:
        socket.getaddrinfo = _NETWORK_ORIGINALS["getaddrinfo"]
        restored = True
    if socket.socket.sendto is _guarded_socket_sendto:
        socket.socket.sendto = _NETWORK_ORIGINALS["sendto"]
        restored = True
    if (
        "sendmsg" in _NETWORK_ORIGINALS
        and getattr(socket.socket, "sendmsg", None) is _guarded_socket_sendmsg
    ):
        socket.socket.sendmsg = _NETWORK_ORIGINALS["sendmsg"]
        restored = True

    if restored:
        _NETWORK_INSTALLED = False

    return restored


def network_guard_installed() -> bool:
    """当前 socket 上是否装有本模块的网络守卫。"""
    return getattr(socket.socket.connect, "_wb_net_guard", False)


def network_guard_owner() -> str:
    """返回当前生效的网络守卫来源：root / subdir / none。"""
    current = socket.socket.connect
    if getattr(current, "_wb_net_guard", False):
        return "root"
    if getattr(current, "_wb_guard", False):
        return "subdir"
    return "none"


# ============================================================
# 7.5 HTTP 传输层真实目标校验（阶段 1-R2.2）
#
# 为什么 socket 守卫不够
# ----------------------
# 第七节的守卫只看「连接对端地址」。走 HTTP 代理时，对端是**本地回环
# 代理**（例如 127.0.0.1:52403），于是被判定为回环而放行 —— 但代理会把
# 请求转发到任意真实外网目标。也就是说：
#   「代理 IP 是回环」≠「请求目标是回环」。
# 仅凭代理地址是否回环来判断安全性是错误的，必须看真正的请求目标。
#
# 拦截点为什么选传输层
# --------------------
# 选 ``httpx.HTTPTransport.handle_request`` /
# ``httpx.AsyncHTTPTransport.handle_async_request``，而不是 ``Client.send``：
#   1. 这两个方法是 httpx 真正发起 IO 的唯一入口。直连与代理两条路径都
#      汇聚于此（代理时 ``self._pool`` 是 ``HTTPProxy``，但 ``request`` 不变）；
#   2. ``request.url`` 始终是**真实目标**，与是否经过代理无关（已实测：
#      显式代理与 trust_env 环境代理下 ``request.url.host`` 均为真实主机）；
#   3. ``httpx.MockTransport`` 继承 ``AsyncBaseTransport``，**不是**
#      ``HTTPTransport`` / ``AsyncHTTPTransport`` 的子类 —— 因此 Fake
#      transport、Fake SDK 完全不受影响。若改在 ``Client.send`` 上拦，
#      ``MockTransport`` 会被一并拦掉，反而破坏离线 Fake 客户端。
#
# 覆盖范围
# --------
# 按「库 × 同步/异步」四个入口矩阵覆盖，不假设只有 httpx：
#   httpx  / HTTPTransport.handle_request
#   httpx  / AsyncHTTPTransport.handle_async_request
#   httpx2 / HTTPTransport.handle_request
#   httpx2 / AsyncHTTPTransport.handle_async_request
# 直连、显式 ``proxy=``、``trust_env`` 环境代理（``HTTP_PROXY`` /
# ``HTTPS_PROXY`` / ``ALL_PROXY`` 及全部小写变体）、openai SDK 内部客户端
# 全部经过同一判定。
#
# ``httpx2`` 必须单独覆盖：它是独立分发的平行实现（不同包、不同类对象），
# 本项目实测 openai 3.19.2 的 ``OpenAI()._client`` 就是 ``httpx2.Client``。
# 只 patch httpx 会留下一条真实的出网通道。
# ============================================================

_HTTP_STATE = {
    "installed": False,
    "modules": [],
    "blocked": 0,
    "targets": [],
}

_HTTP_ORIGINALS: dict = {}
_HTTP_INSTALLED = False

# 需要纳入目标的 HTTP 客户端库。
#
# 注意 ``httpx2``：它是**独立分发**的平行实现（不同包、不同类对象），
# openai 3.19.2 与 starlette 1.6.0 都依赖它。只 patch ``httpx`` 会留下
# 一条真实的出网通道 —— 本项目实测：直接驱动 openai 内部客户端
# （``OpenAI()._client``，其实是 ``httpx2.Client``）可经回环代理真实出网。
# 因此这里按库清单遍历，缺哪个就跳过哪个，不假设只有 httpx。
_HTTP_MODULES = ("httpx", "httpx2")

_HTTP_TRANSPORTS = (
    ("HTTPTransport", "handle_request"),
    ("AsyncHTTPTransport", "handle_async_request"),
)


def _describe_http_target(request) -> str:
    """只取 scheme + host + port。

    绝不记录 path / query / header / body，避免把请求内容或凭据写进日志。
    """
    url = getattr(request, "url", None)
    if url is None:
        return "?"

    scheme = getattr(url, "scheme", None) or "?"
    host = getattr(url, "host", None) or "?"
    port = getattr(url, "port", None)
    if port is None:
        return f"{scheme}://{host}"
    return f"{scheme}://{host}:{port}"


def _http_request_is_allowed(request) -> bool:
    """按**真实目标主机**判定，不看代理地址。

    与方法无关：GET / POST / CONNECT 一视同仁 —— 建 HTTPS 隧道时
    ``request.url`` 同样是真实目标，因此同一判定即可覆盖。
    """
    url = getattr(request, "url", None)
    host = getattr(url, "host", None) if url is not None else None
    return _is_loopback_host(host)


def _reject_http_request(request, kind: str):
    target = _describe_http_target(request)
    _HTTP_STATE["blocked"] += 1
    _HTTP_STATE["targets"].append(target)
    NETWORK_STATE["http_blocked"].append((kind, target))
    raise BlockedNetworkAccessError(kind, target)


def _make_http_guard(key: str, kind: str):
    """按 (库, 入口) 生成一个守卫。

    闭包绑定 ``_HTTP_ORIGINALS`` 的 key，因此 httpx 与 httpx2 各自委托到
    自己的原始实现，不会互相串用。
    """

    def _guarded(self, request):
        if _http_request_is_allowed(request):
            return _HTTP_ORIGINALS[key](self, request)
        _reject_http_request(request, kind)

    _guarded._wb_http_guard = True
    _guarded._wb_http_key = key
    return _guarded


# 预生成：守卫函数是模块级单例，uninstall 才能用 `is` 可靠判定所有权
_HTTP_GUARDS = {
    f"{module_name}:{method_name}": _make_http_guard(
        f"{module_name}:{method_name}", f"{module_name}.{class_name}"
    )
    for module_name in _HTTP_MODULES
    for class_name, method_name in _HTTP_TRANSPORTS
}


def _http_transport_class(module_name: str, class_name: str):
    """取传输类；库或类不存在时返回 None（不触发额外导入）。"""
    module = sys.modules.get(module_name)
    if module is None:
        return None
    return getattr(module, class_name, None)


def http_guard_installed() -> bool:
    """HTTP 守卫是否**当前实际在位**（覆盖全部已导入库的同步入口）。

    阶段 1-R2.2-CLOSE：以被 patch 对象的身份判定，不看历史字典、
    也不看布尔闩锁 —— 后两者只能说明"曾经装过"，不能说明"现在还在"。
    """
    covered = 0

    for module_name in _HTTP_MODULES:
        transport_class = _http_transport_class(module_name, "HTTPTransport")
        if transport_class is None:
            continue
        ours = _HTTP_GUARDS.get(f"{module_name}:handle_request")
        if getattr(transport_class, "handle_request", None) is not ours:
            return False
        covered += 1

    return covered > 0


def http_guard_modules() -> tuple:
    """当前实际处于保护下的客户端库（按在位判定，而非历史记录）。"""
    covered = []

    for module_name in _HTTP_MODULES:
        transport_class = _http_transport_class(module_name, "HTTPTransport")
        if transport_class is None:
            continue
        if (
            getattr(transport_class, "handle_request", None)
            is _HTTP_GUARDS.get(f"{module_name}:handle_request")
        ):
            covered.append(module_name)

    return tuple(covered)


def _http_entrypoint_state():
    """枚举四个 HTTP 入口的当前所有者：[(key, current, ours, is_ours, is_foreign)]。"""
    entries = []

    for module_name in _HTTP_MODULES:
        module = sys.modules.get(module_name)
        if module is None:
            continue
        for class_name, method_name in _HTTP_TRANSPORTS:
            transport_class = getattr(module, class_name, None)
            if transport_class is None:
                continue
            key = f"{module_name}:{method_name}"
            ours = _HTTP_GUARDS.get(key)
            current = getattr(transport_class, method_name, None)
            entries.append(
                (
                    key,
                    current,
                    ours,
                    current is ours,
                    bool(
                        current is not None
                        and current is not ours
                        and getattr(current, "_wb_http_guard", False)
                    ),
                )
            )

    return entries


def http_guard_entrypoints() -> tuple:
    """**本守卫**当前实际在位的入口键名（形如 ``httpx:handle_request``）。

    与 ``_HTTP_ORIGINALS`` 无关：后者只说明"曾经替换过什么"。
    """
    return tuple(key for key, _c, _o, is_ours, _f in _http_entrypoint_state() if is_ours)


def http_foreign_entrypoints() -> tuple:
    """被**他方守卫**接管（带 ``_wb_http_guard`` 但不属于本模块）的入口键名。"""
    return tuple(
        key for key, _c, _o, _ours, is_foreign in _http_entrypoint_state() if is_foreign
    )


def install_http_guard() -> bool:
    """按**实际状态**安装 HTTP 传输层守卫；返回本次是否真正安装。

    - 已在位（对象身份等于本模块守卫）→ 不重复 patch；
    - 已被他人替换且带同类标记 → 让位，绝不覆盖；
    - 底层库不可导入 → 不产生任何"安装成功"的假象。
    """
    global _HTTP_INSTALLED

    applied = []

    try:
        for module_name in _HTTP_MODULES:
            try:
                module = __import__(module_name)
            except Exception:
                # 单个库不可用不得影响其他库
                continue

            for class_name, method_name in _HTTP_TRANSPORTS:
                transport_class = getattr(module, class_name, None)
                if transport_class is None:
                    continue

                key = f"{module_name}:{method_name}"
                ours = _HTTP_GUARDS[key]
                current = getattr(transport_class, method_name, None)

                if current is ours:
                    continue
                if current is None:
                    continue
                if getattr(current, "_wb_http_guard", False):
                    # 他方守卫在位：让位，不覆盖、不叠加
                    continue

                # 先落 patch，成功后才登记原始值与本次安装记录：
                # 若 setattr 本身失败，则不会留下指向"未曾替换"的脏登记。
                setattr(transport_class, method_name, ours)
                _HTTP_ORIGINALS[key] = current
                applied.append((transport_class, method_name, key, current))
    except BaseException:
        # 部分安装失败 → 逆序回滚本次已装入口并清除登记，不留半安装状态
        for transport_class, method_name, key, original in reversed(applied):
            if getattr(transport_class, method_name, None) is _HTTP_GUARDS[key]:
                setattr(transport_class, method_name, original)
            _HTTP_ORIGINALS.pop(key, None)
        _HTTP_INSTALLED = http_guard_installed()
        _HTTP_STATE["installed"] = _HTTP_INSTALLED
        _HTTP_STATE["modules"] = list(http_guard_modules())
        raise

    _HTTP_INSTALLED = http_guard_installed()
    _HTTP_STATE["installed"] = _HTTP_INSTALLED
    _HTTP_STATE["modules"] = list(http_guard_modules())
    return bool(applied)


def uninstall_http_guard() -> bool:
    """还原 HTTP 传输层入口；只还原本模块亲自安装的方法（身份判定）。

    成功还原的 key 会从 ``_HTTP_ORIGINALS`` 移除，避免"卸载后字典仍在"
    被误当作"守卫仍在位"。
    """
    global _HTTP_INSTALLED

    restored = False
    restored_keys = []

    for key, original in list(_HTTP_ORIGINALS.items()):
        module_name, _, method_name = key.partition(":")
        module = sys.modules.get(module_name)
        if module is None:
            continue

        for class_name, entry_name in _HTTP_TRANSPORTS:
            if entry_name != method_name:
                continue
            transport_class = getattr(module, class_name, None)
            if transport_class is None:
                continue
            if getattr(transport_class, entry_name, None) is _HTTP_GUARDS.get(key):
                setattr(transport_class, entry_name, original)
                restored = True
                restored_keys.append(key)

    for key in restored_keys:
        _HTTP_ORIGINALS.pop(key, None)

    _HTTP_INSTALLED = http_guard_installed()
    _HTTP_STATE["installed"] = _HTTP_INSTALLED
    _HTTP_STATE["modules"] = list(http_guard_modules())

    return restored


# ============================================================
# 8. 全仓统一真实 SDK 请求拦截（阶段 1-R2.1）
#
# 不替换 `openai.OpenAI` 类本身（那会破坏 Fake SDK 与既有断言），
# 而是拦截 SDK 的唯一请求出口 `SyncAPIClient.request` /
# `AsyncAPIClient.request`：客户端可正常构造、属性可正常读取，
# 一旦真正发起请求即在发出任何 HTTP 之前抛错。
# ============================================================

SDK_STATE = {
    "installed": False,
    "blocked": 0,
    "targets": [],
    "openai_available": None,
}

_SDK_ORIGINALS: dict = {}


def _describe_sdk_call(args, kwargs) -> str:
    """只取方法名与 URL 路径，绝不记录请求体/提示词/凭据。

    openai 3.x 的签名是 ``request(self, cast_to, options, *, stream, ...)``，
    因此 options 位于第二个位置参数。
    """
    options = kwargs.get("options")
    if options is None:
        if len(args) >= 2:
            options = args[1]
        elif args:
            options = args[0]
    method = getattr(options, "method", None) or "?"
    url = getattr(options, "url", None) or "?"
    return f"{method} {url}"


def _guarded_sdk_request(self, *args, **kwargs):
    target = _describe_sdk_call(args, kwargs)
    SDK_STATE["blocked"] += 1
    SDK_STATE["targets"].append(target)
    NETWORK_STATE["sdk_blocked"].append(target)
    raise BlockedSDKRequestError(target)


_guarded_sdk_request._wb_sdk_guard = True


_SDK_ENTRYPOINTS = ("SyncAPIClient", "AsyncAPIClient")


def _sdk_client_class(class_name: str):
    """取 `openai._base_client.<class_name>`；库未导入时返回 None。"""
    module = sys.modules.get("openai")
    if module is None:
        return None
    base_client = getattr(module, "_base_client", None)
    if base_client is None:
        return None
    return getattr(base_client, class_name, None)


def sdk_guard_installed() -> bool:
    """守卫是否真的在位：逐入口比对对象身份，而不是读历史布尔位。

    任一入口仍是别的实现（被外部覆盖）即视为不在位；一个入口都覆盖不到
    （openai 未导入 / 类缺失）同样不算安装成功。
    """
    covered = 0

    for class_name in _SDK_ENTRYPOINTS:
        client_class = _sdk_client_class(class_name)
        if client_class is None:
            continue
        if getattr(client_class, "request", None) is not _guarded_sdk_request:
            return False
        covered += 1

    return covered > 0


def sdk_guard_classes() -> tuple:
    """当前由本守卫实际覆盖的客户端类名。"""
    covered = []

    for class_name in _SDK_ENTRYPOINTS:
        client_class = _sdk_client_class(class_name)
        if client_class is None:
            continue
        if getattr(client_class, "request", None) is _guarded_sdk_request:
            covered.append(class_name)

    return tuple(covered)


def install_sdk_guard() -> bool:
    """安装 SDK 请求守卫。

    幂等且可自愈：不依赖 ``SDK_STATE["installed"]`` 闩锁，而是逐个入口
    比对当前对象 —— 已被本守卫覆盖则跳过；被第三方守卫覆盖（``_wb_sdk_guard``）
    则让位不覆盖；只有确实由本守卫完成替换才返回 True。
    部分失败时回滚已替换的入口，避免留下半装状态。
    """
    applied = []

    try:
        try:
            import openai
        except Exception:
            openai = None

        if openai is None:
            SDK_STATE["openai_available"] = False
            SDK_STATE["installed"] = sdk_guard_installed()
            return False

        SDK_STATE["openai_available"] = True

        base_client = getattr(openai, "_base_client", None)
        if base_client is not None:
            for class_name in _SDK_ENTRYPOINTS:
                client_class = getattr(base_client, class_name, None)
                if client_class is None:
                    continue
                current = getattr(client_class, "request", None)
                if current is _guarded_sdk_request:
                    continue
                if current is None:
                    continue
                if getattr(current, "_wb_sdk_guard", False):
                    # 已有他方守卫在位，不盲目覆盖
                    continue
                setattr(client_class, "request", _guarded_sdk_request)
                _SDK_ORIGINALS[class_name] = current
                applied.append((client_class, class_name, current))
    except BaseException:
        for client_class, class_name, original in reversed(applied):
            if getattr(client_class, "request", None) is _guarded_sdk_request:
                setattr(client_class, "request", original)
            _SDK_ORIGINALS.pop(class_name, None)
        SDK_STATE["installed"] = sdk_guard_installed()
        raise

    SDK_STATE["installed"] = sdk_guard_installed()
    return bool(applied)


def uninstall_sdk_guard() -> bool:
    """还原 SDK 请求出口。

    只还原"当前仍指向本守卫、且记录在案"的入口；被外部替换过的入口
    保持原样（不覆盖他人的 patch），其记录也保留，留给真正的所有者处理。
    """
    restored = False
    restored_keys = []

    for class_name, original in list(_SDK_ORIGINALS.items()):
        client_class = _sdk_client_class(class_name)
        if client_class is None:
            continue
        if getattr(client_class, "request", None) is not _guarded_sdk_request:
            continue
        setattr(client_class, "request", original)
        restored = True
        restored_keys.append(class_name)

    for class_name in restored_keys:
        _SDK_ORIGINALS.pop(class_name, None)

    SDK_STATE["installed"] = sdk_guard_installed()
    return restored


# ============================================================
# 9. 全仓统一 dotenv / 密钥隔离（阶段 1-R2.1）
#
# 目标：pytest 收集阶段不得加载正式 `.env`，测试进程不得使用真实密钥。
# 做法：在 conftest 导入时（早于任何测试模块 import）把 dotenv 的三个
# 入口换成假实现，并把 DEEPSEEK_API_KEY 固定为无名假值；退出时按"仍是
# 假值才还原"的规则恢复，避免与子目录 conftest 的还原互相覆盖。
# ============================================================

FAKE_API_KEY = "offline-test-key"
_PROTECTED_ENV_KEYS = ("DEEPSEEK_API_KEY",)

ENV_STATE = {
    "installed": False,
    "dotenv_module": "unknown",
    "fake_key_installed": False,
    "had_real_key_before": None,
    "run_real_llm_requested": None,
}

_ENV_ORIGINALS: dict = {}
_DOTENV_ORIGINALS: dict = {}
# 本守卫实际安装到对象上的入口：[(owner, attr, key, fake)]
# 卸载时逐条用 `is` 判定"该位置当前是否仍是本守卫装上去的对象"。
_ENV_APPLIED: list = []
# 本守卫自己伪造的 dotenv 模块（仅当真实 dotenv 不可用时出现）
_ENV_SYNTHETIC_MODULE = None
_ENV_INSTALLED = False

_FAKE_DOTENV_MODULE_FLAG = "_wb_fake_dotenv_module"


def _fake_load_dotenv(*args, **kwargs):
    """假 load_dotenv：不读文件，返回 False（等价于"未找到 .env"）。"""
    return False


def _fake_dotenv_values(*args, **kwargs):
    return {}


def _fake_find_dotenv(*args, **kwargs):
    return ""


_fake_load_dotenv._wb_fake_dotenv = True
_fake_dotenv_values._wb_fake_dotenv = True
_fake_find_dotenv._wb_fake_dotenv = True

_DOTENV_FAKES = (
    ("load_dotenv", _fake_load_dotenv),
    ("dotenv_values", _fake_dotenv_values),
    ("find_dotenv", _fake_find_dotenv),
)


def _dotenv_entry_state():
    """当前可见的 dotenv 入口：[（owner, attr, key, fake）]。

    只包含"真的存在于 ``sys.modules`` 里"的模块 —— 底层库没导入时
    不能凭空认定入口已被隔离。
    """
    entries = []

    dotenv_module = sys.modules.get("dotenv")
    if dotenv_module is not None:
        for name, fake in _DOTENV_FAKES:
            entries.append((dotenv_module, name, name, fake))

    dotenv_main = sys.modules.get("dotenv.main")
    if dotenv_main is not None and not getattr(
        dotenv_main, _FAKE_DOTENV_MODULE_FLAG, False
    ):
        for name, fake in _DOTENV_FAKES:
            entries.append((dotenv_main, name, f"main.{name}", fake))

    return entries


def env_guard_installed() -> bool:
    """dotenv 入口是否都被假实现隔离。

    本守卫的假实现在位 → 算；他方让位后的同类假实现（带
    ``_wb_fake_dotenv`` 标记）→ 同样算：隔离的安全性来自"入口不可用于
    读文件"，而不是"必须是我的函数"。
    但一个入口都看不到（库未导入）时，不得声称安装成功。
    """
    entries = _dotenv_entry_state()
    if not entries:
        return False

    for owner, attr, _key, fake in entries:
        current = getattr(owner, attr, None)
        if current is fake:
            continue
        if getattr(current, "_wb_fake_dotenv", False):
            continue
        return False

    return True


def env_guard_entrypoints() -> tuple:
    """当前由本守卫实际覆盖的入口键名。"""
    return tuple(
        key
        for owner, attr, key, fake in _dotenv_entry_state()
        if getattr(owner, attr, None) is fake
    )


def _restore_dotenv_entry(owner, attr, key) -> bool:
    """把单个入口恢复为安装前记录的对象；未记录则删除该属性。"""
    original = _DOTENV_ORIGINALS.pop(key, None)
    if original is not None:
        setattr(owner, attr, original)
        return True
    try:
        delattr(owner, attr)
    except AttributeError:  # pragma: no cover
        return False
    return True


def install_env_guard() -> bool:
    """安装 dotenv 隔离与假密钥。

    幂等且可自愈：不依赖布尔闩锁，而是逐入口比对当前对象 ——
    已被本守卫替换则跳过；被第三方假实现覆盖（``_wb_fake_dotenv``）则让位；
    部分失败时回滚已替换的入口。只有确实由本守卫完成替换才返回 True。
    """
    global _ENV_INSTALLED, _ENV_SYNTHETIC_MODULE

    applied = []
    created_synthetic = False

    try:
        existing = sys.modules.get("dotenv")
        if existing is not None and getattr(
            existing, _FAKE_DOTENV_MODULE_FLAG, False
        ):
            # 复用本守卫上一次伪造的模块，避免把假实现误当原始值快照
            dotenv_module = existing
            ENV_STATE["dotenv_module"] = "synthetic"
        else:
            try:
                import dotenv as dotenv_module
            except Exception:
                dotenv_module = None

            if dotenv_module is None:
                if _ENV_SYNTHETIC_MODULE is None:
                    fake_module = types.ModuleType("dotenv")
                    for name, fake in _DOTENV_FAKES:
                        setattr(fake_module, name, fake)
                    fake_module.__doc__ = (
                        "离线测试用假 dotenv 模块（阶段 1-R2.1）"
                    )
                    setattr(fake_module, _FAKE_DOTENV_MODULE_FLAG, True)
                    _ENV_SYNTHETIC_MODULE = fake_module
                    created_synthetic = True
                dotenv_module = _ENV_SYNTHETIC_MODULE
                sys.modules["dotenv"] = dotenv_module
                ENV_STATE["dotenv_module"] = "synthetic"
            else:
                ENV_STATE["dotenv_module"] = "real-patched"
                for name, fake in _DOTENV_FAKES:
                    current = getattr(dotenv_module, name, None)
                    if current is fake:
                        continue
                    if current is None:
                        continue
                    if getattr(current, "_wb_fake_dotenv", False):
                        # 已有他方假实现在位，不盲目覆盖
                        continue
                    setattr(dotenv_module, name, fake)
                    _DOTENV_ORIGINALS[name] = current
                    applied.append((dotenv_module, name, name, fake))

        # `from dotenv.main import load_dotenv` 也必须拿到假实现
        dotenv_main = sys.modules.get("dotenv.main")
        if dotenv_main is not None and not getattr(
            dotenv_main, _FAKE_DOTENV_MODULE_FLAG, False
        ):
            for name, fake in _DOTENV_FAKES:
                current = getattr(dotenv_main, name, None)
                if current is fake:
                    continue
                if current is None:
                    continue
                if getattr(current, "_wb_fake_dotenv", False):
                    continue
                key = f"main.{name}"
                setattr(dotenv_main, name, fake)
                _DOTENV_ORIGINALS[key] = current
                applied.append((dotenv_main, name, key, fake))

        # 环境变量：首次安装时快照原始值，重复安装不得把假值当成原始值
        if ENV_STATE["had_real_key_before"] is None:
            ENV_STATE["had_real_key_before"] = os.environ.get(
                "DEEPSEEK_API_KEY"
            ) not in (None, "", FAKE_API_KEY)

        for key in _PROTECTED_ENV_KEYS:
            if key not in _ENV_ORIGINALS:
                _ENV_ORIGINALS[key] = os.environ.get(key)
            os.environ[key] = FAKE_API_KEY

        ENV_STATE["run_real_llm_requested"] = os.environ.get("RUN_REAL_LLM")
    except BaseException:
        for owner, attr, key, fake in reversed(applied):
            if getattr(owner, attr, None) is fake:
                _restore_dotenv_entry(owner, attr, key)
            else:
                _DOTENV_ORIGINALS.pop(key, None)
        if created_synthetic and sys.modules.get("dotenv") is _ENV_SYNTHETIC_MODULE:
            sys.modules.pop("dotenv", None)
            _ENV_SYNTHETIC_MODULE = None
        ENV_STATE["installed"] = env_guard_installed()
        ENV_STATE["fake_key_installed"] = all(
            os.environ.get(key) == FAKE_API_KEY for key in _PROTECTED_ENV_KEYS
        )
        _ENV_INSTALLED = ENV_STATE["installed"]
        raise

    _ENV_APPLIED.extend(applied)
    ENV_STATE["fake_key_installed"] = all(
        os.environ.get(key) == FAKE_API_KEY for key in _PROTECTED_ENV_KEYS
    )
    ENV_STATE["installed"] = env_guard_installed()
    _ENV_INSTALLED = ENV_STATE["installed"]
    return bool(applied) or created_synthetic


def uninstall_env_guard() -> bool:
    """还原 dotenv 入口与环境变量。

    dotenv 入口只在"当前仍是本守卫装上去的对象"时还原；已被其他 conftest
    替换的入口保持原样，其原始值记录也保留，交给真正的所有者处理。
    环境变量同样按"仍是本模块写入的假值"才还原。重复调用无副作用。
    """
    global _ENV_INSTALLED, _ENV_SYNTHETIC_MODULE

    restored = False

    remaining = []
    for owner, attr, key, fake in _ENV_APPLIED:
        if getattr(owner, attr, None) is not fake:
            # 已被外部替换：不覆盖他人 patch，记录保留
            remaining.append((owner, attr, key, fake))
            continue
        if _restore_dotenv_entry(owner, attr, key):
            restored = True

    _ENV_APPLIED[:] = remaining

    if _ENV_SYNTHETIC_MODULE is not None:
        if sys.modules.get("dotenv") is _ENV_SYNTHETIC_MODULE:
            sys.modules.pop("dotenv", None)
            restored = True
        _ENV_SYNTHETIC_MODULE = None

    for key, original in list(_ENV_ORIGINALS.items()):
        if os.environ.get(key) != FAKE_API_KEY:
            # 当前值不是本守卫写入的假值：不覆盖，记录保留
            continue
        if original is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original
        _ENV_ORIGINALS.pop(key, None)
        restored = True

    ENV_STATE["fake_key_installed"] = all(
        os.environ.get(key) == FAKE_API_KEY for key in _PROTECTED_ENV_KEYS
    )
    ENV_STATE["installed"] = env_guard_installed()
    _ENV_INSTALLED = ENV_STATE["installed"]
    return restored


def dotenv_is_fake() -> bool:
    """当前 `dotenv.load_dotenv` 是否为假实现。"""
    module = sys.modules.get("dotenv")
    return bool(getattr(getattr(module, "load_dotenv", None), "_wb_fake_dotenv", False))


def guards_report() -> dict:
    """守卫状态汇总（只含布尔与计数，不含任何密钥或环境变量值）。"""
    owner = network_guard_owner()
    return {
        "network_installed": owner != "none",
        "network_owner": owner,
        "network_blocked": len(NETWORK_STATE["non_loopback"])
        + len(NETWORK_STATE["dns_blocked"])
        + len(NETWORK_STATE["udp_blocked"])
        + len(NETWORK_STATE["http_blocked"]),
        "http_installed": http_guard_installed(),
        "http_blocked": len(NETWORK_STATE["http_blocked"]),
        "sdk_installed": bool(SDK_STATE["installed"]),
        "sdk_blocked": SDK_STATE["blocked"],
        "dotenv_is_fake": dotenv_is_fake(),
        "api_key_is_fake": os.environ.get("DEEPSEEK_API_KEY") == FAKE_API_KEY,
    }


# ============================================================
# 10. 供守卫自测文件访问的稳定入口
# ============================================================

sys.modules.setdefault("_wb_db_guard", sys.modules[__name__])

# 导入即安装：确保在测试模块导入之前生效，而不是等到 fixture 执行。
# 顺序：先数据库，再网络/HTTP/SDK/dotenv —— 后者都带幂等标记，重复加载安全。
install_guard()
install_network_guard()
install_http_guard()
install_sdk_guard()
install_env_guard()
