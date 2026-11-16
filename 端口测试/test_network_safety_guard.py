"""阶段 1-R2.1：全仓网络 / SDK / dotenv 守卫主动验证。

设计原则
--------
1. **只验证真实拦截，不验证配置变量**：每个拦截用例都用
   ``spy_transport`` 把「原始实现」换成探测函数；一旦守卫把调用委托给
   真实底层实现，探测函数立即抛错 —— 这证明「拦截发生在真实连接建立
   之前」，而不是先连上再判断。
2. **不产生任何真实网络流量**：目标地址全部是 TEST-NET / 文档保留地址
   （192.0.2.1、2001:db8::1、example.invalid），凭据全部是假值。
3. **不读取正式 .env、不读取正式数据库**：只断言假实现生效与拦截生效。
4. 每个目录的守卫覆盖用**子进程**验证：只有独立进程才能真实模拟
   「进入目录后直接调用 pytest」时 conftest 的加载环境。
"""

from __future__ import annotations

import json
import os
import socket
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROOT_CONFTEST = PROJECT_ROOT / "conftest.py"
GUARD_MODULE_NAME = "_wb_db_guard"

# 空字符串代表项目根目录本身
TEST_DIRECTORIES = (
    "",
    "自动测试脚本",
    "端口测试",
    "记忆测试",
    "RAG测试",
    "测试多模态",
)


def _load_guard():
    """取根级守卫模块；缺失时响亮失败，绝不静默跳过。"""
    module = sys.modules.get(GUARD_MODULE_NAME)
    if module is None:
        raise RuntimeError(
            f"找不到根级守卫模块 {GUARD_MODULE_NAME}："
            "根 conftest.py 未加载，测试安全基线不成立。"
        )
    return module


@pytest.fixture
def guard():
    return _load_guard()


# ------------------------------------------------------------
# 阶段 1-R2.2-CLOSE（E-1）：拦截证据分级
#
# 旧实现只在 ``_NETWORK_ORIGINALS`` 里找 key。子目录守卫接管时根守卫已按
# 所有权规则让位、字典为空 —— 循环一次都不执行，用例就"空转通过"。
# 现在先确认守卫在位、再按**运行时对象**确认入口真有守卫标记，最后才决定
# 能否植入探测器；拿不到委托链时明确降级为可观测证据。
# ------------------------------------------------------------

EVIDENCE_TRANSPORT_SPY = "transport-spy"
EVIDENCE_OBSERVABLE_ONLY = "observable-only"

# (展示名, ``_NETWORK_ORIGINALS`` 中的 key)
NETWORK_ENTRYPOINTS = (
    ("socket.socket.connect", "connect"),
    ("socket.socket.connect_ex", "connect_ex"),
    ("socket.create_connection", "create_connection"),
    ("socket.getaddrinfo", "getaddrinfo"),
    ("socket.socket.sendto", "sendto"),
    ("socket.socket.sendmsg", "sendmsg"),
)


def _network_entrypoint_targets() -> dict:
    """运行时读取六个出口的当前对象（不能在模块导入时缓存）。"""
    return {
        "connect": getattr(socket.socket, "connect", None),
        "connect_ex": getattr(socket.socket, "connect_ex", None),
        "create_connection": getattr(socket, "create_connection", None),
        "getaddrinfo": getattr(socket, "getaddrinfo", None),
        "sendto": getattr(socket.socket, "sendto", None),
        "sendmsg": getattr(socket.socket, "sendmsg", None),
    }


class _SpyLog(list):
    """调用记录；保持 list 语义，兼容 ``== []`` 与真值判断。"""

    def __init__(self, evidence, targets, owner):
        super().__init__()
        self.evidence = evidence
        self.targets = tuple(targets)
        self.owner = owner

    def __repr__(self):  # pragma: no cover - 仅在失败信息里出现
        return (
            f"<spy_transport calls={list.__len__(self)} evidence={self.evidence} "
            f"owner={self.owner} targets={self.targets}>"
        )


@pytest.fixture
def spy_transport(guard, monkeypatch):
    """把当前**有效**守卫入口的底层实现替换为探测函数。

    守卫若在拦截路径上仍调用了真实实现，探测函数会立刻抛
    ``AssertionError``，用例随之失败 —— 这是「未委托真实实现」的直接证据。

    阶段 1-R2.2-CLOSE：不再假设 ``_NETWORK_ORIGINALS`` 非空。若他方守卫
    （``自动测试脚本`` 的 ``_wb_guard``）接管，根守卫已让位、字典为空，
    此时只能提供"拦截确实发生"的可观测证据，证据等级如实标为
    ``observable-only``，绝不假装证明了"未委托"。
    """
    owner = guard.network_guard_owner()
    assert owner in ("root", "subdir"), (
        f"网络守卫不在位（owner={owner!r}）：探测前提不成立，已 fail-closed 中止"
    )

    current_targets = _network_entrypoint_targets()
    live = [
        (label, key)
        for label, key in NETWORK_ENTRYPOINTS
        if getattr(current_targets.get(key), "_wb_net_guard", False)
        or getattr(current_targets.get(key), "_wb_guard", False)
    ]
    assert live, (
        "六个网络出口上都找不到守卫标记，无法证明存在有效拦截入口："
        f"owner={owner!r}"
    )

    log = _SpyLog(EVIDENCE_OBSERVABLE_ONLY, (), owner)

    def _spy(*args, **kwargs):
        log.append((args, kwargs))
        raise AssertionError("守卫把连接委托给了真实底层实现，拦截失效")

    planted = []

    for _label, key in live:
        if key not in guard._NETWORK_ORIGINALS:
            continue
        # 只有本守卫当前占据该入口时，`_NETWORK_ORIGINALS[key]` 才是它的
        # 委托目标。他方守卫接管后根守卫已让位，此时植入的探测器永远
        # 不会被调用 —— 那是假证据，必须排除。
        if not getattr(current_targets.get(key), "_wb_net_guard", False):
            continue
        monkeypatch.setitem(guard._NETWORK_ORIGINALS, key, _spy)
        planted.append(key)

    log.targets = tuple(planted)
    log.evidence = EVIDENCE_TRANSPORT_SPY if planted else EVIDENCE_OBSERVABLE_ONLY
    return log


def assert_no_network_delegation(log):
    """断言未发生底层委托；证据不足时明确降级，而不是静默通过。"""
    if log.evidence == EVIDENCE_TRANSPORT_SPY:
        assert log == [], "守卫把连接委托给了真实底层实现"
        return

    assert log.evidence == EVIDENCE_OBSERVABLE_ONLY
    assert log.owner == "subdir", (
        f"根守卫在位本应可植入探测器，却只得到 {log.evidence!r}"
        f"（owner={log.owner!r}）"
    )


# ------------------------------------------------------------
# 阶段 1-R2.2：跨目录时的「多所有者」现实
#
# 从项目根跑全仓时，`自动测试脚本/conftest.py` 会用自己的 STATE 统计型
# 守卫接管网络层（标记 `_wb_guard`）。它抛的是 RuntimeError + 统一消息，
# 而不是根守卫的 BlockedNetworkAccessError —— 两者都是有效拦截。
#
# 因此下文的断言改为「被任一守卫拦截」+「未委托真实实现」，不再锁定
# 具体实现类。安全性等价：拦截仍然必须发生在真实传输之前。
# ------------------------------------------------------------

NETWORK_GUARD_MARKERS = ("测试网络保护", "NETWORK_BLOCKED")
SDK_GUARD_MARKERS = ("测试 SDK 保护", "离线测试禁止发起真实 SDK 请求")


def _assert_network_blocked(excinfo):
    """确认异常来自某个网络守卫的拦截，而不是其他偶然错误。"""
    message = str(excinfo.value)
    assert any(marker in message for marker in NETWORK_GUARD_MARKERS), (
        f"未被任一网络守卫拦截：{message!r}"
    )
    return excinfo.value


def _assert_sdk_blocked(excinfo):
    """确认异常来自某个 SDK 守卫的拦截。"""
    message = str(excinfo.value)
    assert any(marker in message for marker in SDK_GUARD_MARKERS), (
        f"未被任一 SDK 守卫拦截：{message!r}"
    )
    return excinfo.value


def _assert_kind(exc, expected: str):
    """根守卫会给出精确拦截类别；专用守卫只有统一消息，存在时才校验。"""
    kind = getattr(exc, "kind", None)
    if kind is not None:
        assert kind == expected, f"拦截类别不符：{kind!r} != {expected!r}"


# ============================================================
# 1. 安装状态（根目录）
# ============================================================


def test_T01_network_guard_covers_every_entrypoint(guard):
    """网络守卫必须在位，且覆盖全部网络出口。

    阶段 1-R2.2：不再锁定「由哪个模块安装」。从项目根跑全仓时，
    `自动测试脚本/conftest.py` 会主动让位自己的 STATE 统计型守卫
    （标记 ``_wb_guard``），此时根守卫虽已让位，网络依然是隔离的 ——
    「谁是所有者」只是实现细节，「五个出口是否都被守住」才是安全性质。
    """
    owner = guard.network_guard_owner()
    assert owner in ("root", "subdir"), f"没有任何网络守卫在位：owner={owner}"

    for label, target in (
        ("socket.socket.connect", socket.socket.connect),
        ("socket.socket.connect_ex", socket.socket.connect_ex),
        ("socket.create_connection", socket.create_connection),
        ("socket.getaddrinfo", socket.getaddrinfo),
        ("socket.socket.sendto", socket.socket.sendto),
    ):
        assert getattr(target, "_wb_net_guard", False) or getattr(
            target, "_wb_guard", False
        ), f"{label} 上没有任何网络守卫"


def test_T01_sdk_and_dotenv_guard_are_installed(guard):
    """SDK 请求守卫与 dotenv 假实现必须已安装。"""
    assert guard.SDK_STATE["installed"] is True
    assert guard.SDK_STATE["openai_available"] is True
    assert guard.dotenv_is_fake() is True


# ============================================================
# 2. 逐目录覆盖（子进程验证，模拟「进入目录后直接调用 pytest」）
# ============================================================

_PROBE_SCRIPT = r'''
import importlib.util, json, socket, sys
from pathlib import Path

conftest_path = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("_probe_conftest", conftest_path)
module = importlib.util.module_from_spec(spec)
sys.modules["_probe_conftest"] = module
spec.loader.exec_module(module)

guard = sys.modules.get("_wb_db_guard") or module
connect = socket.socket.connect
payload = {
    "network": bool(
        getattr(connect, "_wb_net_guard", False)
        or getattr(connect, "_wb_guard", False)
    ),
    "http": bool(guard.http_guard_installed()),
    "sdk": bool(guard.SDK_STATE["installed"]),
    "dotenv_fake": bool(guard.dotenv_is_fake()),
}
print("PROBE_RESULT=" + json.dumps(payload))
'''


def _probe_directory(directory: str) -> dict:
    conftest_path = PROJECT_ROOT / directory / "conftest.py" if directory else ROOT_CONFTEST
    assert conftest_path.is_file(), f"缺少 conftest：{conftest_path}"

    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    completed = subprocess.run(
        [sys.executable, "-B", "-c", _PROBE_SCRIPT, str(conftest_path)],
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )

    marker = "PROBE_RESULT="
    for line in completed.stdout.splitlines():
        if line.startswith(marker):
            return json.loads(line[len(marker):])

    raise AssertionError(
        "子进程未返回探测结果："
        f"directory={directory!r} rc={completed.returncode} "
        f"stdout_tail={completed.stdout.strip()[-300:]!r} "
        f"stderr_tail={completed.stderr.strip()[-300:]!r}"
    )


@pytest.mark.parametrize("directory", TEST_DIRECTORIES)
def test_T02_each_directory_installs_guards(directory):
    """六个入口（根 + 五个测试目录）在独立进程中都必须装上守卫。"""
    result = _probe_directory(directory)
    assert result["network"] is True, f"{directory or '项目根'} 未安装网络守卫"
    assert result["http"] is True, f"{directory or '项目根'} 未安装 HTTP 传输层守卫"
    assert result["sdk"] is True, f"{directory or '项目根'} 未安装 SDK 守卫"
    assert result["dotenv_fake"] is True, f"{directory or '项目根'} 的 dotenv 不是假实现"


# ============================================================
# 3. 拦截：非回环 IPv4 / IPv6 / DNS / UDP
# ============================================================


def test_T03_non_loopback_ipv4_is_blocked_before_connect(guard, spy_transport):
    with pytest.raises(RuntimeError) as excinfo:
        socket.create_connection(("192.0.2.1", 80), timeout=1)

    _assert_kind(_assert_network_blocked(excinfo), "socket.create_connection")
    assert_no_network_delegation(spy_transport)


def test_T04_non_loopback_ipv6_is_blocked_before_connect(guard, spy_transport):
    with pytest.raises(RuntimeError) as excinfo:
        socket.create_connection(("2001:db8::1", 80), timeout=1)

    _assert_kind(_assert_network_blocked(excinfo), "socket.create_connection")
    assert_no_network_delegation(spy_transport)


def test_T05_dns_resolution_is_blocked(guard, spy_transport):
    with pytest.raises(RuntimeError) as excinfo:
        socket.getaddrinfo("example.invalid", 443)

    _assert_kind(_assert_network_blocked(excinfo), "socket.getaddrinfo")
    assert_no_network_delegation(spy_transport)


def test_T06_udp_sendto_is_blocked(guard, spy_transport):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        with pytest.raises(RuntimeError) as excinfo:
            sock.sendto(b"probe", ("192.0.2.1", 53))
        _assert_kind(_assert_network_blocked(excinfo), "socket.sendto")
        assert_no_network_delegation(spy_transport)
    finally:
        sock.close()


def test_T06_socket_connect_on_instance_is_blocked(guard, spy_transport):
    """直接使用 ``socket.socket()`` 实例也不能绕过守卫。"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    try:
        with pytest.raises(RuntimeError) as excinfo:
            sock.connect(("192.0.2.1", 80))
        _assert_network_blocked(excinfo)
        assert_no_network_delegation(spy_transport)
    finally:
        sock.close()


# ============================================================
# 4. 回环必须放行（本机 FastAPI / socket 测试不被误伤）
# ============================================================


def test_T07_loopback_connection_still_works():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]

    client = None
    accepted = None
    try:
        client = socket.create_connection(("127.0.0.1", port), timeout=5)
        accepted, _ = server.accept()
        assert client.getpeername()[0] == "127.0.0.1"
    finally:
        if accepted is not None:
            accepted.close()
        if client is not None:
            client.close()
        server.close()


def test_T07_loopback_hostnames_are_recognised(guard):
    assert guard._is_loopback_host("127.0.0.1") is True
    assert guard._is_loopback_host("127.8.9.10") is True
    assert guard._is_loopback_host("::1") is True
    assert guard._is_loopback_host("[::1]") is True
    assert guard._is_loopback_host("localhost") is True
    assert guard._is_loopback_host("LOCALHOST") is True
    # 非回环必须判为 False
    assert guard._is_loopback_host("192.0.2.1") is False
    assert guard._is_loopback_host("2001:db8::1") is False
    assert guard._is_loopback_host("example.invalid") is False
    assert guard._is_loopback_host("0.0.0.0") is False
    assert guard._is_loopback_host(None) is False


# ============================================================
# 5. 真实 SDK 拦截 / Fake SDK 保留
# ============================================================


def test_T08_real_sdk_request_is_blocked_before_http(guard):
    """真实客户端可以构造，但发起请求必须在 HTTP 之前被拒绝。"""
    import openai

    before_network = len(guard.NETWORK_STATE["non_loopback"])
    before_sdk = guard.SDK_STATE["blocked"]

    client = openai.OpenAI(
        api_key="offline-test-key",
        base_url="https://api.deepseek.com",
        timeout=5.0,
    )

    with pytest.raises(RuntimeError) as excinfo:
        client.chat.completions.create(
            model="deepseek-flash",
            messages=[{"role": "user", "content": "guard probe"}],
        )

    blocked = _assert_sdk_blocked(excinfo)
    # 根守卫会精确计数；专用 SDK 空壳不写本模块计数，只在根守卫时校验
    if isinstance(blocked, guard.BlockedSDKRequestError):
        assert guard.SDK_STATE["blocked"] == before_sdk + 1

    # 未触达 socket 层：说明拦截发生在任何网络请求之前
    assert len(guard.NETWORK_STATE["non_loopback"]) == before_network


def test_T09_fake_sdk_still_works():
    """Fake 客户端不受守卫影响：可正常构造与返回。"""

    class _FakeCompletions:
        def create(self, **kwargs):
            return {"choices": [{"message": {"content": "fake"}}], "kwargs": kwargs}

    class _FakeClient:
        def __init__(self):
            self.chat = type("Chat", (), {"completions": _FakeCompletions()})()

    client = _FakeClient()
    result = client.chat.completions.create(model="fake-model", messages=[])
    assert result["choices"][0]["message"]["content"] == "fake"


# ============================================================
# 6. dotenv：收集阶段不得加载正式 .env
# ============================================================


def test_T10_dotenv_entrypoints_are_fake(guard):
    import dotenv

    assert getattr(dotenv.load_dotenv, "_wb_fake_dotenv", False) is True
    assert getattr(dotenv.dotenv_values, "_wb_fake_dotenv", False) is True
    assert getattr(dotenv.find_dotenv, "_wb_fake_dotenv", False) is True

    env_path = str(PROJECT_ROOT / ".env")
    assert dotenv.load_dotenv(env_path) is False
    assert dotenv.dotenv_values(env_path) == {}
    assert dotenv.find_dotenv() == ""


def test_T10_main_binding_is_fake(guard):
    """`main.py` 模块级 `load_dotenv()` 拿到的必须是假实现。"""
    import main

    assert getattr(main.load_dotenv, "_wb_fake_dotenv", False) is True


def test_T10_api_key_in_process_is_fake(guard):
    assert os.environ.get("DEEPSEEK_API_KEY") == guard.FAKE_API_KEY
    assert guard.ENV_STATE["fake_key_installed"] is True


# ============================================================
# 7. RUN_REAL_LLM=1 不得绕过守卫
# ============================================================


def test_T11_run_real_llm_cannot_bypass_guards(guard, monkeypatch):
    monkeypatch.setenv("RUN_REAL_LLM", "1")

    import openai

    client = openai.OpenAI(
        api_key="offline-test-key",
        base_url="https://api.deepseek.com",
        timeout=5.0,
    )

    with pytest.raises(RuntimeError) as sdk_excinfo:
        client.chat.completions.create(
            model="deepseek-flash",
            messages=[{"role": "user", "content": "RUN_REAL_LLM probe"}],
        )
    _assert_sdk_blocked(sdk_excinfo)

    # 即使开关打开，网络层也不可能出网
    with pytest.raises(RuntimeError) as net_excinfo:
        socket.create_connection(("api.deepseek.com", 443), timeout=1)
    _assert_network_blocked(net_excinfo)


# ============================================================
# 8. 数据库守卫回归
# ============================================================


def test_T12_database_guard_still_blocks_official_databases(guard):
    for relative in ("memory.db", "data/business.db"):
        target = PROJECT_ROOT / relative
        with pytest.raises(guard.ProtectedDatabaseAccessError):
            sqlite3.connect(str(target))


def test_T13_main_get_memory_is_isolated(patched_main_memory, tmp_path):
    import main

    memory = patched_main_memory
    resolved = Path(memory.db_path).resolve()
    assert resolved.is_relative_to(Path(tmp_path).resolve())
    assert main.get_memory.cache_info().currsize == 1


# ============================================================
# 9. 安装幂等 / 不与专用守卫互相叠加
#
# 背景：本轮实施中发现过一个真实缺陷 —— 根守卫若在子目录专用守卫之上
# 再包一层，双方的"原始实现"会互相指向对方，形成无限递归。以下两条
# 用例把该约束固定下来。
# ============================================================


def test_T14_install_is_idempotent_and_never_self_delegates(guard):
    """重复安装必须幂等，且保存的原始实现绝不能是任何守卫自身。

    阶段 1-R2.2：不再断言「安装后一定带根守卫标记」。在专用守卫接管的
    场景下根守卫主动让位（返回 False 且不改动入口）是正确的终态，
    因此改为断言「未改变已生效的入口」+「隔离仍然有效」。
    """
    before = socket.socket.connect

    assert guard.install_network_guard() is False, "重复安装必须返回 False"
    assert socket.socket.connect is before, "重复安装改变了已生效的入口对象"

    # 保存的原始实现不能是守卫自身，否则会自我委托或无限递归
    for name, original in guard._NETWORK_ORIGINALS.items():
        assert not getattr(original, "_wb_net_guard", False), f"{name} 的原始实现是根守卫自身"
        assert not getattr(original, "_wb_guard", False), f"{name} 的原始实现是其他守卫"

    # 幂等调用之后隔离仍然有效（行为断言，不依赖所有者）
    with pytest.raises(RuntimeError) as excinfo:
        socket.create_connection(("192.0.2.1", 80), timeout=1)

    message = str(excinfo.value)
    assert "NETWORK_BLOCKED" in message or "测试网络保护" in message, (
        f"幂等安装后非回环连接未被任一守卫拦截：{message!r}"
    )


def test_T15_foreign_guard_is_never_stacked(guard, monkeypatch):
    """检测到专用守卫（`_wb_guard`）时必须让位，不得叠加、不得自我委托。"""

    def _foreign(self, address, *args, **kwargs):
        raise RuntimeError("NETWORK_BLOCKED_FOREIGN")

    _foreign._wb_guard = True

    monkeypatch.setattr(socket.socket, "connect", _foreign)

    try:
        assert guard.install_network_guard() is False
        assert socket.socket.connect is _foreign, "根守卫覆盖了专用守卫"

        # 让位后不得把调用委托回自己：foreign 应当直接处理并抛它自己的错误
        with pytest.raises(RuntimeError) as excinfo:
            socket.socket.connect(object(), ("192.0.2.1", 80))

        assert "NETWORK_BLOCKED_FOREIGN" in str(excinfo.value), (
            "根守卫把调用委托回了自己，形成自我委托链"
        )
    finally:
        monkeypatch.undo()
        # 恢复本模块守卫的安装状态标记
        guard.install_network_guard()

    # 阶段 1-R2.2：恢复后只要求「仍有守卫在位」，不限定必须是根守卫
    assert guard.network_guard_owner() in ("root", "subdir"), (
        f"恢复后没有任何网络守卫在位：owner={guard.network_guard_owner()}"
    )


# ============================================================
# 10. 安装/卸载的所有权语义（阶段 1-R2.2-CLOSE · CLOSE-01 / CLOSE-03）
#
# 核心约束：守卫的安装状态必须**以被 patch 对象的当前状态为准**。
# 布尔闩锁只能说明"曾经装过"，无法反映"现在还在不在"，也无法在他方
# patch 接管后保持正确 —— 以下用例把这两类失效固定在测试里。
# ============================================================

_SUBDIRECTORIES = ("端口测试", "记忆测试", "RAG测试", "测试多模态")

_INSTALL_SEMANTICS_SCRIPT = r'''
import builtins, importlib.util, json, sys
from pathlib import Path

root = Path(sys.argv[1])
mode = sys.argv[2]
blocked = [item for item in sys.argv[3].split(",") if item]

class _Blocker:
    def find_spec(self, name, path=None, target=None):
        for prefix in blocked:
            if name == prefix or name.startswith(prefix + "."):
                raise ImportError("blocked by probe: " + name)
        return None

if blocked:
    sys.meta_path.insert(0, _Blocker())

spec = importlib.util.spec_from_file_location("_wb_db_guard", root / "conftest.py")
module = importlib.util.module_from_spec(spec)
sys.modules["_wb_db_guard"] = module
spec.loader.exec_module(module)

payload = {}

if mode == "unavailable":
    payload = {
        "http_installed": bool(module.http_guard_installed()),
        "http_modules": list(module.http_guard_modules()),
        "sdk_installed": bool(module.sdk_guard_installed()),
        "sdk_available": module.SDK_STATE["openai_available"],
        "env_installed": bool(module.env_guard_installed()),
        "env_module": module.ENV_STATE["dotenv_module"],
        "reinstall_http": module.install_http_guard(),
        "reinstall_sdk": module.install_sdk_guard(),
    }
elif mode == "rollback":
    import httpx

    real_setattr = builtins.setattr
    module.uninstall_http_guard()
    sync_original = httpx.HTTPTransport.handle_request
    async_original = httpx.AsyncHTTPTransport.handle_async_request

    def failing_setattr(target, name, value):
        if target is httpx.AsyncHTTPTransport and name == "handle_async_request":
            raise RuntimeError("injected setattr failure")
        return real_setattr(target, name, value)

    builtins.setattr = failing_setattr
    try:
        try:
            module.install_http_guard()
            raised = False
        except RuntimeError:
            raised = True
    finally:
        builtins.setattr = real_setattr

    payload = {
        "raised": raised,
        "sync_rolled_back": httpx.HTTPTransport.handle_request is sync_original,
        "async_rolled_back": httpx.AsyncHTTPTransport.handle_async_request is async_original,
        "installed": bool(module.http_guard_installed()),
        "originals": sorted(module._HTTP_ORIGINALS),
        "reinstall_ok": bool(module.install_http_guard())
        and bool(module.http_guard_installed()),
    }

print("SEMANTICS=" + json.dumps(payload))
'''

_LIFECYCLE_CONFIGURE_SCRIPT = r'''
import importlib.util, json, os, sys
from pathlib import Path

root = Path(sys.argv[1])
subdir = root / sys.argv[2]

spec = importlib.util.spec_from_file_location("_wb_db_guard", root / "conftest.py")
module = importlib.util.module_from_spec(spec)
sys.modules["_wb_db_guard"] = module
spec.loader.exec_module(module)

# 模拟"上一轮会话已结束"：五类守卫全部卸载
module.uninstall_http_guard(); module.uninstall_network_guard()
module.uninstall_sdk_guard(); module.uninstall_env_guard(); module.uninstall_guard()

before = {
    "owner": module.network_guard_owner(),
    "http": bool(module.http_guard_installed()),
    "sdk": bool(module.sdk_guard_installed()),
    "env": bool(module.env_guard_installed()),
}

os.chdir(subdir)
import pytest
exit_code = pytest.main(["--collect-only", "-q", "-p", "no:cacheprovider"])

after = {
    "owner": module.network_guard_owner(),
    "http": bool(module.http_guard_installed()),
    "sdk": bool(module.sdk_guard_installed()),
    "env": bool(module.env_guard_installed()),
}

print("LIFECYCLE=" + json.dumps({"before": before, "after": after, "exit": int(exit_code)}))
'''

_ENV_UNINSTALL_SCRIPT = r'''
import importlib.util, json, os, sys
from pathlib import Path

root = Path(sys.argv[1])
mode = sys.argv[2]

spec = importlib.util.spec_from_file_location("_wb_db_guard", root / "conftest.py")
module = importlib.util.module_from_spec(spec)
sys.modules["_wb_db_guard"] = module
spec.loader.exec_module(module)

import dotenv

# 把"原始值"钉成假值：卸载时绝不会把真实密钥写回进程环境
for key in module._PROTECTED_ENV_KEYS:
    module._ENV_ORIGINALS[key] = module.FAKE_API_KEY
os.environ["DEEPSEEK_API_KEY"] = module.FAKE_API_KEY

payload = {}

if mode == "restore":
    originals = dict(module._DOTENV_ORIGINALS)
    applied = [(owner, attr, key, fake) for owner, attr, key, fake in module._ENV_APPLIED]
    payload["applied_count"] = len(applied)
    payload["recorded_count"] = len(originals)
    payload["uninstall_ret"] = module.uninstall_env_guard()
    payload["installed_after"] = bool(module.env_guard_installed())
    payload["all_restored"] = all(
        getattr(owner, attr, None) is originals[key]
        for owner, attr, key, _fake in applied
    )
    payload["no_fake_left"] = not any(
        getattr(owner, attr, None) is fake
        for owner, attr, _key, fake in applied
    )

elif mode == "foreign":
    def _foreign(*args, **kwargs):
        return True

    dotenv.load_dotenv = _foreign
    payload["uninstall_ret"] = module.uninstall_env_guard()
    payload["foreign_preserved"] = dotenv.load_dotenv is _foreign
    payload["other_entrypoints_restored"] = not any(
        getattr(getattr(dotenv, name, None), "_wb_fake_dotenv", False)
        for name in ("dotenv_values", "find_dotenv")
    )

elif mode == "otherdir":
    # 模拟"其他目录安装了同为假实现、只是不属于本守卫"的接管场景
    def _other_load(*args, **kwargs):
        return False

    def _other_values(*args, **kwargs):
        return {}

    def _other_find(*args, **kwargs):
        return ""

    for _fn in (_other_load, _other_values, _other_find):
        _fn._wb_fake_dotenv = True

    dotenv.load_dotenv = _other_load
    dotenv.dotenv_values = _other_values
    dotenv.find_dotenv = _other_find

    payload["uninstall_ret"] = module.uninstall_env_guard()
    payload["other_dir_fakes_preserved"] = (
        dotenv.load_dotenv is _other_load
        and dotenv.dotenv_values is _other_values
        and dotenv.find_dotenv is _other_find
    )
    payload["installed_after"] = bool(module.env_guard_installed())

elif mode == "idempotent":
    payload["first"] = module.uninstall_env_guard()
    payload["second"] = module.uninstall_env_guard()
    payload["installed_after"] = bool(module.env_guard_installed())

elif mode == "reinstall":
    payload["uninstall_ret"] = module.uninstall_env_guard()
    payload["installed_mid"] = bool(module.env_guard_installed())
    payload["reinstall_ret"] = module.install_env_guard()
    payload["installed_after"] = bool(module.env_guard_installed())
    payload["fake_flags"] = [
        bool(getattr(getattr(dotenv, name, None), "_wb_fake_dotenv", False))
        for name in ("load_dotenv", "dotenv_values", "find_dotenv")
    ]

print("ENVSTATE=" + json.dumps(payload))
'''


def _run_probe(script: str, *args, marker: str) -> dict:
    """在独立进程执行探测脚本并解析唯一的结果行。"""
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    completed = subprocess.run(
        [sys.executable, "-B", "-c", script, str(PROJECT_ROOT), *args],
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )

    for line in completed.stdout.splitlines():
        if line.startswith(marker):
            return json.loads(line[len(marker) :])

    raise AssertionError(
        "子进程未输出结果："
        f"returncode={completed.returncode} stdout={completed.stdout!r} "
        f"stderr={completed.stderr!r}"
    )


@pytest.fixture
def restore_guards(guard):
    """用例改动守卫状态后无条件恢复"全部在位"，避免污染后续用例。"""
    yield
    guard.install_guard()
    guard.install_network_guard()
    guard.install_http_guard()
    guard.install_sdk_guard()
    guard.install_env_guard()


def test_T16_guard_presence_is_identity_based(guard):
    """三类守卫的在位判定必须来自对象身份，而不是历史布尔位。"""
    assert guard.http_guard_installed() is True
    assert guard.sdk_guard_installed() is True
    assert guard.env_guard_installed() is True

    http_entries = guard.http_guard_entrypoints()
    assert http_entries, "HTTP 守卫没有任何在位入口"
    for key in http_entries:
        assert key in guard._HTTP_GUARDS

    assert guard.sdk_guard_classes(), "SDK 守卫没有任何在位入口"
    assert guard.env_guard_entrypoints(), "dotenv 守卫没有任何在位入口"


def test_T17_install_is_idempotent_for_all_three(guard):
    """重复安装三类守卫都必须返回 False（已在位，无实际改动）。"""
    assert guard.install_http_guard() is False
    assert guard.install_sdk_guard() is False
    assert guard.install_env_guard() is False

    assert guard.http_guard_installed() is True
    assert guard.sdk_guard_installed() is True
    assert guard.env_guard_installed() is True


def test_T18_replaced_http_entrypoint_is_detected(guard, monkeypatch, restore_guards):
    """入口被外部替换后必须立刻判定为"不在位"。"""
    import httpx

    saved_originals = dict(guard._HTTP_ORIGINALS)
    assert guard.http_guard_installed() is True

    def _foreign(self, request):  # pragma: no cover - 不应被调用
        return None

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _foreign)

    try:
        assert guard.http_guard_installed() is False, "被替换后仍报告为在位"
        assert "httpx:handle_request" not in guard.http_guard_entrypoints()
    finally:
        # 必须先 undo 再恢复登记：否则 install 会把 _foreign 记成"原始实现"
        monkeypatch.undo()
        guard._HTTP_ORIGINALS.clear()
        guard._HTTP_ORIGINALS.update(saved_originals)
        guard.install_http_guard()


def test_T19_reinstall_repairs_replaced_entrypoint(guard, monkeypatch, restore_guards):
    """外部替换后再次安装必须自我修复（而不是被闩锁挡住）。"""
    import httpx

    key = "httpx:handle_request"
    saved_originals = dict(guard._HTTP_ORIGINALS)

    def _foreign(self, request):  # pragma: no cover - 不应被调用
        return None

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _foreign)

    try:
        assert guard.install_http_guard() is True, "再次安装没有修复被替换的入口"
        assert guard.http_guard_installed() is True
        assert httpx.HTTPTransport.handle_request is guard._HTTP_GUARDS[key]
    finally:
        monkeypatch.undo()
        guard._HTTP_ORIGINALS.clear()
        guard._HTTP_ORIGINALS.update(saved_originals)
        guard.install_http_guard()


def test_T20_foreign_marked_http_entrypoint_is_not_overwritten(
    guard, monkeypatch, restore_guards
):
    """带他方标记的入口不得被覆盖，且必须被识别为"他方所有"。"""
    import httpx

    key = "httpx:handle_request"
    saved_originals = dict(guard._HTTP_ORIGINALS)

    def _foreign(self, request):  # pragma: no cover - 不应被调用
        raise AssertionError("他方守卫不应被本模块覆盖")

    _foreign._wb_http_guard = True
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _foreign)

    try:
        assert guard.install_http_guard() is False, "覆盖了他方守卫"
        assert httpx.HTTPTransport.handle_request is _foreign
        assert key in guard.http_foreign_entrypoints()
        assert key not in guard.http_guard_entrypoints()
        assert guard.http_guard_installed() is False
    finally:
        monkeypatch.undo()
        guard._HTTP_ORIGINALS.clear()
        guard._HTTP_ORIGINALS.update(saved_originals)
        guard.install_http_guard()


def test_T21_uninstall_then_reinstall_cycle(guard, restore_guards):
    """卸载后三类守卫都应判定为不在位，且可以重新安装。"""
    saved_http = dict(guard._HTTP_ORIGINALS)
    saved_sdk = dict(guard._SDK_ORIGINALS)

    try:
        assert guard.uninstall_http_guard() is True
        assert guard.uninstall_sdk_guard() is True
        assert guard.http_guard_installed() is False
        assert guard.sdk_guard_installed() is False
        assert guard.http_guard_entrypoints() == ()
        assert guard.sdk_guard_classes() == ()

        assert guard.install_http_guard() is True
        assert guard.install_sdk_guard() is True
        assert guard.http_guard_installed() is True
        assert guard.sdk_guard_installed() is True
    finally:
        guard._HTTP_ORIGINALS.clear()
        guard._HTTP_ORIGINALS.update(saved_http)
        guard._SDK_ORIGINALS.clear()
        guard._SDK_ORIGINALS.update(saved_sdk)
        guard.install_http_guard()
        guard.install_sdk_guard()


def test_T22_missing_library_is_not_reported_as_installed():
    """httpx 与 httpx2 都不可用时，不得声称 HTTP 守卫安装成功。"""
    result = _run_probe(
        _INSTALL_SEMANTICS_SCRIPT, "unavailable", "httpx,httpx2", marker="SEMANTICS="
    )

    assert result["http_installed"] is False
    assert result["http_modules"] == []
    assert result["reinstall_http"] is False
    assert result["sdk_installed"] is False


def test_T22_single_library_outage_does_not_block_the_other():
    """单个库不可用不得影响另一个库的守卫安装。"""
    result = _run_probe(
        _INSTALL_SEMANTICS_SCRIPT, "unavailable", "httpx", marker="SEMANTICS="
    )

    assert result["http_modules"] == ["httpx2"]
    assert result["http_installed"] is True
    assert result["sdk_installed"] is True, "屏蔽 httpx 不应影响依赖 httpx2 的 openai"


def test_T23_partial_failure_rolls_back_without_residue():
    """部分安装失败必须回滚已装入口，且不留下指向"未替换"的脏记录。"""
    result = _run_probe(_INSTALL_SEMANTICS_SCRIPT, "rollback", "", marker="SEMANTICS=")

    assert result["raised"] is True, "注入的 setattr 失败没有被传播"
    assert result["sync_rolled_back"] is True, "同步入口未回滚"
    assert result["async_rolled_back"] is True, "异步入口被意外改动"
    assert result["installed"] is False
    assert result["originals"] == [], f"回滚后仍有脏记录：{result['originals']}"
    assert result["reinstall_ok"] is True, "故障排除后无法重新安装"


# ============================================================
# 11. 子目录 conftest 的生命周期兜底（阶段 1-R2.2-CLOSE · CLOSE-02）
# ============================================================


@pytest.mark.parametrize("directory", _SUBDIRECTORIES)
def test_T24_subdir_conftest_configure_hook_restores_guards(directory):
    """守卫被上一轮会话卸载后，子目录 conftest 的 configure 钩子必须装回来。

    `sys.modules` 里已有模块 ≠ 守卫在位；只有真正被 pytest 识别的
    `pytest_configure` 才能在新一轮会话里兜底。
    """
    result = _run_probe(
        _LIFECYCLE_CONFIGURE_SCRIPT, directory, marker="LIFECYCLE="
    )

    assert result["before"]["owner"] == "none", (
        f"前置条件不成立，卸载后仍报告守卫在位：{result['before']}"
    )
    assert result["before"]["http"] is False
    assert result["before"]["sdk"] is False
    assert result["before"]["env"] is False

    assert result["exit"] in (0, 5), f"收集阶段异常退出：{result['exit']}"
    assert result["after"]["owner"] in ("root", "subdir"), (
        f"{directory} 的 configure 钩子没有把网络守卫装回来：{result['after']}"
    )
    assert result["after"]["http"] is True, f"{directory} 的 HTTP 守卫未恢复"
    assert result["after"]["sdk"] is True, f"{directory} 的 SDK 守卫未恢复"
    assert result["after"]["env"] is True, f"{directory} 的 dotenv 守卫未恢复"


@pytest.mark.parametrize("directory", _SUBDIRECTORIES)
def test_T24_subdir_conftest_hook_is_real_and_unload_free(directory):
    """钩子必须是模块顶层真实定义，且不依赖 sys.modules、不自己卸载。"""
    text = (PROJECT_ROOT / directory / "conftest.py").read_text(encoding="utf-8")

    assert text.count("def pytest_configure(") == 1, (
        f"{directory} 的 pytest_configure 定义数量不为 1"
    )
    assert "def pytest_unconfigure(" not in text, (
        f"{directory} 不应注册卸载钩子，避免错卸他人 patch"
    )

    hook_body = text.split("def pytest_configure(", 1)[1]
    assert "sys.modules" not in hook_body, (
        f"{directory} 的 configure 钩子仍依赖 sys.modules 存在性判断"
    )
    for name in (
        "install_guard",
        "install_network_guard",
        "install_http_guard",
        "install_sdk_guard",
        "install_env_guard",
    ):
        assert name in hook_body, f"{directory} 的 configure 钩子缺少 {name}"


# ============================================================
# 12. dotenv 守卫的所有权语义（阶段 1-R2.2-CLOSE · CLOSE-03）
# ============================================================


def test_T25_env_guard_keeps_three_entrypoints_isolated(guard):
    """三个 dotenv 入口必须同时被隔离，且行为是「未找到 .env」。"""
    import dotenv

    assert getattr(dotenv.load_dotenv, "_wb_fake_dotenv", False) is True
    assert getattr(dotenv.dotenv_values, "_wb_fake_dotenv", False) is True
    assert getattr(dotenv.find_dotenv, "_wb_fake_dotenv", False) is True

    assert dotenv.load_dotenv(str(PROJECT_ROOT / ".env")) is False
    assert dotenv.dotenv_values(str(PROJECT_ROOT / ".env")) == {}
    assert dotenv.find_dotenv() == ""

    # 跨目录运行时入口可能由他方同名假实现接管（`自动测试脚本`），
    # 因此这里只要求"隔离整体生效"，不锁定所有者
    assert guard.env_guard_installed() is True


def test_T26_env_uninstall_restores_recorded_originals():
    """卸载必须按安装时记录的原始对象逐个还原，且不留假实现。"""
    result = _run_probe(_ENV_UNINSTALL_SCRIPT, "restore", marker="ENVSTATE=")

    assert result["applied_count"] > 0, "安装时必须记录本守卫实际安装的入口"
    assert result["recorded_count"] >= result["applied_count"]
    assert result["uninstall_ret"] is True
    assert result["installed_after"] is False
    assert result["all_restored"] is True, "有入口未按记录还原"
    assert result["no_fake_left"] is True, "还原后仍残留本守卫的假实现"


def test_T27_env_uninstall_does_not_override_foreign_entry():
    """入口被外部替换时，卸载不得覆盖他人的 patch。"""
    result = _run_probe(_ENV_UNINSTALL_SCRIPT, "foreign", marker="ENVSTATE=")

    assert result["uninstall_ret"] is True, "其余入口仍应被正常还原"
    assert result["foreign_preserved"] is True, "卸载覆盖了他人的 patch"
    assert result["other_entrypoints_restored"] is True, (
        "未被外部接管的入口应还原为真实实现"
    )


def test_T27_env_uninstall_does_not_override_other_directory_fakes():
    """其他目录安装了"同类假实现"时同样不得覆盖 —— 判据是对象身份，不是标记。

    跨目录运行时 `自动测试脚本` 会用自己的 `_noop_*` 接管三入口（同样带
    `_wb_fake_dotenv`）。它们"看起来像本守卫的假实现"，但对象不同，
    因此卸载必须原样保留。
    """
    result = _run_probe(_ENV_UNINSTALL_SCRIPT, "otherdir", marker="ENVSTATE=")

    assert result["uninstall_ret"] is True, "其余入口仍应被正常还原"
    assert result["other_dir_fakes_preserved"] is True, (
        "卸载把他目录安装的假实现当成本守卫的 patch 覆盖了"
    )
    # 主模块三入口仍是他目录的假实现，但本守卫在 `dotenv.main` 上的入口
    # 已按记录还原为真实实现 —— 因此整体不应再判定为"全部入口都在位"。
    assert result["installed_after"] is False, (
        "dotenv.main 已还原，整体不应继续报告为在位"
    )


def test_T28_env_uninstall_is_idempotent():
    """重复卸载必须无副作用（第二次返回 False，状态不变）。"""
    result = _run_probe(_ENV_UNINSTALL_SCRIPT, "idempotent", marker="ENVSTATE=")

    assert result["first"] is True
    assert result["second"] is False, "重复卸载不是空操作"
    assert result["installed_after"] is False


def test_T29_env_guard_can_be_reinstalled_after_uninstall():
    """卸载后必须可以重新安装（不得被历史布尔位挡住）。"""
    result = _run_probe(_ENV_UNINSTALL_SCRIPT, "reinstall", marker="ENVSTATE=")

    assert result["uninstall_ret"] is True
    assert result["installed_mid"] is False
    assert result["reinstall_ret"] is True, "卸载后无法重新安装"
    assert result["installed_after"] is True
    assert result["fake_flags"] == [True, True, True], (
        f"重装后三入口未全部隔离：{result['fake_flags']}"
    )


# ============================================================
# 13. 拦截证据等级（阶段 1-R2.2-CLOSE · CLOSE-04）
# ============================================================


def test_T30_spy_transport_reports_evidence_level(guard, spy_transport):
    """探测器的证据等级必须如实反映"能否证明未委托真实实现"。"""
    assert spy_transport.owner in ("root", "subdir")
    assert spy_transport.evidence in (
        EVIDENCE_TRANSPORT_SPY,
        EVIDENCE_OBSERVABLE_ONLY,
    )

    if spy_transport.owner == "root":
        assert spy_transport.evidence == EVIDENCE_TRANSPORT_SPY, (
            "根守卫在位时应能植入探测器，实际证据等级不足"
        )
        assert spy_transport.targets, "根守卫在位却没有任何探测点"
    else:
        # 他方守卫接管：拿不到委托链时不得假装证明了「未委托」
        assert spy_transport.evidence == EVIDENCE_OBSERVABLE_ONLY
        assert spy_transport.targets == ()

    with pytest.raises(RuntimeError):
        socket.create_connection(("192.0.2.1", 80), timeout=1)

    assert_no_network_delegation(spy_transport)
