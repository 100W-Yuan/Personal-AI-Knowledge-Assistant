# -*- coding: utf-8 -*-
"""阶段 1-R2.2：HTTP 代理绕过防线与守卫生命周期验证。

要解决的问题（P1-01）
----------------------
第七节的 socket 守卫只看「连接对端地址」。走 HTTP 代理时，对端是**本地
回环代理**（例如 127.0.0.1:52403），于是被判为回环而放行 —— 代理却能把
请求转发到任意真实外网目标。即：

    「代理 IP 是回环」  ≠  「请求目标是回环」

本轮在 ``httpx.HTTPTransport.handle_request`` /
``AsyncHTTPTransport.handle_async_request`` 上按 **真实目标主机** 判定，
不依赖代理地址是否回环。

安全前提（fail-closed）
-----------------------
本文件所有用例都先把「原始传输实现」替换成抛出型探测函数：
- 守卫正确拦截 → 抛 ``BlockedNetworkAccessError``，探测函数不被调用；
- 守卫错误放行 → 探测函数立刻抛 ``AssertionError``，**绝不会真的建连**；
- 守卫本身未安装 → ``http_spy`` fixture 在收集阶段就失败，用例不会执行。
目标地址一律是文档保留域名，代理端口一律 ``1``，不触碰 127.0.0.1:52403。
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import threading
from pathlib import Path

import httpx
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GUARD_MODULE_NAME = "_wb_db_guard"

# 文档保留域名：不解析、不建连，只用于让守卫在传输前拒绝
DOC_TARGET = "https://api.deepseek.com/chat/completions"
DOC_TARGET_HTTP = "http://api.deepseek.com/chat/completions"

# 明确的死端口：即使某处防线整体失效，也只是本机连接被拒，不出网
DEAD_PROXY = "http://127.0.0.1:1"

PROXY_ENV_NAMES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


def _load_guard():
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
# 旧实现只断言 `_HTTP_ORIGINALS` 非空 —— 那只能说明"曾经记录过原始实现"，
# 既不能证明守卫此刻在位，也会在他方守卫接管时误判为空转通过。现在改为：
#   - 先按**实际对象身份**确认 http 入口在位（本守卫 + 他方守卫都算）；
#   - 只有本守卫在位的入口才具备可植入的委托链 → ``transport-spy``；
#   - 他方守卫接管时无法植入探测器，只能提供可观测拦截证据
#     → ``observable-only``，用例必须显式区分，绝不假装证明了"未委托"。
# ------------------------------------------------------------

EVIDENCE_TRANSPORT_SPY = "transport-spy"
EVIDENCE_OBSERVABLE_ONLY = "observable-only"


class _HttpSpyLog(list):
    """调用记录；保持 list 语义，兼容 ``== []`` 与真值判断。"""

    def __init__(self, evidence, targets, ours, foreign):
        super().__init__()
        self.evidence = evidence
        self.targets = tuple(targets)
        self.owner_entrypoints = tuple(ours)
        self.foreign_entrypoints = tuple(foreign)

    def __repr__(self):  # pragma: no cover - 仅在失败信息里出现
        return (
            f"<http_spy calls={list.__len__(self)} evidence={self.evidence} "
            f"targets={self.targets} foreign={self.foreign_entrypoints}>"
        )


def _build_http_spy(guard, monkeypatch):
    """把 HTTP 传输层的「原始实现」替换为抛出型探测函数。

    这是本文件的安全底座：任何「守卫错误放行」都会撞到这个探测器，
    既证明拦截发生在真实传输之前，也保证不会产生真实网络 IO。
    """
    ours = guard.http_guard_entrypoints()
    foreign = guard.http_foreign_entrypoints()

    assert ours or foreign, (
        "HTTP 守卫未在位（本守卫与他方守卫的入口都为空）：探测前提不成立；"
        "已 fail-closed 中止，未发起任何请求"
    )

    log = _HttpSpyLog(EVIDENCE_OBSERVABLE_ONLY, (), ours, foreign)

    def _spy(self, request):
        log.append(str(getattr(request, "url", "?")))
        raise AssertionError("守卫把 HTTP 请求委托给了真实传输层，拦截失效")

    targets = []

    for key in ours:
        assert key in guard._HTTP_ORIGINALS, (
            f"{key} 由本守卫在位，但缺少原始实现记录，无法安全植入探测器"
        )
        monkeypatch.setitem(guard._HTTP_ORIGINALS, key, _spy)
        targets.append(key)

    log.targets = tuple(targets)
    log.evidence = EVIDENCE_TRANSPORT_SPY if targets else EVIDENCE_OBSERVABLE_ONLY
    return log


@pytest.fixture
def http_spy(guard, monkeypatch):
    """安全底座：任何「守卫错误放行」都会撞到探测器，绝不产生真实 IO。"""
    return _build_http_spy(guard, monkeypatch)


def assert_no_http_delegation(log):
    """断言未发生底层委托；证据不足时明确降级，而不是静默通过。"""
    if log.evidence == EVIDENCE_TRANSPORT_SPY:
        assert log == [], "守卫把 HTTP 请求交给了真实传输层"
        return

    assert log.evidence == EVIDENCE_OBSERVABLE_ONLY
    assert log.foreign_entrypoints, (
        "既无可用探测器、也无法定位他方守卫入口：拦截证据不足"
    )


@pytest.fixture
def clean_proxy_env(monkeypatch):
    """清空全部代理相关环境变量，避免宿主环境把用例带偏。"""
    for name in PROXY_ENV_NAMES + ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


# ============================================================
# A. 防线在位
# ============================================================


def test_A01_http_transport_guard_is_installed(guard):
    """HTTP 传输层守卫必须已安装，且 httpx 与 httpx2 都被覆盖。"""
    assert guard.http_guard_installed() is True

    modules = guard.http_guard_modules()
    assert "httpx" in modules
    assert "httpx2" in modules, (
        "openai 3.x / starlette 使用的 httpx2 未被覆盖，仍存在真实出网通道"
    )

    for module_name in modules:
        for entry in ("handle_request", "handle_async_request"):
            assert f"{module_name}:{entry}" in guard._HTTP_ORIGINALS, (
                f"未保存 {module_name} 的 {entry} 原始实现"
            )


def test_A02_guard_only_patches_http_transports(guard):
    """只替换 HTTPTransport 家族，不得影响 MockTransport（Fake 传输）。"""
    import httpx2

    for module in (httpx, httpx2):
        assert (
            getattr(module.HTTPTransport.handle_request, "_wb_http_guard", False) is True
        ), f"{module.__name__}.HTTPTransport 未被守卫覆盖"
        assert (
            getattr(
                module.AsyncHTTPTransport.handle_async_request, "_wb_http_guard", False
            )
            is True
        ), f"{module.__name__}.AsyncHTTPTransport 未被守卫覆盖"

        # MockTransport 不继承 HTTPTransport，因此完全没有被触碰
        assert getattr(module.MockTransport, "_wb_http_guard", False) is False
        assert not issubclass(module.MockTransport, module.HTTPTransport)


# ============================================================
# B. 直连：HTTP / HTTPS 非回环目标必须在传输前被拒
# ============================================================


@pytest.mark.parametrize("url", [DOC_TARGET, DOC_TARGET_HTTP])
def test_B01_sync_non_loopback_target_is_blocked(guard, http_spy, url):
    transport = httpx.HTTPTransport()
    try:
        with pytest.raises(guard.BlockedNetworkAccessError) as excinfo:
            transport.handle_request(httpx.Request("GET", url))
    finally:
        transport.close()

    assert excinfo.value.kind == "httpx.HTTPTransport"
    assert "api.deepseek.com" in excinfo.value.target
    assert_no_http_delegation(http_spy)


def test_B02_async_non_loopback_target_is_blocked(guard, http_spy):
    async def _probe():
        transport = httpx.AsyncHTTPTransport()
        try:
            await transport.handle_async_request(httpx.Request("GET", DOC_TARGET))
        finally:
            await transport.aclose()

    with pytest.raises(guard.BlockedNetworkAccessError) as excinfo:
        asyncio.run(_probe())

    assert excinfo.value.kind == "httpx.AsyncHTTPTransport"
    assert "api.deepseek.com" in excinfo.value.target
    assert_no_http_delegation(http_spy)


# ============================================================
# C. 显式代理不能绕过（本轮 P1-01 的核心证明）
# ============================================================


def test_C01_explicit_loopback_proxy_cannot_bypass(guard, http_spy, clean_proxy_env):
    """显式指定回环代理时，仍按真实目标判定并拒绝。"""
    transport = httpx.HTTPTransport(proxy=DEAD_PROXY)
    try:
        # 前提校验：确实建立了代理路径，而不是空跑成直连
        assert type(transport._pool).__name__ == "HTTPProxy", "本用例未进入代理路径"

        with pytest.raises(guard.BlockedNetworkAccessError) as excinfo:
            transport.handle_request(httpx.Request("GET", DOC_TARGET))
    finally:
        transport.close()

    assert excinfo.value.kind == "httpx.HTTPTransport"
    assert "api.deepseek.com" in excinfo.value.target
    assert_no_http_delegation(http_spy)


def test_C02_client_with_explicit_proxy_cannot_bypass(guard, http_spy, clean_proxy_env):
    """已构造好的客户端（携带 proxy=）同样无法绕过。"""
    client = httpx.Client(proxy=DEAD_PROXY)
    try:
        request = client.build_request("POST", DOC_TARGET)
        with pytest.raises(guard.BlockedNetworkAccessError) as excinfo:
            client.send(request)
    finally:
        client.close()

    assert excinfo.value.kind == "httpx.HTTPTransport"
    assert_no_http_delegation(http_spy)


# ============================================================
# D. 环境代理不能绕过（大小写、HTTP/HTTPS/ALL 全覆盖）
# ============================================================


@pytest.mark.parametrize(
    "env_name,url",
    [
        pytest.param("HTTP_PROXY", DOC_TARGET_HTTP, id="HTTP_PROXY-http"),
        pytest.param("http_proxy", DOC_TARGET_HTTP, id="http_proxy-http"),
        pytest.param("HTTPS_PROXY", DOC_TARGET, id="HTTPS_PROXY-https"),
        pytest.param("https_proxy", DOC_TARGET, id="https_proxy-https"),
        pytest.param("ALL_PROXY", DOC_TARGET_HTTP, id="ALL_PROXY-http"),
        pytest.param("ALL_PROXY", DOC_TARGET, id="ALL_PROXY-https"),
        pytest.param("all_proxy", DOC_TARGET, id="all_proxy-https"),
    ],
)
def test_D01_env_proxy_cannot_bypass(guard, http_spy, clean_proxy_env, env_name, url):
    """任意一个代理环境变量都不得把非回环目标送出去。"""
    clean_proxy_env.setenv(env_name, DEAD_PROXY)

    client = httpx.Client(trust_env=True)
    try:
        request = client.build_request("GET", url)
        transport = client._transport_for_url(request.url)
        # 前提校验：该用例确实走到了代理分支
        assert type(getattr(transport, "_pool", None)).__name__ == "HTTPProxy", (
            f"{env_name} 对 {url} 未产生代理路径，本用例未覆盖预期场景"
        )

        with pytest.raises(guard.BlockedNetworkAccessError) as excinfo:
            client.send(request)
    finally:
        client.close()

    assert excinfo.value.kind == "httpx.HTTPTransport"
    assert "api.deepseek.com" in excinfo.value.target
    assert_no_http_delegation(http_spy)


# ============================================================
# E. OpenAI SDK 内部的 httpx 客户端
# ============================================================


def test_E01_openai_internal_httpx_is_blocked(guard, http_spy):
    """openai 3.x 的内部客户端是 ``httpx2.Client``（独立包），不是 ``httpx``。

    只覆盖 httpx 会留下真实出网通道 —— 本用例为固定这一点而存在。
    跨目录运行时 ``openai.OpenAI`` 可能被替换为离线空壳（无 ``_client``），
    因此实际探测直接驱动 ``httpx2.Client``：与 SDK 内部完全同一个库、
    同一个 ``HTTPTransport``，验证的正是那条真实出网通道。
    """
    import httpx2
    import openai

    client = openai.OpenAI(
        api_key="offline-test-key",
        base_url="https://api.deepseek.com",
        timeout=5.0,
    )

    # 第一层：SDK 请求出口守卫（更早）
    # 跨目录时该层可能由 `自动测试脚本` 的离线空壳提供，抛 RuntimeError，
    # 因此只校验「被某个 SDK 守卫拦住」，不锁定实现类。
    with pytest.raises(RuntimeError) as sdk_excinfo:
        client.chat.completions.create(
            model="deepseek-flash",
            messages=[{"role": "user", "content": "probe"}],
        )

    sdk_message = str(sdk_excinfo.value)
    assert any(
        marker in sdk_message
        for marker in ("测试 SDK 保护", "离线测试禁止发起真实 SDK 请求")
    ), f"SDK 请求未被任一守卫拦截：{sdk_message!r}"

    # 真类场景下顺带固定「SDK 内部用的是 httpx2」这一事实
    inner = getattr(client, "_client", None)
    if inner is not None:
        module_names = {cls.__module__ for cls in type(inner).__mro__}
        assert any(name == "httpx2" or name.startswith("httpx2.") for name in module_names), (
            f"SDK 内部客户端不再是 httpx2：{sorted(module_names)}"
        )

    # 第二层：直接驱动 httpx2 客户端，绕过第一层
    probe = httpx2.Client(trust_env=True)
    try:
        request = probe.build_request("POST", "https://api.deepseek.com/chat/completions")
        with pytest.raises(guard.BlockedNetworkAccessError) as excinfo:
            probe.send(request)
    finally:
        probe.close()

    assert excinfo.value.kind.startswith("httpx2."), (
        f"SDK 内部同源请求应由 httpx2 守卫拦截，实际 kind={excinfo.value.kind!r}"
    )
    assert "api.deepseek.com" in excinfo.value.target
    assert_no_http_delegation(http_spy)


# ============================================================
# F. 回环保留
# ============================================================


def test_F01_loopback_target_is_delegated_to_real_transport(guard, http_spy):
    """回环目标必须放行 —— 守卫会把控制权交回真实实现（由探测函数证明）。"""
    transport = httpx.HTTPTransport()
    try:
        with pytest.raises(AssertionError, match="委托给了真实传输层"):
            transport.handle_request(httpx.Request("GET", "http://127.0.0.1:1/"))
    finally:
        transport.close()

    assert http_spy.evidence == EVIDENCE_TRANSPORT_SPY, (
        f"无法证明回环委托：证据等级只有 {http_spy.evidence}，缺少可植入的委托链"
    )
    assert http_spy, "回环目标没有被交回真实传输层，可能被误拦"


def test_F02_loopback_http_end_to_end_still_works(guard):
    """本机回环 HTTP 端到端仍可跑通（保证本机 FastAPI 类测试不受影响）。

    服务端用裸 socket 手工实现，避免 ``HTTPServer`` 在 bind 时触发
    ``socket.getfqdn()`` 反向解析。全程只访问 127.0.0.1。
    """
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(8)
    listener.settimeout(0.5)
    port = listener.getsockname()[1]

    stop = threading.Event()

    def _serve():
        while not stop.is_set():
            try:
                conn, _ = listener.accept()
            except (socket.timeout, OSError):
                continue
            try:
                conn.settimeout(5.0)
                conn.recv(65536)
                conn.sendall(
                    b"HTTP/1.1 200 OK\r\n"
                    b"Content-Length: 8\r\n"
                    b"Connection: close\r\n\r\nloopback"
                )
            except OSError:
                pass
            finally:
                conn.close()

    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()

    try:
        client = httpx.Client(trust_env=False, timeout=10.0)
        try:
            response = client.get(f"http://127.0.0.1:{port}/")
        finally:
            client.close()

        assert response.status_code == 200
        assert response.text == "loopback"
    finally:
        stop.set()
        listener.close()
        thread.join(timeout=5.0)


# ============================================================
# G. Fake 传输不受影响
# ============================================================


def test_G01_mock_transport_is_not_affected(guard):
    """MockTransport 承载的非回环请求必须正常返回，绝不能被守卫拦掉。"""

    def handler(request):
        return httpx.Response(200, text="fake-ok")

    client = httpx.Client(transport=httpx.MockTransport(handler), trust_env=False)
    try:
        response = client.get("https://api.deepseek.com/anything")
    finally:
        client.close()

    assert response.status_code == 200
    assert response.text == "fake-ok"


# ============================================================
# H. 安装 / 卸载生命周期
# ============================================================


def test_H01_install_is_idempotent(guard):
    assert guard.install_http_guard() is False, "重复安装必须返回 False"
    assert guard.http_guard_installed() is True


def test_H02_uninstall_restores_originals_and_can_reinstall(guard):
    key = "httpx:handle_request"
    original = guard._HTTP_ORIGINALS[key]

    assert guard.uninstall_http_guard() is True
    try:
        assert httpx.HTTPTransport.handle_request is original, "卸载后未还原原始实现"
        assert guard.http_guard_installed() is False
    finally:
        assert guard.install_http_guard() is True

    assert guard.http_guard_installed() is True
    assert getattr(httpx.HTTPTransport.handle_request, "_wb_http_guard", False) is True


def test_H03_uninstall_is_safe_when_not_installed(guard, monkeypatch):
    """未安装状态下卸载必须安全返回 False，且不破坏原始实现。"""
    import httpx2

    key = "httpx:handle_request"
    original_sync = guard._HTTP_ORIGINALS[key]

    # 还原成原始实现，模拟「守卫生效前」的状态（httpx 与 httpx2 都要还原，
    # 否则 uninstall 会因另一个库仍是守卫而返回 True）
    for module_name, module in (("httpx", httpx), ("httpx2", httpx2)):
        for class_name, entry in (
            ("HTTPTransport", "handle_request"),
            ("AsyncHTTPTransport", "handle_async_request"),
        ):
            monkeypatch.setattr(
                getattr(module, class_name),
                entry,
                guard._HTTP_ORIGINALS[f"{module_name}:{entry}"],
            )

    assert guard.uninstall_http_guard() is False
    assert httpx.HTTPTransport.handle_request is original_sync


# ============================================================
# I. 同进程重复 pytest.main() 的生命周期（R2.2-02）
# ============================================================

_LIFECYCLE_SCRIPT = r'''
import json, os, sys
import pytest

root, basetemp = sys.argv[1], sys.argv[2]
TARGET = "test_network_safety_guard.py::test_T01_network_guard_covers_every_entrypoint"


def run(rel_target, tag, subdir=None):
    args = [
        "-q", "--no-header", "-p", "no:cacheprovider",
        "--basetemp=" + os.path.join(basetemp, tag),
        "-o", "tmp_path_retention_policy=all",
        rel_target,
    ]
    old = os.getcwd()
    cwd = os.path.join(root, subdir) if subdir else root
    os.chdir(cwd)
    try:
        return int(pytest.main(args))
    finally:
        os.chdir(old)


rc_root = run("端口测试/" + TARGET, "r1")
# 第二次进入子目录运行：rootdir 会退化，根 conftest 不再被 pytest 加载
rc_sub = run(TARGET, "r2", "端口测试")
print("LIFECYCLE=" + json.dumps({"root": rc_root, "subdir": rc_sub}))
'''


def test_I01_guards_survive_repeated_pytest_main(tmp_path):
    """同进程连续两次 pytest.main()（第二次 rootdir 退化）后守卫仍在位。

    阶段 1-R2.2 修复前：``sys.modules`` 里已有守卫模块，子目录 conftest
    的 ``if ... not in sys.modules`` 判定为「已加载」而跳过 exec_module，
    模块级安装不执行 —— 第二次会话完全没有守卫。本用例在子进程中复现
    该调用序列，要求两次都通过（T01 会断言守卫在位）。
    """
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    completed = subprocess.run(
        [sys.executable, "-B", "-c", _LIFECYCLE_SCRIPT, str(PROJECT_ROOT), str(tmp_path)],
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )

    marker = "LIFECYCLE="
    payload = None
    for line in completed.stdout.splitlines():
        if line.startswith(marker):
            payload = json.loads(line[len(marker):])
            break

    assert payload is not None, (
        "生命周期子进程未返回结果："
        f"rc={completed.returncode} "
        f"stdout_tail={completed.stdout.strip()[-500:]!r} "
        f"stderr_tail={completed.stderr.strip()[-500:]!r}"
    )
    assert payload["root"] == 0, "首次（项目根）会话未通过"
    assert payload["subdir"] == 0, (
        "第二次（进子目录、rootdir 退化）会话未通过 —— 守卫在重复 pytest.main 后丢失"
    )


# ============================================================
# J. 报告字段
# ============================================================


def test_J01_guards_report_exposes_http_state(guard):
    report = guard.guards_report()
    assert report["http_installed"] is True
    assert isinstance(report["http_blocked"], int)
    assert report["http_blocked"] >= 0


# ============================================================
# K. 拦截证据等级（阶段 1-R2.2-CLOSE · CLOSE-04）
# ============================================================


def test_K01_http_spy_uses_live_entrypoints_not_history(guard, http_spy):
    """探测点必须来自"当前在位的入口"，而不是历史 `_HTTP_ORIGINALS`。

    旧实现只断言字典非空：守卫被外部替换后仍会空转通过。现在改为按
    对象身份取入口，并把实际植入的探测点回报出来。
    """
    live = guard.http_guard_entrypoints()

    assert live, "HTTP 守卫没有任何在位入口，探测前提不成立"
    assert http_spy.evidence == EVIDENCE_TRANSPORT_SPY, (
        f"本守卫在位时证据等级应可证明未委托，实际为 {http_spy.evidence!r}"
    )
    assert set(http_spy.targets) == set(live), (
        f"探测点与在位入口不一致：targets={http_spy.targets} live={live}"
    )


def test_K02_http_spy_fails_closed_when_guard_is_gone(guard, monkeypatch):
    """守卫整体不在位时，探测器必须 fail-closed，而不是空转返回。"""
    import httpx2

    def _foreign(self, request):  # pragma: no cover - 不应被调用
        return None

    for cls in (httpx.HTTPTransport, httpx2.HTTPTransport):
        monkeypatch.setattr(cls, "handle_request", _foreign)
    for cls in (httpx.AsyncHTTPTransport, httpx2.AsyncHTTPTransport):
        monkeypatch.setattr(cls, "handle_async_request", _foreign)

    assert guard.http_guard_installed() is False
    assert guard.http_guard_entrypoints() == ()
    assert guard.http_foreign_entrypoints() == ()

    with pytest.raises(AssertionError, match="HTTP 守卫未在位"):
        _build_http_spy(guard, monkeypatch)

    # monkeypatch 自动还原后，守卫应能被修复且在位
    monkeypatch.undo()
    guard.install_http_guard()
    assert guard.http_guard_installed() is True
