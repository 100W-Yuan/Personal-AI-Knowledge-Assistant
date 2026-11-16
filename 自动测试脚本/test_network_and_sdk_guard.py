# -*- coding: utf-8 -*-
"""R2-04：主动验证网络守卫与 SDK 空壳确实生效。

背景
----
``自动测试脚本/conftest.py`` 提供了四层离线隔离（假 dotenv / 假 Key /
空壳 OpenAI / 回环感知网络守卫）。但"守卫存在"与"守卫有效"是两件事：
只凭测试里有 Fake 客户端，不能推断没有真实网络调用。

因此本文件**主动发起**被禁止的连接与 SDK 请求，断言它们必须在真正
触网之前就被拦下：

1. 非回环主机名（``create_connection``）；
2. 非回环 IP（``socket.connect``）；
3. DNS 解析（``getaddrinfo``）；
4. 回环连接必须**仍然可用**（守卫不能误伤 TestClient 的进程内传输）；
5. 真实 SDK 请求必须被空壳拒绝。

所有被拦截事件都通过 ``allow_isolation_events`` 显式声明，否则
conftest 的 M1 机制会把它判为未声明的隔离违规。

安全边界：以上用例全部依赖 conftest 的守卫在连接前抛异常，因此
**不会真正出网**；目标地址使用保留域名 ``example.invalid`` 与
TEST-NET-1 网段 ``192.0.2.1``。
"""

import socket
import sys
from pathlib import Path

import pytest


def _offline_conftest_module():
    """按文件路径定位本目录的 conftest 模块。

    pytest 会为多个 conftest.py 生成唯一模块名，直接 ``import conftest``
    可能拿到根级 conftest，因此改为按 ``__file__`` 精确匹配。
    """
    target = (Path(__file__).resolve().parent / "conftest.py").resolve()

    for module in list(sys.modules.values()):
        module_file = getattr(module, "__file__", None)
        if not module_file or not str(module_file).endswith("conftest.py"):
            continue
        try:
            if Path(module_file).resolve() == target:
                return module
        except OSError:  # pragma: no cover
            continue

    return None


# ========================================================
# 网络守卫主动验证
# ========================================================


def test_non_loopback_hostname_is_blocked(allow_isolation_events):
    """主机名形式的非回环目标必须在 connect 之前被拒绝。"""

    allow_isolation_events("non_loopback")

    with pytest.raises(RuntimeError, match="NETWORK_BLOCKED"):
        socket.create_connection(("example.invalid", 443), timeout=0.5)


def test_non_loopback_ip_is_blocked(allow_isolation_events):
    """裸 IP 形式（不经 DNS）同样必须被拒绝。"""

    allow_isolation_events("non_loopback")

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        with pytest.raises(RuntimeError, match="NETWORK_BLOCKED"):
            probe.connect(("192.0.2.1", 443))
    finally:
        probe.close()


def test_dns_resolution_is_blocked(allow_isolation_events):
    """DNS 解析必须被拒绝，避免以"只解析不连接"的方式出网。"""

    allow_isolation_events("dns_blocked")

    with pytest.raises(RuntimeError, match="NETWORK_BLOCKED_DNS"):
        socket.getaddrinfo("example.invalid", 443)


def test_loopback_connection_is_still_allowed():
    """回环必须放行：FastAPI TestClient 依赖进程内回环传输。

    本用例不声明任何隔离事件——如果守卫误伤回环，它会先失败。
    """

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]

        client = socket.create_connection(("127.0.0.1", port), timeout=1.0)

        try:
            assert client.getsockname()[1] > 0
        finally:
            client.close()
    finally:
        server.close()


# ========================================================
# SDK 空壳主动验证
# ========================================================


def test_openai_class_is_offline_shell():
    """openai.OpenAI 必须已被替换为空壳。"""

    import openai

    assert getattr(openai.OpenAI, "_wb_offline_shell", False) is True


def test_sdk_create_is_rejected_before_any_request(allow_isolation_events):
    """真实 SDK 请求必须被空壳拒绝，并计入 sdk_create_attempts。"""

    allow_isolation_events("sdk_create_attempts")

    import openai

    offline = _offline_conftest_module()

    assert offline is not None, "未定位到本目录的离线隔离 conftest 模块"

    before = offline.STATE["sdk_create_attempts"]

    client = openai.OpenAI(
        api_key="offline-test-key",
        base_url="https://api.example.invalid",
    )

    with pytest.raises(RuntimeError, match="离线测试禁止发起真实 SDK 请求"):
        client.chat.completions.create(
            model="fake-model",
            messages=[{"role": "user", "content": "不应发出"}],
        )

    assert offline.STATE["sdk_create_attempts"] == before + 1


def test_dotenv_is_fake():
    """load_dotenv 必须是假实现，避免读入真实密钥。"""

    assert getattr(sys.modules.get("dotenv"), "load_dotenv", None) is not None
    assert getattr(
        sys.modules["dotenv"].load_dotenv, "_wb_fake_dotenv", False
    ) is True
