# -*- coding: utf-8 -*-
"""第八阶段：视觉客户端工厂的离线集成测试。

本轮唯一目标：验证 main.get_vision_client() 能否构造出**项目真实的**
DeepSeekVisionClient 实例（而不是 Mock 替身），且全程零网络。

网络隔离采用三层证明，不以“没有看到网络日志”作为依据：
1. 在 image_client 模块的 SDK 导入位置注入 FakeSDKFactory，
   使 DeepSeekVisionClient.__init__ 拿到 fake，而不是真实 OpenAI 客户端；
2. Fake SDK 的 create() 一旦被调用即计数并抛 AssertionError；
3. autouse fixture 封堵 socket.socket / socket.create_connection，
   任何建链尝试都会立即失败。

本文件不调用 DeepSeekVisionClient.analyze()，不使用真实图片、真实密钥。

未验证（不得据本文件声称）：
- 真实 DeepSeek 视觉模型的可用性、请求格式兼容性与时延；
- 真实 HTTP 传输层行为；
- 真实 API Key 的有效性（测试只使用虚构值）。
"""

import socket
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import image_client
import main
from image_client import DeepSeekVisionClient


# 仅用于 monkeypatch 的虚构值，不是真实密钥。
FAKE_API_KEY = "offline-test-key-factory-integration"

# DeepSeekVisionClient 的默认值（image_client.py:29-31）。
EXPECTED_BASE_URL = "https://api.deepseek.com"
EXPECTED_TIMEOUT = 30.0
EXPECTED_MODEL = "deepseek-flash"


# ============================================================
# 网络隔离
# ============================================================

@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    """第三层防护：任何网络套接字创建或连接尝试都立即失败。"""

    def _blocked(*args, **kwargs):
        raise AssertionError("离线测试禁止创建网络套接字或发起连接")

    monkeypatch.setattr(socket, "socket", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    yield


class FakeSDK:
    """冒充 openai.OpenAI 实例；任何请求都会失败并计数。"""

    def __init__(self, factory):
        self._factory = factory
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):
        self._factory.create_calls += 1
        raise AssertionError("离线测试禁止发起真实 SDK 请求")


class FakeSDKFactory:
    """替换 image_client.OpenAI，记录构造次数与构造入参。"""

    def __init__(self):
        self.constructed = 0
        self.create_calls = 0
        self.init_kwargs = []

    def __call__(self, **kwargs):
        self.constructed += 1
        self.init_kwargs.append(kwargs)
        return FakeSDK(self)


@pytest.fixture
def factory_getter():
    """取得真实工厂，并在用例前后清空 lru_cache。"""
    getter = getattr(main, "get_vision_client", None)
    assert getter is not None, "main 必须提供 get_vision_client"
    getter.cache_clear()
    yield getter
    getter.cache_clear()


# ============================================================
# 测试 1：真实客户端工厂集成
# ============================================================

def test_factory_builds_real_client_with_fake_sdk(monkeypatch, factory_getter):
    """[真实构造 + 交互计数] 工厂返回真实 DeepSeekVisionClient 实例。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", FAKE_API_KEY)
    sdk = FakeSDKFactory()
    monkeypatch.setattr(image_client, "OpenAI", sdk)

    client = factory_getter()

    # 返回的是项目真实类，而不是替代它的 Mock 对象。
    assert isinstance(client, DeepSeekVisionClient)
    assert type(client) is DeepSeekVisionClient
    assert not isinstance(client, Mock)

    # 底层 SDK 只构造一次，且拿到的是虚构 Key。
    assert sdk.constructed == 1
    init_kwargs = sdk.init_kwargs[0]
    assert init_kwargs["api_key"] == FAKE_API_KEY
    assert init_kwargs["base_url"] == EXPECTED_BASE_URL
    assert init_kwargs["timeout"] == EXPECTED_TIMEOUT

    # 客户端保留默认模型名，并持有 fake SDK 而非真实客户端。
    assert client.model == EXPECTED_MODEL
    assert isinstance(client.client, FakeSDK)

    # 构造过程未触发任何请求。
    assert sdk.create_calls == 0


# ============================================================
# 测试 2：缓存行为
# ============================================================

def test_factory_returns_cached_single_instance(monkeypatch, factory_getter):
    """[交互计数] 连续两次调用返回同一对象，底层只构造一次。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", FAKE_API_KEY)
    sdk = FakeSDKFactory()
    monkeypatch.setattr(image_client, "OpenAI", sdk)

    first = factory_getter()
    second = factory_getter()

    assert first is second
    assert sdk.constructed == 1
    assert factory_getter.cache_info().currsize == 1
    assert sdk.create_calls == 0


# ============================================================
# 测试 3：缺少配置时的安全行为
# ============================================================

def test_factory_returns_none_without_api_key(monkeypatch, factory_getter):
    """[实际执行行为] 缺 key 时返回 None，不构造 SDK、不发送请求。"""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    sdk = FakeSDKFactory()
    monkeypatch.setattr(image_client, "OpenAI", sdk)

    assert factory_getter() is None
    assert sdk.constructed == 0
    assert sdk.create_calls == 0


# ============================================================
# 测试 4：重试次数显式配置
# ============================================================

def test_factory_passes_explicit_max_retries(monkeypatch, factory_getter):
    """[静态参数 + 交互计数] 工厂构造的 SDK 客户端显式传入 max_retries=0。

    只断言构造入参，不发送任何请求：image_client.OpenAI 已被
    FakeSDKFactory 替换，autouse 的 block_network 同时封堵 socket。
    注意：该断言证明的是构造参数，不能据此声称底层 HTTP 交换次数。
    """
    monkeypatch.setenv("DEEPSEEK_API_KEY", FAKE_API_KEY)
    sdk = FakeSDKFactory()
    monkeypatch.setattr(image_client, "OpenAI", sdk)

    factory_getter()

    assert sdk.constructed == 1
    assert sdk.init_kwargs[0]["max_retries"] == 0
    assert sdk.create_calls == 0
