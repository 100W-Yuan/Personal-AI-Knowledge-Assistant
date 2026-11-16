
import base64
from io import BytesIO
from types import SimpleNamespace

import httpx2
import pytest

from openai import APITimeoutError
from PIL import Image

from image_client import DeepSeekVisionClient, VisionClientTimeoutError
from image_understanding import understand_image


class FakeCompletions:
    def __init__(self, content="图片中有一只猫"):
        self.content = content
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.content)
                )
            ]
        )


class FakeSDKClient:
    def __init__(self, content="图片中有一只猫"):
        self.completions = FakeCompletions(content)
        self.chat = SimpleNamespace(completions=self.completions)


def make_png() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (2, 2), color="red").save(
        buffer,
        format="PNG",
    )
    return buffer.getvalue()


def test_client_encodes_image_and_returns_answer():
    fake_sdk = FakeSDKClient()
    client = DeepSeekVisionClient(
        api_key="offline-test-key",
        sdk_client=fake_sdk,
    )

    answer = client.analyze(make_png(), "图片里有什么？")

    assert answer == "图片中有一只猫"

    request = fake_sdk.completions.kwargs
    assert request["model"] == "deepseek-flash"

    content = request["messages"][0]["content"]
    assert content[0] == {
        "type": "text",
        "text": "图片里有什么？",
    }

    image_url = content[1]["image_url"]["url"]
    assert image_url.startswith("data:image/png;base64,")

    encoded = image_url.split(",", 1)[1]
    assert base64.b64decode(encoded) == make_png()


def test_client_uses_injected_sdk_without_network():
    fake_sdk = FakeSDKClient("模拟结果")

    client = DeepSeekVisionClient(
        api_key="offline-test-key",
        sdk_client=fake_sdk,
    )

    assert client.analyze(make_png(), "描述图片") == "模拟结果"
    assert fake_sdk.completions.kwargs is not None


# ============================================================
# 超时异常转换
# ============================================================

class TimeoutFakeCompletions:
    """create 立即抛 APITimeoutError，用于验证客户端层的异常转换。"""

    def __init__(self):
        self.kwargs = None
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        self.kwargs = kwargs
        raise APITimeoutError(
            request=httpx2.Request(
                "POST",
                "https://api.deepseek.com/chat/completions",
            )
        )


class TimeoutFakeSDKClient:
    def __init__(self):
        self.completions = TimeoutFakeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


def test_api_timeout_error_becomes_model_timeout():
    """[实际执行行为] APITimeoutError 经客户端转换后由理解层映射为 model_timeout。

    分两段验证不同边界：
    第一段——DeepSeekVisionClient.analyze 把 SDK 抛出的 APITimeoutError
    转成 VisionClientTimeoutError，断言 __cause__ 确为 APITimeoutError，
    以确保被测的是真实转换逻辑，而非直接抛出的目标异常；
    第二段——同一个客户端实例经 understand_image 调用后落到 model_timeout。

    全程使用假 SDK，不建立连接、不发起 HTTP 请求。
    httpx2.Request 只是请求描述对象，构造它不会触网。
    """
    fake_sdk = TimeoutFakeSDKClient()
    client = DeepSeekVisionClient(
        api_key="offline-test-key",
        sdk_client=fake_sdk,
    )

    # 第一段：客户端层转换
    with pytest.raises(VisionClientTimeoutError) as exc_info:
        client.analyze(make_png(), "描述图片")

    assert isinstance(exc_info.value.__cause__, APITimeoutError)
    assert fake_sdk.completions.calls == 1

    # 第二段：理解层映射
    result = understand_image(make_png(), "描述图片", client)

    assert result.success is False
    assert result.answer is None
    assert result.error is not None
    assert result.error.type == "model_timeout"
    assert fake_sdk.completions.calls == 2
