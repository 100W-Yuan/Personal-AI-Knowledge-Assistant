
"""P1 遗留风险补测 A：invalid_model_response 分支覆盖（纯离线，仅新增测试）。

覆盖三条理解层路径与两条适配层路径，并区分二者：
    - 理解层：`client.analyze()` 直接返回非文本 → invalid_model_response。
    - 适配层：`DeepSeekVisionClient.analyze()` 收到 SDK 的 content=None 时，
      会先在 `image_client.py:80-81` 抛 ValueError，最终落到 model_error。
      这两条路径**不是**同一分支，测试分别断言，不合并。

安全说明：
    本文件所有客户端均为 fake/stub，`DeepSeekVisionClient` 一律通过
    `sdk_client=` 注入假 SDK（`image_client.py:36` 的 `or` 短路），
    不会构造真实 OpenAI 客户端，不产生任何网络请求。
    正式运行时由外部隔离启动脚本先注入假 dotenv 与空壳 openai.OpenAI；
    本文件顶部的 dotenv 兜底仅用于防止误直接运行时读取 .env。
"""

import sys
import types

# --- 兜底：确保导入生产模块前 dotenv 已被替换为空操作模块 ---
if "dotenv" not in sys.modules:
    _fake_dotenv = types.ModuleType("dotenv")
    _fake_dotenv.load_dotenv = lambda *args, **kwargs: False
    sys.modules["dotenv"] = _fake_dotenv

from io import BytesIO
from types import SimpleNamespace

from fastapi.testclient import TestClient
from PIL import Image

from image_client import DeepSeekVisionClient
from image_understanding import understand_image


# 与 image_understanding.py:57 保持一致
EXPECTED_MESSAGE = "视觉客户端没有返回有效文本"


def make_png() -> bytes:
    """生成真实有效的 2x2 PNG 字节。"""
    buffer = BytesIO()
    Image.new("RGB", (2, 2), color="red").save(buffer, format="PNG")
    return buffer.getvalue()


class StubVisionClient:
    """按固定值返回的 fake 客户端，记录调用次数与实参。"""

    def __init__(self, answer):
        self._answer = answer
        self.calls: list[tuple[bytes, str]] = []

    def analyze(self, image_data: bytes, question: str):
        self.calls.append((image_data, question))
        return self._answer


class FakeUpload:
    """最小 UploadFile 替身，用于直接调用端点函数并观察关闭次数。"""

    def __init__(self, data: bytes):
        self.data = data
        self.close_calls = 0

    async def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            return self.data
        return self.data[:size]

    async def close(self) -> None:
        self.close_calls += 1


class _FakeCompletions:
    def __init__(self, content):
        self._content = content
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=self._content))
            ]
        )


class _FakeSDKClient:
    """假 OpenAI SDK，仅暴露 chat.completions.create。"""

    def __init__(self, content):
        self.completions = _FakeCompletions(content)
        self.chat = SimpleNamespace(completions=self.completions)


# ============================================================
# 一、理解层：非文本返回值
# ============================================================


def test_empty_string_answer_is_invalid_model_response():
    """空字符串：isinstance 通过，但 strip() 为空 → invalid_model_response。"""
    fake = StubVisionClient("")

    result = understand_image(make_png(), "图片里有什么？", fake)

    assert result.success is False
    assert result.answer is None
    assert result.error is not None
    assert result.error.type == "invalid_model_response"
    assert result.error.message == EXPECTED_MESSAGE
    assert len(fake.calls) == 1


def test_whitespace_only_answer_is_invalid_model_response():
    """纯空白字符串：strip() 为空 → invalid_model_response。"""
    fake = StubVisionClient("  \n\t ")

    result = understand_image(make_png(), "图片里有什么？", fake)

    assert result.success is False
    assert result.answer is None
    assert result.error is not None
    assert result.error.type == "invalid_model_response"
    assert result.error.message == EXPECTED_MESSAGE
    assert len(fake.calls) == 1


def test_none_answer_is_invalid_model_response_at_understanding_layer():
    """None：`not isinstance(answer, str)` 成立 → invalid_model_response。

    注意：这是**理解层**对注入客户端的契约检查，
    与下面适配层 content=None 的 model_error 不是同一条路径。
    """
    fake = StubVisionClient(None)

    result = understand_image(make_png(), "图片里有什么？", fake)

    assert result.success is False
    assert result.answer is None
    assert result.error is not None
    assert result.error.type == "invalid_model_response"
    assert result.error.message == EXPECTED_MESSAGE
    assert len(fake.calls) == 1


# ============================================================
# 二、路由层：HTTP 状态码、JSON 响应体、上传文件关闭
# ============================================================


def test_route_returns_200_and_structured_error(monkeypatch):
    """理解层返回 invalid_model_response 时，接口仍为 200 + 结构化错误体。"""
    import main

    fake = StubVisionClient("")
    monkeypatch.setattr(main, "get_vision_client", lambda: fake, raising=False)

    client = TestClient(main.app)
    response = client.post(
        "/analyze-image",
        files={"file": ("blank.png", make_png(), "image/png")},
        data={"question": "图片里有什么？"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "success": False,
        "answer": None,
        "error": {
            "type": "invalid_model_response",
            "message": EXPECTED_MESSAGE,
        },
    }
    assert len(fake.calls) == 1


def test_route_closes_upload_once_on_invalid_model_response(monkeypatch):
    """同一路径下，上传文件仍只关闭一次。"""
    import asyncio

    import main

    fake = StubVisionClient("  \n\t ")
    monkeypatch.setattr(main, "get_vision_client", lambda: fake, raising=False)

    upload = FakeUpload(make_png())
    result = asyncio.run(
        main.analyze_image(file=upload, question="图片里有什么？")
    )

    assert result["success"] is False
    assert result["answer"] is None
    assert result["error"]["type"] == "invalid_model_response"
    assert result["error"]["message"] == EXPECTED_MESSAGE
    assert upload.close_calls == 1


# ============================================================
# 三、适配层：真实客户端类 + 假 SDK
# ============================================================


def test_adapter_content_empty_string_reaches_invalid_model_response():
    """SDK 返回 content="" 时，适配层原样返回空串 → invalid_model_response。"""
    sdk = _FakeSDKClient("")
    vision_client = DeepSeekVisionClient(
        api_key="offline-test-key",
        sdk_client=sdk,
    )

    result = understand_image(make_png(), "图片里有什么？", vision_client)

    assert result.success is False
    assert result.answer is None
    assert result.error is not None
    assert result.error.type == "invalid_model_response"
    assert result.error.message == EXPECTED_MESSAGE
    assert sdk.completions.calls == 1


def test_adapter_content_none_becomes_model_error():
    """SDK 返回 content=None 时，适配层先抛 ValueError → 映射为 model_error。

    与理解层的 None 分支区分：此处异常发生在客户端内部，
    已被 `image_client.py:80-81` 拦截，理解层看到的是异常而非返回值。
    """
    sdk = _FakeSDKClient(None)
    vision_client = DeepSeekVisionClient(
        api_key="offline-test-key",
        sdk_client=sdk,
    )

    result = understand_image(make_png(), "图片里有什么？", vision_client)

    assert result.success is False
    assert result.answer is None
    assert result.error is not None
    assert result.error.type == "model_error"
    assert result.error.message == "视觉客户端调用失败"
    assert sdk.completions.calls == 1
