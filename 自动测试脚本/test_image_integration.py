
from io import BytesIO
from types import SimpleNamespace

from PIL import Image

from image_client import DeepSeekVisionClient
from image_understanding import understand_image


class FakeCompletions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="模拟集成结果：图片里有一只猫"
                    )
                )
            ]
        )


class FakeSDKClient:
    def __init__(self):
        self.completions = FakeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


def make_png() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (2, 2), color="red").save(buffer, format="PNG")
    return buffer.getvalue()


def test_image_understanding_integrates_with_fake_deepseek_client():
    fake_sdk = FakeSDKClient()

    vision_client = DeepSeekVisionClient(
        api_key="offline-test-key",
        sdk_client=fake_sdk,
    )

    result = understand_image(
        make_png(),
        "图片里有什么？",
        vision_client,
    )

    assert result.success is True
    assert result.answer == "模拟集成结果：图片里有一只猫"
    assert result.error is None
    assert fake_sdk.completions.kwargs is not None


def test_invalid_image_does_not_reach_sdk():
    fake_sdk = FakeSDKClient()

    vision_client = DeepSeekVisionClient(
        api_key="offline-test-key",
        sdk_client=fake_sdk,
    )

    result = understand_image(
        b"invalid image",
        "图片里有什么？",
        vision_client,
    )

    assert result.success is False
    assert result.error is not None
    assert result.error.type == "invalid_image"
    assert fake_sdk.completions.kwargs is None


def test_over_pixel_limit_does_not_reach_client(monkeypatch):
    """[实际执行行为] 像素超限时不会调用注入的视觉客户端。

    使用自定义 fake client 直接验证 understand_image 的调用链，
    不构造 DeepSeekVisionClient，也不触及 OpenAI SDK。
    2x2 图片共 4 像素，应用常量临时缩小到 3 使其超限。
    """
    import image_validator

    class RecordingClient:
        def __init__(self) -> None:
            self.calls: list[tuple[bytes, str]] = []

        def analyze(self, image_data: bytes, question: str) -> str:
            self.calls.append((image_data, question))
            return "不应被调用"

    monkeypatch.setattr(image_validator, "MAX_IMAGE_PIXELS", 3)

    client = RecordingClient()
    result = understand_image(make_png(), "图片里有什么？", client)

    assert result.success is False
    assert result.answer is None
    assert result.error is not None
    assert result.error.type == "image_pixels_exceeded"
    assert client.calls == []
