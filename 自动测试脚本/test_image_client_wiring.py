# -*- coding: utf-8 -*-
"""第八阶段：视觉客户端最小接入的离线接线测试。

验证层次标注（沿用本项目既有约定）：
[实际执行行为]  直接 await main.analyze_image(...)，真实执行
                main.analyze_image -> understand_image -> client.analyze 全链路；
[交互计数]      用 fake client / 计数型工厂 / fake SDK 记录调用次数与实参；
[静态结构]      只确认符号或签名存在，不声称运行期效果。

关键约束：
1. 本文件不 mock main.understand_image。个别测试使用「委托型 spy」
   （先调用真实函数再记录参数），真实逻辑完整执行，仅用于证明组合调用发生。
2. 全局 network_guard fixture 把 image_client 模块内的 OpenAI 入口替换为
   计数型 fake：任何路径只要尝试创建或调用 SDK，都会被记录或直接失败。
3. 不读取 .env、不读取真实 API Key、不发起任何网络请求。

本轮明确未验证（不得据本文件声称）：
1. 真实 DeepSeek 视觉模型的可用性、回答质量与时延；
2. 真实 HTTP / multipart 传输层行为（本文件直接调用端点函数，不经 ASGI 栈）；
3. SDK 对 image_url 内容块的真实类型校验（被 fake SDK 绕过）。
"""

import asyncio
from io import BytesIO

import pytest
from fastapi import HTTPException
from PIL import Image

import image_client
import main
from image_client import VisionClient, VisionClientTimeoutError


# 仅用于 monkeypatch 的假值，不是真实密钥。
FAKE_API_KEY = "offline-test-key-not-real"


# ============================================================
# 离线护栏
# ============================================================

class NetworkGuard:
    """替换 SDK 构建入口，确保测试期间不可能发出真实模型请求。"""

    def __init__(self):
        self.constructed = 0
        self.create_calls = 0

    def build_client(self, **kwargs):
        """冒充 openai.OpenAI：被调用即计数，返回可计数的 fake SDK。"""
        self.constructed += 1
        guard = self

        class _Completions:
            def create(self, **create_kwargs):
                guard.create_calls += 1
                raise AssertionError("离线测试禁止调用真实模型接口")

        class _Chat:
            completions = _Completions()

        class _FakeSDK:
            chat = _Chat()

        return _FakeSDK()


@pytest.fixture(autouse=True)
def network_guard(monkeypatch):
    """全局生效：替换 image_client 中的 OpenAI 入口。"""
    guard = NetworkGuard()
    monkeypatch.setattr(image_client, "OpenAI", guard.build_client)
    yield guard
    # 全文件共同保证：没有任何用例走到真实请求分支。
    assert guard.create_calls == 0


@pytest.fixture
def clear_vision_cache():
    """清理 get_vision_client 的 lru_cache，避免用例间相互污染。

    修复前 main 尚无该工厂，此时不报错，让用例在真正的断言处失败。
    """
    getter = getattr(main, "get_vision_client", None)
    if getter is not None:
        getter.cache_clear()
    yield
    getter = getattr(main, "get_vision_client", None)
    if getter is not None:
        getter.cache_clear()


# ============================================================
# 测试替身
# ============================================================

class FakeUpload:
    """UploadFile 替身（鸭子类型），记录 read / close 调用。"""

    def __init__(self, data: bytes | None = None, total_bytes: int | None = None):
        self.data = data
        self.total_bytes = len(data) if data is not None else total_bytes
        self.read_calls: list[int] = []
        self.close_calls = 0
        self.closed = False

    async def read(self, size: int = -1) -> bytes:
        self.read_calls.append(size)
        start = 0
        end = (
            self.total_bytes
            if size is None or size < 0
            else min(size, self.total_bytes)
        )
        if self.data is not None:
            return self.data[start:end]
        return b"x" * (end - start)

    async def close(self) -> None:
        self.close_calls += 1
        self.closed = True


class RecordingVisionClient:
    """记录实参的 fake 视觉客户端，符合 VisionClient 结构约定。"""

    def __init__(self, answer: str = "离线模拟回答", error: Exception | None = None):
        self.answer = answer
        self.error = error
        self.calls: list[tuple[bytes, str]] = []

    def analyze(self, image_data: bytes, question: str) -> str:
        self.calls.append((image_data, question))
        if self.error is not None:
            raise self.error
        return self.answer


class RecordingClientFactory:
    """计数型工厂，用于「正常配置只创建一次」与「413 不创建」。"""

    def __init__(self):
        self.constructed = 0
        self.init_kwargs: list[dict] = []
        self.instance = RecordingVisionClient(answer="工厂创建实例的回答")

    def __call__(self, **kwargs):
        self.constructed += 1
        self.init_kwargs.append(kwargs)
        return self.instance


def make_png() -> bytes:
    """生成真实有效的 2x2 PNG 字节。"""
    buffer = BytesIO()
    Image.new("RGB", (2, 2), color="red").save(buffer, format="PNG")
    return buffer.getvalue()


def run_endpoint(file, question="图片里有什么？"):
    """同步执行异步端点函数。"""
    return asyncio.run(main.analyze_image(file=file, question=question))


# ============================================================
# 1. 真实注入路径
# ============================================================

def test_route_injects_fake_client_into_real_understanding(monkeypatch):
    """[实际执行行为 + 交互计数] 端点把 fake client 真正注入 understand_image。

    使用委托型 spy 包裹真实 understand_image：先记录实参，再调用原函数，
    因此断言的是「真实组合调用已发生」，而不是被 mock 替换后的假象。
    """
    png = make_png()
    question = "请描述这张图片"
    fake_client = RecordingVisionClient(answer="离线识别的结果")
    assert isinstance(fake_client, VisionClient)

    monkeypatch.setattr(
        main, "get_vision_client", lambda: fake_client, raising=False
    )

    real_understand = main.understand_image
    observed: list[dict] = []

    def delegating_spy(*args, **kwargs):
        observed.append(kwargs)
        return real_understand(*args, **kwargs)

    monkeypatch.setattr(main, "understand_image", delegating_spy)

    fake_upload = FakeUpload(data=png)
    result = run_endpoint(fake_upload, question)

    # understand_image 被真实调用一次，且 client 来自工厂注入。
    assert len(observed) == 1
    assert observed[0]["client"] is fake_client

    # fake client 的 analyze 被调用一次，且图片字节与问题文本准确传入。
    assert fake_client.calls == [(png, question)]

    # 成功响应结构保持不变。
    assert result == {
        "success": True,
        "answer": "离线识别的结果",
        "error": None,
    }


def test_valid_upload_no_longer_degrades_to_model_not_configured(monkeypatch):
    """[实际执行行为] 有效图片不再退化为 model_not_configured。

    该用例在修复前必须失败：当时端点不传 client，understand_image
    会在客户端检查分支直接返回 model_not_configured。
    """
    fake_client = RecordingVisionClient(answer="离线回答")
    monkeypatch.setattr(
        main, "get_vision_client", lambda: fake_client, raising=False
    )

    result = run_endpoint(FakeUpload(data=make_png()))

    assert result["success"] is True, (
        f"未注入 client 时返回了 {result['error']}"
    )
    assert result["error"] is None
    assert len(fake_client.calls) == 1


# ============================================================
# 2. 缺少配置
# ============================================================

def test_missing_api_key_returns_model_not_configured_without_sdk(
    monkeypatch, clear_vision_cache, network_guard
):
    """[实际执行行为 + 交互计数] 缺 key 时结构化降级，且不创建 SDK 客户端。"""
    getter = getattr(main, "get_vision_client", None)
    assert getter is not None, "main 尚未提供 get_vision_client 工厂"

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    getter.cache_clear()

    assert getter() is None, "缺配置时工厂必须返回 None，而不是抛异常"

    fake_client_factory = RecordingClientFactory()
    monkeypatch.setattr(
        main, "DeepSeekVisionClient", fake_client_factory, raising=False
    )

    result = run_endpoint(FakeUpload(data=make_png()))

    assert result["success"] is False
    assert result["answer"] is None
    assert result["error"] == {
        "type": "model_not_configured",
        "message": "尚未配置视觉模型客户端",
    }

    # 未构造任何视觉客户端，SDK 入口未被使用，网络请求为零。
    assert fake_client_factory.constructed == 0
    assert network_guard.constructed == 0
    assert network_guard.create_calls == 0


# ============================================================
# 3. 工厂缓存
# ============================================================

def test_vision_client_factory_caches_single_instance(
    monkeypatch, clear_vision_cache
):
    """[交互计数] 正常配置下重复调用工厂只创建一个客户端。"""
    getter = getattr(main, "get_vision_client", None)
    assert getter is not None, "main 尚未提供 get_vision_client 工厂"

    monkeypatch.setenv("DEEPSEEK_API_KEY", FAKE_API_KEY)
    factory = RecordingClientFactory()
    monkeypatch.setattr(main, "DeepSeekVisionClient", factory, raising=False)
    getter.cache_clear()

    first = getter()
    second = getter()

    assert factory.constructed == 1
    assert first is second
    assert first is factory.instance

    # 只传 api_key：base_url / model / timeout 沿用客户端默认值。
    assert factory.init_kwargs == [{"api_key": FAKE_API_KEY}]


# ============================================================
# 4. 413 不触发客户端创建
# ============================================================

def test_oversized_upload_returns_413_without_creating_client(monkeypatch):
    """[实际执行行为 + 交互计数] 超限在进入理解函数前返回 413，工厂未被调用。"""
    monkeypatch.setattr(main, "MAX_IMAGE_SIZE_BYTES", 1024)

    factory = RecordingClientFactory()
    monkeypatch.setattr(main, "get_vision_client", factory, raising=False)

    fake_upload = FakeUpload(total_bytes=1025)

    with pytest.raises(HTTPException) as exc_info:
        run_endpoint(fake_upload)

    assert exc_info.value.status_code == 413
    assert exc_info.value.detail["type"] == "image_too_large"
    assert factory.constructed == 0
    assert fake_upload.close_calls == 1


# ============================================================
# 5. 异常与文件清理
# ============================================================

@pytest.mark.parametrize(
    "case, error, expected_error_type",
    [
        ("success", None, None),
        ("timeout", VisionClientTimeoutError("模拟超时"), "model_timeout"),
        ("model_error", RuntimeError("模拟模型异常"), "model_error"),
    ],
)
def test_file_closed_once_across_outcomes(
    monkeypatch, case, error, expected_error_type
):
    """[实际执行行为] 成功 / 超时 / 普通异常下，文件均只关闭一次。"""
    fake_client = RecordingVisionClient(
        answer="离线回答",
        error=error,
    )
    monkeypatch.setattr(
        main, "get_vision_client", lambda: fake_client, raising=False
    )

    fake_upload = FakeUpload(data=make_png())
    result = run_endpoint(fake_upload)

    if expected_error_type is None:
        assert result["success"] is True
        assert result["error"] is None
    else:
        assert result["success"] is False
        assert result["error"]["type"] == expected_error_type

    assert len(fake_client.calls) == 1, f"{case} 场景未进入 fake client"
    assert fake_upload.close_calls == 1
    assert fake_upload.closed is True


def test_factory_exception_propagates_out_of_endpoint(monkeypatch):
    """[实际执行行为] 工厂异常被记录一次后原样穿透端点向上传播。

    端点对 get_vision_client() 的调用包了一层 try/except：记录日志后
    用裸 raise 重抛，因此异常类型与对象都不变，HTTP 层仍按 500 处理。
    本用例同时验证三件事：异常对象原样传播、logger.exception 恰好调用
    一次、finally 仍关闭上传文件。

    全程离线：桩在 main.get_vision_client 层即抛出，不会进入
    image_client，也不会构造 SDK 或发起请求。
    """

    class FactoryBoom(RuntimeError):
        """仅用于本用例的哨兵异常，避免与项目既有异常类型混淆。"""

    expected_exception = FactoryBoom("模拟视觉客户端工厂初始化失败")

    def boom():
        raise expected_exception

    monkeypatch.setattr(main, "get_vision_client", boom, raising=False)

    logged: list[str] = []

    def fake_exception(message, *args, **kwargs):
        logged.append(message)

    monkeypatch.setattr(main.logger, "exception", fake_exception)

    fake_upload = FakeUpload(data=make_png())

    with pytest.raises(FactoryBoom) as exc_info:
        run_endpoint(fake_upload)

    assert str(exc_info.value) == "模拟视觉客户端工厂初始化失败"
    assert exc_info.value is expected_exception
    assert len(logged) == 1
    assert "视觉客户端初始化失败" in logged[0]
    assert fake_upload.close_calls == 1
    assert fake_upload.closed is True
