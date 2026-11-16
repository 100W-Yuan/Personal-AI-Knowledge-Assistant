# -*- coding: utf-8 -*-
"""P1 专项：图片理解模块的上游异常分类（纯离线，零网络）。

覆盖 `image_understanding.py` 新增的五类分支与兜底分支：
- AuthenticationError   -> api_auth_error
- RateLimitError        -> api_rate_limited
- InternalServerError   -> api_upstream_error
- APIConnectionError    -> api_connection_error
- 直接抛 APITimeoutError -> model_timeout
- RuntimeError          -> model_error（并验证 logger.exception 被调用）

安全说明：
- 全部使用注入的 fake client 抛异常，不构造真实 OpenAI 客户端；
- `httpx2.Request` / `httpx2.Response` 只是请求/响应描述对象，构造它们不触网；
- 断言公开 message 为固定文案，且不包含上游哨兵字符串。
"""

import logging
from io import BytesIO

import httpx2
import pytest
from PIL import Image
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    InternalServerError,
    RateLimitError,
)

from image_understanding import understand_image


SENTINEL = "upstream-secret-sentinel-9f3a"


def make_png() -> bytes:
    """生成真实有效的 2x2 PNG 字节。"""
    buffer = BytesIO()
    Image.new("RGB", (2, 2), color="red").save(buffer, format="PNG")
    return buffer.getvalue()


def _request() -> httpx2.Request:
    return httpx2.Request("POST", "https://api.deepseek.com/chat/completions")


def _status_error(exc_cls, status_code):
    """按当前 SDK 签名构造 APIStatusError 子类：message, response=, body=。"""
    response = httpx2.Response(status_code, request=_request())
    return exc_cls(SENTINEL, response=response, body=None)


class RaisingClient:
    """符合 VisionClient 结构约定的 fake 客户端：analyze 直接抛指定异常。"""

    def __init__(self, exc: BaseException):
        self.exc = exc
        self.calls = 0

    def analyze(self, image_data: bytes, question: str) -> str:
        self.calls += 1
        raise self.exc


@pytest.mark.parametrize(
    "exc_factory, expected_type, expected_message",
    [
        (
            lambda: _status_error(AuthenticationError, 401),
            "api_auth_error",
            "视觉模型认证失败",
        ),
        (
            lambda: _status_error(RateLimitError, 429),
            "api_rate_limited",
            "视觉模型请求被限流",
        ),
        (
            lambda: _status_error(InternalServerError, 500),
            "api_upstream_error",
            "视觉模型服务返回错误",
        ),
        (
            lambda: APIConnectionError(request=_request()),
            "api_connection_error",
            "无法连接视觉模型服务",
        ),
    ],
    ids=["auth_401", "rate_limit_429", "upstream_500", "connection_error"],
)
def test_api_errors_are_classified(exc_factory, expected_type, expected_message):
    fake = RaisingClient(exc_factory())

    result = understand_image(make_png(), "图片里有什么？", fake)

    assert result.success is False
    assert result.answer is None
    assert result.error is not None
    assert result.error.type == expected_type
    assert result.error.message == expected_message
    assert SENTINEL not in result.error.message
    assert fake.calls == 1


def test_direct_api_timeout_maps_to_model_timeout():
    """未经客户端转换的 APITimeoutError 也必须落到 model_timeout。"""
    fake = RaisingClient(APITimeoutError(_request()))

    result = understand_image(make_png(), "图片里有什么？", fake)

    assert result.success is False
    assert result.answer is None
    assert result.error is not None
    assert result.error.type == "model_timeout"
    assert result.error.message == "视觉客户端调用超时"
    assert fake.calls == 1


def test_unknown_exception_maps_to_model_error_and_is_logged(caplog):
    """未知异常仍为 model_error，且兜底分支会记录日志。"""
    fake = RaisingClient(RuntimeError(SENTINEL))

    with caplog.at_level(logging.ERROR, logger="image_understanding"):
        result = understand_image(make_png(), "图片里有什么？", fake)

    assert result.success is False
    assert result.answer is None
    assert result.error is not None
    assert result.error.type == "model_error"
    assert result.error.message == "视觉客户端调用失败"
    assert SENTINEL not in result.error.message
    assert any(record.levelno >= logging.ERROR for record in caplog.records)


def test_endpoint_returns_200_with_api_auth_error(monkeypatch):
    """接口层保持既有契约：业务错误仍为 HTTP 200 + 结构化 error。"""
    import main
    from fastapi.testclient import TestClient

    fake = RaisingClient(_status_error(AuthenticationError, 401))
    monkeypatch.setattr(main, "get_vision_client", lambda: fake, raising=False)

    client = TestClient(main.app)
    response = client.post(
        "/analyze-image",
        files={"file": ("auth.png", make_png(), "image/png")},
        data={"question": "图片里有什么？"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is False
    assert body["answer"] is None
    assert body["error"]["type"] == "api_auth_error"
    assert body["error"]["message"] == "视觉模型认证失败"
    assert SENTINEL not in body["error"]["message"]
    assert fake.calls == 1
