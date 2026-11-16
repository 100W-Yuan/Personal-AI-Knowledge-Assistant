
from io import BytesIO
from unittest.mock import patch
import pytest

from fastapi.testclient import TestClient
from PIL import Image

import main
from image_result import ImageError, ImageResult


client = TestClient(main.app)


def test_request_module_and_patch_target_are_the_same_object():
    """R6-3：处理请求的模块对象必须与 patch.object(main, ...) 的目标一致。

    若二者分裂（例如 sys.modules["main"] 被替换成另一个模块对象），
    字符串形式的 patch 会打到错误模块，Mock 静默失效而请求仍走真实实现。
    """
    endpoint = next(
        route.endpoint
        for route in main.app.routes
        if getattr(route, "path", None) == "/analyze-image"
    )

    assert endpoint is main.analyze_image
    assert client.app is main.app


def make_png() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (2, 2), color="red").save(buffer, format="PNG")
    return buffer.getvalue()


def test_existing_chat_route_still_exists():
    """确认原有文字聊天路由仍然存在。"""
    assert any(
        getattr(route, "path", None) == "/chat" and "POST" in getattr(route, "methods", ())
        for route in main.app.routes
    )


def test_image_api_returns_successful_result():
    """验证上传图片后，接口能够返回统一的成功结果。"""
    expected = ImageResult(
        success=True,
        answer="图片里有一只猫",
        error=None,
    )

    with patch.object(main, "understand_image", return_value=expected) as mock_understand:
        response = client.post(
            "/analyze-image",
            files={"file": ("test.png", make_png(), "image/png")},
            data={"question": "图片里有什么？"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "answer": "图片里有一只猫",
        "error": None,
    }
    mock_understand.assert_called_once()


def test_image_api_reports_invalid_image():
    """验证无效图片会返回明确错误，不调用真实模型。"""
    expected = ImageResult(
        success=False,
        answer=None,
        error=ImageError(
            type="invalid_image",
            message="图片验证失败",
        ),
    )

    with patch.object(main, "understand_image", return_value=expected) as mock_understand:
        response = client.post(
            "/analyze-image",
            files={"file": ("test.png", b"not-an-image", "image/png")},
            data={"question": "图片里有什么？"},
        )

    assert response.status_code == 200
    assert response.json()["success"] is False
    assert response.json()["answer"] is None
    assert response.json()["error"]["type"] == "invalid_image"
    mock_understand.assert_called_once()


def test_image_api_rejects_invalid_image_without_mock():
    """验证实际图片校验逻辑能拦截无效图片。"""
    response = client.post(
        "/analyze-image",
        files={"file": ("broken.png", b"not-an-image", "image/png")},
        data={"question": "图片里有什么？"},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["success"] is False
    assert result["answer"] is None
    assert result["error"]["type"] == "invalid_image"


def test_image_api_rejects_empty_file_without_mock():
    """验证实际图片校验逻辑能拦截空文件。"""
    response = client.post(
        "/analyze-image",
        files={"file": ("empty.png", b"", "image/png")},
        data={"question": "图片里有什么？"},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["success"] is False
    assert result["answer"] is None
    assert result["error"]["type"] == "empty_image"



@pytest.mark.parametrize("blank_question", ["", "   "])
def test_image_api_defaults_blank_question(blank_question):
    """验证空字符串与纯空白问题都会被替换为默认描述提示词。"""
    expected = ImageResult(
        success=True,
        answer="默认描述",
        error=None,
    )

    with patch.object(main, "understand_image", return_value=expected) as mock_understand:
        response = client.post(
            "/analyze-image",
            files={"file": ("test.png", make_png(), "image/png")},
            data={"question": blank_question},
        )

    assert response.status_code == 200
    assert response.json()["success"] is True
    assert mock_understand.call_args.kwargs["question"] == main.DEFAULT_IMAGE_QUESTION


def test_image_api_missing_question_uses_default():
    """验证缺失 question 字段不再返回 422，而是使用默认描述提示词。"""
    expected = ImageResult(
        success=True,
        answer="默认描述",
        error=None,
    )

    with patch.object(main, "understand_image", return_value=expected) as mock_understand:
        response = client.post(
            "/analyze-image",
            files={"file": ("test.png", make_png(), "image/png")},
        )

    assert response.status_code == 200
    assert response.json()["success"] is True
    assert mock_understand.call_args.kwargs["question"] == main.DEFAULT_IMAGE_QUESTION


def test_image_api_preserves_nonempty_question():
    """验证非空问题（含首尾空白）原样透传，不被替换或 strip。"""
    expected = ImageResult(
        success=True,
        answer="回答",
        error=None,
    )

    with patch.object(main, "understand_image", return_value=expected) as mock_understand:
        response = client.post(
            "/analyze-image",
            files={"file": ("test.png", make_png(), "image/png")},
            data={"question": "  图片里有什么？  "},
        )

    assert response.status_code == 200
    assert mock_understand.call_args.kwargs["question"] == "  图片里有什么？  "


def test_image_api_rejects_oversized_image():
    """验证超出暂定 10 MiB 限制的上传会被接口拒绝。"""
    from image_validator import MAX_IMAGE_SIZE_BYTES

    oversized_data = b"x" * (MAX_IMAGE_SIZE_BYTES + 1)

    response = client.post(
        "/analyze-image",
        files={
            "file": (
                "large.png",
                oversized_data,
                "image/png",
            )
        },
        data={"question": "图片里有什么？"},
    )

    assert response.status_code == 413
    assert response.json()["detail"]["type"] == "image_too_large"




def test_image_api_rejects_oversized_upload_before_image_analysis():
    """验证超大文件在进入图片理解模块前就被拒绝。"""
    from image_validator import MAX_IMAGE_SIZE_BYTES

    oversized_data = b"x" * (MAX_IMAGE_SIZE_BYTES + 1)

    with patch.object(main, "understand_image") as mock_understand:
        response = client.post(
            "/analyze-image",
            files={
                "file": (
                    "large.png",
                    oversized_data,
                    "image/png",
                )
            },
            data={"question": "图片里有什么？"},
        )

    assert response.status_code == 413
    assert response.json()["detail"]["type"] == "image_too_large"
    mock_understand.assert_not_called()


def test_image_api_accepts_normal_png_upload():
    """验证正常 PNG 上传仍会进入图片理解模块。"""
    expected = ImageResult


def test_image_api_accepts_normal_png_upload():
    """验证正常 PNG 上传仍会进入图片理解模块。"""
    expected = ImageResult(
        success=True,
        answer="这是一张测试图片",
        error=None,
    )

    with patch.object(
        main,
        "understand_image",
        return_value=expected,
    ) as mock_understand:
        response = client.post(
            "/analyze-image",
            files={
                "file": (
                    "test.png",
                    make_png(),
                    "image/png",
                )
            },
            data={"question": "请描述这张图片"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "answer": "这是一张测试图片",
        "error": None,
    }
    mock_understand.assert_called_once()
