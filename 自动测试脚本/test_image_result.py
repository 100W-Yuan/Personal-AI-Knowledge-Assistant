
import pytest
import sys
from pathlib import Path

# Make the project root importable when this test file is run from its subfolder.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from image_result import ImageError, ImageResult  # pyright: ignore[reportMissingImports]


def test_success_result():
    result = ImageResult(
        success=True,
        answer="图片里有一只猫",
        error=None,
    )

    assert result.success is True
    assert result.answer == "图片里有一只猫"
    assert result.error is None


def test_failure_result():
    error = ImageError(
        type="model_timeout",
        message="图片识别请求超时",
    )

    result = ImageResult(
        success=False,
        answer=None,
        error=error,
    )

    assert result.success is False
    assert result.answer is None
    assert result.error.type == "model_timeout"


def test_failure_result_rejects_answer():
    with pytest.raises(ValueError, match="失败结果的 answer 必须为 None"):
        ImageResult(
            success=False,
            answer="图片里有一只猫",
            error=None,
        )


def test_success_result_rejects_error():
    error = ImageError(
        type="model_error",
        message="模型请求失败",
    )

    with pytest.raises(ValueError, match="成功结果不能包含错误信息"):
        ImageResult(
            success=True,
            answer="图片里有一只猫",
            error=error,
        )


def test_failure_result_requires_error():
    with pytest.raises(ValueError, match="失败结果必须包含 ImageError"):
        ImageResult(
            success=False,
            answer=None,
            error=None,
        )
def test_image_error_rejects_non_string_type():
    with pytest.raises(TypeError, match="错误类型必须是字符串"):
        ImageError(type=123, message="模型请求失败")


def test_image_error_rejects_non_string_message():
    with pytest.raises(TypeError, match="错误说明必须是字符串"):
        ImageError(type="model_error", message=123)


def test_image_error_rejects_empty_type():
    with pytest.raises(ValueError, match="错误类型不能为空"):
        ImageError(type="   ", message="模型请求失败")


def test_image_error_rejects_empty_message():
    with pytest.raises(ValueError, match="错误说明不能为空"):
        ImageError(type="model_error", message="   ")