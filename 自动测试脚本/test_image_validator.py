
import warnings

from io import BytesIO

from PIL import Image

import image_validator
from image_validator import (
    MAX_IMAGE_SIZE_BYTES,
    validate_image_bytes,
    validate_question,
)


def make_test_image(image_format: str) -> bytes:
    """在内存中生成有效的测试图片。"""
    buffer = BytesIO()
    image = Image.new("RGB", (2, 2), color="red")
    image.save(buffer, format=image_format)
    return buffer.getvalue()


# 图片验证测试


def test_empty_image_bytes():
    assert validate_image_bytes(b"") == "empty_image"


def test_string_input_is_rejected():
    assert validate_image_bytes("photo.png") == "invalid_input"


def test_invalid_image_bytes_are_rejected():
    assert validate_image_bytes(b"abc") == "invalid_image"


def test_fake_png_signature_is_rejected():
    fake_png = b"\x89PNG\r\n\x1a\n" + b"not a real image"
    assert validate_image_bytes(fake_png) == "invalid_image"


def test_valid_png_is_accepted():
    assert validate_image_bytes(make_test_image("PNG")) is None


def test_valid_jpeg_is_accepted():
    assert validate_image_bytes(make_test_image("JPEG")) is None


def test_image_over_size_limit_is_rejected_before_decoding():
    oversized_data = b"x" * (MAX_IMAGE_SIZE_BYTES + 1)
    assert validate_image_bytes(oversized_data) == "image_too_large"


def test_image_exactly_at_size_limit_is_not_rejected_for_size_alone():
    data_at_limit = b"x" * MAX_IMAGE_SIZE_BYTES
    assert validate_image_bytes(data_at_limit) == "invalid_image"


# 问题文本验证测试


def test_empty_question_is_rejected():
    assert validate_question("") == "empty_question"


def test_whitespace_only_question_is_rejected():
    assert validate_question("   \t\n") == "empty_question"


def test_non_string_question_is_rejected():
    assert validate_question(None) == "invalid_input"


def test_normal_question_is_accepted():
    assert validate_question("这张图片里有什么？") is None


def test_question_with_surrounding_spaces_is_accepted():
    assert validate_question("  分析这张图片  ") is None


# ============================================================
# 像素上限与解压炸弹异常
# ============================================================

def test_pixel_count_below_threshold_passes_pixel_rule(monkeypatch):
    """[实际执行行为] 像素数低于阈值时不被像素规则拒绝。

    2x2 图片共 4 像素；应用常量临时缩小到 5，仅用于验证比较方向。
    """
    monkeypatch.setattr(image_validator, "MAX_IMAGE_PIXELS", 5)

    assert validate_image_bytes(make_test_image("PNG")) is None


def test_pixel_count_equal_to_threshold_passes_pixel_rule(monkeypatch):
    """[实际执行行为] 像素数恰好等于阈值时不被像素规则拒绝。

    4 像素 == 阈值 4。断言通过，证明判定条件是严格大于而非大于等于。
    """
    monkeypatch.setattr(image_validator, "MAX_IMAGE_PIXELS", 4)

    assert validate_image_bytes(make_test_image("PNG")) is None


def test_pixel_count_over_threshold_is_rejected(monkeypatch):
    """[实际执行行为] 像素数超过阈值时返回 image_pixels_exceeded。

    4 像素 > 阈值 3，由应用层检查拦截。
    """
    monkeypatch.setattr(image_validator, "MAX_IMAGE_PIXELS", 3)

    assert validate_image_bytes(make_test_image("PNG")) == "image_pixels_exceeded"


def test_decompression_bomb_error_maps_to_pixels_exceeded(monkeypatch):
    """[实际执行行为] Pillow 抛 DecompressionBombError 时映射为 image_pixels_exceeded。

    把 Pillow 的 MAX_IMAGE_PIXELS 设为 1，2x2 图片的 4 像素超过 2*1，
    Image.open 内部抛出 DecompressionBombError。应用层常量保持真实值
    （33_554_432），不会先行拦截，因此命中的是新增的 except 分支。
    """
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1)

    assert validate_image_bytes(make_test_image("PNG")) == "image_pixels_exceeded"


def test_decompression_bomb_warning_as_error_maps_to_pixels_exceeded(monkeypatch):
    """[实际执行行为] 警告被转为异常时，DecompressionBombWarning 映射为 image_pixels_exceeded。

    Pillow 的 MAX_IMAGE_PIXELS 设为 2：4 像素未超过 2*2（不抛 Error），
    但超过 2（发出 Warning）。测试内把该警告临时转为异常，验证对应
    except 分支可达。警告过滤仅在 with 块内生效，退出即恢复。
    """
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 2)

    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        result = validate_image_bytes(make_test_image("PNG"))

    assert result == "image_pixels_exceeded"


def test_gif_format_is_rejected_as_unsupported():
    """[实际执行行为] 有效 GIF 因不在支持格式内被拒绝。"""
    assert validate_image_bytes(make_test_image("GIF")) == "invalid_image"


def test_bmp_format_is_rejected_as_unsupported():
    """[实际执行行为] 有效 BMP 因不在支持格式内被拒绝。"""
    assert validate_image_bytes(make_test_image("BMP")) == "invalid_image"


def test_corrupt_image_still_maps_to_invalid_image():
    """[实际执行行为] 普通损坏图片仍映射为 invalid_image，不受新增像素逻辑影响。"""
    assert validate_image_bytes(b"this is not an image") == "invalid_image"


def test_app_pixel_limit_rejects_under_normal_warning_mode(monkeypatch):
    """[实际执行行为] 普通警告态下，应用层像素上限独立生效。

    与 test_decompression_bomb_warning_as_error_... 的区别：本用例
    不把 DecompressionBombWarning 转为异常，也不修改 Pillow 的
    MAX_IMAGE_PIXELS。2x2 图只有 4 像素，远低于 Pillow 默认阈值，
    因此 Pillow 既不会抛错也不会发警告；拒绝完全来自应用层常量。

    与 test_pixel_count_over_threshold_is_rejected 的区别：该用例只断言
    返回码，本用例额外用 record=True 观察警告流，断言没有任何
    DecompressionBombWarning 产生，从而证明 Pillow 炸弹机制未介入。
    """
    monkeypatch.setattr(image_validator, "MAX_IMAGE_PIXELS", 3)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", Image.DecompressionBombWarning)
        result = validate_image_bytes(make_test_image("PNG"))

    assert result == "image_pixels_exceeded"

    bomb_warnings = [
        item
        for item in caught
        if issubclass(item.category, Image.DecompressionBombWarning)
    ]
    assert bomb_warnings == []
