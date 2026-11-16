
"""R3 回归测试：截断 JPEG 必须被图片校验拒绝（纯离线，仅内存样本）。

背景：
    修复前 `validate_image_bytes()` 使用 `image.verify()`。Pillow 中
    `JpegImageFile` 未重写 `verify`，继承的是 `ImageFile.ImageFile.verify`
    （仅关闭文件句柄，不做任何校验），因此截断 JPEG 会被放行；而在
    同一份字节上完整解码会抛 `OSError: image file is truncated`。
    修复后改为 `image.load()`，强制完整解码。

关于"不得在同对象上 verify 后 load"：
    Pillow 在 `verify()` 之后再调用 `load()` 会抛 `AssertionError`。
    本文件的 T1/T2（有效 JPEG/PNG 返回 None）即为此约束的行为级守卫：
    若实现误写为"保留 verify 再追加 load"，这两项会立刻失败。

样本参数（与已知问题样本一致，Pillow 13.0.0 / Python 3.12.10）：
    64x64 RGB 噪声，随机种子 20261021；
    JPEG 原始 3097 字节，SOS 偏移 609，截断至 1853 字节；
    PNG 原始 12420 字节，IDAT 偏移 37，截断至 6228 字节。

样本全部在内存中构造，不写入任何图片文件。

安全说明：
    本文件不导入 `main`/`image_client`，不构造任何 OpenAI 客户端，不访问网络。
    正式运行时由外部隔离启动脚本注入假 dotenv 与空壳 openai.OpenAI。
"""

import sys
import types

# --- 兜底：确保导入生产模块前 dotenv 已被替换为空操作模块 ---
if "dotenv" not in sys.modules:
    _fake_dotenv = types.ModuleType("dotenv")
    _fake_dotenv.load_dotenv = lambda *args, **kwargs: False
    sys.modules["dotenv"] = _fake_dotenv

import random
import struct
import zlib
from io import BytesIO

import pytest
from PIL import Image

from image_validator import (
    MAX_IMAGE_PIXELS,
    MAX_IMAGE_SIZE_BYTES,
    validate_image_bytes,
)


PNG_MAGIC = bytes([137, 80, 78, 71, 13, 10, 26, 10])
JPEG_SOS = bytes([255, 218])
PNG_IDAT = b"IDAT"

WIDTH = 64
HEIGHT = 64
SEED = 20261021
KEEP_RATIO = 0.5


def make_noise_image(image_format: str) -> bytes:
    """内存中生成 64x64 RGB 噪声图，返回指定格式字节。"""
    raw = random.Random(SEED).randbytes(WIDTH * HEIGHT * 3)
    image = Image.frombytes("RGB", (WIDTH, HEIGHT), raw)
    buffer = BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()


def truncate_inside_payload(data: bytes, marker: bytes) -> tuple[bytes, dict]:
    """在 marker 之后保留剩余字节的 50%，使截断点落在图像数据内部。"""
    offset = data.find(marker)
    assert offset > 0, f"未找到标记 {marker!r}"

    remaining = len(data) - offset
    cut_offset = offset + max(16, int(remaining * KEEP_RATIO))

    return data[:cut_offset], {
        "original_length": len(data),
        "marker_offset": offset,
        "cut_offset": cut_offset,
    }


def _png_chunk(kind: bytes, payload: bytes, bad_crc: bool = False) -> bytes:
    body = kind + payload
    crc = zlib.crc32(body) & 0xFFFFFFFF
    if bad_crc:
        crc ^= 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + body + struct.pack(">I", crc)


def make_bad_crc_png() -> bytes:
    """70 字节、2x2、IDAT CRC 故意的坏样本（R1 的问题样本）。"""
    ihdr = struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0)
    idat = zlib.compress(bytes([0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0]))
    return (
        PNG_MAGIC
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", idat, bad_crc=True)
        + _png_chunk(b"IEND", b"")
    )


def make_oversized_header_png(bad_crc: bool = False) -> bytes:
    """10000x10000 声明尺寸的轻量图片头，不分配像素缓冲。"""
    ihdr = struct.pack(">IIBBBBB", 10000, 10000, 8, 2, 0, 0, 0)
    return (
        PNG_MAGIC
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", b"", bad_crc=bad_crc)
        + _png_chunk(b"IEND", b"")
    )


# ============================================================
# T1 / T2：有效图片不回归（同时守卫"不得 verify 后 load"）
# ============================================================


def test_valid_jpeg_is_accepted():
    """有效 JPEG 仍返回 None。

    若实现误为"先 verify() 再 load()"，此处会抛 AssertionError 而失败。
    """
    assert validate_image_bytes(make_noise_image("JPEG")) is None


def test_valid_png_is_accepted():
    """有效 PNG 仍返回 None（同上，守卫 verify→load 误实现）。"""
    assert validate_image_bytes(make_noise_image("PNG")) is None


# ============================================================
# T3：已知截断 JPEG 必须被拒绝（本轮验收标志）
# ============================================================


def test_known_truncated_jpeg_is_rejected():
    """复现已知样本（3097 → 1853 字节，SOS@609），断言返回 invalid_image。"""
    original = make_noise_image("JPEG")
    truncated, meta = truncate_inside_payload(original, JPEG_SOS)

    # 样本核对：确保使用的就是已确认的问题样本，而非新造的其他样本
    assert meta["original_length"] == 3097, meta
    assert meta["marker_offset"] == 609, meta
    assert meta["cut_offset"] == 1853, meta
    assert len(truncated) == 1853

    # 头部完好：截断样本仍可被 Image.open 打开（缺口成立的前提）
    with Image.open(BytesIO(truncated)) as image:
        assert image.format == "JPEG"

    # 对照：同一份字节完整解码会失败，说明截断确实落在扫描数据内
    with pytest.raises(OSError):
        with Image.open(BytesIO(truncated)) as image:
            image.load()

    # 生产行为：修复后必须拒绝
    assert validate_image_bytes(truncated) == "invalid_image"


# ============================================================
# T4 / T5：截断 PNG 与坏 CRC PNG 不回归
# ============================================================


def test_truncated_png_is_rejected():
    """截断 PNG 仍返回 invalid_image。"""
    original = make_noise_image("PNG")
    truncated, meta = truncate_inside_payload(original, PNG_IDAT)

    assert meta["original_length"] == 12420, meta
    assert meta["marker_offset"] == 37, meta
    assert meta["cut_offset"] == 6228, meta

    assert validate_image_bytes(truncated) == "invalid_image"


def test_bad_crc_png_returns_invalid_image_without_syntax_error_escape():
    """70 字节坏 CRC PNG 仍返回 invalid_image，SyntaxError 不得逃逸（R1 守卫）。"""
    sample = make_bad_crc_png()

    assert len(sample) == 70, f"样本应为已知的 70 字节，实际 {len(sample)}"

    try:
        result = validate_image_bytes(sample)
    except SyntaxError as exc:
        pytest.fail(f"SyntaxError 逃逸到调用方：{exc}")

    assert result == "invalid_image"


# ============================================================
# T6 / T7：像素上限仍早于完整解码
# ============================================================


def test_oversized_header_rejected_as_pixels_exceeded():
    """10000x10000 声明尺寸的轻量图片头仍被像素上限拒绝。"""
    sample = make_oversized_header_png()

    assert len(sample) < 200, f"样本应保持轻量，实际 {len(sample)} 字节"
    assert 10000 * 10000 > MAX_IMAGE_PIXELS

    assert validate_image_bytes(sample) == "image_pixels_exceeded"


def test_oversized_bad_crc_header_still_rejected_as_pixels_exceeded():
    """超限且 CRC 损坏时仍返回 image_pixels_exceeded，证明像素判断早于解码。

    若顺序反转（先 load 再判像素），该样本会因 CRC 损坏先返回 invalid_image，
    本用例随即失败。
    """
    sample = make_oversized_header_png(bad_crc=True)

    assert validate_image_bytes(sample) == "image_pixels_exceeded"


# ============================================================
# T8：文件大小上限与既有错误分类不回归
# ============================================================


def test_size_limit_and_existing_error_classifications_unchanged():
    """10 MiB 上限与既有错误分类保持不变。"""
    assert MAX_IMAGE_SIZE_BYTES == 10 * 1024 * 1024
    assert MAX_IMAGE_PIXELS == 33_554_432

    # 超限 1 字节：必须先被大小规则拒绝，不进入解码
    oversized = b"x" * (MAX_IMAGE_SIZE_BYTES + 1)
    assert validate_image_bytes(oversized) == "image_too_large"

    # 其他既有分类
    assert validate_image_bytes(b"") == "empty_image"
    assert validate_image_bytes("photo.png") == "invalid_input"
    assert validate_image_bytes(b"not an image") == "invalid_image"
    assert validate_image_bytes(b"this is not an image") == "invalid_image"
