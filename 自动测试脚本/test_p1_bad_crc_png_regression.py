
"""P1 / R1 回归测试：坏 CRC PNG 不得让 SyntaxError 逃逸（纯离线，仅内存样本）。

背景（已实测确认的原始缺陷）：
    一个 70 字节、IDAT 块 CRC 错误的 2x2 PNG 在 `image_validator.validate_image_bytes()`
    内部触发 `PngImageFile.verify()` 抛出
    `SyntaxError: broken PNG file (bad header checksum in b'IDAT')`。
    修复前该异常不在 `except (UnidentifiedImageError, OSError, ValueError)` 元组内，
    因此直接逃逸到调用方（`image_understanding.py` 的校验调用位于 try 之外，
    最终会变成 HTTP 500）。修复后应统一映射为 `"invalid_image"`。

本文件的作用：
    1. 先证明样本确实能触发原始 SyntaxError（防止用普通无效字节冒充）；
    2. 再断言修复后 `validate_image_bytes()` 正常返回 `"invalid_image"`；
    3. 对有效 JPEG/PNG、像素上限、10 MiB 常量做不回归检查。

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

import image_validator
from image_validator import (
    MAX_IMAGE_PIXELS,
    MAX_IMAGE_SIZE_BYTES,
    validate_image_bytes,
)


PNG_MAGIC = bytes([137, 80, 78, 71, 13, 10, 26, 10])
BAD_CRC_MASK = 0xFFFFFFFF


def _png_chunk(kind: bytes, payload: bytes, bad_crc: bool = False) -> bytes:
    """构造一个 PNG 块；bad_crc=True 时故意写错 CRC。"""
    body = kind + payload
    crc = zlib.crc32(body) & 0xFFFFFFFF
    if bad_crc:
        crc ^= BAD_CRC_MASK
    return struct.pack(">I", len(payload)) + body + struct.pack(">I", crc)


def make_bad_crc_png() -> bytes:
    """构造已确认的问题样本：2x2、IDAT CRC 错误、约 70 字节。"""
    ihdr = struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0)
    # 1 个 filter 字节 + 4 个像素 × RGB 3 字节 = 13 字节扫描数据
    idat = zlib.compress(bytes([0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0]))
    return (
        PNG_MAGIC
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", idat, bad_crc=True)
        + _png_chunk(b"IEND", b"")
    )


def make_oversized_header_png(bad_crc: bool = False) -> bytes:
    """仅构造超限尺寸的图片头，不分配像素缓冲（应远小于 200 字节）。"""
    ihdr = struct.pack(">IIBBBBB", 10000, 10000, 8, 2, 0, 0, 0)
    return (
        PNG_MAGIC
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", b"", bad_crc=bad_crc)
        + _png_chunk(b"IEND", b"")
    )


def make_noise_image(image_format: str) -> bytes:
    """内存中生成 64x64 噪声图，避免纯色图压缩后过小。"""
    raw = random.Random(20261021).randbytes(64 * 64 * 3)
    image = Image.frombytes("RGB", (64, 64), raw)
    buffer = BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()


# ============================================================
# 1. 前置证明：样本确实触发原始 SyntaxError
# ============================================================


def test_sample_really_triggers_syntax_error_in_pillow():
    """证明样本不是普通无效字节：Pillow 的 verify() 确实抛 SyntaxError。

    该用例直接调用 Pillow，不经过生产代码，因此修复前后都应通过。
    它的作用是保证下面的回归测试使用的是真实触发样本。
    """
    sample = make_bad_crc_png()

    # 大小核对：确认与已确认的约 70 字节样本一致
    assert len(sample) < 200, f"样本过大，可能不是预期样本：{len(sample)} 字节"

    # 头部完好，Image.open 应成功
    with Image.open(BytesIO(sample)) as image:
        assert image.format == "PNG"
        assert image.size == (2, 2)

    # 但 verify() 必须抛出 SyntaxError —— 这是本缺陷的触发点
    with pytest.raises(SyntaxError):
        with Image.open(BytesIO(sample)) as image:
            image.verify()


# ============================================================
# 2. 核心回归：不得逃逸，必须返回 invalid_image
# ============================================================


def test_bad_crc_png_returns_invalid_image_without_raising():
    """修复目标：坏 CRC PNG 返回 "invalid_image"，不向调用方抛 SyntaxError。"""
    sample = make_bad_crc_png()

    try:
        result = validate_image_bytes(sample)
    except SyntaxError as exc:
        pytest.fail(f"SyntaxError 仍然逃逸到调用方：{exc}")

    assert result == "invalid_image"


# ============================================================
# 3. 不回归：有效 JPEG / PNG
# ============================================================


def test_valid_jpeg_still_accepted():
    """有效 JPEG 仍返回 None。"""
    assert validate_image_bytes(make_noise_image("JPEG")) is None


def test_valid_png_still_accepted():
    """有效 PNG 仍返回 None。"""
    assert validate_image_bytes(make_noise_image("PNG")) is None


# ============================================================
# 4. 不回归：像素上限，且仍早于 verify()
# ============================================================


def test_oversized_header_still_rejected_as_pixels_exceeded():
    """10000x10000 的轻量图片头仍被应用层像素上限拒绝，不分配像素缓冲。"""
    sample = make_oversized_header_png()

    assert len(sample) < 200, f"样本应保持轻量，实际 {len(sample)} 字节"
    assert 10000 * 10000 > MAX_IMAGE_PIXELS

    assert validate_image_bytes(sample) == "image_pixels_exceeded"


def test_pixel_check_still_runs_before_verify():
    """顺序保护：超限样本即使 CRC 也坏，仍应返回 image_pixels_exceeded。

    若像素检查被挪到 verify() 之后，verify() 会先抛 SyntaxError，
    修复后会被映射成 invalid_image —— 该用例随即失败，从而锁住顺序。
    """
    sample = make_oversized_header_png(bad_crc=True)

    assert validate_image_bytes(sample) == "image_pixels_exceeded"


# ============================================================
# 5. 常量护栏：10 MiB 与像素上限未被改动
# ============================================================


def test_limits_are_unchanged():
    """本轮不得改动两个上限常量。"""
    assert MAX_IMAGE_SIZE_BYTES == 10 * 1024 * 1024
    assert MAX_IMAGE_PIXELS == 33_554_432
    assert image_validator.MAX_IMAGE_SIZE_BYTES == MAX_IMAGE_SIZE_BYTES
    assert image_validator.MAX_IMAGE_PIXELS == MAX_IMAGE_PIXELS
