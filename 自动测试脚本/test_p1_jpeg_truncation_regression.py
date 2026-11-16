
"""P1 遗留风险补测 B：JPEG / PNG 截断图片的实际验证行为（纯离线）。

本文件只做两件事：
    1. 用项目现有的 `validate_image_bytes()` 观测四类样本的实际返回值；
    2. 记录截断 JPEG 在完整解码（`.load()`）时的真实结果，作为对照。

样本生成方式（全部在内存中，不写任何图片文件）：
    - 用固定随机种子生成 64x64 RGB 噪声图，避免纯色图被压缩到极小后
      截断点落不到扫描数据里；
    - JPEG：定位 SOS 标记 0xFFDA，在其后保留剩余字节的 50%，确保
      截断点位于熵编码扫描数据内部（而不仅是文件尾部）；
    - PNG：定位 IDAT 块，在其后保留剩余字节的 50%，确保截断点位于
      图像数据块内部。

重要说明：
    本文件为**观测型**测试。截断样本的验证结果可能是 None（缺口复现）
    也可能是 "invalid_image"（被现有校验拦截），两者都属有效观测。
    因此对截断样本只断言"返回值落在两种可能之一"，并在运行输出中打印
    实际值，由报告给出结论；不预设缺口必然存在。
"""

import sys
import types

# --- 兜底：确保导入生产模块前 dotenv 已被替换为空操作模块 ---
if "dotenv" not in sys.modules:
    _fake_dotenv = types.ModuleType("dotenv")
    _fake_dotenv.load_dotenv = lambda *args, **kwargs: False
    sys.modules["dotenv"] = _fake_dotenv

import random
from io import BytesIO

from PIL import Image

from image_validator import validate_image_bytes


WIDTH = 64
HEIGHT = 64
SEED = 20261021
KEEP_RATIO = 0.5

# JPEG 的 SOS（Start Of Scan）标记；PNG 的图像数据块标识
JPEG_SCAN_MARKER = b"\xff\xda"
PNG_DATA_MARKER = b"IDAT"


def _image_bytes(image_format: str) -> bytes:
    """在内存中生成噪声图，返回指定格式的字节。"""
    raw = random.Random(SEED).randbytes(WIDTH * HEIGHT * 3)
    image = Image.frombytes("RGB", (WIDTH, HEIGHT), raw)
    buffer = BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()


def _truncate_inside_payload(data: bytes, marker: bytes) -> tuple[bytes, dict]:
    """在 marker 之后保留部分字节，使截断点落在图像数据内部。

    返回 (截断字节, 元信息)。元信息用于在报告中记录生成方式。
    """
    marker_offset = data.find(marker)
    assert marker_offset > 0, f"未找到标记 {marker!r}，无法构造有意义的截断样本"

    remaining = len(data) - marker_offset
    cut_offset = marker_offset + max(16, int(remaining * KEEP_RATIO))
    assert cut_offset < len(data), "截断点必须位于文件内部"

    meta = {
        "original_length": len(data),
        "marker_offset": marker_offset,
        "cut_offset": cut_offset,
        "truncated_length": cut_offset,
        "removed_bytes": len(data) - cut_offset,
    }
    return data[:cut_offset], meta


VALID_JPEG = _image_bytes("JPEG")
TRUNCATED_JPEG, JPEG_META = _truncate_inside_payload(VALID_JPEG, JPEG_SCAN_MARKER)

VALID_PNG = _image_bytes("PNG")
TRUNCATED_PNG, PNG_META = _truncate_inside_payload(VALID_PNG, PNG_DATA_MARKER)


# ============================================================
# 1-4：四类样本的验证结果
# ============================================================


def test_complete_jpeg_is_accepted():
    """完整有效 JPEG 应通过验证（作为截断样本的对照）。"""
    assert validate_image_bytes(VALID_JPEG) is None


def test_truncated_jpeg_validation_result_is_recorded():
    """截断 JPEG：记录 validate_image_bytes 的实际返回值。

    截断方式见模块文档；此处先确认样本确实被截断、且截断点位于 SOS 之后，
    再断言返回值落在「放行」或「invalid_image」两种可能之内。
    """
    result = validate_image_bytes(TRUNCATED_JPEG)

    print(f"[OBSERVE] validate_image_bytes(truncated_jpeg) -> {result!r}")
    print(f"[OBSERVE] jpeg_meta -> {JPEG_META}")

    assert JPEG_META["cut_offset"] > JPEG_META["marker_offset"]
    assert len(TRUNCATED_JPEG) < len(VALID_JPEG)
    assert JPEG_META["removed_bytes"] > 0

    # 头部完好：截断样本仍能被 Image.open 打开（这是缺口讨论的前提）
    with Image.open(BytesIO(TRUNCATED_JPEG)) as image:
        assert image.format == "JPEG"

    assert result in (None, "invalid_image")


def test_complete_png_is_accepted():
    """完整有效 PNG 应通过验证。"""
    assert validate_image_bytes(VALID_PNG) is None


def test_truncated_png_validation_result_is_recorded():
    """截断 PNG：记录 validate_image_bytes 的实际返回值（与 JPEG 对照）。"""
    result = validate_image_bytes(TRUNCATED_PNG)

    print(f"[OBSERVE] validate_image_bytes(truncated_png) -> {result!r}")
    print(f"[OBSERVE] png_meta -> {PNG_META}")

    assert PNG_META["cut_offset"] > PNG_META["marker_offset"]
    assert len(TRUNCATED_PNG) < len(VALID_PNG)

    assert result in (None, "invalid_image")


# ============================================================
# 5：截断 JPEG 的完整解码对照
# ============================================================


def test_truncated_jpeg_full_decode_outcome_is_recorded():
    """对截断 JPEG 执行 Image.open().load()，记录是否抛出异常。

    本用例为**观测型**：不断言 Pillow 的具体异常类型，
    只要求得到一个明确结果（异常类型名或 no_exception）供报告引用。
    """
    outcome = "no_exception"
    detail = ""

    try:
        with Image.open(BytesIO(TRUNCATED_JPEG)) as image:
            image.load()
    except Exception as exc:  # noqa: BLE001 - 观测目的，需捕获全部异常类型
        outcome = type(exc).__name__
        detail = str(exc)

    print(f"[OBSERVE] truncated_jpeg.load() -> {outcome}: {detail}")
    print(f"[OBSERVE] validate_image_bytes same sample -> "
          f"{validate_image_bytes(TRUNCATED_JPEG)!r}")

    assert isinstance(outcome, str) and outcome
    assert len(TRUNCATED_JPEG) < len(VALID_JPEG)


def test_truncated_png_full_decode_outcome_is_recorded():
    """对截断 PNG 执行 Image.open().load()，作为 JPEG 的对照观测。"""
    outcome = "no_exception"
    detail = ""

    try:
        with Image.open(BytesIO(TRUNCATED_PNG)) as image:
            image.load()
    except Exception as exc:  # noqa: BLE001 - 观测目的，需捕获全部异常类型
        outcome = type(exc).__name__
        detail = str(exc)

    print(f"[OBSERVE] truncated_png.load() -> {outcome}: {detail}")

    assert isinstance(outcome, str) and outcome
    assert len(TRUNCATED_PNG) < len(VALID_PNG)
