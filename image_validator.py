
from io import BytesIO

from PIL import Image, UnidentifiedImageError


# 图片大小上限：10 MiB
MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024

# 图片像素总数上限：33_554_432（8192x4096，约 33.5 MP）。
# 与字节上限互补：字节数不约束像素数，高压缩比图片可能在很小体积下声明极大尺寸。
MAX_IMAGE_PIXELS = 33_554_432


def validate_image_bytes(image_data: bytes) -> str | None:
    """验证图片输入、大小、格式及基本解码完整性。

    返回 None 表示验证通过；
    返回错误类型字符串表示验证失败。
    """
    if not isinstance(image_data, bytes):
        return "invalid_input"

    if not image_data:
        return "empty_image"

    if len(image_data) > MAX_IMAGE_SIZE_BYTES:
        return "image_too_large"

    try:
        with Image.open(BytesIO(image_data)) as image:
            if image.format not in ("PNG", "JPEG"):
                return "invalid_image"

            width, height = image.size

            if width <= 0 or height <= 0:
                return "invalid_image"

            if width * height > MAX_IMAGE_PIXELS:
                return "image_pixels_exceeded"

            image.load()

    except Image.DecompressionBombError:
        return "image_pixels_exceeded"

    except Image.DecompressionBombWarning:
        return "image_pixels_exceeded"

    except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
        # image.load() 在完整解码阶段失败时抛出异常：
        # - OSError：图像数据截断或解码失败（如 "image file is truncated"）；
        # - SyntaxError：Pillow 解析器判定数据流结构损坏；
        # - ValueError / UnidentifiedImageError：格式或内容非法。
        # 均属输入图片损坏，统一归入 invalid_image，避免异常逃逸到调用方。
        return "invalid_image"

    return None


def validate_question(question: str) -> str | None:
    """验证图片理解问题是否为非空、非纯空白字符串。

    返回 None 表示验证通过；
    返回错误类型字符串表示验证失败。
    """
    if not isinstance(question, str):
        return "invalid_input"

    if not question.strip():
        return "empty_question"

    return None
