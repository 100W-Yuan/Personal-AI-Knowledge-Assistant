
import logging

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)

from image_client import VisionClient, VisionClientTimeoutError
from image_result import ImageError, ImageResult
from image_validator import validate_image_bytes, validate_question

logger = logging.getLogger(__name__)


def understand_image(
    image_data: bytes,
    question: str,
    client: VisionClient | None = None,
) -> ImageResult:
    """验证输入并调用注入的视觉客户端。"""

    image_error = validate_image_bytes(image_data)

    if image_error is not None:
        return ImageResult(
            success=False,
            answer=None,
            error=ImageError(
                type=image_error,
                message="图片验证失败",
            ),
        )

    question_error = validate_question(question)

    if question_error is not None:
        return ImageResult(
            success=False,
            answer=None,
            error=ImageError(
                type=question_error,
                message="问题文本验证失败",
            ),
        )

    if client is None:
        return ImageResult(
            success=False,
            answer=None,
            error=ImageError(
                type="model_not_configured",
                message="尚未配置视觉模型客户端",
            ),
        )

    try:
        answer = client.analyze(image_data, question)

        if not isinstance(answer, str) or not answer.strip():
            return ImageResult(
                success=False,
                answer=None,
                error=ImageError(
                    type="invalid_model_response",
                    message="视觉客户端没有返回有效文本",
                ),
            )

        return ImageResult(
            success=True,
            answer=answer,
            error=None,
        )

    except VisionClientTimeoutError:
        return ImageResult(
            success=False,
            answer=None,
            error=ImageError(
                type="model_timeout",
                message="视觉客户端调用超时",
            ),
        )

    except AuthenticationError:
        return ImageResult(
            success=False,
            answer=None,
            error=ImageError(
                type="api_auth_error",
                message="视觉模型认证失败",
            ),
        )

    except RateLimitError:
        return ImageResult(
            success=False,
            answer=None,
            error=ImageError(
                type="api_rate_limited",
                message="视觉模型请求被限流",
            ),
        )

    except APITimeoutError:
        return ImageResult(
            success=False,
            answer=None,
            error=ImageError(
                type="model_timeout",
                message="视觉客户端调用超时",
            ),
        )

    except APIConnectionError:
        return ImageResult(
            success=False,
            answer=None,
            error=ImageError(
                type="api_connection_error",
                message="无法连接视觉模型服务",
            ),
        )

    except APIStatusError:
        return ImageResult(
            success=False,
            answer=None,
            error=ImageError(
                type="api_upstream_error",
                message="视觉模型服务返回错误",
            ),
        )

    except Exception:
        logger.exception("视觉客户端调用失败")
        return ImageResult(
            success=False,
            answer=None,
            error=ImageError(
                type="model_error",
                message="视觉客户端调用失败",
            ),
        )
