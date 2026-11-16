
import base64
from io import BytesIO
from typing import Protocol, runtime_checkable

from openai import APITimeoutError, OpenAI
from PIL import Image


class VisionClientTimeoutError(Exception):
    """视觉客户端超过允许等待时间时抛出的统一异常。"""


@runtime_checkable
class VisionClient(Protocol):
    """视觉模型客户端需要遵守的接口约定。"""

    def analyze(self, image_data: bytes, question: str) -> str:
        """分析图片并返回文本回答。"""
        ...


class DeepSeekVisionClient:
    """通过 OpenAI 兼容接口调用 DeepSeek 图像理解能力。"""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-flash",
        timeout: float = 30.0,
        sdk_client=None,
    ):
        # sdk_client 仅用于离线测试注入模拟对象。
        # 未传入时才创建 SDK 客户端，但不会立即发送请求。
        self.client = sdk_client or OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=0,
        )
        self.model = model

    def analyze(self, image_data: bytes, question: str) -> str:
        try:
            # 从图片实际内容识别格式，避免仅凭文件名判断。
            with Image.open(BytesIO(image_data)) as image:
                image_format = image.format

            mime_types = {
                "PNG": "image/png",
                "JPEG": "image/jpeg",
            }

            mime_type = mime_types.get(image_format)
            if mime_type is None:
                raise ValueError("当前客户端暂不支持此图片格式")

            encoded_image = base64.b64encode(image_data).decode("ascii")
            image_url = f"data:{mime_type};base64,{encoded_image}"

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question},
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url},
                            },
                        ],
                    }
                ],
            )

            content = response.choices[0].message.content

            if not isinstance(content, str):
                raise ValueError("视觉模型返回了非文本内容")

            return content

        except APITimeoutError as exc:
            raise VisionClientTimeoutError(
                "DeepSeek 视觉请求超时"
            ) from exc
