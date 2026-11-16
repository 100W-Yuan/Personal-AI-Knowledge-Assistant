"""
my-ai 25 图片理解真实 API 冒烟测试脚本（仅创建，未经授权不得执行）

安全约束：
- 本脚本创建本身不会调用 API；只有运行脚本才会发起一次真实请求。
- 运行前必须由用户另行明确授权 API 调用。
- 不读取、打印或修改正式 memory.db。
- 不启动服务，不启动 Ollama，不执行 Git 操作。
- 不写日志文件；只输出必要的脱敏元数据。
- max_retries=0 且禁用重定向，尽量限制为一次 SDK 请求尝试；无法保证硬性费用上限。
- 图片文件只读取用于构造请求，不会复制或修改图片。
"""

from __future__ import annotations

import base64
import os
import sys
import time
from pathlib import Path
from typing import Any

# 用户此前提供的图片路径；执行前会检查文件存在性和大小。
IMAGE_PATH = Path(r"C:\Users\Doro\Desktop\屏幕截图 2026-04-08 213432.png")
MAX_IMAGE_BYTES = 200 * 1024
QUESTION = "请简要描述图片中可见的主要内容。"
MODEL = "deepseek-flash"
BASE_URL = "https://api.deepseek.com"
TIMEOUT_SECONDS = 60.0

# 安全开关：默认关闭。只有用户明确授权真实 API 调用后，才可自行改为 True。
ALLOW_REAL_API_CALL = False


class RecordingProxy:
    """代理 SDK 调用，保留响应对象以读取 usage；不打印请求或响应正文。"""

    def __init__(self, sdk_client: Any) -> None:
        self._sdk_client = sdk_client
        self.last_response: Any = None
        self.create_calls = 0

        class _Completions:
            def __init__(proxy_self: Any, outer: "RecordingProxy") -> None:
                proxy_self._outer = outer

            def create(proxy_self: Any, **kwargs: Any) -> Any:
                proxy_self._outer.create_calls += 1
                response = proxy_self._outer._sdk_client.chat.completions.create(**kwargs)
                proxy_self._outer.last_response = response
                return response

        class _Chat:
            def __init__(proxy_self: Any, outer: "RecordingProxy") -> None:
                proxy_self.completions = _Completions(outer)

        self.chat = _Chat(self)


def main() -> int:
    print("图片理解真实 API 冒烟测试")
    print(f"图片路径参数：{IMAGE_PATH}")
    print("模式：真实请求脚本；默认安全开关关闭。")

    if not ALLOW_REAL_API_CALL:
        print("已停止：ALLOW_REAL_API_CALL=False。未调用 API。")
        print("只有在获得单独的明确 API 调用授权后，才可人工修改开关并运行。")
        return 2

    if not IMAGE_PATH.is_file():
        print("错误：指定图片不存在或不是普通文件。")
        return 3

    size = IMAGE_PATH.stat().st_size
    print(f"图片大小：{size} bytes")
    if size <= 0 or size > MAX_IMAGE_BYTES:
        print(f"错误：图片必须大于 0 且不超过 {MAX_IMAGE_BYTES} bytes。")
        return 4

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("错误：环境变量 DEEPSEEK_API_KEY 未设置。未调用 API。")
        return 5

    # 运行依赖检查不进行安装，也不会读取或显示密钥。
    try:
        from openai import OpenAI, DefaultHttpxClient
    except Exception as exc:
        print(f"错误：无法导入 OpenAI SDK（{type(exc).__name__}）。未调用 API。")
        return 6

    request_count = 0
    response_status: int | None = None
    response_request_id: str | None = None
    response_hook_errors = 0
