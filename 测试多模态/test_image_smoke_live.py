"""
my-ai 25 图片理解真实 API 冒烟测试（独立脚本）
默认安全状态：ALLOW_REAL_API_CALL = False。
只有在本地源码接口、图片路径、依赖和密钥存在性均核验后，
并且用户已授权真实 API 请求，才可将开关改为 True 并单独运行。
本脚本不启动 Web 服务，不读写 memory.db，不操作 Ollama，不写日志文件。
尽力限制为一次 SDK create 尝试；不能保证服务端绝对费用上限。
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(r"C:\Users\Doro\Desktop\my-ai 25")

IMAGE_PATH = Path(r"C:\Users\Doro\Desktop\屏幕截图 2026-04-08 213432.png")

MAX_IMAGE_BYTES = 10 * 1024 * 1024

QUESTION = "请简要描述图片中可见的主要内容。"

MODEL = "deepseek-flash"

BASE_URL = "https://api.deepseek.com"

TIMEOUT_SECONDS = 30.0  # 与报告中的生产客户端默认超时保持一致
# 必须先完成本地核对，再由用户明确确认后才可手动改为 True。

ALLOW_REAL_API_CALL = False

class RecordingSDKProxy:
    """只代理 chat.completions.create，保留响应元数据，不输出请求/响应正文。"""

    def __init__(self, sdk_client: Any) -> None:
        self._sdk_client = sdk_client
        self.last_response: Any = None
        self.create_calls = 0
        class CompletionsProxy:
            def __init__(proxy_self: Any, outer: "RecordingSDKProxy") -> None:
                proxy_self.outer = outer
            def create(proxy_self: Any, **kwargs: Any) -> Any:
                proxy_self.outer.create_calls += 1
                if proxy_self.outer.create_calls > 1:
                    raise RuntimeError("安全闸触发：拒绝第二次 SDK create 调用")
                response = proxy_self.outer._sdk_client.chat.completions.create(**kwargs)
                proxy_self.outer.last_response = response
                return response
        class ChatProxy:
            def __init__(proxy_self: Any, outer: "RecordingSDKProxy") -> None:
                proxy_self.completions = CompletionsProxy(outer)
        self.chat = ChatProxy(self)
# 四种结果状态，必须严格区分，禁止互相替代。

STATE_LOCAL_PRECHECK = "local_precheck_failed"          # 本地预检失败，未开始请求

STATE_ATTEMPTED_UNKNOWN = "attempted_evidence_missing"  # 已尝试，但无足够证据判断服务端是否处理

STATE_HTTP_RESPONSE = "http_response_received"          # 已收到 HTTP 响应

STATE_SUCCESS = "success_nonempty_text"                 # 成功且返回非空文本
# 传输层异常类名标记：用于判断失败发生在哪个阶段。
# PoolTimeout 是连接池资源等待超时，不等于 TCP 连接建立失败，必须单独归类为 pool_phase。

CONNECT_PHASE_MARKERS = ("ConnectError", "ConnectTimeout")

POOL_PHASE_MARKERS = ("PoolTimeout",)

SENT_PHASE_MARKERS = ("ReadTimeout", "ReadError", "RemoteProtocolError")

WRITE_PHASE_MARKERS = ("WriteTimeout", "WriteError")

def _exception_chain(exc: BaseException, limit: int = 6) -> list[BaseException]:
    """展开 __cause__/__context__ 链，带去重与深度上限，避免环状引用。"""
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and len(chain) < limit and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        nxt = current.__cause__ or current.__context__
        current = nxt if isinstance(nxt, BaseException) else None
    return chain

def _chain_names(exc: BaseException) -> list[str]:
    return [type(item).__name__ for item in _exception_chain(exc)]

def _chain_text(exc: BaseException) -> str:
    """只输出异常类型链，绝不输出异常正文（可能含 body/密钥/请求片段）。"""
    return ">".join(_chain_names(exc))

def _transport_phase(exc: BaseException) -> str:
    """判断失败阶段。
    pool_phase：连接池资源等待超时（未占用到连接）
    connect_phase：TCP/TLS 连接未建立
    after_send：请求已写出但未收回完整响应
    write_phase：请求可能已部分写出
    phase_unknown：无法判定
    无论返回哪个阶段，只要没有 HTTP 响应证据，服务端是否处理一律记为未知。
    """
    names = _chain_names(exc)
    if any(marker in names for marker in POOL_PHASE_MARKERS):
        return "pool_phase"
    if any(marker in names for marker in CONNECT_PHASE_MARKERS):
        return "connect_phase"
    if any(marker in names for marker in SENT_PHASE_MARKERS):
        return "after_send"
    if any(marker in names for marker in WRITE_PHASE_MARKERS):
        return "write_phase"
    return "phase_unknown"

def main() -> int:
    parser = argparse.ArgumentParser(
        description="my-ai 25 图片理解真实 API 冒烟测试（独立脚本）"
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=IMAGE_PATH,
        help=f"待测试图片路径（默认：{IMAGE_PATH}）",
    )
    args = parser.parse_args()
    image_path = args.image

    print("图片理解真实 API 冒烟测试")
    print(f"图片路径：{image_path}")

    if not ALLOW_REAL_API_CALL:
        print("停止：ALLOW_REAL_API_CALL=False；未读取密钥、未构造客户端、未调用 API。")
        return 2

    print(f"项目目录存在：{PROJECT_ROOT.is_dir()}")
    if not PROJECT_ROOT.is_dir():
        print("停止：项目目录不存在。")
        return 3

    if not image_path.is_file():
        print("停止：图片不存在或不是普通文件。")
        return 4

    image_size = image_path.stat().st_size
    print(f"图片大小：{image_size} bytes")
    if image_size <= 0 or image_size > MAX_IMAGE_BYTES:
        print(f"停止：图片必须大于 0 且不超过 {MAX_IMAGE_BYTES} bytes。")
        return 5

    # 在构造任何客户端之前读取原始字节；失败即属于本地预检失败。
    try:
        image_data = image_path.read_bytes()
    except OSError as exc:
        print("状态：local_precheck_failed")
        print("原因：image_read_failed")
        print(f"异常类型：{type(exc).__name__}")
        print("提示：未构造 SDK 客户端，未发起任何请求。")
        return 8

    if len(image_data) != image_size:
        print("状态：local_precheck_failed")
        print("原因：image_size_changed_between_stat_and_read")
        print("提示：未构造 SDK 客户端，未发起任何请求。")
        return 8

    # 图片本地校验必须先于 API Key 检查和 SDK 客户端构造。
    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        from image_validator import validate_image_bytes
    except Exception as exc:
        print(f"停止：图片校验器导入失败（{type(exc).__name__}）；未调用 API。")
        return 7

    try:
        validation_error = validate_image_bytes(image_data)
    except Exception as exc:
        print("状态：local_precheck_failed")
        print("原因：image_validation_failed")
        print(f"校验异常类型：{type(exc).__name__}")
        print("提示：图片校验未通过；未检查 API Key，未构造 SDK 客户端，未发起请求。")
        return 13

    if validation_error is not None:
        print("状态：local_precheck_failed")
        print(f"原因：image_validation_failed:{validation_error}")
        print("提示：未构造 SDK 客户端，未发起任何请求。")
        return 13

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("停止：DEEPSEEK_API_KEY 未设置；未调用 API。")
        return 6

    # 图片通过本地校验且确认密钥存在后，才导入 SDK 与生产客户端。
    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        import httpx2
        from openai import (
            APIConnectionError,
            APIStatusError,
            DefaultHttpxClient,
            OpenAI,
        )
        from image_client import DeepSeekVisionClient, VisionClientTimeoutError
    except Exception as exc:
        print(f"停止：导入失败（{type(exc).__name__}）；未调用 API。")
        return 7

    # 三层计数：SDK create 次数（代理内）、HTTP 请求尝试次数、HTTP 响应次数（事件钩子）。
    counters: dict[str, Any] = {
        "http_request": 0,
        "http_response": 0,
        "http_status": None,
        "http_request_id": None,
    }

    def on_request(request: Any) -> None:
        """发送前触发：只代表客户端尝试发送，不能证明服务端已收到。"""
        counters["http_request"] += 1
        if counters["http_request"] > 1:
            # 第二道闩锁：即便 SDK 内部重试或发生重定向，也在传输层硬中断。
            raise RuntimeError("安全闸触发：拒绝第二次 HTTP 请求尝试")

    def on_response(response: Any) -> None:
        """收到响应头后触发：这是"服务端已处理"的可靠证据来源。"""
        counters["http_response"] += 1
        try:
            counters["http_status"] = int(response.status_code)
        except Exception:
            counters["http_status"] = None
        try:
            counters["http_request_id"] = response.headers.get("x-request-id")
        except Exception:
            counters["http_request_id"] = None
        if counters["http_response"] > 1:
            raise RuntimeError("安全闸触发：收到第二次 HTTP 响应")

    # 四项超时显式声明：连接 10s、读取 30s、写入 30s、连接池 10s。
    http_timeout = httpx2.Timeout(
        connect=10.0,
        read=TIMEOUT_SECONDS,
        write=TIMEOUT_SECONDS,
        pool=10.0,
    )

    # 关键：OpenAI 若收到 http_client，其自身 timeout 形参会被忽略

    # （openai/_base_client.py: self._client = http_client or SyncHttpxClientWrapper(timeout=...)），

    # 因此超时、禁止重定向、事件钩子必须全部设置在真实 HTTP 客户端上。
    http_client = DefaultHttpxClient(
        timeout=http_timeout,
        follow_redirects=False,
        event_hooks={"request": [on_request], "response": [on_response]},
    )
    sdk_client = OpenAI(
        api_key=api_key,
        base_url=BASE_URL,
        max_retries=0,
        http_client=http_client,
    )
    proxy = RecordingSDKProxy(sdk_client)
    client = DeepSeekVisionClient(
        api_key=api_key,
        base_url=BASE_URL,
        model=MODEL,
        timeout=TIMEOUT_SECONDS,
        sdk_client=proxy,
    )
    state = ""
    reason = ""
    exception_chain = "None"
    answer: str | None = None
    elapsed = 0.0
    started_at = time.perf_counter()
    try:
        answer = client.analyze(image_data=image_data, question=QUESTION)
    except VisionClientTimeoutError as exc:
        state = STATE_ATTEMPTED_UNKNOWN
        reason = "timeout_" + _transport_phase(exc)
        exception_chain = _chain_text(exc)
    except APIStatusError as exc:
        state = STATE_HTTP_RESPONSE
        reason = "api_status_error"
        exception_chain = type(exc).__name__
        if counters["http_status"] is None:
            counters["http_status"] = getattr(exc, "status_code", None)
        if counters["http_request_id"] is None:
            counters["http_request_id"] = getattr(exc, "request_id", None)
    except APIConnectionError as exc:
        state = STATE_ATTEMPTED_UNKNOWN
        reason = "connection_" + _transport_phase(exc)
        exception_chain = _chain_text(exc)
    except Exception as exc:
        exception_chain = _chain_text(exc)
        if counters["http_response"] > 0:
            # 已观察到 HTTP 响应：即使后续解析/取值失败，也属于"已收到响应"，
            # 不得回退为证据不足。注意：200 不等于分析成功，也不等于费用已知。
            state = STATE_HTTP_RESPONSE
            reason = "api_response_error"
        elif proxy.create_calls == 0 and counters["http_request"] == 0:
            # 依据是 SDK create 从未被调用，而非"钩子计数为 0"的推断。
            state = STATE_LOCAL_PRECHECK
            reason = "pre_request_error"
        else:
            state = STATE_ATTEMPTED_UNKNOWN
            reason = "unexpected_error"
    finally:
        elapsed = time.perf_counter() - started_at
        try:
            http_client.close()
        except Exception:
            pass
    usage = getattr(proxy.last_response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)
    text_present = isinstance(answer, str) and bool(answer.strip())
    if not state:
        if text_present:
            state = STATE_SUCCESS
            reason = "nonempty_text"
        else:
            state = STATE_HTTP_RESPONSE
            reason = "empty_text"

    # 服务端处理状态：只有拿到 HTTP 响应才允许标记为已确认；

    # 任何阶段失败（含 pool_phase）都不得断言服务端未处理、更不得断言无费用。
    if state == STATE_LOCAL_PRECHECK:
        server_processing = "not_started"
    elif state in (STATE_HTTP_RESPONSE, STATE_SUCCESS):
        server_processing = "confirmed_by_http_response"
    else:
        server_processing = "unknown"

    def show(value: Any) -> str:
        return "null" if value is None else str(value)
    print("--- 结果（仅白名单元数据，不含回答正文）---")
    print(f"状态：{state}")
    print(f"原因：{reason}")
    print(f"服务端处理状态：{server_processing}")
    print(f"异常类型链：{exception_chain}")
    print(f"HTTP 状态码：{show(counters['http_status'])}")
    print(f"Request ID：{show(counters['http_request_id'])}")
    print(f"耗时（秒）：{elapsed:.3f}")
    print(f"prompt_tokens：{show(prompt_tokens)}")
    print(f"completion_tokens：{show(completion_tokens)}")
    print(f"total_tokens：{show(total_tokens)}")
    print(f"SDK create 调用次数：{proxy.create_calls}")
    print(f"HTTP 请求尝试次数：{counters['http_request']}")
    print(f"HTTP 响应次数：{counters['http_response']}")
    print(f"模型文本非空：{str(text_present).lower()}")
    print(
        "证据说明：请求钩子计数仅代表客户端尝试发送；"
        "只有 HTTP 状态码或 Request ID 才能证明服务端已处理。"
    )
    if state == STATE_SUCCESS:
        return 0
    if state == STATE_LOCAL_PRECHECK:
        return 9
    if reason == "empty_text":
        return 10
    if state == STATE_ATTEMPTED_UNKNOWN:
        return 11
    return 12
if __name__ == "__main__":
    sys.exit(main())
