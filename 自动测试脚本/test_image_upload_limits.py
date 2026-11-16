# -*- coding: utf-8 -*-
"""第八阶段专项：图片上传资源保护边界测试。

每个测试的 docstring 首行标注验证层次：
[实际执行行为]  直接 await main.analyze_image(...)，观察真实控制流与返回值；
[mock 交互]     用 FakeUpload 记录 endpoint 对 read/close 的调用参数与次数；
[静态结构]      只确认结构/常量存在，不声称运行期效果。

本轮明确未验证（不得据本文件声称）：
1. Starlette/FastAPI 在 endpoint 执行前的 multipart 解析开销：
   starlette/formparsers.py 会把文件 part 写入 SpooledTemporaryFile
   （MultiPartParser.spool_max_size = 1MB，超过后落盘），该阶段发生在
   endpoint 之前；本文件不测量该层的内存或磁盘占用。
2. 真实视觉模型调用（本轮禁止）。
3. 真实 HTTP 传输层的请求体接收行为（本文件用 fake 直接调用端点函数，
   不经过 TestClient 的完整 ASGI 请求栈）。
"""

import asyncio
import inspect
from io import BytesIO
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from PIL import Image
from starlette.formparsers import MultiPartParser

import main
from image_result import ImageResult
from image_validator import MAX_IMAGE_SIZE_BYTES


# 基线常量：暂定 10 MiB 上限。
REAL_LIMIT = 10 * 1024 * 1024


def make_success_result() -> ImageResult:
    """构造一个合法的成功结果，供 patch 使用（不涉及真实模型）。"""
    return ImageResult(success=True, answer="ok", error=None)


def make_small_png() -> bytes:
    """生成一个真实有效的小 PNG（2x2，约几十字节）。"""
    buffer = BytesIO()
    Image.new("RGB", (2, 2), color="red").save(buffer, format="PNG")
    return buffer.getvalue()


class FakeUpload:
    """UploadFile 替身（鸭子类型），可观测 read / close 的调用。

    - read_calls：每次 read 收到的 size 参数，按调用顺序记录；
    - delivered：实际交付给 endpoint 的字节数（用于证明没有读完整文件）；
    - total_bytes：模拟源文件总大小（不实际分配该大小）；
    - data：可选的真实字节内容，提供时 read 返回其切片。
    """

    def __init__(
        self,
        total_bytes: int | None = None,
        raise_on_read: Exception | None = None,
        data: bytes | None = None,
    ):
        self.data = data
        self.total_bytes = len(data) if data is not None else total_bytes
        self.raise_on_read = raise_on_read
        self.read_calls: list[int] = []
        self.delivered = 0
        self.close_calls = 0
        self.closed = False

    async def read(self, size: int = -1) -> bytes:
        self.read_calls.append(size)

        if self.raise_on_read is not None:
            raise self.raise_on_read

        start = self.delivered
        end = (
            self.total_bytes
            if size is None or size < 0
            else min(start + size, self.total_bytes)
        )
        self.delivered = end

        if self.data is not None:
            return self.data[start:end]

        return b"x" * (end - start)

    async def close(self) -> None:
        self.close_calls += 1
        self.closed = True


# ============================================================
# A 组：受限读取行为
# ============================================================

def test_limit_constant_is_ten_mib():
    """[静态结构] 确认模块级上限常量为 10 MiB（与基线一致）。"""
    assert MAX_IMAGE_SIZE_BYTES == REAL_LIMIT


def test_read_size_uses_real_limit_plus_one():
    """[mock 交互] endpoint 对 read 的 size 参数确为真实上限 + 1。

    通过 FakeUpload 捕获 read 的实参，验证的是 endpoint 实际执行时
    传入的读取量；不验证底层流实际返回多少字节。
    """
    fake = FakeUpload(total_bytes=10)  # 小源，避免分配 10 MiB

    with patch.object(main, "understand_image", return_value=make_success_result()):
        asyncio.run(main.analyze_image(file=fake, question="图片里有什么？"))

    assert fake.read_calls == [REAL_LIMIT + 1]


def test_real_png_under_real_limit_reaches_understand_image():
    """[实际执行行为] 真实小 PNG 在真实 10 MiB 上限下进入 understand_image。

    不 monkeypatch 常量、不缩小上限：使用真实 MAX_IMAGE_SIZE_BYTES，
    上传真实有效 PNG（约几十字节），验证正常路径不被误拒。
    """
    png = make_small_png()
    fake = FakeUpload(data=png)

    with patch.object(
        main, "understand_image", return_value=make_success_result()
    ) as mock_understand:
        result = asyncio.run(
            main.analyze_image(file=fake, question="图片里有什么？")
        )

    assert result["success"] is True
    mock_understand.assert_called_once()
    assert mock_understand.call_args.kwargs["image_data"] == png
    assert len(fake.read_calls) == 1


def test_read_is_called_exactly_once():
    """[mock 交互] endpoint 只调用一次 read，不会循环读完整文件。"""
    fake = FakeUpload(total_bytes=10)

    with patch.object(main, "understand_image", return_value=make_success_result()):
        asyncio.run(main.analyze_image(file=fake, question="图片里有什么？"))

    assert len(fake.read_calls) == 1


def test_valid_upload_reaches_understand_image(monkeypatch):
    """[实际执行行为] 未超限的上传会进入 understand_image。

    用 monkeypatch 将上限临时缩到 1024 字节（不改文件），避免分配 10 MiB。
    """
    monkeypatch.setattr(main, "MAX_IMAGE_SIZE_BYTES", 1024)
    fake = FakeUpload(total_bytes=1024)

    with patch.object(
        main, "understand_image", return_value=make_success_result()
    ) as mock_understand:
        result = asyncio.run(
            main.analyze_image(file=fake, question="图片里有什么？")
        )

    assert result["success"] is True
    mock_understand.assert_called_once()
    passed_data = mock_understand.call_args.kwargs["image_data"]
    assert len(passed_data) == 1024


def test_exactly_at_limit_is_accepted(monkeypatch):
    """[实际执行行为] 恰好等于上限（1024）不被拒绝。"""
    monkeypatch.setattr(main, "MAX_IMAGE_SIZE_BYTES", 1024)
    fake = FakeUpload(total_bytes=1024)

    with patch.object(
        main, "understand_image", return_value=make_success_result()
    ) as mock_understand:
        asyncio.run(main.analyze_image(file=fake, question="图片里有什么？"))

    mock_understand.assert_called_once()


def test_limit_plus_one_is_rejected(monkeypatch):
    """[实际执行行为] 上限 + 1 字节触发 413。"""
    monkeypatch.setattr(main, "MAX_IMAGE_SIZE_BYTES", 1024)
    fake = FakeUpload(total_bytes=1025)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            main.analyze_image(file=fake, question="图片里有什么？")
        )

    assert exc_info.value.status_code == 413
    assert exc_info.value.detail["type"] == "image_too_large"


def test_oversized_does_not_call_understand_image(monkeypatch):
    """[实际执行行为 + mock 交互] 超限时 understand_image 完全未被调用。"""
    monkeypatch.setattr(main, "MAX_IMAGE_SIZE_BYTES", 1024)
    fake = FakeUpload(total_bytes=1025)

    with patch.object(main, "understand_image") as mock_understand:
        with pytest.raises(HTTPException):
            asyncio.run(
                main.analyze_image(file=fake, question="图片里有什么？")
            )

    mock_understand.assert_not_called()


# ============================================================
# B 组：资源保护边界（endpoint 内读取量）
# ============================================================

def test_endpoint_does_not_consume_entire_oversized_source(monkeypatch):
    """[mock 交互] 超限源只被取走“上限 + 1”字节，未读完整文件。

    注意：这是 fake 层面的交付量记录，不等价于真实内存占用测量；
    也不能说明 HTTP 请求体或 multipart 临时文件没有被完整接收。
    """
    monkeypatch.setattr(main, "MAX_IMAGE_SIZE_BYTES", 1024)
    fake = FakeUpload(total_bytes=5_000_000)  # 虚拟超大源，不实际分配

    with pytest.raises(HTTPException):
        asyncio.run(
            main.analyze_image(file=fake, question="图片里有什么？")
        )

    assert fake.delivered == 1025
    assert fake.delivered < fake.total_bytes


def test_multipart_parser_buffers_outside_endpoint():
    """[静态结构] 确认 multipart 解析层独立存在于 endpoint 之前。

    只验证框架结构：文件 part 由 SpooledTemporaryFile 承载（1MB 阈值），
    因此请求体在 endpoint 执行前已被接收/缓冲。
    本测试不测量该层的内存或磁盘开销，也不断言应用层限制覆盖了它。
    """
    assert MultiPartParser.spool_max_size == 1024 * 1024
    assert MultiPartParser.max_part_size == 1024 * 1024

    source = inspect.getsource(MultiPartParser)
    assert "SpooledTemporaryFile" in source
    assert "spool_max_size" in source


# ============================================================
# C 组：异常与清理
# ============================================================

def test_close_called_on_success():
    """[mock 交互] 正常路径下文件被关闭且只关闭一次。"""
    fake = FakeUpload(total_bytes=10)

    with patch.object(main, "understand_image", return_value=make_success_result()):
        asyncio.run(main.analyze_image(file=fake, question="图片里有什么？"))

    assert fake.close_calls == 1
    assert fake.closed is True


def test_close_called_on_oversized(monkeypatch):
    """[mock 交互] 超限（413）路径下文件仍被关闭。"""
    monkeypatch.setattr(main, "MAX_IMAGE_SIZE_BYTES", 1024)
    fake = FakeUpload(total_bytes=1025)

    with pytest.raises(HTTPException):
        asyncio.run(
            main.analyze_image(file=fake, question="图片里有什么？")
        )

    assert fake.close_calls == 1
    assert fake.closed is True


def test_close_called_when_read_raises():
    """[mock 交互] 读取抛异常时文件仍被关闭（finally 生效）。"""
    fake = FakeUpload(total_bytes=10, raise_on_read=RuntimeError("read boom"))

    with pytest.raises(RuntimeError):
        asyncio.run(
            main.analyze_image(file=fake, question="图片里有什么？")
        )

    assert fake.close_calls == 1
    assert fake.closed is True


def test_close_called_when_understand_raises():
    """[mock 交互] 图片理解抛异常时文件仍被关闭。"""
    fake = FakeUpload(total_bytes=10)

    with patch.object(main, "understand_image", side_effect=RuntimeError("model boom")):
        with pytest.raises(RuntimeError):
            asyncio.run(
                main.analyze_image(file=fake, question="图片里有什么？")
            )

    assert fake.close_calls == 1



def test_close_error_does_not_mask_original_exception(caplog):
    """原始操作与关闭都失败时，保留原始异常并记录关闭异常。"""

    class CloseFailUpload(FakeUpload):
        async def close(self):
            self.close_calls += 1
            raise RuntimeError("close failed")

    fake = CloseFailUpload(
        total_bytes=10,
        raise_on_read=RuntimeError("read failed"),
    )

    with pytest.raises(RuntimeError, match="read failed") as exc_info:
        asyncio.run(
            main.analyze_image(file=fake, question="图片里有什么？")
        )

    assert str(exc_info.value) == "read failed"
    assert "close failed" in caplog.text
    assert fake.close_calls == 1
 
def test_close_error_is_raised_when_operation_succeeds():
    """图片处理成功但关闭失败时，关闭异常不能被吞掉。"""

    class CloseFailUpload(FakeUpload):
        async def close(self):
            self.close_calls += 1
            raise RuntimeError("close failed")

    fake = CloseFailUpload(total_bytes=10)

    with patch.object(
        main,
        "understand_image",
        return_value=make_success_result(),
    ):
        with pytest.raises(RuntimeError, match="close failed"):
            asyncio.run(
                main.analyze_image(
                    file=fake,
                    question="图片里有什么？",
                )
            )

    assert fake.close_calls == 1
