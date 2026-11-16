
from dataclasses import dataclass


@dataclass(frozen=True)
class ImageError:
    type: str
    message: str

    def __post_init__(self):
        if not isinstance(self.type, str):
            raise TypeError("错误类型必须是字符串")

        if not isinstance(self.message, str):
            raise TypeError("错误说明必须是字符串")

        if not self.type.strip():
            raise ValueError("错误类型不能为空")

        if not self.message.strip():
            raise ValueError("错误说明不能为空")


@dataclass(frozen=True)
class ImageResult:
    success: bool
    answer: str | None
    error: ImageError | None

    def __post_init__(self):
        if not isinstance(self.success, bool):
            raise TypeError("success 必须是布尔值")

        if self.success:
            if not isinstance(self.answer, str) or not self.answer.strip():
                raise ValueError("成功结果必须包含非空回答")

            if self.error is not None:
                raise ValueError("成功结果不能包含错误信息")

        else:
            if self.answer is not None:
                raise ValueError("失败结果的 answer 必须为 None")

            if not isinstance(self.error, ImageError):
                raise ValueError("失败结果必须包含 ImageError")
