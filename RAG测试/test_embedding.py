# -*- coding: utf-8 -*-
"""Embedding 手工冒烟脚本（历史遗留，非 pytest 用例）。

R2-A 安全约束：
- **导入本模块不会加载任何真实模型**（Embedding 在 main() 内部才导入/实例化）；
- 真实 Embedding 仅在环境变量 ``RUN_REAL_EMBEDDING=1`` 时初始化；
- 未授权时打印 SKIP 并以 0 退出，不加载模型、不访问网络。

运行方式（在项目根目录）：
  py -3.12 -m RAG测试.test_embedding                       # 默认：打印 SKIP
  RUN_REAL_EMBEDDING=1 py -3.12 -m RAG测试.test_embedding   # 真实模型

注：真实 Embedding 依赖（sentence-transformers / torch）只安装在 py -3.12，
    裸 ``python``（3.13）未装，真实模式必须使用 py -3.12。
"""

import os


def main():
    if os.environ.get("RUN_REAL_EMBEDDING") != "1":
        print(
            "[SKIP] 未设置 RUN_REAL_EMBEDDING=1，"
            "未加载真实 Embedding 模型。"
        )
        return 0

    # 仅在显式授权后才导入并实例化真实模型。
    from rag.embedding import Embedding

    embedding = Embedding()

    text = "职业目标"

    vector = embedding.encode(text)

    print(vector)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
