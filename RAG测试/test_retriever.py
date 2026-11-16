# -*- coding: utf-8 -*-
"""Retriever 手工冒烟脚本（历史遗留，非 pytest 用例）。

R2-A 安全约束：
- **导入本模块不会加载任何真实模型**（Embedding 在 main() 内部才导入/实例化）；
- 真实 Embedding 仅在环境变量 ``RUN_REAL_EMBEDDING=1`` 时初始化；
- 未授权时打印 SKIP 并以 0 退出，不加载模型、不访问网络。

原脚本逻辑（构造两个文档 → 向量化 → 入库 → 检索 Top-2 → 打印）
保持不变，仅把执行体移入 main() 并加上开关。

运行方式（在项目根目录）：
  py -3.12 -m RAG测试.test_retriever                       # 默认：打印 SKIP
  RUN_REAL_EMBEDDING=1 py -3.12 -m RAG测试.test_retriever   # 真实模型

注：真实 Embedding 依赖（sentence-transformers / torch）只安装在 py -3.12，
    裸 ``python``（3.13）未装，真实模式必须使用 py -3.12。
"""

import os

from rag.document import Document
from rag.retriever import Retriever
from rag.vector_store import VectorStore


def main():
    if os.environ.get("RUN_REAL_EMBEDDING") != "1":
        print(
            "[SKIP] 未设置 RUN_REAL_EMBEDDING=1，"
            "未加载真实 Embedding 模型。"
        )
        return 0

    store = VectorStore()

    doc_a = Document(
        "我的Agent项目",
        {
            "source": "A.txt"
        }
    )

    doc_b = Document(
        "我的健身计划",
        {
            "source": "B.txt"
        }
    )

    # 暂时直接使用 Embedding 生成文档向量（仅在授权后导入/实例化）
    from rag.embedding import Embedding

    embedding = Embedding()

    store.add(
        doc_a,
        embedding.encode(doc_a.page_content)
    )

    store.add(
        doc_b,
        embedding.encode(doc_b.page_content)
    )

    retriever = Retriever(
        store
    )

    query_vector = embedding.encode(
        "我的Agent项目"
    )

    results = retriever.retrieve(
        query_vector,
        top_k=2
    )

    for result in results:

        print(
            result["document"].page_content,
            result["score"]
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
