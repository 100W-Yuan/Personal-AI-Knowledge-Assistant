# -*- coding: utf-8 -*-
"""RAG 全链路手工演示脚本（历史遗留，非 pytest 用例）。

R2-A 安全约束：
- **导入本模块不会加载任何真实模型**（Embedding 在 main() 内部才导入/实例化）；
- 真实 Embedding 仅在环境变量 ``RUN_REAL_EMBEDDING=1`` 时初始化；
- 未授权时打印 SKIP 并以 0 退出，不读文件、不加载模型、不访问网络。

原脚本逻辑（Loader → Splitter → Embedding → VectorStore → Retriever）
保持不变，仅把执行体移入 main() 并加上开关。
已知的历史缺陷（本轮不修，超出 R2-A 授权范围）：
- ``TextLoader("RAG测试/txt")`` 指向目录而非文件；
- ``retriever.retrieve(query)`` 传入的是字符串而非查询向量；
- 输出读取 ``result["distance"]``，而检索结果字段是 ``score``。

运行方式（在项目根目录）：
  py -3.12 -m RAG测试.test_rag_pipeline                       # 默认：打印 SKIP
  RUN_REAL_EMBEDDING=1 py -3.12 -m RAG测试.test_rag_pipeline   # 真实模型

注：真实 Embedding 依赖（sentence-transformers / torch）只安装在 py -3.12，
    裸 ``python``（3.13）未装，真实模式必须使用 py -3.12。
"""

import os

from rag.loader import TextLoader
from rag.splitter import TextSplitter
from rag.vector_store import VectorStore
from rag.retriever import Retriever


def main():
    if os.environ.get("RUN_REAL_EMBEDDING") != "1":
        print(
            "[SKIP] 未设置 RUN_REAL_EMBEDDING=1，"
            "未加载真实 Embedding 模型。"
        )
        return 0

    # 1. 加载文档
    loader = TextLoader(
        "RAG测试/txt"
    )

    documents = loader.load()

    print("========== Loader ==========")

    for document in documents:
        print(document)

    # 2. 文本切分
    splitter = TextSplitter(
        chunk_size=30,
        overlap=5
    )

    chunks = []

    for document in documents:
        chunks.extend(
            splitter.split(document)
        )

    print("\n========== Splitter ==========")

    for chunk in chunks:
        print(
            chunk.metadata,
            chunk.page_content
        )

    # 3. 创建 Embedding（仅在授权后导入/实例化）
    from rag.embedding import Embedding

    embedding = Embedding()

    # 4. 创建 VectorStore
    store = VectorStore()

    # 5. 将 Chunk 转换成向量并保存
    for chunk in chunks:

        vector = embedding.encode(
            chunk.page_content
        )

        store.add(
            chunk,
            vector
        )

    print("\n========== VectorStore ==========")
    print("保存 Chunk 数量：", len(store.items))

    # 6. 创建 Retriever
    retriever = Retriever(
        store
    )

    # 7. 输入查询
    query = "我的Agent项目"

    # 8. 检索
    results = retriever.retrieve(
        query
    )

    print("\n========== Retriever ==========")

    for result in results:

        print(
            "distance:",
            result["distance"]
        )

        print(
            "content:",
            result["document"].page_content
        )

        print(
            "metadata:",
            result["document"].metadata
        )

        print("----------------")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
