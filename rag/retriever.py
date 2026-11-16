class Retriever:

    def __init__(self, vector_store):
        self.vector_store = vector_store

    def retrieve(self, query_vector, top_k=3, threshold=None):

        # 参数校验：top_k 必须是非负整数（bool 是 int 的子类，须显式排除）。
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
            raise ValueError(
                f"top_k 必须是非负整数，得到 {top_k!r}"
            )

        # 处理顺序保持不变：VectorStore 内部先算分并降序排序，
        # 再做 Top-K 截断；threshold 过滤发生在这之后（仅当显式
        # 传入时启用），因此过滤后结果数 <= top_k。
        documents = self.vector_store.search(
            query_vector,
            top_k=top_k
        )

        if threshold is None:
            return documents

        # 仅保留 score >= threshold 的结果（等于阈值时必须保留）。
        return [
            document
            for document in documents
            if document["score"] >= threshold
        ]