import math


class VectorStore:

    def __init__(self):
        self.items = []


    def add(self, document, vector):

        self.items.append(
            {
                "document": document,
                "vector": vector
            }
        )


    def search(self, query_vector, top_k=3):

        results = []

        for item in self.items:

            score = self.similarity(
                query_vector,
                item["vector"]
            )

            result = {
                "document": item["document"],
                "vector": item["vector"],
                "score": score
            }

            results.append(result)

        results.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return results[:top_k]

    def similarity(self, a, b):
        if len(a) != len(b):
            raise ValueError(
                f"向量维度不一致：{len(a)} != {len(b)}"
            )
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)