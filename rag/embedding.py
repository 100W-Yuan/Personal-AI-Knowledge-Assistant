from sentence_transformers import SentenceTransformer


class Embedding:

    def __init__(self):

        self.model = SentenceTransformer(
            "BAAI/bge-small-zh-v1.5"
        )


    def encode(self, text):

        return self.model.encode(
            text,
            normalize_embeddings=True
        )