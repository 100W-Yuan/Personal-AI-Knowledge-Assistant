from rag.document import Document
from rag.vector_store import VectorStore


store = VectorStore()


doc_a = Document(
    "鏂囨。A",
    {"source": "A.txt"}
)

doc_b = Document(
    "鏂囨。B",
    {"source": "B.txt"}
)

doc_c = Document(
    "鏂囨。C",
    {"source": "C.txt"}
)


store.add(
    doc_a,
    [1, 2, 3]
)

store.add(
    doc_b,
    [1, 2, 4]
)

store.add(
    doc_c,
    [10, 20, 30]
)


query_vector = [1, 2, 3]


results = store.search(
    query_vector,
    top_k=2
)


for result in results:

    print(
        result["document"].page_content,
        result["score"]
    )
