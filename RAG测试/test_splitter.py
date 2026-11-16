from rag.document import Document
from rag.splitter import TextSplitter


document = Document(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    {
        "source": "test.txt"
    }
)


splitter = TextSplitter(
    chunk_size=10,
    overlap=2
)


chunks = splitter.split(document)


for chunk in chunks:
    print(chunk)
    print(chunk.page_content)
    print(chunk.metadata)
    print("----------------")