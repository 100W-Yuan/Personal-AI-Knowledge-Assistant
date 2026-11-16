import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag.loader import TextLoader
import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).parent.parent)
)

from rag.loader import TextLoader


def main():

    loader = TextLoader(
        "RAG测试/测试/test.txt"
    )

    documents = loader.load()

    for doc in documents:

        print(doc)

        print("----------------")

        print(doc.page_content)

        print(doc.metadata)


if __name__ == "__main__":

    main()