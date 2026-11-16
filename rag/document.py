class Document:
    
    """
    RAG中的基础数据结构

    page_content:
        文档正文

    metadata:
        文档额外信息
    """

    def __init__(
        self,
        page_content: str,
        metadata: dict | None = None
    ):
        self.page_content = page_content
        self.metadata = metadata or {}


    def __repr__(self):
        return (
            f"Document("
            f"content_length={len(self.page_content)}, "
            f"metadata={self.metadata}"
            f")"
        )