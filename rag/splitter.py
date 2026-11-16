from .document import Document


class TextSplitter:

    def __init__(
        self,
        chunk_size=100,
        overlap=20
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size 必须大于 0")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap 必须满足 0 <= overlap < chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap


    def split(self, document):

        text = document.page_content

        chunks = []

        start = 0
        chunk_id = 0

        while start < len(text):

            end = start + self.chunk_size

            chunk_text = text[start:end]

            metadata = document.metadata.copy()

            # chunk_id 在每个原始 Document 内从 0 开始，不是跨文件全局 ID。
            metadata["chunk_id"] = chunk_id

            chunks.append(
                Document(
                    page_content=chunk_text,
                    metadata=metadata
                )
            )

            chunk_id += 1

            # 若当前块已经完整覆盖到文本末尾，则停止（避免恰等长时
            # 因 overlap 步进产生冗余的重叠尾块）。
            if end >= len(text):
                break

            start += self.chunk_size - self.overlap


        return chunks
