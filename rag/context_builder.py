"""把 Retriever 的 Top-K 结果整理成一段可供 LLM 使用的 Context。

来源标签只读取**当前 chunk 自身**的 metadata：哪个字段存在就展示哪个，
缺失的字段直接不出现，不用 ``None`` / ``unknown`` / ``第 0 页`` 占位。
把「该格式没有此概念」渲染成一个具体数值，会让模型把不存在的定位当成
事实引用，比不显示更糟。
"""


def _positive_int(value):
    """把页码类字段归一为 >=1 的 int；类型不符或越界返回 None。"""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value >= 1 else None


def _non_negative_int(value):
    """把序号类字段归一为 >=0 的 int；类型不符或越界返回 None。"""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value >= 0 else None


def _format_page_location(metadata):
    """PDF：page 由 PdfLoader 写为 1 基页码，无需换算。"""
    page = _positive_int(metadata.get("page"))
    if page is None:
        return ""
    page_count = _positive_int(metadata.get("page_count"))
    # page_count 缺失、或小于当前页（异常数据）时只报当前页，
    # 否则会出现「第 3/2 页」这种自相矛盾的定位。
    if page_count is None or page_count < page:
        return f"第 {page} 页"
    return f"第 {page} 页（共 {page_count} 页）"


def _format_section_location(metadata):
    """DOCX 章节：section_index 在 DocxLoader 内是 0 基，展示时换成 1 基。

    DocxLoader 只对「真正产出了片段」的节递增 section_index（见
    ``rag/loader.py`` 第三遍分配逻辑），因此这里的序号是「第几个有内容的
    节」，与文档里的第几个标题可能不同；标题原文由 section 一并展示，
    模型可据此自行校正。
    """
    section = metadata.get("section")
    if not isinstance(section, str):
        return ""
    title = section.strip()
    if not title:
        # 空标题不能渲染成「第 1 节：」这样的半截标签。
        return ""
    index = _non_negative_int(metadata.get("section_index"))
    if index is None:
        return f"章节：{title}"
    return f"第 {index + 1} 节：{title}"


def _format_table_location(metadata):
    """DOCX 表格：table_index 是「文档内第几个 w:tbl」的 0 基编号。

    空表格与解析失败的表格同样占用编号（见 ``rag/loader.py``），
    所以 ``table_index + 1`` 表示的是表格在文档中的真实次序，而非
    「第几个进入了索引的表格」。
    """
    table_index = _non_negative_int(metadata.get("table_index"))
    if table_index is None:
        return ""
    return f"表格 {table_index + 1}"


# 顺序即读取顺序。三种 Loader 产出的定位字段互不重叠（PDF 没有 section，
# DOCX 表格没有 section，章节没有 table_index），所以这里不是「优先级覆盖」，
# 而是对互斥字段的一次统一读取。
_LOCATION_FORMATTERS = (
    _format_page_location,
    _format_section_location,
    _format_table_location,
)


def _format_location(metadata):
    """返回当前 chunk 的定位描述；无可用定位字段时返回空串。"""
    for formatter in _LOCATION_FORMATTERS:
        location = formatter(metadata)
        if location:
            return location
    return ""


class ContextBuilder:

    """把 Retriever 返回的 Top-K 结果整理成一段可供 LLM 使用的 Context。

    每条结果固定渲染四个部分：``【相关资料】`` 标记、来源行、相似度行、内容行。
    来源行在文件名与片段序号之外，按该 chunk 自身的 metadata 附带定位信息：
    PDF 给页码与总页数、DOCX 章节给节序号与标题、DOCX 表格给表格序号；
    TXT / Markdown 没有定位概念，来源行与引入定位信息之前完全一致。

    本类**不排序、不截断、不设长度预算**：Context 的顺序完全取决于调用方
    传入的 ``results`` 顺序（生产链路上已由 ``VectorStore.search`` 按分数
    降序取 Top-K），总长度等于全部结果渲染长度之和。
    """

    def build(self, results):

        context = ""

        for result in results:
            document = result["document"]
            metadata = getattr(document, "metadata", {}) or {}
            source = metadata.get("source", "unknown")

            label = f"来源：{source}"

            chunk_id = metadata.get("chunk_id")
            if chunk_id is not None:
                label += f"（片段 {chunk_id}）"

            location = _format_location(metadata)
            if location:
                label += f"｜{location}"

            context += (
                "【相关资料】\n"
                f"{label}\n"
                f"相似度：{result['score']}\n"
                f"内容：{document.page_content}\n\n"
            )

        return context
