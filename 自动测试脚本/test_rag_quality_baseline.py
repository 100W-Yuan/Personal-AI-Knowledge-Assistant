# -*- coding: utf-8 -*-
"""P2-3E-STEP1：RAG 离线评估基线（RAG-BENCHMARK-V1）。

本文件建立 my-ai 25 的第一版**检索质量**基线，只评估

    Loader → Splitter → KnowledgeBase → Embedding → Retriever

这一段链路，**不评估**真实 LLM 的回答生成质量，**不调用** DeepSeek 或任何
外部 API。

三个待回答的问题
----------------
1. 正确证据能不能被检索出来？        → Recall@K / HitRate@K
2. 正确证据在结果中的排名如何？      → MRR
3. 返回的来源定位是否与真实位置一致？ → 三层定位准确率（文档级 / chunk 级 / 字段级）

数据集
------
``RAG-BENCHMARK-V1``：4 份**由本文件生成**的测试文档 + 25 条评估问题
（23 条可回答 + 2 条无证据；Gold Evidence 共 25 条，其中 2 条跨文档问题
各含 2 条证据）。

- PDF（3 页，每页一类事实）+ 1 份 3 页全空白 PDF（验证空白页被安全跳过）；
- DOCX（2 个一级标题章节 + 1 个真实表格 + 1 个空表格）；
- TXT（水质要求）、MD（安装指南）。

文档内容刻意不重复：同一事实只出现在一份文档里，保证 gold evidence 唯一。

关键口径（避免指标解释混用）
--------------------------
- **gold evidence 的确定方式**：由数据集声明的定位信息（PDF 页码 / DOCX 章节或
  表格下标 / TXT·MD 的 chunk_seq）**在索引里反查**得到 ``(document_id, chunk_seq)``，
  完全不依赖任何一次检索的排名；
- **排名只算一次**：每条 query 只调用一次 ``search(top_k=10)``，Recall@1/3/5/10
  均为这一条完整排名的前缀截断（IR 评估的标准做法），不是四次独立检索；
- **Precision@K 分母固定为 K**：即使实际返回数 < K 也按 K 计算（保守口径），
  同时另报 ``precision_returned@K``（分母 ``min(K, 实际返回数)``）以免口径混淆；
- **Recall 必须同时给出两种口径**（P2-3E-STEP1-R1 修订）：
  - **微平均（evidence-level micro）** ``Σ命中证据 / Σ|Gold|``——证据权重相等，
    跨文档问题（|Gold|=2）自然占双倍权重；
  - **宏平均（query-level macro）** ``mean(命中数 / |Gold_i|)``——每条问题权重
    相等，跨文档问题只贡献 1/23。
  两者在 ``|Gold_i|`` 不一致时必然不同，**不得只报其一**。报告字段分别为
  ``recall_micro`` / ``recall_macro``；
- **Precision@K 也必须双口径**：``precision_macro``（先按每条算 hits/K 再平均）
  与 ``precision_micro``（``Σhits / (N·K)``）。当且仅当每条返回数相同且分母
  固定为 K 时两者相等；第三口径 ``precision_micro_returned`` 用
  ``Σ min(K, 实际返回数)`` 作分母，用于返回数不足 K 的情况；
- **Recall / MRR 的分母只计可回答问题**（``answerable=True``）；无证据问题
  不进入这两个指标的分母，也不进入 Precision 分母；
- **定位准确率拆三层**，不合并：文档级命中（命中正确 source）、chunk 级命中
  （命中 gold 的 ``(document_id, chunk_seq)``）、字段级正确（命中的那条记录
  其 page / section / table_index 与 gold 完全一致）；
- **Unsupported Retrieval Rate** 只衡量**检索噪声**：无证据问题没有任何相关
  证据，因此 Top-K 中每一条都计为非相关，该指标等于 ``平均(返回条数 / K)``。
  它**不等于**模型幻觉率，也不等于模型拒答率。

运行方式
--------
本文件位于 ``自动测试脚本/``，继承该目录 conftest.py 的离线隔离（dotenv 假模块、
openai 空壳、非回环网络拦截）。真实模型为本地缓存的 ``BAAI/bge-small-zh-v1.5``，
离线模式加载，512 维。禁止 stub embedding 参与「正式评估结果」。
"""

import json
import os
import sys
from pathlib import Path

# 离线环境必须在导入任何 rag / sentence_transformers 之前设置：
# BGE 只从本地缓存读取，不触网（conftest 会拦截任何非回环连接）。
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pytest  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for _path in (str(PROJECT_ROOT), str(SCRIPT_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from rag.knowledge_base import KnowledgeBase  # noqa: E402


# ======================================================================
# 0. 基线元信息
# ======================================================================

BENCHMARK_VERSION = "RAG-BENCHMARK-V1"

# Splitter 的默认参数（rag/splitter.py:9-14 的默认值，本文件只读不改）。
SPLITTER_CHUNK_SIZE = 100
SPLITTER_OVERLAP = 20
_SPLITTER_STEP = SPLITTER_CHUNK_SIZE - SPLITTER_OVERLAP

# 一次检索取满 10（KnowledgeBase.search 的 top_k 上限），所有 K 由前缀截断得到。
RETRIEVAL_TOP_K = 10
KS = (1, 3, 5, 10)

EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
EMBEDDING_DIM_EXPECTED = 512

PDF_NAME = "qingshan_product_manual_v2.pdf"
DOCX_NAME = "qingshan_filter_spec_v2.docx"
TXT_NAME = "qingshan_water_quality.txt"
MD_NAME = "qingshan_install_guide.md"
BLANK_PDF_NAME = "qingshan_blank_appendix.pdf"

CORPUS_SOURCES = (PDF_NAME, DOCX_NAME, TXT_NAME, MD_NAME)

REPORT_DIRNAME = "evidence"
REPORT_FILENAME = "benchmark_report.json"


def _document_id(filename):
    """与 KnowledgeBase._compute_document_id 同算法：sha256(basename)[:32]。

    只由文件名派生，因此与索引、改名规则一致；这里重复一份是为了让数据集
    能**在构建索引之前**声明稳定的文档身份（不 import 私有静态方法）。
    """
    import hashlib

    return hashlib.sha256(filename.encode("utf-8")).hexdigest()[:32]


DOC_ID = {name: _document_id(name) for name in CORPUS_SOURCES}


# ======================================================================
# 1. 测试文档内容（唯一事实来源）
# ======================================================================
# 设计约束：同一事实只出现在一份文档中（由 test_dataset_facts_are_disjoint
# 强制）。这样 gold evidence 才有区分度，不会出现「两份文档都能答」的歧义。

PDF_PAGES = (
    # 第 1 页：产品基础信息
    (
        "青山净水器 QS-2026A 产品维护手册",
        "型号：QS-2026A",
        "额定电压：220 伏",
        "额定功率：1200 瓦",
    ),
    # 第 2 页：滤芯更换
    (
        "第 2 章 滤芯更换",
        "滤芯型号：FC-90A",
        "滤芯更换周期为 90 天",
        "更换后复位滤芯计时器",
    ),
    # 第 3 页：其他维护
    (
        "第 3 章 其他维护",
        "外壳清洁使用中性清洁剂",
        "长期停用需排空水箱",
        "运行噪音不高于 52 分贝",
    ),
)

PDF_BLANK_PAGE_COUNT = 3

DOCX_SECTION_0 = "滤芯维护规范"
DOCX_BODY_0 = "滤芯巡检间隔为 30 天。每次巡检应记录进出水压力值。"
DOCX_TABLE_0_ROWS = (
    ("项目", "要求"),
    ("检测方式", "压力差检测"),
    ("记录频率", "每日记录一次"),
    ("判定阈值", "压差超过 0.15 兆帕"),
    ("异常处理", "压差超标时更换滤芯"),
)
DOCX_SECTION_1 = "其他维护内容"
DOCX_BODY_1 = "滤芯清洗使用清水冲洗 3 分钟，禁止使用洗涤剂。"
# 空表格（0 行）：Loader 必须跳过它，但仍占用 table_index=1。
DOCX_EMPTY_TABLE_ROWS = 0
DOCX_EMPTY_TABLE_COLS = 2

TXT_TEXT = """青山净水器水质要求说明（文档编号 WQ-2026-03）

一、适用范围
本说明适用于青山净水器家用系列产品。安装与使用前应完整阅读本文档。文档内容与随机说明书不一致时，以随机说明书为准。

二、进水水质要求
进水水源应为市政自来水。进水水温范围为 5 至 38 摄氏度。进水浊度不得超过 1 NTU。进水余氯不得超过 0.3 毫克每升。进水硬度建议不超过 300 毫克每升，以碳酸钙计。若进水水质超出上述范围，应先加装前置过滤器并复测。

三、出水水质要求
出水水质应符合 GB 5749 生活饮用水卫生标准。出水余氯每月检测一次并记录。出水流量应不低于每分钟 1.2 升。连续三次检测不合格时应停用设备。

四、检测与记录
水质检测数据应保存两年以上。检测仪器每十二个月校准一次。用户可通过面板查询最近一次检测结果。记录表格由售后服务中心统一提供。

五、异常处理
出现异味或浊度升高时，应立即停止使用并联系售后。滤芯到期未更换会导致出水流量下降。长时间停用后重新启用应先冲洗管路。

六、术语说明
浊度表示水中悬浮颗粒对光线的散射程度。余氯表示水中残留的有效氯浓度。硬度表示水中钙镁离子的总含量，通常以碳酸钙计。

七、版本记录
本文档首次发布于 2026 年 3 月。后续修订以版本号标注。文档解释权归青山售后服务中心。

八、附则
本说明的纸质版本随设备交付。电子版本可在服务网点索取。条款如有变更不另行通知。

九、常见问题
问：进水压力过低会影响出水吗？答：会影响，建议加装增压泵后复测。
"""

MD_TEXT = """# 青山净水器安装指南

## 一、开箱检查
开箱后核对主机、滤芯、进水软管、出水龙头与说明书是否齐全。发现缺件应在签收后三日内联系经销商。主机外观有明显磕碰时不要通电。

## 二、安装环境要求
安装环境温度应为 4 至 40 摄氏度。安装环境相对湿度不得超过 85%。机身四周应预留 10 厘米散热间距。设备应避免阳光直射与雨淋。

## 三、电气要求
电源插座必须可靠接地。供电电压应与铭牌标注一致。禁止使用多孔插排串联供电。电源线不得压在家具下方。

## 四、管路连接
进水软管接入自来水角阀。接头处应加装密封垫圈。安装完成后打开角阀检查是否渗漏。排水管不得插入地漏液面以下。

## 五、首次使用
首次使用应连续冲洗 5 分钟。冲洗期间出水不可饮用。冲洗完成后检查出水是否清澈。若出水仍浑浊应重复冲洗。

## 六、日常检查
每周检查一次接头是否渗漏。每月擦洗一次机身外壳。每季度检查一次电源线绝缘层。发现异常噪音应立即停用。

## 七、拆卸与移机
拆卸前关闭进水角阀并断开电源。排空机内余水后装箱。移机后应由专业人员重新安装。

## 八、常见安装问题
问：墙面为轻质隔墙可以挂装吗？答：需要加装承重背板后再挂装。问：软管长度不够怎么办？答：可联系服务网点更换加长软管。

## 九、售后服务
设备出现故障时先对照本指南排查。无法排除时联系服务网点安排上门。上门服务前请准备好购机凭证。
"""


# ======================================================================
# 2. 文档生成（真实 pypdf / python-docx）
# ======================================================================
# 环境无 reportlab / fpdf / PyMuPDF，因此用 pypdf 组装结构完整的 PDF：
# Type0 复合字体 + Identity-H + ToUnicode CMap（BMP 恒等）承载中文，
# 正文以 UTF-16BE 十六进制串写入内容流。该构造方式与
# run_real_agent_rag_citation.py 的 _build_pdf 同源，这里独立实现一份，
# 使本基线不依赖任何「真机执行」脚本。

_PDF_TOUNICODE_CMAP = b"""/CIDInit /ProcSet findresource begin
12 dict begin
begincmap
/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def
/CMapName /Adobe-Identity-UCS def
/CMapType 2 def
1 begincodespacerange
<0000> <FFFF>
endcodespacerange
1 beginbfrange
<0000> <FFFF> <0000>
endbfrange
endcmap
CMapName currentdict /CMap defineresource pop
end
end
"""


def _pdf_hex_text(text):
    """把文本编码为 PDF 十六进制字符串（UTF-16BE）。"""
    return "<%s>" % text.encode("utf-16-be").hex().upper()


def _build_pdf(path, pages):
    """组装多页真实 PDF；``pages`` 为若干「行列表」，每页一个内容流。"""
    from pypdf import PdfWriter
    from pypdf.generic import (
        ArrayObject,
        DecodedStreamObject,
        DictionaryObject,
        NameObject,
        NumberObject,
        TextStringObject,
    )

    writer = PdfWriter()

    cmap_stream = DecodedStreamObject()
    cmap_stream.set_data(_PDF_TOUNICODE_CMAP)
    cmap_ref = writer._add_object(cmap_stream)

    descriptor = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/FontDescriptor"),
            NameObject("/FontName"): NameObject("/SimSun"),
            NameObject("/Flags"): NumberObject(4),
            NameObject("/FontBBox"): ArrayObject(
                [
                    NumberObject(0),
                    NumberObject(-200),
                    NumberObject(1000),
                    NumberObject(900),
                ]
            ),
            NameObject("/ItalicAngle"): NumberObject(0),
            NameObject("/Ascent"): NumberObject(880),
            NameObject("/Descent"): NumberObject(-120),
            NameObject("/CapHeight"): NumberObject(700),
            NameObject("/StemV"): NumberObject(80),
        }
    )
    descriptor_ref = writer._add_object(descriptor)

    cid_font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/CIDFontType2"),
            NameObject("/BaseFont"): NameObject("/SimSun"),
            NameObject("/CIDSystemInfo"): DictionaryObject(
                {
                    NameObject("/Registry"): TextStringObject("Adobe"),
                    NameObject("/Ordering"): TextStringObject("Identity"),
                    NameObject("/Supplement"): NumberObject(0),
                }
            ),
            NameObject("/FontDescriptor"): descriptor_ref,
            NameObject("/DW"): NumberObject(1000),
            NameObject("/CIDToGIDMap"): NameObject("/Identity"),
        }
    )
    cid_font_ref = writer._add_object(cid_font)

    type0_font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type0"),
            NameObject("/BaseFont"): NameObject("/SimSun"),
            NameObject("/Encoding"): NameObject("/Identity-H"),
            NameObject("/DescendantFonts"): ArrayObject([cid_font_ref]),
            NameObject("/ToUnicode"): cmap_ref,
        }
    )
    font_ref = writer._add_object(type0_font)

    for lines in pages:
        ops = ["BT", "/F1 12 Tf", "72 780 Td", "16 TL"]
        if lines:
            for line in lines:
                ops.append("%s Tj" % _pdf_hex_text(line))
                ops.append("T*")
        # 没有任何 Tj 的页面 = 真实空白页（无文本层），用于空白页场景。
        ops.append("ET")

        content = DecodedStreamObject()
        content.set_data("\n".join(ops).encode("ascii"))
        content_ref = writer._add_object(content)

        page = writer.add_blank_page(width=595, height=842)
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref})}
        )
        page[NameObject("/Contents")] = content_ref

    with open(path, "wb") as handle:
        writer.write(handle)


def _build_blank_pdf(path, page_count):
    """生成纯空白 PDF：Loader 必须抛 NoTextLayerError 并被索引安全跳过。"""
    _build_pdf(path, [() for _ in range(page_count)])


def _build_docx(path):
    """真实 DOCX：两个一级标题章节 + 一个真实表格 + 一个空表格。"""
    from docx import Document as DocxDocument

    document = DocxDocument()

    document.add_heading(DOCX_SECTION_0, level=1)
    document.add_paragraph(DOCX_BODY_0)

    table = document.add_table(rows=len(DOCX_TABLE_0_ROWS), cols=2)
    for row_index, row_values in enumerate(DOCX_TABLE_0_ROWS):
        for col_index, value in enumerate(row_values):
            table.rows[row_index].cells[col_index].text = value

    document.add_heading(DOCX_SECTION_1, level=1)
    document.add_paragraph(DOCX_BODY_1)

    # 空表格：0 行。python-docx 允许；Loader 必须跳过但不复用编号。
    document.add_table(rows=DOCX_EMPTY_TABLE_ROWS, cols=DOCX_EMPTY_TABLE_COLS)

    document.save(str(path))


def _write_text(path, text):
    path.write_text(text, encoding="utf-8")


# ======================================================================
# 3. 切块窗口（与 rag/splitter.py 的默认参数逐字对应）
# ======================================================================


def _split_windows(text):
    """复现 TextSplitter 的窗口：[start, start+chunk_size)，步长 chunk_size-overlap。"""
    windows = []
    start = 0
    while start < len(text):
        end = start + SPLITTER_CHUNK_SIZE
        windows.append((start, text[start:end]))
        if end >= len(text):
            break
        start += _SPLITTER_STEP
    return windows


def _anchor_chunk_seqs(text, anchor):
    """返回「包含 anchor 文本」的 chunk 序号列表（按窗口内容判定，不看排名）。"""
    return [index for index, (_, body) in enumerate(_split_windows(text)) if anchor in body]


def _single_chunk_seq(text, anchor, label):
    """要求 anchor 恰好落在一个 chunk 内，否则数据集设计有问题，直接报错。"""
    seqs = _anchor_chunk_seqs(text, anchor)
    if len(seqs) != 1:
        raise AssertionError(
            f"[{BENCHMARK_VERSION}] anchor 未唯一落块：{label}={anchor!r} -> {seqs}"
            "（需要调整文档内容或改用非跨界锚点）"
        )
    return seqs[0]


# ======================================================================
# 4. 评估数据集（25 条：23 可回答 + 2 无证据）
# ======================================================================


def _locator_pdf(page):
    return {"page": page}


def _locator_docx_section(section, section_index):
    return {"section": section, "section_index": section_index}


def _locator_docx_table(table_index):
    return {"table_index": table_index}


def _locator_chunk(chunk_seq):
    return {"chunk_seq": chunk_seq}


def _case(
    case_id,
    query,
    category,
    answerable,
    expected_source,
    expected_locator,
    expected_fact,
):
    """构造一条数据集记录：gold_evidence 里嵌入稳定的 document_id。"""
    gold_evidence = {}
    if answerable:
        gold_evidence = {
            "source": expected_source,
            "document_id": DOC_ID[expected_source],
        }
        gold_evidence.update(expected_locator)

    return {
        "case_id": case_id,
        "query": query,
        "category": category,
        "answerable": bool(answerable),
        "expected_source": expected_source if answerable else None,
        "expected_locator": dict(expected_locator) if answerable else None,
        "expected_fact": expected_fact if answerable else None,
        "gold_evidence": gold_evidence,
    }


def _build_dataset():
    cases = []

    # ---- PDF 页级事实（5）----
    cases.append(
        _case(
            "pdf_01_model",
            "青山净水器的型号是什么？",
            "pdf_page",
            True,
            PDF_NAME,
            _locator_pdf(1),
            "QS-2026A",
        )
    )
    cases.append(
        _case(
            "pdf_02_power",
            "产品的额定功率是多少瓦？",
            "pdf_page",
            True,
            PDF_NAME,
            _locator_pdf(1),
            "1200",
        )
    )
    cases.append(
        _case(
            "pdf_03_replace_cycle",
            "滤芯更换周期是多少天？",
            "pdf_page",
            True,
            PDF_NAME,
            _locator_pdf(2),
            "90 天",
        )
    )
    cases.append(
        _case(
            "pdf_04_filter_model",
            "滤芯的型号是什么？",
            "pdf_page",
            True,
            PDF_NAME,
            _locator_pdf(2),
            "FC-90A",
        )
    )
    cases.append(
        _case(
            "pdf_05_noise",
            "设备运行噪音是多少分贝？",
            "pdf_page",
            True,
            PDF_NAME,
            _locator_pdf(3),
            "52 分贝",
        )
    )

    # ---- DOCX 章节事实（4）----
    cases.append(
        _case(
            "docx_01_inspect_interval",
            "滤芯巡检间隔是多少天？",
            "docx_section",
            True,
            DOCX_NAME,
            _locator_docx_section(DOCX_SECTION_0, 0),
            "30 天",
        )
    )
    cases.append(
        _case(
            "docx_02_record_pressure",
            "巡检时需要记录哪些压力数据？",
            "docx_section",
            True,
            DOCX_NAME,
            _locator_docx_section(DOCX_SECTION_0, 0),
            "进出水压力值",
        )
    )
    cases.append(
        _case(
            "docx_03_clean_minutes",
            "滤芯清洗需要冲洗多长时间？",
            "docx_section",
            True,
            DOCX_NAME,
            _locator_docx_section(DOCX_SECTION_1, 1),
            "3 分钟",
        )
    )
    cases.append(
        _case(
            "docx_04_clean_forbid",
            "清洗滤芯时可以使用洗涤剂吗？",
            "docx_section",
            True,
            DOCX_NAME,
            _locator_docx_section(DOCX_SECTION_1, 1),
            "禁止使用洗涤剂",
        )
    )

    # ---- DOCX 表格事实（4）----
    cases.append(
        _case(
            "docx_05_table_method",
            "规定的检测方式是什么？",
            "docx_table",
            True,
            DOCX_NAME,
            _locator_docx_table(0),
            "压力差检测",
        )
    )
    cases.append(
        _case(
            "docx_06_table_threshold",
            "压差判定阈值是多少？",
            "docx_table",
            True,
            DOCX_NAME,
            _locator_docx_table(0),
            "0.15 兆帕",
        )
    )
    cases.append(
        _case(
            "docx_07_table_frequency",
            "检测记录频率有什么要求？",
            "docx_table",
            True,
            DOCX_NAME,
            _locator_docx_table(0),
            "每日记录一次",
        )
    )
    cases.append(
        _case(
            "docx_08_table_overflow",
            "压差超标时应当如何处理？",
            "docx_table",
            True,
            DOCX_NAME,
            _locator_docx_table(0),
            "更换滤芯",
        )
    )

    # ---- TXT 检索（4）----
    cases.append(
        _case(
            "txt_01_temperature",
            "进水水温范围是多少？",
            "txt",
            True,
            TXT_NAME,
            _locator_chunk(_single_chunk_seq(TXT_TEXT, "5 至 38 摄氏度", "txt_temp")),
            "5 至 38 摄氏度",
        )
    )
    cases.append(
        _case(
            "txt_02_turbidity",
            "进水浊度上限是多少？",
            "txt",
            True,
            TXT_NAME,
            _locator_chunk(_single_chunk_seq(TXT_TEXT, "1 NTU", "txt_turbidity")),
            "1 NTU",
        )
    )
    cases.append(
        _case(
            "txt_03_hardness",
            "进水硬度建议上限是多少？",
            "txt",
            True,
            TXT_NAME,
            _locator_chunk(_single_chunk_seq(TXT_TEXT, "300 毫克每升", "txt_hardness")),
            "300 毫克每升",
        )
    )
    cases.append(
        _case(
            "txt_04_standard",
            "出水水质应符合哪个标准？",
            "txt",
            True,
            TXT_NAME,
            _locator_chunk(_single_chunk_seq(TXT_TEXT, "GB 5749", "txt_standard")),
            "GB 5749",
        )
    )

    # ---- MD 检索（4）----
    cases.append(
        _case(
            "md_01_env_temp",
            "安装环境温度要求是多少？",
            "md",
            True,
            MD_NAME,
            _locator_chunk(_single_chunk_seq(MD_TEXT, "4 至 40 摄氏度", "md_temp")),
            "4 至 40 摄氏度",
        )
    )
    cases.append(
        _case(
            "md_02_humidity",
            "安装环境相对湿度上限是多少？",
            "md",
            True,
            MD_NAME,
            _locator_chunk(_single_chunk_seq(MD_TEXT, "85%", "md_humidity")),
            "85%",
        )
    )
    cases.append(
        _case(
            "md_03_clearance",
            "机身需要预留多少散热间距？",
            "md",
            True,
            MD_NAME,
            _locator_chunk(_single_chunk_seq(MD_TEXT, "10 厘米散热间距", "md_clearance")),
            "10 厘米",
        )
    )
    cases.append(
        _case(
            "md_04_grounding",
            "对电源插座有什么安全要求？",
            "md",
            True,
            MD_NAME,
            _locator_chunk(_single_chunk_seq(MD_TEXT, "可靠接地", "md_ground")),
            "必须可靠接地",
        )
    )

    # ---- 跨文档检索（2）：每条 gold 含两条证据，来自不同格式 ----
    cases.append(
        {
            "case_id": "cross_01_periods",
            "query": "滤芯更换周期与滤芯巡检间隔分别是多少天？",
            "category": "cross_doc",
            "answerable": True,
            "expected_source": [PDF_NAME, DOCX_NAME],
            "expected_locator": [
                {"source": PDF_NAME, "page": 2},
                {"source": DOCX_NAME, "section": DOCX_SECTION_0, "section_index": 0},
            ],
            "expected_fact": ["90 天", "30 天"],
            "gold_evidence": [
                {"source": PDF_NAME, "document_id": DOC_ID[PDF_NAME], "page": 2},
                {
                    "source": DOCX_NAME,
                    "document_id": DOC_ID[DOCX_NAME],
                    "section": DOCX_SECTION_0,
                    "section_index": 0,
                },
            ],
        }
    )
    cases.append(
        {
            "case_id": "cross_02_water_install",
            "query": "对进水水质和安装环境分别有哪些要求？",
            "category": "cross_doc",
            "answerable": True,
            "expected_source": [TXT_NAME, MD_NAME],
            "expected_locator": [
                {"source": TXT_NAME, "chunk_seq": _single_chunk_seq(TXT_TEXT, "1 NTU", "cross_txt")},
                {"source": MD_NAME, "chunk_seq": _single_chunk_seq(MD_TEXT, "85%", "cross_md")},
            ],
            "expected_fact": ["1 NTU", "85%"],
            "gold_evidence": [
                {"source": TXT_NAME, "document_id": DOC_ID[TXT_NAME], "chunk_seq": _single_chunk_seq(TXT_TEXT, "1 NTU", "cross_txt")},
                {"source": MD_NAME, "document_id": DOC_ID[MD_NAME], "chunk_seq": _single_chunk_seq(MD_TEXT, "85%", "cross_md")},
            ],
        }
    )

    # ---- 无证据问题（2）：文档中确实没有该信息 ----
    cases.append(_case("noans_01_warranty", "整机保修期限是多少年？", "unanswerable", False, None, None, None))
    cases.append(_case("noans_02_price", "滤芯的建议零售价格是多少元？", "unanswerable", False, None, None, None))

    return cases


CASES = _build_dataset()

# 无证据问题必须真的「无据」：这些词不得出现在任何一份测试文档里。
NEGATIVE_TOPIC_KEYWORDS = ("保修", "质保", "零售价", "价格", "费用")


def _raw_documents():
    """``{source: 纯文本}``：用于校验事实分散度与负向主题缺失。"""
    pdf_text = "\n".join("\n".join(lines) for lines in PDF_PAGES)
    docx_text = "\n".join(
        [
            DOCX_SECTION_0,
            DOCX_BODY_0,
            "\n".join(" | ".join(row) for row in DOCX_TABLE_0_ROWS),
            DOCX_SECTION_1,
            DOCX_BODY_1,
        ]
    )
    return {
        PDF_NAME: pdf_text,
        DOCX_NAME: docx_text,
        TXT_NAME: TXT_TEXT,
        MD_NAME: MD_TEXT,
    }


# ======================================================================
# 5. 指标实现
# ======================================================================
# 约定：``ranking`` 是 [{"identity": (document_id, chunk_seq), "source": ...,
# "score": ..., "metadata": {...}}, ...]，按分数降序、长度为 RETRIEVAL_TOP_K。


def _gold_ids(gold_evidence):
    """gold evidence 列表 → {(document_id, chunk_seq)} 集合（由 resolve 填充）。"""
    return {item["identity"] for item in gold_evidence}


def _hits_at_k(ranking, gold_ids, k):
    """Top-K 中命中 Gold 的证据条数（微平均的分子）。"""
    return sum(1 for record in ranking[:k] if record["identity"] in gold_ids)


def _recall_at_k(ranking, gold_ids, k):
    if not gold_ids:
        return None
    hit = sum(1 for record in ranking[:k] if record["identity"] in gold_ids)
    return hit / len(gold_ids)


def _precision_at_k(ranking, gold_ids, k, denominator="fixed"):
    """Precision@K。

    denominator="fixed"    ：分母固定为 K（保守口径，本基线主口径）；
    denominator="returned" ：分母为 min(K, 实际返回条数)，用于对照。
    """
    hit = sum(1 for record in ranking[:k] if record["identity"] in gold_ids)
    if denominator == "fixed":
        return hit / k
    actual = min(k, len(ranking))
    return hit / actual if actual else 0.0


def _reciprocal_rank(ranking, gold_ids):
    """首个相关结果的名次倒数；未命中记 0。"""
    if not gold_ids:
        return None
    for index, record in enumerate(ranking, start=1):
        if record["identity"] in gold_ids:
            return 1.0 / index
    return 0.0


def _source_hit(ranking, k, expected_sources):
    if isinstance(expected_sources, str):
        expected_sources = [expected_sources]
    sources = {record["source"] for record in ranking[:k]}
    return bool(sources & set(expected_sources))


# gold evidence 中不属于「定位字段」的键（用于把定位核对与身份字段分开）。
_GOLD_NON_LOCATOR_KEYS = (
    "source",
    "document_id",
    "identity",
    "chunk_text",
    "resolved",
)


def _field_view(record):
    """组装「可核对的字段视图」：metadata 全部键 + 身份键（chunk_seq 等）。"""
    view = dict(record["metadata"])
    view["source"] = record["source"]
    view["document_id"] = record["document_id"]
    view["chunk_seq"] = record["chunk_seq"]
    return view


def _locator_check(record, gold_item):
    """字段级定位核对：gold 声明的每个定位键都必须存在且相等。

    返回 (ok, 明细)。明细只记录键名与是否相等，不落正文。

    注意：TXT / MD 的定位字段就是 ``chunk_seq``，它与 chunk 级身份同源，
    因此这两种格式的字段级核对在 chunk 命中时必然成立（由构造决定）。
    区分度主要来自 PDF 的 ``page`` 与 DOCX 的 ``section`` / ``table_index``。
    """
    locator_keys = [
        key for key in gold_item if key not in _GOLD_NON_LOCATOR_KEYS
    ]
    view = _field_view(record)
    detail = {}
    ok = True
    for key in locator_keys:
        expected = gold_item[key]
        actual = view.get(key)
        matched = actual == expected and key in view
        detail[key] = {"expected": expected, "actual": actual, "ok": matched}
        ok = ok and matched
    return ok, detail


def _mean(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _gold_identity_set(gold_items):
    """gold 解析结果 → 去重后的 ``(document_id, chunk_seq)`` 集合。

    用集合而非列表：同一身份出现两次（例如两个 gold 条目落到同一 chunk）
    只计一次，避免重复 Gold 把分母放大。
    """
    identities = set()
    for item in gold_items:
        for chunk in item["resolved"]:
            identities.add(chunk["identity"])
    return identities


def _rows_from_cases(cases):
    """从逐条结果提取聚合用行；**无证据问题在这里被排除**（不进任何分母）。"""
    rows = []
    for case in cases:
        if not case["answerable"]:
            continue
        rows.append(
            {
                "case_id": case["case_id"],
                "gold_size": case["gold_id_count"],
                "hits": {k: case["metrics"]["hits@%d" % k] for k in KS},
                "returned": len(case["retrieval"]["ranking"]),
            }
        )
    return rows


def _aggregate_answerable(rows, ks=KS):
    """可回答问题上的聚合指标（纯函数：无 I/O、无模型、无随机性）。

    ``rows`` 每条包含三个字段：

    - ``gold_size``：该问题的 Gold Evidence 条数（>0）；跨文档问题为 2；
    - ``hits``：``{k: 命中证据条数}``，即 Top-K 与 Gold 的交集大小；
    - ``returned``：该问题实际返回条数（用于 returned 口径的分母）。

    微 / 宏两种口径在这里**同时**产出，且不共用中间量，便于独立复核：

    - ``recall_micro@K    = Σ hits / Σ gold_size``
    - ``recall_macro@K    = mean(hits / gold_size)``
    - ``precision_macro@K = mean(hits / K)``
    - ``precision_micro@K = Σ hits / (N · K)``
    - ``precision_micro_returned@K = Σ hits / Σ min(K, returned)``
    """
    n = len(rows)
    result = {
        "n": n,
        "gold_total": sum(row["gold_size"] for row in rows),
        "recall_micro": {},
        "recall_macro": {},
        "precision_macro": {},
        "precision_micro": {},
        "precision_micro_returned": {},
        "hit_rate": {},
    }

    if n == 0:
        for k in ks:
            for key in (
                "recall_micro",
                "recall_macro",
                "precision_macro",
                "precision_micro",
                "precision_micro_returned",
                "hit_rate",
            ):
                result[key]["@%d" % k] = None
        return result

    gold_total = result["gold_total"]

    for k in ks:
        total_hits = sum(row["hits"][k] for row in rows)
        result["recall_micro"]["@%d" % k] = total_hits / gold_total
        result["recall_macro"]["@%d" % k] = (
            sum(row["hits"][k] / row["gold_size"] for row in rows) / n
        )
        result["precision_macro"]["@%d" % k] = (
            sum(row["hits"][k] / k for row in rows) / n
        )
        result["precision_micro"]["@%d" % k] = total_hits / (n * k)
        returned_denom = sum(min(k, row["returned"]) for row in rows)
        result["precision_micro_returned"]["@%d" % k] = (
            total_hits / returned_denom if returned_denom else None
        )
        result["hit_rate"]["@%d" % k] = (
            sum(1 for row in rows if row["hits"][k] > 0) / n
        )

    return result


# ======================================================================
# 6. 夹具：建文档 → 真索引 → 一次性排名 → 指标
# ======================================================================


def _store_records(kb):
    """从真实 VectorStore 取出索引内的全部 chunk（只读，不看排名）。"""
    items = kb._retriever.vector_store.items
    records = []
    for item in items:
        document = item["document"]
        metadata = dict(document.metadata or {})
        records.append(
            {
                "source": metadata.get("source"),
                "document_id": metadata.get("document_id"),
                "chunk_seq": metadata.get("chunk_seq"),
                "metadata": metadata,
                "chunk_text": document.page_content,
            }
        )
    return records


def _resolve_gold(store_records, gold_evidence, case_id):
    """把数据集声明的定位信息在**索引**里反查成 (document_id, chunk_seq)。

    这不是「按排名反推 gold」：匹配只依赖 metadata 与声明的定位字段，
    与任何一次检索结果无关。
    """
    resolved = []
    items = gold_evidence if isinstance(gold_evidence, list) else [gold_evidence]
    for item in items:
        locator_keys = [
            key
            for key in item
            if key not in ("source", "document_id")
        ]
        matched = [
            record
            for record in store_records
            if record["source"] == item["source"]
            and record["document_id"] == item["document_id"]
            and all(record["metadata"].get(key) == item[key] for key in locator_keys)
        ]
        if not matched:
            raise AssertionError(
                f"[{BENCHMARK_VERSION}] gold evidence 在索引中找不到对应 chunk："
                f"case={case_id} locator={locator_keys}"
            )
        entry = dict(item)
        entry["resolved"] = [
            {
                "document_id": record["document_id"],
                "chunk_seq": record["chunk_seq"],
                "identity": (record["document_id"], record["chunk_seq"]),
                "chunk_text": record["chunk_text"],
            }
            for record in matched
        ]
        resolved.append(entry)
    return resolved


@pytest.fixture(scope="session")
def benchmark(tmp_path_factory):
    """构建 RAG-BENCHMARK-V1 语料、真实索引、一次性排名与全部指标。"""
    run_dir = tmp_path_factory.mktemp("rag-benchmark-v1")
    kb_dir = run_dir / "kb"
    evidence_dir = run_dir / REPORT_DIRNAME
    kb_dir.mkdir()
    evidence_dir.mkdir()

    pdf_path = kb_dir / PDF_NAME
    docx_path = kb_dir / DOCX_NAME
    txt_path = kb_dir / TXT_NAME
    md_path = kb_dir / MD_NAME
    blank_pdf_path = kb_dir / BLANK_PDF_NAME

    _build_pdf(pdf_path, PDF_PAGES)
    _build_docx(docx_path)
    _write_text(txt_path, TXT_TEXT)
    _write_text(md_path, MD_TEXT)
    _build_blank_pdf(blank_pdf_path, PDF_BLANK_PAGE_COUNT)

    # 真实 KnowledgeBase：不注入 embedding_factory，使用生产默认（BGE 本地缓存）。
    kb = KnowledgeBase(doc_dir=kb_dir)

    # 真实检索（触发惰性索引）。预热查询不参与任何指标。
    preheat = kb.search("索引构建预热", top_k=1)

    ranking_by_case = {}
    for case in CASES:
        result = kb.search(case["query"], top_k=RETRIEVAL_TOP_K)
        ranking = []
        if result.get("ok"):
            for position, record in enumerate(result["results"], start=1):
                ranking.append(
                    {
                        "rank": position,
                        "source": record.get("source"),
                        "document_id": record.get("document_id"),
                        "chunk_seq": record.get("chunk_seq"),
                        "score": record.get("score"),
                        "identity": (
                            record.get("document_id"),
                            record.get("chunk_seq"),
                        ),
                        "metadata": {
                            key: record.get(key)
                            for key in (
                                "page",
                                "page_count",
                                "section",
                                "section_index",
                                "segment_index",
                                "table_index",
                            )
                            if key in record
                        },
                    }
                )
        ranking_by_case[case["case_id"]] = {
            "ok": bool(result.get("ok")),
            "count": result.get("count"),
            "stage": result.get("stage"),
            "error": result.get("error"),
            "ranking": ranking,
        }

    store_records = _store_records(kb)

    # ---- gold 解析 + 契约核对 ----
    cases_out = []
    for case in CASES:
        entry = {
            "case_id": case["case_id"],
            "query": case["query"],
            "category": case["category"],
            "answerable": case["answerable"],
            "expected_source": case["expected_source"],
            "expected_locator": case["expected_locator"],
            "expected_fact": case["expected_fact"],
            "gold_evidence": case["gold_evidence"],
            "retrieval": ranking_by_case[case["case_id"]],
        }

        gold_items = []
        if case["answerable"]:
            gold_items = _resolve_gold(store_records, case["gold_evidence"], case["case_id"])
            entry["gold_resolved"] = [
                {
                    "source": item["source"],
                    "document_id": item["document_id"],
                    "locator": {
                        key: item[key]
                        for key in item
                        if key not in ("source", "document_id", "resolved")
                    },
                    "chunks": [
                        {
                            "document_id": chunk["document_id"],
                            "chunk_seq": chunk["chunk_seq"],
                            "chunk_text": chunk["chunk_text"],
                        }
                        for chunk in item["resolved"]
                    ],
                }
                for item in gold_items
            ]
        entry["_gold_items"] = gold_items
        cases_out.append(entry)

    # ---- 逐条指标 ----
    for entry in cases_out:
        ranking = entry["retrieval"]["ranking"]
        gold_items = entry["_gold_items"]
        gold_ids = _gold_identity_set(gold_items)
        entry["gold_id_count"] = len(gold_ids)
        entry["metrics"] = {}

        for k in KS:
            if entry["answerable"]:
                # hits@K 是微平均的分子；recall@K 是宏平均的输入，两者分开存。
                entry["metrics"]["hits@%d" % k] = _hits_at_k(ranking, gold_ids, k)
                entry["metrics"]["recall@%d" % k] = _recall_at_k(ranking, gold_ids, k)
                entry["metrics"]["precision@%d" % k] = _precision_at_k(ranking, gold_ids, k)
                entry["metrics"]["precision_returned@%d" % k] = _precision_at_k(
                    ranking, gold_ids, k, denominator="returned"
                )
                entry["metrics"]["source_hit@%d" % k] = _source_hit(
                    ranking, k, entry["expected_source"]
                )
                entry["metrics"]["chunk_hit@%d" % k] = bool(
                    gold_ids & {record["identity"] for record in ranking[:k]}
                )
            else:
                returned = min(k, len(ranking))
                # 无相关证据：返回的每一条都计为非相关。
                entry["metrics"]["unsupported@%d" % k] = (
                    returned / k if k else 0.0
                )
                entry["metrics"]["returned@%d" % k] = returned

        if entry["answerable"]:
            entry["metrics"]["reciprocal_rank"] = _reciprocal_rank(ranking, gold_ids)
        else:
            entry["metrics"]["reciprocal_rank"] = None

        # 字段级定位：只对「命中 gold chunk」的记录核对，避免把无关记录算进分母。
        locator_checks = []
        for record in ranking:
            for item in gold_items:
                for chunk in item["resolved"]:
                    if record["identity"] == chunk["identity"]:
                        ok, detail = _locator_check(record, item)
                        locator_checks.append(
                            {
                                "rank": record["rank"],
                                "identity": record["identity"],
                                "source": item["source"],
                                "ok": ok,
                                "fields": detail,
                            }
                        )
        entry["locator_checks"] = locator_checks

    # ---- 聚合（微 / 宏双口径，纯函数 _aggregate_answerable 产出） ----
    answerable = [entry for entry in cases_out if entry["answerable"]]
    unanswerable = [entry for entry in cases_out if not entry["answerable"]]

    rows = _rows_from_cases(cases_out)
    answerable_aggregate = _aggregate_answerable(rows)

    aggregate = {
        "case_count": len(cases_out),
        "answerable_count": len(answerable),
        "unanswerable_count": len(unanswerable),
        "category_counts": {},
        # 微 / 宏分开，字段名自带口径，不再用含糊的 "recall"
        "recall_micro": answerable_aggregate["recall_micro"],
        "recall_macro": answerable_aggregate["recall_macro"],
        "precision_macro": answerable_aggregate["precision_macro"],
        "precision_micro": answerable_aggregate["precision_micro"],
        "precision_micro_returned": answerable_aggregate["precision_micro_returned"],
        "hit_rate": answerable_aggregate["hit_rate"],
        "evidence_totals": {
            # entries = 未去重的 Gold 条目累计；unique = 去重后的真实分母。
            # 两者不相等即说明数据集里有重复 Gold（会被 _gold_identity_set 消掉）。
            "gold_total": answerable_aggregate["gold_total"],
            "gold_total_entries": sum(
                len(item["resolved"])
                for entry in answerable
                for item in entry["_gold_items"]
            ),
            "gold_total_unique": answerable_aggregate["gold_total"],
            "gold_per_case": {row["case_id"]: row["gold_size"] for row in rows},
            "multi_gold_cases": [
                row["case_id"] for row in rows if row["gold_size"] > 1
            ],
        },
        "mrr": None,
        "mrr_zero_count": None,
        "citation": {},
        "unsupported_retrieval_rate": {},
        "negative_query_diagnostics": {},
        "score_diagnostic": {},
    }

    for entry in cases_out:
        category = entry["category"]
        aggregate["category_counts"][category] = (
            aggregate["category_counts"].get(category, 0) + 1
        )

    for k in KS:
        aggregate["unsupported_retrieval_rate"]["@%d" % k] = _mean(
            [entry["metrics"]["unsupported@%d" % k] for entry in unanswerable]
        )

    rrs = [entry["metrics"]["reciprocal_rank"] for entry in answerable]
    aggregate["mrr"] = _mean(rrs)
    aggregate["mrr_zero_count"] = sum(1 for value in rrs if value == 0.0)

    for k in KS:
        source_hits = sum(
            1 for entry in answerable if entry["metrics"]["source_hit@%d" % k]
        )
        chunk_hits = sum(
            1 for entry in answerable if entry["metrics"]["chunk_hit@%d" % k]
        )
        checks = [
            check
            for entry in answerable
            for check in entry["locator_checks"]
            if check["rank"] <= k
        ]
        ok_checks = [check for check in checks if check["ok"]]
        by_format = {}
        for check in checks:
            bucket = by_format.setdefault(
                check["source"], {"checks": 0, "ok": 0}
            )
            bucket["checks"] += 1
            if check["ok"]:
                bucket["ok"] += 1
        aggregate["citation"]["@%d" % k] = {
            # 三层各自的统计样本数必须显式给出，否则无法判断置信度
            "document_accuracy": source_hits / len(answerable) if answerable else None,
            "document_sample": len(answerable),
            "chunk_accuracy": chunk_hits / len(answerable) if answerable else None,
            "chunk_sample": len(answerable),
            "locator_field_accuracy": (
                len(ok_checks) / len(checks) if checks else None
            ),
            "locator_field_sample": len(checks),
            "locator_field_ok": len(ok_checks),
            "locator_by_source": by_format,
        }

    max_scores_answerable = [
        entry["retrieval"]["ranking"][0]["score"]
        for entry in answerable
        if entry["retrieval"]["ranking"]
    ]
    max_scores_unanswerable = [
        entry["retrieval"]["ranking"][0]["score"]
        for entry in unanswerable
        if entry["retrieval"]["ranking"]
    ]
    aggregate["score_diagnostic"] = {
        "note": (
            "仅作诊断：检索器对任何 query 都返回满 K 条，最高分不能作为"
            "「能否回答」的判据，也不构成拒答依据。"
        ),
        "max_score_mean_answerable": _mean(max_scores_answerable),
        "max_score_mean_unanswerable": _mean(max_scores_unanswerable),
        "max_score_answerable": max_scores_answerable,
        "max_score_unanswerable": max_scores_unanswerable,
    }

    # 无证据问题单独报告：返回数量 / 最高分 / 最低分 / 平均分，并声明无阈值。
    negative_detail = []
    for entry in unanswerable:
        scores = [record["score"] for record in entry["retrieval"]["ranking"]]
        negative_detail.append(
            {
                "case_id": entry["case_id"],
                "query": entry["query"],
                "returned": len(scores),
                "max_score": max(scores) if scores else None,
                "min_score": min(scores) if scores else None,
                "mean_score": _mean(scores),
                "gold_evidence": entry["gold_evidence"],
            }
        )
    aggregate["negative_query_diagnostics"] = {
        "note": (
            "本段只衡量检索侧噪声与相似度分布，不是拒答准确率，"
            "也不是幻觉率；检索器没有任何弃权（abstain）机制。"
        ),
        "threshold_defined": False,
        "threshold_note": (
            "本基线未定义拒答阈值。可回答问题最高分均值 %.4f 与无证据问题"
            "最高分均值 %.4f 存在区间重叠，任何单点阈值都会同时误伤两侧。"
            % (
                aggregate["score_diagnostic"]["max_score_mean_answerable"] or 0.0,
                aggregate["score_diagnostic"]["max_score_mean_unanswerable"] or 0.0,
            )
        ),
        "cases": negative_detail,
    }

    report = {
        "benchmark_version": BENCHMARK_VERSION,
        "retrieval_top_k": RETRIEVAL_TOP_K,
        "ks": list(KS),
        "embedding": {
            "model": EMBEDDING_MODEL_NAME,
            "class": type(getattr(kb, "_embedding", None)).__name__,
            "dim": int(kb._embedding.encode("维度检查").shape[0]) if kb.is_indexed else None,
        },
        "corpus": {
            "files": list(CORPUS_SOURCES),
            "blank_pdf": BLANK_PDF_NAME,
            "file_count": kb.file_count,
            "chunk_count": kb.chunk_count,
            "skipped_count": kb.skipped_count,
            "empty_file_count": kb.empty_file_count,
            "indexed_chunk_counts": kb.indexed_chunk_counts(),
            "chunks": [
                {
                    "source": record["source"],
                    "document_id": record["document_id"],
                    "chunk_seq": record["chunk_seq"],
                    "chars": len(record["chunk_text"]),
                    "page": record["metadata"].get("page"),
                    "section": record["metadata"].get("section"),
                    "table_index": record["metadata"].get("table_index"),
                }
                for record in store_records
            ],
        },
        "search_calls": {
            "preheat": preheat.get("ok"),
            "case_queries": len(CASES),
            "total": len(CASES) + 1,
        },
        "aggregate": aggregate,
        "cases": [
            {
                "case_id": entry["case_id"],
                "query": entry["query"],
                "category": entry["category"],
                "answerable": entry["answerable"],
                "expected_source": entry["expected_source"],
                "expected_locator": entry["expected_locator"],
                "expected_fact": entry["expected_fact"],
                "gold_evidence": entry["gold_evidence"],
                "gold_resolved": entry.get("gold_resolved", []),
                "gold_id_count": entry["gold_id_count"],
                "metrics": entry["metrics"],
                "locator_checks": entry["locator_checks"],
                "retrieval": {
                    "ok": entry["retrieval"]["ok"],
                    "count": entry["retrieval"]["count"],
                    "stage": entry["retrieval"]["stage"],
                    "error": entry["retrieval"]["error"],
                    "top_k": [
                        {
                            "rank": record["rank"],
                            "source": record["source"],
                            "document_id": record["document_id"],
                            "chunk_seq": record["chunk_seq"],
                            "score": record["score"],
                            "page": record["metadata"].get("page"),
                            "page_count": record["metadata"].get("page_count"),
                            "section": record["metadata"].get("section"),
                            "section_index": record["metadata"].get("section_index"),
                            "segment_index": record["metadata"].get("segment_index"),
                            "table_index": record["metadata"].get("table_index"),
                            "is_gold": record["identity"]
                            in {
                                chunk["identity"]
                                for item in entry["_gold_items"]
                                for chunk in item["resolved"]
                            },
                        }
                        for record in entry["retrieval"]["ranking"]
                    ],
                },
            }
            for entry in cases_out
        ],
        "paths": {
            "run_dir": str(run_dir),
            "kb_dir": str(kb_dir),
            "evidence_dir": str(evidence_dir),
            "report": str(evidence_dir / REPORT_FILENAME),
        },
    }

    report_path = evidence_dir / REPORT_FILENAME
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 交付用的可读摘要（同时便于人工核对，不落正文）。
    summary = {
        "benchmark_version": BENCHMARK_VERSION,
        "report_path": str(report_path),
        "case_count": aggregate["case_count"],
        "category_counts": aggregate["category_counts"],
        "corpus": report["corpus"],
        "embedding": report["embedding"],
        "recall_micro": aggregate["recall_micro"],
        "recall_macro": aggregate["recall_macro"],
        "precision_macro": aggregate["precision_macro"],
        "precision_micro": aggregate["precision_micro"],
        "precision_micro_returned": aggregate["precision_micro_returned"],
        "hit_rate": aggregate["hit_rate"],
        "mrr": aggregate["mrr"],
        "mrr_zero_count": aggregate["mrr_zero_count"],
        "evidence_totals": aggregate["evidence_totals"],
        "citation": aggregate["citation"],
        "unsupported_retrieval_rate": aggregate["unsupported_retrieval_rate"],
        "negative_query_diagnostics": aggregate["negative_query_diagnostics"],
        "score_diagnostic": {
            "max_score_mean_answerable": aggregate["score_diagnostic"]["max_score_mean_answerable"],
            "max_score_mean_unanswerable": aggregate["score_diagnostic"]["max_score_mean_unanswerable"],
        },
        "misses": [
            {
                "case_id": entry["case_id"],
                "query": entry["query"],
                "category": entry["category"],
                "recall@10": entry["metrics"]["recall@10"],
                "reciprocal_rank": entry["metrics"]["reciprocal_rank"],
                "top1_source": (
                    entry["retrieval"]["ranking"][0]["source"]
                    if entry["retrieval"]["ranking"]
                    else None
                ),
            }
            for entry in answerable
            if not entry["metrics"]["chunk_hit@10"]
        ],
        "rank_beyond_1": [
            {
                "case_id": entry["case_id"],
                "query": entry["query"],
                "rank": (
                    None
                    if not entry["metrics"]["reciprocal_rank"]
                    else int(round(1.0 / entry["metrics"]["reciprocal_rank"]))
                ),
            }
            for entry in answerable
            if entry["metrics"]["reciprocal_rank"] is not None
            and entry["metrics"]["reciprocal_rank"] < 1.0
        ],
    }

    print("\n===== RAG-BENCHMARK-V1 SUMMARY =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("===== END SUMMARY =====\n")

    return {"report": report, "summary": summary, "kb": kb, "run_dir": run_dir}


# ======================================================================
# 7. 数据集完整性
# ======================================================================


def test_dataset_size_and_distribution():
    """至少 20 条，且分类分布覆盖任务书要求的六类。"""
    assert len(CASES) >= 20, "评估问题不足 20 条"

    counts = {}
    for case in CASES:
        counts[case["category"]] = counts.get(case["category"], 0) + 1

    required = {"pdf_page", "docx_section", "docx_table", "cross_doc", "unanswerable"}
    missing = required - set(counts)
    assert not missing, f"缺少问题类别：{missing}"
    assert counts["txt"] >= 2 and counts["md"] >= 2, "TXT / MD 检索问题不足"
    assert counts["unanswerable"] == 2

    ids = [case["case_id"] for case in CASES]
    assert len(ids) == len(set(ids)), "case_id 重复"

    queries = [case["query"] for case in CASES]
    assert len(queries) == len(set(queries)), "query 重复"


def test_dataset_gold_contract():
    """可回答问题必须有完整 gold evidence；无证据问题必须为空并标记 answerable=False。"""
    for case in CASES:
        if case["answerable"]:
            gold = case["gold_evidence"]
            assert gold, f"{case['case_id']} 缺少 gold_evidence"
            items = gold if isinstance(gold, list) else [gold]
            for item in items:
                assert item.get("source") in CORPUS_SOURCES
                assert item.get("document_id") == DOC_ID[item["source"]]
                locator_keys = {
                    key for key in item if key not in ("source", "document_id")
                }
                assert locator_keys, f"{case['case_id']} gold 缺少定位信息"
                assert case["expected_fact"] is not None
        else:
            assert case["gold_evidence"] == {}
            assert case["expected_locator"] is None
            assert case["expected_fact"] is None
            assert case["expected_source"] is None


def test_dataset_locator_matches_format():
    """定位字段与格式一对一：PDF→page，DOCX 章节→section+section_index，
    DOCX 表格→table_index，TXT/MD→chunk_seq。"""
    for case in CASES:
        if not case["answerable"]:
            continue
        items = case["gold_evidence"]
        if not isinstance(items, list):
            items = [items]
        for item in items:
            keys = {key for key in item if key not in ("source", "document_id")}
            if item["source"] == PDF_NAME:
                assert keys == {"page"}, f"{case['case_id']} PDF 定位键异常：{keys}"
            elif item["source"] == DOCX_NAME:
                assert keys in (
                    {"section", "section_index"},
                    {"table_index"},
                ), f"{case['case_id']} DOCX 定位键异常：{keys}"
            else:
                assert keys == {"chunk_seq"}, f"{case['case_id']} TXT/MD 定位键异常：{keys}"


def test_dataset_facts_are_disjoint():
    """同一事实只能出现在一份文档里，避免 gold evidence 出现歧义。"""
    docs = _raw_documents()

    for case in CASES:
        if not case["answerable"]:
            continue
        facts = case["expected_fact"]
        sources = case["expected_source"]
        if not isinstance(facts, list):
            facts = [facts]
        if not isinstance(sources, list):
            sources = [sources]

        for fact in facts:
            owners = [name for name, text in docs.items() if fact in text]
            assert owners, f"{case['case_id']} 的事实 {fact!r} 不在任何文档中"
            assert set(owners) <= set(sources), (
                f"{case['case_id']} 的事实 {fact!r} 出现在预期之外的文档："
                f"{sorted(set(owners) - set(sources))}"
            )


def test_negative_topics_absent_from_corpus():
    """无证据问题必须真的『无据』：负向主题词不得出现在任何文档里。"""
    docs = _raw_documents()
    for keyword in NEGATIVE_TOPIC_KEYWORDS:
        hits = [name for name, text in docs.items() if keyword in text]
        assert not hits, f"负向主题词 {keyword!r} 出现在：{hits}"


# ======================================================================
# 8. 文档与索引契约
# ======================================================================


def test_documents_build_and_parse_real(benchmark):
    """真实 pypdf / python-docx 读回：页数、章节、表格内容与空白页。"""
    from pypdf import PdfReader
    from docx import Document as DocxDocument

    run_dir = benchmark["run_dir"]
    pdf_path = run_dir / "kb" / PDF_NAME
    docx_path = run_dir / "kb" / DOCX_NAME
    blank_path = run_dir / "kb" / BLANK_PDF_NAME

    reader = PdfReader(str(pdf_path))
    assert len(reader.pages) == len(PDF_PAGES)
    for index, lines in enumerate(PDF_PAGES):
        text = reader.pages[index].extract_text() or ""
        for line in lines:
            assert line in text, f"PDF 第 {index + 1} 页缺少内容：{line}"
    # 事实所在页正确
    page2 = reader.pages[1].extract_text() or ""
    assert "90 天" in page2
    assert "90 天" not in (reader.pages[0].extract_text() or "")

    blank_reader = PdfReader(str(blank_path))
    assert len(blank_reader.pages) == PDF_BLANK_PAGE_COUNT
    assert all(not (page.extract_text() or "").strip() for page in blank_reader.pages)

    docx_document = DocxDocument(str(docx_path))
    headings = [p.text.strip() for p in docx_document.paragraphs if p.style.name.startswith("Heading")]
    assert headings == [DOCX_SECTION_0, DOCX_SECTION_1]
    assert len(docx_document.tables) == 2
    assert len(docx_document.tables[1].rows) == 0, "空表格必须是 0 行"


def test_index_contract(benchmark):
    """索引规模与元数据契约：4 份文档入索引，空白 PDF 被跳过，空表格不入索引。"""
    corpus = benchmark["report"]["corpus"]

    assert corpus["file_count"] == len(CORPUS_SOURCES)
    assert corpus["skipped_count"] == 1, "空白 PDF 必须被记为 skipped"
    assert corpus["empty_file_count"] == 0
    assert sorted(corpus["indexed_chunk_counts"]) == sorted(CORPUS_SOURCES)
    assert BLANK_PDF_NAME not in corpus["indexed_chunk_counts"]

    # 每个源至少 1 个 chunk，总量必须大于 RETRIEVAL_TOP_K，
    # 否则 K=10 会覆盖全部语料，指标失去区分度。
    assert corpus["chunk_count"] >= RETRIEVAL_TOP_K + 5, (
        f"语料 chunk 数过少：{corpus['chunk_count']}，K=10 将覆盖几乎全部语料"
    )

    # PDF 三页各自独立 chunk；DOCX 只有 table_index=0 存在（空表格=1 被跳过）
    pdf_pages = sorted(
        chunk["page"] for chunk in corpus["chunks"] if chunk["source"] == PDF_NAME
    )
    assert pdf_pages == [1, 2, 3]
    docx_tables = [
        chunk["table_index"]
        for chunk in corpus["chunks"]
        if chunk["source"] == DOCX_NAME and chunk["table_index"] is not None
    ]
    assert docx_tables == [0], f"DOCX 表格编号异常：{docx_tables}"
    docx_sections = [
        chunk["section"]
        for chunk in corpus["chunks"]
        if chunk["source"] == DOCX_NAME and chunk["section"] is not None
    ]
    assert DOCX_SECTION_0 in docx_sections and DOCX_SECTION_1 in docx_sections


def test_real_embedding_is_used(benchmark):
    """真实本地 BGE：禁止 stub embedding 参与正式评估。"""
    embedding = benchmark["report"]["embedding"]
    assert embedding["class"] == "Embedding"
    assert embedding["dim"] == EMBEDDING_DIM_EXPECTED
    assert benchmark["kb"].is_indexed is True


def test_gold_evidence_resolves_in_index(benchmark):
    """gold evidence 必须能在索引里反查到 chunk，且该 chunk 真的含有预期事实。"""
    report = benchmark["report"]
    for case in report["cases"]:
        if not case["answerable"]:
            assert case["gold_resolved"] == []
            continue
        assert case["gold_resolved"], f"{case['case_id']} gold 未解析"
        assert case["gold_id_count"] >= 1
        facts = case["expected_fact"]
        if not isinstance(facts, list):
            facts = [facts]
        for fact, item in zip(facts, case["gold_resolved"]):
            assert any(fact in chunk["chunk_text"] for chunk in item["chunks"]), (
                f"{case['case_id']} gold chunk 不含事实 {fact!r}"
            )


# ======================================================================
# 9. 检索与指标
# ======================================================================


def test_every_query_returns_single_ranking(benchmark):
    """每条 query 只检索一次，返回按分数降序的 Top-K，无重复 chunk。"""
    report = benchmark["report"]
    assert report["search_calls"]["total"] == len(CASES) + 1

    for case in report["cases"]:
        retrieval = case["retrieval"]
        assert retrieval["ok"], f"{case['case_id']} 检索失败：{retrieval['error']}"
        assert retrieval["count"] == len(retrieval["top_k"])
        assert retrieval["count"] == min(RETRIEVAL_TOP_K, report["corpus"]["chunk_count"])

        scores = [item["score"] for item in retrieval["top_k"]]
        assert scores == sorted(scores, reverse=True), f"{case['case_id']} 未按分数降序"

        identities = [(item["document_id"], item["chunk_seq"]) for item in retrieval["top_k"]]
        assert len(identities) == len(set(identities)), f"{case['case_id']} 结果有重复 chunk"


def test_recall_metrics(benchmark):
    """Recall@1/3/5/10 微 / 宏两口径均位于 [0,1]，且随 K 单调不减。

    微平均恒等于 ``Σhits / Σ|Gold|``；宏平均恒等于 ``mean(hits/|Gold_i|)``。
    两者在 ``|Gold_i|`` 不一致时（本数据集有 2 条跨文档问题）必须不同，
    这里用独立算式逐点复核，而不是相信聚合函数。
    """
    report = benchmark["report"]
    aggregate = report["aggregate"]
    answerable = [case for case in report["cases"] if case["answerable"]]

    gold_total = sum(case["gold_id_count"] for case in answerable)
    assert gold_total == aggregate["evidence_totals"]["gold_total"]
    assert len(answerable) == aggregate["answerable_count"]
    # V1 数据集无重复 Gold：条目数必须等于去重后的分母
    assert (
        aggregate["evidence_totals"]["gold_total_entries"]
        == aggregate["evidence_totals"]["gold_total_unique"]
    ), "存在重复 Gold Evidence，分母会被放大"

    for k in KS:
        hits = sum(case["metrics"]["hits@%d" % k] for case in answerable)
        micro = hits / gold_total
        macro = sum(
            case["metrics"]["hits@%d" % k] / case["gold_id_count"]
            for case in answerable
        ) / len(answerable)

        assert abs(aggregate["recall_micro"]["@%d" % k] - micro) < 1e-12
        assert abs(aggregate["recall_macro"]["@%d" % k] - macro) < 1e-12
        assert 0.0 <= micro <= 1.0 and 0.0 <= macro <= 1.0

        # 每条问题的 recall 必须等于 hits/|Gold|
        for case in answerable:
            expected = case["metrics"]["hits@%d" % k] / case["gold_id_count"]
            assert abs(case["metrics"]["recall@%d" % k] - expected) < 1e-12

    for key in ("recall_micro", "recall_macro"):
        ordered = [aggregate[key]["@%d" % k] for k in sorted(KS)]
        for previous, current in zip(ordered, ordered[1:]):
            assert current >= previous - 1e-12, f"{key} 未随 K 单调不减"

    # 存在 |Gold|>1 的问题时，微 / 宏在某个 K 上必须出现差异
    if aggregate["evidence_totals"]["multi_gold_cases"]:
        assert any(
            abs(
                aggregate["recall_micro"]["@%d" % k]
                - aggregate["recall_macro"]["@%d" % k]
            )
            > 1e-12
            for k in KS
        ), "存在多证据问题，微 / 宏两口径不可能完全相同"

    for case in report["cases"]:
        if not case["answerable"]:
            assert "recall@10" not in case["metrics"]
            assert "hits@10" not in case["metrics"]
            continue
        values = [case["metrics"]["recall@%d" % k] for k in sorted(KS)]
        for previous, current in zip(values, values[1:]):
            assert current >= previous - 1e-12, f"{case['case_id']} 单条 Recall 非单调"


def test_precision_metrics(benchmark):
    """Precision@K 三口径：macro、micro(fixed)、micro(returned)。"""
    report = benchmark["report"]
    aggregate = report["aggregate"]
    answerable = [case for case in report["cases"] if case["answerable"]]
    n = len(answerable)

    for k in KS:
        hits = sum(case["metrics"]["hits@%d" % k] for case in answerable)
        macro = sum(case["metrics"]["hits@%d" % k] / k for case in answerable) / n
        micro_fixed = hits / (n * k)
        returned_denom = sum(
            min(k, len(case["retrieval"]["top_k"])) for case in answerable
        )
        micro_returned = hits / returned_denom

        assert abs(aggregate["precision_macro"]["@%d" % k] - macro) < 1e-12
        assert abs(aggregate["precision_micro"]["@%d" % k] - micro_fixed) < 1e-12
        assert (
            abs(aggregate["precision_micro_returned"]["@%d" % k] - micro_returned)
            < 1e-12
        )
        for key in ("precision_macro", "precision_micro", "precision_micro_returned"):
            assert 0.0 <= aggregate[key]["@%d" % k] <= 1.0

        # 每条返回数都 ≥ K 时，三个口径必然相同（本例语料 22 > K=10）
        if all(len(case["retrieval"]["top_k"]) >= k for case in answerable):
            assert abs(macro - micro_fixed) < 1e-12
            assert abs(micro_fixed - micro_returned) < 1e-12


def test_mrr(benchmark):
    """MRR = 平均(1/首个相关名次)；未命中记 0，分母只含可回答问题。"""
    report = benchmark["report"]
    rr_values = [
        case["metrics"]["reciprocal_rank"]
        for case in report["cases"]
        if case["answerable"]
    ]
    assert len(rr_values) == report["aggregate"]["answerable_count"]
    assert all(value is not None and 0.0 <= value <= 1.0 for value in rr_values)
    expected = sum(rr_values) / len(rr_values)
    assert abs(report["aggregate"]["mrr"] - expected) < 1e-12
    assert 0.0 <= report["aggregate"]["mrr"] <= 1.0
    assert report["aggregate"]["mrr_zero_count"] == sum(
        1 for value in rr_values if value == 0.0
    )


def test_citation_locator_three_layers(benchmark):
    """三层定位：文档级命中 ⊇ chunk 级命中；字段级只在命中的记录上统计。"""
    report = benchmark["report"]

    for k in KS:
        block = report["aggregate"]["citation"]["@%d" % k]
        # 每层都必须给出统计样本数，否则指标无法判断置信度
        assert block["document_sample"] == report["aggregate"]["answerable_count"]
        assert block["chunk_sample"] == report["aggregate"]["answerable_count"]

        assert block["document_accuracy"] is not None
        assert block["chunk_accuracy"] is not None
        # 命中 chunk 必然命中文档：chunk 准确率不得高于文档准确率
        assert block["chunk_accuracy"] <= block["document_accuracy"] + 1e-12

        if block["locator_field_sample"]:
            assert block["locator_field_accuracy"] is not None
            assert 0.0 <= block["locator_field_accuracy"] <= 1.0
            assert block["locator_field_ok"] <= block["locator_field_sample"]
        else:
            # 没有任何 gold chunk 命中时，字段级指标无定义，不能伪造 0 或 1
            assert block["locator_field_accuracy"] is None

    # 逐条：字段级明细必须与命中记录一一对应
    for case in report["cases"]:
        if not case["answerable"]:
            assert case["locator_checks"] == []
            continue
        hit_ranks = {
            item["rank"]
            for item in case["retrieval"]["top_k"]
            if item["is_gold"]
        }
        assert {check["rank"] for check in case["locator_checks"]} == hit_ranks


def test_unsupported_retrieval_rate(benchmark):
    """无证据问题：Top-K 中返回的每条都计为非相关，比例 = 返回数 / K。"""
    report = benchmark["report"]
    unanswerable = [case for case in report["cases"] if not case["answerable"]]
    assert len(unanswerable) == 2

    for case in unanswerable:
        for k in KS:
            returned = min(k, len(case["retrieval"]["top_k"]))
            expected = returned / k
            assert abs(case["metrics"]["unsupported@%d" % k] - expected) < 1e-12

    for k in KS:
        value = report["aggregate"]["unsupported_retrieval_rate"]["@%d" % k]
        assert value is not None and 0.0 <= value <= 1.0


def test_report_is_written(benchmark):
    """评估报告落盘，且包含每条 query 的 Top-K 明细与 gold evidence。"""
    report_path = Path(benchmark["summary"]["report_path"])
    assert report_path.exists()

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["benchmark_version"] == BENCHMARK_VERSION
    assert len(payload["cases"]) == len(CASES)
    for case in payload["cases"]:
        assert "gold_evidence" in case
        assert len(case["retrieval"]["top_k"]) == RETRIEVAL_TOP_K
        for item in case["retrieval"]["top_k"]:
            assert item["source"] is not None
            assert item["document_id"] is not None
            assert item["chunk_seq"] is not None
            assert 0.0 <= item["score"] <= 1.0
    assert "locator_field_accuracy" in payload["aggregate"]["citation"]["@10"]


# ======================================================================
# 10. 指标口径的独立单元测试（P2-3E-STEP1-R1）
#
# 本节**完全不依赖 benchmark 夹具**：不加载 BGE、不建索引、不执行任何检索、
# 不触网。全部用人工构造的 Gold / Rank 样例验证宏微平均、跨文档多证据、
# 未命中、重复 Gold、无证据排除、Precision 分母。
# ======================================================================

# 复用与夹具相同的 KS，但用固定的小 K 集合让算式可手算。
_TEST_KS = (1, 3)


def test_aggregate_micro_vs_macro_divergence():
    """|Gold| 不一致时，微平均与宏平均必然不同（且各自数值可手算）。

    构造：3 条可回答问题
      A: |Gold|=2, K=1 命中 1  → per-query 0.5
      B: |Gold|=1, K=1 命中 1  → 1.0
      C: |Gold|=1, K=1 命中 0  → 0.0

    K=1:  micro = (1+1+0)/4 = 0.5000
          macro = (0.5+1.0+0.0)/3 = 0.5000
    K=3:  A 命中 2、B 命中 1、C 命中 1  → Σhits=4
          micro = 4/4 = 1.0000
          macro = (2/2 + 1/1 + 1/1)/3 = 1.0000
    再构造一个能拉开差异的样例：|Gold|=2 且只命中 1 条是唯一失分点，
    此时 micro 把 2 条证据各算一次，macro 只把它算成 1/3 条问题。
    """
    rows = [
        {"case_id": "A", "gold_size": 2, "hits": {1: 1, 3: 2}, "returned": 10},
        {"case_id": "B", "gold_size": 1, "hits": {1: 1, 3: 1}, "returned": 10},
        {"case_id": "C", "gold_size": 1, "hits": {1: 0, 3: 1}, "returned": 10},
    ]
    result = _aggregate_answerable(rows, ks=_TEST_KS)

    assert result["gold_total"] == 4
    assert abs(result["recall_micro"]["@1"] - 0.5) < 1e-12
    assert abs(result["recall_macro"]["@1"] - 0.5) < 1e-12

    # 让差异显式出现：A 的 |Gold|=2 在微平均里占两条证据
    rows2 = [
        {"case_id": "A", "gold_size": 2, "hits": {1: 1, 3: 1}, "returned": 10},
        {"case_id": "B", "gold_size": 1, "hits": {1: 1, 3: 1}, "returned": 10},
    ]
    result2 = _aggregate_answerable(rows2, ks=_TEST_KS)
    # micro K=1: (1+1)/3 = 0.6667 ；macro K=1: (0.5+1.0)/2 = 0.75
    assert abs(result2["recall_micro"]["@1"] - 2 / 3) < 1e-12
    assert abs(result2["recall_macro"]["@1"] - 0.75) < 1e-12
    assert result2["recall_micro"]["@1"] < result2["recall_macro"]["@1"]

    # 两者都必须随 K 单调不减
    for key in ("recall_micro", "recall_macro"):
        assert result[key]["@3"] >= result[key]["@1"] - 1e-12


def test_aggregate_cross_document_requires_both_evidence():
    """跨文档问题必须同时命中两个证据才算完全召回（|Gold|=2）。

    - 只命中 1 条 → recall = 0.5（不是 1.0）；RR = 1（首个相关结果即命中）；
    - 命中 2 条 → recall = 1.0。
    """
    partial = _aggregate_answerable(
        [{"case_id": "X", "gold_size": 2, "hits": {1: 1, 3: 1}, "returned": 10}],
        ks=_TEST_KS,
    )
    assert abs(partial["recall_macro"]["@1"] - 0.5) < 1e-12
    assert abs(partial["recall_micro"]["@3"] - 0.5) < 1e-12

    full = _aggregate_answerable(
        [{"case_id": "X", "gold_size": 2, "hits": {1: 2, 3: 2}, "returned": 10}],
        ks=_TEST_KS,
    )
    assert abs(full["recall_macro"]["@1"] - 1.0) < 1e-12
    assert abs(full["recall_micro"]["@1"] - 1.0) < 1e-12

    # 排名函数层面：两条证据都在 Top-1 是不可能的，但 Top-3 可以
    gold = {("d1", 0), ("d2", 1)}
    ranking = [
        {"rank": 1, "identity": ("d1", 0)},
        {"rank": 2, "identity": ("d3", 0)},
        {"rank": 3, "identity": ("d2", 1)},
    ]
    assert _hits_at_k(ranking, gold, 1) == 1
    assert _hits_at_k(ranking, gold, 3) == 2
    assert _recall_at_k(ranking, gold, 1) == 0.5
    assert _recall_at_k(ranking, gold, 3) == 1.0
    assert _reciprocal_rank(ranking, gold) == 1.0


def test_aggregate_miss_gives_zero_rr_and_zero_mrr():
    """完全未命中的问题：RR=0，且单条问题独占数据集时 MRR=0。"""
    ranking = [
        {"rank": 1, "identity": ("d9", 0)},
        {"rank": 2, "identity": ("d9", 1)},
    ]
    gold = {("d1", 0)}
    assert _hits_at_k(ranking, gold, 10) == 0
    assert _recall_at_k(ranking, gold, 10) == 0.0
    assert _reciprocal_rank(ranking, gold) == 0.0
    assert _precision_at_k(ranking, gold, 2) == 0.0

    # 两条问题：一条命中 rank1、一条全未命中 → MRR = 0.5
    mrr_values = [
        _reciprocal_rank([{"rank": 1, "identity": ("g", 0)}], {("g", 0)}),
        _reciprocal_rank(ranking, gold),
    ]
    assert abs(sum(mrr_values) / 2 - 0.5) < 1e-12


def test_duplicate_gold_evidence_not_double_counted():
    """同一 (document_id, chunk_seq) 出现两次只计一次，不放大分母。"""
    duplicated = [
        {
            "resolved": [
                {"identity": ("d1", 0), "document_id": "d1", "chunk_seq": 0},
                {"identity": ("d1", 0), "document_id": "d1", "chunk_seq": 0},
                {"identity": ("d2", 1), "document_id": "d2", "chunk_seq": 1},
            ]
        }
    ]
    identities = _gold_identity_set(duplicated)
    assert identities == {("d1", 0), ("d2", 1)}
    assert len(identities) == 2

    # 去重后 |Gold|=2，命中 1 条 → 0.5（若未去重会算成 1/3）
    ranking = [{"rank": 1, "identity": ("d1", 0)}]
    assert _hits_at_k(ranking, identities, 1) == 1
    assert _recall_at_k(ranking, identities, 1) == 0.5


def test_unanswerable_excluded_from_recall_and_mrr_denominators():
    """无证据问题不得进入 Recall / MRR / Precision 的分母。"""
    mixed = [
        {
            "case_id": "a1",
            "answerable": True,
            "gold_id_count": 1,
            "metrics": {"hits@1": 1, "hits@3": 1, "hits@5": 1, "hits@10": 1},
            "retrieval": {"ranking": [{"rank": 1, "identity": ("d1", 0)}]},
        },
        {
            "case_id": "neg",
            "answerable": False,
            "gold_id_count": 0,
            "metrics": {},
            "retrieval": {"ranking": [{"rank": 1, "identity": ("d9", 0)}]},
        },
    ]
    rows = _rows_from_cases(mixed)
    assert [row["case_id"] for row in rows] == ["a1"]

    result = _aggregate_answerable(rows, ks=(1,))
    assert result["n"] == 1
    assert result["gold_total"] == 1
    assert abs(result["recall_micro"]["@1"] - 1.0) < 1e-12
    assert abs(result["recall_macro"]["@1"] - 1.0) < 1e-12
    assert abs(result["precision_macro"]["@1"] - 1.0) < 1e-12


def test_precision_denominator_variants():
    """Precision 三口径的分母差异：返回数 < K 时 returned 口径才与固定 K 不同。"""
    rows = [
        {"case_id": "A", "gold_size": 1, "hits": {1: 1, 3: 1}, "returned": 3},
        {"case_id": "B", "gold_size": 1, "hits": {1: 0, 3: 1}, "returned": 3},
    ]
    result = _aggregate_answerable(rows, ks=(1, 3))

    # K=3：hits=2，固定分母 2*3=6 → 0.3333；returned 分母 3+3=6 → 同值
    assert abs(result["precision_micro"]["@3"] - 2 / 6) < 1e-12
    assert abs(result["precision_micro_returned"]["@3"] - 2 / 6) < 1e-12

    # K=5 且返回 3 条时，两口径分离：hits=2
    rows_short = [
        {"case_id": "A", "gold_size": 1, "hits": {5: 1}, "returned": 3},
        {"case_id": "B", "gold_size": 1, "hits": {5: 1}, "returned": 3},
    ]
    short = _aggregate_answerable(rows_short, ks=(5,))
    assert abs(short["precision_micro"]["@5"] - 2 / 10) < 1e-12
    assert abs(short["precision_micro_returned"]["@5"] - 2 / 6) < 1e-12
    assert short["precision_micro_returned"]["@5"] > short["precision_micro"]["@5"]


# ======================================================================
# 11. 对既有报告的独立复算（只读 JSON，不重跑检索）
# ======================================================================

_REPORT_PATH_ENV = "RAG_BENCHMARK_REPORT"
_DEFAULT_REPORT_PATH = (
    r"C:\Users\Doro\AppData\Local\Temp\p2_3e_step1_basetemp_ec1195ce13ae4d13a1c3333db88a82d2"
    r"\rag-benchmark-v10\evidence\benchmark_report.json"
)


def _existing_report():
    """读取一份既有报告；路径可用环境变量覆盖，缺失则跳过（不伪造数据）。"""
    path = Path(os.environ.get(_REPORT_PATH_ENV) or _DEFAULT_REPORT_PATH)
    if not path.exists():
        pytest.skip(f"既有报告不存在，跳过只读复算：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _report_hits_at_k(case, k):
    """从既有报告取 hits@K：新报告直接有，旧报告由 recall×|Gold| 反推。"""
    key = "hits@%d" % k
    if key in case["metrics"]:
        return case["metrics"][key]
    if "recall@%d" % k in case["metrics"]:
        return int(round(case["metrics"]["recall@%d" % k] * case["gold_id_count"]))
    raise AssertionError(f"{case['case_id']} 缺少 {key}")


def test_existing_report_gold_evidence_integrity():
    """既有报告：Gold 无重复、is_gold 与 Gold 集合一致、无证据问题 Gold 为空。"""
    payload = _existing_report()

    total_gold = 0
    for case in payload["cases"]:
        identities = []
        for item in case.get("gold_resolved") or []:
            for chunk in item["chunks"]:
                identities.append((chunk["document_id"], chunk["chunk_seq"]))
        unique = set(identities)
        assert len(unique) == len(identities), f"{case['case_id']} Gold Evidence 重复"
        assert case["gold_id_count"] == len(unique)

        recovered = set()
        for record in case["retrieval"]["top_k"]:
            identity = (record["document_id"], record["chunk_seq"])
            assert bool(record["is_gold"]) == (identity in unique), (
                f"{case['case_id']} rank={record['rank']} 的 is_gold 与 Gold 集合不一致"
            )
            if identity in unique:
                recovered.add(identity)

        if not case["answerable"]:
            assert unique == set()
            assert all(not record["is_gold"] for record in case["retrieval"]["top_k"])
        total_gold += len(unique)

    answerable = sum(1 for case in payload["cases"] if case["answerable"])
    assert answerable + sum(
        1 for case in payload["cases"] if not case["answerable"]
    ) == len(payload["cases"])

    stored = payload["aggregate"].get("evidence_totals", {})
    if stored:
        assert stored["gold_total"] == total_gold
    # 与旧报告对照：Σ|Gold| 必须等于「单证据问题数 + 2×跨文档问题数」
    multi = sum(1 for case in payload["cases"] if case["gold_id_count"] > 1)
    single = answerable - multi
    assert total_gold == single + 2 * multi


def test_existing_report_aggregates_recompute():
    """用独立实现（不复用 _aggregate_answerable）复算既有报告的聚合指标。"""
    payload = _existing_report()
    aggregate = payload["aggregate"]
    cases = payload["cases"]
    answerable = [case for case in cases if case["answerable"]]
    n = len(answerable)
    gold_total = sum(case["gold_id_count"] for case in answerable)

    def stored(primary, legacy=None):
        if legacy and legacy in aggregate:
            return aggregate[legacy]
        return aggregate[primary]

    def block_value(block, primary, legacy):
        """兼容旧报告字段名（retrieval_source_accuracy / locator_checks 等）。"""
        return block[primary] if primary in block else block[legacy]

    recall_macro = stored("recall_macro", "recall")
    precision_macro = stored("precision_macro", "precision_fixed")

    for k in KS:
        hits = sum(_report_hits_at_k(case, k) for case in answerable)
        micro = hits / gold_total
        macro = sum(
            _report_hits_at_k(case, k) / case["gold_id_count"] for case in answerable
        ) / n
        assert abs(recall_macro["@%d" % k] - macro) < 1e-9, "宏平均 Recall 不符"
        if "recall_micro" in aggregate:
            assert abs(aggregate["recall_micro"]["@%d" % k] - micro) < 1e-9

        p_macro = sum(_report_hits_at_k(case, k) / k for case in answerable) / n
        assert abs(precision_macro["@%d" % k] - p_macro) < 1e-9
        if "precision_micro" in aggregate:
            assert abs(aggregate["precision_micro"]["@%d" % k] - hits / (n * k)) < 1e-9

    # MRR
    rr_total = 0.0
    for case in answerable:
        gold = {
            (chunk["document_id"], chunk["chunk_seq"])
            for item in case["gold_resolved"]
            for chunk in item["chunks"]
        }
        rank = 0
        for record in case["retrieval"]["top_k"]:
            if (record["document_id"], record["chunk_seq"]) in gold:
                rank = record["rank"]
                break
        rr_total += 1.0 / rank if rank else 0.0
    assert abs(aggregate["mrr"] - rr_total / n) < 1e-9

    # Citation 三层：文档级 / chunk 级 / 字段级（含样本数）
    for k in KS:
        doc_hits = 0
        chunk_hits = 0
        checks = 0
        ok = 0
        for case in answerable:
            expected_sources = case["expected_source"]
            if isinstance(expected_sources, str):
                expected_sources = [expected_sources]
            gold = {
                (chunk["document_id"], chunk["chunk_seq"]): item
                for item in case["gold_resolved"]
                for chunk in item["chunks"]
            }
            top = case["retrieval"]["top_k"][:k]
            if any(record["source"] in expected_sources for record in top):
                doc_hits += 1
            if any(
                (record["document_id"], record["chunk_seq"]) in gold for record in top
            ):
                chunk_hits += 1
            for record in top:
                key = (record["document_id"], record["chunk_seq"])
                if key not in gold:
                    continue
                locator = gold[key]["locator"]
                checks += 1
                if all(record.get(name) == value for name, value in locator.items()):
                    ok += 1

        block = aggregate["citation"]["@%d" % k]
        assert abs(
            block_value(block, "document_accuracy", "retrieval_source_accuracy")
            - doc_hits / n
        ) < 1e-9
        assert abs(
            block_value(block, "chunk_accuracy", "retrieval_chunk_accuracy")
            - chunk_hits / n
        ) < 1e-9
        if checks:
            assert abs(block["locator_field_accuracy"] - ok / checks) < 1e-9
        assert block_value(block, "locator_field_sample", "locator_checks") == checks
        if "document_sample" in block:
            assert block["document_sample"] == n

    # Negative：不进入任何分母，且必须单独报告
    negative = [case for case in cases if not case["answerable"]]
    assert len(negative) == aggregate["unanswerable_count"]
    for case in negative:
        assert case["gold_evidence"] == {}
        scores = [record["score"] for record in case["retrieval"]["top_k"]]
        assert len(scores) == len(case["retrieval"]["top_k"])

    diagnostics = aggregate.get("negative_query_diagnostics")
    if diagnostics is None:
        # 旧报告只存最高分诊断：核对最高分均值，其余字段不存在属已知缺口
        max_mean = sum(
            case["retrieval"]["top_k"][0]["score"] for case in negative
        ) / len(negative)
        assert (
            abs(aggregate["score_diagnostic"]["max_score_mean_unanswerable"] - max_mean)
            < 1e-9
        )
    else:
        assert diagnostics["threshold_defined"] is False
        reported = {item["case_id"] for item in diagnostics["cases"]}
        assert reported == {case["case_id"] for case in negative}
        for case in negative:
            entry = next(
                item
                for item in diagnostics["cases"]
                if item["case_id"] == case["case_id"]
            )
            scores = [record["score"] for record in case["retrieval"]["top_k"]]
            assert entry["returned"] == len(scores)
            assert abs(entry["max_score"] - max(scores)) < 1e-12
            assert abs(entry["min_score"] - min(scores)) < 1e-12
            assert abs(entry["mean_score"] - sum(scores) / len(scores)) < 1e-12
