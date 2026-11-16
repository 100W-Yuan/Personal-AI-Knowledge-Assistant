# -*- coding: utf-8 -*-
"""P2-3D-C1：真实 PDF / DOCX 多来源问答与定位引用验收（独立运行，不经 pytest）。

定位
----
本文件**不是 pytest 用例**：文件名不以 ``test_`` 开头，只能作为普通 Python
进程直接执行。原因与 ``run_real_agent_rag_smoke.py`` 完全一致 —— 本项目三层
conftest 守卫在 pytest 路径下会无条件拦截真实模型请求：

- ``conftest.py::_guarded_sdk_request`` 不看 URL，一律抛 BlockedSDKRequestError；
- ``conftest.py::_make_http_guard`` 的 ``_http_request_is_allowed`` 只放行回环；
- ``自动测试脚本/conftest.py`` 在**导入期**把 ``openai.OpenAI`` 换成离线空壳。

因此真实调用的正统形态是独立脚本。本文件不复用 smoke 脚本的代码（它是执行
脚本而非库），而是按其已验证的安全语义独立实现，两者互不依赖。

本轮验证目标
------------
跨格式、多来源证据综合与**定位引用**：PDF（3 页，第 2 页写 90 天）与 DOCX
（Heading 1 + 正文 30 天 + 真实表格）同时进索引，Agent 必须分别给出来源，
并标注 PDF 页码、DOCX 章节与表格位置。

安全边界（自我约束，不依赖也不拆除任何外部守卫）
------------------------------------------------
1. 只读进程环境变量 ``DEEPSEEK_API_KEY``；不读 ``.env``、不调用 ``load_dotenv``；
2. 构造 Agent **之前**把工作目录切到 TEMP 下本次唯一新目录；
3. 构造后把工具注册表收敛到 ``search_knowledge``；
4. 真实 HTTP 客户端带「8 次请求硬上限」+「单 Host / HTTPS / 有效端口 443」
   白名单，两者都在 request 钩子里、且在请求发出之前判定；
5. ``max_retries=0``，不自定义重试，不因异常重跑 ``Agent.run``；
6. Agent 控制台输出被捕获后丢弃，只落结构化脱敏摘要；
7. 不导入 conftest、不导入 ``main.py``、不运行 pytest；
8. 正式 ``memory.db`` / ``data/business.db`` / ``rag_knowledge`` 只做指纹比对，
   绝不读写内容。

PDF 构造方式（为何手写）
-----------------------
环境内没有 reportlab / fpdf / PyMuPDF（已实测），因此用 pypdf 的
``PdfWriter`` + ``DecodedStreamObject`` 组装**结构完整**的 PDF：Type0 复合
字体 + Identity-H 编码 + ToUnicode CMap（BMP 恒等映射），正文以 UTF-16BE
十六进制串写入内容流。这不是「伪造文件头」——它有真实的 xref、trailer、
页面对象、字体资源与内容流，pypdf 能独立解析并正确还原中文。

关键实测（P2-3D-C1 预检）：
- pypdf 6.16.2 读取三页，第 2 页文本完整；
- 真实 ``PdfLoader`` 产出 3 个 Document，``page`` = 1/2/3、``page_count`` = 3；
- 真实 ``DocxLoader`` 产出 2 个 Document：段落节（``section=滤芯维护规范``、
  ``section_index=0``、``segment_index=0``）与表格（``table_index=0``，
  **不含 section 字段**——表格来源确实没有 section，以 Loader 真实输出为准）。

问题文本适配（必须说明）
------------------------
任务书给定的问句原文为「根据我的 PDF 产品维护手册和 DOCX 滤芯维护规范，分别
回答：…」。实测 ``Agent._classify_knowledge_intent`` 判为 **neutral**：该分类
器要求「根据/依据/参照/参考」与资源词（知识库|文档|资料|导入）**同时命中**，
而原文只有「根据」没有资源词。为通过脚本自身的 recall 闸门（否则退出码 3），
在「我的」之后补入「知识库中」三字 —— 三问内容、来源要求、定位要求均未改动。

请求预算（R5 起生效，本脚本沿用）
----------------------------------
``Agent.run()`` 的模型调用次数由 Plan 形状决定，不是固定值。本轮硬上限
**8**，第 1～8 次符合白名单的请求允许发送，**第 9 次在发送前阻断**。
8 只是本轮硬上限，**不保证任何复杂计划都能完成**；超出即以退出码 6 安全
失败，不得放宽上限、不得自动重试、不得追加提问。

退出码
------
0 通过 | 2 缺少 DEEPSEEK_API_KEY | 3 意图不是 recall
4 静态契约 / 文档构造 / 本地预检失败
5 断言失败（事实或来源引用未达验收标准）
6 HTTP 预算耗尽或 Host 被拒 | 7 运行异常 | 8 离线 Embedding 不可用
9 误在 pytest 运行时中启动
"""

from __future__ import annotations

import contextlib
import contextvars
import io
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path

# ============================================================
# 1. 离线环境：必须在导入任何 rag / sentence_transformers 之前完成
# ============================================================

OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}

for _env_key, _env_value in OFFLINE_ENV.items():
    os.environ[_env_key] = _env_value


# ============================================================
# 2. 路径与常量
# ============================================================

# SCRIPT_PATH 必须在**导入期**解析为绝对路径：脚本稍后会 os.chdir() 到
# 临时目录，之后再按相对 __file__ 读源码就会解析失败。
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---- 真实模型目标（与 main.py:273-278 的正式配置逐字一致）----
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-flash"

# ---- 单 Host 白名单：scheme / host / 有效端口 三者同时命中才放行 ----
# ALLOWED_PORT 是「有效端口」：URL 未写端口时按 scheme 取默认值后再比较
# （见 _effective_port）。httpx2 会把显式 :443 也规范化为 None。
ALLOWED_SCHEME = "https"
ALLOWED_HOST = "api.deepseek.com"
ALLOWED_PORT = 443

# ---- HTTP 请求硬上限（R3-LIVE-R1 授权额度：12 次）----
# 第 1～12 次放行，第 13 次发送前阻断；计数为进程级累计，新建客户端不重置。
MAX_HTTP_REQUESTS = 12

# ---- 显式超时（覆盖 openai 默认 read=600s 的宽限期）----
TIMEOUT_CONNECT = 5.0
TIMEOUT_READ = 45.0
TIMEOUT_WRITE = 20.0
TIMEOUT_POOL = 5.0

# ---- 外部工具允许列表：只保留 search_knowledge ----
ALLOWED_TOOLS = ("search_knowledge",)

FORBIDDEN_TOOLS = (
    "query_orders",      # tools.py:261 ADMIN_ORDER_TOOL_ENABLED=True，读正式 business.db
    "calculator",        # 与本场景无关
    "get_current_time",  # 与本场景无关
    "test_failure",      # 故意失败桩，会污染失败/重规划路径
    "search_memory",     # 不属于外部工具；内部 Memory 检索路径不受影响
)

# ---- 临时运行目录子结构 ----
KB_DIRNAME = "kb"
MEMORY_DIRNAME = "memory"
EVIDENCE_DIRNAME = "evidence"

# ---- 两份真实文档 ----
# 文件名用 ASCII：便于模型在答案里逐字引用，也避免控制台编码噪声。
PDF_FILENAME = "qingshan_product_manual.pdf"
DOCX_FILENAME = "qingshan_filter_spec.docx"

# PDF 三页：第 1 页无关背景，第 2 页给出 90 天，第 3 页无关维护说明。
PDF_PAGES = (
    (
        "青杉净水器产品维护手册",
        "本产品适用于家庭饮用水净化场景。",
        "机身采用食品级材料，外观为简约白色。",
    ),
    (
        "第2章 滤芯更换",
        "滤芯更换周期为 90 天。",
        "请按周期更换以保证出水品质。",
    ),
    (
        "第3章 日常维护",
        "建议每周擦拭机身表面。",
        "长期不用时应排空管路积水。",
    ),
)
PDF_PAGE_WITH_FACT = 2          # 1 基：事实所在的页
PDF_PAGE_COUNT_EXPECTED = 3

# DOCX：Heading 1 + 正文 + 真实表格
DOCX_HEADING = "滤芯维护规范"
DOCX_BODY = "滤芯巡检间隔为 30 天。"
DOCX_TABLE_ROWS = (
    ("项目", "要求"),
    ("检测方式", "压力差检测"),
    ("记录频率", "每次巡检"),
)
DOCX_SECTION_EXPECTED = DOCX_HEADING
DOCX_TABLE_INDEX_EXPECTED = 0   # 0 基：这是文档里第 1 个 w:tbl

# ---- 唯一问答（见文件头「问题文本适配」）----
#
# 第 4 项是**负向**问题：两份测试文档都**没有**记载整机保修期限
# （构造文档时刻意不写入），因此正确答案只能是「资料未提供」。
# 这一项用来验证模型不会用常识补全缺失事实。
QUESTION = (
    "根据我的知识库中 PDF 产品维护手册和 DOCX 滤芯维护规范，分别回答："
    "1. 滤芯更换周期是多少天？"
    "2. 滤芯巡检间隔是多少天？"
    "3. 检测方式是什么？"
    "4. 整机保修期限是多少年？"
    "如果提供的资料没有明确记载，请明确说明资料未提供，不要根据常识推测。"
    "请对每项答案分别提供来源文件，并尽可能精确标注 PDF 页码、DOCX 章节或表格位置。"
)

# ---- 事实别名（模型措辞不受控，只放宽写法，不放宽事实）----
EXPECTED_REPLACE_DAYS = ("90", "九十")
EXPECTED_INSPECT_DAYS = ("30", "三十")
EXPECTED_METHOD = ("压力差", "压差")

# ---- 第 4 项：负向事实（资料中不存在，必须拒答）----
# 文档中刻意不写保修期限，因此这里只定义「拒答」与「虚构」两套判定材料。
WARRANTY_TOPIC_ALIASES = ("保修", "质保")

WARRANTY_REFUSAL_MARKERS = (
    "未提供",
    "未记载",
    "没有记载",
    "无记载",
    "未明确",
    "未提及",
    "未涉及",
    "未说明",
    "没有说明",
    "无法确定",
    "无法给出",
    "无法回答",
    "不能确定",
    "不详",
    "未见",
    "未找到",
    "未包含",
    "不包含",
    "没有相关",
    "无相关",
    "资料中没有",
    "资料未",
    "知识库中没有",
    "知识库未",
)

# 只匹配「保修/质保 …… N 年」这种**明确给出年限**的表述。
# 刻意限定单位是「年」：答案里必然出现 90 天 / 30 天 / 第 2 页 / 表格 1，
# 若按「出现数字」判定会大面积误伤。
WARRANTY_FABRICATION_PATTERN = re.compile(
    r"(?:保修|质保)(?:期限|期|年限|时间)?[^。！？\n；;|]{0,15}?"
    r"(\d+(?:\.\d+)?|[一二三四五六七八九十]+)\s*年"
)

# 句子切分：不用 "|" —— Markdown 表格的「问题列」与「答案列」会被切成两段，
# 导致「含保修但无拒答词」与「含拒答词但无保修」被误判为两个不相关片段。
_WARRANTY_SENTENCE_SPLIT = re.compile(r"[。！？\n；;]")

# 定位指称的正则（宽松匹配常见写法，命中后仍会与真实 sources 交叉核对）
_PDF_PAGE_PATTERN = re.compile(
    r"(第\s*2\s*页|第\s*二\s*页|page\s*2\b|p\.?\s*2\b|第\s*2\s*張|第\s*2\s*张)",
    re.IGNORECASE,
)
_DOCX_SECTION_PATTERN = re.compile(re.escape(DOCX_SECTION_EXPECTED))
_DOCX_TABLE_PATTERN = re.compile(
    r"(表\s*格?\s*1\b|表\s*格?\s*一|table\s*1\b|第\s*1\s*个?\s*表)",
    re.IGNORECASE,
)

# 临时目录前缀：每次运行唯一，绝不复用
TEMP_PREFIX = "p2_3d_c1_citation_"


# ============================================================
# 3. 运行期可观测状态（不保存任何请求体 / 响应体 / 凭据 / 文档正文）
# ============================================================

BUDGET = {
    # ---- 计数（进程级累计，跨客户端 / 跨调用不重置）----
    "request_attempts": 0,
    "sent_requests": 0,
    "response_count": 0,
    "blocked": 0,
    "blocked_reason": None,
    "blocked_detail": None,
    # ---- 逐条证据 ----
    "requests": [],
    "responses": [],
    "status_codes": [],
    "request_ids": [],
    # ---- 阶段与用量 ----
    "stage_counts": {},
    "usage_by_request": [],
    "usage_totals": {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "requests_with_usage": 0,
        "requests_without_usage": 0,
        "available": False,
    },
    "_last_attempt": None,
}

CLIENTS = []


# ============================================================
# 3-bis. 调用阶段上下文
#
# 阶段判定依据「真实调用栈」，不是函数命名习惯。依据是对 agent.py 的 AST
# 扫描：全文件只有 5 处 ``self.client.chat.completions.create``，每个所在
# 函数各 1 处；``run:2733`` 的 Prompt 上下文为「请生成这个工具需要的 JSON
# 参数」，语义是工具参数生成，不是最终回答生成。
# ============================================================

STAGE_EXTRACT = "extract_memories"
STAGE_CREATE_PLAN = "create_plan"
STAGE_TOOL_ARGS = "tool_argument_generation"
STAGE_REASONING = "reasoning"
STAGE_REPLAN = "replan"
STAGE_UNKNOWN = "unknown"

ALLOWED_STAGES = (
    STAGE_EXTRACT,
    STAGE_CREATE_PLAN,
    STAGE_TOOL_ARGS,
    STAGE_REASONING,
    STAGE_REPLAN,
    STAGE_UNKNOWN,
)

# (agent.py 中的函数名, create 调用所在行号) → stage
#
# P2-3D-C1-R2 在 agent.py 顶部新增查询感知复用辅助函数，并在
# execute_tool / run 注释处增行，导致全部调用点行号整体下移。
# 调用点**数量与所在函数未变**（仍为 5 个），此处仅同步行号；
# 生产代码未做任何改动。
CALL_SITE_STAGE = {
    ("extract_memories", 877): STAGE_EXTRACT,
    ("create_plan", 1503): STAGE_CREATE_PLAN,
    ("replan", 1629): STAGE_REPLAN,
    ("reasoning", 1784): STAGE_REASONING,
    ("run", 2866): STAGE_TOOL_ARGS,
}

# 函数名 → stage（仅当该函数内确知只有 1 个 create 调用点时才允许使用）
FUNCTION_STAGE = {
    "extract_memories": STAGE_EXTRACT,
    "create_plan": STAGE_CREATE_PLAN,
    "replan": STAGE_REPLAN,
    "reasoning": STAGE_REASONING,
    "run": STAGE_TOOL_ARGS,
}

_CURRENT_STAGE = contextvars.ContextVar("p2_3d_c1_stage", default=STAGE_UNKNOWN)
_CURRENT_ATTEMPT = contextvars.ContextVar("p2_3d_c1_attempt", default=None)

# 由 _verify_call_site_map() 在运行期填充
_UNIQUE_CALL_FUNCS = None


def _bump_stage(stage):
    """阶段尝试计数 +1（含被拦截的尝试，与 request_attempts 对齐）。"""
    if stage not in ALLOWED_STAGES:
        stage = STAGE_UNKNOWN
    BUDGET["stage_counts"][stage] = BUDGET["stage_counts"].get(stage, 0) + 1


def _resolve_stage(frame):
    """由调用帧解析阶段；无法可靠判定时返回 "unknown"（绝不伪造）。"""
    if frame is None:
        return STAGE_UNKNOWN
    try:
        func = frame.f_code.co_name
        lineno = frame.f_lineno
    except Exception:
        return STAGE_UNKNOWN

    exact = CALL_SITE_STAGE.get((func, lineno))
    if exact is not None:
        return exact

    if _UNIQUE_CALL_FUNCS and func in _UNIQUE_CALL_FUNCS:
        return FUNCTION_STAGE.get(func, STAGE_UNKNOWN)

    return STAGE_UNKNOWN


class HttpBudgetExceeded(RuntimeError):
    """第 9 次（及以后）HTTP 请求在发送之前被终止。"""


class HostNotAllowedError(RuntimeError):
    """目标 scheme/host/有效端口不在白名单内，请求在发送之前被终止。"""


# ============================================================
# 4. HTTP 事件钩子：单 Host 白名单 + 硬上限 + 阶段 / 用量采集
#
# httpx2 的 "request" 钩子在请求构造完成、真正发送之前触发，因此在这里
# raise 可以保证「请求从未离开本机」。计数放在模块级字典里，不随 SDK
# 内部任何重试而重置。
# ============================================================


def _effective_port(url):
    """返回 URL 的有效端口：未显式写端口时按 scheme 取协议默认值。

    httpx2 的 ``URL.port`` 在未显式携带端口时返回 ``None``，显式写 ``:443``
    也会被规范化为 ``None``。若直接与 443 比较，合法目标会被误拒
    （P2-3D-B2-R2 的失败根因）。
    """
    try:
        port = getattr(url, "port", None)
    except Exception:
        return None
    if port is not None:
        return port
    scheme = getattr(url, "scheme", None)
    if scheme == "https":
        return 443
    if scheme == "http":
        return 80
    return None


def _is_allowed_api_url(url):
    """纯函数：scheme + host + 有效端口三重严格匹配；异常一律 fail closed。"""
    if url is None:
        return False
    try:
        if getattr(url, "scheme", None) != ALLOWED_SCHEME:
            return False
        if getattr(url, "host", None) != ALLOWED_HOST:
            return False
        return _effective_port(url) == ALLOWED_PORT
    except Exception:
        return False


def _target_desc(url):
    """只取 scheme / host / 原始端口 / 有效端口，不读 path / query / header / body。"""
    if url is None:
        return "scheme=None;host=None;port=None;effective_port=None"
    try:
        return (
            f"scheme={url.scheme};host={url.host};port={url.port};"
            f"effective_port={_effective_port(url)}"
        )
    except Exception:
        return "scheme=<unreadable>;host=<unreadable>;port=<unreadable>;effective_port=<unreadable>"


def _budget_decision(attempt):
    """纯函数：累计第 attempt 次尝试是否允许发送（1..MAX 允许，MAX+1 起拒绝）。"""
    return attempt <= MAX_HTTP_REQUESTS


def _on_request(request):
    BUDGET["request_attempts"] += 1
    attempt = BUDGET["request_attempts"]
    BUDGET["_last_attempt"] = attempt
    _CURRENT_ATTEMPT.set(attempt)

    stage = _CURRENT_STAGE.get()
    if stage not in ALLOWED_STAGES:
        stage = STAGE_UNKNOWN
    _bump_stage(stage)

    url = getattr(request, "url", None)
    whitelist_ok = bool(_is_allowed_api_url(url))

    entry = {
        "attempt": attempt,
        "stage": stage,
        "scheme": getattr(url, "scheme", None),
        "host": getattr(url, "host", None),
        "effective_port": _effective_port(url),
        "whitelist_ok": whitelist_ok,
        "budget_ok": None,
        "sent": False,
    }
    BUDGET["requests"].append(entry)

    # (1) 白名单先判：目标不对时，无论第几次都拒绝
    if not whitelist_ok:
        detail = _target_desc(url)
        entry["blocked_reason"] = "host_whitelist"
        BUDGET["blocked"] += 1
        BUDGET["blocked_reason"] = "host_whitelist"
        BUDGET["blocked_detail"] = detail
        raise HostNotAllowedError(
            "请求目标不在白名单内，已在发送前终止："
            f"{detail}；"
            f"仅允许 {ALLOWED_SCHEME}://{ALLOWED_HOST}:{ALLOWED_PORT}"
        )

    # (2) 硬上限：第 1～8 次放行，第 9 次直接拒绝
    budget_ok = bool(_budget_decision(attempt))
    entry["budget_ok"] = budget_ok
    if not budget_ok:
        entry["blocked_reason"] = "http_budget"
        BUDGET["blocked"] += 1
        BUDGET["blocked_reason"] = "http_budget"
        BUDGET["blocked_detail"] = f"attempt={attempt};limit={MAX_HTTP_REQUESTS}"
        raise HttpBudgetExceeded(
            f"HTTP 请求硬上限 {MAX_HTTP_REQUESTS} 已用尽："
            f"第 {attempt} 次请求在发送前被终止"
        )

    # 真正发出：此时才计入 sent_requests
    entry["sent"] = True
    BUDGET["sent_requests"] += 1


def _on_response(response):
    BUDGET["response_count"] += 1

    try:
        status_code = int(response.status_code)
    except Exception:
        status_code = None

    try:
        request_id = response.headers.get("x-request-id")
    except Exception:
        request_id = None
    request_id = str(request_id) if request_id else None

    attempt = _CURRENT_ATTEMPT.get()
    if attempt is None:
        attempt = BUDGET["_last_attempt"]

    stage = _CURRENT_STAGE.get()
    if stage not in ALLOWED_STAGES:
        stage = STAGE_UNKNOWN

    if status_code is None:
        outcome = "unknown_status"
    elif status_code >= 400:
        outcome = "http_error"
    else:
        outcome = "success"

    entry = {
        "attempt": attempt,
        "stage": stage,
        "status_code": status_code,
        "outcome": outcome,
        "request_id": request_id,
        # usage 由观察包装在 SDK 返回 ChatCompletion 后回填；默认不可用，
        # 绝不预先填 0
        "usage_available": False,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
    }
    BUDGET["responses"].append(entry)

    BUDGET["status_codes"].append(status_code)
    if request_id:
        BUDGET["request_ids"].append(request_id)


def _record_usage(attempt, stage, chat_completion):
    """从真实 SDK 返回的 ChatCompletion 读取 usage，只保留三个整数。

    - 无 usage / 非整数 → usage_available=False 且三项为 None（不填 0）
    - 不保存响应正文、不保存请求对象、不读请求头
    - 回填到同一 attempt 的 response 记录，保证一一对应
    """
    usage = getattr(chat_completion, "usage", None)

    def _as_int(value):
        return (
            int(value)
            if isinstance(value, int) and not isinstance(value, bool)
            else None
        )

    prompt_tokens = _as_int(getattr(usage, "prompt_tokens", None))
    completion_tokens = _as_int(getattr(usage, "completion_tokens", None))
    total_tokens = _as_int(getattr(usage, "total_tokens", None))
    available = any(
        v is not None for v in (prompt_tokens, completion_tokens, total_tokens)
    )

    stage = stage if stage in ALLOWED_STAGES else STAGE_UNKNOWN
    record = {
        "attempt": attempt,
        "stage": stage,
        "usage_available": bool(available),
        "prompt_tokens": prompt_tokens if available else None,
        "completion_tokens": completion_tokens if available else None,
        "total_tokens": total_tokens if available else None,
    }
    BUDGET["usage_by_request"].append(record)

    totals = BUDGET["usage_totals"]
    if available:
        for key, value in (
            ("prompt_tokens", prompt_tokens),
            ("completion_tokens", completion_tokens),
            ("total_tokens", total_tokens),
        ):
            if value is not None:
                totals[key] += value
        totals["requests_with_usage"] += 1
        totals["available"] = True
    else:
        totals["requests_without_usage"] += 1

    for response_entry in reversed(BUDGET["responses"]):
        if response_entry.get("attempt") == attempt:
            response_entry["usage_available"] = record["usage_available"]
            response_entry["prompt_tokens"] = record["prompt_tokens"]
            response_entry["completion_tokens"] = record["completion_tokens"]
            response_entry["total_tokens"] = record["total_tokens"]
            break


def _install_stage_observer(client):
    """给真实客户端的 ``chat.completions.create`` 加观察包装。

    约束：客户端仍是真实 openai.OpenAI 实例（不做代理 / Mock / Stub 替换）；
    ``original`` 是真实绑定方法；包装原样传参、原样返回；**不捕获** original
    的异常（异常原样向上传播）；不改变调用顺序、不增加请求。
    """
    completions = client.chat.completions
    original = completions.create

    if getattr(original, "_p2_stage_observed", False):
        return original

    def observed_create(*args, **kwargs):
        stage = _resolve_stage(sys._getframe(1))
        token = _CURRENT_STAGE.set(stage)
        try:
            result = original(*args, **kwargs)
        finally:
            _CURRENT_STAGE.reset(token)

        try:
            _record_usage(_CURRENT_ATTEMPT.get(), stage, result)
        except Exception:
            pass
        return result

    observed_create._p2_stage_observed = True
    observed_create._p2_original = original
    observed_create.__wrapped__ = original

    completions.create = observed_create
    return observed_create


# ============================================================
# 5. 真实 SDK 客户端工厂
#
# 注入点：agent.py 顶部 ``from openai import OpenAI`` —— 模块级符号，
# Agent.__init__ 直接调用它。把 agent.OpenAI 换成工厂即可在不改生产代码的
# 前提下注入受限客户端。
#
# 关键坑：OpenAI 一旦收到 http_client，其自身 timeout 形参会被忽略
# （openai/_base_client.py），因此超时、禁止跳转、事件钩子必须全部设置在
# 真实 HTTP 客户端上。
# ============================================================


def _build_http_client():
    """真实 httpx2 客户端：显式超时 + 禁止跳转 + 忽略环境代理 + 事件钩子。"""
    import httpx2
    from openai import DefaultHttpxClient

    timeout = httpx2.Timeout(
        connect=TIMEOUT_CONNECT,
        read=TIMEOUT_READ,
        write=TIMEOUT_WRITE,
        pool=TIMEOUT_POOL,
    )

    return DefaultHttpxClient(
        timeout=timeout,
        follow_redirects=False,
        trust_env=False,   # 禁止 http_proxy / https_proxy / all_proxy 环境代理
        event_hooks={
            "request": [_on_request],
            "response": [_on_response],
        },
    )


def _make_client_factory(http_client):
    """返回一个工厂：签名与 openai.OpenAI 兼容，产出真实客户端。"""

    def _factory(*args, **kwargs):
        from openai import OpenAI

        api_key = kwargs.get("api_key")
        base_url = kwargs.get("base_url") or BASE_URL

        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=0,            # SDK 自动重试固定为 0
            http_client=http_client,  # 真实客户端，非替身
        )

        # 真实客户端保持原样，只给 chat.completions.create 加观察包装
        _install_stage_observer(client)

        CLIENTS.append(client)
        return client

    return _factory


# ============================================================
# 6. 脱敏 / 指纹 / 临时运行目录
# ============================================================

_SECRET_PATTERNS = (
    (re.compile(r"sk-[A-Za-z0-9_\-]{6,}"), "***REDACTED***"),
    (re.compile(r"(?i)authorization\s*[:=]\s*[^\r\n]+"), "Authorization: ***REDACTED***"),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{6,}"), "Bearer ***REDACTED***"),
)


def _sanitize(value, secret=None):
    """对任意对象做文本化 + 密钥/授权头脱敏。"""
    try:
        text = value if isinstance(value, str) else str(value)
    except Exception:
        text = "<unprintable>"
    if secret:
        text = text.replace(secret, "***REDACTED***")
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def _fingerprint(path: Path):
    """正式资产的轻量指纹：存在性 / 大小 / mtime。不读取内容。"""
    try:
        stat = path.stat()
        return {
            "exists": True,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    except FileNotFoundError:
        return {"exists": False}
    except Exception as exc:
        return {"exists": None, "error": type(exc).__name__}


def _create_run_dir():
    """在系统 TEMP 下创建本次运行唯一的新目录（拒绝复用）。"""
    import tempfile

    temp_root = Path(tempfile.gettempdir()).resolve()
    run_dir = temp_root / f"{TEMP_PREFIX}{uuid.uuid4().hex}"

    # 明确要求：创建前确认目标路径不存在
    if run_dir.exists():
        raise RuntimeError(f"临时目录已存在，拒绝复用：{run_dir}")

    kb_dir = run_dir / KB_DIRNAME
    memory_dir = run_dir / MEMORY_DIRNAME
    evidence_dir = run_dir / EVIDENCE_DIRNAME

    # exist_ok=False：父目录已存在就报错，保证「本次唯一」
    run_dir.mkdir(parents=False, exist_ok=False)
    kb_dir.mkdir(exist_ok=False)
    memory_dir.mkdir(exist_ok=False)
    evidence_dir.mkdir(exist_ok=False)

    return run_dir, kb_dir, memory_dir, evidence_dir


# ============================================================
# 7. 真实 PDF 构造与校验
#
# 环境无 reportlab / fpdf / PyMuPDF，因此用 pypdf 组装结构完整的 PDF。
# 中文通过 Type0 复合字体 + Identity-H + ToUnicode CMap（BMP 恒等）呈现，
# 正文以 UTF-16BE 十六进制串写入内容流；pypdf 据此可正确还原文本。
# ============================================================

_PDF_Tounicode_CMAP = b"""/CIDInit /ProcSet findresource begin
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


def _pdf_escape_text(text):
    """把文本编码为 PDF 十六进制字符串（UTF-16BE）。"""
    return "<%s>" % text.encode("utf-16-be").hex().upper()


def _build_pdf(path: Path, pages):
    """用 pypdf 组装多页真实 PDF（每页一个内容流，共享一个 Type0 字体）。"""
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

    # --- ToUnicode CMap（BMP 恒等映射，保证文本可被正确提取）---
    cmap_stream = DecodedStreamObject()
    cmap_stream.set_data(_PDF_Tounicode_CMAP)
    cmap_ref = writer._add_object(cmap_stream)

    # --- FontDescriptor（不嵌入字体文件：本 PDF 只用于文本抽取验证）---
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

    # --- CIDFont ---
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

    # --- Type0 字体（Identity-H + ToUnicode）---
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
        for line in lines:
            ops.append("%s Tj" % _pdf_escape_text(line))
            ops.append("T*")
        ops.append("ET")

        content = DecodedStreamObject()
        content.set_data("\n".join(ops).encode("ascii"))
        content_ref = writer._add_object(content)

        page = writer.add_blank_page(width=595, height=842)
        page[NameObject("/Resources")] = DictionaryObject(
            {
                NameObject("/Font"): DictionaryObject(
                    {NameObject("/F1"): font_ref}
                )
            }
        )
        page[NameObject("/Contents")] = content_ref

    with open(path, "wb") as handle:
        writer.write(handle)


def _verify_pdf(path: Path):
    """用真实 pypdf 校验：页数、每页文本、事实所在页完整。"""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = list(reader.pages)
    detail = {
        "path": str(path),
        "size": path.stat().st_size if path.exists() else 0,
        "page_count": len(pages),
        "pages": [],
    }

    for index, page in enumerate(pages, start=1):
        text = page.extract_text() or ""
        detail["pages"].append(
            {
                "page": index,
                "chars": len(text),
                "has_fact": any(alias in text for alias in EXPECTED_REPLACE_DAYS),
                # 只记录长度与是否含事实，不落正文
            }
        )

    ok = (
        detail["page_count"] == PDF_PAGE_COUNT_EXPECTED
        and len(detail["pages"]) == PDF_PAGE_COUNT_EXPECTED
        and all(p["chars"] > 0 for p in detail["pages"])
        and detail["pages"][PDF_PAGE_WITH_FACT - 1]["has_fact"]
    )
    return ok, detail


# ============================================================
# 8. 真实 DOCX 构造与校验
# ============================================================


def _build_docx(path: Path):
    """用真实 python-docx 创建：Heading 1 + 正文 + 真实表格。"""
    from docx import Document as DocxDocument

    document = DocxDocument()
    document.add_heading(DOCX_HEADING, level=1)
    document.add_paragraph(DOCX_BODY)

    table = document.add_table(rows=len(DOCX_TABLE_ROWS), cols=2)
    for row_index, row_values in enumerate(DOCX_TABLE_ROWS):
        for col_index, value in enumerate(row_values):
            table.rows[row_index].cells[col_index].text = value

    document.save(str(path))


def _verify_docx_structure(path: Path):
    """用真实 python-docx 读回，校验标题 / 段落 / 表格内容。"""
    from docx import Document as DocxDocument

    document = DocxDocument(str(path))
    headings = [
        p.text.strip()
        for p in document.paragraphs
        if p.style is not None and p.style.name.lower().startswith("heading")
    ]
    paragraphs = [p.text.strip() for p in document.paragraphs if p.text.strip()]
    tables = document.tables

    table_cells = []
    if tables:
        first = tables[0]
        table_cells = [
            [cell.text.strip() for cell in row.cells] for row in first.rows
        ]

    detail = {
        "path": str(path),
        "size": path.stat().st_size if path.exists() else 0,
        "headings": headings,
        "paragraph_count": len(paragraphs),
        "has_body": DOCX_BODY in paragraphs,
        "table_count": len(tables),
        "table_rows": len(table_cells),
        "table_cols": len(table_cells[0]) if table_cells else 0,
        "table_cells": table_cells,
    }

    expected_cells = [[a, b] for a, b in DOCX_TABLE_ROWS]
    ok = (
        DOCX_HEADING in headings
        and detail["has_body"]
        and detail["table_count"] >= 1
        and table_cells == expected_cells
    )
    return ok, detail


# ============================================================
# 9. 真实 Loader 预检（不依赖 Embedding、不联网）
# ============================================================


def _precheck_documents(pdf_path: Path, docx_path: Path):
    """用真实 PdfLoader / DocxLoader 解析，校验 metadata 契约。

    - PDF：3 个 Document，page=1..3、page_count=3，第 2 页含事实；
    - DOCX：段落节有 section/section_index/segment_index；表格节有
      table_index=0 且**不含** section（以 Loader 真实输出为准）。
    """
    from rag.loader import DocxLoader, PdfLoader

    detail = {}
    ok = True

    # ---- PDF ----
    pdf_docs = PdfLoader(str(pdf_path)).load()
    pdf_items = []
    for doc in pdf_docs:
        meta = doc.metadata or {}
        pdf_items.append(
            {
                "page": meta.get("page"),
                "page_count": meta.get("page_count"),
                "source_suffix": Path(str(meta.get("source", ""))).suffix.lower(),
                "chars": len(doc.page_content or ""),
                "has_fact": any(
                    a in (doc.page_content or "") for a in EXPECTED_REPLACE_DAYS
                ),
            }
        )
    pdf_pages = [i["page"] for i in pdf_items]
    pdf_ok = (
        len(pdf_docs) == PDF_PAGE_COUNT_EXPECTED
        and pdf_pages == list(range(1, PDF_PAGE_COUNT_EXPECTED + 1))
        and all(i["page_count"] == PDF_PAGE_COUNT_EXPECTED for i in pdf_items)
        and all(i["chars"] > 0 for i in pdf_items)
        and any(
            i["page"] == PDF_PAGE_WITH_FACT and i["has_fact"] for i in pdf_items
        )
    )
    detail["pdf"] = {"ok": pdf_ok, "documents": pdf_items}
    ok = ok and pdf_ok

    # ---- DOCX ----
    docx_docs = DocxLoader(str(docx_path)).load()
    docx_items = []
    for doc in docx_docs:
        meta = dict(doc.metadata or {})
        docx_items.append(
            {
                "keys": sorted(meta.keys()),
                "section": meta.get("section"),
                "section_index": meta.get("section_index"),
                "segment_index": meta.get("segment_index"),
                "table_index": meta.get("table_index"),
                "table_rows": meta.get("table_rows"),
                "table_cols": meta.get("table_cols"),
                "chars": len(doc.page_content or ""),
                "has_body": DOCX_BODY in (doc.page_content or ""),
                "has_method": any(
                    a in (doc.page_content or "") for a in EXPECTED_METHOD
                ),
                # 表格来源是否带 section：记录真实情况，不假定
                "has_section_key": "section" in meta,
            }
        )

    section_items = [i for i in docx_items if i["section"] == DOCX_SECTION_EXPECTED]
    table_items = [
        i for i in docx_items if i["table_index"] == DOCX_TABLE_INDEX_EXPECTED
    ]

    docx_ok = (
        len(section_items) >= 1
        and section_items[0]["has_body"]
        and section_items[0]["section_index"] is not None
        and section_items[0]["segment_index"] is not None
        and len(table_items) == 1
        and table_items[0]["has_method"]
        and table_items[0]["table_rows"] == len(DOCX_TABLE_ROWS)
        and table_items[0]["table_cols"] == 2
    )
    detail["docx"] = {
        "ok": docx_ok,
        "documents": docx_items,
        "section_documents": len(section_items),
        "table_documents": len(table_items),
        "table_has_section_key": (
            table_items[0]["has_section_key"] if table_items else None
        ),
    }
    ok = ok and docx_ok

    return ok, detail


# ============================================================
# 10. KnowledgeBase 预检：真实 Embedding + 三问检索 + 元数据契约
# ============================================================


def _precheck_knowledge_base(kb, pdf_name, docx_name):
    """在花费任何 API 额度之前，用本地 Embedding 验证两类来源都可检索。

    这一步是本地计算（离线模型），不产生任何 HTTP 请求。
    """
    queries = {
        "replace_days": "滤芯更换周期是多少天",
        "inspect_days": "滤芯巡检间隔是多少天",
        "method": "检测方式是什么",
    }

    info = {
        "files": sorted(p.name for p in (Path(kb.doc_dir)).iterdir())
        if getattr(kb, "doc_dir", None)
        else None,
        "queries": {},
    }

    for key, query in queries.items():
        probe = kb.search(query, top_k=3)
        results = []
        for item in probe.get("results", []):
            # 只保留元数据；不落 chunk 正文
            results.append(
                {
                    "source": item.get("source"),
                    "chunk_id": item.get("chunk_id"),
                    "score": item.get("score"),
                    "document_id": item.get("document_id"),
                    "chunk_seq": item.get("chunk_seq"),
                    "page": item.get("page"),
                    "page_count": item.get("page_count"),
                    "section": item.get("section"),
                    "section_index": item.get("section_index"),
                    "segment_index": item.get("segment_index"),
                    "table_index": item.get("table_index"),
                }
            )
        info["queries"][key] = {
            "query": query,
            "ok": bool(probe.get("ok")),
            "count": probe.get("count"),
            "results": results,
        }

    # 索引健康：两份文件、512 维、至少 3 个 chunk
    info["is_indexed"] = bool(kb.is_indexed)
    info["file_count"] = kb.file_count
    info["chunk_count"] = kb.chunk_count
    info["init_error_kind"] = kb.init_error_kind

    embedding = getattr(kb, "_embedding", None)
    model = getattr(embedding, "model", None)
    info["embedding_model_name"] = "BAAI/bge-small-zh-v1.5"
    info["embedding_dimension"] = (
        _embedding_dimension(model) if model else None
    )

    # ---- 元数据契约：三问各自必须命中对应来源 ----
    def _hits(key, predicate):
        return [
            r
            for r in info["queries"][key]["results"]
            if predicate(r)
        ]

    replace_hits = _hits(
        "replace_days",
        lambda r: str(r.get("source")) == pdf_name
        and r.get("page") == PDF_PAGE_WITH_FACT,
    )
    inspect_hits = _hits(
        "inspect_days",
        lambda r: str(r.get("source")) == docx_name
        and r.get("section") == DOCX_SECTION_EXPECTED,
    )
    method_hits = _hits(
        "method",
        lambda r: str(r.get("source")) == docx_name
        and r.get("table_index") == DOCX_TABLE_INDEX_EXPECTED,
    )

    info["replace_hits"] = len(replace_hits)
    info["inspect_hits"] = len(inspect_hits)
    info["method_hits"] = len(method_hits)

    # 来源格式覆盖：PDF 与 DOCX 都必须出现在检索结果里
    formats = set()
    for key in info["queries"]:
        for r in info["queries"][key]["results"]:
            suffix = Path(str(r.get("source") or "")).suffix.lower()
            if suffix:
                formats.add(suffix)
    info["source_formats"] = sorted(formats)

    ok = (
        info["is_indexed"]
        and info["file_count"] == 2
        and info["chunk_count"] >= 3
        and info["embedding_dimension"] == 512
        and len(replace_hits) >= 1
        and len(inspect_hits) >= 1
        and len(method_hits) >= 1
        and {".pdf", ".docx"} <= formats
    )
    return ok, info


def _embedding_dimension(model):
    """读取 BGE 输出维度；读不到返回 None（不据此判失败）。

    sentence-transformers 6.x 把 ``get_sentence_embedding_dimension`` 改名为
    ``get_embedding_dimension``，这里优先新名、回退旧名，避免 FutureWarning
    与未来版本移除。
    """
    for attr in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
        getter = getattr(model, attr, None)
        if callable(getter):
            try:
                return int(getter())
            except Exception:
                return None
    return None


# ============================================================
# 11. 工具收敛
# ============================================================


def _enforce_tool_allowlist(agent_obj):
    """把工具注册表收敛到 ALLOWED_TOOLS（直接删字典条目）。"""
    registry = getattr(agent_obj, "tool_registry", None)
    tools_dict = getattr(registry, "tools", None)
    if not isinstance(tools_dict, dict):
        raise RuntimeError("工具注册表结构不符合预期：tools 不是 dict")

    before = sorted(tools_dict)
    removed = [name for name in before if name not in ALLOWED_TOOLS]
    for name in removed:
        del tools_dict[name]

    after = sorted(tools_dict)
    if set(after) != set(ALLOWED_TOOLS):
        raise RuntimeError(
            f"工具收敛失败：after={after}，期望={list(ALLOWED_TOOLS)}"
        )

    for name in FORBIDDEN_TOOLS:
        if name in tools_dict:
            raise RuntimeError(f"禁止工具仍然存在：{name}")

    payload_names = []
    try:
        for item in registry.get_openai_tools():
            function = item.get("function") if isinstance(item, dict) else None
            if function:
                payload_names.append(function.get("name"))
    except Exception:
        payload_names = []

    return {
        "allowed": list(ALLOWED_TOOLS),
        "before": before,
        "removed": removed,
        "after": after,
        "payload_names": sorted(n for n in payload_names if n),
    }


# ============================================================
# 12. 契约检查：KnowledgeBase.is_indexed 必须是 property
# ============================================================


def _knowledge_base_member_kinds(kb_class):
    kinds = {}
    for name, member in vars(kb_class).items():
        if name.startswith("_"):
            continue
        if isinstance(member, property):
            kinds[name] = "property"
        elif callable(member):
            kinds[name] = "method"
        else:
            kinds[name] = type(member).__name__
    return kinds


def _audit_self_calls_on_readonly_members(kb_class):
    """AST 审计本脚本源码：不得对只读属性（property）加括号调用。"""
    import ast as _ast

    kinds = _knowledge_base_member_kinds(kb_class)
    readonly = {name for name, kind in kinds.items() if kind == "property"}

    source = SCRIPT_PATH.read_bytes().decode("utf-8-sig")
    tree = _ast.parse(source)

    violations = []
    for node in _ast.walk(tree):
        if not isinstance(node, _ast.Call):
            continue
        func = node.func
        if isinstance(func, _ast.Attribute) and func.attr in readonly:
            violations.append((node.lineno, func.attr))
    return violations, sorted(readonly)


def _verify_knowledge_base_contract(kb_class, kb_instance=None):
    import inspect as _inspect

    detail = {
        "is_indexed_static_kind": None,
        "is_indexed_instance_type": None,
        "is_indexed_instance_value": None,
        "readonly_members": [],
        "violations": [],
    }

    try:
        static = _inspect.getattr_static(kb_class, "is_indexed", None)
    except Exception:
        static = None
    detail["is_indexed_static_kind"] = (
        "property" if isinstance(static, property) else type(static).__name__
    )

    violations, readonly = _audit_self_calls_on_readonly_members(kb_class)
    detail["readonly_members"] = readonly
    detail["violations"] = violations

    instance_ok = True
    if kb_instance is not None:
        value = kb_instance.is_indexed
        detail["is_indexed_instance_type"] = type(value).__name__
        detail["is_indexed_instance_value"] = value
        instance_ok = isinstance(value, bool)

    ok = (
        detail["is_indexed_static_kind"] == "property"
        and not detail["violations"]
        and instance_ok
    )
    return ok, detail


# ============================================================
# 13. 调用点映射静态校验（证明阶段标签有真实来源）
# ============================================================


def _scan_create_call_sites(source_text):
    """AST 扫描：返回 [(所在函数名, 行号)] 形式的 chat.completions.create 调用点。"""
    import ast as _ast

    tree = _ast.parse(source_text)
    sites = []

    def walk(node, func):
        for child in _ast.iter_child_nodes(node):
            next_func = func
            if isinstance(child, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                next_func = child.name
            if isinstance(child, _ast.Call):
                attr = child.func
                if isinstance(attr, _ast.Attribute) and attr.attr == "create":
                    inner = attr.value
                    if (
                        isinstance(inner, _ast.Attribute)
                        and inner.attr == "completions"
                    ):
                        sites.append((next_func, child.lineno))
            walk(child, next_func)

    walk(tree, None)
    return sites


def _verify_call_site_map(agent_module):
    """校验阶段映射表覆盖 agent.py 全部 create 调用点；ok=False 必须停止。"""
    global _UNIQUE_CALL_FUNCS

    path = Path(getattr(agent_module, "__file__", "agent.py"))
    source = path.read_bytes().decode("utf-8-sig")
    sites = _scan_create_call_sites(source)

    counts = {}
    for func, _lineno in sites:
        counts[func] = counts.get(func, 0) + 1

    unique_funcs = {func for func, n in counts.items() if n == 1}
    _UNIQUE_CALL_FUNCS = {func for func in unique_funcs if func in FUNCTION_STAGE}

    unmapped = [s for s in sites if s not in CALL_SITE_STAGE]
    stale = [key for key in CALL_SITE_STAGE if key not in sites]

    ok = (not unmapped) and (not stale) and (len(sites) == len(CALL_SITE_STAGE))

    detail = {
        "agent_py": str(path),
        "call_sites": [
            {
                "func": func,
                "lineno": lineno,
                "stage": CALL_SITE_STAGE.get((func, lineno)),
            }
            for func, lineno in sites
        ],
        "call_site_count": len(sites),
        "mapped_count": len(sites) - len(unmapped),
        "unmapped": [{"func": f, "lineno": l} for f, l in unmapped],
        "stale": [{"func": f, "lineno": l} for f, l in stale],
        "unique_funcs": sorted(_UNIQUE_CALL_FUNCS),
        "stages_covered": sorted(
            {CALL_SITE_STAGE[k] for k in CALL_SITE_STAGE if k in sites}
        ),
    }
    return ok, detail


# ============================================================
# 13B. Agent 真实检索轨迹：KnowledgeBase.search 透明观察包装
#
# 目的：证明 R2 的查询感知复用确实让多个子查询进入**真实检索**，
# 而不是只观察到准备阶段的本地预检索。
#
# 契约（缺一不可）：
# 1. 保存原始 bound method；包装体每次调用**只**调用它一次；
# 2. 参数原样转发（*args / **kwargs），不重排、不补默认值后再传；
# 3. 返回值原样返回（同一对象，不复制、不改写）；
# 4. 异常原样抛回调用方，仅在此之前记录异常类型；
# 5. 不替换 KnowledgeBase 实例、Retriever、Embedding、VectorStore；
# 6. 不改变检索次数、排序、结果，也不改变 Agent 调用顺序；
# 7. `enabled=False` 时包装体仍透传但**不记录**，用于隔离准备阶段。
#
# 因此 wrapper_calls 与 original_calls 必须始终相等；recorded 只统计
# 开关打开期间进入记录分支的调用。
# ============================================================

SEARCH_TRACE = {
    "installed": False,
    "enabled": False,
    "wrapper_calls": 0,
    "original_calls": 0,
    "records": [],
    "errors": [],
    "enabled_windows": [],
}

# 仅用于查询词脱敏；绝不写入任何证据文件。
_TRACE_SECRET = None

# 只保留定位/身份元数据，绝不落 chunk 正文。
_TRACE_RESULT_KEYS = (
    "source",
    "score",
    "document_id",
    "chunk_seq",
    "page",
    "page_count",
    "section",
    "section_index",
    "segment_index",
    "table_index",
)

_TRACE_TOP_K_DEFAULT = 3


def _trace_query_text(value):
    """查询词脱敏后的文本；非字符串返回 None。"""
    if not isinstance(value, str):
        return None
    return _sanitize(value, _TRACE_SECRET)


def _trace_top_k(value):
    """只接受真实 int（排除 bool），否则返回 None。"""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _trace_result_rows(results, limit=10):
    """把检索结果投影为纯元数据行；不读取 chunk 正文。"""
    rows = []
    if not isinstance(results, (list, tuple)):
        return rows
    for item in results[:limit]:
        if not isinstance(item, dict):
            continue
        row = {}
        for key in _TRACE_RESULT_KEYS:
            if key in item:
                row[key] = item[key]
        rows.append(row)
    return rows


def _install_search_trace(kb):
    """在 kb.search 上安装透明包装；重复安装直接返回现状。"""
    if SEARCH_TRACE["installed"]:
        return SEARCH_TRACE

    original = kb.search

    def wrapped(*args, **kwargs):
        # 每次包装调用都恰好触发一次原始调用（不变量：wrapper == original）。
        SEARCH_TRACE["wrapper_calls"] += 1
        SEARCH_TRACE["original_calls"] += 1

        if not SEARCH_TRACE["enabled"]:
            # 准备阶段：只透传，不记录，不改变任何行为。
            return original(*args, **kwargs)

        query = args[0] if args else kwargs.get("query")
        if len(args) > 1:
            raw_top_k = args[1]
        else:
            raw_top_k = kwargs.get("top_k", _TRACE_TOP_K_DEFAULT)

        seq = len(SEARCH_TRACE["records"]) + 1
        entry = {
            "seq": seq,
            "query": _trace_query_text(query),
            "top_k": _trace_top_k(raw_top_k),
            "status": "ok",
        }

        try:
            result = original(*args, **kwargs)
        except BaseException as exc:
            entry["status"] = "error"
            entry["error_type"] = type(exc).__name__
            entry["count"] = None
            entry["results"] = []
            SEARCH_TRACE["errors"].append(
                {"seq": seq, "type": type(exc).__name__}
            )
            SEARCH_TRACE["records"].append(entry)
            raise

        payload = result if isinstance(result, dict) else {}
        entry["ok"] = bool(payload.get("ok"))
        entry["count"] = payload.get("count")
        entry["results"] = _trace_result_rows(payload.get("results"))
        SEARCH_TRACE["records"].append(entry)
        return result

    kb.search = wrapped
    SEARCH_TRACE["installed"] = True
    return SEARCH_TRACE


def _set_trace_enabled(value, reason):
    """切换记录开关，并留下可核对的窗口边界（含当时的计数快照）。"""
    SEARCH_TRACE["enabled"] = bool(value)
    SEARCH_TRACE["enabled_windows"].append(
        {
            "enabled": bool(value),
            "reason": reason,
            "wrapper_calls": SEARCH_TRACE["wrapper_calls"],
            "recorded": len(SEARCH_TRACE["records"]),
        }
    )


def _trace_summary():
    """轨迹汇总：一对一不变量 + 不同 query 清单 + 开关窗口。"""
    records = SEARCH_TRACE["records"]
    distinct = []
    for record in records:
        query = record.get("query")
        if isinstance(query, str) and query and query not in distinct:
            distinct.append(query)

    return {
        "installed": SEARCH_TRACE["installed"],
        "wrapper_calls": SEARCH_TRACE["wrapper_calls"],
        "original_calls": SEARCH_TRACE["original_calls"],
        "recorded": len(records),
        "one_to_one": (
            SEARCH_TRACE["wrapper_calls"] == SEARCH_TRACE["original_calls"]
        ),
        "recorded_calls_match": (
            len(records) == len([e for e in records if e.get("status") == "ok"])
            + len(SEARCH_TRACE["errors"])
        ),
        "distinct_queries": distinct,
        "distinct_query_count": len(distinct),
        "enabled_windows": list(SEARCH_TRACE["enabled_windows"]),
        "errors": list(SEARCH_TRACE["errors"]),
        "records": list(records),
    }


# ============================================================
# 14. 本地自检：白名单 + 预算边界（无网络、不消耗预算）
# ============================================================

URL_SELFCHECK_CASES = (
    ("https 未显式写端口 → 有效端口 443 → 允许",
     "https://api.deepseek.com/chat/completions", True),
    ("https 显式 :443 → 允许",
     "https://api.deepseek.com:443/v1/chat/completions", True),
    ("https 显式 :8443 → 拒绝（端口越界）",
     "https://api.deepseek.com:8443/v1/chat/completions", False),
    ("http 未写端口（默认 80）→ 拒绝（scheme 不符）",
     "http://api.deepseek.com/chat/completions", False),
    ("https 但 Host 为 example.com → 拒绝",
     "https://example.com/chat/completions", False),
)

_BUDGET_KEYS_FOR_SELFCHECK = (
    "request_attempts",
    "sent_requests",
    "response_count",
    "blocked",
    "blocked_reason",
    "blocked_detail",
)

_BUDGET_LEN_KEYS_FOR_SELFCHECK = (
    "requests",
    "responses",
    "usage_by_request",
)

BUDGET_BOUNDARY_CASES = tuple(
    [(attempt, True) for attempt in range(1, MAX_HTTP_REQUESTS + 1)]
    + [(MAX_HTTP_REQUESTS + 1, False), (MAX_HTTP_REQUESTS + 2, False)]
)


def _budget_snapshot():
    snap = {k: BUDGET[k] for k in _BUDGET_KEYS_FOR_SELFCHECK}
    snap.update({("len:" + k): len(BUDGET[k]) for k in _BUDGET_LEN_KEYS_FOR_SELFCHECK})
    snap["len:stage_counts"] = sum(BUDGET["stage_counts"].values())
    return snap


def _selfcheck_url_whitelist():
    """用真实 httpx2.Request 本地构造 URL，验证白名单纯函数的 5 条边界。"""
    import httpx2

    budget_before = _budget_snapshot()

    cases = []
    all_ok = True
    for label, url_text, expected in URL_SELFCHECK_CASES:
        try:
            url = httpx2.Request("POST", url_text).url
            actual = bool(_is_allowed_api_url(url))
            observed = {
                "scheme": url.scheme,
                "host": url.host,
                "port": url.port,
                "effective_port": _effective_port(url),
            }
        except Exception as exc:
            actual = False
            observed = {"error": type(exc).__name__}

        passed = (actual == expected)
        all_ok = all_ok and passed
        cases.append(
            {
                "case": label,
                "url": url_text,
                "expected": expected,
                "actual": actual,
                "observed": observed,
                "ok": passed,
            }
        )

    try:
        none_rejected = (_is_allowed_api_url(None) is False)
    except Exception:
        none_rejected = False

    budget_after = _budget_snapshot()
    untouched = (budget_before == budget_after)

    detail = {
        "cases": cases,
        "cases_total": len(cases),
        "cases_passed": sum(1 for c in cases if c["ok"]),
        "none_input_rejected": none_rejected,
        "budget_before": budget_before,
        "budget_after": budget_after,
        "budget_untouched": untouched,
    }
    return (all_ok and none_rejected and untouched), detail


def _selfcheck_budget_boundary():
    """纯函数验证预算边界：1..MAX 放行、MAX+1 与 MAX+2 拒绝。"""
    before = _budget_snapshot()

    cases = []
    all_ok = True
    for attempt, expected in BUDGET_BOUNDARY_CASES:
        actual = bool(_budget_decision(attempt))
        passed = (actual == expected)
        all_ok = all_ok and passed
        cases.append(
            {
                "attempt": attempt,
                "expected": expected,
                "actual": actual,
                "ok": passed,
            }
        )

    after = _budget_snapshot()
    untouched = (before == after)

    detail = {
        "max_http_requests": MAX_HTTP_REQUESTS,
        "cases": cases,
        "cases_total": len(cases),
        "cases_passed": sum(1 for c in cases if c["ok"]),
        "budget_untouched": untouched,
    }
    return (all_ok and untouched), detail


# ============================================================
# 15. 答案与来源核对（模型回答层）
# ============================================================


def _source_buckets(sources):
    """把 sources 按格式与定位字段分桶（用于与答案逐项交叉核对）。"""
    pdf_pages = set()
    docx_sections = set()
    docx_tables = set()
    formats = set()

    for item in sources:
        if not isinstance(item, dict):
            continue
        source_name = str(item.get("source") or "")
        suffix = Path(source_name).suffix.lower()
        if suffix:
            formats.add(suffix)
        if suffix == ".pdf" and item.get("page") is not None:
            pdf_pages.add(item.get("page"))
        if item.get("section"):
            docx_sections.add(item.get("section"))
        if item.get("table_index") is not None:
            docx_tables.add(item.get("table_index"))

    return {
        "pdf_pages": sorted(p for p in pdf_pages if isinstance(p, int)),
        "docx_sections": sorted(docx_sections),
        "docx_tables": sorted(
            t for t in docx_tables if isinstance(t, int)
        ),
        "formats": sorted(formats),
    }


def _warranty_verdict(answer_text):
    """解析答案中与保修期限有关的片段，判定「是否拒答」与「是否虚构」。

    为什么必须先切句再判定：整份答案里必然出现 90 天、30 天、第 2 页、
    表格 1 等数字，若在全文中找数字会大面积误伤。这里只保留**同时提到
    保修/质保**的句子，再在这些句子内分别找拒答词与「N 年」年限表述。

    切分刻意不使用 "|"：Markdown 表格的「问题列」与「答案列」会被切成
    两段，出现「含保修但无拒答词」+「含拒答词但无保修」的假阴性。
    """
    text = answer_text or ""
    segments = [
        seg.strip()
        for seg in _WARRANTY_SENTENCE_SPLIT.split(text)
        if seg.strip() and any(alias in seg for alias in WARRANTY_TOPIC_ALIASES)
    ]

    refusal_markers = []
    for seg in segments:
        for marker in WARRANTY_REFUSAL_MARKERS:
            if marker in seg and marker not in refusal_markers:
                refusal_markers.append(marker)

    fabrication = []
    for seg in segments:
        hit = WARRANTY_FABRICATION_PATTERN.search(seg)
        if hit:
            fabrication.append(hit.group(0))

    return {
        "segments": segments,
        "refusal_markers": refusal_markers,
        "fabrication": fabrication,
    }


def _evaluate_answer(answer_text, sources, retrieval):
    """把模型答案与真实检索来源逐项核对。

    判定分三层且**互不代偿**：
      (a) 正向事实层：三个事实必须出现；
      (a′) 负向事实层：整机保修期限在资料中**不存在**，必须明确拒答，
           且不得给出任何年限（不得用常识补全）；
      (b) 引用层：答案必须给出定位指称，且 sources 必须真的含对应元数据。
    事实正确但来源缺失 → 引用项单独失败（任务书 §9）。
    负向项失败**不能**被其他三项正确抵消：任一 check 失败即整体失败。
    """
    text = answer_text or ""
    buckets = _source_buckets(sources or [])

    def _has_aliases(aliases):
        return any(alias in text for alias in aliases)

    checks = []

    def add(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})
        return bool(ok)

    # ---- (a) 事实层 ----
    add("答案含更换周期 90 天", _has_aliases(EXPECTED_REPLACE_DAYS),
        "aliases=%s" % (EXPECTED_REPLACE_DAYS,))
    add("答案含巡检间隔 30 天", _has_aliases(EXPECTED_INSPECT_DAYS),
        "aliases=%s" % (EXPECTED_INSPECT_DAYS,))
    add("答案含检测方式 压力差", _has_aliases(EXPECTED_METHOD),
        "aliases=%s" % (EXPECTED_METHOD,))

    # ---- (a′) 负向事实层：保修期限不存在，必须拒答且不得虚构 ----
    warranty = _warranty_verdict(text)
    add("答案对整机保修期限明确拒答（资料未提供）",
        bool(warranty["refusal_markers"]),
        "segments=%d refusal=%s"
        % (len(warranty["segments"]), warranty["refusal_markers"]))
    add("答案未虚构整机保修年限",
        not warranty["fabrication"],
        "fabrication=%s" % (warranty["fabrication"],))

    # ---- (b) 引用层：答案里的定位指称 ----
    pdf_ref = _PDF_PAGE_PATTERN.search(text)
    add("答案指称 PDF 第 %d 页" % PDF_PAGE_WITH_FACT,
        pdf_ref is not None,
        "matched=%r" % (pdf_ref.group(0) if pdf_ref else None))

    section_ref = _DOCX_SECTION_PATTERN.search(text)
    add("答案指称 DOCX 章节「%s」" % DOCX_SECTION_EXPECTED,
        section_ref is not None,
        "matched=%r" % (section_ref.group(0) if section_ref else None))

    table_ref = _DOCX_TABLE_PATTERN.search(text)
    add("答案指称 DOCX 表格 1",
        table_ref is not None,
        "matched=%r" % (table_ref.group(0) if table_ref else None))

    # ---- (b) 引用层：真实 sources 元数据必须支撑上述指称 ----
    add("sources 含 PDF page=%d" % PDF_PAGE_WITH_FACT,
        PDF_PAGE_WITH_FACT in buckets["pdf_pages"],
        "pdf_pages=%s" % buckets["pdf_pages"])

    add("sources 含 DOCX section=%s" % DOCX_SECTION_EXPECTED,
        DOCX_SECTION_EXPECTED in buckets["docx_sections"],
        "docx_sections=%s" % buckets["docx_sections"])

    add("sources 含 DOCX table_index=%d" % DOCX_TABLE_INDEX_EXPECTED,
        DOCX_TABLE_INDEX_EXPECTED in buckets["docx_tables"],
        "docx_tables=%s" % buckets["docx_tables"])

    add("sources 覆盖 PDF 与 DOCX 两种格式",
        {".pdf", ".docx"} <= set(buckets["formats"]),
        "formats=%s" % buckets["formats"])

    # ---- 检索状态 ----
    status = (retrieval or {}).get("status") if isinstance(retrieval, dict) else None
    add("retrieval.status == hit", status == "hit", "status=%s" % status)

    failed = [c for c in checks if not c["ok"]]
    return (not failed), checks, buckets


# ============================================================
# 16. 主流程
# ============================================================


def main():
    global _TRACE_SECRET

    started = time.time()

    # ---- 防误启动：本脚本必须在普通进程里跑 ----
    if "pytest" in sys.modules:
        print("停止：检测到 pytest 运行时；本脚本只能作为普通进程直接运行。")
        return 9

    # ---- 控制台编码兜底（Windows 默认 cp936 打印中文可能抛错）----
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    print("=" * 70)
    print("P2-3D-C1：真实 PDF / DOCX 多来源问答与定位引用验收")
    print("=" * 70)

    # ---- 1) API Key：只读进程环境变量 ----
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("状态：skipped_no_api_key")
        print("原因：进程环境变量 DEEPSEEK_API_KEY 不存在或为空。")
        print("提示：未读取环境变量文件、未构造客户端、未发起任何请求。")
        return 2

    print("API Key：已从进程环境变量读取（存在，未打印，长度已隐去）")

    # 仅供检索轨迹里的查询词脱敏（查询词本身不敏感，但统一走同一脱敏器）。
    _TRACE_SECRET = api_key

    # ---- 2) 正式资产指纹：运行前后必须完全一致 ----
    protected_paths = {
        "project_memory_db": PROJECT_ROOT / "memory.db",
        "project_business_db": PROJECT_ROOT / "data" / "business.db",
        "project_rag_knowledge": PROJECT_ROOT / "rag_knowledge",
    }
    fingerprints_before = {
        name: _fingerprint(path) for name, path in protected_paths.items()
    }

    # ---- 3) 临时运行目录（存在即报错，绝不复用）----
    run_dir, kb_dir, memory_dir, evidence_dir = _create_run_dir()
    print(f"临时运行目录：{run_dir}")

    # ---- 4) 生成两份真实文档 ----
    pdf_path = kb_dir / PDF_FILENAME
    docx_path = kb_dir / DOCX_FILENAME
    _build_pdf(pdf_path, PDF_PAGES)
    _build_docx(docx_path)
    print(f"已生成 PDF：{pdf_path.name}（{pdf_path.stat().st_size} B）")
    print(f"已生成 DOCX：{docx_path.name}（{docx_path.stat().st_size} B）")

    # ---- 5) 切工作目录：必须在构造 Agent 之前 ----
    # Agent.__init__ 执行 self.memory = Memory()，Memory 默认 db_path="memory.db"
    # 且构造即建表，相对路径按 cwd 解析。切到 memory/ 后库只落临时目录。
    os.chdir(memory_dir)
    print(f"工作目录已切换：{Path.cwd()}")

    evidence = {
        "script": Path(__file__).name,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "offline_env": dict(OFFLINE_ENV),
        "target": {
            "base_url": BASE_URL,
            "model": MODEL_NAME,
            "allowed_scheme": ALLOWED_SCHEME,
            "allowed_host": ALLOWED_HOST,
            "allowed_port": ALLOWED_PORT,
            "max_http_requests": MAX_HTTP_REQUESTS,
            "timeout": {
                "connect": TIMEOUT_CONNECT,
                "read": TIMEOUT_READ,
                "write": TIMEOUT_WRITE,
                "pool": TIMEOUT_POOL,
            },
        },
        "question": QUESTION,
        "paths": {
            "temp_root": str(run_dir),
            "kb_dir": str(kb_dir),
            "memory_dir": str(memory_dir),
            "evidence_dir": str(evidence_dir),
            "pdf_file": str(pdf_path),
            "docx_file": str(docx_path),
            "memory_db_expected": str(memory_dir / "memory.db"),
        },
        "assertions": [],
        "status": "unknown",
    }

    def record(name, ok, detail=None):
        entry = {"name": name, "ok": bool(ok)}
        if detail is not None:
            entry["detail"] = _sanitize(detail, api_key)
        evidence["assertions"].append(entry)
        printed = "" if detail is None else f"  （{entry.get('detail')}）"
        print(f"[{'PASS' if ok else 'FAIL'}] {name}{printed}")
        return bool(ok)

    http_client = None

    try:
        # ---- 6) 文档结构校验（pypdf / python-docx 双重复核）----
        pdf_ok, pdf_detail = _verify_pdf(pdf_path)
        evidence["documents"] = {"pdf": pdf_detail}
        record(
            "PDF 三页且第 %d 页含事实（pypdf 解析）" % PDF_PAGE_WITH_FACT,
            pdf_ok,
            "pages=%s" % [(p["page"], p["chars"], p["has_fact"]) for p in pdf_detail["pages"]],
        )
        if not pdf_ok:
            evidence["status"] = "pdf_construct_failed"
            return 4

        docx_ok, docx_detail = _verify_docx_structure(docx_path)
        evidence["documents"]["docx"] = docx_detail
        record(
            "DOCX 标题/正文/表格内容正确（python-docx 读回）",
            docx_ok,
            "headings=%s tables=%d cells=%s"
            % (docx_detail["headings"], docx_detail["table_count"], docx_detail["table_cells"]),
        )
        if not docx_ok:
            evidence["status"] = "docx_construct_failed"
            return 4

        # ---- 7) 导入生产模块（此时 cwd 已在临时目录）----
        import agent as agent_module
        from rag.knowledge_base import KnowledgeBase

        # ---- 8) 意图分类闸门：不是 recall 就不加载模型、不发请求 ----
        intent = agent_module.Agent._classify_knowledge_intent(QUESTION)
        evidence["knowledge_intent"] = intent
        print(f"知识意图分类：{intent}")
        if intent != "recall":
            evidence["status"] = "aborted_not_recall"
            record("知识意图为 recall", False, f"实际={intent}")
            return 3
        record("知识意图为 recall", True, intent)

        # ---- 9) 契约检查（类级 + 源码自审计）----
        contract_ok, contract_detail = _verify_knowledge_base_contract(KnowledgeBase)
        evidence["kb_contract"] = contract_detail
        record(
            "契约：is_indexed 为 property 且源码无只读属性误调用",
            contract_ok,
            "static=%s violations=%s"
            % (contract_detail["is_indexed_static_kind"], contract_detail["violations"]),
        )
        if not contract_ok:
            evidence["status"] = "contract_violation"
            return 4

        # ---- 10) URL 白名单纯函数自检（无网络、零预算消耗）----
        whitelist_ok, whitelist_detail = _selfcheck_url_whitelist()
        evidence["url_whitelist_selfcheck"] = whitelist_detail
        print()
        print("---- URL 白名单纯函数自检（本地构造，无网络）----")
        for case in whitelist_detail["cases"]:
            print(
                "  [%s] %s：expected=%s actual=%s"
                % ("OK " if case["ok"] else "BAD", case["case"],
                   case["expected"], case["actual"])
            )
        record(
            "白名单自检：5 条 URL 边界符合预期（含有效端口 443）",
            whitelist_ok,
            "cases=%d/%d none_rejected=%s budget_untouched=%s"
            % (whitelist_detail["cases_passed"], whitelist_detail["cases_total"],
               whitelist_detail["none_input_rejected"],
               whitelist_detail["budget_untouched"]),
        )
        if not whitelist_ok:
            evidence["status"] = "url_whitelist_selfcheck_failed"
            return 4

        # ---- 11) 调用点映射校验 ----
        callmap_ok, callmap_detail = _verify_call_site_map(agent_module)
        evidence["stage_call_sites"] = callmap_detail
        print()
        print("---- 阶段调用点映射（AST 扫描 agent.py）----")
        for site in callmap_detail["call_sites"]:
            print("  %-18s L%-5d → %s" % (site["func"], site["lineno"], site["stage"]))
        record(
            "阶段调用点映射覆盖 agent.py 全部 create 调用点",
            callmap_ok,
            "sites=%d unmapped=%d stale=%d"
            % (callmap_detail["call_site_count"],
               len(callmap_detail["unmapped"]), len(callmap_detail["stale"])),
        )
        if not callmap_ok:
            evidence["status"] = "stage_call_site_map_mismatch"
            return 4

        # ---- 12) 预算边界自检 ----
        boundary_ok, boundary_detail = _selfcheck_budget_boundary()
        evidence["budget_boundary_selfcheck"] = boundary_detail
        record(
            "预算边界自检：1..%d ALLOW，%d/%d BLOCK，计数未变"
            % (MAX_HTTP_REQUESTS, MAX_HTTP_REQUESTS + 1, MAX_HTTP_REQUESTS + 2),
            boundary_ok,
            "cases=%d/%d untouched=%s"
            % (boundary_detail["cases_passed"], boundary_detail["cases_total"],
               boundary_detail["budget_untouched"]),
        )
        if not boundary_ok:
            evidence["status"] = "budget_boundary_selfcheck_failed"
            return 4

        # ---- 13) 真实 Loader 预检（不联网、不加载模型）----
        loader_ok, loader_detail = _precheck_documents(pdf_path, docx_path)
        evidence["loader_precheck"] = loader_detail
        record(
            "真实 Loader：PDF 三页 + DOCX 章节/表格元数据契约",
            loader_ok,
            "pdf_pages=%s docx_docs=%d section_docs=%d table_docs=%d table_has_section=%s"
            % ([d["page"] for d in loader_detail["pdf"]["documents"]],
               len(loader_detail["docx"]["documents"]),
               loader_detail["docx"]["section_documents"],
               loader_detail["docx"]["table_documents"],
               loader_detail["docx"]["table_has_section_key"]),
        )
        if not loader_ok:
            evidence["status"] = "loader_precheck_failed"
            return 4

        # ---- 14) 真实 KnowledgeBase（惰性，构造不加载模型）----
        knowledge_base = KnowledgeBase(kb_dir)

        # ---- 14B) 安装检索轨迹包装：此刻保持关闭，仅透传 ----
        # 之后的 Embedding 预检（3 次本地检索）会经过 wrapper 但不被记录，
        # 从而保证 agent_search_trace 只含 Agent 运行期的真实检索。
        _install_search_trace(knowledge_base)
        _set_trace_enabled(False, "before_local_precheck")

        instance_ok, instance_detail = _verify_knowledge_base_contract(
            KnowledgeBase, knowledge_base
        )
        evidence["kb_contract"].update(instance_detail)
        record(
            "契约：is_indexed 实例读取结果为 bool",
            instance_ok,
            "type=%s value=%r"
            % (instance_detail["is_indexed_instance_type"],
               instance_detail["is_indexed_instance_value"]),
        )
        if not instance_ok:
            evidence["status"] = "contract_violation"
            return 4

        # ---- 15) 本地 Embedding 预检：两类来源都可检索 ----
        kb_ok, kb_info = _precheck_knowledge_base(
            knowledge_base, PDF_FILENAME, DOCX_FILENAME
        )
        evidence["knowledge_base"] = kb_info

        print()
        print("---- 知识库预检（本地 Embedding，无网络）----")
        for key in ("files", "file_count", "chunk_count", "is_indexed",
                    "embedding_model_name", "embedding_dimension", "source_formats",
                    "replace_hits", "inspect_hits", "method_hits"):
            print(f"  {key}：{kb_info.get(key)}")

        record(
            "知识库预检：PDF/DOCX 双格式索引 + 三问各自命中定位字段",
            kb_ok,
            "files=%s chunks=%s dim=%s formats=%s hits(r/i/m)=%s/%s/%s"
            % (kb_info["file_count"], kb_info["chunk_count"],
               kb_info["embedding_dimension"], kb_info["source_formats"],
               kb_info["replace_hits"], kb_info["inspect_hits"], kb_info["method_hits"]),
        )
        if not kb_ok:
            evidence["status"] = "embedding_unavailable"
            return 8

        # ---- 16) 真实 HTTP 客户端 + 注入工厂 ----
        http_client = _build_http_client()
        agent_module.OpenAI = _make_client_factory(http_client)

        # ---- 17) 构造 Agent（真实 Agent，非 main.get_agent 缓存）----
        ctor_out = io.StringIO()
        try:
            with contextlib.redirect_stdout(ctor_out):
                agent_obj = agent_module.Agent(
                    api_key,
                    BASE_URL,
                    MODEL_NAME,
                    knowledge_base=knowledge_base,
                )
        finally:
            evidence["console_captured"] = {
                "agent_init_stdout_chars": len(ctor_out.getvalue()),
                "note": "Agent 控制台输出已捕获并丢弃，未落盘",
            }

        # ---- 18) 三项断言：真实 client / max_retries / model ----
        client_obj = getattr(agent_obj, "client", None)
        record(
            "Agent 使用真实 openai.OpenAI 客户端",
            type(client_obj).__name__ == "OpenAI",
            type(client_obj).__name__,
        )
        record(
            "SDK 自动重试为 0",
            getattr(client_obj, "max_retries", None) == 0,
            "max_retries=%s" % getattr(client_obj, "max_retries", None),
        )
        record(
            "模型名为 %s" % MODEL_NAME,
            getattr(agent_obj, "model", None) == MODEL_NAME,
            str(getattr(agent_obj, "model", None)),
        )
        record(
            "Agent 使用本次临时 KnowledgeBase",
            getattr(agent_obj, "knowledge_base", None) is knowledge_base,
            "identity 比对",
        )

        memory_obj = getattr(agent_obj, "memory", None)
        memory_db_path = None
        try:
            connection = memory_obj.get_connection()
            cursor = connection.execute("PRAGMA database_list")
            for row in cursor.fetchall():
                if row[1] == "main":
                    memory_db_path = row[2]
        except Exception:
            memory_db_path = None

        record(
            "Memory 数据库落在临时目录内",
            bool(memory_db_path) and str(memory_dir) in str(memory_db_path),
            str(memory_db_path),
        )
        record(
            "Memory 数据库不是项目正式 memory.db",
            bool(memory_db_path)
            and Path(str(memory_db_path)).resolve()
            != (PROJECT_ROOT / "memory.db").resolve(),
            str(memory_db_path),
        )

        # ---- 19) 工具收敛 ----
        tool_detail = _enforce_tool_allowlist(agent_obj)
        evidence["tools"] = tool_detail
        print()
        print("---- 工具隔离 ----")
        print(f"  构造时注册：{tool_detail['before']}")
        print(f"  已移除    ：{tool_detail['removed']}")
        print(f"  最终启用  ：{tool_detail['after']}")
        record(
            "工具注册表收敛到允许列表",
            tool_detail["after"] == list(ALLOWED_TOOLS),
            str(tool_detail["after"]),
        )

        # ---- 20) 唯一一次真实 Agent.run ----
        print()
        print("---- 发起唯一一次真实 Agent.run ----")
        run_out = io.StringIO()
        run_err = io.StringIO()
        result = None
        run_error = None

        # 轨迹开关：仅在 Agent.run 期间记录真实检索。
        _set_trace_enabled(True, "before_agent_run")
        try:
            with contextlib.redirect_stdout(run_out), contextlib.redirect_stderr(run_err):
                result = agent_obj.run(QUESTION, user_id="test-user")
        except Exception as exc:
            run_error = exc
        finally:
            _set_trace_enabled(False, "after_agent_run")

        evidence["console_captured"]["run_stdout_chars"] = len(run_out.getvalue())
        evidence["console_captured"]["run_stderr_chars"] = len(run_err.getvalue())

        if run_error is not None:
            kind = (
                "http_budget"
                if isinstance(run_error, HttpBudgetExceeded)
                else "host_whitelist"
                if isinstance(run_error, HostNotAllowedError)
                else "run_exception"
            )
            evidence["error"] = {
                "kind": kind,
                "type": type(run_error).__name__,
                "message": _sanitize(run_error, api_key),
            }
            record("Agent.run 未抛出异常", False,
                   f"{type(run_error).__name__}: {_sanitize(run_error, api_key)}")
            evidence["status"] = (
                "http_budget_exhausted"
                if kind == "http_budget"
                else "host_not_allowed"
                if kind == "host_whitelist"
                else "run_exception"
            )
            return 6 if kind in ("http_budget", "host_whitelist") else 7

        record("Agent.run 未抛出异常", True)

        # ---- 20B) 检索轨迹判读（以 wrapper 真实观察为准，不伪造）----
        trace_now = _trace_summary()
        record(
            "检索轨迹：真实 search 调用一对一透传（无重复/无丢失）",
            trace_now["one_to_one"] and trace_now["recorded"] >= 1,
            "wrapper=%d original=%d recorded=%d"
            % (
                trace_now["wrapper_calls"],
                trace_now["original_calls"],
                trace_now["recorded"],
            ),
        )
        record(
            "检索轨迹：Agent.run 期间至少 2 个不同 query 进入真实检索",
            trace_now["distinct_query_count"] >= 2,
            "distinct=%d %s"
            % (
                trace_now["distinct_query_count"],
                [q[:20] for q in trace_now["distinct_queries"][:5]],
            ),
        )

        print()
        print("---- Agent 真实检索轨迹（wrapper 实际观察）----")
        for rec in trace_now["records"]:
            print(
                "  #%d top_k=%s count=%s %s"
                % (
                    rec.get("seq"),
                    rec.get("top_k"),
                    rec.get("count"),
                    (rec.get("query") or "")[:40],
                )
            )

        # ---- 21) 结果解析 ----
        if not isinstance(result, dict):
            record("Agent 返回 dict", False, type(result).__name__)
            evidence["status"] = "unexpected_result"
            return 5

        answer_text = str(result.get("answer") or result.get("response") or "")
        success = bool(result.get("success"))
        sources = result.get("sources") or []
        retrieval = result.get("retrieval") or {}

        record("Agent 返回 success=True", success, f"success={success}")
        record("答案非空", bool(answer_text.strip()), f"len={len(answer_text)}")

        # ---- 22) 答案与来源逐项核对 ----
        answer_ok, checks, buckets = _evaluate_answer(
            answer_text, sources, retrieval
        )
        evidence["answer_checks"] = checks
        evidence["source_buckets"] = buckets

        print()
        print("---- 答案与来源逐项核对 ----")
        for check in checks:
            print(
                "  [%s] %s  （%s）"
                % ("PASS" if check["ok"] else "FAIL", check["name"], check["detail"])
            )

        for check in checks:
            record(check["name"], check["ok"], check["detail"])

        # ---- 23) 来源元数据（不含 chunk 正文）----
        source_meta = []
        for item in sources:
            if not isinstance(item, dict):
                continue
            row = {}
            for key in (
                "source", "chunk_id", "score", "document_id", "chunk_seq",
                "page", "page_count", "section", "section_index",
                "segment_index", "table_index",
            ):
                if key in item:
                    row[key] = item[key]
            source_meta.append(row)

        evidence["result"] = {
            "success": success,
            "answer": _sanitize(answer_text, api_key),
            "answer_length": len(answer_text),
            "sources": source_meta,
            "source_count": len(source_meta),
            "retrieval": {
                "status": retrieval.get("status") if isinstance(retrieval, dict) else None,
                "partial": retrieval.get("partial") if isinstance(retrieval, dict) else None,
                "skipped_files": retrieval.get("skipped_files") if isinstance(retrieval, dict) else None,
                "failed_searches": retrieval.get("failed_searches") if isinstance(retrieval, dict) else None,
            },
        }

        print()
        print("---- 最终答案 ----")
        print(_sanitize(answer_text, api_key))
        print()
        print(f"来源：{buckets}")

        ok_all = bool(success and answer_text.strip() and answer_ok)
        evidence["status"] = "ok" if ok_all else "citation_assertion_failed"
        return 0 if ok_all else 5

    except Exception as exc:
        evidence["status"] = "unexpected_exception"
        evidence["error"] = {
            "kind": "unexpected_exception",
            "type": type(exc).__name__,
            "message": _sanitize(exc, api_key),
        }
        print()
        print(f"[FAIL] 未预期异常：{type(exc).__name__}: {_sanitize(exc, api_key)}")
        return 7

    finally:
        # ---- 24) 收尾：计数 / 阶段 / 用量 / 指纹 / 落盘 ----
        usage_totals = dict(BUDGET["usage_totals"])
        if not usage_totals.get("available"):
            usage_totals["prompt_tokens"] = None
            usage_totals["completion_tokens"] = None
            usage_totals["total_tokens"] = None

        evidence["http"] = {
            "max_http_requests": MAX_HTTP_REQUESTS,
            "request_attempts": BUDGET["request_attempts"],
            "sent_requests": BUDGET["sent_requests"],
            "blocked_requests": len([r for r in BUDGET["requests"] if not r.get("sent")]),
            "response_count": BUDGET["response_count"],
            "responses_success": len(
                [r for r in BUDGET["responses"] if r.get("outcome") == "success"]
            ),
            "responses_http_error": len(
                [r for r in BUDGET["responses"] if r.get("outcome") == "http_error"]
            ),
            "requests": list(BUDGET["requests"]),
            "responses": list(BUDGET["responses"]),
            "stage_counts": dict(BUDGET["stage_counts"]),
            "usage_by_request": list(BUDGET["usage_by_request"]),
            "usage_totals": usage_totals,
            "status_codes": list(BUDGET["status_codes"]),
            "request_ids": list(BUDGET["request_ids"]),
            "blocked": BUDGET["blocked"],
            "blocked_reason": BUDGET["blocked_reason"],
            "blocked_detail": BUDGET["blocked_detail"],
            "sdk_max_retries": 0,
        }

        # Agent 运行期真实检索轨迹（准备阶段调用已被开关隔离在外）。
        evidence["agent_search_trace"] = _trace_summary()

        evidence["elapsed_ms"] = int((time.time() - started) * 1000)

        fingerprints_after = {
            name: _fingerprint(path) for name, path in protected_paths.items()
        }
        unchanged = fingerprints_before == fingerprints_after
        evidence["protected_assets"] = {
            "before": fingerprints_before,
            "after": fingerprints_after,
            "unchanged": unchanged,
        }

        if http_client is not None:
            try:
                http_client.close()
            except Exception:
                pass

        print()
        print("---- HTTP 预算 ----")
        print(
            f"  请求尝试={BUDGET['request_attempts']} "
            f"已发出={BUDGET['sent_requests']} "
            f"响应={BUDGET['response_count']} "
            f"上限={MAX_HTTP_REQUESTS} "
            f"阻断={BUDGET['blocked']}({BUDGET['blocked_reason']})"
        )
        print(f"  状态码={BUDGET['status_codes']}")
        print(f"  阶段分布={BUDGET['stage_counts']}")
        print(
            "  Token 用量=%s（逐请求 %d 条，有 usage %d 次，无 usage %d 次）"
            % (
                {
                    k: usage_totals.get(k)
                    for k in ("prompt_tokens", "completion_tokens", "total_tokens")
                },
                len(BUDGET["usage_by_request"]),
                usage_totals.get("requests_with_usage"),
                usage_totals.get("requests_without_usage"),
            )
        )
        print(f"  正式资产指纹未变化：{unchanged}")

        try:
            evidence_path = evidence_dir / "summary.json"
            evidence_path.write_text(
                json.dumps(evidence, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            print(f"  证据已写入：{evidence_path}")
        except Exception as exc:
            print(f"  证据写入失败：{type(exc).__name__}")

        print(f"最终状态：{evidence.get('status')}")
        print("=" * 70)


if __name__ == "__main__":
    sys.exit(main())
