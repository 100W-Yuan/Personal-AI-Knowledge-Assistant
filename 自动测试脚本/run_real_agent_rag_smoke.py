# -*- coding: utf-8 -*-
"""P2-3D-B1：真实 Agent + DeepSeek + RAG 单次问答验收脚本（独立运行，不经 pytest）。

定位
----
本文件**不是 pytest 用例**：文件名不以 ``test_`` 开头，只能作为普通 Python
进程直接执行。原因已由 P2-3D-A 逐层确认：本项目三层 conftest 守卫在 pytest
路径下会无条件拦截真实模型请求——

- ``conftest.py::_guarded_sdk_request`` 不看 URL，一律抛 BlockedSDKRequestError；
- ``conftest.py::_make_http_guard`` 的 ``_http_request_is_allowed`` 只放行回环；
- ``自动测试脚本/conftest.py`` 更在**导入期**把 ``openai.OpenAI`` 换成离线空壳。

因此真实调用的正统形态是独立脚本，与仓库既有先例一致：
``RAG测试/test_rag_llm.py``（RUN_REAL_LLM=1）与
``测试多模态/test_image_smoke_live.py``（ALLOW_REAL_API_CALL）。

安全边界（本脚本自我约束，不依赖任何外部守卫，也不拆除任何外部守卫）
--------------------------------------------------------------------
1. 只读进程环境变量 ``DEEPSEEK_API_KEY``；不读 ``.env``、不调用 ``load_dotenv``；
2. 构造 Agent **之前**把工作目录切到 TEMP 下本次唯一新目录，
   ``Agent.__init__`` 内的 ``Memory()`` 只能落在临时目录；
3. 构造后把工具注册表收敛到允许列表，直接删除字典条目
   （``ToolRegistry`` 无 unregister/remove API，``tools`` 是公开普通字典）；
4. 真实 HTTP 客户端带「5 次请求硬上限」+「单 Host / HTTPS / 有效端口 443」
   白名单，两者都在 request 事件钩子里、且在请求发出之前判定；
5. ``max_retries=0``，不自定义重试，不因异常重跑 ``Agent.run``；
6. Agent 的全部控制台输出被捕获后丢弃，只落结构化脱敏摘要；
7. 运行前执行 KnowledgeBase 契约检查（``is_indexed`` 必须是 property、
   实例读取必须得到 bool、源码不得对只读属性加括号调用），不通过即退出 4。
8. 运行前执行白名单判定本地自检（5 条 URL 边界，真实 ``httpx2.Request``
   构造、不发请求、不消耗请求预算），不通过即退出 4（P2-3D-B2-R3）。

白名单语义（P2-3D-B2-R3，修正 R2 的误拒）
----------------------------------------
``httpx2.URL.port`` 在 URL **未显式携带端口**时返回 ``None``
（``https://api.deepseek.com:443/...`` 也会被规范化为 ``None``），
而不是 443。因此判定改为「有效端口」：

- ``https`` 且 ``port is None``        → 有效端口 443 → 允许
- ``https`` 且显式 ``port=443``        → 有效端口 443 → 允许
- ``https`` 且显式 ``port=8443``       → 有效端口 8443 → 拒绝
- ``http``（任意端口）                 → 拒绝（scheme 不符）
- host 非 ``api.deepseek.com``         → 拒绝
- 无法解析 / url 为 None / 任何异常    → 拒绝（fail closed）

判定集中在纯函数 ``_is_allowed_api_url(url)``，``_on_request`` 只调用它。
不扩大 Host 范围、不扩大端口范围、不放宽 scheme。

请求预算（P2-3D-B2-R5 任务书 §3 R5-01）
----------------------------------------
``Agent.run()`` 的模型调用次数由 Plan 的实际形状决定，**不是固定值**。
以下只是预算情景说明，不构成对单次真实问答的固定次数承诺：

- 最少逻辑调用数为 **4**：``extract_memories``（1）+ ``create_plan``（1）
  + 工具参数生成（1）+ 无 pending 步骤时的最终 ``reasoning``（1）。
- 若 Plan 为「1 个知识检索工具步骤 + 1 个计划内 Reasoning 步骤」，
  则计划内 Reasoning 会额外产生 1 次调用，合计可能需要 **5** 次；
  该形状由 Planner 自行决定，脚本不保证也不强制。
- 已实测（P2-3D-B2-R4）5 次上限不足：5 次全部 200 后仍需第 6 次。
  本轮的 **8** 次上限即针对该实测结果调整。
- 本轮硬上限为 **8**（``MAX_HTTP_REQUESTS``）。第 1～8 次符合白名单的
  请求允许发送，**第 9 次在发送之前阻断**。
- **8 只是本轮硬上限，不保证任何复杂计划都能完成。** 更复杂的 Plan
  （多工具步骤、额外 reasoning 步骤、Retry 或 Replan）一旦超出预算，
  **必须安全失败**：以退出码 6 结束并记录
  ``blocked_reason="http_budget"``。届时**不得**放宽上限、不得自动重试、
  不得继续调用，也不得为省钱而改写问题。
- 预算计数是**整个进程生命周期内的累计值**：新建客户端、多次 SDK 调用、
  SDK 内部任何重试都不会重置它（``max_retries=0``，且计数在模块级字典）。
- 确定性预检索（``execute_tool("search_knowledge", top_k=3)``）属于
  **本地检索**，只做 Embedding + 向量检索，不产生任何模型 HTTP 请求，
  因此不计入上述次数。

阶段追踪与 Token 用量（P2-3D-B2-R5 任务书 §4 R5-02 / §5 R5-03）
---------------------------------------------------------------
- 每个模型请求都带阶段标签，取值域：
  ``extract_memories`` / ``create_plan`` / ``tool_argument_generation`` /
  ``reasoning`` / ``replan`` / ``unknown``。
- 阶段由**真实调用栈**判定（``sys._getframe(1)`` 的函数名 + 行号），
  与 ``agent.py`` 的 AST 扫描结果比对；未命中或存在歧义一律记
  ``unknown``，**绝不伪造**标签。
- 判定表来源：对 ``agent.py`` 做 AST 扫描得到的 5 个
  ``chat.completions.create`` 调用点（extract_memories:755、
  create_plan:1381、replan:1507、reasoning:1662、run:2733）。
  ``run:2733`` 的上下文是「请生成这个工具需要的 JSON 参数」，
  故其语义为 ``tool_argument_generation``（非按函数名推测）。
- 观察包装只包裹真实客户端的 ``chat.completions.create`` 绑定方法：
  原样传参、原样返回、原样抛异常，不吞异常、不重试、不改变调用顺序、
  不增加请求，也不把客户端替换成代理 / Mock / Stub。
- Token 用量取自真实 SDK 返回对象的 ``usage``，只记录三个整数；
  无 ``usage`` 时记 ``usage_available=false`` 且三项为 ``null``
  （**不填 0**）。不保存响应正文、请求对象、请求头、Authorization。
- 即使后续 ``Agent.run()`` 抛异常，已取得的请求 / 响应 / 用量证据
  仍会写入 ``summary.json``。

退出码
------
0 通过 | 2 缺少 DEEPSEEK_API_KEY | 3 意图不是 recall
4 静态契约 / 工具隔离 / 白名单自检无法证明
5 断言失败 | 6 HTTP 预算耗尽 | 7 运行异常 | 8 离线 Embedding 不可用
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
#
# 目的：只从本地 HuggingFace 缓存加载 BAAI/bge-small-zh-v1.5，
# 禁止联网下载、禁止遥测。此处一旦设置，后续任何分支都不得取消。
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

# 本文件位于 <项目根>/自动测试脚本/ 下；sys.path[0] 是该子目录，
# 因此必须把项目根加入 path 才能 import agent / memory / rag 包。
#
# SCRIPT_PATH 必须在**导入期**解析为绝对路径：脚本稍后会
# os.chdir() 到临时目录，之后再按相对 __file__ 读源码就会解析失败。
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---- 真实模型目标（与 main.py:273-278 的正式配置逐字一致）----
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-flash"

# ---- 单 Host 白名单：scheme / host / 有效端口 三者同时命中才放行 ----
# 注意 ALLOWED_PORT 是「有效端口」：URL 未写端口时按 scheme 取默认值后再比较
# （见 _effective_port）。R2 直接比较 url.port 导致 https 默认端口被误判为 None。
ALLOWED_SCHEME = "https"
ALLOWED_HOST = "api.deepseek.com"
ALLOWED_PORT = 443

# ---- HTTP 请求硬上限（P2-3D-B2-R5 任务书 §3 R5-01：最多 8 次）----
# 第 1～8 次符合白名单的请求允许发送；第 9 次在发送之前阻断。
# 计数 BUDGET 是进程级模块字典，跨客户端、跨 SDK 调用、跨异常持续累计，
# 新建客户端不会重置；异常后也不会自动重跑，因此「累计」语义成立。
#
# 明确声明：8 只是本轮硬上限，**不保证**任何复杂计划都能完成。
# 若第 9 次被阻断，必须安全失败（退出码 6），不得放宽上限、不得自动重试、
# 不得继续追加提问。修改此常量等同于改变授权额度，必须由新任务书授权。
MAX_HTTP_REQUESTS = 8

# ---- 显式超时（覆盖 openai 默认 read=600s 的宽限期）----
TIMEOUT_CONNECT = 5.0
TIMEOUT_READ = 45.0
TIMEOUT_WRITE = 20.0
TIMEOUT_POOL = 5.0

# ---- 外部工具允许列表（P2-3D-B1-R1 任务书 §2 R1-02：收窄为 1 个）----
# 只保留 search_knowledge。注意 ``search_memory`` 已被移出外部工具列表，
# 但 Agent 内部的 self.memory.search_memories() / retrieve_long_term_memories()
# 属于 Memory 检索路径而不是外部工具，仍然可用且仍指向本次临时数据库。
# 元组末尾的逗号不可省略：漏掉会退化成字符串。
ALLOWED_TOOLS = ("search_knowledge",)

# 明确列为禁止的工具（用于断言与报告，不是全部清单）
FORBIDDEN_TOOLS = (
    "query_orders",     # tools.py:261 ADMIN_ORDER_TOOL_ENABLED=True，读正式 business.db
    "calculator",       # 与本场景无关
    "get_current_time",  # 与本场景无关
    "test_failure",     # 故意失败桩，会污染失败/重规划路径
    "search_memory",    # R1-02 移出外部工具；内部 Memory 检索路径不受影响
)

# ---- 临时知识库 ----
KB_DIRNAME = "kb"
KB_FILENAME = "kb_qingshan_filter.txt"
KB_CONTENT = "产品青杉净水器的滤芯更换周期为90天。\n"

# ---- 唯一问答场景 ----
QUESTION = (
    "请根据我的知识库资料，回答青杉净水器的滤芯更换周期是多少天？"
    "请说明依据，并指出来源文件。"
)

# 事实核对：接受阿拉伯数字与中文数字两种写法（模型措辞不受控）
EXPECTED_FACT_ALIASES = ("90", "九十")

# 临时目录前缀：每次运行唯一，绝不复用
TEMP_PREFIX = "p2_3d_b1_agent_real_"


# ============================================================
# 3. 运行期可观测状态（不保存任何请求体 / 响应体 / 凭据）
# ============================================================

BUDGET = {
    # ---- 计数（全部为进程级累计值，跨客户端 / 跨调用不重置）----
    "request_attempts": 0,   # 进入 _on_request 的次数（含被拦截的）
    "sent_requests": 0,      # 白名单与预算都通过、真正发出的次数
    "response_count": 0,     # 真实收到响应的次数
    "blocked": 0,
    "blocked_reason": None,
    "blocked_detail": None,
    # ---- 逐条证据（R5-04）----
    "requests": [],          # 每次尝试一条；含被拦截的
    "responses": [],         # 每次真实响应一条；usage 由观察包装回填
    "status_codes": [],      # 兼容保留：响应状态码序列
    "request_ids": [],       # 兼容保留：安全请求 ID 序列
    # ---- 阶段与用量（R5-02 / R5-03）----
    "stage_counts": {},      # 阶段 → 尝试次数
    "usage_by_request": [],  # 与 responses 对齐的用量视图
    "usage_totals": {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "requests_with_usage": 0,
        "requests_without_usage": 0,
        "available": False,
    },
    # ---- 当前调用关联（同步单线程，一次 create 只发 1 个 HTTP）----
    "_last_attempt": None,
}

CLIENTS = []


# ============================================================
# 3-bis. 调用阶段上下文（R5-02）
#
# 阶段判定依据「真实调用栈」，而不是函数命名习惯。依据是对 agent.py 的
# AST 扫描结果：全文件只有 5 处 ``self.client.chat.completions.create``，
# 且每个所在函数各 1 处：
#     extract_memories:755  → extract_memories
#     create_plan:1381      → create_plan
#     replan:1507           → replan
#     reasoning:1662        → reasoning
#     run:2733              → tool_argument_generation
# （run:2733 的 Prompt 上下文为「请生成这个工具需要的 JSON 参数」，
#   故语义是工具参数生成，不是最终回答生成。）
#
# 命中规则：
#   1. (函数名, 行号) 精确命中 → 用映射的 stage；
#   2. 行号漂移但该函数在 agent.py 中确知只有 1 个 create 调用点
#      → 用该函数唯一的 stage（不会产生歧义，不是猜测）；
#   3. 其余情况（含包装层无法取得调用帧）→ "unknown"，绝不伪造。
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
CALL_SITE_STAGE = {
    ("extract_memories", 755): STAGE_EXTRACT,
    ("create_plan", 1381): STAGE_CREATE_PLAN,
    ("replan", 1507): STAGE_REPLAN,
    ("reasoning", 1662): STAGE_REASONING,
    ("run", 2733): STAGE_TOOL_ARGS,
}

# 函数名 → stage（仅在「该函数内确知只有 1 个 create 调用点」时才允许使用）
FUNCTION_STAGE = {
    "extract_memories": STAGE_EXTRACT,
    "create_plan": STAGE_CREATE_PLAN,
    "replan": STAGE_REPLAN,
    "reasoning": STAGE_REASONING,
    "run": STAGE_TOOL_ARGS,
}

# 当前阶段 / 当前 attempt：同一同步调用栈内传递，不跨线程、不改业务逻辑
_CURRENT_STAGE = contextvars.ContextVar("p2_3d_b2_r5_stage", default=STAGE_UNKNOWN)
_CURRENT_ATTEMPT = contextvars.ContextVar("p2_3d_b2_r5_attempt", default=None)

# 由 _verify_call_site_map() 在运行期填充：确知「函数内仅 1 个 create」的函数名集合
_UNIQUE_CALL_FUNCS = None


def _bump_stage(stage):
    """阶段尝试计数 +1（含被拦截的尝试，故与 request_attempts 对齐）。"""
    if stage not in ALLOWED_STAGES:
        stage = STAGE_UNKNOWN
    BUDGET["stage_counts"][stage] = BUDGET["stage_counts"].get(stage, 0) + 1


def _resolve_stage(frame):
    """由调用帧解析阶段；无法可靠判定时返回 "unknown"。"""
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

    # 行号漂移兜底：仅当该函数在 agent.py 中确知只有 1 个调用点时成立
    if _UNIQUE_CALL_FUNCS and func in _UNIQUE_CALL_FUNCS:
        return FUNCTION_STAGE.get(func, STAGE_UNKNOWN)

    return STAGE_UNKNOWN


class HttpBudgetExceeded(RuntimeError):
    """第 9 次（及以后）HTTP 请求在发送之前被终止。"""


class HostNotAllowedError(RuntimeError):
    """目标 scheme/host/port 不在白名单内，请求在发送之前被终止。"""


# ============================================================
# 4. HTTP 事件钩子：硬上限 + 单 Host 白名单
#
# 说明：httpx2 的 "request" 钩子在请求构造完成、真正发送之前触发，
# 因此在这里 raise 可以保证「请求从未离开本机」。
# 计数放在模块级字典里，不随 SDK 内部任何重试而重置 —— 重试也无法绕过。
# ============================================================


def _effective_port(url):
    """返回 URL 的有效端口：未显式写端口时按 scheme 取协议默认值。

    P2-3D-B2-R2 失败根因：``httpx2.URL.port`` 在 URL 未显式携带端口时返回
    ``None``（例如 ``https://api.deepseek.com/chat/completions``），
    而不是 443。若直接与 443 严格比较，合法目标会被白名单误拒。
    这里按 RFC 语义把「未写端口」折算为协议默认端口，再交给白名单判定。
    未知 / 缺失 scheme 一律返回 None —— 由调用方 fail closed。
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
    """纯函数：判断 URL 是否命中唯一允许的目标（scheme + host + 有效端口）。

    约束严格，不扩大范围：
      (1) scheme 必须恰好 "https"
      (2) host   必须恰好 "api.deepseek.com"
      (3) 有效端口必须恰好 443（显式 443 与「未写端口」等价，8443 一律拒绝）
    任何异常 / 无法解析 / url 为 None 的情况一律返回 False（fail closed）。
    本函数只读 URL 属性，不发请求、不改动 BUDGET，可独立自检。
    """
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
    """纯函数：累计第 attempt 次尝试是否允许发送（1..MAX 允许，MAX+1 起拒绝）。

    抽成纯函数是为了让本地自检能验证边界，而不真正发请求、也不改动计数。
    """
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

    # 逐条证据：只取 scheme / host / 有效端口，不读 path / query / header / body
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
    # 判定委托给纯函数 _is_allowed_api_url（有效端口语义，P2-3D-B2-R3 修复点）
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
    # 计数 BUDGET 是进程级模块字典：新建客户端、多次 SDK 调用、SDK 内部
    # 任何重试都不会重置它，因此「整个进程生命周期内累计」成立。
    # 抛出后不捕获重跑 —— 预算耗尽即安全失败（退出码 6）。
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

    # x-request-id 是服务端追踪号，不含任何凭据，可用于核对真实响应
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

    # 区分「成功响应」与「错误响应」：只依据状态码，不猜测
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
        # usage 由观察包装在 SDK 返回 ChatCompletion 后回填；此处默认不可用，
        # 绝不预先填 0（任务书 §5：无 usage 记 usage_available=false）
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

    - 无 usage / 非整数值 → usage_available=False 且三项为 None（不填 0）
    - 不保存响应正文、不保存请求对象、不读请求头
    - 回填到同一 attempt 的 response 记录，便于一一对应
    """
    usage = getattr(chat_completion, "usage", None)

    def _as_int(value):
        return int(value) if isinstance(value, int) and not isinstance(value, bool) else None

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

    # 回填到同一 attempt 的响应证据（从后往前找最近一条）
    for response_entry in reversed(BUDGET["responses"]):
        if response_entry.get("attempt") == attempt:
            response_entry["usage_available"] = record["usage_available"]
            response_entry["prompt_tokens"] = record["prompt_tokens"]
            response_entry["completion_tokens"] = record["completion_tokens"]
            response_entry["total_tokens"] = record["total_tokens"]
            break


def _install_stage_observer(client):
    """给真实客户端的 ``chat.completions.create`` 加观察包装（R5-02 / R5-03）。

    约束（任务书 §4）：
      - 客户端仍是真实 openai.OpenAI 实例，不做代理 / Mock / Stub 替换；
      - ``original`` 是真实绑定方法，包装函数原样传参、原样返回；
      - 不捕获 original 的异常 → 异常原样向上传播，不吞、不重试；
      - 不改变调用顺序、不增加任何请求。
    包装只做两件事：设置阶段上下文（供 HTTP 钩子读取）、读取返回对象的 usage。
    """
    completions = client.chat.completions
    original = completions.create

    if getattr(original, "_p2_stage_observed", False):
        return original

    def observed_create(*args, **kwargs):
        # 直接调用者 = agent.py 中真正发起模型请求的那个方法
        stage = _resolve_stage(sys._getframe(1))
        token = _CURRENT_STAGE.set(stage)
        try:
            result = original(*args, **kwargs)   # 原样调用；异常自然向上传播
        finally:
            _CURRENT_STAGE.reset(token)

        # 只有正常返回才会执行到这里（异常已在上方传播出去）
        try:
            _record_usage(_CURRENT_ATTEMPT.get(), stage, result)
        except Exception:
            # 采集失败不得影响主流程，也不得改变返回值
            pass
        return result

    observed_create._p2_stage_observed = True
    observed_create._p2_original = original
    observed_create.__wrapped__ = original

    completions.create = observed_create
    return observed_create


# ============================================================
# 4-bis. 白名单判定本地自检（无网络、零预算消耗）
#
# P2-3D-B2-R2 的失败出现在真实请求阶段（port=None 被误拒）。为了不再用
# 一次真实 API 调用换取一次事实，这里在本地用真实 httpx2.Request 构造 URL
# 对象（只构造、不发送），直接调用纯函数 _is_allowed_api_url 做边界验证。
#
# 约束：
#   - 不调用 _on_request，因此不增加 request_attempts / blocked 计数；
#   - 不创建客户端、不使用 Mock / Fake Client、不建立任何连接；
#   - 必须在任何真实 API 请求之前执行（main() 中位于构造客户端之前）。
# ============================================================

URL_SELFCHECK_CASES = (
    (
        "https 未显式写端口 → 有效端口 443 → 允许",
        "https://api.deepseek.com/chat/completions",
        True,
    ),
    (
        "https 显式 :443 → 允许",
        "https://api.deepseek.com:443/v1/chat/completions",
        True,
    ),
    (
        "https 显式 :8443 → 拒绝（端口越界）",
        "https://api.deepseek.com:8443/v1/chat/completions",
        False,
    ),
    (
        "http 未写端口（默认 80）→ 拒绝（scheme 不符）",
        "http://api.deepseek.com/chat/completions",
        False,
    ),
    (
        "https 但 Host 为 example.com → 拒绝",
        "https://example.com/chat/completions",
        False,
    ),
)

_BUDGET_KEYS_FOR_SELFCHECK = (
    "request_attempts",
    "sent_requests",
    "response_count",
    "blocked",
    "blocked_reason",
    "blocked_detail",
)

# 列表型证据只能比长度（同一对象引用，比对象本身无意义）
_BUDGET_LEN_KEYS_FOR_SELFCHECK = (
    "requests",
    "responses",
    "usage_by_request",
)


def _budget_snapshot():
    """自检用：计数 + 列表长度的快照，用于证明自检零副作用。"""
    snap = {k: BUDGET[k] for k in _BUDGET_KEYS_FOR_SELFCHECK}
    snap.update({("len:" + k): len(BUDGET[k]) for k in _BUDGET_LEN_KEYS_FOR_SELFCHECK})
    snap["len:stage_counts"] = sum(BUDGET["stage_counts"].values())
    return snap


# 预算边界用例（R5-05）：1～8 ALLOW，9～10 BLOCK
BUDGET_BOUNDARY_CASES = tuple(
    [(attempt, True) for attempt in range(1, MAX_HTTP_REQUESTS + 1)]
    + [(MAX_HTTP_REQUESTS + 1, False), (MAX_HTTP_REQUESTS + 2, False)]
)


def _selfcheck_budget_boundary():
    """纯函数验证预算边界：1..MAX 放行、MAX+1 与 MAX+2 拒绝。

    全程只调用 ``_budget_decision``，不进入 ``_on_request``，
    因此不消耗任何请求预算、不产生任何网络活动。
    """
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
        "budget_before": before,
        "budget_after": after,
        "budget_untouched": untouched,
    }
    return (all_ok and untouched), detail


def _selfcheck_url_whitelist():
    """用真实 httpx2.Request 本地构造 URL，验证白名单纯函数的 5 条边界。

    返回 (ok, detail)。ok 同时要求：5 条用例全部符合预期、「None 输入」被
    fail closed 拒绝、BUDGET 相关计数在自检前后完全一致。
    """
    import httpx2

    budget_before = _budget_snapshot()

    cases = []
    all_ok = True
    for label, url_text, expected in URL_SELFCHECK_CASES:
        try:
            # 只构造 Request 对象，取其 url（真实 httpx2.URL）；不会发送
            url = httpx2.Request("POST", url_text).url
            actual = bool(_is_allowed_api_url(url))
            observed = {
                "scheme": url.scheme,
                "host": url.host,
                "port": url.port,
                "effective_port": _effective_port(url),
            }
        except Exception as exc:  # 构造/解析失败也按拒绝处理（fail closed）
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

    # 额外 fail-closed 断言：非 URL 对象必须被拒绝，不能抛异常、不能放行
    try:
        none_rejected = (_is_allowed_api_url(None) is False)
    except Exception:
        none_rejected = False

    budget_after = _budget_snapshot()
    budget_untouched = (budget_before == budget_after)

    detail = {
        "cases": cases,
        "cases_total": len(cases),
        "cases_passed": sum(1 for c in cases if c["ok"]),
        "none_input_rejected": none_rejected,
        "budget_before": budget_before,
        "budget_after": budget_after,
        "budget_untouched": budget_untouched,
    }
    return (all_ok and none_rejected and budget_untouched), detail


# ============================================================
# 5. 真实 SDK 客户端工厂
#
# 注入点：agent.py:5 ``from openai import OpenAI`` —— 模块级符号，
# Agent.__init__（agent.py:539-542）直接调用它。把 agent.OpenAI 换成
# 本工厂即可在不改生产代码的前提下注入受限客户端。
#
# 关键坑（已有先例注释，见 测试多模态/test_image_smoke_live.py:245-249）：
# OpenAI 一旦收到 http_client，其自身 timeout 形参会被忽略
# （openai/_base_client.py: self._client = http_client or SyncHttpxClientWrapper(...)），
# 因此超时、禁止跳转、事件钩子必须全部设置在真实 HTTP 客户端上。
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
            max_retries=0,          # SDK 自动重试固定为 0
            http_client=http_client,  # 真实客户端，非替身
        )

        # 真实客户端保持原样；只是给它的 chat.completions.create 加观察包装
        # （不改类型、不改参数、不改返回、不吞异常，见 _install_stage_observer）
        _install_stage_observer(client)

        CLIENTS.append(client)
        return client

    return _factory


# ============================================================
# 6. 脱敏
# ============================================================

_SECRET_PATTERNS = (
    # 形如 sk-xxxx 的密钥
    (re.compile(r"sk-[A-Za-z0-9_\-]{6,}"), "***REDACTED***"),
    # Authorization 头（无论 Bearer 还是裸值）
    (re.compile(r"(?i)authorization\s*[:=]\s*[^\r\n]+"), "Authorization: ***REDACTED***"),
    # 裸 Bearer 令牌
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{6,}"), "Bearer ***REDACTED***"),
)


def _sanitize(value, secret=None):
    """对任意对象做文本化 + 密钥/授权头脱敏。"""
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    if secret:
        value = value.replace(secret, "***REDACTED***")
    for pattern, replacement in _SECRET_PATTERNS:
        value = pattern.sub(replacement, value)
    return value


# ============================================================
# 7. 文件指纹（只 stat，不读内容）
# ============================================================


def _fingerprint(path: Path):
    try:
        stat_result = path.stat()
    except OSError:
        return {"exists": False}
    return {
        "exists": True,
        "size": stat_result.st_size,
        "mtime_ns": stat_result.st_mtime_ns,
    }


# ============================================================
# 8. 临时运行目录
# ============================================================


def _create_run_dir():
    """在系统 TEMP 下创建本次运行唯一的新目录（三缺一即失败）。"""
    import tempfile

    temp_root = Path(tempfile.gettempdir()).resolve()
    run_dir = temp_root / f"{TEMP_PREFIX}{uuid.uuid4().hex}"

    # 明确要求：创建前确认目标路径不存在
    if run_dir.exists():
        raise RuntimeError(f"临时目录已存在，拒绝复用：{run_dir}")

    kb_dir = run_dir / KB_DIRNAME
    memory_dir = run_dir / "memory"
    evidence_dir = run_dir / "evidence"

    # exist_ok=False：父目录已存在就报错，保证「本次唯一」
    run_dir.mkdir(parents=False, exist_ok=False)
    kb_dir.mkdir(exist_ok=False)
    memory_dir.mkdir(exist_ok=False)
    evidence_dir.mkdir(exist_ok=False)

    return run_dir, kb_dir, memory_dir, evidence_dir


# ============================================================
# 9. 工具隔离
# ============================================================


def _enforce_tool_allowlist(agent_obj):
    """把注册表收敛到允许列表，并给出可验证的证据。

    ToolRegistry（tool_registry.py:1-10）只有 ``self.tools = {}`` 与
    ``register``，**没有** unregister / remove_tool 之类的 API。
    ``tools`` 是公开普通字典，因此可靠做法是删除字典键，
    再用三重断言证明隔离生效：
      1) 剩余键集合 == 允许集合；
      2) 禁止工具一个都不在；
      3) 交给模型的 tools 载荷（get_openai_tools）里也只有允许项。
    """
    registry = agent_obj.tool_registry

    tools_dict = getattr(registry, "tools", None)
    if not isinstance(tools_dict, dict):
        raise RuntimeError(
            "ToolRegistry.tools 不是可预期的字典，无法证明工具隔离可靠"
        )

    before = sorted(tools_dict)

    removed = [name for name in before if name not in ALLOWED_TOOLS]
    for name in removed:
        del tools_dict[name]

    after = sorted(tools_dict)

    # 断言 1：只剩允许项
    if set(after) != set(ALLOWED_TOOLS):
        raise RuntimeError(
            f"工具收敛失败：期望 {sorted(ALLOWED_TOOLS)}，实际 {after}"
        )

    # 断言 2：禁止工具一个都不在
    leftover = [name for name in FORBIDDEN_TOOLS if name in tools_dict]
    if leftover:
        raise RuntimeError(f"禁止工具仍然存在：{leftover}")

    # 断言 3：模型侧载荷只含允许项
    payload_names = {
        item["function"]["name"]
        for item in registry.get_openai_tools()
    }
    if payload_names != set(ALLOWED_TOOLS):
        raise RuntimeError(
            f"tools 载荷未收敛：{sorted(payload_names)}"
        )

    return before, removed, after


# ============================================================
# 10. KnowledgeBase 契约检查（P2-3D-B2-R1 防回归）
#
# 背景：P2-3D-B2 首跑在本地预检阶段崩溃，报
# ``TypeError: 'bool' object is not callable``。根因是
# ``KnowledgeBase.is_indexed`` 是 ``@property``（只读、返回 bool，
# 见 rag/knowledge_base.py:443-446），脚本却按方法调用。
#
# compile() 与常规 AST 结构检查无法发现「属性被当方法调用」，
# 因此这里增加两层运行前契约检查，全部在**任何模型请求之前**完成，
# 且不加载模型、不触发 Embedding、不联网：
#   1) 类型契约：``is_indexed`` 在类上必须是 property 对象；
#   2) 实例契约：构造后读取它必须直接得到 bool（而不是方法对象）；
#   3) 自审计：本脚本源码中不得出现 ``*.is_indexed(...)`` 这种 Call；
#   4) 自审计：KnowledgeBase 的**其他**只读属性也不得被当方法调用。
# ============================================================


def _knowledge_base_member_kinds(kb_class):
    """把 KnowledgeBase 的公开成员分类为 property / method / 其它。

    只读类属性（property / 数据描述符）一旦被加括号调用就会抛
    ``TypeError: '<type>' object is not callable``，是本次事故的形态。
    """
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
    """扫描本脚本源码：不得对 KnowledgeBase 的只读属性做 Call。

    这是第 3、4 项检查的合并实现——``is_indexed`` 与其余只读属性
    使用同一套判定，避免以后新增 property 时再次踩坑。
    返回违规列表 [(lineno, attr_name)]，空列表表示通过。
    """
    import ast as _ast

    kinds = _knowledge_base_member_kinds(kb_class)
    readonly = {name for name, kind in kinds.items() if kind == "property"}

    # 用导入期已解析的绝对路径：本函数在 os.chdir() 之后调用，
    # 直接用相对 __file__ 会因工作目录变化而读不到源码。
    source = SCRIPT_PATH.read_text(encoding="utf-8")
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
    """执行 §4 的契约检查（4 项；第 3、4 项合并为一次源码审计）。

    返回 ``(ok, detail)``；``detail`` 全部为可 JSON 序列化的基础类型。
    """
    detail = {}

    # (1) 类型契约：类上的 is_indexed 必须是 property
    import inspect as _inspect

    static_member = _inspect.getattr_static(kb_class, "is_indexed", None)
    detail["is_indexed_static_kind"] = type(static_member).__name__
    detail["is_indexed_is_property"] = isinstance(static_member, property)

    # (3)(4) 自审计：源码里不存在对只读属性的 Call
    violations, readonly_names = _audit_self_calls_on_readonly_members(kb_class)
    detail["readonly_members"] = readonly_names
    detail["violations"] = violations

    # (2) 实例契约：读取必须直接得到 bool
    if kb_instance is not None:
        value = kb_instance.is_indexed
        detail["instance_value"] = value
        detail["instance_type"] = type(value).__name__
        detail["instance_is_bool"] = isinstance(value, bool)
        instance_ok = detail["instance_is_bool"]
    else:
        instance_ok = True

    ok = (
        detail["is_indexed_is_property"]
        and not violations
        and instance_ok
    )
    return ok, detail


def _embedding_dimension(model):
    """读取模型输出维度；新旧方法名都尝试（sentence-transformers 6.x 已改名）。"""
    for method_name in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
        method = getattr(model, method_name, None)
        if not callable(method):
            continue
        try:
            return int(method())
        except Exception:
            continue
    return None


def _precheck_knowledge_base(kb, kb_dir):
    """在花费任何 API 额度之前，先用本地 Embedding 验证知识库可用。

    这一步是本地计算（离线模型），不产生任何 HTTP 请求。
    """
    info = {
        "kb_dir": str(kb_dir),
        "files": sorted(p.name for p in kb_dir.iterdir()),
    }

    probe = kb.search("青杉净水器滤芯更换周期", top_k=3)

    info["search_ok"] = bool(probe.get("ok"))
    info["search_stage"] = probe.get("stage")
    info["count"] = probe.get("count")
    # is_indexed 是 @property（rag/knowledge_base.py:443-446），只读属性
    # 不触发构建；必须按属性访问，不能加括号调用（P2-3D-B2-R1 修复点）。
    info["is_indexed"] = bool(kb.is_indexed)
    info["file_count"] = kb.file_count
    info["chunk_count"] = kb.chunk_count
    info["init_error_kind"] = kb.init_error_kind

    embedding = getattr(kb, "_embedding", None)
    model = getattr(embedding, "model", None)
    info["embedding_model_name"] = "BAAI/bge-small-zh-v1.5"
    info["embedding_dimension"] = _embedding_dimension(model) if model else None

    if not info["search_ok"]:
        info["error"] = _sanitize(probe.get("error"))
        return info, False

    # 索引健康：至少 1 个文件、1 个 chunk；命中的来源必须是临时文档
    hit_sources = {
        item.get("source") for item in probe.get("results", [])
    }
    info["hit_sources"] = sorted(s for s in hit_sources if s)
    info["contains_kb_file"] = KB_FILENAME in hit_sources

    ok = (
        info["count"] >= 1
        and info["chunk_count"] >= 1
        and info["contains_kb_file"]
    )

    # 维度核对：能读到就必须等于 512；读不到不据此判失败（记入证据，不猜）
    dimension = info["embedding_dimension"]
    if dimension is not None and dimension != 512:
        ok = False
        info["dimension_mismatch"] = True

    return info, ok


# ============================================================
# 10-bis. 调用点映射静态校验（R5-02）
#
# 目的：证明阶段标签来自真实调用点，而不是按函数名猜。若 agent.py 新增或
# 移动了 chat.completions.create 调用点，映射表就会失配 —— 此时必须停止，
# 不允许用「看起来合理」的标签继续跑。
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
                    if isinstance(inner, _ast.Attribute) and inner.attr == "completions":
                        sites.append((next_func, child.lineno))
            walk(child, next_func)

    walk(tree, None)
    return sites


def _verify_call_site_map(agent_module):
    """校验阶段映射表覆盖 agent.py 的全部 create 调用点。

    返回 (ok, detail)。ok=False 表示基线漂移，必须停止真实执行。
    同时填充 _UNIQUE_CALL_FUNCS（函数内仅 1 个 create 的程序集），
    供 _resolve_stage 在行号漂移时安全兜底。
    """
    global _UNIQUE_CALL_FUNCS

    path = Path(getattr(agent_module, "__file__", "agent.py"))
    source = path.read_bytes().decode("utf-8-sig")
    sites = _scan_create_call_sites(source)

    counts = {}
    for func, _lineno in sites:
        counts[func] = counts.get(func, 0) + 1

    unique_funcs = {func for func, n in counts.items() if n == 1}
    # 只有「函数内唯一 create」才允许用函数名兜底，避免歧义标注
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
# 11. 主流程
# ============================================================


def main():
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

    print("=" * 68)
    print("P2-3D-B1：真实 Agent + DeepSeek + RAG 单次问答验收脚本")
    print("=" * 68)

    # ---- 1) API Key：只读进程环境变量 ----
    api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        print("状态：skipped_no_api_key")
        print("原因：进程环境变量 DEEPSEEK_API_KEY 不存在或为空。")
        print("提示：未读取环境变量文件、未构造客户端、未发起任何请求。")
        return 2

    print("API Key：已从进程环境变量读取（存在，未打印，长度已隐去）")

    # ---- 2) 正式资产指纹：运行前后必须完全一致 ----
    protected_paths = {
        "project_memory_db": PROJECT_ROOT / "memory.db",
        "project_business_db": PROJECT_ROOT / "data" / "business.db",
        "project_rag_knowledge": PROJECT_ROOT / "rag_knowledge",
    }
    fingerprints_before = {
        name: _fingerprint(path) for name, path in protected_paths.items()
    }

    # ---- 3) 临时运行目录 ----
    run_dir, kb_dir, memory_dir, evidence_dir = _create_run_dir()
    print(f"临时运行目录：{run_dir}")

    # 写入临时知识库文档（内容唯一，便于来源归属）
    kb_file = kb_dir / KB_FILENAME
    kb_file.write_text(KB_CONTENT, encoding="utf-8")

    # ---- 4) 切工作目录：必须在构造 Agent 之前 ----
    # Agent.__init__（agent.py:548）执行 self.memory = Memory()，
    # Memory.__init__（memory.py:48-53）默认 db_path="memory.db" 且构造即建表，
    # 相对路径按 cwd 解析。切到 memory/ 子目录后，数据库只会落在本次临时目录。
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
        "paths": {
            "temp_root": str(run_dir),
            "kb_dir": str(kb_dir),
            "memory_dir": str(memory_dir),
            "evidence_dir": str(evidence_dir),
            "kb_file": str(kb_file),
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
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  （{entry.get('detail')}）" if detail else ""))
        return bool(ok)

    http_client = None
    stdout_chars = 0
    stderr_chars = 0

    try:
        # ---- 5) 导入生产模块（此时 cwd 已在临时目录）----
        import agent as agent_module
        from openai import OpenAI as RealOpenAIClient
        from rag.knowledge_base import KnowledgeBase

        # ---- 6) 意图分类闸门：不是 recall 就不加载模型、不发请求 ----
        # 分类器是纯正则实现（agent.py:1134-1204），不触网、不读文件、
        # 不加载模型，因此可以放在最前面做门禁。
        intent = agent_module.Agent._classify_knowledge_intent(QUESTION)
        evidence["knowledge_intent"] = intent
        print(f"知识意图分类：{intent}")
        if intent != "recall":
            evidence["status"] = "aborted_not_recall"
            record("知识意图为 recall", False, f"实际={intent}")
            return 3
        record("知识意图为 recall", True, intent)

        # ---- 7a) 契约检查（类级 + 源码自审计）：在任何模型请求之前 ----
        contract_ok, contract_detail = _verify_knowledge_base_contract(KnowledgeBase)
        evidence["kb_contract"] = contract_detail

        record(
            "契约：is_indexed 为 property 且源码无只读属性误调用",
            contract_ok,
            "static=%s readonly=%s violations=%s" % (
                contract_detail["is_indexed_static_kind"],
                contract_detail["readonly_members"],
                contract_detail["violations"],
            ),
        )
        if not contract_ok:
            evidence["status"] = "contract_violation"
            return 4

        # ---- 7a2) URL 白名单纯函数本地自检：无网络、零预算消耗 ----
        # 必须在任何真实 API 请求之前完成；只调用纯函数，不触发 _on_request。
        whitelist_ok, whitelist_detail = _selfcheck_url_whitelist()
        evidence["url_whitelist_selfcheck"] = whitelist_detail

        print()
        print("---- URL 白名单纯函数自检（本地构造，无网络）----")
        for case in whitelist_detail["cases"]:
            flag = "OK " if case["ok"] else "BAD"
            print(
                f"  [{flag}] {case['case']}：expected={case['expected']} "
                f"actual={case['actual']} observed={case['observed']}"
            )
        print(
            "  预算计数：before=%s after=%s untouched=%s"
            % (
                whitelist_detail["budget_before"],
                whitelist_detail["budget_after"],
                whitelist_detail["budget_untouched"],
            )
        )

        record(
            "白名单自检：5 条 URL 边界符合预期（含有效端口 443）",
            whitelist_ok,
            "cases=%d/%d none_rejected=%s budget_untouched=%s"
            % (
                whitelist_detail["cases_passed"],
                whitelist_detail["cases_total"],
                whitelist_detail["none_input_rejected"],
                whitelist_detail["budget_untouched"],
            ),
        )
        if not whitelist_ok:
            evidence["status"] = "url_whitelist_selfcheck_failed"
            return 4

        # ---- 7a3) 调用点映射校验：证明阶段标签有真实来源 ----
        callmap_ok, callmap_detail = _verify_call_site_map(agent_module)
        evidence["stage_call_sites"] = callmap_detail

        print()
        print("---- 阶段调用点映射（AST 扫描 agent.py）----")
        for site in callmap_detail["call_sites"]:
            print(
                "  %-18s L%-5d → %s"
                % (site["func"], site["lineno"], site["stage"])
            )
        record(
            "阶段调用点映射覆盖 agent.py 全部 create 调用点",
            callmap_ok,
            "sites=%d mapped=%d unmapped=%d stale=%d stages=%s"
            % (
                callmap_detail["call_site_count"],
                callmap_detail["mapped_count"],
                len(callmap_detail["unmapped"]),
                len(callmap_detail["stale"]),
                callmap_detail["stages_covered"],
            ),
        )
        if not callmap_ok:
            evidence["status"] = "stage_call_site_map_mismatch"
            return 4

        # ---- 7a4) 预算边界自检：1..8 ALLOW / 9..10 BLOCK，零副作用 ----
        boundary_ok, boundary_detail = _selfcheck_budget_boundary()
        evidence["budget_boundary_selfcheck"] = boundary_detail

        print()
        print("---- 请求预算边界自检（纯函数，无网络）----")
        for case in boundary_detail["cases"]:
            print(
                "  attempt=%-3d expected=%-5s actual=%-5s %s"
                % (
                    case["attempt"],
                    case["expected"],
                    case["actual"],
                    "OK" if case["ok"] else "BAD",
                )
            )
        record(
            "预算边界自检：1..%d ALLOW，%d/%d BLOCK，计数未变"
            % (MAX_HTTP_REQUESTS, MAX_HTTP_REQUESTS + 1, MAX_HTTP_REQUESTS + 2),
            boundary_ok,
            "cases=%d/%d budget_untouched=%s"
            % (
                boundary_detail["cases_passed"],
                boundary_detail["cases_total"],
                boundary_detail["budget_untouched"],
            ),
        )
        if not boundary_ok:
            evidence["status"] = "budget_boundary_selfcheck_failed"
            return 4

        # ---- 7b) 真实 KnowledgeBase（惰性，构造不加载模型）----
        knowledge_base = KnowledgeBase(kb_dir)

        # ---- 7c) 实例契约：读取必须直接得到 bool（不能是方法对象）----
        is_indexed_value = knowledge_base.is_indexed
        instance_is_bool = isinstance(is_indexed_value, bool)
        evidence["kb_contract"]["instance_value"] = is_indexed_value
        evidence["kb_contract"]["instance_type"] = type(is_indexed_value).__name__
        evidence["kb_contract"]["instance_is_bool"] = instance_is_bool

        record(
            "契约：is_indexed 实例读取结果为 bool",
            instance_is_bool,
            "type=%s value=%r" % (type(is_indexed_value).__name__, is_indexed_value),
        )
        if not instance_is_bool:
            evidence["status"] = "contract_violation"
            return 4

        kb_info, kb_ok = _precheck_knowledge_base(knowledge_base, kb_dir)
        evidence["knowledge_base"] = kb_info

        print()
        print("---- 知识库预检（本地 Embedding，无网络）----")
        for key in (
            "files", "count", "is_indexed", "file_count", "chunk_count",
            "embedding_model_name", "embedding_dimension", "hit_sources",
        ):
            print(f"  {key}：{kb_info.get(key)}")

        if not kb_ok:
            evidence["status"] = "embedding_unavailable"
            record("知识库预检通过", False, kb_info.get("error") or "预检条件不满足")
            return 8

        record("知识库预检通过（真实 Embedding 512 维，命中临时文档）", True)

        # ---- 8) 构造真实 HTTP 客户端 + 注入工厂 ----
        http_client = _build_http_client()
        agent_module.OpenAI = _make_client_factory(http_client)

        # ---- 9) 构造 Agent（真实 Agent，非 main.get_agent 缓存）----
        # Agent.__init__（agent.py:672）会 print 全部工具定义，同样属于未经
        # 筛选的控制台内容，这里一并捕获后丢弃。
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
            evidence.setdefault("console_captured", {})
            evidence["console_captured"]["agent_init_stdout_chars"] = len(ctor_out.getvalue())
            ctor_out.close()

        # ---- 10) 证明三件事：客户端真实 / 重试为 0 / Memory 在临时目录 ----
        record(
            "Agent 使用真实 openai.OpenAI 客户端",
            type(agent_obj.client) is RealOpenAIClient,
            type(agent_obj.client).__name__,
        )
        record(
            "SDK 自动重试为 0",
            getattr(agent_obj.client, "max_retries", None) == 0,
            f"max_retries={getattr(agent_obj.client, 'max_retries', None)}",
        )
        record(
            "模型名为 deepseek-flash",
            agent_obj.model == MODEL_NAME,
            agent_obj.model,
        )
        record(
            "Agent 使用本次临时 KnowledgeBase",
            getattr(agent_obj, "knowledge_base", None) is knowledge_base,
            "identity 比对",
        )

        memory_db_path = Path(agent_obj.memory.db_path)
        if not memory_db_path.is_absolute():
            memory_db_path = (Path.cwd() / memory_db_path)
        memory_db_resolved = memory_db_path.resolve()

        record(
            "Memory 数据库落在临时目录内",
            memory_db_resolved.is_relative_to(run_dir.resolve()),
            str(memory_db_resolved),
        )
        record(
            "Memory 数据库不是项目正式 memory.db",
            memory_db_resolved != (PROJECT_ROOT / "memory.db").resolve(),
            str(memory_db_resolved),
        )
        evidence["paths"]["memory_db_actual"] = str(memory_db_resolved)

        # ---- 11) 工具允许列表 ----
        tools_before, tools_removed, tools_after = _enforce_tool_allowlist(agent_obj)
        evidence["tools"] = {
            "allowed": list(ALLOWED_TOOLS),
            "before": tools_before,
            "removed": tools_removed,
            "after": tools_after,
        }

        print()
        print("---- 工具隔离 ----")
        print(f"  构造时注册：{tools_before}")
        print(f"  已移除    ：{tools_removed}")
        print(f"  最终启用  ：{tools_after}")

        record(
            "工具注册表收敛到允许列表",
            set(tools_after) == set(ALLOWED_TOOLS),
            str(tools_after),
        )
        # R1-02 强断言：最终唯一外部工具必须恰好是 search_knowledge。
        # 用列表相等（而非集合）保证「有且仅有一个、且没有多出来的」。
        record(
            "最终唯一外部工具为 search_knowledge",
            tools_after == ["search_knowledge"],
            f"after={tools_after}",
        )
        record(
            "query_orders / search_memory 等工具已移除",
            not any(name in tools_after for name in FORBIDDEN_TOOLS),
            f"remaining={tools_after}",
        )

        # ---- 12) 唯一一次真实问答（控制台输出全部捕获后丢弃）----
        print()
        print("---- 发起唯一一次真实 Agent.run ----")

        captured_out = io.StringIO()
        captured_err = io.StringIO()

        run_error = None
        result = None

        try:
            with contextlib.redirect_stdout(captured_out), \
                    contextlib.redirect_stderr(captured_err):
                result = agent_obj.run(QUESTION, user_id="test-user")
        except HttpBudgetExceeded as exc:
            run_error = ("http_budget", exc)
        except HostNotAllowedError as exc:
            run_error = ("host_not_allowed", exc)
        except Exception as exc:  # 不重跑、不重试，只记录脱敏后的类型与摘要
            run_error = ("exception", exc)
        finally:
            stdout_chars = len(captured_out.getvalue())
            stderr_chars = len(captured_err.getvalue())
            # 缓冲区立即丢弃：绝不写入 evidence、日志或控制台
            captured_out.close()
            captured_err.close()

        console_captured = evidence.setdefault("console_captured", {})
        console_captured["run_stdout_chars"] = stdout_chars
        console_captured["run_stderr_chars"] = stderr_chars
        console_captured["note"] = "Agent 控制台输出已捕获并丢弃，未落盘"

        if run_error is not None:
            kind, exc = run_error
            evidence["error"] = {
                "kind": kind,
                "type": type(exc).__name__,
                "message": _sanitize(exc, api_key),
            }
            record("Agent.run 未抛出异常", False, f"{type(exc).__name__}: {exc}")

            if kind == "http_budget":
                evidence["status"] = "http_budget_exhausted"
                return 6
            if kind == "host_not_allowed":
                evidence["status"] = "host_not_allowed"
                return 6
            evidence["status"] = "run_failed"
            return 7

        record("Agent.run 未抛出异常", True)

        # ---- 13) 结果断言 ----
        if not isinstance(result, dict):
            evidence["status"] = "unexpected_result_type"
            record("Agent 返回字典结构", False, type(result).__name__)
            return 5

        answer = result.get("answer")
        sources = result.get("sources") or []
        retrieval = result.get("retrieval")

        ok_all = True

        ok_all &= record("Agent 返回 success=True", bool(result.get("success")))

        answer_text = answer if isinstance(answer, str) else ""
        has_fact = any(alias in answer_text for alias in EXPECTED_FACT_ALIASES)
        ok_all &= record("答案包含正确事实（90 天）", has_fact)

        source_names = [
            item.get("source") for item in sources if isinstance(item, dict)
        ]
        ok_all &= record(
            "sources 包含临时知识库文档",
            KB_FILENAME in source_names,
            f"sources={source_names}",
        )

        hit_status = None
        if isinstance(retrieval, dict):
            hit_status = retrieval.get("status")
        ok_all &= record(
            "检索状态为 hit",
            hit_status == "hit",
            f"retrieval.status={hit_status}",
        )

        ok_all &= record(
            "知识库确为本次临时索引（真实工具链证据）",
            bool(knowledge_base.is_indexed)
            and knowledge_base.chunk_count >= 1
            and knowledge_base.file_count >= 1,
            f"file_count={knowledge_base.file_count};chunk_count={knowledge_base.chunk_count}",
        )

        ok_all &= record(
            "存在可解释来源（非无来源答案）",
            len([s for s in source_names if s]) >= 1,
        )

        ok_all &= record(
            "HTTP 请求次数未超上限",
            BUDGET["sent_requests"] <= MAX_HTTP_REQUESTS,
            f"sent={BUDGET['sent_requests']};attempts={BUDGET['request_attempts']}",
        )

        # ---- 14) 来源元数据（不含 chunk 正文）----
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
            "success": bool(result.get("success")),
            "answer": _sanitize(answer_text, api_key),  # 允许保存最终答案；已脱敏
            "answer_length": len(answer_text),
            "answer_contains_expected_fact": has_fact,
            "sources": source_meta,
            "source_count": len(source_meta),
            "retrieval": _sanitize(json.dumps(retrieval, ensure_ascii=False, default=str), api_key)
            if isinstance(retrieval, dict) else None,
        }

        print()
        print("---- 最终答案 ----")
        print(_sanitize(answer_text, api_key))
        print()
        print(f"来源：{source_names}")
        print(f"检索状态：{hit_status}")

        evidence["status"] = "ok" if ok_all else "assertion_failed"
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
        # ---- 15) 收尾：网络计数、阶段、用量、指纹比对、证据落盘 ----
        # 无论正常结束还是异常退出，都在此落盘：已发出的请求、已收到的响应、
        # 已取得的 usage 全部保留（R5-07）。未完成的请求不会被标成成功。
        usage_totals = dict(BUDGET["usage_totals"])
        if not usage_totals.get("available"):
            # 一次 usage 都没拿到：不把 0 当作真实用量
            usage_totals["prompt_tokens"] = None
            usage_totals["completion_tokens"] = None
            usage_totals["total_tokens"] = None

        requests_sent = [r for r in BUDGET["requests"] if r.get("sent")]
        requests_blocked = [r for r in BUDGET["requests"] if not r.get("sent")]
        responses_success = [
            r for r in BUDGET["responses"] if r.get("outcome") == "success"
        ]
        responses_error = [
            r for r in BUDGET["responses"] if r.get("outcome") == "http_error"
        ]

        evidence["http"] = {
            "max_http_requests": MAX_HTTP_REQUESTS,
            "request_attempts": BUDGET["request_attempts"],
            "sent_requests": BUDGET["sent_requests"],
            "blocked_requests": len(requests_blocked),
            "response_count": BUDGET["response_count"],
            "responses_success": len(responses_success),
            "responses_http_error": len(responses_error),
            # 逐条证据（R5-04 / R5-07）
            "requests": list(BUDGET["requests"]),
            "responses": list(BUDGET["responses"]),
            "stage_counts": dict(BUDGET["stage_counts"]),
            "usage_by_request": list(BUDGET["usage_by_request"]),
            "usage_totals": usage_totals,
            # 兼容保留
            "status_codes": list(BUDGET["status_codes"]),
            "request_ids": list(BUDGET["request_ids"]),
            "blocked": BUDGET["blocked"],
            "blocked_reason": BUDGET["blocked_reason"],
            "blocked_detail": BUDGET["blocked_detail"],
            "sdk_max_retries": 0,
        }
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
        print("=" * 68)


if __name__ == "__main__":
    sys.exit(main())
