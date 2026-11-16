from datetime import datetime


def calculator(expression: str) -> str:
    expression = expression.replace("^", "**")
    result = eval(expression, {"__builtins__": {}}, {})
    return str(result)


def get_current_time() -> str:
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


def test_failure():
    raise RuntimeError("这是一次故意制造的工具错误")


def make_search_memory_tool(memory, identity):
    """
    创建一个 search_memory 工具。

    ``memory``：Memory 实例（全进程共享）。
    ``identity``：无参可调用对象，返回**本次请求**的用户标识。

    PAI-AUTH-MVP：Agent 是全局单例，工具闭包**不能**在初始化时绑定某个
    用户；每次调用时通过 ``identity()`` 取当前请求的身份，因此并发请求
    之间不会串线。
    """

    def search_memory(query: str, category=None) -> dict:
        memories = memory.search_long_term_memory(
            query,
            limit=5,
            category=category,
            user_id=identity()
        )

        return {
            "query": query,
            "count": len(memories),
            "memories": memories
        }

    return search_memory


# ============================================================
# 知识检索状态（N1）
#
# 四种状态用于区分「查到了」「没查到」「库是空的」「链路坏了」，
# 避免把异常静默成“没找到”。判定集中在 classify_knowledge_retrieval，
# 工具层（成功路径）与 Agent（失败路径与汇总）共用同一份规则，
# 防止两处各自推断而漂移。
#
# 状态取值（对外契约，不得混用别名）：
#   hit       命中了知识库内容
#   no_hit    知识库可用，但该问题没有命中
#   kb_empty  知识库存在，但没有任何可用内容
#   degraded  检索链路异常（目录缺失、索引失败、查询期异常）
# ============================================================

KNOWLEDGE_STATUS_HIT = "hit"
KNOWLEDGE_STATUS_NO_HIT = "no_hit"
KNOWLEDGE_STATUS_KB_EMPTY = "kb_empty"
KNOWLEDGE_STATUS_DEGRADED = "degraded"

# 工具「真的执行了检索但失败」时的固定消息前缀。
# Agent 据此把这类失败与「参数校验失败 / 参数生成非法」区分开：
# 后者检索根本没有执行，不应被判定成检索链路异常。
KNOWLEDGE_FAILURE_PREFIX = "知识检索失败"


def _read_kb_int(knowledge_base, name):
    """安全读取 KnowledgeBase 的整数诊断属性。

    测试中的 stub 可能没有这些属性，一律降级为 0，不让诊断信息
    影响正常返回。
    """
    value = getattr(knowledge_base, name, 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def classify_knowledge_retrieval(knowledge_base, count=0, failed_searches=0):
    """判定本轮知识检索的状态，返回结构化状态字典。

    参数：
    - count：本轮命中的最大条数（0 表示本轮没有任何命中）。
      只接受「非布尔类型的非负整数」；True/False、负整数、浮点数、
      字符串、None 及其它非整数对象一律视为无效计数，返回
      degraded + partial=True，既不当作零命中，也不抛异常。
    - failed_searches：本轮「确实执行了检索但失败」的次数。
      参数校验失败、参数生成非法、意图守卫拒绝都不计入，
      因为它们根本没有执行检索。

    判定规则：
    - count 无效 → degraded（检索计数不可信，无法判断检索结果）；
    - count > 0 → hit；
    - count == 0 且 failed_searches > 0 → 按初始化错误类别判定：
        dir_no_files              → kb_empty（库存在但没有可索引内容）
        其余（目录缺失/全部不可解析/初始化或查询异常）→ degraded；
    - count == 0 且无失败 → no_hit（知识库可用，只是没命中）。

    partial 表示「本次判定依据不完整」，而不是「有部分结果」：
    存在被跳过的文档、或本轮存在检索失败，都会置为 True。
    因此 kb_empty / degraded 也可能 partial=True；
    no_hit + partial=True 的含义是「不能据此断言知识库中没有相关内容」。
    """
    skipped = _read_kb_int(knowledge_base, "skipped_count")
    init_error_kind = getattr(knowledge_base, "init_error_kind", None)

    if (
        isinstance(failed_searches, bool)
        or not isinstance(failed_searches, int)
        or failed_searches < 0
    ):
        failed_searches = 0

    # count 只接受「非布尔类型的非负整数」。
    # 这里必须先归一化再比较，否则会出现三类错判：
    #   True  → bool 是 int 子类，True > 0 成立，被误判成 hit；
    #   2.7   → 浮点同样能与 0 比较，被误判成 hit；
    #   -1    → 既不 > 0 也不触发失败分支，被静默吞成 no_hit；
    #   而 "3" / None 直接比较会抛 TypeError，冲出 run()。
    # 统一归一为 None，交给下面的「计数无效」分支，不伪装成零命中。
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        count = None

    notes = []
    if skipped > 0:
        notes.append(f"{skipped} 个文档在索引时被跳过")
    if failed_searches > 0:
        notes.append(f"{failed_searches} 次检索失败")

    partial = bool(notes)
    detail = f"（依据不完整：{'；'.join(notes)}）" if partial else ""

    # 计数无效优先级最高：既不能判 hit（可能其实没命中），
    # 也不能判 no_hit（可能其实命中了），只能如实报 degraded。
    # partial 恒为 True —— 判定依据已经不可用。
    if count is None:
        return {
            "status": KNOWLEDGE_STATUS_DEGRADED,
            "partial": True,
            "skipped_files": skipped,
            "failed_searches": failed_searches,
            "reason": "检索计数无效，无法可靠判断检索结果" + detail,
        }

    if count > 0:
        status = KNOWLEDGE_STATUS_HIT
        reason = "命中知识库内容" + detail

    elif failed_searches > 0:
        if init_error_kind == "dir_no_files":
            status = KNOWLEDGE_STATUS_KB_EMPTY
            reason = "知识库目录存在，但没有可索引的 .md/.txt 文件"
        elif init_error_kind == "dir_missing":
            status = KNOWLEDGE_STATUS_DEGRADED
            reason = "知识库目录不存在或不是目录"
        elif init_error_kind == "all_unparsable":
            status = KNOWLEDGE_STATUS_DEGRADED
            reason = "知识库文档全部无法加载或解析"
        elif init_error_kind == "init_error":
            status = KNOWLEDGE_STATUS_DEGRADED
            reason = "知识库初始化失败"
        else:
            status = KNOWLEDGE_STATUS_DEGRADED
            reason = "知识检索执行失败"
        reason += detail

    elif skipped > 0:
        status = KNOWLEDGE_STATUS_NO_HIT
        reason = (
            f"未命中任何内容，但有 {skipped} 个文档在索引时被跳过，"
            "不能据此断言知识库中没有相关内容"
        )

    else:
        status = KNOWLEDGE_STATUS_NO_HIT
        reason = "知识库可用，但该问题未命中任何内容"

    return {
        "status": status,
        "partial": partial,
        "skipped_files": skipped,
        "failed_searches": failed_searches,
        "reason": reason,
    }


def make_search_knowledge_tool(knowledge_base):
    """
    创建一个 search_knowledge 工具。

    通过闭包保存 KnowledgeBase 实例，
    让 Agent 可以通过 ToolRegistry 检索项目知识资料
    （注意：用于项目知识，不用于查询用户的个人记忆）。

    返回结构：
    - 命中：{"query", "count", "results":[...], "context", "retrieval"}
      results 每项固定含 source / chunk_id / score / chunk，并按来源格式附带
      document_id、chunk_seq 以及 page、page_count、section、section_index、
      segment_index、table_index 等定位元数据（KnowledgeBase 有则透传，
      该格式没有的字段不出现）。
      retrieval 为检索状态：
      {"status", "partial", "skipped_files", "failed_searches", "reason"}
    - 未命中：count=0，context 为明确提示（正常结果，不是失败）
    - 执行失败：抛异常，交由 ToolRegistry 既有错误处理约定；
      失败原因分类由 Agent 依据 KnowledgeBase 的 init_error_kind 判定
      （ToolRegistry 只保留 str(exc)，异常上的附加属性无法传出）
    """

    def search_knowledge(query: str, top_k: int = 3) -> dict:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("知识检索问题不能为空")
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise ValueError("top_k 必须是整数")
        if not 1 <= top_k <= 10:
            raise ValueError("top_k 必须在 1 到 10 之间")

        result = knowledge_base.search(query, top_k=top_k)

        if not result["ok"]:
            raise RuntimeError(
                f"{KNOWLEDGE_FAILURE_PREFIX}"
                f"（{result['stage']}）：{result['error']}"
            )

        count = result["count"]

        context = result["context"]
        if count == 0:
            context = "（知识库中未找到相关内容）"

        return {
            "query": result["query"],
            "count": count,
            "results": result["results"],
            "context": context,
            "retrieval": classify_knowledge_retrieval(
                knowledge_base,
                count=count,
            ),
        }

    return search_knowledge


# ========================================================
# 管理员模式订单查询（R3-AGENT-ORDER-02B）
#
# 边界声明：
# 1. 这是本机单人环境下的「管理员全局订单只读」能力，
#    不是多用户管理员身份认证。项目没有登录 / 会话 / 角色体系。
# 2. 仅允许部署在绑定 127.0.0.1 的进程；禁止 0.0.0.0 / 局域网 / 公网。
# 3. customer_id 只是可选过滤条件，不承担身份认证。
# 4. 授权数据规则：默认只发送 order_id / status / created_at / amount；
#    customer_id 与 paid_at 仅在调用方显式要求时才附加。
# 5. 本段代码不写任何日志，避免把订单内容落到日志文件。
# ========================================================

ADMIN_ORDER_TOOL_ENABLED = True

QUERY_ORDERS_FAILURE_PREFIX = "订单查询失败："

ADMIN_ORDER_SCOPE = "admin_all_customers"

ADMIN_ORDER_DEFAULT_PAGE_SIZE = 20
ADMIN_ORDER_MAX_PAGE_SIZE = 20

# 默认白名单：不含客户编号与付款时间
ORDER_TOOL_DEFAULT_FIELDS = (
    "order_id",
    "status",
    "created_at",
    "amount",
)

# 可选白名单：仅在对应布尔参数为 True 时附加
# 键为字段名，值为控制它的参数名
ORDER_TOOL_OPTIONAL_FIELDS = {
    "customer_id": "include_customer_id",
    "paid_at": "include_paid_at",
}


def build_query_orders_tool_definition():
    """构造 query_orders 的工具定义（供 Tool Registry 注册）。

    ORDER_STATUSES 在此函数内延迟导入：tools.py 会被多个独立测试模块
    直接 import，不能在模块加载期依赖 business 包。
    """
    from business.validators import ORDER_STATUSES

    status_text = "、".join(sorted(ORDER_STATUSES))

    return {
        "name": "query_orders",
        "description": (
            "查询业务数据库中的订单，只读。"
            "默认按创建时间倒序返回（created_at DESC, order_id ASC）。"
            "可按订单编号查询单条详情，或按客户编号、中文订单状态筛选并分页。"
            "这是管理员全局查询（scope=admin_all_customers），"
            "结果覆盖所有客户，不代表当前用户本人的订单；"
            "customer_id 仅作筛选条件，不是身份认证。"
            "默认只返回 order_id / status / created_at / amount 四个字段；"
            "客户编号与付款时间需通过 include_customer_id / include_paid_at "
            "显式开启，默认 false。不支持按时间范围筛选。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": (
                        "订单编号，精确匹配。提供时返回该订单详情，"
                        "忽略筛选与分页。"
                    ),
                },
                "customer_id": {
                    "type": "string",
                    "description": (
                        "客户编号，精确匹配，仅作筛选条件，"
                        "不代表当前用户身份。"
                    ),
                },
                "status": {
                    "type": "string",
                    "description": (
                        "订单状态，必须是以下之一：" + status_text + "。"
                    ),
                },
                "page": {
                    "type": "integer",
                    "description": "页码，必须为正整数，从 1 开始，默认 1。",
                },
                "page_size": {
                    "type": "integer",
                    "description": (
                        "每页条数，必须为正整数；大于 "
                        + str(ADMIN_ORDER_MAX_PAGE_SIZE)
                        + " 时按 " + str(ADMIN_ORDER_MAX_PAGE_SIZE)
                        + " 处理；默认 "
                        + str(ADMIN_ORDER_DEFAULT_PAGE_SIZE) + "。"
                    ),
                },
                "include_customer_id": {
                    "type": "boolean",
                    "description": (
                        "是否在结果中附带客户编号。默认 false；"
                        "仅当用户明确要求查看客户编号时才设为 true。"
                    ),
                },
                "include_paid_at": {
                    "type": "boolean",
                    "description": (
                        "是否在结果中附带付款时间。默认 false；"
                        "仅当用户明确要求查看付款时间时才设为 true。"
                    ),
                },
            },
            "required": [],
        },
    }


def make_query_orders_tool():
    """返回 query_orders 工具函数（管理员全局只读）。

    返回契约（工具原始返回值，未经 Tool Registry 包装）：

    - 列表：
        {"scope", "filters", "count", "total", "page", "page_size",
         "has_more", "items": [ {字段}, ... ]}
    - 详情命中：
        {"scope", "found": True, "order": {...}, "count": 1,
         "items": [{...}]}
    - 详情未命中：
        {"scope", "found": False, "order_id", "count": 0, "items": []}

    失败一律抛 ValueError，消息以 QUERY_ORDERS_FAILURE_PREFIX 开头。
    参数错误给出具体原因；数据库异常只给出安全文案，
    不携带原始异常文本、绝对路径、SQL 参数或实现细节。
    """

    def _fail(message):
        raise ValueError(f"{QUERY_ORDERS_FAILURE_PREFIX}{message}")

    def _is_int(value):
        # bool 是 int 的子类，必须显式排除
        return isinstance(value, int) and not isinstance(value, bool)

    def _clean_text(value, field_name):
        if value is None:
            return None
        if not isinstance(value, str):
            _fail(f"{field_name} 必须是字符串")
        value = value.strip()
        return value or None

    def _project(order, include_customer_id, include_paid_at):
        """字段白名单投影：绝不整体返回数据库原始字典。"""
        flags = {
            "include_customer_id": include_customer_id,
            "include_paid_at": include_paid_at,
        }

        fields = list(ORDER_TOOL_DEFAULT_FIELDS)

        for field_name, flag_name in ORDER_TOOL_OPTIONAL_FIELDS.items():
            if flags[flag_name]:
                fields.append(field_name)

        return {name: order.get(name) for name in fields}

    def query_orders(
        order_id=None,
        customer_id=None,
        status=None,
        page=1,
        page_size=ADMIN_ORDER_DEFAULT_PAGE_SIZE,
        include_customer_id=False,
        include_paid_at=False,
    ):
        # 延迟导入：保持 tools.py 模块加载期不依赖 business 包
        from business.queries import get_order, list_orders
        from business.validators import ORDER_STATUSES

        # --------------------------------------------
        # 1) 布尔参数类型（注册表已校验，这里防内部直接调用）
        # --------------------------------------------
        if not isinstance(include_customer_id, bool):
            _fail("include_customer_id 必须是布尔值")

        if not isinstance(include_paid_at, bool):
            _fail("include_paid_at 必须是布尔值")

        order_id = _clean_text(order_id, "order_id")
        customer_id = _clean_text(customer_id, "customer_id")
        status_value = _clean_text(status, "status")

        # --------------------------------------------
        # 2) 指定订单号：查详情
        #    订单不存在是正常业务结果，不是数据库异常
        # --------------------------------------------
        if order_id is not None:
            try:
                order = get_order(order_id)
            except Exception:
                # 不把原始异常文本 / 路径 / SQL 参数传给模型
                _fail("暂时无法读取订单数据，请稍后重试。")

            if order is None:
                return {
                    "scope": ADMIN_ORDER_SCOPE,
                    "found": False,
                    "order_id": order_id,
                    "count": 0,
                    "items": [],
                }

            projected = _project(
                order,
                include_customer_id,
                include_paid_at,
            )

            return {
                "scope": ADMIN_ORDER_SCOPE,
                "found": True,
                "order": projected,
                "count": 1,
                "items": [projected],
            }

        # --------------------------------------------
        # 3) 分页区间校验
        #    Tool Registry 只校验类型，没有 min/max 能力，
        #    因此上下界必须由本工具兜住。
        # --------------------------------------------
        if not _is_int(page) or page < 1:
            _fail("page 必须是正整数（从 1 开始）")

        if not _is_int(page_size) or page_size < 1:
            _fail("page_size 必须是正整数（至少 1）")

        safe_size = min(page_size, ADMIN_ORDER_MAX_PAGE_SIZE)

        # --------------------------------------------
        # 4) 状态白名单（中文枚举，来自 business.validators）
        # --------------------------------------------
        if status_value is not None and status_value not in ORDER_STATUSES:
            _fail(
                "status 不在允许范围内："
                f"{status_value}；允许值："
                + "、".join(sorted(ORDER_STATUSES))
            )

        # --------------------------------------------
        # 5) 列表查询：复用现有只读分页函数，不写任何 SQL
        # --------------------------------------------
        try:
            page_result = list_orders(
                order_status=status_value,
                customer_id=customer_id,
                page=page,
                page_size=safe_size,
            )
        except Exception:
            _fail("暂时无法读取订单数据，请稍后重试。")

        items = [
            _project(row, include_customer_id, include_paid_at)
            for row in (page_result.get("items") or [])
            if isinstance(row, dict)
        ]

        total = page_result.get("total", 0)
        safe_page = page_result.get("page", page)
        result_size = page_result.get("page_size", safe_size)

        return {
            "scope": ADMIN_ORDER_SCOPE,
            "filters": {
                "customer_id": customer_id,
                "status": status_value,
            },
            "count": len(items),
            "total": total,
            "page": safe_page,
            "page_size": result_size,
            "has_more": (safe_page * result_size) < total,
            "items": items,
        }

    return query_orders
