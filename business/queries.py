
"""
企业业务只读查询模块。

负责：
1. 查询订单列表与订单详情
2. 查询订单下的支付、物流、售后、客服工单
3. 查询物流记录下的物流事件时间线

约束：
1. 本模块只执行 SELECT 查询，不执行任何写语句，不调用 commit
2. 所有用户输入值通过 ? 参数绑定
3. 数据库连接在 finally 中关闭
4. 布尔列在返回前显式转换为 Python bool
5. 返回字段明确列出，禁止 SELECT *

注意：
本模块只读取业务数据库，不读取个人 AI 记忆数据库，也不调用大模型 API。
"""

from business.db import get_connection


# 分页默认值与上限
DEFAULT_PAGE = 1
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100

# 返回字段清单（明确列出，禁止 SELECT *）
ORDER_COLUMNS = "order_id, customer_id, status, created_at, amount, paid_at"

PAYMENT_COLUMNS = "payment_id, order_id, paid_at, method, amount, status"

SHIPMENT_COLUMNS = (
    "shipment_id, order_id, carrier, status, at_hub, departed_hub, t_last"
)

SHIPMENT_EVENT_COLUMNS = (
    "event_id, shipment_id, event_type, event_time, detail, "
    "is_real_advance, is_exception"
)

AFTERSALE_COLUMNS = "aftersale_id, order_id, type, status, created_at"

WORKORDER_COLUMNS = (
    "workorder_id, order_id, type, status, created_at, assigned_to"
)

# 父资源存在性检查语句
ORDER_EXISTS_SQL = "SELECT 1 FROM orders WHERE order_id = ?"
SHIPMENT_EXISTS_SQL = "SELECT 1 FROM shipments WHERE shipment_id = ?"


# =========================
# 行转换：显式列出字段，布尔列转 bool
# =========================

def _row_to_order(row):
    """把订单行转换为可序列化的字典。"""
    return {
        "order_id": row["order_id"],
        "customer_id": row["customer_id"],
        "status": row["status"],
        "created_at": row["created_at"],
        "amount": row["amount"],
        "paid_at": row["paid_at"],
    }


def _row_to_payment(row):
    """把支付行转换为可序列化的字典。"""
    return {
        "payment_id": row["payment_id"],
        "order_id": row["order_id"],
        "paid_at": row["paid_at"],
        "method": row["method"],
        "amount": row["amount"],
        "status": row["status"],
    }


def _row_to_shipment(row):
    """把物流行转换为可序列化的字典，布尔列转 bool。"""
    return {
        "shipment_id": row["shipment_id"],
        "order_id": row["order_id"],
        "carrier": row["carrier"],
        "status": row["status"],
        "at_hub": bool(row["at_hub"]),
        "departed_hub": bool(row["departed_hub"]),
        "t_last": row["t_last"],
    }


def _row_to_shipment_event(row):
    """把物流事件行转换为可序列化的字典，布尔列转 bool。"""
    return {
        "event_id": row["event_id"],
        "shipment_id": row["shipment_id"],
        "event_type": row["event_type"],
        "event_time": row["event_time"],
        "detail": row["detail"],
        "is_real_advance": bool(row["is_real_advance"]),
        "is_exception": bool(row["is_exception"]),
    }


def _row_to_aftersale(row):
    """把售后行转换为可序列化的字典。"""
    return {
        "aftersale_id": row["aftersale_id"],
        "order_id": row["order_id"],
        "type": row["type"],
        "status": row["status"],
        "created_at": row["created_at"],
    }


def _row_to_workorder(row):
    """把工单行转换为可序列化的字典。"""
    return {
        "workorder_id": row["workorder_id"],
        "order_id": row["order_id"],
        "type": row["type"],
        "status": row["status"],
        "created_at": row["created_at"],
        "assigned_to": row["assigned_to"],
    }


# =========================
# 连接与只读事务
# =========================

def _normalize_paging(page, page_size):
    """兜底归一化分页参数。

    API 层已通过 Query(ge=1, le=100) 校验，这里再兜底一次，
    防止内部直接调用时产生负数 OFFSET。
    """
    safe_page = page if page >= 1 else DEFAULT_PAGE
    safe_size = page_size if page_size >= 1 else DEFAULT_PAGE_SIZE

    if safe_size > MAX_PAGE_SIZE:
        safe_size = MAX_PAGE_SIZE

    return safe_page, safe_size


def _apply_read_only_protection(connection):
    """把当前连接切换为只读模式。

    query_only 是连接级 PRAGMA，只影响本连接，
    不影响 service.py 使用独立连接执行的业务写入。
    必须在 BEGIN 之前调用。
    """
    connection.execute("PRAGMA query_only = ON")


def _begin_read_transaction(connection):
    """开启只读事务，使 COUNT 与当前页 SELECT 看到同一份快照。"""
    connection.execute("BEGIN")


def _finish_read_transaction(connection):
    """结束只读事务。

    本模块不执行任何写语句，因此用 rollback 结束事务，
    不调用 commit，也不修改任何业务数据。
    """
    if connection.in_transaction:
        connection.rollback()


def _read_page(count_sql, items_sql, params, page, page_size, converter):
    """在同一连接、同一只读事务中执行 COUNT 与分页 SELECT。"""
    safe_page, safe_size = _normalize_paging(page, page_size)
    offset = (safe_page - 1) * safe_size

    connection = get_connection()

    try:
        _apply_read_only_protection(connection)
        _begin_read_transaction(connection)

        total_row = connection.execute(count_sql, params).fetchone()
        total = int(total_row[0]) if total_row is not None else 0

        rows = connection.execute(
            items_sql, list(params) + [safe_size, offset]
        ).fetchall()

        return {
            "items": [converter(row) for row in rows],
            "total": total,
            "page": safe_page,
            "page_size": safe_size,
        }

    finally:
        # 先结束只读事务，再确保连接关闭（异常路径同样关闭）
        try:
            _finish_read_transaction(connection)
        finally:
            connection.close()


def _read_children(
    parent_exists_sql,
    parent_params,
    count_sql,
    items_sql,
    params,
    page,
    page_size,
    converter,
):
    """先确认父资源存在，再读取子资源分页数据。

    父资源不存在时返回 None，由 API 层转换为 404。
    父资源存在但没有子记录时返回空 items 与 total 0。
    """
    safe_page, safe_size = _normalize_paging(page, page_size)
    offset = (safe_page - 1) * safe_size

    connection = get_connection()

    try:
        _apply_read_only_protection(connection)
        _begin_read_transaction(connection)

        exists_row = connection.execute(
            parent_exists_sql, parent_params
        ).fetchone()

        if exists_row is None:
            return None

        total_row = connection.execute(count_sql, params).fetchone()
        total = int(total_row[0]) if total_row is not None else 0

        rows = connection.execute(
            items_sql, list(params) + [safe_size, offset]
        ).fetchall()

        return {
            "items": [converter(row) for row in rows],
            "total": total,
            "page": safe_page,
            "page_size": safe_size,
        }

    finally:
        try:
            _finish_read_transaction(connection)
        finally:
            connection.close()


# =========================
# 订单查询
# =========================

def list_orders(
    order_status=None,
    customer_id=None,
    created_from=None,
    created_to=None,
    page=DEFAULT_PAGE,
    page_size=DEFAULT_PAGE_SIZE,
):
    """查询订单列表，支持状态、客户与创建时间范围筛选。

    排序为 created_at 倒序，并以 order_id 正序作为第二排序键，
    保证同一时间戳下分页结果稳定。
    """
    conditions = []
    params = []

    if order_status is not None:
        conditions.append("status = ?")
        params.append(order_status)

    if customer_id is not None:
        conditions.append("customer_id = ?")
        params.append(customer_id)

    if created_from is not None:
        conditions.append("created_at >= ?")
        params.append(created_from)

    if created_to is not None:
        conditions.append("created_at <= ?")
        params.append(created_to)

    where_clause = ""
    if conditions:
        # 只拼接固定代码常量，用户输入一律走 ? 绑定
        where_clause = " WHERE " + " AND ".join(conditions)

    count_sql = "SELECT COUNT(*) FROM orders" + where_clause

    items_sql = (
        "SELECT "
        + ORDER_COLUMNS
        + " FROM orders"
        + where_clause
        + " ORDER BY created_at DESC, order_id ASC LIMIT ? OFFSET ?"
    )

    return _read_page(
        count_sql,
        items_sql,
        params,
        page,
        page_size,
        _row_to_order,
    )


def get_order(order_id):
    """按订单编号查询单条订单。

    未找到时返回 None，由 API 层转换为 404。
    单条 SELECT 本身即原子操作，无需显式事务。
    """
    connection = get_connection()

    try:
        _apply_read_only_protection(connection)

        row = connection.execute(
            "SELECT " + ORDER_COLUMNS + " FROM orders WHERE order_id = ?",
            (order_id,),
        ).fetchone()

        if row is None:
            return None

        return _row_to_order(row)

    finally:
        connection.close()


# =========================
# 订单子资源查询
# =========================

def list_order_payments(
    order_id,
    page=DEFAULT_PAGE,
    page_size=DEFAULT_PAGE_SIZE,
):
    """查询指定订单的支付记录，按支付时间倒序。"""
    count_sql = "SELECT COUNT(*) FROM payments WHERE order_id = ?"

    items_sql = (
        "SELECT "
        + PAYMENT_COLUMNS
        + " FROM payments WHERE order_id = ?"
        + " ORDER BY paid_at DESC, payment_id ASC LIMIT ? OFFSET ?"
    )

    return _read_children(
        ORDER_EXISTS_SQL,
        (order_id,),
        count_sql,
        items_sql,
        (order_id,),
        page,
        page_size,
        _row_to_payment,
    )


def list_order_shipments(
    order_id,
    page=DEFAULT_PAGE,
    page_size=DEFAULT_PAGE_SIZE,
):
    """查询指定订单的物流记录，按最近更新时间倒序。"""
    count_sql = "SELECT COUNT(*) FROM shipments WHERE order_id = ?"

    items_sql = (
        "SELECT "
        + SHIPMENT_COLUMNS
        + " FROM shipments WHERE order_id = ?"
        + " ORDER BY t_last DESC, shipment_id ASC LIMIT ? OFFSET ?"
    )

    return _read_children(
        ORDER_EXISTS_SQL,
        (order_id,),
        count_sql,
        items_sql,
        (order_id,),
        page,
        page_size,
        _row_to_shipment,
    )


def list_shipment_events(
    shipment_id,
    page=DEFAULT_PAGE,
    page_size=DEFAULT_PAGE_SIZE,
):
    """查询指定物流记录的物流事件，按发生时间正序（时间线顺序）。"""
    count_sql = "SELECT COUNT(*) FROM shipment_events WHERE shipment_id = ?"

    items_sql = (
        "SELECT "
        + SHIPMENT_EVENT_COLUMNS
        + " FROM shipment_events WHERE shipment_id = ?"
        + " ORDER BY event_time ASC, event_id ASC LIMIT ? OFFSET ?"
    )

    return _read_children(
        SHIPMENT_EXISTS_SQL,
        (shipment_id,),
        count_sql,
        items_sql,
        (shipment_id,),
        page,
        page_size,
        _row_to_shipment_event,
    )


def list_order_aftersales(
    order_id,
    page=DEFAULT_PAGE,
    page_size=DEFAULT_PAGE_SIZE,
):
    """查询指定订单的售后记录，按创建时间倒序。"""
    count_sql = "SELECT COUNT(*) FROM aftersales WHERE order_id = ?"

    items_sql = (
        "SELECT "
        + AFTERSALE_COLUMNS
        + " FROM aftersales WHERE order_id = ?"
        + " ORDER BY created_at DESC, aftersale_id ASC LIMIT ? OFFSET ?"
    )

    return _read_children(
        ORDER_EXISTS_SQL,
        (order_id,),
        count_sql,
        items_sql,
        (order_id,),
        page,
        page_size,
        _row_to_aftersale,
    )


def list_order_workorders(
    order_id,
    page=DEFAULT_PAGE,
    page_size=DEFAULT_PAGE_SIZE,
):
    """查询指定订单的客服工单，按创建时间倒序。"""
    count_sql = "SELECT COUNT(*) FROM workorders WHERE order_id = ?"

    items_sql = (
        "SELECT "
        + WORKORDER_COLUMNS
        + " FROM workorders WHERE order_id = ?"
        + " ORDER BY created_at DESC, workorder_id ASC LIMIT ? OFFSET ?"
    )

    return _read_children(
        ORDER_EXISTS_SQL,
        (order_id,),
        count_sql,
        items_sql,
        (order_id,),
        page,
        page_size,
        _row_to_workorder,
    )
