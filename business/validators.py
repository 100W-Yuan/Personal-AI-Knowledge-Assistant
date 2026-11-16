"""
企业业务数据校验模块。

负责：
1. 校验订单数据
2. 校验支付数据
3. 校验物流数据
4. 校验售后数据
5. 校验客服工单数据

本模块只负责检查数据，不连接数据库，也不调用大模型 API。
"""

from datetime import datetime


# 允许的业务状态
ORDER_STATUSES = {
    "待支付",
    "已支付",
    "待发货",
    "已发货",
    "已完成",
    "已取消",
}

PAYMENT_STATUSES = {
    "待支付",
    "支付成功",
    "支付失败",
    "已退款",
}

AFTERSALE_STATUSES = {
    "待处理",
    "处理中",
    "已完成",
    "已关闭",
}

WORKORDER_STATUSES = {
    "待处理",
    "处理中",
    "已完成",
    "已关闭",
}


def validate_required(value, field_name):
    """检查必填字段是否为空。"""

    if value is None or str(value).strip() == "":
        raise ValueError(f"{field_name}不能为空")


def validate_amount(amount):
    """检查金额是否合法。"""

    try:
        value = float(amount)
    except (TypeError, ValueError):
        raise ValueError("金额必须是有效数字")

    if value < 0:
        raise ValueError("金额不能为负数")

    return value

def validate_boolean(value, field_name):
    """严格校验布尔值，避免 bool('false') 返回 True。"""

    if isinstance(value, bool):
        return value

    if isinstance(value, int) and value in (0, 1):
        return bool(value)

    if isinstance(value, str):
        normalized = value.strip().lower()

        if normalized in ("true", "1"):
            return True

        if normalized in ("false", "0"):
            return False

    raise ValueError(f"{field_name}必须是布尔值、0 或 1")

def validate_datetime(value, field_name):
    """检查日期时间格式。"""

    validate_required(value, field_name)

    try:
        datetime.fromisoformat(str(value))
    except ValueError:
        raise ValueError(
            f"{field_name}格式不正确，请使用 ISO 格式，例如 2026-11-04T15:30:00"
        )


def validate_order(data):
    """校验订单数据。"""

    required_fields = [
        "order_id",
        "customer_id",
        "status",
        "created_at",
        "amount",
    ]

    for field in required_fields:
        validate_required(data.get(field), field)

    if data["status"] not in ORDER_STATUSES:
        raise ValueError("订单状态不合法")

    validate_datetime(data["created_at"], "created_at")

    amount = validate_amount(data["amount"])

    if data.get("paid_at") not in (None, ""):
        validate_datetime(data["paid_at"], "paid_at")

    return {
        "order_id": str(data["order_id"]).strip(),
        "customer_id": str(data["customer_id"]).strip(),
        "status": data["status"],
        "created_at": data["created_at"],
        "amount": amount,
        "paid_at": data.get("paid_at") or None,
    }


def validate_payment(data):
    """校验支付数据。"""

    required_fields = [
        "payment_id",
        "order_id",
        "method",
        "amount",
        "status",
    ]

    for field in required_fields:
        validate_required(data.get(field), field)

    if data["status"] not in PAYMENT_STATUSES:
        raise ValueError("支付状态不合法")

    amount = validate_amount(data["amount"])

    if data.get("paid_at") not in (None, ""):
        validate_datetime(data["paid_at"], "paid_at")

    return {
        "payment_id": str(data["payment_id"]).strip(),
        "order_id": str(data["order_id"]).strip(),
        "paid_at": data.get("paid_at") or None,
        "method": str(data["method"]).strip(),
        "amount": amount,
        "status": data["status"],
    }


def validate_shipment(data):
    """校验物流数据。"""

    required_fields = [
        "shipment_id",
        "order_id",
        "carrier",
        "status",
    ]

    for field in required_fields:
        validate_required(data.get(field), field)

    if data.get("t_last") not in (None, ""):
        validate_datetime(data["t_last"], "t_last")

    return {
        "shipment_id": str(data["shipment_id"]).strip(),
        "order_id": str(data["order_id"]).strip(),
        "carrier": str(data["carrier"]).strip(),
        "status": str(data["status"]).strip(),
        "at_hub": validate_boolean(data.get("at_hub", False), "at_hub"),
        "departed_hub": validate_boolean(data.get("departed_hub", False), "departed_hub"),
        "t_last": data.get("t_last") or None,
    }


def validate_aftersale(data):
    """校验售后数据。"""

    required_fields = [
        "aftersale_id",
        "order_id",
        "type",
        "status",
        "created_at",
    ]

    for field in required_fields:
        validate_required(data.get(field), field)

    if data["status"] not in AFTERSALE_STATUSES:
        raise ValueError("售后状态不合法")

    validate_datetime(data["created_at"], "created_at")

    return {
        "aftersale_id": str(data["aftersale_id"]).strip(),
        "order_id": str(data["order_id"]).strip(),
        "type": str(data["type"]).strip(),
        "status": data["status"],
        "created_at": data["created_at"],
    }


def validate_workorder(data):
    """校验客服工单数据。"""

    required_fields = [
        "workorder_id",
        "order_id",
        "type",
        "status",
        "created_at",
    ]

    for field in required_fields:
        validate_required(data.get(field), field)

    if data["status"] not in WORKORDER_STATUSES:
        raise ValueError("工单状态不合法")

    validate_datetime(data["created_at"], "created_at")

    return {
        "workorder_id": str(data["workorder_id"]).strip(),
        "order_id": str(data["order_id"]).strip(),
        "type": str(data["type"]).strip(),
        "status": data["status"],
        "created_at": data["created_at"],
        "assigned_to": data.get("assigned_to") or None,
    }

def validate_shipment_event(data):
    """校验物流事件数据。"""

    required_fields = [
        "event_id",
        "shipment_id",
        "event_type",
        "event_time",
    ]

    for field in required_fields:
        validate_required(data.get(field), field)

    validate_datetime(data["event_time"], "event_time")

    return {
        "event_id": str(data["event_id"]).strip(),
        "shipment_id": str(data["shipment_id"]).strip(),
        "event_type": str(data["event_type"]).strip(),
        "event_time": data["event_time"],
        "detail": str(data.get("detail") or "").strip(),
        "is_real_advance": validate_boolean(
            data.get("is_real_advance", False),
            "is_real_advance",
        ),
        "is_exception": validate_boolean(
            data.get("is_exception", False),
            "is_exception",
        ),
    }
