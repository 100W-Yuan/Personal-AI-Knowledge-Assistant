
"""
企业业务服务模块。

当前已实现：
- 创建订单
- 创建支付记录
- 创建物流记录
- 创建物流事件
"""

import sqlite3

from business.db import get_connection
from business.validators import (
    validate_order,
    validate_payment,
    validate_shipment,
    validate_shipment_event,
    validate_aftersale,
    validate_workorder,
)

def create_order(data):
    """校验订单数据，并将合法订单写入数据库。"""

    order = validate_order(data)
    connection = get_connection()

    try:
        connection.execute(
            """
            INSERT INTO orders (
                order_id, customer_id, status,
                created_at, amount, paid_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                order["order_id"],
                order["customer_id"],
                order["status"],
                order["created_at"],
                order["amount"],
                order["paid_at"],
            ),
        )
        connection.commit()

    except sqlite3.IntegrityError as exc:
        connection.rollback()

        if "UNIQUE constraint failed: orders.order_id" in str(exc):
            raise ValueError("订单编号已存在") from exc

        raise ValueError("订单数据违反数据库约束") from exc

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()

    return order


def create_payment(data):
    """校验支付数据，并将合法支付记录写入数据库。"""

    payment = validate_payment(data)
    connection = get_connection()

    try:
        connection.execute(
            """
            INSERT INTO payments (
                payment_id, order_id, paid_at,
                method, amount, status
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                payment["payment_id"],
                payment["order_id"],
                payment["paid_at"],
                payment["method"],
                payment["amount"],
                payment["status"],
            ),
        )
        connection.commit()

    except sqlite3.IntegrityError as exc:
        connection.rollback()

        if "UNIQUE constraint failed: payments.payment_id" in str(exc):
            raise ValueError("支付编号已存在") from exc

        if "FOREIGN KEY constraint failed" in str(exc):
            raise ValueError("关联订单不存在") from exc

        raise ValueError("支付数据违反数据库约束") from exc

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()

    return payment


def create_shipment(data):
    """校验物流数据，并将合法物流记录写入数据库。"""

    shipment = validate_shipment(data)
    connection = get_connection()

    try:
        connection.execute(
            """
            INSERT INTO shipments (
                shipment_id, order_id, carrier,
                status, at_hub, departed_hub, t_last
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                shipment["shipment_id"],
                shipment["order_id"],
                shipment["carrier"],
                shipment["status"],
                int(shipment["at_hub"]),
                int(shipment["departed_hub"]),
                shipment["t_last"],
            ),
        )
        connection.commit()

    except sqlite3.IntegrityError as exc:
        connection.rollback()

        if "UNIQUE constraint failed: shipments.shipment_id" in str(exc):
            raise ValueError("物流编号已存在") from exc

        if "FOREIGN KEY constraint failed" in str(exc):
            raise ValueError("关联订单不存在") from exc

        raise ValueError("物流数据违反数据库约束") from exc

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()

    return shipment


def create_shipment_event(data):
    """校验物流事件，并将合法事件写入数据库。"""

    event = validate_shipment_event(data)
    connection = get_connection()

    try:
        connection.execute(
            """
            INSERT INTO shipment_events (
                event_id, shipment_id, event_type,
                event_time, detail, is_real_advance, is_exception
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event["event_id"],
                event["shipment_id"],
                event["event_type"],
                event["event_time"],
                event["detail"],
                int(event["is_real_advance"]),
                int(event["is_exception"]),
            ),
        )
        connection.commit()

    except sqlite3.IntegrityError as exc:
        connection.rollback()

        if "UNIQUE constraint failed: shipment_events.event_id" in str(exc):
            raise ValueError("物流事件编号已存在") from exc

        if "FOREIGN KEY constraint failed" in str(exc):
            raise ValueError("关联物流记录不存在") from exc

        raise ValueError("物流事件数据违反数据库约束") from exc

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()

    return event

def create_aftersale(data):
    """校验售后数据，并将合法售后记录写入数据库。"""

    aftersale = validate_aftersale(data)
    connection = get_connection()

    try:
        connection.execute(
            """
            INSERT INTO aftersales (
                aftersale_id,
                order_id,
                type,
                status,
                created_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                aftersale["aftersale_id"],
                aftersale["order_id"],
                aftersale["type"],
                aftersale["status"],
                aftersale["created_at"],
            ),
        )

        connection.commit()

    except sqlite3.IntegrityError as exc:
        connection.rollback()

        if "UNIQUE constraint failed: aftersales.aftersale_id" in str(exc):
            raise ValueError("售后编号已存在") from exc

        if "FOREIGN KEY constraint failed" in str(exc):
            raise ValueError("关联订单不存在") from exc

        raise ValueError("售后数据违反数据库约束") from exc

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()

    return aftersale


def create_workorder(data):
    """校验工单数据，并将合法工单写入数据库。"""

    workorder = validate_workorder(data)
    connection = get_connection()

    try:
        connection.execute(
            """
            INSERT INTO workorders (
                workorder_id,
                order_id,
                type,
                status,
                created_at,
                assigned_to
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                workorder["workorder_id"],
                workorder["order_id"],
                workorder["type"],
                workorder["status"],
                workorder["created_at"],
                workorder["assigned_to"],
            ),
        )

        connection.commit()

    except sqlite3.IntegrityError as exc:
        connection.rollback()

        if "UNIQUE constraint failed: workorders.workorder_id" in str(exc):
            raise ValueError("工单编号已存在") from exc

        if "FOREIGN KEY constraint failed" in str(exc):
            raise ValueError("关联订单不存在") from exc

        raise ValueError("工单数据违反数据库约束") from exc

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()

    return workorder
