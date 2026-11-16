from typing import Any

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, StrictBool

from business.queries import (
    get_order,
    list_order_aftersales,
    list_order_payments,
    list_order_shipments,
    list_order_workorders,
    list_orders,
    list_shipment_events,
)
from business.service import (
    create_order,
    create_payment,
    create_shipment,
    create_shipment_event,
    create_aftersale,
    create_workorder,
)
from business.validators import ORDER_STATUSES


router = APIRouter(prefix="/business", tags=["Business"])


# =========================
# 请求模型
# =========================

class BaseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OrderCreateRequest(BaseRequest):
    order_id: str
    customer_id: str
    status: str
    created_at: str
    amount: float
    paid_at: str | None = None


class PaymentCreateRequest(BaseRequest):
    payment_id: str
    order_id: str
    paid_at: str | None = None
    method: str
    amount: float
    status: str


class ShipmentCreateRequest(BaseRequest):
    shipment_id: str
    order_id: str
    carrier: str
    status: str
    at_hub: StrictBool = False
    departed_hub: StrictBool = False
    t_last: str | None = None


class ShipmentEventCreateRequest(BaseRequest):
    event_id: str
    shipment_id: str
    event_type: str
    event_time: str
    detail: str | None = None
    is_real_advance: StrictBool = False
    is_exception: StrictBool = False


class AfterSaleCreateRequest(BaseRequest):
    aftersale_id: str
    order_id: str
    type: str
    status: str
    created_at: str


class WorkOrderCreateRequest(BaseRequest):
    workorder_id: str
    order_id: str
    type: str
    status: str
    created_at: str
    assigned_to: str | None = None


# =========================
# 统一异常处理
# =========================

def handle_business_error(error: ValueError) -> HTTPException:
    message = str(error)

    if "已存在" in message:
        error_type = "duplicate_record"
        http_status = status.HTTP_409_CONFLICT
    elif "不存在" in message:
        error_type = "invalid_reference"
        http_status = status.HTTP_422_UNPROCESSABLE_ENTITY
    else:
        error_type = "validation_error"
        http_status = status.HTTP_422_UNPROCESSABLE_ENTITY

    return HTTPException(
        status_code=http_status,
        detail={
            "type": error_type,
            "message": message,
        },
    )


def execute_service(service_function: Any, data: BaseModel) -> dict:
    try:
        result = service_function(data.model_dump())
        return {
            "ok": True,
            "data": result,
        }
    except ValueError as error:
        raise handle_business_error(error) from error


# =========================
# 业务 API
# =========================

@router.post("/orders", status_code=status.HTTP_201_CREATED)
def create_order_api(data: OrderCreateRequest):
    return execute_service(create_order, data)


@router.post("/payments", status_code=status.HTTP_201_CREATED)
def create_payment_api(data: PaymentCreateRequest):
    return execute_service(create_payment, data)


@router.post("/shipments", status_code=status.HTTP_201_CREATED)
def create_shipment_api(data: ShipmentCreateRequest):
    return execute_service(create_shipment, data)


@router.post("/shipment-events", status_code=status.HTTP_201_CREATED)
def create_shipment_event_api(data: ShipmentEventCreateRequest):
    return execute_service(create_shipment_event, data)


@router.post("/aftersales", status_code=status.HTTP_201_CREATED)
def create_aftersale_api(data: AfterSaleCreateRequest):
    return execute_service(create_aftersale, data)


@router.post("/workorders", status_code=status.HTTP_201_CREATED)
def create_workorder_api(data: WorkOrderCreateRequest):
    return execute_service(create_workorder, data)


# =========================
# 只读查询异常与响应工具
# =========================

def not_found(error_type: str, message: str) -> HTTPException:
    """构造 404 异常。

    查询接口的"资源未找到"不走 handle_business_error，
    避免被 "不存在" 子串映射成 422 invalid_reference。
    """
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={
            "type": error_type,
            "message": message,
        },
    )


def success(data: Any) -> dict:
    """包装只读查询的成功响应。"""
    return {
        "ok": True,
        "data": data,
    }


# =========================
# 只读查询 API
# =========================

@router.get("/orders")
def list_orders_api(
    order_status: str | None = Query(default=None, alias="status"),
    customer_id: str | None = Query(default=None),
    created_from: str | None = Query(default=None),
    created_to: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
):
    """查询订单列表，支持筛选与分页。"""
    if order_status is not None and order_status not in ORDER_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "type": "validation_error",
                "message": "订单状态不在允许范围内",
            },
        )

    return success(
        list_orders(
            order_status=order_status,
            customer_id=customer_id,
            created_from=created_from,
            created_to=created_to,
            page=page,
            page_size=page_size,
        )
    )


@router.get("/orders/{order_id}")
def get_order_api(order_id: str):
    """按订单编号查询订单详情。"""
    order = get_order(order_id)

    if order is None:
        raise not_found("order_not_found", "订单不存在")

    return success(order)


@router.get("/orders/{order_id}/payments")
def list_order_payments_api(
    order_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
):
    """查询指定订单的支付记录。"""
    data = list_order_payments(order_id, page=page, page_size=page_size)

    if data is None:
        raise not_found("order_not_found", "订单不存在")

    return success(data)


@router.get("/orders/{order_id}/shipments")
def list_order_shipments_api(
    order_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
):
    """查询指定订单的物流记录。"""
    data = list_order_shipments(order_id, page=page, page_size=page_size)

    if data is None:
        raise not_found("order_not_found", "订单不存在")

    return success(data)


@router.get("/shipments/{shipment_id}/events")
def list_shipment_events_api(
    shipment_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
):
    """查询指定物流记录的物流事件。"""
    data = list_shipment_events(shipment_id, page=page, page_size=page_size)

    if data is None:
        raise not_found("shipment_not_found", "物流记录不存在")

    return success(data)


@router.get("/orders/{order_id}/aftersales")
def list_order_aftersales_api(
    order_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
):
    """查询指定订单的售后记录。"""
    data = list_order_aftersales(order_id, page=page, page_size=page_size)

    if data is None:
        raise not_found("order_not_found", "订单不存在")

    return success(data)


@router.get("/orders/{order_id}/workorders")
def list_order_workorders_api(
    order_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
):
    """查询指定订单的客服工单。"""
    data = list_order_workorders(order_id, page=page, page_size=page_size)

    if data is None:
        raise not_found("order_not_found", "订单不存在")

    return success(data)