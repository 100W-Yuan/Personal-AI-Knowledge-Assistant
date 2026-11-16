
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class Order:
    """订单信息"""
    order_id: str
    customer_id: str
    status: str
    created_at: datetime
    amount: float
    paid_at: Optional[datetime] = None


@dataclass
class Payment:
    """支付信息"""
    payment_id: str
    order_id: str
    method: str
    amount: float
    status: str
    paid_at: Optional[datetime] = None


@dataclass
class ShipmentEvent:
    """物流事件"""
    event_id: str
    shipment_id: str
    event_type: str
    event_time: datetime
    detail: str
    is_real_advance: bool = False
    is_exception: bool = False


@dataclass
class Shipment:
    """物流信息"""
    shipment_id: str
    order_id: str
    carrier: str
    status: str
    at_hub: bool = False
    departed_hub: bool = False
    t_last: Optional[datetime] = None
    events: List[ShipmentEvent] = field(default_factory=list)


@dataclass
class AfterSale:
    """售后信息"""
    aftersale_id: str
    order_id: str
    type: str
    status: str
    created_at: datetime


@dataclass
class WorkOrder:
    """客服工单"""
    workorder_id: str
    order_id: str
    type: str
    status: str
    created_at: datetime
    assigned_to: Optional[str] = None
