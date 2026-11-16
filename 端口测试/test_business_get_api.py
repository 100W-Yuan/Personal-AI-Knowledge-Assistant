

"""
B4-04 只读查询接口测试。

覆盖范围：
1. 订单列表（空库、分页、状态筛选、非法参数、超出页数）
2. 订单详情（成功、未找到）
3. 订单子资源（支付、物流、售后、工单）
4. 物流事件时间线（正序、布尔类型、父资源未找到）
5. 只读性验证（查询前后数据一致）
6. 索引初始化幂等性
7. 查询异常时连接仍被关闭

隔离方式：
每个测试使用 pytest tmp_path + monkeypatch 替换 business.db 的
DATA_DIR / DATABASE_PATH，绝不触达正式 data/business.db。
"""

import sqlite3

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import business.db as business_db
import business.queries as business_queries
from business.api import router


# =========================
# fixture
# =========================

@pytest.fixture
def temp_database(tmp_path, monkeypatch):
    """把业务数据库锁定在 tmp_path 内，并初始化表与索引。"""
    database_dir = tmp_path / "data"
    database_dir.mkdir()

    database_path = database_dir / "business_get_test.db"

    monkeypatch.setattr(business_db, "DATA_DIR", database_dir)
    monkeypatch.setattr(business_db, "DATABASE_PATH", database_path)

    # 显式断言：隔离必须真实生效，且绝不指向正式 data/business.db
    assert business_db.DATA_DIR == database_dir
    assert business_db.DATABASE_PATH == database_path
    assert business_db.DATABASE_PATH.is_relative_to(tmp_path)
    assert business_db.DATABASE_PATH.name == "business_get_test.db"

    business_db.init_database()

    assert database_path.exists()
    assert database_path.is_relative_to(tmp_path)

    return database_path


@pytest.fixture
def client(temp_database):
    """提供绑定到临时数据库的测试客户端。"""
    app = FastAPI()
    app.include_router(router)

    with TestClient(app) as test_client:
        yield test_client


# =========================
# 造数辅助函数
# =========================

def create_order(
    client,
    order_id="ORDER-001",
    created_at="2026-11-04T10:00:00",
    status="已支付",
    customer_id="CUSTOMER-001",
    amount=99.9,
):
    return client.post(
        "/business/orders",
        json={
            "order_id": order_id,
            "customer_id": customer_id,
            "status": status,
            "created_at": created_at,
            "amount": amount,
            "paid_at": created_at,
        },
    )


def create_payment(
    client,
    payment_id="PAY-001",
    order_id="ORDER-001",
    paid_at="2026-11-04T10:05:00",
    method="支付宝",
    amount=99.9,
    status="支付成功",
):
    return client.post(
        "/business/payments",
        json={
            "payment_id": payment_id,
            "order_id": order_id,
            "paid_at": paid_at,
            "method": method,
            "amount": amount,
            "status": status,
        },
    )


def create_shipment(
    client,
    shipment_id="SHIP-001",
    order_id="ORDER-001",
    carrier="顺丰",
    status="运输中",
    at_hub=True,
    departed_hub=False,
    t_last="2026-11-04T11:00:00",
):
    return client.post(
        "/business/shipments",
        json={
            "shipment_id": shipment_id,
            "order_id": order_id,
            "carrier": carrier,
            "status": status,
            "at_hub": at_hub,
            "departed_hub": departed_hub,
            "t_last": t_last,
        },
    )


def create_shipment_event(
    client,
    event_id,
    shipment_id="SHIP-001",
    event_type="到达转运中心",
    event_time="2026-11-04T12:00:00",
    detail="包裹已到达",
    is_real_advance=False,
    is_exception=False,
):
    return client.post(
        "/business/shipment-events",
        json={
            "event_id": event_id,
            "shipment_id": shipment_id,
            "event_type": event_type,
            "event_time": event_time,
            "detail": detail,
            "is_real_advance": is_real_advance,
            "is_exception": is_exception,
        },
    )


def create_aftersale(
    client,
    aftersale_id="AFTER-001",
    order_id="ORDER-001",
    aftersale_type="退款",
    status="待处理",
    created_at="2026-11-04T13:00:00",
):
    return client.post(
        "/business/aftersales",
        json={
            "aftersale_id": aftersale_id,
            "order_id": order_id,
            "type": aftersale_type,
            "status": status,
            "created_at": created_at,
        },
    )


def create_workorder(
    client,
    workorder_id="WORK-001",
    order_id="ORDER-001",
    workorder_type="订单异常",
    status="待处理",
    created_at="2026-11-04T14:00:00",
    assigned_to="员工A",
):
    return client.post(
        "/business/workorders",
        json={
            "workorder_id": workorder_id,
            "order_id": order_id,
            "type": workorder_type,
            "status": status,
            "created_at": created_at,
            "assigned_to": assigned_to,
        },
    )


# =========================
# 1. 空订单列表
# =========================

def test_empty_order_list(client):
    response = client.get("/business/orders")

    assert response.status_code == 200

    body = response.json()
    assert body["ok"] is True
    assert body["data"]["items"] == []
    assert body["data"]["total"] == 0
    assert body["data"]["page"] == 1
    assert body["data"]["page_size"] == 20


# =========================
# 2. 订单分页
# =========================

def test_order_pagination(client):
    create_order(client, order_id="ORDER-001", created_at="2026-11-04T10:00:00")
    create_order(client, order_id="ORDER-002", created_at="2026-11-04T11:00:00")
    create_order(client, order_id="ORDER-003", created_at="2026-11-04T12:00:00")

    first_page = client.get(
        "/business/orders", params={"page": 1, "page_size": 2}
    )

    assert first_page.status_code == 200
    first_data = first_page.json()["data"]
    assert first_data["total"] == 3
    assert first_data["page"] == 1
    assert first_data["page_size"] == 2
    assert [item["order_id"] for item in first_data["items"]] == [
        "ORDER-003",
        "ORDER-002",
    ]

    second_page = client.get(
        "/business/orders", params={"page": 2, "page_size": 2}
    )

    assert second_page.status_code == 200
    second_data = second_page.json()["data"]
    assert second_data["total"] == 3
    assert [item["order_id"] for item in second_data["items"]] == ["ORDER-001"]


# =========================
# 3. 订单状态筛选
# =========================

def test_order_status_filter(client):
    create_order(client, order_id="ORDER-001", status="已支付")
    create_order(client, order_id="ORDER-002", status="待发货")
    create_order(client, order_id="ORDER-003", status="已支付")

    paid = client.get("/business/orders", params={"status": "已支付"})

    assert paid.status_code == 200
    paid_data = paid.json()["data"]
    assert paid_data["total"] == 2
    assert {item["order_id"] for item in paid_data["items"]} == {
        "ORDER-001",
        "ORDER-003",
    }
    assert all(item["status"] == "已支付" for item in paid_data["items"])

    cancelled = client.get("/business/orders", params={"status": "已取消"})

    assert cancelled.status_code == 200
    assert cancelled.json()["data"]["items"] == []
    assert cancelled.json()["data"]["total"] == 0


# =========================
# 4. 非法状态返回 422
# =========================

def test_invalid_order_status_rejected(client):
    response = client.get("/business/orders", params={"status": "未知状态"})

    assert response.status_code == 422
    assert response.json()["detail"]["type"] == "validation_error"


# =========================
# 5. 非法分页参数返回 422
# =========================

@pytest.mark.parametrize(
    "params",
    [
        {"page": 0},
        {"page": -1},
        {"page_size": 0},
        {"page_size": 101},
        {"page": "abc"},
        {"page_size": "abc"},
    ],
)
def test_invalid_pagination_rejected(client, params):
    response = client.get("/business/orders", params=params)

    assert response.status_code == 422


# =========================
# 6. 超出页数返回空列表
# =========================

def test_page_beyond_range_returns_empty_list(client):
    create_order(client, order_id="ORDER-001")

    response = client.get("/business/orders", params={"page": 999})

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["items"] == []
    assert data["total"] == 1


# =========================
# 7. 订单详情成功
# =========================

def test_get_order_detail_success(client):
    create_order(
        client,
        order_id="ORDER-001",
        created_at="2026-11-04T10:00:00",
        status="已支付",
        customer_id="CUSTOMER-009",
        amount=123.45,
    )

    response = client.get("/business/orders/ORDER-001")

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True

    order = body["data"]
    assert order["order_id"] == "ORDER-001"
    assert order["customer_id"] == "CUSTOMER-009"
    assert order["status"] == "已支付"
    assert order["created_at"] == "2026-11-04T10:00:00"
    assert order["amount"] == 123.45
    assert order["paid_at"] == "2026-11-04T10:00:00"


# =========================
# 8. 订单详情不存在返回 404
# =========================

def test_get_order_detail_not_found(client):
    response = client.get("/business/orders/NOT-EXIST")

    assert response.status_code == 404
    detail = response.json()["detail"]
    assert detail["type"] == "order_not_found"
    assert detail["message"] == "订单不存在"


# =========================
# 9. 支付记录查询
# =========================

def test_order_payments_query(client):
    create_order(client, order_id="ORDER-001")
    create_payment(
        client,
        payment_id="PAY-001",
        paid_at="2026-11-04T10:05:00",
        method="支付宝",
    )
    create_payment(
        client,
        payment_id="PAY-002",
        paid_at="2026-11-04T10:06:00",
        method="微信支付",
    )

    response = client.get("/business/orders/ORDER-001/payments")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["total"] == 2
    assert [item["payment_id"] for item in data["items"]] == [
        "PAY-002",
        "PAY-001",
    ]
    assert data["items"][0]["method"] == "微信支付"


# =========================
# 10. 存在订单但没有支付记录
# =========================

def test_order_exists_without_payments(client):
    create_order(client, order_id="ORDER-001")

    response = client.get("/business/orders/ORDER-001/payments")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["items"] == []
    assert data["total"] == 0


# =========================
# 11. 不存在订单查询支付记录返回 404
# =========================

def test_payments_for_missing_order_returns_404(client):
    response = client.get("/business/orders/NOT-EXIST/payments")

    assert response.status_code == 404
    assert response.json()["detail"]["type"] == "order_not_found"


# =========================
# 12. 物流查询及布尔类型
# =========================

def test_order_shipments_boolean_type(client):
    create_order(client, order_id="ORDER-001")
    create_shipment(
        client,
        shipment_id="SHIP-001",
        at_hub=True,
        departed_hub=False,
    )
    create_shipment(
        client,
        shipment_id="SHIP-002",
        at_hub=False,
        departed_hub=True,
        t_last="2026-11-04T12:00:00",
    )

    response = client.get("/business/orders/ORDER-001/shipments")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["total"] == 2

    shipments = {item["shipment_id"]: item for item in data["items"]}

    assert shipments["SHIP-001"]["at_hub"] is True
    assert shipments["SHIP-001"]["departed_hub"] is False
    assert shipments["SHIP-002"]["at_hub"] is False
    assert shipments["SHIP-002"]["departed_hub"] is True

    for item in data["items"]:
        assert isinstance(item["at_hub"], bool)
        assert isinstance(item["departed_hub"], bool)


# =========================
# 13. 物流事件正序排序及布尔类型
# =========================

def test_shipment_events_ordered_ascending_with_boolean(client):
    create_order(client, order_id="ORDER-001")
    create_shipment(client, shipment_id="SHIP-001")

    create_shipment_event(
        client,
        event_id="EVENT-012",
        event_time="2026-11-04T12:00:00",
        is_real_advance=False,
        is_exception=True,
    )
    create_shipment_event(
        client,
        event_id="EVENT-010",
        event_time="2026-11-04T10:00:00",
        is_real_advance=True,
        is_exception=False,
    )
    create_shipment_event(
        client,
        event_id="EVENT-011",
        event_time="2026-11-04T11:00:00",
        is_real_advance=False,
        is_exception=False,
    )

    response = client.get("/business/shipments/SHIP-001/events")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["total"] == 3
    assert [item["event_id"] for item in data["items"]] == [
        "EVENT-010",
        "EVENT-011",
        "EVENT-012",
    ]
    assert [item["event_time"] for item in data["items"]] == [
        "2026-11-04T10:00:00",
        "2026-11-04T11:00:00",
        "2026-11-04T12:00:00",
    ]

    first_event = data["items"][0]
    assert first_event["is_real_advance"] is True
    assert first_event["is_exception"] is False

    last_event = data["items"][2]
    assert last_event["is_real_advance"] is False
    assert last_event["is_exception"] is True

    for item in data["items"]:
        assert isinstance(item["is_real_advance"], bool)
        assert isinstance(item["is_exception"], bool)


# =========================
# 14. 不存在的物流查询事件返回 404
# =========================

def test_events_for_missing_shipment_returns_404(client):
    response = client.get("/business/shipments/NOT-EXIST/events")

    assert response.status_code == 404
    detail = response.json()["detail"]
    assert detail["type"] == "shipment_not_found"
    assert detail["message"] == "物流记录不存在"


# =========================
# 15. 售后和工单查询
# =========================

def test_order_aftersales_and_workorders(client):
    create_order(client, order_id="ORDER-001")
    create_aftersale(
        client,
        aftersale_id="AFTER-001",
        aftersale_type="退款",
        status="待处理",
    )
    create_workorder(
        client,
        workorder_id="WORK-001",
        workorder_type="订单异常",
        status="处理中",
        assigned_to="员工A",
    )

    aftersales = client.get("/business/orders/ORDER-001/aftersales")

    assert aftersales.status_code == 200
    aftersale_data = aftersales.json()["data"]
    assert aftersale_data["total"] == 1
    assert aftersale_data["items"][0]["aftersale_id"] == "AFTER-001"
    assert aftersale_data["items"][0]["type"] == "退款"
    assert aftersale_data["items"][0]["status"] == "待处理"

    workorders = client.get("/business/orders/ORDER-001/workorders")

    assert workorders.status_code == 200
    workorder_data = workorders.json()["data"]
    assert workorder_data["total"] == 1
    assert workorder_data["items"][0]["workorder_id"] == "WORK-001"
    assert workorder_data["items"][0]["type"] == "订单异常"
    assert workorder_data["items"][0]["status"] == "处理中"
    assert workorder_data["items"][0]["assigned_to"] == "员工A"

    # 父资源不存在时同样返回 404
    assert (
        client.get("/business/orders/NOT-EXIST/aftersales").status_code == 404
    )
    assert (
        client.get("/business/orders/NOT-EXIST/workorders").status_code == 404
    )


# =========================
# 16. 查询前后业务数据保持一致
# =========================

TABLES = (
    "orders",
    "payments",
    "shipments",
    "shipment_events",
    "aftersales",
    "workorders",
)


def snapshot_business_data():
    """读取六张表的全部行，用于比对我是否被查询接口改动。"""
    connection = business_db.get_connection()

    try:
        snapshot = {}

        for table in TABLES:
            rows = connection.execute(
                "SELECT * FROM " + table + " ORDER BY 1"
            ).fetchall()
            snapshot[table] = [tuple(row) for row in rows]

        return snapshot

    finally:
        connection.close()


def test_queries_do_not_modify_business_data(client):
    create_order(client, order_id="ORDER-001")
    create_payment(client, payment_id="PAY-001")
    create_shipment(client, shipment_id="SHIP-001")
    create_shipment_event(client, event_id="EVENT-001")
    create_aftersale(client, aftersale_id="AFTER-001")
    create_workorder(client, workorder_id="WORK-001")

    before = snapshot_business_data()

    assert client.get("/business/orders").status_code == 200
    assert client.get("/business/orders/ORDER-001").status_code == 200
    assert client.get("/business/orders/ORDER-001/payments").status_code == 200
    assert client.get("/business/orders/ORDER-001/shipments").status_code == 200
    assert client.get("/business/shipments/SHIP-001/events").status_code == 200
    assert client.get("/business/orders/ORDER-001/aftersales").status_code == 200
    assert client.get("/business/orders/ORDER-001/workorders").status_code == 200

    after = snapshot_business_data()

    assert before == after

    for table in TABLES:
        assert len(before[table]) == 1


# =========================
# 17. 索引初始化与重复初始化
# =========================

EXPECTED_INDEXES = {
    "idx_payments_order_id": "payments",
    "idx_shipments_order_id": "shipments",
    "idx_shipment_events_shipment_id": "shipment_events",
    "idx_aftersales_order_id": "aftersales",
    "idx_workorders_order_id": "workorders",
}


def test_index_initialization_is_idempotent(client):
    # fixture 已执行过一次初始化，这里再连续执行两次
    business_db.init_database()
    business_db.init_database()

    connection = business_db.get_connection()

    try:
        rows = connection.execute(
            "SELECT name, tbl_name FROM sqlite_master "
            "WHERE type = 'index' AND name LIKE 'idx_%'"
        ).fetchall()

        actual = {row["name"]: row["tbl_name"] for row in rows}

    finally:
        connection.close()

    assert actual == EXPECTED_INDEXES


# =========================
# 18. 数据库连接在异常情况下也会关闭
# =========================

class CloseTracker:
    """包装数据库连接，记录 close() 是否被调用。"""

    def __init__(self, connection):
        self._connection = connection
        self.closed = False

    def __getattr__(self, name):
        return getattr(self._connection, name)

    def close(self):
        self.closed = True
        self._connection.close()


def test_connection_closed_when_query_fails(temp_database, monkeypatch):
    # 本用例覆盖 _read_page 的异常关闭路径。
    # temp_database 已把业务库路径锁定在 tmp_path 内，构成第二道防线。
    # 构造一个没有任何表的空数据库，使 SELECT 必然抛异常
    empty_path = temp_database.parent / "empty.db"
    raw_connection = sqlite3.connect(empty_path)
    raw_connection.row_factory = sqlite3.Row

    tracker = CloseTracker(raw_connection)
    monkeypatch.setattr(business_queries, "get_connection", lambda: tracker)

    with pytest.raises(sqlite3.OperationalError):
        business_queries.list_orders()

    assert tracker.closed is True


# =========================
# 19. 关闭测试辅助
# =========================

def build_empty_database_tracker(directory):
    """建立一个没有任何表的空库连接，用于制造查询异常。"""
    connection = sqlite3.connect(directory / "empty.db")
    connection.row_factory = sqlite3.Row
    return CloseTracker(connection)


def build_tracked_query_connection():
    """用当前已锁定到 tmp_path 的路径建立可跟踪连接。"""
    return CloseTracker(business_db.get_connection())


def install_tracked_connection(monkeypatch, tracker_list):
    """把查询层取连接函数替换为"真实连接 + 关闭跟踪"的工厂。"""

    def factory():
        tracker = CloseTracker(business_db.get_connection())
        tracker_list.append(tracker)
        return tracker

    monkeypatch.setattr(business_queries, "get_connection", factory)


# =========================
# 20. 只读保护生效（Task E）
# =========================

def test_query_connection_rejects_write(temp_database):
    """开启只读保护后，查询连接上的写入必须被拒绝且不落库。"""
    connection = business_db.get_connection()

    try:
        business_queries._apply_read_only_protection(connection)

        with pytest.raises(sqlite3.OperationalError):
            connection.execute(
                "INSERT INTO orders ("
                "order_id, customer_id, status, created_at, amount"
                ") VALUES (?, ?, ?, ?, ?)",
                (
                    "ORDER-READONLY",
                    "CUSTOMER-READONLY",
                    "待支付",
                    "2026-11-04T10:00:00",
                    1.0,
                ),
            )

    finally:
        connection.close()

    verify_connection = business_db.get_connection()

    try:
        total = verify_connection.execute(
            "SELECT COUNT(*) FROM orders"
        ).fetchone()[0]

    finally:
        verify_connection.close()

    assert total == 0


# =========================
# 21. PRAGMA 失败时连接关闭（Task B3）
# =========================

ENTRY_POINTS = [
    pytest.param(
        lambda: business_queries.list_orders(),
        id="read_page",
    ),
    pytest.param(
        lambda: business_queries.list_order_payments("ORDER-001"),
        id="read_children",
    ),
    pytest.param(
        lambda: business_queries.get_order("ORDER-001"),
        id="get_order",
    ),
]


@pytest.mark.parametrize("invoke", ENTRY_POINTS)
def test_connection_closed_when_read_only_pragma_fails(
    temp_database, monkeypatch, invoke
):
    """只读保护自身失败时，连接仍须关闭，且原异常必须向上传播。"""
    tracker = build_tracked_query_connection()

    # 证明本次连接指向临时库，而非正式 data/business.db
    assert business_db.DATABASE_PATH == temp_database
    assert business_db.DATABASE_PATH.name == "business_get_test.db"

    def failing_protection(connection):
        raise sqlite3.OperationalError("simulated read-only pragma failure")

    monkeypatch.setattr(business_queries, "get_connection", lambda: tracker)
    monkeypatch.setattr(
        business_queries,
        "_apply_read_only_protection",
        failing_protection,
    )

    with pytest.raises(sqlite3.OperationalError) as error_info:
        invoke()

    assert "simulated read-only pragma failure" in str(error_info.value)
    assert tracker.closed is True


# =========================
# 22. _read_children / get_order 连接关闭（Task B1、B2）
# =========================

def test_read_children_connection_closed_on_success(client, monkeypatch):
    create_order(client, order_id="ORDER-001")

    trackers = []
    install_tracked_connection(monkeypatch, trackers)

    result = business_queries.list_order_payments("ORDER-001")

    assert result is not None
    assert result["total"] == 0
    assert len(trackers) == 1
    assert trackers[0].closed is True


def test_read_children_connection_closed_on_error(temp_database, monkeypatch):
    tracker = build_empty_database_tracker(temp_database.parent)
    monkeypatch.setattr(business_queries, "get_connection", lambda: tracker)

    with pytest.raises(sqlite3.OperationalError):
        business_queries.list_order_payments("ORDER-001")

    assert tracker.closed is True


def test_get_order_connection_closed_on_success(client, monkeypatch):
    create_order(client, order_id="ORDER-001")

    trackers = []
    install_tracked_connection(monkeypatch, trackers)

    order = business_queries.get_order("ORDER-001")

    assert order is not None
    assert order["order_id"] == "ORDER-001"
    assert len(trackers) == 1
    assert trackers[0].closed is True


def test_get_order_connection_closed_on_error(temp_database, monkeypatch):
    tracker = build_empty_database_tracker(temp_database.parent)
    monkeypatch.setattr(business_queries, "get_connection", lambda: tracker)

    with pytest.raises(sqlite3.OperationalError):
        business_queries.get_order("ORDER-001")

    assert tracker.closed is True


# =========================
# 23. 接口测试补齐（Task D）
# =========================

def test_shipments_for_missing_order_returns_404(client):
    response = client.get("/business/orders/NOT-EXIST/shipments")

    assert response.status_code == 404
    detail = response.json()["detail"]
    assert detail["type"] == "order_not_found"
    assert detail["message"] == "订单不存在"


def test_order_customer_id_filter(client):
    create_order(client, order_id="ORDER-001", customer_id="CUSTOMER-001")
    create_order(client, order_id="ORDER-002", customer_id="CUSTOMER-002")
    create_order(client, order_id="ORDER-003", customer_id="CUSTOMER-001")

    response = client.get(
        "/business/orders", params={"customer_id": "CUSTOMER-001"}
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["total"] == 2
    assert {item["order_id"] for item in data["items"]} == {
        "ORDER-001",
        "ORDER-003",
    }
    assert all(
        item["customer_id"] == "CUSTOMER-001" for item in data["items"]
    )


def test_order_created_from_filter(client):
    create_order(client, order_id="ORDER-001", created_at="2026-11-04T10:00:00")
    create_order(client, order_id="ORDER-002", created_at="2026-11-04T12:00:00")

    response = client.get(
        "/business/orders",
        params={"created_from": "2026-11-04T11:00:00"},
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["total"] == 1
    assert [item["order_id"] for item in data["items"]] == ["ORDER-002"]


def test_order_created_to_filter(client):
    create_order(client, order_id="ORDER-001", created_at="2026-11-04T10:00:00")
    create_order(client, order_id="ORDER-002", created_at="2026-11-04T12:00:00")
    create_order(client, order_id="ORDER-003", created_at="2026-11-04T11:00:00")

    response = client.get(
        "/business/orders",
        params={"created_to": "2026-11-04T11:00:00"},
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["total"] == 2
    assert {item["order_id"] for item in data["items"]} == {
        "ORDER-001",
        "ORDER-003",
    }


def test_order_payments_pagination(client):
    create_order(client, order_id="ORDER-001")
    create_payment(client, payment_id="PAY-001", paid_at="2026-11-04T10:01:00")
    create_payment(client, payment_id="PAY-002", paid_at="2026-11-04T10:02:00")
    create_payment(client, payment_id="PAY-003", paid_at="2026-11-04T10:03:00")

    first_page = client.get(
        "/business/orders/ORDER-001/payments",
        params={"page": 1, "page_size": 2},
    )

    assert first_page.status_code == 200
    first_data = first_page.json()["data"]
    assert first_data["total"] == 3
    assert first_data["page"] == 1
    assert first_data["page_size"] == 2
    assert [item["payment_id"] for item in first_data["items"]] == [
        "PAY-003",
        "PAY-002",
    ]

    second_page = client.get(
        "/business/orders/ORDER-001/payments",
        params={"page": 2, "page_size": 2},
    )

    assert second_page.status_code == 200
    second_data = second_page.json()["data"]
    assert second_data["total"] == 3
    assert [item["payment_id"] for item in second_data["items"]] == ["PAY-001"]
