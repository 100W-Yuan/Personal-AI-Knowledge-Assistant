

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import business.db as business_db
from business.api import router


@pytest.fixture
def client(tmp_path, monkeypatch):
    """为每个测试创建独立的临时业务数据库。"""
    database_dir = tmp_path / "data"
    database_dir.mkdir()

    database_path = database_dir / "business_test.db"

    monkeypatch.setattr(business_db, "DATA_DIR", database_dir)
    monkeypatch.setattr(business_db, "DATABASE_PATH", database_path)

    business_db.init_database()

    app = FastAPI()
    app.include_router(router)

    with TestClient(app) as test_client:
        yield test_client


def create_order(client, order_id="ORDER-001"):
    return client.post(
        "/business/orders",
        json={
            "order_id": order_id,
            "customer_id": "CUSTOMER-001",
            "status": "已支付",
            "created_at": "2026-11-04T10:00:00",
            "amount": 99.9,
            "paid_at": "2026-11-04T10:05:00",
        },
    )


def test_create_order_success(client):
    response = create_order(client)

    assert response.status_code == 201
    body = response.json()
    assert body["ok"] is True
    assert body["data"]["order_id"] == "ORDER-001"


def test_duplicate_order_rejected(client):
    first = create_order(client)
    second = create_order(client)

    assert first.status_code == 201
    assert second.status_code == 409
    assert second.json()["detail"]["type"] == "duplicate_record"


def test_invalid_order_data_rejected(client):
    response = client.post(
        "/business/orders",
        json={
            "order_id": "ORDER-002",
            "customer_id": "CUSTOMER-001",
            "status": "未知状态",
            "created_at": "2026-11-04T10:00:00",
            "amount": 99.9,
        },
    )

    assert response.status_code == 422


def test_extra_field_rejected(client):
    response = client.post(
        "/business/orders",
        json={
            "order_id": "ORDER-003",
            "customer_id": "CUSTOMER-001",
            "status": "已支付",
            "created_at": "2026-11-04T10:00:00",
            "amount": 99.9,
            "unexpected_field": "not allowed",
        },
    )

    assert response.status_code == 422


def test_create_payment_success(client):
    create_order(client)

    response = client.post(
        "/business/payments",
        json={
            "payment_id": "PAY-001",
            "order_id": "ORDER-001",
            "paid_at": "2026-11-04T10:05:00",
            "method": "支付宝",
            "amount": 99.9,
            "status": "支付成功",
        },
    )

    assert response.status_code == 201
    assert response.json()["data"]["payment_id"] == "PAY-001"


def test_payment_invalid_order_rejected(client):
    response = client.post(
        "/business/payments",
        json={
            "payment_id": "PAY-002",
            "order_id": "NOT-EXIST",
            "method": "支付宝",
            "amount": 99.9,
            "status": "支付成功",
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"]["type"] == "invalid_reference"


def test_create_shipment_success(client):
    create_order(client)

    response = client.post(
        "/business/shipments",
        json={
            "shipment_id": "SHIP-001",
            "order_id": "ORDER-001",
            "carrier": "顺丰",
            "status": "运输中",
            "at_hub": True,
            "departed_hub": False,
            "t_last": "2026-11-04T11:00:00",
        },
    )

    assert response.status_code == 201
    assert response.json()["data"]["shipment_id"] == "SHIP-001"


def test_shipment_boolean_must_be_json_boolean(client):
    create_order(client)

    response = client.post(
        "/business/shipments",
        json={
            "shipment_id": "SHIP-002",
            "order_id": "ORDER-001",
            "carrier": "顺丰",
            "status": "运输中",
            "at_hub": "true",
        },
    )

    assert response.status_code == 422


def test_create_shipment_event_success(client):
    create_order(client)

    client.post(
        "/business/shipments",
        json={
            "shipment_id": "SHIP-003",
            "order_id": "ORDER-001",
            "carrier": "顺丰",
            "status": "运输中",
        },
    )

    response = client.post(
        "/business/shipment-events",
        json={
            "event_id": "EVENT-001",
            "shipment_id": "SHIP-003",
            "event_type": "到达转运中心",
            "event_time": "2026-11-04T12:00:00",
            "detail": "包裹已到达",
            "is_real_advance": True,
            "is_exception": False,
        },
    )

    assert response.status_code == 201
    assert response.json()["data"]["event_id"] == "EVENT-001"


def test_create_aftersale_success(client):
    create_order(client)

    response = client.post(
        "/business/aftersales",
        json={
            "aftersale_id": "AFTER-001",
            "order_id": "ORDER-001",
            "type": "退款",
            "status": "待处理",
            "created_at": "2026-11-04T13:00:00",
        },
    )

    assert response.status_code == 201
    assert response.json()["data"]["aftersale_id"] == "AFTER-001"


def test_create_workorder_success(client):
    create_order(client)

    response = client.post(
        "/business/workorders",
        json={
            "workorder_id": "WORK-001",
            "order_id": "ORDER-001",
            "type": "订单异常",
            "status": "待处理",
            "created_at": "2026-11-04T14:00:00",
            "assigned_to": "员工A",
        },
    )

    assert response.status_code == 201
    assert response.json()["data"]["workorder_id"] == "WORK-001"


def test_workorder_invalid_order_rejected(client):
    response = client.post(
        "/business/workorders",
        json={
            "workorder_id": "WORK-002",
            "order_id": "NOT-EXIST",
            "type": "订单异常",
            "status": "待处理",
            "created_at": "2026-11-04T14:00:00",
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"]["type"] == "invalid_reference"
