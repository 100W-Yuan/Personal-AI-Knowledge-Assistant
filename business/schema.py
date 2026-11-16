
"""
企业业务数据库结构定义。

负责：
1. 创建订单表
2. 创建支付表
3. 创建物流表
4. 创建物流事件表
5. 创建售后表
6. 创建客服工单表

注意：
这里只定义数据库结构，不存放员工录入的业务数据。
业务数据由网页提交，经后端校验后写入数据库。
"""

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

-- 订单表
CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    amount REAL NOT NULL CHECK (amount >= 0),
    paid_at TEXT
);

-- 支付表
CREATE TABLE IF NOT EXISTS payments (
    payment_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL,
    paid_at TEXT,
    method TEXT NOT NULL,
    amount REAL NOT NULL CHECK (amount >= 0),
    status TEXT NOT NULL,
    FOREIGN KEY (order_id) REFERENCES orders(order_id)
);

-- 物流表
CREATE TABLE IF NOT EXISTS shipments (
    shipment_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL,
    carrier TEXT NOT NULL,
    status TEXT NOT NULL,
    at_hub INTEGER NOT NULL DEFAULT 0
        CHECK (at_hub IN (0, 1)),
    departed_hub INTEGER NOT NULL DEFAULT 0
        CHECK (departed_hub IN (0, 1)),
    t_last TEXT,
    FOREIGN KEY (order_id) REFERENCES orders(order_id)
);

-- 物流事件表
CREATE TABLE IF NOT EXISTS shipment_events (
    event_id TEXT PRIMARY KEY,
    shipment_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    event_time TEXT NOT NULL,
    detail TEXT,
    is_real_advance INTEGER NOT NULL DEFAULT 0
        CHECK (is_real_advance IN (0, 1)),
    is_exception INTEGER NOT NULL DEFAULT 0
        CHECK (is_exception IN (0, 1)),
    FOREIGN KEY (shipment_id) REFERENCES shipments(shipment_id)
);

-- 售后表
CREATE TABLE IF NOT EXISTS aftersales (
    aftersale_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL,
    type TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (order_id) REFERENCES orders(order_id)
);

-- 客服工单表
CREATE TABLE IF NOT EXISTS workorders (
    workorder_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL,
    type TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    assigned_to TEXT,
    FOREIGN KEY (order_id) REFERENCES orders(order_id)
);

-- 外键查询索引
-- 说明：仅新增索引，不修改任何表结构、字段、外键与约束。
CREATE INDEX IF NOT EXISTS idx_payments_order_id
    ON payments(order_id);

CREATE INDEX IF NOT EXISTS idx_shipments_order_id
    ON shipments(order_id);

CREATE INDEX IF NOT EXISTS idx_shipment_events_shipment_id
    ON shipment_events(shipment_id);

CREATE INDEX IF NOT EXISTS idx_aftersales_order_id
    ON aftersales(order_id);

CREATE INDEX IF NOT EXISTS idx_workorders_order_id
    ON workorders(order_id);
"""
