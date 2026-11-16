# -*- coding: utf-8 -*-
"""R3-AGENT-ORDER-03A：订单查询「有数据」场景隔离测试。

目的：在**隔离临时数据库**中构造虚构订单，补齐 02E 无法验证的有数据场景：
默认字段投影、可选项开关、分页与 has_more、非法参数、中文状态、空库结构。

数据隔离（硬要求，沿用 test_agent_order_tool.py 已验证的范式）：
1. business/db.py 使用绝对路径（__file__ 推导），monkeypatch.chdir() 无效，
   必须替换 `business_db.DATA_DIR` 与 `business_db.DATABASE_PATH`。
2. 所有写入只发生在 pytest tmp_path 内的临时库，绝不触达正式 data/business.db。
3. 会话级守卫：任何 socket.connect / connect_ex 一律拒绝，确保零网络请求。
4. 会话级守卫：测试前后比对正式 business.db 的 (size, mtime_ns, sha256) 与 sidecar，
   不一致即判定测试污染正式库。

本文件只读业务源码，不修改任何既有文件。
"""

import hashlib
import socket
import sys
from pathlib import Path
from unittest import mock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from business import db as business_db                 # noqa: E402
from business.validators import ORDER_STATUSES         # noqa: E402
from tools import (                                    # noqa: E402
    ADMIN_ORDER_DEFAULT_PAGE_SIZE,
    ADMIN_ORDER_MAX_PAGE_SIZE,
    ADMIN_ORDER_SCOPE,
    ORDER_TOOL_DEFAULT_FIELDS,
    ORDER_TOOL_OPTIONAL_FIELDS,
    QUERY_ORDERS_FAILURE_PREFIX,
    make_query_orders_tool,
)


# =========================================================
# 正式库守卫常量
# =========================================================

FORMAL_DB = PROJECT_ROOT / "data" / "business.db"
FORMAL_SIDECARS = tuple(
    FORMAL_DB.parent / (FORMAL_DB.name + suffix)
    for suffix in ("-journal", "-wal", "-shm")
)


def _file_fingerprint(path):
    """返回 (字节大小, mtime_ns, sha256)；不打开 SQLite。"""
    stat = path.stat()
    return (stat.st_size, stat.st_mtime_ns,
            hashlib.sha256(path.read_bytes()).hexdigest())


# =========================================================
# 会话级守卫
# =========================================================

@pytest.fixture(scope="session", autouse=True)
def network_guard():
    """阻断一切出站连接：任何真实模型/HTTP 调用都会失败，从而零请求。"""

    def _refuse(*args, **kwargs):
        raise AssertionError("测试期间禁止任何网络连接")

    with mock.patch.object(socket.socket, "connect", _refuse), \
            mock.patch.object(socket.socket, "connect_ex", _refuse):
        yield


@pytest.fixture(scope="session", autouse=True)
def formal_database_guard():
    """测试前后比对正式 business.db，任何变化都判失败。"""
    before = _file_fingerprint(FORMAL_DB)
    before_sidecars = tuple(p.exists() for p in FORMAL_SIDECARS)

    yield

    after = _file_fingerprint(FORMAL_DB)
    after_sidecars = tuple(p.exists() for p in FORMAL_SIDECARS)

    assert after == before, "正式 business.db 在测试期间被修改"
    assert after_sidecars == before_sidecars, "正式库 sidecar 状态发生变化"


# =========================================================
# 虚构订单数据（全部虚构，与正式库无关）
# =========================================================

# 字段顺序：order_id, customer_id, status, created_at, amount, paid_at
FICTIONAL_ORDERS = (
    ("ORD-TEST-0001", "CUST-FAKE-01", "待支付", "2026-01-01T09:00:00", 100.10, None),
    ("ORD-TEST-0002", "CUST-FAKE-01", "已支付", "2026-01-02T09:00:00", 200.20, "2026-01-02T10:00:00"),
    ("ORD-TEST-0003", "CUST-FAKE-02", "待发货", "2026-01-03T09:00:00", 300.30, "2026-01-03T10:00:00"),
    ("ORD-TEST-0004", "CUST-FAKE-02", "已发货", "2026-01-04T09:00:00", 400.40, "2026-01-04T10:00:00"),
    ("ORD-TEST-0005", "CUST-FAKE-03", "已完成", "2026-01-05T09:00:00", 500.50, "2026-01-05T10:00:00"),
    ("ORD-TEST-0006", "CUST-FAKE-03", "已取消", "2026-01-06T09:00:00", 600.60, None),
    ("ORD-TEST-0007", "CUST-FAKE-01", "已完成", "2026-01-07T09:00:00", 700.70, "2026-01-07T10:00:00"),
)

# 由 created_at 倒序推导的期望顺序（与 queries.py 的 ORDER BY 一致）
EXPECTED_ORDER_BY_CREATED_DESC = tuple(
    row[0] for row in sorted(FICTIONAL_ORDERS, key=lambda r: r[3], reverse=True)
)

DEFAULT_FIELD_SET = set(ORDER_TOOL_DEFAULT_FIELDS)

PRIVACY_FIELDS = {
    "name", "customer_name", "phone", "mobile", "tel", "address",
    "receiver", "receiver_phone", "email", "id_card",
}


# =========================================================
# Fixtures
# =========================================================

@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """把业务数据库锁定在 tmp_path 内，并写入 7 条虚构订单。"""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    db_path = data_dir / "business_projection_test.db"

    monkeypatch.setattr(business_db, "DATA_DIR", data_dir)
    monkeypatch.setattr(business_db, "DATABASE_PATH", db_path)
    business_db.init_database()

    assert business_db.DATABASE_PATH.is_relative_to(tmp_path)
    assert business_db.DATABASE_PATH != FORMAL_DB

    connection = business_db.get_connection()
    try:
        connection.executemany(
            "INSERT INTO orders "
            "(order_id, customer_id, status, created_at, amount, paid_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            FICTIONAL_ORDERS,
        )
        connection.commit()
    finally:
        connection.close()

    return db_path


@pytest.fixture
def empty_db(tmp_path, monkeypatch):
    """独立空库：只建结构，不插数据，用于空结果结构验证。"""
    data_dir = tmp_path / "empty_data"
    data_dir.mkdir()
    db_path = data_dir / "business_empty_test.db"

    monkeypatch.setattr(business_db, "DATA_DIR", data_dir)
    monkeypatch.setattr(business_db, "DATABASE_PATH", db_path)
    business_db.init_database()

    assert business_db.DATABASE_PATH.is_relative_to(tmp_path)
    assert business_db.DATABASE_PATH != FORMAL_DB

    return db_path


@pytest.fixture
def tool():
    """被测对象：query_orders 工具原始函数（未经 Tool Registry 包装）。"""
    return make_query_orders_tool()


# =========================================================
# T00：防护自检（证明守卫确实生效）
# =========================================================

def test_T00_network_guard_active():
    """网络守卫必须真的拦住连接，否则「零请求」无从证明。"""
    probe = socket.socket()
    try:
        with pytest.raises(AssertionError):
            probe.connect(("127.0.0.1", 9))
    finally:
        probe.close()


# =========================================================
# T01：默认字段投影
# =========================================================

def test_T01_default_field_projection(temp_db, tool):
    """默认只返回 4 个白名单字段，不多不少。"""
    result = tool()

    assert result["count"] == len(FICTIONAL_ORDERS) == 7

    for item in result["items"]:
        assert set(item.keys()) == DEFAULT_FIELD_SET
        assert set(item.keys()) == {"order_id", "status", "created_at", "amount"}
        assert len(item) == 4


# =========================================================
# T02：可选字段默认关闭
# =========================================================

def test_T02_optional_fields_off_by_default(temp_db, tool):
    """不传 include_* 时，customer_id / paid_at 与隐私字段都不得出现。"""
    result = tool()

    for item in result["items"]:
        assert "customer_id" not in item
        assert "paid_at" not in item
        assert not (set(item.keys()) & PRIVACY_FIELDS)
        for field in ORDER_TOOL_OPTIONAL_FIELDS:
            assert field not in item


# =========================================================
# T03：可选字段显式启用
# =========================================================

def test_T03_include_customer_id_only(temp_db, tool):
    """只开 include_customer_id：多出 customer_id，仍无 paid_at。"""
    result = tool(include_customer_id=True)

    expected = DEFAULT_FIELD_SET | {"customer_id"}
    for item in result["items"]:
        assert set(item.keys()) == expected
        assert item["customer_id"].startswith("CUST-FAKE-")


def test_T03_include_paid_at_only(temp_db, tool):
    """只开 include_paid_at：多出 paid_at，仍无 customer_id。"""
    result = tool(include_paid_at=True)

    expected = DEFAULT_FIELD_SET | {"paid_at"}
    for item in result["items"]:
        assert set(item.keys()) == expected
        assert "customer_id" not in item


def test_T03_both_optional_enabled(temp_db, tool):
    """两个开关都开：6 个字段，无隐私字段。"""
    result = tool(include_customer_id=True, include_paid_at=True)

    expected = DEFAULT_FIELD_SET | set(ORDER_TOOL_OPTIONAL_FIELDS)
    for item in result["items"]:
        assert set(item.keys()) == expected
        assert len(item) == 6
        assert not (set(item.keys()) & PRIVACY_FIELDS)


def test_T03_optional_values_match_seed_data(temp_db, tool):
    """可选字段的取值必须来自本库虚构数据，不是默认值或占位符。"""
    result = tool(include_customer_id=True, include_paid_at=True)
    by_id = {item["order_id"]: item for item in result["items"]}

    for order_id, customer_id, _status, _created, _amount, paid_at in FICTIONAL_ORDERS:
        assert by_id[order_id]["customer_id"] == customer_id
        assert by_id[order_id]["paid_at"] == paid_at


# =========================================================
# T04：默认分页
# =========================================================

def test_T04_default_page_size(temp_db, tool):
    """不传 page_size 时取实现默认值，并核对设计常量。"""
    result = tool()

    assert ADMIN_ORDER_DEFAULT_PAGE_SIZE == 20
    assert result["page_size"] == 20
    assert result["page_size"] == ADMIN_ORDER_DEFAULT_PAGE_SIZE
    assert result["page"] == 1
    assert result["count"] == 7
    assert result["total"] == 7
    assert result["has_more"] is False


# =========================================================
# T05：最大分页限制
# =========================================================

def test_T05_page_size_within_range(temp_db, tool):
    """正常值不做截断。"""
    result = tool(page_size=1)
    assert result["page_size"] == 1
    assert result["count"] == 1
    assert result["has_more"] is True

    result = tool(page_size=5)
    assert result["page_size"] == 5
    assert result["count"] == 5
    assert result["has_more"] is True


def test_T05_page_size_equals_max(temp_db, tool):
    """page_size=20（等于上限）保持 20。"""
    assert ADMIN_ORDER_MAX_PAGE_SIZE == 20
    result = tool(page_size=20)
    assert result["page_size"] == 20
    assert result["count"] == 7
    assert result["has_more"] is False


@pytest.mark.parametrize("requested", [21, 50, 100, 1000])
def test_T05_page_size_above_max_is_capped(temp_db, tool, requested):
    """超过上限时按上限 20 处理，且元数据与返回数量一致。"""
    result = tool(page_size=requested)

    assert result["page_size"] == ADMIN_ORDER_MAX_PAGE_SIZE == 20
    assert result["count"] == 7
    assert result["total"] == 7
    assert result["has_more"] is False


# =========================================================
# T06：分页正确性
# =========================================================

def test_T06_pagination_order_follows_real_sort(temp_db, tool):
    """按实现真实排序（created_at DESC, order_id ASC）切页。"""
    result = tool(page=1, page_size=3)
    actual = tuple(item["order_id"] for item in result["items"])
    assert actual == EXPECTED_ORDER_BY_CREATED_DESC[:3]


def test_T06_pages_have_no_overlap_and_no_gap(temp_db, tool):
    """第一页到最后一页：无重复、无遗漏、元数据自洽。"""
    page_size = 3
    total_pages = 3  # 7 条 / 每页 3 条 → 3 页

    collected = []
    pages_meta = []

    for page in range(1, total_pages + 1):
        result = tool(page=page, page_size=page_size)
        pages_meta.append(
            (result["page"], result["count"], result["total"], result["has_more"])
        )
        collected.extend(item["order_id"] for item in result["items"])

    # 元数据：当前页正确、总数正确、返回数量正确
    assert pages_meta == [
        (1, 3, 7, True),
        (2, 3, 7, True),
        (3, 1, 7, False),
    ]

    # 无重复
    assert len(collected) == len(set(collected)) == 7
    # 无遗漏，且整体顺序等于真实排序
    assert tuple(collected) == EXPECTED_ORDER_BY_CREATED_DESC


def test_T06_page_beyond_last_is_empty(temp_db, tool):
    """越界页：返回空 items，但 total / page_size / has_more 仍然自洽。"""
    result = tool(page=4, page_size=3)

    assert result["items"] == []
    assert result["count"] == 0
    assert result["total"] == 7
    assert result["page"] == 4
    assert result["page_size"] == 3
    assert result["has_more"] is False


# =========================================================
# T07：非法分页与参数
# =========================================================

@pytest.mark.parametrize(
    "kwargs, needle",
    [
        ({"page": 0}, "page 必须是正整数"),
        ({"page": -3}, "page 必须是正整数"),
        ({"page": "1"}, "page 必须是正整数"),
        ({"page": 1.0}, "page 必须是正整数"),
        ({"page": True}, "page 必须是正整数"),
        ({"page_size": 0}, "page_size 必须是正整数"),
        ({"page_size": -1}, "page_size 必须是正整数"),
        ({"page_size": "5"}, "page_size 必须是正整数"),
        ({"page_size": False}, "page_size 必须是正整数"),
        ({"order_id": 123}, "order_id 必须是字符串"),
        ({"customer_id": 123}, "customer_id 必须是字符串"),
        ({"status": 123}, "status 必须是字符串"),
        ({"status": "未定义状态"}, "status 不在允许范围内"),
        ({"include_customer_id": "true"}, "include_customer_id 必须是布尔值"),
        ({"include_paid_at": 1}, "include_paid_at 必须是布尔值"),
    ],
)
def test_T07_illegal_arguments_rejected(temp_db, tool, kwargs, needle):
    """非法参数一律 ValueError，且带统一安全前缀。"""
    with pytest.raises(ValueError) as excinfo:
        tool(**kwargs)

    message = str(excinfo.value)
    assert message.startswith(QUERY_ORDERS_FAILURE_PREFIX)
    assert needle in message


def test_T07_valid_edge_arguments_accepted(temp_db, tool):
    """边界合法值不应被误判。"""
    result = tool(page=1, page_size=1)
    assert result["page"] == 1
    assert result["page_size"] == 1

    result = tool(include_customer_id=False, include_paid_at=False)
    assert set(result["items"][0].keys()) == DEFAULT_FIELD_SET


# =========================================================
# T08：中文订单状态映射
# =========================================================

def test_T08_status_enum_is_exactly_six_chinese_values():
    """状态白名单就是 6 个中文值，不存在英文/底层编码。"""
    assert ORDER_STATUSES == {
        "待支付", "已支付", "待发货", "已发货", "已完成", "已取消",
    }


@pytest.mark.parametrize("status", sorted(ORDER_STATUSES))
def test_T08_status_filter_round_trip(temp_db, tool, status):
    """输入中文状态 → 命中对应订单 → 返回的 status 仍是该中文值。"""
    result = tool(status=status)

    assert result["filters"]["status"] == status

    expected_ids = {row[0] for row in FICTIONAL_ORDERS if row[2] == status}
    assert expected_ids, f"测试数据未覆盖状态：{status}"
    assert {item["order_id"] for item in result["items"]} == expected_ids

    for item in result["items"]:
        assert item["status"] == status
        assert item["status"] in ORDER_STATUSES


def test_T08_all_returned_statuses_stay_in_enum(temp_db, tool):
    """全量返回中不出现白名单之外的状态值。"""
    result = tool()
    statuses = {item["status"] for item in result["items"]}
    assert statuses == ORDER_STATUSES
    assert statuses <= ORDER_STATUSES


# =========================================================
# T09：空数据场景
# =========================================================

def test_T09_empty_database_structure(empty_db, tool):
    """空库：total/count 为 0，items 为空，has_more 为 False。"""
    assert empty_db.exists()

    result = tool()

    assert result["scope"] == ADMIN_ORDER_SCOPE
    assert result["total"] == 0
    assert result["count"] == 0
    assert result["items"] == []
    assert result["has_more"] is False
    assert result["page"] == 1
    assert result["page_size"] == ADMIN_ORDER_DEFAULT_PAGE_SIZE


def test_T09_empty_database_with_filters(empty_db, tool):
    """空库 + 状态筛选：结构保持稳定，不抛异常。"""
    result = tool(status="已完成", page=1, page_size=3)

    assert result["total"] == 0
    assert result["count"] == 0
    assert result["items"] == []
    assert result["has_more"] is False
    assert result["filters"]["status"] == "已完成"


# =========================================================
# T10：正式数据库完整性
# =========================================================

def test_T10_tests_run_against_temporary_database(temp_db, tmp_path):
    """被测库必须位于 tmp_path，且绝不等于正式库路径。"""
    assert temp_db.is_relative_to(tmp_path)
    assert temp_db != FORMAL_DB
    assert business_db.DATABASE_PATH == temp_db
    assert business_db.DATABASE_PATH.is_relative_to(tmp_path)


def test_T10_formal_database_still_present_unchanged_contract():
    """正式库仍存在且可读；会话结束后的完整比对由 formal_database_guard 负责。"""
    assert FORMAL_DB.exists()
    size, mtime_ns, digest = _file_fingerprint(FORMAL_DB)
    assert size > 0
    assert mtime_ns > 0
    assert len(digest) == 64
