# -*- coding: utf-8 -*-
"""N1 缺陷修正回归：classify_knowledge_retrieval 的 count 类型校验。

覆盖场景（8 项）
----------------
count = 0 / 3            → 有效计数，走原有四状态判定（no_hit / hit）；
count = True / False     → bool 是 int 子类，必须判为无效；
count = -1               → 负整数，必须判为无效；
count = 2.7              → 浮点，必须判为无效；
count = "3" / None       → 直接比较会抛 TypeError，必须判为无效且不抛异常。

无效计数一律返回 degraded + partial=True，reason 说明计数无效：
既不能当 hit（可能其实没命中），也不能当 no_hit（可能其实命中了）。

隔离说明
--------
只导入 tools（模块级仅依赖 datetime），不导入 main / agent，
不构造 Agent、不访问网络、不读写知识库与任何数据库，
可在无 .env、无模型、无服务的环境下重复运行。
"""

import pytest

import tools


class _StubKB:
    """最小 KnowledgeBase 替身：只暴露诊断属性，不做任何 IO。"""

    def __init__(self, skipped_count=0, init_error_kind=None):
        self.skipped_count = skipped_count
        self.init_error_kind = init_error_kind


# ============================================================
# 有效计数：必须保持原有语义，不被本次校验误伤
# ============================================================


@pytest.mark.parametrize(
    "count, expected_status",
    [
        (0, "no_hit"),
        (3, "hit"),
    ],
    ids=["count-0", "count-3"],
)
def test_valid_count_keeps_original_semantics(count, expected_status):
    result = tools.classify_knowledge_retrieval(_StubKB(), count=count)

    assert result["status"] == expected_status
    assert result["partial"] is False
    assert result["skipped_files"] == 0
    assert result["failed_searches"] == 0
    # 有效计数不应出现「计数无效」的措辞
    assert "计数无效" not in result["reason"]


# ============================================================
# 无效计数：必须稳定降级、不得抛异常、不得伪装成正常状态
# ============================================================


@pytest.mark.parametrize(
    "count",
    [
        pytest.param(True, id="count-True"),
        pytest.param(False, id="count-False"),
        pytest.param(-1, id="count-negative-int"),
        pytest.param(2.7, id="count-float"),
        pytest.param("3", id="count-str"),
        pytest.param(None, id="count-None"),
    ],
)
def test_invalid_count_degrades_without_raising(count):
    # 关键：字符串/None 若直接参与 `count > 0` 比较会抛 TypeError，
    # 这里若抛出即视为缺陷复现。
    result = tools.classify_knowledge_retrieval(_StubKB(), count=count)

    assert result["status"] == "degraded"
    assert result["partial"] is True
    assert result["skipped_files"] == 0
    assert result["failed_searches"] == 0
    assert "计数无效" in result["reason"]
    # 不得伪装成「没命中」或「命中」
    assert result["status"] != "no_hit"
    assert result["status"] != "hit"
