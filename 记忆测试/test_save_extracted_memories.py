# -*- coding: utf-8 -*-
"""
Agent 长期记忆保存流程测试 —— save_extracted_memories()

覆盖范围：
    只测 Agent 层的「保存」流程（agent.py: save_extracted_memories），
    即「一份已经提取好的 extraction_result → 写入 SQLite」这一段。

不覆盖：
    - 不测 extract_memories()（那需要调用真实 DeepSeek API，本测试全程零网络请求）
    - 不重复 test_memory_save.py 已有的 Memory.save_memory 基础用例
    - 不测 messages 表、不测 Agent Loop / 工具调用

数据安全：
    每个用例使用 pytest 的 tmp_path（独立临时目录）+ monkeypatch.chdir，
    因此 Agent 内部的 Memory() 与显式构造的 Memory 都指向临时数据库，
    绝不可能碰项目正式的 memory.db。

运行：
    见文件末尾 __main__ 注释，或直接 python -m pytest。
"""

import sqlite3
import sys
from pathlib import Path

import pytest

# ---- 解决「测试子目录找不到项目根模块」的问题 ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent import Agent  # noqa: E402
from memory import Memory  # noqa: E402


# =====================================================================
# 辅助函数
# =====================================================================

DUMMY_API_KEY = "dummy-key-never-used"
DUMMY_BASE_URL = "https://example.invalid"
DUMMY_MODEL = "dummy-model-never-used"


def build_agent(tmp_path, monkeypatch):
    """
    构造一个可用的 Agent，但完全不产生网络请求。

    Agent.__init__ 中会用默认路径构造 Memory()（相对当前工作目录）。
    这里先 chdir 到临时目录，再用绝对路径显式替换 agent.memory，
    双重保证所有数据库操作都落在临时目录内。
    """
    monkeypatch.chdir(tmp_path)

    agent = Agent(
        api_key=DUMMY_API_KEY,
        base_url=DUMMY_BASE_URL,
        model=DUMMY_MODEL,
    )

    agent.memory = Memory(db_path=str(tmp_path / "test_memory.db"))
    return agent


def physical_rows(db_path):
    """
    直接用 sqlite3 读取 memories 表的物理内容。

    刻意绕开 Memory 的封装方法，用来证明数据真的落到了数据库里，
    而不只是返回对象或字典里带着这些字段。
    """
    connection = sqlite3.connect(str(db_path))
    try:
        cursor = connection.execute(
            """
            SELECT key, value, source, confidence, category, reason
            FROM memories
            ORDER BY id
            """
        )
        return cursor.fetchall()
    finally:
        connection.close()


def memory_item(key, value, category="profile", reason="测试用保存原因"):
    """构造一条形如大模型输出的记忆字典。"""
    return {
        "key": key,
        "value": value,
        "category": category,
        "reason": reason,
    }


def extraction(should_remember, memories):
    """构造形如 extract_memories() 返回值的字典。"""
    return {
        "should_remember": should_remember,
        "memories": memories,
    }


THREE_MEMORIES = [
    memory_item("name", "钦", category="identity", reason="用户说明姓名"),
    memory_item("major", "计算机专业", category="profile", reason="用户说明专业"),
    memory_item("career_goal", "大模型落地工程师", category="goal", reason="用户说明职业目标"),
]


# =====================================================================
# 场景 1：新增 3 条记忆
# =====================================================================

def test_add_three_memories_returns_added_3(tmp_path, monkeypatch):
    agent = build_agent(tmp_path, monkeypatch)
    db_path = tmp_path / "test_memory.db"

    outcome = agent.save_extracted_memories(
        extraction(True, THREE_MEMORIES)
    )

    assert outcome["added"] == 3
    assert outcome["updated"] == 0
    assert outcome["ignored"] == 0

    rows = physical_rows(db_path)
    assert len(rows) == 3
    assert [row[0] for row in rows] == ["name", "major", "career_goal"]
    assert [row[1] for row in rows] == ["钦", "计算机专业", "大模型落地工程师"]


# =====================================================================
# 场景 2：重复保存相同内容 —— 应全部 ignore，且不产生重复记录
# =====================================================================

def test_saving_same_content_twice_is_ignored_without_duplicates(tmp_path, monkeypatch):
    agent = build_agent(tmp_path, monkeypatch)
    db_path = tmp_path / "test_memory.db"

    first = agent.save_extracted_memories(extraction(True, THREE_MEMORIES))
    assert first["added"] == 3

    second = agent.save_extracted_memories(extraction(True, THREE_MEMORIES))

    assert second["added"] == 0
    assert second["updated"] == 0
    assert second["ignored"] == 3

    rows = physical_rows(db_path)
    assert len(rows) == 3, "重复保存后不应新增记录"

    keys = [row[0] for row in rows]
    assert len(keys) == len(set(keys)), "同一 key 不应出现重复行"


# =====================================================================
# 场景 3：修改已有记忆的 value —— 应 update，且值确实变了
# =====================================================================

def test_updating_value_returns_updated_1_and_persists_new_value(tmp_path, monkeypatch):
    agent = build_agent(tmp_path, monkeypatch)
    db_path = tmp_path / "test_memory.db"

    agent.save_extracted_memories(
        extraction(True, [memory_item("career_goal", "大模型落地工程师", category="goal")])
    )

    outcome = agent.save_extracted_memories(
        extraction(True, [
            memory_item(
                "career_goal",
                "AI 应用工程师",
                category="goal",
                reason="用户修正了职业目标",
            )
        ])
    )

    assert outcome["added"] == 0
    assert outcome["updated"] == 1
    assert outcome["ignored"] == 0

    rows = physical_rows(db_path)
    assert len(rows) == 1, "更新不应新增行"
    assert rows[0][1] == "AI 应用工程师", "数据库中的 value 应被更新"
    assert rows[0][5] == "用户修正了职业目标", "reason 应随更新一起写入"


# =====================================================================
# 场景 4：category / reason 是否真的写进了数据库
# =====================================================================

def test_category_and_reason_are_persisted_in_sqlite(tmp_path, monkeypatch):
    agent = build_agent(tmp_path, monkeypatch)
    db_path = tmp_path / "test_memory.db"

    agent.save_extracted_memories(extraction(True, THREE_MEMORIES))

    rows = physical_rows(db_path)
    by_key = {row[0]: row for row in rows}

    # 逐条核对：category 与 reason 必须等于传入值，而不是默认值
    assert by_key["name"][4] == "identity"
    assert by_key["name"][5] == "用户说明姓名"

    assert by_key["major"][4] == "profile"
    assert by_key["major"][5] == "用户说明专业"

    assert by_key["career_goal"][4] == "goal"
    assert by_key["career_goal"][5] == "用户说明职业目标"

    # 反向确认：不能落成 Memory 里的默认文案
    for row in rows:
        assert row[4] != "", "category 不应为空"
        assert row[5] != "未提供具体保存原因", "reason 不应落到默认说明"


def test_source_and_confidence_are_written_by_agent(tmp_path, monkeypatch):
    """save_extracted_memories 写死了 source='user'、confidence=1.0，此处固定住该行为。"""
    agent = build_agent(tmp_path, monkeypatch)
    db_path = tmp_path / "test_memory.db"

    agent.save_extracted_memories(extraction(True, THREE_MEMORIES))

    for row in physical_rows(db_path):
        assert row[2] == "user"
        assert row[3] == 1.0


# =====================================================================
# 场景 5：should_remember 为 False 时不应保存
# =====================================================================

def test_should_remember_false_saves_nothing(tmp_path, monkeypatch):
    agent = build_agent(tmp_path, monkeypatch)
    db_path = tmp_path / "test_memory.db"

    outcome = agent.save_extracted_memories(
        extraction(False, THREE_MEMORIES)
    )

    assert outcome == {"added": 0, "updated": 0, "ignored": 0}
    assert physical_rows(db_path) == []


def test_empty_memories_list_saves_nothing(tmp_path, monkeypatch):
    agent = build_agent(tmp_path, monkeypatch)
    db_path = tmp_path / "test_memory.db"

    outcome = agent.save_extracted_memories(extraction(True, []))

    assert outcome == {"added": 0, "updated": 0, "ignored": 0}
    assert physical_rows(db_path) == []


# =====================================================================
# 场景 6：缺少必要字段 / 字段值非法时的实际处理结果
# =====================================================================

def test_missing_should_remember_raises_keyerror(tmp_path, monkeypatch):
    """实际实现直接下标取值，缺字段时抛 KeyError（不是友好提示）。"""
    agent = build_agent(tmp_path, monkeypatch)

    with pytest.raises(KeyError):
        agent.save_extracted_memories({"memories": []})


def test_missing_memories_key_raises_keyerror(tmp_path, monkeypatch):
    agent = build_agent(tmp_path, monkeypatch)

    with pytest.raises(KeyError):
        agent.save_extracted_memories({"should_remember": True})


@pytest.mark.parametrize("missing_field", ["key", "value", "category", "reason"])
def test_missing_item_field_raises_keyerror(tmp_path, monkeypatch, missing_field):
    agent = build_agent(tmp_path, monkeypatch)
    db_path = tmp_path / "test_memory.db"

    incomplete = memory_item("some_key", "some_value")
    del incomplete[missing_field]

    with pytest.raises(KeyError):
        agent.save_extracted_memories(extraction(True, [incomplete]))

    assert physical_rows(db_path) == [], "失败时不应留下任何记录"


def test_empty_key_raises_valueerror(tmp_path, monkeypatch):
    agent = build_agent(tmp_path, monkeypatch)

    with pytest.raises(ValueError):
        agent.save_extracted_memories(
            extraction(True, [memory_item("", "任意值")])
        )


def test_blank_key_raises_valueerror(tmp_path, monkeypatch):
    agent = build_agent(tmp_path, monkeypatch)

    with pytest.raises(ValueError):
        agent.save_extracted_memories(
            extraction(True, [memory_item("   ", "任意值")])
        )


def test_empty_value_raises_valueerror(tmp_path, monkeypatch):
    agent = build_agent(tmp_path, monkeypatch)

    with pytest.raises(ValueError):
        agent.save_extracted_memories(
            extraction(True, [memory_item("some_key", "")])
        )


def test_non_string_value_raises_valueerror(tmp_path, monkeypatch):
    agent = build_agent(tmp_path, monkeypatch)

    with pytest.raises(ValueError):
        agent.save_extracted_memories(
            extraction(True, [memory_item("age", 25)])
        )


# =====================================================================
# 场景 7：保存中途出现异常时，数据是否一致
# =====================================================================

def test_batch_failure_rolls_back_all_rows_on_empty_db(tmp_path, monkeypatch):
    """
    空库 + 一批 3 条、第 2 条非法 → 整批回滚，物理表 0 行。

    本用例取代原先断言「部分提交」的旧用例
    （test_partial_failure_leaves_partial_committed_rows）：
    实施批量原子性后，期望行为已从「留下部分记录」变为「零残留」。
    """
    agent = build_agent(tmp_path, monkeypatch)
    db_path = tmp_path / "test_memory.db"

    batch = [
        memory_item("first_ok", "合法"),
        memory_item("", "空 key 会抛 ValueError"),
        memory_item("third_never", "不应被执行"),
    ]

    with pytest.raises(ValueError):
        agent.save_extracted_memories(extraction(True, batch))

    assert physical_rows(db_path) == [], "整批失败后不能留下任何记录"


def test_batch_failure_does_not_affect_existing_rows(tmp_path, monkeypatch):
    """已有数据不受失败批次影响：不新增、不改动、不删除。"""
    agent = build_agent(tmp_path, monkeypatch)
    db_path = tmp_path / "test_memory.db"

    baseline = agent.save_extracted_memories(
        extraction(True, [
            memory_item("name", "钦", category="identity", reason="基线原因1"),
            memory_item("major", "计算机专业", category="profile", reason="基线原因2"),
        ])
    )
    assert baseline["added"] == 2

    before = physical_rows(db_path)
    assert len(before) == 2

    with pytest.raises(ValueError):
        agent.save_extracted_memories(
            extraction(True, [
                memory_item("name", "阿钦", category="identity", reason="想改基线"),
                memory_item("city", "杭州", category="profile", reason="想新增"),
                memory_item("", "空 key 触发失败"),
            ])
        )

    after = physical_rows(db_path)

    assert after == before, "失败批次不得改动任何已有记录"
    by_key = {row[0]: row for row in after}
    assert len(by_key) == 2
    assert by_key["name"][1] == "钦", "基线的 value 不应被失败批次污染"
    assert by_key["name"][5] == "基线原因1", "基线的 reason 不应被失败批次污染"
    assert "city" not in by_key, "失败批次新增的记录不应残留"


def test_intra_batch_same_key_add_then_update(tmp_path, monkeypatch):
    """批内同 key：第 2 条必须看得到第 1 条尚未提交的写入，判为 update 而非再插一行。"""
    agent = build_agent(tmp_path, monkeypatch)
    db_path = tmp_path / "test_memory.db"

    outcome = agent.save_extracted_memories(
        extraction(True, [
            memory_item("career_goal", "大模型落地工程师", category="goal", reason="第一次"),
            memory_item("career_goal", "AI 应用工程师", category="goal", reason="第二次修正"),
        ])
    )

    assert outcome == {"added": 1, "updated": 1, "ignored": 0}

    rows = physical_rows(db_path)
    assert len(rows) == 1, "批内同 key 不应产生两行"
    assert rows[0][0] == "career_goal"
    assert rows[0][1] == "AI 应用工程师"
    assert rows[0][5] == "第二次修正"


def test_intra_batch_same_key_add_then_ignore(tmp_path, monkeypatch):
    """批内同 key 且 value 相同：第 2 条应判为 ignore。"""
    agent = build_agent(tmp_path, monkeypatch)
    db_path = tmp_path / "test_memory.db"

    outcome = agent.save_extracted_memories(
        extraction(True, [
            memory_item("name", "钦", category="identity"),
            memory_item("name", "钦", category="identity"),
        ])
    )

    assert outcome == {"added": 1, "updated": 0, "ignored": 1}
    assert len(physical_rows(db_path)) == 1, "批内同 key 同值不应产生两行"


def test_rollback_then_success_still_works(tmp_path, monkeypatch):
    """回滚之后连接与事务状态必须干净，后续批次仍能正常提交。"""
    agent = build_agent(tmp_path, monkeypatch)
    db_path = tmp_path / "test_memory.db"

    with pytest.raises(ValueError):
        agent.save_extracted_memories(
            extraction(True, [memory_item("", "触发失败")])
        )
    assert physical_rows(db_path) == []

    outcome = agent.save_extracted_memories(extraction(True, THREE_MEMORIES))

    assert outcome == {"added": 3, "updated": 0, "ignored": 0}
    rows = physical_rows(db_path)
    assert len(rows) == 3
    assert [row[0] for row in rows] == ["name", "major", "career_goal"]


def test_category_and_reason_persist_through_batch_update(tmp_path, monkeypatch):
    """批内的更新也要把 category / reason 真正写进数据库，而不是只留在入参字典里。"""
    agent = build_agent(tmp_path, monkeypatch)
    db_path = tmp_path / "test_memory.db"

    agent.save_extracted_memories(
        extraction(True, [
            memory_item("career_goal", "大模型落地工程师", category="goal", reason="初始原因"),
        ])
    )

    agent.save_extracted_memories(
        extraction(True, [
            memory_item("career_goal", "AI 应用工程师", category="skill", reason="用户改口后的原因"),
        ])
    )

    rows = physical_rows(db_path)
    assert len(rows) == 1
    assert rows[0][1] == "AI 应用工程师"
    assert rows[0][4] == "skill", "category 应随更新写入数据库"
    assert rows[0][5] == "用户改口后的原因", "reason 应随更新写入数据库"
    assert rows[0][5] != "未提供具体保存原因", "不应落到默认原因"


def test_after_rollback_find_memory_sees_nothing(tmp_path, monkeypatch):
    """回滚后再走公共读取接口，也不能看到任何残留。"""
    agent = build_agent(tmp_path, monkeypatch)

    with pytest.raises(ValueError):
        agent.save_extracted_memories(
            extraction(True, [
                memory_item("ghost", "这条不应存在"),
                memory_item("", "触发失败"),
            ])
        )

    assert agent.memory.find_memory("ghost", user_id="test-user") is None
    assert agent.memory.get_memories(user_id="test-user") == []


def test_failed_save_does_not_upgrade_return_value(tmp_path, monkeypatch):
    """异常必须向上抛出，不能被吞掉后返回一个「看起来成功」的计数。"""
    agent = build_agent(tmp_path, monkeypatch)

    with pytest.raises(ValueError):
        agent.save_extracted_memories(
            extraction(True, [memory_item("some_key", "")])
        )


# =====================================================================
# 返回值结构
# =====================================================================

def test_return_value_has_exactly_three_counters(tmp_path, monkeypatch):
    agent = build_agent(tmp_path, monkeypatch)

    outcome = agent.save_extracted_memories(extraction(True, []))

    assert isinstance(outcome, dict)
    assert set(outcome.keys()) == {"added", "updated", "ignored"}
    assert all(isinstance(v, int) for v in outcome.values())


def test_mixed_batch_counts_add_ignore_and_update_separately(tmp_path, monkeypatch):
    """一批里同时出现 新增 / 忽略 / 更新 时，三个计数器应分别统计。"""
    agent = build_agent(tmp_path, monkeypatch)
    db_path = tmp_path / "test_memory.db"

    agent.save_extracted_memories(
        extraction(True, [
            memory_item("name", "钦", category="identity"),
            memory_item("major", "计算机专业", category="profile"),
        ])
    )

    outcome = agent.save_extracted_memories(
        extraction(True, [
            memory_item("name", "钦", category="identity"),          # 忽略
            memory_item("major", "软件工程", category="profile"),     # 更新
            memory_item("city", "杭州", category="profile"),          # 新增
        ])
    )

    assert outcome == {"added": 1, "updated": 1, "ignored": 1}

    rows = physical_rows(db_path)
    assert len(rows) == 3
    by_key = {row[0]: row[1] for row in rows}
    assert by_key["major"] == "软件工程"
    assert by_key["city"] == "杭州"


# =====================================================================
# 场景 8：清理阶段（rollback / close）自身出错时的行为
#
# 真实的 sqlite3 连接几乎不会在 rollback/close 上抛错，
# 因此这里用最小假连接精确构造「清理失败」场景。
# =====================================================================

class _FakeConnection:
    """只实现批量保存会用到的三个动作的最小假连接。"""

    def __init__(self, rollback_error=None, close_error=None):
        self.rollback_error = rollback_error
        self.close_error = close_error
        self.commit_called = False
        self.rollback_called = False
        self.close_called = False

    def commit(self):
        self.commit_called = True

    def rollback(self):
        self.rollback_called = True
        if self.rollback_error is not None:
            raise self.rollback_error

    def close(self):
        self.close_called = True
        if self.close_error is not None:
            raise self.close_error


class _FakeMemory:
    """把 get_connection / save_memory 换成可控实现的假 Memory。"""

    def __init__(self, connection, save_error=None):
        self._connection = connection
        self._save_error = save_error

    def get_connection(self):
        return self._connection

    def save_memory(self, **_kwargs):
        if self._save_error is not None:
            raise self._save_error
        return "add"


def notes_of(exc):
    """取出异常上挂的附注（Python 3.11+ 的 add_note）。"""
    return list(getattr(exc, "__notes__", []))


def test_rollback_failure_does_not_mask_original_error(tmp_path, monkeypatch):
    """
    rollback 抛错时：原始业务异常必须原样抛出，
    不能被清理异常替换；同时仍要尽力 close，并留下诊断附注。
    """
    agent = build_agent(tmp_path, monkeypatch)

    original = ValueError("原始业务异常")
    fake = _FakeConnection(rollback_error=sqlite3.OperationalError("rollback 炸了"))
    agent.memory = _FakeMemory(fake, save_error=original)

    with pytest.raises(ValueError) as excinfo:
        agent.save_extracted_memories(extraction(True, [memory_item("k", "v")]))

    assert excinfo.value is original, "抛出的必须是原始异常对象本身"
    assert fake.rollback_called is True, "必须尽力执行 rollback"
    assert fake.close_called is True, "rollback 失败后仍必须尽力 close"
    assert any("rollback" in note for note in notes_of(original)),         "清理异常应作为附注保留下来"
    assert "rollback 炸了" in "".join(notes_of(original))


def test_close_failure_does_not_mask_original_error(tmp_path, monkeypatch):
    """close 抛错时同样不能覆盖原始业务异常。"""
    agent = build_agent(tmp_path, monkeypatch)

    original = ValueError("原始业务异常")
    fake = _FakeConnection(close_error=sqlite3.OperationalError("close 炸了"))
    agent.memory = _FakeMemory(fake, save_error=original)

    with pytest.raises(ValueError) as excinfo:
        agent.save_extracted_memories(extraction(True, [memory_item("k", "v")]))

    assert excinfo.value is original
    assert fake.rollback_called is True
    assert fake.close_called is True
    assert any("close" in note for note in notes_of(original))
    assert "close 炸了" in "".join(notes_of(original))


def test_rollback_and_close_both_fail_original_error_survives(tmp_path, monkeypatch):
    """rollback 与 close 同时失败，原始异常仍必须胜出，且两条诊断都保留。"""
    agent = build_agent(tmp_path, monkeypatch)

    original = ValueError("原始业务异常")
    fake = _FakeConnection(
        rollback_error=sqlite3.OperationalError("rollback 炸了"),
        close_error=sqlite3.OperationalError("close 炸了"),
    )
    agent.memory = _FakeMemory(fake, save_error=original)

    with pytest.raises(ValueError) as excinfo:
        agent.save_extracted_memories(extraction(True, [memory_item("k", "v")]))

    assert excinfo.value is original
    joined = "".join(notes_of(original))
    assert "rollback 炸了" in joined
    assert "close 炸了" in joined
    assert len(notes_of(original)) >= 2


def test_close_failure_on_success_path_is_not_swallowed(tmp_path, monkeypatch):
    """
    没有业务异常时，close 失败不能被悄悄吞掉，
    否则连接泄漏会无人知晓。
    """
    agent = build_agent(tmp_path, monkeypatch)

    fake = _FakeConnection(close_error=sqlite3.OperationalError("close 炸了"))
    agent.memory = _FakeMemory(fake)

    with pytest.raises(sqlite3.OperationalError) as excinfo:
        agent.save_extracted_memories(extraction(True, [memory_item("k", "v")]))

    assert "close 炸了" in str(excinfo.value)
    assert fake.commit_called is True
    assert fake.close_called is True


def test_success_path_commits_and_closes_without_rollback(tmp_path, monkeypatch):
    """正常路径：commit 与 close 各执行一次，rollback 完全不执行。"""
    agent = build_agent(tmp_path, monkeypatch)

    fake = _FakeConnection()
    agent.memory = _FakeMemory(fake)

    outcome = agent.save_extracted_memories(
        extraction(True, [memory_item("a", "1"), memory_item("b", "2")])
    )

    assert outcome == {"added": 2, "updated": 0, "ignored": 0}
    assert fake.commit_called is True
    assert fake.close_called is True
    assert fake.rollback_called is False
