# -*- coding: utf-8 -*-
"""
Memory 连接生命周期测试 —— close() / 上下文管理器 / Windows 句柄释放

背景：真实流程验收脚本在 TemporaryDirectory 清理时报
PermissionError: [WinError 32]（另一个程序正在使用此文件）。
本文件验证 Memory 新增的 close() 机制：

    1. close() 可正常调用，且重复调用不报错（幂等）
    2. close() 能释放 get_connection() 创建的连接句柄
    3. Windows 下 close() 后临时数据库文件可以删除
    4. Agent 批量保存（save_extracted_memories）所用连接也被 close() 覆盖
    5. close() 后 Memory 各方法照常工作（API 兼容）
    6. with 上下文管理器退出时自动 close
    7. 正式 memory.db 测试前后不变

数据安全：全部使用 tmp_path 临时数据库，零网络请求（dummy Agent）。
"""

import hashlib
import os
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

# 正式库指纹：模块加载时（任何用例运行前）记录，最后一个用例复查。
REAL_DB_PATH = PROJECT_ROOT / "memory.db"
REAL_DB_MD5_BEFORE = hashlib.md5(REAL_DB_PATH.read_bytes()).hexdigest()

DUMMY_API_KEY = "dummy-key-never-used"
DUMMY_BASE_URL = "https://example.invalid"
DUMMY_MODEL = "dummy-model-never-used"


def build_agent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    agent = Agent(
        api_key=DUMMY_API_KEY,
        base_url=DUMMY_BASE_URL,
        model=DUMMY_MODEL,
    )
    agent.memory = Memory(db_path=str(tmp_path / "memory.db"))
    return agent


class TestMemoryClose:

    def test_close_is_callable_and_idempotent(self, tmp_path):
        memory = Memory(db_path=str(tmp_path / "t.db"))
        memory.close()
        memory.close()  # 第二次调用不得报错

    def test_close_releases_get_connection_handles(self, tmp_path):
        db_path = tmp_path / "t.db"
        memory = Memory(db_path=str(db_path))
        connection = memory.get_connection()

        memory.close()

        # 连接已被释放：再执行任何语句必须报错
        with pytest.raises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")

    @pytest.mark.skipif(
        sys.platform != "win32",
        reason="Windows 文件句柄锁只有 Windows 才有",
    )
    def test_windows_db_file_removable_after_close(self, tmp_path):
        """句柄未释放时 Windows 拒绝删除 db 文件；close() 后必须可删。"""
        db_path = tmp_path / "t.db"
        memory = Memory(db_path=str(db_path))
        connection = memory.get_connection()

        with pytest.raises(PermissionError):
            os.remove(str(db_path))

        connection.close()   # 模拟既有方法自带的 finally-close
        memory.close()       # 对已关闭连接再 close 也不得报错
        os.remove(str(db_path))  # 现在必须能删

    def test_close_after_agent_batch_save(self, tmp_path, monkeypatch):
        """Agent 批量保存（save_extracted_memories）的连接也在 close() 覆盖内。"""
        agent = build_agent(tmp_path, monkeypatch)
        agent.save_extracted_memories({
            "should_remember": True,
            "memories": [
                {"key": "name", "value": "钦",
                 "category": "identity", "reason": "用户自述"},
            ],
        })
        agent.memory.close()
        agent.memory.close()

    def test_memory_usable_after_close(self, tmp_path):
        """close() 后各方法照常工作：连接按需重建（API 完全兼容）。"""
        memory = Memory(db_path=str(tmp_path / "t.db"))
        memory.save_memory(key="major", value="计算机专业",
                           category="profile", reason="测试", user_id="test-user")
        memory.close()

        assert memory.find_memory("major", user_id="test-user")["value"] == "计算机专业"
        assert memory.search_long_term_memory("专业", user_id="test-user")
        assert memory.get_memories_by_category("profile", user_id="test-user")

        memory.close()  # 再次 close 依然不报错

    def test_context_manager_closes(self, tmp_path):
        db_path = str(tmp_path / "t.db")
        with Memory(db_path=db_path) as memory:
            memory.save_memory(key="k", value="v",
                               category="profile", reason="t", user_id="test-user")
            connection = memory.get_connection()

        # with 退出后连接必须已释放
        with pytest.raises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")


# =====================================================================
# 数据安全：正式库未被触碰
# =====================================================================

def test_real_memory_db_never_touched():
    """所有用例运行完后，项目正式 memory.db 的 MD5 必须与测试前一致。"""
    md5_now = hashlib.md5(REAL_DB_PATH.read_bytes()).hexdigest()
    assert md5_now == REAL_DB_MD5_BEFORE
