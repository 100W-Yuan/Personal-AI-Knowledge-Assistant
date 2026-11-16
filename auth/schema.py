"""认证相关表结构与幂等迁移。

设计约束（对应任务书"修正 4"）：
- 每条 DDL 单独 ``execute()``，禁止 ``executescript()``（后者失败时无法回滚）；
- 只做「加表 / 加列 / 加索引」，绝不删表、删列、改列类型；
- 全部 ``IF NOT EXISTS`` 或"先查 PRAGMA 再加列"，可重复执行。
"""

from __future__ import annotations

import sqlite3

USERS_TABLE = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('admin', 'user')),
    is_active INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
    must_change_password INTEGER NOT NULL DEFAULT 0 CHECK (must_change_password IN (0, 1)),
    created_at TEXT NOT NULL
)
"""

SESSIONS_TABLE = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    token_hash TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    revoked_at TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id)
)
"""

AUDIT_LOGS_TABLE = """
CREATE TABLE IF NOT EXISTS audit_logs (
    id TEXT PRIMARY KEY,
    actor_user_id TEXT,
    actor_type TEXT NOT NULL,
    target_user_id TEXT,
    action TEXT NOT NULL,
    result TEXT NOT NULL,
    detail TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (actor_user_id) REFERENCES users(id)
)
"""

SESSION_USER_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions (user_id)"
)

AUDIT_LOGS_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_audit_logs_created_at ON audit_logs (created_at)"
)

AUTH_STATEMENTS = (
    USERS_TABLE,
    SESSIONS_TABLE,
    SESSION_USER_INDEX,
    AUDIT_LOGS_TABLE,
    AUDIT_LOGS_INDEX,
)

# (表名, 新增列名, 列定义) —— 认证表幂等补列（兼容旧库）
AUTH_SCOPED_COLUMNS = (
    ("users", "must_change_password", "INTEGER NOT NULL DEFAULT 0"),
)

# (表名, 新增列名, 列定义) —— 用户私有数据归属
USER_SCOPED_COLUMNS = (
    ("conversations", "user_id", "TEXT"),
    ("memories", "user_id", "TEXT"),
)

USER_SCOPED_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_conversations_user_id "
    "ON conversations (user_id)",
    "CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories (user_id)",
)


def table_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    """返回表的列名集合；表不存在时返回空集合。"""
    try:
        rows = connection.execute("PRAGMA table_info(%s)" % table).fetchall()
    except sqlite3.Error:
        return set()
    return {row[1] for row in rows}


def ensure_auth_schema(connection: sqlite3.Connection) -> None:
    """建 users / sessions 表与索引（幂等）。"""
    for statement in AUTH_STATEMENTS:
        connection.execute(statement)


def ensure_auth_columns(connection: sqlite3.Connection) -> None:
    """给 users 表补认证列（幂等，不动既有数据）。"""
    for table, column, column_type in AUTH_SCOPED_COLUMNS:
        columns = table_columns(connection, table)
        if not columns:
            # 表还不存在：由 ensure_auth_schema 负责建表，这里跳过
            continue
        if column not in columns:
            connection.execute(
                "ALTER TABLE %s ADD COLUMN %s %s" % (table, column, column_type)
            )


def ensure_user_scoped_columns(connection: sqlite3.Connection) -> None:
    """给 conversations / memories 补 user_id 列（幂等，不动既有数据）。"""
    for table, column, column_type in USER_SCOPED_COLUMNS:
        columns = table_columns(connection, table)
        if not columns:
            # 表还不存在：由 Memory.init_database 负责建表，这里跳过
            continue
        if column not in columns:
            connection.execute(
                "ALTER TABLE %s ADD COLUMN %s %s" % (table, column, column_type)
            )
    for statement in USER_SCOPED_INDEXES:
        try:
            connection.execute(statement)
        except sqlite3.Error:
            # 表不存在时建索引会失败；不影响后续流程
            continue


def apply_all(connection: sqlite3.Connection) -> None:
    ensure_auth_schema(connection)
    ensure_auth_columns(connection)
    ensure_user_scoped_columns(connection)
