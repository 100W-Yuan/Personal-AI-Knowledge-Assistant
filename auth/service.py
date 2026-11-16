"""认证业务逻辑：用户、会话、登录限流、历史数据归属。

只依赖标准库 + ``auth.security``；不 import ``main``（避免循环依赖），
数据库路径通过 ``Memory`` 的默认路径规则（相对当前工作目录的 ``memory.db``）
或环境变量 ``PAI_MEMORY_DB`` 解析。
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from typing import Any

from auth import schema as auth_schema
from auth import security

MAX_LOGIN_FAILURES = 5
LOGIN_FAILURE_WINDOW_SECONDS = 900
LOGIN_LOCK_SECONDS = 900

# 审计操作者来源：HTTP 管理员操作 / HTTP 用户自助 / CLI（MVP 暂不写审计）。
ACTOR_TYPE_HTTP_ADMIN = "http_admin"
ACTOR_TYPE_HTTP_USER = "http_user"
ACTOR_TYPE_CLI = "cli"

# 管理操作动作名（审计 action 字段取值）。
AUDIT_ACTION_CREATE_USER = "create_user"
AUDIT_ACTION_RESET_PASSWORD = "reset_password"
AUDIT_ACTION_SET_ACTIVE = "set_active"
AUDIT_ACTION_REVOKE_SESSIONS = "revoke_sessions"
AUDIT_ACTION_CHANGE_PASSWORD = "change_password"

logger = logging.getLogger(__name__)

_failure_lock = threading.Lock()
_failures: dict[str, list[float]] = {}


class AuthError(Exception):
    """认证类错误基类；``code`` 供 HTTP 层映射状态码与错误体。"""

    code = "auth_error"

    def __init__(self, message: str = ""):
        super().__init__(message or self.code)
        self.message = message or self.code


class InvalidCredentials(AuthError):
    code = "invalid_credentials"


class AccountDisabled(AuthError):
    code = "account_disabled"


class RateLimited(AuthError):
    code = "rate_limited"


class UsernameTaken(AuthError):
    code = "username_taken"


class UserNotFound(AuthError):
    code = "user_not_found"


def _default_db_path() -> str:
    """优先复用 ``main.get_memory()`` 的库路径，保证认证与业务同库。"""
    configured = os.environ.get("PAI_MEMORY_DB")
    if configured:
        return configured
    try:
        import main  # 延迟导入：避免模块级循环依赖

        memory = main.get_memory()
        path = getattr(memory, "db_path", None)
        if isinstance(path, str) and path:
            return path
    except Exception:
        pass
    return "memory.db"


def _reset_failures_for_tests() -> None:
    with _failure_lock:
        _failures.clear()


class AuthService:
    """认证服务。每次操作独立开关连接，不持有长连接。"""

    def __init__(self, db_path: str | None = None):
        self._db_path = db_path

    # ------------------------------------------------------------
    # 连接与库表
    # ------------------------------------------------------------

    @property
    def db_path(self) -> str:
        return self._db_path or _default_db_path()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def ensure_schema(self) -> None:
        """幂等建表 + 补列；单条 execute，失败整体回滚。"""
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            try:
                auth_schema.apply_all(connection)
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        finally:
            connection.close()

    # ------------------------------------------------------------
    # 用户
    # ------------------------------------------------------------

    @staticmethod
    def _row_to_user(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "username": row["username"],
            "role": row["role"],
            "is_active": bool(row["is_active"]),
            "must_change_password": bool(row["must_change_password"]),
            "created_at": row["created_at"],
        }

    def count_admins(self) -> int:
        connection = self.connect()
        try:
            row = connection.execute(
                "SELECT COUNT(*) FROM users WHERE role = 'admin'"
            ).fetchone()
            return int(row[0])
        finally:
            connection.close()

    def get_user(self, username: str) -> dict[str, Any] | None:
        try:
            name = security.normalize_username(username)
        except ValueError:
            return None
        connection = self.connect()
        try:
            row = connection.execute(
                "SELECT id, username, role, is_active, must_change_password, created_at "
                "FROM users WHERE username = ?",
                (name,),
            ).fetchone()
            return self._row_to_user(row) if row is not None else None
        finally:
            connection.close()

    def list_users(self) -> list[dict[str, Any]]:
        """列出全部用户（不含 password_hash）；供管理员接口使用。"""
        connection = self.connect()
        try:
            rows = connection.execute(
                "SELECT id, username, role, is_active, must_change_password, created_at "
                "FROM users ORDER BY username"
            ).fetchall()
            return [self._row_to_user(row) for row in rows]
        finally:
            connection.close()

    @staticmethod
    def _audit_insert(connection, actor_user_id, actor_type, target_user_id,
                      action, result, detail):
        """在当前事务内插入一条审计（detail 只允许安全字段）。"""
        connection.execute(
            "INSERT INTO audit_logs "
            "(id, actor_user_id, actor_type, target_user_id, action, result, detail, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                security.new_identifier(),
                actor_user_id,
                actor_type,
                target_user_id,
                action,
                result,
                detail,
                security.format_time(security.utcnow()),
            ),
        )

    @staticmethod
    def _error_code(exc: Exception) -> str:
        return getattr(exc, "code", None) or type(exc).__name__

    def _record_failure_audit(self, actor_type, actor_user_id, action, exc,
                              username=None):
        """失败审计：独立连接、best-effort；MVP CLI（actor_type=None）不写。"""
        if actor_type is None:
            return
        detail_payload = {"error_code": self._error_code(exc)}
        if username:
            detail_payload["username"] = username
        detail = json.dumps(detail_payload, ensure_ascii=False)
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._audit_insert(
                    connection, actor_user_id, actor_type, None,
                    action, "failure", detail,
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        except Exception:
            # best-effort：绝不覆盖原始业务异常，也不让审计失败影响 CLI/HTTP 结果
            logger.error(
                "失败审计写入失败（action=%s, error=%s）",
                action, self._error_code(exc),
            )
        finally:
            connection.close()

    def create_user(
        self,
        username: str,
        password: str,
        role: str = "user",
        actor_user_id: str | None = None,
        actor_type: str | None = None,
        must_change_password: bool = False,
    ) -> dict:
        if role not in ("admin", "user"):
            raise ValueError("role 只能是 admin 或 user")
        name = security.normalize_username(username)
        password_hash = security.hash_password(password)

        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            try:
                existing = connection.execute(
                    "SELECT id FROM users WHERE username = ?", (name,)
                ).fetchone()
                if existing is not None:
                    raise UsernameTaken("用户名已存在")
                user_id = security.new_identifier()
                connection.execute(
                    "INSERT INTO users "
                    "(id, username, password_hash, role, is_active, "
                    " must_change_password, created_at) "
                    "VALUES (?, ?, ?, ?, 1, ?, ?)",
                    (
                        user_id,
                        name,
                        password_hash,
                        role,
                        1 if must_change_password else 0,
                        security.format_time(security.utcnow()),
                    ),
                )
                if actor_type is not None:
                    self._audit_insert(
                        connection, actor_user_id, actor_type, user_id,
                        AUDIT_ACTION_CREATE_USER, "success", None,
                    )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        except Exception as exc:
            self._record_failure_audit(
                actor_type, actor_user_id, AUDIT_ACTION_CREATE_USER, exc, name
            )
            raise
        finally:
            connection.close()

        return {
            "id": user_id,
            "username": name,
            "role": role,
            "is_active": True,
            "must_change_password": bool(must_change_password),
        }

    def set_password(self, username: str, password: str) -> dict:
        """重置密码并撤销全部会话（CLI 旧行为：不强制改密、不写审计）。"""
        return self._set_password_impl(
            username, password,
            force_change=False, actor_user_id=None, actor_type=None,
            audit_action=None,
        )

    def reset_password(self, username: str, password: str,
                       actor_user_id: str | None = None,
                       actor_type: str | None = None) -> dict:
        """管理员 HTTP API 重置密码：改哈希 + 强制改密 + 撤销全部会话 + 审计。"""
        return self._set_password_impl(
            username, password,
            force_change=True, actor_user_id=actor_user_id,
            actor_type=actor_type, audit_action=AUDIT_ACTION_RESET_PASSWORD,
        )

    def _set_password_impl(self, username, password, *, force_change,
                           actor_user_id, actor_type, audit_action):
        """重置密码的单一实现：Session 撤销与审计各只发生一次，无重复。"""
        name = security.normalize_username(username)
        password_hash = security.hash_password(password)

        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT id FROM users WHERE username = ?", (name,)
                ).fetchone()
                if row is None:
                    raise UserNotFound("用户不存在")
                user_id = row["id"]
                connection.execute(
                    "UPDATE users SET password_hash = ? WHERE id = ?",
                    (password_hash, user_id),
                )
                if force_change:
                    connection.execute(
                        "UPDATE users SET must_change_password = 1 WHERE id = ?",
                        (user_id,),
                    )
                now = security.format_time(security.utcnow())
                revoked = connection.execute(
                    "UPDATE sessions SET revoked_at = ? "
                    "WHERE user_id = ? AND revoked_at IS NULL",
                    (now, user_id),
                ).rowcount
                if audit_action is not None and actor_type is not None:
                    self._audit_insert(
                        connection, actor_user_id, actor_type, user_id,
                        audit_action, "success", None,
                    )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        except Exception as exc:
            if audit_action is not None:
                self._record_failure_audit(
                    actor_type, actor_user_id, audit_action, exc, name
                )
            raise
        finally:
            connection.close()

        result = {"user_id": user_id, "revoked_sessions": revoked}
        if force_change:
            result["must_change_password"] = True
        return result

    def set_active(self, username: str, is_active: bool,
                   actor_user_id: str | None = None,
                   actor_type: str | None = None) -> dict:
        """启用/禁用账号；禁用时立即撤销其全部会话。HTTP 管理操作写审计。"""
        name = security.normalize_username(username)

        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT id, role FROM users WHERE username = ?", (name,)
                ).fetchone()
                if row is None:
                    raise UserNotFound("用户不存在")
                user_id = row["id"]
                if actor_user_id is not None and not is_active and user_id == actor_user_id:
                    raise ValueError("不能禁用自己")
                if row["role"] == "admin" and not is_active:
                    admins = connection.execute(
                        "SELECT COUNT(*) FROM users WHERE role='admin' AND is_active=1"
                    ).fetchone()[0]
                    if admins <= 1:
                        raise ValueError("不能禁用最后一个启用状态的管理员")
                connection.execute(
                    "UPDATE users SET is_active = ? WHERE id = ?",
                    (1 if is_active else 0, user_id),
                )
                revoked = 0
                if not is_active:
                    now = security.format_time(security.utcnow())
                    revoked = connection.execute(
                        "UPDATE sessions SET revoked_at = ? "
                        "WHERE user_id = ? AND revoked_at IS NULL",
                        (now, user_id),
                    ).rowcount
                if actor_type is not None:
                    self._audit_insert(
                        connection, actor_user_id, actor_type, user_id,
                        AUDIT_ACTION_SET_ACTIVE, "success", None,
                    )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        except Exception as exc:
            self._record_failure_audit(
                actor_type, actor_user_id, AUDIT_ACTION_SET_ACTIVE, exc, name
            )
            raise
        finally:
            connection.close()

        return {"user_id": user_id, "is_active": is_active, "revoked_sessions": revoked}

    # ------------------------------------------------------------
    # 登录限流（进程内，单 Worker）
    # ------------------------------------------------------------

    @staticmethod
    def _rate_key(username: str) -> str:
        return username.strip().lower()[:64]

    def _check_limited(self, key: str) -> None:
        now = time.monotonic()
        with _failure_lock:
            stamps = [t for t in _failures.get(key, []) if now - t < LOGIN_LOCK_SECONDS]
            _failures[key] = stamps
            if len(stamps) >= MAX_LOGIN_FAILURES:
                raise RateLimited("登录失败次数过多，请稍后再试")

    def _record_failure(self, key: str) -> None:
        now = time.monotonic()
        with _failure_lock:
            stamps = [t for t in _failures.get(key, []) if now - t < LOGIN_LOCK_SECONDS]
            stamps.append(now)
            _failures[key] = stamps

    def _clear_failures(self, key: str) -> None:
        with _failure_lock:
            _failures.pop(key, None)

    # ------------------------------------------------------------
    # 登录 / 会话
    # ------------------------------------------------------------

    def login(self, username: str, password: str) -> dict:
        """校验账号密码并签发新会话；返回 ``{"token", "user"}``。"""
        key = self._rate_key(username)
        self._check_limited(key)

        connection = self.connect()
        try:
            row = connection.execute(
                "SELECT id, username, password_hash, role, is_active, "
                "       must_change_password, created_at "
                "FROM users WHERE username = ?",
                (username.strip(),),
            ).fetchone()
        finally:
            connection.close()

        # 用户不存在也走一次哈希校验，避免用响应时间区分「用户不存在」与「密码错误」
        stored_hash = row["password_hash"] if row is not None else security.hash_password(
            "dummy-password-for-timing"
        )
        password_ok = security.verify_password(stored_hash, password)

        if row is None or not password_ok:
            self._record_failure(key)
            raise InvalidCredentials("用户名或密码不正确")

        if not row["is_active"]:
            self._record_failure(key)
            raise AccountDisabled("账号已被禁用")

        self._clear_failures(key)
        user = self._row_to_user(
            {
                "id": row["id"],
                "username": row["username"],
                "role": row["role"],
                "is_active": row["is_active"],
                "must_change_password": row["must_change_password"],
                "created_at": row["created_at"],
            }
        )
        token = self._issue_session(user["id"])
        return {"token": token, "user": user}

    def _issue_session(self, user_id: str) -> str:
        token = security.new_session_token()
        digest = security.token_digest(token)
        now = security.utcnow()
        created = now
        expires = min(now + security.IDLE_TIMEOUT, created + security.ABSOLUTE_TIMEOUT)

        connection = self.connect()
        try:
            connection.execute(
                "INSERT INTO sessions "
                "(id, user_id, token_hash, created_at, last_seen_at, expires_at, revoked_at) "
                "VALUES (?, ?, ?, ?, ?, ?, NULL)",
                (
                    security.new_identifier(),
                    user_id,
                    digest,
                    security.format_time(now),
                    security.format_time(now),
                    security.format_time(expires),
                ),
            )
            connection.commit()
        finally:
            connection.close()
        return token

    def validate_session(self, token: str | None) -> dict | None:
        """校验会话；有效则按需续期，返回用户信息。"""
        if not token or not isinstance(token, str):
            return None
        digest = security.token_digest(token)

        connection = self.connect()
        try:
            row = connection.execute(
                "SELECT s.id AS sid, s.created_at, s.last_seen_at, s.expires_at, "
                "       s.revoked_at, u.id, u.username, u.role, u.is_active, "
                "       u.must_change_password, u.created_at AS ucreated "
                "FROM sessions AS s JOIN users AS u ON u.id = s.user_id "
                "WHERE s.token_hash = ?",
                (digest,),
            ).fetchone()
            if row is None or row["revoked_at"] is not None:
                return None
            if not row["is_active"]:
                self._revoke_session_id(row["sid"])
                return None

            now = security.utcnow()
            created = security.parse_time(row["created_at"])
            last_seen = security.parse_time(row["last_seen_at"])
            expires_at = security.parse_time(row["expires_at"])

            if now >= expires_at or now >= created + security.ABSOLUTE_TIMEOUT:
                self._revoke_session_id(row["sid"])
                return None
            if now - last_seen > security.IDLE_TIMEOUT:
                self._revoke_session_id(row["sid"])
                return None

            if now - last_seen > security.RENEW_THROTTLE:
                new_expiry = min(
                    now + security.IDLE_TIMEOUT, created + security.ABSOLUTE_TIMEOUT
                )
                connection.execute(
                    "UPDATE sessions SET last_seen_at = ?, expires_at = ? "
                    "WHERE id = ? AND revoked_at IS NULL",
                    (
                        security.format_time(now),
                        security.format_time(new_expiry),
                        row["sid"],
                    ),
                )
                connection.commit()

            return {
                "id": row["id"],
                "username": row["username"],
                "role": row["role"],
                "is_active": bool(row["is_active"]),
                "must_change_password": bool(row["must_change_password"]),
                "created_at": row["ucreated"],
            }
        finally:
            connection.close()

    def _revoke_session_id(self, session_id: str) -> None:
        connection = self.connect()
        try:
            connection.execute(
                "UPDATE sessions SET revoked_at = ? WHERE id = ? AND revoked_at IS NULL",
                (security.format_time(security.utcnow()), session_id),
            )
            connection.commit()
        finally:
            connection.close()

    def logout(self, token: str | None) -> bool:
        if not token:
            return False
        digest = security.token_digest(token)
        connection = self.connect()
        try:
            changed = connection.execute(
                "UPDATE sessions SET revoked_at = ? "
                "WHERE token_hash = ? AND revoked_at IS NULL",
                (security.format_time(security.utcnow()), digest),
            ).rowcount
            connection.commit()
            return changed > 0
        finally:
            connection.close()

    def revoke_all_sessions(self, user_id: str,
                            actor_user_id: str | None = None,
                            actor_type: str | None = None) -> int:
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            try:
                changed = connection.execute(
                    "UPDATE sessions SET revoked_at = ? "
                    "WHERE user_id = ? AND revoked_at IS NULL",
                    (security.format_time(security.utcnow()), user_id),
                ).rowcount
                if actor_type is not None:
                    self._audit_insert(
                        connection, actor_user_id, actor_type, user_id,
                        AUDIT_ACTION_REVOKE_SESSIONS, "success", None,
                    )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        except Exception as exc:
            self._record_failure_audit(
                actor_type, actor_user_id, AUDIT_ACTION_REVOKE_SESSIONS, exc
            )
            raise
        finally:
            connection.close()
        return changed

    def change_password(self, user_id: str, current_password: str,
                        new_password: str, current_session_token: str) -> dict:
        """当前用户自助改密：验旧密码 → 改哈希 → 清改密标记 → 撤其它会话保留当前 → 审计。"""
        new_hash = security.hash_password(new_password)

        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT password_hash FROM users WHERE id = ?", (user_id,)
                ).fetchone()
                if row is None or not security.verify_password(
                    row["password_hash"], current_password
                ):
                    raise InvalidCredentials("当前密码不正确")
                connection.execute(
                    "UPDATE users SET password_hash = ?, must_change_password = 0 "
                    "WHERE id = ?",
                    (new_hash, user_id),
                )
                digest = security.token_digest(current_session_token)
                connection.execute(
                    "UPDATE sessions SET revoked_at = ? "
                    "WHERE user_id = ? AND token_hash != ? AND revoked_at IS NULL",
                    (security.format_time(security.utcnow()), user_id, digest),
                )
                self._audit_insert(
                    connection, user_id, ACTOR_TYPE_HTTP_USER, user_id,
                    AUDIT_ACTION_CHANGE_PASSWORD, "success", None,
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        except Exception as exc:
            self._record_failure_audit(
                ACTOR_TYPE_HTTP_USER, user_id, AUDIT_ACTION_CHANGE_PASSWORD, exc
            )
            raise
        finally:
            connection.close()

        return {"must_change_password": False}

    # ------------------------------------------------------------
    # 历史数据归属
    # ------------------------------------------------------------

    def legacy_status(self) -> dict:
        """统计尚无归属的历史数据（只读）。"""
        connection = self.connect()
        try:
            def count(sql: str) -> int:
                return int(connection.execute(sql).fetchone()[0])

            return {
                "unowned_conversations": count(
                    "SELECT COUNT(*) FROM conversations WHERE user_id IS NULL"
                ),
                "unowned_memories": count(
                    "SELECT COUNT(*) FROM memories WHERE user_id IS NULL"
                ),
                "orphan_messages": count(
                    "SELECT COUNT(*) FROM messages AS m "
                    "WHERE NOT EXISTS (SELECT 1 FROM conversations AS c "
                    "                  WHERE c.conversation_id = m.conversation_id)"
                ),
                "total_conversations": count("SELECT COUNT(*) FROM conversations"),
                "total_messages": count("SELECT COUNT(*) FROM messages"),
                "total_memories": count("SELECT COUNT(*) FROM memories"),
            }
        finally:
            connection.close()

    def claim_legacy_data(self, user_id: str) -> dict:
        """把**尚无归属**的会话与记忆归属到指定用户；已有归属的一行不动。"""
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            try:
                conversations = connection.execute(
                    "UPDATE conversations SET user_id = ? WHERE user_id IS NULL",
                    (user_id,),
                ).rowcount
                memories = connection.execute(
                    "UPDATE memories SET user_id = ? WHERE user_id IS NULL",
                    (user_id,),
                ).rowcount
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        finally:
            connection.close()
        return {"conversations": conversations, "memories": memories}
