"""密码哈希、会话令牌与基础校验。

只用标准库 + argon2-cffi：
- 密码：Argon2id（time_cost=3, memory_cost=64 MiB, parallelism=4）；
- 令牌：``secrets.token_urlsafe(32)``（256 bit 熵），库里只存 SHA-256；
- 时间：统一 UTC ISO8601（定宽，可直接按字符串比较）。
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
from datetime import datetime, timedelta, timezone

from argon2 import PasswordHasher, Type
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

# Argon2id 参数：本机实测 hash 55.6 ms / verify 57.4 ms（2C2G 目标机约 100-150 ms）
_PASSWORD_HASHER = PasswordHasher(
    time_cost=3,
    memory_cost=65536,
    parallelism=4,
    hash_len=32,
    salt_len=16,
    type=Type.ID,
)

SESSION_TOKEN_BYTES = 32
_USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
PASSWORD_MIN_LENGTH = 10
PASSWORD_MAX_LENGTH = 128

IDLE_TIMEOUT = timedelta(hours=24)
ABSOLUTE_TIMEOUT = timedelta(days=7)
RENEW_THROTTLE = timedelta(seconds=60)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def format_time(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_time(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def normalize_username(username: str) -> str:
    if not isinstance(username, str):
        raise ValueError("用户名必须是字符串")
    value = username.strip()
    if not _USERNAME_RE.match(value):
        raise ValueError("用户名只能是 3-32 位字母、数字、下划线、点或短横线")
    return value


def validate_password(password: str) -> str:
    if not isinstance(password, str):
        raise ValueError("密码必须是字符串")
    if len(password) < PASSWORD_MIN_LENGTH:
        raise ValueError("密码至少 %d 位" % PASSWORD_MIN_LENGTH)
    if len(password) > PASSWORD_MAX_LENGTH:
        raise ValueError("密码不能超过 %d 位" % PASSWORD_MAX_LENGTH)
    return password


def hash_password(password: str) -> str:
    return _PASSWORD_HASHER.hash(validate_password(password))


def verify_password(password_hash: str, password: str) -> bool:
    if not password_hash or not isinstance(password, str):
        return False
    try:
        return _PASSWORD_HASHER.verify(password_hash, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def password_needs_rehash(password_hash: str) -> bool:
    try:
        return _PASSWORD_HASHER.check_needs_rehash(password_hash)
    except InvalidHashError:
        return True


def new_session_token() -> str:
    return secrets.token_urlsafe(SESSION_TOKEN_BYTES)


def new_temporary_password() -> str:
    """生成一次性临时密码（管理员创建/重置账号时仅回显一次）。

    ``token_urlsafe(12)`` 约 16 字符，满足 ``validate_password`` 的 10–128 位
    长度约束；明文绝不入库、不入日志。
    """
    return secrets.token_urlsafe(12)


def new_identifier() -> str:
    return secrets.token_hex(16)


def token_digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def constant_time_equals(left: str, right: str) -> bool:
    return hmac.compare_digest(left or "", right or "")


def looks_hashed(value: str) -> bool:
    """粗判数据库里存的密码字段是否是 Argon2 哈希（用于「无明文」自检）。"""
    return isinstance(value, str) and value.startswith("$argon2id$")
