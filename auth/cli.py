"""账号管理与数据库迁移命令行工具。

用法（密码一律交互输入，**不接受**命令行参数）：

    python -m auth.cli create-admin --username owner
    python -m auth.cli create-user  --username hr
    python -m auth.cli reset-password --username hr
    python -m auth.cli disable --username hr
    python -m auth.cli enable  --username hr
    python -m auth.cli status
    python -m auth.cli migrate --backup-dir .backup_auth
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import os
import sqlite3
import sys
from datetime import datetime, timezone

from auth import schema as auth_schema
from auth import security
from auth.service import AuthError, AuthService, UserNotFound, UsernameTaken


# ------------------------------------------------------------
# 输出与交互
# ------------------------------------------------------------


def _fail(message: str, code: int = 1) -> int:
    print("[失败] %s" % message, file=sys.stderr)
    return code


def _read_password_twice(prompt: str, allow_env: bool = False) -> str:
    """读取并确认新密码。

    默认只用两个通道，二者都不把密码写进 argv：
      - 交互式终端：``getpass`` 两次确认（不回显）；
      - 非交互标准输入：读一行（CI / 管道 / 远程会话兜底），「一次输入」
        即视为已确认，这是 CI 场景的通行做法。
    环境变量通道 ``PAI_CLI_PASSWORD`` **默认禁用**：进程环境可能被子进程
    继承或经调试信息带出，仅当调用方显式传入 ``allow_env=True``（对应
    ``--password-env`` 命令行开关）时才读取。
    """
    if allow_env:
        from_env = os.environ.get("PAI_CLI_PASSWORD")
        if from_env:
            return security.validate_password(from_env)

    if not sys.stdin.isatty():
        line = sys.stdin.readline()
        if line:
            return security.validate_password(line.rstrip("\r\n"))

    first = getpass.getpass(prompt)
    second = getpass.getpass("再次输入以确认: ")
    if first != second:
        raise ValueError("两次输入的密码不一致")
    return security.validate_password(first)


# ------------------------------------------------------------
# 备份与校验
# ------------------------------------------------------------


def _table_digest(connection: sqlite3.Connection, table: str, columns: str) -> dict:
    rows = connection.execute(
        "SELECT %s FROM %s ORDER BY 1" % (columns, table)
    ).fetchall()
    digest = hashlib.sha256()
    for row in rows:
        digest.update("|".join("" if v is None else str(v) for v in row).encode("utf-8"))
        digest.update(b"\n")
    return {"rows": len(rows), "sha256": digest.hexdigest()}


def database_report(path: str) -> dict:
    """结构与内容摘要：表数量、每表行数与内容哈希。"""
    connection = sqlite3.connect(path, timeout=10.0)
    try:
        tables = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        report: dict = {"path": path, "tables": tables, "detail": {}}
        for table in tables:
            report["detail"][table] = _table_digest(connection, table, "*")
        report["integrity_check"] = connection.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0]
        report["foreign_key_check"] = len(
            connection.execute("PRAGMA foreign_key_check").fetchall()
        )
        report["size_bytes"] = os.path.getsize(path)
        return report
    finally:
        connection.close()


def backup_database(source: str, backup_dir: str) -> str:
    """用 SQLite backup API 做一致性备份（不复制正在使用的文件）。"""
    os.makedirs(backup_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = os.path.join(backup_dir, "memory.db.%s.bak" % stamp)
    if os.path.exists(target):
        raise RuntimeError("备份文件已存在：%s" % target)

    source_connection = sqlite3.connect(source, timeout=10.0)
    target_connection = sqlite3.connect(target)
    try:
        source_connection.backup(target_connection)
        target_connection.commit()
    finally:
        target_connection.close()
        source_connection.close()

    check = database_report(target)
    if check["integrity_check"] != "ok":
        raise RuntimeError("备份完整性校验失败：%s" % check["integrity_check"])
    return target


# ------------------------------------------------------------
# 子命令
# ------------------------------------------------------------


def cmd_migrate(args: argparse.Namespace) -> int:
    service = AuthService()
    db_path = service.db_path
    if not os.path.exists(db_path):
        return _fail("数据库不存在：%s" % db_path, 2)

    before = database_report(db_path)
    print("数据库：%s（%d 字节）" % (db_path, before["size_bytes"]))
    print("迁移前表：%s" % ", ".join(before["tables"]))
    print(
        "迁移前行数：%s"
        % ", ".join(
            "%s=%d" % (name, info["rows"]) for name, info in before["detail"].items()
        )
    )

    target = backup_database(db_path, args.backup_dir)
    backup = database_report(target)
    print("备份完成：%s（%d 字节，integrity=%s）" % (target, backup["size_bytes"], backup["integrity_check"]))

    service.ensure_schema()
    after = database_report(db_path)
    print("迁移后表：%s" % ", ".join(after["tables"]))

    changed = []
    for name, info in before["detail"].items():
        if name not in after["detail"]:
            return _fail("迁移后表丢失：%s" % name, 1)
        if after["detail"][name]["rows"] != info["rows"]:
            changed.append(
                "%s: %d -> %d"
                % (name, info["rows"], after["detail"][name]["rows"])
            )
        elif not name.startswith("users") and not name.startswith("sessions"):
            if after["detail"][name]["sha256"] != info["sha256"]:
                changed.append("%s: 行数相同但内容哈希变化" % name)

    if changed:
        return _fail("迁移改动了既有数据（应只新增表与列）：%s" % "; ".join(changed), 1)

    status = service.legacy_status()
    print("迁移后校验：integrity=%s，外键问题=%d" % (after["integrity_check"], after["foreign_key_check"]))
    print(
        "历史数据归属：会话 %d（无归属 %d）、消息 %d、记忆 %d（无归属 %d）、孤儿消息 %d"
        % (
            status["total_conversations"],
            status["unowned_conversations"],
            status["total_messages"],
            status["total_memories"],
            status["unowned_memories"],
            status["orphan_messages"],
        )
    )
    if status["unowned_conversations"] or status["unowned_memories"]:
        print("提示：历史数据尚未归属，请执行 create-admin（会自动归属到该管理员）")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    service = AuthService()
    try:
        service.ensure_schema()
    except Exception as exc:  # noqa: BLE001
        return _fail("读取失败：%s" % exc)
    status = service.legacy_status()
    print("数据库：%s" % service.db_path)
    print("管理员数量：%d" % service.count_admins())
    for key, value in status.items():
        print("%s: %d" % (key, value))
    return 0


def cmd_create_admin(args: argparse.Namespace) -> int:
    service = AuthService()
    service.ensure_schema()
    if service.count_admins() > 0:
        return _fail(
            "已存在管理员账号，禁止重复 bootstrap；如需新增管理员请使用 create-user --role admin",
            3,
        )
    try:
        password = _read_password_twice("设置管理员密码: ", allow_env=args.password_env)
        user = service.create_user(args.username, password, role="admin")
    except (ValueError, UsernameTaken) as exc:
        return _fail(str(exc), 2)

    claimed = service.claim_legacy_data(user["id"])
    print("[成功] 管理员已创建：%s" % user["username"])
    print(
        "历史数据归属：会话 %d 行、记忆 %d 行"
        % (claimed["conversations"], claimed["memories"])
    )
    return 0


def cmd_create_user(args: argparse.Namespace) -> int:
    service = AuthService()
    service.ensure_schema()
    try:
        password = _read_password_twice("设置用户密码: ", allow_env=args.password_env)
        user = service.create_user(args.username, password, role=args.role)
    except (ValueError, UsernameTaken) as exc:
        return _fail(str(exc), 2)
    print("[成功] 账号已创建：%s（角色 %s）" % (user["username"], user["role"]))
    return 0


def cmd_reset_password(args: argparse.Namespace) -> int:
    service = AuthService()
    service.ensure_schema()
    try:
        password = _read_password_twice("设置新密码: ", allow_env=args.password_env)
        result = service.set_password(args.username, password)
    except (ValueError, UserNotFound) as exc:
        return _fail(str(exc), 2)
    print(
        "[成功] 密码已重置：%s（同时撤销 %d 个会话）"
        % (args.username, result["revoked_sessions"])
    )
    return 0


def cmd_set_active(args: argparse.Namespace, active: bool) -> int:
    service = AuthService()
    service.ensure_schema()
    try:
        result = service.set_active(args.username, active)
    except (ValueError, UserNotFound) as exc:
        return _fail(str(exc), 2)
    action = "启用" if active else "禁用"
    print(
        "[成功] %s账号：%s（撤销 %d 个会话）"
        % (action, args.username, result["revoked_sessions"])
    )
    return 0


def cmd_disable(args: argparse.Namespace) -> int:
    return cmd_set_active(args, False)


def cmd_enable(args: argparse.Namespace) -> int:
    return cmd_set_active(args, True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m auth.cli", description="PAI 认证账号管理（密码交互输入）"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("migrate", help="备份并创建认证表与 user_id 列")
    p.add_argument("--backup-dir", default=".backup_auth", help="备份目录")
    p.set_defaults(func=cmd_migrate)

    p = sub.add_parser("status", help="查看账号与归属状态")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("create-admin", help="创建首个管理员并归属历史数据")
    p.add_argument("--username", required=True)
    p.add_argument(
        "--password-env",
        action="store_true",
        help="从环境变量 PAI_CLI_PASSWORD 读取密码（默认禁用该通道）",
    )
    p.set_defaults(func=cmd_create_admin)

    p = sub.add_parser("create-user", help="创建普通用户")
    p.add_argument("--username", required=True)
    p.add_argument("--role", default="user", choices=("user", "admin"))
    p.add_argument(
        "--password-env",
        action="store_true",
        help="从环境变量 PAI_CLI_PASSWORD 读取密码（默认禁用该通道）",
    )
    p.set_defaults(func=cmd_create_user)

    p = sub.add_parser("reset-password", help="重置密码并撤销其会话")
    p.add_argument("--username", required=True)
    p.add_argument(
        "--password-env",
        action="store_true",
        help="从环境变量 PAI_CLI_PASSWORD 读取密码（默认禁用该通道）",
    )
    p.set_defaults(func=cmd_reset_password)

    p = sub.add_parser("disable", help="禁用账号并撤销其会话")
    p.add_argument("--username", required=True)
    p.set_defaults(func=cmd_disable)

    p = sub.add_parser("enable", help="启用账号")
    p.add_argument("--username", required=True)
    p.set_defaults(func=cmd_enable)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except AuthError as exc:
        return _fail(str(exc), 2)
    except KeyboardInterrupt:
        return _fail("已取消", 130)


if __name__ == "__main__":
    raise SystemExit(main())
