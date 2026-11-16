import sqlite3
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_DIR / "memory.db"
BACKUP_PATH = PROJECT_DIR / "memory_before_schema_migration.db"


def main():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"找不到原数据库：{DB_PATH}")

    if not BACKUP_PATH.exists():
        raise FileNotFoundError(f"找不到迁移前备份：{BACKUP_PATH}")

    connection = sqlite3.connect(DB_PATH)

    try:
        connection.execute("PRAGMA busy_timeout = 5000")

        integrity = connection.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0]

        if integrity != "ok":
            raise RuntimeError(f"原数据库完整性检查失败：{integrity}")

        columns = [
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(memories)"
            ).fetchall()
        ]

        expected_columns = [
            "id", "key", "value", "source",
            "confidence", "created_at", "updated_at"
        ]

        if columns != expected_columns:
            raise RuntimeError(
                "当前表结构与预期不一致，停止迁移。\n"
                f"实际字段：{columns}"
            )

        count_before = connection.execute(
            "SELECT COUNT(*) FROM memories"
        ).fetchone()[0]

        if count_before != 5:
            raise RuntimeError(
                f"预期 5 条记录，实际为 {count_before} 条，停止迁移。"
            )

        print("迁移前检查通过。")
        print(f"原数据库：{DB_PATH}")
        print(f"备份文件：{BACKUP_PATH}")
        print(f"原有记忆数量：{count_before}")

        with connection:
            connection.execute("""
                ALTER TABLE memories
                ADD COLUMN category TEXT NOT NULL
                DEFAULT 'profile'
            """)

            connection.execute("""
                ALTER TABLE memories
                ADD COLUMN reason TEXT NOT NULL
                DEFAULT '历史记忆：迁移前未记录保存原因'
            """)

            connection.execute("""
                UPDATE memories
                SET category = CASE key
                    WHEN 'name' THEN 'identity'
                    WHEN 'major' THEN 'profile'
                    WHEN 'career_goal' THEN 'goal'
                    WHEN 'test_name' THEN 'profile'
                    ELSE 'profile'
                END
            """)

            connection.execute("""
                UPDATE memories
                SET reason = CASE key
                    WHEN 'name'
                        THEN '历史记录：保存的姓名字段；迁移时未重新核实'
                    WHEN 'major'
                        THEN '历史记录：保存的专业字段；迁移时未重新核实'
                    WHEN 'career_goal'
                        THEN '历史记录：保存的职业目标字段；迁移时未重新核实'
                    WHEN 'test_name'
                        THEN '历史测试数据：仅用于测试，不应视为已核实的用户事实'
                    ELSE '历史记忆：迁移前未记录保存原因'
                END
            """)

            count_after = connection.execute(
                "SELECT COUNT(*) FROM memories"
            ).fetchone()[0]

            if count_after != count_before:
                raise RuntimeError(
                    f"记录数量变化：迁移前 {count_before}，"
                    f"迁移后 {count_after}，已回滚本次事务。"
                )

            new_columns = [
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(memories)"
                ).fetchall()
            ]

            if "category" not in new_columns or "reason" not in new_columns:
                raise RuntimeError("新字段验证失败，已回滚本次事务。")

        integrity_after = connection.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0]

        print()
        print("【迁移完成】")
        print("新增字段：category、reason")
        print(f"迁移后记忆数量：{count_after}")
        print(f"数据库完整性：{integrity_after}")

        print()
        print("【迁移后的记忆】")

        rows = connection.execute("""
            SELECT id, key, value, category, reason
            FROM memories
            ORDER BY id
        """).fetchall()

        for row in rows:
            print(
                f"id={row[0]} | key={row[1]} | value={row[2]} "
                f"| category={row[3]} | reason={row[4]}"
            )

        if integrity_after != "ok":
            print("警告：迁移后的完整性检查未通过，请停止后续操作。")

    finally:
        connection.close()


if __name__ == "__main__":
    main()
