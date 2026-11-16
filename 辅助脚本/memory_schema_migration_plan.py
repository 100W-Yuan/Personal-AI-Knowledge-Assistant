import sqlite3
from pathlib import Path


# 定位项目根目录中的数据库，避免因为运行目录不同而检查错库
PROJECT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_DIR / "memory.db"

TARGET_COLUMNS = [
    "id",
    "key",
    "value",
    "category",
    "source",
    "confidence",
    "reason",
    "created_at",
    "updated_at",
]


def main():
    if not DB_PATH.exists():
        print(f"错误：没有找到数据库文件：{DB_PATH}")
        return

    connection = sqlite3.connect(DB_PATH)

    try:
        cursor = connection.cursor()

        print()
        print("=" * 60)
        print("Memory Schema Migration Plan（数据库结构迁移方案检查）")
        print("=" * 60)
        print(f"数据库位置：{DB_PATH}")

        # 1. 检查当前表结构
        print()
        print("【1. 当前数据库结构】")

        cursor.execute("PRAGMA table_info(memories)")
        columns = cursor.fetchall()

        if not columns:
            print("错误：没有找到 memories 表。")
            return

        current_columns = [column[1] for column in columns]

        for name in current_columns:
            print(f"- {name}")

        # 2. 显示计划中的目标结构
        print()
        print("【2. 目标数据库结构】")

        for name in TARGET_COLUMNS:
            print(f"- {name}")

        # 3. 检查需要新增的字段
        print()
        print("【3. 需要新增的字段】")

        missing_columns = [
            name
            for name in TARGET_COLUMNS
            if name not in current_columns and name != "id"
        ]

        if missing_columns:
            for name in missing_columns:
                print(f"- {name}")
        else:
            print("没有需要新增的字段。")

        # 4. 检查目标之外的现有字段
        print()
        print("【4. 目标之外的现有字段】")

        extra_columns = [
            name
            for name in current_columns
            if name not in TARGET_COLUMNS
        ]

        if extra_columns:
            for name in extra_columns:
                print(f"- {name}")
        else:
            print("没有。")

        # 5. 检查长期记忆数量
        print()
        print("【5. 当前数据数量】")

        cursor.execute("SELECT COUNT(*) FROM memories")
        memory_count = cursor.fetchone()[0]
        print(f"长期记忆数量：{memory_count}")

        print()
        print("=" * 60)
        print("结构差异检查完成。")
        print("本脚本只读，不修改数据库。")
        print("=" * 60)

    finally:
        connection.close()


if __name__ == "__main__":
    main()
