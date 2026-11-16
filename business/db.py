
"""
企业业务数据库管理模块。

负责：
1. 确定业务数据库文件路径
2. 创建 SQLite 数据库连接
3. 开启外键约束
4. 初始化业务数据库表结构

注意：
本模块只管理业务数据库，不操作个人 AI 记忆数据库。
"""

import sqlite3
from pathlib import Path

from business.schema import SCHEMA_SQL


# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 业务数据库目录
DATA_DIR = PROJECT_ROOT / "data"

# 业务数据库文件
DATABASE_PATH = DATA_DIR / "business.db"


def get_connection():
    """创建并返回业务数据库连接。"""

    # 确保 data 目录存在
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 建立 SQLite 连接
    connection = sqlite3.connect(DATABASE_PATH)

    # 将查询结果转换为可以通过字段名访问的对象
    connection.row_factory = sqlite3.Row

    # 开启外键约束
    connection.execute("PRAGMA foreign_keys = ON")

    return connection


def init_database():
    """初始化业务数据库表结构。"""

    connection = get_connection()

    try:
        # 执行 schema.py 中定义的全部建表语句
        connection.executescript(SCHEMA_SQL)

        # 提交数据库结构变更
        connection.commit()

    except Exception:
        # 出错时回滚本次事务
        connection.rollback()
        raise

    finally:
        # 无论成功还是失败，都关闭连接
        connection.close()
