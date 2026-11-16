
import os
import sys
import json
import tempfile
from pathlib import Path

from dotenv import load_dotenv

# 获取项目根目录（测试文件所在目录的上一级）
PROJECT_DIR = Path(__file__).resolve().parent.parent

# 将项目根目录加入 Python 模块搜索路径
sys.path.insert(0, str(PROJECT_DIR))

# 加载项目根目录下的 .env 文件
load_dotenv(PROJECT_DIR / ".env")

# 导入项目中的 Agent
from agent import Agent


def main():
    # 读取 DeepSeek API 密钥
    api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        raise RuntimeError(
            "没有找到 DEEPSEEK_API_KEY，请检查项目根目录下的 .env 文件"
        )

    # 允许的记忆分类
    allowed_categories = {
        "identity",
        "profile",
        "preference",
        "goal",
        "project",
        "skill",
    }

    # 使用临时目录，避免操作正式数据库
    original_dir = os.getcwd()

    try:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:

            agent = Agent(
                api_key=api_key,
                base_url="https://api.deepseek.com",
                model="deepseek-flash",
            )

            # 用于测试的用户信息
            user_message = (
                "我叫钦，是计算机专业的学生，"
                "我的职业目标是大模型落地工程师。"
            )

            print("开始测试长期记忆提取……")

            # 只提取，不调用保存记忆的函数
            result = agent.extract_memories(user_message)

            print("\n========== 模型返回结果 ==========")
            print(json.dumps(result, ensure_ascii=False, indent=2))

            if not isinstance(result, dict):
                raise TypeError("提取结果不是字典")

            memories = result.get("memories", [])

            if not isinstance(memories, list):
                raise TypeError("memories 字段不是列表")

            required_fields = {
                "key",
                "value",
                "category",
                "reason",
            }

            print("\n========== 字段与分类检查 ==========")

            all_valid = True

            for index, memory in enumerate(memories, start=1):
                if not isinstance(memory, dict):
                    print(f"第 {index} 条记忆格式错误：不是字典")
                    all_valid = False
                    continue

                missing_fields = required_fields - set(memory.keys())

                if missing_fields:
                    print(
                        f"第 {index} 条记忆缺少字段："
                        f"{sorted(missing_fields)}"
                    )
                    all_valid = False
                    continue

                category = memory["category"]

                if category not in allowed_categories:
                    print(
                        f"第 {index} 条记忆分类不合法：{category!r}"
                    )
                    all_valid = False
                    continue

                print(
                    f"第 {index} 条检查通过："
                    f"key={memory['key']}, "
                    f"category={category}"
                )

            if memories and all_valid:
                print("\n测试通过：所有记忆字段和分类均合法。")
            elif not memories:
                print("\n测试未通过：模型没有返回任何记忆。")
            else:
                print("\n测试未通过：存在字段缺失或分类错误。")

    finally:
        os.chdir(original_dir)
        agent = None

if __name__ == "__main__":
    main()
