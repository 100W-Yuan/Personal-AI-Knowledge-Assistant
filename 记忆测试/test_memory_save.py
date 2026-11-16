import sys
import tempfile
from pathlib import Path

# 获取项目根目录：测试文件的上一级目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 把项目根目录加入 Python 的模块搜索路径
sys.path.insert(0, str(PROJECT_ROOT))

from memory import Memory


def main():
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_memory.db"
        memory = Memory(db_path=str(db_path))

        print("=" * 50)
        print("Memory 保存功能测试")
        print("=" * 50)

        # 测试 1：新增记忆
        result_add = memory.save_memory(
            key="career_goal",
            value="大模型落地工程师",
            category="goal",
            reason="用户明确表达了自己的职业目标", user_id="test-user"
        )
        print("新增测试：", result_add)

        # 测试 2：保存相同内容，应该忽略
        result_ignore = memory.save_memory(
            key="career_goal",
            value="大模型落地工程师",
            category="goal",
            reason="重复保存测试", user_id="test-user"
        )
        print("忽略测试：", result_ignore)

        # 测试 3：相同 key、不同 value，应该更新
        result_update = memory.save_memory(
            key="career_goal",
            value="AI 应用工程师",
            category="goal",
            reason="测试更新记忆", user_id="test-user"
        )
        print("更新测试：", result_update)

        # 测试 4：确认查询能读出新字段
        item = memory.find_memory("career_goal", user_id="test-user")
        print("最终记忆：", item)

        assert result_add == "add"
        assert result_ignore == "ignore"
        assert result_update == "update"
        assert item["value"] == "AI 应用工程师"
        assert item["category"] == "goal"
        assert item["reason"] == "测试更新记忆"

        print()
        print("全部测试通过。")
        print("测试使用临时数据库，不修改项目的 memory.db。")


if __name__ == "__main__":
    main()
