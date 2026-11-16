# -*- coding: utf-8 -*-
"""第 1 阶段补充：长时记忆持久化重读与端到端链路测试。

测试一：新实例重新读取持久化记忆
    保存（第一实例）→ close → 新建第二实例（同库路径）→ find_memory
    断言 key/value/category/reason 与保存值一致（非同一实例内读取）。

测试二：端到端链路（仅模拟模型响应，真实入口串联）
    用户消息 → extract_memories（真实入口 + FakeClient）
    → save_extracted_memories（真实）→ close → 新 Memory 实例（模拟重启）
    → retrieve_long_term_memories（真实）→ agent.run() 全流程 →
    断言 Planner 实际请求文本包含预期记忆（注入环节，而非仅查询返回值）。

约束：不调用真实 API（无网络、无费用）；全程临时数据库；
      不触碰正式 memory.db；不修改任何生产代码。
"""

import io
import contextlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent import Agent      # noqa: E402
from memory import Memory    # noqa: E402

DUMMY = dict(api_key="dummy", base_url="https://example.invalid", model="dummy")


class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)


class _Resp:
    def __init__(self, content):
        self.choices = [_Choice(content)]


class _Completions:
    def __init__(self, script):
        self._script = list(script)
        self.calls = []

    def create(self, model=None, messages=None, **kw):
        self.calls.append(messages)
        return _Resp(self._script.pop(0))


class _Chat:
    def __init__(self, script):
        self.completions = _Completions(script)


class FakeClient:
    """仅模拟模型响应；不替换任何被测业务逻辑。"""

    def __init__(self, script):
        self.chat = _Chat(script)

    @property
    def calls(self):
        return self.chat.completions.calls


# =====================================================================
# 测试一：新实例重新读取持久化记忆
# =====================================================================

def test_persisted_memory_read_by_new_instance(tmp_path):
    db_path = tmp_path / "memory.db"

    # 1) 第一个实例：保存
    first = Memory(db_path=str(db_path))
    saved = first.save_memory(
        key="major",
        value="计算机科学与技术",
        category="profile",
        reason="用户明确说明专业",
        source="user",
        confidence=1.0, user_id="test-user"
    )
    assert saved == "add"

    # 2) 正常关闭
    first.close()

    # 3) 第二个全新实例（同一数据库文件）——模拟程序重启
    second = Memory(db_path=str(db_path))
    try:
        # 4) 通过真实读取接口查回
        record = second.find_memory("major", user_id="test-user")
        assert record is not None, "新实例未能读取到已持久化的记忆"
        assert record["value"] == "计算机科学与技术"
        assert record["category"] == "profile"
        assert record["reason"] == "用户明确说明专业"
        assert record["source"] == "user"

        # 5) 列表接口同样可见（非单一方法偶然命中）
        all_memories = second.get_memories(user_id="test-user")
        assert any(m["key"] == "major" for m in all_memories)
    finally:
        second.close()  # 6) 清理临时资源


# =====================================================================
# 测试二：端到端链路（提取 → 保存 → 重启 → 检索 → 注入）
# =====================================================================

def test_e2e_extract_save_restart_retrieve_inject(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db_path = tmp_path / "memory.db"

    agent = Agent(**DUMMY)
    agent.memory = Memory(db_path=str(db_path))

    # ---- 环节 1：提取（真实入口；仅模拟模型返回的结构化结果）----
    agent.client = FakeClient([json.dumps({
        "should_remember": True,
        "memories": [
            {"key": "major", "value": "计算机科学与技术",
             "category": "profile", "reason": "用户明确说明专业"},
            {"key": "career_goal", "value": "机器人工程师",
             "category": "goal", "reason": "用户明确说明职业目标"},
        ],
    })])
    extraction = agent.extract_memories(
        "我是计算机专业的学生，我的职业目标是机器人工程师。"
    )
    assert extraction["should_remember"] is True
    assert len(extraction["memories"]) == 2

    # ---- 环节 2：保存（真实入口）----
    result = agent.save_extracted_memories(extraction)
    assert result == {"added": 2, "updated": 0, "ignored": 0}

    # ---- 环节 3：关闭第一实例；环节 4：新实例（模拟重启）----
    agent.memory.close()
    agent.memory = Memory(db_path=str(db_path))

    # ---- 环节 5：检索（真实入口 + 新实例）----
    cand = agent.retrieve_long_term_memories("我是什么专业的")
    keys = [m["key"] for m in cand]
    assert "major" in keys, f"新实例检索失败：{keys}"
    major_record = next(m for m in cand if m["key"] == "major")
    assert major_record["value"] == "计算机科学与技术"
    assert major_record["category"] == "profile"

    # ---- 环节 6：注入（真实 run 全流程；断言模型实际请求文本）----
    agent.client = FakeClient([
        json.dumps({"should_remember": False, "memories": []}),
        json.dumps({"plan": [{"description": "回答", "type": "reasoning"}]}),
        "中间推理",
        "最终答案。",
    ])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run_result = agent.run("我是什么专业的", user_id="test-user")
    assert run_result["success"] is True

    planner_prompt = agent.client.calls[1][0]["content"]
    assert "- major = 计算机科学与技术" in planner_prompt, (
        "记忆未被注入后端模型请求（注入环节验证失败）"
    )
    assert "（分类：profile）" in planner_prompt

    agent.memory.close()
