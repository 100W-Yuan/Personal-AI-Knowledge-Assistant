# -*- coding: utf-8 -*-
"""POST /chat 的 HTTP 层契约测试（FastAPI TestClient，离线）。

此前 /chat 只在函数级（main.chat 直调）验证过；本文件补齐 HTTP 层验证：
- 用 FakeAgent 替换真实 Agent（monkeypatch main.get_agent），
  不初始化真实模型客户端、不访问正式记忆数据库；
- 依赖 自动测试脚本/conftest.py 的离线隔离（假 dotenv / 假 Key / 空壳 OpenAI）。

断言只覆盖「HTTP 请求→路由→响应」的契约，不重复 Agent 内部逻辑测试。
"""

import pytest
from fastapi.testclient import TestClient

import main


client = TestClient(main.app)

SOURCE_ITEM = {
    "source": "faq.md",
    "chunk_id": 0,
    "score": 0.9,
    "chunk": "依据片段",
}


class FakeAgent:
    def __init__(self, result):
        self.result = result
        self.messages = []

    def run(self, message, conversation_id=None, *, user_id=None):
        self.messages.append(message)
        return self.result


class BoomAgent:
    def run(self, message, conversation_id=None, *, user_id=None):
        raise RuntimeError("敏感 sk-secret-123 / C:/Users/Doro/secret")


@pytest.fixture(autouse=True)
def isolated_chat_memory(tmp_path, monkeypatch):
    """/chat 会先写会话记录：把 Memory 锁定在 tmp_path。

    不隔离的话 get_memory() 会打开工程根目录的正式 memory.db，
    测试就会对真实用户数据执行迁移。
    """
    from memory import Memory

    memory = Memory(db_path=str(tmp_path / "memory.db"))

    monkeypatch.setattr(main, "get_memory", lambda: memory)

    return memory


def test_chat_http_returns_success_answer_sources(monkeypatch):
    monkeypatch.setattr(
        main,
        "get_agent",
        lambda: FakeAgent(
            {"success": True, "answer": "答案", "sources": [SOURCE_ITEM]}
        ),
    )

    response = client.post("/chat", json={"message": "问题"})

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["answer"] == "答案"
    assert body["sources"] == [SOURCE_ITEM]
    # 来源条目字段符合契约。
    assert set(body["sources"][0]) == {"source", "chunk_id", "score", "chunk"}


def test_chat_http_no_sources_returns_empty_list(monkeypatch):
    monkeypatch.setattr(
        main,
        "get_agent",
        lambda: FakeAgent({"success": True, "answer": "答案"}),
    )

    response = client.post("/chat", json={"message": "问题"})

    assert response.status_code == 200
    assert response.json()["sources"] == []


def test_chat_http_legacy_string_result(monkeypatch):
    monkeypatch.setattr(main, "get_agent", lambda: FakeAgent("字符串答案"))

    response = client.post("/chat", json={"message": "问题"})

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["answer"] == "字符串答案"
    assert body["sources"] == []


def test_chat_http_failure_keeps_contract(monkeypatch):
    monkeypatch.setattr(
        main,
        "get_agent",
        lambda: FakeAgent({"success": False, "answer": "失败", "error": "工具失败"}),
    )

    response = client.post("/chat", json={"message": "问题"})

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is False
    assert body["answer"] == "执行失败：工具失败"
    assert body["error"] == "工具失败"
    assert body["sources"] == []


def test_chat_http_empty_message_is_422(monkeypatch):
    response = client.post("/chat", json={"message": "   "})

    assert response.status_code == 422
    assert response.json()["detail"]["type"] == "invalid_request"


def test_chat_http_internal_error_does_not_leak(monkeypatch):
    monkeypatch.setattr(main, "get_agent", lambda: BoomAgent())

    response = client.post("/chat", json={"message": "问题"})

    assert response.status_code == 502
    detail = response.json()["detail"]
    assert detail["type"] == "internal_error"
    text = str(detail)
    assert "sk-secret-123" not in text
    assert "C:/Users/Doro/secret" not in text
    assert "BoomAgent" not in text


def test_chat_http_does_not_touch_real_agent(monkeypatch):
    """确认请求路径只经由 FakeAgent，未触发真实 get_agent 的缓存/构造。"""
    fake = FakeAgent({"success": True, "answer": "ok"})
    monkeypatch.setattr(main, "get_agent", lambda: fake)

    response = client.post("/chat", json={"message": "问题"})

    assert response.status_code == 200
    assert fake.messages == ["问题"]
