# -*- coding: utf-8 -*-
"""KnowledgeBase 健壮性测试（stub，零真实模型、零网络）。

覆盖「文档格式错误 / 解析失败」的容错语义：
- 单个坏文档不应拖垮整个索引；
- 全部坏文档时给出明确的 init 失败诊断。
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.knowledge_base import KnowledgeBase  # noqa: E402


class StubEmbedding:
    def encode(self, text):
        return [1.0, 0.0] if "项目" in text else [0.0, 1.0]


def test_one_bad_file_does_not_break_index(tmp_path):
    (tmp_path / "good.md").write_text("智能体项目实现了工具调用", encoding="utf-8")
    # 无效 UTF-8 字节：TextLoader 读取时抛 UnicodeDecodeError。
    (tmp_path / "bad.md").write_bytes(b"\xff\xfe\x00garbage")

    kb = KnowledgeBase(str(tmp_path), embedding_factory=StubEmbedding)
    result = kb.search("项目")

    assert result["ok"] is True
    assert kb.file_count == 1                      # 只有 good.md 成功加载
    assert all(item["source"] == "good.md" for item in result["results"])


def test_all_bad_files_is_init_error(tmp_path):
    (tmp_path / "a.md").write_bytes(b"\xff\xfe\x00")
    (tmp_path / "b.md").write_bytes(b"\x80\x81\x82\x83")

    kb = KnowledgeBase(str(tmp_path), embedding_factory=StubEmbedding)
    result = kb.search("项目")

    assert result["ok"] is False
    assert result["stage"] == "init"
    assert "所有知识文档均无法加载或解析" in result["error"]
