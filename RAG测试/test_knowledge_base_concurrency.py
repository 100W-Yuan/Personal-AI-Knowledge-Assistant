# -*- coding: utf-8 -*-
"""RAG 7.6：KnowledgeBase 首次并发初始化专项测试。

验证（真实线程并发，非顺序调用）：
- 两个线程通过 threading.Barrier 同时触发首次检索；
- Embedding 工厂总调用次数严格等于 1；
- 索引只构建一次（chunk_count 稳定）；
- 两个线程均取得符合接口约定的结果；
- 初始化失败时并发调用不重复执行工厂（仍为 1 次）；
- 普通构造仍是零加载。

全程 stub（不加载真实模型）、临时目录（虚构知识）、零网络、不读正式库。
"""

import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.knowledge_base import KnowledgeBase   # noqa: E402

DOC_ONE = "智能体项目已经实现了工具调用和长期记忆功能。"


class SlowStubEmbedding:
    """带小延时的确定性 stub（拉长初始化窗口，提高并发重叠概率）。"""

    def __init__(self, delay=0.05):
        self.delay = delay

    def encode(self, text):
        time.sleep(self.delay)
        return [
            float(text.count("智能") + text.count("项目")),
            0.0,
            float(text.count("工具")),
        ]


def prepare_dir(tmp_path):
    (tmp_path / "one.md").write_text(DOC_ONE, encoding="utf-8")


def run_two_threads(kb, query):
    """两个线程用 Barrier 同时发起首次检索；返回 (结果列表, 异常列表)。"""
    barrier = threading.Barrier(2)
    results = [None, None]
    errors = [None, None]

    def worker(index):
        try:
            barrier.wait(timeout=10)
            results[index] = kb.search(query)
        except Exception as exc:  # noqa: BLE001
            errors[index] = exc

    threads = [threading.Thread(target=worker, args=(i,)) for i in (0, 1)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    return results, errors


# ----------------------------------------------------------------------
# 1. 并发首次检索：工厂严格执行一次、索引只建一次、双方结果有效
# ----------------------------------------------------------------------

def test_concurrent_first_search_single_init(tmp_path):
    prepare_dir(tmp_path)
    stats = {"calls": 0}

    def factory():
        stats["calls"] += 1
        return SlowStubEmbedding()

    kb = KnowledgeBase(tmp_path, embedding_factory=factory)
    assert stats["calls"] == 0            # 构造零加载

    results, errors = run_two_threads(kb, "项目的智能功能")

    assert errors == [None, None], f"并发检索出现异常：{errors}"
    assert stats["calls"] == 1, f"工厂被调用 {stats['calls']} 次（应为 1）"
    assert kb._ready is True
    assert kb.file_count == 1
    assert kb.chunk_count >= 1

    for r in results:
        assert r is not None
        assert r["ok"] is True
        assert r["count"] >= 1
        assert r["results"][0]["source"] == "one.md"


# ----------------------------------------------------------------------
# 2. 索引只构建一次（并发后再次检索不新增构建）
# ----------------------------------------------------------------------

def test_index_built_once_under_concurrency(tmp_path):
    prepare_dir(tmp_path)
    stats = {"calls": 0}

    def factory():
        stats["calls"] += 1
        return SlowStubEmbedding(delay=0.0)

    kb = KnowledgeBase(tmp_path, embedding_factory=factory)
    run_two_threads(kb, "项目的智能功能")

    chunks_after_concurrent = kb.chunk_count
    kb.search("工具调用")

    assert stats["calls"] == 1
    assert kb.chunk_count == chunks_after_concurrent


# ----------------------------------------------------------------------
# 3. 初始化失败：并发调用不重复执行工厂（严格 1 次）+ 错误缓存
# ----------------------------------------------------------------------

def test_concurrent_init_failure_factory_once(tmp_path):
    prepare_dir(tmp_path)
    stats = {"calls": 0}

    def factory():
        stats["calls"] += 1
        time.sleep(0.05)                  # 拉长失败窗口
        raise RuntimeError("stub 初始化失败（并发注入）")

    kb = KnowledgeBase(tmp_path, embedding_factory=factory)
    results, errors = run_two_threads(kb, "项目的智能功能")

    assert errors == [None, None]
    assert stats["calls"] == 1, f"失败场景工厂被调用 {stats['calls']} 次（应为 1）"
    for r in results:
        assert r is not None
        assert r["ok"] is False
        assert r["stage"] == "init"
        assert "stub 初始化失败" in r["error"]


# ----------------------------------------------------------------------
# 4. 普通构造仍然零加载
# ----------------------------------------------------------------------

def test_construct_still_zero_load(tmp_path):
    prepare_dir(tmp_path)
    stats = {"calls": 0}

    def factory():
        stats["calls"] += 1
        return SlowStubEmbedding()

    kb = KnowledgeBase(tmp_path, embedding_factory=factory)
    assert stats["calls"] == 0
    assert kb._ready is False
