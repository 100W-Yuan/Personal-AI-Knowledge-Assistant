# -*- coding: utf-8 -*-
"""真实语义 Embedding 的离线冒烟测试（默认跳过）。

设计目标：
- 证明 rag.embedding.Embedding 是**真实语义向量**（BAAI/bge-small-zh-v1.5），
  而非字符编码 / 哈希等演示实现；
- 全程离线：强制 HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE，模型走本地缓存，
  零网络、零 API 费用；
- 默认**不运行**（`RUN_REAL_EMBEDDING != "1"` 时跳过），避免默认离线套件
  加载真实模型；只有显式授权时才加载本地缓存模型。

运行方式（显式授权）：
    RUN_REAL_EMBEDDING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        python -B -m pytest 自动测试脚本/test_embedding_offline_smoke.py \
        --basetemp=<唯一新目录> -o tmp_path_retention_policy=all -p no:cacheprovider
"""

import math
import os

import pytest

# 模块加载即强制离线，杜绝任何下载行为（即使调用方忘了设环境变量）。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# 默认关：只有显式设置 RUN_REAL_EMBEDDING=1 才加载真实模型。
RUN_REAL = os.environ.get("RUN_REAL_EMBEDDING") == "1"


def _cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


@pytest.mark.skipif(
    not RUN_REAL,
    reason="需显式设置 RUN_REAL_EMBEDDING=1 才加载真实（本地缓存）Embedding",
)
def test_real_embedding_dimension_and_normalization():
    from rag.embedding import Embedding

    embedding = Embedding()
    v1 = embedding.encode("苹果是一种常见的水果")
    v2 = embedding.encode("香蕉也是一种水果，很好吃")

    # SentenceTransformer.encode 返回 numpy 数组；此处只关心长度与数值语义。
    v1 = list(v1)
    v2 = list(v2)
    assert v1
    assert len(v1) == len(v2)
    assert len(v1) >= 128  # bge-small-zh-v1.5 为 512 维；下限仅作防御
    # normalize_embeddings=True 时向量应为单位向量
    norm = math.sqrt(sum(x * x for x in v1))
    assert norm == pytest.approx(1.0, abs=1e-4)


@pytest.mark.skipif(
    not RUN_REAL,
    reason="需显式设置 RUN_REAL_EMBEDDING=1 才加载真实（本地缓存）Embedding",
)
def test_real_embedding_is_semantic():
    from rag.embedding import Embedding

    embedding = Embedding()
    a = embedding.encode("苹果是一种常见的水果")
    b = embedding.encode("香蕉也是一种水果，很好吃")
    c = embedding.encode("今天天气晴朗适合跑步")

    # 语义相关句子的相似度应显著高于无关句子。
    assert _cos(a, b) > _cos(a, c)
