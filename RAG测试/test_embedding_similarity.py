# -*- coding: utf-8 -*-
"""RAG 第 6 阶段（真实 Embedding）6.4：验证真实 Embedding 的语义相似度。

目标：验证本地模型 BAAI/bge-small-zh-v1.5 的嵌入能体现基本语义相似关系——
不要求固定数值，只要求 sim(A,B) > sim(A,C)：
  A：我的职业目标是什么？      （职业目标）
  B：我未来想从事什么工作？    （职业目标——与 A 语义相近）
  C：今天晚上吃什么？          （饮食话题——与 A 语义明显不同）

方法：SentenceTransformer 本地加载（不调用任何云端 API）、
      normalize_embeddings=True（归一化向量）、NumPy 点积 = cosine 相似度。

两种运行方式：
  python "RAG测试/test_embedding_similarity.py"   （脚本模式：打印全部指标 + 退出码）
  pytest "RAG测试/test_embedding_similarity.py"   （测试模式）
"""

import sys

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-zh-v1.5"

TEXT_A = "我的职业目标是什么？"
TEXT_B = "我未来想从事什么工作？"
TEXT_C = "今天晚上吃什么？"


def compute_similarities():
    """返回 (模型名, 向量维度, sim_AB, sim_AC)。"""
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        [TEXT_A, TEXT_B, TEXT_C],
        normalize_embeddings=True,
    )

    dim = embeddings.shape[1]
    vec_a, vec_b, vec_c = embeddings[0], embeddings[1], embeddings[2]

    # 归一化向量的点积即 cosine 相似度
    sim_ab = float(np.dot(vec_a, vec_b))
    sim_ac = float(np.dot(vec_a, vec_c))

    return MODEL_NAME, dim, sim_ab, sim_ac


# ----------------------------------------------------------------------
# 测试模式（pytest）
# ----------------------------------------------------------------------

def test_ab_closer_than_ac():
    model_name, dim, sim_ab, sim_ac = compute_similarities()

    print(f"\n模型: {model_name}")
    print(f"向量维度: {dim}")
    print(f"A-B 相似度: {sim_ab:.6f}")
    print(f"A-C 相似度: {sim_ac:.6f}")
    print(f"是否满足 A-B > A-C: {sim_ab > sim_ac}")

    assert sim_ab > sim_ac, (
        f"语义相似度不符合预期：A-B={sim_ab:.6f} 应大于 A-C={sim_ac:.6f}"
    )


# ----------------------------------------------------------------------
# 脚本模式（直接运行：打印指标并返回退出码）
# ----------------------------------------------------------------------

def main():
    model_name, dim, sim_ab, sim_ac = compute_similarities()
    satisfied = sim_ab > sim_ac

    print("=" * 52)
    print("RAG 6.4 真实 Embedding 语义相似度验证")
    print("=" * 52)
    print(f"模型名称：{model_name}")
    print(f"向量维度：{dim}")
    print(f"A：{TEXT_A}")
    print(f"B：{TEXT_B}")
    print(f"C：{TEXT_C}")
    print("-" * 52)
    print(f"A-B 相似度：{sim_ab:.6f}")
    print(f"A-C 相似度：{sim_ac:.6f}")
    print(f"是否满足 A-B > A-C：{satisfied}")
    print("=" * 52)

    return 0 if satisfied else 1


if __name__ == "__main__":
    sys.exit(main())
