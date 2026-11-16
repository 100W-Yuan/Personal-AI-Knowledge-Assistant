# -*- coding: utf-8 -*-
"""统一文档格式清单（阶段二 P0-1）。

本模块是**唯一的**格式权威来源：受支持后缀、枚举分组顺序都只在这里定义一次，
``main``（上传校验）、``knowledge_store``（枚举/删除/去重）、
``rag.knowledge_base``（索引枚举）一律引用本模块，不得再写后缀字面量。

设计边界：

- 只描述「哪些后缀受支持」，**不负责解析**：各格式的解析实现见
  ``rag.loader``（``SUFFIX_LOADERS``），本模块不描述、也不限制解析能力；
- 不含任何业务逻辑、不 import 任何项目模块（避免层次倒置与循环导入）；
- 顺序即语义：``GROUP_ORDER`` 决定索引枚举的分组顺序，**MD 必须排在 TXT 前**，
  与阶段一 ``_iter_knowledge_files`` 的既有排序逐字节一致。
"""

import os


# 受支持的后缀（小写，含点）。顺序即枚举分组顺序，不要随意调整。
SUPPORTED_SUFFIXES = (".md", ".txt", ".pdf", ".docx")

# 枚举分组顺序，与 SUPPORTED_SUFFIXES 同源（不要单独维护第二份顺序）。
GROUP_ORDER = SUPPORTED_SUFFIXES


def suffix_of(filename):
    """返回 ``filename`` 的小写后缀（含点）；无后缀或不可解析时返回空串。

    大小写不敏感：``"A.MD"`` → ``".md"``。
    点开头的文件名（如 ``".gitignore"``）与 ``splitext`` 一致，视为无后缀。
    """
    try:
        path = os.fspath(filename)
    except TypeError:
        return ""
    if not isinstance(path, str):
        return ""
    return os.path.splitext(path)[1].lower()


def is_supported_suffix(suffix):
    """判断**后缀**（含点，如 ``".pdf"``）是否受支持；大小写不敏感。

    参数必须是后缀而不是完整文件名；未知后缀、空串、非字符串一律返回 False。
    """
    if not isinstance(suffix, str) or not suffix:
        return False
    return suffix.lower() in SUPPORTED_SUFFIXES
