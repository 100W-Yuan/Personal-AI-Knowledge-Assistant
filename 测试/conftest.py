"""`测试/` 目录的路径引导（PAI-UI-LOGIN-R1）。

在本目录内调用 pytest 时，项目根不在 `sys.path` 上，`import main` 会失败。
这里只做一件事：把项目根追加到 `sys.path`。

其余隔离能力**全部复用根 `conftest.py`**，本文件不重复安装任何守卫：
- 正式数据库（memory.db / data/business.db）连接拦截与指纹哨兵；
- 非回环网络、DNS、UDP、HTTP 传输层与模型 SDK 请求拦截；
- dotenv 假模块与假 DEEPSEEK_API_KEY。

因此本目录下的测试天然满足任务书「不得触发真实模型 API 调用」。
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT_CONFTEST = Path(__file__).resolve().parent.parent / "conftest.py"
_PROJECT_ROOT = str(_ROOT_CONFTEST.parent)

if _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)
