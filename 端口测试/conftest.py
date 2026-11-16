# -*- coding: utf-8 -*-
"""端口测试/ 的根级测试守卫加载器（阶段 1-R2）。

为什么需要这个文件
------------------
pytest 的 rootdir 由调用参数推导：在项目根目录执行
``pytest 端口测试/test_x.py`` 时 rootdir 是项目根，本目录会自动继承
根级 ``conftest.py``；但若在本目录内直接执行 ``pytest``，rootdir 会退化
为 ``端口测试``，根级 ``conftest.py`` 不再被加载，数据库守卫随之失效。

本文件只做两件事：把根级 ``conftest.py`` 加载进来并复用它的哨兵
fixture，使两种调用方式下的保护一致；以及注册一个 ``pytest_configure``
钩子做生命周期兜底。**不定义任何新断言、不改动任何测试行为。**

不注册 ``pytest_unconfigure`` / ``pytest_terminal_summary``：卸载与摘要
仍由根级 conftest 统一负责，避免重复输出，也避免在他方守卫接管时抢先还原。
"""

import sys
from pathlib import Path

_ROOT_CONFTEST = Path(__file__).resolve().parent.parent / "conftest.py"
_GUARD_MODULE_NAME = "_wb_db_guard"

if _GUARD_MODULE_NAME not in sys.modules:
    import importlib.util

    _spec = importlib.util.spec_from_file_location(
        _GUARD_MODULE_NAME, _ROOT_CONFTEST
    )
    _module = importlib.util.module_from_spec(_spec)
    # 先登记再执行：根 conftest 内部用的是 setdefault，必须命中同一对象
    sys.modules[_GUARD_MODULE_NAME] = _module
    _spec.loader.exec_module(_module)

# 在本目录内调用 pytest 时，项目根不在 sys.path 上，`import main` /
# `import memory` 都会失败。追加（而不是插入）项目根，保持测试目录优先。
_PROJECT_ROOT = str(_ROOT_CONFTEST.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)

_guard = sys.modules[_GUARD_MODULE_NAME]

# ------------------------------------------------------------
# 阶段 1-R2.2：`sys.modules` 里有模块 ≠ 守卫已安装
#
# 根 conftest.py 的 pytest_unconfigure 会在每次会话结束时卸载四类守卫，
# 而模块对象仍留在 sys.modules 里。同进程再次 pytest.main() 时（尤其是
# 「先在项目根跑、再进本目录跑」这种 rootdir 退化组合），上面的
# `if ... not in sys.modules` 会判定为「已加载」而跳过 exec_module，
# 模块级的「导入即安装」随之不执行 —— 整个会话在没有守卫的状态下运行。
#
# 五个 install_* 都是幂等的：已在位时返回 False，不重复 patch、
# 不覆盖其他目录已安装的有效守卫。因此这里每次加载都无条件调用。
# ------------------------------------------------------------
_guard.install_guard()
_guard.install_network_guard()
_guard.install_http_guard()
_guard.install_sdk_guard()
_guard.install_env_guard()

# 复用根级的全部 fixture：pytest 按模块属性发现 fixture，赋值即完成注册。
# 这里遍历而不是逐个列名，根级新增 fixture 时本文件无需同步修改。
# pytest 9 用 _fixture_function_marker，pytest 8 用 _pytestfixturefunction，
# 两者都识别以保持兼容。
for _name, _value in list(vars(_guard).items()):
    if (
        getattr(_value, "_fixture_function_marker", None) is not None
        or getattr(_value, "_pytestfixturefunction", None) is not None
    ):
        globals()[_name] = _value


# ------------------------------------------------------------
# 阶段 1-R2.2-CLOSE：`pytest_configure` 生命周期兜底
#
# 上面那次无条件安装只在本模块被 exec_module 的那一刻执行一次。若同一
# 进程内先跑完一轮会话（根 conftest 的 `pytest_unconfigure` 已卸载全部
# 守卫），而本模块仍在 sys.modules 里、不会再被 exec_module，新一轮会话
# 就会在没有守卫的状态下运行。
#
# 这里注册真正的 pytest 钩子 —— pytest 只识别 conftest 模块顶层的
# `pytest_configure` 属性，与 sys.modules 里有没有同名模块无关，因此
# 不依赖任何存在性判断：
#   - 已在位        → install_* 返回 False，不重复 patch、不递归包装；
#   - 他方守卫在位  → 各守卫按所有权规则让位，不覆盖别人的 patch；
#   - 只装不卸      → 结束后由根 conftest 统一卸载，不会错卸他人 patch。
# ------------------------------------------------------------


def pytest_configure(config):
    """会话配置阶段兜底：再次确认五类守卫在位（均幂等）。"""
    _guard.install_guard()
    _guard.install_network_guard()
    _guard.install_http_guard()
    _guard.install_sdk_guard()
    _guard.install_env_guard()
