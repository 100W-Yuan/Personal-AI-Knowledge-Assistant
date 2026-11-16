# -*- coding: utf-8 -*-
"""多模态测试离线隔离（R6-1 + R6-1B 加固）。

加载时机保证
------------
pytest 在收集任何测试模块之前导入 conftest（`_pytest/main.py:731-737` 取
hook proxy 时加载 conftest，`_pytest/python.py:202` 在其后创建模块收集器，
`:560-561` 才真正 importtestmodule），因此这里的注入发生在 `import main` /
`import image_client` 之前。

四层隔离
--------
1. dotenv：`sys.modules["dotenv"]` 替换为假模块；若真实 dotenv 已被提前导入，
   同时中和其 `load_dotenv`。假函数带 `_wb_fake_dotenv` 标记，用于跨模块
   重新导入时仍能识别身份（不依赖函数对象 identity）。
2. API Key：`DEEPSEEK_API_KEY` 固定为 `offline-test-key`；每次 pytest 配置阶段
   重新施加，会话结束恢复原值并清空保存字典（不打印任何值）。
3. SDK：`openai.OpenAI` 替换为空壳（不保存 api_key，create 即抛错并计数）。
   本版本 SDK 中 `openai.client` 模块不存在，故不做该入口兼容；
   `openai.AsyncOpenAI` 项目内零调用，未替换（见文件末尾"已知缺口"）。
4. 网络：回环感知守卫，拦截 connect / connect_ex / create_connection /
   getaddrinfo / sendto。放行 127.0.0.0/8、::1、localhost。

R6-1B 加固点
------------
M1 违规即失败：按用例统计四类违规事件（sdk_create_attempts、non_loopback、
   dns_blocked、udp_blocked）的增量；未声明即 `pytest.fail`，失败信息含
   nodeid 与事件类别。`sdk_inits` 属正常构造，不计入违规。
   需要验证守卫本身的用例，用 `allow_isolation_events("dns")` 显式声明。
M2 启动 fail-fast：若 `main` 在 conftest 加载前已被导入且 `load_dotenv`
   不是假实现（或无法判断），直接抛错使 pytest 非零退出。
   边界：`--noconftest` 使本文件根本不执行，无法由自身防护。
M3 生命周期：保存原始 socket 方法与 `openai.OpenAI` 引用，正常结束时还原，
   避免污染嵌入调用者；patch 函数带 `_wb_guard` / `_wb_offline_shell` 标记，
   重复 `pytest.main()` 时不会重复保存已替换的对象。

已知缺口（不得据此宣称"绝对无法联网"）
--------------------------------------
- `socket.socketpair` 按设计放行（asyncio self-pipe 依赖）；
- `socket.sendmsg`、C 层 `_socket` 直调、子进程不在守卫范围；
- `--noconftest` 完全绕过本文件；
- conftest 导入中途失败、进程被强杀时不会执行还原；
- 收集期（模块导入）发生的违规事件会被归到本次会话的第一个用例上报。
"""

import ipaddress
import os
import socket
import sys
import types

import pytest

FAKE_API_KEY = "offline-test-key"
_ENV_KEYS = ("DEEPSEEK_API_KEY",)

# 违规事件（sdk_inits 不在其中：API 测试会正常构造离线空壳）
VIOLATION_KEYS = (
    "sdk_create_attempts",
    "non_loopback",
    "dns_blocked",
    "udp_blocked",
)

STATE = {
    "sdk_inits": 0,
    "sdk_create_attempts": 0,
    "non_loopback": [],
    "dns_blocked": [],
    "udp_blocked": [],
}

# 已保存的原始对象（还原用；不保存任何密钥值）
_SAVED = {}


# ============================================================
# 1. dotenv 隔离
# ============================================================


def _noop_load_dotenv(*args, **kwargs):
    """假 load_dotenv：不读文件，返回 False（与"未找到 .env"一致）。"""
    return False


def _noop_dotenv_values(*args, **kwargs):
    return {}


def _noop_find_dotenv(*args, **kwargs):
    return ""


# 身份标记：模块被重新导入后仍可识别"这是假实现"
# 阶段 1-R2.2：三个入口都要打标记。此前只有 load_dotenv 带标记，
# 导致跨目录运行时「dotenv_values / find_dotenv 是否为假实现」无法判定
# （行为本身是假的，只是标记缺失，测试因此误报）。
_noop_load_dotenv._wb_fake_dotenv = True
_noop_dotenv_values._wb_fake_dotenv = True
_noop_find_dotenv._wb_fake_dotenv = True

_fake_dotenv = types.ModuleType("dotenv")
_fake_dotenv.load_dotenv = _noop_load_dotenv
_fake_dotenv.dotenv_values = _noop_dotenv_values
_fake_dotenv.find_dotenv = _noop_find_dotenv
_fake_dotenv.__doc__ = "离线测试用假 dotenv 模块（R6-1B）"

_real_dotenv = sys.modules.get("dotenv")
if _real_dotenv is not None:
    # 真实 dotenv 已被提前导入：中和其函数属性，避免后续 from-import 拿到真实现。
    _real_dotenv.load_dotenv = _noop_load_dotenv

sys.modules["dotenv"] = _fake_dotenv


# ============================================================
# 2. 环境变量隔离 + 启动 fail-fast
# ============================================================

_ORIGINAL_ENV = {key: os.environ.get(key) for key in _ENV_KEYS}
os.environ["DEEPSEEK_API_KEY"] = FAKE_API_KEY


def _assert_isolation_intact(stage):
    """若 main 已在隔离建立前被导入，立即中止（fail-fast）。

    判定依据不是函数对象 identity（模块重导入后会变），而是假函数上的
    `_wb_fake_dotenv` 标记。属性缺失时无法可靠判断，按保守策略拒绝。
    """
    main_module = sys.modules.get("main")
    if main_module is None:
        return

    bound = getattr(main_module, "load_dotenv", None)
    if bound is None or not getattr(bound, "_wb_fake_dotenv", False):
        raise RuntimeError(
            "[测试隔离失败] 检测到 main 在 conftest 加载前已被导入"
            f"（检查阶段：{stage}），其 load_dotenv 已绑定非假实现。"
            "请勿先 import main 再调用 pytest；"
            "请使用独立 Python 进程直接执行 `python -B -m pytest`。"
        )


_assert_isolation_intact("conftest 导入")


# ============================================================
# 3. SDK 隔离
# ============================================================

try:
    import openai
except Exception:  # pragma: no cover - openai 是项目依赖，正常不会触发
    openai = None


class _OfflineOpenAI:
    """空壳 OpenAI：不建立任何连接，不保存 api_key。"""

    _wb_offline_shell = True

    def __init__(self, *args, **kwargs):
        STATE["sdk_inits"] += 1
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._blocked_create)
        )

    def _blocked_create(self, **kwargs):
        STATE["sdk_create_attempts"] += 1
        raise RuntimeError("离线测试禁止发起真实 SDK 请求")


def _install_openai_shell():
    """幂等安装；仅保存尚未被替换的原始类。"""
    if openai is None:
        return
    if getattr(openai.OpenAI, "_wb_offline_shell", False):
        return

    saved = _SAVED.setdefault("openai", {})
    saved.setdefault("OpenAI", openai.OpenAI)
    openai.OpenAI = _OfflineOpenAI


def _restore_openai_shell():
    saved = _SAVED.pop("openai", None)
    if not saved or openai is None:
        return
    openai.OpenAI = saved["OpenAI"]


_install_openai_shell()


# ============================================================
# 4. 网络隔离（回环放行）
# ============================================================


def _is_loopback_host(host):
    if host is None or host == "":
        return False
    if not isinstance(host, str):
        return False
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _is_loopback_address(address):
    if not isinstance(address, tuple) or len(address) < 2:
        return False
    return _is_loopback_host(address[0])


def _guard_connect(self, address, *args, **kwargs):
    if _is_loopback_address(address):
        return _SAVED["network"]["connect"](self, address, *args, **kwargs)
    STATE["non_loopback"].append(("connect", repr(address)))
    raise RuntimeError("NETWORK_BLOCKED_NON_LOOPBACK")


def _guard_connect_ex(self, address, *args, **kwargs):
    if _is_loopback_address(address):
        return _SAVED["network"]["connect_ex"](self, address, *args, **kwargs)
    STATE["non_loopback"].append(("connect_ex", repr(address)))
    raise RuntimeError("NETWORK_BLOCKED_NON_LOOPBACK")


def _guard_create_connection(address, *args, **kwargs):
    if _is_loopback_address(address):
        return _SAVED["network"]["create_connection"](address, *args, **kwargs)
    STATE["non_loopback"].append(("create_connection", repr(address)))
    raise RuntimeError("NETWORK_BLOCKED_NON_LOOPBACK")


def _guard_getaddrinfo(host, port, *args, **kwargs):
    if host is None or host == "" or _is_loopback_host(host):
        return _SAVED["network"]["getaddrinfo"](host, port, *args, **kwargs)
    STATE["dns_blocked"].append((repr(host), port))
    raise RuntimeError("NETWORK_BLOCKED_DNS")


def _guard_sendto(self, data, *args, **kwargs):
    address = args[-1] if args else None
    if _is_loopback_address(address):
        return _SAVED["network"]["sendto"](self, data, *args, **kwargs)
    STATE["udp_blocked"].append(repr(address))
    raise RuntimeError("NETWORK_BLOCKED_UDP")


for _guard in (
    _guard_connect,
    _guard_connect_ex,
    _guard_create_connection,
    _guard_getaddrinfo,
    _guard_sendto,
):
    _guard._wb_guard = True


def _install_network_guards():
    """幂等安装网络守卫；保存原始实现供还原使用。"""
    if getattr(socket.socket.connect, "_wb_guard", False):
        return

    _SAVED["network"] = {
        "connect": socket.socket.connect,
        "connect_ex": socket.socket.connect_ex,
        "create_connection": socket.create_connection,
        "getaddrinfo": socket.getaddrinfo,
        "sendto": socket.socket.sendto,
    }

    socket.socket.connect = _guard_connect
    socket.socket.connect_ex = _guard_connect_ex
    socket.create_connection = _guard_create_connection
    socket.getaddrinfo = _guard_getaddrinfo
    socket.socket.sendto = _guard_sendto


def _restore_network_guards():
    saved = _SAVED.pop("network", None)
    if not saved:
        return
    socket.socket.connect = saved["connect"]
    socket.socket.connect_ex = saved["connect_ex"]
    socket.create_connection = saved["create_connection"]
    socket.getaddrinfo = saved["getaddrinfo"]
    socket.socket.sendto = saved["sendto"]


_install_network_guards()


# ============================================================
# 5. pytest 生命周期
# ============================================================


def pytest_configure(config):
    """每次会话重新施加隔离，覆盖"同进程重复 pytest.main()"的缓存场景。"""
    sys.modules["dotenv"] = _fake_dotenv
    os.environ["DEEPSEEK_API_KEY"] = FAKE_API_KEY
    _assert_isolation_intact("pytest 配置阶段")
    _install_network_guards()
    _install_openai_shell()
    config.addinivalue_line(
        "markers",
        "fresh_process: 在全新 Python 子进程中运行该用例，隔离 sys.modules 模块导入污染",
    )


def pytest_unconfigure(config):
    """恢复环境变量与被替换的全局对象；不输出任何值。"""
    for key, value in _ORIGINAL_ENV.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    _ORIGINAL_ENV.clear()

    _restore_network_guards()
    _restore_openai_shell()


# ============================================================
# 6. M1：隔离违规必须成为失败条件
# ============================================================


def _snapshot():
    return {
        key: len(STATE[key]) if isinstance(STATE[key], list) else STATE[key]
        for key in VIOLATION_KEYS
    }


_BASELINE_CHECKED = {"done": False}


@pytest.fixture(autouse=True)
def _isolation_event_guard(request):
    """按用例检查违规事件增量；未声明即失败。

    - 只统计四类违规事件，`sdk_inits`（正常构造）不计入；
    - 声明只对本用例生效，不会影响其他用例；
    - 用例本身的异常不会被吞掉：本 fixture 仅追加一条失败报告。
    """
    if not _BASELINE_CHECKED["done"]:
        _BASELINE_CHECKED["done"] = True
        pre = _snapshot()
        if any(pre.values()):
            pytest.fail(
                "[隔离] 用例开始前已存在隔离事件（可能发生在收集期/模块导入阶段）："
                f"{ {k: v for k, v in pre.items() if v} }，首个用例 nodeid={request.node.nodeid}",
                pytrace=False,
            )

    before = _snapshot()
    yield
    after = _snapshot()

    delta = {key: after[key] - before[key] for key in VIOLATION_KEYS if after[key] > before[key]}
    if not delta:
        return

    declared = set(getattr(request.node, "_wb_allowed_isolation_events", ()))
    unexpected = {key: value for key, value in delta.items() if key not in declared}
    if unexpected:
        pytest.fail(
            f"[隔离] 检测到未声明的隔离事件 nodeid={request.node.nodeid} "
            f"events={unexpected}；若为守卫自身的预期拦截，"
            f"请在用例参数中声明 allow_isolation_events(\"{'、'.join(unexpected)}\")",
            pytrace=False,
        )


@pytest.fixture
def allow_isolation_events(request):
    """声明本用例预期触发的隔离事件类别（仅当前用例有效）。

    用法：def test_x(allow_isolation_events):
              allow_isolation_events("dns")
    """

    allowed = set()
    request.node._wb_allowed_isolation_events = allowed

    def _allow(*kinds):
        allowed.update(kinds)

    return _allow


# ============================================================
# 7. 汇总输出（只报计数，不报任何密钥或环境变量值）
# ============================================================


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    dotenv_is_fake = sys.modules.get("dotenv") is _fake_dotenv

    main_module = sys.modules.get("main")
    if main_module is None:
        main_binding = "not-imported"
    elif getattr(getattr(main_module, "load_dotenv", None), "_wb_fake_dotenv", False):
        main_binding = "fake"
    else:
        main_binding = "REAL"

    terminalreporter.write_sep("-", "离线隔离（conftest.py · R6-1B）")
    terminalreporter.write_line(
        f"dotenv_module_is_fake={dotenv_is_fake} main.load_dotenv={main_binding} "
        f"api_key_is_fake={os.environ.get('DEEPSEEK_API_KEY') == FAKE_API_KEY}"
    )
    terminalreporter.write_line(
        f"sdk_shell_inits={STATE['sdk_inits']} "
        f"sdk_create_attempts={STATE['sdk_create_attempts']}"
    )
    terminalreporter.write_line(
        f"non_loopback_blocked={len(STATE['non_loopback'])} "
        f"dns_blocked={len(STATE['dns_blocked'])} "
        f"udp_blocked={len(STATE['udp_blocked'])}"
    )


# ============================================================
# 8. 根级测试数据库守卫复用（阶段 1-R2 增量）
#
# 本目录在项目根调用 pytest 时会自动继承根级 conftest.py；在本目录内
# 直接调用 pytest 时 rootdir 会退化，根级守卫不会被加载。以下代码把
# 根级 conftest.py 加载进来并复用其哨兵 fixture，不新增任何钩子，
# 因此上面的 dotenv / SDK / 网络隔离逻辑与输出完全不受影响。
# ============================================================

from pathlib import Path  # noqa: E402  （追加段落所需，避免改动原有 import 区）

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
# 根 conftest.py 的 pytest_unconfigure 会在每次会话结束时卸载守卫，
# 模块对象却仍留在 sys.modules 里。同进程再次 pytest.main() 时，上面的
# `if ... not in sys.modules` 判定为「已加载」而跳过 exec_module，
# 模块级的「导入即安装」随之不执行 —— 会话在没有守卫的状态下运行。
#
# 网络守卫例外：本目录保留自己的「STATE 统计型」守卫（下一段详述），
# 因此这里只安装数据库 / HTTP / SDK / dotenv 四类。网络守卫交给下面
# 那次 uninstall 处理 —— 它会移除可能被上次会话残留、或被根
# pytest_configure 装上的那层根级守卫，保证本文件的守卫在最外层。
# ------------------------------------------------------------
_guard.install_guard()
_guard.install_http_guard()
_guard.install_sdk_guard()
_guard.install_env_guard()

# 阶段 1-R2.1：本目录保留自己的「STATE 统计型」网络守卫（第六节的
# `_isolation_event_guard` 依赖它的计数），因此让根级统一网络守卫在此目录
# 让位。两种调用方式下最终都是本文件的守卫位于最外层：
# - 从项目根调用：根守卫先装 → 本文件守卫叠在其上 → 此处 uninstall 不动它；
# - 在本目录内调用：本文件守卫先装 → 加载根 conftest 时根守卫叠在其上 →
#   此处 uninstall 移除根守卫，恢复本文件守卫。
# dotenv 与 SDK 空壳同理由本文件提供；根级的 SDK 请求守卫属于额外防线，
# 保留不影响 `_OfflineOpenAI` 空壳（空壳不继承 SyncAPIClient）。
_guard.uninstall_network_guard()

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


# ============================================================
# PAI-AUTH-MVP：既有接口测试的鉴权替身
#
# 本目录的用例聚焦业务语义（知识库检索/上传、图片接口、/chat 错误映射），
# 在接入认证之前它们默认「匿名即可调用」。引入登录与角色后，这些用例
# 需要先登录才能到达业务代码，但那样每个文件都要复制一遍登录流程。
#
# 这里把四个身份依赖替换为固定管理员，让业务用例继续测它关心的东西；
# 鉴权本身（401/403/302、未分类路由即失败）由 测试/test_auth_mvp.py 与
# 测试/test_auth_ui_login.py 用真实依赖覆盖，不会被这里削弱。
#
# 只在本目录生效：其它目录仍走真实鉴权。
# ============================================================


@pytest.fixture(autouse=True)
def _stub_auth_for_business_tests():
    # 关键：不能用 import main 主动把 main 拉进 sys.modules。
    # 本目录有用例（test_knowledge_rag_real_http_e2e）以「main 尚未导入」
    # 为前提，主动导入会让它必然失败。这里只对「已导入」的情况装替身。
    import auth.deps as auth_deps

    main_module = sys.modules.get("main")
    if main_module is None:
        yield None
        return

    stub_user = {
        "id": "test-user",
        "username": "test-admin",
        "role": "admin",
        "is_active": True,
        "created_at": "2026-01-01T00:00:00Z",
    }

    overrides = main_module.app.dependency_overrides
    saved = {
        auth_deps.get_current_user: overrides.get(auth_deps.get_current_user),
        auth_deps.get_optional_user: overrides.get(auth_deps.get_optional_user),
        auth_deps.get_page_user: overrides.get(auth_deps.get_page_user),
        auth_deps.require_admin: overrides.get(auth_deps.require_admin),
    }

    overrides[auth_deps.get_current_user] = lambda: stub_user
    overrides[auth_deps.get_optional_user] = lambda: stub_user
    overrides[auth_deps.get_page_user] = lambda: stub_user
    overrides[auth_deps.require_admin] = lambda: stub_user

    yield stub_user

    for dependency, previous in saved.items():
        if previous is None:
            overrides.pop(dependency, None)
        else:
            overrides[dependency] = previous


# ============================================================
# PAI-AUTH-MVP：模块污染测试的独立进程隔离
#
# 本目录有两项用例以「某个模块尚未被导入」为前提：
#   - test_knowledge_rag_real_http_e2e.py::test_real_http_rag_end_to_end
#       运行时 main / sentence_transformers 均不得已导入；
#   - test_rag_offline_e2e.py::test_F01_no_model_runtime_is_imported
#       运行时 sentence_transformers / torch / transformers /
#       huggingface_hub / rag.embedding 均不得已导入。
# 而本目录另有 9 个文件在模块顶层 `import main`（test_chat_http_contract.py、
# test_image_api.py、test_image_client_wiring.py、test_image_upload_limits.py、
# test_knowledge_documents_api.py、test_knowledge_documents_upload.py、
# test_knowledge_import_legacy_contract.py、test_knowledge_endpoints.py、
# test_vision_factory_integration.py），且 test_real_http_rag_end_to_end 内部会
# 真实导入 sentence_transformers。这些用例按字母序在同一进程内运行，必然互相
# 污染，导致上述两项的「未导入」断言失败。
#
# 解决：给这两项用例打 @pytest.mark.fresh_process，本钩子在 runtest 协议层拦截，
# 改在全新 Python 子进程里执行 `python -m pytest <nodeid>`。子进程是干净解释器，
# conftest 的假 dotenv / 离线守卫照常加载，断言因此成立。
# 不删除、不跳过、不弱化任何断言，不改动任何认证业务代码。
#
# 防递归：子进程通过环境变量 PAI_FRESH_PROCESS_CHILD=1 标记为「子进程」，
# 钩子看到该标记后直接放行，让用例在本（干净）进程内正常执行。
# ============================================================

import hashlib
import subprocess
import tempfile
import time

from _pytest.reports import TestReport

_FRESH_PROCESS_ENV = "PAI_FRESH_PROCESS_CHILD"


def _run_fresh_process(item):
    """在全新 Python 子进程运行单个用例，返回 (outcome, longrepr, duration)。"""
    nodeid = item.nodeid
    rootpath = str(item.config.rootpath)

    # 唯一 basetemp：沿用项目约定（pytest 9.1.1 默认临时目录清理会触发沙箱
    # 批量删除），零删除、不触碰历史残留。
    tag = hashlib.sha1(nodeid.encode("utf-8")).hexdigest()[:12]
    # 平铺唯一目录：中间目录不存在时 pathlib.mkdir(parents=False) 会失败。
    basetemp = os.path.join(tempfile.gettempdir(), "pai_fresh_process_" + tag)

    cmd = [
        sys.executable, "-B", "-m", "pytest",
        nodeid,
        "-q", "--no-header",
        "--basetemp=" + basetemp,
        "-o", "tmp_path_retention_policy=all",
    ]

    env = dict(os.environ)
    env[_FRESH_PROCESS_ENV] = "1"

    start = time.monotonic()
    proc = subprocess.run(
        cmd,
        cwd=rootpath,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    duration = time.monotonic() - start
    output = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()

    if proc.returncode == 0:
        return "passed", None, duration
    if proc.returncode == pytest.ExitCode.NO_TESTS_COLLECTED:
        return "failed", "独立进程未收集到用例（NO_TESTS_COLLECTED）\n" + output, duration
    return "failed", "独立进程运行失败（exit=%d）\n%s" % (proc.returncode, output), duration


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    # 子进程内放行：让用例在本（干净）解释器里正常执行。
    if os.environ.get(_FRESH_PROCESS_ENV) == "1":
        return None
    if item.get_closest_marker("fresh_process") is None:
        return None

    outcome, longrepr, duration = _run_fresh_process(item)

    item.ihook.pytest_runtest_logstart(nodeid=item.nodeid, location=item.location)
    report = TestReport(
        nodeid=item.nodeid,
        location=item.location,
        keywords=dict(item.keywords),
        outcome=outcome,
        longrepr=longrepr,
        when="call",
        duration=duration,
    )
    item.ihook.pytest_runtest_logreport(report=report)
    # 返回非 None：firstresult 短路，跳过本进程内的默认 setup/call/teardown。
    return True
