# PAI-AUTH-R2-1-R3 设计修订报告

> 阶段：PAI-AUTH-R2-1 第三轮定点修订（R3）
> 基线 HEAD：`81a8f2dec62df00a82b9d838cfa18d7b98962b61`（分支 `main`）
> 性质：**只修订设计文档**。本轮不写一行业务代码、不动一次数据库、不装一个包。
> 前序文件：`docs/PAI-AUTH-R2-1_设计报告.md`（R1/R2-1）、`docs/PAI-AUTH-R2-1-R2_设计修订报告.md`（R2）。两份原文**未覆盖、未删除、未重命名**。

---

## 0. 文档定位与三类放行

### 0.1 R1 / R2 / R3 的关系

| 轮次 | 产出 | 定位 | 遗留 |
|---|---|---|---|
| R2-1 | 架构与库表详细设计 | 建立了 7 项产品决策 + 13 张表雏形 | 13 项 P0/P1 缺陷 |
| R2-1-R1 | 定点修订 13 项 | 修掉闭包捕获身份、messages 冗余 user_id、NULL 比较、锁定型 DoS | 11 项深化要求 |
| R2-1-R2 | 定点修订 11 项 | 修掉状态回滚、actor NOT NULL 冲突、锁字典泄漏、非原子预扣 | 11 项机制级缺口（本轮） |
| **R2-1-R3（本轮）** | 定点修订 11 项 | 从「写对了」推进到「**机制上不可能写错**」 | 见 §8 待核验 |

R3 的判断标准与 R1/R2 不同：**不再满足于「设计里写了禁止」，而要求「写错了会被测试或机制抓住」**。本轮 11 项中 8 项是 P0，共同特征是「R2 的方案在正常路径下正确，在并发、崩溃、误用路径下失效」。

### 0.2 三类放行必须分开

任务书第 3 条要求区分三种放行。本报告在 §7 分别给出结论，此处先定义判定口径：

| 放行类型 | 含义 | 判定依据 | 本轮结论 |
|---|---|---|---|
| **设计放行** | 设计文档可作为编码依据 | 11 项均有明确方案 + 可执行验收标准 + P0 阻断项为 0 | ✅ 通过（§7.1） |
| **编码放行（R2-2）** | 允许开始写代码 | 设计放行 + 用户**明确授权** + 无「编码前必须知道」的未知项 | ⏸ 建议进入，但**不自动授权**（§7.2） |
| **生产上线放行** | 允许部署到腾讯云公网 | 编码放行 + 目标机实测通过（V-01/V-02/V-03/V-09） | ❌ 不通过（§7.3） |

> 三者不是递进承诺。设计通过 ≠ 可以上线；本轮连编码都未获授权。

### 0.3 证据标记约定

本报告所有技术断言标注证据来源，**不写「符合最佳实践」这类无法核验的话**：

| 标记 | 含义 |
|---|---|
| `【实测】` | 本轮在本机用 Python 3.12.10 实跑得到，附可复现命令（§1.3） |
| `【代码】` | 从现有源码直接读出，附 `文件:行号` |
| `【文档】` | 来自 R1/R2 原文或框架语义，非本轮实测 |
| `【待核验】` | 现有证据不足，**不猜测**，列入 §8 |

---

## 1. 基线与本轮取证

### 1.1 Git 与文件指纹（开工时，2026-11-09 15:25）

| 项目 | 值 |
|---|---|
| HEAD | `81a8f2dec62df00a82b9d838cfa18d7b98962b61` |
| 分支 | `main` |
| `git status --porcelain` 条目 | **59**（25 项 `M`/`D` + 34 项 `??`） |
| `git stash list` | 0 |
| `docs/` 目录文件数 | 21 |

核心文件 SHA-256 前 16 位（全量见 §9.2）：

| 文件 | size | SHA-256（前 16） |
|---|---|---|
| `main.py` | 39,559 | `a8397b5433a4e680` |
| `memory.py` | 32,571 | `8a79e394607022e5` |
| `agent.py` | 118,574 | `b8a8e32f20669106` |
| `knowledge_store.py` | 21,587 | `5d46ff6abf9189fa` |
| `tools.py` | 21,036 | `bee0341bad95af5b` |
| `business/api.py` | 8,356 | `6385020cdf6fc0a9` |
| `business/db.py` | 1,541 | `4afa0fac166160ae` |
| `conftest.py` | 57,791 | `5fde88c5e28dc142` |
| `rag/knowledge_base.py` | 26,097 | `c35f36d3f4a7f078` |
| `rag/vector_store.py` | 1,286 | `ce61b2b5c2cadd05` |
| `rag/embedding.py` | 335 | `06acb4f5d39c5893` |
| `task.py` | 4,236 | `7a6f30eb3f3e0fa5` |
| `tool_registry.py` | 4,962 | `af52be93e96a79c8` |

数据库：

| 文件 | size | mtime | SHA-256（前 16） |
|---|---|---|---|
| `memory.db` | 98,304 | 2026-11-07 18:40:03.071817100 +0800 | `dbe0d2a497b02236` |
| `data/business.db` | 73,728 | 2026-11-07 17:57:07.405455100 +0800 | `678a9b871bf7403f` |

### 1.2 运行环境实测值

| 项 | 实测值 | 证据 |
|---|---|---|
| Python | 3.12.10 (tags/v3.12.10:0cf2128, Apr 8 2025, 12:21:36) [MSC v.1943 64 bit (AMD64)] | 【实测】 |
| SQLite | 3.49.1 | 【实测】 |
| fastapi / starlette / anyio | 0.141.1 / 1.6.0 / **4.15.1** | 【实测】 |
| uvicorn / httpx / pydantic | 0.52.4 / 0.28.1 / 2.13.5 | 【实测】 |
| Pillow / python-multipart | **13.0.0** / **0.0.32**（均未写入 `requirements.txt`） | 【实测】 |
| argon2-cffi / cryptography / bcrypt | **均未安装** | 【实测】 |
| tzdata | **未安装** | 【实测】 |
| 两库 `journal_mode` | **`delete`**（非 WAL） | 【实测】`PRAGMA journal_mode` |
| 两库 `page_size` / `page_count` | 4096 / 24（memory.db）、4096 / 18（business.db） | 【实测】 |
| 两库 `user_version` | 0 / 0（**迁移版本号当前未使用**） | 【实测】 |

### 1.3 本轮实测手段与边界

为避免「未经验证的推断」，本轮在系统临时目录 `%TEMP%\pai_r3_probe\` 写了 7 个一次性探针脚本，全部**只 import 标准库与已安装框架**，**不 import 任何项目模块**（因此不触发 `dotenv.load_dotenv`、不打开项目数据库、不写 `__pycache__`）。

| 探针 | 验证对象 | 关键结论（见 §3 引用） |
|---|---|---|
| `probe_ctx.py` | ContextVar 在 6 条通道的传播 | create_task / to_thread / BackgroundTasks **传播**；run_in_executor / threading.Thread **不传播**；anyio 线程池**传播** |
| `probe_isolation.py` | 4 种隔离原语的可用性 | `create_task(context=Context())` 与「子任务首句 set(None)」均有效，父侧不受影响 |
| `probe_sqlite.py` | WAL / checkpoint / backup / data_version | checkpoint busy 返回 `(1,1,1)`；非 WAL 返回 `(0,-1,-1)`；备份是**时点快照**，与源库最终行数不同 |
| `probe_walfile.py` | WAL/SHM 句柄占用 | 连接打开时 `os.remove` → `PermissionError`；关闭后文件自动消失 |
| `probe_sql.py` | NULL 比较 / NOT IN 陷阱 / 原子预扣 | `NOT IN` 漏判 3 vs `NOT EXISTS` 正确 2；`<>` 得 0 而 `IS NOT` 得 1；预扣 rowcount `[1,1,0]` |
| `probe_routes.py` | 路由（**AST 静态解析，不 import**） | 38 条 = `main.py` 25 + `business/api.py` 13；**33 条 sync / 5 条 async** |
| `probe_script.py` | `executescript` 原子性 | 失败后表已建、行已插，`in_transaction=False` → `rollback()` **无效** |

只读数据查询（`file:memory.db?mode=ro`，未开写连接）：

```
conversations 2 / messages 53 / distinct conversation_id = 1（全部 'legacy'）
孤儿消息（NOT EXISTS）= 0；无主 user_id（列尚不存在）
memories 5
```

> **这条数据很关键**：现网 53 条消息全部落在 `legacy` 会话，且 0 条孤儿。它把 R3-02 的「隔离」从「必然发生」降级为「预期 0 行，但必须可判定、可回滚」。设计不能因为「现在没有」就省掉逆向恢复 —— 省掉后一旦出现孤儿就无退路。

### 1.4 R2 结论中需要修正的表述（本轮自我纠正）

R3 复核 R2 时发现 3 处表述不准确，先纠正再往下设计：

| # | R2 原表述 | 实测/代码事实 | 影响 |
|---|---|---|---|
| C-1 | 「`memory.Memory.get_connection` 无 `busy_timeout`」 | `sqlite3.connect()` 默认 `timeout=5.0` → **`PRAGMA busy_timeout` 实测 = 5000 ms** | 不是「没有」，而是「有 5 秒但未显式声明」。R3-07 的并发设计可以依赖这个等待窗口，但必须显式写 `timeout=` 以免默认值变化 |
| C-2 | 「`asyncio.to_thread` 不传播 / 只提 `run_in_executor`」 | 实测 **`to_thread` 传播**（读到 `user-42`），`run_in_executor` **不传播**（读到 `None`） | R2-01 的禁用清单漏了 `to_thread`，这是真实泄漏点，R3-01 补上 |
| C-3 | 「`KnowledgeBase` 不支持并发」 | 实际是**读并发安全、写不安全**（`VectorStore.search` 只遍历 `items`，`items` 只在初始化期写） | R3-06 据此把约束表述为「实例不可变」，而非「禁止并发」 |

另有一处**代码级新发现**（不是 R2 的错，是 R2 没查）：

> `business/db.py:56` 用 `connection.executescript(SCHEMA_SQL)`，第 63 行的 `connection.rollback()` 在脚本中途失败时**是空操作**。实测：脚本第 4 句语法错误后，前 2 张表已建、1 行数据已插、`in_transaction=False`。也就是说**今天的 `business.init_database()` 本身就不是原子迁移**。R3-03 的状态机必须覆盖这个既有缺陷。

---

## 2. 11 项修订逐项对照表

| 编号 | 主题 | R2 结论 | R3 发现的问题 | R3 修订要点 | 级别 | 阻断对象 |
|---|---|---|---|---|---|---|
| R3-01 | RuntimeContext 后台隔离 | 「禁止后台任务隐式继承」 | 只有文字禁令，无机制；且漏了 `to_thread` / `BackgroundTasks` 两个真实泄漏点 | 空 Context 创建 + 首句清零 + 快照显式传参 + AST 静态守卫 | P0 | 编码 |
| R3-02 | 孤儿消息 | `NOT IN` + `UPDATE ... SET user_id=NULL` 回滚 | `NOT IN` 在 NULL 子查询下漏判；无条件回滚会波及后续合法修改 | `NOT EXISTS` + 隔离清单表 + batch 精确回填 + 四条件 `UPDATE` | P0 | 编码 |
| R3-03 | 迁移状态机 | 版本级 `running/succeeded` + 双连接 | 无步骤级状态；`executescript` 非原子；多 Worker 无 claim | 版本/步骤两级 + 幂等探针 + `changes()` 抢锁 + 四边界崩溃矩阵 | P0 | 编码 |
| R3-04 | 审计主体 | 三类 actor + 条件 CHECK | 未定「审计失败能否继续」；未定 bootstrap 唯一来源；未定脱敏字段 | 分级失败策略 + CLI-only bootstrap + 一次性 token + `scrub()` 白名单 | P1 | 编码 |
| R3-05 | 路由矩阵 | 48 行清单 | 未逐条核对现存 38；未给 Origin/限速/数据范围字段；未给 docs 关闭验证法 | AST 全量核对 38 + 48 行八字段矩阵 + 404 断言法 | P0 | 上线 |
| R3-06 | KB 缓存 | `_Entry` + refcount + LRU=8 | 获取与淘汰不在同一临界区；忙时淘汰语义未定；上限是硬是软未定 | 占位构建 + `ready_event` + 软上限 8/硬上限 16 + 全忙拒绝新私有 KB | P0 | 编码 |
| R3-07 | 预算状态机 | 原子预扣 + 汇总 | 无 reservation 身份，无法结算/释放/崩溃恢复；未知用量语义未定 | 事件级 reservation + 四态迁移 + 保守 `kept` + 对账 | P0 | 编码 |
| R3-08 | CSRF/Session | 双提交 + 严格 Origin + no-store | 无匿名态、无轮换、无优先级、无并发续期/撤销竞态 | 8 态生命周期 + Origin>Referer>Sec-Fetch-Site>拒绝 + 节流续期 | P0 | 编码 |
| R3-09 | 备份恢复 | 6 道门禁 + 恢复 6 步 | 未处理 checkpoint busy/非 WAL 返回值；用行数判完整不可靠；WAL/SHM 删除时机未定 | 8 道门禁 + 主键集合哈希 + 同卷 `os.replace` + 连接全关闭 | P0 | 上线 |
| R3-10 | 依赖与平台 | 生产 9 项 + argon2-cffi | 未区分 Win/Linux；未补 Pillow/multipart 版本；未定哈希锁定 | 双环境矩阵 + 精确 pin + lock 生成流程 + 平台差异清单 | P1 | 上线 |
| R3-11 | 测试矩阵 | 11 组「全绿」 | 「全绿」不是设计；无前置/数据/时序/DB 预期/失败判定 | 8 字段逐案补全，共 80 个用例 | P0 | 编码 |

**P0 阻断项统计：0**（11 项均有方案与可执行验收标准）。**上线阻断项：4**（V-01/V-02/V-03/V-09，见 §8）。

---

## 3. 逐项完整设计

### R3-01 RuntimeContext 后台任务上下文隔离（P0）

#### 3.1.1 原设计（R2-01）

`auth/context.py` 提供 `RuntimeContext` 冻结数据类 + `ContextVar` + `runtime_context(ctx)` 上下文管理器；规则为「端点 body 内 `with runtime_context(...)`、禁止嵌套、禁止后台任务隐式继承」。

#### 3.1.2 发现的问题

**(a) 「禁止」不是机制。** R2 用一句文字要求开发者不要在后台任务里读上下文。只要有人写 `asyncio.create_task(do_audit())`，子任务就会读到发起请求的用户 —— 而且会「工作正常」，测试也发现不了。

**(b) 六条通道的传播行为不一致，靠记忆必错。** 实测结果：

| 通道 | 子执行体读到的值 | 判定 | 证据 |
|---|---|---|---|
| `asyncio.create_task(coro())` | `user-42` | **泄漏** | 【实测】`/async-create-task` |
| `asyncio.create_task(coro(), context=Context())` | `None` | 隔离 | 【实测】`probe_isolation` |
| `asyncio.to_thread(fn)` | `user-42` | **泄漏** | 【实测】`/async-to-thread` |
| `loop.run_in_executor(None, fn)` | `None` | 不传播 | 【实测】`/async-run-in-executor` |
| Starlette `BackgroundTasks`（async 任务） | `user-42` | **泄漏** | 【实测】`/background-tasks.holder.async` |
| Starlette `BackgroundTasks`（sync 任务） | `user-42` | **泄漏** | 【实测】`/background-tasks.holder.sync` |
| `threading.Thread` | `None` | 不传播 | 【实测】`plain_thread` |
| anyio 线程池（`def` 端点） | `user-42` | **这是框架正常行为，必须保留** | 【实测】`/sync-endpoint-dep` |

**(c) `copy_context()` 的位置极易搞反。** 实测：父线程 `set()` 后，在**子线程内**调 `contextvars.copy_context()` 得到的是**空**上下文（`thread_child_ctx = null`）；必须在**父侧**捕获再传入才有效（`thread_parent_ctx = user-42`）。这是「看起来用了 contextvars，其实什么都没带过去」的经典陷阱。

**(d) 33/38 个端点是 `def`（同步）。** 【实测】AST 解析：`main.py` 25 条中 20 条 sync、`business/api.py` 13 条全 sync → **33 条同步 / 5 条异步**。同步端点在 anyio 线程池执行，而实测确认 anyio 4.15.1 **会把上下文复制进工作线程**。这既是好消息（一个 `Depends` 就能给同步端点建立上下文），也是风险点（线程池复用是否残留？实测 `/sync-endpoint` 在多次污染后仍读到 `None` → **无跨请求残留**）。

#### 3.1.3 修订方案

**(1) 唯一对外接口（模块只导出 4 个名字）**

```python
# auth/context.py
@dataclass(frozen=True, slots=True)
class IdentitySnapshot:            # 后台任务的唯一身份载体
    user_id: str
    role: str                      # 'user' | 'admin'
    request_id: str                # 沿用发起请求的 id，便于串联日志
    issued_at: str                 # ISO8601 UTC

@dataclass(frozen=True, slots=True)
class RuntimeContext:
    user_id: str
    role: str
    request_id: str
    scopes: tuple[str, ...]
    detached: bool = False         # True 表示后台任务内的受限上下文

_CURRENT: ContextVar[RuntimeContext | None] = ContextVar("pai_ctx", default=None)

def current_context() -> RuntimeContext:      # 无值 → ContextMissingError
def runtime_context(ctx) -> AbstractContextManager:   # 端点用；嵌套 → RuntimeError
def snapshot_identity() -> IdentitySnapshot:  # 只能在 runtime_context 内调用
def run_detached(factory, *, identity: IdentitySnapshot | None = None) -> asyncio.Task
```

**(2) 隔离机制（双保险，实测均有效）**

```python
def run_detached(factory, *, identity=None):
    async def _runner():
        _CURRENT.set(None)                    # 保险 2：首句清零（对 context= 不可用的路径兜底）
        if identity is None:
            return await factory()            # 全程无身份：只能做纯计算/清理
        ctx = RuntimeContext(
            user_id=identity.user_id,
            role=identity.role,
            request_id=identity.request_id,
            scopes=("background",),           # 只保留 background 子集，剥离写类 scope
            detached=True,
        )
        token = _CURRENT.set(ctx)             # 显式重建「受限」上下文
        try:
            return await factory()
        finally:
            _CURRENT.reset(token)

    return asyncio.create_task(_runner(), context=contextvars.Context())   # 保险 1：空上下文创建
```

- 保险 1 从机制上切断继承（实测 `create_task_empty_context = null`）；
- 保险 2 覆盖「有人绕开 `run_detached` 但复用了 `_runner`」的情形，且实测 `wrapper_set_none` 与 `wrapper_set_none_copy` 都为 `null`，`parent_after_all` 仍是 `user-42` → **清零不会污染父上下文**。

**(3) 身份只能显式传递**

- `snapshot_identity()` 只能在 `runtime_context` 内调用，返回**不可变**快照；
- `run_detached(factory, identity=snap)` 是唯一把身份带进后台的通道；
- **禁止**在后台任务里调 `current_context()` 取身份后再切到「用户态」——`detached=True` 由 `RuntimeContext` 携带，写审计时 `actor_type` 仍为 `user` 但 `detail.detached=true`，管理层可据此区分「用户直接操作」与「用户触发的延迟副作用」。

**(4) 五类通道的处置矩阵（写入编码规范，并由静态守卫强制）**

| 通道 | 允许 | 约束 |
|---|---|---|
| `asyncio.create_task()` | ❌ 裸用 | 仅 `auth/context.py` 内允许；其它模块必须 `run_detached` |
| `asyncio.to_thread()` | ❌ | 会传播身份；需要线程时改用 `run_in_executor` + 显式快照 |
| `loop.run_in_executor()` | ✅ | 不传播；参数里必须显式带 `IdentitySnapshot` |
| `threading.Thread()` | ✅ | 同上 |
| `BackgroundTasks.add_task()` | ⚠️ 仅限无身份任务（如临时文件清理） | 若任务需要身份 → 必须 `run_detached` |
| anyio 线程池（`def` 端点） | ✅ 保留 | 这是 `Depends` 建立上下文的唯一正确路径 |

**(5) 机制层强制：AST 静态守卫（不依赖自觉）**

新增 `tests/test_no_raw_spawn.py`，用 `ast` 扫描仓库内所有 `*.py`（排除 `tests/`、`.backup_*`、`auth/context.py`），命中以下任一即**失败**：

```python
BANNED = {
    ("asyncio", "create_task"),        # 裸 create_task
    ("asyncio", "to_thread"),
    ("contextvars", "copy_context"),   # 禁止在各处随手 copy，只允许 auth/context.py
}
BANNED_METHODS = {"add_task"}           # BackgroundTasks.add_task
```

**基线验证**：本轮实测全项目 `create_task|BackgroundTasks|run_in_executor|to_thread` 在业务代码中 **0 命中**（命中的 5 处全在测试里的 `threading.Thread`）→ 该守卫在现有代码上可直接通过，不会误伤。

#### 3.1.4 实现约束

| 约束 | 理由 |
|---|---|
| `RuntimeContext` 与 `IdentitySnapshot` 必须 `frozen=True` | 后台任务持有引用，可变对象会被后续请求改写 |
| `ContextMissingError` 必须继承 `RuntimeError`，且**不带任何身份信息** | 异常消息会被记日志，不能泄漏 user_id |
| `run_detached` 必须在**发起请求的同步栈内**调用（不能在另一个后台任务里再派生） | 否则 `snapshot_identity()` 已失效 |
| 后台任务的异常必须被 `add_done_callback` 捕获并记 `actor_type='system'` 审计 | 否则 `Task exception was never retrieved` 静默丢失 |
| `request_id` 沿用发起请求的值，不重新生成 | 便于把「用户请求 → 延迟副作用」串成一条链 |

#### 3.1.5 失败处理

| 场景 | 行为 |
|---|---|
| 后台任务内 `current_context()`（未传 identity） | 抛 `ContextMissingError` → 被 `done_callback` 捕获 → `logger.error` + `audit(actor_type='system', action='background_context_missing')` → **不重试** |
| 后台任务内 `current_context()`（传了 identity） | 正常返回 `detached=True` 的 ctx |
| `run_detached` 在无事件循环的线程中调用 | 抛 `RuntimeError("no running event loop")`，由调用方改走 `run_in_executor` |
| 嵌套 `runtime_context` | 抛 `RuntimeError("nested runtime context")`，不静默覆盖 |
| 后台任务在请求结束后访问已失效的 `session_id` | 设计上 `IdentitySnapshot` **不含** `session_id`；需要鉴权的后续动作必须重新走登录态校验，不得复用快照做授权 |

#### 3.1.6 验收标准

| 用例 | 前置 | 执行 | 断言 |
|---|---|---|---|
| T-01-A | 已登录请求内 | `run_detached(lambda: current_context())`，`identity=None` | 任务体抛 `ContextMissingError`；异常被 `done_callback` 捕获；审计表新增 1 行 `actor_type='system'` |
| T-01-B | 同上 | `run_detached(..., identity=snapshot_identity())` | `ctx.user_id` 与登录用户相等；`ctx.detached is True`；`ctx.scopes == ("background",)` |
| T-01-C | 同上 | 请求返回后再让后台任务执行（用 `threading.Event` 同步） | 任务仍能完成；读不到父请求的其它 ContextVar（用探针变量断言为 `None`） |
| T-01-D | 仓库源码 | 运行 `tests/test_no_raw_spawn.py` | 0 命中；若人为在 `main.py` 加一行 `asyncio.create_task(...)` → 测试失败并指出文件行号 |
| T-01-E | 两个并发用户 A/B | A 的后台任务与 B 的后台任务同时跑 | 互不读到对方 user_id（各 100 次采样全等） |

**判定标准**：T-01-A 若「任务体没抛异常」即失败（说明隔离未生效）；T-01-D 若「改坏后测试仍通过」即失败（说明守卫无效）。**不接受「人工 review 确认没有裸 create_task」作为证据。**

---

### R3-02 孤儿消息隔离与完整逆向恢复（P0）

#### 3.2.1 原设计（R2-02）

「不猜测归属；N>0 即中止迁移；人工授权后『NULL 所有者会话』隔离，可逆」，回滚方案为 `UPDATE conversations SET user_id = NULL`。

#### 3.2.2 发现的问题

**(a) `NOT IN` 在子查询含 NULL 时会静默漏判。** 实测：

```sql
-- 子查询里有 user_id IS NULL 的行时
SELECT count(*) FROM messages WHERE conversation_id NOT IN
  (SELECT conversation_id FROM conversations WHERE user_id IS NOT NULL);   -- 返回 3
SELECT count(*) FROM messages m WHERE NOT EXISTS
  (SELECT 1 FROM conversations c WHERE c.conversation_id = m.conversation_id); -- 返回 2（正确）
```

差异来自 SQL 三值逻辑：`x NOT IN (..., NULL)` 结果是 `UNKNOWN` 而非 `TRUE`，该行被丢弃。**漏判的正是最需要处理的那批数据。**

**(b) `<>` 比较同样被 NULL 吃掉。** 实测：`WHERE m.user_id <> c.user_id` → **0 行**；`WHERE m.user_id IS NOT c.user_id` → **1 行**（正确）。R2 里那条「owner 一致性校验」在 NULL 存在时等于**永远通过**的空校验。

**(c) 无条件回滚会破坏后续合法修改。** `UPDATE conversations SET user_id = NULL` 的语义是「把这张表所有会话的所有者抹掉」，包括：迁移后新注册用户创建的会话、管理员后来认领的会话、本轮根本没被隔离的会话。**它不区分批次，也不校验当前值。**

**(d) 孤儿消息处于「列表不可见、按 ID 可读」的隐形状态。** 【代码】`list_conversations`（`memory.py:352-369`）用 `conversations LEFT JOIN messages`，孤儿消息没有父行 → 不出现在列表；而 `get_messages`（`memory.py:459` 起）直接 `SELECT ... FROM messages WHERE conversation_id = ?` → **知道 ID 就能读**。鉴权上线后，这种消息既没有 owner 也不会被列表暴露，属于「越权读取盲区」。

**(e) 现网实测 0 条孤儿。** 53 条消息全在 `legacy`，2 个会话，孤儿 0 → 隔离逻辑本轮**预期 0 行**。这既是好消息（迁移风险低），也是陷阱（**没有真实数据可验证逆向恢复**，必须靠构造用例）。

#### 3.2.3 修订方案

**(1) 四条固定的判定 SQL（全部 NULL 安全）**

```sql
-- Q1 孤儿消息：会话行不存在
SELECT m.id FROM messages m
WHERE NOT EXISTS (SELECT 1 FROM conversations c
                  WHERE c.conversation_id = m.conversation_id);

-- Q2 无主消息：会话存在但 owner 为空
SELECT m.id FROM messages m
JOIN conversations c ON c.conversation_id = m.conversation_id
WHERE c.user_id IS NULL;

-- Q3 无主会话
SELECT c.conversation_id FROM conversations c WHERE c.user_id IS NULL;

-- Q4 owner 快照不一致（用于恢复期校验，必须 IS NOT）
SELECT m.id FROM messages m
JOIN conversations c ON c.conversation_id = m.conversation_id
WHERE m.owner_snapshot IS NOT c.user_id;
```

编码规范写死：**`memory.py` 与迁移脚本内禁止出现 `NOT IN (SELECT`**，由 `tests/test_sql_nullsafe.py` 用正则扫描源码拦截（`r"NOT\s+IN\s*\(\s*SELECT"`）。同时禁止 `<>` 用于可能为 NULL 的列比较，改 `IS NOT`。

**(2) 隔离清单表**

```sql
CREATE TABLE IF NOT EXISTS migration_quarantine (
    batch_id                 TEXT NOT NULL,
    entity                   TEXT NOT NULL CHECK (entity IN ('message', 'conversation')),
    row_id                   TEXT NOT NULL,           -- messages.id / conversations.conversation_id 的文本形式
    reason                   TEXT NOT NULL CHECK (reason IN ('orphan_message','unowned_conversation','unowned_message')),
    original_conversation_id TEXT,                    -- ★ 原始归属，必须存；孤儿也存（哪怕它指向不存在的会话）
    original_owner           TEXT,                    -- 隔离前的 user_id（可为 NULL）
    quarantined_at           TEXT NOT NULL,
    restored_at              TEXT,
    PRIMARY KEY (batch_id, entity, row_id)
) STRICT, WITHOUT ROWID;
```

- `batch_id` 形如 `mig_v1_20261109T073012Z_7f3a1c9d`，写入 `schema_versions.batch_id`，**同一次迁移只有一个 batch**；
- 重复执行时 `INSERT ... ON CONFLICT(batch_id, entity, row_id) DO NOTHING`，用 `cursor.rowcount` 累计新增数 → **重入幂等且可观测**；
- 删除清单中任何行都视为「破坏审计」，测试用例会断言「隔离后清单行数 == Q1+Q2+Q3 的和」。

**(3) 隔离动作（同一事务，绝不删数据）**

```sql
BEGIN IMMEDIATE;

-- 1) 先写清单（逐行，来自 Q1/Q2/Q3 的结果集）
INSERT INTO migration_quarantine (batch_id, entity, row_id, reason,
                                  original_conversation_id, original_owner, quarantined_at)
VALUES (:batch, :entity, :row_id, :reason, :orig_cid, :orig_owner, :now)
ON CONFLICT(batch_id, entity, row_id) DO NOTHING;

-- 2) 再打标（不置 NULL，不删除）
UPDATE messages
SET owner_state = 'quarantined', quarantine_batch = :batch
WHERE id = :mid AND owner_state = 'normal';          -- 幂等：第二次 rowcount = 0

UPDATE conversations
SET owner_state = 'quarantined', quarantine_batch = :batch
WHERE conversation_id = :cid AND owner_state = 'normal';

COMMIT;
```

- `owner_state` 是新列：`TEXT NOT NULL DEFAULT 'normal' CHECK (owner_state IN ('normal','quarantined'))`；
- **不写 `user_id = NULL`**，因为 `user_id` 本来就可能为 NULL，置 NULL 等于丢失「原来是空的」这一信息；
- `original_owner` 单独存快照，才能在恢复时区分「原来就无主」与「被隔离清空」。

**(4) 逆向恢复（四条件 UPDATE，绝不覆盖后续合法修改）**

```sql
BEGIN IMMEDIATE;

UPDATE messages
SET owner_state = 'normal', quarantine_batch = NULL
WHERE id = :mid
  AND quarantine_batch = :batch            -- ① 只动本批次
  AND owner_state = 'quarantined'          -- ② 仍处于隔离态
  AND NOT EXISTS (                          -- ③ 期间没有新的合法归属产生
      SELECT 1 FROM conversations c
      WHERE c.conversation_id = :orig_cid AND c.user_id IS NOT NULL );
-- rowcount = 0 → 该行被跳过，记 skipped_conflict，不报错

UPDATE migration_quarantine
SET restored_at = :now
WHERE batch_id = :batch AND entity = 'message' AND row_id = :mid_str
  AND restored_at IS NULL;

COMMIT;
```

恢复前**必须**重算 Q4 做一致性预检：若 `m.owner_snapshot IS NOT c.user_id`，说明恢复对象在隔离期间被外部改过 → **该行不恢复**，写入 `skipped_conflict` 列表并输出到报告，交人工判断。

**(5) 事务边界**

| 操作 | 边界 | 失败语义 |
|---|---|---|
| 隔离 | 每 batch 一个 `BEGIN IMMEDIATE ... COMMIT` | 失败整体回滚，清单与打标同时不存在（不会留半套） |
| 恢复 | 每 batch 一个 `BEGIN IMMEDIATE ... COMMIT` | 同上 |
| 部分失败模拟 | 测试用 `CREATE TRIGGER ... BEFORE UPDATE ON messages WHEN NEW.id = :k BEGIN SELECT RAISE(ABORT,'inject'); END` 注入 | 断言 `migration_quarantine` 中该 batch 行数为 **0**（回滚干净） |

**(6) 明确禁止（写进编码规范 + 测试反向断言）**

| 禁止项 | 拦截方式 |
|---|---|
| `UPDATE conversations SET user_id = NULL`（无条件） | `tests/test_sql_nullsafe.py` 正则扫描 `SET\s+user_id\s*=\s*NULL` → 失败 |
| `DELETE FROM messages` 作为隔离手段 | 迁移脚本内禁止 `DELETE`，正则扫描迁移目录 |
| 按 `title LIKE 'quarantined%'` 之类的前缀删除 | 同上，禁 `LIKE` + `DELETE` 组合 |
| 用 `NOT IN (SELECT ...)` 判孤儿 | 正则扫描 `NOT\s+IN\s*\(\s*SELECT` |

#### 3.2.4 实现约束

| 约束 | 理由 |
|---|---|
| `migration_quarantine` 建成 `STRICT, WITHOUT ROWID` | 主键即业务键；`STRICT` 防止 silent type coercion 把 `id` 写成字符串 |
| `row_id` 统一用 `str()` 文本 | `messages.id` 是 INTEGER、`conversations.conversation_id` 是 TEXT，同表存需统一类型 |
| 隔离与恢复都**只碰本 batch** | 跨批操作会破坏「可精确回滚」的前提 |
| 恢复必须以 batch 为单位整批恢复，不支持单行恢复 | 单行恢复会破坏「清单与数据一致」的不变量 |
| 每次隔离/恢复写 `audit_logs(actor_type='bootstrap')` | 人工介入必须留痕 |

#### 3.2.5 验收标准

| 用例 | 测试数据 | 步骤 | 断言 | DB 预期 | 失败判定 |
|---|---|---|---|---|---|
| T-02-A | 6 条消息：①正常 ②会话存在但无主 ③孤儿有主 ④孤儿无主 ⑤自身 cid 为空串 ⑥会话存在且 owner 一致 | 跑 Q1/Q2 → 隔离 | Q1 命中 ③④，Q2 命中 ②；清单 3 行；所有原行数不变 | `messages` 53+6 行不变；`owner_state='quarantined'` 恰 3 行 | 行数减少或清单 ≠ 3 → 失败 |
| T-02-B | 同 T-02-A | 隔离后**再跑一次** | 第二次 `changes()` 全为 0；清单行数仍为 3 | 无变化 | 出现重复行 → 失败 |
| T-02-C | 同 T-02-A + 注入触发器让第 2 行 `UPDATE` 失败 | 隔离 | 抛出 `IntegrityError`；事务回滚 | 清单 0 行；`owner_state` 全 `normal` | 出现半套（清单有行但未打标）→ 失败 |
| T-02-D | T-02-A 成功后 | 逆向恢复 | `restored_at` 3 行非空；`owner_state` 全回 `normal`；`quarantine_batch` 全 NULL | `(id, conversation_id, user_id, owner_state)` 与隔离前快照**逐行相等** | 任一字段不等 → 失败 |
| T-02-E | T-02-A 成功后，人工把某孤儿的会话「认领」（插一条 `conversations(user_id='u-new')`） | 逆向恢复 | 该行 `rowcount=0` 被跳过；输出 `skipped_conflict` 含该 id | 该行仍 `quarantined`；其余 2 行恢复 | 该行被强行恢复 → 失败 |
| T-02-F | 构造 owner 为 NULL 的会话 | 跑 Q4 | `IS NOT` 检出 1 行；同一数据用 `<>` 检出 0 行 | — | 若实现用 `<>` → 测试 `test_sql_nullsafe` 失败 |

**总验收（任务书原文）**：「构造多个孤儿记录，完成隔离、重复执行、部分失败模拟和逆向恢复后，原始消息关联关系及所有权状态完全一致。」→ 由 T-02-A~E 串联覆盖，最终断言用**隔离前快照 vs 恢复后全表**的哈希比对。

---

### R3-03 迁移状态机重新定义（P0）

#### 3.3.1 原设计（R2-03）

版本级台账 `schema_migrations(version, status ∈ {pending,running,succeeded,failed})`；用「双连接」——ledger 独立事务先提交 `running`，再开业务事务，避免状态随业务回滚。

#### 3.3.2 发现的问题

**(a) 只有版本级状态，崩溃后无法知道「哪一步做完了」。** 一个 version 里含 13 个 DDL/DML 步骤，`running` 状态无法区分「卡在第 1 步」还是「卡在第 12 步」→ resume 只能整版重跑，而部分步骤（如 `ALTER TABLE ADD COLUMN`）**不可重入**。

**(b) 现有 `business.init_database()` 的「回滚」是假的。** 【代码】`business/db.py:56` `connection.executescript(SCHEMA_SQL)`；第 63 行 `connection.rollback()`。实测：

```
executescript 抛出 OperationalError: near "THIS": syntax error
执行后 sqlite_master = ['a','b']   ← 前 3 句已生效
b 表行数 = 1                        ← INSERT 已生效
in_transaction = False             ← rollback() 无处可回
```

对照实验：显式 `BEGIN IMMEDIATE` + 逐条 `execute` + 失败 `ROLLBACK` → 表数为 **0**（真正原子）。**结论：迁移执行器必须逐条 `execute`，不能用 `executescript`。**

**(c) 多 Worker 同时启动无互斥。** 若 `--workers 2`，两个进程会同时进 `init_database()`，都判定「需要迁移」，都去 `ALTER TABLE ADD COLUMN` → 第二个报 `duplicate column name`，进程崩溃，但第一个可能已提交一半。

**(d) 「进程在每个事务边界崩溃」的行为没定义。** 有 4 个边界，其中「业务事务已提交、ledger 尚未标记 succeeded」会让状态永久停在 `running` —— R2 只写了「`running` 残留阻断启动」，但**没说怎么恢复**，等于每次这种崩溃都要人工清库。

#### 3.3.3 修订方案

**(1) 两级状态表**

```sql
CREATE TABLE IF NOT EXISTS schema_versions (
    version      INTEGER PRIMARY KEY,
    status       TEXT NOT NULL CHECK (status IN ('pending','running','succeeded','failed')),
    batch_id     TEXT,
    owner        TEXT,                 -- '<hostname>:<pid>:<uuid4>'
    heartbeat_at TEXT,                 -- UTC ISO8601
    started_at   TEXT,
    finished_at  TEXT,
    error        TEXT
) STRICT;

CREATE TABLE IF NOT EXISTS schema_steps (
    version      INTEGER NOT NULL,
    step_no      INTEGER NOT NULL CHECK (step_no > 0),
    name         TEXT NOT NULL,
    status       TEXT NOT NULL CHECK (status IN ('pending','running','succeeded','failed','skipped')),
    attempts     INTEGER NOT NULL DEFAULT 0,
    owner        TEXT,
    started_at   TEXT,
    finished_at  TEXT,
    error        TEXT,
    PRIMARY KEY (version, step_no)
) STRICT, WITHOUT ROWID;
```

**(2) 版本状态只能由「全部步骤成功」推动（代码 + 谓词双保险）**

```sql
UPDATE schema_versions
SET status='succeeded', finished_at=:now
WHERE version=:v
  AND status='running'
  AND NOT EXISTS (SELECT 1 FROM schema_steps
                  WHERE version=:v AND status NOT IN ('succeeded','skipped'));
-- rowcount=0 → 说明还有步骤未完成 → 保持 running，并写告警
```

**(3) 每个步骤必须自带「幂等探针」**（这是整套状态机的地基）

| step | 名称 | 幂等探针（判「已完成」） |
|---|---|---|
| 01 | `create_users` | `SELECT 1 FROM sqlite_master WHERE type='table' AND name='users'` |
| 02 | `create_sessions` | 同上 `sessions` |
| 03 | `create_invites` | 同上 `invites` |
| 04 | `create_audit_logs` | 同上 `audit_logs` |
| 05 | `create_usage` | 同上 `usage_daily` **且** `usage_reservations` |
| 06 | `create_quarantine` | 同上 `migration_quarantine` |
| 07 | `alter_conversations` | `PRAGMA table_info(conversations)` 含 `user_id` 与 `owner_state` |
| 08 | `alter_messages` | `PRAGMA table_info(messages)` 含 `owner_state` 与 `quarantine_batch` |
| 09 | `create_indexes` | `sqlite_master` 含 `idx_conversations_user_id`、`idx_messages_owner_state` |
| 10 | `quarantine_orphans` | `SELECT count(*) FROM migration_quarantine WHERE batch_id=:b` == 隔离后实际打标数 |
| 11 | `verify` | 下列 4 条全为 0：孤儿数、未隔离无主数、`foreign_key_check` 行数、owner 快照不一致数 |

执行逻辑：

```
for step in steps:
    if probe(step) is True:            # ★ 先探测，已完成就跳过
        mark(step, 'succeeded', attempts=step_attempts)
        continue
    claim_step(step)                   # 写 running + owner + heartbeat，独立事务提交
    try:
        run_step(step)                 # 逐条 execute，单事务
        mark(step, 'succeeded')
    except Exception as e:
        mark(step, 'failed', error=str(e))
        raise
```

**(4) 四个崩溃边界的行为矩阵**

| 崩溃点 | 磁盘上的状态 | 下次启动行为 | 会不会重复执行非幂等操作 |
|---|---|---|---|
| ① 步骤开始前 | step 无记录 / `pending` | 正常执行该步 | 否 |
| ② ledger 写 `running` 后、业务事务未提交 | 业务事务已回滚，step = `running` | 判定 stale（见下）→ 转 `failed` → **探针为假 → 重跑该步** | 否（业务未生效） |
| ③ **业务事务已提交、ledger 未标 `succeeded`** | 业务已生效，step = `running` | 判定 stale → 转 `failed` → **探针为真 → 直接标 `succeeded` 不再执行** | **否（这是幂等探针存在的唯一理由）** |
| ④ ledger 标 `succeeded` 后 | step = `succeeded` | 跳过，进入下一步 | 否 |

stale 判定：`status='running' AND (heartbeat_at IS NULL OR heartbeat_at < now-120s)` 且 `owner != self.owner`。心跳由执行线程每 30 s 更新 `heartbeat_at`。

**(5) 多 Worker 互斥（跨进程，不依赖额外中间件）**

用 SQLite 自身的写锁做两级互斥：

```python
# 第一级：整个迁移流程的进程互斥（BEGIN IMMEDIATE 会拿到 RESERVED 锁）
conn.execute("BEGIN IMMEDIATE")          # 拿不到 → 5s 后抛 database is locked（busy_timeout 实测 5000ms）

# 第二级：版本级 claim，原子且可判定
cur = conn.execute("""
    INSERT INTO schema_versions(version, status, owner, heartbeat_at, started_at)
    VALUES (:v, 'running', :owner, :now, :now)
    ON CONFLICT(version) DO UPDATE SET
        status='running', owner=:owner, heartbeat_at=:now, started_at=:now
    WHERE schema_versions.status IN ('pending','failed')
       OR (schema_versions.status='running' AND schema_versions.heartbeat_at < :stale_before)
""", {...})
if cur.rowcount == 0:
    row = conn.execute("SELECT status, owner FROM schema_versions WHERE version=?", (v,)).fetchone()
    conn.execute("ROLLBACK")
    raise MigrationBusy(row)            # 另一个 Worker 正在迁移 → 本进程启动失败退出
```

**设计选择：抢不到锁的 Worker 直接启动失败退出**（uvicorn 单 Worker 假设下不会发生；若将来多 Worker，由进程管理器重启，等第一个完成后第二个就能正常起）。理由：让第二个 Worker「等待并继续」会引入「两个 Worker 各自认为迁移完成」的窗口，风险大于收益。

**(6) 三类数据库的启动行为**

| 库状态 | 判定条件 | 行为 |
|---|---|---|
| 全新 | `sqlite_master` 无 `schema_versions` 表 | 建台账 → 跑 step 01..11 → `succeeded` |
| 部分迁移 | 有版本行，`status ∈ {running, failed}` 或有 step 非 `succeeded` | `resume`：按探针跳过已完成，继续未完成 |
| 全部完成 | `status='succeeded'` 且所有 step `succeeded` | 直接启动（只读校验 1 次 `verify`） |
| 版本超前（库版本 > 代码目标版本） | `MAX(version) > CODE_TARGET` | **拒绝启动**，报 `schema_newer_than_code`（防回滚降级写坏数据） |
| `failed` 且 `attempts >= 3` | — | **拒绝启动**，要求人工 `python -m auth.migrate --force-step <n>` |

**(7) resume 的幂等性与人工介入**

- `resume` 只处理 `status != 'succeeded'` 的 step，且执行前必跑探针；
- `attempts` 上限 3：超过后 step 标 `failed` 并**停止自动重试**；
- 人工介入只能通过 CLI：`python -m auth.migrate --resume` / `--force-step 08` / `--status`；
- 每次人工介入写 `audit_logs(actor_type='bootstrap', action='migrate_force_step', detail={'step':8,'version':1})`；
- **HTTP 层没有任何触发迁移的端点**（迁移只发生在 `lifespan` 启动期与 CLI）。

#### 3.3.4 实现约束

| 约束 | 理由 |
|---|---|
| 迁移执行器**禁止** `executescript` | 实测非原子，且失败后 `rollback()` 无效 |
| 每个 step 一个事务，不跨 step 合并 | 崩溃粒度即 step 粒度 |
| ledger 使用**独立连接**（R2 结论保留） | 业务事务回滚不能带走 `running` 记录 |
| 迁移期间连接必须 `timeout=10`（大于默认 5） | 多进程等待窗口 |
| `schema_versions` 只允许追加版本，不允许改历史行 | 台账是审计证据 |
| `verify` step 必须可重复执行（只读） | 每次启动都会跑一次 |

#### 3.3.5 验收标准

| 用例 | 注入方式 | 断言 |
|---|---|---|
| T-03-A | 在第 2 步的业务事务中 `RAISE(ABORT)` | step 2 = `failed`；step 1 = `succeeded`；version 未 succeeded；重启后从 step 2 继续 |
| T-03-B | 在边界②崩溃（写 `running` 后用 `os._exit(1)`） | 重启后 step 2 探针为假 → 重跑 → 最终 `succeeded`；`ALTER` 未重复执行（无 `duplicate column name`） |
| T-03-C | 在边界③崩溃（事务提交后、ledger 更新前 `os._exit(1)`） | 重启后 **探针为真** → 直接标 `succeeded`，**不重跑 DDL**（断言 `PRAGMA table_info` 列数不变、无异常） |
| T-03-D | 两个进程同时启动（`multiprocessing` 或两个子进程跑同一 `migrate()`） | 恰好 1 个成功迁移；另一个抛 `MigrationBusy` 并退出；DB 最终只有 1 套表、step 表无重复行 |
| T-03-E | 全部成功后重置 step 10 为 `pending` 再 `--resume` | 不重复插入清单（`migration_quarantine` 行数不变） |
| T-03-F | 手工把 version 改为比 `CODE_TARGET` 大 | 启动抛 `schema_newer_than_code`，进程退出码非 0 |
| T-03-G | `attempts=3, status=failed` | 启动拒绝；`--force-step` 后成功；审计表新增 1 行 `actor_type='bootstrap'` |
| T-03-H | 全新空库 | 11 个 step 全 `succeeded`；`verify` 的 4 条计数全 0 |
| T-03-I | 现网 `memory.db` 副本（53 消息 / 2 会话 / 0 孤儿） | step 10 打标 0 行；清单 0 行；`messages` 53 行不变；`legacy` 会话仍存在 |

> T-03-B/C 必须用**子进程**做崩溃注入（`os._exit`），不能用 `try/except` 模拟 —— 后者不经过真实的「进程消失」路径，测不出 `running` 残留。

---

### R3-04 审计主体与授权约束（P1）

#### 3.4.1 原设计（R2-04）

```sql
actor_type TEXT NOT NULL CHECK (actor_type IN ('user','system','bootstrap')),
actor_user_id TEXT REFERENCES users(user_id),
CHECK ((actor_type='user' AND actor_user_id IS NOT NULL) OR
       (actor_type<>'user' AND actor_user_id IS NULL))
```

即「用 `actor_type` 区分主体，用条件 CHECK 保证 user 必须带 ID」。

#### 3.4.2 发现的问题

**(a) 约束只保证「结构合法」，不保证「身份真实」。** `actor_user_id='u-admin'` 写在 INSERT 里就是合法的 —— 调用方可以随便传。约束没有把「这个 ID 是否等于当前认证身份」钉死。

**(b) bootstrap 的授权来源没定义。** 「首管理员 CLI 一次性初始化」只说了用 CLI，但没说：谁能跑这条命令、跑的时候怎么证明自己是运维、跑第二次怎么办。**如果没有凭证，任何能登录服务器的人都能造一个 admin。**

**(c) 审计写失败时业务能否继续，完全没定。** 这是个必须二选一的问题：要么审计失败即业务失败（强一致，但审计表一次抖动就让登录不可用），要么继续（可用，但会丢审计）。不定就会出现「不同接口不同行为」的随机结果。

**(d) 未定义脱敏字段。** 审计最容易在 `detail` 里顺手塞 `{"password": "...", "invite_code": "..."}`。

#### 3.4.3 修订方案

**(1) 三类 actor 的约束（表结构 + 写入 API 双重）**

| actor_type | actor_user_id | 语义 | 允许的调用来源 |
|---|---|---|---|
| `user` | 非空，且**必须等于 `current_context().user_id`** | 已登录用户的主动操作 | HTTP 请求（经 `auth/deps.py`） |
| `system` | 必须 NULL | 系统自身的副作用（后台任务异常、迁移、对账） | 后台任务 / `lifespan` / 定时器 |
| `bootstrap` | 必须 NULL | CLI 的一次性引导或人工干预 | **仅 CLI 进程**，且必须带 `PAI_BOOTSTRAP_TOKEN` |

`detail` 里对 `bootstrap` 额外强制写入 `{"cli_argv_sha256": "...", "host": "..."}`，便于事后追溯是哪次命令。

**(2) 用户 ID 与真实身份的对应关系（不接受参数）**

```python
# auth/audit.py —— 唯一写入口。注意：签名里没有 user_id 参数
def record(action: str, *, target: str | None = None,
           detail: Mapping[str, Any] | None = None,
           conn: sqlite3.Connection | None = None) -> None:
    ctx = current_context()               # 无上下文 → ContextMissingError，绝不明文传 ID
    actor_type = "system" if ctx.detached else "user"
    actor_user_id = None if actor_type == "system" else ctx.user_id
    ...
```

- **没有 `user_id=` 形参**，物理上无法伪造；
- `system` 场景由 `run_detached(identity=None)` 产生（`current_context()` 会抛错）→ 提供 `record_system(action, ...)` 显式入口，它**不接受** `user_id`；
- CI 断言：`grep -rn "record(" --include=*.py | grep "user_id="` 必须 0 命中。

**(3) bootstrap 的唯一授权来源**

| 环节 | 规则 |
|---|---|
| 入口 | `python -m auth.bootstrap create-admin --username <u> --confirm-create-admin`，`auth/bootstrap.py` 的 `main()` 只在 `if __name__ == "__main__":` 下可达，**不注册任何 HTTP 路由** |
| 凭证 | 环境变量 `PAI_BOOTSTRAP_TOKEN`，要求 ≥ 32 字节随机（`secrets.token_urlsafe(32)`），**首次启动前由运维注入，创建成功后运维必须删除该变量** |
| 校验 | 常量时间比较（`hmac.compare_digest`）；缺失或长度 < 32 → 直接退出码 2 |
| 唯一性 | `INSERT INTO users(role) ... WHERE NOT EXISTS (SELECT 1 FROM users WHERE role='admin')`，`rowcount==0` → 报 `admin_already_exists` 并退出码 3 |
| 一次性 | 成功后在 `system_flags` 表写 `bootstrap_used_at`；再次调用直接拒绝（即使 token 仍在） |
| 后续管理员操作 | 一律 `actor_type='user'` + `role='admin'`；**bootstrap 不可复用**，也不存在「bootstrap 专用 HTTP 端点」 |

**(4) 审计写失败时的业务行为（分级，必须二选一，不许随机）**

| 动作类别 | 具体动作 | 审计失败时 | 理由 |
|---|---|---|---|
| **安全关键** | 登录成功/失败、登出、邀请码创建/撤销、用户禁用/启用、会话撤销、bootstrap、迁移人工介入 | **业务回滚 + 500** | 这些动作没有审计就无法追责，宁可不做 |
| **数据类** | 会话创建、KB 文档删除、business 表写入 | **业务继续**，`logger.error` + 指标 `audit_write_failed_total`，响应头 `X-Audit-Degraded: 1` | 用户的数据操作不能因为审计表抖动而失败 |

实现方式：`record(..., conn=conn)` 传入**同一连接同一事务**时，安全关键动作天然原子（审计 INSERT 失败 → 整个事务回滚）；数据类动作传 `conn=None`，用独立连接写，失败只记日志。

> 跨库限制：`business.db` 与 `memory.db` 是**两个文件**，无法同事务。因此 business 写入天然只能走「降级继续」，这一点由存储结构决定，不是设计偷懒。

**(5) 脱敏与禁止写入字段**

```python
_SENSITIVE_KEYS = {
    "password", "passwd", "password_hash", "token", "csrf_token",
    "session_id", "invite_code", "api_key", "authorization", "cookie",
}
# 写入前统一 scrub：命中即替换为 "sha256:" + hashlib.sha256(v).hexdigest()[:12]
```

- `detail` 只接受 `dict`，键做**白名单 + 黑名单双重过滤**；值若为 `str` 且长度 > 200 一律截断；
- **禁止**把 `request.headers` 整体写入（含 `Cookie`）；
- 测试断言：对 `audit_logs` 全表跑正则 `(?i)(password|token|secret|invite_code)\s*[:=]\s*\S{6,}`，命中即失败。

#### 3.4.4 验收标准

| 用例 | 前置 | 动作 | 断言 |
|---|---|---|---|
| T-04-A | 空库 | `INSERT audit_logs(actor_type='user', actor_user_id=NULL)` | `CHECK constraint failed` |
| T-04-B | 同上 | `actor_type='bootstrap', actor_user_id='u1'` | `CHECK constraint failed` |
| T-04-C | 已存在 1 个 admin | 再跑 bootstrap CLI | 退出码 3；`users` 表 admin 数仍 1 |
| T-04-D | 无 `PAI_BOOTSTRAP_TOKEN` | 跑 bootstrap CLI | 退出码 2；`users` 表无新行 |
| T-04-E | 已登录普通用户 | 调用任意写接口，尝试在 body 里塞 `user_id` | 该字段被忽略（Pydantic 模型无此字段 → 422 或忽略）；审计 `actor_user_id` 等于会话用户 |
| T-04-F | 会话有效 | 让 `audit_logs` 表被 `DROP`（测试用临时库） | 安全关键动作返回 500 且业务未生效；数据类动作返回 200 且带 `X-Audit-Degraded: 1` |
| T-04-G | 写入含 `password='abc123'` 的 detail | 查 `audit_logs` | 存的是 `sha256:...` 12 位，无明文 |

---

### R3-05 路由权限矩阵完整性（P0）

#### 3.5.1 原设计（R2-05）

给出 48 行矩阵（38 现存 + 10 新增），并统一了三口径：`app.routes` 30 / OpenAPI paths 36 / 方法级操作 38。

#### 3.5.2 发现的问题

**(a) 38 条「现存」没有逐条核对过。** R2 的表来自 `app.openapi()`，但没给出「handler 名 + 源码行号」的对照，无法证明「表里的 38 条就是代码里的 38 条」。

**(b) 缺 Origin / 限速 / 数据范围三个字段。** R2 表只有 CSRF 一列，而 R3-08 要求 Origin 与 CSRF 分开判定（二者失败语义不同：403 vs 403，但判定时机不同）。

**(c) 生产关闭 `/docs` 的「验证方式」没写。** 只说「`docs_url=None`」，没说怎么证明它真的关了。

**(d) 三口径会随实现方式漂移。** 新增 10 条如果挂在独立 `APIRouter` 上，`app.routes` 会多一个 `_IncludedRouter` 占位 —— 这个值取决于实现选择，不能提前写成定论。

#### 3.5.3 修订方案

**(1) 38 条现存操作的 AST 全量核对（本轮实测）**

用 `ast` 静态解析（**不 import 项目**，因此不触发 `.env` 与数据库）：

```
总 38 = main.py 25 + business/api.py 13
同步 33 / 异步 5（/knowledge/import、/knowledge/documents、/auth/login、/auth/register、/analyze-image）
方法集合 {GET, POST, DELETE}（无 PUT/PATCH 现存）
```

**核对结论：R2 的 38 条清单与代码逐条对应，无遗漏、无多余。** 下表带源码行号，可直接核对。

**(2) 三口径的定义与预测值（区分「已实测」与「实施后待测」）**

| 口径 | 现值（实测） | 实施后预测 | 说明 |
|---|---|---|---|
| 方法级业务操作 | **38** | **48** | 唯一稳定口径，= 上一行 + 10 新增 |
| OpenAPI 路径条目 | **36** | **45**（待核验） | 现有 2 个路径挂 2 个方法；新增 10 条中 `/admin/invites` 同时有 GET 与 POST → 路径增量 9 |
| `app.routes` 条目 | **30**（25 APIRoute + 4 Route + 1 `_IncludedRouter`） | **31 或 40**（待核验） | 若新增挂在 `APIRouter` → 35 - 25 不变、占位变 2 → 31；若直接 `@app.get` 挂在 `main.py` → 35 APIRoute + 4 + 1 = 40 |
| 内建文档路由 | **4** | **0**（生产） | `docs_url=None` 等三个参数 |

> `app.routes` 的预测值**故意给两个**：它取决于「新增路由挂在 app 还是 router」这一实现选择，现在无法定论。实施后由 T-05-B 断言实际值，**不允许把预测值当成事实写进验收**。

**(3) 生产关闭 `/docs` 的验证方式（可执行断言）**

```python
# 生产配置：FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
assert client.get("/docs").status_code == 404
assert client.get("/redoc").status_code == 404
assert client.get("/openapi.json").status_code == 404
assert client.get("/docs/oauth2-redirect").status_code == 404
# 结构断言：不残留任何内建 Route
from starlette.routing import Route
assert [r.path for r in app.routes if isinstance(r, Route)] == []
```

补充：`/openapi.json` 关闭后，`client.app.openapi()` 仍可调用（它是 Python API，不依赖路由）—— 因此**不能用 `app.openapi()` 是否可用判断文档是否关闭**，必须走 HTTP 断言。

**(4) 48 行权限矩阵（八字段）**

图例：认证 `公` 公开 / `登` 需登录 / `管` 需管理员；CSRF `✓` 需 token；Origin `严` 严格校验（缺失即 403）/ `宽` 仅同源 / `—` 不校验；限速 `L` 登录 / `R` 注册 / `U` 用户级配额 / `—`。

| # | 方法 | 路径 | handler（行号） | 认证 | 角色 | CSRF | Origin | 限速 | 数据范围 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | POST | `/chat` | `chat`（305） | 登 | user/admin | ✓ | 宽 | U | 仅本人会话；越权 404 |
| 2 | GET | `/conversations` | `list_conversations`（476） | 登 | 同上 | — | — | — | 强制 `user_id=ctx.user_id` |
| 3 | GET | `/conversations/{id}/messages` | `conversation_messages`（498） | 登 | 同上 | — | — | — | 他人会话 → 404（不泄露存在性） |
| 4 | GET | `/knowledge/search` | `knowledge_search`（533） | 登 | 同上 | — | — | U | public + 本人 private |
| 5 | POST | `/knowledge/import` | `knowledge_import`（554，**async**） | 登 | user→private / admin→public | ✓ | 宽 | U | 本人 private 或全站 public |
| 6 | POST | `/knowledge/documents` | `knowledge_upload_document`（661，**async**） | 登 | 同上 | ✓ | 宽 | U | 同上 |
| 7 | GET | `/knowledge/documents` | `knowledge_documents`（768） | 登 | 同上 | — | — | — | 同上 |
| 8 | DELETE | `/knowledge/documents/{did}` | `knowledge_delete_document`（820） | 登 | 同上 | ✓ | 宽 | U | 同上；public 仅 admin 可删 |
| 9 | POST | `/analyze-image` | `analyze_image`（1097，**async**） | 登 | 同上 | ✓ | 宽 | U | 不落盘，无持久化 |
| 10 | GET | `/` | `index`（851） | 登 | 同上 | — | — | — | 无 |
| 11 | GET | `/workbench` | `workbench_index`（882） | 登 | 同上 | — | — | — | 无 |
| 12 | GET | `/business-ui` | `business_ui_index`（927） | 登 | 同上 | — | — | — | 无 |
| 13 | GET | `/app-ui/static/style.css` | `app_ui_style`（859） | 公 | — | — | — | — | 静态资源，无用户数据 |
| 14 | GET | `/app-ui/static/app.js` | `app_ui_script`（867） | 公 | — | — | — | — | 同上 |
| 15 | GET | `/workbench/static/style.css` | `workbench_style`（890） | 公 | — | — | — | — | 同上 |
| 16 | GET | `/workbench/static/app.js` | `workbench_script`（898） | 公 | — | — | — | — | 同上 |
| 17 | GET | `/business-ui/app.js` | `business_ui_script`（935） | 公 | — | — | — | — | 同上 |
| 18 | GET | `/business-ui/style.css` | `business_ui_style`（943） | 公 | — | — | — | — | 同上 |
| 19 | GET | `/login` | `login_page`（1050） | 公 | — | — | — | — | 无身份内容 |
| 20 | GET | `/register` | `register_page`（1056） | 公 | — | — | — | — | 同上 |
| 21 | GET | `/auth/static/style.css` | `auth_style`（1062） | 公 | — | — | — | — | 同上 |
| 22 | GET | `/auth/static/app.js` | `auth_script`（1068） | 公 | — | — | — | — | 同上 |
| 23 | GET | `/auth/status` | `auth_status`（1074） | 公 | — | — | — | — | 仅返回「是否已登录」布尔，不含用户信息 |
| 24 | POST | `/auth/login` | `auth_login`（1083，**async**） | 公 | — | ✓ | **严** | L | 无 |
| 25 | POST | `/auth/register` | `auth_register`（1089，**async**） | 公 | — | ✓ | **严** | R | 无 |
| 26 | POST | `/business/orders` | `create_order_api`（134） | 管 | admin | ✓ | 宽 | — | 全站共享写 |
| 27 | POST | `/business/payments` | `create_payment_api`（139） | 管 | admin | ✓ | 宽 | — | 同上 |
| 28 | POST | `/business/shipments` | `create_shipment_api`（144） | 管 | admin | ✓ | 宽 | — | 同上 |
| 29 | POST | `/business/shipment-events` | `create_shipment_event_api`（149） | 管 | admin | ✓ | 宽 | — | 同上 |
| 30 | POST | `/business/aftersales` | `create_aftersale_api`（154） | 管 | admin | ✓ | 宽 | — | 同上 |
| 31 | POST | `/business/workorders` | `create_workorder_api`（159） | 管 | admin | ✓ | 宽 | — | 同上 |
| 32 | GET | `/business/orders` | `list_orders_api`（195） | 登 | user/admin | — | — | — | 共享演示数据，全员只读 |
| 33 | GET | `/business/orders/{oid}` | `get_order_api`（226） | 登 | 同上 | — | — | — | 同上 |
| 34 | GET | `/business/orders/{oid}/payments` | `list_order_payments_api`（237） | 登 | 同上 | — | — | — | 同上 |
| 35 | GET | `/business/orders/{oid}/shipments` | `list_order_shipments_api`（252） | 登 | 同上 | — | — | — | 同上 |
| 36 | GET | `/business/shipments/{sid}/events` | `list_shipment_events_api`（267） | 登 | 同上 | — | — | — | 同上 |
| 37 | GET | `/business/orders/{oid}/aftersales` | `list_order_aftersales_api`（282） | 登 | 同上 | — | — | — | 同上 |
| 38 | GET | `/business/orders/{oid}/workorders` | `list_order_workorders_api`（297） | 登 | 同上 | — | — | — | 同上 |
| 39 | GET | `/auth/me`（新增） | `auth_me` | 登 | 同上 | — | — | — | 仅本人（`user_id`, `role`, `expires_at`） |
| 40 | POST | `/auth/logout`（新增） | `auth_logout` | 登 | 同上 | ✓ | **严** | — | 仅当前会话 |
| 41 | GET | `/admin/users`（新增） | `admin_list_users` | 管 | admin | — | — | — | 用户元数据（不含 hash） |
| 42 | PATCH | `/admin/users/{uid}`（新增） | `admin_patch_user` | 管 | admin | ✓ | **严** | — | 禁改自己 role；审计必写 |
| 43 | POST | `/admin/users/{uid}/revoke-sessions`（新增） | `admin_revoke_sessions` | 管 | admin | ✓ | **严** | — | 可撤销任意用户会话 |
| 44 | GET | `/admin/invites`（新增） | `admin_list_invites` | 管 | admin | — | — | — | 邀请码仅显示前缀 6 位 |
| 45 | POST | `/admin/invites`（新增） | `admin_create_invite` | 管 | admin | ✓ | **严** | — | 单次创建，返回明文一次 |
| 46 | POST | `/admin/invites/{iid}/revoke`（新增） | `admin_revoke_invite` | 管 | admin | ✓ | **严** | — | 仅未使用的邀请码 |
| 47 | GET | `/admin/audit`（新增） | `admin_list_audit` | 管 | admin | — | — | — | 分页上限 200，`detail` 已脱敏 |
| 48 | GET | `/admin/ui`（新增） | `admin_ui` | 管 | admin | — | — | — | 静态页，无数据 |

**逐条断言要求**：T-05-A 对 48 条**逐条**发请求，断言「未登录 → 401/302、普通用户 → 403、admin → 200/201」（公开路由断言 200），共 48 × 3 = 144 个断言点，**不接受抽样**。

**(5) 「不存在未登记路由」的证明方式**

```python
registered = {(r.path, m.upper()) for r in app.routes
              if isinstance(r, APIRoute) for m in r.methods}
declared = {(row.path, row.method) for row in MATRIX}   # 报告里的矩阵
assert registered == declared
```

矩阵以**测试文件里的常量**形式存在（`tests/route_matrix.py`），与本文档同源维护；新增路由不更新矩阵 → 测试失败。这样「文档」和「代码」不可能各说各话。

#### 3.5.4 验收标准

| 用例 | 内容 | 断言 |
|---|---|---|
| T-05-A | 48 条 × 3 种身份 | 每条状态码符合矩阵；无 500 |
| T-05-B | 路由注册表 | `registered == declared`；并打印实测三口径数值（38→48、36→45、30→31/40 中哪一个） |
| T-05-C | 生产配置 | `/docs`、`/redoc`、`/openapi.json`、`/docs/oauth2-redirect` 全 404；`Route` 实例列表为空 |
| T-05-D | 反例 | 临时加一条未登记路由 → T-05-B 失败 |
| T-05-E | 静态资源 | 13–22 号路由无 `Set-Cookie`，无 `Cache-Control: no-store`（可缓存） |

---

### R3-06 KB 缓存生命周期与并发淘汰（P0）

#### 3.6.1 原设计（R2-06）

```python
_PUBLIC_ENTRY = _Entry()            # 不参与淘汰
_PRIVATE: "OrderedDict[str, _Entry]" = OrderedDict()
PRIVATE_MAX = 8
# 删除 R1 的 _KB_LOCKS 字典，锁随 _Entry 生命周期消亡
```

#### 3.6.2 发现的问题

**(a) 获取与淘汰不在同一临界区。** R2 的伪代码是「先 `_PRIVATE.get(scope)`，再 `if len > MAX: popitem()`」。两步之间没有锁保护的话，典型交错是：线程 A 取到 entry（refs=1）→ 线程 B 淘汰把它 pop 掉并置 `CLOSED` → 线程 A 拿着已关闭的 entry 去 `search()`。

**(b) 「全部 entry 都忙」时怎么办没定。** 8 个都在用，第 9 个用户来了：是淘汰一个在用的（会打断别的请求）、还是无限增长（内存爆）、还是拒绝服务？三种都能自圆其说，但必须选一个。

**(c) `PRIVATE_MAX=8` 是硬上限还是软上限没定。** 说「上限 8」但没说不满足时抛什么。

**(d) `KnowledgeBase` 到底能不能并发读写，R2 只说「不支持并发」。** 实际要分开看：

- 【代码】`KnowledgeBase._ensure_index`（`rag/knowledge_base.py:246`）用 `_init_lock` + 双重检查，`_ready` 在初始化完成后置 True；
- 【代码】`VectorStore.search`（`rag/vector_store.py:20-44`）只遍历 `self.items`，不写；
- 【代码】`VectorStore.add`（`:10`）是裸 `list.append`，无锁；
- **结论：实例建成后「读并发安全」，但「读写并发不安全」**（`search` 遍历时 `append` 会让它看到半更新的列表，虽然 Python 不会崩，但语义上是脏读）。

**(e) 失效时旧实例仍在用。** 【代码】`main._invalidate_knowledge_caches`（`main.py:110`）只是 `lru_cache.cache_clear()`。清掉后**进行中的请求仍持有旧实例** → 同时存在 2 份 BGE 模型（若已加载，`sentence-transformers` 6.1.0 每份常驻数百 MB）。R2 没给「在用实例数量」的约束。

#### 3.6.3 修订方案

**(1) entry 状态机 + 引用计数**

```python
class _State(enum.Enum):
    BUILDING = "building"; READY = "ready"; FAILED = "failed"; CLOSED = "closed"

@dataclass
class _Entry:
    scope: str
    state: _State = _State.BUILDING
    kb: KnowledgeBase | None = None
    refs: int = 0
    ready: threading.Event = field(default_factory=threading.Event)
    error: BaseException | None = None
```

**关键不变量**：`refs > 0` 的 entry **永远不会**被置 `CLOSED`，也永远不会从字典里 pop。

**(2) 获取：先占位再构建，一次临界区决定谁构建**

```python
_CACHE_LOCK = threading.Lock()          # 只保护字典与计数，微秒级
_PRIVATE: "OrderedDict[str, _Entry]" = OrderedDict()
_PRIVATE_SOFT_MAX = 8
_PRIVATE_HARD_MAX = 16
_PUBLIC: _Entry | None = None

def _acquire(scope: str, is_public: bool) -> _Entry:
    global _PUBLIC
    builder = False
    with _CACHE_LOCK:
        if is_public:
            entry = _PUBLIC
            if entry is None:
                entry = _PUBLIC = _Entry(scope=scope); builder = True
            entry.refs += 1
        else:
            entry = _PRIVATE.get(scope)
            if entry is None:
                if len(_PRIVATE) >= _PRIVATE_HARD_MAX:
                    raise KbCapacityExhausted()        # 硬上限：拒绝，不淘汰在用
                entry = _Entry(scope=scope)
                _PRIVATE[scope] = entry                 # ★ 先占位，防止并发重复构建
                builder = True
            entry.refs += 1
            _PRIVATE.move_to_end(scope)
    if builder:
        try:
            entry.kb = _build(scope)                    # ★ 锁外构建（BGE 加载可能数十秒）
            entry.state = _State.READY
        except BaseException as exc:
            entry.error = exc; entry.state = _State.FAILED
        finally:
            entry.ready.set()                           # 唤醒所有等待者（成功或失败）
    else:
        entry.ready.wait(timeout=_BUILD_TIMEOUT)        # 等构建方，不自己再建
    if entry.state is not _State.READY:
        _release(entry, is_public)
        raise KbUnavailable(entry.error)
    return entry
```

要点：
- **占位与构建分离**：`_PRIVATE[scope] = entry` 在锁内完成 → 并发的第二个请求拿到同一个 `BUILDING` entry，`wait()` 等结果，**不会出现两个实例**；
- 构建失败也 `set()` 事件并标 `FAILED` → 等待者立即失败返回 503，**不会形成构建风暴**（不会每个请求都重试一次 BGE 加载）；
- `FAILED` entry 保留在字典里 60 s（`_FAILED_TTL`）后被下次访问替换，避免永久毒化。

**(3) 释放与淘汰：只在 `refs==0` 时真正回收**

```python
def _release(entry: _Entry, is_public: bool) -> None:
    with _CACHE_LOCK:
        entry.refs -= 1
        if is_public or entry.refs > 0:
            return
        # 仅私有、且无人在用，才允许回收
        if entry.state is _State.CLOSED:
            _PRIVATE.pop(entry.scope, None)
        _evict_locked()

def _evict_locked() -> None:
    while len(_PRIVATE) > _PRIVATE_SOFT_MAX:
        victim = next((s for s, e in _PRIVATE.items()
                       if e.refs == 0 and e.state in (_State.READY, _State.FAILED)), None)
        if victim is None:
            _metrics.inc("kb_cache_overflow")     # 全忙 → 允许临时超限，记指标
            return
        e = _PRIVATE.pop(victim)
        e.state = _State.CLOSED
```

**(4) 上限语义（明确回答「严格还是软」）**

| 参数 | 值 | 语义 |
|---|---|---|
| `_PRIVATE_SOFT_MAX` | 8 | **目标值**。超过时尝试淘汰空闲项；全部在用则**允许临时超限**并记 `kb_cache_overflow` |
| `_PRIVATE_HARD_MAX` | 16 | **硬上限**。达到后**拒绝新建私有 KB**（503 `kb_capacity_exhausted`），**绝不淘汰在用实例** |

选择理由：KB 实例不是可随意重建的廉价资源（要加载 BGE 模型 + 全量索引）。为满足「8」而淘汰一个正在被 5 个请求使用的实例，会造成「重建风暴 + 内存峰值翻倍」，比短暂超限更危险。

**(5) 锁范围（三把锁，职责不重叠）**

| 锁 | 保护范围 | 持有时长 | 禁止在锁内做的事 |
|---|---|---|---|
| `_CACHE_LOCK`（模块级） | `_PRIVATE` / `_PUBLIC` 字典、`refs` 计数 | 微秒级 | 磁盘 I/O、模型加载、`KnowledgeBase()` 构造 |
| `_Entry.ready`（Event） | 构建完成的发布 | — | 持锁等待 |
| `KnowledgeBase._init_lock`（实例级，既有） | 首次索引构建 | 秒~分钟 | 不与其他锁嵌套 |

**锁序**：`_CACHE_LOCK` → 释放 → `_init_lock`。**禁止反向**（否则死锁）。

**(6) public 缓存独立**

- `_PUBLIC` 是单变量，不进 LRU，`_evict_locked` 不碰它；
- `refs` 同样计数（用于统计在用量），但**永远不因 refs 归零而回收**；
- 唯一失效途径：显式 `invalidate_public()`（由管理员 reload 触发，不在本轮 10 条路由内 → 由 `_invalidate_knowledge_caches()` 内部调用）；
- 失效时旧实例若 `refs>0` 仍可服务到请求结束，但新请求立即拿新实例。

**(7) `KnowledgeBase` 并发语义（明确写入编码规范）**

| 操作 | 并发安全 | 约束 |
|---|---|---|
| `search()` / `indexed_chunk_counts()` | ✅ 读并发安全 | 依赖「初始化后 `items` 不再变」 |
| `_ensure_index()` | ✅ 并发安全 | 既有 `_init_lock` + 双检查 |
| 任何改变索引的操作 | ❌ 不支持 | **实例不可变**：改索引一律「新建实例 + 替换缓存」，禁止 `VectorStore.add` 在运行期被调用 |

执行层面加一条守卫：`VectorStore.add` 增加 `if getattr(self, "_sealed", False): raise RuntimeError("vector store is sealed")`，`_ensure_index` 成功后置 `_sealed = True` → **运行期误加会立刻报错**，而不是悄悄脏读。

#### 3.6.4 验收标准

| 用例 | 并发/时序 | 断言 |
|---|---|---|
| T-06-A | 32 线程同时请求同一 scope 的 KB | `KnowledgeBase` 构造次数 == **1**（用计数器）；32 个请求全部成功；`refs` 最终归 0 |
| T-06-B | 8 个 scope 各 1 个慢请求占住（`Event` 卡住），第 9 个 scope 请求 | 第 9 个成功创建（软超限）；`kb_cache_overflow` == 1；在用 8 个**未被打断** |
| T-06-C | 16 个 scope 全占住，第 17 个请求 | 返回 503 `kb_capacity_exhausted`；已建 16 个实例全部 `state=READY` 且未 CLOSED |
| T-06-D | 占住 1 个 scope，其他 9 个空闲 | 触发淘汰：`len(_PRIVATE)` 回到 8；被淘汰的 9 个 `state=CLOSED`；**磁盘文件数不变** |
| T-06-E | public + 15 个私有占满 | public entry `refs>0`，`_evict_locked` 从不动它；`_PUBLIC is not None` 恒成立 |
| T-06-F | 构建时抛异常（mock `KnowledgeBase` 构造失败） | 所有等待者都收到 `KbUnavailable`；`KnowledgeBase()` 只被调用 1 次；60 s 后可重试 |
| T-06-G | 请求处理中 `invalidate_public()` | 在途请求正常返回；下一个请求拿到新实例；旧实例 `refs` 归 0 后不再被引用 |
| T-06-H | 运行期调用 `VectorStore.add` | 抛 `RuntimeError("vector store is sealed")` |
| T-06-I | `_CACHE_LOCK` 持有时长采样 | 1000 次 `_acquire` 的 p99 < 5 ms（用 `time.perf_counter` 埋点） |

---

### R3-07 模型预算预扣与结算状态机（P0）

#### 3.7.1 原设计（R2-07）

```sql
INSERT INTO usage_daily (user_id, day, calls, image_calls, total_tokens)
VALUES (:uid, :day, 1, :is_image, 0)
ON CONFLICT(user_id, day) DO UPDATE SET calls = calls + 1, image_calls = image_calls + :is_image
WHERE calls < :daily_limit;   -- changes()==0 → 429，不发请求
```

（R2 结论：reserve→settle/release 状态机；`changes()==1` 原子预扣；无 usage 不估算。）

#### 3.7.2 发现的问题

**(a) 这条 SQL 只是个「计数器自增」，没有「预扣凭证」。** 请求发出后进程崩了，`calls` 已经 +1，但没有任何记录告诉你「这 1 次是谁、什么时候、对应哪次模型调用」。所以：

- 无法**结算**（想把真实 token 写回到具体那次调用，没有主键）；
- 无法**释放**（想退还这 1 次额度，不知道退哪一条）；
- 无法**对账**（日汇总与事件明细无法互校）。

**(b) 未知 token 用量的处理没定。** 供应商不返回 `usage` 时（本项目全项目 `usage` 零命中，5 个调用点都没有取用量），如果把 `None` 当 0 写进 `total_tokens`，会**压低**已统计的已知用量；如果累加 `NULL`，`total_tokens` 会变 NULL 把之前的值吃掉。

**(c) 释放的边界没定。** 「Provider 报错就释放」太宽：超时、连接中断、`CancelledError` 这三种情况**无法证明供应商没计费**，释放等于白送额度。

**(d) 并发验证不足。** R2 只给了 SQL，没给「并发 50 个请求、限额 10，恰好 10 个成功」的实测依据。

#### 3.7.3 修订方案

**(1) 事件级预扣表（每次预扣一行，带唯一 ID）**

```sql
CREATE TABLE IF NOT EXISTS usage_reservations (
    reservation_id          TEXT PRIMARY KEY,          -- uuid4 hex（32 位）
    user_id                 TEXT NOT NULL,
    day                     TEXT NOT NULL,             -- UTC 'YYYY-MM-DD'
    kind                    TEXT NOT NULL CHECK (kind IN ('chat','image','embedding')),
    cost_calls              INTEGER NOT NULL DEFAULT 1 CHECK (cost_calls >= 0),
    cost_image              INTEGER NOT NULL DEFAULT 0 CHECK (cost_image >= 0),
    state                   TEXT NOT NULL CHECK (state IN ('reserved','settled','released','kept')),
    created_at              TEXT NOT NULL,
    expires_at              TEXT NOT NULL,             -- created_at + 10 min
    settled_at              TEXT,
    provider_reported_tokens INTEGER,                  -- NULL = 供应商未返回
    outcome                 TEXT                       -- 'ok' | 'provider_unbilled' | 'timeout_kept' | 'crash_kept'
) STRICT;

CREATE INDEX IF NOT EXISTS idx_ures_uid_day ON usage_reservations(user_id, day, state);
```

**(2) 状态机（四态，终态不可变）**

```
                 ┌──────────── settle(tokens|NULL) ──────────► settled
reserved ────────┼──────────── release(reason) ──────────────► released
                 └──────────── keep(timeout|crash|unknown) ───► kept
```

| 迁移 | 触发条件 | 是否退还额度 | `calls` 变化 |
|---|---|---|---|
| → `settled` | 模型正常返回 | 否 | 不变 |
| → `released` | **确认未计费**的 Provider 错误（HTTP 4xx 且非 429/5xx；连接建立前失败） | 是 | `calls -= cost_calls`、`image_calls -= cost_image`（**必须同时**） |
| → `kept` | 超时 / 连接中断 / `CancelledError` / 进程重启 / 结果不明 | 否（保守计费） | 不变 |

**只有这三种终态，没有第四种。**「不确定」一律归 `kept`，绝不归 `released`。

**(3) 预扣：单个写事务内完成「查限额 + 写事件 + 加计数」**

```python
def reserve(conn, *, user_id, day, kind, limits) -> str:
    rid = uuid.uuid4().hex
    cost_calls = 1
    cost_image = 1 if kind == "image" else 0
    conn.execute("BEGIN IMMEDIATE")            # 跨连接/跨进程互斥（busy_timeout 实测 5000ms）
    try:
        cur = conn.execute("""
            INSERT INTO usage_daily(user_id, day, calls, image_calls, total_tokens, unknown_usage_count)
            VALUES (:uid, :day, :c, :i, 0, 0)
            ON CONFLICT(user_id, day) DO UPDATE SET
                calls       = calls + :c,
                image_calls = image_calls + :i
            WHERE usage_daily.calls + :c <= :lim_calls
              AND usage_daily.image_calls + :i <= :lim_images
        """, {...})
        if cur.rowcount == 0:                  # 超限：整体回滚，不发请求
            conn.execute("ROLLBACK")
            raise QuotaExceeded(day, user_id)
        conn.execute("""
            INSERT INTO usage_reservations
              (reservation_id, user_id, day, kind, cost_calls, cost_image,
               state, created_at, expires_at)
            VALUES (:rid, :uid, :day, :kind, :c, :i, 'reserved', :now, :exp)
        """, {...})
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    return rid
```

**实测支撑**（`probe_sql.py`）：同样的 upsert+WHERE 模式连跑 3 次，`rowcount = [1, 1, 0]`，最终 `calls=2` —— **限额 2 时恰好成功 2 次**，第 3 次原子失败。这是「不会超卖」的直接证据。

注意 `WHERE` 里用 `usage_daily.calls` **必须带表名限定**，否则在 `ON CONFLICT` 子句里会与 `excluded` 命名解析混淆。

**(4) 结算（绑定 reservation_id，只允许一次）**

```python
def settle(conn, rid, tokens: int | None, outcome="ok") -> bool:
    conn.execute("BEGIN IMMEDIATE")
    try:
        cur = conn.execute("""
            UPDATE usage_reservations
            SET state='settled', settled_at=:now,
                provider_reported_tokens=:tokens, outcome=:outcome
            WHERE reservation_id=:rid AND state='reserved'
        """, {...})
        if cur.rowcount == 0:                  # 已结算/已释放 → 幂等返回，不报错
            conn.execute("ROLLBACK"); return False
        conn.execute("""
            UPDATE usage_daily SET
                total_tokens        = total_tokens + COALESCE(:tokens, 0),
                unknown_usage_count = unknown_usage_count + (CASE WHEN :tokens IS NULL THEN 1 ELSE 0 END)
            WHERE user_id=:uid AND day=:day
        """, {...})
        conn.execute("COMMIT"); return True
    except BaseException:
        conn.execute("ROLLBACK"); raise
```

**「未知 ≠ 零」的落实**：
- `provider_reported_tokens` 存 `NULL`，不存 0 → 明细层可区分「真的用了 0 token」与「没上报」；
- `total_tokens` 用 `COALESCE(:tokens, 0)` 相加 → **已统计的已知用量不会被覆盖，也不会被置 NULL**；
- 另设 `unknown_usage_count` 记「有几次没上报」 → 对账时可判断「汇总数偏低的原因」。

**(5) 释放（同步处理两类计数）**

```python
def release(conn, rid, reason: str) -> bool:
    conn.execute("BEGIN IMMEDIATE")
    try:
        cur = conn.execute("""
            UPDATE usage_reservations
            SET state='released', settled_at=:now, outcome=:reason
            WHERE reservation_id=:rid AND state='reserved' RETURNING cost_calls, cost_image, user_id, day
        """, {...})
        row = cur.fetchone()
        if row is None:
            conn.execute("ROLLBACK"); return False
        c, i, uid, day = row
        conn.execute("""
            UPDATE usage_daily
            SET calls = calls - :c, image_calls = image_calls - :i
            WHERE user_id=:uid AND day=:day
        """, {"c": c, "i": i, "uid": uid, "day": day})
        conn.execute("COMMIT"); return True
    except BaseException:
        conn.execute("ROLLBACK"); raise
```

`RETURNING` 实测可用（SQLite 3.49.1）→ 退还是按**该次预扣的真实成本**退，不是写死 1。

**(6) 只有「确认未计费」才释放（白名单）**

| 场景 | 释放？ | 依据 |
|---|---|---|
| HTTP 400/401/403/404/422（请求未被执行） | ✅ | 供应商明确拒绝，未产生 token |
| 连接建立失败（DNS/TCP/TLS 错误，未发出请求） | ✅ | 请求根本没到供应商 |
| HTTP 429（限流） | ❌ → `kept` | 供应商可能已计费；且 429 常伴随重试语义 |
| HTTP 5xx | ❌ → `kept` | 可能部分执行 |
| `asyncio.TimeoutError` / socket timeout | ❌ → `kept` | 无法证明未执行 |
| `ConnectionResetError`（已发出后断开） | ❌ → `kept` | 同上 |
| `asyncio.CancelledError`（客户端断开） | ❌ → `kept` | 供应商侧可能仍在生成 |
| 进程崩溃 / 重启 | ❌ → `kept` | 启动时批量收敛 |

保守计费的理由：额度是「防滥用」工具，不是计费系统。**多算一次的成本远低于被绕过的风险。**

**(7) 崩溃恢复：启动时把过期 reserved 收敛为 kept**

```sql
UPDATE usage_reservations
SET state='kept', settled_at=:now, outcome='crash_kept'
WHERE state='reserved' AND expires_at < :now;
-- 逐行写审计 actor_type='system', action='reservation_reaped'
```

**不退还额度**。同时输出 `reaped_count` 到启动日志。

**(8) 全局预算与并发原子性**

- 预扣在同一 `BEGIN IMMEDIATE` 内完成「限额判定 + 事件写入」→ 不存在「查完发现够、再写时已超」的窗口；
- 跨进程：`BEGIN IMMEDIATE` 拿的是 SQLite 的 RESERVED 锁，**跨进程有效**；
- `timeout` 必须显式设 `10.0`（默认 5 s 实测，多 Worker 抢锁可能不够）；
- 失败重试：`sqlite3.OperationalError: database is locked` → 最多重试 2 次（指数退避 50/150 ms），仍失败 → 503 `budget_busy`（**不降级为「跳过计量」**）。

**(9) 事件级用量 vs 日汇总的关系（可重建）**

```
usage_daily.total_tokens  ==  SUM(usage_reservations.provider_reported_tokens)
                               WHERE user_id=? AND day=? AND state IN ('settled','kept')
usage_daily.calls         ==  COUNT(*) WHERE state IN ('reserved','settled','kept')
                               （released 已扣回，不计）
```

提供 `reconcile(user_id, day)` 做双向校验，不一致则：
- 记 `logger.error` + 审计 `actor_type='system', action='budget_reconcile_mismatch'`；
- **不自动修正**（自动修正会掩盖 bug），只告警，由运维决定。

#### 3.7.4 实现约束

| 约束 | 理由 |
|---|---|
| `reservation_id` 必须 `uuid4().hex`（32 位无连字符） | 定长、可索引、不可枚举 |
| `day` 统一 **UTC** | 与旧表 `datetime.now()` 本地时间不同（V-06 技术债），新表只存 UTC，禁止跨表比较日期 |
| `expires_at` = `created_at + 10 min` | 大于最长模型调用超时（当前 300 s），留有收敛余量 |
| 预扣必须先于「发请求」，且 `rid` 传入调用点 | 否则崩溃窗口内无法归因 |
| 5 个模型调用点（`agent.py:877/1503/1629/1784/2866`）**全部**要包 reserve/settle | 现有代码 `usage` 零命中 → 需逐点接入，不允许只接 `run` 一处 |

#### 3.7.5 验收标准

| 用例 | 时序 | 断言 | DB 预期 |
|---|---|---|---|
| T-07-A | 50 并发请求，日限额 10 | 恰好 10 个拿到 `reservation_id`，40 个 429 | `usage_daily.calls == 10`；`reservations` 10 行 `reserved` |
| T-07-B | 同一 rid 连续 `settle` 两次 | 第一次 True、第二次 False；不抛错 | `total_tokens` 只加一次 |
| T-07-C | 同一 rid `settle` 后 `release` | `release` 返回 False | `calls` 不变（`released` 不能覆盖 `settled`） |
| T-07-D | `settle(tokens=None)` | `provider_reported_tokens IS NULL`；`unknown_usage_count += 1` | `total_tokens` 不变（**不写 0**） |
| T-07-E | 先 `settle(tokens=100)`，再对另一 rid `settle(tokens=None)` | 已知量保留 | `total_tokens == 100`，`unknown_usage_count == 1` |
| T-07-F | `kind='image'` 的预扣后 `release` | `calls` 与 `image_calls` **同时**减 1 | 两者都不为负（`CHECK >= 0` 兜底） |
| T-07-G | 模拟超时 → `keep('timeout_kept')` | 额度不退 | `calls` 不变；`state='kept'` |
| T-07-H | 写入 3 条过期 `reserved` → 重启 | 全部转 `kept`；审计 3 行 `actor_type='system'` | `usage_daily.calls` 不变 |
| T-07-I | 构造日汇总与明细不一致 | `reconcile` 报 mismatch；**不自动改数** | `usage_daily` 保持原值 |
| T-07-J | `BEGIN` 后另一进程抢锁（`timeout=10`） | 最多重试 2 次；仍失败 → 503 `budget_busy` | 无脏写 |

---

### R3-08 CSRF 与 Session 完整生命周期（P0）

#### 3.8.1 原设计（R2-08）

「双提交 + 会话绑定摘要 + 登录/注册严格 Origin + `Cache-Control: no-store`」。

#### 3.8.2 发现的问题

**(a) 缺匿名态。** 双提交的前提是「客户端已有一个 CSRF token」。但**登录页本身**的 POST 是从零状态发出的，此时还没有会话。R2 没定义匿名 token 从哪来。

**(b) 没定轮换。** 登录成功后若继续用登录前的 CSRF token，且该 token 在登录前已被攻击者知晓（例如通过子域写入），就构成 CSRF 绕过。必须换。

**(c) Origin / Referer / Sec-Fetch-Site 的优先级与缺失处理没定。** 三者可能同时存在、部分缺失、全部缺失。缺一个就「跳过校验」是最常见的错误实现。

**(d) `PUBLIC_ORIGIN` 从哪来没定。** 从 `Host` 头推导 = host header injection；写死在代码里 = 换域名要改代码。

**(e) 反向代理信任边界没定。** 若部署在 nginx 后，`request.url` 拿到的是内网地址还是公网地址，取决于是否信任 `X-Forwarded-*`。

**(f) 并发续期与撤销的竞态没定。** 同一用户 10 个请求同时续期，会写 10 次；登出瞬间有请求在途，该放行还是拒绝。

**(g) 24h 空闲 vs 7d 绝对的交互规则没给公式。**

#### 3.8.3 修订方案

**(1) 完整生命周期（8 态）**

```
① 匿名访问
     GET /login → 响应设置 pai_csrf（匿名 token，签名 cookie，无 Max-Age）
     → 状态 ANON_WITH_CSRF

② 登录提交
     POST /auth/login  body: {username, password}
     header: X-CSRF-Token == cookie pai_csrf   （双提交）
     Origin 严格校验（见 (3)）
     → 校验通过 → 状态 LOGIN_PENDING

③ Session 轮换
     生成 session_id = secrets.token_urlsafe(32)；DB 只存 sha256(session_id)
     Set-Cookie: __Host-pai_sess=<sid>; Path=/; Secure; HttpOnly; SameSite=Lax
     同时轮换 CSRF：pai_csrf = HMAC(secret, sid || nonce)  ← 与旧值必然不同
     → 状态 AUTHENTICATED

④ 已登录写请求
     校验 cookie 会话有效 + X-CSRF-Token == cookie pai_csrf + 摘要与 sid 绑定
     → 状态 AUTHENTICATED（不变）

⑤ Session 续期
     每请求若 last_seen_at 距今 > 60 s → 更新 last_seen_at（节流）
     expires_at = min(last_seen_at + 24h, created_at + 7d)
     → 状态 RENEWED（数据态同 AUTHENTICATED）

⑥ 登出
     POST /auth/logout：DB 置 revoked_at，清两个 cookie
     → 状态 ANON（无 CSRF，需重新 GET 页面获取）

⑦ 撤销（管理员）
     UPDATE sessions SET revoked_at=... WHERE user_id=?
     → 目标用户下次请求 401

⑧ 过期
     空闲 > 24h 或 绝对 > 7d → 校验失败 401
```

**(2) 匿名 CSRF token 的生成、存储、失效**

| 环节 | 设计 |
|---|---|
| 生成 | `GET /login`、`GET /register`、`GET /`（任意 HTML 页）响应时，若无 `pai_csrf` cookie → 生成 `nonce = secrets.token_urlsafe(24)` |
| 存储 | cookie 值 = `nonce + "." + HMAC-SHA256(PAI_SECRET_KEY, "anon" + nonce)[:16]`；**服务端不存库**（无状态） |
| 读取 | 页面 `<meta name="csrf-token">` 内联同一值；JS 也读 cookie（`pai_csrf` **非 HttpOnly**） |
| 校验 | `HMAC` 重算一致 + header 值 == cookie 值 |
| 失效 | 登录成功时**删除**该 cookie 并下发新值（绑定 sid）；浏览器关闭即失效（无 `Max-Age`）；服务端 `PAI_SECRET_KEY` 轮换后全部失效 |

为什么不用「服务端存 token 表」：匿名态每天可能数万次访问，存表会引入写放大与清理任务，而**签名自校验**在无状态下等价安全（攻击者没有密钥就无法伪造合法签名）。

**(3) Origin / Referer / Sec-Fetch-Site 的优先级与缺失处理**

```
写请求（POST/PATCH/DELETE）判定顺序：
1) 有 Origin 头  → 必须 ∈ 白名单（精确匹配 scheme+host+port），否则 403
2) 无 Origin，有 Referer → 解析出 origin，必须 ∈ 白名单，否则 403
3) 两者都无，有 Sec-Fetch-Site → 必须 == "same-origin"，否则 403
4) 三者全无 → **403**（默认拒绝，不是默认放行）
```

- 「严格」档（登录、注册、登出、全部 `/admin/*` 写）：走完 1–4，**任何一步不满足即 403**；
- 「宽松」档（`/chat`、`/knowledge/*`、`/analyze-image`）：只做 CSRF 双提交，**跳过 Origin**（理由：这些接口的 CSRF 风险由 token 覆盖；加 Origin 会破坏非浏览器客户端的既有契约 —— 现有测试用 `TestClient` 直连，无 Origin）；
- GET / HEAD / OPTIONS 不校验 Origin。

> 这里刻意**不统一**：把 Origin 加到 `/chat` 会改变现有 33 条测试的行为（`TestClient` 默认不发 Origin）。设计选择是「新增的敏感写接口更严，既有业务接口保持契约」。

**(4) `PUBLIC_ORIGIN` 的可信配置来源**

| 项 | 规则 |
|---|---|
| 变量名 | `PAI_PUBLIC_ORIGIN` |
| 格式 | 必须是 `https://<host>[:port]`，**不得含路径/查询**；本地开发允许 `http://127.0.0.1:8000` |
| 校验时机 | 应用启动时（`lifespan` 第一句）；不合法 → **拒绝启动**（退出码 2），不降级为「跳过校验」 |
| 多值 | 逗号分隔（用于 `www` 与非 `www`），最多 3 个 |
| 绝不使用 | `request.headers["host"]`、`X-Forwarded-Host`、`request.base_url` |

**(5) 反向代理与 Host 头的信任边界**

- 应用**不读** `Host` 做任何安全判定 → 免疫 host header injection；
- 默认**不启用** `ProxyHeadersMiddleware`；
- 若部署在 nginx 后：应用 bind `127.0.0.1:8000`，由 nginx 终结 TLS；此时仍不需要信任 `X-Forwarded-*`，因为安全判定只用 `PAI_PUBLIC_ORIGIN`；
- 若必须知道客户端 IP（限速用）：仅当 `PAI_TRUST_PROXY=1` 且 `PAI_TRUSTED_PROXY_IPS=127.0.0.1` 时读取 `X-Forwarded-For` **最右侧**一项，并且只用于限速桶，**不用于鉴权与审计主体**。

**(6) Cookie / CORS / 缓存**

| 项 | 会话 cookie | CSRF cookie |
|---|---|---|
| 名称 | `__Host-pai_sess`（生产）/ `pai_sess`（开发） | `pai_csrf` |
| `Secure` | 是 | 是（生产）/ 否（开发） |
| `HttpOnly` | **是** | **否**（JS 必须可读） |
| `SameSite` | `Lax` | `Lax` |
| `Path` | `/` | `/` |
| `Domain` | **不设置**（`__Host-` 强制） | 不设置 |
| `Max-Age` | 24h（滑动，随续期重发） | 无（会话级） |

> `__Host-` 前缀是浏览器强制的：必须 `Secure` + `Path=/` + 无 `Domain`。因此**开发环境（http）不能用该前缀**，由 `PAI_COOKIE_SECURE=0` 自动切换为 `pai_sess`，并在启动日志打 `WARN: cookie_secure_disabled`。生产环境若检测到 `PAI_COOKIE_SECURE=0`，**拒绝启动**。

CORS：**不启用** `CORSMiddleware`（同源部署）。若将来需要，必须显式白名单 + `allow_credentials=True` 时禁止 `allow_origins=["*"]`。

缓存：所有带身份的响应 `Cache-Control: no-store, private`；静态资源（13–22 号路由）`Cache-Control: public, max-age=3600` 且**不带任何 cookie**。

**(7) 并发续期、登出、撤销的竞态**

| 场景 | 处理 | 语义 |
|---|---|---|
| 10 个并发请求同时续期 | `UPDATE sessions SET last_seen_at=:now WHERE session_id=? AND revoked_at IS NULL AND last_seen_at < :now_minus_60s` | 最多写 1 次（其余 `rowcount=0`）；不报错 |
| 登出瞬间有在途请求 | 登出只改 DB，不等待在途 | **在途请求允许完成**，下一个请求 401 |
| 管理员撤销 + 用户正在请求 | 同上 | 最终一致，不做强同步 |
| 续期与登出并发 | 两个 `UPDATE` 都带 `revoked_at IS NULL`（续期）/`revoked_at IS NULL`（登出幂等） | 登出优先：一旦 `revoked_at` 非空，续期 `rowcount=0` |

明确声明：**不做「撤销即中断在途请求」**。原因：单 Worker + 同步处理，中断在途请求需要追踪所有活动任务，成本远大于收益；且撤销的目的通常是「阻止后续访问」。

**(8) 24h 空闲与 7d 绝对期限的交互公式**

```
created_at          会话创建时刻（不随续期变化）
last_seen_at        最近一次有效请求时刻
expires_at = min(last_seen_at + 24h, created_at + 7d)
```

- 每次续期重算 `expires_at`；
- **绝对期限不因续期延长**：`created_at + 7d` 是硬顶；
- 校验逻辑：`now < expires_at AND revoked_at IS NULL` → 有效；
- 边界：`now == expires_at` 视为**过期**（`<` 而非 `<=`）；
- 剩余不足 24h 但已接近 7d 时，`expires_at` 停在 7d 顶，并在 `GET /auth/me` 返回 `expires_at` 让前端提示重新登录。

#### 3.8.4 验收标准

| 用例 | 场景 | 断言 |
|---|---|---|
| T-08-A | 匿名访问 `/login` | 响应含 `Set-Cookie: pai_csrf=...`；无 `pai_sess`；cookie 值通过 HMAC 自校验 |
| T-08-B | 匿名 POST `/auth/login`，不带 `X-CSRF-Token` | 403；`sessions` 表 0 行 |
| T-08-C | 匿名 POST `/auth/login`，token 正确、无 Origin/Referer/Sec-Fetch-Site | 403（三者全缺 → 拒绝） |
| T-08-D | 同上，Origin 为外域 | 403 |
| T-08-E | 正确登录 | 200；`Set-Cookie: __Host-pai_sess`；**新 `pai_csrf` ≠ 登录前的值**；旧匿名 token 失效 |
| T-08-F | 已登录写请求：token 与 cookie 不一致 / 缺失 / 用登录前的旧 token / 用他人 session 的 token（4 种） | 全部 403；业务未执行 |
| T-08-G | 已登录并发 20 个 GET | `sessions.last_seen_at` 最多被更新 1 次（60 s 节流） |
| T-08-H | 登出 | `revoked_at` 非空；两个 cookie 被清；后续请求 401 |
| T-08-I | 构造 `last_seen_at=now-25h` | 请求 401（空闲过期） |
| T-08-J | 构造 `created_at=now-8d` 但 `last_seen_at=now-1min` | 请求 401（绝对过期），**且续期不能救** |
| T-08-K | `expires_at == now`（精确边界） | 401 |
| T-08-L | 生产配置下 `PAI_PUBLIC_ORIGIN` 缺失或非法 | 启动失败，退出码 2 |
| T-08-M | 带身份的响应 | `Cache-Control` 含 `no-store`；静态资源不含 |

---

### R3-09 SQLite 备份与恢复一致性（P0）

#### 3.9.1 原设计（R2-09）

「6 道门禁（G1–G6），任一失败中止；恢复 6 步」；完整性校验用「三张表行数相同」。

#### 3.9.2 发现的问题

**(a) `checkpoint` 的返回值没处理，而它有三种形态。** 实测：

| 库状态 | `PRAGMA wal_checkpoint(TRUNCATE)` 返回 |
|---|---|
| WAL 库、无竞争者 | `(0, 0, 0)` → busy=0，log=0，checkpointed=0 |
| WAL 库、**有读者持共享锁** | `(1, 1, 1)` → **busy=1**，WAL 仍有 1 帧未合并 |
| **非 WAL 库**（本项目现状 `journal_mode=delete`） | `(0, -1, -1)` → busy=0 但 log/checkpointed 是 **-1（无意义）** |

第三行是陷阱：一个只判 `busy == 0` 的实现会在**非 WAL 库上永远通过**，而它其实什么都没检查。本项目两库当前都是 `delete` 模式（实测）→ 这个Bug 会 100% 发生。

**(b) 「三张表行数相同」不能证明备份完整。** 实测：备份期间有一个线程持续写入，`backup()` 成功返回，备份里 25 行、源库当时已 46 行 → **行数天然不同**。原因：`Connection.backup()` 产出的是**某一时刻的一致快照**，不是最终状态。若实现是「先备份、再比源库当前行数」，必然失败；若是「先记源库行数、再备份」，两次计数之间源库可能已变，也必然失败。两种写法都错。

更要紧的是：**行数相同 ≠ 内容相同**。删一行 + 加一行，行数不变。

**(c) WAL/SHM 的删除时机没定。** 实测（Windows）：连接打开时 `os.remove("...-wal")` → `PermissionError: 另一个程序正在使用此文件`；`-shm` 同样。连接关闭后，SQLite 自己就把两个文件删了（`文件不存在`）。所以「恢复前删 WAL/SHM」如果还有连接开着，**必然报错**；而如果硬删成功，说明有连接被强制断开，可能丢数据。

**(d) 备份产物会改变目标库的 `journal_mode`。** 实测：源库 `wal`，备份出的文件 `journal_mode` 也是 `wal`。把它 `os.replace` 到原本是 `delete` 模式的 `memory.db` 上 → **目标库变成 WAL 模式**，后续 `conftest.py` 的指纹哨兵要处理 `-wal/-shm` 文件（V-05）。

**(e) `PRAGMA foreign_keys` 不持久化。** 实测：新连接读 `PRAGMA foreign_keys` = **0**，尽管 `business/db.py:44` 在 connect 时设了 ON。恢复后校验必须**显式开启**再跑 `foreign_key_check`，否则校验形同虚设。

**(f) `foreign_key_check` 的返回值形态。** 实测：`[('c', 1, 'p', 0)]` = (子表, rowid, 父表, fk 序号)。断言应该是「结果为空列表」，不是「结果为 0」。

#### 3.9.3 修订方案

**(1) 八道门禁（G0–G7，任一失败即中止，不做部分恢复）**

| 门禁 | 检查 | 失败动作 | 实测依据 |
|---|---|---|---|
| **G0** | 磁盘剩余空间 ≥ 3 × 源库大小 | 中止，报 `insufficient_disk` | `page_count × page_size` 可精确算出库大小 |
| **G1** | `journal_mode` 判定 + `checkpoint` 三元组语义校验 | 中止，报 `checkpoint_incomplete(busy=1)` | 见下 |
| **G2** | 备份期间 `data_version` 未变（或变化后重试 ≤3 次仍变 → 中止） | 中止，报 `concurrent_writes` | 实测 `data_version` 2→3 可观测到他人提交 |
| **G3** | 备份文件 `PRAGMA integrity_check` == `'ok'` **且** `quick_check` == `'ok'` | 删除坏备份，中止 | 实测返回 `'ok'` 字符串 |
| **G4** | 关键表 **PK 集合哈希 + 内容哈希** 与源快照一致 | 中止，报 `content_mismatch` | 见 (3) |
| **G5** | 备份库 `PRAGMA foreign_key_check` 为空列表 **（必须先 `PRAGMA foreign_keys=ON`）** | 中止 | 实测默认 OFF |
| **G6** | 恢复前所有连接已关闭（用连接登记表 + `os.remove` 试探） | 中止，报 `connection_leak` | 实测 `PermissionError` |
| **G7** | 备份库 `user_version` == 迁移台账当前版本 | 中止，报 `schema_version_mismatch` | 两库 `user_version` 现为 0 |

**G1 的正确实现（三种返回值全覆盖）**：

```python
def gate_checkpoint(conn) -> None:
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0].lower()
    busy, log, ckpt = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    if mode == "wal":
        if busy != 0:
            raise BackupAbort(f"checkpoint_incomplete busy={busy} log={log} ckpt={ckpt}")
        if log == -1 or ckpt == -1:
            raise BackupAbort("checkpoint_undefined_on_wal")   # 理论不应出现
    else:
        # 非 WAL：checkpoint 无意义，返回 (0,-1,-1)；此时改判「无 -wal 文件残留」
        if log != -1:
            raise BackupAbort(f"unexpected_checkpoint_shape log={log}")
        if pathlib.Path(db_path + "-wal").exists():
            raise BackupAbort("stale_wal_file_on_non_wal_db")
```

关键点：**先看 `journal_mode` 再解释三元组**，而不是无条件信任 `busy == 0`。

> **跨平台修正（与 §3.10.3 (5) 联动）**：G6 的「`os.remove` 试探」只在 Windows 有效。Linux 允许删除正在被打开的文件，试探**恒成功**，等于没检查。因此 G6 的主判据必须是**应用层连接登记表**，`PermissionError` 仅作为 Windows 上的额外佐证。此修正同时影响 T-09-H 的实现方式。

**(2) 备份期间是否允许业务写入**

| 模式 | 允许写入 | 一致性保证 | 使用场景 |
|---|---|---|---|
| `live`（默认） | ✅ 允许 | 备份是**某个时点的快照**；台账记录 `data_version_before/after`、`rows_at_snapshot` | 日常备份 |
| `strict` | ❌ 维护窗口内停写 | G2 必须通过（`data_version` 不变） | 迁移前备份、恢复演练 |

**默认 `live`**：因为 `Connection.backup()` 本身保证快照一致（实测备份文件 `integrity_check = ok`），不需要停服。停服只用于「要拿到精确的最终状态」的场景。

**(3) 一致快照边界与「不能只看行数」**

- 边界：`src.backup(dst)` 单次调用内部对源库持读事务，产出**某一时刻的一致快照**；备份完成后源库的继续写入**不会**进入备份。实测：备份 25 行 vs 源最终 46 行，两者都 `integrity_check=ok`。
- **因此校验必须「备份 vs 快照内的自身一致性」**，不能「备份 vs 源库当前」。
- 替代行数的校验（G4）：

```sql
-- 每张关键表三个量一起比
SELECT COUNT(*)                                   AS n,
       COALESCE(SUM(id), 0)                       AS id_sum,
       COALESCE(SUM(LENGTH(content)), 0)          AS len_sum
FROM messages;

SELECT COUNT(*), COALESCE(SUM(id),0) FROM conversations;
SELECT COUNT(*) FROM memories;

-- 再加一层：主键集合的确定性哈希（用 Python 侧算）
SELECT id FROM messages ORDER BY id;   -- 取出后 hashlib.sha256(",".join(map(str,ids)))
```

断言：`(n, id_sum, len_sum, pk_sha256)` **四元组**逐项相等。删一行加一行时 `n` 相同但 `pk_sha256` 必变。

**(4) 恢复流程（8 步，每步可中止且可回退）**

```
R0  预检：目标库无活动连接（G6）；备份文件通过 G3/G4/G5/G7
R1  把当前目标库改名为 memory.db.pre_restore_<ts>（不删）
R2  删除目标库残留的 -wal / -shm（此时无连接，实测可删）
R3  shutil.copy2(backup, target + ".restoring")  → 同卷临时文件
R4  os.replace(target + ".restoring", target)     → 同卷原子替换
R5  打开目标库：PRAGMA foreign_keys=ON → PRAGMA foreign_key_check == []
R6  PRAGMA integrity_check == 'ok'；user_version 匹配台账；关键表四元组与备份一致
R7  全部通过 → 删除 .pre_restore_*；任一失败 → os.replace 回滚到 .pre_restore_*
```

要点：
- **R4 用 `os.replace`（同卷原子）**，不用 `shutil.move`（跨卷会退化为 copy+delete，非原子）；
- **R1 的改名是保命步骤**：任何后续失败都能回滚；
- R2 必须在 R1 之后、R3 之前 —— 此时无连接，删除安全（实测连接关闭后文件甚至自动消失）。

**(5) 恢复后的验证清单**

| 项 | 断言 |
|---|---|
| 外键 | `PRAGMA foreign_keys=ON` 后 `foreign_key_check` == **空列表** |
| 完整性 | `PRAGMA integrity_check` == `'ok'` |
| 迁移状态 | `schema_versions.status == 'succeeded'` 且 `MAX(version) == CODE_TARGET` |
| 业务数据 | `business.db`：`orders` 行数 ≥ 1（现状 1）；`memory.db`：`messages` 行数 == 备份四元组的 n |
| 归属一致性 | 隔离批次未残留 `owner_state='quarantined'`（除非备份时就处于隔离态） |
| 模式 | 记录恢复后的 `journal_mode`；若与恢复前不同，写日志告警（实测备份会带 `wal` 模式过来） |

#### 3.9.4 验收标准

| 用例 | 场景 | 断言 |
|---|---|---|
| T-09-A | 备份文件尾部截断 1 KB | G3 失败（`integrity_check` 非 ok）；**不进入 R1**，目标库零改动 |
| T-09-B | 备份期间持续写入（`live` 模式） | 备份成功；G2 记录 `data_version` 变化；G4 四元组在**备份自身**上自洽；日志含 `concurrent_writes_detected` |
| T-09-C | 有读者持读事务时 checkpoint | G1 抛 `checkpoint_incomplete busy=1`；不产生备份文件 |
| T-09-D | 非 WAL 库（现状）跑备份 | G1 走非 WAL 分支通过（`log == -1` 且无 `-wal` 文件）；备份成功；**不误判为通过 WAL checkpoint** |
| T-09-E | 备份后删 1 行再加 1 行，使 `n` 不变 | G4 因 `pk_sha256` 不等而失败 |
| T-09-F | R4 之后模拟 R6 失败 | `os.replace` 回滚；目标库字节级等于 `.pre_restore_*`（SHA-256 比对） |
| T-09-G | 恢复后 | `foreign_key_check` 空列表；`integrity_check='ok'`；`user_version` 匹配；`orders ≥ 1` |
| T-09-H | 恢复时仍有连接（人为留一个） | G6 中止；目标库未改名 |

---

### R3-10 生产依赖与平台锁定（P1）

#### 3.10.1 原设计（R2-10）

「生产 9 项（含隐式 multipart）+ 新增 argon2-cffi；测试 5 项；lock 精确锁定」。

#### 3.10.2 发现的问题

**(a) Windows 与 Linux 的依赖集合不可能相同。** `sentence-transformers 6.1.0` 依赖 `torch`，而 `torch` 的 wheel **按平台+后端（CPU/CUDA）区分**。在 Windows 上 `pip install` 拿到的是 `torch-*-win_amd64.whl`，在 Linux 上是 `manylinux*_x86_64.whl`。一份 `pip freeze` 结果**无法跨平台复用**。

**(b) `Pillow` 与 `python-multipart` 装了但没声明。** 【实测】`pillow 13.0.0`、`python-multipart 0.0.32` 已安装，但 `requirements.txt` 里没有（当前文件只有 `pypdf==6.16.2`、`python-docx==1.2.0`）。干净环境安装后会 `ModuleNotFoundError`（`PIL` 用于图片处理，`multipart` 用于 `UploadFile` 解析）。

**(c) `tzdata` 未安装，而代码写本地时间。** 【代码】`memory.py:239/269/502/925`、`tools.py:11` 用 `datetime.now()`（naive，无时区）。Windows 开发机 TZ=Asia/Shanghai，Linux 容器默认 TZ=UTC → **同一份代码写入的时间戳相差 8 小时**。这不是依赖问题，是**行为差异**，但必须在依赖轮一并解决（装 `tzdata` 或显式 `TZ=`）。

**(d) 跨平台语义差异没有被识别：`os.remove` 的失败行为不同。** 实测 Windows 下删除被占用文件 → `PermissionError`。**Linux 允许删除正在被打开的文件**（inode 延迟回收）→ R3-09 的 G6「用 `os.remove` 试探连接是否关闭」在 Linux 上**永远成功**，等于没检查。这是必须在设计阶段修掉的跨平台陷阱（见 (5)）。

**(e) 89 个已安装包里有环境残留。** `edge-tts`、`pywebview`、`zai-sdk` 源码零 import，`PyJWT 2.15.0` 零引用，`cryptography`/`bcrypt` 未装。若不锁定而直接在生产上 `pip install -r requirements.txt`，会把这些残留带进镜像（体积 + 攻击面）。

#### 3.10.3 修订方案

**(1) 双环境矩阵**

| 维度 | 开发（Windows，实测） | 生产（Linux，待核验） | 处理 |
|---|---|---|---|
| Python | 3.12.10 `MSC v.1943 64bit` | 3.12.x（**必须同 minor**） | `requires-python = ">=3.12,<3.13"`；镜像固定 `3.12.10` |
| 架构 | `win_amd64` | `x86_64` **或 arm64（V-09）** | 锁文件按架构分 |
| ABI | `cp312` | `cp312` | wheel tag `cp312-cp312-manylinux_2_17_x86_64` |
| torch | CPU wheel（`win_amd64`） | CPU wheel（`manylinux`） | 显式 `--index-url https://download.pytorch.org/whl/cpu`，**禁止**默认源（会拉 2.5 GB CUDA 依赖） |
| 时区 | Asia/Shanghai | 容器默认 UTC | **必须**装 `tzdata` 并 `ENV TZ=Asia/Shanghai`，或统一改 UTC（V-06） |
| 文件系统 | NTFS（大小写不敏感） | ext4（敏感） | 禁止依赖大小写；`document_id` 已是 `sha256(filename)`，需校验大小写敏感性 |
| 重解析点 | junction / symlink | 仅 symlink | `knowledge_store.inspect_knowledge_directory` 的 `lstat` 判定在 Linux 上语义等价 |
| 文件占用 | 打开时不可删（`PermissionError`） | **可删**（inode 延迟回收） | **G6 不能靠 OS 报错**（见 (5)） |
| 信号 | 无 SIGTERM 语义差异 | SIGTERM 优雅关闭 | `lifespan` 需处理 |

**(2) 直接依赖与精确 pin（生产）**

```text
# requirements.in —— 单一事实来源，只写直接依赖与语义范围
fastapi==0.141.1
uvicorn==0.52.4
pydantic==2.13.5
python-dotenv==1.2.3
openai==3.19.2
sentence-transformers==6.1.0
pypdf==6.16.2
python-docx==1.2.0
Pillow==13.0.0                 # ★ 新增（原先未声明）
python-multipart==0.0.32       # ★ 新增（原先未声明）
argon2-cffi==25.1.0            # ★ 新增（未安装，需 V-02 验证 wheel）
tzdata==2026.1                 # ★ 新增（Linux 时区）
```

测试依赖（开发机，不进生产镜像）：

```text
pytest==9.1.1
httpx==0.28.1
starlette==1.6.0
numpy==2.5.4
```

**不进入生产清单**（环境残留，源码零 import）：`edge-tts`、`pywebview`、`zai-sdk`、`PyJWT`、`cffi`（仅 argon2-cffi 运行时需要，由传递依赖引入，不直接声明）。

**(3) argon2-cffi 的目标平台 wheel 验证方法（本轮禁装，只给命令）**

```bash
pip download argon2-cffi==25.1.0 \
  --only-binary=:all: \
  --platform manylinux_2_17_x86_64 \
  --python-version 312 \
  --implementation cp \
  --abi cp312 \
  -d /tmp/wheelcheck
```

成功判据：目录中出现 `argon2_cffi-25.1.0-cp39-abi3-manylinux_2_17_x86_64.whl`（或 `manylinux2014`）；同时确认传递依赖 `argon2-cffi-bindings` 也有对应 wheel。**若只有 sdist → 需要 gcc + libffi-dev，镜像必须装编译工具链**（当前未知 → V-02）。

**(4) 锁定策略（直接依赖 vs 传递依赖）**

| 文件 | 内容 | 生成方式 | 用途 |
|---|---|---|---|
| `requirements.in` | 直接依赖 + 精确版本 | 手写 | 唯一事实来源 |
| `requirements-linux.lock` | 全量（含传递依赖）+ **哈希** | `pip-compile --generate-hashes`（Linux 上跑） | 生产镜像安装 |
| `requirements-win.lock` | 同上，Windows | 同上（Windows 上跑） | 开发环境复现 |

- **不共用一个 lock**：torch 的 wheel 名不同，共用会在另一平台安装失败；
- 安装命令：`pip install --require-hashes --no-deps -r requirements-linux.lock`（`--no-deps` 因为 hash 已覆盖全部）；
- CI 校验：`pip-compile --generate-hashes` 重跑一次，diff 为空才算通过 → 防止「手改了 lock 但没改 .in」。

**(5) 跨平台修正：G6 不能依赖 `PermissionError`**

R3-09 的 G6 原写法是「`os.remove` 成功则说明无连接」，这在 Linux 上**恒成立**（可以删除打开的文件）。修正为：

```python
# 主判据：应用层连接登记（跨平台可靠）
assert registry.open_count() == 0, "connection_leak"
# 次判据（仅 Windows 有效，作为额外佐证，不作为唯一依据）
if os.name == "nt":
    try:
        os.remove(db_path + "-wal")
    except FileNotFoundError:
        pass
    except PermissionError:
        raise BackupAbort("connection_leak(windows)")
```

即：**Windows 的 `PermissionError` 是「有连接」的充分证据；Linux 上必须靠登记表**。

**(6) `uvicorn` 版本约束**

- `uvicorn==0.52.4`（实测）；
- Windows 上 `uvloop` 不可用 → 不得在 `.in` 里加 `uvloop`；
- 若生产要用 `[standard]`（含 `httptools`/`uvloop`/`watchfiles`），必须**单独**给 Linux lock，且 `uvloop` 只在 `sys_platform == "linux"` 条件下声明；
- 单 Worker（`--workers 1`），与 R3-03 的多 Worker 互斥设计一致；若将来加 Worker，需重新评估 R3-06 的进程内缓存（V-07）。

#### 3.10.4 验收标准

| 用例 | 场景 | 断言 |
|---|---|---|
| T-10-A | 干净 Linux venv（无任何预装）`pip install --require-hashes -r requirements-linux.lock` | 安装成功；`python -c "import fastapi, PIL, multipart, argon2"` 全通过；`pip check` 无冲突 |
| T-10-B | 同上，断网 + 本地 wheel 目录 | 安装成功（证明无隐式网络依赖） |
| T-10-C | 重跑 `pip-compile` | 与仓库内 lock **逐字节相同** |
| T-10-D | 生产镜像内 `python -c "import edge_tts"` | `ModuleNotFoundError`（证明无残留） |
| T-10-E | 生产镜像 `TZ=Asia/Shanghai` + 写入一条消息 | `created_at` 与宿主机本地时间差 < 2 s（证明时区生效） |

---

### R3-11 测试矩阵补全（P0）

#### 3.11.1 原设计（R2-11）

11 组用例（T-01~T-10 + T-R），通过阈值一律写「全绿」。

#### 3.11.2 发现的问题

**(a) 「全绿」不是设计。** 它没有说「测什么数据、什么时序、跑完数据库应该长什么样」。一个只会断言 `status_code == 200` 的测试也能全绿。

**(b) 缺故障注入的时序描述。** R3 的多数缺陷只在「并发/崩溃/误用」路径出现。不写清时序（谁先谁后、在哪个点注入），测试就无法复现，也就无法回归。

**(c) 缺「失败判定」。** 没有写「什么现象算测试失败」，等于把判定权交给实现者 —— 实现者可以改断言来迁就实现。

**(d) 缺数据库预期状态。** 断言 HTTP 200 但不能证明数据没被写坏。R3-02/R3-03/R3-07 的核心风险全在数据层。

#### 3.11.3 修订方案

**八字段强制**（缺一不可，评审时逐格检查）：

| 字段 | 要求 |
|---|---|
| 测试编号 | `T-<项>-<字母>`，与 §3 一一对应，不新增不明来源的编号 |
| 前置条件 | 数据库初始状态、登录态、配置项、是否启用 mock |
| 测试数据 | 具体行数/字段值，不用「一些数据」 |
| 执行步骤 | 按顺序列出，含断言插入点 |
| 并发/故障注入时序 | 明确「谁在第几步做什么」，用 `Event`/`os._exit`/触发器，不用 `sleep` |
| 关键断言 | 可判定的布尔表达式，不用「表现正常」 |
| 数据库预期状态 | 具体表的具体行数/字段值 |
| 失败判定 | 「若出现 X 则失败」，含反例（改坏实现后测试必须红） |

完整的 80 个用例见 **§5 测试验收矩阵**。此处只声明三条方法论约束：

1. **反例必测**：每个 P0 项至少有一个「把实现改坏 → 测试变红」的验证（§6 列出 11 个反例点）。
2. **禁 `sleep` 同步**：并发用 `threading.Event` / `asyncio.Event`；崩溃注入用子进程 + `os._exit`；部分失败用 SQLite 触发器 `RAISE(ABORT)`。
3. **禁断言实现细节**：断言观测点（HTTP 状态、DB 行、日志字段），不断言内部函数被调用几次（除 T-06-A 的构造次数，那是设计约束本身）。

---

## 4. 状态机与数据流汇总

本轮设计的 6 个状态机，全部要求「终态不可变 + 迁移条件写在 SQL 的 WHERE 里 + `rowcount` 判定」。

| # | 状态机 | 状态集 | 迁移触发 | 终态可逆？ | 载体 |
|---|---|---|---|---|---|
| M1 | 迁移版本 | `pending → running → succeeded / failed` | 启动 / CLI | `failed` 可 resume，`succeeded` 不可逆 | `schema_versions` |
| M2 | 迁移步骤 | `pending → running → succeeded / failed / skipped` | 执行器 | 同上（靠幂等探针） | `schema_steps` |
| M3 | 隔离标记 | `normal → quarantined → normal` | 迁移 / 人工回填 | **可逆**（四条件 UPDATE） | `messages.owner_state` + `migration_quarantine` |
| M4 | 预算预扣 | `reserved → settled / released / kept` | 模型调用结果 | **不可逆**（终态唯一） | `usage_reservations` |
| M5 | 会话 | `anon → login_pending → authenticated → renewed → revoked / expired` | 请求 / 登出 / 超时 | `revoked` 不可逆 | `sessions` + cookie |
| M6 | KB 缓存 entry | `building → ready → failed / closed` | 首次访问 / 淘汰 / 失效 | `closed` 不可逆，`failed` 60 s 后可重建 | 进程内 `_Entry` |

**共同不变量**

| 不变量 | 保障手段 |
|---|---|
| 终态只能到一次 | `UPDATE ... WHERE state='<前置态>'` + `rowcount==0` 即忽略 |
| 崩溃后状态可解释 | 每个非终态都有 `owner` + 时间戳，可判定 stale |
| 数据与台账同事务 | M1/M2 用独立连接（ledger），M3/M4 与业务同事务 |
| 无跨库事务 | `business.db` 与 `memory.db` 分开，business 侧审计降级（§3.4.3 (4)） |

**一次带身份的 `/chat` 请求的完整数据流**

```
HTTP POST /chat
  → [中间件] Origin 宽松档（无）→ CSRF 双提交校验（403 或继续）
  → [Depends] 解析 __Host-pai_sess → sha256 → 查 sessions
              ├ 无效/过期/撤销 → 401
              └ 有效 → RuntimeContext(user_id, role, request_id, scopes)
                        sessions.last_seen_at 节流续期（60s）
  → [端点] with runtime_context(ctx):
        normalize_conversation_id
        → 校验 conversation 归属（他人 → 404）
        → reserve()  ← BEGIN IMMEDIATE：限额判定 + 写 reservations + calls+1
              └ 超限 → 429（不发请求）
        → Agent.run(...)  ← 5 个模型调用点均带 reservation_id
        → settle(rid, tokens|None)
              └ 异常分支：白名单内 → release()；其余 → keep()
        → save_message（同事务的审计 INSERT）
  → [响应] Cache-Control: no-store
```

---

## 5. 测试验收矩阵（80 个用例，八字段）

> 缩写：`R` = `RuntimeContext`；`DB` 列写「预期行数/字段值」；「反例」列说明**把实现改坏后本用例必须变红的方式**。所有用例默认在**临时库副本**上跑（不碰 `memory.db` 原文件），由既有 `conftest.py` 的 `isolated_memory` / `isolated_business_db` 提供隔离。

### 5.1 T-01 RuntimeContext（R3-01，5 例）

| 编号 | 前置条件 | 测试数据 | 步骤与时序 | 关键断言 | DB 预期 | 失败判定（反例） |
|---|---|---|---|---|---|---|
| T-01-A | 已登录（u1）；`audit_logs` 空 | 1 个后台任务，`identity=None` | 请求内 `run_detached(lambda: current_context())` → 任务体首句读 ctx | 抛 `ContextMissingError`；`done_callback` 记 1 条审计 | `audit_logs` +1，`actor_type='system'`，`action='background_context_missing'` | 任务体读到 `u1` → 隔离失效 |
| T-01-B | 同上 | `identity=snapshot_identity()` | 同上，任务体打印 ctx | `user_id=='u1'`、`detached is True`、`scopes==('background',)` | 无写入 | 任一项不符 |
| T-01-C | 同上 | 探针 ContextVar `PROBE` | 请求内 set `PROBE='x'` 并创建任务；请求返回后用 `Event` 放行任务 | 任务内 `PROBE.get() is None` | 无 | 读到 `'x'` → 继承未切断 |
| T-01-D | 仓库源码 | — | 运行 `tests/test_no_raw_spawn.py` | 0 命中 | — | 在 `main.py` 加 `asyncio.create_task(...)` 后仍全绿 |
| T-01-E | u1、u2 各 1 会话 | 各 100 次采样 | 两用户后台任务并发（`Barrier` 同步） | 每次采样 `ctx.user_id` 恒等于各自用户 | 无 | 出现交叉 |

### 5.2 T-02 孤儿消息（R3-02，6 例）

| 编号 | 前置条件 | 测试数据 | 步骤与时序 | 关键断言 | DB 预期 | 失败判定 |
|---|---|---|---|---|---|---|
| T-02-A | 临时库：2 会话 + 6 消息 | ①正常 ②会话无主 ③孤儿有主 ④孤儿无主 ⑤cid 空串 ⑥一致 | 跑 Q1/Q2 → 隔离（单事务） | Q1 命中 ③④、Q2 命中 ② | `messages` 仍 8 行；`owner_state='quarantined'` 恰 3 行；清单 3 行 | 行数变化或清单≠3 |
| T-02-B | T-02-A 之后 | 同数据 | 重复隔离 | 第二次 `changes()` 全 0 | 清单仍 3 行 | 出现第 4 行 |
| T-02-C | T-02-A 之前 | 加触发器：`BEFORE UPDATE ON messages WHEN NEW.id=:k RAISE(ABORT)` | 隔离 | 抛 `IntegrityError` | 清单 0 行；`owner_state` 全 `normal` | 半套（清单有、标记无） |
| T-02-D | T-02-A 成功后 | 隔离前快照（id,cid,user_id,state） | 逆向恢复 | `restored_at` 3 行非空 | 全表四元组与快照**逐行相等** | 任一不等 |
| T-02-E | T-02-A 后，为某孤儿插入 `conversations(user_id='u-new')` | 该孤儿 id | 逆向恢复 | 该行 `rowcount=0` 被跳过 | 该行仍 `quarantined`；其余 2 行恢复 | 被强行恢复 |
| T-02-F | 同 T-02-A 数据 | owner 为 NULL 的会话 | 跑 Q4（`IS NOT`）与 `<>` 对照 | `IS NOT` 命中 1，`<>` 命中 0 | — | 实现用 `<>` |

### 5.3 T-03 迁移状态机（R3-03，9 例）

| 编号 | 前置条件 | 测试数据 | 步骤与时序 | 关键断言 | DB 预期 | 失败判定 |
|---|---|---|---|---|---|---|
| T-03-A | 空临时库 | 第 2 步注入 `RAISE(ABORT)` | 跑迁移 | step1 `succeeded`、step2 `failed`、version 非 `succeeded` | `schema_steps` 2 行 | version 被标 `succeeded` |
| T-03-B | 空库 | — | 子进程：写 `running` 后 `os._exit(1)`（边界②） | 重启后 step2 探针为假 → 重跑成功 | 无 `duplicate column name`；step2 `succeeded` | 重启后报列已存在 |
| T-03-C | 空库 | — | 子进程：step 事务提交后、ledger 更新前 `os._exit(1)`（边界③） | 重启后**探针为真** → 直接标 `succeeded`，不再执行 DDL | `PRAGMA table_info` 列数不变 | 重复执行 DDL 抛错 |
| T-03-D | 空库 | 2 个进程 | `multiprocessing` 同时 `migrate()` | 恰 1 个成功；另一个 `MigrationBusy` | 表只有 1 套；`schema_steps` 无重复 | 两进程都执行 |
| T-03-E | 全部成功后 | 把 step10 改 `pending` | `--resume` | 不重复插入 | `migration_quarantine` 行数不变 | 行数增加 |
| T-03-F | 库版本改为 99 | — | 启动 | 抛 `schema_newer_than_code`，退出码非 0 | 无写入 | 正常启动 |
| T-03-G | `attempts=3, status=failed` | — | 启动；再 `--force-step` | 首次拒绝；force 后成功 | `audit_logs` +1 `actor_type='bootstrap'` | force 无审计 |
| T-03-H | 空库 | — | 完整迁移 | 11 步全 `succeeded` | `verify` 4 条计数全 0 | 任一非 0 |
| T-03-I | **现网副本**（53 消息 / 2 会话 / 0 孤儿） | — | 完整迁移 | step10 打标 0 行 | `messages` 53 行；`legacy` 仍在；清单 0 行 | 丢消息 |

### 5.4 T-04 审计主体（R3-04，7 例）

| 编号 | 前置条件 | 测试数据 | 步骤与时序 | 关键断言 | DB 预期 | 失败判定 |
|---|---|---|---|---|---|---|
| T-04-A | 空库 | `actor_type='user', actor_user_id=NULL` | 直接 INSERT | `IntegrityError: CHECK` | 0 行 | 插入成功 |
| T-04-B | 空库 | `actor_type='bootstrap', actor_user_id='u1'` | 直接 INSERT | `IntegrityError: CHECK` | 0 行 | 插入成功 |
| T-04-C | 已有 1 个 admin | — | 再跑 bootstrap CLI | 退出码 3 | `users` admin 仍 1 | 变成 2 |
| T-04-D | 无 `PAI_BOOTSTRAP_TOKEN` | — | 跑 CLI | 退出码 2 | `users` 无新行 | 创建成功 |
| T-04-E | u1 已登录 | body 塞 `{"user_id":"u2"}` | POST `/chat` | 字段被忽略/422 | 审计 `actor_user_id='u1'` | 写成 `u2` |
| T-04-F | 临时库 | `DROP TABLE audit_logs` | 走安全关键动作 + 数据类动作 | 前者 500、后者 200 + `X-Audit-Degraded: 1` | 安全关键动作的业务表无新行 | 前者返回 200 |
| T-04-G | u1 已登录 | `detail={"password":"abc123"}` | 触发审计 | 无明文 | 存 `sha256:...` 12 位 | 出现 `abc123` |

### 5.5 T-05 路由矩阵（R3-05，5 例）

| 编号 | 前置条件 | 测试数据 | 步骤与时序 | 关键断言 | DB 预期 | 失败判定 |
|---|---|---|---|---|---|---|
| T-05-A | 三种身份 token | 48 条 × 3 | 逐条请求 | 全部符合矩阵（401/403/200/201） | 无意外写入 | 任一 500 或与矩阵不符 |
| T-05-B | — | `app.routes` + `tests/route_matrix.py` | 集合比对 | `registered == declared` | — | 出现未登记路由 |
| T-05-C | 生产配置 | — | GET 4 条文档路由 | 全 404；`Route` 实例列表为空 | — | 任一非 404 |
| T-05-D | — | 临时加 1 条未登记路由 | 跑 T-05-B | 失败并指出路径 | — | 仍通过 |
| T-05-E | — | 13–22 号路由 | GET | 无 `Set-Cookie`；`Cache-Control` 含 `max-age` | — | 带 cookie |

### 5.6 T-06 KB 缓存（R3-06，9 例）

| 编号 | 前置条件 | 测试数据 | 步骤与时序 | 关键断言 | DB 预期 | 失败判定 |
|---|---|---|---|---|---|---|
| T-06-A | 缓存空 | 32 线程同 scope | `Barrier` 同时放行 | `KnowledgeBase()` 构造 1 次；32 请求全成功 | 无 | 构造 ≥2 次 |
| T-06-B | 8 个 scope 各被 1 请求占住（`Event` 卡住） | 第 9 个 scope | 第 9 个请求进入 | 成功创建；`kb_cache_overflow==1`；8 个未被打断 | 无 | 第 9 个 503，或前 8 个被 close |
| T-06-C | 16 个 scope 全占住 | 第 17 个 | 第 17 个请求 | 503 `kb_capacity_exhausted` | — | 淘汰在用实例 |
| T-06-D | 1 个占住 + 9 个空闲 | — | 反复 acquire/release | `len(_PRIVATE)==8`；被淘汰的 9 个 `CLOSED` | 磁盘知识文件数不变 | 文件被删 |
| T-06-E | public 常驻（refs>0）+ 15 私有 | — | 触发淘汰 | `_PUBLIC` 非 None 且 `refs>0` | — | public 被替换 |
| T-06-F | mock 构造抛异常 | 10 个并发 | 同时 acquire | 全部 `KbUnavailable`；构造仅 1 次 | — | 构造 10 次（构建风暴） |
| T-06-G | 请求持有 entry | — | 请求中途 `invalidate_public()` | 在途请求正常返回；下一个拿新实例 | — | 在途请求失败 |
| T-06-H | entry `READY` | — | 调 `VectorStore.add` | `RuntimeError("vector store is sealed")` | — | 静默 append |
| T-06-I | — | 1000 次 `_acquire` | 埋点计时 | p99 < 5 ms | — | 超时（锁内做了 I/O） |

### 5.7 T-07 预算状态机（R3-07，10 例）

| 编号 | 前置条件 | 测试数据 | 步骤与时序 | 关键断言 | DB 预期 | 失败判定 |
|---|---|---|---|---|---|---|
| T-07-A | 日限额 10 | 50 并发 | `Barrier` 同时 reserve | 恰 10 个拿到 rid；40 个 429 | `calls==10`；`reservations` 10 行 `reserved` | `calls>10` |
| T-07-B | 1 个 rid | 同 rid settle ×2 | 顺序调用 | 第一次 True，第二次 False | `total_tokens` 只加一次 | 加两次 |
| T-07-C | 1 个 rid 已 settled | — | release | 返回 False | `calls` 不变 | 被退还 |
| T-07-D | 1 个 rid | `settle(None)` | — | `provider_reported_tokens IS NULL` | `total_tokens` 不变；`unknown_usage_count+1` | 写成 0 |
| T-07-E | 2 个 rid | 先 `settle(100)`，再 `settle(None)` | — | 已知量保留 | `total_tokens==100`；unknown==1 | `total_tokens` 变 NULL/0 |
| T-07-F | 日限额 5 | `kind='image'` ×2 后 release ×1 | — | 两类计数同时减 | `calls==1`、`image_calls==0` | 只减一个，或出现负数 |
| T-07-G | 1 个 rid | 模拟 timeout → `keep` | — | 额度不退 | `state='kept'`；`calls` 不变 | 被 release |
| T-07-H | 3 条过期 `reserved` | — | 重启 | 全部转 `kept` | `usage_daily.calls` 不变；审计 +3 | 被退还 |
| T-07-I | 人为改 `total_tokens` | 与明细不一致 | `reconcile()` | 报 mismatch | `usage_daily` 原值不变 | 自动修正 |
| T-07-J | 另一连接持写锁 | `timeout=10` | reserve | 最多重试 2 次 → 503 `budget_busy` | 无脏写 | 跳过计量直接放行 |

### 5.8 T-08 CSRF/Session（R3-08，13 例）

| 编号 | 前置条件 | 测试数据 | 步骤与时序 | 关键断言 | DB 预期 | 失败判定 |
|---|---|---|---|---|---|---|
| T-08-A | 缓存空 | — | GET `/login` | 有 `pai_csrf`；无 `pai_sess`；HMAC 自校验通过 | 无 | 无 cookie 或签名错 |
| T-08-B | 匿名 | 无 `X-CSRF-Token` | POST `/auth/login` | 403 | `sessions` 0 行 | 200 |
| T-08-C | 匿名 | token 正确但三头全缺 | POST login | 403 | 0 行 | 200 |
| T-08-D | 匿名 | `Origin: https://evil.com` | POST login | 403 | 0 行 | 200 |
| T-08-E | 匿名 + token | 正确口令 | POST login | 200；新 `__Host-pai_sess`；**新 csrf ≠ 旧 csrf** | `sessions` 1 行 | csrf 未变（可固定） |
| T-08-F | 已登录 | 4 种错误 token（不一致/缺失/旧/他人） | 写请求 ×4 | 全 403 | 业务表无新行 | 任一 200 |
| T-08-G | 已登录 | 20 并发 GET | 同一秒内 | `last_seen_at` 最多更新 1 次 | `sessions` 1 行 | 更新 20 次 |
| T-08-H | 已登录 | — | POST logout | `revoked_at` 非空；两 cookie 清空；后续 401 | `sessions.revoked_at` 非空 | 后续仍 200 |
| T-08-I | 会话 | `last_seen_at=now-25h` | GET `/auth/me` | 401 | 行保留（不删） | 200 |
| T-08-J | 会话 | `created_at=now-8d`，`last_seen_at=now-1min` | GET | 401；续期不能救 | 行保留 | 200 |
| T-08-K | 会话 | `expires_at == now` | GET | 401（边界用 `<`） | 行保留 | 200 |
| T-08-L | 生产模式 | 缺/非法 `PAI_PUBLIC_ORIGIN` | 启动 | 退出码 2 | — | 启动成功 |
| T-08-M | 已登录 | — | GET `/auth/me` + 静态资源 | `no-store` / `max-age` | — | 带身份响应可缓存 |

### 5.9 T-09 备份恢复（R3-09，8 例）

| 编号 | 前置条件 | 测试数据 | 步骤与时序 | 关键断言 | DB 预期 | 失败判定 |
|---|---|---|---|---|---|---|
| T-09-A | 备份文件截断 1 KB | — | 跑备份校验 | G3 失败 | 目标库零改动；坏备份被删 | 进入恢复 |
| T-09-B | 有持续写入线程 | `live` 模式 | 备份 | 成功；G2 记录变化；G4 在备份自身上自洽 | 源库继续增长；备份是其快照 | 因行数不等而失败 |
| T-09-C | 读者持读事务 | — | checkpoint | G1 抛 `checkpoint_incomplete busy=1` | 无备份文件 | 继续备份 |
| T-09-D | 非 WAL 库（现状） | — | 备份 | 走非 WAL 分支（`log==-1` 且无 `-wal`） | 备份成功 | 误判为 WAL 通过 |
| T-09-E | 备份后删 1 加 1 | `n` 不变 | G4 | `pk_sha256` 不等 → 失败 | — | 因 `n` 相同而通过 |
| T-09-F | R4 后 | 让 R6 故意失败 | 恢复 | 回滚 | 目标库 SHA-256 == `.pre_restore_*` | 目标库被改坏 |
| T-09-G | 恢复完成 | — | 校验 | `foreign_key_check` 空列表；`integrity_check='ok'` | `user_version` 匹配；`orders>=1` | 任一不符 |
| T-09-H | 人为留 1 个连接 | — | 恢复 | G6 中止 | 目标库未改名 | 强行恢复 |

### 5.10 T-10 依赖与平台（R3-10，5 例）

| 编号 | 前置条件 | 测试数据 | 步骤与时序 | 关键断言 | DB 预期 | 失败判定 |
|---|---|---|---|---|---|---|
| T-10-A | 干净 Linux venv | `requirements-linux.lock` | `pip install --require-hashes` | 成功；`import fastapi,PIL,multipart,argon2` 全通过 | — | 任一 ModuleNotFoundError |
| T-10-B | 同上 + 断网 | 本地 wheel 目录 | 安装 | 成功 | — | 需要联网 |
| T-10-C | 源码 | — | 重跑 `pip-compile` | diff 为空 | — | 有差异 |
| T-10-D | 生产镜像 | — | `import edge_tts` | `ModuleNotFoundError` | — | 导入成功（说明带了残留） |
| T-10-E | 镜像 `TZ=Asia/Shanghai` | 写 1 条消息 | 比对时间 | 与宿主机差 < 2 s | `created_at` 正确 | 差 8 h |

### 5.11 T-R 回归（3 例）

| 编号 | 前置条件 | 测试数据 | 步骤与时序 | 关键断言 | DB 预期 | 失败判定 |
|---|---|---|---|---|---|---|
| T-R-A | 现网副本 | 既有 35 项 UI 测试 + RAG/记忆/端口测试 | 全量跑 | 全绿 | 与基线一致 | 任一失败 |
| T-R-B | 现网副本 | `/`、`/workbench` | GET 比对字节 | 响应体与磁盘文件逐字节相同 | — | 有差异 |
| T-R-C | 现网副本 | 53 条消息 / 2 会话 / 5 条记忆 | 迁移后对比 | `messages` 53 行、内容哈希不变 | 会话 `legacy` 仍在 | 行数或内容变化 |

**用例总数：5+6+9+7+5+9+10+13+8+5+3 = 80**（T-10 由 4 例扩为 5 例，故总数为 80，此前 79 的估计按 4 例计）。

---

## 6. 覆盖率与反例清单

### 6.1 修订项 → 测试组 → 用例数

| 修订 | 级别 | 测试组 | 用例数 | 是否有「机制层」验证 | 机制层验证方式 |
|---|---|---|---|---|---|
| R3-01 | P0 | T-01 | 5 | ✅ | T-01-D 静态 AST 守卫（改坏必红） |
| R3-02 | P0 | T-02 | 6 | ✅ | T-02-F + `tests/test_sql_nullsafe.py` 正则拦截 |
| R3-03 | P0 | T-03 | 9 | ✅ | T-03-B/C 子进程 `os._exit` 真实崩溃 |
| R3-04 | P1 | T-04 | 7 | ✅ | T-04-A/B 由 DDL CHECK 强制，非应用层 |
| R3-05 | P0 | T-05 | 5 | ✅ | T-05-B 路由集合比对（`registered == declared`） |
| R3-06 | P0 | T-06 | 9 | ✅ | T-06-A 构造计数、T-06-H `sealed` 守卫 |
| R3-07 | P0 | T-07 | 10 | ✅ | T-07-A 并发限额（DB 层原子性） |
| R3-08 | P0 | T-08 | 13 | ✅ | T-08-E CSRF 必换、T-08-K 边界 `<` |
| R3-09 | P0 | T-09 | 8 | ✅ | T-09-D 非 WAL 陷阱、T-09-E 哈希而非行数 |
| R3-10 | P1 | T-10 | 5 | ⚠️ | 依赖层面，需目标机（V-09） |
| R3-11 | P0 | T-R | 3 | ✅ | T-R-C 内容哈希不变 |

### 6.2 11 个反例点（「把实现改坏，测试必须变红」）

| # | 改坏什么 | 哪个用例必须失败 |
|---|---|---|
| 1 | 把 `create_task(context=Context())` 改成裸 `create_task` | T-01-A |
| 2 | 在 `main.py` 加一行 `asyncio.create_task(...)` | T-01-D |
| 3 | 把 Q1 的 `NOT EXISTS` 改回 `NOT IN (SELECT ...)` | T-02-A / T-02-F |
| 4 | 把逆向恢复的 `AND quarantine_batch=:batch` 删掉 | T-02-E |
| 5 | 把 step 执行改成 `executescript` | T-03-A（失败后表已建） |
| 6 | 把版本成功判定改成「直接 UPDATE」 | T-03-G |
| 7 | 把 `rowcount==0` 判定删掉 | T-07-A（`calls` 会超 10） |
| 8 | 把 `COALESCE(:tokens, 0)` 改成 `:tokens` | T-07-E（`total_tokens` 变 NULL） |
| 9 | 登录后不轮换 CSRF | T-08-E |
| 10 | 把 G1 改成只看 `busy==0` | T-09-D（非 WAL 库误判通过） |
| 11 | 把 G4 从四元组改成只看 `COUNT(*)` | T-09-E |

> 这张表本身就是验收物：实施轮必须逐条改坏一次并记录「测试确实红了」。**只跑一次全绿不算通过。**

---

## 7. 三类放行判定

### 7.1 设计放行：✅ 通过

| 判定条件 | 结果 |
|---|---|
| 11 项修订均有明确方案 | ✅ §3 逐项给出 |
| 每项都有可执行验收标准 | ✅ §3 各小节「验收标准」+ §5 八字段矩阵（80 例） |
| 关键风险均有测试方案 | ✅ §3 + §6.2 反例清单 |
| 未解决的 P0 阻断项 | **0** |
| 未经验证的推断被写成事实 | ✅ 无（全部标 `【实测】/【代码】/【待核验】`） |

**结论：设计可以作为编码依据。**

### 7.2 编码放行（R2-2）：⏸ 建议进入，但**不自动授权**

进入编码前，用户需要确认以下 4 项（这些是「编码前必须知道」的，不是「上线前必须知道」的）：

| # | 待确认 | 影响范围 | 若不确认的后果 |
|---|---|---|---|
| E-1 | `PAI_SECRET_KEY` 的生成与轮换策略（谁生成、存哪、轮换后旧会话是否全失效） | R3-08 全链路 | CSRF 签名与会话摘要都依赖它，事后改会强制全员登出 |
| E-2 | 新增 10 条路由挂在 `APIRouter` 还是 `main.py` 直挂 | R3-05 的 `app.routes` 计数（31 或 40） | 影响 T-05-B 断言值，但不影响功能 |
| E-3 | `business.db` 是否需要同样纳入认证（当前设计：共享演示数据、HR 只读、写限 admin） | R3-04 审计分级、R3-05 数据范围 | 若要求「按用户隔离 business 数据」，需重做表结构 |
| E-4 | 迁移是否允许在**现网 memory.db 原文件**上执行（当前设计：先复制副本，在副本上试跑 T-03-I） | R3-03 执行流程 | 直接在生产库上试跑，失败即无退路 |

> 本报告**不建议**在 E-1/E-4 未定时开工。这两项不涉及技术分歧，只需一句确认。

### 7.3 生产上线放行：❌ 不通过

| 阻断项 | 内容 | 为什么阻断上线 |
|---|---|---|
| V-01 | Argon2id 在目标机（2C2G）的 p95 与 RSS 未实测 | 参数选大了会 OOM，选小了不安全。**这是安全参数，不能猜** |
| V-02 | `argon2-cffi` 在目标平台的 wheel 可得性未验证 | 无 wheel 则需编译（gcc + libffi），镜像构建会失败 |
| V-03 | BGE 模型 + 多实例 KB 缓存在 2C2G 的峰值内存未实测 | R3-06 的 `_PRIVATE_HARD_MAX=16` 是按「未知单实例内存」定的，可能直接 OOM |
| V-09 | 目标 Linux 环境的架构/发行版/glibc 版本未确认 | 决定 lock 文件的 wheel 标签，错了装不上 |

**结论：设计通过 ≠ 可以上线。这 4 项必须在编码完成后、部署前，在目标机上实测。**

---

## 8. 待核验清单（12 项）

| 编号 | 待核验事项 | 为什么现在无法确定 | 需要什么证据 | 阻断级别 |
|---|---|---|---|---|
| V-01 | Argon2id 在 2C2G 的 p95 与 RSS | 无目标机；本机 13.9 GiB 不代表生产 | 目标机跑参数网格（19/32/46/64 MiB × t=2/3 × p=1/2） | **上线** |
| V-02 | `argon2-cffi` 目标平台 wheel | 本轮禁止安装 | §3.10.3 (3) 的 `pip download` 产物清单 | **上线** |
| V-03 | BGE 首次索引 + 常驻 RSS 峰值 | 未加载过模型 | 目标机 `psutil` 采样，单实例 + 8 实例两组 | **上线** |
| V-04 | FastAPI 0.141 `_IncludedRouter` 展平时机 | 只观测到「不展平」，未读其源码 | 锁定版本后跑注册表枚举脚本 | 编码（T-05-B 断言值） |
| V-05 | WAL 模式与 `conftest.py` 指纹哨兵的兼容性 | 现网是 `delete` 模式；若恢复带来 WAL，哨兵可能漏检 `-wal/-shm` | 在副本上改 WAL，跑一次全量测试看哨兵是否覆盖 | 编码 |
| V-06 | 时间格式不一致（旧表本地时间 vs 新表 UTC） | 旧表 53 行是 `datetime.now()` 本地时间 | 确认「新表只存 UTC、禁止跨表比较」的评审规则 | 编码（不阻断） |
| V-07 | 单 Worker 假设下的进程内状态（限速桶、KB 缓存） | uvicorn 现为默认 1 worker | 若改多 Worker，需外部共享存储 | 上线 |
| V-08 | `pillow` / `python-multipart` 未在依赖声明 | 已装但未声明 | 写入 `requirements.in`（§3.10.3 (2) 已给） | 编码 |
| V-09 | 目标 Linux 架构 / 发行版 / glibc | 未确认（x86_64 假设未验证） | `uname -m`、`cat /etc/os-release`、`ldd --version` | **上线** |
| V-10 | `PAI_SECRET_KEY` 的生成/轮换策略 | 产品决策未定（E-1） | 用户确认 | 编码 |
| V-11 | 新增 10 条路由的挂载方式 | 实现选择未定（E-2） | 用户确认 | 编码（仅影响计数） |
| V-12 | `business.db` 是否需要按用户隔离 | 产品决策（E-3） | 用户确认 | 编码（若需隔离则重做表） |

> **P0 阻断项（阻断设计）为 0；阻断编码 4 项（V-04/V-05/V-10/V-11，均可通过一句确认或一次实验解决）；阻断上线 4 项（V-01/V-02/V-03/V-09）。**

---

## 9. 文件指纹与 Git 状态

### 9.1 本轮唯一新增文件

| 文件 | 修改前 | 修改后 |
|---|---|---|
| `docs/PAI-AUTH-R2-1-R3_设计修订报告.md` | 不存在 | 见 §9.3 |

### 9.2 代码 / 配置 / 数据库：前后一致（零改动）

| 文件 | 开工 SHA-256（前 16） | 收工复核 | 结论 |
|---|---|---|---|
| `main.py` | `a8397b5433a4e680` | 相同 | 一致 |
| `memory.py` | `8a79e394607022e5` | 相同 | 一致 |
| `agent.py` | `b8a8e32f20669106` | 相同 | 一致 |
| `knowledge_store.py` | `5d46ff6abf9189fa` | 相同 | 一致 |
| `tools.py` | `bee0341bad95af5b` | 相同 | 一致 |
| `business/api.py` | `6385020cdf6fc0a9` | 相同 | 一致 |
| `business/db.py` | `4afa0fac166160ae` | 相同 | 一致 |
| `conftest.py` | `5fde88c5e28dc142` | 相同 | 一致 |
| `rag/knowledge_base.py` | `c35f36d3f4a7f078` | 相同 | 一致 |
| `rag/vector_store.py` | `ce61b2b5c2cadd05` | 相同 | 一致 |
| `rag/embedding.py` | `06acb4f5d39c5893` | 相同 | 一致 |
| `task.py` | `7a6f30eb3f3e0fa5` | 相同 | 一致 |
| `tool_registry.py` | `af52be93e96a79c8` | 相同 | 一致 |
| `memory.db` | `dbe0d2a497b02236`（98,304 B / mtime 2026-11-07 18:40:03） | 相同 | 一致 |
| `data/business.db` | `678a9b871bf7403f`（73,728 B / mtime 2026-11-07 17:57:07） | 相同 | 一致 |

### 9.3 本轮结束时的实际指纹

本次实测于文档落盘后执行（`sha256sum` / `wc` / `stat` / `git status --porcelain`）。

**新增文件（本文档自身）**

| 文件 | size | 行数 | SHA-256 | mtime |
|---|---|---|---|---|
| `docs/PAI-AUTH-R2-1-R3_设计修订报告.md` | 138,498 B | 2,136 | `c4662354c8175383f27d9fe146b4ab05dc5b8320cf24c61a3f010c6fb28ce600` ① | 2026-11-09 15:35:54 +0800 |

> ① 自引用说明：文档内嵌自身 SHA-256 时，写入该值这一动作会改变文件哈希，因此内嵌值必然对应「写入本节之前」的版本，不可能与定稿文件逐位相同。上表 size / 行数 / mtime / SHA 对应同一版本。**定稿版本的实测 SHA-256 记录在对话交付回执中**，复核命令：
> `sha256sum docs/PAI-AUTH-R2-1-R3_设计修订报告.md`
> 该偏差仅影响本行；其余 15 项（源码 13 + 数据库 2）均可用 `sha256sum` 独立复核并与 §9.2 逐位比对。

**结构完整性自检**

| 检查 | 结果 |
|---|---|
| 代码围栏（```）计数 | 86（偶数，无未闭合块） |
| 一级章节 | 11 个（§0–§10） |
| 三级章节 | 33 个 |
| 显式标记 `【实测】` | 26 处 |
| 显式标记 `【待核验】` | 出现在 §2、§3.10、§8 共 9 处 |
| 测试用例编号 | T-01(5) / T-02(6) / T-03(9) / T-04(7) / T-05(5) / T-06(9) / T-07(10) / T-08(13) / T-09(8) / T-10(5) / T-R(3) = **80** |

**Git**

| 指标 | 开工 | 收工 | 差值 |
|---|---|---|---|
| HEAD | `81a8f2d` | `81a8f2d` | 未变 |
| 工作区条目 | 59（25 M/D + 34 ??） | **60**（25 M/D + 35 ??） | **+1（仅本文档）** |
| `stash` 条目 | 0 | 0 | 未变 |
| `docs/` 文件数 | 21 | **22** | +1（仅本文档） |

唯一新增的 `??` 条目即 `docs/PAI-AUTH-R2-1-R3_设计修订报告.md`；25 项已跟踪 `M`/`D` 集合逐条未变，指纹与 SHA-256 均未变。

---

## 10. 本轮实际操作披露与零改动声明

### 10.1 实际执行过的操作（完整披露，无省略）

| # | 操作 | 位置 | 是否触碰项目 |
|---|---|---|---|
| 1 | `git status` / `git rev-parse` / `git stash list` / `sha256sum` / `stat` / `wc` | 项目目录 | 只读 |
| 2 | 读取源码：`main.py`、`memory.py`、`rag/knowledge_base.py`、`rag/vector_store.py`、`business/db.py`、`business/schema.py` 片段 | 项目目录 | 只读 |
| 3 | `grep` / `Glob` 检索 `asyncio|BackgroundTasks|run_in_executor|to_thread`、`datetime.now()`、`os.getenv` | 项目目录 | 只读 |
| 4 | 写 7 个探针脚本并运行（`probe_ctx.py`、`probe_isolation.py`、`probe_sqlite.py`、`probe_walfile.py`、`probe_sql.py`、`probe_routes.py`、`probe_script.py`） | `%TEMP%\pai_r3_probe\`（**系统临时目录，非项目**） | 否 |
| 5 | 探针在临时目录创建 6 个一次性 SQLite 文件（`wal_probe*.db`、`plain_probe.db`、`script_probe*.db`、`sql_probe.db`） | 同上 | 否 |
| 6 | 只读查询 `memory.db`（`file:memory.db?mode=ro`）统计会话/消息/孤儿数 | 项目目录 | **只读 URI，未开写连接** |
| 7 | 读取 `importlib.metadata` 获取 26 个包的已安装版本 | 系统 | 否 |
| 8 | AST 静态解析 `main.py` / `business/api.py` 提取路由（**不 import 项目模块**） | 项目目录 | 只读 |
| 9 | 新建 `docs/PAI-AUTH-R2-1-R3_设计修订报告.md` | 项目目录 | **唯一写入** |

**没有执行的操作**：没有 import 任何项目模块（因此没有触发 `dotenv.load_dotenv`，**没有读取 `.env`**）；没有启动服务；没有调用任何模型 API；没有 `pip install/uninstall`；没有执行任何 `git` 写操作（`commit`/`reset`/`checkout`/`stash` 均未执行）；没有修改任何 SQLite 数据库；没有删除、覆盖、重命名任何既有文件。

### 10.2 零改动声明

- 未修改任何 Python 源码、测试文件、前端资源、配置文件。
- 未执行任何 DDL/DML；两个 SQLite 文件 size、mtime、SHA-256 与开工时逐位一致（§9.2）。
- 未安装、升级、卸载任何依赖。
- 未读取、未修改 `.env`（代码中只观察到 `DEEPSEEK_API_KEY` 一个变量名，**未读取其值**）。
- 未输出任何密钥、密码、Token、邀请码或对话/记忆正文。
- 唯一写入为新增本文档；R1 与 R2 的设计文档**未覆盖、未删除、未重命名**。
- Git 工作区条目由 59 增至 60（唯一新增即本文档），已跟踪的 25 项 `M`/`D` 集合逐条未变。

---
