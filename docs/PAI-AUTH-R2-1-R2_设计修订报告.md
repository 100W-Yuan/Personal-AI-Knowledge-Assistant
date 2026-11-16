# PAI-AUTH-R2-1-R2：认证系统设计修订报告

- 项目路径：`C:\Users\Doro\Desktop\my-ai 25`
- 基线：HEAD `81a8f2dec62df00a82b9d838cfa18d7b98962b61` / 分支 `main` / 工作区 58 项 / Python 3.12.10 / SQLite 3.49.1
- 文档性质：设计修订，不含代码实现
- 生成时间：2026-11-09 15:0x（本地）

---

## 1. R1 原始设计核对结果

### 1.1 R1 报告的载体

R1 报告（`PAI-AUTH-R2-1-R1`）以会话消息形式交付，**磁盘上不存在同名设计文件**（`find . -iname "*PAI-AUTH*"` 与 `-iname "*R2-1*"` 均无命中；`docs/` 写入前 20 项中无该文件）。因此本轮不存在「覆盖 R1 原始报告」的风险；本文件是**新增**文档，`docs/` 由 20 项变为 21 项。

### 1.2 R1 关键结论的代码复核（逐条取证）

| R1 结论 | 复核证据 | 结论 |
|---|---|---|
| 工具执行不传上下文 | `tool_registry.py:129 def execute(self, name, arguments)`；`:145 result = function(**arguments)` | 成立 |
| 未知参数被注册表拒绝（可挡模型伪造 user_id） | `tool_registry.py:67-74`（`未知参数：xxx`） | 成立 |
| Agent 单例持有共享 Memory 并闭包绑定工具 | `main.py:236-237 @lru_cache(maxsize=1)`；`agent.py:657/670/719` | 成立 |
| 工具执行入口签名带 state 但注册表不接收 | `agent.py:1804 execute_tool(self, state, tool_name, arguments)`；`:1984` | 成立 |
| messages 无 user_id，归属可经 conversation 推导 | `memory.py:486 save_message` 内必调 `ensure_conversation`；`schema`：messages 仅 id/role/content/created_at/conversation_id | 成立 |
| KnowledgeBase 自带初始化锁、embedding 可注入 | `rag/knowledge_base.py:130/140/246`、`:555 _default_embedding_factory` | 成立 |
| 全项目无 usage 读取 | `grep -n usage agent.py main.py image_client.py` 零命中 | 成立 |
| 视觉客户端只回文本 | `image_client.py:40 analyze(...) -> str`；`image_understanding.py:19 -> ImageResult`（无 usage 字段） | 成立 |

### 1.3 R1 中需要修正的表述（本轮自我纠正）

| R1 表述 | 问题 | 修正 |
|---|---|---|
| 「38 条路由 + 3 条内建」「39 行矩阵」 | 口径混用：38 是**方法级操作数**，36 是**路径条目数**，39 是表格行数（含合并行与内建行） | 见 §6：以运行期注册表为准逐项列出，并统一口径 |
| 「business 16 条」 | 过滤 `startswith("/business")` 时误含 `/business-ui` 的 3 条 | business 实为 **13 条**（6 POST + 7 GET） |

---

## 2. 11 项修订逐项对照表

| 编号 | 主题 | 结论 | 是否需改代码（实施轮） | 阻断级别 |
|---|---|---|---|---|
| R2-01 | RuntimeContext 传播 | 采用 ContextVar + 端点 body 内设置 + 禁止嵌套 + 禁止后台任务隐式继承 | 是（`main.py`/`agent.py`/`tools.py`/新增 `auth/context.py`） | P0 |
| R2-02 | 孤儿消息 | 不猜测归属；N>0 即中止迁移；人工授权后「NULL 所有者会话」隔离，可逆 | 是（迁移脚本） | P0 |
| R2-03 | 迁移状态机 | 状态用**独立连接独立事务**先写 running，再开业务事务；失败可持久化；启动门禁 | 是（迁移框架 + lifespan） | P0 |
| R2-04 | 审计主体 | `actor_type ∈ {user, system, bootstrap}` + 条件 CHECK；不伪造用户 ID | 是（`audit_logs` DDL + 写入封装） | P0 |
| R2-05 | 路由矩阵 | 运行期枚举：36 paths / **38 ops** / +4 内建 / 实施后 +10 = 48 ops | 是（路由保护） | P1 |
| R2-06 | KB 缓存 | public 独立槽永不淘汰；私有 LRU=8；锁随 entry 生命周期消亡 | 是（`main.py` 缓存层） | P0 |
| R2-07 | 计量预算 | reserve→settle/release 状态机；`changes()==1` 原子预扣；无 usage 不估算 | 是（`auth/metering.py` + 2 表） | P0 |
| R2-08 | CSRF/Origin/Session | 双提交 + 会话绑定摘要 + 登录/注册严格 Origin + `Cache-Control: no-store` | 是（`auth/deps.py`） | P0 |
| R2-09 | 备份恢复 | 6 道硬门禁（G1–G6），任一失败中止；恢复 6 步 | 是（`migrations/backup.py`） | P0 |
| R2-10 | 生产依赖 | 生产 9 项（含隐式 multipart）+ 新增 argon2-cffi；测试 5 项；lock 精确锁定 | 是（依赖文件） | P1 |
| R2-11 | 测试验收 | 13 组用例覆盖 11 项修订，全部给出可判定断言 | 是（测试文件） | P1 |

---

## 3. 各项修订后的完整设计

### R2-01 RuntimeContext 与 ContextVar 传播规则

#### 设计结论

```python
# auth/context.py
@dataclass(frozen=True)
class RuntimeContext:
    user_id: str
    request_id: str
    scopes: tuple[str, ...]        # ("public", f"u:{user_id}")

_CURRENT: ContextVar[RuntimeContext | None] = ContextVar("pai_ctx", default=None)

@contextmanager
def runtime_context(ctx: RuntimeContext):
    if _CURRENT.get() is not None:
        raise RuntimeError("nested runtime context")   # fail-closed，防泄漏叠加
    token = _CURRENT.set(ctx)
    try:
        yield ctx
    finally:
        _CURRENT.reset(token)                          # 必须恢复，见下方线程池风险

def current_context() -> RuntimeContext:
    ctx = _CURRENT.get()
    if ctx is None:
        raise RuntimeError("runtime context missing")
    return ctx
```

#### 传播规则表（含依据）

| 场景 | 是否继承 | 规则与依据 |
|---|---|---|
| 同步端点 `def` 的 body | 是（同线程） | 在 body **首行** `with runtime_context(...)`。理由：`/chat` 为 `def chat`（`main.py:305`），整个 Agent 执行在同一线程池线程内完成 |
| 同步依赖 `Depends`（`def`） | **不可依赖** | FastAPI 把同步依赖与同步端点分别调度到线程池，**不保证同一线程**；若在依赖里 `set()`，端点 body 可能读不到（读到 `None`）→ 因此身份**只能作为参数**返回，由 body 自行建立上下文 |
| 异步端点 `async def` | 是（同 task） | `/analyze-image` 为 `async def`（`main.py:1096`），await 的子协程继承同一 context；同样在 body 内设置 |
| `asyncio` 内新建 Task | 是（创建时复制） | 新 Task 复制创建时刻的 context；**不允许**把身份借此带给「响应之后才完成」的任务（见后台任务行） |
| `run_in_executor` / `ThreadPoolExecutor` | **否** | 线程不继承 Python context。若 Agent 未来引入并行工具，必须 `ctx = contextvars.copy_context(); loop.run_in_executor(None, ctx.run, fn)` |
| `BackgroundTasks` | 是（同请求上下文） | 但**禁止**依赖：后台任务必须显式接收 `RuntimeContext` **快照**作为参数，任务体内不允许调用 `current_context()`。理由：后台任务可能在响应后执行，隐式继承会把「上一个请求的用户」带到不确定的时间点 |
| 工具调用链 `execute_tool → registry.execute → function(**args)` | 是 | 纯同步调用栈，无线程/任务切换（`agent.py:1804`、`1984`） |
| 线程池线程复用 | **必须 reset** | anyio 线程池会复用线程；若忘记 `reset`，**下一个请求会继承上一个用户的身份**。这是本设计最危险的失效模式，由 ① `try/finally reset` ② 进入时禁止嵌套 ③ 测试断言请求后 `_CURRENT.get() is None` 三重防护 |

#### 实现约束

1. 身份只来自 `Depends(get_current_user)` 的返回值，**不读** 请求体/查询参数/请求头里的 `user_id`。
2. 不得用模块级全局变量保存当前用户（并发下必然串号）。
3. 不得用工具闭包捕获 user_id（单例下会冻结首个用户）。
4. `Agent.run(..., user_id)` 保留显式参数用于所有 SQL；入口断言 `user_id == current_context().user_id`，不一致直接 500。
5. `/analyze-image` 与未来所有「会调模型的异步端点」同样在 body 内建立上下文。

#### 失败处理

| 失效 | 表现 | 处理 |
|---|---|---|
| 忘记 reset | 下一请求读到他人身份 | 禁止嵌套 + 测试断言；生产开启 `PAI_CTX_STRICT=1` 时在请求结束钩子再校验一次 `_CURRENT.get() is None` |
| 上下文缺失 | 工具抛 `RuntimeError` | 端点转 500（不降级为「匿名访问」，避免静默越权） |
| 线程池子线程无上下文 | 同上 | 明确禁止在子线程内调工具；需并行时按上表 `copy_context()` |

#### 测试验收

- T-01-A：两线程并发，各 50 次交替调用同一 Agent 实例，A 的回答/写入只含 A 的数据，B 同理（断言双向不串）。
- T-01-B：同步依赖里 `set()` 的写法必须**失败**（回归保护：证明我们没有在依赖里设上下文）。
- T-01-C：请求结束后 `_CURRENT.get() is None`（在测试中通过中间件/钩子采样）。
- T-01-D：嵌套 `runtime_context` 抛 `RuntimeError`。
- T-01-E：`BackgroundTasks` 任务体内调 `current_context()` 抛错（强制显式传参）。

---

### R2-02 messages 归属与孤儿数据

#### 设计结论

- `messages` **不新增** `user_id`；归属唯一来源是 `conversations.user_id`。
- 迁移只改 `conversations`（回填）与 `memories`（回填），`messages` 一列不动 → 从结构上消灭「消息与会话 owner 不一致」。
- 孤儿消息（`conversation_id` 在 conversations 中不存在）**不猜测归属、不删除、不归给管理员**。

#### 迁移前核验

```sql
-- N1: 孤儿消息数（期望 0）
SELECT COUNT(*) FROM messages m
LEFT JOIN conversations c ON c.conversation_id = m.conversation_id
WHERE c.conversation_id IS NULL;

-- N2: 各表基线数量（写入 ledger.detail）
SELECT (SELECT COUNT(*) FROM conversations),
       (SELECT COUNT(*) FROM messages),
       (SELECT COUNT(*) FROM memories);

-- N3: 内容量守恒基线（不输出正文）
SELECT COUNT(*), SUM(LENGTH(content)), MIN(id), MAX(id) FROM messages;
```

#### 失败停止条件（任一命中即中止迁移，ledger 置 failed，服务拒绝启动）

| 编号 | 条件 |
|---|---|
| S1 | N1 > 0（存在孤儿消息） |
| S2 | N2 三表数量 ≠ 2 / 53 / 5（与 R1 基线不符，说明库被外部改动） |
| S3 | 备份任一门禁失败（见 R2-09） |
| S4 | 回填后出现 `conversations.user_id IS NULL` 且未启用隔离模式 |
| S5 | 内容量守恒（N3）前后不一致 |

#### 孤儿隔离（仅在人工书面授权后执行）

1. 为每个孤儿 `conversation_id` 插入一条会话记录：`user_id = NULL`、`title = '[quarantined] <cid 前 8 位>'`、`created_at = 迁移时刻`。
2. 插入**必须早于** `trg_conversations_owner_required` 触发器创建（触发器只约束后续 INSERT，不约束既有行）。
3. 写审计：`action='orphan.quarantine'`、`actor_type='system'`、`detail_json={"count": N, "cids": ["前8位", ...]}`（不写正文）。
4. 隔离后这些会话对所有 `WHERE user_id = :uid` 查询不可见（NULL 永不等于任何 uid）。

#### 恢复与回滚

| 目标 | 操作 |
|---|---|
| 撤销隔离 | `DELETE FROM conversations WHERE title LIKE '[quarantined] %' AND user_id IS NULL`（只删本轮新增的占位会话，不触碰 messages） |
| 撤销回填 | `UPDATE conversations SET user_id = NULL`（列可空，旧代码不读） |
| 整库回滚 | 备份文件覆盖 + `integrity_check` |

#### 数量守恒检查（迁移后必须全绿）

```sql
SELECT 'messages_delta', (SELECT COUNT(*) FROM messages) - :baseline_messages
UNION ALL SELECT 'conv_delta', (SELECT COUNT(*) FROM conversations) - :baseline_conv - :quarantined
UNION ALL SELECT 'content_delta', (SELECT SUM(LENGTH(content)) FROM messages) - :baseline_len
UNION ALL SELECT 'orphans', (SELECT COUNT(*) FROM messages m LEFT JOIN conversations c
             ON c.conversation_id = m.conversation_id WHERE c.conversation_id IS NULL)
UNION ALL SELECT 'null_owner', (SELECT COUNT(*) FROM conversations WHERE user_id IS NULL);
-- 期望：前四行 0；null_owner 在非隔离模式下为 0，隔离模式下等于 :quarantined
```

#### 测试验收

- T-02-A：构造 3 条孤儿消息 → 迁移中止，ledger=failed，服务启动失败，且 **messages 行数仍为 56**（不删不猜）。
- T-02-B：授权隔离后 → `orphans=0`、所有用户查询不到这些会话、`messages` 行数不变。
- T-02-C：撤销隔离后原状恢复（孤儿重新出现，N1 复原）。
- T-02-D：53 条历史消息在新旧代码下读出的 `(id, length)` 序列完全一致。

---

### R2-03 迁移状态机

#### 缺陷修正

R1 版把 `schema_migrations` 的写入与业务 DDL 放在**同一个事务**：一旦 DDL 失败回滚，状态行也一起消失，等于「失败无记录」。本轮改为**双连接**：

```
连接 L（ledger，独立事务）        连接 M（业务 DDL，独立事务）
  BEGIN IMMEDIATE
  UPDATE/INSERT status='running'   ← 先提交，保证崩溃也可观测
  COMMIT
                                   BEGIN IMMEDIATE
                                   ... ALTER/CREATE/UPDATE ...
                                   COMMIT（成功）/ ROLLBACK（失败）
  BEGIN IMMEDIATE
  status='succeeded' / 'failed'
  COMMIT
```

**关键顺序约束**：`running` 必须先于业务事务**提交**，否则 M 的写锁会阻塞 L，或 L 的回滚吞掉状态。两连接均设 `busy_timeout=5000`。

#### 表结构

```sql
CREATE TABLE IF NOT EXISTS schema_migrations (
    version     INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    status      TEXT NOT NULL CHECK (status IN ('pending','running','succeeded','failed')),
    started_at  TEXT,
    finished_at TEXT,
    attempts    INTEGER NOT NULL DEFAULT 0,
    error       TEXT,
    detail      TEXT
) STRICT;
```

#### 状态转换

```
pending ──begin──> running ──成功──> succeeded
                      │
                      └──异常/崩溃──> failed ──修复后 resume──> running
                                        │
                                        └──禁止服务启动（门禁）
```

- `running` 在启动时被读到 = 上次进程崩溃（正常流程不会把 running 留到退出）。
- `failed` 必须人工处理：`migrate status` 查看 → 修复 → `migrate resume --version N` → 成功后自动 `succeeded`。

#### 启动门禁（硬约束）

```python
async def lifespan(app):
    assert_migrations_clean()      # 任一 failed/running → 抛异常，服务不启动
    init_database()
    yield
```

`assert_migrations_clean()` 语义：允许 `succeeded` 与「全部为 pending 且库为全新」；**不允许** `failed` 或 `running`。抛 `RuntimeError("migration blocked: <version> <status>")`，uvicorn 退出码非 0。

> 注意：`TestClient(main.app)` 不作为上下文管理器时**不会**执行 lifespan（现有 35 项 UI 测试即如此，已实测），因此门禁不会破坏既有测试；门禁测试需显式 `with TestClient(app)`。

#### 失败可持久化的验证方式

测试注入一个必然失败的 DDL（如对不存在的表 ALTER），断言：① 业务事务回滚后表结构未变；② `schema_migrations` 该行 `status='failed'` 且 `error` 非空；③ 重新 `import main` + 启动服务抛错。

#### 崩溃恢复

| 情形 | 行为 |
|---|---|
| 崩溃于业务事务提交前 | 库未变，ledger=running → 启动被拒 → `migrate resume` 重跑该步（该步必须幂等） |
| 崩溃于 commit 之后、ledger 更新前 | 库已变，ledger=running → `resume` 时先做该步的幂等探测（列/表/索引是否存在），已生效则跳过并置 succeeded |
| 崩溃于 ledger 更新中 | ledger 事务未提交 → 仍为 running → 同上 |

---

### R2-04 审计主体与管理员初始化

#### 冲突分析

R1 版 `actor_user_id NOT NULL` 无法表达：① 迁移/清理/隔离等系统动作；② 首个 admin 创建时「还没有任何 admin 可作为操作者」。若用「伪造一个用户 ID」解决，会污染审计真实性。

#### 修订后表结构

```sql
CREATE TABLE audit_logs (
    audit_id     TEXT PRIMARY KEY,
    occurred_at  TEXT NOT NULL,                    -- UTC ISO8601
    actor_type   TEXT NOT NULL CHECK (actor_type IN ('user','system','bootstrap')),
    actor_user_id TEXT REFERENCES users(user_id),  -- 可空
    actor_label  TEXT,                             -- 用户操作时存 username_key；系统操作存 'migrate'/'scheduler'
    action       TEXT NOT NULL,
    target_type  TEXT,
    target_id    TEXT,
    result       TEXT NOT NULL CHECK (result IN ('success','failure')),
    detail_json  TEXT,
    request_id   TEXT,
    ip_hash      TEXT,
    CHECK (
        (actor_type = 'user'   AND actor_user_id IS NOT NULL) OR
        (actor_type <> 'user'  AND actor_user_id IS NULL)
    )
) STRICT;
```

#### 三类主体

| actor_type | actor_user_id | 使用场景 | 示例 action |
|---|---|---|---|
| `user` | 必填 | 已登录用户的操作 | `invite.create`、`invite.revoke`、`user.disable`、`user.enable`、`session.revoke_all`、`user.create`（由现有 admin 执行） |
| `bootstrap` | NULL | 首个管理员初始化（CLI，无既有主体） | `admin.bootstrap` |
| `system` | NULL | 迁移、孤儿隔离、定时清理 | `migration.run`、`orphan.quarantine`、`session.gc` |

#### 管理员初始化流程（CLI）

1. `python -m auth.cli create-admin`（必须用 3.12.10 绝对路径）。
2. 交互读取 username / password 两次；密码不回显、不入日志、不进 shell history。
3. 事务内：`INSERT users(role='admin')` + `INSERT audit_logs(actor_type='bootstrap', actor_user_id=NULL, action='admin.bootstrap', target_type='user', target_id=<新 uid>, result='success')`。
4. 若事务回滚 → 不写审计（原子性），但 CLI 打印失败原因（不含密码）。
5. **多管理员**：后续由已有 admin 通过 CLI `create-admin --as <admin_username>` 创建 → `actor_type='user'`、`actor_user_id=<该 admin>`；若 `--as` 缺失或不是 admin → 拒绝执行。API 层**永不**提供创建管理员或修改 role 的能力。

#### 实现约束

- `detail_json` 写入前过白名单：`{code_prefix, expires_at, revoked_count, reason, count, cids_prefix}`；非白名单键丢弃并计数。
- 禁止写入：password / password_hash / session token / csrf token / 邀请码明文 / 对话或记忆正文。测试用正则全表扫描（`\$argon2id\$`、24 字符 token 形态）断言零命中。

#### 测试验收

- T-04-A：三类主体各写一条，CHECK 约束行为正确（`user` 缺 id 应失败；`system` 带 id 应失败）。
- T-04-B：bootstrap 事件 `actor_user_id IS NULL` 且 `target_id` 指向真实 admin。
- T-04-C：CLI 无 `--as` 创建第二个管理员被拒（exit≠0，无用户产生）。
- T-04-D：审计表全表扫描无敏感串。

---

### R2-05 路由权限矩阵（运行期注册表口径）

#### 数量口径统一（解决 38/39/25+13 冲突）

以 `main.app.openapi()` + `main.app.routes` 实测（FastAPI 0.141.1 / Starlette 1.6.0，导入前已禁用 `dotenv.load_dotenv`，**未读取 .env**）：

| 口径 | 实测值 | 说明 |
|---|---|---|
| `app.routes` 条目 | **30** | 25 个 `APIRoute` + 4 个 `Route`（`/openapi.json`、`/docs`、`/docs/oauth2-redirect`、`/redoc`）+ 1 个 `_IncludedRouter` 占位 |
| OpenAPI 路径条目 | **36** | 两个路径各挂 2 个方法（`/knowledge/documents`、`/business/orders`） |
| **方法级操作（业务）** | **38** | 25（`main.py`）+ 13（`business/api.py`）→ **这是「38」的来源** |
| 内建文档路由 | **4** | 不在 OpenAPI 中，生产需关闭 |
| 实施后新增 | **+10** | 见下表 39–48 |
| 实施后总操作 | **48** | 38 + 10（不含 4 条内建） |

> 说明：FastAPI 0.141 的 `include_router` 不把子路由展平进 `app.routes`，只留 `_IncludedRouter` 占位；因此「25 + 1」不能直接相加，必须以 OpenAPI 契约（38）为准。R1 报告中的「39 行」是表格行数，非路由数。

#### 权限矩阵（38 条现存 + 10 条新增）

图例：认证 `公`=公开 / `登`=需登录 / `管`=需管理员；归属=是否校验资源所有者；CSRF：`✓` 需 token，`O` 需 Origin 校验；限速：`L`=登录限速、`R`=注册限速、`U`=用户级配额、`-`=无；审计：`A`=写 audit_logs。

| # | 方法 | 路径 | handler | 认证 | 角色 | 归属 | CSRF | 限速 | 审计 | 模型 | 用户数据 | 共享写 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | POST | `/chat` | `chat` | 登 | user/admin | 会话归属 | ✓ | U | - | **是** | 是 | 否 |
| 2 | GET | `/conversations` | `list_conversations` | 登 | 同 | 强制 user_id | - | - | - | 否 | 是 | 否 |
| 3 | GET | `/conversations/{cid}/messages` | `conversation_messages` | 登 | 同 | 是（他人→404） | - | - | - | 否 | 是 | 否 |
| 4 | GET | `/knowledge/search` | `knowledge_search` | 登 | 同 | public+own | - | U | - | 否 | 是 | 否 |
| 5 | POST | `/knowledge/import` | `knowledge_import` | 登 | user→private / admin→public | 是 | ✓ | U | - | 否 | 是 | 是 |
| 6 | POST | `/knowledge/documents` | `knowledge_upload_document` | 登 | 同上 | 是 | ✓ | U | - | 否 | 是 | 是 |
| 7 | GET | `/knowledge/documents` | `knowledge_documents` | 登 | 同 | 是 | - | - | - | 否 | 是 | 否 |
| 8 | DELETE | `/knowledge/documents/{did}` | `knowledge_delete_document` | 登 | 同 | 是 | ✓ | U | - | 否 | 是 | 是 |
| 9 | POST | `/analyze-image` | `analyze_image` | 登 | 同 | 无（不落盘） | ✓ | U | - | **是** | 否 | 否 |
| 10 | GET | `/` | `index` | 登 | 同 | 无 | - | - | - | 否 | 否 | 否 |
| 11 | GET | `/workbench` | `workbench_index` | 登 | 同 | 无 | - | - | - | 否 | 否 | 否 |
| 12 | GET | `/business-ui` | `business_ui_index` | 登 | 同 | 无 | - | - | - | 否 | 否 | 否 |
| 13 | GET | `/app-ui/static/style.css` | `app_ui_style` | 公 | - | 无 | - | - | - | 否 | 否 | 否 |
| 14 | GET | `/app-ui/static/app.js` | `app_ui_script` | 公 | - | 无 | - | - | - | 否 | 否 | 否 |
| 15 | GET | `/workbench/static/style.css` | `workbench_style` | 公 | - | 无 | - | - | - | 否 | 否 | 否 |
| 16 | GET | `/workbench/static/app.js` | `workbench_script` | 公 | - | 无 | - | - | - | 否 | 否 | 否 |
| 17 | GET | `/business-ui/app.js` | `business_ui_script` | 公 | - | 无 | - | - | - | 否 | 否 | 否 |
| 18 | GET | `/business-ui/style.css` | `business_ui_style` | 公 | - | 无 | - | - | - | 否 | 否 | 否 |
| 19 | GET | `/login` | `login_page` | 公 | - | 无 | - | - | - | 否 | 否 | 否 |
| 20 | GET | `/register` | `register_page` | 公 | - | 无 | - | - | - | 否 | 否 | 否 |
| 21 | GET | `/auth/static/style.css` | `auth_style` | 公 | - | 无 | - | - | - | 否 | 否 | 否 |
| 22 | GET | `/auth/static/app.js` | `auth_script` | 公 | - | 无 | - | - | - | 否 | 否 | 否 |
| 23 | GET | `/auth/status` | `auth_status` | 公 | - | 无 | - | - | - | 否 | 否 | 否 |
| 24 | POST | `/auth/login` | `auth_login` | 公 | - | 无 | O | L | - | 否 | 否 | 否 |
| 25 | POST | `/auth/register` | `auth_register` | 公 | - | 无 | O | R | - | 否 | 否 | 否 |
| 26 | POST | `/business/orders` | `create_order_api` | 管 | admin | - | ✓ | - | A | 否 | 否 | **是** |
| 27 | POST | `/business/payments` | `create_payment_api` | 管 | admin | - | ✓ | - | A | 否 | 否 | **是** |
| 28 | POST | `/business/shipments` | `create_shipment_api` | 管 | admin | - | ✓ | - | A | 否 | 否 | **是** |
| 29 | POST | `/business/shipment-events` | `create_shipment_event_api` | 管 | admin | - | ✓ | - | A | 否 | 否 | **是** |
| 30 | POST | `/business/aftersales` | `create_aftersale_api` | 管 | admin | - | ✓ | - | A | 否 | 否 | **是** |
| 31 | POST | `/business/workorders` | `create_workorder_api` | 管 | admin | - | ✓ | - | A | 否 | 否 | **是** |
| 32 | GET | `/business/orders` | `list_orders_api` | 登 | 同 | - | - | - | - | 否 | 否 | 否 |
| 33 | GET | `/business/orders/{oid}` | `get_order_api` | 登 | 同 | - | - | - | - | 否 | 否 | 否 |
| 34 | GET | `/business/orders/{oid}/payments` | `list_order_payments_api` | 登 | 同 | - | - | - | - | 否 | 否 | 否 |
| 35 | GET | `/business/orders/{oid}/shipments` | `list_order_shipments_api` | 登 | 同 | - | - | - | - | 否 | 否 | 否 |
| 36 | GET | `/business/shipments/{sid}/events` | `list_shipment_events_api` | 登 | 同 | - | - | - | - | 否 | 否 | 否 |
| 37 | GET | `/business/orders/{oid}/aftersales` | `list_order_aftersales_api` | 登 | 同 | - | - | - | - | 否 | 否 | 否 |
| 38 | GET | `/business/orders/{oid}/workorders` | `list_order_workorders_api` | 登 | 同 | - | - | - | - | 否 | 否 | 否 |
| 39 | GET | `/auth/me`（新增） | `auth_me` | 登 | 同 | 本人 | - | - | - | 否 | 是（本人） | 否 |
| 40 | POST | `/auth/logout`（新增） | `auth_logout` | 登 | 同 | 本人会话 | ✓ | - | - | 否 | 是 | 否 |
| 41 | GET | `/admin/users`（新增） | `admin_list_users` | 管 | admin | - | - | - | - | 否 | 是（元数据） | 否 |
| 42 | PATCH | `/admin/users/{uid}`（新增） | `admin_patch_user` | 管 | admin | - | ✓ | - | A | 否 | 是 | 否 |
| 43 | POST | `/admin/users/{uid}/revoke-sessions`（新增） | `admin_revoke_sessions` | 管 | admin | - | ✓ | - | A | 否 | 是 | 否 |
| 44 | GET | `/admin/invites`（新增） | `admin_list_invites` | 管 | admin | - | - | - | - | 否 | 是（元数据） | 否 |
| 45 | POST | `/admin/invites`（新增） | `admin_create_invite` | 管 | admin | - | ✓ | - | A | 否 | 是 | 否 |
| 46 | POST | `/admin/invites/{iid}/revoke`（新增） | `admin_revoke_invite` | 管 | admin | - | ✓ | - | A | 否 | 是 | 否 |
| 47 | GET | `/admin/audit`（新增） | `admin_list_audit` | 管 | admin | - | - | - | - | 否 | 是（元数据） | 否 |
| 48 | GET | `/admin/ui`（新增） | `admin_ui` | 管 | admin | - | - | - | - | 否 | 否 | 否 |

**生产环境策略（4 条内建）**

| 路由 | 策略 |
|---|---|
| `/docs`、`/redoc`、`/openapi.json` | 生产 **关闭**（`FastAPI(docs_url=None, redoc_url=None, openapi_url=None)`）；如需保留，仅绑定 127.0.0.1 或加 admin 依赖 |
| `/docs/oauth2-redirect` | 随 `/docs` 一并失效 |

**旧路由遗漏检查**：以 OpenAPI 38 条逐一比对 R2-1 清单，无遗漏、无多余；`/business` 前缀实为 13 条（R1 的「16」已在本轮修正）。

---

### R2-06 RAG KB 缓存与隔离

#### 设计结论（修正 R1 的两个缺陷：无界 LRU、`_KB_LOCKS` 无界增长）

```python
@dataclass
class _Entry:
    kb: KnowledgeBase | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

_PUBLIC_ENTRY = _Entry()                       # public 独立槽，永不参与淘汰
_PRIVATE: "OrderedDict[str, _Entry]" = OrderedDict()
_CACHE_LOCK = threading.Lock()
PRIVATE_MAX = 8

def get_kb(scope: str) -> KnowledgeBase:
    if scope == "public":
        entry = _PUBLIC_ENTRY
    else:
        with _CACHE_LOCK:
            entry = _PRIVATE.get(scope)
            if entry is None:
                entry = _Entry()
                _PRIVATE[scope] = entry
            _PRIVATE.move_to_end(scope)
            _evict_locked()
    with entry.lock:                            # 只锁本 scope
        if entry.kb is None:
            entry.kb = KnowledgeBase(_scope_dir(scope), embedding_factory=_shared_embedding)
    return entry.kb
```

```python
def _evict_locked():
    """调用方必须持有 _CACHE_LOCK。"""
    while len(_PRIVATE) > PRIVATE_MAX:
        scope, entry = next(iter(_PRIVATE.items()))
        if entry.lock.locked():                 # 正在构建 → 跳过，下次再试
            if all(e.lock.locked() for e in _PRIVATE.values()):
                return
            _PRIVATE.move_to_end(scope)
            continue
        _PRIVATE.popitem(last=False)            # 只丢实例，不删磁盘文件
```

| 要求 | 落实方式 |
|---|---|
| public 永不被误删 | 独立字段 `_PUBLIC_ENTRY`，不在 `_PRIVATE` 中，淘汰逻辑不可见 |
| 私有严格按用户隔离 | `scope = "u:" + user_id`，只由 `get_current_user` 生成；`_scope_dir` 前校验 `^[0-9a-f]{32}$`，否则抛错 |
| 并发与淘汰锁定 | dict 元数据用 `_CACHE_LOCK`（短临界区）；实例构建用 `entry.lock`（长临界区）；构建中的 entry 不被淘汰 |
| `_KB_LOCKS` 无界增长 | **删除独立锁字典**，锁随 `_Entry` 一起创建/淘汰，条目数恒等于 scope 数（≤ 9） |
| 缓存未命中重载 | `entry.kb is None` 时重建，`KnowledgeBase._ensure_index` 惰性加载（`rag/knowledge_base.py:246`） |
| Embedding 共享 | `_shared_embedding` 单例 + `embedding_factory=lambda: _shared_embedding`（`:555` 支持注入） |
| 磁盘文件 | 淘汰只丢内存对象，`rag_knowledge/**` 不动 |

#### 配额与容量边界

| 维度 | 上限 | 依据 |
|---|---|---|
| 每用户文件数 | 50 | 管理成本与索引时间 |
| 每用户总字节 | 20 MiB | 单文件已限 10 MiB（`MAX_KNOWLEDGE_FILE_BYTES`） |
| 每 scope chunk | 2000 | 见下估算 |
| 私有缓存实例 | 8 | 2C2G 内存约束 |
| 单向量 | 512 × float32 = 2,048 B | bge-small-zh-v1.5 维度 |
| 单 chunk 开销 | ≈ 2.6–3 KB（向量 2 KB + 文本 ≤300 B + 元数据） | 现有 22 chunk 语料实测可忽略 |
| 8 实例满载 | ≈ 40–50 MB | 2000 chunk × 3 KB × 8 |
| BGE 常驻 | 权重 ≈95 MB，进程 RSS 通常 400–700 MB | 需在 2C2G 实测（待核验项 V-3） |

#### 测试验收

- T-06-A：9 个不同用户依次检索 → 缓存条目数 ≤ 9（public + 8 私有），且被淘汰用户的文件仍在磁盘。
- T-06-B：淘汰后该用户再次检索 → 自动重建，结果与首次一致。
- T-06-C：public 连续检索 100 次 + 20 轮淘汰压力 → `_PUBLIC_ENTRY.kb` 始终非 None。
- T-06-D：并发 8 线程检索同一 scope → `KnowledgeBase` 只构建一次（用构造计数器断言）。
- T-06-E：`_PRIVATE` 大小与 `len(entries)` 有界；进程运行 1 万次不同 scope 访问后 `len(_PRIVATE) <= 8`（锁字典已删除）。
- T-06-F：`scope` 传入非法 user_id（非 32 hex）→ 抛错，不拼路径。

---

### R2-07 模型计量与预算状态机

#### 状态机

```
[reserve] --changes()==1--> reserved --(发送)--> settled(ok=1, tokens 有则写)
    │                          │
    │ changes()==0             ├─(provider 4xx)--> released(ok=0, calls-1)
    │                          ├─(5xx/超时/连接)--> kept(ok=0, calls 保留)
    └─> 429，未发送请求        └─(本地校验失败)--> 不预扣，可选 released 事件
```

#### 预扣 SQL（原子）

```sql
INSERT INTO usage_daily (user_id, day, calls, image_calls, total_tokens)
VALUES (:uid, :day, 1, :is_image, 0)
ON CONFLICT(user_id, day) DO UPDATE SET
    calls = calls + 1,
    image_calls = image_calls + :is_image
WHERE calls < :daily_limit;
-- sqlite3 changes() == 0 → 配额不足 → 429，且绝不进入模型调用分支
```

#### 结算 / 释放 / 保留规则

| 情况 | calls | tokens | `ok` | 说明 |
|---|---|---|---|---|
| 成功且有 provider usage | 已+1 | 累加真实值 | 1 | `usage_source='provider'` |
| 成功但无 usage 字段 | 已+1 | **NULL** | 1 | `usage_source='unavailable'`，**禁止估算/置 0** |
| provider 4xx（参数非法） | **-1 释放** | NULL | 0 | 明确未计费 |
| provider 5xx / 超时 / 连接中断 | 保留 | NULL | 0 | 可能已计费，保守保留 |
| 本地校验失败（未发出请求） | 不预扣 | NULL | 0（可选事件） | 无扣减 |
| 配额不足 | 不预扣 | — | — | 429，无事件 |

释放 SQL：`UPDATE usage_daily SET calls = calls - 1 WHERE user_id=? AND day=? AND calls > 0`（与事件写入同事务）。

#### 覆盖入口

- 5 处 `self.client.chat.completions.create`（`agent.py:877/1503/1629/1784/2866`）→ 由 `MeteredCompletions` 代理统一拦截（Agent 构造时包装 `self.client`）。
- `DeepSeekVisionClient.analyze`（`image_client.py:40`）→ 新增可选 `analyze_detail()`；`understand_image` 用 `hasattr` 探测，旧假客户端仍可用。
- 未来入口：只允许通过 `metering.reserve()` 访问模型，代码评审检查「直接 `client.chat.completions.create`」零出现。

#### 并发与持久化

- 原子性由单条条件 UPSERT + SQLite 单写保证；并发测试 50 线程 / limit=10 → 恰好 10 次 `changes()==1`。
- 计数落库（`usage_daily`），**进程重启不清零**；按 UTC `day` 分桶。
- 全局预算：`SELECT SUM(calls) FROM usage_daily WHERE day=?` 与全局上限比较（索引 `idx_usage_daily_day`）。

#### 测试验收

- T-07-A：limit=10，50 并发 → 10 成功 40 拒绝，`usage_daily.calls=10`。
- T-07-B：无 usage 响应 → `usage_source='unavailable'` 且三 token 列为 NULL（断言 `IS NULL`，不是 0）。
- T-07-C：provider 400 → calls 回到预扣前值。
- T-07-D：超时 → calls 保留（不回退），`ok=0`。
- T-07-E：重启进程（新 Memory 实例读同一 db）→ 当日计数不变。
- T-07-F：配额耗尽时模型客户端**未被调用**（fake client 调用计数为 0）。

---

### R2-08 CSRF、Origin 与 Session 逐路由策略

#### Cookie 属性

| Cookie | HttpOnly | Secure | SameSite | Path | Domain | 内容 |
|---|---|---|---|---|---|---|
| `__Host-pai_sess` | **是** | 是 | Lax | `/` | 无（`__Host-` 要求） | 256-bit 随机 token（DB 只存 sha256） |
| `pai_csrf` | 否（供 JS 读取） | 是 | Lax | `/` | 无 | 256-bit 随机 token（DB 存 sha256，绑定会话） |

#### Origin 规则

| 场景 | Origin 缺失 | Origin 不匹配 | `Sec-Fetch-Site: cross-site` |
|---|---|---|---|
| 登录 / 注册（无会话） | **403**（要求 Origin 或 Referer） | 403 | 403 |
| 已登录写请求 | 允许（以 CSRF token + 会话绑定为主要凭据） | 403（双重校验） | 403 |
| 已登录读请求（GET） | 允许 | **不校验**（无副作用），但响应加 `Cache-Control: no-store` + `Vary: Cookie` | 允许 |

#### 校验流程

```
写请求（POST/PUT/PATCH/DELETE，除 /auth/login、/auth/register）：
  1. session = 查 token_sha256 → 校验 revoked/expires/absolute/user.status
  2. h = headers["X-CSRF-Token"]；c = cookies["pai_csrf"]
  3. 若 h 或 c 缺失 → 403
  4. hmac.compare_digest(h, c)                → 403
  5. hmac.compare_digest(sha256(h).hexdigest(), session.csrf_sha256) → 403
  6. 若 Origin 存在且 ≠ PUBLIC_ORIGIN → 403
```

#### Session 绑定与过期

- `csrf_sha256` 与会话同寿命；登出、改密、禁用、管理员撤销 → 该会话立即失效（CSRF 随之失效）。
- 过期：24h 空闲（`expires_at`）+ 7d 绝对（`absolute_expires_at`），二者任一到期即 401。
- 会话固定防护：登录成功**总是新建** session，不复用请求里带来的任何 token。

#### 路由映射

CSRF/Origin 要求已并入 §6 矩阵的 `CSRF` 列（`✓` = token，`O` = Origin，`-` = 无）。

#### 测试验收

- T-08-A：6 种 CSRF 失效场景全部 403（缺 token / 缺 cookie / 两者不等 / 摘要不符 / 会话已撤销 / 跨源）。
- T-08-B：登录页表单跨源提交 → 403；同源 → 200。
- T-08-C：GET 请求带跨源 Origin 仍 200，但响应含 `Cache-Control: no-store`。
- T-08-D：登出后原 cookie 再请求写接口 → 401（而非 403）。

---

### R2-09 SQLite 备份与恢复（硬门禁）

#### 备份流程与 6 道门禁

| 门禁 | 检查 | 失败动作 |
|---|---|---|
| G1 | `src.backup(dst)` 无异常 | **中止**迁移 |
| G2 | `dst` 文件存在且 `size > 0` | 中止 |
| G3 | 对 `dst` 执行 `PRAGMA integrity_check` == `ok` | 中止 |
| G4 | 对 `dst` 执行 `PRAGMA quick_check` == `ok` | 中止 |
| G5 | 备份中各表行数与源库一致（conversations/messages/memories） | 中止 |
| G6 | 记录 `dst` 的 sha256 与字节数到 `ledger.detail` | 中止（无法留证即不迁移） |

- 备份前先 `PRAGMA wal_checkpoint(TRUNCATE)`（WAL 模式下确保数据落主库）。
- 连接策略（两个库统一）：`foreign_keys=ON`、`busy_timeout=5000`、`journal_mode=WAL`、`synchronous=NORMAL`。
- **原库在 G1–G6 全部通过前不得进入任何 DDL/DML**。

#### 恢复步骤

1. 停止服务（门禁会阻止带病启动）。
2. `cp memory.db.bak-<utc> memory.db`（覆盖）；同时删除残留 `-wal/-shm`。
3. 对恢复后的库执行 `PRAGMA integrity_check` 与 `PRAGMA foreign_keys` 检查。
4. 核对三表行数 = 基线（2/53/5）。
5. 删除/标记 `schema_migrations` 中失败版本为 `failed`，保留证据。
6. 重启前先 `migrate status` 确认状态，再决定重跑或放弃。

#### 外键与完整性

- 每个新连接执行 `PRAGMA foreign_keys = ON` 并**断言返回 1**；测试用「插入不存在 user_id 的 session 必须失败」验证真实生效。
- `PRAGMA foreign_key_check` 在迁移前后各跑一次（`users` 与 `sessions/invites` 之间已建物理外键）。

#### 测试验收

- T-09-A：模拟 G3 失败（对损坏文件）→ 迁移零 DDL 执行。
- T-09-B：备份→恢复→`integrity_check`=ok→行数一致。
- T-09-C：`foreign_keys` 返回 1 且违规插入失败。
- T-09-D：备份哈希写入 ledger 后可复算一致。

---

### R2-10 生产依赖与测试依赖

#### 划分依据（源码 AST import 提取，非「已装 89 包」）

| 类别 | 包 | 版本约束（建议） | 说明 |
|---|---|---|---|
| 生产 | `fastapi` | `>=0.141,<0.142` | Web 框架 |
| 生产 | `pydantic` | `>=2.13,<3` | 请求模型 |
| 生产 | `python-dotenv` | `>=1.2,<2` | 配置加载 |
| 生产 | `openai` | `>=3.19,<4` | 模型 SDK |
| 生产 | `sentence-transformers` | `>=6.1,<7` | 本地 BGE |
| 生产 | `pypdf` | `==6.16.2` | PDF 解析 |
| 生产 | `python-docx` | `==1.2.0` | DOCX 解析 |
| 生产 | `pillow` | 未在 requirements 中声明（**缺口**） | `PIL` 用于图片格式识别 |
| 生产（隐式） | `python-multipart` | `>=0.0.32` | `File()`/`Form()` 必需，源码无 import |
| 生产（新增） | `argon2-cffi` | `>=23.1,<26`（待核验 V-2） | 密码哈希 |
| 测试 | `pytest` | `>=9,<10` | |
| 测试 | `httpx` | `>=0.28,<0.29` | TestClient 依赖 |
| 测试 | `starlette` | 由 fastapi 传递，测试直连 | TestClient 类型 |
| 测试 | `uvicorn` | `>=0.52,<0.53` | 启动服务跑集成测试 |
| 测试 | `numpy` | `>=2.5,<3` | 向量断言 |

**明确排除**（源码零 import，属环境残留）：`edge-tts 7.3.1`、`pywebview 6.2.1`、`zai-sdk 0.2.3`。

#### 安装方案

```
# 生产
"<py312>" -m pip install -r requirements.txt
# 测试（在生产基础上叠加）
"<py312>" -m pip install -r requirements-test.txt

# 锁定（干净 venv 中验证后生成）
"<py312>" -m venv .venv-lock
.venv-lock\Scripts\python -m pip install -r requirements.txt -r requirements-test.txt
.venv-lock\Scripts\python -m pip freeze > requirements.lock.txt
```

- `requirements.lock.txt` 保留精确 `==` 版本，可选 `--require-hashes`。
- **不使用 `--no-deps` 作为常规安装方式**（仅诊断冲突时临时使用）。
- 安装前 `pip install --dry-run -r requirements.lock.txt` 对比，任何无关包版本漂移须人工确认。
- 本轮**不执行**任何安装；`pillow` 与 `python-multipart` 缺失声明属既有缺口，实施方案需补入 `requirements.txt`。

#### 测试验收

- T-10-A：干净 venv 仅装 `requirements.txt` → `import main` 成功、`/auth/status` 可响应（不需要模型 Key）。
- T-10-B：安装后 `pip freeze` 与 lock 完全一致。
- T-10-C：`requirements.txt` 中不出现环境残留包。

---

## 4. 关键数据流与状态转换

### 4.1 请求数据流

```
HTTP → Caddy → uvicorn(1 worker)
  ├─ 公开路由：直接返回静态/页面
  └─ 受保护路由
       ├─ Depends(get_current_user)：cookie → sha256 → sessions JOIN users
       │     校验 revoked_at IS NULL / expires_at > now / absolute_expires_at > now / status='active'
       │     → 返回 CurrentUser（纯参数，不写 ContextVar）
       ├─ Depends(require_csrf)：写请求 token 双提交 + 会话摘要比对
       ├─ 端点 body 首行：with runtime_context(RuntimeContext(uid, rid, scopes))
       │     ├─ agent.run(msg, cid, user_id=uid)
       │     │    ├─ memory.*(..., user_id=uid)  → memory.db
       │     │    ├─ execute_tool → tool → current_context().user_id
       │     │    └─ MeteredCompletions → reserve → provider → settle/release → usage_*
       │     └─ get_kb("u:"+uid) / get_kb("public")
       └─ finally: ContextVar.reset + 断言 _CURRENT.get() is None
```

### 4.2 迁移状态机

见 R2-03（pending → running → succeeded / failed，双连接独立事务，启动门禁）。

### 4.3 计量状态机

见 R2-07（reserve → settled / released / kept）。

### 4.4 KB 缓存状态

```
未命中 → 建 _Entry（锁）→ 构建 KnowledgeBase（首次 13s）
     → 命中 → move_to_end
文件变更 → invalidate(scope) → 移除 entry（含锁）→ 下次重建
超过 8 → 淘汰最久未用的非构建中 entry（public 不参与）
```

---

## 5. 数据库迁移与失败恢复方案

| # | 步骤 | 事务边界 | 幂等 | 失败处理 |
|---|---|---|---|---|
| 0 | `wal_checkpoint(TRUNCATE)` + 备份 + G1–G6 | 文件级 | 可重复（带时间戳） | 中止，ledger 不写 |
| 1 | 只读预检（N1/N2/N3、`user_version`、`foreign_key_check`） | 只读 | — | 中止 |
| 2 | 建 `schema_migrations`；写 `0001 pending→running`（连接 L） | L 独立事务 | `IF NOT EXISTS` | 标记 failed |
| 3 | 建 `users/sessions/invites` + 索引（连接 M）→ L 置 succeeded | M 一个事务 | `IF NOT EXISTS` | M 回滚 → L=failed |
| 4 | CLI 创建首个 admin + bootstrap 审计 | M 一个事务 | `username_key` 唯一 | 回滚，无 admin |
| 5 | 加 `conversations.user_id`、`memories.user_id`（`PRAGMA table_info` 预检） | 每表一个事务 | 列存在则跳过 | L=failed |
| 6a | （仅授权时）孤儿隔离：插入 NULL-owner 占位会话 | M 一个事务 | 以 `title LIKE '[quarantined] %'` 判重 | L=failed |
| 6b | 回填：memories → conversations | 每表一个事务 | 只填 NULL | L=failed |
| 6c | 建索引 + `trg_conversations_owner_required` | M 一个事务 | 先 `DROP TRIGGER IF EXISTS` | L=failed |
| 7 | 建 `usage_events/usage_daily/audit_logs` + 索引 | M 一个事务 | `IF NOT EXISTS` | L=failed |
| 8 | 代码改造发布（Memory / RAG / 鉴权依赖） | 代码 | 可回退旧代码 | 回退即恢复 |
| 9 | RAG 目录迁移 `public/` + `users/<uid>/` | 文件 mv | 已存在则跳过 | 可逆（未删除） |
| 10 | 核对 §R2-02 的 5 条 SQL + 内容量守恒 | 只读 | — | 失败 → 备份恢复 |
| 11 | 关闭 `/docs` 等内建路由 | 代码 | — | 回退配置 |

**回滚优先级**：① 回退应用代码（列可空，旧代码不读）；② 撤销隔离/回填（UPDATE 置 NULL）；③ 备份整库覆盖（最后手段）。

---

## 6. 完整路由权限矩阵及实际数量

见 §R2-05。**口径结论**：现存方法级操作 **38**（25 + 13）；内建文档路由 **4**；实施后新增 **10**；实施后总操作 **48**。`app.routes` 条目为 30（含 `_IncludedRouter` 占位），OpenAPI 路径条目为 36——三者不可混用。

---

## 7. 测试验收矩阵

| # | 修订 | 测试 ID | 关键断言 | 通过阈值 |
|---|---|---|---|---|
| 1 | R2-01 | T-01-A/B/C/D/E | 双用户并发不串；依赖内 set 无效；请求后无残留；嵌套抛错；后台任务禁止隐式继承 | 全绿 |
| 2 | R2-02 | T-02-A/B/C/D | 孤儿中止且不删；隔离不可见；可逆；53 条历史消息 (id,length) 一致 | 全绿 |
| 3 | R2-03 | T-03-A/B/C | 失败状态可持久化；running 阻断启动；resume 幂等 | 全绿 |
| 4 | R2-04 | T-04-A/B/C/D | 三类 actor CHECK 正确；bootstrap NULL actor；无 --as 拒绝；审计无敏感串 | 全绿 |
| 5 | R2-05 | T-05-A/B | 48 条路由逐条状态码断言；OpenAPI 操作数 = 48 | 全绿，数量一致 |
| 6 | R2-06 | T-06-A~F | 缓存 ≤9；淘汰不删文件；public 常驻；并发只建一次；锁不增长 | 全绿 |
| 7 | R2-07 | T-07-A~F | 50 并发限 10 恰好 10；无 usage 为 NULL；4xx 释放；超时保留；重启不清零；超限不发请求 | 全绿 |
| 8 | R2-08 | T-08-A~D | 6 种 CSRF 失效 403；跨源登录 403；GET 带 no-store；登出后 401 | 全绿 |
| 9 | R2-09 | T-09-A~D | 门禁中止零 DDL；备份可恢复；外键生效；哈希可复算 | 全绿 |
| 10 | R2-10 | T-10-A~C | 干净 venv 可运行；lock 一致；无残留包 | 全绿 |
| 11 | 回归 | T-R-A/B | 现有 35 项 UI 测试 + RAG/记忆/端口测试全绿；`/`、`/workbench` 响应字节不变 | 全绿 |

---

## 8. 尚未解决的问题与待核验证据

| 编号 | 问题 | 性质 | 需要什么证据 |
|---|---|---|---|
| V-1 | Argon2id 在 2C2G 的 p95 与 RSS 未知 | **待核验**（非阻断设计，阻断**上线**） | 目标机基准脚本输出（参数网格 19/32/46/64 MiB × t=2/3 × p=1/2） |
| V-2 | `argon2-cffi` 在离线/内网环境的 wheel 可得性未验证 | 待核验 | `pip download --only-binary=:all:` 的产物清单 |
| V-3 | BGE 首次索引与常驻 RSS 在 2C2G 的峰值 | 待核验 | 目标机 `GetProcessMemoryInfo`/`psutil` 采样 |
| V-4 | FastAPI 0.141 `_IncludedRouter` 的展平时机 | 已观测（不展平），需在锁定版本后复核 | 锁定版本后重跑注册表枚举脚本 |
| V-5 | WAL 与现有 `conftest.py` 指纹哨兵的兼容性 | 待核验 | 跑一次全量测试，确认 `-wal/-shm` 被快照覆盖 |
| V-6 | 时间格式不一致（旧表本地时间 vs 新表 UTC） | 已知技术债 | 制定「新表只存 UTC、不跨表比较」的代码评审规则 |
| V-7 | 单 Worker 假设下的限速/KB 缓存进程内状态 | 已知限制 | 若改多 Worker，需引入共享存储（本轮不设计） |
| V-8 | `pillow`、`python-multipart` 未在依赖声明中 | 既有缺口 | 实施轮补入 `requirements.txt` |

**P0 阻断项统计：0**（11 项修订均有结论与验收标准；V-1/V-2/V-3 属上线前置实测，不阻断设计定稿与编码，但阻断生产部署）。

---

## 9. 文件修改清单及修改前后指纹

### 9.1 本轮唯一新增文件

| 文件 | 修改前 | 修改后 |
|---|---|---|
| `docs/PAI-AUTH-R2-1-R2_设计修订报告.md` | 不存在 | 见 §9.3 指纹 |

### 9.2 代码/配置/数据库：前后一致（零改动）

| 文件 | 修改前 SHA-256 | 修改后 |
|---|---|---|
| `main.py` | `a8397b54…6f0b96` | 一致 |
| `memory.py` | `8a79e394…2e5f8a` | 一致 |
| `agent.py` | `b8a8e32f…74804e` | 一致 |
| `knowledge_store.py` | `5d46ff6a…63acfc5` | 一致 |
| `tools.py` | `bee0341b…29736` | 一致 |
| `business/api.py` | `6385020c…c9a75` | 一致 |
| `business/db.py` | `4afa0fac…59693` | 一致 |
| `conftest.py` | `5fde88c5…74c83` | 一致 |
| `memory.db` | `dbe0d2a4…64653d`（98,304 B / mtime 2026-11-07 18:40:03） | 一致 |
| `data/business.db` | `678a9b87…c775fc`（73,728 B / mtime 2026-11-07 17:57:07） | 一致 |

### 9.3 本轮结束时的实际指纹

本节数据于本文档落盘后重新实测取得（`sha256sum` / `stat` / `git status --porcelain`）。

**新增文件（本文档自身）**

| 文件 | size | 行数 | SHA-256 | mtime |
|---|---|---|---|---|
| `docs/PAI-AUTH-R2-1-R2_设计修订报告.md` | 50,075 B | 874 | `e747bbceb91cf71be5a46969c513e70d1b9bbef390575e08df00cc5489c9932d` ① | 2026-11-09 15:09:23 +0800 |

> ① 自引用说明：文档内嵌自身 SHA-256 时，写入该值这一动作本身就会改变文件哈希，因此任何内嵌值都必然对应「写入前一刻」的版本，不可能与最终文件逐位相同。上表 size / 行数 / mtime 与 SHA 均对应同一版本（即写入本脚注之前的版本）。**定稿版本的实测 SHA-256 记录在对话交付回执中**，复核命令：
> `sha256sum docs/PAI-AUTH-R2-1-R2_设计修订报告.md`
> 该自引用偏差仅影响本节这一行，不影响其余 10 项指纹（源码 8 项 + 数据库 2 项）的可核验性 —— 它们均可用 `sha256sum` 独立复核并与 §9.2 逐位比对。

**代码（8 项，全文 SHA-256）**

| 文件 | SHA-256（实测） | 与开工基线 |
|---|---|---|
| `main.py` | `a8397b5433a4e680ab75479c03ebdba5fbf735e33a11154fd6cdedbeaa6f0b96` | 逐位一致 |
| `memory.py` | `8a79e394607022e5092fbc19d520c7965c57b8da11cb34032f2b8ccd4a2e5f8a` | 逐位一致 |
| `agent.py` | `b8a8e32f20669106c662b3d0b79c07b872e335d817a1bf7afe6ae07ad374804e` | 逐位一致 |
| `knowledge_store.py` | `5d46ff6abf9189fa95a00e7b7076e10bebae135220ffd95d588ffb22f63acfc5` | 逐位一致 |
| `tools.py` | `bee0341bad95af5bba9e520753363a22605bc65d4bd5bd9769dfb852ba429736` | 逐位一致 |
| `business/api.py` | `6385020cdf6fc0a94791779c12183f569cec0a53a3f326fff691c29e4dbc9a75` | 逐位一致 |
| `business/db.py` | `4afa0fac166160aea9c22b5c734e69e9152300a6b7c27221df80821813859693` | 逐位一致 |
| `conftest.py` | `5fde88c5e28dc1422ad9b4fb322da3f3756dea3a7848084c8aa9d9570ef74c83` | 逐位一致 |

**数据库（2 项）**

| 文件 | size | mtime | SHA-256 | 与开工基线 |
|---|---|---|---|---|
| `memory.db` | 98,304 B | 2026-11-07 18:40:03.071817100 +0800 | `dbe0d2a497b022361aaf7adf7f956519e44c204e1b03e647cb2c41259d64653d` | 逐位一致 |
| `data/business.db` | 73,728 B | 2026-11-07 17:57:07.405455100 +0800 | `678a9b871bf7403f743cde9353901449cb518071e418d3cb61691c1b35c775fc` | 逐位一致 |

**Git**

| 指标 | 开工基线 | 结束时 | 差值 |
|---|---|---|---|
| HEAD | `81a8f2d` | `81a8f2d` | 未变 |
| 工作区条目 | 58（25 M/D + 33 ??） | 59（25 M/D + 34 ??） | **+1（仅本文档）** |
| `stash` 条目 | 0 | 0 | 未变 |
| `docs/` 目录文件数 | 20 | 21 | +1（仅本文档） |

新增的唯一 `??` 条目即 `docs/PAI-AUTH-R2-1-R2_设计修订报告.md`；25 项已跟踪的 `M`/`D` 集合与开工时集合逐条相同，未新增、未消失。

---

## 10. 本轮未修改代码、数据库、依赖和配置的声明

- 未修改任何 Python 源码、测试、前端资源或配置。
- 未执行任何 DDL/DML；两个 SQLite 文件大小、mtime、SHA-256 与开工时逐位一致。
- 未安装、升级或卸载任何依赖。
- 未调用任何真实模型 API；未启动或停止任何服务。
- 未读取、未修改 `.env`（路由枚举时已把 `dotenv.load_dotenv` 替换为空实现，确保不加载该文件）。
- 未执行任何 Git 写操作（commit/reset/checkout/clean/stash 均未执行）。
- 唯一写入为新增本文档（`docs/PAI-AUTH-R2-1-R2_设计修订报告.md`），未覆盖、未删除任何既有文件。
- 未输出任何密钥、密码、Token、邀请码或对话/记忆正文。
