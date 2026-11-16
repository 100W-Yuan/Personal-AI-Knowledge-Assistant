# P0-003 多 Agent 系统架构设计说明书

| 项 | 内容 |
|---|---|
| 文档编号 | P0-003 |
| 版本 | **V2.1** |
| 日期 | 2026-11-04 |
| 状态 | 草案（待负责人批准；**上一版唯一阻塞项"`revision_required` 表示法"已由 D-07 关闭**） |
| 本版性质 | V2.0 为首次落盘；**V2.1 为 P0-008 定点修订**（D-06 异常归并权责、D-07 修订标记落定、D-08 W1 暂缓与同步机制待设计） |
| 不可变更项 | 6 角色划分、9 态任务状态机、"LLM 规划 + 确定性执行器调度"分工、`cancel_requested` 为非状态标记、`partial_success` 为不可逆终态 —— 本版**均未修改**；且 **D-07 明确不新增第 10 个状态** |

---

## 1. 架构总原则

1. **真的多 Agent**：6 个角色各自持有独立工具集与独立权限，不存在"一个 Agent 调所有工具"。
2. **规划与调度分离**：Coordinator 用 LLM 产出**计划**，由**确定性执行器**按 DAG 调度；LLM 不直接决定"下一步调谁"，只产出可被校验的计划。
3. **权限用"能不能拿到数据"实现，不用"提示词里说别拿"实现**：数据权限靠 Port 层不暴露接口，不靠 Agent 自律。
4. **一切动作止于草稿**：V1 无写业务系统的通道，架构上不提供该 Adapter。
5. **所有时间判定必须可注入**：见 §9。

---

## 2. 六角色：职责、工具、权限

| 角色 | 职责 | 可用工具（V1） | 数据权限 | 产出 |
|---|---|---|---|---|
| **Coordinator** | 拆解任务、产出计划 DAG、按状态机驱动、**归并与汇总异常结果**、对外应答 | 无业务工具；仅计划校验器 + 状态机驱动器 | **数据读取：全禁**（不得读订单/物流/售后任何字段）；调度权限：允许 | 计划、任务状态、最终响应。**不得创造异常事实（D-06-6）** |
| **Knowledge Agent** | 检索 SOP/政策，返回片段与来源 | `kb_search`（唯一入口 `KnowledgeBase.search`） | 只读知识库；**无订单数据权限** | 引用片段 + 来源 + 命中态 |
| **Data Agent** | 经 DataPort 取只读聚合视图 | `order_view.get`（只读） | **只读**订单聚合视图；**无写权限** | 订单聚合快照 + `as_of` |
| **Analysis Agent** | 判定异常类型、按 D-06 定主次、结论五要素 | 纯计算，无 IO 工具 | **不直接取数**，只消费 Data/Knowledge 的返回 | **异常记录唯一签发者**（D-06-6） |
| **Service Agent** | 生成方案草稿、客户回复草稿、修订稿（新 `draft_id`） | `draft.create` / `draft.supersede` | **只写草稿**；无业务写权限；**只读**引用所需上下文 | 草稿（含 `draft_id` / 版本 / `parent_draft_id`） |
| **Human Approval** | 审批（批准/驳回/要求修改）、阶段推进（`approval_phase`）、超时标记 | 审批动作接口 | 只读被审草稿与依据；不可读原始数据表 | 审批记录、已批准方案记录 |

### 2.1 四条不可让渡的隔离

1. **Coordinator 不得读数据**：它只看到"任务成功/失败 + 结构化摘要"，看不到原始订单字段。防止编排层成为越权取数的后门。
2. **Analysis Agent 不得取数**：它只能基于别的 Agent 给的事实做判定，避免"自己取数自己判"导致依据不可追溯。
3. **Service Agent 不得写业务系统**：V1 架构中根本不存在业务写 Adapter（不是"禁用"，是"没有"）。
4. **签发权与归并权分离（D-06-6）**：异常事实只能由 Analysis Agent **签发**；Coordinator 只能**归并、排序、汇总**。Coordinator 若"顺手补一条异常"，即越权并破坏可追溯性。

### 2.2 异常归并的权责边界（D-06 落地）

| 动作 | 谁做 | 可否创造新异常事实 |
|---|---|---|
| 判定某订单命中 EX-0x | Analysis Agent | **可以**（唯一） |
| 从多条命中里选 1 条为主 | Analysis Agent（按 D-06 优先级） | 否 |
| 保留其余为关联异常 | Analysis Agent | 否（是保留，不是新建） |
| 按 `is_primary` 排序、拼装响应 | Coordinator | **否** |
| 去重（同周期同类型） | Coordinator（按幂等键） | **否** |
| 补一条"看起来也像异常"的记录 | —— | **任何人禁止** |

### 2.2 旧模块定位（沿用 V1.1，未变）

| 模块 | 定位 |
|---|---|
| `agent.py`（单体 2505 行） | **不得**充当 V1 编排层 |
| `task.py`（Task/ControlState/AgentState） | **不得**充当 V1 状态机 |
| `memory.py`（个人记忆） | **不作为 V1 默认依赖**；个人记忆**不注入**子 Agent |
| `rag/`（8 模块） | **候选复用**，须过"质量门 + 来源传递门"两道验证 |

---

## 3. 数据模型原则（D-03 方案 C）

### 3.1 两层结构

```
┌─────────────────────────────────────────┐
│  独立业务实体（唯一真相源）               │
│  Order / Shipment / TraceEvent /         │
│  AfterSale / ApprovalRequest / Draft     │
└───────────────┬─────────────────────────┘
                │ 投影（只读、无独立事实）
┌───────────────▼─────────────────────────┐
│  订单聚合视图 OrderAggregate（只读）      │
│  按 as_of 生成，供 Agent 消费             │
└───────────────┬─────────────────────────┘
                │ 唯一入口
┌───────────────▼─────────────────────────┐
│  DataPort（只读查询接口）                 │
└─────────────────────────────────────────┘
```

### 3.2 四条约束（D-03 原文落地）

| # | 约束 | 落地方式 |
|---|---|---|
| 1 | 聚合视图只读 | DataPort 不提供任何写方法；视图对象不可变（构造后冻结） |
| 2 | 聚合视图不产生独立业务事实 | 视图中的每个字段必须能追溯到某个实体的某个字段；**禁止**在视图层计算并落盘"派生状态"（如"疑似异常"），派生结论只存在于 Analysis Agent 的产出中 |
| 3 | Agent 不直接访问底层存储实体 | 仅 Data Agent 持有 DataPort 句柄；其他 Agent 不得 import 存储层 |
| 4 | DataPort 提供聚合查询 | 对外只有一个查询入口：`get_order_view(order_id, as_of)` |

### 3.3 本轮明确不做（D-03 尾部）

表结构、索引、API 字段、JSON Schema 契约 —— **全部属 P2**。本文件不定义，下游文档也不得定义。

### 3.4 本地模拟数据形态

- 只读文件（JSON/JSONL），不入库、不建表；
- 时间字段使用固定基准日期 + 相对偏移表达；
- 支持"按 `as_of` 回溯"，即同一订单在不同 `as_of` 下可返回不同视图（这是 T1/T2/T3 可测的前提）。

---

## 4. DataPort 访问边界

| 方法 | 调用者 | 权限 | 说明 |
|---|---|---|---|
| `get_order_view(order_id, as_of)` | **仅 Data Agent** | 只读 | 唯一数据入口 |
| `list_orders(filter)` | 仅 Data Agent（V1 不启用） | 只读 | 批量巡检属 Should，V1 关闭 |
| ~~`write_*` / `update_*`~~ | — | **不存在** | 架构上不提供 |

**越权检测**：任何 Agent 绕过 DataPort 直接读文件/读库，视为架构违规，须在测试中断言"非 Data Agent 无法获得数据句柄"。

---

## 5. Coordinator 权限两维（沿用 V1.1）

| 维度 | 权限 | 理由 |
|---|---|---|
| **数据权限** | **全禁** | 编排层若可读数据，则"最小权限"形同虚设 |
| **调度权限** | 允许（派发/取消/归并/终止） | 调度不需要看到数据内容 |

对应到状态机：Coordinator 可改**任务状态**，不可改**业务字段**。

---

## 6. Agent 消息协议（摘要）

| 字段 | 说明 |
|---|---|
| `task_id` | 任务唯一标识 |
| `parent_task_id` | 父任务（子任务必填） |
| `source_agent` / `target_agent` | 角色枚举 |
| `message_type` | `request` / `response` / `error` / `cancel` |
| `payload` | 结构化载荷（不含原始 PII） |
| `status` | `ok` / `partial` / `failed` |
| `created_at` | 由**注入时钟**提供（见 §9） |
| `correlation_id` | 幂等与链路追踪键 |

**禁止**：子 Agent 之间横向直连（必须经 Coordinator 中转），以保证 DAG 可审计。

---

## 7. 任务状态机（9 态，未修改）

```
pending → assigned → running ─┬→ completed
                              ├→ partial_success（不可逆终态）
                              ├→ failed
                              ├→ cancelled
                              ├→ waiting_for_agent
                              └→ waiting_for_approval
```

- `cancel_requested`：**运行期标记，不是状态**；被置位后由执行器在安全点收敛为 `cancelled`。
- `partial_success`：**不可逆终态**，不可再回到 `running`。
- 并行分支的父任务：全部子任务成功→`completed`；部分成功→`partial_success`；全失败→`failed`。

### 7.1 审批阶段是字段，不是状态（D-07）

> **D-07 已确认：`revision_required` 不作为独立任务状态存在。** 状态机**保持 9 态**，第 10 个状态**不予引入**。

| 层 | 承载物 | 取值 | 说明 |
|---|---|---|---|
| 状态（`status`） | 任务生命周期 | 9 态不变 | `waiting_for_approval` 覆盖整个"等人"区间 |
| 阶段（`approval_phase`） | `waiting_for_approval` **内部**的子阶段 | `waiting_decision` / `waiting_revision` / `exhausted` | 只在 `status = waiting_for_approval` 时有效 |

**职责切分（避免语义重叠）**：

| 问题 | 答在哪个字段 |
|---|---|
| 任务是不是在等人？ | `status == waiting_for_approval` |
| 等谁？ | `approval_phase`：等人批（`waiting_decision`）/ 等人改（`waiting_revision`）/ 只能批或驳（`exhausted`） |
| 改过几次？ | `revision_count`（0/1） |
| 是否已超时？ | `approval_expired`（bool，**不是状态**） |

**不变量（可断言）**：

1. `approval_phase` 非空 ⇒ `status == waiting_for_approval`；反向不成立（历史任务可能无该字段）。
2. `revision_count == 0` ⇒ `approval_phase ∈ {waiting_decision, exhausted}`；`exhausted` 仅在 `revision_count == 1` 后可达。
3. 任何路径下 `status` **不得**出现 `revision_required` 取值。
4. `approval_expired == true` 不改变 `status`，也不自动推进 `approval_phase`。

---

## 8. 审批流程（D-07 已确认：marker 方案）

> **本节为已决议设计，不再是候选方案。** D-07 明确：不新增独立任务状态 `revision_required`；`status` 保持 `waiting_for_approval`，用 `approval_phase` 表达审批内部阶段。

### 8.1 审批全路径（6 条，含修订回路）

```
                       status = waiting_for_approval
                    ┌───────────────────────────────┐
   draft 提交 ──────▶│ approval_phase=waiting_decision│
                    └───┬───────┬────────┬──────────┘
             批准 │      │       │        │ 超时
                  ▼      │       │        ▼
        已批准方案记录    │       │   approval_expired=true
        status=completed │       │   status 不变、不自动推进
        （不执行业务）    │       │   → 须重新发起审批
                         │       │
            驳回 │       │       │ 要求修改
                 ▼       │       ▼
          status=failed  │  approval_phase=waiting_revision
          （终态）        │  revision_count=1，原稿 superseded
                         │       ├─ 提交人修订 → 新 draft_id + v2
                         │       │   → 重新审批（phase 回 waiting_decision，不继承旧结论）
                         │       └─ 再次要求修改 → 拒绝
                         │           approval_phase=exhausted，仅剩批准/驳回
                         ▼
                    （终止路径）
```

### 8.2 已采纳的表示（不是候选）

| 字段 | 取值 | 语义（D-07 原文） |
|---|---|---|
| `status` | 恒为 `waiting_for_approval` | 不改状态机；**不出现 `revision_required`** |
| `approval_phase` | `waiting_decision` | 等待审批人批准或驳回 |
| | `waiting_revision` | 审批人要求修改，等待生成新版本草稿 |
| | `exhausted` | 修改次数已达上限，不允许再次要求修改 |
| `revision_count` | 0 / 1 | 上限常量 1（**D-07-2：最多一次**） |
| `approval_expired` | bool | 超时标记，非状态 |

**可观测性**：所有状态查询必须同时返回 `approval_phase`；看板与审计按 `(status, approval_phase)` 二元组统计。

### 8.3 决策溯源（仅存档，不再作为选项）

| 曾评估的方案 | 结论 | 否决/采纳理由 |
|---|---|---|
| 独立状态 `revision_required`（第 10 态） | **已否决（D-07）** | 会扩展已确认的 9 态状态机；且需为"修订等待"另立一套超时策略，而审批 TTL 本身尚未确定 |
| `waiting_for_approval` + `approval_phase` 标记 | **已采纳** | 状态机零改动；超时策略可共用一条通道；未来如确需拆分，标记→状态是纯增量演进 |

> 本节保留仅为审计可追溯。**后续设计不得再引入 `revision_required` 状态**；如需重启该议题，须由负责人另立决策。

### 8.4 与 US-13 / AC-S7 的对应（D-07-10 要求的一致性检查）

| D-07 约束 | 落点 | 一致性 |
|---|---|---|
| 1 不新增独立状态 | §7.1、§8.2 | ✅ 9 态不变，`status` 无新取值 |
| 2 最多修改一次 | `revision_count` 上限 1 | ✅ 与 P0-004 AC-S7 第 4 点一致 |
| 3 原草稿保留 | `superseded`（非删除） | ✅ 与 AC-S7 第 2 点一致 |
| 4 新 `draft_id` 和版本号 | `parent_draft_id` + `version` | ✅ 与 AC-S7 第 3 点一致 |
| 5 必须重新审批 | `waiting_revision` → 新稿 → `waiting_decision` | ✅ 图中回路与 AC-S7 第 5 点一致 |
| 6 不继承旧结论 | 新 `approval_request_id`（`round=2`） | ✅ 与 AC-S7 第 5 点一致 |
| 7 通过≠执行 | 只写"已批准方案记录" | ✅ 与 AC-S6/AC-S7 红线一致 |
| 8 重复请求幂等 | `correlation_id` 去重 | ✅ 与 AC-S7 第 8 点、AC-S8 一致 |
| 9 失败不产生空稿 | 无占位稿，任务 `failed` | ✅ 与 AC-S7 第 9 点一致 |
| 10 一致性 | 本节 + §7.1 | ✅ 已逐条比对 |

### 8.5 边界（不得越界）

**D-07 明确**：具体字段结构、事件字段、持久化结构**仍属 P2 设计阶段**。
本文件只定义**语义与不变量**，不定义：表结构、索引、API 字段名与类型、事件 payload 的具体 schema、状态存储形态。

---

## 9. 注入时钟约束（本轮新增，建议纳入硬约束）

**规则**：所有与时间相关的判定（T1/T2/T3、TTL、超时、`as_of` 回溯）**必须接收外部注入的 `now`**，业务逻辑内**禁止**直接调用系统时钟（`datetime.now()` / `time.time()` / `System.currentTimeMillis()` 等）。

**理由**：

1. T1=72h / T2=48h / T3=72h 已确认，边界测试（恰好 72h、72h±1s）无法靠真实等待完成；
2. 审批 TTL（建议 24h）同理；
3. 无注入时钟则测试必然依赖真实时间 → 不可重复、CI 不稳定。

**配套约定**：全系统统一边界口径 —— **`elapsed >= T` 判为命中**（含相等时刻）。该口径为**建议**，待负责人确认。

**落地要求**：`ClockPort` 作为与 `DataPort` 并列的基础设施端口，由调用方注入；测试注入固定时钟。

---

## 10. 故障与降级（补充 D-01 / D-08 影响）

| 故障 | 降级行为 | 禁止行为 |
|---|---|---|
| 知识库无命中 | `no_hit`，明确声明"未检索到依据"，方案标注"无依据支持" | 不得编造依据 |
| 知识库不可用 | `unavailable`，任务可 `partial_success` | 不得静默当作 no_hit |
| 订单数据缺失 | `data_missing`，声明缺哪个实体/字段 | **不得判为状态矛盾**（D-01 的直接推论） |
| 数据陈旧（stale） | 标 `data_stale`，不参与矛盾判定 | 不得用 stale 字段判 EX-04 |
| **同步状态不可判定** | **标 `pending_verification`，转待核实/人工路径** | **不得默认当作矛盾**（D-08-4） |
| 子 Agent 超时 | 子任务 `failed`，可重试 1 次 | 不得无限重试 |
| Service 不可用 | 不生成空草稿/占位草稿，任务 `failed` | 不得产出占位稿 |

### 10.1 数据同步四态（D-08：W1 defer）

> **W1 数值未确定。** D-08 明确：W1 未定前，**不得**把不同步的时间差直接当作 EX-04；本文件**不写入 W1 的任何具体数值**（含任何分钟级或小时级示例值）。

**必须区分的四类情形**：

| 情形 | 判定依据（不依赖 W1） | 归类 | 是否可判 EX-04 |
|---|---|---|---|
| **数据缺失** | 实体不存在 / 关键字段为 null | `data_missing` | **否** |
| **数据陈旧** | 字段 `updated_at` 明显早于 `as_of`（"明显"的量化=W1，**未定**） | `data_stale` | **否** |
| **数据同步中** | 存在同步动作在进行（状态已变更、下游未反映，且有同步信号） | `sync_in_progress` | **否** |
| **真实状态矛盾** | 三方均有值、无同步信号、组合落在 C-01~C-06 | `conflict` | 是 |
| **无法判定**（兜底） | 有值差但无足够信息判断是延迟还是矛盾 | `pending_verification` | **否**（转人工） |

**安全降级路径（不依赖 W1 即可实现）**：

```
三方取数
  ├─ 任一缺失 ────────────────▶ data_missing      → 声明缺什么，不判异常
  ├─ 有"同步中"信号 ──────────▶ sync_in_progress → 标注"待同步"，不判异常
  ├─ 无值差，组合合法 ────────▶ 正常
  ├─ 无值差，组合非法 ────────▶ conflict → EX-04（可直接判）
  └─ 有值差，无法确认是否延迟 ─▶ pending_verification → 待核实/人工
```

**关键点**：`conflict` 的**直接判定路径不依赖 W1**——当三方都"已同步到同一时点"（`updated_at` 一致或均由同一批次写入）且组合落在 C-01~C-06 时，可立即判矛盾。W1 只影响"有值差时自动判定"这一条支路；该支路在 W1 未定期间**整体走 `pending_verification`**，因此 V1 可以**在没有 W1 的情况下安全运行**，只是自动化率较低。

### 10.2 W1 待设计事项（D-08-5）

| # | 待设计项 | 为什么现在不能定 |
|---|---|---|
| 1 | W1 数值 | D-08 defer；需依据模拟数据生成机制确定 |
| 2 | 起算点（以哪一侧时间为基准） | 三方时区/时钟源未定 |
| 3 | 适用数据源（订单/物流/售后是否同一 W1） | 三类数据更新频率差异未知 |
| 4 | 超时行为（超 W1 后是判矛盾还是转人工） | 与"自动化率 vs 误报率"的取舍相关 |
| 5 | W1 与 `as_of` 回溯的关系 | 回溯视图下 `updated_at` 语义未定 |

**确定顺序建议**：先定"模拟数据生成机制与数据更新流程"（D-08-7），再据其真实延迟分布反推 W1，避免拍脑袋取整数值。

---

## 11. 审计设计（V1 最小集）

必记事件（每事件含 `task_id` / `correlation_id` / `operator_id` / `timestamp`（注入时钟））：

`task.created` → `plan.generated` → `agent.dispatched` → `agent.responded` → `anomaly.detected`（含主/从）→ `draft.created` → `approval.requested` → `approval.approved` / `approval.rejected` / `approval.revision_requested` / `approval.expired` / `approval.revision_denied_limit` → `draft.superseded` → `task.finished`

**审计不含**：原始 PII 明文、模型原始提示词全文（可存摘要与哈希）。

---

## 12. 本版未解决问题

| # | 问题 | 状态 | 是否阻塞 |
|---|---|---|---|
| 1 | ~~`revision_required` 用独立状态还是修订标记~~ | **已关闭（D-07 = marker）** | — |
| 2 | ~~异常主次优先级~~ | **已关闭（D-06 = EX-04>EX-01>EX-02>EX-03）** | — |
| 3 | ~~W1 具体数值~~ | **转为待设计（D-08 = defer）**，见 §10.1/§10.2 | 否（有安全降级路径） |
| 4 | `elapsed >= T` 边界口径是否采纳 | 待确认 | 中 |
| 5 | `ClockPort` 是否作为硬约束写入架构 | 待确认（本文按建议执行） | 低 |
| 6 | 审批请求 TTL / 子任务 TTL / 结论 TTL 三值 | 待确认 | 中 |
| 7 | 审批超时后重新发起的具体规则 | 待确认 | 中 |
| 8 | 审批人身份与角色映射的最终方案 | 待确认 | 中 |
| 9 | W1 的五项待设计内容（§10.2） | **待设计**（非"待确认"） | 中 |

以上第 4~9 项**待负责人确认**，本文件不自行定案。

**本版相对 V2.0 的阻塞项变化**：V2.0 的唯一阻塞项（第 1 项）**已关闭**，状态机设计不再有待决分叉。
