# 本地 Embedding 原型设计与验证方案

- 日期：2026-10-12（第二阶段）
- 状态：**只读准备，未实测；等待批准下载模型与启动服务**
- 关联文档：`语义检索技术选型评估_2026-10-12.md`（第一阶段评估）

---

## 1. 接口可行性（已核实）

- 本机 Ollama 客户端版本：**0.40.1**（`ollama --version` 实测）；服务当前**未运行**（无进程、无端口占用）
- 官方文档确认（docs.ollama.com/api/embed）：
  - `POST /api/embed`：当前标准端点，`input` 支持**单条或数组批量**，响应 `embeddings: [[...], ...]`，另支持 `truncate`、`dimensions`（MRL 截断）、`keep_alive`
  - `POST /api/embeddings`（旧版单条）仍兼容；`POST /v1/embeddings` 为 OpenAI 兼容端点（base_url=`http://localhost:11434/v1`，api_key 任意非空字符串）
- 对项目的影响：**零新增 Python 依赖**——原型直接用标准库 `urllib` 调 `/api/embed`；未来接入可选用项目已装的 `openai` 包

## 2. 候选模型（中文优先，均未下载）

| 模型 | 下载体积 | 向量维度 | 上下文 | 中文 | 备注 |
|---|---|---|---|---|---|
| **qwen3-embedding:0.6b（首选）** | 639MB | 1024（支持 MRL 截断） | 32K | 优秀（MTEB 多语言 64.33） | sub-1GB 档最强，RTX 3050 4GB 或 CPU 均可 |
| bge-m3（备选） | ~1.2GB | 1024 | 8K | 优秀（老牌中文强项） | 体积翻倍，优势不明显 |
| nomic-embed-text v1.5 | 274MB | 768 | 8K | **差**（英文模型） | 需 `search_document:`/`search_query:` 前缀，中文场景排除 |
| embeddinggemma | 622MB | 768 | 2K | 一般 | 备选 |
| all-minilm | 46MB | 384 | 短 | 弱 | 仅极端省资源时考虑 |

磁盘余量 44.9GB，模型占位无压力；1024 维 float32 每条约 4KB，当前 5 条记忆的向量总量 < 25KB，**无需独立向量数据库**。

## 3. 原型脚本（已创建：`辅助脚本/semantic_prototype.py`）

标准库实现（urllib + math + json），三种模式：

| 模式 | 作用 | 网络行为 | 本轮是否已运行 |
|---|---|---|---|
| `--check` | 离线结构自检（矩阵完整性、rel/forbid 冲突、key 合法性） | 零网络 | ✅ 已运行（见第 7 节） |
| `--errcheck` | 错误路径验证：空输入 / 空白输入 / 端点不可达 / 超时 | 仅本机端口，含错误端口 | ❌ 待批准后运行 |
| `--run` | 完整矩阵：记忆批量向量化 → 查询向量化 → 余弦相似度 → 排序 → 指标 | 本机 Ollama（需服务+模型） | ❌ **被前置条件阻塞** |

脚本设计要点：
- 余弦相似度**纯 Python 实现**（不依赖 numpy）
- 错误分类语义化：`OLLAMA_UNREACHABLE` / `MODEL_NOT_FOUND` / `TIMEOUT` / `HTTP_<code>`，全部优雅返回、不抛裸栈
- 运行记录：模型名、向量维度、批量向量化耗时、单查询延迟（逐条打印）、阈值
- `--json` 输出受保护：**拒绝覆盖已存在文件**

## 4. 离线测试矩阵（11 用例：8 个召回评测 + 3 个无关查询）

语料（6 条虚构记忆）：major=计算机科学与技术 / career_goal=机器人工程师 / learning_goal=学好检索与评估技术 / preference=喜欢健身 / project=社区图书角项目 / name=李小明

| 用例 | 查询 | 目标(rel) | 允许 | 禁止 |
|---|---|---|---|---|
| Q1 | 我是什么专业的？ | major | — | 其余 5 条 |
| Q2 | 我以后想从事什么工作？ | career_goal | learning_goal | 其余 4 条 |
| Q3a | 我学的是什么方向？ | major | — | 其余 5 条 |
| Q3b | 我的专业是什么 | major | — | 其余 5 条 |
| Q4 | 我的名字是什么 | name | — | 其余 5 条 |
| Q5a/b/c | 明天会下雨吗 / 123乘以456等于多少 / 这家店的服务态度很好 | 无 | — | **全部**（任何记忆越过阈值即失败） |
| Q6 | 我有什么长期目标？ | career_goal | learning_goal | 其余 4 条 |
| Q7 | 帮我看看我在学什么技术 | learning_goal | major | 其余 4 条 |
| Q8 | 我是什么专业的？（category=profile 过滤变体） | major | — | 其余 5 条 |

**判定标准（不凭"相似度较高"判成功）**：非空用例 = rel 全部进 Top3 且无禁止项进 Top3；空召回用例 = 全部记忆 cosine < 阈值。

**指标口径**：Recall@3 = Σ|rel∩Top3|/Σ|rel|；Precision@3 = Σ|Top3∩(rel∪allow)|/Σ|Top3|；MRR = 各查询首个 rel 名次倒数的均值；延迟 = 单查询 embed+检索端到端毫秒。

## 5. 所需资源与前置条件

| 项 | 内容 | 状态 |
|---|---|---|
| 下载模型 | `ollama pull qwen3-embedding:0.6b`（639MB） | **需你批准** |
| 启动服务 | Ollama 守护进程（约 1–2GB 内存占用，模型加载后按需释放） | **需你批准** |
| 运行评测 | `python 辅助脚本/semantic_prototype.py --run --model qwen3-embedding:0.6b` | 待两个前置满足 |
| 错误路径 | `--errcheck`（可与 --run 同批执行） | 同上 |
| 新增依赖 | **0 个** | ✅ |
| 写入正式库 | **无**（原型只用内存语料） | ✅ |

## 6. 风险与降级设计

- 端口失联 / 超时 / 模型缺失 → 脚本分类报错并提示下一步，不崩溃
- 阈值 0.45 为初始值：**须由实测的相似度分布校准**，校准前不接入任何生产逻辑
- 中文效果、真实延迟、显存占用三项**均为待实测项**（本文件不给出虚拟数字）
- 原型不写正式库、不建向量表、不改任何生产代码；正式接入方案见第一阶段评估文档第 5–7 节

## 7. 本轮已执行的验证

- `--check` 离线自检：矩阵结构校验（见运行输出）
- 语法检查：AST 解析通过
- 未执行 `--run` / `--errcheck`（前者被模型与服务前置条件阻塞；后者按"不启动服务"要求暂缓）

## 8. 批准后的执行序（申请）

1. `ollama pull qwen3-embedding:0.6b`（下载 639MB）
2. 启动 Ollama 服务（或确认其已运行）
3. `python 辅助脚本/semantic_prototype.py --errcheck`
4. `python 辅助脚本/semantic_prototype.py --run --model qwen3-embedding:0.6b --json 辅助脚本/semantic_prototype_report_run1.json`
5. 若中文效果或延迟不达标 → 换 bge-m3 重跑（需再批准 1.2GB 下载）
