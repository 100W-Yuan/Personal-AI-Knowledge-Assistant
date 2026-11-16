# R3-02_FINAL 封存记录

> 统一工作台外壳整合 · 最终文档封存
>
> | 项 | 值 |
> |---|---|
> | 所属项目 | My AI 25 |
> | 项目根目录 | `C:\Users\Doro\Desktop\my-ai 25` |
> | 阶段 | R3-02（统一工作台静态骨架实施） |
> | 封存轮次 | R3-02-FINAL |
> | 封存日期 | 2026-11-05 |
> | 封存文档 | `R3-02_FINAL_封存记录.md` |
> | 本轮性质 | 只读核对 + 文档归档 |
> | 封存状态 | **R3-02：静态实施与代码复核通过，运行时验收待授权。** |

---

## 1. 任务背景和目标

### 1.1 背景

My AI 25 项目长期存在两个相互独立的前端：

- **AI 助手前端**：`/`，由 `app_ui/index.html` + `app_ui/static/style.css` + `app_ui/static/app.js` 构成，R2-04 完成从 `main.py` 内联 HTML 的抽离。
- **业务管理前端**：`/business-ui`，由 `business_ui/index.html` + `business_ui/static/style.css` + `business_ui/static/app.js` 构成。

两者各自独立可访问，但缺少统一入口。R3-01（统一工作台整合前只读架构审查）对三种整合方案做了比较：

| 方案 | 做法 | 结论 |
|---|---|---|
| A | 外壳页 + iframe | **采纳** |
| B | 单页合并（改源码消除冲突） | 否决 |
| C | 独立部署（新增服务与端口） | 否决 |

否决 B 的两条硬依据（来自 R3-01 只读审查）：

1. `body` 被两套 CSS 同时定义 —— AI 侧 `app_ui/static/style.css` 用 Arial / `#f5f5f5`，业务侧 `business_ui/static/style.css` 用系统字体栈 / `var(--bg)`。二者都是代码基座，不改源码无法共存。
2. `business_ui/static/app.js` 在 `init()` 末尾存在「无 hash 时强制写入 `#/orders`」的逻辑，合并进单页后会接管整页 URL。

方案 A 的优势：零改原有代码、天然样式与脚本隔离、可秒级回退。

### 1.2 本轮目标

对 R3-02 的实施与验收结果做**最终文档封存**，形成一份可追溯、边界清晰的存档记录。

### 1.3 本轮不做什么

本轮**不修改任何项目代码**，不启动服务，不导入 `main.py`，不访问数据库，不执行测试，不进入 R3-03。详见第 9 节。

---

## 2. 四个目标文件及其变更情况

R3-02 共新增 3 个文件、修改 1 个文件，无删除、无移动、无重命名。

| 文件 | 操作 | 字节 | 行数 | 换行符 | 编码 |
|---|---|---|---|---|---|
| `app_ui/workbench.html` | 新建 | 822 | 31 | LF（0 CRLF） | UTF-8 无 BOM |
| `app_ui/static/workbench.css` | 新建 | 2,258 | 124 | LF（0 CRLF） | UTF-8 无 BOM |
| `app_ui/static/workbench.js` | 新建 | 2,391 | 85 | LF（0 CRLF） | UTF-8 无 BOM |
| `main.py` | 修改 | 17,676 → **18,599** | 543 → **574** | CRLF（574 CRLF / 0 裸 LF） | UTF-8 无 BOM |

三项换行符差异是**有意的**：三个新文件按 R3-02 决策统一为 LF；`main.py` 原有换行符为 CRLF，修改时必须保持，避免产生整文件级换行符变更噪音。

### 2.1 各文件职责

**`app_ui/workbench.html`** —— 外壳页骨架。结构为 `<header class="wb-topbar">`（品牌名 + 两个导航项）+ `<main class="wb-main">`（单个 `<iframe id="wb-frame" src="/">`）+ 两个外部资源引用。

静态约束实测：

- `<iframe` 出现 **1** 次；`sandbox` **0** 次；`srcdoc` **0** 次；`document.write` **0** 次
- 内联 `<style>` **0** 处；内联 `onclick` / `onchange` / `onload` / `onsubmit` **0** 处
- `lang="zh-CN"`，`<title>My AI 25 · 工作台`
- 资源引用：`<link rel="stylesheet" href="/workbench/static/style.css">`、`<script src="/workbench/static/app.js"></script>`

**`app_ui/static/workbench.css`** —— 外壳专属样式。定义 6 个 `--wb-*` CSS 变量，仅作用于 `.wb-*` 类与外壳自身的 `html/body`；含 1 个 `@media (max-width: 640px)` 断点。所有选择器均带 `wb-` 前缀，不与两个原有前端的 14 / 46 个类名产生交集。

**`app_ui/static/workbench.js`** —— 外壳交互。首行 `"use strict";`，整体包在 IIFE 内。实测：

- IIFE 开合各 **1** 次；IIFE 外裸 `var` **0** 个；`window.xxx =` 赋值 **0** 次
- 隔离约束：`contentDocument` / `contentWindow` / `postMessage` / `document.write` 均 **不存在**
- 无 `fetch` / `XMLHttpRequest`，无 `/business/` 接口路径
- 仅 1 个 `addEventListener("hashchange", ...)`

**`main.py`** —— 新增 31 行，位于第 431–461 行，插入在 `app_ui_script()` 收尾 `)`（第 428 行）之后、「业务管理前端」注释块之前。

31 行的构成：2 行注释 + 3 行路径常量 + 8 个空行 + 18 行路由代码（3 × 6 行）。空行落在第 436、437、444、445、452、453、460、461 行；非空行 23 行。块总字节 **923 B**（含每行 CRLF），块 SHA256 `79474c3dcc8f0f599a29f7b3bfe008a07a297f4d3b193f5d35acb8c4a3f0bb50`。

块原文：

```python
# 统一工作台外壳：只暴露这三个固定文件。
# 路径在导入时一次性确定，任何请求参数都不参与路径拼接。
WORKBENCH_INDEX = APP_UI_DIR / "workbench.html"
WORKBENCH_CSS = APP_UI_DIR / "static" / "workbench.css"
WORKBENCH_JS = APP_UI_DIR / "static" / "workbench.js"

@app.get("/workbench")
def workbench_index():
    """返回统一工作台页面。"""
    return FileResponse(
        WORKBENCH_INDEX, media_type="text/html; charset=utf-8"
    )

@app.get("/workbench/static/style.css")
def workbench_style():
    """返回统一工作台样式。"""
    return FileResponse(
        WORKBENCH_CSS, media_type="text/css; charset=utf-8"
    )

@app.get("/workbench/static/app.js")
def workbench_script():
    """返回统一工作台脚本。"""
    return FileResponse(
        WORKBENCH_JS, media_type="application/javascript; charset=utf-8"
    )
```

路径常量全部基于既有 `APP_UI_DIR`（`main.py:401`，其值 `Path(__file__).resolve().parent / "app_ui"`），未引入新的目录基准，也不接受任何请求参数参与路径拼装。

---

## 3. 新增路由清单

新增 3 条 GET 路由，全部为固定文件直出：

| 装饰器行 | `def` 行 | 方法 | 路径 | 处理函数 | 媒体类型 |
|---|---|---|---|---|---|
| 438 | 439 | GET | `/workbench` | `workbench_index()` | `text/html; charset=utf-8` |
| 446 | 447 | GET | `/workbench/static/style.css` | `workbench_style()` | `text/css; charset=utf-8` |
| 454 | 455 | GET | `/workbench/static/app.js` | `workbench_script()` | `application/javascript; charset=utf-8` |

路由总数由 **10** 条增至 **13** 条。静态判定：路径唯一 `True`、处理函数名唯一 `True`、重复路径「无」。

全部 13 条显式路由（含原有 10 条，行号为 R3-02 之后的状态）：

```
  156  POST  /chat                              -> chat()
  289  GET   /knowledge/search                  -> knowledge_search()
  310  POST  /knowledge/import                  -> knowledge_import()
  408  GET   /                                  -> index()
  416  GET   /app-ui/static/style.css           -> app_ui_style()
  424  GET   /app-ui/static/app.js              -> app_ui_script()
  439  GET   /workbench                         -> workbench_index()          ← 新增
  447  GET   /workbench/static/style.css        -> workbench_style()          ← 新增
  455  GET   /workbench/static/app.js           -> workbench_script()         ← 新增
  484  GET   /business-ui                       -> business_ui_index()
  492  GET   /business-ui/app.js                -> business_ui_script()
  500  GET   /business-ui/style.css             -> business_ui_style()
  510  POST  /analyze-image                     -> analyze_image()
```

注：上表行号为函数定义行。第 484 / 492 / 500 / 510 条位于插入点之后，因新增 31 行而整体后移 31 行；其方法、路径、处理函数与媒体类型均未变。

### 3.1 统一工作台架构

```
浏览器
  │
  └─ GET /workbench  ──►  app_ui/workbench.html（外壳）
                              ├─ /workbench/static/style.css
                              ├─ /workbench/static/app.js
                              └─ <iframe id="wb-frame" src="/">
                                     │
                                     ├─ #/ai       ──► iframe.src = "/"            （AI 助手）
                                     └─ #/business ──► iframe.src = "/business-ui" （业务管理）
```

- 外层导航使用 `#/ai`、`#/business`，由 `workbench.js` 的 `hashchange` 驱动 iframe 的 `src` 切换。
- 两个原有前端**保持完全独立**，各自仍是完整文档，各自持有自己的样式表与脚本。
- 业务页自己的 `#/orders` 系列 hash 只存在于 iframe 内部，不会与外壳的 `#/ai`、`#/business` 相互干扰 —— 因为二者处于不同文档、不同 `window`。
- `parseHash()` 使用 `/^#\/?/` 剥离前缀；`DEFAULT_MODULE = "ai"`；首次进入且无 hash 时用 `history.replaceState` 补 `#/ai`（而非 `location.hash =`，以避免多触发一次 `hashchange`）。

---

## 4. 改前、改后 SHA256

### 4.1 四个目标文件

| 文件 | 改前 SHA256 | 改后 SHA256 |
|---|---|---|
| `main.py` | `e9cadb99ce6ef0b4fc1a3c802d6c52eb234aeb1d8b0b3ce0ba2d6c0c5aa58220` | `5c14f2354d4750b0088adb22b01da34654032d675fc84a87881c5639a2be74f5` |
| `app_ui/workbench.html` | 不存在 | `24f864cd3170a901f3b601e948bacc1df3f4862b71f818ab02868f8db987c528` |
| `app_ui/static/workbench.css` | 不存在 | `606143374416db68ac5c3069635145ac064e2fe2711992ccc5e07a0084964e28` |
| `app_ui/static/workbench.js` | 不存在 | `d7ce2ae423bfd8912a9c26bfff32bff89e37737c73895144a010e19dce02208a` |

### 4.2 未被改动的七个前端文件

以下七个文件在 R3-02 前后逐字节一致：

| 文件 | SHA256 |
|---|---|
| `app_ui/index.html` | `3ad2ad61685d34c4fba4a3bdf8c440650844e0c31ea194bfaac858521521d364` |
| `app_ui/index.workbench.draft.html` | `2fad76950f40f9ab4b5bc59f8b81f5bbf13c94785fbf366abd96c7c4faec6599` |
| `app_ui/static/style.css` | `7f419627de1cd83814672b7e39313c6391e819e622f5ca61854375555e72ddd1` |
| `app_ui/static/app.js` | `32be8bbabefffa4f4f5e347dcd4b917f267a6278605c4476cdcbda81de9d4429` |
| `business_ui/index.html` | `d9378ce458cb7e1038b13792373b9842c4a88c43bc380c0472ad35c65eb6fbd1` |
| `business_ui/static/app.js` | `91a4da888afcbfa7f4aa0be0e578deba2f24ac4af22bc0731903a7072bf840d9` |
| `business_ui/static/style.css` | `8a7984a5513e260907a07f3a70e4d41b6327905e0dd7b627604fd15acb987b37` |

### 4.3 封存时点复核

封存时（2026-11-05）对上述 11 个文件 + `data/business.db` 重新计算 SHA256，**全部与 R3-02 执行记录逐项一致**。这说明从 R3-02 执行完成到封存之间，没有发生任何文件变动。

---

## 5. 静态验收结果

R3-02 执行时完成 10 项静态验收，结果如下。

| # | 验收项 | 结果 | 证据要点 |
|---|---|---|---|
| 1 | 四个目标文件写入成功 | **通过** | 新建文件用 `open(path, "xb")` 独占创建，写入后立即回读比对，全部 `OK` |
| 2 | 新文件编码与换行格式 | **通过** | 均 UTF-8 可解码、无 BOM、CRLF=0、独立 CR=0、末尾有 LF |
| 3 | `main.py` AST 静态解析 | **通过** | `ast.parse` 成功，未执行导入；模块顶层 58 条语句 |
| 4 | 新增三条路由、原十条保留 | **通过** | 10 → 13 条，路径唯一；原 10 条经"反向还原逐字节一致"佐证 |
| 5 | HTML / CSS / JS 路径对应 | **通过** | 两条引用与两条静态路由一一对应 |
| 6 | iframe 隔离约束静态检查 | **通过** | 单 iframe、无 `sandbox`、无 `srcdoc`、无 `document.write`、JS 不触碰 `contentDocument` / `contentWindow` |
| 7 | 原有七个前端文件指纹未变 | **通过** | 见第 4.2 节 |
| 8 | `main.py` 反向还原后 SHA256 一致 | **通过** | 见第 5.1 节 |
| 9 | 数据库文件前后 SHA256 / 大小 / mtime 未变 | **通过（限文件层）** | 见第 7 节，含证明边界说明 |
| 10 | 项目目录无本轮遗留临时 Python 脚本 | **通过** | 项目内 `*.py` 且 mtime 晚于 2026-11-05 00:40 的仅 `main.py`；脚本均置于系统临时目录 |

### 5.1 关于"原十条路由保留"的证据强度

第 4 项最初用"逐条行号比对"表述，但该方法在插入点之后的 4 条路由上存在循环引用之嫌（行号取自写入后的文件）。因此该项的**真正依据**是第 8 项的反向还原：

> 从当前 `main.py` 中剔除 1-based 第 431–461 行后重新拼接，得到 SHA256 `e9cadb99…c5aa58220`、17,676 B、543 行，与改前基线**逐字节一致**。

既然删除这 31 行即可完整还原改前文件，则改前文件中的每一条路由（包括位于插入点之后的 4 条）必然保持原样。这是比逐条行号比对更强的证据，且不依赖执行脚本自身的中间输出。

未采用 `git diff` 作为依据的原因：工作区存在 R1 ~ R2-04 的累计未提交改动，`git diff` 会把它们一并计入（实测显示 151 insertions / 508 deletions），无法作为单轮变更的基线。

### 5.2 一处执行中止（如实记录）

首次落盘脚本在 preflight 阶段**主动中止**：断言把 `app_ui_script` 的收尾 `)` 写成 `L[428]`，实际位于 `L[427]`（`L[428]`、`L[429]` 是两个空行）。断言失败后脚本以非零状态退出，**未写入任何字节**；事后复核 `main.py` SHA 仍为 `e9cadb99…8220`、三个目标文件仍不存在。修正索引并补两条断言后重跑通过。

该中止未产生任何副作用，但记录在此以保持过程可追溯。

---

## 6. ChatGPT 静态代码复核结论

### 6.1 已转述的结论

据 R3-02-FINAL 任务书载明：

> ChatGPT 已对 HTML、CSS、JS 和新增后端路由进行静态代码复核，未发现阻断性问题。

### 6.2 待核实项

以下信息**未随任务书提供**，本记录无法确认，标记为**待核实**：

- 复核的具体范围清单（哪些文件、哪些关注点）
- 使用的检查方法（人工阅读 / 工具辅助）
- 除"未发现阻断性问题"外是否提出次要问题或改进建议
- 复核对应的文件 SHA256 是否与第 4.1 节一致

即：**"未发现阻断性问题"这一结论本身记录在案，但其覆盖面与依据未独立验证。** 本记录不将其升格为"代码质量已确认"。

---

## 7. 数据库零修改证据及其证明边界

### 7.1 证据

| 观测项 | R3-02 之前 | R3-02 之后 / 封存时点 | 是否一致 |
|---|---|---|---|
| `data/business.db` SHA256 | `3ffabb53488448db9cef7c615963930fdb37106b127fba1847766bde6632d279` | 同左 | 是 |
| `data/business.db` 大小 | 73,728 B | 73,728 B | 是 |
| `data/business.db` mtime | `2026-11-04 22:05:40.484688700 +0800` | 同左 | 是 |
| `data/` 目录 mtime | `2026-11-04 22:05:40.485686600 +0800` | 同左 | 是 |

辅助证据：

- 本轮及 R3-02 全程**未启动服务**，故 `lifespan` 未执行，`init_database()` 未被调用。
- 全程**未导入 `main.py`**，`main.py` 只经 `ast.parse` 做语法解析 —— 该操作不执行模块体、不触发 `load_dotenv()`、不触发任何数据库路径。
- 全程未执行 `pytest` 或其他测试。
- 字节码侧：`find . -name "*.pyc" -newermt "2026-11-05 00:00:00"` 结果为空；根 `__pycache__` 目录 mtime 仍为 `2026-11-04 22:26:02`，说明本轮没有任何 Python 模块被实际编译加载。

### 7.2 证明边界（必须与证据同时阅读）

上述结论的效力**严格限定在文件层面**，不能外推为"数据库逻辑上绝未被访问"。具体边界：

1. **只能证明文件字节与元数据未变化。** 无法排除某进程以只读方式打开过该文件而不产生任何写入 —— 只读打开不会改变 SHA256、大小或 mtime（访问时间 atime 未纳入观测，且本机可能以 `relatime` 挂载）。
2. **未审计进程与句柄。** 未使用 `lsof` / 句柄枚举等手段证明"无进程持有该文件"。
3. **未审计 SQLite 的旁路文件。** 未检查 `-wal` / `-shm` / `-journal` 是否存在或变化（若存在，本记录未覆盖）。
4. **未证明网络侧零请求。** 未抓包，未证明没有其他客户端连向本项目端口（该项目本轮未监听端口，此为间接推断，非直接证据）。

因此第 5 节第 9 项的准确表述应为：**"数据库文件在字节与元数据层面未发生变化（文件级证据），逻辑访问层面未独立验证。"**

---

## 8. 未完成的运行时验收事项

R3-02 未启动服务，以下事项**全部未验收**。**不得**将其标记为通过。

| # | 未验收事项 | 说明 |
|---|---|---|
| 1 | `GET /workbench` 的实际 HTTP 响应 | 未发出任何请求，状态码、响应头、响应体均未观测 |
| 2 | CSS、JS 的实际加载 | 未验证 `/workbench/static/style.css`、`/workbench/static/app.js` 的状态码与 `Content-Type` |
| 3 | iframe 是否成功加载两个原有页面 | 未验证 `/` 与 `/business-ui` 在 iframe 内是否正常渲染 |
| 4 | 导航切换、刷新、前进/后退的浏览器行为 | `#/ai` ↔ `#/business` 切换、F5 刷新后 hash 与 iframe 的一致性、浏览器历史栈行为均未验证 |
| 5 | 窄屏布局、滚动条、页面高度 | `@media (max-width: 640px)` 断点效果未验证；外层 `overflow: hidden` + iframe `height: 100%` 是否产生双滚动条未验证 |
| 6 | 两个模块运行时的视觉协调性 | 顶栏配色与两个子页面的视觉过渡未验证 |

### 8.1 阻塞原因

运行时验收被**"数据库零变化"约束**硬阻塞：

- `main.py` 的 `lifespan` 无条件调用 `init_database()`（该函数会 `mkdir`、执行 `executescript(SCHEMA_SQL)`、`commit()`），只要启动服务就会触碰 `data/business.db`。
- 在约束未解除前，任何启动都违反第 7 节的零修改要求。
- 解除方式需负责人明确授权（R3-01 已将其登记为待决策项 D5）。

### 8.2 静态检查通过 ≠ 运行时功能已验证

本记录中所有"通过"均指**静态**性质：文件存在、字节正确、编码正确、语法可解析、路由声明无冲突、代码中不存在被禁止的 API 调用。这些性质与"页面能正常打开并交互"之间没有推导关系。

---

## 9. 本轮执行边界和禁止事项

### 9.1 本轮允许

- 查看已存在的 R3-02 执行报告与相关审计记录
- 核对封存记录所需的已有信息（只读）
- 创建本封存记录文档

### 9.2 本轮禁止（实际执行情况）

| 禁止项 | 是否遵守 |
|---|---|
| 修改 `main.py` 和任何前端代码 | 遵守，未修改 |
| 修改、删除、移动或重命名现有文件 | 遵守，仅新建本记录 |
| 启动或停止服务 | 遵守，未启动 |
| 导入 `main.py` | 遵守，未导入（仅 `ast.parse` 于 R3-02 阶段，本轮未再执行） |
| 访问或连接数据库 | 遵守，仅做文件指纹只读计算 |
| 读取 `.env`、API Key、Token 或其他凭据 | 遵守，未读取 |
| 执行 `pytest` 或其他测试 | 遵守，未执行 |
| 调用大模型 API | 遵守，未调用 |
| 安装、升级或卸载依赖 | 遵守，未安装 |
| 执行 Git 回退、清理或重置 | 遵守，未执行（本轮仅使用 `git status` / `git diff` 只读查看，且仅在 R3-02 阶段） |
| 擅自进入 R3-03 | 遵守，未进入 |

### 9.3 关于本记录的唯一写入

本轮在项目内**新建了且仅新建了一个文件**：本封存记录 `R3-02_FINAL_封存记录.md`。除此之外未产生任何项目内文件变更，也未在项目目录留下临时脚本（R3-02 的构造与校验脚本均置于系统临时目录）。

---

## 10. 封存状态与后续建议

### 10.1 封存状态

**R3-02：静态实施与代码复核通过，运行时验收待授权。**

拆解：

| 维度 | 状态 |
|---|---|
| 静态实施（4 文件写入） | 通过 |
| 编码 / 换行 / 指纹 | 通过 |
| 语法与路由声明 | 通过 |
| 隔离约束（静态） | 通过 |
| 代码复核（第三方） | 已转述"未发现阻断性问题"，覆盖面待核实 |
| 运行时行为 | **未验收**（6 项，见第 8 节） |
| 数据库零修改 | 文件层通过，逻辑层未验证 |

### 10.2 前序阶段状态

- **R2-04**（AI 助手 HTML 抽离与静态资源路由接入）：保持原封存状态，不追溯修改。
- **R2-05**（静态验收归档）：保持原封存状态，不追溯修改。其归档结论「R2-04 页面抽离及 R2-05 静态验收已经完成。运行时验收因数据库零变化约束暂缓，不代表运行时功能已验证」继续有效。

### 10.3 R3 系列后续阶段（均未启动）

| 阶段 | 内容 | 状态 |
|---|---|---|
| R3-03 | 导航与 hash 同步 | 未启动，等待负责人决定 |
| R3-04 | 布局与响应式 | 未启动 |
| R3-05 | 运行时验收 | **被 D5 硬阻塞** |
| R3-06 | 归档 | 未启动 |

### 10.4 建议

1. **解除 D5 是推进 R3-05 的唯一前置。** 需负责人明确：是否允许在验收期间让 `data/business.db` 发生变化；若不允许，是否改为指向一个临时副本数据库（需评估 `init_database()` 的路径来源是否可参数化，此点尚未审查）。
2. **三条新路由缺 404 保护，建议在 R3-05 一并观察。** 与既有 `app_ui` 三条路由行为一致（`FileResponse` 构造时 `os.stat`，文件缺失将抛 `FileNotFoundError` → 500），而业务侧 `business_ui` 有显式 404 封装。此为 R2-04 时 T4"不建 helper"决策的延续，属设计取舍而非缺陷，但行为不一致值得记录。
3. **窄屏双滚动条风险建议纳入 R3-04。** 外壳 `body { overflow: hidden }` 与 iframe `height: 100%` 组合下，子页面自身若有滚动容器，可能出现嵌套滚动。此为静态推断，未经浏览器验证。
4. **若运行时验收最终无法进行**，建议将方案 A 的封存状态明确记为"静态就绪、未上线"，并在 `/workbench` 入口处（或部署说明中）标注该路径未经运行时验证，避免被误当作可用功能。

### 10.5 证据分级声明

本记录中出现的所有结论，按证据强度分为三级：

- **实测（静态）**：由文件字节计算或 AST 解析直接得出，可复现。
- **实测（只读观测）**：由 `stat` / `sha256sum` / `find` 等只读命令得出。
- **转述**：由任务书或负责人提供，未经独立验证（第 6 节）。

凡不属于以上三级的内容，均标记为**待核实**，不推断为通过。

---

## 附：封存清单

| 项 | 值 |
|---|---|
| 封存文档路径 | `C:\Users\Doro\Desktop\my-ai 25\R3-02_FINAL_封存记录.md` |
| 本轮新建文件数 | 1（本记录） |
| 本轮修改文件数 | 0 |
| 本轮删除文件数 | 0 |
| 封存时点 `main.py` SHA256 | `5c14f2354d4750b0088adb22b01da34654032d675fc84a87881c5639a2be74f5` |
| 封存时点 `data/business.db` SHA256 | `3ffabb53488448db9cef7c615963930fdb37106b127fba1847766bde6632d279` |
| 是否启动服务 | 否 |
| 是否导入 `main.py` | 否 |
| 是否访问数据库 | 否（仅文件指纹只读计算） |
| 封存状态 | R3-02：静态实施与代码复核通过，运行时验收待授权 |
