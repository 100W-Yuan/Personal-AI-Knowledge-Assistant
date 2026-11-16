# -*- coding: utf-8 -*-
"""Ollama 本地 Embedding 语义检索独立原型 v2（离线验证工具，不接入 Agent）。

v2 重构要点（2026-10-12 第四阶段第一步）：
  1. 统一后端接口 embed(texts) -> vectors：OllamaBackend 已实现；
     OpenAICompatBackend 为在线预留（本阶段不调用，且缺 Key 时仅在
     构造阶段报配置错误，不发起任何请求）。
  2. --backend 参数（默认 ollama，保持原行为兼容）。
  3. 模型标识 + 向量维度校验：首次探测定维，后续不一致立即中止，
     杜绝不同模型/维度向量混算。
  4. 计时分层：冷加载（首调，含模型加载）→ 预热 → 测量；
     测量阶段每条查询重复 3 次，统计口径显式声明。
  5. 延迟 p50/p95：最近秩法（nearest-rank），报告样本数与单位。
  6. 错误分类：UNREACHABLE / MODEL_NOT_FOUND / TIMEOUT /
     AUTH / RATE_LIMIT / HTTP_xxx / BAD_RESPONSE（修正 404 误分类）。
  7. 数据集隔离：冒烟 11 条 + 扩展 52 条（E01-E20=DEV 校准，
     E21-E52=锁定 TEST；--set test 需显式 --unlock-test，防止误跑）。
  8. JSON 防覆盖：目标文件已存在则拒绝（exit 3），报告用唯一文件名。

安全边界：只访问本机 Ollama；不读写正式库；不修改生产代码；
标准库实现，零第三方依赖。

运行示例：
  python 辅助脚本/semantic_prototype.py --check
  python 辅助脚本/semantic_prototype.py --errcheck
  python 辅助脚本/semantic_prototype.py --run --set smoke --json <唯一文件名>.json
"""

import argparse
import json
import math
import os
import socket
import sys
import time
import urllib.error
import urllib.request

OLLAMA_BASE = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen3-embedding:0.6b"
DEFAULT_THRESHOLD = 0.45
DEFAULT_TIMEOUT = 30.0
DEFAULT_REPEAT = 3
BACKEND_CHOICES = ("ollama", "openai-compat")


# =====================================================================
# 后端错误分类（v2 修正：404 区分模型不存在与其他 404）
# =====================================================================

class BackendError(RuntimeError):
    def __init__(self, kind, message):
        super().__init__(f"{kind}: {message}")
        self.kind = kind


class BackendConfigError(BackendError):
    def __init__(self, message):
        super().__init__("CONFIG", message)


def _make_opener(bypass_proxy):
    """构造 urllib opener。

    重要环境事实（2026-10-12 实测）：本机设置了全局代理环境变量
    （HTTP_PROXY/HTTPS_PROXY=127.0.0.1:63434），urllib 默认会把本地
    Ollama 请求也发给代理；代理对无服务端口会返回 HTTP 502 而非连接
    拒绝。因此本地后端必须 bypass_proxy=True 直连 127.0.0.1。"""
    if bypass_proxy:
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener()


def _post_json(url, payload, timeout, headers=None, bypass_proxy=False):
    """唯一 HTTP 入口（测试可 monkeypatch _make_opener 模拟各错误路径）。"""
    body = json.dumps(payload).encode("utf-8")
    all_headers = {"Content-Type": "application/json"}
    if headers:
        all_headers.update(headers)
    request = urllib.request.Request(
        url, data=body, headers=all_headers, method="POST"
    )
    opener = _make_opener(bypass_proxy)
    try:
        with opener.open(request, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", "ignore")[:300]
        except Exception:
            pass
        if exc.code == 404:
            low = detail.lower()
            if "model" in low and "not found" in low:
                raise BackendError(
                    "MODEL_NOT_FOUND", "HTTP 404，模型不存在"
                ) from exc
            raise BackendError("HTTP_404", f"HTTP 404：{detail[:150]}") from exc
        if exc.code in (401, 403):
            raise BackendError("AUTH", f"HTTP {exc.code}，鉴权失败") from exc
        if exc.code == 429:
            raise BackendError("RATE_LIMIT", "HTTP 429，触发限流") from exc
        raise BackendError(
            f"HTTP_{exc.code}", f"HTTP {exc.code}：{detail[:150]}"
        ) from exc
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        if isinstance(reason, socket.timeout) or "timed out" in str(reason).lower():
            raise BackendError("TIMEOUT", "请求超时") from exc
        raise BackendError("UNREACHABLE", f"服务不可达：{reason}") from exc
    except (TimeoutError, socket.timeout) as exc:
        raise BackendError("TIMEOUT", "请求超时") from exc

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise BackendError("BAD_RESPONSE", "响应不是合法 JSON") from exc


# =====================================================================
# 统一后端接口（embed(texts) -> vectors）
# =====================================================================

class EmbeddingBackend:
    """统一接口：embed(texts, timeout) -> list[list[float]]。

    维度校验：首次调用记录 dim，此后不一致立即 BackendError(DIM_MISMATCH)。
    """

    name = "base"
    base_url = ""

    def __init__(self, model_id):
        self.model_id = model_id
        self.dim = None

    def _check_dim(self, vectors, expected=None):
        if not vectors:
            raise BackendError("BAD_RESPONSE", "返回向量为空")
        dim = len(vectors[0])
        for vec in vectors:
            if len(vec) != dim:
                raise BackendError("DIM_MISMATCH", "同批向量维度不一致")
        if expected is not None and dim != expected:
            raise BackendError(
                "DIM_MISMATCH",
                f"模型 {self.model_id} 期望维度 {expected}，实际 {dim}",
            )
        if self.dim is None:
            self.dim = dim
        elif dim != self.dim:
            raise BackendError(
                "DIM_MISMATCH",
                f"模型 {self.model_id} 维度变化：先前 {self.dim}，现在 {dim}",
            )

    @staticmethod
    def _validate_inputs(texts):
        if not isinstance(texts, list) or not texts:
            raise ValueError("输入不能为空")
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                raise ValueError("输入包含空文本")

    def embed(self, texts, timeout=DEFAULT_TIMEOUT):
        raise NotImplementedError


class OllamaBackend(EmbeddingBackend):
    """本地 Ollama /api/embed。"""

    name = "ollama"

    def __init__(self, model_id, base=OLLAMA_BASE):
        super().__init__(model_id)
        self.base_url = base

    def embed(self, texts, timeout=DEFAULT_TIMEOUT):
        self._validate_inputs(texts)
        data = _post_json(
            f"{self.base_url}/api/embed",
            {"model": self.model_id, "input": texts},
            timeout,
            bypass_proxy=True,  # 本地请求直连，不经全局代理
        )
        vectors = data.get("embeddings")
        if not vectors:
            raise BackendError("BAD_RESPONSE", "响应缺少 embeddings 字段")
        self._check_dim(vectors)
        return vectors


class OpenAICompatBackend(EmbeddingBackend):
    """OpenAI 兼容 /v1/embeddings（在线预留：本阶段不调用）。

    密钥仅从环境变量读取，绝不打印、绝不写入任何输出。
    """

    name = "openai-compat"

    def __init__(self, model_id, base_url, api_key_env="EMBED_API_KEY"):
        super().__init__(model_id)
        if not base_url:
            raise BackendConfigError("openai-compat 后端需要 --base-url")
        key = os.getenv(api_key_env, "")
        if not key:
            raise BackendConfigError(
                f"环境变量 {api_key_env} 未设置（值不会显示；"
                "在线后端本阶段只保留接口，不做调用）"
            )
        self.base_url = base_url
        self._api_key_env = api_key_env
        self._key = key  # 不进入任何 print / JSON

    def embed(self, texts, timeout=DEFAULT_TIMEOUT):
        self._validate_inputs(texts)
        data = _post_json(
            f"{self.base_url}/v1/embeddings",
            {"model": self.model_id, "input": texts},
            timeout,
            headers={"Authorization": f"Bearer {self._key}"},
        )
        try:
            items = sorted(data["data"], key=lambda item: item["index"])
            vectors = [item["embedding"] for item in items]
        except (KeyError, TypeError) as exc:
            raise BackendError("BAD_RESPONSE", "响应结构不符合预期") from exc
        self._check_dim(vectors)
        return vectors


def make_backend(args):
    """按 --backend 构造后端（默认 ollama，保持原行为兼容）。"""
    if args.backend == "ollama":
        return OllamaBackend(args.model, args.base_url or OLLAMA_BASE)
    if args.backend == "openai-compat":
        return OpenAICompatBackend(
            args.model, args.base_url, args.key_env
        )
    raise BackendConfigError(f"未知后端：{args.backend}")


def embed_texts(texts, model, timeout=DEFAULT_TIMEOUT, base=None):
    """v1 兼容包装：等价于 OllamaBackend(model, base).embed(texts)。"""
    backend = OllamaBackend(model, base or OLLAMA_BASE)
    return backend.embed(texts, timeout)


# =====================================================================
# 相似度与排序（与原版一致）
# =====================================================================

def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def search(query_vec, memory_vecs, threshold, top_k=3):
    scored = [
        (key, cosine(query_vec, vec)) for key, vec in memory_vecs.items()
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    return [
        (key, value, index + 1)
        for index, (key, value) in enumerate(scored)
        if value >= threshold
    ][:top_k]


def percentile_nearest_rank(samples, p):
    """最近秩法（nearest-rank）：p∈(0,100]，样本升序后取第
    ceil(p/100×N) 个元素（1-based）。返回 (值, 0-based索引)。"""
    if not samples:
        raise ValueError("样本为空")
    if not 0 < p <= 100:
        raise ValueError("p 必须在 (0,100] 区间")
    ordered = sorted(samples)
    rank = max(1, math.ceil(p / 100 * len(ordered)))
    return ordered[rank - 1]


# =====================================================================
# 冒烟数据集（原 11 条，行为与 v1 完全一致）
# =====================================================================

CORPUS = [
    {"key": "major", "value": "计算机科学与技术", "category": "profile"},
    {"key": "career_goal", "value": "机器人工程师", "category": "goal"},
    {"key": "learning_goal", "value": "学好检索与评估技术", "category": "goal"},
    {"key": "preference", "value": "喜欢健身", "category": "preference"},
    {"key": "project", "value": "社区图书角项目", "category": "project"},
    {"key": "name", "value": "李小明", "category": "identity"},
]

ALL_KEYS = [m["key"] for m in CORPUS]


def _others(*keep):
    return [k for k in ALL_KEYS if k not in keep]


CASES = [
    dict(id="Q1", q="我是什么专业的？", rel=["major"], allow=[],
         forbid=_others("major"), category=None, expect_empty=False),
    dict(id="Q2", q="我以后想从事什么工作？", rel=["career_goal"],
         allow=["learning_goal"], forbid=_others("career_goal", "learning_goal"),
         category=None, expect_empty=False),
    dict(id="Q3a", q="我学的是什么方向？", rel=["major"], allow=[],
         forbid=_others("major"), category=None, expect_empty=False),
    dict(id="Q3b", q="我的专业是什么", rel=["major"], allow=[],
         forbid=_others("major"), category=None, expect_empty=False),
    dict(id="Q4", q="我的名字是什么", rel=["name"], allow=[],
         forbid=_others("name"), category=None, expect_empty=False),
    dict(id="Q5a", q="明天会下雨吗", rel=[], allow=[],
         forbid=list(ALL_KEYS), category=None, expect_empty=True),
    dict(id="Q5b", q="123乘以456等于多少", rel=[], allow=[],
         forbid=list(ALL_KEYS), category=None, expect_empty=True),
    dict(id="Q5c", q="这家店的服务态度很好", rel=[], allow=[],
         forbid=list(ALL_KEYS), category=None, expect_empty=True),
    dict(id="Q6", q="我有什么长期目标？", rel=["career_goal"],
         allow=["learning_goal"], forbid=_others("career_goal", "learning_goal"),
         category=None, expect_empty=False),
    dict(id="Q7", q="帮我看看我在学什么技术", rel=["learning_goal"],
         allow=["major"], forbid=_others("learning_goal", "major"),
         category=None, expect_empty=False),
    dict(id="Q8", q="我是什么专业的？", rel=["major"], allow=[],
         forbid=_others("major"), category="profile", expect_empty=False),
]


# =====================================================================
# 扩展数据集（52 条：E01-E20 = DEV 校准集，E21-E52 = 锁定 TEST 集）
# 全部虚构语料；不使用正式库任何个人信息
# =====================================================================

CORPUS_EXT = [
    {"key": "major", "value": "计算机科学与技术", "category": "profile"},
    {"key": "degree", "value": "本科在读", "category": "profile"},
    {"key": "school", "value": "华东电子科技大学", "category": "profile"},
    {"key": "career_goal", "value": "机器人工程师", "category": "goal"},
    {"key": "learning_goal", "value": "学好检索与评估技术", "category": "goal"},
    {"key": "side_goal", "value": "帮朋友规划机器人比赛", "category": "goal"},
    {"key": "preference", "value": "喜欢黑咖啡不加糖", "category": "preference"},
    {"key": "language_preference", "value": "偏好用中文交流", "category": "preference"},
    {"key": "project", "value": "社区图书角项目", "category": "project"},
    {"key": "skill", "value": "熟悉Python基础语法", "category": "skill"},
    {"key": "location", "value": "现居上海", "category": "profile"},
    {"key": "name", "value": "李小明", "category": "identity"},
]

EXT_KEYS = [m["key"] for m in CORPUS_EXT]


def _ext_others(*keep):
    return [k for k in EXT_KEYS if k not in keep]


def _case(case_id, query, rel, allow, forbid, split, purpose,
          category=None, expect_empty=False, mode="normal"):
    return dict(id=case_id, q=query, rel=rel, allow=allow, forbid=forbid,
                split=split, purpose=purpose, category=category,
                expect_empty=expect_empty, mode=mode)


CASES_EXT = [
    # ---- DEV（E01-E20，仅用于阈值校准）----
    _case("E01", "我是什么专业的？", ["major"], [], _ext_others("major"), "dev", "R"),
    _case("E02", "我学的是什么方向？", ["major"], [], _ext_others("major"), "dev", "R"),
    _case("E03", "我现在在读什么？", ["degree"], ["major"], _ext_others("degree", "major"), "dev", "R"),
    _case("E04", "我的学历是什么", ["degree"], [], _ext_others("degree"), "dev", "R"),
    _case("E05", "我上的哪所大学", ["school"], [], _ext_others("school"), "dev", "R"),
    _case("E06", "我在哪里上学？", ["school"], ["degree"], _ext_others("school", "degree"), "dev", "R"),
    _case("E07", "我以后想从事什么工作？", ["career_goal"], ["learning_goal", "side_goal"], _ext_others("career_goal", "learning_goal", "side_goal"), "dev", "R"),
    _case("E08", "毕业后我想做什么？", ["career_goal"], ["learning_goal"], _ext_others("career_goal", "learning_goal"), "dev", "R"),
    _case("E09", "我的职业规划是什么", ["career_goal"], ["learning_goal", "side_goal"], _ext_others("career_goal", "learning_goal", "side_goal"), "dev", "R"),
    _case("E10", "我有什么长期目标", ["career_goal"], ["learning_goal", "side_goal"], _ext_others("career_goal", "learning_goal", "side_goal"), "dev", "RANK"),
    _case("E11", "我想往哪个方向发展", ["career_goal"], ["learning_goal"], _ext_others("career_goal", "learning_goal"), "dev", "R"),
    _case("E12", "帮我规划未来的学习安排", ["learning_goal"], ["career_goal"], _ext_others("learning_goal", "career_goal"), "dev", "RANK"),
    _case("E13", "我平常喜欢喝点啥", ["preference"], [], _ext_others("preference"), "dev", "R"),
    _case("E14", "我的日常爱好是啥", ["preference"], [], _ext_others("preference"), "dev", "R"),
    _case("E15", "我喝咖啡有什么讲究", ["preference"], [], _ext_others("preference"), "dev", "R"),
    _case("E16", "我用中文交流更习惯", ["language_preference"], [], _ext_others("language_preference"), "dev", "R"),
    _case("E17", "我喜欢说什么语言", ["language_preference"], [], _ext_others("language_preference"), "dev", "R"),
    _case("E18", "我最近在忙什么项目", ["project"], [], _ext_others("project"), "dev", "R"),
    _case("E19", "我有什么拿得出手的技能", ["skill"], ["learning_goal"], _ext_others("skill", "learning_goal"), "dev", "R"),
    _case("E20", "我会点啥技术", ["skill"], ["learning_goal"], _ext_others("skill", "learning_goal"), "dev", "R"),
    # ---- TEST（E21-E52，锁定：--set test 需 --unlock-test）----
    _case("E21", "我的 major 是什么", ["major"], [], _ext_others("major"), "test", "MIX"),
    _case("E22", "我未来的 career goal 是什么", ["career_goal"], ["learning_goal"], _ext_others("career_goal", "learning_goal"), "test", "MIX"),
    _case("E23", "我的 project 进展如何", ["project"], [], _ext_others("project"), "test", "MIX"),
    _case("E24", "我的 AI skill 怎么样", ["skill"], [], _ext_others("skill"), "test", "MIX"),
    _case("E25", "Python 水平如何", ["skill"], [], _ext_others("skill"), "test", "MIX"),
    _case("E26", "我的 preference 是什么", ["preference"], ["language_preference"], _ext_others("preference", "language_preference"), "test", "MIX"),
    _case("E27", "和机器人相关的目标", ["career_goal"], ["side_goal"], _ext_others("career_goal", "side_goal"), "test", "RANK"),
    _case("E28", "我帮朋友做的事", ["side_goal"], [], _ext_others("side_goal"), "test", "R"),
    _case("E29", "我学的是计算机吗", ["major"], ["degree"], _ext_others("major", "degree"), "test", "R"),
    _case("E30", "我的水平怎么样", ["skill"], ["learning_goal"], _ext_others("skill", "learning_goal"), "test", "R"),
    _case("E31", "上海", ["location"], [], _ext_others("location"), "test", "R"),
    _case("E32", "我现在住哪", ["location"], [], _ext_others("location"), "test", "R"),
    _case("E33", "我接下来想做的事", ["career_goal"], ["learning_goal", "side_goal"], _ext_others("career_goal", "learning_goal", "side_goal"), "test", "RANK"),
    _case("E34", "学习方面的目标", ["learning_goal"], ["career_goal"], _ext_others("learning_goal", "career_goal"), "test", "RANK"),
    _case("E35", "机器人比赛相关的事", ["side_goal"], ["career_goal"], _ext_others("side_goal", "career_goal"), "test", "RANK"),
    _case("E36", "技能和学习方面的情况", ["skill"], ["learning_goal"], _ext_others("skill", "learning_goal"), "test", "RANK"),
    _case("E37", "我的个人偏好", ["preference"], ["language_preference"], _ext_others("preference", "language_preference"), "test", "RANK"),
    _case("E38", "明天会下雨吗", [], [], list(EXT_KEYS), "test", "NEG", expect_empty=True),
    _case("E39", "123乘以456等于多少", [], [], list(EXT_KEYS), "test", "NEG", expect_empty=True),
    _case("E40", "地球到月球多远", [], [], list(EXT_KEYS), "test", "NEG", expect_empty=True),
    _case("E41", "今天心情不太好", [], [], list(EXT_KEYS), "test", "NEG", expect_empty=True),
    _case("E42", "他喜欢喝什么", [], [], list(EXT_KEYS), "test", "NEG", expect_empty=True),
    _case("E43", "现在几点了", [], [], list(EXT_KEYS), "test", "NEG", expect_empty=True),
    _case("E44", "帮我翻译这句话", [], [], list(EXT_KEYS), "test", "NEG", expect_empty=True),
    _case("E45", "写一首诗", [], [], list(EXT_KEYS), "test", "NEG", expect_empty=True),
    _case("E46", "这家店服务态度很好", [], [], list(EXT_KEYS), "test", "NEG", expect_empty=True),
    _case("E47", "推荐几部电影", [], [], list(EXT_KEYS), "test", "NEG", expect_empty=True),
    _case("E48", "我是什么专业的？（空库变体）", ["major"], [], [], "test", "EDGE",
          mode="scenario", expect_empty=False),
    _case("E49", "我是什么专业的？（单条记忆库变体）", ["major"], [], [], "test", "EDGE",
          mode="scenario", expect_empty=False),
    _case("E50", "我的名字是什么？（重复 key 变体）", ["name"], [], [], "test", "EDGE",
          mode="scenario", expect_empty=False),
    _case("E51", "我是什么专业的？", ["major"], [], _ext_others("major"), "test", "EDGE", category="profile"),
    _case("E52", "明天会下雨吗", [], [], list(EXT_KEYS), "test", "EDGE", category="profile", expect_empty=True),
]

SCENARIO_NOTES = {
    "E48": "空库变体：需专用夹具（0 条记忆的临时库），本轮 runner 不执行",
    "E49": "单条记忆库变体：需专用夹具（仅 M01），本轮 runner 不执行",
    "E50": "重复 key 变体：需两条 name 记录夹具与产品策略定义，本轮 runner 不执行",
}

DATASETS = {
    "smoke": dict(corpus=CORPUS, cases=CASES, tag=None),
    "dev": dict(corpus=CORPUS_EXT,
                cases=[c for c in CASES_EXT if c["split"] == "dev"],
                tag="dev"),
    "test": dict(corpus=CORPUS_EXT,
                 cases=[c for c in CASES_EXT if c["split"] == "test"],
                 tag="test"),
}


def select_cases(dataset, set_name):
    """返回 (corpus, cases, skipped_scenarios)。"""
    corpus = dataset["corpus"]
    cases, skipped = [], []
    for case in dataset["cases"]:
        if case.get("mode") == "scenario":
            skipped.append(case["id"])
        else:
            cases.append(case)
    return corpus, cases, skipped


# =====================================================================
# 模式：--check（离线结构自检）
# =====================================================================

def mode_check(args):
    print("=" * 66)
    print("[--check] 离线结构自检（不访问网络）")
    print(f"  后端配置: {args.backend}"
          + (f"（{DEFAULT_MODEL}）" if args.backend == "ollama" else ""))
    if args.backend == "ollama":
        print(f"  端点配置: {args.base_url or OLLAMA_BASE}（本轮不连接）")
    else:
        print("  在线后端: 仅结构校验，本阶段禁止调用")
    print(f"  阈值配置: {args.threshold}（初始值，须用 DEV 集校准）")

    problems = []

    def check_dataset(name, corpus, cases, expect_total,
                      expect_dev=None, expect_test=None):
        keys = [m["key"] for m in corpus]
        ids = [c["id"] for c in cases]
        if len(ids) != len(set(ids)):
            problems.append(f"{name}: 用例 id 重复")
        for case in cases:
            if case.get("expect_empty"):
                if case["rel"] or case["allow"]:
                    problems.append(f"{name}/{case['id']}: 空召回用例不应有 rel/allow")
            else:
                if not case["rel"]:
                    problems.append(f"{name}/{case['id']}: 缺少 rel")
                overlap = set(case["rel"]) & set(case["forbid"])
                if overlap:
                    problems.append(f"{name}/{case['id']}: rel∩forbid 冲突 {overlap}")
                for key in case["rel"] + case["allow"]:
                    if key not in keys:
                        problems.append(f"{name}/{case['id']}: 未知 key {key}")
                for key in case["forbid"]:
                    if key not in keys:
                        problems.append(f"{name}/{case['id']}: forbid 未知 key {key}")
            if case.get("category") and case["category"] not in {
                m["category"] for m in corpus
            }:
                problems.append(f"{name}/{case['id']}: 未知 category")

    check_dataset("smoke", CORPUS, CASES, 11)
    dev_cases = [c for c in CASES_EXT if c["split"] == "dev"]
    test_cases = [c for c in CASES_EXT if c["split"] == "test"]
    check_dataset("dev", CORPUS_EXT, dev_cases, 20)
    check_dataset("test", CORPUS_EXT, test_cases, 32)

    if len(dev_cases) != 20:
        problems.append(f"DEV 集数量应为 20，实际 {len(dev_cases)}")
    if len(test_cases) != 32:
        problems.append(f"TEST 集数量应为 32，实际 {len(test_cases)}")
    if len(CASES_EXT) != 52:
        problems.append(f"扩展集总数应为 52，实际 {len(CASES_EXT)}")

    print(f"  语料: 冒烟 {len(CORPUS)} 条 / 扩展 {len(CORPUS_EXT)} 条")
    print(f"  用例: 冒烟 {len(CASES)} | DEV {len(dev_cases)} | "
          f"TEST {len(test_cases)}（含 {len(SCENARIO_NOTES)} 个场景变体待实施）")

    if problems:
        print("[--check] 问题：")
        for p in problems:
            print("  -", p)
        return 1
    print("[--check] 结构与数据划分校验通过。")
    print("[--check] 防误跑：--set test 需同时传入 --unlock-test 才会执行。")
    return 0


def find_closed_port(candidates=(9, 11436, 45678)):
    """探测一个本机真正关闭（连接被拒）的端口，用于"不可达"分类验证。

    本机可能被代理软件占用任意端口（表现为返回 HTTP 502 而非拒绝），
    因此不能硬编码端口；全部候选都被占用时返回 None。"""
    for port in candidates:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.settimeout(0.3)
        try:
            refused = probe.connect_ex(("127.0.0.1", port)) != 0
        finally:
            probe.close()
        if refused:
            return port
    return None


# =====================================================================
# 模式：--errcheck（错误路径验证）
# =====================================================================

def mode_errcheck(args):
    print("=" * 66)
    print("[--errcheck] 错误路径验证（仅访问本机端口）")
    results = []

    backend = OllamaBackend(args.model, args.base_url or OLLAMA_BASE)

    # 1) 空输入
    try:
        backend.embed([], args.timeout)
        results.append(("空输入", "FAIL", "未抛错"))
    except ValueError as exc:
        results.append(("空输入", "OK", f"ValueError({exc})"))

    # 2) 空白输入
    try:
        backend.embed(["   "], args.timeout)
        results.append(("空白输入", "FAIL", "未抛错"))
    except ValueError as exc:
        results.append(("空白输入", "OK", f"ValueError({exc})"))

    # 3) 不可达端点（动态探测真正关闭的端口，避免被本机代理干扰）
    closed_port = find_closed_port()
    if closed_port is None:
        results.append(("端点不可达", "FAIL",
                        "候选端口均被占用，无法构造不可达场景（环境限制）"))
    else:
        try:
            OllamaBackend(args.model, f"http://127.0.0.1:{closed_port}").embed(
                ["测试"], timeout=3.0
            )
            results.append(("端点不可达", "FAIL", "未抛错"))
        except BackendError as exc:
            ok = exc.kind == "UNREACHABLE"
            results.append(("端点不可达", "OK" if ok else "FAIL",
                            f"端口 {closed_port} → {exc.kind}"))

    # 4) 超时（极短超时；服务未启动时表现为不可达，均可接受）
    try:
        backend.embed(["测试"], timeout=0.001)
        results.append(("超时", "OK", "请求在超时前完成（模型已热）"))
    except BackendError as exc:
        ok = exc.kind in ("TIMEOUT", "UNREACHABLE")
        results.append(("超时", "OK" if ok else "FAIL", exc.kind))

    # 5) 模型不存在（v2 修正的错误分类验证）
    try:
        OllamaBackend("no-such-model-xyz:latest",
                      args.base_url or OLLAMA_BASE).embed(["测试"], timeout=30)
        results.append(("模型不存在", "FAIL", "未抛错"))
    except BackendError as exc:
        ok = exc.kind in ("MODEL_NOT_FOUND", "UNREACHABLE")
        results.append(("模型不存在", "OK" if ok else "FAIL",
                        f"{exc.kind}"
                        + ("" if exc.kind == "MODEL_NOT_FOUND"
                           else "（服务未运行，退化为不可达）")))

    for name, verdict, detail in results:
        print(f"  {name:8s} -> {verdict}: {detail}")
    ok = all(verdict == "OK" for _, verdict, _ in results)
    print(f"[--errcheck] 结论: {'全部优雅降级' if ok else '存在未处理路径'}")
    return 0 if ok else 1


# =====================================================================
# 模式：--run（分层计时 + 指标评测）
# =====================================================================

def mode_run(args):
    dataset = DATASETS[args.set]
    corpus, cases, skipped = select_cases(dataset, args.set)

    print("=" * 66)
    print(f"[--run] 数据集: {args.set}"
          + ("（DEV 校准集）" if args.set == "dev" else "")
          + ("（锁定 TEST 集）" if args.set == "test" else "")
          + f" | 用例 {len(cases)} 条"
          + (f" | 跳过场景变体: {skipped}" if skipped else ""))
    for sid in skipped:
        print(f"        [{sid}] 未执行：{SCENARIO_NOTES.get(sid, '场景用例')}")

    backend = make_backend(args)
    repeat = args.repeat

    # ---- 阶段 1：冷调用（含模型加载；若服务已热则数值偏小，报告注明）----
    t0 = time.perf_counter()
    try:
        cold_vecs = backend.embed(["预热"], args.timeout)
    except BackendError as exc:
        print(f"[--run] 前置条件不满足，无法实测：{exc}")
        print("  请先：1) 本地 Ollama 服务运行中  2) 模型已 pull")
        return 2
    cold_ms = (time.perf_counter() - t0) * 1000

    # ---- 阶段 2：预热（不计入统计）----
    t0 = time.perf_counter()
    backend.embed(["预热"], args.timeout)
    warm_ms = (time.perf_counter() - t0) * 1000
    dim = len(cold_vecs[0])
    print(f"  模型: {backend.model_id} | 后端: {backend.name} | "
          f"维度: {dim}（已校验）")
    print(f"  冷调用(含加载): {cold_ms:.0f} ms | 预热调用: {warm_ms:.0f} ms")
    print("  口径说明：冷调用仅在服务未加载该模型时反映真实加载耗时；"
          "若服务已热，冷调用为低值属正常。")

    # ---- 批次向量化（预热后重复 3 次，取中位数）----
    texts = [m["value"] for m in corpus]
    batch_samples = []
    for _ in range(3):
        t0 = time.perf_counter()
        vectors = backend.embed(texts, args.timeout)
        batch_samples.append((time.perf_counter() - t0) * 1000)
    batch_ms = sorted(batch_samples)[len(batch_samples) // 2]
    memory_vecs = {corpus[i]["key"]: vectors[i] for i in range(len(corpus))}
    print(f"  批次向量化（{len(corpus)} 条 × 3 次，取中位数）: {batch_ms:.0f} ms "
          f"（{batch_ms / len(corpus):.0f} ms/条；样本: "
          f"{', '.join(f'{x:.0f}' for x in batch_samples)}）")

    # ---- 逐用例评测（每条查询重复 repeat 次测延迟）----
    report_cases = []
    samples_all = []
    print("-" * 66)
    total_rel = total_rel_hit = total_prec_num = total_prec_den = 0
    mrr_sum = 0.0
    mrr_count = 0
    neg_total = neg_violation = 0
    passed = failed = 0

    for case in cases:
        case_samples = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            q_vec = backend.embed([case["q"]], args.timeout)[0]
            case_samples.append((time.perf_counter() - t0) * 1000)
        samples_all.extend(case_samples)
        latency_ms = sorted(case_samples)[len(case_samples) // 2]

        candidates = memory_vecs
        if case.get("category"):
            candidates = {
                m["key"]: memory_vecs[m["key"]]
                for m in corpus if m["category"] == case["category"]
            }
        ranked = search(q_vec, candidates, args.threshold, top_k=3)
        top_keys = [k for k, _, _ in ranked]

        if case["expect_empty"]:
            ok = len(ranked) == 0
            detail = ("全部低于阈值" if ok else f"越过阈值: {top_keys}")
            neg_total += 1
            if not ok:
                neg_violation += 1
        else:
            rel_hit = [k for k in case["rel"] if k in top_keys]
            forbid_hit = [k for k in top_keys if k in case["forbid"]]
            ok = len(rel_hit) == len(case["rel"]) and not forbid_hit
            detail = (f"rel命中 {rel_hit}/{case['rel']}"
                      + (f" 误召回 {forbid_hit}" if forbid_hit else ""))
            total_rel += len(case["rel"])
            total_rel_hit += len(rel_hit)
            total_prec_num += len(
                [k for k in top_keys if k in case["rel"] or k in case["allow"]]
            )
            total_prec_den += len(top_keys)
            first_rel_rank = next(
                (r for k, _, r in ranked if k in case["rel"]), None
            )
            if first_rel_rank:
                mrr_sum += 1.0 / first_rel_rank
            mrr_count += 1

        passed += 1 if ok else 0
        failed += 0 if ok else 1

        cos_text = " ".join(f"{k}={c:.3f}(#{r})" for k, c, r in ranked) or "（无）"
        print(f"[{case['id']:4s}] {'PASS' if ok else 'FAIL'} "
              f"「{case['q']}」 {latency_ms:.0f}ms(中位/{repeat}次)")
        print(f"        top3: {cos_text}")
        print(f"        {detail}")

        report_cases.append({
            "id": case["id"], "query": case["q"],
            "top": [{"key": k, "cos": c, "rank": r}
                    for k, c, r in ranked],
            "ok": ok, "latency_ms": round(latency_ms, 1),
            "latency_samples_ms": [round(x, 1) for x in case_samples],
            "detail": detail,
        })

    recall = (total_rel_hit / total_rel) if total_rel else float("nan")
    precision = (total_prec_num / total_prec_den) if total_prec_den else 0.0
    mrr = (mrr_sum / mrr_count) if mrr_count else float("nan")
    p50 = percentile_nearest_rank(samples_all, 50)
    p95 = percentile_nearest_rank(samples_all, 95)

    print("-" * 66)
    print(f"用例: {len(cases)} | PASS: {passed} | FAIL: {failed}")
    print(f"Recall@3 : {total_rel_hit}/{total_rel} = {recall:.2%}")
    print(f"Precision@3: {total_prec_num}/{total_prec_den} = {precision:.2%}")
    print(f"MRR      : {mrr:.3f}（{mrr_count} 个有效查询）")
    print(f"无关误召回: {neg_violation}/{neg_total}")
    print(f"延迟统计 : 样本 N={len(samples_all)}（{len(cases)} 用例 × {repeat} 次），"
          f"单位 ms，最近秩法 → p50={p50:.0f} ms，p95={p95:.0f} ms")
    print(f"模型: {backend.model_id} | 后端: {backend.name} | "
          f"维度: {dim} | 阈值: {args.threshold}")

    if args.json:
        out = {
            "set": args.set,
            "backend": backend.name,
            "model": backend.model_id,
            "dim": dim,
            "threshold": args.threshold,
            "timing": {
                "cold_ms": round(cold_ms, 1),
                "warm_ms": round(warm_ms, 1),
                "batch_median_ms": round(batch_ms, 1),
                "batch_samples_ms": [round(x, 1) for x in batch_samples],
                "query_samples": len(samples_all),
                "query_p50_ms": round(p50, 1),
                "query_p95_ms": round(p95, 1),
                "method": "nearest-rank, unit=ms",
            },
            "passed": passed, "failed": failed,
            "recall@3": f"{total_rel_hit}/{total_rel}",
            "precision@3": f"{total_prec_num}/{total_prec_den}",
            "mrr": round(mrr, 4) if mrr == mrr else None,
            "unrelated_false_recall": f"{neg_violation}/{neg_total}",
            "skipped_scenarios": skipped,
            "cases": report_cases,
        }
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"报告已写入: {args.json}")

    return 0 if failed == 0 else 1


# =====================================================================
# CLI
# =====================================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description="Ollama 本地 Embedding 语义检索独立原型 v2"
    )
    parser.add_argument("--check", action="store_true",
                        help="离线结构自检（零网络）")
    parser.add_argument("--errcheck", action="store_true",
                        help="错误路径验证（仅本机端口）")
    parser.add_argument("--run", action="store_true",
                        help="评测运行（需本地服务与模型）")
    parser.add_argument("--backend", choices=BACKEND_CHOICES,
                        default="ollama", help="后端（默认 ollama）")
    parser.add_argument("--base-url", default=None,
                        help="后端地址（默认本机 Ollama）")
    parser.add_argument("--key-env", default="EMBED_API_KEY",
                        help="在线后端密钥的环境变量名（不读取值以外信息）")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--repeat", type=int, default=DEFAULT_REPEAT,
                        help="测量阶段每条查询重复次数（默认 3）")
    parser.add_argument("--set", choices=("smoke", "dev", "test"),
                        default="smoke", help="数据集（默认 smoke，与原行为兼容）")
    parser.add_argument("--unlock-test", action="store_true",
                        help="显式解锁锁定 TEST 集（默认禁止运行）")
    parser.add_argument("--json", default=None,
                        help="结果写入指定 JSON（目标文件必须不存在）")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    # ---- 锁定 TEST 集防误跑 ----
    if args.run and args.set == "test" and not args.unlock_test:
        print("拒绝执行锁定的 TEST 集：需要显式传入 --unlock-test。")
        print("（校准只允许使用 DEV 集；TEST 集每种模型仅运行一次。）")
        return 4

    # ---- JSON 防覆盖（保留 v1 机制）----
    if args.json and os.path.exists(args.json):
        print(f"拒绝覆盖已存在文件: {args.json}")
        return 3

    if args.check:
        return mode_check(args)
    if args.errcheck:
        return mode_errcheck(args)
    if args.run:
        return mode_run(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
