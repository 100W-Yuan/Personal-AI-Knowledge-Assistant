# -*- coding: utf-8 -*-
"""semantic_prototype.py（v2）专属单元测试——全部离线，不依赖 Ollama。

覆盖范围（第四阶段第一步要求 9）：
  1. 后端接口与构造（含在线预留后端的缺 Key 配置错误）
  2. 输入校验先于网络（空输入不发起请求）
  3. 维度校验（同批不一致 / 跨调用变化 均中止）
  4. 错误分类（404→MODEL_NOT_FOUND、其余 404、401/403、429、
     不可达、超时、非法 JSON）
  5. 余弦相似度与检索排序/阈值/TopK
  6. p50/p95 最近秩法统计
  7. 指标计算（通过假后端走完整 mode_run 管道）
  8. 数据集划分隔离（E01-E20=DEV、E21-E52=TEST、冒烟独立）
  9. JSON 防覆盖与锁定 TEST 防误跑
"""

import importlib.util
import io
import socket
import sys
import urllib.error
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = PROJECT_ROOT / "辅助脚本" / "semantic_prototype.py"

spec = importlib.util.spec_from_file_location("semantic_prototype", SCRIPT_PATH)
sp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sp)


# =====================================================================
# 1. 后端接口与构造
# =====================================================================

class TestBackendConstruction:

    def test_backend_arg_choices_and_default(self):
        parser = sp.build_parser()
        args = parser.parse_args([])
        assert args.backend == "ollama"  # 默认保持原行为兼容

    def test_unknown_backend_rejected(self):
        parser = sp.build_parser()
        with pytest.raises(SystemExit):  # argparse choices 拒绝未知值
            parser.parse_args(["--backend", "nope"])

    def test_make_backend_ollama(self):
        parser = sp.build_parser()
        args = parser.parse_args(["--model", "m1"])
        backend = sp.make_backend(args)
        assert isinstance(backend, sp.OllamaBackend)
        assert backend.name == "ollama"
        assert backend.model_id == "m1"

    def test_openai_compat_requires_key_env(self, monkeypatch):
        monkeypatch.delenv("MY_EMBED_KEY", raising=False)
        with pytest.raises(sp.BackendConfigError) as excinfo:
            sp.OpenAICompatBackend("m", "https://example.invalid/v1",
                                   api_key_env="MY_EMBED_KEY")
        assert "MY_EMBED_KEY" in str(excinfo.value)
        assert "CONFIG" in str(excinfo.value)

    def test_openai_compat_construction_makes_no_request(
        self, monkeypatch
    ):
        monkeypatch.setenv("MY_EMBED_KEY", "sk-test-not-real")
        monkeypatch.setattr(
            sp, "_post_json",
            lambda *a, **k: pytest.fail("构造阶段不得发起网络请求"),
        )
        backend = sp.OpenAICompatBackend(
            "m", "https://example.invalid/v1", api_key_env="MY_EMBED_KEY"
        )
        assert backend.name == "openai-compat"


# =====================================================================
# 2/3. 输入校验与维度校验
# =====================================================================

class TestValidation:

    def test_empty_input_raises_before_network(self, monkeypatch):
        monkeypatch.setattr(
            sp, "_post_json",
            lambda *a, **k: pytest.fail("输入校验应在网络请求之前"),
        )
        backend = sp.OllamaBackend("m")
        with pytest.raises(ValueError):
            backend.embed([])
        with pytest.raises(ValueError):
            backend.embed(["   "])

    def test_din_mismatch_within_batch(self, monkeypatch):
        monkeypatch.setattr(
            sp, "_post_json",
            lambda *a, **k: {"embeddings": [[1.0, 0.0], [1.0, 0.0, 0.0]]},
        )
        backend = sp.OllamaBackend("m")
        with pytest.raises(sp.BackendError) as excinfo:
            backend.embed(["a", "b"])
        assert excinfo.value.kind == "DIM_MISMATCH"

    def test_dim_change_across_calls(self, monkeypatch):
        responses = iter([
            {"embeddings": [[1.0, 0.0]]},
            {"embeddings": [[1.0, 0.0, 0.0]]},
        ])
        monkeypatch.setattr(
            sp, "_post_json", lambda *a, **k: next(responses)
        )
        backend = sp.OllamaBackend("m")
        backend.embed(["a"])
        with pytest.raises(sp.BackendError) as excinfo:
            backend.embed(["b"])
        assert excinfo.value.kind == "DIM_MISMATCH"
        assert backend.dim == 2  # 首次探测值被保留


# =====================================================================
# 4. 错误分类
# =====================================================================

def _http_error(code, body=b""):
    return urllib.error.HTTPError(
        "http://localhost", code, "err", {}, io.BytesIO(body)
    )


class _RaisingOpener:
    def __init__(self, exc):
        self._exc = exc

    def open(self, *a, **k):
        raise self._exc


class TestErrorClassification:

    @pytest.mark.parametrize("code,body,expected", [
        (404, b'{"error":"model \'x\' not found, try pulling it first"}',
         "MODEL_NOT_FOUND"),
        (404, b'{"error":"path not found"}', "HTTP_404"),
        (401, b'{"error":"unauthorized"}', "AUTH"),
        (403, b'{"error":"forbidden"}', "AUTH"),
        (429, b'{"error":"rate limited"}', "RATE_LIMIT"),
        (500, b'{"error":"boom"}', "HTTP_500"),
    ])
    def test_http_error_mapping(self, monkeypatch, code, body, expected):
        monkeypatch.setattr(
            sp, "_make_opener",
            lambda bypass: _RaisingOpener(_http_error(code, body)),
        )
        with pytest.raises(sp.BackendError) as excinfo:
            sp._post_json("http://localhost/x", {}, 5)
        assert excinfo.value.kind == expected

    def test_unreachable(self, monkeypatch):
        refused = urllib.error.URLError(
            ConnectionRefusedError(61, "refused")
        )
        monkeypatch.setattr(
            sp, "_make_opener", lambda bypass: _RaisingOpener(refused)
        )
        with pytest.raises(sp.BackendError) as excinfo:
            sp._post_json("http://localhost/x", {}, 5)
        assert excinfo.value.kind == "UNREACHABLE"

    def test_timeout_socket(self, monkeypatch):
        monkeypatch.setattr(
            sp, "_make_opener",
            lambda bypass: _RaisingOpener(socket.timeout("timed out")),
        )
        with pytest.raises(sp.BackendError) as excinfo:
            sp._post_json("http://localhost/x", {}, 5)
        assert excinfo.value.kind == "TIMEOUT"

    def test_timeout_wrapped_in_urlerror(self, monkeypatch):
        wrapped = urllib.error.URLError(socket.timeout("timed out"))
        monkeypatch.setattr(
            sp, "_make_opener", lambda bypass: _RaisingOpener(wrapped)
        )
        with pytest.raises(sp.BackendError) as excinfo:
            sp._post_json("http://localhost/x", {}, 5)
        assert excinfo.value.kind == "TIMEOUT"

    def test_bad_json_response(self, monkeypatch):
        class _Resp:
            def read(self):
                return b"not-json"

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        class _OkOpener:
            def open(self, *a, **k):
                return _Resp()

        monkeypatch.setattr(
            sp, "_make_opener", lambda bypass: _OkOpener()
        )
        with pytest.raises(sp.BackendError) as excinfo:
            sp._post_json("http://localhost/x", {}, 5)
        assert excinfo.value.kind == "BAD_RESPONSE"

    def test_ollama_bypasses_global_proxy(self, monkeypatch):
        """本机存在全局代理环境变量：本地后端必须 bypass_proxy=True。"""
        captured = {}

        def fake_post(url, payload, timeout, headers=None,
                      bypass_proxy=False):
            captured["bypass"] = bypass_proxy
            return {"embeddings": [[1.0, 0.0]]}

        monkeypatch.setattr(sp, "_post_json", fake_post)
        sp.OllamaBackend("m").embed(["x"])
        assert captured["bypass"] is True

    def test_openai_compat_keeps_env_proxy(self, monkeypatch):
        """在线后端默认沿用环境代理（False）。"""
        monkeypatch.setenv("MY_KEY", "sk-x")
        captured = {}

        def fake_post(url, payload, timeout, headers=None,
                      bypass_proxy=False):
            captured["bypass"] = bypass_proxy
            captured["auth"] = headers.get("Authorization", "")
            return {"data": [{"index": 0, "embedding": [1.0, 0.0]}]}

        monkeypatch.setattr(sp, "_post_json", fake_post)
        sp.OpenAICompatBackend("m", "https://example.invalid",
                               api_key_env="MY_KEY").embed(["x"])
        assert captured["bypass"] is False
        assert captured["auth"].startswith("Bearer ")


# =====================================================================
# 5. 余弦与检索
# =====================================================================

class TestCosineAndSearch:

    def test_cosine_known_values(self):
        assert sp.cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
        assert sp.cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
        assert sp.cosine([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_search_order_threshold_topk(self):
        vectors = {
            "a": [1.0, 0.0],
            "b": [0.8, 0.6],   # cos ≈ 0.8
            "c": [0.0, 1.0],   # cos = 0
        }
        ranked = sp.search([1.0, 0.0], vectors, threshold=0.5, top_k=3)
        assert [k for k, _, _ in ranked] == ["a", "b"]
        assert [r for _, _, r in ranked] == [1, 2]

        limited = sp.search([1.0, 0.0], vectors, threshold=0.0, top_k=2)
        assert len(limited) == 2


# =====================================================================
# 6. p50/p95 统计（最近秩法）
# =====================================================================

class TestPercentile:

    def test_nearest_rank_examples(self):
        samples = [10, 20, 30, 40]
        assert sp.percentile_nearest_rank(samples, 50) == 20   # ceil(2)=2 → 20
        assert sp.percentile_nearest_rank(samples, 95) == 40   # ceil(3.8)=4 → 40
        assert sp.percentile_nearest_rank(samples, 100) == 40
        assert sp.percentile_nearest_rank([5], 95) == 5

    def test_percentile_rejects_bad_input(self):
        with pytest.raises(ValueError):
            sp.percentile_nearest_rank([], 50)
        with pytest.raises(ValueError):
            sp.percentile_nearest_rank([1], 0)


# =====================================================================
# 7. 指标计算（假后端走完整 mode_run 管道）
# =====================================================================

class _FakeBackend(sp.EmbeddingBackend):
    name = "fake"

    def __init__(self, mapping):
        super().__init__("fake-model")
        self._mapping = mapping

    def embed(self, texts, timeout=0):
        return [self._mapping[t] for t in texts]


class TestMetricsPipeline:

    def _run_fake(self, monkeypatch, capsys):
        mapping = {
            "预热": [1.0, 0.0],
            "记忆A": [1.0, 0.0],
            "记忆B": [0.0, 1.0],
            "查询A": [1.0, 0.0],
            "查询B": [0.0, 1.0],
            "无关查询": [-1.0, 0.0],
        }
        fake = _FakeBackend(mapping)
        monkeypatch.setattr(sp, "make_backend", lambda args: fake)

        tiny = dict(
            corpus=[
                {"key": "a", "value": "记忆A", "category": "profile"},
                {"key": "b", "value": "记忆B", "category": "goal"},
            ],
            cases=[
                dict(id="T1", q="查询A", rel=["a"], allow=[],
                     forbid=["b"], category=None, expect_empty=False),
                dict(id="T2", q="查询B", rel=["b"], allow=[],
                     forbid=["a"], category=None, expect_empty=False),
                dict(id="T3", q="无关查询", rel=[], allow=[],
                     forbid=["a", "b"], category=None, expect_empty=True),
            ],
            tag=None,
        )
        monkeypatch.setitem(sp.DATASETS, "smoke", tiny)

        parser = sp.build_parser()
        args = parser.parse_args(
            ["--run", "--set", "smoke", "--repeat", "1"]
        )
        rc = sp.mode_run(args)
        out = capsys.readouterr().out
        return rc, out

    def test_metrics_are_correct(self, monkeypatch, capsys):
        rc, out = self._run_fake(monkeypatch, capsys)
        assert rc == 0
        assert "Recall@3 : 2/2 = 100.00%" in out
        assert "Precision@3: 2/2 = 100.00%" in out
        assert "MRR      : 1.000" in out
        assert "无关误召回: 0/1" in out
        assert "p50=" in out and "p95=" in out

    def test_run_fails_when_expectation_broken(self, monkeypatch, capsys):
        """负向验证：把无关查询向量改到与记忆A同向 → 误召回 → FAIL。"""
        mapping = {
            "预热": [1.0, 0.0],
            "记忆A": [1.0, 0.0],
            "记忆B": [0.0, 1.0],
            "查询A": [1.0, 0.0],
            "查询B": [0.0, 1.0],
            "无关查询": [1.0, 0.0],  # 故意制造误召回
        }
        fake = _FakeBackend(mapping)
        monkeypatch.setattr(sp, "make_backend", lambda args: fake)
        monkeypatch.setitem(sp.DATASETS, "smoke", dict(
            corpus=[
                {"key": "a", "value": "记忆A", "category": "profile"},
                {"key": "b", "value": "记忆B", "category": "goal"},
            ],
            cases=[
                dict(id="T3", q="无关查询", rel=[], allow=[],
                     forbid=["a", "b"], category=None, expect_empty=True),
            ],
            tag=None,
        ))
        parser = sp.build_parser()
        args = parser.parse_args(["--run", "--set", "smoke", "--repeat", "1"])
        rc = sp.mode_run(args)
        out = capsys.readouterr().out
        assert rc == 1  # 失败应返回非零
        assert "无关误召回: 1/1" in out


# =====================================================================
# 8. 数据集划分隔离
# =====================================================================

class TestDatasetSplit:

    def test_ext_dataset_totals_and_split(self):
        assert len(sp.CASES_EXT) == 52
        dev_ids = [c["id"] for c in sp.CASES_EXT if c["split"] == "dev"]
        test_ids = [c["id"] for c in sp.CASES_EXT if c["split"] == "test"]
        assert dev_ids == [f"E{i:02d}" for i in range(1, 21)]
        assert test_ids == [f"E{i:02d}" for i in range(21, 53)]
        all_ids = dev_ids + test_ids
        assert len(set(all_ids)) == 52

    def test_every_case_annotation_is_consistent(self):
        keys = set(sp.EXT_KEYS)
        for case in sp.CASES_EXT:
            if case.get("expect_empty"):
                assert not case["rel"] and not case["allow"]
            else:
                assert case["rel"], case["id"]
                assert not (set(case["rel"]) & set(case["forbid"])), case["id"]
                for key in case["rel"] + case["allow"] + case["forbid"]:
                    assert key in keys, (case["id"], key)

    def test_smoke_dataset_unchanged(self):
        assert len(sp.CASES) == 11
        assert len(sp.CORPUS) == 6


# =====================================================================
# 9. 防覆盖与锁定 TEST 防误跑
# =====================================================================

class TestGuards:

    def test_find_closed_port_returns_refused_port(self):
        port = sp.find_closed_port()
        if port is None:
            pytest.skip("候选端口均被占用（环境限制）")
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.settimeout(0.3)
        try:
            assert probe.connect_ex(("127.0.0.1", port)) != 0
        finally:
            probe.close()

    def test_json_refuses_overwrite(self, tmp_path, capsys):
        existing = tmp_path / "report.json"
        existing.write_text("{}", encoding="utf-8")
        rc = sp.main(["--run", "--json", str(existing)])
        assert rc == 3
        assert "拒绝覆盖" in capsys.readouterr().out

    def test_locked_test_requires_unlock(self, capsys):
        rc = sp.main(["--run", "--set", "test"])
        assert rc == 4
        assert "--unlock-test" in capsys.readouterr().out

    def test_check_mode_offline(self, capsys):
        rc = sp.main(["--check"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "DEV 20" in out and "TEST 32" in out
