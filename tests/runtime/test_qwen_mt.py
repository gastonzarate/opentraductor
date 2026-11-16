"""T-51: QwenMLX (mt_engine="qwen") con un mlx_lm falso: prompt, corte de línea, normalización y preflight."""

from __future__ import annotations

import sys
import types

import pytest

from runtime import components as K
from runtime import config as C
from runtime import preflight


class FakeTok:
    def __init__(self, template: str):
        self.chat_template = template
        self.calls = []

    def apply_chat_template(self, msgs, add_generation_prompt, tokenize, **kw):
        self.calls.append(kw)
        return f"<sys>{msgs[0]['content']}</sys><user>{msgs[1]['content']}</user>"


@pytest.fixture
def fake_mlx(monkeypatch):
    seen = {}

    def load(path):
        seen["path"] = path
        return "model", seen.setdefault("tok", FakeTok(seen.get("template", "")))

    def generate(model, tok, prompt, max_tokens, verbose):
        seen["prompt"], seen["max_tokens"] = prompt, max_tokens
        return seen.get("reply", "  You need to send it.\nNote: extra  ")

    mod = types.SimpleNamespace(load=load, generate=generate)
    monkeypatch.setitem(sys.modules, "mlx_lm", mod)
    return seen


def test_translate_first_line_and_budget(fake_mlx):
    mt = K.QwenMLX("/m/qwen")
    assert mt.translate("Tenés que mandarlo.") == "You need to send it."
    assert fake_mlx["path"] == "/m/qwen"
    assert fake_mlx["max_tokens"] == 3 * 3 + 16
    assert "Spanish:\nTenés que mandarlo." in fake_mlx["prompt"]
    assert mt.translate("   ") == ""


def test_hybrid_template_disables_thinking(fake_mlx):
    fake_mlx["template"] = "{% if enable_thinking %}...{% endif %}"
    mt = K.QwenMLX("/m/q17")
    mt.translate("hola")
    assert fake_mlx["tok"].calls[-1] == {"enable_thinking": False}


def test_instruct_template_gets_no_thinking_kwarg(fake_mlx):
    mt = K.QwenMLX("/m/q4")
    mt.translate("hola")
    assert fake_mlx["tok"].calls[-1] == {}


def test_normalize_applies_bug7_numbers(fake_mlx):
    K.QwenMLX("/m/q", normalize=True).translate("Son 120 mil, a las 19.")
    assert "120.000" in fake_mlx["prompt"] and "7 p. m." in fake_mlx["prompt"]


def test_qwen_dir_from_cfg_or_env(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENTRADUCTOR_QWEN_DIR", raising=False)
    monkeypatch.setenv("OPENTRADUCTOR_QWEN_ROOT", str(tmp_path / "root"))
    monkeypatch.setenv("OPENTRADUCTOR_QWEN_SIZE", "1.7b")
    assert C.qwen_model_dir(C.PipelineConfig(mt_engine="qwen")) == tmp_path / "root" / "qwen3-1.7b-4bit"
    monkeypatch.setenv("OPENTRADUCTOR_QWEN_DIR", str(tmp_path))
    assert C.qwen_model_dir(C.PipelineConfig(mt_engine="qwen")) == tmp_path
    assert C.qwen_model_dir(C.PipelineConfig(mt_engine="qwen", qwen_model_dir="/x")) == C.Path("/x")


def test_qwen_size_by_ram(monkeypatch):
    monkeypatch.delenv("OPENTRADUCTOR_QWEN_SIZE", raising=False)
    gb = 1024**3
    assert C.qwen_size(8 * gb) == "1.7b"
    assert C.qwen_size(16 * gb) == "4b"
    assert C.qwen_size(48 * gb) == "4b"
    assert C.qwen_size_for_ram(None) == "1.7b"
    monkeypatch.setenv("OPENTRADUCTOR_QWEN_SIZE", "4b")
    assert C.qwen_size(8 * gb) == "4b"
    monkeypatch.setenv("OPENTRADUCTOR_QWEN_SIZE", "0.6b")
    with pytest.raises(ValueError):
        C.qwen_size(8 * gb)


def test_mac_memsize_reads_hw_memsize():
    m = C.mac_memsize()
    assert m is None or m >= 4 * 1024**3


def test_mlx_cache_limit_default_and_env(monkeypatch):
    from runtime import pipeline as P

    monkeypatch.delenv("OPENTRADUCTOR_MLX_CACHE_MB", raising=False)
    assert C.mlx_cache_limit_bytes(C.PipelineConfig()) == 256_000_000
    assert C.mlx_cache_limit_bytes(C.PipelineConfig(mlx_cache_limit_mb=None)) is None
    monkeypatch.setenv("OPENTRADUCTOR_MLX_CACHE_MB", "-1")
    assert C.mlx_cache_limit_bytes(C.PipelineConfig()) is None
    monkeypatch.setenv("OPENTRADUCTOR_MLX_CACHE_MB", "128")
    seen = {}
    core = types.SimpleNamespace(set_cache_limit=lambda b: seen.setdefault("b", b))
    monkeypatch.setitem(sys.modules, "mlx", types.SimpleNamespace(core=core))
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    assert P._apply_mlx_cache_limit(C.PipelineConfig()) == 128_000_000 and seen["b"] == 128_000_000


def test_preflight_checks_qwen_not_marian(monkeypatch, tmp_path):
    for f in ("config.json", "model.safetensors", "tokenizer.json"):
        (tmp_path / f).write_text("{}")
    monkeypatch.setitem(sys.modules, "mlx_lm", types.SimpleNamespace(__spec__=object()))
    monkeypatch.setattr(preflight.importlib.util, "find_spec", lambda name: object())
    names = {c.name: c for c in preflight.run_checks(C.PipelineConfig(mt_engine="qwen", qwen_model_dir=str(tmp_path)))}
    assert "mt:marian-ct2" not in names and names["mt:qwen-mlx"].ok
    (tmp_path / "model.safetensors").unlink()
    names = {c.name: c for c in preflight.run_checks(C.PipelineConfig(mt_engine="qwen", qwen_model_dir=str(tmp_path)))}
    assert not names["mt:qwen-mlx"].ok


def test_default_engine_is_marian():
    assert C.PipelineConfig().mt_engine == "marian"
