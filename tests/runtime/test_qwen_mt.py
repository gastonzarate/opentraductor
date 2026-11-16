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
    with pytest.raises(FileNotFoundError):
        C.qwen_model_dir(C.PipelineConfig(mt_engine="qwen"))
    monkeypatch.setenv("OPENTRADUCTOR_QWEN_DIR", str(tmp_path))
    assert C.qwen_model_dir(C.PipelineConfig(mt_engine="qwen")) == tmp_path
    assert C.qwen_model_dir(C.PipelineConfig(mt_engine="qwen", qwen_model_dir="/x")) == C.Path("/x")


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
