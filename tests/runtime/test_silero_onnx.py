"""T-43: Silero VAD sin torch (onnxruntime + numpy) y Marian sin conversión en el dispositivo."""
from __future__ import annotations

import importlib.util
import sys

import numpy as np
import pytest

from runtime import components as K
from runtime import config as C

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("onnxruntime") is None or not C.silero_vad_onnx_path().exists(),
    reason="falta onnxruntime o silero_vad.onnx",
)


def _speechish(seconds: float, seed: int = 0) -> bytes:
    """Señal con armónicos y modulación de amplitud + ruido (determinista)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(16000 * seconds)) / 16000
    f0 = 120 + 30 * np.sin(2 * np.pi * 2 * t)
    ph = 2 * np.pi * np.cumsum(f0) / 16000
    x = sum(np.sin(k * ph) / k for k in range(1, 8))
    x = x * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)) * 0.3 + rng.normal(0, 0.01, t.size)
    return K.float_to_pcm(x.astype(np.float32))


def test_vad_runs_without_torch(monkeypatch):
    # Se resuelve la ruta ANTES de bloquear silero_vad (en desarrollo sale de ese paquete).
    monkeypatch.setenv("OPENTRADUCTOR_SILERO_ONNX", str(C.silero_vad_onnx_path()))
    # None en sys.modules hace que ``import torch`` falle con ImportError.
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "silero_vad", None)
    vad = K.SileroVAD()
    silence = vad.probs(bytes(2 * 16000))
    assert len(silence) == 16000 // K.VAD_WINDOW
    assert all(p < 0.2 for _, p in silence)
    assert all(0.0 <= p <= 1.0 for _, p in vad.probs(_speechish(1.0)))


def test_vad_chunking_does_not_change_probs():
    pcm = _speechish(1.0)
    whole = K.SileroVAD()
    a = whole.probs(pcm)
    split = K.SileroVAD()
    b = []
    consumed = 0
    for n in (1000, 3000, 5000, 7001, len(pcm)):  # tamaños arbitrarios, en bytes
        n = min(n, len(pcm) - consumed) & ~1
        chunk = pcm[consumed : consumed + n]
        b += [(consumed // 2 + off, p) for off, p in split.probs(chunk)]
        consumed += n
    assert [round(p, 5) for _, p in a] == [round(p, 5) for _, p in b][: len(a)]


def test_vad_reset_restores_initial_state():
    pcm = _speechish(0.5)
    vad = K.SileroVAD()
    first = vad.probs(pcm)
    vad.reset()
    assert vad.probs(pcm) == first


def test_vad_reset_states_keeps_pending_samples():
    # El pipeline llama a reset_states() al cerrar cada segmento (regresión: antes usaba vad.model).
    pcm = _speechish(0.5)
    vad = K.SileroVAD()
    vad.probs(pcm)
    vad.reset_states()
    assert not vad._state.any() and not vad._ctx.any()
    vad.probs(b"\x01\x00" * 100)  # menos de una ventana: queda como resto
    rest = vad._rest.copy()
    assert len(rest) > 0
    vad.reset_states()
    assert (vad._rest == rest).all()
    vad.reset()
    assert len(vad._rest) == 0


def test_vad_matches_torch_when_available():
    pytest.importorskip("torch")
    silero = pytest.importorskip("silero_vad")
    import torch

    jit = silero.load_silero_vad()
    pcm = _speechish(2.0, seed=1)
    x = K.pcm_to_float(pcm)
    ref = []
    with torch.inference_mode():
        for i in range(len(x) // K.VAD_WINDOW):
            w = torch.from_numpy(x[i * K.VAD_WINDOW : (i + 1) * K.VAD_WINDOW])
            ref.append(float(jit(w, 16000).item()))
    got = [p for _, p in K.SileroVAD().probs(pcm)]
    assert len(got) == len(ref)
    assert max(abs(a - b) for a, b in zip(got, ref)) < 1e-3


def test_preflight_vad_check_is_onnx():
    from runtime.preflight import run_checks

    vad = [c for c in run_checks(C.PipelineConfig()) if c.name.startswith("vad:")]
    assert [c.name for c in vad] == ["vad:silero-onnx"]
    assert vad[0].ok, vad[0].detail


def test_marian_dir_override(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENTRADUCTOR_MARIAN_CT2", str(tmp_path))
    assert C.marian_ct2_dir() == tmp_path
    monkeypatch.delenv("OPENTRADUCTOR_MARIAN_CT2")
    assert C.marian_ct2_dir().name == "opus-mt-es-en-ct2-int8"


def test_convert_marian_without_torch_explains(monkeypatch, tmp_path):
    from runtime import setup as S

    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(RuntimeError, match="OPENTRADUCTOR_MARIAN_CT2"):
        S.convert_marian(out=tmp_path / "x")
