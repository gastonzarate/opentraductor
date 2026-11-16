"""T-58: recorte del silencio inicial en el 1.er chunk de cada cláusula (flag tts_trim_lead_all_clauses)."""
import sys
import types

import numpy as np
import pytest

from runtime import components as K
from runtime.config import PipelineConfig

SR = K.KokoroTTS.SAMPLE_RATE
LEAD_S = 0.35  # silencio que Kokoro pone al inicio de cada cláusula (medido: 295-380 ms)


def _audio(lead_s=LEAD_S, voiced_s=0.2):
    t = np.arange(int(SR * voiced_s)) / SR
    return np.concatenate([np.zeros(int(SR * lead_s), np.float32), (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)])


class _R:
    def __init__(self, audio):
        self.audio = audio


def _tts(all_clauses, chunks_per_clause=1):
    tts = object.__new__(K.KokoroTTS)
    tts.voice, tts.speed = "af_heart", 1.0
    tts.trim_lead_margin_ms = 30.0
    tts.trim_tail = False
    tts.tail_trimmed_ms_total = 0.0
    tts.last_trim_ms = 0.0
    tts.trim_lead_all_clauses = all_clauses
    tts.pipe = lambda clause, **_: [_R(_audio()) for _ in range(chunks_per_clause)]
    return tts


@pytest.fixture(autouse=True)
def _fake_mlx(monkeypatch):
    mx = types.ModuleType("mlx.core")
    mx.eval = lambda *_: None
    mlx = types.ModuleType("mlx")
    mlx.core = mx
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)


def _lead_ms(ch):
    x = np.frombuffer(ch.pcm, np.int16).astype(np.float32) / 32768
    return K.lead_silence_samples(x, SR) / SR * 1000


def test_off_trims_only_first_chunk():
    chunks = list(_tts(False).synth_clauses(["a b", "c d", "e f"], first=True))
    assert _lead_ms(chunks[0]) == pytest.approx(30, abs=6)
    assert all(_lead_ms(c) == pytest.approx(LEAD_S * 1000, abs=6) for c in chunks[1:])


def test_on_trims_head_of_every_clause_but_not_inner_chunks():
    tts = _tts(True, chunks_per_clause=2)
    chunks = list(tts.synth_clauses(["a b", "c d"], first=True))
    assert [_lead_ms(c) == pytest.approx(30, abs=6) for c in chunks] == [True, False, True, False]
    assert tts.last_trim_ms == pytest.approx(LEAD_S * 1000 - 30, abs=6)  # solo cuenta el de la frase


def test_on_split_equals_whole():
    """plan[:1] con first=True + plan[1:] con first=False == todo junto (lo usa la especulación)."""
    cl = ["a b", "c d", "e f"]
    whole = [c.pcm for c in _tts(True).synth_clauses(cl, first=True)]
    t = _tts(True)
    split = [c.pcm for c in t.synth_clauses(cl[:1], True)] + [c.pcm for c in t.synth_clauses(cl[1:], first=False)]
    assert whole == split


def test_on_without_margin_is_noop():
    tts = _tts(True)
    tts.trim_lead_margin_ms = None
    chunks = list(tts.synth_clauses(["a b", "c d"], first=True))
    assert all(_lead_ms(c) == pytest.approx(LEAD_S * 1000, abs=6) for c in chunks)


def test_config_default_off_and_fingerprint():
    from runtime.pipeline import Engine

    async def emit(*_):
        pass

    assert PipelineConfig().tts_trim_lead_all_clauses is False
    assert PipelineConfig.from_dict({"tts_trim_lead_all_clauses": True}).tts_trim_lead_all_clauses is True
    assert Engine(PipelineConfig(), emit)._spec_fp() != Engine(PipelineConfig(tts_trim_lead_all_clauses=True), emit)._spec_fp()
