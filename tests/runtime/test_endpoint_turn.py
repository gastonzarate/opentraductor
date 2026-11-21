"""T-67 sonda c): detector de fin de turno por audio (smart-turn v3) en los casos "base" del fin de frase adaptativo."""
import os

import numpy as np
import pytest

from runtime import components as K
from runtime.config import PipelineConfig
from runtime.pipeline import Engine, Segment, StreamState


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    for k in ("OPENTRADUCTOR_ENDPOINT_TURN", "OPENTRADUCTOR_ENDPOINT_TURN_P", "OPENTRADUCTOR_ENDPOINT_ADAPTIVE",
              "OPENTRADUCTOR_ENDPOINT_SHORT_MS"):
        monkeypatch.delenv(k, raising=False)


class FakeTurn:
    def __init__(self, p):
        self.p, self.calls = p, []

    def prob(self, pcm):
        self.calls.append(len(pcm))
        return self.p


async def _noemit(kind, f):
    pass


def _state(partial="el granizo destruyó toda la plantación", silence_ms=64.0, caught_up=False):
    st = StreamState("s", vad=None)
    st.in_speech, st.seg = True, Segment(1, start_s=0.0, gen=0)
    st.last_partial, st.silence_ms, st.last_speech_end_s = partial, silence_ms, 1.0
    st.asr_done_pos_s = 1.0 if caught_up else 0.5
    st.turn_buf = bytearray(b"\x01\x00" * 16000)
    return st


def _engine(turn_p, **kw):
    e = Engine(PipelineConfig(**kw), _noemit)
    e.turn = FakeTurn(turn_p) if turn_p is not None else None
    return e


def test_off_by_default():
    c = PipelineConfig()
    assert c.endpoint_turn is False
    e = _engine(0.99)
    assert e._endpoint_wait_ms(_state()) == 160.0 and e.turn.calls == []


def test_base_becomes_turn_once_per_pause(monkeypatch):
    monkeypatch.setenv("OPENTRADUCTOR_ENDPOINT_TURN", "1")
    e, st = _engine(0.9), _state()
    assert e._endpoint_wait_ms(st) == 64.0 and st.ep_wait == (64.0, "turn")
    st.silence_ms += 32
    assert e._endpoint_wait_ms(st) == 64.0
    assert len(e.turn.calls) == 1 and e.turn_stats["calls"] == 1 and e.turn_stats["turn"] == 1


def test_low_prob_keeps_base_and_threshold_env(monkeypatch):
    monkeypatch.setenv("OPENTRADUCTOR_ENDPOINT_TURN", "1")
    e = _engine(0.4)
    assert e._endpoint_wait_ms(_state()) == 160.0
    monkeypatch.setenv("OPENTRADUCTOR_ENDPOINT_TURN_P", "0.3")
    st = _state()
    st.last_speech_end_s = 2.0  # pausa nueva
    assert e._endpoint_wait_ms(st) == 64.0


def test_continue_and_complete_untouched(monkeypatch):
    monkeypatch.setenv("OPENTRADUCTOR_ENDPOINT_TURN", "1")
    e = _engine(0.99)
    assert e._endpoint_wait_ms(_state(partial="fuimos al centro y")) == 480.0
    assert e._endpoint_wait_ms(_state(caught_up=True)) == 64.0
    assert e.turn.calls == []  # solo se consulta en "base"


def test_waits_turn_silence(monkeypatch):
    monkeypatch.setenv("OPENTRADUCTOR_ENDPOINT_TURN", "1")
    e = _engine(0.99)
    assert e._endpoint_wait_ms(_state(silence_ms=0.0)) == 160.0 and e.turn.calls == []


def test_whisper_log_mel_matches_transformers():
    tr = pytest.importorskip("transformers")
    rng = np.random.default_rng(0)
    audio = (rng.standard_normal(16000 * 3) * 0.1).astype(np.float32)
    fe = tr.WhisperFeatureExtractor(chunk_length=8)
    ref = fe(audio, sampling_rate=16000, return_tensors="np", padding="max_length", max_length=8 * 16000,
             truncation=True, do_normalize=True).input_features[0]
    ours = K.whisper_log_mel(audio)
    assert ours.shape == ref.shape == (80, 800)
    assert np.max(np.abs(ours - ref)) < 2e-3


@pytest.mark.skipif(os.environ.get("OPENTRADUCTOR_TEST_SMART_TURN") != "1", reason="carga el ONNX real (8 MB)")
def test_smart_turn_real_model():
    m = K.SmartTurn()
    p = m.prob((np.zeros(16000, dtype=np.int16)).tobytes())
    assert 0.0 <= p <= 1.0


def test_asr_tail_stats_counts():
    """T-67 sonda b): cuenta si el parcial del endpoint ya era el final (techo de la especulación)."""
    e = _engine(None)
    e._asr_tail_count(Segment(1, 0.0, ep_partial="hola que tal", ep_caught_up=True, text_es_raw="Hola que tal."))
    e._asr_tail_count(Segment(2, 0.0, ep_partial="hola que", ep_caught_up=False, text_es_raw="hola que tal"))
    assert e.asr_tail_stats == {"n": 2, "caught_up": 1, "eq": 1, "eq_caught_up": 1}


# ---------------------------------------------------------------- T-80: "complete" exige fin de turno (flag)
@pytest.fixture
def _ct_env(monkeypatch):
    monkeypatch.delenv("OPENTRADUCTOR_ENDPOINT_COMPLETE_TURN", raising=False)
    return monkeypatch


def test_complete_turn_off_by_default(_ct_env):
    assert PipelineConfig().endpoint_complete_turn is False
    e = _engine(0.1)
    assert e._endpoint_wait_ms(_state(caught_up=True)) == 64.0 and e.turn.calls == []


def test_complete_turn_low_prob_waits_base(_ct_env):
    _ct_env.setenv("OPENTRADUCTOR_ENDPOINT_COMPLETE_TURN", "1")
    e, st = _engine(0.2), _state(caught_up=True)
    assert e._endpoint_wait_ms(st) == 160.0 and st.ep_wait == (160.0, "complete_noturn")
    assert len(e.turn.calls) == 1


def test_complete_turn_high_prob_keeps_short(_ct_env):
    _ct_env.setenv("OPENTRADUCTOR_ENDPOINT_COMPLETE_TURN", "1")
    e, st = _engine(0.9), _state(caught_up=True)
    assert e._endpoint_wait_ms(st) == 64.0 and st.ep_wait == (64.0, "complete")


def test_complete_turn_without_model_is_today(_ct_env):
    _ct_env.setenv("OPENTRADUCTOR_ENDPOINT_COMPLETE_TURN", "1")
    e = _engine(None)
    assert e._endpoint_wait_ms(_state(caught_up=True)) == 64.0


def test_complete_turn_does_not_touch_base(_ct_env):
    _ct_env.setenv("OPENTRADUCTOR_ENDPOINT_COMPLETE_TURN", "1")
    e = _engine(0.99)  # ENDPOINT_TURN apagado: la base sigue en 160 sin consultar
    assert e._endpoint_wait_ms(_state(caught_up=False)) == 160.0 and e.turn.calls == []
