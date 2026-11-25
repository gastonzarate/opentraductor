"""T-103: velocidad de la voz por ritmo del orador (OPENTRADUCTOR_PACE_VOICE, encendido por defecto desde el cierre de T-103, tope 1,3x)."""
import pytest

from runtime import components as K
from runtime.config import PipelineConfig, pace_voice_on
from runtime.pipeline import Engine, Segment


def _engine(**kw):
    return Engine(PipelineConfig(mt_engine="qwen", asr_backend="moonshine-tiny", **kw), lambda *a: None)


def test_pace_speed_formula():
    t = "x" * 57  # 57 car / 11,4 = 5 s a 1x
    assert K.pace_speed(t, 5.0) == pytest.approx(1.0)
    assert K.pace_speed(t, 4.0) == pytest.approx(1.25)
    assert K.pace_speed(t, 2.0) == pytest.approx(1.5)  # tope
    assert K.pace_speed(t, 10.0) == 1.0  # nunca más lenta que la base
    assert K.pace_speed("", 3.0) == 1.0
    assert K.pace_speed(t, None) == 1.0
    assert K.pace_speed(t, 0.1) == 1.0  # duración absurda: sin efecto


def test_on_by_default_tope_13(monkeypatch):
    for k in ("OPENTRADUCTOR_PACE_VOICE", "OPENTRADUCTOR_PACE_MAX", "OPENTRADUCTOR_TTS_SPEED_ADAPTIVE"):
        monkeypatch.delenv(k, raising=False)
    e = _engine()
    assert pace_voice_on(e.cfg)
    e.playback_queue_fn = lambda: 0.0
    e._set_pace(Segment(seg_id=1, start_s=0.0, end_s=2.0), "x" * 57)  # pediría 2,5x: tope 1,3x
    assert e._tts_speed() == pytest.approx(1.3)


@pytest.mark.parametrize("how", ["env", "cfg"])
def test_off(monkeypatch, how):
    if how == "env":
        monkeypatch.setenv("OPENTRADUCTOR_PACE_VOICE", "0")
        e = _engine()
    else:
        monkeypatch.delenv("OPENTRADUCTOR_PACE_VOICE", raising=False)
        e = _engine(pace_voice=False)
    assert not pace_voice_on(e.cfg)
    e._set_pace(Segment(seg_id=1, start_s=0.0, end_s=1.0), "x" * 100)
    assert e._pace_v is None
    e.playback_queue_fn = lambda: 0.0
    assert e._tts_speed() == 1.0
    assert "pace_clauses" not in e.speed_stats


def test_on_uses_max_of_pace_and_lag(monkeypatch):
    monkeypatch.setenv("OPENTRADUCTOR_PACE_VOICE", "1")
    monkeypatch.delenv("OPENTRADUCTOR_TTS_SPEED_ADAPTIVE", raising=False)
    e = _engine()
    e.playback_queue_fn = lambda: 0.0
    e._set_pace(Segment(seg_id=1, start_s=10.0, end_s=14.0), "x" * 57)  # 5 s de voz en 4 s de habla
    assert e._tts_speed() == pytest.approx(1.25)
    e.playback_queue_fn = lambda: 4.0  # atraso al tope: manda la adaptativa (1,5x)
    assert e._tts_speed() == pytest.approx(1.5)
    assert e.speed_stats["pace_clauses"] == 2 and e.speed_stats["pace_wins"] == 1
    e._set_pace(Segment(seg_id=2, start_s=0.0, end_s=None), "x" * 57)  # sin fin de habla: sin efecto
    e.playback_queue_fn = lambda: 0.0
    assert e._tts_speed() == 1.0
    e._set_pace(None, None)
    assert e._tts_speed() == 1.0


def test_env_params(monkeypatch):
    monkeypatch.setenv("OPENTRADUCTOR_PACE_VOICE", "1")
    monkeypatch.setenv("OPENTRADUCTOR_PACE_MAX", "1.3")
    monkeypatch.setenv("OPENTRADUCTOR_PACE_CPS", "5.7")
    e = _engine()
    e.playback_queue_fn = lambda: 0.0
    e._set_pace(Segment(seg_id=1, start_s=0.0, end_s=20.0), "x" * 57)  # 10 s a 1x en 20 s
    assert e._tts_speed() == 1.0
    e._set_pace(Segment(seg_id=1, start_s=0.0, end_s=5.0), "x" * 57)
    assert e._tts_speed() == pytest.approx(1.3)
