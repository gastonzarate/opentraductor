"""T-92 (BUG-9): retraso de la voz acotado en charla continua.

Velocidad adaptativa según la cola, frases solo muletilla sin voz y salto de frases viejas en el Player."""
import numpy as np
import pytest

from runtime import audio_io
from runtime import components as K
from runtime.config import PipelineConfig
from runtime.pipeline import Engine


def test_adaptive_speed_ramp():
    assert K.adaptive_speed(0.0) == 1.0
    assert K.adaptive_speed(2.0) == 1.0
    assert K.adaptive_speed(5.0) == pytest.approx(1.15)
    assert K.adaptive_speed(8.0) == pytest.approx(1.30)
    assert K.adaptive_speed(120.0) == pytest.approx(1.30)
    assert K.adaptive_speed(5.0, base=1.4, max_speed=1.3) == 1.4  # nunca más lento que la base


@pytest.mark.parametrize("t", ["Eh.", "eh, eh...", "Em", "¿Se entiende?", "Este...", "Got it?", "Um, you know?",
                               "mmm", "Eh, ¿me explico?"])
def test_filler_only(t):
    assert K.is_filler_only(t)


@pytest.mark.parametrize("t", ["", "...", "No.", "Sí.", "OK.", "Right?", "Eh, el modelo es este.",
                               "Got it, now we deploy.", "Este modelo funciona.", "You know the answer."])
def test_not_filler(t):
    assert not K.is_filler_only(t)


def _player(skip_old_s=20.0):
    d = audio_io.Device("out", "Out", 0, 0, 1, 1000.0, True, True)
    return audio_io.Player(d, skip_old_s=skip_old_s)


def _pcm(seconds, sr=1000, v=1000):
    return (np.full(int(seconds * sr), v, dtype="<i2")).tobytes()


def _play(p, frames):
    out = np.zeros((frames, 1), dtype=np.float32)
    p._cb(out, frames, None, None)
    return out


def test_player_skips_old_whole_segments_not_current_nor_new():
    p = _player(skip_old_s=20.0)
    p.write(_pcm(5), 1000, "a")
    _play(p, 100)  # "a" suena
    for s in "bcd":
        p.write(_pcm(5), 1000, s)
    assert p.skipped_s == 0.0
    p.write(_pcm(5), 1000, "e")  # 24,9 s > 20: sale la más vieja sin sonar (b), nunca a (suena) ni e (llega)
    assert p.skipped_s == pytest.approx(5.0)
    assert p.skipped_segments == 1
    assert list(p._tags) == ["c", "d", "e"]
    assert p.queued_s == pytest.approx(19.9)
    assert p.dropped_s == 0.0


def test_player_never_drops_current_or_incoming_segment():
    p = _player(skip_old_s=2.0)
    p.write(_pcm(1), 1000, "a")
    _play(p, 10)
    p.write(_pcm(3), 1000, "a")  # resto de la frase que suena: no se salta
    p.write(_pcm(3), 1000, "b")
    assert p.skipped_s == 0.0 and p.dropped_s == 0.0
    assert p.queued_s == pytest.approx(6.99)


def test_player_skip_off_keeps_old_policy():
    d = audio_io.Device("out", "Out", 0, 0, 1, 1000.0, True, True)
    p = audio_io.Player(d, max_queue_s=3.0, skip_old_s=None)
    p.write(_pcm(2), 1000, "a")
    p.write(_pcm(2), 1000, "b")
    assert p.dropped_s == pytest.approx(2.0) and p.skipped_s == 0.0


def test_player_clear_resets_tags():
    p = _player()
    p.write(_pcm(1), 1000, "a")
    p.clear()
    assert not p._tags and p._cur_tag is None


def _engine(**kw):
    return Engine(PipelineConfig(mt_engine="qwen", asr_backend="moonshine-tiny", **kw), lambda *a: None)


def test_engine_speed_follows_estimated_queue(monkeypatch):
    monkeypatch.delenv("OPENTRADUCTOR_TTS_SPEED_ADAPTIVE", raising=False)
    e = _engine()
    assert e._speed_adaptive
    assert e._tts_speed() == 1.0
    e._note_tts(_pcm(10, sr=24000), 24000)  # 10 s encolados -> tope
    assert e.playback_queue_s() == pytest.approx(10.0, abs=0.1)
    assert e._tts_speed() == pytest.approx(1.30)
    e.playback_queue_fn = lambda: 5.0  # la cola real del Player manda
    assert e._tts_speed() == pytest.approx(1.15)
    assert e.speed_stats["clauses_fast"] == 2


def test_engine_flags_off(monkeypatch):
    monkeypatch.setenv("OPENTRADUCTOR_TTS_SPEED_ADAPTIVE", "0")
    monkeypatch.setenv("OPENTRADUCTOR_TTS_SKIP_FILLERS", "0")
    e = _engine()
    assert not e._speed_adaptive
    assert not e._is_filler("eh")


def test_kokoro_clause_speed_uses_fn():
    t = K.KokoroTTS.__new__(K.KokoroTTS)
    t.speed = 1.0
    t.speed_fn = None
    assert t._clause_speed() == 1.0
    t.speed_fn = lambda: 1.2
    assert t._clause_speed() == 1.2 and t.last_speed == 1.2
    t.speed_fn = lambda: 1 / 0
    assert t._clause_speed() == 1.0
