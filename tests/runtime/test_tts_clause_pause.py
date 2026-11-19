"""T-67: tope de la pausa entre fragmentos TTS a mitad de frase (tts_clause_pause_ms)."""
import pytest

from runtime import components as K
from runtime.config import PipelineConfig
from runtime.pipeline import tts_clause_pause_ms


def test_keep_ms_by_cut_kind():
    assert K.clause_pause_keep_ms("Edgardo would like", 60, 30) == 30  # corte por palabras: 60 - arranque 30
    assert K.clause_pause_keep_ms("If it's not credited,", 60, 30) == 170  # tras coma: 200 - 30
    assert K.clause_pause_keep_ms("the 20th at 6 PM;", 300, 30) == 270  # tope mayor que 200 manda
    for end in ("account.", "really?", "Stop!", 'he said "no."', ""):
        assert K.clause_pause_keep_ms(end, 60, 30) is None  # fin de oración: cola intacta
    assert K.clause_pause_keep_ms("a b", 10, 30) == 0.0


def test_off_by_default_and_env(monkeypatch):
    monkeypatch.delenv("OPENTRADUCTOR_TTS_CLAUSE_PAUSE_MS", raising=False)
    c = PipelineConfig()
    assert c.tts_clause_pause_ms is None and tts_clause_pause_ms(c) is None
    monkeypatch.setenv("OPENTRADUCTOR_TTS_CLAUSE_PAUSE_MS", "60")
    assert tts_clause_pause_ms(c) == 60.0
    monkeypatch.setenv("OPENTRADUCTOR_TTS_CLAUSE_PAUSE_MS", "off")
    assert tts_clause_pause_ms(PipelineConfig(tts_clause_pause_ms=80.0)) is None
    for bad in ("-5", "abc", ""):
        monkeypatch.setenv("OPENTRADUCTOR_TTS_CLAUSE_PAUSE_MS", bad)
        assert tts_clause_pause_ms(PipelineConfig(tts_clause_pause_ms=80.0)) == 80.0


class _R:
    def __init__(self, audio):
        self.audio = audio


@pytest.mark.parametrize("cap", [None, 60.0])
def test_capped_synth_trims_lead_and_tail_keeps_sentence_end(cap):
    np = pytest.importorskip("numpy")
    pytest.importorskip("mlx.core")
    sr = K.KokoroTTS.SAMPLE_RATE
    sil = lambda ms: np.zeros(int(sr * ms / 1000), np.float32)  # noqa: E731
    tone = 0.3 * np.sin(np.arange(int(sr * 0.2)) * 0.05).astype(np.float32)

    t = K.KokoroTTS.__new__(K.KokoroTTS)
    t.voice, t.speed, t.trim_lead_margin_ms, t.trim_lead_all_clauses = "v", 1.0, 30.0, False
    t.clause_pause_ms, t.last_trim_ms = cap, 0.0
    t.pipe = lambda clause, **kw: iter([_R(np.concatenate([sil(300), tone, sil(430)]))])
    ch = list(t.synth_clauses(["Edgardo would like", "to make a deposit."], first=True))
    n = [len(c.pcm) // 2 for c in ch]
    full = int(sr * 0.93)
    if cap is None:  # camino de hoy: 1.ª con arranque recortado, 2.ª entera
        assert n[1] == full and n[0] < full
    else:
        lead, tail = int(sr * 0.03), int(sr * 0.03)
        assert abs(n[0] - (lead + len(tone) + tail)) <= int(sr * 0.006)
        assert abs(n[1] - (lead + len(tone) + int(sr * 0.43))) <= int(sr * 0.006)  # cierre de oración: cola intacta
