"""T-67 TTS en streaming: $OPENTRADUCTOR_TTS_FIRST_WORDS fuerza el largo del 1.er fragmento de Kokoro."""
from runtime import components as K
from runtime.config import PipelineConfig
from runtime.pipeline import tts_first_words

SENTENCES = [
    "We are going to talk about the new version of the product that we launched last week",
    "The team worked very hard, and the results were better than we expected for this quarter",
    "I think that if we keep the same pace we will finish the migration before the end of the month",
    "Thanks to everyone for coming today to this meeting about the budget",
    "Hello everyone.",
]


def test_sin_env_usa_config(monkeypatch):
    monkeypatch.delenv("OPENTRADUCTOR_TTS_FIRST_WORDS", raising=False)
    assert tts_first_words(PipelineConfig()) == PipelineConfig().tts_first_chunk_max_words
    assert tts_first_words(PipelineConfig(tts_first_chunk_max_words=None)) is None


def test_env_fuerza(monkeypatch):
    monkeypatch.setenv("OPENTRADUCTOR_TTS_FIRST_WORDS", "4")
    assert tts_first_words(PipelineConfig()) == 4
    monkeypatch.setenv("OPENTRADUCTOR_TTS_FIRST_WORDS", "0")
    assert tts_first_words(PipelineConfig()) is None
    monkeypatch.setenv("OPENTRADUCTOR_TTS_FIRST_WORDS", "x")
    assert tts_first_words(PipelineConfig()) == PipelineConfig().tts_first_chunk_max_words


def _fix_at(text: str, n: int) -> int | None:
    """Carácter del prefijo en que first_clause_final fija el 1.er fragmento (None si nunca antes del final)."""
    full = K.split_clauses(text, first_max_words=n, bind_conjunctions=True)[0]
    for i in range(1, len(text) + 1):
        h = K.first_clause_final(text[:i], first_max_words=n, bind_conjunctions=True)
        if h is not None:
            assert h == full  # garantía del camino en streaming: mismo corte que el normal
            return i
    return None


def test_cuatro_palabras_fija_antes_que_ocho():
    earlier = 0
    for s in SENTENCES:
        a4, a8 = _fix_at(s, 4), _fix_at(s, 8)
        if a8 is not None:
            assert a4 is not None and a4 <= a8
            earlier += a4 < a8
    assert earlier >= 3


def test_cuatro_palabras_no_pierde_texto():
    for s in SENTENCES:
        parts = K.split_clauses(s, first_max_words=4, bind_conjunctions=True)
        assert " ".join(parts).split() == s.split()
        assert len(parts[0].split()) <= 6
