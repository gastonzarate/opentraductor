"""T-67 paso 2: local agreement y cortes de cláusula (piezas puras de runtime/simul.py)."""
from runtime.simul import ClauseCommitter, agreed_prefix, next_clause_end


def test_agreed_prefix_ignores_case_and_punct():
    assert agreed_prefix("el granizo destruyó toda la planta", "El granizo, destruyó toda la plantación") == \
        ["El", "granizo,", "destruyó", "toda", "la"]
    assert agreed_prefix("", "hola") == []


def test_clause_end_before_cue_with_min_words():
    w = "fuimos al centro con mi hermana pero no".split()
    assert next_clause_end(w, 0) == 6  # corta antes de "pero"
    assert next_clause_end("fui pero no".split(), 0) is None  # cláusula demasiado corta


def test_clause_end_pair_needs_both_words():
    w = "no llegamos a tiempo al tren o".split()
    assert next_clause_end(w, 0) is None
    assert next_clause_end(w + ["sea"], 0) == 6


def test_clause_end_after_comma_needs_next_word():
    w = "cuando llegamos a la casa,".split()
    assert next_clause_end(w, 0) is None
    assert next_clause_end(w + ["estaba"], 0) == 5


def test_committer_confirms_only_after_two_partials_agree():
    c = ClauseCommitter()
    assert c.update("fuimos al centro con mi hermana pero") == []  # primera hipótesis: nada estable
    assert c.update("fuimos al centro con mi hermana pero no había") == ["fuimos al centro con mi hermana"]
    assert c.update("fuimos al centro con mi hermana pero no había nadie") == []
    assert c.rest("fuimos al centro con mi hermana pero no había nadie") == "pero no había nadie"


def test_committer_rewrite_falls_back_to_full_sentence():
    c = ClauseCommitter()
    c.update("fuimos al centro con mi hermana pero")
    c.update("fuimos al centro con mi hermana pero no")
    assert c.update("fuimos al cine con mi hermana pero no") == []  # contradice lo confirmado
    assert c.rest("fuimos al cine con mi hermana pero no") is None


def test_committer_without_clauses_returns_final():
    c = ClauseCommitter()
    c.update("el granizo destruyó")
    c.update("el granizo destruyó la plantación")
    assert c.rest("el granizo destruyó la plantación") == "el granizo destruyó la plantación"


# ---------------------------------------------------------------- pipeline (flag mt_simul_clauses)
import asyncio  # noqa: E402

from runtime.config import PipelineConfig  # noqa: E402
from runtime.pipeline import Engine  # noqa: E402

from test_asr_spec_finish import PatternVAD  # noqa: E402
from test_speculative import WIN, FakeMT, FakeTTS  # noqa: E402


class GrowASR:
    """Cada add() muestra una palabra más de ``words``; el final es la frase entera (o ``final``)."""

    def __init__(self, words, final=None):
        self.words, self.final, self.i = words, final, 0

    def start(self):
        self.i = 0

    def add(self, pcm):
        if pcm.count(0) < len(pcm):
            self.i = min(len(self.words), self.i + 1)
        return " ".join(self.words[: self.i])

    def finish(self):
        return self.final if self.final is not None else " ".join(self.words)


async def _run_simul(words, final=None, **kw):
    events = []

    async def emit(kind, f):
        events.append((kind, f))

    cfg = PipelineConfig(asr_backend="moonshine-tiny", speculative_mt_tts=False, endpoint_silence_ms=160,
                         asr_chunk_ms=32, min_speech_ms=0, mt_kv_prefill_on_pause=False, asr_normalize_text=False, **kw)
    e = Engine(cfg, emit)
    e.asr, e.mt, e.tts = GrowASR(words, final), FakeMT(delay=0.0), FakeTTS()
    pattern = [1] * (len(words) + 6)
    e.vad_proto = PatternVAD(pattern)
    e._out_q = asyncio.Queue(maxsize=16)
    e._out_task = asyncio.ensure_future(e._speaker_loop())
    await e.start_stream("s")
    for i in range(len(pattern) + 20):
        voice = i < len(pattern)
        await e.push("s", (b"\x01" if voice else b"\x00") * (2 * WIN), 0.0)
        await asyncio.sleep(0.01)
    await e.end_stream("s")
    await e.close()
    return e, events


SENT = "ayer fuimos al centro con mi hermana pero no encontramos nada que valga la pena".split()


def test_simul_off_by_default(monkeypatch):
    monkeypatch.delenv("OPENTRADUCTOR_MT_SIMUL", raising=False)
    e, ev = asyncio.run(_run_simul(SENT))
    assert e.mt.calls == [" ".join(SENT)]
    assert not [k for k, _ in ev if k.startswith("simul_")]


def test_simul_speaks_clause_then_rest_one_mt_final(monkeypatch):
    monkeypatch.setenv("OPENTRADUCTOR_MT_SIMUL", "1")
    e, ev = asyncio.run(_run_simul(SENT))
    assert e.mt.calls == ["ayer fuimos al centro con mi hermana", "pero no encontramos nada que valga la pena"]
    finals = [f for k, f in ev if k == "mt_final"]
    assert len(finals) == 1 and finals[0]["simul_clauses"] == 1
    assert finals[0]["text"] == "EN ayer fuimos al centro con mi hermana EN pero no encontramos nada que valga la pena"
    kinds = [k for k, _ in ev]
    # el audio de la 1.ª cláusula sale antes del fin de frase
    assert kinds.index("tts_chunk") < kinds.index("asr_final")
    assert e.simul_stats["clauses"] == 1 and e.simul_stats["segments"] == 1


def test_simul_asr_rewrite_still_translates_only_rest(monkeypatch):
    monkeypatch.setenv("OPENTRADUCTOR_MT_SIMUL", "1")
    final = "ayer fuimos al centro con mi hermano pero no encontramos nada que valga la pena"
    e, ev = asyncio.run(_run_simul(SENT, final=final))
    assert e.mt.calls[-1] == "pero no encontramos nada que valga la pena"
    assert e.simul_stats["rewrite"] == 1
    assert [f["text_raw"] for k, f in ev if k == "asr_final"] == [final]


def test_extra_cues_and_min_words_are_parameters():
    w = "fuimos al centro y no había nadie".split()
    assert next_clause_end(w, 0) is None  # defaults de hoy: "y" no corta
    assert next_clause_end(w, 0, 3, frozenset({"y"})) == 3
    c = ClauseCommitter(min_words=3, extra_cues=("y", " que ", ""))
    assert c.extra_cues == frozenset({"y", "que"})
    c.update("fuimos al centro y")
    assert c.update("fuimos al centro y no") == ["fuimos al centro"]
    assert c.rest("fuimos al centro y no había nadie") == "y no había nadie"


def test_engine_clauser_reads_config_and_env(monkeypatch):
    from runtime.config import PipelineConfig
    from runtime.pipeline import Engine

    e = Engine.__new__(Engine)
    e.cfg = PipelineConfig()
    monkeypatch.delenv("OPENTRADUCTOR_MT_SIMUL_MIN_WORDS", raising=False)
    monkeypatch.delenv("OPENTRADUCTOR_MT_SIMUL_CUES", raising=False)
    c = e._new_clauser()
    assert (c.min_words, c.extra_cues) == (5, frozenset())
    monkeypatch.setenv("OPENTRADUCTOR_MT_SIMUL_MIN_WORDS", "3")
    monkeypatch.setenv("OPENTRADUCTOR_MT_SIMUL_CUES", "y,que")
    c = e._new_clauser()
    assert (c.min_words, c.extra_cues) == (3, frozenset({"y", "que"}))


def test_join_pieces_fixes_mid_sentence_case_and_period():
    from runtime.simul import join_pieces
    # casos reales de t67-terminos (simul, ctx200)
    assert join_pieces([("Si en 48 horas no se acredita,", "If it's not credited within 48 hours,"),
                        ("hacemos un contracargo.", "We'll do a chargeback.")]) == \
        "If it's not credited within 48 hours, we'll do a chargeback."
    assert join_pieces([("A más tardar el viernes 20,", "At the latest by Friday the 20th."),
                        ("a las 18.", "by 6 PM.")]) == "At the latest by Friday the 20th, by 6 PM."


def test_join_pieces_keeps_names_sentences_and_empty():
    from runtime.simul import join_pieces
    # nombre propio e "I" no se tocan; tras punto real tampoco; piezas vacías se saltan
    assert join_pieces([("Le dije a Martín que venga,", "I told Martin to come,"),
                        ("Martín no vino.", "Martin didn't come.")]) == "I told Martin to come, Martin didn't come."
    assert join_pieces([("listo,", "Ready,"), ("yo voy.", "I'm going.")]) == "Ready, I'm going."
    assert join_pieces([("Terminamos.", "We're done."), ("Nos vemos.", "See you.")]) == "We're done. See you."
    assert join_pieces([("pero bueno", "But well."), ("", "")]) == "But well."
    # cláusula cortada antes de "pero" (sin puntuación en español): se quita el punto de Marian
    assert join_pieces([("Lo hicimos rápido", "We did it fast."), ("pero salió bien.", "But it went well.")]) == \
        "We did it fast but it went well."
