"""Troceo del primer fragmento TTS y normalización de texto ASR (sin modelos)."""
import asyncio

import pytest

from runtime import components as K
from runtime.config import PipelineConfig

LONG = "Give the place all dignity the solemnity and respect it deserves do not joke about the Holocaust or Nazism"


def words(parts):
    return " ".join(parts).split()


def test_long_unpunctuated_first_chunk_is_split_without_loss():
    parts = K.split_clauses(LONG, first_max_words=8)
    assert len(parts) == 2 and 3 <= len(parts[0].split()) <= 8
    assert words(parts) == LONG.split()  # sin pérdida ni duplicación, mismo orden
    assert parts[0].split()[-1].lower() not in K._NO_END


def test_disabled_and_short_unchanged():
    assert K.split_clauses(LONG) == [LONG]
    assert K.split_clauses("Hello there my friend.", first_max_words=8) == ["Hello there my friend."]


def test_punctuated_text_keeps_natural_clauses():
    t = "It is worth the dignity and respect it deserves to the place, not joke about the holocaust or Nazism."
    base = K.split_clauses(t)
    got = K.split_clauses(t, first_max_words=8)
    assert words(got) == words(base) == t.split()
    assert got[1:] == base[1:] or len(got) == len(base) + 1


@pytest.mark.parametrize("text", [
    "The server will restart at 3:30 on Tuesday and then deploy version 2.1 to all the regions",
    "Please check the user_id field in the request and then retry the call after one minute again",
    "We call the getUser endpoint and the API gateway before we return the answer to the client now",
])
def test_numbers_and_identifiers_are_never_split(text):
    parts = K.split_clauses(text, first_max_words=5)
    assert words(parts) == text.split()
    if len(parts) > 1:
        a, b = parts[0].split()[-1], parts[1].split()[0]
        assert not K._risky_token(a) and not K._risky_token(b)


def test_no_valid_cut_keeps_whole():
    t = "a1 b2 c3 d4 e5 f6 g7 h8 i9 j0"
    assert K.split_clauses(t, first_max_words=3) == [t]


def test_normalize_asr_text():
    n = K.normalize_asr_text
    assert n("el granizo destruyó la plantación") == "El granizo destruyó la plantación."
    assert n("Ya está.") == "Ya está."
    assert n("dele al lugar toda la dignidad") == "Dele al lugar toda la dignidad."
    assert "," not in n("cuando es baja cuando es alta")  # no inventa comas
    assert n("  ") == "" and n("¿cómo estás?") == "¿cómo estás?"
    assert n(n("hola")) == n("hola")  # idempotente


def test_config_flags_default_off():
    c = PipelineConfig()
    assert c.tts_first_chunk_max_words == 8 and c.asr_normalize_text is False  # T-58
    c = PipelineConfig.from_dict({"tts_first_chunk_max_words": 8, "asr_normalize_text": True})
    assert c.tts_first_chunk_max_words == 8 and c.asr_normalize_text


def test_engine_keeps_raw_and_normalized_asr_text():
    from runtime.pipeline import Engine, Segment

    events = []

    async def emit(kind, f):
        events.append((kind, f))

    class FakeASR:
        def start(self): pass
        def add(self, pcm): return ""
        def finish(self): return "averiguame precios cuando es temporada baja"

    class FakeMT:
        def translate(self, t): return t

    async def go(norm):
        e = Engine(PipelineConfig(asr_normalize_text=norm, min_speech_ms=0), emit)
        e.asr, e.mt = FakeASR(), FakeMT()
        e._out_q = asyncio.Queue(maxsize=4)
        st = type("S", (), {})()
        st.stream_id = "s"; st.asr_q = asyncio.Queue(); st.asr_pending_s = 0.0; st.last_partial = ""
        seg = Segment(1, 0.0, 1.0)
        await st.asr_q.put(("start", (seg, b""))); await st.asr_q.put(("end", seg)); await st.asr_q.put(("stop", None))
        await e._asr_loop(st)
        await e.close()
        return seg, next(f for k, f in events if k == "asr_final")

    seg, f = asyncio.run(go(True))
    assert seg.text_es_raw == "averiguame precios cuando es temporada baja"
    assert seg.text_es == "Averiguame precios cuando es temporada baja." and f["text"] == seg.text_es and f["text_raw"] == seg.text_es_raw
    events.clear()
    seg, f = asyncio.run(go(False))
    assert seg.text_es == seg.text_es_raw and f["text"] == f["text_raw"]


def test_cap_is_a_flexible_target_not_a_maximum():
    f = K.split_first_by_words
    assert f("hello", 8) == ["hello"]  # 1 palabra
    assert f("one two three four five six seven eight", 8) == ["one two three four five six seven eight"]
    # racha de números: la cabeza supera el objetivo para no cortar dentro de ella
    h, r = f("the 1 2 3 4 5 6 7 8 then we continue to talk about stuff", 4)
    assert len(h.split()) > 4 and h.split()[-1] == "then" and not K._risky_token("then")
    assert (h + " " + r).split() == "the 1 2 3 4 5 6 7 8 then we continue to talk about stuff".split()
    # sin alternativa, el 2.º pase acepta terminar en palabra funcional
    assert f("go to the store and buy", 3)[0].split()[-1] == "the"


def test_bind_conjunctions_rule_keeps_words_and_avoids_conjunction_cuts():
    t = LONG
    base = K.split_first_by_words(t, 8)
    assert base[1].startswith("and respect")  # heurística T1/T2 (sin regla): corte antes de 'and'
    h, r = K.split_first_by_words(t, 8, bind_conjunctions=True)
    assert (h + " " + r).split() == t.split()
    assert h.split()[-1].lower() not in K._CONJ and r.split()[0].lower() not in K._CONJ
    assert h == "Give the place all dignity"
    # sin conjunciones el resultado no cambia
    u = "one two three four five six seven eight nine ten"
    assert K.split_first_by_words(u, 8, True) == K.split_first_by_words(u, 8)
    # nunca termina ni empieza en conjunción, incluso con fallback hacia delante
    # cada frontera toca una conjunción: no hay corte válido, se conserva la cláusula entera
    assert K.split_first_by_words("a and b and c and d", 2, bind_conjunctions=True) == ["a and b and c and d"]
    h, r = K.split_first_by_words("alpha beta and gamma delta epsilon zeta", 2, bind_conjunctions=True)
    assert (h + " " + r).split() == "alpha beta and gamma delta epsilon zeta".split()
    assert h.split()[-1] not in K._CONJ and r.split()[0] not in K._CONJ


def test_bind_conjunctions_flag_default_on_and_threaded():
    assert PipelineConfig().tts_bind_conjunctions is True  # T-58
    assert not PipelineConfig.from_dict({"tts_bind_conjunctions": False}).tts_bind_conjunctions
    assert PipelineConfig.from_dict({"tts_bind_conjunctions": True}).tts_bind_conjunctions
    assert K.split_clauses(LONG, first_max_words=8) == [LONG[: LONG.index(" and respect")], LONG[LONG.index("and respect"):]]
    c = K.split_clauses(LONG, first_max_words=8, bind_conjunctions=True)
    assert c[0] == "Give the place all dignity" and " ".join(c) == LONG


def test_adapter_logs_raw_asr_and_mt_input_in_describe():
    from runtime.adapters.baseline import BaselineAdapter

    async def go():
        a = BaselineAdapter()
        await a._emit("asr_final", {"stream_id": "s1", "text": "Hola.", "text_raw": "hola", "seg_id": 3})
        return a.describe_log() if hasattr(a, "describe_log") else a.asr_text_log

    log = asyncio.run(go())
    assert log == [{"stream_id": "s1", "commit_id": "s1:3", "asr_raw": "hola", "mt_input": "Hola."}]
