"""T-58 mt_stream_first_chunk: el 1.er fragmento de Kokoro sale durante el decode de Qwen, con los mismos cortes
y el mismo texto que el camino normal."""
import asyncio
import itertools

import pytest

import test_asr_spec_finish as tasf
from runtime import components as K
from test_speculative import FakeMT, FakeTTS

SENTENCES = [
    "We need to finish the payment integration before Friday, otherwise the client won't sign.",
    "Hail hit the car yesterday and the insurance still hasn't answered.",
    "Can you check how much the flight costs?",
    "Ok.",
    "Yes, sure, sure.",
    "The v2.1 release breaks user_id parsing in getUser and the API returns 500 at 3:30 pm.",
    "I think that the money comes in on the fifth and then we pay the rent, the salaries and the taxes.",
    "Look, honestly, I don't know, but if it's pointless we drop it and move on to the next ticket.",
    "It costs a thousand pesos per month, which is cheap; however, we have to ask the boss first.",
    "and so the idea is that we keep the team small and focused on the store release for now",
    "Dale.",
    "We talked about dignity, about respect and about the solemnity and respect that the moment deserves, "
    "so let's not rush it.",
    "First: logs. Second: metrics. Third: alerts, all of them by Monday.",
]
CONFIGS = [(None, False), (5, True), (6, True), (7, True), (8, True), (8, False), (3, False)]


@pytest.mark.parametrize("max_words,bind", CONFIGS)
def test_first_clause_final_never_disagrees_with_plan(max_words, bind):
    """Para cada prefijo (carácter a carácter) de cada frase: si first_clause_final fija un fragmento, es
    exactamente split_clauses(final)[0]. Y en frases largas lo fija antes del final."""
    early = 0
    for s in SENTENCES:
        want = K.split_clauses(s, first_max_words=max_words, bind_conjunctions=bind)[0]
        got_at = None
        for i in range(1, len(s) + 1):
            h = K.first_clause_final(s[:i], first_max_words=max_words, bind_conjunctions=bind)
            if h is not None:
                assert h == want, (s, s[:i], h, want)
                # siempre queda cola: un fragmento siguiente lleva planned_text_en (ver _mt_stream_speak)
                assert len(K.split_clauses(s, first_max_words=max_words, bind_conjunctions=bind)) >= 2, s
                got_at = got_at if got_at is not None else i
        if got_at is not None and got_at < len(s):
            early += 1
    assert early >= 4


def test_first_clause_final_with_continuations():
    """Mismo prefijo, continuaciones distintas (cortas, largas, con y sin puntuación): nunca contradice."""
    prefixes = ["We need to finish the payment integration, ", "Look, honestly, I don't know ",
                "The money comes in on the fifth and then ", "First: logs. Second: "]
    tails = ["", "ok.", "and that's it.", "x", "so, yes, we pay, then we rest, then we go home.",
             "3.5 percent", "and the rest of the very long sentence goes on without any punctuation at all"]
    for (mw, bind), p, t in itertools.product(CONFIGS, prefixes, tails):
        h = K.first_clause_final(p, first_max_words=mw, bind_conjunctions=bind)
        if h is not None:
            assert h == K.split_clauses(p + t, first_max_words=mw, bind_conjunctions=bind)[0], (mw, p, t)


# --- pipeline ---------------------------------------------------------------------------------------------------

LOG: list[str] = []
EN = "We need to finish the payment integration before Friday, otherwise the client won't sign."


class StreamTTS(FakeTTS):
    def plan(self, t):
        return K.split_clauses(t, first_max_words=5, bind_conjunctions=True)

    def first_clause_final(self, p):
        return K.first_clause_final(p, first_max_words=5, bind_conjunctions=True)

    def synth_clauses(self, clauses, first=True):
        for ch in super().synth_clauses(clauses, first):
            LOG.append("tts:" + ch.text)
            yield ch


class StreamMT(FakeMT):
    def translate(self, t, context=()):
        self.calls.append(t)
        return EN

    def translate_stream(self, t, context=(), on_text=None):
        self.calls.append(t)
        acc = ""
        for w in EN.split(" "):
            acc += (" " if acc else "") + w
            LOG.append("tok")
            if on_text:
                on_text(acc)
        return EN


def _go(monkeypatch, env=None, **kw):
    LOG.clear()
    monkeypatch.setattr(tasf, "FakeTTS", StreamTTS)
    if env is not None:
        monkeypatch.setenv("OPENTRADUCTOR_MT_STREAM_FIRST_CHUNK", env)
    return asyncio.run(tasf._run([1] * 10 + [0] * 10 + [1] * 10, mt=StreamMT(), **kw))


def _tts_texts(ev):
    return [f["pcm"].decode() for k, f in ev if k == "tts_chunk"]


def test_off_by_default_uses_normal_path(monkeypatch):
    e, ev = _go(monkeypatch, mt_engine="qwen")
    assert "tok" not in LOG
    assert e.mt_stream_stats["segments"] == 0
    assert _tts_texts(ev) and _tts_texts(ev)[: len(StreamTTS().plan(EN))] == StreamTTS().plan(EN)


def test_stream_same_chunks_and_text_and_head_before_mt_ends(monkeypatch):
    _, ev_base = _go(monkeypatch, mt_engine="qwen")
    e, ev = _go(monkeypatch, mt_engine="qwen", mt_stream_first_chunk=True)
    assert _tts_texts(ev) == _tts_texts(ev_base)
    assert [f["text"] for k, f in ev if k == "mt_final"] == [f["text"] for k, f in ev_base if k == "mt_final"]
    head = StreamTTS().plan(EN)[0]
    i = LOG.index("tts:" + head)
    assert "tok" in LOG[i + 1:]  # el 1.er fragmento se sintetizó antes del último token
    s = e.mt_stream_stats
    assert s["segments"] >= 1 and s["head_early"] == s["segments"] and s["mismatch"] == 0
    kinds = [k for k, f in ev if k in ("mt_final", "tts_chunk")]
    assert kinds[0] == "tts_chunk"  # el audio adelantado llega antes que su mt_final
    assert all(f.get("mt_stream") for k, f in ev if k == "mt_final")
    # texto planeado por frase (1.er no vacío por seg_id, como benchmark.metrics.commit_texts) igual al normal
    def planned(evs):
        out = {}
        for k, f in evs:
            if k == "tts_chunk" and f.get("planned_text_en"):
                out.setdefault(f["seg_id"], f["planned_text_en"])
        return list(out.values())
    assert planned(ev) == planned(ev_base) == [EN] * len(planned(ev_base))
    assert s["head_only"] == 0


def test_env_forces_off(monkeypatch):
    e, _ = _go(monkeypatch, env="0", mt_engine="qwen", mt_stream_first_chunk=True)
    assert e.mt_stream_stats["segments"] == 0


def test_ignored_for_marian(monkeypatch):
    e, _ = _go(monkeypatch, mt_engine="marian", mt_stream_first_chunk=True)
    assert e.mt_stream_stats["segments"] == 0 and "tok" not in LOG


def _reference_sentences() -> list[str]:
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    out = []
    for p in sorted((root / "data" / "manifests").glob("*.jsonl")) + [root / "scripts/eval/sets/mt_sentido_ar.jsonl"]:
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                for k in ("reference_en", "text_en", "en"):
                    if isinstance(r.get(k), str) and r[k].strip():
                        out.append(r[k].strip())
    return out


@pytest.mark.parametrize("max_words,bind", CONFIGS)
def test_first_clause_final_on_reference_sets(max_words, bind):
    """Mismo invariante sobre todas las referencias EN de los sets del repo (reuniones AR, T-58, sentido):
    cualquier prefijo que fije un fragmento coincide con plan(final)[0] y deja cola para el texto planeado."""
    sents = _reference_sentences()
    assert len(sents) >= 50
    for s in sents:
        plan = K.split_clauses(s, first_max_words=max_words, bind_conjunctions=bind)
        for i in range(1, len(s) + 1):
            h = K.first_clause_final(s[:i], first_max_words=max_words, bind_conjunctions=bind)
            if h is not None:
                assert h == plan[0] and len(plan) >= 2, (max_words, bind, s[:i], h, plan)
