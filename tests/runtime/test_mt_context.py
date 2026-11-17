"""Pedido 17/11: Qwen traduce por sentido (prompt) y con las frases previas de la sesión como contexto."""
import asyncio

from runtime import components as K

from test_asr_spec_finish import _run
from test_speculative import FakeMT


class CtxMT(FakeMT):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.contexts = []

    def translate(self, t, context=()):
        self.contexts.append(context)
        return super().translate(t)


TWO = [1] * 10 + [0] * 10 + [1] * 10


def _go(**kw):
    async def go():
        mt = CtxMT()
        e, ev = await _run(TWO, mt=mt, **kw)
        return e, ev
    return asyncio.run(go())


def test_context_off_by_default_passes_nothing():
    e, ev = _go(mt_engine="qwen")
    assert e.mt.contexts == [(), ()]


def test_context_passes_previous_turn_per_stream():
    e, ev = _go(mt_engine="qwen", mt_context_turns=1)
    finals = [f["text"] for k, f in ev if k == "asr_final"]
    assert len(finals) == 2
    assert e.mt.contexts[0] == ()
    assert e.mt.contexts[1] == ((finals[0], "EN " + finals[0]),)


def test_context_ignored_for_marian():
    e, ev = _go(mt_engine="marian", mt_context_turns=2)
    assert e.mt.contexts == [(), ()]


class FakeTok:
    chat_template = ""

    def apply_chat_template(self, msgs, **kw):
        return "|".join(f"{m['role']}:{m['content']}" for m in msgs)


def _qwen(sense):
    q = K.QwenMLX.__new__(K.QwenMLX)
    q.tok, q._hybrid = FakeTok(), False
    q.system = K.QWEN_SYSTEM + (K.QWEN_SENSE if sense else "")
    return q


def test_prompt_unchanged_without_context_and_sense():
    q = _qwen(False)
    assert q.prompt("hola") == f"system:{K.QWEN_SYSTEM}|user:Spanish:\nhola"


def test_prompt_with_sense_and_context_keeps_current_turn_last():
    q = _qwen(True)
    p = q.prompt("me sale mucha plata", (("hablemos de pagos", "let's talk about payments"),))
    assert p.startswith("system:" + K.QWEN_SYSTEM + K.QWEN_SENSE)
    assert p.endswith("|user:Spanish:\nhablemos de pagos|assistant:let's talk about payments"
                      "|user:Spanish:\nme sale mucha plata")
    # el contexto va antes del turno actual: el prefijo que carga prefill() lo incluye
    assert q.prompt("me sale", (("hablemos de pagos", "let's talk about payments"),)).startswith(
        p[: p.rindex("|user:")])


def test_glossary_only_terms_present():
    g = K.qwen_glossary("No hay PLATA y laburamos al pedo")
    assert g == ["plata = money", "laburo = work, job; laburar = to work", "al pedo = for nothing, pointless"]
    assert K.qwen_glossary("Ni Mariana ni Gonzalo revisaron el PR.") == []
    assert K.qwen_glossary("Consiguió un laburo en un banco, pateamos la patente") == [
        "laburo = work, job; laburar = to work", "patear = to push back, postpone"]


def test_glossary_off_prompt_unchanged():
    q = _qwen(False)
    q.glossary = False
    assert q.prompt("no hay plata") == f"system:{K.QWEN_SYSTEM}|user:Spanish:\nno hay plata"


def test_glossary_on_goes_after_text_only_with_term():
    q = _qwen(False)
    q.glossary = True
    assert q.prompt("Ni Mariana ni Gonzalo") == f"system:{K.QWEN_SYSTEM}|user:Spanish:\nNi Mariana ni Gonzalo"
    p = q.prompt("Dale, no hay plata")
    assert p.startswith(f"system:{K.QWEN_SYSTEM}|user:Spanish:\nDale, no hay plata\n\nMeaning of the Argentine slang in it: ")
    assert "plata = money" in p and "dale = OK, sure" in p and p.endswith(K.QWEN_GUARD)
    # historia con el mismo formato: el prompt de la frase anterior es prefijo de la siguiente (caché de sesión)
    prev = q.prompt("Dale, no hay plata")
    nxt = q.prompt("Seguimos", (("Dale, no hay plata", "OK, there's no money"),))
    assert nxt.startswith(prev[: prev.rindex(K.QWEN_GUARD) + len(K.QWEN_GUARD)])

