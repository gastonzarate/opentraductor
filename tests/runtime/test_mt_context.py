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


def test_glossary_on_goes_before_text_only_with_term():
    q = _qwen(False)
    q.glossary = True
    assert q.prompt("Ni Mariana ni Gonzalo") == f"system:{K.QWEN_SYSTEM}|user:Spanish:\nNi Mariana ni Gonzalo"
    p = q.prompt("Dale, no hay plata")
    assert p.startswith(f"system:{K.QWEN_SYSTEM}|user:Meaning of the Argentine slang below: ")
    assert p.endswith(f"{K.QWEN_GUARD}\n\nSpanish:\nDale, no hay plata")
    assert "plata = money" in p and "dale" not in p.split("slang below: ")[1].split("\n")[0]
    assert K.qwen_glossary("Che, dale, lo miro") == []
    # el parcial estable con el mismo término comparte el prefijo con glosario (entra en la caché del prefill)
    part = q.prompt("Dale, no hay plata y")
    assert part.startswith(p)
    # historia con el mismo formato: el prompt de la frase anterior es prefijo de la siguiente (caché de sesión)
    prev = q.prompt("Dale, no hay plata")
    nxt = q.prompt("Seguimos", (("Dale, no hay plata", "OK, there's no money"),))
    assert nxt.startswith(prev)



def _remember_engine(**cfg):
    """T-68: Engine._mt_remember con un MT que cuenta 1 token por carácter del español."""
    from types import SimpleNamespace

    from runtime.config import PipelineConfig
    from runtime.pipeline import Engine

    e = SimpleNamespace(cfg=PipelineConfig(mt_engine="qwen", **cfg), _mt_ctx={},
                        mt=SimpleNamespace(turn_tokens=lambda es, en: len(es)))
    return e, (lambda es: Engine._mt_remember(e, "s", es, "en"))


def test_context_tokens_block_trim_to_half():
    e, rem = _remember_engine(mt_context_turns=200, mt_context_tokens=100, mt_session_kv=True)
    for i in range(10):
        rem(f"{i}" * 10)  # 10 tokens cada una: 100 no pasa el tope
    assert len(e._mt_ctx["s"]) == 10
    rem("x" * 10)  # 110 > 100: quedan las más nuevas hasta <= 50
    h = e._mt_ctx["s"]
    assert [es for es, _ in h] == ["6" * 10, "7" * 10, "8" * 10, "9" * 10, "x" * 10]
    rem("y" * 10)  # 60: no recorta (el prefijo de la caché de sesión no cambia)
    assert len(e._mt_ctx["s"]) == 6


def test_context_tokens_keeps_last_turn_even_if_huge():
    e, rem = _remember_engine(mt_context_turns=200, mt_context_tokens=100)
    rem("a" * 10)
    rem("b" * 500)
    assert [es for es, _ in e._mt_ctx["s"]] == ["b" * 500]


def test_context_tokens_off_keeps_turn_rule():
    e, rem = _remember_engine(mt_context_turns=2)
    for t in "abc":
        rem(t)
    assert [es for es, _ in e._mt_ctx["s"]] == ["b", "c"]


def test_glossary_general_rioplatense_terms():
    # T-68: modismos generales (no salen del set del 17/11)
    assert K.qwen_glossary("Eso sale dos palos") == [
        "'sale' with a price = it costs; '¿sale?' about a plan = is it on?", "un palo = a million (pesos or dollars)"]
    assert K.qwen_glossary("Lo cobra en negro, posta") == [
        "en negro (pay, jobs, contracts, rent) = off the books, undeclared", "posta = for real, the truth"]
    assert K.qwen_glossary("Lo miro al toque") == ["al toque = right away"]
    assert K.qwen_glossary("¿Lo hacemos? De una.") == ["de una = sure, right away"]


def test_glossary_general_no_false_positives():
    # palabras comunes en charlas de software que no deben disparar el glosario
    for t in ("Los tests están verdes", "El texto en negro sobre fondo blanco", "Cambiamos la manguera",
              "Hagámoslo de una vez", "De una manera u otra", "El palo de golf", "Pon las pruebas en verde",
              "Hay un banco en la esquina", "El banco me cobra comisión", "Precio final con IVA", "La línea de base"):
        assert K.qwen_glossary(t) == [], t


def test_glossary_t74_terms():
    # T-74: léxico general (AAL / Gobello), no sale del set del 17/11
    assert K.qwen_glossary("Los cambio al blue")[0].startswith("dólar blue = 'blue dollar'")
    assert K.qwen_glossary("así no me ahorco a fin de mes")[0].startswith("ahorcarse")
    assert K.qwen_glossary("me hizo precio") == ["hacer precio = to give a discount"]
    assert K.qwen_glossary("Tranqui, te banco")[0].startswith("bancar =")
    assert K.qwen_glossary("el contrato no está en negro")[0].startswith("en negro")
    # sentido literal: no disparan (pantalla en negro, rescatar personas, mangos fruta, camiseta en el lavarropas)
    for s in ("La pantalla está en negro", "Se rescató a tres personas", "Compré dos mangos",
              "Pongan la camiseta en el lavarropas", "Baja la línea de producción"):
        assert K.qwen_glossary(s) == [], s
    assert K.qwen_glossary("es medio garca")[0].startswith("garca")
