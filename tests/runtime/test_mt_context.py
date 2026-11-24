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


def test_context_off_passes_nothing():
    e, ev = _go(mt_engine="qwen", mt_context_turns=0)
    assert e.mt.contexts == [(), ()]


def test_context_on_by_default():
    # T-77: contexto largo con caché KV de sesión y tope de 2000 tokens; el especulativo sigue apagado
    from runtime.config import PipelineConfig
    c = PipelineConfig()
    assert (c.mt_context_turns, c.mt_session_kv, c.mt_context_tokens, c.mt_draft_dir) == (200, True, 2000, None)
    e, ev = _go(mt_engine="qwen")
    finals = [f["text"] for k, f in ev if k == "asr_final"]
    assert e.mt.contexts == [(), ((finals[0], "EN " + finals[0]),)]


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


def test_ctx_es_history_spanish_only_before_current_turn():
    q = _qwen(False)
    q.glossary, q.ctx_es = True, True
    ctx = (("hablemos de pagos", "let's talk about payments"), ("dale", "OK"))
    p = q.prompt("no hay plata", ctx)
    assert "let's talk" not in p and "assistant:" not in p
    assert p.startswith(f"system:{K.QWEN_SYSTEM}|user:{K.QWEN_CTX_ES}\nhablemos de pagos\ndale\n\n")
    assert p.endswith("Spanish:\nno hay plata")
    # sin contexto, igual que sin el flag; con una frase más, la historia anterior sigue siendo prefijo
    assert q.prompt("hola") == f"system:{K.QWEN_SYSTEM}|user:Spanish:\nhola"
    nxt = q.prompt("seguimos", ctx + (("no hay plata", "no money"),))
    assert nxt.startswith(p[: p.index("dale\n") + len("dale\n")])
    # el parcial estable entra en la caché de prefill(): anclado al último «Spanish:»
    assert q.prompt("no hay plata y", ctx).startswith(p[: p.rindex("Spanish:\n")])
    q.ctx_es = False
    assert "assistant:let's talk about payments" in q.prompt("no hay plata", ctx)


def test_topic_from_history_only():
    # tema claro: 2+ términos distintos del dominio y sin empate; solo la historia, no el parcial actual
    assert K.qwen_topic(["Hoy hacemos el deploy a producción", "Ojo con la rama de release"]) == \
        "deploying and maintaining software"
    assert K.qwen_topic(["El inquilino no pagó las expensas del depto"]) == "renting a home"
    assert K.qwen_topic(["Hola, ¿cómo andás?", "Bien, todo tranqui"]) is None
    assert K.qwen_topic(["Revisamos el servidor"]) is None  # 1 solo término
    assert K.qwen_topic([]) is None


def test_topic_flag_prompt():
    q = _qwen(False)
    ctx = (("Hoy hacemos el deploy a producción", "We deploy to production today"),
           ("Revisá la rama antes del merge", "Check the branch before the merge"))
    q.topic = False
    off = q.prompt("dale, lo subo", ctx)
    assert "Topic of the conversation" not in off
    q.topic = True
    p = q.prompt("dale, lo subo", ctx)
    assert p.endswith("|user:Topic of the conversation: deploying and maintaining software.\nSpanish:\ndale, lo subo")
    assert p[: p.rindex("|user:")] == off[: off.rindex("|user:")]  # historia igual: misma caché de sesión
    # sin contexto o sin tema claro, el prompt no cambia
    assert q.prompt("hola") == f"system:{K.QWEN_SYSTEM}|user:Spanish:\nhola"
    assert "Topic" not in q.prompt("hola", (("che", "hey"),))
    # el parcial estable sigue entrando en la caché de prefill()
    assert q.prompt("dale, lo subo y", ctx).startswith(p[: p.rindex("Spanish:\n")])
    # con historia en español (palanca 2) va después del bloque de historia
    q.ctx_es = True
    pe = q.prompt("dale, lo subo", ctx)
    assert "rama antes del merge\n\nTopic of the conversation: deploying" in pe


def test_topic_literal_controls_keep_no_glossary():
    # los controles de sentido literal (seña como gesto, se me cayó el cargador) no reciben glosa: solo el tema
    for s in ("Me hizo una seña con la mano para que pase", "Se me cayó el cargador atrás del escritorio"):
        assert K.qwen_glossary(s, ext=True) == []



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
    e, rem = _remember_engine(mt_context_turns=2, mt_context_tokens=0, mt_session_kv=False)  # camino off (T-77)
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
        # solo la lista base: con DOM (por defecto desde 18/11) «dos mangos» dispara «mangos = pesos» (número + mangos)
        assert K.qwen_glossary(s, ext=False, dom=False) == [], s
    assert K.qwen_glossary("es medio garca")[0].startswith("garca")


def test_glossary_ext_dom_on_by_default(monkeypatch):
    # T-74 (18/11): EXT y DOM encendidos por defecto; =0 los apaga
    monkeypatch.delenv("OPENTRADUCTOR_GLOSSARY_EXT", raising=False)
    monkeypatch.delenv("OPENTRADUCTOR_GLOSSARY_DOM", raising=False)
    assert K.qwen_glossary("no funca el build") == ["funcar = to work, function (no funca = it doesn't work)"]
    assert "contracargo = chargeback" in K.qwen_glossary("El cliente pidió un contracargo")
    monkeypatch.setenv("OPENTRADUCTOR_GLOSSARY_EXT", "0")
    monkeypatch.setenv("OPENTRADUCTOR_GLOSSARY_DOM", "0")
    assert K.qwen_glossary("no funca el build") == []
    assert K.qwen_glossary("El cliente pidió un contracargo") == []


def test_glossary_ext_terms():
    # T-74 palanca 3: Wiktionary (CC BY-SA 4.0), Rioplatense/Argentine Spanish/Lunfardo; no sale del 17/11 ni de v1
    g = lambda s: K.qwen_glossary(s, ext=True)
    assert g("ponele que son diez")[0].startswith("ponele")
    assert g("el tipo tiene mucha cancha")[0].startswith("tener cancha")
    assert g("quedé hecho pelota")[0].startswith("hecho pelota")
    assert g("hay mucha rosca en el directorio")[0].startswith("rosca")
    assert g("te lo digo sin joda")[0].startswith("en joda")
    assert g("compré dólar MEP")[0].startswith("dólar MEP")
    # la lista base no cambia con ext
    assert g("Tranqui, te banco")[0].startswith("bancar =")


def test_glossary_ext_literal():
    # sentido literal o común: no disparan
    for s in ("Los tests están verdes", "Jugamos en la cancha de fútbol", "La rosca del tornillo",
              "Pasame el coso de la mesa", "El manual de usuario", "La pelota está pinchada",
              "Me pinta la pared", "Hay un problema en el servidor", "El dólar oficial subió",
              "El cajón del escritorio", "Una joda bárbara"):
        assert [e for e in K.qwen_glossary(s, ext=True) if e not in K.qwen_glossary(s, ext=False)] == [], s


def _dom_only(s):
    return [g for g in K.qwen_glossary(s, ext=True, dom=True) if g not in K.qwen_glossary(s, ext=True, dom=False)]


def test_glossary_dom_fires_in_technical_and_tax_context():
    assert any(g.startswith("pisar") for g in _dom_only("Ojo que me pisaste el commit de ayer"))
    assert any(g.startswith("subir") for g in _dom_only("Subimos el hotfix a producción"))
    assert any(g.startswith("bajar") for g in _dom_only("Bajá el release, volvemos a la 1.2.3"))
    assert any(g.startswith("se cayó") for g in _dom_only("Se cayó el servidor de pagos"))
    assert any(g.startswith("sacar") for g in _dom_only("Saquemos el login del alcance del sprint"))
    assert "contracargo = chargeback" in _dom_only("El cliente pidió un contracargo")
    assert any(g.startswith("retención") for g in _dom_only("Te hacen la retención de IVA en la factura"))
    assert "mangos = pesos, bucks" in _dom_only("Me cobró ochocientos mangos")
    assert _dom_only("Nos vemos el 12/10 en la oficina") == ["12/10 = October 12 (Argentine dates are day/month)"]


def test_glossary_dom_subir_bajar_by_sense():
    # run14: un solo sentido por objeto (rama/repo, producción, servicio, release, archivo)
    assert _dom_only("Subí los commits a la rama de staging") == ["subir (code, a fix, commits) to a branch or repo = to push"]
    assert any("to deploy, release" in g for g in _dom_only("Subimos el hotfix a producción"))
    assert _dom_only("Mañana bajamos el servidor de reportes") == ["bajar (a service, server, site) = to take down, shut down"]
    assert any("to pull" in g for g in _dom_only("Bajate los cambios del remoto antes de empezar"))
    assert _dom_only("Bajá el release, volvemos a la 1.2.3") == ["bajar (a deploy, release or version) = to roll back"]
    assert _dom_only("Bajate el instalador de la página") == ["bajar (a file, app, installer) = to download"]


def test_glossary_dom_literal_controls():
    # sentido literal o sin contexto técnico: DOM no agrega nada
    for s in ("Subí la escalera hasta el tercer piso", "Pisé un charco en la vereda", "Bajá el volumen de la tele",
              "Se me cayó el cargador atrás del escritorio", "Me hizo una seña con la mano", "Sacá la basura",
              "Comí tres cuartos, o sea 3/4 de la pizza", "Retención de líquidos", "Compré mangos en la verdulería"):
        assert _dom_only(s) == [], s
    # «de plata» como metal: no se glosa como money
    g = K.qwen_glossary("Le regalé una cadena de plata", ext=True, dom=True)
    assert "plata = money" not in g and "de plata (an object) = made of silver" in g
    assert "plata = money" in K.qwen_glossary("No hay plata para el viaje", ext=True, dom=True)
    assert "plata = money" in K.qwen_glossary("Le regalé una cadena de plata", ext=True, dom=False)


def test_glossary_dom_not_in_history_turns(monkeypatch):
    # T-74: la glosa DOM va solo en la frase que la dispara, no en los turnos de historia (se arrastraba)
    monkeypatch.setenv("OPENTRADUCTOR_GLOSSARY_DOM", "1")
    q = _qwen(True)
    q.glossary = True
    ctx = (("Y si nada funciona, bajá el servicio entero", "If nothing works, roll back the whole service"),)
    p = q.prompt("Bajá la pantalla de la notebook", ctx)
    assert "= to take down" not in p and "= to roll back" not in p
    p2 = q.prompt("Bajá el release, volvemos a la 1.2.3", ctx)
    assert p2.count("bajar (a deploy") == 1 and p2.rindex("bajar (a deploy") > p2.rindex("|user:")
    assert any("credited, deposited" in g for g in _dom_only("Hoy me acreditaron el sueldo"))


def _qwen_init(monkeypatch, env=None, **kw):
    import sys
    import types

    fake = types.ModuleType("mlx_lm")
    fake.load = lambda d: (object(), FakeTok())
    monkeypatch.setitem(sys.modules, "mlx_lm", fake)
    monkeypatch.delenv("OPENTRADUCTOR_MT_CONCISE", raising=False)
    if env is not None:
        monkeypatch.setenv("OPENTRADUCTOR_MT_CONCISE", env)
    return K.QwenMLX("x", sense_prompt=True, **kw)


def test_concise_off_by_default_keeps_system(monkeypatch):
    from runtime.config import PipelineConfig

    assert PipelineConfig().mt_concise is False
    q = _qwen_init(monkeypatch)
    assert q.system == K.QWEN_SYSTEM + K.QWEN_SENSE and not q.concise


def test_concise_flag_and_env_append_to_system(monkeypatch):
    assert _qwen_init(monkeypatch, concise=True).system == K.QWEN_SYSTEM + K.QWEN_SENSE + K.QWEN_CONCISE
    assert _qwen_init(monkeypatch, env="1").system.endswith(K.QWEN_CONCISE)
    assert _qwen_init(monkeypatch, env="0", concise=True).system == K.QWEN_SYSTEM + K.QWEN_SENSE
