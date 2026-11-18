"""Componentes de la baseline: VAD (Silero), ASR (Parakeet MLX), MT (Marian
CTranslate2) y TTS (Kokoro MLX). Sin red: todo desde disco.

Nota MLX: los arrays y streams de MLX son por hilo. Cargar y usar ASR/TTS en
un MISMO hilo dedicado (ver ``pipeline.ModelWorker``).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np

from . import config as C

VAD_WINDOW = 512  # muestras @16 kHz (32 ms), requerido por Silero


def pcm_to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def float_to_pcm(x: np.ndarray) -> bytes:
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


# ----------------------------------------------------------------------- VAD
_VAD_CONTEXT = 64  # muestras de contexto que el modelo ONNX espera delante de cada ventana @16 kHz


class SileroVAD:
    """Probabilidad de voz por ventana de 32 ms, con estado. Una instancia por
    stream (``reset`` entre streams).

    Corre el ONNX oficial de Silero con onnxruntime (CPU, 1 hilo) y numpy, sin
    torch. Mismo cálculo que ``silero_vad.OnnxWrapper``: entrada = 64 muestras de
    contexto + 512 de la ventana, estado (2, 1, 128).
    """

    def __init__(self) -> None:
        import onnxruntime as ort

        path = C.silero_vad_onnx_path()
        if not path.exists():
            raise FileNotFoundError(
                f"Falta el modelo Silero VAD ONNX en {path}. "
                "Copia silero_vad.onnx a runtime/data/ o define OPENTRADUCTOR_SILERO_ONNX."
            )
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self._sess = ort.InferenceSession(
            str(path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._sr = np.array(16000, dtype=np.int64)
        self._rest = np.zeros(0, dtype=np.float32)
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._ctx = np.zeros((1, _VAD_CONTEXT), dtype=np.float32)

    def reset_states(self) -> None:
        """Reinicia solo la memoria del modelo (fin de segmento). Conserva el resto
        de muestras sin procesar, igual que ``model.reset_states()`` del Silero original."""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._ctx = np.zeros((1, _VAD_CONTEXT), dtype=np.float32)

    def reset(self) -> None:
        self.reset_states()
        self._rest = np.zeros(0, dtype=np.float32)

    def _window_prob(self, w: np.ndarray) -> float:
        x = np.concatenate([self._ctx, w[None, :]], axis=1)
        out, self._state = self._sess.run(
            None, {"input": x, "state": self._state, "sr": self._sr}
        )
        self._ctx = x[:, -_VAD_CONTEXT:]
        return float(out[0, 0])

    def probs(self, pcm: bytes) -> list[tuple[int, float]]:
        """Devuelve [(fin_de_ventana_en_muestras_desde_el_inicio_de_ESTE_chunk, prob)].

        El offset descuenta el resto de la llamada anterior, así que la
        posición absoluta es ``muestras_previas_al_chunk + offset`` (puede ser
        <= 0 para la primera ventana si el resto era largo).
        """
        prev = len(self._rest)
        x = np.concatenate([self._rest, pcm_to_float(pcm)])
        out = []
        n = len(x) // VAD_WINDOW
        for i in range(n):
            w = x[i * VAD_WINDOW : (i + 1) * VAD_WINDOW]
            out.append(((i + 1) * VAD_WINDOW - prev, self._window_prob(w)))
        self._rest = x[n * VAD_WINDOW :]
        return out


# ----------------------------------------------------------------------- ASR
class ParakeetStreamer:
    """ASR streaming con caché de Parakeet-TDT (parakeet-mlx)."""

    def __init__(self, context: tuple[int, int]) -> None:
        import mlx.core as mx
        from parakeet_mlx import from_pretrained

        self._mx = mx
        path = C.snapshot(C.PARAKEET_REPO, C.PARAKEET_REV, ("config.json", "model.safetensors"))
        self.model = from_pretrained(str(path))
        self.context = context
        self._ctx = None
        self._st = None

    def start(self) -> None:
        self.finish()
        self._ctx = self.model.transcribe_stream(context_size=self.context)
        self._st = self._ctx.__enter__()

    def add(self, pcm: bytes) -> str:
        if self._st is None:
            self.start()
        self._st.add_audio(self._mx.array(pcm_to_float(pcm)))
        return self._st.result.text.strip()

    def finish(self) -> str:
        text = ""
        if self._st is not None:
            text = self._st.result.text.strip()
            self._ctx.__exit__(None, None, None)
        self._ctx = self._st = None
        return text


class MoonshineStreamer:
    """ASR streaming con Moonshine (CPU/ONNX, ``moonshine-voice``). Misma
    interfaz que ``ParakeetStreamer`` (start/add/finish).

    ``add`` devuelve el texto acumulado según los eventos que el motor haya
    emitido hasta ese momento: el motor decide cuándo recalcula (piso
    ``update_interval``), así que ``add`` puede devolver el texto anterior.
    ``finish`` llama a ``stop()`` del stream, que transcribe el audio restante
    y completa las líneas (vaciado final). Solo disco, sin red.
    """

    def __init__(self, model_dir, arch: str, update_interval: float) -> None:
        try:
            from moonshine_voice import ModelArch, Transcriber
        except ImportError as exc:  # noqa: BLE001
            raise ModuleNotFoundError(
                "Falta moonshine-voice en este entorno (pip: moonshine-voice==0.1.6)"
            ) from exc

        self.update_interval = update_interval
        self.tr = Transcriber(
            model_path=str(model_dir),
            model_arch=getattr(ModelArch, arch),
            update_interval=update_interval,
        )
        self._stream = None
        self._lines: dict[int, str] = {}
        self.last_latency_ms = 0

    def _on_event(self, ev) -> None:
        line = getattr(ev, "line", None)
        if line is None:
            return
        self._lines[line.line_id] = line.text
        self.last_latency_ms = getattr(line, "last_transcription_latency_ms", 0)

    def _text(self) -> str:
        return " ".join(t.strip() for _, t in sorted(self._lines.items()) if t.strip())

    def start(self) -> None:
        self.finish()
        self._lines = {}
        self._stream = self.tr.create_stream(update_interval=self.update_interval)
        self._stream.add_listener(self._on_event)
        self._stream.start()

    def add(self, pcm: bytes) -> str:
        if self._stream is None:
            self.start()
        self._stream.add_audio(pcm_to_float(pcm).tolist(), 16000)
        return self._text()

    def close(self) -> None:
        """Libera el stream abierto y el Transcriber nativo. Idempotente."""
        try:
            self.finish()
        finally:
            tr, self.tr = self.tr, None
            if tr is not None:
                tr.close()

    def finish(self) -> str:
        text = ""
        if self._stream is not None:
            try:
                self._stream.stop()
                text = self._text()
            finally:
                self._stream.close()
                self._stream = None
        return text


# ------------------------------------------------------------------------ MT
# BUG-7: Marian traduce "120 mil" como "$120 billion" y omite la 2.ª oración de una entrada.
# Separa solo tras . ! ? seguidos de mayúscula o ¿ ¡ (no corta "1,2" ni "2.0.0").
_MT_SENT = re.compile(r"(?<=[.!?])\s+(?=[¿¡A-ZÁÉÍÓÚÑ])")
_MT_MIL = re.compile(r"\b(\d{1,3}) mil\b")


def mt_split_sentences(text: str) -> list[str]:
    return [p for p in _MT_SENT.split(text.strip()) if p]


# BUG-7 (frase 28): Marian pasa "a las 19" a "7:00" y pierde la tarde; con "7 p. m." la conserva.
# "HH:MM" (13-23) siempre; "HH" solo si sigue fin, puntuación o h/hs/horas (no toca "las 19 personas").
_MT_HORA = re.compile(
    r"\b(las|la)\s+(1[3-9]|2[0-3])"
    r"(?:(:[0-5]\d)(?:\s*(?:h|hs|horas)\b)?|(?:\s*(?:h|hs|horas)\b)?(?=\s*(?:[.,;:!?)]|$)))"
)


def _hora_12(g: re.Match) -> str:
    h = int(g.group(2)) - 12
    return f"{'la' if h == 1 else 'las'} {h}{g.group(3) or ''} p. m."


def mt_normalize_numbers(text: str) -> str:
    """'120 mil' -> '120.000' (1 a 3 cifras; no toca '1,2 millones') y 'a las 19' -> 'a las 7 p. m.'."""
    return _MT_HORA.sub(_hora_12, _MT_MIL.sub(lambda g: g.group(1) + ".000", text)).replace("p. m..", "p. m.")


class MarianCT2:
    def __init__(self, beam_size: int = 1, threads: int = 4, normalize: bool = False) -> None:
        import ctranslate2
        import sentencepiece as spm

        d = C.marian_ct2_dir()
        if not (d / "model.bin").exists():
            raise FileNotFoundError(
                f"Falta el artefacto CTranslate2 en {d}. "
                "Ejecuta: uv run --extra runtime python -m runtime.setup"
            )
        self.sp_s = spm.SentencePieceProcessor(model_file=str(d / "source.spm"))
        self.sp_t = spm.SentencePieceProcessor(model_file=str(d / "target.spm"))
        self.tr = ctranslate2.Translator(
            str(d), device="cpu", inter_threads=1, intra_threads=threads
        )
        self.beam = beam_size
        self.normalize = normalize

    def translate(self, text_es: str) -> str:
        text_es = text_es.strip()
        if not text_es:
            return ""
        if self.normalize:
            parts = mt_split_sentences(mt_normalize_numbers(text_es))
            return " ".join(t for t in (self._translate_one(p) for p in parts) if t)
        return self._translate_one(text_es)

    def _translate_one(self, text_es: str) -> str:
        toks = self.sp_s.encode(text_es, out_type=str) + ["</s>"]
        res = self.tr.translate_batch(
            [toks], beam_size=self.beam, max_decoding_length=256
        )
        return self.sp_t.decode(res[0].hypotheses[0]).strip()


# ----------------------------------------------------------------------- TTS
_CLAUSE = re.compile(r"(?<=[.!?;:,])\s+")


# Palabras que no conviene dejar al final de un fragmento (prosodia): el corte se mueve antes.
_NO_END = frozenset(
    "a an the of to in on at for from with by and or but nor so if as than that this these those "
    "is are was were be been it's i'm i you he she we they my your his her our their not no do does did "
    "can could will would should may might must when while because which who where how what then about into over under".split()
)


# Conjunciones coordinantes: con ``bind_conjunctions`` no se corta justo ANTES ni justo DESPUÉS de ellas
# (la conjunción queda dentro de su grupo, a mitad de fragmento).
_CONJ = frozenset({"and", "or", "but", "nor", "yet"})


# T-51: mismo prompt que la app diaria (MP4-A) y que T-50 (probe_qwen_sizes.py).
QWEN_SYSTEM = ("You translate Argentine Spanish speech from software team meetings into natural US English. "
               "Output only the English translation, with no notes, quotes or explanations. "
               "Keep numbers, negations, names and technical terms exact.")
# Pedido de la persona (17/11): traducir por el sentido según el tema de la charla, no palabra por palabra.
# Se agrega a QWEN_SYSTEM con ``mt_sense_prompt`` (apagado hasta el A/B de frases ambiguas argentinas).
QWEN_SENSE = (" Translate the meaning in the context of the conversation, not word by word: pick the sense that "
              "fits the topic being discussed (e.g. 'plata' in a talk about payments is money, never silver). "
              "Render Argentine slang and idioms with natural English equivalents (plata/guita = money, "
              "laburo = work/job, re = really, bárbaro = great, 'salir' a price = to cost, 'dale' = OK/sure, "
              "luca = a thousand pesos, afano = rip-off/theft, al pedo = for nothing/pointless, "
              "fiaca = laziness). "
              "Earlier turns are context only: translate just the last Spanish message.")
# T-74 palanca 2 (OPENTRADUCTOR_MT_CTX_ES=1): encabezado del bloque de historia en español, ver QwenMLX.prompt().
QWEN_CTX_ES = "Earlier in the meeting (Spanish, context only, do not translate):"
# T-58 opción acotada (17/11): QWEN_SENSE completo bajó el BLEU del gate 72,8 -> 64,6 (perdió «ni... ni», imperativos,
# «gotta»). Con ``mt_sense_glossary`` solo las frases con un término de esta lista llevan, ANTES del texto y dentro
# del prefijo que cachea prefill() (ver QwenMLX._user), el glosario de los términos que aparecen y QWEN_GUARD. Sin instrucción de parafrasear.
QWEN_GLOSSARY = (
    (r"\bplata\b", "plata = money"),
    (r"\bguita\b", "guita = money"),
    (r"\bun mango\b", "no tener un mango = to be broke, not a penny"),
    (r"\blucas?\b", "luca = a thousand pesos (diez lucas = ten thousand)"),
    (r"\blabur\w*", "laburo = work, job; laburar = to work"),
    (r"\bsal(e|en|ió|dría|ía)\b", "'sale' with a price = it costs; '¿sale?' about a plan = is it on?"),
    (r"\bre\b", "re + adjective = really"),
    (r"\bbárbaro\b", "bárbaro = great"),
    # dale y che fuera: Qwen ya los traduce bien y la glosa cambiaba el registro (gate 6f57a1a: -1,4 BLEU solo por dev-03/dev-13)
    (r"\bboludo\b", "boludo = dude"),
    (r"\bquilombo\b", "quilombo = mess"),
    (r"\bbanc(ar|ame|ate|ás|amos|ó|an|aste)\b|\b(te|lo|la|los|las|me|nos|les?) banco\b",
     "bancar = to handle, put up with; to back or cover someone (te banco = I've got you / I'll cover you); "
     "¿te bancás + verb? = are you up for; bancame = hold on"),
    (r"\bfiaca\b", "fiaca = laziness, not feeling like it"),
    (r"\bal pedo\b", "al pedo = for nothing, pointless"),
    (r"\bzaf(ar|amos|ó|é|aste|aron)\b", "zafar = to get by, scrape through, be OK; to get away with it"),
    (r"\bchamuyo\b", "chamuyo = smooth talk, BS"),
    (r"\bgarrón\b", "garrón = bummer"),
    (r"\bcopad[oa]s?\b", "copado = cool"),
    (r"\bpiola\b", "piola = cool, nice"),
    (r"\bpate(ar|amos|ás|a|ó|o|emos|émos)\w*", "patear = to push back, postpone"),
    (r"\bpila\b", "con toda la pila = full of energy; una pila de = a lot of"),
    (r"\bafan\w*", "afano = rip-off; afanar = to steal"),
    # T-68 (18/11, pedido de @lead): modismos rioplatenses generales, de lunfardo de uso común en trabajo y plata.
    # La lista NO sale de los fallos del set del 17/11 (inflaría la métrica); el banco v1 controla el sobreajuste.
    # Patrones acotados para no disparar en frases comunes (ver test_glossary_general_no_false_positives).
    (r"\b(un|medio|dos|tres|cuatro|cinco|diez|\d+) palos?\b", "un palo = a million (pesos or dollars)"),
    (r"\b(\d+|cien|mil|quinientos|doscientos) verdes\b", "verdes = US dollars"),
    (r"\bd[óo]lar blue\b|\b(el|al) blue\b", "dólar blue = 'blue dollar' (keep the word 'blue'; it is Argentina's parallel-market dollar)"),
    (r"\bgarp\w*", "garpar = to pay"),
    (r"\bmangue(ar|o|a|ás|ando|é|aste|amos)\b", "manguear = to bum, ask for money or favors"),
    (r"\b(cobr|pag|trabaj|labur|contrat|ten)\w*( \w+)? en negro\b|\b(contrato|sueldo|alquiler|trabajo|laburo|empleo|pago)s?\b.*\best[áa]n? en negro\b",
     "en negro (pay, jobs, contracts, rent) = off the books, undeclared"),
    (r"\bchang(a|as|uita)\b", "changa = odd job, side gig"),
    (r"\bcurr(o|os|ar|ando)\b", "curro = racket, easy money scheme"),
    (r"\bñoquis?\b", "ñoqui = no-show employee who still gets paid"),
    (r"\braj(ar|aron|ó|aste|amos|en|ame)\b", "rajar = to fire someone; rajarse = to leave fast"),
    (r"\bposta\b", "posta = for real, the truth"),
    (r"\bni en pedo\b", "ni en pedo = no way"),
    (r"\bal toque\b", "al toque = right away"),
    (r"\bun toque\b", "un toque = a little, a bit"),
    (r"\bde una\b(?=\s*[.,!?]|\s*$)", "de una = sure, right away"),
    (r"\bbocha\b", "una bocha = a lot"),
    (r"\buna banda\b", "una banda (de) = a lot (of)"),
    (r"\blas pilas\b", "ponerse las pilas = to step up, get one's act together"),
    (r"\bbardo\b", "bardo = trouble, mess"),
    (r"\bboludez\w*|\bpavada\w*", "boludez/pavada = silly thing, trifle"),
    (r"\bchant(a|as)\b", "chanta = phony, con artist"),
    (r"\btruch[oa]s?\b", "trucho = fake, bogus"),
    (r"\bgros[s]?[oa]s?\b", "groso = great, a big shot"),
    (r"\bjoya\b", "joya = great, perfect"),
    (r"\bzarpad[oa]s?\b|\bsarpad[oa]s?\b", "zarpado = over the top; awesome"),
    (r"\bembol\w*", "embole = drag, boring; embolarse = to get bored"),
    (r"\bbaj[óo]n\b", "bajón = bummer, downer"),
    (r"\bmacana\b", "macana = blunder; qué macana = what a shame"),
    (r"\bcagad(a|as)\b", "cagada = screwup"),
    (r"\bpifi\w*|\bpifie\w*", "pifiar = to get it wrong, miss"),
    (r"\bflashe\w*", "flashear = to imagine things, get carried away"),
    (r"\bhinch(ar|a|ás|ando|ó)\b", "hinchar = to bug, annoy"),
    (r"\bmanija\b", "manija = hyped, eager"),
    (r"\bmufa\b|\byeta\b", "mufa/yeta = jinx, bad luck"),
    (r"\bortiva\b", "ortiva = killjoy"),
    (r"\bchab[óo]n\b|\bchabona\b", "chabón = guy"),
    (r"\bpib[ea]s?\b", "pibe/piba = kid, young guy/girl"),
    (r"\btranqui\b", "tranqui = relax, no worries"),
    (r"\bla gamba\b", "hacer la gamba = to back someone up"),
    # T-74 (18/11): léxico rioplatense general. Polisémicas acotadas al sentido lunfardo (al horno, la camiseta, con alambre,
    # chorro de...) y fuera careta y buena/mala onda (Qwen ya las traduce y careta = máscara mete ruido). Fuentes: Diccionario del habla de los argentinos (AAL) y Diccionario
    # lunfardo (Gobello) para garca, chorro, ahorcado, hacer precio, morfar, bondi, birra, cana, cheto, careta, mangos,
    # al horno, cajonear, sarasa, ni ahí, de onda/buena onda, hacerse el boludo, chupar un huevo, rescatarse;
    # uso común en reuniones de trabajo para bajar línea, ponerse la camiseta, atar con alambre, agarrar viaje, un cacho.
    # Ninguna sale de los fallos ni de las traducciones esperadas del set del 17/11.
    (r"\bgarc(a|as|ada|adas|ó|aron|ar|arte|arme)\b", "garca = crook, swindler; garcar = to screw someone over"),
    (r"\bchorr(o|os|a|as)\b(?! de)", "chorro = thief"),
    (r"\b(me|te|nos|se) ahorc(o|a|ás|amos|an)\b|\bahorcad[oa]s?\b",
     "ahorcarse / andar ahorcado (with money or payments) = to be squeezed, stretched thin financially (never literally hang)"),
    (r"\b(hizo|hice|hacés|hace|hacen|hicieron|haceme|hacer|hago) (un )?precio\b", "hacer precio = to give a discount"),
    (r"\bmorf(ar|amos|ás|é|ó|aron|o)\b", "morfar = to eat"),
    (r"\bbondis?\b", "bondi = bus"),
    (r"\bbirras?\b", "birra = beer"),
    (r"\bla cana\b|\ben cana\b", "la cana = the cops; en cana = in jail"),
    (r"\bchet(o|a|os|as)\b", "cheto = posh, snobby"),
    (r"\b(\d+|diez|veinte|cien|mil) mangos\b", "mangos = pesos, bucks"),
    (r"\b(estoy|estás|está|estamos|están|estaba|estábamos|quedé|quedó|quedamos) al horno\b", "estar al horno = to be in deep trouble"),
    (r"\bcajone(ar|aron|ó|amos|ado|ada)\b", "cajonear = to shelve, sit on something"),
    (r"\bsarasa\b", "sarasa = blah blah, empty talk"),
    (r"\bni ahí\b", "ni ahí = not at all, no way"),
    (r"\bhac(e|és|erse|iendo|ete)(se|te)? el (boludo|otario|sota|tonto)\b|\bse hizo el (boludo|otario|sota|tonto)\b",
     "hacerse el boludo = to play dumb"),
    (r"\bchupa un huevo\b|\bchupa tres pepinos\b", "me chupa un huevo = I couldn't care less"),
    (r"\brescat[aá]te\b", "rescatate = get it together, wise up"),
    (r"\bbaj(ar|ó|aron|a|ando) l[íi]nea\b", "bajar línea = to lecture, tell people what to think"),
    (r"\b(me|te|se|nos) pon\w* la camiseta\b|\bpon(erse|ete|ganse|ernos|értela|érsela)\w* la camiseta\b", "ponerse la camiseta = to commit fully to the team"),
    (r"\b(atad|atat|emparchad)[oa]s? con alambre\b|\batad[oa] con alambre\b", "atado con alambre = held together with duct tape, jury-rigged"),
    (r"\bagarr(ó|aron|ar|a|ás|amos) viaje\b", "agarrar viaje = to go along with it, take up an offer"),
    (r"\bun cacho\b", "un cacho = a bit, a while"),
)
# 1.ª versión («Keep ... exactly as in the Spanish») hacía copiar el modismo (put more guita, ten lucas): 21/30.
QWEN_GUARD = ("Do not copy those Argentine words into the English: write their meaning instead. "
              "Translate the rest of the sentence as usual.")


# T-74 palanca 3 (OPENTRADUCTOR_GLOSSARY_EXT=1, apagada por defecto): ampliación con fuente general y licencia
# compatible. Fuente única: Wiktionary en inglés, categorías «Rioplatense Spanish», «Argentine Spanish» y «Lunfardo»
# (CC BY-SA 4.0; solo se toma la lista de lemas y su sentido, redactado aquí). Elegidas por uso en charlas de trabajo,
# plata y reuniones; ninguna sale de los fallos del set del 17/11 ni del banco v1. Las polisémicas solo disparan con
# el contexto que fija el sentido rioplatense (ver test_glossary_ext_literal).
QWEN_GLOSSARY_EXT = (
    (r"\bfunc(a|an|ó|aba|aban|ar|ando)\b", "funcar = to work, function (no funca = it doesn't work)"),
    (r"\bponele\b", "ponele = let's say, say"),
    (r"\bberretas?\b", "berreta = cheap, shoddy"),
    (r"\bcos(o|ito)\b(?! de)", "coso = thingy, whatchamacallit"),
    (r"\brelojea\w*", "relojear = to eye, check out"),
    (r"\bchusme(ar|o|á|ás|aste|é|ó|ando)\b", "chusmear = to snoop, take a quick look; to gossip"),
    (r"\bengrup\w*", "engrupir = to fool, con"),
    (r"\bdeschav\w*", "deschavar = to give away, reveal"),
    (r"\bes un montón\b", "es un montón = it's a lot, it's too much"),
    (r"\bcuentapropistas?\b", "cuentapropista = self-employed, freelancer"),
    (r"\bmonotributistas?\b", "monotributista = self-employed contractor (Argentine simplified tax regime)"),
    (r"\bel cepo\b|\bcepo cambiario\b", "el cepo = Argentina's currency controls"),
    (r"\bd[óo]lar mep\b", "dólar MEP = MEP dollar (keep 'MEP'; legal stock-market exchange rate)"),
    (r"\b(tiene|tenés|tengo|tienen|tenemos|con|falta|le falta|les falta) (mucha |poca )?cancha\b", "tener cancha = to be experienced, savvy"),
    (r"\bcancher[oa]s?\b", "canchero = savvy, self-assured"),
    (r"\bfuler[oa]s?\b", "fulero = ugly, bad, dodgy"),
    (r"\bbarde(ar|ó|aron|ás|a|ando)\b", "bardear = to trash-talk, mess with"),
    (r"\bghoste\w*", "ghostear = to ghost"),
    (r"\bpon(er|é|emos|ele|erle|gan|gamos)(le)? (más |un poco de |toda la )?onda\b", "ponerle onda = to put some effort, enthusiasm into it"),
    (r"\bhech[oa]s? pelota\b", "hecho pelota = wrecked, exhausted"),
    (r"\bno te hagas problema\b", "no te hagas problema = don't worry about it"),
    (r"\bse me prendió la lamparita\b|\bprend\w* la lamparita\b", "prendérsele la lamparita = to have a bright idea"),
    (r"\bsi te digo te miento\b", "si te digo te miento = honestly, I don't know"),
    (r"\bde manual\b", "de manual = textbook, classic"),
    (r"\bdato de color\b", "dato de color = fun fact, side detail"),
    (r"\bgatill(ar|ó|ás|á|aste|amos|aron)\b", "gatillar = to pay up, shell out"),
    (r"\bde cajón\b", "de cajón = obviously, a given"),
    (r"\bsi (te |me |nos )?pinta\b", "si pinta = if we feel like it, if it works out"),
    (r"\bni a cañonazos\b", "ni a cañonazos = no way, not a chance"),
    (r"\b(hac\w+|mucha|pura) rosca\b", "rosca = backroom politics, scheming"),
    (r"\b(en|sin) joda\b", "en joda = joking; sin joda = seriously"),
    (r"\batenti\b", "atenti = heads up, pay attention"),
)


def qwen_glossary(text_es: str, ext: bool | None = None) -> list[str]:
    """Entradas de QWEN_GLOSSARY cuyos términos aparecen en ``text_es`` (sin distinguir mayúsculas), en orden.
    Con ``ext`` (por defecto, env OPENTRADUCTOR_GLOSSARY_EXT=1) suma QWEN_GLOSSARY_EXT al final."""
    if ext is None:
        ext = os.environ.get("OPENTRADUCTOR_GLOSSARY_EXT") == "1"
    pats = QWEN_GLOSSARY + QWEN_GLOSSARY_EXT if ext else QWEN_GLOSSARY
    return [g for pat, g in pats if re.search(pat, text_es, re.I)]


class QwenMLX:
    """MT ES->EN con Qwen3 4bit en MLX (mlx-lm). Greedy, corte en el primer salto de línea, max_tokens = 3n+16.

    Usa la GPU: el pipeline lo llama desde el mismo hilo que Kokoro (MLX no se comparte entre hilos).
    ``normalize`` aplica el mismo preproceso de BUG-7 que Marian (cifras y horas, sin partir oraciones:
    Qwen no omite la 2.ª oración y partirla costaría una llamada más)."""

    def __init__(self, model_dir, normalize: bool = False, sense_prompt: bool = False,
                 session_kv: bool = False, sense_glossary: bool = False, draft_dir=None,
                 num_draft_tokens: int = 3) -> None:
        from mlx_lm import load

        self.model, self.tok = load(str(model_dir))
        # T-68 decodificación especulativa: Qwen3 chico (mismo tokenizer) como borrador; greedy da la misma salida.
        # La caché KV es la del modelo + la del borrador en una sola lista (como la espera mlx-lm). Env manda.
        draft_dir = os.environ.get("OPENTRADUCTOR_MT_DRAFT_DIR", draft_dir) or None
        self.draft = load(str(draft_dir))[0] if draft_dir else None
        self.num_draft_tokens = int(os.environ.get("OPENTRADUCTOR_MT_DRAFT_TOKENS", num_draft_tokens))
        self.normalize = normalize
        v = os.environ.get("OPENTRADUCTOR_MT_SENSE")  # 1/0 fuerza el prompt por sentido (A/B y gate sin tocar configs)
        sense_prompt = (v == "1") if v in ("0", "1") else sense_prompt
        self.system = QWEN_SYSTEM + (QWEN_SENSE if sense_prompt else "")
        v = os.environ.get("OPENTRADUCTOR_MT_GLOSSARY")  # 1/0 fuerza el glosario acotado (gate sin tocar configs)
        self.glossary = (v == "1") if v in ("0", "1") else sense_glossary
        self._hybrid = "enable_thinking" in (getattr(self.tok, "chat_template", None) or "")
        self._kv = None  # (prompt_cache, tokens) del último prefill(); lo consume el próximo translate()
        # T-58 contexto largo: caché KV de sesión (system + historia) que sobrevive entre frases. 1/0 lo fuerza.
        v = os.environ.get("OPENTRADUCTOR_MT_SESSION_KV")
        self.session_kv = (v == "1") if v in ("0", "1") else session_kv
        v = os.environ.get("OPENTRADUCTOR_MT_CTX_ES")  # T-74: historia solo en español (A/B, apagado por defecto)
        self.ctx_es = v == "1"
        self._sess = None  # (prompt_cache, tokens): prompt exacto de la última frase, sin lo generado
        self._post = None
        self.kv_stats = {"used": 0, "tail_tokens": []}

    def prompt(self, text_es: str, context: tuple = ()) -> str:
        """``context``: pares (es, en) de las frases anteriores de la sesión, como turnos previos del chat.
        Van antes del turno actual, así que entran en el prefijo que carga ``prefill()``."""
        msgs = [{"role": "system", "content": getattr(self, "system", QWEN_SYSTEM)}]
        kw = {"enable_thinking": False} if self._hybrid else {}
        if context and getattr(self, "ctx_es", False):
            # T-74 palanca 2: historia solo en español, en el mismo turno y antes del texto. Sin nuestro inglés
            # previo como ejemplo. Cada frase nueva agrega una línea al final del bloque: sigue siendo prefijo.
            hist = "\n".join(es for es, _ in context)
            msgs.append({"role": "user", "content": f"{QWEN_CTX_ES}\n{hist}\n\n{self._user(text_es)}"})
            return self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **kw)
        for es, en in context:
            msgs += [{"role": "user", "content": self._user(es)}, {"role": "assistant", "content": en}]
        msgs.append({"role": "user", "content": self._user(text_es)})
        return self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **kw)

    def _user(self, text_es: str) -> str:
        """Turno de usuario. Con ``glossary`` y algún término de QWEN_GLOSSARY, el glosario y QWEN_GUARD van ANTES
        del texto. prefill() arma el glosario con el parcial estable, así queda dentro de la caché KV y translate()
        solo paga la cola (con el glosario después del texto, bench 70bb991 midió p95 +63 ms en las 28 frases con
        modismo). Si la última palabra trae un término nuevo, el prefijo común se corta en el glosario y se
        recalcula desde ahí: misma salida, más lento solo en ese caso. Los turnos de historia usan el mismo
        formato, así la caché de sesión de la frase anterior sigue siendo prefijo de la siguiente."""
        g = qwen_glossary(text_es) if getattr(self, "glossary", False) else []
        if not g:
            return f"Spanish:\n{text_es}"
        return f"Meaning of the Argentine slang below: {'; '.join(g)}.\n{QWEN_GUARD}\n\nSpanish:\n{text_es}"

    def translate(self, text_es: str, context: tuple = ()) -> str:
        from mlx_lm import generate

        args = self._gen_args(text_es, context)
        if args is None:
            return ""
        prompt, kw = args
        out = generate(self.model, self.tok, prompt=prompt, verbose=False, **kw)
        self._after_gen()
        return out.strip().split("\n")[0].strip()

    def translate_stream(self, text_es: str, context: tuple = (), on_text=None) -> str:
        """T-58 (mt_stream_first_chunk): igual que translate() (mismo prompt, caché KV y greedy), pero llama
        ``on_text(prefijo)`` tras cada token con la 1.ª línea decodificada hasta ahora, y corta al primer salto
        de línea (translate() se queda con esa línea, así que el resultado es el mismo)."""
        from mlx_lm import stream_generate

        args = self._gen_args(text_es, context)
        if args is None:
            return ""
        prompt, kw = args
        acc = ""
        for r in stream_generate(self.model, self.tok, prompt, **kw):
            acc += r.text
            if "\n" in acc.lstrip():
                break
            if on_text is not None:
                on_text(acc.lstrip())
        self._after_gen()
        return acc.strip().split("\n")[0].strip()

    def _gen_args(self, text_es: str, context: tuple = ()):
        text_es = text_es.strip()
        if not text_es:
            return None
        if self.normalize:
            text_es = mt_normalize_numbers(text_es)
        prompt, kw = self.prompt(text_es, context), {"max_tokens": 3 * len(text_es.split()) + 16}
        if getattr(self, "draft", None) is not None:
            kw.update(draft_model=self.draft, num_draft_tokens=self.num_draft_tokens)
        kv, self._kv = self._kv, None  # un solo uso, sea de esta frase o no
        session = getattr(self, "session_kv", False)
        if kv is None and session:
            kv, self._sess = getattr(self, "_sess", None), None
        self._post, full, used = None, None, None
        if kv is not None:
            from mlx_lm.models.cache import trim_prompt_cache

            cache, cached = kv
            full = self._enc(prompt)
            n = 0  # prefijo común de tokens (la cola del parcial puede haber cambiado)
            while n < min(len(cached), len(full) - 1) and cached[n] == full[n]:
                n += 1
            if n > 0:
                if len(cached) > n:
                    trim_prompt_cache(cache, len(cached) - n)
                prompt, kw = full[n:], {**kw, "prompt_cache": cache}
                used = cache
                self.kv_stats["used"] += 1
                self.kv_stats["tail_tokens"].append(len(full) - n)
                del self.kv_stats["tail_tokens"][:-64]
        if session:  # T-58 contexto largo: la caché sobrevive a la frase (prompt + historia), ver _after_gen()
            from mlx_lm.models.cache import can_trim_prompt_cache, make_prompt_cache

            if used is None:
                cache = self._new_cache()
                if can_trim_prompt_cache(cache):
                    full = self._enc(prompt) if isinstance(prompt, str) else full
                    prompt, kw, used = full, {**kw, "prompt_cache": cache}, cache
            if used is not None:
                self._post = (used, full)
        return prompt, kw

    def _after_gen(self) -> None:
        """Modo sesión: saca de la caché los tokens generados y la deja con el prompt exacto de esta frase. La
        frase siguiente (historia + esta frase + nueva) comparte ese prefijo y solo paga sus tokens nuevos."""
        post, self._post = getattr(self, "_post", None), None
        if post is None:
            return
        from mlx_lm.models.cache import trim_prompt_cache

        cache, full = post
        # Por capa (modelo y borrador pueden quedar con offsets distintos tras el especulativo).
        for c in cache:
            off = getattr(c, "offset", None)
            if not isinstance(off, int) or off < len(full):
                self._sess = None
                return
            if off > len(full):
                trim_prompt_cache([c], off - len(full))
        self._sess = (cache, full)
        self.kv_stats["session_tokens"] = len(full)

    def _enc(self, s: str) -> list[int]:
        return list(self.tok.encode(s, add_special_tokens=False))

    def _new_cache(self) -> list:
        from mlx_lm.models.cache import make_prompt_cache

        d = getattr(self, "draft", None)
        return make_prompt_cache(self.model) + (make_prompt_cache(d) if d is not None else [])

    def turn_tokens(self, es: str, en: str) -> int:
        """T-68: tokens aprox. que suma un par (es, en) a la historia (turno user + assistant con su plantilla)."""
        if getattr(self, "ctx_es", False):
            return len(self._enc(es)) + 1
        return len(self._enc(self._user(es))) + len(self._enc(en)) + 10

    def prefill(self, partial_es: str, context: tuple = ()) -> None:
        """T-58: carga en una caché KV el prompt con el parcial SIN su última palabra (suele cambiar); el
        translate() siguiente solo procesa la cola y el decode. También despierta la GPU, como prewarm().
        Salida idéntica a translate() sin caché (greedy; sonda c7aec17, 9/9 en 3 frases)."""
        import mlx.core as mx
        from mlx_lm.models.cache import can_trim_prompt_cache, make_prompt_cache

        t = partial_es.strip()
        if self.normalize:
            t = mt_normalize_numbers(t)
        stable = " ".join(t.split()[:-1])
        if not stable:
            self._kv = None
            self.prewarm()
            return
        p = self.prompt(stable, context)
        head = "Spanish:\n" + stable  # anclar al texto del turno actual (el glosario, si hay, va antes y entra)
        toks = self._enc(p[: p.rindex(head) + len(head)])
        sess, n = (getattr(self, "_sess", None) if getattr(self, "session_kv", False) else None), 0
        if sess is not None:  # contexto largo: extender la caché de sesión en vez de rehacer system + historia
            self._sess = None
            from mlx_lm.models.cache import trim_prompt_cache

            cache, cached = sess
            while n < min(len(cached), len(toks) - 1) and cached[n] == toks[n]:
                n += 1
            if n == 0:
                cache = self._new_cache()
            elif len(cached) > n:
                trim_prompt_cache(cache, len(cached) - n)
        else:
            cache = self._new_cache()
        if not can_trim_prompt_cache(cache):
            self._kv = None
            self.prewarm()
            return
        x = mx.array(toks[n:])[None]
        if getattr(self, "draft", None) is None:
            self.model(x, cache=cache)
        else:
            k = len(self.model.layers)
            self.model(x, cache=cache[:k])
            self.draft(x, cache=cache[k:])
        mx.eval([c.state for c in cache])
        self._kv = (cache, toks)

    def prewarm(self) -> None:
        """Forward de 1 token con un prompt mínimo: despierta la GPU tras inactividad (T-58, GPU fría)."""
        from mlx_lm import generate

        generate(self.model, self.tok, prompt=self.prompt("hola"), max_tokens=1, verbose=False)


def _risky_token(w: str) -> bool:
    """Números e identificadores que no deben cortarse ni separarse de su vecino."""
    return (
        any(c.isdigit() for c in w)
        or any(c in w for c in "_/@#\\:")
        or ("." in w.strip(".,;:!?") or "-" in w.strip("-"))
        or (w.isupper() and len(w) > 1)
        or (any(c.isupper() for c in w[1:]) and any(c.islower() for c in w))  # camelCase
    )


def split_first_by_words(clause: str, max_words: int, bind_conjunctions: bool = False) -> list[str]:
    """Si ``clause`` tiene más de ``max_words`` palabras, la parte en (cabeza, resto).

    ``max_words`` es un OBJETIVO flexible, no un tope. Orden de prioridad:
    1. Conservar todas las palabras en orden (separadas por un espacio).
    2. Nunca cortar junto a un token con dígitos o aspecto de identificador
       (``v2.1``, ``user_id``, ``3:30``, ``API``, ``getUser``): una racha larga de
       números/identificadores puede hacer que la cabeza supere ``max_words``.
    3. Preferir cabezas de <= ``max_words`` palabras que no terminen en palabra
       funcional (``the``, ``of``, ``and``...), buscando hacia atrás desde el objetivo.
    4. Si no hay tal corte, se acepta terminar en palabra funcional (2.º pase).
    5. Si no hay corte antes del objetivo, se busca el primero válido DESPUÉS
       (cabeza > ``max_words``). Sin ningún corte válido devuelve la cláusula entera.
    Con 1 palabra o con <= ``max_words`` devuelve la cláusula sin cambios.

    ``bind_conjunctions`` (opcional, apagado por defecto: las corridas T1/T2 usan la
    heurística sin esta regla): además prohíbe cortar justo antes o justo después de
    and/or/but/nor/yet, en ambos pases y en la búsqueda hacia delante, de modo que la
    conjunción quede dentro de un grupo (p. ej. se corta antes de ``the solemnity and
    respect`` y no entre ``solemnity`` y ``and``). Solo se garantiza la propiedad
    textual (sin pérdida, sin cortes junto a conjunciones); NO se afirma que resuelva
    el artefacto acústico hasta medirlo.
    """
    words = clause.split()
    if max_words < 1 or len(words) <= max_words:
        return [clause]

    def ok(i: int, strict: bool) -> bool:  # ¿se puede cortar después de words[i]?
        a, b = words[i], words[i + 1]
        if _risky_token(a) or _risky_token(b):
            return False
        if bind_conjunctions and (
            a.lower().strip(".,;:!?\"'") in _CONJ or b.lower().strip(".,;:!?\"'") in _CONJ
        ):
            return False
        return not (strict and a.lower().strip(".,;:!?\"'") in _NO_END)

    for strict in (True, False):
        for i in range(max_words - 1, 0, -1):  # preferir cabezas de ~max_words hacia atrás
            if ok(i, strict):
                return [" ".join(words[: i + 1]), " ".join(words[i + 1 :])]
    for i in range(max_words, len(words) - 1):  # nada antes: primer corte válido después
        if ok(i, False):
            return [" ".join(words[: i + 1]), " ".join(words[i + 1 :])]
    return [clause]


# T-67 fin de frase adaptativo: palabras con las que una frase hablada en español casi nunca termina (conectores,
# preposiciones, artículos, muletillas). Sin las ambiguas que el ASR escribe sin tilde y sí cierran ("si"/"sí",
# "se"/"sé", "no", "más", "bueno", "tipo").
ES_CONTINUE_WORDS = frozenset({
    "y", "e", "o", "u", "ni", "que", "pero", "porque", "pues", "entonces", "sino", "aunque", "cuando", "donde",
    "mientras", "como", "de", "del", "a", "al", "en", "con", "por", "para", "sin", "sobre", "entre", "hacia", "hasta",
    "desde", "según", "segun", "el", "la", "los", "las", "un", "una", "unos", "unas", "lo", "mi", "mis", "tus", "su",
    "sus", "nuestro", "nuestra", "nuestros", "nuestras", "este", "esta", "ese", "esa", "eh", "em", "mm", "ehh",
    "osea", "digamos", "me", "te", "le", "les", "nos", "muy",
})
ES_CONTINUE_PAIRS = frozenset({("o", "sea"), ("es", "decir"), ("o", "bien")})
_WORD_STRIP = ".,;:!?¿¡…\"'()«»-—"


def phrase_endpoint_kind(partial: str) -> str:
    """T-67: ``"complete"`` si el parcial del ASR puede cerrar una frase, ``"continue"`` si termina en conector,
    preposición, artículo, muletilla o coma (el hablante sigue), ``"unknown"`` si está vacío.
    Puntuación final ``. ? ! …`` = complete aunque la palabra sea de la lista."""
    t = (partial or "").strip()
    if not t:
        return "unknown"
    if t[-1] in ".!?…":
        return "complete"
    if t[-1] in ",;:":
        return "continue"
    words = [w.strip(_WORD_STRIP).lower() for w in t.split()]
    words = [w for w in words if w]
    if not words:
        return "unknown"
    if words[-1] in ES_CONTINUE_WORDS:
        return "continue"
    if len(words) >= 2 and (words[-2], words[-1]) in ES_CONTINUE_PAIRS:
        return "continue"
    return "complete"


def normalize_asr_text(text: str) -> str:
    """Mayúscula inicial y punto final si faltan. No inventa comas ni toca el resto."""
    t = text.strip()
    if not t:
        return t
    if t[0].isalpha() and t[0].islower():
        t = t[0].upper() + t[1:]
    if t[-1] not in ".!?…":
        t += "."
    return t


def first_clause_final(prefix: str, min_chars: int = 24, first_max_words: int | None = None,
                       bind_conjunctions: bool = False) -> str | None:
    """T-58 (mt_stream_first_chunk): primera cláusula de ``split_clauses(texto)`` si ya queda fija con solo
    ``prefix`` (lo decodificado hasta ahora), para cualquier continuación. None si todavía puede cambiar.

    Garantía: si devuelve ``h``, entonces ``split_clauses(prefix + resto, ...)[0] == h`` para todo ``resto``.
    - Caso A: las partes cerradas (seguidas de puntuación + espacio ya visto) suman >= min_chars -> out[0] fijo,
      siempre que lo que viene después ya mida >= min_chars (si no, la cola corta final se le pega a out[0]).
    - Caso B (con first_max_words): todavía no hay out[0] cerrado, pero out[0] tiene ya >= first_max_words + 1
      palabras completas y la búsqueda hacia atrás de split_first_by_words encuentra corte con ellas.
    Conservadora: ante la duda devuelve None (se cae al camino normal)."""
    t = prefix.lstrip()
    if not t:
        return None
    parts = _CLAUSE.split(t)
    closed = [p.strip() for p in parts[:-1] if p.strip()]
    tail = parts[-1].strip()
    buf = ""
    for i, p in enumerate(closed):
        buf = f"{buf} {p}".strip()
        if len(buf) >= min_chars:
            rest = " ".join([q for q in closed[i + 1:]] + ([tail] if tail else []))
            if len(rest) < min_chars:
                return None
            return split_first_by_words(buf, first_max_words, bind_conjunctions)[0] if first_max_words else buf
    if not first_max_words:
        return None
    words = " ".join(closed + [parts[-1]]).split()
    if parts[-1] and not parts[-1][-1].isspace():
        words = words[:-1]  # la última palabra puede seguir creciendo
    if len(words) < first_max_words + 1:
        return None
    r = split_first_by_words(" ".join(words[: first_max_words + 1]), first_max_words, bind_conjunctions)
    return r[0] if len(r) == 2 else None


def split_clauses(text: str, min_chars: int = 24, first_max_words: int | None = None,
                  bind_conjunctions: bool = False) -> list[str]:
    """Parte el texto en cláusulas para sintetizar la primera cuanto antes.
    Une fragmentos muy cortos con el siguiente. Con ``first_max_words`` y si la
    primera cláusula es larga (sin puntuación útil), la parte por palabras."""
    parts = [p.strip() for p in _CLAUSE.split(text.strip()) if p.strip()]
    out: list[str] = []
    buf = ""
    for p in parts:
        buf = f"{buf} {p}".strip()
        if len(buf) >= min_chars:
            out.append(buf)
            buf = ""
    if buf:
        if out and len(buf) < min_chars:
            out[-1] = f"{out[-1]} {buf}"
        else:
            out.append(buf)
    if first_max_words and out:
        out[0:1] = split_first_by_words(out[0], first_max_words, bind_conjunctions)
    return out


def lead_silence_samples(x: np.ndarray, sr: int, thr_dbfs: float = -60.0, frame_ms: float = 5.0) -> int:
    """Muestras de silencio inicial: antes del primer bloque de ``frame_ms``
    cuyo RMS supere ``thr_dbfs``. Devuelve ``len(x)`` si todo es silencio."""
    n = max(1, int(sr * frame_ms / 1000))
    thr = 10 ** (thr_dbfs / 20)
    for i in range(0, len(x), n):
        blk = x[i : i + n]
        if float(np.sqrt(np.mean(blk * blk))) > thr:
            return i
    return len(x)


TAIL_THR_DBFS = -60.0
TAIL_FRAME_MS = 5.0
TAIL_MARGIN_MS = 80.0


def tail_keep_samples(x: np.ndarray, sr: int, thr_dbfs: float = TAIL_THR_DBFS, frame_ms: float = TAIL_FRAME_MS,
                      margin_ms: float = TAIL_MARGIN_MS) -> int:
    """Muestras a conservar al recortar el silencio FINAL (T-47; regla de ``incremental-tts-tail-v1/trim.py``).

    ``x``: muestras s16 (cualquier dtype numérico). Bloques de ``frame_ms`` alineados DESDE EL FINAL; un bloque es
    sonoro si su RMS (x/32768) en dBFS es > ``thr_dbfs``. Se conserva hasta el fin del último bloque sonoro más
    ``margin_ms``, sin pasar del largo. Todo silencio o vacío: se conserva todo (nunca vacía el chunk)."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    L = x.size
    if L == 0:
        return 0
    n = max(1, int(sr * frame_ms / 1000))
    thr = 32768.0 * 10 ** (thr_dbfs / 20)
    k, r = divmod(L, n)
    end = 0
    if k:
        rms = np.sqrt(np.mean(x[r:].reshape(k, n) ** 2, axis=1))
        idx = np.nonzero(rms > thr)[0]
        if idx.size:
            end = r + (int(idx[-1]) + 1) * n
    if not end and r and float(np.sqrt(np.mean(x[:r] ** 2))) > thr:
        end = r
    if not end:
        return L
    return min(L, end + int(sr * margin_ms / 1000))


def trim_tail_pcm(pcm: bytes, sr: int) -> bytes:
    """PCM s16le con el silencio final recortado; lo conservado es un prefijo EXACTO de ``pcm``. Si no hay nada que
    recortar, o el PCM no es válido (largo impar, sr inválido), devuelve ``pcm`` sin cambios."""
    if not pcm or len(pcm) % 2 or not isinstance(sr, int) or sr <= 0:
        return pcm
    keep = tail_keep_samples(np.frombuffer(pcm, dtype="<i2"), sr)
    return pcm if keep * 2 >= len(pcm) else pcm[: 2 * keep]


@dataclass
class TTSChunk:
    pcm: bytes  # s16le mono
    sample_rate: int
    text: str


class KokoroTTS:
    SAMPLE_RATE = 24000

    def __init__(self, voice: str, speed: float, trim_lead_margin_ms: float | None = None,
                 first_chunk_max_words: int | None = None, bind_conjunctions: bool = False,
                 trim_tail: bool = False, trim_lead_all_clauses: bool = False) -> None:
        from mlx_audio.tts.models.kokoro import KokoroPipeline
        from mlx_audio.tts.utils import load_model

        path = C.snapshot(
            C.KOKORO_REPO, C.KOKORO_REV,
            ("config.json", "kokoro-v1_0.safetensors", f"voices/{voice}.safetensors"),
        )
        self.model = load_model(path)
        self.pipe = KokoroPipeline(
            lang_code="a", model=self.model, repo_id=C.KOKORO_REPO, model_path=path
        )
        self.voice = voice
        self.speed = speed
        # None = sin recorte. Si no, se recorta el silencio inicial del PRIMER
        # chunk de cada frase dejando este margen antes del primer sonido
        # (umbral -60 dBFS, por debajo del -50 con que el benchmark detecta audio).
        self.trim_lead_margin_ms = trim_lead_margin_ms
        self.first_chunk_max_words = first_chunk_max_words
        self.bind_conjunctions = bind_conjunctions
        self.last_trim_ms = 0.0
        # T-47: recorte del silencio FINAL de cada chunk (-60 dBFS, bloques de 5 ms, margen 80 ms). False = sin cambio.
        self.trim_tail = trim_tail
        self.tail_trimmed_ms_total = 0.0
        # T-58: recorte inicial también en el 1.er chunk de cada cláusula (no solo de la frase). False = sin cambio.
        self.trim_lead_all_clauses = trim_lead_all_clauses

    def plan(self, text_en: str) -> list[str]:
        """Cláusulas que se sintetizarán (determinista para un mismo texto y config)."""
        return split_clauses(text_en, first_max_words=self.first_chunk_max_words,
                             bind_conjunctions=self.bind_conjunctions)

    def first_clause_final(self, prefix_en: str) -> str | None:
        """Lo mismo que ``plan(texto)[0]`` si ``prefix_en`` ya lo fija (MT en streaming); si no, None."""
        return first_clause_final(prefix_en, first_max_words=self.first_chunk_max_words,
                                  bind_conjunctions=self.bind_conjunctions)

    def synth(self, text_en: str):
        """Genera TTSChunk por cláusula (generador: el primer chunk sale en
        cuanto la primera cláusula está sintetizada)."""
        yield from self.synth_clauses(self.plan(text_en), first=True)

    def synth_clauses(self, clauses: list[str], first: bool = True):
        """Sintetiza ``clauses`` en orden. ``first=True``: el primer chunk de la
        primera cláusula recibe el recorte de silencio inicial (si está activo).
        Sintetizar ``plan(t)[:1]`` con first=True y luego ``plan(t)[1:]`` con
        first=False da exactamente los mismos chunks que ``synth(t)``."""
        import mlx.core as mx

        if first:
            self.last_trim_ms = 0.0
        for clause in clauses:
            head = True  # 1.er chunk con audio de esta cláusula
            for r in self.pipe(
                clause, voice=self.voice, speed=self.speed, split_pattern=None
            ):
                a = r.audio
                if a is None:
                    continue
                mx.eval(a)
                x = np.asarray(a, dtype=np.float32).reshape(-1)
                trim = first or (head and getattr(self, "trim_lead_all_clauses", False))
                if x.size and trim and self.trim_lead_margin_ms is not None:
                    keep = int(self.SAMPLE_RATE * self.trim_lead_margin_ms / 1000)
                    cut = max(0, lead_silence_samples(x, self.SAMPLE_RATE) - keep)
                    if cut < x.size:  # nunca vaciar el chunk
                        x = x[cut:]
                        if first:
                            self.last_trim_ms = cut / self.SAMPLE_RATE * 1000
                if x.size:
                    first = False
                    head = False
                    yield self._chunk(float_to_pcm(x), clause)

    def _chunk(self, pcm: bytes, clause: str) -> "TTSChunk":
        # T-47: el tail trim NO va acá (cada chunk es una cláusula: recortarlo acortaba las pausas entre frases).
        # Se aplica al final de la frase con TailHold, en el pipeline.
        return TTSChunk(pcm, self.SAMPLE_RATE, clause)


class TailHold:
    """T-47: recorte del silencio final de la FRASE, sin tocar las pausas entre cláusulas ni atrasar el audio.

    ``feed(pcm)`` devuelve el chunk sin su silencio final (sale ya) y guarda ese silencio. Si llega otro chunk de la
    misma frase, el silencio guardado va delante de él: la pausa interna queda igual, menos el tiempo que el player
    ya estuvo sin audio esperando a la síntesis (así no se suma a la pausa de hoy). ``finish()`` descarta el
    silencio pendiente (la cola de la frase) y devuelve cuántos ms recortó."""

    def __init__(self, sr: int, clock=None) -> None:
        import time

        self.sr = sr
        self.clock = clock or time.perf_counter
        self.pending = b""
        self.play_end: float | None = None  # estimación de cuándo termina de sonar lo ya emitido

    def feed(self, pcm: bytes) -> bytes:
        now = self.clock()
        if self.pending:
            rest = self.pending
            idle = now - self.play_end if self.play_end is not None else 0.0
            if idle > 0:
                rest = rest[min(len(rest) // 2, round(idle * self.sr)) * 2:]
            pcm = rest + pcm
            self.pending = b""
        keep = trim_tail_pcm(pcm, self.sr)
        self.pending = pcm[len(keep):]
        start = now if self.play_end is None else max(now, self.play_end)
        self.play_end = start + len(keep) / 2 / self.sr
        return keep

    def finish(self) -> float:
        ms = len(self.pending) / 2 / self.sr * 1000
        self.pending = b""
        return ms
