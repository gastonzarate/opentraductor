"""Componentes de la baseline: VAD (Silero), ASR (Parakeet MLX), MT (Marian
CTranslate2) y TTS (Kokoro MLX). Sin red: todo desde disco.

Nota MLX: los arrays y streams de MLX son por hilo. Cargar y usar ASR/TTS en
un MISMO hilo dedicado (ver ``pipeline.ModelWorker``).
"""
from __future__ import annotations

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


class QwenMLX:
    """MT ES->EN con Qwen3 4bit en MLX (mlx-lm). Greedy, corte en el primer salto de línea, max_tokens = 3n+16.

    Usa la GPU: el pipeline lo llama desde el mismo hilo que Kokoro (MLX no se comparte entre hilos).
    ``normalize`` aplica el mismo preproceso de BUG-7 que Marian (cifras y horas, sin partir oraciones:
    Qwen no omite la 2.ª oración y partirla costaría una llamada más)."""

    def __init__(self, model_dir, normalize: bool = False) -> None:
        from mlx_lm import load

        self.model, self.tok = load(str(model_dir))
        self.normalize = normalize
        self._hybrid = "enable_thinking" in (getattr(self.tok, "chat_template", None) or "")
        self._kv = None  # (prompt_cache, tokens) del último prefill(); lo consume el próximo translate()
        self.kv_stats = {"used": 0, "tail_tokens": []}

    def prompt(self, text_es: str) -> str:
        msgs = [{"role": "system", "content": QWEN_SYSTEM}, {"role": "user", "content": f"Spanish:\n{text_es}"}]
        kw = {"enable_thinking": False} if self._hybrid else {}
        return self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **kw)

    def translate(self, text_es: str) -> str:
        from mlx_lm import generate

        text_es = text_es.strip()
        if not text_es:
            return ""
        if self.normalize:
            text_es = mt_normalize_numbers(text_es)
        prompt, kw = self.prompt(text_es), {}
        kv, self._kv = self._kv, None  # un solo uso, sea de esta frase o no
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
                prompt, kw = full[n:], {"prompt_cache": cache}
                self.kv_stats["used"] += 1
                self.kv_stats["tail_tokens"].append(len(full) - n)
                del self.kv_stats["tail_tokens"][:-64]
        out = generate(self.model, self.tok, prompt=prompt, max_tokens=3 * len(text_es.split()) + 16,
                       verbose=False, **kw)
        return out.strip().split("\n")[0].strip()

    def _enc(self, s: str) -> list[int]:
        return list(self.tok.encode(s, add_special_tokens=False))

    def prefill(self, partial_es: str) -> None:
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
        p = self.prompt(stable)
        toks = self._enc(p[: p.rindex(stable) + len(stable)])
        cache = make_prompt_cache(self.model)
        if not can_trim_prompt_cache(cache):
            self._kv = None
            self.prewarm()
            return
        self.model(mx.array(toks)[None], cache=cache)
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
        if self.trim_tail:
            out = trim_tail_pcm(pcm, self.SAMPLE_RATE)
            self.tail_trimmed_ms_total += (len(pcm) - len(out)) / 2 / self.SAMPLE_RATE * 1000
            pcm = out
        return TTSChunk(pcm, self.SAMPLE_RATE, clause)
