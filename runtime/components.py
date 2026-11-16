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
class SileroVAD:
    """Probabilidad de voz por ventana de 32 ms, con estado. Una instancia por
    stream (``reset`` entre streams)."""

    def __init__(self) -> None:
        import torch
        from silero_vad import load_silero_vad

        torch.set_num_threads(1)
        self._torch = torch
        self.model = load_silero_vad()
        self._rest = np.zeros(0, dtype=np.float32)

    def reset(self) -> None:
        self.model.reset_states()
        self._rest = np.zeros(0, dtype=np.float32)

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
        with self._torch.inference_mode():
            for i in range(n):
                w = self._torch.from_numpy(x[i * VAD_WINDOW : (i + 1) * VAD_WINDOW])
                out.append(((i + 1) * VAD_WINDOW - prev, float(self.model(w, 16000).item())))
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
class MarianCT2:
    def __init__(self, beam_size: int = 1, threads: int = 4) -> None:
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

    def translate(self, text_es: str) -> str:
        text_es = text_es.strip()
        if not text_es:
            return ""
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


@dataclass
class TTSChunk:
    pcm: bytes  # s16le mono
    sample_rate: int
    text: str


class KokoroTTS:
    SAMPLE_RATE = 24000

    def __init__(self, voice: str, speed: float, trim_lead_margin_ms: float | None = None,
                 first_chunk_max_words: int | None = None, bind_conjunctions: bool = False) -> None:
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
            for r in self.pipe(
                clause, voice=self.voice, speed=self.speed, split_pattern=None
            ):
                a = r.audio
                if a is None:
                    continue
                mx.eval(a)
                x = np.asarray(a, dtype=np.float32).reshape(-1)
                if x.size and first and self.trim_lead_margin_ms is not None:
                    keep = int(self.SAMPLE_RATE * self.trim_lead_margin_ms / 1000)
                    cut = max(0, lead_silence_samples(x, self.SAMPLE_RATE) - keep)
                    if cut < x.size:  # nunca vaciar el chunk
                        x = x[cut:]
                        self.last_trim_ms = cut / self.SAMPLE_RATE * 1000
                if x.size:
                    first = False
                    yield TTSChunk(float_to_pcm(x), self.SAMPLE_RATE, clause)
