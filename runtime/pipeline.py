"""Motor de traducción en streaming (independiente del arnés y de la UI).

Flujo por stream:

    push(pcm16k) -> VAD (Silero, 32 ms) -> segmentador con endpoint ajustable
                 -> ASR streaming (Parakeet, hilo GPU dedicado) -> parciales
    endpoint     -> ASR final -> MT (Marian CT2, hilo CPU) -> TTS por cláusulas
                 -> chunks PCM (24 kHz) por el callback ``emit``

Reglas:
* Nunca se usa audio futuro: cada decisión depende solo de lo ya recibido.
* ``push`` no bloquea: el cómputo pesado va a hilos. Si el ASR se atrasa, el
  audio se encola (acotado) y el atraso queda registrado, no oculto.
* No se descarta traducción pendiente cuando el hablante vuelve a hablar:
  los segmentos se traducen en orden. Si la cola de segmentos excede
  ``max_pending_segments`` se descarta el más viejo SIN traducir y se emite un
  evento ``overflow`` visible.
* No se guarda audio. Solo se emiten textos, métricas y PCM de salida.
"""
from __future__ import annotations

import asyncio
import collections
import concurrent.futures as cf
import os
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from . import components as K
from .config import PipelineConfig

SR = 16000
Emit = Callable[[str, dict], Awaitable[None]]


def _apply_mlx_cache_limit(cfg: PipelineConfig) -> int | None:
    """T-50: ``mx.set_cache_limit`` (global del proceso). Sin MLX instalado no hace nada."""
    from .config import mlx_cache_limit_bytes

    b = mlx_cache_limit_bytes(cfg)
    if b is None:
        return None
    try:
        import mlx.core as mx
    except ImportError:
        return None
    mx.set_cache_limit(b)
    return b


def _timed_call(fn, *a):
    """Ejecuta ``fn`` y devuelve (resultado, ms medidos dentro del hilo del executor)."""
    t = time.perf_counter()
    r = fn(*a)
    return r, (time.perf_counter() - t) * 1000


def _append_capped(xs: list, v, cap: int = 64) -> None:
    xs.append(v)
    del xs[:-cap]


class ModelWorker:
    """Un hilo dedicado: MLX liga arrays/streams al hilo que los crea."""

    def __init__(self, name: str) -> None:
        self._ex = cf.ThreadPoolExecutor(1, thread_name_prefix=name)
        self.inflight = 0  # trabajos encolados o corriendo (solo se toca desde el loop)
        self.last_done = 0.0  # time.monotonic() del último trabajo terminado

    async def run(self, fn, *a):
        self.inflight += 1
        try:
            return await asyncio.get_running_loop().run_in_executor(self._ex, fn, *a)
        finally:
            self.inflight -= 1
            self.last_done = time.monotonic()

    def shutdown(self) -> None:
        self._ex.shutdown(wait=True, cancel_futures=True)


@dataclass
class Segment:
    seg_id: int
    start_s: float  # inicio de voz según VAD (posición de audio)
    end_s: float | None = None  # último instante de voz
    endpoint_pos_s: float | None = None
    forced: bool = False
    text_es: str = ""  # texto usado por MT (normalizado si asr_normalize_text)
    text_es_raw: str = ""  # salida cruda del ASR
    gen: int = 0  # generación del stream que creó el segmento


@dataclass
class SpecEntry:
    """Resultado especulativo (en memoria, nunca emitido) de MT + primer fragmento TTS de un parcial."""
    stream_id: str
    seg_id: int
    text_es: str  # texto fuente EXACTO sobre el que se calculó
    gen: int = 0
    fp: tuple = ()  # configuración que afecta MT/TTS al calcular; si cambia, no se reutiliza
    started: float = field(default_factory=time.perf_counter)
    task: asyncio.Task | None = None
    text_en: str | None = None
    clauses: list[str] | None = None
    first_chunks: list | None = None  # TTSChunk de la primera cláusula, si se pudo
    mt_ms: float = 0.0
    tts_ms: float = 0.0
    cancelled: bool = False
    error: str | None = None

    @property
    def compute_ms(self) -> float:
        return self.mt_ms + self.tts_ms


@dataclass
class StreamState:
    stream_id: str
    vad: K.SileroVAD
    pos_samples: int = 0  # muestras recibidas
    t_base_s: float = 0.0
    in_speech: bool = False
    seg: Segment | None = None
    last_speech_end_s: float = 0.0
    silence_ms: float = 0.0
    seg_counter: int = 0
    preroll: collections.deque = field(default_factory=lambda: collections.deque(maxlen=12))
    asr_pending_s: float = 0.0  # audio encolado para el ASR (acotado)
    dropped_s: float = 0.0
    asr_buf: bytearray = field(default_factory=bytearray)
    asr_buf_start_s: float = 0.0
    asr_q: asyncio.Queue | None = None
    asr_task: asyncio.Task | None = None
    last_partial: str = ""
    buf_end_s: float = 0.0  # fin (posición de audio) del chunk que se está procesando en push
    skipped_tail_ms: float = 0.0  # cola silenciosa no enviada al ASR (medido)
    asr_done_pos_s: float = 0.0  # fin de audio del último chunk ya procesado por el ASR
    spec_seg_id: int | None = None  # segmento para el que ya se especuló
    prewarm_seg_id: int | None = None  # segmento para el que ya se evaluó el prewarm de MT
    kv_pause: tuple | None = None  # (seg_id, last_speech_end_s) de la última pausa evaluada para prefill KV
    kv_stable: tuple | None = None  # (seg_id, prefijo estable) ya cargado en la caché KV
    asr_spec_pause: tuple | None = None  # (seg_id, last_speech_end_s) de la pausa con finish especulativo encolado
    asr_spec_open: bool = False  # hay un finish especulativo vigente (stream del ASR cerrado) en el segmento actual
    gen: int = 0  # generación (cada start_stream incrementa; separa sesiones con el mismo id)


class Engine:
    def __init__(self, cfg: PipelineConfig, emit: Emit) -> None:
        self.cfg = cfg
        self.emit = emit
        self.gpu = ModelWorker("mlx")
        self.cpu = ModelWorker("cpu")
        # Parakeet usa MLX (hilo GPU dedicado, compartido con el TTS). Moonshine
        # es CPU/ONNX: hilo propio, para que synth() no bloquee el ASR de la
        # frase siguiente ni la MT bloquee al ASR.
        self.asr_worker = self.gpu if cfg.asr_backend == "parakeet" else ModelWorker("asr")
        self.asr: K.ParakeetStreamer | K.MoonshineStreamer | None = None
        self.tts: K.KokoroTTS | None = None
        self.mt: K.MarianCT2 | None = None
        self.vad_proto: K.SileroVAD | None = None
        self._streams: dict[str, StreamState] = {}
        self._out_q: asyncio.Queue | None = None
        self._out_task: asyncio.Task | None = None
        self.load_ms: dict[str, float] = {}
        self._spec: dict[tuple[str, int, int], SpecEntry] = {}
        self._speaking = False  # trabajo confirmado de MT/TTS en curso (la especulación cede)
        self.spec_stats = {
            "started": 0, "hit": 0, "miss_text": 0, "error": 0, "cancelled": 0,
            "skipped_busy": 0, "skipped_limit": 0, "miss_config": 0, "inflight_unmeasured": 0,
            "discarded_observed_ms": 0.0, "overlap_estimate_ms": 0.0,
        }
        self.prewarm_stats: dict[str, Any] = {"started": 0, "skipped_hot": 0, "error": 0, "cost_ms": []}
        self._prewarm_tasks: set[asyncio.Future] = set()
        self._mt_ctx: dict[str, list[tuple[str, str]]] = {}  # stream_id -> últimas frases (es, en) emitidas
        # T-58 fin de frase especulativo: started = finish adelantados; hit = usados como final; discard_voice = volvió
        # la voz (stream reabierto y re-alimentado); finish_ms / refeed_ms = cómputo medido en el hilo del ASR.
        self.asr_spec_stats: dict[str, Any] = {"started": 0, "hit": 0, "discard_voice": 0, "finish_ms": [],
                                               "refeed_ms": []}
        # T-58 MT en streaming hacia Kokoro: segments = frases por este camino; head_early = 1.er fragmento sintetizado
        # antes de terminar la MT; mismatch = plan(final)[0] != fragmento adelantado (debe ser 0); lead_ms = cuánto
        # antes del fin de la MT quedó listo el 1.er audio; tts_in_mt_ms = Kokoro intercalado en el decode.
        self.mt_stream_stats: dict[str, Any] = {"segments": 0, "head_early": 0, "mismatch": 0, "head_only": 0, "lead_ms": [],
                                                "tts_in_mt_ms": []}
        self._gen = 0
        self._live_gen: dict[str, int] = {}  # generación vigente por stream; lo de otras generaciones no se emite
        self.stale_dropped = 0  # emisiones/trabajos de sesiones cerradas descartados (medido)

    def spec_metrics(self) -> dict:
        """Contadores de especulación.
        discarded_observed_ms: tiempo de cómputo OBSERVADO y COMPLETADO de resultados descartados; las operaciones en
          vuelo canceladas (inflight_unmeasured) no se miden, así que NO es el desperdicio total.
        overlap_estimate_ms: estimación de espera evitada en aciertos (cómputo especulativo menos espera en el final);
          NO es un ahorro causal de latencia."""
        d = dict(self.spec_stats)
        d["discarded_observed_ms"] = round(d["discarded_observed_ms"], 1)
        d["overlap_estimate_ms"] = round(d["overlap_estimate_ms"], 1)
        d["enabled"] = self.cfg.speculative_mt_tts
        return d

    @property
    def _mt_exec(self):
        """Ejecutor del MT: Marian en el hilo CPU; Qwen (MLX) en el hilo GPU, el mismo de Kokoro."""
        return self.gpu if self.cfg.mt_engine == "qwen" else self.cpu

    def _spec_fp(self) -> tuple:
        c = self.cfg
        return (c.voice, c.speed, c.tts_trim_lead_margin_ms, c.tts_first_chunk_max_words,
                c.tts_bind_conjunctions, c.asr_normalize_text, c.tts_trim_tail, c.mt_normalize, c.mt_engine,
                c.tts_trim_lead_all_clauses)

    def backlog(self) -> dict:
        """Estado de colas para métricas (valores medidos, no estimados)."""
        return {
            "asr_backlog_s": round(sum(st.asr_pending_s for st in self._streams.values()), 3),
            "dropped_ms": round(sum(st.dropped_s for st in self._streams.values()) * 1000),
            "queue_depth": self._out_q.qsize() if self._out_q else 0,
        }

    # ---------------------------------------------------------------- ciclo
    async def open(self) -> None:
        def timed(name, fn):
            t = time.perf_counter()
            r = fn()
            self.load_ms[name] = round((time.perf_counter() - t) * 1000, 1)
            return r

        from .preflight import require_ready

        require_ready(self.cfg)
        c = self.cfg
        await self.gpu.run(lambda: _apply_mlx_cache_limit(c))  # T-50: antes de cargar nada en MLX
        self.asr = await self.asr_worker.run(lambda: timed("asr", self._make_asr))
        self.tts = await self.gpu.run(lambda: timed("tts", lambda: K.KokoroTTS(c.voice, c.speed, c.tts_trim_lead_margin_ms,
                                                   c.tts_first_chunk_max_words, c.tts_bind_conjunctions,
                                                   c.tts_trim_tail, c.tts_trim_lead_all_clauses)))
        if c.mt_engine == "qwen":  # T-51: Qwen MLX comparte el hilo GPU con Kokoro
            from .config import qwen_model_dir

            self.mt = await self.gpu.run(lambda: timed("mt", lambda: K.QwenMLX(qwen_model_dir(c), normalize=c.mt_normalize,
                                                                                         sense_prompt=c.mt_sense_prompt,
                                                                                         sense_glossary=c.mt_sense_glossary,
                                                                                         session_kv=c.mt_session_kv)))
        else:
            self.mt = await self.cpu.run(lambda: timed("mt", lambda: K.MarianCT2(normalize=c.mt_normalize)))
        self.vad_proto = timed("vad", K.SileroVAD)
        self._out_q = asyncio.Queue(maxsize=max(1, c.max_pending_segments))
        self._out_task = asyncio.ensure_future(self._speaker_loop())
        if c.warmup_in_open:
            await self._warmup()

    def _make_asr(self):
        c = self.cfg
        if c.asr_backend == "parakeet":
            return K.ParakeetStreamer(c.asr_context)
        from .config import MOONSHINE_MODELS, moonshine_model_dir

        return K.MoonshineStreamer(
            moonshine_model_dir(c.asr_backend), MOONSHINE_MODELS[c.asr_backend][1],
            c.moonshine_update_interval_s,
        )

    async def _warmup(self) -> None:
        silence = bytes(2 * SR // 2)
        await self.asr_worker.run(lambda: (self.asr.start(), self.asr.add(silence), self.asr.finish()))
        await self._mt_exec.run(lambda: self.mt.translate("hola"))
        await self.gpu.run(lambda: list(self.tts.synth("Hello there.")))

    async def close(self) -> None:
        for st in list(self._streams.values()):
            await self._stop_asr(st)
        self._spec_drop(None)
        if self._out_task:
            self._out_task.cancel()
            try:
                await self._out_task
            except (asyncio.CancelledError, Exception):
                pass
        # Liberar recursos nativos del ASR (Transcriber de Moonshine) en su propio hilo.
        close_asr = getattr(self.asr, "close", None)
        if close_asr is not None:
            try:
                await self.asr_worker.run(close_asr)
            except Exception as exc:  # noqa: BLE001
                await self.emit("error", {"stream_id": "", "error_kind": f"asr_close: {type(exc).__name__}: {exc}"})
        self.gpu.shutdown()
        self.cpu.shutdown()
        if self.asr_worker is not self.gpu:
            self.asr_worker.shutdown()

    async def start_stream(self, stream_id: str) -> None:
        if stream_id in self._streams:
            # Reinicio con un id vivo: se cierra ordenadamente la sesión anterior (fuerza endpoint, detiene su
            # ASR, drena sus salidas confirmadas y descarta su especulación) ANTES de crear la nueva.
            await self.end_stream(stream_id)
        vad = K.SileroVAD() if self._streams else self.vad_proto
        vad.reset()
        self._spec_drop(stream_id)
        self._mt_ctx.pop(stream_id, None)  # sesión nueva: sin contexto de la anterior
        self._gen += 1
        self._live_gen[stream_id] = self._gen
        st = StreamState(stream_id=stream_id, vad=vad, gen=self._gen)
        st.asr_q = asyncio.Queue()
        st.asr_task = asyncio.ensure_future(self._asr_loop(st))
        self._streams[stream_id] = st

    async def _stop_asr(self, st: StreamState) -> None:
        if st.asr_task:
            st.asr_task.cancel()
            try:
                await st.asr_task
            except (asyncio.CancelledError, Exception):
                pass
            st.asr_task = None

    def _is_live(self, stream_id: str, gen: int) -> bool:
        return self._live_gen.get(stream_id) == gen

    def _purge_queue(self, stream_id: str, gen: int) -> int:
        """Quita de la cola de salida los segmentos de esa generación (no se tocan los de otras)."""
        q = self._out_q
        if q is None:
            return 0
        keep, dropped = [], 0
        while True:
            try:
                item = q.get_nowait()
            except asyncio.QueueEmpty:
                break
            q.task_done()
            if item[0] == stream_id and item[1].gen == gen:
                dropped += 1
            else:
                keep.append(item)
        for item in keep:
            q.put_nowait(item)
        return dropped

    async def end_stream(self, stream_id: str) -> None:
        st = self._streams.get(stream_id)
        if not st:
            return
        gen = st.gen
        try:
            if st.in_speech and st.seg is not None:
                await self._endpoint(st, st.pos_samples / SR + st.t_base_s, forced=True)
            await st.asr_q.put(("stop", None))
            if st.asr_task:
                await st.asr_task
            if self._out_q is not None:
                try:
                    await asyncio.wait_for(self._out_q.join(), self.cfg.end_stream_drain_timeout_s)
                except asyncio.TimeoutError:
                    await self.emit("error", {"stream_id": stream_id, "error_kind": "end_stream_drain_timeout"})
        finally:
            # A partir de aquí NADA de esta generación se emite (trabajo confirmado viejo tras un timeout incluido).
            if self._live_gen.get(stream_id) == gen:
                del self._live_gen[stream_id]
            self.stale_dropped += self._purge_queue(stream_id, gen)
            self._spec_drop(stream_id, gen=gen)
            self._streams.pop(stream_id, None)

    # ---------------------------------------------------------------- audio
    async def push(self, stream_id: str, pcm: bytes, t_start_s: float) -> None:
        st = self._streams[stream_id]
        c = self.cfg
        win_s = K.VAD_WINDOW / SR
        base = st.pos_samples
        chunk_end_s = (base + len(pcm) // 2) / SR + st.t_base_s
        st.buf_end_s = chunk_end_s
        appended = False  # ¿ya se añadió este chunk al buffer del ASR?
        if st.in_speech:
            st.asr_buf.extend(pcm)
            appended = True
        for consumed, p in st.vad.probs(pcm):
            end_s = (base + consumed) / SR + st.t_base_s
            if p >= c.vad_threshold:
                if st.asr_spec_open and st.in_speech:
                    st.asr_spec_open = False  # volvió la voz: el finish especulativo no sirve
                    await st.asr_q.put(("spec_resume", None))
                st.silence_ms = 0.0
                st.last_speech_end_s = end_s
                if not st.in_speech:
                    st.in_speech = True
                    st.seg_counter += 1
                    st.seg = Segment(st.seg_counter, start_s=end_s - win_s, gen=st.gen)
                    pre = b"".join(st.preroll)
                    st.preroll.clear()
                    await st.asr_q.put(("start", (st.seg, pre)))
                    if not appended:
                        st.asr_buf.extend(pcm)
                        appended = True
            elif st.in_speech:
                st.silence_ms += win_s * 1000
                if st.silence_ms >= c.endpoint_silence_ms:
                    await self._endpoint(st, end_s, forced=False)
                    appended = False
            if (
                st.in_speech
                and st.seg is not None
                and end_s - st.seg.start_s >= c.max_segment_s
            ):
                await self._endpoint(st, end_s, forced=True)
                appended = False
        st.pos_samples += len(pcm) // 2
        await self._maybe_spec_finish(st)
        self._maybe_speculate(st)
        self._maybe_prewarm(st)
        if st.in_speech:
            if len(st.asr_buf) // 2 >= SR * c.asr_chunk_ms / 1000:
                await self._queue_audio(st, chunk_end_s)
        elif not appended:
            st.preroll.append(pcm)

    async def _queue_audio(self, st: StreamState, pos_s: float) -> None:
        """Encola audio para el ASR con límite en segundos. Si el ASR va
        atrasado más que ``max_asr_backlog_s`` se descarta el audio NUEVO
        (nunca el ya encolado, para no corromper el estado del ASR), se cuenta
        y se emite ``overflow`` visible."""
        dur = len(st.asr_buf) / 2 / SR
        pcm = bytes(st.asr_buf)
        st.asr_buf.clear()
        if st.asr_pending_s + dur > self.cfg.max_asr_backlog_s:
            st.dropped_s += dur
            await self.emit(
                "overflow",
                {"stream_id": st.stream_id, "reason": "asr_backlog",
                 "dropped_ms": round(dur * 1000), "total_dropped_ms": round(st.dropped_s * 1000)},
            )
            return
        st.asr_pending_s += dur
        await st.asr_q.put(("audio", (pcm, pos_s, dur)))

    async def _endpoint(self, st: StreamState, pos_s: float, forced: bool) -> None:
        seg = st.seg
        if seg is None:
            return
        seg.end_s = st.last_speech_end_s
        seg.endpoint_pos_s = pos_s
        seg.forced = forced
        if st.asr_buf:
            tail_start_s = st.buf_end_s - len(st.asr_buf) / 2 / SR
            # Cola solo-silencio: el VAD ya vio silencio desde last_speech_end_s y
            # TODA la voz fue enviada al ASR en chunks anteriores; procesarla solo
            # añade una pasada del ASR (~120 ms) tras el fin de habla.
            if (
                self.cfg.asr_skip_silent_tail
                and not forced
                and tail_start_s >= st.last_speech_end_s
            ):
                st.skipped_tail_ms += len(st.asr_buf) / 2 / SR * 1000
                st.asr_buf.clear()
            else:
                await self._queue_audio(st, pos_s)
        await st.asr_q.put(("end", seg))
        st.asr_spec_open = False
        await self.emit(
            "endpoint",
            {"stream_id": st.stream_id, "audio_pos_s": pos_s, "forced": forced,
             "seg_id": seg.seg_id},
        )
        st.in_speech = False
        st.seg = None
        st.silence_ms = 0.0
        st.vad.reset_states()

    # ------------------------------------------------------- contexto de la MT (pedido 17/11)
    def _mt_args(self, stream_id: str) -> tuple:
        """Argumentos extra de translate()/prefill(): () sin contexto (Marian y Qwen por defecto)."""
        if self.cfg.mt_engine != "qwen" or self.cfg.mt_context_turns <= 0:
            return ()
        return (tuple(self._mt_ctx.get(stream_id, ())),)

    def _mt_remember(self, stream_id: str, es: str, en: str) -> None:
        n = self.cfg.mt_context_turns
        if self.cfg.mt_engine != "qwen" or n <= 0 or not es.strip() or not en.strip():
            return
        h = self._mt_ctx.setdefault(stream_id, [])
        h.append((es.strip(), en.strip()))
        if self.cfg.mt_session_kv:  # recorte en bloque: el prefijo (y la caché KV de sesión) cambia solo cada n/2 frases
            if len(h) > n:
                del h[: len(h) - max(1, n // 2)]
        else:
            del h[:-n]

    # ------------------------------------------------------- fin de frase especulativo (T-58)
    @property
    def _asr_spec_enabled(self) -> bool:
        v = os.environ.get("OPENTRADUCTOR_ASR_SPEC_FINISH")
        return (v == "1") if v in ("0", "1") else self.cfg.asr_spec_finish_on_pause

    async def _maybe_spec_finish(self, st: StreamState) -> None:
        """Una vez por pausa VAD: manda al ASR la voz pendiente y le pide el finish sin esperar al endpoint."""
        c = self.cfg
        seg = st.seg
        if seg is None or not st.in_speech or st.asr_spec_open or not self._asr_spec_enabled:
            return
        if st.silence_ms < c.asr_spec_finish_silence_ms or st.silence_ms >= c.endpoint_silence_ms:
            return
        pause = (seg.seg_id, st.last_speech_end_s)
        if pause == st.asr_spec_pause or not st.last_partial.strip():
            return
        if (st.last_speech_end_s - seg.start_s) * 1000 < c.min_speech_ms:
            return
        st.asr_spec_pause = pause
        st.asr_spec_open = True
        if st.asr_buf:
            await self._queue_audio(st, st.buf_end_s)
        await st.asr_q.put(("spec_end", pause))

    # ------------------------------------------------------------------ ASR
    async def _asr_loop(self, st: StreamState) -> None:
        """Serializa el ASR de un stream. Los parciales salen a medida que
        llega el audio; al final de segmento se entrega a la cola de salida."""
        cur: Segment | None = None
        seg_pcm: list[bytes] = []  # audio del segmento ya pasado al ASR (para re-alimentar si vuelve la voz)
        held: list[bytes] = []  # audio recibido con el stream cerrado por un finish especulativo
        closed: tuple | None = None  # (pausa, texto) del finish especulativo vigente
        try:
            while True:
                kind, payload = await st.asr_q.get()
                if kind == "stop":
                    return
                if kind == "start":
                    cur, pre = payload
                    seg_pcm, held, closed = [], [], None
                    await self.asr_worker.run(self.asr.start)
                    if pre:
                        await self.asr_worker.run(self.asr.add, pre)
                        seg_pcm.append(pre)
                    st.last_partial = ""
                elif kind == "spec_end":
                    if closed is None:
                        r = await self.asr_worker.run(_timed_call, self.asr.finish)
                        closed = (payload, r[0])
                        self.asr_spec_stats["started"] += 1
                        _append_capped(self.asr_spec_stats["finish_ms"], round(r[1], 1))
                elif kind == "spec_resume":
                    if closed is not None:
                        audio = b"".join(seg_pcm + held)
                        seg_pcm, held, closed = ([audio] if audio else []), [], None

                        def refeed(a=audio):
                            self.asr.start()
                            return self.asr.add(a) if a else ""

                        r = await self.asr_worker.run(_timed_call, refeed)
                        self.asr_spec_stats["discard_voice"] += 1
                        _append_capped(self.asr_spec_stats["refeed_ms"], round(r[1], 1))
                elif kind == "audio":
                    pcm, pos_s, dur = payload
                    if closed is not None:
                        held.append(pcm)
                        st.asr_pending_s = max(0.0, st.asr_pending_s - dur)
                        st.asr_done_pos_s = max(st.asr_done_pos_s, pos_s)
                        continue
                    try:
                        text = await self.asr_worker.run(self.asr.add, pcm)
                    finally:
                        st.asr_pending_s = max(0.0, st.asr_pending_s - dur)
                    seg_pcm.append(pcm)
                    st.asr_done_pos_s = max(st.asr_done_pos_s, pos_s)
                    if text and text != st.last_partial:
                        st.last_partial = text
                        await self.emit(
                            "asr_partial",
                            {"stream_id": st.stream_id, "text": text, "audio_pos_s": pos_s,
                             "seg_id": cur.seg_id if cur else None},
                        )
                    self._maybe_speculate(st)
                elif kind == "end":
                    seg = payload
                    if closed is not None and closed[0] == (seg.seg_id, seg.end_s):
                        seg.text_es_raw = closed[1]  # sin voz nueva desde la pausa: el finish adelantado es el final
                        self.asr_spec_stats["hit"] += 1
                    else:
                        if closed is not None:  # no debería pasar (la voz nueva reabre antes); camino normal
                            audio = b"".join(seg_pcm + held)
                            await self.asr_worker.run(lambda a=audio: (self.asr.start(), a and self.asr.add(a)))
                            self.asr_spec_stats["miss"] = self.asr_spec_stats.get("miss", 0) + 1
                        seg.text_es_raw = await self.asr_worker.run(self.asr.finish)
                    seg_pcm, held, closed = [], [], None
                    seg.text_es = (
                        K.normalize_asr_text(seg.text_es_raw)
                        if self.cfg.asr_normalize_text else seg.text_es_raw
                    )
                    if ((seg.end_s or seg.start_s) - seg.start_s) * 1000 < self.cfg.min_speech_ms:
                        self._spec_drop(st.stream_id, seg.seg_id, st.gen)
                        cur = None  # pico breve: no genera traducción, pero se avisa (T-55)
                        await self.emit(
                            "discarded",
                            {"stream_id": st.stream_id, "seg_id": seg.seg_id, "text": seg.text_es or st.last_partial,
                             "reason": "short"},
                        )
                        continue
                    if not (seg.text_es or "").strip():
                        self._spec_drop(st.stream_id, seg.seg_id, st.gen)
                        cur = None  # ASR final vacío: la fila del parcial queda como descartada (T-55)
                        await self.emit(
                            "discarded",
                            {"stream_id": st.stream_id, "seg_id": seg.seg_id, "text": st.last_partial,
                             "reason": "empty"},
                        )
                        continue
                    await self.emit(
                        "asr_final",
                        {"stream_id": st.stream_id, "text": seg.text_es, "text_raw": seg.text_es_raw,
                         "seg_id": seg.seg_id},
                    )
                    await self._enqueue_segment(st, seg)
                    cur = None
        except asyncio.CancelledError:
            await self.asr_worker.run(self.asr.finish)
            raise

    # ------------------------------------------------------- MT + TTS (cola)
    async def _enqueue_segment(self, st: StreamState, seg: Segment) -> None:
        if not seg.text_es.strip():
            self._spec_drop(st.stream_id, seg.seg_id, st.gen)
            return
        q = self._out_q
        if q.full():  # desbordamiento explícito y visible
            try:
                old_stream, old = q.get_nowait()
                q.task_done()
                self._spec_drop(old_stream, old.seg_id, old.gen)
                await self.emit(
                    "overflow",
                    {"stream_id": old_stream, "seg_id": old.seg_id, "text_es": old.text_es},
                )
            except asyncio.QueueEmpty:
                pass
        await q.put((st.stream_id, seg))

    # ------------------------------------------------------- prewarm MT (T-58)
    @property
    def _prewarm_enabled(self) -> bool:
        v = os.environ.get("OPENTRADUCTOR_MT_PREWARM")
        return (v == "1") if v in ("0", "1") else self.cfg.mt_prewarm_on_pause

    @property
    def _kv_prefill_enabled(self) -> bool:
        v = os.environ.get("OPENTRADUCTOR_MT_KV_PREFILL")
        return (v == "1") if v in ("0", "1") else self.cfg.mt_kv_prefill_on_pause

    def _maybe_prewarm(self, st: StreamState) -> None:
        """Al empezar la pausa VAD de un segmento, si el hilo MLX está ocioso hace >= mt_prewarm_idle_s,
        corre un forward de 1 token de Qwen para que la MT de esa frase no pague la GPU fría. No emite nada."""
        c = self.cfg
        seg = st.seg
        if seg is None or not st.in_speech:
            return
        kv = self._kv_prefill_enabled
        if not kv and seg.seg_id == st.prewarm_seg_id:
            return
        if st.silence_ms < c.mt_prewarm_silence_ms or not (kv or self._prewarm_enabled):
            return
        if c.mt_engine != "qwen" or self.mt is None or not hasattr(self.mt, "prewarm") or self.asr_worker is self.gpu:
            return
        g = self.gpu
        if kv and hasattr(self.mt, "prefill"):
            # Prefill KV: se evalúa en CADA pausa VAD del segmento (no una vez por segmento). En frases
            # largas la 1.ª pausa >= 32 ms llega a mitad de frase, a veces antes del 1.er parcial; si se
            # gastaba ahí, la pausa final no tenía prefill (dignidad 21 pal., @bench 90a0051).
            pause = (seg.seg_id, st.last_speech_end_s)
            if pause == st.kv_pause:
                return
            st.kv_pause = pause
            text = st.last_partial.strip()
            stable = " ".join(text.split()[:-1])
            if stable and (seg.seg_id, stable) != st.kv_stable:
                if g.inflight:
                    self.prewarm_stats["skipped_hot"] += 1
                    return
                st.kv_stable = (seg.seg_id, stable)
                st.prewarm_seg_id = seg.seg_id  # ya despierta la GPU: no hace falta el prewarm de 1 token
                self.prewarm_stats["started"] += 1
                self.prewarm_stats["kv_prefill"] = self.prewarm_stats.get("kv_prefill", 0) + 1
                ws = self.prewarm_stats.setdefault("kv_words", [])
                ws.append(len(text.split()))
                del ws[:-64]
                t = asyncio.ensure_future(g.run(_timed_call, self.mt.prefill, text, *self._mt_args(st.stream_id)))
                self._prewarm_tasks.add(t)
                t.add_done_callback(self._prewarm_done)
                return
            if not stable:
                self.prewarm_stats["kv_no_partial"] = self.prewarm_stats.get("kv_no_partial", 0) + 1
            if seg.seg_id == st.prewarm_seg_id:
                return
            # Sin parcial todavía: prewarm de 1 token (una vez por segmento, solo con GPU fría).
        st.prewarm_seg_id = seg.seg_id  # una vez por segmento
        if g.inflight or time.monotonic() - g.last_done < c.mt_prewarm_idle_s:
            self.prewarm_stats["skipped_hot"] += 1
            return
        self.prewarm_stats["started"] += 1
        t = asyncio.ensure_future(g.run(_timed_call, self.mt.prewarm))
        self._prewarm_tasks.add(t)
        t.add_done_callback(self._prewarm_done)

    def _prewarm_done(self, t: asyncio.Future) -> None:
        self._prewarm_tasks.discard(t)
        if t.cancelled():
            return
        if t.exception() is not None:
            self.prewarm_stats["error"] += 1
            return
        self.prewarm_stats["cost_ms"].append(round(t.result()[1], 1))

    # ------------------------------------------------------- especulación
    def _maybe_speculate(self, st: StreamState) -> None:
        """Lanza (como mucho una vez por segmento) MT + primer fragmento TTS del parcial, solo cuando:
        hay pausa VAD >= spec_silence_ms, el ASR ya procesó TODA la voz recibida (cola vacía y la cola
        de audio pendiente es solo silencio) y no hay trabajo confirmado en curso. No emite nada."""
        c = self.cfg
        seg = st.seg
        if not c.speculative_mt_tts or self.mt is None or seg is None or not st.in_speech:
            return
        if seg.seg_id == st.spec_seg_id or st.silence_ms < c.spec_silence_ms:
            return
        if st.asr_pending_s > 0 or (st.asr_q is not None and not st.asr_q.empty()):
            return
        if st.asr_done_pos_s + 1e-9 < st.last_speech_end_s:
            return
        if st.asr_buf and (st.pos_samples / SR + st.t_base_s - len(st.asr_buf) / 2 / SR) < st.last_speech_end_s:
            return
        if (st.last_speech_end_s - seg.start_s) * 1000 < c.min_speech_ms:
            return
        text = st.last_partial.strip()
        if c.asr_normalize_text:
            text = K.normalize_asr_text(text)
        if not text:
            return
        if self._speaking or (self._out_q is not None and not self._out_q.empty()):
            self.spec_stats["skipped_busy"] += 1  # prioridad al audio confirmado
            return
        if len(self._spec) >= max(1, c.spec_max_entries):
            self.spec_stats["skipped_limit"] += 1
            return
        st.spec_seg_id = seg.seg_id
        e = SpecEntry(st.stream_id, seg.seg_id, text, gen=st.gen, fp=self._spec_fp())
        self._spec[(st.stream_id, st.gen, seg.seg_id)] = e
        self.spec_stats["started"] += 1
        e.task = asyncio.ensure_future(self._spec_run(e))

    async def _spec_run(self, e: SpecEntry) -> None:
        try:
            t = time.perf_counter()
            text_en = await self._mt_exec.run(self.mt.translate, e.text_es, *self._mt_args(e.stream_id))
            e.mt_ms = (time.perf_counter() - t) * 1000
            if e.cancelled:
                return
            e.text_en = text_en
            # TTS especulativo solo si el ASR NO comparte el hilo MLX (si no, retrasaría al propio ASR).
            if self.asr_worker is not self.gpu and not self._speaking and hasattr(self.tts, "plan"):
                clauses = self.tts.plan(text_en)
                e.clauses = clauses
                if clauses:
                    t = time.perf_counter()
                    chunks = await self.gpu.run(lambda: list(self.tts.synth_clauses(clauses[:1], True)))
                    e.tts_ms = (time.perf_counter() - t) * 1000
                    if not e.cancelled:
                        e.first_chunks = chunks
        except asyncio.CancelledError:
            e.cancelled = True
            raise
        except Exception as exc:  # noqa: BLE001
            e.error = f"{type(exc).__name__}: {exc}"

    def _spec_drop(self, stream_id: str | None, seg_id: int | None = None, gen: int | None = None) -> None:
        """Descarta entradas (stop/restart/error/segmento descartado). Nunca emite; el cómputo hecho cuenta como desperdicio."""
        for key in [k for k in self._spec if (stream_id is None or k[0] == stream_id)
                    and (gen is None or k[1] == gen) and (seg_id is None or k[2] == seg_id)]:
            e = self._spec.pop(key)
            self._spec_discard(e)
            self.spec_stats["cancelled"] += 1

    def _spec_discard(self, e: SpecEntry) -> None:
        """Marca la entrada como descartada. Un trabajo ya enviado a un executor (FIFO) NO es cancelable y terminará
        en segundo plano (ocupa su hilo hasta entonces); aquí solo se evita lanzar trabajo nuevo (TTS) y se ignora el
        resultado. Solo el cómputo ya completado se suma a discarded_observed_ms."""
        e.cancelled = True
        if e.task is not None and not e.task.done():
            self.spec_stats["inflight_unmeasured"] += 1
            e.task.cancel()
        self.spec_stats["discarded_observed_ms"] += e.compute_ms

    async def _spec_claim(self, stream_id: str, seg: Segment) -> tuple[SpecEntry | None, str | None]:
        """Devuelve (entrada, estado). Solo es acierto con generación, configuración y texto fuente EXACTAMENTE
        iguales y sin error. Se compara ANTES de esperar la tarea: un fallo conocido no espera ni lanza TTS nuevo."""
        e = self._spec.pop((stream_id, seg.gen, seg.seg_id), None)
        if e is None:
            return None, ("none" if self.cfg.speculative_mt_tts else None)
        if e.fp != self._spec_fp():
            self.spec_stats["miss_config"] += 1
            self._spec_discard(e)
            return None, "miss"
        if e.text_es != seg.text_es:
            self.spec_stats["miss_text"] += 1
            self._spec_discard(e)
            return None, "miss"
        t = time.perf_counter()
        try:
            if e.task is not None:
                await asyncio.shield(e.task)
        except asyncio.CancelledError:
            if not (e.task is not None and e.task.cancelled()):
                raise  # cancelaron al speaker loop, no a la especulación
        waited_ms = (time.perf_counter() - t) * 1000
        if e.cancelled or e.error is not None or e.text_en is None:
            self.spec_stats["error" if e.error else "cancelled"] += 1
            self._spec_discard(e)
            return None, "error" if e.error else "cancelled"
        self.spec_stats["hit"] += 1
        self.spec_stats["overlap_estimate_ms"] += max(0.0, e.compute_ms - waited_ms)
        return e, "hit"

    async def _speaker_loop(self) -> None:
        while True:
            stream_id, seg = await self._out_q.get()
            if not self._is_live(stream_id, seg.gen):  # sesión ya cerrada: ni MT ni TTS
                self.stale_dropped += 1
                self._out_q.task_done()
                continue
            self._speaking = True
            try:
                t0 = time.perf_counter()
                entry, spec = await self._spec_claim(stream_id, seg)
                if entry is not None:
                    text_en = entry.text_en
                elif self._mt_stream_enabled:
                    await self._mt_stream_speak(stream_id, seg, t0)
                    continue
                else:
                    text_en, mt_compute_ms = await self._mt_exec.run(_timed_call, self.mt.translate, seg.text_es,
                                                                      *self._mt_args(stream_id))
                mt_ms = (time.perf_counter() - t0) * 1000
                if not self._is_live(stream_id, seg.gen):  # se cerró durante la MT: se descarta
                    self.stale_dropped += 1
                    continue
                ev = {"stream_id": stream_id, "text": text_en, "seg_id": seg.seg_id, "mt_ms": round(mt_ms, 1)}
                if entry is None:  # T-58: cómputo dentro del hilo vs espera (cola del executor + bucle)
                    ev["mt_compute_ms"] = round(mt_compute_ms, 1)
                if spec is not None:
                    ev["spec"] = spec
                await self.emit("mt_final", ev)
                self._mt_remember(stream_id, seg.text_es, text_en)
                span = (seg.start_s, seg.end_s if seg.end_s is not None else seg.start_s)
                await self._speak(stream_id, seg, text_en, span, entry)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if self._is_live(stream_id, seg.gen):
                    await self.emit(
                        "error",
                        {"stream_id": stream_id, "error_kind": f"{type(exc).__name__}: {exc}"},
                    )
            finally:
                self._speaking = False
                self._out_q.task_done()

    @property
    def _mt_stream_enabled(self) -> bool:
        v = os.environ.get("OPENTRADUCTOR_MT_STREAM_FIRST_CHUNK")
        on = (v == "1") if v in ("0", "1") else self.cfg.mt_stream_first_chunk
        return bool(on) and self.cfg.mt_engine == "qwen" and hasattr(self.mt, "translate_stream")

    async def _mt_stream_speak(self, stream_id, seg, t0: float) -> None:
        """T-58 mt_stream_first_chunk: decode de Qwen token a token y, apenas el prefijo fija ``plan(final)[0]``
        (first_clause_final), Kokoro sintetiza ese 1.er fragmento en el mismo hilo GPU, entre dos pasos del decode,
        y se emite sin esperar el resto de la MT. Después del mt_final, el resto con synth_clauses(plan[1:], False):
        mismos cortes y mismo audio que el camino normal. Con el flag, el tts_chunk del 1.er fragmento puede llegar
        antes que el mt_final de su frase (con planned_text_en=None; los fragmentos siguientes traen el texto
        completo, como en el camino normal)."""
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        stats = self.mt_stream_stats
        stats["segments"] += 1

        def put(item) -> None:
            loop.call_soon_threadsafe(q.put_nowait, item)

        def produce():
            head: str | None = None
            head_t = None
            tts_ms = 0.0

            def on_text(prefix: str) -> None:
                nonlocal head, head_t, tts_ms
                if head is not None or not self._is_live(stream_id, seg.gen):
                    return
                h = self.tts.first_clause_final(prefix)
                if h is None:
                    return
                head = h
                ts = time.perf_counter()
                # planned_text_en=None: el benchmark (metrics.commit_texts) se queda con el 1.er texto no vacío de
                # cada seg_id; si fuera h, el texto planeado de la frase quedaría recortado al fragmento.
                for ch in self.tts.synth_clauses([h], True):
                    put(("chunk", ch, None))
                head_t = time.perf_counter()
                tts_ms += (head_t - ts) * 1000

            try:
                tm = time.perf_counter()
                text_en = self.mt.translate_stream(seg.text_es, *self._mt_args(stream_id), on_text=on_text)
                t_end = time.perf_counter()
                put(("mt", text_en, (t_end - tm) * 1000 - tts_ms))
                clauses = self.tts.plan(text_en) if text_en else []
                if head is None:
                    gen = self.tts.synth_clauses(clauses, True)
                else:
                    stats["head_early"] += 1
                    _append_capped(stats["lead_ms"], round((t_end - head_t) * 1000, 1))
                    _append_capped(stats["tts_in_mt_ms"], round(tts_ms, 1))
                    if clauses[:1] != [head]:
                        stats["mismatch"] += 1
                    if len(clauses) < 2:  # no debería pasar (first_clause_final exige cola): sin texto planeado
                        stats["head_only"] += 1
                    gen = self.tts.synth_clauses(clauses[1:], first=False)
                for ch in gen:
                    if not self._is_live(stream_id, seg.gen):
                        self.stale_dropped += 1
                        break
                    put(("chunk", ch, text_en))
            except Exception as exc:  # noqa: BLE001
                put(("error", exc, None))
            finally:
                put(("done", None, None))

        fut = asyncio.ensure_future(self.gpu.run(produce))
        span = (seg.start_s, seg.end_s if seg.end_s is not None else seg.start_s)
        hold = self._tail_hold()
        try:
            while True:
                kind, v, extra = await q.get()
                if kind == "done":
                    self._tail_done(hold)
                    break
                if not self._is_live(stream_id, seg.gen):
                    self.stale_dropped += 1
                    continue
                if kind == "error":
                    await self.emit("error", {"stream_id": stream_id, "error_kind": f"{type(v).__name__}: {v}"})
                elif kind == "mt":
                    ev = {"stream_id": stream_id, "text": v, "seg_id": seg.seg_id,
                          "mt_ms": round((time.perf_counter() - t0) * 1000, 1), "mt_compute_ms": round(extra, 1),
                          "mt_stream": True}
                    await self.emit("mt_final", ev)
                    self._mt_remember(stream_id, seg.text_es, v)
                else:
                    await self.emit(
                        "tts_chunk",
                        {"stream_id": stream_id, "pcm": hold.feed(v.pcm) if hold else v.pcm, "sample_rate": v.sample_rate,
                         "planned_text_en": extra, "source_span_s": span, "seg_id": seg.seg_id},
                    )
            await fut
        finally:
            if not fut.done():
                fut.cancel()

    async def _speak(self, stream_id, seg, text_en, span, entry: SpecEntry | None = None) -> None:
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        pre = entry.first_chunks if entry is not None and entry.first_chunks else None
        for ch in pre or ():  # fragmento ya sintetizado (acierto exacto): primero, en orden
            q.put_nowait(("chunk", ch))

        def produce():
            try:
                if pre:
                    gen = self.tts.synth_clauses(entry.clauses[1:], first=False) if entry.clauses[1:] else ()
                else:
                    gen = self.tts.synth(text_en)
                for ch in gen:
                    if not self._is_live(stream_id, seg.gen):  # cerrada: dejar de sintetizar
                        self.stale_dropped += 1
                        break
                    loop.call_soon_threadsafe(q.put_nowait, ("chunk", ch))
            except Exception as exc:  # noqa: BLE001
                loop.call_soon_threadsafe(q.put_nowait, ("error", exc))
            finally:
                loop.call_soon_threadsafe(q.put_nowait, ("done", None))

        fut = asyncio.ensure_future(self.gpu.run(produce))
        try:
            await self._drain_tts(q, stream_id, seg, text_en, span)
            await fut
        finally:
            if not fut.done():
                fut.cancel()

    def _tail_hold(self):
        """T-47: TailHold por frase si el tail trim está activo (solo la cola de la frase; ver K.TailHold)."""
        return K.TailHold(K.KokoroTTS.SAMPLE_RATE) if getattr(self.tts, "trim_tail", False) else None

    def _tail_done(self, hold) -> None:
        if hold is not None:
            self.tts.tail_trimmed_ms_total = getattr(self.tts, "tail_trimmed_ms_total", 0.0) + hold.finish()

    async def _drain_tts(self, q, stream_id, seg, text_en, span) -> None:
        hold = self._tail_hold()
        while True:
            kind, v = await q.get()
            if kind == "done":
                self._tail_done(hold)
                break
            if not self._is_live(stream_id, seg.gen):  # sesión cerrada: se descarta lo que llegue
                self.stale_dropped += 1
                continue
            if kind == "error":
                await self.emit(
                    "error",
                    {"stream_id": stream_id, "error_kind": f"{type(v).__name__}: {v}"},
                )
                continue
            await self.emit(
                "tts_chunk",
                {"stream_id": stream_id, "pcm": hold.feed(v.pcm) if hold else v.pcm, "sample_rate": v.sample_rate,
                 "planned_text_en": text_en, "source_span_s": span, "seg_id": seg.seg_id},
            )
