"""Replay causal de audio a ritmo real y recolección de eventos.

Reglas de reloj (ver docs/benchmark.md):

* El instante ``t0`` del stream es el instante del reloj del arnés en que el
  audio en la posición 0 "ocurre". Una muestra en posición ``s`` ocurre en
  ``t0 + s``.
* Un chunk que cubre ``[a, b)`` solo se entrega al candidato en ``t0 + b``
  (cuando su última muestra ya ocurrió). El candidato nunca ve audio futuro.
* El origen de latencia de fin de fragmento es ``t0 + utterance_end_s``
  (anotado), jamás un evento del candidato ni del VAD del sistema.
* Cada evento se estampa con el reloj del arnés AL RECIBIRLO; ``claimed_ns``
  del candidato se conserva solo como dato no confiable.
* El audio de relleno final (silencio) cubre toda la ventana de censura para
  que ``end_stream`` no actúe como pista de fin de frase.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Protocol

from . import audio as A
from .clock import Clock
from .contracts import Adapter, AdapterEvent, AdapterMode, EventKind


@dataclass
class Segment:
    """Un fragmento de habla dentro de un stream (posiciones en s del stream)."""

    item_id: str
    start_s: float | None
    end_s: float | None
    boundary_method: str = "unknown"
    boundary_uncertainty_s: float | None = None
    # Metadatos que deben sobrevivir hasta el informe:
    tags: tuple[str, ...] = ()  # fenómenos del ítem (voseo, numeros, negacion...)
    cluster: str | None = None  # hablante si se conoce; si no, grabación
    speaker_group: str | None = None
    transcript_provenance: str | None = None
    reference_provenance: str | None = None


@dataclass
class TraceEvent:
    recv_ns: int  # reloj del arnés
    kind: str
    stream_id: str
    text: str | None = None
    audio_pos_s: float | None = None
    pcm_samples: int | None = None
    sample_rate: int | None = None
    rms_dbfs: float | None = None
    first_nonsilent_offset_s: float | None = None
    planned_text_en: str | None = None
    source_span_s: tuple[float, float] | None = None
    commit_id: str | None = None
    error_kind: str | None = None
    claimed_ns: int | None = None  # NO confiable
    revision: int = 0
    unit_n_chunks: int | None = None
    unit_total_samples: int | None = None
    before_emit_claim: bool | None = None
    unit_reason: str | None = None
    pcm: bytes | None = field(default=None, repr=False)  # solo en memoria


@dataclass
class SendRecord:
    audio_end_s: float
    scheduled_ns: int
    actual_ns: int

    @property
    def lateness_ns(self) -> int:
        return self.actual_ns - self.scheduled_ns


@dataclass
class StreamTrace:
    stream_id: str
    adapter_mode: str
    clock_kind: str
    t0_ns: int
    input_duration_s: float  # incluye relleno
    segments: list[Segment]
    sends: list[SendRecord] = field(default_factory=list)
    events: list[TraceEvent] = field(default_factory=list)
    end_ns: int | None = None
    error: str | None = None
    seed: int = 0
    start_stream_ms: float | None = None  # carga diferida: cuenta para cold start
    produces_audio: bool = True
    deadline_s: float = 0.0
    continuous: bool = False

    def abs_ns(self, audio_s: float) -> int:
        return self.t0_ns + round(audio_s * 1e9)


class ReplayObserver(Protocol):
    """Hook opt-in; el replay normal no depende del módulo incremental."""

    def register_stream(self, trace: StreamTrace) -> None: ...

    def record_input_delivery(
        self, stream_id: str, pcm: bytes, start_s: float, end_s: float, actual_ns: int
    ) -> None: ...

    def record_adapter_event(self, ev: AdapterEvent, recv_ns: int) -> None: ...

    def finish_stream(self, trace: StreamTrace) -> None: ...

    def record_terminal_trace(self, trace: StreamTrace) -> None: ...


class EventCollector:
    """Consume ``adapter.events()`` y estampa el reloj del arnés al recibir."""

    def __init__(
        self,
        adapter: Adapter,
        clock: Clock,
        *,
        threshold_dbfs: float = A.DEFAULT_SILENCE_THRESHOLD_DBFS,
        keep_pcm: bool = False,
        observer: ReplayObserver | None = None,
    ) -> None:
        self.adapter = adapter
        self.clock = clock
        self.threshold_dbfs = threshold_dbfs
        self.keep_pcm = keep_pcm
        self.observer = observer
        if observer is not None:
            configure = getattr(observer, "configure_adapter", None)
            if callable(configure):
                configure(adapter)
        self._traces: dict[str, StreamTrace] = {}
        self.stray: list[TraceEvent] = []
        self._task: asyncio.Task | None = None
        self.collector_error: str | None = None

    def start(self) -> None:
        self._task = asyncio.ensure_future(self._run())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass

    def register(self, trace: StreamTrace) -> None:
        self._traces[trace.stream_id] = trace
        try:
            if self.observer is not None:
                self.observer.register_stream(trace)
        except Exception as exc:
            self._mark_collector_error("observer.register_stream", exc)
            raise
        if self.collector_error is not None:
            trace.error = trace.error or self.collector_error
            raise RuntimeError(f"EventCollector no disponible: {self.collector_error}")

    def _mark_collector_error(self, where: str, exc: Exception) -> None:
        message = f"{where}: {type(exc).__name__}: {exc}"
        self.collector_error = message
        # El iterador de eventos es compartido por los streams. Si muere, ningún
        # stream activo ni futuro puede presentarse como observado exitosamente.
        for trace in self._traces.values():
            if trace.error is None:
                trace.error = message

    def unregister(self, stream_id: str) -> None:
        trace = self._traces.pop(stream_id, None)
        if trace is not None and self.observer is not None:
            try:
                self.observer.finish_stream(trace)
            except Exception as exc:
                self._mark_collector_error("observer.finish_stream", exc)
                if trace.error is None:
                    trace.error = self.collector_error

    def record_terminal_trace(self, trace: StreamTrace) -> None:
        if self.observer is None:
            return
        try:
            self.observer.record_terminal_trace(trace)
        except Exception as exc:
            self._mark_collector_error("observer.record_terminal_trace", exc)
            if trace.error is None:
                trace.error = self.collector_error

    async def _run(self) -> None:
        try:
            async for ev in self.adapter.events():
                recv = self.clock.now_ns()  # estampa inmediata
                if self.observer is not None:
                    self.observer.record_adapter_event(ev, recv)
                te = self._to_trace_event(ev, recv)
                tr = self._traces.get(ev.stream_id)
                if tr is None:
                    self.stray.append(te)
                else:
                    tr.events.append(te)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # error del candidato, del iterador o del observer
            self._mark_collector_error("event_collection", exc)

    def _to_trace_event(self, ev: AdapterEvent, recv_ns: int) -> TraceEvent:
        te = TraceEvent(
            recv_ns=recv_ns,
            kind=ev.kind.value,
            stream_id=ev.stream_id,
            text=ev.text,
            audio_pos_s=ev.audio_pos_s,
            planned_text_en=ev.planned_text_en,
            source_span_s=ev.source_span_s,
            commit_id=ev.commit_id,
            error_kind=ev.error_kind,
            claimed_ns=ev.claimed_ns,
            revision=ev.revision,
            unit_n_chunks=ev.unit_n_chunks,
            unit_total_samples=ev.unit_total_samples,
            before_emit_claim=ev.before_emit_claim,
            unit_reason=ev.unit_reason,
        )
        if ev.pcm is not None:
            sr = ev.sample_rate or A.TARGET_SR
            te.pcm_samples = len(ev.pcm) // 2
            te.sample_rate = sr
            te.rms_dbfs = A.rms_dbfs(ev.pcm)
            te.first_nonsilent_offset_s = A.first_nonsilent_offset_s(
                ev.pcm, sr, threshold_dbfs=self.threshold_dbfs
            )
            if self.keep_pcm:
                te.pcm = ev.pcm
        return te


class AdapterHang(Exception):
    """Una llamada al adaptador no retornó dentro del límite."""


async def _call(coro, timeout_s: float, what: str):
    try:
        return await asyncio.wait_for(coro, timeout_s)
    except asyncio.TimeoutError:
        raise AdapterHang(f"{what} no retornó en {timeout_s:.1f}s") from None


async def run_stream(
    adapter: Adapter,
    clock: Clock,
    collector: EventCollector,
    stream_id: str,
    pcm: bytes,
    segments: list[Segment],
    *,
    seed: int = 0,
    chunk_ms: float = 20.0,
    deadline_s: float = 5.0,
    drain_grace_s: float = 0.5,
    idle_ms: float = 500.0,
    text_es: str | None = None,
    continuous: bool = False,
    call_timeout_s: float = 10.0,
) -> StreamTrace:
    """Reproduce un stream y devuelve su traza. Un stream a la vez.

    Toda llamada al adaptador tiene límite (``call_timeout_s``). Un cuelgue se
    registra en ``trace.error`` y la limpieza (unregister) está garantizada.
    Nota: un adaptador que bloquea el event loop con código síncrono no puede
    interrumpirse aquí; lo cubre el watchdog de proceso (``watchdog.py``).
    """
    mode = adapter.mode
    audio_dur = A.pcm_duration_s(pcm)
    last_end = max([s.end_s for s in segments if s.end_s is not None] + [audio_dur])
    if mode is AdapterMode.STREAMING:
        # Relleno de silencio hasta cubrir la ventana de censura completa.
        pcm = pcm + A.silence(max(0.0, last_end + deadline_s - audio_dur))
    total_dur = A.pcm_duration_s(pcm)
    trace = StreamTrace(
        stream_id=stream_id,
        adapter_mode=mode.value,
        clock_kind=clock.kind,
        t0_ns=0,
        input_duration_s=total_dur,
        segments=segments,
        seed=seed,
        deadline_s=deadline_s,
        continuous=continuous,
        produces_audio=adapter.produces_audio,
    )
    try:
        s0 = clock.now_ns()
        await _call(adapter.start_stream(stream_id, seed), call_timeout_s, "start_stream")
        trace.start_stream_ms = (clock.now_ns() - s0) / 1e6  # carga diferida cuenta
    except Exception as exc:
        trace.t0_ns = clock.now_ns()
        trace.error = f"{type(exc).__name__}: {exc}"
        trace.end_ns = trace.t0_ns
        collector.record_terminal_trace(trace)
        return trace
    trace.t0_ns = clock.now_ns()
    try:
        collector.register(trace)
        if mode is AdapterMode.STREAMING:
            await _replay_causal(
                adapter, clock, trace, pcm, chunk_ms, call_timeout_s,
                observer=collector.observer,
            )
            await clock.sleep_until_ns(clock.now_ns() + round(drain_grace_s * 1e9))
            await _call(adapter.end_stream(stream_id), call_timeout_s, "end_stream")
        elif mode is AdapterMode.OFFLINE_FILE:
            await _call(adapter.process_file(stream_id, pcm), call_timeout_s + deadline_s, "process_file")
            await _wait_quiescent(clock, trace, deadline_s, adapter.quiescence_ms or idle_ms)
        elif mode is AdapterMode.TEXT:
            if text_es is None:
                raise ValueError("modo text requiere text_es")
            await _call(adapter.process_text(stream_id, text_es), call_timeout_s + deadline_s, "process_text")
            await _wait_quiescent(clock, trace, deadline_s, adapter.quiescence_ms or idle_ms)
    except Exception as exc:
        trace.error = f"{type(exc).__name__}: {exc}"
    finally:
        trace.end_ns = clock.now_ns()
        collector.unregister(stream_id)
    return trace


async def _replay_causal(
    adapter: Adapter,
    clock: Clock,
    trace: StreamTrace,
    pcm: bytes,
    chunk_ms: float,
    call_timeout_s: float,
    *,
    observer: ReplayObserver | None = None,
) -> None:
    sr = A.TARGET_SR
    n = max(1, round(sr * chunk_ms / 1000)) * 2  # bytes por chunk
    for off in range(0, len(pcm), n):
        chunk = pcm[off : off + n]
        start_s = off / 2 / sr
        end_s = (off + len(chunk)) / 2 / sr
        scheduled = trace.abs_ns(end_s)  # recién entonces "ocurrió"
        await clock.sleep_until_ns(scheduled)

        async def deliver() -> None:
            # ``actual`` se toma cuando este coroutine realmente entra, no antes
            # de que wait_for lo programe. El SendRecord y el observer comparten
            # exactamente ese instante, inmediatamente antes de push_audio.
            actual = clock.now_ns()
            trace.sends.append(SendRecord(end_s, scheduled, actual))
            if observer is not None:
                observer.record_input_delivery(
                    trace.stream_id, chunk, start_s, end_s, actual
                )
            await adapter.push_audio(trace.stream_id, chunk, start_s)

        await _call(deliver(), call_timeout_s, "push_audio")


async def _wait_quiescent(
    clock: Clock, trace: StreamTrace, deadline_s: float, idle_ms: float
) -> None:
    """Espera hasta recibir audio y quedar inactivo, o hasta el deadline."""
    hard = trace.t0_ns + round(deadline_s * 1e9)
    step = 5_000_000
    while clock.now_ns() < hard:
        await clock.sleep_until_ns(clock.now_ns() + step)
        out = [e for e in trace.events if (e.pcm_samples if trace.produces_audio else (e.kind == "mt_final" and e.text))]
        if out and clock.now_ns() - max(e.recv_ns for e in trace.events) > idle_ms * 1e6:
            return
