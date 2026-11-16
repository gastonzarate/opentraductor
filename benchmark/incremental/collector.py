"""Collector causal opt-in para alimentar el contrato incremental desde ``benchmark.replay``.

El collector conserva números, texto y metadatos. Nunca retiene PCM de entrada ni de
salida. La reproducción que consumen las métricas incrementales sigue siendo simulada;
este módulo no observa CoreAudio ni certifica ningún umbral de latencia.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmark import audio as A
from benchmark.contracts import AdapterEvent, EventKind

from .contracts import (
    CandidateFields,
    ChunkRecord,
    HarnessStamp,
    IncrementalRun,
    InputDelivery,
    UnitCancelledRecord,
    UnitDoneRecord,
)


class CollectorValidationError(ValueError):
    """Un valor en el borde replay/adaptador viola el contrato del collector."""


@dataclass(frozen=True)
class TimelineEntry:
    """Orden total de una entrega de entrada o evento de salida dentro del stream."""

    event_seq: int
    kind: str
    observed_ns: int
    input_seq: int
    delivered_until_s: float
    chunk_id: int | None = None


@dataclass(frozen=True)
class ObservedAdapterEvent:
    """Evento persistible sin los bytes PCM."""

    stamp: HarnessStamp
    kind: str
    stream_id: str
    text: str | None
    audio_pos_s: float | None
    planned_text_en: str | None
    source_span_claim: tuple[float, float] | None
    source_span_claim_error: str | None
    emission_id: str | None
    error_kind: str | None
    claimed_ns: int | None
    revision: int
    unit_n_chunks: int | None
    unit_total_samples: int | None
    before_emit_claim: bool | None
    unit_reason: str | None
    chunk_id: int | None = None
    pcm_bytes: int | None = None
    pcm_samples: int | None = None
    sample_rate: int | None = None
    pcm_duration_s: float | None = None
    pcm_sha256: str | None = None
    rms_dbfs: float | None = None
    rms_state: str | None = None
    first_nonsilent_offset_s: float | None = None


@dataclass
class _StreamState:
    stream_id: str
    t0_ns: int
    input_duration_s: float
    unit_events_supported: bool = False
    next_event_seq: int = 0
    next_input_seq: int = 0
    next_chunk_id: int = 0
    delivered_until_s: float = 0.0
    last_input_seq: int = -1
    input_deliveries: list[InputDelivery] = field(default_factory=list)
    chunks: list[ChunkRecord] = field(default_factory=list)
    unit_done: list[UnitDoneRecord] = field(default_factory=list)
    unit_cancelled: list[UnitCancelledRecord] = field(default_factory=list)
    events: list[ObservedAdapterEvent] = field(default_factory=list)
    timeline: list[TimelineEntry] = field(default_factory=list)
    validation_errors: list[str] = field(default_factory=list)
    trace_error: str | None = None
    end_ns: int | None = None
    closed: bool = False

    def take_event_seq(self) -> int:
        value = self.next_event_seq
        self.next_event_seq += 1
        return value


class IncrementalCollector:
    """Observer opt-in de ``EventCollector`` y del replay causal.

    Una instancia puede observar varios streams secuenciales. ``to_incremental_run``
    entrega ``alignment=None`` y ``ref_units=[]``: este collector no inventa verdad
    semántica a partir de ``commit_id`` ni ``source_span_s``.
    """

    schema = "opentraductor-incremental-collector/v2"

    def __init__(self, *, threshold_dbfs: float = A.DEFAULT_SILENCE_THRESHOLD_DBFS) -> None:
        if not math.isfinite(threshold_dbfs):
            raise ValueError("threshold_dbfs debe ser finito")
        self.threshold_dbfs = float(threshold_dbfs)
        self.unit_events_supported = False
        self._streams: dict[str, _StreamState] = {}
        self.stray_events: list[dict[str, Any]] = []

    def configure_adapter(self, adapter: Any) -> None:
        """Captura la capability antes del primer stream; no infiere soporte por ausencia."""
        if self._streams:
            raise CollectorValidationError("no se puede cambiar capability con streams registrados")
        self.unit_events_supported = bool(getattr(adapter, "unit_events_supported", False))

    # ---------------------------------------------------------- hooks de replay
    def register_stream(self, trace: Any) -> None:
        stream_id = str(trace.stream_id)
        if stream_id in self._streams:
            raise CollectorValidationError(f"stream duplicado: {stream_id}")
        t0_ns = self._nonnegative_int(trace.t0_ns, "t0_ns")
        duration = self._finite_nonnegative(trace.input_duration_s, "input_duration_s")
        self._streams[stream_id] = _StreamState(
            stream_id, t0_ns, duration, unit_events_supported=self.unit_events_supported
        )

    def record_input_delivery(
        self,
        stream_id: str,
        pcm: bytes,
        start_s: float,
        end_s: float,
        actual_ns: int,
    ) -> None:
        """Registra el bloque inmediatamente antes de llamar ``adapter.push_audio``.

        ``end_s`` es el final de muestra. El hook rechaza usar el inicio como frontera,
        huecos/solapamientos, PCM que no sea s16le y rangos fuera del stream.
        """
        st = self._state(stream_id)
        if st.closed:
            raise CollectorValidationError(f"entrega después del cierre: {stream_id}")
        raw = self._pcm_bytes(pcm, "PCM de entrada")
        if not raw:
            raise CollectorValidationError("PCM de entrada vacío")
        start = self._finite_nonnegative(start_s, "input start_s")
        end = self._finite_nonnegative(end_s, "input end_s")
        samples = len(raw) // 2
        computed_end = start + samples / A.TARGET_SR
        half_sample = 0.5 / A.TARGET_SR
        if abs(end - computed_end) > half_sample + 1e-12:
            raise CollectorValidationError(
                f"end_s no coincide con el final de muestra: {end:.9f} != {computed_end:.9f}"
            )
        if abs(start - st.delivered_until_s) > half_sample + 1e-12:
            raise CollectorValidationError(
                f"entrada no contigua: start_s={start:.9f}, esperado={st.delivered_until_s:.9f}"
            )
        if end > st.input_duration_s + half_sample + 1e-12:
            raise CollectorValidationError(
                f"entrada termina fuera del stream: {end:.9f} > {st.input_duration_s:.9f}"
            )
        now = self._nonnegative_int(actual_ns, "actual_ns")
        input_seq = st.next_input_seq
        event_seq = st.take_event_seq()
        st.next_input_seq += 1
        st.last_input_seq = input_seq
        st.delivered_until_s = end
        delivery = InputDelivery(input_seq, end, now)
        st.input_deliveries.append(delivery)
        st.timeline.append(TimelineEntry(event_seq, "input_delivery", now, input_seq, end))

    def record_adapter_event(self, ev: AdapterEvent, recv_ns: int) -> None:
        """Estampa un evento al recibirlo y conserva solo metadatos persistibles."""
        if not isinstance(ev, AdapterEvent):
            raise CollectorValidationError(f"evento no es AdapterEvent: {type(ev).__name__}")
        if not isinstance(ev.kind, EventKind):
            raise CollectorValidationError(f"kind inválido: {ev.kind!r}")
        st = self._streams.get(ev.stream_id)
        if st is None or st.closed:
            self.stray_events.append(
                {"stream_id": ev.stream_id, "kind": ev.kind.value, "recv_ns": int(recv_ns)}
            )
            return
        event_seq = st.take_event_seq()
        stamp = HarnessStamp(
            recv_ns=self._nonnegative_int(recv_ns, "recv_ns"),
            input_seq=st.last_input_seq,
            delivered_until_s=st.delivered_until_s,
            event_seq=event_seq,
        )
        try:
            observed, chunk, done, cancelled = self._convert_event(st, ev, stamp)
        except CollectorValidationError as exc:
            message = f"event_seq={event_seq} {ev.kind.value}: {exc}"
            st.validation_errors.append(message)
            raise
        st.events.append(observed)
        if ev.kind is EventKind.ERROR and st.trace_error is None:
            detail = ev.error_kind or ev.text or "error sin detalle"
            st.trace_error = f"adapter_event: {detail}"
        if chunk is not None:
            st.chunks.append(chunk)
        if done is not None:
            st.unit_events_supported = True
            st.unit_done.append(done)
        if cancelled is not None:
            st.unit_events_supported = True
            st.unit_cancelled.append(cancelled)
        st.timeline.append(
            TimelineEntry(
                event_seq,
                f"adapter:{ev.kind.value}",
                stamp.recv_ns,
                stamp.input_seq,
                stamp.delivered_until_s,
                observed.chunk_id,
            )
        )

    def finish_stream(self, trace: Any) -> None:
        st = self._streams.get(str(trace.stream_id))
        if st is None:
            return
        st.trace_error = str(trace.error) if trace.error else st.trace_error
        st.end_ns = int(trace.end_ns) if trace.end_ns is not None else None
        st.closed = True

    def record_terminal_trace(self, trace: Any) -> None:
        """Registra un fallo ocurrido antes de que el replay pudiera registrar el stream."""
        if str(trace.stream_id) not in self._streams:
            self.register_stream(trace)
        self.finish_stream(trace)

    # ---------------------------------------------------------- resultados/export
    def stream_ids(self) -> tuple[str, ...]:
        return tuple(self._streams)

    def to_incremental_run(self, stream_id: str) -> IncrementalRun:
        st = self._state(stream_id)
        err = st.trace_error or ("; ".join(st.validation_errors) if st.validation_errors else None)
        timed_out = bool(
            err and ("AdapterHang:" in err or err.startswith("TimeoutError:"))
        )
        return IncrementalRun(
            t0_ns=st.t0_ns,
            input_duration_s=st.input_duration_s,
            ref_units=[],
            chunks=list(st.chunks),
            alignment=None,
            input_deliveries=list(st.input_deliveries),
            source_speech_intervals=None,
            unit_done=list(st.unit_done) if st.unit_events_supported else None,
            unit_cancelled=list(st.unit_cancelled) if st.unit_events_supported else None,
            unit_ready=None,
            n_streams=1,
            n_failed=int(bool(err) and not timed_out),
            n_timeout=int(timed_out),
            notes=["collector real opt-in; reproducción simulada; sin anotación semántica"],
        )

    def export_record(self, stream_id: str) -> dict[str, Any]:
        """Devuelve JSON persistible. No contiene claves con PCM ni audio codificado."""
        st = self._state(stream_id)
        run = self.to_incremental_run(stream_id)
        return {
            "schema": self.schema,
            "stream_id": st.stream_id,
            "playback": "simulated",
            "t0_ns": st.t0_ns,
            "end_ns": st.end_ns,
            "input_duration_s": st.input_duration_s,
            "closed": st.closed,
            "error": st.trace_error,
            "validation_errors": list(st.validation_errors),
            "threshold_dbfs": self.threshold_dbfs,
            "input_deliveries": [dataclasses.asdict(x) for x in st.input_deliveries],
            "timeline": [dataclasses.asdict(x) for x in st.timeline],
            "events": [self._event_dict(x) for x in st.events],
            "incremental": {
                "alignment": None,
                "ref_units": [],
                "n_chunks": len(run.chunks),
                "n_failed": run.n_failed,
                "n_timeout": run.n_timeout,
                "semantic_metrics": "unavailable_without_independent_alignment",
                "unit_events": {
                    "support": "declared" if st.unit_events_supported else "legacy_unknown",
                    "unit_done": None if run.unit_done is None else len(run.unit_done),
                    "unit_cancelled": None if run.unit_cancelled is None else len(run.unit_cancelled),
                    "verification": (
                        "claims del candidato estampados por el arnés; el cierre/cancelación se contrasta después con chunks/playout"
                        if st.unit_events_supported
                        else "adapter legacy: ausencia de eventos no equivale a cero cierres/cancelaciones"
                    ),
                },
            },
            "limitations": [
                "No se persiste PCM ni audio; solo tamaños, hashes, texto y metadatos.",
                "La reproducción es simulada; no mide CoreAudio ni salida física.",
                "Sin alignment independiente no hay métricas semánticas ni afirmación de 300 ms.",
            ],
        }

    def write_json(self, stream_id: str, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(
            json.dumps(self.export_record(stream_id), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        tmp.replace(p)
        return p

    # ---------------------------------------------------------- implementación
    def _state(self, stream_id: str) -> _StreamState:
        try:
            return self._streams[stream_id]
        except KeyError as exc:
            raise CollectorValidationError(f"stream no registrado: {stream_id}") from exc

    def _convert_event(
        self, st: _StreamState, ev: AdapterEvent, stamp: HarnessStamp
    ) -> tuple[ObservedAdapterEvent, ChunkRecord | None, UnitDoneRecord | None, UnitCancelledRecord | None]:
        claim, claim_error = self._source_claim(ev.source_span_s)
        emission = ev.commit_id if isinstance(ev.commit_id, str) and ev.commit_id.strip() else None
        chunk: ChunkRecord | None = None
        done: UnitDoneRecord | None = None
        cancelled: UnitCancelledRecord | None = None
        chunk_id: int | None = None
        pcm_bytes = pcm_samples = sample_rate = None
        pcm_duration = pcm_hash = rms = rms_state = offset = None
        lifecycle_kind = ev.kind in (EventKind.UNIT_DONE, EventKind.UNIT_CANCELLED)
        revision = self._nonnegative_int(ev.revision, "revision")
        unit_reason = self._optional_text(ev.unit_reason, "unit_reason")

        if lifecycle_kind and emission is None:
            raise CollectorValidationError(f"{ev.kind.value} requiere commit_id/emission_id no vacío")

        if ev.kind is EventKind.TTS_CHUNK:
            if ev.pcm is None:
                raise CollectorValidationError("tts_chunk sin PCM")
            raw = self._pcm_bytes(ev.pcm, "PCM de salida")
            if not raw:
                raise CollectorValidationError("PCM de salida vacío")
            sample_rate = A.TARGET_SR if ev.sample_rate is None else self._positive_int(ev.sample_rate, "sample_rate")
            pcm_bytes = len(raw)
            pcm_samples = pcm_bytes // 2
            pcm_duration = pcm_samples / sample_rate
            pcm_hash = hashlib.sha256(raw).hexdigest()
            measured_rms = A.rms_dbfs(raw)
            if math.isfinite(measured_rms):
                rms, rms_state = measured_rms, "measured"
            else:
                rms, rms_state = None, "silence"
            offset = A.first_nonsilent_offset_s(
                raw, sample_rate, threshold_dbfs=self.threshold_dbfs
            )
            if offset is not None and not (0.0 <= offset <= pcm_duration):
                raise CollectorValidationError("offset no silencioso fuera del PCM")
            chunk_id = st.next_chunk_id
            st.next_chunk_id += 1
            candidate_id = emission or f"unclaimed-chunk-{chunk_id}"
            cand = CandidateFields(
                emission_id=candidate_id,
                revision=revision,
                supersedes=None,
                committed=None,
                last_chunk=False,
                speed=None,
                source_span_claim=claim,
            )
            chunk = ChunkRecord(
                stamp=stamp,
                cand=cand,
                pcm_samples=pcm_samples,
                sample_rate=sample_rate,
                nonsilent_offset_s=offset,
                text_en=ev.planned_text_en,
                chunk_id=chunk_id,
            )
        elif ev.kind is EventKind.UNIT_DONE:
            if ev.pcm is not None:
                raise CollectorValidationError("unit_done debe ser standalone y no transportar PCM")
            if ev.unit_n_chunks is None or ev.unit_total_samples is None:
                raise CollectorValidationError("unit_done requiere unit_n_chunks y unit_total_samples")
            done = UnitDoneRecord(
                stamp,
                emission,  # validado arriba
                revision,
                self._nonnegative_int(ev.unit_n_chunks, "unit_n_chunks"),
                self._nonnegative_int(ev.unit_total_samples, "unit_total_samples"),
            )
        elif ev.kind is EventKind.UNIT_CANCELLED:
            if ev.pcm is not None:
                raise CollectorValidationError("unit_cancelled debe ser standalone y no transportar PCM")
            if not isinstance(ev.before_emit_claim, bool):
                raise CollectorValidationError("unit_cancelled requiere before_emit_claim bool")
            if ev.unit_n_chunks is not None or ev.unit_total_samples is not None:
                raise CollectorValidationError("unit_cancelled no acepta conteos de unit_done")
            cancelled = UnitCancelledRecord(
                stamp,
                emission,  # validado arriba
                revision,
                ev.before_emit_claim,
            )
        elif ev.pcm is not None:
            raise CollectorValidationError(f"{ev.kind.value} no puede transportar PCM")

        observed = ObservedAdapterEvent(
            stamp=stamp,
            kind=ev.kind.value,
            stream_id=ev.stream_id,
            text=ev.text,
            audio_pos_s=self._optional_finite(ev.audio_pos_s, "audio_pos_s"),
            planned_text_en=ev.planned_text_en,
            source_span_claim=claim,
            source_span_claim_error=claim_error,
            emission_id=emission,
            error_kind=ev.error_kind,
            claimed_ns=ev.claimed_ns,
            revision=revision,
            unit_n_chunks=ev.unit_n_chunks,
            unit_total_samples=ev.unit_total_samples,
            before_emit_claim=ev.before_emit_claim,
            unit_reason=unit_reason,
            chunk_id=chunk_id,
            pcm_bytes=pcm_bytes,
            pcm_samples=pcm_samples,
            sample_rate=sample_rate,
            pcm_duration_s=pcm_duration,
            pcm_sha256=pcm_hash,
            rms_dbfs=rms,
            rms_state=rms_state,
            first_nonsilent_offset_s=offset,
        )
        return observed, chunk, done, cancelled

    @staticmethod
    def _event_dict(ev: ObservedAdapterEvent) -> dict[str, Any]:
        return dataclasses.asdict(ev)

    @staticmethod
    def _pcm_bytes(value: Any, label: str) -> bytes:
        if not isinstance(value, (bytes, bytearray, memoryview)):
            raise CollectorValidationError(f"{label} debe ser bytes s16le")
        raw = bytes(value)
        if len(raw) % 2:
            raise CollectorValidationError(f"{label} tiene cantidad impar de bytes")
        return raw

    @staticmethod
    def _positive_int(value: Any, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise CollectorValidationError(f"{label} debe ser entero positivo")
        return value

    @staticmethod
    def _nonnegative_int(value: Any, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CollectorValidationError(f"{label} debe ser entero no negativo")
        return value

    @staticmethod
    def _finite_nonnegative(value: Any, label: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CollectorValidationError(f"{label} debe ser numérico")
        out = float(value)
        if not math.isfinite(out) or out < 0:
            raise CollectorValidationError(f"{label} debe ser finito y no negativo")
        return out

    @staticmethod
    def _optional_finite(value: Any, label: str) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise CollectorValidationError(f"{label} debe ser finito o None")
        return float(value)

    @staticmethod
    def _optional_text(value: Any, label: str) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise CollectorValidationError(f"{label} debe ser str o None")
        return value

    @staticmethod
    def _source_claim(value: Any) -> tuple[tuple[float, float] | None, str | None]:
        if value is None:
            return None, None
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            return None, "source_span_s no es un par"
        a, b = value
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)) for x in (a, b)):
            return None, "source_span_s no es finito"
        af, bf = float(a), float(b)
        if af < 0 or bf < af:
            return None, "source_span_s invertido o negativo"
        return (af, bf), None
