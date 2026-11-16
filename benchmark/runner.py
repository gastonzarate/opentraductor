"""Orquestador de corridas: cold start, calentamiento, repeticiones, semillas."""
from __future__ import annotations

import asyncio
import dataclasses
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import audio as A
from . import export as X
from . import metrics, quality, snapshot
from .clock import Clock
from .contracts import Adapter, AdapterMode, ManifestItem, resolve_audio_path
from .replay import AdapterHang, EventCollector, Segment, StreamTrace, run_stream
from .watchdog import Watchdog


class RunError(RuntimeError):
    """La corrida no pudo completarse; el fallo queda en FAILURE.json."""


@dataclass
class BenchItem:
    item_id: str
    recording_id: str
    pcm: bytes  # PCM16 mono 16 kHz (público; nunca audio de conversaciones)
    segment: Segment
    split: str = "development"
    text_es: str | None = None
    reference_en: tuple[str, ...] | None = None
    infra_synthetic: bool = False

    @property
    def cluster(self) -> str:
        return self.segment.cluster or self.recording_id


def items_from_manifest(items: list[ManifestItem], audio_root: str | Path) -> list[BenchItem]:
    out = []
    for it in items:
        pcm = A.read_wav_pcm16_mono_16k(resolve_audio_path(it, audio_root))
        out.append(
            BenchItem(
                item_id=it.item_id,
                recording_id=f"{it.source_id}:{it.recording_id}",
                pcm=pcm,
                split=it.split,
                text_es=it.transcript_es,
                reference_en=it.reference_en,
                segment=Segment(
                    item_id=it.item_id,
                    start_s=it.utterance_start_s,
                    end_s=it.utterance_end_s,
                    boundary_method=it.boundary_method,
                    boundary_uncertainty_s=it.boundary_uncertainty_s,
                    # Cluster conservador: hablante, o fuente si es desconocido.
                    cluster=it.speaker_group,
                    speaker_group=it.speaker_group,
                    transcript_provenance=it.transcript_es_provenance,
                    reference_provenance=it.reference_en_provenance,
                ),
            )
        )
    return out


@dataclass
class RunConfig:
    adapter_config: dict[str, Any] = field(default_factory=dict)
    repetitions: int = 3
    warmup_streams: int = 2
    seed: int = 1234
    chunk_ms: float = 20.0
    deadline_s: float = 5.0
    drain_grace_s: float = 0.5
    silence_threshold_dbfs: float = A.DEFAULT_SILENCE_THRESHOLD_DBFS
    continuous: bool = False
    continuous_gap_s: float = 1.5
    n_boot: int = 1000
    keep_pcm: bool = False
    purpose: str = "development"  # "development" | "final_test"
    out_dir: str | None = None
    # Observer causal incremental: opt-in; no cambia defaults ni crea alineaciones.
    incremental_collector: bool = False
    # Robustez:
    call_timeout_s: float = 10.0  # por llamada al adaptador
    open_timeout_s: float = 600.0  # carga del modelo
    watchdog_stall_s: float = 30.0  # solo reloj real; 0 = desactivado
    abort_after_hangs: int = 2  # cuelgues consecutivos antes de abortar
    # Export de audio de SALIDA del candidato (solo benchmark) para escucha/proxies:
    export_audio_dir: str | None = None
    system_name: str | None = None  # etiqueta del sistema en el índice; por defecto adapter.name


@dataclass
class RunResult:
    meta: dict
    segment_results: list[metrics.SegmentResult]
    traces: list[StreamTrace]
    warmup_traces: list[StreamTrace]
    aggregate: dict
    playout: list[dict]
    cold_start: dict
    quality: dict


def _build_continuous(items: list[BenchItem], gap_s: float) -> tuple[bytes, list[Segment]]:
    pcm = bytearray(A.silence(gap_s))
    segs: list[Segment] = []
    for it in items:
        off = len(pcm) / 2 / A.TARGET_SR
        pcm.extend(it.pcm)
        s = it.segment
        segs.append(
            dataclasses.replace(
                s,
                start_s=None if s.start_s is None else off + s.start_s,
                end_s=None if s.end_s is None else off + s.end_s,
            )
        )
        pcm.extend(A.silence(gap_s))
    return bytes(pcm), segs


def _write_failure(out_dir: str | None, rec: dict) -> None:
    if not out_dir:
        return
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    (p / "FAILURE.json").write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")


def _write_incremental_records(
    observer: Any,
    warmup: list[StreamTrace],
    traces: list[StreamTrace],
    out_dir: str | Path,
) -> dict[str, Any]:
    """Persiste timeline/metadatos por stream, nunca PCM ni alineación inventada."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "incremental-collector.jsonl"
    tmp = path.with_suffix(path.suffix + ".tmp")
    observed = set(observer.stream_ids())
    rows = []
    for trace, is_warmup in [(t, True) for t in warmup] + [(t, False) for t in traces]:
        if trace.stream_id not in observed:
            continue
        row = observer.export_record(trace.stream_id)
        row["warmup"] = is_warmup
        rows.append(row)
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, allow_nan=False, default=str) + "\n")
    tmp.replace(path)
    def group_summary(group_rows: list[dict[str, Any]], group_traces: list[StreamTrace]) -> dict[str, Any]:
        aborted = [t.stream_id for t in group_traces if t.error and t.error.startswith("AdapterHang: omitido")]
        expected = {t.stream_id for t in group_traces if t.stream_id not in aborted}
        recorded = {r["stream_id"] for r in group_rows}
        inc = [r["incremental"] for r in group_rows]
        return {
            "n_records": len(group_rows),
            "n_failed": sum(r["n_failed"] for r in inc),
            "n_timeout": sum(r["n_timeout"] for r in inc),
            "n_expected_without_aborted_skips": len(expected),
            "missing_stream_ids": sorted(expected - recorded),
            "n_aborted_skips": len(aborted),
            "aborted_skip_stream_ids": sorted(aborted),
        }

    warm_rows = [r for r in rows if r["warmup"]]
    measured_rows = [r for r in rows if not r["warmup"]]
    all_inc = [r["incremental"] for r in rows]
    return {
        "enabled": True,
        "path": str(path),
        "schema": observer.schema,
        "playback": "simulated",
        "contains_pcm": False,
        "alignment": None,
        "semantic_metrics": "unavailable_without_independent_alignment",
        "total": {
            "scope": "warmup_plus_measured",
            "n_records": len(rows),
            "n_failed": sum(r["n_failed"] for r in all_inc),
            "n_timeout": sum(r["n_timeout"] for r in all_inc),
        },
        "warmup": group_summary(warm_rows, warmup),
        "measured": group_summary(measured_rows, traces),
    }


def _skipped_trace(stream_id: str, clock: Clock, mode: str, batch: list[BenchItem], cfg: RunConfig, why: str, continuous: bool) -> StreamTrace:
    segs = _build_continuous(batch, cfg.continuous_gap_s)[1] if continuous else [batch[0].segment]
    now = clock.now_ns()
    return StreamTrace(
        stream_id=stream_id, adapter_mode=mode, clock_kind=clock.kind, t0_ns=now,
        input_duration_s=0.0, segments=segs, error=why, deadline_s=cfg.deadline_s,
        continuous=continuous, end_ns=now,
    )


async def run_benchmark(
    adapter: Adapter,
    clock: Clock,
    items: list[BenchItem],
    cfg: RunConfig,
    *,
    manifest_path: str | Path | None = None,
    audible_asr: quality.AudibleAsr | None = None,
) -> RunResult:
    splits = {i.split for i in items}
    if len(splits) > 1:
        raise ValueError(f"no mezclar splits en una corrida: {splits}")
    split = splits.pop() if splits else None
    if cfg.purpose == "development" and split == "test":
        raise ValueError("purpose=development no puede usar el split test")
    if cfg.purpose == "final_test" and split != "test":
        raise ValueError("purpose=final_test requiere split test")
    if not items:
        raise ValueError("sin items")
    if cfg.incremental_collector and not cfg.out_dir:
        raise ValueError("incremental_collector requiere out_dir para persistir evidencia")
    if cfg.incremental_collector and adapter.mode is not AdapterMode.STREAMING:
        raise ValueError("incremental_collector solo admite adaptadores streaming")

    # Sin hablante conocido, el cluster mínimo es la grabación (nunca el stream:
    # repetir un clip no aporta muestras independientes).
    for it in items:
        if it.segment.cluster is None:
            it.segment = dataclasses.replace(it.segment, cluster=it.recording_id)

    env = snapshot.environment_snapshot()
    t_wall_start = time.time()
    incremental_observer = None
    if cfg.incremental_collector:
        from .incremental.collector import IncrementalCollector

        incremental_observer = IncrementalCollector(threshold_dbfs=cfg.silence_threshold_dbfs)
    collector = EventCollector(
        adapter,
        clock,
        threshold_dbfs=cfg.silence_threshold_dbfs,
        keep_pcm=cfg.keep_pcm or audible_asr is not None or cfg.export_audio_dir is not None,
        observer=incremental_observer,
    )
    warm: list[StreamTrace] = []
    traces: list[StreamTrace] = []
    results: list[metrics.SegmentResult] = []
    playout: list[dict] = []
    progress: dict[str, Any] = {"phase": "init", "stream_id": None}
    aborted: dict[str, Any] | None = None
    open_ms: float | None = None

    wd: Watchdog | None = None
    if clock.kind == "real" and cfg.watchdog_stall_s > 0:
        wd = Watchdog(
            cfg.watchdog_stall_s,
            Path(cfg.out_dir) / "FAILURE.json" if cfg.out_dir else None,
            lambda: {
                **progress,
                "n_segment_results_so_far": len(results),
                "partial_outcomes": [r.outcome for r in results],
            },
        )
        wd.start()

    async def _one(stream_id: str, batch: list[BenchItem], seed: int) -> StreamTrace:
        progress["stream_id"] = stream_id
        if cfg.continuous:
            pcm, segs = _build_continuous(batch, cfg.continuous_gap_s)
        else:
            assert len(batch) == 1
            pcm, segs = batch[0].pcm, [batch[0].segment]
        return await run_stream(
            adapter, clock, collector, stream_id, pcm, segs,
            seed=seed, chunk_ms=cfg.chunk_ms, deadline_s=cfg.deadline_s,
            drain_grace_s=cfg.drain_grace_s, text_es=batch[0].text_es,
            continuous=cfg.continuous, call_timeout_s=cfg.call_timeout_s,
        )

    def _batches(order: list[BenchItem]) -> list[list[BenchItem]]:
        return [order] if cfg.continuous else [[i] for i in order]

    consecutive_hangs = 0
    export_records: list[dict] = []

    def _is_hang(t: StreamTrace) -> bool:
        return bool(t.error) and t.error.startswith("AdapterHang")

    try:
        # ---- cold start: carga + primer start_stream/stream ----
        progress["phase"] = "open"
        c0 = clock.now_ns()
        try:
            await asyncio.wait_for(adapter.open(cfg.adapter_config), cfg.open_timeout_s)
        except Exception as exc:
            rec = {"failure": "open_failed", "error": f"{type(exc).__name__}: {exc}"}
            _write_failure(cfg.out_dir, rec)
            raise RunError(rec["error"]) from exc
        open_ms = (clock.now_ns() - c0) / 1e6
        collector.start()

        progress["phase"] = "warmup"
        nb = len(_batches(list(items)))
        for w in range(cfg.warmup_streams):
            t = await _one(f"warmup{w}", _batches(list(items))[w % nb], cfg.seed + 10_000 + w)
            warm.append(t)
            consecutive_hangs = consecutive_hangs + 1 if _is_hang(t) else 0

        progress["phase"] = "measure"
        for rep in range(cfg.repetitions):
            order = list(items)
            random.Random(cfg.seed + rep).shuffle(order)
            for bi, batch in enumerate(_batches(order)):
                sid = f"r{rep}-b{bi}-{batch[0].item_id}"
                if consecutive_hangs >= cfg.abort_after_hangs:
                    if aborted is None:
                        aborted = {"reason": f"{consecutive_hangs} cuelgues consecutivos", "at_stream": sid}
                    t = _skipped_trace(sid, clock, adapter.mode.value, batch, cfg,
                                       "AdapterHang: omitido, corrida abortada tras cuelgues consecutivos",
                                       cfg.continuous)
                else:
                    t = await _one(sid, batch, cfg.seed + rep)
                    consecutive_hangs = consecutive_hangs + 1 if _is_hang(t) else 0
                traces.append(t)
                srs = metrics.segment_results(t)
                results.extend(srs)
                if cfg.export_audio_dir:
                    rec = X.export_stream_audio(
                        t, cfg.export_audio_dir,
                        system=cfg.system_name or adapter.name,
                        outcome_by_item={r.item_id: r.outcome for r in srs},
                        run_label=("INFRA-TEST dummy" if adapter.infra_test_only else "benchmark"),
                    )
                    if rec:
                        export_records.append(rec)
                pa = metrics.playout_analysis(t)
                pa["send_jitter"] = metrics.send_jitter(t)
                playout.append(pa)
        progress["phase"] = "done"
    finally:
        # Cierre garantizado de adaptador, colector y watchdog.
        try:
            await asyncio.wait_for(adapter.close(), cfg.call_timeout_s)
        except Exception:
            pass
        await collector.stop()
        if wd is not None:
            wd.stop()

    agg = metrics.aggregate(results, n_boot=cfg.n_boot, seed=cfg.seed)
    refs = {i.item_id: (i.reference_en, i.segment.reference_provenance) for i in items}
    transcripts = {i.item_id: (i.text_es, i.segment.transcript_provenance) for i in items}
    qual = quality.evaluate_run(results, traces, refs, audible_asr, transcripts)
    first = warm[0] if warm else (traces[0] if traces else None)
    first_sr = metrics.segment_results(first) if first else []
    cold = {
        "open_ms": open_ms,
        "first_start_stream_ms": first.start_stream_ms if first else None,
        "first_stream_end_to_first_ms": next((r.end_to_first_ms for r in first_sr if r.outcome == "ok"), None),
        "first_stream_request_to_first_ms": next((r.request_to_first_ms for r in first_sr if r.outcome == "ok"), None),
        "first_stream_is_warmup": bool(warm),
        "warmup_streams": cfg.warmup_streams,
        "note": "open() y start_stream() (carga diferida) medidos aparte; calentamiento excluido de estadísticas, guardado en traces",
    }
    incremental_meta: dict[str, Any] = {"enabled": False}
    if incremental_observer is not None:
        try:
            incremental_meta = _write_incremental_records(
                incremental_observer, warm, traces, cfg.out_dir
            )
        except Exception as exc:
            rec = {"failure": "incremental_collector_export_failed", "error": f"{type(exc).__name__}: {exc}"}
            _write_failure(cfg.out_dir, rec)
            raise RunError(rec["error"]) from exc
    load_end = snapshot.load_snapshot()
    contended = bool(env["load_at_start"].get("contention_possible") or load_end.get("contention_possible"))
    meta = {
        "label": "INFRA-TEST (dummy): NO es un benchmark real" if adapter.infra_test_only else "benchmark",
        "infra_test_only": adapter.infra_test_only,
        "adapter": {"name": adapter.name, "version": adapter.version, "mode": adapter.mode.value},
        "adapter_config": cfg.adapter_config,
        "run_config": {k: v for k, v in cfg.__dict__.items() if k != "adapter_config"},
        "split": split,
        "n_items": len(items),
        "items_infra_synthetic": any(i.infra_synthetic for i in items),
        "manifest_sha256": snapshot.file_sha256(manifest_path) if manifest_path else None,
        "environment": env,
        "load_at_end": load_end,
        "contention_possible": contended,
        "run_status_label": "PILOTO: contención posible (procesos ajenos activos); no es comparación final"
        if contended else "sin contención observada al inicio y al final (no garantiza reposo)",
        "clock_kind": clock.kind,
        "started_unix": t_wall_start,
        "collector_error": collector.collector_error,
        "incremental_collector": incremental_meta,
        "stray_events": len(collector.stray),
        "silence_threshold_dbfs": cfg.silence_threshold_dbfs,
        "aborted": aborted,
        "audible_asr_configured": audible_asr is not None,
    }
    try:
        meta["adapter_metadata"] = adapter.describe()
    except Exception as exc:  # no tumbar la corrida, pero dejar constancia
        meta["adapter_metadata"] = {"error": f"{type(exc).__name__}: {exc}"}
    if not meta["adapter_metadata"]:
        meta["adapter_metadata_warning"] = "el adaptador no declaró describe(): pesos/versiones del candidato NO quedan registrados"
    if cfg.export_audio_dir:
        idx = X.write_index(export_records, cfg.export_audio_dir) if export_records else None
        meta["audio_export"] = {
            "dir": cfg.export_audio_dir,
            "index": str(idx) if idx else None,
            "n_exported": sum(1 for r in export_records if r.get("exported")),
            "n_failed": sum(1 for r in export_records if not r.get("exported")),
            "scope": "solo benchmark; nunca audio de conversaciones",
        }
    meta["audible_asr_label"] = getattr(audible_asr, "label", None) if audible_asr else None
    res = RunResult(meta, results, traces, warm, agg, playout, cold, qual)
    if cfg.out_dir:
        write_run(res, cfg.out_dir)
    return res


def trace_to_dict(t: StreamTrace) -> dict:
    return {
        "stream_id": t.stream_id,
        "adapter_mode": t.adapter_mode,
        "clock_kind": t.clock_kind,
        "t0_ns": t.t0_ns,
        "input_duration_s": t.input_duration_s,
        "seed": t.seed,
        "deadline_s": t.deadline_s,
        "continuous": t.continuous,
        "error": t.error,
        "start_stream_ms": t.start_stream_ms,
        "segments": [s.__dict__ for s in t.segments],
        "sends_lateness_ms_max": max((s.lateness_ns for s in t.sends), default=0) / 1e6,
        "events": [{k: v for k, v in e.__dict__.items() if k != "pcm"} for e in t.events],
    }


def write_run(res: RunResult, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(
        json.dumps(
            {
                "meta": res.meta,
                "cold_start": res.cold_start,
                "aggregate": res.aggregate,
                "quality": res.quality,
                "playout": res.playout,
                "segment_results": [r.to_dict() for r in res.segment_results],
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    with (out / "traces.jsonl").open("w", encoding="utf-8") as fh:
        for t in res.warmup_traces:
            fh.write(json.dumps({**trace_to_dict(t), "warmup": True}, default=str) + "\n")
        for t in res.traces:
            fh.write(json.dumps({**trace_to_dict(t), "warmup": False}, default=str) + "\n")
