"""Métricas de latencia y lag a partir de ``StreamTrace``.

Tres instantes distintos de "primer audio" (no se mezclan):

* **disponible** (``first_pcm_ns``, titular): ``recv_ns`` del arnés del primer
  chunk TTS con bloque no silencioso. Es cuando el audio existe en el arnés.
* **playout estimado** (``first_pcm_playout_est_ns``): cola de reproducción
  simulada con TODOS los chunks previos (también los silenciosos); inicio del
  chunk = max(recv, fin de lo encolado) + offset no silencioso. Es una
  estimación, no una medición de dispositivo.
* **dispositivo** (``first_device_ns``): solo con eventos ``device_output``
  reales; si no, ``None``/"unavailable".

Orígenes (no intercambiables):

* streaming: ``end_to_first`` desde el fin anotado de habla (``t0 + end``).
* text/offline_file: NO hay fin de habla en tiempo real. Se mide
  ``request_to_first`` desde el inicio de la petición (``t0``) y se etiqueta así.

Clases (ver ``stats``): observed / timeout / failure / anticipated /
unmeasurable. No medible no se convierte en fallo ni en censura.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

from . import stats
from .replay import Segment, StreamTrace, TraceEvent

OUTCOMES = (
    "ok",
    "anticipated",
    "timeout",
    "silent_output",
    "error",
    "no_correspondence",
    "no_boundary",
)
_STATUS = {
    "ok": "observed",
    "timeout": "timeout",
    "silent_output": "failure",
    "error": "failure",
    "anticipated": "anticipated",
    "no_correspondence": "unmeasurable",
    "no_boundary": "unmeasurable",
}


@dataclass
class SegmentResult:
    stream_id: str
    item_id: str
    outcome: str
    adapter_mode: str
    clock_kind: str
    boundary_method: str
    boundary_uncertainty_s: float | None
    end_to_first_ms: float | None  # streaming, solo si outcome == ok
    signed_end_to_first_ms: float | None
    request_to_first_ms: float | None  # text/offline_file, solo si ok
    start_to_first_ms: float | None
    start_origin: str  # utterance_start | stream_t0 | request_start
    start_censor_ms: float | None  # cota de censura para start_to_first
    deadline_ms: float
    first_pcm_ns: int | None  # disponible
    first_pcm_playout_est_ns: int | None  # estimado
    end_to_playout_est_ms: float | None
    first_device_ns: int | None
    n_attributed_chunks: int
    planned_text_en: str | None
    output_kind: str = "audio"  # audio | text (adaptador sin TTS: el "primer output" es mt_final)
    mt_text_en: str | None = None  # concatenación ordenada de mt_final atribuidos
    tags: tuple = ()
    n_commits: int = 0
    planned_text_ambiguous: bool = False  # sin commit_id: no se distinguen frases repetidas seguidas
    asr_text_es: str | None = None  # concatenación ordenada de asr_final atribuidos
    asr_status: str = "none"  # ok | none (sin asr_final) | no_correspondence
    cluster: str | None = None
    speaker_group: str | None = None
    transcript_provenance: str | None = None
    reference_provenance: str | None = None
    detail: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _tts_events(trace: StreamTrace) -> list[TraceEvent]:
    return [e for e in trace.events if e.kind == "tts_chunk" and e.pcm_samples]


def commit_texts(events: Sequence[TraceEvent], *, per_event: bool) -> tuple[list[str], bool]:
    """Textos por commit en orden de llegada, sin duplicar chunks ni quitar
    frases repetidas.

    * Con ``commit_id``: una entrada por commit (los chunks de un commit no se
      duplican; dos commits con el mismo texto cuentan los dos).
    * Sin ``commit_id`` y ``per_event=False`` (chunks TTS): runs consecutivos de
      igual texto se funden en uno. Una frase repetida pegada a otra queda
      indistinguible: se devuelve ``ambiguous=True``.
    * Sin ``commit_id`` y ``per_event=True`` (asr_final/mt_final): cada evento es
      un commit.
    """
    order: list[str] = []
    by_id: dict[str, str] = {}
    out_no_id: list[tuple[int, str]] = []
    ambiguous = False
    last_text: str | None = None
    for i, e in enumerate(sorted(events, key=lambda e: e.recv_ns)):
        txt = (e.planned_text_en if not per_event else e.text) or ""
        if not txt:
            continue
        if e.commit_id is not None:
            if e.commit_id not in by_id:
                order.append(e.commit_id)
            by_id[e.commit_id] = txt if per_event else by_id.get(e.commit_id, txt)
            last_text = None
            continue
        if per_event:
            order.append(f"__ev{i}")
            by_id[f"__ev{i}"] = txt
        else:
            ambiguous = True
            if txt != last_text:
                order.append(f"__run{i}")
                by_id[f"__run{i}"] = txt
            last_text = txt
    return [by_id[k] for k in order], ambiguous


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def playout_schedule(trace: StreamTrace) -> dict[int, tuple[int, int]]:
    """id(evento) -> (inicio_ns, fin_ns) en cola de reproducción simulada."""
    sched: dict[int, tuple[int, int]] = {}
    playhead = None
    for c in sorted(_tts_events(trace), key=lambda e: e.recv_ns):
        dur = round(c.pcm_samples / (c.sample_rate or 16000) * 1e9)
        start = c.recv_ns if playhead is None else max(c.recv_ns, playhead)
        playhead = start + dur
        sched[id(c)] = (start, playhead)
    return sched


def _attribute(
    trace: StreamTrace, seg: Segment, events: list[TraceEvent] | None = None
) -> tuple[list[TraceEvent], str | None]:
    chunks = _tts_events(trace) if events is None else events
    if not trace.continuous:
        return chunks, None
    if not chunks:
        return [], None
    if all(c.source_span_s is None for c in chunks):
        return [], "no_correspondence"
    best: list[TraceEvent] = []
    for c in chunks:
        if c.source_span_s is None:
            continue
        best_seg, best_ov = None, 0.0
        for sg in trace.segments:
            g0 = sg.start_s if sg.start_s is not None else 0.0
            g1 = sg.end_s if sg.end_s is not None else trace.input_duration_s
            ov = _overlap(c.source_span_s[0], c.source_span_s[1], g0, g1)
            if ov > best_ov:
                best_seg, best_ov = sg, ov
        if best_seg is seg:
            best.append(c)
    return best, None


def segment_results(trace: StreamTrace) -> list[SegmentResult]:
    out: list[SegmentResult] = []
    streaming = trace.adapter_mode == "streaming"
    has_error = bool(trace.error) or any(e.kind == "error" for e in trace.events)
    sched = playout_schedule(trace)
    dev = [e for e in trace.events if e.kind == "device_output"]
    deadline_ms = trace.deadline_s * 1000
    for seg in trace.segments:
        chunks, status = _attribute(trace, seg)
        mt_all = [e for e in trace.events if e.kind == "mt_final"]
        mt_ev, _mt_st = _attribute(trace, seg, mt_all)
        mt_texts, _ = commit_texts(mt_ev, per_event=True)
        mt_text = " ".join(mt_texts) if mt_texts else None
        text_only = not trace.produces_audio
        nonsilent = [c for c in chunks if c.first_nonsilent_offset_s is not None]
        first_ns = playout_ns = None
        if text_only:
            # Salida de texto: el primer mt_final con texto es el "primer output".
            texts = [e for e in mt_ev if e.text]
            if texts:
                first_ns = min(e.recv_ns for e in texts)
        elif nonsilent:
            c = min(nonsilent, key=lambda e: e.recv_ns)
            first_ns = c.recv_ns  # titular: disponible
            playout_ns = sched[id(c)][0] + round(c.first_nonsilent_offset_s * 1e9)
        if streaming:
            start_origin = "utterance_start" if seg.start_s is not None else "stream_t0"
            start_abs = trace.abs_ns(seg.start_s) if seg.start_s is not None else trace.t0_ns
            end_abs = trace.abs_ns(seg.end_s) if seg.end_s is not None else None
        else:
            start_origin = "request_start"
            start_abs = trace.t0_ns
            end_abs = trace.t0_ns  # origen = petición, no fin de habla
        s2f = (first_ns - start_abs) / 1e6 if first_ns is not None else None
        rel = (first_ns - end_abs) / 1e6 if (first_ns is not None and end_abs is not None) else None
        rel_play = (playout_ns - end_abs) / 1e6 if (playout_ns is not None and end_abs is not None) else None
        ptexts, p_amb = commit_texts(chunks, per_event=False)
        planned = " ".join(ptexts) if ptexts else None
        asr_all = [e for e in trace.events if e.kind == "asr_final"]
        asr_ev, asr_st = _attribute(trace, seg, asr_all)
        if asr_st == "no_correspondence":
            asr_status, asr_text = "no_correspondence", None
        elif not asr_ev:
            asr_status, asr_text = "none", None
        else:
            atexts, _ = commit_texts(asr_ev, per_event=True)
            # Hubo evento asr_final: aunque el texto sea vacío, el ASR respondió
            # (penaliza como hipótesis vacía). "none" = ningún evento asr_final.
            asr_status, asr_text = "ok", " ".join(atexts)
        if streaming and seg.end_s is not None:
            start_censor = ((seg.end_s - (seg.start_s or 0.0)) * 1000 + deadline_ms) if seg.start_s is not None else (seg.end_s * 1000 + deadline_ms)
        else:
            start_censor = deadline_ms if not streaming else None
        first_dev = min((e.recv_ns for e in dev), default=None) if (dev and not trace.continuous) else None

        detail = None
        if status == "no_correspondence":
            outcome = "no_correspondence"
            detail = "chunks TTS sin source_span_s en stream continuo"
        elif has_error and first_ns is None:
            outcome = "error"
            detail = trace.error or "evento error del candidato"
        elif first_ns is None:
            outcome = "silent_output" if (chunks and not text_only) else "timeout"
        elif end_abs is None:
            outcome = "no_boundary"
            detail = "sin fin de habla anotado; solo start_to_first"
        elif streaming and rel < 0:
            outcome = "anticipated"
            detail = "primer PCM antes del fin anotado; contenido sin verificar"
            u = seg.boundary_uncertainty_s
            if u is not None and -rel <= u * 1000:
                detail += " (dentro de la incertidumbre del límite)"
        elif rel > deadline_ms:
            outcome = "timeout"
            detail = "primer PCM después del deadline"
        else:
            outcome = "ok"
        out.append(
            SegmentResult(
                stream_id=trace.stream_id,
                item_id=seg.item_id,
                outcome=outcome,
                adapter_mode=trace.adapter_mode,
                clock_kind=trace.clock_kind,
                boundary_method=seg.boundary_method,
                boundary_uncertainty_s=seg.boundary_uncertainty_s,
                end_to_first_ms=rel if (outcome == "ok" and streaming) else None,
                signed_end_to_first_ms=rel if streaming else None,
                request_to_first_ms=rel if (outcome == "ok" and not streaming) else None,
                start_to_first_ms=s2f,
                start_origin=start_origin,
                start_censor_ms=start_censor,
                deadline_ms=deadline_ms,
                first_pcm_ns=first_ns,
                first_pcm_playout_est_ns=playout_ns,
                end_to_playout_est_ms=rel_play if outcome == "ok" else None,
                first_device_ns=first_dev,
                n_attributed_chunks=len(chunks),
                planned_text_en=planned,
                output_kind="text" if text_only else "audio",
                mt_text_en=mt_text,
                tags=tuple(seg.tags),
                n_commits=len(ptexts),
                planned_text_ambiguous=p_amb,
                asr_text_es=asr_text,
                asr_status=asr_status,
                cluster=seg.cluster,
                speaker_group=seg.speaker_group,
                transcript_provenance=seg.transcript_provenance,
                reference_provenance=seg.reference_provenance,
                detail=detail,
            )
        )
    return out


# --------------------------------------------------------------------------
# Lag continuo
# --------------------------------------------------------------------------


def playout_analysis(trace: StreamTrace) -> dict:
    """Cola de reproducción simulada.

    * ``backlog``: segundos encolados aún no reproducidos al llegar cada chunk.
      Duración acumulada, NO lag semántico.
    * ``semantic_lag``: solo si todos los chunks traen ``source_span_s``.
    * ``rtf_request``: en offline_file/text, tiempo de pared hasta el último
      evento / duración del audio de entrada (RTF de proceso).
    """
    chunks = sorted(_tts_events(trace), key=lambda e: e.recv_ns)
    res: dict[str, Any] = {"stream_id": trace.stream_id, "n_chunks": len(chunks)}
    # Velocidad: tres cosas distintas, nunca mezcladas.
    # (1) Throughput/RTF offline: SOLO text/offline_file (el arnés no rellena ni
    #     reproduce a ritmo real). En streaming wall/duración-de-replay NO es RTF
    #     de cómputo (el replay dura audio+relleno por construcción).
    if trace.adapter_mode != "streaming" and trace.events:
        last = max(e.recv_ns for e in trace.events)
        wall = (last - trace.t0_ns) / 1e9
        dur_in = trace.input_duration_s
        res["offline_throughput"] = {
            "wall_s": wall,
            "input_s": dur_in,
            "rtf": wall / dur_in if dur_in else None,
            "x_realtime": dur_in / wall if wall > 0 else None,
            "label": "tiempo de petición completa / duración de audio de entrada (incluye ASR+MT+TTS)",
        }
    else:
        res["offline_throughput"] = {
            "status": "unavailable",
            "reason": "streaming: wall/replay_duration no mide cómputo (el replay dura audio+relleno)",
        }
    # (2) RTF de emisión TTS: tiempo entre primer y último chunk recibido /
    #     duración del audio emitido. Depende de si el candidato pacea su salida;
    #     con salida a ritmo real vale ~1 y no mide cómputo.
    tts_sorted = sorted(_tts_events(trace), key=lambda e: e.recv_ns)
    if len(tts_sorted) >= 2:
        span = (tts_sorted[-1].recv_ns - tts_sorted[0].recv_ns) / 1e9
        out_s = sum(e.pcm_samples / (e.sample_rate or 16000) for e in tts_sorted[:-1])
        res["tts_emission_rtf"] = {
            "rtf": span / out_s if out_s > 0 else None,
            "formula": "(recv_último_chunk − recv_primer_chunk) / Σ duración de los chunks salvo el último",
            "label": ("intervalo de recepción de chunks / duración emitida (excluye último chunk). NO es costo de síntesis: en streaming incluye la "
                      "espera de audio de entrada/replay entre chunks (el TTS no puede hablar lo que aún no se dijo), así que un valor > 1 no "
                      "demuestra que el TTS calcule más lento que tiempo real. En modo texto/sin esperas de entrada ~1 si pacea a ritmo real, "
                      "<1 si sintetiza más rápido. La evidencia de espera real son backlog, underrun_s y el lag por segmento."),
            "valid_as_synthesis_cost": trace.adapter_mode == "text" and not trace.continuous,
        }
    else:
        res["tts_emission_rtf"] = {"status": "unavailable", "reason": "menos de 2 chunks TTS"}
    # (3) Backlog de reproducción: ver más abajo.
    if not chunks:
        res["backlog"] = None
        res["semantic_lag"] = {"status": "unavailable", "reason": "sin salida"}
        return res
    sched = playout_schedule(trace)
    backlogs, starts, underrun_s, total_out = [], [], 0.0, 0.0
    prev_end = None
    for c in chunks:
        s, e = sched[id(c)]
        backlogs.append(max(0.0, ((prev_end or c.recv_ns) - c.recv_ns) / 1e9) if prev_end else 0.0)
        if prev_end is not None and c.recv_ns > prev_end:
            underrun_s += (c.recv_ns - prev_end) / 1e9
        starts.append(s)
        total_out += (e - s) / 1e9
        prev_end = e
    res["total_output_s"] = total_out
    res["backlog"] = {
        "label": "backlog de reproducción (duración acumulada); NO es lag semántico",
        "max_s": max(backlogs),
        "mean_s": sum(backlogs) / len(backlogs),
        "underrun_s": underrun_s,
    }
    if all(c.source_span_s is not None for c in chunks):
        lags = [(st - trace.abs_ns(c.source_span_s[1])) / 1e9 for st, c in zip(starts, chunks)]
        # Granularidad: ``source_span_s`` lo declara el adaptador por SEGMENTO/chunk (no hay
        # alineación palabra a palabra ni se verifica contenido); si el span es interpolado,
        # el valor hereda esa imprecisión. Lag < 0 = salida antes del fin del span fuente:
        # anticipación NO verificada (puede ser span impreciso o traducción adelantada).
        n_early = sum(1 for x in lags if x < 0)
        # Pendiente = crecimiento del lag a lo largo de la CONVERSACIÓN: un punto por span fuente
        # distinto (lag del primer chunk de cada segmento) frente al fin de ese span en el reloj
        # de la fuente. Pendiente entre chunks de un mismo segmento (mismo span, reloj de
        # recepción) mide la velocidad de emisión del TTS, no deriva de conversación: se excluye.
        first_by_span: dict[tuple[float, float], tuple[float, float]] = {}
        for c, lag in zip(chunks, lags):
            first_by_span.setdefault(tuple(c.source_span_s), (c.source_span_s[1], lag))
        pts = sorted(first_by_span.values())
        if len(pts) >= TREND_MIN_N:
            trend, trend_note = _slope([x for x, _ in pts], [y for _, y in pts]), None
        else:
            trend = None
            trend_note = (f"{len(pts)} span(s) fuente distinto(s) (< {TREND_MIN_N}): pendiente no calculada; "
                          "la pendiente intra-segmento no es crecimiento de conversación ni lag por palabra")
        res["semantic_lag"] = {
            "status": "available",
            "granularity": "por segmento (span fuente declarado por el adaptador); NO alineación palabra a palabra",
            "n": len(lags),
            "min_s": min(lags),
            "max_s": max(lags),
            "mean_s": sum(lags) / len(lags),
            "n_negative_unverified_anticipation": n_early,
            "n_distinct_source_spans": len(pts),
            "trend_s_per_s": trend,  # lag del 1.er chunk de cada segmento vs fin del segmento (reloj fuente)
            "trend_basis": "primer chunk por span fuente; eje x = fin del span en el reloj de la fuente",
            "trend_note": trend_note,
            "trend_min_n": TREND_MIN_N,
        }
    else:
        res["semantic_lag"] = {
            "status": "unavailable",
            "reason": "chunks TTS sin source_span_s (sin alineación fuente/salida)",
        }
    return res


TREND_MIN_N = 5  # con menos segmentos distintos la pendiente es ruido (p. ej. n=3 daba ~23 s/s)


def _slope(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    return None if den == 0 else sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


def send_jitter(trace: StreamTrace) -> dict:
    if not trace.sends:
        return {"n": 0}
    late = sorted(s.lateness_ns / 1e6 for s in trace.sends)
    return {
        "n": len(late),
        "lateness_ms_p50": stats.nearest_rank(late, 0.5),
        "lateness_ms_p99": stats.nearest_rank(late, 0.99),
        "lateness_ms_max": late[-1],
    }


# --------------------------------------------------------------------------
# Agregación
# --------------------------------------------------------------------------


def _obs(r: SegmentResult, metric: str) -> stats.Obs:
    cl = r.cluster
    st = _STATUS[r.outcome]
    if metric in ("end_to_first_ms", "request_to_first_ms"):
        if st == "observed":
            return stats.Obs("observed", getattr(r, metric), cl)
        if st == "timeout":
            return stats.Obs("timeout", r.deadline_ms, cl)
        return stats.Obs(st, None, cl)
    if metric == "end_to_playout_est_ms":
        if st == "observed":
            return stats.Obs("observed", r.end_to_playout_est_ms, cl)
        if st == "timeout":
            return stats.Obs("timeout", r.deadline_ms, cl)
        return stats.Obs(st, None, cl)
    # start_to_first_ms: observado si hay primer PCM atribuido
    if r.outcome == "no_correspondence":
        return stats.Obs("unmeasurable", None, cl)
    if r.start_to_first_ms is not None:
        return stats.Obs("observed", r.start_to_first_ms, cl)
    if r.outcome == "timeout" and r.start_censor_ms is not None:
        return stats.Obs("timeout", r.start_censor_ms, cl)
    if r.outcome in ("silent_output", "error"):
        return stats.Obs("failure", None, cl)
    return stats.Obs("unmeasurable", None, cl)


def aggregate(
    results: Sequence[SegmentResult],
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """Agrega por (modo, reloj, precisión de límite) sin mezclar."""
    groups: dict[tuple, list[SegmentResult]] = {}
    for r in results:
        precise = r.boundary_method in ("human_annotation", "forced_alignment_reviewed")
        key = (r.adapter_mode, r.clock_kind, "precise" if precise else "imprecise_or_unknown", r.output_kind)
        groups.setdefault(key, []).append(r)
    out = []
    for (mode, ck, prec, okind), rr in groups.items():
        n = len(rr)
        counts = {o: sum(1 for r in rr if r.outcome == o) for o in OUTCOMES}
        primary = "end_to_first_ms" if mode == "streaming" else "request_to_first_ms"
        sm = lambda metric: stats.summarize_latency(  # noqa: E731
            [_obs(r, metric) for r in rr], n_boot=n_boot, seed=seed
        )
        ant = sorted(r.signed_end_to_first_ms for r in rr if r.outcome == "anticipated")
        out.append(
            {
                "adapter_mode": mode,
                "clock_kind": ck,
                "boundary_precision": prec,
                "output_kind": okind,
                "infra_note": "reloj virtual: SOLO infraestructura, no es medición de hardware" if ck == "virtual" else None,
                "n_segments": n,
                "outcome_counts": counts,
                "primary_metric": primary,
                "primary_origin": (
                    "fin de habla anotado" if mode == "streaming"
                    else "inicio de la petición (sin fin de habla en tiempo real); primer output = "
                    + ("primer mt_final (SALIDA DE TEXTO, sin audio)" if okind == "text" else "primer PCM no silencioso")
                ),
                primary: sm(primary),
                "end_to_playout_est_ms": sm("end_to_playout_est_ms") if mode == "streaming" else None,
                "start_to_first_ms": sm("start_to_first_ms"),
                "device_output": (
                    {"status": "available", "n": sum(1 for r in rr if r.first_device_ns is not None)}
                    if any(r.first_device_ns is not None for r in rr)
                    else {"status": "unavailable", "reason": "sin eventos device_output"}
                ),
                "anticipation": {
                    "count": len(ant),
                    "rate": len(ant) / n,
                    "wilson95_assumes_independence": stats.wilson_interval(len(ant), n),
                    "signed_end_to_first_ms": ant,
                    "note": "contenido sin verificar; no se penaliza ni se premia en la distribución observada",
                },
            }
        )
    return {"groups": out}
