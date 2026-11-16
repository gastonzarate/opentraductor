"""Métricas diagnósticas de traducción incremental (v2). Sin modelos, sin alineaciones inventadas.

Principios:
* Todo ``None`` es DESCONOCIDO y se propaga como desconocido; falta de correspondencia, ``n_evaluable=0`` o alineación incoherente
  dan ``unavailable``, nunca cero (excepto la cobertura con evaluación NEGATIVA completa del ledger: 0 con omisiones verificadas).
* Los IDs/spans/cierres que declara el candidato son telemetría o claims; no sustentan integridad, cobertura ni desfase.
* El tiempo audible de una realización sale de la reproducción EXACTA del chunk del arnés (inicio de reproducción con huecos) más el
  offset de muestras del ``OutputSpan``; no del «primer onset de la emisión».
* Categorías separadas: transporte (entregas del arnés), generado-antes-del-soporte (recepción), audible-antes-del-soporte
  (reproducción), revisión audible, y cola física (hueco entre chunks ≠ underrun demostrado).
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any

from .contracts import AtomRealization, ChunkRecord, IncrementalRun

NS = 1e9
EPS_S = 1e-6  # ruido numérico de punto flotante (sub-microsegundo); NO es una holgura de transporte
MIN_POINTS_SLOPE = 5
EVALUABLE = ("realized", "partial", "contradicted")


@dataclass(frozen=True)
class Params:
    temporal_support_fraction_heuristic: float = 0.8  # fracción de DURACIÓN fuente: heurística temporal, no cobertura semántica
    during_speech_margin_s: float = 1.0
    claim_iou_min: float = 0.5


# ------------------------------------------------------------------ línea de tiempo de salida (arnés)
def _ordered(run: IncrementalRun) -> list[ChunkRecord]:
    ids = [c.chunk_id for c in run.chunks]
    if any(i < 0 for i in ids) or len(ids) != len(set(ids)):
        raise ValueError("chunk_id ausente, negativo o duplicado: lo asigna el arnés y debe ser único")
    return sorted(run.chunks, key=lambda c: (c.stamp.recv_ns, c.stamp.event_seq, c.chunk_id))


def _play(run: IncrementalRun) -> dict[int, dict[str, Any]]:
    """Reproducción reconstruida de UN canal: inicio = max(recv, fin del anterior); incluye huecos entre chunks."""
    out: dict[int, dict[str, Any]] = {}
    playhead = None
    for c in _ordered(run):
        dur = round(c.pcm_samples / (c.sample_rate or 16000) * NS)
        start = c.stamp.recv_ns if playhead is None else max(c.stamp.recv_ns, playhead)
        gap = 0 if playhead is None else max(0, c.stamp.recv_ns - playhead)
        playhead = start + dur
        onset = start + round(c.nonsilent_offset_s * NS) if c.nonsilent_offset_s is not None else None
        out[c.chunk_id] = {"c": c, "start": start, "end": playhead, "dur": dur, "onset": onset, "gap_before": gap}
    return out


def _rel(run: IncrementalRun, ns: int) -> float:
    return (ns - run.t0_ns) / NS


def _slope(points: list[tuple[float, float]]) -> float | None:
    if len(points) < MIN_POINTS_SLOPE:
        return None
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    den = sum((x - mx) ** 2 for x in xs)
    return None if den == 0 else sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


def _add(*xs: float | None) -> float | None:
    return None if any(x is None for x in xs) else float(sum(xs))  # type: ignore[arg-type]


def _union_len(iv: list[tuple[float, float]]) -> float:
    total, cs, ce = 0.0, None, None
    for a, b in sorted(iv):
        if ce is None or a > ce:
            if ce is not None:
                total += ce - cs
            cs, ce = a, b
        else:
            ce = max(ce, b)
    return total + ((ce - cs) if ce is not None else 0.0)


# ------------------------------------------------------------------ alineación del evaluador
def _alignment_state(run: IncrementalRun) -> tuple[Any, str | None]:
    a = run.alignment
    if a is None:
        return None, "sin correspondencia independiente (alignment=None)"
    prob = a.validate(run.ref_units)
    if prob:
        return None, "alineación incoherente: " + "; ".join(prob[:2])
    return a, None


def _resolve(run: IncrementalRun, plays: dict[int, dict], r: AtomRealization) -> dict[str, Any]:
    """Tiempo audible EXACTO de la realización. Valida el span contra el chunk del arnés (0 ≤ inicio < fin ≤ muestras).
    El audio solo empieza en max(inicio del span, primer audio no silencioso del chunk): el silencio inicial no es voz."""
    if r.span is None or r.span.chunk_id not in plays:
        return {"r": r, "invalid": "span ausente o chunk_id inexistente"}
    p = plays[r.span.chunk_id]
    c: ChunkRecord = p["c"]
    sr = c.sample_rate or 16000
    if not (0 <= r.span.sample_start < r.span.sample_end <= c.pcm_samples):
        return {"r": r, "invalid": f"span {r.span.sample_start}:{r.span.sample_end} fuera de [0,{c.pcm_samples}] del chunk {c.chunk_id}"}
    off = c.nonsilent_offset_s
    audible = off is not None and (r.span.sample_end / sr) > off
    start_s = max(r.span.sample_start / sr, off if off is not None else 0.0)
    return {"r": r, "start_ns": p["start"] + round(start_s * NS), "end_ns": p["start"] + round(r.span.sample_end / sr * NS),
            "span_start_ns": p["start"] + round(r.span.sample_start / sr * NS), "audible": audible, "chunk": c}


def _evaluable(run: IncrementalRun, plays: dict[int, dict]) -> tuple[list[dict], list[dict]]:
    """(realizaciones evaluables y resueltas, inválidas/no resueltas con su motivo)."""
    rows, bad = [], []
    for r in run.alignment.realizations:  # type: ignore[union-attr]
        if r.status not in EVALUABLE:
            continue
        x = _resolve(run, plays, r)
        (bad if "invalid" in x else rows).append(x)
    return rows, bad


# ------------------------------------------------------------------ métricas
def accounting(run: IncrementalRun) -> dict[str, Any]:
    n = run.n_streams
    return {"n_streams": n, "n_failed": run.n_failed, "n_timeout_censored": run.n_timeout, "n_observed": n - run.n_failed - run.n_timeout,
            "failure_rate": (run.n_failed / n) if n else None, "timeout_rate": (run.n_timeout / n) if n else None,
            "note": "fallos y timeouts quedan en el denominador; las demás métricas describen solo los streams observados"}


def first_onset(run: IncrementalRun, p: Params = Params()) -> dict[str, Any]:
    on = [v["onset"] for v in _play(run).values() if v["onset"] is not None]
    if not on:
        return {"first_onset_s": None, "reason": "ninguna emisión audible"}
    t_ns = min(on)
    t = _rel(run, t_ns)
    out: dict[str, Any] = {"first_onset_s": t, "before_recording_end": t < run.input_duration_s,
                           "during_speech_margin_s": p.during_speech_margin_s}
    iv = run.source_speech_intervals
    if iv is None:
        out.update(source_active_at_onset=None, future_speech_remaining_s=None, during_speech=None,
                   speech_note="sin source_speech_intervals (actividad de habla fuente): no se infiere desde las unidades de referencia")
    else:
        active = any(a <= t <= b for a, b in iv)
        remaining = sum(max(0.0, b - max(a, t)) for a, b in iv)
        out.update(source_active_at_onset=active, future_speech_remaining_s=remaining,
                   during_speech=bool(active and remaining >= p.during_speech_margin_s))
    out["note"] = "primer onset audible de CUALQUIER emisión, sin umbral de tamaño; no implica utilidad ni cobertura"
    return out


def physical_queue(run: IncrementalRun) -> dict[str, Any]:
    """Cola FÍSICA. Un hueco entre chunks NO es un underrun: puede ser una pausa legítima."""
    plays = _play(run)
    if not plays:
        return {"status": "unavailable", "reason": "sin chunks"}
    order = sorted(plays.values(), key=lambda v: (v["c"].stamp.recv_ns, v["c"].stamp.event_seq, v["c"].chunk_id))
    pts, prev_end = [], None
    for v in order:
        recv = v["c"].stamp.recv_ns
        ahead = max(0, (prev_end if prev_end is not None else recv) - recv)
        pts.append((_rel(run, recv), (ahead + v["dur"]) / NS))
        prev_end = v["end"]
    gaps = [v["gap_before"] / NS for v in order if v["gap_before"] > 0]
    bl = [b for _, b in pts]
    speeds = [v["c"].cand.speed for v in order if v["c"].cand.speed is not None]
    ready = run.unit_ready
    return {
        "max_s": max(bl), "mean_s": statistics.fmean(bl), "growth_s_per_s": _slope(pts),
        "growth_note": f"pendiente solo con ≥{MIN_POINTS_SLOPE} chunks; cola física, no desfase semántico",
        "interchunk_gap_s": sum(gaps), "n_interchunk_gaps": len(gaps), "max_interchunk_gap_s": max(gaps) if gaps else 0.0,
        "underrun_demonstrated_s": None,
        "underrun_note": ("None: un hueco entre chunks (incluida una pausa legítima) no demuestra underrun; hace falta evidencia del Player/arnés "
                          "con pacing explícito. unit_ready es telemetría del candidato y no lo prueba"),
        "unit_ready_events": None if ready is None else len(ready),
        "speed_factor_mean": statistics.fmean(speeds) if speeds else None, "speed_factor_max": max(speeds) if speeds else None,
        "n_chunks": len(order),
    }


def _delivered_at(run: IncrementalRun, t_ns: int) -> float | None:
    if run.input_deliveries is None:
        return None
    ends = [d.end_s for d in run.input_deliveries if d.actual_ns <= t_ns]
    return max(ends) if ends else 0.0


def transport_frontier(run: IncrementalRun, p: Params = Params()) -> dict[str, Any]:
    """Violación de TRANSPORTE: el arnés entregó más audio que el reloj. Entregas independientes de los chunks de salida."""
    tol = 0.0
    out: dict[str, Any] = {"tolerance_s": tol, "tolerance_basis": ("estricto: se usa end_s exacto y el reloj monótono del replay; solo se absorbe 1 µs de ruido numérico. "
                                                                   "La incertidumbre física medida es una fase futura")}
    if run.input_deliveries is None:
        out["deliveries"] = {"status": "unavailable", "reason": "sin línea de tiempo de entrega (input_deliveries)"}
    else:
        d = sorted(run.input_deliveries, key=lambda x: x.actual_ns)
        viol = [{"seq": x.seq, "end_s": x.end_s, "elapsed_s": _rel(run, x.actual_ns)} for x in d if x.end_s > _rel(run, x.actual_ns) + tol + EPS_S]
        nonmono = sum(1 for a, b in zip(d, d[1:]) if b.end_s < a.end_s - 1e-9 or b.seq < a.seq)
        out["deliveries"] = {"status": "ok", "n": len(d), "violations": len(viol), "examples": viol[:3], "non_monotonic": nonmono}
    chunk_v = [c.chunk_id for c in run.chunks if c.stamp.delivered_until_s > _rel(run, c.stamp.recv_ns) + tol + EPS_S]
    out["chunk_stamps"] = {"n_chunks": len(run.chunks), "violations": len(chunk_v), "chunk_ids": chunk_v[:3]}
    out["note"] = "fuga de TRANSPORTE del arnés (entregó futuro). NO es anticipación del contenido ni prueba acceso del candidato"
    return out


def _lag_rows(run: IncrementalRun, rows: list[dict]) -> list[dict[str, Any]]:
    units = {u.ref_unit_id: u for u in run.ref_units}
    out = []
    for x in rows:
        r: AtomRealization = x["r"]
        u = units.get(r.ref_unit_id)
        t = _rel(run, x["start_ns"])
        unc_support = _add(r.out_uncertainty_s, r.support_uncertainty_s)
        lag_s = (t - r.support_end_s) if r.support_end_s is not None else None
        lag_u = (t - u.src_end_s) if u is not None else None
        unc_u = _add(r.out_uncertainty_s, u.boundary_uncertainty_s) if u is not None else None
        out.append({
            "start_ns": x["start_ns"], "realization_id": r.realization_id, "atom_id": r.atom_id, "ref_unit_id": r.ref_unit_id, "audible": x["audible"], "status": r.status,
            "audible_start_s": t, "support_end_s": r.support_end_s, "lag_to_support_end_s": lag_s, "lag_to_support_end_uncertainty_s": unc_support if lag_s is not None else None,
            "lag_to_unit_end_s": lag_u, "lag_to_unit_end_uncertainty_s": unc_u,
            "negative_beyond_uncertainty": ((lag_s + unc_support < 0) if (lag_s is not None and unc_support is not None) else None),
        })
    return out


def semantic_lag(run: IncrementalRun) -> dict[str, Any]:
    """Titular SOLO con realizaciones `realized` y audibles. `partial`/`contradicted` se muestran aparte y nunca mejoran el titular.
    La pendiente se mide contra el tiempo FUENTE (support_end, o fin de unidad), no contra audible_start."""
    a, why = _alignment_state(run)
    if a is None:
        return {"status": "unavailable", "reason": why}
    rows, invalid = _evaluable(run, _play(run))
    aud = [x for x in rows if x["audible"]]
    if not aud:
        return {"status": "unavailable", "reason": "n_evaluable=0: ninguna realización evaluable y audible", "n_invalid_spans": len(invalid)}
    lr = _lag_rows(run, aud)
    units = {u.ref_unit_id: u for u in run.ref_units}
    realized_all = sorted((r for r in lr if r["status"] == "realized"), key=lambda r: (r["start_ns"], r["realization_id"]))
    head, reemitted, seen_atoms = [], [], set()
    for r in realized_all:  # UNA observación por (ref_unit_id, atom_id): la primera realización correcta audible; las reemisiones van aparte
        k = (r["ref_unit_id"], r["atom_id"])
        (reemitted if k in seen_atoms else head).append(r)
        seen_atoms.add(k)
    other = [r for r in lr if r["status"] != "realized"]
    sup_pts = [(r["support_end_s"], r["lag_to_support_end_s"]) for r in head if r["lag_to_support_end_s"] is not None]
    unit_pts = [(units[r["ref_unit_id"]].src_end_s, r["lag_to_unit_end_s"]) for r in head if r["lag_to_unit_end_s"] is not None and r["ref_unit_id"] in units]
    if not head:
        return {"status": "unavailable", "reason": "ninguna realización `realized` audible para el titular", "n_partial": sum(r["status"] == "partial" for r in other),
                "n_contradicted": sum(r["status"] == "contradicted" for r in other), "n_invalid_spans": len(invalid)}
    return {
        "status": "ok", "tier": a.tier, "n_realizations": len(head), "n_atoms_in_headline": len(head),
        "n_realized_realizations_total": len(realized_all), "n_reemitted_excluded_from_headline": len(reemitted),
        "reemitted_realizations": reemitted, "headline_basis": "una observación por (ref_unit_id, atom_id): primera realización correcta audible",
        "n_support_end_known": len(sup_pts), "n_support_end_missing": len(head) - len(sup_pts),
        "median_lag_to_support_end_s": statistics.median([y for _, y in sup_pts]) if sup_pts else None,
        "median_lag_to_unit_end_s": statistics.median([y for _, y in unit_pts]) if unit_pts else None,
        "slope_support_end_s_per_s": _slope(sup_pts), "slope_unit_end_s_per_s": _slope(unit_pts),
        "slope_basis": "lag frente al tiempo FUENTE (support_end / fin de unidad), no frente a audible_start; ≥5 puntos",
        "n_uncertainty_unknown": sum(1 for r in head if r["lag_to_support_end_uncertainty_s"] is None),
        "n_partial_excluded_from_headline": sum(r["status"] == "partial" for r in other),
        "n_contradicted_excluded_from_headline": sum(r["status"] == "contradicted" for r in other),
        "other_status_realizations": other, "n_invalid_spans": len(invalid),
        "invalid_spans": [{"realization_id": u["r"].realization_id, "motivo": u["invalid"]} for u in invalid[:3]], "realizations": head,
        "note": ("principal: audible_start − support_end (disponibilidad causal del soporte); secundario: contra el fin de la unidad. "
                 "Incertidumbre None = desconocida: no se clasifica el signo. partial/contradicted no mejoran el titular"),
    }


def anticipation(run: IncrementalRun, p: Params = Params()) -> dict[str, Any]:
    """generated_before_support (recepción) y audible_before_support (reproducción) por separado. Ninguna prueba fuga.
    La clasificación audible usa el intervalo TEMPORAL INCIERTO [start − out_unc, start + out_unc]: «antes» solo si el soporte no
    había sido entregado ni siquiera al final del intervalo; «no antes» solo si ya estaba entregado al comienzo; si la frontera cambia
    dentro del intervalo es «ambigua»; sin incertidumbre de salida conocida no se clasifica."""
    a, why = _alignment_state(run)
    if a is None:
        return {"status": "unavailable", "reason": why}
    plays = _play(run)
    rows, bad = _evaluable(run, plays)
    if not rows:
        return {"status": "unavailable", "reason": "n_evaluable=0: ninguna realización evaluable y resuelta", "n_invalid_spans": len(bad)}
    gen, unclass = [], 0
    aud_before, aud_not, aud_amb, aud_unclass = [], 0, [], 0
    for x in rows:
        r: AtomRealization = x["r"]
        if r.support_end_s is None or r.support_uncertainty_s is None:
            unclass += 1
            continue
        if r.support_end_s > x["chunk"].stamp.delivered_until_s + r.support_uncertainty_s:
            gen.append(r.realization_id)
        if not x["audible"]:
            continue
        if run.input_deliveries is None:
            continue
        if r.out_uncertainty_s is None:
            aud_unclass += 1
            continue
        u_ns = round(r.out_uncertainty_s * NS)
        d_early, d_late = _delivered_at(run, x["start_ns"] - u_ns), _delivered_at(run, x["start_ns"] + u_ns)
        need = r.support_end_s - r.support_uncertainty_s  # soporte necesario en el caso más favorable
        need_hi = r.support_end_s + r.support_uncertainty_s
        if d_late is not None and d_late + 0.0 < need:
            aud_before.append(r.realization_id)  # ni siquiera al final del intervalo estaba entregado
        elif d_early is not None and d_early >= need_hi:
            aud_not += 1
        else:
            aud_amb.append(r.realization_id)
    has_t = run.input_deliveries is not None
    return {
        "status": "ok", "n_classified": len(rows) - unclass, "n_unclassified_unknown_support_or_uncertainty": unclass,
        "n_invalid_spans": len(bad),
        "generated_before_support": {"n": len(gen), "ids": gen[:5], "note": "PCM generado/recibido antes del soporte: preparación privada, NO es pronunciación anticipada"},
        "audible_before_support": ({"n": len(aud_before), "ids": aud_before[:5], "n_not_before": aud_not, "n_ambiguous": len(aud_amb), "ids_ambiguous": aud_amb[:5],
                                    "n_unclassified_unknown_output_uncertainty": aud_unclass,
                                    "note": "se OYÓ antes de que el soporte llegara según la entrega de entrada, considerando la incertidumbre temporal de salida; anticipación no sustentada, no prueba fuga"}
                                   if has_t else {"status": "unavailable", "reason": "sin línea de tiempo de entrega (input_deliveries)"}),
    }


def coverage_and_integrity(run: IncrementalRun, p: Params = Params()) -> dict[str, Any]:
    """Cobertura semántica por átomos con LEDGER de evaluación explícito (positiva y negativa).
    Una evaluación completa con 0 realizaciones correctas es cobertura 0 con omisiones verificadas; sin ninguna evaluación es unavailable."""
    a, why = _alignment_state(run)
    if a is None:
        return {"status": "unavailable", "reason": why}
    if not run.ref_units or not any(u.atoms for u in run.ref_units):
        return {"status": "unavailable", "reason": "referencia desconocida: sin unidades/átomos de referencia definidos por el evaluador"}
    if not any(x.status != "not_assessed" for x in a.assessments) and not a.realizations:
        return {"status": "unavailable", "reason": "sin evaluación: ni assessments ni realizaciones del evaluador"}
    plays = _play(run)
    rows, unresolved = _evaluable(run, plays)
    invalid_atoms = {(x["r"].ref_unit_id, x["r"].atom_id) for x in unresolved}
    assess = {(x.ref_unit_id, x.atom_id): x.status for x in a.assessments}
    by_atom: dict[tuple[str, str], list[dict]] = {}
    for x in rows:
        by_atom.setdefault((x["r"].ref_unit_id, x["r"].atom_id), []).append(x)
    complete = a.completeness == "complete" and not invalid_atoms  # spans inválidos hacen no atribuibles esos átomos
    total = realized = 0
    atom_state: dict[tuple[str, str], str] = {}
    atom_time: dict[tuple[str, str], int] = {}
    atom_end: dict[tuple[str, str], int] = {}
    reps, contradicted, conflicts, heard = [], [], [], {}
    for u in run.ref_units:
        for at in u.atoms:
            k = (u.ref_unit_id, at.atom_id)
            total += 1
            xs = by_atom.get(k, [])
            real = [x for x in xs if x["r"].status == "realized" and x["audible"]]
            part = [x for x in xs if x["r"].status == "partial" and x["audible"]]
            contra = [x for x in xs if x["r"].status == "contradicted" and x["audible"]]
            distinct = {x["r"].realization_id for x in real + part}
            if len(distinct) >= 2:
                reps.append({"ref_unit_id": u.ref_unit_id, "atom_id": at.atom_id, "n_audible_realizations": len(distinct)})
            if contra:
                heard[k] = at.critical  # se OYÓ una contradicción: el historial no se borra aunque luego se corrija
            ast = assess.get(k)
            if ast is None:  # sin veredicto explícito: se infiere SOLO de las realizaciones audibles; sin ellas no se observó
                ast = "realized" if real else "contradicted" if contra else "not_assessed"
            if k in invalid_atoms and not real:
                state = "unattributable"
            elif ast == "realized":
                state = "realized" if real else "unattributable"  # veredicto «realizado» sin evidencia audible resoluble
            elif ast == "omitted":
                state = "omitted" if not (real or contra) else "unattributable"  # omitido pero hay una realización audible: conflicto
                if real or contra:
                    conflicts.append(f"{k[0]}:{k[1]}")
            elif ast == "contradicted":
                if real and contra and min(x["start_ns"] for x in real) > min(x["start_ns"] for x in contra):
                    state = "realized"  # corregida DESPUÉS de la contradicción: cobertura eventual, historial conservado
                else:
                    state = "contradicted"
                    if real:
                        conflicts.append(f"{k[0]}:{k[1]}")  # el ledger dice contradicho pero solo se oyó la correcta (o antes de la contradicción)
            elif ast == "ambiguous":
                state = "unattributable"
                if real or contra:
                    conflicts.append(f"{k[0]}:{k[1]}")
            else:
                state = "not_observed"
            atom_state[k] = state
            if state == "realized":
                atom_time[k] = min(x["start_ns"] for x in real)
                atom_end[k] = min(x["end_ns"] for x in real)
                realized += 1
            if state == "contradicted":
                contradicted.append(k)
    units_out, n_complete, n_crit_incomplete = [], 0, 0
    for u in run.ref_units:
        req = [at for at in u.atoms if at.required]
        st = {at.atom_id: atom_state[(u.ref_unit_id, at.atom_id)] for at in u.atoms}
        is_complete = bool(req) and all(st[at.atom_id] == "realized" for at in req)
        crit_bad = [at.atom_id for at in u.atoms if at.critical and st[at.atom_id] != "realized"]
        complete_at = _rel(run, max(atom_end[(u.ref_unit_id, at.atom_id)] for at in req)) if is_complete else None  # cuando termina de oírse el último requerido
        n_complete += is_complete
        n_crit_incomplete += bool(crit_bad)
        units_out.append({"ref_unit_id": u.ref_unit_id, "unit_complete": is_complete, "unit_complete_at_s": complete_at,
                          "critically_incomplete": bool(crit_bad), "critical_atoms_not_realized": crit_bad, "atoms": st,
                          "had_contradiction": any((u.ref_unit_id, at.atom_id) in heard for at in u.atoms),
                          "critical_contradiction_heard": any(heard.get((u.ref_unit_id, at.atom_id)) for at in u.atoms)})
    first = min(atom_time.values()) if atom_time else None
    heur = []
    for u in run.ref_units:
        length = max(u.src_end_s - u.src_start_s, 1e-9)
        items = sorted(((x["start_ns"], x["r"]) for x in rows if x["r"].ref_unit_id == u.ref_unit_id and x["r"].status == "realized" and x["audible"]
                        and x["r"].support_start_s is not None and x["r"].support_end_s is not None), key=lambda t: t[0])
        acc, cross = [], None
        for t_ns, r in items:
            acc.append((max(r.support_start_s, u.src_start_s), min(r.support_end_s, u.src_end_s)))
            frac = _union_len([i for i in acc if i[1] > i[0]]) / length
            if cross is None and frac >= p.temporal_support_fraction_heuristic:
                cross = _rel(run, t_ns)
        heur.append({"ref_unit_id": u.ref_unit_id, "fraction": (_union_len([i for i in acc if i[1] > i[0]]) / length) if acc else None, "crossing_time_s": cross})
    frac_atoms = realized / total if total else None
    not_assessed = [f"{k[0]}:{k[1]}" for k, v in atom_state.items() if v == "not_observed"]
    return {
        "status": "ok", "tier": a.tier, "completeness": a.completeness, "complete_effective": complete, "n_atoms": total, "n_atoms_realized": realized,
        "semantic_atom_coverage": frac_atoms if complete else None,
        "semantic_atom_coverage_lower_bound": None if complete else frac_atoms,
        "coverage_note": ("cobertura SEMÁNTICA por átomos con ledger de evaluación; con completeness=complete incluye el caso 0 (evaluación negativa explícita); "
                          "en otro caso es una cota inferior etiquetada y los átomos no evaluados NO son omisiones"),
        "first_useful_atom_s": _rel(run, first) if first is not None else None,
        "n_units": len(run.ref_units), "n_units_complete": n_complete, "n_units_critically_incomplete": n_crit_incomplete, "units": units_out,
        "repeated_atoms": reps, "n_repeated_atoms": len(reps), "n_contradicted_atoms": len(contradicted),
        "n_atoms_contradiction_heard": len(heard), "contradiction_heard_atoms": [f"{k[0]}:{k[1]}" for k in heard],
        "n_critical_contradiction_heard": sum(1 for v in heard.values() if v),
        "contradiction_note": "historial: cualquier contradicción AUDIBLE cuenta aunque luego se corrija; la cobertura es la eventual y el estado final no la borra",
        "omitted_atoms": [f"{k[0]}:{k[1]}" for k, v in atom_state.items() if v == "omitted"],  # solo omisiones VERIFICADAS por el ledger
        "not_assessed_atoms": not_assessed, "n_not_assessed": len(not_assessed), "ledger_conflicts": conflicts,
        "n_unattributable": sum(1 for v in atom_state.values() if v == "unattributable"),
        "n_invalid_spans": len(unresolved), "invalid_spans": [{"realization_id": u["r"].realization_id, "motivo": u["invalid"]} for u in unresolved[:3]],
        "temporal_support_fraction_heuristic": {"threshold": p.temporal_support_fraction_heuristic, "units": heur,
                                                "note": "HEURÍSTICA TEMPORAL de duración fuente; NO es cobertura, omisión, utilidad ni integridad"},
    }


def audible_revisions(run: IncrementalRun) -> dict[str, Any]:
    """Revisión de audio YA pronunciado, verificable solo con el arnés (la sustituta llegó cuando lo sustituido ya se oía)."""
    plays = _play(run)
    onset: dict[tuple[str, int], int] = {}
    first_recv: dict[tuple[str, int], int] = {}
    sup: dict[tuple[str, int], str] = {}
    for v in plays.values():
        c = v["c"]
        k = (c.cand.emission_id, c.cand.revision)
        first_recv[k] = min(first_recv.get(k, c.stamp.recv_ns), c.stamp.recv_ns)
        if v["onset"] is not None:
            onset[k] = min(onset.get(k, v["onset"]), v["onset"])
        if c.cand.supersedes:
            sup[k] = c.cand.supersedes
    spoken, silent, malformed = [], 0, []
    for k, s in sup.items():
        eid, sep, rev = s.rpartition("@")
        if not sep or not eid or not rev.isascii() or not rev.isdigit():  # gramática exigida: `emission_id@revision` (revision entera)
            malformed.append({"by": f"{k[0]}@{k[1]}", "supersedes": s})
            continue
        rv = int(rev)
        on = onset.get((eid, rv))
        if on is not None and on <= first_recv[k]:
            spoken.append({"superseded": s, "by": f"{k[0]}@{k[1]}", "heard_before_replacement_s": (first_recv[k] - on) / NS})
        else:
            silent += 1
    return {"spoken_then_revised": len(spoken), "examples": spoken[:3], "revised_before_audible": silent,
            "n_malformed_supersedes": len(malformed), "malformed_supersedes": malformed[:3],
            "note": "lo ya pronunciado no se puede des-decir; basado en `supersedes` (telemetría) contrastado con la reproducción del arnés"}


def unit_closure(run: IncrementalRun) -> dict[str, Any]:
    """Cierre/cancelación declarados, con ESTAMPA y orden. Un desajuste es una inconsistencia DECLARATIVA del candidato:
    no se atribuye automáticamente a pérdida/duplicación de transporte."""
    chunks = _ordered(run)
    by_key: dict[tuple[str, int], list[ChunkRecord]] = {}
    for c in chunks:
        by_key.setdefault((c.cand.emission_id, c.cand.revision), []).append(c)
    out: dict[str, Any] = {"n_emissions_seen": len(by_key)}
    if run.unit_done is None:
        out["done"] = {"status": "unavailable", "reason": "el candidato no emite unit_done (opcional)"}
    else:
        issues: dict[str, list[str]] = {"double_done": [], "negative_counts": [], "unknown_emission": [], "chunk_after_done": [],
                                        "premature_done": [], "claim_mismatch_final": []}
        seen_done: dict[tuple[str, int], int] = {}
        for d in sorted(run.unit_done, key=lambda d: (d.stamp.recv_ns, d.stamp.event_seq)):
            k = (d.emission_id, d.revision)
            tag = f"{k[0]}@{k[1]}"
            seen_done[k] = seen_done.get(k, 0) + 1
            if seen_done[k] == 2:
                issues["double_done"].append(tag)
            if d.n_chunks < 0 or d.total_samples < 0:
                issues["negative_counts"].append(tag)
            cs = by_key.get(k)
            if not cs:
                issues["unknown_emission"].append(tag)
                continue
            dk = (d.stamp.recv_ns, d.stamp.event_seq)
            upto = [c for c in cs if (c.stamp.recv_ns, c.stamp.event_seq) <= dk]
            after = [c for c in cs if (c.stamp.recv_ns, c.stamp.event_seq) > dk]
            if after:
                issues["chunk_after_done"].append(tag)
            if (len(upto), sum(c.pcm_samples for c in upto)) != (d.n_chunks, d.total_samples):
                if (len(cs), sum(c.pcm_samples for c in cs)) == (d.n_chunks, d.total_samples):
                    issues["premature_done"].append(tag)
                else:
                    issues["claim_mismatch_final"].append(tag)
        n_issue = sum(len(v) for v in issues.values())
        out["done"] = {"status": "ok", "n_declared": len(run.unit_done), "n_issues": n_issue, "issues": {k: v[:3] for k, v in issues.items() if v},
                       "n_unclosed": sum(1 for k in by_key if k not in seen_done),
                       "note": "inconsistencia DECLARATIVA del candidato; no prueba pérdida ni duplicación de transporte; sin cierre no es error (last_chunk/unit_done opcionales)"}
    if run.unit_cancelled is None:
        out["cancelled"] = {"status": "unavailable", "reason": "el candidato no emite unit_cancelled (opcional)"}
    else:
        plays = _play(run)
        aud_before, chunk_after, claim_false = [], [], []
        for cn in run.unit_cancelled:
            k = (cn.emission_id, cn.revision)
            tag = f"{k[0]}@{k[1]}"
            cs = by_key.get(k, [])
            ck = (cn.stamp.recv_ns, cn.stamp.event_seq)
            if any(v["c"].cand.emission_id == k[0] and v["c"].cand.revision == k[1] and v["onset"] is not None and v["onset"] <= cn.stamp.recv_ns
                   and (v["c"].stamp.recv_ns, v["c"].stamp.event_seq) <= ck for v in plays.values()):  # el chunk debe haber llegado ANTES del cancel en el arnés
                aud_before.append(tag)
            if any((c.stamp.recv_ns, c.stamp.event_seq) > ck for c in cs):
                chunk_after.append(tag)
            if cn.before_emit_claim and any((c.stamp.recv_ns, c.stamp.event_seq) <= ck for c in cs):
                claim_false.append(tag)
        out["cancelled"] = {"status": "ok", "n_cancelled": len(run.unit_cancelled), "n_already_audible_at_cancel": len(aud_before), "audible": aud_before[:3],
                            "n_chunk_after_cancel": len(chunk_after), "n_before_emit_claim_false": len(claim_false),
                            "note": "cancelar algo ya audible no lo des-dice; lo declarado por el candidato se contrasta con la reproducción"}
    out["ready"] = {"status": "unavailable", "reason": "sin unit_ready"} if run.unit_ready is None else {"status": "ok", "n": len(run.unit_ready),
                                                                                                         "note": "telemetría: NO prueba underrun"}
    return out


def claims_vs_evaluator(run: IncrementalRun, p: Params = Params()) -> dict[str, Any]:
    a, why = _alignment_state(run)
    if a is None:
        return {"status": "unavailable", "reason": why}
    rows, _ = _evaluable(run, _play(run))
    n_cmp, bad = 0, []
    for x in rows:
        r, cl = x["r"], x["chunk"].cand.source_span_claim
        if not cl or r.support_start_s is None or r.support_end_s is None:
            continue
        n_cmp += 1
        inter = max(0.0, min(cl[1], r.support_end_s) - max(cl[0], r.support_start_s))
        union = max(cl[1], r.support_end_s) - min(cl[0], r.support_start_s)
        if union <= 0 or inter / union < p.claim_iou_min:
            bad.append(r.realization_id)
    return {"status": "ok", "n_compared": n_cmp, "n_disagree": len(bad), "ids": bad[:5],
            "note": "los claims del candidato no se usan para medir; esto solo detecta autoasignaciones inconsistentes"}


def compute_all(run: IncrementalRun, p: Params = Params()) -> dict[str, Any]:
    return {
        "accounting": accounting(run), "first_onset": first_onset(run, p), "physical_queue": physical_queue(run),
        "semantic_lag": semantic_lag(run), "coverage_integrity": coverage_and_integrity(run, p), "anticipation": anticipation(run, p),
        "audible_revisions": audible_revisions(run), "transport_frontier": transport_frontier(run, p),
        "unit_closure": unit_closure(run), "claims_vs_evaluator": claims_vs_evaluator(run, p),
    }
