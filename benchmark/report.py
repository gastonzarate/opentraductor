"""Informe Markdown reproducible a partir de ``run.json``."""
from __future__ import annotations

import json
from pathlib import Path


def _fmt(v, nd=1):
    return "n/d" if v is None else f"{v:.{nd}f}"


def _q_table(title: str, label: str, qs: dict, continuous: bool = False) -> list[str]:
    lines = ["", f"**{title}**", f"_{label}_", "",
             "| cuantil | valor | IC95 bootstrap por cluster | n | clusters indep. | aviso |",
             "|---|---|---|---|---|---|"]
    for name, q in qs.items():
        lo, hi = q["ci"]
        val = _fmt(q["value"]) if q["identifiable"] else "no identificable"
        ci = f"[{_fmt(lo)}, {_fmt(hi) if hi is not None else '∞'}]"
        warn = []
        if q["low_sample"]:
            warn.append(f"muestra baja (necesita ≥{q['min_n_for_quantile']} clusters)")
        if q.get("clusters_assumed_independent"):
            warn.append("sin clusters: se asume independencia")
        if continuous:
            warn.append("stream continuo: el IC por hablante/cluster NO captura la dependencia temporal (backlog y lag se arrastran entre ítems); optimista")
        lines.append(f"| {name} | {val} | {ci} | {q['n']} | {q.get('n_clusters', '')} | {'; '.join(warn)} |")
    return lines


def _summary_block(title: str, s: dict, continuous: bool = False) -> list[str]:
    c = s["counts"]
    lines = [
        "",
        f"### {title}",
        f"N={s['n_total']} · observados={c['observed']} · timeout={c['timeout']} · fallo={c['failure']} · "
        f"anticipados={c['anticipated']} · no medibles={c['unmeasurable']} · deadlines(ms)={s['timeout_deadlines_ms']}",
    ]
    lines += _q_table("Latencia observada con censura (Kaplan-Meier)", s["censoring_aware"]["label"], s["censoring_aware"]["quantiles"], continuous)
    comp = s["composite_time_to_valid_output"]
    if not comp["ordering_check"]["identifiable_ordering"]:
        lines += ["", f"_Compuesto no identificable: {comp['ordering_check']['reason']}_"]
    else:
        lines += _q_table("Compuesto tiempo-hasta-salida-válida (NO es latencia observada)", comp["label"], comp["quantiles"], continuous)
    return lines


def render_markdown(run: dict) -> str:
    meta = run["meta"]
    continuous = bool(meta.get("run_config", {}).get("continuous"))
    lines = ["# Informe de corrida", "", f"**{meta['label']}**"]
    if meta.get("infra_test_only"):
        lines += ["", "> Resultados de adaptador DUMMY. Sirven solo para validar la infraestructura.",
                  "> No representan ningún candidato ni hardware."]
    if meta.get("clock_kind") == "virtual":
        lines += ["", "> Reloj VIRTUAL: las latencias son tiempo simulado, no mediciones."]
    if meta.get("contention_possible"):
        lines += ["", "> **PILOTO con contención posible**: había procesos ajenos activos (ver carga abajo). La exclusividad es solo del "
                  "equipo opentraductor; no se afirma máquina en reposo ni GPU exclusiva. No es una comparación final."]
    if meta.get("aborted"):
        lines += ["", f"> **CORRIDA ABORTADA**: {meta['aborted']}. Los streams omitidos cuentan como fallo."]
    env = meta["environment"]
    lines += [
        "",
        f"- Adaptador: `{meta['adapter']['name']}` v{meta['adapter']['version']} modo `{meta['adapter']['mode']}`",
        f"- Split: `{meta['split']}` · ítems: {meta['n_items']}",
        f"- Umbral de silencio: {meta['silence_threshold_dbfs']} dBFS",
        f"- sha256 código benchmark: `{env['benchmark_code_sha256'][:16]}…` · manifest: `{(meta.get('manifest_sha256') or 'n/d')[:16]}`",
        f"- Hardware: {env['hardware'].get('cpu_brand')} · {env['hardware'].get('platform')}",
        f"- Estado de la corrida: {meta.get('run_status_label')}",
        f"- Carga al inicio: {env['load_at_start']}",
        f"- Carga al final: {meta['load_at_end']}",
        f"- Eventos fuera de stream: {meta['stray_events']} · error del colector: {meta['collector_error']}",
        "",
        "## Cold start",
        "",
        f"- open(): {_fmt(run['cold_start']['open_ms'])} ms · primer start_stream(): {_fmt(run['cold_start']['first_start_stream_ms'])} ms",
        f"- Primer stream (calentamiento={run['cold_start']['first_stream_is_warmup']}): "
        f"fin→primer output {_fmt(run['cold_start']['first_stream_end_to_first_ms'])} ms · "
        f"petición→primer output {_fmt(run['cold_start']['first_stream_request_to_first_ms'])} ms",
        "",
    ]
    for g in run["aggregate"]["groups"]:
        lines += [f"## Grupo: modo `{g['adapter_mode']}` · salida `{g['output_kind']}` · reloj `{g['clock_kind']}` · límites `{g['boundary_precision']}`", ""]
        if g["infra_note"]:
            lines += [f"_{g['infra_note']}_", ""]
        if g["boundary_precision"] != "precise":
            lines += ["> Límites estimados o desconocidos: la tolerancia declarada es una SUPOSICIÓN, no una cota validada ni un IC. "
                      "Solo diagnóstico; no sirve para afirmar el objetivo de 300 ms.", ""]
        lines += [f"Origen de la métrica principal `{g['primary_metric']}`: {g['primary_origin']}.", "",
                  "| outcome | n |", "|---|---|"]
        lines += [f"| {o} | {c} |" for o, c in g["outcome_counts"].items()]
        first_what = "primer audio DISPONIBLE" if g["output_kind"] == "audio" else "primer mt_final recibido (texto)"
        lines += _summary_block(f"{g['primary_metric']} — {first_what} (titular)", g[g["primary_metric"]], continuous)
        if g["end_to_playout_est_ms"]:
            lines += _summary_block("end_to_playout_est_ms — reproducción ESTIMADA (cola simulada; no es dispositivo)", g["end_to_playout_est_ms"], continuous)
        if g["adapter_mode"] == "streaming":
            lines += _summary_block("start_to_first_ms (otra métrica; no se mezcla con la anterior)", g["start_to_first_ms"], continuous)
        d = g["device_output"]
        lines += ["", f"Salida a dispositivo: {d['status']}" + (f" ({d['reason']})" if d["status"] == "unavailable" else f" (n={d['n']})")]
        a = g["anticipation"]
        lines += ["", f"**Anticipación** (publicada junto al lag): {a['count']} fragmentos ({a['rate']:.3f}); valores con signo (ms): {a['signed_end_to_first_ms']}. {a['note']}.", ""]
    lines += ["## Reproducción / lag (backlog ≠ lag semántico)", ""]
    for p in run["playout"][:20]:
        sl = p["semantic_lag"]
        b = p.get("backlog")
        ot = p.get("offline_throughput", {})
        extra = f"; RTF offline {_fmt(ot['rtf'], 3)} ({_fmt(ot['x_realtime'], 1)}× tiempo real)" if "rtf" in ot else ""
        te = p.get("tts_emission_rtf", {})
        extra += f"; RTF emisión TTS {_fmt(te['rtf'], 2)}" if te.get("rtf") is not None else ""
        lines.append(
            f"- `{p['stream_id']}`: chunks={p['n_chunks']}; backlog máx={_fmt(b['max_s'], 3) if b else 'n/d'} s; lag por segmento: {sl['status']}"
            + (f" (media {_fmt(sl['mean_s'], 3)} s, mín {_fmt(sl['min_s'], 3)} s, n={sl['n']}; "
               + (f"tendencia {_fmt(sl['trend_s_per_s'], 4)} s/s" if sl["trend_s_per_s"] is not None else f"tendencia no calculada ({sl['n_distinct_source_spans']} segmento(s) fuente distinto(s) < {sl['trend_min_n']})")
               + (f"; {sl['n_negative_unverified_anticipation']} con lag<0 = anticipación NO verificada" if sl["n_negative_unverified_anticipation"] else "")
               + "; por segmento, no por palabra)" if sl["status"] == "available" else f" ({sl['reason']})")
            + extra
        )
    if any(p.get("tts_emission_rtf", {}).get("rtf") is not None for p in run["playout"]):
        lines.append("- _RTF emisión TTS = (recv último chunk − recv primer chunk) / Σ duración de los chunks salvo el último. NO es costo de síntesis: en streaming incluye la espera de audio de entrada/replay entre chunks, así que > 1 no demuestra que el TTS sea más lento que tiempo real. La evidencia de espera real son backlog, underrun_s y el lag por segmento._")
    if len(run["playout"]) > 20:
        lines.append(f"- … {len(run['playout']) - 20} streams más en run.json")
    q = run.get("quality")
    if q:
        lines += ["", "## Calidad", ""]
        for key in ("mt_text", "planned_text", "audible_text"):
            if key not in q:
                continue
            blk = q[key]
            lines.append(f"### {key}")
            if blk.get("status") == "unavailable":
                lines.append(f"- unavailable: {blk.get('reason')}")
                continue
            for prov in ("human", "generated", "unknown"):
                if prov in blk:
                    s = blk[prov]
                    if s.get("status") == "ok":
                        lines.append(f"- referencias `{prov}`: BLEU {_fmt(s['bleu'])} · chrF {_fmt(s['chrf'])} (n={s['n_scored']}, hipótesis vacías={s['n_empty_hypothesis']})")
                    else:
                        lines.append(f"- referencias `{prov}`: unavailable ({s.get('reason')})")
            lines.append(f"- ítems sin referencia: {blk.get('items_without_reference')}")
        tc = q.get("text_checks")
        if tc:
            lines.append("### Chequeos de superficie (números / negaciones / términos técnicos)")
            lines.append(f"_{tc['label']}_")
            for name, c in tc["checks"].items():
                lines.append(f"- {name}: aplicables {c['n_applicable']}, marcados {c['n_flagged']}")
                for e in c["flagged_examples"]:
                    lines.append(f"  - `{e['item_id']}`: «{e['src']}» → «{e['hyp']}»")
            lines += [f"- límite: {l}" for l in tc["limitations"]]
        asr = q.get("candidate_asr_es")
        if asr:
            lines.append("### ASR es del candidato vs transcripción (separado de la traducción)")
            if asr.get("status") == "unavailable":
                lines.append(f"- unavailable: {asr.get('reason')}")
            for prov in ("human", "generated", "unknown"):
                if prov in asr:
                    a = asr[prov]
                    lines.append(f"- transcripción `{prov}`: WER {_fmt(a['wer'], 3)} · CER {_fmt(a['cer'], 3)} (n={a['n']}, hipótesis vacías={a['n_empty_hypothesis']}; {a['normalization']})")
        lines.append(f"- métrica neural: {q['neural_metric']['status']} ({q['neural_metric']['reason']})")
        lines += [f"- Aviso: {c}" for c in q["caveats"]]
    return "\n".join(lines) + "\n"


def _relabel_legacy(run: dict, run_dir: Path) -> dict:
    """Corrige el texto de razones de corridas viejas SOLO desde datos guardados (sin rerun)."""
    q = run.get("quality") or {}
    mt = q.get("mt_text") or {}
    if mt.get("status") == "unavailable" and mt.get("reason") == "el candidato no emitió mt_final" and (run_dir / "traces.jsonl").exists():
        n = sum(1 for line in (run_dir / "traces.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()
                for e in json.loads(line)["events"] if e["kind"] == "mt_final")
        if n:
            mt["reason"] = (f"el candidato SÍ emitió {n} mt_final, pero no se atribuyen a ítems (sin correspondencia fuente→salida por ítem; "
                            "típico del stream continuo): sin calidad de MT por ítem. No significa que falte la MT")
    for p in run.get("playout", []):
        if p.get("tts_emission_rtf", {}).get("rtf") is not None:
            p["tts_emission_rtf"]["label"] = "ver nota de RTF de emisión en el informe (no es costo de síntesis)"
    return run


def write_report(run_dir: str | Path, out_path: str | Path | None = None) -> Path:
    """``out_path``: escribir en otro lugar (p. ej. al regenerar el informe de una corrida ajena sin tocar sus archivos)."""
    run_dir = Path(run_dir)
    run = _relabel_legacy(json.loads((run_dir / "run.json").read_text(encoding="utf-8")), run_dir)
    path = Path(out_path) if out_path else run_dir / "report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(run), encoding="utf-8")
    return path
