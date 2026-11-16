"""Informe del benchmark incremental (v2): cada familia en su sección, nivel de procedencia visible, sin certificar nada."""
from __future__ import annotations

from typing import Any

from .contracts import IncrementalRun
from .metrics import Params, compute_all

NOT_CERTIFIED = [
    "No certifica 300 ms ni ningún umbral de experiencia de usuario: son métricas DIAGNÓSTICAS de un módulo sin integrar.",
    "Sin alineación independiente del evaluador (`alignment`) no hay cobertura semántica, repetición, omisión ni desfase: aparece «unavailable», nunca cero. Con n_evaluable=0 el desfase y la anticipación son «unavailable»; la cobertura solo es 0 si el evaluador entrega una evaluación NEGATIVA completa (ledger por átomo con omisiones verificadas); sin evaluación es «unavailable».",
    "La cobertura semántica por átomos solo es completa con `completeness=complete`; si no, es una cota inferior etiquetada y lo no observado no es omisión.",
    "`temporal_support_fraction_heuristic` es una heurística de DURACIÓN fuente; no es cobertura, omisión, utilidad ni integridad.",
    "Los IDs, spans, `committed`, `unit_done` y `unit_ready` del candidato son claims/telemetría; un desajuste de cierre es una inconsistencia declarativa, no prueba pérdida de transporte.",
    "Transporte (entregas del arnés), generado-antes-del-soporte (recepción), audible-antes-del-soporte (reproducción) y revisión audible son categorías distintas; ninguna prueba acceso al futuro por acertar un número (una diferencia pre-fork bajo controles idénticos indica dependencia/no-determinismo a investigar).",
    "Un hueco entre chunks (p. ej. una pausa legítima) no es un underrun; `underrun_demonstrated` solo con evidencia del Player/arnés. Cola física ≠ desfase semántico; la reproducción es simulada, no dispositivo.",
    "Procedencia `test_oracle` = verdad inyectada por pruebas, nunca anotación humana; `auto_unvalidated` = solo diagnóstico.",
    "Sin escucha humana no hay afirmación de naturalidad; sin modelos de calidad no hay BLEU/chrF/MOS aquí; con pocos hablantes los IC no son poblacionales.",
]


def build_report(run: IncrementalRun, p: Params = Params()) -> dict[str, Any]:
    a = run.alignment
    return {
        "schema": "opentraductor-incremental-report/v1",
        "provenance": {"alignment_provenance": a.provenance if a else None, "alignment_tier": a.tier if a else None,
                       "correction_state": a.correction_state if a else None, "completeness": a.completeness if a else None,
                       "test_only": bool(a and a.provenance == "test_oracle"), "params": p.__dict__,
                       "ref_unit_provenance": sorted({u.provenance for u in run.ref_units}) or None},
        "sections": compute_all(run, p), "not_certified": NOT_CERTIFIED,
    }


def _v(x: Any, nd: int = 3) -> str:
    return "n/d" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def render_markdown(rep: dict[str, Any]) -> str:
    s, pr = rep["sections"], rep["provenance"]
    L = ["# Informe incremental (diagnóstico, módulo 1 v3)", ""]
    if pr["test_only"]:
        L += ["> **SOLO PRUEBAS**: la alineación usa procedencia `test_oracle` (verdad inyectada por adaptadores falsos); no es una medición.", ""]
    elif pr["alignment_tier"] == "diagnostic":
        L += ["> **DIAGNÓSTICO**: alineación `auto_unvalidated`; no es validación humana.", ""]
    L += [f"- alineación: {pr['alignment_provenance']} · corrección {pr['correction_state']} · completitud {pr['completeness']}"]
    a = s["accounting"]
    L += ["", "## Contabilidad (denominadores)", f"- streams {a['n_streams']} · fallos {a['n_failed']} · timeouts (censura) {a['n_timeout_censored']} · observados {a['n_observed']}"]
    f = s["first_onset"]
    L += ["", "## Primer onset (sin umbral de tamaño)",
          f"- primer onset: {_v(f.get('first_onset_s'))} s · antes del fin de la grabación: {_v(f.get('before_recording_end'))} · fuente activa en el onset: {_v(f.get('source_active_at_onset'))} · habla futura restante: {_v(f.get('future_speech_remaining_s'))} s · durante habla: {_v(f.get('during_speech'))}"]
    ci = s["coverage_integrity"]
    L += ["", "## Cobertura SEMÁNTICA por átomos e integridad (contra referencia independiente)"]
    if ci.get("status") != "ok":
        L.append(f"- unavailable: {ci.get('reason')}")
    else:
        L += [f"- cobertura semántica: {_v(ci['semantic_atom_coverage'])} (cota inferior si no es completa: {_v(ci['semantic_atom_coverage_lower_bound'])}) · átomos {ci['n_atoms_realized']}/{ci['n_atoms']} · completitud {ci['completeness']}",
              f"- primer átomo útil: {_v(ci['first_useful_atom_s'])} s · unidades completas {ci['n_units_complete']}/{ci['n_units']} · críticamente incompletas {ci['n_units_critically_incomplete']}",
              f"- átomos repetidos (realizaciones audibles distintas) {ci['n_repeated_atoms']} · contradichos finales {ci['n_contradicted_atoms']} · contradicción OÍDA (aunque corregida) {ci['n_atoms_contradiction_heard']} (críticas {ci['n_critical_contradiction_heard']}) · no atribuibles {ci['n_unattributable']}",
              f"- evaluación por átomo: sin evaluar {ci.get('n_not_assessed')} · conflictos del ledger {len(ci.get('ledger_conflicts') or [])} · omisiones verificadas {len(ci.get('omitted_atoms') or [])} · completitud efectiva {ci.get('complete_effective')}",
              f"- heurística temporal (NO cobertura): umbral {_v(ci['temporal_support_fraction_heuristic']['threshold'])} de duración fuente"]
    q = s["physical_queue"]
    L += ["", "## Cola FÍSICA (no es desfase semántico)"]
    if q.get("status") == "unavailable":
        L.append(f"- unavailable: {q['reason']}")
    else:
        L.append(f"- backlog máx {_v(q['max_s'])} s · media {_v(q['mean_s'])} s · crecimiento {_v(q['growth_s_per_s'], 4)} s/s · huecos entre chunks {_v(q['interchunk_gap_s'])} s ({q['n_interchunk_gaps']}) · underrun demostrado: {_v(q['underrun_demonstrated_s'])} · speed medio {_v(q['speed_factor_mean'])}")
    sl = s["semantic_lag"]
    L += ["", "## Desfase SEMÁNTICO por realización audible (con signo)"]
    if sl.get("status") != "ok":
        L.append(f"- unavailable: {sl.get('reason')}")
    else:
        L += [f"- n={sl['n_realizations']} · mediana vs fin de soporte {_v(sl['median_lag_to_support_end_s'])} s · vs fin de unidad {_v(sl['median_lag_to_unit_end_s'])} s · pendiente vs fuente (fin de soporte) {_v(sl['slope_support_end_s_per_s'], 4)} s/s · (fin de unidad) {_v(sl['slope_unit_end_s_per_s'], 4)} s/s · con incertidumbre desconocida {sl['n_uncertainty_unknown']} · sin fin de soporte {sl['n_support_end_missing']}",
              f"- titular SOLO `realized`; reemisiones {sl['n_reemitted_excluded_from_headline']} (titular: una observación por átomo, n átomos {sl['n_atoms_in_headline']} de {sl['n_realized_realizations_total']} realizaciones) · parciales {sl['n_partial_excluded_from_headline']} · contradichas {sl['n_contradicted_excluded_from_headline']} (visibles aparte, nunca mejoran el titular)"]
    an = s["anticipation"]
    L += ["", "## Categorías separadas"]
    if an.get("status") != "ok":
        L.append(f"- anticipación: unavailable ({an.get('reason')})")
    else:
        ab = an["audible_before_support"]
        L += [f"- generado antes del soporte (preparación privada, no es pronunciación): {an['generated_before_support']['n']}",
              "- audible antes del soporte: " + (f"{ab['n']} seguras · ambiguas {ab['n_ambiguous']} · no antes {ab['n_not_before']} · sin clasificar por incertidumbre de salida desconocida {ab['n_unclassified_unknown_output_uncertainty']}" if "n" in ab else f"unavailable ({ab['reason']})"),
              f"- sin clasificar por soporte/incertidumbre desconocidos: {an['n_unclassified_unknown_support_or_uncertainty']}"]
    t = s["transport_frontier"]
    dv = t["deliveries"]
    r = s["audible_revisions"]
    L += [f"- violación de frontera de TRANSPORTE (entregas): " + (f"{dv['violations']} de {dv['n']} (no monótonas {dv['non_monotonic']})" if dv.get("status") == "ok" else f"unavailable ({dv['reason']})")
          + f" · estampas de chunks: {t['chunk_stamps']['violations']} · tolerancia {t['tolerance_s']} s ({t['tolerance_basis']})",
          f"- revisión AUDIBLE de lo ya pronunciado: {r['spoken_then_revised']} (revisadas antes de oírse: {r['revised_before_audible']}) · `supersedes` malformados (diagnóstico): {r['n_malformed_supersedes']}"]
    uc = s["unit_closure"]
    d, cn = uc["done"], uc["cancelled"]
    L += ["", "## Cierre y cancelación declarados (opcionales; inconsistencia declarativa, no transporte)",
          "- unit_done: " + (f"{d['n_declared']} · inconsistencias {d['n_issues']} {d['issues']} · sin cerrar {d['n_unclosed']}" if d.get("status") == "ok" else f"unavailable ({d.get('reason')})"),
          "- unit_cancelled: " + (f"{cn['n_cancelled']} · ya audibles al cancelar {cn['n_already_audible_at_cancel']} · chunks tras cancelar {cn['n_chunk_after_cancel']}" if cn.get("status") == "ok" else f"unavailable ({cn.get('reason')})")]
    c = s["claims_vs_evaluator"]
    L += ["", "## Claims del candidato vs evaluador (diagnóstico)", "- " + (f"comparados {c['n_compared']}, en desacuerdo {c['n_disagree']}" if c.get("status") == "ok" else f"unavailable: {c.get('reason')}")]
    L += ["", "## Lo que NO certifica"] + [f"- {x}" for x in rep["not_certified"]]
    return "\n".join(L) + "\n"
