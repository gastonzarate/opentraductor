#!/usr/bin/env python3
"""T-67 paso 2 / T-73: demora al audio cuando parte de la traducción puede sonar ANTES del fin de habla.

t58_stage_breakdown.py toma el 1er tts_chunk menos el fin de habla: con OPENTRADUCTOR_MT_SIMUL=1 la 1.ª cláusula
suena antes del fin de habla y ese total da negativo, y las etapas (mt, tts) se mezclan. Este script separa:
  primer_audio = 1er tts_chunk - fin de habla              (negativo = empezó a sonar mientras la persona hablaba)
  resto        = 1er tts_chunk desde el último mt_final - fin de habla  (lo que se espera por lo que faltaba)
  clausulas    = simul_clause por frase (run.json -> describe -> engine_events, desde 922dd7d)
Origen = t0_ns + segments[0].end_s, igual que end_to_first_ms.
Uso: python3 scripts/eval/t67_simul_lag.py artifacts/<run>/<brazo>/latency [...más brazos]"""
import json
import statistics as st
import sys
from collections import Counter
from pathlib import Path


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, round(q * (len(v) - 1)))] if v else float("nan")


def arm(d: Path) -> str:
    first, rest, n = [], [], 0
    for line in open(d / "traces.jsonl"):
        t = json.loads(line)
        if t.get("warmup") or t.get("error") or not t.get("segments"):
            continue
        end_ns = t["t0_ns"] + t["segments"][0]["end_s"] * 1e9
        tts = sorted(e["recv_ns"] for e in t["events"] if e["kind"] == "tts_chunk")
        mf = [e["recv_ns"] for e in t["events"] if e["kind"] == "mt_final"]
        if not tts or not mf:
            continue
        n += 1
        first.append((tts[0] - end_ns) / 1e6)
        after = [x for x in tts if x >= max(mf)]
        if after:
            rest.append((after[0] - end_ns) / 1e6)
    clauses = Counter()
    try:
        run = json.load(open(d / "run.json"))
        evs = run["meta"]["adapter_metadata"].get("engine_events", [])  # describe() del adaptador
        clauses = Counter(e["kind"] for e in evs)
    except (OSError, ValueError, KeyError):
        pass
    f = lambda v: f"{st.median(v):.0f} / {pct(v, .95):.0f}" if v else "-"
    return (f"| {d.parent.name} | {n} | {f(first)} | {f(rest)} | {sum(x < 0 for x in first)} | "
            f"{clauses.get('simul_clause', 0)} | {clauses.get('spec_result', 0)} |")


if __name__ == "__main__":
    print("| brazo | n | primer audio p50/p95 ms | resto p50/p95 ms | antes del fin | simul_clause | spec_result |")
    print("|---|---|---|---|---|---|---|")
    for a in sys.argv[1:]:
        print(arm(Path(a)))
