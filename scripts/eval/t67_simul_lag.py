#!/usr/bin/env python3
"""T-67 paso 2 / T-73: demora al audio cuando parte de la traducción puede sonar ANTES del fin de habla.

t58_stage_breakdown.py toma el 1er tts_chunk menos el fin de habla: con OPENTRADUCTOR_MT_SIMUL=1 la 1.ª cláusula
suena antes del fin de habla y ese total da negativo, y las etapas (mt, tts) se mezclan. Este script separa:
  primer_audio = 1er tts_chunk - fin de habla              (negativo = empezó a sonar mientras la persona hablaba)
  resto        = 1er tts_chunk desde el último mt_final - fin de habla  (lo que se espera por lo que faltaba)
  clausulas    = simul_clause por frase (run.json -> describe -> engine_events, desde 922dd7d)
  piezas/frase = (cláusulas + resto) por frase ok; final contradice = frases cuyo asr_text_es final no empieza con
                 las cláusulas que ya sonaron (lo hablado no se deshace)
Origen = t0_ns + segments[0].end_s, igual que end_to_first_ms.
Métrica del objetivo (decisión del 18/11), por frase desde segment_results.signed_end_to_first_ms:
  objetivo = max(0, primer audio) si el final del ASR no contradice las cláusulas que ya sonaron; ∞ si las contradice.
  pesimista = la compuesta del benchmark: anticipadas (sin verificar) = ∞.
  IC95 del p50 con bootstrap por cluster (segment_results.cluster, 2000 remuestreos, semilla fija).
Uso: python3 scripts/eval/t67_simul_lag.py artifacts/<run>/<brazo>/latency [...más brazos]"""
import json
import math
import random
import statistics as st
import sys
from collections import Counter, defaultdict
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
    frases, piezas, contra = 0, 0, 0
    obj, pes = defaultdict(list), defaultdict(list)  # cluster -> ms por frase
    try:
        run = json.load(open(d / "run.json"))
        evs = run["meta"]["adapter_metadata"].get("engine_events", [])  # describe() del adaptador
        clauses = Counter(e["kind"] for e in evs)
        by_stream = defaultdict(list)  # stream_id -> textos de simul_clause (en orden)
        for e in evs:
            if e["kind"] == "simul_clause":
                by_stream[e["stream_id"]].append(e.get("text_es", ""))
        for s in run.get("segment_results", []):
            final = s.get("asr_text_es")
            sig = s.get("signed_end_to_first_ms")
            if s.get("outcome") not in ("ok", "anticipated") or not final or sig is None:
                continue
            cl = by_stream.get(s["stream_id"], [])
            frases += 1
            piezas += 1 + len(cl)
            spoken = [_norm(w) for w in " ".join(cl).split()]
            bad = bool(spoken) and [_norm(w) for w in final.split()][: len(spoken)] != spoken
            contra += bad  # el final del ASR ya no empieza con lo que sonó
            c = s.get("cluster") or s.get("item_id")
            obj[c].append(math.inf if bad else max(0.0, sig))
            pes[c].append(math.inf if s["outcome"] == "anticipated" else sig)
    except (OSError, ValueError, KeyError):
        pass
    f = lambda v: f"{st.median(v):.0f} / {pct(v, .95):.0f}" if v else "-"
    ppf = f"{piezas / frases:.2f}" if frases else "-"
    return (f"| {d.parent.name} | {n} | {f(first)} | {f(rest)} | {sum(x < 0 for x in first)} | "
            f"{clauses.get('simul_clause', 0)} | {ppf} | {contra}/{frases} | {clauses.get('spec_result', 0)} | "
            f"{_p50ic(obj)} | {_p50ic(pes)} | {len(obj)} |")


def _ms(x: float) -> str:
    return "∞" if math.isinf(x) else f"{x:.0f}"


def _p50ic(by_cluster: dict, n_boot: int = 2000) -> str:
    """p50 de todas las frases [IC95 por bootstrap de clusters]."""
    allv = [x for v in by_cluster.values() for x in v]
    if not allv:
        return "-"
    keys, rng, meds = list(by_cluster), random.Random(67), []
    for _ in range(n_boot):
        sample = [x for k in rng.choices(keys, k=len(keys)) for x in by_cluster[k]]
        meds.append(st.median_low(sorted(sample)))
    meds.sort()
    return f"{_ms(st.median_low(sorted(allv)))} [{_ms(meds[int(.025 * n_boot)])}, {_ms(meds[int(.975 * n_boot) - 1])}]"


def _norm(w: str) -> str:
    return w.strip(".,;:¿?¡!\"'()…-").lower()


if __name__ == "__main__":
    print("| brazo | n | primer audio p50/p95 ms | resto p50/p95 ms | antes del fin | simul_clause | piezas/frase | final contradice | spec_result | p50 objetivo [IC95] | p50 pesimista [IC95] | clusters |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for a in sys.argv[1:]:
        print(arm(Path(a)))
