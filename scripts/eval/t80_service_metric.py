"""T-80: recalcula sobre trazas del banco la métrica de la app (`last_endpoint_to_audio_ms` de service.py).

service.py mide endpoint -> primer tts_chunk del mismo seg_id, y solo si el primer tts_chunk llega DESPUÉS del
endpoint (si ya sonó una cláusula antes, la frase queda sin medida). Sirve para comparar con la sesión de tienda
de QA, que lee esa métrica, en vez del «total desde fin de habla» de t67_intercalado.

Uso: python scripts/eval/t80_service_metric.py artifacts/t80/latencia ref simul
"""
from __future__ import annotations

import glob
import json
import os
import statistics
import sys


def arm_values(root: str, arm: str):
    vals, unmeasured, clause_first = [], 0, 0
    for f in sorted(glob.glob(os.path.join(root, arm, "r*", "latency", "traces.jsonl"))):
        for line in open(f):
            d = json.loads(line)
            if str(d.get("stream_id", "")).startswith("warmup"):
                continue
            ep, first = {}, {}
            for e in d["events"]:
                k = e.get("commit_id")
                if k is None:
                    continue
                if e["kind"] == "endpoint":
                    ep.setdefault(k, e["recv_ns"])
                elif e["kind"] == "tts_chunk":
                    first.setdefault(k, e["recv_ns"])
            for k, t0 in ep.items():
                t1 = first.get(k)
                if t1 is None:
                    unmeasured += 1
                elif t1 < t0:
                    unmeasured += 1
                    clause_first += 1
                else:
                    vals.append((t1 - t0) / 1e6)
    return vals, unmeasured, clause_first


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))]


def main() -> None:
    root, arms = sys.argv[1], sys.argv[2:]
    for arm in arms:
        v, un, cf = arm_values(root, arm)
        if not v:
            print(f"{arm}: sin datos")
            continue
        print(f"{arm}: n={len(v)} sin_medida={un} (audio de cláusula antes del endpoint: {cf}) "
              f"p50={statistics.median(v):.0f} p95={pct(v, 0.95):.0f} media={statistics.mean(v):.0f}")


if __name__ == "__main__":
    main()
