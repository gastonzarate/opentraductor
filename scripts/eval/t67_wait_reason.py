"""T-67: cuenta wait_reason/wait_ms del endpoint por brazo de un RUNDIR de t69_bench.sh.

El runtime manda wait_reason y wait_ms en ``unit_reason`` del evento endpoint ("ep:<reason>:<ms>"), porque
AdapterEvent no tiene esos campos. Uso: python scripts/eval/t67_wait_reason.py artifacts/t67-ab/run
"""
from __future__ import annotations

import collections
import glob
import json
import statistics
import sys


def main(run: str) -> None:
    for path in sorted(glob.glob(f"{run}/*/latency/traces.jsonl")):
        arm = path[len(run) + 1:].split("/")[0]
        reasons: collections.Counter = collections.Counter()
        waits: dict = collections.defaultdict(list)
        segs = finals = 0
        for line in open(path, encoding="utf-8"):
            t = json.loads(line)
            if t.get("warmup") or str(t.get("stream_id", "")).startswith("warmup"):
                continue
            segs += len(t.get("segments", []))
            for e in t["events"]:
                if e["kind"] == "asr_final":
                    finals += 1
                if e["kind"] != "endpoint":
                    continue
                ur = e.get("unit_reason") or ""
                if ur.startswith("ep:"):
                    _, reason, ms = ur.split(":", 2)
                    reasons[reason] += 1
                    waits[reason].append(float(ms))
                else:
                    reasons[None] += 1
        med = {k: statistics.median(v) for k, v in waits.items()}
        print(f"{arm}: frases={segs} asr_final={finals} wait_reason={dict(reasons)} mediana_wait_ms={med}")


if __name__ == "__main__":
    main(sys.argv[1].rstrip("/"))
