"""T-83: cuenta finales de ASR con la última palabra cortada en las trazas de t83_tail_ab.sh.

Por brazo y clip: completas (``policial`` en 1584, ``nazismo`` en 1516), cortadas y cortes de frase
(cantidad de asr_final). Muestra los textos cortados y el motivo del endpoint previo (wait_reason).
Uso: python3 scripts/eval/t83_tail_count.py artifacts/t83/ab
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

WORD = {"1584": "policial", "1516": "nazismo", "1652": "mayoritaria"}


def main(out: str) -> int:
    res: dict = collections.defaultdict(lambda: collections.Counter())
    cuts: list = []
    for f in sorted(Path(out).glob("*/r*/traces.jsonl")):
        arm = f.parts[-3]
        for line in f.open():
            d = json.loads(line)
            if d.get("warmup"):
                continue
            sid = d["stream_id"]
            clip = next((k for k in WORD if k in sid), None)
            if clip is None:
                continue
            var = sid.rsplit("-", 1)[-1] if sid.endswith(("-bh", "-g07")) else "orig"
            fins = [e["text"] or "" for e in d["events"] if e["kind"] == "asr_final"]
            eps = [e for e in d["events"] if e["kind"] == "endpoint"]
            joined = " | ".join(fins)
            ok = WORD[clip] in joined.lower()
            k = (arm, clip, var)
            res[k]["ok" if ok else "cut"] += 1
            res[k]["segs"] += len(fins)
            if not ok:
                cuts.append((arm, clip, var, f.parts[-2], joined[-70:],
                             [(e.get("wait_ms"), e.get("wait_reason")) for e in eps]))
    print("brazo clip variante: completas/total, finales por pasada")
    for k, c in sorted(res.items()):
        n = c["ok"] + c["cut"]
        print(f"{k[0]:8} {k[1]} {k[2]:5} {c['ok']}/{n}  segs {c['segs'] / max(n, 1):.1f}")
    for c in cuts:
        print("CORTE", *c)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
