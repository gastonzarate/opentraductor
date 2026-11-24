#!/usr/bin/env python3
"""T-88: cuántos trozos de voz salen por brazo y cuántos son de 1-2 palabras (suenan entrecortados).

Uso: python3 scripts/eval/t88_trozos.py artifacts/t88b/latencia ref k5 k5ti16
Lee <dir>/<brazo>/r*/latency/**/traces.jsonl, toma planned_text_en de los eventos tts_chunk (sin repetir el
mismo texto seguido dentro de un stream) y omite warmup.
"""
import json
import pathlib
import sys


def trozos(arm_dir: pathlib.Path):
    out = []
    for f in sorted(arm_dir.glob("r*/latency/**/traces.jsonl")):
        for line in open(f):
            r = json.loads(line)
            if r.get("warmup") or str(r.get("stream_id", "")).startswith("warmup"):
                continue
            prev = None
            for e in r["events"]:
                t = e.get("planned_text_en")
                if e.get("kind") == "tts_chunk" and t and t != prev:
                    out.append(t)
                    prev = t
    return out


def main():
    base = pathlib.Path(sys.argv[1])
    print("| brazo | trozos | de 1-2 palabras | ejemplos |")
    print("|---|---|---|---|")
    for arm in sys.argv[2:]:
        ts = trozos(base / arm)
        cortos = [t for t in ts if len(t.split()) <= 2]
        ej = ", ".join(sorted({f"«{t}»" for t in cortos})[:4])
        pct = 100 * len(cortos) / len(ts) if ts else 0
        print(f"| {arm} | {len(ts)} | {len(cortos)} ({pct:.0f} %) | {ej} |")


if __name__ == "__main__":
    main()
