#!/usr/bin/env python3
"""T-67 [bench]: cortes de frase por brazo (riesgo de partir frases con esperas cortas) a partir de traces.jsonl.

Uso: python3 scripts/eval/t67_cuts.py <brazo>/latency [<brazo2>/latency ...]   (el primero es el control: off)

Referencia = 1 segmento por clip (los clips dev16 son una frase). Por traza no warmup:
  segmentos   = eventos 'endpoint' (cada uno abre un commit_id).
  corte falso = endpoint con audio_pos_s < end_s de referencia - 150 ms (misma definición que t67_vad_tail.py).
Por brazo: trazas, segmentos totales, trazas con >1 segmento, cortes falsos, y clips (item_id) con más segmentos
que el control (mediana por clip entre repeticiones). Lista los clips partidos con su audio_pos_s y unit_reason.
"""
import json
import statistics as st
import sys
from collections import defaultdict

MARGEN_S = 0.15


def cargar(path):
    por_clip = defaultdict(list)  # item_id -> [(n_seg, [(audio_pos_s, reason)], end_s)]
    with open(path + "/traces.jsonl") as fh:
        for line in fh:
            t = json.loads(line)
            if t.get("warmup") or str(t.get("stream_id", "")).startswith("warmup"):
                continue
            seg = (t.get("segments") or [{}])[0]
            item = seg.get("item_id") or t["stream_id"]
            end_s = seg.get("end_s")
            eps = [(e.get("audio_pos_s"), e.get("unit_reason")) for e in t["events"] if e["kind"] == "endpoint"]
            por_clip[item].append((len(eps), eps, end_s))
    return por_clip


def main(paths):
    datos = {p: cargar(p) for p in paths}
    ctrl = datos[paths[0]]
    print("| brazo | trazas | segmentos | trazas >1 seg | cortes falsos | clips con más seg. que control |")
    print("|---|---|---|---|---|---|")
    detalle = []
    for p in paths:
        d = datos[p]
        nombre = p.rstrip("/").split("/")[-2] if p.rstrip("/").endswith("latency") else p
        trazas = sum(len(v) for v in d.values())
        segs = sum(n for v in d.values() for n, _, _ in v)
        multi = sum(1 for v in d.values() for n, _, _ in v if n > 1)
        falsos = sum(
            1
            for v in d.values()
            for _, eps, end_s in v
            for pos, _ in eps
            if end_s is not None and pos is not None and pos < end_s - MARGEN_S
        )
        mas = []
        for item, v in d.items():
            c = ctrl.get(item)
            if not c:
                continue
            if st.median(n for n, _, _ in v) > st.median(n for n, _, _ in c):
                mas.append(item[-12:])
        print(f"| {nombre} | {trazas} | {segs} ({segs / max(trazas, 1):.2f}/traza) | {multi} | {falsos} | {len(mas)} |")
        detalle.append((nombre, d))
    print("\n## Clips partidos (>1 segmento) por brazo\n")
    for nombre, d in detalle:
        for item, v in sorted(d.items()):
            for n, eps, end_s in v:
                if n > 1:
                    pts = " ".join(f"{pos:.2f}s:{r}" for pos, r in eps)
                    print(f"- {nombre} {item[-16:]} fin ref {end_s}: {n} seg -> {pts}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
