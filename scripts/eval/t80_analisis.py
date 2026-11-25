#!/usr/bin/env python3
"""T-80: resume artifacts/t80 (scripts/eval/t80_intercalado.sh). Imprime Markdown.

Secciones: (1) cortes falsos en charlas (t67_cuts.py) y motivos de espera por brazo; (2) latencia desde el fin de habla
(resumen de t67_intercalado.py); (3) 1.er audio desde el INICIO de habla en clips de más de 5 s; (4) términos ctx0/ctx200.
Uso: python3 scripts/eval/t80_analisis.py artifacts/t80
Todo lo que falte (fase sin terminar) se omite con una nota."""
import json
import pathlib
import shutil
import statistics as st
import subprocess
import sys
from collections import Counter, defaultdict

HERE = pathlib.Path(__file__).resolve().parent
out = pathlib.Path(sys.argv[1])
ORDEN = ["ref", "ct", "simul", "ctsimul", "k5", "k5ti32", "k5ti16"]
pct = lambda v, q: sorted(v)[min(len(v) - 1, round(q * (len(v) - 1)))]


def trazas(path):
    res = []
    if not path.exists():
        return res
    for line in path.read_text().splitlines():
        t = json.loads(line)
        if t.get("warmup") or t.get("error") or not t.get("segments") or str(t.get("stream_id", "")).startswith("warmup"):
            continue
        res.append(t)
    return res


def juntar(base, glob):
    """Une OUT/.../<brazo>/<glob>/latency/traces.jsonl en <brazo>/latency/traces.jsonl (para t67_cuts)."""
    arms = [a for a in ORDEN if (base / a).is_dir()]
    for a in arms:
        dst = base / a / "latency"
        dst.mkdir(exist_ok=True)
        partes = sorted((base / a).glob(f"{glob}/latency/traces.jsonl"), key=lambda p: p.parent.parent.name.rjust(4))
        with open(dst / "traces.jsonl", "w") as fh:
            for p in partes:
                txt = p.read_text()
                fh.write(txt if txt.endswith("\n") else txt + "\n")
        if partes and not (dst / "run.json").exists() and (partes[0].parent / "run.json").exists():
            shutil.copy(partes[0].parent / "run.json", dst / "run.json")
    return arms


def eventos(t, kind):
    return [e for e in t["events"] if e["kind"] == kind]


def primero_desde_inicio(t):
    seg = t["segments"][0]
    tcs = [e["recv_ns"] for e in eventos(t, "tts_chunk")]
    if not tcs or seg.get("start_s") is None:
        return None
    return (min(tcs) - t["t0_ns"] - seg["start_s"] * 1e9) / 1e6


def largo(t):
    s = t["segments"][0]
    return (s.get("end_s") or 0) - (s.get("start_s") or 0)


print(f"# T-80 · sha {(out / 'sha.txt').read_text().strip() if (out / 'sha.txt').exists() else '?'}\n")
print((out / "arms.txt").read_text().replace("\n", "  \n") if (out / "arms.txt").exists() else "")

# ---- 1. cortes en charlas
print("\n## 1. Cortes falsos en charlas (114 turnos, audio sintético `say`, 1 pasada, con carga de la Mac)\n")
ch = out / "charlas"
arms = juntar(ch, "c*") if ch.is_dir() else []
arms = [a for a in arms if (ch / a / "latency" / "traces.jsonl").stat().st_size > 0]
if arms:
    r = subprocess.run([sys.executable, str(HERE / "t67_cuts.py")] + [str(ch / a / "latency") for a in arms],
                       capture_output=True, text=True)
    print(r.stdout.replace(str(ch) + "/", "").split("\n## Clips partidos")[0])
    print("Brazos (en el mismo orden): " + ", ".join(arms) + ". Corte falso = endpoint 150 ms o más antes del fin de habla de referencia.\n")
    print("| brazo | motivos de espera (unit_reason del endpoint) | eventos simul_clause |")
    print("|---|---|---|")
    for a in arms:
        ts = trazas(ch / a / "latency" / "traces.jsonl")
        mot = Counter(e.get("unit_reason") for t in ts for e in eventos(t, "endpoint"))
        sc = sum(len(eventos(t, "simul_clause")) for t in ts)
        print(f"| {a} | " + ", ".join(f"{k} {v}" for k, v in mot.most_common()) + f" | {sc} |")
    (out / "cortes.txt").write_text(r.stdout.replace(str(ch) + "/", ""))
    print("\nDetalle por clip: artifacts/t80/cortes.txt")
else:
    print("(sin trazas de charlas todavía)")

# ---- 2. latencia
print("\n## 2. Latencia desde el fin de habla (8 clips t67-clips, intercalado)\n")
lat = out / "latencia"
rs = lat / "RESUMEN.md"
if rs.exists():
    txt = rs.read_text().split("\n", 2)[2] if rs.read_text().startswith("#") else rs.read_text()
    print(txt)
    print("\nCon simul el primer audio puede salir antes del fin de habla: total negativo es válido (no se recorta).")
    print("En brazos simul, «MT» y «TTS 1er audio» del desglose usan el primer mt_final (puede ser una cláusula): mirar solo «total».")
else:
    print("(latencia sin terminar)")

# ---- 3. 1.er audio desde inicio de habla, clips > 5 s
print("\n## 3. 1.er audio desde el inicio de habla, clips de más de 5 s\n")
print("| fuente | brazo | n | p50 ms | p95 ms | p50 por clip (mín–máx) |")
print("|---|---|---|---|---|---|")
for nombre, base in (("t67-clips (reposo relativo)", lat), ("charlas (con carga)", ch)):
    if not base.is_dir():
        continue
    if base is lat:
        juntar(lat, "r*")
    for a in ORDEN:
        p = base / a / "latency" / "traces.jsonl"
        ts = [t for t in trazas(p) if largo(t) > 5.0]
        vs, por = [], defaultdict(list)
        for t in ts:
            v = primero_desde_inicio(t)
            if v is not None:
                vs.append(v)
                por[t["segments"][0]["item_id"]].append(v)
        if vs:
            meds = [st.median(x) for x in por.values()]
            print(f"| {nombre} | {a} | {len(vs)} | {st.median(vs):.0f} | {pct(vs, .95):.0f} | {min(meds):.0f}–{max(meds):.0f} |")

# ---- 3b. motivos de espera en latencia
print("\n## 3b. Motivos de espera en el banco de latencia\n")
print("| brazo | motivos (unit_reason) | simul_clause |")
print("|---|---|---|")
for a in ORDEN:
    ts = trazas(lat / a / "latency" / "traces.jsonl") if lat.is_dir() else []
    if ts:
        mot = Counter(e.get("unit_reason") for t in ts for e in eventos(t, "endpoint"))
        print(f"| {a} | " + ", ".join(f"{k} {v}" for k, v in mot.most_common()) + f" | {sum(len(eventos(t, 'simul_clause')) for t in ts)} |")

# ---- 4. términos
print("\n## 4. Términos con contexto (96 términos, 7 charlas, MT sola)\n")
print("ct no cambia el texto de la MT (solo la espera del endpoint), así que ct = ref y ctsimul = simul en este eval.\n")
print("| brazo | ctx0 | ctx200 | MT p50 ctx200 |")
print("|---|---|---|---|")
for a, d in (("ref (= ct si se corrió)", "ref"), ("simul (= ctsimul si se corrió)", "simul"), ("k5", "k5")):
    p = out / "terminos" / d / "summary.json"
    if p.exists():
        s = json.loads(p.read_text())["arms"]
        print(f"| {a} | {s['ctx0']['terminos_ok']}/{s['ctx0']['terminos']} ({s['ctx0']['pct']} %) | "
              f"{s['ctx200']['terminos_ok']}/{s['ctx200']['terminos']} ({s['ctx200']['pct']} %) | {s['ctx200']['mt_p50_ms']:.0f} ms |")
    else:
        print(f"| {a} | (sin medir) | | |")
