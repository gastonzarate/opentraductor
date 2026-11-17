#!/usr/bin/env python3
"""T-69: tabla A/B desde la salida de scripts/eval/t69_bench.sh (OUTDIR/<brazo>/{stages.md,charlas/summary.json,reposo.txt}).
Uso: python3 scripts/eval/t69_summary.py OUTDIR"""
import json
import re
import sys
from pathlib import Path

out = Path(sys.argv[1])
print(f"# T-69 resumen · {out.name}\n")
env = out / "entorno.txt"
if env.exists():
    print("```\n" + "\n".join(env.read_text().splitlines()[:8]) + "\n```\n")
print("## Latencia por etapa (ms, mediana; n = 3 clips × REPS)\n")
print("| brazo | reposo | n | espera fin de habla | ASR final | MT | TTS 1er audio | **total p50** | p95 total | load ini→fin |")
print("|---|---|---|---|---|---|---|---|---|---|")
for d in sorted(p for p in out.iterdir() if p.is_dir()):
    s = d / "stages.md"
    if not s.exists():
        continue
    txt = s.read_text()
    m = re.search(r"^\| todas \| (\d+) \| (\d+) \| (\d+) \| (\d+) \| (\d+) \| (\d+) \|", txt, re.M)
    p95 = re.search(r"p95 todas:.*total (\d+)", txt)
    rep = (d / "reposo.txt").read_text() if (d / "reposo.txt").exists() else ""
    quiet = "NO REPOSO" if "NO REPOSO" in rep else ("reposo" if "reposo:" in rep else "?")
    ld = lambda f: (d / f).read_text().split("load averages:")[-1].split()[0] if (d / f).exists() else "?"  # noqa: E731
    if m:
        print(f"| {d.name} | {quiet} | {m[1]} | {m[2]} | {m[3]} | {m[4]} | {m[5]} | **{m[6]}** | {p95[1] if p95 else '?'} | {ld('load-start.txt')}→{ld('load-end.txt')} |")
    else:
        print(f"| {d.name} | {quiet} | sin datos | | | | | | | |")
print("\n## Términos bien traducidos en charlas con contexto\n")
print("| brazo | contexto | % términos | ok/total | por charla | MT p50 ms | MT 1er token p50 ms |")
print("|---|---|---|---|---|---|---|")
for d in sorted(p for p in out.iterdir() if p.is_dir()):
    s = d / "charlas" / "summary.json"
    if not s.exists():
        continue
    for arm, v in json.loads(s.read_text())["arms"].items():
        pc = " ".join(f"{k} {x}" for k, x in v["por_charla"].items())
        print(f"| {d.name} | {arm.replace('ctx', '')} turnos | **{v['pct']}** | {v['terminos_ok']}/{v['terminos']} | {pc} | {v['mt_p50_ms']} | {v.get('mt_1er_token_p50_ms', '-')} |")
