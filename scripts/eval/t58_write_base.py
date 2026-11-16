#!/usr/bin/env python3
"""[bench] T-58: escribe scripts/eval/store_baseline.json con las métricas de artifacts/t58 (umbral = base +10 %).
Uso: python3 scripts/eval/t58_write_base.py artifacts/t58 --qwen-dir DIR [--asr moonshine-tiny] [--reps 7] [--dry]
Usa los mismos extractores que gate.py, así que lo que fija como base es lo que el gate luego compara."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--qwen-dir", required=True); ap.add_argument("--asr", default="moonshine-tiny")
ap.add_argument("--reps", type=int, default=7); ap.add_argument("--margin", type=float, default=0.10); ap.add_argument("--dry", action="store_true")
a = ap.parse_args()
out = Path(a.out)
lat, mt = gate.load(out / "latency/store/run.json"), gate.load(out / "mt/run.json")
vals = {"latency_p50_ms": gate.find_quantile(lat, "p50"), "latency_p95_ms": gate.find_quantile(lat, "p95"),
        "asr_wer": gate.find_key(lat, "wer") or gate.find_key(lat, "wer_corpus"), "mt_p95_ms": gate.find_quantile(mt, "p95")}
res = {"config": {"asr_backend": a.asr, "mt_engine": "qwen", "qwen_model_dir": a.qwen_dir}, "python": ".venv-inc-llm/bin/python",
       "latency_reps": a.reps, "mlx_cache_limit_mb": 256}
for k, v in vals.items():
    if v is None:
        sys.exit(f"falta {k} en run.json")
    base = round(v[1], 3)
    # WER: con 3 clips es ruidoso; piso de umbral 0,06 (el del estándar) para no fallar por un solo error de palabra.
    thr = max(round(base * (1 + a.margin), 3), 0.06) if k == "asr_wer" else round(base * (1 + a.margin), 1)
    res[k] = {"base": base, "threshold": thr, "path": v[0]}
# conserva el piso BLEU/chrF por tamaño de Qwen que ya esté en el archivo (no sale de run.json de latencia)
try:
    old = json.loads(gate.STORE_BASE_FILE.read_text(encoding="utf-8"))
    if "bleu_chrf_min" in old:
        res["bleu_chrf_min"] = old["bleu_chrf_min"]
except (OSError, ValueError):
    pass
print(json.dumps(res, indent=1, ensure_ascii=False))
if not a.dry:
    gate.STORE_BASE_FILE.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("escrito", gate.STORE_BASE_FILE)
