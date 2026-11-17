#!/usr/bin/env python
"""T-58 (@bench): latencia de MT con mt_sense_glossary apagado vs encendido, en serie, mismo proceso y modelo.

Cada (frase, repeticion) mide los dos brazos back-to-back con orden alternado (off,on / on,off) para que la deriva
termica y de carga caiga igual en ambos. Dos modos:
  kv    como en el pipeline: prefill(parcial) -> espera WAIT_S (endpoint + finish del ASR, GPU ociosa) -> translate(final).
        Parcial = el final sin su ultima palabra (prefill descarta ademas la ultima del parcial).
  cold  translate(final) sin prefill (comparable con t58-mt-timing y mt_sentido_eval).
Sets: idiom = frases de mt_sentido_ar con algun termino de QWEN_GLOSSARY (llevan prompt mas largo);
      gate  = las 54 del gate MT (text_meetings_ar_dev) -> con 70bb991 el prompt es identico: control de ruido.

Uso: OT_MLX_CACHE_LIMIT_MB=256 .venv-inc-llm/bin/python scripts/eval/t58_mt_glossary_latency.py --model <dir> --out <dir>
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from runtime import components as K  # noqa: E402


def pct(xs, p):
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def one(q, final, mode, wait_s):
    words = final.split()
    if mode == "kv":
        q.prefill(" ".join(words[:-1]))
        time.sleep(wait_s)
    t = time.perf_counter()
    en = q.translate(final)
    ms = (time.perf_counter() - t) * 1000
    q._kv = None
    return ms, en


def run_set(q, name, texts, mode, reps, wait_s, rows):
    res = {"off": [], "on": []}
    for rep in range(reps):
        for i, es in enumerate(texts):
            order = ("off", "on") if (rep + i) % 2 == 0 else ("on", "off")
            for arm in order:
                q.glossary = arm == "on"
                ms, en = one(q, es, mode, wait_s)
                res[arm].append(ms)
                rows.write(json.dumps({"set": name, "mode": mode, "rep": rep, "arm": arm, "es": es, "en": en,
                                       "ms": round(ms, 1), "gloss": bool(K.qwen_glossary(es))}, ensure_ascii=False) + "\n")
    rows.flush()
    s = {a: {"n": len(v), "p50": round(statistics.median(v), 1), "p95": round(pct(v, 0.95), 1),
             "mean": round(statistics.mean(v), 1)} for a, v in res.items()}
    s["delta_p50"] = round(s["on"]["p50"] - s["off"]["p50"], 1)
    s["delta_p95"] = round(s["on"]["p95"] - s["off"]["p95"], 1)
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--gate-reps", type=int, default=3)
    ap.add_argument("--wait-s", type=float, default=0.3)
    a = ap.parse_args()
    import mlx.core as mx
    mx.set_cache_limit(int(float(os.environ.get("OT_MLX_CACHE_LIMIT_MB", "256")) * 1024 * 1024))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sent = [json.loads(line) for line in open(ROOT / "scripts/eval/sets/mt_sentido_ar.jsonl") if line.strip()]
    idiom = [it["es"] for it in sent if K.qwen_glossary(it["es"])]
    gate = [json.loads(line)["text_es"] if "text_es" in json.loads(line) else None
            for line in open(ROOT / "data/manifests/text_meetings_ar_dev.jsonl") if line.strip()]
    if any(g is None for g in gate):
        raise SystemExit("text_meetings_ar_dev: clave de texto desconocida " + repr(open(ROOT / "data/manifests/text_meetings_ar_dev.jsonl").readline()[:300]))
    q = K.QwenMLX(a.model)
    for es in idiom[:3] + gate[:3]:  # calentamiento de ambos brazos
        for g in (False, True):
            q.glossary = g
            one(q, es, "kv", 0.1)
            one(q, es, "cold", 0)
    env = {"commit": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                                    text=True).stdout.strip(), "uptime_start": subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip(),
           "n_idiom": len(idiom), "n_gate": len(gate), "gate_con_glosario": sum(bool(K.qwen_glossary(g)) for g in gate),
           "reps": a.reps, "gate_reps": a.gate_reps, "wait_s": a.wait_s,
           "cache_mb": os.environ.get("OT_MLX_CACHE_LIMIT_MB", "256"), "glossary_default": os.environ.get("OPENTRADUCTOR_MT_GLOSSARY")}
    summary = {"env": env}
    with open(out / "rows.jsonl", "w") as rows:
        for name, texts, mode, reps in (("idiom", idiom, "kv", a.reps), ("idiom", idiom, "cold", a.reps),
                                        ("gate", gate, "kv", a.gate_reps)):
            summary[f"{name}-{mode}"] = run_set(q, name, texts, mode, reps, a.wait_s, rows)
            print(name, mode, summary[f"{name}-{mode}"], flush=True)
    env["uptime_end"] = subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
