#!/usr/bin/env python
"""T-69 (@bench): % de términos bien traducidos en charlas con contexto (scripts/eval/sets/charlas_ctx_ar.jsonl).

Cada charla se traduce turno a turno con Qwen, como en el pipeline: los últimos N turnos (es, en propio) van de contexto.
Brazos = valores de N (0 = sin contexto). Los flags del MT se toman del entorno, igual que en la app:
OPENTRADUCTOR_MT_GLOSSARY=0/1, OPENTRADUCTOR_MT_SENSE=0/1, OPENTRADUCTOR_MT_SESSION_KV=0/1.
Un término acierta si ``must`` aparece y ``bad`` no. Mide también la MT por turno (sin prefill KV: orientativo, no es end_to_first).

Uso:
  .venv-inc-llm/bin/python scripts/eval/t69_charlas_eval.py --out artifacts/t69-charlas/base [--ctx 0,6] [--set ...]
  OPENTRADUCTOR_MT_GLOSSARY=1 .venv-inc-llm/bin/python scripts/eval/t69_charlas_eval.py --out artifacts/t69-charlas/gloss
"""
import argparse
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.environ.get("T69_CODE_ROOT", str(ROOT)))  # T69_CODE_ROOT: carpeta con el commit a medir (git archive)
DEFAULT_MODEL = ROOT / "artifacts/analysis/llm-mp4-a/models/qwen3-4b-instruct-2507-4bit"


def term_ok(en: str, term: dict) -> bool:
    return bool(re.search(term["must"], en, re.I)) and not (term["bad"] and re.search(term["bad"], en, re.I))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("OPENTRADUCTOR_QWEN_DIR", str(DEFAULT_MODEL)))
    ap.add_argument("--set", default=str(ROOT / "scripts/eval/sets/charlas_ctx_ar.jsonl"))
    ap.add_argument("--ctx", default="0,6", help="turnos de contexto por brazo, separados por coma (0 = sin contexto)")
    ap.add_argument("--first-token", action="store_true",
                    help="usa translate_stream (mismo texto que translate) y mide ms hasta el 1.er token de la MT")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from runtime import components as K  # noqa: E402

    charlas = [json.loads(line) for line in open(a.set) if line.strip()]
    q = K.QwenMLX(a.model)
    q.translate("hola")  # calentamiento
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"flags": {k: v for k, v in os.environ.items() if k.startswith("OPENTRADUCTOR_MT_")},
               "set": a.set, "arms": {}}
    with open(out / "rows.jsonl", "w") as f:
        for n in [int(x) for x in a.ctx.split(",")]:
            arm = f"ctx{n}"
            hits = total = 0
            ms: list[float] = []
            ft: list[float] = []
            by_charla: dict[str, list[int]] = {}
            for ch in charlas:
                hist: list[tuple[str, str]] = []
                h = t_ = 0
                for i, turn in enumerate(ch["turnos"]):
                    context = tuple(hist[-n:]) if n > 0 else ()
                    t0 = time.perf_counter()
                    if a.first_token:
                        first: list[float] = []
                        en = q.translate_stream(turn["es"], context,
                                                lambda _p: first.append(time.perf_counter()) if not first else None)
                        if first:
                            ft.append((first[0] - t0) * 1000)
                    else:
                        en = q.translate(turn["es"], context)
                    ms.append((time.perf_counter() - t0) * 1000)
                    hist.append((turn["es"], en))
                    res = [term_ok(en, t) for t in turn["terms"]]
                    h += sum(res)
                    t_ += len(res)
                    f.write(json.dumps({"arm": arm, "charla": ch["id"], "turno": i, "es": turn["es"], "en": en,
                                        "terms": [{"t": t["t"], "ok": r} for t, r in zip(turn["terms"], res)],
                                        "ms": round(ms[-1], 1)}, ensure_ascii=False) + "\n")
                by_charla[ch["id"]] = [h, t_]
                hits += h
                total += t_
            summary["arms"][arm] = {"terminos_ok": hits, "terminos": total, "pct": round(100 * hits / total, 1),
                                    "por_charla": {k: f"{v[0]}/{v[1]}" for k, v in by_charla.items()},
                                    "mt_p50_ms": round(statistics.median(ms), 1),
                                    "mt_p95_ms": round(sorted(ms)[int(0.95 * (len(ms) - 1))], 1), "turnos": len(ms),
                                    "loadavg_fin": os.getloadavg()[0]}
            if ft:
                summary["arms"][arm]["mt_1er_token_p50_ms"] = round(statistics.median(ft), 1)
                summary["arms"][arm]["mt_1er_token_p95_ms"] = round(sorted(ft)[int(0.95 * (len(ft) - 1))], 1)
            print(arm, json.dumps(summary["arms"][arm], ensure_ascii=False), flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
