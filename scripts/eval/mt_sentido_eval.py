#!/usr/bin/env python
"""Aciertos de sentido de Qwen en frases ambiguas argentinas (scripts/eval/sets/mt_sentido_ar.jsonl).

Brazos: base (prompt actual, sin contexto), sense (QWEN_SENSE), ctx (sin QWEN_SENSE, frase previa como turno),
sense_ctx (ambos). Acierto = algún regex de ``must`` aparece y ninguno de ``must_not`` (sin distinguir mayúsculas).
También mide la MT de cada llamada (sin prefill KV, igual en todos los brazos) para ver cuánto suma el contexto.

Uso:
  .venv-inc-llm/bin/python scripts/eval/mt_sentido_eval.py \
      --model artifacts/analysis/llm-mp4-a/models/qwen3-4b-instruct-2507-4bit --out artifacts/t58-sentido
"""
import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from runtime import components as K  # noqa: E402

ARMS = {"base": (False, False), "sense": (True, False), "ctx": (False, True), "sense_ctx": (True, True)}


def score(en: str, item: dict) -> bool:
    hit = any(re.search(p, en, re.I) for p in item["must"])
    bad = any(re.search(p, en, re.I) for p in item["must_not"])
    return hit and not bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--set", default=str(ROOT / "scripts/eval/sets/mt_sentido_ar.jsonl"))
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    items = [json.loads(line) for line in open(a.set) if line.strip()]
    q = K.QwenMLX(a.model)
    q.translate("hola")  # calentamiento
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    with open(out / "rows.jsonl", "w") as f:
        for arm in a.arms.split(","):
            sense, ctx = ARMS[arm]
            q.system = K.QWEN_SYSTEM + (K.QWEN_SENSE if sense else "")
            hits, ms = 0, []
            for it in items:
                context = ((it["prev_es"], it["prev_en"]),) if ctx else ()
                t = time.perf_counter()
                en = q.translate(it["es"], context)
                ms.append((time.perf_counter() - t) * 1000)
                ok = score(en, it)
                hits += ok
                f.write(json.dumps({"arm": arm, "id": it["id"], "es": it["es"], "en": en, "ok": ok,
                                    "ms": round(ms[-1], 1)}, ensure_ascii=False) + "\n")
            summary[arm] = {"hits": hits, "n": len(items), "mt_p50_ms": round(statistics.median(ms), 1)}
            print(arm, summary[arm], flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
