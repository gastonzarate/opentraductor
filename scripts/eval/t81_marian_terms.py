"""T-81: % de términos de Marian (CTranslate2 int8, sin contexto) en el mismo set y con el mismo evaluador que
t69_charlas_eval.py (term_ok). Marian no usa contexto, así que ctx200 = ctx0. Con --simul traduce por cláusulas
como el Engine con MT_SIMUL=1. Uso: python scripts/eval/t81_marian_terms.py --set S --out DIR [--simul]"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.environ.get("T69_CODE_ROOT", str(ROOT)))
sys.path.insert(0, str(ROOT / "scripts/eval"))
from t69_charlas_eval import norm_charla, simul_split, term_ok  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--simul", action="store_true")
    a = ap.parse_args()
    from runtime import components as K  # noqa: E402
    from runtime.config import PipelineConfig  # noqa: E402
    from runtime.simul import join_pieces  # noqa: E402

    m = K.MarianCT2(normalize=PipelineConfig().mt_normalize)
    m.translate("hola")
    charlas = [norm_charla(json.loads(x)) for x in open(a.set) if x.strip()]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    hits = total = 0
    ms: list[float] = []
    with open(out / "rows.jsonl", "w") as f:
        for ch in charlas:
            for i, turn in enumerate(ch["turnos"]):
                t0 = time.perf_counter()
                if a.simul:
                    en = join_pieces([(p, m.translate(p)) for p in simul_split(turn["es"])])
                else:
                    en = m.translate(turn["es"])
                ms.append((time.perf_counter() - t0) * 1000)
                res = [term_ok(en, t) for t in turn["terms"]]
                hits += sum(res)
                total += len(res)
                f.write(json.dumps({"charla": ch["id"], "turno": i, "es": turn["es"], "en": en,
                                    "terms": [{"t": t["t"], "ok": r} for t, r in zip(turn["terms"], res)]},
                                   ensure_ascii=False) + "\n")
    s = {"arms": {"ctx200": {"terminos_ok": hits, "terminos": total, "pct": round(100 * hits / total, 1),
                             "mt_p50_ms": round(statistics.median(ms), 1)}}, "simul": a.simul}
    (out / "summary.json").write_text(json.dumps(s, indent=2, ensure_ascii=False))
    print(json.dumps(s, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
