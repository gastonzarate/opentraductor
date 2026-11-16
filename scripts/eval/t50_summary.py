"""T-50: tabla de los tamaños de Qwen a partir de los JSON de t50_pipeline_ram.py.

BLEU/chrF de las 54 frases (text_meetings_ar_dev) contra reference_en, frases cuya traducción difiere del 4B
(para revisar a mano el sentido), RAM pico (phys_footprint) frente al criterio de 8 GB (≤ 3,5 GB), swap y latencia.
  .venv/bin/python scripts/eval/t50_summary.py artifacts/t50/ram > artifacts/t50/RESULTS.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import sacrebleu

ROOT = Path(__file__).resolve().parents[2]
TXT = ROOT / "data" / "manifests" / "text_meetings_ar_dev.jsonl"
LIMIT_MB = 3500
ORDER = ("qwen3-4b", "qwen3-1.7b", "qwen3-0.6b")


def main(d: Path) -> None:
    refs = {json.loads(x)["item_id"]: json.loads(x) for x in TXT.read_text().splitlines()}
    res = {n: json.loads((d / f"{n}.json").read_text()) for n in ORDER if (d / f"{n}.json").exists()}
    hyp = {n: {r["id"]: r["en"] for r in v["rows"] if r["id"] in refs} for n, v in res.items()}
    ids = sorted(refs)
    print("| modelo | BLEU | chrF | footprint pico MB | ≤3,5 GB | swap Δ MB | pageouts Δ | MT p50/p95 ms | TTS 1.º p50 ms | sesión s |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for n, v in res.items():
        h = [hyp[n].get(i, "") for i in ids]
        r = [[refs[i]["reference_en"] for i in ids]]
        fp = v.get("footprint_peak_mb")
        print(f"| {n} | {sacrebleu.corpus_bleu(h, r).score:.1f} | {sacrebleu.corpus_chrf(h, r).score:.1f} | {fp} | "
              f"{'sí' if fp is not None and fp <= LIMIT_MB else 'no'} | {v['swap_delta_mb']:.0f} | {v['pageouts_delta']} | "
              f"{v['mt_ms']['p50']:.0f}/{v['mt_ms']['p95']:.0f} | {v['tts_first_ms']['p50']:.0f} | {v.get('session_s')} |")
    if "qwen3-4b" not in hyp:
        return
    for n in ORDER[1:]:
        if n not in hyp:
            continue
        diff = [i for i in ids if hyp[n].get(i, "").strip().lower() != hyp["qwen3-4b"].get(i, "").strip().lower()]
        print(f"\n### {n}: {len(diff)} de {len(ids)} frases distintas del 4B (revisar sentido a mano)\n")
        print("| id | ES | 4B | " + n + " |\n|---|---|---|---|")
        for i in diff:
            print(f"| {i} | {refs[i]['text_es']} | {hyp['qwen3-4b'].get(i, '')} | {hyp[n].get(i, '')} |")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/t50/ram"))
