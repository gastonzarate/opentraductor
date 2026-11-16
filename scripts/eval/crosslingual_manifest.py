"""Manifest de texto (transcripción HUMANA) para el piloto interlingüe ES-AR → EN desde docs/crosslingual-voice-pilot.json.

Sin inferencia. Verifica hash de cada audio y que el split sea development (nunca held-out).
    .venv/bin/python scripts/eval/crosslingual_manifest.py --out artifacts/benchmark/crosslingual-pilot/text_manifest.jsonl
Evalúa MT + TTS/clonación con transcripción humana; NO es ASR ni latencia extremo a extremo.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def build(pilot_json: Path, out: Path) -> list[dict]:
    doc = json.loads(pilot_json.read_text(encoding="utf-8"))
    if doc["selection"].get("split") != "development":
        raise SystemExit("el piloto solo admite split development (nunca held-out)")
    rows = []
    for it in doc["items"]:
        audio = ROOT / it["audio_path"]
        if hashlib.sha256(audio.read_bytes()).hexdigest() != it["sha256"]:
            raise SystemExit(f"sha256 no coincide: {it['item_id']}")
        if it.get("transcript_es_provenance") != "human":
            raise SystemExit(f"{it['item_id']}: transcripción no humana")
        rows.append({"item_id": it["item_id"], "split": "development", "dialect": it.get("dialect_label", "unknown"),
                     "text_es": it["transcript_es"], "reference_en": None, "phenomena": [],
                     "transcript_es_provenance": "human", "reference_en_provenance": "unknown"})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", default=str(ROOT / "docs" / "crosslingual-voice-pilot.json"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = build(Path(a.pilot), Path(a.out))
    print(f"{len(rows)} ítems → {a.out}")
