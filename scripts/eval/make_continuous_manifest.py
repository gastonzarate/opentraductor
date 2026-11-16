"""Construye el manifest de la receta continua: SOLO split development, ids explícitos y fijos.

Uso: .venv/bin/python scripts/eval/make_continuous_manifest.py [OUT.jsonl]
No toca el split test. Falla si algún id falta, no es development o no tiene límites anotados.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ["data/manifests/openslr61_es_ar_sample.jsonl", "data/manifests/fleurs_es_sample.jsonl"]
# 8 ítems: los 3 de offline-review + 5 más (4 es-AR de OpenSLR 61, 2 FLEURS). Orden de colocación lo decide el arnés (semilla).
IDS = [
    "openslr61-arf_02121_02074400931", "openslr61-arm_02484_01249298980", "fleurs-es_419-validation-1516",
    "openslr61-arf_02121_01122665566", "openslr61-arm_02484_01813972301", "openslr61-arm_07049_01264801258",
    "fleurs-es_419-validation-1549", "fleurs-es_419-validation-1612",
]


def main(out: str) -> None:
    by_id = {}
    for rel in SOURCES:
        for line in (ROOT / rel).read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            by_id[d["item_id"]] = d
    rows = []
    for i in IDS:
        d = by_id.get(i)
        assert d is not None, f"falta {i}"
        assert d["split"] == "development", f"{i} no es development"
        assert d.get("utterance_end_s") is not None, f"{i} sin límites anotados"
        assert (ROOT / "data/audio" / d["audio_path"]).exists(), f"falta audio de {i}"
        rows.append(d)
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    dur = sum(r["duration_s"] for r in rows)
    print(f"{len(rows)} ítems development -> {p} (audio {dur:.1f} s + huecos de 1.5 s entre ítems)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "artifacts/runtime-continuous/manifest.jsonl"))
