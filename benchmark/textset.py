"""Conjunto de texto (sin audio) para evaluar MT/TTS por separado del streaming.

Formato JSONL: ``item_id, split, dialect, text_es, reference_en, phenomena,
transcript_es_provenance, reference_en_provenance``. Sin audio: la latencia de
estos ítems es petición→primer output (modo ``text``) y no se mezcla con la de
streaming.
"""
from __future__ import annotations

import json
from pathlib import Path

from .contracts import SPLITS, ManifestError
from .replay import Segment
from .runner import BenchItem

PROVENANCE = ("human", "generated", "unknown")


def load_text_items(path: str | Path, *, split: str = "development", ids: list[str] | None = None) -> list[BenchItem]:
    items: list[BenchItem] = []
    seen: set[str] = set()
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        d = json.loads(line)
        for k in ("item_id", "split", "text_es"):
            if k not in d:
                raise ManifestError(f"línea {n}: falta {k}")
        if d["split"] not in SPLITS:
            raise ManifestError(f"línea {n}: split inválido {d['split']!r}")
        if d["item_id"] in seen:
            raise ManifestError(f"item_id duplicado: {d['item_id']}")
        seen.add(d["item_id"])
        if d["split"] != split or (ids is not None and d["item_id"] not in ids):
            continue
        ref = d.get("reference_en")
        refs = tuple([ref] if isinstance(ref, str) else ref) if ref else None
        for k in ("transcript_es_provenance", "reference_en_provenance"):
            if d.get(k, "unknown") not in PROVENANCE:
                raise ManifestError(f"línea {n}: {k} inválido")
        items.append(
            BenchItem(
                item_id=d["item_id"],
                recording_id=d["item_id"],
                pcm=b"",
                split=d["split"],
                text_es=d["text_es"],
                reference_en=refs,
                segment=Segment(
                    item_id=d["item_id"], start_s=None, end_s=None, boundary_method="unknown",
                    tags=tuple(d.get("phenomena", ())),
                    cluster=d["item_id"],  # cada frase es independiente
                    transcript_provenance=d.get("transcript_es_provenance", "unknown"),
                    reference_provenance=d.get("reference_en_provenance", "unknown"),
                ),
            )
        )
    if ids is not None:
        missing = [i for i in ids if i not in {x.item_id for x in items}]
        if missing:
            raise ManifestError(f"item_ids no encontrados en el split {split!r}: {missing}")
        items.sort(key=lambda x: ids.index(x.item_id))
    if not items:
        raise ManifestError(f"sin ítems del split {split!r} en {path}")
    return items
