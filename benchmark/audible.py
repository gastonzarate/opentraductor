"""Puntuación a posteriori del texto AUDIBLE de WAV exportados.

Lee uno o más ``export_index.jsonl`` (verificándolos), transcribe cada WAV con un
ASR local independiente y compara: (a) con el texto planeado (índice o run.json) →
WER/omisión sospechada, evidencia de inteligibilidad del TTS; (b) con las
referencias del set → BLEU/chrF por procedencia. No usa los WAV para nada más.
El ASR es evidencia sesgada (puede corregir habla poco clara); ver docs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Sequence

from . import audio as A
from . import export as X
from . import quality


def score_exports(
    index_paths: Sequence[str | Path],
    asr: Callable[[bytes, int], str],
    *,
    refs: dict[str, tuple[Sequence[str] | None, str | None]] | None = None,
    planned_from_run: dict[str, str] | None = None,
    min_word_ratio: float = 0.8,
) -> dict:
    rows = []
    for ip in index_paths:
        r = X.read_index(ip)
        problems = X.verify_index(r)
        if problems:
            raise ValueError(f"índice {ip} con problemas: {problems}")
        rows.extend(r)
    items, skipped = [], []
    for r in rows:
        if r.get("continuous"):
            skipped.append({"stream_id": r["stream_id"], "reason": "stream continuo: sin atribución por ítem"})
            continue
        pcm, sr = A.read_wav_pcm16_mono(Path(r["_dir"]) / r["wav"])
        audible = asr(pcm, sr)
        planned = r.get("planned_text_en") or (planned_from_run or {}).get(r["stream_id"])
        row = {"system": r["system"], "stream_id": r["stream_id"], "item_id": r["item_ids"][0], "wav": r["wav"],
               "duration_s": r["duration_s"], "planned_text": planned, "audible_text": audible,
               "planned_text_source": "index" if r.get("planned_text_en") else ("run.json" if planned else None)}
        if planned:
            row["planned_vs_audible_wer"] = quality.wer(quality.normalize_text(planned), quality.normalize_text(audible))
            row["omission_suspected"] = len(audible.split()) < min_word_ratio * len(planned.split())
        else:
            row["planned_vs_audible_wer"] = quality.unavailable("sin texto planeado")
            row["omission_suspected"] = None
        items.append(row)
    by_sys: dict[str, dict] = {}
    for sysname in sorted({i["system"] for i in items}):
        sub = [i for i in items if i["system"] == sysname]
        planned_rows = [i for i in sub if i["planned_text"]]
        entry: dict = {"n": len(sub), "n_omission_suspected": sum(1 for i in sub if i["omission_suspected"])}
        if planned_rows:
            try:
                import jiwer

                entry["planned_vs_audible_wer_corpus"] = jiwer.wer(
                    [quality.normalize_text(i["planned_text"]) for i in planned_rows],
                    [quality.normalize_text(i["audible_text"]) for i in planned_rows])
            except ImportError:
                entry["planned_vs_audible_wer_corpus"] = quality.unavailable("jiwer no instalado")
        else:
            entry["planned_vs_audible_wer_corpus"] = quality.unavailable("sin texto planeado")
        if refs is not None:
            entry["audible_vs_reference"] = quality._by_provenance(
                [(i["audible_text"], *refs.get(i["item_id"], (None, None))) for i in sub])
        by_sys[sysname] = entry
    return {
        "asr_label": getattr(asr, "label", None),
        "asr_params": asr.describe() if hasattr(asr, "describe") else None,
        "n_scored": len(items), "skipped": skipped, "per_system": by_sys, "items": items,
        "caveats": [
            "ASR local independiente con sesgo propio: puede 'corregir' habla poco clara y alucinar con audio no inglés.",
            "WER planeado-vs-audible mide inteligibilidad relativa a ESE ASR, no calidad de la traducción.",
            "BLEU/chrF audible vs referencias es un proxy automático; no es MOS ni juicio humano.",
        ],
    }


def write_json(result: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
