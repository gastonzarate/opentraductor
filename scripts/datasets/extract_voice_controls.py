"""Controles de voz humana EN (FLEURS en_us, CC-BY-4.0) para anclar rangos de escucha/proxies.

Extrae N grabaciones humanas en inglés del parquet en_us/validation ya cacheado
(~/.cache/opentraductor/fleurs), con los mismos ids que la muestra es_419 del
manifest (así el texto coincide con ``reference_en``). Convierte a PCM16 mono 16 kHz
con el mismo método que fetch_fleurs_sample.py y escribe:

  data/audio/controls/fleurs_en_us_<id>.wav
  data/manifests/voice_controls.jsonl

No modifica manifests existentes ni usa test (validation = "development"). Son
CONTROLES de rango (anclas), NO ítems de evaluación del traductor ni voz de
clonación. Un solo hablante por grabación; el hablante no se identifica (FLEURS
publica género, no identidad).

Uso (pyarrow no está en pyproject; se usa efímero):
    uv run --with pyarrow python scripts/datasets/extract_voice_controls.py --n 4
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from benchmark import audio as A  # noqa: E402

CACHE = Path.home() / ".cache" / "opentraductor" / "fleurs" / "en_us_validation.parquet"
SAMPLE = REPO / "data" / "manifests" / "fleurs_es_sample.jsonl"
OUT_DIR = REPO / "data" / "audio" / "controls"
OUT_MANIFEST = REPO / "data" / "manifests" / "voice_controls.jsonl"
GENDER = {0: "male", 1: "female", 2: "other"}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    args = ap.parse_args()
    if not CACHE.exists():
        sys.exit(f"falta el parquet {CACHE}; ejecutar fetch_fleurs_sample.py primero")
    sample_ids = sorted(json.loads(l)["fleurs_id"] for l in SAMPLE.read_text().splitlines() if l.strip())
    first: dict[str, dict] = {}
    for r in pq.read_table(CACHE).to_pylist():
        first.setdefault(str(r["id"]), r)  # primera aparición, igual que la muestra
    chosen = [i for i in sample_ids if i in first][: args.n]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for fid in chosen:
        r = first[fid]
        src = r["audio"]["bytes"]
        data, sr = sf.read(io.BytesIO(src), dtype="float32", always_2d=True)
        if sr != A.TARGET_SR:
            raise ValueError(f"sample rate inesperado {sr}")
        mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
        pcm = np.clip(np.round(mono * 32767.0), -32768, 32767).astype("<i2").tobytes()
        rel = Path("controls") / f"fleurs_en_us_{fid}.wav"
        A.write_wav_pcm16_mono_16k(REPO / "data" / "audio" / rel, pcm)
        rows.append({
            "control_id": f"fleurs-en_us-validation-{fid}",
            "role": "human_en_control",
            "purpose": "ancla de rango para escucha ciega y proxies (UTMOSv2); NO es ítem de evaluación ni voz de clonación",
            "language": "en-US",
            "audio_path": rel.as_posix(),
            "sha256": sha256_file(REPO / "data" / "audio" / rel),
            "sample_rate": A.TARGET_SR,
            "channels": 1,
            "duration_s": round(A.pcm_duration_s(pcm), 3),
            "split": "development",
            "speaker_gender": GENDER.get(r.get("gender"), "unknown"),
            "speaker_id": None,
            "transcript_en": r["raw_transcription"],
            "transcript_en_provenance": "human",
            "license": "CC-BY-4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "source_url": "https://huggingface.co/datasets/google/fleurs",
            "attribution": "Conneau et al. (2022), FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech, google/fleurs, CC-BY-4.0",
            "fleurs_id": fid,
            "source_parquet": str(CACHE),
            "source_parquet_sha256": sha256_file(CACHE),
            "source_audio_sha256": hashlib.sha256(src).hexdigest(),
            "conversion": "sf.read float32 -> mono -> PCM16 little-endian (igual que fetch_fleurs_sample.py)",
            "listening_license_arg": "CC-BY-4.0 (Conneau et al. 2022, FLEURS en_us)",
        })
    OUT_MANIFEST.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
    for x in rows:
        print(x["control_id"], x["duration_s"], x["speaker_gender"], x["sha256"][:12])


if __name__ == "__main__":
    main()
