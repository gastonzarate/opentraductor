"""Muestra piloto reproducible de FLEURS (google/fleurs, CC-BY-4.0).

Selecciona N utterances de es_419/validation con id paralelo en en_us/validation,
con semilla declarada. Convierte cada WAV original a PCM16 mono 16 kHz, escribe
data/audio/fleurs/..., genera el manifest data/manifests/fleurs_es_sample.jsonl
y la metadata data/manifests/fleurs_es_sample.metadata.json. Valida con
benchmark.contracts.load_manifest (sha256) y benchmark.runner.items_from_manifest.

Uso desde la raíz del repo (pyarrow no está en pyproject; se usa efímero):
    uv run --with pyarrow python scripts/datasets/fetch_fleurs_sample.py --n 10 --seed 20260514

Los parquet se cachean en ~/.cache/opentraductor/fleurs (~545 MB) y no se
vuelven a descargar si el cache existe. Split: validation -> "development".
Test queda sin tocar.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from benchmark import audio as A  # noqa: E402
from benchmark import contracts  # noqa: E402
from benchmark.runner import items_from_manifest  # noqa: E402

DATASET = "google/fleurs"
PARQUET_URL = "https://huggingface.co/api/datasets/google/fleurs/parquet/{config}/validation/0.parquet"
DATASET_API = "https://huggingface.co/api/datasets/google/fleurs"
CACHE = Path.home() / ".cache" / "opentraductor" / "fleurs"
AUDIO_ROOT = REPO / "data" / "audio"
MANIFEST = REPO / "data" / "manifests" / "fleurs_es_sample.jsonl"
METADATA = REPO / "data" / "manifests" / "fleurs_es_sample.metadata.json"
SOURCE_URL = "https://huggingface.co/datasets/google/fleurs"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
ATTRIBUTION = (
    "Conneau et al. (2022), FLEURS: Few-shot Learning Evaluation of Universal "
    "Representations of Speech, google/fleurs, CC-BY-4.0"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch_parquet(config: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    dest = CACHE / f"{config}_validation.parquet"
    if not dest.exists():
        url = PARQUET_URL.format(config=config)
        print(f"descargando {url} -> {dest}", flush=True)
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dest)
    return dest


def dataset_revision() -> dict:
    with urllib.request.urlopen(DATASET_API, timeout=60) as resp:
        meta = json.load(resp)
    return {"commit_sha": meta.get("sha"), "last_modified": meta.get("lastModified")}


def load_rows(path: Path) -> tuple[dict[str, dict], int]:
    rows = pq.read_table(path).to_pylist()
    by_id: dict[str, dict] = {}
    dups = 0
    for r in rows:
        key = str(r["id"])
        if key in by_id:
            # Mismo id = misma frase leída por otra voz (audio distinto, transcripción igual).
            # Se conserva la primera aparición del parquet; los demás no se usan.
            dups += 1
            continue
        by_id[key] = r
    return by_id, dups


def to_pcm16_mono_16k(wav_bytes: bytes) -> tuple[bytes, dict]:
    data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=True)
    src_info = {"source_samplerate": int(sr), "source_channels": int(data.shape[1])}
    if sr != A.TARGET_SR:
        raise ValueError(f"sample rate inesperado {sr}; FLEURS se esperaba 16 kHz")
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    pcm16 = np.clip(np.round(mono * 32767.0), -32768, 32767).astype("<i2")
    src_info["conversion"] = "sf.read float32 -> mono -> PCM16 little-endian"
    return pcm16.tobytes(), src_info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260514)
    args = ap.parse_args()

    es_path = fetch_parquet("es_419")
    en_path = fetch_parquet("en_us")
    parquet_info = {
        "es_419_validation": {
            "file": str(es_path),
            "sha256": sha256_file(es_path),
        },
        "en_us_validation": {
            "file": str(en_path),
            "sha256": sha256_file(en_path),
        },
    }
    revision = dataset_revision()
    print("parquet:", json.dumps(parquet_info, ensure_ascii=False))
    print("revisión del dataset:", revision)

    es_rows, es_dups = load_rows(es_path)
    en_rows, en_dups = load_rows(en_path)
    shared = sorted(set(es_rows) & set(en_rows))
    rng = random.Random(args.seed)
    chosen = sorted(rng.sample(shared, args.n))
    print(
        f"ids es_419={len(es_rows)} (grabaciones extra no usadas {es_dups}) "
        f"en_us={len(en_rows)} (grabaciones extra no usadas {en_dups}) "
        f"paralelos={len(shared)} seleccionados={len(chosen)} seed={args.seed}"
    )

    items = []
    for fid in chosen:
        es, en = es_rows[fid], en_rows[fid]
        src_bytes = es["audio"]["bytes"]
        pcm, conv = to_pcm16_mono_16k(src_bytes)
        out_rel = Path("fleurs") / "es_419" / "validation" / f"{fid}.wav"
        out = AUDIO_ROOT / out_rel
        out.parent.mkdir(parents=True, exist_ok=True)
        A.write_wav_pcm16_mono_16k(out, pcm)
        converted_sha = sha256_file(out)
        duration = A.pcm_duration_s(pcm)
        item_id = f"fleurs-es_419-validation-{fid}"
        items.append(
            {
                "item_id": item_id,
                "recording_id": item_id,
                "source_id": "fleurs",
                "speaker_id": None,
                "split": "development",
                "audio_path": out_rel.as_posix(),
                "sha256": converted_sha,
                "sample_rate": A.TARGET_SR,
                "channels": 1,
                "duration_s": round(duration, 3),
                "dialect": "unknown",
                "license": "CC-BY-4.0",
                "license_url": LICENSE_URL,
                "source_url": SOURCE_URL,
                "attribution": ATTRIBUTION,
                "transcript_es": es["raw_transcription"],
                "transcript_es_provenance": "human",
                "reference_en": [en["raw_transcription"]],
                "reference_en_provenance": "human",
                "utterance_start_s": None,
                "utterance_end_s": None,
                "boundary_method": "unknown",
                "boundary_uncertainty_s": None,
                "consent_note": None,
                "fleurs_id": fid,
                "source_audio_sha256": hashlib.sha256(src_bytes).hexdigest(),
                "conversion": conv,
                "reference_note": (
                    "en_us con mismo id FLEURS; texto paralelo humano, "
                    "no garantizado como traducción directa del audio es_419"
                ),
            }
        )

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")

    METADATA.write_text(
        json.dumps(
            {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "dataset": DATASET,
                "dataset_revision": revision,
                "parquet": parquet_info,
                "selection": {
                    "method": "random.Random(seed).sample sobre ids paralelos ordenados",
                    "seed": args.seed,
                    "n": args.n,
                    "parallel_ids": len(shared),
                    "es_extra_recordings_not_used": es_dups,
                    "en_extra_recordings_not_used": en_dups,
                    "selected_ids": chosen,
                },
                "split_policy": "validation -> development; test no usado",
                "speaker_id": "no disponible en la versión HF; agrupación por fuente",
                "boundary": "sin límites; boundary_method=unknown",
                "license": "CC-BY-4.0",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    loaded = contracts.load_manifest(MANIFEST, audio_root=AUDIO_ROOT, verify_sha256=True)
    bench_items = items_from_manifest(loaded, AUDIO_ROOT)
    print(
        f"manifest OK: {len(loaded)} items, benchmark.runner.items_from_manifest OK "
        f"({len(bench_items)} items) -> {MANIFEST}"
    )
    print(f"duración total: {sum(i.duration_s for i in loaded):.1f} s")


if __name__ == "__main__":
    main()
