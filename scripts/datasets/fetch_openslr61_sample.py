"""Muestra piloto de OpenSLR 61 (Crowdsourced Argentinian Spanish, CC-BY-SA 4.0).

Fuente primaria: https://openslr.org/61/ (Guevara-Rukoz et al., 2020).
Lee los índices line_index_{female,male}.tsv y el zip de cada género por HTTP
Range, sin descargar el zip completo. Elige con semilla declarada 4 hablantes
para development y 4 para test (2 mujeres + 2 hombres en cada split), 3 clips
por hablante. Convierte 48 kHz -> PCM16 mono 16 kHz con filtro sinc en numpy
(sin dependencias nuevas), escribe data/audio/openslr61/... y añade entradas al
manifest data/manifests/openslr61_es_ar_sample.jsonl. Speaker ID =
segundo campo del FileID (arf_<spk>_<utt> / arm_<spk>_<utt>).

Licencia: CC-BY-SA 4.0. Se mantiene en su propio manifest y en su propio
directorio. El audio no se mezcla con el de otras fuentes.

Uso (desde la raíz del repo):
    .venv/bin/python scripts/datasets/fetch_openslr61_sample.py --seed 20260514 --per-speaker 3
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import sys
import urllib.request
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from benchmark import audio as A  # noqa: E402
from benchmark import contracts  # noqa: E402
from benchmark.runner import items_from_manifest  # noqa: E402

BASE = "https://openslr.org/resources/61"
INDEX = {"female": f"{BASE}/line_index_female.tsv", "male": f"{BASE}/line_index_male.tsv"}
ZIPS = {"female": f"{BASE}/es_ar_female.zip", "male": f"{BASE}/es_ar_male.zip"}
LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"
SOURCE_URL = "https://openslr.org/61/"
CACHE = Path.home() / ".cache" / "opentraductor" / "openslr61"
AUDIO_ROOT = REPO / "data" / "audio"
MANIFEST = REPO / "data" / "manifests" / "openslr61_es_ar_sample.jsonl"
METADATA = REPO / "data" / "manifests" / "openslr61_es_ar_sample.metadata.json"
ATTRIBUTION = (
    "Guevara-Rukoz et al. (2020), Crowdsourcing Latin American Spanish for "
    "Low-Resource Text-to-Speech, LREC, OpenSLR 61, CC-BY-SA-4.0 (Google)"
)
SRC_SR = 48000
DST_SR = A.TARGET_SR


class HTTPRangeFile:
    """Objeto de solo lectura con seek sobre una URL que admite Range."""

    def __init__(self, url: str):
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            self._size = int(r.headers["Content-Length"])
        self.url = url
        self._pos = 0

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, off: int, whence: int = 0) -> int:
        base = {0: 0, 1: self._pos, 2: self._size}[whence]
        self._pos = max(0, base + off)
        return self._pos

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self._size - self._pos
        if n == 0 or self._pos >= self._size:
            return b""
        end = min(self._size - 1, self._pos + n - 1)
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self._pos}-{end}"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        self._pos += len(data)
        return data


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fetch_cached(url: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    dest = CACHE / url.rsplit("/", 1)[1]
    if not dest.exists():
        urllib.request.urlretrieve(url, dest)
    return dest


def read_index(path: Path, gender: str) -> dict[str, list[str]]:
    """speaker_id -> [FileID, ...] con transcripción. Guarda texto aparte."""
    by_spk: dict[str, list[str]] = defaultdict(list)
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            fid, _text = line.split("\t", 1)
            spk = fid.split("_")[1]
            by_spk[f"{gender}:{spk}"].append(fid)
    return by_spk


def read_text_map(path: Path) -> dict[str, str]:
    out = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.strip():
                fid, text = line.split("\t", 1)
                out[fid] = text
    return out


def resample_48k_to_16k(x: np.ndarray) -> np.ndarray:
    """Factor 3 exacto. Filtro pasa-bajos sinc con ventana de Hamming (corte 7 kHz)."""
    factor = SRC_SR // DST_SR
    cutoff = 7000.0 / (SRC_SR / 2)  # normalizado a Nyquist de la entrada
    half = 64
    t = np.arange(-half, half + 1)
    h = np.sinc(cutoff * t) * np.hamming(t.size)
    h = h / np.sum(h)
    y = np.convolve(x, h, mode="same")
    return y[::factor]


def to_pcm16_16k(wav_bytes: bytes) -> tuple[bytes, int]:
    data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float64", always_2d=True)
    if data.shape[1] > 1:
        data = data.mean(axis=1, keepdims=True)
    mono = data[:, 0]
    if sr == DST_SR:
        y = mono
    elif sr == SRC_SR:
        y = resample_48k_to_16k(mono)
    else:
        raise ValueError(f"sample rate no soportado: {sr}")
    pcm = np.clip(np.round(y * 32767.0), -32768, 32767).astype("<i2").tobytes()
    return pcm, int(sr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260514)
    ap.add_argument("--per-speaker", type=int, default=3)
    ap.add_argument("--min-utts", type=int, default=10)
    args = ap.parse_args()

    idx_files = {g: fetch_cached(u) for g, u in INDEX.items()}
    lic = fetch_cached(f"{BASE}/LICENSE")
    text_map: dict[str, str] = {}
    spk_map: dict[str, list[str]] = {}
    for g, p in idx_files.items():
        text_map.update(read_text_map(p))
        spk_map.update(read_index(p, g))
    eligible = {s: v for s, v in spk_map.items() if len(v) >= args.min_utts}

    rng = random.Random(args.seed)
    chosen_spk: dict[str, list[str]] = {"development": [], "test": []}
    for split in ("development", "test"):
        for g in ("female", "male"):
            pool = sorted(s for s in eligible if s.startswith(f"{g}:") and s not in sum(chosen_spk.values(), []))
            picks = rng.sample(pool, 2)
            chosen_spk[split].extend(picks)

    zips = {g: zipfile.ZipFile(HTTPRangeFile(u)) for g, u in ZIPS.items()}
    manifest_items = []
    for split, spks in chosen_spk.items():
        for spk in spks:
            g = spk.split(":")[0]
            fids = sorted(eligible[spk])
            picked = rng.sample(fids, args.per_speaker)
            for fid in picked:
                member = f"{fid}.wav"
                wav = zips[g].read(member)
                pcm, src_sr = to_pcm16_16k(wav)
                rel = Path("openslr61") / split / f"{fid}.wav"
                out = AUDIO_ROOT / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                A.write_wav_pcm16_mono_16k(out, pcm)
                converted = out.read_bytes()
                item_id = f"openslr61-{fid}"
                manifest_items.append(
                    {
                        "item_id": item_id,
                        "recording_id": item_id,
                        "source_id": "openslr61",
                        "speaker_id": spk.replace(":", "_"),  # género + id: evita colisiones entre géneros
                        "split": split,
                        "audio_path": rel.as_posix(),
                        "sha256": sha256_bytes(converted),
                        "sample_rate": DST_SR,
                        "channels": 1,
                        "duration_s": round(A.pcm_duration_s(pcm), 3),
                        "dialect": "es-AR",
                        "license": "CC-BY-SA-4.0",
                        "license_url": LICENSE_URL,
                        "source_url": SOURCE_URL,
                        "attribution": ATTRIBUTION,
                        "transcript_es": text_map[fid],
                        "transcript_es_provenance": "human",
                        "reference_en": None,
                        "reference_en_provenance": "unknown",
                        "utterance_start_s": None,
                        "utterance_end_s": None,
                        "boundary_method": "unknown",
                        "boundary_uncertainty_s": None,
                        "consent_note": "Voluntarios; la fuente no publica consentimiento individual en el índice",
                        "source_audio_sha256": sha256_bytes(wav),
                        "source_samplerate": src_sr,
                        "gender_from_fileid_prefix": g,
                    }
                )

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", encoding="utf-8") as fh:
        for it in manifest_items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")

    METADATA.write_text(
        json.dumps(
            {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "source": SOURCE_URL,
                "license": "CC-BY-SA-4.0",
                "license_text_sha256": sha256_bytes(lic.read_bytes()),
                "index_sha256": {g: sha256_bytes(p.read_bytes()) for g, p in idx_files.items()},
                "zip_urls": ZIPS,
                "seed": args.seed,
                "per_speaker": args.per_speaker,
                "min_utts": args.min_utts,
                "eligible_speakers": len(eligible),
                "chosen_speakers": chosen_spk,
                "split_policy": "speakers disjoint between development and test",
                "resampling": "48 kHz -> 16 kHz, FIR sinc 129 taps, hamming, corte 7 kHz, numpy",
                "boundary": "sin límites; boundary_method=unknown (se estima después)",
                "shareAlike_note": "CC-BY-SA-4.0: derivados (manifest, extractos) bajo el mismo tipo de licencia. Mantener separado de la licencia de la app.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    loaded = contracts.load_manifest(MANIFEST, audio_root=AUDIO_ROOT, verify_sha256=True)
    bench = items_from_manifest(loaded, AUDIO_ROOT)
    dev = [i for i in loaded if i.split == "development"]
    tst = [i for i in loaded if i.split == "test"]
    print(f"items {len(loaded)} (dev {len(dev)}, test {len(tst)}); runner OK {len(bench)}")
    print("speakers:", chosen_spk)
    print(f"duración total {sum(i.duration_s for i in loaded):.1f} s")


if __name__ == "__main__":
    main()
