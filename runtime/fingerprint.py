"""Huellas reproducibles de pesos, motores y código para ``describe()``."""
from __future__ import annotations

import hashlib
import importlib.metadata as md
from pathlib import Path

from . import config as C

ENGINE_PKGS = ("mlx", "mlx-audio", "parakeet-mlx", "ctranslate2", "torch", "transformers",
               "sentencepiece", "silero-vad", "onnxruntime", "misaki", "huggingface-hub", "numpy")


def code_hash() -> str:
    root = Path(__file__).parent
    h = hashlib.sha256()
    for p in sorted(root.rglob("*.py")):
        h.update(str(p.relative_to(root)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def _file_info(path: Path, full_hash: bool = False) -> dict:
    d = {"bytes": path.stat().st_size}
    if full_hash:
        d["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return d


def weights(cfg: C.PipelineConfig) -> dict:
    out: dict = {}
    if cfg.asr_backend == "parakeet":
        asr = C.snapshot(C.PARAKEET_REPO, C.PARAKEET_REV, ("config.json", "model.safetensors"))
        out["asr"] = {"repo": C.PARAKEET_REPO, "revision": C.PARAKEET_REV,
                      "model.safetensors": _file_info((asr / "model.safetensors").resolve())}
    else:
        d = C.moonshine_model_dir(cfg.asr_backend)
        out["asr"] = {"backend": cfg.asr_backend, "revision": C.MOONSHINE_REV, "dir": str(d),
                      "files": {f.name: _file_info(f, True) for f in sorted(d.glob("*")) if f.is_file()}}
    tts = C.snapshot(C.KOKORO_REPO, C.KOKORO_REV, ("config.json", "kokoro-v1_0.safetensors", f"voices/{cfg.voice}.safetensors"))
    out["tts"] = {"repo": C.KOKORO_REPO, "revision": C.KOKORO_REV, "voice": cfg.voice,
                  "kokoro-v1_0.safetensors": _file_info((tts / "kokoro-v1_0.safetensors").resolve()),
                  "voice_file": _file_info((tts / "voices" / f"{cfg.voice}.safetensors").resolve(), True)}
    d = C.marian_ct2_dir()
    out["mt"] = {"source_repo": C.MARIAN_REPO, "source_revision": C.MARIAN_REV, "ct2_dir": str(d),
                 "quantization": "int8", "model.bin": _file_info(d / "model.bin", True)}
    return out


def engines() -> dict:
    out = {}
    for p in ENGINE_PKGS:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return out
