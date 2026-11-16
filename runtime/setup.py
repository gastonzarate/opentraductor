"""Setup explícito de modelos (ÚNICO paso que usa red).

    uv run --extra runtime python -m runtime.setup

Descarga los pesos con revisión fija y convierte Marian a CTranslate2 int8.
Después, el runtime funciona sin red.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from . import config as C


def download() -> None:
    from huggingface_hub import snapshot_download

    print("ASR  Parakeet ...", flush=True)
    snapshot_download(C.PARAKEET_REPO, revision=C.PARAKEET_REV,
                      allow_patterns=["config.json", "model.safetensors", "tokenizer.*", "vocab.txt"])
    print("TTS  Kokoro ...", flush=True)
    snapshot_download(C.KOKORO_REPO, revision=C.KOKORO_REV,
                      allow_patterns=["config.json", "kokoro-v1_0.safetensors"]
                      + [f"voices/{v}.safetensors" for v in C.KOKORO_VOICES])
    print("MT   Marian (fuente) ...", flush=True)
    snapshot_download(C.MARIAN_REPO, revision=C.MARIAN_REV,
                      allow_patterns=["*.json", "*.spm", "pytorch_model.bin", "README.md"])


def convert_marian(force: bool = False) -> Path:
    import ctranslate2
    from ctranslate2.converters import TransformersConverter
    import importlib.metadata as md
    from huggingface_hub import snapshot_download

    out = C.marian_ct2_dir()
    if (out / "model.bin").exists() and not force:
        return out
    src = Path(snapshot_download(C.MARIAN_REPO, revision=C.MARIAN_REV, local_files_only=True,
                                 allow_patterns=["*.json", "*.spm", "pytorch_model.bin"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    TransformersConverter(str(src)).convert(str(out), quantization="int8", force=True)
    for f in ("source.spm", "target.spm", "vocab.json", "tokenizer_config.json"):
        (out / f).write_bytes((src / f).read_bytes())
    (out / "conversion.json").write_text(json.dumps({
        "source_repo": C.MARIAN_REPO, "source_revision": C.MARIAN_REV, "quantization": "int8",
        "ctranslate2": ctranslate2.__version__, "transformers": md.version("transformers"),
        "torch": md.version("torch"),
    }, indent=2), encoding="utf-8")
    return out


def main() -> int:
    from .config import PipelineConfig
    from .preflight import run_checks

    download()
    print("MT   convirtiendo a CTranslate2 int8 ...", flush=True)
    convert_marian()
    bad = [c for c in run_checks(PipelineConfig()) if not c.ok]
    for c in run_checks(PipelineConfig()):
        print(("OK   " if c.ok else "FALTA"), c.name, "-", c.detail)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
