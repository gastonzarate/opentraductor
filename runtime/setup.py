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


def _marian_ready() -> bool:
    return (C.marian_ct2_dir() / "model.bin").exists()


def download_marian_source() -> None:
    from huggingface_hub import snapshot_download

    print("MT   Marian (fuente) ...", flush=True)
    snapshot_download(C.MARIAN_REPO, revision=C.MARIAN_REV,
                      allow_patterns=["*.json", "*.spm", "pytorch_model.bin", "README.md"])


def download() -> None:
    """Descarga los modelos que el dispositivo necesita (Parakeet y Kokoro).

    Marian NO se baja aquí: en la edición de tienda viene ya convertido en el
    bundle (``scripts/convert_marian.py`` en el build). Solo en desarrollo, si
    falta, ``main`` lo baja y lo convierte.
    """
    from huggingface_hub import snapshot_download

    print("ASR  Parakeet ...", flush=True)
    snapshot_download(C.PARAKEET_REPO, revision=C.PARAKEET_REV,
                      allow_patterns=["config.json", "model.safetensors", "tokenizer.*", "vocab.txt"])
    print("TTS  Kokoro ...", flush=True)
    snapshot_download(C.KOKORO_REPO, revision=C.KOKORO_REV,
                      allow_patterns=["config.json", "kokoro-v1_0.safetensors"]
                      + [f"voices/{v}.safetensors" for v in C.KOKORO_VOICES])


def convert_marian(force: bool = False, out: Path | None = None) -> Path:
    """Convierte Marian a CTranslate2 int8. Necesita torch y transformers (solo
    en el Mac de build o de desarrollo; la app de tienda no los incluye)."""
    import importlib.metadata as md

    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Convertir Marian necesita torch y transformers, que la app de tienda no incluye. "
            "El build debe traer Marian convertido (scripts/convert_marian.py) y "
            "OPENTRADUCTOR_MARIAN_CT2 apuntando a esa carpeta."
        ) from exc
    import ctranslate2
    from ctranslate2.converters import TransformersConverter
    from huggingface_hub import snapshot_download

    out = out or C.marian_ct2_dir()
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
    if _marian_ready():
        print(f"MT   Marian CT2 ya presente en {C.marian_ct2_dir()}", flush=True)
    else:
        download_marian_source()
        print("MT   convirtiendo a CTranslate2 int8 ...", flush=True)
        convert_marian()
    checks = run_checks(PipelineConfig())
    for c in checks:
        print(("OK   " if c.ok else "FALTA"), c.name, "-", c.detail)
    return 1 if any(not c.ok for c in checks) else 0


if __name__ == "__main__":
    sys.exit(main())
