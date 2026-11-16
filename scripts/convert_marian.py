"""Convierte Marian es->en a CTranslate2 int8 EN EL BUILD (Mac de desarrollo).

    uv run --extra runtime --extra build python scripts/convert_marian.py --out <carpeta>

Necesita torch y transformers, que NO van en la app de tienda. El resultado
(``model.bin``, ``source.spm``, ``target.spm``, ...) se copia a Resources y el
shell lanza el runtime con ``OPENTRADUCTOR_MARIAN_CT2=<carpeta en Resources>``.
Sin ``--out`` usa ``runtime.config.marian_ct2_dir()``.

Descarga los pesos de Hugging Face con la revisión fijada (usa red; es del
build, no del dispositivo).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from runtime import setup as S  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, help="carpeta de salida (por defecto marian_ct2_dir())")
    ap.add_argument("--force", action="store_true", help="reconvertir aunque ya exista")
    a = ap.parse_args()
    S.download_marian_source()
    out = S.convert_marian(force=a.force, out=a.out)
    needed = ("model.bin", "source.spm", "target.spm")
    missing = [f for f in needed if not (out / f).exists()]
    if missing:
        print(f"FALTA {missing} en {out}", file=sys.stderr)
        return 1
    size = sum(f.stat().st_size for f in out.iterdir() if f.is_file())
    print(f"OK {out} ({size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
