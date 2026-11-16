"""Instala de forma aislada el candidato ASR Moonshine (streaming, español).

Hace solo esto:
1. Descarga el wheel fijado de moonshine-voice 0.1.6 (macOS arm64) desde PyPI y verifica su sha256.
2. Crea un venv propio en models/moonshine/venv e instala ese wheel. No toca .venv ni pyproject.toml.
3. Descarga los pesos streaming en español (tiny y small) con la CLI del paquete, a
   models/moonshine/assets, y verifica el sha256 de cada archivo.
4. Escribe models/moonshine/manifest.json con URLs, tamaños, hashes y la nota de licencia.

No carga modelos, no hace inferencia y no abre el micrófono. Es un paso de preparación,
no una evaluación. Ver docs/moonshine.md.

Uso desde la raíz del repo:
    python3 scripts/setup_moonshine.py
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "models" / "moonshine"
VENV = ROOT / "venv"
WHEEL_DIR = ROOT / "wheels"
ASSETS = ROOT / "assets"
CACHE = ROOT / "cache"
MANIFEST = ROOT / "manifest.json"

MOONSHINE_VERSION = "0.1.6"
WHEEL_NAME = "moonshine_voice-0.1.6-py3-none-macosx_15_0_arm64.whl"
WHEEL_URL = (
    "https://files.pythonhosted.org/packages/25/ad/"
    "91ef5908f592faff47dd4e03073b3eb073ce88f984f6bc86031e02057a69/" + WHEEL_NAME
)
WHEEL_SHA256 = "71f0fa9d0b01fa317f86a85e846e9faa247be985f92b50970817dd999e400ff4"
PY_VERSION = "3.12"
DOWNLOAD_BASE = "https://download.moonshine.ai/model"

# Arquitecturas según core/moonshine-c-api.h: 2 = TINY_STREAMING, 4 = SMALL_STREAMING.
MODELS = [
    {
        "arch": 2,
        "arch_name": "TINY_STREAMING",
        "params": "34 M",
        "name": "tiny-streaming-es",
        "revision": "quantized_26_08_24",
        "files": {
            "adapter.ort": ("2efd97cd2ee7d89578dd1d429a849be9fe37ac81dd9c4e1d31c8a2e7f85d43e6", 1318472),
            "cross_kv.ort": ("9e268b1b3bb0f9ff79bca6e76055d41298de0a5ba52505ee04907a2642efa3ae", 1288120),
            "decoder_kv.ort": ("7adf56b982df98e385d71823b9a40cf9296ae3a746efcf03f3d7bb4604032db0", 19717336),
            "encoder.ort": ("ec9eb43a01fbfe242a2c6a86050cffe4a31050e1c9f084e98560d6c100004505", 7772792),
            "frontend.model.ort": ("9d19434b861de0c1d2aff52556911ae565cf3212ebe6d7f797c76c5c5066c2fc", 23176),
            "frontend.weights.ort": ("5e4d5471cd5d984d4b94a603d861539ebf1dc72de11b0cf13f38316725d5ce48", 2093280),
            "streaming_config.json": ("d31afd7c2dd5ee9ae3fa232312c5b42cb3e4fe1c6e9c53d885f2bea63023790a", 509),
            "tokenizer.bin": ("5fbb7d4314dcb18e03c5f975609e4a4cd572b22b01d2b2accb1b3e6830696f36", 102888),
        },
    },
    {
        "arch": 4,
        "arch_name": "SMALL_STREAMING",
        "params": "123 M",
        "name": "small-streaming-es",
        "revision": "quantized_26_08_24",
        "files": {
            "adapter.ort": ("04b54114c8aab534222922640f7ca0882948ff9f6ed76777f6e184e55a8e8b15", 2869296),
            "cross_kv.ort": ("4bfd0a641d72ccdae22751f86bbb2e25ff70c1fba4f81213f2415a1accff9618", 5358752),
            "decoder_kv.ort": ("5b77c3d6baf801ef925a5bc54d7eb3db0c35ebbb86f8eaf5390d7f5bb42fef37", 61314512),
            "encoder.ort": ("a9b8d6d5d9348d0e319cceffdb0196ef622df8e4e9f3fb4797dcd8dfafd50857", 44358376),
            "frontend.model.ort": ("69c76287f49db365aa278d4908ec450e69ca1b4bfb7e836159d963d623ee0c13", 26776),
            "frontend.weights.ort": ("2f5a0eb5f3004c9447810d74274a352746319a32144e31d95ef410f425b84dda", 7769280),
            "streaming_config.json": ("12d16c7f5ea6734d197b79baf47914ea7e996d6fca8d303a19aca10d0617cecc", 512),
            "tokenizer.bin": ("5fbb7d4314dcb18e03c5f975609e4a4cd572b22b01d2b2accb1b3e6830696f36", 102888),
        },
    },
]

LICENSE_NOTE = (
    "Documentación del vendor (github.com/moonshine-ai/moonshine, docs/models/available-models.md "
    "y docs/license.md): los modelos streaming en español son MIT. El modelo Spanish Base "
    "(no streaming) es Community no comercial y NO se usa. La CLI imprime un aviso "
    "de Community License para cualquier idioma distinto de inglés; no es específico del modelo. "
    "Sin archivo de licencia por peso en el paquete ni en los assets. Pendiente de confirmación del coordinador."
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch_verified(url: str, dest: Path, sha256: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and sha256_file(dest) == sha256:
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    got = sha256_file(tmp)
    if got != sha256:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"sha256 no coincide para {dest.name}: esperado {sha256}, obtenido {got}")
    tmp.rename(dest)


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def ensure_venv() -> Path:
    py = VENV / "bin" / "python"
    if not py.exists():
        uv = shutil.which("uv")
        if uv:
            run([uv, "venv", str(VENV), "--python", PY_VERSION, "-q"])
        else:
            run([sys.executable, "-m", "venv", str(VENV)])
    wheel = WHEEL_DIR / WHEEL_NAME
    uv = shutil.which("uv")
    if uv:
        run([uv, "pip", "install", "--python", str(py), str(wheel)])
    else:
        run([str(py), "-m", "pip", "install", str(wheel)])
    return py


def download_models() -> None:
    cli = VENV / "bin" / "moonshine-voice"
    env = dict(os.environ, MOONSHINE_VOICE_CACHE=str(CACHE))
    for m in MODELS:
        run(
            [str(cli), "download", "--stt", "--language", "es",
             "--model-arch", str(m["arch"]), "--root", str(ASSETS)],
            env=env,
        )


def verify_models() -> list[dict]:
    report = []
    for m in MODELS:
        base = ASSETS / "download.moonshine.ai" / "model" / m["name"] / m["revision"]
        entries = []
        for fname, (sha, size) in sorted(m["files"].items()):
            path = base / fname
            if not path.exists():
                raise SystemExit(f"falta {path}")
            got = sha256_file(path)
            if got != sha:
                raise SystemExit(f"sha256 distinto en {path}: {got}")
            if path.stat().st_size != size:
                raise SystemExit(f"tamaño distinto en {path}")
            entries.append({
                "file": fname,
                "sha256": sha,
                "bytes": size,
                "url": f"{DOWNLOAD_BASE}/{m['name']}/{m['revision']}/{fname}",
            })
        report.append({
            "arch": m["arch"],
            "arch_name": m["arch_name"],
            "params": m["params"],
            "name": m["name"],
            "revision": m["revision"],
            "path": str(base.relative_to(REPO)),
            "files": entries,
        })
    return report


def main() -> None:
    if sys.platform != "darwin":
        raise SystemExit("este script está preparado para macOS arm64")
    for d in (ROOT, WHEEL_DIR, ASSETS, CACHE):
        d.mkdir(parents=True, exist_ok=True)

    fetch_verified(WHEEL_URL, WHEEL_DIR / WHEEL_NAME, WHEEL_SHA256)
    ensure_venv()
    download_models()
    models = verify_models()

    MANIFEST.write_text(
        json.dumps({
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "package": {
                "name": "moonshine-voice",
                "version": MOONSHINE_VERSION,
                "wheel": WHEEL_NAME,
                "wheel_sha256": WHEEL_SHA256,
                "wheel_url": WHEEL_URL,
                "license": "MIT (metadatos de PyPI)",
            },
            "venv": str(VENV.relative_to(REPO)),
            "license_note": LICENSE_NOTE,
            "models": models,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"listo: {MANIFEST.relative_to(REPO)}")


if __name__ == "__main__":
    main()
