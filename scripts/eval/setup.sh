#!/usr/bin/env bash
# Entorno AISLADO de evaluación (UTMOSv2 + faster-whisper, CPU). No toca .venv del runtime.
# Todo queda bajo models/eval/ (ignorado por el proyecto): venv, pesos, caché de Hugging Face.
# Uso: scripts/eval/setup.sh            (idempotente; verifica sha256 de pesos y fija revisiones)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
EVAL="$ROOT/models/eval"
VENV="$EVAL/venv"
export HF_HOME="$EVAL/hf"
mkdir -p "$EVAL/utmosv2" "$HF_HOME"

command -v uv >/dev/null || { echo "falta uv" >&2; exit 1; }
[ -x "$VENV/bin/python" ] || uv venv --python 3.12 "$VENV"
VIRTUAL_ENV="$VENV" uv pip install -r "$ROOT/scripts/eval/requirements-eval.txt"

# Pesos UTMOSv2: descarga explícita (el instalador del paquete usa wget, ausente en macOS) + sha256.
"$VENV/bin/python" - "$ROOT" <<'PY'
import hashlib, json, subprocess, sys
from pathlib import Path
root = Path(sys.argv[1])
cfg = json.loads((root / "scripts/eval/weights.json").read_text())
ck = cfg["utmosv2"]["checkpoint"]
dst = root / "models/eval/utmosv2" / ck["file"]
def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
if not dst.exists() or sha(dst) != ck["sha256"]:
    subprocess.run(["curl", "-fL", "-o", str(dst), ck["url"]], check=True)
got = sha(dst)
if got != ck["sha256"]:
    dst.unlink()
    sys.exit(f"sha256 de {dst.name} no coincide: {got} != {ck['sha256']}")
print("OK pesos UTMOSv2", got)

# Modelos HF a revisión fija; refs/main apunta a esa revisión para que la carga offline resuelva a lo fijado.
from huggingface_hub import snapshot_download
import os
hub = Path(os.environ["HF_HOME"]) / "hub"
for m in cfg["hf_models"]:
    snapshot_download(m["repo"], revision=m["revision"])
    ref = hub / ("models--" + m["repo"].replace("/", "--")) / "refs" / "main"
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_text(m["revision"])
    print("OK", m["repo"], m["revision"])
PY
echo "Entorno de evaluación listo en $VENV (usar scripts/eval/run.sh)"
