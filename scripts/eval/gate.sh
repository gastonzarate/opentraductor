#!/usr/bin/env bash
# Gate de calidad (T-45): corre las métricas del estándar y sale con error si un umbral se rompe.
# Uso: scripts/eval/gate.sh [--out DIR] [--only torch,tests,vad,latency,mt,tts] [--skip ...] [--reps 3]
# Tarda varios minutos (carga de modelos, pytest, 3 repeticiones). Cerrá la app de la persona (:8877) o anotalo.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$ROOT"
exec "$ROOT/.venv/bin/python" "$ROOT/scripts/eval/gate.py" "$@"
