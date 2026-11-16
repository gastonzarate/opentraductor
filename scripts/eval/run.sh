#!/usr/bin/env bash
# Ejecuta el CLI del benchmark dentro del entorno de evaluación, sin red (modelos fijados).
# Ej.: scripts/eval/run.sh proxy-score --index X/export_index.jsonl --checkpoint models/eval/utmosv2/fold0_s42_best_model.pth --out out.json
#      scripts/eval/run.sh run ... --audible-asr faster-whisper:small.en
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export HF_HOME="$ROOT/models/eval/hf" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
exec "$ROOT/models/eval/venv/bin/python" -m benchmark "$@"
