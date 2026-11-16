#!/usr/bin/env bash
# Comparación acotada de configuraciones del runtime sobre el manifest DEVELOPMENT (no test). Offline forzado.
# Uso: scripts/eval/compare_configs.sh OUTDIR   (secuencial; una carga de modelos a la vez)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
OUT="${1:-artifacts/runtime-opt}"; REPS="${REPS:-3}"
PYBIN="${PYBIN:-.venv/bin/python}"  # T-58: PYBIN=.venv-inc-llm/bin/python para la config de la tienda (Qwen/MLX)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$ROOT"
run() { # nombre json
  "$PYBIN" -m benchmark run --manifest artifacts/benchmark/offline-review/manifest.jsonl \
    --audio-root data/audio --split development --adapter runtime.adapters.baseline:BaselineAdapter \
    --adapter-config "$2" --repetitions "$REPS" --warmup 1 --out "$OUT/$1" \
    --export-audio-dir "$OUT/$1/audio" --system-name "$1"
}
# CONFIGS="nombre=json;nombre2=json2" para elegir; por defecto el conjunto acotado.
DEFAULT='A-baseline={};B-ep160-chunk96={"endpoint_silence_ms":160,"asr_chunk_ms":96};D-ep160-chunk160={"endpoint_silence_ms":160};E-ep160-trim30={"endpoint_silence_ms":160,"tts_trim_lead_margin_ms":30};F-ep160-trim30-skiptail={"endpoint_silence_ms":160,"tts_trim_lead_margin_ms":30,"asr_skip_silent_tail":true}'
IFS=';' read -ra ITEMS <<< "${CONFIGS:-$DEFAULT}"
for it in "${ITEMS[@]}"; do run "${it%%=*}" "${it#*=}"; done
