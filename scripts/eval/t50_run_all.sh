#!/usr/bin/env bash
# T-50: RAM pico (phys_footprint), latencia de MT/TTS y traducciones de las 54 frases para Qwen3 4B / 1.7B / 0.6B,
# en serie, una sesión de SESSION_MIN minutos por tamaño (criterio 8 GB: 10 min, pico <= 3,5 GB).
# Uso: ASR=moonshine-tiny SESSION_MIN=10 scripts/eval/t50_run_all.sh artifacts/t50/ram
# Tomar la máquina en #t-45 antes de correrlo. Sin red (HF_HUB_OFFLINE=1). ~35-40 min con SESSION_MIN=10.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${1:-artifacts/t50/ram}"
ASR="${ASR:-moonshine-tiny}"
SESSION_MIN="${SESSION_MIN:-10}"
PY=.venv-inc-llm/bin/python
mkdir -p "$OUT"
declare -a SIZES=(
  "qwen3-4b=artifacts/analysis/llm-mp4-a/models/qwen3-4b-instruct-2507-4bit"
  "qwen3-1.7b=artifacts/analysis/t50-qwen-sizes/models/qwen3-1.7b-4bit"
  "qwen3-0.6b=artifacts/analysis/t50-qwen-sizes/models/qwen3-0.6b-4bit"
)
for s in "${SIZES[@]}"; do
  name="${s%%=*}"; dir="${s#*=}"
  echo "== $name ($dir) asr=$ASR session=${SESSION_MIN}min load=$(sysctl -n vm.loadavg)"
  HF_HUB_OFFLINE=1 PYTHONPATH=. /usr/bin/time -l "$PY" scripts/eval/t50_pipeline_ram.py \
    --qwen-dir "$dir" --asr "$ASR" --session-min "$SESSION_MIN" --out "$OUT/$name.json" 2> "$OUT/$name.time.txt"
  grep -E "maximum resident|peak memory footprint" "$OUT/$name.time.txt" || true
done
