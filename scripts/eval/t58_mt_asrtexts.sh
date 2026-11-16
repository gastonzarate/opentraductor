#!/usr/bin/env bash
# T-58: MT sola (Qwen 4B y 1.7B) sobre los 3 textos asr_final de los traces del gate, en caliente (REPS=7, warmup 2).
# Sirve para separar costo del pipeline (hilo GPU compartido con Kokoro) de largo de frase.
# Uso: scripts/eval/t58_mt_asrtexts.sh artifacts/t58-mt-asrtexts   (medir solo en reposo/tras el "listo" del build)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
OUT="${1:?uso: t58_mt_asrtexts.sh OUTDIR}"; REPS="${REPS:-7}"; PYBIN=".venv-inc-llm/bin/python"
Q4="${Q4:-artifacts/analysis/llm-mp4-a/models/qwen3-4b-instruct-2507-4bit}"
Q17="${Q17:-artifacts/analysis/t50-qwen-sizes/models/qwen3-1.7b-4bit}"
mkdir -p "$OUT"
{ echo "commit: $(git rev-parse --short HEAD) · REPS=$REPS"; date; uptime
  lsof -iTCP:8877 -sTCP:LISTEN >/dev/null 2>&1 && echo ":8877 ABIERTA" || echo ":8877 libre"; } | tee "$OUT/entorno.txt"
export OT_MLX_CACHE_LIMIT_MB="${CACHE_MB:-256}" EXTRA_PYTHONPATH="$ROOT/scripts/eval/mlxcap"
for pair in "qwen4b:$Q4" "qwen1.7b:$Q17"; do
  name="${pair%%:*}"; dir="${pair#*:}"
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$ROOT:$EXTRA_PYTHONPATH" OPENTRADUCTOR_QWEN_DIR="$dir" "$PYBIN" -m benchmark run \
    --text-manifest data/manifests/t58_asr_final_texts.jsonl --adapter benchmark.adapters.text_mt_tts:TextMtTtsAdapter \
    --adapter-config "{\"stages\":\"mt\",\"mt_engine\":\"qwen\",\"qwen_model_dir\":\"$dir\"}" \
    --split development --repetitions "$REPS" --warmup 2 --out "$OUT/$name"
done
uptime | tee -a "$OUT/entorno.txt"
