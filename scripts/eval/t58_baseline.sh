#!/usr/bin/env bash
# T-58: base de la config de la tienda (Moonshine + Qwen). EN SERIE y con la máquina en reposo (espera load<3).
# Uso:  ASR=moonshine-tiny QWEN_DIR=<carpeta Qwen MLX> scripts/eval/t58_baseline.sh artifacts/t58 [REPS=7]
#   Latencia/ASR WER: set del gate (3 clips dev, offline-review) con REPS=7 → n=21 por métrica (≥20, mínimo del benchmark para p95).
#   MT p95: 54 frases de text_meetings_ar_dev con Qwen, REPS=3 (n=162).
# MODE=incremental mide inc_llm.adapter:IncrementalLLMAdapter (Moonshine tiny + Qwen fijados por el adaptador; ASR/QWEN_DIR no aplican a la latencia).
#   Por defecto MODE=baseline (BaselineAdapter + Qwen). El modo lo confirma @audio en #t-58 tras la fila de T-50.
# Corre en .venv-inc-llm (tiene mlx_lm); .venv no.  Anotar en el informe: ASR de T-48, Qwen de T-50, :8877 y loadavg.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
OUT="${1:?uso: t58_baseline.sh OUTDIR}"; LAT_REPS="${LAT_REPS:-7}"; MT_REPS="${MT_REPS:-3}"
ASR="${ASR:-moonshine-tiny}"; QWEN_DIR="${QWEN_DIR:?QWEN_DIR=carpeta del Qwen elegido en T-50}"
PYBIN=".venv-inc-llm/bin/python"; MODE="${MODE:-baseline}"
mkdir -p "$OUT"
{ echo "commit: $(git rev-parse --short HEAD) · MODE=$MODE · ASR=$ASR · QWEN_DIR=$QWEN_DIR"; date; uptime
  lsof -iTCP:8877 -sTCP:LISTEN >/dev/null 2>&1 && echo ":8877 ABIERTA (app de la persona)" || echo ":8877 libre"; } | tee "$OUT/entorno.txt"
scripts/eval/wait_quiet.sh 3 4 "${QUIET_TIMEOUT_S:-600}" | tee -a "$OUT/entorno.txt"
export OPENTRADUCTOR_QWEN_DIR="$QWEN_DIR"
if [ "$MODE" = "incremental" ]; then
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=".:artifacts/analysis/incremental-adapter-v1:artifacts/analysis/incremental-llm-v1" "$PYBIN" -m benchmark run \
    --manifest artifacts/benchmark/offline-review/manifest.jsonl --audio-root data/audio --split development \
    --adapter inc_llm.adapter:IncrementalLLMAdapter --adapter-config '{}' --repetitions "$LAT_REPS" --warmup 1 \
    --out "$OUT/latency/store" --system-name store
else
  CONFIGS="store={\"asr_backend\":\"$ASR\",\"mt_engine\":\"qwen\",\"qwen_model_dir\":\"$QWEN_DIR\"}" REPS="$LAT_REPS" PYBIN="$PYBIN" \
    scripts/eval/compare_configs.sh "$OUT/latency"
fi
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$ROOT" "$PYBIN" -m benchmark run --text-manifest data/manifests/text_meetings_ar_dev.jsonl \
  --adapter benchmark.adapters.text_mt_tts:TextMtTtsAdapter --adapter-config "{\"stages\":\"mt\",\"mt_engine\":\"qwen\",\"qwen_model_dir\":\"$QWEN_DIR\"}" \
  --split development --repetitions "$MT_REPS" --warmup 2 --out "$OUT/mt"
uptime | tee -a "$OUT/entorno.txt"
echo "listo: $OUT/latency/store/run.json y $OUT/mt/run.json"
