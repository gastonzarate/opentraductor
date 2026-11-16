#!/usr/bin/env bash
# T-50 fila de modo (pedido de @lead, #t-48): baseline + Qwen 4B contra incremental (app diaria), mismo Qwen 4B y Moonshine tiny.
# Mide sobre openslr61-test (12 clips, con referencias): latencia a primer audio, WER de ASR y BLEU/chrF de MT por el harness.
# Brazos intercalados B, I, B2, I2 para repartir la carga. Las 54 frases de texto usan el mismo prompt en los dos modos
# (SYSTEM idéntico a MP4-A); la diferencia de sentido sale del prefijo comprometido del incremental, que solo se ve con audio.
# Uso: scripts/eval/t50_mode_row.sh artifacts/t50/mode   (tomar la máquina en #t-45; ~10-15 min; sin red)
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${1:-artifacts/t50/mode}"
REPS="${REPS:-3}"
PY=.venv-inc-llm/bin/python
QWEN4B=artifacts/analysis/llm-mp4-a/models/qwen3-4b-instruct-2507-4bit
COMMON=(--manifest data/manifests/openslr61_es_ar_sample.jsonl --audio-root data/audio --split test
        --repetitions "$REPS" --warmup 1)
B_CFG="{\"asr_backend\":\"moonshine-tiny\",\"mt_engine\":\"qwen\",\"qwen_model_dir\":\"$QWEN4B\"}"
mkdir -p "$OUT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
B() {
  echo "== $1 baseline+qwen4b load=$(sysctl -n vm.loadavg)" | tee -a "$OUT/load.txt"
  PYTHONPATH=. /usr/bin/time -l "$PY" -m benchmark run "${COMMON[@]}" \
    --adapter runtime.adapters.baseline:BaselineAdapter --adapter-config "$B_CFG" \
    --system-name "t50-$1" --out "$OUT/$1" 2> "$OUT/$1.time.txt"
}
I() {
  echo "== $1 incremental load=$(sysctl -n vm.loadavg)" | tee -a "$OUT/load.txt"
  PYTHONPATH=.:artifacts/analysis/incremental-adapter-v1:artifacts/analysis/incremental-llm-v1 /usr/bin/time -l "$PY" \
    -m benchmark run "${COMMON[@]}" --adapter inc_llm.adapter:IncrementalLLMAdapter --adapter-config '{}' \
    --system-name "t50-$1" --out "$OUT/$1" 2> "$OUT/$1.time.txt"
}
B base1; I inc1; B base2; I inc2
grep -H -E "peak memory footprint" "$OUT"/*.time.txt || true
echo "listo $(date +%H:%M)" > "$OUT/DONE"
