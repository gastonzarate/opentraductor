#!/usr/bin/env bash
# T-83: «policial»/«nazismo» cortados al cerrar frase. Reproducción offline con la config de la app
# (parakeet + marian, defaults de PipelineConfig) sobre 1584 y 1516, intercalando brazos por ronda.
# Uso: scripts/eval/t83_tail_ab.sh OUTDIR   (ROUNDS=5, ARMS="skip=;noskip=..." como t67_intercalado.sh)
# Brazos por defecto: skip (asr_skip_silent_tail true, default) vs noskip (false). Escribe OUT/DONE al terminar.
# Después: python3 scripts/eval/t83_tail_count.py OUT
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
OUT="${1:?uso: t83_tail_ab.sh OUTDIR}"; case "$OUT" in /*) ;; *) OUT="$ROOT/$OUT";; esac
ROUNDS="${ROUNDS:-5}"; PYBIN="${PYBIN:-.venv/bin/python}"
MAN="${MAN:-$ROOT/artifacts/t83/manifest.jsonl}"
[ -n "${ARMS:-}" ] || ARMS='skip={};noskip={"asr_skip_silent_tail":false}'
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p "$OUT"; echo "ARMS=$ARMS ROUNDS=$ROUNDS $(git rev-parse --short HEAD)" > "$OUT/arms.txt"
IFS=';' read -ra ITEMS <<< "$ARMS"
for r in $(seq 1 "$ROUNDS"); do
  for it in "${ITEMS[@]}"; do
    name="${it%%=*}"; cfg="${it#*=}"; D="$OUT/$name/r$r"
    [ -s "$D/traces.jsonl" ] && continue
    mkdir -p "$D"
    "$PYBIN" -m benchmark run --manifest "$MAN" --audio-root data/audio --split development \
      --adapter runtime.adapters.baseline:BaselineAdapter --adapter-config "$cfg" \
      --repetitions 1 --warmup 1 --out "$D" --system-name "$name" > "$D/run.log" 2>&1
    echo "$r,$name,$?,$(sysctl -n vm.loadavg)" >> "$OUT/rc.csv"
  done
done
date +%T > "$OUT/DONE"
