#!/usr/bin/env bash
# T-48: parakeet vs moonshine-tiny vs moonshine-small en el pipeline baseline (Marian), en serie y offline.
# Sets: gate (offline-review, 3 ítems), openslr61-test (12 clips es-AR, el de la referencia 409 ms) y short
# (frases de 1-4 palabras de scripts/eval/t48_short_set.py). Correr con la máquina sola (sin :8877 traduciendo).
# Uso: scripts/eval/t48_asr_compare.sh [OUT]   · BACKENDS="parakeet moonshine-tiny" SETS="short" para acotar
#      REPS (gate/openslr, def. 3) · SHORT_REPS (def. 1)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
OUT="${1:-artifacts/t48/runs}"; REPS="${REPS:-3}"; SHORT_REPS="${SHORT_REPS:-1}"
BACKENDS="${BACKENDS:-parakeet moonshine-tiny moonshine-small c2-ref}"; SETS="${SETS:-gate openslr61-test short}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$ROOT"
[ -f artifacts/t48/short/manifest.jsonl ] || .venv/bin/python scripts/eval/t48_short_set.py artifacts/t48/short
for set in $SETS; do
  case "$set" in
    gate) man=artifacts/benchmark/offline-review/manifest.jsonl; root=data/audio; split=development; reps=$REPS ;;
    openslr61-test) man=data/manifests/openslr61_es_ar_sample.jsonl; root=data/audio; split=test; reps=$REPS ;;
    short) man=artifacts/t48/short/manifest.jsonl; root=artifacts/t48/short; split=development; reps=$SHORT_REPS ;;
    *) echo "set desconocido: $set" >&2; exit 2 ;;
  esac
  mkdir -p "$OUT/$set"; echo "$ROOT/$man" > "$OUT/$set/manifest.path"
  for b in $BACKENDS; do
    if [ "$b" = c2-ref ]; then  # config congelada de la referencia 409 ms (artifacts/final-validation/tiny-ep160-v1-config.json)
      cfg='{"asr_backend":"moonshine-tiny","endpoint_silence_ms":160,"tts_trim_lead_margin_ms":30,"asr_skip_silent_tail":true,"tts_first_chunk_max_words":8,"tts_bind_conjunctions":true}'
    else cfg="{\"asr_backend\":\"$b\"}"; fi
    echo "== $set / $b ($(uptime | sed 's/.*load averages*: //'))"
    # RAM pico del proceso entero (ASR + Marian + Kokoro): /usr/bin/time -l, a $OUT/$set/$b.rss.txt
    { /usr/bin/time -l .venv/bin/python -m benchmark run --manifest "$man" --audio-root "$root" --split "$split" \
      --adapter runtime.adapters.baseline:BaselineAdapter --adapter-config "$cfg" \
      --repetitions "$reps" --warmup 1 --out "$OUT/$set/$b" --system-name "t48-$b" ; } 2> >(tee "$OUT/$set/$b.rss.txt" >&2)
  done
done
.venv/bin/python scripts/eval/t48_summary.py "$OUT" | tee "$OUT/../RESULTS.md"
