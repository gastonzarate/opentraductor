#!/usr/bin/env bash
# Receta de cola: CONCATENACIÓN DE GRABACIONES REALES CON PAUSAS SINTÉTICAS DE 1.5 s en un único stream
# (8 ítems DEVELOPMENT; no es conversación natural ni pausas naturales; no usa split test).
# NO ejecutar sin ventana de cómputo asignada.
#
# Uso: CONFIG_JSON='{"asr_backend":"moonshine-tiny",...}' scripts/eval/continuous_recipe.sh NOMBRE [OUTDIR]
#   REPS=1 (cada repetición es un stream completo; orden de ítems por semilla del arnés)
# Mide: PCM y audibleSim por segmento (source_span_s), lag semántico y backlog (run.json), y ASR crudo/entrada MT
#       (run.json -> meta.adapter_metadata.asr_text_log).
# Cobertura por ítem (recuentos, texto concatenado, WER, palabras de referencia borradas, duplicados exactos):
#       .venv/bin/python scripts/eval/continuous_coverage.py OUTDIR/NOMBRE
#   Más de 8 asr_final NO prueba omisión, duplicación ni sobre-segmentación incorrecta (puede ser fragmentación
#   legítima dentro de un ítem): el script separa duplicados exactos de segmentos múltiples.
# Texto AUDIBLE en continuo: PENDIENTE. El arnés no exporta WAV por ítem en streams continuos; esta receta no lo mide.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
NAME="${1:?nombre de la corrida}"; OUT="${2:-artifacts/runtime-continuous}/$NAME"
: "${CONFIG_JSON:?define CONFIG_JSON}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$ROOT"
.venv/bin/python scripts/eval/make_continuous_manifest.py "$OUT/manifest.jsonl"
.venv/bin/python -m benchmark run --manifest "$OUT/manifest.jsonl" --audio-root data/audio --split development \
  --adapter runtime.adapters.baseline:BaselineAdapter --adapter-config "$CONFIG_JSON" --continuous \
  --repetitions "${REPS:-1}" --warmup 1 --deadline-s 30 --out "$OUT" --export-audio-dir "$OUT/audio" --system-name "$NAME"
