#!/usr/bin/env bash
# T-69 (@bench): script único para A/B de latencia por etapa y de aciertos con contexto.
#
# Uso:  scripts/eval/t69_bench.sh OUTDIR
#   ARMS="nombre=VAR=1 VAR2=0;nombre2=VAR=0"   brazos en orden; cada uno es nombre=ENV (ENV puede ser vacío: "base=").
#                                              Repetir nombres con sufijo (off;on;on2;off2) reparte la deriva. Por defecto: "base=".
#   REPS=5            repeticiones del set de latencia (3 clips dev del gate, n=3×REPS por brazo)
#   CTX=0,6           turnos de contexto de las charlas (0 = sin contexto); el pipeline usa mt_context_turns
#   CFG='{...}'       config JSON de PipelineConfig para el adaptador (por defecto: tienda = Moonshine tiny + Qwen 4B)
#   COMMIT=<sha>      mide ese commit (git archive en /tmp, con venvs/modelos/datos enlazados) y no el árbol de trabajo, que
#                     puede traer cambios sin commitear de otros. Sin COMMIT mide el árbol de trabajo y lo anota.
#   SKIP_LAT=1 / SKIP_CHARLAS=1   saltar una de las dos mediciones
#   QUIET_MAX=1.5 QUIET_TMO=600   reposo (loadavg 1 min) y espera máxima; si vence se mide igual y queda marcado NO REPOSO
# Ejemplos:
#   ARMS="off=OPENTRADUCTOR_MT_GLOSSARY=0;on=OPENTRADUCTOR_MT_GLOSSARY=1" scripts/eval/t69_bench.sh artifacts/t69-glosario
#   scripts/eval/t69_bench.sh artifacts/t69-base                                   # línea base del HEAD actual
# Salida: OUTDIR/<brazo>/{latency,charlas}, OUTDIR/<brazo>/stages.md y OUTDIR/RESUMEN.md (tabla A/B).
# Latencia por etapa = fin de habla→endpoint (espera) · ASR final · MT · TTS 1er audio (end_to_first = suma).
# Sin cobertura aquí: salida al dispositivo (BlackHole, scripts/eval/device_loopback.py, necesita lease) y MT 1er token
# en el pipeline (se mide aparte con translate_stream en charlas, ver mt_1er_token_*).
# Debe correr SOLO, sin otros benchmarks. No mata la app de la persona en :8877: la anota.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
OUT="${1:?uso: t69_bench.sh OUTDIR}"; case "$OUT" in /*) ;; *) OUT="$ROOT/$OUT";; esac; REPS="${REPS:-5}"; CTX="${CTX:-0,6}"; PYBIN="${PYBIN:-.venv-inc-llm/bin/python}"
QWEN_DIR="${QWEN_DIR:-$ROOT/artifacts/analysis/llm-mp4-a/models/qwen3-4b-instruct-2507-4bit}"
CFG="${CFG:-{\"asr_backend\":\"moonshine-tiny\",\"mt_engine\":\"qwen\",\"qwen_model_dir\":\"$QWEN_DIR\"}}"
ARMS="${ARMS:-base=}"
mkdir -p "$OUT"; CODE="$ROOT"
if [ -n "${COMMIT:-}" ]; then
  CODE="/tmp/ot-t69-$COMMIT"
  if [ ! -d "$CODE" ]; then mkdir -p "$CODE" && git archive "$COMMIT" | tar -x -C "$CODE" || exit 2
    for p in .venv .venv-inc-llm artifacts models data; do rm -rf "$CODE/$p"; ln -s "$ROOT/$p" "$CODE/$p"; done; fi
  CINFO="commit: $COMMIT (git archive, árbol limpio)"
else
  CINFO="commit: $(git rev-parse --short HEAD)$(git diff --quiet -- runtime || echo ' (+ cambios sin commitear en runtime/)')"
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$CODE" T69_CODE_ROOT="$CODE"
{ echo "$CINFO"; date; uptime
  lsof -iTCP:8877 -sTCP:LISTEN >/dev/null 2>&1 && echo ":8877 ABIERTA (app de la persona)" || echo ":8877 libre"
  echo "ARMS=$ARMS REPS=$REPS CTX=$CTX"; echo "CFG=$CFG"; } | tee "$OUT/entorno.txt"
IFS=';' read -ra ITEMS <<< "$ARMS"
for it in "${ITEMS[@]}"; do
  name="${it%%=*}"; envs="${it#*=}"; D="$OUT/$name"; mkdir -p "$D"
  echo "== $name  ENV=[$envs]  $(date +%T)" | tee -a "$OUT/entorno.txt"
  if [ -z "${SKIP_LAT:-}" ]; then
    if "$ROOT/scripts/eval/wait_quiet.sh" "${QUIET_MAX:-1.5}" 4 "${QUIET_TMO:-600}" | tee "$D/reposo.txt"; then :; else echo "NO REPOSO" >> "$D/reposo.txt"; fi
    uptime > "$D/load-start.txt"
    (cd "$CODE" && env $envs "$PYBIN" -m benchmark run --manifest "$ROOT/artifacts/benchmark/offline-review/manifest.jsonl" --audio-root data/audio \
      --split development --adapter runtime.adapters.baseline:BaselineAdapter --adapter-config "$CFG" \
      --repetitions "$REPS" --warmup 1 --out "$D/latency" --system-name "$name") > "$D/latency.log" 2>&1 || echo "latency rc!=0 ($name)" | tee -a "$OUT/entorno.txt"
    uptime > "$D/load-end.txt"
    "$PYBIN" "$ROOT/scripts/eval/t58_stage_breakdown.py" "$D/latency" > "$D/stages.md" 2>&1 || true
  fi
  if [ -z "${SKIP_CHARLAS:-}" ]; then
    env $envs "$PYBIN" "$ROOT/scripts/eval/t69_charlas_eval.py" --out "$D/charlas" --ctx "$CTX" --first-token > "$D/charlas.log" 2>&1 || echo "charlas rc!=0 ($name)" | tee -a "$OUT/entorno.txt"
  fi
done
"$PYBIN" "$ROOT/scripts/eval/t69_summary.py" "$OUT" | tee "$OUT/RESUMEN.md"
