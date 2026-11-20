#!/usr/bin/env bash
# @bench T-67: A/B de latencia INTERCALADO por repetición, sin esperar reposo (pedido @lead 20/11).
# Ronda r: off, new, fw2, fw3 (1 repetición de los 3 clips dev cada una, warmup 1). Así la carga de la Mac pega parecido a todos.
# Uso: scripts/eval/t67_intercalado.sh OUTDIR
#   COMMIT=da867a6   commit a medir (git archive en /tmp, igual que t69_bench.sh)
#   ROUNDS=6         rondas (n = 3 x ROUNDS por brazo)
#   ARMS="off=ENV;new=;fw2=ENV;fw3=ENV"   brazos en orden (como t69_bench.sh)
# Salida: OUT/<brazo>/r<N>/latency, OUT/loads.csv (carga 1 min antes/después de cada corrida), OUT/DONE al terminar.
# Después: python3 scripts/eval/t67_intercalado.py OUT
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
OUT="${1:?uso: t67_intercalado.sh OUTDIR}"; case "$OUT" in /*) ;; *) OUT="$ROOT/$OUT";; esac
ROUNDS="${ROUNDS:-6}"; PYBIN="${PYBIN:-.venv-inc-llm/bin/python}"; COMMIT="${COMMIT:-$(git rev-parse --short HEAD)}"
OFF="OPENTRADUCTOR_ENDPOINT_ADAPTIVE=0 OPENTRADUCTOR_MT_STREAM_TTS=0 OPENTRADUCTOR_TTS_FIRST_WORDS=8"
ARMS="${ARMS:-off=$OFF;new=;fw2=OPENTRADUCTOR_TTS_FIRST_WORDS=2;fw3=OPENTRADUCTOR_TTS_FIRST_WORDS=3}"
QWEN_DIR="${QWEN_DIR:-$ROOT/artifacts/analysis/llm-mp4-a/models/qwen3-4b-instruct-2507-4bit}"
CFG="${CFG:-{\"asr_backend\":\"moonshine-tiny\",\"mt_engine\":\"qwen\",\"qwen_model_dir\":\"$QWEN_DIR\"}}"
LAT_MANIFEST="${LAT_MANIFEST:-$ROOT/artifacts/benchmark/offline-review/manifest.jsonl}"
CODE="/tmp/ot-t69-$COMMIT"
if [ ! -d "$CODE" ]; then mkdir -p "$CODE" && git archive "$COMMIT" | tar -x -C "$CODE" || exit 2
  for p in .venv .venv-inc-llm artifacts models data; do rm -rf "$CODE/$p"; ln -s "$ROOT/$p" "$CODE/$p"; done; fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$CODE" T69_CODE_ROOT="$CODE"
mkdir -p "$OUT"; echo "$COMMIT" > "$OUT/sha.txt"; echo "ARMS=$ARMS" > "$OUT/arms.txt"; echo "ROUNDS=$ROUNDS" >> "$OUT/arms.txt"
[ -e "$OUT/loads.csv" ] || echo "ronda,brazo,inicio,load1_antes,load1_despues,rc" > "$OUT/loads.csv"
lsof -iTCP:8877 -sTCP:LISTEN >/dev/null 2>&1 && echo ":8877 ABIERTA (app de la persona)" >> "$OUT/arms.txt" || echo ":8877 libre" >> "$OUT/arms.txt"
IFS=';' read -ra ITEMS <<< "$ARMS"
ld(){ sysctl -n vm.loadavg | awk '{print $2}'; }
for r in $(seq 1 "$ROUNDS"); do
  for it in "${ITEMS[@]}"; do
    name="${it%%=*}"; envs="${it#*=}"; D="$OUT/$name/r$r"
    [ -s "$D/latency/traces.jsonl" ] && continue   # reanudable
    mkdir -p "$D"; a=$(ld); t=$(date +%T)
    (cd "$CODE" && env $envs "$PYBIN" -m benchmark run --manifest "$LAT_MANIFEST" --audio-root data/audio \
      --split development --adapter runtime.adapters.baseline:BaselineAdapter --adapter-config "$CFG" \
      --repetitions 1 --warmup 1 --out "$D/latency" --system-name "$name") > "$D/latency.log" 2>&1; rc=$?
    echo "$r,$name,$t,$a,$(ld),$rc" >> "$OUT/loads.csv"
  done
done
date +%T > "$OUT/DONE"
