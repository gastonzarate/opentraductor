#!/usr/bin/env bash
# @audio T-81: costo en calidad de cada palanca hacia 0,3 s. Intercalado (la Mac nunca está libre).
# Brazo: nombre|ENV1=a ENV2=b|{"campo_extra_de_PipelineConfig": ...}   (separados por ;)
# Fases (reanudables, OUT/<fase>.DONE): latencia (t67-clips, ROUNDS rondas) -> terminos (ctx200) -> charlas (cortes).
# Uso: COMMIT=45eb5b7 ROUNDS=5 bash scripts/eval/t81_bench.sh artifacts/t81
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
OUT="${1:?uso: t81_bench.sh OUTDIR}"; case "$OUT" in /*) ;; *) OUT="$ROOT/$OUT";; esac
COMMIT="${COMMIT:-$(git rev-parse --short HEAD)}"; ROUNDS="${ROUNDS:-5}"
PYBIN="${PYBIN:-.venv-inc-llm/bin/python}"
M="$ROOT/artifacts/analysis/llm-mp4-a/models"
Q4="$M/qwen3-4b-instruct-2507-4bit"; Q06="$M/qwen3-0.6b-4bit"
DEF_ARMS='ref||{};spec|OPENTRADUCTOR_ASR_SPEC_FINISH=1|{};short32|OPENTRADUCTOR_ENDPOINT_SHORT_MS=32|{};sil96||{"endpoint_silence_ms":96};ctx500|OPENTRADUCTOR_MT_CONTEXT_TOKENS=500|{};q06||{"qwen_model_dir":"Q06"};marian||{"mt_engine":"marian"}'
DEF_ARMS="${DEF_ARMS//Q06/$Q06}"
ARMS="${ARMS:-$DEF_ARMS}"
CUT_ARMS="${CUT_ARMS:-ref spec short32 sil96}"
BASE="{\"asr_backend\":\"moonshine-tiny\",\"mt_engine\":\"qwen\",\"qwen_model_dir\":\"$Q4\"}"
CODE="/tmp/ot-t69-$COMMIT"
if [ ! -d "$CODE" ]; then mkdir -p "$CODE" && git archive "$COMMIT" | tar -x -C "$CODE" || exit 2
  for p in .venv .venv-inc-llm artifacts models data; do rm -rf "$CODE/$p"; ln -s "$ROOT/$p" "$CODE/$p"; done; fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$CODE" T69_CODE_ROOT="$CODE"
mkdir -p "$OUT"; echo "$COMMIT" > "$OUT/sha.txt"; printf 'ARMS=%s\nROUNDS=%s\n' "$ARMS" "$ROUNDS" > "$OUT/arms.txt"
ld(){ sysctl -n vm.loadavg | awk '{print $2}'; }
log(){ echo "$(date +%T) $* | load $(ld)" >> "$OUT/run.log"; }
cfg(){ python3 -c 'import json,sys; b=json.loads(sys.argv[1]); b.update(json.loads(sys.argv[2] or "{}")); print(json.dumps(b))' "$BASE" "$1"; }
IFS=';' read -ra ITEMS <<< "$ARMS"
run_bench(){ # $1 manifest $2 outdir $3 name $4 envs $5 cfgjson
  mkdir -p "$2"
  (cd "$CODE" && env $4 "$PYBIN" -m benchmark run --manifest "$1" --audio-root data/audio \
    --split development --adapter runtime.adapters.baseline:BaselineAdapter --adapter-config "$5" \
    --repetitions 1 --warmup 1 --out "$2/latency" --system-name "$3") > "$2/latency.log" 2>&1
}
# ---- latencia intercalada (orden rotado por ronda)
if [ ! -e "$OUT/latencia.DONE" ]; then
  L="$OUT/latencia"; mkdir -p "$L"; [ -e "$L/loads.csv" ] || echo "ronda,brazo,inicio,load1_antes,load1_despues,rc" > "$L/loads.csv"
  N=${#ITEMS[@]}
  for r in $(seq 1 "$ROUNDS"); do for j in $(seq 0 $((N-1))); do
    it="${ITEMS[$(( (j + r - 1) % N ))]}"; IFS='|' read -r name envs extra <<< "$it"; D="$L/$name/r$r"
    [ -s "$D/latency/traces.jsonl" ] && continue
    a=$(ld); t=$(date +%T); run_bench "$ROOT/artifacts/t67-clips/manifest.jsonl" "$D" "$name" "$envs" "$(cfg "$extra")"; rc=$?
    echo "$r,$name,$t,$a,$(ld),$rc" >> "$L/loads.csv"; log "lat r$r $name rc=$rc"
  done; done
  python3 scripts/eval/t67_intercalado.py "$L" --ref ref > "$L/resumen.log" 2>&1
  date +%T > "$OUT/latencia.DONE"
fi
# ---- terminos ctx200 (solo cambian el texto de la MT: contexto y modelo)
if [ ! -e "$OUT/terminos.DONE" ]; then
  T="$OUT/terminos"; mkdir -p "$T"; S="$ROOT/artifacts/eval/contexto-largo/charlas.jsonl"
  ev(){ n=$1; shift; [ -s "$T/$n/summary.json" ] && return; env "$@" "$PYBIN" scripts/eval/t69_charlas_eval.py --out "$T/$n" --ctx 200 --prefill --simul --set "$S" > "$T/$n.log" 2>&1; log "terminos $n rc=$?"; }
  ev ref OPENTRADUCTOR_QWEN_DIR="$Q4"
  ev ctx500 OPENTRADUCTOR_QWEN_DIR="$Q4" OPENTRADUCTOR_MT_CONTEXT_TOKENS=500
  ev ctx200tok OPENTRADUCTOR_QWEN_DIR="$Q4" OPENTRADUCTOR_MT_CONTEXT_TOKENS=200
  ev q06 OPENTRADUCTOR_QWEN_DIR="$Q06"
  date +%T > "$OUT/terminos.DONE"
fi
# ---- cortes falsos en charlas (114 turnos) para las palancas de fin de frase
if [ ! -e "$OUT/charlas.DONE" ]; then
  MAN="$ROOT/data/audio/synth-charlas/manifest.jsonl"
  for name in $CUT_ARMS; do for it in "${ITEMS[@]}"; do IFS='|' read -r n envs extra <<< "$it"; [ "$n" = "$name" ] || continue
    D="$OUT/charlas/$name/c1"; [ -s "$D/latency/traces.jsonl" ] && continue
    run_bench "$MAN" "$D" "$name" "$envs" "$(cfg "$extra")"; log "charlas $name rc=$?"
  done; done
  date +%T > "$OUT/charlas.DONE"
fi
date +%T > "$OUT/DONE"; log fin
