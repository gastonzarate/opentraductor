#!/usr/bin/env bash
# T-80: A/B intercalado de ENDPOINT_COMPLETE_TURN (ct) y MT_SIMUL (simul) sobre un commit.
# Brazos por defecto: ref= ; ct=OPENTRADUCTOR_ENDPOINT_COMPLETE_TURN=1 ; simul=OPENTRADUCTOR_MT_SIMUL=1 ; ctsimul=ambos.
# Fases (cada una reanudable; escribe OUT/<fase>.DONE):
#   1 charlas : cortes falsos. 114 turnos de data/audio/synth-charlas en CHUNKS tandas; en cada tanda corren los 4 brazos
#               (orden rotado) para que la carga de la Mac pegue parecido. Salida OUT/charlas/<brazo>/c<k>/latency + loads.csv
#   2 latencia: t67_intercalado.sh sobre artifacts/t67-clips (8 clips), ROUNDS rondas. Salida OUT/latencia/
#   3 terminos: t69_charlas_eval.py ctx 0,200 (ref y simul; ct no cambia el texto de la MT, ctsimul = simul)
#   4 analisis: scripts/eval/t80_analisis.py OUT
# Uso: COMMIT=4806494 ROUNDS=6 CHUNKS=3 bash scripts/eval/t80_intercalado.sh artifacts/t80
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
OUT="${1:?uso: t80_intercalado.sh OUTDIR}"; case "$OUT" in /*) ;; *) OUT="$ROOT/$OUT";; esac
COMMIT="${COMMIT:-$(git rev-parse --short HEAD)}"; ROUNDS="${ROUNDS:-6}"; CHUNKS="${CHUNKS:-3}"
PYBIN="${PYBIN:-.venv-inc-llm/bin/python}"
ARMS="${ARMS:-ref=;ct=OPENTRADUCTOR_ENDPOINT_COMPLETE_TURN=1;simul=OPENTRADUCTOR_MT_SIMUL=1;ctsimul=OPENTRADUCTOR_ENDPOINT_COMPLETE_TURN=1 OPENTRADUCTOR_MT_SIMUL=1}"
QWEN_DIR="${QWEN_DIR:-$ROOT/artifacts/analysis/llm-mp4-a/models/qwen3-4b-instruct-2507-4bit}"
CFG="${CFG:-{\"asr_backend\":\"moonshine-tiny\",\"mt_engine\":\"qwen\",\"qwen_model_dir\":\"$QWEN_DIR\"}}"
CODE="/tmp/ot-t69-$COMMIT"
if [ ! -d "$CODE" ]; then mkdir -p "$CODE" && git archive "$COMMIT" | tar -x -C "$CODE" || exit 2
  for p in .venv .venv-inc-llm artifacts models data; do rm -rf "$CODE/$p"; ln -s "$ROOT/$p" "$CODE/$p"; done; fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$CODE" T69_CODE_ROOT="$CODE"
mkdir -p "$OUT/charlas"; echo "$COMMIT" > "$OUT/sha.txt"; echo "ARMS=$ARMS" > "$OUT/arms.txt"
echo "ROUNDS=$ROUNDS CHUNKS=$CHUNKS" >> "$OUT/arms.txt"
lsof -iTCP:8877 -sTCP:LISTEN >/dev/null 2>&1 && echo ":8877 ABIERTA (app de la persona)" >> "$OUT/arms.txt" || echo ":8877 libre" >> "$OUT/arms.txt"
ld(){ sysctl -n vm.loadavg | awk '{print $2}'; }
log(){ echo "$(date +%T) $* | load $(ld)" >> "$OUT/run.log"; }

# ---- fase 1: cortes en charlas
if [ ! -e "$OUT/charlas.DONE" ]; then
  MAN="$ROOT/data/audio/synth-charlas/manifest.jsonl"
  python3 - "$MAN" "$OUT/charlas" "$CHUNKS" <<'PY'
import sys, pathlib
man, out, n = sys.argv[1], pathlib.Path(sys.argv[2]), int(sys.argv[3])
rows = [l for l in open(man) if l.strip()]
k = -(-len(rows) // n)
for i in range(n):
    (out / f"c{i+1}.jsonl").write_text("".join(rows[i*k:(i+1)*k]))
PY
  [ -e "$OUT/charlas/loads.csv" ] || echo "tanda,brazo,inicio,load1_antes,load1_despues,rc" > "$OUT/charlas/loads.csv"
  IFS=';' read -ra ITEMS <<< "$ARMS"; N=${#ITEMS[@]}
  for c in $(seq 1 "$CHUNKS"); do
    for j in $(seq 0 $((N-1))); do
      it="${ITEMS[$(( (j + c - 1) % N ))]}"; name="${it%%=*}"; envs="${it#*=}"; D="$OUT/charlas/$name/c$c"
      [ -s "$D/latency/traces.jsonl" ] && continue
      mkdir -p "$D"; a=$(ld); t=$(date +%T)
      (cd "$CODE" && env $envs "$PYBIN" -m benchmark run --manifest "$OUT/charlas/c$c.jsonl" --audio-root data/audio \
        --split development --adapter runtime.adapters.baseline:BaselineAdapter --adapter-config "$CFG" \
        --repetitions 1 --warmup 1 --out "$D/latency" --system-name "$name") > "$D/latency.log" 2>&1; rc=$?
      echo "$c,$name,$t,$a,$(ld),$rc" >> "$OUT/charlas/loads.csv"; log "charlas c$c $name rc=$rc"
    done
  done
  date +%T > "$OUT/charlas.DONE"
fi

# ---- fase 2: latencia intercalada
if [ ! -e "$OUT/latencia.DONE" ]; then
  log "latencia inicio"
  COMMIT="$COMMIT" ROUNDS="$ROUNDS" ARMS="$ARMS" LAT_MANIFEST="$ROOT/artifacts/t67-clips/manifest.jsonl" \
    bash scripts/eval/t67_intercalado.sh "$OUT/latencia" > "$OUT/latencia.run.log" 2>&1
  python3 scripts/eval/t67_intercalado.py "$OUT/latencia" --ref ref > "$OUT/latencia/resumen.log" 2>&1
  date +%T > "$OUT/latencia.DONE"; log "latencia fin"
fi

# ---- fase 3: terminos
if [ ! -e "$OUT/terminos.DONE" ]; then
  mkdir -p "$OUT/terminos"; S="$ROOT/artifacts/eval/contexto-largo/charlas.jsonl"
  echo "commit $COMMIT, ctx 0,200 --prefill, $(date), $(uptime)" > "$OUT/terminos/entorno.txt"
  "$PYBIN" scripts/eval/t69_charlas_eval.py --out "$OUT/terminos/ref" --ctx 0,200 --prefill --set "$S" > "$OUT/terminos/ref.log" 2>&1; log "terminos ref rc=$?"
  "$PYBIN" scripts/eval/t69_charlas_eval.py --out "$OUT/terminos/simul" --ctx 0,200 --prefill --simul --set "$S" > "$OUT/terminos/simul.log" 2>&1; log "terminos simul rc=$?"
  date +%T > "$OUT/terminos.DONE"
fi

# ---- fase 4: analisis
python3 scripts/eval/t80_analisis.py "$OUT" > "$OUT/ANALISIS.md" 2>"$OUT/analisis.err"; log "analisis rc=$?"
date +%T > "$OUT/DONE"; log fin
