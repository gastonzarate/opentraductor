#!/usr/bin/env bash
# Cola nocturna: espera a que termine T-48 (PID dado) y el A/B de QA (/tmp/qa-ab/DONE),
# después corre en serie T-50 (3 tamaños x 10 min). La fila baseline vs incremental salió en T-70. Escribe artifacts/t50/QUEUE_DONE.
# Uso: nohup scripts/eval/t48_t50_queue.sh <pid_t48> > artifacts/t50/queue.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
T48_PID="${1:?pid de t48_asr_compare.sh}"
mkdir -p artifacts/t50
while kill -0 "$T48_PID" 2>/dev/null; do sleep 20; done
echo "T-48 terminó $(date +%H:%M:%S)"
for _ in $(seq 1 60); do [ -e /tmp/qa-ab/DONE ] && break; sleep 10; done   # A/B de QA: ~2 min, espera hasta 10
echo "qa-ab: $(ls /tmp/qa-ab/DONE 2>/dev/null || echo 'sin DONE, sigo') $(date +%H:%M:%S)"
ASR=moonshine-tiny SESSION_MIN=10 scripts/eval/t50_run_all.sh artifacts/t50/ram
echo "T-50 tamaños terminó $(date +%H:%M:%S)"
date +%H:%M:%S > artifacts/t50/QUEUE_DONE
