#!/usr/bin/env bash
# T-58: espera a que la máquina esté en reposo antes de medir latencias.
# Uso: scripts/eval/wait_quiet.sh [MAX_LOAD=3] [MUESTRAS_SEGUIDAS=4] [TIMEOUT_S=600]
# Sale 0 si loadavg(1 min) < MAX_LOAD durante MUESTRAS_SEGUIDAS lecturas separadas 15 s; 1 si vence el timeout (imprime la carga y los 5 procesos que más CPU usan).
MAX="${1:-3}"; NEED="${2:-4}"; TMO="${3:-600}"; ok=0; t0=$(date +%s)
while :; do
  l=$(sysctl -n vm.loadavg | awk '{print $2}')
  if awk -v l="$l" -v m="$MAX" 'BEGIN{exit !(l<m)}'; then ok=$((ok+1)); else ok=0; fi
  [ "$ok" -ge "$NEED" ] && { echo "reposo: load1=$l <$MAX x$NEED"; exit 0; }
  [ $(( $(date +%s) - t0 )) -ge "$TMO" ] && { echo "sin reposo tras ${TMO}s: load1=$l (máx. $MAX)"; ps -axo pcpu,comm | sort -nr | head -5; exit 1; }
  sleep 15
done
