#!/usr/bin/env python3
"""[bench] Resumen del benchmark largo (T-47): backlog, retraso semantico, duracion de salida, cola y pausas. Uso: long_summary.py DIR..."""
import json, subprocess, sys
for d in sys.argv[1:]:
    p = json.load(open(f"{d}/run.json"))["playout"][0]
    st = json.loads(subprocess.check_output([sys.executable, "scripts/eval/audio_tail_stats.py", f"{d}/audio"]))
    print(json.dumps({"dir": d, "chunks": p["n_chunks"], "out_s": round(p["total_output_s"], 1),
        "backlog_max_s": round(p["backlog"]["max_s"], 2), "backlog_mean_s": round(p["backlog"]["mean_s"], 2), "underrun_s": round(p["backlog"]["underrun_s"], 1),
        "lag_max_s": round(p["semantic_lag"]["max_s"], 2), "lag_mean_s": round(p["semantic_lag"]["mean_s"], 2), "lag_trend_s_per_s": round(p["semantic_lag"]["trend_s_per_s"], 3),
        "tail_ms": st["tail_ms_max"], "pauses_ge_300ms": st["pauses_ge_300ms"], "max_pause_ms": st["max_internal_pause_ms"]}))
