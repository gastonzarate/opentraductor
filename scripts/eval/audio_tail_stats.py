#!/usr/bin/env python3
"""[bench] Cola de silencio y pausas internas de los WAV exportados por compare_configs.sh (T-47).

Uso: audio_tail_stats.py DIR_CON_export_index.jsonl [...]  ->  JSON por directorio en stdout.
Silencio = bloques de 5 ms bajo -60 dBFS (misma definición que el tail trim). Pausa = silencio interno >= 300 ms.
"""
import json
import sys
import wave
from pathlib import Path

import numpy as np

BLOCK_MS, THR_DB, PAUSE_MS = 5, -60.0, 300


def stats(path: Path) -> dict:
    with wave.open(str(path)) as w:
        sr, n = w.getframerate(), w.getnframes()
        x = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float64) / 32768.0
    b = int(sr * BLOCK_MS / 1000)
    nb = len(x) // b
    if nb == 0:
        return {"dur_s": len(x) / sr, "tail_ms": 0.0, "pauses": 0, "max_pause_ms": 0.0}
    rms = np.sqrt((x[: nb * b].reshape(nb, b) ** 2).mean(axis=1) + 1e-20)
    sil = 20 * np.log10(rms) < THR_DB
    tail = 0
    while tail < nb and sil[nb - 1 - tail]:
        tail += 1
    # silencios internos (sin cabeza ni cola)
    first = int(np.argmax(~sil)) if (~sil).any() else nb
    runs, cur = [], 0
    for s in sil[first: nb - tail]:
        if s:
            cur += 1
        else:
            if cur:
                runs.append(cur)
            cur = 0
    ms = [r * BLOCK_MS for r in runs]
    p = [m for m in ms if m >= PAUSE_MS]
    return {"dur_s": round(len(x) / sr, 3), "tail_ms": tail * BLOCK_MS, "pauses": len(p), "max_pause_ms": float(max(ms, default=0))}


def main() -> int:
    for d in map(Path, sys.argv[1:]):
        rows = []
        for line in (d / "export_index.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r.get("exported"):
                rows.append({"stream": r["stream_id"], **stats(d / r["wav"])})
        tails = [r["tail_ms"] for r in rows]
        out = {"dir": str(d), "n": len(rows), "tail_ms_max": max(tails, default=None), "tail_ms_mean": round(float(np.mean(tails)), 1) if tails else None,
               "pauses_ge_300ms": sum(r["pauses"] for r in rows), "max_internal_pause_ms": max((r["max_pause_ms"] for r in rows), default=None),
               "dur_s_total": round(sum(r["dur_s"] for r in rows), 2), "rows": rows}
        print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
