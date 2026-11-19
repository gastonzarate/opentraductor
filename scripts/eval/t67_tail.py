#!/usr/bin/env python3
"""[bench] T-67: asr_tail y turn de adapter_metadata (da867a6+) por brazo.

Uso: t67_tail.py <dir latency de brazo> [...]   (cada dir con run.json)
asr_tail: n, caught_up, eq, eq_caught_up (frases con ASR al dia en el endpoint / parcial ya final).
turn: calls, turn, error, enabled, cost_ms (p50/p95 por consulta). Los contadores del engine son
acumulados de toda la corrida (incluye el warmup). Solo medicion.
"""
import json, pathlib, sys


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))] if v else float("nan")


for d in map(pathlib.Path, sys.argv[1:]):
    arm = d.parent.name
    try:
        m = json.loads((d / "run.json").read_text()).get("meta", {}).get("adapter_metadata", {})
    except Exception as e:
        print(f"{arm}: sin run.json ({e})"); continue
    a, t = m.get("asr_tail"), m.get("turn")
    if a is None and t is None:
        print(f"{arm}: sin asr_tail/turn (commit anterior a da867a6)"); continue
    a, t = a or {}, t or {}
    n = a.get("n") or 0
    r = lambda k: f"{a.get(k)} ({100 * (a.get(k) or 0) / n:.0f}%)" if n else str(a.get(k))
    c = t.get("cost_ms") or []
    print(f"{arm}: asr_tail n={n} caught_up={r('caught_up')} eq={r('eq')} eq_caught_up={r('eq_caught_up')} | "
          f"turn enabled={t.get('enabled')} calls={t.get('calls')} turn={t.get('turn')} error={t.get('error')} "
          f"cost_ms p50={pct(c, .5):.1f} p95={pct(c, .95):.1f} n={len(c)}")
