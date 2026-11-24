"""T-88: simulación offline del primer corte en frases largas sobre trazas reales del banco (sin modelos).

Repite los asr_partial de cada traza (con su hora de llegada) a través de varias políticas de corte del
primer trozo y estima cuándo sale la primera voz: hora del parcial que dispara el corte + demora MT+TTS
medida en la propia traza (primer tts_chunk - parcial que disparó la 1.ª cláusula con la regla de hoy).

Uso: python3 scripts/eval/t88_first_cut_sim.py artifacts/t80/latencia/simul [--min-len 5]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from runtime.simul import ClauseCommitter  # noqa: E402


def trazas(d: pathlib.Path):
    for f in sorted(d.glob("r*/latency/traces.jsonl")):
        for line in open(f):
            t = json.loads(line)
            if not t.get("warmup"):
                yield f.parent.parent.name, t


def first_cut(partials, policy) -> tuple[float | None, str]:
    """(recv_ns del parcial que dispara la 1.ª pieza, texto de la pieza) del primer segmento."""
    c = ClauseCommitter(**policy)
    for ns, text in partials:
        out = c.update(text)
        if out:
            return ns, out[0]
    return None, ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--min-len", type=float, default=5.0)
    a = ap.parse_args()
    policies = {
        "hoy": dict(min_words=5),
        "coma_sin_espera": dict(min_words=5, comma_lookahead=False),
        "k5": dict(min_words=5, first_max_words=5),
        "k6": dict(min_words=5, first_max_words=6),
        "k5+coma": dict(min_words=5, first_max_words=5, comma_lookahead=False),
        "k4+coma": dict(min_words=5, first_max_words=4, comma_lookahead=False),
    }
    res = {p: [] for p in policies}
    pieces = {p: [] for p in policies}
    for rnd, t in trazas(pathlib.Path(a.dir)):
        s = t["segments"][0]
        if (s.get("end_s") or 0) - (s.get("start_s") or 0) <= a.min_len:
            continue
        ev = t["events"]
        t0 = int(t["t0_ns"])
        seg1 = ev[0].get("commit_id") if ev else None
        parts = [(int(e["recv_ns"]), e["text"]) for e in ev
                 if e["kind"] == "asr_partial" and e.get("commit_id") == seg1 and e.get("text")]
        tts = [int(e["recv_ns"]) for e in ev if e["kind"] == "tts_chunk"]
        if not parts or not tts:
            continue
        start_ns = t0 + int(s["start_s"] * 1e9)
        real_first = (min(tts) - start_ns) / 1e6
        ns_hoy, _ = first_cut(parts, policies["hoy"])
        # demora MT+TTS de una pieza: de la traza si hoy hubo cláusula antes del endpoint; si no, 400 ms
        d = (min(tts) - ns_hoy) / 1e6 if ns_hoy else 0
        d = d if 0 < d < 1500 else 400
        for p, pol in policies.items():
            ns, txt = first_cut(parts, pol)
            est = (ns - start_ns) / 1e6 + d if ns else real_first
            est = min(est, real_first)
            res[p].append(est)
            pieces[p].append(f"{s['item_id'][-4:]} {rnd}: {txt!r}")
    print("| política | n | 1.ª voz p50 ms | p95 ms |\n|---|---|---|---|")
    for p, v in res.items():
        v = sorted(v)
        if v:
            print(f"| {p} | {len(v)} | {statistics.median(v):.0f} | {v[min(len(v) - 1, int(0.95 * len(v)))]:.0f} |")
    for p in policies:
        print(f"\n### {p}: primer trozo\n" + "\n".join(sorted(set(x.split(': ', 1)[0][:4] + ': ' + x.split(': ', 1)[1] for x in pieces[p]))))


if __name__ == "__main__":
    main()
