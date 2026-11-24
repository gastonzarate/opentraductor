"""T-88: simulación offline del primer trozo «con un solo parcial» (sin esperar a que el parcial siguiente
confirme las palabras) sobre trazas reales del banco (sin modelos).

La regla de hoy (local agreement) necesita dos parciales seguidos que coincidan: con un parcial cada ~0,64 s
eso suma ~0,64 s al primer trozo. Aquí se prueba cortar el primer trozo apenas un parcial trae k palabras
más un margen de m palabras (las m últimas del parcial son las inestables y no se usan). Se mide:
- hora estimada de la primera voz desde el inicio de habla (parcial que dispara + demora MT+TTS de la traza);
- si el trozo cortado es prefijo del asr_final (si no, el motor tiene que retraducir: trozo «malo»).

Uso: python3 scripts/eval/t88_one_partial_sim.py artifacts/t80/latencia/simul [--min-len 5]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from runtime.simul import ClauseCommitter, _norm, first_wait_k_end  # noqa: E402


def trazas(d: pathlib.Path):
    for f in sorted(d.glob("r*/latency/traces.jsonl")):
        for line in open(f):
            t = json.loads(line)
            if not t.get("warmup"):
                yield f.parent.parent.name, t


def cut_two(parts, k):
    c = ClauseCommitter(min_words=5, first_max_words=k, comma_lookahead=False)
    for ns, text in parts:
        out = c.update(text)
        if out:
            return ns, out[0]
    return None, ""


def cut_one(parts, k, m):
    for ns, text in parts:
        w = text.split()
        if m:
            w = w[:-m]
        if len(w) < k:
            continue
        end = first_wait_k_end(w, k)
        if end:
            return ns, " ".join(w[:end])
    return None, ""


def es_prefijo(piece: str, final: str) -> bool:
    a, b = [_norm(x) for x in piece.split()], [_norm(x) for x in final.split()]
    return bool(a) and b[: len(a)] == a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--min-len", type=float, default=5.0)
    a = ap.parse_args()
    pols = {"k5 (2 parciales, hoy flag)": ("two", 5, 0)}
    for k in (4, 5):
        for m in (1, 2):
            pols[f"k{k} 1 parcial m{m}"] = ("one", k, m)
    res = {p: [] for p in pols}
    bad = {p: 0 for p in pols}
    ej = {p: [] for p in pols}
    for d in a.dirs:
        for rnd, t in trazas(pathlib.Path(d)):
            s = t["segments"][0]
            if (s.get("end_s") or 0) - (s.get("start_s") or 0) <= a.min_len:
                continue
            ev = t["events"]
            t0 = int(t["t0_ns"])
            seg1 = ev[0].get("commit_id") if ev else None
            parts = [(int(e["recv_ns"]), e["text"]) for e in ev
                     if e["kind"] == "asr_partial" and e.get("commit_id") == seg1 and e.get("text")]
            fin = next((e["text"] for e in ev if e["kind"] == "asr_final" and e.get("commit_id") == seg1), "")
            tts = [int(e["recv_ns"]) for e in ev if e["kind"] == "tts_chunk"]
            if not parts or not tts or not fin:
                continue
            start_ns = t0 + int(s["start_s"] * 1e9)
            real_first = (min(tts) - start_ns) / 1e6
            ns_ref, _ = cut_two(parts, 5)
            d_mt = (min(tts) - ns_ref) / 1e6 if ns_ref else 0
            d_mt = d_mt if 0 < d_mt < 1500 else 400
            for p, (kind, k, m) in pols.items():
                ns, txt = cut_two(parts, k) if kind == "two" else cut_one(parts, k, m)
                est = min((ns - start_ns) / 1e6 + d_mt if ns else real_first, real_first)
                res[p].append(est)
                if txt and not es_prefijo(txt, fin):
                    bad[p] += 1
                    ej[p].append(f"{s['item_id'][-4:]}: {txt!r} vs {fin[:60]!r}")
    print("| política | n | 1.ª voz p50 ms | p95 ms | trozo no es prefijo del final |\n|---|---|---|---|---|")
    for p, v in res.items():
        v = sorted(v)
        if v:
            print(f"| {p} | {len(v)} | {statistics.median(v):.0f} | "
                  f"{v[min(len(v) - 1, int(0.95 * len(v)))]:.0f} | {bad[p]} |")
    for p in pols:
        if ej[p]:
            print(f"\n### {p}: trozos que no coinciden con el final\n" + "\n".join(sorted(set(ej[p]))))


if __name__ == "__main__":
    main()
