#!/usr/bin/env python3
"""T-67 [bench]: huecos de reproducción entre fragmentos TTS (Kokoro en streaming) a partir de traces.jsonl.

Uso: python3 scripts/eval/t67_tts_gaps.py <brazo>/latency [<brazo2>/latency ...]

Por frase (commit_id de cada traza no warmup) con >=2 eventos tts_chunk, simula la reproducción secuencial:
  el chunk i empieza a sonar en max(recv_i, fin_{i-1}); fin_i = inicio_i + pcm_samples/sample_rate.
  hueco_i = max(0, recv_i - fin_{i-1})  (silencio audible entre fragmentos; 0 = sin corte).
Además: nº de chunks, 1er audio antes del último mt_final (adelanto, ms) y fin de habla -> 1er chunk.
Es un modelo de la cola de salida (el adaptador entrega chunks a recv_ns); no mide el dispositivo de audio.
"""
import json
import statistics as st
import sys


def pct(v, q):
    v = sorted(v)
    if not v:
        return float("nan")
    k = (len(v) - 1) * q
    f = int(k)
    c = min(f + 1, len(v) - 1)
    return v[f] + (v[c] - v[f]) * (k - f)


def analyze(path):
    n = multi = con_hueco = antes_mt = 0
    chunks, gaps, maxgaps, adelanto = [], [], [], []
    with open(path + "/traces.jsonl") as fh:
        for line in fh:
            t = json.loads(line)
            if t.get("warmup") or str(t.get("stream_id", "")).startswith("warmup"):
                continue
            por_commit = {}
            for e in t["events"]:
                por_commit.setdefault(e.get("commit_id"), []).append(e)
            for ev in por_commit.values():
                tts = sorted((e for e in ev if e["kind"] == "tts_chunk"), key=lambda e: e["recv_ns"])
                if not tts:
                    continue
                n += 1
                chunks.append(len(tts))
                mts = [e["recv_ns"] for e in ev if e["kind"] == "mt_final"]
                if mts and tts[0]["recv_ns"] < max(mts):
                    antes_mt += 1
                    adelanto.append((max(mts) - tts[0]["recv_ns"]) / 1e6)
                if len(tts) < 2:
                    continue
                multi += 1
                fin_prev = tts[0]["recv_ns"] + tts[0]["pcm_samples"] / tts[0]["sample_rate"] * 1e9
                fg = []
                for e in tts[1:]:
                    g = max(0.0, (e["recv_ns"] - fin_prev) / 1e6)
                    fg.append(g)
                    ini = max(e["recv_ns"], fin_prev)
                    fin_prev = ini + e["pcm_samples"] / e["sample_rate"] * 1e9
                gaps.extend(fg)
                maxgaps.append(max(fg))
                if max(fg) > 0:
                    con_hueco += 1
    return dict(n=n, multi=multi, con_hueco=con_hueco, antes_mt=antes_mt, chunks=chunks, gaps=gaps,
                maxgaps=maxgaps, adelanto=adelanto)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    print("| brazo | frases | con >=2 chunks | chunks/frase | frases con hueco | hueco máx p50 | hueco máx p95 | hueco máx max | 1er audio antes de mt_final | adelanto p50 (ms) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for p in sys.argv[1:]:
        r = analyze(p.rstrip("/"))
        name = p.rstrip("/").split("/")[-2] if p.rstrip("/").endswith("latency") else p
        mg = r["maxgaps"]
        print(f"| {name} | {r['n']} | {r['multi']} | {st.mean(r['chunks']) if r['chunks'] else float('nan'):.2f} | "
              f"{r['con_hueco']}/{r['multi']} | {pct(mg, .5):.0f} | {pct(mg, .95):.0f} | {max(mg) if mg else 0:.0f} | "
              f"{r['antes_mt']}/{r['n']} | {pct(r['adelanto'], .5):.0f} |")


if __name__ == "__main__":
    main()
