#!/usr/bin/env python3
"""[bench] T-94: tiempo de fin de habla -> hipótesis final de Moonshine (stop() del stream), a ritmo de reloj.

Uso: t94_finish_probe.py --system moonshine-tiny|moonshine-small --lines RAW.json --audio WAV --n 12 [--keyterms F] --out JSON
Toma N frases de la transcripción (ya hecha) de 3 a 9 s, repartidas por la charla; para cada una alimenta su audio en
trozos de 100 ms en tiempo real + 300 s de cola de silencio NO (se llama stop() justo al acabar el audio = fin de habla),
y mide el tiempo de stop(). Mismo comando que usa MoonshineStreamer.finish(). Mide también CPU de proceso total.
"""
import argparse, json, resource, statistics, sys, time, wave
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from t94_asr_run import load_wav  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--lines", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--keyterms")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from moonshine_voice import ModelArch, Transcriber
    from runtime import config as C

    audio = load_wav(a.audio)
    ls = [l for l in json.loads(Path(a.lines).read_text())["lines"] if 3 <= l["end"] - l["start"] <= 9]
    step = max(1, len(ls) // a.n)
    pick = ls[::step][: a.n]
    dname, arch = C.MOONSHINE_MODELS[a.system]
    tr = Transcriber(model_path=str(C.moonshine_model_dir(a.system)), model_arch=getattr(ModelArch, arch), update_interval=0.1)
    if a.keyterms:
        tr.set_keyterms([x.strip() for x in Path(a.keyterms).read_text().splitlines() if x.strip()])
    fin, rows = [], []
    c0 = time.process_time()
    w0 = time.perf_counter()
    for l in pick:
        seg = audio[int(l["start"] * 16000): int(l["end"] * 16000)]
        lines = {}
        st = tr.create_stream(update_interval=0.1)
        st.add_listener(lambda ev, lines=lines: lines.__setitem__(ev.line.line_id, ev.line.text) if getattr(ev, "line", None) else None)
        st.start()
        t_start = time.perf_counter()
        for i in range(0, len(seg), 1600):
            st.add_audio(seg[i:i + 1600].tolist(), 16000)
            tgt = t_start + (i + 1600) / 16000
            d = tgt - time.perf_counter()
            if d > 0:
                time.sleep(d)
        t0 = time.perf_counter()
        st.stop()
        ms = (time.perf_counter() - t0) * 1000
        st.close()
        fin.append(ms)
        rows.append({"start": l["start"], "dur": round(len(seg) / 16000, 1), "stop_ms": round(ms), "text": " ".join(t for _, t in sorted(lines.items()))})
    tr.close()
    q = lambda v, p: sorted(v)[min(len(v) - 1, int(round(p * (len(v) - 1))))]
    res = {"system": a.system, "keyterms": bool(a.keyterms), "n": len(fin), "stop_ms_p50": round(statistics.median(fin)),
           "stop_ms_p95": round(q(fin, 0.95)), "cpu_s": round(time.process_time() - c0, 1), "wall_s": round(time.perf_counter() - w0, 1),
           "cpu_cores_avg": round((time.process_time() - c0) / (time.perf_counter() - w0), 2),
           "maxrss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6), "rows": rows}
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print({k: v for k, v in res.items() if k != "rows"})


if __name__ == "__main__":
    main()
