"""Análisis OFFLINE (sin modelos) de silencio/quietud inicial en WAV exportados vs línea de tiempo de chunks.

Explica la brecha entre «primer PCM no silencioso disponible» (recv del primer chunk con algún bloque
> umbral) y «reproducción estimada» (inicio del chunk en la cola simulada + offset del primer bloque
no silencioso DENTRO del chunk). Usa el mismo detector que el benchmark (``audio.first_nonsilent_offset_s``,
bloques de 10 ms, -50 dBFS por defecto).

    .venv/bin/python scripts/eval/leading_quiet.py --run DIR [DIR ...] --out artifacts/benchmark/clone-pilot1/leading_quiet.json
Cada DIR contiene run.json, traces.jsonl y audio/export_index.jsonl.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmark import audio as A  # noqa: E402


def analyze(run_dir: Path, threshold_dbfs: float) -> dict:
    index = {}
    for line in (run_dir / "audio" / "export_index.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        index["+".join(r["item_ids"])] = r
    out = []
    for line in (run_dir / "traces.jsonl").read_text(encoding="utf-8").splitlines():
        t = json.loads(line)
        if t.get("warmup") or not t["segments"]:
            continue
        item = t["segments"][0]["item_id"]
        chunks = [e for e in t["events"] if e["kind"] == "tts_chunk"]
        if not chunks:
            continue
        t0 = t["t0_ns"]
        first_nonsil = next((i for i, c in enumerate(chunks)
                             if c["first_nonsilent_offset_s"] is not None), None)
        # línea de tiempo de la cola simulada (mismo criterio que metrics.playout_schedule)
        playhead, starts = None, []
        for c in chunks:
            dur = round(c["pcm_samples"] / c["sample_rate"] * 1e9)
            s = c["recv_ns"] if playhead is None else max(c["recv_ns"], playhead)
            playhead = s + dur
            starts.append(s)
        wav = Path(index[item]["_dir"]) / index[item]["wav"] if "_dir" in index[item] else run_dir / "audio" / index[item]["wav"]
        pcm, sr = A.read_wav_pcm16_mono(wav)
        wav_lead = A.first_nonsilent_offset_s(pcm, sr, threshold_dbfs=threshold_dbfs)
        c = chunks[first_nonsil] if first_nonsil is not None else None
        avail_ms = (c["recv_ns"] - t0) / 1e6 if c else None
        play_ms = ((starts[first_nonsil] - t0) / 1e6 + c["first_nonsilent_offset_s"] * 1000) if c else None
        quiet_before_ms = sum(x["pcm_samples"] / x["sample_rate"] for x in chunks[:first_nonsil]) * 1000 if c else None
        out.append({
            "item_id": item, "n_chunks": len(chunks),
            "first_chunk_s": chunks[0]["pcm_samples"] / chunks[0]["sample_rate"],
            "first_nonsilent_chunk_index": first_nonsil,
            "available_ms": avail_ms, "playout_est_ms": play_ms,
            "gap_ms": (play_ms - avail_ms) if c else None,
            "offset_inside_first_nonsilent_chunk_ms": c["first_nonsilent_offset_s"] * 1000 if c else None,
            "wav_leading_quiet_ms": wav_lead * 1000 if wav_lead is not None else None,
            "chunk_timeline_leading_quiet_ms": (quiet_before_ms + c["first_nonsilent_offset_s"] * 1000) if c else None,
            "wav_vs_timeline_diff_ms": (wav_lead * 1000 - (quiet_before_ms + c["first_nonsilent_offset_s"] * 1000)) if (c and wav_lead is not None) else None,
        })
    def p50(k):
        v = [x[k] for x in out if x[k] is not None]
        return statistics.median(v) if v else None
    return {"run_dir": str(run_dir), "threshold_dbfs": threshold_dbfs, "n": len(out), "items": out,
            "p50_available_ms": p50("available_ms"), "p50_playout_est_ms": p50("playout_est_ms"),
            "p50_gap_ms": p50("gap_ms"), "p50_wav_leading_quiet_ms": p50("wav_leading_quiet_ms"),
            "caveat": f"n={len(out)}: mediana descriptiva, sin IC ni afirmación sobre 300 ms; modo texto (sin ASR/endpoint)."}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="+", required=True)
    ap.add_argument("--threshold-dbfs", type=float, default=A.DEFAULT_SILENCE_THRESHOLD_DBFS)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = [analyze(Path(d), a.threshold_dbfs) for d in a.run]
    Path(a.out).write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    for r in res:
        print(r["run_dir"], {k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items() if k.startswith("p50")})
        for x in r["items"]:
            print("  ", x["item_id"], {k: (round(v, 1) if isinstance(v, float) else v) for k, v in x.items() if k not in ("item_id",)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
