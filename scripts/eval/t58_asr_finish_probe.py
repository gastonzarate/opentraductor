"""T-58: ¿cuánto cuesta ``MoonshineStreamer.finish()`` y cuánto difiere del último parcial?

Alimenta cada clip en chunks de ``asr_chunk_ms`` hasta el fin de habla del manifest
(aprox. de ``asr_skip_silent_tail``: la cola de silencio no llega al ASR), y mide:
- ms de ``finish()`` (stop + pasada final),
- audio sin pasar por el modelo al llegar al endpoint (``_stream_time - _last_update_time``),
- WER del último parcial y del final contra la referencia, y si son iguales.

Solo disco, sin red. Uso:
  uv run python scripts/eval/t58_asr_finish_probe.py [--backend moonshine-tiny] [--out ruta.jsonl]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmark.quality import normalize_text  # noqa: E402
from runtime import components as K  # noqa: E402
from runtime.config import MOONSHINE_MODELS, PipelineConfig, moonshine_model_dir  # noqa: E402

MANIFESTS = (
    "artifacts/benchmark/offline-review/manifest.jsonl",
    "data/manifests/openslr61_es_ar_sample.jsonl",
    "data/manifests/fleurs_es_sample.jsonl",
)


def _wer(ref: str, hyp: str) -> float:
    import jiwer

    return jiwer.wer(normalize_text(ref), normalize_text(hyp) or "<vacío>")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="moonshine-tiny")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = PipelineConfig()
    chunk = int(16000 * cfg.asr_chunk_ms / 1000) * 2
    asr = K.MoonshineStreamer(moonshine_model_dir(a.backend), MOONSHINE_MODELS[a.backend][1],
                              cfg.moonshine_update_interval_s)
    asr.start(); asr.add(bytes(16000)); asr.finish()  # warmup
    rows, seen = [], set()
    for m in MANIFESTS:
        for line in (ROOT / m).read_text().splitlines():
            it = json.loads(line)
            if it["audio_path"] in seen:
                continue
            seen.add(it["audio_path"])
            with wave.open(str(ROOT / "data/audio" / it["audio_path"])) as w:
                pcm = w.readframes(w.getnframes())
            end_b = int((it.get("utterance_end_s") or it["duration_s"]) * 16000) * 2
            end_b = min(len(pcm), -(-end_b // chunk) * chunk)  # redondea al chunk siguiente
            asr.start()
            for i in range(0, end_b, chunk):
                asr.add(pcm[i:i + chunk])
            st = asr._stream
            stale_ms = (st._stream_time - st._last_update_time) * 1000
            partial = asr._text()
            t0 = time.perf_counter()
            final = asr.finish()
            fin_ms = (time.perf_counter() - t0) * 1000
            ref = it["transcript_es"]
            rows.append({
                "audio": it["audio_path"], "words": len(ref.split()), "fed_s": end_b / 32000,
                "finish_ms": round(fin_ms, 1), "stale_ms": round(stale_ms), "same": normalize_text(partial) == normalize_text(final),
                "wer_partial": round(_wer(ref, partial), 3), "wer_final": round(_wer(ref, final), 3),
                "partial": partial, "final": final,
            })
            r = rows[-1]
            print(f"{r['audio'][-34:]:34} fin {r['finish_ms']:6.1f} ms stale {r['stale_ms']:4} ms "
                  f"same {r['same']!s:5} wer p/f {r['wer_partial']:.3f}/{r['wer_final']:.3f}")
    asr.close()
    fin = [r["finish_ms"] for r in rows]
    print(f"\nn={len(rows)} finish p50 {statistics.median(fin):.0f} ms max {max(fin):.0f}; "
          f"iguales {sum(r['same'] for r in rows)}/{len(rows)}; "
          f"WER medio parcial {statistics.mean(r['wer_partial'] for r in rows):.3f} "
          f"final {statistics.mean(r['wer_final'] for r in rows):.3f}")
    if a.out:
        Path(a.out).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")


if __name__ == "__main__":
    main()
