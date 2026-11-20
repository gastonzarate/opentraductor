"""T-67 sonda: de dónde salen los ~130-300 ms de "ASR fin" con Moonshine (CPU/ONNX).

En t67-intercalado/new el parcial del endpoint pierde las últimas 1-4 palabras en 16/16 frases
("...toda la planta" -> "...toda la plantación de lechuga"), así que la MT especulativa con el parcial no
puede acertar. Esta sonda alimenta cada clip en tiempo real a MoonshineStreamer (sin Qwen ni Kokoro) y mide,
tras SIL_MS de silencio:
  - finish_ms: lo que tarda stop() (el "ASR fin" del pipeline),
  - words_missing: palabras que le faltan al último parcial respecto del final,
  - lag del parcial: cuánto tarda en aparecer cada palabra nueva del parcial.
Variantes por proceso (el env de ORT se lee al cargar libmoonshine):
  MOONSHINE_ORT_SINGLE_THREAD=1 vs sin env; UPDATE_S (update_interval) 0.1 / 0.05.

Uso (con la Mac sin A/B corriendo):
  .venv/bin/python scripts/eval/t67_asr_tail_probe.py --out artifacts/t67-asrtail/default.json
  MOONSHINE_ORT_SINGLE_THREAD=1 .venv/bin/python scripts/eval/t67_asr_tail_probe.py --out .../single.json
Solo medición: no toca el runtime.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as stx
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from runtime.components import MoonshineStreamer  # noqa: E402
from runtime.config import MOONSHINE_MODELS, moonshine_model_dir  # noqa: E402


def read_pcm(p: Path) -> bytes:
    with wave.open(str(p)) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2, p
        return w.readframes(w.getnframes())


def norm_words(t: str) -> list[str]:
    return "".join(c.lower() if c.isalnum() or c.isspace() else " " for c in t).split()


def run_clip(asr: MoonshineStreamer, pcm: bytes, chunk_ms: int, sil_ms: int, realtime: bool) -> dict:
    step = 16000 * chunk_ms // 1000 * 2
    asr.start()
    t0 = time.perf_counter()
    last, partial_changes = "", []
    for i in range(0, len(pcm), step):
        if realtime:
            due = t0 + (i // 2) / 16000
            d = due - time.perf_counter()
            if d > 0:
                time.sleep(d)
        txt = asr.add(pcm[i:i + step])
        if txt != last:
            partial_changes.append(round((time.perf_counter() - t0) * 1000))
            last = txt
    speech_end = t0 + len(pcm) / 2 / 16000
    silence = b"\x00\x00" * (16000 * sil_ms // 1000)
    for i in range(0, len(silence), step):
        if realtime:
            d = speech_end + (i // 2) / 16000 - time.perf_counter()
            if d > 0:
                time.sleep(d)
        txt = asr.add(silence[i:i + step])
        if txt != last:
            last = txt
    ep_partial = last
    tf = time.perf_counter()
    final = asr.finish()
    finish_ms = (time.perf_counter() - tf) * 1000
    pw, fw = norm_words(ep_partial), norm_words(final)
    common = 0
    while common < min(len(pw), len(fw)) and pw[common] == fw[common]:
        common += 1
    return dict(finish_ms=round(finish_ms, 1), words_final=len(fw), words_partial=len(pw),
                words_missing=len(fw) - common, eq=pw == fw, partial=ep_partial, final=final,
                n_partial_changes=len(partial_changes), last_lat_ms=asr.last_latency_ms)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "artifacts/t67-clips/manifest.jsonl"))
    ap.add_argument("--audio-root", default=str(ROOT / "data/audio"))
    ap.add_argument("--backend", default="moonshine-tiny")
    ap.add_argument("--update-s", type=float, default=float(os.environ.get("UPDATE_S", "0.1")))
    ap.add_argument("--chunk-ms", type=int, default=32)
    ap.add_argument("--sil-ms", type=int, default=64, help="silencio antes del finish (endpoint_short_ms)")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--no-realtime", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gil-hog", action="store_true",
                    help="hilo Python en bucle (sin soltar el GIL más que el intervalo de switch), como el decode de Qwen")
    ap.add_argument("--process", action="store_true", help="ASR en proceso aparte (runtime/asr_process.py)")
    ap.add_argument("--switch-ms", type=float, default=None, help="sys.setswitchinterval (ms) para el proceso")
    a = ap.parse_args()
    if a.switch_ms:
        sys.setswitchinterval(a.switch_ms / 1000)
    if a.gil_hog:
        import threading

        def hog() -> None:
            x = 0
            while True:
                x = (x * 31 + 7) % 1000003

        threading.Thread(target=hog, daemon=True).start()

    items = [json.loads(l) for l in open(a.manifest)]
    if a.process:
        from runtime.asr_process import ProcessMoonshineStreamer as cls
    else:
        cls = MoonshineStreamer
    asr = cls(moonshine_model_dir(a.backend), MOONSHINE_MODELS[a.backend][1], a.update_s)
    run_clip(asr, read_pcm(Path(a.audio_root) / items[0]["audio_path"]), a.chunk_ms, a.sil_ms, False)  # warmup
    rows = []
    for r in range(a.reps):
        for it in items:
            res = run_clip(asr, read_pcm(Path(a.audio_root) / it["audio_path"]), a.chunk_ms, a.sil_ms,
                           not a.no_realtime)
            res.update(item=it["item_id"], rep=r)
            rows.append(res)
            print(f"{it['item_id'][:34]:34} fin {res['finish_ms']:6.1f} ms  faltan {res['words_missing']} pal.",
                  flush=True)
    fm = [x["finish_ms"] for x in rows]
    summ = dict(env_single_thread=os.environ.get("MOONSHINE_ORT_SINGLE_THREAD"), update_s=a.update_s,
                sil_ms=a.sil_ms, n=len(rows), finish_p50=round(stx.median(fm), 1),
                finish_p95=round(sorted(fm)[min(len(fm) - 1, -(-95 * len(fm) // 100) - 1)], 1),
                gil_hog=a.gil_hog, process=a.process, switch_ms=a.switch_ms,
                missing_p50=stx.median(x["words_missing"] for x in rows),
                eq=sum(x["eq"] for x in rows), load1=os.getloadavg()[0])
    print(json.dumps(summ, ensure_ascii=False))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(summary=summ, rows=rows), ensure_ascii=False, indent=1))
    asr.close()


if __name__ == "__main__":
    main()
