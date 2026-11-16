"""T-50: RAM pico del pipeline completo de la tienda (Silero ONNX + Moonshine + Qwen MLX + Kokoro) con un tamaño de Qwen.

Un proceso por tamaño, los tres modelos cargados a la vez, como en la app. Recorre los 12 clips openslr61-test
(VAD + ASR + Qwen + Kokoro) y después las 54 frases de text_meetings_ar_dev (Qwen + Kokoro). Mide:
  * ru_maxrss del proceso y mx.get_peak_memory() (memoria Metal; en Apple Silicon es la misma RAM unificada),
  * swap usado y pageouts del sistema antes y después (vm.swapusage, vm_stat): si crecen, no entra cómodo,
  * p50/p95 de MT y del primer chunk de TTS por frase.
Envolverlo con /usr/bin/time -l da además el «peak memory footprint», que es la cifra que usa macOS para la presión de memoria.
Correr con la máquina sola y en serie (turno en #t-45). Sin red: HF_HUB_OFFLINE=1.

  HF_HUB_OFFLINE=1 PYTHONPATH=. /usr/bin/time -l .venv-inc-llm/bin/python scripts/eval/t50_pipeline_ram.py \\
      --qwen-dir artifacts/analysis/t50-qwen-sizes/models/qwen3-1.7b-4bit --out artifacts/t50/ram/qwen3-1.7b.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import resource
import statistics
import subprocess
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SLR = ROOT / "data" / "manifests" / "openslr61_es_ar_sample.jsonl"
TXT = ROOT / "data" / "manifests" / "text_meetings_ar_dev.jsonl"
# mismo prompt que probe_qwen_sizes.py / MP4-A
SYSTEM = ("You translate Argentine Spanish speech from software team meetings into natural US English. "
          "Output only the English translation, with no notes, quotes or explanations. "
          "Keep numbers, negations, names and technical terms exact.")


def sysmem() -> dict:
    swap = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.strip()
    used = re.search(r"used = ([\d.]+)M", swap)
    vm = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    po = re.search(r"Pageouts:\s+(\d+)", vm)
    return {"swap_used_mb": float(used.group(1)) if used else None, "pageouts": int(po.group(1)) if po else None,
            "loadavg": os.getloadavg()}


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))] if xs else None


def read_pcm16(path: Path) -> bytes:
    with wave.open(str(path)) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2, path
        return w.readframes(w.getnframes())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen-dir", required=True)
    ap.add_argument("--asr", default="moonshine-tiny", choices=["moonshine-tiny", "moonshine-small"])
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import mlx.core as mx
    from mlx_lm import generate, load

    from runtime import components as K
    from runtime.config import MOONSHINE_MODELS, moonshine_model_dir

    before = sysmem()
    t0 = time.monotonic()
    vad = K.SileroVAD()
    asr = K.MoonshineStreamer(moonshine_model_dir(a.asr), MOONSHINE_MODELS[a.asr][1], 0.16)
    model, tok = load(a.qwen_dir)
    tts = K.KokoroTTS(a.voice, 1.0)
    load_s = time.monotonic() - t0
    rss_after_load = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # bytes en macOS

    def translate(es: str) -> tuple[str, float]:
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"Spanish:\n{es}"}]
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        t = time.monotonic()
        out = generate(model, tok, prompt=prompt, max_tokens=3 * len(es.split()) + 16, verbose=False)
        return out.strip().split("\n")[0].strip(), (time.monotonic() - t) * 1000

    def speak(en: str) -> float:
        t = time.monotonic()
        first = None
        for _ in tts.synth(en):
            first = first if first is not None else (time.monotonic() - t) * 1000
        return first or 0.0

    rows, mt_ms, tts_ms = [], [], []
    translate("Hola, ¿me escuchan?")  # warmup
    speak("Hello, can you hear me?")
    for line in SLR.read_text().splitlines():
        it = json.loads(line)
        if it["split"] != "test":
            continue
        pcm = read_pcm16(ROOT / "data" / "audio" / it["audio_path"])
        vad.reset()
        vad.probs(pcm)
        asr.start()
        asr.add(pcm)
        es = asr.finish()
        en, ms = translate(es) if es else ("", 0.0)
        mt_ms.append(ms)
        tts_ms.append(speak(en) if en else 0.0)
        rows.append({"id": it["item_id"], "asr": es, "en": en})
    for line in TXT.read_text().splitlines():
        it = json.loads(line)
        en, ms = translate(it["text_es"])
        mt_ms.append(ms)
        tts_ms.append(speak(en))
        rows.append({"id": it["item_id"], "es": it["text_es"], "en": en})
    after = sysmem()
    res = {
        "qwen_dir": a.qwen_dir, "asr": a.asr, "load_s": round(load_s, 1),
        "rss_after_load_mb": round(rss_after_load), "rss_peak_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6),
        "metal_peak_mb": round(mx.get_peak_memory() / 1e6),
        "swap_delta_mb": (after["swap_used_mb"] or 0) - (before["swap_used_mb"] or 0),
        "pageouts_delta": (after["pageouts"] or 0) - (before["pageouts"] or 0),
        "mt_ms": {"p50": statistics.median(mt_ms), "p95": pct(mt_ms, 0.95)},
        "tts_first_ms": {"p50": statistics.median(tts_ms), "p95": pct(tts_ms, 0.95)},
        "before": before, "after": after, "n": len(rows), "rows": rows,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("rows", "before", "after")}, ensure_ascii=False))
    asr.close()


if __name__ == "__main__":
    main()
