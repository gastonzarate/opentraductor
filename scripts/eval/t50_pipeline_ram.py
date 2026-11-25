"""T-50: RAM pico del pipeline completo de la tienda (Silero ONNX + Moonshine + Qwen MLX + Kokoro) con un tamaño de Qwen.

Un proceso por tamaño, los tres modelos cargados a la vez, como en la app. Recorre los 12 clips openslr61-test
(VAD + ASR + Qwen + Kokoro) y después las 54 frases de text_meetings_ar_dev (Qwen + Kokoro). Mide:
  * ru_maxrss del proceso y mx.get_peak_memory() (memoria Metal; en Apple Silicon es la misma RAM unificada),
  * swap usado y pageouts del sistema antes y después (vm.swapusage, vm_stat): si crecen, no entra cómodo,
  * p50/p95 de MT y del primer chunk de TTS por frase.
Envolverlo con /usr/bin/time -l da además el «peak memory footprint», que es la cifra que usa macOS para la presión de memoria.
Con --session-min 10 repite la carga hasta 10 min (criterio de 8 GB: footprint pico ≤ 3,5 GB) y guarda
el phys_footprint pico del proceso (proc_pid_rusage, igual que `footprint -p`).
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


def phys_footprint(pid: int | None = None) -> dict:
    """phys_footprint actual y pico de vida del proceso (proc_pid_rusage, RUSAGE_INFO_V4): la cifra de
    `footprint -p` y del Monitor de Actividad, la que usa macOS para la presión de memoria."""
    import ctypes

    class RU(ctypes.Structure):
        _fields_ = [("uuid", ctypes.c_uint8 * 16), ("f", ctypes.c_uint64 * 40)]

    r = RU()
    if ctypes.CDLL("/usr/lib/libproc.dylib").proc_pid_rusage(int(pid or os.getpid()), 4, ctypes.byref(r)) != 0:
        return {"now_mb": None, "peak_mb": None}
    return {"now_mb": round(r.f[7] / 1e6), "peak_mb": round(r.f[28] / 1e6)}


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
    ap.add_argument("--session-min", type=float, default=0.0,
                    help="tras la pasada de calidad, repetir clips y frases hasta N minutos (criterio 8 GB: 10)")
    ap.add_argument("--cache-limit-mb", type=float, default=None,
                    help="mx.set_cache_limit: tope de la caché de buffers libres de MLX (por defecto MLX no la limita)")
    ap.add_argument("--clear-between", action="store_true",
                    help="T-51: mx.clear_cache() después de cada frase (en el silencio), en vez de un tope fijo")
    a = ap.parse_args()

    import mlx.core as mx
    from mlx_lm import generate, load

    from runtime import components as K
    from runtime.config import MOONSHINE_MODELS, moonshine_model_dir

    if a.cache_limit_mb is not None:
        mx.set_cache_limit(int(a.cache_limit_mb * 1e6))
    before = sysmem()
    t0 = time.monotonic()
    vad = K.SileroVAD()
    asr = K.MoonshineStreamer(moonshine_model_dir(a.asr), MOONSHINE_MODELS[a.asr][1], 0.16)
    model, tok = load(a.qwen_dir)
    tts = K.KokoroTTS(a.voice, 1.0)
    load_s = time.monotonic() - t0
    rss_after_load = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # bytes en macOS
    fp_after_load = phys_footprint()

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

    clear_ms: list[float] = []

    def idle() -> None:  # fin de frase: lo que el pipeline haría en el silencio
        if a.clear_between:
            t = time.monotonic()
            mx.clear_cache()
            clear_ms.append((time.monotonic() - t) * 1000)

    rows, mt_ms, tts_ms = [], [], []
    translate("Hola, ¿me escuchan?")  # warmup
    speak("Hello, can you hear me?")
    def one_pass(keep: bool) -> None:
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
            idle()
            if keep:
                rows.append({"id": it["item_id"], "asr": es, "en": en})
        for line in TXT.read_text().splitlines():
            it = json.loads(line)
            en, ms = translate(it["text_es"])
            mt_ms.append(ms)
            tts_ms.append(speak(en))
            idle()
            if keep:
                rows.append({"id": it["item_id"], "es": it["text_es"], "en": en})

    t_sess = time.monotonic()
    one_pass(True)
    passes, samples = 1, [{"t_s": 0, **phys_footprint(), **sysmem()}]
    while time.monotonic() - t_sess < a.session_min * 60:  # sesión larga: misma carga, solo memoria y latencia
        one_pass(False)
        passes += 1
        samples.append({"t_s": round(time.monotonic() - t_sess), **phys_footprint(), **sysmem()})
    after = sysmem()
    fp = phys_footprint()
    res = {
        "qwen_dir": a.qwen_dir, "asr": a.asr, "cache_limit_mb": a.cache_limit_mb, "clear_between": a.clear_between,
        "clear_ms": {"p50": statistics.median(clear_ms), "p95": pct(clear_ms, 0.95)} if clear_ms else None, "load_s": round(load_s, 1),
        "rss_after_load_mb": round(rss_after_load), "rss_peak_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6),
        "metal_peak_mb": round(mx.get_peak_memory() / 1e6),
        "footprint_after_load_mb": fp_after_load["now_mb"], "footprint_peak_mb": fp["peak_mb"],
        "session_s": round(time.monotonic() - t_sess), "passes": passes, "samples": samples,
        "swap_delta_mb": (after["swap_used_mb"] or 0) - (before["swap_used_mb"] or 0),
        "pageouts_delta": (after["pageouts"] or 0) - (before["pageouts"] or 0),
        "mt_ms": {"p50": statistics.median(mt_ms), "p95": pct(mt_ms, 0.95)},
        "tts_first_ms": {"p50": statistics.median(tts_ms), "p95": pct(tts_ms, 0.95)},
        "before": before, "after": after, "n": len(rows), "rows": rows,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("rows", "before", "after", "samples")}, ensure_ascii=False))
    asr.close()


if __name__ == "__main__":
    main()
