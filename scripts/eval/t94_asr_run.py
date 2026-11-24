#!/usr/bin/env python3
"""[bench] T-94: transcribe un WAV 16 kHz mono con un reconocedor y guarda líneas con tiempos + CPU/memoria.

Uso: t94_asr_run.py --system moonshine-tiny|moonshine-small|parakeet [--keyterms FILE] [--boost X] --audio WAV --out JSON
Moonshine: stream continuo (VAD y segmentación propios de libmoonshine), audio en trozos de 100 ms sin pausas
(precisión en lote, no latencia). Parakeet: transcribe() por bloques de 60 s con solape de 15 s (MLX, GPU).
Salida: {"system","lines":[{"start","end","text"}],"wall_s","cpu_s","maxrss_mb","audio_s"}. Solo disco, sin red.
"""
import argparse, json, resource, sys, time, wave
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def load_wav(p):
    with wave.open(str(p)) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1, "se espera 16 kHz mono"
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


def run_moonshine(system, audio, keyterms, boost, interval, batch=True):
    from moonshine_voice import ModelArch, Transcriber
    from runtime import config as C

    d, arch = C.MOONSHINE_MODELS[system]
    opts = {}
    if keyterms and boost is not None:
        opts["keyterm_boost"] = str(boost)
    if interval:
        opts["transcription_interval"] = str(interval)
    kw = dict(model_path=str(C.moonshine_model_dir(system)), model_arch=getattr(ModelArch, arch), update_interval=0.1)
    tr = Transcriber(**kw, options=opts) if opts else Transcriber(**kw)
    if keyterms:
        tr.set_keyterms(keyterms)
    lines = {}

    def on(ev):
        ln = getattr(ev, "line", None)
        if ln is not None:
            lines[ln.line_id] = (ln.start_time, ln.start_time + ln.duration, ln.text)

    if batch:  # sin pacing: libmoonshine segmenta con su VAD y transcribe lo más rápido que puede
        t = tr.transcribe_without_streaming(audio.tolist(), 16000)
        out = [{"start": l.start_time, "end": l.start_time + l.duration, "text": l.text.strip()} for l in t.lines if l.text.strip()]
        tr.close()
        return out
    st = tr.create_stream(update_interval=0.1)
    st.add_listener(on)
    st.start()
    step = 1600
    for i in range(0, len(audio), step):
        st.add_audio(audio[i:i + step].tolist(), 16000)
    st.stop()
    st.close()
    tr.close()
    return [{"start": a, "end": b, "text": t.strip()} for _, (a, b, t) in sorted(lines.items()) if t.strip()]


def run_parakeet(audio_path):
    from parakeet_mlx import from_pretrained
    from runtime import config as C

    path = C.snapshot(C.PARAKEET_REPO, C.PARAKEET_REV, ("config.json", "model.safetensors"))
    m = from_pretrained(str(path))
    r = m.transcribe(str(audio_path), chunk_duration=60.0, overlap_duration=15.0)
    return [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in r.sentences if s.text.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keyterms", help="archivo de texto, un término por línea")
    ap.add_argument("--boost", type=float)
    ap.add_argument("--interval", type=float)
    ap.add_argument("--stream", action="store_true", help="stream continuo a ritmo de reloj (lento: 1x)")
    a = ap.parse_args()
    audio = load_wav(a.audio)
    kt = [l.strip() for l in Path(a.keyterms).read_text().splitlines() if l.strip()] if a.keyterms else None
    t0, c0 = time.perf_counter(), time.process_time()
    if a.system == "parakeet":
        lines = run_parakeet(a.audio)
    else:
        lines = run_moonshine(a.system, audio, kt, a.boost, a.interval, batch=not a.stream)
    wall, cpu = time.perf_counter() - t0, time.process_time() - c0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # bytes en macOS -> MB
    Path(a.out).write_text(json.dumps({
        "system": a.system, "keyterms": kt, "boost": a.boost, "lines": lines, "wall_s": round(wall, 1),
        "cpu_s": round(cpu, 1), "maxrss_mb": round(rss, 0), "audio_s": round(len(audio) / 16000, 1)},
        ensure_ascii=False, indent=1))
    print(f"{a.system}: {len(lines)} lineas wall={wall:.1f}s cpu={cpu:.1f}s rss={rss:.0f}MB audio={len(audio)/16000:.0f}s")


if __name__ == "__main__":
    main()
