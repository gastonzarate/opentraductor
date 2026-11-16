"""T-58: sonda de "fin de frase especulativo" (Moonshine finish adelantado durante la espera de endpoint).

Simula el segmentador del pipeline (VAD Silero 32 ms, chunks ASR de ``asr_chunk_ms``, ``asr_skip_silent_tail``)
sobre cada clip, empujando audio en ventanas de 32 ms SIN tiempo real (mide cómputo, no espera). En la primera
ventana de cada pausa con silencio >= ``--spec-ms`` y parcial no vacío, abre un SEGUNDO stream en el mismo
Transcriber con exactamente el audio del segmento recibido hasta ahí (preroll + voz + silencio), lo cierra
(stop) y guarda texto, ms y huella (seg, last_speech_end). Si vuelve la voz antes del endpoint: descarte.
Al endpoint corre el finish real del stream principal y compara.

Mide por clip y en total: aciertos (misma huella y texto idéntico al final), descartes por voz nueva,
fallos por texto distinto, ms del finish especulativo y del finish real.
Solo disco, sin red. Uso:
  uv run python scripts/eval/t58_spec_finish_probe.py [--backend moonshine-tiny] [--spec-ms 80] [--reps 3] [--out f.jsonl]
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

from runtime import components as K  # noqa: E402
from runtime.config import MOONSHINE_MODELS, PipelineConfig, moonshine_model_dir  # noqa: E402

MANIFESTS = ("artifacts/benchmark/offline-review/manifest.jsonl",)
SR = 16000


def spec_finish(asr: K.MoonshineStreamer, pcm: bytes) -> tuple[str, float]:
    """Transcribe ``pcm`` en un stream aparte del mismo Transcriber, sin tocar el stream principal."""
    t = time.perf_counter()
    lines: dict[int, str] = {}

    def on(ev):
        line = getattr(ev, "line", None)
        if line is not None:
            lines[line.line_id] = line.text

    s = asr.tr.create_stream(update_interval=asr.update_interval)
    s.add_listener(on)
    s.start()
    try:
        s.add_audio(K.pcm_to_float(pcm).tolist(), SR)
        s.stop()
    finally:
        s.close()
    text = " ".join(v.strip() for _, v in sorted(lines.items()) if v.strip())
    return text, (time.perf_counter() - t) * 1000


def run_clip(asr, vad, pcm: bytes, cfg: PipelineConfig, spec_ms: float, name: str, mode: str = "stream",
             realtime: bool = False) -> list[dict]:
    t_start = time.perf_counter()
    win = K.VAD_WINDOW
    win_s = win / SR
    chunk_b = int(SR * cfg.asr_chunk_ms / 1000) * 2
    vad.reset()
    preroll: list[bytes] = []
    in_speech, silence_ms, last_end, seg_start = False, 0.0, 0.0, 0.0
    asr_buf = bytearray()
    seg_audio = bytearray()  # todo lo enviado o por enviar al ASR del segmento (preroll incluido)
    spec = None  # (fp, text, ms)
    spec_done_pause = None
    events, seg_id, pos = [], 0, 0
    rows = []
    partial = ""
    for i in range(0, len(pcm) - win * 2 + 1, win * 2):
        frame = pcm[i:i + win * 2]
        if in_speech:
            asr_buf.extend(frame)
            seg_audio.extend(frame)
        appended = in_speech
        for consumed, p in vad.probs(frame):
            end_s = (pos + consumed) / SR
            if p >= cfg.vad_threshold:
                if in_speech and spec is not None and silence_ms > 0:
                    ev = {"kind": "discard_voice", "seg": seg_id, "spec_ms": spec[2]}
                    if mode == "main":  # el stream quedó cerrado: se reabre y se re-alimenta el segmento
                        t = time.perf_counter()
                        asr.start()
                        partial = asr.add(bytes(seg_audio)) or partial
                        asr_buf.clear()  # ya va dentro de seg_audio
                        ev["refeed_ms"] = round((time.perf_counter() - t) * 1000, 1)
                    events.append(ev)
                    spec = None
                silence_ms = 0.0
                last_end = end_s
                if not in_speech:
                    in_speech, seg_id, seg_start = True, seg_id + 1, end_s - win_s
                    pre = b"".join(preroll)
                    preroll.clear()
                    asr.start()
                    if pre:
                        asr.add(pre)
                    seg_audio = bytearray(pre)
                    partial = ""
                    if not appended:
                        asr_buf.extend(frame)
                        seg_audio.extend(frame)
                        appended = True
            elif in_speech:
                silence_ms += win_s * 1000
                if (silence_ms >= spec_ms and spec_done_pause != (seg_id, last_end) and partial.strip()
                        and (last_end - seg_start) * 1000 >= cfg.min_speech_ms):
                    spec_done_pause = (seg_id, last_end)
                    if mode == "main":  # finish adelantado en el stream principal (lo que queda pendiente, primero)
                        t = time.perf_counter()
                        if asr_buf:
                            asr.add(bytes(asr_buf))
                            asr_buf.clear()
                        text = asr.finish()
                        ms = (time.perf_counter() - t) * 1000
                    elif mode == "stream":
                        text, ms = spec_finish(asr, bytes(seg_audio))
                    else:
                        text, ms = None, 0.0
                    if text is not None:
                        spec = ((seg_id, last_end), text, ms)
                if silence_ms >= cfg.endpoint_silence_ms:
                    tail_start = (pos + len(frame) // 2) / SR - len(asr_buf) / 2 / SR
                    if mode == "main" and spec is not None:
                        pass  # stream ya cerrado en la pausa; la cola es silencio
                    elif asr_buf and not (cfg.asr_skip_silent_tail and tail_start >= last_end):
                        partial = asr.add(bytes(asr_buf)) or partial
                    asr_buf.clear()
                    t = time.perf_counter()
                    final = asr.finish()
                    fin_ms = (time.perf_counter() - t) * 1000
                    if mode == "main" and spec is not None:
                        final = spec[1]  # el stream ya se cerró en la pausa: el final ES el especulativo
                    r = {"clip": name, "seg": seg_id, "final": final, "finish_ms": round(fin_ms, 1),
                         "speech_s": round(last_end - seg_start, 2), "spec": None}
                    if spec is not None:
                        same_fp = spec[0] == (seg_id, last_end)
                        r["spec"] = {"text": spec[1], "ms": round(spec[2], 1), "same_fp": same_fp,
                                     "same_text": spec[1] == final}
                        r["result"] = "hit" if same_fp and spec[1] == final else "miss_text"
                    else:
                        r["result"] = "none"
                    rows.append(r)
                    in_speech, silence_ms, spec = False, 0.0, None
                    vad.reset_states()
                    appended = False
        pos += len(frame) // 2
        if realtime:
            nxt = t_start + pos / SR
            time.sleep(max(0.0, nxt - time.perf_counter()))
        if in_speech and len(asr_buf) >= chunk_b and not (mode == "main" and spec is not None):
            partial = asr.add(bytes(asr_buf)) or partial
            asr_buf.clear()
        elif not in_speech and not appended:
            preroll.append(frame)
            del preroll[:-12]
    if in_speech:
        if asr_buf:
            asr.add(bytes(asr_buf))
        final = asr.finish()
        rows.append({"clip": name, "seg": seg_id, "final": final, "finish_ms": None, "spec": None,
                     "result": "forced_end"})
    for e in events:
        e["clip"] = name
    return rows + events


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="moonshine-tiny")
    ap.add_argument("--spec-ms", type=float, default=80.0)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default=None)
    ap.add_argument("--mode", choices=("none", "stream", "main"), default="stream",
                    help="none: sin especular | stream: 2.o stream desde cero | main: finish adelantado en el principal")
    ap.add_argument("--realtime", action="store_true", help="empuja el audio a velocidad real (como el pipeline)")
    a = ap.parse_args()
    cfg = PipelineConfig()
    asr = K.MoonshineStreamer(moonshine_model_dir(a.backend), MOONSHINE_MODELS[a.backend][1],
                              cfg.moonshine_update_interval_s)
    asr.start(); asr.add(bytes(SR)); asr.finish()  # warmup
    vad = K.SileroVAD()
    out = []
    for m in MANIFESTS:
        for line in (ROOT / m).read_text().splitlines():
            it = json.loads(line)
            with wave.open(str(ROOT / "data/audio" / it["audio_path"])) as w:
                pcm = w.readframes(w.getnframes())
            pcm += bytes(SR)  # 0,5 s de silencio final para que llegue el endpoint
            for rep in range(a.reps):
                for r in run_clip(asr, vad, pcm, cfg, a.spec_ms, it["audio_path"], a.mode, a.realtime):
                    r["rep"] = rep
                    out.append(r)
    asr.close()
    segs = [r for r in out if "result" in r]
    disc = [r for r in out if r.get("kind") == "discard_voice"]
    by = lambda k: sum(1 for r in segs if r["result"] == k)  # noqa: E731
    spec_n = by("hit") + by("miss_text") + len(disc)
    print(f"mode={a.mode} realtime={a.realtime} spec_ms={a.spec_ms} backend={a.backend} reps={a.reps}")
    for clip in sorted({r["clip"] for r in out}):
        cs = [r for r in segs if r["clip"] == clip]
        cd = [r for r in disc if r["clip"] == clip]
        sp = [r["spec"]["ms"] for r in cs if r["spec"]] + [r["spec_ms"] for r in cd]
        fi = [r["finish_ms"] for r in cs if r["finish_ms"] is not None]
        print(f"{clip[-36:]:36} segs {len(cs)} hit {sum(r['result'] == 'hit' for r in cs)} "
              f"miss_text {sum(r['result'] == 'miss_text' for r in cs)} none {sum(r['result'] == 'none' for r in cs)} "
              f"discard_voice {len(cd)} | spec p50 {statistics.median(sp) if sp else 0:.0f} ms "
              f"finish p50 {statistics.median(fi) if fi else 0:.0f} ms")
    print(f"TOTAL especulaciones {spec_n}: hit {by('hit')} miss_text {by('miss_text')} discard_voice {len(disc)} "
          f"(descartes {100 * (spec_n - by('hit')) / max(1, spec_n):.0f} %)")
    for r in segs:
        if r["result"] == "miss_text":
            print(f"  miss {r['clip'][-20:]} seg {r['seg']}: final={r['final']!r} spec={r['spec']['text']!r}")
    if a.out:
        Path(a.out).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n")


if __name__ == "__main__":
    main()
