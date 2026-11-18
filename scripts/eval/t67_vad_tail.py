#!/usr/bin/env python3
"""T-67 punto 2: de dónde sale la «espera fin de habla» (~250 ms p50 con endpoint_silence_ms=160).

Corre el Silero VAD del runtime (ONNX) sobre los clips de un manifest, ventana a ventana como Pipeline.push, y
separa la espera en: cola del VAD (última ventana con p >= umbral de voz - fin de habla de referencia) + silencio
contado hasta el endpoint. Prueba umbrales para CONTAR silencio dentro de la frase (p < off_thr) y espera base:
cuánto se adelanta el endpoint y cuántos cortes falsos aparecen (endpoint antes del fin de referencia - 150 ms,
es decir, la frase se parte). Solo análisis offline, sin cambio de runtime.
Uso: python scripts/eval/t67_vad_tail.py artifacts/t67-fin/manifest_dev16.jsonl [data/audio]"""
import json
import statistics as st
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from runtime import components as K  # noqa: E402

man = Path(sys.argv[1])
audio_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "data" / "audio"
ON = 0.5
WIN_MS = K.VAD_WINDOW / 16
items = [json.loads(line) for line in man.read_text().splitlines() if line.strip()]
vad = K.SileroVAD()
clips = []
for it in items:
    with wave.open(str(audio_dir / it["audio_path"])) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1
        pcm = w.readframes(w.getnframes())
    vad.reset()
    probs = [(off / 16000, p) for off, p in vad.probs(pcm)]
    clips.append((it, probs))


def simulate(probs, off_thr, wait_ms):
    """Endpoints (pos_s, last_speech_end_s) con la lógica de Pipeline.push: voz si p >= ON; dentro de frase, silencio
    si p < off_thr (off_thr = ON es lo de hoy)."""
    eps, in_speech, sil, last = [], False, 0.0, 0.0
    for end_s, p in probs:
        if p >= (off_thr if in_speech else ON):
            in_speech, sil, last = True, 0.0, end_s
        elif in_speech:
            sil += WIN_MS
            if sil >= wait_ms:
                eps.append((end_s, last))
                in_speech, sil = False, 0.0
    if in_speech:
        eps.append((None, last))
    return eps


print(f"# T-67 cola del VAD · {man.name} · {len(clips)} clips\n")
print("| off_thr | espera | n | cola VAD p50 | endpoint - fin ref p50 | p90 | cortes falsos | sin endpoint |")
print("|---|---|---|---|---|---|---|---|")
for off_thr in (0.5, 0.65, 0.8, 0.9):
    for wait in (160, 128, 96, 64):
        tails, lat, false_cuts, noep = [], [], 0, 0
        for it, probs in clips:
            ref = it["utterance_end_s"]
            eps = simulate(probs, off_thr, wait)
            false_cuts += sum(1 for pos, _ in eps if pos is not None and pos < ref - 0.15)
            final = [e for e in eps if e[0] is not None and e[0] >= ref - 0.15]
            if not final:
                noep += 1
                continue
            pos, last = final[0]
            tails.append((last - ref) * 1000)
            lat.append((pos - ref) * 1000)
        q = sorted(lat)
        p90 = q[int(0.9 * (len(q) - 1))] if q else float("nan")
        print(f"| {off_thr} | {wait} | {len(lat)} | {st.median(tails):.0f} | {st.median(lat):.0f} | {p90:.0f} | "
              f"{false_cuts} | {noep} |")

print("\n## Probabilidad del VAD alrededor del fin de referencia (off_thr 0,5, 160 ms)\n")
for it, probs in clips:
    ref = it["utterance_end_s"]
    near = [f"{(e - ref) * 1000:+.0f}:{p:.2f}" for e, p in probs if -0.1 <= e - ref <= 0.3]
    print(f"- {it['item_id'][-24:]} fin {ref:.2f}s  " + " ".join(near))
