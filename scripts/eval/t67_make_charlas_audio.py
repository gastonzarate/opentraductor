#!/usr/bin/env python3
"""T-67 [bench]: audio sintético (macOS `say`) de los turnos de las charlas del eval de términos, para medir cortes falsos
de frase (t67_cuts.py) con el endpoint adaptativo a 64 ms contra el control.

Uso: .venv-inc-llm/bin/python scripts/eval/t67_make_charlas_audio.py [--voice Paulina] [--out data/audio/synth-charlas]
Salida: <out>/<talk>-<idx>.wav (16 kHz mono, 0,6 s de silencio antes y 2,0 s después) y <out>/manifest.jsonl
(formato del benchmark; límites = energía del propio audio, boundary_method 'estimated').

Límite: voz sintética (es_MX, no rioplatense), sin muletillas ni dudas dentro de la frase. Subestima los cortes que
produce una persona que duda; sirve para comparar brazos, no para estimar la tasa absoluta de cortes en uso real."""
import argparse
import hashlib
import json
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LEAD_S, TAIL_S, SR = 0.6, 2.0, 16000


def synth(text: str, voice: str) -> np.ndarray:
    with tempfile.TemporaryDirectory() as td:
        a, w = Path(td) / "x.aiff", Path(td) / "x.wav"
        subprocess.run(["say", "-v", voice, "-o", str(a), text], check=True)
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(a), str(w)], check=True)
        with wave.open(str(w)) as f:
            assert f.getframerate() == SR and f.getnchannels() == 1
            return np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16).copy()


def bounds(x: np.ndarray, floor_dbfs: float = -50.0, frame: int = 160):
    n = len(x) // frame
    rms = np.sqrt((x[: n * frame].astype(np.float64).reshape(n, frame) ** 2).mean(axis=1)) / 32768.0 + 1e-9
    act = np.where(20 * np.log10(rms) > floor_dbfs)[0]
    return (act[0] * frame / SR, (act[-1] + 1) * frame / SR) if len(act) else (0.0, len(x) / SR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="Paulina")
    ap.add_argument("--set", default=str(ROOT / "artifacts/eval/contexto-largo/charlas.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "data/audio/synth-charlas"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for line in Path(a.set).read_text().splitlines():
        if not line.strip():
            continue
        talk = json.loads(line)
        for s in talk["sentences"]:
            iid = f"{talk['talk_id']}-{s['idx']:02d}"
            x = synth(s["es"], a.voice)
            s0, s1 = bounds(x)
            x = x[int(s0 * SR): int(s1 * SR)]
            pcm = np.concatenate([np.zeros(int(LEAD_S * SR), np.int16), x, np.zeros(int(TAIL_S * SR), np.int16)])
            p = out / f"{iid}.wav"
            with wave.open(str(p), "wb") as f:
                f.setnchannels(1), f.setsampwidth(2), f.setframerate(SR)
                f.writeframes(pcm.tobytes())
            rows.append({
                "item_id": f"synth-{iid}", "recording_id": f"synth-{iid}", "source_id": "synth-say",
                "speaker_id": a.voice, "split": "development", "audio_path": str(p.relative_to(ROOT / "data/audio")),
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "sample_rate": SR, "channels": 1,
                "duration_s": round(len(pcm) / SR, 3), "dialect": "other", "license": "generado (macOS say)",
                "license_url": "n/a", "source_url": "artifacts/eval/contexto-largo/charlas.jsonl",
                "attribution": "texto generado por el equipo; voz sintética macOS", "transcript_es": s["es"],
                "transcript_es_provenance": "generated", "reference_en": s.get("en_ref"),
                "reference_en_provenance": "generated", "utterance_start_s": LEAD_S,
                "utterance_end_s": round(LEAD_S + len(x) / SR, 3), "boundary_method": "estimated",
                "boundary_uncertainty_s": 0.05,
            })
    (out / "manifest.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"{len(rows)} turnos -> {out}/manifest.jsonl; audio total {sum(r['duration_s'] for r in rows):.0f} s")


if __name__ == "__main__":
    main()
