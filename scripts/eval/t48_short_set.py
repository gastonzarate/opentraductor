"""T-48: set sintético de frases cortas (1-4 palabras, léxico rioplatense) para medir si el ASR las pasa a inglés.

Genera WAV 16 kHz mono con las voces en español de macOS (`say`), con 0,5 s de silencio antes y 1,2 s después
(para que el endpoint cierre), y un manifest compatible con `python -m benchmark run`.

Límite conocido: macOS no trae voz es-AR; son voces es-MX/es-ES leyendo frases rioplatenses ("dialect": "other").
La voz sintética es más limpia que una persona en un micrófono de notebook, así que el % en inglés que salga es un
piso, no el valor real de una llamada.

Uso: .venv/bin/python scripts/eval/t48_short_set.py [OUTDIR]   (por defecto artifacts/t48/short)
"""
from __future__ import annotations

import array
import hashlib
import json
import math
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

PHRASES = [
    "Sí.", "Sí, dale.", "Dale.", "No.", "No, pará.", "Ya está.", "Perfecto.", "Bárbaro.", "Joya.", "Listo.",
    "¿Me escuchás?", "¿Me ves?", "Hola, hola.", "Buen día.", "Gracias.", "De una.", "Claro.", "Obvio.",
    "Exacto.", "Tal cual.", "Esperá un segundo.", "Ahora vuelvo.", "Se cortó.", "No te escucho.",
    "¿Se ve la pantalla?", "Compartí pantalla.", "Vamos de nuevo.", "Tenés razón.", "Ni idea.", "Mirá vos.",
    "Un toque.", "Che, ¿estás?", "Bueno.", "¿Cómo andás?", "Todo bien.", "Sí, sí, claro.",
]
# (nombre de voz en `say`, id corto). Solo las que vienen con macOS en español.
VOICES = [("Paulina", "paulina-mx"), ("Eddy (Spanish (Mexico))", "eddy-mx"),
          ("Flo (Spanish (Mexico))", "flo-mx"), ("Eddy (Spanish (Spain))", "eddy-es")]
SR = 16000
PRE_S, POST_S = 0.5, 1.2


def _bounds(samples: array.array, sr: int, floor_dbfs: float = -45.0) -> tuple[float, float]:
    """Primer y último bloque de 10 ms sobre el piso (estimado, no humano)."""
    n = sr // 100
    on = []
    for i in range(0, len(samples) - n + 1, n):
        blk = samples[i:i + n]
        rms = math.sqrt(sum(x * x for x in blk) / n) / 32768.0
        on.append(rms > 0 and 20 * math.log10(rms) > floor_dbfs)
    idx = [i for i, v in enumerate(on) if v]
    if not idx:
        return 0.0, len(samples) / sr
    return idx[0] * n / sr, (idx[-1] + 1) * n / sr


def main(out: Path) -> None:
    audio = out / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    rows = []
    for voice, vid in VOICES:
        for k, text in enumerate(PHRASES):
            name = f"{vid}-{k:02d}.wav"
            with tempfile.TemporaryDirectory() as td:
                raw = Path(td) / "raw.wav"
                subprocess.run(["say", "-v", voice, "--file-format=WAVE", "--data-format=LEI16@16000",
                                "-o", str(raw), text], check=True)
                with wave.open(str(raw)) as w:
                    assert w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2
                    speech = array.array("h", w.readframes(w.getnframes()))
            s0, s1 = _bounds(speech, SR)
            pcm = array.array("h", [0] * int(PRE_S * SR)) + speech + array.array("h", [0] * int(POST_S * SR))
            dst = audio / name
            with wave.open(str(dst), "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
            rows.append({
                "item_id": f"t48short-{vid}-{k:02d}", "recording_id": f"t48short-{vid}-{k:02d}",
                "source_id": "t48short", "speaker_id": vid, "split": "development",
                "audio_path": str(dst.resolve()), "sha256": hashlib.sha256(dst.read_bytes()).hexdigest(),
                "sample_rate": SR, "channels": 1, "duration_s": round(len(pcm) / SR, 3), "dialect": "other",
                "license": "generado localmente (macOS say)", "license_url": "", "source_url": "",
                "attribution": f"macOS say -v '{voice}'", "transcript_es": text, "transcript_es_provenance": "generated",
                "reference_en": None, "reference_en_provenance": "unknown",
                "utterance_start_s": round(PRE_S + s0, 3), "utterance_end_s": round(PRE_S + s1, 3),
                "boundary_method": "estimated", "boundary_uncertainty_s": 0.05,
                "consent_note": "voz sintética del sistema", "words": len(text.split()),
            })
    (out / "manifest.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"{len(rows)} clips en {out}/manifest.jsonl")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/t48/short"))
