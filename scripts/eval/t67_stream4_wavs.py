"""T-67: WAV de muestra para comparar la prosodia del 1.er fragmento TTS cortado a 4 palabras (default desde 8f01737)
contra el corte previo a 8 palabras, con el mismo texto y la misma voz de Kokoro.

Por frase escribe <n>-w4.wav y <n>-w8.wav: los chunks de synth() pegados sin hueco (como suenan en la app con
MT_STREAM_TTS, tts_gaps 0/80 en t67-asrfin) y un .txt con los cortes. Solo Kokoro (sin ASR ni MT), sin red.

  HF_HUB_OFFLINE=1 PYTHONPATH=. .venv-inc-llm/bin/python scripts/eval/t67_stream4_wavs.py --out <dir>
"""
from __future__ import annotations

import argparse
import wave
from pathlib import Path

from runtime.components import KokoroTTS
from runtime.config import PipelineConfig

# mt_final reales de t67-asrfin/n2 (stream4adapt64) y del eval de términos: frases de 8+ palabras, con coma o sin.
SENTENCES = [
    "Edgardo would like to make a deposit into his cousin's account.",
    "If it's not credited within 48 hours, we'll do a chargeback.",
    "By Friday the 20th at 6 PM at the latest, we need the final budget approved.",
]


def write(path: Path, pcm: bytes, sr: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--pause-ms", type=float, default=None, help="tts_clause_pause_ms (default: sin tope)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    c = PipelineConfig()
    tts = KokoroTTS(c.voice, c.speed, trim_lead_margin_ms=c.tts_trim_lead_margin_ms,
                    first_chunk_max_words=c.tts_first_chunk_max_words, bind_conjunctions=c.tts_bind_conjunctions,
                    trim_tail=c.tts_trim_tail, trim_lead_all_clauses=c.tts_trim_lead_all_clauses)
    tts.clause_pause_ms = a.pause_ms
    list(tts.synth("Warm up."))
    notes = []
    for i, text in enumerate(SENTENCES, 1):
        for words in (4, 8):
            tts.first_chunk_max_words = words
            chunks = list(tts.synth(text))
            name = f"{i}-w{words}.wav"
            write(a.out / name, b"".join(ch.pcm for ch in chunks), tts.SAMPLE_RATE)
            notes.append(f"{name}: " + " | ".join(dict.fromkeys(ch.text for ch in chunks)))
    (a.out / "cortes.txt").write_text("\n".join(notes) + "\n")
    print("\n".join(notes))


if __name__ == "__main__":
    main()
