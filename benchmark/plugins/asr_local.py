"""ASR local e independiente para texto audible (faster-whisper, CPU).

Independencia: el ASR debe ser un sistema distinto del candidato. El modelo y
sus parámetros quedan registrados en ``label``/``describe()`` y en run.json.
Corre en CPU (int8) para no contaminar mediciones de GPU del candidato.
Licencias: faster-whisper MIT; pesos Whisper (OpenAI) MIT; conversiones
Systran/faster-whisper-* MIT (verificar la ficha del modelo elegido).

Sesgo: Whisper puede "corregir" habla poco clara con un modelo de lenguaje, lo
que subestima errores de TTS. Es evidencia de inteligibilidad, no verdad.
"""
from __future__ import annotations

import numpy as np

from .. import audio as A
from . import PluginUnavailable


class LocalWhisperAsr:
    def __init__(self, model_size: str = "small.en", *, compute_type: str = "int8",
                 device: str = "cpu", language: str = "en", beam_size: int = 5, cpu_threads: int = 4):
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except ImportError as exc:
            raise PluginUnavailable("faster-whisper no instalado") from exc
        self.params = dict(model_size=model_size, compute_type=compute_type, device=device,
                           language=language, beam_size=beam_size, cpu_threads=cpu_threads,
                           temperature=0.0, vad_filter=False)
        self.label = f"faster-whisper:{model_size}:{compute_type}:{device}"
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type, cpu_threads=cpu_threads)

    def __call__(self, pcm: bytes, sample_rate: int) -> str:
        pcm16k = A.resample_fft(pcm, sample_rate, A.TARGET_SR)
        audio = np.frombuffer(pcm16k, dtype="<i2").astype(np.float32) / 32768.0
        segs, _info = self._model.transcribe(
            audio, language=self.params["language"], beam_size=self.params["beam_size"],
            temperature=0.0, vad_filter=False, condition_on_previous_text=False,
        )
        return " ".join(s.text.strip() for s in segs).strip()

    def describe(self) -> dict:
        import importlib.metadata as md

        return {**self.params, "label": self.label, "faster_whisper": md.version("faster-whisper"),
                "caveat": "ASR con modelo de lenguaje: puede corregir habla poco clara (sesgo hacia lo inteligible)"}
