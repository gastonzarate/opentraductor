"""Utilidades de audio mínimas: WAV PCM16 mono 16 kHz y detección de PCM audible."""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

TARGET_SR = 16000

# Umbral documentado: un bloque de 10 ms es "no silencioso" si su RMS supera
# -50 dBFS (relativo a int16 de escala completa). Es configurable por corrida y
# queda en el manifiesto de la corrida.
DEFAULT_SILENCE_THRESHOLD_DBFS = -50.0
DEFAULT_FRAME_MS = 10.0


class AudioFormatError(ValueError):
    pass


def read_wav_pcm16_mono_16k(path: str | Path) -> bytes:
    """Lee un WAV PCM16 mono 16 kHz. No remuestrea: el corpus lo entrega así.

    Se rechaza cualquier otro formato para no introducir remuestreo implícito
    de calidad desconocida dentro del benchmark.
    """
    with wave.open(str(path), "rb") as w:
        if (w.getnchannels(), w.getsampwidth(), w.getframerate()) != (1, 2, TARGET_SR):
            raise AudioFormatError(
                f"{path}: se requiere WAV PCM16 mono {TARGET_SR} Hz, "
                f"obtenido ch={w.getnchannels()} width={w.getsampwidth()} sr={w.getframerate()}"
            )
        return w.readframes(w.getnframes())


def write_wav_pcm16_mono_16k(path: str | Path, pcm: bytes) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(TARGET_SR)
        w.writeframes(pcm)


def pcm_duration_s(pcm: bytes, sample_rate: int = TARGET_SR) -> float:
    return len(pcm) / 2 / sample_rate


def silence(duration_s: float, sample_rate: int = TARGET_SR) -> bytes:
    return bytes(2 * round(duration_s * sample_rate))


def to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2").astype(np.float64)


def rms_dbfs(pcm: bytes) -> float:
    x = to_float(pcm)
    if x.size == 0:
        return float("-inf")
    rms = float(np.sqrt(np.mean(x * x)))
    return 20 * np.log10(rms / 32768.0) if rms > 0 else float("-inf")


def first_nonsilent_offset_s(
    pcm: bytes,
    sample_rate: int,
    *,
    threshold_dbfs: float = DEFAULT_SILENCE_THRESHOLD_DBFS,
    frame_ms: float = DEFAULT_FRAME_MS,
) -> float | None:
    """Offset (s) dentro de ``pcm`` del primer bloque no silencioso, o None."""
    x = to_float(pcm)
    n = max(1, round(sample_rate * frame_ms / 1000))
    for start in range(0, x.size, n):
        frame = x[start : start + n]
        rms = float(np.sqrt(np.mean(frame * frame)))
        if rms > 0 and 20 * np.log10(rms / 32768.0) > threshold_dbfs:
            return start / sample_rate
    return None


def tone(
    duration_s: float,
    *,
    freq_hz: float = 220.0,
    amp: float = 0.3,
    sample_rate: int = TARGET_SR,
) -> bytes:
    t = np.arange(round(duration_s * sample_rate)) / sample_rate
    return (np.sin(2 * np.pi * freq_hz * t) * amp * 32767).astype("<i2").tobytes()


def write_wav_pcm16(path: str | Path, pcm: bytes, sample_rate: int) -> None:
    """WAV PCM16 mono a la frecuencia nativa (para exportar salidas de candidatos)."""
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)


def read_wav_pcm16_mono(path: str | Path) -> tuple[bytes, int]:
    with wave.open(str(path), "rb") as w:
        if (w.getnchannels(), w.getsampwidth()) != (1, 2):
            raise AudioFormatError(f"{path}: se requiere PCM16 mono")
        return w.readframes(w.getnframes()), w.getframerate()


def resample_fft(pcm: bytes, sr_from: int, sr_to: int) -> bytes:
    """Remuestreo limitado en banda (FFT). Solo para EVALUACIÓN de salidas
    (ASR/proxy); nunca para el audio de entrada del benchmark."""
    if sr_from == sr_to:
        return pcm
    x = to_float(pcm)
    n_out = round(x.size * sr_to / sr_from)
    spec = np.fft.rfft(x)
    out_spec = np.zeros(n_out // 2 + 1, dtype=complex)
    k = min(spec.size, out_spec.size)
    out_spec[:k] = spec[:k]
    y = np.fft.irfft(out_spec, n=n_out) * (n_out / x.size)
    return np.clip(np.round(y), -32768, 32767).astype("<i2").tobytes()
