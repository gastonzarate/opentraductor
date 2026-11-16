"""Rutas y revisiones fijas de los modelos de la baseline.

Todo se resuelve desde disco (``local_files_only``): el runtime nunca
descarga nada al arrancar. Las descargas son un paso de setup explícito
(``python -m runtime.setup``).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PARAKEET_REPO = "mlx-community/parakeet-tdt-0.6b-v3"
PARAKEET_REV = "ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15"
KOKORO_REPO = "mlx-community/Kokoro-82M-bf16"
KOKORO_REV = "a71e4d38b236d968966a2002c4c895dbd12b1c3c"
MARIAN_REPO = "Helsinki-NLP/opus-mt-es-en"
MARIAN_REV = "c96e2c5399ebfae4fc43d9669556b9afa74bb69d"

# Voces de Kokoro que el setup descarga (solo en-US).
KOKORO_VOICES = ("af_heart", "af_bella", "af_nicole", "am_michael", "am_adam")


def models_dir() -> Path:
    return Path(
        os.environ.get("OPENTRADUCTOR_MODELS", "~/.cache/opentraductor/models")
    ).expanduser()


MOONSHINE_REV = "quantized_26_08_24"
# backend -> (directorio del modelo, ModelArch de moonshine_voice)
MOONSHINE_MODELS = {
    "moonshine-tiny": ("tiny-streaming-es", "TINY_STREAMING"),
    "moonshine-small": ("small-streaming-es", "SMALL_STREAMING"),
}


def moonshine_model_dir(backend: str) -> Path:
    """Pesos Moonshine en disco: ``$OPENTRADUCTOR_MOONSHINE_DIR`` o ``<repo>/models/moonshine/assets``."""
    name = MOONSHINE_MODELS[backend][0]
    root = os.environ.get("OPENTRADUCTOR_MOONSHINE_DIR")
    base = Path(root).expanduser() if root else Path(__file__).resolve().parent.parent / "models" / "moonshine" / "assets" / "download.moonshine.ai" / "model"
    return base / name / MOONSHINE_REV


def marian_ct2_dir() -> Path:
    """Marian ya convertido a CTranslate2 int8.

    Edición de tienda: el build lo convierte (``scripts/convert_marian.py``) y lo
    copia a Resources; el shell lanza el runtime con ``OPENTRADUCTOR_MARIAN_CT2``
    apuntando a esa carpeta (solo lectura). Sin la variable, va a ``models_dir()``.
    """
    override = os.environ.get("OPENTRADUCTOR_MARIAN_CT2")
    if override:
        return Path(override).expanduser()
    return models_dir() / "opus-mt-es-en-ct2-int8"


def silero_vad_onnx_path() -> Path:
    """Modelo Silero VAD en ONNX (MIT). Se lee como archivo de datos, sin importar
    el paquete ``silero_vad`` (que arrastra torch/torchaudio).

    Orden: ``$OPENTRADUCTOR_SILERO_ONNX``; ``runtime/data/silero_vad.onnx``
    (incluido en el bundle); en desarrollo, el archivo del paquete instalado
    (se ubica con ``find_spec``, sin importarlo).
    """
    override = os.environ.get("OPENTRADUCTOR_SILERO_ONNX")
    if override:
        return Path(override).expanduser()
    vendored = Path(__file__).resolve().parent / "data" / "silero_vad.onnx"
    if vendored.exists():
        return vendored
    import importlib.util

    spec = importlib.util.find_spec("silero_vad")
    for loc in (spec.submodule_search_locations or []) if spec else []:
        cand = Path(loc) / "data" / "silero_vad.onnx"
        if cand.exists():
            return cand
    return vendored  # no existe: el que llama falla con un mensaje claro


def snapshot(repo: str, rev: str, required: tuple[str, ...] = ("config.json",)) -> Path:
    """Directorio del snapshot fijado por revisión, solo desde disco.

    No usa ``snapshot_download``: el setup descarga archivos selectivos y esa
    función rechaza snapshots parciales. Falla claro si falta algo necesario.
    """
    from huggingface_hub import constants

    d = Path(constants.HF_HUB_CACHE) / ("models--" + repo.replace("/", "--")) / "snapshots" / rev
    missing = [f for f in required if not (d / f).exists()]
    if not d.is_dir() or missing:
        raise FileNotFoundError(
            f"Falta el modelo {repo}@{rev[:8]} en disco ({missing or 'snapshot'}). "
            "Ejecuta: uv run --extra runtime python -m runtime.setup"
        )
    return d


@dataclass
class PipelineConfig:
    """Parámetros ajustables. Todos quedan registrados en la corrida."""

    # Endpoint
    endpoint_silence_ms: float = 250.0
    vad_threshold: float = 0.5
    min_speech_ms: float = 120.0
    max_segment_s: float = 20.0  # corte forzado de seguridad
    # ASR: "parakeet" (MLX, GPU) | "moonshine-tiny" | "moonshine-small" (CPU, hilo propio)
    asr_backend: str = "parakeet"
    moonshine_update_interval_s: float = 0.1
    asr_chunk_ms: float = 160.0
    asr_context: tuple[int, int] = (256, 256)  # (izq, der) en frames de 80 ms; derecho pequeño degrada el texto
    # Al endpoint, no pasar al ASR la cola de audio que es solo silencio (toda la voz ya se procesó).
    asr_skip_silent_tail: bool = False
    # TTS
    voice: str = "af_heart"
    speed: float = 1.0
    # Recorte del silencio inicial del TTS; None = desactivado. Valor = margen (ms) que se conserva antes del primer sonido.
    tts_trim_lead_margin_ms: float | None = None
    # T-47: recorte del silencio FINAL de cada chunk TTS (-60 dBFS, bloques de 5 ms, conserva 80 ms; prefijo PCM exacto).
    # False = camino de hoy, sin cambios. Reversible por flag (--tts-trim-tail off) o env OPENTRADUCTOR_TTS_TRIM_TAIL.
    tts_trim_tail: bool = False
    # Primer fragmento del TTS: máx. de palabras (None = solo por puntuación). Ayuda con ASR sin puntuación.
    tts_first_chunk_max_words: int | None = None
    # Con el anterior: no cortar justo antes/después de and/or/but/nor/yet (conjunción dentro de su grupo).
    tts_bind_conjunctions: bool = False
    # Normaliza el texto ASR para MT (mayúscula inicial + punto final; sin comas). El crudo se conserva aparte.
    asr_normalize_text: bool = False
    # Especulación (OPCIONAL, apagada): tras una pausa VAD y con el ASR ya al día se precalcula MT (y, si el ASR
    # no usa el hilo MLX, el primer fragmento TTS) del parcial. Nada especulativo se emite jamás: el resultado
    # solo se usa si el texto final es EXACTAMENTE igual; si no, se descarta y corre la ruta normal.
    speculative_mt_tts: bool = False
    spec_silence_ms: float = 64.0  # silencio VAD mínimo para especular (< endpoint_silence_ms)
    spec_max_entries: int = 4  # entradas en caché simultáneas (acotado)
    end_stream_drain_timeout_s: float = 5.0  # espera máx. a drenar salidas al cerrar un stream; luego se abandona (y se descarta lo viejo)
    # Colas / backpressure
    max_pending_segments: int = 8
    max_asr_backlog_s: float = 6.0
    warmup_in_open: bool = False

    def __post_init__(self) -> None:
        if self.asr_backend != "parakeet" and self.asr_backend not in MOONSHINE_MODELS:
            raise ValueError(f"asr_backend desconocido: {self.asr_backend!r}")

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineConfig":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        if "asr_context" in known:
            known["asr_context"] = tuple(known["asr_context"])
        return cls(**known)
