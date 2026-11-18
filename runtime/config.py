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


# T-50/T-51: tamaño de Qwen según la RAM de la Mac (decisión 16/11: 4B si >=16 GB, 1.7B si menos; 0.6B descartado).
# tamaño -> (repo mlx-community, carpeta bajo la raíz de Qwen)
QWEN_MODELS = {
    "4b": ("mlx-community/Qwen3-4B-Instruct-2507-4bit", "qwen3-4b-instruct-2507-4bit"),
    "1.7b": ("mlx-community/Qwen3-1.7B-4bit", "qwen3-1.7b-4bit"),
}
QWEN_4B_MIN_RAM = 16 * 1024**3
# Tope de la caché de buffers libres de MLX: sin tope crecía ~4 GB en 10 min (T-50). Ojo: el tope SÍ cuesta
# latencia de TTS (Kokoro MLX vuelve a reservar buffers): 1er audio p50 +27..+69 ms en 4B/1.7B (soak 17/11,
# memory dato/t51-mlx-cache-limit-soak.md). 768/1536 MB no lo arreglan y pasan los 3,5 GB del 1.7B.
MLX_CACHE_LIMIT_MB_DEFAULT = 256


def mac_memsize() -> int | None:
    """RAM física (``hw.memsize``) sin subprocess (sirve en el sandbox). None si no se puede leer."""
    try:
        import ctypes
        import ctypes.util

        libc = ctypes.CDLL(ctypes.util.find_library("c"))
        v = ctypes.c_uint64(0)
        n = ctypes.c_size_t(ctypes.sizeof(v))
        if libc.sysctlbyname(b"hw.memsize", ctypes.byref(v), ctypes.byref(n), None, ctypes.c_size_t(0)) == 0:
            return int(v.value)
    except Exception:
        pass
    return None


def qwen_size_for_ram(memsize: int | None) -> str:
    """'4b' con >=16 GB; '1.7b' con menos. Si no se sabe la RAM, '1.7b' (el que entra en 8 GB)."""
    return "4b" if memsize is not None and memsize >= QWEN_4B_MIN_RAM else "1.7b"


def qwen_size(memsize: int | None = None) -> str:
    """Tamaño elegido: ``$OPENTRADUCTOR_QWEN_SIZE`` (4b|1.7b) o según ``hw.memsize``."""
    s = os.environ.get("OPENTRADUCTOR_QWEN_SIZE")
    if s:
        if s not in QWEN_MODELS:
            raise ValueError(f"OPENTRADUCTOR_QWEN_SIZE inválido: {s!r} ({'|'.join(QWEN_MODELS)})")
        return s
    return qwen_size_for_ram(mac_memsize() if memsize is None else memsize)


def qwen_root() -> Path:
    """Carpeta con los Qwen por tamaño: ``$OPENTRADUCTOR_QWEN_ROOT`` o ``models_dir()/qwen``."""
    r = os.environ.get("OPENTRADUCTOR_QWEN_ROOT")
    return Path(r).expanduser() if r else models_dir() / "qwen"


def qwen_model_dir(cfg: "PipelineConfig") -> Path:
    """T-51: pesos de Qwen en disco (sin descarga aquí).

    Orden: ``cfg.qwen_model_dir``, ``$OPENTRADUCTOR_QWEN_DIR`` (carpeta exacta) y, si no hay ninguno,
    ``qwen_root()/<carpeta del tamaño elegido por RAM>`` (ver ``qwen_size``)."""
    d = cfg.qwen_model_dir or os.environ.get("OPENTRADUCTOR_QWEN_DIR")
    if d:
        return Path(d).expanduser()
    return qwen_root() / QWEN_MODELS[qwen_size()][1]


def mlx_cache_limit_bytes(cfg: "PipelineConfig") -> int | None:
    """Tope de caché MLX en bytes: ``$OPENTRADUCTOR_MLX_CACHE_MB`` o ``cfg.mlx_cache_limit_mb``; <0 o None = sin tope."""
    v = os.environ.get("OPENTRADUCTOR_MLX_CACHE_MB")
    mb = float(v) if v else cfg.mlx_cache_limit_mb
    if mb is None or mb < 0:
        return None
    return int(mb * 1e6)  # igual que t50_pipeline_ram.py --cache-limit-mb (la config medida)


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


# T-58: default anterior de endpoint_silence_ms. La app guardaba todo el config al cambiar cualquier opción,
# así que un 250 guardado casi nunca es una elección del usuario: se trata como "sin valor".
LEGACY_ENDPOINT_SILENCE_MS = 250.0


@dataclass
class PipelineConfig:
    """Parámetros ajustables. Todos quedan registrados en la corrida."""

    # Endpoint
    # T-58: 160 (antes 250). Brazo store-tuned del A/B T-58: p50 1063 vs 1234 ms, mismo WER 0,047.
    endpoint_silence_ms: float = 160.0
    # T-67 fin de frase adaptativo (apagado hasta el A/B de @bench; env OPENTRADUCTOR_ENDPOINT_ADAPTIVE=1/0 lo fuerza).
    # En cada ventana de silencio se elige la espera según el último parcial del ASR: endpoint_long_ms si termina en
    # conector/preposición/artículo/muletilla/coma ("y", "que", "pero", "o sea", "de", "este"...);
    # endpoint_short_ms si el ASR ya procesó toda la voz y el parcial puede cerrar frase; si no, endpoint_silence_ms.
    endpoint_adaptive: bool = False
    # T-67 paso 2: traducción simultánea por cláusulas (runtime/simul.py). Una cláusula confirmada por dos parciales
    # seguidos del ASR y cerrada por coma/"pero"/"porque"/"o sea"... (mín. 5 palabras) se traduce y se habla antes del
    # fin de frase; en el endpoint solo se traduce el resto. Env OPENTRADUCTOR_MT_SIMUL=1/0 manda. Off = camino de hoy.
    mt_simul_clauses: bool = False
    endpoint_short_ms: float = 96.0
    endpoint_long_ms: float = 480.0
    vad_threshold: float = 0.5
    min_speech_ms: float = 120.0
    max_segment_s: float = 20.0  # corte forzado de seguridad
    # ASR: "parakeet" (MLX, GPU) | "moonshine-tiny" | "moonshine-small" (CPU, hilo propio)
    asr_backend: str = "parakeet"
    moonshine_update_interval_s: float = 0.1
    asr_chunk_ms: float = 160.0
    asr_context: tuple[int, int] = (256, 256)  # (izq, der) en frames de 80 ms; derecho pequeño degrada el texto
    # Al endpoint, no pasar al ASR la cola de audio que es solo silencio (toda la voz ya se procesó).
    asr_skip_silent_tail: bool = True  # T-58: default activo (WER sin cambio en el A/B)
    # TTS
    voice: str = "af_heart"
    speed: float = 1.0
    # Recorte del silencio inicial del TTS; None = desactivado. Valor = margen (ms) que se conserva antes del primer sonido.
    tts_trim_lead_margin_ms: float | None = 30.0  # T-58: default 30 ms (antes None)
    # T-47: recorte del silencio FINAL de cada chunk TTS (-60 dBFS, bloques de 5 ms, conserva 80 ms; prefijo PCM exacto).
    # False = camino de hoy, sin cambios. Reversible por flag (--tts-trim-tail off) o env OPENTRADUCTOR_TTS_TRIM_TAIL.
    tts_trim_tail: bool = False
    # T-58: aplica el recorte de silencio inicial (tts_trim_lead_margin_ms) al 1.er chunk de CADA cláusula, no solo
    # de la frase. Kokoro abre cada cláusula con ~300-380 ms de silencio: sin esto, cortar el 1.er fragmento más corto
    # mete pausas >=300 ms a mitad de frase. False = camino de hoy.
    tts_trim_lead_all_clauses: bool = False
    # BUG-7: antes de Marian, "120 mil" -> "120.000" y una llamada por oración (Marian omitía la 2.ª).
    # False = camino de hoy. Reversible por flag (--mt-normalize off) o env OPENTRADUCTOR_MT_NORMALIZE.
    mt_normalize: bool = False
    # Pedido de la persona (17/11), solo Qwen, apagados hasta el A/B (set de frases ambiguas, BLEU -1 máx.,
    # MT p50 +20 ms máx.): prompt de traducción por sentido y N frases previas ES->EN de la sesión como contexto.
    # T-74 (18/11): encendido por defecto (el 94,8 % del 17/11 se midió con SENSE+GLOSSARY+EXT+DOM). Env =0 lo apaga.
    mt_sense_prompt: bool = True
    # T-58 opción acotada: solo a las frases con un término de QWEN_GLOSSARY, glosario + «mantener negaciones, sujeto,
    # tiempo, cifras y nombres» después del texto. Gate 2 de 70bb991 pasa en sentido (20->28/30, sin must_not nuevos;
    # BLEU 72.8->72.8), pero NO en latencia: en frases con modismo MT p95 +63 ms (límite +30), porque el glosario queda
    # fuera del prefijo del prefill KV. Ahora el glosario va antes del texto, armado con el parcial estable (dentro de
    # la caché); falta repetir gate 2 y latencia. Apagado por defecto; se enciende con env OPENTRADUCTOR_MT_GLOSSARY=1.
    # T-74 (18/11): encendido por defecto junto con mt_sense_prompt; env OPENTRADUCTOR_MT_GLOSSARY=0 lo apaga.
    mt_sense_glossary: bool = True
    mt_context_turns: int = 0
    # T-58 contexto largo (pedido 17/11): caché KV de Qwen que sobrevive entre frases; cada frase solo hace prefill de
    # sus tokens nuevos. Con mt_context_turns > 0 la historia se recorta en bloque (n -> n/2) para no romper el prefijo.
    mt_session_kv: bool = False
    # T-68 contexto largo por tokens: tope (aprox.) de tokens de historia en el prompt de Qwen. Al pasarlo se descartan
    # las frases más viejas hasta quedar en la mitad (recorte en bloque: la caché de sesión se rehace solo entonces).
    # 0 = solo mt_context_turns. Con este tope, mt_context_turns es el máximo de frases (poner alto, p. ej. 200).
    mt_context_tokens: int = 0
    # T-68 decodificación especulativa: carpeta de un Qwen3 chico (0.6B 4bit) como borrador del 4B. None = sin borrador.
    # También por env OPENTRADUCTOR_MT_DRAFT_DIR (y OPENTRADUCTOR_MT_DRAFT_TOKENS, default 3).
    mt_draft_dir: str | None = None
    # T-51: traductor. "marian" (CTranslate2, CPU) = camino de hoy | "qwen" (Qwen3 4bit MLX, en el hilo GPU con Kokoro).
    mt_engine: str = "marian"
    # Carpeta del Qwen MLX (config.json + model.safetensors + tokenizer). None = $OPENTRADUCTOR_QWEN_DIR.
    qwen_model_dir: str | None = None
    # T-50: tope de la caché de buffers libres de MLX (MB), para todo el proceso (Qwen + Kokoro + Parakeet). None = sin tope.
    mlx_cache_limit_mb: float | None = MLX_CACHE_LIMIT_MB_DEFAULT
    # Primer fragmento del TTS: máx. de palabras (None = solo por puntuación). Ayuda con ASR sin puntuación.
    tts_first_chunk_max_words: int | None = 8  # T-58: default 8 (antes None)
    # Con el anterior: no cortar justo antes/después de and/or/but/nor/yet (conjunción dentro de su grupo).
    tts_bind_conjunctions: bool = True  # T-58: default activo
    # Normaliza el texto ASR para MT (mayúscula inicial + punto final; sin comas). El crudo se conserva aparte.
    asr_normalize_text: bool = False
    # Especulación (OPCIONAL, apagada): tras una pausa VAD y con el ASR ya al día se precalcula MT (y, si el ASR
    # no usa el hilo MLX, el primer fragmento TTS) del parcial. Nada especulativo se emite jamás: el resultado
    # solo se usa si el texto final es EXACTAMENTE igual; si no, se descarta y corre la ruta normal.
    speculative_mt_tts: bool = False
    # T-58: fin de frase especulativo. En la 1.ª ventana de cada pausa VAD >= asr_spec_finish_silence_ms (con parcial)
    # se cierra el stream del ASR (finish) sin esperar al endpoint. Si el endpoint llega sin voz nueva, ese texto ES el
    # final; si vuelve la voz, se reabre el stream y se le re-pasa el audio del segmento. Sonda 7b1b37b (Moonshine,
    # tiempo real): hit 9/9, finish 153-201 ms desde la pausa. $OPENTRADUCTOR_ASR_SPEC_FINISH=1/0 lo fuerza.
    asr_spec_finish_on_pause: bool = False
    # T-58: MT en streaming hacia Kokoro (solo Qwen). El 1.er fragmento se sintetiza entre pasos del decode apenas el
    # prefijo lo fija (components.first_clause_final) y sale antes del mt_final. Mismos cortes y texto que el camino
    # normal. Apagado hasta el A/B de @bench. $OPENTRADUCTOR_MT_STREAM_FIRST_CHUNK=1/0 lo fuerza.
    mt_stream_first_chunk: bool = False
    # T-73 Idea B: alias de mt_stream_first_chunk con el nombre de la tarea ($OPENTRADUCTOR_MT_STREAM_TTS=1/0).
    mt_stream_tts: bool = False
    # T-73 Idea A: especulación por PAUSA (no una vez por frase). En cada pausa VAD con el ASR al día se lanza MT + 1.er
    # fragmento TTS del parcial; si llega el finish especulativo del ASR (asr_spec_finish_on_pause) con otro texto, se
    # relanza con ese. Si vuelve la voz, se descarta. Acierto con comparación normalizada (minúsculas, sin
    # puntuación). Implica speculative_mt_tts. Evento spec_result por frase. $OPENTRADUCTOR_SPEC_ON_SILENCE=1/0.
    spec_on_silence: bool = False
    asr_spec_finish_silence_ms: float = 80.0
    spec_silence_ms: float = 64.0  # silencio VAD mínimo para especular (< endpoint_silence_ms)
    spec_max_entries: int = 4  # entradas en caché simultáneas (acotado)
    # T-58 GPU fría: tras >=1 s de GPU ociosa la 1.ª generación de Qwen tarda ~170 ms más (sonda t58_mt_gpu_probe:
    # idle 2 s p50 472-499 vs en caliente 306). Un forward de 1 token de "hola" (~240 ms) lanzado al empezar la pausa
    # VAD despierta la GPU durante endpoint + ASR finish y la MT de la frase corre en caliente (309). Solo Qwen y con
    # el ASR fuera del hilo MLX. $OPENTRADUCTOR_MT_PREWARM=1/0 lo fuerza (para medir sin tocar la UI).
    mt_prewarm_on_pause: bool = False
    mt_prewarm_silence_ms: float = 32.0  # silencio VAD para lanzarlo (< endpoint_silence_ms)
    mt_prewarm_idle_s: float = 1.0  # solo si el hilo MLX lleva al menos esto sin trabajo
    # T-58: en vez del forward de 1 token, prefill KV del prompt con el parcial sin su última palabra (mismo
    # momento, mismo silencio). Con el final solo se procesa la cola: sonda c7aec17 -52..-58 ms sobre el prewarm,
    # salida idéntica 9/9. Corre aunque la GPU esté caliente (no si está ocupada). Reemplaza al prewarm si están
    # los dos. $OPENTRADUCTOR_MT_KV_PREFILL=1/0 lo fuerza. Encendido por defecto desde el A/B de @bench
    # (d654025, n=21, artifacts/t58-kv2): MT p50 447 -> 288, end_to_first 1056 -> 929, mt_final idéntico 21/21.
    # Solo actúa con mt_engine == "qwen"; con Marian no hace nada.
    mt_kv_prefill_on_pause: bool = True
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
