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


def smart_turn_onnx_path() -> Path:
    """T-67 sonda c): modelo smart-turn v3 (ONNX). ``$OPENTRADUCTOR_SMART_TURN_ONNX``, si no
    ``models_dir()/smartturn/smart-turn-v3.2-cpu.onnx`` y, en desarrollo, ``<repo>/models/smartturn/``."""
    override = os.environ.get("OPENTRADUCTOR_SMART_TURN_ONNX")
    if override:
        return Path(override).expanduser()
    name = "smart-turn-v3.2-cpu.onnx"
    cand = models_dir() / "smartturn" / name
    if cand.exists():
        return cand
    dev = Path(__file__).resolve().parent.parent / "models" / "smartturn" / name
    return dev if dev.exists() else cand


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
    # T-67 (19/11): encendido por defecto. A/B t67-asrfin/n2 (3801bd9, dev16 n=80): stream4adapt64fin p50 695 vs off 837.
    endpoint_adaptive: bool = True
    # T-67 paso 2: traducción simultánea por cláusulas (runtime/simul.py). Una cláusula confirmada por dos parciales
    # seguidos del ASR y cerrada por coma/"pero"/"porque"/"o sea"... (mín. 5 palabras) se traduce y se habla antes del
    # fin de frase; en el endpoint solo se traduce el resto. Env OPENTRADUCTOR_MT_SIMUL=1/0 manda.
    # T-80 (21/11): encendido por defecto. A/B intercalado artifacts/t80 (4806494, 8 rondas x 8 clips, n=64/brazo):
    # p50 562 vs 599, pareado -224 ms, p95 673 vs 763; 1.er audio en clips >5 s 4,3 vs 6,3 s; cortes 100 vs 99 (114
    # turnos); términos ctx200 90/96 (93,8 %) vs 91/96. Apagar: OPENTRADUCTOR_MT_SIMUL=0.
    mt_simul_clauses: bool = True
    # Mínimo de palabras por cláusula y palabras extra que abren cláusula (separadas por coma, p. ej. "y,que").
    # Env OPENTRADUCTOR_MT_SIMUL_MIN_WORDS / OPENTRADUCTOR_MT_SIMUL_CUES mandan. Defaults = reglas de 02512a5.
    mt_simul_min_words: int = 5
    mt_simul_extra_cues: str = ""
    # T-88: primer trozo antes en frases largas. mt_simul_first_k > 0 = si no hubo coma ni nexo, el primer trozo
    # sale con k palabras confirmadas (sin terminar en artículo/preposición/nexo). mt_simul_comma_nowait = la coma
    # en la última palabra confirmada corta sin esperar la palabra siguiente. Env OPENTRADUCTOR_MT_SIMUL_FIRST_K /
    # OPENTRADUCTOR_MT_SIMUL_COMMA_NOWAIT=1/0 mandan.
    mt_simul_first_k: int = 0
    mt_simul_comma_nowait: bool = False
    # T-67 (19/11): 64 (antes 96). T-81 (21/11): 32. A/B intercalado artifacts/t81 (5 rondas x 8 clips, n=40/brazo):
    # p50 335 vs 372 ms, pareado -37 ms, p95 625 vs 604; cortes falsos en charlas 99 vs 100 (114 turnos); no cambia la
    # MT (términos ctx200 iguales). Volver: OPENTRADUCTOR_ENDPOINT_SHORT_MS=64.
    endpoint_short_ms: float = 32.0
    endpoint_long_ms: float = 480.0
    # T-67 sonda c): en los casos "base" (parcial que puede cerrar pero ASR atrasado, o vacío) se consulta una vez
    # por pausa al detector de fin de turno por audio (SmartTurn, smart-turn v3 ONNX) a los endpoint_turn_silence_ms
    # de silencio; si da p >= endpoint_turn_threshold, la espera pasa a la corta (motivo "turn"). Apagado: env
    # OPENTRADUCTOR_ENDPOINT_TURN=1/0 manda; OPENTRADUCTOR_ENDPOINT_TURN_P=0.x fija el umbral.
    endpoint_turn: bool = False
    endpoint_turn_threshold: float = 0.5
    endpoint_turn_silence_ms: float = 32.0
    # T-80: con endpoint_adaptive, la espera corta del caso "complete" (parcial que cierra frase y ASR al día) solo
    # se aplica si smart-turn también da p >= endpoint_turn_threshold; si no, endpoint_silence_ms (motivo
    # "complete_noturn"). Evita los cortes a mitad de frase vistos en la app (QA 21/11: 3/15 frases cortas partidas).
    # Carga smart-turn aunque endpoint_turn esté apagado. Env OPENTRADUCTOR_ENDPOINT_COMPLETE_TURN=1/0 manda.
    endpoint_complete_turn: bool = False
    vad_threshold: float = 0.5
    min_speech_ms: float = 120.0
    max_segment_s: float = 20.0  # corte forzado de seguridad
    # ASR: "parakeet" (MLX, GPU) | "moonshine-tiny" | "moonshine-small" (CPU, hilo propio)
    asr_backend: str = "parakeet"
    moonshine_update_interval_s: float = 0.1
    # T-88: opción transcription_interval de libmoonshine (s). None = default de la librería (~0,64 s entre
    # parciales). Env OPENTRADUCTOR_ASR_TRANSCRIPTION_INTERVAL_S manda ("0" = default de la librería).
    moonshine_transcription_interval_s: float | None = None
    asr_chunk_ms: float = 160.0
    # T-67: Moonshine en un proceso aparte (runtime/asr_process.py), fuera del GIL del motor. Default activo desde el
    # A/B final (artifacts/t67-final, junto con tts_first_chunk_max_words=2 y switch_interval_ms=1).
    # $OPENTRADUCTOR_ASR_PROCESS=1/0 manda. Sin efecto con parakeet.
    asr_process: bool = True
    # T-67: switch interval del GIL (ms, 1-50) que fija el Engine al crearse; None = no se toca (5 ms de Python).
    # $OPENTRADUCTOR_SWITCH_INTERVAL_MS=N lo fuerza; "off" lo apaga.
    switch_interval_ms: int | None = 1
    asr_context: tuple[int, int] = (256, 256)  # (izq, der) en frames de 80 ms; derecho pequeño degrada el texto
    # Al endpoint, no pasar al ASR la cola de audio que es solo silencio (toda la voz ya se procesó).
    asr_skip_silent_tail: bool = True  # T-58: default activo (WER sin cambio en el A/B)
    # T-83: de esa cola sí se pasan al ASR los primeros ms tras el último frame con voz. El VAD da silencio en
    # finales suaves («policia(l)», «nazi(smo)») y sin ese audio Moonshine pierde la última sílaba.
    # 0 = descartar toda la cola (comportamiento T-58). $OPENTRADUCTOR_ASR_TAIL_KEEP_MS manda (A/B de bench).
    asr_tail_keep_ms: float = 96.0
    # TTS
    voice: str = "af_heart"
    speed: float = 1.0
    # T-92/T-93 (BUG-9): interruptor general de las 3 medidas contra el atraso en charla continua (velocidad
    # adaptativa, muletillas sin voz, Player salta frases viejas con cola > player_skip_old_s). Off o
    # $OPENTRADUCTOR_TTS_CATCHUP=0 = camino de antes: speed fijo, se dicen las muletillas, el Player descarta lo
    # NUEVO pasados 60 s. Ver tts_catchup_on().
    tts_catchup: bool = True
    # T-93: 6 s = mínimo con p95 fin de habla->voz <= 5 s en la charla 1 (simulación QA: salta ~10 % de la voz,
    # que sigue en subtítulos). $OPENTRADUCTOR_PLAYER_SKIP_OLD_S lo cambia para A/B.
    # T-93 (criterio del lead: voz saltada <= 5 %, nunca frases con cifras o nombres): 8 s. Sim charla 1 con
    # recorte 5 + rampa 1->3 + silencio <= 80 ms: p95 5,6 s, voz no oída 3,6 %.
    player_skip_old_s: float = 8.0
    # T-93: al llegar una frase nueva con cola > este valor, se cortan las cláusulas no empezadas de la frase en
    # curso (la que suena termina; nunca si tiene cifras o nombres). $OPENTRADUCTOR_PLAYER_CUT_CUR_S (vacío o < 0 =
    # apagado).
    player_cut_cur_s: float | None = 5.0
    # T-93: topes del recorte (None = sin tope): nunca más de N s ni de esta fracción de la frase entera (charla 3:
    # 4 recortes de 5-7 s = frases enteras sin voz). $OPENTRADUCTOR_PLAYER_CUT_MAX_S / _CUT_MAX_FRAC (vacío o < 0 =
    # sin tope).
    player_cut_max_s: float | None = None
    player_cut_max_frac: float | None = None
    # T-93: silencio máximo (ms) al inicio y al final de cada chunk de Kokoro (en la charla 1 había 68 s de silencio
    # inicial en 433 chunks). $OPENTRADUCTOR_TTS_TRIM_KEEP_MS (vacío o < 0 = sin recorte).
    tts_trim_keep_ms: float | None = 80.0
    # T-92 (BUG-9): velocidad adaptativa por cláusula según la voz encolada sin reproducir: speed con <= lo_s,
    # sube lineal hasta tts_speed_max con >= hi_s. Off o $OPENTRADUCTOR_TTS_SPEED_ADAPTIVE=0 = speed fijo.
    tts_speed_adaptive: bool = True
    tts_speed_queue_lo_s: float = 1.0  # T-93: era 2
    tts_speed_queue_hi_s: float = 3.0  # T-93: era 5 (antes 8); acelera antes para no tener que saltar
    tts_speed_max: float = 1.30
    # T-92: frases que son solo muletilla (eh, em, ¿se entiende?, got it?) no se sintetizan; siguen en subtítulos.
    # $OPENTRADUCTOR_TTS_SKIP_FILLERS=0 = se dicen como hoy.
    tts_skip_fillers: bool = True
    # Recorte del silencio inicial del TTS; None = desactivado. Valor = margen (ms) que se conserva antes del primer sonido.
    tts_trim_lead_margin_ms: float | None = 30.0  # T-58: default 30 ms (antes None)
    # T-47: recorte del silencio FINAL de cada chunk TTS (-60 dBFS, bloques de 5 ms, conserva 80 ms; prefijo PCM exacto).
    # False = camino de hoy, sin cambios. Reversible por flag (--tts-trim-tail off) o env OPENTRADUCTOR_TTS_TRIM_TAIL.
    tts_trim_tail: bool = False
    # T-58: aplica el recorte de silencio inicial (tts_trim_lead_margin_ms) al 1.er chunk de CADA cláusula, no solo
    # de la frase. Kokoro abre cada cláusula con ~300-380 ms de silencio: sin esto, cortar el 1.er fragmento más corto
    # mete pausas >=300 ms a mitad de frase. False = camino de hoy.
    tts_trim_lead_all_clauses: bool = False
    # T-67: tope (ms) de la pausa entre fragmentos TTS a mitad de frase. Kokoro deja ~430 ms de cola + ~300 de
    # arranque por fragmento: sin tope cada corte suena como ~730 ms de pausa. Con N: arranque de cada cláusula a
    # tts_trim_lead_margin_ms y cola a N (corte por palabras) o max(N, 200) (tras , ; :); fin de oración sin tocar.
    # None = camino de hoy. Env OPENTRADUCTOR_TTS_CLAUSE_PAUSE_MS=N / off manda.
    tts_clause_pause_ms: float | None = None
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
    # T-77 (18/11): contexto largo por defecto (pedido de la persona: «mucho contexto»). Máximo de frases; el tope
    # real lo pone mt_context_tokens. 0 = sin historia (camino viejo).
    mt_context_turns: int = 200
    # T-58 contexto largo (pedido 17/11): caché KV de Qwen que sobrevive entre frases; cada frase solo hace prefill de
    # sus tokens nuevos. Con mt_context_turns > 0 la historia se recorta en bloque (n -> n/2) para no romper el prefijo.
    # T-77: encendido por defecto (A/B de T-68 en artifacts/t68-kv95: MT p50 −5 ms, 91/96 igual). Env =0 lo apaga.
    mt_session_kv: bool = True
    # T-68 contexto largo por tokens: tope (aprox.) de tokens de historia en el prompt de Qwen. Al pasarlo se descartan
    # las frases más viejas hasta quedar en la mitad (recorte en bloque: la caché de sesión se rehace solo entonces).
    # 0 = solo mt_context_turns. Con este tope, mt_context_turns es el máximo de frases (poner alto, p. ej. 200).
    # T-77: 2000 por defecto (lo medido en T-68). Env OPENTRADUCTOR_MT_CONTEXT_TOKENS=0 vuelve a solo turnos.
    mt_context_tokens: int = 2000
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
    tts_first_chunk_max_words: int | None = 2  # T-67 (20/11): 2 (antes 4; 8 en T-58), A/B final artifacts/t67-final
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
    # T-67 (19/11): queda apagado (decisión de @lead): sobre stream4adapt64 da -4 ms, dentro del ruido. Env =1 lo enciende.
    asr_spec_finish_on_pause: bool = False
    # T-58: MT en streaming hacia Kokoro (solo Qwen). El 1.er fragmento se sintetiza entre pasos del decode apenas el
    # prefijo lo fija (components.first_clause_final) y sale antes del mt_final. Mismos cortes y texto que el camino
    # normal. Apagado hasta el A/B de @bench. $OPENTRADUCTOR_MT_STREAM_FIRST_CHUNK=1/0 lo fuerza.
    mt_stream_first_chunk: bool = False
    # T-73 Idea B: alias de mt_stream_first_chunk con el nombre de la tarea ($OPENTRADUCTOR_MT_STREAM_TTS=1/0).
    # T-67 (19/11): encendido por defecto (solo actúa con Qwen). t67-asrfin: 0/80 huecos, mismo texto que off.
    mt_stream_tts: bool = True
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
    # T-67 sonda d): al empezar la voz de cada frase, si el hilo MLX está ocioso hace >= mt_prewarm_idle_s, una
    # síntesis descartada de un fonema de Kokoro (KokoroTTS.prewarm) para que el 1.er fragmento no pague Kokoro frío.
    # Solo con el ASR fuera del hilo MLX (Moonshine). Apagado; $OPENTRADUCTOR_TTS_PREWARM=1/0 manda.
    tts_prewarm_on_speech: bool = False
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


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def env_opt_seconds(name: str, default: float | None) -> float | None:
    """T-93: segundos opcionales por env. Sin env = default; vacío o < 0 = apagado (None); basura = default."""
    if name not in os.environ:
        return default
    v = os.environ[name].strip()
    if not v:
        return None
    try:
        f = float(v)
    except ValueError:
        return default
    return None if f < 0 else f


def tts_catchup_on(cfg) -> bool:
    """T-93: interruptor general (env OPENTRADUCTOR_TTS_CATCHUP=0/1 manda sobre cfg.tts_catchup)."""
    v = os.environ.get("OPENTRADUCTOR_TTS_CATCHUP")
    return (v == "1") if v in ("0", "1") else bool(getattr(cfg, "tts_catchup", True))


def tts_speed_params(cfg) -> tuple[float, float, float]:
    """(lo_s, hi_s, max_speed) de la velocidad adaptativa; env OPENTRADUCTOR_TTS_SPEED_LO_S / _HI_S / _MAX mandan
    (para A/B del banco sin tocar código)."""
    return (_env_float("OPENTRADUCTOR_TTS_SPEED_LO_S", cfg.tts_speed_queue_lo_s),
            _env_float("OPENTRADUCTOR_TTS_SPEED_HI_S", cfg.tts_speed_queue_hi_s),
            _env_float("OPENTRADUCTOR_TTS_SPEED_MAX", cfg.tts_speed_max))
