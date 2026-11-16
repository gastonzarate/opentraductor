"""Seleccion de modo del motor: baseline (por defecto) o incremental experimental (integrado en runtime/).

Todo lo pesado es perezoso: importar este modulo no importa la variante, mlx_lm ni modelos. Falla cerrado: sin variante de hash aprobado,
sin pesos locales o sin la venv de LLM, la comprobacion previa (preflight) marca FALTA y el servicio no inicia. Nunca descarga nada.
Se ejecuta con el python de la venv .venv-inc-llm:  .venv-inc-llm/bin/python -m runtime run --engine-mode incremental_experimental
Rutas y pines FIJOS para esta Mac (sin flags de ruta): variante, motor base y modelo Qwen del repositorio.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

from .preflight import Check

MODE_BASELINE = "baseline"
MODE_INCREMENTAL = "incremental_experimental"
ENGINE_MODES = (MODE_BASELINE, MODE_INCREMENTAL)

REPO = Path(__file__).resolve().parent.parent
ANALYSIS = REPO / "artifacts" / "analysis"
VARIANT_DIR = ANALYSIS / "incremental-llm-v1"  # contiene inc_llm/ (v0.1.10)
BASE_DIR = ANALYSIS / "incremental-adapter-v1"  # contiene inc_adapter/ (motor base fijado)
QWEN_DIR = ANALYSIS / "llm-mp4-a" / "models" / "qwen3-4b-instruct-2507-4bit"  # = MODEL_DIR fijo de qwen_prefix_factory
MODEL_MANIFEST = ANALYSIS / "llm-mp4-a" / "model_manifest.json"
# Pines APROBADOS por el coordinador. Cada uno es el sha256 del archivo de manifiesto; ademas se verifican los archivos listados.
# MLX real sigue sin validar (2A pendiente; sin lease GPU ni aprobacion 2B).
APPROVED_VARIANT_SHA256: str | None = "edf536e0dd60f877a2a928474870d3a15b75f4928c00e44000248d0be99b1a0b"  # HASHES.sha256 de la variante v0.1.10
APPROVED_BASE_SHA256: str | None = "d5fc4ae710e287de40170f22ad466288d54fffc6cbf73aacaa98159fec00ff83"  # HASHES.sha256 del motor base
APPROVED_MODEL_MANIFEST_SHA256: str | None = "a4a9d6ca046f468b5067c2d2acaee1492f5d9a7b2d7de8d18a3e43acab2ce296"
ADAPTER_SPEC = "inc_llm.adapter:IncrementalLLMAdapter"
IGNORED_CONTROLS = ("voice", "endpoint_silence_ms")  # no se ofrecen ni se aceptan en modo incremental (el motor fija su politica)
# flags del CLI que solo afectan al motor baseline: en modo incremental se rechazan (no se ignoran en silencio)
BASELINE_ONLY_FLAGS = (("asr_backend", "parakeet", "--asr-backend"), ("skip_silent_tail", None, "--skip-silent-tail"),
                       ("tts_trim_lead_ms", None, "--tts-trim-lead-ms"), ("tts_trim_tail", None, "--tts-trim-tail"), ("mt_normalize", None, "--mt-normalize"), ("tts_first_chunk_max_words", None, "--tts-first-chunk-max-words"),
                       ("normalize_asr_text", False, "--normalize-asr-text"), ("tts_bind_conjunctions", None, "--tts-bind-conjunctions"),
                       ("endpoint_ms", None, "--endpoint-ms"), ("no_warmup", False, "--no-warmup"))
LABEL = "Traducción simultánea · Experimental"  # texto corto de la UI; el detalle tecnico va en DETAIL (diagnostico/docs)
DETAIL = ("Qwen3-4B sin cache + Moonshine tiny + Kokoro; sin promesa de 300 ms; voz estandar fija. 'Entregado' = PCM enviado al reproductor, "
          "no reproduccion verificada. ES por linea y EN por unidad, sin relacion unidad-linea.")


@dataclass
class IncOptions:
    """Rutas fijas por defecto (esta Mac). Los tests pueden sustituirlas; el CLI NO las expone."""
    variant_dir: Path = VARIANT_DIR
    base_dir: Path = BASE_DIR
    qwen_dir: Path = QWEN_DIR
    model_manifest: Path = MODEL_MANIFEST
    store_path: Path | None = None  # SQLite propia de unidades (no toca el historial)


def verified_manifest_sha256(root: Path, subdir: str) -> str:
    """sha256 del manifiesto ``HASHES.sha256`` de ``root`` (el pin aprobado) tras verificar que cada archivo listado bajo ``subdir/`` coincide
    y que no hay .py de ``subdir/`` fuera del manifiesto. Lanza si algo difiere. No es un algoritmo propio de arbol: el pin es el del manifiesto."""
    root = Path(root)
    man = root / "HASHES.sha256"
    entries = {}
    for line in man.read_text().splitlines():
        if line.strip():
            digest, rel = line.split(None, 1)
            entries[rel.strip()] = digest
    for rel, digest in entries.items():
        if rel.startswith(subdir + "/"):
            f = root / rel
            if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest() != digest:
                raise ValueError(f"{rel} no coincide con HASHES.sha256")
    extra = [str(f.relative_to(root)) for f in (root / subdir).rglob("*.py") if str(f.relative_to(root)) not in entries]
    if extra:
        raise ValueError(f"archivos fuera del manifiesto: {extra}")
    return hashlib.sha256(man.read_bytes()).hexdigest()


def _venv_ok() -> bool:
    return Path(sys.prefix).name == ".venv-inc-llm"


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _baseline_checks() -> list[Check]:
    from . import config as C
    from .preflight import run_checks

    keep = ("asr:moonshine-tiny", "tts:kokoro", "g2p:", "vad:silero")
    return [c for c in run_checks(C.PipelineConfig(asr_backend="moonshine-tiny")) if c.name.startswith(keep)]


def checks(o: IncOptions) -> list[Check]:
    """Solo mira disco e importabilidad. No carga modelos ni importa la variante."""
    out: list[Check] = []

    def chk(name, fn):
        try:
            out.append(Check(name, True, str(fn())))
        except Exception as exc:  # noqa: BLE001
            out.append(Check(name, False, f"{type(exc).__name__}: {exc}"))

    def need(cond, msg):
        if not cond:
            raise FileNotFoundError(msg)

    def venv():
        need(_venv_ok(), f"entorno incorrecto ({sys.prefix}): ejecutar con .venv-inc-llm/bin/python")
        need(_has_module("mlx_lm"), "mlx_lm no importable en .venv-inc-llm")
        return "mlx_lm"

    def pinned(name, root, subdir, approved):
        need(approved is not None, f"pin de {name} no definido: bloqueado")
        need(Path(root, subdir).is_dir(), f"no existe {root}/{subdir}")
        got = verified_manifest_sha256(root, subdir)  # lanza si los archivos difieren del manifiesto
        need(got == approved, f"manifiesto de {name} distinto del aprobado ({got[:12]}...)")
        return f"{name} sha256 {got[:12]}..."

    def variant():
        return pinned("variante v0.1.10", o.variant_dir, "inc_llm", APPROVED_VARIANT_SHA256)

    def base():
        return pinned("motor base", o.base_dir, "inc_adapter", APPROVED_BASE_SHA256)

    def qwen():
        d = Path(o.qwen_dir)
        need(APPROVED_MODEL_MANIFEST_SHA256 is not None, "pin del manifiesto del modelo no definido")
        got = hashlib.sha256(Path(o.model_manifest).read_bytes()).hexdigest()
        need(got == APPROVED_MODEL_MANIFEST_SHA256, f"model_manifest.json distinto del aprobado ({got[:12]}...)")
        need((d / "config.json").is_file() and (d / "model.safetensors").is_file(), f"faltan config.json/model.safetensors en {d} (no se descarga)")
        return f"{d.name} (manifiesto {got[:12]}...)"

    chk("inc:venv-inc-llm", venv)
    chk("inc:variant", variant)
    chk("inc:base-engine", base)
    chk("inc:qwen-model", qwen)
    out.extend(_baseline_checks())
    return out


def mode_info(o: IncOptions) -> dict:
    return {"mode": MODE_INCREMENTAL, "experimental": True, "label": LABEL, "detail": DETAIL, "ignored_controls": list(IGNORED_CONTROLS),
            "delivery": "pcm_delivered_to_player", "playback": "not_verified", "no_300ms_claim": True,
            "applied": {"asr": "moonshine-tiny (transcription_interval 0.16)", "mt": "qwen3-4b-instruct-2507-4bit sin cache (head_cache=False)",
                        "tts": "kokoro (voz estandar fija)", "variant_sha256": APPROVED_VARIANT_SHA256, "base_sha256": APPROVED_BASE_SHA256,
                        "model_manifest_sha256": APPROVED_MODEL_MANIFEST_SHA256, "model_dir": str(o.qwen_dir)}}


BASELINE_INFO = {"mode": MODE_BASELINE, "experimental": False, "label": "", "ignored_controls": [], "applied": None}


def _adapter_factory(o: IncOptions):
    def make():
        if (APPROVED_VARIANT_SHA256 is None or APPROVED_BASE_SHA256 is None
                or verified_manifest_sha256(o.variant_dir, "inc_llm") != APPROVED_VARIANT_SHA256
                or verified_manifest_sha256(o.base_dir, "inc_adapter") != APPROVED_BASE_SHA256
                or APPROVED_MODEL_MANIFEST_SHA256 is None
                or hashlib.sha256(Path(o.model_manifest).read_bytes()).hexdigest() != APPROVED_MODEL_MANIFEST_SHA256):
            raise RuntimeError("variante/motor base incremental no aprobados o modificados: no se importan")
        for d in (o.base_dir, o.variant_dir):
            if str(d) not in sys.path:
                sys.path.insert(0, str(d))
        mod, cls = ADAPTER_SPEC.split(":")
        return getattr(importlib.import_module(mod), cls)()  # import perezoso, solo al abrir el motor

    return make


def build(o: IncOptions):
    """-> (engine_factory, mode_info, preflight_fn). No importa la variante ni abre nada."""
    from .inc_app.bridge import IncrementalModeConfig, SqliteUnitStore, make_engine_factory

    if o.store_path is None:
        raise ValueError('store_path requerido')
    path = str(o.store_path)
    factory = make_engine_factory(IncrementalModeConfig(llm_mt={"head_cache": False}), _adapter_factory(o), store_factory=lambda: SqliteUnitStore(path))
    return factory, mode_info(o), lambda cfg: checks(o)


def reject_baseline_only_flags(a) -> list[str]:
    return [flag for attr, default, flag in BASELINE_ONLY_FLAGS if getattr(a, attr, default) != default]
