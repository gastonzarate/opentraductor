"""CLI: ``uv run --extra runtime python -m runtime <comando>``.

  check     comprueba modelos y dependencias en disco (sin red)
  devices   lista micrófonos y salidas (no abre ninguno)
  run       arranca el servicio y la interfaz en http://127.0.0.1:PUERTO/ (8877 por defecto;
            8765 suele ser de Agent Hub)
            (el micrófono solo se abre al pulsar Iniciar en la interfaz)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import threading
from pathlib import Path


def default_db() -> Path:
    base = Path(os.environ.get("OPENTRADUCTOR_DATA", "~/.local/share/opentraductor")).expanduser()
    return base / "history.sqlite3"


def cmd_check(a) -> int:
    from .config import PipelineConfig
    from .preflight import run_checks

    if a.engine_mode == "incremental_experimental":
        from . import inc_mode

        cs = inc_mode.checks(_inc_options(a))
    else:
        cs = run_checks(PipelineConfig(
            asr_backend=a.asr_backend,
            mt_engine=_mt_engine(a),
            qwen_model_dir=getattr(a, "qwen_dir", None),
        ))
    for c in cs:
        print(("OK    " if c.ok else "FALTA "), c.name, "-", c.detail)
    return 0 if all(c.ok for c in cs) else 1


def cmd_devices(a) -> int:
    from .audio_io import devices_payload

    print(json.dumps(devices_payload(), ensure_ascii=False, indent=2))
    return 0


def _inc_options(a):
    from . import inc_mode

    db = Path(a.db) if getattr(a, "db", None) else default_db()
    return inc_mode.IncOptions(store_path=db.with_name("incremental_units.sqlite3"))  # rutas y pines fijos de esta Mac


def _tts_trim_tail(a) -> bool:
    """T-47: el flag explícito manda; sin flag, OPENTRADUCTOR_TTS_TRIM_TAIL=1 (lo fija la .app); si no, off."""
    v = getattr(a, "tts_trim_tail", None)
    if v is None:
        return os.environ.get("OPENTRADUCTOR_TTS_TRIM_TAIL", "").strip().lower() in ("1", "on", "true")
    return v == "on"


def _mt_normalize(a) -> bool:
    """BUG-7: igual que T-47. Flag explícito manda; sin flag, OPENTRADUCTOR_MT_NORMALIZE=1; si no, off."""
    v = getattr(a, "mt_normalize", None)
    if v is None:
        return os.environ.get("OPENTRADUCTOR_MT_NORMALIZE", "").strip().lower() in ("1", "on", "true")
    return v == "on"


def _mt_engine(a) -> str:
    """T-51: flag explícito manda; sin flag, OPENTRADUCTOR_MT_ENGINE (lo fija la .app); si no, marian."""
    v = getattr(a, "mt_engine", None)
    if v is None:
        v = os.environ.get("OPENTRADUCTOR_MT_ENGINE", "").strip().lower() or "marian"
    if v not in ("marian", "qwen"):
        raise SystemExit(f"OPENTRADUCTOR_MT_ENGINE inválido: {v!r} (marian|qwen)")
    return v


def build_service(a):
    """Crea el servicio con la configuración de la línea de comandos. No carga modelos."""
    from .config import PipelineConfig
    from .service import RuntimeService

    cfg = PipelineConfig(
        asr_backend=a.asr_backend,
        asr_skip_silent_tail=a.skip_silent_tail,
        tts_trim_lead_margin_ms=a.tts_trim_lead_ms,
        tts_trim_tail=_tts_trim_tail(a),
        mt_normalize=_mt_normalize(a),
        mt_engine=_mt_engine(a),
        qwen_model_dir=getattr(a, "qwen_dir", None),
        tts_first_chunk_max_words=a.tts_first_chunk_max_words,
        asr_normalize_text=a.normalize_asr_text,
        tts_bind_conjunctions=a.tts_bind_conjunctions,
        warmup_in_open=not a.no_warmup,
    )
    explicit = set()
    if a.endpoint_ms is not None:
        cfg.endpoint_silence_ms = a.endpoint_ms
        explicit.add("endpoint_silence_ms")
    if a.audio_latency is not None:
        explicit.add("audio_latency")
    db = Path(a.db) if a.db else default_db()
    extra = {}
    if a.engine_mode == "incremental_experimental":  # import perezoso: el baseline nunca toca inc_mode ni la variante
        from . import inc_mode

        factory, info, pre = inc_mode.build(_inc_options(a))
        extra = {"engine_factory": factory, "engine_mode": info, "preflight": pre}
    return RuntimeService(db, pipeline_cfg=cfg, explicit=explicit, audio_latency=a.audio_latency, **extra), cfg


def cmd_run(a) -> int:
    from .http_control import ControlServer
    from .preflight import force_offline, run_checks

    force_offline()

    svc, cfg = build_service(a)
    if cfg.asr_backend != "parakeet":
        print(f"ASR: {cfg.asr_backend} (experimental, opción de prueba)")
    for c in svc._preflight(cfg):  # el mismo preflight que usa el servicio (baseline: run_checks)
        if not c.ok:
            print(f"FALTA {c.name} - {c.detail}")
    db = Path(a.db) if a.db else default_db()
    # La licencia vive junto a la base de datos: con --db o OPENTRADUCTOR_DATA apuntando a otra carpeta,
    # la prueba y la clave quedan ahí (útil para pruebas de QA con una prueba nueva o vencida).
    # Edición de la Mac App Store: la tienda cobra y la guía 2.4.5(vi) prohíbe claves propias. Sin licencia, sin red,
    # y sin importar runtime/license.py (sign_store.sh lo borra del bundle de tienda).
    store = os.environ.get("OPENTRADUCTOR_EDITION") == "store"
    lic = None
    if not store:
        from .license import LicenseManager, default_license_path

        lic = LicenseManager(db.with_name("license.json") if a.db else default_license_path())
    try:
        srv = ControlServer(svc, port=a.port, license=lic, store=store)
        srv.start()
    except OSError as exc:
        svc.close()  # el servicio ya creó hilos y una base: se cierran antes de salir
        print(f"No se pudo abrir el puerto {a.port}: {exc}. Probá otro con --port.", file=sys.stderr)
        return 1
    if a.engine_mode == "incremental_experimental":
        print("MODO INCREMENTAL EXPERIMENTAL (sin promesa de 300 ms; audio entregado, no reproduccion verificada).")
    print(f"opentraductor en {srv.url}  ("
          + ("local; sin conexiones tras la descarga de modelos)" if store else "local; única conexión: la activación de la licencia, una vez)"))
    print(f"historial de textos: {db}")
    print("Ctrl+C para salir. El micrófono se abre solo al pulsar Iniciar.")
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    try:
        stop.wait()
    finally:
        print("cerrando...")
        srv.stop()
        svc.close()
    return 0


ASR_BACKENDS = ("parakeet", "moonshine-tiny", "moonshine-small")


def _finite(text: str) -> float:
    try:
        v = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"no es un número: {text!r}") from None
    if not math.isfinite(v):
        raise argparse.ArgumentTypeError(f"debe ser un número finito, no {text!r}")
    return v


def _nonneg_ms(text: str) -> float:
    v = _finite(text)
    if v < 0:
        raise argparse.ArgumentTypeError("debe ser >= 0")
    return v


def _positive_int(text: str) -> int:
    try:
        v = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"debe ser un entero, no {text!r}") from None
    if v < 1:
        raise argparse.ArgumentTypeError("debe ser un entero mayor que 0")
    return v


def _endpoint_ms(text: str) -> float:
    # mismo rango que la API de configuración (RuntimeService._validate)
    v = _finite(text)
    if not 60.0 <= v <= 2000.0:
        raise argparse.ArgumentTypeError("debe estar entre 60 y 2000 ms")
    return v


ENGINE_MODES = ("baseline", "incremental_experimental")


def _add_mode_args(sp) -> None:
    sp.add_argument("--engine-mode", choices=ENGINE_MODES, default="baseline",
                    help="baseline (por defecto, estable) o incremental_experimental (ejecutar con .venv-inc-llm; rutas y pines fijos; sin promesa de 300 ms)")


def _add_engine_args(sp) -> None:
    sp.add_argument("--asr-backend", choices=ASR_BACKENDS, default="parakeet",
                    help="motor de reconocimiento (por defecto parakeet; moonshine-* es experimental)")


def _add_mt_args(sp):
    """T-51: run y check eligen el mismo traductor (el preflight verifica el que se va a usar)."""
    sp.add_argument("--mt-engine", choices=("marian", "qwen"), default=None,
                    help="traductor (T-51). Sin el flag: OPENTRADUCTOR_MT_ENGINE; si no, marian")
    sp.add_argument("--qwen-dir", default=None,
                    help="carpeta del Qwen MLX. Sin el flag: OPENTRADUCTOR_QWEN_DIR")


def build_parser():
    p = argparse.ArgumentParser(prog="python -m runtime")
    sub = p.add_subparsers(dest="cmd", required=True)
    ck = sub.add_parser("check")
    _add_engine_args(ck)
    _add_mode_args(ck)
    _add_mt_args(ck)
    ck.set_defaults(fn=cmd_check)
    sub.add_parser("devices").set_defaults(fn=cmd_devices)
    r = sub.add_parser("run")
    r.add_argument("--port", type=int, default=8877)
    r.add_argument("--db", default=None, help="ruta de la base SQLite de textos")
    r.add_argument("--endpoint-ms", type=_endpoint_ms, default=None,
                   help="silencio para cerrar un segmento; si se indica, prevalece sobre el valor guardado")
    r.add_argument("--no-warmup", action="store_true", help="no precalentar modelos al iniciar")
    _add_engine_args(r)
    _add_mode_args(r)
    r.add_argument("--tts-trim-lead-ms", type=_nonneg_ms, default=None,
                   help="recorta el silencio inicial del TTS, conservando este margen en ms (experimental)")
    r.add_argument("--tts-trim-tail", choices=("on", "off"), default=None,
                   help="recorta el silencio final de cada fragmento TTS (-60 dBFS, conserva 80 ms). "
                        "Sin el flag: OPENTRADUCTOR_TTS_TRIM_TAIL=1 lo activa; si no, off")
    _add_mt_args(r)
    r.add_argument("--mt-normalize", choices=("on", "off"), default=None,
                   help="antes de Marian: '120 mil' -> '120.000' y traduce oración por oración. "
                        "Sin el flag: OPENTRADUCTOR_MT_NORMALIZE=1 lo activa; si no, off")
    r.add_argument("--skip-silent-tail", action="store_true",
                   help="no envía al ASR la cola silenciosa del segmento (experimental)")
    r.add_argument("--tts-first-chunk-max-words", type=_positive_int, default=None,
                   help="palabras máximas del primer fragmento de TTS (entero > 0; experimental)")
    r.add_argument("--audio-latency", choices=("default", "low", "high"), default=None,
                   help="latencia pedida a PortAudio: default (sounddevice, 'high'), low o high. "
                        "Si se indica, prevalece sobre el valor guardado en la base")
    r.add_argument("--tts-bind-conjunctions", action="store_true",
                   help="no corta el TTS justo antes ni después de conjunciones (experimental)")
    r.add_argument("--normalize-asr-text", action="store_true",
                   help="normaliza el texto del ASR antes de traducir; el original se guarda aparte (experimental)")
    r.set_defaults(fn=cmd_run)
    return p


def main(argv=None) -> int:
    p = build_parser()
    a = p.parse_args(argv)
    if a.cmd == "run" and a.engine_mode == "incremental_experimental":
        from . import inc_mode

        bad = inc_mode.reject_baseline_only_flags(a)
        if bad:  # no se ofrecen controles que el modo incremental ignoraria
            p.error("no aplican en modo incremental_experimental: " + ", ".join(bad))
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
