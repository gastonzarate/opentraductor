"""Servicio de la aplicación: une Engine (ASR→MT→TTS), audio y TextStore.

Cumple la interfaz que espera ``http_control.ControlServer`` (síncrona, segura
entre hilos). El motor corre en un loop asyncio propio (hilo ``ot-loop``).

Reglas:
* El micrófono se abre solo en ``start()`` (acción explícita) y se cierra en
  ``stop()``. No se guarda audio en ningún caso.
* Las métricas son valores medidos; lo que no se mide va ``null``.
* La cola de entrada es acotada; si se llena se descarta audio NUEVO y se
  informa ``overflow`` (nunca en silencio).
* Los textos se escriben en SQLite desde un hilo escritor, no desde callbacks
  de audio.
"""
from __future__ import annotations

import asyncio
import concurrent.futures as cf
import queue
import sqlite3
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import audio_io as default_audio
from . import config as C
from .config import PipelineConfig
from .http_control import ServiceError
from .text_store import TextStore, TextStoreError

HOT_FIELDS = {"monitor_volume", "monitor_id", "endpoint_silence_ms"}
ALL_FIELDS = {"input_id", "output_id", "monitor_id", "monitor_volume", "voice", "endpoint_silence_ms",
              "audio_latency"}
VOICE_LABELS = {
    "af_heart": "Heart · inglés (EE. UU.)",
    "af_bella": "Bella · inglés (EE. UU.)",
    "af_nicole": "Nicole · inglés (EE. UU.)",
    "am_michael": "Michael · inglés (EE. UU.)",
    "am_adam": "Adam · inglés (EE. UU.)",
}
INPUT_QUEUE_MAX = 1500  # bloques de ~20 ms = 30 s
PLAYBACK_DRAIN_MAX_S = 10.0
# Mismo dispositivo para entrada y salida (p. ej. BlackHole): el captura se abre tras la salida
# y tras esta pausa. Abiertos casi a la vez, en macOS la captura queda muerta sin excepción.
SAME_DEVICE_SETTLE_S = 0.3
FIRST_BLOCK_TIMEOUT_S = 2.0
# Valor de output_id para «solo subtítulos»: no se abre ningún reproductor y la voz no suena.
NO_OUTPUT = "__none__"


def _saved_endpoint_ms(saved: dict, default: float) -> float:
    """endpoint_silence_ms guardado, o ``default`` si no hay o si es el default viejo (250, T-58).

    La app guardaba todo el config al cambiar cualquier opción, así que un 250 guardado casi
    nunca es una elección del usuario. Cualquier otro valor guardado se respeta."""
    v = saved.get("endpoint_silence_ms")
    if v is None:
        return float(default)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return float(default)
    return float(default) if v == C.LEGACY_ENDPOINT_SILENCE_MS else v


def _ms(seconds: float | None) -> float | None:
    return round(seconds * 1000, 1) if seconds is not None else None


def _stream_info(latency_s: float | None, blocksize: int | None) -> dict:
    """Lo que informa PortAudio al abrir el stream. No es una medición física.
    blocksize 0 significa tamaño variable, no cero muestras."""
    return {"stream_reported_latency_ms": _ms(latency_s), "blocksize": blocksize,
            "blocksize_variable": blocksize == 0}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class RuntimeService:
    def __init__(
        self,
        db_path: str | Path,
        *,
        pipeline_cfg: PipelineConfig | None = None,
        engine_factory: Callable | None = None,
        audio: Any = None,
        preflight: Callable | None = None,
        max_records: int = 10_000,
        explicit: frozenset[str] | set[str] = frozenset(),
        audio_latency: str | None = None,  # None: rige el guardado o 'default'; si se indica, prevalece
        first_block_timeout_s: float = FIRST_BLOCK_TIMEOUT_S,
    ) -> None:
        self._first_block_timeout_s = float(first_block_timeout_s)
        # latencia y blocksize que informó el último stream abierto (no es medición física)
        self._audio_info: dict[str, Any] = {}
        # Secuencia de micrófonos: cambia solo al abrir un micrófono nuevo, no en stop. Así los
        # bloques ya capturados se vacían al parar, y los callbacks de un micrófono anterior se descartan.
        self._stream_seq = 0
        # ``explicit``: campos que la persona fijó en esta corrida (p. ej. --endpoint-ms).
        # Prevalecen sobre el valor guardado en SQLite; sin ellos rige el guardado o el default.
        self._explicit = frozenset(explicit)
        self._pcfg = pipeline_cfg or PipelineConfig()
        self._engine_factory = engine_factory or self._default_engine_factory
        self._audio = audio or default_audio
        self._preflight = preflight or self._default_preflight
        self._store = TextStore(db_path, max_records=max_records)
        self._lock = threading.RLock()
        self._state = "idle"
        self._error: str | None = None
        self._subs: dict[int, Callable[[dict], None]] = {}
        self._sub_seq = 0
        self._models = {k: {"state": "not_loaded", "detail": None} for k in ("asr", "mt", "tts")}
        saved = self._store.get_config()
        self._cfg: dict[str, Any] = {
            "input_id": saved.get("input_id"),
            "output_id": saved.get("output_id"),
            "monitor_id": saved.get("monitor_id"),
            "monitor_volume": float(saved.get("monitor_volume", 0.8)),
            "voice": saved.get("voice", self._pcfg.voice),
            "endpoint_silence_ms": _saved_endpoint_ms(saved, self._pcfg.endpoint_silence_ms),
        }
        if "endpoint_silence_ms" in self._explicit:
            self._cfg["endpoint_silence_ms"] = float(self._pcfg.endpoint_silence_ms)
        if "voice" in self._explicit:
            self._cfg["voice"] = self._pcfg.voice
        mode = saved.get("audio_latency", "default")
        if mode not in default_audio.LATENCY_MODES:
            mode = "default"  # valor guardado inválido: se usa el default
        if audio_latency is not None:
            mode = audio_latency
        default_audio.latency_arg(mode)  # ValueError si el modo no existe
        self._cfg["audio_latency"] = mode
        self._pcfg.endpoint_silence_ms = self._cfg["endpoint_silence_ms"]
        self._pcfg.voice = self._cfg["voice"]
        # runtime
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, name="ot-loop", daemon=True)
        self._thread.start()
        self._engine = None
        self._engine_voice: str | None = None
        self._mic = None
        self._player = None
        self._monitor = None
        self._in_q: asyncio.Queue | None = None
        self._feeder: asyncio.Task | None = None
        self._feeding = False  # True mientras el feeder procesa un bloque (push al motor)
        self._metrics_task: asyncio.Task | None = None
        self._session_id: str | None = None
        self._stream_id: str | None = None
        self._audio_clock_s = 0.0
        self._pending_es: dict[str, str] = {}
        self._pending_raw: dict[str, str | None] = {}  # salida cruda del ASR por segmento
        self._shown_es: dict[str, str] = {}  # T-55: último texto ES publicado por fila (para `discarded`)
        self._endpoint_t: dict[str, tuple[float, str | None]] = {}
        self._last_plan: dict[str, str | None] = {}  # planned_text_en del último tts_chunk por seg_id
        self._first_chunk_seen: set[str] = set()
        self._m = {"overflow_count": 0, "input_dropped_ms": 0.0, "engine_dropped_ms": 0.0,
                   "last_endpoint_to_audio_ms": None, "audio_status": None}
        self._writer_q: queue.Queue = queue.Queue(maxsize=1000)
        self._writer = threading.Thread(target=self._writer_loop, name="ot-writer", daemon=True)
        self._writer.start()
        self._busy: cf.Future | None = None
        self._refreshing = False  # refresco de PortAudio en curso: start() devuelve 409 hasta que termine
        self._stale_device_keys: set = set()  # ids guardados AMBIGUOS (mismo nombre y dirección) tras un refresco: hay que volver a elegirlos
        self._gen = 0  # sube con cada start/stop; un start con gen vieja no abre nada
        self._closed = False

    # ------------------------------------------------------------ defaults
    @staticmethod
    def _default_engine_factory(cfg, emit):
        from .pipeline import Engine

        return Engine(cfg, emit)

    @staticmethod
    def _default_preflight(cfg):
        from .preflight import run_checks

        return run_checks(cfg)

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    @property
    def _audio_latency(self) -> str:
        return self._cfg["audio_latency"]

    @property
    def _latency_arg(self):
        return default_audio.latency_arg(self._cfg["audio_latency"])

    def _submit(self, coro) -> cf.Future:
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    # --------------------------------------------------------- suscripción
    def subscribe(self, callback: Callable[[dict], None]):
        with self._lock:
            self._sub_seq += 1
            self._subs[self._sub_seq] = callback
            return self._sub_seq

    def unsubscribe(self, handle) -> None:
        with self._lock:
            self._subs.pop(handle, None)

    def _publish(self, event: dict) -> None:
        with self._lock:
            subs = list(self._subs.values())
        for cb in subs:
            try:
                cb(event)
            except Exception:  # noqa: BLE001  un suscriptor roto no afecta al resto
                pass

    def _set_state(self, state: str, error: str | None = None) -> None:
        with self._lock:
            self._state = state
            self._error = error
        ev = {"type": "state", "state": state}
        if error:
            ev["error"] = error
        self._publish(ev)

    # ------------------------------------------------------------ consultas
    def devices(self) -> dict:
        return self._audio.devices_payload()

    def refresh_devices(self) -> dict:
        """Reinicializa PortAudio para ver dispositivos nuevos. SOLO con el servicio ocioso: nunca durante start/running/stopping, carga
        del motor o con streams abiertos. Reserva (bajo el mismo lock que start) el flag ``_refreshing``: un start concurrente recibe 409
        y un refresco concurrente con un start ya reservado también. La config (ids por nombre) no se toca; se informa cuáles ya no existen."""
        refresh = getattr(self._audio, "refresh_devices", None)
        if refresh is None:
            raise ServiceError("unsupported", "Este audio no permite actualizar dispositivos.", 501)
        with self._lock:
            if self._closed:
                raise ServiceError("closed", "El servicio se está cerrando.", 409)
            if self._state != "idle":
                raise ServiceError("bad_state", "No se pueden actualizar los dispositivos con una sesión en curso.", 409)
            if self._refreshing:
                raise ServiceError("busy", "Ya se están actualizando los dispositivos.", 409)
            self._refreshing = True
            cfg = dict(self._cfg)
        try:
            try:
                info = refresh()
            except default_audio.AudioBusy as exc:
                raise ServiceError("bad_state", f"No se pueden actualizar los dispositivos: {exc}.", 409) from exc
            except Exception as exc:  # noqa: BLE001
                raise ServiceError("audio_refresh", f"No se pudo actualizar la lista de dispositivos: {exc}", 500) from exc
            missing, candidates, ambiguous = [], {}, []
            cand = getattr(self._audio, "device_candidates", None)
            is_amb = getattr(self._audio, "device_is_ambiguous", None)
            for key, kind in (("input_id", "in"), ("output_id", "out"), ("monitor_id", "out")):
                if cfg.get(key) and cfg[key] != NO_OUTPUT:
                    try:
                        self._audio.find_device(cfg[key], kind)  # estricto: un id cuya numeración cambió NO se reasigna solo
                        if is_amb is not None and is_amb(cfg[key]):
                            ambiguous.append(key)  # existe, pero sin UID no se sabe si sigue siendo el MISMO dispositivo físico
                    except LookupError:
                        missing.append(key)
                        if cand is not None:
                            candidates[key] = cand(cfg[key], kind)  # solo informativo
            with self._lock:
                self._stale_device_keys = set(ambiguous)  # bloquea start() hasta que el usuario los vuelva a elegir (configure)
            return {"devices": self.devices(), "refresh": info, "config_missing": missing, "config_candidates": candidates,
                    "config_ambiguous": ambiguous}
        finally:
            with self._lock:
                self._refreshing = False

    def voices(self) -> dict:
        out = []
        try:
            d = C.snapshot(C.KOKORO_REPO, C.KOKORO_REV, ("config.json",)) / "voices"
            have = {p.stem for p in d.glob("*.safetensors")}
        except FileNotFoundError:
            have = set()
        for v in C.KOKORO_VOICES:
            if v in have:
                # id interno sin cambios; la etiqueta es la que ve la persona
                out.append({"id": v, "label": VOICE_LABELS.get(v, v)})
        return {"voices": out}

    def metrics(self) -> dict:
        b = {"asr_backlog_s": None, "dropped_ms": None, "queue_depth": None}
        eng = self._engine
        if eng is not None and hasattr(eng, "backlog") and self._state in ("running", "stopping"):
            try:
                b = eng.backlog()
            except Exception:  # noqa: BLE001
                pass
        # dropped_ms = entrada descartada + backlog del motor (acumulado, medido en el motor).
        # Las pérdidas de reproducción van aparte para no mezclar fuentes.
        dropped = None
        if b["dropped_ms"] is not None or self._m["input_dropped_ms"]:
            dropped = round((b["dropped_ms"] or 0) + self._m["input_dropped_ms"])
        pl = self._player
        pl_dropped = round(pl.dropped_ms) if pl is not None else None
        return {
            "queue_depth": b["queue_depth"],
            "asr_backlog_s": b["asr_backlog_s"],
            "dropped_ms": dropped,
            "overflow_count": self._m["overflow_count"],
            "playback_queue_s": round(pl.queued_s, 3) if pl is not None else None,
            "playback_dropped_ms": pl_dropped,
            # T-92: voz vieja saltada (frases enteras sin reproducir) cuando la cola pasó de 20 s.
            "playback_skipped_s": round(getattr(pl, "skipped_s", 0.0), 2) if pl is not None else None,
            # T-93: parte de playback_skipped_s que salió de recortar la frase en curso (player_cut_cur_s).
            "playback_cut_cur_s": round(getattr(pl, "cut_cur_s_total", 0.0), 2) if pl is not None else None,
            # T-93: un registro por salto o recorte (seg_id = id del historial, text_en, s, reason, queue_s, t).
            "playback_skipped_segs": list(getattr(pl, "skipped_segs", ())) if pl is not None else None,
            "playback_partial_blocks": pl.partial_blocks if pl is not None else None,
            "playback_status_events": pl.status_events if pl is not None else None,
            "last_first_audio_ms": None,  # el fin real de habla no se conoce en vivo
            "last_endpoint_to_audio_ms": self._m["last_endpoint_to_audio_ms"],
        }

    def engine_config(self) -> dict:
        """Configuración efectiva del motor (sin audio). Se registra en status y en cada sesión."""
        d = asdict(self._pcfg)
        d["asr_context"] = list(d["asr_context"])
        return d

    def status(self) -> dict:
        try:
            pre = [{"name": c.name, "ok": c.ok, "detail": c.detail} for c in self._preflight(self._pcfg)]
        except Exception as exc:  # noqa: BLE001
            pre = [{"name": "preflight", "ok": False, "detail": f"{type(exc).__name__}: {exc}"}]
        with self._lock:
            return {
                "state": self._state,
                "error": self._error,
                "preflight": pre,
                "models": {k: dict(v) for k, v in self._models.items()},
                "config": dict(self._cfg),
                "engine": self.engine_config(),
                # latencia y blocksize REALES del último stream abierto (no los pedidos)
                "audio": {"latency_mode": self._audio_latency, **{k: dict(v) for k, v in self._audio_info.items()}},
                "metrics": self.metrics(),
            }

    # ---------------------------------------------------------- configuración
    def configure(self, values: dict) -> dict:
        if not isinstance(values, dict):
            raise ServiceError("invalid", "La configuración debe ser un objeto.", 400)
        unknown = set(values) - ALL_FIELDS
        if unknown:
            raise ServiceError("invalid", f"Campos desconocidos: {sorted(unknown)}", 400)
        with self._lock:
            active = self._state in ("starting", "running", "stopping")
        if active:
            blocked = set(values) - HOT_FIELDS
            changed = {k for k in blocked if values[k] != self._cfg.get(k)}
            if changed:
                raise ServiceError(
                    "conflict",
                    f"No se puede cambiar {sorted(changed)} con la sesión en curso. Detené la sesión primero.",
                    409,
                )
        clean: dict[str, Any] = {}
        for k, v in values.items():
            clean[k] = self._validate(k, v)
        with self._lock:
            self._cfg.update(clean)
            self._stale_device_keys -= set(clean)  # elegir de nuevo (explícito) levanta la marca de ids ambiguos
        if "endpoint_silence_ms" in clean:
            self._pcfg.endpoint_silence_ms = clean["endpoint_silence_ms"]
            if self._engine is not None:
                self._engine.cfg.endpoint_silence_ms = clean["endpoint_silence_ms"]
        if "voice" in clean:
            self._pcfg.voice = clean["voice"]
        if "monitor_volume" in clean and self._monitor is not None:
            self._monitor.set_volume(clean["monitor_volume"])
        if "monitor_id" in clean and active:
            self._submit(self._swap_monitor(clean["monitor_id"]))
        try:
            self._store.set_config(dict(self._cfg))
        except TextStoreError as exc:
            raise ServiceError("storage", str(exc), 500) from exc
        return self.status()

    def _validate(self, k: str, v: Any) -> Any:
        def bad(msg):
            raise ServiceError("invalid", msg, 400)

        if k in ("input_id", "output_id", "monitor_id"):
            if v is None or v == "":
                if k == "monitor_id":
                    return None
                bad(f"{k} es obligatorio.")
            if not isinstance(v, str) or len(v) > 300:
                bad(f"{k} inválido.")
            return v
        if k == "monitor_volume":
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0.0 <= float(v) <= 1.0:
                bad("monitor_volume debe estar entre 0 y 1.")
            return float(v)
        if k == "audio_latency":
            if v not in default_audio.LATENCY_MODES:
                bad("audio_latency debe ser default, low o high.")
            return v
        if k == "endpoint_silence_ms":
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not 60.0 <= float(v) <= 2000.0:
                bad("endpoint_silence_ms debe estar entre 60 y 2000.")
            return float(v)
        if k == "voice":
            ids = {x["id"] for x in self.voices()["voices"]}
            if v not in ids:
                bad("Voz no disponible.")
            return v
        bad("Campo inválido.")

    # ------------------------------------------------------------ ciclo de vida
    def _is_current(self, gen: int) -> bool:
        with self._lock:
            return self._gen == gen

    def start(self) -> None:
        with self._lock:
            if self._state not in ("idle", "error"):
                raise ServiceError("bad_state", "La sesión ya está en curso.", 409)
            cfg = dict(self._cfg)
        if not cfg["input_id"] or not cfg["output_id"]:
            raise ServiceError("invalid", "Elegí micrófono y salida antes de iniciar.", 400)
        bad = [c for c in self._preflight(self._pcfg) if not c.ok]
        if bad:
            raise ServiceError(
                "preflight", "Faltan modelos o dependencias: " + "; ".join(c.name for c in bad), 503
            )
        # Reserva, envío y registro del future en un solo bloque: un stop concurrente
        # ve el start completo (generación nueva y future) o nada.
        with self._lock:
            if self._closed:
                raise ServiceError("closed", "El servicio se está cerrando.", 409)
            if self._state not in ("idle", "error"):
                raise ServiceError("bad_state", "La sesión ya está en curso.", 409)
            if self._refreshing:
                raise ServiceError("busy", "Se están actualizando los dispositivos de audio; reintentá en un momento.", 409)
            if self._stale_device_keys & {"input_id", "output_id", "monitor_id"}:
                raise ServiceError("device_ambiguous", "Hay dispositivos guardados con el mismo nombre que otro (ids ambiguos tras actualizar la lista). "
                                   "Elegilos de nuevo antes de iniciar.", 409)
            self._gen += 1
            gen = self._gen
            self._state = "starting"
            self._error = None
            self._busy = self._submit(self._start_async(cfg, gen))
        self._publish({"type": "state", "state": "starting"})

    def stop(self) -> None:
        with self._lock:
            prev = self._state
            if prev in ("idle", "stopping"):
                return
            busy = self._busy
            if prev != "error":
                self._gen += 1  # invalida cualquier start en curso
                self._state = "stopping"
        if prev == "error":
            self._set_state("idle")
            return
        self._publish({"type": "state", "state": "stopping"})
        self._submit(self._stop_async(busy))

    async def _start_async(self, cfg: dict, gen: int) -> None:
        try:
            await self._ensure_engine()
            if not self._is_current(gen):
                return  # stop llegó durante la carga: no abrir nada; _stop_async cierra
            dev_in = self._audio.find_device(cfg["input_id"], "in")
            self._audio_info.pop("output", None)
            if cfg["output_id"] != NO_OUTPUT:  # solo subtítulos: sin reproductor, el resto del flujo no cambia
                dev_out = self._audio.find_device(cfg["output_id"], "out")
                self._player = self._audio.Player(dev_out, latency=self._latency_arg)
                self._player.skip_old_s = self._skip_old_s()
                self._player.cut_cur_s = self._cut_cur_s()
                self._player.start()
                self._audio_info["output"] = _stream_info(self._player.latency_s, self._player.blocksize)
            if cfg.get("monitor_id"):
                await self._open_monitor(cfg["monitor_id"])
            if not self._is_current(gen):
                return
            if cfg["input_id"] == cfg["output_id"]:
                await asyncio.sleep(SAME_DEVICE_SETTLE_S)
                if not self._is_current(gen):
                    return
            # snapshot de la sesión: ajustes de la UI más la configuración efectiva del motor
            self._session_id = self._store.start_session({**cfg, "engine": self.engine_config()})
            self._stream_id = f"live-{self._session_id[:8]}"
            self._pending_es.clear()
            self._pending_raw.clear()
            self._shown_es.clear()
            self._endpoint_t.clear()  # mapeos por segmento: solo viven durante una sesión
            self._first_chunk_seen.clear()
            self._last_plan.clear()
            self._audio_clock_s = 0.0
            # Primero se invalida el micrófono anterior y después se instala la cola nueva:
            # un callback viejo durante start_stream (await) ya no puede entrar en la cola nueva.
            self._stream_seq += 1
            seq = self._stream_seq
            self._in_q = asyncio.Queue(maxsize=INPUT_QUEUE_MAX)
            await self._engine.start_stream(self._stream_id)
            if not self._is_current(gen):
                return
            self._feeder = asyncio.ensure_future(self._feed())
            self._metrics_task = asyncio.ensure_future(self._metrics_loop())
            loop = self._loop

            seen = threading.Event()  # primer bloque de ESTE micrófono
            def on_block(pcm: bytes, _s: int = seq, _ev: threading.Event = seen) -> None:
                # hilo de PortAudio: solo encola. Un callback de un micrófono anterior se descarta.
                if self._stream_seq != _s:
                    return
                _ev.set()
                loop.call_soon_threadsafe(self._enqueue_block, pcm, _s)

            mic = self._audio.MicCapture(dev_in, on_block, lambda st, _s=seq: self._on_mic_status(st, _s),
                                         latency=self._latency_arg)
            if not self._is_current(gen):
                return
            self._mic = mic
            mic.start()
            self._audio_info["input"] = _stream_info(mic.latency_s, mic.blocksize)
            if not self._is_current(gen):
                return
            # Un micrófono sin bloques no es una sesión válida: error visible, no 'running' mudo.
            got = await asyncio.to_thread(seen.wait, self._first_block_timeout_s)
            if not got:
                raise RuntimeError("El micrófono no entregó audio al iniciar. Revisá la entrada elegida.")
            if not self._is_current(gen):
                return  # nunca 'running' tras un stop
            if self._session_id:  # diagnóstico del stream en el snapshot de la sesión (sin audio)
                try:
                    self._writer_q.put_nowait(("cfg", self._session_id,
                                               {"audio": {"latency_mode": self._audio_latency, **self._audio_info}}))
                except queue.Full:
                    self._publish({"type": "error", "message": "Cola de escritura llena: diagnóstico no guardado."})
            self._set_state("running")
        except Exception as exc:  # noqa: BLE001
            if not self._is_current(gen):
                return  # el stop ya es dueño del cierre
            await self._teardown()
            self._set_state("error", f"{type(exc).__name__}: {exc}")

    async def _ensure_engine(self) -> None:
        if self._engine is not None and self._engine_voice == self._pcfg.voice:
            return
        if self._engine is not None:
            await self._engine.close()
            self._engine = None
        for k in self._models:
            self._models[k] = {"state": "loading", "detail": None}
        self._publish({"type": "state", "state": "starting"})
        try:
            eng = self._engine_factory(self._pcfg, self._on_engine_event)
            await eng.open()
        except Exception as exc:  # noqa: BLE001
            for k in self._models:
                self._models[k] = {"state": "error", "detail": f"{type(exc).__name__}: {exc}"}
            raise
        self._engine = eng
        self._engine_voice = self._pcfg.voice
        if hasattr(eng, "playback_queue_fn"):  # T-92: velocidad de la voz según la cola REAL del Player
            eng.playback_queue_fn = lambda: self._player.queued_s if self._player is not None else 0.0
        load = getattr(eng, "load_ms", {}) or {}
        for k in self._models:
            ms = load.get(k)
            self._models[k] = {"state": "ready", "detail": f"carga {round(ms)} ms" if ms else None}

    def _skip_old_s(self) -> float | None:
        """T-92/T-93: umbral de cola (s) a partir del cual el Player salta frases viejas; None = política vieja."""
        if not C.tts_catchup_on(self._pcfg):
            return None
        return C.env_opt_seconds("OPENTRADUCTOR_PLAYER_SKIP_OLD_S", float(self._pcfg.player_skip_old_s))

    def _cut_cur_s(self) -> float | None:
        """T-93: cola (s) a partir de la cual una frase nueva recorta lo no empezado de la que suena; None = no."""
        if not C.tts_catchup_on(self._pcfg):
            return None
        d = getattr(self._pcfg, "player_cut_cur_s", None)
        return C.env_opt_seconds("OPENTRADUCTOR_PLAYER_CUT_CUR_S", None if d is None else float(d))

    async def _open_monitor(self, dev_id: str) -> None:
        dev = self._audio.find_device(dev_id, "out")
        m = self._audio.Player(dev, volume=self._cfg["monitor_volume"])
        m.skip_old_s = self._skip_old_s()
        m.cut_cur_s = self._cut_cur_s()
        m.start()
        self._monitor = m

    async def _swap_monitor(self, dev_id: str | None) -> None:
        old, self._monitor = self._monitor, None
        if old is not None:
            old.stop()
        if dev_id:
            try:
                await self._open_monitor(dev_id)
            except Exception as exc:  # noqa: BLE001
                self._publish({"type": "error", "message": f"No se pudo abrir el monitor: {exc}"})

    def _on_mic_status(self, status: str, seq: int) -> None:
        # estado de PortAudio del micrófono: solo si sigue siendo el de la sesión actual
        if self._stream_seq == seq:
            self._m["audio_status"] = status

    def _enqueue_block(self, pcm: bytes, seq: int | None = None) -> None:
        if seq is not None and seq != self._stream_seq:
            return  # bloque de un micrófono anterior
        q = self._in_q
        if q is None:
            return
        try:
            q.put_nowait(pcm)
        except asyncio.QueueFull:
            ms = len(pcm) / 2 / 16000 * 1000
            self._m["input_dropped_ms"] += ms
            self._m["overflow_count"] += 1
            self._publish({"type": "overflow", "reason": "input_queue", "dropped_ms": round(ms)})

    async def _feed(self) -> None:
        while True:
            pcm = await self._in_q.get()
            t = self._audio_clock_s
            self._audio_clock_s += len(pcm) / 2 / 16000
            self._feeding = True
            try:
                await self._engine.push(self._stream_id, pcm, t)
            except Exception as exc:  # noqa: BLE001
                self._publish({"type": "error", "message": f"Error en el motor: {type(exc).__name__}: {exc}"})
            finally:
                self._feeding = False

    async def _metrics_loop(self) -> None:
        while True:
            await asyncio.sleep(1.0)
            self._publish({"type": "metrics", **self.metrics()})

    async def _stop_async(self, busy: cf.Future | None = None) -> None:
        err = None
        # Esperar a que termine el start que estaba en curso al pedir el stop (p. ej. carga
        # del motor). Ese start ve la generación cambiada y no abre micrófono; después se
        # cierra todo aquí.
        if busy is not None and not busy.done():
            try:
                await asyncio.wrap_future(busy)
            except Exception:  # noqa: BLE001
                pass
        try:
            mic, self._mic = self._mic, None
            if mic is not None:
                mic.stop()  # primero el micrófono: no entra audio nuevo
            if self._feeder is not None:
                # procesar lo ya capturado antes de cerrar
                t0 = time.monotonic()
                # cola vacía Y sin push en curso: el último bloque también debe terminar en el motor
                while self._in_q is not None and (not self._in_q.empty() or self._feeding) \
                        and time.monotonic() - t0 < 5:
                    await asyncio.sleep(0.02)
            if self._engine is not None and self._stream_id:
                await self._engine.end_stream(self._stream_id)
            t0 = time.monotonic()
            while self._player is not None and self._player.queued_s > 0.05 \
                    and time.monotonic() - t0 < PLAYBACK_DRAIN_MAX_S:
                await asyncio.sleep(0.05)
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
        finally:
            await self._teardown()
        self._set_state("error", err) if err else self._set_state("idle")

    async def _teardown(self) -> None:
        mic, self._mic = self._mic, None
        if mic is not None:
            try:
                mic.stop()
            except Exception:  # noqa: BLE001
                pass
        for t in (self._feeder, self._metrics_task):
            if t is not None:
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
        self._feeder = self._metrics_task = None
        self._in_q = None
        for attr in ("_player", "_monitor"):
            p = getattr(self, attr)
            setattr(self, attr, None)
            if p is not None:
                try:
                    p.stop()
                except Exception:  # noqa: BLE001
                    pass
        sid, self._session_id = self._session_id, None
        if sid:
            self._writer_q.put(("end", sid))
        self._stream_id = None
        self._endpoint_t.clear()
        self._first_chunk_seen.clear()
        self._last_plan.clear()
        self._pending_raw.clear()
        self._shown_es.clear()

    # ------------------------------------------------------ eventos del motor
    async def _on_engine_event(self, kind: str, f: dict) -> None:
        seg = f.get("seg_id")
        key = str(seg) if seg is not None else None
        # Contrato T-55 (memory/decisiones/t55-contrato-eventos-frase.md): un seg_id por frase, nunca un
        # partial con text_es vacío, `from_seg_id` al pasar de la fila live al id de commit y `discarded`
        # en vez de descartes mudos.
        if kind == "asr_partial":
            text = f.get("text") or ""
            if key is not None and text.strip():
                self._shown_es[key] = text
                self._publish({"type": "partial", "seg_id": key, "text_es": text})
        elif kind == "endpoint":
            if key is not None:
                # Con cláusulas simultáneas puede haber sonado audio antes del endpoint: se guarda la cláusula
                # que sonaba y se mide hasta el primer audio de una cláusula nueva (lo que falta oír).
                self._endpoint_t[key] = (time.monotonic(), self._last_plan.get(key))
                if len(self._endpoint_t) > 1000:
                    self._endpoint_t.clear()
        elif kind == "asr_final":
            text = f.get("text") or ""
            src = f.get("from_seg_id")
            src = str(src) if src is not None and str(src) != key else None
            if not text.strip():  # ASR final vacío: la fila queda visible como descartada
                row = src or key
                if row is not None:
                    shown = self._shown_es.pop(row, "")
                    if key is not None:
                        self._shown_es.pop(key, None)
                    self._publish_discarded(row, shown, "empty")
                return
            self._pending_es[key] = text
            self._pending_raw[key] = f.get("text_raw")
            ev = {"type": "partial", "seg_id": key, "text_es": text}
            if src is not None:
                ev["from_seg_id"] = src
                self._shown_es.pop(src, None)
            self._shown_es[key] = text
            self._publish(ev)
        elif kind == "discarded":
            if key is None:
                return
            src = f.get("from_seg_id")
            row = str(src) if src is not None else key
            text = (f.get("text") or self._pending_es.pop(key, None) or self._shown_es.get(row)
                    or self._shown_es.get(key) or "")
            self._pending_es.pop(key, None)
            self._pending_raw.pop(key, None)
            self._shown_es.pop(row, None)
            self._shown_es.pop(key, None)
            self._publish_discarded(row, text, f.get("reason") or "cancelled")
        elif kind == "mt_final":
            es = self._pending_es.pop(key, "")
            raw = self._pending_raw.pop(key, None)  # None si el motor no informó el original
            self._shown_es.pop(key, None)
            ts = _now_iso()
            self._publish({"type": "segment", "seg_id": key, "text_es": es, "text_es_raw": raw,
                           "text_en": f.get("text", ""), "ts": ts})
            if self._session_id:
                try:
                    self._writer_q.put_nowait(("seg", self._session_id, key, es, f.get("text", ""), ts, raw))
                except queue.Full:
                    self._publish({"type": "error", "message": "Cola de escritura de historial llena."})
        elif kind == "tts_chunk":
            plan = f.get("planned_text_en")
            ep = self._endpoint_t.get(key)
            if ep is not None and (key not in self._first_chunk_seen or plan != ep[1]):
                del self._endpoint_t[key]
                self._m["last_endpoint_to_audio_ms"] = round((time.monotonic() - ep[0]) * 1000)
            if key not in self._first_chunk_seen:
                self._first_chunk_seen.add(key)
                if len(self._first_chunk_seen) > 1000:
                    self._first_chunk_seen.clear()
                    self._last_plan.clear()
            self._last_plan[key] = plan
            for p in (self._player, self._monitor):
                if p is not None:
                    before = p.dropped_s
                    p.write(f["pcm"], f["sample_rate"], key, text_en=plan)
                    lost = p.dropped_s - before
                    if lost > 0:
                        self._m["overflow_count"] += 1
                        self._publish({"type": "overflow", "reason": "playback_queue",
                                       "dropped_ms": round(lost * 1000), "seg_id": key})
        elif kind == "overflow":
            self._m["overflow_count"] += 1
            self._publish({"type": "overflow", "reason": f.get("reason") or "queue_full",
                           "dropped_ms": f.get("dropped_ms"), "seg_id": key})
        elif kind == "error":
            self._publish({"type": "error", "message": str(f.get("error_kind") or "Error desconocido")})

    _DISCARD_REASONS = ("short", "empty", "cancelled")

    def _publish_discarded(self, seg_id: str, text_es: str, reason: str) -> None:
        reason = reason if reason in self._DISCARD_REASONS else "cancelled"
        self._publish({"type": "discarded", "seg_id": seg_id, "text_es": text_es, "reason": reason})

    # ---------------------------------------------------------------- historial
    def _writer_loop(self) -> None:
        while True:
            item = self._writer_q.get()
            if item is None:
                return
            try:
                if item[0] == "seg":
                    _, sid, seg, es, en, ts, raw = item
                    self._store.add_segment(sid, seg, es, en, ts, text_es_raw=raw)
                elif item[0] == "end":
                    self._store.end_session(item[1])
                elif item[0] == "cfg":
                    self._store.update_session_config(item[1], item[2])
                elif item[0] == "flush":
                    item[1].set()
            except (TextStoreError, sqlite3.Error) as exc:
                # Visible para la UI: el texto no quedó guardado (sesión borrada, disco lleno,
                # base bloqueada). El hilo sigue vivo para los siguientes elementos.
                self._publish({"type": "error", "message": f"Historial no guardado: {exc}"})
            except Exception as exc:  # noqa: BLE001  nunca dejar morir al escritor
                self._publish({"type": "error", "message": f"Escritor del historial: {type(exc).__name__}: {exc}"})

    def history(self, limit: int, offset: int) -> dict:
        self._flush_writer()
        try:
            return self._store.history(limit, offset)
        except TextStoreError as exc:
            raise ServiceError("storage", str(exc), 500) from exc

    def export_history(self, format: str) -> str:  # noqa: A002
        self._flush_writer()
        try:
            return self._store.export(format)
        except TextStoreError as exc:
            raise ServiceError("invalid", str(exc), 400) from exc

    def delete_history(self, session_id: str | None) -> int:
        """Borra todo el historial (``session_id=None``) o una sesión. La comprobación y el borrado ocurren bajo
        ``self._lock`` (el mismo que usa start): no puede abrirse una sesión entre medias."""
        self._flush_writer()
        with self._lock:
            try:
                if session_id is None:
                    try:
                        return self._store.delete_all()
                    except TextStoreError as exc:
                        raise ServiceError("storage", f"No se borró el historial ({exc}).", 500) from exc
                return 1 if self._store.delete_session(session_id) else 0  # una sesión abierta se rechaza aquí
            except TextStoreError as exc:
                raise ServiceError("conflict", str(exc), 409) from exc

    def _flush_writer(self, timeout: float = 2.0) -> None:
        ev = threading.Event()
        self._writer_q.put(("flush", ev))
        # el escritor procesa en orden: al llegar a 'flush' ya escribió todo lo previo
        ev.wait(timeout)

    # ------------------------------------------------------------------- cierre
    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True  # ningún start nuevo a partir de aquí
            active = self._state in ("starting", "running", "stopping")
            busy = self._busy
            if active:
                self._gen += 1  # invalida un start en curso: no debe abrir mic al terminar la carga
        if active:
            try:
                self._submit(self._stop_async(busy)).result(timeout=30)
            except Exception:  # noqa: BLE001
                pass
        if busy is not None:
            try:
                busy.result(timeout=5)
            except Exception:  # noqa: BLE001
                pass
        if self._engine is not None:
            try:
                self._submit(self._engine.close()).result(timeout=15)
            except Exception:  # noqa: BLE001
                pass
        self._writer_q.put(None)
        self._writer.join(timeout=3)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=3)
        self._store.close()
