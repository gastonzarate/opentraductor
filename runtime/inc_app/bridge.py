"""Puente del motor incremental (cualquier Adapter del contrato del benchmark) como motor del servicio de la app local. Experimental.

Integrado en runtime/ (modo ``--engine-mode incremental_experimental``). El servicio (runtime.service.RuntimeService) acepta
``engine_factory(cfg, emit)``; este modulo la implementa (ver runtime/inc_mode.py):

    factory = make_engine_factory(IncrementalModeConfig(...), adapter_factory=lambda: IncrementalLLMAdapter())
    RuntimeService(db, pipeline_cfg, engine_factory=factory, ...)

Interfaz de motor que el servicio usa: open(), start_stream(id), push(id, pcm, t), end_stream(id), close(), backlog(), cfg, load_ms.
Eventos que emite hacia el servicio (compatibles con ``_on_engine_event``): asr_partial, endpoint, asr_final, mt_final (una vez por unidad
EMITIDA), tts_chunk (por fragmento, mientras la captura sigue), overflow, error; mas ``inc_unit`` (informativo; un servicio sin parche lo ignora).
Reglas: el audio entra continuo (cola acotada, sin acumular frases); nada especulativo o cancelado se reproduce ni se persiste; los fallos son
visibles; lo persistido son textos y metricas con estado explicito. No promete 300 ms ni ha corrido con modelos reales.
"""
from __future__ import annotations

import asyncio
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Protocol

EmitFn = Callable[[str, dict], Awaitable[None]]

# estados explicitos de una unidad en el puente
# "delivered" = PCM entregado al servicio/player (observado en tts_chunk). NO prueba que se reprodujo: el player no informa reproduccion
# efectiva al motor; por eso se persiste ``playback = not_verified``.
UNIT_STATES = ("translated", "delivering", "delivered", "delivered_partial", "interrupted", "cancelled", "discarded")
PERSISTED_STATES = ("delivered", "delivered_partial", "interrupted")  # solo unidades con PCM observado
TERMINAL_STATES = ("cancelled", "discarded") + PERSISTED_STATES


@dataclass
class IncrementalModeConfig:
    mode: str = "incremental_experimental"
    backend_label: str = "qwen3-4b-uncached + moonshine tiny (transcription_interval 0.16) + kokoro (M5 Pro)"
    input_queue_chunks: int = 250  # chunks de entrada acotados (no se acumula una frase)
    drain_timeout_s: float = 60.0
    llm_mt: dict | None = None

    def __post_init__(self):
        if self.mode != "incremental_experimental":
            raise ValueError(f"modo desconocido: {self.mode!r}")
        if self.input_queue_chunks < 1:
            raise ValueError("input_queue_chunks >= 1")

    def status(self, state: str) -> dict:
        return {"mode": self.mode, "state": state, "backend": self.backend_label, "experimental": True,
                "claims": "experimental; no validado; sin promesa de 300 ms; latencias por medir"}


class UnitStore(Protocol):
    def record_line(self, rec: dict) -> None: ...
    def record_unit(self, rec: dict) -> bool: ...  # False si ya existia (idempotente por session+unit_id)
    def close(self) -> None: ...


class MemoryUnitStore:
    def __init__(self):
        self.lines: list[dict] = []
        self.units: dict[tuple, dict] = {}
        self.closed = False

    def record_line(self, rec):
        self.lines.append(dict(rec))

    def record_unit(self, rec):
        k = (rec["session"], rec["unit_id"])
        if k in self.units:
            return False
        self.units[k] = dict(rec)
        return True

    def by_unit(self, unit_id):
        return [v for (_, u), v in self.units.items() if u == unit_id]

    def close(self):
        self.closed = True


class SqliteUnitStore:
    """Tablas propias (no modifica text_store): inc_lines e inc_units, aisladas por ``session`` (= stream_id de cada arranque).
    Solo unidades con PCM entregado, con estado; sin audio. ``playback`` siempre 'not_verified'."""

    def __init__(self, path: str):
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        with self._lock:
            self._db.executescript("""
            CREATE TABLE IF NOT EXISTS inc_lines(session TEXT NOT NULL, line_id TEXT NOT NULL, text_es TEXT NOT NULL, ts TEXT, PRIMARY KEY(session, line_id));
            CREATE TABLE IF NOT EXISTS inc_units(session TEXT NOT NULL, unit_id TEXT NOT NULL, text_en TEXT NOT NULL,
              state TEXT NOT NULL CHECK(state IN ('delivered','delivered_partial','interrupted')),
              playback TEXT NOT NULL DEFAULT 'not_verified' CHECK(playback='not_verified'),
              n_chunks INTEGER NOT NULL CHECK(n_chunks>0), total_samples INTEGER NOT NULL CHECK(total_samples>0), reason TEXT, ts TEXT,
              PRIMARY KEY(session, unit_id));""")
            self._db.commit()

    def record_line(self, rec):
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO inc_lines VALUES(?,?,?,?)", (rec["session"], rec["line_id"], rec["text_es"], rec.get("ts")))
            self._db.commit()

    def record_unit(self, rec):
        with self._lock:
            if self._db.execute("SELECT 1 FROM inc_units WHERE session=? AND unit_id=?", (rec["session"], rec["unit_id"])).fetchone():
                return False
            self._db.execute("INSERT INTO inc_units(session,unit_id,text_en,state,n_chunks,total_samples,reason,ts) VALUES(?,?,?,?,?,?,?,?)",
                             (rec["session"], rec["unit_id"], rec["text_en"], rec["state"], rec["n_chunks"], rec["total_samples"],
                              rec.get("reason"), rec.get("ts")))
            self._db.commit()
            return True

    def rows(self, table):
        with self._lock:
            return self._db.execute(f"SELECT * FROM {table}").fetchall()

    def close(self):
        with self._lock:
            try:
                self._db.close()
            except sqlite3.Error:
                pass


@dataclass
class _Unit:
    uid: str
    sid: str
    text_en: str = ""
    state: str = "translated"
    chunks: int = 0  # ENTREGADOS: emit(tts_chunk) exitoso (no claims del adapter, no solo recibidos)
    recv_chunks: int = 0
    emit_failed: int = 0
    samples: int = 0
    reason: str | None = None


_UNIT_KINDS = ("mt_partial", "tts_chunk", "unit_done", "unit_cancelled")


class IncrementalEngineBridge:
    """Implementa la interfaz de motor del servicio sobre un Adapter incremental."""

    def __init__(self, mode: IncrementalModeConfig, adapter, emit: EmitFn, store: UnitStore | None = None, owns_store: bool = False,
                 quiet_s: float = 0.05, quiesce_timeout_s: float = 2.0):
        self.mode, self.adapter, self.emit = mode, adapter, emit
        self.store = store or MemoryUnitStore()
        self.owns_store = owns_store or store is None
        self.quiet_s, self.quiesce_timeout_s = quiet_s, quiesce_timeout_s
        self.cfg = SimpleNamespace(endpoint_silence_ms=None, voice=None)  # el servicio puede asignar endpoint_silence_ms: no aplica aqui
        self.load_ms: dict[str, float] = {}
        self.state = "idle"
        self.units: dict[tuple, _Unit] = {}
        self.stats = {"input_dropped_chunks": 0, "input_dropped_ms": 0.0, "rejected_chunks": 0, "lost_in_queue_chunks": 0,
                      "stale_chunks_dropped": 0, "units_delivered": 0, "units_partial": 0, "units_interrupted": 0, "units_cancelled": 0,
                      "units_discarded_unemitted": 0, "units_done_without_pcm": 0, "claim_mismatch": 0, "stale_stream_events": 0, "tts_emit_failures": 0, "errors": 0}
        self._q: asyncio.Queue | None = None
        self._feeder: asyncio.Task | None = None
        self._pump: asyncio.Task | None = None
        self._sid: str | None = None
        self._accepting = False
        self._closing = False
        self._closed = False
        self._in_handle = False
        self._last_activity = 0.0
        self._sid_rate = 16000

    # ---------------------------------------------------------------- ciclo
    async def open(self) -> None:
        self.state = "starting"
        t = time.perf_counter()
        try:
            await self.adapter.open({"llm_mt": self.mode.llm_mt} if self.mode.llm_mt else {})
        except Exception as exc:  # noqa: BLE001
            self.state = "error"
            raise RuntimeError(f"no se pudo abrir el motor incremental: {type(exc).__name__}: {exc}") from exc
        self.load_ms["open"] = round((time.perf_counter() - t) * 1000, 1)
        eng = getattr(self.adapter, "engine", None)
        self.load_ms.update(getattr(eng, "load_ms", {}) or {})
        self._pump = asyncio.ensure_future(self._pump_events())
        self.state = "ready"

    async def start_stream(self, stream_id: str) -> None:
        if self._closed or self._closing:
            raise RuntimeError("puente cerrado")
        if self._sid is not None:
            raise RuntimeError(f"stream activo ({self._sid}): llamar end_stream antes de reiniciar")
        if self._pump is None or self._pump.done():
            raise RuntimeError("el bombeo de eventos del motor no esta activo (open fallo o termino)")
        self._sid = stream_id  # identidad activa ANTES de start_stream: el adapter puede emitir al arrancar
        try:
            await self.adapter.start_stream(stream_id, 0)
        except BaseException:
            self._sid = None  # revertir: ningun evento posterior de ese stream es valido
            self._drop_units(stream_id)
            raise
        self._q = asyncio.Queue(maxsize=self.mode.input_queue_chunks)
        self._feeder = asyncio.ensure_future(self._feed(self._q, stream_id))
        self._accepting = True
        self.state = "running"

    async def push(self, stream_id: str, pcm: bytes, t_start_s: float) -> None:
        """No bloquea la captura: cola acotada. Llena: se descarta el chunk NUEVO y se informa (overflow). Tras fallo/stop/cierre no acepta audio."""
        ms = len(pcm) / 2 / self._sid_rate * 1000
        if not self._accepting or self._q is None or stream_id != self._sid:
            self.stats["rejected_chunks"] += 1  # el error/stop que lo causo ya se informo; se cuenta, no se acumula
            return
        try:
            self._q.put_nowait((stream_id, pcm, t_start_s))
        except asyncio.QueueFull:
            self.stats["input_dropped_chunks"] += 1
            self.stats["input_dropped_ms"] += ms
            await self.emit("overflow", {"stream_id": stream_id, "reason": "input_queue", "dropped_ms": round(ms)})

    async def _feed(self, q: asyncio.Queue, stream_id: str) -> None:
        try:
            while True:
                item = await q.get()
                try:
                    await self.adapter.push_audio(*item)
                finally:
                    q.task_done()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            await self._fail(stream_id, f"audio_bridge: {type(exc).__name__}: {exc}")

    async def _report(self, stream_id, msg: str) -> None:
        """Error visible que no detiene el stream."""
        self.stats["errors"] += 1
        try:
            await self.emit("error", {"stream_id": stream_id, "error_kind": msg})
        except Exception:  # noqa: BLE001
            pass

    async def _fail(self, stream_id, msg: str) -> None:
        """Fallo del puente: estado error, no se acepta mas audio, se cancela el alimentador y se informa."""
        self.state = "error"
        self._accepting = False
        f = self._feeder
        if f is not None and f is not asyncio.current_task() and not f.done():
            f.cancel()
        await self._report(stream_id, msg)

    def _discard_queue(self) -> None:
        q = self._q
        while q is not None and not q.empty():
            q.get_nowait()
            q.task_done()
            self.stats["lost_in_queue_chunks"] += 1

    async def _quiesce(self) -> bool:
        """Barrera explicita: ningun evento en proceso y sin actividad durante quiet_s (acotado por quiesce_timeout_s)."""
        end = time.monotonic() + self.quiesce_timeout_s
        while time.monotonic() < end:
            if self._pump is None or self._pump.done():
                return True
            if not self._in_handle and time.monotonic() - self._last_activity >= self.quiet_s:
                return True
            await asyncio.sleep(0.005)
        return False

    async def end_stream(self, stream_id: str) -> None:
        """Parada limpia. Deadline unico drain_timeout_s para (drenaje de la cola + adapter.end_stream); despues una espera de quietud
        acotada por quiesce_timeout_s. Maximo total = drain_timeout_s + quiesce_timeout_s. La quietud es una heuristica provisional:
        NO es prueba de que el motor drenó todos sus eventos."""
        sid = self._sid or stream_id
        deadline = time.monotonic() + self.mode.drain_timeout_s  # UNICO para drenaje de cola + adapter.end_stream
        self._accepting = False
        q, feeder = self._q, self._feeder
        if feeder is not None:
            join = asyncio.ensure_future(q.join())
            try:
                done, _ = await asyncio.wait({join, feeder}, timeout=max(0.0, deadline - time.monotonic()), return_when=asyncio.FIRST_COMPLETED)
                if not done:
                    await self._fail(sid, "audio_bridge: drain_timeout")
            finally:
                join.cancel()
                feeder.cancel()
                await asyncio.gather(join, feeder, return_exceptions=True)
            self._discard_queue()  # feeder muerto o timeout: lo no entregado se cuenta, no queda colgado
        self._feeder = None
        try:
            await asyncio.wait_for(self.adapter.end_stream(sid), max(0.001, deadline - time.monotonic()))
        except Exception as exc:  # noqa: BLE001  (incluye TimeoutError)
            await self._fail(sid, f"end_stream: {type(exc).__name__}: {exc}")
        self._last_activity = time.monotonic()
        settled = await self._quiesce()
        if not settled:
            await self._report(sid, "end_stream: eventos del motor sin asentarse; unidades abiertas se marcan interrumpidas")
        self._finalize_units(sid, "stream_end" if settled else "stream_end_events_unsettled")
        self._sid = None
        self._q = None
        if self.state != "error":
            self.state = "ready"

    def _finalize_units(self, sid, reason: str) -> None:
        for (usid, _), u in list(self.units.items()):
            if usid != sid:
                continue
            if u.state == "delivering":  # cortada por la parada: interrumpida, no entregada completa
                self._finish(u, "interrupted", reason)
            elif u.state == "translated":
                u.state = "discarded"
                self.stats["units_discarded_unemitted"] += 1
        self._drop_units(sid)

    def _drop_units(self, sid) -> None:
        for k in [k for k in self.units if k[0] == sid]:
            del self.units[k]  # libera memoria; los estados ya quedaron persistidos/contados. La identidad activa (_sid) filtra eventos tardios

    async def close(self) -> None:
        if self._closed:
            return
        self._closing = True
        self._accepting = False
        self._closed = True
        if self._feeder is not None:
            self._feeder.cancel()
            await asyncio.gather(self._feeder, return_exceptions=True)
            self._feeder = None
        self._discard_queue()
        if self._sid is not None:
            self._finalize_units(self._sid, "closed")
            self._sid = None
        try:
            await self.adapter.close()
        except Exception as exc:  # noqa: BLE001
            await self._report("", f"close: {type(exc).__name__}: {exc}")
        if self._pump is not None:
            try:
                await asyncio.wait_for(self._pump, 10.0)
            except asyncio.TimeoutError:
                self._pump.cancel()
                await asyncio.gather(self._pump, return_exceptions=True)
            except asyncio.CancelledError:
                pass
        if self.owns_store:
            try:
                self.store.close()
            except Exception:  # noqa: BLE001
                pass
        self.state = "closed"

    def backlog(self) -> dict:
        pending = sum(1 for u in self.units.values() if u.state in ("translated", "delivering"))
        return {"asr_backlog_s": None, "dropped_ms": round(self.stats["input_dropped_ms"]),
                "queue_depth": (self._q.qsize() if self._q else 0) + pending}

    # ------------------------------------------------------- eventos del motor
    def _finish(self, u: _Unit, state: str, reason: str | None = None) -> None:
        if u.state in TERMINAL_STATES:
            return  # idempotente: nunca se persiste ni se cuenta dos veces
        if u.chunks <= 0:
            u.state = "discarded"  # sin PCM observado no hay nada emitido que persistir
            self.stats["units_done_without_pcm"] += 1
            return
        u.state, u.reason = state, reason or u.reason
        key = {"delivered": "units_delivered", "delivered_partial": "units_partial", "interrupted": "units_interrupted"}[state]
        self.stats[key] += 1
        self.store.record_unit({"session": u.sid, "unit_id": u.uid, "text_en": u.text_en, "state": state, "n_chunks": u.chunks,
                                "total_samples": u.samples, "reason": u.reason, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})

    async def _pump_events(self) -> None:
        try:
            async for ev in self.adapter.events():
                self._in_handle = True
                self._last_activity = time.monotonic()
                try:
                    await self._handle(ev)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    await self._report(getattr(ev, "stream_id", ""), f"bridge_event: {type(exc).__name__}: {exc}")
                finally:
                    self._in_handle = False
                    self._last_activity = time.monotonic()
            if not self._closing:
                await self._fail(self._sid or "", "events: el iterador de eventos del motor termino inesperadamente")
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            await self._fail(self._sid or "", f"events: {type(exc).__name__}: {exc}")

    def _unit(self, sid, cid, **kw) -> _Unit:
        k = (sid, cid)
        if k not in self.units:
            self.units[k] = _Unit(cid, sid, **kw)
        return self.units[k]

    async def _handle(self, ev) -> None:
        k = getattr(ev.kind, "value", ev.kind)
        sid = ev.stream_id
        cid = getattr(ev, "commit_id", None)
        if k != "error" and sid != self._sid:
            self.stats["stale_stream_events"] += 1  # evento de un stream inactivo (anterior, o ya cerrado): no suena, no se muestra, no se persiste
            return
        if k in _UNIT_KINDS and not cid:
            await self._report(sid, f"evento {k} sin commit_id: descartado")
            return
        if k == "asr_partial":
            await self.emit("asr_partial", {"stream_id": sid, "text": ev.text or "", "audio_pos_s": ev.audio_pos_s, "seg_id": f"{sid}:live"})
        elif k == "endpoint":
            await self.emit("endpoint", {"stream_id": sid, "audio_pos_s": ev.audio_pos_s, "seg_id": cid or f"{sid}:live"})
        elif k == "asr_final":
            if cid:
                self.store.record_line({"session": sid, "line_id": cid, "text_es": ev.text or "", "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})
            await self.emit("asr_final", {"stream_id": sid, "text": ev.text or "", "text_raw": None, "seg_id": cid})
        elif k == "mt_partial":
            u = self._unit(sid, cid)
            if u.state not in TERMINAL_STATES:
                u.text_en = ev.text or u.text_en
        elif k == "tts_chunk":
            u = self._unit(sid, cid)
            if u.state in TERMINAL_STATES:
                self.stats["stale_chunks_dropped"] += 1  # tardio de unidad cancelada/cerrada: no se reproduce
                return
            if not ev.pcm:
                await self._report(sid, f"tts_chunk sin pcm en unidad {cid}")
                return
            u.recv_chunks += 1  # recibido del adapter; "entregado" solo tras emit exitoso
            try:
                await self.emit("tts_chunk", {"stream_id": sid, "pcm": ev.pcm, "sample_rate": ev.sample_rate, "seg_id": cid,
                                              "planned_text_en": ev.planned_text_en})
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                u.emit_failed += 1
                self.stats["tts_emit_failures"] += 1
                await self._report(sid, f"tts_chunk no entregado ({cid}): {type(exc).__name__}: {exc}")
                return
            u.state = "delivering"
            u.text_en = u.text_en or ev.planned_text_en or ""
            u.chunks += 1
            u.samples += len(ev.pcm) // 2
        elif k == "unit_done":
            u = self._unit(sid, cid)
            if u.state in TERMINAL_STATES:
                return  # cancelada, ya cerrada o duplicado: no repite mt_final
            claimed = ev.unit_n_chunks
            partial = bool(getattr(ev, "unit_reason", None))
            if u.chunks <= 0:  # los claims terminales no sustituyen PCM observado
                self._finish(u, "delivered")  # -> discarded + contador
                await self._report(sid, f"unit_done de {cid} sin PCM observado: no se persiste ni se muestra")
                return
            reason = getattr(ev, "unit_reason", None)
            if claimed is not None and claimed != u.chunks:
                self.stats["claim_mismatch"] += 1
                reason = f"{reason + ';' if reason else ''}claim_mismatch(claimed={claimed},observed={u.chunks})"
            if u.emit_failed:
                partial = True
                reason = f"{reason + ';' if reason else ''}tts_emit_failed({u.emit_failed} de {u.recv_chunks})"
            self._finish(u, "delivered_partial" if partial else "delivered", reason)
            await self.emit("mt_final", {"stream_id": sid, "text": u.text_en, "seg_id": cid, "mt_ms": None})
            await self.emit("inc_unit", {"stream_id": sid, "unit_id": cid, "state": u.state, "playback": "not_verified",
                                         "text_en": u.text_en, "n_chunks": u.chunks, "reason": u.reason})
        elif k == "unit_cancelled":
            u = self._unit(sid, cid)
            if u.state in PERSISTED_STATES or u.state in ("cancelled", "discarded"):
                return  # ya entregada: no se reescribe
            u.state = "cancelled"
            u.reason = getattr(ev, "unit_reason", None)
            self.stats["units_cancelled"] += 1
        elif k == "error":
            self.stats["errors"] += 1
            await self.emit("error", {"stream_id": sid, "error_kind": ev.error_kind or "error del motor incremental", "seg_id": cid})


def make_engine_factory(mode: IncrementalModeConfig, adapter_factory: Callable[[], Any], store: UnitStore | None = None,
                        store_factory: Callable[[], UnitStore] | None = None):
    """factory(cfg, emit) para ``RuntimeService(engine_factory=...)``. ``cfg`` (PipelineConfig) se ignora.
    ``store`` externo: lo cierra quien lo creo. ``store_factory``: cada puente crea su store y lo cierra en close()."""

    def factory(cfg, emit):
        if store_factory is not None:
            return IncrementalEngineBridge(mode, adapter_factory(), emit, store_factory(), owns_store=True)
        return IncrementalEngineBridge(mode, adapter_factory(), emit, store, owns_store=False)

    return factory
