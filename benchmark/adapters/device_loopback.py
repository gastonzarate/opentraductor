"""Envoltorio de adaptador STREAMING que reproduce el TTS en BlackHole y mide la captura.

Reutiliza el arnés existente (replay causal, colector, métricas): el candidato interno
(``inner``) no cambia y sus eventos pasan intactos. Este envoltorio, con el reloj del arnés:
  1. abre SOLO BlackHole (captura y ``runtime.audio_io.Player``), verifica silencio previo;
  2. entre clips exige silencio de captura y cola vacía (un clip a la vez);
  3. al llegar cada ``tts_chunk`` lo escribe al Player y anota el instante;
  4. al cerrar escribe ``report_path`` (JSON: solo métricas; no se guarda audio).

Comando exacto: ver ``scripts/eval/device_loopback.py``. No ejecutar sin lease de cómputo.
"""
from __future__ import annotations

import asyncio
import importlib
import json
from pathlib import Path
from typing import Any, AsyncIterator

from .. import device_loopback as DL
from ..contracts import Adapter, AdapterEvent, AdapterMode, EventKind

DEFAULTS: dict[str, Any] = {
    "inner": "runtime.adapters.baseline:BaselineAdapter",
    "inner_config": {},
    "pre_silence_s": 1.0,
    "threshold_dbfs": DL.DEFAULT_THRESHOLD_DBFS,
    "settle_s": 0.3,
    "drain_timeout_s": 20.0,
    "volume": 1.0,
    "player_to_capture_settle_s": 0.3,
    "player_latency": None,  # None (defecto de sounddevice) | "low" | "high": solo el Player; la captura NO cambia entre brazos
    "report_path": None,
}


def _load(spec: str):
    mod, _, cls = spec.partition(":")
    return getattr(importlib.import_module(mod), cls)


class DeviceLoopbackAdapter(Adapter):
    name = "device-loopback(blackhole)"
    version = "0.1"
    mode = AdapterMode.STREAMING

    def __init__(self) -> None:
        self.cfg: dict[str, Any] = {}
        self.inner: Adapter | None = None
        self.capture = None
        self.player = None
        self.rec: DL.Recorder | None = None
        self._closed = False
        self._pump_idle = True  # el consumidor de events() espera el siguiente evento del candidato
        self.late_events = 0

    # ---- fábricas sobrescribibles en pruebas ----
    def _make_inner(self, spec: str) -> Adapter:
        return _load(spec)()

    def _make_capture(self):
        return DL.BlackHoleCapture()

    def _make_player(self, dev, volume: float, latency=None):
        from runtime import audio_io

        return audio_io.Player(dev, volume=volume, latency=latency)

    # ---- ciclo de vida ----
    async def open(self, config: dict[str, Any]) -> None:
        self.cfg = {**DEFAULTS, **config}
        if not self.cfg["report_path"]:
            raise ValueError("report_path es obligatorio")
        self.inner = self._make_inner(self.cfg["inner"])
        self.mode = self.inner.mode
        self.produces_audio = self.inner.produces_audio
        try:
            await self.inner.open(self.cfg["inner_config"])  # modelos primero: calibramos justo antes de medir
            self.capture = self._make_capture()
            dev = self.capture.pick_device()
            # ORDEN OBLIGATORIO: Player primero, captura después. Con la captura ya corriendo, arrancar el
            # OutputStream de BlackHole da AUHAL -10863 y la captura muere en silencio (medido en este equipo).
            self.player = self._make_player(dev, self.cfg["volume"], self.cfg["player_latency"])
            self.player.start()
            # Medido en este equipo: capture.start() <~10 ms después de player.start() deja la captura muerta tras ~17 bloques
            # (AUHAL -50/-10863); con >=0.2 s de separación entrega 100 bloques/s. Se exige el hueco.
            await asyncio.sleep(self.cfg["player_to_capture_settle_s"])
            self.capture.start()
            self.rec = DL.Recorder(self.capture, threshold_dbfs=self.cfg["threshold_dbfs"], settle_s=self.cfg["settle_s"])
            await self.rec.calibrate(self.cfg["pre_silence_s"])
        except BaseException:
            await self._cleanup()
            try:
                await self.inner.close()
            except Exception:  # noqa: BLE001
                pass
            raise

    async def start_stream(self, stream_id: str, seed: int) -> None:
        await self.rec.wait_quiet(lambda: self.player.queued_s, self.cfg["drain_timeout_s"])
        self.rec.begin(stream_id)
        await self.inner.start_stream(stream_id, seed)

    async def push_audio(self, stream_id, pcm, t_audio_start_s) -> None:
        await self.inner.push_audio(stream_id, pcm, t_audio_start_s)

    async def end_stream(self, stream_id: str) -> None:
        await self.inner.end_stream(stream_id)
        try:
            # Barrera de entrega: el candidato pudo volver con eventos aún en su cola. Hasta que el
            # consumidor de events() esté ocioso (ya escribió al Player y anotó), el silencio de
            # captura/cola vacía no significa «terminó». Los chunks posteriores se rechazan y se cuentan.
            await self._delivery_barrier()
            await self.rec.wait_quiet(lambda: self.player.queued_s, self.cfg["drain_timeout_s"])
        finally:
            self.rec.end(stream_id)
        if self.rec.foreign_since_last_clip():
            raise DL.ContaminationError("audio fuera de las ventanas de reproducción detectado en BlackHole")

    async def _delivery_barrier(self) -> None:
        for _ in range(3):  # deja correr los despertares ya programados (put del candidato → get del consumidor)
            await asyncio.sleep(0)
        t_end = DL.MONO_NS() + round(self.cfg["drain_timeout_s"] * 1e9)
        while not self._pump_idle:
            if DL.MONO_NS() > t_end:
                raise DL.DeviceLoopbackError("barrera de entrega: el consumidor de eventos no quedó ocioso a tiempo")
            await asyncio.sleep(0.005)

    async def events(self) -> AsyncIterator[AdapterEvent]:
        it = self.inner.events().__aiter__()
        while True:
            self._pump_idle = True
            try:
                ev = await it.__anext__()
            except StopAsyncIteration:
                return
            self._pump_idle = False
            if ev.kind == EventKind.TTS_CHUNK and ev.pcm:
                try:
                    self.rec.write(ev.stream_id, ev.pcm, ev.sample_rate or 24000, ev.commit_id)
                    self.player.write(ev.pcm, ev.sample_rate or 24000)
                except DL.DeviceLoopbackError as exc:
                    self.late_events += 1
                    yield AdapterEvent(EventKind.ERROR, ev.stream_id, error_kind=f"device_loopback: {exc}")
            yield ev

    async def _cleanup(self) -> None:
        for fn in (getattr(self.player, "stop", None), getattr(self.capture, "stop", None)):
            if fn:
                try:
                    fn()
                except Exception:  # noqa: BLE001
                    pass

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        rep = None
        try:
            if self.rec is not None:
                rep = self.rec.report()
                pb = getattr(self.player, "partial_blocks", getattr(self.player, "underruns", None))
                player = {"sample_rate": getattr(self.player, "sr", None),
                          "underruns": pb,  # = bloques parcialmente rellenados (incluye el vaciado normal); nombre histórico
                          "requested_latency_mode": self.cfg["player_latency"] or "default",
                          "reported_stream_latency_s": getattr(self.player, "latency_s", None),
                          "reported_stream_latency_note": "valor que declara PortAudio al abrir el stream; NO es una medición física",
                          "blocksize": getattr(self.player, "blocksize", None),
                          "dropped_s": getattr(self.player, "dropped_s", None),
                          "status_events": getattr(self.player, "status_events", None)}
                rep.update(
                    validity=DL.validity(getattr(self.capture, "status_events", None), player, self.late_events,
                                         max_block_gap_ms=DL.max_block_gap_ms(self.capture.blocks),
                                         block_ms=self.capture.blocksize_ms),
                    late_events=self.late_events, player_last_status=getattr(self.player, "last_status", None),
                    uncertainty_note=("incertidumbre = bloque + jitter p95 de callbacks: NO cubre la calibración del reloj de "
                                      "PortAudio, ni la latencia de salida del Player/BlackHole, ni la incertidumbre del límite anotado"),
                    unambiguous_note="«inequívoco» = asignación temporal interna; no autentica el contenido de audio externo",
                )
                rep.update(
                    schema="opentraductor-device-loopback/v1",
                    scope=("pipeline → Player → BlackHole → captura del arnés. NO es audio escuchado en Meet/Discord, "
                           "NO usa micrófono físico ni acústica. Incluye latencia de bloque/callback de la captura."),
                    clock="time.monotonic_ns (mismo reloj que RealClock) leído en el callback de captura",
                    device=getattr(self.capture.dev, "name", None) if self.capture else None,
                    calibration=self.rec.calibration, threshold_dbfs=self.cfg["threshold_dbfs"],
                    player=player,
                    capture_status_events=getattr(self.capture, "status_events", None),
                    audio_saved=False, physical_microphone_opened=False,
                )
        finally:
            await self._cleanup()
            if self.inner is not None:
                await self.inner.close()
        if rep is not None:
            Path(self.cfg["report_path"]).parent.mkdir(parents=True, exist_ok=True)
            Path(self.cfg["report_path"]).write_text(json.dumps(DL.json_safe(rep), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")

    def describe(self) -> dict[str, Any]:
        d = {"device_loopback": {k: v for k, v in self.cfg.items() if k != "inner_config"},
             "inner": self.inner.describe() if self.inner else None}
        return d
