"""Entrada y salida de audio (PortAudio vía sounddevice).

* El micrófono solo se abre cuando ``MicCapture.start()`` se llama por una
  acción explícita del usuario (POST /api/start).
* No se guarda audio: los bloques pasan por memoria hacia el motor y se
  descartan. Ningún método escribe PCM a disco.
* Los callbacks de PortAudio no asignan memoria pesada, no toman locks largos
  ni llaman a SQLite.
"""
from __future__ import annotations

import collections
import inspect
import math
import re
import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

VIRTUAL_HINTS = ("blackhole", "loopback", "virtual", "soundflower", "vb-cable", "aggregate")
TARGET_SR = 16000
EDGE_FADE_MS = 2.0    # fundido de entrada y salida en los bordes del audio (evita clics)
CLEAR_FADE_MS = 8.0   # fundido de salida al cortar o interrumpir lo que suena

# ---- Refresco de la lista de dispositivos. PortAudio (CoreAudio) enumera dispositivos SOLO en Pa_Initialize: un auricular conectado
# después no aparece hasta reinicializarlo. sounddevice no expone API pública para eso; se usan sus funciones PRIVADAS _terminate /
# _initialize, encapsuladas aquí, con versión verificada, bloqueo y recuperación.
SUPPORTED_SD_VERSIONS = ("0.5.6",)  # versiones cuyo _initialize/_terminate/_initialized se revisaron
REFRESH_INIT_RETRIES = 3


class AudioBusy(RuntimeError):
    """Hay streams de audio abiertos: PortAudio no se puede reiniciar."""


class AudioRefreshError(RuntimeError):
    """No se pudo refrescar PortAudio (versión no verificada, referencias ajenas, terminate/initialize fallido)."""


class StaleDevice(RuntimeError):
    """El dispositivo se resolvió antes de un refresco: su índice ya no es válido. Hay que volver a resolverlo."""


_pa_lock = threading.RLock()  # serializa enumeración, apertura de streams y refresco de PortAudio
_open_streams: set = set()  # streams creados por ESTE módulo y NO cerrados con confirmación (sounddevice 0.5 no cuenta streams)
_unsafe_streams: set = set()  # subconjunto cuyo close() falló: su estado es incierto; el refresco se rechaza hasta reiniciar
_generation = 0  # +1 en cada refresco; los Device guardan la generación con la que se resolvieron
_broken = False  # un refresco dejó PortAudio sin inicializar (se reintenta en el próximo)


def open_stream_count() -> int:
    with _pa_lock:
        return len(_open_streams)


def generation() -> int:
    return _generation


def unsafe_stream_count() -> int:
    with _pa_lock:
        return len(_unsafe_streams)


def _register(stream) -> None:
    with _pa_lock:
        _open_streams.add(stream)


def _close_kwargs(stream) -> dict:
    """sounddevice.Stream.close(ignore_errors=True) OCULTA por defecto los errores de Pa_CloseStream (y anula el puntero igual): para poder CONFIRMAR
    el cierre se pide ignore_errors=False. Dobles de prueba sin ese parámetro se cierran sin argumentos."""
    try:
        return {"ignore_errors": False} if "ignore_errors" in inspect.signature(stream.close).parameters else {}
    except (TypeError, ValueError):
        return {}


def _close_confirmed(stream) -> None:
    """Cierra y SOLO si close() termina sin error saca el stream del registro. Si falla, queda registrado y marcado inseguro (el estado real
    del stream es incierto) y se relanza el error. No se llama con _pa_lock tomado salvo en la ruta de fallo de apertura; los callbacks de audio
    nunca toman _pa_lock, así que Pa_CloseStream (que espera al callback) no puede interbloquearse con él."""
    try:
        stream.close(**_close_kwargs(stream))
    except BaseException:
        with _pa_lock:
            _unsafe_streams.add(stream)
        raise
    with _pa_lock:
        _open_streams.discard(stream)
        _unsafe_streams.discard(stream)


def _check_fresh(dev) -> None:
    g = getattr(dev, "gen", None)
    if g is not None and g != _generation:
        raise StaleDevice("Los dispositivos de audio se actualizaron: volvé a elegirlos e iniciá de nuevo.")


@dataclass(frozen=True)
class Device:
    id: str
    name: str
    index: int
    max_in: int
    max_out: int
    default_sr: float
    default_in: bool
    default_out: bool
    gen: int = 0  # generación de PortAudio con la que se resolvió (ver refresh_devices)
    ambiguous: bool = False  # otro dispositivo comparte nombre Y dirección: su id nombre#índice no es estable tras reenumerar

    @property
    def virtual(self) -> bool:
        n = self.name.lower()
        return any(h in n for h in VIRTUAL_HINTS)


def _sd():
    import sounddevice as sd

    return sd


def _dir_class(d) -> str:
    i, o = int(d["max_input_channels"]) > 0, int(d["max_output_channels"]) > 0
    return "io" if i and o else "in" if i else "out" if o else "none"


def list_devices() -> list[Device]:
    """Dispositivos con id estable (nombre; si se repite, nombre#índice)."""
    sd = _sd()
    with _pa_lock:  # nunca se enumera mientras refresh_devices() tiene PortAudio terminado
        if _broken:
            return []
        try:
            raw = sd.query_devices()
            din, dout = sd.default.device
        except Exception:
            return []
        gen = _generation
    names = collections.Counter(d["name"] for d in raw)
    dirs = collections.Counter((d["name"], _dir_class(d)) for d in raw)
    out = []
    for i, d in enumerate(raw):
        cls = _dir_class(d)
        if names[d["name"]] == 1:
            dev_id, amb = d["name"], False
        elif dirs[(d["name"], cls)] == 1:  # homónimos que difieren en dirección (p. ej. Sony: entrada / salida): id estable
            dev_id, amb = f"{d['name']}#{cls}", False
        else:  # mismo nombre Y dirección: solo se distinguen por índice, que cambia al reenumerar
            dev_id, amb = f"{d['name']}#{i}", True
        out.append(
            Device(dev_id, d["name"], i, int(d["max_input_channels"]), int(d["max_output_channels"]),
                   float(d["default_samplerate"]), i == din, i == dout, gen, amb)
        )
    return out


def devices_payload() -> dict:
    devs = list_devices()
    return {
        "generation": _generation,
        "audio_ok": not _broken,
        "inputs": [
            {"id": d.id, "name": d.name, "default": d.default_in, "virtual": d.virtual, "ambiguous": d.ambiguous}
            for d in devs if d.max_in > 0
        ],
        "outputs": [
            {"id": d.id, "name": d.name, "default": d.default_out, "virtual": d.virtual, "ambiguous": d.ambiguous}
            for d in devs if d.max_out > 0
        ],
    }


def find_device(dev_id: str, kind: str) -> Device:
    """Resuelve un id guardado de forma ESTRICTA: solo coincide el id exacto. Un id ``nombre#índice`` cuya numeración cambió tras un refresco NO se
    reasigna a otro dispositivo (aunque el nombre coincida): LookupError y el usuario elige de nuevo. Ver ``device_candidates`` (solo informativo)."""
    for d in list_devices():
        if d.id == dev_id and (d.max_in if kind == "in" else d.max_out) > 0:
            return d
    raise LookupError(f"Dispositivo no encontrado: {dev_id}")


def device_candidates(dev_id: str, kind: str) -> list[str]:
    """Ids de dispositivos con el mismo NOMBRE y dirección que un id guardado que ya no existe. Solo para MOSTRAR al usuario: nada se asigna solo."""
    m = re.match(r"^(.*)#\d+$", dev_id)
    base = m.group(1) if m else dev_id
    return sorted(d.id for d in list_devices() if d.name == base and (d.max_in if kind == "in" else d.max_out) > 0 and d.id != dev_id)


def device_is_ambiguous(dev_id: str) -> bool:
    """True si ``dev_id`` es un id ``nombre#índice`` de un dispositivo que comparte nombre Y dirección con otro (no hay UID para distinguirlos tras reordenar)."""
    return any(d.id == dev_id and d.ambiguous for d in list_devices())


def _device_names(sd) -> list:
    return [(d["name"], int(d["max_input_channels"]), int(d["max_output_channels"])) for d in sd.query_devices()]


def refresh_devices() -> dict:
    """Reinicializa PortAudio para que vea dispositivos conectados/desconectados después del arranque.

    Seguridad: exige 0 streams abiertos (registro propio, porque sounddevice no los cuenta) y que PortAudio tenga exactamente la
    referencia del propio sounddevice (``sd._initialized == 1``); toma el mismo lock que la enumeración y la apertura de streams;
    reintenta ``_initialize`` si falla y, si no se recupera, deja ``_broken`` (lista vacía) y se reintenta en el siguiente refresco.
    Solo para versiones de sounddevice verificadas. El llamador (servicio) además lo prohíbe con sesión en curso."""
    global _generation, _broken
    sd = _sd()
    ver = getattr(sd, "__version__", None)
    if ver not in SUPPORTED_SD_VERSIONS or not all(hasattr(sd, n) for n in ("_terminate", "_initialize", "_initialized")):
        raise AudioRefreshError(f"sounddevice {ver}: refresco no verificado para esta versión")
    with _pa_lock:
        if _unsafe_streams:
            raise AudioBusy(f"hay {len(_unsafe_streams)} stream(s) cuyo cierre falló (estado incierto): reiniciá la app para actualizar dispositivos")
        if _open_streams:
            raise AudioBusy(f"hay {len(_open_streams)} stream(s) de audio abiertos")
        if not _broken and sd._initialized != 1:
            raise AudioRefreshError(f"PortAudio tiene {sd._initialized} referencias (se esperaba 1): otro componente lo usa")
        before = [] if _broken else _device_names(sd)
        if not _broken:
            try:
                sd._terminate()
            except Exception as exc:  # noqa: BLE001 - sounddevice no decrementa si Pa_Terminate falla: el estado no cambió
                raise AudioRefreshError(f"Pa_Terminate falló; la lista no cambió: {exc}") from exc
        last = None
        for attempt in range(REFRESH_INIT_RETRIES):
            try:
                sd._initialize()
                break
            except Exception as exc:  # noqa: BLE001
                last = exc
                time.sleep(0.2 * (attempt + 1))
        else:
            _broken = True
            raise AudioRefreshError(f"PortAudio no pudo reinicializarse tras {REFRESH_INIT_RETRIES} intentos: {last}")
        _broken = False
        _generation += 1
        after = _device_names(sd)
        added = collections.Counter(after) - collections.Counter(before)
        removed = collections.Counter(before) - collections.Counter(after)
        return {"generation": _generation, "devices_before": len(before), "devices_after": len(after),
                "added": sorted(n for n, _i, _o in added.elements()), "removed": sorted(n for n, _i, _o in removed.elements())}


def resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to or x.size == 0:
        return x
    from scipy.signal import resample_poly

    g = math.gcd(int(sr_from), int(sr_to))
    return resample_poly(x, int(sr_to) // g, int(sr_from) // g).astype(np.float32)


LATENCY_MODES = ("default", "low", "high")


def latency_arg(mode: str) -> str | None:
    """Traduce el modo a ``latency`` de sounddevice. ``default`` no cambia nada: sounddevice usa
    su valor por defecto (``'high'`` para entrada y salida, según ``sd.default.latency``)."""
    if mode not in LATENCY_MODES:
        raise ValueError(f"modo de latencia desconocido: {mode!r} (usar {', '.join(LATENCY_MODES)})")
    return None if mode == "default" else mode


def _actual(stream, idx: int | None):
    """Latencia real que informa el stream abierto (segundos) y su blocksize."""
    lat = getattr(stream, "latency", None)
    if isinstance(lat, tuple):
        lat = lat[idx] if idx is not None and idx < len(lat) else None
    return (float(lat) if lat is not None else None), getattr(stream, "blocksize", None)


def prewarm_resample() -> None:
    """Carga scipy antes de abrir streams, para que el callback nunca importe nada."""
    import scipy.signal  # noqa: F401


class MicCapture:
    """Captura mono a 16 kHz int16. ``on_block(pcm_bytes)`` se llama desde el
    hilo de PortAudio y solo debe encolar y volver."""

    def __init__(self, dev: Device, on_block: Callable[[bytes], None],
                 on_status: Callable[[str], None] | None = None, latency: str | None = None) -> None:
        self.dev = dev
        self.on_block = on_block
        self.on_status = on_status
        self.latency = latency  # None = por defecto de sounddevice; 'low' | 'high'
        self.latency_s: float | None = None  # real, tras abrir el stream
        self.blocksize: int | None = None
        self._stream = None
        self.sr = TARGET_SR

    def start(self) -> None:
        sd = _sd()
        last = None
        for sr in (TARGET_SR, int(self.dev.default_sr)):
            if sr != TARGET_SR:
                prewarm_resample()
            stream = None
            try:
                # La tasa real se fija ANTES de arrancar: el callback puede llegar ya.
                self.sr = sr
                with _pa_lock:  # sin refresco de PortAudio entre validar el índice, abrir y registrar
                    _check_fresh(self.dev)
                    stream = sd.InputStream(
                        device=self.dev.index, channels=1, samplerate=sr, dtype="float32",
                        blocksize=int(sr * 0.02), callback=self._cb, latency=self.latency,
                    )
                    _register(stream)  # registrado DESDE la construcción: hasta un cierre confirmado cuenta como abierto
                    stream.start()
                self._stream = stream
                self.latency_s, self.blocksize = _actual(stream, 0)
                return
            except StaleDevice:
                self.sr = TARGET_SR
                self._stream = None
                raise
            except Exception as exc:  # noqa: BLE001
                last = exc
                self._stream = None
                if stream is not None:
                    try:
                        _close_confirmed(stream)  # un stream que no arrancó también debe cerrarse; si el cierre falla queda marcado inseguro
                    except Exception:  # noqa: BLE001
                        pass
        self.sr = TARGET_SR
        raise RuntimeError(f"No se pudo abrir el micrófono: {last}")

    def _cb(self, indata, frames, t, status) -> None:
        if status and self.on_status:
            self.on_status(str(status))
        x = indata[:, 0]
        if self.sr != TARGET_SR:
            x = resample(x, self.sr, TARGET_SR)
        self.on_block((np.clip(x, -1, 1) * 32767.0).astype("<i2").tobytes())

    def stop(self) -> None:
        s, self._stream = self._stream, None
        if s is not None:
            try:
                s.stop()
            finally:
                _close_confirmed(s)


class Player:
    """Reproduce PCM mono (cualquier tasa) en un dispositivo de salida.

    Cola acotada en segundos: ``write`` descarta lo MÁS NUEVO si se excede
    ``max_queue_s`` y lo cuenta (``dropped_s``), nunca corta lo ya encolado.
    """

    def __init__(self, dev: Device, volume: float = 1.0, max_queue_s: float = 60.0,
                 latency: str | None = None) -> None:
        self.dev = dev
        self.volume = float(volume)
        self.max_queue_s = max_queue_s
        self.latency = latency  # None = por defecto de sounddevice; 'low' | 'high'
        self.latency_s: float | None = None  # real, tras abrir el stream
        self.blocksize: int | None = None
        self._buf: collections.deque[np.ndarray] = collections.deque()
        self._lock = threading.Lock()
        self._queued = 0  # muestras a la tasa del dispositivo
        self._cur: np.ndarray | None = None
        self._tail = np.zeros(0, dtype=np.float32)  # cola ya atenuada tras clear(): se reproduce antes que nada
        self._was_playing = False  # el bloque anterior salió lleno de audio (sin silencio en el medio)
        self._stream = None
        self.sr = int(dev.default_sr) or 48000
        self.dropped_s = 0.0
        # Bloques de callback rellenados solo en parte. Incluye el FINAL NORMAL de cada fragmento
        # (la cola se vacía al terminar de hablar). No es un underrun físico: los cortes reales de
        # reproducción aparecen en status_events (banderas de PortAudio).
        self.partial_blocks = 0
        self.status_events = 0  # banderas de PortAudio (underflow/overflow) ya no se ignoran
        self.last_status: str | None = None
        self._ch = 1

    def start(self) -> None:
        sd = _sd()
        self._ch = 2 if self.dev.max_out >= 2 else 1
        with _pa_lock:  # sin refresco de PortAudio entre validar el índice, abrir y registrar
            _check_fresh(self.dev)
            stream = sd.OutputStream(
                device=self.dev.index, channels=self._ch, samplerate=self.sr, dtype="float32",
                callback=self._cb, latency=self.latency,
            )
            _register(stream)  # registrado DESDE la construcción: hasta un cierre confirmado cuenta como abierto
            try:
                stream.start()
            except Exception:
                try:
                    _close_confirmed(stream)
                except Exception:  # noqa: BLE001
                    pass
                raise
        self._stream = stream
        self.latency_s, self.blocksize = _actual(stream, 1)

    @property
    def underruns(self) -> int:
        """Alias obsoleto de ``partial_blocks``. No es un conteo de cortes de reproducción."""
        return self.partial_blocks

    @property
    def dropped_ms(self) -> float:
        return self.dropped_s * 1000.0

    @property
    def queued_s(self) -> float:
        return self._queued / self.sr

    def set_volume(self, v: float) -> None:
        self.volume = max(0.0, min(1.0, float(v)))

    def write(self, pcm_s16: bytes, sample_rate: int) -> None:
        x = np.frombuffer(pcm_s16, dtype="<i2").astype(np.float32) / 32768.0
        x = resample(x, sample_rate, self.sr)
        with self._lock:
            if (self._queued + x.size) / self.sr > self.max_queue_s:
                self.dropped_s += x.size / self.sr
                return
            self._buf.append(x)
            self._queued += x.size

    def _cb(self, outdata, frames, t, status) -> None:
        if status:
            self.status_events += 1
            self.last_status = str(status)
        out = np.zeros(frames, dtype=np.float32)
        filled = 0
        from_tail = False
        with self._lock:
            if self._tail.size:  # final atenuado de un clear(): sale primero y sin fundido de entrada
                take = min(frames, self._tail.size)
                out[:take] = self._tail[:take]
                self._tail = self._tail[take:]
                filled = take
                from_tail = True
            while filled < frames:
                if self._cur is None or self._cur.size == 0:
                    if not self._buf:
                        break
                    self._cur = self._buf.popleft()
                take = min(frames - filled, self._cur.size)
                out[filled:filled + take] = self._cur[:take]
                self._cur = self._cur[take:]
                self._queued -= take
                filled += take
        if filled < frames and filled > 0:
            self.partial_blocks += 1
        # Bordes sin clics: fundido de entrada (2 ms) si se sale del silencio y de salida si la cola se seca
        # a mitad de frase. Entre fragmentos contiguos no se toca nada.
        k = max(2, int(self.sr * EDGE_FADE_MS / 1000))
        if filled > 0 and not self._was_playing and not from_tail:
            n = min(k, filled)
            out[:n] *= np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float32)
        if 0 < filled < frames:
            n = min(k, filled)
            out[filled - n:filled] *= np.linspace(1.0, 0.0, n, endpoint=True, dtype=np.float32)
        self._was_playing = filled == frames and not (from_tail and not self._tail.size)
        out *= self.volume
        outdata[:] = out.reshape(-1, 1) if self._ch == 1 else np.repeat(out.reshape(-1, 1), self._ch, axis=1)

    def clear(self) -> None:
        """Descarta lo encolado sin cortar la onda: los próximos ``CLEAR_FADE_MS`` salen atenuados hasta cero."""
        with self._lock:
            n = max(2, int(self.sr * CLEAR_FADE_MS / 1000))
            parts: list[np.ndarray] = []
            got = 0
            for b in ([self._cur] if self._cur is not None and self._cur.size else []) + list(self._buf):
                parts.append(b[: n - got])
                got += parts[-1].size
                if got >= n:
                    break
            if parts:
                tail = np.concatenate(parts).astype(np.float32, copy=True)
                tail *= np.linspace(1.0, 0.0, tail.size, endpoint=True, dtype=np.float32)
                self._tail = tail
            self._buf.clear()
            self._cur = None
            self._queued = 0

    def stop(self) -> None:
        s, self._stream = self._stream, None
        if s is not None:
            try:
                # Si todavía suena algo, se atenúa en vez de cortar (sin clic) y se espera a que el fundido salga.
                if self._queued > 0 or self._tail.size or self._cur is not None:
                    self.clear()
                    time.sleep(CLEAR_FADE_MS / 1000 + (self.latency_s or 0.05) + 0.02)
                s.stop()
            finally:
                _close_confirmed(s)
