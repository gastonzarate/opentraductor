"""Medición de la salida a un dispositivo VIRTUAL (BlackHole) hecha por el arnés.

Qué mide: tiempo desde que el arnés entrega PCM de TTS al ``Player`` real hasta que
aparece audio > umbral en la captura de BlackHole (Player → CoreAudio → BlackHole →
captura). Con el origen anotado de cada clip se obtiene fin-de-frase → primer audio en
BlackHole. NO es audio escuchado en Meet/Discord, NO incluye micrófono físico ni
acústica, y la captura añade su propia latencia de bloque/callback (se publica como
incertidumbre).

Reglas:
* Solo se abre un dispositivo cuyo nombre contenga «blackhole» (entrada y salida). Nunca el
  dispositivo por defecto, nunca un micrófono físico ni altavoces.
* Reloj: ``time.monotonic_ns`` (el mismo que ``RealClock`` del arnés), leído DENTRO del callback
  de captura. Ningún tiempo lo declara el candidato; el evento ``device_output`` del candidato
  sigue sin existir: este informe es independiente (``device_loopback.json``).
* Silencio previo obligatorio: si algún bloque supera el umbral antes de reproducir, se aborta
  (``ContaminationError``). Entre clips se exige silencio de captura y cola de reproducción vacía
  antes de empezar el siguiente; audio fuera de las ventanas de un clip aborta.
* Asociación solo inequívoca: un clip a la vez; el primer audio > umbral cuenta para un clip solo
  si aparece tras su primera escritura audible al Player (menos la incertidumbre). Audio previo ⇒
  ``contaminated_before_write``; sin audio capturado ⇒ ``no_device_onset``.
* No se graba nada: de cada bloque solo se guardan instante, RMS y latencia de callback.
"""
from __future__ import annotations

import asyncio
import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from . import audio as A

MONO_NS: Callable[[], int] = time.monotonic_ns  # mismo reloj que benchmark.clock.RealClock
DEFAULT_THRESHOLD_DBFS = A.DEFAULT_SILENCE_THRESHOLD_DBFS
MS_NS = 1_000_000
HINT = "blackhole"


class DeviceLoopbackError(RuntimeError):
    pass


class ContaminationError(DeviceLoopbackError):
    """Audio ajeno detectado en la captura: la medición no es válida."""


@dataclass
class Block:
    t_cb_ns: int  # instante del callback (reloj del arnés)
    start_est_ns: int  # inicio estimado del bloque: t_cb − (currentTime − inputBufferAdcTime)
    dur_ns: int
    rms_dbfs: float
    in_lat_ns: int  # currentTime − inputBufferAdcTime del callback


def rms_dbfs_f32(x: np.ndarray) -> float:
    if x.size == 0:
        return float("-inf")
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    return 20 * float(np.log10(rms)) if rms > 0 else float("-inf")


def nonsilent_seconds(pcm: bytes, sr: int, threshold_dbfs: float, frame_ms: float = 10.0) -> float:
    """Duración con bloques de 10 ms sobre el umbral (mismo criterio que el detector del arnés)."""
    x = A.to_float(pcm)
    n = max(1, round(sr * frame_ms / 1000))
    cnt = 0
    for s in range(0, x.size, n):
        f = x[s : s + n]
        r = float(np.sqrt(np.mean(f * f)))
        if r > 0 and 20 * np.log10(r / 32768.0) > threshold_dbfs:
            cnt += 1
    return cnt * n / sr


# ------------------------------------------------------------------ captura
class BlackHoleCapture:
    """Captura de BlackHole (solo métricas por bloque). Inyectable para pruebas."""

    def __init__(self, *, blocksize_ms: float = 10.0, sd_module=None, devices: list | None = None,
                 clock: Callable[[], int] = MONO_NS) -> None:
        self.blocksize_ms = blocksize_ms
        self._sd = sd_module
        self._devices = devices
        self._clock = clock
        self.blocks: list[Block] = []
        self.status_events = 0
        self.dev = None
        self.sr = 0
        self.channels = 1
        self._stream = None

    def pick_device(self):
        """Dispositivo BlackHole (entrada y salida). Falla si no existe: nunca cae a otro."""
        if self._devices is None:
            from runtime import audio_io

            self._devices = audio_io.list_devices()
        cands = [d for d in self._devices if HINT in d.name.lower() and d.max_in > 0 and d.max_out > 0]
        if not cands:
            raise DeviceLoopbackError("no hay dispositivo BlackHole con entrada y salida; no se usa ningún otro")
        if len(cands) > 1:
            raise DeviceLoopbackError(f"más de un BlackHole ({[d.name for d in cands]}): indicar uno explícito")
        self.dev = cands[0]
        return self.dev

    def start(self) -> None:
        dev = self.dev or self.pick_device()
        if HINT not in dev.name.lower():
            raise DeviceLoopbackError(f"dispositivo no virtual: {dev.name}")
        sd = self._sd
        if sd is None:
            import sounddevice as sd  # noqa: PLW0127
        self.sr = int(dev.default_sr)
        self.channels = 2 if dev.max_in >= 2 else 1
        frames = max(1, round(self.sr * self.blocksize_ms / 1000))
        stream = sd.InputStream(device=dev.index, channels=self.channels, samplerate=self.sr, dtype="float32",
                                blocksize=frames, callback=self._cb)
        try:
            stream.start()
        except Exception:
            try:
                stream.close()
            except Exception:  # noqa: BLE001
                pass
            raise
        self._stream = stream

    def _cb(self, indata, frames, t, status) -> None:
        now = self._clock()
        if status:
            self.status_events += 1
        lat = int(round((float(t.currentTime) - float(t.inputBufferAdcTime)) * 1e9))
        rms = max(rms_dbfs_f32(indata[:, c]) for c in range(indata.shape[1]))
        dur = int(round(frames / self.sr * 1e9))
        self.blocks.append(Block(now, now - lat, dur, rms, lat))

    def stop(self) -> None:
        s, self._stream = self._stream, None
        if s is not None:
            try:
                s.stop()
            finally:
                s.close()


# ------------------------------------------------------------------ análisis puro
@dataclass
class Write:
    t_ns: int
    samples: int
    sample_rate: int
    nonsilent_offset_s: float | None
    nonsilent_s: float
    commit_id: str | None


@dataclass
class Clip:
    stream_id: str
    begin_ns: int
    end_ns: int | None = None
    writes: list[Write] = field(default_factory=list)


def first_audible_write(clip: Clip) -> Write | None:
    return next((w for w in clip.writes if w.nonsilent_offset_s is not None), None)


def uncertainty_ms(blocks: list[Block], block_ms: float) -> dict[str, float]:
    """Incertidumbre de la medición de captura: un bloque + jitter de callback (p95) + latencia de entrada."""
    gaps = [abs((b.start_est_ns - a.start_est_ns) - a.dur_ns) / 1e6 for a, b in zip(blocks, blocks[1:])]
    jit = sorted(gaps)[int(0.95 * (len(gaps) - 1))] if gaps else 0.0
    lat = statistics.median([b.in_lat_ns for b in blocks]) / 1e6 if blocks else 0.0
    return {"block_ms": block_ms, "callback_jitter_p95_ms": jit, "median_input_latency_ms": lat,
            "total_ms": block_ms + jit}


def analyze(blocks: list[Block], clips: list[Clip], *, threshold_dbfs: float, unc_ms: float,
            settle_ms: float = 0.0) -> dict[str, Any]:
    """Asocia el primer audio capturado de cada clip (solo si es inequívoco)."""
    unc = round(unc_ms * 1e6)
    loud = [b for b in blocks if b.rms_dbfs > threshold_dbfs]
    results, covered = [], []
    for c in clips:
        end = c.end_ns if c.end_ns is not None else (blocks[-1].t_cb_ns if blocks else c.begin_ns)
        win_end = end + round(settle_ms * 1e6)
        covered.append((c.begin_ns - unc, win_end))
        w = first_audible_write(c)
        res: dict[str, Any] = {"stream_id": c.stream_id, "n_writes": len(c.writes),
                               "commit_id": w.commit_id if w else None,
                               "first_audible_write_ns": w.t_ns if w else None, "status": None}
        in_win = [b for b in loud if c.begin_ns - unc <= b.start_est_ns <= win_end]
        if w is None:
            res["status"] = "contaminated_no_audible_write" if in_win else "no_audible_write"
        else:
            before = [b for b in in_win if b.start_est_ns < w.t_ns - unc]
            after = [b for b in in_win if b.start_est_ns >= w.t_ns - unc]
            if before:
                res["status"] = "contaminated_before_write"
                res["contaminating_blocks"] = len(before)
            elif not after:
                res["status"] = "no_device_onset"
            else:
                b = after[0]
                res.update(
                    status="ok", device_onset_ns=b.start_est_ns, device_onset_raw_callback_ns=b.t_cb_ns,
                    device_minus_first_write_ms=(b.start_est_ns - w.t_ns) / 1e6,
                    uncertainty_ms=unc_ms,
                    note="ventana de captura incluye latencia del Player+CoreAudio+BlackHole+bloque de captura",
                )
        # energía capturada vs escrita (sospecha de audio ajeno SUPERPUESTO con el propio)
        exp_s = sum(x.nonsilent_s for x in c.writes)
        got_s = sum(b.dur_ns for b in in_win) / 1e9
        res["captured_nonsilent_s"], res["written_nonsilent_s"] = got_s, exp_s
        if res["status"] == "ok" and got_s > exp_s * 1.25 + 0.2:
            res["status"] = "contamination_suspected_energy"
        res["unambiguous"] = res["status"] == "ok"
        results.append(res)
    foreign = [b for b in loud if not any(a <= b.start_est_ns <= z for a, z in covered)]
    return {"clips": results, "foreign_audio_blocks": len(foreign),
            "foreign_first_ns": foreign[0].start_est_ns if foreign else None}


def max_block_gap_ms(blocks: list[Block]) -> float:
    """Mayor hueco entre callbacks consecutivos: una captura que se detiene NO es silencio."""
    return max(((b.t_cb_ns - a.t_cb_ns) / 1e6 for a, b in zip(blocks, blocks[1:])), default=0.0)


def json_safe(o):
    """JSON estándar: -inf/inf/NaN → null (nunca un valor finito inventado)."""
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, dict):
        return {k: json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_safe(v) for v in o]
    return o


def validity(capture_status_events: int | None, player: dict, late_events: int, *,
             max_block_gap_ms: float = 0.0, block_ms: float = 10.0) -> dict[str, Any]:
    """Condiciones que INVALIDAN la medición (no se reporta «ok» en silencio) y avisos que no la invalidan."""
    reasons, warnings = [], []
    if capture_status_events:
        reasons.append(f"capture_status_events={capture_status_events} (overflow/underflow de captura)")
    if player.get("dropped_s"):
        reasons.append(f"player_dropped_s={player['dropped_s']:.3f} (audio descartado por cola llena)")
    if player.get("status_events"):
        reasons.append(f"player_status_events={player['status_events']} (banderas PortAudio de salida)")
    if max_block_gap_ms > max(100.0, 10 * block_ms):
        reasons.append(f"capture_gap_ms={max_block_gap_ms:.0f} (la captura dejó de entregar bloques: no se puede distinguir silencio de captura detenida)")
    if late_events:
        reasons.append(f"late_or_unassigned_tts_events={late_events} (chunks fuera de un clip activo: barrera de entrega incumplida)")
    if player.get("underruns"):
        warnings.append(f"player_partial_blocks={player['underruns']} (bloques parcialmente rellenados: compatibles con el vaciado normal "
                        "al final de cada frase; NO demuestran un underrun físico; no afectan el primer onset)")
    return {"valid": not reasons, "invalidating_reasons": reasons, "warnings": warnings}


def merge_with_traces(device_report: dict, traces: list[dict]) -> list[dict]:
    """Une el informe con ``traces.jsonl`` (mismo reloj): fin anotado → audio en BlackHole.

    Solo con EXACTAMENTE un segmento fuente por stream y con el mismo ``commit_id`` en el primer
    chunk audible del colector y de la captura. Streams continuos o multi-segmento: no se adjudica
    el primer audio al fin de una frase concreta. Con medición inválida no se publican latencias.
    Se conservan la latencia bruta de callback y el onset corregido (estimación por ADC).
    «Inequívoco» = asignación temporal interna; no autentica el contenido de audio externo."""
    by_id = {t["stream_id"]: t for t in traces}
    valid = device_report.get("validity", {"valid": True, "invalidating_reasons": []})
    out = []
    for c in device_report["clips"]:
        t = by_id.get(c["stream_id"])
        row: dict[str, Any] = {"stream_id": c["stream_id"], "status": c["status"], "warmup": bool(t and t.get("warmup"))}
        if c["status"] == "ok" and not valid["valid"]:
            row.update(status="invalid_measurement", reasons=valid["invalidating_reasons"])
        elif t and c["status"] == "ok":
            segs = t.get("segments", [])
            first_pcm = next((e for e in t["events"] if e["kind"] == "tts_chunk"
                              and e.get("first_nonsilent_offset_s") is not None), None)
            if t.get("continuous") or len(segs) != 1:
                row.update(status="not_attributable_multi_segment",
                           reason=f"{len(segs)} segmentos fuente (continuous={bool(t.get('continuous'))}): el primer audio no se adjudica a un fin de frase")
            elif first_pcm is None or first_pcm.get("commit_id") != c.get("commit_id"):
                row.update(status="commit_mismatch",
                           reason=f"commit del colector {first_pcm and first_pcm.get('commit_id')!r} ≠ commit de la captura {c.get('commit_id')!r}")
            else:
                seg = segs[0]
                raw = c["device_onset_raw_callback_ns"]
                est = c["device_onset_ns"]
                row["available_to_device_ms"] = (est - first_pcm["recv_ns"]) / 1e6
                row["write_to_device_ms"] = c["device_minus_first_write_ms"]
                row["input_latency_correction_ms"] = (raw - est) / 1e6
                row["uncertainty_ms"] = c["uncertainty_ms"]
                if seg.get("end_s") is not None:
                    end_ns = t["t0_ns"] + round(seg["end_s"] * 1e9)
                    row["end_to_device_raw_callback_ms"] = (raw - end_ns) / 1e6  # bruto: instante del callback
                    row["end_to_device_estimated_ms"] = (est - end_ns) / 1e6  # corregido por latencia ADC (estimación)
                    row["boundary_method"] = seg.get("boundary_method")
                    row["boundary_uncertainty_s"] = seg.get("boundary_uncertainty_s")
        out.append(row)
    return out


# ------------------------------------------------------------------ grabador de clips
class Recorder:
    def __init__(self, capture, *, threshold_dbfs: float, settle_s: float, clock: Callable[[], int] = MONO_NS) -> None:
        self.capture = capture
        self.threshold = threshold_dbfs
        self.settle_s = settle_s
        self.clock = clock
        self.clips: list[Clip] = []
        self.calibration: dict[str, Any] = {}
        self._cur: Clip | None = None

    async def calibrate(self, pre_silence_s: float) -> dict[str, Any]:
        n0 = len(self.capture.blocks)
        await asyncio.sleep(pre_silence_s)
        blocks = self.capture.blocks[n0:]
        expected = pre_silence_s * 1000 / self.capture.blocksize_ms
        if len(blocks) < 0.5 * expected:
            raise DeviceLoopbackError(f"captura sin bloques suficientes ({len(blocks)} de ~{expected:.0f}): ¿dispositivo parado?")
        peak = max(b.rms_dbfs for b in blocks)
        if peak > self.threshold:
            raise ContaminationError(f"audio ajeno antes de reproducir: {peak:.1f} dBFS > {self.threshold} dBFS; medición abortada")
        u = uncertainty_ms(blocks, self.capture.blocksize_ms)
        self.calibration = {"n_blocks": len(blocks), "peak_dbfs": peak if math.isfinite(peak) else None,
                            "peak_state": "silence_digital" if not math.isfinite(peak) else "signal_below_threshold",
                            "pre_silence_s": pre_silence_s,
                            "sample_rate": self.capture.sr, "channels": self.capture.channels, **u}
        return self.calibration

    def _quiet_for(self, seconds: float) -> bool:
        now = self.clock()
        recent = [b for b in self.capture.blocks if b.start_est_ns >= now - round(seconds * 1e9)]
        return len(recent) > 0 and all(b.rms_dbfs <= self.threshold for b in recent)

    async def wait_quiet(self, player_queued: Callable[[], float], timeout_s: float) -> None:
        t_end = self.clock() + round(timeout_s * 1e9)
        while True:
            if self.capture.blocks and self.clock() - self.capture.blocks[-1].t_cb_ns > 250 * MS_NS:
                raise DeviceLoopbackError("la captura dejó de entregar bloques (no se interpreta como silencio)")
            if player_queued() <= 0.0 and self._quiet_for(self.settle_s):
                return
            if self.clock() > t_end:
                raise ContaminationError("no se alcanzó silencio de captura con cola vacía a tiempo (¿audio ajeno o cola atascada?)")
            await asyncio.sleep(0.02)

    def begin(self, stream_id: str) -> None:
        self._cur = Clip(stream_id, self.clock())
        self.clips.append(self._cur)

    def write(self, stream_id: str, pcm: bytes, sr: int, commit_id: str | None) -> int:
        t = self.clock()
        c = self._cur
        if c is None or c.stream_id != stream_id:
            raise DeviceLoopbackError(f"escritura al Player fuera de un clip activo ({stream_id})")
        c.writes.append(Write(t, len(pcm) // 2, sr, A.first_nonsilent_offset_s(pcm, sr, threshold_dbfs=self.threshold),
                              nonsilent_seconds(pcm, sr, self.threshold), commit_id))
        return t

    def end(self, stream_id: str) -> None:
        if self._cur is not None and self._cur.stream_id == stream_id:
            self._cur.end_ns = self.clock()
            self._cur = None

    def foreign_since_last_clip(self) -> int:
        a = analyze(self.capture.blocks, self.clips, threshold_dbfs=self.threshold,
                    unc_ms=self.calibration.get("total_ms", 10.0), settle_ms=self.settle_s * 1000)
        return a["foreign_audio_blocks"]

    def report(self) -> dict[str, Any]:
        unc = self.calibration.get("total_ms", self.capture.blocksize_ms)
        a = analyze(self.capture.blocks, self.clips, threshold_dbfs=self.threshold, unc_ms=unc,
                    settle_ms=self.settle_s * 1000)
        for c in a["clips"]:
            if c.get("uncertainty_ms") is None and c["status"] == "ok":
                c["uncertainty_ms"] = unc
        return a
