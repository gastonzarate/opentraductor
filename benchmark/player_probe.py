"""Prueba corta y SIN modelos de la ruta Player → BlackHole → captura: ¿cambia ``latency='low'`` el tiempo
escritura→audio frente al valor por defecto? Aísla el buffer del Player antes de repetir el pipeline completo.

Diseño (declarado antes de medir):
* Estímulo: ráfagas de tono sintético precalculado (440 Hz, −14 dBFS, 300 ms, fade-in 5 ms, 24 kHz). Sin modelos ni voz.
* Un brazo = un Player nuevo con su modo + una captura nueva, con el mismo orden y separación que el arnés
  (Player primero, ≥0.3 s después la captura; ver hallazgo AUHAL). La captura es idéntica entre brazos.
* Orden de brazos ABBA (default, low, low, default) × rondas para repartir la deriva temporal.
* Medida: escritura al Player → primer bloque > −50 dBFS en BlackHole (``Recorder``/``analyze`` del arnés, mismo reloj).
* Se informan valores crudos, mediana/cuartiles por modo y por brazo, incertidumbre de captura y la latencia que
  DECLARA PortAudio (no medida). Regla descriptiva: «efecto candidato» solo si |Δmediana| > 2×incertidumbre y el
  signo coincide en todas las parejas de brazos; si no, «sin efecto distinguible». No es una prueba estadística.
* Brazos con estado de captura, drops del Player o contaminación quedan marcados inválidos y se excluyen.
"""
from __future__ import annotations

import asyncio
import statistics
from typing import Any, Callable

import numpy as np

from . import device_loopback as DL

SR = 24000


def make_tone_pcm(dur_s: float = 0.3, freq_hz: float = 440.0, level_dbfs: float = -14.0, fade_ms: float = 5.0, sr: int = SR) -> bytes:
    n = round(dur_s * sr)
    t = np.arange(n) / sr
    x = np.sin(2 * np.pi * freq_hz * t) * (10 ** (level_dbfs / 20)) * np.sqrt(2)
    f = max(1, round(fade_ms / 1000 * sr))
    x[:f] *= np.linspace(0.0, 1.0, f)
    x[-f:] *= np.linspace(1.0, 0.0, f)
    return np.clip(np.round(x * 32767), -32768, 32767).astype("<i2").tobytes()


def arm_order(rounds: int) -> list[str]:
    return ["default", "low", "low", "default"] * rounds


async def run_arm(mode: str, index: int, *, make_capture: Callable[[], Any], make_player: Callable[[Any, str | None], Any],
                  bursts: int = 6, pre_silence_s: float = 0.6, settle_s: float = 0.3, gap_s: float = 0.4,
                  start_gap_s: float = 0.3, threshold_dbfs: float = DL.DEFAULT_THRESHOLD_DBFS, drain_timeout_s: float = 5.0) -> dict:
    pcm = make_tone_pcm()
    dur_s = len(pcm) / 2 / SR
    capture = make_capture()
    dev = capture.pick_device()
    player = make_player(dev, None if mode == "default" else mode)
    out: dict[str, Any] = {"arm": index, "mode": mode, "requested_latency_mode": mode, "clips": []}
    try:
        player.start()
        await asyncio.sleep(start_gap_s)  # Player primero; captura después (AUHAL)
        capture.start()
        rec = DL.Recorder(capture, threshold_dbfs=threshold_dbfs, settle_s=settle_s)
        out["calibration"] = await rec.calibrate(pre_silence_s)
        for k in range(bursts):
            await rec.wait_quiet(lambda: player.queued_s, drain_timeout_s)
            sid = f"arm{index}-{mode}-{k}"
            rec.begin(sid)
            rec.write(sid, pcm, SR, f"{sid}:0")
            player.write(pcm, SR)
            await asyncio.sleep(dur_s)
            await rec.wait_quiet(lambda: player.queued_s, drain_timeout_s)
            rec.end(sid)
            await asyncio.sleep(gap_s)
        rep = rec.report()
        pstate = {"dropped_s": getattr(player, "dropped_s", 0.0), "status_events": getattr(player, "status_events", 0),
                  "underruns": getattr(player, "partial_blocks", getattr(player, "underruns", 0))}
        out["validity"] = DL.validity(getattr(capture, "status_events", 0), pstate, 0,
                                      max_block_gap_ms=DL.max_block_gap_ms(capture.blocks), block_ms=capture.blocksize_ms)
        out["player"] = {**pstate, "reported_stream_latency_s": getattr(player, "latency_s", None),
                         "reported_stream_latency_note": "declarado por PortAudio al abrir; NO es una medición física",
                         "blocksize": getattr(player, "blocksize", None), "sample_rate": getattr(player, "sr", None)}
        out["foreign_audio_blocks"] = rep["foreign_audio_blocks"]
        # Se conserva el registro COMPLETO por ráfaga (onset estimado por ADC y callback bruto incluidos).
        out["clips"] = [dict(c) for c in rep["clips"]]
        out["capture_block_ms"] = capture.blocksize_ms
    finally:
        for fn in (getattr(player, "stop", None), getattr(capture, "stop", None)):
            if fn:
                try:
                    fn()
                except Exception:  # noqa: BLE001
                    pass
    return out


def _q(vals: list[float], p: float) -> float:
    s = sorted(vals)
    return s[round(p * (len(s) - 1))]


def temporal_pairs(arms: list[dict]) -> list[tuple[int, int]]:
    """Parejas (default, low) PREDEFINIDAS por posición ABBA, nunca por magnitud observada:
    en cada ronda r (4 brazos 4r..4r+3 = default, low, low, default): (4r, 4r+1) y (4r+3, 4r+2)."""
    idx = {a["arm"]: a for a in arms}
    out = []
    for r in range(len(arms) // 4):
        for d, l in ((4 * r, 4 * r + 1), (4 * r + 3, 4 * r + 2)):
            if d in idx and l in idx and idx[d]["mode"] == "default" and idx[l]["mode"] == "low":
                out.append((d, l))
    return out


def summarize(arms: list[dict]) -> dict[str, Any]:
    per_mode: dict[str, list[float]] = {}
    arm_medians: list[dict] = []
    excluded = []
    for a in arms:
        ok = [c["device_minus_first_write_ms"] for c in a["clips"] if c["status"] == "ok"]
        if not a.get("validity", {}).get("valid", False) or a.get("foreign_audio_blocks") or len(ok) != len(a["clips"]) or not ok:
            excluded.append({"arm": a["arm"], "mode": a["mode"], "reasons": a.get("validity", {}).get("invalidating_reasons"),
                             "foreign_audio_blocks": a.get("foreign_audio_blocks"),
                             "clip_statuses": [c["status"] for c in a["clips"]]})
            continue
        per_mode.setdefault(a["mode"], []).extend(ok)
        arm_medians.append({"arm": a["arm"], "mode": a["mode"], "median_ms": statistics.median(ok), "n": len(ok)})
    modes = {m: {"n": len(v), "values_ms": [round(x, 2) for x in v], "median_ms": statistics.median(v), "p25_ms": _q(v, 0.25),
                 "p75_ms": _q(v, 0.75), "min_ms": min(v), "max_ms": max(v)} for m, v in per_mode.items()}
    unc = max((c["uncertainty_ms"] or 0.0 for a in arms for c in a["clips"] if c.get("uncertainty_ms")), default=None)
    out: dict[str, Any] = {
        "analysis_version": 2, "modes": modes, "arm_medians": arm_medians, "excluded_arms": excluded, "capture_uncertainty_ms": unc,
        "reported_stream_latency_s": {a["mode"]: a["player"]["reported_stream_latency_s"] for a in arms if "player" in a},
        "independence_note": ("las ráfagas de un mismo brazo NO son muestras independientes (mismo Player/captura abiertos, misma deriva); "
                              "la unidad de análisis es el brazo (y la pareja temporal de brazos); n de ráfagas solo describe dispersión"),
        "calibration_limit": ("la incertidumbre de captura (bloque + jitter p95) NO cubre la calibración del reloj de PortAudio ni la validez de la "
                              "estimación por ADC; el onset ya está corregido restando currentTime − inputBufferAdcTime"),
    }
    if "default" in modes and "low" in modes and unc is not None:
        d = modes["low"]["median_ms"] - modes["default"]["median_ms"]
        by_arm = {m["arm"]: m["median_ms"] for m in arm_medians}
        pairs, excluded_pairs = [], []
        for dflt, low in temporal_pairs(arms):
            if dflt in by_arm and low in by_arm:
                pairs.append({"default_arm": dflt, "low_arm": low, "low_minus_default_ms": by_arm[low] - by_arm[dflt]})
            else:
                excluded_pairs.append({"default_arm": dflt, "low_arm": low, "reason": "algún brazo excluido por validez"})
        same_sign = bool(pairs) and d != 0 and all(p["low_minus_default_ms"] * d > 0 for p in pairs)
        out["comparison"] = {"low_minus_default_median_ms": d, "decision_threshold_ms": 2 * unc,
                             "temporal_pairs": pairs, "excluded_pairs": excluded_pairs, "n_arm_pairs": len(pairs),
                             "pairs_same_sign": same_sign,
                             "verdict": "efecto candidato" if abs(d) > 2 * unc and same_sign else "sin efecto distinguible",
                             "note": ("descriptivo, regla fijada antes de medir; parejas temporales predefinidas por posición ABBA (no por magnitud); "
                                      "no es una prueba estadística; tono sintético, no el pipeline")}
    return out
