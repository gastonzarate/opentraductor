"""Estimación independiente de límites de habla (inicio/fin) para un manifest.

Versión 2 (auditoría de límites). Método: energía por bloque de 10 ms en dBFS.

* Umbral = max(piso de ruido P10 + 10 dB, PISO ABSOLUTO -50 dBFS). El piso relativo
  solo falla cuando la grabación tiene cola casi digital (ruido P10 < -60 dBFS): el
  umbral relativo (< -55 dBFS) contaba dither/ruido de cola como habla y corría el fin
  hasta ~1.5 s (FLEURS 1516, 1593: 11.26 s vs ~9.6 s real).
* Cierre y apertura morfológicos sobre los bloques sobre umbral: se unen huecos de
  <= 150 ms (cierres oclusivos, pausas cortas) y se descartan tramos de < 120 ms
  (clics, respiraciones, ruidos de boca aislados). Corrige FLEURS 1549: un ruido
  aislado a ~10 s extendía el fin 0.9 s sobre el último tramo sostenido (~9.3 s).
* Tramos de borde cortos (< 0.5 s) aislados por un hueco > 0.5 s se descartan como
  ráfagas/ruido (inicio de archivo en FLEURS 1584/1624/1652, respiración de cola en
  1549); cada descarte queda registrado en ``edge_runs_discarded``.
* Verificación cruzada con un segundo método independiente (misma lógica, energía
  en la banda 300-3400 Hz). Si el fin de ambos difiere > 0.3 s, el item queda
  "unknown" (no hay acuerdo; el error no cabe en la tolerancia supuesta).
* Si el último tramo de habla toca el final del archivo (recortado), el fin real es
  desconocido: item "unknown".

No usa timestamps del candidato ni inferencia de modelos (tampoco el VAD del
candidato como referencia). No es anotación humana ni alineación forzada revisada:
boundary_method = "estimated", con incertidumbre declarada de +/- 0.3 s por borde.

IMPORTANTE: 0.3 s es una tolerancia SUPUESTA, no una cota validada ni un intervalo
de confianza. No hay ground truth para medirla. Si el inicio estimado es mayor que
LATE_ONSET_S (3 s; en FLEURS el silencio/ruido inicial llega a ~2.4 s y los inicios
de la v1, 0.3-0.4 s, eran ruido de fondo) el item se marca "unknown". Escribe el manifest actualizado in place,
conserva el historial de cambios por item (``boundary_history``) y guarda parámetros,
validación y antes/después en el metadata JSON. No modifica audios ni referencias.

Uso:
    uv run python scripts/datasets/estimate_boundaries.py \
        --manifest data/manifests/fleurs_es_sample.jsonl \
        --audio-root data/audio
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from benchmark import audio as A  # noqa: E402
from benchmark import contracts  # noqa: E402

ESTIMATOR_VERSION = 2
FRAME_S = 0.010
NOISE_PCT = 10
SPEECH_PCT = 90
MARGIN_DB = 10.0
ABS_FLOOR_DBFS = -50.0  # el umbral nunca baja de aquí (dither/ruido de cola no es habla)
CLOSE_GAP_S = 0.15  # huecos más cortos se unen
MIN_RUN_S = 0.12  # tramos más cortos se descartan (clics, respiraciones)
BAND_HZ = (300.0, 3400.0)
METHOD_DISAGREE_S = 0.3
TOUCH_END_S = 0.15  # habla que termina a menos de esto del final del archivo = recortada
EDGE_RUN_MAX_S = 0.5  # tramo de borde más corto que esto y aislado por un hueco largo = ruido
EDGE_GAP_MIN_S = 0.5
MIN_SEPARATION_DB = 6.0
UNCERTAINTY_S = 0.3  # tolerancia SUPUESTA por borde, no validada
LATE_ONSET_S = 3.0  # inicio más tardío que esto -> item desconocido (FLEURS: silencio inicial de hasta ~2.4 s observado y verificado)
CPS_MIN, CPS_MAX = 6.0, 28.0  # caracteres/s de habla; fuera de rango -> desconocido (lectura rápida es-AR verificada en desarrollo: 19-23)
MIN_SPEECH_S = 0.3


def frame_dbfs(pcm: bytes, band: tuple[float, float] | None = None) -> tuple[np.ndarray, int]:
    x = A.to_float(pcm)
    if band is not None:  # pasabanda por FFT sobre toda la señal
        spec = np.fft.rfft(x)
        f = np.fft.rfftfreq(x.size, 1 / A.TARGET_SR)
        spec[(f < band[0]) | (f > band[1])] = 0
        x = np.fft.irfft(spec, n=x.size)
    n = max(1, round(A.TARGET_SR * FRAME_S))
    usable = (x.size // n) * n
    frames = x[:usable].reshape(-1, n)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    with np.errstate(divide="ignore"):
        db = 20 * np.log10(np.maximum(rms, 1e-9) / 32768.0)
    return db, n


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(int), [0]))))
    return [(int(idx[i]), int(idx[i + 1])) for i in range(0, len(idx), 2)]


def sustained_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Cierre (une huecos <= CLOSE_GAP_S) y apertura (descarta tramos < MIN_RUN_S)."""
    gap = round(CLOSE_GAP_S / FRAME_S)
    minrun = round(MIN_RUN_S / FRAME_S)
    merged: list[list[int]] = []
    for s_, e_ in _runs(mask):
        if merged and s_ - merged[-1][1] <= gap:
            merged[-1][1] = e_
        else:
            merged.append([s_, e_])
    return [(s_, e_) for s_, e_ in merged if e_ - s_ >= minrun]


def trim_isolated_edges(runs: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], list[tuple[float, float]]]:
    """Descarta tramos cortos (< EDGE_RUN_MAX_S) en los bordes separados del resto por
    un hueco largo (> EDGE_GAP_MIN_S): ráfagas/clics de inicio de archivo, respiraciones
    de cola. Es una suposición: una palabra real aislada al borde también se descartaría;
    queda registrada en ``edge_runs_discarded``."""
    runs = list(runs)
    dropped: list[tuple[float, float]] = []
    mx, mg = round(EDGE_RUN_MAX_S / FRAME_S), round(EDGE_GAP_MIN_S / FRAME_S)
    changed = True
    while changed and len(runs) > 1:
        changed = False
        if runs[0][1] - runs[0][0] < mx and runs[1][0] - runs[0][1] > mg:
            dropped.append((round(runs[0][0] * FRAME_S, 3), round(runs[0][1] * FRAME_S, 3)))
            runs.pop(0)
            changed = True
        if len(runs) > 1 and runs[-1][1] - runs[-1][0] < mx and runs[-1][0] - runs[-2][1] > mg:
            dropped.append((round(runs[-1][0] * FRAME_S, 3), round(runs[-1][1] * FRAME_S, 3)))
            runs.pop()
            changed = True
    return runs, dropped


def _estimate_one(db: np.ndarray) -> dict:
    noise = float(np.percentile(db, NOISE_PCT))
    speech = float(np.percentile(db, SPEECH_PCT))
    threshold = max(noise + MARGIN_DB, ABS_FLOOR_DBFS)
    result = {
        "noise_floor_dbfs": round(noise, 2),
        "speech_level_dbfs": round(speech, 2),
        "threshold_dbfs": round(threshold, 2),
        "threshold_from": "piso_absoluto" if threshold == ABS_FLOOR_DBFS and noise + MARGIN_DB < ABS_FLOOR_DBFS else "piso_relativo",
        "frame_s": FRAME_S,
    }
    if speech - noise < MIN_SEPARATION_DB or threshold > speech - MIN_SEPARATION_DB:
        result["status"] = "no_separation"
        return result
    runs, dropped = trim_isolated_edges(sustained_runs(db > threshold))
    if dropped:
        result["edge_runs_discarded"] = dropped
    if not runs:
        result["status"] = "no_speech_above_threshold"
        return result
    start = runs[0][0] * FRAME_S
    end = runs[-1][1] * FRAME_S
    total = db.size * FRAME_S
    result.update(
        {
            "status": "ok",
            "start_s": round(float(start), 3),
            "end_s": round(float(end), 3),
            "speech_span_s": round(float(end - start), 3),
            "n_sustained_runs": len(runs),
            "touches_end_of_file": bool(total - end < TOUCH_END_S),
        }
    )
    return result


def estimate(pcm: bytes) -> dict:
    """Método A (banda completa) + verificación cruzada B (300-3400 Hz)."""
    db, _ = frame_dbfs(pcm)
    res = _estimate_one(db)
    dbb, _ = frame_dbfs(pcm, BAND_HZ)
    b = _estimate_one(dbb)
    res["band_check"] = {"status": b["status"], "start_s": b.get("start_s"), "end_s": b.get("end_s"),
                         "threshold_dbfs": b["threshold_dbfs"], "edge_runs_discarded": b.get("edge_runs_discarded")}
    if res["status"] == "ok":
        if b["status"] != "ok":
            res["status"] = "band_check_failed"
        elif max(abs(b["end_s"] - res["end_s"]), abs(b["start_s"] - res["start_s"])) > METHOD_DISAGREE_S:
            res["status"] = "methods_disagree"
            res["band_check"]["end_diff_s"] = round(abs(b["end_s"] - res["end_s"]), 3)
            res["band_check"]["start_diff_s"] = round(abs(b["start_s"] - res["start_s"]), 3)
            res["note"] = ("los dos métodos no coinciden: no se elige uno; requiere anotación manual. "
                           f"fin banda completa {res['end_s']} s vs banda de voz {b['end_s']} s")
        elif res["touches_end_of_file"]:
            res["status"] = "speech_touches_end_of_file"
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--audio-root", type=Path, required=True)
    ap.add_argument("--metadata", type=Path, default=None)
    args = ap.parse_args()

    raw = [json.loads(l) for l in args.manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    report = []
    for d in raw:
        pcm = A.read_wav_pcm16_mono_16k(args.audio_root / d["audio_path"])
        dur = A.pcm_duration_s(pcm)
        est = estimate(pcm)
        # Plausibilidad: caracteres por segundo de habla con el texto humano.
        chars = len(d["transcript_es"] or "")
        cps = chars / est["speech_span_s"] if est.get("speech_span_s") else None
        est["duration_s"] = round(dur, 3)
        est["chars_per_s_speech"] = None if cps is None else round(cps, 2)
        est["item_id"] = d["item_id"]
        report.append(est)

        before = {"utterance_start_s": d.get("utterance_start_s"), "utterance_end_s": d.get("utterance_end_s"),
                  "boundary_method": d.get("boundary_method"),
                  "estimator_version": (d.get("boundary_estimator") or {}).get("version", 1)}
        late = est.get("start_s") is not None and est["start_s"] > LATE_ONSET_S
        implausible = cps is not None and not (CPS_MIN <= cps <= CPS_MAX)
        est["late_onset_flag"] = bool(late)
        est["implausible_rate_flag"] = bool(implausible)
        if (
            est["status"] == "ok"
            and not late
            and not implausible
            and est["speech_span_s"] >= MIN_SPEECH_S
            and est["end_s"] <= dur + 1e-6
        ):
            d["utterance_start_s"] = est["start_s"]
            d["utterance_end_s"] = min(est["end_s"], round(dur, 3))
            d["boundary_method"] = "estimated"
            d["boundary_uncertainty_s"] = UNCERTAINTY_S
        else:
            d["utterance_start_s"] = None
            d["utterance_end_s"] = None
            d["boundary_method"] = "unknown"
            d["boundary_uncertainty_s"] = None
            est["unknown_reason"] = (
                "late_onset" if late else "implausible_rate" if implausible else est["status"]
            )
        after = {"utterance_start_s": d["utterance_start_s"], "utterance_end_s": d["utterance_end_s"],
                 "boundary_method": d["boundary_method"], "estimator_version": ESTIMATOR_VERSION}
        est["before"], est["after"] = before, after
        if (before["utterance_start_s"], before["utterance_end_s"], before["boundary_method"]) != (
            after["utterance_start_s"], after["utterance_end_s"], after["boundary_method"]
        ):
            hist = d.setdefault("boundary_history", [])
            hist.append({**before, "replaced_by_estimator_version": ESTIMATOR_VERSION,
                         "reason": "auditoría de límites: piso absoluto, tramos sostenidos y verificación por banda"})
        d["boundary_estimator"] = {
            "version": ESTIMATOR_VERSION,
            "method": "energy_frames_10ms+sustained_runs+band_check",
            "abs_floor_dbfs": ABS_FLOOR_DBFS,
            "close_gap_s": CLOSE_GAP_S,
            "min_run_s": MIN_RUN_S,
            "band_hz": list(BAND_HZ),
            "method_disagree_s": METHOD_DISAGREE_S,
            "noise_percentile": NOISE_PCT,
            "speech_percentile": SPEECH_PCT,
            "margin_db": MARGIN_DB,
            "min_separation_db": MIN_SEPARATION_DB,
            "uncertainty_s_each_edge": UNCERTAINTY_S,
            "uncertainty_status": "supuesta, no validada ni IC",
            "late_onset_threshold_s": LATE_ONSET_S,
            "cps_range": [CPS_MIN, CPS_MAX],
            "independent_of_candidate": True,
        }

    args.manifest.write_text(
        "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in raw), encoding="utf-8"
    )
    # Revalidar con el contrato y con el runner.
    loaded = contracts.load_manifest(args.manifest, audio_root=args.audio_root, verify_sha256=True)
    from benchmark.runner import items_from_manifest

    items_from_manifest(loaded, args.audio_root)

    meta_path = args.metadata or args.manifest.with_suffix(".metadata.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    meta["boundaries"] = {
        "estimator": "scripts/datasets/estimate_boundaries.py",
        "method": "estimated",
        "estimator_version": ESTIMATOR_VERSION,
        "uncertainty_s_each_edge": UNCERTAINTY_S,
        "uncertainty_status": "0.3 s es tolerancia SUPUESTA, no cota validada ni intervalo de confianza",
        "late_onset_threshold_s": LATE_ONSET_S,
        "cps_range": [CPS_MIN, CPS_MAX],
        "note": "No son límites humanos ni alineación revisada. Sin ground truth: solo plausibilidad y diagnóstico.",
        "items": report,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    ok = sum(1 for d in raw if d["boundary_method"] == "estimated")
    print(f"items: {len(report)}, con límites estimados: {ok}, desconocidos: {len(report) - ok}")
    for r in report:
        print(
            r["item_id"],
            r["status"],
            r.get("start_s"),
            r.get("end_s"),
            f"dur={r['duration_s']}",
            f"cps={r['chars_per_s_speech']}",
        )


if __name__ == "__main__":
    main()
