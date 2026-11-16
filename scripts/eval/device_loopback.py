"""Orquesta la medición de salida a BlackHole con el arnés (replay causal a 20 ms).

    # solo imprime el comando exacto; no abre dispositivos ni carga modelos
    .venv/bin/python scripts/eval/device_loopback.py command --out artifacts/benchmark/device-loopback-1

    # ejecuta (REQUIERE lease de cómputo y BlackHole 2ch cargado; abre SOLO BlackHole)
    .venv/bin/python scripts/eval/device_loopback.py run --out artifacts/benchmark/device-loopback-1

    # re-unir un informe de dispositivo con traces.jsonl ya existentes (offline)
    .venv/bin/python scripts/eval/device_loopback.py merge --out artifacts/benchmark/device-loopback-1

Entrada: WAV público del manifest, reproducido causalmente a 20 ms hacia el Engine (arnés existente).
Salida: ``device_loopback.json`` (clips, estado, incertidumbre) y ``device_loopback_summary.json``
(fin anotado → audio en BlackHole, solo clips inequívocos). Solo métricas y textos: no se graba audio.
No es audio escuchado en Meet ni mide micrófono físico ni acústica.
"""
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DEFAULT_ITEMS = ["fleurs-es_419-validation-1516"]
MANIFESTS = ["data/manifests/fleurs_es_sample.jsonl", "data/manifests/openslr61_es_ar_sample.jsonl"]


def build(args) -> tuple[list[str], Path]:
    out = Path(args.out)
    sub = out / "manifest_subset.jsonl"
    cfg = {
        "inner": args.inner, "inner_config": json.loads(args.inner_config),
        "pre_silence_s": args.pre_silence_s, "threshold_dbfs": args.threshold_dbfs,
        "settle_s": args.settle_s, "player_latency": (None if args.player_latency == "default" else args.player_latency),
        "report_path": str(out / "device_loopback.json"),
    }
    argv = [
        "run", "--manifest", str(sub), "--audio-root", "data/audio",
        "--adapter", "benchmark.adapters.device_loopback:DeviceLoopbackAdapter",
        "--adapter-config", json.dumps(cfg), "--split", args.split,
        "--repetitions", str(args.repetitions), "--warmup", str(args.warmup),
        "--deadline-s", "5", "--call-timeout-s", "60", "--watchdog-stall-s", "120",
        "--out", str(out), "--system-name", "device-loopback-blackhole",
    ]
    return argv, sub


def write_subset(sub: Path, ids: list[str]) -> None:
    by_id = {}
    for m in MANIFESTS:  # subconjunto mixto: FLEURS + OpenSLR61 (mismo audio-root data/audio)
        for l in (ROOT / m).read_text(encoding="utf-8").splitlines():
            if l.strip():
                by_id.setdefault(json.loads(l)["item_id"], l)
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise SystemExit(f"ítems no encontrados en {MANIFESTS}: {missing}")
    rows = [by_id[i] for i in ids]  # orden pedido
    sub.parent.mkdir(parents=True, exist_ok=True)
    sub.write_text("\n".join(rows) + "\n", encoding="utf-8")


def merge(out: Path) -> Path:
    from benchmark import device_loopback as DL

    rep = json.loads((out / "device_loopback.json").read_text(encoding="utf-8"))
    traces = [json.loads(l) for l in (out / "traces.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = DL.merge_with_traces(rep, traces)
    ok = [r for r in rows if r["status"] == "ok" and not r["warmup"]]
    meas = [r for r in rows if "end_to_device_estimated_ms" in r and not r["warmup"]]
    summ = {
        "schema": "opentraductor-device-loopback-summary/v1",
        "scope": rep["scope"], "n_clips": len(rows), "n_ok_unambiguous": len(ok), "n_with_end_to_device": len(meas),
        "validity": rep.get("validity"), "capture_status_events": rep.get("capture_status_events"),
        "player": rep.get("player"), "late_events": rep.get("late_events"),
        "statuses": {s: sum(1 for r in rows if r["status"] == s) for s in sorted({r["status"] for r in rows})},
        "rows": rows, "calibration": rep["calibration"],
        "caveats": [
            "n pequeño: sin cuantiles ni IC; no es un test del objetivo de 300 ms",
            "fin de habla anotado puede ser estimado (ver boundary_method del manifest)",
            "uncertainty_ms = bloque + jitter p95 de callbacks: NO cubre la calibración del reloj de PortAudio, la latencia de salida del Player/BlackHole ni la incertidumbre del límite anotado (boundary_uncertainty_s)",
            "se publican el callback BRUTO y el onset estimado (corregido por currentTime − inputBufferAdcTime)",
            "«inequívoco» = asignación temporal interna; no autentica audio externo superpuesto",
            "no es audio escuchado en Meet/Discord ni micrófono físico",
        ],
    }
    p = out / "device_loopback_summary.json"
    p.write_text(json.dumps(DL.json_safe(summ), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["command", "run", "merge"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--items", nargs="+", default=DEFAULT_ITEMS)
    ap.add_argument("--repetitions", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--pre-silence-s", type=float, default=1.0)
    ap.add_argument("--settle-s", type=float, default=0.3)
    ap.add_argument("--threshold-dbfs", type=float, default=-50.0)
    ap.add_argument("--player-latency", choices=["default", "low", "high"], default="default",
                    help="latency del OutputStream del Player (solo el Player; la captura no cambia entre brazos)")
    ap.add_argument("--split", choices=["development", "test"], default="development",
                    help="split del manifest; test = held-out (solo con congelado y autorización; purpose=final_test automático)")
    ap.add_argument("--inner", default="runtime.adapters.baseline:BaselineAdapter")
    ap.add_argument("--inner-config", default="{}")
    a = ap.parse_args()
    out = Path(a.out)
    if a.action == "merge":
        print(merge(out))
        return 0
    argv, sub = build(a)
    cmd = "cd %s && .venv/bin/python -m benchmark %s" % (shlex.quote(str(ROOT)), " ".join(shlex.quote(x) for x in argv))
    write_subset(sub, a.items)  # solo archivos; no abre dispositivos ni modelos
    if a.action == "command":
        print(cmd)
        return 0
    from benchmark.__main__ import main as bench_main

    rc = bench_main(argv)
    if (out / "device_loopback.json").exists() and (out / "traces.jsonl").exists():
        print(merge(out))
    elif rc != 0:
        print(f"sin traces.jsonl: la corrida falló (ver {out}/FAILURE.json); no se genera resumen", file=sys.stderr)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
