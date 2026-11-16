"""Prueba corta (sin modelos) default vs low del Player hacia BlackHole. NO ejecutar sin lease: abre BlackHole.

    .venv/bin/python scripts/eval/player_latency_probe.py command --out artifacts/benchmark/player-latency-probe-1   # imprime
    .venv/bin/python scripts/eval/player_latency_probe.py run     --out artifacts/benchmark/player-latency-probe-1   # ejecuta (~1 min)

Cada brazo abre un Player nuevo (modo pedido) y una captura nueva idéntica; orden ABBA × rondas. Ver benchmark/player_probe.py.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


async def run(a) -> dict:
    from benchmark import device_loopback as DL
    from benchmark import player_probe as PP
    from runtime import audio_io

    arms = []
    for i, mode in enumerate(PP.arm_order(a.rounds)):
        arms.append(await PP.run_arm(
            mode, i, make_capture=lambda: DL.BlackHoleCapture(),
            make_player=lambda dev, lat: audio_io.Player(dev, volume=1.0, latency=lat), bursts=a.bursts))
        await asyncio.sleep(0.5)
    return {"schema": "opentraductor-player-latency-probe/v1", "scope": "tono sintético Player→BlackHole→captura; sin modelos, sin micrófono ni altavoces; "
            "no es latencia del pipeline", "rounds": a.rounds, "bursts_per_arm": a.bursts, "arms": arms, "summary": PP.summarize(arms)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["command", "run", "reanalyze"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--bursts", type=int, default=6)
    a = ap.parse_args()
    if a.action == "reanalyze":
        from benchmark import device_loopback as DL
        from benchmark import player_probe as PP

        src = Path(a.out) / "player_latency_probe.json"
        data = json.loads(src.read_text(encoding="utf-8"))
        data["summary_v1_original"] = data.pop("summary")
        data["summary"] = PP.summarize(data["arms"])
        data["correction"] = ("análisis v2 recalculado desde los datos por brazo SIN nueva captura: las parejas de la comparación de signo pasan "
                              "de zip(sorted(default), sorted(low)) (emparejaba por magnitud) a parejas temporales predefinidas por índice ABBA. "
                              "Las ráfagas guardadas en esta corrida NO conservan el callback bruto ni el onset ADC (no se recuperan).")
        dst = Path(a.out) / "player_latency_probe_v2.json"
        dst.write_text(json.dumps(DL.json_safe(data), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        print(json.dumps(data["summary"]["comparison"], indent=1, ensure_ascii=False))
        return 0
    if a.action == "command":
        print("cd %s && .venv/bin/python scripts/eval/player_latency_probe.py run %s" % (
            shlex.quote(str(ROOT)), " ".join(shlex.quote(x) for x in sys.argv[2:] if x != "command")))
        return 0
    from benchmark import device_loopback as DL

    res = DL.json_safe(asyncio.run(run(a)))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "player_latency_probe.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps(res["summary"].get("comparison"), indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
