"""T-55 integración runtime → UI: graba lo que publica /events para frases reales del flujo.

Usa el código real del runtime (RuntimeService._on_engine_event y
Engine baseline con ASR/MT/TTS falsos) y escribe tests/fixtures/t55_runtime_events.json.
Ese archivo lo reproducen la web (tests/ui/t55_integration.test.js) y la app de la tienda
(packaging/macos/Tests/main.swift): así la UI se prueba con la secuencia que sale del motor, no con
eventos escritos a mano. No carga modelos ni abre audio (unos segundos de CPU).

    .venv/bin/python -m tests.integration.t55_dump_events
"""
import asyncio
import json
import sys
import tempfile
from pathlib import Path

from runtime.config import PipelineConfig
from runtime.pipeline import Engine
from runtime.service import RuntimeService
from tests.runtime.test_service import FakeAudio, FakeEngine, ok_preflight
from tests.runtime.test_speculative import WIN, FakeASR, FakeMT, FakeTTS, FakeVAD

OUT = Path(__file__).resolve().parents[1] / "fixtures" / "t55_runtime_events.json"
UI_KINDS = {"partial", "segment", "discarded"}
SVC_KINDS = {"asr_partial", "endpoint", "asr_final", "discarded", "mt_final"}


def new_service(tmp):
    Path(tmp).mkdir(parents=True, exist_ok=True)
    s = RuntimeService(Path(tmp) / "t.db", pipeline_cfg=PipelineConfig(), engine_factory=FakeEngine,
                       audio=FakeAudio, preflight=ok_preflight)
    out = []
    s.subscribe(out.append)
    return s, out


async def to_service(svc, engine_events):
    for kind, f in engine_events:
        if kind in SVC_KINDS:
            await svc._on_engine_event(kind, f)


def ui_events(out):
    # Sin "ts" (cambia en cada corrida) para que el fixture sea estable en git.
    return [{k: v for k, v in e.items() if k != "ts"} for e in out if e.get("type") in UI_KINDS]


# ------------------------------------------------------------------ baseline (service real)
def scenario_baseline_overlap(tmp):
    """Frase 1 se traduce cuando la frase 2 ya muestra parciales (lo que vio la persona en la tienda)."""
    svc, out = new_service(tmp)
    engine = [
        ("asr_partial", {"seg_id": 1, "text": "primera"}),
        ("endpoint", {"seg_id": 1}),
        ("asr_final", {"seg_id": 1, "text": "Primera frase.", "text_raw": "primera frase"}),
        ("asr_partial", {"seg_id": 2, "text": "segunda"}),
        ("asr_partial", {"seg_id": 2, "text": "segunda en curso"}),
        ("mt_final", {"seg_id": 1, "text": "First sentence."}),
        ("asr_partial", {"seg_id": 2, "text": ""}),
        ("asr_final", {"seg_id": 2, "text": "Segunda en curso."}),
        ("asr_partial", {"seg_id": 3, "text": "mmm"}),
        ("asr_final", {"seg_id": 3, "text": ""}),
        ("mt_final", {"seg_id": 2, "text": "Second in progress."}),
    ]
    try:
        asyncio.run(to_service(svc, engine))
    finally:
        svc.close()
    return {
        "name": "baseline: segment de N con parcial de N+1 en pantalla, y final vacío",
        "events": ui_events(out),
        "expect": [
            ["1", "final", "Primera frase.", "First sentence."],
            ["2", "final", "Segunda en curso.", "Second in progress."],
            ["3", "discarded:empty", "mmm", ""],
        ],
    }


async def _pipeline(final, min_speech_ms, speech=10):
    events = []

    async def emit(kind, f):
        events.append((kind, dict(f)))

    cfg = PipelineConfig(asr_backend="moonshine-tiny", speculative_mt_tts=False, endpoint_silence_ms=160,
                         asr_chunk_ms=160, min_speech_ms=min_speech_ms)
    e = Engine(cfg, emit)
    e.asr, e.mt, e.tts = FakeASR("hola", final), FakeMT(), FakeTTS()
    e.vad_proto = FakeVAD(speech)
    e._out_q = asyncio.Queue(maxsize=8)
    e._out_task = asyncio.ensure_future(e._speaker_loop())
    await e.start_stream("s")
    for _ in range(speech + 8):
        await e.push("s", bytes(2 * WIN), 0.0)
        await asyncio.sleep(0.02)
    await e.end_stream("s")
    await e.close()
    return events


def scenario_pipeline(tmp, name, final, min_speech_ms, expect):
    svc, out = new_service(tmp)

    async def flow():
        await to_service(svc, await _pipeline(final, min_speech_ms))
    try:
        asyncio.run(flow())
    finally:
        svc.close()
    return {"name": name, "events": ui_events(out), "expect": expect}


def main():
    with tempfile.TemporaryDirectory() as tmp:
        scenarios = [
            scenario_baseline_overlap(Path(tmp) / "b"),
        ]
        scenarios += [
            scenario_pipeline(Path(tmp) / "c", "pipeline: frase normal", "hola", 0, None),
            scenario_pipeline(Path(tmp) / "d", "pipeline: frase corta (min_speech_ms)", "hola", 60_000, None),
            scenario_pipeline(Path(tmp) / "e", "pipeline: ASR final vacío", "", 0, None),
        ]
    # En el pipeline el seg_id lo asigna el motor: la expectativa usa el id que salió.
    for sc, st, en in zip(scenarios[1:], ("final", "discarded:short", "discarded:empty"), ("EN hola", "", "")):
        ids = list(dict.fromkeys(str(e["seg_id"]) for e in sc["events"]))
        if len(ids) != 1:
            sys.exit(f"{sc['name']}: esperaba 1 frase, salieron {ids}")
        sc["expect"] = [[ids[0], st, "hola", en]]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"source": "tests/integration/t55_dump_events.py", "scenarios": scenarios},
                              ensure_ascii=False, indent=1) + "\n")
    for sc in scenarios:
        print(f"{len(sc['events']):3d} eventos · {sc['name']}")
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
