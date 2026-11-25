"""T-55: contrato de eventos de frase (ver docs/api-local.md).

Secuencias que hacían desaparecer frases de la pantalla, del motor a /events (el modo incremental salió en T-70):
1. ASR final vacío publicaba `partial ""` y la fila quedaba en blanco;
2. frase bajo min_speech_ms se descartaba sin aviso.
"""
import asyncio

from tests.runtime.test_service import svc  # noqa: F401  fixture


def run(svc, kind, **f):
    asyncio.run(svc._on_engine_event(kind, f))


def published(svc):
    ev = []
    svc.subscribe(ev.append)
    return ev


def test_empty_asr_final_never_publishes_empty_partial_and_marks_row_discarded(svc):  # noqa: F811
    out = published(svc)
    run(svc, "asr_partial", seg_id=7, text="hola")
    run(svc, "asr_partial", seg_id=7, text="")
    run(svc, "asr_final", seg_id=7, text="")
    assert all(e["text_es"] for e in out if e["type"] == "partial")
    assert out[-1] == {"type": "discarded", "seg_id": "7", "text_es": "hola", "reason": "empty"}


def test_short_phrase_is_published_as_discarded(svc):  # noqa: F811
    out = published(svc)
    run(svc, "asr_partial", seg_id=3, text="eh")
    run(svc, "discarded", seg_id=3, text="eh", reason="short")
    assert out[-1] == {"type": "discarded", "seg_id": "3", "text_es": "eh", "reason": "short"}


def test_unknown_discard_reason_is_normalized_and_unseen_rows_carry_empty_text(svc):  # noqa: F811
    out = published(svc)
    run(svc, "discarded", seg_id="x", reason="raro")
    assert out[-1] == {"type": "discarded", "seg_id": "x", "text_es": "", "reason": "cancelled"}


# --------------------------------------------------------- pipeline baseline (componentes falsos)
from runtime.config import PipelineConfig  # noqa: E402
from runtime.pipeline import Engine  # noqa: E402
from tests.runtime.test_speculative import WIN, FakeASR, FakeMT, FakeTTS, FakeVAD  # noqa: E402


async def _pipeline_events(final, min_speech_ms, speech=10):
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


def test_pipeline_emits_discarded_short_instead_of_dropping_silently():
    ev = asyncio.run(_pipeline_events("hola", min_speech_ms=60_000))
    kinds = [k for k, _ in ev]
    assert "asr_final" not in kinds and "mt_final" not in kinds
    disc = [f for k, f in ev if k == "discarded"]
    assert len(disc) == 1 and disc[0]["reason"] == "short" and disc[0]["text"] == "hola"


def test_pipeline_emits_discarded_empty_instead_of_empty_final():
    ev = asyncio.run(_pipeline_events("", min_speech_ms=0))
    kinds = [k for k, _ in ev]
    assert "asr_final" not in kinds and "mt_final" not in kinds
    disc = [f for k, f in ev if k == "discarded"]
    assert len(disc) == 1 and disc[0]["reason"] == "empty" and disc[0]["text"] == "hola"
