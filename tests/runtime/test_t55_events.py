"""T-55: contrato de eventos de frase (memory/decisiones/t55-contrato-eventos-frase.md).

Las 4 secuencias que hacían desaparecer frases de la pantalla, del motor a /events:
1. incremental: el parcial «live» de la frase N se pisaba con la N+1 (un solo id live);
2. baseline: ASR final vacío publicaba `partial ""` y la fila quedaba en blanco;
3. baseline: frase bajo min_speech_ms se descartaba sin aviso;
4. incremental: unidad cancelada o sin PCM quedaba sin traducción y sin explicación.
"""
import asyncio
from types import SimpleNamespace

from runtime.inc_app.bridge import IncrementalEngineBridge
from tests.runtime.test_service import svc  # noqa: F401  fixture


def run(svc, kind, **f):
    asyncio.run(svc._on_engine_event(kind, f))


def published(svc):
    ev = []
    svc.subscribe(ev.append)
    return ev


class Recorder:
    def __init__(self):
        self.events = []

    async def __call__(self, kind, f):
        self.events.append((kind, dict(f)))


def make_bridge():
    rec = Recorder()
    b = IncrementalEngineBridge(SimpleNamespace(input_queue_chunks=8), adapter=None, emit=rec)
    b._sid = "s1"
    return b, rec


def ev(kind, **kw):
    return SimpleNamespace(kind=kind, stream_id="s1", **kw)


def test_incremental_each_phrase_has_own_live_row_and_commit_reassigns_it(svc):  # noqa: F811
    b, rec = make_bridge()

    async def flow():
        await b._handle(ev("asr_partial", text="hola", audio_pos_s=0.5))
        await b._handle(ev("asr_final", text="hola mundo", commit_id="c1"))
        await b._handle(ev("asr_partial", text="cómo", audio_pos_s=1.5))
    asyncio.run(flow())

    out = published(svc)
    for kind, f in rec.events:
        run(svc, kind, **f)
    partials = [e for e in out if e["type"] == "partial"]
    assert [p["seg_id"] for p in partials] == ["s1:live:0", "c1", "s1:live:1"]
    assert partials[1]["from_seg_id"] == "s1:live:0"  # la UI reasigna la fila, no crea otra
    assert "from_seg_id" not in partials[2]
    assert partials[2]["text_es"] == "cómo"  # la frase N+1 no pisa la fila de N


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


def test_cancelled_or_pcm_less_unit_is_published_as_discarded_with_its_text(svc):  # noqa: F811
    b, rec = make_bridge()

    async def flow():
        await b._handle(ev("asr_partial", text="una", audio_pos_s=0.2))
        await b._handle(ev("asr_final", text="una frase", commit_id="c1"))
        await b._handle(ev("unit_cancelled", commit_id="c1"))
        await b._handle(ev("asr_final", text="otra frase", commit_id="c2"))
        await b._handle(ev("unit_done", commit_id="c2", unit_n_chunks=0))
    asyncio.run(flow())

    out = published(svc)
    for kind, f in rec.events:
        run(svc, kind, **f)
    disc = [e for e in out if e["type"] == "discarded"]
    assert disc == [
        {"type": "discarded", "seg_id": "c1", "text_es": "una frase", "reason": "cancelled"},
        {"type": "discarded", "seg_id": "c2", "text_es": "otra frase", "reason": "cancelled"},
    ]
    assert not [e for e in out if e["type"] == "segment"]


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
