"""T-83: al cerrar frase, el ASR recibe los primeros asr_tail_keep_ms de la cola «silenciosa».

El VAD da silencio en finales suaves («policia(l)», «nazi(smo)») y, si se descarta toda la cola (T-58), Moonshine
pierde la última sílaba. Se conserva un margen corto y el resto de la cola se sigue descartando.
"""
import asyncio

import pytest

from runtime.config import PipelineConfig
from runtime.pipeline import Engine, asr_tail_keep_ms

from test_asr_spec_finish import PatternVAD
from test_speculative import WIN, FakeMT, FakeTTS


class ByteASR:
    """Cuenta todos los bytes que recibe cada segmento, voz o silencio."""

    def __init__(self):
        self.n, self.finishes = 0, []

    def start(self):
        self.n = 0

    def add(self, pcm):
        self.n += len(pcm)
        return "el granizo destruyó la plantación"

    def finish(self):
        self.finishes.append(self.n)
        return "el granizo destruyó la plantación"


async def _run(pattern, **cfg_kw):
    events = []

    async def emit(kind, f):
        events.append((kind, f))

    # Espera fija de 160 ms (5 ventanas) y chunks de 160 ms: la voz (10 ventanas) entra en 2 chunks justos y
    # la cola de 5 ventanas de silencio arranca en last_speech_end_s, que es el caso que T-58 descarta.
    cfg = PipelineConfig(asr_backend="moonshine-tiny", speculative_mt_tts=False, endpoint_adaptive=False,
                         endpoint_silence_ms=160, asr_chunk_ms=160, min_speech_ms=0,
                         mt_kv_prefill_on_pause=False, asr_spec_finish_on_pause=False, mt_stream_tts=False,
                         **cfg_kw)
    e = Engine(cfg, emit)
    e.asr, e.mt, e.tts = ByteASR(), FakeMT(), FakeTTS()
    e.vad_proto = PatternVAD(pattern)
    e._out_q = asyncio.Queue(maxsize=8)
    e._out_task = asyncio.ensure_future(e._speaker_loop())
    await e.start_stream("s")
    for i in range(len(pattern) + 20):
        voice = i < len(pattern) and pattern[i]
        await e.push("s", (b"\x01" if voice else b"\x00") * (2 * WIN), 0.0)
        await asyncio.sleep(0.02)
    await e.end_stream("s")
    await e.close()
    return e, events


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    monkeypatch.delenv("OPENTRADUCTOR_ASR_TAIL_KEEP_MS", raising=False)


VOICE = [1] * 10


def test_default_keeps_96ms():
    assert PipelineConfig().asr_tail_keep_ms == 96.0
    e, _ = asyncio.run(_run(VOICE))
    assert e.asr.finishes[0] == (10 + 3) * 2 * WIN  # voz + 3 ventanas (96 ms) de la cola


def test_keep_zero_is_t58_behaviour():
    e, _ = asyncio.run(_run(VOICE, asr_tail_keep_ms=0.0))
    assert e.asr.finishes[0] == 10 * 2 * WIN


def test_keep_larger_than_tail_sends_whole_tail():
    e, _ = asyncio.run(_run(VOICE, asr_tail_keep_ms=1000.0))
    assert e.asr.finishes[0] == (10 + 5) * 2 * WIN  # la cola entera hasta el endpoint, nada más


def test_skip_off_still_sends_everything():
    e, _ = asyncio.run(_run(VOICE, asr_skip_silent_tail=False, asr_tail_keep_ms=0.0))
    assert e.asr.finishes[0] == (10 + 5) * 2 * WIN


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("OPENTRADUCTOR_ASR_TAIL_KEEP_MS", "x")
    assert asr_tail_keep_ms(PipelineConfig()) == 96.0
    monkeypatch.setenv("OPENTRADUCTOR_ASR_TAIL_KEEP_MS", "0")
    assert asr_tail_keep_ms(PipelineConfig()) == 0.0
    e, _ = asyncio.run(_run(VOICE, asr_tail_keep_ms=96.0))
    assert e.asr.finishes[0] == 10 * 2 * WIN  # el env manda
