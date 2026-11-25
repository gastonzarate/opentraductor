"""T-105 (BUG-10): con la Mac saturada el atraso del ASR queda acotado saltando frases enteras viejas."""
import asyncio

import pytest

from runtime.config import PipelineConfig, asr_skip_params
from runtime.pipeline import SR, Engine, Segment, StreamState

SCALE = 0.01  # 1 s de audio = 10 ms reales
CHUNK_S = 0.5
SEG_CHUNKS = 8  # frases de 4 s
SEGS = 75  # 300 s de habla continua


class SlowWorker:
    """Hilo del ASR falso: procesar d s de audio tarda d / speed (en tiempo escalado)."""

    def __init__(self, speed):
        self.speed = speed

    async def run(self, fn, *a):
        if a and isinstance(a[0], (bytes, bytearray)):
            await asyncio.sleep(len(a[0]) / 2 / SR / self.speed * SCALE)
        return fn(*a)


class FakeASR:
    def __init__(self):
        self.added_s = 0.0

    def start(self): pass

    def add(self, pcm):
        self.added_s += len(pcm) / 2 / SR
        return "hola mundo"

    def finish(self): return "hola mundo."


async def _session(speed):
    cfg = PipelineConfig(asr_backend="moonshine-tiny", speculative_mt_tts=False, min_speech_ms=0)
    finals = []

    async def emit(kind, f):
        if kind == "asr_final":
            finals.append(f["seg_id"])

    e = Engine(cfg, emit)
    e.asr = FakeASR()
    e.asr_worker = SlowWorker(speed)
    e._out_q = asyncio.Queue()
    st = StreamState(stream_id="s", vad=None)
    st.asr_q = asyncio.Queue()
    e._streams["s"] = st
    task = asyncio.ensure_future(e._asr_loop(st))
    peak, pos = 0.0, 0.0
    for k in range(1, SEGS + 1):
        seg = Segment(k, start_s=pos)
        await st.asr_q.put(("start", (seg, b"")))
        for _ in range(SEG_CHUNKS):
            await asyncio.sleep(CHUNK_S * SCALE)  # el audio llega en tiempo real (escalado)
            pos += CHUNK_S
            st.asr_buf.extend(bytes(int(CHUNK_S * SR) * 2))
            await e._queue_audio(st, pos)
            peak = max(peak, st.asr_pending_s)
        seg.end_s = pos
        await st.asr_q.put(("end", seg))
    await st.asr_q.put(("stop", None))
    await task
    return e, st, peak, finals


def test_asr_at_70pct_realtime_backlog_stays_near_threshold(monkeypatch):
    monkeypatch.setenv("OPENTRADUCTOR_SIMUL_VOICE", "0")
    monkeypatch.delenv("OPENTRADUCTOR_ASR_SKIP_OLD_S", raising=False)
    monkeypatch.delenv("OPENTRADUCTOR_ASR_SKIP_TARGET_S", raising=False)
    e, st, peak, finals = asyncio.run(_session(0.7))
    # Sin salto el backlog llegaría a ~90 s (0,3 s por s de habla en 300 s). Con salto: umbral 20 s + una frase.
    assert peak <= 20.0 + SEG_CHUNKS * CHUNK_S + 1.0
    assert st.asr_skipped_s > 20.0 and st.asr_skip_events >= 1 and st.asr_skip_segs >= 5
    assert st.asr_pending_s == pytest.approx(0.0)
    # Lo saltado son frases enteras: lo reconocido + lo saltado = todo el audio, y cada frase reconocida está entera.
    total = SEGS * SEG_CHUNKS * CHUNK_S
    assert e.asr.added_s + st.asr_skipped_s == pytest.approx(total)
    assert len(finals) + st.asr_skip_segs == SEGS
    b = e.backlog()
    assert b["asr_skipped_s"] == pytest.approx(st.asr_skipped_s, abs=0.01) and b["asr_skip_events"] == st.asr_skip_events


def test_asr_skip_off_by_env_keeps_everything(monkeypatch):
    monkeypatch.setenv("OPENTRADUCTOR_SIMUL_VOICE", "0")
    monkeypatch.setenv("OPENTRADUCTOR_ASR_SKIP_OLD_S", "0")
    e, st, peak, finals = asyncio.run(_session(0.7))
    assert peak > 40.0  # el atraso crece sin cota (BUG-10)
    assert st.asr_skipped_s == 0 and st.asr_skip_events == 0 and len(finals) == SEGS


def test_asr_fast_enough_skips_nothing(monkeypatch):
    """Carga normal: el ASR va más rápido que el tiempo real y no se salta nada."""
    monkeypatch.setenv("OPENTRADUCTOR_SIMUL_VOICE", "0")
    monkeypatch.delenv("OPENTRADUCTOR_ASR_SKIP_OLD_S", raising=False)
    e, st, peak, finals = asyncio.run(_session(3.0))
    assert st.asr_skipped_s == 0 and st.asr_skip_events == 0 and len(finals) == SEGS
    assert e.backlog()["asr_skipped_s"] == 0


def test_asr_skip_params_env(monkeypatch):
    cfg = PipelineConfig(asr_backend="moonshine-tiny")
    for k in ("OPENTRADUCTOR_ASR_SKIP_OLD_S", "OPENTRADUCTOR_ASR_SKIP_TARGET_S"):
        monkeypatch.delenv(k, raising=False)
    assert asr_skip_params(cfg) == (20.0, 3.0)
    for off in ("", "0", "-1"):
        monkeypatch.setenv("OPENTRADUCTOR_ASR_SKIP_OLD_S", off)
        assert asr_skip_params(cfg) is None
    monkeypatch.setenv("OPENTRADUCTOR_ASR_SKIP_OLD_S", "30")
    monkeypatch.setenv("OPENTRADUCTOR_ASR_SKIP_TARGET_S", "5")
    assert asr_skip_params(cfg) == (30.0, 5.0)
