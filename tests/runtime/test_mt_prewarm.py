"""T-58 GPU fría: prewarm de 1 token de Qwen al empezar la pausa VAD (componentes falsos)."""
import asyncio
import time

from runtime.pipeline import Engine

from test_speculative import FakeMT, run_segment


class PrewarmMT(FakeMT):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.prewarms = 0

    def prewarm(self):
        self.prewarms += 1
        time.sleep(0.01)


def _run(**kw):
    async def go():
        mt = PrewarmMT()
        e, ev = await run_segment(partial="hola mundo", final="hola mundo", spec=False, mt=mt, **kw)
        await e.close()
        return e, ev, mt
    return asyncio.run(go())


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("OPENTRADUCTOR_MT_PREWARM", raising=False)
    e, ev, mt = _run(mt_engine="qwen")
    assert mt.prewarms == 0 and e.prewarm_stats["started"] == 0


def test_prewarm_once_per_segment_and_output_unchanged(monkeypatch):
    monkeypatch.delenv("OPENTRADUCTOR_MT_PREWARM", raising=False)
    e, ev, mt = _run(mt_engine="qwen", mt_prewarm_on_pause=True)
    assert mt.prewarms == 1 and e.prewarm_stats["started"] == 1
    assert len(e.prewarm_stats["cost_ms"]) == 1
    assert mt.calls == ["hola mundo"]  # la MT real corre igual, una vez
    assert [f["text"] for k, f in ev if k == "mt_final"] == ["EN hola mundo"]


def test_env_forces_on_and_off(monkeypatch):
    monkeypatch.setenv("OPENTRADUCTOR_MT_PREWARM", "1")
    _, _, mt = _run(mt_engine="qwen")
    assert mt.prewarms == 1
    monkeypatch.setenv("OPENTRADUCTOR_MT_PREWARM", "0")
    _, _, mt = _run(mt_engine="qwen", mt_prewarm_on_pause=True)
    assert mt.prewarms == 0


def test_skipped_for_marian_and_parakeet(monkeypatch):
    monkeypatch.setenv("OPENTRADUCTOR_MT_PREWARM", "1")
    _, _, mt = _run(mt_engine="marian")
    assert mt.prewarms == 0
    _, _, mt = _run(mt_engine="qwen", backend="parakeet")  # ASR en el hilo MLX: no se le roba GPU
    assert mt.prewarms == 0


def test_hot_gpu_skips(monkeypatch):
    monkeypatch.setenv("OPENTRADUCTOR_MT_PREWARM", "1")

    async def go():
        from runtime.config import PipelineConfig
        e = Engine(PipelineConfig(mt_engine="qwen", asr_backend="moonshine-tiny"), lambda *a: None)
        e.mt = PrewarmMT()
        e.gpu.last_done = time.monotonic()
        from runtime.pipeline import Segment, StreamState
        st = StreamState(stream_id="s", vad=None)
        st.seg, st.in_speech, st.silence_ms = Segment(1, start_s=0.0), True, 100.0
        e._maybe_prewarm(st)
        e._maybe_prewarm(st)
        assert e.prewarm_stats == {"started": 0, "skipped_hot": 1, "error": 0, "cost_ms": []}
        e.gpu.shutdown(); e.cpu.shutdown()
    asyncio.run(go())
