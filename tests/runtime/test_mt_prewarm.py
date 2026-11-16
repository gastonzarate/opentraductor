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


class PrefillMT(PrewarmMT):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.prefills = []

    def prefill(self, partial):
        self.prefills.append(partial)


def _run_kv(**kw):
    async def go():
        mt = PrefillMT()
        e, ev = await run_segment(partial="hola mundo", final="hola mundo", spec=False, mt=mt, **kw)
        await e.close()
        return e, ev, mt
    return asyncio.run(go())


def test_kv_prefill_replaces_prewarm_and_output_unchanged(monkeypatch):
    monkeypatch.delenv("OPENTRADUCTOR_MT_PREWARM", raising=False)
    monkeypatch.delenv("OPENTRADUCTOR_MT_KV_PREFILL", raising=False)
    e, ev, mt = _run_kv(mt_engine="qwen", mt_kv_prefill_on_pause=True, mt_prewarm_on_pause=True)
    assert mt.prefills == ["hola mundo"] and mt.prewarms == 0
    assert e.prewarm_stats["kv_prefill"] == 1 and len(e.prewarm_stats["cost_ms"]) == 1
    assert mt.calls == ["hola mundo"]
    assert [f["text"] for k, f in ev if k == "mt_final"] == ["EN hola mundo"]


def test_kv_prefill_off_by_default_and_env(monkeypatch):
    monkeypatch.delenv("OPENTRADUCTOR_MT_PREWARM", raising=False)
    monkeypatch.delenv("OPENTRADUCTOR_MT_KV_PREFILL", raising=False)
    _, _, mt = _run_kv(mt_engine="qwen")
    assert mt.prefills == [] and mt.prewarms == 0
    monkeypatch.setenv("OPENTRADUCTOR_MT_KV_PREFILL", "1")
    _, _, mt = _run_kv(mt_engine="qwen")
    assert mt.prefills == ["hola mundo"]
    monkeypatch.setenv("OPENTRADUCTOR_MT_KV_PREFILL", "0")
    _, _, mt = _run_kv(mt_engine="qwen", mt_kv_prefill_on_pause=True)
    assert mt.prefills == []


def test_kv_prefill_runs_hot_but_not_busy(monkeypatch):
    monkeypatch.setenv("OPENTRADUCTOR_MT_KV_PREFILL", "1")

    async def go():
        from runtime.config import PipelineConfig
        from runtime.pipeline import Segment, StreamState
        e = Engine(PipelineConfig(mt_engine="qwen", asr_backend="moonshine-tiny"), lambda *a: None)
        e.mt = PrefillMT()
        e.gpu.last_done = time.monotonic()  # caliente: el prefill igual corre
        st = StreamState(stream_id="s", vad=None, last_partial="hola mundo")
        st.seg, st.in_speech, st.silence_ms = Segment(1, start_s=0.0), True, 100.0
        e._maybe_prewarm(st)
        await asyncio.gather(*list(e._prewarm_tasks))
        assert e.mt.prefills == ["hola mundo"]
        e.gpu.inflight = 1  # ocupada: no se encola delante del final
        st.seg = Segment(2, start_s=1.0)
        e._maybe_prewarm(st)
        assert e.mt.prefills == ["hola mundo"] and e.prewarm_stats["skipped_hot"] == 1
        e.gpu.inflight = 0
        e.gpu.shutdown(); e.cpu.shutdown()
    asyncio.run(go())


def test_kv_prefill_each_pause_after_early_empty_pause(monkeypatch):
    """Frase larga: la 1.ª pausa llega sin parcial (prewarm de 1 token); la pausa final igual hace prefill KV."""
    monkeypatch.setenv("OPENTRADUCTOR_MT_KV_PREFILL", "1")

    async def go():
        from runtime.config import PipelineConfig
        from runtime.pipeline import Segment, StreamState
        e = Engine(PipelineConfig(mt_engine="qwen", asr_backend="moonshine-tiny"), lambda *a: None)
        e.mt = PrefillMT()
        e.gpu.last_done = 0.0  # fría
        st = StreamState(stream_id="s", vad=None)
        st.seg, st.in_speech, st.silence_ms = Segment(1, start_s=0.0), True, 100.0
        st.last_speech_end_s = 0.5
        e._maybe_prewarm(st)  # pausa temprana, sin parcial
        await asyncio.gather(*list(e._prewarm_tasks))
        assert e.mt.prewarms == 1 and e.mt.prefills == [] and e.prewarm_stats["kv_no_partial"] == 1
        st.last_partial = "la dignidad no se"
        e._maybe_prewarm(st)  # misma pausa: no se reevalúa
        assert e.mt.prefills == []
        st.last_speech_end_s = 2.0  # pausa nueva, con parcial
        e._maybe_prewarm(st)
        await asyncio.gather(*list(e._prewarm_tasks))
        assert e.mt.prefills == ["la dignidad no se"]
        st.last_partial = "la dignidad no sé"  # cambia solo la última palabra: mismo prefijo estable
        st.last_speech_end_s = 2.5
        e._maybe_prewarm(st)
        assert e.mt.prefills == ["la dignidad no se"]
        st.last_partial = "la dignidad no se negocia nunca"
        st.last_speech_end_s = 3.5
        e._maybe_prewarm(st)
        await asyncio.gather(*list(e._prewarm_tasks))
        assert e.mt.prefills[-1] == "la dignidad no se negocia nunca" and e.mt.prewarms == 1
        assert e.prewarm_stats["kv_prefill"] == 2 and e.prewarm_stats["kv_words"] == [4, 6]
        e.gpu.shutdown(); e.cpu.shutdown()
    asyncio.run(go())
