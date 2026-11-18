"""Especulación MT + primer fragmento TTS: aciertos exactos, fallos, stop/restart/error, sin pérdidas ni duplicados.
Todo con componentes falsos (sin modelos ni inferencia)."""
import asyncio
import time
import types

import pytest

from runtime import components as K
from runtime.config import PipelineConfig
from runtime.pipeline import Engine

WIN = 512


class FakeVAD:
    """Voz en las ventanas [0, speech_windows); silencio después."""

    def __init__(self, speech_windows):
        self.n, self.speech = 0, speech_windows

    def reset(self): pass

    def reset_states(self): pass

    def probs(self, pcm):
        out = []
        for i in range(len(pcm) // 2 // WIN):
            out.append(((i + 1) * WIN, 0.9 if self.n < self.speech else 0.0))
            self.n += 1
        return out


class FakeASR:
    def __init__(self, partial, final):
        self.partial, self.final = partial, final

    def start(self): pass
    def add(self, pcm): return self.partial
    def finish(self): return self.final


class FakeMT:
    def __init__(self, delay=0.02, fail=False):
        self.calls, self.delay, self.fail = [], delay, fail

    def translate(self, t, context=()):  # T-77: el pipeline pasa la historia (contexto on por defecto)
        self.calls.append(t)
        time.sleep(self.delay)
        if self.fail and len(self.calls) == 1:
            raise RuntimeError("mt boom")
        return "EN " + t.replace(" ", " ")


class FakeTTS:
    def __init__(self):
        self.synth_calls = []  # (cláusulas, first)

    def plan(self, t): return t.split(" | ")

    def synth_clauses(self, clauses, first=True):
        self.synth_calls.append((list(clauses), first))
        for c in clauses:
            yield K.TTSChunk(pcm=c.encode(), sample_rate=24000, text=c)

    def synth(self, t):
        yield from self.synth_clauses(self.plan(t), True)


async def run_segment(*, partial, final, spec=True, backend="moonshine-tiny", mt=None, tts=None, speech=10,
                      stop_before_final=False, finish=True, settle=False, **cfg_kw):
    events = []

    async def emit(kind, f):
        events.append((kind, f))

    cfg = PipelineConfig(asr_backend=backend, speculative_mt_tts=spec, endpoint_silence_ms=160,
                         asr_chunk_ms=160, min_speech_ms=0, **cfg_kw)
    e = Engine(cfg, emit)
    e.asr, e.mt, e.tts = FakeASR(partial, final), mt or FakeMT(), tts or FakeTTS()
    e.vad_proto = FakeVAD(speech)
    e._out_q = asyncio.Queue(maxsize=8)
    e._out_task = asyncio.ensure_future(e._speaker_loop())
    await e.start_stream("s")
    pcm = bytes(2 * WIN)
    for _ in range(speech + 8):  # voz y luego silencio hasta pasar el endpoint
        await e.push("s", pcm, 0.0)
        await asyncio.sleep(0.03)
        if stop_before_final and e._spec and any(x.task and not x.task.done() for x in e._spec.values()):
            return e, events
        if settle:  # determinista bajo carga: el endpoint espera a que la especulación termine
            pend = [x.task for x in e._spec.values() if x.task]
            if pend:
                await asyncio.wait(pend, timeout=10)
    if finish:
        await e.end_stream("s")
    return e, events


def kinds(events): return [k for k, _ in events]


def chunk_texts(events): return [f["pcm"].decode() for k, f in events if k == "tts_chunk"]


def test_exact_hit_uses_cache_and_loses_nothing():
    async def go():
        mt, tts = FakeMT(), FakeTTS()
        e, ev = await run_segment(partial="hola mundo", final="hola mundo", mt=mt, tts=tts, settle=True)
        # una sola MT (la especulativa) y la 1.ª cláusula se sintetizó una vez
        assert mt.calls == ["hola mundo"]
        assert tts.synth_calls == [(["EN hola mundo"], True)]
        assert e.spec_stats["hit"] == 1 and e.spec_stats["started"] == 1 and e.spec_stats["overlap_estimate_ms"] > 0
        assert [f for k, f in ev if k == "mt_final"][0]["spec"] == "hit"
        # nada especulativo se emitió antes de asr_final
        assert kinds(ev).index("asr_final") < kinds(ev).index("mt_final") < kinds(ev).index("tts_chunk")
        assert chunk_texts(ev) == ["EN hola mundo"]
        await e.close()
    asyncio.run(go())


def test_multi_clause_no_loss_no_duplicates_and_same_as_normal_path():
    async def go():
        tts = FakeTTS()
        e, ev = await run_segment(partial="a | b | c", final="a | b | c", tts=tts, settle=True)
        spec_chunks = chunk_texts(ev)
        assert spec_chunks == ["EN a", "b", "c"]  # MT convierte 'a' solo en el 1.º; el resto va tal cual
        assert tts.synth_calls == [(["EN a", "b", "c"][:1], True), (["b", "c"], False)]
        await e.close()
        # misma salida que sin especulación
        e2, ev2 = await run_segment(partial="a | b | c", final="a | b | c", spec=False)
        assert chunk_texts(ev2) == spec_chunks
        assert all("spec" not in f for k, f in ev2 if k == "mt_final")
        assert e2.spec_stats["started"] == 0
        await e2.close()
    asyncio.run(go())


def test_changed_final_text_discards_cache_and_runs_normal_path():
    async def go():
        mt, tts = FakeMT(), FakeTTS()
        e, ev = await run_segment(partial="hola", final="hola mundo", mt=mt, tts=tts, settle=True)
        assert mt.calls == ["hola", "hola mundo"]  # especulativa + real
        assert e.spec_stats["miss_text"] == 1 and e.spec_stats["hit"] == 0 and e.spec_stats["discarded_observed_ms"] > 0
        assert chunk_texts(ev) == ["EN hola mundo"]  # solo el texto final; nada del parcial
        assert [f for k, f in ev if k == "mt_final"][0]["spec"] == "miss"
        assert all(c != "EN hola" for c in chunk_texts(ev))
        await e.close()
    asyncio.run(go())


def test_normalization_must_match_exactly():
    async def go():
        e, ev = await run_segment(partial="hola mundo", final="hola mundo", asr_normalize_text=True)
        assert e.spec_stats["hit"] == 1  # el parcial se normalizó igual que el final
        await e.close()
    asyncio.run(go())


def test_parakeet_shares_mlx_thread_so_only_mt_is_speculated():
    async def go():
        mt, tts = FakeMT(), FakeTTS()
        e, ev = await run_segment(partial="x | y", final="x | y", backend="parakeet", mt=mt, tts=tts)
        assert mt.calls == ["x | y"] and e.spec_stats["hit"] == 1
        assert tts.synth_calls == [(["EN x", "y"], True)]  # TTS completo en la ruta normal, una sola vez
        assert chunk_texts(ev) == ["EN x", "y"]
        await e.close()
    asyncio.run(go())


def test_speculative_error_falls_back_to_normal_path():
    async def go():
        mt = FakeMT(fail=True)
        e, ev = await run_segment(partial="hola", final="hola", mt=mt)
        assert e.spec_stats["error"] == 1 and e.spec_stats["hit"] == 0
        assert mt.calls == ["hola", "hola"]
        assert chunk_texts(ev) == ["EN hola"] and "error" not in kinds(ev)
        await e.close()
    asyncio.run(go())


def test_close_during_speculation_emits_nothing_and_counts_cancelled():
    async def go():
        mt = FakeMT(delay=0.4)
        e, ev = await run_segment(partial="hola", final="hola", mt=mt, stop_before_final=True)
        assert len(e._spec) == 1  # en vuelo
        await e.close()
        assert e._spec == {} and e.spec_stats["cancelled"] == 1
        assert "mt_final" not in kinds(ev) and "tts_chunk" not in kinds(ev)
    asyncio.run(go())


def test_busy_confirmed_work_has_priority():
    async def go():
        mt = FakeMT()
        events = []

        async def emit(k, f): events.append((k, f))

        e = Engine(PipelineConfig(asr_backend="moonshine-tiny", speculative_mt_tts=True, endpoint_silence_ms=160, min_speech_ms=0), emit)
        e.asr, e.mt, e.tts, e.vad_proto = FakeASR("hola", "hola"), mt, FakeTTS(), FakeVAD(10)
        e._out_q = asyncio.Queue(maxsize=8)
        e._speaking = True  # hay audio confirmado en curso
        await e.start_stream("s")
        for _ in range(16):
            await e.push("s", bytes(2 * WIN), 0.0)
            await asyncio.sleep(0.03)
        assert mt.calls == [] and e.spec_stats["skipped_busy"] >= 1 and e.spec_stats["started"] == 0
        e._speaking = False
        await e.close()
    asyncio.run(go())


def test_default_is_off():
    c = PipelineConfig()
    assert c.speculative_mt_tts is False and 0 < c.spec_silence_ms < c.endpoint_silence_ms


def test_restart_with_final_and_claim_in_flight_drains_old_session_first(monkeypatch):
    """Reinicio del mismo id mientras el segmento viejo ya fue finalizado y su claim espera una especulación lenta:
    la sesion vieja se drena completa ANTES de crear la nueva; nada viejo se mezcla con la nueva."""
    monkeypatch.setattr(K, "SileroVAD", lambda: FakeVAD(10))

    async def go():
        mt, tts = FakeMT(delay=0.3), FakeTTS()
        e, ev = await run_segment(partial="viejo", final="viejo", mt=mt, tts=tts, finish=False)
        # el segmento viejo ya esta en la cola de salida / claim en curso
        old_task = e._streams["s"].asr_task
        old_gen = e._streams["s"].gen
        await e.start_stream("s")  # reinicio con id vivo
        assert old_task.done()  # el ASR viejo se detuvo (no queda huerfano)
        assert chunk_texts(ev) == ["EN viejo"]  # la salida vieja se emitio completa, una sola vez
        assert e._streams["s"].gen == old_gen + 1 and e._spec == {}
        n = len(ev)
        await asyncio.sleep(0.2)
        assert len(ev) == n  # nada viejo aparece despues del reinicio
        await e.close()
    asyncio.run(go())


def test_stale_generation_never_claims_new_entry():
    async def go():
        events = []

        async def emit(k, f): events.append((k, f))

        e = Engine(PipelineConfig(asr_backend="moonshine-tiny", speculative_mt_tts=True), emit)
        e.mt, e.tts = FakeMT(), FakeTTS()
        from runtime.pipeline import Segment, SpecEntry
        ent = SpecEntry("s", 1, "hola", gen=2, fp=e._spec_fp())
        ent.text_en = "EN hola"
        e._spec[("s", 2, 1)] = ent
        seg_old = Segment(1, 0.0, 1.0, text_es="hola", gen=1)  # misma id/seg_id, generacion vieja
        got, state = await e._spec_claim("s", seg_old)
        assert got is None and state == "none" and ("s", 2, 1) in e._spec
        await e.close()
    asyncio.run(go())


def test_config_change_invalidates_cache():
    async def go():
        mt, tts = FakeMT(), FakeTTS()
        events = []

        async def emit(k, f): events.append((k, f))

        e = Engine(PipelineConfig(asr_backend="moonshine-tiny", speculative_mt_tts=True, endpoint_silence_ms=160, min_speech_ms=0), emit)
        e.asr, e.mt, e.tts, e.vad_proto = FakeASR("hola", "hola"), mt, tts, FakeVAD(10)
        e._out_q = asyncio.Queue(maxsize=8)
        e._out_task = asyncio.ensure_future(e._speaker_loop())
        await e.start_stream("s")
        for _ in range(12):
            await e.push("s", bytes(2 * WIN), 0.0)
            await asyncio.sleep(0.03)
        assert e.spec_stats["started"] == 1
        e.cfg.speed = 1.3  # cambia la voz/config antes del final
        for _ in range(8):
            await e.push("s", bytes(2 * WIN), 0.0)
            await asyncio.sleep(0.03)
        await e.end_stream("s")
        assert e.spec_stats["miss_config"] == 1 and e.spec_stats["hit"] == 0
        assert [f for k, f in events if k == "mt_final"][0]["spec"] == "miss"
        assert chunk_texts(events) == ["EN hola"] and mt.calls == ["hola", "hola"]
        await e.close()
    asyncio.run(go())


def test_known_miss_does_not_wait_or_launch_new_tts():
    async def go():
        mt, tts = FakeMT(delay=0.25), FakeTTS()
        events = []

        async def emit(k, f): events.append((k, f))

        e = Engine(PipelineConfig(asr_backend="moonshine-tiny", speculative_mt_tts=True, endpoint_silence_ms=160, min_speech_ms=0), emit)
        e.asr, e.mt, e.tts, e.vad_proto = FakeASR("hola", "hola mundo"), mt, tts, FakeVAD(10)
        e._out_q = asyncio.Queue(maxsize=8)
        e._out_task = asyncio.ensure_future(e._speaker_loop())
        await e.start_stream("s")
        for _ in range(18):
            await e.push("s", bytes(2 * WIN), 0.0)
            await asyncio.sleep(0.03)
        await e.end_stream("s")
        # el parcial viejo no llego a sintetizarse: solo se sintetizo el texto final
        assert [c for c, _ in tts.synth_calls] == [["EN hola mundo"]]
        assert e.spec_stats["miss_text"] == 1 and e.spec_stats["inflight_unmeasured"] >= 0
        assert chunk_texts(events) == ["EN hola mundo"]
        await e.close()
    asyncio.run(go())


def test_metric_names_are_honest():
    m = Engine(PipelineConfig(), None).spec_metrics()
    assert "wasted_ms" not in m and "saved_ms" not in m
    assert {"discarded_observed_ms", "overlap_estimate_ms", "inflight_unmeasured", "miss_config"} <= m.keys()


class SlowTTS(FakeTTS):
    def __init__(self, delay):
        super().__init__()
        self.delay = delay

    def synth_clauses(self, clauses, first=True):
        for c in clauses:
            time.sleep(self.delay)
            yield K.TTSChunk(pcm=c.encode(), sample_rate=24000, text=c)


def test_timeout_then_restart_emits_no_old_audio_slow_mt(monkeypatch):
    monkeypatch.setattr(K, "SileroVAD", lambda: FakeVAD(10))

    async def go():
        mt = FakeMT(delay=0.5)
        e, ev = await run_segment(partial="viejo", final="viejo", spec=False, mt=mt, finish=False,
                                  end_stream_drain_timeout_s=0.1)
        await e.end_stream("s")  # el drenado expira (MT lenta en curso)
        assert any(k == "error" and f.get("error_kind") == "end_stream_drain_timeout" for k, f in ev)
        n = len(ev)
        e.asr = FakeASR("nuevo", "nuevo")
        e.vad_proto = FakeVAD(10)
        await e.start_stream("s")  # nueva generacion con el speaker loop viejo aun ocupado
        await asyncio.sleep(0.8)  # la MT vieja termina en segundo plano
        old = [f for k, f in ev[n:] if k in ("mt_final", "tts_chunk") and "viejo" in str(f.get("text", f.get("pcm")))]
        assert old == [] and e.stale_dropped >= 1
        assert "EN viejo" not in chunk_texts(ev)
        # la sesion nueva funciona despues (con tiempo de drenado normal)
        e.cfg.end_stream_drain_timeout_s = 5.0
        for _ in range(18):
            await e.push("s", bytes(2 * WIN), 0.0)
            await asyncio.sleep(0.03)
        await e.end_stream("s")
        assert chunk_texts(ev) == ["EN nuevo"]
        await e.close()
    asyncio.run(go())


def test_timeout_with_slow_tts_stops_emitting_after_close():
    async def go():
        tts = SlowTTS(0.25)
        e, ev = await run_segment(partial="a | b | c | d", final="a | b | c | d", spec=False, tts=tts, finish=False,
                                  end_stream_drain_timeout_s=0.3)
        await e.end_stream("s")
        n = len(ev)
        await asyncio.sleep(1.2)  # el TTS viejo sigue sintetizando en el hilo
        assert len(ev) == n, "no debe emitirse audio ni eventos de la sesion cerrada despues del timeout"
        assert e.stale_dropped >= 1
        # no se sintetizan todas las clausulas: el hilo corta cuando la sesion deja de estar vigente
        assert len(chunk_texts(ev)) < 4
        await e.close()
    asyncio.run(go())


def test_queued_segments_of_closed_session_are_purged(monkeypatch):
    async def go():
        events = []

        async def emit(k, f): events.append((k, f))

        e = Engine(PipelineConfig(asr_backend="moonshine-tiny"), emit)
        from runtime.pipeline import Segment
        e._out_q = asyncio.Queue(maxsize=8)
        e._live_gen = {"a": 1, "b": 1}
        for sid in ("a", "b", "a"):
            e._out_q.put_nowait((sid, Segment(1, 0.0, 1.0, text_es="x", gen=1)))
        assert e._purge_queue("a", 1) == 2 and e._out_q.qsize() == 1
        assert e._out_q.get_nowait()[0] == "b"
    asyncio.run(go())


def test_mt_final_reports_compute_ms_only_without_spec_hit():
    """T-58: mt_compute_ms (dentro del hilo) acompaña a mt_ms en la ruta normal; en un hit no se recalcula."""
    async def go():
        e, ev = await run_segment(partial="hola", final="hola mundo", spec=False)
        f = [f for k, f in ev if k == "mt_final"][0]
        assert 0 <= f["mt_compute_ms"] <= f["mt_ms"]
        await e.close()
        e, ev = await run_segment(partial="hola mundo", final="hola mundo", settle=True)
        f = [f for k, f in ev if k == "mt_final"][0]
        assert f["spec"] == "hit" and "mt_compute_ms" not in f
        await e.close()
    asyncio.run(go())
