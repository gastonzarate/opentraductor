import asyncio
import json
from types import SimpleNamespace

import pytest

from benchmark import audio as A
from benchmark import device_loopback as DL
from benchmark.adapters.device_loopback import DeviceLoopbackAdapter
from benchmark.contracts import Adapter, AdapterEvent, AdapterMode, EventKind

THR = -50.0
MS = 1_000_000


def blk(t_ms, rms=-90.0, dur_ms=10, lat_ms=5):
    return DL.Block(t_ms * MS, (t_ms - lat_ms) * MS, dur_ms * MS, rms, lat_ms * MS)


def silent_blocks(a_ms, b_ms):
    return [blk(t) for t in range(a_ms, b_ms, 10)]


def clip(begin_ms, end_ms, write_ms, offset=0.0, nonsilent_s=0.5, commit="c1"):
    c = DL.Clip("s1", begin_ms * MS, end_ms * MS)
    c.writes.append(DL.Write(write_ms * MS, 1000, 24000, offset, nonsilent_s, commit))
    return c


# ---------------------------------------------------------------- análisis puro
def test_onset_associated_after_write_with_uncertainty_and_commit():
    blocks = silent_blocks(0, 1000) + [blk(t, -20) for t in range(1000, 1500, 10)] + silent_blocks(1500, 2000)
    res = DL.analyze(blocks, [clip(500, 1800, 700)], threshold_dbfs=THR, unc_ms=12.0)
    c = res["clips"][0]
    assert c["status"] == "ok" and c["unambiguous"] and c["commit_id"] == "c1"
    assert c["device_minus_first_write_ms"] == pytest.approx(295.0)  # bloque en t=1000 con 5 ms de latencia de entrada
    assert c["uncertainty_ms"] == 12.0 and c["device_onset_raw_callback_ns"] == 1000 * MS
    assert res["foreign_audio_blocks"] == 0


def test_audio_before_first_write_is_contamination_not_latency():
    blocks = silent_blocks(0, 400) + [blk(t, -20) for t in range(400, 450, 10)] + silent_blocks(450, 800) \
        + [blk(t, -20) for t in range(800, 900, 10)]
    c = DL.analyze(blocks, [clip(300, 1000, 700)], threshold_dbfs=THR, unc_ms=10.0)["clips"][0]
    assert c["status"] == "contaminated_before_write" and not c["unambiguous"] and "device_onset_ns" not in c


def test_no_onset_and_silent_clip_and_foreign_between_clips():
    blocks = silent_blocks(0, 3000)
    assert DL.analyze(blocks, [clip(0, 900, 100)], threshold_dbfs=THR, unc_ms=10)["clips"][0]["status"] == "no_device_onset"
    silent_clip = DL.Clip("s1", 0, 900 * MS)
    silent_clip.writes.append(DL.Write(100 * MS, 10, 24000, None, 0.0, "c"))
    assert DL.analyze(blocks, [silent_clip], threshold_dbfs=THR, unc_ms=10)["clips"][0]["status"] == "no_audible_write"
    loud_gap = silent_blocks(0, 1000) + [blk(1100, -10)] + silent_blocks(1110, 2000)
    r = DL.analyze(loud_gap, [clip(0, 500, 100)], threshold_dbfs=THR, unc_ms=10)
    assert r["foreign_audio_blocks"] == 1 and r["clips"][0]["status"] == "no_device_onset"


def test_energy_mismatch_flags_overlapping_foreign_audio():
    blocks = [blk(t, -20) for t in range(100, 1500, 10)]  # 1.4 s capturados, 0.2 s escritos
    c = DL.analyze(blocks, [clip(0, 1600, 100, nonsilent_s=0.2)], threshold_dbfs=THR, unc_ms=10)["clips"][0]
    assert c["status"] == "contamination_suspected_energy" and not c["unambiguous"]


def test_uncertainty_has_block_and_jitter():
    bs = [blk(t) for t in (0, 10, 20, 35, 45)]  # un salto de 5 ms de jitter
    u = DL.uncertainty_ms(bs, 10.0)
    assert u["block_ms"] == 10.0 and u["callback_jitter_p95_ms"] >= 0 and u["total_ms"] >= 10.0


def _rep(**kw):
    c = {"stream_id": "s1", "status": "ok", "commit_id": "s1:0", "device_onset_ns": 5_000 * MS,
         "device_onset_raw_callback_ns": 5_020 * MS, "device_minus_first_write_ms": 40.0, "uncertainty_ms": 12.0}
    return {"clips": [c], **kw}


def _trace(segments=1, commit="s1:0", continuous=False):
    return {"stream_id": "s1", "t0_ns": 1_000 * MS, "continuous": continuous,
            "segments": [{"end_s": 3.0, "boundary_method": "estimated", "boundary_uncertainty_s": 0.3}] * segments,
            "events": [{"kind": "tts_chunk", "recv_ns": 4_900 * MS, "first_nonsilent_offset_s": 0.0, "commit_id": commit}]}


def test_merge_keeps_raw_callback_and_estimated_onset_and_boundary_uncertainty():
    row = DL.merge_with_traces(_rep(), [_trace()])[0]
    assert row["end_to_device_estimated_ms"] == pytest.approx(1000.0) and row["end_to_device_raw_callback_ms"] == pytest.approx(1020.0)
    assert row["available_to_device_ms"] == pytest.approx(100.0) and row["input_latency_correction_ms"] == pytest.approx(20.0)
    assert row["boundary_method"] == "estimated" and row["boundary_uncertainty_s"] == 0.3


def test_merge_refuses_multi_segment_continuous_and_commit_mismatch():
    r = DL.merge_with_traces(_rep(), [_trace(segments=3)])[0]
    assert r["status"] == "not_attributable_multi_segment" and "end_to_device_estimated_ms" not in r
    assert DL.merge_with_traces(_rep(), [_trace(continuous=True)])[0]["status"] == "not_attributable_multi_segment"
    m = DL.merge_with_traces(_rep(), [_trace(commit="s1:7")])[0]
    assert m["status"] == "commit_mismatch" and "end_to_device_estimated_ms" not in m
    assert DL.merge_with_traces({"clips": [{"stream_id": "s1", "status": "contaminated_before_write"}]}, [_trace()])[0].get("end_to_device_estimated_ms") is None


def test_invalidating_capture_status_or_playback_drops_never_reported_as_ok():
    assert DL.validity(0, {"dropped_s": 0.0, "status_events": 0, "underruns": 0}, 0)["valid"]
    for cap, pl, late in ((2, {}, 0), (0, {"dropped_s": 0.5}, 0), (0, {"status_events": 1}, 0), (0, {}, 1)):
        v = DL.validity(cap, pl, late)
        assert not v["valid"] and v["invalidating_reasons"]
        row = DL.merge_with_traces(_rep(validity=v), [_trace()])[0]
        assert row["status"] == "invalid_measurement" and "end_to_device_estimated_ms" not in row and row["reasons"]
    w = DL.validity(0, {"underruns": 3}, 0)
    assert w["valid"] and w["warnings"]  # underruns avisan pero no invalidan el primer onset


# ---------------------------------------------------------------- captura (sounddevice falso)
def dev(name, i, mi=2, mo=2):
    return SimpleNamespace(id=name, name=name, index=i, max_in=mi, max_out=mo, default_sr=48000.0)


class FakeStream:
    def __init__(self, fail_start=False, **kw):
        self.kw, self.fail_start, self.stopped, self.closed = kw, fail_start, False, False

    def start(self):
        if self.fail_start:
            raise RuntimeError("no arranca")

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


def test_capture_only_opens_blackhole_never_other_devices():
    cap = DL.BlackHoleCapture(devices=[dev("MacBook Pro Microphone", 0, 1, 0), dev("MacBook Pro Speakers", 1, 0, 2)])
    with pytest.raises(DL.DeviceLoopbackError, match="BlackHole"):
        cap.pick_device()
    cap2 = DL.BlackHoleCapture(devices=[dev("BlackHole 2ch", 3), dev("BlackHole 16ch", 4)])
    with pytest.raises(DL.DeviceLoopbackError, match="más de un"):
        cap2.pick_device()
    streams = []

    class SD:
        @staticmethod
        def InputStream(**kw):
            s = FakeStream(fail_start=True, **kw)
            streams.append(s)
            return s

    cap3 = DL.BlackHoleCapture(devices=[dev("MacBook Pro Microphone", 0, 1, 0), dev("BlackHole 2ch", 3)], sd_module=SD)
    with pytest.raises(RuntimeError, match="no arranca"):
        cap3.start()
    assert streams[0].closed and streams[0].kw["device"] == 3  # abrió BlackHole y cerró el stream que no arrancó


def test_capture_callback_uses_harness_clock_and_input_latency():
    import numpy as np

    ticks = iter([1_000_000_000, 1_010_000_000])
    cap = DL.BlackHoleCapture(devices=[dev("BlackHole 2ch", 3)], clock=lambda: next(ticks))
    cap.sr, cap.channels = 48000, 2
    loud = np.full((480, 2), 0.1, dtype="float32")
    t = SimpleNamespace(currentTime=10.020, inputBufferAdcTime=10.000)
    cap._cb(loud, 480, t, None)
    cap._cb(np.zeros((480, 2), dtype="float32"), 480, t, "input overflow")
    b0, b1 = cap.blocks
    assert b0.t_cb_ns == 1_000_000_000 and b0.in_lat_ns == 20_000_000 and b0.start_est_ns == 980_000_000
    assert b0.rms_dbfs == pytest.approx(-20.0, abs=0.1) and b1.rms_dbfs == float("-inf") and cap.status_events == 1
    assert b0.dur_ns == 10_000_000


# ---------------------------------------------------------------- envoltorio con fakes
class FakeCapture:
    """Genera bloques de 10 ms en tiempo real; el audio sale 'delay_ms' tras escribir al FakePlayer."""

    blocksize_ms = 10.0
    sr, channels = 48000, 2

    def __init__(self, world):
        self.world, self.blocks, self.dev, self.status_events = world, [], dev("BlackHole 2ch", 3), 0
        self.task = None
        self.stopped = False

    def pick_device(self):
        return self.dev

    def start(self):
        self.task = asyncio.ensure_future(self._run())

    async def _run(self):
        while True:
            now = DL.MONO_NS()
            rms = -20.0 if any(a <= now <= z for a, z in self.world.sounding) or self.world.noise else self.world.floor
            self.blocks.append(DL.Block(now, now - 5 * MS, 10 * MS, rms, 5 * MS))
            await asyncio.sleep(0.01)

    def stop(self):
        self.stopped = True
        if self.task:
            self.task.cancel()


class World:
    def __init__(self, delay_ms=30, noise=False):
        self.sounding, self.delay_ms, self.noise, self.floor = [], delay_ms, noise, -90.0


class FakePlayer:
    def __init__(self, world):
        self.world, self.sr, self.underruns, self.dropped_s, self.status_events = world, 48000, 0, 0.0, 0
        self.started = self.stopped = False
        self.writes = 0

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    @property
    def queued_s(self):
        now = DL.MONO_NS()
        return max([0.0] + [(z - now) / 1e9 for a, z in self.world.sounding if z > now])

    def write(self, pcm, sr):
        self.writes += 1
        now = DL.MONO_NS()
        start = max(now + self.world.delay_ms * MS, max([0] + [z for _, z in self.world.sounding]))
        self.world.sounding.append((start, start + round(len(pcm) / 2 / sr * 1e9)))


class FakeInner(Adapter):
    mode = AdapterMode.STREAMING

    def __init__(self):
        self.q, self.closed, self.opened = asyncio.Queue(), False, False

    async def open(self, config):
        self.opened = True

    async def start_stream(self, stream_id, seed):
        pass

    async def push_audio(self, *a):
        pass

    async def end_stream(self, stream_id):
        pass

    async def events(self):
        while True:
            ev = await self.q.get()
            if ev is None:
                return
            yield ev

    async def close(self):
        self.closed = True
        await self.q.put(None)


class Wrapped(DeviceLoopbackAdapter):
    def __init__(self, world):
        super().__init__()
        self.world, self.fake_inner = world, FakeInner()

    def _make_inner(self, spec):
        return self.fake_inner

    def _make_capture(self):
        return FakeCapture(self.world)

    def _make_player(self, dev_, volume, latency=None):
        p = FakePlayer(self.world)
        p.requested, p.latency_s, p.blocksize = latency, 0.011, 0
        return p


def cfg(tmp_path, **kw):
    return {"report_path": str(tmp_path / "dl.json"), "pre_silence_s": 0.15, "settle_s": 0.1, "drain_timeout_s": 3.0, **kw}


def tone_event(stream_id="s1", dur=0.2, commit="s1:0"):
    return AdapterEvent(EventKind.TTS_CHUNK, stream_id, pcm=A.tone(dur, sample_rate=24000), sample_rate=24000, commit_id=commit)


def test_wrapper_measures_onset_and_writes_report_without_audio(tmp_path):
    async def go():
        w = Wrapped(World(delay_ms=60))
        await w.open(cfg(tmp_path))
        got = []

        async def consume():
            async for ev in w.events():
                got.append(ev)

        task = asyncio.ensure_future(consume())
        await w.start_stream("s1", 0)
        await w.fake_inner.q.put(tone_event())
        await asyncio.sleep(0.05)
        await w.end_stream("s1")
        await w.close()
        await task
        return w, got

    w, got = asyncio.run(go())
    rep = json.loads((tmp_path / "dl.json").read_text())
    c = rep["clips"][0]
    assert c["status"] == "ok" and c["commit_id"] == "s1:0" and c["unambiguous"]
    assert 40 <= c["device_minus_first_write_ms"] <= 130  # 60 ms simulados + bloque/tick de 10 ms
    assert rep["audio_saved"] is False and rep["physical_microphone_opened"] is False and "NO es audio escuchado" in rep["scope"]
    assert len(got) == 1 and got[0].kind == EventKind.TTS_CHUNK  # el evento del candidato pasa intacto
    assert w.player.stopped and w.capture.stopped and w.fake_inner.closed  # cleanup


def test_ambient_noise_before_playback_aborts_and_cleans_up(tmp_path):
    async def go():
        w = Wrapped(World(noise=True))
        with pytest.raises(DL.ContaminationError, match="antes de reproducir"):
            await w.open(cfg(tmp_path))
        return w

    w = asyncio.run(go())
    assert w.player.stopped and w.capture.stopped and w.fake_inner.closed


def test_foreign_audio_between_clips_aborts_next_stream(tmp_path):
    async def go():
        w = Wrapped(World())
        await w.open(cfg(tmp_path, drain_timeout_s=0.6))
        t = asyncio.ensure_future(_drain(w))
        await w.start_stream("s1", 0)
        await w.fake_inner.q.put(tone_event(dur=0.1))
        await asyncio.sleep(0.05)
        await w.end_stream("s1")
        w.world.noise = True  # alguien reproduce audio ajeno
        await asyncio.sleep(0.15)
        with pytest.raises(DL.ContaminationError):
            await w.start_stream("s2", 0)
        await w.close()
        await t

    async def _drain(w):
        async for _ in w.events():
            pass

    asyncio.run(go())


def test_write_outside_active_clip_is_reported_as_error_event_not_played(tmp_path):
    async def go():
        w = Wrapped(World())
        await w.open(cfg(tmp_path))
        out = []

        async def consume():
            async for ev in w.events():
                out.append(ev)

        t = asyncio.ensure_future(consume())
        await w.fake_inner.q.put(tone_event("s9"))  # sin start_stream
        await asyncio.sleep(0.05)
        await w.close()
        await t
        return w, out

    w, out = asyncio.run(go())
    assert w.player.writes == 0 and any(e.kind == EventKind.ERROR and "fuera de un clip" in e.error_kind for e in out)


def test_end_stream_barrier_closes_race_with_events_still_queued(tmp_path):
    """El candidato deja el evento en su cola y end_stream vuelve SIN esperar: el clip no se cierra antes de escribir."""
    async def go():
        w = Wrapped(World(delay_ms=40))
        await w.open(cfg(tmp_path))

        async def consume():
            async for _ in w.events():
                pass

        t = asyncio.ensure_future(consume())
        await w.start_stream("s1", 0)
        w.fake_inner.q.put_nowait(tone_event(dur=0.15))  # sin sleep: el consumidor aún no lo vio
        await w.end_stream("s1")
        await w.close()
        await t
        return w

    w = asyncio.run(go())
    rep = json.loads((tmp_path / "dl.json").read_text())
    c = rep["clips"][0]
    assert c["n_writes"] == 1 and c["status"] == "ok" and w.player.writes == 1 and w.late_events == 0
    assert rep["validity"]["valid"] and "NO cubre la calibración del reloj" in rep["uncertainty_note"]


def test_chunks_arriving_after_end_stream_are_rejected_and_invalidate(tmp_path):
    async def go():
        w = Wrapped(World())
        await w.open(cfg(tmp_path))
        got = []

        async def consume():
            async for ev in w.events():
                got.append(ev)

        t = asyncio.ensure_future(consume())
        await w.start_stream("s1", 0)
        await w.end_stream("s1")
        w.fake_inner.q.put_nowait(tone_event())  # tarde: después de cerrar el clip
        await asyncio.sleep(0.05)
        await w.close()
        await t
        return w, got

    w, got = asyncio.run(go())
    rep = json.loads((tmp_path / "dl.json").read_text())
    assert w.player.writes == 0 and w.late_events == 1 and not rep["validity"]["valid"]
    assert any(e.kind == EventKind.ERROR for e in got)


def test_player_is_started_before_capture_and_dead_capture_is_not_silence(tmp_path):
    order, times = [], []

    class W(Wrapped):
        def _make_capture(self):
            c = super()._make_capture()
            orig = c.start
            c.start = lambda: (order.append("capture"), times.append(DL.MONO_NS()), orig())
            return c

        def _make_player(self, d, v, latency=None):
            p = super()._make_player(d, v, latency)
            orig = p.start
            p.start = lambda: (order.append("player"), times.append(DL.MONO_NS()), orig())
            return p

    async def go():
        w = W(World())
        await w.open(cfg(tmp_path))
        await w.close()

    asyncio.run(go())
    assert order == ["player", "capture"] and (times[1] - times[0]) >= 0.29e9  # hueco mínimo medido
    # con captura primero, AUHAL -10863 mata la captura
    assert not DL.validity(0, {}, 0, max_block_gap_ms=900.0)["valid"]
    assert DL.validity(0, {}, 0, max_block_gap_ms=12.0)["valid"]

    async def dead():
        w = Wrapped(World())
        await w.open(cfg(tmp_path))
        w.capture.stop()  # deja de entregar bloques
        await asyncio.sleep(0.4)
        with pytest.raises(DL.DeviceLoopbackError, match="dejó de entregar"):
            await w.start_stream("s1", 0)
        await w.close()

    asyncio.run(dead())


def test_json_is_standard_silence_is_null_and_partial_blocks_wording(tmp_path):
    assert DL.json_safe({"a": float("-inf"), "b": [float("nan"), 1.5]}) == {"a": None, "b": [None, 1.5]}

    async def go():
        world = World()
        world.floor = float("-inf")  # silencio digital exacto
        w = Wrapped(world)
        await w.open(cfg(tmp_path))
        await w.close()
        return w

    asyncio.run(go())
    txt = (tmp_path / "dl.json").read_text()
    rep = json.loads(txt, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))  # sin -Infinity/NaN
    assert rep["calibration"]["peak_dbfs"] is None and rep["calibration"]["peak_state"] == "silence_digital"
    v = DL.validity(0, {"underruns": 4}, 0)
    assert v["valid"] and "NO demuestran un underrun físico" in v["warnings"][0]


def test_wrapper_records_requested_latency_mode_and_declared_stream_latency(tmp_path):
    async def go(mode):
        w = Wrapped(World())
        await w.open(cfg(tmp_path, player_latency=mode))
        await w.close()
        return w

    w = asyncio.run(go("low"))
    p = json.loads((tmp_path / "dl.json").read_text())["player"]
    assert w.player.requested == "low" and p["requested_latency_mode"] == "low"
    assert p["reported_stream_latency_s"] == 0.011 and "NO es una medición física" in p["reported_stream_latency_note"]
    asyncio.run(go(None))
    assert json.loads((tmp_path / "dl.json").read_text())["player"]["requested_latency_mode"] == "default"
