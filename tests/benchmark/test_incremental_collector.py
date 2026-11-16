"""Collector incremental real sobre replay causal. Sin modelos ni audio físico."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmark import audio as A
from benchmark import replay
from benchmark.clock import RealClock, run_virtual
from benchmark.contracts import Adapter, AdapterEvent, AdapterMode, EventKind
from benchmark.incremental.collector import CollectorValidationError, IncrementalCollector
from benchmark.replay import EventCollector, Segment, run_stream


_END = object()


class ScriptedAdapter(Adapter):
    mode = AdapterMode.STREAMING
    produces_audio = True

    def __init__(self, behavior: str = "normal") -> None:
        self.behavior = behavior
        self.q: asyncio.Queue = asyncio.Queue()
        self.tasks: list[asyncio.Task] = []
        self.pushes = 0

    async def start_stream(self, stream_id: str, seed: int) -> None:
        async def before_first_push() -> None:
            await asyncio.sleep(0)
            await self.q.put(AdapterEvent(EventKind.ASR_PARTIAL, stream_id, text="before"))

        self.tasks.append(asyncio.create_task(before_first_push()))

    async def push_audio(self, stream_id: str, pcm: bytes, t_audio_start_s: float) -> None:
        self.pushes += 1
        if self.behavior == "timeout":
            await asyncio.Event().wait()
            return
        if self.behavior == "malformed":
            await self.q.put(
                AdapterEvent(EventKind.TTS_CHUNK, stream_id, pcm=b"\x00", sample_rate=A.TARGET_SR)
            )
            await asyncio.sleep(0)
            return
        if self.behavior == "bad-rate":
            await self.q.put(
                AdapterEvent(EventKind.TTS_CHUNK, stream_id, pcm=A.tone(0.01), sample_rate=0)
            )
            await asyncio.sleep(0)
            return
        if self.behavior == "error":
            await self.q.put(AdapterEvent(EventKind.ERROR, stream_id, error_kind="synthetic_error"))
            await asyncio.sleep(0)
            return

        await self.q.put(
            AdapterEvent(
                EventKind.ASR_PARTIAL,
                stream_id,
                text=f"during-{self.pushes}",
                audio_pos_s=t_audio_start_s + len(pcm) / 2 / A.TARGET_SR,
                commit_id=f"commit-{self.pushes}",
            )
        )
        await asyncio.sleep(0)  # el EventCollector recibe durante push_audio
        out = A.silence(0.01) if self.behavior == "silent" else A.tone(0.01)
        await self.q.put(
            AdapterEvent(
                EventKind.TTS_CHUNK,
                stream_id,
                pcm=out,
                sample_rate=A.TARGET_SR,
                planned_text_en="hello",
                source_span_s=(t_audio_start_s, t_audio_start_s + len(pcm) / 2 / A.TARGET_SR),
                commit_id=f"commit-{self.pushes}",
                claimed_ns=0,
            )
        )

        if self.pushes == 1:
            async def after_first_push() -> None:
                await asyncio.sleep(0.005)
                await self.q.put(AdapterEvent(EventKind.MT_PARTIAL, stream_id, text="after"))

            self.tasks.append(asyncio.create_task(after_first_push()))

    async def end_stream(self, stream_id: str) -> None:
        if self.tasks:
            await asyncio.gather(*self.tasks)
        await asyncio.sleep(0)

    async def events(self):
        while True:
            item = await self.q.get()
            if item is _END:
                return
            yield item

    async def close(self) -> None:
        await self.q.put(_END)


class IteratorFailureAdapter(ScriptedAdapter):
    async def events(self):
        await self.q.get()
        raise RuntimeError("synthetic iterator failure")
        yield  # pragma: no cover - mantiene esta función como async generator


class MalformedKindAdapter(ScriptedAdapter):
    async def push_audio(self, stream_id: str, pcm: bytes, t_audio_start_s: float) -> None:
        await self.q.put(AdapterEvent("malformed-kind", stream_id))  # type: ignore[arg-type]
        await asyncio.sleep(0)


class DeliveryGapAdapter(ScriptedAdapter):
    async def start_stream(self, stream_id: str, seed: int) -> None:
        async def between_schedule_and_delivery() -> None:
            await asyncio.sleep(0.022)
            await self.q.put(AdapterEvent(EventKind.ASR_PARTIAL, stream_id, text="in-gap"))

        self.tasks.append(asyncio.create_task(between_schedule_and_delivery()))

    async def push_audio(self, stream_id: str, pcm: bytes, t_audio_start_s: float) -> None:
        return None


async def exercise(clock, adapter: ScriptedAdapter, *, pcm: bytes | None = None, timeout: float = 1.0):
    observer = IncrementalCollector()
    collector = EventCollector(adapter, clock, observer=observer)
    collector.start()
    source = pcm if pcm is not None else A.tone(0.04)
    trace = await run_stream(
        adapter,
        clock,
        collector,
        "opaque-stream",
        source,
        [Segment("fixture", 0.0, len(source) / 2 / A.TARGET_SR)],
        chunk_ms=20.0,
        deadline_s=0.0,
        drain_grace_s=0.01,
        call_timeout_s=timeout,
    )
    await asyncio.sleep(0)
    await collector.stop()
    return observer, collector, trace


def test_virtual_replay_stamps_input_frontier_and_events_in_one_order():
    observer, collector, trace = run_virtual(lambda clock: exercise(clock, ScriptedAdapter()))
    assert trace.error is None and collector.collector_error is None

    run = observer.to_incremental_run(trace.stream_id)
    assert run.alignment is None and run.ref_units == []
    assert [d.seq for d in run.input_deliveries] == [0, 1]
    assert [d.end_s for d in run.input_deliveries] == pytest.approx([0.02, 0.04])
    assert [c.chunk_id for c in run.chunks] == [0, 1]
    assert all(c.pcm_samples == 160 and c.sample_rate == 16000 for c in run.chunks)
    assert all(c.nonsilent_offset_s == pytest.approx(0.0) for c in run.chunks)

    record = observer.export_record(trace.stream_id)
    seqs = [x["event_seq"] for x in record["timeline"]]
    assert seqs == list(range(len(seqs)))
    before = next(e for e in record["events"] if e["text"] == "before")
    during = next(e for e in record["events"] if e["text"] == "during-1")
    after = next(e for e in record["events"] if e["text"] == "after")
    assert before["stamp"]["input_seq"] == -1 and before["stamp"]["delivered_until_s"] == 0.0
    assert during["stamp"]["input_seq"] == 0 and during["stamp"]["delivered_until_s"] == pytest.approx(0.02)
    assert after["stamp"]["delivered_until_s"] == pytest.approx(0.02)
    first_input = next(x for x in record["timeline"] if x["kind"] == "input_delivery")
    assert first_input["event_seq"] < during["stamp"]["event_seq"]
    assert first_input["observed_ns"] == during["stamp"]["recv_ns"]  # empate real del reloj virtual, ordenado por event_seq


def test_export_persists_metadata_but_never_pcm(tmp_path: Path):
    observer, _collector, trace = run_virtual(lambda clock: exercise(clock, ScriptedAdapter()))
    path = observer.write_json(trace.stream_id, tmp_path / "collector.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["incremental"] == {
        "alignment": None,
        "ref_units": [],
        "n_chunks": 2,
        "n_failed": 0,
        "n_timeout": 0,
        "semantic_metrics": "unavailable_without_independent_alignment",
        "unit_events": {
            "support": "legacy_unknown",
            "unit_done": None,
            "unit_cancelled": None,
            "verification": "adapter legacy: ausencia de eventos no equivale a cero cierres/cancelaciones",
        },
    }
    assert all("pcm" not in event for event in data["events"])
    assert all(event.get("pcm_sha256") for event in data["events"] if event["kind"] == "tts_chunk")
    assert b"RIFF" not in path.read_bytes()


def test_silent_pcm_is_observed_without_becoming_successful_audio():
    observer, collector, trace = run_virtual(lambda clock: exercise(clock, ScriptedAdapter("silent")))
    assert collector.collector_error is None and trace.error is None
    run = observer.to_incremental_run(trace.stream_id)
    assert run.n_failed == run.n_timeout == 0 and run.chunks
    assert all(c.nonsilent_offset_s is None for c in run.chunks)
    events = observer.export_record(trace.stream_id)["events"]
    assert all(e["rms_state"] == "silence" for e in events if e["kind"] == "tts_chunk")


def test_malformed_pcm_and_adapter_error_are_failures_not_successes():
    malformed, malformed_collector, trace = run_virtual(
        lambda clock: exercise(clock, ScriptedAdapter("malformed"))
    )
    assert "cantidad impar" in trace.error
    assert "cantidad impar" in malformed_collector.collector_error
    assert malformed.to_incremental_run(trace.stream_id).n_failed == 1
    assert malformed.to_incremental_run(trace.stream_id).chunks == []

    bad_rate, bad_rate_collector, bad_rate_trace = run_virtual(
        lambda clock: exercise(clock, ScriptedAdapter("bad-rate"))
    )
    assert "sample_rate debe ser entero positivo" in bad_rate_collector.collector_error
    assert bad_rate.to_incremental_run(bad_rate_trace.stream_id).n_failed == 1

    errored, error_collector, error_trace = run_virtual(
        lambda clock: exercise(clock, ScriptedAdapter("error"))
    )
    assert error_collector.collector_error is None and error_trace.error is None
    assert errored.to_incremental_run(error_trace.stream_id).n_failed == 1
    assert "synthetic_error" in errored.export_record(error_trace.stream_id)["error"]


def test_iterator_failure_and_malformed_kind_fail_current_and_future_streams():
    async def iterator_case(clock):
        adapter = IteratorFailureAdapter()
        observer = IncrementalCollector()
        collector = EventCollector(adapter, clock, observer=observer)
        collector.start()
        first = await run_stream(
            adapter, clock, collector, "s1", A.tone(0.02),
            [Segment("fixture", 0.0, 0.02)], chunk_ms=20.0,
            deadline_s=0.0, drain_grace_s=0.0, call_timeout_s=1.0,
        )
        second = await run_stream(
            adapter, clock, collector, "s2", A.tone(0.02),
            [Segment("fixture", 0.0, 0.02)], chunk_ms=20.0,
            deadline_s=0.0, drain_grace_s=0.0, call_timeout_s=1.0,
        )
        await collector.stop()
        return observer, collector, first, second

    observer, collector, first, second = run_virtual(iterator_case)
    assert "synthetic iterator failure" in collector.collector_error
    assert "event_collection" in first.error
    assert "EventCollector no disponible" in second.error
    assert observer.to_incremental_run("s1").n_failed == 1
    assert observer.to_incremental_run("s2").n_failed == 1
    assert observer.to_incremental_run("s1").chunks == observer.to_incremental_run("s2").chunks == []

    malformed, malformed_collector, malformed_trace = run_virtual(
        lambda clock: exercise(clock, MalformedKindAdapter())
    )
    assert "kind inválido" in malformed_collector.collector_error
    assert malformed_trace.error and malformed.to_incremental_run(malformed_trace.stream_id).n_failed == 1


def test_delivery_timestamp_is_taken_inside_deliver_after_scheduler_gap(monkeypatch):
    original_call = replay._call

    async def delayed_call(coro, timeout_s, what):
        if what == "push_audio":
            await asyncio.sleep(0.005)
        return await original_call(coro, timeout_s, what)

    monkeypatch.setattr(replay, "_call", delayed_call)
    observer, collector, trace = run_virtual(
        lambda clock: exercise(clock, DeliveryGapAdapter(), pcm=A.tone(0.02))
    )
    assert collector.collector_error is None and trace.error is None
    record = observer.export_record(trace.stream_id)
    gap_event = next(e for e in record["events"] if e["text"] == "in-gap")
    delivery = record["input_deliveries"][0]
    timeline = record["timeline"]
    gap_timeline = next(x for x in timeline if x["kind"] == "adapter:asr_partial")
    input_timeline = next(x for x in timeline if x["kind"] == "input_delivery")
    assert gap_event["stamp"]["delivered_until_s"] == 0.0
    assert gap_event["stamp"]["recv_ns"] == 22_000_000
    assert delivery["actual_ns"] == trace.sends[0].actual_ns == 25_000_000
    assert gap_timeline["event_seq"] < input_timeline["event_seq"]
    assert [x["observed_ns"] for x in timeline] == sorted(x["observed_ns"] for x in timeline)


def test_push_timeout_is_preserved_as_censored_timeout():
    observer, collector, trace = run_virtual(
        lambda clock: exercise(clock, ScriptedAdapter("timeout"), timeout=0.01)
    )
    assert collector.collector_error is None
    assert trace.error.startswith("AdapterHang: push_audio")
    run = observer.to_incremental_run(trace.stream_id)
    assert run.n_timeout == 1 and run.n_failed == 0 and run.chunks == []


def test_input_boundary_rejects_malformed_or_noncontiguous_ranges():
    observer = IncrementalCollector()
    observer.register_stream(SimpleNamespace(stream_id="s", t0_ns=0, input_duration_s=0.04))
    with pytest.raises(CollectorValidationError, match="impar"):
        observer.record_input_delivery("s", b"\x00", 0.0, 0.02, 20_000_000)
    with pytest.raises(CollectorValidationError, match="no coincide"):
        observer.record_input_delivery("s", bytes(640), 0.0, 0.03, 20_000_000)
    with pytest.raises(CollectorValidationError, match="no contigua"):
        observer.record_input_delivery("s", bytes(640), 0.01, 0.03, 30_000_000)
    observer.record_input_delivery("s", bytes(640), 0.0, 0.02, 20_000_000)
    with pytest.raises(CollectorValidationError, match="fuera del stream"):
        observer.record_input_delivery("s", bytes(800), 0.02, 0.045, 45_000_000)


def test_real_clock_smoke_without_physical_audio():
    async def main():
        observer, collector, trace = await exercise(
            RealClock(), ScriptedAdapter(), pcm=A.tone(0.02), timeout=1.0
        )
        assert collector.collector_error is None and trace.clock_kind == "real"
        record = observer.export_record(trace.stream_id)
        assert record["input_deliveries"][0]["actual_ns"] >= trace.t0_ns + 20_000_000
        assert record["events"] and record["playback"] == "simulated"

    asyncio.run(main())
