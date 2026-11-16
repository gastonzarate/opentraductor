"""Hook standalone UNIT_DONE/UNIT_CANCELLED. Sin modelos ni cambios al contrato incremental."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchmark import audio as A
from benchmark.contracts import Adapter, AdapterEvent, EventKind
from benchmark.incremental import metrics
from benchmark.incremental.collector import CollectorValidationError, IncrementalCollector
from benchmark.replay import EventCollector


class DeclaredAdapter(Adapter):
    unit_events_supported = True


class LegacyAdapter(Adapter):
    pass


def observer_for(adapter: Adapter, stream_id: str = "s") -> IncrementalCollector:
    observer = IncrementalCollector()
    EventCollector(adapter, SimpleNamespace(), observer=observer)  # configura capability; no inicia tareas
    observer.register_stream(SimpleNamespace(stream_id=stream_id, t0_ns=0, input_duration_s=10.0))
    return observer


def tts(stream="s", emission="X", revision=0, recv_ns=1_000_000_000):
    return AdapterEvent(
        EventKind.TTS_CHUNK,
        stream,
        pcm=A.tone(0.01),
        sample_rate=A.TARGET_SR,
        planned_text_en="hello",
        commit_id=emission,
        revision=revision,
    ), recv_ns


def done(stream="s", emission="X", revision=0, chunks=1, samples=160, recv_ns=2_000_000_000):
    return AdapterEvent(
        EventKind.UNIT_DONE,
        stream,
        commit_id=emission,
        revision=revision,
        unit_n_chunks=chunks,
        unit_total_samples=samples,
        unit_reason="tts_complete",
    ), recv_ns


def cancelled(stream="s", emission="X", revision=0, before=True, recv_ns=500_000_000):
    return AdapterEvent(
        EventKind.UNIT_CANCELLED,
        stream,
        commit_id=emission,
        revision=revision,
        before_emit_claim=before,
        unit_reason="revision",
    ), recv_ns


def emit(observer: IncrementalCollector, pair):
    observer.record_adapter_event(*pair)


def test_declared_done_maps_to_record_and_counts_match_observed_chunks():
    observer = observer_for(DeclaredAdapter())
    emit(observer, tts())
    emit(observer, done())
    run = observer.to_incremental_run("s")
    assert run.unit_cancelled == [] and len(run.unit_done) == 1
    assert run.unit_done[0].emission_id == "X" and run.unit_done[0].n_chunks == 1
    assert metrics.unit_closure(run)["done"]["n_issues"] == 0
    record = observer.export_record("s")
    assert record["schema"] == "opentraductor-incremental-collector/v2"
    assert record["incremental"]["unit_events"]["support"] == "declared"
    assert record["incremental"]["unit_events"]["unit_done"] == 1


def test_done_mismatch_and_duplicate_remain_visible_for_unit_closure():
    mismatch = observer_for(DeclaredAdapter())
    emit(mismatch, tts())
    emit(mismatch, done(chunks=2, samples=999))
    issues = metrics.unit_closure(mismatch.to_incremental_run("s"))["done"]["issues"]
    assert issues == {"claim_mismatch_final": ["X@0"]}

    duplicate = observer_for(DeclaredAdapter())
    emit(duplicate, tts())
    emit(duplicate, done())
    emit(duplicate, done(recv_ns=3_000_000_000))
    issues = metrics.unit_closure(duplicate.to_incremental_run("s"))["done"]["issues"]
    assert issues == {"double_done": ["X@0"]}


def test_cancel_before_chunk_and_after_chunk_are_distinguished_by_harness_order():
    observer = observer_for(DeclaredAdapter())
    emit(observer, cancelled(emission="before", recv_ns=500_000_000))
    emit(observer, tts(emission="before", recv_ns=1_000_000_000))
    emit(observer, tts(emission="after", recv_ns=2_000_000_000))
    emit(observer, cancelled(emission="after", recv_ns=3_000_000_000))
    out = metrics.unit_closure(observer.to_incremental_run("s"))["cancelled"]
    assert out["n_cancelled"] == 2
    assert out["n_chunk_after_cancel"] == 1
    assert out["n_already_audible_at_cancel"] == 1
    assert out["n_before_emit_claim_false"] == 1


def test_same_emission_id_revisions_match_separate_chunks_and_done_records():
    observer = observer_for(DeclaredAdapter())
    emit(observer, tts(revision=0, recv_ns=1_000_000_000))
    emit(observer, done(revision=0, recv_ns=2_000_000_000))
    emit(observer, tts(revision=1, recv_ns=3_000_000_000))
    emit(observer, done(revision=1, recv_ns=4_000_000_000))
    run = observer.to_incremental_run("s")
    assert [c.cand.revision for c in run.chunks] == [0, 1]
    assert [(x.emission_id, x.revision) for x in run.unit_done] == [("X", 0), ("X", 1)]
    assert metrics.unit_closure(run)["done"]["n_issues"] == 0


@pytest.mark.parametrize(
    "event,match",
    [
        (AdapterEvent(EventKind.UNIT_DONE, "s", commit_id=None, unit_n_chunks=0, unit_total_samples=0), "commit_id"),
        (AdapterEvent(EventKind.UNIT_DONE, "s", commit_id="X", unit_n_chunks=True, unit_total_samples=0), "unit_n_chunks"),
        (AdapterEvent(EventKind.UNIT_DONE, "s", commit_id="X", revision=-1, unit_n_chunks=0, unit_total_samples=0), "revision"),
        (AdapterEvent(EventKind.UNIT_CANCELLED, "s", commit_id="X"), "before_emit_claim"),
        (AdapterEvent(EventKind.UNIT_CANCELLED, "s", commit_id="X", before_emit_claim=True, unit_n_chunks=0), "no acepta conteos"),
    ],
)
def test_malformed_unit_events_fail_visibility_instead_of_becoming_valid(event, match):
    observer = observer_for(DeclaredAdapter())
    with pytest.raises(CollectorValidationError, match=match):
        observer.record_adapter_event(event, 1_000_000_000)
    assert observer.to_incremental_run("s").n_failed == 1


def test_legacy_unknown_differs_from_declared_zero_events():
    legacy = observer_for(LegacyAdapter())
    old = legacy.to_incremental_run("s")
    assert old.unit_done is None and old.unit_cancelled is None
    assert legacy.export_record("s")["incremental"]["unit_events"]["support"] == "legacy_unknown"

    declared = observer_for(DeclaredAdapter())
    new = declared.to_incremental_run("s")
    assert new.unit_done == [] and new.unit_cancelled == []
    assert declared.export_record("s")["incremental"]["unit_events"] == {
        "support": "declared",
        "unit_done": 0,
        "unit_cancelled": 0,
        "verification": "claims del candidato estampados por el arnés; el cierre/cancelación se contrasta después con chunks/playout",
    }


def test_trace_event_persists_unit_fields_without_incremental_observer():
    event = done(revision=3, chunks=2, samples=320, recv_ns=123)[0]
    trace_event = EventCollector(DeclaredAdapter(), SimpleNamespace())._to_trace_event(event, 123)
    assert trace_event.kind == "unit_done" and trace_event.commit_id == "X"
    assert trace_event.revision == 3 and trace_event.unit_n_chunks == 2 and trace_event.unit_total_samples == 320
    assert trace_event.unit_reason == "tts_complete" and trace_event.pcm is None


def test_error_keeps_unit_id_revision_and_reason_while_marking_failure():
    observer = observer_for(DeclaredAdapter())
    observer.record_adapter_event(
        AdapterEvent(
            EventKind.ERROR,
            "s",
            commit_id="X",
            revision=2,
            error_kind="tts_error",
            unit_reason="partial_tts_failure",
        ),
        1_000_000_000,
    )
    run = observer.to_incremental_run("s")
    event = observer.export_record("s")["events"][0]
    assert run.n_failed == 1 and event["emission_id"] == "X"
    assert event["revision"] == 2 and event["unit_reason"] == "partial_tts_failure"
