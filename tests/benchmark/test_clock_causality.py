"""Pruebas del reloj, del origen de latencia y de fuga temporal."""
import asyncio

import pytest
from conftest import make_item, run_dummy

from benchmark import audio as A
from benchmark import runner
from benchmark.adapters.dummy import DummyAdapter
from benchmark.clock import VirtualClock, run_virtual
from benchmark.replay import EventCollector, run_stream, Segment


def test_virtual_clock_is_exact_and_instant():
    import time

    async def main(clock: VirtualClock):
        t0 = clock.now_ns()
        await clock.sleep_until_ns(t0 + 3_000_000_000)
        return clock.now_ns() - t0

    w = time.monotonic()
    assert run_virtual(main) == 3_000_000_000
    assert time.monotonic() - w < 1.0


def test_audio_is_delivered_only_after_it_has_occurred():
    """Un chunk [a,b) llega al candidato en t0+b, nunca antes."""
    ad = DummyAdapter()
    res = run_dummy(ad, [make_item(0)])
    tr = res.traces[0]
    assert tr.sends
    for s in tr.sends:
        assert s.actual_ns >= tr.t0_ns + round(s.audio_end_s * 1e9)
    # mismo dato visto por el candidato: loop time del push >= t0 + fin de audio
    for p in ad.push_log:
        if p["stream_id"] == tr.stream_id:
            assert round(p["loop_time_s"] * 1e9) >= tr.t0_ns + round(p["t_audio_end_s"] * 1e9)


def test_no_future_leak_candidate_view_up_to_T_is_independent_of_later_audio():
    """Dos streams idénticos hasta T y distintos después: lo que el candidato
    observó hasta T debe ser idéntico, y sus eventos antes de T también."""
    T = 0.9
    sr = A.TARGET_SR

    def build(tail_freq):
        head = A.silence(0.3) + A.tone(0.6, freq_hz=300)  # hasta 0.9 s
        tailpcm = A.tone(0.5, freq_hz=tail_freq) + A.silence(0.2)
        return head + tailpcm

    out = []
    for f in (500, 900):
        ad = DummyAdapter()
        it = make_item(0)
        it.pcm = build(f)
        it.segment = Segment("it0", 0.3, 1.4, "human_annotation", None)
        res = run_dummy(ad, [it], drain_grace_s=0.1, deadline_s=2.0)
        t = res.traces[0]
        seen = [(p["t_audio_end_s"], p["sha256"]) for p in ad.push_log if p["t_audio_end_s"] <= T + 1e-9]
        ev_before = [
            (e.kind, e.recv_ns - t.t0_ns)
            for e in t.events
            if e.recv_ns - t.t0_ns <= round(T * 1e9)
        ]
        out.append((seen, ev_before))
    assert out[0][0] == out[1][0] and len(out[0][0]) > 0
    assert out[0][1] == out[1][1]


def test_end_stream_is_not_an_end_of_speech_cue():
    class Spy(DummyAdapter):
        end_loop_time = None

        async def end_stream(self, stream_id):
            self.end_loop_time = asyncio.get_running_loop().time()

    ad = Spy()
    res = run_dummy(ad, [make_item(0)], deadline_s=3.0)
    tr = res.traces[0]
    true_end = tr.segments[0].end_s
    # end_stream sólo tras cubrir toda la ventana de censura con silencio
    assert ad.end_loop_time * 1e9 >= tr.t0_ns + (true_end + 3.0) * 1e9 - 1


def test_origin_is_annotated_end_not_candidate_endpoint():
    """Si el endpoint del candidato tarda más, la latencia sube 1:1; el origen
    no se mueve al endpoint ni al fin del VAD."""
    lat = {}
    for ep in (200.0, 600.0):
        res = run_dummy(DummyAdapter(), [make_item(0)], adapter_config={"endpoint_silence_ms": ep, "processing_ms": 100.0})
        r = res.segment_results[0]
        assert r.outcome == "ok"
        lat[ep] = r.end_to_first_ms
    assert lat[200.0] == pytest.approx(300.0, abs=25)  # 200 de endpoint + 100 de proceso
    assert lat[600.0] == pytest.approx(700.0, abs=25)


def test_candidate_claimed_timestamps_are_ignored():
    res = run_dummy(DummyAdapter(), [make_item(0)])
    tts = [e for e in res.traces[0].events if e.kind == "tts_chunk"]
    assert tts and all(e.claimed_ns == 0 for e in tts)  # el dummy miente: 0
    assert all(e.recv_ns > 0 for e in tts)
    assert res.segment_results[0].first_pcm_ns > 0


def test_anticipated_output_is_not_a_success_and_is_reported_separately():
    res = run_dummy(
        DummyAdapter(),
        [make_item(0)],
        adapter_config={"behavior": "anticipate", "anticipate_at_s": 0.6},
    )
    r = res.segment_results[0]
    assert r.outcome == "anticipated"
    assert r.end_to_first_ms is None  # no entra como latencia válida
    assert r.signed_end_to_first_ms < 0
    g = res.aggregate["groups"][0]
    assert g["outcome_counts"]["anticipated"] == 1
    sm = g["end_to_first_ms"]
    assert sm["counts"]["anticipated"] == 1 and sm["counts"]["failure"] == 0  # no es fallo
    assert g["anticipation"]["count"] == 1 and g["anticipation"]["signed_end_to_first_ms"][0] < 0
    # sin salida válida verificada: ni distribución observada ni compuesto la premian
    assert sm["censoring_aware"]["quantiles"]["p50"]["identifiable"] is False
    assert sm["composite_time_to_valid_output"]["quantiles"]["p50"]["identifiable"] is False


def test_first_pcm_is_nonsilent_not_first_chunk():
    """Salida con solo silencio no cuenta como primer PCM."""
    res = run_dummy(DummyAdapter(), [make_item(0)], adapter_config={"behavior": "silent"})
    r = res.segment_results[0]
    assert r.outcome == "silent_output"
    assert r.first_pcm_ns is None


def test_leading_silence_inside_chunk_is_not_counted_as_audio():
    from benchmark.audio import first_nonsilent_offset_s

    pcm = A.silence(0.05) + A.tone(0.1)
    off = first_nonsilent_offset_s(pcm, A.TARGET_SR)
    assert 0.04 <= off <= 0.06


def test_failures_and_timeouts_stay_in_denominator_with_distinct_classes():
    for beh, expected, status in (("never", "timeout", "timeout"), ("error", "error", "failure")):
        res = run_dummy(DummyAdapter(), [make_item(0), make_item(1)], adapter_config={"behavior": beh})
        assert {r.outcome for r in res.segment_results} == {expected}
        sm = res.aggregate["groups"][0]["end_to_first_ms"]
        assert sm["n_total"] == 2 and sm["counts"][status] == 2
