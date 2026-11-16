import dataclasses

import pytest
from test_incremental_fakes import alignment, chunk, ideal_run, real, run_, units

from benchmark.incremental import contracts, metrics
from benchmark.incremental.contracts import AlignmentBinding, AlignmentResult, AtomRealization, OutputSpan


def test_spaces_are_disjoint_and_candidate_fields_are_only_claims():
    h = {f.name for f in dataclasses.fields(contracts.HarnessStamp)}
    c = {f.name for f in dataclasses.fields(contracts.CandidateFields)}
    e = {f.name for f in dataclasses.fields(contracts.AtomRealization)} | {f.name for f in dataclasses.fields(contracts.RefUnit)}
    assert h == {"recv_ns", "input_seq", "delivered_until_s", "event_seq"} and not (h & c) and not ({"recv_ns", "delivered_until_s"} & e)
    with pytest.raises(dataclasses.FrozenInstanceError):
        contracts.HarnessStamp(1, 0, 0.0).recv_ns = 2


def test_nothing_has_a_default_quality_or_provenance():
    with pytest.raises(TypeError):  # sin status ni incertidumbres explícitas no se puede construir una realización
        AtomRealization(realization_id="x", atom_id="a", ref_unit_id="u", span=OutputSpan(0, 0, 1))
    with pytest.raises(TypeError):
        AlignmentResult(alignment_id="a", method="m", version="1", binding=AlignmentBinding(fixture_id="f"))


def test_alignment_validation_rejects_incoherent_results():
    ref = units(1)
    ok = alignment([], ref)
    assert ok.validate() == []
    bad = dataclasses.replace(ok, binding=AlignmentBinding())
    assert any("binding" in p for p in bad.validate())
    assert any("raw" in p for p in dataclasses.replace(ok, provenance="human_validated", correction_state="raw", binding=AlignmentBinding(output_audio_sha256="a" * 64)).validate())
    assert any("complete" in p for p in dataclasses.replace(ok, n_atoms_assessed=1).validate())
    dup = alignment([real("r", "u0a0", "u0", 0, 0, 1), real("r", "u0a1", "u0", 0, 1, 2)], ref)
    assert any("duplicado" in p for p in dup.validate())
    nospan = dataclasses.replace(ok, realizations=(AtomRealization("r", "u0a0", "u0", "realized", None, 0.0, 0.0, 1.0, 0.1),))
    assert any("sin span" in p for p in nospan.validate())
    assert dataclasses.replace(ok, binding=AlignmentBinding(output_audio_sha256="a" * 64), provenance="auto_unvalidated").tier == "diagnostic"


def test_incoherent_alignment_makes_every_dependent_metric_unavailable_not_zero():
    r = run_([chunk("a", 4.0)], units(1), dataclasses.replace(alignment([], units(1)), binding=AlignmentBinding()))
    for fn in (metrics.coverage_and_integrity, metrics.semantic_lag, metrics.anticipation, metrics.claims_vs_evaluator):
        out = fn(r)
        assert out["status"] == "unavailable" and "incoherente" in out["reason"]


def test_chunk_ids_are_harness_assigned_unique_and_required():
    from benchmark.incremental.contracts import IncrementalRun

    c = chunk("a", 1.0)
    with pytest.raises(ValueError, match="chunk_id"):
        metrics.first_onset(IncrementalRun(0, 5.0, [], [c]))
    with pytest.raises(ValueError, match="chunk_id"):
        metrics.first_onset(IncrementalRun(0, 5.0, [], [dataclasses.replace(c, chunk_id=0), dataclasses.replace(c, chunk_id=0)]))


def test_candidate_ids_and_claims_do_not_change_any_independent_metric():
    r1, r2 = ideal_run(), ideal_run()
    chunks = [chunk(f"zz{i}", c.stamp.recv_ns / 1e9, 1.0, claim=(0.0, 99.0), speed=3.0, last=True, seq=i, delivered=c.stamp.delivered_until_s) for i, c in enumerate(r2.chunks)]
    r3 = run_(chunks, r2.ref_units, r2.alignment)
    a, b = metrics.coverage_and_integrity(r1), metrics.coverage_and_integrity(r3)
    assert a["semantic_atom_coverage"] == b["semantic_atom_coverage"] == 1.0 and a["n_units_complete"] == b["n_units_complete"] == 6
    assert metrics.semantic_lag(r1)["median_lag_to_support_end_s"] == metrics.semantic_lag(r3)["median_lag_to_support_end_s"]


def test_missing_alignment_gives_unavailable_but_independent_metrics_still_work():
    r = run_([chunk("a", 4.0)], units(), None)
    for fn in (metrics.coverage_and_integrity, metrics.semantic_lag, metrics.anticipation, metrics.claims_vs_evaluator):
        assert fn(r)["status"] == "unavailable"
    assert metrics.first_onset(r)["first_onset_s"] == pytest.approx(4.0)
    assert metrics.first_onset(run_([]))["first_onset_s"] is None and metrics.physical_queue(run_([]))["status"] == "unavailable"
