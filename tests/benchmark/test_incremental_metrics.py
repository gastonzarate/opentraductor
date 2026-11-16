import dataclasses

import pytest
from test_incremental_fakes import alignment, chunk, ideal_run, real, run_, units

from benchmark.incremental import metrics
from benchmark.incremental.contracts import RefAtom, RefUnit


def test_ideal_run_is_fully_covered_with_expected_lags():
    r = ideal_run(delay=0.5)
    ci = metrics.coverage_and_integrity(r)
    assert ci["semantic_atom_coverage"] == 1.0 and ci["n_units_complete"] == 6 and ci["n_repeated_atoms"] == 0 and ci["omitted_atoms"] == []
    assert ci["first_useful_atom_s"] == pytest.approx(3.5) and ci["units"][0]["unit_complete_at_s"] == pytest.approx(4.5)
    sl = metrics.semantic_lag(r)
    assert sl["median_lag_to_support_end_s"] == pytest.approx(1.5) and sl["median_lag_to_unit_end_s"] == pytest.approx(0.75)
    assert sl["n_uncertainty_unknown"] == 0 and metrics.audible_revisions(r)["spoken_then_revised"] == 0


def test_reused_emission_id_cannot_hide_an_audible_repeat():
    ref = units(1)
    mk = lambda e0, e1, off0=0.0: run_([chunk(e0, 4.0, 1.0, offset=off0), chunk(e1, 6.0, 1.0, rev=1 if e0 == e1 else 0)], ref, alignment([
        real("r0", "u0a0", "u0", 0, 0, 8000, eid=e0, rev=0), real("r1", "u0a0", "u0", 1, 0, 8000, eid=e1, rev=1 if e0 == e1 else 0),
        real("r2", "u0a1", "u0", 1, 8000, 16000, eid=e1)], ref))
    same_id, diff_id = mk("X", "X"), mk("X", "Y")
    a, b = metrics.coverage_and_integrity(same_id), metrics.coverage_and_integrity(diff_id)
    assert a["n_repeated_atoms"] == b["n_repeated_atoms"] == 1  # mismo PCM, mismo resultado con IDs X/X o X/Y
    # rev0 retirada antes de oírse (chunk silencioso): no es repetición audible
    retired = mk("X", "X", off0=None)
    assert metrics.coverage_and_integrity(retired)["n_repeated_atoms"] == 0


def test_one_long_emission_covering_two_units_gets_a_different_lag_per_audible_interval():
    ref = units(2)
    rs = [real("a", "u0a0", "u0", 0, 0, 8000, sup=(0, 1.5)), real("b", "u0a1", "u0", 0, 8000, 16000, sup=(1.5, 3)),
          real("c", "u1a0", "u1", 0, 16000, 24000, sup=(3, 4.5)), real("d", "u1a1", "u1", 0, 24000, 32000, sup=(4.5, 6))]
    r = run_([chunk("X", 6.5, 2.0)], ref, alignment(rs, ref))
    by = {x["realization_id"]: x for x in metrics.semantic_lag(r)["realizations"]}
    assert by["a"]["audible_start_s"] == pytest.approx(6.5) and by["c"]["audible_start_s"] == pytest.approx(7.5)
    assert by["a"]["lag_to_unit_end_s"] == pytest.approx(3.5) and by["c"]["lag_to_unit_end_s"] == pytest.approx(1.5)  # ya no «primer onset para todas»
    units_ = {u["ref_unit_id"]: u for u in metrics.coverage_and_integrity(r)["units"]}
    assert units_["u0"]["unit_complete_at_s"] == pytest.approx(7.5) and units_["u1"]["unit_complete_at_s"] == pytest.approx(8.5)


def test_minimal_first_fragment_does_not_give_the_whole_unit_its_lag():
    # unidad [0,10] con átomos a(0-1) b(1-8) c(8-10); e1 (t=2) dice a; e2 (t=20) dice b; c no se dice
    u = RefUnit("u0", 0.0, 10.0, "human", 0.1, (RefAtom("a", "content", True, 0, 1), RefAtom("b", "content", True, 1, 8), RefAtom("c", "content", True, 8, 10)))
    rs = [real("r1", "a", "u0", 0, 0, 8000, sup=(0, 1)), real("r2", "b", "u0", 1, 0, 8000, sup=(1, 8))]
    r = run_([chunk("e1", 2.0), chunk("e2", 20.0)], [u], alignment(rs, [u]), input_duration_s=30.0)
    ci = metrics.coverage_and_integrity(r)
    assert ci["units"][0]["unit_complete"] is False and ci["units"][0]["unit_complete_at_s"] is None and ci["omitted_atoms"] == ["u0:c"]
    assert ci["first_useful_atom_s"] == pytest.approx(2.0)
    assert ci["temporal_support_fraction_heuristic"]["units"][0]["crossing_time_s"] == pytest.approx(20.0)  # la unión cruza 0.8 recién en e2
    lags = {x["realization_id"]: x["lag_to_support_end_s"] for x in metrics.semantic_lag(r)["realizations"]}
    assert lags == {"r1": pytest.approx(1.0), "r2": pytest.approx(12.0)} and -8.0 not in lags.values()


def test_semantic_coverage_needs_completeness_and_unobserved_is_not_omitted():
    u = RefUnit("u0", 0.0, 10.0, "human", 0.1, (RefAtom("a", "content"), RefAtom("b", "content"), RefAtom("c", "content")))
    rs = [real("r1", "a", "u0", 0, 0, 8000, sup=(0, 1))]
    part = metrics.coverage_and_integrity(run_([chunk("e1", 2.0)], [u], alignment(rs, [u], completeness="partial", assessed=1)))
    assert part["semantic_atom_coverage"] is None and part["semantic_atom_coverage_lower_bound"] == pytest.approx(1 / 3)
    assert part["omitted_atoms"] == [] and part["units"][0]["atoms"]["b"] == "not_observed"
    comp = metrics.coverage_and_integrity(run_([chunk("e1", 2.0)], [u], alignment(rs, [u])))
    assert comp["semantic_atom_coverage"] == pytest.approx(1 / 3) and comp["omitted_atoms"] == ["u0:b", "u0:c"]


def test_missing_critical_negation_makes_the_unit_critically_incomplete_even_with_most_atoms_covered():
    u = RefUnit("u0", 0.0, 5.0, "human", 0.1, (RefAtom("a", "content"), RefAtom("b", "content"), RefAtom("c", "content"), RefAtom("d", "content"),
                                              RefAtom("neg", "negation")))
    rs = [real(f"r{n}", n, "u0", 0, i * 1000, i * 1000 + 1000, sup=(0, 5)) for i, n in enumerate("abcd")]
    ci = metrics.coverage_and_integrity(run_([chunk("e", 6.0, 1.0)], [u], alignment(rs, [u])))
    assert ci["semantic_atom_coverage"] == pytest.approx(0.8)
    assert ci["units"][0]["unit_complete"] is False and ci["units"][0]["critically_incomplete"] and ci["units"][0]["critical_atoms_not_realized"] == ["neg"]


def test_unit_completeness_requires_all_required_atoms_not_only_critical_ones():
    u = RefUnit("u0", 0.0, 4.0, "human", 0.1, (RefAtom("a", "content"), RefAtom("b", "content"), RefAtom("n", "number")))
    rs = [real("r1", "n", "u0", 0, 0, 8000, sup=(0, 4)), real("r2", "a", "u0", 0, 8000, 16000, sup=(0, 4))]
    ci = metrics.coverage_and_integrity(run_([chunk("e", 5.0, 1.0)], [u], alignment(rs, [u])))
    assert ci["units"][0]["unit_complete"] is False and ci["units"][0]["atoms"]["b"] == "omitted" and not ci["units"][0]["critically_incomplete"]


def test_exact_playthrough_with_gaps_is_used_not_first_onset_plus_offset():
    ref = [RefUnit("u0", 0, 2, "human", 0.1, (RefAtom("a", "content"), RefAtom("b", "content"), RefAtom("c", "content")))]
    # chunk0: 0.3 s de silencio inicial; chunk1 llega 3 s después (hueco)
    ch = [chunk("e0", 1.0, 1.0, offset=0.3), chunk("e1", 5.0, 1.0, offset=0.0)]
    rs = [real("a", "a", "u0", 0, 8000, 16000, sup=(0, 1)), real("b", "b", "u0", 1, 8000, 16000, sup=(0, 1)),
          real("c", "c", "u0", 0, 0, 3000, sup=(0, 1))]  # c cae dentro del silencio inicial: no audible
    r = run_(ch, ref, alignment(rs, ref))
    by = {x["realization_id"]: x["audible_start_s"] for x in metrics.semantic_lag(r)["realizations"]}
    assert by == {"a": pytest.approx(1.5), "b": pytest.approx(5.5)}  # inicio de reproducción del chunk + muestras, con el hueco incluido
    assert metrics.coverage_and_integrity(r)["units"][0]["atoms"]["c"] == "unattributable"  # el ledger dice «realizado» pero su span no es audible: ni cubierto ni omisión verificada


def test_interchunk_gap_is_not_an_underrun_and_growth_is_physical_only():
    q = metrics.physical_queue(run_([chunk("a", 0.0, 1.0), chunk("b", 3.0, 1.0)], units(1)))
    assert q["interchunk_gap_s"] == pytest.approx(2.0) and q["n_interchunk_gaps"] == 1 and q["underrun_demonstrated_s"] is None
    ch = [chunk(f"e{i}", i * 0.8, 1.0, seq=i) for i in range(10)]
    g = metrics.physical_queue(run_(ch, units(1)))
    assert g["growth_s_per_s"] > 0.1 and g["max_s"] > g["mean_s"] > 1.0 and g["interchunk_gap_s"] == 0
    assert metrics.physical_queue(run_(ch[:3], units(1)))["growth_s_per_s"] is None
    ready = metrics.physical_queue(run_([chunk("a", 0.0), chunk("b", 3.0)], units(1), unit_ready=[]))
    assert ready["underrun_demonstrated_s"] is None and ready["unit_ready_events"] == 0


def test_source_activity_is_independent_of_reference_units_and_a_pause_is_not_speech():
    ch = [chunk("a", 4.0, 1.0)]
    pause = metrics.first_onset(run_(ch, units(6), None, source_speech_intervals=[(0.0, 3.0), (10.0, 12.0)]))
    assert pause["source_active_at_onset"] is False and pause["during_speech"] is False and pause["future_speech_remaining_s"] == pytest.approx(2.0)
    inside = metrics.first_onset(run_([chunk("a", 11.0, 1.0)], units(6), None, source_speech_intervals=[(0.0, 3.0), (10.0, 12.0)]))
    assert inside["source_active_at_onset"] is True and inside["future_speech_remaining_s"] == pytest.approx(1.0) and inside["during_speech"] is True
    none = metrics.first_onset(run_(ch, units(6), None))  # antes se infería del último ref y marcaba «durante habla»
    assert none["during_speech"] is None and none["source_active_at_onset"] is None and none["before_recording_end"] is True


def test_unknown_uncertainty_stays_unknown_and_disables_sign_classification():
    ref = [RefUnit("u0", 0, 3, "unknown", None, (RefAtom("a", "content"),))]
    r = run_([chunk("a", 2.0, 1.0)], ref, alignment([real("r", "a", "u0", 0, 0, 8000, sup=(0, 3), sup_unc=None)], ref))
    row = metrics.semantic_lag(r)["realizations"][0]
    assert row["lag_to_support_end_uncertainty_s"] is None and row["negative_beyond_uncertainty"] is None and row["lag_to_unit_end_uncertainty_s"] is None
    an = metrics.anticipation(r)
    assert an["n_unclassified_unknown_support_or_uncertainty"] == 1


def test_no_evaluable_realizations_is_unavailable_not_zero():
    ref = units(1)
    r = run_([chunk("a", 4.0)], ref, alignment([], ref))
    for fn in (metrics.semantic_lag, metrics.anticipation):
        out = fn(r)
        assert out["status"] == "unavailable" and "n_evaluable=0" in out["reason"]
    neg = metrics.coverage_and_integrity(r)  # evaluación NEGATIVA explícita y completa: cobertura 0 con omisiones verificadas
    assert neg["status"] == "ok" and neg["semantic_atom_coverage"] == 0.0 and neg["omitted_atoms"] == ["u0:u0a0", "u0:u0a1"]
    none = metrics.coverage_and_integrity(run_([chunk("a", 4.0)], ref, alignment([], ref, completeness="partial", assessments=[])))
    assert none["status"] == "unavailable" and "sin evaluación" in none["reason"]
    unk = run_([chunk("a", 4.0)], ref, alignment([real("r", "u0a0", "u0", 0, 0, 100, status="unknown")], ref))
    assert metrics.semantic_lag(unk)["status"] == "unavailable"
    assert metrics.coverage_and_integrity(unk)["n_unattributable"] == 1


def test_audible_revision_by_harness_and_failures_in_denominator():
    spoken = run_([chunk("A", 2.0, 1.0), chunk("B", 3.5, 1.0, sup="A@0")])
    assert metrics.audible_revisions(spoken)["spoken_then_revised"] == 1
    silent = run_([chunk("A", 2.0, 1.0, offset=None), chunk("B", 3.5, 1.0, sup="A@0")])
    assert metrics.audible_revisions(silent)["spoken_then_revised"] == 0 and metrics.audible_revisions(silent)["revised_before_audible"] == 1
    a = metrics.accounting(run_([chunk("a", 4.0)], n_streams=4, n_failed=1, n_timeout=1))
    assert a["n_observed"] == 2 and a["failure_rate"] == 0.25 and a["timeout_rate"] == 0.25


def test_one_word_answer_and_filler():
    ref = [RefUnit("u0", 0.0, 0.6, "human", 0.05, (RefAtom("no", "negation"),))]
    ok = run_([chunk("no", 1.0, 0.4)], ref, alignment([real("r", "no", "u0", 0, 0, 6400, sup=(0, 0.6))], ref))
    ci = metrics.coverage_and_integrity(ok)
    assert ci["n_units_complete"] == 1 and ci["first_useful_atom_s"] == pytest.approx(1.0)
    filler = run_([chunk("uh", 0.5, 0.3), chunk("no", 1.0, 0.4)], ref, alignment([real("r", "no", "u0", 1, 0, 6400, sup=(0, 0.6)),
                                                                              real("f", "no", "u0", 0, 0, 4800, status="unknown")], ref))
    assert metrics.first_onset(filler)["first_onset_s"] == pytest.approx(0.5)  # el relleno es el primer onset...
    assert metrics.coverage_and_integrity(filler)["first_useful_atom_s"] == pytest.approx(1.0)  # ...pero no es la primera unidad útil


def test_claims_vs_evaluator_diagnostic_only():
    ref = units(1)
    ch = [chunk("a", 4.0, 1.0, claim=(0.0, 99.0))]
    rs = [real("r", "u0a0", "u0", 0, 0, 8000, sup=(0, 1.5))]
    r = run_(ch, ref, alignment(rs, ref))
    assert metrics.claims_vs_evaluator(r)["n_disagree"] == 1
    no_claim = dataclasses.replace(r.chunks[0], cand=dataclasses.replace(r.chunks[0].cand, source_span_claim=None))
    same = metrics.coverage_and_integrity(dataclasses.replace(r, chunks=[no_claim]))
    assert metrics.coverage_and_integrity(r)["n_atoms_realized"] == same["n_atoms_realized"]  # el claim no cambia la cobertura


# ---------------------------------------------------------------- spans inválidos y silencio inicial (revisión root)
def _one(span, *, off=0.0, status="realized", completeness="complete", recv=2.0, dur=1.0):
    ref = [RefUnit("u0", 0.0, 2.0, "human", 0.1, (RefAtom("a", "content"), RefAtom("b", "content")))]
    rs = [real("r", "a", "u0", span[0], span[1], span[2], status=status, sup=(0.0, 1.0)), real("s", "b", "u0", 0, 0, 8000, sup=(1.0, 2.0))]
    return run_([chunk("e", recv, dur, offset=off)], ref, alignment(rs, ref, completeness=completeness))


def test_span_must_lie_inside_the_chunk_and_invalid_spans_are_not_omissions():
    # chunk de 1 s a 16 kHz (16000 muestras); el span 32000:40000 terminaría a los 3 s aunque el audio acabó a los 3 s del reloj
    r = _one((0, 32000, 40000))
    sl = metrics.semantic_lag(r)
    assert sl["n_invalid_spans"] == 1 and "fuera de [0,16000]" in sl["invalid_spans"][0]["motivo"]
    assert [x["realization_id"] for x in sl["realizations"]] == ["s"]  # la inválida no recibe audible_start
    ci = metrics.coverage_and_integrity(r)
    assert ci["units"][0]["atoms"]["a"] == "unattributable" and ci["omitted_atoms"] == [] and ci["n_unattributable"] == 1
    assert ci["completeness"] == "complete" and ci["semantic_atom_coverage"] is None and ci["semantic_atom_coverage_lower_bound"] == pytest.approx(0.5)
    assert ci["n_invalid_spans"] == 1


@pytest.mark.parametrize("span", [(0, 100, 100), (0, -5, 100), (7, 0, 100), (0, 0, 16001)])
def test_other_invalid_spans_are_rejected(span):
    out = metrics.semantic_lag(_one(span))
    assert out["n_invalid_spans"] == 1 and "r" not in [x["realization_id"] for x in out["realizations"]]


def test_span_crossing_initial_silence_starts_at_first_audio_and_matches_first_onset():
    r = _one((0, 0, 8000), off=0.3, recv=2.0)
    by = {x["realization_id"]: x for x in metrics.semantic_lag(r)["realizations"]}
    assert by["r"]["audible_start_s"] == pytest.approx(2.3)  # no 2.0: el silencio inicial no cuenta como voz
    assert metrics.first_onset(r)["first_onset_s"] == pytest.approx(2.3)
    inside = _one((0, 0, 4000), off=0.3)  # termina a 0.25 s < 0.3 s: todo silencio
    assert "r" not in [x["realization_id"] for x in metrics.semantic_lag(inside)["realizations"]]
    late = _one((0, 8000, 16000), off=0.3)  # empieza después del primer audio: usa su propio inicio
    assert {x["realization_id"]: x for x in metrics.semantic_lag(late)["realizations"]}["r"]["audible_start_s"] == pytest.approx(2.5)


def test_inverted_span_makes_the_whole_alignment_incoherent_not_zero():
    out = metrics.semantic_lag(_one((0, 9000, 8000)))
    assert out["status"] == "unavailable" and "incoherente" in out["reason"] and "invertido" in out["reason"]


# ---------------------------------------------------------------- v3: validación contextual, ledger, titular, pendiente
from benchmark.incremental.contracts import AtomAssessment  # noqa: E402


def _incoherent(r):
    return [metrics.coverage_and_integrity(r), metrics.semantic_lag(r)]


def test_ghost_realization_or_ghost_atom_makes_the_alignment_incoherent():
    ref = units(1)
    ghost_unit = alignment([real("g", "u0a0", "NO_EXISTE", 0, 0, 8000, sup=(0, 1))], ref)
    ghost_atom = alignment([real("g", "zzz", "u0", 0, 0, 8000, sup=(0, 1))], ref)
    ghost_assess = alignment([], ref, assessments=[AtomAssessment("u0", "fantasma", "realized")], completeness="partial")
    for al in (ghost_unit, ghost_atom, ghost_assess):
        for out in _incoherent(run_([chunk("a", 4.0)], ref, al)):
            assert out["status"] == "unavailable" and "incoherente" in out["reason"]


def test_counters_must_match_the_real_references_and_ids_must_be_unique():
    ref = units(2)
    al = alignment([], ref)
    wrong_total = dataclasses.replace(al, n_atoms_total=99)
    assert any("n_atoms_total" in x for x in wrong_total.validate(ref))
    dup_unit = [ref[0], dataclasses.replace(ref[1], ref_unit_id="u0")]
    assert any("ref_unit_id duplicado" in x for x in al.validate(dup_unit))
    dup_atom = [dataclasses.replace(ref[0], atoms=(ref[0].atoms[0], ref[0].atoms[0])), ref[1]]
    assert any("atom_id duplicado" in x for x in alignment([], dup_atom).validate(dup_atom))
    assert al.validate(ref) == []


def test_full_negative_assessment_is_coverage_zero_with_verified_omissions_and_no_assessment_is_unavailable():
    ref = units(2)
    r = run_([chunk("a", 4.0)], ref, alignment([], ref))
    out = metrics.coverage_and_integrity(r)
    assert out["status"] == "ok" and out["semantic_atom_coverage"] == 0.0 and len(out["omitted_atoms"]) == 4 and out["n_not_assessed"] == 0
    none = metrics.coverage_and_integrity(run_([chunk("a", 4.0)], ref, alignment([], ref, completeness="partial")))
    assert none["status"] == "unavailable" and "sin evaluación" in none["reason"]


def test_not_assessed_atoms_are_neither_covered_nor_omitted_and_coverage_is_a_lower_bound():
    ref = units(1)
    rs = [real("r", "u0a0", "u0", 0, 0, 8000, sup=(0, 1))]
    out = metrics.coverage_and_integrity(run_([chunk("a", 4.0)], ref, alignment(rs, ref, completeness="partial")))
    assert out["semantic_atom_coverage"] is None and out["semantic_atom_coverage_lower_bound"] == 0.5
    assert out["omitted_atoms"] == [] and out["not_assessed_atoms"] == ["u0:u0a1"] and out["n_not_assessed"] == 1


def test_headline_lag_uses_only_realized_and_partial_contradicted_stay_visible_but_never_improve_it():
    ref = units(1)
    rs = [real("ok", "u0a0", "u0", 0, 0, 8000, sup=(0, 1)),
          real("p", "u0a1", "u0", 1, 0, 8000, sup=(0, 1), status="partial"),
          real("c", "u0a1", "u0", 2, 0, 8000, sup=(0, 1), status="contradicted")]
    ch = [chunk("a", 3.0), chunk("b", 4.0), chunk("c", 5.0)]
    lag = metrics.semantic_lag(run_(ch, ref, alignment(rs, ref)))
    assert [x["realization_id"] for x in lag["realizations"]] == ["ok"] and lag["n_realizations"] == 1
    assert lag["n_partial_excluded_from_headline"] == 1 and lag["n_contradicted_excluded_from_headline"] == 1
    assert {x["realization_id"] for x in lag["other_status_realizations"]} == {"p", "c"}
    only_partial = metrics.semantic_lag(run_(ch[:2], ref, alignment(rs[1:2], ref)))
    assert only_partial["status"] == "unavailable" and only_partial["n_partial"] == 1


def test_lag_slope_is_measured_against_source_time_not_audible_start():
    n = 6  # soporte que termina en 1..5(+) y lag 0..4 ⇒ crecimiento 1.0 contra tiempo fuente (no 0.5)
    ref = units(n, 3.0)
    ch, rs = [], []
    for i in range(n):
        end = 1.0 + i
        ch.append(chunk(f"e{i}", end + i, 1.0, seq=i))  # lag = i (audible_start = end + i)
        rs.append(real(f"r{i}", f"u{i}a0", f"u{i}", i, 0, 8000, sup=(end - 1, end)))
    lag = metrics.semantic_lag(run_(ch, ref, alignment(rs, ref)))
    assert lag["slope_support_end_s_per_s"] == pytest.approx(1.0) and lag["status"] == "ok"
    assert "FUENTE" in lag["slope_basis"]


def test_missing_support_end_and_unknown_uncertainty_are_counted():
    ref = units(1)
    rs = [real("a", "u0a0", "u0", 0, 0, 8000, sup=(0, 1), sup_unc=None), real("b", "u0a1", "u0", 0, 8000, 16000, sup=(None, None))]
    lag = metrics.semantic_lag(run_([chunk("a", 4.0)], ref, alignment(rs, ref)))
    assert lag["n_support_end_known"] == 1 and lag["n_support_end_missing"] == 1 and lag["n_uncertainty_unknown"] >= 1


def test_malformed_supersedes_is_a_diagnostic_not_an_exception():
    ch = [chunk("a", 1.0), chunk("b", 2.0, sup="a@xyz")]
    out = metrics.audible_revisions(run_(ch))
    assert out["n_malformed_supersedes"] == 1 and out["malformed_supersedes"][0]["supersedes"] == "a@xyz"


# ---------------------------------------------------------------- v3.1: una observación por átomo, historial de contradicción, gramática supersedes
def test_headline_lag_is_one_observation_per_atom_not_per_reemission():
    ref = units(2, 3.0, kinds=("content", "content"))
    # átomo a: lags 1, 2, 3 (reemisiones); átomo b: lag 9. Titular = mediana de [1, 9] = 5, nunca la de [1, 2, 3, 9] = 2.5
    ch = [chunk("e1", 3.0 + 1, seq=0), chunk("e2", 3.0 + 2, seq=1), chunk("e3", 3.0 + 3, seq=2), chunk("e4", 3.0 + 9, seq=3)]
    rs = [real("a1", "u0a0", "u0", 0, 0, 8000, sup=(0, 3)), real("a2", "u0a0", "u0", 1, 0, 8000, sup=(0, 3)), real("a3", "u0a0", "u0", 2, 0, 8000, sup=(0, 3)),
          real("b", "u0a1", "u0", 3, 0, 8000, sup=(0, 3))]
    lag = metrics.semantic_lag(run_(ch, ref, alignment(rs, ref)))
    assert lag["n_atoms_in_headline"] == 2 and lag["n_realized_realizations_total"] == 4 and lag["n_reemitted_excluded_from_headline"] == 2
    assert lag["median_lag_to_support_end_s"] == pytest.approx(5.0)
    assert {x["realization_id"] for x in lag["realizations"]} == {"a1", "b"} and len(lag["reemitted_realizations"]) == 2


def test_audible_contradiction_is_kept_in_history_even_if_later_corrected():
    ref = [RefUnit("u0", 0, 3, "human", 0.1, (RefAtom("n", "number"), RefAtom("w", "content")))]
    rs = [real("c", "n", "u0", 0, 0, 8000, status="contradicted", sup=(0, 1)), real("ok", "n", "u0", 1, 0, 8000, sup=(0, 1)), real("w", "w", "u0", 1, 8000, 16000, sup=(1, 2))]
    ch = [chunk("a", 2.0, seq=0), chunk("b", 10.0, seq=1)]
    out = metrics.coverage_and_integrity(run_(ch, ref, alignment(rs, ref)))
    assert out["semantic_atom_coverage"] == 1.0 and out["units"][0]["atoms"]["n"] == "realized"  # cobertura EVENTUAL
    assert out["units"][0]["had_contradiction"] is True and out["units"][0]["critical_contradiction_heard"] is True
    assert out["n_atoms_contradiction_heard"] == 1 and out["n_critical_contradiction_heard"] == 1
    # el ledger explícito «contradicho» no borra el historial ni niega la corrección posterior
    led = alignment(rs, ref, assessments=[AtomAssessment("u0", "n", "contradicted"), AtomAssessment("u0", "w", "realized")])
    out2 = metrics.coverage_and_integrity(run_(ch, ref, led))
    assert out2["units"][0]["atoms"]["n"] == "realized" and out2["n_atoms_contradiction_heard"] == 1 and out2["ledger_conflicts"] == []


def test_ledger_omitted_with_any_audible_realization_is_a_conflict():
    ref = units(1)
    rs = [real("c", "u0a0", "u0", 0, 0, 8000, status="contradicted", sup=(0, 1))]
    led = alignment(rs, ref, assessments=[AtomAssessment("u0", "u0a0", "omitted"), AtomAssessment("u0", "u0a1", "omitted")])
    out = metrics.coverage_and_integrity(run_([chunk("a", 4.0)], ref, led))
    assert out["ledger_conflicts"] == ["u0:u0a0"] and "u0:u0a0" not in out["omitted_atoms"] and out["omitted_atoms"] == ["u0:u0a1"]


def test_supersedes_requires_id_at_integer_revision():
    for bad in ("a", "a@", "@1", "a@x", "a@-1", "a@1.5"):
        out = metrics.audible_revisions(run_([chunk("a", 1.0), chunk("b", 2.0, sup=bad)]))
        assert out["n_malformed_supersedes"] == 1, bad
    ok = metrics.audible_revisions(run_([chunk("a", 1.0), chunk("b", 2.0, sup="a@0")]))
    assert ok["n_malformed_supersedes"] == 0
