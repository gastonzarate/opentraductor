import dataclasses

import pytest
from test_incremental_fakes import alignment, chunk, ideal_run, real, run_, units

from benchmark.incremental import metrics
from benchmark.incremental.contracts import HarnessStamp, IncrementalRun, InputDelivery, RefAtom, RefUnit, UnitCancelledRecord, UnitDoneRecord, UnitReadyRecord

S = 1_000_000_000


def st(recv_s, seq=0):
    return HarnessStamp(round(recv_s * S), seq, recv_s)


def one_atom(support_end, *, sup_unc=0.0, cid=0, s1=8000):
    ref = [RefUnit("u0", 0.0, 6.0, "human", 0.1, (RefAtom("a", "content"),))]
    return ref, [real("r", "a", "u0", cid, 0, s1, sup=(0.0, support_end), sup_unc=sup_unc)]


def test_input_deliveries_are_validated_independently_of_output_chunks():
    ok = run_([], units(1), None, input_deliveries=[InputDelivery(0, 1.0, S), InputDelivery(1, 2.0, 2 * S)])
    assert metrics.transport_frontier(ok)["deliveries"]["violations"] == 0
    future = run_([], units(1), None, input_deliveries=[InputDelivery(0, 9.0, S)])  # entregó 9 s a t=1 s y NO hay chunks de salida
    d = metrics.transport_frontier(future)["deliveries"]
    assert d["violations"] == 1 and d["examples"][0]["end_s"] == 9.0
    nm = run_([], units(1), None, input_deliveries=[InputDelivery(1, 2.0, 2 * S), InputDelivery(0, 1.0, 3 * S)])
    assert metrics.transport_frontier(nm)["deliveries"]["non_monotonic"] == 1
    assert metrics.transport_frontier(run_([chunk("a", 1.0)], units(1)))["deliveries"]["status"] == "unavailable"


def test_silent_first_chunk_then_late_audible_chunk_is_not_flagged():
    # chunk0 silencioso a t=2 (frontera 2); chunk1 audible a t=5 (frontera 5); el soporte termina en 4
    ref, rs = one_atom(4.0, cid=1)
    r = run_([chunk("E", 2.0, 1.0, offset=None), chunk("E", 5.0, 1.0, delivered=5.0, seq=1)], ref, alignment(rs, ref),
             input_deliveries=[InputDelivery(0, 2.0, 2 * S), InputDelivery(1, 5.0, 5 * S)])
    an = metrics.anticipation(r)
    assert an["generated_before_support"]["n"] == 0 and an["audible_before_support"]["n"] == 0  # la silenciosa no cuenta


def test_queued_pcm_generated_early_but_heard_after_support_is_private_preparation_not_anticipation():
    # chunk A (10 s de PCM) a t=1 ocupa la reproducción; chunk B llega a t=2 (frontera 2) pero suena recién a t=11; soporte hasta 4
    ref, rs = one_atom(4.0, cid=1)
    r = run_([chunk("A", 1.0, 10.0), chunk("B", 2.0, 1.0, delivered=2.0, seq=1)], ref, alignment(rs, ref),
             input_deliveries=[InputDelivery(0, 2.0, 2 * S), InputDelivery(1, 11.0, 11 * S)])
    an = metrics.anticipation(r)
    assert an["generated_before_support"]["n"] == 1  # se GENERÓ antes de que el soporte llegara (preparación privada)
    assert an["audible_before_support"]["n"] == 0  # pero SE OYÓ cuando el soporte ya había sido entregado


def test_audible_before_support_is_flagged_but_it_is_not_a_transport_leak():
    ref, rs = one_atom(6.0)
    r = run_([chunk("E", 2.0, 1.0, delivered=2.0)], ref, alignment(rs, ref), input_deliveries=[InputDelivery(0, 2.0, 2 * S)])
    an = metrics.anticipation(r)
    assert an["audible_before_support"]["n"] == 1 and an["generated_before_support"]["n"] == 1
    assert "no prueba fuga" in an["audible_before_support"]["note"]
    assert metrics.transport_frontier(r)["deliveries"]["violations"] == 0  # transporte correcto
    leak = run_([chunk("E", 4.0, 1.0, delivered=9.0)], *(lambda a: (a[0], alignment(a[1], a[0])))(one_atom(3.0)), input_deliveries=[InputDelivery(0, 9.0, 4 * S)])
    assert metrics.transport_frontier(leak)["deliveries"]["violations"] == 1 and metrics.anticipation(leak)["audible_before_support"]["n"] == 0


def test_audible_before_support_needs_a_delivery_timeline_and_known_uncertainty():
    ref, rs = one_atom(6.0)
    no_t = metrics.anticipation(run_([chunk("E", 2.0, 1.0)], ref, alignment(rs, ref)))
    assert no_t["audible_before_support"]["status"] == "unavailable"
    ref2, rs2 = one_atom(6.0, sup_unc=None)
    unk = metrics.anticipation(run_([chunk("E", 2.0, 1.0)], ref2, alignment(rs2, ref2), input_deliveries=[InputDelivery(0, 2.0, 2 * S)]))
    assert unk["n_unclassified_unknown_support_or_uncertainty"] == 1 and unk["generated_before_support"]["n"] == 0


def test_negative_lag_with_support_already_heard_is_legitimate():
    ref, rs = one_atom(1.9, sup_unc=0.05)
    r = run_([chunk("a", 2.0, 1.0, delivered=2.0)], ref, alignment(rs, ref), input_deliveries=[InputDelivery(0, 2.0, 2 * S)])
    assert metrics.anticipation(r)["audible_before_support"]["n"] == 0
    row = metrics.semantic_lag(r)["realizations"][0]
    assert row["lag_to_support_end_s"] == pytest.approx(0.1) and row["lag_to_unit_end_s"] == pytest.approx(-4.0)  # sale antes del fin de la unidad, con soporte ya oído


def test_long_continuous_speech_60s_has_no_future_delivered():
    ref = units(20)
    ch = [chunk(f"e{i}", (i + 1) * 3.0 + 0.4, 1.0, seq=i, delivered=(i + 1) * 3.0 + 0.4) for i in range(20)]
    dl = [InputDelivery(i, i * 0.02 + 0.02, round((i * 0.02 + 0.02) * S)) for i in range(3000)]
    t = metrics.transport_frontier(run_(ch, ref, None, input_deliveries=dl))
    assert t["deliveries"]["violations"] == 0 and t["chunk_stamps"]["violations"] == 0


# ---------------------------------------------------------------- unit_done / cancel con estampa y orden
def test_unit_done_before_its_last_chunk_double_done_and_chunk_after_done_are_flagged():
    ch = [chunk("X", 4.0, 1.0), chunk("X", 5.0, 1.0, seq=1)]
    early = metrics.unit_closure(run_(ch, unit_done=[UnitDoneRecord(st(4.5), "X", 0, 2, 32000)]))["done"]
    assert early["n_issues"] == 2 and set(early["issues"]) == {"chunk_after_done", "premature_done"}
    double = metrics.unit_closure(run_(ch, unit_done=[UnitDoneRecord(st(6.0), "X", 0, 2, 32000), UnitDoneRecord(st(6.1), "X", 0, 2, 32000)]))["done"]
    assert double["issues"] == {"double_done": ["X@0"]}
    fine = metrics.unit_closure(run_(ch, unit_done=[UnitDoneRecord(st(6.0), "X", 0, 2, 32000)]))["done"]
    assert fine["n_issues"] == 0 and fine["n_unclosed"] == 0
    mism = metrics.unit_closure(run_(ch, unit_done=[UnitDoneRecord(st(6.0), "X", 0, 3, 99)]))["done"]
    assert mism["issues"] == {"claim_mismatch_final": ["X@0"]} and "no prueba pérdida" in mism["note"]


def test_unit_done_edge_cases_and_optional_closure():
    ch = [chunk("X", 4.0, 1.0)]
    r = metrics.unit_closure(run_(ch, unit_done=[UnitDoneRecord(st(5.0), "ghost", 0, 1, 10), UnitDoneRecord(st(5.0), "X", 0, -1, -5)]))["done"]
    assert "unknown_emission" in r["issues"] and "negative_counts" in r["issues"]
    assert metrics.unit_closure(run_(ch))["done"]["status"] == "unavailable"  # sin unit_done es válido; last_chunk no se exige
    assert metrics.unit_closure(run_(ch, unit_done=[]))["done"]["n_unclosed"] == 1  # sin cerrar ≠ error


def test_cancelled_after_audible_and_chunks_after_cancel_and_ready_is_not_underrun():
    ch = [chunk("A", 2.0, 1.0), chunk("A", 6.0, 1.0, seq=1), chunk("B", 5.0, 1.0, offset=None)]
    c = metrics.unit_closure(run_(ch, unit_cancelled=[UnitCancelledRecord(st(4.0), "A", 0), UnitCancelledRecord(st(5.5), "B", 0)],
                                  unit_ready=[UnitReadyRecord(st(3.0), "A", 0)]))
    assert c["cancelled"]["n_already_audible_at_cancel"] == 1 and c["cancelled"]["n_chunk_after_cancel"] == 1
    assert c["cancelled"]["n_before_emit_claim_false"] == 2 and "NO prueba underrun" in c["ready"]["note"]


# ---------------------------------------------------------------- incertidumbre de salida y tolerancia explícita (revisión root)
def _aud(support_end, out_unc, deliveries, *, sup_unc=0.0):
    ref = [RefUnit("u0", 0.0, 6.0, "human", 0.1, (RefAtom("a", "content"),))]
    rs = [real("r", "a", "u0", 0, 0, 8000, sup=(0.0, support_end), sup_unc=sup_unc, out_unc=out_unc)]
    return run_([chunk("E", 5.0, 1.0, delivered=5.0)], ref, alignment(rs, ref), input_deliveries=deliveries)


def test_audible_before_support_considers_the_uncertain_time_interval_not_a_point():
    dl = [InputDelivery(0, 2.0, 2 * S), InputDelivery(1, 6.0, 5 * S + 100_000_000)]  # la frontera salta a 6 s en t=5.1
    sure_before = metrics.anticipation(_aud(8.0, 0.5, dl))["audible_before_support"]  # ni a 5.5 s estaba entregado 8 s
    assert sure_before["n"] == 1 and sure_before["n_ambiguous"] == 0
    sure_not = metrics.anticipation(_aud(1.5, 0.05, dl))["audible_before_support"]  # ya estaba entregado desde t=2
    assert sure_not["n"] == 0 and sure_not["n_not_before"] == 1
    amb = metrics.anticipation(_aud(5.5, 0.5, dl))["audible_before_support"]  # a 4.5 s no, a 5.5 s sí: la frontera cambia dentro del intervalo
    assert amb["n"] == 0 and amb["n_ambiguous"] == 1 and amb["ids_ambiguous"] == ["r"]


def test_unknown_output_uncertainty_leaves_audible_anticipation_unclassified():
    dl = [InputDelivery(0, 2.0, 2 * S)]
    out = metrics.anticipation(_aud(8.0, None, dl))["audible_before_support"]
    assert out["n"] == 0 and out["n_not_before"] == 0 and out["n_unclassified_unknown_output_uncertainty"] == 1  # no se clasifica como cero/seguro


def test_transport_is_strict_by_default_and_has_no_free_tolerance():
    assert "input_precision_s" not in {f.name for f in dataclasses.fields(IncrementalRun)}
    ahead = [InputDelivery(0, 1.03, S)]  # 30 ms por delante del reloj: SIEMPRE violación
    none = metrics.transport_frontier(run_([], units(1), None, input_deliveries=ahead))
    assert none["deliveries"]["violations"] == 1 and none["tolerance_s"] == 0.0 and "estricto" in none["tolerance_basis"]
    with pytest.raises(TypeError):
        run_([], units(1), None, input_deliveries=ahead, input_precision_s=0.05)
    noise = metrics.transport_frontier(run_([], units(1), None, input_deliveries=[InputDelivery(i, i * 0.02 + 0.02, round((i * 0.02 + 0.02) * S)) for i in range(500)]))
    assert noise["deliveries"]["violations"] == 0  # ruido de coma flotante (1 µs) no es una fuga


def test_event_seq_breaks_timestamp_ties_for_unit_done_versus_chunks():
    def stamp(ns_s, ev):
        return HarnessStamp(round(ns_s * S), 0, ns_s, event_seq=ev)
    c0 = chunk("X", 5.0, 1.0)
    c1 = chunk("X", 5.0, 1.0)
    ch = [dataclasses.replace(c0, stamp=stamp(5.0, 1)), dataclasses.replace(c1, stamp=stamp(5.0, 2))]
    ok = metrics.unit_closure(run_(ch, unit_done=[UnitDoneRecord(stamp(5.0, 3), "X", 0, 2, 32000)]))["done"]  # el cierre llegó DESPUÉS en el arnés
    assert ok["n_issues"] == 0
    early = metrics.unit_closure(run_(ch, unit_done=[UnitDoneRecord(stamp(5.0, 1), "X", 0, 2, 32000)]))["done"]  # mismo timestamp, antes del 2.º chunk
    assert early["n_issues"] >= 1 and "chunk_after_done" in early["issues"]


def test_already_audible_at_cancel_requires_the_chunk_to_precede_the_cancel_in_the_harness():
    def stamp(ev):
        return HarnessStamp(round(5.0 * S), 0, 5.0, event_seq=ev)
    ch = [dataclasses.replace(chunk("X", 5.0, 1.0), stamp=stamp(2))]
    after = metrics.unit_closure(run_(ch, unit_cancelled=[UnitCancelledRecord(stamp(1), "X", 0, False)]))["cancelled"]  # cancel(5, seq1) < chunk(5, seq2)
    assert after["n_chunk_after_cancel"] == 1 and after["n_already_audible_at_cancel"] == 0
    before = metrics.unit_closure(run_(ch, unit_cancelled=[UnitCancelledRecord(stamp(3), "X", 0, False)]))["cancelled"]  # espejo: el chunk llegó antes
    assert before["n_chunk_after_cancel"] == 0 and before["n_already_audible_at_cancel"] == 1
