from test_incremental_fakes import alignment, chunk, ideal_run, real, run_, units

from benchmark.incremental import report
from benchmark.incremental.contracts import AlignmentBinding, InputDelivery, RefAtom, RefUnit, UnitDoneRecord

S = 1_000_000_000


def test_report_sections_provenance_and_not_certified_block():
    rep = report.build_report(ideal_run())
    md = report.render_markdown(rep)
    for h in ("Contabilidad", "Primer onset", "Cobertura SEMÁNTICA por átomos", "Cola FÍSICA", "Desfase SEMÁNTICO", "Categorías separadas", "Cierre y cancelación", "Lo que NO certifica"):
        assert h in md
    assert rep["provenance"]["test_only"] is True and "SOLO PRUEBAS" in md and "heurística temporal (NO cobertura)" in md
    assert any("temporal_support_fraction_heuristic" in x and "no es cobertura" in x for x in rep["not_certified"])
    assert any("pre-fork" in x for x in rep["not_certified"]) and any("Un hueco entre chunks" in x for x in rep["not_certified"])


def test_report_shows_unavailable_not_zero_without_alignment():
    md = report.render_markdown(report.build_report(run_([chunk("a", 4.0)], units(), None)))
    assert "unavailable: sin correspondencia independiente" in md and "cobertura semántica: 0" not in md
    assert "anticipación: unavailable" in md


def test_report_keeps_generated_audible_transport_and_revision_apart():
    ref = [RefUnit("u0", 0.0, 6.0, "human", 0.1, (RefAtom("a", "content"),))]
    rs = [real("r", "a", "u0", 0, 0, 8000, sup=(0.0, 6.0), sup_unc=0.0)]
    ch = [chunk("a", 2.0, 1.0, delivered=2.0), chunk("b", 3.5, 1.0, sup="a@0", seq=1, delivered=9.0)]
    md = report.render_markdown(report.build_report(run_(ch, ref, alignment(rs, ref), input_deliveries=[InputDelivery(0, 9.0, 3 * S)],
                                                          unit_done=None)))
    assert "generado antes del soporte (preparación privada, no es pronunciación): 1" in md and "audible antes del soporte: 1" in md
    assert "violación de frontera de TRANSPORTE (entregas): 1 de 1" in md and "revisión AUDIBLE de lo ya pronunciado: 1" in md


def test_diagnostic_tier_is_labelled_and_never_validated():
    import dataclasses

    ref = units(1)
    a = dataclasses.replace(alignment([], ref), provenance="auto_unvalidated", correction_state="raw", binding=AlignmentBinding(output_audio_sha256="b" * 64))
    md = report.render_markdown(report.build_report(run_([chunk("a", 4.0)], ref, a)))
    assert "DIAGNÓSTICO" in md and "auto_unvalidated" in md and "SOLO PRUEBAS" not in md
