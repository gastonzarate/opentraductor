"""Ejecuciones FALSAS con oracle de evaluación SEPARADO del candidato. Sin modelos.

Convenciones: ``run_`` asigna ``chunk_id`` = posición en la lista de chunks (lo hace el «arnés» de la prueba); las realizaciones del
evaluador apuntan a ``(chunk_id, muestras)``. Procedencia ``test_oracle`` = verdad inyectada por la prueba, nunca anotación humana."""
import dataclasses

from benchmark.incremental.contracts import (AlignmentBinding, AtomAssessment, AlignmentResult, AtomRealization, CandidateFields, ChunkRecord, HarnessStamp,
                                            IncrementalRun, OutputSpan, RefAtom, RefUnit)

S = 1_000_000_000
SR = 16000


def chunk(eid, recv_s, dur_s=1.0, *, rev=0, sup=None, delivered=None, offset=0.0, claim=None, speed=None, seq=0, last=False):
    return ChunkRecord(
        stamp=HarnessStamp(recv_ns=round(recv_s * S), input_seq=seq, delivered_until_s=recv_s if delivered is None else delivered),
        cand=CandidateFields(emission_id=eid, revision=rev, supersedes=sup, last_chunk=last, speed=speed, source_span_claim=claim),
        pcm_samples=round(dur_s * SR), sample_rate=SR, nonsilent_offset_s=offset, text_en=eid)


def units(n=6, length=3.0, kinds=("content", "content")):
    out = []
    for i in range(n):
        atoms = tuple(RefAtom(f"u{i}a{k}", kd, True, i * length + k * length / len(kinds), i * length + (k + 1) * length / len(kinds)) for k, kd in enumerate(kinds))
        out.append(RefUnit(f"u{i}", i * length, (i + 1) * length, provenance="human", boundary_uncertainty_s=0.1, atoms=atoms))
    return out


def real(rid, atom, unit, cid, s0, s1, *, status="realized", out_unc=0.05, sup=(None, None), sup_unc=0.1, eid=None, rev=None):
    return AtomRealization(realization_id=rid, atom_id=atom, ref_unit_id=unit, status=status, span=OutputSpan(cid, s0, s1), out_uncertainty_s=out_unc,
                           support_start_s=sup[0], support_end_s=sup[1], support_uncertainty_s=sup_unc, emission_id=eid, revision=rev)


def alignment(realizations, ref, *, completeness="complete", prov="test_oracle", correction="reviewed", assessed=None, fixture="fx", assessments=None):
    """Genera el ledger de evaluación: realized/contradicted si hay realización de ese estado; si no, omitted cuando es complete; si no, not_assessed.
    `assessed` solo fuerza el contador declarado (para probar incoherencias); `assessments` reemplaza el ledger entero."""
    total = sum(len(u.atoms) for u in ref)
    if assessments is None:
        by = {}
        for r in realizations:
            by.setdefault((r.ref_unit_id, r.atom_id), []).append(r.status)
        assessments = []
        for u in ref:
            for at in u.atoms:
                sts = by.get((u.ref_unit_id, at.atom_id), [])
                if "realized" in sts:
                    st = "realized"
                elif "contradicted" in sts:
                    st = "contradicted"
                elif sts:
                    st = "ambiguous"
                else:
                    st = "omitted" if completeness == "complete" else "not_assessed"
                assessments.append(AtomAssessment(u.ref_unit_id, at.atom_id, st))
    n_ass = sum(1 for x in assessments if x.status != "not_assessed")
    return AlignmentResult(alignment_id="t", method="fixture", version="1", provenance=prov, correction_state=correction, completeness=completeness,
                           n_atoms_total=total, n_atoms_assessed=n_ass if assessed is None else assessed,
                           binding=AlignmentBinding(fixture_id=fixture if prov == "test_oracle" else None, output_audio_sha256=None if prov == "test_oracle" else "0" * 64),
                           realizations=tuple(realizations), assessments=tuple(assessments))


def run_(chunks, ref=None, align=None, **kw):
    chunks = [c if c.chunk_id >= 0 else dataclasses.replace(c, chunk_id=i) for i, c in enumerate(chunks)]
    return IncrementalRun(t0_ns=0, input_duration_s=kw.pop("input_duration_s", 20.0), ref_units=ref if ref is not None else units(),
                          chunks=chunks, alignment=align, **kw)


def ideal_run(n=6, delay=0.5, length=3.0):
    """Cada unidad (2 átomos) se dice entera tras su fin + delay, en UNA emisión de 1 s (átomo a0 en 0-0.5 s, a1 en 0.5-1 s)."""
    ref = units(n, length)
    chunks = [chunk(f"e{i}", (i + 1) * length + delay, 1.0, seq=i, delivered=(i + 1) * length + delay) for i in range(n)]
    rs = []
    for i in range(n):
        rs.append(real(f"r{i}a0", f"u{i}a0", f"u{i}", i, 0, 8000, sup=(i * length, i * length + length / 2), eid=f"e{i}"))
        rs.append(real(f"r{i}a1", f"u{i}a1", f"u{i}", i, 8000, 16000, sup=(i * length + length / 2, (i + 1) * length), eid=f"e{i}"))
    return run_(chunks, ref, alignment(rs, ref))


def test_helpers_build_a_consistent_ideal_run():
    r = ideal_run()
    assert len(r.chunks) == 6 and [c.chunk_id for c in r.chunks] == list(range(6)) and r.alignment.validate() == []
    assert r.alignment.tier == "test_only" and r.speech_end_s == 18.0
