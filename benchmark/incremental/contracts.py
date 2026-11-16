"""Contrato v2 de traducción incremental (PROVISIONAL hasta integrar con runtime).

Espacios que no se mezclan:

* ARNÉS: ``HarnessStamp`` (recv, input_seq, delivered_until), ``ChunkRecord.chunk_id`` (id GENERADO por el arnés), línea de tiempo
  de ENTREGA de entrada (``InputDelivery``) independiente de los chunks, y la reproducción reconstruida exactamente (con huecos).
* CANDIDATO: ``CandidateFields``, ``UnitDoneRecord``, ``UnitCancelledRecord``, ``UnitReadyRecord`` — solo CLAIMS/telemetría.
* EVALUADOR: átomos de referencia (``RefAtom``), actividad de habla fuente, y ``AlignmentResult`` con realizaciones audibles
  (``AtomRealization``) ubicadas en la salida por ``OutputSpan`` = (chunk_id del arnés, muestras). El tiempo audible se obtiene
  del instante de reproducción EXACTO del chunk + offset de muestras; nunca «primer onset de la emisión + offset».

Todo ``None`` significa DESCONOCIDO y se propaga como desconocido. Nada tiene calidad/procedencia por defecto.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

AtomKind = Literal["content", "negation", "number", "entity", "function"]
RealizationStatus = Literal["realized", "partial", "contradicted", "ambiguous", "unknown"]
Provenance = Literal["human_validated", "validated_alignment", "auto_unvalidated", "test_oracle"]
CorrectionState = Literal["raw", "reviewed", "human_corrected"]
Completeness = Literal["complete", "partial", "unknown"]
RefProvenance = Literal["human", "auto", "unknown"]
CRITICAL_KINDS = ("negation", "number", "entity")


# ------------------------------------------------------------------ ARNÉS
@dataclass(frozen=True)
class HarnessStamp:
    recv_ns: int
    input_seq: int
    delivered_until_s: float  # posición de audio fuente ya entregada al candidato cuando salió este evento
    event_seq: int = 0  # orden de llegada asignado por el arnés: desempata marcas de tiempo iguales


@dataclass(frozen=True)
class InputDelivery:
    """Entrega de audio de entrada al candidato (independiente de que haya salida): hasta ``end_s`` a ``actual_ns``."""

    seq: int
    end_s: float
    actual_ns: int


@dataclass(frozen=True)
class CandidateFields:
    emission_id: str
    revision: int = 0
    supersedes: str | None = None  # "emission_id@revision"
    committed: bool | None = None
    last_chunk: bool = False  # opcional: nunca debe obligar a retener audio
    speed: float | None = None
    source_span_claim: tuple[float, float] | None = None  # SOLO claim


@dataclass(frozen=True)
class ChunkRecord:
    stamp: HarnessStamp
    cand: CandidateFields
    pcm_samples: int
    sample_rate: int
    nonsilent_offset_s: float | None  # medido por el arnés (−50 dBFS); None = chunk silencioso
    text_en: str | None = None
    chunk_id: int = -1  # asignado por el ARNÉS; debe ser único y ≥0 (se valida)


@dataclass(frozen=True)
class UnitDoneRecord:
    stamp: HarnessStamp  # cuándo llegó el cierre respecto de los chunks
    emission_id: str
    revision: int
    n_chunks: int
    total_samples: int


@dataclass(frozen=True)
class UnitCancelledRecord:
    stamp: HarnessStamp
    emission_id: str
    revision: int
    before_emit_claim: bool = True  # claim; el arnés verifica contra la reproducción


@dataclass(frozen=True)
class UnitReadyRecord:
    """Telemetría opcional: trabajo listo para emitir. NO prueba underrun."""

    stamp: HarnessStamp
    emission_id: str
    revision: int


# ------------------------------------------------------------------ EVALUADOR
@dataclass(frozen=True)
class RefAtom:
    atom_id: str
    kind: AtomKind
    required: bool = True
    src_start_s: float | None = None
    src_end_s: float | None = None

    @property
    def critical(self) -> bool:
        return self.kind in CRITICAL_KINDS


@dataclass(frozen=True)
class RefUnit:
    ref_unit_id: str
    src_start_s: float
    src_end_s: float
    provenance: RefProvenance = "unknown"
    boundary_uncertainty_s: float | None = None  # None = desconocida
    atoms: tuple[RefAtom, ...] = ()
    anchors: tuple[str, ...] = ()


@dataclass(frozen=True)
class OutputSpan:
    """Tramo de la SALIDA: muestras [sample_start, sample_end) del chunk ``chunk_id`` generado por el arnés."""

    chunk_id: int
    sample_start: int
    sample_end: int


@dataclass(frozen=True)
class AtomRealization:
    realization_id: str  # id del EVALUADOR (independiente de emission_id/revision del candidato)
    atom_id: str
    ref_unit_id: str
    status: RealizationStatus
    span: OutputSpan | None
    out_uncertainty_s: float | None  # None = desconocida
    support_start_s: float | None
    support_end_s: float | None
    support_uncertainty_s: float | None
    emission_id: str | None = None  # solo correlación con la telemetría del candidato
    revision: int | None = None


AssessmentStatus = Literal["realized", "omitted", "contradicted", "ambiguous", "not_assessed"]


@dataclass(frozen=True)
class AtomAssessment:
    """Veredicto EXPLÍCITO del evaluador sobre un átomo (incluida la evaluación NEGATIVA: «omitido»).
    ``not_assessed`` = el evaluador no lo miró: ni cubierto ni omitido."""

    ref_unit_id: str
    atom_id: str
    status: AssessmentStatus


@dataclass(frozen=True)
class AlignmentBinding:
    """Ata la alineación al audio real exportado o a un fixture explícito. El hash del ASR es opcional (solo si el método lo usa)."""

    output_audio_sha256: str | None = None
    fixture_id: str | None = None
    asr_sha256: str | None = None


@dataclass(frozen=True)
class AlignmentResult:
    alignment_id: str
    method: str
    version: str
    provenance: Provenance
    correction_state: CorrectionState
    completeness: Completeness
    n_atoms_total: int | None
    n_atoms_assessed: int | None
    binding: AlignmentBinding
    realizations: tuple[AtomRealization, ...] = ()
    assessments: tuple[AtomAssessment, ...] = ()  # ledger de evaluación por átomo (positiva y negativa)
    notes: str = ""

    def validate(self, ref_units: "list[RefUnit] | None" = None) -> list[str]:
        """Problemas de coherencia; vacío = coherente. Con ``ref_units`` valida también el contexto real de referencias."""
        p: list[str] = []
        if not (self.binding.output_audio_sha256 or self.binding.fixture_id):
            p.append("binding: falta output_audio_sha256 o fixture_id explícito")
        if self.provenance == "human_validated" and self.correction_state == "raw":
            p.append("human_validated no puede estar en correction_state=raw")
        if self.provenance == "test_oracle" and not self.binding.fixture_id:
            p.append("test_oracle requiere fixture_id explícito")
        ids = [r.realization_id for r in self.realizations]
        if len(ids) != len(set(ids)):
            p.append("realization_id duplicado")
        for r in self.realizations:
            if r.status in ("realized", "partial", "contradicted") and r.span is None:
                p.append(f"{r.realization_id}: estado {r.status} sin span de salida")
            if r.span is not None and r.span.sample_end < r.span.sample_start:
                p.append(f"{r.realization_id}: span invertido")
        akeys = [(a.ref_unit_id, a.atom_id) for a in self.assessments]
        if len(akeys) != len(set(akeys)):
            p.append("assessment duplicado para un mismo átomo")
        n_assessed = sum(1 for a in self.assessments if a.status != "not_assessed")
        if self.n_atoms_assessed is not None and self.n_atoms_assessed != n_assessed:
            p.append(f"n_atoms_assessed={self.n_atoms_assessed} no coincide con las evaluaciones explícitas ({n_assessed})")
        if self.completeness == "complete" and (self.n_atoms_total is None or self.n_atoms_assessed != self.n_atoms_total):
            p.append("completeness=complete exige n_atoms_assessed == n_atoms_total")
        if ref_units is not None:
            uids = [u.ref_unit_id for u in ref_units]
            if len(uids) != len(set(uids)):
                p.append("ref_unit_id duplicado en las referencias")
            atoms = {(u.ref_unit_id, at.atom_id) for u in ref_units for at in u.atoms}
            if sum(len(u.atoms) for u in ref_units) != len(atoms):
                p.append("atom_id duplicado dentro de una unidad de referencia")
            if self.n_atoms_total is not None and self.n_atoms_total != len(atoms):
                p.append(f"n_atoms_total={self.n_atoms_total} no coincide con los átomos reales de la referencia ({len(atoms)})")
            for r in self.realizations:
                if (r.ref_unit_id, r.atom_id) not in atoms:
                    p.append(f"{r.realization_id}: átomo inexistente en la referencia ({r.ref_unit_id}:{r.atom_id})")
            for a in self.assessments:
                if (a.ref_unit_id, a.atom_id) not in atoms:
                    p.append(f"evaluación de un átomo inexistente ({a.ref_unit_id}:{a.atom_id})")
            if self.completeness == "complete":
                done = {(a.ref_unit_id, a.atom_id) for a in self.assessments if a.status != "not_assessed"}
                missing = atoms - done
                if missing:
                    p.append(f"completeness=complete pero faltan evaluaciones de {len(missing)} átomos")
        return p

    @property
    def tier(self) -> str:
        """validated | test_only | diagnostic (auto sin validar)."""
        return {"human_validated": "validated", "validated_alignment": "validated", "test_oracle": "test_only"}.get(self.provenance, "diagnostic")


@dataclass
class IncrementalRun:
    t0_ns: int
    input_duration_s: float
    ref_units: list[RefUnit]
    chunks: list[ChunkRecord]
    alignment: AlignmentResult | None = None  # None = no hay correspondencia independiente
    input_deliveries: list[InputDelivery] | None = None  # None = sin línea de tiempo de entrega
    source_speech_intervals: list[tuple[float, float]] | None = None  # actividad de habla fuente (≠ ref units); None = sin anotación
    unit_done: list[UnitDoneRecord] | None = None
    unit_cancelled: list[UnitCancelledRecord] | None = None
    unit_ready: list[UnitReadyRecord] | None = None
    n_streams: int = 1
    n_failed: int = 0
    n_timeout: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def speech_end_s(self) -> float | None:
        """Último fin de unidad de REFERENCIA (no sirve como actividad de habla)."""
        return max((u.src_end_s for u in self.ref_units), default=None)
