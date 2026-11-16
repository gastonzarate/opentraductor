"""Contratos v0 del benchmark: manifest de audio, adaptador y eventos.

Contrato confirmado por el coordinador (Hub, bench-contract-coord-v1).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Iterable

# --------------------------------------------------------------------------
# Manifest de audio
# --------------------------------------------------------------------------

SPLITS = ("development", "test")
DIALECTS = ("es-AR", "other", "unknown")
TEXT_PROVENANCE = ("human", "generated", "unknown")
BOUNDARY_METHODS = (
    "human_annotation",
    "forced_alignment_reviewed",
    "forced_alignment_auto",
    "estimated",
    "unknown",
)
# Métodos que NO pueden llamarse "humanos"/precisos sin revisión.
PRECISE_BOUNDARY_METHODS = ("human_annotation", "forced_alignment_reviewed")

REQUIRED_FIELDS = (
    "item_id",
    "recording_id",
    "source_id",
    "split",
    "audio_path",
    "sha256",
    "sample_rate",
    "channels",
    "duration_s",
    "dialect",
    "license",
    "license_url",
    "source_url",
    "attribution",
    "transcript_es_provenance",
    "reference_en_provenance",
    "boundary_method",
)


class ManifestError(ValueError):
    pass


@dataclass(frozen=True)
class ManifestItem:
    item_id: str
    recording_id: str
    source_id: str  # corpus/fuente; sirve para agrupar conservadoramente
    speaker_id: str | None  # None = desconocido
    split: str
    audio_path: str  # relativo a data/audio (o absoluto)
    sha256: str
    sample_rate: int
    channels: int
    duration_s: float
    dialect: str
    license: str
    license_url: str
    source_url: str
    attribution: str
    transcript_es: str | None
    transcript_es_provenance: str
    reference_en: tuple[str, ...] | None
    reference_en_provenance: str
    utterance_start_s: float | None
    utterance_end_s: float | None
    boundary_method: str
    boundary_uncertainty_s: float | None
    consent_note: str | None = None
    extra: dict = field(default_factory=dict, compare=False)

    @property
    def speaker_group(self) -> str:
        """Clave de agrupación para el split.

        Hablante desconocido: se agrupa conservadoramente por fuente completa.
        """
        if self.speaker_id:
            return f"spk:{self.source_id}:{self.speaker_id}"
        return f"unknown-source:{self.source_id}"

    @property
    def has_precise_end(self) -> bool:
        return (
            self.utterance_end_s is not None
            and self.boundary_method in PRECISE_BOUNDARY_METHODS
        )


def _item_from_dict(d: dict[str, Any], line_no: int) -> ManifestItem:
    missing = [k for k in REQUIRED_FIELDS if k not in d]
    if missing:
        raise ManifestError(f"línea {line_no}: faltan campos {missing}")
    if d["split"] not in SPLITS:
        raise ManifestError(f"línea {line_no}: split inválido {d['split']!r}")
    if d["dialect"] not in DIALECTS:
        raise ManifestError(f"línea {line_no}: dialect inválido {d['dialect']!r}")
    for key in ("transcript_es_provenance", "reference_en_provenance"):
        if d[key] not in TEXT_PROVENANCE:
            raise ManifestError(f"línea {line_no}: {key} inválido {d[key]!r}")
    if d["boundary_method"] not in BOUNDARY_METHODS:
        raise ManifestError(f"línea {line_no}: boundary_method inválido")
    end = d.get("utterance_end_s")
    start = d.get("utterance_start_s")
    if end is not None or start is not None:
        if d["boundary_method"] == "unknown":
            raise ManifestError(
                f"línea {line_no}: límites sin boundary_method declarado"
            )
        if d["boundary_method"] != "human_annotation" and d.get(
            "boundary_uncertainty_s"
        ) is None:
            raise ManifestError(
                f"línea {line_no}: límites no humanos requieren boundary_uncertainty_s"
            )
    if end is not None and start is not None and end < start:
        raise ManifestError(f"línea {line_no}: utterance_end_s < utterance_start_s")
    if end is not None and end > d["duration_s"] + 1e-6:
        raise ManifestError(f"línea {line_no}: utterance_end_s > duration_s")
    ref = d.get("reference_en")
    if isinstance(ref, str):
        ref = [ref]
    known = set(REQUIRED_FIELDS) | {
        "speaker_id",
        "transcript_es",
        "reference_en",
        "utterance_start_s",
        "utterance_end_s",
        "boundary_uncertainty_s",
        "consent_note",
    }
    return ManifestItem(
        item_id=d["item_id"],
        recording_id=d["recording_id"],
        source_id=d["source_id"],
        speaker_id=d.get("speaker_id"),
        split=d["split"],
        audio_path=d["audio_path"],
        sha256=d["sha256"],
        sample_rate=int(d["sample_rate"]),
        channels=int(d["channels"]),
        duration_s=float(d["duration_s"]),
        dialect=d["dialect"],
        license=d["license"],
        license_url=d["license_url"],
        source_url=d["source_url"],
        attribution=d["attribution"],
        transcript_es=d.get("transcript_es"),
        transcript_es_provenance=d["transcript_es_provenance"],
        reference_en=tuple(ref) if ref else None,
        reference_en_provenance=d["reference_en_provenance"],
        utterance_start_s=start,
        utterance_end_s=end,
        boundary_method=d["boundary_method"],
        boundary_uncertainty_s=d.get("boundary_uncertainty_s"),
        consent_note=d.get("consent_note"),
        extra={k: v for k, v in d.items() if k not in known},
    )


def validate_split_isolation(items: Iterable[ManifestItem]) -> None:
    """Falla si un hablante (o su agrupación conservadora) o una grabación
    aparece en más de un split."""
    by_group: dict[str, set[str]] = {}
    by_rec: dict[str, set[str]] = {}
    for it in items:
        by_group.setdefault(it.speaker_group, set()).add(it.split)
        by_rec.setdefault(f"{it.source_id}:{it.recording_id}", set()).add(it.split)
    bad = [k for k, v in by_group.items() if len(v) > 1]
    bad += [k for k, v in by_rec.items() if len(v) > 1]
    if bad:
        raise ManifestError(f"fuga entre splits (hablante/grabación): {sorted(bad)}")


def load_manifest(
    path: str | Path,
    *,
    audio_root: str | Path | None = None,
    verify_sha256: bool = False,
) -> list[ManifestItem]:
    path = Path(path)
    items: list[ManifestItem] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestError(f"línea {n}: JSON inválido: {exc}") from exc
            it = _item_from_dict(d, n)
            if it.item_id in seen:
                raise ManifestError(f"item_id duplicado: {it.item_id}")
            seen.add(it.item_id)
            items.append(it)
    validate_split_isolation(items)
    if verify_sha256:
        root = Path(audio_root) if audio_root else path.parent
        for it in items:
            p = resolve_audio_path(it, root)
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            if h != it.sha256:
                raise ManifestError(f"sha256 no coincide: {it.item_id}")
    return items


def resolve_audio_path(item: ManifestItem, audio_root: str | Path) -> Path:
    p = Path(item.audio_path)
    return p if p.is_absolute() else Path(audio_root) / p


def select_split(items: Iterable[ManifestItem], split: str) -> list[ManifestItem]:
    if split not in SPLITS:
        raise ManifestError(f"split inválido {split!r}")
    return [i for i in items if i.split == split]


# --------------------------------------------------------------------------
# Eventos y adaptador
# --------------------------------------------------------------------------


class AdapterMode(str, Enum):
    TEXT = "text"  # MT/TTS con texto de entrada. No es streaming de voz.
    OFFLINE_FILE = "offline_file"  # recibe archivo completo. No es streaming.
    STREAMING = "streaming"  # causal: solo ve audio ya ocurrido.


class EventKind(str, Enum):
    ASR_PARTIAL = "asr_partial"
    ASR_FINAL = "asr_final"
    ENDPOINT = "endpoint"
    MT_PARTIAL = "mt_partial"
    MT_FINAL = "mt_final"
    TTS_CHUNK = "tts_chunk"
    # Eventos standalone de ciclo de vida; no obligan a retener el último chunk.
    UNIT_DONE = "unit_done"
    UNIT_CANCELLED = "unit_cancelled"
    ERROR = "error"
    # Derivados/registrados por el arnés (un candidato no los emite):
    FIRST_PCM = "first_pcm"
    DEVICE_OUTPUT = "device_output"


@dataclass
class AdapterEvent:
    """Evento emitido por un candidato.

    Ningún tiempo del candidato se usa para medir. ``claimed_ns`` solo se
    guarda como dato no confiable. El arnés estampa su propio reloj al recibir.
    """

    kind: EventKind
    stream_id: str
    text: str | None = None  # asr_*, mt_*
    audio_pos_s: float | None = None  # asr_partial/endpoint: posición de audio
    pcm: bytes | None = None  # tts_chunk: s16le mono
    sample_rate: int | None = None
    planned_text_en: str | None = None  # tts_chunk: texto que se pretendía decir
    # Correspondencia fuente→salida (opcional pero necesaria para latencia
    # por fragmento y lag semántico en streams continuos):
    source_span_s: tuple[float, float] | None = None
    # Identificador de commit/frase dentro del stream (único por frase emitida,
    # NO por chunk). Todos los tts_chunk y el mt_final/asr_final de la misma
    # frase comparten ``commit_id``. Dos frases iguales tienen commit_id
    # distintos y ambas cuentan; los chunks de un mismo commit no se duplican.
    commit_id: str | None = None
    error_kind: str | None = None
    claimed_ns: int | None = None
    # Ciclo de vida incremental. Son claims del candidato; el arnés estampa
    # recepción/frontier y los contrasta con los chunks observados.
    revision: int = 0
    unit_n_chunks: int | None = None
    unit_total_samples: int | None = None
    before_emit_claim: bool | None = None
    unit_reason: str | None = None


class Adapter:
    """Interfaz base de candidato. Un candidato por proceso."""

    name: str = "unnamed"
    version: str = "0"
    mode: AdapterMode = AdapterMode.STREAMING
    # Los adaptadores de infraestructura deben ponerlo en True.
    infra_test_only: bool = False
    # False = el adaptador solo produce texto (p. ej. MT sin TTS): el primer
    # "output" medido es el primer mt_final, etiquetado como salida de texto.
    produces_audio: bool = True
    # False = adapter legacy: ausencia de UNIT_DONE/UNIT_CANCELLED es instrumentación
    # desconocida. True = soporte declarado; cero eventos se conserva como lista vacía.
    unit_events_supported: bool = False
    # Para text/offline_file: ms sin eventos tras el último antes de dar la petición
    # por terminada. None = valor por defecto del arnés (500). Un adaptador que
    # espera a terminar todo dentro de process_* puede declarar un valor menor.
    quiescence_ms: float | None = None

    def describe(self) -> dict[str, Any]:
        """Metadatos de reproducibilidad del candidato, registrados en run.json.

        Debe incluir lo que cambia entre iteraciones y que el arnés no puede
        deducir: revisión/hash de pesos (usar ``snapshot.weights_fingerprint``),
        versiones de motores (mlx, ctranslate2, torch...), commit/hash del código
        del adaptador, parámetros de decodificación. ``version = "0"`` no basta."""
        return {}

    async def open(self, config: dict[str, Any]) -> None:
        """Carga el modelo. El arnés mide ``cold_start_ms`` alrededor de esta
        llamada y del primer stream."""

    async def start_stream(self, stream_id: str, seed: int) -> None: ...

    # --- STREAMING ---
    async def push_audio(
        self, stream_id: str, pcm_s16le_16k_mono: bytes, t_audio_start_s: float
    ) -> None:
        raise NotImplementedError

    async def end_stream(self, stream_id: str) -> None:
        """No hay más audio (fin de la entrada, no del habla)."""

    # --- OFFLINE_FILE ---
    async def process_file(self, stream_id: str, pcm_s16le_16k_mono: bytes) -> None:
        raise NotImplementedError

    # --- TEXT ---
    async def process_text(self, stream_id: str, text_es: str) -> None:
        raise NotImplementedError

    def events(self) -> AsyncIterator[AdapterEvent]:
        raise NotImplementedError

    async def close(self) -> None: ...
