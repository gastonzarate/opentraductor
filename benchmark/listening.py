"""Escucha ciega: naturalidad, inteligibilidad, acento, expresividad, similitud.

Qué recibe el evaluador (``stimuli/``): audios con nombre ciego ``S####.wav`` y,
cuando existe, un audio de referencia ``R####.wav`` cuyo TIPO se declara por estímulo
(``reference_kind``: ``source_utterance`` = original público del mismo ítem; ``speaker_sample``
= muestra de voz que NO es el original ni tiene idioma verificado). Qué NO recibe: nombres de
sistema (``blind_key.json`` aparte).

Dimensiones no aplicables se pre-rellenan ``NA``: ``similitud_voz`` si el sistema
usa voz fija (no clona: no es un fallo de clonación) y ``expresividad`` /
``similitud_voz`` si el ítem no tiene audio de referencia. El evaluador también
puede escribir ``NA``.

Controles para interpretar el rango (no sustituyen anclas P.800 ni convierten un
proxy en MOS humano): grabación humana EN (si se aporta, con licencia) y copias
degradadas del mismo audio.

Aviso: puntuar 1-5 con pocos evaluadores no equivale a un MOS formal (ITU-T P.800).
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
import statistics
from pathlib import Path
from typing import Sequence

import numpy as np

from . import audio as A
from . import export as X

DIMENSIONS = ("naturalidad", "inteligibilidad", "acento", "expresividad", "similitud_voz")
SCALE = "1 (muy mala) a 5 (excelente)"
NA_TOKENS = {"na", "n/a", "n.a."}

INSTRUCTIONS = """# Instrucciones de escucha ciega

Escuchá cada estímulo `S####.wav` con auriculares, en un lugar silencioso, UNA vez completa
(podés repetir una segunda vez si dudás). Si hay audio de referencia `R####.wav` (columna
`reference_id`), la pantalla dice de qué tipo es: léelo antes de usarlo. Escuchalo primero.

Puntuá de 1 (muy mala) a 5 (excelente) en cada columna:
- naturalidad: qué tan natural suena como habla humana (ritmo, entonación, artefactos).
- inteligibilidad: qué tan fácil es entender lo que dice, sin ver texto.
- acento: qué tan cercano a inglés estadounidense general suena. Juzgá solo la salida; no se
  afirma ningún acento de la fuente (el dialecto de origen puede ser desconocido).
- expresividad: cuánto conserva la intención/énfasis/emoción del original del ítem. Solo se
  pide si la referencia es el original; si no, la celda ya dice NA.
- similitud_voz: qué tan parecida es la voz a la de la referencia (original o muestra de voz). Solo se pide a sistemas que
  intentan clonar la voz. Una voz fija distinta NO es un fallo: la celda dice NA.

Usá `NA` si una dimensión no se puede juzgar. No dejes celdas en blanco sin motivo (en `comentario`).
El orden de los estímulos es distinto para cada evaluador. No intentes adivinar el sistema.
Los resultados son puntuaciones subjetivas; no constituyen un MOS formal (ITU-T P.800).
"""


SOURCE_REF_INSTRUCTIONS = """## Referencia: original del ítem (`R####.wav`)
En los estímulos marcados «original del ítem», `R####.wav` es el audio fuente público (en español) de lo que dice el
estímulo. Úsala para juzgar `expresividad` y, si la celda no dice NA, `similitud_voz`.
"""

VOICE_REF_INSTRUCTIONS = """## Referencia: muestra de voz (`R####.wav`)
En los estímulos marcados «muestra de voz», `R####.wav` es una MUESTRA DE VOZ de un voluntario (licencia CC0), NO el original
de lo que dice el estímulo. Úsala solo para `similitud_voz` (si la celda no dice NA): qué tan parecida es la voz del estímulo
a la de la muestra. No se afirma en qué idioma o con qué acento habla la muestra. `expresividad` es NA en esos estímulos.
"""

REF_KINDS = {"source_utterance": SOURCE_REF_INSTRUCTIONS, "speaker_sample": VOICE_REF_INSTRUCTIONS}


def degrade_pcm(pcm: bytes, sr: int, *, cutoff_hz: float = 3400.0, snr_db: float = 15.0, seed: int = 0) -> bytes:
    """Copia degradada (paso bajo FFT + ruido blanco a SNR fijo) para anclar el
    rango de puntuaciones. Determinista por semilla."""
    x = A.to_float(pcm)
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.size, 1 / sr)
    spec[freqs > cutoff_hz] = 0
    y = np.fft.irfft(spec, n=x.size)
    sig = float(np.sqrt(np.mean(y * y))) or 1.0
    noise = np.random.default_rng(seed).normal(0, sig / (10 ** (snr_db / 20)), y.size)
    return np.clip(np.round(y + noise), -32768, 32767).astype("<i2").tobytes()


def make_blind_template(
    stimuli: Sequence[dict],
    out_dir: str | Path,
    *,
    raters: Sequence[str],
    seed: int = 0,
    extra_instructions: str = "",
) -> dict:
    """stimuli: {"system", "item_id", "audio_ref", ["audio_path", "reference_path",
    "reference_id_key", "na_dims", "is_control"]}"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    ids = [f"S{n:04d}" for n in range(len(stimuli))]
    rng.shuffle(ids)
    ref_ids: dict[str, str] = {}  # item -> R####
    key: dict[str, dict] = {}
    na_by_sid: dict[str, set] = {}
    for sid, st in zip(ids, stimuli):
        rid = None
        if st.get("reference_path"):
            rid = ref_ids.setdefault(st.get("reference_id_key", st["item_id"]), f"R{len(ref_ids):04d}")
        na = set(st.get("na_dims", ()))
        if rid is None:
            na |= {"expresividad", "similitud_voz"}
        na_by_sid[sid] = na
        key[sid] = {
            "system": st["system"],
            "item_id": st["item_id"],
            "audio_ref": st["audio_ref"],
            "reference_id": rid,
            "reference_kind": (st.get("reference_kind", "source_utterance") if rid else None),
            "is_control": bool(st.get("is_control")),
            "na_dims": sorted(na),
        }
        if st.get("audio_path"):
            (out / "stimuli").mkdir(exist_ok=True)
            shutil.copyfile(st["audio_path"], out / "stimuli" / f"{sid}.wav")
        if rid and st.get("reference_path"):
            (out / "stimuli").mkdir(exist_ok=True)
            shutil.copyfile(st["reference_path"], out / "stimuli" / f"{rid}.wav")
    rows = []
    for rater in raters:
        order = list(key)
        random.Random(f"{seed}:{rater}").shuffle(order)
        for sid in order:
            rows.append({
                "rater": rater, "stimulus_id": sid, "reference_id": key[sid]["reference_id"] or "",
                **{d: ("NA" if d in na_by_sid[sid] else "") for d in DIMENSIONS}, "comentario": "",
            })
    with (out / "ratings_template.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["rater", "stimulus_id", "reference_id", *DIMENSIONS, "comentario"])
        w.writeheader()
        w.writerows(rows)
    (out / "blind_key.json").write_text(
        json.dumps({"seed": seed, "scale": SCALE, "key": key}, indent=2, ensure_ascii=False), encoding="utf-8")
    kinds = sorted({v["reference_kind"] for v in key.values() if v["reference_kind"]})
    sections = "".join("\n" + REF_KINDS[k].strip() + "\n" for k in kinds)
    (out / "INSTRUCCIONES.md").write_text(
        INSTRUCTIONS + sections + (("\n" + extra_instructions.strip() + "\n") if extra_instructions else ""), encoding="utf-8")
    return {"n_stimuli": len(stimuli), "n_rows": len(rows), "dimensions": DIMENSIONS,
            "n_references": len(ref_ids)}


def prepare_from_indexes(
    index_paths: Sequence[str | Path],
    out_dir: str | Path,
    *,
    raters: Sequence[str],
    seed: int = 0,
    reference_audio: dict[str, str] | None = None,  # item_id -> WAV fuente (público)
    voice_reference_audio: dict[str, str] | None = None,  # item_id -> WAV de MUESTRA DE VOZ (no es el original del ítem)
    system_meta: dict[str, dict] | None = None,  # system -> {"voice_cloning": bool}
    human_controls: Sequence[tuple[str, str]] = (),  # (wav_pcm16_mono, licencia/atribución)
    degraded_controls: int = 0,
    extra_instructions: str = "",
) -> dict:
    """Arma estímulos ciegos desde índices de export. Rechaza índices corruptos.
    Solo audio sintético de benchmark y controles explícitos."""
    system_meta = system_meta or {}
    reference_audio = reference_audio or {}
    voice_reference_audio = voice_reference_audio or {}
    out = Path(out_dir)
    stimuli: list[dict] = []
    for ip in index_paths:
        rows = X.read_index(ip)
        problems = X.verify_index(rows)
        if problems:
            raise ValueError(f"índice {ip} con problemas: {problems}")
        for r in rows:
            if r.get("continuous"):
                continue
            item = "+".join(r["item_ids"])
            na = set()
            if system_meta.get(r["system"], {}).get("voice_cloning") is not True:
                na.add("similitud_voz")  # voz fija o no declarada: no aplica
            ref_path = reference_audio.get(r["item_ids"][0])
            ref_kind = "source_utterance"
            if ref_path is None and r["item_ids"][0] in voice_reference_audio:
                # Muestra de voz (p. ej. CC0), no el original del ítem: solo sirve para similitud_voz.
                # expresividad = NA (no hay audio del que conservar intención) y, si el sistema no clona, similitud = NA.
                ref_path = voice_reference_audio[r["item_ids"][0]]
                ref_kind = "speaker_sample"
                na.add("expresividad")
            stimuli.append({
                "system": r["system"], "item_id": item, "audio_ref": r["wav"],
                "audio_path": str(Path(r["_dir"]) / r["wav"]),
                "reference_path": ref_path, "reference_kind": ref_kind,
                "reference_id_key": item, "na_dims": na, "run_label": r.get("run_label"),
            })
    if not stimuli and not human_controls:
        raise ValueError("sin estímulos exportados")
    # controles
    ctrl_dir = out / "_controls_src"
    ctrl_dir.mkdir(parents=True, exist_ok=True)
    control_index: list[dict] = []
    for n, (path, lic) in enumerate(human_controls):
        pcm, sr = A.read_wav_pcm16_mono(path)
        stimuli.append({"system": "control:human_en", "item_id": f"human_control_{n}", "audio_ref": Path(path).name,
                        "audio_path": str(path), "na_dims": {"similitud_voz"}, "is_control": True,
                        "run_label": f"control humano ({lic})"})
        control_index.append({"system": "control:human_en", "item_ids": [f"human_control_{n}"], "wav": Path(path).name,
                              "_src": str(path), "license": lic})
    base = [s for s in stimuli if not s.get("is_control")]
    for n, s in enumerate(base[:degraded_controls]):
        pcm, sr = A.read_wav_pcm16_mono(s["audio_path"])
        name = f"degraded_{n}.wav"
        A.write_wav_pcm16(ctrl_dir / name, degrade_pcm(pcm, sr, seed=seed + n), sr)
        stimuli.append({"system": f"control:degraded({s['system']})", "item_id": s["item_id"], "audio_ref": name,
                        "audio_path": str(ctrl_dir / name), "reference_path": s.get("reference_path"), "reference_kind": s.get("reference_kind", "source_utterance"),
                        "reference_id_key": s["item_id"], "na_dims": {"similitud_voz"}, "is_control": True,
                        "run_label": "control degradado (paso bajo 3.4 kHz + ruido 15 dB)"})
        control_index.append({"system": f"control:degraded({s['system']})", "item_ids": [s["item_id"]], "wav": name,
                              "_src": str(ctrl_dir / name)})
    info = make_blind_template(stimuli, out, raters=raters, seed=seed, extra_instructions=extra_instructions)
    sets = {}
    for st in stimuli:
        if not st.get("is_control"):
            sets.setdefault(st["system"], set()).add(st["item_id"])
    if len(sets) > 1 and len({frozenset(v) for v in sets.values()}) > 1:
        info["warning_unpaired"] = ("los sistemas no tienen los mismos textos: las puntuaciones NO son comparables entre sistemas "
                                    "(sistema y texto quedan confundidos). Solo sirve como muestra para escuchar")
    # Índice de controles en formato export (para proxy-score), con sha256.
    recs = []
    for c in control_index:
        p = Path(c.pop("_src"))
        recs.append({**c, "run_label": "control", "outcomes": {}, "continuous": False, "roundtrip_ok": True,
                     "exported": True, "scope": "control", "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                     "_path": str(p)})
    with (out / "controls_index.jsonl").open("w", encoding="utf-8") as fh:
        for r in recs:
            r2 = dict(r)
            # verify_index espera archivo junto al índice: copiar controles allí
            shutil.copyfile(r2.pop("_path"), out / r2["wav"])
            fh.write(json.dumps(r2, ensure_ascii=False) + "\n")
    labels = sorted({s["run_label"] for s in stimuli if s.get("run_label")})
    info["run_labels"] = labels
    info["n_controls"] = len(control_index)
    if any("INFRA-TEST" in (l or "") for l in labels):
        info["warning"] = "incluye audio de adaptador DUMMY (infraestructura): no usar para juicios de calidad"
    return info


def load_ratings_rows(path: str | Path) -> list[dict]:
    """Lee puntuaciones desde CSV (plantilla) o JSON exportado por ``listening_ui``
    (``opentraductor-listening-ratings/v1``, uno o varios evaluadores en una lista).
    Valores JSON: número 1-5, ``"NA"`` o ``""``; se normalizan a texto como en el CSV."""
    p = Path(path)
    if p.suffix.lower() != ".json":
        with p.open(newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))
    doc = json.loads(p.read_text(encoding="utf-8"))
    docs = doc if isinstance(doc, list) else [doc]
    rows: list[dict] = []
    for d in docs:
        if d.get("schema") != "opentraductor-listening-ratings/v1":
            raise ValueError(f"{p}: esquema JSON no reconocido: {d.get('schema')!r}")
        for r in d["ratings"]:
            rows.append({"_package_id": d.get("package_id"), "rater": d["rater"], "stimulus_id": r["stimulus_id"], "reference_id": r.get("reference_id") or "",
                         **{k: ("" if r.get(k) is None else str(r.get(k))) for k in DIMENSIONS},
                         "comentario": r.get("comentario", "")})
    return rows


def summarize_ratings(ratings_csv: str | Path | Sequence[str | Path], blind_key: str | Path) -> dict:
    """Resume puntuaciones humanas por sistema (y controles aparte) y dimensión.
    Acepta CSV o JSON de ``listening_ui`` (un archivo o varios).
    ``NA`` = no aplica (contado aparte); vacío = sin dato; inválido se cuenta."""
    key = json.loads(Path(blind_key).read_text(encoding="utf-8"))["key"]
    paths = [ratings_csv] if isinstance(ratings_csv, (str, Path)) else list(ratings_csv)
    all_rows = [r for pth in paths for r in load_ratings_rows(pth)]
    pids = {r["_package_id"] for r in all_rows if r.get("_package_id")}
    if len(pids) > 1:
        raise ValueError(f"puntuaciones de paquetes distintos mezcladas (package_id): {sorted(pids)}")
    acc: dict[str, dict[str, list[float]]] = {}
    raters: dict[str, set[str]] = {}
    na_counts: dict[str, dict[str, int]] = {}
    n_empty = n_invalid = 0
    for row in all_rows:
        if row["stimulus_id"] not in key:
            raise ValueError(f"stimulus_id desconocido en puntuaciones: {row['stimulus_id']!r}")
        sysname = key[row["stimulus_id"]]["system"]
        for d in DIMENSIONS:
            v = (row.get(d) or "").strip()
            if not v:
                n_empty += 1
                continue
            if v.lower() in NA_TOKENS:
                na_counts.setdefault(sysname, {}).setdefault(d, 0)
                na_counts[sysname][d] += 1
                continue
            try:
                f = float(v)
            except ValueError:
                n_invalid += 1
                continue
            if not 1 <= f <= 5:
                n_invalid += 1
                continue
            acc.setdefault(sysname, {}).setdefault(d, []).append(f)
            raters.setdefault(sysname, set()).add(row["rater"])
    out: dict = {"systems": {}, "controls": {}, "n_empty_cells": n_empty, "n_invalid_cells": n_invalid,
                 "caveat": "Puntuación humana subjetiva en escala 1-5; no es MOS formal (ITU-T P.800) sin panel y diseño adecuados. "
                           "Nunca se mezcla con proxies automáticos. Los controles solo ayudan a interpretar el rango."}
    for sysname in sorted(set(acc) | set(na_counts)):
        dims = acc.get(sysname, {})
        entry = {
            "n_raters": len(raters.get(sysname, ())),
            "dimensions": {d: {"n": len(v), "mean": statistics.fmean(v), "sd": statistics.stdev(v) if len(v) > 1 else None}
                           for d, v in dims.items()},
            "not_applicable_counts": na_counts.get(sysname, {}),
        }
        (out["controls"] if sysname.startswith("control:") else out["systems"])[sysname] = entry
    return out
