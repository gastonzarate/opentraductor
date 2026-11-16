"""CLI: ``python -m benchmark <comando>``.

* ``selftest``: corre el adaptador DUMMY con audio sintético y reloj virtual
  (solo infraestructura) y escribe un informe etiquetado.
* ``run``: corre un adaptador real sobre un manifest (reloj real).
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import sys
from pathlib import Path

from . import audio as A
from . import report, runner
from .adapters.dummy import DummyAdapter
from .clock import RealClock, run_virtual
from .replay import Segment
from .contracts import load_manifest, select_split
from .runner import RunError
from .plugins import PluginUnavailable


def _synthetic_items(n: int) -> list[runner.BenchItem]:
    items = []
    for i in range(n):
        speech = A.tone(1.0 + 0.1 * i, freq_hz=300 + 20 * i)
        pcm = A.silence(0.3) + speech + A.silence(0.2)
        end = 0.3 + (1.0 + 0.1 * i)
        items.append(
            runner.BenchItem(
                item_id=f"synthetic-{i}",
                recording_id=f"synthetic-rec-{i}",
                pcm=pcm,
                segment=Segment(f"synthetic-{i}", 0.3, end, "human_annotation", None),
                infra_synthetic=True,
            )
        )
    return items


def _cmd_selftest(a) -> int:
    cfg = runner.RunConfig(repetitions=3, warmup_streams=1, out_dir=a.out, n_boot=200)

    async def main(clock):
        return await runner.run_benchmark(DummyAdapter(), clock, _synthetic_items(5), cfg)

    run_virtual(main)
    print(report.write_report(a.out))
    return 0


def _load_adapter(spec: str):
    mod, _, cls = spec.partition(":")
    return getattr(importlib.import_module(mod), cls)()


def _load_callable(spec: str):
    mod, _, fn = spec.partition(":")
    return getattr(importlib.import_module(mod), fn)


def _load_asr(spec: str):
    """``faster-whisper:<modelo>`` (local, CPU) o ``modulo:funcion``."""
    if spec.startswith("faster-whisper:"):
        from .plugins.asr_local import LocalWhisperAsr

        return LocalWhisperAsr(spec.split(":", 1)[1])
    return _load_callable(spec)


def _cmd_run(a) -> int:
    if a.text_manifest:
        from .textset import load_text_items

        items = load_text_items(a.text_manifest, split=a.split, ids=a.item_ids.split(",") if a.item_ids else None)
        manifest_path = a.text_manifest
    else:
        if not a.manifest or not a.audio_root:
            print("run requiere --manifest y --audio-root, o --text-manifest", file=sys.stderr)
            return 2
        items_m = select_split(
            load_manifest(a.manifest, audio_root=a.audio_root, verify_sha256=not a.no_verify_sha256),
            a.split,
        )
        items = runner.items_from_manifest(items_m, a.audio_root)
        manifest_path = a.manifest
    cfg = runner.RunConfig(
        adapter_config=json.loads(a.adapter_config),
        repetitions=a.repetitions,
        warmup_streams=a.warmup,
        purpose="final_test" if a.split == "test" else "development",
        continuous=a.continuous,
        deadline_s=a.deadline_s,
        call_timeout_s=a.call_timeout_s,
        watchdog_stall_s=a.watchdog_stall_s,
        export_audio_dir=a.export_audio_dir,
        system_name=a.system_name,
        out_dir=a.out,
        incremental_collector=a.incremental_collector,
    )
    try:
        asr = _load_asr(a.audible_asr) if a.audible_asr else None
    except PluginUnavailable as exc:
        print(f"ASR no disponible: {exc}", file=sys.stderr)
        return 3
    try:
        asyncio.run(
            runner.run_benchmark(
                _load_adapter(a.adapter), RealClock(), items, cfg,
                manifest_path=manifest_path, audible_asr=asr,
            )
        )
    except RunError as exc:
        print(f"FALLO registrado en {a.out}/FAILURE.json: {exc}", file=sys.stderr)
        return 2
    print(report.write_report(a.out))
    return 0


def _cmd_listening_prepare(a) -> int:
    from . import listening

    ref_audio: dict[str, str] = {}
    if a.manifest:
        from .contracts import resolve_audio_path

        for it in load_manifest(a.manifest, audio_root=a.audio_root, verify_sha256=True):
            ref_audio[it.item_id] = str(resolve_audio_path(it, a.audio_root))
    meta: dict[str, dict] = {}
    for spec in a.system_meta or []:  # NOMBRE:voice_cloning=true|false
        name, _, kv = spec.rpartition(":")
        k, _, v = kv.partition("=")
        meta.setdefault(name, {})[k] = v.lower() == "true"
    info = listening.prepare_from_indexes(
        a.index, a.out, raters=a.raters, seed=a.seed, reference_audio=ref_audio, system_meta=meta,
        human_controls=[tuple(x) for x in (a.human_control or [])], degraded_controls=a.degraded_controls,
        voice_reference_audio=dict(x.split("=", 1) for x in (a.voice_reference or [])),
        extra_instructions=Path(a.note).read_text(encoding="utf-8") if a.note else "",
    )
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0


def _cmd_listening_summary(a) -> int:
    from . import listening

    print(json.dumps(listening.summarize_ratings(a.ratings, a.key), ensure_ascii=False, indent=2))
    return 0


def _cmd_listening_ui(a) -> int:
    from . import listening_ui

    print(json.dumps(listening_ui.build(a.package, a.out, seed=a.seed), ensure_ascii=False, indent=2))
    return 0


def _cmd_proxy_score(a) -> int:
    """Puntúa WAV exportados con UTMOSv2. Resultado = predictor automático, NO MOS humano."""
    from . import export as X

    try:
        from .plugins.utmos import UtmosV2

        model = UtmosV2(a.checkpoint, fold=a.fold, seed=a.seed, num_repetitions=a.num_repetitions,
                        predict_dataset=a.predict_dataset)
    except PluginUnavailable as exc:
        print(f"UTMOSv2 no disponible: {exc}", file=sys.stderr)
        return 3
    rows = [r for ix in a.index for r in X.read_index(ix)]
    problems = X.verify_index(rows)
    if problems:
        print(f"índice con problemas: {problems}", file=sys.stderr)
        return 2
    scored, errors = [], []
    for r in rows:
        if r.get("continuous"):
            continue
        try:
            scored.append({"system": r["system"], "item_ids": r["item_ids"], "wav": r["wav"], "sha256": r["sha256"],
                           "run_label": r.get("run_label"),
                           "predicted_naturalness_utmosv2": model.predict_file(Path(r["_dir"]) / r["wav"])})
        except Exception as exc:  # un WAV malo no tumba el lote; se registra
            errors.append({"wav": r["wav"], "error": f"{type(exc).__name__}: {exc}"})
    by_sys: dict = {}
    for x in scored:
        by_sys.setdefault(x["system"], []).append(x["predicted_naturalness_utmosv2"])
    out = {
        "plugin": model.describe(),
        "n_scored": len(scored), "n_errors": len(errors), "errors": errors,
        "per_system": {k: {"n": len(v), "mean_predicted": sum(v) / len(v)} for k, v in by_sys.items()},
        "scored": scored,
        "caveat": "Predictor automático de naturalidad; no es MOS humano ni sustituye la escucha ciega. "
                  "Validez para TTS es->en y voces clonadas no demostrada.",
    }
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(a.out)
    return 0


def _refs_from(a) -> dict:
    refs: dict = {}
    if a.text_manifest:
        for line in Path(a.text_manifest).read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                r = d.get("reference_en")
                refs[d["item_id"]] = ((r,) if isinstance(r, str) else tuple(r or ()) or None, d.get("reference_en_provenance"))
    if a.manifest:
        for it in load_manifest(a.manifest, audio_root=a.audio_root or Path(a.manifest).parent):
            refs[it.item_id] = (it.reference_en, it.reference_en_provenance)
    return refs


def _cmd_audible_score(a) -> int:
    from . import audible

    try:
        asr = _load_asr(a.asr)
    except PluginUnavailable as exc:
        print(f"ASR no disponible: {exc}", file=sys.stderr)
        return 3
    planned = {}
    for rj in a.run_json or []:
        for r in json.loads(Path(rj).read_text(encoding="utf-8"))["segment_results"]:
            if r.get("planned_text_en"):
                planned[r["stream_id"]] = r["planned_text_en"]
    res = audible.score_exports(a.index, asr, refs=_refs_from(a) or None, planned_from_run=planned)
    audible.write_json(res, a.out)
    print(a.out)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="benchmark")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("selftest")
    s.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--manifest", default=None)
    r.add_argument("--audio-root", default=None)
    r.add_argument("--item-ids", default=None, help="subconjunto (ids separados por coma) del text-manifest; queda en el informe")
    r.add_argument("--text-manifest", default=None, help="JSONL de texto (sin audio) para adaptadores de modo text")
    r.add_argument("--adapter", required=True, help="modulo:Clase")
    r.add_argument("--adapter-config", default="{}")
    r.add_argument("--split", choices=["development", "test"], default="development")
    r.add_argument("--repetitions", type=int, default=3)
    r.add_argument("--warmup", type=int, default=2)
    r.add_argument("--deadline-s", type=float, default=5.0)
    r.add_argument("--continuous", action="store_true")
    r.add_argument("--no-verify-sha256", action="store_true", help="NO recomendado: omite la verificación de audio")
    r.add_argument("--call-timeout-s", type=float, default=10.0)
    r.add_argument("--watchdog-stall-s", type=float, default=30.0)
    r.add_argument("--audible-asr", default=None, help="faster-whisper:<modelo> (local, CPU) o modulo:funcion(pcm, sr)->texto; ASR independiente del candidato")
    r.add_argument("--out", required=True)
    r.add_argument("--incremental-collector", action="store_true",
                   help="observer causal opt-in: exporta timeline/metadatos sin PCM; sin alineación, semánticas unavailable")
    r.add_argument("--export-audio-dir", default=None, help="exporta WAV de SALIDA del candidato (solo benchmark)")
    r.add_argument("--system-name", default=None)
    lp = sub.add_parser("listening-prepare", help="estímulos y plantilla de escucha ciega desde índices de export")
    lp.add_argument("--index", nargs="+", required=True)
    lp.add_argument("--raters", nargs="+", required=True)
    lp.add_argument("--seed", type=int, default=0)
    lp.add_argument("--manifest", default=None, help="manifest para audio fuente (referencia) por ítem")
    lp.add_argument("--audio-root", default=None)
    lp.add_argument("--system-meta", action="append", help="NOMBRE:voice_cloning=true|false (similitud_voz solo si true)")
    lp.add_argument("--human-control", nargs=2, action="append", metavar=("WAV", "LICENCIA"), help="grabación humana EN (con licencia) como control")
    lp.add_argument("--degraded-controls", type=int, default=0, help="N copias degradadas de estímulos como control de rango")
    lp.add_argument("--voice-reference", action="append", metavar="ITEM_ID=WAV", help="muestra de VOZ (p. ej. CC0) para similitud_voz de ese ítem; no es el original: expresividad pasa a NA")
    lp.add_argument("--note", default=None, help="archivo de texto (markdown) que se agrega a INSTRUCCIONES.md (alcance del piloto)")
    lp.add_argument("--out", required=True)
    ls = sub.add_parser("listening-summary", help="resume puntuaciones humanas rellenadas")
    ls.add_argument("--ratings", required=True, nargs="+", help="CSV o JSON exportados por la UI (uno o varios evaluadores)")
    ls.add_argument("--key", required=True)
    lu = sub.add_parser("listening-ui", help="HTML local de escucha ciega desde un paquete de listening-prepare")
    lu.add_argument("--package", required=True, help="carpeta con stimuli/ y blind_key.json")
    lu.add_argument("--out", required=True, help="carpeta de salida APARTE (sin blind_key.json)")
    lu.add_argument("--seed", type=int, default=None, help="semilla de aleatorización por evaluador (por defecto la del paquete)")
    au = sub.add_parser("audible-score", help="ASR local sobre WAV exportados: texto audible vs planeado y vs referencias")
    au.add_argument("--index", nargs="+", required=True)
    au.add_argument("--asr", default="faster-whisper:small.en")
    au.add_argument("--run-json", nargs="*", help="run.json con planned_text_en por stream (índices antiguos)")
    au.add_argument("--manifest", default=None)
    au.add_argument("--audio-root", default=None)
    au.add_argument("--text-manifest", default=None)
    au.add_argument("--out", required=True)
    ps = sub.add_parser("proxy-score", help="UTMOSv2 sobre WAV exportados (proxy automático, no MOS humano)")
    ps.add_argument("--index", nargs="+", required=True, help="uno o más export_index.jsonl (incluye controls_index.jsonl)")
    ps.add_argument("--checkpoint", required=True)
    ps.add_argument("--fold", type=int, default=0)
    ps.add_argument("--seed", type=int, default=42)
    ps.add_argument("--num-repetitions", type=int, default=1)
    ps.add_argument("--predict-dataset", default="sarulab")
    ps.add_argument("--out", required=True)
    a = p.parse_args(argv)
    return {"selftest": _cmd_selftest, "run": _cmd_run, "listening-prepare": _cmd_listening_prepare,
            "listening-summary": _cmd_listening_summary, "listening-ui": _cmd_listening_ui, "proxy-score": _cmd_proxy_score, "audible-score": _cmd_audible_score}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
