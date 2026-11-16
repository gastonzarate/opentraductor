#!/usr/bin/env python3
"""Gate de calidad: corre las métricas de memory/decisiones/estandar-de-calidad.md y falla si un umbral se rompe.

Uso:  scripts/eval/gate.sh [--out DIR] [--only a,b] [--skip a,b] [--reps 3] [--allow-unmeasured]
Códigos de salida: 0 todo PASA (las OMITIDAS se listan) · 1 algún umbral FALLA o una métrica NO SE PUDO MEDIR · 2 uso incorrecto.
Cada corrida guarda sus salidas en --out (por defecto artifacts/gate/<fecha-hora>) y escribe gate.md con la tabla.

ESTADO: escrito en T-45 SIN poder ejecutarlo (el run no tenía terminal). Los extractores de run.json buscan
la clave de forma recursiva e imprimen la ruta usada; verificar la primera corrida contra PILOT_RESULTS.md.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import socket
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = str(ROOT / ".venv/bin/python")  # runtime + benchmark (como compare_configs.sh)
EVAL_RUN = str(ROOT / "scripts/eval/run.sh")  # entorno models/eval/venv: faster-whisper, UTMOSv2
TEXT_MANIFEST = "data/manifests/text_meetings_ar_dev.jsonl"
TTS_ITEMS = "txt-meet-ar-dev-01,txt-meet-ar-dev-02,txt-meet-ar-dev-03,txt-meet-ar-dev-04,txt-meet-ar-dev-06,txt-meet-ar-dev-07"
UTMOS_CKPT = "models/eval/utmosv2/fold0_s42_best_model.pth"
KNOWN_FAILING_FILES = ("tests/runtime/test_speculative.py",)  # BUG-1
BASE_PASSED = 299

ENV = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
       "PYTHONPATH": str(ROOT) + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")}


@dataclass
class Row:
    metric: str
    value: str
    threshold: str
    status: str  # PASA | FALLA | NO MEDIDO | OMITIDO | DE @qa
    note: str = ""


def sh(cmd, out: Path, log: str, env=None, timeout=3600) -> int:
    """Corre un comando desde ROOT y guarda stdout+stderr en out/log."""
    out.mkdir(parents=True, exist_ok=True)
    with open(out / log, "w") as f:
        f.write("$ " + " ".join(map(str, cmd)) + "\n")
        f.flush()
        return subprocess.run(cmd, cwd=ROOT, env=env or ENV, stdout=f, stderr=subprocess.STDOUT, timeout=timeout).returncode


def find_key(obj, key, path=""):
    """Primera aparición (en profundidad) de `key` con valor numérico: (ruta, valor) o None."""
    if isinstance(obj, dict):
        v = obj.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return f"{path}.{key}", float(v)
        for k, x in obj.items():
            r = find_key(x, key, f"{path}.{k}")
            if r:
                return r
    elif isinstance(obj, list):
        for i, x in enumerate(obj):
            r = find_key(x, key, f"{path}[{i}]")
            if r:
                return r
    return None


def find_quantile(run_json: dict, q: str):
    """Valor de censoring_aware.quantiles.<q>.value del primer grupo (la métrica titular)."""
    def walk(o, path=""):
        if isinstance(o, dict):
            qs = o.get("censoring_aware", {}).get("quantiles") if isinstance(o.get("censoring_aware"), dict) else None
            if qs and q in qs and isinstance(qs[q], dict) and "value" in qs[q]:
                return f"{path}.censoring_aware.quantiles.{q}.value", float(qs[q]["value"])
            for k, x in o.items():
                r = walk(x, f"{path}.{k}")
                if r:
                    return r
        elif isinstance(o, list):
            for i, x in enumerate(o):
                r = walk(x, f"{path}[{i}]")
                if r:
                    return r
        return None
    return walk(run_json)


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def num(v, nd=1):
    return f"{v:.{nd}f}".replace(".", ",") if v is not None else "n/d"


# ----------------------------------------------------------------------------- checks
def check_torch(out: Path, a) -> list[Row]:
    code = ("import sys; import runtime.components, runtime.preflight; "
            "bad=sorted(m for m in sys.modules if m.split('.')[0] in ('torch','torchaudio','silero_vad')); "
            "print('LOADED', bad)")
    log = out / "torch"
    rc = sh([PY, "-c", code], log, "import.log")
    txt = (log / "import.log").read_text()
    onnx = ROOT / "runtime/data/silero_vad.onnx"
    if rc != 0:
        return [Row("Runtime sin torch", "no importa", "no carga torch/torchaudio/silero_vad", "NO MEDIDO", f"ver {log}/import.log")]
    ok = "LOADED []" in txt and onnx.exists()
    return [Row("Runtime sin torch", "limpio" if ok else ("torch cargado: " + txt.split("LOADED")[-1].strip() if "LOADED []" not in txt else "falta runtime/data/silero_vad.onnx"),
                "ningún módulo torch/torchaudio/silero_vad + silero_vad.onnx presente", "PASA" if ok else "FALLA")]


def check_tests(out: Path, a) -> list[Row]:
    log = out / "tests"
    sh(["uv", "run", "pytest", "tests/runtime", "-q", "-p", "no:cacheprovider"], log, "pytest.log")
    txt = (log / "pytest.log").read_text()
    failed = re.search(r"(\d+) failed", txt)
    passed = re.search(r"(\d+) passed", txt)
    if not passed and not failed:
        return [Row("Tests runtime", "sin resumen", f"≥ {BASE_PASSED} pasan, solo fallos conocidos", "NO MEDIDO", f"ver {log}/pytest.log")]
    nf, npass = int(failed.group(1)) if failed else 0, int(passed.group(1)) if passed else 0
    new = [l for l in txt.splitlines() if l.startswith(("FAILED", "ERROR")) and not any(k in l for k in KNOWN_FAILING_FILES)]
    ok = not new and npass >= BASE_PASSED and nf <= 2
    note = f"fallos nuevos: {new[:3]}" if new else ("BUG-1 conocido" if nf else "")
    return [Row("Tests runtime", f"{npass} pasan, {nf} fallan", f"≥ {BASE_PASSED} pasan; solo BUG-1 (test_speculative)", "PASA" if ok else "FALLA", note)]


def check_vad(out: Path, a) -> list[Row]:
    code = ("import time,statistics as s; from runtime.components import SileroVAD; v=SileroVAD(); b=bytes(1024)\n"
            "for _ in range(100): v.probs(b)\n"
            "t=[]\n"
            "for _ in range(2000):\n"
            "    x=time.perf_counter(); v.probs(b); t.append((time.perf_counter()-x)*1000)\n"
            "print('VAD_MS', s.median(t))")
    log = out / "vad"
    sh([PY, "-c", code], log, "vad.log")
    m = re.search(r"VAD_MS ([0-9.]+)", (log / "vad.log").read_text())
    if not m:
        return [Row("VAD por ventana (mediana)", "n/d", "≤ 0,3 ms", "NO MEDIDO", f"ver {log}/vad.log")]
    v = float(m.group(1))
    return [Row("VAD por ventana (mediana)", num(v, 3) + " ms", "≤ 0,3 ms (base 0,23)", "PASA" if v <= 0.3 else "FALLA")]


def check_latency(out: Path, a) -> list[Row]:
    d = out / "latency"
    env = {**ENV, "CONFIGS": "A-baseline={}", "REPS": str(a.reps)}
    rc = sh([str(ROOT / "scripts/eval/compare_configs.sh"), str(d)], d, "run.log", env=env)
    rj = d / "A-baseline/run.json"
    rows = []
    if rc != 0 or not rj.exists():
        rows.append(Row("Latencia motor, mediana (sin red)", "n/d", "≤ 450 ms (base 409)", "NO MEDIDO", f"ver {d}/run.log"))
        rows.append(Row("ASR WER", "n/d", "≤ 0,06", "NO MEDIDO", "depende de la corrida de latencia"))
        return rows
    data = load(rj)
    q = find_quantile(data, "p50")
    if q:
        rows.append(Row("Latencia motor, mediana (sin red)", num(q[1], 0) + " ms", "≤ 450 ms (base 409)", "PASA" if q[1] <= 450 else "FALLA", f"ruta: {q[0]}"))
    else:
        rows.append(Row("Latencia motor, mediana (sin red)", "n/d", "≤ 450 ms", "NO MEDIDO", "no hay censoring_aware.quantiles.p50 en run.json"))
    w = find_key(data, "wer") or find_key(data, "wer_corpus")
    if w:
        rows.append(Row("ASR WER", num(w[1], 3), "≤ 0,06 (base 0,050)", "PASA" if w[1] <= 0.06 else "FALLA", f"ruta: {w[0]} (verificar que es el ASR del candidato)"))
    else:
        rows.append(Row("ASR WER", "n/d", "≤ 0,06", "NO MEDIDO", "sin clave 'wer' en run.json; revisar el extractor"))
    return rows


def check_mt(out: Path, a) -> list[Row]:
    d = out / "mt"
    cmd = [PY, "-m", "benchmark", "run", "--text-manifest", TEXT_MANIFEST, "--adapter", "benchmark.adapters.text_mt_tts:TextMtTtsAdapter",
           "--adapter-config", '{"stages": "mt"}', "--split", "development", "--repetitions", str(a.reps), "--warmup", "2", "--out", str(d)]
    rc = sh(cmd, d, "run.log")
    rj = d / "run.json"
    names = ("MT petición→mt_final p95", "MT BLEU (54 casos)", "MT chrF (54 casos)")
    if rc != 0 or not rj.exists():
        return [Row(n, "n/d", t, "NO MEDIDO", f"ver {d}/run.log") for n, t in zip(names, ("≤ 70 ms", "≥ 56", "≥ 71"))]
    data = load(rj)
    rows = []
    q = find_quantile(data, "p95")
    rows.append(Row(names[0], num(q[1]) + " ms", "≤ 70 ms (base 51,3)", "PASA" if q[1] <= 70 else "FALLA", f"ruta: {q[0]}") if q
                else Row(names[0], "n/d", "≤ 70 ms", "NO MEDIDO", "sin p95 en run.json"))
    for name, key, thr in ((names[1], "bleu", 56.0), (names[2], "chrf", 71.0)):
        r = find_key(data, key)
        rows.append(Row(name, num(r[1]), f"≥ {thr:g}".replace(".", ","), "PASA" if r[1] >= thr else "FALLA", f"ruta: {r[0]}; las referencias son 'generated'") if r
                    else Row(name, "n/d", f"≥ {thr:g}", "NO MEDIDO", f"sin clave '{key}' en run.json; ver {d}/report.md"))
    return rows


def check_tts(out: Path, a) -> list[Row]:
    d = out / "mt-tts"
    cmd = [PY, "-m", "benchmark", "run", "--text-manifest", TEXT_MANIFEST, "--item-ids", TTS_ITEMS,
           "--adapter", "benchmark.adapters.text_mt_tts:TextMtTtsAdapter", "--adapter-config", '{"stages": "mt+tts"}',
           "--split", "development", "--repetitions", "1", "--warmup", "2", "--out", str(d), "--export-audio-dir", str(d / "audio"),
           "--system-name", "kokoro-af_heart"]
    rc = sh(cmd, d, "run.log")
    idx = d / "audio/export_index.jsonl"
    rows = []
    if rc != 0 or not idx.exists():
        return [Row("Texto audible WER (TTS→Whisper)", "n/d", "≤ 0,05", "NO MEDIDO", f"ver {d}/run.log"),
                Row("UTMOSv2 voz por defecto (proxy)", "n/d", "≥ 3,8", "NO MEDIDO", "depende de la corrida TTS")]
    sh([EVAL_RUN, "audible-score", "--index", str(idx), "--asr", "faster-whisper:small.en", "--text-manifest", TEXT_MANIFEST,
        "--out", str(d / "audible_score.json")], d, "audible.log")
    try:
        ps = load(d / "audible_score.json")["per_system"]
        s = next(iter(ps.values()))
        w = s["planned_vs_audible_wer_corpus"]
        rows.append(Row("Texto audible WER (TTS→Whisper)", num(w, 3), "≤ 0,05 (base 0,036)", "PASA" if w <= 0.05 else "FALLA", f"n={s['n']}"))
    except Exception as e:  # noqa: BLE001
        rows.append(Row("Texto audible WER (TTS→Whisper)", "n/d", "≤ 0,05", "NO MEDIDO", f"{type(e).__name__}; ver {d}/audible.log"))
    sh([EVAL_RUN, "proxy-score", "--index", str(idx), "--checkpoint", UTMOS_CKPT, "--out", str(d / "proxy_utmosv2.json")], d, "proxy.log")
    try:
        ps = load(d / "proxy_utmosv2.json")["per_system"]
        m = next(iter(ps.values()))["mean_predicted"]
        rows.append(Row("UTMOSv2 voz por defecto (proxy)", num(m, 2), "≥ 3,8 (base 4,00)", "PASA" if m >= 3.8 else "FALLA", "predictor automático, no opinión humana"))
    except Exception as e:  # noqa: BLE001
        rows.append(Row("UTMOSv2 voz por defecto (proxy)", "n/d", "≥ 3,8", "NO MEDIDO", f"{type(e).__name__}; ver {d}/proxy.log"))
    return rows


CHECKS = {"torch": check_torch, "tests": check_tests, "vad": check_vad, "latency": check_latency, "mt": check_mt, "tts": check_tts}


def environment_note() -> list[str]:
    notes = []
    try:
        socket.create_connection(("127.0.0.1", 8877), timeout=0.3).close()
        notes.append("La app de la persona responde en :8877 (no se mató): las latencias pueden tener contención.")
    except OSError:
        notes.append("Puerto :8877 libre.")
    notes.append("loadavg 1/5/15 min: " + ", ".join(f"{x:.2f}" for x in os.getloadavg()))
    try:
        notes.append("commit: " + subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
                     + (" (con cambios sin commitear)" if subprocess.run(["git", "status", "--porcelain", "--", "runtime", "packaging", "pyproject.toml", "uv.lock"],
                                                                        cwd=ROOT, capture_output=True, text=True).stdout.strip() else ""))
    except Exception:  # noqa: BLE001
        pass
    return notes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", default="")
    ap.add_argument("--skip", default="")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--allow-unmeasured", action="store_true", help="NO MEDIDO no hace fallar el gate (solo para depurar)")
    a = ap.parse_args()
    names = [n for n in CHECKS if (not a.only or n in a.only.split(",")) and n not in a.skip.split(",")]
    bad = [n for n in (a.only + "," + a.skip).split(",") if n and n not in CHECKS]
    if bad:
        print(f"chequeos desconocidos: {bad}; válidos: {list(CHECKS)}", file=sys.stderr)
        return 2
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = Path(a.out) if a.out else ROOT / "artifacts/gate" / stamp
    out = out if out.is_absolute() else ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows: list[Row] = []
    for n in names:
        print(f"[gate] {n} ...", flush=True)
        try:
            rows += CHECKS[n](out, a)
        except Exception as e:  # noqa: BLE001
            rows.append(Row(n, "n/d", "-", "NO MEDIDO", f"{type(e).__name__}: {e}"))
    for n in CHECKS:
        if n not in names:
            rows.append(Row(n, "-", "-", "OMITIDO", "excluido con --only/--skip"))
    rows += [Row("Latencia a BlackHole (mediana)", "-", "≤ 600 ms (base 548)", "OMITIDO", "necesita BlackHole; scripts/eval/device_loopback.py, a mano"),
             Row("Latencia percibida E2E (QA)", "-", "mediana ≤ 1,5 s, p90 ≤ 2,5 s", "DE @qa", "con la .app"),
             Row("Estabilidad 60 min (QA)", "-", "sin cortes", "DE @qa", "con la .app")]
    md = ["# Gate de calidad", "", *[f"- {x}" for x in environment_note()], f"- salidas: `{out}` · duración {time.time() - t0:.0f} s", "",
          "| Métrica | Valor | Umbral | Estado | Nota |", "|---|---|---|---|---|"]
    md += [f"| {r.metric} | {r.value} | {r.threshold} | **{r.status}** | {r.note} |" for r in rows]
    fails = [r for r in rows if r.status == "FALLA"]
    unm = [r for r in rows if r.status == "NO MEDIDO"]
    verdict = "FALLA" if fails or (unm and not a.allow_unmeasured) else "PASA"
    md += ["", f"**GATE: {verdict}** · fallan {len(fails)} · sin medir {len(unm)}"]
    (out / "gate.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0 if verdict == "PASA" else 1


if __name__ == "__main__":
    sys.exit(main())
