#!/usr/bin/env python3
"""Gate de calidad: corre las métricas del estándar de calidad del proyecto y falla si un umbral se rompe.

Uso:  scripts/eval/gate.sh [--out DIR] [--only a,b] [--skip a,b] [--reps 3] [--allow-unmeasured]
Códigos de salida: 0 todo PASA (las OMITIDAS se listan) · 1 algún umbral FALLA o una métrica NO SE PUDO MEDIR · 2 uso incorrecto.
Cada corrida guarda sus salidas en --out (por defecto artifacts/gate/<fecha-hora>) y escribe gate.md con la tabla.

ESTADO: ejecutado el 16/11 (torch, tests, vad, latency, mt verificados; tts pendiente). Los extractores de run.json buscan
la clave de forma recursiva e imprimen la ruta usada en la columna Nota.
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
import threading
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


# T-45 (16/11, decisión del proyecto): la base 409 ms / 0,050 / 51,3 ms es de otra config (Moonshine tiny ep160, 12 clips test, sin Qwen).
# La tienda sale con Moonshine + Qwen (T-49). Hasta fijar la base con esa config (tras T-48 y T-50), estas filas se miden y se
# muestran, pero no aprueban ni fallan. T-58: se reactivan solas al existir scripts/eval/store_baseline.json (ver STORE_BASE).
NO_VALID_BASE = ("Latencia motor", "ASR WER", "MT petición→mt_final p95")

# T-58: la base de la config de tienda vive en scripts/eval/store_baseline.json (la escribe el equipo de medición tras medir con
# scripts/eval/t58_baseline.sh, retirado en T-70 con el modo incremental). Si existe, las filas con base dejan de salir SIN BASE y se comparan contra su umbral.
# Formato: {"config": {...adapter-config de la tienda...}, "python": ".venv-inc-llm/bin/python", "latency_reps": 7,
#           "latency_p50_ms": {"base": N, "threshold": N}, "latency_p95_ms": {...}, "mt_p95_ms": {...}, "asr_wer": {...}}
STORE_BASE_FILE = ROOT / "scripts/eval/store_baseline.json"
STORE_BASE: dict = json.loads(STORE_BASE_FILE.read_text(encoding="utf-8")) if STORE_BASE_FILE.exists() else {}
if STORE_BASE:  # T-58: la tienda fija el tope de caché MLX (decisión T-50); lo aplica scripts/eval/mlxcap/sitecustomize.py (solo si hay mlx)
    _cap = str(ROOT / "scripts/eval/mlxcap")
    ENV.update({"OT_MLX_CACHE_LIMIT_MB": str(STORE_BASE.get("mlx_cache_limit_mb", 256)), "EXTRA_PYTHONPATH": _cap,
                "PYTHONPATH": ENV["PYTHONPATH"] + os.pathsep + _cap})
BASE_KEY = {"Latencia motor, mediana": "latency_p50_ms", "Latencia motor, p95": "latency_p95_ms",
            "ASR WER": "asr_wer", "MT petición→mt_final p95": "mt_p95_ms"}


def store_thr(key: str):
    """(base, umbral) de la base de tienda o None si esa fila aún no tiene base."""
    e = STORE_BASE.get(key)
    return (float(e["base"]), float(e["threshold"])) if isinstance(e, dict) and "base" in e and "threshold" in e else None


@dataclass
class Row:
    metric: str
    value: str
    threshold: str
    status: str  # PASA | FALLA | CONTENDIDO | SIN BASE | NO MEDIDO | OMITIDO | DE QA
    note: str = ""


def sh(cmd, out: Path, log: str, env=None, timeout=3600) -> int:
    """Corre un comando desde ROOT y guarda stdout+stderr en out/log."""
    out.mkdir(parents=True, exist_ok=True)
    with open(out / log, "w") as f:
        f.write("$ " + " ".join(map(str, cmd)) + "\n")
        f.flush()
        return subprocess.run(cmd, cwd=ROOT, env=env or ENV, stdout=f, stderr=subprocess.STDOUT, timeout=timeout).returncode


COMPETITORS = re.compile(r"pytest|benchmark run|compare_configs|final_validation|xcodebuild|build_app|swift-build|swiftc|ab\.sh|OpenTraductor\.app|/tmp/\S+\.py|\bsay -v")
AGENT_PROC = re.compile(r"kiro-cli|claude --print|/bin/zsh -c|bash -c")  # la línea de comandos de un agente puede contener las palabras de arriba
TIMED_METRICS = ("Latencia motor", "ASR WER", "MT petición")  # (con base de tienda, ver STORE_BASE)  # BLEU/chrF/WER audible no dependen de la carga


def competing_processes() -> set[str]:
    """Otros procesos de benchmark/test/build que no descienden de este gate (contaminan las latencias)."""
    ps = subprocess.run(["ps", "-axo", "pid=,ppid=,command="], capture_output=True, text=True).stdout.splitlines()
    rows = []
    for ln in ps:
        parts = ln.split(None, 2)
        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
            rows.append((int(parts[0]), int(parts[1]), parts[2]))
    mine, grew = {os.getpid()}, True
    while grew:
        grew = False
        for pid, ppid, _ in rows:
            if ppid in mine and pid not in mine:
                mine.add(pid)
                grew = True
    return {re.sub(r"\s+", " ", cmd)[:90] for pid, _, cmd in rows
            if pid not in mine and COMPETITORS.search(cmd) and not AGENT_PROC.search(cmd)}


class ContentionSampler:
    """Muestrea cada 3 s mientras corre un chequeo: procesos competidores y el pico de loadavg."""
    def __init__(self):
        self.seen: set[str] = set()
        self.peak_load = 0.0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            self.seen |= competing_processes()
            self.peak_load = max(self.peak_load, os.getloadavg()[0])
            self._stop.wait(3)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()
        self.seen |= competing_processes()
        return False


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
            if qs and q in qs and isinstance(qs[q], dict) and isinstance(qs[q].get("value"), (int, float)):
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
    store = bool(STORE_BASE.get("config"))
    name = "store" if store else "A-baseline"
    cfg_json = json.dumps(STORE_BASE["config"], separators=(",", ":")) if store else "{}"
    reps = int(STORE_BASE.get("latency_reps", a.reps)) if store else a.reps
    env = {**ENV, "CONFIGS": f"{name}={cfg_json}", "REPS": str(max(reps, a.reps)) if store else str(a.reps)}
    if store and STORE_BASE.get("python"):
        env["PYBIN"] = STORE_BASE["python"]
    rc = sh([str(ROOT / "scripts/eval/compare_configs.sh"), str(d)], d, "run.log", env=env)
    rj = d / name / "run.json"
    rows = []
    tag = "tienda" if store else "sin red"
    b50, b95, bw = store_thr("latency_p50_ms"), store_thr("latency_p95_ms"), store_thr("asr_wer")
    def lab(b, old): return f"≤ {num(b[1], 0) if b else old} ms (base {num(b[0], 0) if b else '409'})"
    if rc != 0 or not rj.exists():
        rows.append(Row(f"Latencia motor, mediana ({tag})", "n/d", lab(b50, "450"), "NO MEDIDO", f"ver {d}/run.log"))
        rows.append(Row(f"Latencia motor, p95 ({tag})", "n/d", lab(b95, "-"), "NO MEDIDO" if b95 else "SIN BASE", "depende de la corrida de latencia"))
        rows.append(Row("ASR WER", "n/d", "≤ 0,06", "NO MEDIDO", "depende de la corrida de latencia"))
        return rows
    data = load(rj)
    for q_name, b, lbl in (("p50", b50, "mediana"), ("p95", b95, "p95")):
        q = find_quantile(data, q_name)
        thr = b[1] if b else (450.0 if q_name == "p50" else None)
        if q:
            st = "PASA" if thr is None or q[1] <= thr else "FALLA"
            rows.append(Row(f"Latencia motor, {lbl} ({tag})", num(q[1], 0) + " ms", lab(b, "450") if thr else "-", st, f"ruta: {q[0]}"))
        else:
            rows.append(Row(f"Latencia motor, {lbl} ({tag})", "n/d", lab(b, "450"), "NO MEDIDO", f"no hay censoring_aware.quantiles.{q_name} en run.json"))
    w = find_key(data, "wer") or find_key(data, "wer_corpus")
    wthr = bw[1] if bw else 0.06
    wl = f"≤ {num(wthr, 3)} (base {num(bw[0], 3) if bw else '0,050'})"
    if w:
        rows.append(Row("ASR WER", num(w[1], 3), wl, "PASA" if w[1] <= wthr else "FALLA", f"ruta: {w[0]} (verificar que es el ASR del candidato)"))
    else:
        rows.append(Row("ASR WER", "n/d", wl, "NO MEDIDO", "sin clave 'wer' en run.json; revisar el extractor"))
    return rows


def check_mt(out: Path, a) -> list[Row]:
    d = out / "mt"
    store = bool(STORE_BASE.get("config"))
    mt_cfg = {"stages": "mt"}
    if store:  # T-58: mismo traductor que la tienda (Qwen) si la base de tienda lo declara
        mt_cfg |= {k: v for k, v in STORE_BASE["config"].items() if k in ("mt_engine", "qwen_model_dir", "mt_normalize")}
    py = str(ROOT / STORE_BASE["python"]) if store and STORE_BASE.get("python") else PY
    cmd = [py, "-m", "benchmark", "run", "--text-manifest", TEXT_MANIFEST, "--adapter", "benchmark.adapters.text_mt_tts:TextMtTtsAdapter",
           "--adapter-config", json.dumps(mt_cfg), "--split", "development", "--repetitions", str(a.reps), "--warmup", "2", "--out", str(d)]
    rc = sh(cmd, d, "run.log")
    rj = d / "run.json"
    names = ("MT petición→mt_final p95", "MT BLEU (54 casos)", "MT chrF (54 casos)")
    if rc != 0 or not rj.exists():
        return [Row(n, "n/d", t, "NO MEDIDO", f"ver {d}/run.log") for n, t in zip(names, ("≤ 70 ms", "≥ 56", "≥ 71"))]
    data = load(rj)
    rows = []
    q = find_quantile(data, "p95")
    bm = store_thr("mt_p95_ms")
    mthr = bm[1] if bm else 70.0
    rows.append(Row(names[0], num(q[1]) + " ms", f"≤ {num(mthr)} ms (base {num(bm[0]) if bm else '51,3'})", "PASA" if q[1] <= mthr else "FALLA", f"ruta: {q[0]}") if q
                else Row(names[0], "n/d", "≤ 70 ms", "NO MEDIDO", "sin p95 en run.json"))
    qd = str(mt_cfg.get("qwen_model_dir", "")).lower()  # T-58: piso de calidad según el tamaño de Qwen de la tienda (decisión del 16/11)
    size = "1.7b" if "1.7b" in qd else "4b" if re.search(r"(?<![\d.])4b(?!it)", qd) else None
    bmin = (STORE_BASE.get("bleu_chrf_min") or {}).get(size, [56.0, 71.0]) if store else [56.0, 71.0]
    for name, key, thr in ((names[1], "bleu", float(bmin[0])), (names[2], "chrf", float(bmin[1]))):
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
    la = os.getloadavg()
    notes.append("loadavg 1/5/15 min: " + ", ".join(f"{x:.2f}" for x in la))
    if la[0] > 2:
        notes.append(f"AVISO: carga {la[0]:.1f} > 2 al empezar: las latencias no son comparables con la línea base (medida sin otras cargas); repetir con la máquina quieta antes de culpar a un commit.")
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
        smp = ContentionSampler()
        try:
            with smp:
                new = CHECKS[n](out, a)
            if smp.seen:
                for r in new:  # una FALLA de tiempo con otros procesos corriendo no es concluyente
                    if r.status == "FALLA" and r.metric.startswith(TIMED_METRICS):
                        r.status = "CONTENDIDO"
                        r.note += " · NO CONCLUYENTE, corrían a la vez: " + "; ".join(sorted(smp.seen))
            for r in new:
                key = next((v for k, v in BASE_KEY.items() if r.metric.startswith(k)), None)
                if r.metric.startswith(NO_VALID_BASE) and not (key and store_thr(key)) and r.status in ("PASA", "FALLA", "CONTENDIDO"):
                    r.note += f" · valor medido ({r.status} contra la base vieja); SIN BASE VÁLIDA hasta T-48/T-50"
                    r.status = "SIN BASE"
            rows += new
        except Exception as e:  # noqa: BLE001
            rows.append(Row(n, "n/d", "-", "NO MEDIDO", f"{type(e).__name__}: {e}"))
    for n in CHECKS:
        if n not in names:
            rows.append(Row(n, "-", "-", "OMITIDO", "excluido con --only/--skip"))
    rows += [Row("Latencia a BlackHole (mediana)", "-", "≤ 600 ms (base 548)", "OMITIDO", "necesita BlackHole; scripts/eval/device_loopback.py, a mano"),
             Row("Latencia percibida E2E (QA)", "-", "mediana ≤ 1,5 s, p90 ≤ 2,5 s", "DE QA", "con la .app"),
             Row("Estabilidad 60 min (QA)", "-", "sin cortes", "DE QA", "con la .app")]
    md = ["# Gate de calidad", "", *[f"- {x}" for x in environment_note()], f"- salidas: `{out}` · duración {time.time() - t0:.0f} s", "",
          "| Métrica | Valor | Umbral | Estado | Nota |", "|---|---|---|---|---|"]
    md += [f"| {r.metric} | {r.value} | {r.threshold} | **{r.status}** | {r.note} |" for r in rows]
    fails = [r for r in rows if r.status == "FALLA"]
    unm = [r for r in rows if r.status in ("NO MEDIDO", "CONTENDIDO")]
    verdict = "FALLA" if fails or (unm and not a.allow_unmeasured) else "PASA"
    nb = [r for r in rows if r.status == "SIN BASE"]
    md += ["", f"**GATE: {verdict}** · fallan {len(fails)} · sin medir o contendidos {len(unm)} · sin base válida {len(nb)} (no cuentan)"]
    (out / "gate.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0 if verdict == "PASA" else 1


if __name__ == "__main__":
    sys.exit(main())
