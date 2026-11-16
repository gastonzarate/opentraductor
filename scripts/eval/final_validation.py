"""Validación final sobre el split reservado (test): congelar → comandos → lista de evidencia.

Exposición parcial del held-out reconocida y registrada en FREEZE.json (no se afirma inédito).

NO ejecuta modelos, NO abre audio ni transcripciones del held-out: solo cuenta filas/hablantes del manifest y calcula
hashes de archivos. Usa el arnés existente (``benchmark run``); no es un framework nuevo. Ver docs/final-validation-protocol.md.

    # 1) congelar (una vez por candidato/config; inmutable)
    .venv/bin/python scripts/eval/final_validation.py freeze --config docs/final-validation.config.example.json --out artifacts/final-validation/<id>
    # 2) imprimir los comandos exactos (no ejecuta)
    .venv/bin/python scripts/eval/final_validation.py command --freeze artifacts/final-validation/<id>
    # 3) tras UNA ejecución autorizada: lista de evidencia (pass/fail) contra lo congelado
    .venv/bin/python scripts/eval/final_validation.py check --freeze artifacts/final-validation/<id> --run artifacts/final-validation/<id>/streaming-1

Reglas: el congelado es inmutable (no se sobrescribe); cada intento (también el fallido) queda en ATTEMPTS.jsonl; ningún ajuste tras ver
resultados del held-out: un cambio de código/config exige un congelado NUEVO y el anterior queda documentado como intento.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Exposición PARCIAL reconocida del held-out (no se afirma «inédito»). Texto exacto, tomado del informe del ingeniero del benchmark.
HELDOUT_EXPOSURE = {
    "status": "exposicion_parcial_reconocida",
    "what_was_seen": ("una vez, antes de este protocolo, se listaron en pantalla las filas del manifest de OpenSLR 61 incluidas las del split test: "
                      "item_id, duración, hablante y los primeros ~60 caracteres de la transcripción española"),
    "audio_listened": False,
    "model_or_pipeline_results_on_test_used": False,
    "used_for_tuning": False,
    "claim": "NO se afirma que el held-out sea inédito; la evaluación es reservada con esta limitación documentada y sin ajuste posterior",
}
REQUIRED_CONFIG_KEYS = ("adapter", "adapter_config", "manifest", "audio_root", "repetitions", "warmup", "deadline_s")
MIN_CLUSTERS = {"p95": 20, "p99": 100}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _code_hashes() -> dict:
    """sha256 por archivo de runtime/, benchmark/ y scripts/eval/ propios (mismo alcance que registra el arnés + este script)."""
    out = {}
    for d in ("runtime", "benchmark", "scripts/eval"):  # scripts/eval: validación y dispositivo que se ejecutan
        if (ROOT / d).exists():
            for p in sorted((ROOT / d).rglob("*.py")):
                out[str(p.relative_to(ROOT))] = _sha(p)
    return out


def _lock_hashes() -> dict:
    return {n: _sha(ROOT / n) for n in ("uv.lock", "pyproject.toml", ".python-version") if (ROOT / n).exists()}


def _test_counts(manifest: Path) -> dict:
    """Solo conteos: ítems y hablantes del split test. No imprime ni guarda transcripciones ni rutas de audio."""
    items, speakers = 0, {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("split") == "test":
            items += 1
            k = d.get("speaker_id") or f"source:{d.get('source_id')}"
            speakers[k] = speakers.get(k, 0) + 1
    return {"n_items": items, "n_speakers_clusters": len(speakers), "items_per_cluster": sorted(speakers.values())}


def freeze(a) -> int:
    cfg_path = Path(a.config)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if cfg.get("_example"):
        raise SystemExit("la config es un EJEMPLO (_example=true): no se congela")
    missing = [k for k in REQUIRED_CONFIG_KEYS if k not in cfg]
    if missing:
        raise SystemExit(f"config incompleta: faltan {missing}")
    out = Path(a.out)
    if (out / "FREEZE.json").exists():
        raise SystemExit(f"{out}/FREEZE.json ya existe: el congelado es inmutable; usar otra carpeta (nuevo intento) y documentar el anterior")
    manifest = ROOT / cfg["manifest"]
    counts = _test_counts(manifest)
    if counts["n_items"] == 0:
        raise SystemExit("el manifest no tiene ítems test")
    code = _code_hashes()
    frz = {
        "schema": "opentraductor-final-validation-freeze/v1",
        "created_unix": time.time(),
        "config": cfg,
        "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
        "manifest_file_sha256": _sha(manifest),
        "test_counts": counts,
        "code_sha256_per_file": code,
        "lock_files_sha256": _lock_hashes(),
        "freeze_id": None,
        "declared": {
            "unit_of_analysis": "hablante (cluster); las repeticiones técnicas NO son muestras independientes",
            "heldout_exposure": HELDOUT_EXPOSURE,
            "tuning_after_heldout": "prohibido: un cambio exige un congelado nuevo",
        },
    }
    frz["freeze_id"] = hashlib.sha256(json.dumps({k: frz[k] for k in ("config_sha256", "manifest_file_sha256", "code_sha256_per_file", "lock_files_sha256")},
                                                  sort_keys=True).encode()).hexdigest()[:12]
    out.mkdir(parents=True, exist_ok=True)
    (out / "FREEZE.json").write_text(json.dumps(frz, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "ATTEMPTS.jsonl").write_text("", encoding="utf-8")
    print(json.dumps({"freeze_id": frz["freeze_id"], "test_counts": counts, "n_code_files": len(code)}, indent=1))
    return 0


def _load(freeze_dir: Path) -> dict:
    return json.loads((freeze_dir / "FREEZE.json").read_text(encoding="utf-8"))


def _verify_unchanged(frz: dict) -> list[str]:
    problems = []
    now = _code_hashes()
    for f, h in frz["code_sha256_per_file"].items():
        if now.get(f) != h:
            problems.append(f"código cambió desde el congelado: {f}")
    for f in set(now) - set(frz["code_sha256_per_file"]):
        problems.append(f"archivo de código nuevo desde el congelado: {f}")
    if _sha(ROOT / frz["config"]["manifest"]) != frz["manifest_file_sha256"]:
        problems.append("el manifest cambió")
    locks_now, locks_frz = _lock_hashes(), frz["lock_files_sha256"]
    for n in sorted(set(locks_now) | set(locks_frz)):  # ambos sentidos: también detecta un lock ELIMINADO o añadido
        if locks_now.get(n) != locks_frz.get(n):
            problems.append(f"lock/config {'eliminado' if n not in locks_now else 'añadido' if n not in locks_frz else 'cambió'}: {n}")
    return problems


def command(a) -> int:
    fdir = Path(a.freeze)
    frz = _load(fdir)
    problems = _verify_unchanged(frz)
    if problems:
        print("NO SE EMITEN COMANDOS: lo congelado ya no coincide:\n- " + "\n- ".join(problems), file=sys.stderr)
        return 3
    cfg = frz["config"]
    n = sum(1 for _ in (fdir / "ATTEMPTS.jsonl").read_text().splitlines()) + 1
    run_dir = fdir / f"streaming-{n}"
    run = ["cd", shlex.quote(str(ROOT)), "&&", ".venv/bin/python", "-m", "benchmark", "run",
           "--manifest", shlex.quote(cfg["manifest"]), "--audio-root", shlex.quote(cfg["audio_root"]),
           "--adapter", shlex.quote(cfg["adapter"]), "--adapter-config", shlex.quote(json.dumps(cfg["adapter_config"])),
           "--split", "test", "--repetitions", str(cfg["repetitions"]), "--warmup", str(cfg["warmup"]),
           "--deadline-s", str(cfg["deadline_s"]), "--call-timeout-s", str(cfg.get("call_timeout_s", 60)),
           "--watchdog-stall-s", str(cfg.get("watchdog_stall_s", 120)),
           "--out", shlex.quote(str(run_dir)), "--export-audio-dir", shlex.quote(str(run_dir / "audio")),
           "--system-name", shlex.quote(f"final-{frz['freeze_id']}-a{n}")]
    audible = (f"cd {shlex.quote(str(ROOT))} && HF_HOME=$PWD/models/eval/hf HF_HUB_OFFLINE=1 PYTHONPATH=. models/eval/venv/bin/python -m benchmark audible-score "
               f"--index {shlex.quote(str(run_dir / 'audio' / 'export_index.jsonl'))} --asr faster-whisper:small.en "
               f"--run-json {shlex.quote(str(run_dir / 'run.json'))} --out {shlex.quote(str(run_dir / 'audible.json'))}")
    print("# (1) STREAMING aislado por clip, split test (UNA ejecución por congelado; requiere lease exclusivo; registrar el intento)")
    print(" ".join(run))
    print("\n# (2) texto audible con ASR independiente (CPU, solo salida inglesa; después de (1); sin inferencia del candidato)")
    print(audible)
    print("\n# (3) SALIDA A DISPOSITIVO (BlackHole), SEPARADA: scripts/eval/device_loopback.py. REQUISITO PENDIENTE: hoy fija --split development "
          "y toma ítems de los manifests; hay que parametrizar el split ANTES del congelado (cambia benchmark/ => afecta el hash).")
    off_dir = fdir / f"offline-{n}"
    canary = ("import json,socket,sys; out={'sandbox_deny_network':True}\n"
              "try:\n socket.create_connection(('1.1.1.1',53),2); out['connected']=True\n"
              "except OSError as e: out.update(connected=False, errno=e.errno, error=str(e))\n"
              "json.dump(out, open(sys.argv[1],'w')); sys.exit(0 if (not out.get('connected') and out.get('errno')==1) else 9)")
    prof = "(version 1) (allow default) (deny network*)"
    print("\n# (4) «OFFLINE» = SIN RED: el MISMO adaptador streaming y la misma config congelada bajo sandbox macOS con red denegada. NO es calidad sin latencia")
    print("#     ni un adaptador batch. Va como intento APARTE (otro registro en ATTEMPTS). Primero el canario: debe dar EPERM (errno 1) o no se corre nada.")
    print(f"mkdir -p {shlex.quote(str(off_dir))} && sandbox-exec -p {shlex.quote(prof)} {shlex.quote(sys.executable)} -c {shlex.quote(canary)} {shlex.quote(str(off_dir / 'offline_canary.json'))}")
    off = list(run)
    off[off.index("--out") + 1] = shlex.quote(str(off_dir))
    off[off.index("--export-audio-dir") + 1] = shlex.quote(str(off_dir / "audio"))
    off[off.index("--system-name") + 1] = shlex.quote(f"final-{frz['freeze_id']}-offline{n}")
    i = off.index(".venv/bin/python")
    off = off[:i] + ["sandbox-exec", "-p", shlex.quote(prof)] + off[i:]
    print(" ".join(off))
    print("\n# (5) VALIDACIÓN CONTINUA: reservada y separada (otro congelado). Sin afirmaciones de conversación natural por concatenación de clips.")
    print(f"\n# Tras ejecutar (1): registrar el intento (ok o fallido) con: final_validation.py attempt --freeze {fdir} --run-dir {run_dir} --status <ok|failed> --note '...'")
    print(f"# Luego: final_validation.py check --freeze {fdir} --run {run_dir}")
    return 0


def attempt(a) -> int:
    fdir = Path(a.freeze)
    rec = {"unix": time.time(), "run_dir": a.run_dir, "status": a.status, "note": a.note}
    with (fdir / "ATTEMPTS.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("intento registrado")
    return 0


def check(a) -> int:
    fdir, rdir = Path(a.freeze), Path(a.run)
    frz = _load(fdir)
    cfg = frz["config"]
    run = json.loads((rdir / "run.json").read_text(encoding="utf-8"))
    m = run["meta"]
    rows: list[tuple[str, bool, str]] = []

    def chk(name: str, ok: bool, detail: str = "") -> None:
        rows.append((name, bool(ok), detail))

    chk("código sin cambios desde el congelado", not _verify_unchanged(frz), "; ".join(_verify_unchanged(frz)))
    chk("purpose=final_test y split=test", m["run_config"]["purpose"] == "final_test" and m["split"] == "test")
    chk("adapter_config igual al congelado", m["adapter_config"] == cfg["adapter_config"])
    chk("repeticiones/warmup/deadline como lo congelado",
        (m["run_config"]["repetitions"], m["run_config"]["warmup_streams"], m["run_config"]["deadline_s"]) ==
        (cfg["repetitions"], cfg["warmup"], cfg["deadline_s"]))
    chk("n de ítems = ítems test congelados", m["n_items"] == frz["test_counts"]["n_items"], f"{m['n_items']} vs {frz['test_counts']['n_items']}")
    chk("no abortada ni error del colector", not m.get("aborted") and not m.get("collector_error"), str(m.get("aborted") or m.get("collector_error")))
    chk("aislado por clip (no continuo)", not m["run_config"].get("continuous"))
    traces = [json.loads(l) for l in (rdir / "traces.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    nw = sum(1 for t in traces if t.get("warmup"))
    expected = frz["test_counts"]["n_items"] * cfg["repetitions"]
    chk("streams medidos = ítems × repeticiones (warmup aparte, guardado)", len(traces) - nw == expected, f"medidos {len(traces)-nw}, esperados {expected}, warmup {nw}")
    g = run["aggregate"]["groups"][0]
    counts = g["outcome_counts"]
    chk("fallos/timeouts/anticipados contados (suma = N)", sum(counts.values()) == g["n_segments"], str(counts))
    pm = g[g["primary_metric"]]
    q95 = pm["censoring_aware"]["quantiles"].get("p95", {})
    nclu = q95.get("n_clusters")
    chk("hablantes (clusters) reportados", nclu == frz["test_counts"]["n_speakers_clusters"], f"{nclu}")
    chk("p95/p99 marcados de muestra baja (clusters < mínimo)", all(pm["censoring_aware"]["quantiles"][k]["low_sample"] for k in ("p95", "p99")
                                                                   if k in pm["censoring_aware"]["quantiles"]) or (nclu or 0) >= MIN_CLUSTERS["p99"])
    chk("límites imprecisos declarados", g["boundary_precision"] in ("imprecise_or_unknown",), g["boundary_precision"])
    chk("primer PCM / reproducción simulada / dispositivo separados", g["primary_metric"] in ("end_to_first_ms",) and "end_to_playout_est_ms" in g and "device_output" in g)
    chk("calidad: referencias humanas y generadas por separado", "quality" in run and "planned_text" in run["quality"])
    from benchmark import export as X

    idx_path = rdir / "audio" / "export_index.jsonl"
    exp_ok, exp_detail, wavs = False, "índice ausente", set()
    if idx_path.exists():
        rows_ix = [r for r in X.read_index(idx_path) if r.get("exported")]
        problems = X.verify_index(rows_ix)
        wavs = {r["wav"] for r in rows_ix}
        exp_ok, exp_detail = bool(rows_ix) and not problems, f"{len(rows_ix)} WAV exportados; problemas: {problems[:3]}"
    chk("export de audio: sha256 del índice verificado contra los archivos", exp_ok, exp_detail)
    aud_ok, aud_detail = False, "audible.json ausente"
    if (rdir / "audible.json").exists():
        aud = json.loads((rdir / "audible.json").read_text(encoding="utf-8"))
        ids = {t.get("stream_id") for t in traces}
        aw = {i["wav"] for i in aud.get("items", [])}
        aud_ok = bool(aud.get("asr_label")) and bool(aw) and aw <= wavs and all(i["stream_id"] in ids for i in aud["items"])
        aud_detail = f"ASR {aud.get('asr_label')}; {len(aw)} WAV, subconjunto del índice de ESTA corrida: {aw <= wavs}"
    chk("audible.json vinculado a esta corrida (mismos WAV y stream_id) con ASR independiente", aud_ok, aud_detail)
    sysname = (m.get("run_config") or {}).get("system_name") or ""
    chk("system_name de la corrida referencia el freeze_id", frz["freeze_id"] in sysname, sysname)
    att = [json.loads(l) for l in (fdir / "ATTEMPTS.jsonl").read_text().splitlines() if l.strip()]
    mine = [x for x in att if Path(x["run_dir"]).resolve() == rdir.resolve()]
    chk("esta corrida figura en ATTEMPTS.jsonl (ok)", bool(mine) and mine[-1]["status"] == "ok", f"{len(mine)} registro(s) para esta carpeta")
    if rdir.name.startswith("offline-"):
        cj = rdir / "offline_canary.json"
        c = json.loads(cj.read_text()) if cj.exists() else {}
        chk("offline (sin red): canario EPERM bajo sandbox deny network", c.get("connected") is False and c.get("errno") == 1, str(c))
    allok = all(r[1] for r in rows)
    lines = ["# Lista de evidencia — validación final", "", f"- freeze_id: `{frz['freeze_id']}` · run: `{rdir}`", "",
             "| comprobación | resultado | detalle |", "|---|---|---|"]
    lines += [f"| {n} | {'OK' if ok else '**FALLA**'} | {d} |" for n, ok, d in rows]
    lines += ["", f"**Estado de la evidencia: {'COMPLETA' if allok else 'INCOMPLETA'}**", "",
              "## Lo que este resultado NO certifica (siempre)",
              f"- NO certifica 300 ms: límites {g['boundary_precision']} (anotados `estimated`, tolerancia supuesta ±0.3 s), {nclu} hablantes (regla práctica del arnés: ≥ {MIN_CLUSTERS['p95']}/{MIN_CLUSTERS['p99']} clusters para p95/p99), "
              "repeticiones técnicas no independientes, un solo equipo.",
              "- primer PCM «disponible», reproducción SIMULADA y salida a dispositivo son métricas distintas; la de dispositivo requiere su propia corrida.",
              "- el held-out tuvo una exposición PARCIAL reconocida (ver FREEZE.json → declared.heldout_exposure): no se afirma que sea inédito.",
              "- p95/p99 con tan pocos hablantes no tienen precisión fiable; el mínimo de 20/100 clusters es una regla práctica del arnés, no una garantía estadística universal.",
              "- la calidad con referencias generadas no es calidad humana; sin MOS; sin escucha humana no hay afirmación de naturalidad/similitud.",
              "- «offline» aquí = sin red (mismo adaptador, sandbox con red denegada y canario EPERM), no calidad sin latencia; si esta carpeta no es `offline-*`, no hay evidencia de operación sin red. Tampoco hay validación continua.",
              "- ningún ajuste posterior al held-out: cualquier cambio requiere un congelado nuevo y este queda como intento."]
    (rdir / "EVIDENCE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:4 + len(rows) + 2]))
    return 0 if allok else 4


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--config", required=True)
    f.add_argument("--out", required=True)
    c = sub.add_parser("command")
    c.add_argument("--freeze", required=True)
    at = sub.add_parser("attempt")
    at.add_argument("--freeze", required=True)
    at.add_argument("--run-dir", required=True)
    at.add_argument("--status", choices=["ok", "failed"], required=True)
    at.add_argument("--note", default="")
    k = sub.add_parser("check")
    k.add_argument("--freeze", required=True)
    k.add_argument("--run", required=True)
    a = ap.parse_args()
    return {"freeze": freeze, "command": command, "attempt": attempt, "check": check}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
