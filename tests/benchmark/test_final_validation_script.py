import importlib.util
import json
from types import SimpleNamespace

import pytest
from conftest import ROOT

spec = importlib.util.spec_from_file_location("fv", ROOT / "scripts" / "eval" / "final_validation.py")
fv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fv)


@pytest.fixture()
def env(tmp_path, monkeypatch):
    man = tmp_path / "m.jsonl"
    rows = [{"item_id": f"t{i}", "split": "test", "speaker_id": f"sp{i % 2}", "transcript_es": f"SECRETO-{i}", "audio_path": f"SECRETO/{i}.wav",
             "source_id": "x"} for i in range(4)] + [{"item_id": "d0", "split": "development", "speaker_id": "sd", "source_id": "x"}]
    man.write_text("\n".join(json.dumps(r) for r in rows))
    cfg = {"adapter": "a:B", "adapter_config": {"k": 1}, "manifest": str(man.relative_to(ROOT)) if man.is_relative_to(ROOT) else str(man),
           "audio_root": "data/audio", "repetitions": 3, "warmup": 2, "deadline_s": 5.0}
    cp = tmp_path / "cfg.json"
    cp.write_text(json.dumps(cfg))
    monkeypatch.setattr(fv, "ROOT", tmp_path)  # manifest relativo/absoluto: (tmp / abs) = abs
    monkeypatch.setattr(fv, "_code_hashes", lambda: {"runtime/x.py": "h1"})
    monkeypatch.setattr(fv, "_lock_hashes", lambda: {"uv.lock": "l1"})
    return SimpleNamespace(tmp=tmp_path, cfg=cp, man=man)


def _freeze(env, name="f1"):
    out = env.tmp / name
    assert fv.freeze(SimpleNamespace(config=str(env.cfg), out=str(out))) == 0
    return out


def test_freeze_counts_only_never_stores_heldout_content_and_is_immutable(env):
    out = _freeze(env)
    txt = (out / "FREEZE.json").read_text()
    assert "SECRETO" not in txt
    frz = json.loads(txt)
    assert frz["test_counts"] == {"n_items": 4, "n_speakers_clusters": 2, "items_per_cluster": [2, 2]}
    exp = frz["declared"]["heldout_exposure"]
    assert exp["status"] == "exposicion_parcial_reconocida" and exp["audio_listened"] is False and exp["used_for_tuning"] is False
    assert "NO se afirma que el held-out sea inédito" in exp["claim"] and "heldout_content_examined_before_freeze" not in frz["declared"]
    assert len(frz["freeze_id"]) == 12
    with pytest.raises(SystemExit, match="inmutable"):
        fv.freeze(SimpleNamespace(config=str(env.cfg), out=str(out)))


def test_example_config_and_missing_keys_are_rejected(env, tmp_path):
    ex = json.loads((ROOT / "docs" / "final-validation.config.example.json").read_text())
    p = tmp_path / "ex.json"
    p.write_text(json.dumps(ex))
    with pytest.raises(SystemExit, match="EJEMPLO"):
        fv.freeze(SimpleNamespace(config=str(p), out=str(tmp_path / "o")))
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"adapter": "a:B"}))
    with pytest.raises(SystemExit, match="incompleta"):
        fv.freeze(SimpleNamespace(config=str(bad), out=str(tmp_path / "o2")))


def test_command_prints_exact_split_test_run_and_refuses_after_code_change(env, capsys, monkeypatch):
    out = _freeze(env)
    assert fv.command(SimpleNamespace(freeze=str(out))) == 0
    txt = capsys.readouterr().out
    assert "--split test" in txt and "--repetitions 3" in txt and "--warmup 2" in txt and "PENDIENTE" in txt and "VALIDACIÓN CONTINUA" in txt
    monkeypatch.setattr(fv, "_code_hashes", lambda: {"runtime/x.py": "CAMBIADO"})
    assert fv.command(SimpleNamespace(freeze=str(out))) == 3
    assert "código cambió" in capsys.readouterr().err


def _fake_run(d, frz, **over):
    import hashlib

    d.mkdir(parents=True)
    cfg = frz["config"]
    meta = {"run_config": {"purpose": "final_test", "repetitions": 3, "warmup_streams": 2, "deadline_s": 5.0, "continuous": False,
                           "system_name": f"final-{frz['freeze_id']}-a1"},
            "split": "test", "adapter_config": cfg["adapter_config"], "n_items": 4, "aborted": None, "collector_error": None}
    meta.update(over.get("meta", {}))
    q = {"p95": {"n_clusters": 2, "low_sample": True}, "p99": {"n_clusters": 2, "low_sample": True}}
    g = {"outcome_counts": {"ok": 11, "timeout": 1}, "n_segments": 12, "primary_metric": "end_to_first_ms", "boundary_precision": "imprecise_or_unknown",
         "end_to_first_ms": {"censoring_aware": {"quantiles": q}}, "end_to_playout_est_ms": {}, "device_output": {}}
    run = {"meta": meta, "aggregate": {"groups": [g]}, "quality": {"planned_text": {}}}
    (d / "run.json").write_text(json.dumps(run))
    n = 12 + 2
    (d / "traces.jsonl").write_text("\n".join(json.dumps({"warmup": i < 2, "stream_id": f"s{i}"}) for i in range(over.get("n_traces", n))))
    (d / "audio").mkdir()
    wav = d / "audio" / "x__s5.wav"
    wav.write_bytes(b"RIFFfake")
    sha = hashlib.sha256(wav.read_bytes()).hexdigest() if not over.get("bad_hash") else "0" * 64
    (d / "audio" / "export_index.jsonl").write_text(json.dumps({"stream_id": "s5", "wav": wav.name, "sha256": sha, "exported": True}) + "\n")
    (d / "audible.json").write_text(json.dumps({"asr_label": "faster-whisper:small.en", "items": [{"wav": over.get("audible_wav", wav.name), "stream_id": "s5"}]}))


def test_check_passes_complete_evidence_and_always_states_what_is_not_certified(env, capsys):
    out = _freeze(env)
    frz = json.loads((out / "FREEZE.json").read_text())
    run = out / "streaming-1"
    fv.attempt(SimpleNamespace(freeze=str(out), run_dir=str(run), status="ok", note=""))
    _fake_run(run, frz)
    assert fv.check(SimpleNamespace(freeze=str(out), run=str(run))) == 0
    ev = (run / "EVIDENCE.md").read_text()
    assert "COMPLETA" in ev and "NO certifica 300 ms" in ev and "continua" in ev and "ningún ajuste posterior" in ev
    assert "exposición PARCIAL reconocida" in ev and "regla práctica del arnés" in ev and "no hay evidencia de operación sin red" in ev


def test_check_flags_wrong_config_missing_streams_and_code_drift(env, monkeypatch):
    out = _freeze(env)
    fv.attempt(SimpleNamespace(freeze=str(out), run_dir="r", status="failed", note="x"))
    frz = json.loads((out / "FREEZE.json").read_text())
    run = out / "streaming-1"
    _fake_run(run, frz, meta={"adapter_config": {"k": 2}}, n_traces=10)
    assert fv.check(SimpleNamespace(freeze=str(out), run=str(run))) == 4
    ev = (run / "EVIDENCE.md").read_text()
    assert "INCOMPLETA" in ev and ev.count("**FALLA**") >= 2


def test_deleted_or_added_lock_is_detected_in_both_directions(env, monkeypatch):
    out = _freeze(env)
    frz = json.loads((out / "FREEZE.json").read_text())
    monkeypatch.setattr(fv, "_lock_hashes", lambda: {})  # lock ELIMINADO
    assert any("eliminado: uv.lock" in p for p in fv._verify_unchanged(frz))
    monkeypatch.setattr(fv, "_lock_hashes", lambda: {"uv.lock": "l1", "extra.lock": "z"})
    assert any("añadido: extra.lock" in p for p in fv._verify_unchanged(frz))


def test_code_scope_includes_scripts_eval():
    hashes = fv._code_hashes()  # sin fixture: lee el árbol real
    assert any(k.startswith("scripts/eval/") for k in hashes) and "scripts/eval/final_validation.py" in hashes


def test_check_verifies_hashes_links_audible_and_attempt_to_this_run(env):
    out = _freeze(env)
    frz = json.loads((out / "FREEZE.json").read_text())
    run = out / "streaming-1"
    _fake_run(run, frz, bad_hash=True, audible_wav="otro.wav")
    fv.attempt(SimpleNamespace(freeze=str(out), run_dir=str(out / "streaming-9"), status="ok", note=""))  # otra carpeta
    assert fv.check(SimpleNamespace(freeze=str(out), run=str(run))) == 4
    ev = (run / "EVIDENCE.md").read_text()
    assert "sha256 del índice verificado" in ev and ev.count("**FALLA**") >= 3  # hash, audible vinculado, intento de esta corrida


def test_offline_means_no_network_sandbox_with_canary_and_separate_attempt(env, capsys):
    out = _freeze(env)
    fv.command(SimpleNamespace(freeze=str(out)))
    txt = capsys.readouterr().out
    assert "sandbox-exec -p" in txt and "deny network" in txt and "offline_canary.json" in txt and "errno" in txt
    assert "NO es calidad sin latencia" in txt and "offline-1" in txt and "-offline1" in txt
    frz = json.loads((out / "FREEZE.json").read_text())
    run = out / "offline-1"
    _fake_run(run, frz)
    fv.attempt(SimpleNamespace(freeze=str(out), run_dir=str(run), status="ok", note="offline"))
    (run / "offline_canary.json").write_text(json.dumps({"connected": False, "errno": 1}))
    fv.check(SimpleNamespace(freeze=str(out), run=str(run)))
    assert "canario EPERM" in (run / "EVIDENCE.md").read_text()
    (run / "offline_canary.json").write_text(json.dumps({"connected": True}))
    assert fv.check(SimpleNamespace(freeze=str(out), run=str(run))) == 4


def test_device_loopback_script_accepts_split_without_touching_benchmark_sources():
    spec2 = importlib.util.spec_from_file_location("dl_script", ROOT / "scripts" / "eval" / "device_loopback.py")
    m = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(m)
    args = SimpleNamespace(out="o", inner="x:Y", inner_config="{}", pre_silence_s=1.0, threshold_dbfs=-50.0, settle_s=0.3,
                           player_latency="default", repetitions=1, warmup=1, split="test")
    argv, _ = m.build(args)
    assert argv[argv.index("--split") + 1] == "test"
    args.split = "development"
    assert m.build(args)[0][m.build(args)[0].index("--split") + 1] == "development"
