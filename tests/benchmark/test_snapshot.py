import hashlib
import json

from conftest import make_item, run_dummy

from benchmark import snapshot
from benchmark.adapters.dummy import DummyAdapter


def test_project_snapshot_hashes_runtime_code_and_locks_and_changes_with_edits(tmp_path):
    (tmp_path / "runtime").mkdir()
    (tmp_path / "runtime" / "a.py").write_text("x = 1\n")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "ignored.py").write_text("no")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "t.py").write_text("no")
    (tmp_path / "uv.lock").write_text("lock-v1")
    (tmp_path / "pyproject.toml").write_text("[project]\n")
    s1 = snapshot.project_snapshot(tmp_path)
    assert set(s1["runtime_code"]) == {"runtime"}  # data/ y tests/ no son runtime
    assert s1["lock_files"]["uv.lock"] == hashlib.sha256(b"lock-v1").hexdigest()
    (tmp_path / "runtime" / "a.py").write_text("x = 2\n")
    (tmp_path / "uv.lock").write_text("lock-v2")
    s2 = snapshot.project_snapshot(tmp_path)
    assert s2["runtime_code_sha256"] != s1["runtime_code_sha256"]  # cambia con cada iteración
    assert s2["lock_files"]["uv.lock"] != s1["lock_files"]["uv.lock"]


def test_environment_snapshot_lists_all_packages_and_key_versions():
    env = snapshot.environment_snapshot()
    assert "numpy" in env["all_packages"] and env["key_packages"]["numpy"] == env["all_packages"]["numpy"]
    assert {"mlx", "ctranslate2", "torch", "transformers", "misaki", "parakeet-mlx"} <= set(env["key_packages"])
    assert env["project"]["runtime_code_sha256"]


def test_weights_fingerprint_uses_hf_blob_names_without_rereading_and_caches(tmp_path):
    blobs = tmp_path / "models--org--m" / "blobs"
    snap = tmp_path / "models--org--m" / "snapshots" / "abc123"
    blobs.mkdir(parents=True)
    snap.mkdir(parents=True)
    data = b"pesos"
    sha = hashlib.sha256(data).hexdigest()
    (blobs / sha).write_bytes(data)
    (snap / "model.bin").symlink_to(blobs / sha)
    (snap / "config.json").write_text("{}")  # no LFS: se calcula y se cachea
    cache = tmp_path / "cache.json"
    fp = snapshot.weights_fingerprint(snap, cache_path=cache)
    assert fp["hf_revision"] == "abc123" and fp["hf_repo"] == "org/m"
    assert fp["files"]["model.bin"] == {"size": 5, "sha256": sha, "source": "hf_blob_name"}
    assert fp["files"]["config.json"]["source"] == "computed"
    fp2 = snapshot.weights_fingerprint(snap, cache_path=cache)
    assert fp2["files"]["config.json"]["source"] == "cache"  # no relee
    assert fp2["sha256_aggregate"] == fp["sha256_aggregate"]


def test_adapter_describe_is_recorded_and_missing_describe_is_warned():
    class Described(DummyAdapter):
        def describe(self):
            return {"models": {"asr": {"repo": "x/y", "revision": "r1"}}, "adapter_code_sha256": "ab"}

    res = run_dummy(Described(), [make_item(0)])
    assert res.meta["adapter_metadata"]["models"]["asr"]["revision"] == "r1"
    assert "adapter_metadata_warning" not in res.meta
    res = run_dummy(DummyAdapter(), [make_item(0)])
    assert "NO quedan registrados" in res.meta["adapter_metadata_warning"]
    json.dumps(res.meta, default=str)  # serializable


def test_load_snapshot_records_contention_without_claiming_idle():
    s = snapshot.load_snapshot()
    assert {"loadavg_1_5_15", "ncpu", "contention_possible", "other_processes_over_50pct_cpu", "note"} <= set(s)
    assert "exclusividad solo del equipo" in s["note"]


def test_run_meta_and_report_label_contention(tmp_path, monkeypatch):
    from benchmark import report

    fake = {"loadavg_1_5_15": (9.0, 8.0, 7.0), "ncpu": 4, "other_processes_over_50pct_cpu": [{"comm": "python", "cpu_pct": 99.0}],
            "other_cpu_pct_total": 120.0, "contention_possible": True, "note": "exclusividad solo del equipo opentraductor"}
    monkeypatch.setattr(snapshot, "load_snapshot", lambda: fake)
    res = run_dummy(DummyAdapter(), [make_item(0)], out_dir=str(tmp_path))
    assert res.meta["contention_possible"] is True and "PILOTO" in res.meta["run_status_label"]
    md = report.write_report(tmp_path).read_text()
    assert "PILOTO con contención posible" in md and "no se afirma máquina en reposo" in md
