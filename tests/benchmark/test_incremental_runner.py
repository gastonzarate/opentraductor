"""Integración opt-in del collector incremental con CLI y runner. Sin modelos."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from benchmark import audio as A
from benchmark.__main__ import main as cli_main
from benchmark.runner import _write_incremental_records


def make_manifest(root: Path) -> Path:
    pcm = A.tone(0.10) + A.silence(0.10)
    wav = root / "sample.wav"
    A.write_wav_pcm16_mono_16k(wav, pcm)
    row = {
        "item_id": "collector-cli-fixture",
        "recording_id": "collector-cli-fixture",
        "source_id": "collector-test",
        "speaker_id": None,
        "split": "development",
        "audio_path": wav.name,
        "sha256": hashlib.sha256(wav.read_bytes()).hexdigest(),
        "sample_rate": 16000,
        "channels": 1,
        "duration_s": len(pcm) / 2 / 16000,
        "dialect": "unknown",
        "license": "test fixture",
        "license_url": "https://example.invalid/test-fixture",
        "source_url": "https://example.invalid/test-fixture",
        "attribution": "synthetic test fixture; not human speech",
        "transcript_es": None,
        "transcript_es_provenance": "unknown",
        "reference_en": None,
        "reference_en_provenance": "unknown",
        "utterance_start_s": 0.0,
        "utterance_end_s": 0.10,
        "boundary_method": "human_annotation",
        "boundary_uncertainty_s": None,
        "infra_synthetic": True,
    }
    manifest = root / "manifest.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return manifest


def run_cli(root: Path, out: Path, *, incremental: bool) -> int:
    args = [
        "run",
        "--manifest", str(make_manifest(root)),
        "--audio-root", str(root),
        "--split", "development",
        "--adapter", "benchmark.adapters.dummy:DummyAdapter",
        "--adapter-config", json.dumps({"behavior": "error", "endpoint_silence_ms": 20}),
        "--repetitions", "1",
        "--warmup", "0",
        "--deadline-s", "0.05",
        "--call-timeout-s", "1",
        "--watchdog-stall-s", "0",
        "--out", str(out),
    ]
    if incremental:
        args.append("--incremental-collector")
    return cli_main(args)


def test_cli_runner_opt_in_exports_timeline_without_pcm_and_preserves_failure(tmp_path: Path):
    out = tmp_path / "with-collector"
    assert run_cli(tmp_path, out, incremental=True) == 0

    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    inc = run["meta"]["incremental_collector"]
    assert inc["enabled"] is True and inc["contains_pcm"] is False
    assert inc["alignment"] is None and inc["semantic_metrics"] == "unavailable_without_independent_alignment"
    assert inc["total"] == {"scope": "warmup_plus_measured", "n_records": 1, "n_failed": 1, "n_timeout": 0}
    assert inc["warmup"]["n_records"] == 0 and inc["warmup"]["n_failed"] == 0
    assert inc["measured"]["n_records"] == 1 and inc["measured"]["n_failed"] == 1 and inc["measured"]["n_timeout"] == 0
    assert run["aggregate"]["groups"][0]["outcome_counts"]["error"] == 1

    rows = [json.loads(line) for line in (out / "incremental-collector.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1 and rows[0]["warmup"] is False
    assert rows[0]["incremental"]["alignment"] is None and rows[0]["incremental"]["ref_units"] == []
    assert rows[0]["incremental"]["n_failed"] == 1
    assert any(e["kind"] == "error" for e in rows[0]["events"])
    assert all("pcm" not in e for e in rows[0]["events"])
    assert rows[0]["timeline"] and rows[0]["input_deliveries"]


def test_summary_separates_failed_warmup_from_measured_and_aborted_skips(tmp_path: Path):
    records = {
        "warmup0": {"schema": "x", "stream_id": "warmup0", "incremental": {"n_failed": 1, "n_timeout": 0}},
        "measured-ok": {"schema": "x", "stream_id": "measured-ok", "incremental": {"n_failed": 0, "n_timeout": 0}},
    }

    class Observer:
        schema = "x"

        def stream_ids(self):
            return tuple(records)

        def export_record(self, stream_id):
            return dict(records[stream_id])

    warmup = [SimpleNamespace(stream_id="warmup0", error="synthetic warmup failure")]
    measured = [
        SimpleNamespace(stream_id="measured-ok", error=None),
        SimpleNamespace(stream_id="measured-missing", error="synthetic failure without observer record"),
        SimpleNamespace(stream_id="measured-skipped", error="AdapterHang: omitido, corrida abortada tras cuelgues consecutivos"),
    ]
    summary = _write_incremental_records(Observer(), warmup, measured, tmp_path)
    assert summary["total"] == {"scope": "warmup_plus_measured", "n_records": 2, "n_failed": 1, "n_timeout": 0}
    assert summary["warmup"]["n_failed"] == 1 and summary["warmup"]["n_records"] == 1
    assert summary["measured"]["n_failed"] == 0 and summary["measured"]["n_records"] == 1
    assert summary["measured"]["n_expected_without_aborted_skips"] == 2
    assert summary["measured"]["missing_stream_ids"] == ["measured-missing"]
    assert summary["measured"]["aborted_skip_stream_ids"] == ["measured-skipped"]
    rows = [json.loads(x) for x in (tmp_path / "incremental-collector.jsonl").read_text().splitlines()]
    assert [r["warmup"] for r in rows] == [True, False]


def test_cli_runner_default_does_not_create_incremental_artifact(tmp_path: Path):
    root = tmp_path / "plain-input"
    root.mkdir()
    out = tmp_path / "without-collector"
    assert run_cli(root, out, incremental=False) == 0
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run["meta"]["incremental_collector"] == {"enabled": False}
    assert run["meta"]["run_config"]["incremental_collector"] is False
    assert not (out / "incremental-collector.jsonl").exists()
    assert run["aggregate"]["groups"][0]["outcome_counts"]["error"] == 1
