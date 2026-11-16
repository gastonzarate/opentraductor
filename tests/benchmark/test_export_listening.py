import csv
import json
import wave

import numpy as np
import pytest
from conftest import make_item, run_dummy

from benchmark import audio as A
from benchmark import export as X
from benchmark import listening
from benchmark.__main__ import main as cli_main
from benchmark.adapters.dummy import DummyAdapter
from benchmark.plugins import PluginUnavailable


def _export(tmp_path, name="sysA", **kw):
    d = tmp_path / name
    res = run_dummy(DummyAdapter(), [make_item(0), make_item(1)], export_audio_dir=str(d), system_name=name, **kw)
    return d, res


def test_export_writes_verified_wavs_and_index(tmp_path):
    d, res = _export(tmp_path)
    rows = X.read_index(d / "export_index.jsonl")
    assert len(rows) == 2 and all(r["exported"] and r["roundtrip_ok"] for r in rows)
    assert all(r["scope"] == "benchmark_only_synthetic_output" for r in rows)
    pcm, sr = A.read_wav_pcm16_mono(d / rows[0]["wav"])
    assert sr == 16000 and len(pcm) / 2 / sr == pytest.approx(rows[0]["duration_s"])
    assert X.verify_index(rows) == []
    assert res.meta["audio_export"]["n_exported"] == 2


def test_export_skips_silent_output_and_detects_corruption(tmp_path):
    d = tmp_path / "s"
    run_dummy(DummyAdapter(), [make_item(0)], adapter_config={"behavior": "silent"}, export_audio_dir=str(d))
    assert not (d / "export_index.jsonl").exists()  # nada audible que escuchar
    d2, _ = _export(tmp_path, "sysB")
    rows = X.read_index(d2 / "export_index.jsonl")
    (d2 / rows[0]["wav"]).write_bytes(b"corrupto")
    assert any("corrupto" in p for p in X.verify_index(rows))
    (d2 / rows[1]["wav"]).unlink()
    assert any("ausente" in p for p in X.verify_index(rows))


def test_listening_prepare_hides_systems_and_rejects_corrupt_index(tmp_path):
    a, _ = _export(tmp_path, "sysA")
    b, _ = _export(tmp_path, "sysB")
    out = tmp_path / "listen"
    info = listening.prepare_from_indexes([a / "export_index.jsonl", b / "export_index.jsonl"], out, raters=["r1", "r2"], seed=1)
    assert info["n_stimuli"] == 4 and "warning" in info  # dummy: advertencia
    files = sorted(p.name for p in (out / "stimuli").iterdir())
    assert len(files) == 4 and all(f.startswith("S") for f in files)
    txt = (out / "ratings_template.csv").read_text()
    assert "sysA" not in txt and "sysB" not in txt
    assert not any("sys" in f for f in files)
    key = json.loads((out / "blind_key.json").read_text())["key"]
    assert {v["system"] for v in key.values()} == {"sysA", "sysB"}
    # corromper un WAV: la preparación se niega
    rows = X.read_index(a / "export_index.jsonl")
    (a / rows[0]["wav"]).write_bytes(b"x")
    with pytest.raises(ValueError, match="corrupto"):
        listening.prepare_from_indexes([a / "export_index.jsonl"], tmp_path / "l2", raters=["r1"])


def test_listening_summary_per_system_ignores_empty_and_invalid(tmp_path):
    key = tmp_path / "k.json"
    key.write_text(json.dumps({"key": {"S1": {"system": "A"}, "S2": {"system": "B"}}}))
    csvp = tmp_path / "r.csv"
    with csvp.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["rater", "stimulus_id", *listening.DIMENSIONS, "comentario"])
        w.writeheader()
        w.writerow({"rater": "r1", "stimulus_id": "S1", "naturalidad": "4", "inteligibilidad": "5"})
        w.writerow({"rater": "r2", "stimulus_id": "S1", "naturalidad": "2", "acento": "9"})
        w.writerow({"rater": "r1", "stimulus_id": "S2", "naturalidad": "x"})
    out = listening.summarize_ratings(csvp, key)
    assert out["systems"]["A"]["dimensions"]["naturalidad"]["mean"] == 3.0
    assert out["systems"]["A"]["n_raters"] == 2
    assert "B" not in out["systems"]  # solo entradas inválidas
    assert out["n_invalid_cells"] == 2 and "no es MOS formal" in out["caveat"]


def test_resample_fft_preserves_duration_and_tone():
    pcm = A.tone(1.0, freq_hz=1000, sample_rate=24000)
    out = A.resample_fft(pcm, 24000, 16000)
    assert len(out) // 2 == 16000
    x = np.frombuffer(out, dtype="<i2").astype(float)
    peak = np.argmax(np.abs(np.fft.rfft(x))) 
    assert peak == 1000


def test_plugins_report_unavailable_instead_of_inventing_scores(tmp_path):
    from benchmark.plugins.utmos import UtmosV2

    with pytest.raises(PluginUnavailable):
        UtmosV2(tmp_path / "no-existe.pth")  # sin utmosv2 o sin pesos
    d, _ = _export(tmp_path)
    code = cli_main(["proxy-score", "--index", str(d / "export_index.jsonl"),
                     "--checkpoint", str(tmp_path / "no.pth"), "--out", str(tmp_path / "p.json")])
    assert code == 3 and not (tmp_path / "p.json").exists()


def test_asr_label_is_recorded_in_run_meta():
    def asr(pcm, sr):
        return "x"

    asr.label = "fake-asr:v1"
    res = run_dummy(DummyAdapter(), [make_item(0)], audible_asr=asr)
    assert res.meta["audible_asr_label"] == "fake-asr:v1"


def test_listening_reference_pairing_na_dimensions_controls_and_instructions(tmp_path):
    a, _ = _export(tmp_path, "kokoro-fixed")
    c, _ = _export(tmp_path, "cloner")
    src = tmp_path / "src_it0.wav"
    A.write_wav_pcm16_mono_16k(src, A.tone(0.5, freq_hz=200))
    human = tmp_path / "human_en.wav"
    A.write_wav_pcm16_mono_16k(human, A.tone(0.5, freq_hz=250))
    out = tmp_path / "listen"
    info = listening.prepare_from_indexes(
        [a / "export_index.jsonl", c / "export_index.jsonl"], out, raters=["r1", "r2"], seed=2,
        reference_audio={"it0": str(src)},  # it1 no tiene referencia
        system_meta={"cloner": {"voice_cloning": True}},
        human_controls=[(str(human), "CC0 de prueba")], degraded_controls=2,
    )
    assert info["n_controls"] == 3 and info["n_references"] == 1
    key = json.loads((out / "blind_key.json").read_text())["key"]
    by = lambda sysn, item: next(v for v in key.values() if v["system"] == sysn and v["item_id"] == item)  # noqa: E731
    # voz fija: similitud NA (no es fallo de clonación); con referencia: expresividad aplica
    assert "similitud_voz" in by("kokoro-fixed", "it0")["na_dims"]
    assert "expresividad" not in by("kokoro-fixed", "it0")["na_dims"]
    assert by("kokoro-fixed", "it0")["reference_id"] is not None
    # clonador con referencia: ambas aplican; sin referencia: ambas NA
    assert by("cloner", "it0")["na_dims"] == []
    assert set(by("cloner", "it1")["na_dims"]) == {"expresividad", "similitud_voz"}
    assert by("cloner", "it1")["reference_id"] is None
    # el mismo ítem comparte referencia entre sistemas; archivos con nombre ciego
    assert by("kokoro-fixed", "it0")["reference_id"] == by("cloner", "it0")["reference_id"]
    files = {p.name for p in (out / "stimuli").iterdir()}
    assert any(f.startswith("R") for f in files) and not any("kokoro" in f or "cloner" in f for f in files)
    rows = list(csv.DictReader((out / "ratings_template.csv").read_text().splitlines()))
    assert {"reference_id", "similitud_voz"} <= set(rows[0]) and any(r["similitud_voz"] == "NA" for r in rows)
    ins = (out / "INSTRUCCIONES.md").read_text()
    assert "1 (muy mala) a 5 (excelente)" in ins and "NA" in ins and "no constituyen un MOS formal" in ins
    assert sum(1 for v in key.values() if v["is_control"]) == 3
    # índice de controles verificable para proxy-score
    assert X.verify_index(X.read_index(out / "controls_index.jsonl")) == []


def test_degraded_control_is_deterministic_and_band_limited():
    pcm = A.tone(0.5, freq_hz=6000)
    d1 = listening.degrade_pcm(pcm, 16000, seed=1)
    assert d1 == listening.degrade_pcm(pcm, 16000, seed=1)
    assert A.rms_dbfs(d1) < A.rms_dbfs(pcm) - 10  # 6 kHz cae por el paso bajo


def test_summary_separates_controls_and_not_applicable(tmp_path):
    key = tmp_path / "k.json"
    key.write_text(json.dumps({"key": {"S1": {"system": "A"}, "S2": {"system": "control:human_en"}}}))
    csvp = tmp_path / "r.csv"
    with csvp.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["rater", "stimulus_id", *listening.DIMENSIONS])
        w.writeheader()
        w.writerow({"rater": "r1", "stimulus_id": "S1", "naturalidad": "3", "similitud_voz": "NA"})
        w.writerow({"rater": "r1", "stimulus_id": "S2", "naturalidad": "5"})
    out = listening.summarize_ratings(csvp, key)
    assert out["systems"]["A"]["not_applicable_counts"] == {"similitud_voz": 1} and out["n_invalid_cells"] == 0
    assert "control:human_en" in out["controls"] and "control:human_en" not in out["systems"]


def test_export_index_carries_planned_text_and_audible_score_uses_it(tmp_path):
    from benchmark import audible

    d, _ = _export(tmp_path, "sysA", adapter_config={"text_en": "one two three four five six"})
    rows = X.read_index(d / "export_index.jsonl")
    assert rows[0]["planned_text_en"] == "one two three four five six"
    refs = {"it0": (("one two three four five six",), "human"), "it1": (("one two three four five six",), "human")}
    res = audible.score_exports([d / "export_index.jsonl"], lambda pcm, sr: "one two three", refs=refs)
    s = res["per_system"]["sysA"]
    assert s["n"] == 2 and s["n_omission_suspected"] == 2  # la mitad de las palabras faltan
    assert s["planned_vs_audible_wer_corpus"] == pytest.approx(0.5)
    assert s["audible_vs_reference"]["human"]["status"] == "ok"
    # índice corrupto => se niega
    (d / rows[0]["wav"]).write_bytes(b"x")
    with pytest.raises(ValueError, match="corrupto"):
        audible.score_exports([d / "export_index.jsonl"], lambda p, s: "")
