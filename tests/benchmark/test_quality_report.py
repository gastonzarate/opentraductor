import pytest
import csv
import json

from conftest import make_item, run_dummy

from benchmark import listening, quality, report, runner
from benchmark.adapters.dummy import DummyAdapter
from benchmark.clock import RealClock


def test_translation_scores_unavailable_without_references():
    r = quality.translation_scores(["hello"], [None])
    assert r["status"] == "unavailable"


def test_translation_scores_counts_missing_refs_and_empty_hyps():
    r = quality.translation_scores(
        ["the cat sits on the mat", None, "dogs bark"],
        [["the cat sits on the mat"], ["a bird flies"], None],
    )
    assert r["status"] == "ok"
    assert r["n_missing_reference"] == 1 and r["n_empty_hypothesis"] == 1
    assert r["n_scored"] == 2
    assert r["bleu"] < 100  # la hipótesis vacía penaliza, no se omite


def test_neural_metric_is_not_faked():
    assert quality.neural_metric_status()["status"] == "unavailable"


def test_audible_text_requires_independent_asr():
    assert quality.audible_text_report([], None)["status"] == "unavailable"


def test_audible_text_vs_planned_flags_omission():
    res = run_dummy(
        DummyAdapter(), [make_item(0)], keep_pcm=True, adapter_config={"text_en": "one two three four five six"}
    )
    out = quality.audible_text_report(res.traces, lambda pcm, sr: "one two")
    assert out["status"] == "ok"
    it = next(iter(out["items"].values()))
    assert it["planned_text"].startswith("one two three") and it["audible_text"] == "one two"
    assert it["omission_suspected"] is True
    ok = quality.audible_text_report(res.traces, lambda pcm, sr: "one two three four five six")
    assert next(iter(ok["items"].values()))["omission_suspected"] is False


def test_blind_template_hides_system_names(tmp_path):
    st = [{"system": s, "item_id": f"i{k}", "audio_ref": f"{s}_{k}.wav"} for s in ("sysA", "sysB") for k in range(3)]
    listening.make_blind_template(st, tmp_path, raters=["r1", "r2"], seed=3)
    txt = (tmp_path / "ratings_template.csv").read_text()
    assert "sysA" not in txt and "sysB" not in txt
    rows = list(csv.DictReader(txt.splitlines()))
    assert len(rows) == 12
    assert set(listening.DIMENSIONS) <= set(rows[0])
    key = json.loads((tmp_path / "blind_key.json").read_text())
    assert {v["system"] for v in key["key"].values()} == {"sysA", "sysB"}


def test_report_labels_dummy_and_virtual_clock(tmp_path):
    run_dummy(DummyAdapter(), [make_item(0)], out_dir=str(tmp_path))
    md = report.write_report(tmp_path).read_text()
    assert "INFRA-TEST" in md and "NO es un benchmark real" in md
    assert "VIRTUAL" in md
    data = json.loads((tmp_path / "run.json").read_text())
    assert data["meta"]["infra_test_only"] is True
    assert data["meta"]["environment"]["benchmark_code_sha256"]
    assert (tmp_path / "traces.jsonl").exists()
    # las trazas no guardan PCM
    assert '"pcm"' not in (tmp_path / "traces.jsonl").read_text()


def test_real_clock_smoke_dummy():
    """Humo con reloj real: el replay corre a ritmo real (~2 s)."""
    import asyncio
    import time

    it = make_item(0, lead=0.1, speech=0.3, tail=0.1)
    cfg = runner.RunConfig(repetitions=1, warmup_streams=0, deadline_s=1.0, drain_grace_s=0.1, n_boot=10)

    async def main():
        return await runner.run_benchmark(DummyAdapter(), RealClock(), [it], cfg)

    w = time.monotonic()
    res = asyncio.run(main())
    wall = time.monotonic() - w
    r = res.segment_results[0]
    assert r.outcome == "ok" and r.clock_kind == "real"
    assert 400 <= r.end_to_first_ms <= 800  # 400 ms de endpoint + 100 ms de proceso
    assert wall >= 1.3  # ritmo real: audio + ventana de censura (no instantáneo)


def test_asr_wer_cer_against_human_transcript_separate_from_mt():
    rows = [("hola mundo cruel", "Hola, mundo.", "human"), (None, "buen día", "human"), ("x", "texto generado", "generated")]
    out = quality.asr_error_rates(rows)
    h = out["human"]
    assert h["n"] == 2 and h["n_empty_hypothesis"] == 1  # vacío penaliza, no se omite
    # 2 palabras ref + 2 ref; errores: 1 inserción + 2 borrados = 3 / 4
    assert h["wer"] == pytest.approx(0.75)
    assert out["generated"]["n"] == 1  # procedencias separadas
    assert quality.normalize_text("¡Hola, Mundo!") == "hola mundo"


def test_candidate_asr_es_integrated_and_unavailable_without_asr_events():
    it = make_item(0)
    it.text_es = "misma frase"
    it.segment.transcript_provenance = "human"
    res = run_dummy(DummyAdapter(), [it], adapter_config={"asr_text": "misma frase"})
    assert res.quality["candidate_asr_es"]["human"]["wer"] == 0.0

    # ASR que responde con texto vacío: penaliza (WER 1.0), no es "unavailable"
    res = run_dummy(DummyAdapter(), [it], adapter_config={"asr_text": ""})
    assert res.segment_results[0].asr_status == "ok"
    assert res.quality["candidate_asr_es"]["human"]["wer"] == 1.0
    assert res.quality["candidate_asr_es"]["human"]["n_empty_hypothesis"] == 1
    # sin ningún evento asr_final (end-to-end): unavailable
    res = run_dummy(DummyAdapter(), [it], adapter_config={"asr_text": None})
    assert res.segment_results[0].asr_status == "none"
    assert res.quality["candidate_asr_es"]["status"] == "unavailable"
