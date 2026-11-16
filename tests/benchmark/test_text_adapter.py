import json

import pytest
from conftest import ROOT, run_dummy

from benchmark import audio as A
from benchmark import export as X
from benchmark import textchecks
from benchmark.adapters.text_mt_tts import TextMtTtsAdapter
from benchmark.textset import load_text_items


class FakeMT:
    TABLE = {
        "Vos tenés que mandar el informe antes del viernes.": "You need to send the report before Friday.",
        "No, no creo que el deploy esté listo hoy.": "I think the deploy is ready today.",  # pierde la negación
        "El presupuesto subió de 120 mil a 1,2 millones de dólares.": "The budget went up from 100 thousand to 1.2 million dollars.",  # 120 -> 100
    }

    def translate(self, t):
        return self.TABLE.get(t, "hello world. second clause here")


class FakeChunk:
    def __init__(self, text):
        self.pcm, self.sample_rate, self.text = A.tone(0.1, sample_rate=24000), 24000, text


class FakeTTS:
    def synth(self, text):
        yield FakeChunk(text)
        yield FakeChunk(text)


class InlineWorker:
    """El reloj virtual no espera hilos reales: las pruebas ejecutan en línea."""

    async def run(self, fn, *a):
        return fn(*a)

    def shutdown(self):
        pass


class Fake(TextMtTtsAdapter):
    def __init__(self):
        super().__init__()
        self._cpu = self._gpu = InlineWorker()

    def _make_mt(self, cfg):
        return FakeMT()

    def _make_tts(self, cfg):
        return FakeTTS()

    @staticmethod
    def _split(text):
        return [p.strip() + "." for p in text.rstrip(".").split(".") if p.strip()]


def text_items():
    path = ROOT / "data" / "manifests" / "text_meetings_ar_dev.jsonl"
    return load_text_items(path)


def _subset(ids):
    return [i for i in text_items() if i.item_id in ids]


IDS = {"txt-meet-ar-dev-01", "txt-meet-ar-dev-02", "txt-meet-ar-dev-05"}


def test_text_manifest_loads_54_dev_items_without_audio():
    items = text_items()
    assert len(items) == 54 and all(i.pcm == b"" and i.text_es for i in items)
    assert all(i.segment.reference_provenance == "generated" for i in items)  # no se presentan como humanas
    assert any("negacion" in i.segment.tags for i in items)


def test_mt_only_run_is_text_output_with_own_group_and_quality():
    res = run_dummy(Fake(), _subset(IDS), adapter_config={"stages": "mt"}, repetitions=1, warmup_streams=0)
    assert {r.output_kind for r in res.segment_results} == {"text"}
    assert all(r.outcome == "ok" and r.mt_text_en for r in res.segment_results)
    g = res.aggregate["groups"][0]
    assert g["output_kind"] == "text" and "mt_final" in g["primary_origin"] and g["adapter_mode"] == "text"
    assert res.quality["mt_text"]["generated"]["status"] == "ok"  # BLEU/chrF sobre la MT, por procedencia
    assert res.meta["adapter"]["name"] == "text-marian-ct2"


def test_mt_plus_tts_audio_output_and_clause_commits():
    res = run_dummy(Fake(), _subset({"txt-meet-ar-dev-01"}), adapter_config={"stages": "mt+tts"}, repetitions=1, warmup_streams=0)
    r = res.segment_results[0]
    assert r.output_kind == "audio" and r.outcome == "ok"
    assert r.mt_text_en == "hello world. second clause here" or r.mt_text_en.startswith("You need")
    assert r.n_commits >= 1 and r.planned_text_en  # chunks de una cláusula no se duplican
    assert res.meta["adapter"]["name"] == "text-marian-ct2+kokoro"


def test_two_clauses_one_commit_each_and_repeated_clause_is_kept():
    class Rep(Fake):
        @staticmethod
        def _split(text):
            return ["same clause.", "same clause."]

    res = run_dummy(Rep(), _subset({"txt-meet-ar-dev-01"}), adapter_config={"stages": "mt+tts"}, repetitions=1, warmup_streams=0)
    r = res.segment_results[0]
    assert r.n_commits == 2 and r.planned_text_en == "same clause. same clause."


def test_text_and_audio_runs_are_never_in_the_same_aggregate_group():
    from benchmark import metrics

    a = run_dummy(Fake(), _subset(IDS), adapter_config={"stages": "mt"}, repetitions=1, warmup_streams=0)
    b = run_dummy(Fake(), _subset(IDS), adapter_config={"stages": "mt+tts"}, repetitions=1, warmup_streams=0)
    agg = metrics.aggregate(a.segment_results + b.segment_results, n_boot=10)
    assert sorted(g["output_kind"] for g in agg["groups"]) == ["audio", "text"]


def test_surface_checks_flag_lost_negation_and_changed_number_without_claiming_correctness():
    res = run_dummy(Fake(), _subset(IDS), adapter_config={"stages": "mt"}, repetitions=1, warmup_streams=0)
    tc = res.quality["text_checks"]
    assert "NO miden exactitud semántica" in tc["label"]
    neg = tc["checks"]["negation"]
    assert neg["n_applicable"] >= 1 and neg["n_flagged"] == 1  # 'No, no creo...' -> sin negación
    assert neg["flagged_examples"][0]["item_id"] == "txt-meet-ar-dev-02"
    num = tc["checks"]["numbers"]
    assert any(e["item_id"] == "txt-meet-ar-dev-05" and "120" in e["missing_in_output"] for e in num["flagged_examples"])
    assert "negacion" in tc["by_phenomenon"]


def test_textchecks_unit():
    assert textchecks.number_tokens("de 120 mil a 1,2 millones, 15:30") == ["120", "12", "15", "30"]
    assert textchecks.check_numbers("1,2 millones", "1.2 million")["flag"] is False
    assert textchecks.check_negation("No sé", "I don't know")["flag"] is False  # "don't" cuenta
    assert textchecks.check_negation("Nunca falla", "It fails")["flag"] is True
    assert textchecks.check_negation("Hola", "Hi") is None
    assert textchecks.check_terms("subí el deploy a la API", "I pushed the rollout to the interface")["missing_in_output"] == ["deploy", "api"]
    assert textchecks.check_terms("buenos días", "good morning") is None


def test_describe_never_crashes_without_runtime_models_and_records_engine_versions():
    ad = Fake()
    d = ad.describe()
    assert "engine_versions" in d and ("describe_error" in d or "models" in d)


def test_text_adapter_is_offline_and_exports_audio_for_listening(tmp_path):
    res = run_dummy(Fake(), _subset({"txt-meet-ar-dev-01"}), adapter_config={"stages": "mt+tts"},
                    repetitions=1, warmup_streams=0, export_audio_dir=str(tmp_path / "x"), system_name="kokoro-fixed")
    rows = X.read_index(tmp_path / "x" / "export_index.jsonl")
    assert rows[0]["sample_rate"] == 24000 and X.verify_index(rows) == []


def test_real_threads_and_real_clock_path_with_fake_components():
    """Hilos dedicados reales + reloj real (el reloj virtual no los espera)."""
    import asyncio

    from benchmark import runner
    from benchmark.clock import RealClock

    class Real(TextMtTtsAdapter):
        def _make_mt(self, cfg):
            return FakeMT()

        def _make_tts(self, cfg):
            return FakeTTS()

        @staticmethod
        def _split(text):
            return [text]

    cfg = runner.RunConfig(repetitions=1, warmup_streams=0, deadline_s=2.0, adapter_config={"stages": "mt+tts"}, n_boot=10)

    async def main():
        return await runner.run_benchmark(Real(), RealClock(), _subset({"txt-meet-ar-dev-01"}), cfg)

    res = asyncio.run(main())
    r = res.segment_results[0]
    assert r.outcome == "ok" and r.clock_kind == "real" and r.output_kind == "audio"
    assert r.request_to_first_ms < 500


def test_item_ids_subset_selection_preserves_order_and_rejects_unknown():
    from benchmark.textset import load_text_items

    path = ROOT / "data" / "manifests" / "text_meetings_ar_dev.jsonl"
    ids = ["txt-meet-ar-dev-04", "txt-meet-ar-dev-01"]
    assert [i.item_id for i in load_text_items(path, ids=ids)] == ids
    with pytest.raises(Exception, match="no encontrados"):
        load_text_items(path, ids=["nope"])
