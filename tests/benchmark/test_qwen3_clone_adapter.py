import inspect
import json
import shutil

import pytest
from conftest import ROOT, run_dummy
from test_text_adapter import FakeMT, InlineWorker, _subset

from benchmark import audio as A
from benchmark.adapters import text_mt_qwen3_clone as Q

REF = ROOT / "data" / "audio" / "examples" / "kyutai-0a67.wav"


class FakeResult:
    def __init__(self, audio, sr=24000):
        self.audio, self.sample_rate = audio, sr


class FakeModel:
    sample_rate = 24000

    def __init__(self, results=None):
        self.calls = []
        self.results = results if results is not None else [FakeResult(b"a"), FakeResult(None), FakeResult(b"b")]

    def generate(self, text, **kw):
        self.calls.append((text, kw))
        yield from self.results


def _tts(model=None, **cfg):
    full = {**Q.DEFAULTS, "ref_audio": str(REF), **cfg}
    return Q.Qwen3CloneTTS(full, model=model or FakeModel(), to_pcm=lambda a: A.tone(0.05, sample_rate=24000))


class FakeAdapter(Q.TextMtQwen3CloneAdapter):
    def __init__(self, model):
        super().__init__()
        self._cpu = self._gpu = InlineWorker()
        self._model = model

    def _make_mt(self, cfg):
        return FakeMT()

    def _make_tts(self, cfg):
        return Q.Qwen3CloneTTS(cfg, model=self._model, to_pcm=lambda a: A.tone(0.05, sample_rate=24000))


def test_reference_is_public_licensed_and_hash_checked(tmp_path):
    ref = Q.load_reference(str(REF))
    assert ref["license"] == "CC0-1.0" and "no es voz del usuario" not in ref and "user voice" in ref["note"]
    bad = tmp_path / "r.wav"
    shutil.copy(REF, bad)
    meta = json.loads(REF.with_suffix(".json").read_text())
    (tmp_path / "r.json").write_text(json.dumps({**meta, "sha256": "0" * 64}))
    with pytest.raises(ValueError, match="sha256"):
        Q.load_reference(str(bad))
    (tmp_path / "r.json").write_text(json.dumps({k: v for k, v in meta.items() if k != "license"}))
    with pytest.raises(ValueError, match="license"):
        Q.load_reference(str(bad))
    (tmp_path / "r.json").unlink()
    with pytest.raises(FileNotFoundError, match="licencia"):
        Q.load_reference(str(bad))


def test_clone_mode_follows_ref_text_and_kwargs_reach_the_api():
    t = _tts(streaming_interval=0.25, temperature=0.7, seed=None)
    assert t.clone_mode == "speaker_embedding_only"  # ref_text=None: no se inventa transcripción
    chunks = list(t.synth("hello there"))
    assert len(chunks) == 2  # el resultado sin audio se omite
    _, kw = t.model.calls[0]
    assert kw["stream"] is True and kw["streaming_interval"] == 0.25 and kw["ref_text"] is None
    assert kw["ref_audio"] == str(REF) and kw["lang_code"] == "english" and kw["temperature"] == 0.7
    assert "top_k" not in kw  # sin valor: lo decide la config del modelo
    t2 = _tts(ref_text="texto de referencia")
    assert t2.clone_mode == "icl" and list(t2.synth("x")) and t2.model.calls[0][1]["ref_text"] == "texto de referencia"


def test_kwargs_match_installed_qwen3_generate_signature():
    try:
        from mlx_audio.tts.models.qwen3_tts.qwen3_tts import Model
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"mlx_audio no disponible: {exc}")
    params = set(inspect.signature(Model.generate).parameters)
    kw = _tts(temperature=1.0, top_k=5, top_p=0.9, max_tokens=100)._kwargs()
    assert set(kw) <= params


def test_adapter_whole_utterance_by_default_and_per_clause_on_request():
    m = FakeModel([FakeResult(b"a"), FakeResult(b"b")])
    a = FakeAdapter(m)
    res = run_dummy(a, _subset({"txt-meet-ar-dev-01"}), adapter_config={}, repetitions=1, warmup_streams=0)
    r = res.segment_results[0]
    assert r.outcome == "ok" and r.output_kind == "audio" and r.n_commits == 1 and len(m.calls) == 1
    assert res.meta["adapter"]["name"] == "text-marian-ct2+qwen3-tts-clone"
    assert res.meta["adapter_metadata"]["params"]["streaming_interval"] == 0.5
    m2 = FakeModel([FakeResult(b"a")])
    b = FakeAdapter(m2)
    res2 = run_dummy(b, _subset({"txt-meet-ar-dev-01"}), adapter_config={"split_clauses": True}, repetitions=1, warmup_streams=0)
    assert res2.segment_results[0].outcome == "ok" and len(m2.calls) == len(b._split(res2.segment_results[0].mt_text_en)) >= 1


def test_adapter_rejects_mt_only_and_bad_interval():
    import asyncio

    for cfg, msg in (({"stages": "mt"}, "mt\\+tts"), ({"streaming_interval": 0}, "streaming_interval")):
        with pytest.raises(ValueError, match=msg):
            asyncio.run(FakeAdapter(FakeModel()).open(cfg))


def test_default_text_adapter_still_uses_kokoro_hook():
    from benchmark.adapters.text_mt_tts import TextMtTtsAdapter

    assert TextMtTtsAdapter._describe_tts is not Q.TextMtQwen3CloneAdapter._describe_tts
    assert TextMtTtsAdapter.name == "text-marian-ct2"


# ---------------------------------------------------------------- condicionamiento por ítem (piloto interlingüe)
def _pilot(tmp_path, texts=("hola uno", "hola dos"), provenance="human"):
    items = []
    for n, t in enumerate(texts):
        wav = tmp_path / f"c{n}.wav"
        A.write_wav_pcm16(wav, A.tone(0.2, sample_rate=16000), 16000)
        import hashlib

        items.append({"item_id": f"it{n}", "audio_path": wav.name, "sha256": hashlib.sha256(wav.read_bytes()).hexdigest(),
                      "transcript_es": t, "transcript_es_provenance": provenance, "license": "CC-BY-SA-4.0", "speaker_id": f"s{n}"})
    p = tmp_path / "pilot.json"
    p.write_text(json.dumps({"items": items}))
    return p


def test_per_item_conditioning_uses_own_audio_and_human_transcript(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = _pilot(tmp_path)
    t = Q.Qwen3CloneTTS({**Q.DEFAULTS, "conditioning_file": str(p)}, model=FakeModel(), to_pcm=lambda a: A.tone(0.05, sample_rate=24000))
    assert t.clone_mode == "icl_per_item" and t.ref is None
    with pytest.raises(RuntimeError, match="select_item"):
        list(t.synth("hello"))
    t.select_item("hola dos")
    list(t.synth("hello two"))
    kw = t.model.calls[0][1]
    assert kw["ref_audio"].endswith("c1.wav") and kw["ref_text"] == "hola dos"
    with pytest.raises(KeyError):
        t.select_item("texto desconocido")  # nunca cae a otra voz


def test_per_item_conditioning_rejects_non_human_transcript_hash_mismatch_and_duplicates(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="humana"):
        Q.load_conditioning(str(_pilot(tmp_path, provenance="generated")))
    with pytest.raises(ValueError, match="repetido"):
        Q.load_conditioning(str(_pilot(tmp_path, texts=("igual", "igual"))))
    p = _pilot(tmp_path)
    d = json.loads(p.read_text())
    d["items"][0]["sha256"] = "0" * 64
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="sha256"):
        Q.load_conditioning(str(p))


def test_adapter_selects_reference_per_text_and_records_mode(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = _pilot(tmp_path, texts=("Vos tenés que mandar el informe antes del viernes.", "No, no creo que el deploy esté listo hoy."))
    m = FakeModel([FakeResult(b"a")])

    class PerItem(FakeAdapter):
        def _make_tts(self, cfg):
            return Q.Qwen3CloneTTS(cfg, model=self._model, to_pcm=lambda a: A.tone(0.05, sample_rate=24000))

    a = PerItem(m)
    res = run_dummy(a, _subset({"txt-meet-ar-dev-01", "txt-meet-ar-dev-02"}), adapter_config={"conditioning_file": str(p)},
                    repetitions=1, warmup_streams=0)
    assert all(r.outcome == "ok" for r in res.segment_results)
    refs = {kw["ref_text"] for _, kw in m.calls}
    assert refs == {"Vos tenés que mandar el informe antes del viernes.", "No, no creo que el deploy esté listo hoy."}
