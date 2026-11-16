import hashlib
import json

import pytest

from benchmark import audio as A
from benchmark import runner
from benchmark.contracts import ManifestError, load_manifest, select_split


def rec(**kw):
    d = dict(
        item_id="a",
        recording_id="r1",
        source_id="corpusX",
        speaker_id="s1",
        split="development",
        audio_path="a.wav",
        sha256="0" * 64,
        sample_rate=16000,
        channels=1,
        duration_s=2.0,
        dialect="es-AR",
        license="CC-BY-4.0",
        license_url="https://example.org/l",
        source_url="https://example.org/s",
        attribution="X",
        transcript_es="hola",
        transcript_es_provenance="human",
        reference_en=["hello"],
        reference_en_provenance="human",
        utterance_start_s=0.2,
        utterance_end_s=1.5,
        boundary_method="human_annotation",
        boundary_uncertainty_s=None,
    )
    d.update(kw)
    return d


def write(tmp_path, rows):
    p = tmp_path / "m.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return p


def test_valid_manifest_and_nullable_references(tmp_path):
    p = write(tmp_path, [rec(), rec(item_id="b", recording_id="r2", speaker_id="s2", split="test", reference_en=None, transcript_es=None, dialect="unknown", transcript_es_provenance="unknown", reference_en_provenance="unknown")])
    items = load_manifest(p)
    assert len(items) == 2 and items[1].reference_en is None
    assert [i.item_id for i in select_split(items, "test")] == ["b"]


def test_speaker_in_both_splits_fails(tmp_path):
    p = write(tmp_path, [rec(), rec(item_id="b", recording_id="r2", split="test")])
    with pytest.raises(ManifestError, match="fuga"):
        load_manifest(p)


def test_recording_in_both_splits_fails(tmp_path):
    p = write(tmp_path, [rec(), rec(item_id="b", speaker_id="s2", split="test")])
    with pytest.raises(ManifestError, match="fuga"):
        load_manifest(p)


def test_unknown_speakers_are_grouped_conservatively_by_source(tmp_path):
    rows = [
        rec(speaker_id=None),
        rec(item_id="b", recording_id="r2", speaker_id=None, split="test"),
    ]
    with pytest.raises(ManifestError, match="fuga"):
        load_manifest(write(tmp_path, rows))
    # distinta fuente: permitido
    rows[1]["source_id"] = "corpusY"
    assert len(load_manifest(write(tmp_path, rows))) == 2


def test_boundaries_require_declared_method_and_uncertainty(tmp_path):
    with pytest.raises(ManifestError, match="boundary_method"):
        load_manifest(write(tmp_path, [rec(boundary_method="unknown")]))
    with pytest.raises(ManifestError, match="boundary_uncertainty_s"):
        load_manifest(write(tmp_path, [rec(boundary_method="forced_alignment_auto")]))
    ok = load_manifest(write(tmp_path, [rec(boundary_method="forced_alignment_auto", boundary_uncertainty_s=0.08)]))
    assert not ok[0].has_precise_end  # alineación automática no es "preciso"


def test_sha256_verification(tmp_path):
    pcm = A.tone(0.5)
    wav = tmp_path / "a.wav"
    A.write_wav_pcm16_mono_16k(wav, pcm)
    good = hashlib.sha256(wav.read_bytes()).hexdigest()
    p = write(tmp_path, [rec(sha256=good, duration_s=0.5, utterance_end_s=0.4)])
    assert load_manifest(p, audio_root=tmp_path, verify_sha256=True)
    p = write(tmp_path, [rec(sha256="1" * 64, duration_s=0.5, utterance_end_s=0.4)])
    with pytest.raises(ManifestError, match="sha256"):
        load_manifest(p, audio_root=tmp_path, verify_sha256=True)


def test_non_16k_mono_wav_is_rejected_not_silently_resampled(tmp_path):
    import wave

    p = tmp_path / "x.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(44100)
        w.writeframes(b"\0\0" * 100)
    with pytest.raises(A.AudioFormatError):
        A.read_wav_pcm16_mono_16k(p)


def test_runner_refuses_to_tune_on_test_and_to_mix_splits():
    from conftest import make_item, run_dummy
    from benchmark.adapters.dummy import DummyAdapter

    t = make_item(0)
    t.split = "test"
    with pytest.raises(ValueError, match="test"):
        run_dummy(DummyAdapter(), [t], purpose="development")
    d = make_item(1)
    with pytest.raises(ValueError, match="mezclar"):
        run_dummy(DummyAdapter(), [t, d], purpose="final_test")
    # final_test sobre split test: permitido
    run_dummy(DummyAdapter(), [t], purpose="final_test")
