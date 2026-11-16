"""Auditoría del estimador de límites (scripts/datasets/estimate_boundaries.py)."""
import json
import sys

import numpy as np
import pytest

from conftest import ROOT

sys.path.insert(0, str(ROOT))
from benchmark import audio as A  # noqa: E402
from scripts.datasets import estimate_boundaries as E  # noqa: E402

SR = A.TARGET_SR


def speechlike(dur, level_db=-20.0, f=(500, 1500, 2500)):
    t = np.arange(round(dur * SR)) / SR
    x = sum(np.sin(2 * np.pi * fi * t) for fi in f) * (1 + 0.5 * np.sin(2 * np.pi * 4 * t))
    x = x / np.max(np.abs(x)) * 10 ** (level_db / 20) * 32767
    return x


def noise(dur, level_db, seed=0):
    return np.random.default_rng(seed).normal(0, 10 ** (level_db / 20) * 32767, round(dur * SR))


def pcm(*parts):
    return np.clip(np.round(np.concatenate(parts)), -32768, 32767).astype("<i2").tobytes()


def test_low_level_tail_noise_does_not_extend_the_end():
    """Cola de ruido/dither a -60 dBFS (piso relativo bajo): el fin sigue siendo el del habla.
    Con el umbral relativo de la v1 (P10 + 10 dB ~ -60) el fin se corría ~1.5 s."""
    x = pcm(noise(1.0, -75, 1), speechlike(1.5), noise(2.0, -62, 2))
    r = E.estimate(x)
    assert r["status"] == "ok"
    assert r["end_s"] == pytest.approx(2.5, abs=0.1) and r["start_s"] == pytest.approx(1.0, abs=0.1)
    assert r["threshold_from"] == "piso_absoluto"


def test_isolated_tail_burst_is_ignored_but_recorded():
    """Ráfaga de 0.3 s a -40 dBFS separada 0.6 s del habla (respiración/golpe): no es el fin."""
    x = pcm(noise(1.0, -75, 1), speechlike(1.5), noise(0.6, -75, 2), speechlike(0.3, -38), noise(0.8, -75, 3))
    r = E.estimate(x)
    assert r["status"] == "ok" and r["end_s"] == pytest.approx(2.5, abs=0.1)
    assert r["edge_runs_discarded"], "el descarte debe quedar registrado"


def test_isolated_leading_burst_is_ignored():
    x = pcm(noise(0.3, -75, 1), speechlike(0.15, -35), noise(1.0, -75, 2), speechlike(1.5), noise(1.0, -75, 3))
    r = E.estimate(x)
    assert r["status"] == "ok" and r["start_s"] == pytest.approx(1.45, abs=0.1)


def test_short_clicks_inside_noise_are_not_speech():
    x = pcm(noise(3.0, -75, 1))
    x_arr = np.frombuffer(x, dtype="<i2").astype(float)
    x_arr[16000:16000 + 800] += speechlike(0.05, -30)  # clic de 50 ms
    r = E.estimate(pcm(x_arr))
    assert r["status"] in ("no_speech_above_threshold", "no_separation")


def test_pauses_inside_speech_are_bridged():
    x = pcm(noise(0.5, -75, 1), speechlike(1.0), noise(0.1, -75, 2), speechlike(1.0), noise(1.0, -75, 3))
    r = E.estimate(x)
    assert r["start_s"] == pytest.approx(0.5, abs=0.1) and r["end_s"] == pytest.approx(2.6, abs=0.1)
    assert r["n_sustained_runs"] == 1


def test_speech_touching_end_of_file_is_not_trusted():
    x = pcm(noise(0.5, -75, 1), speechlike(2.0))
    assert E.estimate(x)["status"] == "speech_touches_end_of_file"


def test_methods_disagree_when_energy_is_outside_speech_band():
    """Zumbido de 80 Hz fuerte (no es habla): el método de banda de voz no lo ve."""
    hum = speechlike(2.0, -20, f=(80,))
    x = pcm(noise(0.5, -75, 1), hum, noise(1.0, -75, 2))
    r = E.estimate(x)
    assert r["status"] in ("band_check_failed", "methods_disagree")


def test_updated_manifests_are_valid_and_keep_references_and_history():
    from benchmark.contracts import load_manifest

    for name in ("fleurs_es_sample", "openslr61_es_ar_sample"):
        path = ROOT / "data" / "manifests" / f"{name}.jsonl"
        if not path.exists():
            pytest.skip("manifest ausente")
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        items = load_manifest(path, audio_root=ROOT / "data" / "audio", verify_sha256=True)
        assert len(items) == len(rows)
        for r in rows:
            assert r["transcript_es"]  # transcripción intacta
            if name == "fleurs_es_sample":
                assert r["reference_en"]  # OpenSLR no trae referencia en inglés (nunca la tuvo)
            assert r["boundary_estimator"]["version"] == 2
            if r["boundary_method"] == "estimated":
                assert r["boundary_uncertainty_s"] == 0.3
                assert r["boundary_history"], "el cambio de v1 debe quedar registrado"
                assert r["boundary_history"][0]["estimator_version"] == 1


def test_audited_fleurs_ends_match_independent_waveform_levels():
    """1516/1549 (dev): el fin estimado coincide con la última zona sostenida >-45 dBFS,
    no con la cola de ruido (v1: 11.26 s / 10.18 s)."""
    path = ROOT / "data" / "manifests" / "fleurs_es_sample.jsonl"
    if not path.exists():
        pytest.skip("manifest ausente")
    rows = {r["item_id"][-4:]: r for r in (json.loads(l) for l in path.read_text().splitlines() if l.strip())}
    assert rows["1516"]["utterance_end_s"] == pytest.approx(9.59, abs=0.05)
    assert rows["1549"]["utterance_end_s"] == pytest.approx(9.23, abs=0.05)


def test_voice_controls_manifest_has_provenance_hash_and_valid_audio():
    path = ROOT / "data" / "manifests" / "voice_controls.jsonl"
    if not path.exists():
        pytest.skip("controles ausentes")
    import hashlib

    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    assert rows
    for r in rows:
        f = ROOT / "data" / "audio" / r["audio_path"]
        assert hashlib.sha256(f.read_bytes()).hexdigest() == r["sha256"]
        A.read_wav_pcm16_mono_16k(f)  # formato exigido por el arnés
        assert r["license"] == "CC-BY-4.0" and r["source_url"] and r["attribution"]
        assert r["split"] == "development" and r["role"] == "human_en_control"
        assert "NO es ítem de evaluación" in r["purpose"]
