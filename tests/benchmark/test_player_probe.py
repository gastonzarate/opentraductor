import asyncio

import pytest
from test_device_loopback import FakeCapture, FakePlayer, World

from benchmark import audio as A
from benchmark import player_probe as PP

FAST = dict(bursts=2, pre_silence_s=0.12, settle_s=0.05, gap_s=0.02, start_gap_s=0.02)


def _run(delays):
    """delays: modo -> retardo simulado (ms) del Player hasta BlackHole."""
    async def go():
        arms = []
        for i, mode in enumerate(PP.arm_order(1)):
            world = World(delay_ms=delays[mode])

            def mk_player(dev, lat, world=world):
                p = FakePlayer(world)
                p.latency_s, p.blocksize = (0.011 if lat == "low" else 0.1), 0
                p.requested = lat
                return p

            arms.append(await PP.run_arm(mode, i, make_capture=lambda world=world: FakeCapture(world), make_player=mk_player,
                                         **FAST))
        return arms

    return asyncio.run(go())


def test_tone_is_precomputed_audible_and_deterministic():
    pcm = PP.make_tone_pcm()
    assert pcm == PP.make_tone_pcm() and len(pcm) // 2 == round(0.3 * PP.SR)
    assert A.first_nonsilent_offset_s(pcm, PP.SR) is not None and A.rms_dbfs(pcm) > -20


def test_arm_order_is_abba_repeated():
    assert PP.arm_order(2) == ["default", "low", "low", "default"] * 2


def test_probe_detects_a_real_buffer_saving_and_reports_declared_latency_separately():
    arms = _run({"default": 80, "low": 25})
    s = PP.summarize(arms)
    assert {a["mode"] for a in arms} == {"default", "low"} and not s["excluded_arms"]
    assert s["modes"]["low"]["median_ms"] < s["modes"]["default"]["median_ms"]
    c = s["comparison"]
    assert c["verdict"] == "efecto candidato" and c["low_minus_default_median_ms"] < -30 and c["pairs_same_sign"]
    assert s["reported_stream_latency_s"] == {"default": 0.1, "low": 0.011}
    assert all("NO es una medición física" in a["player"]["reported_stream_latency_note"] for a in arms)


def test_probe_says_no_distinguishable_effect_when_modes_match():
    s = PP.summarize(_run({"default": 50, "low": 50}))
    assert s["comparison"]["verdict"] == "sin efecto distinguible"


def test_invalid_arm_is_excluded_not_averaged():
    arms = _run({"default": 50, "low": 50})
    arms[1]["validity"] = {"valid": False, "invalidating_reasons": ["capture_status_events=3"]}
    s = PP.summarize(arms)
    assert [e["arm"] for e in s["excluded_arms"]] == [1] and arms[1]["mode"] == "low"
    assert s["modes"]["low"]["n"] == 2  # solo el otro brazo low


def _arm(i, mode, v):
    return {"arm": i, "mode": mode, "validity": {"valid": True, "invalidating_reasons": []}, "foreign_audio_blocks": 0,
            "clips": [{"status": "ok", "device_minus_first_write_ms": x, "uncertainty_ms": 10.0} for x in v], "player": {"reported_stream_latency_s": 0.0}}


def test_pairs_are_temporal_by_abba_index_not_by_magnitude():
    # emparejar por magnitud (sorted) daría signos iguales; por posición ABBA los signos difieren
    arms = [_arm(0, "default", [100] * 3), _arm(1, "low", [90] * 3), _arm(2, "low", [130] * 3), _arm(3, "default", [120] * 3)]
    assert PP.temporal_pairs(arms) == [(0, 1), (3, 2)]
    c = PP.summarize(arms)["comparison"]
    assert [round(p["low_minus_default_ms"]) for p in c["temporal_pairs"]] == [-10, 10]
    assert c["pairs_same_sign"] is False and c["verdict"] == "sin efecto distinguible"


def test_excluded_arm_excludes_its_pair_and_independence_note_is_present():
    arms = [_arm(0, "default", [100] * 3), _arm(1, "low", [60] * 3), _arm(2, "low", [60] * 3), _arm(3, "default", [100] * 3)]
    arms[2]["validity"] = {"valid": False, "invalidating_reasons": ["x"]}
    s = PP.summarize(arms)
    assert [p["low_arm"] for p in s["comparison"]["temporal_pairs"]] == [1] and s["comparison"]["excluded_pairs"][0]["low_arm"] == 2
    assert "NO son muestras independientes" in s["independence_note"] and "NO cubre la calibración" in s["calibration_limit"]


def test_run_arm_preserves_full_clip_record_with_raw_callback_and_adc_onset():
    arms = _run({"default": 40, "low": 40})
    c = arms[0]["clips"][0]
    assert c["status"] == "ok" and c["device_onset_raw_callback_ns"] and c["device_onset_ns"] and c["first_audible_write_ns"]
    assert c["device_onset_raw_callback_ns"] >= c["device_onset_ns"]  # callback bruto ≥ onset corregido
