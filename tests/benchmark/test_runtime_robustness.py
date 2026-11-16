"""Cuelgues, limpieza, CLI, orígenes por modo, playout estimado, metadatos, calidad."""
import asyncio
import hashlib
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from conftest import ROOT, make_item, run_dummy

from benchmark import audio as A
from benchmark import metrics, report, runner
from benchmark.__main__ import main as cli_main
from benchmark.adapters.dummy import DummyAdapter
from benchmark.clock import VirtualClock, run_virtual
from benchmark.contracts import AdapterMode, ManifestError
from benchmark.replay import Segment, StreamTrace, TraceEvent


# ------------------------------ cuelgues ----------------------------------


class HangPush(DummyAdapter):
    closed = False

    async def push_audio(self, stream_id, pcm, t):
        await asyncio.Event().wait()  # nunca retorna

    async def close(self):
        HangPush.closed = True
        await super().close()


def test_adapter_that_never_returns_yields_failure_and_run_finishes():
    HangPush.closed = False
    res = run_dummy(
        HangPush(), [make_item(0), make_item(1), make_item(2)],
        call_timeout_s=1.0, abort_after_hangs=2, repetitions=1,
    )
    assert HangPush.closed  # cleanup garantizado
    assert all(r.outcome == "error" for r in res.segment_results)
    assert len(res.segment_results) == 3  # nada sale del denominador
    assert "AdapterHang" in res.traces[0].error and "push_audio" in res.traces[0].error
    assert res.meta["aborted"] is not None  # abortó tras 2 cuelgues
    assert "omitido" in res.traces[2].error  # el tercero no se ejecutó, pero cuenta
    sm = res.aggregate["groups"][0]["end_to_first_ms"]
    assert sm["counts"]["failure"] == 3


def test_process_file_hang_is_bounded():
    class HangFile(DummyAdapter):
        async def process_file(self, stream_id, pcm):
            await asyncio.Event().wait()

    res = run_dummy(HangFile(AdapterMode.OFFLINE_FILE), [make_item(0)], call_timeout_s=1.0, deadline_s=1.0)
    assert res.segment_results[0].outcome == "error"
    assert "process_file" in res.traces[0].error


class BadStart(DummyAdapter):
    closed = False

    async def start_stream(self, stream_id, seed):
        raise RuntimeError("boom en start_stream")

    async def close(self):
        BadStart.closed = True
        await super().close()


def test_warmup_failure_still_closes_adapter_and_collector():
    BadStart.closed = False
    res = run_dummy(BadStart(), [make_item(0)], warmup_streams=2, repetitions=1)
    assert BadStart.closed
    assert res.segment_results[0].outcome == "error"
    assert "boom" in res.traces[0].error


def test_open_failure_raises_runerror_records_failure_and_closes(tmp_path):
    class BadOpen(DummyAdapter):
        closed = False

        async def open(self, config):
            raise RuntimeError("no hay modelo")

        async def close(self):
            BadOpen.closed = True
            await super().close()

    with pytest.raises(runner.RunError, match="no hay modelo"):
        run_dummy(BadOpen(), [make_item(0)], out_dir=str(tmp_path))
    assert BadOpen.closed
    assert json.loads((tmp_path / "FAILURE.json").read_text())["failure"] == "open_failed"


def test_watchdog_kills_process_blocked_by_synchronous_inference(tmp_path):
    """Un push_audio que bloquea el event loop con time.sleep no puede
    interrumpirse con wait_for: el watchdog termina el proceso (código 70) y
    deja FAILURE.json."""
    script = textwrap.dedent(
        f"""
        import asyncio, sys, time
        sys.path.insert(0, {str(ROOT)!r})
        sys.path.insert(0, {str(ROOT / 'tests' / 'benchmark')!r})
        from conftest import make_item
        from benchmark import runner
        from benchmark.adapters.dummy import DummyAdapter
        from benchmark.clock import RealClock

        class Blocker(DummyAdapter):
            async def push_audio(self, stream_id, pcm, t):
                time.sleep(30)  # inferencia síncrona

        cfg = runner.RunConfig(repetitions=1, warmup_streams=0, deadline_s=1.0,
                               watchdog_stall_s=0.5, out_dir={str(tmp_path)!r})
        asyncio.run(runner.run_benchmark(Blocker(), RealClock(), [make_item(0)], cfg))
        """
    )
    p = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=25)
    assert p.returncode == 70, p.stderr
    rec = json.loads((tmp_path / "FAILURE.json").read_text())
    assert rec["failure"] == "event_loop_stalled"
    assert rec["context"]["stream_id"].startswith("r0-")


# ------------------------------- CLI --------------------------------------


def _write_corpus(tmp_path, sha_ok=True):
    pcm = A.silence(0.1) + A.tone(0.3) + A.silence(0.1)
    wav = tmp_path / "a.wav"
    A.write_wav_pcm16_mono_16k(wav, pcm)
    sha = hashlib.sha256(wav.read_bytes()).hexdigest() if sha_ok else "1" * 64
    from test_manifest_splits import rec

    row = rec(sha256=sha, duration_s=0.5, utterance_start_s=0.1, utterance_end_s=0.4)
    m = tmp_path / "m.jsonl"
    m.write_text(json.dumps(row))
    return m


def test_cli_verifies_sha256_by_default(tmp_path):
    m = _write_corpus(tmp_path, sha_ok=False)
    with pytest.raises(ManifestError, match="sha256"):
        cli_main(["run", "--manifest", str(m), "--audio-root", str(tmp_path),
                  "--adapter", "benchmark.adapters.dummy:DummyAdapter", "--out", str(tmp_path / "o")])


def test_cli_fails_with_recorded_failure_when_open_fails(tmp_path, capsys):
    m = _write_corpus(tmp_path)
    (tmp_path / "badmod.py").write_text(
        "from benchmark.adapters.dummy import DummyAdapter\n"
        "class Bad(DummyAdapter):\n"
        "    async def open(self, c):\n        raise RuntimeError('sin pesos')\n"
    )
    sys.path.insert(0, str(tmp_path))
    try:
        code = cli_main(["run", "--manifest", str(m), "--audio-root", str(tmp_path),
                         "--adapter", "badmod:Bad", "--out", str(tmp_path / "o")])
    finally:
        sys.path.remove(str(tmp_path))
    assert code == 2 and (tmp_path / "o" / "FAILURE.json").exists()


# --------------------- playout estimado / orígenes -------------------------


def _tts(recv_s, dur_s, off=None):
    return TraceEvent(recv_ns=round(recv_s * 1e9), kind="tts_chunk", stream_id="s",
                      pcm_samples=round(dur_s * 16000), sample_rate=16000,
                      first_nonsilent_offset_s=off)


def test_playout_estimate_includes_previous_silent_chunks_in_queue():
    seg = Segment("a", 0.0, 1.0, "human_annotation")
    # chunk silencioso de 0.5 s llega en 1.0; el audible llega en 1.1
    ev = [_tts(1.0, 0.5, None), _tts(1.1, 0.5, 0.0)]
    tr = StreamTrace("s", "streaming", "virtual", 0, 10.0, [seg], events=ev, deadline_s=5.0)
    r = metrics.segment_results(tr)[0]
    assert r.end_to_first_ms == pytest.approx(100.0)  # disponible
    assert r.end_to_playout_est_ms == pytest.approx(500.0)  # cola: empieza a 1.5 s
    assert r.first_device_ns is None


def test_device_output_is_separate_and_unavailable_by_default():
    res = run_dummy(DummyAdapter(), [make_item(0)])
    assert res.aggregate["groups"][0]["device_output"]["status"] == "unavailable"


@pytest.mark.parametrize("mode", [AdapterMode.TEXT, AdapterMode.OFFLINE_FILE])
def test_text_and_offline_use_request_origin_not_speech_end(mode):
    it = make_item(0)
    it.text_es = "hola"
    res = run_dummy(DummyAdapter(mode), [it], adapter_config={"processing_ms": 150.0})
    r = res.segment_results[0]
    assert r.outcome == "ok"
    assert r.end_to_first_ms is None and r.signed_end_to_first_ms is None
    assert r.start_origin == "request_start"
    assert r.request_to_first_ms == pytest.approx(150.0, abs=30)
    g = res.aggregate["groups"][0]
    assert g["primary_metric"] == "request_to_first_ms"
    assert "petición" in g["primary_origin"]
    assert g["end_to_playout_est_ms"] is None
    assert res.playout[0]["offline_throughput"]["rtf"] is not None


def test_cold_start_includes_deferred_load_in_start_stream():
    class Lazy(DummyAdapter):
        async def start_stream(self, stream_id, seed):
            await asyncio.sleep(0.4)  # carga diferida del modelo
            await super().start_stream(stream_id, seed)

    res = run_dummy(Lazy(), [make_item(0)], warmup_streams=1)
    assert res.cold_start["first_start_stream_ms"] == pytest.approx(400.0, abs=5)
    assert res.cold_start["open_ms"] < 5


# --------------------- metadatos y calidad integrada -----------------------


def test_manifest_metadata_survives_to_results_and_quality_splits_by_provenance(tmp_path):
    from test_manifest_splits import rec
    from benchmark.contracts import load_manifest

    rows = []
    for i, (spk, prov) in enumerate((("s1", "human"), ("s2", "generated"))):
        pcm = A.silence(0.1) + A.tone(0.3) + A.silence(0.1)
        A.write_wav_pcm16_mono_16k(tmp_path / f"{i}.wav", pcm)
        rows.append(rec(item_id=f"i{i}", recording_id=f"r{i}", speaker_id=spk, audio_path=f"{i}.wav",
                        sha256=hashlib.sha256((tmp_path / f"{i}.wav").read_bytes()).hexdigest(),
                        duration_s=0.5, utterance_start_s=0.1, utterance_end_s=0.4,
                        reference_en=["the cat sits on the mat"], reference_en_provenance=prov))
    m = tmp_path / "m.jsonl"
    m.write_text("\n".join(json.dumps(r) for r in rows))
    items = runner.items_from_manifest(load_manifest(m, audio_root=tmp_path, verify_sha256=True), tmp_path)
    res = run_dummy(DummyAdapter(), items, adapter_config={"text_en": "the cat sits on the mat"},
                    audible_asr=None, out_dir=str(tmp_path / "out"))
    by = {r.item_id: r for r in res.segment_results}
    assert by["i0"].speaker_group == "spk:corpusX:s1" and by["i0"].cluster == "spk:corpusX:s1"
    assert by["i1"].reference_provenance == "generated" and by["i0"].transcript_provenance == "human"
    pt = res.quality["planned_text"]
    assert pt["human"]["status"] == "ok" and pt["generated"]["status"] == "ok"  # separados
    assert pt["human"]["n_scored"] == 1 and pt["generated"]["n_scored"] == 1
    assert pt["human"]["bleu"] > 90  # texto idéntico a la referencia
    md = report.write_report(tmp_path / "out").read_text()
    assert "## Calidad" in md and "referencias `human`" in md
    # clusters del agregado = hablantes
    q = res.aggregate["groups"][0]["end_to_first_ms"]["censoring_aware"]["quantiles"]["p50"]
    assert q["n_clusters"] == 2 and not q.get("clusters_assumed_independent")


def test_run_with_independent_asr_scores_audible_text_and_flags_omission():
    it = make_item(0)
    it.reference_en = ("one two three four five six",)
    it.segment.reference_provenance = "human"
    res = run_dummy(
        DummyAdapter(), [it],
        adapter_config={"text_en": "one two three four five six"},
        audible_asr=lambda pcm, sr: "one two three",
    )
    q = res.quality
    assert q["planned_text"]["human"]["bleu"] > 90  # lo planeado era correcto
    assert q["audible_text"]["human"]["bleu"] < q["planned_text"]["human"]["bleu"]  # lo audible, no
    assert q["audible_text_report"]["n_omission_suspected"] == 1


def test_streaming_has_no_fake_rtf_and_speed_metrics_are_separated():
    res = run_dummy(DummyAdapter(), [make_item(0)])
    p = res.playout[0]
    assert p["offline_throughput"]["status"] == "unavailable"  # wall/replay no es RTF
    assert p["tts_emission_rtf"]["rtf"] is not None  # separado
    assert p["backlog"]["max_s"] is not None  # separado
    # el dummy emite a ritmo real: RTF de emisión ~1, no cómputo
    assert p["tts_emission_rtf"]["rtf"] == pytest.approx(1.0, abs=0.3)
