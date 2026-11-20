"""CLI: puerto ocupado sale con error humano y cierra el servicio."""
import socket

import pytest

from runtime import __main__ as cli


def test_default_port_is_8877():
    import inspect

    from runtime.http_control import ControlServer

    assert inspect.signature(ControlServer).parameters["port"].default == 8877


def test_busy_port_exits_cleanly(tmp_path, capsys):
    busy = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    busy.bind(("127.0.0.1", 0))
    busy.listen(1)
    port = busy.getsockname()[1]
    try:
        rc = cli.main(["run", "--port", str(port), "--db", str(tmp_path / "h.sqlite3")])
    finally:
        busy.close()
    assert rc == 1
    err = capsys.readouterr().err
    assert "No se pudo abrir el puerto" in err and "Traceback" not in err


def _args(*extra, db):
    return cli.build_parser().parse_args(["run", "--db", str(db), *extra])


def test_defaults_come_from_pipeline_config(tmp_path):
    # T-58: sin flags rige el brazo store-tuned (defaults de PipelineConfig)
    a = _args(db=tmp_path / "h.sqlite3")
    assert a.asr_backend == "parakeet"
    assert a.skip_silent_tail is None and a.tts_trim_lead_ms is None and a.endpoint_ms is None
    svc, cfg = cli.build_service(a)
    try:
        assert cfg.asr_backend == "parakeet" and cfg.asr_skip_silent_tail is True
        assert cfg.tts_trim_lead_margin_ms == 30.0 and cfg.endpoint_silence_ms == 160.0
        assert cfg.tts_first_chunk_max_words == 2 and cfg.tts_bind_conjunctions is True
    finally:
        svc.close()


def test_tuned_defaults_can_be_turned_off(tmp_path):
    a = _args("--no-skip-silent-tail", "--tts-trim-lead-ms", "off", "--tts-first-chunk-max-words", "off",
              "--no-tts-bind-conjunctions", db=tmp_path / "h.sqlite3")
    svc, cfg = cli.build_service(a)
    try:
        assert cfg.asr_skip_silent_tail is False and cfg.tts_trim_lead_margin_ms is None
        assert cfg.tts_first_chunk_max_words is None and cfg.tts_bind_conjunctions is False
    finally:
        svc.close()


def test_options_reach_pipeline_config(tmp_path):
    a = _args("--asr-backend", "moonshine-tiny", "--skip-silent-tail", "--tts-trim-lead-ms", "80",
              db=tmp_path / "h.sqlite3")
    svc, cfg = cli.build_service(a)
    try:
        assert cfg.asr_backend == "moonshine-tiny"
        assert cfg.asr_skip_silent_tail is True
        assert cfg.tts_trim_lead_margin_ms == 80.0
    finally:
        svc.close()


def test_negative_trim_is_rejected(tmp_path, capsys):
    with pytest.raises(SystemExit):
        _args("--tts-trim-lead-ms", "-5", db=tmp_path / "h.sqlite3")


def test_unknown_backend_is_rejected(tmp_path):
    with pytest.raises(SystemExit):
        _args("--asr-backend", "whisper", db=tmp_path / "h.sqlite3")


def test_explicit_endpoint_beats_saved_value(tmp_path):
    from runtime.service import RuntimeService

    db = tmp_path / "h.sqlite3"
    s0 = RuntimeService(db)
    s0.configure({"endpoint_silence_ms": 500})  # valor guardado en SQLite
    s0.close()

    # sin flag: rige el valor guardado
    svc, _ = cli.build_service(_args(db=db))
    try:
        assert svc.status()["config"]["endpoint_silence_ms"] == 500.0
    finally:
        svc.close()

    # con flag: prevalece la línea de comandos
    svc, _ = cli.build_service(_args("--endpoint-ms", "150", db=db))
    try:
        st = svc.status()
        assert st["config"]["endpoint_silence_ms"] == 150.0
        assert st["engine"]["endpoint_silence_ms"] == 150.0
    finally:
        svc.close()


def test_engine_config_is_in_status_and_session_snapshot(tmp_path):
    from runtime.service import RuntimeService
    from runtime.config import PipelineConfig
    import json

    db = tmp_path / "h.sqlite3"
    cfg = PipelineConfig(asr_backend="moonshine-small", asr_skip_silent_tail=True, tts_trim_lead_margin_ms=60.0)
    s = RuntimeService(db, pipeline_cfg=cfg, audio=None)
    try:
        eng = s.status()["engine"]
        assert eng["asr_backend"] == "moonshine-small"
        assert eng["asr_skip_silent_tail"] is True
        assert eng["tts_trim_lead_margin_ms"] == 60.0
        # el snapshot de sesión guarda la configuración del motor, sin audio
        sid = s._store.start_session({"x": 1, "engine": s.engine_config()})
        row = s._store._conn.execute("SELECT config FROM sessions WHERE id = ?", (sid,)).fetchone()
        assert json.loads(row["config"])["engine"]["asr_backend"] == "moonshine-small"
    finally:
        s.close()


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf", "-1", "abc"])
def test_trim_rejects_non_finite_and_negative(tmp_path, bad, monkeypatch):
    monkeypatch.setattr(cli, "build_service", lambda a: pytest.fail("no debe crear servicio"))
    with pytest.raises(SystemExit):
        cli.main(["run", "--db", str(tmp_path / "h.sqlite3"), "--tts-trim-lead-ms", bad])


@pytest.mark.parametrize("bad", ["nan", "inf", "59.9", "2000.1", "x"])
def test_endpoint_rejects_non_finite_and_out_of_range(tmp_path, bad, monkeypatch):
    monkeypatch.setattr(cli, "build_service", lambda a: pytest.fail("no debe crear servicio"))
    with pytest.raises(SystemExit):
        cli.main(["run", "--db", str(tmp_path / "h.sqlite3"), "--endpoint-ms", bad])


def test_endpoint_accepts_api_range_bounds(tmp_path):
    assert _args("--endpoint-ms", "60", db=tmp_path / "h.sqlite3").endpoint_ms == 60.0
    assert _args("--endpoint-ms", "2000", db=tmp_path / "h.sqlite3").endpoint_ms == 2000.0
    assert _args("--tts-trim-lead-ms", "0", db=tmp_path / "h.sqlite3").tts_trim_lead_ms == 0.0


def test_new_flags_default_and_reach_config(tmp_path):
    a = _args(db=tmp_path / "h.sqlite3")
    assert a.tts_first_chunk_max_words is None and a.normalize_asr_text is False
    svc, cfg = cli.build_service(a)
    try:
        assert cfg.tts_first_chunk_max_words == 2 and cfg.asr_normalize_text is False
    finally:
        svc.close()
    a = _args("--tts-first-chunk-max-words", "5", "--normalize-asr-text", db=tmp_path / "h.sqlite3")
    svc, cfg = cli.build_service(a)
    try:
        assert cfg.tts_first_chunk_max_words == 5 and cfg.asr_normalize_text is True
    finally:
        svc.close()


@pytest.mark.parametrize("bad", ["0", "-3", "2.5", "nan", "inf", "abc", ""])
def test_first_chunk_words_rejects_invalid(tmp_path, bad, monkeypatch):
    monkeypatch.setattr(cli, "build_service", lambda a: pytest.fail("no debe crear servicio"))
    with pytest.raises(SystemExit):
        cli.main(["run", "--db", str(tmp_path / "h.sqlite3"), "--tts-first-chunk-max-words", bad])


def test_audio_latency_and_bind_conjunctions_defaults_and_flow(tmp_path):
    a = _args(db=tmp_path / "h.sqlite3")
    assert a.audio_latency is None and a.tts_bind_conjunctions is None  # sin flag: rige guardado o default
    svc, cfg = cli.build_service(a)
    try:
        assert cfg.tts_bind_conjunctions is True and svc._audio_latency == "default"
    finally:
        svc.close()
    a = _args("--audio-latency", "low", "--tts-bind-conjunctions", db=tmp_path / "h.sqlite3")
    svc, cfg = cli.build_service(a)
    try:
        assert cfg.tts_bind_conjunctions is True and svc._audio_latency == "low"
    finally:
        svc.close()


@pytest.mark.parametrize("bad", ["turbo", "LOW", ""])
def test_audio_latency_rejects_unknown_modes(tmp_path, bad, monkeypatch):
    monkeypatch.setattr(cli, "build_service", lambda a: pytest.fail("no debe crear servicio"))
    with pytest.raises(SystemExit):
        cli.main(["run", "--db", str(tmp_path / "h.sqlite3"), "--audio-latency", bad])


def test_mt_engine_defaults_to_marian_and_env_or_flag_selects_qwen(tmp_path, monkeypatch):
    """T-51: la .app elige Qwen con OPENTRADUCTOR_MT_ENGINE=qwen (+ OPENTRADUCTOR_QWEN_DIR); el flag manda."""
    monkeypatch.delenv("OPENTRADUCTOR_MT_ENGINE", raising=False)
    svc, cfg = cli.build_service(_args(db=tmp_path / "a.sqlite3"))
    svc.close()
    assert cfg.mt_engine == "marian" and cfg.qwen_model_dir is None

    monkeypatch.setenv("OPENTRADUCTOR_MT_ENGINE", "qwen")
    svc, cfg = cli.build_service(_args(db=tmp_path / "b.sqlite3"))
    svc.close()
    assert cfg.mt_engine == "qwen"

    svc, cfg = cli.build_service(_args("--mt-engine", "marian", "--qwen-dir", "/x", db=tmp_path / "c.sqlite3"))
    svc.close()
    assert cfg.mt_engine == "marian" and cfg.qwen_model_dir == "/x"

    monkeypatch.setenv("OPENTRADUCTOR_MT_ENGINE", "gpt")
    with pytest.raises(SystemExit):
        cli.build_service(_args(db=tmp_path / "d.sqlite3"))


def test_check_verifies_the_selected_mt_engine(monkeypatch, capsys):
    """T-51: `check` arma el PipelineConfig con mt_engine; si no, el preflight de la .app con Qwen verificaba Marian."""
    import runtime.preflight as pf

    seen = []

    def fake_run_checks(cfg):
        seen.append((cfg.mt_engine, cfg.qwen_model_dir, cfg.asr_backend))
        return []

    monkeypatch.setattr(pf, "run_checks", fake_run_checks)
    monkeypatch.delenv("OPENTRADUCTOR_MT_ENGINE", raising=False)
    p = cli.build_parser()
    a = p.parse_args(["check", "--asr-backend", "moonshine-tiny"])
    assert a.fn(a) == 0
    monkeypatch.setenv("OPENTRADUCTOR_MT_ENGINE", "qwen")
    a = p.parse_args(["check", "--asr-backend", "moonshine-tiny"])
    assert a.fn(a) == 0
    a = p.parse_args(["check", "--mt-engine", "marian", "--qwen-dir", "/q"])
    assert a.fn(a) == 0
    assert seen[0][:2] == ("marian", None)
    assert seen[1] == ("qwen", None, "moonshine-tiny")
    assert seen[2][:2] == ("marian", "/q")
