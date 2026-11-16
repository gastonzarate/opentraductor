"""Seleccion de modo baseline/incremental experimental. Sin modelos, micro ni servicio de audio."""
import asyncio, hashlib, importlib, textwrap
import pytest
from runtime import __main__ as M
from runtime.http_control import ServiceError


def parse(*argv):
    return M.build_parser().parse_args(["run", *argv])


def test_baseline_is_default_and_unchanged(tmp_path):
    a = parse("--db", str(tmp_path / "h.db"))
    assert a.engine_mode == "baseline"
    svc, cfg = M.build_service(a)
    try:
        assert svc._engine_factory == svc._default_engine_factory
        st = svc.status()
        assert st["engine_mode"]["mode"] == "baseline" and st["engine_mode"]["ignored_controls"] == []
        assert "asr_backend" in st["engine"]
        svc.configure({"endpoint_silence_ms": 300.0})
    finally:
        svc.close()


def test_no_path_flags_and_baseline_only_flags_rejected(capsys):
    with pytest.raises(SystemExit):
        M.build_parser().parse_args(["run", "--inc-qwen-dir", "/otro"])          # rutas arbitrarias: no existen como flag
    with pytest.raises(SystemExit):
        M.main(["run", "--engine-mode", "incremental_experimental", "--skip-silent-tail", "--endpoint-ms", "200"])
    err = capsys.readouterr().err
    assert "--skip-silent-tail" in err and "--endpoint-ms" in err


def manifest(root, subdir):
    lines = [f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(root)}" for f in sorted((root / subdir).rglob("*.py"))]
    (root / "HASHES.sha256").write_text("\n".join(lines) + "\n")
    return hashlib.sha256((root / "HASHES.sha256").read_bytes()).hexdigest()


@pytest.fixture
def inc_env(tmp_path, monkeypatch):
    from runtime import inc_mode
    v = tmp_path / "variant" / "inc_llm"; v.mkdir(parents=True)
    (v / "__init__.py").write_text("")
    (v / "adapter.py").write_text(textwrap.dedent('''
        import asyncio
        class IncrementalLLMAdapter:
            async def open(self, cfg): self.cfg = cfg; self.q = asyncio.Queue()
            async def start_stream(self, sid, t): pass
            async def push_audio(self, sid, pcm, t): pass
            async def end_stream(self, sid): pass
            async def events(self):
                while True:
                    ev = await self.q.get()
                    if ev is None: return
                    yield ev
            async def close(self): await self.q.put(None)
    '''))
    b = tmp_path / "base" / "inc_adapter"; b.mkdir(parents=True); (b / "__init__.py").write_text("x=1")
    q = tmp_path / "qwen"; q.mkdir(); (q / "config.json").write_text("{}"); (q / "model.safetensors").write_bytes(b"x")
    mm = tmp_path / "model_manifest.json"; mm.write_text("{}")
    pins = {"v": manifest(tmp_path / "variant", "inc_llm"), "b": manifest(tmp_path / "base", "inc_adapter"),
            "m": hashlib.sha256(b"{}").hexdigest()}
    monkeypatch.setattr(inc_mode, "_has_module", lambda n: True)
    monkeypatch.setattr(inc_mode, "_venv_ok", lambda: True)
    monkeypatch.setattr(inc_mode, "_baseline_checks", lambda: [])
    opts = inc_mode.IncOptions(tmp_path / "variant", tmp_path / "base", q, mm, tmp_path / "u.db")
    return inc_mode, opts, pins, monkeypatch


def pin(env, **over):
    m, o, pins, mp = env
    mp.setattr(m, "APPROVED_VARIANT_SHA256", over.get("v", pins["v"]))
    mp.setattr(m, "APPROVED_BASE_SHA256", over.get("b", pins["b"]))
    mp.setattr(m, "APPROVED_MODEL_MANIFEST_SHA256", over.get("m", pins["m"]))


def service(env, tmp_path):
    from runtime.service import RuntimeService
    m, o, *_ = env
    f, info, pre = m.build(o)
    return RuntimeService(tmp_path / "h.db", engine_factory=f, engine_mode=info, preflight=pre)


def names(svc):
    return {c["name"]: c["ok"] for c in svc.status()["preflight"]}


def test_real_constants_and_fixed_paths():
    from runtime import inc_mode as m
    m = importlib.reload(m)
    assert m.APPROVED_VARIANT_SHA256 == "edf536e0dd60f877a2a928474870d3a15b75f4928c00e44000248d0be99b1a0b"
    assert m.APPROVED_BASE_SHA256 == "d5fc4ae710e287de40170f22ad466288d54fffc6cbf73aacaa98159fec00ff83"
    assert m.APPROVED_MODEL_MANIFEST_SHA256 == "a4a9d6ca046f468b5067c2d2acaee1492f5d9a7b2d7de8d18a3e43acab2ce296"
    o = m.IncOptions()
    assert o.variant_dir.name == "incremental-llm-v1" and o.base_dir.name == "incremental-adapter-v1"
    assert o.qwen_dir.name == "qwen3-4b-instruct-2507-4bit" and o.model_manifest.name == "model_manifest.json"


def test_incremental_selected_applied_ignored_controls_and_head_cache(tmp_path, inc_env):
    pin(inc_env)
    svc = service(inc_env, tmp_path)
    try:
        st = svc.status()
        assert all(c["ok"] for c in st["preflight"]) and "inc:venv-inc-llm" in names(svc)
        em = st["engine_mode"]
        assert em["mode"] == "incremental_experimental" and em["label"] == "Traducción simultánea · Experimental" and em["no_300ms_claim"]
        assert em["playback"] == "not_verified" and "relacion unidad-linea" in em["detail"] and len(em["label"]) < 50
        assert em["applied"]["model_dir"].endswith("qwen") and "head_cache=False" in em["applied"]["mt"]
        assert st["engine"]["mode"] == "incremental_experimental" and "asr_backend" not in st["engine"]
        for k, val in (("voice", "af_heart"), ("endpoint_silence_ms", 300.0)):
            with pytest.raises(ServiceError): svc.configure({k: val})
        svc.configure({"input_id": "x", "output_id": "y", "monitor_volume": 0.5})
        async def go():
            eng = svc._engine_factory(svc._pcfg, lambda *a: asyncio.sleep(0))
            assert type(eng).__name__ == "IncrementalEngineBridge" and eng.mode.llm_mt == {"head_cache": False}
            await eng.open(); assert eng.adapter.cfg == {"llm_mt": {"head_cache": False}}; await eng.close()
        asyncio.run(go())
    finally:
        svc.close()


@pytest.mark.parametrize("which", ["v", "b", "m"])
def test_wrong_or_missing_pin_blocks(tmp_path, inc_env, which):
    pin(inc_env, **{which: None})
    svc = service(inc_env, tmp_path)
    try:
        assert not all(names(svc).values())
        with pytest.raises(Exception):
            asyncio.run(asyncio.wait_for(svc._engine_factory(svc._pcfg, lambda *a: None).open(), 2))
    finally:
        svc.close()


def test_modified_files_extra_files_wrong_env_missing_weights(tmp_path, inc_env):
    m, o, pins, mp = inc_env
    pin(inc_env)
    assert all(c.ok for c in m.checks(o))
    mp.setattr(m, "_venv_ok", lambda: False)
    assert not next(c for c in m.checks(o) if c.name == "inc:venv-inc-llm").ok          # .venv-llm-probe u otra no sirve
    mp.setattr(m, "_venv_ok", lambda: True)
    (o.qwen_dir / "model.safetensors").unlink()
    assert not next(c for c in m.checks(o) if c.name == "inc:qwen-model").ok
    (o.qwen_dir / "model.safetensors").write_bytes(b"x")
    o.model_manifest.write_text("{ }")                                                    # manifiesto del modelo cambiado
    assert not next(c for c in m.checks(o) if c.name == "inc:qwen-model").ok
    (o.variant_dir / "inc_llm" / "adapter.py").write_text("# cambiado")
    assert not next(c for c in m.checks(o) if c.name == "inc:variant").ok
    (o.base_dir / "inc_adapter" / "extra.py").write_text("x=1")
    assert not next(c for c in m.checks(o) if c.name == "inc:base-engine").ok
