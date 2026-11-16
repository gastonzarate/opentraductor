"""Exportar y borrar textos incrementales (inc_lines/inc_units) desde el servicio. Sin modelos ni audio."""
import json, sqlite3
from types import SimpleNamespace
import pytest
from runtime.http_control import ServiceError
from runtime.inc_app.bridge import SqliteUnitStore
from runtime.inc_history import IncHistory, default_path, session_key
from runtime.service import RuntimeService

SID = "0123456789abcdef"            # id de sesion del servicio
KEY = session_key(SID)               # live-01234567


def make_inc(path, extra_session=True):
    s = SqliteUnitStore(str(path))
    s.record_line({"session": KEY, "line_id": "L1", "text_es": "hola sin traducir", "ts": "2026-01-01T00:00:01"})
    s.record_line({"session": KEY, "line_id": "L2", "text_es": "otra linea", "ts": "2026-01-01T00:00:02"})
    s.record_unit({"session": KEY, "unit_id": "u1", "text_en": "hello", "state": "delivered", "n_chunks": 2, "total_samples": 100, "ts": "2026-01-01T00:00:03"})
    s.record_unit({"session": KEY, "unit_id": "u2", "text_en": "partial", "state": "interrupted", "n_chunks": 1, "total_samples": 10, "reason": "stream_end", "ts": "2026-01-01T00:00:04"})
    if extra_session:
        s.record_line({"session": "live-other000", "line_id": "L9", "text_es": "otra sesion", "ts": "t"})
        s.record_unit({"session": "live-other000", "unit_id": "u9", "text_en": "other", "state": "delivered", "n_chunks": 1, "total_samples": 1, "ts": "t"})
    s.close()


@pytest.fixture
def svc(tmp_path):
    s = RuntimeService(tmp_path / "history.sqlite3", preflight=lambda c: [SimpleNamespace(name="x", ok=True, detail="ok")])
    yield s, tmp_path
    s.close()


def test_no_db_created_by_reading_or_deleting_when_absent(svc):
    s, tmp = svc
    assert s.export_history("json") == "[]" and s.delete_history(None) == 0 and s.delete_history("abc") == 0
    assert not default_path(tmp / "history.sqlite3").exists()


def test_export_in_baseline_includes_es_lines_without_translation_and_en_states(svc):
    s, tmp = svc                                                       # servicio en BASELINE reabierto con base incremental previa
    make_inc(default_path(tmp / "history.sqlite3"))
    data = json.loads(s.export_history("json"))
    lines = [d for d in data if d["kind"] == "inc_line_es"]
    units = [d for d in data if d["kind"] == "inc_unit_en"]
    assert {d["text_es"] for d in lines} == {"hola sin traducir", "otra linea", "otra sesion"}      # ES sin traduccion incluido
    assert {(d["id"], d["state"], d["playback"]) for d in units} == {("u1", "delivered", "not_verified"), ("u2", "interrupted", "not_verified"), ("u9", "delivered", "not_verified")}
    assert all("text_en" not in d for d in lines) and all("text_es" not in d for d in units)       # sin alineaciones inventadas
    txt = s.export_history("txt")
    assert "ES (línea" in txt and "hola sin traducir" in txt and "estado=interrupted" in txt and "reproducción no verificada" in txt


def test_delete_all_removes_both_incremental_tables_and_baseline(svc):
    s, tmp = svc
    make_inc(default_path(tmp / "history.sqlite3"))
    n = s.delete_history(None)
    assert n == 6                                                        # 3 lineas + 3 unidades (+0 segmentos baseline)
    assert IncHistory(default_path(tmp / "history.sqlite3")).read() == {"lines": [], "units": []}
    assert s.export_history("json") == "[]"


def real_session(s):
    sid = s._store.start_session({})
    s._store.add_segment(sid, "1", "hola", "hello")
    s._store.end_session(sid)
    return sid


def test_delete_by_session_only_that_session_legacy_known_unique(svc):
    s, tmp = svc
    p = default_path(tmp / "history.sqlite3")
    sid = real_session(s)
    st = SqliteUnitStore(str(p)); key = session_key(sid)
    st.record_line({"session": key, "line_id": "L1", "text_es": "uno", "ts": "t"})
    st.record_line({"session": "live-other000", "line_id": "L9", "text_es": "otra", "ts": "t"}); st.close()
    assert s.delete_history(sid) == 1
    assert [r["line_id"] for r in IncHistory(p).read()["lines"]] == ["L9"]


def test_unknown_id_with_same_prefix_never_deletes_another_session(svc):
    s, tmp = svc
    p = default_path(tmp / "history.sqlite3")
    sid = real_session(s)
    st = SqliteUnitStore(str(p)); st.record_line({"session": session_key(sid), "line_id": "L1", "text_es": "mio", "ts": "t"}); st.close()
    assert s.delete_history(sid[:8] + "zzzzzzzzzzzz") == 0                  # id inexistente con mismos 8 primeros
    assert len(IncHistory(p).read()["lines"]) == 1 and s.history(10, 0)["total"] == 1


def test_special_characters_in_path_are_not_uri_syntax(tmp_path):
    d = tmp_path / "a#b?c%d"; d.mkdir()
    p = d / "incremental_units.sqlite3"; make_inc(p)
    h = IncHistory(p)
    assert len(h.read()["lines"]) == 3 and h.delete(None) == 6 and h.read()["lines"] == []


def test_delete_all_holds_service_lock_so_start_cannot_interleave(svc):
    import threading, time
    s, tmp = svc
    make_inc(default_path(tmp / "history.sqlite3"))
    inside, release, done = threading.Event(), threading.Event(), {}
    real = s._inc.delete
    def slow(session=None):
        inside.set(); release.wait(3); return real(session)
    s._inc.delete = slow
    t = threading.Thread(target=lambda: done.update(n=s.delete_history(None))); t.start()
    assert inside.wait(3)
    def starter():                                                       # como start(): toma el lock del servicio para pasar a 'starting'
        with s._lock: done["start_at"] = time.monotonic()
    t2 = threading.Thread(target=starter); t2.start()
    time.sleep(0.2); assert "start_at" not in done                      # start espera: el borrado tiene el lock
    done["release"] = time.monotonic(); release.set(); t.join(3); t2.join(3)
    assert done["n"] == 6 and done["start_at"] >= done["release"]


def test_delete_rejected_while_incremental_session_active(tmp_path):
    p = default_path(tmp_path / "history.sqlite3"); make_inc(p)
    s = RuntimeService(tmp_path / "history.sqlite3", engine_mode={"mode": "incremental_experimental", "ignored_controls": []},
                       preflight=lambda c: [])
    try:
        s._session_id = SID                                              # sesion activa
        with pytest.raises(ServiceError) as e: s.delete_history(SID)    # por sesion: la activa se rechaza
        assert len(IncHistory(p).read()["lines"]) == 3
        with pytest.raises(ServiceError) as e: s.delete_history(None)
        assert e.value.status == 409 if hasattr(e.value, "status") else True
        assert len(IncHistory(p).read()["lines"]) == 3                   # nada se borro
        s._session_id = None
        assert s.delete_history(None) >= 6
    finally:
        s._session_id = None; s.close()


def test_incremental_db_failure_not_reported_as_complete(svc):
    s, tmp = svc
    p = default_path(tmp / "history.sqlite3")
    p.write_bytes(b"esto no es sqlite" * 100)                            # base corrupta
    with pytest.raises(ServiceError) as e: s.export_history("json")
    assert "Exportación no generada" in str(e.value)
    with pytest.raises(ServiceError) as e: s.delete_history(None)
    assert "No se borró nada" in str(e.value)


def test_partial_delete_is_reported_when_baseline_fails(svc, monkeypatch):
    s, tmp = svc
    p = default_path(tmp / "history.sqlite3"); make_inc(p)
    from runtime.text_store import TextStoreError
    monkeypatch.setattr(s._store, "delete_all", lambda: (_ for _ in ()).throw(TextStoreError("disco lleno")))
    with pytest.raises(ServiceError) as e: s.delete_history(None)
    assert "PARCIAL" in str(e.value) and "incrementales se borraron" in str(e.value)
    assert IncHistory(p).read() == {"lines": [], "units": []}


def test_delete_is_atomic_across_both_tables(svc):
    s, tmp = svc
    p = default_path(tmp / "history.sqlite3"); make_inc(p)
    con = sqlite3.connect(p); con.execute("CREATE TRIGGER boom BEFORE DELETE ON inc_units BEGIN SELECT RAISE(ABORT,'no'); END"); con.commit(); con.close()
    with pytest.raises(ServiceError): s.delete_history(None)
    r = IncHistory(p).read()
    assert len(r["lines"]) == 3 and len(r["units"]) == 3                 # inc_lines NO quedo borrada sola (todo o nada)


# ---- mapping explicito stream -> sesion del historial ----
def test_mapped_session_delete_uses_mapping_not_prefix(svc):
    s, tmp = svc
    p = default_path(tmp / "history.sqlite3"); make_inc(p, extra_session=False)   # filas bajo live-01234567
    other = "0123456799999999"                                                      # mismo prefijo de 8 que SID (colision)
    IncHistory(p).map_session("live-01234567", other)                              # el mapping dice que esas filas son de OTRA sesion
    s.delete_history(SID)                                                           # SID no tiene mapping ni legado valido (clave mapeada a otra)
    assert len(IncHistory(p).read()["lines"]) == 2                                  # nada borrado por prefijo
    assert s.delete_history(other) == 1
    r = IncHistory(p).read()
    assert r["lines"] == [] and r["units"] == [] and not IncHistory(p).resolve(other)["mapped"]


def test_export_reports_history_session_only_when_mapped(svc):
    s, tmp = svc
    p = default_path(tmp / "history.sqlite3"); make_inc(p)
    IncHistory(p).map_session(KEY, SID)
    data = json.loads(s.export_history("json"))
    by = {d["session"]: d["history_session_id"] for d in data if d["kind"].startswith("inc_")}
    assert by == {KEY: SID, "live-other000": None}                                  # legado: None, no se adivina


def test_legacy_ambiguity_blocks_session_delete_without_deleting(svc, monkeypatch):
    s, tmp = svc
    p = default_path(tmp / "history.sqlite3"); make_inc(p, extra_session=False)    # legado, sin mapping
    monkeypatch.setattr(s._store, "export", lambda f="json": json.dumps([{"session_id": SID}, {"session_id": SID[:8] + "ffffffff"}]))
    with pytest.raises(ServiceError) as e: s.delete_history(SID)
    assert "ambigua" in str(e.value) and len(IncHistory(p).read()["lines"]) == 2


def test_map_session_conflict_and_missing_db_are_errors(svc):
    s, tmp = svc
    p = default_path(tmp / "history.sqlite3")
    from runtime.inc_history import IncHistoryError
    with pytest.raises(IncHistoryError): IncHistory(p).map_session("live-x", "a")   # no crea la base
    assert not p.exists()
    make_inc(p, extra_session=False)
    IncHistory(p).map_session("live-x", "a"); IncHistory(p).map_session("live-x", "a")   # idempotente
    with pytest.raises(IncHistoryError, match="colision"): IncHistory(p).map_session("live-x", "b")


def test_delete_all_also_clears_mapping(svc):
    s, tmp = svc
    p = default_path(tmp / "history.sqlite3"); make_inc(p); IncHistory(p).map_session(KEY, SID)
    s.delete_history(None)
    c = sqlite3.connect(p); assert c.execute("SELECT COUNT(*) FROM inc_sessions").fetchone()[0] == 0; c.close()


def test_service_start_registers_mapping(tmp_path):
    import os, sys, time
    sys.path.insert(0, os.path.dirname(__file__))
    from test_service import FakeAudio, FakeEngine, wait
    p = default_path(tmp_path / "history.sqlite3")
    SqliteUnitStore(str(p)).close()                                                 # el almacen del puente ya creo la base
    s = RuntimeService(tmp_path / "history.sqlite3", engine_factory=FakeEngine, audio=FakeAudio, engine_mode={"mode": "incremental_experimental", "ignored_controls": []},
                       preflight=lambda c: [SimpleNamespace(name="x", ok=True, detail="ok")])
    s.voices = lambda: {"voices": [{"id": "af_heart", "label": "af_heart"}]}
    try:
        s.configure({"input_id": "mic", "output_id": "spk"}); s.start()
        assert wait(lambda: s.status()["state"] == "running")
        c = sqlite3.connect(p); rows = c.execute("SELECT stream_id, history_session_id FROM inc_sessions").fetchall(); c.close()
        assert len(rows) == 1 and rows[0][0] == f"live-{rows[0][1][:8]}"
        s.stop(); assert wait(lambda: s.status()["state"] == "idle")
    finally:
        s.close()
