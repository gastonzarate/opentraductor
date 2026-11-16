"""Pruebas de runtime/text_store.py: retención, Unicode, borrado aislado,
exportación, reapertura y acceso desde varios hilos."""
from __future__ import annotations

import json
import threading

import pytest

from runtime.text_store import TextStore, TextStoreError


@pytest.fixture
def store(tmp_path):
    s = TextStore(tmp_path / "db" / "textos.sqlite")
    yield s
    s.close()


def test_config_roundtrip_and_overwrite(store):
    assert store.get_config() == {}
    store.set_config({"voice": "en-US-a", "monitor_volume": 0.4})
    store.set_config({"monitor_volume": 0.8, "monitor_id": None})
    assert store.get_config() == {"voice": "en-US-a", "monitor_volume": 0.8, "monitor_id": None}


def test_history_is_newest_first_with_pagination(store):
    sid = store.start_session({"voice": "en-US-a"})
    for i in range(5):
        store.add_segment(sid, f"s{i}", f"es {i}", f"en {i}", ts=f"2026-01-01T00:00:0{i}Z")
    page = store.history(limit=2, offset=0)
    assert page["total"] == 5
    assert [it["text_es"] for it in page["items"]] == ["es 4", "es 3"]
    page2 = store.history(limit=2, offset=2)
    assert [it["text_es"] for it in page2["items"]] == ["es 2", "es 1"]


def test_retention_drops_oldest_segments(tmp_path):
    s = TextStore(tmp_path / "t.sqlite", max_records=3)
    try:
        sid = s.start_session()
        for i in range(7):
            s.add_segment(sid, f"s{i}", f"es {i}", f"en {i}")
        h = s.history(limit=100)
        assert h["total"] == 3
        assert [it["text_es"] for it in h["items"]] == ["es 6", "es 5", "es 4"]
    finally:
        s.close()


def test_retention_removes_closed_sessions_left_empty(tmp_path):
    s = TextStore(tmp_path / "t.sqlite", max_records=2)
    try:
        old = s.start_session()
        s.add_segment(old, "a", "uno", "one")
        s.end_session(old)
        new = s.start_session()
        s.add_segment(new, "b", "dos", "two")
        s.add_segment(new, "c", "tres", "three")  # desplaza el segmento de "old"
        with pytest.raises(TextStoreError):
            s.add_segment(old, "d", "x", "y")  # la sesión vacía se borró
    finally:
        s.close()


def test_unicode_quotes_and_sql_text_are_stored_literally(store):
    sid = store.start_session()
    tricky_es = "¿Dónde está la «ñ»? '); DROP TABLE segments; --"
    tricky_en = 'He said "hi" — ok\u2603 \u0000'.replace("\u0000", "")
    store.add_segment(sid, "x'\"", tricky_es, tricky_en)
    item = store.history(limit=1)["items"][0]
    assert item["text_es"] == tricky_es
    assert item["text_en"] == tricky_en
    assert store.history(limit=1)["total"] == 1  # la tabla sigue existiendo


def test_delete_session_is_isolated(store):
    a = store.start_session()
    b = store.start_session()
    store.add_segment(a, "1", "a1", "A1")
    store.add_segment(b, "2", "b1", "B1")
    store.add_segment(b, "3", "b2", "B2")
    store.end_session(a)
    assert store.delete_session(a) is True
    assert store.delete_session(a) is False
    h = store.history(limit=10)
    assert h["total"] == 2
    assert {it["session_id"] for it in h["items"]} == {b}


def test_delete_all_keeps_config(store):
    store.set_config({"voice": "en-US-a"})
    sid = store.start_session()
    store.add_segment(sid, "1", "hola", "hello")
    assert store.delete_all() == 1
    assert store.history()["total"] == 0
    assert store.get_config() == {"voice": "en-US-a"}


def test_delete_all_during_open_session_keeps_it_writable(store):
    """Caso de captura: borrar textos con la traducción en marcha no debe romperla."""
    sid = store.start_session({"voice": "en-US-a"})
    store.add_segment(sid, "1", "uno", "one")
    store.add_segment(sid, "2", "dos", "two")
    assert store.delete_all() == 2
    assert store.history()["total"] == 0
    store.add_segment(sid, "3", "tres", "three")  # la sesión sigue abierta
    h = store.history()
    assert h["total"] == 1 and h["items"][0]["text_en"] == "three"
    assert h["items"][0]["session_id"] == sid


def test_delete_all_removes_closed_sessions_only(store):
    closed = store.start_session()
    store.add_segment(closed, "1", "a", "A")
    store.end_session(closed)
    open_id = store.start_session()
    store.delete_all()
    with pytest.raises(TextStoreError):
        store.add_segment(closed, "2", "b", "B")  # sesión cerrada se borró
    store.add_segment(open_id, "3", "c", "C")     # sesión abierta sigue


def test_open_session_cannot_be_deleted(store):
    sid = store.start_session()
    store.add_segment(sid, "1", "hola", "hello")
    with pytest.raises(TextStoreError):
        store.delete_session(sid)
    assert store.history()["total"] == 1
    store.end_session(sid)
    assert store.delete_session(sid) is True
    assert store.history()["total"] == 0


def test_export_json_and_txt(store):
    sid = store.start_session()
    store.add_segment(sid, "1", "¿qué tal?", "how are you?", ts="2026-01-01T10:00:00Z")
    data = json.loads(store.export("json"))
    assert data[0]["text_es"] == "¿qué tal?"
    assert data[0]["text_en"] == "how are you?"
    txt = store.export("txt")
    assert "ES: ¿qué tal?" in txt and "EN: how are you?" in txt
    with pytest.raises(TextStoreError):
        store.export("pdf")


def test_reopen_keeps_data(tmp_path):
    path = tmp_path / "t.sqlite"
    s = TextStore(path)
    s.set_config({"voice": "en-US-a"})
    sid = s.start_session()
    s.add_segment(sid, "1", "hola", "hello")
    s.close()

    s2 = TextStore(path)
    try:
        assert s2.get_config() == {"voice": "en-US-a"}
        assert s2.history()["items"][0]["text_en"] == "hello"
    finally:
        s2.close()


def test_unknown_session_and_closed_store(store, tmp_path):
    with pytest.raises(TextStoreError):
        store.add_segment("no-existe", "1", "a", "b")
    with pytest.raises(TextStoreError):
        store.end_session("no-existe")
    store.close()
    with pytest.raises(TextStoreError):
        store.history()


def test_invalid_max_records_rejected(tmp_path):
    with pytest.raises(TextStoreError):
        TextStore(tmp_path / "t.sqlite", max_records=0)


def test_concurrent_writers_do_not_lose_rows(store):
    sid = store.start_session()
    errors = []

    def writer(tag: int) -> None:
        try:
            for i in range(50):
                store.add_segment(sid, f"{tag}-{i}", f"es {tag} {i}", f"en {tag} {i}")
        except Exception as exc:  # pragma: no cover - el fallo se reporta abajo
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(t,)) for t in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert store.history(limit=1)["total"] == 200
