"""Texto original del ASR (text_es_raw): migración aditiva, export y errores. Sin audio."""
import json
import sqlite3

import pytest

from runtime.text_store import TextStore, TextStoreError

OLD_SCHEMA = """
CREATE TABLE config (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE sessions (id TEXT PRIMARY KEY, started_at TEXT NOT NULL, ended_at TEXT, config TEXT NOT NULL);
CREATE TABLE segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    seg_id TEXT NOT NULL, text_es TEXT NOT NULL, text_en TEXT NOT NULL, ts TEXT NOT NULL
);
"""


def make_old_db(path):
    c = sqlite3.connect(str(path))
    c.executescript(OLD_SCHEMA)
    c.execute("INSERT INTO sessions VALUES ('s1', 't', NULL, '{}')")
    c.execute("INSERT INTO segments(session_id, seg_id, text_es, text_en, ts) VALUES ('s1','1','hola','hello','t')")
    c.commit()
    c.close()


def test_migration_adds_column_and_old_rows_read_null(tmp_path):
    db = tmp_path / "old.sqlite3"
    make_old_db(db)
    st = TextStore(db)
    try:
        item = st.history(10, 0)["items"][0]
        assert item["text_es"] == "hola" and item["text_es_raw"] is None
        assert json.loads(st.export("json"))[0]["text_es_raw"] is None
        assert "ASR original" not in st.export("txt")  # no se inventa el original
    finally:
        st.close()


def test_migration_is_idempotent_and_keeps_data(tmp_path):
    db = tmp_path / "old.sqlite3"
    make_old_db(db)
    TextStore(db).close()
    st = TextStore(db)  # segunda apertura: no falla ni duplica la columna
    try:
        assert st.history(10, 0)["total"] == 1
        cols = [r["name"] for r in st._conn.execute("PRAGMA table_info(segments)").fetchall()]
        assert cols.count("text_es_raw") == 1
    finally:
        st.close()


def test_new_raw_is_stored_returned_and_exported(tmp_path):
    st = TextStore(tmp_path / "n.sqlite3")
    try:
        sid = st.start_session({})
        st.add_segment(sid, "1", "hola mundo", "hello world", "t", text_es_raw="hola  mundo!")
        item = st.history(10, 0)["items"][0]
        assert item["text_es"] == "hola mundo" and item["text_es_raw"] == "hola  mundo!"
        assert json.loads(st.export("json"))[0]["text_es_raw"] == "hola  mundo!"
        txt = st.export("txt")
        assert "ES: hola mundo" in txt and "ES (ASR original): hola  mundo!" in txt
    finally:
        st.close()


def test_equal_or_missing_raw_is_not_repeated_in_txt(tmp_path):
    st = TextStore(tmp_path / "n.sqlite3")
    try:
        sid = st.start_session({})
        st.add_segment(sid, "1", "hola", "hello", "t", text_es_raw="hola")
        st.add_segment(sid, "2", "adiós", "bye", "t")
        txt = st.export("txt")
        assert "ASR original" not in txt
    finally:
        st.close()


def test_invalid_raw_type_is_rejected(tmp_path):
    st = TextStore(tmp_path / "n.sqlite3")
    try:
        sid = st.start_session({})
        with pytest.raises(TextStoreError):
            st.add_segment(sid, "1", "hola", "hello", "t", text_es_raw=123)
    finally:
        st.close()


def test_update_session_config_merges_without_audio(tmp_path):
    st = TextStore(tmp_path / "n.sqlite3")
    try:
        sid = st.start_session({"voice": "af_heart"})
        st.update_session_config(sid, {"audio": {"latency_mode": "high"}})
        row = st._conn.execute("SELECT config FROM sessions WHERE id = ?", (sid,)).fetchone()
        cfg = json.loads(row["config"])
        assert cfg == {"voice": "af_heart", "audio": {"latency_mode": "high"}}
        with pytest.raises(TextStoreError):
            st.update_session_config("nope", {"x": 1})
    finally:
        st.close()
