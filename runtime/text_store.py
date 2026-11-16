"""Almacén SQLite de textos y configuración. Solo texto y metadatos: nunca audio.

La ruta la pasa el backend. Este módulo no fija ubicaciones en disco.
Todas las consultas son parametrizadas. Las operaciones están protegidas por
un RLock: el hilo que escribe puede ser cualquiera, pero nunca el callback de
audio. Llamar desde el callback está prohibido.

Esquema:
    config(key TEXT PK, value TEXT)            -- JSON por clave
    sessions(id TEXT PK, started_at TEXT, ended_at TEXT NULL, config TEXT)
    segments(id INTEGER PK, session_id TEXT FK, seg_id TEXT,
             text_es TEXT, text_en TEXT, ts TEXT, text_es_raw TEXT NULL)

text_es es el texto que usó la traducción (normalizado si está activado).
text_es_raw es la salida cruda del ASR. Es NULL en registros anteriores a esa
columna: no se reconstruye un original que no se guardó.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_MAX_RECORDS = 10_000
_EXPORT_FORMATS = ("json", "txt")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS config (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    id         TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    ended_at   TEXT,
    config     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS segments (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    seg_id     TEXT NOT NULL,
    text_es    TEXT NOT NULL,
    text_en    TEXT NOT NULL,
    ts         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS segments_session ON segments(session_id);
"""


class TextStoreError(ValueError):
    pass


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class TextStore:
    """Textos originales y traducciones de una sesión de traducción.

    ``max_records`` limita el número total de segmentos guardados. Al superarlo se
    borran los más antiguos, y después las sesiones cerradas que quedan vacías.
    """

    def __init__(self, path: str | Path, max_records: int = DEFAULT_MAX_RECORDS):
        if not isinstance(max_records, int) or max_records < 1:
            raise TextStoreError("max_records debe ser un entero mayor que 0")
        self._path = Path(path)
        self._max = max_records
        self._lock = threading.RLock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._migrate()

    def _migrate(self) -> None:
        """Migración aditiva e idempotente: añade columnas que bases antiguas no tienen."""
        cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(segments)").fetchall()}
        if "text_es_raw" not in cols:
            self._conn.execute("ALTER TABLE segments ADD COLUMN text_es_raw TEXT")

    # ------------------------------------------------------------ utilidades
    def _tx(self):
        """Transacción explícita: BEGIN IMMEDIATE, COMMIT o ROLLBACK."""
        self._require_open()
        return _Tx(self._conn)

    def _require_open(self) -> None:
        if self._conn is None:
            raise TextStoreError("el almacén está cerrado")

    # ---------------------------------------------------------- configuración
    def get_config(self) -> dict[str, Any]:
        with self._lock:
            self._require_open()
            rows = self._conn.execute("SELECT key, value FROM config ORDER BY key").fetchall()
        return {r["key"]: json.loads(r["value"]) for r in rows}

    def set_config(self, values: dict[str, Any]) -> None:
        if not isinstance(values, dict):
            raise TextStoreError("la configuración debe ser un dict")
        with self._lock, self._tx():
            self._require_open()
            for key, value in values.items():
                if not isinstance(key, str) or not key:
                    raise TextStoreError("clave de configuración inválida")
                self._conn.execute(
                    "INSERT INTO config(key, value) VALUES (?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (key, json.dumps(value, ensure_ascii=False)),
                )

    # ----------------------------------------------------------------- sesiones
    def start_session(self, config: dict[str, Any] | None = None) -> str:
        session_id = uuid.uuid4().hex
        snapshot = json.dumps(config or {}, ensure_ascii=False)
        with self._lock, self._tx():
            self._require_open()
            self._conn.execute(
                "INSERT INTO sessions(id, started_at, ended_at, config) VALUES (?, ?, NULL, ?)",
                (session_id, _now_iso(), snapshot),
            )
        return session_id

    def update_session_config(self, session_id: str, values: dict[str, Any]) -> None:
        """Añade o reemplaza claves del snapshot JSON de una sesión. Nunca guarda audio."""
        if not isinstance(values, dict):
            raise TextStoreError("el snapshot debe ser un dict")
        with self._lock, self._tx():
            self._require_open()
            row = self._conn.execute("SELECT config FROM sessions WHERE id = ?", (session_id,)).fetchone()
            if row is None:
                raise TextStoreError(f"sesión desconocida: {session_id}")
            cfg = json.loads(row["config"] or "{}")
            cfg.update(values)
            self._conn.execute("UPDATE sessions SET config = ? WHERE id = ?",
                               (json.dumps(cfg, ensure_ascii=False), session_id))

    def end_session(self, session_id: str) -> None:
        """Cierra la sesión. Si no tiene segmentos, la borra."""
        with self._lock, self._tx():
            self._require_open()
            self._require_session(session_id)
            self._conn.execute(
                "UPDATE sessions SET ended_at = ? WHERE id = ?", (_now_iso(), session_id)
            )
            self._drop_empty_closed_sessions()

    def delete_session(self, session_id: str) -> bool:
        """Borra una sesión cerrada y sus segmentos. Las demás sesiones no cambian.

        Una sesión abierta no se borra: hay que detenerla antes.
        """
        with self._lock, self._tx():
            self._require_open()
            row = self._conn.execute(
                "SELECT ended_at FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if row is None:
                return False
            if row["ended_at"] is None:
                raise TextStoreError("la sesión está abierta; detenla antes de borrarla")
            cur = self._conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            return cur.rowcount > 0

    def _require_session(self, session_id: str) -> None:
        row = self._conn.execute("SELECT 1 FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if row is None:
            raise TextStoreError(f"sesión desconocida: {session_id}")

    def _drop_empty_closed_sessions(self) -> None:
        self._conn.execute(
            "DELETE FROM sessions WHERE ended_at IS NOT NULL "
            "AND NOT EXISTS (SELECT 1 FROM segments s WHERE s.session_id = sessions.id)"
        )

    # ----------------------------------------------------------------- segmentos
    def add_segment(
        self,
        session_id: str,
        seg_id: str,
        text_es: str,
        text_en: str,
        ts: str | None = None,
        text_es_raw: str | None = None,
    ) -> int:
        """Guarda un segmento. Devuelve su id interno. ``text_es_raw`` es opcional."""
        if not isinstance(text_es, str) or not isinstance(text_en, str):
            raise TextStoreError("text_es y text_en deben ser str")
        if text_es_raw is not None and not isinstance(text_es_raw, str):
            raise TextStoreError("text_es_raw debe ser str o None")
        with self._lock, self._tx():
            self._require_open()
            self._require_session(session_id)
            cur = self._conn.execute(
                "INSERT INTO segments(session_id, seg_id, text_es, text_en, ts, text_es_raw) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, str(seg_id), text_es, text_en, ts or _now_iso(), text_es_raw),
            )
            self._enforce_limit()
            return int(cur.lastrowid)

    def _enforce_limit(self) -> None:
        total = self._conn.execute("SELECT COUNT(*) AS n FROM segments").fetchone()["n"]
        excess = total - self._max
        if excess > 0:
            self._conn.execute(
                "DELETE FROM segments WHERE id IN "
                "(SELECT id FROM segments ORDER BY id ASC LIMIT ?)",
                (excess,),
            )
            self._drop_empty_closed_sessions()

    def history(self, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        """Segmentos más recientes primero. Devuelve {items, total}."""
        limit = min(max(int(limit), 1), 500)
        offset = max(int(offset), 0)
        with self._lock:
            self._require_open()
            total = self._conn.execute("SELECT COUNT(*) AS n FROM segments").fetchone()["n"]
            rows = self._conn.execute(
                "SELECT id, session_id, ts, text_es, text_en, text_es_raw FROM segments "
                "ORDER BY id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        items = [
            {
                "id": str(r["id"]),
                "session_id": r["session_id"],
                "ts": r["ts"],
                "text_es": r["text_es"],
                "text_es_raw": r["text_es_raw"],
                "text_en": r["text_en"],
            }
            for r in rows
        ]
        return {"items": items, "total": int(total)}

    def delete_all(self) -> int:
        """Vacía los textos. Conserva la configuración y las sesiones abiertas,
        para que una traducción en curso pueda seguir guardando segmentos.
        Las sesiones cerradas se borran. Devuelve la cantidad de segmentos borrados."""
        with self._lock, self._tx():
            self._require_open()
            n = self._conn.execute("SELECT COUNT(*) AS n FROM segments").fetchone()["n"]
            self._conn.execute("DELETE FROM segments")
            self._conn.execute("DELETE FROM sessions WHERE ended_at IS NOT NULL")
            return int(n)

    def export(self, format: str = "json") -> str:
        if format not in _EXPORT_FORMATS:
            raise TextStoreError(f"formato no soportado: {format}")
        with self._lock:
            self._require_open()
            rows = self._conn.execute(
                "SELECT id, session_id, ts, text_es, text_en, text_es_raw FROM segments ORDER BY id ASC"
            ).fetchall()
        if format == "json":
            data = [
                {
                    "id": str(r["id"]),
                    "session_id": r["session_id"],
                    "ts": r["ts"],
                    "text_es": r["text_es"],
                    "text_es_raw": r["text_es_raw"],
                    "text_en": r["text_en"],
                }
                for r in rows
            ]
            return json.dumps(data, ensure_ascii=False, indent=2)
        lines = []
        for r in rows:
            lines.append(f"[{r['ts']}] ES: {r['text_es']}")
            raw = r["text_es_raw"]
            if raw is not None and raw != r["text_es"]:
                lines.append(f"[{r['ts']}] ES (ASR original): {raw}")
            lines.append(f"[{r['ts']}] EN: {r['text_en']}")
            lines.append("")
        return "\n".join(lines)

    # --------------------------------------------------------------- cierre
    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None


class _Tx:
    """Context manager: BEGIN IMMEDIATE al entrar, COMMIT o ROLLBACK al salir."""

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def __enter__(self):
        self._conn.execute("BEGIN IMMEDIATE")
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self._conn.execute("COMMIT")
        else:
            self._conn.execute("ROLLBACK")
        return False
