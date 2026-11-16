"""Lectura y borrado de los textos del modo incremental (tablas inc_lines e inc_units) con ruta EXPLICITA.

Sin imports de modelos. Nunca crea la base: si el archivo no existe, leer devuelve vacio y borrar devuelve 0. Funciona igual en modo
baseline si quedo una base incremental de una sesion anterior. El español (lineas) y el ingles (unidades, con estado) se exportan por
separado: no se inventa ninguna alineacion entre ellos.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

TABLES = ("inc_lines", "inc_units", "inc_sessions")  # inc_sessions: mapping explicito stream -> sesion del historial
INC_DB_NAME = "incremental_units.sqlite3"


class IncHistoryError(Exception):
    pass


def default_path(history_db: str | Path) -> Path:
    return Path(history_db).with_name(INC_DB_NAME)


def legacy_key(session_id: str) -> str:
    """Formato LEGADO del stream_id (``live-<8 primeros del id>``). Solo para datos previos al mapping; el prefijo puede ser ambiguo."""
    return f"live-{session_id[:8]}"


session_key = legacy_key  # alias


class IncHistory:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _connect(self, mode: str) -> sqlite3.Connection:
        uri = f"{self.path.resolve().as_uri()}?mode={mode}"  # as_uri codifica '#', '?' y '%' del nombre: no se interpretan
        conn = sqlite3.connect(uri, uri=True, timeout=5.0)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _tables(conn) -> set[str]:
        return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")} & set(TABLES)

    def read(self) -> dict:
        if not self.path.is_file():
            return {"lines": [], "units": []}
        try:
            conn = self._connect("ro")
            try:
                have = self._tables(conn)
                lines = [dict(r) for r in conn.execute("SELECT * FROM inc_lines ORDER BY rowid")] if "inc_lines" in have else []
                units = [dict(r) for r in conn.execute("SELECT * FROM inc_units ORDER BY rowid")] if "inc_units" in have else []
                maps = ({r["stream_id"]: r["history_session_id"] for r in conn.execute("SELECT * FROM inc_sessions")}
                        if "inc_sessions" in have else {})
            finally:
                conn.close()
        except sqlite3.Error as exc:
            raise IncHistoryError(f"{type(exc).__name__}: {exc}") from exc
        for r in lines + units:  # sesion del historial habitual si hay mapping explicito; None = dato legado (no se adivina)
            r["history_session_id"] = maps.get(r["session"])
        return {"lines": lines, "units": units}

    def map_session(self, stream_id: str, history_session_id: str) -> None:
        """Registra el mapping explicito de una sesion nueva. NO crea la base: debe existir (la crea el almacen del puente)."""
        if not self.path.is_file():
            raise IncHistoryError("la base incremental no existe todavia: mapping no registrado")
        try:
            conn = self._connect("rw")
            try:
                with conn:
                    conn.execute("CREATE TABLE IF NOT EXISTS inc_sessions(stream_id TEXT PRIMARY KEY, history_session_id TEXT NOT NULL)")
                    row = conn.execute("SELECT history_session_id FROM inc_sessions WHERE stream_id=?", (stream_id,)).fetchone()
                    if row is not None and row[0] != history_session_id:
                        raise IncHistoryError(f"{stream_id} ya esta mapeado a otra sesion del historial (colision del prefijo del stream_id)")
                    conn.execute("INSERT OR IGNORE INTO inc_sessions VALUES(?,?)", (stream_id, history_session_id))
            finally:
                conn.close()
        except sqlite3.Error as exc:
            raise IncHistoryError(f"{type(exc).__name__}: {exc}") from exc

    def resolve(self, history_session_id: str) -> dict:
        """Que filas incrementales corresponden a una sesion del historial: ``mapped`` (stream_ids con mapping explicito) o ``legacy``
        (clave live-<8> sin mapping, solo si ningun otro mapping la usa)."""
        out = {"mapped": [], "legacy": None}
        if not self.path.is_file():
            return out
        try:
            conn = self._connect("ro")
            try:
                have = self._tables(conn)
                maps = ({r["stream_id"]: r["history_session_id"] for r in conn.execute("SELECT * FROM inc_sessions")}
                        if "inc_sessions" in have else {})
                out["mapped"] = sorted(k for k, v in maps.items() if v == history_session_id)
                if not out["mapped"]:
                    lk = legacy_key(history_session_id)
                    if lk not in maps:
                        n = sum(conn.execute(f"SELECT COUNT(*) FROM {t} WHERE session=?", (lk,)).fetchone()[0] for t in ("inc_lines", "inc_units") if t in have)
                        out["legacy"] = lk if n else None
            finally:
                conn.close()
        except sqlite3.Error as exc:
            raise IncHistoryError(f"{type(exc).__name__}: {exc}") from exc
        return out

    def delete_streams(self, stream_ids: list[str], history_session_id: str | None = None) -> int:
        """Borra filas de esos stream_ids (y su mapping) en UNA transaccion."""
        if not self.path.is_file() or not stream_ids:
            return 0
        try:
            conn = self._connect("rw")
            try:
                conn.isolation_level = None
                conn.execute("BEGIN IMMEDIATE")
                n = 0
                try:
                    have = self._tables(conn)
                    for t in ("inc_lines", "inc_units"):
                        if t in have:
                            for sid in stream_ids:
                                n += conn.execute(f"DELETE FROM {t} WHERE session = ?", (sid,)).rowcount
                    if "inc_sessions" in have:
                        for sid in stream_ids:
                            conn.execute("DELETE FROM inc_sessions WHERE stream_id = ?", (sid,))
                    conn.execute("COMMIT")
                except BaseException:
                    conn.execute("ROLLBACK")
                    raise
            finally:
                conn.close()
        except sqlite3.Error as exc:
            raise IncHistoryError(f"{type(exc).__name__}: {exc}") from exc
        return n

    def delete(self, session: str | None = None) -> int:
        """Borra las DOS tablas en una sola transaccion (todo o nada). ``session`` None = todo. Devuelve filas borradas."""
        if not self.path.is_file():
            return 0
        try:
            conn = self._connect("rw")
            try:
                conn.isolation_level = None
                conn.execute("BEGIN IMMEDIATE")
                n = 0
                try:
                    for t in sorted(self._tables(conn)):
                        if session is not None and t == "inc_sessions":
                            continue
                        cur = conn.execute(f"DELETE FROM {t}" if session is None else f"DELETE FROM {t} WHERE session = ?",
                                           () if session is None else (session,))
                        n += cur.rowcount
                    conn.execute("COMMIT")
                except BaseException:
                    conn.execute("ROLLBACK")
                    raise
            finally:
                conn.close()
        except sqlite3.Error as exc:
            raise IncHistoryError(f"{type(exc).__name__}: {exc}") from exc
        return n


def merge_json(base_json: str, inc: dict) -> str:
    """Agrega al export JSON existente (lista) los textos incrementales, marcados con ``kind``. Sin alineacion entre lineas y unidades."""
    import json

    data = json.loads(base_json)
    for r in inc["lines"]:
        data.append({"kind": "inc_line_es", "session": r["session"], "history_session_id": r["history_session_id"], "id": r["line_id"], "ts": r["ts"], "text_es": r["text_es"]})
    for r in inc["units"]:
        data.append({"kind": "inc_unit_en", "session": r["session"], "history_session_id": r["history_session_id"], "id": r["unit_id"], "ts": r["ts"], "text_en": r["text_en"],
                     "state": r["state"], "playback": r["playback"], "n_chunks": r["n_chunks"], "reason": r["reason"]})
    return json.dumps(data, ensure_ascii=False, indent=2)


def merge_txt(base_txt: str, inc: dict) -> str:
    out = [base_txt.rstrip("\n"), "", "== Modo incremental experimental (español por línea e inglés por unidad; sin relación entre ellos) =="]
    for r in inc["lines"]:
        out.append(f"[{r['ts']}] ES (línea, sesión {r['session']}): {r['text_es']}")
    for r in inc["units"]:
        out.append(f"[{r['ts']}] EN (unidad, estado={r['state']}, reproducción no verificada, sesión {r['session']}): {r['text_en']}")
    out.append("")
    return "\n".join(out)
