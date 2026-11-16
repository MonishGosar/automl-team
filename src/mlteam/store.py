"""DuckDB-backed table store.

Lifecycle: open read-write, load the raw files, export Parquet, then ``seal()``.
A sealed store is read-only and cannot touch the filesystem or network, which
is what makes it safe to let an agent write SQL against it.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

from mlteam.intake import Manifest
from mlteam.runstore import RunPaths


class StoreError(Exception):
    """Raised when data cannot be loaded or a query is not allowed."""


@dataclass(frozen=True)
class TableInfo:
    name: str
    file: str
    rows: int
    columns: tuple[tuple[str, str], ...]
    load_note: str = ""


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def quote_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


_SUFFIX = re.compile(r"(\.csv|\.tsv)?(\.gz)?$|\.parquet$", re.IGNORECASE)


def table_name_for(file_name: str, used: set[str]) -> str:
    stem = _SUFFIX.sub("", file_name) or file_name
    name = re.sub(r"[^a-z0-9_]+", "_", stem.lower()).strip("_") or "table"
    if not name[0].isalpha():
        name = f"t_{name}"
    candidate, counter = name, 2
    while candidate in used:
        candidate = f"{name}_{counter}"
        counter += 1
    used.add(candidate)
    return candidate


# Tried in order. The first that loads wins; anything after the first is
# reported to the user because it means the file needed special handling.
_CSV_ATTEMPTS: tuple[tuple[str, str], ...] = (
    ("", ""),
    ("sample_size=-1", "types detected from the whole file"),
    ("encoding='latin-1'", "read as Latin-1 text encoding"),
    (
        "all_varchar=true, null_padding=true, ignore_errors=true",
        "loaded as text only; malformed rows were skipped",
    ),
    (
        "encoding='latin-1', all_varchar=true, null_padding=true, ignore_errors=true",
        "read as Latin-1, loaded as text only; malformed rows were skipped",
    ),
)

_READ_ONLY_START = re.compile(r"^\s*(select|with|describe|summarize|show|explain)\b", re.IGNORECASE)
_SQL_COMMENT = re.compile(r"--[^\n]*|/\*.*?\*/", re.DOTALL)


class Store:
    def __init__(self, db_path: Path) -> None:
        self._path = db_path
        self._lock = threading.RLock()
        self._sealed = False
        self._con = duckdb.connect(str(db_path))

    # -- loading ---------------------------------------------------------

    def load(self, manifest: Manifest, run: RunPaths) -> list[TableInfo]:
        if self._sealed:
            raise StoreError("Store is sealed; loading is no longer possible.")
        infos: list[TableInfo] = []
        used: set[str] = set()
        with self._lock:
            for raw in manifest.files:
                table = table_name_for(raw.name, used)
                note = self._load_file(table, run.raw / raw.name)
                rows = self._con.execute(f"SELECT count(*) FROM {quote_ident(table)}").fetchone()[0]
                described = self._con.execute(f"DESCRIBE {quote_ident(table)}").fetchall()
                columns = tuple((str(row[0]), str(row[1])) for row in described)
                if rows == 0:
                    note = (note + "; " if note else "") + "table is empty"
                infos.append(
                    TableInfo(name=table, file=raw.name, rows=int(rows), columns=columns, load_note=note)
                )
        return infos

    @classmethod
    def beside(cls, db_path: Path, file_path: Path, table: str) -> Store:
        """A sealed store that sees a finished run's tables plus one new file,
        loaded as a temporary table. The run's database is attached read-only
        and is never written to."""
        self = cls.__new__(cls)
        self._path = db_path
        self._lock = threading.RLock()
        self._sealed = False
        self._con = duckdb.connect(":memory:")
        try:
            self._con.execute(f"ATTACH {quote_literal(Path(db_path).as_posix())} AS run (READ_ONLY)")
            self._con.execute("USE run")
            self._load_file(table, Path(file_path), temp=True)
            self._con.execute("SET enable_external_access = false")
            self._con.execute("SET lock_configuration = true")
        except duckdb.Error as exc:
            self._con.close()
            raise StoreError(f"Could not open {Path(file_path).name} beside this run: {exc}") from exc
        except StoreError:
            self._con.close()
            raise
        self._sealed = True
        return self

    def _load_file(self, table: str, path: Path, *, temp: bool = False) -> str:
        source = quote_literal(path.as_posix())
        target = quote_ident(table)
        create = "CREATE TEMP TABLE" if temp else "CREATE TABLE"
        if path.name.lower().endswith(".parquet"):
            try:
                self._con.execute(f"{create} {target} AS SELECT * FROM read_parquet({source})")
            except duckdb.Error as exc:
                raise StoreError(f"Could not read {path.name}: {exc}") from exc
            return ""
        last_error: Exception | None = None
        for options, note in _CSV_ATTEMPTS:
            args = f"{source}, {options}" if options else source
            try:
                self._con.execute(f"{create} {target} AS SELECT * FROM read_csv({args})")
                return note
            except duckdb.Error as exc:
                last_error = exc
                self._con.execute(f"DROP TABLE IF EXISTS {target}")
        raise StoreError(f"Could not read {path.name}: {last_error}")

    def export_parquet(self, folder: Path, tables: list[TableInfo]) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        with self._lock:
            for info in tables:
                dest = quote_literal((folder / f"{info.name}.parquet").as_posix())
                self._con.execute(f"COPY {quote_ident(info.name)} TO {dest} (FORMAT parquet)")

    def seal(self) -> None:
        """Reopen read-only with filesystem and network access switched off."""
        with self._lock:
            self._con.close()
            self._con = duckdb.connect(str(self._path), read_only=True)
            self._con.execute("SET enable_external_access = false")
            self._con.execute("SET lock_configuration = true")
            self._sealed = True

    @property
    def sealed(self) -> bool:
        return self._sealed

    # -- querying --------------------------------------------------------

    def query(self, sql: str, params: list[Any] | None = None) -> tuple[list[str], list[tuple[Any, ...]]]:
        """Trusted query path, used by harness code only."""
        with self._lock:
            try:
                cursor = self._con.execute(sql, params or [])
                columns = [d[0] for d in cursor.description] if cursor.description else []
                return columns, cursor.fetchall()
            except duckdb.Error as exc:
                raise StoreError(str(exc)) from exc

    def scalar(self, sql: str, params: list[Any] | None = None) -> Any:
        _, rows = self.query(sql, params)
        return rows[0][0] if rows else None

    def agent_query(self, sql: str, max_rows: int = 50) -> tuple[list[str], list[tuple[Any, ...]], bool]:
        """Untrusted query path for agent-written SQL: one read-only statement,
        on a sealed store, with a row cap. Returns (columns, rows, truncated)."""
        if not self._sealed:
            raise StoreError("Agent queries need a sealed store.")
        cleaned = _SQL_COMMENT.sub(" ", sql).strip().rstrip(";").strip()
        if not cleaned:
            raise StoreError("Empty query.")
        if ";" in cleaned:
            raise StoreError("Only one statement per call (remove the ';').")
        if not _READ_ONLY_START.match(cleaned):
            raise StoreError("Only read-only statements are allowed: SELECT, WITH, DESCRIBE, SUMMARIZE.")
        with self._lock:
            try:
                cursor = self._con.execute(cleaned)
                columns = [d[0] for d in cursor.description] if cursor.description else []
                rows = cursor.fetchmany(max_rows + 1)
            except duckdb.Error as exc:
                raise StoreError(str(exc)) from exc
        truncated = len(rows) > max_rows
        return columns, rows[:max_rows], truncated

    def close(self) -> None:
        with self._lock:
            self._con.close()
