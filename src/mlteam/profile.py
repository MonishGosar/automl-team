"""Column and table profiling.

Everything here is computed by SQL over the loaded data. No model is involved,
so the same data always gives the same profile.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from mlteam.store import Store, TableInfo, quote_ident

MAX_PROFILED_COLUMNS = 400
TOP_K = 5
LOW_CARDINALITY = 50

_INT_TYPES = {"TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT",
              "UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT", "UHUGEINT"}
_FLOAT_TYPES = {"FLOAT", "DOUBLE", "REAL"}

_ID_NAME_SNAKE = re.compile(r"(^|_)(id|uuid|guid|key|code|no|number|ref|reference)$|^id(_|$)", re.IGNORECASE)
_ID_NAME_CAMEL = re.compile(r"[a-z](Id|ID)$")


def is_id_name(name: str) -> bool:
    """True for names like id, user_id, Loan_ID, PassengerId. Not for 'paid' or 'valid'."""
    return bool(_ID_NAME_SNAKE.search(name) or _ID_NAME_CAMEL.search(name))


_TIME_NAME = re.compile(r"date|time|timestamp|(^|_)(dt|at|year|month|day)$", re.IGNORECASE)
_TARGET_NAME = re.compile(
    r"^(target|label|labels|class|y|outcome|output|result|response)$"
    r"|(default(ed)?|churn(ed)?|survived|fraud|approved|status|converted|cancell?ed|clicked"
    r"|purchased|returned|price|saleprice|sales|revenue|score)$"
    r"|^(is|has|did)_",
    re.IGNORECASE,
)


def type_class(dtype: str) -> str:
    upper = dtype.upper()
    base = upper.split("(")[0].strip()
    if base in _INT_TYPES:
        return "integer"
    if base in _FLOAT_TYPES or base == "DECIMAL":
        return "float"
    if base == "BOOLEAN":
        return "boolean"
    if base.startswith("TIMESTAMP") or base in {"DATE", "TIME"}:
        return "datetime"
    if base == "VARCHAR":
        return "string"
    return "other"


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    kind: str
    rows: int
    nulls: int
    null_pct: float
    distinct: int
    unique: bool
    min: str | None = None
    max: str | None = None
    mean: float | None = None
    std: float | None = None
    avg_length: float | None = None
    top: list[tuple[str, int]] = field(default_factory=list)
    samples: list[str] = field(default_factory=list)
    hints: list[str] = field(default_factory=list)


@dataclass
class TableProfile:
    name: str
    file: str
    rows: int
    columns: list[ColumnProfile]
    load_note: str = ""
    truncated_columns: int = 0

    def column(self, name: str) -> ColumnProfile | None:
        for col in self.columns:
            if col.name == name:
                return col
        return None

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]


@dataclass
class DatasetProfile:
    tables: list[TableProfile]

    def table(self, name: str) -> TableProfile | None:
        for table in self.tables:
            if table.name == name:
                return table
        return None

    def to_json(self, *, stats_only: bool = False) -> dict[str, Any]:
        """Serialise. ``stats_only`` strips every raw value so nothing from the
        data itself (only counts and types) leaves the machine."""
        data = asdict(self)
        if stats_only:
            for table in data["tables"]:
                for col in table["columns"]:
                    col["top"] = []
                    col["samples"] = []
                    if col["kind"] not in {"numeric", "datetime"}:
                        col["min"] = None
                        col["max"] = None
        return data


def infer_kind(cls: str, rows: int, nonnull: int, distinct: int, avg_len: float | None, name: str,
               span: float | None = None) -> str:
    """``span`` is max - min + 1 for whole numbers. A row number fills its range
    (1, 2, 3, ...); a price or an amount that happens to be different on every row does not."""
    if nonnull == 0:
        return "empty"
    if distinct <= 1:
        return "constant"
    all_unique = distinct == nonnull
    id_named = is_id_name(name)
    near_unique = id_named and distinct >= 0.9 * nonnull and nonnull >= 20
    # A text column that is different on almost every row is a reference, not a category.
    text_reference = cls == "string" and nonnull >= 50 and distinct >= 0.9 * nonnull and (avg_len or 0) <= 40
    if cls == "boolean":
        return "boolean"
    if cls == "datetime":
        return "datetime"
    if cls == "integer":
        dense = span is None or span <= 2 * distinct
        if (all_unique and (id_named or (rows >= 50 and dense))) or near_unique:
            return "id"
        return "categorical" if distinct <= 10 else "numeric"
    if cls == "float":
        return "numeric"
    if cls == "string":
        if (all_unique and (id_named or rows >= 20)) or near_unique or text_reference:
            return "id"
        if avg_len is not None and avg_len > 40 and distinct / max(nonnull, 1) > 0.5:
            return "text"
        return "categorical"
    return "other"


def name_hints(name: str, kind: str) -> list[str]:
    hints: list[str] = []
    if is_id_name(name):
        hints.append("id_name")
    if _TIME_NAME.search(name):
        hints.append("time_name")
    if _TARGET_NAME.search(name) and kind not in {"id", "text", "empty", "constant"}:
        hints.append("target_name")
    return hints


def _to_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if len(text) <= 80 else text[:77] + "..."


def profile_table(store: Store, info: TableInfo) -> TableProfile:
    table = quote_ident(info.name)
    columns = list(info.columns)
    truncated = max(0, len(columns) - MAX_PROFILED_COLUMNS)
    columns = columns[:MAX_PROFILED_COLUMNS]
    rows = info.rows

    stats: dict[str, dict[str, Any]] = {}
    for start in range(0, len(columns), 40):
        chunk = columns[start:start + 40]
        parts: list[str] = []
        for offset, (name, dtype) in enumerate(chunk):
            i = start + offset
            col = quote_ident(name)
            cls = type_class(dtype)
            parts.append(f"count({col}) AS n_{i}")
            if cls == "other":
                parts.append(f"count(DISTINCT {col}::VARCHAR) AS d_{i}")
                continue
            parts.append(f"count(DISTINCT {col}) AS d_{i}")
            parts.append(f"min({col})::VARCHAR AS mn_{i}")
            parts.append(f"max({col})::VARCHAR AS mx_{i}")
            if cls in {"integer", "float"}:
                parts.append(f"avg({col})::DOUBLE AS av_{i}")
                parts.append(f"stddev_samp({col})::DOUBLE AS sd_{i}")
            if cls == "string":
                parts.append(f"avg(length({col}))::DOUBLE AS ln_{i}")
        if not parts:
            continue
        names, result = store.query(f"SELECT {', '.join(parts)} FROM {table}")
        row = result[0] if result else tuple([None] * len(names))
        values = dict(zip(names, row, strict=False))
        for offset, (name, _dtype) in enumerate(chunk):
            i = start + offset
            stats[name] = {
                "nonnull": int(values.get(f"n_{i}") or 0),
                "distinct": int(values.get(f"d_{i}") or 0),
                "min": values.get(f"mn_{i}"),
                "max": values.get(f"mx_{i}"),
                "mean": values.get(f"av_{i}"),
                "std": values.get(f"sd_{i}"),
                "avg_len": values.get(f"ln_{i}"),
            }

    sample_names, sample_rows = store.query(f"SELECT * FROM {table} LIMIT {TOP_K}")
    sample_index = {name: idx for idx, name in enumerate(sample_names)}

    profiles: list[ColumnProfile] = []
    for name, dtype in columns:
        s = stats.get(name, {})
        nonnull = s.get("nonnull", 0)
        distinct = s.get("distinct", 0)
        cls = type_class(dtype)
        span = None
        if cls == "integer" and s.get("min") is not None and s.get("max") is not None:
            try:
                span = float(s["max"]) - float(s["min"]) + 1
            except (TypeError, ValueError):
                span = None
        kind = infer_kind(cls, rows, nonnull, distinct, s.get("avg_len"), name, span)
        nulls = rows - nonnull
        top: list[tuple[str, int]] = []
        if 0 < distinct <= LOW_CARDINALITY and kind not in {"id", "text"}:
            col = quote_ident(name)
            _, top_rows = store.query(
                f"SELECT {col}::VARCHAR AS v, count(*) AS c FROM {table} "
                f"WHERE {col} IS NOT NULL GROUP BY 1 ORDER BY c DESC, v LIMIT {TOP_K}"
            )
            top = [(_to_text(v) or "", int(c)) for v, c in top_rows]
        samples: list[str] = []
        if name in sample_index:
            idx = sample_index[name]
            samples = [t for t in (_to_text(r[idx]) for r in sample_rows) if t is not None]
        mean = s.get("mean")
        std = s.get("std")
        profiles.append(
            ColumnProfile(
                name=name,
                dtype=dtype,
                kind=kind,
                rows=rows,
                nulls=nulls,
                null_pct=round(100.0 * nulls / rows, 2) if rows else 0.0,
                distinct=distinct,
                unique=bool(rows and nonnull == rows and distinct == rows),
                min=_to_text(s.get("min")),
                max=_to_text(s.get("max")),
                mean=round(float(mean), 6) if mean is not None else None,
                std=round(float(std), 6) if std is not None else None,
                avg_length=round(float(s["avg_len"]), 2) if s.get("avg_len") is not None else None,
                top=top,
                samples=samples,
                hints=name_hints(name, kind),
            )
        )
    return TableProfile(
        name=info.name,
        file=info.file,
        rows=rows,
        columns=profiles,
        load_note=info.load_note,
        truncated_columns=truncated,
    )


def profile_dataset(store: Store, infos: list[TableInfo]) -> DatasetProfile:
    return DatasetProfile(tables=[profile_table(store, info) for info in infos])
