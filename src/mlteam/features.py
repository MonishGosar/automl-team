"""Features: the clues the model is allowed to see.

A feature is one DuckDB expression evaluated for one row of the base table,
aliased ``b``. It may look into other tables with a correlated subquery:

    b."plan"
    (SELECT count(*) FROM orders o WHERE o.customer_id = b.customer_id)

Two properties hold by construction, whoever wrote the expression:

* The row count cannot change. Features are a SELECT list over the base table,
  so a join that multiplies rows is impossible here.
* No feature can see past the prediction moment. When the contract sets one,
  every dated, summarised table is replaced, for the whole query, by only the
  rows that exist at that moment. Naming the unfiltered table is refused.

Features are stored as versioned definitions, not as a copy of the data: the
raw files are hashed and the SQL is saved, so the values can be recomputed.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from mlteam.contract import (
    DataContract,
    base_sql,
    target_stats,
    time_safe_child_sql,
)
from mlteam.profile import DatasetProfile
from mlteam.runstore import RunPaths, now_iso, write_json
from mlteam.store import Store, StoreError, quote_ident, quote_literal

MAX_FEATURES = 60
MAX_SQL_LENGTH = 1500
MAX_AGG_NUMERIC = 4
MAX_CATEGORIES = 50

_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,59}$")
_QUALIFIED = re.compile(r"\b(main|temp|system|information_schema|pg_catalog)\s*\.", re.IGNORECASE)
_FORBIDDEN = re.compile(
    r"\b(read_csv|read_csv_auto|read_parquet|read_json|read_text|read_blob|glob|copy|attach|install|load|pragma)\b",
    re.IGNORECASE,
)
_NUMERIC_TYPES = {"TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT", "UTINYINT", "USMALLINT",
                  "UINTEGER", "UBIGINT", "UHUGEINT", "FLOAT", "DOUBLE", "REAL", "DECIMAL"}


@dataclass
class Feature:
    name: str
    sql: str
    why: str = ""
    origin: str = "rules"  # "rules" or "engineer"
    kind: str = ""
    null_pct: float = 0.0
    distinct: int = 0
    error: str | None = None    # unusable: cannot be selected
    leak: str | None = None     # usable but looks like it gives the answer away: off by default
    warning: str | None = None  # usable, with a caveat
    suggested_off: str | None = None  # usable, but understanding the data says leave it out

    @property
    def usable(self) -> bool:
        return self.error is None

    @property
    def default_on(self) -> bool:
        return self.error is None and self.leak is None and self.suggested_off is None


@dataclass
class FeatureSet:
    dataset_hash: str
    contract_version: int
    horizon_days: int | None
    features: list[Feature] = field(default_factory=list)
    selected: list[str] = field(default_factory=list)
    version: int = 0
    approved_by: str | None = None
    approved_at: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------


def time_rule_ctes(contract: DataContract) -> str:
    """``WITH`` clause that shadows each dated, summarised table with only the
    rows that exist at the prediction moment. Empty when no time rule applies."""
    parts = []
    for step in contract.joins:
        safe = time_safe_child_sql(contract, step)
        if safe is not None:
            parts.append(f"{quote_ident(step.table)} AS ({safe})")
    return ("WITH " + ", ".join(parts) + " ") if parts else ""


def _frame_select(contract: DataContract, features: list[Feature]) -> str:
    selects = [f"b.{quote_ident(k)}" for k in contract.grain_keys]
    if contract.target:
        selects.append(f"b.{quote_ident(contract.target)}")
    selects += [f"({f.sql}) AS {quote_ident(f.name)}" for f in features]
    return f"SELECT {', '.join(selects)} FROM {base_sql(contract)} AS b"


def frame_sql(contract: DataContract, features: list[Feature]) -> str:
    """The modelling table: row key(s), target, then one column per feature."""
    return f"{time_rule_ctes(contract)}{_frame_select(contract, features)}"


def features_query_sql(contract: DataContract, features: list[Feature], query: str) -> str:
    """Wrap an agent's SELECT so it can read a table called ``features`` (the
    modelling table) with the time rule applied to the whole statement. Raises
    ValueError with a reason when the query is not allowed."""
    text = query.strip().rstrip(";").strip()
    if not text:
        raise ValueError("The query is empty.")
    if ";" in text or "--" in text or "/*" in text:
        raise ValueError("Use one statement, without ';' or comments.")
    if not re.match(r"^(select|with)\b", text, re.IGNORECASE):
        raise ValueError("Only SELECT queries are allowed here.")
    if _QUALIFIED.search(text):
        raise ValueError("Do not name a schema (such as main.orders). It would bypass the time rule.")
    if _FORBIDDEN.search(text):
        raise ValueError("File and system functions are not allowed.")
    ctes = time_rule_ctes(contract)
    head = (ctes.rstrip() + ", ") if ctes else "WITH "
    return f"{head}features AS ({_frame_select(contract, features)}) SELECT * FROM ({text})"


def check_expression(name: str, sql: str, contract: DataContract, taken: set[str]) -> str | None:
    """Reasons to refuse an expression before running it. None means it may run."""
    if not _NAME.match(name):
        return "The name must start with a letter and use only letters, digits and underscores."
    if name.lower() in taken:
        return f"A feature called {name} already exists."
    reserved = {k.lower() for k in contract.grain_keys} | ({contract.target.lower()} if contract.target else set())
    if name.lower() in reserved:
        return f"{name} is the name of the target or the row key."
    text = sql.strip()
    if not text:
        return "The expression is empty."
    if len(text) > MAX_SQL_LENGTH:
        return f"The expression is longer than {MAX_SQL_LENGTH} characters."
    if ";" in text or "--" in text or "/*" in text:
        return "Use one expression, without ';' or comments."
    if _QUALIFIED.search(text):
        return "Do not name a schema (such as main.orders). It would bypass the time rule."
    if _FORBIDDEN.search(text):
        return "File and system functions are not allowed in a feature."
    if contract.target:
        target = re.escape(contract.target)
        if re.search(rf'(?<![A-Za-z0-9_]){target}(?![A-Za-z0-9_])', text, re.IGNORECASE):
            return f"The expression uses the target ({contract.target}). A feature cannot be built from the answer."
    return None


# ---------------------------------------------------------------------------
# Rule-based starting set
# ---------------------------------------------------------------------------


def default_features(contract: DataContract, profile: DatasetProfile) -> list[Feature]:
    """What fixed rules would use: the kept base columns, the columns of each
    lookup table, and a few summaries of each many-row table."""
    base = profile.table(contract.base_table)
    if base is None:
        return []
    features: list[Feature] = []
    taken: set[str] = set()

    def add(name: str, sql: str, why: str) -> None:
        clean = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_") or "feature"
        if not clean[0].isalpha():
            clean = f"f_{clean}"
        clean = clean[:60]
        if clean.lower() in taken or len(features) >= MAX_FEATURES:
            return
        taken.add(clean.lower())
        features.append(Feature(name=clean, sql=sql, why=why))

    excluded = {contract.target, *contract.drop_columns, *[s.base_key for s in contract.joins]}
    for col in base.columns:
        if col.name in excluded:
            continue
        ref = f"b.{quote_ident(col.name)}"
        if col.name == contract.time_column or col.kind == "datetime":
            add(f"{col.name}_month", f"date_part('month', {ref})", f"Month of {col.name}, to catch seasonal patterns.")
            add(f"{col.name}_weekday", f"date_part('isodow', {ref})", f"Day of the week of {col.name}.")
        elif col.kind in {"numeric", "categorical", "boolean"}:
            add(col.name, ref, f"{col.name} as it is in {base.name}.")

    for step in contract.joins:
        other = profile.table(step.table)
        if other is None:
            continue
        match = (f"c.{quote_ident(step.other_key)}::VARCHAR = b.{quote_ident(step.base_key)}::VARCHAR")
        source = f"{quote_ident(step.table)} AS c"
        if step.kind == "lookup":
            for col in other.columns:
                if col.name == step.other_key or col.kind not in {"numeric", "categorical", "boolean"}:
                    continue
                add(f"{step.table}__{col.name}",
                    f"(SELECT c.{quote_ident(col.name)} FROM {source} WHERE {match})",
                    f"{col.name} looked up from {step.table}.")
            continue
        add(f"{step.table}__count", f"(SELECT count(*) FROM {source} WHERE {match})",
            f"How many {step.table} rows belong to this row.")
        numeric = [c for c in other.columns if c.kind == "numeric" and c.name != step.other_key]
        for col in numeric[:MAX_AGG_NUMERIC]:
            add(f"{step.table}__{col.name}__mean",
                f"(SELECT avg(c.{quote_ident(col.name)}) FROM {source} WHERE {match})",
                f"Average {col.name} across this row's {step.table}.")
            add(f"{step.table}__{col.name}__sum",
                f"(SELECT coalesce(sum(c.{quote_ident(col.name)}), 0) FROM {source} WHERE {match})",
                f"Total {col.name} across this row's {step.table}.")
        if step.time_column and contract.time_column and contract.horizon_days is not None:
            moment = (f"b.{quote_ident(contract.time_column)} + INTERVAL '{int(contract.horizon_days)}' DAY")
            add(f"{step.table}__days_since_last",
                f"(SELECT date_diff('day', max(c.{quote_ident(step.time_column)}), {moment}) FROM {source} WHERE {match})",
                f"Days between the latest {step.table} row and the prediction moment. Recent activity often matters.")
    return features


# ---------------------------------------------------------------------------
# Evaluation: run each feature, describe it, and look for leaks.
# ---------------------------------------------------------------------------


def _kind(dtype: str | None) -> str:
    base = (dtype or "").upper().split("(")[0].strip()
    if base in _NUMERIC_TYPES:
        return "numeric"
    if base == "BOOLEAN":
        return "boolean"
    if base == "VARCHAR":
        return "categorical"
    if base.startswith("TIMESTAMP") or base in {"DATE", "TIME"}:
        return "datetime"
    return "other"


def _target_as_number(contract: DataContract, stats: dict[str, Any]) -> str | None:
    """SQL turning the target column ``t`` into a number, for correlation."""
    if contract.task == "regression":
        return "try_cast(t AS DOUBLE)"
    top = stats.get("top") or []
    if stats.get("distinct") == 2 and top:
        return f"CASE WHEN t::VARCHAR = {quote_literal(str(top[0][0]))} THEN 1.0 ELSE 0.0 END"
    return None


def evaluate_feature(
    store: Store, contract: DataContract, feature: Feature, taken: set[str], stats: dict[str, Any]
) -> Feature:
    """Fill in ``kind``, ``null_pct``, ``distinct`` and any error, leak or warning."""
    feature.error = feature.leak = feature.warning = None
    feature.error = check_expression(feature.name, feature.sql, contract, taken)
    if feature.error:
        return feature
    ctes = time_rule_ctes(contract)
    target = f"b.{quote_ident(contract.target)}" if contract.target else "NULL"
    inner = f"SELECT ({feature.sql}) AS v, {target} AS t FROM {base_sql(contract)} AS b"
    try:
        _, rows = store.query(
            f"{ctes}SELECT count(*), count(v), count(DISTINCT v), "
            f"any_value(typeof(v)) FILTER (WHERE v IS NOT NULL) FROM ({inner})"
        )
    except StoreError as exc:
        feature.error = str(exc).splitlines()[0][:300]
        return feature
    total, nonnull, distinct, dtype = rows[0]
    total, nonnull, distinct = int(total or 0), int(nonnull or 0), int(distinct or 0)
    feature.kind = _kind(dtype)
    feature.distinct = distinct
    feature.null_pct = round(100.0 * (total - nonnull) / total, 2) if total else 0.0
    if nonnull == 0:
        feature.error = "It is empty on every row."
        return feature
    if distinct <= 1:
        feature.error = "It has the same value on every row, so it carries no information."
        return feature
    if feature.kind == "datetime":
        feature.error = "It returns a date. Turn it into a number, such as days since, or a part, such as the month."
        return feature
    if feature.kind == "other":
        feature.error = f"It returns {dtype}, which a model cannot use directly."
        return feature

    try:
        if contract.target and contract.task == "classification" and 2 <= distinct <= MAX_CATEGORIES and total >= 30:
            top = stats.get("top") or []
            seen = sum(c for _, c in top) or 1
            baseline = max((c for _, c in top), default=seen) / seen
            if baseline < 0.95:
                purity = store.scalar(
                    f"{ctes}SELECT sum(mx)::DOUBLE / sum(n) FROM (SELECT max(cnt) AS mx, sum(cnt) AS n FROM "
                    f"(SELECT v, t, count(*) AS cnt FROM ({inner}) WHERE v IS NOT NULL AND t IS NOT NULL "
                    f"GROUP BY 1, 2) GROUP BY v)"
                )
                if purity is not None and float(purity) >= 0.995:
                    feature.leak = f"Knowing it alone gives the target {float(purity):.1%} of the time."
        number = _target_as_number(contract, stats)
        if feature.leak is None and feature.kind in {"numeric", "boolean"} and number and total >= 30:
            corr = store.scalar(f"{ctes}SELECT corr(v::DOUBLE, {number}) FROM ({inner})")
            if corr is not None and abs(float(corr)) >= 0.98:
                feature.leak = f"It moves almost exactly with the target (correlation {float(corr):.2f})."
    except StoreError:
        pass

    if feature.kind == "categorical" and distinct > MAX_CATEGORIES:
        feature.warning = f"{distinct:,} different values. Too many categories usually hurts more than it helps."
    elif distinct == nonnull and nonnull == total and feature.kind != "numeric":
        feature.warning = "Different on every row, like an identifier."
    elif feature.null_pct >= 80:
        feature.warning = f"Missing on {feature.null_pct:.0f}% of rows."
    return feature


def evaluate_features(store: Store, contract: DataContract, features: list[Feature]) -> list[Feature]:
    stats = target_stats(store, contract)
    taken: set[str] = set()
    for feature in features:
        evaluate_feature(store, contract, feature, taken, stats)
        if feature.error is None or "already exists" not in feature.error:
            taken.add(feature.name.lower())
    return features


def time_rule_summary(contract: DataContract) -> str:
    dated = [s for s in contract.joins if s.kind == "aggregate" and s.time_column]
    if contract.horizon_days is not None and contract.time_column:
        hidden = ", ".join(s.table for s in dated) or "no joined table is dated"
        return (f"Prediction moment: {contract.horizon_days} days after {contract.time_column}. "
                f"Rows after it are hidden from every feature ({hidden}).")
    if dated:
        return ("No prediction moment is set, so features see every row of "
                + ", ".join(s.table for s in dated) + ", including any from after the outcome.")
    return "No time rule: nothing joined here is dated."


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_feature_set(run: RunPaths, feature_set: FeatureSet, approved_by: str) -> Path:
    version = 1
    while run.features_version(version).exists():
        version += 1
    feature_set.version = version
    feature_set.approved_by = approved_by
    feature_set.approved_at = now_iso()
    path = run.features_version(version)
    write_json(path, feature_set.to_json())
    write_json(run.features_latest, feature_set.to_json())
    return path
