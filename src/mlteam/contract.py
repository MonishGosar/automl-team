"""The data contract: the agreed answer to "what exactly are we predicting, and
what is one row?"

A draft is proposed by fixed rules, may be refined by the Analyst agent, is
checked by the harness, and only becomes real when a person approves it.
Nothing downstream is allowed to run without an approved contract.
"""

from __future__ import annotations

import copy
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from mlteam.profile import ColumnProfile, DatasetProfile, TableProfile
from mlteam.relations import Layout
from mlteam.runstore import RunPaths, now_iso, write_json
from mlteam.store import Store, StoreError, quote_ident

TASKS = ("classification", "regression")
METRICS: dict[str, tuple[str, ...]] = {
    "classification": ("roc_auc", "pr_auc", "f1", "macro_f1", "accuracy", "log_loss"),
    "regression": ("rmse", "mae", "r2"),
}
# Issues with this field can only be cleared by a person, never by an agent.
QUESTIONS_FIELD = "questions"
MAX_LEAKAGE_COLUMNS = 100
MAX_AGG_COLUMNS = 10


@dataclass
class JoinStep:
    table: str
    kind: str  # "lookup": many rows here -> one there. "aggregate": one row here -> many there.
    base_key: str
    other_key: str
    matched: float
    # For "aggregate" joins: the date column of the other table. With a prediction
    # moment set, only its rows dated on or before that moment are ever used.
    time_column: str | None = None

    def label(self) -> str:
        how = "look up" if self.kind == "lookup" else "summarise"
        dated = f" · dated by {self.time_column}" if self.kind == "aggregate" and self.time_column else ""
        return f"{how} {self.table} on {self.base_key} = {self.other_key} ({self.matched:.0%} matched){dated}"


@dataclass
class Suspect:
    column: str
    reason: str


@dataclass
class DataContract:
    dataset_hash: str
    layout: str
    base_table: str
    stack: list[str] = field(default_factory=list)
    test_table: str | None = None
    target: str | None = None
    task: str | None = None
    metric: str | None = None
    grain: str = ""
    grain_keys: list[str] = field(default_factory=list)
    dedupe: bool = False  # drop rows that are exact copies of another row, before anything else
    time_column: str | None = None
    # The prediction moment: this many days after ``time_column``. None means
    # "no time rule": every row of every joined table is used.
    horizon_days: int | None = None
    # Rows that belong together (e.g. several orders of one customer). Kept on one side
    # of every split, so the model is tested on groups it has never seen.
    group_column: str | None = None
    # Columns to check the model's fairness on, beyond the ones found by name (age, sex, ...).
    fairness_columns: list[str] = field(default_factory=list)
    # What a wrong yes/no decision costs, in any unit. When both are set the cut-off
    # is chosen for the lowest cost instead of the best F1.
    cost_false_alarm: float | None = None
    cost_miss: float | None = None
    drop_columns: list[str] = field(default_factory=list)
    joins: list[JoinStep] = field(default_factory=list)
    leakage_suspects: list[Suspect] = field(default_factory=list)
    rationale: dict[str, str] = field(default_factory=dict)
    # Things the data cannot answer and only a person can, e.g. "at what moment
    # is the prediction made?". Asked at the gate instead of guessed.
    open_questions: list[str] = field(default_factory=list)
    # The person's answers, keyed by question. Approval is refused while any
    # question has no answer; later stages read these.
    answers: dict[str, str] = field(default_factory=dict)
    version: int = 0
    proposed_by: str = "rules"
    approved_by: str | None = None
    approved_at: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def contract_from_json(data: dict[str, Any]) -> DataContract:
    payload = dict(data)
    payload["joins"] = [JoinStep(**j) for j in payload.get("joins", [])]
    payload["leakage_suspects"] = [Suspect(**s) for s in payload.get("leakage_suspects", [])]
    known = DataContract.__dataclass_fields__.keys()
    return DataContract(**{k: v for k, v in payload.items() if k in known})


@dataclass
class Issue:
    level: str  # "error" blocks approval, "warning" does not
    field: str
    message: str


# ---------------------------------------------------------------------------
# SQL builders
# ---------------------------------------------------------------------------


def base_sql(contract: DataContract) -> str:
    """The base relation: one table, or shards stacked by column name, with exact
    duplicate rows dropped when the contract says so."""
    tables = contract.stack if len(contract.stack) > 1 else [contract.base_table]
    if len(tables) == 1:
        relation = quote_ident(tables[0])
    else:
        relation = "(" + " UNION ALL BY NAME ".join(f"SELECT * FROM {quote_ident(t)}" for t in tables) + ")"
    return f"(SELECT DISTINCT * FROM {relation})" if contract.dedupe else relation


def exact_copies(store: Store, contract: DataContract) -> int:
    raw = copy.copy(contract)
    raw.dedupe = False
    rel = base_sql(raw)
    total = int(store.scalar(f"SELECT count(*) FROM {rel} AS b") or 0)
    distinct = int(store.scalar(f"SELECT count(*) FROM (SELECT DISTINCT * FROM {rel} AS b)") or 0)
    return total - distinct


def model_input_sql(contract: DataContract, profile: DatasetProfile) -> str:
    """Base rows plus every approved join. Lookups add the other table's
    columns; aggregates first collapse the other table to one row per key so
    the base row count cannot change."""
    selects = ["b.*"]
    joins: list[str] = []
    for i, step in enumerate(contract.joins):
        other = profile.table(step.table)
        if other is None:
            continue
        alias = f"j{i}"
        base_key = f"b.{quote_ident(step.base_key)}::VARCHAR"
        if step.kind == "lookup":
            for col in other.columns:
                if col.name == step.other_key:
                    continue
                selects.append(
                    f"{alias}.{quote_ident(col.name)} AS {quote_ident(f'{step.table}__{col.name}')}"
                )
            joins.append(
                f"LEFT JOIN {quote_ident(step.table)} AS {alias} "
                f"ON {base_key} = {alias}.{quote_ident(step.other_key)}::VARCHAR"
            )
        else:
            aggs = [f"count(*) AS {quote_ident(f'{step.table}__rows')}"]
            numeric = [c for c in other.columns if c.kind == "numeric" and c.name != step.other_key]
            for col in numeric[:MAX_AGG_COLUMNS]:
                aggs.append(
                    f"avg({quote_ident(col.name)}) AS {quote_ident(f'{step.table}__{col.name}__mean')}"
                )
            sub = (
                f"(SELECT {quote_ident(step.other_key)}::VARCHAR AS _k, {', '.join(aggs)} "
                f"FROM {quote_ident(step.table)} WHERE {quote_ident(step.other_key)} IS NOT NULL GROUP BY 1)"
            )
            selects.append(f"{alias}.* EXCLUDE (_k)")
            joins.append(f"LEFT JOIN {sub} AS {alias} ON {base_key} = {alias}._k")
    return f"SELECT {', '.join(selects)} FROM {base_sql(contract)} AS b " + " ".join(joins)


def time_safe_child_sql(contract: DataContract, step: JoinStep) -> str | None:
    """Rows of ``step.table`` that already exist at the prediction moment, as a
    SELECT. None when no time rule applies to this join."""
    if (
        step.kind != "aggregate"
        or not step.time_column
        or contract.horizon_days is None
        or not contract.time_column
    ):
        return None
    days = int(contract.horizon_days)
    return (
        f"SELECT c.* FROM main.{quote_ident(step.table)} AS c WHERE EXISTS ("
        f"SELECT 1 FROM {base_sql(contract)} AS p "
        f"WHERE p.{quote_ident(step.base_key)}::VARCHAR = c.{quote_ident(step.other_key)}::VARCHAR "
        f"AND c.{quote_ident(step.time_column)} <= p.{quote_ident(contract.time_column)} + INTERVAL '{days}' DAY)"
    )


def attach_time_columns(joins: list[JoinStep], profile: DatasetProfile) -> list[JoinStep]:
    """Fill in the date column of each summarised table, when it has one."""
    for step in joins:
        if step.kind == "aggregate" and not step.time_column:
            other = profile.table(step.table)
            dated = [c.name for c in other.columns if c.kind == "datetime"] if other else []
            step.time_column = dated[0] if dated else None
    return joins


def target_stats(store: Store, contract: DataContract) -> dict[str, Any]:
    if not contract.target:
        return {}
    col = quote_ident(contract.target)
    rel = base_sql(contract)
    try:
        _, head = store.query(
            f"SELECT count(*), count({col}), count(DISTINCT {col}) FROM {rel} AS b"
        )
        rows, nonnull, distinct = (int(v or 0) for v in head[0])
        _, top = store.query(
            f"SELECT {col}::VARCHAR AS v, count(*) AS c FROM {rel} AS b "
            f"WHERE {col} IS NOT NULL GROUP BY 1 ORDER BY c DESC, v LIMIT 10"
        )
    except StoreError:
        return {}
    return {
        "rows": rows,
        "nulls": rows - nonnull,
        "distinct": distinct,
        "top": [(str(v), int(c)) for v, c in top],
    }


# ---------------------------------------------------------------------------
# Proposal
# ---------------------------------------------------------------------------


def _entity_from(name: str) -> str:
    words = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    words = re.sub(r"[_\-\s]+", " ", words).strip().lower()
    words = re.sub(r"\b(id|uuid|key|code|no|number)\b", "", words).strip()
    return words


def _singular_word(name: str) -> str:
    word = re.sub(r"[_\-]+", " ", name).strip().lower()
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _pick_base(profile: DatasetProfile, layout: Layout) -> tuple[str, str]:
    if len(profile.tables) == 1:
        return profile.tables[0].name, "Only one table was provided."
    if layout.kind == "train_test" and layout.train:
        return layout.train, f"{layout.train} is the training split; {layout.test} is held for scoring."
    if layout.kind == "shards" and layout.shard_groups:
        group = layout.shard_groups[0]
        return group[0], f"{len(group)} files share the same columns, so they are stacked into one table."
    follower = {name for group in layout.shard_groups for name in group[1:]}
    best: tuple[float, int, str] | None = None
    for table in profile.tables:
        if table.name in follower or table.name in {layout.test, layout.submission}:
            continue
        score = 0.0
        if any("target_name" in c.hints for c in table.columns):
            score += 3
        score += sum(1 for r in layout.relations_for(table.name) if r.cardinality == "many-to-one")
        candidate = (score, table.rows, table.name)
        if best is None or candidate > best:
            best = candidate
    assert best is not None
    return best[2], (
        "Chosen because it holds an outcome-like column and points at the other tables. "
        "Change it if another table holds what you want to predict."
    )


def _target_candidates(
    base: TableProfile, profile: DatasetProfile, layout: Layout, join_keys: set[str]
) -> list[tuple[str, int, str]]:
    scores: dict[str, tuple[int, list[str]]] = {}

    def bump(name: str, points: int, reason: str) -> None:
        score, reasons = scores.get(name, (0, []))
        scores[name] = (score + points, [*reasons, reason])

    usable = {
        c.name: c for c in base.columns
        if c.kind in {"categorical", "boolean", "numeric"} and c.name not in join_keys
    }
    if layout.train == base.name:
        for name in layout.train_only_columns:
            if name in usable:
                bump(name, 10, f"it is in {layout.train} but missing from {layout.test}")
    if layout.submission:
        sub = profile.table(layout.submission)
        if sub is not None:
            for col in sub.columns:
                match = next((n for n in usable if n.lower() == col.name.lower()), None)
                if match:
                    bump(match, 8, "the sample submission asks for it")
    for name, col in usable.items():
        if "target_name" in col.hints:
            bump(name, 5, "its name suggests an outcome")
        if col.distinct == 2:
            bump(name, 1, "it has exactly two values")
    if base.columns and base.columns[-1].name in usable:
        bump(base.columns[-1].name, 2, "it is the last column, a common place for the label")
    ranked = sorted(scores.items(), key=lambda item: (-item[1][0], item[0]))
    return [(name, score, "; ".join(reasons)) for name, (score, reasons) in ranked]


def infer_task(col: ColumnProfile) -> str:
    if col.kind in {"boolean", "categorical"}:
        return "classification"
    if col.kind == "numeric":
        is_float = col.dtype.upper().split("(")[0] in {"DOUBLE", "FLOAT", "REAL", "DECIMAL"}
        return "regression" if is_float or col.distinct > 20 else "classification"
    return "classification"


def default_metric(task: str, distinct: int, minority_share: float | None) -> tuple[str, str]:
    if task == "regression":
        return "rmse", "RMSE measures typical error in the target's own units and punishes big misses."
    if distinct > 2:
        return "macro_f1", "Macro F1 treats every class as equally important, even rare ones."
    if minority_share is not None and minority_share < 0.2:
        return "pr_auc", (
            f"Only {minority_share:.0%} of rows are the rare class. PR AUC focuses on finding "
            "those; accuracy would look good while missing all of them."
        )
    return "roc_auc", "ROC AUC measures how well the model ranks one class above the other."


def _minority_share(stats: dict[str, Any]) -> float | None:
    top = stats.get("top") or []
    total = sum(c for _, c in top)
    if len(top) < 2 or total == 0:
        return None
    return min(c for _, c in top) / total


def _leakage_suspects(
    store: Store, contract: DataContract, base: TableProfile, task: str
) -> list[Suspect]:
    if not contract.target:
        return []
    suspects: list[Suspect] = []
    target = quote_ident(contract.target)
    rel = base_sql(contract)
    target_col = base.column(contract.target)
    target_norm = re.sub(r"[^a-z0-9]", "", contract.target.lower())
    features = [
        c for c in base.columns
        if c.name != contract.target and c.kind in {"categorical", "boolean", "numeric"}
    ][:MAX_LEAKAGE_COLUMNS]
    stats = target_stats(store, contract)
    top = stats.get("top") or []
    total = sum(c for _, c in top) or 1
    baseline = (max(c for _, c in top) / total) if top else 1.0
    target_numeric = target_col is not None and target_col.dtype.upper().split("(")[0] not in {"VARCHAR"}

    for col in features:
        name = quote_ident(col.name)
        if len(target_norm) >= 4 and _names_target(col.name, contract.target or ""):
            suspects.append(Suspect(col.name, f"its name contains the target's name ({contract.target})"))
            continue
        try:
            if task == "classification" and 2 <= col.distinct <= 50 and base.rows >= 30 and baseline < 0.95:
                purity = store.scalar(
                    f"SELECT sum(mx)::DOUBLE / sum(n) FROM (SELECT max(cnt) AS mx, sum(cnt) AS n FROM "
                    f"(SELECT {name} AS c, {target} AS t, count(*) AS cnt FROM {rel} AS b "
                    f"WHERE {target} IS NOT NULL AND {name} IS NOT NULL GROUP BY 1, 2) GROUP BY c)"
                )
                if purity is not None and float(purity) >= 0.995:
                    suspects.append(
                        Suspect(col.name, f"knowing it alone gives the target {float(purity):.1%} of the time")
                    )
                    continue
            if col.kind == "numeric" and target_numeric and base.rows >= 30:
                corr = store.scalar(
                    f"SELECT corr({name}::DOUBLE, {target}::DOUBLE) FROM {rel} AS b"
                )
                if corr is not None and abs(float(corr)) >= 0.98:
                    suspects.append(
                        Suspect(col.name, f"it moves almost exactly with the target (correlation {float(corr):.2f})")
                    )
        except StoreError:
            continue
    return suspects


def candidate_joins(
    layout: Layout, base_name: str, stack: list[str] | None = None
) -> tuple[list[JoinStep], list[str]]:
    """Every join the rules consider safe from ``base_name``: at most one per
    other table, best match first. Also returns tables left out because their
    keys repeat on both sides. A frontend offers these so a person can re-add a
    join an agent removed."""
    stack = stack or []
    excluded = {layout.test, layout.submission, *[n for g in layout.shard_groups for n in g[1:]]}
    best: dict[str, JoinStep] = {}
    skipped_many: list[str] = []
    for rel in layout.relations_for(base_name):
        if rel.right_table in excluded or rel.right_table in stack:
            continue
        if rel.cardinality == "many-to-many":
            skipped_many.append(rel.right_table)
            continue
        if rel.left_in_right < 0.5:
            continue
        kind = "aggregate" if rel.cardinality == "one-to-many" else "lookup"
        step = JoinStep(rel.right_table, kind, rel.left_column, rel.right_column, rel.left_in_right)
        if rel.right_table not in best or step.matched > best[rel.right_table].matched:
            best[rel.right_table] = step
    return list(best.values()), skipped_many


def _words(name: str) -> list[str]:
    """'churnDate' / 'churn_date' / 'Churn Date' -> ['churn', 'date']."""
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    return [w for w in re.split(r"[^a-z0-9]+", spaced.lower()) if w]


def _names_target(column: str, target: str) -> bool:
    """True when the target's words appear as whole words in the column's name
    (churn -> churn_date, is_churned). 'class' inside 'workclass' does not count."""
    words, wanted = _words(column), _words(target)
    if not wanted:
        return False
    for i in range(len(words) - len(wanted) + 1):
        window = words[i:i + len(wanted)]
        if window[:-1] == wanted[:-1] and window[-1].startswith(wanted[-1]):
            return True
    return False


# Names that point at another thing (customer_id, accountRef), not codes or numbers.
_ENTITY_ID = re.compile(r"(?i:(^|_)(id|uuid|guid|ref|reference)$)|[a-z](Id|ID|Ref)$")


def propose_contract(
    store: Store,
    profile: DatasetProfile,
    layout: Layout,
    dataset_hash: str,
    *,
    base_override: str | None = None,
) -> DataContract:
    rationale: dict[str, str] = {}
    if base_override and profile.table(base_override) is not None:
        base_name, rationale["base"] = base_override, "Chosen by you."
    else:
        base_name, rationale["base"] = _pick_base(profile, layout)
    base = profile.table(base_name)
    assert base is not None

    stack: list[str] = []
    for group in layout.shard_groups:
        if base_name in group:
            stack = list(group)
    contract = DataContract(
        dataset_hash=dataset_hash,
        layout=layout.kind,
        base_table=base_name,
        stack=stack,
        test_table=layout.test if layout.train == base_name else None,
    )

    contract.joins, skipped_many = candidate_joins(layout, base_name, stack)
    attach_time_columns(contract.joins, profile)
    if contract.joins:
        rationale["joins"] = (
            "Lookups attach one matching row. Tables with many rows per key are summarised first, "
            "so the row count stays the same and nothing is counted twice."
        )
    if skipped_many:
        rationale["joins"] = (rationale.get("joins", "") + " Not joined (keys repeat on both sides): "
                              + ", ".join(sorted(set(skipped_many))) + ".").strip()
    join_keys = {step.base_key for step in contract.joins}

    # Target, task, metric.
    candidates = _target_candidates(base, profile, layout, join_keys)
    if candidates and candidates[0][1] >= 2:
        name, _score, reason = candidates[0]
        contract.target = name
        rationale["target"] = f"{name}: {reason}."
        others = [c[0] for c in candidates[1:4]]
        if others:
            rationale["target"] += f" Other candidates: {', '.join(others)}."
    else:
        rationale["target"] = "No column clearly looks like the outcome. Pick the one you want to predict."
    target_col = base.column(contract.target) if contract.target else None
    if target_col is not None:
        contract.task = infer_task(target_col)
        stats = target_stats(store, contract)
        distinct = stats.get("distinct", target_col.distinct)
        contract.metric, rationale["metric"] = default_metric(
            contract.task, distinct, _minority_share(stats)
        )
        rationale["task"] = (
            f"{contract.target} has {distinct} distinct values, so this is "
            + ("predicting a category (classification)." if contract.task == "classification"
               else "predicting a number (regression).")
        )

    # Duplicates: rows that are exact copies of another row are dropped up front.
    copies = exact_copies(store, contract)
    if copies:
        contract.dedupe = True
        rationale["dedupe"] = (
            f"{copies:,} rows are exact copies of another row, as from a double export. They are dropped, so "
            "no copy can land in both the training and the test rows and flatter the score."
        )

    # Grain.
    keys = [c for c in base.columns if c.unique and c.kind == "id"]
    if contract.dedupe and not keys:  # an ID that only repeated because of the copies
        rel = base_sql(contract)
        rows_left = int(store.scalar(f"SELECT count(*) FROM {rel} AS b") or 0)
        for c in base.columns:
            if c.kind == "id" and int(store.scalar(
                    f"SELECT count(DISTINCT {quote_ident(c.name)}) FROM {rel} AS b") or 0) == rows_left:
                keys.append(c)
    keys.sort(key=lambda c: ("id_name" not in c.hints, base.column_names.index(c.name)))
    if keys:
        contract.grain_keys = [keys[0].name]
        entity = _entity_from(keys[0].name) or _singular_word(base.name)
        contract.grain = f"one row = one {entity}"
        rationale["grain"] = f"{keys[0].name} is different on every row, so it identifies a row." + (
            " (After dropping the copies.)" if contract.dedupe and not keys[0].unique else "")
    else:
        contract.grain = f"one row = one {_singular_word(base.name)}"
        rationale["grain"] = (
            "No single column identifies a row. Check that rows are not repeated; "
            "duplicates quietly inflate scores."
        )

    # Groups: an ID-named column whose values repeat (several orders of one customer).
    # Rows of one group must stay on one side of a split, or the test rows are not new.
    rel = base_sql(contract)
    rows_now = int(store.scalar(f"SELECT count(*) FROM {rel} AS b") or 0)
    for c in base.columns:
        if not _ENTITY_ID.search(c.name) or c.null_pct > 5 or c.name in contract.grain_keys                 or c.name == contract.target:
            continue
        distinct = int(store.scalar(f"SELECT count(DISTINCT {quote_ident(c.name)}) FROM {rel} AS b") or 0)
        if distinct >= 5 and rows_now >= 1.5 * distinct:
            contract.group_column = c.name
            rationale["group"] = (
                f"{c.name} repeats: {distinct:,} values over {rows_now:,} rows (about {rows_now / distinct:.1f} "
                f"each). All rows of one {c.name} are kept on the same side of every split, so the model is "
                "tested on ones it has never seen."
            )
            break

    # Time.
    time_cols = [c for c in base.columns if c.kind == "datetime"]
    if time_cols:
        contract.time_column = time_cols[0].name
        rationale["time"] = (
            f"{time_cols[0].name} is a date. Splitting by time later stops the model from "
            "learning from the future."
        )

    # Columns kept out of the features.
    drops: list[str] = []
    reasons: list[str] = []
    for col in base.columns:
        if col.name in {contract.target, contract.time_column}:
            continue
        if col.kind == "id":
            drops.append(col.name)
            reasons.append(f"{col.name} (identifier)")
        elif col.kind in {"constant", "empty"}:
            drops.append(col.name)
            reasons.append(f"{col.name} (never changes)")
        elif col.null_pct >= 95:
            drops.append(col.name)
            reasons.append(f"{col.name} ({col.null_pct:.0f}% missing)")
        elif col.kind == "text":
            drops.append(col.name)
            reasons.append(f"{col.name} (free text, not handled yet)")
    if contract.target and contract.task:
        contract.leakage_suspects = _leakage_suspects(store, contract, base, contract.task)
        for suspect in contract.leakage_suspects:
            if suspect.column not in drops:
                drops.append(suspect.column)
                reasons.append(f"{suspect.column} (possible leakage)")
    contract.drop_columns = drops
    if reasons:
        rationale["drops"] = ", ".join(reasons) + "."
    contract.rationale = rationale
    return contract


def retarget(
    store: Store, profile: DatasetProfile, contract: DataContract, target: str | None
) -> DataContract:
    """Return a copy of ``contract`` with a different target. Task, metric and
    the leakage scan all depend on the target, so they are redone."""
    new = copy.deepcopy(contract)
    base = profile.table(new.base_table)
    old_suspects = {s.column for s in new.leakage_suspects}
    new.target = target
    new.drop_columns = [c for c in new.drop_columns if c != target and c not in old_suspects]
    new.leakage_suspects = []
    col = base.column(target) if base is not None and target else None
    if base is None or col is None:
        new.task = None
        new.metric = None
        return new
    new.task = infer_task(col)
    stats = target_stats(store, new)
    distinct = stats.get("distinct", col.distinct)
    new.metric, metric_why = default_metric(new.task, distinct, _minority_share(stats))
    new.rationale = {
        **new.rationale,
        "target": f"{target}: chosen by you.",
        "task": f"{target} has {distinct} distinct values, so this is "
                + ("predicting a category (classification)." if new.task == "classification"
                   else "predicting a number (regression)."),
        "metric": metric_why,
    }
    new.leakage_suspects = _leakage_suspects(store, new, base, new.task)
    for suspect in new.leakage_suspects:
        if suspect.column not in new.drop_columns:
            new.drop_columns.append(suspect.column)
    return new


# ---------------------------------------------------------------------------
# Validation: the harness checks, it does not trust.
# ---------------------------------------------------------------------------


def validate_contract(store: Store, profile: DatasetProfile, contract: DataContract) -> list[Issue]:
    issues: list[Issue] = []
    base = profile.table(contract.base_table)
    if base is None:
        return [Issue("error", "base", f"Table not found: {contract.base_table}")]
    for name in contract.stack:
        if profile.table(name) is None:
            issues.append(Issue("error", "base", f"Stacked table not found: {name}"))
    if any(i.level == "error" for i in issues):
        return issues
    columns = set(base.column_names)
    rel = base_sql(contract)
    base_rows = int(store.scalar(f"SELECT count(*) FROM {rel} AS b") or 0)
    if contract.dedupe:
        dropped = exact_copies(store, contract)
        issues.append(Issue("info", "dedupe", f"{dropped:,} exact copy rows dropped; {base_rows:,} rows left."))
    if base_rows == 0:
        issues.append(Issue("error", "base", "The base table has no rows."))
        return issues
    if base_rows < 50:
        issues.append(Issue("warning", "base", f"Only {base_rows} rows. Any score will be unstable."))

    # Target.
    if not contract.target:
        issues.append(Issue("error", "target", "No target chosen. Pick the column to predict."))
    elif contract.target not in columns:
        issues.append(Issue("error", "target", f"{contract.target} is not a column of {base.name}."))
    else:
        stats = target_stats(store, contract)
        if stats.get("distinct", 0) <= 1:
            issues.append(Issue("error", "target", f"{contract.target} has only one value. Nothing to learn."))
        if stats.get("nulls", 0):
            issues.append(Issue(
                "warning", "target",
                f"{stats['nulls']:,} rows have no {contract.target}. They will be left out of training.",
            ))
        if contract.target in contract.drop_columns:
            issues.append(Issue("error", "drops", "The target is in the excluded columns."))
        if contract.task not in TASKS:
            issues.append(Issue("error", "task", "Choose classification or regression."))
        target_col = base.column(contract.target)
        if contract.task == "regression" and target_col is not None and target_col.dtype.upper() == "VARCHAR":
            issues.append(Issue("error", "task", f"{contract.target} is text, so it cannot be a regression target."))
        if contract.task == "classification" and stats.get("distinct", 0) > 50:
            issues.append(Issue(
                "warning", "task",
                f"{contract.target} has {stats['distinct']} classes. That looks like a number or an ID.",
            ))
        if contract.task == "classification":
            share = _minority_share(stats)
            if share is not None and share < 0.05 and contract.metric == "accuracy":
                issues.append(Issue(
                    "warning", "metric",
                    f"The rare class is {share:.1%} of rows. Accuracy will look high while missing it.",
                ))
        if contract.task in METRICS and contract.metric not in METRICS[contract.task]:
            issues.append(Issue("error", "metric", f"{contract.metric} does not fit {contract.task}."))

    # Grain.
    missing_keys = [k for k in contract.grain_keys if k not in columns]
    if missing_keys:
        issues.append(Issue("error", "grain", f"Key column not found: {', '.join(missing_keys)}"))
    elif contract.grain_keys:
        key_list = ", ".join(quote_ident(k) for k in contract.grain_keys)
        distinct = int(store.scalar(
            f"SELECT count(*) FROM (SELECT DISTINCT {key_list} FROM {rel} AS b)"
        ) or 0)
        if distinct < base_rows:
            issues.append(Issue(
                "error", "grain",
                f"{', '.join(contract.grain_keys)} repeats: {base_rows - distinct:,} extra rows. "
                "Either the key is wrong or the data has duplicates.",
            ))
    else:
        distinct_rows = int(store.scalar(f"SELECT count(*) FROM (SELECT DISTINCT * FROM {rel} AS b)") or 0)
        if distinct_rows < base_rows:
            issues.append(Issue(
                "warning", "grain",
                f"{base_rows - distinct_rows:,} rows are exact copies of another row.",
            ))
        else:
            issues.append(Issue("warning", "grain", "No key column chosen, so repeated entities cannot be detected."))

    if contract.time_column and contract.time_column not in columns:
        issues.append(Issue("error", "time", f"Time column not found: {contract.time_column}"))
    unknown_drops = [c for c in contract.drop_columns if c not in columns]
    if unknown_drops:
        issues.append(Issue("error", "drops", f"Unknown columns: {', '.join(unknown_drops)}"))

    # Joins: prove the row count did not change.
    join_ok = True
    for step in contract.joins:
        other = profile.table(step.table)
        if other is None or step.other_key not in other.column_names:
            issues.append(Issue("error", "joins", f"Join target not found: {step.table}.{step.other_key}"))
            join_ok = False
        elif step.base_key not in columns:
            issues.append(Issue("error", "joins", f"{step.base_key} is not a column of {base.name}."))
            join_ok = False
    if contract.joins and join_ok:
        try:
            joined = int(store.scalar(
                f"SELECT count(*) FROM ({model_input_sql(contract, profile)})"
            ) or 0)
        except StoreError as exc:
            issues.append(Issue("error", "joins", f"The join plan failed to run: {exc}"))
        else:
            if joined != base_rows:
                issues.append(Issue(
                    "error", "joins",
                    f"Joining changed the row count from {base_rows:,} to {joined:,}. "
                    "A lookup table has repeated keys; it must be summarised first.",
                ))
            else:
                issues.append(Issue("info", "joins", f"Row count checked: {base_rows:,} before and after joining."))

    # Time rule: which rows of a dated, summarised table exist at the prediction moment.
    if contract.horizon_days is not None:
        if not isinstance(contract.horizon_days, int) or contract.horizon_days < 0:
            issues.append(Issue("error", "horizon", "The prediction moment must be a whole number of days, 0 or more."))
        elif not contract.time_column:
            issues.append(Issue("error", "horizon", "Choose a time column: the prediction moment is counted from it."))
    horizon_ok = not any(i.field == "horizon" for i in issues)
    for step in contract.joins if join_ok and horizon_ok else []:
        other = profile.table(step.table)
        if step.kind != "aggregate" or not step.time_column or other is None:
            continue
        if step.time_column not in other.column_names:
            issues.append(Issue("error", "joins", f"{step.table} has no column {step.time_column}."))
            continue
        safe = time_safe_child_sql(contract, step)
        if safe is None:
            issues.append(Issue(
                "warning", "horizon",
                f"{step.table} is dated by {step.time_column} but no prediction moment is set, so all "
                f"{other.rows:,} rows are used, including any from after the outcome.",
            ))
            continue
        try:
            usable = int(store.scalar(f"SELECT count(*) FROM ({safe})") or 0)
        except StoreError as exc:
            issues.append(Issue("error", "horizon", f"The time rule failed for {step.table}: {exc}"))
            continue
        level = "warning" if usable == 0 else "info"
        issues.append(Issue(
            level, "horizon",
            f"{step.table}: {usable:,} of {other.rows:,} rows exist {contract.horizon_days} days after "
            f"{contract.time_column}. Later rows are hidden from every feature.",
        ))

    # Test split.
    if contract.test_table:
        test = profile.table(contract.test_table)
        if test is None:
            issues.append(Issue("error", "test", f"Test table not found: {contract.test_table}"))
        else:
            test_cols = {c.lower() for c in test.column_names}
            excluded = {contract.target, *contract.drop_columns}
            missing = [c for c in base.column_names if c not in excluded and c.lower() not in test_cols]
            if missing:
                issues.append(Issue(
                    "warning", "test",
                    f"{test.name} lacks columns the model would use: {', '.join(missing[:6])}",
                ))

    # Questions the team could not settle from the data need a person's answer.
    for question in contract.open_questions:
        if not contract.answers.get(question, "").strip():
            short = question if len(question) <= 90 else question[:87] + "..."
            issues.append(Issue("error", QUESTIONS_FIELD, f"Unanswered question: {short}"))
    hint = horizon_from_answers(contract)
    if hint is not None and contract.horizon_days is None and contract.joins and contract.time_column:
        issues.append(Issue("warning", "horizon",
                            f"Your answer says the prediction is made {hint} days after {contract.time_column}, "
                            f"but the prediction moment is not set. Approving sets it to {hint} days."))
    issues += _check_extras(store, contract, columns, base_rows)
    return issues


_DELAY = re.compile(r"(\d+)\s*(day|week|month)s?\s+(after|from|since|past)", re.IGNORECASE)


def horizon_from_answers(contract: DataContract) -> int | None:
    """'made 30 days after signup' -> 30. Only a clear 'N days/weeks/months after' counts."""
    for answer in contract.answers.values():
        match = _DELAY.search(answer or "")
        if match:
            n, unit = int(match.group(1)), match.group(2).lower()
            return n * {"day": 1, "week": 7, "month": 30}[unit]
    return None


def _check_extras(store: Store, contract: DataContract, columns: set[str], base_rows: int) -> list[Issue]:
    issues: list[Issue] = []
    if contract.group_column:
        if contract.group_column not in columns:
            issues.append(Issue("error", "group", f"{contract.group_column} is not a column of {contract.base_table}."))
        elif contract.group_column == contract.target:
            issues.append(Issue("error", "group", "The group column cannot be the target."))
        else:
            groups = int(store.scalar(
                f"SELECT count(DISTINCT b.{quote_ident(contract.group_column)}) FROM {base_sql(contract)} AS b") or 0)
            if groups < 5:
                issues.append(Issue("error", "group",
                                    f"{contract.group_column} has only {groups} values: too few groups to split on."))
            elif groups >= base_rows:
                issues.append(Issue("warning", "group", f"Every {contract.group_column} has one row, so grouping "
                                                       "changes nothing."))
            else:
                issues.append(Issue("info", "group", f"{groups:,} {contract.group_column} values, about "
                                                     f"{base_rows / groups:.1f} rows each, kept on one side of "
                                                     "every split."))
    for column in contract.fairness_columns:
        if column not in columns:
            issues.append(Issue("warning", "fairness", f"Fairness column {column} is not in {contract.base_table}."))
    costs = (contract.cost_false_alarm, contract.cost_miss)
    if (costs[0] is None) != (costs[1] is None):
        issues.append(Issue("error", "cost", "Give both costs (a false alarm and a miss), or neither."))
    elif costs[0] is not None:
        if contract.task != "classification":
            issues.append(Issue("warning", "cost", "Costs only apply to yes/no predictions; they will be ignored."))
        elif costs[0] <= 0 or costs[1] <= 0:  # type: ignore[operator]
            issues.append(Issue("error", "cost", "Costs must be above zero."))
    return issues


def has_errors(issues: list[Issue]) -> bool:
    return any(i.level == "error" for i in issues)


# ---------------------------------------------------------------------------
# Persistence: every approval is a new version.
# ---------------------------------------------------------------------------


def next_version(run: RunPaths) -> int:
    version = 1
    while run.contract_version(version).exists():
        version += 1
    return version


def save_contract(run: RunPaths, contract: DataContract, approved_by: str) -> Path:
    contract.version = next_version(run)
    contract.approved_by = approved_by
    contract.approved_at = now_iso()
    path = run.contract_version(contract.version)
    write_json(path, contract.to_json())
    write_json(run.contract_latest, contract.to_json())
    return path
