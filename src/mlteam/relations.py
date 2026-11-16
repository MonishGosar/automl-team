"""Work out how several tables fit together.

"A bunch of CSVs" is one of three different problems:

* shards      same columns split across files      -> stack them
* related     different tables sharing a key       -> join them
* train/test  a Kaggle-style split                 -> keep the roles

The decision is made from the data (column sets, key uniqueness, how many of
one table's key values exist in the other), not from file names alone.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from mlteam.profile import ColumnProfile, DatasetProfile, TableProfile, is_id_name, type_class
from mlteam.store import Store, quote_ident

MIN_CONTAINMENT = 0.8
MAX_TABLES_FOR_PAIRING = 40


@dataclass
class Relation:
    left_table: str
    left_column: str
    right_table: str
    right_column: str
    left_in_right: float
    right_in_left: float
    left_unique: bool
    right_unique: bool
    cardinality: str

    def describe(self) -> str:
        return (
            f"{self.left_table}.{self.left_column} -> {self.right_table}.{self.right_column} "
            f"({self.cardinality}, {self.left_in_right:.0%} of keys matched)"
        )


@dataclass
class Layout:
    kind: str
    train: str | None = None
    test: str | None = None
    submission: str | None = None
    train_only_columns: list[str] = field(default_factory=list)
    shard_groups: list[list[str]] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    def relations_for(self, table: str) -> list[Relation]:
        """Relations touching ``table``, oriented so ``table`` is on the left."""
        out: list[Relation] = []
        for rel in self.relations:
            if rel.left_table == table:
                out.append(rel)
            elif rel.right_table == table:
                out.append(flip(rel))
        return out


def flip(rel: Relation) -> Relation:
    swap = {"many-to-one": "one-to-many", "one-to-many": "many-to-one"}
    return Relation(
        left_table=rel.right_table,
        left_column=rel.right_column,
        right_table=rel.left_table,
        right_column=rel.left_column,
        left_in_right=rel.right_in_left,
        right_in_left=rel.left_in_right,
        left_unique=rel.right_unique,
        right_unique=rel.left_unique,
        cardinality=swap.get(rel.cardinality, rel.cardinality),
    )


def _tokens(name: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", name.lower()) if t]


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _singular(name: str) -> str:
    base = _norm(name)
    if base.endswith("ies"):
        return base[:-3] + "y"
    if base.endswith("ses"):
        return base[:-2]
    if base.endswith("s") and not base.endswith("ss"):
        return base[:-1]
    return base


def _key_unique(col: ColumnProfile) -> bool:
    nonnull = col.rows - col.nulls
    return nonnull > 0 and col.distinct == nonnull


def _joinable(col: ColumnProfile) -> bool:
    return col.kind in {"id", "categorical", "numeric"} and type_class(col.dtype) in {"integer", "string"}


def _find_roles(profile: DatasetProfile) -> tuple[str | None, str | None, str | None]:
    trains = [t.name for t in profile.tables if "train" in _tokens(t.name)]
    tests = [t.name for t in profile.tables if "test" in _tokens(t.name)]
    subs = [t.name for t in profile.tables if "submission" in t.name.lower()]
    train = trains[0] if len(trains) == 1 else None
    test = tests[0] if len(tests) == 1 else None
    if train is None or test is None:
        return None, None, None
    return train, test, (subs[0] if len(subs) == 1 else None)


def _candidate_pairs(a: TableProfile, b: TableProfile) -> list[tuple[ColumnProfile, ColumnProfile]]:
    pairs: list[tuple[ColumnProfile, ColumnProfile]] = []
    b_by_norm = {_norm(c.name): c for c in b.columns}
    a_by_norm = {_norm(c.name): c for c in a.columns}
    seen: set[tuple[str, str]] = set()

    def add(ca: ColumnProfile, cb: ColumnProfile) -> None:
        key = (ca.name, cb.name)
        if key in seen or not (_joinable(ca) and _joinable(cb)):
            return
        if not (_key_unique(ca) or _key_unique(cb) or (is_id_name(ca.name) and is_id_name(cb.name))):
            return
        seen.add(key)
        pairs.append((ca, cb))

    for norm, ca in a_by_norm.items():
        if norm in b_by_norm:
            add(ca, b_by_norm[norm])
    # orders.customer_id <-> customers.id
    for owner, other, flip_pair in ((a, b, False), (b, a, True)):
        fk = _singular(other.name) + "id"
        owner_cols = {_norm(c.name): c for c in owner.columns}
        other_cols = {_norm(c.name): c for c in other.columns}
        if fk in owner_cols and "id" in other_cols:
            ca, cb = owner_cols[fk], other_cols["id"]
            add(cb, ca) if flip_pair else add(ca, cb)
    return pairs


def _containment(store: Store, a: str, ca: str, b: str, cb: str) -> tuple[float, float]:
    sql = (
        f"WITH l AS (SELECT DISTINCT {quote_ident(ca)}::VARCHAR AS k FROM {quote_ident(a)} "
        f"WHERE {quote_ident(ca)} IS NOT NULL), "
        f"r AS (SELECT DISTINCT {quote_ident(cb)}::VARCHAR AS k FROM {quote_ident(b)} "
        f"WHERE {quote_ident(cb)} IS NOT NULL) "
        "SELECT (SELECT count(*) FROM l), (SELECT count(*) FROM r), "
        "(SELECT count(*) FROM l JOIN r USING (k))"
    )
    _, rows = store.query(sql)
    left, right, both = (int(v or 0) for v in rows[0])
    return (both / left if left else 0.0, both / right if right else 0.0)


def _cardinality(left_unique: bool, right_unique: bool) -> str:
    if left_unique and right_unique:
        return "one-to-one"
    if right_unique:
        return "many-to-one"
    if left_unique:
        return "one-to-many"
    return "many-to-many"


def detect_layout(store: Store, profile: DatasetProfile) -> Layout:
    tables = profile.tables
    if len(tables) == 1:
        return Layout(kind="single")

    layout = Layout(kind="unrelated")
    train, test, submission = _find_roles(profile)
    layout.train, layout.test, layout.submission = train, test, submission
    role_tables = {name for name in (train, test, submission) if name}

    if train and test:
        train_cols = {c.lower(): c for c in profile.table(train).column_names}  # type: ignore[union-attr]
        test_cols = {c.lower() for c in profile.table(test).column_names}  # type: ignore[union-attr]
        layout.train_only_columns = [orig for low, orig in train_cols.items() if low not in test_cols]
        extra = sorted(test_cols - set(train_cols))
        if extra:
            layout.notes.append(f"{test} has columns that {train} lacks: {', '.join(extra)}")
        if not layout.train_only_columns:
            layout.notes.append(
                f"{train} and {test} have the same columns, so the target cannot be inferred from the split."
            )

    # Shards: identical column sets outside the train/test/submission roles.
    groups: dict[frozenset[str], list[str]] = {}
    for table in tables:
        if table.name in role_tables:
            continue
        groups.setdefault(frozenset(c.lower() for c in table.column_names), []).append(table.name)
    layout.shard_groups = [sorted(names) for names in groups.values() if len(names) >= 2]
    shard_followers = {name for group in layout.shard_groups for name in group[1:]}

    # Relations between logical tables.
    logical = [
        t for t in tables
        if t.name not in shard_followers and t.name not in {test, submission}
    ]
    if len(logical) > MAX_TABLES_FOR_PAIRING:
        layout.notes.append(
            f"{len(logical)} tables: relation search limited to the first {MAX_TABLES_FOR_PAIRING}."
        )
        logical = logical[:MAX_TABLES_FOR_PAIRING]
    for i, a in enumerate(logical):
        for b in logical[i + 1:]:
            for ca, cb in _candidate_pairs(a, b):
                left_in_right, right_in_left = _containment(store, a.name, ca.name, b.name, cb.name)
                if max(left_in_right, right_in_left) < MIN_CONTAINMENT:
                    continue
                layout.relations.append(
                    Relation(
                        left_table=a.name,
                        left_column=ca.name,
                        right_table=b.name,
                        right_column=cb.name,
                        left_in_right=round(left_in_right, 4),
                        right_in_left=round(right_in_left, 4),
                        left_unique=_key_unique(ca),
                        right_unique=_key_unique(cb),
                        cardinality=_cardinality(_key_unique(ca), _key_unique(cb)),
                    )
                )

    for rel in layout.relations:
        if rel.cardinality == "many-to-many":
            layout.notes.append(
                f"{rel.left_table}.{rel.left_column} and {rel.right_table}.{rel.right_column} "
                "repeat on both sides. Joining them directly would multiply rows."
            )

    non_role = [t.name for t in tables if t.name not in role_tables]
    if train and test:
        layout.kind = "train_test"
    elif len(layout.shard_groups) == 1 and sorted(non_role) == layout.shard_groups[0]:
        layout.kind = "shards"
    elif layout.relations:
        layout.kind = "related"
    else:
        layout.kind = "unrelated"
        layout.notes.append("No shared key found between the tables. Pick the table to model by hand.")
    return layout
