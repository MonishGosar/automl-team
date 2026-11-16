"""Understand the data before training on it, and let what is found change the training.

Run on the drafted features and the target, on the training rows only (the
locked test rows never inform a decision). Each finding is a plain sentence,
and several lead to a concrete change:

* Signal       which features tell most about the target, measured by mutual
               information: how much knowing the feature narrows down the answer.
               Shown as a ranking; also handed to the Engineer.
* Skew         a number with a long tail (a few huge values). Linear models are
               pulled around by those; a log version is added as a feature.
* Redundancy   two features that move together almost perfectly. One is enough;
               the weaker one is suggested for switching off.
* Missingness  when a value being missing is itself linked to the outcome. Kept
               as a signal (the training step adds a "was missing" flag).
* Labels       the same category written several ways ("North " and "north"), or
               a placeholder such as "unknown" posing as a real value. A cleaned
               version is added and the raw one switched off.
* Duplicates   identical rows. Copies can land in both the training and test
               rows, which inflates the score. Counted and reported.
* Balance      a rare outcome, and what that means for the metric.
* Size         rows per feature, and what that means for how complex a model
               can be trusted. Handed to the training plan.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy import stats

from mlteam.contract import DataContract
from mlteam.features import Feature
from mlteam.store import Store
from mlteam.train import _load

PLACEHOLDERS = {"unknown", "n/a", "na", "none", "null", "-", "?", "missing", "not available", "nan"}
SKEW_LIMIT = 2.0
REDUNDANT = 0.95
MIN_MISSING_SHARE = 0.05
RARE_CLASS = 0.20


@dataclass
class Finding:
    kind: str          # signal | skew | redundant | missing | duplicates | balance | size | weak
    severity: str      # info | warn
    text: str
    features: list[str] = field(default_factory=list)
    action: str = ""   # what was done or suggested because of it


@dataclass
class DataInsights:
    rows: int
    features: int
    task: str
    target: str
    signal: list[tuple[str, float]] = field(default_factory=list)   # feature, mutual information (higher = more)
    findings: list[Finding] = field(default_factory=list)
    new_features: list[Feature] = field(default_factory=list)       # added because of findings
    switch_off: dict[str, str] = field(default_factory=dict)        # feature -> why it should be off by default
    advice: list[str] = field(default_factory=list)                 # for the training plan

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    def text_for_agents(self) -> str:
        lines = [f"{self.rows:,} training rows, {self.features} features, task {self.task}."]
        if self.signal:
            top = ", ".join(f"{n} ({v:.3f})" for n, v in self.signal[:8])
            lines.append(f"Signal ranking by mutual information with {self.target}: {top}.")
        lines += [f"- {f.text}" + (f" Action: {f.action}" if f.action else "") for f in self.findings
                  if f.kind != "signal"]
        lines += [f"Advice: {a}" for a in self.advice]
        return "\n".join(lines)


def _numeric_column(column: np.ndarray) -> np.ndarray:
    return np.array([np.nan if v is None else float(v) for v in column], dtype=float)


def _codes(column: np.ndarray, keep: int = 20) -> np.ndarray:
    """Categories as integer codes for mutual information, the way training sees them:
    the most common values each get a code, the rest share one. Hundreds of rare codes
    would otherwise look highly informative by pure chance."""
    labels = ["\x00missing" if v is None else str(v) for v in column]
    counts: dict[str, int] = {}
    for k in labels:
        counts[k] = counts.get(k, 0) + 1
    common = set(sorted(counts, key=lambda k: -counts[k])[:keep])
    labels = [k if k in common else "\x00other" for k in labels]
    index = {k: i for i, k in enumerate(sorted(set(labels)))}
    return np.array([index[k] for k in labels], dtype=float)


def _mutual_information(x: np.ndarray, y: np.ndarray, kinds: list[str], classify: bool) -> list[float]:
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

    columns, discrete = [], []
    for i, kind in enumerate(kinds):
        if kind in {"numeric", "boolean"}:
            col = _numeric_column(x[:, i])
            fill = np.nanmedian(col) if np.isfinite(col).any() else 0.0
            columns.append(np.where(np.isnan(col), fill, col))
            discrete.append(kind == "boolean" or len(np.unique(col[~np.isnan(col)])) <= 10)
        else:
            columns.append(_codes(x[:, i]))
            discrete.append(True)
    matrix = np.column_stack(columns) if columns else np.empty((len(y), 0))
    fn = mutual_info_classif if classify else mutual_info_regression
    return [float(v) for v in fn(matrix, y, discrete_features=np.array(discrete), random_state=0)]


def understand(store: Store, contract: DataContract, features: list[Feature]) -> DataInsights:
    usable = [f for f in features if f.usable]
    data = _load(store, contract, usable)
    x, y = data.x_train, data.y_train
    task = contract.task or ""
    classify = task == "classification"
    kinds = [f.kind for f in usable]
    names = [f.name for f in usable]
    result = DataInsights(rows=len(y), features=len(usable), task=task, target=contract.target or "")

    # Signal.
    try:
        mi = _mutual_information(x, y, kinds, classify)
        result.signal = sorted(zip(names, mi), key=lambda item: -item[1])
    except Exception:  # noqa: BLE001 - the rest of the findings still stand
        mi = []
    if result.signal:
        top_name, top_value = result.signal[0]
        weak = [n for n, v in result.signal if v < 0.005]
        result.findings.append(Finding(
            "signal", "info",
            f"{top_name} tells the most about {result.target}. "
            + (f"{len(weak)} feature(s) show almost no link on their own: {', '.join(weak[:5])}."
               if weak else "Every feature shows some link to the target."),
            [top_name, *weak],
        ))
        if top_value < 0.01:
            result.findings.append(Finding(
                "weak", "warn",
                "No feature on its own says much about the target. Expect the models to struggle; a better "
                "feature, or different data, will matter more than tuning.",
            ))

    # Skew and outliers.
    for i, f in enumerate(usable):
        if f.kind != "numeric":
            continue
        col = _numeric_column(x[:, i])
        values = col[~np.isnan(col)]
        if len(values) < 20 or len(np.unique(values)) <= 10:
            continue
        skew = float(stats.skew(values))
        q1, q3 = np.percentile(values, [25, 75])
        far = float(np.mean((values > q3 + 3 * (q3 - q1)) | (values < q1 - 3 * (q3 - q1)))) if q3 > q1 else 0.0
        if skew > SKEW_LIMIT and values.min() >= 0:
            added = log_feature(f)
            name = added.name
            result.findings.append(Finding(
                "skew", "info",
                f"{f.name} has a long tail (skew {skew:.1f}; {_share(far)} of values are extreme). A few huge values "
                "pull linear models around; trees do not mind.",
                [f.name],
                f"Added {name}, the log of it, which spreads the values out.",
            ))
            result.new_features.append(added)
        elif far >= 0.02:
            result.findings.append(Finding(
                "skew", "info",
                f"{f.name} has extreme values on {far:.1%} of rows. Worth checking they are real, not typos.",
                [f.name],
            ))

    # Redundancy: groups of numeric features that rank-correlate almost perfectly.
    # Keep the one with the strongest link to the target in each group; suggest the rest off.
    numeric = [i for i, f in enumerate(usable) if f.kind in {"numeric", "boolean"}]
    strength = dict(result.signal)
    parent = {i: i for i in numeric}

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    links: dict[tuple[int, int], float] = {}
    for a_pos, a in enumerate(numeric):
        for b in numeric[a_pos + 1:]:
            ca, cb = _numeric_column(x[:, a]), _numeric_column(x[:, b])
            both = ~np.isnan(ca) & ~np.isnan(cb)
            if both.sum() < 20 or np.std(ca[both]) == 0 or np.std(cb[both]) == 0:
                continue
            rho = float(stats.spearmanr(ca[both], cb[both]).statistic)
            if abs(rho) >= REDUNDANT:
                links[(a, b)] = rho
                parent[root(a)] = root(b)
    groups: dict[int, list[int]] = {}
    for i in numeric:
        groups.setdefault(root(i), []).append(i)
    for members in groups.values():
        if len(members) < 2:
            continue
        member_names = [names[i] for i in members]
        keeper = max(member_names, key=lambda n: strength.get(n, 0.0))
        others = [n for n in member_names if n != keeper]
        lowest = min(abs(v) for (a, b), v in links.items() if a in members and b in members)
        for other in others:
            result.switch_off[other] = f"Moves almost exactly with {keeper}, which tells more about the target."
        listed = ", ".join(member_names)
        result.findings.append(Finding(
            "redundant", "info",
            f"{listed} move together almost perfectly ("
            + ("rank correlation 1.00" if lowest >= 0.995 else f"rank correlations of {lowest:.2f} or more")
            + "). One is enough.",
            member_names,
            f"Keeping {keeper}, the one most linked to the target; suggesting {', '.join(others)} off.",
        ))

    # No log version of a feature that is being switched off as a near-copy.
    for f in list(result.new_features):
        source = f.name[: -len("__log")] if f.name.endswith("__log") else None
        if source in result.switch_off:
            result.new_features.remove(f)
            for finding in result.findings:
                if finding.kind == "skew" and finding.features == [source]:
                    finding.action = "No log version added: it is being switched off as a near-copy."

    # Labels written several ways, and placeholders posing as values.
    for i, f in enumerate(usable):
        if f.kind != "categorical":
            continue
        raw = sorted({str(v) for v in x[:, i] if v is not None})
        groups: dict[str, list[str]] = {}
        for value in raw:
            groups.setdefault(value.strip().lower(), []).append(value)
        variants = [g for k, g in groups.items() if len(g) > 1 and k not in PLACEHOLDERS]
        placeholders = sorted(v for v in raw if v.strip().lower() in PLACEHOLDERS)
        if not variants and not placeholders:
            continue
        parts = []
        if variants:
            shown = "; ".join(" / ".join(repr(v) for v in g) for g in variants[:3])
            parts.append(f"the same label is written several ways ({shown})")
        if placeholders:
            share = float(np.mean([v is not None and str(v) in placeholders for v in x[:, i]]))
            parts.append(f"{', '.join(repr(p) for p in placeholders)} on {share:.0%} of rows looks like a "
                         "placeholder for 'not known' rather than a real value")
        cleaned = clean_feature(f, has_placeholders=bool(placeholders))
        result.new_features.append(cleaned)
        result.switch_off[f.name] = f"Replaced by {cleaned.name}, the cleaned version."
        result.findings.append(Finding(
            "labels", "warn", f"In {f.name}, " + " and ".join(parts) + ".", [f.name],
            f"Added {cleaned.name} (trimmed, lower case" + (", placeholders as missing" if placeholders else "")
            + f") and switched {f.name} off.",
        ))

    # Missingness that carries signal.
    for i, f in enumerate(usable):
        column = x[:, i]
        missing = np.array([v is None or (isinstance(v, float) and math.isnan(v)) for v in column])
        share = float(missing.mean())
        if share < MIN_MISSING_SHARE or share > 0.95:
            continue
        if classify:
            table = np.array([[np.sum((y == k) & missing), np.sum((y == k) & ~missing)] for k in np.unique(y)])
            if (table.sum(axis=0) == 0).any():
                continue
            p = float(stats.chi2_contingency(table)[1])
            rate_missing = float(np.mean(y[missing] == y.max()))
            rate_present = float(np.mean(y[~missing] == y.max()))
            detail = f"the outcome rate is {rate_missing:.0%} when it is missing against {rate_present:.0%} when not"
        else:
            p = float(stats.mannwhitneyu(y[missing], y[~missing]).pvalue)
            detail = (f"the average target is {np.mean(y[missing]):,.2f} when it is missing against "
                      f"{np.mean(y[~missing]):,.2f} when not")
        if p < 0.01:
            result.findings.append(Finding(
                "missing", "info",
                f"{f.name} is missing on {share:.0%} of rows, and that is itself a clue: {detail}.",
                [f.name],
                "Kept as a signal: training adds a 'was missing' flag next to the filled-in value.",
            ))

    # Duplicates among the training rows.
    keys = [tuple("" if v is None else str(v) for v in row) for row in x]
    counts: dict[tuple, int] = {}
    for k in keys:
        counts[k] = counts.get(k, 0) + 1
    extra = sum(c - 1 for c in counts.values() if c > 1)
    if extra:
        share = extra / len(keys)
        result.findings.append(Finding(
            "duplicates", "warn" if share >= 0.01 else "info",
            f"{extra:,} training rows ({_share(share)}) repeat another row's features exactly. If they are copies, "
            "a copy in the test rows makes the score look better than it is.",
            action="Check the row key in the contract; repeated rows usually mean a join or an export went wrong.",
        ))

    # Balance.
    if classify:
        counts_y = np.bincount(y)
        share = counts_y.min() / counts_y.sum()
        if share < RARE_CLASS:
            result.findings.append(Finding(
                "balance", "warn",
                f"The rarest outcome is {share:.0%} of rows. Accuracy would look high while missing it, so the "
                f"contract's metric ({contract.metric}) matters more than usual.",
            ))

    # Size.
    per_feature = len(y) / max(len(usable), 1)
    if per_feature < 20:
        advice = (f"Only {per_feature:.0f} training rows per feature. Prefer simple models (linear, shallow trees) "
                  "and keep forests and boosting shallow; deep models will memorise.")
        result.advice.append(advice)
        result.findings.append(Finding("size", "warn", advice))
    elif len(y) < 1000:
        result.advice.append(f"{len(y):,} training rows: moderate settings; very deep models are unlikely to pay off.")
    else:
        result.advice.append(f"{len(y):,} training rows: there is room for forests and boosting to show their strength.")
    return result


def clean_feature(source: Feature, *, has_placeholders: bool) -> Feature:
    text = f"lower(trim(CAST(({source.sql}) AS VARCHAR)))"
    if has_placeholders:
        listed = ", ".join("'" + p.replace("'", "''") + "'" for p in sorted(PLACEHOLDERS))
        text = f"CASE WHEN {text} IN ({listed}) THEN NULL ELSE {text} END"
    return Feature(
        name=f"{source.name}__clean"[:60], sql=text, origin="rules",
        why=f"{source.name} with labels tidied: spaces trimmed and case unified"
            + (", and placeholders such as 'unknown' treated as missing." if has_placeholders else "."),
    )


def log_feature(source: Feature) -> Feature:
    return Feature(
        name=f"{source.name}__log"[:60],
        sql=f"ln(1 + greatest(({source.sql}), 0))",
        why=f"The log of {source.name}, which has a long tail. It spreads out the few huge values.",
        origin="rules",
    )


def _share(value: float) -> str:
    """0.0004 -> 'under 0.1%': a share that rounds to zero is still not zero."""
    return "under 0.1%" if 0 < value < 0.001 else f"{value:.1%}"
