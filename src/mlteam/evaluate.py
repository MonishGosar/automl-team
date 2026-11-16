"""What a real team checks before a model goes out, beyond one score.

Everything that is *chosen* here (the cut-off, whether to calibrate, how wide a
prediction interval is) is chosen on cross-validated predictions for the
training rows. The unseen test rows are only used to report how those choices
hold up, never to make them.

* Cut-off       a yes/no model gives a probability; something has to turn it
                into a decision. 0.5 is rarely right when the outcome is rare.
                Picked for the best F1, or the lowest cost when the contract
                says what a false alarm and a miss cost.
* Calibration   does "70%" mean 70%? Checked in ten bins. A model that is off
                gets a two-number correction (Platt scaling) that keeps the
                ranking and fixes the scale.
* Cost          expected cost per 1,000 rows at the cut-off, next to "act on
                nobody" and "act on everybody".
* Groups        the same numbers per group for sensitive columns (age, sex,
                ...). Big gaps are shown, not hidden.
* Intervals     for numbers: a range that held the true value for 80% of
                cross-validated rows, and how often it holds on unseen rows.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

SENSITIVE = re.compile(
    r"(^|[_\- ])(age|sex|gender|race|ethnic\w*|religion|marital|nationality|native|citizenship|disab\w*|pregnan\w*)($|[_\- ])",
    re.IGNORECASE,
)
CALIBRATE_ABOVE = 0.05   # expected calibration error worth fixing; below this it is mostly bin noise
INTERVAL = 0.8
MIN_GROUP_ROWS = 20      # shown
GAP_GROUP_ROWS = 100     # big enough to make a claim about
GAP_POSITIVES = 20       # real cases needed before comparing how many are caught


@dataclass
class Counts:
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def precision(self) -> float | None:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else None

    @property
    def recall(self) -> float | None:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else None

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p and r else 0.0

    @property
    def flagged(self) -> float:
        n = self.tp + self.fp + self.fn + self.tn
        return (self.tp + self.fp) / n if n else 0.0

    def summary(self) -> dict[str, Any]:
        return {**asdict(self), "precision": self.precision, "recall": self.recall, "f1": self.f1,
                "flagged": self.flagged}


@dataclass
class GroupRow:
    value: str
    rows: int
    actual: float              # outcome rate (yes/no) or mean target (numbers)
    predicted: float           # flagged rate at the cut-off, or mean prediction
    recall: float | None = None
    score: float | None = None  # ROC AUC within the group, or MAE for numbers


@dataclass
class GroupReport:
    column: str
    rows: list[GroupRow]
    gap: str | None = None       # something the model does unevenly
    context: str | None = None   # an uneven outcome that is already in the data


@dataclass
class Evaluation:
    kind: str                        # binary | multiclass | regression
    oof_rows: int = 0
    notes: list[str] = field(default_factory=list)
    # yes/no
    positive_class: str | None = None
    base_rate: float | None = None
    pr_auc: float | None = None
    cutoff: float | None = None
    cutoff_reason: str = ""
    at_cutoff: dict[str, Any] | None = None
    at_half: dict[str, Any] | None = None
    ece_before: float | None = None
    ece_after: float | None = None
    brier_before: float | None = None
    brier_after: float | None = None
    bins: list[tuple[float, float, int]] = field(default_factory=list)  # (mean predicted, actual rate, rows)
    platt: tuple[float, float] | None = None
    cost: dict[str, Any] | None = None
    # several classes
    per_class: list[dict[str, Any]] = field(default_factory=list)
    confusion: list[list[int]] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    # numbers
    half_width: float | None = None
    coverage: float | None = None
    bias: float | None = None
    groups: list[GroupReport] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def evaluation_from_json(data: dict[str, Any] | None) -> Evaluation | None:
    if not data:
        return None
    payload = dict(data)
    payload["groups"] = [GroupReport(g["column"], [GroupRow(**r) for r in g["rows"]], g.get("gap"), g.get("context"))
                         for g in payload.get("groups", [])]
    payload["bins"] = [tuple(b) for b in payload.get("bins", [])]
    if payload.get("platt"):
        payload["platt"] = tuple(payload["platt"])
    known = Evaluation.__dataclass_fields__.keys()
    return Evaluation(**{k: v for k, v in payload.items() if k in known})


# ---------------------------------------------------------------------------
# Small pieces
# ---------------------------------------------------------------------------


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def apply_platt(p: np.ndarray, platt: tuple[float, float] | None) -> np.ndarray:
    if platt is None:
        return p
    a, b = platt
    return 1.0 / (1.0 + np.exp(-(a * _logit(p) + b)))


def fit_platt(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    model = LogisticRegression(C=1e6, max_iter=1000)
    model.fit(_logit(p).reshape(-1, 1), y)
    return float(model.coef_[0][0]), float(model.intercept_[0])


def _platt_helps(p: np.ndarray, y: np.ndarray) -> bool:
    """Fit the rescaling on one half, check it on the other, both ways round. In-sample
    it always looks better; it only counts if it helps rows it was not fitted on."""
    rng = np.random.default_rng(0)
    half = rng.permutation(len(p)) < len(p) // 2
    gains = []
    for fit, check in ((half, ~half), (~half, half)):
        if len(set(y[fit].tolist())) < 2 or check.sum() < 50:
            return False
        a, b = fit_platt(p[fit], y[fit])
        gains.append(ece(p[check], y[check]) - ece(apply_platt(p[check], (a, b)), y[check]))
    return min(gains) > 0.01


def calibration_bins(p: np.ndarray, y: np.ndarray, bins: int = 10) -> list[tuple[float, float, int]]:
    """Equal-count bins: in each, the average predicted probability and the actual rate."""
    if len(p) == 0:
        return []
    order = np.argsort(p, kind="stable")
    out = []
    for chunk in np.array_split(order, min(bins, len(p))):
        if len(chunk):
            out.append((float(p[chunk].mean()), float(y[chunk].mean()), int(len(chunk))))
    return out


def ece(p: np.ndarray, y: np.ndarray) -> float:
    """Expected calibration error: the row-weighted gap between predicted and actual, over bins."""
    rows = calibration_bins(p, y)
    total = sum(n for _, _, n in rows)
    return float(sum(abs(m - a) * n for m, a, n in rows) / total) if total else 0.0


def counts_at(p: np.ndarray, y: np.ndarray, cutoff: float) -> Counts:
    flag = p >= cutoff
    return Counts(tp=int(np.sum(flag & (y == 1))), fp=int(np.sum(flag & (y == 0))),
                  fn=int(np.sum(~flag & (y == 1))), tn=int(np.sum(~flag & (y == 0))))


def _cost_per_1000(c: Counts, false_alarm: float, miss: float) -> float:
    n = c.tp + c.fp + c.fn + c.tn
    return 1000.0 * (c.fp * false_alarm + c.fn * miss) / n if n else 0.0


def choose_cutoff(p: np.ndarray, y: np.ndarray, false_alarm: float | None, miss: float | None) -> tuple[float, str]:
    candidates = np.unique(np.quantile(p, np.linspace(0.01, 0.99, 99)))
    if false_alarm is not None and miss is not None:
        best = min(candidates, key=lambda t: (_cost_per_1000(counts_at(p, y, t), false_alarm, miss), -t))
        return float(best), (f"lowest cost on cross-validated predictions, with a false alarm costing "
                             f"{false_alarm:g} and a miss costing {miss:g}")
    best = max(candidates, key=lambda t: (counts_at(p, y, t).f1 or 0.0, t))
    return float(best), "best F1 (balance of catching cases and avoiding false alarms) on cross-validated predictions"


def sensitive_columns(names: list[str], extra: list[str] | None = None,
                      derived: set[str] | None = None) -> list[int]:
    """Columns as they came in whose names point at a protected attribute, plus any asked
    for. Features built from them (age__squared, age_hours) are not groups anyone is in."""
    wanted = {e.lower() for e in (extra or [])}
    derived = derived or set()
    return [i for i, n in enumerate(names)
            if n.lower() in wanted or (n not in derived and "__" not in n and SENSITIVE.search(n))]


def _group_labels(values: np.ndarray, numeric: bool) -> np.ndarray:
    if numeric:
        x = np.array([np.nan if v is None else float(v) for v in values], dtype=float)
        ok = ~np.isnan(x)
        if len(np.unique(x[ok])) <= 6:  # a code such as sex = 1 / 2: each value is its own group
            return np.array(["missing" if math.isnan(v) else f"{v:g}" for v in x], dtype=object)
        labels = np.array(["missing"] * len(x), dtype=object)
        if ok.sum() >= 4:
            edges = np.unique(np.quantile(x[ok], [0.25, 0.5, 0.75]))
            idx = np.searchsorted(edges, x[ok], side="right")
            bounds = [-math.inf, *edges.tolist(), math.inf]
            names = []
            for i in range(len(bounds) - 1):
                lo, hi = bounds[i], bounds[i + 1]
                names.append(f"under {hi:,.4g}" if lo == -math.inf else f"{lo:,.4g} or more" if hi == math.inf
                             else f"{lo:,.4g} to {hi:,.4g}")
            labels[ok] = [names[i] for i in idx]
        return labels
    text = np.array(["missing" if v is None else str(v) for v in values], dtype=object)
    values_, counts = np.unique(text, return_counts=True)
    top = {v for v, _ in sorted(zip(values_, counts, strict=True), key=lambda vc: -vc[1])[:6]}
    return np.array([t if t in top else "other" for t in text], dtype=object)


# ---------------------------------------------------------------------------
# The evaluation
# ---------------------------------------------------------------------------


def evaluate(
    *,
    task: str,
    classes: list[str],
    y_oof: np.ndarray,
    oof_pred: np.ndarray,
    oof_proba: np.ndarray | None,
    y_test: np.ndarray,
    test_pred: np.ndarray,
    test_proba: np.ndarray | None,
    x_test: np.ndarray,
    names: list[str],
    kinds: list[str],
    cost_false_alarm: float | None = None,
    cost_miss: float | None = None,
    fairness_columns: list[str] | None = None,
    derived: set[str] | None = None,
) -> Evaluation:
    if task == "regression":
        ev = _regression(y_oof, oof_pred, y_test, test_pred)
    elif len(classes) == 2:
        assert oof_proba is not None and test_proba is not None
        ev = _binary(classes, y_oof, oof_proba[:, 1], y_test, test_proba[:, 1], cost_false_alarm, cost_miss)
    else:
        ev = _multiclass(classes, y_test, test_pred)
    ev.oof_rows = int(len(y_oof))

    for i in sensitive_columns(names, fairness_columns, derived):
        report = _group_report(ev, task, classes, names[i], x_test[:, i], kinds[i] in {"numeric", "boolean"},
                               y_test, test_pred, test_proba)
        if report is not None:
            ev.groups.append(report)
    return ev


def _binary(classes, y_oof, p_oof, y_test, p_test, false_alarm, miss) -> Evaluation:
    ev = Evaluation(kind="binary", positive_class=classes[1])
    ev.base_rate = float(np.mean(y_test))
    if len(set(y_test.tolist())) == 2:
        ev.pr_auc = float(average_precision_score(y_test, p_test))

    ev.ece_before = ece(p_test, y_test)
    ev.brier_before = float(np.mean((p_test - y_test) ** 2))
    oof_ece = ece(p_oof, y_oof)
    if oof_ece > CALIBRATE_ABOVE and len(set(y_oof.tolist())) == 2:
        platt = fit_platt(p_oof, y_oof)
        if platt[0] > 0 and _platt_helps(p_oof, y_oof):
            ev.platt = platt
    if ev.platt is not None:
        p_oof = apply_platt(p_oof, ev.platt)
        p_test = apply_platt(p_test, ev.platt)
        ev.ece_after = ece(p_test, y_test)
        ev.brier_after = float(np.mean((p_test - y_test) ** 2))
        ev.notes.append(
            f"Its probabilities were off by {oof_ece:.1%} on average in cross-validation, so they were "
            "rescaled (Platt scaling). The ranking, and so ROC AUC, does not change."
        )
    ev.bins = calibration_bins(p_test, y_test)

    ev.cutoff, ev.cutoff_reason = choose_cutoff(p_oof, y_oof, false_alarm, miss)
    at_cut, at_half = counts_at(p_test, y_test, ev.cutoff), counts_at(p_test, y_test, 0.5)
    ev.at_cutoff, ev.at_half = at_cut.summary(), at_half.summary()
    if false_alarm is not None and miss is not None:
        nobody = Counts(0, 0, int(np.sum(y_test == 1)), int(np.sum(y_test == 0)))
        everybody = Counts(int(np.sum(y_test == 1)), int(np.sum(y_test == 0)), 0, 0)
        ev.cost = {
            "false_alarm": false_alarm, "miss": miss,
            "model": _cost_per_1000(at_cut, false_alarm, miss),
            "at_half": _cost_per_1000(at_half, false_alarm, miss),
            "act_on_nobody": _cost_per_1000(nobody, false_alarm, miss),
            "act_on_everybody": _cost_per_1000(everybody, false_alarm, miss),
        }
    if ev.base_rate is not None and ev.base_rate < 0.2:
        ev.notes.append(
            f"The outcome is rare ({ev.base_rate:.0%} of unseen rows). Accuracy and ROC AUC can look good while "
            "few cases are caught: read recall and PR AUC."
        )
    return ev


def _multiclass(classes, y_test, test_pred) -> Evaluation:
    ev = Evaluation(kind="multiclass", classes=list(classes))
    k = len(classes)
    matrix = np.zeros((k, k), dtype=int)
    for actual, predicted in zip(y_test.tolist(), np.asarray(test_pred).astype(int).tolist(), strict=True):
        matrix[actual, predicted] += 1
    ev.confusion = matrix.tolist()
    for i, c in enumerate(classes):
        tp, predicted, actual = matrix[i, i], matrix[:, i].sum(), matrix[i, :].sum()
        ev.per_class.append({"class": c, "rows": int(actual),
                             "precision": float(tp / predicted) if predicted else None,
                             "recall": float(tp / actual) if actual else None})
    weak = [p["class"] for p in ev.per_class if p["recall"] is not None and p["recall"] < 0.5]
    if weak:
        ev.notes.append(f"Fewer than half the rows of {', '.join(weak)} are found. Check if that matters.")
    return ev


def _regression(y_oof, oof_pred, y_test, test_pred) -> Evaluation:
    ev = Evaluation(kind="regression")
    residual = np.abs(np.asarray(y_oof, dtype=float) - np.asarray(oof_pred, dtype=float))
    if len(residual):
        n = len(residual)
        # Split-conformal style: the quantile that makes the range hold for 80% of rows.
        level = min(1.0, math.ceil((n + 1) * INTERVAL) / n)
        ev.half_width = float(np.quantile(residual, level))
        inside = np.abs(np.asarray(y_test, dtype=float) - np.asarray(test_pred, dtype=float)) <= ev.half_width
        ev.coverage = float(np.mean(inside)) if len(inside) else None
    ev.bias = float(np.mean(np.asarray(test_pred, dtype=float) - np.asarray(y_test, dtype=float)))
    if ev.coverage is not None and ev.coverage < INTERVAL - 0.08:
        ev.notes.append(
            f"The ±{ev.half_width:,.4g} range held for only {ev.coverage:.0%} of unseen rows, against "
            f"{INTERVAL:.0%} in cross-validation: unseen rows are harder than the training rows suggested."
        )
    return ev


def _group_report(ev, task, classes, column, values, numeric, y_test, test_pred, test_proba) -> GroupReport | None:
    labels = _group_labels(values, numeric)
    rows: list[GroupRow] = []
    for value in sorted(set(labels.tolist()), key=str):
        m = labels == value
        n = int(m.sum())
        if n < MIN_GROUP_ROWS:
            continue
        y = y_test[m]
        if task == "regression":
            pred = np.asarray(test_pred, dtype=float)[m]
            rows.append(GroupRow(str(value), n, float(np.mean(y)), float(np.mean(pred)),
                                 score=float(np.mean(np.abs(pred - y)))))
        elif len(classes) == 2 and test_proba is not None:
            p = apply_platt(test_proba[m, 1], ev.platt)
            c = counts_at(p, y, ev.cutoff if ev.cutoff is not None else 0.5)
            auc = float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) and min(y.sum(), len(y) - y.sum()) >= 5 else None
            rows.append(GroupRow(str(value), n, float(np.mean(y)), c.flagged,
                                 recall=c.recall if c.tp + c.fn >= 5 else None, score=auc))
        else:
            pred = np.asarray(test_pred).astype(int)[m]
            rows.append(GroupRow(str(value), n, float("nan"), float("nan"), score=float(np.mean(pred == y))))
    if len(rows) < 2:
        return None
    report = GroupReport(column, rows)
    gaps: list[str] = []
    big = [r for r in rows if r.rows >= GAP_GROUP_ROWS and r.value not in {"missing", "other"}]
    if task == "classification" and len(classes) == 2:
        # 1. Of the real cases, are some groups' found far less often? (equal opportunity)
        recalls = [(r.recall, r.value) for r in big
                   if r.recall is not None and r.actual * r.rows >= GAP_POSITIVES]
        if len(recalls) >= 2:
            (low, low_v), (high, high_v) = min(recalls), max(recalls)
            if high - low > 0.15:
                gaps.append(f"of the real cases, it finds {high:.0%} in {high_v} but only {low:.0%} in {low_v}")
        # 2. Flag rates (the four-fifths rule), separated into what is already in the data
        #    and what the model adds on top: flagged rate relative to the group's real rate.
        flagged = [r for r in big if r.actual > 0]
        if len(flagged) >= 2:
            hi_r, lo_r = max(flagged, key=lambda r: r.predicted), min(flagged, key=lambda r: r.predicted)
            if hi_r.predicted > 0 and lo_r.predicted / hi_r.predicted < 0.8 and hi_r.predicted - lo_r.predicted > 0.05:
                lift = {r.value: r.predicted / r.actual for r in flagged if r.actual >= 0.02}
                widened = (hi_r.value in lift and lo_r.value in lift
                           and lift[lo_r.value] / lift[hi_r.value] < 0.8)
                sentence = (f"it flags {hi_r.predicted:.0%} of {hi_r.value} but {lo_r.predicted:.0%} of "
                            f"{lo_r.value}, below the four-fifths rule; their real rates are {hi_r.actual:.0%} "
                            f"and {lo_r.actual:.0%}")
                if widened:
                    gaps.append(sentence + ", so the model widens a gap that is in the data")
                else:
                    report.context = (sentence + ". The gap is in the data itself; the model follows it "
                                      "without widening it. Whether that is acceptable is a decision, not a statistic.")
    elif task == "regression":
        errors = [(r.score, r.value) for r in big if r.score is not None]
        if len(errors) >= 2:
            (lo, lo_v), (hi, hi_v) = min(errors), max(errors)
            if lo > 0 and hi / lo > 1.5:
                gaps.append(f"its typical error is {hi:,.4g} for {hi_v} against {lo:,.4g} for {lo_v}")
    if gaps:
        report.gap = f"By {column}: " + "; ".join(gaps) + ". Check whether this is fair and intended."
    return report
