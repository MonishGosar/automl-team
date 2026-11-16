"""Does each feature actually help? Tested, not assumed.

For every feature the same reference model is cross-validated on the training
rows twice, fold by fold: once with the feature and once without it. The gain
is measured per fold, so both scores see exactly the same rows (a paired
comparison). Features that are usable but not switched on (for example ones
flagged as leaks) are tested the other way round: what adding them would do.

Folds share most of their training rows, so their scores are not independent;
the t-test is the Nadeau-Bengio corrected version, which accounts for that.

Testing many features at once means some will look helpful by luck. With 20
features and a 5% threshold, about one would "pass" by chance. So the p-values
are corrected with the Benjamini-Hochberg method (at most about 5% of the features
called real are expected to be flukes) before anything is called a real effect.

The locked test rows are never used here. Everything runs inside a time budget;
features it did not reach are listed as untested.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy import stats

from mlteam.contract import DataContract
from mlteam.features import Feature
from mlteam.store import Store
from mlteam.train import HIGHER_IS_BETTER, Candidate, _load, cv_folds, cv_scores, gain

REFERENCE = Candidate("reference", "boosting", {"learning_rate": 0.1, "rounds": 100.0, "max_depth": 3.0})
ALPHA = 0.05
MAX_TEST_ROWS = 8000  # training rows used for feature tests on big data


@dataclass
class FeatureTest:
    name: str
    kind: str                   # "drop": currently on, scored without it. "add": currently off, scored with it.
    gains: list[float]          # per fold, positive = the feature helps
    mean_gain: float
    folds_helped: int
    p_value: float | None
    p_adjusted: float | None = None
    verdict: str = ""           # helps | likely helps | no clear effect | likely hurts | hurts
    members: list[str] = field(default_factory=list)  # a feature and its derived versions, tested together
    seconds: float = 0.0

    def sentence(self) -> str:
        n = len(self.gains)
        together = f" (with {', '.join(m for m in self.members if m != self.name)})" if len(self.members) > 1 else ""
        what = (f"Removing it{together}" if self.kind == "drop" else "Adding it")
        change = -self.mean_gain if self.kind == "drop" else self.mean_gain
        direction = "costs" if change < 0 else "gains"
        base = (f"{what} {direction} {abs(change):.3f} on average; it helped in "
                f"{self.folds_helped} of {n} folds")
        if self.verdict == "helps":
            return base + ". A real effect, after correcting for testing many features."
        if self.verdict == "hurts":
            return base + ". It makes the model worse, after correcting for testing many features."
        if self.verdict == "likely helps":
            return base + (". Consistent, but not proven once corrected for testing many features at once: "
                           "probably useful.")
        if self.verdict == "likely hurts":
            return base + ". Consistently a little worse with it, though not proven: probably better left out."
        return base + ". Too small or too uneven to tell from luck."


@dataclass
class FeatureTestRun:
    metric: str
    higher_is_better: bool
    reference: str
    folds: int
    fold_kind: str
    base_scores: list[float]
    budget_seconds: float
    spent_seconds: float = 0.0
    tests: list[FeatureTest] = field(default_factory=list)
    untested: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    def by_name(self) -> dict[str, FeatureTest]:
        """Every feature name, including derived versions, mapped to the test that covered it."""
        out: dict[str, FeatureTest] = {}
        for t in self.tests:
            for member in t.members or [t.name]:
                out[member] = t
            out[t.name] = t
        return out

    def summary(self) -> str:
        counts: dict[str, int] = {}
        for t in self.tests:
            counts[t.verdict] = counts.get(t.verdict, 0) + 1
        text = (f"Tested {len(self.tests)} feature(s) in {self.spent_seconds:.0f} s: {counts.get('helps', 0)} help, "
                f"{counts.get('likely helps', 0)} likely help, {counts.get('no clear effect', 0)} no clear effect, "
                f"{counts.get('likely hurts', 0) + counts.get('hurts', 0)} hurt or likely hurt.")
        if self.untested:
            text += f" Not reached within the budget: {len(self.untested)}."
        return text


def _benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Controls the share of false discoveries among the features called real, rather
    than the chance of any single one (Holm). The usual choice when screening many."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        running = min(running, p_values[i] * m / (rank + 1))
        adjusted[i] = min(1.0, running)
    return adjusted


def _holm(p_values: list[float]) -> list[float]:
    """Holm step-down adjustment: controls the chance of any false 'real effect'."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def _corrected_p(gains: list[float], val_share: float) -> float:
    """Nadeau-Bengio corrected resampled t-test. A plain t-test treats the folds as
    independent and so claims far too much certainty; this widens the variance by
    the overlap between training sets."""
    n = len(gains)
    mean = float(np.mean(gains))
    var = float(np.var(gains, ddof=1))
    if var == 0:
        return 1.0 if mean == 0 else 0.0
    t = mean / np.sqrt((1.0 / n + val_share) * var)
    return float(2 * stats.t.sf(abs(t), df=n - 1))


def run_feature_tests(
    store: Store,
    contract: DataContract,
    on: list[Feature],
    off: list[Feature],
    budget_seconds: float = 30.0,
    repeats: int = 2,
    clock: Callable[[], float] = time.perf_counter,
) -> FeatureTestRun:
    """``on``: features currently switched on. ``off``: usable features currently off."""
    everything = [*on, *off]
    data = _load(store, contract, everything)
    sampled_from = None
    if len(data.y_train) > MAX_TEST_ROWS:
        # Ten fits per feature on tens of thousands of rows would spend the whole budget on
        # a handful of features. A sample keeps every feature tested; time order is kept.
        sampled_from = len(data.y_train)
        if data.time_ordered:
            keep = np.arange(len(data.y_train))[-MAX_TEST_ROWS:]
        else:
            keep = np.sort(np.random.default_rng(0).choice(len(data.y_train), MAX_TEST_ROWS, replace=False))
        data.x_train, data.y_train = data.x_train[keep], data.y_train[keep]
        if data.groups_train is not None:
            data.groups_train = data.groups_train[keep]
    task = contract.task or ""
    classify = task == "classification"
    metric = contract.metric or ("roc_auc" if classify else "rmse")
    n_classes = len(data.classes)
    if metric == "f1" and n_classes > 2:
        metric = "macro_f1"
    higher = HIGHER_IS_BETTER[metric]
    folds = cv_folds(data.y_train, classify=classify, time_ordered=data.time_ordered,
                     k=10 if data.time_ordered else 5, repeats=1 if data.time_ordered else repeats,
                     groups=data.groups_train)
    kinds = {i: f.kind for i, f in enumerate(everything)}

    def score(columns: list[int]) -> list[float]:
        x = data.x_train[:, columns]
        numeric = [j for j, c in enumerate(columns) if kinds[c] in {"numeric", "boolean"}]
        categorical = [j for j, c in enumerate(columns) if kinds[c] not in {"numeric", "boolean"}]
        return cv_scores(REFERENCE, task, metric, x, data.y_train, numeric, categorical, folds, n_classes)

    # Folds overlap (each row trains in most of them), so their scores are not independent.
    val_share = float(np.mean([len(v) / len(t) for t, v in folds]))
    started = clock()
    on_cols = list(range(len(on)))
    base = score(on_cols)
    run = FeatureTestRun(
        metric=metric, higher_is_better=higher, reference="gradient boosting, default settings",
        folds=len(folds),
        fold_kind=("expanding windows in time order" if data.time_ordered else f"{repeats} × shuffled 5-fold")
        + (f", on {MAX_TEST_ROWS:,} of the {sampled_from:,} training rows" if sampled_from else ""),
        base_scores=base, budget_seconds=budget_seconds,
    )
    if len(base) != len(folds):
        run.notes.append("Some folds could not be scored with all features, so the comparison uses fewer folds.")

    # A feature and its derived versions (x, x__log, x__clean) carry the same information, so
    # removing one alone shows nothing: the other covers for it. They are removed together.
    def family(name: str) -> str:
        for suffix in ("__log", "__clean"):
            if name.endswith(suffix):
                return name[: -len(suffix)]
        return name

    groups: dict[str, list[int]] = {}
    for i, feature in enumerate(on):
        groups.setdefault(family(feature.name), []).append(i)
    plan: list[tuple[str, str, list[int], list[str]]] = []
    for base_name, members in groups.items():
        names = [on[i].name for i in members]
        label = base_name if base_name in names else names[0]
        plan.append((label, "drop", [c for c in on_cols if c not in members], names))
    for j, feature in enumerate(off):
        plan.append((feature.name, "add", [*on_cols, len(on) + j], [feature.name]))

    for name, kind, columns, members in plan:
        if clock() - started >= budget_seconds:
            run.untested.append(name)
            continue
        t0 = clock()
        other = score(columns) if columns else []
        if len(other) != len(base) or not base:
            run.untested.append(name)
            continue
        # Positive gain = the feature helps, whichever way the test ran.
        per_fold = [gain(b, o, higher) if kind == "drop" else gain(o, b, higher) for b, o in zip(base, other)]
        p_value = _corrected_p(per_fold, val_share) if len(per_fold) >= 3 else None
        run.tests.append(FeatureTest(
            name=name, kind=kind, gains=[round(g, 5) for g in per_fold], mean_gain=float(np.mean(per_fold)),
            folds_helped=sum(1 for g in per_fold if g > 0), p_value=p_value, seconds=round(clock() - t0, 2),
            members=members,
        ))

    tested = [t for t in run.tests if t.p_value is not None]
    for t, adjusted in zip(tested, _benjamini_hochberg([t.p_value for t in tested])):  # type: ignore[misc]
        t.p_adjusted = adjusted
    for t in run.tests:
        consistent = max(t.folds_helped, len(t.gains) - t.folds_helped) >= 0.8 * len(t.gains)
        if t.p_adjusted is not None and t.p_adjusted < ALPHA:
            t.verdict = "helps" if t.mean_gain > 0 else "hurts"
        elif t.p_value is not None and t.p_value < ALPHA and consistent:
            t.verdict = "likely helps" if t.mean_gain > 0 else "likely hurts"
        else:
            t.verdict = "no clear effect"
    if run.tests and not any(t.verdict in {"helps", "likely helps"} for t in run.tests):
        run.notes.append(
            "None stood out on its own. Often several features carry the same information and cover for each "
            "other when one is removed; the model can still be good."
        )
    if len(data.y_train) < 300:
        run.notes.append(
            f"With {len(data.y_train)} training rows only large effects can be told apart from luck. "
            "'No clear effect' means unproven, not useless."
        )
    run.spent_seconds = round(clock() - started, 2)
    return run
