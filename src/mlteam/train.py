"""Training and scoring.

The rule that matters most: **nobody scores their own model.** Whoever proposed
a candidate (fixed rules or an agent), the harness builds it fresh, fits it on
the training rows, and scores it on test rows it has never seen. Every
candidate gets the same rows and the same metric.

Other rules enforced here:

* Gap-filling, scaling and category encoding are fitted on training rows only.
* With a time column, the model trains on the earlier rows and is tested on
  the later ones.
* A "do nothing" baseline is always trained. A model that cannot beat it
  cannot be approved.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from mlteam.contract import DataContract, base_sql
from mlteam.evaluate import Evaluation, apply_platt, evaluate
from mlteam.features import Feature, time_rule_ctes
from mlteam.store import Store, quote_ident

SEED = 0
TEST_SHARE = 0.2
MIN_ROWS = 40
MAX_CANDIDATES = 12  # everything trained in one run: the plan, plus tuned versions
PLAN_LIMIT = 8       # what an agent may propose, baseline included
CV_FOLDS = 5
BOOTSTRAP_ROUNDS = 500
HIGHER_IS_BETTER = {"roc_auc": True, "pr_auc": True, "f1": True, "macro_f1": True, "accuracy": True,
                    "r2": True, "log_loss": False, "rmse": False, "mae": False}
BOUNDED = {"roc_auc", "pr_auc", "f1", "macro_f1", "accuracy", "r2"}
METRIC_WORDS = {
    "roc_auc": "ROC AUC (1.0 = ranks every case right, 0.5 = coin flip)",
    "pr_auc": "PR AUC (how well it finds the rare class; higher is better)",
    "f1": "F1 (balance of catching cases and not raising false alarms; higher is better)",
    "macro_f1": "macro F1 (F1 averaged over classes; higher is better)",
    "accuracy": "accuracy (share of rows predicted right)",
    "log_loss": "log loss (penalty for confident wrong answers; lower is better)",
    "rmse": "RMSE (typical error in the target's units; lower is better)",
    "mae": "MAE (average error in the target's units; lower is better)",
    "r2": "R² (share of the variation explained; 1.0 is perfect)",
}

# family -> {parameter: (low, high, default)}. An agent can only pick from this menu.
FAMILIES: dict[str, dict[str, tuple[float, float, float]]] = {
    "baseline": {},
    "linear": {"regularization": (0.001, 100.0, 1.0)},
    "tree": {"max_depth": (1, 6, 3)},
    "forest": {"n_trees": (50, 500, 200), "max_depth": (2, 20, 8)},
    "boosting": {"learning_rate": (0.01, 0.3, 0.1), "rounds": (50, 500, 100), "max_depth": (2, 10, 4)},
}
FAMILY_WORDS = {
    "baseline": "Does nothing clever: always answers with the most common outcome. The bar to beat.",
    "linear": "Adds up each clue with a weight. Simple, fast, easy to explain.",
    "tree": "A short chain of yes/no questions. You can read every step.",
    "forest": "Many trees that vote. Usually strong, harder to explain.",
    "boosting": "Trees built one after another, each fixing the last one's mistakes. Often the most accurate.",
}
# Options every family except the baseline accepts. balanced = 1 weighs the rarer
# outcomes up, so a model cannot do well by ignoring them (yes/no and classes only).
OPTIONS: dict[str, tuple[float, float, float]] = {"balanced": (0, 1, 0)}
ENSEMBLE_WORDS = "The average of the best model from each family. Often a little more accurate, harder to explain."
RARE_SHARE = 0.2
_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,39}$")


class TrainError(Exception):
    """Raised when training cannot start at all."""


@dataclass
class Candidate:
    name: str
    family: str
    params: dict[str, float] = field(default_factory=dict)
    why: str = ""
    origin: str = "rules"


@dataclass
class ModelResult:
    name: str
    family: str
    params: dict[str, float]
    why: str
    origin: str
    test_score: float | None = None
    train_score: float | None = None
    extra: dict[str, float] = field(default_factory=dict)
    beats_baseline: bool = False
    recommended: bool = False
    notes: list[str] = field(default_factory=list)
    importance: list[tuple[str, float]] = field(default_factory=list)
    tree_text: str | None = None
    error: str | None = None
    fit_seconds: float = 0.0
    cv_mean: float | None = None        # cross-validated on training rows only
    cv_std: float | None = None
    ties_with: list[str] = field(default_factory=list)  # models it cannot be told apart from
    members: list[dict[str, Any]] = field(default_factory=list)  # for an ensemble: what it averages
    evaluation: dict[str, Any] | None = None  # cut-off, calibration, cost, groups: see evaluate.py


@dataclass
class TrainingRun:
    task: str
    metric: str
    higher_is_better: bool
    split: str
    train_rows: int
    test_rows: int
    dropped_rows: int
    positive_class: str | None
    features: list[str]
    classes: list[str] = field(default_factory=list)  # label order the models use; positive class last
    results: list[ModelResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def result(self, name: str) -> ModelResult | None:
        return next((r for r in self.results if r.name == name), None)

    @property
    def baseline(self) -> ModelResult | None:
        return next((r for r in self.results if r.family == "baseline"), None)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------


def default_plan() -> list[Candidate]:
    return [
        Candidate("baseline", "baseline", {}, FAMILY_WORDS["baseline"]),
        Candidate("linear", "linear", {}, FAMILY_WORDS["linear"]),
        Candidate("small_tree", "tree", {}, FAMILY_WORDS["tree"]),
        Candidate("forest", "forest", {}, FAMILY_WORDS["forest"]),
        Candidate("boosting", "boosting", {}, FAMILY_WORDS["boosting"]),
    ]


def check_candidate(candidate: Candidate, taken: set[str]) -> str | None:
    """Reason to refuse a candidate, or None. Also fills in default parameters."""
    if not _NAME.match(candidate.name):
        return "The name must start with a letter and use only letters, digits and underscores."
    if candidate.name.lower() in taken:
        return f"A candidate called {candidate.name} already exists."
    if candidate.family not in FAMILIES:
        return f"Unknown family {candidate.family}. Choose from: {', '.join(FAMILIES)}."
    menu = FAMILIES[candidate.family]
    options = OPTIONS if candidate.family != "baseline" else {}
    for key, (low, high, _default) in options.items():
        if key in candidate.params:
            value = candidate.params[key]
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not low <= float(value) <= high:
                return f"{key} must be a number between {low:g} and {high:g}."
            candidate.params[key] = float(value)
    unknown = [k for k in candidate.params if k not in menu and k not in options]
    if unknown:
        allowed = ", ".join([*menu, *options]) or "none"
        return f"{candidate.family} has no parameter {', '.join(unknown)}. Allowed: {allowed}."
    for key, (low, high, default) in menu.items():
        value = candidate.params.get(key, default)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or math.isnan(float(value)):
            return f"{key} must be a number."
        if not low <= float(value) <= high:
            return f"{key} must be between {low:g} and {high:g}, got {value:g}."
        candidate.params[key] = float(value)
    return None


def _estimator(candidate: Candidate, task: str) -> Any:
    p = candidate.params
    classify = task == "classification"
    weight = "balanced" if classify and p.get("balanced", 0) >= 0.5 else None
    if candidate.family == "baseline":
        return DummyClassifier(strategy="prior") if classify else DummyRegressor(strategy="mean")
    if candidate.family == "linear":
        if classify:
            return LogisticRegression(C=1.0 / p["regularization"], max_iter=2000, class_weight=weight)
        return Ridge(alpha=p["regularization"])
    if candidate.family == "tree":
        if classify:
            return DecisionTreeClassifier(max_depth=int(p["max_depth"]), min_samples_leaf=5, random_state=SEED,
                                          class_weight=weight)
        return DecisionTreeRegressor(max_depth=int(p["max_depth"]), min_samples_leaf=5, random_state=SEED)
    if candidate.family == "forest":
        shared = dict(n_estimators=int(p["n_trees"]), max_depth=int(p["max_depth"]), min_samples_leaf=2,
                      random_state=SEED, n_jobs=1)
        return RandomForestClassifier(**shared, class_weight=weight) if classify else RandomForestRegressor(**shared)
    shared = dict(learning_rate=p["learning_rate"], max_iter=int(p["rounds"]), max_depth=int(p["max_depth"]),
                  random_state=SEED)
    if classify:
        return HistGradientBoostingClassifier(**shared, class_weight=weight)
    return HistGradientBoostingRegressor(**shared)


def _to_float(values: np.ndarray) -> np.ndarray:
    return values.astype(float)


def _to_text(values: np.ndarray) -> np.ndarray:
    flat = ["missing" if v is None or (isinstance(v, float) and math.isnan(v)) else str(v)
            for v in values.ravel()]
    return np.array(flat, dtype=object).reshape(values.shape)


def _pipeline(candidate: Candidate, task: str, numeric: list[int], categorical: list[int]) -> Pipeline:
    """Everything that learns from data lives inside this pipeline, so it is
    fitted on training rows only."""
    steps_num: list[tuple[str, Any]] = [
        ("float", FunctionTransformer(_to_float, feature_names_out="one-to-one")),
        ("fill", SimpleImputer(strategy="median", add_indicator=True)),
    ]
    if candidate.family == "linear":
        steps_num.append(("scale", StandardScaler()))
    transformers: list[tuple[str, Any, list[int]]] = []
    if numeric:
        transformers.append(("num", Pipeline(steps_num), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([
            ("text", FunctionTransformer(_to_text, feature_names_out="one-to-one")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=20)),
        ]), categorical))
    return Pipeline([
        ("prepare", ColumnTransformer(transformers, verbose_feature_names_out=True)),
        ("model", _estimator(candidate, task)),
    ])


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def training_sql(contract: DataContract, features: list[Feature]) -> str:
    selects = [f"b.{quote_ident(contract.target)} AS __y"]
    time_col = f"b.{quote_ident(contract.time_column)}" if contract.time_column else "NULL"
    selects.append(f"{time_col} AS __t")
    group_col = f"b.{quote_ident(contract.group_column)}" if contract.group_column else "NULL"
    selects.append(f"{group_col} AS __g")
    selects += [f"({f.sql}) AS {quote_ident(f.name)}" for f in features]
    order = ", ".join(f"b.{quote_ident(k)}" for k in contract.grain_keys)
    tail = f" ORDER BY {order}" if order else ""
    return f"{time_rule_ctes(contract)}SELECT {', '.join(selects)} FROM {base_sql(contract)} AS b{tail}"


def _column(values: list[Any], kind: str) -> np.ndarray:
    if kind in {"numeric", "boolean"}:
        return np.array([np.nan if v is None else float(v) for v in values], dtype=object)
    return np.array([None if v is None else str(v) for v in values], dtype=object)


@dataclass
class _Data:
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    numeric: list[int]
    categorical: list[int]
    classes: list[str]
    split: str
    dropped: int
    warnings: list[str]
    time_ordered: bool = False  # x_train is sorted by time, so folds must respect it
    groups_train: np.ndarray | None = None  # e.g. customer: all of one group's rows stay on one side


def _load(store: Store, contract: DataContract, features: list[Feature]) -> _Data:
    if not contract.target or contract.task not in {"classification", "regression"}:
        raise TrainError("The contract has no target or task.")
    if not features:
        raise TrainError("No features were approved.")
    _, rows = store.query(training_sql(contract, features))
    kept = [r for r in rows if r[0] is not None]
    dropped = len(rows) - len(kept)
    if len(kept) < MIN_ROWS:
        raise TrainError(f"Only {len(kept)} rows have a target. At least {MIN_ROWS} are needed.")
    warnings: list[str] = []
    classify = contract.task == "classification"

    columns = [_column([r[i + 3] for r in kept], f.kind) for i, f in enumerate(features)]
    groups = np.array(["" if r[2] is None else str(r[2]) for r in kept], dtype=object) \
        if contract.group_column else None
    x = np.empty((len(kept), len(features)), dtype=object)
    for i, col in enumerate(columns):
        x[:, i] = col
    numeric = [i for i, f in enumerate(features) if f.kind in {"numeric", "boolean"}]
    categorical = [i for i, f in enumerate(features) if f.kind not in {"numeric", "boolean"}]

    classes: list[str] = []
    if classify:
        labels = [str(r[0]) for r in kept]
        counts: dict[str, int] = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
        classes = sorted(counts)
        if len(classes) < 2:
            raise TrainError("The target has only one value in the rows that are left.")
        if len(classes) == 2:
            truthy = [c for c in classes if c.lower() in {"1", "true", "yes", "y", "1.0"}]
            positive = truthy[0] if len(truthy) == 1 else min(classes, key=lambda c: (counts[c], c))
            classes = [c for c in classes if c != positive] + [positive]  # positive class is index 1
        index = {c: i for i, c in enumerate(classes)}
        y = np.array([index[label] for label in labels])
    else:
        y = np.array([float(r[0]) for r in kept])

    times = [r[1] for r in kept]
    n_test = max(1, math.ceil(len(kept) * TEST_SHARE))
    split = ""
    order: np.ndarray | None = None
    if contract.time_column and all(t is not None for t in times):
        order = np.argsort(np.array([str(t) for t in times]), kind="stable")
        test_y = y[order[-n_test:]]
        if classify and len(set(test_y.tolist())) < len(classes):
            warnings.append(
                f"The latest {n_test} rows by {contract.time_column} do not contain every outcome, "
                "so a random split was used instead of a split by time."
            )
            order = None
        else:
            split = (f"Trained on the earliest {len(kept) - n_test:,} rows by {contract.time_column}, "
                     f"tested on the latest {n_test:,}.")
    if order is not None:
        train_idx, test_idx = order[:-n_test], order[-n_test:]
    elif groups is not None and len(set(groups.tolist())) >= 5:
        splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SHARE, random_state=SEED)
        train_idx, test_idx = next(splitter.split(np.zeros(len(kept)), groups=groups))
        split = (f"Trained on {len(train_idx):,} rows, tested on {len(test_idx):,} held-out rows, split by "
                 f"{contract.group_column}: no {contract.group_column} has rows on both sides.")
    else:
        stratify = y if classify and min(np.bincount(y)) >= 2 else None
        train_idx, test_idx = train_test_split(
            np.arange(len(kept)), test_size=n_test, random_state=SEED, stratify=stratify
        )
        how = "keeping the same mix of outcomes in both" if stratify is not None else "at random"
        split = f"Trained on {len(train_idx):,} rows, tested on {len(test_idx):,} held-out rows, split {how}."
    if n_test < 100:
        warnings.append(
            f"Only {n_test} test rows. Scores can move by several points by chance; treat small gaps as ties."
        )
    return _Data(x[train_idx], x[test_idx], y[train_idx], y[test_idx], numeric, categorical,
                 classes, split, dropped, warnings, time_ordered=order is not None,
                 groups_train=None if groups is None or order is not None else groups[train_idx])


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _outputs(model: Pipeline, x: np.ndarray, task: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Predictions once, so a metric can be recomputed on many resamples cheaply."""
    pred = model.predict(x)
    proba = model.predict_proba(x) if task == "classification" else None
    return pred, proba


def _metric_from(metric: str, y: np.ndarray, pred: np.ndarray, proba: np.ndarray | None, n_classes: int) -> float:
    if metric in {"rmse", "mae", "r2"}:
        if metric == "rmse":
            return float(root_mean_squared_error(y, pred))
        return float(mean_absolute_error(y, pred) if metric == "mae" else r2_score(y, pred))
    if metric in {"accuracy", "f1", "macro_f1"}:
        if metric == "accuracy":
            return float(accuracy_score(y, pred))
        average = "binary" if metric == "f1" and n_classes == 2 else "macro"
        return float(f1_score(y, pred, average=average, zero_division=0))
    assert proba is not None
    labels = list(range(n_classes))
    if metric == "log_loss":
        return float(log_loss(y, proba, labels=labels))
    if n_classes == 2:
        positive = proba[:, 1]
        return float(roc_auc_score(y, positive) if metric == "roc_auc" else average_precision_score(y, positive))
    if metric == "roc_auc":
        return float(roc_auc_score(y, proba, multi_class="ovr", labels=labels))
    onehot = np.eye(n_classes)[y]
    return float(average_precision_score(onehot, proba, average="macro"))


def _score(metric: str, model: Pipeline, x: np.ndarray, y: np.ndarray, n_classes: int) -> float:
    task = "regression" if metric in {"rmse", "mae", "r2"} else "classification"
    pred, proba = _outputs(model, x, task)
    return _metric_from(metric, y, pred, proba, n_classes)


def gain(value: float, reference: float, higher_is_better: bool) -> float:
    """How much better ``value`` is than ``reference``, positive meaning better, whatever the metric."""
    return (value - reference) if higher_is_better else (reference - value)


def cv_folds(
    y: np.ndarray, *, classify: bool, time_ordered: bool, k: int = CV_FOLDS, repeats: int = 1, seed: int = SEED,
    groups: np.ndarray | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Folds over the training rows only; the locked test rows are never in here.
    Rows sorted by time get expanding-window folds: always train on the past,
    validate on the next block."""
    n = len(y)
    if time_ordered:
        from sklearn.model_selection import TimeSeriesSplit

        k = max(2, min(k, n // 10))
        return list(TimeSeriesSplit(n_splits=k).split(np.zeros(n)))
    from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold

    if groups is not None and len(set(groups.tolist())) >= 2:
        splits = max(2, min(k, len(set(groups.tolist()))))
        if classify and int(min(np.bincount(y))) >= splits:
            return list(StratifiedGroupKFold(n_splits=splits, shuffle=True, random_state=seed)
                        .split(np.zeros(n), y, groups))
        return list(GroupKFold(n_splits=splits).split(np.zeros(n), y, groups))
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for repeat in range(repeats):
        if classify:
            smallest = int(min(np.bincount(y))) if len(y) else 0
            splits = max(2, min(k, smallest))
            if smallest >= 2:
                splitter = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed + repeat)
                folds += list(splitter.split(np.zeros(n), y))
                continue
        splitter = KFold(n_splits=max(2, min(k, n)), shuffle=True, random_state=seed + repeat)
        folds += list(splitter.split(np.zeros(n)))
    return folds


def cv_scores(
    candidate: Candidate,
    task: str,
    metric: str,
    x: np.ndarray,
    y: np.ndarray,
    numeric: list[int],
    categorical: list[int],
    folds: list[tuple[np.ndarray, np.ndarray]],
    n_classes: int,
    on_fold: Any = None,
    collect: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray | None]] | None = None,
) -> list[float]:
    """Fit and score once per fold. ``on_fold(i, scores)`` may return True to stop
    early (used to drop a hopeless try after its first folds). A fold that cannot
    be scored, for example one with a single outcome in it, is skipped."""
    import warnings

    scores: list[float] = []
    for i, (train_idx, val_idx) in enumerate(folds):
        model = _pipeline(candidate, task, numeric, categorical)
        try:
            with warnings.catch_warnings():
                # A fold can see a feature that is empty in its rows; the imputer says so and moves on.
                warnings.simplefilter("ignore", UserWarning)
                model.fit(x[train_idx], y[train_idx])
                pred, proba = _outputs(model, x[val_idx], task)
                scores.append(_metric_from(metric, y[val_idx], pred, proba, n_classes))
                if collect is not None:
                    collect[i] = (val_idx, pred, proba)
        except ValueError:
            continue
        if on_fold is not None and on_fold(i, scores):
            break
    return scores


def compare_on_test(
    metric: str,
    higher_is_better: bool,
    y: np.ndarray,
    a: tuple[np.ndarray, np.ndarray | None],
    b: tuple[np.ndarray, np.ndarray | None],
    n_classes: int,
    rounds: int = BOOTSTRAP_ROUNDS,
) -> tuple[float, float, float]:
    """Paired bootstrap on the same test rows: (gain of a over b, 5th percentile, 95th percentile)."""
    point = gain(_metric_from(metric, y, a[0], a[1], n_classes), _metric_from(metric, y, b[0], b[1], n_classes),
                 higher_is_better)
    rng = np.random.default_rng(SEED)
    diffs: list[float] = []
    for _ in range(rounds):
        idx = rng.integers(0, len(y), len(y))
        try:
            score_a = _metric_from(metric, y[idx], a[0][idx], None if a[1] is None else a[1][idx], n_classes)
            score_b = _metric_from(metric, y[idx], b[0][idx], None if b[1] is None else b[1][idx], n_classes)
        except ValueError:
            continue
        diffs.append(gain(score_a, score_b, higher_is_better))
    if len(diffs) < rounds // 2:
        return point, float("nan"), float("nan")
    return point, float(np.percentile(diffs, 5)), float(np.percentile(diffs, 95))


def _extra_metrics(task: str, metric: str, n_classes: int) -> list[str]:
    if task == "regression":
        return [m for m in ("rmse", "mae", "r2") if m != metric]
    names = ["accuracy", "roc_auc", "f1" if n_classes == 2 else "macro_f1"] + (["pr_auc"] if n_classes == 2 else [])
    return [m for m in names if m != metric]


def _friendly(name: str) -> str:
    """'num__orders__count' -> 'orders__count'; 'cat__plan_pro' -> 'plan = pro'."""
    if name.startswith("num__"):
        rest = name[len("num__"):]
        if rest.startswith("missingindicator_"):
            return rest[len("missingindicator_"):] + " is missing"
        return rest
    if name.startswith("cat__"):
        rest = name[len("cat__"):]
        if rest.endswith("_infrequent_sklearn"):
            return rest[: -len("_infrequent_sklearn")] + " = a rare value"
        feature, _, value = rest.rpartition("_")
        return f"{feature} = {value}" if feature else rest
    return name


def tree_text(model: Pipeline, classes: list[str], task: str, feature_names: list[str]) -> str:
    """Draw a fitted decision tree as indented yes/no questions."""
    tree = model.named_steps["model"].tree_
    prepare = model.named_steps["prepare"]
    try:
        names = [_friendly(n) for n in prepare.get_feature_names_out(feature_names)]
    except Exception:  # noqa: BLE001 - fall back to positions rather than fail the run
        names = [f"column {i}" for i in range(tree.n_features)]
    lines: list[str] = []

    def leaf(node: int) -> str:
        rows = int(tree.n_node_samples[node])
        value = tree.value[node][0]
        if task == "classification":
            share = value / value.sum() if value.sum() else value
            top = int(np.argmax(share))
            return f"→ {classes[top]}  ({share[top]:.0%} of {rows} rows)"
        return f"→ {float(value[0]):,.2f}  (average of {rows} rows)"

    def question(node: int) -> str:
        name = names[tree.feature[node]]
        threshold = float(tree.threshold[node])
        if " = " in name or name.endswith(" is missing"):
            return f"{name}?"
        return f"{name} > {threshold:,.2f}?"

    def walk(node: int, prefix: str) -> None:
        # sklearn sends "feature <= threshold" left. For a yes/no column that is "no".
        left, right = int(tree.children_left[node]), int(tree.children_right[node])
        for child, label, last in ((right, "yes", False), (left, "no ", True)):
            branch, extend = ("└─", "   ") if last else ("├─", "│  ")
            if tree.children_left[child] == -1:
                lines.append(f"{prefix}{branch} {label} {leaf(child)}")
            else:
                lines.append(f"{prefix}{branch} {label}  {question(child)}")
                walk(child, prefix + extend)

    if tree.children_left[0] == -1:
        return leaf(0)
    lines.append(question(0))
    walk(0, "")
    return "\n".join(lines)


class Ensemble:
    """The average of several fitted pipelines: probabilities for classes, values for numbers."""

    def __init__(self, members: list[Any], task: str) -> None:
        self.members = members
        self.task = task

    def fit(self, x: np.ndarray, y: np.ndarray) -> Ensemble:
        for member in self.members:
            member.fit(x, y)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.mean([m.predict_proba(x) for m in self.members], axis=0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.task == "classification":
            return np.argmax(self.predict_proba(x), axis=1)
        return np.mean([m.predict(x) for m in self.members], axis=0)


class Delivered:
    """A fitted model plus the decisions made on top of it: rescaled probabilities,
    a yes/no cut-off, and the half-width of an 80% range for numbers."""

    def __init__(self, model: Any, *, platt: tuple[float, float] | None = None, cutoff: float | None = None,
                 half_width: float | None = None) -> None:
        self.model = model
        self.platt = tuple(platt) if platt else None
        self.cutoff = cutoff
        self.half_width = half_width

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        proba = np.asarray(self.model.predict_proba(x), dtype=float)
        if self.platt is not None and proba.shape[1] == 2:
            positive = apply_platt(proba[:, 1], self.platt)
            proba = np.column_stack([1 - positive, positive])
        return proba

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.cutoff is not None:
            proba = self.predict_proba(x)
            if proba.shape[1] == 2:
                return (proba[:, 1] >= self.cutoff).astype(int)
        return self.model.predict(x)

    def __getattr__(self, name: str) -> Any:
        if name == "model":  # not set yet, e.g. while unpickling
            raise AttributeError(name)
        return getattr(self.model, name)


def build_model(family: str, params: dict[str, Any], members: list[dict[str, Any]] | None, task: str,
                numeric: list[int], categorical: list[int]) -> Any:
    """An unfitted model from what a model card records, ensembles included."""
    if family == "ensemble":
        return Ensemble([_pipeline(Candidate(m["name"], m["family"], {k: float(v) for k, v in m["params"].items()}),
                                   task, numeric, categorical) for m in members or []], task)
    return _pipeline(Candidate("model", family, {k: float(v) for k, v in params.items()}), task, numeric, categorical)


def _importance(model: Any, metric: str, x: np.ndarray, y: np.ndarray, names: list[str], n_classes: int,
                base: float, higher: bool) -> list[tuple[str, float]]:
    """How much the score drops when one feature's values are shuffled on the unseen
    rows: a model-agnostic measure of how much it relies on that feature."""
    rng = np.random.default_rng(SEED)
    drops: list[tuple[str, float]] = []
    for i, name in enumerate(names):
        losses = []
        for _ in range(5):
            shuffled = x.copy()
            shuffled[:, i] = shuffled[rng.permutation(len(shuffled)), i]
            score = _score(metric, model, shuffled, y, n_classes)
            losses.append((base - score) if higher else (score - base))
        drops.append((name, float(np.mean(losses))))
    return sorted(drops, key=lambda item: -item[1])


def _stack_oof(folds_out: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray | None]], y: np.ndarray):
    """Out-of-fold outputs, in row order of the folds that were scored."""
    if not folds_out:
        return None
    keys = sorted(folds_out)
    idx = np.concatenate([folds_out[k][0] for k in keys])
    pred = np.concatenate([folds_out[k][1] for k in keys])
    probas = [folds_out[k][2] for k in keys]
    proba = None if any(p is None for p in probas) else np.vstack(probas)
    return y[idx], pred, proba


def train_all(
    store: Store,
    contract: DataContract,
    features: list[Feature],
    candidates: list[Candidate],
    models_dir: Path,
) -> TrainingRun:
    """Fit every candidate on the same training rows and score every one on the
    same unseen rows."""
    data = _load(store, contract, features)
    metric = contract.metric or ("roc_auc" if contract.task == "classification" else "rmse")
    n_classes = len(data.classes)
    if metric == "f1" and n_classes > 2:
        metric = "macro_f1"
    higher = HIGHER_IS_BETTER[metric]
    names = [f.name for f in features]
    run = TrainingRun(
        task=contract.task or "",
        metric=metric,
        higher_is_better=higher,
        split=data.split,
        train_rows=len(data.y_train),
        test_rows=len(data.y_test),
        dropped_rows=data.dropped,
        positive_class=data.classes[1] if n_classes == 2 else None,
        features=names,
        classes=list(data.classes),
        warnings=list(data.warnings),
    )
    models_dir.mkdir(parents=True, exist_ok=True)

    taken: set[str] = set()
    plan = list(candidates)
    if not any(c.family == "baseline" for c in plan):
        plan.insert(0, default_plan()[0])
    plan = plan[:MAX_CANDIDATES]
    if run.task == "classification":
        share = float(min(np.bincount(data.y_train))) / len(data.y_train)
        if share < RARE_SHARE and not any(c.params.get("balanced") for c in plan) \
                and len(plan) < MAX_CANDIDATES and not any(c.name == "boosting_balanced" for c in plan):
            plan.append(Candidate(
                "boosting_balanced", "boosting", {"balanced": 1.0},
                f"The rarest outcome is {share:.0%} of rows. This version gives rare rows more weight, so it "
                "cannot score well by ignoring them.", "harness"))
    folds = cv_folds(data.y_train, classify=run.task == "classification", time_ordered=data.time_ordered,
                     groups=data.groups_train)
    outputs: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}
    oof: dict[str, dict[int, tuple[np.ndarray, np.ndarray, np.ndarray | None]]] = {}
    for candidate in plan:
        result = ModelResult(candidate.name, candidate.family, dict(candidate.params), candidate.why, candidate.origin)
        run.results.append(result)
        result.error = check_candidate(candidate, taken)
        if result.error:
            continue
        taken.add(candidate.name.lower())
        result.params = dict(candidate.params)
        started = time.perf_counter()
        try:
            model = _pipeline(candidate, run.task, data.numeric, data.categorical)
            model.fit(data.x_train, data.y_train)
            result.test_score = _score(metric, model, data.x_test, data.y_test, n_classes)
            result.train_score = _score(metric, model, data.x_train, data.y_train, n_classes)
            for other in _extra_metrics(run.task, metric, n_classes):
                result.extra[other] = _score(other, model, data.x_test, data.y_test, n_classes)
            outputs[candidate.name] = _outputs(model, data.x_test, run.task)
            oof[candidate.name] = {}
            fold_scores = cv_scores(candidate, run.task, metric, data.x_train, data.y_train,
                                    data.numeric, data.categorical, folds, n_classes, collect=oof[candidate.name])
            if len(fold_scores) >= 2:
                result.cv_mean = float(np.mean(fold_scores))
                result.cv_std = float(np.std(fold_scores, ddof=1))
            if candidate.family != "baseline":
                result.importance = _importance(model, metric, data.x_test, data.y_test, names, n_classes,
                                                result.test_score, higher)
            if candidate.family == "tree":
                result.tree_text = tree_text(model, data.classes, run.task, names)
            joblib.dump(model, models_dir / f"{candidate.name}.joblib")
        except Exception as exc:  # noqa: BLE001 - one bad candidate must not stop the others
            result.error = f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"
        result.fit_seconds = round(time.perf_counter() - started, 2)

    _add_ensemble(run, data, plan, outputs, oof, metric, n_classes, names, models_dir)
    _judge(run)
    _prefer_simple_over_ensemble(run)
    _check_ties(run, data.y_test, outputs, n_classes)
    _evaluate_all(run, contract, features, data, outputs, oof)
    return run


def _add_ensemble(run: TrainingRun, data: _Data, plan: list[Candidate], outputs: dict, oof: dict, metric: str,
                  n_classes: int, names: list[str], models_dir: Path) -> None:
    """Average the best model of each family, when there are at least two families."""
    best_per_family: dict[str, ModelResult] = {}
    for r in run.results:
        if r.family == "baseline" or r.error or r.cv_mean is None or r.name not in outputs:
            continue
        held = best_per_family.get(r.family)
        if held is None or gain(r.cv_mean, held.cv_mean or 0.0, run.higher_is_better) > 0:
            best_per_family[r.family] = r
    picked = sorted(best_per_family.values(), key=lambda r: -gain(r.cv_mean or 0.0, 0.0, run.higher_is_better))[:3]
    if len(picked) < 2 or any(r.name == "ensemble" for r in run.results):
        return
    started = time.perf_counter()
    result = ModelResult("ensemble", "ensemble", {}, ENSEMBLE_WORDS + " Averages " +
                         ", ".join(r.name for r in picked) + ".", "harness")
    result.members = [{"name": r.name, "family": r.family, "params": dict(r.params)} for r in picked]
    try:
        common = set.intersection(*(set(oof[r.name]) for r in picked))
        fold_scores = []
        combined: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray | None]] = {}
        for k in sorted(common):
            val_idx = oof[picked[0].name][k][0]
            if run.task == "classification":
                proba = np.mean([oof[r.name][k][2] for r in picked], axis=0)
                pred = np.argmax(proba, axis=1)
            else:
                proba, pred = None, np.mean([oof[r.name][k][1] for r in picked], axis=0)
            combined[k] = (val_idx, pred, proba)
            fold_scores.append(_metric_from(metric, data.y_train[val_idx], pred, proba, n_classes))
        oof["ensemble"] = combined
        if len(fold_scores) >= 2:
            result.cv_mean = float(np.mean(fold_scores))
            result.cv_std = float(np.std(fold_scores, ddof=1))
        members = [joblib.load(models_dir / f"{r.name}.joblib") for r in picked]
        model = Ensemble(members, run.task)
        outputs["ensemble"] = _outputs(model, data.x_test, run.task)
        result.test_score = _metric_from(metric, data.y_test, *outputs["ensemble"], n_classes)
        result.train_score = _score(metric, model, data.x_train, data.y_train, n_classes)
        for other in _extra_metrics(run.task, metric, n_classes):
            result.extra[other] = _score(other, model, data.x_test, data.y_test, n_classes)
        result.importance = _importance(model, metric, data.x_test, data.y_test, names, n_classes,
                                        result.test_score, run.higher_is_better)
        joblib.dump(model, models_dir / "ensemble.joblib")
        result.notes.append("Its members were picked by the same cross-validation, so its cross-validated "
                            "score is a little optimistic.")
    except Exception as exc:  # noqa: BLE001
        result.error = f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"
    result.fit_seconds = round(time.perf_counter() - started, 2)
    run.results.append(result)


def _prefer_simple_over_ensemble(run: TrainingRun) -> None:
    """An ensemble is harder to explain and slower. It is recommended only when it is
    clearly ahead of the best single model in cross-validation."""
    ensemble = run.result("ensemble")
    if ensemble is None or not ensemble.recommended:
        return
    singles = [r for r in run.results if r.beats_baseline and r.family not in {"baseline", "ensemble"}
               and r.cv_mean is not None]
    if not singles:
        return
    best = max(singles, key=lambda r: gain(r.cv_mean or 0.0, 0.0, run.higher_is_better))
    lead = gain(ensemble.cv_mean or 0.0, best.cv_mean or 0.0, run.higher_is_better)
    margin = 0.005 if run.metric in BOUNDED else 0.01 * abs(best.cv_mean or 0.0)
    if lead < margin:
        ensemble.recommended = False
        best.recommended = True
        ensemble.notes.append(f"Only {lead:+.4f} ahead of {best.name} in cross-validation: not worth averaging "
                              f"{len(ensemble.members)} models, so {best.name} is recommended instead.")


def _evaluate_all(run: TrainingRun, contract: DataContract, features: list[Feature], data: _Data,
                  outputs: dict, oof: dict) -> None:
    kinds = [f.kind for f in features]
    for result in run.results:
        if result.family == "baseline" or result.error or not result.beats_baseline or result.name not in outputs:
            continue
        stacked = _stack_oof(oof.get(result.name, {}), data.y_train)
        if stacked is None:
            continue
        y_oof, oof_pred, oof_proba = stacked
        try:
            ev: Evaluation = evaluate(
                task=run.task, classes=run.classes, y_oof=y_oof, oof_pred=oof_pred, oof_proba=oof_proba,
                y_test=data.y_test, test_pred=outputs[result.name][0], test_proba=outputs[result.name][1],
                x_test=data.x_test, names=run.features, kinds=kinds,
                cost_false_alarm=contract.cost_false_alarm, cost_miss=contract.cost_miss,
                fairness_columns=contract.fairness_columns,
                derived={f.name for f in features if f.origin == "engineer"},
            )
            result.evaluation = ev.to_json()
        except Exception as exc:  # noqa: BLE001 - a report must not stop training
            result.notes.append(f"The extra checks could not run: {type(exc).__name__}: {str(exc)[:120]}")


def _rank(result: ModelResult) -> float:
    return result.cv_mean if result.cv_mean is not None else float(result.test_score or 0.0)


def _check_ties(run: TrainingRun, y_test: np.ndarray, outputs: dict, n_classes: int) -> None:
    """Is the recommended model really better than the others, or only by luck of
    which rows ended up in the test set?"""
    best = next((r for r in run.results if r.recommended), None)
    if best is None or best.name not in outputs:
        return
    for other in run.results:
        if other is best or other.family == "baseline" or other.name not in outputs or not other.beats_baseline:
            continue
        point, low, high = compare_on_test(run.metric, run.higher_is_better, y_test,
                                           outputs[best.name], outputs[other.name], n_classes)
        if not (low == low) or low <= 0:  # nan, or the interval reaches zero
            best.ties_with.append(other.name)
    if best.ties_with:
        names = ", ".join(best.ties_with)
        best.notes.append(
            f"Not clearly better than {names}: resampling the {run.test_rows} unseen rows, its lead could be zero. "
            "Treat them as tied and prefer the simpler or more readable one."
        )


def _judge(run: TrainingRun) -> None:
    """Compare with the baseline, flag anything suspicious, and mark the best.

    The best is chosen by cross-validated score on the training rows, not by the
    test score: picking the top test score among many candidates quietly fits the
    test rows. The test score stays the honest, once-only estimate."""
    baseline = run.baseline
    floor = baseline.test_score if baseline and baseline.test_score is not None else None
    best: ModelResult | None = None
    for result in run.results:
        if result.error or result.test_score is None:
            continue
        test, train = result.test_score, result.train_score
        if result.family == "baseline":
            continue
        if floor is None:
            result.beats_baseline = True
        else:
            result.beats_baseline = (test > floor + 1e-9) if run.higher_is_better else (test < floor - 1e-9)
        if not result.beats_baseline:
            result.notes.append("No better than doing nothing (the baseline). It cannot be approved.")
        if run.metric in BOUNDED and test >= 0.995:
            result.notes.append(
                "Almost perfect on unseen rows. That is rare with real data: look for a feature that gives the answer away."
            )
        if train is not None:
            if run.metric in BOUNDED and train - test > 0.10:
                result.notes.append(
                    f"Much better on the rows it trained on ({train:.3f}) than on unseen rows ({test:.3f}). "
                    "It is partly memorising."
                )
            elif run.metric in {"rmse", "mae"} and train > 0 and test > 1.5 * train:
                result.notes.append(
                    f"Its error on unseen rows ({test:,.3f}) is far above its error on training rows ({train:,.3f}). "
                    "It is partly memorising."
                )
        if result.beats_baseline and (best is None or gain(_rank(result), _rank(best), run.higher_is_better) > 0):
            best = result
    if best is not None:
        best.recommended = True
    else:
        run.warnings.append("No candidate beat the baseline. The features may not carry enough signal.")
