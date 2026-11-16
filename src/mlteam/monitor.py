"""After the model is live: is it still fit for the data it sees?

Three questions, answered for every batch of rows the model scores:

* Have the inputs moved?   Per feature, the batch is compared with the training
  rows using the Population Stability Index (PSI). The training values are cut
  into ten bins (or categories); PSI adds up, over the bins, how far the share
  of rows has moved: sum((now - then) * ln(now / then)). Rules of thumb:
  under 0.10 stable, 0.10 to 0.25 worth watching, above 0.25 a real shift.
* Have the outputs moved?  The same measure on the model's own predictions. If
  the inputs look fine but the predictions moved, something subtle changed.
* Is it still right?       Only knowable when the batch carries the answers.
  Then the model is scored on it and compared with its score at approval.

Drift in a feature the model barely uses matters little; drift in one it leans
on matters a lot, so the advice weighs each feature by its importance.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

STABLE, WATCH = 0.10, 0.25
EPS = 1e-4
MAX_CATEGORIES = 20
IMPORTANT_SHARE = 0.10   # a feature carrying at least this share of the model's reliance counts as important


# ---------------------------------------------------------------------------
# Reference: a snapshot of the training rows, saved when a model is approved
# ---------------------------------------------------------------------------


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _numeric_profile(values: list[Any]) -> dict[str, Any]:
    present = np.array([float(v) for v in values if not _is_missing(v)], dtype=float)
    missing = (len(values) - len(present)) / len(values) if values else 0.0
    if len(present) == 0:
        return {"type": "numeric", "edges": [], "shares": [], "missing": missing}
    distinct = np.unique(present)
    if len(distinct) <= 10:  # few distinct values: treat each as its own bin
        shares = [float(np.mean(present == v)) * (1 - missing) for v in distinct]
        return {"type": "discrete", "values": [float(v) for v in distinct], "shares": shares, "missing": missing}
    edges = sorted(set(float(q) for q in np.quantile(present, np.linspace(0.1, 0.9, 9))))
    counts = np.bincount(np.searchsorted(edges, present, side="left"), minlength=len(edges) + 1)
    shares = (counts / len(present) * (1 - missing)).tolist()
    return {"type": "numeric", "edges": edges, "shares": shares, "missing": missing}


def _category_profile(values: list[Any]) -> dict[str, Any]:
    present = [str(v) for v in values if not _is_missing(v)]
    missing = (len(values) - len(present)) / len(values) if values else 0.0
    counts: dict[str, int] = {}
    for v in present:
        counts[v] = counts.get(v, 0) + 1
    top = sorted(counts, key=lambda k: -counts[k])[:MAX_CATEGORIES]
    total = len(values) or 1
    shares = {k: counts[k] / total for k in top}
    other = sum(counts[k] for k in counts if k not in shares) / total
    return {"type": "category", "shares": shares, "other": other, "missing": missing}


def profile_values(values: list[Any], kind: str) -> dict[str, Any]:
    return _numeric_profile(values) if kind in {"numeric", "boolean"} else _category_profile(values)


def _shares_like(reference: dict[str, Any], values: list[Any]) -> tuple[list[float], list[float]]:
    """(reference shares, batch shares) over the same bins, missing included."""
    n = len(values) or 1
    missing_now = sum(1 for v in values if _is_missing(v)) / n
    if reference["type"] == "category":
        keys = list(reference["shares"])
        now = {k: 0 for k in keys}
        other = 0
        for v in values:
            if _is_missing(v):
                continue
            key = str(v)
            if key in now:
                now[key] += 1
            else:
                other += 1
        then = [*reference["shares"].values(), reference["other"], reference["missing"]]
        return then, [*(now[k] / n for k in keys), other / n, missing_now]
    present = [float(v) for v in values if not _is_missing(v)]
    if reference["type"] == "discrete":
        known = reference["values"]
        counts = [sum(1 for v in present if v == k) for k in known]
        unknown = len(present) - sum(counts)
        return ([*reference["shares"], 0.0, reference["missing"]],
                [*(c / n for c in counts), unknown / n, missing_now])
    edges = reference["edges"]
    counts = np.bincount(np.searchsorted(edges, np.array(present), side="left"), minlength=len(edges) + 1) \
        if present else np.zeros(len(edges) + 1)
    return [*reference["shares"], reference["missing"]], [*(counts / n).tolist(), missing_now]


def psi(reference: dict[str, Any], values: list[Any]) -> float:
    return psi_and_noise(reference, values, 0)[0]


def psi_and_noise(reference: dict[str, Any], values: list[Any], reference_rows: int) -> tuple[float, float]:
    """PSI, and the PSI that pure chance gives at these sample sizes. Two samples of
    the same data still differ a little; the smaller they are, the more. Expected
    chance level is about (bins - 1) * (1/batch rows + 1/reference rows)."""
    then, now = _shares_like(reference, values)
    total = 0.0
    used = 0
    for a, e in zip(now, then):
        if a == 0 and e == 0:
            continue
        used += 1
        a, e = max(a, EPS), max(e, EPS)
        total += (a - e) * math.log(a / e)
    n = max(len(values), 1)
    noise = (max(used, 2) - 1) * (1.0 / n + (1.0 / reference_rows if reference_rows else 0.0))
    return float(total), float(noise)


def prediction_values(fitted: Any, matrix: np.ndarray, task: str, classes: list[str]) -> tuple[list[Any], str]:
    """What gets watched on the output side: the probability of the positive class
    for yes/no problems, the predicted label for several classes, the value for numbers."""
    if task == "classification" and len(classes) == 2:
        return [float(p[1]) for p in fitted.predict_proba(matrix)], "numeric"
    if task == "classification":
        return [classes[int(i)] for i in fitted.predict(matrix)], "category"
    return [float(v) for v in fitted.predict(matrix)], "numeric"


def build_reference(
    features: list[Any], matrix: np.ndarray, predictions: list[Any], prediction_kind: str
) -> dict[str, Any]:
    return {
        "rows": int(matrix.shape[0]),
        "features": {f.name: profile_values(list(matrix[:, i]), f.kind) for i, f in enumerate(features)},
        "prediction": profile_values(predictions, prediction_kind),
        "prediction_kind": prediction_kind,
    }


# ---------------------------------------------------------------------------
# Drift and advice for one batch
# ---------------------------------------------------------------------------


def level(value: float) -> str:
    return "stable" if value < STABLE else "watch" if value < WATCH else "shifted"


@dataclass
class Health:
    status: str                       # stable | watch | retrain
    reasons: list[str] = field(default_factory=list)
    feature_psi: dict[str, float] = field(default_factory=dict)     # as measured
    prediction_psi: float | None = None
    chance_psi: float = 0.0           # what chance alone gives at this batch size; judged on the excess
    small_batch: bool = False


def assess(
    reference: dict[str, Any],
    features: list[Any],
    matrix: np.ndarray,
    predictions: list[Any],
    importance: list[tuple[str, float]] | None = None,
    score: dict[str, Any] | None = None,
    score_dropped: bool = False,
) -> Health:
    health = Health(status="stable")
    ref_rows = int(reference.get("rows") or 0)
    excess: dict[str, float] = {}
    noises: list[float] = []
    for i, f in enumerate(features):
        ref = reference["features"].get(f.name)
        if ref is not None:
            value, noise = psi_and_noise(ref, list(matrix[:, i]), ref_rows)
            health.feature_psi[f.name] = round(value, 4)
            excess[f.name] = max(0.0, value - noise)
            noises.append(noise)
    prediction_excess: float | None = None
    if reference.get("prediction") is not None:
        value, noise = psi_and_noise(reference["prediction"], predictions, ref_rows)
        health.prediction_psi = round(value, 4)
        prediction_excess = max(0.0, value - noise)
        noises.append(noise)
    health.chance_psi = round(float(np.median(noises)) if noises else 0.0, 4)
    health.small_batch = health.chance_psi >= STABLE / 2

    weights = {n: max(0.0, v) for n, v in (importance or [])}
    total = sum(weights.values())
    important = {n for n, v in weights.items() if total > 0 and v / total >= IMPORTANT_SHARE}

    # Judged on the excess over chance, so a small batch of ordinary rows is not a false alarm.
    shifted = sorted((n for n, v in excess.items() if v >= WATCH), key=lambda n: -excess[n])
    watched = sorted((n for n, v in excess.items() if STABLE <= v < WATCH), key=lambda n: -excess[n])
    key_shifted = [n for n in shifted if n in important] if important else shifted

    if score_dropped and score is not None:
        expected = score.get("expected", score.get("original"))
        spread = f" ± {score['expected_spread']:.3f}" if score.get("expected_spread") else ""
        health.reasons.append(
            f"On this batch the model scored {score['metric']} {score['value']:.3f}, against an expected "
            f"{expected:.3f}{spread} at approval. It has got worse, by more than chance explains."
        )
    if key_shifted:
        names = ", ".join(f"{n} (PSI {health.feature_psi[n]:.2f})" for n in key_shifted[:4])
        health.reasons.append(f"Features the model relies on have shifted: {names}.")
    if prediction_excess is not None and prediction_excess >= WATCH:
        health.reasons.append(
            f"The model's own predictions have shifted (PSI {health.prediction_psi:.2f}): it is answering "
            "differently from how it did on the training rows."
        )
    if health.reasons:
        health.status = "retrain"
    else:
        minor = [n for n in shifted if n not in key_shifted] + watched
        if minor or (prediction_excess is not None and prediction_excess >= STABLE):
            health.status = "watch"
            if minor:
                names = ", ".join(f"{n} (PSI {health.feature_psi[n]:.2f})" for n in minor[:4])
                health.reasons.append(f"Some inputs are moving: {names}. Not yet a reason to retrain.")
            if prediction_excess is not None and prediction_excess >= STABLE:
                health.reasons.append(f"Predictions are moving a little (PSI {health.prediction_psi:.2f}).")
    if health.small_batch:
        health.reasons.append(
            f"Small batch: chance alone gives a PSI near {health.chance_psi:.2f} here, so only larger shifts count."
        )
    if health.status == "retrain" and score is None:
        health.reasons.append("Score a batch that has the answers to measure the damage, then train a challenger.")
    return health


ADVICE = {
    "stable": "Stable. Inputs and predictions look like the training data.",
    "watch": "Watch. Something is moving; keep scoring and check again.",
    "retrain": "Retrain recommended.",
}


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


def append_record(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")


def read_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def health_to_json(health: Health) -> dict[str, Any]:
    return asdict(health)
