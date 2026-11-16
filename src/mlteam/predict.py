"""Scoring rows the model has not seen.

Two cases, one code path:

* the held-back test file that came in with the data (a Kaggle-style layout);
* a brand-new file, scored later against a finished run.

In both, features are computed by the approved expressions under the same time
rule, and the fitted model is applied. Before trusting the output the harness
compares the new rows with the training rows: a model can only be relied on
for rows that look like what it learned from. Differences are reported, not
hidden. If the new file happens to contain the answers, the model is also
scored on it, which is the most honest check there is.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from mlteam.contract import DataContract, base_sql, contract_from_json, target_stats
from mlteam.features import Feature, time_rule_ctes
from mlteam.monitor import ADVICE, append_record, assess, health_to_json, prediction_values
from mlteam.profile import DatasetProfile
from mlteam.relations import Layout
from mlteam.runstore import RunPaths, now_iso
from mlteam.store import Store, StoreError, quote_ident
from mlteam.train import HIGHER_IS_BETTER, TrainingRun, _column, _score

MISSING_SHIFT = 20.0   # percentage points
RATE_SHIFT = 0.20      # share of rows predicted as the positive class vs. seen in training
SCORE_DROP = 0.05      # how far below the original score counts as "worse on this file"
NEW_ROWS = "__scoring_rows"


class PredictError(Exception):
    """Raised when a table cannot be scored."""


@dataclass
class Prediction:
    table: str
    rows: int
    model: str
    model_version: int
    task: str
    output: str
    submission: str | None = None
    positive_class: str | None = None
    counts: dict[str, int] = field(default_factory=dict)       # classification: label -> rows
    summary: dict[str, float] = field(default_factory=dict)    # regression: min / mean / max
    notes: list[str] = field(default_factory=list)
    source: str | None = None                                   # a new file: its name
    source_sha256: str | None = None
    score: dict[str, Any] | None = None                         # only when the file contains the answers
    health: dict[str, Any] | None = None                        # drift against the training rows, and advice
    version: int = 0
    created_at: str = ""

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _Model:
    name: str
    version: int
    task: str
    classes: list[str]
    positive_class: str | None
    metric: str
    test_score: float | None
    path: Path
    reference: dict[str, Any] | None = None
    importance: list[tuple[str, float]] | None = None
    expected: float | None = None         # cross-validated score at approval: the stable expectation
    expected_spread: float | None = None  # its spread across folds


def scoring_sql(
    contract: DataContract, table: str, features: list[Feature], keys: list[str], answer: str | None = None
) -> str:
    """The approved feature expressions, evaluated for the rows of ``table``
    instead of the training table. The time rule follows the new rows."""
    scoped = copy.copy(contract)
    scoped.base_table, scoped.stack = table, []
    scoped.dedupe = False  # every row handed in gets a prediction, copies included
    selects = [f"b.{quote_ident(k)}" for k in keys]
    selects += [f"({f.sql}) AS {quote_ident(f.name)}" for f in features]
    if answer:
        selects.append(f"b.{quote_ident(answer)} AS __y")
    order = ", ".join(f"b.{quote_ident(k)}" for k in keys)
    tail = f" ORDER BY {order}" if order else ""
    return f"{time_rule_ctes(scoped)}SELECT {', '.join(selects)} FROM {quote_ident(table)} AS b{tail}"


def _next_version(run: RunPaths) -> int:
    version = 1
    while run.predictions_version(version).exists():
        version += 1
    return version


def _compare_with_training(
    store: Store, contract: DataContract, features: list[Feature], matrix: np.ndarray
) -> list[str]:
    """Plain-language notes on how the new rows differ from the training rows."""
    notes: list[str] = []
    ctes = time_rule_ctes(contract)
    for i, feature in enumerate(features):
        column = matrix[:, i]
        if feature.kind in {"numeric", "boolean"}:
            missing = sum(1 for v in column if v is None or v != v)
        else:
            missing = sum(1 for v in column if v is None)
        share = 100.0 * missing / len(column) if len(column) else 0.0
        if abs(share - feature.null_pct) >= MISSING_SHIFT:
            notes.append(
                f"{feature.name} is missing on {share:.0f}% of these rows, against {feature.null_pct:.0f}% "
                "in training. The model has little experience with rows like that."
            )
        if feature.kind not in {"numeric", "boolean"}:
            try:
                _, seen = store.query(
                    f"{ctes}SELECT DISTINCT ({feature.sql})::VARCHAR FROM {base_sql(contract)} AS b"
                )
            except StoreError:
                continue
            known = {row[0] for row in seen if row[0] is not None}
            new = sorted({str(v) for v in column if v is not None} - known)
            if new:
                rows = sum(1 for v in column if v is not None and str(v) not in known)
                shown = ", ".join(new[:4]) + (" ..." if len(new) > 4 else "")
                notes.append(
                    f"{feature.name} has {len(new)} value(s) never seen in training ({shown}), on {rows:,} rows. "
                    "The model treats them as unknown."
                )
    return notes


def _score_rows(
    store: Store,
    contract: DataContract,
    features: list[Feature],
    model: _Model,
    run: RunPaths,
    *,
    table: str,
    columns: list[str],
    expected_rows: int,
    sample_columns: list[str] | None = None,
) -> Prediction:
    if not features:
        raise PredictError("No approved features to compute.")
    if not model.path.exists():
        raise PredictError(f"The trained model file for {model.name} is missing.")
    available = {c.lower(): c for c in columns}
    keys = [available[k.lower()] for k in contract.grain_keys if k.lower() in available]
    answer = available.get(contract.target.lower()) if contract.target else None

    try:
        _, rows = store.query(scoring_sql(contract, table, features, keys, answer))
    except StoreError as exc:
        raise PredictError(
            "This data cannot be scored: a feature needs a column it does not have, or a column holds "
            f"the wrong kind of value. {str(exc).splitlines()[0]}"
        ) from exc
    if not rows:
        raise PredictError("There are no rows to score.")
    if len(rows) != expected_rows:
        raise PredictError(
            f"Scoring produced {len(rows):,} rows from {expected_rows:,}. Nothing was written."
        )

    offset = len(keys)
    matrix = np.empty((len(rows), len(features)), dtype=object)
    for i, feature in enumerate(features):
        try:
            matrix[:, i] = _column([r[offset + i] for r in rows], feature.kind)
        except (TypeError, ValueError) as exc:
            raise PredictError(
                f"{feature.name} should be a number, but this data has something else in it ({exc})."
            ) from exc

    fitted = joblib.load(model.path)
    version = _next_version(run)
    output = run.predictions_version(version)
    prediction = Prediction(
        table=table, rows=len(rows), model=model.name, model_version=model.version, task=model.task,
        output=output.name, positive_class=model.positive_class, version=version, created_at=now_iso(),
    )
    header = list(keys)
    lines: list[list[Any]] = []
    answers: list[Any] = []
    classes = model.classes
    try:
        if model.task == "classification":
            proba = fitted.predict_proba(matrix)
            labels = [classes[int(i)] for i in fitted.predict(matrix)]
            answers = labels
            if len(classes) == 2:
                header += ["prediction", f"probability_of_{classes[1]}"]
                lines = [[*r[:offset], label, round(float(p[1]), 6)]
                         for r, label, p in zip(rows, labels, proba, strict=True)]
            else:
                header += ["prediction", "probability"]
                lines = [[*r[:offset], label, round(float(max(p)), 6)]
                         for r, label, p in zip(rows, labels, proba, strict=True)]
            for label in labels:
                prediction.counts[label] = prediction.counts.get(label, 0) + 1
        else:
            values = [float(v) for v in fitted.predict(matrix)]
            answers = values
            header += ["prediction"]
            lines = [[*r[:offset], round(v, 6)] for r, v in zip(rows, values, strict=True)]
            width = getattr(fitted, "half_width", None)
            if width:  # the range that held the true value for 80% of cross-validated rows
                header += ["low_80", "high_80"]
                lines = [[*line, round(v - width, 6), round(v + width, 6)]
                         for line, v in zip(lines, values, strict=True)]
            prediction.summary = {"min": min(values), "mean": sum(values) / len(values), "max": max(values)}
    except Exception as exc:  # noqa: BLE001 - a model that cannot run on these rows is a user-facing problem
        raise PredictError(f"The model could not be applied to these rows: {str(exc).splitlines()[0][:200]}") from exc

    if model.task == "classification" and len(classes) == 2:
        seen = dict(target_stats(store, contract).get("top") or [])
        total = sum(seen.values())
        if total:
            trained_rate = seen.get(classes[1], 0) / total
            rate = prediction.counts.get(classes[1], 0) / len(rows)
            if abs(rate - trained_rate) >= RATE_SHIFT:
                prediction.notes.append(
                    f"{rate:.0%} of these rows are predicted as {classes[1]}, against {trained_rate:.0%} in "
                    "the training data. Either these rows really are different, or the model is off here."
                )
    prediction.notes += _compare_with_training(store, contract, features, matrix)
    if not keys:
        prediction.notes.append(
            "There is no row key in this data, so predictions are listed in its own order with no identifier."
        )
    dropped = False
    if answer:
        dropped = _score_against_answers(prediction, fitted, matrix, [r[-1] for r in rows], model)
    if model.reference is not None:
        watched, _kind = prediction_values(fitted, matrix, model.task, classes)
        prediction.health = health_to_json(assess(
            model.reference, features, matrix, watched, model.importance, prediction.score, dropped))

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(lines)

    # A Kaggle-style sample submission names the exact two columns expected: an id and the target.
    if sample_columns and contract.target and len(keys) == 1 and len(sample_columns) == 2:
        id_col = next((n for n in sample_columns if n.lower() == keys[0].lower()), None)
        target_col = next((n for n in sample_columns if n.lower() == contract.target.lower()), None)
        if id_col and target_col:
            path = run.submission_version(version)
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(sample_columns)
                for row, value in zip(rows, answers, strict=True):
                    pair = {id_col: row[0], target_col: value}
                    writer.writerow([pair[n] for n in sample_columns])
            prediction.submission = path.name
    return prediction


def _score_against_answers(
    prediction: Prediction, fitted: Any, matrix: np.ndarray, truth: list[Any], model: _Model
) -> bool:
    """The file carries the real answers, so measure the model on it. True when it did clearly worse."""
    classes = model.classes
    if model.task == "classification":
        index = {c: i for i, c in enumerate(classes)}
        usable = [i for i, v in enumerate(truth) if v is not None and str(v) in index]
        unknown = sorted({str(v) for v in truth if v is not None and str(v) not in index})
        if unknown:
            prediction.notes.append(
                f"The answers in this data include value(s) the model was never trained on: {', '.join(unknown[:4])}."
            )
        y = np.array([index[str(truth[i])] for i in usable])
        if len(usable) < 10 or len(set(y.tolist())) < len(classes):
            prediction.notes.append(
                "This data has answers, but too few of each outcome to score the model on it."
            )
            return False
    else:
        usable = [i for i, v in enumerate(truth) if v is not None]
        if len(usable) < 10:
            return False
        y = np.array([float(truth[i]) for i in usable])
    try:
        value = _score(model.metric, fitted, matrix[usable], y, len(classes))
    except Exception:  # noqa: BLE001 - the predictions are still valid without a score
        return False
    # Compare with the cross-validated score where there is one: a single test score on a few
    # dozen rows can be lucky, and a fresh batch would then look "worse" for no reason.
    expected = model.expected if model.expected is not None else model.test_score
    spread = model.expected_spread or 0.0
    prediction.score = {"metric": model.metric, "value": value, "rows": len(usable),
                        "original": model.test_score, "expected": expected, "expected_spread": spread}
    if expected is not None:
        higher = HIGHER_IS_BETTER.get(model.metric, True)
        worse = (expected - value) if higher else (value - expected)
        floor = SCORE_DROP if model.metric not in {"rmse", "mae", "log_loss"} else 0.25 * abs(expected)
        limit = max(floor, 2 * spread)
        if worse > limit:
            prediction.notes.append(
                f"The model does worse on this data ({model.metric} {value:.3f}) than expected at approval "
                f"({expected:.3f} ± {spread:.3f}). Something has changed since it was trained."
            )
            return True
    return False


def record_batch(run: RunPaths, prediction: Prediction) -> None:
    """One line per scored batch, so drift can be followed over time."""
    health = prediction.health or {}
    feature_psi = health.get("feature_psi") or {}
    top = max(feature_psi.items(), key=lambda kv: kv[1]) if feature_psi else (None, None)
    append_record(run.monitoring, {
        "at": prediction.created_at, "source": prediction.source or prediction.table, "rows": prediction.rows,
        "model": prediction.model, "model_version": prediction.model_version, "output": prediction.output,
        "status": health.get("status"), "reasons": health.get("reasons", []),
        "feature_psi": feature_psi, "prediction_psi": health.get("prediction_psi"),
        "max_feature": top[0], "max_psi": top[1], "score": prediction.score,
    })


def _card_extras(run: RunPaths, card: dict[str, Any]) -> tuple[dict[str, Any] | None, list[tuple[str, float]] | None]:
    reference = None
    if card.get("reference") and (run.root / str(card["reference"])).exists():
        reference = json.loads((run.root / str(card["reference"])).read_text(encoding="utf-8"))
    importance = [(str(n), float(v)) for n, v in card.get("importance") or []]
    return reference, importance or None


def _apply_card(model: _Model, run: RunPaths, card: dict[str, Any]) -> None:
    model.reference, model.importance = _card_extras(run, card)
    model.expected = card.get("cv_mean")
    model.expected_spread = card.get("cv_std")


def predict_table(
    store: Store,
    contract: DataContract,
    profile: DatasetProfile,
    layout: Layout,
    features: list[Feature],
    training: TrainingRun,
    model_name: str,
    model_version: int,
    run: RunPaths,
    table: str,
    card: dict[str, Any] | None = None,
) -> Prediction:
    """Score a table that came in with the data: the held-back test file."""
    target_table = profile.table(table)
    if target_table is None:
        raise PredictError(f"No table called {table}.")
    best = training.result(model_name)
    model = _Model(
        name=model_name, version=model_version, task=training.task, classes=list(training.classes),
        positive_class=training.positive_class, metric=training.metric,
        test_score=best.test_score if best else None, path=run.models / f"{model_name}.joblib",
    )
    if card is not None:
        _apply_card(model, run, card)
        model.path = run.root / str(card.get("artifact", f"models/{model_name}.joblib"))
    sample = profile.table(layout.submission) if layout.submission else None
    prediction = _score_rows(
        store, contract, features, model, run, table=table, columns=target_table.column_names,
        expected_rows=target_table.rows, sample_columns=sample.column_names if sample else None,
    )
    record_batch(run, prediction)
    return prediction


def _load_json(path: Path, what: str) -> dict[str, Any]:
    if not path.exists():
        raise PredictError(f"This run has no {what} ({path.name} is missing).")
    return json.loads(path.read_text(encoding="utf-8"))


def predict_file(run: RunPaths, file: Path) -> Prediction:
    """Score a brand-new file against a finished run, using exactly what was
    approved in it: that contract version, those features, that fitted model."""
    file = Path(file).expanduser()
    if not file.is_file():
        raise PredictError(f"File not found: {file}")
    card = _load_json(run.model_latest, "approved model")
    contract = contract_from_json(_load_json(run.contract_version(int(card["contract_version"])), "contract"))
    feature_set = _load_json(run.features_version(int(card["features_version"])), "feature list")
    by_name = {f["name"]: f for f in feature_set["features"]}
    features = [Feature(**by_name[n]) for n in feature_set["selected"] if n in by_name]
    classes, task = card.get("classes"), card.get("task")
    if classes is None or task is None:  # a model approved before these were written to the card
        training = _load_json(run.training, "training record")
        classes, task = training.get("classes", []), training.get("task", contract.task)
    model = _Model(
        name=str(card["model"]), version=int(card["version"]), task=str(task), classes=[str(c) for c in classes],
        positive_class=card.get("positive_class"), metric=str(card["metric"]),
        test_score=card.get("test_score"), path=run.root / str(card["artifact"]),
    )
    _apply_card(model, run, card)

    digest = hashlib.sha256()
    with file.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)

    try:
        store = Store.beside(run.db, file, NEW_ROWS)
    except StoreError as exc:
        raise PredictError(str(exc)) from exc
    try:
        described, _ = [r[0] for r in store.query(f"DESCRIBE {quote_ident(NEW_ROWS)}")[1]], None
        total = int(store.scalar(f"SELECT count(*) FROM {quote_ident(NEW_ROWS)}") or 0)
        base_columns = [c for c in _base_columns(store, contract) if c != contract.target]
        have = {c.lower() for c in described}
        missing = [c for c in base_columns if c.lower() not in have and c not in contract.drop_columns]
        try:
            prediction = _score_rows(
                store, contract, features, model, run, table=NEW_ROWS, columns=described, expected_rows=total,
            )
        except PredictError as exc:
            if missing:
                raise PredictError(
                    f"{exc} Columns the training table had that this file lacks: {', '.join(missing)}."
                ) from exc
            raise
    finally:
        store.close()
    prediction.table = file.name
    prediction.source = file.name
    prediction.source_sha256 = digest.hexdigest()
    record_batch(run, prediction)
    return prediction


def _base_columns(store: Store, contract: DataContract) -> list[str]:
    try:
        return [str(r[0]) for r in store.query(f"DESCRIBE {quote_ident(contract.base_table)}")[1]]
    except StoreError:
        return []
