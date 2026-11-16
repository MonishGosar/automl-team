"""Champion and challenger: retrain on new answers, but switch only on evidence.

When a batch arrives with the real answers, a challenger is trained: the same
model family and settings as the live model (the champion), fitted on the
original training rows plus part of the new rows. The rest of the new rows are
held back. Both models are then scored on those held-back new rows, which
neither has seen, and the gap is checked by resampling them: is the challenger
better by more than luck?

Both are also scored on the original locked test rows, to catch a challenger
that learned the new pattern by forgetting the old one.

Nothing switches automatically. Promotion is a separate, recorded decision, and
a challenger that is clearly worse cannot be promoted.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from mlteam.contract import DataContract, contract_from_json
from mlteam.features import Feature
from mlteam.monitor import build_reference, prediction_values
from mlteam.predict import NEW_ROWS, PredictError, scoring_sql
from mlteam.runstore import RunPaths, Trace, now_iso, write_json
from mlteam.store import Store, StoreError, quote_ident
from mlteam.train import (
    Delivered,
    build_model,
    HIGHER_IS_BETTER,
    SEED,
    _column,
    _load,
    _metric_from,
    _outputs,
    compare_on_test,
)

HOLDOUT_SHARE = 0.3
MIN_NEW_ROWS = 40


@dataclass
class Challenge:
    version: int
    champion: str
    champion_version: int
    family: str
    params: dict[str, float]
    source: str
    source_sha256: str
    metric: str
    higher_is_better: bool
    new_rows: int
    new_train_rows: int
    holdout_rows: int
    holdout_how: str
    original_train_rows: int
    champion_new: float
    challenger_new: float
    challenger_spread: float          # bootstrap spread of its score on the held-back rows
    gain_new: float
    gain_low: float
    gain_high: float
    champion_old: float | None
    challenger_old: float | None
    verdict: str                      # better | no clear difference | worse
    artifact: str
    notes: list[str] = field(default_factory=list)
    created_at: str = ""
    promoted_as: int | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    def sentence(self) -> str:
        return {
            "better": "The challenger is better on the new rows, by more than luck.",
            "worse": "The challenger is worse on the new rows. Keep the champion.",
            "no clear difference": "No clear difference on the new rows: the gap could be luck.",
        }[self.verdict]


def _read(path: Path, what: str) -> dict[str, Any]:
    if not path.exists():
        raise PredictError(f"This run has no {what} ({path.name} is missing).")
    return json.loads(path.read_text(encoding="utf-8"))


def _approved(run: RunPaths) -> tuple[dict[str, Any], DataContract, list[Feature]]:
    card = _read(run.model_latest, "approved model")
    contract = contract_from_json(_read(run.contract_version(int(card["contract_version"])), "contract"))
    feature_set = _read(run.features_version(int(card["features_version"])), "feature list")
    by_name = {f["name"]: f for f in feature_set["features"]}
    features = [Feature(**by_name[n]) for n in feature_set["selected"] if n in by_name]
    return card, contract, features


def _next(run: RunPaths, pattern: str) -> int:
    version = 1
    while (run.root / pattern.format(version)).exists():
        version += 1
    return version


def challenge(run: RunPaths, file: Path) -> Challenge:
    file = Path(file).expanduser()
    if not file.is_file():
        raise PredictError(f"File not found: {file}")
    card, contract, features = _approved(run)
    if not contract.target:
        raise PredictError("The approved contract has no target.")
    classes = [str(c) for c in card.get("classes") or []]
    task = str(card.get("task") or contract.task)
    metric = str(card["metric"])
    higher = HIGHER_IS_BETTER[metric]
    digest = hashlib.sha256(file.read_bytes()).hexdigest()

    try:
        store = Store.beside(run.db, file, NEW_ROWS)
    except StoreError as exc:
        raise PredictError(str(exc)) from exc
    try:
        columns = [r[0] for r in store.query(f"DESCRIBE {quote_ident(NEW_ROWS)}")[1]]
        by_lower = {c.lower(): c for c in columns}
        answer = by_lower.get(contract.target.lower())
        if answer is None:
            raise PredictError(
                f"This file has no {contract.target} column. A challenger needs the real answers to learn from."
            )
        keys = [by_lower[k.lower()] for k in contract.grain_keys if k.lower() in by_lower]
        time_col = by_lower.get(contract.time_column.lower()) if contract.time_column else None
        sql = scoring_sql(contract, NEW_ROWS, features, keys, answer)
        if time_col:  # carry the time along so the newest rows can be held back
            sql = sql.replace(" AS __y FROM", f" AS __y, b.{quote_ident(time_col)} AS __t FROM", 1)
        try:
            _, rows = store.query(sql)
        except StoreError as exc:
            raise PredictError(f"This file cannot be used: {str(exc).splitlines()[0]}") from exc
        original = _load(store, contract, features)
    finally:
        store.close()

    offset = len(keys)
    label_at = offset + len(features)
    labelled = [r for r in rows if r[label_at] is not None]
    notes: list[str] = []
    if len(labelled) < len(rows):
        notes.append(f"{len(rows) - len(labelled)} rows had no answer and were left out.")
    if task == "classification":
        index = {c: i for i, c in enumerate(classes)}
        unknown = sorted({str(r[label_at]) for r in labelled if str(r[label_at]) not in index})
        if unknown:
            raise PredictError(f"The answers include value(s) the model has never seen: {', '.join(unknown[:4])}.")
        y_new = np.array([index[str(r[label_at])] for r in labelled])
    else:
        y_new = np.array([float(r[label_at]) for r in labelled])
    if len(labelled) < MIN_NEW_ROWS:
        raise PredictError(f"Only {len(labelled)} rows with answers. At least {MIN_NEW_ROWS} are needed.")
    x_new = np.empty((len(labelled), len(features)), dtype=object)
    for i, f in enumerate(features):
        x_new[:, i] = _column([r[offset + i] for r in labelled], f.kind)

    n_hold = max(10, int(round(len(labelled) * HOLDOUT_SHARE)))
    if time_col and all(r[-1] is not None for r in labelled):
        order = np.argsort(np.array([str(r[-1]) for r in labelled]), kind="stable")
        train_idx, hold_idx = order[:-n_hold], order[-n_hold:]
        how = f"the latest {n_hold} rows by {time_col}"
    else:
        stratify = y_new if task == "classification" and min(np.bincount(y_new)) >= 2 else None
        train_idx, hold_idx = train_test_split(np.arange(len(labelled)), test_size=n_hold, random_state=SEED,
                                               stratify=stratify)
        how = f"{n_hold} rows picked at random" + (", same mix of outcomes" if stratify is not None else "")
    n_classes = len(classes)
    if task == "classification" and len(set(y_new[hold_idx].tolist())) < n_classes:
        raise PredictError("The held-back new rows do not contain every outcome, so the models cannot be compared.")

    champion = joblib.load(run.root / str(card["artifact"]))
    family = str(card["family"])
    params = {k: float(v) for k, v in (card.get("params") or {}).items()}
    challenger = build_model(family, params, card.get("members"), task, original.numeric, original.categorical)
    challenger.fit(np.vstack([original.x_train, x_new[train_idx]]),
                   np.concatenate([original.y_train, y_new[train_idx]]))
    delivery = card.get("delivery") or {}
    if any(delivery.get(k) is not None for k in ("platt", "cutoff", "half_width")):
        # Same decisions on top as the champion, so the two are compared like for like.
        challenger = Delivered(challenger, platt=delivery.get("platt"), cutoff=delivery.get("cutoff"),
                               half_width=delivery.get("half_width"))

    x_hold, y_hold = x_new[hold_idx], y_new[hold_idx]
    champ_out, chall_out = _outputs(champion, x_hold, task), _outputs(challenger, x_hold, task)
    champion_new = _metric_from(metric, y_hold, *champ_out, n_classes)
    challenger_new = _metric_from(metric, y_hold, *chall_out, n_classes)
    point, low, high = compare_on_test(metric, higher, y_hold, chall_out, champ_out, n_classes)
    rng = np.random.default_rng(SEED)
    resampled = []
    for _ in range(300):
        idx = rng.integers(0, len(y_hold), len(y_hold))
        try:
            resampled.append(_metric_from(metric, y_hold[idx], chall_out[0][idx],
                                          None if chall_out[1] is None else chall_out[1][idx], n_classes))
        except ValueError:
            continue
    spread = float(np.std(resampled)) if len(resampled) > 10 else 0.0
    verdict = "better" if low == low and low > 0 else "worse" if high == high and high < 0 else "no clear difference"

    champion_old = challenger_old = None
    try:
        champion_old = _metric_from(metric, original.y_test, *_outputs(champion, original.x_test, task), n_classes)
        challenger_old = _metric_from(metric, original.y_test, *_outputs(challenger, original.x_test, task), n_classes)
        forgot = (champion_old - challenger_old) if higher else (challenger_old - champion_old)
        if forgot > 0.05:
            notes.append(
                f"On the original test rows the challenger scores {challenger_old:.3f} against the champion's "
                f"{champion_old:.3f}: it has partly forgotten the old pattern. Fine if the old pattern is gone for good."
            )
    except ValueError:
        pass
    if n_hold < 60:
        notes.append(f"Only {n_hold} held-back new rows: only a large gap can count as more than luck.")

    version = _next(run, "challenger.v{}.json")
    artifact = f"models/{card['model']}__challenger_v{version}.joblib"
    joblib.dump(challenger, run.root / artifact)
    result = Challenge(
        version=version, champion=str(card["model"]), champion_version=int(card["version"]), family=family,
        params=params, source=file.name, source_sha256=digest, metric=metric, higher_is_better=higher,
        new_rows=len(labelled), new_train_rows=len(train_idx), holdout_rows=n_hold, holdout_how=how,
        original_train_rows=len(original.y_train), champion_new=champion_new, challenger_new=challenger_new,
        challenger_spread=spread,
        gain_new=point, gain_low=low, gain_high=high, champion_old=champion_old, challenger_old=challenger_old,
        verdict=verdict, artifact=artifact, notes=notes, created_at=now_iso(),
    )
    write_json(run.root / f"challenger.v{version}.json", result.to_json())
    # Keep what is needed to rebuild the drift reference if this challenger is promoted.
    joblib.dump({"x": np.vstack([original.x_train, x_new[train_idx]]), "features": [f.name for f in features]},
                run.root / f"models/challenger_v{version}_rows.joblib")
    Trace(run.trace).log("challenger", **result.to_json())
    return result


def load_challenge(run: RunPaths, version: int) -> Challenge:
    return Challenge(**_read(run.root / f"challenger.v{version}.json", f"challenger v{version}"))


def promote(run: RunPaths, version: int, approved_by: str = "human") -> Path:
    """Make a challenger the live model. A new model version is written; the old
    one stays on record. A challenger that did clearly worse is refused."""
    result = load_challenge(run, version)
    if result.promoted_as is not None:
        raise PredictError(f"Challenger v{version} was already promoted as model v{result.promoted_as}.")
    if result.verdict == "worse":
        raise PredictError("This challenger did worse than the champion on the new rows, so it cannot be promoted.")
    card, contract, features = _approved(run)
    if int(card["version"]) != result.champion_version:
        raise PredictError(
            f"The live model changed since this challenger was trained (now v{card['version']}). Train a new one."
        )
    model_version = _next(run, "model.v{}.json")
    fitted = joblib.load(run.root / result.artifact)
    rows = joblib.load(run.root / f"models/challenger_v{version}_rows.joblib")
    watched, kind = prediction_values(fitted, rows["x"], str(card.get("task")), [str(c) for c in card["classes"]])
    reference = run.reference_version(model_version)
    write_json(reference, build_reference(features, rows["x"], watched, kind))

    new_card = dict(card)
    new_card.update({
        "version": model_version,
        "artifact": result.artifact,
        "reference": reference.name,
        "replaces": card["version"],
        "challenger_version": version,
        "trained_on_extra": {"file": result.source, "sha256": result.source_sha256, "rows": result.new_train_rows},
        # The freshest unseen evidence: the held-back new rows. The cross-validated estimate no longer
        # applies, so the expectation for later batches becomes this score.
        "test_score": result.challenger_new,
        "test_rows": result.holdout_rows,
        "cv_mean": result.challenger_new,
        "cv_std": result.challenger_spread,
        "was_recommended": result.verdict == "better",
        "harness_notes": [result.sentence(), *result.notes],
        "approved_by": approved_by,
        "approved_at": now_iso(),
    })
    path = run.model_version(model_version)
    write_json(path, new_card)
    write_json(run.model_latest, new_card)
    result.promoted_as = model_version
    write_json(run.root / f"challenger.v{version}.json", result.to_json())
    Trace(run.trace).log("promotion", challenger=version, model_version=model_version, replaces=card["version"],
                         verdict=result.verdict, approved_by=approved_by)
    return path
