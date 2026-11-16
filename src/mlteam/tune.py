"""Bayesian optimisation of model settings, on a budget.

How one try works:
  1. A sampler proposes settings for a model family. Optuna's TPE sampler keeps
     two density models, one of settings that scored well so far and one of the
     rest, and proposes settings where "good" is much likelier than "bad". The
     first few tries per family are random, to give it something to learn from.
  2. The harness scores the settings by cross-validation on the training rows
     only. The locked test rows are never touched here.
  3. After each fold the running score is reported. If it is already below the
     median of earlier tries at the same fold, the try is stopped ("pruned"):
     a cheap partial result spares the cost of the full one.

How the budget is spent:
  * A wall-clock budget, plus a cap on the number of tries.
  * Every family starts with a few tries, the first being its default settings
    (a warm start). Then the weaker half of the families is dropped and the
    budget goes to the rest (successive halving across families).
  * The search stops early once many tries in a row have failed to beat the best
    by more than the noise between folds.

What comes out is one tuned candidate per surviving family. Those go through
the normal training step, where they are fitted on all training rows and scored
once on the locked test rows, like every other candidate.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.trial import TrialState

from mlteam.contract import DataContract
from mlteam.features import Feature
from mlteam.store import Store
from mlteam.train import (
    FAMILIES,
    HIGHER_IS_BETTER,
    SEED,
    Candidate,
    TrainError,
    _load,
    cv_folds,
    cv_scores,
    gain,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

TUNABLE = ("linear", "tree", "forest", "boosting")
INT_PARAMS = {"max_depth", "n_trees", "rounds"}
LOG_PARAMS = {"learning_rate", "regularization"}
FIRST_RUNG = 4     # tries per family before the first cut
SECOND_RUNG = 12   # tries per surviving family before the second cut


@dataclass
class Budget:
    seconds: float = 60.0
    max_trials: int = 80
    patience: int = 20  # tries in a row without a real improvement before stopping

    def describe(self) -> str:
        return f"{self.seconds:g} seconds or {self.max_trials} tries, whichever comes first"


@dataclass
class TuneTrial:
    number: int
    family: str
    params: dict[str, float]
    state: str                     # complete | pruned | failed
    fold_scores: list[float]
    cv_mean: float | None
    cv_std: float | None
    seconds: float
    elapsed: float                 # budget spent when this try finished
    best_so_far: float | None      # best complete score across all families at that point
    improved: bool = False


@dataclass
class TuneRun:
    metric: str
    higher_is_better: bool
    folds: int
    fold_kind: str
    budget: dict[str, Any]
    spent_seconds: float = 0.0
    stop_reason: str = ""
    trials: list[TuneTrial] = field(default_factory=list)
    best: dict[str, dict[str, Any]] = field(default_factory=dict)  # family -> params, cv_mean, cv_std, trial, tries
    dropped: dict[str, str] = field(default_factory=dict)          # family -> why it stopped getting budget
    warnings: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def counts(self) -> dict[str, int]:
        out = {"complete": 0, "pruned": 0, "failed": 0}
        for t in self.trials:
            out[t.state] = out.get(t.state, 0) + 1
        return out

    def candidates(self) -> list[Candidate]:
        """One tuned candidate per family that reached the end, best first."""
        ranked = sorted(self.best.items(), key=lambda kv: kv[1]["cv_mean"], reverse=self.higher_is_better)
        out = []
        for family, info in ranked:
            if family in self.dropped:
                continue
            out.append(Candidate(
                name=f"tuned_{family}", family=family, params=dict(info["params"]), origin="tuner",
                why=(f"Best of {info['tries']} tries for {family} by Bayesian optimisation: "
                     f"{self.metric} {info['cv_mean']:.3f} ± {info['cv_std']:.3f} across folds of the training rows."),
            ))
        return out


def _suggest(trial: optuna.Trial, family: str) -> dict[str, float]:
    params: dict[str, float] = {}
    for name, (low, high, _default) in FAMILIES[family].items():
        if name in INT_PARAMS:
            params[name] = float(trial.suggest_int(name, int(low), int(high), log=name == "n_trees"))
        else:
            params[name] = float(trial.suggest_float(name, low, high, log=name in LOG_PARAMS))
    return params


def _defaults(family: str) -> dict[str, float]:
    return {k: (int(v[2]) if k in INT_PARAMS else v[2]) for k, v in FAMILIES[family].items()}


def tune(
    store: Store,
    contract: DataContract,
    features: list[Feature],
    budget: Budget,
    families: list[str] | None = None,
    on_trial: Callable[[TuneTrial], None] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> TuneRun:
    data = _load(store, contract, features)
    task = contract.task or ""
    classify = task == "classification"
    metric = contract.metric or ("roc_auc" if classify else "rmse")
    n_classes = len(data.classes)
    if metric == "f1" and n_classes > 2:
        metric = "macro_f1"
    higher = HIGHER_IS_BETTER[metric]
    folds = cv_folds(data.y_train, classify=classify, time_ordered=data.time_ordered, groups=data.groups_train)
    families = [f for f in (families or list(TUNABLE)) if f in TUNABLE] or list(TUNABLE)
    run = TuneRun(
        metric=metric, higher_is_better=higher, folds=len(folds),
        fold_kind="expanding windows in time order" if data.time_ordered else "shuffled folds",
        budget=asdict(budget),
    )
    if len(data.y_train) < 60:
        run.warnings.append(
            f"Only {len(data.y_train)} training rows. Differences between settings will mostly be noise; "
            "a large search is likely to fit that noise."
        )

    direction = "maximize" if higher else "minimize"
    studies = {
        family: optuna.create_study(
            direction=direction,
            sampler=TPESampler(seed=SEED, n_startup_trials=FIRST_RUNG),
            pruner=MedianPruner(n_startup_trials=FIRST_RUNG, n_warmup_steps=1),
        )
        for family in families
    }
    for family, study in studies.items():
        study.enqueue_trial(_defaults(family))  # warm start: begin from the known-reasonable settings

    tries: dict[str, int] = {f: 0 for f in families}
    alive = list(families)
    started = clock()
    best_value: float | None = None
    anchor: float | None = None   # the best score at the last real improvement
    anchor_std = 0.0
    since_improvement = 0
    turn = 0
    rungs_done: set[int] = set()

    def family_best(family: str) -> float | None:
        info = run.best.get(family)
        return None if info is None else info["cv_mean"]

    def cut(rung: int) -> None:
        nonlocal alive
        if rung in rungs_done or len(alive) <= 1 or any(tries[f] < rung for f in alive):
            return
        rungs_done.add(rung)
        ranked = sorted(alive, key=lambda f: (family_best(f) is None, -(family_best(f) or 0.0) if higher
                                              else (family_best(f) or 0.0)))
        keep = max(1, math.ceil(len(ranked) / 2))
        for family in ranked[keep:]:
            score = family_best(family)
            run.dropped[family] = (
                f"Stopped after {tries[family]} tries: its best ({'none' if score is None else f'{score:.3f}'}) "
                "was in the weaker half."
            )
        alive = ranked[:keep]

    while True:
        elapsed = clock() - started
        if elapsed >= budget.seconds:
            run.stop_reason = f"Time budget spent ({budget.seconds:g} s)."
            break
        if len(run.trials) >= budget.max_trials:
            run.stop_reason = f"Reached the cap of {budget.max_trials} tries."
            break
        if since_improvement >= budget.patience and len(run.trials) >= FIRST_RUNG * len(families):
            run.stop_reason = (f"{budget.patience} tries in a row without beating the best by more than "
                               "the noise between folds.")
            break
        cut(FIRST_RUNG)
        cut(SECOND_RUNG)
        family = alive[turn % len(alive)]
        turn += 1
        study = studies[family]
        trial = study.ask()
        params = _suggest(trial, family)
        candidate = Candidate(name=f"try_{len(run.trials)}", family=family, params=dict(params))
        trial_start = clock()
        pruned = out_of_time = False

        def on_fold(i: int, scores: list[float], _trial: optuna.Trial = trial) -> bool:
            nonlocal pruned, out_of_time
            # On big data one try can take longer than the whole budget: stop it between
            # folds once time is up, as long as something has already been scored.
            if best_value is not None and clock() - started >= budget.seconds:
                out_of_time = True
                return True
            _trial.report(float(np.mean(scores)), step=i)
            if _trial.should_prune():
                pruned = True
                return True
            return False

        try:
            scores = cv_scores(candidate, task, metric, data.x_train, data.y_train,
                               data.numeric, data.categorical, folds, n_classes, on_fold=on_fold)
        except Exception:  # noqa: BLE001 - a setting that crashes is just a failed try
            scores = []
        seconds = clock() - trial_start
        tries[family] += 1
        mean = float(np.mean(scores)) if scores else None
        std = float(np.std(scores, ddof=1)) if len(scores) >= 2 else None
        if out_of_time:
            study.tell(trial, state=TrialState.FAIL)
            state = "out of time"
        elif pruned:
            study.tell(trial, state=TrialState.PRUNED)
            state = "pruned"
        elif mean is None or len(scores) < 2:
            study.tell(trial, state=TrialState.FAIL)
            state = "failed"
        else:
            study.tell(trial, mean)
            state = "complete"

        improved = False
        if state == "complete" and mean is not None:
            if best_value is None or gain(mean, best_value, higher) > 0:
                best_value = mean
            # A real improvement beats the anchor by more than the standard error between folds.
            noise = anchor_std / math.sqrt(len(folds)) if anchor is not None else 0.0
            if anchor is None or gain(best_value, anchor, higher) > noise:
                improved = True
                anchor, anchor_std = best_value, std or 0.0
                since_improvement = 0
            else:
                since_improvement += 1
            current = run.best.get(family)
            if current is None or gain(mean, current["cv_mean"], higher) > 0:
                run.best[family] = {"params": params, "cv_mean": mean, "cv_std": std or 0.0,
                                    "trial": len(run.trials), "tries": 0}
        else:
            since_improvement += 1
        record = TuneTrial(
            number=len(run.trials), family=family, params=params, state=state, fold_scores=scores,
            cv_mean=mean, cv_std=std, seconds=round(seconds, 3), elapsed=round(clock() - started, 3),
            best_so_far=best_value, improved=improved,
        )
        run.trials.append(record)
        if on_trial is not None:
            on_trial(record)

    for family, info in run.best.items():
        info["tries"] = tries[family]
    run.spent_seconds = round(clock() - started, 2)
    if not run.best:
        raise TrainError("Tuning finished without a single complete try.")
    return run


def best_curve(run: TuneRun, width: int = 40) -> str:
    """Best score so far, against tries, as one line of blocks."""
    values = [t.best_so_far for t in run.trials if t.best_so_far is not None]
    if not values:
        return ""
    if len(values) > width:
        step = len(values) / width
        values = [values[min(len(values) - 1, int(i * step + step - 1))] for i in range(width)]
    low, high = min(values), max(values)
    blocks = "▁▂▃▄▅▆▇█"
    if high - low < 1e-12:
        return blocks[-1] * len(values)
    scaled = [(v - low) / (high - low) for v in values]
    if not run.higher_is_better:
        scaled = [1 - s for s in scaled]
    return "".join(blocks[min(7, int(s * 7.999))] for s in scaled)
