"""Budgeted Bayesian tuning, feature tests, and the honesty rules around them."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mlteam import samples
from mlteam.feature_tests import _benjamini_hochberg, _corrected_p, _holm, run_feature_tests
from mlteam.features import Feature, default_features, evaluate_features
from mlteam.runstore import new_run
from mlteam.team import Team
from mlteam.train import cv_folds, train_all, default_plan
from mlteam.tune import Budget, best_curve, tune


def _features(loaded) -> list[Feature]:
    return [f for f in evaluate_features(loaded.store, loaded.draft, default_features(loaded.draft, loaded.profile))
            if f.default_on]


class FakeClock:
    """Each reading advances one second, so time budgets are exact and fast in tests."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        self.now += 1.0
        return self.now


# -- folds ---------------------------------------------------------------


def test_folds_never_validate_on_the_past_when_rows_are_in_time_order() -> None:
    y = np.array([0, 1] * 50)
    for train_idx, val_idx in cv_folds(y, classify=True, time_ordered=True):
        assert train_idx.max() < val_idx.min()
    shuffled = cv_folds(y, classify=True, time_ordered=False, repeats=2)
    assert len(shuffled) == 10
    for train_idx, val_idx in shuffled:
        assert not set(train_idx) & set(val_idx)
        assert set(np.bincount(y[val_idx])) == {10}  # stratified: same mix of outcomes in every fold


# -- the tuner -----------------------------------------------------------


def test_tuning_respects_the_budget_and_halves_the_families(load, scenario) -> None:
    loaded = load(scenario("titanic"))
    run = tune(loaded.store, loaded.draft, _features(loaded), Budget(seconds=1e9, max_trials=30, patience=999))

    assert len(run.trials) == 30 and run.stop_reason == "Reached the cap of 30 tries."
    tries = {f: sum(1 for t in run.trials if t.family == f) for f in ("linear", "tree", "forest", "boosting")}
    assert sorted(tries.values())[:2] == [4, 4]  # the weaker half stopped after the first rung
    assert len(run.dropped) >= 2 and all("weaker half" in why for why in run.dropped.values())
    first = {t.family: t for t in reversed(run.trials)}
    assert first["tree"].params == {"max_depth": 3.0}  # warm start: each family's first try is its default
    for t in run.trials:
        if t.state == "complete":
            assert t.cv_mean is not None and len(t.fold_scores) == run.folds
    best_values = [t.best_so_far for t in run.trials if t.best_so_far is not None]
    assert best_values == sorted(best_values)  # best so far never gets worse
    assert len(best_curve(run)) == 30

    candidates = run.candidates()
    assert candidates and all(c.origin == "tuner" and c.name == f"tuned_{c.family}" for c in candidates)
    assert {c.family for c in candidates}.isdisjoint(run.dropped)


def test_time_budget_and_patience_stop_the_search(load, scenario) -> None:
    loaded = load(scenario("loans"))
    feats = _features(loaded)
    timed = tune(loaded.store, loaded.draft, feats, Budget(seconds=12, max_trials=500), clock=FakeClock())
    assert timed.stop_reason.startswith("Time budget spent") and len(timed.trials) < 12

    patient = tune(loaded.store, loaded.draft, feats, Budget(seconds=1e9, max_trials=500, patience=3),
                   families=["tree"])
    assert "in a row without beating the best" in patient.stop_reason
    assert len(patient.trials) < 500 and list(patient.best) == ["tree"]


def test_tuned_candidates_are_judged_on_the_locked_test_rows_like_any_other(load, scenario, tmp_path: Path) -> None:
    loaded = load(scenario("titanic"))
    feats = _features(loaded)
    tuned = tune(loaded.store, loaded.draft, feats, Budget(seconds=1e9, max_trials=16, patience=999))
    run = train_all(loaded.store, loaded.draft, feats, [*default_plan(), *tuned.candidates()], tmp_path / "m")

    for candidate in tuned.candidates():
        result = run.result(candidate.name)
        assert result is not None and result.test_score is not None and result.origin == "tuner"
        assert result.cv_mean == pytest.approx(tuned.best[candidate.family]["cv_mean"])  # same folds, same score
    best = next(r for r in run.results if r.recommended)
    eligible = [r for r in run.results if r.beats_baseline]
    # Picked by cross-validation, not the test score. An ensemble only wins when clearly ahead of the best single.
    singles = [r for r in eligible if r.family != "ensemble"]
    assert best.recommended and (best.family == "ensemble" or best.cv_mean == max(r.cv_mean for r in singles))
    # 60 test rows cannot separate models this close; the harness says so instead of crowning a winner.
    assert best.ties_with and any("Not clearly better" in n for n in best.notes)


# -- feature tests -------------------------------------------------------


def test_corrections_make_luck_harder_to_mistake_for_signal() -> None:
    assert _holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert _benjamini_hochberg([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.04, 0.04])
    gains = [0.02, 0.01, 0.03, -0.01, 0.02, 0.02, 0.0, 0.01, 0.03, 0.01]
    from scipy import stats

    naive = stats.ttest_1samp(gains, 0.0).pvalue
    assert _corrected_p(gains, 0.25) > naive  # overlapping folds: less certainty than a plain t-test claims


def test_feature_tests_find_the_real_signal_and_not_the_noise(load, scenario) -> None:
    loaded = load(scenario("loans"))
    on = _features(loaded) + [Feature("noise", 'hash(b."Loan_ID") % 7', origin="engineer")]
    leak = Feature("decision", 'b."Decision_Code"', origin="engineer")
    evaluate_features(loaded.store, loaded.draft, [*on, leak])
    run = run_feature_tests(loaded.store, loaded.draft, on, [leak], budget_seconds=1e9)

    verdicts = {t.name: t for t in run.tests}
    assert verdicts["Credit_History"].verdict == "helps" and verdicts["Credit_History"].kind == "drop"
    assert verdicts["noise"].verdict == "no clear effect"
    assert verdicts["decision"].kind == "add" and verdicts["decision"].verdict == "helps"  # the leak is obvious
    assert all(len(t.gains) == run.folds == 10 for t in run.tests)
    assert all(t.p_adjusted is not None and t.p_adjusted >= t.p_value for t in run.tests)
    assert "Credit_History" in run.summary() or "help" in run.summary()


def test_feature_tests_stop_at_the_budget_and_say_what_they_skipped(load, scenario) -> None:
    loaded = load(scenario("titanic"))
    on = _features(loaded)
    clock = FakeClock()
    run = run_feature_tests(loaded.store, loaded.draft, on, [], budget_seconds=6, clock=clock)
    assert run.untested and len(run.tests) + len(run.untested) == len(on)
    assert "Not reached within the budget" in run.summary()


# -- in the pipeline -----------------------------------------------------


async def test_pipeline_runs_tests_and_tuning_and_saves_both(tmp_path: Path) -> None:
    team = Team(new_run(tmp_path / "runs"), tune_budget=Budget(seconds=1e9, max_trials=10, patience=999),
                test_seconds=1e9)
    try:
        draft = await team.prepare([str(samples.make("titanic", tmp_path / "data"))])
        await team.consult()
        team.approve(draft, "tester")
        features = await team.build_features()
        assert team.status["feature_tests"] == "done" and team.feature_tests is not None
        tested = {m for t in team.feature_tests.tests for m in t.members}  # derived versions travel with their source
        assert tested == {f.name for f in features if f.default_on}
        assert team.run.feature_tests.exists()

        team.approve_features([f.name for f in features if f.default_on], "tester")
        run = await team.train_models()
        assert team.status["tune"] == "done" and team.tuning is not None and len(team.tuning.trials) == 10
        assert any(r.origin == "tuner" for r in run.results)
        saved = json.loads(team.run.tuning.read_text(encoding="utf-8"))
        assert len(saved["trials"]) == 10 and saved["budget"]["max_trials"] == 10
        trace = [json.loads(line) for line in team.run.trace.read_text(encoding="utf-8").splitlines()]
        assert sum(1 for r in trace if r["kind"] == "tune_trial") == 10

        # A zero budget skips both, and nothing else changes.
        team.tune_budget = Budget(seconds=0)
        await team.train_models()
        assert team.status["tune"] == "skipped" and not any(r.origin == "tuner" for r in team.training.results)
    finally:
        team.close()
