from __future__ import annotations

import json
from pathlib import Path

import joblib
import pytest
from tau_ai.fake import FakeProvider
from textual.widgets import Button, DataTable, TabbedContent

from mlteam import samples
from mlteam.agents.brain import Brain
from mlteam.features import Feature, default_features, evaluate_features
from mlteam.runstore import new_run
from mlteam.team import Team
from mlteam.train import Candidate, TrainError, check_candidate, default_plan, train_all
from mlteam.tui.app import MLTeamApp
from tests.test_team import tool_turn
from tests.test_tui import SIZE, settle, shot, text_of


def _features(loaded, contract=None) -> list[Feature]:
    contract = contract or loaded.draft
    return [f for f in evaluate_features(loaded.store, contract, default_features(contract, loaded.profile))
            if f.default_on]


# -- the engine ----------------------------------------------------------


def test_every_candidate_is_scored_on_the_same_unseen_rows(load, scenario, tmp_path: Path) -> None:
    loaded = load(scenario("titanic"))
    run = train_all(loaded.store, loaded.draft, _features(loaded), default_plan(), tmp_path / "models")

    assert run.metric == "roc_auc" and run.train_rows == 240 and run.test_rows == 60
    assert run.positive_class == "1" and "same mix of outcomes" in run.split
    assert [r.name for r in run.results] == ["baseline", "linear", "small_tree", "forest", "boosting", "ensemble"]
    assert all(r.error is None and r.test_score is not None for r in run.results)
    assert run.baseline is not None and run.baseline.test_score == pytest.approx(0.5)
    assert not run.baseline.beats_baseline and not run.baseline.recommended
    winners = [r for r in run.results if r.recommended]
    assert len(winners) == 1 and winners[0].beats_baseline
    # Picked by cross-validation on training rows, never by the test score.
    eligible = [r for r in run.results if r.beats_baseline]
    singles = [r for r in eligible if r.family != "ensemble"]
    assert winners[0].family == "ensemble" or winners[0].cv_mean == max(r.cv_mean for r in singles)
    ensemble = run.result("ensemble")
    if ensemble is not None and not ensemble.recommended and ensemble.cv_mean > winners[0].cv_mean:
        assert any("not worth averaging" in n for n in ensemble.notes)  # ahead, but not by enough
    assert all(r.cv_mean is not None and r.cv_std is not None for r in run.results)
    assert any("Only 60 test rows" in w for w in run.warnings)

    tree = run.result("small_tree")
    assert tree is not None and tree.tree_text is not None
    assert tree.tree_text.splitlines()[0] == "Sex = female?" and "→" in tree.tree_text
    assert tree.importance[0][0] == "Sex" and tree.importance[0][1] > 0.1

    # The saved file is the scored model, and it loads without retraining.
    saved = joblib.load(tmp_path / "models" / "small_tree.joblib")
    assert saved.named_steps["model"].get_depth() <= 3


def test_time_column_means_train_on_earlier_test_on_later(load, scenario, tmp_path: Path) -> None:
    loaded = load(scenario("shop"))
    run = train_all(loaded.store, loaded.draft, _features(loaded), default_plan(), tmp_path / "models")
    assert run.split == "Trained on the earliest 200 rows by signup_date, tested on the latest 50."


def test_forest_memorising_is_noted(load, scenario, tmp_path: Path) -> None:
    loaded = load(scenario("loans"))
    run = train_all(loaded.store, loaded.draft, _features(loaded), default_plan(), tmp_path / "models")
    forest = run.result("forest")
    assert forest is not None and forest.train_score - forest.test_score > 0.10
    assert any("memorising" in n for n in forest.notes)


def test_a_leaky_feature_makes_scores_too_good_and_that_is_flagged(load, scenario, tmp_path: Path) -> None:
    loaded = load(scenario("loans"))
    leak = Feature("decision", 'b."Decision_Code"', origin="engineer")
    evaluate_features(loaded.store, loaded.draft, [leak])
    run = train_all(loaded.store, loaded.draft, [*_features(loaded), leak], default_plan(), tmp_path / "models")
    tree = run.result("small_tree")
    assert tree is not None and tree.test_score >= 0.995
    assert any("gives the answer away" in n for n in tree.notes)
    assert tree.importance[0][0] == "decision"


def test_no_signal_means_nothing_can_be_recommended(load, scenario, tmp_path: Path) -> None:
    loaded = load(scenario("shards"))  # the target is noise by construction
    run = train_all(loaded.store, loaded.draft, _features(loaded), default_plan(), tmp_path / "models")
    assert run.task == "regression" and run.metric == "rmse" and not run.higher_is_better
    assert not any(r.recommended for r in run.results)
    assert any("No candidate beat the baseline" in w for w in run.warnings)


@pytest.mark.parametrize(
    ("candidate", "reason"),
    [
        (Candidate("x", "neural_net"), "Unknown family"),
        (Candidate("x", "tree", {"max_depth": 40}), "between 1 and 6"),
        (Candidate("x", "tree", {"leaves": 3}), "no parameter leaves"),
        (Candidate("x", "forest", {"n_trees": "many"}), "must be a number"),
        (Candidate("bad name", "tree"), "name must start"),
        (Candidate("linear", "linear"), "already exists"),
    ],
)
def test_plan_menu_is_enforced(candidate: Candidate, reason: str) -> None:
    assert reason in (check_candidate(candidate, {"linear"}) or "")


def test_a_bad_candidate_does_not_stop_the_others(load, scenario, tmp_path: Path) -> None:
    loaded = load(scenario("titanic"))
    plan = [*default_plan()[:2], Candidate("deep", "tree", {"max_depth": 99})]
    run = train_all(loaded.store, loaded.draft, _features(loaded), plan, tmp_path / "models")
    assert run.result("deep").error and run.result("linear").error is None
    with pytest.raises(TrainError, match="No features"):
        train_all(loaded.store, loaded.draft, [], default_plan(), tmp_path / "models")


# -- the harness ---------------------------------------------------------


async def _team_with_features(tmp_path: Path, scenario: str = "titanic") -> Team:
    team = Team(new_run(tmp_path / "runs"))
    draft = await team.prepare([str(samples.make(scenario, tmp_path / "data"))])
    await team.consult()
    team.approve(draft, "tester")
    features = await team.build_features()
    team.approve_features([f.name for f in features if f.default_on], "tester")
    return team


async def test_training_needs_approved_features(tmp_path: Path) -> None:
    team = Team(new_run(tmp_path / "runs"))
    try:
        draft = await team.prepare([str(samples.make("titanic", tmp_path / "data"))])
        await team.consult()
        team.approve(draft, "tester")
        await team.build_features()
        with pytest.raises(ValueError, match="Approve the features first"):
            await team.train_models()
    finally:
        team.close()


async def test_rules_only_training_and_model_approval(tmp_path: Path) -> None:
    team = await _team_with_features(tmp_path)
    try:
        run = await team.train_models()
        assert team.status["train_plan"] == "skipped" and team.status["train"] == "done"
        assert team.status["model_gate"] == "waiting" and team.run.training.exists()
        best = next(r for r in run.results if r.recommended)

        with pytest.raises(ValueError, match="bar to beat"):
            team.approve_model("baseline")
        with pytest.raises(ValueError, match="No candidate called"):
            team.approve_model("nope")
        path = team.approve_model("small_tree", "tester")  # a person may prefer the readable one

        card = json.loads(path.read_text(encoding="utf-8"))
        assert path.name == "model.v1.json" and card["model"] == "small_tree"
        assert card["was_recommended"] is (best.name == "small_tree")
        assert card["baseline_test_score"] == pytest.approx(0.5) and card["test_score"] > 0.5
        assert card["contract_version"] == 1 and card["features_version"] == 1
        assert card["fitted_on"] == "training rows only"
        assert (team.run.root / card["artifact"]).exists()
        trace = [json.loads(line) for line in team.run.trace.read_text(encoding="utf-8").splitlines()]
        assert any(r["kind"] == "model_approval" and r["model"] == "small_tree" for r in trace)

        # Re-approving features throws the trained models away.
        team.approve_features(team.feature_set.selected, "tester")
        assert team.training is None and team.status["model_gate"] == "pending"
    finally:
        team.close()


async def test_a_model_that_cannot_beat_the_baseline_is_refused(tmp_path: Path) -> None:
    team = await _team_with_features(tmp_path, "shards")
    try:
        await team.train_models()
        with pytest.raises(ValueError, match="no better than doing nothing"):
            team.approve_model("forest")
        assert not team.run.model_latest.exists()
    finally:
        team.close()


async def test_engineer_plans_harness_trains_skeptic_reviews(tmp_path: Path) -> None:
    team = await _team_with_features(tmp_path)
    streams = [
        tool_turn("p1", "submit_training_plan", {"candidates": [
            {"name": "tiny_tree", "family": "tree", "params": {"max_depth": 2}, "why": "Readable."},
            {"name": "huge", "family": "forest", "params": {"n_trees": 9000}, "why": "More is better?"}]}),
        tool_turn("p2", "submit_training_plan", {"candidates": [
            {"name": "tiny_tree", "family": "tree", "params": {"max_depth": 2}, "why": "Readable."},
            {"name": "simple", "family": "linear", "why": "Few rows favour a simple model."}]}),
        tool_turn("s1", "submit_review", {
            "verdict": "concerns", "summary": "Pick the simple model.",
            "concerns": [{"severity": "medium", "field": "simple",
                          "message": "It leans almost entirely on Sex.", "evidence": "shuffling Sex costs 0.35"}]}),
    ]
    provider = FakeProvider(streams)
    team.brain = Brain(provider=provider, model="scripted", label="fake/scripted")
    events: list = []
    team._emit_to_frontend = events.append
    try:
        run = await team.train_models()
        names = [r.name for r in run.results]
        assert names[:3] == ["baseline", "tiny_tree", "simple"]
        # Tuning searched only the families the Engineer chose, and added one tuned candidate per family.
        assert sorted(names[3:]) == ["ensemble", "tuned_linear", "tuned_tree"]  # the ensemble is the harness's
        assert all(r.origin == ("harness" if r.family == "ensemble" else "tuner") for r in run.results[3:])
        assert run.result("tiny_tree").origin == "engineer" and run.result("tiny_tree").params == {"max_depth": 2.0}
        plans = [e for e in events if e.kind == "tool_end" and e.data["tool"] == "submit_training_plan"]
        assert [e.data["is_error"] for e in plans] == [True, False] and "between 50 and 500" in plans[0].text
        assert team.model_review is not None and team.model_review.concerns[0].field == "simple"
        # The Engineer has no tool that trains or scores; the Skeptic sees the harness's numbers.
        assert "submit_training_plan" in {t.name for t in provider.calls[0][3]}
        assert not {t.name for t in provider.calls[0][3]} & {"train", "score", "fit", "try_feature"}
        assert "Results from the harness" in provider.calls[-1][2][0].text
        assert "test roc_auc=" in provider.calls[-1][2][0].text
    finally:
        team.close()


# -- the app -------------------------------------------------------------


async def test_app_three_gates_to_an_approved_model(tmp_path: Path) -> None:
    app = MLTeamApp(["demo:titanic"], runs_dir=tmp_path / "runs")
    async with app.run_test(size=SIZE) as pilot:
        await settle(app, pilot)
        assert "Approve the features first" in text_of(app, "#model_summary")
        await pilot.press("ctrl+s")  # contract
        await settle(app, pilot)
        await pilot.press("ctrl+s")  # features
        await settle(app, pilot)

        assert app.team is not None and app.team.status["model_gate"] == "waiting"
        assert app.query_one("#main", TabbedContent).active == "models"
        assert "Scored by the harness" in text_of(app, "#model_summary")
        table = app.query_one("#model_table", DataTable)
        assert table.row_count >= 5  # the plan, plus tuned candidates
        best = next(r for r in app.team.training.results if r.recommended)
        assert best.name in text_of(app, "#model_detail")  # the cursor starts on the best model
        shot(app, "models_best")

        # The baseline cannot be approved, by button or by shortcut.
        table.move_cursor(row=0)
        await settle(app, pilot)
        assert app.query_one("#approve_model", Button).disabled is True
        await pilot.press("ctrl+s")
        await settle(app, pilot)
        assert not app.team.run.model_latest.exists()

        # The person picks the readable tree and sees it drawn.
        table.move_cursor(row=2)
        await settle(app, pilot)
        detail = text_of(app, "#model_detail")
        assert "The whole model, as questions" in detail and ("Sex = female?" in detail or "Sex = male?" in detail)
        assert "What it relies on" in detail
        shot(app, "models_tree")
        await pilot.press("ctrl+s")
        await settle(app, pilot)
        card = json.loads(app.team.run.model_latest.read_text(encoding="utf-8"))
        assert card["model"] == "small_tree" and card["version"] == 1
        assert "Approved small_tree" in str(app.query_one("#approve_model", Button).label)
        assert "your call" not in text_of(app, "#stages")
