"""Small behaviours added after real-model runs showed agents tripping on them."""

from __future__ import annotations

from pathlib import Path

from tau_ai.fake import FakeProvider

from mlteam.agents.brain import Brain
from mlteam.train import ModelResult, TrainingRun
from mlteam.tui import render
from tests.test_team import text_turn, tool_turn
from tests.test_train import _team_with_features


async def test_run_sql_on_the_features_table_points_to_the_right_tool(tmp_path: Path) -> None:
    """Seen in a real run: the model Skeptic queried `features` with run_sql, where it does not exist."""
    team = await _team_with_features(tmp_path)
    streams = [
        tool_turn("p1", "submit_training_plan", {"candidates": [
            {"name": "simple", "family": "linear", "why": "Few rows."}]}),
        tool_turn("s1", "run_sql", {"sql": "SELECT count(*) FROM features"}),
        tool_turn("s2", "run_sql", {"sql": "SELECT count(*) FROM no_such_table"}),
        tool_turn("s3", "submit_review", {"verdict": "approve", "summary": "Fine."}),
    ]
    team.brain = Brain(provider=FakeProvider(streams), model="scripted", label="fake/scripted")
    events: list = []
    team._emit_to_frontend = events.append
    try:
        await team.train_models()
        failed = [e.text for e in events if e.kind == "tool_end" and e.data["tool"] == "run_sql"]
        assert len(failed) == 2
        assert "query_features" in failed[0]
        assert "query_features" not in failed[1]  # the hint is only for the features table
    finally:
        team.close()


def test_importance_shows_what_matters_and_folds_the_rest() -> None:
    run = TrainingRun(task="classification", metric="roc_auc", higher_is_better=True, split="", train_rows=240,
                      test_rows=60, dropped_rows=0, positive_class="1", features=[])
    result = ModelResult("m", "tree", {}, "", "rules", test_score=0.8, train_score=0.82, beats_baseline=True,
                         importance=[("Sex", 0.345), ("Age", 0.004), ("Pclass", 0.0), ("Fare", -0.02)])
    text = str(render.model_detail(result, run))
    assert "Sex" in text and "Age" in text
    assert "2 other feature(s): no measurable effect" in text
    assert "Pclass" not in text and "Fare " not in text

    result.importance = [("Sex", 0.0), ("Age", -0.01)]
    assert "it has learned little" in str(render.model_detail(result, run))


async def test_skeptic_that_stops_without_submitting_is_reminded_once(tmp_path: Path) -> None:
    """Seen in a real run: the Skeptic queried, then ended its turn with no review at all."""
    team = await _team_with_features(tmp_path)
    streams = [
        tool_turn("p1", "submit_training_plan", {"candidates": [
            {"name": "simple", "family": "linear", "why": "Few rows."}]}),
        tool_turn("s1", "run_sql", {"sql": "SELECT 1"}),
        text_turn(""),  # stops: nothing submitted
        tool_turn("s2", "submit_review", {"verdict": "approve", "summary": "Fine after the reminder."}),
    ]
    team.brain = Brain(provider=FakeProvider(streams), model="scripted", label="fake/scripted")
    events: list = []
    team._emit_to_frontend = events.append
    try:
        await team.train_models()
        assert team.status["model_skeptic"] == "done"
        assert team.model_review is not None and team.model_review.summary == "Fine after the reminder."
        assert any(e.agent == "harness" and "reminding" in e.text for e in events)
    finally:
        team.close()


def test_missing_review_never_reads_as_ok() -> None:
    from types import SimpleNamespace

    from mlteam.cli import _skeptic_note

    team = SimpleNamespace(status={"model_skeptic": "failed"})
    note = _skeptic_note(team, "model_skeptic", "openai-codex/some-model")
    assert "did not finish" in str(render.review(None, note))
