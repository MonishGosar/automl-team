from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from textual.widgets import DataTable

from mlteam import samples
from mlteam.runstore import new_run
from mlteam.team import Team
from mlteam.tui.app import MLTeamApp
from tests.test_tui import SIZE, settle, shot, text_of


async def _team_with_model(tmp_path: Path, source: str | Path, model: str = "linear") -> Team:
    team = Team(new_run(tmp_path / "runs"))
    draft = await team.prepare([str(source)])
    await team.consult()
    team.approve(draft, "tester")
    features = await team.build_features()
    team.approve_features([f.name for f in features if f.default_on], "tester")
    await team.train_models()
    team.approve_model(model, "tester")
    return team


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


async def test_held_back_test_file_is_scored_and_written(tmp_path: Path) -> None:
    team = await _team_with_model(tmp_path, samples.make("titanic", tmp_path / "data"))
    try:
        result = team.predict()
        assert result is not None and result.table == "test" and result.rows == 120
        assert result.model == "linear" and result.model_version == 1 and result.positive_class == "1"
        assert sum(result.counts.values()) == 120 and set(result.counts) <= {"0", "1"}
        assert result.notes == []  # the sample's test rows are drawn like its training rows
        assert team.status["predict"] == "done"

        rows = _read(team.run.root / result.output)
        assert len(rows) == 120 and list(rows[0]) == ["PassengerId", "prediction", "probability_of_1"]
        assert [int(r["PassengerId"]) for r in rows] == list(range(301, 421))  # every test row, once, in order
        for row in rows:  # the label agrees with the probability at the default 0.5 cut-off
            assert (float(row["probability_of_1"]) >= 0.5) == (row["prediction"] == "1")

        submission = _read(team.run.root / str(result.submission))
        assert list(submission[0]) == ["PassengerId", "Survived"] and len(submission) == 120
        assert [(r["PassengerId"], r["Survived"]) for r in submission] == [
            (r["PassengerId"], r["prediction"]) for r in rows]

        trace = [json.loads(line) for line in team.run.trace.read_text(encoding="utf-8").splitlines()]
        assert any(r["kind"] == "prediction" and r["rows"] == 120 for r in trace)

        # Approving a different model and scoring again keeps the earlier output.
        team.approve_model("small_tree", "tester")
        assert team.prediction is None and team.status["predict"] == "pending"
        second = team.predict()
        assert second is not None and second.output == "predictions.v2.csv" and second.model == "small_tree"
        assert (team.run.root / "predictions.v1.csv").exists()
    finally:
        team.close()


async def test_no_test_file_means_the_step_is_skipped(tmp_path: Path) -> None:
    team = await _team_with_model(tmp_path, samples.make("loans", tmp_path / "data"))
    try:
        assert team.predict() is None
        assert team.status["predict"] == "skipped"
        assert not list(team.run.root.glob("predictions.*"))
    finally:
        team.close()


async def test_predicting_needs_an_approved_model(tmp_path: Path) -> None:
    team = Team(new_run(tmp_path / "runs"))
    try:
        await team.prepare([str(samples.make("titanic", tmp_path / "data"))])
        await team.consult()
        with pytest.raises(ValueError, match="Approve a model first"):
            team.predict()
    finally:
        team.close()


async def test_test_rows_unlike_training_rows_are_reported(tmp_path: Path) -> None:
    folder = tmp_path / "data" / "titanic"
    samples.make_titanic(folder)
    test_path = folder / "test.csv"
    with test_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    header, body = rows[0], rows[1:]
    sex, age = header.index("Sex"), header.index("Age")
    for i, row in enumerate(body):
        if i < 30:
            row[sex] = "unknown"   # a category the model never saw
        if i % 10 != 0:
            row[age] = ""          # far more gaps than in training
    with test_path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows([header, *body])

    team = await _team_with_model(tmp_path, folder)
    try:
        result = team.predict()
        assert result is not None and result.rows == 120
        notes = " | ".join(result.notes)
        assert "Sex has 1 value(s) never seen in training (unknown), on 30 rows" in notes
        blank = sum(1 for row in body if row[age] == "")  # the sample already had some gaps
        assert f"Age is missing on {100 * blank / len(body):.0f}% of these rows, against 21% in training" in notes
        assert blank > 100
        assert (team.run.root / result.output).exists()  # still written: the person decides what to do with it
    finally:
        team.close()


async def test_app_scores_the_test_file_after_model_approval(tmp_path: Path) -> None:
    app = MLTeamApp(["demo:titanic"], runs_dir=tmp_path / "runs")
    async with app.run_test(size=SIZE) as pilot:
        await settle(app, pilot)
        for _ in range(2):  # contract, then features
            await pilot.press("ctrl+s")
            await settle(app, pilot)
        assert text_of(app, "#prediction") == ""
        await pilot.press("ctrl+s")  # the recommended model
        await settle(app, pilot)

        assert app.team is not None and app.team.status["predict"] == "done"
        shown = text_of(app, "#prediction")
        assert "120 rows of test scored" in shown and "predictions.v1.csv" in shown and "submission.v1.csv" in shown
        assert "has no score" in shown
        assert (app.team.run.root / "submission.v1.csv").exists()
        assert app.query_one("#model_table", DataTable).row_count >= 5  # plan plus tuned candidates
        shot(app, "models_predicted")
