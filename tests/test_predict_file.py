"""Scoring a brand-new file against a finished run."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import pytest

from mlteam import samples
from mlteam.cli import find_run, main
from mlteam.predict import PredictError, predict_file
from mlteam.runstore import new_run
from mlteam.team import Team
from mlteam.tui import render
from mlteam.tui.app import MLTeamApp, ScoreFileScreen
from tests.test_predict import _read, _team_with_model
from tests.test_tui import SIZE, settle, text_of

COLUMNS = ["PassengerId", "Pclass", "Sex", "Age", "Fare"]


def _passengers(path: Path, rows: int, *, answers: bool, seed: int = 99, start: int = 5000) -> Path:
    """Fresh passengers drawn the same way as the sample, but not the same rows."""
    rng = random.Random(seed)
    out = []
    for i in range(start, start + rows):
        sex = rng.choice(["male", "female"])
        pclass = rng.choices([1, 2, 3], weights=[0.24, 0.21, 0.55])[0]
        age = round(max(1.0, rng.gauss(30, 13)), 1)
        fare = round(rng.lognormvariate(3.0, 0.9) / pclass, 2)
        row = [i, pclass, sex, age if rng.random() > 0.2 else "", fare]
        if answers:
            chance = 0.2 + (0.5 if sex == "female" else 0.0) + (0.15 if pclass == 1 else 0.0)
            row.append(1 if rng.random() < chance else 0)
        out.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([*COLUMNS, "Survived"] if answers else COLUMNS)
        writer.writerows(out)
    return path


async def test_new_files_are_scored_with_what_was_approved(tmp_path: Path, capsys) -> None:
    team = await _team_with_model(tmp_path, samples.make("titanic", tmp_path / "data"))
    run = team.run
    try:
        # 1. A file with no answers: predictions, and no score.
        plain = _passengers(tmp_path / "new" / "arrivals.csv", 80, answers=False)
        result = team.predict_file(plain)
        assert result.source == "arrivals.csv" and result.rows == 80 and result.score is None
        assert result.model == "linear" and len(result.source_sha256 or "") == 64
        rows = _read(run.root / result.output)
        assert len(rows) == 80 and list(rows[0]) == ["PassengerId", "prediction", "probability_of_1"]
        assert rows[0]["PassengerId"] == "5000" and result.submission is None
        assert "has no score" in str(render.prediction(result))

        # 2. A file that carries the answers: the model is checked on it.
        labelled = _passengers(tmp_path / "new" / "outcomes.csv", 200, answers=True, seed=7)
        checked = team.predict_file(labelled)
        assert checked.score is not None and checked.score["metric"] == "roc_auc" and checked.score["rows"] == 200
        assert 0.7 < checked.score["value"] <= 1.0
        assert checked.score["original"] == pytest.approx(team.model_card["test_score"])
        assert "the model was checked on it" in str(render.prediction(checked))
        assert "Survived" not in _read(run.root / checked.output)[0]  # answers are never copied to the output
        assert checked.output != result.output

        # 3. The run's own data is untouched, and the app's connection still works.
        assert team.store is not None and team.store.scalar("SELECT count(*) FROM train") == 300
        with pytest.raises(Exception, match="scoring_rows"):
            team.store.query('SELECT count(*) FROM "__scoring_rows"')

        # 4. A file missing a column the features need is refused, and says which.
        with (tmp_path / "new" / "no_fare.csv").open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerows([["PassengerId", "Pclass", "Sex", "Age"], [1, 3, "male", 30]])
        with pytest.raises(ValueError, match="this file lacks: Fare"):
            team.predict_file(tmp_path / "new" / "no_fare.csv")

        # 5. Text where a number belongs is refused before any output is written.
        before = sorted(p.name for p in run.root.glob("predictions.*"))
        with (tmp_path / "new" / "bad_age.csv").open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerows([COLUMNS, [1, 3, "male", "thirty", 7.5], [2, 1, "female", "old", 80]])
        with pytest.raises(ValueError, match="Age should be a number"):
            team.predict_file(tmp_path / "new" / "bad_age.csv")
        with pytest.raises(ValueError, match="File not found"):
            team.predict_file(tmp_path / "new" / "nope.csv")
        assert sorted(p.name for p in run.root.glob("predictions.*")) == before

        # 6. Each scoring is on the record.
        trace = [json.loads(line) for line in run.trace.read_text(encoding="utf-8").splitlines()]
        assert [r["source"] for r in trace if r["kind"] == "prediction"] == ["arrivals.csv", "outcomes.csv"]
    finally:
        team.close()

    # 7. The same thing from the command line, later, with the app closed.
    assert find_run("latest", run.root.parent).root == run.root
    assert main(["predict", "latest", str(plain), "--runs-dir", str(run.root.parent)]) == 0
    printed = capsys.readouterr().out
    assert "80 rows of arrivals.csv scored with linear" in printed and "predictions.v3.csv" in printed
    assert main(["predict", run.root.name, str(tmp_path / "nope.csv"), "--runs-dir", str(run.root.parent)]) == 1
    assert main(["predict", "no_such_run", str(plain), "--runs-dir", str(run.root.parent)]) == 1


async def test_a_run_without_an_approved_model_cannot_score(tmp_path: Path) -> None:
    team = Team(new_run(tmp_path / "runs"))
    try:
        await team.prepare([str(samples.make("titanic", tmp_path / "data"))])
        await team.consult()
        plain = _passengers(tmp_path / "new.csv", 5, answers=False)
        with pytest.raises(ValueError, match="Approve a model first"):
            team.predict_file(plain)
        with pytest.raises(PredictError, match="no approved model"):
            predict_file(team.run, plain)
    finally:
        team.close()


async def test_new_rows_still_use_the_other_tables(tmp_path: Path) -> None:
    """Features that look into joined tables (orders, regions) must work for rows that are not in the run."""
    team = await _team_with_model(tmp_path, samples.make("shop", tmp_path / "data"), model="small_tree")
    try:
        new = tmp_path / "new_customers.csv"
        with new.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerows([
                ["customer_id", "region_id", "signup_date", "plan"],
                [1, 2, "2024-03-01", "pro"],        # a customer the run already has orders for
                [99999, 3, "2024-06-01", "basic"],  # a customer with no history at all
            ])
        result = team.predict_file(new)
        assert result.rows == 2 and sum(result.counts.values()) == 2
        rows = _read(team.run.root / result.output)
        assert [r["customer_id"] for r in rows] == ["1", "99999"]
        assert all(0.0 <= float(r["probability_of_1"]) <= 1.0 for r in rows)
    finally:
        team.close()


async def test_app_scores_a_new_file_on_request(tmp_path: Path) -> None:
    plain = _passengers(tmp_path / "arrivals.csv", 40, answers=False)
    app = MLTeamApp(["demo:titanic"], runs_dir=tmp_path / "runs")
    async with app.run_test(size=SIZE) as pilot:
        await settle(app, pilot)
        await pilot.press("ctrl+n")  # too early: no model yet
        await settle(app, pilot)
        assert not isinstance(app.screen, ScoreFileScreen)

        for _ in range(3):  # contract, features, model
            await pilot.press("ctrl+s")
            await settle(app, pilot)
        await pilot.press("ctrl+n")
        await pilot.pause()
        assert isinstance(app.screen, ScoreFileScreen)
        app.screen.dismiss(str(plain))
        await settle(app, pilot)

        shown = text_of(app, "#prediction")
        assert "40 rows of arrivals.csv scored" in shown and "predictions.v2.csv" in shown
        assert (app.team.run.root / "predictions.v2.csv").exists()

        app.score_file(str(tmp_path / "missing.csv"))
        await settle(app, pilot)
        assert "could not be scored" in text_of(app, "#prediction")
