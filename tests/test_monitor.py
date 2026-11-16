"""After go-live: drift per batch, champion against challenger, promotion."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import pytest
from textual.widgets import Button

from mlteam.challenger import challenge, load_challenge, promote
from mlteam.cli import main
from mlteam.monitor import Health, assess, build_reference, psi, psi_and_noise, read_history
from mlteam.predict import PredictError, predict_file
from mlteam.tui import render
from mlteam.tui.app import MLTeamApp, ScoreFileScreen
from tests.test_predict import _team_with_model
from tests.test_predict_file import COLUMNS, _passengers
from tests.test_tui import SIZE, settle, text_of
from mlteam import samples

import numpy as np


def changed_world(path: Path, rows: int, seed: int) -> Path:
    """Same kinds of passengers, but the outcome rule has flipped: men in 3rd class now survive."""
    rng = random.Random(seed)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([*COLUMNS, "Survived"])
        for i in range(rows):
            sex = rng.choice(["male", "female"])
            pclass = rng.choices([1, 2, 3], weights=[0.24, 0.21, 0.55])[0]
            chance = 0.15 + (0.6 if sex == "male" else 0.0) + (0.1 if pclass == 3 else 0.0)
            writer.writerow([20000 + i, pclass, sex, round(max(1.0, rng.gauss(30, 13)), 1),
                             round(rng.lognormvariate(3.0, 0.9) / pclass, 2), 1 if rng.random() < chance else 0])
    return path


def moved_inputs(path: Path, rows: int, seed: int) -> Path:
    """Old, rich, first-class passengers only: inputs far from the training rows."""
    rng = random.Random(seed)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for i in range(rows):
            writer.writerow([30000 + i, 1, "female" if rng.random() < 0.85 else "male",
                             round(rng.gauss(60, 8), 1), round(rng.uniform(80, 300), 2)])
    return path


# -- the measure ---------------------------------------------------------


class _F:
    def __init__(self, name: str, kind: str) -> None:
        self.name, self.kind = name, kind


def test_psi_is_near_zero_for_the_same_data_and_large_for_moved_data() -> None:
    rng = np.random.default_rng(0)
    train = rng.normal(0, 1, 5000)
    reference = build_reference([_F("x", "numeric")], train.reshape(-1, 1).astype(object), list(train), "numeric")
    same = psi(reference["features"]["x"], list(rng.normal(0, 1, 5000)))
    moved = psi(reference["features"]["x"], list(rng.normal(1.5, 1, 5000)))
    assert same < 0.02 and moved > 0.25

    # Small batches differ by chance; the expected chance level grows as the batch shrinks.
    _, noise_small = psi_and_noise(reference["features"]["x"], list(rng.normal(0, 1, 80)), 5000)
    _, noise_big = psi_and_noise(reference["features"]["x"], list(rng.normal(0, 1, 5000)), 5000)
    assert noise_small > 0.1 > noise_big

    cats = build_reference([_F("c", "categorical")], np.array([["a"], ["b"]] * 500, dtype=object), ["a", "b"] * 500,
                           "category")
    assert psi(cats["features"]["c"], ["a"] * 990 + ["new"] * 10) > 0.25  # one category vanished


def test_drift_weighs_features_by_how_much_the_model_relies_on_them() -> None:
    rng = np.random.default_rng(1)
    base = rng.normal(0, 1, (2000, 2))
    reference = build_reference([_F("used", "numeric"), _F("ignored", "numeric")], base.astype(object),
                                list(rng.uniform(0, 1, 2000)), "numeric")
    batch = np.column_stack([rng.normal(0, 1, 2000), rng.normal(3, 1, 2000)]).astype(object)
    preds = list(rng.uniform(0, 1, 2000))
    features = [_F("used", "numeric"), _F("ignored", "numeric")]
    calm = assess(reference, features, batch, preds, importance=[("used", 0.3), ("ignored", 0.0)])
    assert calm.status == "watch" and "ignored" in calm.reasons[0]
    alarmed = assess(reference, features, batch, preds, importance=[("used", 0.1), ("ignored", 0.3)])
    assert alarmed.status == "retrain" and "ignored" in alarmed.reasons[0]


# -- on real runs ---------------------------------------------------------


async def test_ordinary_batches_stay_stable_and_moved_ones_raise_the_alarm(tmp_path: Path) -> None:
    team = await _team_with_model(tmp_path, samples.make("titanic", tmp_path / "data"))
    run = team.run
    try:
        card = team.model_card
        assert card is not None and (run.root / card["reference"]).exists()
        assert card["cv_mean"] is not None and card["importance"]
        test = team.predict()
        assert test is not None and test.health["status"] == "stable"  # same data, small batch: no false alarm
    finally:
        team.close()

    ordinary = predict_file(run, _passengers(tmp_path / "ordinary.csv", 200, answers=True, seed=3))
    assert ordinary.health["status"] == "stable", ordinary.health["reasons"]
    assert ordinary.score is not None and ordinary.score["expected"] == pytest.approx(card["cv_mean"])

    moved = predict_file(run, moved_inputs(tmp_path / "moved.csv", 200, seed=5))
    assert moved.health["status"] == "retrain"
    assert any("relies on have shifted" in r for r in moved.health["reasons"])
    assert moved.health["prediction_psi"] > 0.25
    assert "Score a batch that has the answers" in moved.health["reasons"][-1]

    flipped = predict_file(run, changed_world(tmp_path / "flipped.csv", 300, seed=8))
    assert flipped.health["status"] == "retrain"
    assert any("got worse, by more than chance" in r for r in flipped.health["reasons"])

    history = read_history(run.monitoring)
    assert [h["status"] for h in history] == ["stable", "stable", "retrain", "retrain"]
    assert [h["source"] for h in history][1:] == ["ordinary.csv", "moved.csv", "flipped.csv"]
    report = str(render.monitor_report(history, card))
    assert "Retrain recommended" in report and "Trend for features that moved" in report


async def test_challenger_wins_only_when_the_world_changed_and_promotion_is_guarded(tmp_path: Path) -> None:
    team = await _team_with_model(tmp_path, samples.make("titanic", tmp_path / "data"), model="forest")
    run = team.run
    team.close()

    same = challenge(run, _passengers(tmp_path / "same.csv", 300, answers=True, seed=4))
    assert same.verdict == "no clear difference" and same.gain_low < 0 < same.gain_high
    assert same.holdout_rows == 90 and same.new_train_rows == 210 and same.original_train_rows == 240

    changed = challenge(run, changed_world(tmp_path / "changed.csv", 400, seed=8))
    assert changed.verdict == "better" and changed.gain_low > 0.2
    assert changed.challenger_new > changed.champion_new
    assert changed.challenger_old < changed.champion_old  # it gave up the old pattern
    assert any("forgotten the old pattern" in n for n in changed.notes)
    assert (run.root / changed.artifact).exists()

    # The live model only changes on an explicit promotion.
    assert json.loads(run.model_latest.read_text(encoding="utf-8"))["version"] == 1
    path = promote(run, changed.version, "tester")
    live = json.loads(path.read_text(encoding="utf-8"))
    assert path.name == "model.v2.json" and live["replaces"] == 1 and live["artifact"] == changed.artifact
    assert live["trained_on_extra"]["file"] == "changed.csv" and live["approved_by"] == "tester"
    assert (run.root / live["reference"]).exists() and live["reference"] != "reference.v1.json"
    assert load_challenge(run, changed.version).promoted_as == 2
    assert (run.root / "model.v1.json").exists()  # the old model stays on record

    with pytest.raises(PredictError, match="already promoted"):
        promote(run, changed.version)
    with pytest.raises(PredictError, match="live model changed"):
        promote(run, same.version)  # trained against v1, which is no longer live

    after = predict_file(run, changed_world(tmp_path / "later.csv", 300, seed=9))
    assert after.model_version == 2
    assert after.score["value"] > changed.champion_new + 0.2  # far better than the old model in the new world

    trace = [json.loads(line) for line in run.trace.read_text(encoding="utf-8").splitlines()]
    assert [r["kind"] for r in trace if r["kind"] in {"challenger", "promotion"}] == [
        "challenger", "challenger", "promotion"]


async def test_a_worse_challenger_cannot_be_promoted_and_bad_files_are_refused(tmp_path: Path) -> None:
    team = await _team_with_model(tmp_path, samples.make("titanic", tmp_path / "data"))
    run = team.run
    team.close()
    with pytest.raises(PredictError, match="needs the real answers"):
        challenge(run, _passengers(tmp_path / "no_answers.csv", 100, answers=False))
    with pytest.raises(PredictError, match="At least 40"):
        challenge(run, _passengers(tmp_path / "tiny.csv", 20, answers=True))

    worse = challenge(run, changed_world(tmp_path / "changed.csv", 300, seed=2))
    worse.verdict = "worse"  # force the case: a challenger that lost on the held-back rows
    (run.root / f"challenger.v{worse.version}.json").write_text(json.dumps(worse.to_json()), encoding="utf-8")
    with pytest.raises(PredictError, match="cannot be promoted"):
        promote(run, worse.version)
    assert json.loads(run.model_latest.read_text(encoding="utf-8"))["version"] == 1


async def test_cli_monitor_retrain_promote(tmp_path: Path, capsys) -> None:
    team = await _team_with_model(tmp_path, samples.make("titanic", tmp_path / "data"), model="forest")
    runs = str(team.run.root.parent)
    team.close()
    assert main(["monitor", "latest", "--runs-dir", runs]) == 0
    assert "Live model: forest v1" in capsys.readouterr().out
    labelled = changed_world(tmp_path / "changed.csv", 400, seed=8)
    assert main(["retrain", "latest", str(labelled), "--runs-dir", runs]) == 0
    out = capsys.readouterr().out
    assert "CHALLENGER v1" in out and "better" in out and "mlteam promote latest 1" in out
    assert main(["promote", "latest", "1", "--runs-dir", runs]) == 0
    assert main(["promote", "latest", "1", "--runs-dir", runs]) == 1  # already promoted
    capsys.readouterr()
    assert main(["monitor", "latest", "--runs-dir", runs]) == 0
    assert "replaced v1 after a challenger" in capsys.readouterr().out


async def test_app_monitor_tab_scores_retrains_and_promotes(tmp_path: Path) -> None:
    labelled = changed_world(tmp_path / "changed.csv", 400, seed=8)
    moved = moved_inputs(tmp_path / "moved.csv", 150, seed=5)
    app = MLTeamApp(["demo:titanic"], runs_dir=tmp_path / "runs")
    async with app.run_test(size=SIZE) as pilot:
        await settle(app, pilot)
        for _ in range(3):  # contract, features, model
            await pilot.press("ctrl+s")
            await settle(app, pilot)
        assert "test" in text_of(app, "#monitor_report")  # the test-file batch is already on record

        app.score_file(str(moved))
        await settle(app, pilot)
        assert "Retrain recommended" in text_of(app, "#monitor_report")
        assert "DRIFT" in text_of(app, "#prediction")

        await pilot.press("f6")
        await pilot.pause()
        app.query_one("#monitor_retrain", Button).press()
        await pilot.pause()
        assert isinstance(app.screen, ScoreFileScreen)
        app.screen.dismiss(str(labelled))
        await settle(app, pilot)
        assert "CHALLENGER v1" in text_of(app, "#challenge_report")
        promote_button = app.query_one("#monitor_promote", Button)
        assert promote_button.disabled is False
        promote_button.press()
        await settle(app, pilot)
        assert app.team is not None and app.team.model_card["version"] == 2
        assert "Promoted: live as model v2" in text_of(app, "#challenge_report")
        assert "replaced v1 after a challenger" in text_of(app, "#monitor_report")
