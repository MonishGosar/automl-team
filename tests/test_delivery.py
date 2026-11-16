"""What a real team checks before handing a model over: cut-off, calibration, cost,
ranges, fairness by group, grouped splits, an ensemble, and the results report."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np

from mlteam import samples
from mlteam.cli import main as cli_main
from mlteam.contract import horizon_from_answers, validate_contract
from mlteam.evaluate import ece, evaluate
from mlteam.predict import predict_file
from mlteam.runstore import new_run
from mlteam.team import Team
from mlteam.tui import render


def _binary_case(n: int = 3000, seed: int = 0, overconfident: bool = False):
    rng = np.random.default_rng(seed)
    score = rng.normal(size=n)
    truth = 1 / (1 + np.exp(-(score - 1.5)))           # a rare-ish outcome
    y = (rng.random(n) < truth).astype(int)
    p = 1 / (1 + np.exp(-3 * (score - 1.5))) if overconfident else truth
    proba = np.column_stack([1 - p, p])
    return y, proba


def _evaluate_binary(y, proba, **kwargs):
    half = len(y) // 2
    return evaluate(task="classification", classes=["0", "1"], y_oof=y[:half], oof_pred=(proba[:half, 1] > .5).astype(int),
                    oof_proba=proba[:half], y_test=y[half:], test_pred=(proba[half:, 1] > .5).astype(int),
                    test_proba=proba[half:], x_test=np.empty((len(y) - half, 0), dtype=object), names=[], kinds=[],
                    **kwargs)


def test_cut_off_follows_the_costs() -> None:
    y, proba = _binary_case()
    plain = _evaluate_binary(y, proba)
    costly_miss = _evaluate_binary(y, proba, cost_false_alarm=1, cost_miss=20)
    assert plain.cutoff is not None and costly_miss.cutoff is not None
    assert costly_miss.cutoff < plain.cutoff  # a miss costs a lot: flag more rows
    assert costly_miss.at_cutoff["recall"] > plain.at_cutoff["recall"]
    cost = costly_miss.cost
    assert cost and cost["model"] < min(cost["act_on_nobody"], cost["act_on_everybody"])
    assert "lowest cost" in costly_miss.cutoff_reason and "best F1" in plain.cutoff_reason


def test_overconfident_probabilities_are_rescaled_and_good_ones_are_not() -> None:
    y, honest = _binary_case(seed=1)
    assert _evaluate_binary(y, honest).platt is None
    y, loud = _binary_case(seed=1, overconfident=True)
    ev = _evaluate_binary(y, loud)
    assert ev.platt is not None and ev.ece_after < ev.ece_before
    assert any("rescaled" in n for n in ev.notes)


def test_range_for_numbers_holds_about_eighty_percent() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=4000)
    y = 3 * x + rng.normal(scale=2.0, size=4000)
    pred = 3 * x
    ev = evaluate(task="regression", classes=[], y_oof=y[:2000], oof_pred=pred[:2000], oof_proba=None,
                  y_test=y[2000:], test_pred=pred[2000:], test_proba=None,
                  x_test=np.empty((2000, 0), dtype=object), names=[], kinds=[])
    assert ev.half_width is not None and 2.2 < ev.half_width < 3.0  # about 1.28 standard deviations
    assert 0.76 <= ev.coverage <= 0.84


def test_fairness_gap_is_shown_by_group() -> None:
    rng = np.random.default_rng(3)
    n = 2000
    sex = np.array(rng.choice(["f", "m"], n), dtype=object)
    y = (rng.random(n) < 0.3).astype(int)
    # The model is good for one group and guesses for the other.
    p = np.where(sex == "f", np.where(y == 1, 0.8, 0.2), rng.random(n) * 0.5)
    proba = np.column_stack([1 - p, p])
    ev = evaluate(task="classification", classes=["0", "1"], y_oof=y, oof_pred=(p > .5).astype(int), oof_proba=proba,
                  y_test=y, test_pred=(p > .5).astype(int), test_proba=proba, x_test=sex.reshape(-1, 1),
                  names=["sex"], kinds=["categorical"])
    assert [g.column for g in ev.groups] == ["sex"]
    assert ev.groups[0].gap and "finds" in ev.groups[0].gap
    assert "By sex" in str(render.evaluation_view(ev))


def test_answers_set_the_prediction_moment() -> None:
    from mlteam.contract import DataContract

    contract = DataContract("h", "single", "t")
    contract.answers = {"When?": "The prediction is made 2 weeks after signup."}
    assert horizon_from_answers(contract) == 14
    contract.answers = {"When?": "At signup."}
    assert horizon_from_answers(contract) is None


def _orders(folder: Path, customers: int = 150, seed: int = 4) -> Path:
    """Several rows per customer: the same customer must not be in training and test."""
    rng = random.Random(seed)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "orders.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle)
        w.writerow(["order_ref", "customer_id", "amount", "items", "returned"])
        k = 0
        for c in range(customers):
            taste = rng.random()
            for _ in range(rng.randint(2, 6)):
                k += 1
                items = rng.randint(1, 8)
                w.writerow([f"O{k:05d}", f"C{c:04d}", round(rng.uniform(5, 200), 2), items,
                            int(rng.random() < 0.15 + 0.5 * taste * (items > 4))])
    return path


async def test_repeated_customers_are_kept_on_one_side(tmp_path: Path) -> None:
    team = Team(new_run(tmp_path / "runs"), test_seconds=0)
    try:
        draft = await team.prepare([str(_orders(tmp_path / "orders"))])
        assert draft.group_column == "customer_id" and "repeats" in draft.rationale["group"]
        issues = validate_contract(team.store, team.profile, draft)
        assert any(i.field == "group" and "kept on one side" in i.message for i in issues)
        await team.consult()
        team.approve(draft, "tester")
        features = await team.build_features()
        team.approve_features([f.name for f in features if f.default_on], "tester")
        run = await team.train_models()
        assert "split by customer_id" in run.split

        from mlteam.train import _load

        data = _load(team.store, team.contract, [f for f in team.features if f.name in run.features])
        assert data.groups_train is not None
        rows = team.store.query("SELECT customer_id FROM orders ORDER BY order_ref")[1]
        assert len(rows) == len(data.y_train) + len(data.y_test)
    finally:
        team.close()


async def test_rare_outcome_ensemble_and_report(tmp_path: Path) -> None:
    team = Team(new_run(tmp_path / "runs"), test_seconds=0)
    try:
        source = samples.make("loans", tmp_path / "data")
        draft = await team.prepare([str(source)])
        await team.consult()
        draft.cost_false_alarm, draft.cost_miss = 1.0, 4.0
        team.approve(draft, "tester")
        features = await team.build_features()
        team.approve_features([f.name for f in features if f.default_on], "tester")
        run = await team.train_models()

        names = {r.name for r in run.results}
        assert "ensemble" in names
        ensemble = run.result("ensemble")
        assert ensemble and len(ensemble.members) >= 2 and ensemble.error is None
        best = next(r for r in run.results if r.recommended)
        assert best.evaluation and best.evaluation["kind"] == "binary"
        assert "lowest cost" in best.evaluation["cutoff_reason"]
        detail = str(render.model_detail(best, run))
        assert "Before you use it" in detail and "Cut-off" in detail

        path = team.approve_model(best.name, "tester")
        card = json.loads(path.read_text(encoding="utf-8"))
        assert card["delivery"]["cutoff"] == best.evaluation["cutoff"]
        assert card["artifact"].endswith(".delivered.joblib")
        report = team.report_path
        assert report and report.exists() and (team.run.root / "report.html").exists()
        html = report.read_text(encoding="utf-8")
        for part in ("What was decided", "Models compared", "Before you use it", "Risks and open concerns",
                     "How to use it", best.name):
            assert part in html

        # The approved ensemble also works end to end.
        team.status["model_gate"] = "waiting"
        card_path = team.approve_model("ensemble", "tester")
        card = json.loads(card_path.read_text(encoding="utf-8"))
        assert card["family"] == "ensemble" and len(card["members"]) >= 2
        scored = predict_file(team.run, source)
        assert scored.rows > 0
    finally:
        team.close()


async def test_numbers_get_a_range_in_the_output(tmp_path: Path) -> None:
    folder = tmp_path / "houses"
    folder.mkdir()
    rng = random.Random(5)
    path = folder / "houses.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle)
        w.writerow(["house_id", "rooms", "area", "age", "price"])
        for i in range(400):
            rooms, area = rng.randint(1, 6), rng.uniform(40, 220)
            w.writerow([i, rooms, round(area, 1), rng.randint(0, 80), round(1000 * area + 5000 * rooms + rng.gauss(0, 20000))])
    team = Team(new_run(tmp_path / "runs"), test_seconds=0)
    try:
        draft = await team.prepare([str(path)])
        await team.consult()
        team.approve(draft, "tester")
        features = await team.build_features()
        team.approve_features([f.name for f in features if f.default_on], "tester")
        run = await team.train_models()
        best = next(r for r in run.results if r.recommended)
        assert best.evaluation and best.evaluation["half_width"] > 0
        team.approve_model(best.name, "tester")
        scored = predict_file(team.run, path)
        header = (team.run.root / scored.output).read_text(encoding="utf-8").splitlines()[0]
        assert "low_80" in header and "high_80" in header
    finally:
        team.close()


def test_runs_command_lists_runs(tmp_path: Path, capsys) -> None:
    run = new_run(tmp_path / "runs")
    (run.root / "manifest.json").write_text(json.dumps({"files": [{"name": "a.csv"}]}), encoding="utf-8")
    (run.root / "contract.json").write_text(json.dumps({"target": "churn", "approved_by": "x"}), encoding="utf-8")
    assert cli_main(["runs", "--runs-dir", str(tmp_path / "runs")]) == 0
    out = capsys.readouterr().out
    assert "churn" in out and "contract" in out


def test_gap_already_in_the_data_is_told_apart_from_one_the_model_adds() -> None:
    rng = np.random.default_rng(6)
    n = 4000
    group = np.array(rng.choice(["a", "b"], n), dtype=object)
    y = (rng.random(n) < np.where(group == "a", 0.40, 0.10)).astype(int)  # real rates differ
    p = np.where(y == 1, 0.8, 0.1)  # the model treats both groups alike
    proba = np.column_stack([1 - p, p])
    ev = evaluate(task="classification", classes=["0", "1"], y_oof=y, oof_pred=(p > .5).astype(int), oof_proba=proba,
                  y_test=y, test_pred=(p > .5).astype(int), test_proba=proba, x_test=group.reshape(-1, 1),
                  names=["sex"], kinds=["categorical"])
    report = ev.groups[0]
    assert report.gap is None and report.context and "in the data itself" in report.context

    # Engineered features named after a sensitive column are not groups.
    from mlteam.evaluate import sensitive_columns

    assert sensitive_columns(["age", "age__squared", "age_hours", "marital-status"], derived={"age_hours"}) == [0, 3]
