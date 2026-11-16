from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from tau_ai.fake import FakeProvider
from textual.widgets import Button, Input, SelectionList, Static, TabbedContent

from mlteam import samples
from mlteam.agents.brain import Brain
from mlteam.contract import has_errors, validate_contract
from mlteam.features import Feature, default_features, evaluate_features, frame_sql
from mlteam.runstore import new_run
from mlteam.team import Team
from mlteam.tui.app import MLTeamApp
from tests.test_team import tool_turn
from tests.test_tui import SIZE, settle, shot, text_of

ORDERS = "(SELECT count(*) FROM orders o WHERE o.customer_id = b.customer_id)"


def _by_name(features: list[Feature]) -> dict[str, Feature]:
    return {f.name: f for f in features}


# -- the engine ----------------------------------------------------------


def test_default_features_cover_base_lookup_and_summaries(load, scenario) -> None:
    loaded = load(scenario("shop"))
    features = evaluate_features(loaded.store, loaded.draft, default_features(loaded.draft, loaded.profile))
    names = set(_by_name(features))

    assert {"plan", "signup_date_month", "orders__count", "orders__amount__mean", "regions__tier"} <= names
    assert not names & {"churned", "customer_id", "region_id"}  # target, row key, join key stay out
    assert all(f.usable for f in features)
    # One column per feature, one row per base row: the row count cannot change here.
    columns, rows = loaded.store.query(f"SELECT * FROM ({frame_sql(loaded.draft, features)})")
    assert len(rows) == 250 and columns[:2] == ["customer_id", "churned"]
    assert set(columns[2:]) == names


def test_prediction_moment_hides_later_rows_from_any_feature(load, scenario) -> None:
    loaded = load(scenario("shop"))
    store = loaded.store
    engineered = Feature("n_orders", ORDERS, origin="engineer")

    open_contract = copy.deepcopy(loaded.draft)
    evaluate_features(store, open_contract, [engineered])
    assert store.scalar(f'SELECT sum("n_orders") FROM ({frame_sql(open_contract, [engineered])})') == 645

    timed = copy.deepcopy(loaded.draft)
    timed.horizon_days = 200
    issues = validate_contract(store, loaded.profile, timed)
    assert not has_errors(issues)
    assert any(i.field == "horizon" and "of 645 rows exist 200 days after signup_date" in i.message for i in issues)

    expected = store.scalar(
        "SELECT count(*) FROM orders o JOIN customers c USING (customer_id) "
        "WHERE o.order_date <= c.signup_date + INTERVAL '200' DAY"
    )
    assert 0 < expected < 645
    # The Engineer's expression is unchanged, yet it now sees only the earlier orders.
    assert store.scalar(f'SELECT sum("n_orders") FROM ({frame_sql(timed, [engineered])})') == expected
    defaults = _by_name(evaluate_features(store, timed, default_features(timed, loaded.profile)))
    assert "orders__days_since_last" in defaults and defaults["orders__days_since_last"].usable


def test_no_prediction_moment_on_dated_table_is_a_warning(load, scenario) -> None:
    loaded = load(scenario("shop"))
    issues = validate_contract(loaded.store, loaded.profile, loaded.draft)
    assert any(i.level == "warning" and i.field == "horizon" and "no prediction moment" in i.message
               for i in issues)
    bad = copy.deepcopy(loaded.draft)
    bad.horizon_days, bad.time_column = 30, None
    assert any(i.level == "error" and i.field == "horizon"
               for i in validate_contract(loaded.store, loaded.profile, bad))


@pytest.mark.parametrize(
    ("name", "sql", "reason"),
    [
        ("uses_target", 'CASE WHEN b."churned" = 1 THEN 1 ELSE 0 END', "uses the target"),
        ("bypass", "(SELECT count(*) FROM main.orders)", "bypass the time rule"),
        ("two_things", "1; DROP TABLE customers", "one expression"),
        ("reads_file", "(SELECT count(*) FROM read_csv('C:/x.csv'))", "not allowed"),
        ("bad name!", "1", "name must start"),
        ("plan", 'b."plan"', "already exists"),
        ("constant", "42", "same value on every row"),
        ("a_date", 'b."signup_date"', "returns a date"),
        ("broken", "b.no_such_column + 1", "no_such_column"),
        ("many_rows", "(SELECT o.amount FROM orders o WHERE o.customer_id = b.customer_id)", "ore than one row"),
    ],
)
def test_bad_features_are_rejected_with_a_reason(load, scenario, name, sql, reason) -> None:
    loaded = load(scenario("shop"))
    features = [Feature("plan", 'b."plan"'), Feature(name, sql, origin="engineer")]
    evaluate_features(loaded.store, loaded.draft, features)
    assert features[0].usable
    assert features[1].error is not None and reason in features[1].error


def test_leaky_feature_is_usable_but_flagged_and_off_by_default(load, scenario) -> None:
    loaded = load(scenario("loans"))
    features = [Feature("income", 'b."ApplicantIncome"'), Feature("decision", 'b."Decision_Code"', origin="engineer")]
    evaluate_features(loaded.store, loaded.draft, features)
    assert features[0].default_on and features[0].leak is None
    assert features[1].usable and not features[1].default_on
    assert "100.0%" in (features[1].leak or "")


# -- the harness with agents ---------------------------------------------


def _engineer_streams() -> list:
    return [
        tool_turn("e1", "try_feature", {"name": "big_spender", "sql": f"{ORDERS} >= 4"}),
        # First submission contains a feature built from the target: rejected.
        tool_turn("e2", "submit_features", {"add": [
            {"name": "big_spender", "sql": f"{ORDERS} >= 4", "why": "Frequent buyers rarely leave."},
            {"name": "cheat", "sql": 'b."churned" * 1', "why": "oops"}]}),
        tool_turn("e3", "submit_features", {
            "add": [{"name": "big_spender", "sql": f"{ORDERS} >= 4", "why": "Frequent buyers rarely leave."}],
            "remove": ["signup_date_weekday"]}),
        tool_turn("s1", "submit_review", {
            "verdict": "concerns", "summary": "Usable, one thing to check.",
            "concerns": [{"severity": "low", "field": "orders__amount__sum",
                          "message": "Total and average spend say nearly the same thing.",
                          "evidence": "both come from orders.amount"}]}),
    ]


async def _team_at_feature_stage(tmp_path: Path, streams: list | None) -> tuple[Team, list, FakeProvider | None]:
    events: list = []
    provider = FakeProvider(streams) if streams is not None else None
    team = Team(new_run(tmp_path / "runs"), emit=events.append)
    draft = await team.prepare([str(samples.make("shop", tmp_path / "data"))])
    await team.consult()
    draft.horizon_days = 200
    team.approve(draft, "tester")
    if provider is not None:
        team.brain = Brain(provider=provider, model="scripted", label="fake/scripted")
    return team, events, provider


async def test_features_need_an_approved_contract(tmp_path: Path) -> None:
    team = Team(new_run(tmp_path / "runs"))
    try:
        await team.prepare([str(samples.make("loans", tmp_path / "data"))])
        await team.consult()
        with pytest.raises(ValueError, match="Approve the data contract first"):
            await team.build_features()
    finally:
        team.close()


async def test_rules_only_feature_stage_and_approval(tmp_path: Path) -> None:
    team, _, _ = await _team_at_feature_stage(tmp_path, None)
    try:
        features = await team.build_features()
        assert team.status["engineer"] == "skipped" and team.status["feature_gate"] == "waiting"
        picked = [f.name for f in features if f.default_on]

        with pytest.raises(ValueError, match="at least one"):
            team.approve_features([])
        with pytest.raises(ValueError, match="Unknown features"):
            team.approve_features(["nope"])
        path = team.approve_features(picked, "tester")

        saved = json.loads(path.read_text(encoding="utf-8"))
        assert path.name == "features.v1.json" and saved["selected"] == picked
        assert saved["contract_version"] == 1 and saved["horizon_days"] == 200
        assert {f["name"]: f["sql"] for f in saved["features"]}["orders__count"].startswith("(SELECT count(*)")
        assert team.status["feature_gate"] == "done"
        trace = [json.loads(line) for line in team.run.trace.read_text(encoding="utf-8").splitlines()]
        assert any(r["kind"] == "feature_approval" and r["selected"] == picked for r in trace)

        # Approving the contract again makes the old features stale.
        team.approve(team.contract, "tester")
        assert team.features == [] and team.status["feature_gate"] == "pending"
    finally:
        team.close()


async def test_engineer_and_skeptic_work_on_features(tmp_path: Path) -> None:
    team, events, provider = await _team_at_feature_stage(tmp_path, _engineer_streams())
    try:
        features = _by_name(await team.build_features())

        assert team.status["engineer"] == "done" and team.status["feature_skeptic"] == "done"
        assert "big_spender" in features and features["big_spender"].origin == "engineer"
        assert features["big_spender"].kind == "boolean" and features["big_spender"].usable
        assert "cheat" not in features and "signup_date_weekday" not in features
        submits = [e for e in events if e.kind == "tool_end" and e.data["tool"] == "submit_features"]
        assert [e.data["is_error"] for e in submits] == [True, False]
        assert "uses the target" in submits[0].text
        assert team.feature_review is not None and team.feature_review.concerns[0].field == "orders__amount__sum"
        assert team.review is None  # the contract review is a separate thing

        assert provider is not None
        engineer_call, skeptic_call = provider.calls[0], provider.calls[-1]
        assert "Engineer" in engineer_call[1] and "feature list" in skeptic_call[1]
        assert {t.name for t in engineer_call[3]} >= {"try_feature", "submit_features", "run_sql"}
        assert "200 days after signup_date" in engineer_call[2][0].text
    finally:
        team.close()


# -- the app -------------------------------------------------------------


async def test_app_contract_then_features_gate(tmp_path: Path) -> None:
    app = MLTeamApp(["demo:shop"], runs_dir=tmp_path / "runs")
    async with app.run_test(size=SIZE) as pilot:
        await settle(app, pilot)
        assert "Approve the data contract first" in text_of(app, "#feat_rule")
        assert "no prediction moment is set" in text_of(app, "#checks")

        # Setting the prediction moment changes the checks, then the features.
        app.query_one("#horizon", Input).value = "200"
        await settle(app, pilot)
        assert app.current is not None and app.current.horizon_days == 200
        assert "exist 200 days after signup_date" in text_of(app, "#checks")
        shot(app, "contract_horizon")

        await pilot.press("ctrl+s")
        await settle(app, pilot)
        assert app.team is not None and app.team.status["feature_gate"] == "waiting"
        assert app.query_one("#main", TabbedContent).active == "features"
        assert "Prediction moment: 200 days after signup_date" in text_of(app, "#feat_rule")
        listing = app.query_one("#feature_list", SelectionList)
        assert "orders__days_since_last" in listing.selected and "plan" in listing.selected
        assert "How it is computed" in text_of(app, "#feat_detail")
        shot(app, "features_shop")

        listing.deselect("regions__region_name")
        await settle(app, pilot)
        assert "ticked" in text_of(app, "#feat_hint")
        await pilot.press("ctrl+s")  # on the Features tab this approves features
        await settle(app, pilot)
        saved = json.loads(app.team.run.features_latest.read_text(encoding="utf-8"))
        assert saved["version"] == 1 and "regions__region_name" not in saved["selected"]
        assert "orders__count" in saved["selected"] and saved["horizon_days"] == 200
        assert "Approved v1" in str(app.query_one("#approve_features", Button).label)
        # Approving features starts training; the next gate is the model review.
        assert app.team.status["feature_gate"] == "done" and app.team.status["model_gate"] == "waiting"
        assert app.query_one("#main", TabbedContent).active == "models"


async def test_app_shows_engineer_features_and_leak_flag_off(tmp_path: Path) -> None:
    streams = [
        tool_turn("a1", "submit_contract", {"target": "Loan_Status", "task": "classification",
                                            "metric": "roc_auc", "grain": "one row = one loan",
                                            "drop_columns": ["Loan_ID"]}),
        tool_turn("s1", "submit_review", {"verdict": "approve", "summary": "Sound."}),
        tool_turn("e1", "submit_features", {"add": [
            {"name": "income_per_loan", "sql": 'b."ApplicantIncome" / nullif(b."LoanAmount", 0)',
             "why": "Income relative to the amount asked for shows how stretched the applicant is."},
            {"name": "decision", "sql": 'b."Decision_Code"', "why": "The recorded decision code."}]}),
        tool_turn("s2", "submit_review", {
            "verdict": "concerns", "summary": "One feature gives the answer away.",
            "concerns": [{"severity": "high", "field": "decision",
                          "message": "decision matches the target on every row.", "evidence": "purity 100%"}]}),
    ]
    brain = Brain(provider=FakeProvider(streams), model="scripted", label="fake/scripted")
    app = MLTeamApp(["demo:loans"], runs_dir=tmp_path / "runs", brain=brain)
    async with app.run_test(size=SIZE) as pilot:
        await settle(app, pilot)
        await pilot.press("ctrl+s")
        await settle(app, pilot)

        assert app.team is not None and app.team.status["engineer"] == "done"
        listing = app.query_one("#feature_list", SelectionList)
        assert "income_per_loan" in listing.selected
        assert "decision" not in listing.selected  # leak-flagged: present, but off
        labels = " | ".join(str(listing.get_option_at_index(i).prompt) for i in range(listing.option_count))
        assert "leak?" in labels and "Engineer" in labels
        assert "matches the target on every row" in text_of(app, "#feat_review")
        shot(app, "features_loans_engineer")

        await pilot.press("ctrl+s")
        await settle(app, pilot)
        saved = json.loads(app.team.run.features_latest.read_text(encoding="utf-8"))
        assert "income_per_loan" in saved["selected"] and "decision" not in saved["selected"]


# -- querying the feature table ------------------------------------------


def test_features_query_sees_one_table_cut_at_the_prediction_moment(load, scenario) -> None:
    from mlteam.features import features_query_sql

    loaded = load(scenario("shop"))
    timed = copy.deepcopy(loaded.draft)
    timed.horizon_days = 200
    feats = [f for f in evaluate_features(loaded.store, timed, default_features(timed, loaded.profile)) if f.usable]

    sql = features_query_sql(timed, feats, "SELECT count(*), sum(orders__count), (SELECT count(*) FROM orders) FROM features")
    _, rows, _ = loaded.store.agent_query(sql)
    rows_n, total, visible = rows[0]
    assert rows_n == 250 and total == visible and 0 < visible < 645  # joined tables are cut too

    for bad, reason in (
        ("SELECT count(*) FROM main.orders", "bypass the time rule"),
        ("DROP TABLE customers", "Only SELECT"),
        ("SELECT 1; SELECT 2", "one statement"),
        ("SELECT * FROM read_csv('x.csv')", "not allowed"),
    ):
        with pytest.raises(ValueError, match=reason):
            features_query_sql(timed, feats, bad)


async def test_feature_skeptic_can_query_the_feature_table(tmp_path: Path) -> None:
    streams = [
        tool_turn("e1", "submit_features", {"add": []}),
        tool_turn("s1", "query_features", {"sql": "SELECT count(*) AS n, sum(orders__count) AS orders_used FROM features"}),
        tool_turn("s2", "query_features", {"sql": "SELECT count(*) FROM main.orders"}),
        tool_turn("s3", "submit_review", {"verdict": "approve", "summary": "Sound."}),
    ]
    team, events, provider = await _team_at_feature_stage(tmp_path, streams)
    try:
        await team.build_features()
        results = [e for e in events if e.kind == "tool_end" and e.data["tool"] == "query_features"]
        assert [e.data["is_error"] for e in results] == [False, True]
        assert "n | orders_used" in results[0].text and "250 | 141" in results[0].text
        assert "bypass the time rule" in results[1].text
        assert team.feature_review is not None and team.feature_review.verdict == "approve"
        assert provider is not None
        assert "query_features" in {t.name for t in provider.calls[-1][3]}
    finally:
        team.close()
