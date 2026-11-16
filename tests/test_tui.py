"""Drive the real Textual app headlessly."""

from __future__ import annotations

import json
import os
from pathlib import Path

from textual.widgets import Button, DataTable, Input, OptionList, Select, SelectionList, Static, TabbedContent

from mlteam.agents.brain import Brain
from mlteam.tui.app import IntakeScreen, MLTeamApp
from tests.test_team import tool_turn

SIZE = (150, 48)
SHOTS = os.environ.get("MLTEAM_SHOTS")


async def settle(app: MLTeamApp, pilot) -> None:
    for _ in range(4):
        await app.workers.wait_for_complete()
        await pilot.pause()


def shot(app: MLTeamApp, name: str) -> None:
    if SHOTS:
        Path(SHOTS).mkdir(parents=True, exist_ok=True)
        (Path(SHOTS) / f"{name}.svg").write_text(app.export_screenshot(), encoding="utf-8")


def text_of(app: MLTeamApp, selector: str) -> str:
    return str(app.query_one(selector, Static).render())


async def test_rules_only_flow_to_approval(tmp_path: Path) -> None:
    app = MLTeamApp(["demo:loans"], runs_dir=tmp_path / "runs", brain_note="Not run: no model connected.")
    async with app.run_test(size=SIZE) as pilot:
        await settle(app, pilot)

        assert app.team is not None and app.team.status["gate"] == "waiting"
        assert app.query_one("#main", TabbedContent).active == "contract"
        assert app.query_one("#target", Select).value == "Loan_Status"
        assert app.query_one("#task", Select).value == "classification"
        assert app.query_one("#key", Select).value == "Loan_ID"
        assert app.query_one("#grain", Input).value == "one row = one loan"
        assert set(app.query_one("#drops", SelectionList).selected) == {"Loan_ID", "Decision_Code"}
        assert "Nothing blocks approval" in text_of(app, "#checks")
        assert "Decision_Code" in text_of(app, "#reasons")
        assert "your call" in text_of(app, "#stages")
        assert app.query_one("#approve", Button).disabled is False
        shot(app, "contract_loans")

        # Data tab shows what the contract does with each column.
        grid = app.query_one("#columns", DataTable)
        roles = {str(grid.get_row_at(i)[0]): str(grid.get_row_at(i)[1]) for i in range(grid.row_count)}
        assert roles["Loan_Status"] == "TARGET" and roles["Decision_Code"] == "leak?"
        assert roles["Loan_ID"] == "row key" and roles["Gender"] == "feature"

        await pilot.press("ctrl+s")
        await settle(app, pilot)
        saved = json.loads(app.team.run.contract_latest.read_text(encoding="utf-8"))
        assert saved["version"] == 1 and saved["target"] == "Loan_Status" and saved["approved_by"] == "human"
        assert app.team.status["gate"] == "done"


async def test_editing_the_form_revalidates_and_blocks_bad_contracts(tmp_path: Path) -> None:
    app = MLTeamApp(["demo:loans"], runs_dir=tmp_path / "runs")
    async with app.run_test(size=SIZE) as pilot:
        await settle(app, pilot)

        # A key that repeats is an error, and approval is blocked.
        app.query_one("#key", Select).value = "Gender"
        await settle(app, pilot)
        assert "repeats" in text_of(app, "#checks")
        assert app.query_one("#approve", Button).disabled is True
        await pilot.press("ctrl+s")
        await settle(app, pilot)
        assert not app.team.run.contract_latest.exists()

        # Changing the target redoes task, metric and the leakage scan.
        app.query_one("#key", Select).value = "Loan_ID"
        app.query_one("#target", Select).value = "ApplicantIncome"
        await settle(app, pilot)
        assert app.current is not None and app.current.target == "ApplicantIncome"
        assert app.query_one("#task", Select).value == "regression"
        assert app.query_one("#metric", Select).value == "rmse"
        assert "Decision_Code" not in app.query_one("#drops", SelectionList).selected
        assert app.query_one("#approve", Button).disabled is False

        # Reset goes back to what the team proposed.
        await pilot.press("ctrl+r")
        await settle(app, pilot)
        assert app.query_one("#target", Select).value == "Loan_Status"
        assert "Decision_Code" in app.query_one("#drops", SelectionList).selected


async def test_related_tables_show_join_plan_and_unchecking_a_join_updates_it(tmp_path: Path) -> None:
    app = MLTeamApp(["demo:shop"], runs_dir=tmp_path / "runs")
    async with app.run_test(size=SIZE) as pilot:
        await settle(app, pilot)

        plan = text_of(app, "#relations_text")
        assert "summarise orders" in plan and "look up regions" in plan
        assert "250 before and after" in plan
        assert [str(o.prompt).split()[0] for o in app.query_one("#tables", OptionList).options] == [
            "customers", "orders", "regions"]
        app.query_one("#main", TabbedContent).active = "relations"
        await pilot.pause()
        shot(app, "relations_shop")

        joins = app.query_one("#joins", SelectionList)
        joins.deselect(0)
        await settle(app, pilot)
        assert app.current is not None and [j.table for j in app.current.joins] == ["regions"]
        assert "summarise orders" not in text_of(app, "#relations_text")

        # Switching the base table re-proposes everything for that table.
        app.query_one("#base", Select).value = "orders"
        await settle(app, pilot)
        assert app.current.base_table == "orders" and app.current.grain_keys == ["order_id"]
        assert app.query_one("#approve", Button).disabled is True  # no target on orders


async def test_intake_screen_then_agents_feed_the_team_log(tmp_path: Path) -> None:
    from tau_ai.fake import FakeProvider

    streams = [
        tool_turn("a1", "run_sql", {"sql": "SELECT Survived, count(*) FROM train GROUP BY 1"},
                  text="Checking how balanced the target is."),
        tool_turn("a2", "submit_contract", {"target": "Survived", "task": "classification",
                                            "metric": "roc_auc", "grain": "one row = one passenger",
                                            "rationale": {"target": "Survived is what the test file leaves out."}}),
        tool_turn("s1", "submit_review", {
            "verdict": "concerns", "summary": "Usable. One column needs a decision.",
            "concerns": [{"severity": "medium", "field": "Sex",
                          "message": "Sex is a protected attribute and a strong predictor here.",
                          "evidence": "survival 70% vs 20% by sex"}]}),
    ]
    brain = Brain(provider=FakeProvider(streams), model="scripted", label="fake/scripted")
    app = MLTeamApp(runs_dir=tmp_path / "runs", brain=brain)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        assert isinstance(app.screen, IntakeScreen)
        shot(app, "intake")
        for key in "demo:titanic":
            await pilot.press("colon" if key == ":" else key)
        await pilot.press("enter")
        await settle(app, pilot)

        assert not isinstance(app.screen, IntakeScreen)
        assert app.team is not None and app.team.contract is not None
        assert app.team.contract.proposed_by == "analyst"
        assert "the Analyst" in text_of(app, "#reasons")
        assert "Survived is what the test file leaves out" in text_of(app, "#reasons")
        assert "protected attribute" in text_of(app, "#review")
        assert "fake/scripted" in text_of(app, "#statusbar")
        shot(app, "contract_titanic_agents")

        app.query_one("#main", TabbedContent).active = "team"
        await pilot.pause()
        shot(app, "team_log")
        app.query_one("#main", TabbedContent).active = "data"
        await pilot.pause()
        shot(app, "data_titanic")


async def test_a_join_the_analyst_removed_can_be_put_back_and_its_question_is_shown(tmp_path: Path) -> None:
    """Seen in the first real-model run: the Analyst dropped the orders join on a
    guess about prediction time. The person must be able to see why and undo it."""
    from tau_ai.fake import FakeProvider

    streams = [
        tool_turn("a1", "submit_contract", {
            "target": "churned", "task": "classification", "metric": "roc_auc",
            "grain": "one row = one customer",
            "joins": [{"table": "regions", "kind": "lookup", "base_key": "region_id", "other_key": "id"}],
            "leakage_suspects": [{"column": "order_date", "reason": "orders may fall after the churn date."}],
            "open_questions": ["At what moment do you predict churn: at signup, or later once orders exist?"],
        }),
        tool_turn("s1", "submit_review", {"verdict": "approve", "summary": "Sound."}),
    ]
    brain = Brain(provider=FakeProvider(streams), model="scripted", label="fake/scripted")
    app = MLTeamApp(["demo:shop"], runs_dir=tmp_path / "runs", brain=brain)
    async with app.run_test(size=SIZE) as pilot:
        await settle(app, pilot)

        assert app.current is not None and [j.table for j in app.current.joins] == ["regions"]
        assert app.team is not None and app.team.contract.proposed_by == "analyst"
        asked = " ".join(str(w.render()) for w in app.query("#answers .question").results(Static))
        assert "At what moment do you predict churn" in asked
        # An unanswered question blocks approval, by button and by shortcut.
        assert "Unanswered question" in text_of(app, "#checks")
        assert app.query_one("#approve", Button).disabled is True
        await pilot.press("ctrl+s")
        await settle(app, pilot)
        assert not app.team.run.contract_latest.exists()
        why = " ".join(text_of(app, "#reasons").split())
        assert "after the churn date." in why and "after the churn date.." not in why

        joins = app.query_one("#joins", SelectionList)
        labels = [str(joins.get_option_at_index(i).prompt) for i in range(joins.option_count)]
        assert any("summarise orders" in label for label in labels)
        orders_index = next(i for i, label in enumerate(labels) if "orders" in label)
        assert orders_index not in joins.selected
        shot(app, "contract_shop_question")

        joins.select(orders_index)
        await settle(app, pilot)
        assert sorted(j.table for j in app.current.joins) == ["orders", "regions"]
        assert "250 before and after" in text_of(app, "#checks")
        assert app.query_one("#approve", Button).disabled is True  # still unanswered

        # Answering unlocks approval; the answer survives a target change and is saved.
        app.query_one("#answer_0", Input).value = "Thirty days after signup."
        await settle(app, pilot)
        assert "Unanswered question" not in text_of(app, "#checks")
        assert app.query_one("#approve", Button).disabled is False
        shot(app, "contract_shop_answered")
        app.query_one("#target", Select).value = "plan"
        await settle(app, pilot)
        app.query_one("#target", Select).value = "churned"
        await settle(app, pilot)
        assert app.query_one("#answer_0", Input).value == "Thirty days after signup."

        await pilot.press("ctrl+s")
        await settle(app, pilot)
        saved = json.loads(app.team.run.contract_latest.read_text(encoding="utf-8"))
        assert sorted(j["table"] for j in saved["joins"]) == ["orders", "regions"]
        assert saved["proposed_by"] == "analyst"
        assert saved["answers"] == {saved["open_questions"][0]: "Thirty days after signup."}


async def test_bad_source_returns_to_intake_with_the_reason(tmp_path: Path) -> None:
    app = MLTeamApp([str(tmp_path / "missing.csv")], runs_dir=tmp_path / "runs")
    async with app.run_test(size=SIZE) as pilot:
        await settle(app, pilot)
        assert isinstance(app.screen, IntakeScreen)
        assert "Not found" in str(app.screen.query_one("#intake_error", Static).render())
