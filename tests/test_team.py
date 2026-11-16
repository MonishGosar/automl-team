"""The harness with two Tau agents, driven by a scripted model so the run is
deterministic and costs nothing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tau_agent import AssistantMessage, TextContent, ToolCall
from tau_agent.provider_events import AssistantDoneEvent, AssistantErrorEvent
from tau_ai.fake import FakeProvider

from mlteam import samples
from mlteam.agents.brain import Brain
from mlteam.agents.roles import TeamEvent
from mlteam.runstore import new_run
from mlteam.team import Team


def tool_turn(call_id: str, name: str, args: dict, text: str = "") -> list:
    content: list = [TextContent(text=text)] if text else []
    content.append(ToolCall(id=call_id, name=name, arguments=args))
    message = AssistantMessage(content=content, stop_reason="toolUse")
    return [AssistantDoneEvent(reason="toolUse", message=message)]


def text_turn(text: str) -> list:
    return [AssistantDoneEvent(reason="stop", message=AssistantMessage(content=text))]


def make_team(tmp_path: Path, streams: list | None, **kwargs) -> tuple[Team, list[TeamEvent], FakeProvider | None]:
    events: list[TeamEvent] = []
    provider = FakeProvider(streams) if streams is not None else None
    brain = Brain(provider=provider, model="scripted", label="fake/scripted") if provider else None
    team = Team(new_run(tmp_path / "runs"), brain=brain, emit=events.append, **kwargs)
    return team, events, provider


async def test_rules_only_run_reaches_the_gate_and_approves(tmp_path: Path) -> None:
    team, events, _ = make_team(tmp_path, None, brain_note="No model connected.")
    try:
        draft = await team.prepare([str(samples.make("loans", tmp_path / "data"))])
        await team.consult()

        assert draft.target == "Loan_Status"
        assert team.status["analyst"] == "skipped" and team.status["skeptic"] == "skipped"
        assert team.status["gate"] == "waiting"

        path = team.approve(draft, approved_by="tester")
        assert path.name == "contract.v1.json"
        assert team.status["gate"] == "done"

        trace = [json.loads(line) for line in team.run.trace.read_text(encoding="utf-8").splitlines()]
        approval = next(r for r in trace if r["kind"] == "approval")
        assert approval["approved_by"] == "tester" and approval["contract"]["target"] == "Loan_Status"
        assert [e.stage for e in events if e.kind == "stage" and e.data["status"] == "done"][:4] == [
            "intake", "load", "profile", "draft"]
    finally:
        team.close()


async def test_analyst_then_skeptic_run_as_separate_tau_agents(tmp_path: Path) -> None:
    streams = [
        # Analyst: look at the data, get rejected once by the harness, then fix it.
        tool_turn("a1", "run_sql", {"sql": "SELECT churned, count(*) FROM customers GROUP BY 1"},
                  text="Checking the target balance."),
        tool_turn("a2", "submit_contract", {
            "target": "churned", "task": "classification", "metric": "pr_auc",
            "grain": "one row = one customer",
            "joins": [{"table": "orders", "kind": "lookup", "base_key": "customer_id", "other_key": "customer_id"}],
        }),
        tool_turn("a3", "submit_contract", {
            "target": "churned", "task": "classification", "metric": "pr_auc",
            "grain": "one row = one customer",
            "joins": [
                {"table": "orders", "kind": "aggregate", "base_key": "customer_id", "other_key": "customer_id"},
                {"table": "regions", "kind": "lookup", "base_key": "region_id", "other_key": "id"},
            ],
            "rationale": {"joins": "Orders has many rows per customer, so it is summarised first."},
        }),
        # Skeptic: one review.
        tool_turn("s1", "submit_review", {
            "verdict": "concerns", "summary": "Sound, with one thing to watch.",
            "concerns": [{"severity": "medium", "field": "orders",
                          "message": "Orders placed after the churn date would leak the outcome.",
                          "evidence": "orders.order_date runs to 2025-09"}],
        }),
    ]
    team, events, provider = make_team(tmp_path, streams)
    try:
        await team.prepare([str(samples.make("shop", tmp_path / "data"))])
        await team.consult()

        # The Analyst's contract replaced the draft, after the harness rejected the fan-out join.
        assert team.contract is not None and team.contract.proposed_by == "analyst"
        assert team.contract.metric == "pr_auc"
        assert {j.table: j.kind for j in team.contract.joins} == {"orders": "aggregate", "regions": "lookup"}
        assert "summarised first" in team.contract.rationale["joins"]
        rejected = [e for e in events if e.kind == "tool_end" and e.data["tool"] == "submit_contract"]
        assert [e.data["is_error"] for e in rejected] == [True, False]
        assert "row count" in rejected[0].text

        # The Skeptic's review is recorded separately.
        assert team.review is not None and team.review.verdict == "concerns"
        assert team.review.concerns[0].field == "orders"
        assert team.status == {**team.status, "analyst": "done", "skeptic": "done", "gate": "waiting"}

        # Two independent agents: different system prompts, different tools, no shared history.
        assert provider is not None and len(provider.calls) == 4
        analyst_call, skeptic_call = provider.calls[0], provider.calls[3]
        assert "Analyst" in analyst_call[1] and "Skeptic" in skeptic_call[1]
        assert {t.name for t in analyst_call[3]} == {"get_profile", "get_layout", "run_sql", "submit_contract"}
        assert {t.name for t in skeptic_call[3]} == {"get_profile", "get_layout", "run_sql", "submit_review"}
        assert len(skeptic_call[2]) == 1  # the Skeptic starts from its own single prompt

        sql_events = [e for e in events if e.kind == "tool_end" and e.data["tool"] == "run_sql"]
        assert sql_events and "churned" in sql_events[0].text
    finally:
        team.close()


async def test_stats_only_hides_sql_and_raw_values_from_the_model(tmp_path: Path) -> None:
    streams = [
        tool_turn("a1", "get_profile", {"table": "loans"}),
        tool_turn("a2", "submit_contract", {"target": "Loan_Status", "task": "classification",
                                            "metric": "roc_auc", "grain": "one row = one loan"}),
        tool_turn("s1", "submit_review", {"verdict": "approve", "summary": "Looks sound."}),
    ]
    team, events, provider = make_team(tmp_path, streams, stats_only=True)
    try:
        await team.prepare([str(samples.make("loans", tmp_path / "data"))])
        await team.consult()

        assert provider is not None
        for _model, system, messages, tools in provider.calls:
            assert "run_sql" not in {t.name for t in tools}
            assert "Raw rows are not available" in system
            sent = json.dumps([m.model_dump(mode="json") for m in messages])
            for raw in ("LP01000", "Semiurban", "Female"):
                assert raw not in sent
        assert team.review is not None and team.review.verdict == "approve"
    finally:
        team.close()


async def test_agent_failure_falls_back_to_the_rule_based_draft(tmp_path: Path) -> None:
    failed = AssistantMessage(content=[], stop_reason="error", error_message="rate limited")
    streams = [
        [AssistantErrorEvent(reason="error", error=failed)],  # Analyst dies
        text_turn("I have no concerns."),                      # Skeptic never submits
    ]
    team, events, _ = make_team(tmp_path, streams)
    try:
        draft = await team.prepare([str(samples.make("titanic", tmp_path / "data"))])
        await team.consult()

        assert team.status["analyst"] == "failed" and team.status["skeptic"] == "failed"
        assert team.contract is draft and team.review is None
        assert team.status["gate"] == "waiting"
        assert any(e.kind == "error" and "rate limited" in e.text for e in events)
        team.approve(draft)
    finally:
        team.close()


async def test_approval_is_refused_while_a_check_fails(tmp_path: Path) -> None:
    team, _, _ = make_team(tmp_path, None)
    try:
        draft = await team.prepare([str(samples.make("loans", tmp_path / "data"))])
        draft.target = None
        with pytest.raises(ValueError, match="No target chosen"):
            team.approve(draft)
        assert not team.run.contract_latest.exists()
    finally:
        team.close()


async def test_bad_source_marks_the_stage_failed(tmp_path: Path) -> None:
    team, events, _ = make_team(tmp_path, None)
    try:
        with pytest.raises(Exception, match="Not found"):
            await team.prepare([str(tmp_path / "missing.csv")])
        assert team.status["intake"] == "failed"
    finally:
        team.close()
