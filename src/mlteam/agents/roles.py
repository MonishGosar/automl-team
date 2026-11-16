"""The two roles in this stage, and the loop that runs one of them.

Each role is its own Tau ``AgentHarness``: separate system prompt, separate
tools, separate message history. They share nothing but the artefacts the
harness hands over.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from tau_agent import (
    AgentHarness,
    AgentHarnessConfig,
    AgentTool,
    AssistantMessage,
    MessageEndEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
)

from mlteam.agents.brain import Brain
from mlteam.agents.tools import (
    TeamContext,
    data_tools,
    feature_list_text,
    query_features_tool,
    submit_contract_tool,
    submit_features_tool,
    submit_review_tool,
    submit_training_plan_tool,
    training_text,
    try_feature_tool,
)
from mlteam.contract import DataContract, Issue

ANALYST_SYSTEM = """\
You are the Analyst on a small machine-learning team. At this stage your one job is to agree a \
data contract for a tabular prediction problem, before any modelling happens.

You are given a draft contract produced by fixed rules. Check it against the data and fix what is wrong.

How to work:
- Decide from tool results, never from column names alone. Look before you claim.
- Check in this order: (1) which table holds the thing we predict about and what one row means, \
(2) that the row key really is unique, (3) the target and whether this is classification or \
regression, (4) each join and which side repeats, (5) columns that give the answer away because \
they were created after or because of the outcome, (6) columns to keep out: identifiers, \
constants, mostly empty columns.
- A table with many rows per key is never a lookup. Mark it kind "aggregate".
- A split by time trains on earlier rows and tests on later rows, never the other way round. \
Say it that way.
- Some decisions depend on a fact the data cannot give you. The most common is the moment the \
prediction is made: whether a column or a joined table is a fair clue or a leak depends on it. \
When that happens, do not guess and do not quietly remove data. Leave the draft's choice in \
place, name the risk in leakage_suspects, and ask the person in open_questions.
- The person reading your rationale is new to machine learning. Write one or two plain sentences \
per decision. Explain any term you cannot avoid.
- Finish by calling submit_contract exactly once. If it is rejected, fix the listed problems and \
submit again.
- Keep messages short. The contract is the output.
"""

SKEPTIC_SYSTEM = """\
You are the Skeptic on a small machine-learning team. You review the Analyst's data contract \
once, before a person approves it. Your job is to find what would make a model look better than \
it really is, or be unfair or useless in practice.

Look for:
- Leakage: columns that would not be known at the moment of prediction, or that are derived from the target.
- Wrong grain: the same real-world thing appearing in more than one row.
- Joins that multiply rows, or lookups where few keys match.
- The wrong task, or a metric that hides failure (accuracy on a rare outcome, for example).
- Columns describing protected attributes such as sex, age, ethnicity or religion. Do not remove \
them. Flag them so the person can decide.
- A target with many missing values or one dominant class.

How to work:
- Back every concern with a number you got from a tool. Only raise what you checked.
- If the contract is sound, say so. Inventing concerns wastes the person's time.
- Write for someone new to machine learning: plain sentences, no unexplained terms.
- Finish by calling submit_review exactly once.
"""

ENGINEER_SYSTEM = """\
You are the Engineer on a small machine-learning team. The data contract is approved. Your job \
now is to improve the list of features: the clues the model will be allowed to see.

What a feature is here:
- One DuckDB expression evaluated for one row of the base table, which is aliased b.
- It may look into another table with a correlated subquery, for example \
(SELECT count(*) FROM orders o WHERE o.customer_id = b.customer_id).
- It must return a number, a true/false value or a short category. Not a date, not free text.

Rules the harness enforces, so do not fight them:
- A feature may never use the target column.
- If the contract sets a prediction moment, tables dated after it are cut off for you: inside a \
feature you only ever see rows that exist at that moment. Do not name a schema to get round it.
- Your exploratory run_sql queries are not cut off. Do not conclude from rows a feature cannot see.

How to work:
- Start from the rule-based list you are given. Keep what is useful, remove what is redundant.
- Add features that carry meaning: ratios, counts, recency, flags, simple combinations. A few \
good ones beat many weak ones. Add at most 12.
- Do not encode categories, scale numbers or fill in missing values. The training stage does \
that, fitted on training rows only, so that nothing leaks from the test rows.
- Do not add "is missing" flags for numbers: training already adds one next to every number \
that has missing values.
- Use try_feature on each idea before you submit it.
- There is no stored feature table to query with run_sql. To look at the features together, use \
query_features, which gives you a table called features.
- The person reading is new to machine learning. For each feature, write one plain sentence \
saying why it could help.
- Finish by calling submit_features exactly once. If it is rejected, fix the listed features and \
submit again.
"""

FEATURE_SKEPTIC_SYSTEM = """\
You are the Skeptic on a small machine-learning team. You review the feature list once, before \
a person approves it.

Look for:
- Leakage: a feature that would not be known at the prediction moment, or that restates the target.
- Features that are almost copies of each other.
- Features that are missing on most rows, or nearly constant.
- Features that stand in for a protected attribute such as sex, age, ethnicity or religion. Do \
not remove them. Flag them so the person can decide.
- An expression that does something other than what its description claims.

How to work:
- Back every concern with a number you got from a tool. Only raise what you checked.
- The list you are given already shows the harness's result for every feature. Do not re-run \
them one by one with try_feature.
- There is no stored feature table to query with run_sql. To compare features with each other or \
with the target, use query_features, which gives you a table called features.
- If the list is sound, say so.
- Write for someone new to machine learning.
- Finish by calling submit_review exactly once. Use the feature name as the field.
"""

TRAINER_SYSTEM = """\
You are the Engineer on a small machine-learning team. The contract and the features are \
approved. Your job now is to choose which models to try.

How it works here:
- You do not train or score anything. You submit a plan: model families and settings from a \
fixed menu. The harness builds each one, fits it on the training rows and scores it on rows it \
has never seen. Every candidate gets the same rows and the same metric.
- The harness always adds a baseline that does nothing clever. A model that cannot beat it \
cannot be approved.
- Filling gaps, scaling and encoding are done by the harness, fitted on training rows only.

How to choose:
- Match the model to the amount of data. With a few hundred rows, simple models and shallow \
trees usually beat large ones, which memorise.
- Always include one model a person can read, such as a small tree.
- Vary settings on purpose, for example a shallow and a deeper forest, rather than repeating \
near-copies. Propose at most 6.
- The person reading is new to machine learning. For each candidate write one plain sentence \
saying why it is worth trying.
- Finish by calling submit_training_plan exactly once. If it is rejected, fix it and submit again.
"""

MODEL_SKEPTIC_SYSTEM = """\
You are the Skeptic on a small machine-learning team. The harness has trained and scored every \
candidate. You review the results once, before a person picks a model.

Look for:
- Scores that are too good to be true, which usually means a feature gives the answer away.
- A large gap between the score on training rows and on unseen rows: the model is memorising.
- Differences between models that are too small to trust, given how few test rows there are.
- A model that leans almost entirely on one feature, or on one describing a protected \
attribute such as sex or age. Flag it; the person decides.
- A recommended model that is harder to explain than a simpler one scoring about the same.

How to work:
- Use the numbers in the results and, where needed, query_features. Only raise what you checked.
- Do not ask for more training. Say which model you would pick and why, in plain words.
- Finish by calling submit_review exactly once. Use the model name as the field.
"""

STATS_ONLY_NOTE = (
    "\nRaw rows are not available in this session. You can see counts, types and statistics only."
)


@dataclass
class TeamEvent:
    """What the harness tells a frontend. Frontends render these; they never
    reach into the agents."""

    kind: str  # stage | agent_text | tool_start | tool_end | note | error
    stage: str = ""
    agent: str = ""
    text: str = ""
    data: dict[str, Any] = field(default_factory=dict)


Emit = Callable[[TeamEvent], None]


def _issues_text(issues: list[Issue]) -> str:
    if not issues:
        return "(none)"
    return "\n".join(f"- {i.level}: [{i.field}] {i.message}" for i in issues)


def _tables_text(ctx: TeamContext) -> str:
    lines = []
    for table in ctx.profile.tables:
        lines.append(f"- {table.name}: {table.rows:,} rows, {len(table.columns)} columns")
    return "\n".join(lines)


def analyst_prompt(ctx: TeamContext, issues: list[Issue]) -> str:
    return (
        f"Layout detected: {ctx.layout.kind}\n"
        f"Tables:\n{_tables_text(ctx)}\n\n"
        f"Draft contract from the rules:\n{json.dumps(ctx.draft.to_json(), indent=1, default=str)}\n\n"
        f"Harness checks on the draft:\n{_issues_text(issues)}\n\n"
        "Check the draft against the data, then submit the contract."
    )


def skeptic_prompt(contract: DataContract, issues: list[Issue], ctx: TeamContext) -> str:
    return (
        f"Tables:\n{_tables_text(ctx)}\n\n"
        f"Contract to review (proposed by: {contract.proposed_by}):\n"
        f"{json.dumps(contract.to_json(), indent=1, default=str)}\n\n"
        f"Harness checks on it:\n{_issues_text(issues)}\n\n"
        "Review it once, then submit your review."
    )


def engineer_prompt(ctx: TeamContext) -> str:
    assert ctx.contract is not None
    contract = ctx.contract
    answers = "\n".join(f"- {q} -> {a}" for q, a in contract.answers.items()) or "(none)"
    return (
        f"Approved contract (v{contract.version}):\n"
        f"{json.dumps(contract.to_json(), indent=1, default=str)}\n\n"
        f"What the person told us:\n{answers}\n\n"
        f"What understanding the data found (training rows only):\n{ctx.insights_text or '(not run)'}\n\n"
        f"Rule-based feature list, as evaluated by the harness:\n{feature_list_text(ctx)}\n\n"
        "Features marked as suggested off were switched off by those findings; you may disagree and say why. "
        "Improve the list, then submit your changes."
    )


def feature_skeptic_prompt(ctx: TeamContext) -> str:
    assert ctx.contract is not None
    return (
        f"Target: {ctx.contract.target} ({ctx.contract.task}). {ctx.contract.grain}.\n\n"
        f"Feature list to review, as evaluated by the harness:\n{feature_list_text(ctx)}\n\n"
        "Review it once, then submit your review."
    )


def _feature_data_tools(ctx: TeamContext) -> list[AgentTool]:
    tools = data_tools(ctx)
    if not ctx.stats_only:  # query_features returns rows, so it follows the same rule as run_sql
        tools.append(query_features_tool(ctx))
    return tools


def trainer_prompt(ctx: TeamContext, shape: str) -> str:
    assert ctx.contract is not None
    contract = ctx.contract
    plan = "\n".join(f"- {c.name}: {c.family} {c.params or '(defaults)'}" for c in ctx.candidates)
    return (
        f"Target: {contract.target} ({contract.task}). Metric: {contract.metric}. {contract.grain}.\n"
        f"{shape}\n\n"
        f"What understanding the data found:\n{ctx.insights_text or '(not run)'}\n\n"
        f"Approved features:\n{feature_list_text(ctx)}\n\n"
        f"The rule-based plan, which you may replace:\n{plan}\n\n"
        "Choose the models to try, then submit the plan."
    )


def model_skeptic_prompt(ctx: TeamContext) -> str:
    assert ctx.contract is not None and ctx.training is not None
    return (
        f"Target: {ctx.contract.target} ({ctx.contract.task}). {ctx.contract.grain}.\n\n"
        f"Results from the harness:\n{training_text(ctx.training)}\n\n"
        "Review them once, then submit your review."
    )


def trainer_tools(ctx: TeamContext) -> list[AgentTool]:
    return [*_feature_data_tools(ctx), submit_training_plan_tool(ctx)]


def model_skeptic_tools(ctx: TeamContext) -> list[AgentTool]:
    return [*_feature_data_tools(ctx), submit_review_tool(ctx, "model_review")]


def engineer_tools(ctx: TeamContext) -> list[AgentTool]:
    return [*_feature_data_tools(ctx), try_feature_tool(ctx), submit_features_tool(ctx)]


def feature_skeptic_tools(ctx: TeamContext) -> list[AgentTool]:
    return [*_feature_data_tools(ctx), try_feature_tool(ctx), submit_review_tool(ctx, "feature_review")]


def analyst_tools(ctx: TeamContext) -> list[AgentTool]:
    return [*data_tools(ctx), submit_contract_tool(ctx)]


def skeptic_tools(ctx: TeamContext) -> list[AgentTool]:
    return [*data_tools(ctx), submit_review_tool(ctx)]


def _short(text: str, limit: int = 240) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."


async def run_role(
    *,
    name: str,
    stage: str,
    brain: Brain,
    system: str,
    tools: list[AgentTool],
    prompt: str,
    emit: Emit,
    max_turns: int = 14,
    submitted: Callable[[], bool] | None = None,
) -> str | None:
    """Run one role to completion. Returns an error message, or None on success.

    Models sometimes investigate and then stop without calling their submit tool. When
    `submitted` says nothing was handed in, the role is reminded once, in the same
    conversation, to submit what it has found.
    """
    harness = AgentHarness(
        AgentHarnessConfig(
            provider=brain.provider,
            model=brain.model,
            system=system,
            tools=tools,
            max_turns=max_turns,
        )
    )
    error = await _drain(harness.prompt(prompt), name, stage, emit)
    if error is None and submitted is not None and not submitted():
        emit(TeamEvent("agent_text", stage=stage, agent="harness",
                       text=f"{name} stopped without submitting; reminding it once."))
        error = await _drain(harness.prompt(NUDGE), name, stage, emit)
    return error


NUDGE = ("You stopped without calling your submit tool, so nothing was handed in. Do not run more "
         "queries. Call the submit tool now with what you have already found.")


async def _drain(events, name: str, stage: str, emit: Emit) -> str | None:
    error: str | None = None
    async for event in events:
        if isinstance(event, MessageEndEvent) and isinstance(event.message, AssistantMessage):
            message = event.message
            if message.text.strip():
                emit(TeamEvent("agent_text", stage=stage, agent=name, text=message.text.strip()))
            if message.stop_reason == "error":
                error = message.error_message or "The model returned an error."
                emit(TeamEvent("error", stage=stage, agent=name, text=error))
        elif isinstance(event, ToolExecutionStartEvent):
            args = event.args.get("sql") if event.tool_name == "run_sql" else None
            detail = _short(str(args)) if args else ""
            emit(TeamEvent("tool_start", stage=stage, agent=name, text=detail,
                           data={"tool": event.tool_name}))
        elif isinstance(event, ToolExecutionEndEvent):
            emit(TeamEvent("tool_end", stage=stage, agent=name, text=_short(event.result.text),
                           data={"tool": event.tool_name, "is_error": event.is_error}))
    return error
