"""Tools the agents can call.

Agents see the data only through these. ``run_sql`` goes through the sealed,
read-only store; in stats-only mode it is not offered at all, so no raw row
can reach the model provider.
"""

from __future__ import annotations

import asyncio
import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from tau_agent import AgentTool, AgentToolResult
from tau_agent.types import JSONValue

from mlteam.contract import (
    METRICS,
    QUESTIONS_FIELD,
    DataContract,
    JoinStep,
    Suspect,
    propose_contract,
    validate_contract,
)
from mlteam.contract import attach_time_columns, target_stats
from mlteam.features import (
    MAX_FEATURES,
    Feature,
    evaluate_feature,
    evaluate_features,
    features_query_sql,
    time_rule_summary,
)
from mlteam.profile import DatasetProfile
from mlteam.train import FAMILIES, Candidate, TrainingRun, check_candidate, default_plan
from mlteam.train import PLAN_LIMIT as MAX_CANDIDATES
from mlteam.relations import Layout
from mlteam.store import Store, StoreError

MAX_CELL = 60


@dataclass
class Concern:
    severity: str
    field: str
    message: str
    evidence: str = ""


@dataclass
class Review:
    verdict: str
    summary: str
    concerns: list[Concern] = field(default_factory=list)


@dataclass
class TeamContext:
    store: Store
    profile: DatasetProfile
    layout: Layout
    draft: DataContract
    dataset_hash: str
    stats_only: bool = False
    proposed: DataContract | None = None
    review: Review | None = None
    # Feature stage
    contract: DataContract | None = None  # the approved contract
    features: list[Feature] = field(default_factory=list)
    engineered: bool = False  # True once the Engineer has submitted
    feature_review: Review | None = None
    insights_text: str = ""  # what understanding the data found; shown to the Engineer
    # Training stage
    candidates: list[Candidate] = field(default_factory=list)
    planned: bool = False  # True once the Engineer has submitted a plan
    training: TrainingRun | None = None
    model_review: Review | None = None


class ToolError(Exception):
    """Raised inside a tool; Tau turns it into an error result the agent can read."""


def _result(text: str, *, terminate: bool | None = None, details: JSONValue = None) -> AgentToolResult:
    return AgentToolResult(content=text, details=details, terminate=terminate)


def _str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(v) for v in value if isinstance(v, (str, int, float))]


def format_rows(columns: list[str], rows: list[tuple[Any, ...]], truncated: bool) -> str:
    def cell(value: Any) -> str:
        text = "NULL" if value is None else str(value)
        return text if len(text) <= MAX_CELL else text[: MAX_CELL - 3] + "..."

    lines = [" | ".join(columns)]
    lines.extend(" | ".join(cell(v) for v in row) for row in rows)
    if not rows:
        lines.append("(no rows)")
    if truncated:
        lines.append(f"(only the first {len(rows)} rows are shown; add LIMIT or aggregate)")
    return "\n".join(lines)


def profile_tool(ctx: TeamContext) -> AgentTool:
    async def execute(_id: str, args: Mapping[str, JSONValue], _signal=None, _update=None) -> AgentToolResult:
        data = ctx.profile.to_json(stats_only=ctx.stats_only)
        wanted = args.get("table")
        if wanted:
            tables = [t for t in data["tables"] if t["name"] == wanted]
            if not tables:
                names = ", ".join(t["name"] for t in data["tables"])
                raise ToolError(f"No table named {wanted}. Tables: {names}")
            data = {"tables": tables}
        for table in data["tables"]:
            for col in table["columns"]:
                col.pop("rows", None)
                col.pop("avg_length", None)
                if ctx.stats_only:
                    col.pop("top", None)
                    col.pop("samples", None)
        return _result(json.dumps(data, default=str))

    return AgentTool(
        name="get_profile",
        label="Profile",
        description=(
            "Column-level profile of the loaded tables: type, kind, missing share, distinct count, "
            "uniqueness, range and most common values. Pass a table name to limit the output."
        ),
        parameters={
            "type": "object",
            "properties": {"table": {"type": "string", "description": "Optional table name."}},
        },
        execute_fn=execute,
    )


def layout_tool(ctx: TeamContext) -> AgentTool:
    async def execute(_id: str, _args: Mapping[str, JSONValue], _signal=None, _update=None) -> AgentToolResult:
        return _result(json.dumps(ctx.layout.to_json(), default=str))

    return AgentTool(
        name="get_layout",
        label="Layout",
        description=(
            "How the tables relate: train/test roles, files that are shards of one table, and key "
            "relations with their cardinality and the share of keys that match."
        ),
        parameters={"type": "object", "properties": {}},
        execute_fn=execute,
    )


def sql_tool(ctx: TeamContext) -> AgentTool:
    async def execute(_id: str, args: Mapping[str, JSONValue], _signal=None, _update=None) -> AgentToolResult:
        sql = args.get("sql")
        if not isinstance(sql, str) or not sql.strip():
            raise ToolError("Pass the query as the 'sql' string.")
        try:
            columns, rows, truncated = await asyncio.to_thread(ctx.store.agent_query, sql)
        except StoreError as exc:
            message = str(exc)
            if ctx.features and "features" in message.lower() and "does not exist" in message:
                message += ("\nThe features table is not stored. It exists only inside the "
                            "query_features tool; run this query there.")
            raise ToolError(message) from exc
        return _result(format_rows(columns, rows, truncated))

    return AgentTool(
        name="run_sql",
        label="SQL",
        description=(
            "Run one read-only DuckDB query (SELECT / WITH / DESCRIBE / SUMMARIZE) against the loaded "
            "tables. At most 50 rows come back, so aggregate. Use it to check every claim you make."
        ),
        parameters={
            "type": "object",
            "properties": {"sql": {"type": "string", "description": "A single read-only DuckDB statement."}},
            "required": ["sql"],
        },
        execute_fn=execute,
    )


def _build_contract(ctx: TeamContext, args: Mapping[str, JSONValue]) -> DataContract:
    base = args.get("base_table")
    if isinstance(base, str) and base != ctx.draft.base_table:
        if ctx.profile.table(base) is None:
            raise ToolError(f"No table named {base}.")
        contract = propose_contract(
            ctx.store, ctx.profile, ctx.layout, ctx.dataset_hash, base_override=base
        )
    else:
        contract = copy.deepcopy(ctx.draft)
    contract.proposed_by = "analyst"

    for key in ("target", "task", "metric", "grain", "time_column"):
        if key in args:
            value = args[key]
            setattr(contract, key, value if isinstance(value, str) and value else None)
    contract.grain = contract.grain or ""
    if isinstance(args.get("dedupe"), bool):
        contract.dedupe = bool(args["dedupe"])
    if "group_column" in args:
        value = args["group_column"]
        contract.group_column = value if isinstance(value, str) and value else None
    if "fairness_columns" in args:
        contract.fairness_columns = _str_list(args["fairness_columns"])
    if "grain_keys" in args:
        contract.grain_keys = _str_list(args["grain_keys"])
    if "drop_columns" in args:
        contract.drop_columns = _str_list(args["drop_columns"])
    if "joins" in args:
        known = {(r.right_table, r.left_column, r.right_column): r.left_in_right
                 for r in ctx.layout.relations_for(contract.base_table)}
        joins: list[JoinStep] = []
        for item in args["joins"] if isinstance(args["joins"], list) else []:
            if not isinstance(item, dict):
                continue
            try:
                table, kind = str(item["table"]), str(item["kind"])
                base_key, other_key = str(item["base_key"]), str(item["other_key"])
            except KeyError as exc:
                raise ToolError(f"Each join needs table, kind, base_key, other_key. Missing: {exc}") from exc
            if kind not in {"lookup", "aggregate"}:
                raise ToolError(f"Join kind must be 'lookup' or 'aggregate', got {kind}.")
            joins.append(JoinStep(table, kind, base_key, other_key, known.get((table, base_key, other_key), 0.0)))
        contract.joins = attach_time_columns(joins, ctx.profile)
    if "leakage_suspects" in args and isinstance(args["leakage_suspects"], list):
        contract.leakage_suspects = [
            Suspect(str(s.get("column", "")), str(s.get("reason", "")))
            for s in args["leakage_suspects"] if isinstance(s, dict) and s.get("column")
        ]
    if "open_questions" in args:
        contract.open_questions = [q.strip() for q in _str_list(args["open_questions"]) if q.strip()]
    contract.answers = {}  # only a person answers
    rationale = args.get("rationale")
    if isinstance(rationale, dict):
        contract.rationale = {**contract.rationale, **{str(k): str(v) for k, v in rationale.items()}}
    return contract


def submit_contract_tool(ctx: TeamContext) -> AgentTool:
    async def execute(_id: str, args: Mapping[str, JSONValue], _signal=None, _update=None) -> AgentToolResult:
        contract = await asyncio.to_thread(_build_contract, ctx, args)
        issues = await asyncio.to_thread(validate_contract, ctx.store, ctx.profile, contract)
        # An agent cannot answer its own questions, so those do not count against it here.
        blocking = [i for i in issues if i.level == "error" and i.field != QUESTIONS_FIELD]
        if blocking:
            problems = "\n".join(f"- [{i.field}] {i.message}" for i in blocking)
            raise ToolError("The harness rejected this contract. Fix these and submit again:\n" + problems)
        ctx.proposed = contract
        warnings = "\n".join(f"- [{i.field}] {i.message}" for i in issues if i.level == "warning")
        text = "Contract accepted and passed to the Skeptic and the human gate."
        if warnings:
            text += "\nWarnings the person will see:\n" + warnings
        return _result(text, terminate=True)

    metrics = sorted({m for group in METRICS.values() for m in group})
    return AgentTool(
        name="submit_contract",
        label="Submit contract",
        description=(
            "Submit the data contract. Fields you leave out keep the draft's value. The harness "
            "re-checks everything (key uniqueness, row counts after joins) and rejects it if a check fails."
        ),
        parameters={
            "type": "object",
            "properties": {
                "base_table": {"type": "string", "description": "Table whose rows are the things we predict about."},
                "target": {"type": "string", "description": "Column to predict."},
                "task": {"type": "string", "enum": ["classification", "regression"]},
                "metric": {"type": "string", "enum": metrics},
                "grain": {"type": "string", "description": "Plain sentence: 'one row = one <thing>'."},
                "dedupe": {"type": "boolean",
                           "description": "Drop rows that are exact copies of another row before anything else."},
                "group_column": {"type": "string",
                                 "description": "Column whose rows belong together (e.g. customer_id when one "
                                                "customer has many rows). Kept on one side of every split. "
                                                "Empty for none."},
                "fairness_columns": {"type": "array", "items": {"type": "string"},
                                     "description": "Columns to check the model's fairness on, beyond age, sex "
                                                    "and similar names, which are checked anyway."},
                "grain_keys": {"type": "array", "items": {"type": "string"},
                               "description": "Columns that together identify one row."},
                "time_column": {"type": "string", "description": "Date column to split on, or empty."},
                "drop_columns": {"type": "array", "items": {"type": "string"},
                                 "description": "Base-table columns kept out of the model."},
                "joins": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "table": {"type": "string"},
                            "kind": {"type": "string", "enum": ["lookup", "aggregate"]},
                            "base_key": {"type": "string"},
                            "other_key": {"type": "string"},
                        },
                        "required": ["table", "kind", "base_key", "other_key"],
                    },
                },
                "leakage_suspects": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"column": {"type": "string"}, "reason": {"type": "string"}},
                        "required": ["column", "reason"],
                    },
                },
                "rationale": {
                    "type": "object",
                    "description": "One or two plain sentences per decision. Keys: base, target, task, "
                                   "metric, grain, time, joins, drops.",
                },
                "open_questions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Questions only the person can answer because the data does not say, "
                                   "for example the moment at which the prediction is made. One plain "
                                   "sentence each. Ask instead of guessing.",
                },
            },
            "required": ["target", "task", "metric", "grain"],
        },
        execute_fn=execute,
        execution_mode="sequential",
    )


def feature_report(feature: Feature) -> str:
    if feature.error:
        return f"{feature.name}: REJECTED. {feature.error}"
    line = f"{feature.name}: ok. {feature.kind}, {feature.distinct:,} distinct values, {feature.null_pct:.0f}% missing."
    if feature.suggested_off:
        line += f" Suggested off: {feature.suggested_off}"
    if feature.leak:
        line += f" LEAK SUSPECT: {feature.leak}"
    if feature.warning:
        line += f" Warning: {feature.warning}"
    return line


def try_feature_tool(ctx: TeamContext) -> AgentTool:
    async def execute(_id: str, args: Mapping[str, JSONValue], _signal=None, _update=None) -> AgentToolResult:
        assert ctx.contract is not None
        name, sql = args.get("name"), args.get("sql")
        if not isinstance(name, str) or not isinstance(sql, str):
            raise ToolError("Pass 'name' and 'sql' as strings.")
        feature = Feature(name=name, sql=sql, origin="engineer")
        stats = await asyncio.to_thread(target_stats, ctx.store, ctx.contract)
        await asyncio.to_thread(evaluate_feature, ctx.store, ctx.contract, feature, set(), stats)
        return _result(feature_report(feature))

    return AgentTool(
        name="try_feature",
        label="Try feature",
        description=(
            "Evaluate one candidate feature exactly as the harness will: under the time rule, with the "
            "leak scan. Returns its type, distinct count, missing share, and any rejection or leak warning. "
            "Nothing is saved."
        ),
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "letters, digits, underscores"},
                "sql": {"type": "string", "description": "One DuckDB expression for one base row, aliased b."},
            },
            "required": ["name", "sql"],
        },
        execute_fn=execute,
    )


def query_features_tool(ctx: TeamContext) -> AgentTool:
    async def execute(_id: str, args: Mapping[str, JSONValue], _signal=None, _update=None) -> AgentToolResult:
        assert ctx.contract is not None
        sql = args.get("sql")
        if not isinstance(sql, str):
            raise ToolError("Pass the query as the 'sql' string.")
        usable = [f for f in ctx.features if f.usable]
        try:
            full = features_query_sql(ctx.contract, usable, sql)
            columns, rows, truncated = await asyncio.to_thread(ctx.store.agent_query, full)
        except (ValueError, StoreError) as exc:
            raise ToolError(str(exc)) from exc
        return _result(format_rows(columns, rows, truncated))

    return AgentTool(
        name="query_features",
        label="Query features",
        description=(
            "Run one read-only SELECT against a table called features: one row per base row, with the "
            "row key, the target and every usable feature in the current list as columns. The time rule "
            "applies to the whole query, so dated tables you join here are also cut at the prediction "
            "moment. Use it to compare features with each other and with the target. At most 50 rows "
            "come back, so aggregate."
        ),
        parameters={
            "type": "object",
            "properties": {"sql": {"type": "string", "description": "A single SELECT. Example: "
                                   "SELECT corr(orders__count, orders__amount__sum) FROM features"}},
            "required": ["sql"],
        },
        execute_fn=execute,
    )


def submit_features_tool(ctx: TeamContext) -> AgentTool:
    async def execute(_id: str, args: Mapping[str, JSONValue], _signal=None, _update=None) -> AgentToolResult:
        assert ctx.contract is not None
        remove = {n.lower() for n in _str_list(args.get("remove"))}
        unknown = remove - {f.name.lower() for f in ctx.features}
        if unknown:
            raise ToolError("Cannot remove features that do not exist: " + ", ".join(sorted(unknown)))
        kept = [copy.deepcopy(f) for f in ctx.features if f.name.lower() not in remove]
        added: list[Feature] = []
        for item in args.get("add") or []:
            if not isinstance(item, dict) or not item.get("name") or not item.get("sql"):
                raise ToolError("Each added feature needs 'name', 'sql' and 'why'.")
            added.append(Feature(name=str(item["name"]), sql=str(item["sql"]),
                                 why=str(item.get("why", "")), origin="engineer"))
        if len(kept) + len(added) > MAX_FEATURES:
            raise ToolError(f"Too many features. The limit is {MAX_FEATURES}.")
        candidate = kept + added
        await asyncio.to_thread(evaluate_features, ctx.store, ctx.contract, candidate)
        broken = [f for f in added if f.error]
        if broken:
            raise ToolError(
                "The harness rejected these features. Fix or drop them, then submit again:\n"
                + "\n".join(f"- {feature_report(f)}" for f in broken)
            )
        ctx.features = candidate
        ctx.engineered = True
        flagged = [f for f in added if f.leak]
        text = f"Accepted: {len(candidate)} features ({len(added)} new, {len(remove)} removed)."
        if flagged:
            text += " Flagged as possible leaks, off by default for the person: " + ", ".join(f.name for f in flagged)
        return _result(text, terminate=True)

    return AgentTool(
        name="submit_features",
        label="Submit features",
        description=(
            "Submit your changes to the feature list: features to add and rule-based features to remove. "
            "The harness evaluates every one and rejects the submission if a feature you added fails."
        ),
        parameters={
            "type": "object",
            "properties": {
                "add": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "sql": {"type": "string",
                                    "description": "One DuckDB expression for one base row, aliased b."},
                            "why": {"type": "string",
                                    "description": "One plain sentence: why this could help predict the target."},
                        },
                        "required": ["name", "sql", "why"],
                    },
                },
                "remove": {"type": "array", "items": {"type": "string"},
                           "description": "Names of existing features to drop, for example redundant ones."},
            },
        },
        execute_fn=execute,
        execution_mode="sequential",
    )


def submit_training_plan_tool(ctx: TeamContext) -> AgentTool:
    async def execute(_id: str, args: Mapping[str, JSONValue], _signal=None, _update=None) -> AgentToolResult:
        items = args.get("candidates")
        if not isinstance(items, list) or not items:
            raise ToolError("Pass 'candidates' as a non-empty list.")
        plan = [default_plan()[0]]  # the baseline is always trained; the harness adds it
        taken = {plan[0].name.lower()}
        problems: list[str] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            params = item.get("params") if isinstance(item.get("params"), dict) else {}
            candidate = Candidate(
                name=str(item.get("name", "")), family=str(item.get("family", "")),
                params={str(k): v for k, v in params.items()}, why=str(item.get("why", "")), origin="engineer",
            )
            if candidate.family == "baseline":
                continue
            error = check_candidate(candidate, taken)
            if error:
                problems.append(f"- {candidate.name or '(no name)'}: {error}")
            else:
                taken.add(candidate.name.lower())
                plan.append(candidate)
        if problems:
            raise ToolError("The harness rejected these candidates. Fix them and submit again:\n" + "\n".join(problems))
        if len(plan) < 2:
            raise ToolError("Propose at least one model besides the baseline.")
        if len(plan) > MAX_CANDIDATES:
            raise ToolError(f"Too many candidates. The limit is {MAX_CANDIDATES - 1} plus the baseline.")
        ctx.candidates = plan
        ctx.planned = True
        return _result(f"Plan accepted: {len(plan) - 1} candidates plus the baseline. "
                       "The harness will train and score them; you do not.", terminate=True)

    menu = {family: {k: {"min": low, "max": high, "default": default} for k, (low, high, default) in params.items()}
            for family, params in FAMILIES.items() if family != "baseline"}
    return AgentTool(
        name="submit_training_plan",
        label="Submit training plan",
        description=(
            "Submit the models to try. You choose a family and its settings from a fixed menu; the "
            "harness builds, trains and scores every one itself on the same unseen rows. The baseline "
            "is always added. Menu: " + json.dumps(menu)
        ),
        parameters={
            "type": "object",
            "properties": {
                "candidates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "letters, digits, underscores"},
                            "family": {"type": "string", "enum": [f for f in FAMILIES if f != "baseline"]},
                            "params": {"type": "object", "description": "Settings from the menu. Omit for defaults."},
                            "why": {"type": "string", "description": "One plain sentence: why this model, with these settings."},
                        },
                        "required": ["name", "family", "why"],
                    },
                },
            },
            "required": ["candidates"],
        },
        execute_fn=execute,
        execution_mode="sequential",
    )


def training_text(run: TrainingRun) -> str:
    """The scoreboard as plain text, for an agent to review."""
    lines = [
        f"Task: {run.task}. Metric: {run.metric} ({'higher' if run.higher_is_better else 'lower'} is better).",
        f"{run.split} Rows without a target, left out: {run.dropped_rows}.",
    ]
    if run.positive_class:
        lines.append(f"Positive class: {run.positive_class}.")
    lines += [f"Harness warning: {w}" for w in run.warnings]
    for r in run.results:
        if r.error:
            lines.append(f"- {r.name} ({r.family}): FAILED. {r.error}")
            continue
        extra = ", ".join(f"{k}={v:.3f}" for k, v in r.extra.items())
        line = (f"- {r.name} ({r.family}, {r.params}): test {run.metric}={r.test_score:.4f}, "
                f"train {run.metric}={r.train_score:.4f}; {extra}.")
        if r.cv_mean is not None:
            line += f" Cross-validated on training rows: {r.cv_mean:.4f} ± {r.cv_std:.4f}."
        if r.family != "baseline":
            line += " Beats baseline." if r.beats_baseline else " Does NOT beat the baseline."
        if r.recommended:
            line += " Best on the metric."
        if r.importance:
            top = ", ".join(f"{n} ({v:+.3f})" for n, v in r.importance[:5])
            line += f" Score lost when a feature is shuffled: {top}."
        lines.append(line)
        lines += [f"    harness note: {n}" for n in r.notes]
    return "\n".join(lines)


def feature_list_text(ctx: TeamContext) -> str:
    lines = [time_rule_summary(ctx.contract)] if ctx.contract else []
    for f in ctx.features:
        lines.append(f"- {feature_report(f)}  SQL: {f.sql}")
    return "\n".join(lines)


def submit_review_tool(ctx: TeamContext, attribute: str = "review") -> AgentTool:
    async def execute(_id: str, args: Mapping[str, JSONValue], _signal=None, _update=None) -> AgentToolResult:
        verdict = args.get("verdict")
        if verdict not in {"approve", "concerns"}:
            raise ToolError("verdict must be 'approve' or 'concerns'.")
        concerns: list[Concern] = []
        for item in args.get("concerns") or []:
            if isinstance(item, dict) and item.get("message"):
                severity = str(item.get("severity", "medium"))
                concerns.append(Concern(
                    severity=severity if severity in {"high", "medium", "low"} else "medium",
                    field=str(item.get("field", "")),
                    message=str(item["message"]),
                    evidence=str(item.get("evidence", "")),
                ))
        if verdict == "concerns" and not concerns:
            raise ToolError("verdict 'concerns' needs at least one concern.")
        setattr(ctx, attribute,
                Review(verdict=str(verdict), summary=str(args.get("summary", "")), concerns=concerns))
        return _result("Review recorded. The person decides next.", terminate=True)

    return AgentTool(
        name="submit_review",
        label="Submit review",
        description="Submit your one review. The person sees it at the gate.",
        parameters={
            "type": "object",
            "properties": {
                "verdict": {"type": "string", "enum": ["approve", "concerns"]},
                "summary": {"type": "string", "description": "Two plain sentences at most."},
                "concerns": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "severity": {"type": "string", "enum": ["high", "medium", "low"]},
                            "field": {"type": "string",
                                      "description": "target, grain, joins, drops, task, metric or a column name."},
                            "message": {"type": "string"},
                            "evidence": {"type": "string", "description": "The number or query result behind it."},
                        },
                        "required": ["severity", "message"],
                    },
                },
            },
            "required": ["verdict", "summary"],
        },
        execute_fn=execute,
        execution_mode="sequential",
    )


def data_tools(ctx: TeamContext) -> list[AgentTool]:
    tools = [profile_tool(ctx), layout_tool(ctx)]
    if not ctx.stats_only:
        tools.append(sql_tool(ctx))
    return tools
