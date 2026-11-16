"""Turn harness data into terminal text.

Pure functions: data in, ``rich.text.Text`` out. No widgets, so they can be
tested without a terminal and reused by a different frontend's text mode.
"""

from __future__ import annotations

from typing import Any

from rich.text import Text

from mlteam.agents.tools import Review
from mlteam.contract import DataContract, Issue
from mlteam.feature_tests import FeatureTest, FeatureTestRun
from mlteam.challenger import Challenge
from mlteam.features import Feature
from mlteam.insights import DataInsights
from mlteam.monitor import ADVICE, STABLE, WATCH
from mlteam.tune import TuneRun, best_curve
from mlteam.predict import Prediction
from mlteam.profile import ColumnProfile, DatasetProfile, TableProfile
from mlteam.relations import Layout
from mlteam.team import GATES, LOCKED_STAGES, STAGES
from mlteam.evaluate import Evaluation, evaluation_from_json
from mlteam.train import BOUNDED, ENSEMBLE_WORDS, FAMILY_WORDS, METRIC_WORDS, ModelResult, TrainingRun

STATUS_STYLE: dict[str, tuple[str, str]] = {
    "pending": ("○", "dim"),
    "running": ("●", "bold yellow"),
    "done": ("✔", "green"),
    "skipped": ("–", "dim"),
    "failed": ("✘", "bold red"),
    "waiting": ("◆", "bold cyan"),
}
BAR_WIDTH = 8

LAYOUT_WORDS = {
    "single": "one table",
    "train_test": "a train / test split",
    "shards": "one table split across several files",
    "related": "several related tables",
    "unrelated": "several tables with no shared key",
}

RATIONALE_ORDER = (
    ("dedupe", "Duplicates"),
    ("base", "Table"),
    ("target", "Target"),
    ("task", "Task"),
    ("metric", "Metric"),
    ("grain", "One row"),
    ("group", "Groups"),
    ("time", "Time"),
    ("horizon", "Predict at"),
    ("joins", "Joins"),
    ("drops", "Kept out"),
)

LEARN: tuple[tuple[str, str], ...] = (
    ("Target", "The column the model learns to predict. Everything else is a clue."),
    ("One row", "What a single row stands for. If one customer fills two rows, the model can "
                "recognise them instead of learning, and the score looks better than it is."),
    ("Leakage", "A column that already contains the answer, usually because it was filled in "
                "after the outcome happened. The model scores brilliantly, then fails in real use."),
    ("Joins", "Attaching another table. If the other table has many rows per key, rows get "
              "copied and counted twice, so it is summarised to one row per key first."),
    ("Metric", "How we keep score. The right one depends on which mistakes cost more."),
)


def bar(share: float, width: int = BAR_WIDTH) -> str:
    share = max(0.0, min(1.0, share))
    filled = round(share * width)
    if share > 0 and filled == 0:
        filled = 1
    return "█" * filled + "░" * (width - filled)


def rail(status: dict[str, str], notes: dict[str, str] | None = None) -> Text:
    notes = notes or {}
    text = Text()
    text.append("PIPELINE\n", style="bold")
    for key, label in STAGES:
        state = status.get(key, "pending")
        icon, style = STATUS_STYLE.get(state, STATUS_STYLE["pending"])
        text.append(f" {icon} ", style=style)
        text.append(label, style="bold" if state in {"running", "waiting"} else "")
        if key in GATES and state == "waiting":
            text.append("  your call", style="bold cyan")
        elif state in {"skipped", "failed"}:
            text.append(f"  {state}", style=style)
        text.append("\n")
    text.append(" " + "─" * 18 + "\n", style="dim")
    for label in LOCKED_STAGES:
        text.append(f" · {label}\n", style="dim")
    text.append(" not built yet\n", style="dim italic")
    return text


def table_summary(table: TableProfile) -> Text:
    text = Text()
    text.append(table.name, style="bold")
    text.append(f"  {table.rows:,} rows · {len(table.columns)} columns · from {table.file}", style="dim")
    if table.load_note:
        text.append(f"\n! {table.load_note}", style="yellow")
    if table.truncated_columns:
        text.append(f"\n! {table.truncated_columns} more columns were not profiled", style="yellow")
    return text


def column_role(col: ColumnProfile, contract: DataContract | None, table: str) -> tuple[str, str]:
    """(label, style) describing what the contract does with this column."""
    if contract is None or table not in ({contract.base_table} | set(contract.stack)):
        return "", ""
    if col.name == contract.target:
        return "TARGET", "bold green"
    if col.name in contract.grain_keys:
        return "row key", "cyan"
    if any(s.column == col.name for s in contract.leakage_suspects) and col.name in contract.drop_columns:
        return "leak?", "bold red"
    if col.name in contract.drop_columns:
        return "kept out", "dim"
    if col.name == contract.time_column:
        return "time", "magenta"
    if any(j.base_key == col.name for j in contract.joins):
        return "join key", "cyan"
    return "feature", ""


def missing_cell(col: ColumnProfile) -> Text:
    share = col.null_pct / 100.0
    style = "red" if share >= 0.5 else "yellow" if share > 0 else "dim"
    return Text(f"{bar(share)} {col.null_pct:4.0f}%", style=style)


def values_cell(col: ColumnProfile) -> Text:
    if col.kind in {"empty", "constant"}:
        return Text("never changes" if col.kind == "constant" else "all missing", style="dim")
    if col.top and col.kind in {"categorical", "boolean"}:
        nonnull = max(col.rows - col.nulls, 1)
        parts = [f"{value} {count / nonnull:.0%}" for value, count in col.top[:3]]
        extra = col.distinct - min(len(col.top), 3)
        return Text(" · ".join(parts) + (f" · +{extra}" if extra > 0 else ""))
    if col.kind == "numeric" and col.min is not None:
        mean = f"  mean {col.mean:,.2f}" if col.mean is not None else ""
        return Text(f"{col.min} … {col.max}{mean}")
    if col.kind == "datetime" and col.min is not None:
        return Text(f"{col.min} → {col.max}")
    return Text(", ".join(col.samples[:3]), style="dim")


def relations(profile: DatasetProfile, layout: Layout, contract: DataContract | None,
              issues: list[Issue]) -> Text:
    text = Text()
    text.append("What you gave me: ", style="bold")
    text.append(LAYOUT_WORDS.get(layout.kind, layout.kind) + "\n\n")

    if layout.train and layout.test:
        text.append("Split\n", style="bold")
        text.append(f"  {layout.train}", style="green")
        text.append("  is used to learn\n")
        text.append(f"  {layout.test}", style="cyan")
        text.append("  is held back for scoring")
        if layout.train_only_columns:
            text.append(f" (it has no {', '.join(layout.train_only_columns)})")
        text.append("\n")
        if layout.submission:
            text.append(f"  {layout.submission}", style="dim")
            text.append("  shows the answer format\n")
        text.append("\n")

    for group in layout.shard_groups:
        text.append("Same columns, stacked into one table\n", style="bold")
        total = 0
        for name in group:
            table = profile.table(name)
            rows = table.rows if table else 0
            total += rows
            text.append(f"  ├─ {name}", style="green")
            text.append(f"  {rows:,} rows\n", style="dim")
        text.append(f"  └─ total {total:,} rows\n\n")

    if contract is not None:
        base = profile.table(contract.base_table)
        text.append("Join plan\n", style="bold")
        text.append(f"  {contract.base_table}", style="bold green")
        text.append(f"  {base.rows if base else 0:,} rows · {contract.grain}\n", style="dim")
        if not contract.joins:
            text.append("  └─ nothing joined\n", style="dim")
        for i, step in enumerate(contract.joins):
            branch = "└─" if i == len(contract.joins) - 1 else "├─"
            other = profile.table(step.table)
            text.append(f"  {branch} ")
            if step.kind == "lookup":
                text.append("look up ", style="cyan")
                text.append(step.table, style="bold")
                text.append(f"  {step.base_key} = {step.other_key} · one match per row · "
                            f"{step.matched:.0%} of keys found\n", style="dim")
            else:
                text.append("summarise ", style="magenta")
                text.append(step.table, style="bold")
                text.append(f"  {other.rows if other else 0:,} rows → one row per {step.base_key} · "
                            f"{step.matched:.0%} of keys found\n", style="dim")
        for issue in issues:
            if issue.field == "joins":
                icon, style = ("✘", "bold red") if issue.level == "error" else ("✔", "green")
                text.append(f"\n  {icon} {issue.message}\n", style=style)
        text.append("\n")

    unused = []
    if contract is not None:
        used = {j.table for j in contract.joins}
        for rel in layout.relations:
            if contract.base_table in {rel.left_table, rel.right_table}:
                other = rel.right_table if rel.left_table == contract.base_table else rel.left_table
                if other in used:
                    continue
            unused.append(rel)
    if unused:
        text.append("Other links found, not used\n", style="bold")
        for rel in unused:
            text.append(f"  {rel.describe()}\n", style="dim")
        text.append("\n")
    for note in layout.notes:
        text.append(f"! {note}\n", style="yellow")
    return text


def checks(issues: list[Issue]) -> Text:
    text = Text()
    text.append("HARNESS CHECKS\n", style="bold")
    errors = [i for i in issues if i.level == "error"]
    for issue in issues:
        icon, style = {"error": ("✘", "bold red"), "warning": ("!", "yellow")}.get(issue.level, ("✔", "green"))
        text.append(f" {icon} ", style=style)
        text.append(issue.message + "\n")
    if errors:
        text.append(f"\n {len(errors)} problem(s) block approval.\n", style="bold red")
    else:
        text.append("\n ✔ Nothing blocks approval.\n", style="bold green")
    return text


def target_distribution(contract: DataContract, stats: dict[str, Any]) -> Text:
    text = Text()
    if not contract.target or not stats:
        return text
    text.append(f"{contract.target.upper()}\n", style="bold")
    rows = max(stats.get("rows", 0), 1)
    top = stats.get("top", [])
    distinct = stats.get("distinct", 0)
    if contract.task == "classification" or distinct <= 10:
        width = max((len(str(v)) for v, _ in top), default=1)
        for value, count in top:
            text.append(f" {str(value):<{width}} ")
            text.append(bar(count / rows, 20), style="green")
            text.append(f" {count / rows:5.1%}  {count:,}\n", style="dim")
        if distinct > len(top):
            text.append(f" … and {distinct - len(top)} more values\n", style="dim")
    else:
        text.append(f" {distinct:,} different numbers\n", style="dim")
    if stats.get("nulls"):
        text.append(f" missing: {stats['nulls']:,} rows\n", style="yellow")
    return text


def reasons(contract: DataContract) -> Text:
    text = Text()
    who = {"rules": "fixed rules", "analyst": "the Analyst"}.get(contract.proposed_by, contract.proposed_by)
    text.append("WHY THIS CONTRACT", style="bold")
    text.append(f"  proposed by {who}\n", style="dim")
    for key, label in RATIONALE_ORDER:
        why = contract.rationale.get(key)
        if why:
            text.append(f" {label}: ", style="bold")
            text.append(why + "\n")
    for suspect in contract.leakage_suspects:
        text.append(" Leak? ", style="bold red")
        text.append(f"{suspect.column}: {suspect.reason.rstrip('. ')}.\n")
    return text


def questions(contract: DataContract) -> Text:
    """Things the team could not settle from the data and is asking the person."""
    text = Text()
    if not contract.open_questions:
        return text
    text.append("QUESTIONS FOR YOU\n", style="bold cyan")
    for question in contract.open_questions:
        text.append(" ? ", style="bold cyan")
        text.append(question.strip() + "\n")
        answer = contract.answers.get(question, "").strip()
        if answer:
            text.append("   → ", style="green")
            text.append(answer + "\n")
        else:
            text.append("   not answered yet\n", style="yellow")
    return text


def review(result: Review | None, skipped_reason: str = "") -> Text:
    text = Text()
    text.append("SKEPTIC'S REVIEW\n", style="bold")
    if result is None:
        text.append(f" {skipped_reason or 'Not run.'}\n", style="dim")
        return text
    if result.verdict == "approve":
        text.append(" ✔ No objections. ", style="green")
    else:
        text.append(f" ! {len(result.concerns)} concern(s). ", style="yellow")
    text.append(result.summary + "\n")
    for concern in result.concerns:
        style = {"high": "bold red", "medium": "yellow", "low": "dim"}.get(concern.severity, "")
        text.append(f" {concern.severity.upper():<6} ", style=style)
        if concern.field:
            text.append(f"[{concern.field}] ", style="bold")
        text.append(concern.message)
        if concern.evidence:
            text.append(f"  ({concern.evidence})", style="dim")
        text.append("\n")
    return text


VERDICT_TAG = {"helps": ("helps ", "bold green"), "likely helps": ("likely", "green"),
               "hurts": ("hurts ", "bold red"), "likely hurts": ("worse?", "red"),
               "no clear effect": ("  ?   ", "dim")}


def feature_label(feature: Feature, test: FeatureTest | None = None) -> Text:
    """One line for the feature list."""
    text = Text()
    text.append(f"{feature.name:<26.26} ", style="bold" if feature.usable else "dim")
    if feature.error:
        text.append("rejected", style="bold red")
        return text
    # Flags come before the statistics so a narrow terminal cuts numbers, not warnings.
    flag, flag_style = (("leak? ", "bold red") if feature.leak else ("off?  ", "magenta") if feature.suggested_off
                        else ("check ", "yellow") if feature.warning else ("      ", ""))
    text.append(flag, style=flag_style)
    tag, tag_style = VERDICT_TAG.get(test.verdict, ("      ", "")) if test else ("      ", "")
    text.append(tag, style=tag_style)
    text.append("Engineer " if feature.origin == "engineer" else "         ", style="cyan")
    text.append(f"{feature.kind:<12}", style="dim")
    text.append(f"{bar(feature.null_pct / 100.0, 5)} {feature.null_pct:3.0f}% missing", style="dim")
    return text


def _gain_bars(test: FeatureTest) -> Text:
    """One mark per fold: up = the feature helped in that fold."""
    text = Text(" ")
    top = max((abs(g) for g in test.gains), default=0.0) or 1.0
    for g in test.gains:
        level = min(3, int(abs(g) / top * 3.999))
        text.append("▁▂▄█"[level], style="green" if g > 0 else "red")
    return text


def feature_detail(feature: Feature | None, test: FeatureTest | None = None) -> Text:
    text = Text()
    if feature is None:
        text.append("Move through the list to see what each feature is and why it is there.", style="dim")
        return text
    if test is not None:
        text.append("TESTED  ", style="bold")
        tag, style = VERDICT_TAG.get(test.verdict, ("", ""))
        text.append(test.verdict + "\n", style=style or "bold")
        text.append(test.sentence() + "\n")
        text.append(_gain_bars(test))
        text.append("  one mark per fold: green helped, red hurt\n", style="dim")
        if test.p_adjusted is not None:
            text.append(f" chance this is luck, after correcting for many tests: {test.p_adjusted:.2f}\n", style="dim")
        if feature.leak and test.verdict == "helps":
            text.append(" It helps because it gives the answer away. See the leak warning below.\n", style="bold red")
        text.append("\n")
    text.append(feature.name + "\n", style="bold")
    who = "the Engineer" if feature.origin == "engineer" else "fixed rules"
    text.append(f"proposed by {who}\n\n", style="dim")
    if feature.why:
        text.append("Why: ", style="bold")
        text.append(feature.why + "\n\n")
    if feature.error:
        text.append("✘ Rejected by the harness: ", style="bold red")
        text.append(feature.error + "\n\n")
    else:
        text.append("What it is: ", style="bold")
        text.append(f"{feature.kind}, {feature.distinct:,} different values, "
                    f"missing on {feature.null_pct:.0f}% of rows\n\n")
    if feature.suggested_off:
        text.append("Suggested off: ", style="bold magenta")
        text.append(feature.suggested_off + " It starts unticked; tick it to use it anyway.\n\n")
    if feature.leak:
        text.append("Leak? ", style="bold red")
        text.append(feature.leak + " It is switched off unless you turn it on.\n\n")
    if feature.warning:
        text.append("! ", style="yellow")
        text.append(feature.warning + "\n\n")
    text.append("How it is computed\n", style="bold")
    text.append(feature.sql + "\n", style="dim")
    return text


def feature_table(features: list[Feature], selected: list[str] | None = None,
                  tests: FeatureTestRun | None = None) -> Text:
    """All features as text, for the headless report."""
    text = Text()
    text.append("FEATURES\n", style="bold")
    chosen = set(selected) if selected is not None else {f.name for f in features if f.default_on}
    by_name = tests.by_name() if tests else {}
    for feature in features:
        mark = "✔" if feature.name in chosen and feature.usable else "✘" if feature.error else "·"
        style = "green" if mark == "✔" else "red" if mark == "✘" else "dim"
        text.append(f" {mark} ", style=style)
        text.append(feature_label(feature, by_name.get(feature.name)))
        text.append("\n")
        note = feature.error or feature.leak or feature.warning
        if note:
            text.append(f"     {note}\n", style="dim")
        if feature.name in by_name:
            text.append(f"     tested: {by_name[feature.name].sentence()}\n", style="dim")
    if tests is not None:
        text.append("\n")
        text.append(feature_tests_summary(tests))
    return text


def feature_tests_summary(tests: FeatureTestRun | None) -> Text:
    text = Text()
    if tests is None:
        return text
    text.append(tests.summary() + "\n", style="bold")
    text.append(f"Each feature scored with and without it, fold by fold ({tests.folds} folds, {tests.fold_kind}, "
                f"{tests.reference}), on training rows only. Corrected for testing many features at once.",
                style="dim")
    for note in tests.notes:
        text.append(f"\n! {note}", style="yellow")
    return text


def learn_features() -> Text:
    text = Text()
    text.append("NEW TO THIS?\n", style="bold")
    for term, meaning in (
        ("Feature", "One clue the model may use for each row. Good clues matter more than a clever model."),
        ("Prediction moment", "The point in time you would make the prediction in real life. Anything "
                              "recorded after it is hidden, because you would not have it yet."),
        ("Leak?", "The feature predicts the target almost perfectly. That is usually the answer "
                  "sneaking in, not a discovery. It stays off unless you decide otherwise."),
        ("Not done here", "Filling gaps, scaling numbers and turning categories into numbers happen "
                          "during training, using training rows only."),
    ):
        text.append(f" {term}: ", style="bold")
        text.append(meaning + "\n", style="dim")
    return text


def _score(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}" if abs(value) < 100 else f"{value:,.1f}"


def tuning_summary(tuning: TuneRun | None) -> Text:
    text = Text()
    if tuning is None:
        return text
    counts = tuning.counts
    text.append("Tuning: ", style="bold")
    text.append(f"{len(tuning.trials)} tries in {tuning.spent_seconds:.0f} s, {counts['pruned']} stopped early "
                f"once clearly worse. {tuning.stop_reason}\n")
    curve = best_curve(tuning)
    if curve:
        text.append(" best so far ", style="dim")
        text.append(curve, style="green")
        text.append(f"  {len(tuning.trials)} tries\n", style="dim")
    return text


def model_summary(run: TrainingRun, tuning: TuneRun | None = None) -> Text:
    text = Text()
    text.append("Scored by the harness, not by the agents. ", style="bold")
    text.append(run.split + "\n")
    text.append("The best is picked by cross-validation on training rows; the unseen-row score is reported once "
                "and never used to pick.\n", style="dim")
    text.append(tuning_summary(tuning))
    text.append(f"Metric: {METRIC_WORDS.get(run.metric, run.metric)}.", style="dim")
    if run.positive_class:
        text.append(f" Counting {run.positive_class} as the outcome to find.", style="dim")
    if run.dropped_rows:
        text.append(f" {run.dropped_rows:,} rows without a target were left out.", style="dim")
    for warning in run.warnings:
        text.append(f"\n! {warning}", style="yellow")
    return text


def model_verdict(result: ModelResult) -> Text:
    if result.error:
        return Text("failed", style="bold red")
    if result.family == "baseline":
        return Text("the bar", style="dim")
    if not result.beats_baseline:
        return Text("no gain", style="red")
    if result.recommended:
        return Text("best", style="bold green")
    return Text("check" if result.notes else "ok", style="yellow" if result.notes else "green")


def model_cells(result: ModelResult, run: TrainingRun) -> tuple[Text, str, str, Text, str, Text]:
    """Cells for the models table: name, kind, cross-validated score, test score (with bar), train score, verdict."""
    name = Text(result.name, style="bold" if result.recommended else "")
    if result.origin == "engineer":
        name.append("  Engineer", style="cyan")
    elif result.origin == "tuner":
        name.append("  tuned", style="magenta")
    cv = "-" if result.cv_mean is None else f"{_score(result.cv_mean)} ± {_score(result.cv_std)}"
    test = Text(_score(result.test_score))
    if result.test_score is not None and run.metric in BOUNDED:
        test = Text(f"{bar(max(0.0, result.test_score), 10)} {_score(result.test_score)}",
                    style="green" if result.beats_baseline else "dim")
    return name, result.family, cv, test, _score(result.train_score), model_verdict(result)


def model_detail(result: ModelResult | None, run: TrainingRun | None, tuning: TuneRun | None = None) -> Text:
    text = Text()
    if result is None or run is None:
        text.append("Move through the list to see how each model did and what it relies on.", style="dim")
        return text
    text.append(result.name + "\n", style="bold")
    who = {"engineer": "the Engineer", "tuner": "the tuner", "harness": "the harness"}.get(result.origin, "fixed rules")
    text.append(f"{result.family}, proposed by {who}\n", style="dim")
    words = FAMILY_WORDS.get(result.family, ENSEMBLE_WORDS if result.family == "ensemble" else "")
    text.append(words + "\n\n")
    if result.why and result.why != words and not (words and result.why.startswith(words)):
        text.append("Why try it: ", style="bold")
        text.append(result.why + "\n\n")
    if result.error:
        text.append("✘ Did not train: ", style="bold red")
        text.append(result.error + "\n")
        return text
    settings = ", ".join(f"{k} {v:g}" for k, v in result.params.items() if not (k == "balanced" and not v))
    if result.members:
        settings = (settings + "; " if settings else "") + "averages " + ", ".join(m["name"] for m in result.members)
    if settings:
        text.append("Settings: ", style="bold")
        text.append(settings + "\n")
    baseline = run.baseline
    text.append(f"On unseen rows: ", style="bold")
    text.append(f"{run.metric} {_score(result.test_score)}")
    if baseline is not None and result.family != "baseline":
        text.append(f"  (doing nothing scores {_score(baseline.test_score)})", style="dim")
    text.append("\nOn the rows it trained on: ", style="bold")
    text.append(f"{run.metric} {_score(result.train_score)}\n")
    if result.cv_mean is not None:
        text.append("Cross-validated on training rows: ", style="bold")
        text.append(f"{run.metric} {_score(result.cv_mean)} ± {_score(result.cv_std)}  (what it was picked on)\n")
    if result.origin == "tuner" and tuning is not None:
        text.append(_search_history(result.family, tuning))
    if result.extra:
        text.append("Also, on unseen rows: ", style="bold")
        text.append(", ".join(f"{k} {_score(v)}" for k, v in result.extra.items()) + "\n")
    for note in result.notes:
        text.append("\n! ", style="yellow")
        text.append(note)
    if result.notes:
        text.append("\n")
    if result.importance:
        text.append("\nWhat it relies on\n", style="bold")
        text.append("score lost when that feature is scrambled\n", style="dim")
        # Show the features that matter; fold the ones with no measurable effect into one line.
        shown = [(n, v) for n, v in result.importance if v >= 0.0005][:8]
        rest = len(result.importance) - len(shown)
        top = max((v for _, v in shown), default=0.0)
        width = max((len(n) for n, _ in shown), default=0)
        for name, value in shown:
            text.append(f" {name:<{width}} ")
            text.append(bar(value / top if top > 0 else 0.0, 12), style="green")
            text.append(f" {value:+.3f}\n", style="dim")
        if not shown:
            text.append(" Scrambling any single feature costs nothing: it has learned little.\n", style="yellow")
        elif rest:
            text.append(f" {rest} other feature(s): no measurable effect on unseen rows\n", style="dim")
    if result.evaluation:
        text.append(evaluation_view(evaluation_from_json(result.evaluation), run))
    if result.tree_text:
        text.append("\nThe whole model, as questions\n", style="bold")
        text.append("read top to bottom; each row of data follows one path\n", style="dim")
        for line in result.tree_text.splitlines():
            stem, arrow, leaf = line.partition("→")
            text.append(" " + stem)
            if arrow:
                text.append(arrow + leaf, style="bold green")
            text.append("\n")
    return text


def _pct(value: float | None) -> str:
    return "—" if value is None or value != value else f"{value:.0%}"


def evaluation_view(ev: Evaluation | None, run: TrainingRun | None = None) -> Text:
    """Cut-off, calibration, cost, ranges and groups: what decides if a model is usable."""
    text = Text()
    if ev is None:
        return text
    text.append("\nBefore you use it\n", style="bold")
    text.append(f"choices made on {ev.oof_rows:,} cross-validated training rows, checked on the unseen rows\n",
                style="dim")
    if ev.kind == "binary" and ev.cutoff is not None:
        cut, half = ev.at_cutoff or {}, ev.at_half or {}
        positive = ev.positive_class or "yes"
        text.append(" Cut-off: ", style="bold")
        text.append(f"call it {positive} when the probability is {ev.cutoff:.2f} or more\n")
        text.append(f"  ({ev.cutoff_reason})\n", style="dim")
        text.append(f"  {'':<20}{'at ' + format(ev.cutoff, '.2f'):>9}{'at 0.50':>9}\n", style="dim")
        for label, key in (("cases caught", "recall"), ("right when flagged", "precision"),
                           ("rows flagged", "flagged")):
            text.append(f"  {label:<20}{_pct(cut.get(key)):>9}{_pct(half.get(key)):>9}\n")
        if cut:
            total = cut["tp"] + cut["fp"] + cut["fn"] + cut["tn"]
            text.append(f"  On {total:,} unseen rows: {cut['tp']} caught, {cut['fn']} missed, "
                        f"{cut['fp']} false alarms.\n", style="dim")
        if ev.pr_auc is not None and ev.base_rate is not None:
            text.append(f"  PR AUC {ev.pr_auc:.3f} (doing nothing: {ev.base_rate:.3f})\n", style="dim")
        if ev.cost:
            c = ev.cost
            text.append(" Cost per 1,000 rows: ", style="bold")
            text.append(f"{c['model']:,.0f} with the model, {c['act_on_nobody']:,.0f} acting on nobody, "
                        f"{c['act_on_everybody']:,.0f} acting on everybody\n")
        text.append(" Probabilities: ", style="bold")
        if ev.platt is not None and ev.ece_after is not None:
            text.append(f"rescaled; off by {ev.ece_before:.1%} before, {ev.ece_after:.1%} after, on unseen rows\n")
        elif ev.ece_before is not None:
            text.append(f"off by {ev.ece_before:.1%} on average on unseen rows"
                        + (", close enough\n" if ev.ece_before <= 0.05 else "\n"))
        if ev.bins:
            text.append("  said  →  happened\n", style="dim")
            for said, happened, rows in ev.bins[:: max(1, len(ev.bins) // 5)]:
                text.append(f"  {said:>4.0%}  →  {happened:>4.0%}  ", style="dim")
                text.append(bar(happened, 10), style="green" if abs(said - happened) <= 0.1 else "yellow")
                text.append(f"  {rows} rows\n", style="dim")
    elif ev.kind == "multiclass":
        text.append(" Per class on unseen rows\n", style="bold")
        width = max((len(str(p["class"])) for p in ev.per_class), default=5)
        for p in ev.per_class:
            text.append(f"  {str(p['class']):<{width}}  found {_pct(p['recall']):>5}  right when said "
                        f"{_pct(p['precision']):>5}  ({p['rows']} rows)\n")
    elif ev.kind == "regression" and ev.half_width is not None:
        text.append(" Range: ", style="bold")
        text.append(f"prediction ± {ev.half_width:,.4g} held the true value for 80% of cross-validated rows; "
                    f"on unseen rows it held for {_pct(ev.coverage)}.\n")
        if ev.bias is not None:
            text.append(f"  On average it predicts {abs(ev.bias):,.4g} too {'high' if ev.bias > 0 else 'low'}.\n",
                        style="dim")
    for note in ev.notes:
        text.append(" ! ", style="yellow")
        text.append(note + "\n")
    for group in ev.groups:
        text.append(f" By {group.column}\n", style="bold")
        width = max(len(r.value) for r in group.rows)
        for r in group.rows:
            if ev.kind == "binary":
                text.append(f"  {r.value:<{width}}  {r.rows:>4} rows  outcome {_pct(r.actual):>4}  flagged "
                            f"{_pct(r.predicted):>4}  caught {_pct(r.recall):>4}\n")
            elif ev.kind == "regression":
                text.append(f"  {r.value:<{width}}  {r.rows:>4} rows  average {r.actual:,.4g}  predicted "
                            f"{r.predicted:,.4g}  typical error {r.score:,.4g}\n")
            else:
                text.append(f"  {r.value:<{width}}  {r.rows:>4} rows  right {_pct(r.score):>4}\n")
        if group.gap:
            text.append("  ! " + group.gap + "\n", style="yellow")
        if group.context:
            text.append("  · " + group.context[0].upper() + group.context[1:] + "\n", style="dim")
        if not group.gap and not group.context:
            text.append("  No large gap between groups.\n", style="dim")
    return text


def _search_history(family: str, tuning: TuneRun) -> Text:
    text = Text()
    tries = [t for t in tuning.trials if t.family == family]
    done = sorted((t for t in tries if t.state == "complete" and t.cv_mean is not None),
                  key=lambda t: t.cv_mean or 0.0, reverse=tuning.higher_is_better)
    pruned = sum(1 for t in tries if t.state == "pruned")
    text.append(f"\nHow it was found\n", style="bold")
    text.append(f" {len(tries)} tries for {family}, {pruned} stopped early. The first used the default settings; "
                "later ones were chosen where good results were likelier.\n", style="dim")
    for t in done[:4]:
        settings = ", ".join(f"{k} {v:g}" for k, v in t.params.items())
        text.append(f"  try {t.number:<3} {_score(t.cv_mean)} ± {_score(t.cv_std)}  {settings}\n")
    if family in tuning.dropped:
        text.append(f" {tuning.dropped[family]}\n", style="dim")
    return text


def model_table(run: TrainingRun) -> Text:
    """The scoreboard as text, for the headless report."""
    text = Text()
    text.append("MODELS\n", style="bold")
    width = max((len(r.name) for r in run.results), default=8)
    for result in run.results:
        mark = "✘" if result.error else "★" if result.recommended else "·"
        text.append(f" {mark} ", style="bold green" if result.recommended else "red" if result.error else "dim")
        text.append(f"{result.name:<{width}}  {result.family:<9}", style="bold" if result.recommended else "")
        cv = "-" if result.cv_mean is None else f"{_score(result.cv_mean)}±{_score(result.cv_std)}"
        text.append(f" cross-val {cv:>13}   unseen {_score(result.test_score):>8}   "
                    f"trained-on {_score(result.train_score):>8}   ")
        text.append(model_verdict(result))
        text.append("\n")
        for note in result.notes:
            text.append(f"     ! {note}\n", style="yellow")
    return text


def prediction(result: Prediction | None, skipped: str = "") -> Text:
    text = Text()
    if result is None:
        if skipped:
            text.append("PREDICTIONS\n", style="bold")
            text.append(f" {skipped}\n", style="dim")
        return text
    text.append("PREDICTIONS\n", style="bold")
    text.append(f" {result.rows:,} rows of {result.table} scored with {result.model} (model v{result.model_version}).\n")
    if result.counts:
        width = max(len(str(k)) for k in result.counts)
        for label, count in sorted(result.counts.items(), key=lambda item: -item[1]):
            text.append(f"  {str(label):<{width}} ")
            text.append(bar(count / result.rows, 16), style="green")
            text.append(f" {count / result.rows:5.1%}  {count:,}\n", style="dim")
    if result.summary:
        text.append(f"  lowest {result.summary['min']:,.2f} · average {result.summary['mean']:,.2f} · "
                    f"highest {result.summary['max']:,.2f}\n", style="dim")
    for note in result.notes:
        text.append(" ! ", style="yellow")
        text.append(note + "\n")
    if not result.notes:
        text.append(" ✔ These rows look like the training rows: same gaps, no new categories.\n", style="green")
    if result.score:
        text.append(" This data includes the answers, so the model was checked on it: ", style="bold")
        text.append(f"{result.score['metric']} {_score(result.score['value'])} on {result.score['rows']:,} rows")
        if result.score.get("original") is not None:
            text.append(f" (it scored {_score(result.score['original'])} on its original unseen rows)", style="dim")
        text.append("\n")
    if result.health:
        text.append(health(result.health))
    text.append(f" Saved: {result.output}", style="bold")
    if result.submission:
        text.append(f" and {result.submission} (in the sample submission's format)", style="bold")
    text.append("\n")
    if not result.score:
        text.append(" Nobody knows the right answers for these rows, so this output has no score.\n", style="dim")
    return text


STATUS_STYLE_HEALTH = {"stable": "bold green", "watch": "bold yellow", "retrain": "bold red"}


def _psi_style(value: float) -> str:
    return "green" if value < STABLE else "yellow" if value < WATCH else "red"


def health(info: dict) -> Text:
    """Drift against the training rows, for one scored batch."""
    text = Text()
    status = info.get("status") or "stable"
    text.append(" DRIFT  ", style="bold")
    text.append(ADVICE.get(status, status) + "\n", style=STATUS_STYLE_HEALTH.get(status, "bold"))
    for reason in info.get("reasons") or []:
        text.append(f"  · {reason}\n")
    feature_psi = info.get("feature_psi") or {}
    if feature_psi:
        width = max(len(n) for n in feature_psi)
        for name, value in sorted(feature_psi.items(), key=lambda kv: -kv[1])[:8]:
            text.append(f"  {name:<{width}} ")
            text.append(bar(min(1.0, value / (2 * WATCH)), 12), style=_psi_style(value))
            text.append(f" PSI {value:.2f}\n", style="dim")
        if info.get("prediction_psi") is not None:
            value = info["prediction_psi"]
            text.append(f"  {'predictions':<{width}} ")
            text.append(bar(min(1.0, value / (2 * WATCH)), 12), style=_psi_style(value))
            text.append(f" PSI {value:.2f}\n", style="dim")
        text.append("  PSI under 0.10 stable, 0.10 to 0.25 moving, above 0.25 shifted, after allowing for chance.\n",
                    style="dim")
    return text


def monitor_report(history: list[dict], card: dict | None = None) -> Text:
    """Every scored batch, oldest first, with the drift verdict and score."""
    text = Text()
    text.append("MONITORING\n", style="bold")
    if card is not None:
        text.append(f" Live model: {card.get('model')} v{card.get('version')}")
        if card.get("replaces"):
            text.append(f" (replaced v{card['replaces']} after a challenger)", style="dim")
        text.append("\n")
    if not history:
        text.append(" Nothing scored yet. Score a file to start watching for drift.\n", style="dim")
        return text
    width = max(len(str(h.get("source"))) for h in history)
    for h in history:
        status = h.get("status") or "-"
        text.append(f" {str(h.get('at', ''))[:16]:<16}  {str(h.get('source')):<{width}}  {h.get('rows', 0):>6,} rows  ")
        text.append(f"{status:<8}", style=STATUS_STYLE_HEALTH.get(status, "dim"))
        if h.get("max_feature") is not None:
            text.append(f"  most moved: {h['max_feature']} ", style="dim")
            text.append(f"{h['max_psi']:.2f}", style=_psi_style(h["max_psi"]))
        if h.get("score"):
            text.append(f"  score {h['score']['metric']} {h['score']['value']:.3f}", style="bold")
        text.append(f"  model v{h.get('model_version')}\n", style="dim")
    latest = history[-1]
    text.append("\n")
    text.append(" Now: ", style="bold")
    text.append(ADVICE.get(latest.get("status") or "stable", ""), style=STATUS_STYLE_HEALTH.get(latest.get("status"), ""))
    text.append("\n")
    for reason in latest.get("reasons") or []:
        text.append(f"  · {reason}\n")
    moving = sorted({n for h in history for n, v in (h.get("feature_psi") or {}).items() if v >= STABLE})
    if moving and len(history) > 1:
        text.append("\n Trend for features that moved (PSI per batch, oldest first)\n", style="bold")
        for name in moving[:8]:
            values = [(h.get("feature_psi") or {}).get(name) for h in history]
            text.append(f"  {name:<20} ")
            for v in values:
                text.append("  -  " if v is None else f"{v:5.2f}", style="dim" if v is None else _psi_style(v))
                text.append(" ")
            text.append("\n")
    return text


def challenge_report(c: Challenge) -> Text:
    text = Text()
    style = {"better": "bold green", "worse": "bold red"}.get(c.verdict, "bold yellow")
    text.append(f"CHALLENGER v{c.version}  ", style="bold")
    text.append(c.sentence() + "\n", style=style)
    text.append(f" Same model ({c.family}, same settings) retrained on the original {c.original_train_rows:,} "
                f"training rows plus {c.new_train_rows:,} new rows from {c.source}.\n")
    text.append(f" Both scored on {c.holdout_how} that neither had seen:\n", style="dim")
    best = max(c.champion_new, c.challenger_new) or 1.0
    for label, value in (("champion  ", c.champion_new), ("challenger", c.challenger_new)):
        text.append(f"  {label} ")
        text.append(bar(max(0.0, value / best) if c.higher_is_better else 0.5, 16), style="green")
        text.append(f" {c.metric} {value:.3f}\n")
    text.append(f"  gain {c.gain_new:+.3f}, and when the held-back rows are resampled it ranges from "
                f"{c.gain_low:+.3f} to {c.gain_high:+.3f}\n", style="dim")
    if c.champion_old is not None and c.challenger_old is not None:
        text.append(f" On the original test rows: champion {c.champion_old:.3f}, challenger {c.challenger_old:.3f}\n",
                    style="dim")
    for note in c.notes:
        text.append(f" ! {note}\n", style="yellow")
    if c.promoted_as is not None:
        text.append(f" Promoted: live as model v{c.promoted_as}.\n", style="bold green")
    elif c.verdict == "worse":
        text.append(" Cannot be promoted.\n", style="red")
    else:
        text.append(" Promotion is your call; nothing switches on its own.\n", style="dim")
    return text


FINDING_TITLE = {
    "signal": "Signal", "weak": "Weak signal", "skew": "Long tail", "redundant": "Near-copies",
    "labels": "Messy labels", "missing": "Missing values that matter", "duplicates": "Duplicate rows",
    "balance": "Rare outcome", "size": "Few rows",
}


def insights_view(ins: DataInsights | None, contract_note: str = "") -> Text:
    """What understanding the data found, and what changed because of it."""
    text = Text()
    if ins is None:
        text.append("Approve the data contract first. The data is examined against the target you chose.",
                    style="dim")
        return text
    text.append("UNDERSTANDING THE DATA\n", style="bold")
    text.append(f"{ins.rows:,} training rows (the locked test rows are not looked at), {ins.features} features, "
                f"predicting {ins.target} ({ins.task}).\n", style="dim")
    if contract_note:
        text.append(contract_note + "\n", style="dim")
    if ins.signal:
        text.append("\nWhich features tell most about the target\n", style="bold")
        text.append("mutual information: how much knowing the feature narrows down the answer, on its own\n",
                    style="dim")
        top = ins.signal[0][1] or 1.0
        width = max(len(n) for n, _ in ins.signal[:12])
        for name, value in ins.signal[:12]:
            text.append(f" {name:<{width}} ")
            text.append(bar(value / top if top > 0 else 0.0, 20), style="green" if value >= 0.005 else "dim")
            text.append(f" {value:.3f}\n", style="dim")
        if len(ins.signal) > 12:
            text.append(f" ... and {len(ins.signal) - 12} more\n", style="dim")
        text.append(" On its own does not mean in combination: a weak feature can still help next to others.\n",
                    style="dim")
    findings = [f for f in ins.findings if f.kind != "signal"]
    if findings:
        text.append("\nWhat was found, and what was done about it\n", style="bold")
        for f in findings:
            style = "yellow" if f.severity == "warn" else "cyan"
            text.append(f" {FINDING_TITLE.get(f.kind, f.kind):<26}", style="bold " + style)
            text.append(f.text + "\n")
            if f.action:
                text.append(f" {'':<26}→ {f.action}\n", style="green")
    else:
        text.append("\nNothing unusual: no long tails, near-copies, messy labels or duplicates.\n", style="green")
    for advice in ins.advice:
        text.append(f"\nFor training: {advice}\n", style="bold")
    return text


def learn_models() -> Text:
    text = Text()
    text.append("NEW TO THIS?\n", style="bold")
    for term, meaning in (
        ("Unseen rows", "Rows held back from training. The score on them is the honest one: it shows "
                        "how the model does on data it has never met."),
        ("Baseline", "A model that ignores every clue and always gives the most common answer. If a "
                     "real model cannot beat it, the features hold no usable signal."),
        ("Memorising", "Scoring far higher on training rows than on unseen rows. The model learned "
                       "those rows by heart instead of a pattern that carries over."),
        ("Which to pick", "The best score is a starting point. A simpler model you can read, scoring "
                          "almost the same, is often the better choice."),
    ):
        text.append(f" {term}: ", style="bold")
        text.append(meaning + "\n", style="dim")
    return text


def learn() -> Text:
    text = Text()
    text.append("NEW TO THIS?\n", style="bold")
    for term, meaning in LEARN:
        text.append(f" {term}: ", style="bold")
        text.append(meaning + "\n", style="dim")
    return text
