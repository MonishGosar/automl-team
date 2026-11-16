"""Command line entry point.

    mlteam                         open the app and ask for data
    mlteam data.csv                open the app on a file
    mlteam data/ kaggle:owner/ds   several sources at once
    mlteam demo:shop               try a built-in sample
    mlteam --headless demo:loans   no app: print the proposal and the checks
    mlteam predict latest new.csv  score a new file with an approved model
    mlteam monitor latest          drift and score for every batch scored so far
    mlteam retrain latest new.csv  train a challenger on new rows that have the answers
    mlteam promote latest 1        make challenger 1 the live model
    mlteam serve latest            serve the approved model: POST /predict
    mlteam resume <run>            pick a run up where it stopped
    mlteam runs                    every run side by side
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.text import Text

from mlteam import __version__, samples
from mlteam.agents.brain import Brain, resolve_brain
from mlteam.agents.roles import TeamEvent
from mlteam.contract import has_errors, target_stats
from mlteam.features import time_rule_summary
from mlteam.intake import IntakeError
from mlteam.challenger import challenge, promote
from mlteam.monitor import read_history
from mlteam.predict import PredictError, predict_file
from mlteam.tune import Budget
from mlteam.runstore import RunPaths, Trace, default_runs_dir, new_run
from mlteam.store import StoreError
from mlteam.team import STAGES, Team
from mlteam.tui import render


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlteam",
        description="An AutoML team you steer: agents do the work, the harness sets the rules, "
                    "you decide at the gates.",
    )
    parser.add_argument("sources", nargs="*",
                        help="CSV/TSV/Parquet files, folders, globs, zips, kaggle:owner/dataset, "
                             "or demo:loans|titanic|shop|shards")
    parser.add_argument("--headless", action="store_true", help="print a report instead of opening the app")
    parser.add_argument("--approve", action="store_true",
                        help="with --headless: approve the proposed contract if every check passes")
    parser.add_argument("--approve-features", action="store_true",
                        help="with --headless --approve: also approve the features that are on by default")
    parser.add_argument("--tune-seconds", type=float, default=60.0, metavar="S",
                        help="time budget for Bayesian tuning of the model settings (0 skips it)")
    parser.add_argument("--tune-trials", type=int, default=80, metavar="N", help="cap on tuning tries")
    parser.add_argument("--test-seconds", type=float, default=30.0, metavar="S",
                        help="time budget for testing each feature with and without it (0 skips it)")
    parser.add_argument("--approve-model", action="store_true",
                        help="with --headless --approve --approve-features: train, then approve the best model")
    parser.add_argument("--horizon-days", type=int, metavar="N",
                        help="with --headless: the prediction moment, N days after the time column")
    parser.add_argument("--cost-false-alarm", type=float, metavar="X",
                        help="what acting on a 'yes' that turns out 'no' costs (any unit); needs --cost-miss")
    parser.add_argument("--cost-miss", type=float, metavar="X", help="what missing a real 'yes' costs")
    parser.add_argument("--group-column", metavar="COL",
                        help="column whose rows belong together (e.g. customer_id); 'none' to turn grouping off")
    parser.add_argument("--fairness", action="append", default=[], metavar="COL",
                        help="extra column to check fairness on (age, sex and similar are checked anyway)")
    parser.add_argument("--redo-contract", metavar="TEXT",
                        help="feedback for the Analyst: redo the contract once with it")
    parser.add_argument("--redo-features", metavar="TEXT",
                        help="feedback for the Engineer: redo the features once with it")
    parser.add_argument("--redo-models", metavar="TEXT",
                        help="feedback for the model plan: redo training once with it")
    parser.add_argument("--answer", action="append", default=[], metavar="TEXT",
                        help="with --headless: answer to the team's questions, in the order asked; repeatable")
    parser.add_argument("--no-llm", action="store_true", help="rules only: do not call a model")
    parser.add_argument("--stats-only", action="store_true",
                        help="agents see counts and statistics only; no raw rows are sent to the model")
    parser.add_argument("--provider", help="Tau provider name (default: Tau's default provider)")
    parser.add_argument("--model", help="model id for that provider")
    parser.add_argument("--runs-dir", type=Path, help="where runs are stored (default: ./mlteam_runs)")
    parser.add_argument("--approver", default="human", help="name recorded on approvals")
    parser.add_argument("--version", action="version", version=f"mlteam {__version__}")
    return parser


def _brain(args: argparse.Namespace) -> tuple[Brain | None, str]:
    if args.no_llm:
        return None, "Not run: started with --no-llm."
    return resolve_brain(args.provider, args.model)


async def run_headless(args: argparse.Namespace, brain: Brain | None, note: str,
                       resume: RunPaths | None = None) -> int:
    console = Console()
    labels = dict(STAGES)

    def emit(event: TeamEvent) -> None:
        if event.kind == "stage" and event.data.get("status") in {"done", "failed", "skipped"}:
            icon, style = render.STATUS_STYLE[event.data["status"]]
            console.print(Text(f"{icon} {labels.get(event.stage, event.stage)}  {event.text}", style=style))
        elif event.kind == "agent_text":
            console.print(Text(f"{event.agent}: {event.text}"))
        elif event.kind in {"note", "error"}:
            console.print(Text(f"! {event.text}", style="yellow" if event.kind == "note" else "red"))

    options = dict(brain=brain, brain_note=note, stats_only=args.stats_only, emit=emit,
                   tune_budget=Budget(seconds=args.tune_seconds, max_trials=args.tune_trials),
                   test_seconds=args.test_seconds)
    if resume is not None:
        try:
            team = Team.resume(resume.root, **options)
        except (ValueError, StoreError) as exc:
            console.print(Text(str(exc), style="red"))
            return 2
        run = team.run
    else:
        run = new_run(args.runs_dir)
        team = Team(run, **options)
    try:
        if resume is None:
            try:
                await team.prepare(samples.resolve_specs(args.sources, run.root / "demo"))
            except (IntakeError, StoreError, ValueError) as exc:
                console.print(Text(str(exc), style="red"))
                return 2
            await team.consult()

        # -- the data contract ------------------------------------------
        if team.status.get("gate") != "done":
            if args.redo_contract:
                await team.redo("contract", args.redo_contract)
            assert team.contract and team.profile and team.layout and team.store
            unanswered = [q for q in team.contract.open_questions if not team.contract.answers.get(q, "").strip()]
            for question, answer in zip(unanswered, args.answer, strict=False):
                team.contract.answers[question] = answer
            if args.horizon_days is not None:
                team.contract.horizon_days = args.horizon_days
            if args.cost_false_alarm is not None or args.cost_miss is not None:
                team.contract.cost_false_alarm, team.contract.cost_miss = args.cost_false_alarm, args.cost_miss
            if args.group_column:
                team.contract.group_column = None if args.group_column.lower() == "none" else args.group_column
            if args.fairness:
                team.contract.fairness_columns = list(args.fairness)
            issues = team.validate(team.contract)
            console.print()
            console.print(render.relations(team.profile, team.layout, team.contract, issues))
            console.print(render.questions(team.contract))
            console.print(render.reasons(team.contract))
            console.print(render.target_distribution(team.contract, target_stats(team.store, team.contract)))
            console.print(render.review(team.review, _skeptic_note(team, "skeptic", note)))
            console.print(render.checks(issues))
            console.print(Text(f"Run folder: {run.root}", style="dim"))
            team._checkpoint()
            if has_errors(issues):
                return 1
            if not args.approve:
                return 0
            path = team.approve(team.contract, args.approver)
            console.print(Text(f"Approved: {path}", style="green"))
            console.print()

        # -- the features -----------------------------------------------
        if team.status.get("feature_gate") != "done":
            if team.status.get("feature_gate") != "waiting" or not team.features:
                await team.build_features()
            if args.redo_features:
                await team.redo("features", args.redo_features)
            features = team.features
            assert team.contract is not None
            console.print(Text(time_rule_summary(team.contract), style="bold"))
            console.print(render.insights_view(team.insights))
            console.print(render.feature_table(features, tests=team.feature_tests))
            console.print(render.review(team.feature_review, _skeptic_note(team, "feature_skeptic", note)))
            if not args.approve_features:
                console.print(Text(f"Paused at the features. Continue with: mlteam resume {run.root}", style="dim"))
                return 0
            picked = [f.name for f in features if f.default_on]
            path = team.approve_features(picked, args.approver)
            console.print(Text(f"Approved {len(picked)} features: {path}", style="green"))
            console.print()

        # -- the models -------------------------------------------------
        if team.status.get("model_gate") != "done":
            if team.status.get("model_gate") != "waiting" or team.training is None:
                await team.train_models()
            if args.redo_models:
                await team.redo("models", args.redo_models)
            run_result = team.training
            assert run_result is not None
            console.print(render.model_summary(run_result, team.tuning))
            console.print(render.model_table(run_result))
            best = next((r for r in run_result.results if r.recommended), None)
            if best is not None:
                console.print(render.model_detail(best, run_result, team.tuning))
            readable = next((r for r in run_result.results if r.tree_text), None)
            if readable is not None and readable is not best:
                console.print(render.model_detail(readable, run_result, team.tuning))
            console.print(render.review(team.model_review, _skeptic_note(team, "model_skeptic", note)))
            if not args.approve_model:
                console.print(Text(f"Paused at the models. Continue with: mlteam resume {run.root}", style="dim"))
                return 0
            if best is None:
                console.print(Text("No model beat the baseline, so none was approved.", style="red"))
                return 1
            path = team.approve_model(best.name, args.approver)
            console.print(Text(f"Approved model {best.name}: {path}", style="green"))
            if team.report_path is not None:
                console.print(Text(f"Results report: {team.report_path}", style="bold green"))
            console.print()
            try:
                scored = team.predict()
            except ValueError as exc:
                console.print(Text(f"The test file could not be scored: {exc}", style="red"))
                return 1
            console.print(render.prediction(scored, "This data has no held-back test file to score."))
        else:
            card = team.model_card or {}
            console.print(Text(f"Model v{card.get('version')} ({card.get('model')}) is already approved in this run. "
                               f"Serve it with: mlteam serve {run.root}", style="green"))
        return 0
    finally:
        team.close()


def find_run(spec: str, runs_dir: Path | None) -> RunPaths:
    """A run folder, a run id inside the runs folder, or 'latest' (newest run with an approved model)."""
    base = runs_dir or default_runs_dir()
    if spec == "latest":
        runs = sorted((p for p in base.glob("*") if (p / "model.json").exists()), key=lambda p: p.name)
        if not runs:
            raise PredictError(f"No run with an approved model in {base}.")
        return RunPaths(runs[-1])
    for candidate in (Path(spec), base / spec):
        if candidate.is_dir():
            return RunPaths(candidate.resolve())
    raise PredictError(f"Run not found: {spec}")


def run_predict(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="mlteam predict",
        description="Score a new file with the model approved in a finished run.",
    )
    parser.add_argument("run", help="run folder, run id, or 'latest'")
    parser.add_argument("file", type=Path, help="CSV, TSV or Parquet file with the training table's columns")
    parser.add_argument("--runs-dir", type=Path, help="where runs are stored (default: ./mlteam_runs)")
    args = parser.parse_args(argv)
    console = Console()
    try:
        run = find_run(args.run, args.runs_dir)
        result = predict_file(run, args.file)
    except PredictError as exc:
        console.print(Text(str(exc), style="red"))
        return 1
    Trace(run.trace).log("prediction", **result.to_json())
    console.print(render.prediction(result))
    console.print(Text(f"Written to {run.root / result.output}", style="dim"))
    return 0


def _run_or_fail(console: Console, spec: str, runs_dir: Path | None) -> RunPaths | None:
    try:
        return find_run(spec, runs_dir)
    except PredictError as exc:
        console.print(Text(str(exc), style="red"))
        return None


def run_monitor(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="mlteam monitor", description="Drift and score for every batch scored so far.")
    parser.add_argument("run", help="run folder, run id, or 'latest'")
    parser.add_argument("--runs-dir", type=Path)
    args = parser.parse_args(argv)
    console = Console()
    run = _run_or_fail(console, args.run, args.runs_dir)
    if run is None:
        return 1
    card = json.loads(run.model_latest.read_text(encoding="utf-8")) if run.model_latest.exists() else None
    console.print(render.monitor_report(read_history(run.monitoring), card))
    return 0


def run_retrain(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="mlteam retrain",
        description="Train a challenger on new rows that carry the answers, and compare it with the live model.",
    )
    parser.add_argument("run", help="run folder, run id, or 'latest'")
    parser.add_argument("file", type=Path, help="file with the training table's columns plus the target")
    parser.add_argument("--runs-dir", type=Path)
    args = parser.parse_args(argv)
    console = Console()
    run = _run_or_fail(console, args.run, args.runs_dir)
    if run is None:
        return 1
    try:
        result = challenge(run, args.file)
    except PredictError as exc:
        console.print(Text(str(exc), style="red"))
        return 1
    console.print(render.challenge_report(result))
    if result.verdict != "worse":
        console.print(Text(f"To make it live: mlteam promote {args.run} {result.version}", style="dim"))
    return 0


def run_promote(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="mlteam promote", description="Make a challenger the live model.")
    parser.add_argument("run", help="run folder, run id, or 'latest'")
    parser.add_argument("challenger", type=int, help="challenger number, from mlteam retrain")
    parser.add_argument("--runs-dir", type=Path)
    parser.add_argument("--approver", default="human")
    args = parser.parse_args(argv)
    console = Console()
    run = _run_or_fail(console, args.run, args.runs_dir)
    if run is None:
        return 1
    try:
        path = promote(run, args.challenger, args.approver)
    except PredictError as exc:
        console.print(Text(str(exc), style="red"))
        return 1
    console.print(Text(f"Challenger v{args.challenger} is now the live model: {path}", style="green"))
    return 0


def _skeptic_note(team, stage: str, note: str | None) -> str | None:
    """What to show when there is no review: never the model's name, which reads like 'ok'."""
    if team.status.get(stage) == "failed":
        return "The Skeptic did not finish its review."
    return note or "Not run: no model connected."


def run_runs(argv: list[str]) -> int:
    """Every run side by side: what was predicted, which model, how well."""
    import json

    from rich.console import Console
    from rich.table import Table

    from mlteam.runstore import default_runs_dir

    parser = argparse.ArgumentParser(prog="mlteam runs", description="Compare runs side by side.")
    parser.add_argument("--runs-dir", type=Path)
    args = parser.parse_args(argv)
    root = args.runs_dir or default_runs_dir()
    folders = sorted((p for p in root.glob("*") if (p / "manifest.json").exists()), key=lambda p: p.name)
    if not folders:
        print(f"No runs in {root}.")
        return 0

    def read(path: Path) -> dict:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    table = Table(title="Runs", caption=str(root), show_lines=False)
    for name, justify in (("run", "left"), ("data", "left"), ("target", "left"), ("metric", "left"),
                          ("model", "left"), ("cross-val", "right"), ("unseen", "right"), ("nothing", "right"),
                          ("status", "left")):
        table.add_column(name, justify=justify)  # type: ignore[arg-type]
    for folder in folders:
        manifest, contract = read(folder / "manifest.json"), read(folder / "contract.json")
        card, training = read(folder / "model.json"), read(folder / "training.json")
        files = [f.get("name", "") for f in manifest.get("files", [])]
        data = files[0] + (f" +{len(files) - 1}" if len(files) > 1 else "") if files else "?"
        best = next((r for r in training.get("results", []) if r.get("recommended")), {})
        shown = card or best

        def num(value: object) -> str:
            return f"{value:,.4g}" if isinstance(value, (int, float)) else "—"

        status = (f"model v{card['version']} approved" if card else "trained" if training
                  else "contract approved" if contract.get("approved_by") else "draft")
        table.add_row(folder.name, data, str(contract.get("target") or "—"),
                      str(card.get("metric") or training.get("metric") or "—"), str(shown.get("model") or
                                                                                    shown.get("name") or "—"),
                      num(shown.get("cv_mean")), num(shown.get("test_score")),
                      num(card.get("baseline_test_score")), status)
    console = Console()
    if not console.is_terminal:  # piped or captured: no terminal width to fit, so do not cut values off
        console = Console(width=160)
    console.print(table)
    return 0


def run_resume(argv: list[str]) -> int:
    """Open a run where it stopped, in the app or headless, with the usual options."""
    parser = build_parser()
    parser.prog = "mlteam resume"
    parser.description = "Pick a run up where it stopped: at its last open gate."
    args = parser.parse_args(argv)
    if len(args.sources) != 1:
        print("Give one run: a run folder, a run id, or 'latest'.", file=sys.stderr)
        return 2
    spec = args.sources[0]
    try:
        if spec == "latest":
            base = args.runs_dir or default_runs_dir()
            folders = sorted((p for p in base.glob("*") if (p / "manifest.json").exists()), key=lambda p: p.name)
            if not folders:
                raise PredictError(f"No runs in {base}.")
            run = RunPaths(folders[-1])
        else:
            run = find_run(spec, args.runs_dir)
    except PredictError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    brain, note = _brain(args)
    if args.headless:
        return asyncio.run(run_headless(args, brain, note, resume=run))
    return _open_app(args, brain, note, resume=run.root)


def run_serve(argv: list[str]) -> int:
    from mlteam.serve import ServeError, make_server, model_info

    parser = argparse.ArgumentParser(prog="mlteam serve", description="Serve the approved model over HTTP.")
    parser.add_argument("run", help="run folder, run id, or 'latest'")
    parser.add_argument("--host", default="127.0.0.1", help="address to listen on (default: this machine only)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--runs-dir", type=Path)
    parser.add_argument("--quiet", action="store_true", help="do not log each request")
    args = parser.parse_args(argv)
    try:
        run = find_run(args.run, args.runs_dir)
        server = make_server(run, args.host, args.port, quiet=args.quiet)
        info = model_info(run)
    except (PredictError, ServeError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    host, port = server.server_address[:2]
    print(f"Serving {info['model']} v{info['version']} from run {run.run_id}")
    print(f"  POST http://{host}:{port}/predict   body: {{\"rows\": [{{...}}, ...]}}")
    print(f"  GET  http://{host}:{port}/model     GET http://{host}:{port}/health")
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        print("  ! Listening beyond this machine with no authentication. Put it behind a proxy that has some.")
    print("Ctrl+C stops it.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    raw = list(sys.argv[1:] if argv is None else argv)
    commands = {"predict": run_predict, "monitor": run_monitor, "retrain": run_retrain, "promote": run_promote,
                "runs": run_runs, "serve": run_serve}
    if raw and raw[0] == "resume":
        return run_resume(raw[1:])
    if raw and raw[0] in commands:
        return commands[raw[0]](raw[1:])
    args = build_parser().parse_args(raw)
    brain, note = _brain(args)
    if args.headless:
        if not args.sources:
            print("--headless needs at least one source.", file=sys.stderr)
            return 2
        return asyncio.run(run_headless(args, brain, note))

    return _open_app(args, brain, note)


def _open_app(args: argparse.Namespace, brain: Brain | None, note: str, resume: Path | None = None) -> int:
    from mlteam.tui.app import MLTeamApp

    MLTeamApp(
        args.sources if resume is None else [],
        resume=resume,
        runs_dir=args.runs_dir,
        brain=brain,
        brain_note=note,
        stats_only=args.stats_only,
        approver=args.approver,
        tune_budget=Budget(seconds=args.tune_seconds, max_trials=args.tune_trials),
        test_seconds=args.test_seconds,
    ).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
