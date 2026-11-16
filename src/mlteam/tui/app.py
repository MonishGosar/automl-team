"""The terminal app.

Left: the pipeline and the tables. Right: four tabs.
  Data       what was loaded, column by column
  Relations  how the files fit together and what gets joined
  Contract   the first gate: edit, see the harness checks live, approve
  Features   the second gate: pick the clues the model may use
  Models     the third gate: every model scored by the harness; pick one
  Monitor    after go-live: drift per batch, challengers, promotion
  Team       what the agents said and ran
"""

from __future__ import annotations

import asyncio
import copy
import threading
from pathlib import Path

from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Checkbox,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    OptionList,
    RichLog,
    Select,
    SelectionList,
    Static,
    TabbedContent,
    TabPane,
)
from textual.widgets.option_list import Option
from textual.widgets.selection_list import Selection

from mlteam import samples
from mlteam.agents.brain import Brain
from mlteam.agents.roles import TeamEvent
from mlteam.contract import (
    METRICS,
    TASKS,
    DataContract,
    Issue,
    JoinStep,
    attach_time_columns,
    candidate_joins,
    default_metric,
    has_errors,
    retarget,
    target_stats,
)
from mlteam.features import Feature, time_rule_summary
from mlteam.tune import Budget
from mlteam.intake import IntakeError, KaggleFetcher, fetch_from_kaggle
from mlteam.runstore import new_run
from mlteam.store import StoreError
from mlteam.team import STAGES, Team
from mlteam.tui import render

STAGE_LABELS = dict(STAGES)
AGENT_STYLE = {"Analyst": "bold cyan", "Skeptic": "bold magenta"}


class IntakeScreen(ModalScreen[str | None]):
    """Ask for data. Shown at start when no source was given, and again if intake fails."""

    BINDINGS = [Binding("escape", "cancel", "Quit")]

    def __init__(self, error: str = "") -> None:
        super().__init__()
        self._error = error

    def compose(self) -> ComposeResult:
        with Vertical(id="intake_box"):
            yield Static("mlteam", id="intake_title")
            yield Static(
                "Point me at data. I load it, work out how the files fit together, "
                "and propose what to predict. Nothing is decided until you approve it.",
                id="intake_lead",
            )
            yield Input(placeholder="file.csv · a folder · data/*.csv · bundle.zip · kaggle:owner/dataset",
                        id="source")
            yield Static("Several sources: separate them with ;", classes="hint")
            yield Static("No data handy? Type demo:loans, demo:titanic, demo:shop or demo:shards",
                         classes="hint")
            yield Static(self._error, id="intake_error")

    def on_mount(self) -> None:
        self.query_one("#source", Input).focus()

    @on(Input.Submitted, "#source")
    def submitted(self, event: Input.Submitted) -> None:
        value = event.value.strip()
        if value:
            self.dismiss(value)

    def action_cancel(self) -> None:
        self.dismiss(None)


class ScoreFileScreen(ModalScreen[str | None]):
    """Ask for a new file: to score with the live model, or to train a challenger on."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, retrain: bool = False) -> None:
        super().__init__()
        self._retrain = retrain

    def compose(self) -> ComposeResult:
        with Vertical(id="intake_box"):
            if self._retrain:
                yield Static("Retrain with answers", id="intake_title")
                yield Static(
                    "Give a file with the training table's columns plus the real answers. A challenger is "
                    "trained on it and compared with the live model on rows neither has seen.",
                    id="intake_lead",
                )
            else:
                yield Static("Score a new file", id="intake_title")
                yield Static(
                    "Give a file with the same columns as the training table. The approved model is "
                    "applied to it. If the file also has the answers, the model is checked against them.",
                    id="intake_lead",
                )
            yield Input(placeholder="path/to/new_rows.csv", id="score_path")
            yield Static("csv, tsv or parquet · escape cancels", classes="hint")

    def on_mount(self) -> None:
        self.query_one("#score_path", Input).focus()

    @on(Input.Submitted, "#score_path")
    def submitted(self, event: Input.Submitted) -> None:
        value = event.value.strip().strip('"')
        if value:
            self.dismiss(value)

    def action_cancel(self) -> None:
        self.dismiss(None)


class MLTeamApp(App[None]):
    TITLE = "mlteam"
    SUB_TITLE = "an AutoML team you steer"

    CSS = """
    #body { height: 1fr; }
    #rail { width: 34; padding: 1 1 0 1; border-right: solid $panel; }
    #stages { height: auto; }
    .rail-title { margin-top: 1; text-style: bold; }
    #tables { height: auto; max-height: 12; border: none; background: transparent; padding: 0; }
    #main { width: 1fr; }
    #table_summary { height: auto; padding: 0 1 1 1; }
    #columns { height: 1fr; }
    #relations_scroll { padding: 1 2; }
    #contract_body { height: 1fr; }
    #form { width: 3fr; min-width: 50; padding: 0 1 1 1; }
    #why { width: 2fr; padding: 0 1 1 2; border-left: solid $panel; }
    #why Static { height: auto; margin-bottom: 1; }
    #answers { height: auto; margin-top: 1; padding: 0 1; border: round $accent; }
    .question-title { color: $accent; text-style: bold; height: auto; }
    .question { height: auto; margin-top: 1; }
    .pair { height: auto; }
    .cell { width: 1fr; height: auto; padding-right: 2; }
    .field { margin-top: 1; color: $text-muted; }
    #form Select { width: 100%; }
    #form Input { width: 100%; }
    #form SelectionList { height: auto; max-height: 10; }
    #dedupe { margin-top: 1; }
    .costs { height: auto; }
    .costs Input { width: 1fr; }
    .feedback { height: auto; padding: 1 1 0 1; border-top: solid $panel; }
    .feedback Input { width: 1fr; margin-right: 2; }
    #buttons { height: auto; padding: 1 1 0 1; }
    #buttons Button { margin-right: 2; }
    #approve_hint { width: 1fr; height: 3; content-align: left middle; color: $text-muted; }
    #features_body { height: 1fr; }
    #feat_left { width: 3fr; min-width: 50; padding: 0 1; }
    #feat_rule { height: auto; padding: 1 0; }
    #feature_list { height: 1fr; }
    #feat_right { width: 2fr; padding: 0 1 1 2; border-left: solid $panel; }
    #feat_right Static { height: auto; margin-bottom: 1; }
    #feat_buttons { height: auto; padding: 1 1 0 1; }
    #feat_buttons Button { margin-right: 2; }
    #feat_hint { width: 1fr; height: 3; content-align: left middle; color: $text-muted; }
    #tune_label { width: auto; height: 3; content-align: right middle; color: $text-muted; padding-right: 1; }
    #tune_seconds { width: 8; margin-top: 1; }
    #models_body { height: 1fr; }
    #model_left { width: 3fr; min-width: 50; padding: 0 1; }
    #model_summary { height: auto; padding: 1 0; }
    #model_table { height: 1fr; }
    #prediction { height: auto; max-height: 14; padding-top: 1; }
    #model_right { width: 2fr; padding: 0 1 1 2; border-left: solid $panel; }
    #model_right Static { height: auto; margin-bottom: 1; }
    #model_buttons { height: auto; padding: 1 1 0 1; }
    #model_buttons Button { margin-right: 2; }
    #model_hint { width: 1fr; height: 3; content-align: left middle; color: $text-muted; }
    #insights_scroll { padding: 1 2; }
    #monitor_scroll { padding: 1 2; }
    #monitor_scroll Static { height: auto; margin-bottom: 1; }
    #monitor_buttons { height: auto; padding: 1 1 0 1; border-top: solid $panel; }
    #monitor_buttons Button { margin-right: 2; }
    #teamlog { padding: 0 1; }
    #statusbar { height: 1; padding: 0 1; background: $panel; color: $text-muted; }
    IntakeScreen { align: center middle; }
    ScoreFileScreen { align: center middle; }
    #intake_box { width: 86; height: auto; padding: 1 3; border: round $accent; background: $surface; }
    #intake_title { text-style: bold; color: $accent; }
    #intake_lead { margin: 1 0; }
    .hint { color: $text-muted; }
    #intake_error { color: $error; margin-top: 1; height: auto; }
    """

    BINDINGS = [
        Binding("ctrl+s", "approve", "Approve"),
        Binding("ctrl+r", "reset", "Reset"),
        Binding("f1", "tab('data')", "Data"),
        Binding("f2", "tab('relations')", "Relations"),
        Binding("f3", "tab('contract')", "Contract"),
        Binding("f4", "tab('features')", "Features"),
        Binding("f5", "tab('models')", "Models"),
        Binding("f6", "tab('monitor')", "Monitor"),
        Binding("f7", "tab('team')", "Team"),
        Binding("f8", "tab('insights')", "Insights"),
        Binding("ctrl+n", "score_file", "Score a file"),
        Binding("ctrl+o", "open", "New data"),
        Binding("ctrl+q", "quit", "Quit"),
    ]

    def __init__(
        self,
        sources: list[str] | None = None,
        *,
        runs_dir: Path | None = None,
        brain: Brain | None = None,
        brain_note: str = "",
        stats_only: bool = False,
        approver: str = "human",
        fetch_kaggle: KaggleFetcher = fetch_from_kaggle,
        tune_budget: Budget | None = None,
        test_seconds: float = 30.0,
        resume: Path | None = None,
    ) -> None:
        super().__init__()
        self._resume = resume
        self._tune_budget = tune_budget if tune_budget is not None else Budget()
        self._test_seconds = test_seconds
        self._sources = sources or []
        self._runs_dir = runs_dir
        self._brain = brain
        self._brain_note = brain_note
        self._stats_only = stats_only
        self._approver = approver
        self._fetch_kaggle = fetch_kaggle
        self._ui_thread = threading.get_ident()
        self.team: Team | None = None
        self.current: DataContract | None = None
        self.proposal: DataContract | None = None
        self.issues: list[Issue] = []
        self._join_options: list[JoinStep] = []
        self._shown_table: str | None = None

    # -- layout ----------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="body"):
            with Vertical(id="rail"):
                yield Static(id="stages")
                yield Static("TABLES", classes="rail-title")
                yield OptionList(id="tables")
            with TabbedContent(id="main", initial="data"):
                with TabPane("Data", id="data"):
                    yield Static(id="table_summary")
                    yield DataTable(id="columns", zebra_stripes=True, cursor_type="row")
                with TabPane("Relations", id="relations"):
                    with VerticalScroll(id="relations_scroll"):
                        yield Static(id="relations_text")
                with TabPane("Contract", id="contract"):
                    with Horizontal(id="contract_body"):
                        with VerticalScroll(id="form"):
                            yield Vertical(id="answers")
                            with Horizontal(classes="pair"):
                                with Vertical(classes="cell"):
                                    yield Label("Table to predict from", classes="field")
                                    yield Select[str]([], id="base", allow_blank=True, compact=True,
                                                      prompt="choose a table")
                                with Vertical(classes="cell"):
                                    yield Label("Target: the column to predict", classes="field")
                                    yield Select[str]([], id="target", allow_blank=True, compact=True,
                                                      prompt="choose the target")
                            with Horizontal(classes="pair"):
                                with Vertical(classes="cell"):
                                    yield Label("Task", classes="field")
                                    yield Select[str]([], id="task", allow_blank=True, compact=True,
                                                      prompt="choose a task")
                                with Vertical(classes="cell"):
                                    yield Label("Metric: how we keep score", classes="field")
                                    yield Select[str]([], id="metric", allow_blank=True, compact=True,
                                                      prompt="choose a metric")
                            with Horizontal(classes="pair"):
                                with Vertical(classes="cell"):
                                    yield Label("Row key: different on every row", classes="field")
                                    yield Select[str]([], id="key", allow_blank=True, compact=True,
                                                      prompt="no key")
                                with Vertical(classes="cell"):
                                    yield Label("Time column: to split by date", classes="field")
                                    yield Select[str]([], id="time", allow_blank=True, compact=True,
                                                      prompt="no time column")
                            with Horizontal(classes="pair"):
                                with Vertical(classes="cell"):
                                    yield Label("One row means", classes="field")
                                    yield Input(id="grain", placeholder="one row = one ...", compact=True)
                                with Vertical(classes="cell"):
                                    yield Label("Predict N days after time column", classes="field")
                                    yield Input(id="horizon", type="integer", compact=True,
                                                placeholder="blank = no time rule")
                            with Horizontal(classes="pair"):
                                with Vertical(classes="cell"):
                                    yield Label("Rows that belong together", classes="field")
                                    yield Select[str]([], id="group", allow_blank=True, compact=True,
                                                      prompt="no groups")
                                with Vertical(classes="cell"):
                                    yield Label("Cost of a false alarm / a miss", classes="field")
                                    with Horizontal(classes="costs"):
                                        yield Input(id="cost_fa", type="number", compact=True, placeholder="—")
                                        yield Input(id="cost_miss", type="number", compact=True, placeholder="—")
                            yield Checkbox("Drop exact duplicate rows", id="dedupe", compact=True)
                            yield Label("Joins", classes="field", id="joins_label")
                            yield SelectionList[int](id="joins", compact=True)
                            yield Label("Kept out of the model (space toggles)", classes="field")
                            yield SelectionList[str](id="drops", compact=True)
                        with VerticalScroll(id="why"):
                            yield Static(id="checks")
                            yield Static(id="target_dist")
                            yield Static(id="reasons")
                            yield Static(id="review")
                            yield Static(render.learn(), id="learn")
                    with Horizontal(classes="feedback"):
                        yield Input(id="fb_contract", placeholder="Feedback for the agents, e.g. 'also try a ratio of visits to tickets'. Enter or Redo.", compact=True)
                        yield Button("Redo with feedback", id="redo_contract", compact=True)
                    with Horizontal(id="buttons"):
                        yield Button("Approve contract", id="approve", variant="success", disabled=True)
                        yield Button("Reset to proposal", id="reset")
                        yield Static("ctrl+s approves · each approval is saved as a new version",
                                     id="approve_hint")
                with TabPane("Insights", id="insights"):
                    with VerticalScroll(id="insights_scroll"):
                        yield Static(render.insights_view(None), id="insights_view")
                with TabPane("Features", id="features"):
                    with Horizontal(id="features_body"):
                        with Vertical(id="feat_left"):
                            yield Static("Approve the data contract first. Features are built from it.",
                                         id="feat_rule")
                            yield SelectionList[str](id="feature_list", compact=True)
                        with VerticalScroll(id="feat_right"):
                            yield Static(render.feature_detail(None), id="feat_detail")
                            yield Static(id="feat_review")
                            yield Static(render.learn_features(), id="feat_learn")
                    with Horizontal(classes="feedback"):
                        yield Input(id="fb_features", placeholder="Feedback for the agents, e.g. 'also try a ratio of visits to tickets'. Enter or Redo.", compact=True)
                        yield Button("Redo with feedback", id="redo_features", compact=True)
                    with Horizontal(id="feat_buttons"):
                        yield Button("Approve features", id="approve_features", variant="success", disabled=True)
                        yield Static("tuning budget, seconds", id="tune_label")
                        yield Input(str(int(self._tune_budget.seconds)), id="tune_seconds", type="integer",
                                    compact=True, tooltip="0 skips tuning")
                        yield Static("space toggles a feature · ctrl+s approves", id="feat_hint")
                with TabPane("Models", id="models"):
                    with Horizontal(id="models_body"):
                        with Vertical(id="model_left"):
                            yield Static("Approve the features first. Models are trained on them.",
                                         id="model_summary")
                            yield DataTable(id="model_table", zebra_stripes=True, cursor_type="row")
                            yield Static(id="prediction")
                        with VerticalScroll(id="model_right"):
                            yield Static(render.model_detail(None, None), id="model_detail")
                            yield Static(id="model_review")
                            yield Static(render.learn_models(), id="model_learn")
                    with Horizontal(classes="feedback"):
                        yield Input(id="fb_models", compact=True,
                                    placeholder="Feedback for the model plan, e.g. 'try deeper forests, "
                                                "skip linear'. Enter or Redo.")
                        yield Button("Redo with feedback", id="redo_models", compact=True)
                    with Horizontal(id="model_buttons"):
                        yield Button("Approve this model", id="approve_model", variant="success", disabled=True)
                        yield Static("move to a model · ctrl+s approves it", id="model_hint")
                with TabPane("Monitor", id="monitor"):
                    with VerticalScroll(id="monitor_scroll"):
                        yield Static(render.monitor_report([], None), id="monitor_report")
                        yield Static(id="challenge_report")
                    with Horizontal(id="monitor_buttons"):
                        yield Button("Score a new file", id="monitor_score")
                        yield Button("Retrain with answers", id="monitor_retrain")
                        yield Button("Promote challenger", id="monitor_promote", variant="success", disabled=True)
                with TabPane("Team", id="team"):
                    yield RichLog(id="teamlog", wrap=True, markup=False)
        yield Static(id="statusbar")
        yield Footer()

    def on_mount(self) -> None:
        self._ui_thread = threading.get_ident()
        self.query_one("#columns", DataTable).add_columns(
            "Column", "Role", "Type", "Kind", "Missing", "Distinct", "Values"
        )
        self.query_one("#model_table", DataTable).add_columns(
            "Model", "Kind", "Cross-val", "Unseen rows", "Trained-on", "Verdict"
        )
        self._refresh_rail()
        self._refresh_status()
        if self._resume is not None:
            self.resume_pipeline(self._resume)
        elif self._sources:
            self.run_pipeline(self._sources)
        else:
            self._ask_for_source()

    def on_unmount(self) -> None:
        if self.team is not None:
            self.team.close()

    # -- intake ----------------------------------------------------------

    def _ask_for_source(self, error: str = "") -> None:
        def chosen(value: str | None) -> None:
            if value is None:
                if self.team is None or self.team.profile is None:
                    self.exit()
                return
            self.run_pipeline([part.strip() for part in value.split(";") if part.strip()])

        self.push_screen(IntakeScreen(error), chosen)

    def action_open(self) -> None:
        self._ask_for_source()

    # -- pipeline --------------------------------------------------------

    @work(exclusive=True, group="pipeline")
    async def run_pipeline(self, sources: list[str]) -> None:
        if self.team is not None:
            self.team.close()
        run = new_run(self._runs_dir)
        self.team = Team(
            run,
            brain=self._brain,
            brain_note=self._brain_note,
            stats_only=self._stats_only,
            emit=self.on_team_event,
            fetch_kaggle=self._fetch_kaggle,
            tune_budget=Budget(**vars(self._tune_budget)),
            test_seconds=self._test_seconds,
        )
        self.current = None
        self.proposal = None
        self.query_one("#teamlog", RichLog).clear()
        self._clear_features("Approve the data contract first. Features are built from it.")
        self._clear_models("Approve the features first. Models are trained on them.")
        self.query_one("#insights_view", Static).update(render.insights_view(None))
        self._refresh_rail()
        self._refresh_status()
        try:
            resolved = samples.resolve_specs(sources, run.root / "demo")
            await self.team.prepare(resolved)
        except (IntakeError, StoreError, ValueError) as exc:
            self._ask_for_source(str(exc))
            return
        self._show_tables()
        self._refresh_status()
        assert self.team.contract is not None
        self.proposal = copy.deepcopy(self.team.contract)
        await self._load_form(self.team.contract, new_base=True)
        if self._brain is None:
            self._log(Text(
                "No model connected, so the contract comes from fixed rules only. "
                "To add the Analyst and the Skeptic: run `tau`, then /login.", style="yellow"))
        await self.team.consult()
        assert self.team.contract is not None
        self.proposal = copy.deepcopy(self.team.contract)
        await self._load_form(self.team.contract, new_base=True)
        self.query_one("#main", TabbedContent).active = "contract"

    @work(exclusive=True, group="pipeline")
    async def resume_pipeline(self, root: Path) -> None:
        """Open a run where it stopped: every tab filled in from what was saved."""
        try:
            self.team = await asyncio.to_thread(
                Team.resume, root, brain=self._brain, brain_note=self._brain_note, stats_only=self._stats_only,
                emit=self.on_team_event, fetch_kaggle=self._fetch_kaggle,
                tune_budget=Budget(**vars(self._tune_budget)), test_seconds=self._test_seconds,
            )
        except (ValueError, StoreError, OSError) as exc:
            self._ask_for_source(f"Could not resume {root}: {exc}")
            return
        team = self.team
        self._show_tables()
        assert team.contract is not None
        self.proposal = copy.deepcopy(team.draft or team.contract)
        await self._load_form(team.contract, new_base=True)
        if team.status.get("gate") == "done":
            self.query_one("#approve", Button).label = f"Approved v{team.contract.version} · approve again"
        self._refresh_rail()
        self._refresh_status()
        if team.status.get("gate") != "done":
            self.query_one("#main", TabbedContent).active = "contract"
        elif team.status.get("feature_gate") not in {"waiting", "done"} or not team.features:
            self.run_features()
        else:
            self._show_features()
            self.query_one("#insights_view", Static).update(render.insights_view(team.insights, self._contract_note()))
            if team.feature_set is not None and team.status.get("feature_gate") == "done":
                self.query_one("#approve_features", Button).label = f"Approved v{team.feature_set.version}"
            if team.status.get("feature_gate") != "done":
                self.query_one("#main", TabbedContent).active = "features"
            elif team.training is None:
                self.run_training()
            else:
                self._show_models()
                self._refresh_monitor()
                self.query_one("#main", TabbedContent).active = (
                    "monitor" if team.status.get("model_gate") == "done" else "models")
        self.notify(f"Picked up at {team.where()}.", title=f"Resumed {team.run.run_id}", timeout=8)

    # -- feedback --------------------------------------------------------

    @on(Button.Pressed, "#redo_contract")
    @on(Input.Submitted, "#fb_contract")
    def redo_contract_pressed(self) -> None:
        self._redo("contract", "#fb_contract")

    @on(Button.Pressed, "#redo_features")
    @on(Input.Submitted, "#fb_features")
    def redo_features_pressed(self) -> None:
        self._redo("features", "#fb_features")

    @on(Button.Pressed, "#redo_models")
    @on(Input.Submitted, "#fb_models")
    def redo_models_pressed(self) -> None:
        self._redo("models", "#fb_models")

    def _redo(self, step: str, field: str) -> None:
        box = self.query_one(field, Input)
        text = box.value.strip()
        if self.team is None:
            return
        if not text:
            self.notify("Write what you want changed first.", severity="warning")
            return
        ready = {"contract": self.team.status.get("gate") in {"waiting", "done"},
                 "features": self.team.status.get("feature_gate") in {"waiting", "done"},
                 "models": self.team.status.get("model_gate") in {"waiting", "done"}}[step]
        if not ready:
            self.notify("That step has not reached its gate yet.", severity="warning")
            return
        box.value = ""
        self.redo_step(step, text)

    @work(exclusive=True, group="redo")
    async def redo_step(self, step: str, text: str) -> None:
        assert self.team is not None
        if self.team.brain is None:
            self.notify("No model is connected, so only the rules run again. Run `tau`, then /login, for agents.",
                        severity="warning", timeout=8)
        if step == "contract":
            self._clear_features("Approve the data contract first. Features are built from it.")
            self._clear_models("Approve the features first. Models are trained on them.")
            await self.team.redo("contract", text)
            assert self.team.contract is not None
            self.proposal = copy.deepcopy(self.team.contract)
            await self._load_form(self.team.contract, new_base=True)
            self.query_one("#approve", Button).label = "Approve contract"
            self.query_one("#main", TabbedContent).active = "contract"
        elif step == "features":
            self._clear_features("The Engineer is redoing the features with your feedback...")
            self._clear_models("Approve the features first. Models are trained on them.")
            try:
                await self.team.redo("features", text)
            except (ValueError, StoreError) as exc:
                self._clear_features(f"Features could not be built: {exc}")
                return
            self._show_features()
            self.query_one("#insights_view", Static).update(
                render.insights_view(self.team.insights, self._contract_note()))
            self.query_one("#main", TabbedContent).active = "features"
        else:
            self._clear_models("Redoing the model plan with your feedback...")
            try:
                await self.team.redo("models", text)
            except (ValueError, StoreError) as exc:
                self._clear_models(f"Training could not run: {exc}")
                return
            self._show_models()
            self.query_one("#main", TabbedContent).active = "models"
        self._refresh_rail()
        self._refresh_status()

    def on_team_event(self, event: TeamEvent) -> None:
        """Called by the harness. May arrive from a worker thread."""
        if threading.get_ident() != self._ui_thread:
            self.call_from_thread(self._handle_team_event, event)
        else:
            self._handle_team_event(event)

    def _handle_team_event(self, event: TeamEvent) -> None:
        if event.kind == "tune_trial":
            data = event.data
            best = data.get("best_so_far")
            line = Text()
            line.append("Tuning on a budget... ", style="bold")
            line.append(f"try {data['number'] + 1} ({data['family']}, {data['state']})  ·  "
                        f"best so far {'-' if best is None else f'{best:.3f}'}  ·  {data['elapsed']:.0f} s spent")
            self.query_one("#model_summary", Static).update(line)
            if data.get("improved") and best is not None:
                self._log(Text(f"   tuning: new best {best:.3f} at try {data['number'] + 1} ({data['family']})",
                               style="magenta"))
            return
        self._refresh_rail()
        if event.kind == "stage":
            status = event.data.get("status", "")
            if status in {"done", "failed", "skipped", "waiting"}:
                icon, style = render.STATUS_STYLE.get(status, ("", ""))
                line = Text()
                line.append(f"{icon} {STAGE_LABELS.get(event.stage, event.stage)}", style=style)
                if status == "waiting":
                    line.append("  waiting for you", style="bold cyan")
                elif event.text:
                    line.append(f"  {event.text}", style="dim")
                self._log(line)
        elif event.kind == "agent_text":
            line = Text()
            line.append(f"{event.agent}: ", style=AGENT_STYLE.get(event.agent, "bold"))
            line.append(event.text)
            self._log(line)
        elif event.kind == "tool_start":
            line = Text(f"   → {event.agent} runs {event.data.get('tool', '')}", style="dim")
            if event.text:
                line.append(f"  {event.text}", style="dim italic")
            self._log(line)
        elif event.kind == "tool_end":
            failed = bool(event.data.get("is_error"))
            line = Text("   ← ", style="dim")
            line.append("rejected: " if failed else "", style="bold red")
            line.append(event.text, style="red" if failed else "dim")
            self._log(line)
        elif event.kind == "note":
            self._log(Text(f"! {event.text}", style="yellow"))
        elif event.kind == "error":
            self._log(Text(f"✘ {event.agent or event.stage}: {event.text}", style="bold red"))

    def _log(self, line: Text) -> None:
        self.query_one("#teamlog", RichLog).write(line)

    def _refresh_rail(self) -> None:
        status = self.team.status if self.team is not None else {}
        self.query_one("#stages", Static).update(render.rail(status))

    def _refresh_status(self) -> None:
        parts = [f"model: {self._brain.label}" if self._brain else "model: none (rules only)"]
        if self._stats_only:
            parts.append("stats only: no raw rows leave this machine")
        if self.team is not None:
            parts.append(f"run {self.team.run.run_id}")
            if self.team.manifest is not None:
                parts.append(f"data {self.team.manifest.dataset_hash[:10]}")
            if self.current is not None and self.current.version:
                parts.append(f"contract v{self.current.version} approved")
        self.query_one("#statusbar", Static).update(" · ".join(parts))

    # -- data tab --------------------------------------------------------

    def _show_tables(self) -> None:
        assert self.team is not None and self.team.profile is not None
        tables = self.query_one("#tables", OptionList)
        tables.clear_options()
        for table in self.team.profile.tables:
            tables.add_option(Option(f"{table.name}  {table.rows:,}", id=table.name))
        if self.team.profile.tables:
            names = [t.name for t in self.team.profile.tables]
            base = self.team.contract.base_table if self.team.contract is not None else names[0]
            index = names.index(base) if base in names else 0
            self._shown_table = names[index]
            tables.highlighted = index
            self._show_table(self._shown_table)

    @on(OptionList.OptionHighlighted, "#tables")
    def table_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if event.option.id:
            self._show_table(event.option.id)

    def _show_table(self, name: str) -> None:
        if self.team is None or self.team.profile is None:
            return
        table = self.team.profile.table(name)
        if table is None:
            return
        self._shown_table = name
        self.query_one("#table_summary", Static).update(render.table_summary(table))
        grid = self.query_one("#columns", DataTable)
        grid.clear()
        for col in table.columns:
            role, style = render.column_role(col, self.current, name)
            grid.add_row(
                Text(col.name, style="bold" if role == "TARGET" else ""),
                Text(role, style=style),
                col.dtype.lower(),
                col.kind,
                render.missing_cell(col),
                f"{col.distinct:,}",
                render.values_cell(col),
            )

    # -- contract form ---------------------------------------------------

    async def _load_form(self, contract: DataContract, *, new_base: bool = False) -> None:
        """Fill the form from ``contract`` without the widgets echoing change events back."""
        assert self.team is not None and self.team.profile is not None and self.team.layout is not None
        profile, layout = self.team.profile, self.team.layout
        self.current = copy.deepcopy(contract)
        if new_base:
            # Offer every join the rules consider safe, not only the ones in this
            # contract, so a join an agent removed can be put back by hand.
            candidates, _ = candidate_joins(layout, contract.base_table, contract.stack)
            attach_time_columns(candidates, profile)
            offered = {self._join_key(j) for j in candidates}
            self._join_options = candidates + [j for j in contract.joins if self._join_key(j) not in offered]
        base = profile.table(contract.base_table)
        columns = base.columns if base is not None else []
        hidden = {layout.test, layout.submission, *[n for g in layout.shard_groups for n in g[1:]]}

        with self.prevent(Select.Changed, Input.Changed, SelectionList.SelectedChanged, Checkbox.Changed):
            base_select = self.query_one("#base", Select)
            base_select.set_options(
                [(f"{t.name}  ·  {t.rows:,} rows", t.name) for t in profile.tables if t.name not in hidden]
            )
            base_select.value = contract.base_table

            target = self.query_one("#target", Select)
            target.set_options(
                [(f"{c.name}  ·  {c.kind}", c.name)
                 for c in columns if c.kind not in {"empty", "constant"}]
            )
            target.value = contract.target if contract.target else Select.NULL

            task = self.query_one("#task", Select)
            task.set_options([(t, t) for t in TASKS])
            task.value = contract.task if contract.task in TASKS else Select.NULL

            self._set_metric_options(contract.task, contract.metric)
            self.query_one("#grain", Input).value = contract.grain
            self.query_one("#dedupe", Checkbox).value = contract.dedupe
            group = self.query_one("#group", Select)
            base = self.team.profile.table(contract.base_table) if self.team and self.team.profile else None
            names = [n for n in (base.column_names if base else []) if n != contract.target]
            group.set_options([(n, n) for n in names])
            group.value = contract.group_column if contract.group_column in names else Select.NULL
            self.query_one("#cost_fa", Input).value = "" if contract.cost_false_alarm is None \
                else f"{contract.cost_false_alarm:g}"
            self.query_one("#cost_miss", Input).value = "" if contract.cost_miss is None else f"{contract.cost_miss:g}"
            self.query_one("#horizon", Input).value = (
                "" if contract.horizon_days is None else str(contract.horizon_days)
            )

            key = self.query_one("#key", Select)
            key.set_options(
                [(f"{c.name}{'  ·  unique' if c.unique else ''}", c.name) for c in columns]
            )
            key.value = contract.grain_keys[0] if contract.grain_keys else Select.NULL

            time = self.query_one("#time", Select)
            time_cols = [c for c in columns if c.kind == "datetime" or "time_name" in c.hints]
            time.set_options([(f"{c.name}  ·  {c.kind}", c.name) for c in time_cols])
            names = {c.name for c in time_cols}
            time.value = contract.time_column if contract.time_column in names else Select.NULL

            joins = self.query_one("#joins", SelectionList)
            joins.display = bool(self._join_options)
            self.query_one("#joins_label", Label).display = bool(self._join_options)
            joins.clear_options()
            chosen = {self._join_key(j) for j in contract.joins}
            joins.add_options([
                Selection(step.label(), i, self._join_key(step) in chosen)
                for i, step in enumerate(self._join_options)
            ])

            drops = self.query_one("#drops", SelectionList)
            drops.clear_options()
            suspects = {s.column for s in contract.leakage_suspects}
            drops.add_options([
                Selection(
                    f"{c.name}  ·  " + ("possible leak" if c.name in suspects else c.kind),
                    c.name,
                    c.name in contract.drop_columns,
                )
                for c in columns if c.name != contract.target
            ])

        # One input per question the team could not settle from the data.
        box = self.query_one("#answers", Vertical)
        await box.remove_children()
        box.display = bool(contract.open_questions)
        if contract.open_questions:
            widgets: list[Static | Input] = [
                Static("QUESTIONS FOR YOU  ·  answer each one to unlock approval", classes="question-title")
            ]
            for i, question in enumerate(contract.open_questions):
                widgets.append(Static(f"? {question}", classes="question"))
                widgets.append(Input(
                    value=contract.answers.get(question, ""), id=f"answer_{i}", classes="answer",
                    compact=True, placeholder="type your answer",
                ))
            await box.mount(*widgets)
        await self._revalidate()

    def _set_metric_options(self, task: str | None, metric: str | None) -> None:
        select = self.query_one("#metric", Select)
        options = METRICS.get(task or "", ())
        select.set_options([(m, m) for m in options])
        select.value = metric if metric in options else Select.NULL

    @staticmethod
    def _join_key(step: JoinStep) -> tuple[str, str, str, str]:
        return (step.table, step.kind, step.base_key, step.other_key)

    @staticmethod
    def _value(select: Select) -> str | None:
        value = select.value
        return None if value is Select.NULL or value is None else str(value)

    def _read_form(self) -> None:
        """Copy the simple fields from the widgets into ``self.current``."""
        if self.current is None:
            return
        self.current.metric = self._value(self.query_one("#metric", Select))
        self.current.grain = self.query_one("#grain", Input).value.strip()
        self.current.dedupe = self.query_one("#dedupe", Checkbox).value
        self.current.group_column = self._value(self.query_one("#group", Select))
        self.current.cost_false_alarm = _number(self.query_one("#cost_fa", Input).value)
        self.current.cost_miss = _number(self.query_one("#cost_miss", Input).value)
        days = self.query_one("#horizon", Input).value.strip()
        self.current.horizon_days = int(days) if days.isdigit() else None
        key = self._value(self.query_one("#key", Select))
        self.current.grain_keys = [key] if key else []
        self.current.time_column = self._value(self.query_one("#time", Select))
        picked = set(self.query_one("#joins", SelectionList).selected)
        self.current.joins = [step for i, step in enumerate(self._join_options) if i in picked]
        self.current.drop_columns = [str(v) for v in self.query_one("#drops", SelectionList).selected]
        # Keep an answer we already hold if its input is not on screen right now.
        answers = dict(self.current.answers)
        for i, question in enumerate(self.current.open_questions):
            for field in self.query(f"#answer_{i}").results(Input):
                answers[question] = field.value.strip()
        self.current.answers = {q: a for q, a in answers.items() if q in self.current.open_questions}

    @on(Select.Changed, "#base")
    async def base_changed(self, event: Select.Changed) -> None:
        value = self._value(event.select)
        if self.team is None or self.current is None or value is None or value == self.current.base_table:
            return
        proposal = await asyncio.to_thread(self.team.repropose, value)
        await self._load_form(proposal, new_base=True)

    @on(Select.Changed, "#target")
    async def target_changed(self, event: Select.Changed) -> None:
        value = self._value(event.select)
        if self.team is None or self.current is None or value == self.current.target:
            return
        self._read_form()
        assert self.team.store is not None and self.team.profile is not None
        updated = await asyncio.to_thread(retarget, self.team.store, self.team.profile, self.current, value)
        await self._load_form(updated)

    @on(Select.Changed, "#task")
    async def task_changed(self, event: Select.Changed) -> None:
        value = self._value(event.select)
        if self.current is None or value == self.current.task:
            return
        self.current.task = value
        distinct = 2
        if self.team is not None and self.team.profile is not None and self.current.target:
            base = self.team.profile.table(self.current.base_table)
            col = base.column(self.current.target) if base is not None else None
            distinct = col.distinct if col is not None else 2
        metric = default_metric(value, distinct, None)[0] if value else None
        with self.prevent(Select.Changed):
            self._set_metric_options(value, metric)
        self._form_changed()

    @on(Select.Changed, "#metric")
    @on(Select.Changed, "#key")
    @on(Select.Changed, "#time")
    @on(Input.Changed, "#grain")
    @on(Input.Changed, "#horizon")
    @on(Checkbox.Changed, "#dedupe")
    @on(Select.Changed, "#group")
    @on(Input.Changed, "#cost_fa")
    @on(Input.Changed, "#cost_miss")
    @on(Input.Changed, ".answer")
    @on(SelectionList.SelectedChanged, "#joins")
    @on(SelectionList.SelectedChanged, "#drops")
    def field_changed(self) -> None:
        self._form_changed()

    def _form_changed(self) -> None:
        if self.current is None:
            return
        self._read_form()
        self.revalidate_later()

    @work(exclusive=True, group="validate")
    async def revalidate_later(self) -> None:
        await self._revalidate()

    async def _revalidate(self) -> None:
        """Ask the harness to check the contract as it stands, then redraw everything that depends on it."""
        if self.team is None or self.current is None or self.team.store is None:
            return
        team, contract = self.team, copy.deepcopy(self.current)
        assert team.profile is not None and team.layout is not None
        issues = await asyncio.to_thread(team.validate, contract)
        stats = await asyncio.to_thread(target_stats, team.store, contract)
        self.issues = issues
        self.query_one("#checks", Static).update(render.checks(issues))
        self.query_one("#target_dist", Static).update(render.target_distribution(contract, stats))
        self.query_one("#reasons", Static).update(render.reasons(contract))
        skipped = team.brain_note or "Not run: no model connected."
        if team.brain is not None:
            skipped = "The Skeptic has not reviewed this yet." if team.status.get("skeptic") != "failed" \
                else "The Skeptic did not finish its review."
        self.query_one("#review", Static).update(render.review(team.review, skipped))
        self.query_one("#relations_text", Static).update(
            render.relations(team.profile, team.layout, contract, issues)
        )
        gate_open = team.status.get("gate") in {"waiting", "done"}
        self.query_one("#approve", Button).disabled = has_errors(issues) or not gate_open
        if self._shown_table:
            self._show_table(self._shown_table)

    # -- the gate --------------------------------------------------------

    @on(Button.Pressed, "#approve")
    async def approve_pressed(self) -> None:
        await self.action_approve()

    @on(Button.Pressed, "#reset")
    async def reset_pressed(self) -> None:
        await self.action_reset()

    async def action_approve(self) -> None:
        """ctrl+s approves whichever gate you are looking at."""
        active = self.query_one("#main", TabbedContent).active
        if active == "features":
            await self.approve_features()
            return
        if active == "models":
            await self.approve_model()
            return
        if self.team is None or self.current is None:
            return
        if self.team.status.get("gate") not in {"waiting", "done"}:
            self.notify("The team is still working. Approve once the gate opens.", severity="warning")
            return
        self._read_form()
        contract = copy.deepcopy(self.current)
        try:
            path = await asyncio.to_thread(self.team.approve, contract, self._approver)
        except ValueError as exc:
            self.notify(str(exc), title="Not approved", severity="error", timeout=8)
            await self._revalidate()
            return
        self.current = contract
        self.query_one("#approve", Button).label = f"Approved v{contract.version} · approve again"
        self._refresh_status()
        self.notify(
            f"Saved to {path.name}. Building features from it now.",
            title=f"Contract v{contract.version} approved",
            timeout=8,
        )
        self._clear_models("Approve the features first. Models are trained on them.")
        self.run_features()

    # -- features --------------------------------------------------------

    def _clear_features(self, message: str) -> None:
        self.query_one("#feature_list", SelectionList).clear_options()
        self.query_one("#feat_rule", Static).update(message)
        self.query_one("#feat_detail", Static).update(render.feature_detail(None))
        self.query_one("#feat_review", Static).update("")
        self.query_one("#approve_features", Button).disabled = True
        self.query_one("#approve_features", Button).label = "Approve features"

    @work(exclusive=True, group="features")
    async def run_features(self) -> None:
        if self.team is None:
            return
        self._clear_features("Building features...")
        try:
            await self.team.build_features()
        except (ValueError, StoreError) as exc:
            self._clear_features(f"Features could not be built: {exc}")
            return
        self._show_features()
        self.query_one("#insights_view", Static).update(render.insights_view(self.team.insights, self._contract_note()))
        self.query_one("#main", TabbedContent).active = "features"

    def _show_features(self) -> None:
        assert self.team is not None and self.team.contract is not None
        team = self.team
        usable = sum(1 for f in team.features if f.usable)
        rule = Text()
        rule.append(time_rule_summary(team.contract) + "\n", style="bold")
        rule.append(f"{usable} usable features from contract v{team.contract.version}. "
                    "Ticked ones go to training.", style="dim")
        if team.feature_tests is not None:
            rule.append("\n")
            rule.append(render.feature_tests_summary(team.feature_tests))
        self.query_one("#feat_rule", Static).update(rule)
        listing = self.query_one("#feature_list", SelectionList)
        with self.prevent(SelectionList.SelectedChanged):
            listing.clear_options()
            listing.add_options([
                Selection(render.feature_label(f, self._feature_test(f.name)), f.name, f.default_on,
                          disabled=not f.usable)
                for f in team.features
            ])
        skipped = team.brain_note or "Not run: no model connected."
        if team.brain is not None:
            skipped = "The Skeptic did not finish its review."
        self.query_one("#feat_review", Static).update(render.review(team.feature_review, skipped))
        if team.features:
            listing.highlighted = 0
            first = team.features[0]
            self.query_one("#feat_detail", Static).update(
                render.feature_detail(first, self._feature_test(first.name)))
        self._refresh_feature_hint()

    def _contract_note(self) -> str:
        if self.team is None or self.team.contract is None:
            return ""
        notes = []
        if self.team.contract.dedupe:
            notes.append("Exact duplicate rows were dropped by the contract.")
        return " ".join(notes)

    def _feature_test(self, name: object):  # -> FeatureTest | None
        if self.team is None or self.team.feature_tests is None:
            return None
        return self.team.feature_tests.by_name().get(str(name))

    def _feature(self, name: object) -> Feature | None:
        if self.team is None:
            return None
        return next((f for f in self.team.features if f.name == name), None)

    def _refresh_feature_hint(self) -> None:
        listing = self.query_one("#feature_list", SelectionList)
        picked = len(listing.selected)
        gate_open = self.team is not None and self.team.status.get("feature_gate") in {"waiting", "done"}
        self.query_one("#approve_features", Button).disabled = not gate_open or picked == 0
        self.query_one("#feat_hint", Static).update(
            f"{picked} of {listing.option_count} ticked · space toggles · ctrl+s approves"
        )

    @on(SelectionList.SelectionHighlighted, "#feature_list")
    def feature_highlighted(self, event: SelectionList.SelectionHighlighted) -> None:
        self.query_one("#feat_detail", Static).update(
            render.feature_detail(self._feature(event.selection.value), self._feature_test(event.selection.value)))

    @on(SelectionList.SelectedChanged, "#feature_list")
    def features_picked(self) -> None:
        self._refresh_feature_hint()

    @on(Button.Pressed, "#approve_features")
    async def approve_features_pressed(self) -> None:
        await self.approve_features()

    async def approve_features(self) -> None:
        if self.team is None or self.team.status.get("feature_gate") not in {"waiting", "done"}:
            self.notify("Features are not ready to approve yet.", severity="warning")
            return
        picked = {str(v) for v in self.query_one("#feature_list", SelectionList).selected}
        ordered = [f.name for f in self.team.features if f.name in picked]
        seconds = self.query_one("#tune_seconds", Input).value.strip()
        self.team.tune_budget.seconds = float(seconds) if seconds.isdigit() else self._tune_budget.seconds
        try:
            path = await asyncio.to_thread(self.team.approve_features, ordered, self._approver)
        except ValueError as exc:
            self.notify(str(exc), title="Not approved", severity="error", timeout=8)
            return
        assert self.team.feature_set is not None
        version = self.team.feature_set.version
        self.query_one("#approve_features", Button).label = f"Approved v{version} · approve again"
        leaks = [n for n in ordered if (f := self._feature(n)) is not None and f.leak]
        note = f"Saved to {path.name}. Training the models on them now."
        if leaks:
            note += f" You kept {len(leaks)} leak-flagged feature(s): {', '.join(leaks)}."
        self.notify(note, title=f"Features v{version} approved: {len(ordered)} features", timeout=10)
        self.run_training()

    def action_score_file(self) -> None:
        if self.team is None or self.team.model_card is None:
            self.notify("Approve a model first. Then any new file can be scored with it.", severity="warning")
            return

        def chosen(path: str | None) -> None:
            if path:
                self.score_file(path)

        self.push_screen(ScoreFileScreen(), chosen)

    @on(Button.Pressed, "#monitor_score")
    def monitor_score_pressed(self) -> None:
        self.action_score_file()

    @on(Button.Pressed, "#monitor_retrain")
    def monitor_retrain_pressed(self) -> None:
        if self.team is None or self.team.model_card is None:
            self.notify("Approve a model first.", severity="warning")
            return

        def chosen(path: str | None) -> None:
            if path:
                self.retrain_file(path)

        self.push_screen(ScoreFileScreen(retrain=True), chosen)

    @work(exclusive=True, group="scoring")
    async def retrain_file(self, path: str) -> None:
        if self.team is None:
            return
        panel = self.query_one("#challenge_report", Static)
        self.query_one("#main", TabbedContent).active = "monitor"
        panel.update(Text("Training a challenger and comparing it with the live model...", style="dim"))
        try:
            result = await asyncio.to_thread(self.team.challenge, path)
        except ValueError as exc:
            panel.update(Text(f"No challenger: {exc}", style="red"))
            return
        self._challenge = result
        panel.update(render.challenge_report(result))
        self.query_one("#monitor_promote", Button).disabled = result.verdict == "worse"

    @on(Button.Pressed, "#monitor_promote")
    async def monitor_promote_pressed(self) -> None:
        result = getattr(self, "_challenge", None)
        if self.team is None or result is None:
            return
        try:
            path = await asyncio.to_thread(self.team.promote, result.version, self._approver)
        except ValueError as exc:
            self.notify(str(exc), title="Not promoted", severity="error", timeout=8)
            return
        result.promoted_as = int(self.team.model_card["version"])
        self.query_one("#challenge_report", Static).update(render.challenge_report(result))
        self.query_one("#monitor_promote", Button).disabled = True
        self._refresh_monitor()
        self.notify(f"Saved to {path.name}. New files are now scored with it.",
                    title=f"Challenger v{result.version} is live", timeout=8)

    def _refresh_monitor(self) -> None:
        if self.team is None:
            return
        self.query_one("#monitor_report", Static).update(
            render.monitor_report(self.team.history(), self.team.model_card))

    @work(exclusive=True, group="scoring")
    async def score_file(self, path: str) -> None:
        if self.team is None:
            return
        panel = self.query_one("#prediction", Static)
        self.query_one("#main", TabbedContent).active = "models"
        try:
            result = await asyncio.to_thread(self.team.predict_file, path)
        except ValueError as exc:
            panel.update(Text(f"That file could not be scored: {exc}", style="red"))
            return
        panel.update(render.prediction(result))
        self._refresh_monitor()

    # -- models ----------------------------------------------------------

    def _clear_models(self, message: str) -> None:
        self.query_one("#model_table", DataTable).clear()
        self.query_one("#model_summary", Static).update(message)
        self.query_one("#model_detail", Static).update(render.model_detail(None, None))
        self.query_one("#model_review", Static).update("")
        self.query_one("#prediction", Static).update("")
        button = self.query_one("#approve_model", Button)
        button.disabled = True
        button.label = "Approve this model"

    @work(exclusive=True, group="training")
    async def run_training(self) -> None:
        if self.team is None:
            return
        self._clear_models("Training and scoring every candidate...")
        try:
            await self.team.train_models()
        except (ValueError, StoreError) as exc:
            self._clear_models(f"Training could not run: {exc}")
            return
        self._show_models()
        self.query_one("#main", TabbedContent).active = "models"

    def _show_models(self) -> None:
        assert self.team is not None and self.team.training is not None
        team, run = self.team, self.team.training
        self.query_one("#model_summary", Static).update(render.model_summary(run, team.tuning))
        table = self.query_one("#model_table", DataTable)
        table.clear()
        for result in run.results:
            table.add_row(*render.model_cells(result, run), key=result.name)
        skipped = team.brain_note or "Not run: no model connected."
        if team.brain is not None:
            skipped = "The Skeptic did not finish its review."
        self.query_one("#model_review", Static).update(render.review(team.model_review, skipped))
        best = next((i for i, r in enumerate(run.results) if r.recommended), 0)
        if run.results:
            table.move_cursor(row=best)
            self._show_model(run.results[best].name)

    def _show_model(self, name: str | None) -> None:
        if self.team is None or self.team.training is None or name is None:
            return
        run = self.team.training
        result = run.result(name)
        self._model_name = name
        self.query_one("#model_detail", Static).update(render.model_detail(result, run, self.team.tuning))
        gate_open = self.team.status.get("model_gate") in {"waiting", "done"}
        choosable = result is not None and not result.error and result.beats_baseline
        self.query_one("#approve_model", Button).disabled = not (gate_open and choosable)
        hint = "ctrl+s approves this model"
        if result is not None and result.family == "baseline":
            hint = "the baseline is the bar to beat, not a model to ship"
        elif result is not None and not choosable:
            hint = "this one cannot be approved: it did not beat the baseline"
        self.query_one("#model_hint", Static).update(hint)

    @on(DataTable.RowHighlighted, "#model_table")
    def model_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.row_key is not None and event.row_key.value is not None:
            self._show_model(str(event.row_key.value))

    @on(Button.Pressed, "#approve_model")
    async def approve_model_pressed(self) -> None:
        await self.approve_model()

    async def approve_model(self) -> None:
        name = getattr(self, "_model_name", None)
        if self.team is None or name is None or self.team.status.get("model_gate") not in {"waiting", "done"}:
            self.notify("No model is ready to approve yet.", severity="warning")
            return
        try:
            path = await asyncio.to_thread(self.team.approve_model, name, self._approver)
        except ValueError as exc:
            self.notify(str(exc), title="Not approved", severity="error", timeout=8)
            return
        assert self.team.model_card is not None
        version = self.team.model_card["version"]
        self.query_one("#approve_model", Button).label = f"Approved {name} (v{version})"
        report = self.team.report_path
        self.notify(
            f"Saved to {path.name}." + (f" Results report: {report}" if report else ""),
            title=f"Model v{version} approved: {name}", timeout=12,
        )
        self.query_one("#prediction", Static).update("")
        try:
            result = await asyncio.to_thread(self.team.predict)
        except ValueError as exc:
            self.query_one("#prediction", Static).update(
                Text(f"The test file could not be scored: {exc}", style="red"))
            return
        self.query_one("#prediction", Static).update(
            render.prediction(result, "This data has no held-back test file to score."))
        self._refresh_monitor()

    async def action_reset(self) -> None:
        if self.proposal is not None:
            await self._load_form(self.proposal, new_base=True)

    def action_tab(self, name: str) -> None:
        self.query_one("#main", TabbedContent).active = name


def _number(text: str) -> float | None:
    try:
        value = float(text.strip())
    except ValueError:
        return None
    return value if value == value else None
