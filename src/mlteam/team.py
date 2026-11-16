"""The harness: it fixes the order of the stages and the rules. Agents fill in
the work; a person decides at the gate.

    intake -> load -> profile -> draft -> Analyst -> Skeptic -> [you] data contract
           -> draft features -> understand the data -> Engineer -> Skeptic -> test features -> [you] feature review
           -> training plan -> tune on a budget -> train and score -> Skeptic -> [you] model review
           -> score the held-back test file

No frontend code lives here. A frontend passes an ``emit`` callback and renders
the ``TeamEvent`` stream, which is also written to the run's trace file.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mlteam.agents.brain import Brain
from mlteam.agents.roles import (
    ANALYST_SYSTEM,
    ENGINEER_SYSTEM,
    FEATURE_SKEPTIC_SYSTEM,
    MODEL_SKEPTIC_SYSTEM,
    SKEPTIC_SYSTEM,
    STATS_ONLY_NOTE,
    TRAINER_SYSTEM,
    Emit,
    TeamEvent,
    analyst_prompt,
    analyst_tools,
    engineer_prompt,
    engineer_tools,
    feature_skeptic_prompt,
    feature_skeptic_tools,
    model_skeptic_prompt,
    model_skeptic_tools,
    run_role,
    trainer_prompt,
    trainer_tools,
    skeptic_prompt,
    skeptic_tools,
)
from mlteam.agents.tools import Review, TeamContext
from mlteam.contract import (
    QUESTIONS_FIELD,
    contract_from_json,
    DataContract,
    Issue,
    has_errors,
    horizon_from_answers,
    propose_contract,
    save_contract,
    validate_contract,
)
from mlteam.feature_tests import FeatureTestRun, run_feature_tests
from mlteam.insights import DataInsights, understand
from mlteam.features import (
    Feature,
    FeatureSet,
    default_features,
    evaluate_features,
    save_feature_set,
)
from mlteam.intake import KaggleFetcher, Manifest, fetch_from_kaggle, ingest
from mlteam.predict import PredictError, Prediction, predict_file, predict_table
from mlteam.profile import DatasetProfile, profile_dataset
from mlteam.relations import Layout, detect_layout
from mlteam.runstore import RunPaths, Trace, now_iso, write_json
from mlteam.store import Store, StoreError, TableInfo
from mlteam.train import Candidate, TrainError, TrainingRun, default_plan, train_all
from mlteam.tune import TUNABLE, Budget, TuneRun, TuneTrial, tune

# (key, label). Stages after "gate" are shown as locked until they are built.
STAGES: tuple[tuple[str, str], ...] = (
    ("intake", "Intake"),
    ("load", "Load"),
    ("profile", "Profile"),
    ("draft", "Draft contract"),
    ("analyst", "Analyst"),
    ("skeptic", "Skeptic"),
    ("gate", "Data contract"),
    ("feature_draft", "Draft features"),
    ("understand", "Understand data"),
    ("engineer", "Engineer"),
    ("feature_skeptic", "Skeptic"),
    ("feature_tests", "Test features"),
    ("feature_gate", "Feature review"),
    ("train_plan", "Training plan"),
    ("tune", "Tune on budget"),
    ("train", "Train and score"),
    ("model_skeptic", "Skeptic"),
    ("model_gate", "Model review"),
    ("predict", "Score test file"),
)
FEATURE_STAGES = ("feature_draft", "understand", "engineer", "feature_skeptic", "feature_tests", "feature_gate")
TRAIN_STAGES = ("train_plan", "tune", "train", "model_skeptic", "model_gate", "predict")
LOCKED_STAGES: tuple[str, ...] = ("Serve live",)
GATES = {"gate", "feature_gate", "model_gate"}


class Team:
    def __init__(
        self,
        run: RunPaths,
        *,
        brain: Brain | None = None,
        brain_note: str = "",
        stats_only: bool = False,
        emit: Emit | None = None,
        fetch_kaggle: KaggleFetcher = fetch_from_kaggle,
        tune_budget: Budget | None = None,
        test_seconds: float = 30.0,
    ) -> None:
        self.run = run
        self.tune_budget = tune_budget if tune_budget is not None else Budget()
        self.test_seconds = test_seconds
        self.feature_tests: FeatureTestRun | None = None
        self.insights: DataInsights | None = None
        self.tuning: TuneRun | None = None
        self.brain = brain
        self.brain_note = brain_note
        self.stats_only = stats_only
        self._emit_to_frontend = emit
        self._fetch_kaggle = fetch_kaggle
        self._trace = Trace(run.trace)
        self.status: dict[str, str] = {key: "pending" for key, _ in STAGES}
        self.manifest: Manifest | None = None
        self.store: Store | None = None
        self.infos: list[TableInfo] = []
        self.profile: DatasetProfile | None = None
        self.layout: Layout | None = None
        self.draft: DataContract | None = None
        self.contract: DataContract | None = None
        self.review: Review | None = None
        self.features: list[Feature] = []
        self.feature_review: Review | None = None
        self.feature_set: FeatureSet | None = None
        self.candidates: list[Candidate] = []
        self.training: TrainingRun | None = None
        self.model_review: Review | None = None
        self.model_card: dict[str, object] | None = None
        self.report_path: Path | None = None
        self.prediction: Prediction | None = None
        # What the person said back to the agents, oldest first: {stage, text, at}.
        self.feedback: list[dict[str, str]] = []

    # -- events ----------------------------------------------------------

    def emit(self, event: TeamEvent) -> None:
        self._trace.log(event.kind, stage=event.stage, agent=event.agent, text=event.text, data=event.data)
        if self._emit_to_frontend is not None:
            self._emit_to_frontend(event)

    def _set(self, stage: str, status: str, text: str = "") -> None:
        self.status[stage] = status
        self.emit(TeamEvent("stage", stage=stage, text=text, data={"status": status}))
        if stage in GATES or stage == "predict":
            self._checkpoint()

    # -- pause and resume ------------------------------------------------

    def _checkpoint(self) -> None:
        """Save where the run stands, so it can be picked up later. Written whenever a
        gate opens or closes; approved artefacts are saved separately and are the truth."""
        try:
            write_json(self.run.state, {
                "saved_at": now_iso(),
                "status": self.status,
                "draft": self.draft.to_json() if self.draft else None,
                "contract": self.contract.to_json() if self.contract else None,
                "review": asdict(self.review) if self.review else None,
                "feature_review": asdict(self.feature_review) if self.feature_review else None,
                "model_review": asdict(self.model_review) if self.model_review else None,
                "features": [asdict(f) for f in self.features],
                "candidates": [asdict(c) for c in self.candidates],
                "feedback": self.feedback,
            })
        except Exception as exc:  # noqa: BLE001 - a checkpoint must never break the run
            self._trace.log("checkpoint_failed", error=str(exc))

    @classmethod
    def resume(cls, root: Path, **kwargs: Any) -> Team:
        """Pick a run up where it stopped: at its last open gate, with its proposals,
        reviews, scores and approvals as they were. A stage that was cut off while
        running starts again from the gate before it."""
        from mlteam.restore import from_dict

        run = RunPaths(Path(root))
        if not run.manifest.exists():
            raise ValueError(f"{root} is not a run folder (no manifest.json).")
        if not run.db.exists() or not run.profile.exists():
            raise ValueError("This run stopped before its data was loaded. Start a new run instead.")

        def read(path: Path) -> Any:
            return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

        team = cls(run, **kwargs)
        team.manifest = from_dict(Manifest, read(run.manifest))
        team.store = Store.open_sealed(run.db)
        team.profile = from_dict(DatasetProfile, read(run.profile))
        team.layout = from_dict(Layout, read(run.layout))
        state = read(run.state) or {}

        latest = contract_from_json(read(run.contract_latest)) if run.contract_latest.exists() else None
        team.draft = contract_from_json(state["draft"]) if state.get("draft") else propose_contract(
            team.store, team.profile, team.layout, team.manifest.dataset_hash)
        team.contract = contract_from_json(state["contract"]) if state.get("contract") else (latest or team.draft)
        for key in ("review", "feature_review", "model_review"):
            if state.get(key):
                setattr(team, key, from_dict(Review, state[key]))
        team.features = [from_dict(Feature, f) for f in state.get("features") or []]
        team.candidates = [from_dict(Candidate, c) for c in state.get("candidates") or []]
        team.feedback = list(state.get("feedback") or [])

        saved = dict(state.get("status") or {})
        status = {key: saved.get(key, "pending") for key, _ in STAGES}
        if not saved:  # an older run without a checkpoint: work it out from what was approved
            status.update({k: "done" for k in ("intake", "load", "profile", "draft")})
            status["gate"] = "done" if latest else "waiting"
            if run.features_latest.exists():
                status["feature_gate"] = "done"
            if run.model_latest.exists():
                status["model_gate"] = "done"
        # Anything cut off mid-stage is not done: the run restarts from the last gate.
        status = {k: ("pending" if v == "running" else v) for k, v in status.items()}
        if status.get("gate") != "done":
            status["gate"] = "waiting"
        team.status = status

        if status.get("gate") == "done" and latest is not None:
            team.contract = latest if latest.version >= (team.contract.version or 0) else team.contract
        if run.features_latest.exists() and status.get("feature_gate") == "done":
            team.feature_set = from_dict(FeatureSet, read(run.features_latest))
            if not team.features:
                team.features = list(team.feature_set.features)
        if status.get("gate") == "done":
            if run.insights.exists():
                team.insights = from_dict(DataInsights, read(run.insights))
            if run.feature_tests.exists() and status.get("feature_tests") == "done":
                team.feature_tests = from_dict(FeatureTestRun, read(run.feature_tests))
        if status.get("feature_gate") == "done" and status.get("train") == "done" and run.training.exists():
            team.training = from_dict(TrainingRun, read(run.training))
            if run.tuning.exists():
                team.tuning = from_dict(TuneRun, read(run.tuning))
            if status.get("model_gate") != "done":
                status["model_gate"] = "waiting"
        elif status.get("feature_gate") == "done":
            for stage in TRAIN_STAGES:
                status[stage] = "pending"
        if status.get("model_gate") == "done" and run.model_latest.exists():
            team.model_card = read(run.model_latest)
            report = run.root / "report.html"
            team.report_path = report if report.exists() else None
        if status.get("gate") == "done" and status.get("feature_gate") != "done" and not team.features:
            for stage in FEATURE_STAGES:
                status[stage] = "pending"  # features were never shown: build them again
        team._trace.log("resume", status=status)
        team.emit(TeamEvent("note", stage="intake", text=f"Resumed run {run.run_id} at: {team.where()}"))
        return team

    def where(self) -> str:
        """The next thing waiting on someone, in plain words."""
        s = self.status
        if s.get("gate") != "done":
            return "the data contract, waiting for your approval"
        if s.get("feature_gate") == "waiting":
            return "the features, waiting for your approval"
        if s.get("feature_gate") != "done":
            return "building features (approved contract)"
        if s.get("model_gate") == "waiting":
            return "the models, waiting for your approval"
        if s.get("model_gate") != "done":
            return "training (approved features)"
        return "an approved model, ready to score and serve"

    # -- feedback --------------------------------------------------------

    def _note_feedback(self, stage: str, text: str | None) -> str:
        """Record what the person said and return it as a block for the agent's prompt."""
        text = (text or "").strip()
        if not text:
            return ""
        self.feedback.append({"stage": stage, "text": text, "at": now_iso()})
        self._trace.log("feedback", stage=stage, text=text)
        self.emit(TeamEvent("note", stage=stage, text=f"Your feedback: {text}"))
        if self.brain is None:
            self.emit(TeamEvent("note", stage=stage, text=(
                "No model is connected, so the rules run again unchanged; the feedback is saved with the run.")))
        return feedback_block(text)

    async def redo(self, step: str, feedback: str) -> object:
        """Send feedback to the agents of one step and run that step again.
        ``step``: contract, features or models. Later steps reset, as with any change."""
        if step == "contract":
            return await self.consult(feedback=feedback)
        if step == "features":
            return await self.build_features(feedback=feedback)
        if step == "models":
            return await self.train_models(feedback=feedback)
        raise ValueError("Feedback goes to one of: contract, features, models.")

    # -- stages ----------------------------------------------------------

    async def prepare(self, sources: Sequence[str]) -> DataContract:
        """Intake through draft contract. Raises IntakeError / StoreError with
        a message fit to show the user."""
        current = "intake"
        try:
            self._set("intake", "running")
            self.manifest = await asyncio.to_thread(
                ingest, list(sources), self.run, fetch_kaggle=self._fetch_kaggle
            )
            size = sum(f.size_bytes for f in self.manifest.files)
            self._set("intake", "done",
                      f"{len(self.manifest.files)} file(s), {size / 1024:.0f} KB, hash {self.manifest.dataset_hash[:10]}")
            for skipped in self.manifest.skipped:
                self.emit(TeamEvent("note", stage="intake", text=f"Skipped {skipped}"))

            current = "load"
            self._set("load", "running")
            self.store = Store(self.run.db)
            self.infos = await asyncio.to_thread(self.store.load, self.manifest, self.run)
            await asyncio.to_thread(self.store.export_parquet, self.run.parquet, self.infos)
            await asyncio.to_thread(self.store.seal)
            self._set("load", "done", ", ".join(f"{i.name} ({i.rows:,} rows)" for i in self.infos))
            for info in self.infos:
                if info.load_note:
                    self.emit(TeamEvent("note", stage="load", text=f"{info.file}: {info.load_note}"))

            current = "profile"
            self._set("profile", "running")
            self.profile = await asyncio.to_thread(profile_dataset, self.store, self.infos)
            self.layout = await asyncio.to_thread(detect_layout, self.store, self.profile)
            write_json(self.run.profile, self.profile.to_json())
            write_json(self.run.layout, self.layout.to_json())
            self._set("profile", "done", f"layout: {self.layout.kind}")
            for note in self.layout.notes:
                self.emit(TeamEvent("note", stage="profile", text=note))

            current = "draft"
            self._set("draft", "running")
            self.draft = await asyncio.to_thread(
                propose_contract, self.store, self.profile, self.layout, self.manifest.dataset_hash
            )
            self.contract = self.draft
            self._set("draft", "done", f"target: {self.draft.target or 'not found'}")
        except Exception as exc:
            self._set(current, "failed", str(exc))
            raise
        return self.draft

    def _context(self) -> TeamContext:
        assert self.store and self.profile and self.layout and self.draft and self.manifest
        return TeamContext(
            store=self.store,
            profile=self.profile,
            layout=self.layout,
            draft=self.draft,
            dataset_hash=self.manifest.dataset_hash,
            stats_only=self.stats_only,
        )

    async def consult(self, feedback: str | None = None) -> None:
        """Analyst refines the draft, then the Skeptic reviews the result once.
        Without a connected model both are skipped and the rule-based draft
        goes straight to the gate. With feedback, the Analyst starts from the
        current proposal and is told what the person wants changed."""
        said = self._note_feedback("analyst", feedback)
        if self.brain is None:
            reason = self.brain_note or "No model connected."
            self._set("analyst", "skipped", reason)
            self._set("skeptic", "skipped", reason)
            self._set("gate", "waiting")
            return
        ctx = self._context()
        if said and self.contract is not None:
            ctx.draft = self.contract  # build on what the person already saw
        note = STATS_ONLY_NOTE if self.stats_only else ""

        self._set("analyst", "running", self.brain.label)
        draft_issues = await asyncio.to_thread(self.validate, ctx.draft)
        try:
            error = await run_role(
                name="Analyst", stage="analyst", brain=self.brain,
                system=ANALYST_SYSTEM + note, tools=analyst_tools(ctx),
                prompt=analyst_prompt(ctx, draft_issues) + said, emit=self.emit,
                submitted=lambda: ctx.proposed is not None,
            )
        except Exception as exc:  # noqa: BLE001 - an agent failure must not kill the run
            error = str(exc)
        if ctx.proposed is not None:
            self.contract = ctx.proposed
            self._set("analyst", "done", f"target: {ctx.proposed.target}")
        else:
            self._set("analyst", "failed", error or "The Analyst did not submit a contract. Using the rule-based draft.")

        self._set("skeptic", "running", self.brain.label)
        contract = self.contract or ctx.draft
        issues = [i for i in await asyncio.to_thread(self.validate, contract) if i.field != QUESTIONS_FIELD]
        try:
            error = await run_role(
                name="Skeptic", stage="skeptic", brain=self.brain,
                system=SKEPTIC_SYSTEM + note, tools=skeptic_tools(ctx),
                prompt=skeptic_prompt(contract, issues, ctx) + said, emit=self.emit,
                submitted=lambda: ctx.review is not None,
            )
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
        if ctx.review is not None:
            self.review = ctx.review
            self._set("skeptic", "done", f"{ctx.review.verdict}, {len(ctx.review.concerns)} concern(s)")
        else:
            self._set("skeptic", "failed", error or "The Skeptic did not submit a review.")
        self._set("gate", "waiting")

    # -- the gate --------------------------------------------------------

    def repropose(self, base_table: str) -> DataContract:
        assert self.store and self.profile and self.layout and self.manifest
        return propose_contract(
            self.store, self.profile, self.layout, self.manifest.dataset_hash, base_override=base_table
        )

    def validate(self, contract: DataContract) -> list[Issue]:
        assert self.store and self.profile
        return validate_contract(self.store, self.profile, contract)

    def approve(self, contract: DataContract, approved_by: str = "human") -> Path:
        """The only way a contract becomes real. Refuses if a check fails."""
        issues = self.validate(contract)
        if has_errors(issues):
            problems = "; ".join(i.message for i in issues if i.level == "error")
            raise ValueError(f"Cannot approve: {problems}")
        hint = horizon_from_answers(contract)
        if hint is not None and contract.horizon_days is None and contract.joins and contract.time_column:
            contract.horizon_days = hint
            contract.rationale["horizon"] = (f"Set from your answer: the prediction is made {hint} days after "
                                             f"{contract.time_column}.")
        path = save_contract(self.run, contract, approved_by)
        self.contract = contract
        self._trace.log(
            "approval",
            version=contract.version,
            approved_by=approved_by,
            contract=contract.to_json(),
            warnings=[asdict(i) for i in issues if i.level == "warning"],
            review=asdict(self.review) if self.review else None,
        )
        self._set("gate", "done", f"contract v{contract.version} approved by {approved_by}")
        for stage in (*FEATURE_STAGES, *TRAIN_STAGES):
            self.status[stage] = "pending"
        self.feature_tests, self.tuning, self.insights = None, None, None
        self.features, self.feature_review, self.feature_set = [], None, None
        self.candidates, self.training, self.model_review, self.model_card = [], None, None, None
        self.prediction = None
        return path

    # -- features --------------------------------------------------------

    async def build_features(self, feedback: str | None = None) -> list[Feature]:
        """Rule-based features, then the Engineer's changes, then one Skeptic
        review. Needs an approved contract; a new approval means a rebuild."""
        contract = self.contract
        if contract is None or not contract.version or self.status.get("gate") != "done":
            raise ValueError("Approve the data contract first.")
        assert self.store and self.profile
        said = self._note_feedback("engineer", feedback)
        for stage in FEATURE_STAGES:
            self.status[stage] = "pending"
        self.features, self.feature_review, self.feature_set = [], None, None

        self._set("feature_draft", "running")
        try:
            drafted = default_features(contract, self.profile)
            self.features = await asyncio.to_thread(evaluate_features, self.store, contract, drafted)
        except Exception as exc:
            self._set("feature_draft", "failed", str(exc))
            raise
        usable = sum(1 for f in self.features if f.usable)
        self._set("feature_draft", "done", f"{usable} usable of {len(self.features)}")
        await self._understand()

        if self.brain is None:
            reason = self.brain_note or "No model connected."
            self._set("engineer", "skipped", reason)
            self._set("feature_skeptic", "skipped", reason)
            await self._run_feature_tests()
            self._set("feature_gate", "waiting")
            return self.features

        ctx = self._context()
        ctx.contract = contract
        ctx.features = self.features
        ctx.insights_text = self.insights.text_for_agents() if self.insights else ""
        note = STATS_ONLY_NOTE if self.stats_only else ""

        self._set("engineer", "running", self.brain.label)
        try:
            error = await run_role(
                name="Engineer", stage="engineer", brain=self.brain,
                system=ENGINEER_SYSTEM + note, tools=engineer_tools(ctx),
                prompt=engineer_prompt(ctx) + said, emit=self.emit, max_turns=20,
                submitted=lambda: ctx.engineered,
            )
        except Exception as exc:  # noqa: BLE001 - an agent failure must not kill the run
            error = str(exc)
        if ctx.engineered:
            self.features = ctx.features
            added = sum(1 for f in self.features if f.origin == "engineer")
            self._set("engineer", "done", f"{len(self.features)} features, {added} from the Engineer")
        else:
            self._set("engineer", "failed",
                      error or "The Engineer did not submit. Using the rule-based features.")

        self._set("feature_skeptic", "running", self.brain.label)
        try:
            error = await run_role(
                name="Skeptic", stage="feature_skeptic", brain=self.brain,
                system=FEATURE_SKEPTIC_SYSTEM + note, tools=feature_skeptic_tools(ctx),
                prompt=feature_skeptic_prompt(ctx) + said, emit=self.emit,
                submitted=lambda: ctx.feature_review is not None,
            )
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
        if ctx.feature_review is not None:
            self.feature_review = ctx.feature_review
            self._set("feature_skeptic", "done",
                      f"{ctx.feature_review.verdict}, {len(ctx.feature_review.concerns)} concern(s)")
        else:
            self._set("feature_skeptic", "failed", error or "The Skeptic did not submit a review.")
        await self._run_feature_tests()
        self._set("feature_gate", "waiting")
        return self.features

    async def _understand(self) -> None:
        """Look at the drafted features against the target (training rows only) and act on
        what is found: add log and cleaned versions, switch near-copies off."""
        self.insights = None
        assert self.store and self.contract
        if not any(f.usable for f in self.features):
            self._set("understand", "skipped", "No usable feature to look at.")
            return
        self._set("understand", "running")
        try:
            self.insights = await asyncio.to_thread(understand, self.store, self.contract, self.features)
        except Exception as exc:  # noqa: BLE001 - informative; the features stand without it
            self._set("understand", "failed", str(exc))
            return
        taken = {f.name.lower() for f in self.features}
        fresh = [f for f in self.insights.new_features if f.name.lower() not in taken]
        if fresh:
            fresh = await asyncio.to_thread(evaluate_features, self.store, self.contract, fresh)
            self.features += [f for f in fresh if f.usable]
        for feature in self.features:
            reason = self.insights.switch_off.get(feature.name)
            if reason and feature.usable:
                feature.suggested_off = reason
        write_json(self.run.insights, self.insights.to_json())
        warns = sum(1 for f in self.insights.findings if f.severity == "warn")
        self._set("understand", "done",
                  f"{len(self.insights.findings)} finding(s), {warns} warning(s); added "
                  f"{sum(1 for f in fresh if f.usable)} feature(s), suggested {len(self.insights.switch_off)} off")

    async def _run_feature_tests(self) -> None:
        """Measure each feature with and without it, within the test budget. Informs
        the person at the gate; never blocks it."""
        self.feature_tests = None
        if self.test_seconds <= 0:
            self._set("feature_tests", "skipped", "No test budget set.")
            return
        on = [f for f in self.features if f.default_on]
        off = [f for f in self.features if f.usable and not f.default_on]
        if not on:
            self._set("feature_tests", "skipped", "No feature is switched on to test against.")
            return
        assert self.store and self.contract
        self._set("feature_tests", "running", f"{len(on) + len(off)} features, {self.test_seconds:g} s budget")
        try:
            self.feature_tests = await asyncio.to_thread(
                run_feature_tests, self.store, self.contract, on, off, self.test_seconds
            )
        except Exception as exc:  # noqa: BLE001 - informative only; the gate still opens
            self._set("feature_tests", "failed", str(exc))
            return
        write_json(self.run.feature_tests, self.feature_tests.to_json())
        self._set("feature_tests", "done", self.feature_tests.summary())

    def approve_features(self, selected: list[str], approved_by: str = "human") -> Path:
        """The only way a feature list becomes real. Re-evaluates what was
        picked instead of trusting the earlier numbers."""
        if self.status.get("feature_gate") not in {"waiting", "done"}:
            raise ValueError("The feature review is not open yet.")
        assert self.store and self.contract and self.manifest
        by_name = {f.name: f for f in self.features}
        unknown = [n for n in selected if n not in by_name]
        if unknown:
            raise ValueError(f"Unknown features: {', '.join(unknown)}")
        if not selected:
            raise ValueError("Pick at least one feature.")
        chosen = [Feature(name=by_name[n].name, sql=by_name[n].sql, why=by_name[n].why,
                          origin=by_name[n].origin) for n in selected]
        evaluate_features(self.store, self.contract, chosen)
        broken = [f for f in chosen if f.error]
        if broken:
            raise ValueError("Cannot approve: " + "; ".join(f"{f.name}: {f.error}" for f in broken))
        feature_set = FeatureSet(
            dataset_hash=self.manifest.dataset_hash,
            contract_version=self.contract.version,
            horizon_days=self.contract.horizon_days,
            features=self.features,
            selected=list(selected),
        )
        path = save_feature_set(self.run, feature_set, approved_by)
        self.feature_set = feature_set
        self._trace.log(
            "feature_approval",
            version=feature_set.version,
            approved_by=approved_by,
            contract_version=self.contract.version,
            selected=list(selected),
            leak_flagged=[f.name for f in chosen if f.leak],
            review=asdict(self.feature_review) if self.feature_review else None,
        )
        self._set("feature_gate", "done", f"features v{feature_set.version}: {len(selected)} approved by {approved_by}")
        for stage in TRAIN_STAGES:
            self.status[stage] = "pending"
        self.candidates, self.training, self.model_review, self.model_card = [], None, None, None
        self.prediction = None
        return path

    # -- training --------------------------------------------------------

    def _selected_features(self) -> list[Feature]:
        assert self.feature_set is not None
        by_name = {f.name: f for f in self.feature_set.features}
        return [by_name[n] for n in self.feature_set.selected if n in by_name]

    async def train_models(self, feedback: str | None = None) -> TrainingRun:
        """Plan, then train and score every candidate in the harness, then one
        Skeptic review. Needs approved features."""
        if self.feature_set is None or self.status.get("feature_gate") != "done":
            raise ValueError("Approve the features first.")
        assert self.store and self.contract and self.profile
        contract, features = self.contract, self._selected_features()
        said = self._note_feedback("train_plan", feedback)
        for stage in TRAIN_STAGES:
            self.status[stage] = "pending"
        self.training, self.model_review, self.model_card = None, None, None
        self.prediction = None
        self.tuning = None
        self.candidates = default_plan()

        ctx = self._context()
        ctx.contract = contract
        ctx.features = features
        ctx.candidates = self.candidates
        ctx.insights_text = self.insights.text_for_agents() if self.insights else ""
        note = STATS_ONLY_NOTE if self.stats_only else ""
        skipped = self.brain_note or "No model connected."

        if self.brain is None:
            self._set("train_plan", "skipped", f"{skipped} Using the rule-based plan.")
        else:
            self._set("train_plan", "running", self.brain.label)
            base = self.profile.table(contract.base_table)
            shape = (f"{base.rows if base else 0:,} rows in {contract.base_table}, {len(features)} features. "
                     "About a fifth of the rows are held back for testing.")
            try:
                error = await run_role(
                    name="Engineer", stage="train_plan", brain=self.brain,
                    system=TRAINER_SYSTEM + note, tools=trainer_tools(ctx),
                    prompt=trainer_prompt(ctx, shape) + said, emit=self.emit,
                    submitted=lambda: ctx.planned,
                )
            except Exception as exc:  # noqa: BLE001 - an agent failure must not kill the run
                error = str(exc)
            if ctx.planned:
                self.candidates = ctx.candidates
                self._set("train_plan", "done", f"{len(self.candidates) - 1} candidates plus the baseline")
            else:
                self._set("train_plan", "failed", error or "The Engineer did not submit a plan. Using the rule-based plan.")

        await self._run_tuning(contract, features)

        self._set("train", "running", f"{len(self.candidates)} candidates")
        try:
            self.training = await asyncio.to_thread(
                train_all, self.store, contract, features, self.candidates, self.run.models
            )
        except (TrainError, Exception) as exc:  # noqa: BLE001 - report, do not crash the app
            self._set("train", "failed", str(exc))
            raise ValueError(str(exc)) from exc
        write_json(self.run.training, self.training.to_json())
        best = next((r for r in self.training.results if r.recommended), None)
        summary = (f"best: {best.name} {self.training.metric}={best.test_score:.3f}" if best and best.test_score is not None
                   else "no candidate beat the baseline")
        self._set("train", "done", summary)
        for warning in self.training.warnings:
            self.emit(TeamEvent("note", stage="train", text=warning))

        if self.brain is None:
            self._set("model_skeptic", "skipped", skipped)
        else:
            ctx.training = self.training
            self._set("model_skeptic", "running", self.brain.label)
            try:
                error = await run_role(
                    name="Skeptic", stage="model_skeptic", brain=self.brain,
                    system=MODEL_SKEPTIC_SYSTEM + note, tools=model_skeptic_tools(ctx),
                    prompt=model_skeptic_prompt(ctx) + said, emit=self.emit,
                    submitted=lambda: ctx.model_review is not None,
                )
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
            if ctx.model_review is not None:
                self.model_review = ctx.model_review
                self._set("model_skeptic", "done",
                          f"{ctx.model_review.verdict}, {len(ctx.model_review.concerns)} concern(s)")
            else:
                self._set("model_skeptic", "failed", error or "The Skeptic did not submit a review.")
        self._set("model_gate", "waiting")
        return self.training

    def _save_reference(self, artifact: Path, version: int) -> Path:
        """Snapshot of the training rows (and the model's answers on them) that later
        batches are compared with to spot drift."""
        import joblib

        from mlteam.monitor import build_reference, prediction_values
        from mlteam.train import _load

        assert self.store and self.contract and self.training
        features = self._selected_features()
        data = _load(self.store, self.contract, features)
        fitted = joblib.load(artifact)
        watched, kind = prediction_values(fitted, data.x_train, self.training.task, self.training.classes)
        path = self.run.reference_version(version)
        write_json(path, build_reference(features, data.x_train, watched, kind))
        return path

    async def _run_tuning(self, contract: DataContract, features: list[Feature]) -> None:
        """Bayesian optimisation within the budget, for the families in the plan.
        Adds one tuned candidate per family that survives; the training step then
        scores them on the locked test rows like any other candidate."""
        self.tuning = None
        budget = self.tune_budget
        families = [c.family for c in self.candidates if c.family in TUNABLE]
        families = list(dict.fromkeys(families))
        if budget.seconds <= 0 or budget.max_trials <= 0:
            self._set("tune", "skipped", "No tuning budget set.")
            return
        if not families:
            self._set("tune", "skipped", "No tunable model family in the plan.")
            return
        assert self.store
        self._set("tune", "running", f"{', '.join(families)} · {budget.describe()}")

        def on_trial(trial: TuneTrial) -> None:
            self.emit(TeamEvent("tune_trial", stage="tune", data={
                "number": trial.number, "family": trial.family, "state": trial.state, "cv_mean": trial.cv_mean,
                "elapsed": trial.elapsed, "best_so_far": trial.best_so_far, "improved": trial.improved,
            }))

        try:
            self.tuning = await asyncio.to_thread(tune, self.store, contract, features, budget, families, on_trial)
        except Exception as exc:  # noqa: BLE001 - tuning is an extra; training still runs on the plan
            self._set("tune", "failed", str(exc))
            return
        write_json(self.run.tuning, self.tuning.to_json())
        taken = {c.name.lower() for c in self.candidates}
        added = [c for c in self.tuning.candidates() if c.name.lower() not in taken]
        self.candidates = [*self.candidates, *added]
        counts = self.tuning.counts
        self._set("tune", "done",
                  f"{len(self.tuning.trials)} tries in {self.tuning.spent_seconds:.0f} s "
                  f"({counts['pruned']} stopped early), {len(added)} tuned candidate(s). {self.tuning.stop_reason}")
        for warning in self.tuning.warnings:
            self.emit(TeamEvent("note", stage="tune", text=warning))

    def approve_model(self, name: str, approved_by: str = "human") -> Path:
        """Pick the model. Refused for the baseline, a failed candidate, or one
        that did not beat the baseline."""
        if self.training is None or self.status.get("model_gate") not in {"waiting", "done"}:
            raise ValueError("The model review is not open yet.")
        assert self.contract and self.feature_set and self.manifest
        result = self.training.result(name)
        if result is None:
            raise ValueError(f"No candidate called {name}.")
        if result.error:
            raise ValueError(f"{name} failed to train: {result.error}")
        if result.family == "baseline":
            raise ValueError("The baseline is the bar to beat, not a model to ship.")
        if not result.beats_baseline:
            raise ValueError(f"{name} is no better than doing nothing, so it cannot be approved.")
        artifact = self.run.models / f"{name}.joblib"
        if not artifact.exists():
            raise ValueError(f"The trained model file for {name} is missing.")
        delivery = _delivery(result)
        if delivery:
            # The decisions made on cross-validated predictions travel with the model.
            import joblib

            from mlteam.train import Delivered

            wrapped = Delivered(joblib.load(artifact), **delivery)
            artifact = self.run.models / f"{name}.delivered.joblib"
            joblib.dump(wrapped, artifact)
        version = 1
        while self.run.model_version(version).exists():
            version += 1
        reference_path = self._save_reference(artifact, version)
        baseline = self.training.baseline
        card: dict[str, object] = {
            "version": version,
            "model": result.name,
            "family": result.family,
            "params": result.params,
            "task": self.training.task,
            "classes": self.training.classes,
            "metric": self.training.metric,
            "test_score": result.test_score,
            "train_score": result.train_score,
            "cv_mean": result.cv_mean,
            "cv_std": result.cv_std,
            "baseline_test_score": baseline.test_score if baseline else None,
            "other_test_scores": result.extra,
            "was_recommended": result.recommended,
            "harness_notes": result.notes,
            "split": self.training.split,
            "train_rows": self.training.train_rows,
            "test_rows": self.training.test_rows,
            "positive_class": self.training.positive_class,
            "features": self.training.features,
            "features_version": self.feature_set.version,
            "contract_version": self.contract.version,
            "dataset_hash": self.manifest.dataset_hash,
            "artifact": f"models/{artifact.name}",
            "members": result.members,
            "delivery": delivery,
            "evaluation": result.evaluation,
            "reference": reference_path.name,
            "importance": [[n, v] for n, v in result.importance],
            "fitted_on": "training rows only",
            "approved_by": approved_by,
            "approved_at": now_iso(),
        }
        path = self.run.model_version(version)
        write_json(path, card)
        write_json(self.run.model_latest, card)
        self.model_card = card
        self._trace.log("model_approval", **{k: card[k] for k in (
            "version", "model", "metric", "test_score", "baseline_test_score", "was_recommended", "approved_by")},
            review=asdict(self.model_review) if self.model_review else None)
        self._set("model_gate", "done", f"model v{version}: {name} approved by {approved_by}")
        try:
            from mlteam.report import write_report

            self.report_path = write_report(self)
            self.emit(TeamEvent("note", stage="model_gate", text=f"Results report: {self.report_path}"))
        except Exception as exc:  # noqa: BLE001 - a report problem must not undo an approval
            self.report_path = None
            self.emit(TeamEvent("note", stage="model_gate", text=f"The results report could not be written: {exc}"))
        self.status["predict"] = "pending"
        self.prediction = None
        return path

    # -- prediction ------------------------------------------------------

    def predict(self) -> Prediction | None:
        """Score the held-back test file with the approved model. Returns None
        when the data has no such file. No agent is involved."""
        if self.model_card is None or self.status.get("model_gate") != "done":
            raise ValueError("Approve a model first.")
        assert self.store and self.contract and self.profile and self.layout and self.feature_set and self.training
        table = self.contract.test_table
        if not table:
            self._set("predict", "skipped", "This data has no held-back test file to score.")
            return None
        self._set("predict", "running", table)
        try:
            self.prediction = predict_table(
                self.store, self.contract, self.profile, self.layout, self._selected_features(), self.training,
                str(self.model_card["model"]), int(self.model_card["version"]), self.run, table,  # type: ignore[call-overload]
                card=self.model_card,
            )
        except (PredictError, StoreError) as exc:
            self._set("predict", "failed", str(exc))
            raise ValueError(str(exc)) from exc
        self._trace.log("prediction", **self.prediction.to_json())
        self._set("predict", "done", f"{self.prediction.rows:,} rows of {table} scored with {self.prediction.model}")
        for note in self.prediction.notes:
            self.emit(TeamEvent("note", stage="predict", text=note))
        return self.prediction

    def predict_file(self, file: str | Path) -> Prediction:
        """Score a brand-new file with the approved model."""
        if self.model_card is None or self.status.get("model_gate") != "done":
            raise ValueError("Approve a model first.")
        try:
            result = predict_file(self.run, Path(file))
        except (PredictError, StoreError) as exc:
            self.emit(TeamEvent("error", stage="predict", text=str(exc)))
            raise ValueError(str(exc)) from exc
        self._trace.log("prediction", **result.to_json())
        self.emit(TeamEvent("note", stage="predict",
                            text=f"{result.rows:,} rows of {result.source} scored with {result.model}"))
        return result

    # -- after go-live: drift, challengers, promotion -------------------------

    def challenge(self, file: str | Path):  # -> Challenge
        """Train a challenger on a labelled file and compare it with the live model."""
        from mlteam.challenger import challenge

        if self.model_card is None:
            raise ValueError("Approve a model first.")
        try:
            result = challenge(self.run, Path(file))
        except (PredictError, StoreError) as exc:
            self.emit(TeamEvent("error", stage="predict", text=str(exc)))
            raise ValueError(str(exc)) from exc
        self.emit(TeamEvent("note", stage="predict", text=f"challenger v{result.version}: {result.sentence()}"))
        return result

    def promote(self, version: int, approved_by: str = "human") -> Path:
        from mlteam.challenger import promote

        try:
            path = promote(self.run, version, approved_by)
        except PredictError as exc:
            raise ValueError(str(exc)) from exc
        self.model_card = json.loads(path.read_text(encoding="utf-8"))
        self.emit(TeamEvent("note", stage="predict",
                            text=f"challenger v{version} promoted: live model is now v{self.model_card['version']}"))
        return path

    def history(self) -> list[dict]:
        from mlteam.monitor import read_history

        return read_history(self.run.monitoring)

    def close(self) -> None:
        if self.store is not None:
            self.store.close()
            self.store = None


def _delivery(result) -> dict[str, object]:
    """Cut-off, probability rescaling and range width chosen for this model, if any."""
    ev = result.evaluation or {}
    out: dict[str, object] = {}
    if ev.get("platt"):
        out["platt"] = list(ev["platt"])
    if ev.get("cutoff") is not None:
        out["cutoff"] = float(ev["cutoff"])
    if ev.get("half_width") is not None:
        out["half_width"] = float(ev["half_width"])
    return out


def feedback_block(text: str) -> str:
    """The person's feedback, as it is handed to an agent."""
    return ("\n\nFEEDBACK FROM THE PERSON who reviewed the last attempt at this step. Act on it; where you "
            "cannot or should not (it would leak the answer, or break a rule), say so in your submission:\n"
            + text.strip())
