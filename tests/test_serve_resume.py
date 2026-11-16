"""Serving the approved model over HTTP, picking a run up where it stopped, and
sending feedback to the agents."""

from __future__ import annotations

import csv
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from tau_ai.fake import FakeProvider

from mlteam import samples
from mlteam.agents.brain import Brain
from mlteam.runstore import new_run
from mlteam.serve import make_server
from mlteam.team import Team
from tests.test_team import tool_turn
from tests.test_train import _team_with_features


async def _approved_run(tmp_path: Path) -> tuple[Team, Path]:
    source = samples.make("loans", tmp_path / "data")
    team = Team(new_run(tmp_path / "runs"), test_seconds=0)
    draft = await team.prepare([str(source)])
    await team.consult()
    team.approve(draft, "tester")
    features = await team.build_features()
    team.approve_features([f.name for f in features if f.default_on], "tester")
    run = await team.train_models()
    best = next(r for r in run.results if r.recommended)
    team.approve_model(best.name, "tester")
    return team, Path(source)


def _call(url: str, body: object | None = None) -> tuple[int, dict]:
    data = None if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())
    request = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET",
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


async def test_the_approved_model_is_served_over_http(tmp_path: Path) -> None:
    team, source = await _approved_run(tmp_path)
    team.close()
    with source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))[:6]
    target = team.contract.target
    key = team.contract.grain_keys[0]
    sent = [{k: v for k, v in r.items() if k != target} for r in reversed(rows)]  # not in key order

    server = make_server(team.run, port=0, quiet=True)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    try:
        status, health = _call(f"{base}/health")
        assert status == 200 and health["status"] == "ok" and health["model"] == team.model_card["model"]
        status, info = _call(f"{base}/model")
        assert status == 200 and info["metric"] == team.model_card["metric"] and info["features"]

        status, out = _call(f"{base}/predict", {"rows": sent})
        assert status == 200, out
        assert out["rows"] == 6 and len(out["predictions"]) == 6
        assert [str(p[key]) for p in out["predictions"]] == [str(r[key]) for r in sent]  # order kept
        assert all("prediction" in p for p in out["predictions"])
        assert out["health"] == "too few rows"  # six rows cannot show drift; the batch is still recorded
        history = [json.loads(line) for line in team.run.monitoring.read_text(encoding="utf-8").splitlines()]
        assert history[-1]["rows"] == 6  # live traffic shows up in monitoring

        status, out = _call(f"{base}/predict", b"not json")
        assert status == 400 and "not valid JSON" in out["error"]
        status, out = _call(f"{base}/predict", {"rows": []})
        assert status == 400 and "rows" in out["error"]
        broken = [{k: v for k, v in r.items() if k not in {target, *team.model_card["features"][:1]}} for r in rows]
        status, out = _call(f"{base}/predict", {"rows": broken})
        assert status in {200, 400}  # a missing column is either imputed or explained, never a crash
        status, out = _call(f"{base}/nowhere")
        assert status == 404
    finally:
        server.shutdown()
        server.server_close()


def test_serving_refuses_a_run_without_an_approved_model(tmp_path: Path) -> None:
    from mlteam.serve import ServeError

    run = new_run(tmp_path / "runs")
    with pytest.raises(ServeError, match="no approved model"):
        make_server(run, port=0)


async def test_a_run_is_picked_up_at_its_open_gate(tmp_path: Path) -> None:
    source = samples.make("loans", tmp_path / "data")
    team = Team(new_run(tmp_path / "runs"), test_seconds=0)
    draft = await team.prepare([str(source)])
    await team.consult()
    team.approve(draft, "tester")
    features = await team.build_features()
    names = [f.name for f in features]
    root = team.run.root
    team.close()  # the person closes the app at the feature gate

    resumed = Team.resume(root, test_seconds=0)
    try:
        assert resumed.status["gate"] == "done" and resumed.status["feature_gate"] == "waiting"
        assert [f.name for f in resumed.features] == names
        assert resumed.contract.version == 1 and "features" in resumed.where()
        resumed.approve_features([f.name for f in resumed.features if f.default_on], "tester")
        run = await resumed.train_models()
        best = next(r for r in run.results if r.recommended)
        resumed.approve_model(best.name, "tester")
    finally:
        resumed.close()

    again = Team.resume(root)
    try:
        assert again.status["model_gate"] == "done" and again.model_card is not None
        assert again.training is not None and "ready to score" in again.where()
        assert again.predict_file(source).rows > 0
    finally:
        again.close()


async def test_a_stage_cut_off_mid_run_restarts_from_its_gate(tmp_path: Path) -> None:
    team = await _team_with_features(tmp_path)
    root = team.run.root
    team.close()
    state = json.loads((root / "state.json").read_text(encoding="utf-8"))
    state["status"]["train"] = "running"  # the app was closed while training
    (root / "state.json").write_text(json.dumps(state), encoding="utf-8")
    resumed = Team.resume(root)
    try:
        assert resumed.status["train"] == "pending" and resumed.status["feature_gate"] == "done"
        assert resumed.training is None and "training" in resumed.where()
        await resumed.train_models()
        assert resumed.status["model_gate"] == "waiting"
    finally:
        resumed.close()


async def test_an_older_run_without_a_checkpoint_still_resumes(tmp_path: Path) -> None:
    team = await _team_with_features(tmp_path)
    root = team.run.root
    team.close()
    (root / "state.json").unlink()
    resumed = Team.resume(root)
    try:
        assert resumed.status["gate"] == "done" and resumed.status["feature_gate"] == "done"
        assert resumed.feature_set is not None and resumed.features
    finally:
        resumed.close()


def test_resume_refuses_a_folder_that_is_not_a_run(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a run folder"):
        Team.resume(tmp_path)


async def test_feedback_reaches_the_agents_and_the_step_is_redone(tmp_path: Path) -> None:
    team = await _team_with_features(tmp_path)
    plan = {"candidates": [{"name": "simple", "family": "linear", "why": "Few rows."}]}
    redo = {"candidates": [{"name": "deep_forest", "family": "forest", "params": {"max_depth": 12},
                            "why": "The person asked for deeper forests."}]}
    streams = [
        tool_turn("p1", "submit_training_plan", plan),
        tool_turn("s1", "submit_review", {"verdict": "approve", "summary": "Fine."}),
        tool_turn("p2", "submit_training_plan", redo),
        tool_turn("s2", "submit_review", {"verdict": "approve", "summary": "Fine again."}),
    ]
    provider = FakeProvider(streams)
    team.brain = Brain(provider=provider, model="scripted", label="fake/scripted")
    try:
        first = await team.train_models()
        assert first.result("simple") is not None and first.result("deep_forest") is None
        second = await team.redo("models", "Try deeper forests; skip the linear model.")
        assert second.result("deep_forest") is not None and second.result("simple") is None
        assert team.status["model_gate"] == "waiting"
        redo_prompt = provider.calls[2][2][0].text
        assert "FEEDBACK FROM THE PERSON" in redo_prompt and "deeper forests" in redo_prompt
        assert team.feedback[-1]["text"].startswith("Try deeper forests")
        state = json.loads(team.run.state.read_text(encoding="utf-8"))
        assert state["feedback"][-1]["stage"] == "train_plan"
        trace = team.run.trace.read_text(encoding="utf-8")
        assert '"kind": "feedback"' in trace

        best = next(r for r in second.results if r.recommended)
        team.approve_model(best.name, "tester")
        html = team.report_path.read_text(encoding="utf-8")
        assert "Feedback given to the agents" in html and "deeper forests" in html
    finally:
        team.close()


async def test_redo_needs_a_known_step(tmp_path: Path) -> None:
    team = await _team_with_features(tmp_path)
    try:
        with pytest.raises(ValueError, match="contract, features, models"):
            await team.redo("serving", "faster please")
    finally:
        team.close()


async def test_the_app_reopens_a_paused_run_at_its_gate(tmp_path: Path) -> None:
    from textual.widgets import Input, SelectionList, TabbedContent

    from mlteam.tui.app import MLTeamApp
    from tests.test_tui import SIZE, settle, text_of

    source = samples.make("loans", tmp_path / "data")
    team = Team(new_run(tmp_path / "runs"), test_seconds=0)
    draft = await team.prepare([str(source)])
    await team.consult()
    team.approve(draft, "tester")
    await team.build_features()
    root = team.run.root
    team.close()

    app = MLTeamApp(resume=root, test_seconds=0)
    async with app.run_test(size=SIZE) as pilot:
        await settle(app, pilot)
        assert app.query_one("#main", TabbedContent).active == "features"
        assert app.query_one("#feature_list", SelectionList).option_count > 0
        assert "Approved v1" in str(app.query_one("#approve").label)
        assert "Resumed run" in text_of(app, "#statusbar") or app.team.status["feature_gate"] == "waiting"

        # Feedback with no model connected: saved, and the rules run the step again.
        box = app.query_one("#fb_features", Input)
        box.value = "add a ratio of income to loan amount"
        await pilot.press("tab")
        app.redo_features_pressed()
        await settle(app, pilot)
        assert app.team.feedback and app.team.feedback[-1]["stage"] == "engineer"
        assert app.query_one("#feature_list", SelectionList).option_count > 0
