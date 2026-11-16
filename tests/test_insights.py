"""Understanding the data, and the cleaning it leads to."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from textual.widgets import SelectionList, Static

from mlteam.contract import validate_contract
from mlteam.features import Feature, default_features, evaluate_features
from mlteam.insights import understand
from mlteam.runstore import new_run
from mlteam.team import Team
from mlteam.tui import render
from mlteam.tui.app import MLTeamApp
from tests.test_tui import SIZE, settle, text_of


def messy(folder: Path, rows: int = 600, seed: int = 1) -> Path:
    """Planted problems: duplicate rows, an ID that repeats only because of them, a skewed
    number and its near-copy, labels written several ways, informative missing values."""
    rng = random.Random(seed)
    folder.mkdir(parents=True, exist_ok=True)
    data = []
    for i in range(rows):
        visits = int(rng.expovariate(1 / 6))
        income = round(rng.lognormvariate(10.5, 0.9), 2)
        segment = rng.choice(["consumer", "smb", "enterprise"])
        pages = round(rng.uniform(1, 12), 2) if visits > 0 else ""
        score = 0.4 * (visits > 4) + 0.3 * (segment == "enterprise") + rng.gauss(0, 0.2)
        data.append([f"C{1000 + i}", rng.choice(["north", "North ", "south", "unknown"]), segment, visits, pages,
                     income, round(income + rng.gauss(0, 30), 2), 1 if score > 0.45 else 0])
    data += [list(r) for r in rng.sample(data, 30)]
    path = folder / "customers.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["customer_ref", "region", "segment", "visits", "pages_per_visit", "income", "income_usd",
                         "upgraded"])
        writer.writerows(data)
    return path


def _by_kind(insights) -> dict[str, list]:
    out: dict[str, list] = {}
    for f in insights.findings:
        out.setdefault(f.kind, []).append(f)
    return out


def test_contract_drops_copies_and_finds_the_id_they_hid(load, tmp_path: Path) -> None:
    loaded = load(messy(tmp_path / "messy"))
    draft = loaded.draft
    assert draft.dedupe is True and "30 rows are exact copies" in draft.rationale["dedupe"]
    assert draft.grain_keys == ["customer_ref"] and "customer_ref" in draft.drop_columns
    issues = validate_contract(loaded.store, loaded.profile, draft)
    assert any(i.field == "dedupe" and "600 rows left" in i.message for i in issues)
    assert not [i for i in issues if i.level == "error"]
    rows = loaded.store.scalar(f"SELECT count(*) FROM (SELECT * FROM {__import__('mlteam.contract', fromlist=['base_sql']).base_sql(draft)} AS b)")
    assert rows == 600


def test_understanding_finds_what_was_planted(load, tmp_path: Path) -> None:
    loaded = load(messy(tmp_path / "messy"))
    features = evaluate_features(loaded.store, loaded.draft, default_features(loaded.draft, loaded.profile))
    ins = understand(loaded.store, loaded.draft, features)
    found = _by_kind(ins)

    assert ins.rows == 480  # training rows only: the locked test rows are never looked at
    assert ins.signal[0][0] in {"visits", "segment"}  # the real drivers, not an ID
    assert {n for n, _ in ins.signal} >= {"visits", "segment", "region"}
    redundant = found["redundant"][0]
    assert set(redundant.features) == {"income", "income_usd"}
    off = set(ins.switch_off)
    assert len(off & {"income", "income_usd"}) == 1 and "region" in off
    labels = found["labels"][0]
    assert "'North ' / 'north'" in labels.text and "'unknown'" in labels.text
    assert any(f.kind == "missing" and f.features == ["pages_per_visit"] for f in ins.findings)
    names = {f.name for f in ins.new_features}
    assert "region__clean" in names and any(n.startswith("income") and n.endswith("__log") for n in names)
    kept = ({"income", "income_usd"} - off).pop()
    assert f"{kept}__log" in names and f"{({'income', 'income_usd'} & off).pop()}__log" not in names

    # The added features are real, working features.
    added = evaluate_features(loaded.store, loaded.draft, list(ins.new_features))
    clean = next(f for f in added if f.name == "region__clean")
    assert clean.error is None and clean.distinct == 2  # north and south; 'unknown' became missing
    values = loaded.store.query(f"SELECT DISTINCT {clean.sql} AS v FROM customers AS b")[1]
    assert {v[0] for v in values} == {None, "north", "south"}


def test_size_advice_and_quiet_data(load, scenario) -> None:
    loaded = load(scenario("titanic"))
    ins = understand(loaded.store, loaded.draft, evaluate_features(
        loaded.store, loaded.draft, default_features(loaded.draft, loaded.profile)))
    assert ins.signal[0][0] == "Sex"
    assert ins.advice and "training rows" in ins.advice[0]
    assert "UNDERSTANDING THE DATA" in str(render.insights_view(ins))


async def test_pipeline_acts_on_what_it_finds(tmp_path: Path) -> None:
    team = Team(new_run(tmp_path / "runs"), test_seconds=0)
    try:
        draft = await team.prepare([str(messy(tmp_path / "messy"))])
        await team.consult()
        team.approve(draft, "tester")
        features = await team.build_features()
        assert team.status["understand"] == "done" and team.run.insights.exists()
        by_name = {f.name: f for f in features}
        assert "region__clean" in by_name and by_name["region__clean"].default_on
        assert by_name["region"].suggested_off and not by_name["region"].default_on
        assert by_name["region"].usable  # off by default, still the person's choice
        saved = json.loads(team.run.insights.read_text(encoding="utf-8"))
        assert any(f["kind"] == "labels" for f in saved["findings"])

        team.approve_features([f.name for f in features if f.default_on], "tester")
        run = await team.train_models()
        assert any(r.beats_baseline for r in run.results)
        assert "region" not in run.features and "region__clean" in run.features
    finally:
        team.close()


async def test_app_shows_insights_and_unticks_suggested_off(tmp_path: Path) -> None:
    source = messy(tmp_path / "messy")
    app = MLTeamApp([str(source)], runs_dir=tmp_path / "runs", test_seconds=0)
    async with app.run_test(size=SIZE) as pilot:
        await settle(app, pilot)
        assert "Approve the data contract first" in text_of(app, "#insights_view")
        assert "exact copies" in text_of(app, "#reasons")
        from textual.widgets import Checkbox

        box = app.query_one("#dedupe", Checkbox)
        assert box.value is True
        box.value = False  # keeping the copies: the key now repeats, and approval is blocked
        await settle(app, pilot)
        assert "repeats" in text_of(app, "#checks")
        box.value = True
        await settle(app, pilot)
        assert "exact copy rows dropped" in text_of(app, "#checks")
        await pilot.press("ctrl+s")
        await settle(app, pilot)
        view = text_of(app, "#insights_view")
        assert "Which features tell most about the target" in view and "Messy labels" in view
        assert "Exact duplicate rows were dropped" in view
        listing = app.query_one("#feature_list", SelectionList)
        assert "region__clean" in listing.selected and "region" not in listing.selected
        labels = " | ".join(str(listing.get_option_at_index(i).prompt) for i in range(listing.option_count))
        assert "off?" in labels
