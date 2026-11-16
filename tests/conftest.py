from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from mlteam import samples
from mlteam.contract import DataContract, propose_contract
from mlteam.intake import Manifest, ingest
from mlteam.profile import DatasetProfile, profile_dataset
from mlteam.relations import Layout, detect_layout
from mlteam.runstore import RunPaths, new_run
from mlteam.store import Store


@dataclass
class Loaded:
    run: RunPaths
    manifest: Manifest
    store: Store
    profile: DatasetProfile
    layout: Layout
    draft: DataContract


@pytest.fixture
def load(tmp_path: Path) -> Iterator[Callable[..., Loaded]]:
    """Run intake -> load -> seal -> profile -> layout -> draft on given sources."""
    stores: list[Store] = []

    def _load(*sources: str | Path) -> Loaded:
        run = new_run(tmp_path / "runs")
        manifest = ingest([str(s) for s in sources], run)
        store = Store(run.db)
        stores.append(store)
        infos = store.load(manifest, run)
        store.export_parquet(run.parquet, infos)
        store.seal()
        profile = profile_dataset(store, infos)
        layout = detect_layout(store, profile)
        draft = propose_contract(store, profile, layout, manifest.dataset_hash)
        return Loaded(run, manifest, store, profile, layout, draft)

    yield _load
    for store in stores:
        store.close()


@pytest.fixture
def scenario(tmp_path: Path) -> Callable[[str], Path]:
    return lambda name: samples.make(name, tmp_path / "data")


@pytest.fixture(autouse=True)
def small_budgets(monkeypatch):
    """The product tunes for 60 s by default. Tests use a small budget so the suite stays fast;
    tests that are about budgets pass their own."""
    import mlteam.team
    import mlteam.tui.app
    from mlteam.tune import Budget

    def small(**overrides):
        return Budget(**{"seconds": 3.0, "max_trials": 8, "patience": 6, **overrides})

    monkeypatch.setattr(mlteam.team, "Budget", small)
    monkeypatch.setattr(mlteam.tui.app, "Budget", small)
