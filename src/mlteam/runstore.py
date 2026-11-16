"""Where a run lives on disk, and the append-only trace of what happened in it.

Every run gets its own folder. Nothing in it is overwritten: raw files are
read-only, contracts are versioned, and the trace only grows. That record is
what lets a later run start from this one.
"""

from __future__ import annotations

import json
import os
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def default_runs_dir() -> Path:
    return Path(os.environ.get("MLTEAM_RUNS", "mlteam_runs")).resolve()


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True)
class RunPaths:
    root: Path

    @property
    def run_id(self) -> str:
        return self.root.name

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def parquet(self) -> Path:
        return self.root / "parquet"

    @property
    def db(self) -> Path:
        return self.root / "data.duckdb"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def profile(self) -> Path:
        return self.root / "profile.json"

    @property
    def layout(self) -> Path:
        return self.root / "layout.json"

    @property
    def trace(self) -> Path:
        return self.root / "trace.jsonl"

    @property
    def contract_latest(self) -> Path:
        return self.root / "contract.json"

    def contract_version(self, version: int) -> Path:
        return self.root / f"contract.v{version}.json"

    @property
    def features_latest(self) -> Path:
        return self.root / "features.json"

    def features_version(self, version: int) -> Path:
        return self.root / f"features.v{version}.json"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def training(self) -> Path:
        return self.root / "training.json"

    @property
    def insights(self) -> Path:
        return self.root / "insights.json"

    @property
    def tuning(self) -> Path:
        return self.root / "tuning.json"

    @property
    def feature_tests(self) -> Path:
        return self.root / "feature_tests.json"

    @property
    def model_latest(self) -> Path:
        return self.root / "model.json"

    def model_version(self, version: int) -> Path:
        return self.root / f"model.v{version}.json"

    @property
    def state(self) -> Path:
        """Where the run stopped: stage statuses, the open proposals and reviews."""
        return self.root / "state.json"

    @property
    def monitoring(self) -> Path:
        return self.root / "monitoring.jsonl"

    def reference_version(self, version: int) -> Path:
        return self.root / f"reference.v{version}.json"

    def predictions_version(self, version: int) -> Path:
        return self.root / f"predictions.v{version}.csv"

    def submission_version(self, version: int) -> Path:
        return self.root / f"submission.v{version}.csv"


def new_run(runs_dir: Path | None = None) -> RunPaths:
    base = runs_dir or default_runs_dir()
    run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}"
    root = base / run_id
    root.mkdir(parents=True, exist_ok=False)
    return RunPaths(root)


class Trace:
    """Append-only JSONL log of every stage, agent step and human decision."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def log(self, kind: str, **data: Any) -> None:
        record = {"ts": now_iso(), "kind": kind, **data}
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
