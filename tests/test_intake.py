from __future__ import annotations

import os
import stat
import zipfile
from pathlib import Path

import pytest

from mlteam import samples
from mlteam.intake import IntakeError, ingest, parse_kaggle
from mlteam.runstore import new_run


def test_single_file_is_copied_hashed_and_read_only(tmp_path: Path) -> None:
    source = samples.make_loans(tmp_path / "data")
    run = new_run(tmp_path / "runs")
    manifest = ingest([str(source)], run)

    assert [f.name for f in manifest.files] == ["loans.csv"]
    copied = run.raw / "loans.csv"
    assert copied.read_bytes() == source.read_bytes()
    assert not os.stat(copied).st_mode & stat.S_IWRITE
    assert len(manifest.dataset_hash) == 64
    assert run.manifest.exists()


def test_same_bytes_give_same_hash_and_changed_bytes_do_not(tmp_path: Path) -> None:
    source = samples.make_loans(tmp_path / "data")
    first = ingest([str(source)], new_run(tmp_path / "runs")).dataset_hash
    second = ingest([str(source)], new_run(tmp_path / "runs")).dataset_hash
    assert first == second

    source.write_text(source.read_text(encoding="utf-8") + "LP99999,Male,No,1,1,1,Urban,A1,Y\n", encoding="utf-8")
    third = ingest([str(source)], new_run(tmp_path / "runs")).dataset_hash
    assert third != first


def test_folder_takes_every_data_file_and_reports_the_rest(tmp_path: Path) -> None:
    folder = tmp_path / "data" / "shop"
    samples.make_shop(folder)
    (folder / "README.md").write_text("notes", encoding="utf-8")
    manifest = ingest([str(folder)], new_run(tmp_path / "runs"))

    assert sorted(f.name for f in manifest.files) == ["customers.csv", "orders.csv", "regions.csv"]
    assert any("README.md" in s for s in manifest.skipped)


def test_glob_and_duplicate_names(tmp_path: Path) -> None:
    samples.make_shards(tmp_path / "a")
    samples.make_shards(tmp_path / "b")
    manifest = ingest([str(tmp_path / "a" / "*.csv"), str(tmp_path / "b")], new_run(tmp_path / "runs"))

    names = [f.name for f in manifest.files]
    assert len(names) == 6
    assert len({n.lower() for n in names}) == 6
    assert "b__tickets_jan.csv" in names


def test_same_path_given_twice_is_taken_once(tmp_path: Path) -> None:
    source = samples.make_loans(tmp_path / "data")
    manifest = ingest([str(source), str(source.parent)], new_run(tmp_path / "runs"))
    assert len(manifest.files) == 1


def test_zip_is_extracted_safely(tmp_path: Path) -> None:
    source = samples.make_loans(tmp_path / "data")
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(source, "nested/loans.csv")
        zf.writestr("../escape.csv", "a,b\n1,2\n")
        zf.writestr("notes.txt", "hello")
    run = new_run(tmp_path / "runs")
    manifest = ingest([str(archive)], run)

    assert [f.name for f in manifest.files] == ["nested__loans.csv"]
    assert any("unsafe path" in s for s in manifest.skipped)
    assert any("notes.txt" in s for s in manifest.skipped)
    assert not (tmp_path / "escape.csv").exists()


def test_missing_path_and_empty_folder_fail_with_a_usable_message(tmp_path: Path) -> None:
    with pytest.raises(IntakeError, match="kaggle:owner/dataset"):
        ingest([str(tmp_path / "nope.csv")], new_run(tmp_path / "runs"))
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(IntakeError, match="No CSV"):
        ingest([str(empty)], new_run(tmp_path / "runs"))


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("kaggle:uciml/iris", ("dataset", "uciml/iris")),
        ("kaggle:competition/titanic", ("competition", "titanic")),
        ("kaggle-competition:titanic", ("competition", "titanic")),
        ("https://www.kaggle.com/datasets/uciml/iris", ("dataset", "uciml/iris")),
        ("https://www.kaggle.com/datasets/uciml/iris/data?select=Iris.csv", ("dataset", "uciml/iris")),
        ("https://www.kaggle.com/competitions/titanic/data", ("competition", "titanic")),
        ("https://kaggle.com/c/titanic", ("competition", "titanic")),
        ("data/train.csv", None),
        ("C:/data/train.csv", None),
    ],
)
def test_parse_kaggle(spec: str, expected: tuple[str, str] | None) -> None:
    assert parse_kaggle(spec) == expected


def test_malformed_kaggle_handle_is_rejected() -> None:
    with pytest.raises(IntakeError):
        parse_kaggle("kaggle:just-a-name")


def test_kaggle_source_uses_the_fetcher_and_records_origin(tmp_path: Path) -> None:
    cache = tmp_path / "kaggle_cache"
    samples.make_titanic(cache)
    calls: list[tuple[str, str]] = []

    def fake_fetch(kind: str, handle: str) -> Path:
        calls.append((kind, handle))
        return cache

    manifest = ingest(["kaggle:someone/titanic"], new_run(tmp_path / "runs"), fetch_kaggle=fake_fetch)
    assert calls == [("dataset", "someone/titanic")]
    assert sorted(f.name for f in manifest.files) == ["sample_submission.csv", "test.csv", "train.csv"]
    assert all(f.origin == "kaggle:dataset:someone/titanic" for f in manifest.files)
