from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from mlteam.contract import (
    JoinStep,
    contract_from_json,
    has_errors,
    model_input_sql,
    save_contract,
    validate_contract,
)
from mlteam.store import StoreError


def _errors(issues, field=None):
    return [i for i in issues if i.level == "error" and (field is None or i.field == field)]


def test_single_csv_proposal_and_leakage(load, scenario) -> None:
    loaded = load(scenario("loans"))
    draft = loaded.draft

    assert loaded.layout.kind == "single"
    assert draft.target == "Loan_Status"
    assert draft.task == "classification"
    assert draft.grain_keys == ["Loan_ID"]
    assert draft.grain == "one row = one loan"
    assert [s.column for s in draft.leakage_suspects] == ["Decision_Code"]
    assert {"Loan_ID", "Decision_Code"} <= set(draft.drop_columns)
    assert not has_errors(validate_contract(loaded.store, loaded.profile, draft))


def test_kaggle_layout_finds_target_from_the_split(load, scenario) -> None:
    loaded = load(scenario("titanic"))

    assert loaded.layout.kind == "train_test"
    assert loaded.layout.train_only_columns == ["Survived"]
    assert loaded.draft.base_table == "train"
    assert loaded.draft.test_table == "test"
    assert loaded.draft.target == "Survived"
    assert not has_errors(validate_contract(loaded.store, loaded.profile, loaded.draft))


def test_related_tables_get_lookup_and_aggregate_joins(load, scenario) -> None:
    loaded = load(scenario("shop"))
    draft = loaded.draft

    assert loaded.layout.kind == "related"
    assert draft.base_table == "customers"
    assert draft.target == "churned"
    kinds = {j.table: j.kind for j in draft.joins}
    assert kinds == {"orders": "aggregate", "regions": "lookup"}

    issues = validate_contract(loaded.store, loaded.profile, draft)
    assert not has_errors(issues)
    assert any(i.field == "joins" and "250" in i.message for i in issues)

    columns, rows = loaded.store.query(f"SELECT * FROM ({model_input_sql(draft, loaded.profile)}) LIMIT 1")
    assert "orders__rows" in columns
    assert "regions__region_name" in columns
    assert len(rows) == 1


def test_a_join_that_multiplies_rows_is_an_error(load, scenario) -> None:
    loaded = load(scenario("shop"))
    bad = copy.deepcopy(loaded.draft)
    bad.joins = [JoinStep("orders", "lookup", "customer_id", "customer_id", 0.9)]

    issues = validate_contract(loaded.store, loaded.profile, bad)
    joins = _errors(issues, "joins")
    assert joins and "row count" in joins[0].message


def test_shards_are_stacked_and_key_checked_across_files(load, scenario) -> None:
    loaded = load(scenario("shards"))
    draft = loaded.draft

    assert loaded.layout.kind == "shards"
    assert sorted(draft.stack) == ["tickets_feb", "tickets_jan", "tickets_mar"]
    assert draft.task == "regression"
    assert not has_errors(validate_contract(loaded.store, loaded.profile, draft))
    assert loaded.store.scalar(f"SELECT count(*) FROM ({model_input_sql(draft, loaded.profile)})") == 360


def test_duplicate_key_across_shards_is_caught(load, tmp_path: Path) -> None:
    folder = tmp_path / "dup"
    folder.mkdir()
    # Same ids in both files, different values: not copies, so the key really repeats.
    for name, scale in (("part_a.csv", 1.5), ("part_b.csv", 2.5)):
        rows = "\n".join(f"{i},{i % 3},{i * scale}" for i in range(1, 61))
        (folder / name).write_text("id,grp,value\n" + rows + "\n", encoding="utf-8")
    loaded = load(folder)

    assert loaded.layout.kind == "shards" and loaded.draft.dedupe is False
    grain = _errors(validate_contract(loaded.store, loaded.profile, loaded.draft), "grain")
    assert grain and "repeats" in grain[0].message


def test_shards_that_are_copies_of_each_other_are_deduplicated(load, tmp_path: Path) -> None:
    folder = tmp_path / "copies"
    folder.mkdir()
    rows = "\n".join(f"{i},{i % 3},{i * 1.5}" for i in range(1, 61))
    for name in ("part_a.csv", "part_b.csv"):
        (folder / name).write_text("id,grp,value\n" + rows + "\n", encoding="utf-8")
    loaded = load(folder)
    assert loaded.draft.dedupe is True and loaded.draft.grain_keys == ["id"]
    issues = validate_contract(loaded.store, loaded.profile, loaded.draft)
    assert not _errors(issues) and any("60 exact copy rows dropped" in i.message for i in issues)


@pytest.mark.parametrize(
    ("change", "field"),
    [
        ({"target": None}, "target"),
        ({"target": "NoSuchColumn"}, "target"),
        ({"task": "regression"}, "task"),
        ({"metric": "rmse"}, "metric"),
        ({"grain_keys": ["Gender"]}, "grain"),
        ({"drop_columns": ["Loan_Status"]}, "drops"),
    ],
)
def test_validation_blocks_bad_edits(load, scenario, change, field) -> None:
    loaded = load(scenario("loans"))
    contract = copy.deepcopy(loaded.draft)
    for key, value in change.items():
        setattr(contract, key, value)
    assert _errors(validate_contract(loaded.store, loaded.profile, contract), field)


def test_open_questions_block_until_every_one_is_answered(load, scenario) -> None:
    loaded = load(scenario("loans"))
    contract = copy.deepcopy(loaded.draft)
    contract.open_questions = ["When is the prediction made?", "Who acts on it?"]

    assert len(_errors(validate_contract(loaded.store, loaded.profile, contract), "questions")) == 2
    contract.answers = {"When is the prediction made?": "At application.", "Who acts on it?": "   "}
    left = _errors(validate_contract(loaded.store, loaded.profile, contract), "questions")
    assert len(left) == 1 and "Who acts on it?" in left[0].message
    contract.answers["Who acts on it?"] = "The credit team."
    assert not has_errors(validate_contract(loaded.store, loaded.profile, contract))

    saved = contract_from_json(json.loads(save_contract(loaded.run, contract, "tester").read_text(encoding="utf-8")))
    assert saved.answers == contract.answers and saved.open_questions == contract.open_questions


def test_approval_writes_a_new_version_each_time(load, scenario) -> None:
    loaded = load(scenario("loans"))
    first = save_contract(loaded.run, copy.deepcopy(loaded.draft), "tester")
    second = save_contract(loaded.run, copy.deepcopy(loaded.draft), "tester")

    assert first.name == "contract.v1.json" and second.name == "contract.v2.json"
    saved = contract_from_json(json.loads(second.read_text(encoding="utf-8")))
    assert saved.version == 2 and saved.approved_by == "tester" and saved.approved_at
    assert saved.joins == loaded.draft.joins
    assert json.loads(loaded.run.contract_latest.read_text(encoding="utf-8"))["version"] == 2


def test_sealed_store_refuses_writes_files_and_multiple_statements(load, scenario, tmp_path: Path) -> None:
    loaded = load(scenario("loans"))
    store = loaded.store
    outside = tmp_path / "outside.csv"
    outside.write_text("a\n1\n", encoding="utf-8")

    columns, rows, truncated = store.agent_query("SELECT count(*) AS n FROM loans")
    assert columns == ["n"] and rows == [(400,)] and truncated is False
    _, rows, truncated = store.agent_query("SELECT * FROM loans")
    assert len(rows) == 50 and truncated is True

    for sql in (
        "DROP TABLE loans",
        "CREATE TABLE x AS SELECT 1",
        "SELECT 1; DROP TABLE loans",
        f"COPY loans TO '{(tmp_path / 'leak.csv').as_posix()}'",
        "SET enable_external_access = true",
    ):
        with pytest.raises(StoreError):
            store.agent_query(sql)
    with pytest.raises(StoreError):
        store.agent_query(f"SELECT * FROM read_csv('{outside.as_posix()}')")
    assert not (tmp_path / "leak.csv").exists()


def test_stats_only_profile_carries_no_raw_values(load, scenario) -> None:
    loaded = load(scenario("loans"))
    text = json.dumps(loaded.profile.to_json(stats_only=True))
    for raw in ("LP01000", "Semiurban", "Female", "A1"):
        assert raw not in text
    assert "Semiurban" in json.dumps(loaded.profile.to_json())


def test_target_name_inside_another_word_is_not_a_leak() -> None:
    """Seen on the Adult census data: 'workclass' was kept out because it contains 'class'."""
    from mlteam.contract import _names_target

    assert not _names_target("workclass", "class")
    assert _names_target("churn_date", "churn") and _names_target("is_churned", "churn")
    assert _names_target("churnDate", "churn")
