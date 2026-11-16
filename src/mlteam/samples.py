"""Small synthetic datasets, one per intake situation.

Used by the tests and by ``mlteam demo`` so the tool can be tried without
finding data first. Everything is generated from a fixed seed.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

SCENARIOS = ("loans", "titanic", "shop", "shards")


def _write(path: Path, header: list[str], rows: list[list[object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def make_loans(folder: Path, rows: int = 400, *, with_leak: bool = True) -> Path:
    """One CSV. Binary target, an ID, missing values, and (optionally) a column
    that gives the answer away."""
    rng = random.Random(7)
    header = ["Loan_ID", "Gender", "Married", "ApplicantIncome", "LoanAmount",
              "Credit_History", "Property_Area", "Loan_Status"]
    if with_leak:
        header.insert(-1, "Decision_Code")
    out: list[list[object]] = []
    for i in range(rows):
        income = int(rng.lognormvariate(8.4, 0.5))
        amount = int(rng.gauss(140, 45))
        history = rng.choices([1, 0], weights=[0.82, 0.18])[0]
        chance = 0.15 + 0.6 * history + min(income, 9000) / 60000
        approved = "Y" if rng.random() < chance else "N"
        row: list[object] = [
            f"LP{1000 + i:05d}",
            rng.choice(["Male", "Female"]),
            rng.choice(["Yes", "No"]),
            income,
            amount if rng.random() > 0.05 else "",
            history if rng.random() > 0.08 else "",
            rng.choice(["Urban", "Semiurban", "Rural"]),
            approved,
        ]
        if with_leak:
            row.insert(-1, "A1" if approved == "Y" else "R9")
        out.append(row)
    return _write(folder / "loans.csv", header, out)


def make_titanic(folder: Path, rows: int = 300) -> list[Path]:
    """Kaggle layout: train, test without the target, and a sample submission."""
    rng = random.Random(11)

    def passenger(i: int) -> list[object]:
        sex = rng.choice(["male", "female"])
        pclass = rng.choices([1, 2, 3], weights=[0.24, 0.21, 0.55])[0]
        age = round(max(1.0, rng.gauss(30, 13)), 1)
        fare = round(rng.lognormvariate(3.0, 0.9) / pclass, 2)
        return [i, pclass, sex, age if rng.random() > 0.2 else "", fare]

    train: list[list[object]] = []
    for i in range(1, rows + 1):
        row = passenger(i)
        chance = 0.2 + (0.5 if row[2] == "female" else 0.0) + (0.15 if row[1] == 1 else 0.0)
        train.append([*row, 1 if rng.random() < chance else 0])
    test = [passenger(i) for i in range(rows + 1, rows + 121)]
    cols = ["PassengerId", "Pclass", "Sex", "Age", "Fare"]
    return [
        _write(folder / "train.csv", [*cols, "Survived"], train),
        _write(folder / "test.csv", cols, test),
        _write(folder / "sample_submission.csv", ["PassengerId", "Survived"], [[r[0], 0] for r in test]),
    ]


def make_shop(folder: Path, customers: int = 250) -> list[Path]:
    """Related tables: customers (what we predict about), a regions lookup, and
    orders with many rows per customer."""
    rng = random.Random(23)
    regions = [[i, name, tier] for i, (name, tier) in enumerate(
        [("North", "A"), ("South", "B"), ("East", "A"), ("West", "C"), ("Central", "B")], start=1)]
    cust: list[list[object]] = []
    orders: list[list[object]] = []
    order_id = 1
    for cid in range(1, customers + 1):
        region = rng.randint(1, 5)
        n_orders = rng.choices([0, 1, 2, 4, 8], weights=[0.1, 0.25, 0.3, 0.25, 0.1])[0]
        churned = 1 if rng.random() < (0.55 if n_orders <= 1 else 0.12) else 0
        cust.append([cid, region, f"2024-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
                     rng.choice(["basic", "plus", "pro"]), churned])
        for _ in range(n_orders):
            orders.append([order_id, cid, round(rng.uniform(5, 300), 2), rng.randint(1, 6),
                           f"2025-{rng.randint(1, 9):02d}-{rng.randint(1, 28):02d}"])
            order_id += 1
    return [
        _write(folder / "customers.csv", ["customer_id", "region_id", "signup_date", "plan", "churned"], cust),
        _write(folder / "regions.csv", ["id", "region_name", "tier"], regions),
        _write(folder / "orders.csv", ["order_id", "customer_id", "amount", "items", "order_date"], orders),
    ]


def make_shards(folder: Path, per_file: int = 120) -> list[Path]:
    """The same table split by month."""
    rng = random.Random(31)
    paths: list[Path] = []
    ticket = 1
    for month in ("jan", "feb", "mar"):
        rows: list[list[object]] = []
        for _ in range(per_file):
            minutes = round(rng.lognormvariate(3.2, 0.6), 1)
            rows.append([ticket, rng.choice(["email", "chat", "phone"]),
                         rng.choice(["low", "medium", "high"]), rng.randint(0, 5), minutes])
            ticket += 1
        paths.append(_write(folder / f"tickets_{month}.csv",
                            ["ticket_id", "channel", "priority", "reopens", "resolution_minutes"], rows))
    return paths


def resolve_specs(specs: list[str], folder: Path) -> list[str]:
    """Replace ``demo:<scenario>`` specs with freshly generated sample data."""
    resolved: list[str] = []
    for spec in specs:
        if spec.strip().lower().startswith("demo:"):
            resolved.append(str(make(spec.split(":", 1)[1].strip().lower(), folder)))
        else:
            resolved.append(spec)
    return resolved


def make(scenario: str, folder: Path) -> Path:
    """Create ``scenario`` under ``folder`` and return the path to hand to intake."""
    if scenario == "loans":
        return make_loans(folder / "loans")
    if scenario == "titanic":
        make_titanic(folder / "titanic")
        return folder / "titanic"
    if scenario == "shop":
        make_shop(folder / "shop")
        return folder / "shop"
    if scenario == "shards":
        make_shards(folder / "shards")
        return folder / "shards"
    raise ValueError(f"Unknown scenario: {scenario}. Choose from: {', '.join(SCENARIOS)}")
