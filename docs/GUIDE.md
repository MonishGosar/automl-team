# mlteam: the full guide

An AutoML team you steer from the terminal. Agents do the work, the harness sets the
rules, and you decide at the gates. Built on [Tau](https://github.com/huggingface/tau)'s
agent brain (`tau_agent`) with its own Textual interface.

Built so far: data intake, three gates (data contract, features, model), feature tests,
budgeted Bayesian tuning, scoring, drift monitoring, and champion/challenger retraining.
The approved model can be served over HTTP (`mlteam serve`), a run can be paused and
resumed at any gate (`mlteam resume`), and every gate takes feedback for its agents.

```text
intake -> load -> profile -> draft -> Analyst -> Skeptic -> [you] data contract
       -> draft features -> understand the data -> Engineer -> Skeptic -> test features -> [you] feature review
       -> training plan -> tune on a budget -> train and score -> Skeptic -> [you] model review
       -> score the held-back test file
after go-live:
       score new files -> drift per batch -> retrain advice
       labelled file -> challenger -> fair comparison -> [you] promote
```

## Run it

```bash
uv sync
```

```bash
uv run python -m mlteam demo:shop
```

Give it real data the same way:

```bash
uv run python -m mlteam path/to/data.csv
```

```bash
uv run python -m mlteam data_folder/ kaggle:owner/dataset
```

Without the app, as a text report:

```bash
uv run python -m mlteam --headless --no-llm demo:loans
```

Run the tests:

```bash
uv run python -m pytest
```

Use `python -m mlteam` rather than the `mlteam` launcher if Windows Application
Control blocks unsigned executables on your machine (it blocks the generated
`mlteam.exe` on the machine this was built on).

## What it accepts

| Source | Example |
|---|---|
| One file | `sales.csv`, `.tsv`, `.csv.gz`, `.parquet` |
| A folder | every data file inside, zips included |
| A glob | `data/*.csv` |
| A zip | `bundle.zip` |
| Kaggle dataset | `kaggle:owner/dataset` or the dataset URL |
| Kaggle competition | `kaggle:competition/name` or the competition URL |
| Built-in samples | `demo:loans`, `demo:titanic`, `demo:shop`, `demo:shards` |

Kaggle downloads go through `kagglehub`. Credentials are read by `kagglehub` from
`KAGGLE_API_TOKEN` or `~/.kaggle`; agents never see them.

Files are copied into the run folder read-only and hashed. The dataset hash changes
if and only if a file's bytes or name change.

## Several files

"A bunch of CSVs" is one of three different problems. The layout is decided from the
data (column sets, key uniqueness, share of keys that match), not from file names alone.

| Layout | What it looks like | What happens |
|---|---|---|
| Shards | same columns in every file | stacked into one table |
| Related | tables sharing a key | joined: lookups directly, many-row tables summarised first |
| Train / test | Kaggle-style split | roles kept; the target is the column the test file lacks |

After every join the harness counts rows again. If the count changed, the contract is
rejected: a join that multiplies rows inflates scores without any error message.

## The data contract

The gate before any modelling. It records:

- the table to predict from, and what one row means
- the row key (checked for duplicates)
- the target, the task and the metric
- the time column, if any
- the join plan
- columns kept out of the model: identifiers, constants, mostly empty columns, and
  columns that look like they leak the answer

- questions the team could not settle from the data (for example, the moment the
  prediction is made), each with your answer. Approval is refused until every question
  is answered. Agents cannot answer their own questions. In `--headless` mode pass
  `--answer "..."` once per question, in the order asked.

A draft comes from fixed rules. With a model connected, the Analyst refines it and the
Skeptic reviews it once. The harness re-checks whatever they submit. Nothing is real
until you approve it, and every approval is saved as a new version.

## Features

A feature is one DuckDB expression evaluated for one row of the base table (`b`). It may
look into other tables with a correlated subquery:

```sql
(SELECT count(*) FROM orders o WHERE o.customer_id = b.customer_id)
```

Fixed rules draft a starting list: kept base columns, lookup columns, and counts, sums
and averages of each many-row table. With a model connected, the Engineer adds and
removes features and the Skeptic reviews the list once. You tick the ones that go to
training.

Two things hold whoever wrote the expression:

- **The row count cannot change.** Features are a SELECT list over the base table.
- **No feature sees past the prediction moment.** Set "Predict N days after time column"
  on the contract (`--horizon-days N` headless). Every dated, summarised table is then
  replaced, for the whole query, by the rows that exist at that moment. Naming the
  unfiltered table (`main.orders`) is refused.

The harness rejects a feature that uses the target, reads files, returns a date, is
empty or constant, or fails to run. A feature that predicts the target almost perfectly
is kept but flagged as a possible leak and switched off until you turn it on.

Not done at this stage, on purpose: filling gaps, scaling, and encoding categories.
Those belong to training, fitted on training rows only.

Agents can read the features together through one tool, `query_features`, which exposes a
table called `features` (row key, target, every usable feature) under the same time rule.

Features are saved as versioned definitions (`features.v1.json`: SQL, statistics, your
selection, the contract version), not as a copy of the data.

## Understanding the data

Before the Engineer starts, the drafted features are examined against the target, on the
training rows only (the locked test rows never inform a decision). Each finding is a
plain sentence, and most change something:

| Finding | What happens |
|---|---|
| Which features tell most about the target (mutual information) | Ranked and shown; handed to the Engineer |
| A number with a long tail (skew above 2) | A `__log` version is added |
| Features that move together almost perfectly | Grouped; the one most linked to the target is kept, the rest start unticked |
| One label written several ways (`"North "`, `"north"`), or placeholders like `"unknown"` | A `__clean` version is added (trimmed, lower case, placeholders as missing) and the raw one starts unticked |
| Missing values that themselves predict the outcome | Kept as a signal: training adds a "was missing" flag |
| Duplicate rows, a rare outcome, few rows per feature | Warned about; size advice goes to the training plan |

Nothing is removed: features suggested off start unticked at the feature gate, with the
reason, and can be ticked back. The Insights tab (`f8`) shows the full report.

The contract also handles duplicates: rows that are exact copies of another are dropped
before anything else (a checkbox on the Contract tab), and an ID column that repeated
only because of those copies is recognised as the row key.

## Tried on outside data

Besides the built-in samples, the full pipeline (rules only, 30-second budgets) was run
on three datasets that ship with scikit-learn and one deliberately messy random table:

| Data | Rows | Task | What it exercised |
|---|---|---|---|
| Breast cancer | 569 | two outcomes | 30 heavily correlated measurements |
| Wine | 178 | three outcomes | multi-class, very few rows |
| Diabetes | 442 | a number | regression end to end |
| Messy customers (random) | 2,460 | two outcomes | duplicates, a skewed number and its near-copy, messy labels, informative gaps, dates |

The first round found problems that are now fixed: the duplicates were kept and the ID
column used as a feature; messy labels counted as different categories; and feature
tests were so strict that the strongest driver read "no clear effect".

## Testing features

Most features look helpful by luck. Before the feature gate, each feature is tested:
the same reference model (gradient boosting) is cross-validated on the training rows
with and without it, fold by fold, so both scores see exactly the same rows. Features
that are usable but switched off (such as leak suspects) are tested the other way:
what adding them would do.

- **Paired, per fold.** The gain is measured in each fold, not on averages.
- **Corrected for overlap.** Folds share most of their training rows, so a plain
  t-test claims too much certainty. The Nadeau–Bengio corrected test is used.
- **Corrected for testing many at once.** With 20 features, about one passes a 5%
  test by luck. p-values are adjusted with the Benjamini–Hochberg method: of the
  features called real, at most about 5% are expected to be flukes.
- Verdicts: *helps* (proven after both corrections), *likely helps* (better in at least
  80% of folds and significant on its own, but not after correcting for many tests),
  *no clear effect* (unproven, not useless), and the same two for hurting.
- A feature and its derived versions (`x`, `x__log`, `x__clean`) are removed together:
  removed alone, the other version covers for it and nothing shows.
- Runs within a time budget (`--test-seconds`, default 30); features not reached are
  listed. Results appear in the Features tab, with one mark per fold.

On the built-in samples this separates the real signals (`Sex`, `Credit_History`) and a
known leak from a feature of pure noise, which a raw test alone would have passed.

## Tuning on a budget

Model settings are tuned by Bayesian optimisation (Optuna's TPE sampler) inside a budget
you set: seconds (`--tune-seconds`, default 60, or the box beside "Approve features")
and a cap on tries (`--tune-trials`, default 80).

- **How a try is chosen.** The sampler models which settings scored well so far and
  which did not, and proposes settings where "good" is much likelier. The first try
  for each family is its default settings (a warm start).
- **Cheap tries first.** Each try is cross-validated fold by fold. If its running score
  is already below the median of earlier tries, it is stopped early (pruned).
- **Budget goes to the strong.** Every family in the plan gets a few tries, then the
  weaker half is dropped, twice (successive halving).
- **Stops when it stalls.** After many tries without beating the best by more than the
  noise between folds, it stops even if budget is left.
- **Locked test stays locked.** Tuning uses training rows only. Each surviving family
  adds a `tuned_<family>` candidate, scored once on the test rows like every other.
- Tuning searches only the families in the training plan, so the Engineer's choice of
  families is respected.

The Models tab shows the best score against tries spent, and for each tuned model how
it was found.

## Training and scoring

Nobody scores their own model. Fixed rules or the Engineer propose candidates from a
fixed menu (linear, small tree, forest, boosting, with bounded settings). The harness
builds each one, fits it on the training rows and scores it on rows it has never seen.
Every candidate gets the same rows and the same metric.

- A baseline that always gives the most common answer is always trained. A model that
  cannot beat it cannot be approved.
- With a time column, training uses the earlier rows and testing the later ones.
  Otherwise a fifth of the rows are held back, keeping the mix of outcomes.
- Gap-filling, scaling and category encoding are fitted on training rows only.
- The harness flags scores that are too good to be true, and models that do far better
  on training rows than on unseen rows.
- For each model you see what it relies on (score lost when one feature is scrambled).
  The small tree is drawn in full as yes/no questions.
- **Picked by cross-validation, not by the test score.** Every candidate is also
  cross-validated on the training rows. The recommended model is the best by that
  measure. Picking the top test score among many candidates quietly fits the test rows,
  so the test score stays a once-only estimate.
- **Ties are called ties.** The test rows are resampled to see whether the recommended
  model's lead could be zero. If it could, the harness says so and suggests preferring
  the simpler or more readable one.

The Models tab is the third gate. You pick; the best score is only a starting point.
Approval writes `model.v1.json` (scores, baseline, split, versions it was built from)
next to the trained model in `models/<name>.joblib`.

## Predictions

When the data has a held-back test file (the Kaggle layout), approving a model scores it
straight away. No agent is involved. The approved feature expressions are computed for
the test rows under the same time rule, and the fitted model is applied.

- `predictions.v1.csv`: row key, predicted answer, and for two-outcome problems the
  probability of the positive outcome. Labels use a 0.5 cut-off.
- `submission.v1.csv`: written when the data includes a two-column sample submission;
  same column names and order.
- The harness compares the test rows with the training rows and says where they differ:
  a feature missing far more often, a category never seen in training, or a share of
  positive predictions far from the share seen in training.
- Nobody knows the right answers for these rows, so this output carries no score. The
  model's score is the one from the Models tab.

Data without a test file skips this step.

### Scoring a new file later

Any CSV, TSV or Parquet file with the training table's columns can be scored against a
finished run, with exactly what was approved in it: that contract version, those
features, that fitted model.

```bash
uv run python -m mlteam predict latest path/to/new_rows.csv
```

`latest` is the newest run with an approved model; a run id or run folder also works.
In the app, press `ctrl+n` once a model is approved.

- The file is loaded beside the run in a sealed, read-only session. The run's data is
  never changed. Features that look into other tables (orders, regions) still use the
  run's copies of those tables.
- The same checks as for the test file apply, and each scoring is logged in the trace
  with the file's name and hash.
- A file missing a column the features need, or holding text where a number belongs, is
  refused with the reason, and nothing is written.
- **If the file contains the answers**, the model is scored on it and compared with its
  original score. A large drop is reported: something changed since training. The
  answers are never copied into the output.

scikit-learn is pinned to 1.7.2. On the machine this was built on, Windows Application
Control blocked compiled files of 1.9.1.

## Monitoring drift

When a model is approved, a snapshot of its training rows is saved (`reference.v1.json`):
each feature's distribution, and the model's own answers on those rows. Every batch
scored afterwards, including the test file, is compared with it and logged in
`monitoring.jsonl`.

- **Have the inputs moved?** Population Stability Index per feature. Under 0.10 is
  stable, 0.10 to 0.25 is moving, above 0.25 has shifted.
- **Has the output moved?** The same measure on the model's predictions.
- **Is it still right?** When a batch has the answers, the model is scored on it and
  compared with its cross-validated score at approval, allowing for its spread. A single
  test score on a few dozen rows can be lucky, so it is not the comparison.
- **Small batches are judged fairly.** Two samples of the same data differ by chance,
  more so when small. The expected chance level is subtracted before judging.
- **Weighted by reliance.** Drift in a feature the model barely uses counts less than
  drift in one it leans on.

Advice per batch: *stable*, *watch*, or *retrain recommended*, with the reasons.

```bash
uv run python -m mlteam monitor latest
```

## Retraining: champion against challenger

```bash
uv run python -m mlteam retrain latest new_rows_with_answers.csv
```

```bash
uv run python -m mlteam promote latest 1
```

`retrain` trains a challenger: the live model's family and settings, fitted on the
original training rows plus 70% of the new labelled rows. The other 30% (the latest
ones, when there is a time column) are held back. Both models are scored on those rows,
and the gap is resampled to see whether it could be luck. Verdict: *better*, *no clear
difference*, or *worse*. Both are also scored on the original test rows, to catch a
challenger that learned the new pattern by forgetting the old one.

Nothing switches on its own. `promote` writes a new model version that records what it
replaced and the file it learned from, rebuilds the drift snapshot, and from then on
new files are scored with it. A challenger that is clearly worse, or one trained against
a live model that has since changed, cannot be promoted.

In the app, the Monitor tab (`f6`) shows the batch history and has buttons to score a
file, retrain with answers, and promote.

## Before a model is handed over

A score is not enough to use a model. For every model that beats the baseline, the
harness also works out what a team checks before handing it over. Every choice below
is made on **cross-validated predictions for the training rows**; the unseen rows only
show how the choice holds up.

| Check | What it gives you |
|---|---|
| Cut-off (yes/no) | The probability above which to say "yes". Best F1 by default; with costs, the lowest cost. Shown next to 0.50: cases caught, right when flagged, rows flagged. |
| Costs | `--cost-false-alarm 1 --cost-miss 5` (or the Contract tab). Cost per 1,000 rows with the model, acting on nobody, acting on everybody. |
| Calibration | Does "70%" mean 70%? Five bins, said vs happened. Off by more than 5%, and a rescaling (Platt) that helps on held-apart rows: it is applied, and the ranking is unchanged. |
| Rare outcomes | Under 20% of rows: a `boosting_balanced` candidate that weighs rare rows up, PR AUC, and a reminder to read recall. Any family accepts `balanced: 1`. |
| Several classes | Found and right-when-said for each class. |
| Ranges (numbers) | `prediction ± w`, where `w` held the true value for 80% of cross-validated rows; the output gets `low_80` and `high_80` columns. |
| Fairness | Columns named like age, sex, gender, race, religion, marital status, nationality or disability (and any `--fairness COL`) are split into groups. Large gaps in cases caught, rows flagged (four-fifths rule) or error are shown. |
| Groups | An ID column that repeats (several orders per `customer_id`) is suggested as a group: all its rows stay on one side of every split and fold, so the test is on customers the model has never seen. `--group-column none` turns it off. |
| Ensemble | The best model of each family, averaged. Recommended only when at least 0.005 ahead of the best single model in cross-validation; otherwise the simpler one wins. |

Approving a model saves these decisions with it (`models/<name>.delivered.joblib`), so
`mlteam predict` uses the same cut-off, rescaling and range.

### The results report

Approving a model also writes `report.html` (and `report.v1.html`, ...) in the run
folder: one self-contained page for the person who asked the question. What was
predicted and how well against doing nothing, what was decided and answered, what the
data showed, the features, every model compared, the checks above, every reviewer's
concern, and the commands to use it. It opens offline in any browser.

### Comparing runs

```bash
mlteam runs
```

lists every run with its data, target, metric, model, cross-validated and unseen
scores, and status.

## Models

mlteam does not manage API keys. It reuses whatever provider you connected in Tau:

```bash
tau
```

then `/login` inside Tau. Without a model, the app runs on rules only and says so.

`--stats-only` keeps raw rows on your machine: agents lose the SQL tool and see counts,
types and statistics only. Column names are still sent.

## Run folder

Each run writes to `mlteam_runs/<run id>/`:

| File | Content |
|---|---|
| `raw/` | read-only copies of the input files |
| `parquet/`, `data.duckdb` | the loaded tables |
| `manifest.json` | files, hashes, sources |
| `profile.json`, `layout.json` | column profiles and how the tables relate |
| `contract.v1.json`, `contract.v2.json`, ... | every approved contract |
| `features.v1.json`, ... | every approved feature list |
| `training.json`, `models/` | every candidate's scores, and the fitted models |
| `model.v1.json`, ... | every approved model, with what it was built from |
| `feature_tests.json`, `tuning.json` | every feature test and every tuning try |
| `reference.v1.json`, `monitoring.jsonl` | drift snapshot per model, one line per scored batch |
| `challenger.v1.json`, ... | every challenger and how it compared |
| `predictions.v1.csv`, `submission.v1.csv` | scored test file, per model approval |
| `report.html`, `report.v1.html` | the results report, per model approval |
| `trace.jsonl` | every stage, agent step, tool call and approval |

## How it is built

```text
Textual TUI (mlteam.tui)          renders events, edits the contract
        |  TeamEvent stream
Harness (mlteam.team)             fixes stage order, validates, owns the gate
        |
Agents (mlteam.agents)            one Tau AgentHarness per role, own prompt and tools
        |
Data (intake, store, profile,     DuckDB, sealed read-only before any agent touches it
      relations, contract)
```

- The harness, not a model, decides who runs and in what order.
- Agents never call each other. They hand over artefacts: a contract, a review.
- Agent SQL runs against a sealed store: read-only, one statement, 50-row cap, no
  filesystem or network access.
- The interface holds no logic. A different frontend can consume the same `TeamEvent`
  stream.

## Limits so far

- The prediction moment is one number of days after one time column. It hides rows of
  summarised tables only; lookup tables are assumed not to change over time, and base
  columns are not checked for when they became known.
- An agent's exploratory SQL is not cut off by the time rule. Only features are.
- Approving the contract again discards the features built on the old one, and
  approving features again discards the trained models.
- One locked test split. Cross-validation runs on the training rows only. On small
  data, scores move by several points by chance, and the harness says so.
- The approved model is the one fitted on training rows only. It is not refitted on all
  rows before scoring the test file.
- `mlteam serve` is a small standard-library HTTP server with no authentication; it
  scores one request at a time.
- New rows are matched to the run's own copies of joined tables. If those tables have
  grown since, add the new data as a new run.
- Tuning is not cost-aware within a family: a slow setting costs budget like a fast
  one. The budget is wall-clock time, not money or tokens.
- Feature tests use one reference model (gradient boosting); a feature can matter to a
  different model and not to this one.
- A challenger keeps the live model's family and settings. It is not re-tuned, and it
  weighs old and new rows equally.
- Drift checks compare each batch with the training snapshot, not with the batch before.
  Gradual drift shows as a trend in the monitoring report, not as an alert of its own.

- Tabular, supervised problems only (classification, regression).
- Joins go one hop from the base table.
- Free-text columns are kept out, not used.
- The leakage scan is a heuristic (near-perfect purity or correlation, or the target's
  name inside a column name). It misses leaks that need domain knowledge; that is what
  the Skeptic and you are for.
- No query time limit on agent SQL yet.
