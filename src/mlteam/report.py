"""The deliverable: one self-contained HTML page per approved model.

Written for the person who asked the question, not for the person who built the
model. It says what was predicted, how well, against what bar, what was decided
along the way and by whom, what to watch out for, and how to use it. It opens in
any browser and can be attached to an email; nothing is loaded from the web.
"""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

from mlteam.evaluate import Evaluation, evaluation_from_json
from mlteam.train import BOUNDED, METRIC_WORDS

CSS = """
:root { --bg:#fbfaf7; --fg:#1f2328; --muted:#5f6670; --line:#e3e0d8; --card:#ffffff;
        --good:#1a7f37; --warn:#9a6700; --bad:#cf222e; --accent:#0b5cad; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#14161a; --fg:#e6e6e6; --muted:#9aa3ad; --line:#2c3038; --card:#1b1e24;
          --good:#3fb950; --warn:#d29922; --bad:#f85149; --accent:#58a6ff; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg);
       font:15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 960px; margin: 0 auto; padding: 32px 16px 64px; }
h1 { font-size: 26px; margin: 0 0 4px; }
h2 { font-size: 18px; margin: 36px 0 10px; padding-bottom: 6px; border-bottom: 1px solid var(--line); }
.sub { color: var(--muted); margin: 0 0 20px; }
.cards { display:grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px; }
.card .k { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.04em; }
.card .v { font-size:24px; font-weight:650; margin-top:4px; }
.card .n { color:var(--muted); font-size:13px; }
table { width:100%; border-collapse:collapse; margin:8px 0; font-size:14px; }
th, td { text-align:left; padding:6px 8px; border-bottom:1px solid var(--line); vertical-align:top; }
th { color:var(--muted); font-weight:600; }
td.num, th.num { text-align:right; font-variant-numeric: tabular-nums; }
tr.best td { font-weight:650; }
.tag { display:inline-block; padding:1px 8px; border-radius:99px; font-size:12px; border:1px solid var(--line); }
.good { color:var(--good); } .warn { color:var(--warn); } .bad { color:var(--bad); }
ul { padding-left: 20px; } li { margin: 4px 0; }
code, pre { font-family: ui-monospace, Consolas, monospace; font-size: 13px; }
pre { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px; overflow-x:auto; }
.muted { color: var(--muted); }
.wrap { overflow-x:auto; }
"""


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if value != value:
            return "—"
        return f"{value:,.{digits}f}" if abs(value) < 1000 else f"{value:,.0f}"
    return escape(str(value))


def _pct(value: float | None) -> str:
    return "—" if value is None or value != value else f"{value:.0%}"


def _li(items: list[str]) -> str:
    return "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>" if items else "<p class='muted'>None.</p>"


def build_report(team: Any) -> str:
    contract, training, card = team.contract, team.training, team.model_card
    assert contract is not None and training is not None and card is not None
    result = training.result(str(card["model"]))
    ev: Evaluation | None = evaluation_from_json(card.get("evaluation"))  # type: ignore[arg-type]
    metric = training.metric
    higher = training.higher_is_better
    words = METRIC_WORDS.get(metric, metric)
    baseline = card.get("baseline_test_score")
    test = card.get("test_score")
    parts: list[str] = []

    files = team.manifest.files if team.manifest else []
    rows = training.train_rows + training.test_rows
    title = f"Predicting {contract.target}"
    parts.append(f"<h1>{escape(title)}</h1>")
    parts.append(
        f"<p class='sub'>{escape(contract.task or '')} · {len(files)} file(s) · {rows:,} rows used · "
        f"model v{card['version']} <b>{escape(str(card['model']))}</b> approved by "
        f"{escape(str(card['approved_by']))} on {escape(str(card['approved_at'])[:10])}</p>")

    # Headline.
    if test is not None and baseline is not None:
        if metric in BOUNDED:
            line = f"{metric} {test:.3f} on rows it never saw, against {baseline:.3f} for doing nothing."
        else:
            line = f"typical error {test:,.4g} on rows it never saw, against {baseline:,.4g} for always guessing the average."
        parts.append(f"<p><b>In one line:</b> {escape(line)} "
                     f"<span class='muted'>({escape(words)})</span></p>")
    cards = [
        ("Unseen rows", _fmt(test), f"{metric}, scored once, never used to pick"),
        ("Doing nothing", _fmt(baseline), "the bar to beat"),
        ("Cross-validated", f"{_fmt(card.get('cv_mean'))}", f"± {_fmt(card.get('cv_std'))} on training rows"),
        ("Data", f"{training.train_rows:,} / {training.test_rows:,}", "training / unseen rows"),
    ]
    if ev and ev.kind == "binary" and ev.at_cutoff:
        cards.append(("Cases caught", _pct(ev.at_cutoff.get("recall")),
                      f"right {_pct(ev.at_cutoff.get('precision'))} of the time when it flags"))
    if ev and ev.kind == "regression" and ev.half_width is not None:
        cards.append(("80% range", f"± {ev.half_width:,.4g}", f"held for {_pct(ev.coverage)} of unseen rows"))
    parts.append("<div class='cards'>" + "".join(
        f"<div class='card'><div class='k'>{escape(k)}</div><div class='v'>{v}</div><div class='n'>{escape(n)}</div></div>"
        for k, v, n in cards) + "</div>")

    # Decisions.
    parts.append("<h2>What was decided</h2>")
    decided = [f"<b>Target:</b> {escape(contract.target or '')} ({escape(contract.task or '')}, scored by {escape(metric)})",
               f"<b>One row:</b> {escape(contract.grain or '—')}",
               f"<b>Split:</b> {escape(training.split)}"]
    if contract.dedupe:
        decided.append("<b>Duplicates:</b> exact copy rows were dropped before anything else.")
    if contract.group_column:
        decided.append(f"<b>Groups:</b> all rows of one {escape(contract.group_column)} stay on the same side of every split.")
    if contract.horizon_days is not None and contract.time_column:
        decided.append(f"<b>Prediction moment:</b> {contract.horizon_days} days after {escape(contract.time_column)}; "
                       "joined data after that is not used.")
    if contract.drop_columns:
        decided.append("<b>Kept out:</b> " + escape(", ".join(contract.drop_columns)))
    for question, answer in contract.answers.items():
        decided.append(f"<b>Asked:</b> {escape(question)}<br><b>Answer:</b> {escape(answer)}")
    parts.append(_li(decided))
    feedback = getattr(team, "feedback", None) or []
    if feedback:
        step = {"analyst": "contract", "engineer": "features", "train_plan": "models"}
        parts.append("<p><b>Feedback given to the agents:</b></p>" + _li(
            [f"<span class='muted'>{escape(str(f.get('at', ''))[:16])} · {escape(step.get(f.get('stage', ''), ''))}"
             f"</span> {escape(f.get('text', ''))}" for f in feedback]))
    if contract.leakage_suspects:
        parts.append("<p><b>Possible leaks checked:</b></p>" + _li(
            [f"{escape(s.column)}: {escape(s.reason)}" for s in contract.leakage_suspects]))

    # Data.
    insights = team.insights
    if insights is not None:
        parts.append("<h2>What the data showed</h2>")
        if insights.signal:
            parts.append("<p>Most informative on their own: " + escape(
                ", ".join(n for n, _ in insights.signal[:6])) + ".</p>")
        parts.append(_li([escape(f.text) + (f" <span class='muted'>→ {escape(f.action)}</span>" if f.action else "")
                          for f in insights.findings if f.kind != "signal"]))

    # Features.
    if team.feature_set is not None:
        used = training.features
        parts.append("<h2>Features</h2>")
        parts.append(f"<p>{len(used)} features used.</p>")
        tests = team.feature_tests
        if tests is not None and tests.tests:
            strong = [t for t in tests.tests if t.verdict in {"helps", "likely helps"}]
            parts.append("<p>Tested one at a time (score with and without, corrected for testing many): "
                         + (escape(", ".join(f"{t.name} ({t.verdict})" for t in strong)) if strong else
                            "none stood out on its own; several carry the same information.") + "</p>")

    # Models.
    parts.append("<h2>Models compared</h2>")
    parts.append("<p class='muted'>The harness trained and scored every model on the same rows. The best is picked "
                 "by cross-validation on training rows; the unseen rows are scored once.</p>")
    head = ("<tr><th>Model</th><th>Family</th><th class='num'>Cross-val</th><th class='num'>Unseen</th>"
            "<th class='num'>Trained-on</th><th>Note</th></tr>")
    body = []
    for r in training.results:
        if r.error:
            body.append(f"<tr><td>{escape(r.name)}</td><td>{escape(r.family)}</td><td colspan='4' class='bad'>"
                        f"did not train</td></tr>")
            continue
        cls = " class='best'" if r.name == card["model"] else ""
        note = "approved" if r.name == card["model"] else ("the bar" if r.family == "baseline"
                                                           else "recommended" if r.recommended else "")
        cv = f"{_fmt(r.cv_mean)} ± {_fmt(r.cv_std)}" if r.cv_mean is not None else "—"
        body.append(f"<tr{cls}><td>{escape(r.name)}</td><td>{escape(r.family)}</td><td class='num'>{cv}</td>"
                    f"<td class='num'>{_fmt(r.test_score)}</td><td class='num'>{_fmt(r.train_score)}</td>"
                    f"<td>{escape(note)}</td></tr>")
    parts.append(f"<div class='wrap'><table>{head}{''.join(body)}</table></div>")
    if result is not None and result.importance:
        shown = [(n, v) for n, v in result.importance if v >= 0.0005][:8]
        if shown:
            parts.append("<p><b>What the approved model relies on</b> <span class='muted'>(score lost when a feature "
                         "is scrambled)</span></p><table>" + "".join(
                             f"<tr><td>{escape(n)}</td><td class='num'>{v:+.3f}</td></tr>" for n, v in shown)
                         + "</table>")

    # Using it.
    if ev is not None:
        parts.append("<h2>Before you use it</h2>")
        parts.append(_evaluation_html(ev))

    # Risks.
    parts.append("<h2>Risks and open concerns</h2>")
    risks: list[str] = [escape(n) for n in (card.get("harness_notes") or [])]  # type: ignore[union-attr]
    risks += [escape(w) for w in training.warnings]
    for label, review in (("Contract", team.review), ("Features", team.feature_review), ("Model", team.model_review)):
        if review is None:
            continue
        for c in review.concerns:
            tone = "bad" if c.severity.lower() == "high" else "warn" if c.severity.lower() == "medium" else "muted"
            risks.append(f"<span class='tag {tone}'>{escape(c.severity)}</span> {escape(label)} review, "
                         f"{escape(c.field)}: {escape(c.message)}")
    parts.append(_li(risks))

    # How to use.
    run_id = team.run.run_id
    parts.append("<h2>How to use it</h2>")
    parts.append("<pre>"
                 f"mlteam predict {escape(run_id)} new_rows.csv     # predictions plus a drift check\n"
                 f"mlteam monitor {escape(run_id)}                  # every batch scored so far\n"
                 f"mlteam retrain {escape(run_id)} rows_with_answers.csv  # train a challenger\n"
                 "</pre>")
    parts.append(f"<p class='muted'>Run folder: {escape(str(team.run.root))}. Every step, agent message and approval "
                 "is in trace.jsonl there.</p>")

    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>{escape(title)}</title><style>{CSS}</style></head><body><main>"
            + "".join(parts) + "</main></body></html>")


def _evaluation_html(ev: Evaluation) -> str:
    out: list[str] = []
    if ev.kind == "binary" and ev.cutoff is not None:
        cut, half = ev.at_cutoff or {}, ev.at_half or {}
        out.append(f"<p><b>Cut-off:</b> call it {escape(str(ev.positive_class))} when the probability is "
                   f"{ev.cutoff:.2f} or more <span class='muted'>({escape(ev.cutoff_reason)})</span>.</p>")
        out.append("<table><tr><th></th><th class='num'>at the cut-off</th><th class='num'>at 0.50</th></tr>"
                   + "".join(f"<tr><td>{label}</td><td class='num'>{_pct(cut.get(k))}</td>"
                             f"<td class='num'>{_pct(half.get(k))}</td></tr>"
                             for label, k in (("cases caught (recall)", "recall"),
                                              ("right when it flags (precision)", "precision"),
                                              ("rows flagged", "flagged")))
                   + "</table>")
        if cut:
            out.append(f"<p class='muted'>On the unseen rows: {cut['tp']} caught, {cut['fn']} missed, "
                       f"{cut['fp']} false alarms, {cut['tn']} correctly left alone.</p>")
        if ev.cost:
            c = ev.cost
            out.append(f"<p><b>Cost per 1,000 rows</b> (a false alarm costs {c['false_alarm']:g}, a miss "
                       f"{c['miss']:g}): {c['model']:,.0f} with the model, {c['act_on_nobody']:,.0f} acting on "
                       f"nobody, {c['act_on_everybody']:,.0f} acting on everybody.</p>")
        if ev.ece_before is not None:
            if ev.platt is not None and ev.ece_after is not None:
                out.append(f"<p><b>Probabilities</b> were rescaled: off by {ev.ece_before:.1%} on average before, "
                           f"{ev.ece_after:.1%} after.</p>")
            else:
                out.append(f"<p><b>Probabilities</b> are off by {ev.ece_before:.1%} on average.</p>")
    elif ev.kind == "multiclass":
        out.append("<table><tr><th>Class</th><th class='num'>Found</th><th class='num'>Right when said</th>"
                   "<th class='num'>Rows</th></tr>" + "".join(
                       f"<tr><td>{escape(str(p['class']))}</td><td class='num'>{_pct(p['recall'])}</td>"
                       f"<td class='num'>{_pct(p['precision'])}</td><td class='num'>{p['rows']}</td></tr>"
                       for p in ev.per_class) + "</table>")
    elif ev.kind == "regression" and ev.half_width is not None:
        out.append(f"<p><b>Range:</b> each prediction comes with ± {ev.half_width:,.4g}. That range held the true "
                   f"value for 80% of cross-validated rows and {_pct(ev.coverage)} of unseen rows.</p>")
    out += [f"<p class='warn'>! {escape(n)}</p>" for n in ev.notes]
    for group in ev.groups:
        out.append(f"<p><b>By {escape(group.column)}</b></p>")
        if ev.kind == "binary":
            head = "<tr><th>Group</th><th class='num'>Rows</th><th class='num'>Outcome</th><th class='num'>Flagged</th><th class='num'>Caught</th></tr>"
            rows = "".join(f"<tr><td>{escape(r.value)}</td><td class='num'>{r.rows}</td><td class='num'>{_pct(r.actual)}</td>"
                           f"<td class='num'>{_pct(r.predicted)}</td><td class='num'>{_pct(r.recall)}</td></tr>"
                           for r in group.rows)
        elif ev.kind == "regression":
            head = "<tr><th>Group</th><th class='num'>Rows</th><th class='num'>Average</th><th class='num'>Predicted</th><th class='num'>Typical error</th></tr>"
            rows = "".join(f"<tr><td>{escape(r.value)}</td><td class='num'>{r.rows}</td><td class='num'>{_fmt(r.actual)}</td>"
                           f"<td class='num'>{_fmt(r.predicted)}</td><td class='num'>{_fmt(r.score)}</td></tr>"
                           for r in group.rows)
        else:
            head = "<tr><th>Group</th><th class='num'>Rows</th><th class='num'>Right</th></tr>"
            rows = "".join(f"<tr><td>{escape(r.value)}</td><td class='num'>{r.rows}</td><td class='num'>{_pct(r.score)}</td></tr>"
                           for r in group.rows)
        out.append(f"<div class='wrap'><table>{head}{rows}</table></div>")
        if group.gap:
            out.append(f"<p class='warn'>! {escape(group.gap)}</p>")
        if group.context:
            out.append(f"<p class='muted'>{escape(group.context[0].upper() + group.context[1:])}</p>")
        if not group.gap and not group.context:
            out.append("<p class='muted'>No large gap between groups.</p>")
    return "".join(out)


def write_report(team: Any) -> Path:
    version = team.model_card["version"]
    path = team.run.root / f"report.v{version}.html"
    path.write_text(build_report(team), encoding="utf-8")
    (team.run.root / "report.html").write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return path
