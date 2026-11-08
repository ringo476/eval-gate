"""
A single self-contained HTML page summarising the latest full run of every
(task, model): how each model did, whether it regressed, how retrieval scored,
and exactly which case failed which check and why. No server, no dependencies,
nothing fetched from the network, so it can be opened from disk or uploaded as a
CI artifact.

Everything from the runs goes through html.escape: model answers and judge
reasons are untrusted text, and a report must not execute what it displays.
"""
from __future__ import annotations

from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

from harness.core.baseline import find_baseline
from harness.core.diff import diff_runs
from harness.core.stats import wilson_interval
from harness.core.store import RunStore

_STATUS = {"pass": ("✓", "Pass"), "flaky": ("~", "Flaky"), "fail": ("✗", "Fail")}

_STYLE = """
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #6f6d68;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10); --series-1: #2a78d6;
  --good: #0ca30c; --good-ink: #006300; --warning: #fab219; --warning-ink: #8a5a00; --critical: #d03b3b;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #a3a29a;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10); --series-1: #3987e5;
    --good-ink: #0ca30c; --warning-ink: #fab219;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #a3a29a;
  --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10); --series-1: #3987e5;
  --good-ink: #0ca30c; --warning-ink: #fab219;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
       font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1120px; margin: 0 auto; padding: 32px 16px 64px; }
h1 { font-size: 24px; margin: 0 0 4px; }
h2 { font-size: 18px; margin: 40px 0 12px; }
h3 { font-size: 14px; margin: 24px 0 8px; color: var(--ink-2); font-weight: 600; }
.sub { color: var(--ink-2); margin: 0; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; }
th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid var(--grid); white-space: nowrap; }
tr:last-child td { border-bottom: 0; }
th { color: var(--muted); font-weight: 500; font-size: 12px; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
td.wrap { white-space: normal; min-width: 280px; }
code { font: 12px ui-monospace, Consolas, monospace; color: var(--ink-2); }
.bar { position: relative; width: 120px; height: 14px; display: inline-block; vertical-align: middle; margin-right: 8px; }
.bar i { position: absolute; left: 0; top: 4px; height: 6px; background: var(--grid); width: 100%; border-radius: 3px; }
.bar b { position: absolute; left: 0; top: 4px; height: 6px; background: var(--series-1); border-radius: 3px; }
.bar u { position: absolute; top: 1px; height: 2px; background: var(--ink-2); }
.ci { color: var(--muted); font-size: 12px; }
.spark { vertical-align: middle; }
.cell { font-variant-numeric: tabular-nums; }
.cell.pass { background: color-mix(in srgb, var(--good) 10%, transparent); }
.cell.flaky { background: color-mix(in srgb, var(--warning) 22%, transparent); }
.cell.fail { background: color-mix(in srgb, var(--critical) 16%, transparent); }
.icon { font-weight: 700; display: inline-block; width: 14px; }
.pass .icon, .ok { color: var(--good-ink); }
.flaky .icon, .warn { color: var(--warning-ink); }
.fail .icon, .bad { color: var(--critical); }
.legend { display: flex; gap: 16px; color: var(--ink-2); font-size: 12px; margin: 0 0 8px; flex-wrap: wrap; }
.legend .pass, .legend .flaky, .legend .fail { background: none; }
.note { color: var(--muted); font-size: 12px; margin: 8px 0 0; }
summary { cursor: pointer; color: var(--ink-2); font-weight: 600; margin: 24px 0 8px; }
#tip { position: fixed; display: none; max-width: 340px; padding: 8px 10px; background: var(--surface); color: var(--ink);
       border: 1px solid var(--border); border-radius: 8px; box-shadow: 0 4px 16px rgba(0,0,0,0.18);
       font-size: 12px; pointer-events: none; z-index: 10; white-space: pre-line; }
[data-tip]:focus-visible { outline: 2px solid var(--series-1); outline-offset: -2px; }
"""

_SCRIPT = """
(function () {
  var tip = document.getElementById('tip');
  function show(el, x, y) {
    tip.textContent = el.getAttribute('data-tip');
    tip.style.display = 'block';
    var w = tip.offsetWidth, h = tip.offsetHeight;
    tip.style.left = Math.max(8, Math.min(x + 12, innerWidth - w - 8)) + 'px';
    tip.style.top = (y + h + 20 > innerHeight ? y - h - 12 : y + 12) + 'px';
  }
  function hide() { tip.style.display = 'none'; }
  document.querySelectorAll('[data-tip]').forEach(function (el) {
    el.addEventListener('mousemove', function (e) { show(el, e.clientX, e.clientY); });
    el.addEventListener('mouseleave', hide);
    el.addEventListener('focus', function () {
      var r = el.getBoundingClientRect(); show(el, r.left, r.bottom);
    });
    el.addEventListener('blur', hide);
  });
  addEventListener('keydown', function (e) { if (e.key === 'Escape') hide(); });
})();
"""


def collect_dashboard_data(
    store: RunStore, baselines_dir: str | Path | None = None, include_mock: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """{task: [one entry per model, best pass rate first]} from each model's latest full run."""
    tasks: dict[str, list[dict[str, Any]]] = {}
    for task, model in store.task_models():
        if model == "mock" and not include_mock:
            continue
        run_id = store.latest_run_id(task, model, full_only=True)
        if run_id is None:
            continue
        run = store.get_run(run_id)
        # Only runs on this same dataset: a harder golden set is not a model getting worse.
        history = [
            r for r in reversed(store.run_history(task, model, limit=30))
            if not r["is_partial"] and not r["is_aborted"]
            and r["dataset_sha256"] == run["meta"].get("dataset_sha256")
        ]
        has_baseline = baselines_dir is not None and find_baseline(baselines_dir, task, model) is not None
        tasks.setdefault(task, []).append({
            "model": model,
            "run": run,
            "cases": store.case_details(run_id),
            "history": history,
            "has_baseline": has_baseline,
            "diff": diff_runs(store, task, model, run_id, baselines_dir=baselines_dir) if has_baseline else None,
        })
    for entries in tasks.values():
        entries.sort(key=lambda e: (-e["run"]["pass_rate"], e["model"]))
    return tasks


def _pass_bar(run: dict[str, Any]) -> str:
    n, rate = run["case_count"], run["pass_rate"]
    lo, hi = wilson_interval(round(rate * n), n)
    return (
        f'<span class="bar" role="img" aria-label="{rate:.0%} passed, 95% interval {lo:.0%} to {hi:.0%}">'
        f'<i></i><b style="width:{rate * 100:.1f}%"></b>'
        f'<u style="left:{lo * 100:.1f}%;width:{(hi - lo) * 100:.1f}%"></u></span>'
        f'{rate:.1%} <span class="ci">({lo:.0%}&ndash;{hi:.0%})</span>'
    )


def _sparkline(history: list[dict[str, Any]]) -> str:
    if len(history) < 2:
        return '<span class="ci">1 run</span>' if history else ""
    width, height, pad = 96, 24, 4
    step = (width - 2 * pad) / (len(history) - 1)
    points = [
        (pad + i * step, height - pad - r["pass_rate"] * (height - 2 * pad)) for i, r in enumerate(history)
    ]
    path = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    first, last = history[0]["pass_rate"], history[-1]["pass_rate"]
    x, y = points[-1]
    return (
        f'<svg class="spark" width="{width}" height="{height}" role="img" '
        f'aria-label="Pass rate over {len(history)} runs, from {first:.0%} to {last:.0%}">'
        f'<polyline points="{path}" fill="none" stroke="var(--series-1)" stroke-width="2" stroke-linejoin="round"/>'
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="var(--series-1)" stroke="var(--surface)" stroke-width="2"/></svg>'
    )


def _baseline_status(entry: dict[str, Any]) -> str:
    diff = entry["diff"]
    if not entry["has_baseline"] or diff is None:
        return '<span class="ci">no baseline</span>'
    if diff.regressions:
        return f'<span class="bad">&#10007; {len(diff.regressions)} regressed</span>'
    if diff.degraded:
        return f'<span class="warn">~ {len(diff.degraded)} flakier</span>'
    return '<span class="ok">&#10003; no regressions</span>'


def _case_status(case: dict[str, Any]) -> str:
    if case["pass_rate"] >= 1.0:
        return "pass"
    return "fail" if case["pass_rate"] <= 0.0 else "flaky"


def _models_table(entries: list[dict[str, Any]]) -> str:
    rows = []
    for e in entries:
        run = e["run"]
        repeats = f' x {run["repeats"]}' if run["repeats"] > 1 else ""
        rows.append(
            f'<tr><td>{escape(e["model"])}</td><td>{_pass_bar(run)}</td>'
            f'<td class="num">{run["mean_score"]:.3f}</td>'
            f'<td>{_sparkline(e["history"])}</td><td>{_baseline_status(e)}</td>'
            f'<td class="num">{run["case_count"]}{repeats}</td><td class="num">{run["error_count"]}</td>'
            f'<td class="num">${run["total_cost_usd"]:.4f}</td>'
            f'<td>{escape(str(run["meta"].get("judge_model", "?")))}</td>'
            f'<td><code>{escape((run["meta"].get("git_sha") or "")[:7])}</code></td></tr>'
        )
    return (
        '<div class="card scroll"><table><thead><tr><th>Model</th><th>Pass rate (95% interval)</th>'
        '<th class="num">Mean score</th><th>Trend</th><th>Vs baseline</th><th class="num">Cases</th>'
        '<th class="num">Errors</th><th class="num">Cost</th><th>Judge</th><th>Code</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>'
    )


def _retrieval_table(entries: list[dict[str, Any]]) -> str:
    with_metrics = [e for e in entries if e["run"]["meta"].get("retrieval")]
    if not with_metrics:
        return ""
    columns = ("recall@1", "recall@3", "recall@5", "precision@1", "precision@3", "ndcg@3", "mrr")
    head = "".join(f'<th class="num">{c.replace("ndcg", "nDCG").replace("mrr", "MRR")}</th>' for c in columns)
    rows = "".join(
        f'<tr><td>{escape(e["model"])}</td>'
        + "".join(f'<td class="num">{e["run"]["meta"]["retrieval"][c]:.3f}</td>' for c in columns)
        + "</tr>"
        for e in with_metrics
    )
    return (
        f'<h3>Retrieval quality</h3><div class="card scroll"><table><thead><tr><th>Model</th>{head}</tr></thead>'
        f"<tbody>{rows}</tbody></table></div>"
    )


def _matrix(entries: list[dict[str, Any]]) -> str:
    case_ids = sorted({cid for e in entries for cid in e["cases"]})
    head = "".join(f"<th>{escape(e['model'])}</th>" for e in entries)
    rows = []
    for cid in case_ids:
        cells = []
        for e in entries:
            case = e["cases"].get(cid)
            if case is None:
                cells.append('<td class="ci">not run</td>')
                continue
            status = _case_status(case)
            icon, label = _STATUS[status]
            value = f'{round(case["pass_rate"] * case["n_samples"])}/{case["n_samples"]}' if status == "flaky" \
                else f'{case["score"]:.2f}'
            lines = [f'{cid} on {e["model"]}: {label}, score {case["score"]:.2f}']
            if case["error"]:
                lines.append(f'Error: {case["error"]}')
            lines += [f'{f["scorer"]}: {f["reason"] or "failed"}' for f in case["failed"]]
            tip = escape("\n".join(lines), quote=True)
            cells.append(f'<td class="cell {status}" tabindex="0" data-tip="{tip}">'
                         f'<span class="icon">{icon}</span> {value}</td>')
        rows.append(f'<tr><td><code>{escape(cid)}</code></td>{"".join(cells)}</tr>')
    legend = "".join(
        f'<span class="{key}"><span class="icon">{icon}</span> {label}</span>' for key, (icon, label) in _STATUS.items()
    )
    return (
        f'<h3>Every case, by model</h3><p class="legend">{legend}</p>'
        f'<div class="card scroll"><table><thead><tr><th>Case</th>{head}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>'
    )


def _failures(entries: list[dict[str, Any]]) -> str:
    rows = []
    for e in entries:
        for cid, case in sorted(e["cases"].items()):
            if case["error"]:
                rows.append((e["model"], cid, "error", case["error"]))
            rows += [(e["model"], cid, f["scorer"], f["reason"]) for f in case["failed"]]
    if not rows:
        return ""
    body = "".join(
        f"<tr><td>{escape(m)}</td><td><code>{escape(c)}</code></td><td>{escape(s)}</td>"
        f'<td class="wrap">{escape(r or "failed the check")}</td></tr>'
        for m, c, s, r in rows
    )
    return (
        f"<details><summary>What failed and why ({len(rows)})</summary>"
        '<div class="card scroll"><table><thead><tr><th>Model</th><th>Case</th><th>Check</th><th>Reason</th></tr></thead>'
        f"<tbody>{body}</tbody></table></div></details>"
    )


def render_dashboard(tasks: dict[str, list[dict[str, Any]]], generated_at: str | None = None) -> str:
    generated_at = generated_at or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    if tasks:
        sections = "".join(
            f"<section><h2>{escape(task)}</h2>{_models_table(entries)}{_retrieval_table(entries)}"
            f"{_matrix(entries)}{_failures(entries)}</section>"
            for task, entries in sorted(tasks.items())
        )
    else:
        sections = '<p class="note">No full runs yet. Run <code>harness run</code> without --smoke first.</p>'
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>Eval Dashboard</title><style>{_STYLE}</style></head><body><main>"
        f'<h1>Eval Dashboard</h1><p class="sub">Latest full run of each model. Generated {escape(generated_at)}.</p>'
        f'{sections}<p class="note">With tens of cases, pass rates carry wide intervals. '
        "Use <code>harness compare</code> for a paired test before calling one model better.</p>"
        f'</main><div id="tip" role="tooltip"></div><script>{_SCRIPT}</script></body></html>'
    )


def write_dashboard(store: RunStore, out_path: str | Path, baselines_dir: str | Path | None = None,
                    include_mock: bool = False) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_dashboard(collect_dashboard_data(store, baselines_dir, include_mock)), encoding="utf-8")
    return out_path
