"""Renders a RunSummary (and optionally a DiffResult) as a Markdown
report -- this is the artifact you'd actually paste into a PR description
or interview writeup, not a database row -- plus a JSON twin for machines,
and a side-by-side multi-model comparison."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from harness.core.diff import DiffResult
from harness.core.models import RunSummary
from harness.core.stats import percentile, wilson_interval


def _cell(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _scorer_stats(summary: RunSummary) -> dict[str, dict[str, float]]:
    """Per-scorer pass rate over every (non-errored) sample -- shows *which*
    kind of check is failing, which the overall pass rate hides."""
    stats: dict[str, dict[str, float]] = {}
    for cr in summary.case_results:
        for s in cr.scores:
            entry = stats.setdefault(s.scorer_name, {"passed": 0, "total": 0, "score_sum": 0.0})
            entry["total"] += 1
            entry["passed"] += int(s.passed)
            entry["score_sum"] += s.score
    return {
        name: {"passed": e["passed"], "total": e["total"], "pass_rate": e["passed"] / e["total"],
               "mean_score": e["score_sum"] / e["total"]}
        for name, e in stats.items()
    }


def _latency_stats(summary: RunSummary) -> tuple[float, float]:
    values = [c.trace.latency_ms for c in summary.case_results if not c.trace.error and not c.trace.cache_hit]
    return percentile(values, 50), percentile(values, 95)


def render_run_report(summary: RunSummary, diff: DiffResult | None = None) -> str:
    cases = summary.cases
    meta = summary.meta
    n_passed = sum(1 for c in cases if c.passed)
    lo, hi = wilson_interval(n_passed, len(cases))
    p50, p95 = _latency_stats(summary)

    lines = [
        f"# Eval run `{summary.run_id}`",
        "",
        f"- **Task:** {summary.task}",
        f"- **Model:** {summary.model}" + (f" ({meta['provider']})" if meta.get("provider") else ""),
        f"- **Judge:** {meta.get('judge_model', 'n/a')}" + (f" ({meta['judge_provider']})" if meta.get("judge_provider") else ""),
        f"- **Timestamp:** {summary.timestamp}",
        f"- **Cases:** {len(cases)}" + (f" x {summary.repeats} samples" if summary.repeats > 1 else ""),
        f"- **Pass rate:** {summary.pass_rate:.1%}  (95% CI {lo:.0%}-{hi:.0%}, n={len(cases)})",
        f"- **Mean score:** {summary.mean_score:.3f}",
        f"- **Total cost:** ${summary.total_cost_usd:.4f}",
        f"- **Latency (uncached) p50 / p95:** {p50 / 1000:.1f}s / {p95 / 1000:.1f}s",
        f"- **Errors:** {summary.error_count}",
    ]
    if summary.flaky_cases:
        lines.append(f"- **Flaky cases ({len(summary.flaky_cases)}):** {', '.join(summary.flaky_cases)}")
    if meta.get("temperature") is not None:
        lines.append(f"- **Temperature:** {meta['temperature']}")
    if meta.get("git_sha"):
        dirty = " (uncommitted changes)" if meta.get("git_dirty") else ""
        lines.append(f"- **Code:** `{meta['git_sha'][:10]}`{dirty}")
    if meta.get("dataset_sha256"):
        lines.append(f"- **Dataset:** `{meta.get('dataset_path', '?')}` sha256 `{meta['dataset_sha256'][:12]}`")
    lines.append("")

    if summary.aborted:
        lines += [f"> **ABORTED:** {summary.abort_reason}", "> Results below are incomplete and not eligible as a baseline.", ""]
    if summary.is_partial:
        lines += [
            f"> **PARTIAL RUN:** {len(cases)} of {meta.get('dataset_case_count', '?')} cases (--smoke). "
            "Not eligible as a baseline, and the gate will refuse it.",
            "",
        ]
    if meta.get("self_judged"):
        lines += [
            "> **Warning:** the judge is the same model as the task model. LLM judges favour their own output, "
            "so `llm_judge` scores are optimistic here. Use a different judge for gating.",
            "",
        ]

    if diff is not None:
        lines += ["## Regression check", ""]
        if diff.baseline_kind == "none":
            lines.append("_No baseline for this (task, model), so regressions can't be detected. "
                         "Run `harness baseline set` after a good full run and commit the file._")
        else:
            kind = "committed baseline" if diff.baseline_kind == "blessed" else "previous local run (not a committed baseline)"
            lines.append(f"Compared against the {kind}: `{diff.baseline_label}`.")
            if diff.baseline_pass_rate is not None:
                lines.append(f"Baseline pass rate {diff.baseline_pass_rate:.1%}, mean score "
                             f"{(diff.baseline_mean_score or 0):.3f}; this run {summary.pass_rate:.1%}, {summary.mean_score:.3f}.")
            lines.append("")
            if diff.regressions:
                lines.append(f"**REGRESSIONS ({len(diff.regressions)}):** {', '.join(diff.regressions)}")
            else:
                lines.append("No regressions.")
            if diff.degraded:
                lines.append(f"**Flakier than baseline ({len(diff.degraded)}):** {', '.join(diff.degraded)}")
            if diff.improvements:
                lines.append(f"**Improvements ({len(diff.improvements)}):** {', '.join(diff.improvements)}")
            if diff.new_cases:
                lines.append(f"**New cases:** {', '.join(diff.new_cases)}")
            if diff.missing_cases and not summary.is_partial:
                lines.append(f"**Missing vs baseline ({len(diff.missing_cases)}):** {', '.join(diff.missing_cases)}")
            lines.append(f"Unchanged: {diff.unchanged_pass} passing, {diff.unchanged_fail} still failing.")
        lines.append("")

    scorer_stats = _scorer_stats(summary)
    if scorer_stats:
        lines += ["## By scorer", "", "| Scorer | Passed | Pass rate | Mean score |", "|---|---|---|---|"]
        for name, s in scorer_stats.items():
            lines.append(f"| {name} | {s['passed']}/{s['total']} | {s['pass_rate']:.0%} | {s['mean_score']:.2f} |")
        lines.append("")

    multi = summary.repeats > 1
    header = "| Case | Status | " + ("Pass rate | " if multi else "") + "Score | Cost | Latency | Error |"
    sep = "|---|---|" + ("---|" if multi else "") + "---|---|---|---|"
    lines += ["## Case results", "", header, sep]
    for agg in cases:
        status = {"pass": "PASS", "fail": "FAIL", "flaky": "FLAKY"}[agg.status]
        rate = f"{agg.n_passed}/{agg.n_samples} | " if multi else ""
        err = _cell(agg.representative.trace.error or "") if agg.error_count else ""
        lines.append(
            f"| {agg.case.id} | {status} | {rate}{agg.mean_score:.2f} | "
            f"${agg.total_cost_usd:.4f} | {agg.mean_latency_ms:.0f}ms | {err} |"
        )

    failing = [c for c in cases if c.status != "pass"]
    if failing:
        lines += ["", "## Failure detail", ""]
        for agg in failing:
            rep = agg.representative
            lines.append(f"### {agg.case.id}" + (f" ({agg.n_passed}/{agg.n_samples} samples passed)" if multi else ""))
            lines.append(f"- Input: `{agg.case.input}`")
            lines.append(f"- Output: {_cell(rep.trace.output_text[:300]) or '(empty)'}")
            if rep.trace.error:
                lines.append(f"- Error: {_cell(rep.trace.error)}")
            for s in rep.scores:
                if not s.passed:
                    lines.append(f"- **{s.scorer_name}** failed (score={s.score:.2f}): {s.details}")
            lines.append("")

    return "\n".join(lines)


def summary_to_dict(summary: RunSummary, diff: DiffResult | None = None) -> dict[str, Any]:
    cases = summary.cases
    n_passed = sum(1 for c in cases if c.passed)
    lo, hi = wilson_interval(n_passed, len(cases))
    p50, p95 = _latency_stats(summary)
    return {
        "run_id": summary.run_id,
        "task": summary.task,
        "model": summary.model,
        "timestamp": summary.timestamp,
        "aborted": summary.aborted,
        "abort_reason": summary.abort_reason,
        "meta": summary.meta,
        "metrics": {
            "n_cases": len(cases),
            "repeats": summary.repeats,
            "pass_rate": summary.pass_rate,
            "pass_rate_ci95": [lo, hi],
            "mean_score": summary.mean_score,
            "total_cost_usd": summary.total_cost_usd,
            "latency_p50_ms": p50,
            "latency_p95_ms": p95,
            "error_count": summary.error_count,
            "flaky_cases": summary.flaky_cases,
        },
        "by_scorer": _scorer_stats(summary),
        "cases": [
            {
                "id": c.case.id, "status": c.status, "pass_rate": c.pass_rate, "n_samples": c.n_samples,
                "mean_score": c.mean_score, "cost_usd": c.total_cost_usd, "mean_latency_ms": c.mean_latency_ms,
                "error": c.representative.trace.error if c.error_count else None,
                "tags": c.case.tags,
            }
            for c in cases
        ],
        "diff": asdict(diff) if diff is not None else None,
    }


def write_report(summary: RunSummary, diff: DiffResult | None, out_dir: str = "reports") -> Path:
    """Writes <run_id>.md and <run_id>.json; returns the Markdown path."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    md_path = Path(out_dir) / f"{summary.run_id}.md"
    md_path.write_text(render_run_report(summary, diff), encoding="utf-8")
    json_path = Path(out_dir) / f"{summary.run_id}.json"
    json_path.write_text(json.dumps(summary_to_dict(summary, diff), indent=2, default=str) + "\n", encoding="utf-8")
    return md_path


def render_comparison(task: str, runs: list[dict[str, Any]], results: dict[str, dict[str, dict[str, Any]]]) -> str:
    """Side-by-side view of several models on the same task.

    runs:    store.get_run() dicts, one per model (the latest full run of each)
    results: {run_id: store.case_results_for_run(run_id)}
    """
    labels = [r["model"] for r in runs]
    lines = [f"# Model comparison: `{task}`", ""]

    lines += ["| Model | Pass rate (95% CI) | Mean score | Cost | Errors | Judge | Run |", "|---|---|---|---|---|---|---|"]
    for r in runs:
        n = r["case_count"]
        k = round(r["pass_rate"] * n)
        lo, hi = wilson_interval(k, n)
        judge = r["meta"].get("judge_model", "?")
        flag = " (self-judged)" if r["meta"].get("self_judged") else ""
        lines.append(
            f"| {r['model']} | {r['pass_rate']:.1%} ({lo:.0%}-{hi:.0%}) | {r['mean_score']:.3f} | "
            f"${r['total_cost_usd']:.4f} | {r['error_count']} | {judge}{flag} | `{r['run_id']}` |"
        )
    lines.append("")
    lines.append("_With a small golden set the confidence intervals are wide: differences smaller than the "
                 "interval overlap are not evidence one model is better._")
    lines.append("")

    all_cases = sorted({cid for res in results.values() for cid in res})
    disagree = []
    for cid in all_cases:
        outcomes = [results[r["run_id"]].get(cid) for r in runs]
        flags = [None if o is None else o["passed"] for o in outcomes]
        if len(set(flags)) > 1:
            disagree.append((cid, outcomes))

    lines += [f"## Cases where models disagree ({len(disagree)} of {len(all_cases)})", ""]
    if disagree:
        lines += ["| Case | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
        for cid, outcomes in disagree:
            cells = [
                "(not run)" if o is None else f"{'PASS' if o['passed'] else 'FAIL'} ({o['score']:.2f})"
                for o in outcomes
            ]
            lines.append(f"| {cid} | " + " | ".join(cells) + " |")
    else:
        lines.append("All models agree on every case.")
    lines.append("")
    return "\n".join(lines)
