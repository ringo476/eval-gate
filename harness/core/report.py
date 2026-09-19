"""Renders a RunSummary (and optionally a DiffResult) as a Markdown
report -- this is the artifact you'd actually paste into a PR description
or interview writeup, not a database row."""
from __future__ import annotations

from pathlib import Path

from harness.core.diff import DiffResult
from harness.core.models import RunSummary


def render_run_report(summary: RunSummary, diff: DiffResult | None = None) -> str:
    lines = [
        f"# Eval run `{summary.run_id}`",
        "",
        f"- **Task:** {summary.task}",
        f"- **Model:** {summary.model}",
        f"- **Timestamp:** {summary.timestamp}",
        f"- **Cases:** {len(summary.case_results)}",
        f"- **Pass rate:** {summary.pass_rate:.1%}",
        f"- **Mean score:** {summary.mean_score:.3f}",
        f"- **Total cost:** ${summary.total_cost_usd:.4f}",
        f"- **Total latency:** {summary.total_latency_ms / 1000:.1f}s",
        f"- **Errors:** {summary.error_count}",
        "",
    ]

    if diff is not None:
        lines += ["## Regression check", ""]
        if diff.baseline_run_id is None:
            lines.append("_No prior run for this (task, model) to compare against -- this is the baseline._")
        else:
            lines.append(f"Compared against `{diff.baseline_run_id}`.")
            lines.append("")
            if diff.regressions:
                lines.append(f"**REGRESSIONS ({len(diff.regressions)}):** {', '.join(diff.regressions)}")
            else:
                lines.append("No regressions.")
            if diff.improvements:
                lines.append(f"**Improvements ({len(diff.improvements)}):** {', '.join(diff.improvements)}")
            if diff.new_cases:
                lines.append(f"**New cases:** {', '.join(diff.new_cases)}")
            lines.append(f"Unchanged: {diff.unchanged_pass} passing, {diff.unchanged_fail} still failing.")
        lines.append("")

    lines += ["## Case results", "", "| Case | Passed | Score | Cost | Latency | Error |",
              "|---|---|---|---|---|---|"]
    for cr in summary.case_results:
        err = cr.trace.error or ""
        lines.append(
            f"| {cr.case.id} | {'PASS' if cr.passed else 'FAIL'} | {cr.aggregate_score:.2f} | "
            f"${cr.trace.cost_usd:.4f} | {cr.trace.latency_ms:.0f}ms | {err} |"
        )

    failing = [cr for cr in summary.case_results if not cr.passed]
    if failing:
        lines += ["", "## Failure detail", ""]
        for cr in failing:
            lines.append(f"### {cr.case.id}")
            lines.append(f"- Query: `{cr.case.input}`")
            lines.append(f"- Output: {cr.trace.output_text[:300] or '(empty)'}")
            if cr.trace.error:
                lines.append(f"- Error: {cr.trace.error}")
            for s in cr.scores:
                if not s.passed:
                    lines.append(f"- **{s.scorer_name}** failed (score={s.score:.2f}): {s.details}")
            lines.append("")

    return "\n".join(lines)


def write_report(summary: RunSummary, diff: DiffResult | None, out_dir: str = "reports") -> Path:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / f"{summary.run_id}.md"
    path.write_text(render_run_report(summary, diff), encoding="utf-8")
    return path
