"""
The CI gate policy, as a pure function.

`harness gate` is what a pipeline calls to block a merge/deploy. Keeping the
decision logic here (no I/O, no argparse) means every rule is unit-tested
directly, and the CLI is just plumbing around it.

A run PASSES the gate only if all of these hold:
  1. it wasn't aborted (e.g. budget cap hit mid-run);
  2. it covered the full dataset (a --smoke subset can't vouch for the rest),
     unless --allow-partial;
  3. it ran against the *current* dataset file (not a stale run from before
     the golden set changed);
  4. no sample errored (more than --max-errors);
  5. mean score >= --min-score (and pass rate >= --min-pass-rate if given);
  6. every pinned case passed every sample;
  7. no case regressed against the baseline, and none that the baseline
     covered have gone missing (deleting a failing case must not be a way
     to make the gate green);
  8. a committed baseline exists, if --require-baseline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from harness.core.diff import DiffResult


@dataclass
class GateVerdict:
    ok: bool
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    mean_score: float = 0.0
    pass_rate: float = 0.0


def evaluate_gate(
    *,
    run: dict[str, Any],
    case_results: dict[str, dict[str, Any]],
    pinned_ids: set[str],
    diff: DiffResult,
    min_score: float,
    min_pass_rate: float | None = None,
    max_errors: int = 0,
    require_baseline: bool = False,
    allow_partial: bool = False,
    current_dataset_sha: str | None = None,
    fail_on_degraded: bool = False,
) -> GateVerdict:
    failures: list[str] = []
    warnings: list[str] = []

    n = max(len(case_results), 1)
    mean_score = sum(v["score"] for v in case_results.values()) / n
    pass_rate = sum(1 for v in case_results.values() if v["passed"]) / n

    if run.get("is_aborted"):
        failures.append("run was aborted before completing (budget cap or fatal error) -- results are incomplete")

    if run.get("is_partial") and not allow_partial:
        failures.append(
            f"run covered only {run.get('case_count')} cases (--smoke); the gate requires a full run "
            "(pass --allow-partial to override)"
        )

    run_sha = (run.get("meta") or {}).get("dataset_sha256")
    if current_dataset_sha and run_sha and run_sha != current_dataset_sha:
        failures.append("run is stale: the golden dataset has changed since it was run -- re-run `harness run`")
    elif current_dataset_sha and not run_sha:
        warnings.append("run has no dataset fingerprint (recorded by an older version); staleness not checked")

    if run.get("error_count", 0) > max_errors:
        failures.append(f"{run['error_count']} sample(s) errored (allowed: {max_errors})")

    if mean_score < min_score:
        failures.append(f"mean_score {mean_score:.3f} < required {min_score:.3f}")
    if min_pass_rate is not None and pass_rate < min_pass_rate:
        failures.append(f"pass_rate {pass_rate:.1%} < required {min_pass_rate:.1%}")

    unevaluated = sorted(pid for pid in pinned_ids if pid not in case_results)
    if unevaluated:
        failures.append(f"pinned case(s) were not evaluated: {', '.join(unevaluated)}")
    pinned_failed = sorted(pid for pid in pinned_ids if pid in case_results and case_results[pid]["pass_rate"] < 1.0)
    if pinned_failed:
        failures.append(f"pinned regression case(s) not passing every sample: {', '.join(pinned_failed)}")

    if diff.baseline_kind == "none":
        if require_baseline:
            failures.append(
                "no committed baseline for this (task, model); create one with `harness baseline set` and commit it"
            )
        else:
            warnings.append("no baseline found -- regressions cannot be detected (run `harness baseline set`)")
    else:
        if require_baseline and diff.baseline_kind != "blessed":
            failures.append(
                "--require-baseline needs a committed baseline file, but only the previous local run was available"
            )
        if diff.regressions:
            failures.append(
                f"{len(diff.regressions)} case(s) regressed vs baseline {diff.baseline_label}: "
                f"{', '.join(diff.regressions)}"
            )
        if diff.missing_cases and not run.get("is_partial"):
            failures.append(
                f"{len(diff.missing_cases)} case(s) in the baseline are missing from this run: "
                f"{', '.join(diff.missing_cases)}"
            )
        if diff.degraded:
            message = f"{len(diff.degraded)} case(s) became flakier vs baseline: {', '.join(diff.degraded)}"
            (failures if fail_on_degraded else warnings).append(message)

    return GateVerdict(ok=not failures, failures=failures, warnings=warnings,
                       mean_score=mean_score, pass_rate=pass_rate)
