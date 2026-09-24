"""
CLI entry point.

    python -m harness.cli run --model gpt-6-luna --smoke 8
    python -m harness.cli gate --model gpt-6-luna --min-score 0.75
    python -m harness.cli calibrate --provider openai
    python -m harness.cli history --model gpt-6-luna

Every subcommand defaults to --provider mock unless --provider openai (or
--provider anthropic) is passed explicitly, so an unfamiliar reader
cannot accidentally spend money by copy-pasting a command from the
README. OpenAI is the primary/documented provider for this project;
Anthropic is supported by the same ModelProvider abstraction but isn't
the default -- see PROJECT_GUIDE.md.
"""
from __future__ import annotations

import argparse
import logging
import sys

from harness.config import load_config
from harness.core.budget import BudgetExceededError, BudgetGuard
from harness.core.cache import ResponseCache
from harness.core.dataset import filter_smoke, load_jsonl
from harness.core.diff import diff_runs
from harness.core.models import RunTrace
from harness.core.providers import get_provider
from harness.core.report import render_run_report, write_report
from harness.core.runner import Runner
from harness.core.store import RunStore
from harness.logging_setup import setup_logging
from harness.tasks import get_task

logger = logging.getLogger("harness.cli")

DEFAULT_CALIBRATION = "golden/calibration_labels.jsonl"

# Used when HARNESS_JUDGE_MODEL isn't set in .env -- a judge model name
# only makes sense paired with a matching provider, so the default is
# looked up by whichever --provider was actually chosen, not hardcoded
# to one vendor.
DEFAULT_JUDGE_MODEL_BY_PROVIDER = {
    "openai": "gpt-6-luna",
    "anthropic": "claude-haiku-4-5-20251001",
    "mock": "mock",
}


def _resolve_judge_model(config, provider_name: str) -> str:
    return config.judge_model or DEFAULT_JUDGE_MODEL_BY_PROVIDER.get(provider_name, "gpt-6-luna")


def cmd_run(args: argparse.Namespace) -> int:
    config = load_config()
    cache = None if args.no_cache else ResponseCache(config.cache_db_path)
    budget = BudgetGuard(args.max_cost_usd or config.max_cost_usd)

    api_key = None if args.provider == "mock" else config.require_api_key(args.provider)
    judge_model = _resolve_judge_model(config, args.provider)

    provider = get_provider(args.provider, args.model, api_key, cache, budget)
    judge_provider = get_provider(args.provider, judge_model, api_key, cache, budget)

    task_spec = get_task(args.task)
    dataset_path = args.dataset or task_spec.dataset_path
    cases = load_jsonl(dataset_path)
    if args.smoke is not None:
        cases = filter_smoke(cases, n=args.smoke)
    logger.info("loaded %d cases from %s", len(cases), dataset_path)

    adapter = task_spec.adapter_factory()
    scorers = task_spec.scorers_factory(judge_provider)
    runner = Runner(adapter, scorers, provider, max_workers=args.max_workers or config.max_workers)

    try:
        summary = runner.run(cases)
    except BudgetExceededError as e:
        logger.error(str(e))
        return 2

    store = RunStore(config.runs_db_path)
    store.save(summary)
    diff = diff_runs(store, summary.task, summary.model, summary.run_id)
    report_path = write_report(summary, diff)
    store.close()
    if cache:
        cache.close()

    print(render_run_report(summary, diff))
    print(f"\nReport written to {report_path}")
    print(f"Budget spent: ${budget.spent:.4f} / ${budget.max_cost_usd:.2f}")

    return 0 if summary.error_count == 0 else 1


def cmd_diff(args: argparse.Namespace) -> int:
    config = load_config()
    store = RunStore(config.runs_db_path)
    run_id = args.run_id or store.latest_run_id(args.task, args.model)
    if run_id is None:
        print(f"No runs found for task={args.task} model={args.model}", file=sys.stderr)
        return 2
    diff = diff_runs(store, args.task, args.model, run_id)
    store.close()

    print(f"Baseline: {diff.baseline_run_id or '(none)'}")
    print(f"Current:  {diff.current_run_id}")
    print(f"Regressions ({len(diff.regressions)}): {', '.join(diff.regressions) or '(none)'}")
    print(f"Improvements ({len(diff.improvements)}): {', '.join(diff.improvements) or '(none)'}")
    print(f"Unchanged: {diff.unchanged_pass} passing, {diff.unchanged_fail} still failing")
    return 1 if diff.regressions else 0


def cmd_gate(args: argparse.Namespace) -> int:
    """Exit non-zero if the latest run does not meet the bar. This is what
    a CI workflow calls to block a merge/deploy on a quality regression."""
    config = load_config()
    store = RunStore(config.runs_db_path)
    run_id = args.run_id or store.latest_run_id(args.task, args.model)
    if run_id is None:
        print(f"No runs found for task={args.task} model={args.model}", file=sys.stderr)
        return 2

    dataset_path = args.dataset or get_task(args.task).dataset_path
    cases = load_jsonl(dataset_path)
    pinned_ids = {c.id for c in cases if c.is_pinned}
    case_results = store.case_results_for_run(run_id)
    diff = diff_runs(store, args.task, args.model, run_id)
    store.close()

    mean_score = sum(v["score"] for v in case_results.values()) / max(len(case_results), 1)
    pinned_failed = [cid for cid in pinned_ids if cid in case_results and not case_results[cid]["passed"]]

    ok = True
    if mean_score < args.min_score:
        print(f"FAIL: mean_score {mean_score:.3f} < required {args.min_score:.3f}")
        ok = False
    if pinned_failed:
        print(f"FAIL: pinned regression case(s) failing: {', '.join(pinned_failed)}")
        ok = False
    if diff.regressions:
        print(f"FAIL: {len(diff.regressions)} case(s) regressed vs baseline {diff.baseline_run_id}: "
              f"{', '.join(diff.regressions)}")
        ok = False

    if ok:
        print(f"PASS: mean_score={mean_score:.3f}, no pinned failures, no regressions (run {run_id})")
    return 0 if ok else 1


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Tests the judge, not the target app: for each (question, candidate
    answer, human_label) triple in golden/calibration_labels.jsonl, asks
    the judge to score the candidate and compares it to the human label.
    Run this whenever the judge model or its prompt changes."""
    import numpy as np

    config = load_config()
    api_key = None if args.provider == "mock" else config.require_api_key(args.provider)
    judge_model = _resolve_judge_model(config, args.provider)
    cache = None if args.no_cache else ResponseCache(config.cache_db_path)
    judge_provider = get_provider(args.provider, judge_model, api_key, cache)

    from harness.scorers.llm_judge import LLMJudgeScorer
    judge = LLMJudgeScorer(judge_provider)

    cases = load_jsonl(args.dataset)
    human_scores, judge_scores, rows = [], [], []

    for case in cases:
        trace = RunTrace(case_id=case.id, output_text=case.input.get("candidate_answer", ""), model=judge_provider.name)
        result = judge.score(case, trace)
        human_scores.append(case.human_label if case.human_label is not None else 0.0)
        judge_scores.append(result.score)
        rows.append((case.id, case.human_label, result.score, abs((case.human_label or 0) - result.score)))

    if cache:
        cache.close()

    human_arr, judge_arr = np.array(human_scores), np.array(judge_scores)
    mae = float(np.mean(np.abs(human_arr - judge_arr)))
    close_agreement = float(np.mean(np.abs(human_arr - judge_arr) <= 0.2))

    # corrcoef is undefined (divide by zero) when either series has zero
    # variance -- e.g. every judge score came back 0 because the judge
    # errored on every case. That's a real calibration failure, not a
    # crash, so report it as such instead of letting numpy warn and emit nan.
    if len(cases) > 1 and np.std(human_arr) > 0 and np.std(judge_arr) > 0:
        corr = float(np.corrcoef(human_arr, judge_arr)[0, 1])
    else:
        corr = None

    from tabulate import tabulate
    print(tabulate(rows, headers=["case", "human_label", "judge_score", "abs_diff"], floatfmt=".2f"))
    print()
    print(f"Mean absolute error:        {mae:.3f}")
    print(f"Pearson correlation:        {corr:.3f}" if corr is not None else "Pearson correlation:        undefined (a score series had zero variance)")
    print(f"Agreement within +/-0.2:    {close_agreement:.1%}")
    print(f"Judge cost this run:        ${judge.total_cost_usd:.4f}")
    print()
    if mae > 0.25 or corr is None or corr < 0.5:
        print("WARNING: judge calibration is weak. Do not trust llm_judge scores "
              "for gating until the judge prompt or model is improved.")
        return 1
    print("Judge calibration looks reasonable -- see CALIBRATION.md for the methodology.")
    return 0


def cmd_history(args: argparse.Namespace) -> int:
    config = load_config()
    store = RunStore(config.runs_db_path)
    rows = store.run_history(args.task, args.model, limit=args.limit)
    store.close()

    from tabulate import tabulate
    print(tabulate(
        [(r["run_id"], r["timestamp"], f"{r['mean_score']:.3f}", f"{r['pass_rate']:.1%}",
          f"${r['total_cost_usd']:.4f}", r["error_count"]) for r in rows],
        headers=["run_id", "timestamp", "mean_score", "pass_rate", "cost", "errors"],
    ))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="harness", description="Eval harness CLI")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--task", default="rag_fusion_qa")
    common.add_argument("--model", default="gpt-6-luna")

    p_run = sub.add_parser("run", parents=[common], help="Run the golden set against the target app")
    p_run.add_argument("--provider", choices=["openai", "anthropic", "mock"], default="mock")
    p_run.add_argument("--dataset", default=None, help="Overrides the task's default golden set")
    p_run.add_argument("--smoke", type=int, default=None, help="Run only the first N cases (+ all pinned)")
    p_run.add_argument("--max-cost-usd", type=float, default=None)
    p_run.add_argument("--max-workers", type=int, default=None)
    p_run.add_argument("--no-cache", action="store_true")
    p_run.set_defaults(func=cmd_run)

    p_diff = sub.add_parser("diff", parents=[common], help="Show regressions vs the previous run")
    p_diff.add_argument("--run-id", default=None)
    p_diff.set_defaults(func=cmd_diff)

    p_gate = sub.add_parser("gate", parents=[common], help="Exit non-zero if quality bar is not met (for CI)")
    p_gate.add_argument("--dataset", default=None, help="Overrides the task's default golden set")
    p_gate.add_argument("--run-id", default=None)
    p_gate.add_argument("--min-score", type=float, default=0.7)
    p_gate.set_defaults(func=cmd_gate)

    p_cal = sub.add_parser("calibrate", help="Check the LLM judge against hand-labeled examples")
    p_cal.add_argument("--provider", choices=["openai", "anthropic", "mock"], default="mock")
    p_cal.add_argument("--dataset", default=DEFAULT_CALIBRATION)
    p_cal.add_argument("--no-cache", action="store_true")
    p_cal.set_defaults(func=cmd_calibrate)

    p_hist = sub.add_parser("history", parents=[common], help="Show run history for a task/model")
    p_hist.add_argument("--limit", type=int, default=20)
    p_hist.set_defaults(func=cmd_history)

    return parser


def main() -> int:
    # Model output (or a case's query text) can contain arbitrary Unicode;
    # Windows consoles default to a narrow codepage (cp1252) that can't
    # represent it. Re-configuring stdout/stderr to UTF-8 here means a
    # printed report never crashes the run over a character it didn't
    # expect -- worst case a glyph doesn't render, the process doesn't die.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.verbose)
    try:
        return args.func(args)
    except Exception as e:
        logger.error("fatal: %s", e, exc_info=args.verbose)
        return 3


if __name__ == "__main__":
    sys.exit(main())
