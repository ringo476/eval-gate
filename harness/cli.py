"""
CLI entry point.

    python -m harness.cli doctor
    python -m harness.cli run --task rag_fusion_qa --provider ollama --model llama3.1:8b --judge-provider ollama --judge-model gpt-oss:20b
    python -m harness.cli run --task rag_fusion_qa --provider openai --model gpt-6-luna --repeats 3 --temperature 0.7
    python -m harness.cli baseline set --task rag_fusion_qa --model gpt-6-luna
    python -m harness.cli gate --task rag_fusion_qa --model gpt-6-luna --min-score 0.75 --require-baseline
    python -m harness.cli compare --task rag_fusion_qa --models gpt-6-luna llama3.1:8b
    python -m harness.cli calibrate --provider openai
    python -m harness.cli history --model gpt-6-luna

Every subcommand defaults to --provider mock unless another provider is
passed explicitly, so an unfamiliar reader cannot accidentally spend money
by copy-pasting a command from the README.

Exit codes (stable, so CI can rely on them):
    0  success / gate passed
    1  checked and failed: gate failed, regressions found, samples errored,
       or the judge is poorly calibrated
    2  nothing to act on: no run / no baseline found, or the run was aborted
       (e.g. budget cap hit)
    3  fatal error (bad config, missing key, invalid dataset, ...)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from harness import __version__
from harness.config import PROVIDERS, Config, load_config
from harness.core.baseline import (
    BaselineError,
    baseline_from_store,
    baseline_path,
    find_baseline,
    load_baseline,
    slugify,
    write_baseline,
)
from harness.core.budget import BudgetExceededError, BudgetGuard
from harness.core.cache import ResponseCache
from harness.core.dataset import filter_smoke, load_jsonl
from harness.core.diff import DEFAULT_REGRESSION_THRESHOLD, diff_runs
from harness.core.gate import evaluate_gate
from harness.core.models import RunTrace
from harness.core.providers import GenerationSettings, check_ollama, get_provider, list_ollama_models
from harness.core.report import render_comparison, render_run_report, write_report
from harness.core.runmeta import collect_meta, file_sha256
from harness.core.runner import Runner
from harness.core.stats import pearson
from harness.core.store import RunStore
from harness.logging_setup import setup_logging
from harness.tasks import TASKS, get_task

logger = logging.getLogger("harness.cli")

DEFAULT_CALIBRATION = "golden/calibration_labels.jsonl"

# Used when no judge model is specified -- a judge model name only makes
# sense paired with a matching provider, so the default is looked up by
# whichever provider the judge runs on, not hardcoded to one vendor.
DEFAULT_JUDGE_MODEL_BY_PROVIDER = {
    "openai": "gpt-6-luna",
    "anthropic": "claude-haiku-4-5-20251001",
    "ollama": "gpt-oss:20b",
    "mock": "mock",
}

# Judge replies are tiny, but reasoning models spend output tokens thinking
# before they answer; too small a cap truncates the verdict to nothing.
MIN_JUDGE_OUTPUT_TOKENS = 2048

# "Dangerous" miss: a clearly bad answer (human label <= this) that the judge
# nevertheless passes. For a regression gate this is the failure that matters.
BAD_ANSWER_LABEL_MAX = 0.2
GOOD_ANSWER_LABEL_MIN = 0.9
CAL_MAX_MAE = 0.25
CAL_MIN_CORRELATION = 0.5
CAL_MAX_FALSE_ACCEPT_RATE = 0.2
CAL_MAX_INVALID_RATE = 0.1


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _task_settings(config: Config, args: argparse.Namespace) -> GenerationSettings:
    temperature = args.temperature if args.temperature is not None else config.temperature
    return GenerationSettings(
        temperature=temperature,
        max_output_tokens=args.max_output_tokens or config.max_output_tokens,
    )


def _judge_settings(config: Config, args: argparse.Namespace) -> GenerationSettings:
    # Always temperature 0: the judge should be as repeatable as the API allows.
    return GenerationSettings(
        temperature=0.0,
        max_output_tokens=max(args.max_output_tokens or config.max_output_tokens, MIN_JUDGE_OUTPUT_TOKENS),
    )


def _make_provider(provider_name, model, config, cache, budget, settings, allow_unpriced=False):
    api_key = config.require_api_key(provider_name)
    if provider_name == "ollama":
        check_ollama(model, config.ollama_base_url)
    return get_provider(
        provider_name, model, api_key, cache, budget,
        settings=settings, base_url=config.ollama_base_url, allow_unpriced=allow_unpriced,
    )


def _resolve_judge(args: argparse.Namespace, config: Config, provider_name: str) -> tuple[str, str]:
    judge_provider = args.judge_provider or provider_name
    judge_model = args.judge_model or config.judge_model or DEFAULT_JUDGE_MODEL_BY_PROVIDER[judge_provider]
    return judge_provider, judge_model


def _baseline_kwargs(config: Config, args: argparse.Namespace) -> dict:
    return {
        "baselines_dir": config.baselines_dir,
        "baseline_path": getattr(args, "baseline", None),
        "threshold": getattr(args, "regression_threshold", DEFAULT_REGRESSION_THRESHOLD),
    }


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> int:
    config = load_config()
    if args.repeats < 1:
        raise RuntimeError("--repeats must be >= 1")

    task_settings = _task_settings(config, args)
    if args.repeats > 1 and task_settings.temperature == 0:
        logger.warning("--repeats %d with temperature 0: samples will be near-identical, so repeating buys "
                       "little. Use --temperature 0.5 (or similar) to measure real run-to-run variance.", args.repeats)

    judge_provider_name, judge_model = _resolve_judge(args, config, args.provider)
    self_judged = args.provider != "mock" and judge_provider_name == args.provider and judge_model == args.model
    if self_judged:
        message = (f"judge and task model are both '{args.model}': LLM judges favour their own output, "
                   "so llm_judge scores will be optimistic")
        if args.strict_judge:
            raise RuntimeError(f"{message}. Choose a different --judge-model/--judge-provider "
                               "(or drop --strict-judge to accept the bias).")
        logger.warning(message)

    cache = None if args.no_cache else ResponseCache(config.cache_db_path)
    budget = BudgetGuard(args.max_cost_usd or config.max_cost_usd)
    store = None
    try:
        provider = _make_provider(args.provider, args.model, config, cache, budget, task_settings, args.allow_unpriced)
        judge_provider = _make_provider(
            judge_provider_name, judge_model, config, cache, budget, _judge_settings(config, args), args.allow_unpriced,
        )

        task_spec = get_task(args.task)
        dataset_path = args.dataset or task_spec.dataset_path
        all_cases = load_jsonl(dataset_path, expected_task=args.task)
        dataset_sha = file_sha256(dataset_path)
        cases = filter_smoke(all_cases, n=args.smoke) if args.smoke is not None else all_cases
        logger.info("loaded %d cases from %s (running %d)", len(all_cases), dataset_path, len(cases))

        adapter = task_spec.adapter_factory()
        scorers = task_spec.scorers_factory(judge_provider)
        runner = Runner(adapter, scorers, provider, max_workers=args.max_workers or config.max_workers,
                        repeats=args.repeats)

        summary = runner.run(cases)
        summary.meta = collect_meta(
            provider=args.provider, model=args.model,
            judge_provider=judge_provider_name, judge_model=judge_model, self_judged=self_judged,
            temperature=task_settings.temperature, max_output_tokens=task_settings.max_output_tokens,
            repeats=args.repeats, dataset_path=str(dataset_path), dataset_sha256=dataset_sha,
            dataset_case_count=len(all_cases), partial=len(cases) < len(all_cases), smoke=args.smoke,
        )

        store = RunStore(config.runs_db_path)
        store.save(summary)
        diff = diff_runs(store, summary.task, summary.model, summary.run_id, **_baseline_kwargs(config, args))
        report_path = write_report(summary, diff, out_dir=config.reports_dir)

        print(render_run_report(summary, diff))
        print(f"\nReport written to {report_path}")
        print(f"Budget spent: ${budget.spent:.4f} / ${budget.max_cost_usd:.2f}")

        if summary.aborted:
            return 2
        return 0 if summary.error_count == 0 else 1
    except BudgetExceededError as e:  # e.g. raised while building providers/scorers outside the runner
        logger.error(str(e))
        return 2
    finally:
        if store:
            store.close()
        if cache:
            cache.close()


# ---------------------------------------------------------------------------
# diff / gate
# ---------------------------------------------------------------------------

def cmd_diff(args: argparse.Namespace) -> int:
    config = load_config()
    store = RunStore(config.runs_db_path)
    try:
        run_id = args.run_id or store.latest_run_id(args.task, args.model)
        if run_id is None:
            print(f"No runs found for task={args.task} model={args.model}", file=sys.stderr)
            return 2
        diff = diff_runs(store, args.task, args.model, run_id, **_baseline_kwargs(config, args))
    finally:
        store.close()

    kind = {"blessed": "committed baseline", "previous_run": "previous local run", "none": "(none)"}[diff.baseline_kind]
    print(f"Baseline: {kind} {diff.baseline_label or ''}".rstrip())
    print(f"Current:  {diff.current_run_id}")
    print(f"Regressions ({len(diff.regressions)}): {', '.join(diff.regressions) or '(none)'}")
    print(f"Flakier ({len(diff.degraded)}): {', '.join(diff.degraded) or '(none)'}")
    print(f"Improvements ({len(diff.improvements)}): {', '.join(diff.improvements) or '(none)'}")
    print(f"Unchanged: {diff.unchanged_pass} passing, {diff.unchanged_fail} still failing")
    return 1 if diff.regressions else 0


def cmd_gate(args: argparse.Namespace) -> int:
    """Exit non-zero if the latest run does not meet the bar. This is what
    a CI workflow calls to block a merge/deploy on a quality regression.
    The rules themselves live in harness/core/gate.py."""
    config = load_config()
    store = RunStore(config.runs_db_path)
    try:
        run_id = args.run_id or store.latest_run_id(args.task, args.model)
        if run_id is None:
            print(f"No runs found for task={args.task} model={args.model}", file=sys.stderr)
            return 2
        run = store.get_run(run_id)
        if run is None:
            print(f"Run '{run_id}' not found", file=sys.stderr)
            return 2

        dataset_path = args.dataset or get_task(args.task).dataset_path
        cases = load_jsonl(dataset_path, expected_task=args.task)
        pinned_ids = {c.id for c in cases if c.is_pinned}
        case_results = store.case_results_for_run(run_id)
        diff = diff_runs(store, args.task, args.model, run_id, **_baseline_kwargs(config, args))
    finally:
        store.close()

    verdict = evaluate_gate(
        run=run, case_results=case_results, pinned_ids=pinned_ids, diff=diff,
        min_score=args.min_score, min_pass_rate=args.min_pass_rate, max_errors=args.max_errors,
        require_baseline=args.require_baseline, allow_partial=args.allow_partial,
        current_dataset_sha=file_sha256(dataset_path), fail_on_degraded=args.fail_on_degraded,
    )

    for warning in verdict.warnings:
        print(f"WARN: {warning}")
    for failure in verdict.failures:
        print(f"FAIL: {failure}")
    if verdict.ok:
        print(f"PASS: mean_score={verdict.mean_score:.3f}, pass_rate={verdict.pass_rate:.1%}, "
              f"no pinned failures, no regressions (run {run_id})")
        return 0
    if run["is_aborted"]:
        return 2
    return 1


# ---------------------------------------------------------------------------
# baseline
# ---------------------------------------------------------------------------

def cmd_baseline_set(args: argparse.Namespace) -> int:
    config = load_config()
    store = RunStore(config.runs_db_path)
    try:
        run_id = args.run_id or store.latest_run_id(args.task, args.model, full_only=True)
        if run_id is None:
            print(f"No full (non-smoke, non-aborted) run found for task={args.task} model={args.model}. "
                  "Run `harness run` without --smoke first.", file=sys.stderr)
            return 2
        baseline = baseline_from_store(store, run_id, allow_errors=args.force)
    finally:
        store.close()

    path = Path(args.out) if args.out else baseline_path(config.baselines_dir, args.task, args.model)
    previous = load_baseline(path) if path.exists() else None
    write_baseline(baseline, path)

    s = baseline.summary
    print(f"Wrote baseline {path}")
    print(f"  from run {run_id}: pass_rate={s['pass_rate']:.1%} mean_score={s['mean_score']:.3f} over {s['n_cases']} cases")
    if previous:
        p = previous.summary
        print(f"  replaced previous baseline (pass_rate={p['pass_rate']:.1%} mean_score={p['mean_score']:.3f}, "
              f"run {previous.source_run_id})")
    print(f"Review the change and commit it:  git add {path}")
    return 0


def cmd_baseline_show(args: argparse.Namespace) -> int:
    config = load_config()
    baseline = find_baseline(config.baselines_dir, args.task, args.model)
    if baseline is None:
        print(f"No baseline at {baseline_path(config.baselines_dir, args.task, args.model)}", file=sys.stderr)
        return 2
    s = baseline.summary
    print(f"{baseline.path}")
    print(f"  task={baseline.task} model={baseline.model} created={baseline.created_at}")
    print(f"  source run: {baseline.source_run_id}")
    print(f"  pass_rate={s['pass_rate']:.1%} mean_score={s['mean_score']:.3f} cases={s['n_cases']} repeats={s.get('repeats', 1)}")
    for key in ("git_sha", "judge_model", "temperature", "dataset_sha256"):
        if key in baseline.meta:
            print(f"  {key}: {baseline.meta[key]}")
    failing = sorted(cid for cid, c in baseline.cases.items() if c["pass_rate"] < 1.0)
    print(f"  cases not at 100% in the baseline ({len(failing)}): {', '.join(failing) or '(none)'}")
    return 0


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

def cmd_compare(args: argparse.Namespace) -> int:
    config = load_config()
    store = RunStore(config.runs_db_path)
    try:
        runs, results, missing = [], {}, []
        for model in args.models:
            run_id = store.latest_run_id(args.task, model, full_only=True)
            if run_id is None:
                missing.append(model)
                continue
            runs.append(store.get_run(run_id))
            results[run_id] = store.case_results_for_run(run_id)
    finally:
        store.close()

    if missing:
        print(f"No full run found for: {', '.join(missing)} (task={args.task}). Run `harness run` for each first.",
              file=sys.stderr)
        return 2
    text = render_comparison(args.task, runs, results)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"Written to {args.out}")
    return 0


# ---------------------------------------------------------------------------
# calibrate
# ---------------------------------------------------------------------------

def cmd_calibrate(args: argparse.Namespace) -> int:
    """Tests the judge, not the target app: for each (question, candidate
    answer, human_label) triple in golden/calibration_labels.jsonl, asks
    the judge to score the candidate and compares it to the human label.
    Run this whenever the judge model or its prompt changes."""
    from tabulate import tabulate

    from harness.scorers.llm_judge import LLMJudgeScorer

    config = load_config()
    model = args.judge_model or config.judge_model or DEFAULT_JUDGE_MODEL_BY_PROVIDER[args.provider]
    settings = GenerationSettings(
        temperature=args.temperature,
        max_output_tokens=max(args.max_output_tokens or config.max_output_tokens, MIN_JUDGE_OUTPUT_TOKENS),
    )
    cache = None if args.no_cache else ResponseCache(config.cache_db_path)
    budget = BudgetGuard(args.max_cost_usd or config.max_cost_usd)
    try:
        judge_provider = _make_provider(args.provider, model, config, cache, budget, settings, args.allow_unpriced)
        cases = load_jsonl(args.dataset, expected_task="judge_calibration")

        rows, valid_human, valid_judge, invalid = [], [], [], 0
        by_tag: dict[str, list[float]] = {}
        stabilities: list[float] = []
        total_cost = 0.0

        for i, case in enumerate(cases, start=1):
            samples, bad = [], 0
            for r in range(args.repeats):
                judge = LLMJudgeScorer(judge_provider.scoped(r))
                trace = RunTrace(case_id=case.id, output_text=case.input.get("candidate_answer", ""),
                                 model=judge_provider.name)
                result = judge.score(case, trace)
                total_cost += judge.total_cost_usd
                if "error" in result.details:
                    bad += 1
                else:
                    samples.append(result.score)
            human = case.human_label if case.human_label is not None else 0.0
            if not samples:
                invalid += 1
                rows.append((case.id, human, "INVALID", "-", ",".join(case.tags)))
                continue
            mean = sum(samples) / len(samples)
            if len(samples) > 1:
                stabilities.append((sum((x - mean) ** 2 for x in samples) / len(samples)) ** 0.5)
            valid_human.append(human)
            valid_judge.append(mean)
            for tag in case.tags:
                by_tag.setdefault(tag, []).append(abs(human - mean))
            rows.append((case.id, human, round(mean, 2), round(abs(human - mean), 2), ",".join(case.tags)))
            logger.info("calibrated %d/%d", i, len(cases))
    finally:
        if cache:
            cache.close()

    n_valid = len(valid_human)
    n_total = len(cases)
    invalid_rate = invalid / n_total
    mae = sum(abs(h - j) for h, j in zip(valid_human, valid_judge, strict=True)) / n_valid if n_valid else float("nan")
    close_agreement = sum(abs(h - j) <= 0.2 for h, j in zip(valid_human, valid_judge, strict=True)) / n_valid if n_valid else 0.0
    corr = pearson(valid_human, valid_judge)

    bad_pairs = [(h, j) for h, j in zip(valid_human, valid_judge, strict=True) if h <= BAD_ANSWER_LABEL_MAX]
    good_pairs = [(h, j) for h, j in zip(valid_human, valid_judge, strict=True) if h >= GOOD_ANSWER_LABEL_MIN]
    false_accept = sum(j >= 0.7 for _, j in bad_pairs) / len(bad_pairs) if bad_pairs else 0.0
    false_reject = sum(j < 0.7 for _, j in good_pairs) / len(good_pairs) if good_pairs else 0.0

    print(tabulate(rows, headers=["case", "human", "judge", "abs_diff", "tags"], floatfmt=".2f"))
    print()
    print(f"Judge:                      {model} ({args.provider}), temperature {args.temperature}, repeats {args.repeats}")
    print(f"Cases scored / invalid:     {n_valid} / {invalid}  (invalid = reply wasn't a usable score)")
    print(f"Mean absolute error:        {mae:.3f}   (max {CAL_MAX_MAE})")
    print(f"Pearson correlation:        {corr:.3f}   (min {CAL_MIN_CORRELATION})" if corr is not None
          else "Pearson correlation:        undefined (a score series had zero variance)")
    print(f"Agreement within +/-0.2:    {close_agreement:.1%}")
    print(f"False accepts:              {false_accept:.1%} of {len(bad_pairs)} clearly-bad answers passed (max {CAL_MAX_FALSE_ACCEPT_RATE:.0%})")
    print(f"False rejects:              {false_reject:.1%} of {len(good_pairs)} clearly-good answers failed")
    if stabilities:
        print(f"Self-inconsistency:         mean per-case std {sum(stabilities) / len(stabilities):.3f} across {args.repeats} repeats")
    worst_tags = sorted(((sum(v) / len(v), t, len(v)) for t, v in by_tag.items()), reverse=True)[:6]
    if worst_tags:
        print("Hardest categories (mean abs error): " + ", ".join(f"{t} {e:.2f} (n={n})" for e, t, n in worst_tags))
    print(f"Judge cost this run:        ${total_cost:.4f}")
    print()

    weak_reasons = []
    if invalid_rate > CAL_MAX_INVALID_RATE:
        weak_reasons.append(f"{invalid_rate:.0%} of replies were unusable")
    if n_valid and mae > CAL_MAX_MAE:
        weak_reasons.append(f"MAE {mae:.3f} > {CAL_MAX_MAE}")
    if corr is None or corr < CAL_MIN_CORRELATION:
        weak_reasons.append("correlation undefined or too low")
    if false_accept > CAL_MAX_FALSE_ACCEPT_RATE:
        weak_reasons.append(f"false-accept rate {false_accept:.0%} > {CAL_MAX_FALSE_ACCEPT_RATE:.0%}")

    reports_dir = Path(config.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = reports_dir / f"calibration-{slugify(model)}-{stamp}.json"
    out_path.write_text(json.dumps({
        "judge_model": model, "provider": args.provider, "temperature": args.temperature,
        "repeats": args.repeats, "n_cases": n_total, "n_valid": n_valid, "invalid": invalid,
        "mae": mae, "pearson": corr, "agreement_within_0.2": close_agreement,
        "false_accept_rate": false_accept, "false_reject_rate": false_reject,
        "self_inconsistency": (sum(stabilities) / len(stabilities)) if stabilities else None,
        "weak_reasons": weak_reasons, "dataset_sha256": file_sha256(args.dataset),
        "rows": [list(r) for r in rows],
    }, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Calibration written to {out_path}")

    if weak_reasons:
        print("WARNING: judge calibration is weak (" + "; ".join(weak_reasons) + "). "
              "Do not trust llm_judge scores for gating until the judge model or prompt improves.")
        return 1
    print("Judge calibration looks reasonable -- record it in CALIBRATION.md.")
    return 0


# ---------------------------------------------------------------------------
# history
# ---------------------------------------------------------------------------

def cmd_history(args: argparse.Namespace) -> int:
    config = load_config()
    store = RunStore(config.runs_db_path)
    rows = store.run_history(args.task, args.model, limit=args.limit)
    store.close()

    from tabulate import tabulate
    print(tabulate(
        [(r["run_id"], r["timestamp"][:19], f"{r['mean_score']:.3f}", f"{r['pass_rate']:.1%}",
          f"${r['total_cost_usd']:.4f}", r["error_count"], r["repeats"],
          "aborted" if r["is_aborted"] else ("partial" if r["is_partial"] else "full"), r["git_sha"])
         for r in rows],
        headers=["run_id", "timestamp", "mean_score", "pass_rate", "cost", "errors", "repeats", "kind", "git"],
    ))
    return 0


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------

def cmd_doctor(args: argparse.Namespace) -> int:
    """Preflight: everything the harness needs, checked in one place, with
    an actionable message for anything missing. Never prints key values."""
    import importlib
    import platform

    problems = 0

    def report(status: str, text: str) -> None:
        nonlocal problems
        if status == "fail":
            problems += 1
        print(f"[{status:>4}] {text}")

    report("ok" if sys.version_info >= (3, 10) else "fail", f"Python {platform.python_version()} (need >= 3.10)")

    for module, required in (("openai", True), ("chromadb", True), ("sentence_transformers", True),
                             ("numpy", True), ("tabulate", True), ("tenacity", True), ("dotenv", True),
                             ("anthropic", False)):
        try:
            importlib.import_module(module)
            report("ok", f"package {module}")
        except ImportError:
            report("fail" if required else "warn",
                   f"package {module} not installed" + ("" if required else " (only needed for --provider anthropic)"))

    try:
        config = load_config()
    except RuntimeError as e:
        report("fail", f"config: {e}")
        return 3
    report("ok", "config loaded")

    target = Path("targets/rag-fusion/main.py")
    report("ok" if target.exists() else "fail",
           "vendored target app present" if target.exists() else
           "targets/rag-fusion missing -- run `bash scripts/setup_target.sh` (scripts/setup_target.ps1 on Windows)")

    for name, spec in TASKS.items():
        try:
            cases = load_jsonl(spec.dataset_path, expected_task=name)
            pinned = sum(1 for c in cases if c.is_pinned)
            report("ok", f"dataset {spec.dataset_path}: {len(cases)} cases, {pinned} pinned")
        except (OSError, ValueError) as e:
            report("fail", f"dataset {spec.dataset_path}: {e}")
        baselines_dir = Path(config.baselines_dir)
        found = sorted(baselines_dir.glob(f"{slugify(name)}__*.json")) if baselines_dir.exists() else []
        report("ok" if found else "info",
               f"baselines for {name}: {', '.join(p.stem.split('__', 1)[1] for p in found) or 'none yet'}")

    report("ok" if config.openai_api_key else "info", "OPENAI_API_KEY " + ("is set" if config.openai_api_key else "not set (needed for --provider openai)"))
    report("ok" if config.anthropic_api_key else "info", "ANTHROPIC_API_KEY " + ("is set" if config.anthropic_api_key else "not set (optional)"))

    try:
        names = list_ollama_models(config.ollama_base_url)
        report("ok", f"Ollama reachable at {config.ollama_base_url}: {', '.join(names) or 'no models pulled'}")
    except RuntimeError as e:
        report("info", f"Ollama not reachable ({str(e).split('(')[0].strip()}) -- only needed for --provider ollama")

    print()
    print("All required checks passed." if not problems else f"{problems} required check(s) failed.")
    return 0 if not problems else 3


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="harness", description="Eval harness for LLM applications")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--version", action="version", version=f"harness {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--task", default="rag_fusion_qa", choices=sorted(TASKS))
    common.add_argument("--model", default="gpt-6-luna")

    baseline_opts = argparse.ArgumentParser(add_help=False)
    baseline_opts.add_argument("--baseline", default=None, help="Compare against this baseline file instead of the default location")
    baseline_opts.add_argument("--regression-threshold", type=float, default=DEFAULT_REGRESSION_THRESHOLD,
                               help="A case regresses when its pass rate drops by at least this much (default 0.5)")

    gen_opts = argparse.ArgumentParser(add_help=False)
    gen_opts.add_argument("--max-output-tokens", type=int, default=None, help="Cap on tokens per model reply")
    gen_opts.add_argument("--allow-unpriced", action="store_true",
                          help="Run a paid model that has no entry in pricing.py (the budget cap can't protect it)")
    gen_opts.add_argument("--no-cache", action="store_true")
    gen_opts.add_argument("--max-cost-usd", type=float, default=None)

    p_run = sub.add_parser("run", parents=[common, baseline_opts, gen_opts], help="Run the golden set against the target app")
    p_run.add_argument("--provider", choices=PROVIDERS, default="mock")
    p_run.add_argument("--judge-provider", choices=PROVIDERS, default=None, help="Provider for the judge (default: same as --provider)")
    p_run.add_argument("--judge-model", default=None, help="Judge model (default: HARNESS_JUDGE_MODEL or a per-provider default)")
    p_run.add_argument("--strict-judge", action="store_true", help="Refuse to run if judge and task model are the same")
    p_run.add_argument("--dataset", default=None, help="Overrides the task's default golden set")
    p_run.add_argument("--smoke", type=int, default=None, help="Run only the first N cases (+ all pinned). Not gateable.")
    p_run.add_argument("--repeats", type=int, default=1, help="Samples per case; >1 measures flakiness (use with --temperature > 0)")
    p_run.add_argument("--temperature", type=float, default=None, help="Sampling temperature (default HARNESS_TEMPERATURE, 0)")
    p_run.add_argument("--max-workers", type=int, default=None)
    p_run.set_defaults(func=cmd_run)

    p_diff = sub.add_parser("diff", parents=[common, baseline_opts], help="Show regressions vs the baseline")
    p_diff.add_argument("--run-id", default=None)
    p_diff.set_defaults(func=cmd_diff)

    p_gate = sub.add_parser("gate", parents=[common, baseline_opts], help="Exit non-zero if the quality bar is not met (for CI)")
    p_gate.add_argument("--dataset", default=None, help="Overrides the task's default golden set")
    p_gate.add_argument("--run-id", default=None)
    p_gate.add_argument("--min-score", type=float, default=0.7)
    p_gate.add_argument("--min-pass-rate", type=float, default=None)
    p_gate.add_argument("--max-errors", type=int, default=0, help="Errored samples tolerated (default 0)")
    p_gate.add_argument("--require-baseline", action="store_true", help="Fail unless a committed baseline file exists")
    p_gate.add_argument("--allow-partial", action="store_true", help="Accept a --smoke run (not recommended)")
    p_gate.add_argument("--fail-on-degraded", action="store_true", help="Treat flakier-than-baseline cases as failures")
    p_gate.set_defaults(func=cmd_gate)

    p_base = sub.add_parser("baseline", help="Manage committed baselines")
    base_sub = p_base.add_subparsers(dest="baseline_command", required=True)
    p_bset = base_sub.add_parser("set", parents=[common], help="Bless a full run as the baseline (then commit the file)")
    p_bset.add_argument("--run-id", default=None, help="Default: latest full run for (task, model)")
    p_bset.add_argument("--out", default=None, help="Default: baselines/<task>__<model>.json")
    p_bset.add_argument("--force", action="store_true", help="Allow blessing a run that had errored samples")
    p_bset.set_defaults(func=cmd_baseline_set)
    p_bshow = base_sub.add_parser("show", parents=[common], help="Print the committed baseline")
    p_bshow.set_defaults(func=cmd_baseline_show)

    p_cmp = sub.add_parser("compare", help="Side-by-side comparison of models on one task")
    p_cmp.add_argument("--task", default="rag_fusion_qa", choices=sorted(TASKS))
    p_cmp.add_argument("--models", nargs="+", required=True, help="Models with an existing full run")
    p_cmp.add_argument("--out", default=None, help="Also write the comparison to this Markdown file")
    p_cmp.set_defaults(func=cmd_compare)

    p_cal = sub.add_parser("calibrate", parents=[gen_opts], help="Check the LLM judge against hand-labeled examples")
    p_cal.add_argument("--provider", choices=PROVIDERS, default="mock", help="Provider the judge runs on")
    p_cal.add_argument("--judge-model", default=None)
    p_cal.add_argument("--dataset", default=DEFAULT_CALIBRATION)
    p_cal.add_argument("--repeats", type=int, default=1, help="Judge each case N times to measure self-consistency")
    p_cal.add_argument("--temperature", type=float, default=0.0)
    p_cal.set_defaults(func=cmd_calibrate)

    p_hist = sub.add_parser("history", parents=[common], help="Show run history for a task/model")
    p_hist.add_argument("--limit", type=int, default=20)
    p_hist.set_defaults(func=cmd_history)

    p_doc = sub.add_parser("doctor", help="Check the environment: packages, datasets, keys, local models")
    p_doc.set_defaults(func=cmd_doctor)

    return parser


def main(argv: list[str] | None = None) -> int:
    # Model output (or a case's query text) can contain arbitrary Unicode;
    # Windows consoles default to a narrow codepage (cp1252) that can't
    # represent it. Re-configuring stdout/stderr to UTF-8 here means a
    # printed report never crashes the run over a character it didn't
    # expect -- worst case a glyph doesn't render, the process doesn't die.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        logger.error("interrupted")
        return 130
    except (BaselineError, RuntimeError, ValueError, OSError) as e:
        logger.error("fatal: %s", e, exc_info=args.verbose)
        return 3
    except Exception as e:  # noqa: BLE001 -- last resort: never show a raw traceback unless -v
        logger.error("fatal: %s: %s", type(e).__name__, e, exc_info=args.verbose)
        return 3


if __name__ == "__main__":
    sys.exit(main())
