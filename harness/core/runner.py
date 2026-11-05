"""
Orchestrates one harness run: executes the adapter for every case
concurrently (bounded worker pool), applies every scorer, and never lets
one case's exception take down the run. A per-case failure -- a network
error the retry policy gave up on, a malformed judge response -- becomes a
failed CaseResult with a recorded error, not a crashed process. That's the
difference between "the run finished and 3 cases errored" (useful) and
"the run died at case 47 of 150" (useless).

Two things are deliberately NOT isolated per case:

  * BudgetExceededError. Once the spend cap is hit, every remaining case
    would fail the same way (and burn wall-clock time doing it), so the run
    is aborted: pending work is cancelled and the summary is flagged
    `aborted`, which keeps it out of baselines and fails any gate.

  * Repeats. With repeats=N every case is executed N times as independent
    samples (each a fresh draw -- see CachingProvider.scoped), so a case's
    outcome is a pass *rate*, which is what lets regression detection
    tell a flaky case from a genuinely broken one.
"""
from __future__ import annotations

import logging
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed

from harness.adapters.base import Adapter
from harness.core.budget import BudgetExceededError
from harness.core.models import CaseResult, EvalCase, RunSummary, ScoreResult
from harness.core.providers import ModelProvider
from harness.core.store import new_run_id, now_iso
from harness.scorers.base import Scorer

logger = logging.getLogger("harness.runner")


class Runner:
    def __init__(
        self,
        adapter: Adapter,
        scorers: list[Scorer],
        provider: ModelProvider,
        max_workers: int = 4,
        repeats: int = 1,
    ):
        if repeats < 1:
            raise ValueError("repeats must be >= 1")
        self.adapter = adapter
        self.scorers = scorers
        self.provider = provider
        self.max_workers = max_workers
        self.repeats = repeats

    def _provider_for(self, sample_idx: int) -> ModelProvider:
        scoped = getattr(self.provider, "scoped", None)
        return scoped(sample_idx) if scoped else self.provider

    def _run_one(self, case: EvalCase, sample_idx: int = 0) -> CaseResult:
        trace = self.adapter.run(case, self._provider_for(sample_idx))

        if trace.error:
            logger.warning("case %s (sample %d) errored: %s", case.id, sample_idx, trace.error)
            return CaseResult(case=case, trace=trace, scores=[], sample_idx=sample_idx)

        # A case can opt out of scorers that don't apply to it, e.g. embedding
        # similarity when `reference_answer` describes *behavior* ("should
        # decline") rather than giving a sample answer to compare against.
        skip = set(case.expected.get("skip_scorers", []))
        unknown = skip - {s.name for s in self.scorers}
        if unknown:
            logger.warning("case %s skips unknown scorer(s): %s", case.id, ", ".join(sorted(unknown)))

        scores = []
        for scorer in self.scorers:
            if scorer.name in skip:
                continue
            try:
                scores.append(scorer.score(case, trace))
            except BudgetExceededError:
                raise
            except Exception as e:  # noqa: BLE001 -- a broken scorer must not sink the case
                logger.error("scorer %s raised on case %s: %s", scorer.name, case.id, e)
                scores.append(ScoreResult(scorer.name, score=0.0, passed=False,
                                          details={"scorer_error": str(e)}))
        return CaseResult(case=case, trace=trace, scores=scores, sample_idx=sample_idx)

    def run(self, cases: list[EvalCase]) -> RunSummary:
        run_id = new_run_id()
        total = len(cases) * self.repeats
        logger.info("starting run %s: task=%s model=%s cases=%d repeats=%d",
                    run_id, self.adapter.task, self.provider.name, len(cases), self.repeats)

        results: list[CaseResult] = []
        abort_reason: str | None = None
        done = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self._run_one, case, sample_idx): (case, sample_idx)
                for case in cases for sample_idx in range(self.repeats)
            }
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except CancelledError:
                    continue
                except BudgetExceededError as e:
                    if abort_reason is None:
                        abort_reason = str(e)
                        logger.error("aborting run %s: %s", run_id, e)
                        for pending in futures:
                            pending.cancel()
                    continue
                done += 1
                if done % 10 == 0 or done == total:
                    logger.info("progress: %d/%d samples done", done, total)

        # keep deterministic ordering matching the input dataset, not
        # completion order, so reports/diffs are stable across reruns
        order = {c.id: i for i, c in enumerate(cases)}
        results.sort(key=lambda cr: (order[cr.case.id], cr.sample_idx))

        summary = RunSummary(
            run_id=run_id,
            task=self.adapter.task,
            model=self.provider.name,
            timestamp=now_iso(),
            case_results=results,
            aborted=abort_reason is not None,
            abort_reason=abort_reason,
        )
        logger.info(
            "run %s %s: pass_rate=%.2f mean_score=%.2f cost=$%.4f errors=%d",
            run_id, "ABORTED" if summary.aborted else "complete",
            summary.pass_rate, summary.mean_score, summary.total_cost_usd, summary.error_count,
        )
        return summary
