"""
Orchestrates one harness run: executes the adapter for every case
concurrently (bounded worker pool), applies every scorer, and never lets
one case's exception take down the run. A per-case failure -- a budget
cap hit, a network error the retry policy gave up on, a malformed judge
response -- becomes a failed CaseResult with a recorded error, not a
crashed process. That's the difference between "the run finished and
3 cases errored" (useful) and "the run died at case 47 of 150" (useless).
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from harness.adapters.base import Adapter
from harness.core.models import CaseResult, EvalCase, RunSummary
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
    ):
        self.adapter = adapter
        self.scorers = scorers
        self.provider = provider
        self.max_workers = max_workers

    def _run_one(self, case: EvalCase) -> CaseResult:
        trace = self.adapter.run(case, self.provider)

        if trace.error:
            logger.warning("case %s errored: %s", case.id, trace.error)
            return CaseResult(case=case, trace=trace, scores=[])

        scores = []
        for scorer in self.scorers:
            try:
                scores.append(scorer.score(case, trace))
            except Exception as e:  # noqa: BLE001 -- a broken scorer must not sink the case
                logger.error("scorer %s raised on case %s: %s", scorer.name, case.id, e)
                from harness.core.models import ScoreResult
                scores.append(ScoreResult(scorer.name, score=0.0, passed=False,
                                           details={"scorer_error": str(e)}))
        return CaseResult(case=case, trace=trace, scores=scores)

    def run(self, cases: list[EvalCase]) -> RunSummary:
        run_id = new_run_id()
        logger.info("starting run %s: task=%s model=%s cases=%d",
                     run_id, self.adapter.task, self.provider.name, len(cases))

        case_results: list[CaseResult] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(self._run_one, case): case for case in cases}
            for future in as_completed(futures):
                case_results.append(future.result())

        # keep deterministic ordering matching the input dataset, not
        # completion order, so reports/diffs are stable across reruns
        order = {c.id: i for i, c in enumerate(cases)}
        case_results.sort(key=lambda cr: order[cr.case.id])

        summary = RunSummary(
            run_id=run_id,
            task=self.adapter.task,
            model=self.provider.name,
            timestamp=now_iso(),
            case_results=case_results,
        )
        logger.info(
            "run %s complete: pass_rate=%.2f mean_score=%.2f cost=$%.4f errors=%d",
            run_id, summary.pass_rate, summary.mean_score, summary.total_cost_usd, summary.error_count,
        )
        return summary
