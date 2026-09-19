"""
LLM-as-judge scorer.

This is the scorer most people build carelessly: they trust a model to
grade a model with no evidence it's any good at grading. Treat the judge
as an unverified classifier until `harness calibrate` has been run
against the hand-labeled set in golden/calibration_labels.jsonl and the
agreement rate is reported in the README -- see harness/cli.py's
`calibrate` command and CALIBRATION.md.
"""
from __future__ import annotations

import json
import re

from harness.core.models import EvalCase, RunTrace, ScoreResult
from harness.core.pricing import estimate_cost_usd
from harness.core.providers import ModelProvider
from harness.scorers.base import Scorer

JUDGE_SYSTEM_PROMPT = """You are a strict grader for an AI system's output. \
Score the CANDIDATE answer against the REFERENCE answer and QUESTION on a \
0-10 scale for correctness and faithfulness. The candidate must not state \
anything not supported by the reference, must not contradict it, and must \
not fabricate specifics (numbers, technologies, comparisons) that aren't in \
the reference. If the reference describes an appropriate refusal (e.g. the \
question is out of scope) and the candidate also appropriately declines, \
score that highly -- a correct refusal is a correct answer. If the \
candidate confidently answers something the reference says should have \
been declined, score it very low regardless of how plausible it sounds.

Respond with ONLY a JSON object: {"score": <0-10 integer>, "reason": "<one sentence>"}"""


class LLMJudgeScorer(Scorer):
    name = "llm_judge"

    def __init__(self, judge_provider: ModelProvider, pass_threshold: float = 0.7):
        self.judge_provider = judge_provider
        self.pass_threshold = pass_threshold
        self.total_cost_usd = 0.0  # running total for this scorer instance, for cost reporting

    def score(self, case: EvalCase, trace: RunTrace) -> ScoreResult:
        reference = case.expected.get("reference_answer", "(no reference provided)")
        question = case.input.get("question") or case.input.get("query", "")
        candidate = case.input.get("candidate_answer") or trace.output_text

        if not candidate.strip():
            return ScoreResult(self.name, score=0.0, passed=False, details={"note": "empty candidate answer"})

        prompt = (
            f"QUESTION: {question}\n\n"
            f"REFERENCE ANSWER: {reference}\n\n"
            f"CANDIDATE ANSWER: {candidate}"
        )

        response = self.judge_provider.generate(prompt, system=JUDGE_SYSTEM_PROMPT)
        if not response.cache_hit:
            self.total_cost_usd += estimate_cost_usd(
                self.judge_provider.name, response.input_tokens, response.output_tokens
            )

        parsed = _extract_json(response.text)
        if parsed is None:
            return ScoreResult(
                self.name, score=0.0, passed=False,
                details={"error": "judge did not return valid JSON", "raw": response.text[:200]},
            )

        try:
            raw_score = float(parsed.get("score", 0)) / 10.0
        except (TypeError, ValueError):
            raw_score = 0.0

        return ScoreResult(
            self.name,
            score=raw_score,
            passed=raw_score >= self.pass_threshold,
            details={"reason": parsed.get("reason", ""), "judge_model": self.judge_provider.name},
        )


def _extract_json(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
