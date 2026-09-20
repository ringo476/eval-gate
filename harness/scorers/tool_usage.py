"""
Deterministic check on agent behavior: did it call the tool(s) the case
expects, with plausible arguments? Checked independently of whether the
final answer sounds right -- a correct-sounding answer built by skipping
the tool (i.e. the model just guessed) is exactly the failure mode this
scorer exists to catch.
"""
from __future__ import annotations

from harness.core.models import EvalCase, RunTrace, ScoreResult
from harness.scorers.base import Scorer


class ToolUsageScorer(Scorer):
    name = "tool_usage"

    def score(self, case: EvalCase, trace: RunTrace) -> ScoreResult:
        expected_tools = case.expected.get("expected_tools", [])
        if not expected_tools:
            return ScoreResult(self.name, score=1.0, passed=True, details={"note": "no requirement"})

        called_names = [tc["name"] for tc in trace.tool_calls]
        called_set = set(called_names)
        expected_set = set(expected_tools)

        hits = expected_set & called_set
        hit_rate = len(hits) / len(expected_set)

        arg_substr = case.expected.get("expected_arg_contains")
        args_ok = True
        if arg_substr and hit_rate == 1.0:
            all_args = " ".join(str(tc.get("input", "")) for tc in trace.tool_calls).lower()
            args_ok = arg_substr.lower() in all_args

        passed = (hit_rate == 1.0) and args_ok
        details = {}
        if not passed:
            details = {
                "expected_tools": sorted(expected_set),
                "called_tools": called_names,
                "args_ok": args_ok,
            }
        return ScoreResult(self.name, score=hit_rate if args_ok else hit_rate * 0.5, passed=passed, details=details)
