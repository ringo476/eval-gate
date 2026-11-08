"""
Deterministic check on agent behavior: did it call the tool(s) the case
expects, with plausible arguments? Checked independently of whether the
final answer sounds right -- a correct-sounding answer built by skipping
the tool (i.e. the model just guessed) is exactly the failure mode this
scorer exists to catch.

`expected_tools` has three meanings, and the difference matters:

  key absent            -> no requirement; the case doesn't constrain tools
  ["calculator", ...]   -> each listed tool must be called
  []  (explicit empty)  -> NO tool may be called. This is how a golden case
                           asserts "the model must not force a tool call
                           when neither tool applies".
"""
from __future__ import annotations

from harness.core.models import EvalCase, RunTrace, ScoreResult
from harness.scorers.base import Scorer


class ToolUsageScorer(Scorer):
    name = "tool_usage"

    def score(self, case: EvalCase, trace: RunTrace) -> ScoreResult:
        if "expected_tools" not in case.expected:
            return ScoreResult(self.name, score=1.0, passed=True, details={"note": "no requirement"}, applicable=False)

        expected_tools = case.expected["expected_tools"] or []
        called_names = [tc["name"] for tc in trace.tool_calls]

        if not expected_tools:
            if not called_names:
                return ScoreResult(self.name, score=1.0, passed=True)
            return ScoreResult(
                self.name, score=0.0, passed=False,
                details={"expected_tools": [], "called_tools": called_names,
                         "note": "case requires that no tool be called"},
            )

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
