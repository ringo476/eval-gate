from harness.core.models import EvalCase, RunTrace
from harness.scorers.tool_usage import ToolUsageScorer


def make_case(expected):
    return EvalCase(id="c1", task="tool_agent_qa", input={"query": "q"}, expected=expected)


def make_trace(tool_calls):
    return RunTrace(case_id="c1", output_text="answer", tool_calls=tool_calls)


def test_passes_when_expected_tool_called():
    case = make_case({"expected_tools": ["calculator"]})
    trace = make_trace([{"name": "calculator", "input": {"expression": "1+1"}, "id": "1"}])
    result = ToolUsageScorer().score(case, trace)
    assert result.passed is True
    assert result.score == 1.0


def test_fails_when_no_tool_called():
    case = make_case({"expected_tools": ["calculator"]})
    trace = make_trace([])
    result = ToolUsageScorer().score(case, trace)
    assert result.passed is False
    assert result.score == 0.0


def test_partial_credit_for_multi_tool_case():
    case = make_case({"expected_tools": ["calculator", "kb_lookup"]})
    trace = make_trace([{"name": "calculator", "input": {}, "id": "1"}])
    result = ToolUsageScorer().score(case, trace)
    assert result.passed is False
    assert result.score == 0.5


def test_no_requirement_auto_passes():
    case = make_case({"expected_tools": []})
    trace = make_trace([])
    result = ToolUsageScorer().score(case, trace)
    assert result.passed is True


def test_checks_argument_content_when_specified():
    case = make_case({"expected_tools": ["calculator"], "expected_arg_contains": "42"})
    trace_good = make_trace([{"name": "calculator", "input": {"expression": "42*2"}, "id": "1"}])
    trace_bad = make_trace([{"name": "calculator", "input": {"expression": "1*2"}, "id": "1"}])

    assert ToolUsageScorer().score(case, trace_good).passed is True
    assert ToolUsageScorer().score(case, trace_bad).passed is False
