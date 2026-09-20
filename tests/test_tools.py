from harness.adapters.tools import calculator, execute_tool


def test_calculator_basic_arithmetic():
    assert calculator("2 + 3") == "5"
    assert calculator("(12 + 8) * 3") == "60"
    assert calculator("144 / 12") == "12.0"


def test_calculator_rejects_non_arithmetic():
    # must not execute arbitrary code -- these should error, not run
    assert calculator("__import__('os').system('echo hi')").startswith("error")
    assert calculator("open('/etc/passwd').read()").startswith("error")
    assert calculator("[x for x in range(10)]").startswith("error")


def test_calculator_division_by_zero_is_a_clean_error_not_a_crash():
    result = calculator("100 / 0")
    assert result.startswith("error")


def test_calculator_malformed_expression_is_a_clean_error():
    result = calculator("2 +")
    assert result.startswith("error")


def test_execute_tool_dispatches_by_name():
    assert execute_tool("calculator", {"expression": "1+1"}) == "2"
    assert execute_tool("unknown_tool", {}).startswith("error")


def test_kb_lookup_returns_relevant_entry():
    result = execute_tool("kb_lookup", {"topic": "coral reef bleaching"})
    assert "coral" in result.lower() or "bleach" in result.lower()
