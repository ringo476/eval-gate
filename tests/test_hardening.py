"""Calculator limits, dataset validation, and report rendering."""
import json

import pytest

from harness.adapters.tools import MAX_EXPRESSION_CHARS, calculator
from harness.core.dataset import load_jsonl
from harness.core.diff import BaselineRef, diff_cases
from harness.core.report import render_comparison, render_run_report, summary_to_dict, write_report
from tests.helpers import make_sample, make_summary

# --- calculator -------------------------------------------------------------


def test_calculator_survives_pathologically_nested_input():
    nested = "(" * 150 + "1" + ")" * 150
    assert calculator(nested).startswith("error") or calculator(nested) == "1"
    deep = "-" * 5000 + "1"
    assert calculator(deep).startswith("error")  # too long: refused before parsing


def test_calculator_rejects_overlong_and_non_string_input():
    assert calculator("1+" * (MAX_EXPRESSION_CHARS // 2 + 1) + "1").startswith("error: expression too long")
    assert calculator(None).startswith("error")  # type: ignore[arg-type]


def test_calculator_rejects_power_and_huge_work():
    assert calculator("9**9**9").startswith("error")  # exponentiation is not in the grammar


# --- dataset validation -----------------------------------------------------


def write(tmp_path, *rows):
    path = tmp_path / "d.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def row(**overrides):
    base = {"id": "a", "task": "t", "input": {"q": 1}, "expected": {"x": 1}}
    base.update(overrides)
    return base


@pytest.mark.parametrize("bad,message", [
    ({"id": ""}, "non-empty string"),
    ({"id": 5}, "non-empty string"),
    ({"input": "text"}, "'input' must be an object"),
    ({"expected": [1]}, "'expected' must be an object"),
    ({"tags": "pinned"}, "'tags' must be a list"),
    ({"tags": [1]}, "'tags' must be a list"),
    ({"human_label": 1.5}, "between 0 and 1"),
    ({"human_label": "high"}, "between 0 and 1"),
    ({"human_label": True}, "between 0 and 1"),
])
def test_invalid_rows_are_rejected_with_the_line_number(tmp_path, bad, message):
    with pytest.raises(ValueError, match=message) as exc:
        load_jsonl(write(tmp_path, row(**bad)))
    assert ":1:" in str(exc.value)


def test_non_object_line_is_rejected(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text("[1, 2]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_jsonl(path)


def test_expected_task_catches_a_case_in_the_wrong_file(tmp_path):
    path = write(tmp_path, row(id="a", task="t"), row(id="b", task="other"))
    assert len(load_jsonl(path)) == 2  # without the check it loads silently
    with pytest.raises(ValueError, match="has task 'other', expected 't'"):
        load_jsonl(path, expected_task="t")


def test_valid_human_label_boundaries(tmp_path):
    cases = load_jsonl(write(tmp_path, row(id="a", human_label=0), row(id="b", human_label=1.0)))
    assert [c.human_label for c in cases] == [0, 1.0]


def test_shipped_datasets_are_valid_and_well_formed():
    from harness.tasks import TASKS

    for name, spec in TASKS.items():
        cases = load_jsonl(spec.dataset_path, expected_task=name)
        assert len(cases) >= 10
        assert any(c.is_pinned for c in cases), f"{name} has no pinned cases"
    labelled = load_jsonl("golden/calibration_labels.jsonl", expected_task="judge_calibration")
    assert all(c.human_label is not None for c in labelled)
    assert len(labelled) >= 30
    assert any(c.human_label <= 0.2 for c in labelled) and any(c.human_label >= 0.9 for c in labelled)
    assert any("judge-injection" in c.tags for c in labelled)


def test_agent_dataset_asserts_the_no_tool_case_explicitly():
    cases = {c.id: c for c in load_jsonl("golden/tool_agent_qa.jsonl")}
    assert cases["ta-010"].expected["expected_tools"] == []  # explicit empty list => must not call a tool


# --- report -----------------------------------------------------------------


def _summary(**meta):
    base = {"provider": "ollama", "judge_model": "j", "judge_provider": "ollama", "git_sha": "a" * 40,
            "git_dirty": True, "temperature": 0.0, "dataset_path": "d.jsonl", "dataset_sha256": "b" * 64}
    base.update(meta)
    return make_summary("run-1", "task", "model-x", "2026-01-01T00:00:00Z", {"a": True, "b": False}, meta=base)


def test_report_shows_provenance_ci_and_scorer_breakdown():
    text = render_run_report(_summary())
    assert "95% CI" in text and "n=2" in text
    assert "`aaaaaaaaaa`" in text and "uncommitted changes" in text
    assert "## By scorer" in text and "dummy" in text
    assert "Judge:" in text and "(ollama)" in text


def test_report_warns_when_judge_is_the_task_model():
    assert "same model as the task model" in render_run_report(_summary(self_judged=True))
    assert "same model as the task model" not in render_run_report(_summary(self_judged=False))


def test_report_flags_partial_and_aborted_runs():
    partial = render_run_report(_summary(partial=True, dataset_case_count=25))
    assert "PARTIAL RUN" in partial and "of 25 cases" in partial
    aborted = _summary()
    aborted.aborted, aborted.abort_reason = True, "budget"
    assert "ABORTED" in render_run_report(aborted)


def test_report_says_when_there_is_no_baseline_and_when_there_is():
    summary = _summary()
    no_base = render_run_report(summary, diff_cases(None, "run-1", {}))
    assert "can't be detected" in no_base

    ref = BaselineRef(kind="blessed", label="baselines/x.json", run_id="old",
                      cases={"a": {"pass_rate": 1.0}, "b": {"pass_rate": 1.0}}, mean_score=0.9, pass_rate=1.0)
    current = {"a": {"passed": True, "pass_rate": 1.0, "score": 1.0, "n_samples": 1},
               "b": {"passed": False, "pass_rate": 0.0, "score": 0.0, "n_samples": 1}}
    text = render_run_report(summary, diff_cases(ref, "run-1", current))
    assert "committed baseline" in text and "**REGRESSIONS (1):** b" in text


def test_report_repeats_show_flaky_and_sample_counts():
    summary = make_summary("r", "t", "m", "ts", {"a": [True, True], "b": [True, False]})
    text = render_run_report(summary)
    assert "FLAKY" in text and "1/2" in text and "Flaky cases (1):** b" in text


def test_report_escapes_pipes_in_errors_so_tables_dont_break():
    summary = make_summary("r", "t", "m", "ts", {"a": True})
    summary.case_results.append(make_sample("b", False, error="bad | pipe\nnewline"))
    row_line = next(line for line in render_run_report(summary).splitlines() if line.startswith("| b "))
    assert "bad \\| pipe newline" in row_line


def test_json_report_is_machine_readable(tmp_path):
    summary = _summary()
    md = write_report(summary, None, out_dir=str(tmp_path))
    data = json.loads(md.with_suffix(".json").read_text(encoding="utf-8"))
    assert data["run_id"] == "run-1" and data["metrics"]["n_cases"] == 2
    assert len(data["metrics"]["pass_rate_ci95"]) == 2
    assert {c["id"] for c in data["cases"]} == {"a", "b"}
    assert summary_to_dict(summary)["diff"] is None


def test_comparison_lists_only_the_disagreements_and_warns_about_noise():
    runs = [
        {"run_id": "r1", "model": "gpt", "case_count": 4, "pass_rate": 0.75, "mean_score": 0.8,
         "total_cost_usd": 0.01, "error_count": 0, "meta": {"judge_model": "j"}},
        {"run_id": "r2", "model": "local", "case_count": 4, "pass_rate": 0.5, "mean_score": 0.6,
         "total_cost_usd": 0.0, "error_count": 0, "meta": {"judge_model": "j", "self_judged": True}},
    ]

    def res(flags):
        return {cid: {"passed": p, "score": 1.0 if p else 0.0} for cid, p in flags.items()}

    results = {"r1": res({"a": True, "b": True, "c": True, "d": False}),
               "r2": res({"a": True, "b": False, "c": False, "d": False})}
    text = render_comparison("task", runs, results)
    assert "disagree (2 of 4)" in text
    assert "| b |" in text and "| c |" in text and "| a |" not in text
    assert "self-judged" in text and "interval" in text
