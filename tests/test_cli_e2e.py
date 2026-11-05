"""End-to-end CLI tests: real argument parsing, real SQLite, real files, a
controllable fake task instead of a real model. These pin the behavior CI
depends on -- exit codes and the run -> bless -> gate -> regress workflow."""
import json

import pytest

from harness import cli
from harness.adapters.base import Adapter
from harness.core.models import RunTrace
from harness.scorers.contains import ContainsScorer
from harness.tasks import TASKS, TaskSpec

CASES = [
    {"id": "c1", "task": "fake_task", "input": {"needle": "one"}, "expected": {"answer_contains": ["one"]}, "tags": ["pinned"]},
    {"id": "c2", "task": "fake_task", "input": {"needle": "two"}, "expected": {"answer_contains": ["two"]}},
    {"id": "c3", "task": "fake_task", "input": {"needle": "three"}, "expected": {"answer_contains": ["three"]}},
    {"id": "c4", "task": "fake_task", "input": {"needle": "four"}, "expected": {"answer_contains": ["four"]}},
]


class ControlledAdapter(Adapter):
    task = "fake_task"
    broken: set = set()

    def run(self, case, provider):
        text = "nothing useful" if case.id in self.broken else f"the answer is {case.input['needle']}"
        return RunTrace(case_id=case.id, output_text=text, model=provider.name)


def write_dataset(path, cases):
    path.write_text("\n".join(json.dumps(c) for c in cases) + "\n", encoding="utf-8")


@pytest.fixture
def env(tmp_path, monkeypatch):
    dataset = tmp_path / "fake.jsonl"
    write_dataset(dataset, CASES)
    for key, value in {
        "HARNESS_RUNS_DB": tmp_path / "runs.db", "HARNESS_CACHE_DB": tmp_path / "cache.db",
        "HARNESS_REPORTS_DIR": tmp_path / "reports", "HARNESS_BASELINES_DIR": tmp_path / "baselines",
    }.items():
        monkeypatch.setenv(key, str(value))
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "HARNESS_JUDGE_MODEL"):
        monkeypatch.delenv(key, raising=False)
    ControlledAdapter.broken = set()
    monkeypatch.setitem(TASKS, "fake_task", TaskSpec(
        dataset_path=str(dataset), adapter_factory=ControlledAdapter,
        scorers_factory=lambda judge: [ContainsScorer()],
    ))
    return SimpleEnv(tmp_path, dataset)


class SimpleEnv:
    def __init__(self, root, dataset):
        self.root, self.dataset = root, dataset

    @property
    def baseline_file(self):
        return self.root / "baselines" / "fake_task__mock.json"


def run(*args):
    return cli.main(["--version"] if not args else list(args))


def do_run(*extra):
    return cli.main(["run", "--task", "fake_task", "--provider", "mock", *extra])


def gate(*extra):
    return cli.main(["gate", "--task", "fake_task", "--model", "mock", "--min-score", "0.5", *extra])


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        run("--version")
    assert exc.value.code == 0
    assert "harness" in capsys.readouterr().out


def test_run_writes_report_json_and_database(env, capsys):
    assert do_run() == 0
    out = capsys.readouterr().out
    assert "Pass rate:** 100.0%" in out and "95% CI" in out
    reports = list((env.root / "reports").iterdir())
    assert {p.suffix for p in reports} == {".md", ".json"}
    assert (env.root / "runs.db").exists()


def test_full_regression_workflow(env, capsys):
    # 1. a good full run, blessed as the baseline
    assert do_run() == 0
    assert cli.main(["baseline", "set", "--task", "fake_task", "--model", "mock"]) == 0
    assert env.baseline_file.exists()
    assert gate("--require-baseline") == 0

    # 2. a change breaks c2; the run itself still "succeeds" (no errors) but the gate blocks it
    ControlledAdapter.broken = {"c2"}
    assert do_run() == 0
    capsys.readouterr()
    assert gate("--require-baseline") == 1
    out = capsys.readouterr().out
    assert "FAIL" in out and "regressed" in out and "c2" in out
    assert cli.main(["diff", "--task", "fake_task", "--model", "mock"]) == 1

    # 3. fix it; green again
    ControlledAdapter.broken = set()
    assert do_run() == 0
    assert gate("--require-baseline") == 0


def test_baseline_is_not_silently_replaced_by_a_worse_run(env):
    do_run()
    cli.main(["baseline", "set", "--task", "fake_task", "--model", "mock"])
    before = env.baseline_file.read_text(encoding="utf-8")
    ControlledAdapter.broken = {"c2", "c3"}
    do_run()
    do_run()  # even after several bad runs the committed bar hasn't moved...
    assert env.baseline_file.read_text(encoding="utf-8") == before
    assert gate("--require-baseline") == 1  # ...so they keep failing


def test_require_baseline_fails_when_none_exists_but_plain_gate_only_warns(env, capsys):
    do_run()
    capsys.readouterr()
    assert gate() == 0
    assert "no baseline" in capsys.readouterr().out.lower()
    assert gate("--require-baseline") == 1


def test_smoke_runs_cannot_be_blessed_or_gated(env, capsys):
    assert do_run("--smoke", "2") == 0
    assert cli.main(["baseline", "set", "--task", "fake_task", "--model", "mock"]) == 2  # no full run exists
    assert not env.baseline_file.exists()
    capsys.readouterr()
    assert gate() == 1
    assert "--smoke" in capsys.readouterr().out


def test_stale_run_is_rejected_after_the_dataset_changes(env, capsys):
    do_run()
    extra = {"id": "c5", "task": "fake_task", "input": {"needle": "five"}, "expected": {"answer_contains": ["five"]}}
    write_dataset(env.dataset, CASES + [extra])
    capsys.readouterr()
    assert gate() == 1
    assert "stale" in capsys.readouterr().out


def test_dropping_a_case_from_the_dataset_cannot_hide_a_regression(env, capsys):
    do_run()
    cli.main(["baseline", "set", "--task", "fake_task", "--model", "mock"])
    write_dataset(env.dataset, CASES[:3])  # quietly delete c4...
    do_run()  # ...and re-run
    capsys.readouterr()
    assert gate("--require-baseline") == 1
    out = capsys.readouterr().out
    assert "missing from this run" in out and "c4" in out


def test_wrong_task_in_dataset_is_a_fatal_error(env):
    write_dataset(env.dataset, CASES[:2] + [{**CASES[2], "task": "some_other_task"}])
    assert do_run() == 3


def test_unknown_run_id_and_missing_runs_exit_2(env):
    assert gate() == 2  # nothing has been run yet
    do_run()
    assert gate("--run-id", "no-such-run") == 2
    assert cli.main(["diff", "--task", "fake_task", "--model", "nobody"]) == 2


def test_repeats_are_recorded(env, capsys):
    assert do_run("--repeats", "3", "--temperature", "0.5") == 0
    out = capsys.readouterr().out
    assert "4 x 3 samples" in out
    assert cli.main(["history", "--task", "fake_task", "--model", "mock"]) == 0
    assert "full" in capsys.readouterr().out


def test_strict_judge_refuses_self_judging_before_spending_anything(env):
    code = cli.main(["run", "--task", "fake_task", "--provider", "ollama", "--model", "m",
                     "--judge-model", "m", "--strict-judge"])
    assert code == 3


def test_missing_api_key_is_a_clean_fatal_error(env, capsys):
    assert cli.main(["run", "--task", "fake_task", "--provider", "openai", "--model", "gpt-6-luna"]) == 3
    assert "OPENAI_API_KEY" in capsys.readouterr().err


def test_compare(env, capsys):
    assert cli.main(["compare", "--task", "fake_task", "--models", "mock", "ghost"]) == 2  # ghost has no run
    do_run()
    capsys.readouterr()
    out_file = env.root / "cmp.md"
    assert cli.main(["compare", "--task", "fake_task", "--models", "mock", "--out", str(out_file)]) == 0
    assert "Model comparison" in out_file.read_text(encoding="utf-8")


def test_baseline_show(env, capsys):
    assert cli.main(["baseline", "show", "--task", "fake_task", "--model", "mock"]) == 2
    do_run()
    cli.main(["baseline", "set", "--task", "fake_task", "--model", "mock"])
    capsys.readouterr()
    assert cli.main(["baseline", "show", "--task", "fake_task", "--model", "mock"]) == 0
    assert "pass_rate=100.0%" in capsys.readouterr().out


def test_calibrate_with_the_mock_judge_reports_it_as_uncalibrated(env, capsys):
    cal = env.root / "cal.jsonl"
    rows = [
        {"id": f"cal-{i}", "task": "judge_calibration", "input": {"question": "q", "candidate_answer": "a"},
         "expected": {"reference_answer": "r"}, "human_label": label}
        for i, label in enumerate([1.0, 0.0, 0.5])
    ]
    cal.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    assert cli.main(["calibrate", "--provider", "mock", "--dataset", str(cal)]) == 1  # mock can't judge
    out = capsys.readouterr().out
    assert "INVALID" in out and "WARNING" in out
    written = list((env.root / "reports").glob("calibration-*.json"))
    assert len(written) == 1
    assert json.loads(written[0].read_text(encoding="utf-8"))["invalid"] == 3
