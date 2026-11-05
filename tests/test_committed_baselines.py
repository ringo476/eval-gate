"""Guards the committed baselines/ directory. A baseline that has drifted
from its dataset would make the nightly gate fail for a reason nobody can
see until the (paid) run is over; catching it here is free and immediate."""
from pathlib import Path

import pytest

from harness.core.baseline import load_baseline
from harness.core.dataset import load_jsonl
from harness.tasks import TASKS

BASELINE_FILES = sorted(Path("baselines").glob("*.json"))


@pytest.mark.skipif(not BASELINE_FILES, reason="no baselines committed yet")
@pytest.mark.parametrize("path", BASELINE_FILES, ids=lambda p: p.name)
def test_committed_baseline_is_valid_and_matches_its_dataset(path):
    baseline = load_baseline(path)
    assert baseline.task in TASKS, f"{path.name} is for unknown task '{baseline.task}'"

    dataset_ids = {c.id for c in load_jsonl(TASKS[baseline.task].dataset_path, expected_task=baseline.task)}
    stale = sorted(set(baseline.cases) - dataset_ids)
    assert not stale, (
        f"{path.name} references case(s) no longer in the dataset: {', '.join(stale)}. "
        "Re-run and `harness baseline set`, or remove those entries from the baseline by hand."
    )
    for cid, entry in baseline.cases.items():
        assert 0.0 <= entry["pass_rate"] <= 1.0, f"{path.name}: {cid} has an impossible pass_rate"
