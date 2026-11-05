import json

import pytest

from harness.core.baseline import (
    BaselineError,
    baseline_from_store,
    baseline_path,
    find_baseline,
    load_baseline,
    slugify,
    write_baseline,
)
from harness.core.store import RunStore
from tests.helpers import make_sample, make_summary


def test_slugify_makes_model_names_filesystem_safe():
    assert slugify("llama3.1:8b") == "llama3.1_8b"
    assert slugify("gpt-6-luna") == "gpt-6-luna"
    assert slugify("a/b\\c d") == "a_b_c_d"
    assert slugify("???") == "unnamed"
    assert baseline_path("baselines", "rag", "llama3.1:8b").name == "rag__llama3.1_8b.json"


def _store_with_run(tmp_path, **kwargs):
    store = RunStore(str(tmp_path / "runs.db"))
    meta = {"git_sha": "abc", "dataset_sha256": "d" * 64, "judge_model": "j", "provider": "p",
            "python": "3.12", "noise": "dropped"}
    meta.update(kwargs.pop("meta", {}))
    store.save(make_summary("r1", "t", "m", "2026-01-01T00:00:00Z", {"a": True, "b": [True, False]}, meta=meta, **kwargs))
    return store


def test_baseline_roundtrip_is_deterministic(tmp_path):
    store = _store_with_run(tmp_path)
    baseline = baseline_from_store(store, "r1")
    store.close()

    path = write_baseline(baseline, tmp_path / "b.json")
    text1 = path.read_text(encoding="utf-8")
    write_baseline(baseline, path)
    assert path.read_text(encoding="utf-8") == text1           # stable => clean git diffs
    assert text1.endswith("\n")

    loaded = load_baseline(path)
    assert loaded.task == "t" and loaded.model == "m" and loaded.source_run_id == "r1"
    assert loaded.cases["a"]["pass_rate"] == 1.0
    assert loaded.cases["b"]["pass_rate"] == 0.5 and loaded.cases["b"]["n_samples"] == 2
    assert loaded.meta["git_sha"] == "abc"
    assert "noise" not in loaded.meta and "python" not in loaded.meta   # only reviewable provenance is kept


def test_cannot_bless_a_partial_run(tmp_path):
    store = _store_with_run(tmp_path, meta={"partial": True})
    with pytest.raises(BaselineError, match="partial"):
        baseline_from_store(store, "r1")
    store.close()


def test_cannot_bless_an_aborted_run(tmp_path):
    store = _store_with_run(tmp_path, aborted=True)
    with pytest.raises(BaselineError, match="aborted"):
        baseline_from_store(store, "r1")
    store.close()


def test_errored_run_needs_force(tmp_path):
    store = RunStore(str(tmp_path / "runs.db"))
    summary = make_summary("r1", "t", "m", "2026-01-01T00:00:00Z", {"a": True})
    summary.case_results.append(make_sample("b", False, error="boom"))
    store.save(summary)
    with pytest.raises(BaselineError, match="errored"):
        baseline_from_store(store, "r1")
    assert baseline_from_store(store, "r1", allow_errors=True).cases["b"]["pass_rate"] == 0.0
    store.close()


def test_unknown_run_is_an_error(tmp_path):
    store = RunStore(str(tmp_path / "runs.db"))
    with pytest.raises(BaselineError, match="not found"):
        baseline_from_store(store, "nope")
    store.close()


def test_load_rejects_bad_files(tmp_path):
    with pytest.raises(BaselineError, match="not found"):
        load_baseline(tmp_path / "missing.json")
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{nope", encoding="utf-8")
    with pytest.raises(BaselineError, match="invalid JSON"):
        load_baseline(bad_json)
    wrong_schema = tmp_path / "schema.json"
    wrong_schema.write_text(json.dumps({"schema": 99}), encoding="utf-8")
    with pytest.raises(BaselineError, match="schema"):
        load_baseline(wrong_schema)
    missing_field = tmp_path / "field.json"
    missing_field.write_text(json.dumps({"schema": 1, "task": "t"}), encoding="utf-8")
    with pytest.raises(BaselineError, match="missing required"):
        load_baseline(missing_field)


def test_find_baseline(tmp_path):
    store = _store_with_run(tmp_path)
    write_baseline(baseline_from_store(store, "r1"), baseline_path(tmp_path / "bl", "t", "m"))
    store.close()
    assert find_baseline(tmp_path / "bl", "t", "m").source_run_id == "r1"
    assert find_baseline(tmp_path / "bl", "t", "other-model") is None
