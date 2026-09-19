import json

import pytest

from harness.core.dataset import filter_smoke, load_jsonl


def _write(tmp_path, rows):
    path = tmp_path / "cases.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


def test_load_valid_jsonl(tmp_path):
    rows = [
        {"id": "a1", "task": "t", "input": {"q": "x"}, "expected": {"y": 1}},
        {"id": "a2", "task": "t", "input": {"q": "x"}, "expected": {"y": 1}, "tags": ["pinned"]},
    ]
    path = _write(tmp_path, rows)
    cases = load_jsonl(path)
    assert len(cases) == 2
    assert cases[1].is_pinned is True
    assert cases[0].is_pinned is False


def test_missing_required_field_raises(tmp_path):
    rows = [{"id": "a1", "task": "t", "input": {}}]  # missing "expected"
    path = _write(tmp_path, rows)
    with pytest.raises(ValueError, match="missing required field"):
        load_jsonl(path)


def test_duplicate_id_raises(tmp_path):
    rows = [
        {"id": "dup", "task": "t", "input": {}, "expected": {}},
        {"id": "dup", "task": "t", "input": {}, "expected": {}},
    ]
    path = _write(tmp_path, rows)
    with pytest.raises(ValueError, match="duplicate case id"):
        load_jsonl(path)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_jsonl(tmp_path / "nope.jsonl")


def test_empty_dataset_raises(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_jsonl(path)


def test_filter_smoke_always_includes_pinned(tmp_path):
    rows = [{"id": f"c{i}", "task": "t", "input": {}, "expected": {}} for i in range(20)]
    rows[15]["tags"] = ["pinned"]
    path = _write(tmp_path, rows)
    cases = load_jsonl(path)

    smoke = filter_smoke(cases, n=5)
    ids = {c.id for c in smoke}
    assert "c15" in ids  # pinned case survives even though it's outside the first 5
    assert len(smoke) == 5
