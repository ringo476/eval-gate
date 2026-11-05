import sqlite3

from harness.core.store import SCHEMA_VERSION, RunStore
from tests.helpers import make_summary


def test_roundtrip_with_meta_and_repeats(tmp_path):
    store = RunStore(str(tmp_path / "runs.db"))
    summary = make_summary(
        "r1", "t", "m", "2026-01-01T00:00:00Z", {"a": [True, True, False], "b": [False, False, False]},
        meta={"git_sha": "abcdef1234567", "partial": False, "judge_model": "j"},
    )
    store.save(summary)

    run = store.get_run("r1")
    assert run["meta"]["git_sha"] == "abcdef1234567"
    assert run["repeats"] == 3 and run["case_count"] == 2
    assert run["is_partial"] is False and run["is_aborted"] is False

    cases = store.case_results_for_run("r1")
    assert cases["a"]["n_samples"] == 3
    assert abs(cases["a"]["pass_rate"] - 2 / 3) < 1e-9
    assert cases["a"]["passed"] is True and cases["b"]["passed"] is False
    assert store.get_run("nope") is None
    store.close()


def test_latest_run_id_skips_partial_and_respects_before(tmp_path):
    store = RunStore(str(tmp_path / "runs.db"))
    store.save(make_summary("r1", "t", "m", "2026-01-01T00:00:00Z", {"a": True}))
    store.save(make_summary("r2", "t", "m", "2026-01-02T00:00:00Z", {"a": True}, meta={"partial": True}))
    store.save(make_summary("r3", "t", "m", "2026-01-03T00:00:00Z", {"a": True}))
    store.save(make_summary("r4", "t", "m", "2026-01-04T00:00:00Z", {"a": True}, aborted=True))

    assert store.latest_run_id("t", "m") == "r4"
    assert store.latest_run_id("t", "m", full_only=True) == "r3"          # not the aborted r4
    assert store.latest_run_id("t", "m", before_run_id="r3", full_only=True) == "r1"  # not partial r2
    assert store.latest_run_id("t", "m", before_run_id="r3") == "r2"
    # "before" means strictly older: diffing an OLD run must not pick a newer one
    assert store.latest_run_id("t", "m", before_run_id="r1") is None
    assert store.latest_run_id("t", "other") is None
    store.close()


def test_run_history_reports_kind_and_sha(tmp_path):
    store = RunStore(str(tmp_path / "runs.db"))
    store.save(make_summary("r1", "t", "m", "2026-01-01T00:00:00Z", {"a": True},
                            meta={"git_sha": "1234567890", "partial": True}))
    (row,) = store.run_history("t", "m")
    assert row["is_partial"] is True and row["git_sha"] == "1234567"
    store.close()


def test_opens_and_migrates_a_legacy_v1_database(tmp_path):
    path = str(tmp_path / "legacy.db")
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE runs (run_id TEXT PRIMARY KEY, task TEXT NOT NULL, model TEXT NOT NULL,
            timestamp TEXT NOT NULL, mean_score REAL NOT NULL, pass_rate REAL NOT NULL,
            total_cost_usd REAL NOT NULL, total_latency_ms REAL NOT NULL, error_count INTEGER NOT NULL,
            case_count INTEGER NOT NULL);
        CREATE TABLE case_results (run_id TEXT NOT NULL, case_id TEXT NOT NULL, passed INTEGER NOT NULL,
            aggregate_score REAL NOT NULL, output_text TEXT NOT NULL, cost_usd REAL NOT NULL,
            latency_ms REAL NOT NULL, error TEXT, scores_json TEXT NOT NULL, PRIMARY KEY (run_id, case_id));
        INSERT INTO runs VALUES ('old', 't', 'm', '2025-01-01T00:00:00Z', 0.5, 0.5, 0, 0, 0, 2);
        INSERT INTO case_results VALUES ('old', 'a', 1, 1.0, 'x', 0, 0, NULL, '[]');
        INSERT INTO case_results VALUES ('old', 'b', 0, 0.0, 'x', 0, 0, NULL, '[]');
    """)
    conn.commit()
    conn.close()

    store = RunStore(path)  # must not raise
    cases = store.case_results_for_run("old")
    assert cases["a"]["pass_rate"] == 1.0 and cases["b"]["pass_rate"] == 0.0
    assert cases["a"]["n_samples"] == 1
    assert store.get_run("old")["meta"] == {}
    assert store._conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION

    store.save(make_summary("new", "t", "m", "2026-01-01T00:00:00Z", {"a": True}))  # new writes work too
    store.close()
    RunStore(path).close()  # and reopening an already-migrated DB is a no-op
