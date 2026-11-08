"""
SQLite-backed run history. Every `harness run` writes a row per case here,
which is what makes `harness diff` (regression detection) and a
quality-over-time chart possible. Plain SQLite is intentional -- this is
a startup-scale eval harness, not a service with concurrent writers, so a
managed DB would be pure overhead.

IMPORTANT for CI: this file is local state. A fresh CI runner starts with
no runs.db, so history-based "diff vs the previous run" cannot work there.
Regression gating in CI therefore compares against a *committed baseline
file* (harness/core/baseline.py), not against this database.

Schema versioning: `PRAGMA user_version` records the schema; opening an older
database adds the missing columns in place (_migrate) instead of failing.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from harness.core.models import RunSummary

SCHEMA_VERSION = 2

_RUN_COLUMNS_V2 = {
    "meta_json": "TEXT NOT NULL DEFAULT '{}'",
    "is_partial": "INTEGER NOT NULL DEFAULT 0",
    "is_aborted": "INTEGER NOT NULL DEFAULT 0",
    "repeats": "INTEGER NOT NULL DEFAULT 1",
}
_CASE_COLUMNS_V2 = {
    "pass_rate": "REAL",
    "n_samples": "INTEGER NOT NULL DEFAULT 1",
    "samples_json": "TEXT NOT NULL DEFAULT '[]'",
}


class RunStore:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()
        self._migrate()

    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                task TEXT NOT NULL,
                model TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                mean_score REAL NOT NULL,
                pass_rate REAL NOT NULL,
                total_cost_usd REAL NOT NULL,
                total_latency_ms REAL NOT NULL,
                error_count INTEGER NOT NULL,
                case_count INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS case_results (
                run_id TEXT NOT NULL REFERENCES runs(run_id),
                case_id TEXT NOT NULL,
                passed INTEGER NOT NULL,
                aggregate_score REAL NOT NULL,
                output_text TEXT NOT NULL,
                cost_usd REAL NOT NULL,
                latency_ms REAL NOT NULL,
                error TEXT,
                scores_json TEXT NOT NULL,
                PRIMARY KEY (run_id, case_id)
            );

            CREATE INDEX IF NOT EXISTS idx_runs_task_model_ts
                ON runs (task, model, timestamp DESC);
        """)
        self._conn.commit()

    def _columns(self, table: str) -> set[str]:
        return {row[1] for row in self._conn.execute(f"PRAGMA table_info({table})")}

    def _migrate(self) -> None:
        version = self._conn.execute("PRAGMA user_version").fetchone()[0]
        if version >= SCHEMA_VERSION:
            return
        with self._conn:
            have = self._columns("runs")
            for col, decl in _RUN_COLUMNS_V2.items():
                if col not in have:
                    self._conn.execute(f"ALTER TABLE runs ADD COLUMN {col} {decl}")
            have = self._columns("case_results")
            for col, decl in _CASE_COLUMNS_V2.items():
                if col not in have:
                    self._conn.execute(f"ALTER TABLE case_results ADD COLUMN {col} {decl}")
            # rows written before repeats existed were single-sample: pass_rate == passed
            self._conn.execute("UPDATE case_results SET pass_rate = CAST(passed AS REAL) WHERE pass_rate IS NULL")
            self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def save(self, summary: RunSummary) -> None:
        cases = summary.cases
        with self._conn:
            self._conn.execute(
                """INSERT INTO runs
                   (run_id, task, model, timestamp, mean_score, pass_rate,
                    total_cost_usd, total_latency_ms, error_count, case_count,
                    meta_json, is_partial, is_aborted, repeats)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (summary.run_id, summary.task, summary.model, summary.timestamp,
                 summary.mean_score, summary.pass_rate, summary.total_cost_usd,
                 summary.total_latency_ms, summary.error_count, len(cases),
                 json.dumps(summary.meta, default=str, sort_keys=True),
                 int(summary.is_partial), int(summary.aborted), summary.repeats),
            )
            for agg in cases:
                rep = agg.representative
                errors = [s.trace.error for s in agg.samples if s.trace.error]
                self._conn.execute(
                    """INSERT INTO case_results
                       (run_id, case_id, passed, aggregate_score, output_text,
                        cost_usd, latency_ms, error, scores_json,
                        pass_rate, n_samples, samples_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (summary.run_id, agg.case.id, int(agg.passed), agg.mean_score,
                     rep.trace.output_text, agg.total_cost_usd, agg.mean_latency_ms,
                     errors[0] if errors else None,
                     json.dumps([{"scorer": s.scorer_name, "score": s.score,
                                  "passed": s.passed, "details": s.details}
                                 for s in rep.scores], default=str),
                     agg.pass_rate, agg.n_samples,
                     json.dumps([{"sample": s.sample_idx, "passed": s.passed,
                                  "score": s.aggregate_score, "error": s.trace.error,
                                  "output": s.trace.output_text[:500]}
                                 for s in agg.samples], default=str)),
                )

    def latest_run_id(
        self, task: str, model: str, before_run_id: str | None = None, full_only: bool = False,
    ) -> str | None:
        """Most recent run for (task, model).

        before_run_id: only consider runs strictly *older* than that run
            (used to find "the run before this one" for diffing).
        full_only: skip smoke/partial and aborted runs -- the only runs
            that are legitimate baselines, since a subset can't vouch for
            the cases it didn't run.
        """
        query = "SELECT run_id FROM runs WHERE task = ? AND model = ?"
        params: list = [task, model]
        if before_run_id is not None:
            ref = self._conn.execute("SELECT timestamp FROM runs WHERE run_id = ?", (before_run_id,)).fetchone()
            query += " AND run_id != ?"
            params.append(before_run_id)
            if ref:
                query += " AND timestamp < ?"
                params.append(ref[0])
        if full_only:
            query += " AND is_partial = 0 AND is_aborted = 0"
        query += " ORDER BY timestamp DESC LIMIT 1"
        row = self._conn.execute(query, params).fetchone()
        return row[0] if row else None

    def get_run(self, run_id: str) -> dict | None:
        row = self._conn.execute(
            """SELECT run_id, task, model, timestamp, mean_score, pass_rate, total_cost_usd,
                      total_latency_ms, error_count, case_count, meta_json, is_partial, is_aborted, repeats
               FROM runs WHERE run_id = ?""",
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        keys = ("run_id", "task", "model", "timestamp", "mean_score", "pass_rate", "total_cost_usd",
                "total_latency_ms", "error_count", "case_count", "meta_json", "is_partial", "is_aborted", "repeats")
        data = dict(zip(keys, row, strict=True))
        data["meta"] = json.loads(data.pop("meta_json") or "{}")
        data["is_partial"] = bool(data["is_partial"])
        data["is_aborted"] = bool(data["is_aborted"])
        return data

    def case_results_for_run(self, run_id: str) -> dict[str, dict]:
        rows = self._conn.execute(
            "SELECT case_id, passed, aggregate_score, pass_rate, n_samples FROM case_results WHERE run_id = ?",
            (run_id,),
        ).fetchall()
        return {
            r[0]: {
                "passed": bool(r[1]),
                "score": r[2],
                "pass_rate": float(r[1]) if r[3] is None else r[3],
                "n_samples": r[4],
            }
            for r in rows
        }

    def task_models(self) -> list[tuple[str, str]]:
        """Every (task, model) pair with at least one stored run."""
        return [(r[0], r[1]) for r in self._conn.execute(
            "SELECT DISTINCT task, model FROM runs ORDER BY task, model"
        )]

    def case_details(self, run_id: str) -> dict[str, dict]:
        """Per-case outcome plus which checks failed and why, for display."""
        rows = self._conn.execute(
            "SELECT case_id, passed, aggregate_score, pass_rate, n_samples, error, scores_json "
            "FROM case_results WHERE run_id = ? ORDER BY case_id",
            (run_id,),
        ).fetchall()
        details = {}
        for case_id, passed, score, pass_rate, n_samples, error, scores_json in rows:
            scores = json.loads(scores_json or "[]")
            details[case_id] = {
                "passed": bool(passed), "score": score, "n_samples": n_samples,
                "pass_rate": float(passed) if pass_rate is None else pass_rate, "error": error,
                "failed": [
                    {"scorer": s["scorer"], "reason": (s.get("details") or {}).get("reason", "")}
                    for s in scores if not s.get("passed")
                ],
            }
        return details

    def run_history(self, task: str, model: str, limit: int = 20) -> list[dict]:
        rows = self._conn.execute(
            """SELECT run_id, timestamp, mean_score, pass_rate, total_cost_usd, error_count,
                      repeats, is_partial, is_aborted, meta_json
               FROM runs WHERE task = ? AND model = ? ORDER BY timestamp DESC LIMIT ?""",
            (task, model, limit),
        ).fetchall()
        history = []
        for r in rows:
            meta = json.loads(r[9] or "{}")
            history.append({
                "run_id": r[0], "timestamp": r[1], "mean_score": r[2], "pass_rate": r[3],
                "total_cost_usd": r[4], "error_count": r[5], "repeats": r[6],
                "is_partial": bool(r[7]), "is_aborted": bool(r[8]),
                "git_sha": (meta.get("git_sha") or "")[:7],
                "dataset_sha256": meta.get("dataset_sha256"),
            })
        return history

    def close(self) -> None:
        self._conn.close()


def new_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}-{uuid.uuid4().hex[:8]}"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
