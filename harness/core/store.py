"""
SQLite-backed run history. Every `harness run` writes a row per case here,
which is what makes `harness diff` (regression detection) and a
quality-over-time chart possible. Plain SQLite is intentional -- this is
a startup-scale eval harness, not a service with concurrent writers, so a
managed DB would be pure overhead.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from harness.core.models import RunSummary


class RunStore:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

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

    def save(self, summary: RunSummary) -> None:
        with self._conn:
            self._conn.execute(
                """INSERT INTO runs
                   (run_id, task, model, timestamp, mean_score, pass_rate,
                    total_cost_usd, total_latency_ms, error_count, case_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (summary.run_id, summary.task, summary.model, summary.timestamp,
                 summary.mean_score, summary.pass_rate, summary.total_cost_usd,
                 summary.total_latency_ms, summary.error_count, len(summary.case_results)),
            )
            for cr in summary.case_results:
                self._conn.execute(
                    """INSERT INTO case_results
                       (run_id, case_id, passed, aggregate_score, output_text,
                        cost_usd, latency_ms, error, scores_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (summary.run_id, cr.case.id, int(cr.passed), cr.aggregate_score,
                     cr.trace.output_text, cr.trace.cost_usd, cr.trace.latency_ms,
                     cr.trace.error,
                     json.dumps([{"scorer": s.scorer_name, "score": s.score,
                                  "passed": s.passed, "details": s.details}
                                 for s in cr.scores])),
                )

    def latest_run_id(self, task: str, model: str, before_run_id: str | None = None) -> str | None:
        """Most recent completed run for (task, model), excluding a given run_id
        (used to find 'the run before this one' for diffing)."""
        query = "SELECT run_id FROM runs WHERE task = ? AND model = ?"
        params: list = [task, model]
        if before_run_id is not None:
            query += " AND run_id != ?"
            params.append(before_run_id)
        query += " ORDER BY timestamp DESC LIMIT 1"
        row = self._conn.execute(query, params).fetchone()
        return row[0] if row else None

    def case_results_for_run(self, run_id: str) -> dict[str, dict]:
        rows = self._conn.execute(
            "SELECT case_id, passed, aggregate_score FROM case_results WHERE run_id = ?",
            (run_id,),
        ).fetchall()
        return {r[0]: {"passed": bool(r[1]), "score": r[2]} for r in rows}

    def run_history(self, task: str, model: str, limit: int = 20) -> list[dict]:
        rows = self._conn.execute(
            """SELECT run_id, timestamp, mean_score, pass_rate, total_cost_usd, error_count
               FROM runs WHERE task = ? AND model = ? ORDER BY timestamp DESC LIMIT ?""",
            (task, model, limit),
        ).fetchall()
        return [
            {"run_id": r[0], "timestamp": r[1], "mean_score": r[2],
             "pass_rate": r[3], "total_cost_usd": r[4], "error_count": r[5]}
            for r in rows
        ]

    def close(self) -> None:
        self._conn.close()


def new_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}-{uuid.uuid4().hex[:8]}"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
