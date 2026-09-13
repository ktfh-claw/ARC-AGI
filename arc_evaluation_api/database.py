"""SQLite persistence and atomic evaluation attempt accounting."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .dataset import Grid

MAX_ATTEMPTS = 3
AUDITED_METHODS = frozenset({"DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"})


@dataclass(frozen=True)
class AttemptResult:
    correct: bool
    attempts_used: int
    attempts_remaining: int
    status: str


class TerminalTaskError(RuntimeError):
    def __init__(self, result: AttemptResult):
        super().__init__(result.status)
        self.result = result


class SubmissionDatabase:
    def __init__(self, path: Path):
        self.path = path

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS submissions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL CHECK (attempt_count BETWEEN 1 AND 3),
                    proposed_output_json TEXT NOT NULL,
                    reasoning TEXT NOT NULL CHECK (length(trim(reasoning)) > 0),
                    correct INTEGER NOT NULL CHECK (correct IN (0, 1)),
                    submitted_at TEXT NOT NULL,
                    UNIQUE (session_id, task_id, attempt_count)
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS submissions_lookup "
                "ON submissions(session_id, task_id, attempt_count)"
            )
            # Request auditing intentionally lives outside the submissions table.  In
            # particular, rejected requests must never acquire columns that could be
            # populated from an untrusted body, header, URL, or validation error.
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS accepted_requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    method TEXT NOT NULL,
                    route TEXT NOT NULL,
                    status_code INTEGER NOT NULL CHECK (status_code BETWEEN 100 AND 399),
                    requested_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS rejected_requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    method TEXT NOT NULL,
                    status_code INTEGER NOT NULL CHECK (status_code BETWEEN 400 AND 599),
                    requested_at TEXT NOT NULL
                )
                """
            )

    def ready(self) -> bool:
        try:
            with self._connect() as connection:
                connection.execute("SELECT 1").fetchone()
            return True
        except sqlite3.Error:
            return False

    def status(self, session_id: str, task_id: str) -> AttemptResult:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS attempts_used, COALESCE(MAX(correct), 0) AS solved
                FROM submissions WHERE session_id = ? AND task_id = ?
                """,
                (session_id, task_id),
            ).fetchone()
        return self._result(bool(row["solved"]), int(row["attempts_used"]))

    def submit(
        self,
        *,
        session_id: str,
        task_id: str,
        outputs: list[Grid],
        reasoning: str,
        correct: bool,
    ) -> AttemptResult:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT COUNT(*) AS attempts_used, COALESCE(MAX(correct), 0) AS solved
                FROM submissions WHERE session_id = ? AND task_id = ?
                """,
                (session_id, task_id),
            ).fetchone()
            current = self._result(bool(row["solved"]), int(row["attempts_used"]))
            if current.status != "active":
                raise TerminalTaskError(current)
            attempt_count = current.attempts_used + 1
            connection.execute(
                """
                INSERT INTO submissions (
                    session_id, task_id, attempt_count, proposed_output_json,
                    reasoning, correct, submitted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    task_id,
                    attempt_count,
                    json.dumps(outputs, separators=(",", ":")),
                    reasoning,
                    int(correct),
                    datetime.now(UTC).isoformat(),
                ),
            )
            return self._result(correct, attempt_count)

    def audit_request(self, *, method: str, route: str | None, status_code: int) -> None:
        """Persist allowlisted request metadata; no caller content crosses this boundary."""
        requested_at = datetime.now(UTC).isoformat()
        safe_method = method if method in AUDITED_METHODS else "OTHER"
        with self._connect() as connection:
            if status_code < 400:
                # Successful routing gives us a server-defined template.  Fall back to a
                # fixed label for unusual ASGI responses rather than persisting a raw URL.
                safe_route = route if route is not None else "unmatched"
                connection.execute(
                    """
                    INSERT INTO accepted_requests (method, route, status_code, requested_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (safe_method, safe_route, status_code, requested_at),
                )
            else:
                connection.execute(
                    """
                    INSERT INTO rejected_requests (method, status_code, requested_at)
                    VALUES (?, ?, ?)
                    """,
                    (safe_method, status_code, requested_at),
                )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    @staticmethod
    def _result(solved: bool, attempts_used: int) -> AttemptResult:
        status = "solved" if solved else "exhausted" if attempts_used >= MAX_ATTEMPTS else "active"
        return AttemptResult(
            correct=solved,
            attempts_used=attempts_used,
            attempts_remaining=max(0, MAX_ATTEMPTS - attempts_used),
            status=status,
        )
