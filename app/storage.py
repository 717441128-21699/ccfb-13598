"""SQLite-backed job store for upload / analysis / result lifecycle."""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from .config import DB_PATH

_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    with _LOCK, _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                job_id      TEXT PRIMARY KEY,
                filename    TEXT NOT NULL,
                status      TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL,
                error       TEXT,
                result      TEXT
            )
            """
        )


def create_job(job_id: str, filename: str) -> None:
    init_db()
    ts = _now()
    with _LOCK, _connect() as conn:
        conn.execute(
            "INSERT INTO jobs(job_id, filename, status, created_at, updated_at) "
            "VALUES (?, ?, 'queued', ?, ?)",
            (job_id, filename, ts, ts),
        )


def set_status(job_id: str, status: str, error: str | None = None) -> None:
    with _LOCK, _connect() as conn:
        conn.execute(
            "UPDATE jobs SET status=?, error=?, updated_at=? WHERE job_id=?",
            (status, error, _now(), job_id),
        )


def save_result(job_id: str, result: dict) -> None:
    with _LOCK, _connect() as conn:
        conn.execute(
            "UPDATE jobs SET status='done', result=?, updated_at=? WHERE job_id=?",
            (json.dumps(result, ensure_ascii=False), _now(), job_id),
        )


def get_job(job_id: str) -> dict | None:
    with _LOCK, _connect() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
    if row is None:
        return None
    out = dict(row)
    out["result"] = json.loads(out["result"]) if out["result"] else None
    return out


def list_jobs(limit: int = 50) -> list[dict]:
    with _LOCK, _connect() as conn:
        rows = conn.execute(
            "SELECT job_id, filename, status, created_at, updated_at, error "
            "FROM jobs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]
