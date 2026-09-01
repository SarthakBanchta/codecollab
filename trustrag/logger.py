"""
TrustRAG — Observability Logger
Logs every query to SQLite with full pipeline telemetry:
timestamps, retrieval scores, generation output, validation verdicts, and latency breakdown.
"""

import json
import sqlite3
import threading
from datetime import datetime, timezone

import config


# Thread-local storage for SQLite connections (SQLite objects can't be shared across threads)
_local = threading.local()


def _get_connection() -> sqlite3.Connection:
    """Get a thread-local SQLite connection."""
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(config.SQLITE_DB_PATH)
        _local.conn.row_factory = sqlite3.Row  # enable dict-like row access
        _local.conn.execute("PRAGMA journal_mode=WAL")  # better concurrent read perf
    return _local.conn


def init_db() -> None:
    """Create the query_logs table if it doesn't exist."""
    conn = _get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS query_logs (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp           TEXT    NOT NULL,
            query_text          TEXT    NOT NULL,
            retrieved_chunks    TEXT,
            generated_answer    TEXT,
            groundedness_score  INTEGER,
            claims_verdict      TEXT,
            fabricated_citations TEXT,
            retrieval_time_ms   REAL,
            generation_time_ms  REAL,
            validation_time_ms  REAL,
            total_latency_ms    REAL,
            decision            TEXT,
            is_low_confidence   BOOLEAN
        )
    """)
    conn.commit()


def log_query(
    query_text: str,
    retrieved_chunks: list[dict],
    generated_answer: str,
    groundedness_score: int,
    claims_verdict: list[dict],
    fabricated_citations: list[str],
    retrieval_time_ms: float,
    generation_time_ms: float,
    validation_time_ms: float,
    decision: str,
    is_low_confidence: bool,
) -> None:
    """
    Insert a complete query log row.
    All list/dict fields are JSON-serialized for storage.
    """
    conn = _get_connection()
    total_latency = retrieval_time_ms + generation_time_ms + validation_time_ms

    conn.execute(
        """
        INSERT INTO query_logs (
            timestamp, query_text, retrieved_chunks, generated_answer,
            groundedness_score, claims_verdict, fabricated_citations,
            retrieval_time_ms, generation_time_ms, validation_time_ms,
            total_latency_ms, decision, is_low_confidence
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now(timezone.utc).isoformat(),
            query_text,
            json.dumps(retrieved_chunks),
            generated_answer,
            groundedness_score,
            json.dumps(claims_verdict),
            json.dumps(fabricated_citations),
            retrieval_time_ms,
            generation_time_ms,
            validation_time_ms,
            total_latency,
            decision,
            is_low_confidence,
        ),
    )
    conn.commit()


def get_recent_queries(limit: int = 50) -> list[dict]:
    """Fetch the most recent query logs, newest first."""
    conn = _get_connection()
    cursor = conn.execute(
        "SELECT * FROM query_logs ORDER BY id DESC LIMIT ?", (limit,)
    )
    rows = cursor.fetchall()
    results = []
    for row in rows:
        record = dict(row)
        # Deserialize JSON fields
        for json_field in ("retrieved_chunks", "claims_verdict", "fabricated_citations"):
            if record.get(json_field):
                try:
                    record[json_field] = json.loads(record[json_field])
                except json.JSONDecodeError:
                    record[json_field] = []
        results.append(record)
    return results


def get_stats() -> dict:
    """
    Compute aggregate statistics for the observability dashboard.
    Returns dict with: total_queries, avg_groundedness, avg_latency_ms,
    refused_count, warned_count, shown_count.
    """
    conn = _get_connection()
    cursor = conn.execute("""
        SELECT
            COUNT(*)                                    AS total_queries,
            COALESCE(AVG(groundedness_score), 0)        AS avg_groundedness,
            COALESCE(AVG(total_latency_ms), 0)          AS avg_latency_ms,
            COALESCE(AVG(retrieval_time_ms), 0)         AS avg_retrieval_ms,
            COALESCE(AVG(generation_time_ms), 0)        AS avg_generation_ms,
            COALESCE(AVG(validation_time_ms), 0)        AS avg_validation_ms,
            SUM(CASE WHEN decision = 'refuse' THEN 1 ELSE 0 END)           AS refused_count,
            SUM(CASE WHEN decision = 'show_with_warning' THEN 1 ELSE 0 END) AS warned_count,
            SUM(CASE WHEN decision = 'show' THEN 1 ELSE 0 END)             AS shown_count
        FROM query_logs
    """)
    row = cursor.fetchone()
    return dict(row) if row else {
        "total_queries": 0, "avg_groundedness": 0, "avg_latency_ms": 0,
        "avg_retrieval_ms": 0, "avg_generation_ms": 0, "avg_validation_ms": 0,
        "refused_count": 0, "warned_count": 0, "shown_count": 0,
    }


def get_flagged_queries(limit: int = 50) -> list[dict]:
    """Fetch queries that were refused or shown with warnings."""
    conn = _get_connection()
    cursor = conn.execute(
        """
        SELECT id, timestamp, query_text, groundedness_score, decision,
               is_low_confidence, total_latency_ms
        FROM query_logs
        WHERE decision != 'show'
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    )
    return [dict(row) for row in cursor.fetchall()]
