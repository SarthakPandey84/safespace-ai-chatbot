"""
backend/database.py

Repository pattern — SQLite (local) vs PostgreSQL/Supabase (production).
  - DATABASE_URL set   → PostgreSQL / Supabase
  - DATABASE_URL unset → SQLite at data/safespace.db
"""

import os
import logging
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Environment ───────────────────────────────────────────────────────────────

DATABASE_URL = os.getenv("DATABASE_URL")
USE_POSTGRES = bool(DATABASE_URL)

if USE_POSTGRES:
    import psycopg2
    import psycopg2.extras
    logger.info("Database: PostgreSQL / Supabase")
else:
    import sqlite3
    DB_PATH = Path(__file__).parent.parent / "data" / "safespace.db"
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Database: SQLite @ {DB_PATH}")


# ── Connection helpers ────────────────────────────────────────────────────────

@contextmanager
def _pg_conn():
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@contextmanager
def _sqlite_conn():
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _conn():
    return _pg_conn() if USE_POSTGRES else _sqlite_conn()


def _exec(conn, sql, params=()):
    if USE_POSTGRES:
        with conn.cursor() as cur:
            cur.execute(sql, params)
    else:
        conn.execute(sql, params)


def _fetchall(conn, sql, params=()):
    if USE_POSTGRES:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]
    else:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]


def _fetchone(conn, sql, params=()):
    if USE_POSTGRES:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
            return dict(row) if row else None
    else:
        row = conn.execute(sql, params).fetchone()
        return dict(row) if row else None


# ── Schema ────────────────────────────────────────────────────────────────────

_PG_SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_logs (
    id                    SERIAL PRIMARY KEY,
    session_id            TEXT        NOT NULL,
    timestamp             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    user_message_raw      TEXT        NOT NULL,
    user_message_anon     TEXT        NOT NULL,
    ai_response           TEXT        NOT NULL,
    emotion               TEXT,
    pii_entities_found    TEXT,
    response_latency_ms   INTEGER,
    crisis_detected       BOOLEAN     NOT NULL DEFAULT FALSE
);
"""

_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_logs (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id            TEXT    NOT NULL,
    timestamp             TEXT    NOT NULL,
    user_message_raw      TEXT    NOT NULL,
    user_message_anon     TEXT    NOT NULL,
    ai_response           TEXT    NOT NULL,
    emotion               TEXT,
    pii_entities_found    TEXT,
    response_latency_ms   INTEGER,
    crisis_detected       INTEGER NOT NULL DEFAULT 0
);
"""

_CRISIS_KEYWORDS = [
    "kill myself", "want to die", "harm myself", "hurt myself",
    "end it all", "not want to live", "better off without me",
    "want to disappear", "take my own life", "ending my life",
]


# ── Public functions (imported by main.py) ────────────────────────────────────

def initialize_database() -> None:
    """Create tables if they don't exist. Called once at FastAPI startup."""
    schema = _PG_SCHEMA if USE_POSTGRES else _SQLITE_SCHEMA
    try:
        with _conn() as conn:
            _exec(conn, schema)
        logger.info("✅ Database initialised.")
    except Exception as e:
        logger.error(f"❌ Database init failed: {e}")
        raise


def log_chat_turn(
    session_id:              str,
    user_message_raw:        str,
    user_message_anonymized: str,
    ai_response:             str,
    detected_emotion:        str,
    pii_entities_found:      str | None,
    response_latency_ms:     int,
) -> int | None:
    """Insert one anonymised chat log row. Returns new row id."""
    crisis = any(kw in user_message_raw.lower() for kw in _CRISIS_KEYWORDS)
    timestamp = datetime.utcnow().isoformat()

    if USE_POSTGRES:
        sql = """
            INSERT INTO chat_logs
                (session_id, timestamp, user_message_raw, user_message_anon,
                 ai_response, emotion, pii_entities_found, response_latency_ms, crisis_detected)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """
        params = (session_id, timestamp, user_message_raw, user_message_anonymized,
                  ai_response, detected_emotion, pii_entities_found, response_latency_ms, crisis)
    else:
        sql = """
            INSERT INTO chat_logs
                (session_id, timestamp, user_message_raw, user_message_anon,
                 ai_response, emotion, pii_entities_found, response_latency_ms, crisis_detected)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (session_id, timestamp, user_message_raw, user_message_anonymized,
                  ai_response, detected_emotion, pii_entities_found, response_latency_ms, int(crisis))

    try:
        with _conn() as conn:
            if USE_POSTGRES:
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    row = cur.fetchone()
                    new_id = dict(row)["id"] if row else None
            else:
                cur = conn.execute(sql, params)
                new_id = cur.lastrowid

        logger.info(
            f"✅ Logged — id={new_id} session={session_id[:8]} "
            f"emotion={detected_emotion} crisis={crisis}"
        )
        return new_id

    except Exception as e:
        logger.error(f"❌ log_chat_turn failed: {e}")
        raise


def get_all_chat_logs(limit: int = 100) -> list[dict]:
    sql = (
        "SELECT * FROM chat_logs ORDER BY timestamp DESC LIMIT %s"
        if USE_POSTGRES else
        "SELECT * FROM chat_logs ORDER BY timestamp DESC LIMIT ?"
    )
    try:
        with _conn() as conn:
            rows = _fetchall(conn, sql, (limit,))
        for r in rows:
            if "timestamp" in r and hasattr(r["timestamp"], "isoformat"):
                r["timestamp"] = r["timestamp"].isoformat()
        return rows
    except Exception as e:
        logger.error(f"❌ get_all_chat_logs failed: {e}")
        return []


def get_all_session_metrics() -> list[dict]:
    sql = """
        SELECT
            session_id,
            COUNT(*)                                          AS turn_count,
            MIN(timestamp)                                    AS first_message,
            MAX(timestamp)                                    AS last_message,
            SUM(CASE WHEN crisis_detected THEN 1 ELSE 0 END) AS crisis_count,
            AVG(response_latency_ms)                          AS avg_latency_ms
        FROM chat_logs
        GROUP BY session_id
        ORDER BY last_message DESC
    """
    try:
        with _conn() as conn:
            rows = _fetchall(conn, sql)
        for r in rows:
            for k in ("first_message", "last_message"):
                if k in r and hasattr(r[k], "isoformat"):
                    r[k] = r[k].isoformat()
            if r.get("avg_latency_ms") is not None:
                r["avg_latency_ms"] = round(float(r["avg_latency_ms"]), 1)
        return rows
    except Exception as e:
        logger.error(f"❌ get_all_session_metrics failed: {e}")
        return []


def get_emotion_distribution() -> list[dict]:
    sql = """
        SELECT emotion, COUNT(*) AS count
        FROM chat_logs
        WHERE emotion IS NOT NULL
        GROUP BY emotion
        ORDER BY count DESC
    """
    try:
        with _conn() as conn:
            return _fetchall(conn, sql)
    except Exception as e:
        logger.error(f"❌ get_emotion_distribution failed: {e}")
        return []


def get_daily_activity() -> list[dict]:
    sql = """
        SELECT DATE(timestamp) AS date, COUNT(*) AS message_count
        FROM chat_logs
        GROUP BY DATE(timestamp)
        ORDER BY date DESC
        LIMIT 30
    """
    try:
        with _conn() as conn:
            rows = _fetchall(conn, sql)
        for r in rows:
            if "date" in r and hasattr(r["date"], "isoformat"):
                r["date"] = r["date"].isoformat()
        return rows
    except Exception as e:
        logger.error(f"❌ get_daily_activity failed: {e}")
        return []


def get_summary_stats() -> dict:
    sql = """
        SELECT
            COUNT(DISTINCT session_id)     AS total_sessions,
            COUNT(*)                       AS total_messages,
            AVG(response_latency_ms)       AS avg_latency_ms,
            SUM(CASE WHEN pii_entities_found IS NOT NULL
                      AND pii_entities_found != ''
                     THEN 1 ELSE 0 END)    AS total_pii_scrubbed
        FROM chat_logs
    """
    try:
        with _conn() as conn:
            row = _fetchone(conn, sql)
        if not row:
            return {}
        total_s = int(row.get("total_sessions") or 0)
        total_m = int(row.get("total_messages") or 0)
        return {
            "total_sessions":        total_s,
            "total_messages":        total_m,
            "avg_turns_per_session": round(total_m / total_s, 2) if total_s else 0.0,
            "avg_latency_ms":        round(float(row.get("avg_latency_ms") or 0), 1),
            "total_pii_scrubbed":    int(row.get("total_pii_scrubbed") or 0),
        }
    except Exception as e:
        logger.error(f"❌ get_summary_stats failed: {e}")
        return {}


# ── Aliases for any legacy references ────────────────────────────────────────
init_db       = initialize_database
save_chat_log = log_chat_turn
get_all_logs  = get_all_chat_logs