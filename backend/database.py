import sqlite3
import os
import logging
from datetime import datetime
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("https://safespace-dashboard.streamlit.app")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
DB_PATH  = os.path.join(DATA_DIR, 'safespace.db')


def get_connection():
    if DATABASE_URL:
        import psycopg2
        conn = psycopg2.connect(https://safespace-dashboard.streamlit.app)
        return conn
    else:
        os.makedirs(DATA_DIR, exist_ok=True)
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

def initialize_database() -> None:
    logger.info(f"Initializing database at: {DB_PATH}")
    conn = get_connection()

    try:
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_logs (
                id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id              TEXT NOT NULL,
                user_message_raw        TEXT NOT NULL,
                user_message_anonymized TEXT NOT NULL,
                ai_response             TEXT NOT NULL,
                detected_emotion        TEXT,
                pii_entities_found      TEXT,
                response_latency_ms     INTEGER,
                timestamp               TEXT NOT NULL DEFAULT (datetime('now'))
            );
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_metrics (
                session_id          TEXT PRIMARY KEY,
                total_turns         INTEGER DEFAULT 0,
                dominant_emotion    TEXT,
                session_start       TEXT NOT NULL DEFAULT (datetime('now')),
                last_active         TEXT NOT NULL DEFAULT (datetime('now')),
                total_pii_detected  INTEGER DEFAULT 0
            );
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_chat_logs_session_id
            ON chat_logs(session_id);
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_chat_logs_timestamp
            ON chat_logs(timestamp);
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_chat_logs_emotion
            ON chat_logs(detected_emotion);
        """)

        conn.commit()
        logger.info("Database initialized successfully. All tables and indexes are ready.")

    except sqlite3.Error as e:
        logger.error(f"CRITICAL: Database initialization failed: {e}")
        raise
    finally:
        conn.close()


def log_chat_turn(
    session_id:              str,
    user_message_raw:        str,
    user_message_anonymized: str,
    ai_response:             str,
    detected_emotion:        Optional[str],
    pii_entities_found:      Optional[str],
    response_latency_ms:     Optional[int]
) -> int:
    conn      = get_connection()
    new_row_id = -1
    pii_count  = len(pii_entities_found.split(',')) if pii_entities_found else 0

    try:
        with conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO chat_logs (
                    session_id, user_message_raw, user_message_anonymized,
                    ai_response, detected_emotion, pii_entities_found,
                    response_latency_ms, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                user_message_raw,
                user_message_anonymized,
                ai_response,
                detected_emotion,
                pii_entities_found,
                response_latency_ms,
                datetime.utcnow().isoformat()
            ))

            new_row_id = cursor.lastrowid
            logger.info(f"Chat log inserted: row_id={new_row_id}, session={session_id[:8]}...")

            cursor.execute("""
                INSERT INTO session_metrics (
                    session_id, total_turns, dominant_emotion,
                    session_start, last_active, total_pii_detected
                ) VALUES (?, 1, ?, datetime('now'), datetime('now'), ?)

                ON CONFLICT(session_id) DO UPDATE SET
                    total_turns        = total_turns + 1,
                    dominant_emotion   = excluded.dominant_emotion,
                    last_active        = datetime('now'),
                    total_pii_detected = total_pii_detected + ?
            """, (
                session_id,
                detected_emotion,
                pii_count,
                pii_count
            ))

        logger.info(f"Session metrics upserted for session={session_id[:8]}...")
        return new_row_id

    except sqlite3.Error as e:
        logger.error(f"Database write failed for session {session_id[:8]}: {e}")
        raise
    finally:
        conn.close()


def get_all_chat_logs(limit: int = 1000) -> list[dict]:
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                id, session_id, user_message_anonymized,
                ai_response, detected_emotion, pii_entities_found,
                response_latency_ms, timestamp
            FROM chat_logs
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        rows = [dict(row) for row in cursor.fetchall()]
        logger.info(f"Fetched {len(rows)} chat log records for dashboard.")
        return rows
    except sqlite3.Error as e:
        logger.error(f"Failed to fetch chat logs: {e}")
        return []
    finally:
        conn.close()


def get_all_session_metrics() -> list[dict]:
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                session_id, total_turns, dominant_emotion,
                session_start, last_active, total_pii_detected
            FROM session_metrics
            ORDER BY last_active DESC
        """)
        rows = [dict(row) for row in cursor.fetchall()]
        logger.info(f"Fetched {len(rows)} session metric records for dashboard.")
        return rows
    except sqlite3.Error as e:
        logger.error(f"Failed to fetch session metrics: {e}")
        return []
    finally:
        conn.close()


def get_emotion_distribution() -> list[dict]:
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                COALESCE(detected_emotion, 'unknown') AS emotion,
                COUNT(*) as count
            FROM chat_logs
            WHERE detected_emotion IS NOT NULL
            GROUP BY detected_emotion
            ORDER BY count DESC
        """)
        rows = [dict(row) for row in cursor.fetchall()]
        return rows
    except sqlite3.Error as e:
        logger.error(f"Failed to fetch emotion distribution: {e}")
        return []
    finally:
        conn.close()


def get_daily_activity() -> list[dict]:
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                DATE(timestamp) AS date,
                COUNT(*) AS message_count
            FROM chat_logs
            GROUP BY DATE(timestamp)
            ORDER BY date ASC
        """)
        rows = [dict(row) for row in cursor.fetchall()]
        return rows
    except sqlite3.Error as e:
        logger.error(f"Failed to fetch daily activity: {e}")
        return []
    finally:
        conn.close()


def get_summary_stats() -> dict:
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                (SELECT COUNT(DISTINCT session_id) FROM chat_logs)       AS total_sessions,
                (SELECT COUNT(*) FROM chat_logs)                          AS total_messages,
                (SELECT ROUND(AVG(total_turns), 1) FROM session_metrics)  AS avg_turns_per_session,
                (SELECT ROUND(AVG(response_latency_ms), 0)
                    FROM chat_logs
                    WHERE response_latency_ms IS NOT NULL)                AS avg_latency_ms,
                (SELECT SUM(total_pii_detected) FROM session_metrics)     AS total_pii_scrubbed
        """)
        row = cursor.fetchone()
        return dict(row) if row else {}
    except sqlite3.Error as e:
        logger.error(f"Failed to fetch summary stats: {e}")
        return {}
    finally:
        conn.close()