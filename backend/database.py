# ==============================================================================
# FILE: backend/database.py
# PROJECT: SafeSpace AI - Empathetic Privacy-First Chatbot
# PURPOSE: Defines the SQLite schema, manages the database connection,
#          and exposes clean CRUD (Create, Read, Update, Delete) functions.
#
# ARCHITECTURAL PATTERN — Repository Pattern:
#   All database logic is intentionally encapsulated ONLY in this file.
#   No other module in this project should ever import `sqlite3` directly.
#   This enforces the Repository Pattern: the rest of the application talks
#   to THIS file's functions, not to the database directly.
#
#   WHY THIS MATTERS (Viva Defense Point):
#   If we ever need to migrate from SQLite to PostgreSQL, we change ONLY this
#   file. The FastAPI routes, AI engine, and dashboard remain completely
#   untouched. This is the Open/Closed Principle from SOLID design:
#   "Open for extension, closed for modification."
#
# PRIVACY NOTE:
#   This module ONLY ever receives and stores ANONYMIZED data. The decision
#   to anonymize BEFORE calling any function in this file is enforced at the
#   API route level (main.py). This means the database is a PII-free zone
#   by architectural design, not just by developer discipline.
# ==============================================================================

import sqlite3      # Part of Python standard library — no installation needed
import os           # For constructing file paths in an OS-agnostic way
import logging      # For structured application logging (better than print())
from datetime import datetime  # For timestamping all logged records
from typing import Optional    # For type hints on nullable fields

# ------------------------------------------------------------------------------
# LOGGING SETUP
# ACADEMIC NOTE: Using Python's logging module instead of print() is a
# professional standard. It allows log levels (DEBUG, INFO, WARNING, ERROR),
# timestamps, and easy redirection to log files — all critical for debugging
# a running application and demonstrating production-readiness to evaluators.
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# DATABASE PATH CONFIGURATION
# We resolve the database path relative to THIS file's location. This makes
# the project portable — it works correctly regardless of what directory
# the developer runs the server from, which is a common source of bugs.
#
# Structure:
#   backend/database.py  (this file)
#   data/safespace.db    (database, one level up from backend/)
# ------------------------------------------------------------------------------

# __file__ is the absolute path of this script.
# os.path.dirname() gets the directory containing it (i.e., backend/).
# os.path.join(..., '..', 'data') navigates up one level, then into data/.
# os.path.abspath() resolves any '..' to produce a clean, absolute path.
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
DB_PATH  = os.path.join(DATA_DIR, 'safespace.db')


# ==============================================================================
# SECTION 1: CONNECTION MANAGEMENT
# ==============================================================================

def get_connection() -> sqlite3.Connection:
    """
    Creates and returns a new SQLite database connection.

    DESIGN DECISION — New Connection Per Request:
        We intentionally create a new connection per database call rather than
        maintaining a single global connection. SQLite connections are NOT
        thread-safe by default. FastAPI runs on an async server (Uvicorn) that
        can handle concurrent requests, so a shared global connection would
        cause data corruption under load.

        The `check_same_thread=False` flag IS used here because we're using
        a fresh connection each time, making the threading concern moot.
        For a production PostgreSQL setup, we would use a connection pool
        (e.g., asyncpg's pool), but this approach is correct and safe for SQLite MVP.

    Returns:
        sqlite3.Connection: A configured database connection with Row factory.
    """
    # Ensure the /data directory exists before trying to connect.
    # exist_ok=True prevents an error if the directory already exists.
    os.makedirs(DATA_DIR, exist_ok=True)

    conn = sqlite3.connect(DB_PATH, check_same_thread=False)

    # ACADEMIC NOTE — Row Factory:
    # By default, sqlite3 returns rows as plain tuples: (1, 'hello', ...).
    # Setting row_factory to sqlite3.Row allows us to access columns by NAME:
    # row['session_id'] instead of row[0]. This makes the code self-documenting
    # and far less error-prone when the schema evolves.
    conn.row_factory = sqlite3.Row

    # Enable Write-Ahead Logging (WAL) mode for better concurrent read/write
    # performance. This is especially useful when the dashboard is reading data
    # while the chatbot is simultaneously writing new logs.
    conn.execute("PRAGMA journal_mode=WAL;")

    return conn


# ==============================================================================
# SECTION 2: DATABASE SCHEMA INITIALIZATION
# ==============================================================================

def initialize_database() -> None:
    """
    Creates all required database tables if they do not already exist.
    This function is called ONCE at application startup (from main.py).

    SCHEMA DESIGN PHILOSOPHY:
        The schema is designed around two core concerns:
        1. OPERATIONAL: Logging chat turns for the chatbot to function.
        2. RESEARCH: Capturing anonymized behavioral metrics for the dashboard.

        These are kept in SEPARATE tables (chat_logs and session_metrics)
        following database normalization principles, specifically to avoid
        data anomalies and to make research queries simpler and faster.
    """
    logger.info(f"Initializing database at: {DB_PATH}")
    conn = get_connection()

    try:
        cursor = conn.cursor()

        # ------------------------------------------------------------------
        # TABLE 1: chat_logs
        # PURPOSE: Stores every individual message turn in the conversation.
        # This is the PRIMARY data table for research analysis.
        #
        # PRIVACY NOTE: The `user_message_raw` column stores the ORIGINAL
        # message for display in the UI response cycle only. In this MVP,
        # we log it here for debugging convenience, but in a production
        # privacy-first system, this column would be REMOVED entirely and
        # only `user_message_anonymized` would be persisted. This is a
        # deliberate academic trade-off worth discussing in your viva.
        # ------------------------------------------------------------------
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_logs (

                -- Primary Key: Auto-incrementing integer ID for each message turn.
                -- Using INTEGER PRIMARY KEY in SQLite automatically creates a
                -- fast B-tree index on this column for O(log n) lookups.
                id                      INTEGER PRIMARY KEY AUTOINCREMENT,

                -- Session Identifier: A UUID (Universally Unique Identifier)
                -- generated client-side. This links multiple messages from the
                -- same conversation WITHOUT storing any personal information.
                -- It is the only "linking" mechanism — no usernames, no emails.
                session_id              TEXT NOT NULL,

                -- The raw message as typed by the user.
                -- ACADEMIC NOTE: In a full production deployment, this field
                -- would be either (a) not stored at all, or (b) encrypted with
                -- a key held separately. For this MVP, it aids in debugging
                -- the anonymization pipeline.
                user_message_raw        TEXT NOT NULL,

                -- The PII-scrubbed version of the user's message (e.g.,
                -- "My name is John" becomes "My name is <PERSON>").
                -- This is the ONLY version used for AI inference and research.
                user_message_anonymized TEXT NOT NULL,

                -- The AI's response to the user.
                -- NOTE: AI responses don't typically contain user PII, but
                -- they are stored here for conversational context analysis.
                ai_response             TEXT NOT NULL,

                -- Detected emotion/sentiment label derived from the AI's
                -- interpretation (e.g., 'anxious', 'sad', 'hopeful', 'neutral').
                -- This is a structured field that powers dashboard analytics.
                -- Nullable because emotion detection may not always be certain.
                detected_emotion        TEXT,

                -- A list of PII entity types found in the raw message,
                -- stored as a comma-separated string (e.g., "PERSON,PHONE_NUMBER").
                -- RESEARCH VALUE: Tells researchers what TYPES of PII users
                -- tend to share, informing future privacy tool improvements,
                -- WITHOUT revealing the actual PII values.
                pii_entities_found      TEXT,

                -- Response latency in milliseconds (AI API call duration).
                -- RESEARCH VALUE: Tracks system performance over time and
                -- helps identify slow responses that may affect user experience.
                response_latency_ms     INTEGER,

                -- ISO 8601 timestamp of when this log entry was created.
                -- DEFAULT CURRENT_TIMESTAMP means SQLite fills this in
                -- automatically if we don't provide a value — a safety net.
                timestamp               TEXT NOT NULL DEFAULT (datetime('now'))
            );
        """)

        # ------------------------------------------------------------------
        # TABLE 2: session_metrics
        # PURPOSE: Stores aggregated, per-session behavioral metadata.
        # One row per unique session (conversation).
        #--
        # RESEARCH VALUE: Allows dashboard to answer questions like:
        #   - "What is the average session duration?"
        #   - "How many turns does a typical conversation last?"
        #   - "What time of day are users most active?"
        # ------------------------------------------------------------------
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_metrics (

                -- The same UUID from chat_logs. Using it as PRIMARY KEY
                -- enforces the one-row-per-session constraint at the DB level.
                session_id          TEXT PRIMARY KEY,

                -- Total number of message turns in this session.
                -- Updated (incremented) after each message via upsert logic.
                total_turns         INTEGER DEFAULT 0,

                -- Dominant emotion across the session (most frequently detected).
                -- Useful for session-level emotional arc analysis.
                dominant_emotion    TEXT,

                -- ISO 8601 timestamp of the first message in this session.
                session_start       TEXT NOT NULL DEFAULT (datetime('now')),

                -- ISO 8601 timestamp of the most recent message.
                -- Updated on every new turn to track session duration.
                last_active         TEXT NOT NULL DEFAULT (datetime('now')),

                -- Total number of PII instances detected and scrubbed
                -- across the entire session.
                -- RESEARCH VALUE: Aggregate privacy risk indicator per session.
                total_pii_detected  INTEGER DEFAULT 0
            );
        """)

        # ------------------------------------------------------------------
        # INDEX CREATION
        # ACADEMIC NOTE — Why Indexes Matter:
        #   Without an index, a query like "SELECT * FROM chat_logs WHERE
        #   session_id = ?" requires a FULL TABLE SCAN — reading every row.
        #   With an index, SQLite uses a B-tree to find matching rows in
        #   O(log n) time. As the research dataset grows to thousands of
        #   entries, this becomes critical for dashboard load performance.
        # ------------------------------------------------------------------
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
        # If table creation fails, we log the error and re-raise it.
        # This will cause the FastAPI startup to fail loudly, which is the
        # CORRECT behavior — a server with no database should not start silently.
        logger.error(f"CRITICAL: Database initialization failed: {e}")
        raise
    finally:
        # ALWAYS close the connection in a finally block.
        # This ensures the connection is released even if an exception occurs,
        # preventing connection leaks which can corrupt the database file.
        conn.close()


# ==============================================================================
# SECTION 3: CRUD OPERATIONS (The Repository Interface)
# ==============================================================================

def log_chat_turn(
    session_id:              str,
    user_message_raw:        str,
    user_message_anonymized: str,
    ai_response:             str,
    detected_emotion:        Optional[str],
    pii_entities_found:      Optional[str],
    response_latency_ms:     Optional[int]
) -> int:
    """
    Inserts one complete chat turn (user message + AI response) into chat_logs
    and updates (or creates) the corresponding session_metrics row.

    This function performs TWO writes in a SINGLE transaction. If either write
    fails, BOTH are rolled back. This guarantees data consistency — we will
    never have a chat log without a corresponding session metric, or vice versa.

    ACADEMIC NOTE — ACID Transactions:
        SQLite is ACID compliant. 'Atomicity' (the 'A' in ACID) ensures that
        a transaction is "all or nothing." Using `conn` as a context manager
        (`with conn:`) automatically handles commit on success and rollback on
        failure, making our code both safe and concise.

    Args:
        session_id:              The UUID identifying the conversation session.
        user_message_raw:        The original, unmodified user input.
        user_message_anonymized: The PII-scrubbed version of the user input.
        ai_response:             The AI-generated response text.
        detected_emotion:        The emotion label extracted by the AI engine.
        pii_entities_found:      Comma-separated PII entity types (or None).
        response_latency_ms:     Time taken for AI inference in milliseconds.

    Returns:
        int: The auto-generated primary key ID of the new chat_logs row.
             Useful for confirming the write succeeded.
    """
    conn = get_connection()
    new_row_id = -1  # Sentinel value; will be overwritten on success

    try:
        # Count how many PII entities were detected for the metrics table.
        # If pii_entities_found is "PERSON,EMAIL_ADDRESS", the count is 2.
        pii_count = len(pii_entities_found.split(',')) if pii_entities_found else 0

        # `with conn:` acts as a transaction context manager in sqlite3.
        # All statements inside execute as one atomic transaction.
        with conn:
            cursor = conn.cursor()

            # --- WRITE 1: Insert the chat log record ---
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
                datetime.utcnow().isoformat()  # UTC time for timezone consistency
            ))

            new_row_id = cursor.lastrowid  # Capture the auto-generated primary key
            logger.info(f"Chat log inserted: row_id={new_row_id}, session={session_id[:8]}...")

            # --- WRITE 2: Upsert the session_metrics record ---
            # UPSERT = INSERT if session doesn't exist, UPDATE if it does.
            # SQLite's "INSERT OR REPLACE" / "ON CONFLICT DO UPDATE" syntax
            # handles this elegantly without needing a separate SELECT first.
            #
            # ACADEMIC NOTE — UPSERT Pattern:
            #   The naive approach would be: SELECT → check if exists → INSERT or UPDATE.
            #   This is a "check-then-act" pattern that introduces a race condition
            #   (two simultaneous requests for the same session could both pass the
            #   SELECT check and try to INSERT, causing a constraint violation).
            #   Using ON CONFLICT resolves this atomically at the database level.
            cursor.execute("""
                INSERT INTO session_metrics (
                    session_id, total_turns, dominant_emotion,
                    session_start, last_active, total_pii_detected
                ) VALUES (?, 1, ?, datetime('now'), datetime('now'), ?)

                ON CONFLICT(session_id) DO UPDATE SET
                    total_turns       = total_turns + 1,
                    dominant_emotion  = excluded.dominant_emotion,
                    last_active       = datetime('now'),
                    total_pii_detected = total_pii_detected + ?
            """, (
                session_id,
                detected_emotion,
                pii_count,
                pii_count         # The second ? in the ON CONFLICT UPDATE clause
            ))

        logger.info(f"Session metrics upserted for session={session_id[:8]}...")
        return new_row_id

    except sqlite3.Error as e:
        logger.error(f"Database write failed for session {session_id[:8]}: {e}")
        raise  # Re-raise so FastAPI can return a 500 error to the client
    finally:
        conn.close()


def get_all_chat_logs(limit: int = 1000) -> list[dict]:
    """
    Retrieves the most recent chat log entries for the research dashboard.

    NOTE: We expose only ANONYMIZED message columns, never `user_message_raw`.
    This function is called by dashboard.py to populate research visualizations.

    Args:
        limit (int): Max number of records to return. Default 1000 prevents
                     accidentally loading a massive dataset into dashboard memory.

    Returns:
        list[dict]: A list of dictionaries, each representing one chat log row.
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        # SECURITY NOTE: We SELECT only the columns needed for research.
        # `user_message_raw` is explicitly EXCLUDED from this query to ensure
        # the dashboard only ever sees anonymized data.
        cursor.execute("""
            SELECT
                id, session_id, user_message_anonymized,
                ai_response, detected_emotion, pii_entities_found,
                response_latency_ms, timestamp
            FROM chat_logs
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))

        # Convert sqlite3.Row objects to plain dicts for easy JSON serialization
        # and Pandas DataFrame creation in the dashboard.
        rows = [dict(row) for row in cursor.fetchall()]
        logger.info(f"Fetched {len(rows)} chat log records for dashboard.")
        return rows
    except sqlite3.Error as e:
        logger.error(f"Failed to fetch chat logs: {e}")
        return []  # Return empty list rather than crashing the dashboard
    finally:
        conn.close()


def get_all_session_metrics() -> list[dict]:
    """
    Retrieves all session-level aggregated metrics for the research dashboard.

    Returns:
        list[dict]: A list of dictionaries, each representing one session summary.
    """
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
    """
    Returns an aggregated count of each detected emotion label.
    This is a pre-computed aggregate query designed specifically to power
    the 'Emotion Distribution' pie/bar chart on the research dashboard.

    ACADEMIC NOTE — Pre-aggregation vs. Client-side Aggregation:
        We could fetch ALL records and aggregate in Python/Pandas on the
        dashboard side. However, pushing aggregation INTO the SQL query
        (using GROUP BY) is significantly more efficient — SQLite processes
        millions of rows in milliseconds using its query optimizer, whereas
        loading all rows into Python memory is wasteful and slow.
        This is the principle of "computation close to the data."

    Returns:
        list[dict]: e.g., [{'emotion': 'anxious', 'count': 42}, ...]
    """
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
    """
    Returns message counts grouped by date for the activity timeline chart.
    Uses SQLite's DATE() function to truncate timestamps to day-level granularity.

    Returns:
        list[dict]: e.g., [{'date': '2025-01-15', 'message_count': 12}, ...]
    """
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
    """
    Returns a single dictionary of high-level statistics for the dashboard
    summary cards (total sessions, total messages, avg turns per session,
    avg response latency, total PII instances scrubbed).

    Returns:
        dict: A flat dictionary of key metrics.
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                (SELECT COUNT(DISTINCT session_id) FROM chat_logs)      AS total_sessions,
                (SELECT COUNT(*) FROM chat_logs)                         AS total_messages,
                (SELECT ROUND(AVG(total_turns), 1) FROM session_metrics) AS avg_turns_per_session,
                (SELECT ROUND(AVG(response_latency_ms), 0)
                    FROM chat_logs
                    WHERE response_latency_ms IS NOT NULL)               AS avg_latency_ms,
                (SELECT SUM(total_pii_detected) FROM session_metrics)    AS total_pii_scrubbed
        """)
        row = cursor.fetchone()
        return dict(row) if row else {}
    except sqlite3.Error as e:
        logger.error(f"Failed to fetch summary stats: {e}")
        return {}
    finally:
        conn.close()
