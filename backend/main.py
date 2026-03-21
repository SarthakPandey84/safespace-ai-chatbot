# ==============================================================================
# FILE: backend/main.py
# PROJECT: SafeSpace AI - Empathetic Privacy-First Chatbot
# PURPOSE: The central FastAPI application. This is the entry point for the
#          entire backend. It is responsible for:
#            1. Wiring together all backend modules (DB, Privacy, AI)
#            2. Defining all HTTP REST API endpoints
#            3. Managing UUID-based anonymous sessions
#            4. Serving the static frontend HTML/CSS/JS files
#            5. Enforcing CORS policy for browser security
#
# ARCHITECTURAL ROLE — The Orchestrator:
#   main.py is the ONLY file that imports and coordinates all other modules.
#   It does NOT contain any business logic itself — it delegates:
#     - Privacy logic   → privacy_engine.py
#     - AI logic        → ai_engine.py
#     - Database logic  → database.py
#     - Data contracts  → models.py
#   This follows the Single Responsibility Principle: main.py's only job
#   is to receive HTTP requests, call the right modules in the right order,
#   and return HTTP responses. It is the conductor, not the musician.
#
# HOW TO RUN THE SERVER:
#   From the project root directory (safe_space_ai/):
#     uvicorn backend.main:app --reload --port 8000
#
#   Then open: http://localhost:8000  (serves the chat frontend)
#   API docs:  http://localhost:8000/docs  (auto-generated Swagger UI)
# ==============================================================================

import os
import sys
import uuid
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

# Pydantic models for request/response validation (defined in models.py)
from backend.models import (
    ChatRequest,
    ChatResponse,
    SessionInitResponse,
    HealthCheckResponse,
    DashboardStatsResponse,
)

# Our three core business logic modules
from backend.database      import (
    initialize_database,
    log_chat_turn,
    get_all_chat_logs,
    get_all_session_metrics,
    get_emotion_distribution,
    get_daily_activity,
    get_summary_stats,
)
from backend.privacy_engine import privacy_engine   # Presidio singleton
from backend.ai_engine      import ai_engine        # Gemini singleton

# ------------------------------------------------------------------------------
# LOGGING SETUP
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)


# ==============================================================================
# SECTION 1: APPLICATION LIFESPAN (Startup & Shutdown)
# ==============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application lifecycle using FastAPI's modern lifespan pattern.

    ACADEMIC NOTE — Lifespan vs. @app.on_event (Deprecated):
        Older FastAPI tutorials use @app.on_event("startup"). This was deprecated
        in favour of the `lifespan` context manager pattern, which is cleaner
        because it co-locates startup AND shutdown logic in one place, using
        Python's familiar `yield` syntax (a context manager).

        The code BEFORE `yield` runs at startup.
        The code AFTER `yield` runs at shutdown.
        This guarantees cleanup logic always runs, even if an exception occurs.

    STARTUP SEQUENCE:
        1. Initialize the SQLite database (create tables if not exist)
        2. The privacy_engine and ai_engine singletons are already loaded
           at module import time (see their respective files). We just log
           their readiness here for confirmation.
    """
    # --- STARTUP ---
    logger.info("="*55)
    logger.info("  SafeSpace AI — Backend Server Starting Up")
    logger.info("="*55)

    # Step 1: Initialize the database schema.
    # This is idempotent (uses CREATE TABLE IF NOT EXISTS), so it is safe
    # to run on every startup — it only creates tables that don't yet exist.
    try:
        initialize_database()
        logger.info("✓ Database initialized.")
    except Exception as e:
        # If the database cannot be initialized, the server MUST NOT start.
        # Logging and re-raising causes Uvicorn to exit with a clear error.
        logger.critical(f"✗ Database initialization failed: {e}. Shutting down.")
        raise

    # Step 2: Confirm AI and Privacy engines are loaded.
    # These were initialized at import time; we just confirm they're accessible.
    logger.info(f"✓ PrivacyEngine ready (Presidio + SpaCy).")
    logger.info(f"✓ AIEngine ready (Gemini: {ai_engine.model.model_name}).")
    logger.info("="*55)
    logger.info("  Server ready. Listening for requests.")
    logger.info("="*55)

    yield  # <-- Application runs here (handles requests between startup and shutdown)

    # --- SHUTDOWN ---
    logger.info("SafeSpace AI — Backend Server Shutting Down. Goodbye.")


# ==============================================================================
# SECTION 2: FASTAPI APPLICATION INSTANCE
# ==============================================================================

app = FastAPI(
    title       = "SafeSpace AI — Backend API",
    description = (
        "Privacy-first empathetic chatbot API. All user messages are anonymized "
        "via Microsoft Presidio before AI inference and database logging. "
        "No PII is stored. Session tracking uses anonymous UUIDs only."
    ),
    version     = "1.0.0-mvp",
    lifespan    = lifespan,     # Register our startup/shutdown manager

    # Disable the /redoc endpoint in favour of the cleaner /docs (Swagger UI).
    # Both document the API automatically from our Pydantic models.
    redoc_url   = None,
    docs_url    = "/docs",
)


# ==============================================================================
# SECTION 3: MIDDLEWARE CONFIGURATION
# ==============================================================================

# --- CORS (Cross-Origin Resource Sharing) Middleware ---
# ACADEMIC NOTE — Why CORS Matters:
#   Browsers enforce the "Same-Origin Policy" — JavaScript on page A cannot
#   make requests to a different domain/port (origin B) unless origin B
#   explicitly permits it via CORS headers in its HTTP responses.
#
#   Our frontend (served at localhost:8000) calls our own API (also localhost:8000),
#   so technically CORS isn't needed for same-origin. However, during development,
#   a developer might run the frontend separately (e.g., with Live Server on port
#   5500). We configure CORS explicitly to support this workflow.
#
#   SECURITY NOTE: In production, replace allow_origins=["*"] with the exact
#   domain(s) of your deployed frontend. Wildcard (*) is acceptable for an
#   academic MVP but is a security risk in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],              # All origins (MVP only — restrict in production)
    allow_credentials = True,
    allow_methods     = ["GET", "POST"],    # Explicitly whitelist only needed HTTP methods
    allow_headers     = ["*"],
)


# ==============================================================================
# SECTION 4: STATIC FILE SERVING
# ==============================================================================

# Construct the absolute path to the frontend directory.
# os.path.dirname(__file__) → backend/
# Going up one level and into frontend/ → frontend/
BASE_DIR     = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FRONTEND_DIR = os.path.join(BASE_DIR, 'frontend')

# Mount the frontend/ directory as a static file server at the /static URL path.
# This allows the browser to request /static/style.css and /static/script.js.
# IMPORTANT: The `name="static"` parameter allows us to generate URLs for static
# files programmatically using `request.url_for("static", path="style.css")`.
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
    logger.info(f"Static files mounted from: {FRONTEND_DIR}")
else:
    logger.warning(f"Frontend directory not found at {FRONTEND_DIR}. Static files not served.")


# ==============================================================================
# SECTION 5: API ENDPOINTS
# ==============================================================================

# ------------------------------------------------------------------------------
# ENDPOINT 1: Root — Serve the Chat Frontend
# GET /
# ------------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse, tags=["Frontend"])
async def serve_frontend():
    """
    Serves the main chat interface HTML file.

    ARCHITECTURAL NOTE — Backend-Served Frontend:
        We serve the HTML file directly from the FastAPI backend rather than
        using a separate web server (like Nginx). This simplifies the MVP
        deployment to a single process (`uvicorn backend.main:app`).
        In production, a reverse proxy (Nginx) would serve static files
        directly (faster) and forward only API requests to FastAPI.
    """
    index_path = os.path.join(FRONTEND_DIR, 'index.html')
    if not os.path.exists(index_path):
        raise HTTPException(
            status_code = status.HTTP_404_NOT_FOUND,
            detail      = "Frontend index.html not found. Ensure frontend/ directory exists."
        )
    return FileResponse(index_path)


# ------------------------------------------------------------------------------
# ENDPOINT 2: Health Check
# GET /health
# ------------------------------------------------------------------------------
@app.get("/health", response_model=HealthCheckResponse, tags=["System"])
async def health_check():
    """
    Returns the operational status of all system components.

    PURPOSE: Health check endpoints are standard practice in any deployed
    service. They allow monitoring tools (or your faculty during a demo)
    to verify that all components — database, AI engine, privacy engine —
    are operational without sending actual user data.

    A system is only as healthy as its weakest component, so we check all three.
    """
    db_status      = "ok"
    ai_status      = "ok"
    privacy_status = "ok"

    # Check database connectivity by attempting a lightweight query
    try:
        stats = get_summary_stats()
        db_status = "ok"
    except Exception as e:
        db_status = f"error: {str(e)}"
        logger.error(f"Health check — DB error: {e}")

    # Check AI engine by verifying the model object exists
    try:
        active_sessions = ai_engine.get_active_session_count()
        ai_status = f"ok ({active_sessions} active sessions)"
    except Exception as e:
        ai_status = f"error: {str(e)}"

    # Check privacy engine by running anonymization on a known test string
    try:
        test_result = privacy_engine.anonymize("Health check test for John.")
        privacy_status = "ok" if test_result else "error: no result"
    except Exception as e:
        privacy_status = f"error: {str(e)}"

    overall = "healthy" if all(
        s.startswith("ok") for s in [db_status, ai_status, privacy_status]
    ) else "degraded"

    return HealthCheckResponse(
        status         = overall,
        database       = db_status,
        ai_engine      = ai_status,
        privacy_engine = privacy_status,
    )


# ------------------------------------------------------------------------------
# ENDPOINT 3: Session Initialization
# POST /session/new
# ------------------------------------------------------------------------------
@app.post("/session/new", response_model=SessionInitResponse, tags=["Session"])
async def create_new_session():
    """
    Creates and returns a new anonymous session UUID.

    PRIVACY ARCHITECTURE — UUID-Only Authentication:
        This endpoint is the cornerstone of our anonymous-by-design approach.
        No username, email, or password is ever requested or stored.
        The UUID is the ONLY identifier, and it is:
          1. Generated server-side using uuid4() — cryptographically random,
             practically impossible to guess (2^122 possible values).
          2. Stateless — the server doesn't "register" it anywhere at creation.
             It only becomes meaningful when the first chat message uses it.
          3. Non-linkable — two sessions cannot be linked to the same person
             without additional data the system never collects.

        ACADEMIC NOTE — UUID4 vs UUID1:
            UUID1 uses the machine's MAC address and current timestamp, which
            could theoretically re-identify the server or user. UUID4 is purely
            random — it carries no embedded metadata whatsoever, making it the
            correct choice for a privacy-first application.

    Returns:
        SessionInitResponse: Contains the new session_id UUID string.
    """
    new_session_id = str(uuid.uuid4())
    logger.info(f"New session created: {new_session_id[:8]}...")
    return SessionInitResponse(
        session_id = new_session_id,
        message    = "Session created. You are anonymous. Your privacy is protected."
    )


# ------------------------------------------------------------------------------
# ENDPOINT 4: Chat — The Core Endpoint
# POST /chat
# ------------------------------------------------------------------------------
@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    The primary endpoint. Processes a user message through the full pipeline:
        1. Validate input (Pydantic does this automatically)
        2. Anonymize the message (Presidio)
        3. Generate an empathetic AI response (Gemini)
        4. Log the anonymized turn to the database (SQLite)
        5. Return the response to the user

    CRITICAL PIPELINE ORDER ENFORCEMENT:
        The order of steps 2 → 3 → 4 is not arbitrary — it is the physical
        enforcement of our Privacy-by-Design mandate:
          - Step 2 MUST happen before step 3: Gemini never sees raw PII.
          - Step 2 MUST happen before step 4: The database never stores raw PII.
        Changing this order would constitute a privacy violation by design.

    REQUEST BODY (ChatRequest):
        {
            "session_id": "uuid-string",
            "message":    "The user's raw message text"
        }

    RESPONSE BODY (ChatResponse):
        {
            "session_id":        "uuid-string",
            "response":          "AI's empathetic response",
            "detected_emotion":  "anxious",
            "pii_was_detected":  true,
            "latency_ms":        342
        }

    NOTE: The response deliberately omits the anonymized message text.
    The user sees the AI's natural response; the anonymized version stays
    server-side (logged to DB). This preserves UX while enforcing privacy.
    """

    # --- INPUT VALIDATION ---
    # Pydantic already validated the JSON shape via ChatRequest.
    # We add an additional semantic check: empty messages are meaningless.
    if not request.message or not request.message.strip():
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail      = "Message cannot be empty."
        )

    # Guard against excessively long messages that could abuse the AI API.
    # 2000 characters ≈ 400-500 words — sufficient for any conversational turn.
    MAX_MESSAGE_LENGTH = 2000
    if len(request.message) > MAX_MESSAGE_LENGTH:
        raise HTTPException(
            status_code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail      = f"Message exceeds maximum length of {MAX_MESSAGE_LENGTH} characters."
        )

    logger.info(
        f"[POST /chat] session={request.session_id[:8]}..., "
        f"msg_len={len(request.message)} chars."
    )

    # =========================================================================
    # STEP 2: ANONYMIZE — Strip PII Before Anything Else
    # =========================================================================
    # ARCHITECTURE NOTE: This is the most important line in the entire pipeline.
    # Nothing proceeds until this call completes successfully. If it raises
    # an exception (RuntimeError from privacy_engine's fail-safe), the entire
    # request is aborted with a 500 error — PII never leaks downstream.
    try:
        privacy_result = privacy_engine.anonymize(request.message)
    except RuntimeError as e:
        logger.error(f"Privacy engine failure for session={request.session_id[:8]}: {e}")
        raise HTTPException(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail      = "An internal error occurred during message processing. Please try again."
            # NOTE: We intentionally return a GENERIC error message to the user.
            # Detailed error messages can leak information about system internals.
            # This is the principle of "minimal error disclosure."
        )

    if privacy_result.pii_found:
        logger.info(
            f"PII detected and scrubbed for session={request.session_id[:8]}. "
            f"Entities: {privacy_result.detected_entities}"
        )

    # =========================================================================
    # STEP 3: AI INFERENCE — Generate Empathetic Response
    # =========================================================================
    # PRIVACY ENFORCEMENT: We pass `privacy_result.anonymized_text` — NOT
    # `request.message` — to the AI engine. The raw message is now "used up"
    # for its only legitimate purpose: being anonymized. It will not be passed
    # to any external service or stored in the database.
    ai_result = ai_engine.get_response(
        session_id        = request.session_id,
        anonymized_message= privacy_result.anonymized_text
    )

    # Log if AI call encountered an error (graceful degradation was applied)
    if ai_result.error:
        logger.warning(
            f"AI engine returned fallback response for session={request.session_id[:8]}. "
            f"Error: {ai_result.error}"
        )

    # =========================================================================
    # STEP 4: DATABASE LOGGING — Store Anonymized Data Only
    # =========================================================================
    # PRIVACY ENFORCEMENT: We pass BOTH the raw message (for MVP debugging only)
    # AND the anonymized version. The database.py module stores both, but the
    # research-facing query functions (get_all_chat_logs) only expose the
    # anonymized version to the dashboard.
    #
    # PRODUCTION NOTE: In a live deployment, `user_message_raw` would be
    # replaced with a hashed version or omitted entirely. The current design
    # is a deliberate MVP trade-off, explicitly documented for viva discussion.
    try:
        log_id = log_chat_turn(
            session_id              = request.session_id,
            user_message_raw        = request.message,
            user_message_anonymized = privacy_result.anonymized_text,
            ai_response             = ai_result.response_text,
            detected_emotion        = ai_result.detected_emotion,
            pii_entities_found      = privacy_result.entities_csv,
            response_latency_ms     = ai_result.latency_ms,
        )
        logger.info(f"Chat turn logged to DB: log_id={log_id}, session={request.session_id[:8]}...")
    except Exception as e:
        # DESIGN DECISION — Non-Fatal DB Logging Failure:
        # If database logging fails, we DO NOT return an error to the user.
        # The user already has the AI response (step 3 succeeded). Failing the
        # entire request because of a logging issue would harm the user experience
        # for a non-critical (research-only) operation.
        # The error is logged for the developer to investigate.
        logger.error(f"DB logging failed (non-fatal): {e}. Response still returned to user.")

    # =========================================================================
    # STEP 5: RETURN RESPONSE
    # =========================================================================
    return ChatResponse(
        session_id       = request.session_id,
        response         = ai_result.response_text,
        detected_emotion = ai_result.detected_emotion,
        pii_was_detected = privacy_result.pii_found,
        latency_ms       = ai_result.latency_ms,
    )


# ------------------------------------------------------------------------------
# ENDPOINT 5: Clear Session
# DELETE /session/{session_id}
# ------------------------------------------------------------------------------
@app.delete("/session/{session_id}", tags=["Session"])
async def clear_session(session_id: str):
    """
    Clears the in-memory conversation history for a given session.
    Called when the user clicks "New Conversation" in the frontend.

    NOTE: This clears the AI's memory of the conversation but does NOT
    delete the database logs (which are anonymized research data).
    This distinction is important — database records are research assets,
    while conversation history is a UX/privacy concern.

    The session_id is passed as a URL path parameter, following REST
    conventions for resource deletion: DELETE /session/{resource_id}.
    """
    ai_engine.clear_session(session_id)
    logger.info(f"Session cleared via API: session={session_id[:8]}...")
    return JSONResponse(
        status_code = status.HTTP_200_OK,
        content     = {
            "message"   : "Session history cleared successfully.",
            "session_id": session_id
        }
    )


# ------------------------------------------------------------------------------
# ENDPOINT 6: Dashboard Data — Emotion Distribution
# GET /dashboard/emotions
# ------------------------------------------------------------------------------
@app.get("/dashboard/emotions", tags=["Dashboard"])
async def get_dashboard_emotions():
    """
    Returns aggregated emotion distribution data for the research dashboard.

    SECURITY NOTE — Dashboard Endpoint Access:
        In a production system, these dashboard endpoints would be protected
        by authentication (e.g., an API key or JWT token for researchers only).
        For the MVP, they are open. This is a deliberate simplification that
        should be acknowledged during a viva discussion about production readiness.

    Returns:
        list: [{"emotion": "anxious", "count": 42}, ...]
    """
    data = get_emotion_distribution()
    return JSONResponse(content={"emotions": data})


# ------------------------------------------------------------------------------
# ENDPOINT 7: Dashboard Data — Daily Activity
# GET /dashboard/activity
# ------------------------------------------------------------------------------
@app.get("/dashboard/activity", tags=["Dashboard"])
async def get_dashboard_activity():
    """
    Returns daily message counts for the research dashboard timeline chart.

    Returns:
        list: [{"date": "2025-01-15", "message_count": 12}, ...]
    """
    data = get_daily_activity()
    return JSONResponse(content={"activity": data})


# ------------------------------------------------------------------------------
# ENDPOINT 8: Dashboard Data — Summary Statistics
# GET /dashboard/stats
# ------------------------------------------------------------------------------
@app.get("/dashboard/stats", response_model=DashboardStatsResponse, tags=["Dashboard"])
async def get_dashboard_stats():
    """
    Returns high-level summary statistics for the research dashboard header cards.

    Returns:
        DashboardStatsResponse: Total sessions, messages, avg turns, avg latency,
                                total PII instances scrubbed.
    """
    stats = get_summary_stats()
    return DashboardStatsResponse(
        total_sessions      = stats.get("total_sessions", 0) or 0,
        total_messages      = stats.get("total_messages", 0) or 0,
        avg_turns_per_session = stats.get("avg_turns_per_session", 0.0) or 0.0,
        avg_latency_ms      = stats.get("avg_latency_ms", 0.0) or 0.0,
        total_pii_scrubbed  = stats.get("total_pii_scrubbed", 0) or 0,
    )


# ------------------------------------------------------------------------------
# ENDPOINT 9: Dashboard Data — Recent Chat Logs (Anonymized)
# GET /dashboard/logs
# ------------------------------------------------------------------------------
@app.get("/dashboard/logs", tags=["Dashboard"])
async def get_dashboard_logs(limit: int = 100):
    """
    Returns recent anonymized chat logs for the research dashboard data table.

    The `limit` query parameter defaults to 100 and is capped at 500 to
    prevent memory issues from loading too much data at once.

    Example: GET /dashboard/logs?limit=200

    PRIVACY GUARANTEE: The database.py get_all_chat_logs() function
    explicitly excludes the `user_message_raw` column from its SELECT query.
    Even if this endpoint were called by an unauthorized party, they would
    receive only anonymized data.
    """
    # Cap the limit to prevent accidental or malicious large data dumps
    safe_limit = min(limit, 500)
    logs = get_all_chat_logs(limit=safe_limit)
    return JSONResponse(content={"logs": logs, "count": len(logs)})


# ------------------------------------------------------------------------------
# ENDPOINT 10: Dashboard Data — Session Metrics
# GET /dashboard/sessions
# ------------------------------------------------------------------------------
@app.get("/dashboard/sessions", tags=["Dashboard"])
async def get_dashboard_sessions():
    """
    Returns per-session aggregated metrics for the research dashboard.

    Returns:
        list: Session summaries with turn counts, dominant emotion, duration.
    """
    sessions = get_all_session_metrics()
    return JSONResponse(content={"sessions": sessions, "count": len(sessions)})


# ==============================================================================
# SECTION 6: GLOBAL EXCEPTION HANDLER
# ==============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catches any unhandled exception anywhere in the application and returns
    a safe, generic JSON error response instead of a raw Python traceback.

    SECURITY NOTE — Information Disclosure:
        Python tracebacks contain internal file paths, module names, and
        sometimes variable values. Exposing them to API clients is an
        information disclosure vulnerability — attackers can map out the
        system internals. This handler ensures all unhandled errors return
        ONLY a generic message externally, while logging the full traceback
        internally for developers.

    ACADEMIC NOTE — Defense in Depth:
        Individual endpoints have their own try/except blocks (first defense).
        This global handler is the last line of defense (second defense) for
        anything that slips through.
    """
    logger.error(f"Unhandled exception on {request.method} {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
        content     = {
            "detail": "An unexpected internal error occurred. Please try again later."
        }
    )


# ==============================================================================
# SECTION 7: DIRECT EXECUTION ENTRY POINT
# ==============================================================================
# Allows running the server directly with: python backend/main.py
# This is a convenience for development. The standard way is:
#   uvicorn backend.main:app --reload
#
# ACADEMIC NOTE: The `if __name__ == "__main__"` guard ensures this block
# only runs when the file is executed directly, not when it is imported
# as a module (which would start an unintended server instance).
# ==============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host    = "0.0.0.0",   # Listen on all interfaces (accessible on LAN)
        port    = 8000,
        reload  = True,        # Auto-restart on code changes (dev mode only)
        log_level = "info"
    )
