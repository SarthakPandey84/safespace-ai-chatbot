import os
import uuid
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from backend.models import (
    ChatRequest,
    ChatResponse,
    SessionInitResponse,
    HealthCheckResponse,
    DashboardStatsResponse,
)
from backend.database import (
    initialize_database,
    log_chat_turn,
    get_all_chat_logs,
    get_all_session_metrics,
    get_emotion_distribution,
    get_daily_activity,
    get_summary_stats,
)
from backend.privacy_engine import privacy_engine
from backend.ai_engine      import ai_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)


CRISIS_KEYWORDS = [
    "better off without me",
    "better off if i",
    "want to disappear",
    "don't want to be here",
    "do not want to be here",
    "no point in continuing",
    "end it all",
    "ending things",
    "ending my life",
    "not want to live",
    "want to die",
    "kill myself",
    "harm myself",
    "hurt myself",
    "disappear forever",
    "wish i was gone",
    "wish i weren't here",
    "wish i was dead",
    "take my own life",
]

CRISIS_RESPONSE = (
    "What you are sharing with me sounds incredibly painful, and I am truly "
    "glad you felt you could say this here. You matter deeply.\n\n"
    "Please reach out to iCall right now — they are free and confidential: "
    "📞 9152987821\n\n"
    "Vandrevala Foundation is available 24 hours, 7 days a week: "
    "📞 1860-2662-345\n\n"
    "Is there someone physically near you right now that you could be with? "
    "You do not have to face this alone."
)


def is_crisis_message(text: str) -> bool:
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in CRISIS_KEYWORDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("="*55)
    logger.info("  SafeSpace AI — Backend Server Starting Up")
    logger.info("="*55)

    try:
        initialize_database()
        logger.info("✓ Database initialized.")
    except Exception as e:
        logger.critical(
            f"✗ Database initialization failed: {e}\n"
            "  Server will start but /chat will fail until DB is reachable.\n"
            "  Fix: use the Supabase pooler (IPv4) URL in DATABASE_URL env var."
        )
        # Do NOT re-raise — server stays up so the issue is diagnosable via /health

    logger.info("✓ PrivacyEngine ready (Presidio + SpaCy).")
    logger.info(f"✓ AIEngine ready (Groq: {ai_engine.model_name}).")
    logger.info("="*55)
    logger.info("  Server ready. Listening for requests.")
    logger.info("="*55)

    yield

    logger.info("SafeSpace AI — Backend Server Shutting Down. Goodbye.")


app = FastAPI(
    title       = "SafeSpace AI — Backend API",
    description = "Privacy-first empathetic chatbot API. All messages anonymized via Presidio.",
    version     = "1.0.0-mvp",
    lifespan    = lifespan,
    redoc_url   = None,
    docs_url    = "/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["GET", "POST"],
    allow_headers     = ["*"],
)

BASE_DIR     = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FRONTEND_DIR = os.path.join(BASE_DIR, 'frontend')

if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
    logger.info(f"Static files mounted from: {FRONTEND_DIR}")
else:
    logger.warning(f"Frontend directory not found at {FRONTEND_DIR}.")


@app.get("/", response_class=HTMLResponse, tags=["Frontend"])
async def serve_frontend():
    index_path = os.path.join(FRONTEND_DIR, 'index.html')
    if not os.path.exists(index_path):
        raise HTTPException(
            status_code = status.HTTP_404_NOT_FOUND,
            detail      = "Frontend index.html not found."
        )
    return FileResponse(index_path)


@app.get("/health", response_model=HealthCheckResponse, tags=["System"])
async def health_check():
    db_status      = "ok"
    ai_status      = "ok"
    privacy_status = "ok"

    try:
        get_summary_stats()
    except Exception as e:
        db_status = f"error: {str(e)}"
        logger.error(f"Health check — DB error: {e}")

    try:
        active_sessions = ai_engine.get_active_session_count()
        ai_status = f"ok ({active_sessions} active sessions)"
    except Exception as e:
        ai_status = f"error: {str(e)}"

    try:
        test_result    = privacy_engine.anonymize("Health check test for John.")
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


@app.post("/session/new", response_model=SessionInitResponse, tags=["Session"])
async def create_new_session():
    new_session_id = str(uuid.uuid4())
    logger.info(f"New session created: {new_session_id[:8]}...")
    return SessionInitResponse(
        session_id = new_session_id,
        message    = "Session created. You are anonymous. Your privacy is protected."
    )


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    if not request.message or not request.message.strip():
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail      = "Message cannot be empty."
        )

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

    try:
        privacy_result = privacy_engine.anonymize(request.message)
    except RuntimeError as e:
        logger.error(f"Privacy engine failure for session={request.session_id[:8]}: {e}")
        raise HTTPException(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail      = "An internal error occurred during message processing. Please try again."
        )

    if privacy_result.pii_found:
        logger.info(
            f"PII detected and scrubbed for session={request.session_id[:8]}. "
            f"Entities: {privacy_result.detected_entities}"
        )

    ai_result = ai_engine.get_response(
        session_id         = request.session_id,
        anonymized_message = privacy_result.anonymized_text
    )

    if ai_result.error:
        logger.warning(
            f"AI engine returned fallback response for session={request.session_id[:8]}. "
            f"Error: {ai_result.error}"
        )

    if is_crisis_message(request.message) and (
        "9152987821" not in ai_result.response_text and
        "1860"       not in ai_result.response_text
    ):
        logger.warning(
            f"CRISIS OVERRIDE triggered for session={request.session_id[:8]}. "
            f"LLM missed crisis keywords — injecting hardcoded crisis response."
        )
        ai_result.response_text  = CRISIS_RESPONSE
        ai_result.detected_emotion = "sad"

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
        logger.error(f"DB logging failed (non-fatal): {e}. Response still returned to user.")

    return ChatResponse(
        session_id       = request.session_id,
        response         = ai_result.response_text,
        detected_emotion = ai_result.detected_emotion,
        pii_was_detected = privacy_result.pii_found,
        latency_ms       = ai_result.latency_ms,
    )


@app.delete("/session/{session_id}", tags=["Session"])
async def clear_session(session_id: str):
    ai_engine.clear_session(session_id)
    logger.info(f"Session cleared via API: session={session_id[:8]}...")
    return JSONResponse(
        status_code = status.HTTP_200_OK,
        content     = {
            "message"   : "Session history cleared successfully.",
            "session_id": session_id
        }
    )


@app.get("/dashboard/emotions", tags=["Dashboard"])
async def get_dashboard_emotions():
    data = get_emotion_distribution()
    return JSONResponse(content={"emotions": data})


@app.get("/dashboard/activity", tags=["Dashboard"])
async def get_dashboard_activity():
    data = get_daily_activity()
    return JSONResponse(content={"activity": data})


@app.get("/dashboard/stats", response_model=DashboardStatsResponse, tags=["Dashboard"])
async def get_dashboard_stats():
    stats = get_summary_stats()
    return DashboardStatsResponse(
        total_sessions        = stats.get("total_sessions", 0) or 0,
        total_messages        = stats.get("total_messages", 0) or 0,
        avg_turns_per_session = stats.get("avg_turns_per_session", 0.0) or 0.0,
        avg_latency_ms        = stats.get("avg_latency_ms", 0.0) or 0.0,
        total_pii_scrubbed    = stats.get("total_pii_scrubbed", 0) or 0,
    )


@app.get("/dashboard/logs", tags=["Dashboard"])
async def get_dashboard_logs(limit: int = 100):
    safe_limit = min(limit, 500)
    logs = get_all_chat_logs(limit=safe_limit)
    return JSONResponse(content={"logs": logs, "count": len(logs)})


@app.get("/dashboard/sessions", tags=["Dashboard"])
async def get_dashboard_sessions():
    sessions = get_all_session_metrics()
    return JSONResponse(content={"sessions": sessions, "count": len(sessions)})


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.method} {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
        content     = {"detail": "An unexpected internal error occurred. Please try again later."}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host      = "0.0.0.0",
        port      = 8000,
        reload    = True,
        log_level = "info"
    )