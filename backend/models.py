# ==============================================================================
# FILE: backend/models.py
# PROJECT: SafeSpace AI - Empathetic Privacy-First Chatbot
# PURPOSE: Defines all Pydantic data models (schemas) for API request and
#          response bodies. These act as a strict data contract between the
#          frontend and the backend.
#
# ACADEMIC NOTE — Why Pydantic Models at API Boundaries?
#   FastAPI uses these models to:
#     1. VALIDATE incoming JSON automatically — if the frontend sends a request
#        with a missing field or wrong type, FastAPI returns a clear 422 error
#        without our code ever executing. This is the "fail fast" principle.
#     2. SERIALIZE outgoing responses — Python objects are automatically
#        converted to JSON using the response_model schema.
#     3. DOCUMENT the API — Pydantic models are read by FastAPI to auto-generate
#        the Swagger UI at /docs, giving us free interactive API documentation.
#
#   Keeping all models in ONE file (models.py) means any developer can
#   understand the entire API contract by reading a single file.
# ==============================================================================

from pydantic import BaseModel, Field, field_validator
from typing import Optional


class ChatRequest(BaseModel):
    """
    Schema for the POST /chat request body.
    Defines what the frontend MUST send for a chat message.
    """
    session_id: str = Field(
        ...,                                    # '...' means this field is required
        min_length = 36,
        max_length = 36,
        description = "A valid UUID v4 session identifier (36 characters).",
        example     = "a3f1c2d4-e5b6-7890-abcd-ef1234567890"
    )
    message: str = Field(
        ...,
        min_length  = 1,
        max_length  = 2000,
        description = "The user's raw message text. PII will be scrubbed server-side.",
        example     = "I've been feeling really anxious about my exams lately."
    )

    # ACADEMIC NOTE — Custom Validators:
    # Pydantic v2 uses @field_validator for custom per-field validation logic.
    # This runs AFTER the type check, giving us a second validation layer.
    @field_validator('session_id')
    @classmethod
    def validate_uuid_format(cls, v: str) -> str:
        """Validates that session_id looks like a UUID (contains hyphens at right positions)."""
        import re
        uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
        if not re.match(uuid_pattern, v.lower()):
            raise ValueError("session_id must be a valid UUID v4 format.")
        return v


class ChatResponse(BaseModel):
    """
    Schema for the POST /chat response body.
    Defines exactly what the frontend will receive after a chat message.

    NOTE: The anonymized version of the user's message is intentionally
    EXCLUDED from this response. The user sees the AI's reply; the
    anonymized text stays server-side. This prevents confusion
    (users seeing their own words replaced with tags) while maintaining
    the privacy guarantee in the backend.
    """
    session_id:       str            = Field(..., description="Echo of the request session_id.")
    response:         str            = Field(..., description="The AI's empathetic response text.")
    detected_emotion: str            = Field(..., description="Primary emotion detected in the user's message.")
    pii_was_detected: bool           = Field(..., description="True if PII was found and scrubbed.")
    latency_ms:       int            = Field(..., description="Total AI inference time in milliseconds.")


class SessionInitResponse(BaseModel):
    """Schema for the POST /session/new response."""
    session_id: str = Field(..., description="The newly generated anonymous UUID session identifier.")
    message:    str = Field(..., description="A welcome/confirmation message for the user.")


class HealthCheckResponse(BaseModel):
    """Schema for the GET /health response."""
    status:         str = Field(..., description="Overall system status: 'healthy' or 'degraded'.")
    database:       str = Field(..., description="Database component status.")
    ai_engine:      str = Field(..., description="AI engine component status.")
    privacy_engine: str = Field(..., description="Privacy engine component status.")


class DashboardStatsResponse(BaseModel):
    """Schema for the GET /dashboard/stats response."""
    total_sessions:        int   = Field(0,   description="Total unique conversation sessions.")
    total_messages:        int   = Field(0,   description="Total chat messages logged.")
    avg_turns_per_session: float = Field(0.0, description="Average number of turns per session.")
    avg_latency_ms:        float = Field(0.0, description="Average AI response latency in ms.")
    total_pii_scrubbed:    int   = Field(0,   description="Total PII instances anonymized across all sessions.")
