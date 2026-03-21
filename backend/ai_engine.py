# ==============================================================================
# FILE: backend/ai_engine.py
# PROJECT: SafeSpace AI - Empathetic Privacy-First Chatbot
# PURPOSE: Manages all interaction with the Google Gemini API. Responsible for:
#            1. Constructing the empathetic system prompt (the AI's "personality")
#            2. Maintaining per-session conversation history (multi-turn memory)
#            3. Sending anonymized messages to Gemini and receiving responses
#            4. Extracting structured metadata (emotion label) from responses
#
# ARCHITECTURAL POSITION IN THE PIPELINE:
#   This module sits at Position 3, receiving ONLY anonymized text from the
#   privacy engine. It is intentionally isolated from all database logic —
#   it knows nothing about SQLite, sessions, or storage. Its sole job is:
#   "Given this conversation history, generate an empathetic response."
#
#   [Privacy Engine] → [THIS FILE] → [Response + Emotion Label] → [main.py → DB]
#
# WHY ISOLATE THE AI ENGINE? (Façade + Dependency Inversion)
#   By wrapping Gemini in this module, we create an abstraction boundary.
#   If Google changes the Gemini API, or if we decide to switch to OpenAI GPT
#   or a local Ollama model, we change ONLY this file. The rest of the
#   application is completely unaffected. This is the Dependency Inversion
#   Principle: high-level modules (main.py) should not depend on low-level
#   details (Gemini SDK specifics) — both should depend on abstractions.
#
# ON SYSTEM PROMPTS AS AN ETHICAL ARTIFACT:
#   The system prompt in this file is not just a technical configuration.
#   It is an ETHICAL DOCUMENT — a set of behavioral constraints that defines
#   how the AI treats vulnerable users. Its design reflects principles from:
#     - Counseling ethics (non-judgment, unconditional positive regard)
#     - AI safety (refusing harmful outputs, no medical diagnosis)
#     - HCI research (trauma-informed design)
#   This makes the system prompt one of the most academically defensible
#   components in the entire project.
# ==============================================================================

import os
import time
import logging
import json
from typing import Optional
from dataclasses import dataclass, field

import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file.
# SECURITY NOTE: The Gemini API key lives in .env (git-ignored), never in code.
# This file only reads the key from the environment — it never knows its value
# at development time, preventing accidental key exposure in version control.
load_dotenv()

# ------------------------------------------------------------------------------
# LOGGING SETUP
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)


# ==============================================================================
# SECTION 1: CONFIGURATION CONSTANTS
# ==============================================================================

# The Gemini model identifier. We use gemini-1.5-flash for the MVP because it
# offers the best balance of speed, cost, and capability for conversational tasks.
# ACADEMIC NOTE: gemini-1.5-pro would yield higher quality responses but at
# higher latency and cost — a classic engineering trade-off. For an MVP with
# a research focus (not production scale), flash is the correct choice.
GEMINI_MODEL_ID = "gemini-2.0-flash"

# Maximum number of conversation turns to retain in memory per session.
# ACADEMIC NOTE — Context Window Management:
#   LLMs have a finite "context window" (maximum tokens they can process at once).
#   Sending an entire long conversation history every time is wasteful and may
#   exceed the model's limit. We cap at 20 turns (10 user + 10 assistant) as a
#   practical balance between conversational coherence and token efficiency.
#   A production system would use a summarization strategy: compress older turns
#   into a summary instead of truncating them.
MAX_HISTORY_TURNS = 20

# Generation configuration: Controls the AI's output characteristics.
# These are hyperparameters for the language model's sampling process.
GENERATION_CONFIG = {
    # Temperature: Controls randomness. 0.0 = deterministic, 1.0 = very random.
    # 0.75 is chosen to allow natural, varied conversational responses while
    # avoiding the incoherence of high-temperature outputs. For empathetic
    # conversation, some variation is desirable — responses shouldn't feel robotic.
    "temperature": 0.75,

    # Top-P (Nucleus Sampling): Only sample from the smallest set of tokens whose
    # cumulative probability exceeds P. Works with temperature to control output
    # quality. 0.95 is a standard value that balances diversity and coherence.
    "top_p": 0.95,

    # Top-K: Limit sampling to the K most probable next tokens.
    # 40 is a standard value that prevents very unlikely (often nonsensical) tokens
    # from being selected, improving response quality.
    "top_k": 40,

    # Maximum output tokens: Hard cap on response length.
    # 1024 tokens ≈ 750-800 words — sufficient for empathetic, substantive
    # responses without being overwhelming to a user in emotional distress.
    "max_output_tokens": 1024,
}

# Safety settings: Gemini's built-in content filtering.
# ACADEMIC NOTE — Layered Safety Architecture:
#   We use BLOCK_MEDIUM_AND_ABOVE for most harm categories. This means Gemini
#   will refuse to generate content that is sexually explicit, dangerous, or
#   hateful. This is our SECOND safety layer (after our system prompt's
#   behavioral instructions). Having two independent layers (prompt + API filter)
#   is a defense-in-depth strategy — if the system prompt is somehow bypassed,
#   the API-level filter still blocks harmful outputs.
SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_LOW_AND_ABOVE"},
    # DANGEROUS_CONTENT is set to BLOCK_LOW_AND_ABOVE (stricter) because our
    # users may be emotionally vulnerable. We must be extra cautious about any
    # content that could be misinterpreted as instructions for self-harm,
    # even if the original intent was innocent.
]


# ==============================================================================
# SECTION 2: THE SYSTEM PROMPT — THE AI'S ETHICAL CONSTITUTION
# ==============================================================================

# ACADEMIC NOTE — Why System Prompts Matter:
#   A system prompt is injected before the conversation begins and sets the
#   AI's persistent behavioral constraints. It is analogous to the briefing
#   a therapist receives during training — it shapes HOW the model responds
#   to all subsequent user inputs. Unlike user messages, the system prompt
#   is not visible to the user, making its careful design a matter of
#   ethical responsibility (the user trusts the system to behave safely).
#
# DESIGN PRINCIPLES APPLIED IN THIS PROMPT:
#   1. Rogerian Person-Centered Therapy: Non-directive, non-judgmental,
#      emphasizing empathy and unconditional positive regard.
#   2. Crisis Protocol: Explicit instructions to redirect to professional
#      help for emergencies — the AI must NEVER replace a therapist.
#   3. Scope Limitation: Hard prohibitions on medical diagnosis, advice,
#      and impersonating licensed professionals.
#   4. Emotion Extraction: A structured sub-task (JSON output) that allows
#      us to capture behavioral metadata for research without asking users
#      intrusive questions about their emotional state.

SYSTEM_PROMPT = """
You are SafeSpace, a compassionate and empathetic AI companion. Your sole purpose
is to provide users with a warm, non-judgmental space to express their thoughts,
feelings, and struggles freely.

## YOUR CORE IDENTITY & PERSONA
- You are a patient, gentle, and thoughtful listener — not a therapist, doctor,
  or advice-giver. You do not diagnose, prescribe, or provide clinical guidance.
- You speak with warmth, using natural, conversational language. Avoid clinical
  jargon, bullet-point lists, or formal report-like responses. Speak like a
  caring, wise friend who is fully present.
- You have NO memory across different conversations. Each conversation is a
  fresh start. However, you remember everything within the CURRENT conversation.

## EMPATHY-FIRST RESPONSE FRAMEWORK
For every user message, follow this internal process (do not narrate it):
  1. ACKNOWLEDGE: Validate the user's emotion before anything else.
     Example: Instead of "Here are some tips for anxiety...", say
     "It sounds like you're carrying a lot right now, and that feeling of
     anxiety can be incredibly heavy. I'm really glad you're sharing this."
  2. REFLECT: Mirror back what you heard to show you were truly listening.
  3. EXPLORE (GENTLY): Ask ONE open-ended question to invite deeper sharing,
     if appropriate. Never pepper the user with multiple questions.
  4. SUPPORT: Offer gentle perspective or grounding thoughts only if it feels
     natural — never as unsolicited advice.

## ABSOLUTE RULES (NEVER VIOLATE THESE)
1. NEVER provide medical diagnoses, clinical assessments, or treatment plans.
2. NEVER claim to be a human, therapist, psychologist, or licensed counselor.
3. NEVER dismiss, minimize, or judge the user's feelings (e.g., "That's not
   such a big deal" or "You should just think positively").
4. NEVER provide specific methods, means, or details related to self-harm or
   suicide — even hypothetically or in a fictional context.
5. NEVER give specific legal or financial advice.
6. ALWAYS recommend professional help for clinical concerns. Use language like:
   "What you're describing sounds really significant, and while I'm here to
   listen, speaking with a counselor or mental health professional could give
   you the structured support you deserve."

## CRISIS PROTOCOL (HIGHEST PRIORITY)
If the user expresses IMMEDIATE intent to harm themselves or others, or if
they indicate a life-threatening emergency:
  - Respond with deep empathy and WITHOUT PANIC.
  - Immediately and clearly provide the iCall helpline (India): 9152987821
  - Provide the Vandrevala Foundation 24/7 helpline: 1860-2662-345
  - Gently encourage them to reach out to a trusted person physically present.
  - Do NOT lecture, argue, or try to "fix" the situation with logic.
  - Keep your response short, warm, and focused entirely on their safety.

## CULTURAL SENSITIVITY
- You are deployed primarily in India. Be sensitive to Indian cultural contexts:
  family pressure, academic stress, arranged marriage dynamics, career anxiety,
  and the social stigma around mental health discussions.
- Avoid Western-centric assumptions about family structures or social norms.
- If a user writes in Hinglish (Hindi + English mixed), respond naturally in
  the same style without making it feel awkward.

## NOTE ON ANONYMIZED TEXT
You may sometimes see placeholder tags like <PERSON>, <LOCATION>, or
<PHONE_NUMBER> in user messages. These are PRIVACY REDACTIONS applied by
the system. Treat them naturally in context — e.g., if a user says "I miss
<PERSON>", respond as if they mentioned a real person by name. Do not
point out or comment on these tags to the user.

## STRUCTURED RESPONSE FORMAT
You MUST structure EVERY response as a valid JSON object with exactly two fields:

{
  "response": "Your full empathetic response to the user goes here as a string.",
  "emotion": "One single word: the PRIMARY emotion you detected in the USER'S message.
              Choose from: anxious, sad, angry, hopeful, lonely, overwhelmed,
              confused, numb, grateful, fearful, ashamed, frustrated, neutral.
              If genuinely unclear, use: neutral"
}

CRITICAL JSON RULES:
- Output ONLY the JSON object. No preamble, no explanation, no markdown fences.
- The "response" field must be a properly escaped JSON string.
- The "emotion" field must be a single word from the list above — no sentences.
- If you cannot generate a safe response, set "response" to a gentle deflection
  and "emotion" to "neutral".
"""


# ==============================================================================
# SECTION 3: RESULT DATA STRUCTURE
# ==============================================================================

@dataclass
class AIResponse:
    """
    Structured container for the AI engine's output.
    Separates the user-facing response text from the research metadata (emotion),
    preventing any downstream code from accidentally mixing the two.

    Attributes:
        response_text:    The empathetic message to display to the user.
        detected_emotion: Single-word emotion label extracted from the response.
        latency_ms:       Time taken for the Gemini API call in milliseconds.
        raw_json:         The raw JSON string returned by Gemini (for debugging).
        error:            Error message if the call failed, None on success.
    """
    response_text:    str
    detected_emotion: str           = "neutral"
    latency_ms:       int           = 0
    raw_json:         Optional[str] = None
    error:            Optional[str] = None


# ==============================================================================
# SECTION 4: THE AI ENGINE CLASS
# ==============================================================================

class AIEngine:
    """
    Manages the Gemini API client, per-session conversation history,
    and the full prompt-response lifecycle.

    STATEFULNESS — In-Memory Session History:
        This class maintains a dictionary mapping session_id → conversation history.
        This gives the AI memory within a conversation (multi-turn context).

        TRADE-OFF DISCUSSION (Important for Viva):
            - ADVANTAGE: Simple, fast, zero-dependency — no Redis or external cache needed.
            - DISADVANTAGE: History lives in RAM. If the server restarts, all
              conversation history is lost. Users would start fresh.
            - PRODUCTION SOLUTION: Store history in Redis (fast key-value store)
              or in the SQLite database itself, rebuilding the history object
              from the chat_logs table on demand.
            - MVP JUSTIFICATION: For a single-server academic project with no
              persistent login, in-memory history is entirely appropriate.

    PRIVACY NOTE ON HISTORY:
        The history stored in memory contains ANONYMIZED messages (from the
        privacy engine). The AI never sees raw PII, even within its own
        conversational context. This is enforced in main.py before calling
        add_turn_and_respond().
    """

    def __init__(self):
        """
        Configures the Gemini API client and initializes the session store.
        Validates that the API key is present before proceeding.
        """
        api_key = os.getenv("GEMINI_API_KEY")

        # FAIL-FAST VALIDATION: If the API key is missing, crash immediately
        # with a clear error message rather than failing cryptically on the
        # first API call. This is the "fail fast" principle — surface
        # configuration errors as early as possible, ideally at startup.
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY not found in environment variables. "
                "Please create a .env file with GEMINI_API_KEY=your_key_here. "
                "See .env.example for reference."
            )

        # Configure the SDK with the API key.
        genai.configure(api_key=api_key)

        # Instantiate the GenerativeModel with our configuration and safety settings.
        # ARCHITECTURE NOTE: We attach generation_config and safety_settings HERE,
        # at model instantiation, so they are enforced on EVERY call automatically.
        # This is safer than passing them per-call, where a developer might forget.
        self.model = genai.GenerativeModel(
            model_name        = GEMINI_MODEL_ID,
            generation_config = GENERATION_CONFIG,
            safety_settings   = SAFETY_SETTINGS,
            system_instruction= SYSTEM_PROMPT,  # Injected as system-role context
        )

        # In-memory store: { session_id (str) → conversation history (list) }
        # Each history entry is a dict: {"role": "user"|"model", "parts": ["text"]}
        # This is the format the Gemini SDK's start_chat() method expects.
        self._session_histories: dict[str, list[dict]] = {}

        logger.info(f"AIEngine initialized with model: {GEMINI_MODEL_ID}")

    def _get_or_create_history(self, session_id: str) -> list[dict]:
        """
        Retrieves the conversation history for a session, or creates a new
        empty history list if this is the first message of the session.

        PATTERN — Lazy Initialization:
            We don't pre-allocate history for every possible session at startup.
            Instead, we create the history only when a session sends its first
            message. This is memory-efficient and avoids unnecessary initialization.

        Args:
            session_id (str): The UUID identifying the conversation session.

        Returns:
            list[dict]: The mutable conversation history list for this session.
        """
        if session_id not in self._session_histories:
            logger.info(f"New session detected. Creating history for session={session_id[:8]}...")
            self._session_histories[session_id] = []
        return self._session_histories[session_id]

    def _trim_history(self, history: list[dict]) -> list[dict]:
        """
        Ensures conversation history does not exceed MAX_HISTORY_TURNS turns.
        Trims the OLDEST messages when the limit is reached, preserving
        the most recent conversational context.

        ACADEMIC NOTE — Sliding Window Context:
            This implements a "sliding window" strategy for context management.
            Turns are stored in pairs (user + model). We convert MAX_HISTORY_TURNS
            (in turns) to MAX_HISTORY_TURNS * 2 (in individual messages),
            then slice the list to keep only the most recent entries.

            ALTERNATIVE STRATEGY (for viva discussion):
            A more sophisticated approach is "recursive summarization" — when
            the history exceeds the limit, ask the AI to summarize the oldest
            N turns into a single context message, then replace those N turns
            with the summary. This preserves long-term context at the cost
            of one extra API call.

        Args:
            history (list[dict]): The current full conversation history.

        Returns:
            list[dict]: Trimmed history, max MAX_HISTORY_TURNS * 2 messages.
        """
        max_messages = MAX_HISTORY_TURNS * 2  # Each "turn" = 1 user + 1 model message
        if len(history) > max_messages:
            # Slice from the end — keep the most recent max_messages entries.
            trimmed = history[-max_messages:]
            logger.info(
                f"History trimmed from {len(history)} to {len(trimmed)} messages "
                f"(MAX_HISTORY_TURNS={MAX_HISTORY_TURNS} reached)."
            )
            return trimmed
        return history

    def _parse_ai_response(self, raw_text: str) -> tuple[str, str]:
        """
        Parses the JSON-structured response from Gemini into its two components:
        the user-facing response text and the emotion label.

        ROBUSTNESS NOTE — Defensive Parsing:
            LLMs are probabilistic systems — they may occasionally deviate from
            the requested format despite clear instructions. This parser implements
            multiple fallback strategies to handle malformed output gracefully,
            ensuring the application never crashes due to an unexpected AI response.

            This is the "Robustness Principle" (also known as Postel's Law):
            "Be conservative in what you send, be liberal in what you accept."

        Args:
            raw_text (str): The raw text response string from the Gemini API.

        Returns:
            tuple[str, str]: (response_text, emotion_label)
        """
        # --- Strategy 1: Clean and direct JSON parse ---
        # Strip whitespace and any accidental markdown fences the model may add.
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            # Remove markdown code fences if present (e.g., ```json ... ```)
            lines = cleaned.split('\n')
            # Remove first and last line if they are fence markers
            cleaned = '\n'.join(
                line for line in lines
                if not line.strip().startswith('```')
            ).strip()

        try:
            parsed = json.loads(cleaned)
            response_text = parsed.get("response", "").strip()
            emotion_label = parsed.get("emotion", "neutral").strip().lower()

            # Validate emotion is from our expected list; default to neutral if not.
            valid_emotions = {
                "anxious", "sad", "angry", "hopeful", "lonely", "overwhelmed",
                "confused", "numb", "grateful", "fearful", "ashamed",
                "frustrated", "neutral"
            }
            if emotion_label not in valid_emotions:
                logger.warning(f"Unexpected emotion label '{emotion_label}'. Defaulting to 'neutral'.")
                emotion_label = "neutral"

            if response_text:
                return response_text, emotion_label

        except json.JSONDecodeError:
            logger.warning("Direct JSON parse failed. Attempting fallback extraction...")

        # --- Strategy 2: Substring search fallback ---
        # If JSON is malformed but the text contains our key fields, extract them.
        try:
            if '"response"' in cleaned and '"emotion"' in cleaned:
                # Find response field using string manipulation
                resp_start = cleaned.find('"response"') + len('"response"') + 2
                resp_end   = cleaned.rfind('"emotion"') - 3
                emotion_start = cleaned.rfind('"emotion"') + len('"emotion"') + 2
                emotion_end   = cleaned.rfind('}') - 1

                response_text = cleaned[resp_start:resp_end].strip().strip('"')
                emotion_label = cleaned[emotion_start:emotion_end].strip().strip('"').lower()

                if response_text:
                    logger.info("Fallback JSON extraction succeeded.")
                    return response_text, emotion_label or "neutral"

        except Exception:
            pass  # Fall through to final fallback

        # --- Strategy 3: Final fallback — treat entire response as plain text ---
        # If all parsing fails, use the raw text as the response.
        # This is a last resort that maintains UX (user gets a response)
        # at the cost of losing the emotion metadata.
        logger.error(
            "All JSON parsing strategies failed. Using raw text as response. "
            f"Raw output (first 200 chars): {raw_text[:200]}"
        )
        return raw_text.strip(), "neutral"

    def get_response(self, session_id: str, anonymized_message: str) -> AIResponse:
        """
        The primary public method. Takes a session ID and an anonymized user
        message, generates an empathetic AI response, and updates the
        conversation history.

        FULL LIFECYCLE:
            1. Retrieve/create session history
            2. Start a Gemini chat session with existing history
            3. Send the new anonymized message
            4. Parse the structured JSON response
            5. Append the new turn to history
            6. Return a structured AIResponse object

        Args:
            session_id (str):          The UUID of the conversation session.
            anonymized_message (str):  The PII-scrubbed user message.

        Returns:
            AIResponse: Structured object with response text, emotion, and metrics.
        """
        logger.info(
            f"Generating response for session={session_id[:8]}... "
            f"Message length: {len(anonymized_message)} chars."
        )

        # --- Retrieve and trim history ---
        history = self._get_or_create_history(session_id)
        history = self._trim_history(history)

        start_time = time.time()

        try:
            # --- Start a stateful chat session with the existing history ---
            # ACADEMIC NOTE — Gemini's Chat API vs. Single-Turn API:
            #   The `start_chat(history=...)` method creates a stateful chat
            #   session that automatically manages the user/model turn alternation.
            #   The alternative (single-turn `generate_content`) would require us
            #   to manually construct the full conversation prompt on every call.
            #   Using start_chat is cleaner and less error-prone for multi-turn
            #   conversation management.
            chat_session = self.model.start_chat(history=history)

            # --- Send the anonymized message ---
            # PRIVACY ENFORCEMENT: Only the anonymized_message reaches this point.
            # The raw user input was consumed by privacy_engine.py and never
            # passed to this function. This is enforced at the call site (main.py).
            gemini_response = chat_session.send_message(anonymized_message)

            # --- Measure latency ---
            latency_ms = int((time.time() - start_time) * 1000)
            logger.info(f"Gemini API response received in {latency_ms}ms.")

            # --- Extract the raw text from the response object ---
            raw_text = gemini_response.text

            # --- Parse the structured JSON response ---
            response_text, detected_emotion = self._parse_ai_response(raw_text)

            # --- Update session history ---
            # We append both the user's message and the AI's response to history.
            # This forms the "turns" that are fed back on the next request.
            # IMPORTANT: We store the AI's FULL raw JSON text in history, not just
            # the parsed response. This is because Gemini expects the history to
            # reflect exactly what the model output — rewriting it could confuse
            # the model's understanding of prior context.
            self._session_histories[session_id] = history + [
                {"role": "user",  "parts": [anonymized_message]},
                {"role": "model", "parts": [raw_text]},
            ]

            logger.info(
                f"Response generated. Emotion: '{detected_emotion}'. "
                f"History length: {len(self._session_histories[session_id])} messages."
            )

            return AIResponse(
                response_text    = response_text,
                detected_emotion = detected_emotion,
                latency_ms       = latency_ms,
                raw_json         = raw_text,
                error            = None
            )

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            logger.error(f"Gemini API call failed for session={session_id[:8]}: {e}", exc_info=True)

            # GRACEFUL DEGRADATION:
            # Return a safe, pre-written fallback response rather than propagating
            # the exception to the user. A user in distress should ALWAYS receive
            # a response — even if the AI is temporarily unavailable.
            # The error is logged for debugging while the user experience is preserved.
            fallback_response = (
                "I'm so sorry — I seem to be having a little trouble right now, "
                "but please know I'm here and I care about what you're going through. "
                "Could you give me just a moment and try sharing again? "
                "You're not alone in this."
            )
            return AIResponse(
                response_text    = fallback_response,
                detected_emotion = "neutral",
                latency_ms       = latency_ms,
                raw_json         = None,
                error            = str(e)
            )

    def clear_session(self, session_id: str) -> None:
        """
        Removes the conversation history for a given session from memory.
        Called when a user explicitly starts a new conversation or when the
        session is considered expired.

        Args:
            session_id (str): The session UUID to clear.
        """
        if session_id in self._session_histories:
            del self._session_histories[session_id]
            logger.info(f"Session history cleared for session={session_id[:8]}...")

    def get_active_session_count(self) -> int:
        """
        Returns the number of sessions currently held in memory.
        Useful for a health-check endpoint to monitor server memory usage.

        Returns:
            int: Number of active sessions in the in-memory store.
        """
        return len(self._session_histories)


# ==============================================================================
# SECTION 5: MODULE-LEVEL SINGLETON INSTANTIATION
# ==============================================================================
# ARCHITECTURE NOTE: Same Singleton pattern as privacy_engine.py.
# The AIEngine is instantiated once at import time. This means:
#   1. The API key is validated at server startup (fail-fast).
#   2. The Gemini SDK is configured once and reused across all requests.
#   3. The session history store is a single shared dictionary —
#      all requests for the same session_id access the same history.
#
# Import this in other modules as:
#   from backend.ai_engine import ai_engine
#   result = ai_engine.get_response(session_id, anonymized_text)
# ==============================================================================
logger.info("Loading AIEngine singleton at module import...")
ai_engine = AIEngine()


# ==============================================================================
# SECTION 6: STANDALONE TEST HARNESS
# ==============================================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  AIEngine — Standalone Test Harness")
    print("  NOTE: Requires GEMINI_API_KEY in your .env file.")
    print("="*60)

    TEST_SESSION_ID = "test-session-abc123"

    # Simulate a multi-turn conversation with anonymized inputs
    # (as they would arrive from the privacy engine in production).
    turns = [
        "I've been feeling really anxious about my college exams lately.",
        "Yeah, my parents expect me to get into IIT but I don't think I can do it.",
        "Sometimes I feel like <PERSON> is the only one who understands me, but even they seem distant now.",
        "I just feel really alone and like nothing is going to get better."
    ]

    for i, message in enumerate(turns, 1):
        print(f"\n[Turn {i}]")
        print(f"  USER (anonymized): {message}")

        result = ai_engine.get_response(TEST_SESSION_ID, message)

        print(f"  AI RESPONSE:  {result.response_text}")
        print(f"  EMOTION:      {result.detected_emotion}")
        print(f"  LATENCY:      {result.latency_ms}ms")
        if result.error:
            print(f"  ERROR:        {result.error}")

    print(f"\n  Active sessions in memory: {ai_engine.get_active_session_count()}")
    print("\n" + "="*60)
    print("  All test turns complete.")
    print("="*60 + "\n")
