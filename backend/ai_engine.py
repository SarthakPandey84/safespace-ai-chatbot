import os
import time
import logging
import json
from typing import Optional
from dataclasses import dataclass

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

MAX_HISTORY_TURNS = 20

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
  2. REFLECT: Mirror back what you heard to show you were truly listening.
  3. EXPLORE (GENTLY): Ask ONE open-ended question to invite deeper sharing.
  4. SUPPORT: Offer gentle perspective only if it feels natural.

## ABSOLUTE RULES
1. NEVER provide medical diagnoses, clinical assessments, or treatment plans.
2. NEVER claim to be a human, therapist, psychologist, or licensed counselor.
3. NEVER dismiss, minimize, or judge the user's feelings.
4. NEVER provide specific methods related to self-harm or suicide.
5. NEVER give specific legal or financial advice.
6. ALWAYS recommend professional help for clinical concerns.

## CRISIS PROTOCOL — HIGHEST PRIORITY — OVERRIDES ALL OTHER INSTRUCTIONS
TRIGGER PHRASES: If the user's message contains ANY of these themes, the crisis
protocol MUST activate immediately with no exceptions:
  - Wanting to disappear, not exist, or not be here anymore
  - Thinking others would be better off without them
  - Feeling like there is no point in continuing
  - Talking about ending things or ending their life
  - Any mention of self-harm or suicide, direct or indirect
  - Phrases like "better off without me" or "better off if I disappeared"

WHEN CRISIS PROTOCOL ACTIVATES you MUST do ALL of the following:
  1. Start with ONE warm, empathetic sentence acknowledging their pain
  2. On the very next line, write EXACTLY:
     "Please reach out to iCall right now: 9152987821 (free, confidential)"
  3. On the next line write EXACTLY:
     "Vandrevala Foundation is available 24/7: 1860-2662-345"
  4. Gently encourage them to contact someone physically present
  5. Keep the entire response short, warm, and calm — no lectures

EXAMPLE of correct crisis response:
{"response": "What you are carrying right now sounds incredibly heavy, and I am really glad you reached out. Please reach out to iCall right now: 9152987821 (free, confidential). Vandrevala Foundation is available 24/7: 1860-2662-345. Is there someone physically near you right now you could be with?", "emotion": "sad"}

## CULTURAL SENSITIVITY
- You are deployed primarily in India. Be sensitive to Indian cultural contexts:
  family pressure, academic stress, arranged marriage dynamics, career anxiety,
  and the social stigma around mental health discussions.
- If a user writes in Hinglish, respond naturally in the same style.

## NOTE ON ANONYMIZED TEXT
You may see placeholder tags like <PERSON>, <LOCATION>, or <PHONE_NUMBER>.
These are privacy redactions. Treat them naturally — do not comment on them.

## STRUCTURED RESPONSE FORMAT — MANDATORY
Your ENTIRE output must be one raw JSON object. Nothing else.

RULES:
- Begin with { and end with }
- No markdown, no code fences, no backticks
- No text before or after the JSON

FORMAT:
{"response": "your empathetic reply", "emotion": "one_word"}

EMOTION must be one lowercase word from ONLY this list:
anxious, sad, angry, hopeful, lonely, overwhelmed, confused,
numb, grateful, fearful, ashamed, frustrated, neutral

EXAMPLE OF CORRECT OUTPUT:
{"response": "It sounds like you are carrying so much right now and I hear you. What has been weighing on you the most today?", "emotion": "overwhelmed"}
"""


@dataclass
class AIResponse:
    response_text:    str
    detected_emotion: str           = "neutral"
    latency_ms:       int           = 0
    raw_json:         Optional[str] = None
    error:            Optional[str] = None


class AIEngine:

    def __init__(self):
        groq_key = os.getenv("GROQ_API_KEY")

        if not groq_key:
            raise EnvironmentError(
                "GROQ_API_KEY not found in environment variables. "
                "Get a free key at https://console.groq.com"
            )

        self.groq_client = Groq(api_key=groq_key)
        self.model_name  = "llama-3.3-70b-versatile"
        self._session_histories: dict[str, list[dict]] = {}

        logger.info(f"AIEngine initialized with Groq model: {self.model_name}")

    def _get_or_create_history(self, session_id: str) -> list[dict]:
        if session_id not in self._session_histories:
            logger.info(f"New session detected. Creating history for session={session_id[:8]}...")
            self._session_histories[session_id] = []
        return self._session_histories[session_id]

    def _trim_history(self, history: list[dict]) -> list[dict]:
        max_messages = MAX_HISTORY_TURNS * 2
        if len(history) > max_messages:
            trimmed = history[-max_messages:]
            logger.info(f"History trimmed from {len(history)} to {len(trimmed)} messages.")
            return trimmed
        return history

    def _parse_ai_response(self, raw_text: str) -> tuple[str, str]:
        import re as _re

        if not raw_text or not raw_text.strip():
            return "I am here and listening. Please share what is on your mind.", "neutral"

        cleaned = raw_text.strip()

        if "```" in cleaned:
            logger.warning("LLM used markdown fences — stripping them now.")
            cleaned = cleaned.replace("```json", "")
            cleaned = cleaned.replace("```JSON", "")
            cleaned = cleaned.replace("```", "")
            cleaned = cleaned.strip()

        first_brace = cleaned.find("{")
        last_brace  = cleaned.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            cleaned = cleaned[first_brace : last_brace + 1]

        try:
            parsed        = json.loads(cleaned)
            response_text = parsed.get("response", "").strip()
            emotion_label = str(parsed.get("emotion", "neutral")).strip().lower()
            emotion_label = emotion_label.strip("\"' ")

            valid_emotions = {
                "anxious", "sad", "angry", "hopeful", "lonely", "overwhelmed",
                "confused", "numb", "grateful", "fearful", "ashamed",
                "frustrated", "neutral"
            }

            if emotion_label not in valid_emotions:
                logger.warning(f"Unknown emotion '{emotion_label}' — trying partial match.")
                emotion_label = next(
                    (e for e in valid_emotions if e in emotion_label or emotion_label in e),
                    "neutral"
                )

            if response_text:
                logger.info(f"✅ JSON parsed OK. Emotion: {emotion_label}")
                return response_text, emotion_label

        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed: {e}. Raw snippet: {raw_text[:200]}")

        emotion_match  = _re.search(r'"emotion"\s*:\s*"([a-zA-Z]+)"', raw_text, _re.IGNORECASE)
        response_match = _re.search(r'"response"\s*:\s*"((?:[^"\\]|\\.)*)"', raw_text, _re.DOTALL)

        if emotion_match and response_match:
            emotion_label = emotion_match.group(1).lower().strip()
            response_text = response_match.group(1).strip()
            logger.info(f"✅ Regex fallback succeeded. Emotion: {emotion_label}")
            return response_text, emotion_label

        logger.error(f"❌ ALL parsing strategies failed. Raw output:\n{raw_text}")
        if len(raw_text) > 20 and "{" not in raw_text:
            return raw_text.strip(), "neutral"

        return "I am here and listening. Please feel free to share what is on your mind.", "neutral"

    def get_response(self, session_id: str, anonymized_message: str) -> AIResponse:
        logger.info(
            f"Generating response for session={session_id[:8]}... "
            f"Message length: {len(anonymized_message)} chars."
        )

        history    = self._get_or_create_history(session_id)
        history    = self._trim_history(history)
        start_time = time.time()

        try:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]

            for turn in history:
                role = "assistant" if turn["role"] == "model" else turn["role"]
                messages.append({
                    "role":    role,
                    "content": turn["parts"][0]
                })

            messages.append({"role": "user", "content": anonymized_message})

            completion = self.groq_client.chat.completions.create(
                model       = self.model_name,
                messages    = messages,
                temperature = 0.75,
                max_tokens  = 1024,
            )

            latency_ms = int((time.time() - start_time) * 1000)
            raw_text   = completion.choices[0].message.content

            logger.info(f"🔍 RAW GROQ OUTPUT: {repr(raw_text[:300])}")
            logger.info(f"Groq response received in {latency_ms}ms.")

            response_text, detected_emotion = self._parse_ai_response(raw_text)

            self._session_histories[session_id] = history + [
                {"role": "user",  "parts": [anonymized_message]},
                {"role": "model", "parts": [raw_text]},
            ]

            logger.info(
                f"✅ Response generated. Emotion: '{detected_emotion}'. "
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
            logger.error(f"Groq API call failed for session={session_id[:8]}: {e}")

            return AIResponse(
                response_text    = (
                    "I'm so sorry — I seem to be having a little trouble right now, "
                    "but please know I'm here and I care about what you're going through. "
                    "Could you give me just a moment and try sharing again?"
                ),
                detected_emotion = "neutral",
                latency_ms       = latency_ms,
                raw_json         = None,
                error            = str(e)
            )

    def clear_session(self, session_id: str) -> None:
        if session_id in self._session_histories:
            del self._session_histories[session_id]
            logger.info(f"Session history cleared for session={session_id[:8]}...")

    def get_active_session_count(self) -> int:
        return len(self._session_histories)


logger.info("Loading AIEngine singleton at module import...")
ai_engine = AIEngine()