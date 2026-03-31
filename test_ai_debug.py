"""
test_ai_debug.py
Tests raw Groq API output and JSON parsing logic end-to-end.
Run: python test_ai_debug.py
"""

import os
import sys
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")
if not groq_key:
    print("ERROR: GROQ_API_KEY not found in .env file.")
    sys.exit(1)

print(f"Groq key loaded: {groq_key[:10]}...")

client = Groq(api_key=groq_key)

SYSTEM_PROMPT = """
You are an empathetic AI assistant. For EVERY message, respond ONLY with this exact JSON format:
{"response": "your empathetic response here", "emotion": "one word from: anxious, sad, angry, hopeful, lonely, overwhelmed, confused, numb, grateful, fearful, ashamed, frustrated, neutral"}

Rules:
- Output ONLY raw JSON. No markdown, no code fences, no extra text.
- The "emotion" field must be exactly one word from the list above.
""".strip()

TEST_MESSAGES = [
    "I feel really anxious about my exams.",
    "I am so sad and lonely lately.",
    "I feel hopeful today for the first time.",
    "Everyone would be better off without me.",
]


def parse_response(raw: str) -> dict | None:
    """Strip markdown fences and extract the first JSON object from raw text."""
    cleaned = raw.strip()

    if "```" in cleaned:
        print("  ⚠️  WARNING: Model wrapped output in markdown code fences — stripping.")
        cleaned = cleaned.replace("```json", "").replace("```JSON", "").replace("```", "").strip()

    start = cleaned.find("{")
    end   = cleaned.rfind("}")
    if start == -1 or end == -1:
        return None

    try:
        return json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None


print("\n" + "=" * 60)
print("RAW GROQ OUTPUT DEBUG")
print("=" * 60)

for msg in TEST_MESSAGES:
    print(f"\nINPUT : {msg}")
    print("-" * 40)

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": msg},
            ],
            temperature=0.75,
            max_tokens=512,
        )

        raw = completion.choices[0].message.content
        print(f"RAW   : {repr(raw)}")
        print(f"VISUAL:\n{raw}")

        parsed = parse_response(raw)
        if parsed:
            print(f"\n✅  JSON parsed OK")
            print(f"    response : {str(parsed.get('response', 'MISSING'))[:80]}...")
            print(f"    emotion  : {parsed.get('emotion', 'MISSING')}")
        else:
            print("\n❌  JSON parse FAILED — check model output above.")

    except Exception as e:
        print(f"\n❌  API ERROR: {e}")

print("\n" + "=" * 60)
print("All tests complete.")
print("=" * 60)