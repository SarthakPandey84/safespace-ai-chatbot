import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")
if not groq_key:
    print("ERROR: GROQ_API_KEY not found in .env file")
    exit()

print(f"Using Groq key: {groq_key[:10]}...")

client = Groq(api_key=groq_key)

TEST_PROMPT = """
You are an empathetic AI. For every message, respond ONLY with this exact JSON format:
{"response": "your empathetic response here", "emotion": "one word from: anxious, sad, angry, hopeful, lonely, overwhelmed, confused, numb, grateful, fearful, ashamed, frustrated, neutral"}
Output ONLY the JSON. No markdown. No code fences. No extra text.
"""

test_messages = [
    "I feel really anxious about my exams",
    "I am so sad and lonely lately",
    "I feel hopeful today for the first time",
    "Everyone would be better off without me",
]

print("=" * 60)
print("RAW GROQ OUTPUT DEBUG")
print("=" * 60)

for msg in test_messages:
    print(f"\nINPUT: {msg}")
    print("-" * 40)

    try:
        completion = client.chat.completions.create(
            model    = "llama-3.3-70b-versatile",
            messages = [
                {"role": "system", "content": TEST_PROMPT},
                {"role": "user",   "content": msg}
            ],
            temperature = 0.75,
            max_tokens  = 512,
        )

        raw = completion.choices[0].message.content
        print(f"RAW OUTPUT:\n{repr(raw)}")
        print(f"\nVISUAL:\n{raw}")

        try:
            cleaned = raw.strip()

            if "```" in cleaned:
                print("\n⚠️  WARNING: LLM wrapped output in markdown code fences!")
                cleaned = cleaned.replace("```json", "").replace("```JSON", "").replace("```", "").strip()

            first = cleaned.find("{")
            last  = cleaned.rfind("}")
            if first != -1 and last != -1:
                cleaned = cleaned[first:last+1]

            parsed = json.loads(cleaned)
            print(f"\n✅ JSON PARSED OK")
            print(f"   response: {parsed.get('response', 'MISSING')[:80]}...")
            print(f"   emotion:  {parsed.get('emotion', 'MISSING')}")

        except json.JSONDecodeError as e:
            print(f"\n❌ JSON PARSE FAILED: {e}")

    except Exception as e:
        print(f"\n❌ API ERROR: {e}")

print("\n" + "=" * 60)
print("All tests complete.")
print("=" * 60)