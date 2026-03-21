# test_ai_debug.py
# Run with: python test_ai_debug.py
# This shows the RAW output from Gemini so we can see what's going wrong.

import os
from dotenv import load_dotenv
import google.generativeai as genai
import json

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("ERROR: GEMINI_API_KEY not found in .env file")
    exit()

genai.configure(api_key=api_key)

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config={
        "temperature": 0.75,
        "max_output_tokens": 1024,
    }
)

# Simplified system prompt — pure JSON instruction
TEST_PROMPT = """
You are an empathetic AI. For every message, respond ONLY with this exact JSON format:
{
  "response": "your empathetic response here",
  "emotion": "one word from: anxious, sad, angry, hopeful, lonely, overwhelmed, confused, numb, grateful, fearful, ashamed, frustrated, neutral"
}

Output ONLY the JSON. No markdown. No code fences. No extra text.
"""

test_messages = [
    "I feel really anxious about my exams",
    "I am so sad and lonely lately",
    "I feel hopeful today for the first time",
]

print("=" * 60)
print("RAW GEMINI OUTPUT DEBUG")
print("=" * 60)

for msg in test_messages:
    print(f"\nINPUT: {msg}")
    print("-" * 40)
    
    response = model.generate_content(TEST_PROMPT + "\n\nUser message: " + msg)
    raw = response.text
    
    print(f"RAW OUTPUT:\n{repr(raw)}")
    print(f"\nVISUAL:\n{raw}")
    
    # Try parsing
    try:
        cleaned = raw.strip()
        if "```" in cleaned:
            print("\n⚠️  WARNING: Gemini wrapped output in markdown code fences!")
            lines = cleaned.split('\n')
            cleaned = '\n'.join(
                line for line in lines
                if not line.strip().startswith('```')
            ).strip()
        
        parsed = json.loads(cleaned)
        print(f"\n✅ JSON PARSED OK")
        print(f"   response: {parsed.get('response', 'MISSING')[:60]}...")
        print(f"   emotion:  {parsed.get('emotion', 'MISSING')}")
    except json.JSONDecodeError as e:
        print(f"\n❌ JSON PARSE FAILED: {e}")

print("\n" + "=" * 60)