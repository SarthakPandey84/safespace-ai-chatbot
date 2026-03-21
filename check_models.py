# check_models.py — Lists all Gemini models available on your API key
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

print("\nAvailable models on your API key:")
print("=" * 50)
for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        print(f"  ✅ {m.name}")
print("=" * 50)