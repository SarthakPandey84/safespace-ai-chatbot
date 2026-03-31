"""
check_models.py
Lists all Groq models available on your account.
Run: python check_models.py
"""

import os
import sys
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")
if not groq_key:
    print("ERROR: GROQ_API_KEY not found in .env file.")
    sys.exit(1)

print(f"\nGroq key loaded: {groq_key[:10]}...")
print("=" * 50)

client = Groq(api_key=groq_key)
models = client.models.list()

print("\nAvailable models on your Groq account:")
print("=" * 50)
for m in sorted(models.data, key=lambda x: x.id):
    print(f"  ✅  {m.id}")

print("=" * 50)
print(f"\nTotal: {len(models.data)} models")
print("\nRecommended for SafeSpace AI:")
print("  ✅  llama-3.3-70b-versatile   (best quality, free tier)")
print("  ✅  llama-3.1-8b-instant      (faster, lightweight)")