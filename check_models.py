import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")

if not groq_key:
    print("ERROR: GROQ_API_KEY not found in .env file")
    exit()

print(f"\nUsing Groq key: {groq_key[:10]}...")
print("=" * 50)

client = Groq(api_key=groq_key)

print("\nAvailable models on your Groq account:")
print("=" * 50)

models = client.models.list()
for m in models.data:
    print(f"  ✅ {m.id}")

print("=" * 50)
print(f"\nTotal models available: {len(models.data)}")
print("\nRecommended model for SafeSpace AI:")
print("  ✅ llama-3.3-70b-versatile  (best quality, free)")
print("  ✅ llama-3.1-8b-instant     (faster, lightweight)")