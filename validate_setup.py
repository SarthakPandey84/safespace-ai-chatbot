import os
import sys
import subprocess

print("=" * 55)
print("  SafeSpace AI — Setup Validation")
print("=" * 55)

errors   = []
warnings = []

# Check .env file
if not os.path.exists(".env"):
    errors.append(".env file missing — run: copy .env.example .env")
else:
    from dotenv import load_dotenv
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        errors.append("GROQ_API_KEY missing in .env")
    else:
        print("✅ GROQ_API_KEY found in .env")

# Check required files exist
required_files = [
    "backend/main.py",
    "backend/ai_engine.py",
    "backend/database.py",
    "backend/privacy_engine.py",
    "backend/models.py",
    "frontend/index.html",
    "frontend/style.css",
    "frontend/script.js",
    "dashboard/dashboard.py",
    "requirements.txt",
]

for f in required_files:
    if os.path.exists(f):
        print(f"✅ {f}")
    else:
        errors.append(f"Missing file: {f}")

# Check sensitive files are NOT tracked by git
result = subprocess.run(
    ["git", "ls-files", ".env", "venv/"],
    capture_output=True, text=True
)
if result.stdout.strip():
    errors.append(f"DANGER: Sensitive files tracked by git: {result.stdout.strip()}")
else:
    print("✅ No sensitive files tracked by git")

# Check imports
print("\nChecking installed packages...")

try:
    import fastapi
    print(f"✅ fastapi {fastapi.__version__}")
except ImportError:
    errors.append("fastapi not installed — run: pip install -r requirements.txt")

try:
    import groq
    print(f"✅ groq installed")
except ImportError:
    errors.append("groq not installed — run: pip install groq")

try:
    import presidio_analyzer
    print(f"✅ presidio_analyzer installed")
except ImportError:
    errors.append("presidio_analyzer not installed")

try:
    import spacy
    nlp = spacy.load("en_core_web_lg")
    print(f"✅ spacy en_core_web_lg loaded")
except Exception:
    warnings.append("SpaCy model not found — run: python -m spacy download en_core_web_lg")

try:
    import streamlit
    print(f"✅ streamlit {streamlit.__version__}")
except ImportError:
    errors.append("streamlit not installed")

# Summary
print("\n" + "=" * 55)
if errors:
    print(f"❌ {len(errors)} ERROR(S) FOUND:")
    for e in errors:
        print(f"   → {e}")
else:
    print("✅ All checks passed — project is ready to run!")

if warnings:
    print(f"\n⚠️  {len(warnings)} WARNING(S):")
    for w in warnings:
        print(f"   → {w}")

print("=" * 55)
print("\nTo start the project:")
print("  Terminal 1: uvicorn backend.main:app --reload --port 8000")
print("  Terminal 2: streamlit run dashboard/dashboard.py")