"""
validate_setup.py
Pre-flight check — run before demo to verify the entire project is ready.
Run: python validate_setup.py
"""

import os
import sys
import subprocess

print("=" * 55)
print("  SafeSpace AI — Setup Validation")
print("=" * 55)

errors   = []
warnings = []


# ── 1. .env file & API key ────────────────────────────────
if not os.path.exists(".env"):
    errors.append(".env file missing — run: cp .env.example .env")
else:
    from dotenv import load_dotenv
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        errors.append("GROQ_API_KEY missing in .env")
    else:
        print("✅  GROQ_API_KEY found in .env")


# ── 2. Required source files ──────────────────────────────
REQUIRED_FILES = [
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

for path in REQUIRED_FILES:
    if os.path.exists(path):
        print(f"✅  {path}")
    else:
        errors.append(f"Missing file: {path}")


# ── 3. Sensitive files not tracked by git ────────────────
result = subprocess.run(
    ["git", "ls-files", ".env", "venv/"],
    capture_output=True, text=True
)
if result.stdout.strip():
    errors.append(f"DANGER — sensitive files tracked by git: {result.stdout.strip()}")
else:
    print("✅  No sensitive files tracked by git")


# ── 4. Package imports ────────────────────────────────────
print("\nChecking installed packages...")

def check_import(package: str, label: str | None = None) -> bool:
    """Try to import a package; log success or add to errors."""
    display = label or package
    try:
        mod = __import__(package)
        version = getattr(mod, "__version__", "")
        print(f"✅  {display}" + (f" {version}" if version else ""))
        return True
    except ImportError:
        errors.append(f"{display} not installed — run: pip install -r requirements.txt")
        return False

check_import("fastapi")
check_import("groq")
check_import("presidio_analyzer", "presidio-analyzer")
check_import("streamlit")

try:
    import spacy
    spacy.load("en_core_web_lg")
    print("✅  spacy en_core_web_lg loaded")
except OSError:
    warnings.append("SpaCy model missing — run: python -m spacy download en_core_web_lg")
except ImportError:
    errors.append("spacy not installed — run: pip install -r requirements.txt")


# ── 5. Data directory ─────────────────────────────────────
if os.path.exists("data/safespace.db"):
    print("✅  data/safespace.db exists")
else:
    warnings.append("data/safespace.db not found — will be created on first run")


# ── Summary ───────────────────────────────────────────────
print("\n" + "=" * 55)

if errors:
    print(f"❌  {len(errors)} ERROR(S) FOUND:")
    for e in errors:
        print(f"   → {e}")
else:
    print("✅  All checks passed — project is ready!")

if warnings:
    print(f"\n⚠️   {len(warnings)} WARNING(S):")
    for w in warnings:
        print(f"   → {w}")

print("=" * 55)
print("\nTo start the project:")
print("  Terminal 1 : uvicorn backend.main:app --reload --port 8000")
print("  Terminal 2 : streamlit run dashboard/dashboard.py")