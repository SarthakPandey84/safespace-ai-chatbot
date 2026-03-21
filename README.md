# 🌿 SafeSpace AI
### Empathetic, Privacy-First AI Chatbot for Behavioral Research

> *B.Tech Final Year Minor Project — Artificial Intelligence & Data Science*

---

## Project Overview

SafeSpace AI is an empathetic chatbot that provides users with a judgment-free space
to express their thoughts and feelings. It is built on a **Privacy-by-Design** architecture
where all Personally Identifiable Information (PII) is anonymized **before** any data
is sent to an AI model or stored in the database.

Anonymized behavioral data is visualized on a Streamlit research dashboard for
academic mental health and HCI research.

---

## Architecture

```
User Input → FastAPI Backend → Presidio Anonymization → Groq AI → SQLite Logging
                                        ↑
                               PII destroyed here.
                               Never travels further.

Streamlit Dashboard ← reads anonymized data ← SQLite
```

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Frontend | HTML · CSS · Vanilla JS | Chat interface |
| Backend API | FastAPI + Uvicorn | REST endpoints, session management |
| AI Engine | Groq (Llama 3.3 70B) | Empathetic response generation |
| Anonymization | Microsoft Presidio + SpaCy | PII detection and scrubbing |
| Database | SQLite | Anonymized chat log storage |
| Dashboard | Streamlit + Plotly | Research data visualization |

---

## Quick Start

### 1. Clone & Set Up Environment
```bash
git clone <your-repo-url>
cd safe_space_ai

python -m venv venv
venv\Scripts\activate.bat        # Windows
# source venv/bin/activate       # Linux/Mac

pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

### 2. Configure Secrets
```bash
copy .env.example .env
```

Edit `.env` and add your API key:
```
GROQ_API_KEY=your_groq_key_here
```

Get a free Groq API key at: https://console.groq.com

### 3. Run the Backend (Terminal 1)
```bash
uvicorn backend.main:app --reload --port 8000
```

| URL | Purpose |
|---|---|
| http://localhost:8000 | Chat interface |
| http://localhost:8000/docs | API docs (Swagger) |
| http://localhost:8000/health | Health check |

### 4. Run the Dashboard (Terminal 2)
```bash
streamlit run dashboard/dashboard.py
```

| URL | Purpose |
|---|---|
| http://localhost:8501 | Research dashboard |

### 5. Validate Setup
```bash
python validate_setup.py
```

---

## Project Structure

```
safe_space_ai/
├── requirements.txt          # Pinned Python dependencies
├── .env.example              # Secrets template
├── .gitignore                # Excludes .env, *.db from git
├── validate_setup.py         # Pre-demo setup verification script
├── backend/
│   ├── main.py               # FastAPI app, all REST endpoints, crisis override
│   ├── database.py           # SQLite schema + CRUD (Repository Pattern)
│   ├── privacy_engine.py     # Presidio PII anonymization (Façade Pattern)
│   ├── ai_engine.py          # Groq API integration + empathy system prompt
│   └── models.py             # Pydantic request/response schemas
├── frontend/
│   ├── index.html            # Chat UI markup (semantic HTML5 + ARIA)
│   ├── style.css             # Trauma-informed UI design (WCAG 2.1 AA)
│   └── script.js             # Vanilla JS: session mgmt, API calls, XSS safety
├── dashboard/
│   └── dashboard.py          # Streamlit research analytics dashboard
└── data/                     # Auto-created at runtime (git-ignored)
    └── safespace.db          # SQLite database
```

---

## Key Privacy Features

- **No registration required** — anonymous UUID sessions only
- **PII scrubbed in memory** — names, phones, emails, Aadhaar, PAN never reach the AI or database
- **Anonymized storage** — `<PERSON>`, `<PHONE_NUMBER>` placeholders stored, not real values
- **Session expiry** — browser tab close = session UUID gone (sessionStorage)
- **Fail-safe pipeline** — anonymization failure aborts the request; raw PII never leaks
- **Crisis override** — server-side keyword detection guarantees helpline numbers are shown

---

## Crisis Safety Features

If a user expresses distress or suicidal ideation, the system:
1. Detects crisis keywords server-side (independent of AI response)
2. Injects both helpline numbers directly into the response
3. Always displays iCall India (9152987821) and Vandrevala Foundation (1860-2662-345)

---

## Academic References

- Microsoft Presidio — https://microsoft.github.io/presidio/
- Privacy by Design (Cavoukian, 2009)
- Rogerian Person-Centered Therapy principles
- WCAG 2.1 Accessibility Guidelines
- 12-Factor App Methodology — https://12factor.net/
- Groq API — https://console.groq.com/docs