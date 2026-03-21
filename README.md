# 🌿 SafeSpace AI
### Empathetic, Privacy-First AI Chatbot for Behavioral Research

> *A B.Tech Final Year Minor Project in Artificial Intelligence & Data Science*

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
User Input → FastAPI Backend → Presidio Anonymization → Gemini AI → SQLite Logging
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
| AI Engine | Google Gemini 1.5 Flash | Empathetic response generation |
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
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

### 2. Configure Secrets
```bash
cp .env.example .env
# Edit .env and add your Gemini API key:
# GEMINI_API_KEY=your_key_here
```

### 3. Run the Backend (Terminal 1)
```bash
uvicorn backend.main:app --reload --port 8000
```
- Chat interface: http://localhost:8000
- API docs (Swagger): http://localhost:8000/docs
- Health check: http://localhost:8000/health

### 4. Run the Dashboard (Terminal 2)
```bash
streamlit run dashboard/dashboard.py
```
- Research dashboard: http://localhost:8501

---

## Project Structure
```
safe_space_ai/
├── requirements.txt          # Pinned Python dependencies
├── .env.example              # Secrets template
├── .gitignore                # Excludes .env, *.db from git
├── backend/
│   ├── main.py               # FastAPI app, all REST endpoints
│   ├── database.py           # SQLite schema + CRUD (Repository Pattern)
│   ├── privacy_engine.py     # Presidio PII anonymization (Façade Pattern)
│   ├── ai_engine.py          # Gemini API integration + empathy prompts
│   └── models.py             # Pydantic request/response schemas
├── frontend/
│   ├── index.html            # Chat UI markup (semantic HTML5 + ARIA)
│   ├── style.css             # Trauma-informed UI design (WCAG 2.1 AA)
│   └── script.js             # Vanilla JS: session mgmt, API calls, XSS safety
├── dashboard/
│   └── dashboard.py         # Streamlit research analytics dashboard
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

---

## Academic References

- Microsoft Presidio — https://microsoft.github.io/presidio/
- Privacy by Design (Cavoukian, 2009)
- Rogerian Person-Centered Therapy principles
- WCAG 2.1 Accessibility Guidelines
- 12-Factor App Methodology — https://12factor.net/
