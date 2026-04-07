# 🌿 SafeSpace AI

### Empathetic, Privacy-First AI Chatbot for Behavioral Research

> *B.Tech Final Project — Artificial Intelligence & Data Science*

[![Backend](https://img.shields.io/badge/Backend-Render-46E3B7?logo=render)](https://safespace-ai-backend.onrender.com/)
[![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit%20Cloud-FF4B4B?logo=streamlit)](https://safespace-ai-dashboard.streamlit.app/)
[![DB](https://img.shields.io/badge/Database-Supabase%20PostgreSQL-3ECF8E?logo=supabase)](https://supabase.com/)
[![AI](https://img.shields.io/badge/AI-Groq%20Llama%203.3%2070B-F55036?logo=meta)](https://console.groq.com/)

---

## Project Overview

SafeSpace AI is an empathetic chatbot that provides users a judgment-free space to express their thoughts and feelings. It is built on a **Privacy-by-Design** architecture where all Personally Identifiable Information (PII) — including Indian names, phone numbers, and government IDs — is anonymized **before** any data is sent to an AI model or stored in the database.

Anonymized behavioral data is visualized on a Streamlit research dashboard for academic mental health and HCI research.

---

## Architecture

```
User Input → FastAPI Backend → Presidio Anonymization → Groq AI (Llama 3.3 70B)
                                        ↑                        ↓
                               PII destroyed here.        Anonymized response
                               Never travels further.           ↓
                                                    Supabase (PostgreSQL) Logging

Streamlit Dashboard ← reads anonymized data ← Supabase/PostgreSQL
```

### Deployment Architecture

```
┌─────────────────────┐     ┌──────────────────────────┐     ┌─────────────────────┐
│   Frontend (HTML)   │────▶│  FastAPI on Render        │────▶│  Supabase           │
│  Served by FastAPI  │     │  (Free tier, IPv6-only)   │     │  PostgreSQL DB      │
└─────────────────────┘     │  Port: 10000              │     │  Transaction Pooler │
                            └──────────────────────────┘     │  Port: 6543         │
                                        │                     └─────────────────────┘
                            ┌──────────────────────────┐              ▲
                            │  Streamlit Cloud          │──────────────┘
                            │  Dashboard (IPv6-only)    │
                            └──────────────────────────┘
```

> **Note on Free Tier Networking:** Both Render and Streamlit Cloud use IPv6-only outbound networking. Supabase's direct connection URL is incompatible — the **Transaction Pooler URL** (port 6543) must be used for both services.

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Frontend | HTML · CSS · Vanilla JS | Chat interface |
| Backend API | FastAPI + Uvicorn | REST endpoints, session management |
| AI Engine | Groq (Llama 3.3 70B Versatile) | Empathetic response generation |
| Anonymization | Microsoft Presidio + SpaCy `en_core_web_lg` | PII detection and scrubbing |
| Database | Supabase (PostgreSQL) | Cloud-hosted anonymized chat log storage |
| Dashboard | Streamlit + Plotly | Research data visualization (live, auto-refresh) |
| Deployment | Render (backend) · Streamlit Cloud (dashboard) | Cloud hosting |

---

## Quick Start (Local Development)

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/SarthakPandey84/safespace-ai-chatbot.git
cd safespace-ai-chatbot

python -m venv venv
venv\Scripts\activate.bat        # Windows
# source venv/bin/activate       # Linux/Mac

pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

> **Important:** This project uses `en_core_web_lg` (large model), not `en_core_web_sm`. The large model has significantly better NER for South Asian names, which is required for reliable PII anonymization.

### 2. Configure Secrets

```bash
copy env.example .env       # Windows
# cp env.example .env       # Linux/Mac
```

Edit `.env` with your credentials:

```env
GROQ_API_KEY=your_groq_key_here

# For cloud DB (Supabase Transaction Pooler — required for IPv6-only hosts)
DATABASE_URL=postgresql://postgres.YOURPROJECTREF:PASSWORD@aws-0-region.pooler.supabase.com:6543/postgres

# Leave DATABASE_URL unset to use SQLite locally
```

Get a free Groq API key at: https://console.groq.com

### 3. Run the Backend (Terminal 1)

```bash
uvicorn backend.main:app --reload --port 8000
```

| URL | Purpose |
|---|---|
| http://localhost:8000 | Chat interface |
| http://localhost:8000/docs | API docs (Swagger UI) |
| http://localhost:8000/health | Health check |

### 4. Run the Dashboard (Terminal 2)

```bash
streamlit run dashboard/dashboard.py
```

| URL | Purpose |
|---|---|
| http://localhost:8501 | Research analytics dashboard |

### 5. Validate Setup

```bash
python validate_setup.py
```

---

## Database Configuration

The project uses an **environment-aware database pattern**:

- **Local dev** (`DATABASE_URL` not set): SQLite at `data/safespace.db` — zero config, works out of the box.
- **Production** (`DATABASE_URL` set): Supabase PostgreSQL via the Transaction Pooler URL.

### Setting up Supabase (Production)

1. Create a project at [supabase.com](https://supabase.com)
2. Run this in the Supabase **SQL Editor**:

```sql
CREATE TABLE IF NOT EXISTS chat_logs (
    id SERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    user_message TEXT,
    bot_response TEXT,
    emotion_detected TEXT,
    is_crisis BOOLEAN DEFAULT FALSE,
    anonymized_user_message TEXT,
    message_length INTEGER,
    response_time_ms INTEGER
);
```

3. Copy the **Transaction Pooler** connection string from:
   `Project Settings → Database → Connection string → Transaction`
4. Set it as `DATABASE_URL` in your environment / Render config.

> **Why Transaction Pooler?** Free-tier Render and Streamlit Cloud use IPv6-only outbound networking. Supabase's direct connection URL resolves to IPv4-only and silently fails. The Transaction Pooler (port 6543) is IPv6-compatible and is required for both platforms.

---

## Project Structure

```
safespace-ai-chatbot/
├── requirements.txt          # Pinned Python dependencies
├── env.example               # Secrets template
├── render.yaml               # Render deployment config (downloads en_core_web_lg)
├── Procfile                  # Process definition for deployment
├── .gitignore                # Excludes .env, *.db from git
├── validate_setup.py         # Pre-demo setup verification script
├── check_models.py           # SpaCy model availability checker
├── test_ai_debug.py          # AI engine debug/test script
│
├── backend/
│   ├── main.py               # FastAPI app, REST endpoints, crisis override
│   ├── database.py           # PostgreSQL/SQLite CRUD (env-aware, Repository Pattern)
│   ├── privacy_engine.py     # Presidio PII anonymization (Façade Pattern)
│   ├── ai_engine.py          # Groq API integration + empathy system prompt
│   └── models.py             # Pydantic request/response schemas
│
├── frontend/
│   ├── index.html            # Chat UI markup (semantic HTML5 + ARIA)
│   ├── style.css             # Trauma-informed UI design (WCAG 2.1 AA)
│   └── script.js             # Vanilla JS: session management, API calls, XSS safety
│
└── dashboard/
    └── dashboard.py          # Streamlit research analytics dashboard (live auto-refresh)
```

---

## Key Privacy Features

- **No registration required** — anonymous UUID sessions only
- **PII scrubbed in memory** — names, phones, emails, Aadhaar, PAN never reach the AI or database
- **Indian name coverage** — deny-list of 60+ common Indian first names + regex recognizer for name-introduction phrases (e.g., *"hi I am Arjun"*) + `en_core_web_lg` for NER
- **Anonymized storage** — `<PERSON>`, `<PHONE_NUMBER>`, `<EMAIL_ADDRESS>` placeholders stored, never real values
- **Session expiry** — browser tab close = session UUID gone (`sessionStorage`)
- **Fail-safe pipeline** — anonymization failure aborts the request; raw PII never leaks
- **Crisis override** — server-side keyword detection guarantees helpline numbers are shown regardless of AI response

---

## Crisis Safety Features

If a user expresses distress or suicidal ideation, the system:

1. Detects crisis keywords **server-side**, independent of the AI response
2. Injects both helpline numbers directly into the response
3. Always displays **iCall India** (9152987821) and **Vandrevala Foundation** (1860-2662-345)

---

## PII Anonymization — Technical Details

The anonymization pipeline (Microsoft Presidio) is configured with:

- **SpaCy `en_core_web_lg`** — large NLP model with better NER accuracy for South Asian names
- **Fallback chain** — tries `en_core_web_lg` → `en_core_web_sm` → blank model to prevent startup failures
- **Indian name deny-list** — 60+ common Indian first names hardcoded as a `PatternRecognizer`
- **Name-introduction regex** — catches patterns like *"my name is X"*, *"hi I am X"*, *"I'm X"* regardless of NER
- **Standard Presidio recognizers** — phone numbers, email addresses, Aadhaar (12-digit), PAN card, credit cards, IP addresses

---

## Deployment Notes (Render + Streamlit Cloud)

### Backend (Render)

`render.yaml` downloads `en_core_web_lg` at build time:

```yaml
buildCommand: pip install -r requirements.txt && python -m spacy download en_core_web_lg
```

Ensure `DATABASE_URL` is set as an environment variable in Render's dashboard pointing to the Supabase Transaction Pooler URL.

### Dashboard (Streamlit Cloud)

Set `DATABASE_URL` in Streamlit Cloud's **Secrets** (TOML format):

```toml
DATABASE_URL = "postgresql://postgres.YOURREF:PASSWORD@aws-0-region.pooler.supabase.com:6543/postgres"
```

The dashboard uses `st.cache_data(ttl=30)` + a `st.rerun()` timer for live data refresh.

---

## Academic References

- Microsoft Presidio — https://microsoft.github.io/presidio/
- Privacy by Design (Cavoukian, 2009)
- Rogerian Person-Centered Therapy principles
- WCAG 2.1 Accessibility Guidelines
- 12-Factor App Methodology — https://12factor.net/
- Groq API — https://console.groq.com/docs
- SpaCy NER — https://spacy.io/models/en

---

## Live Demo

| Service | URL |
|---|---|
| Chat Interface | https://safespace-ai-backend.onrender.com/ |
| API Docs | https://safespace-ai-backend.onrender.com/docs |

> *Free-tier Render services spin down after inactivity. First load may take 30–60 seconds.*

---

*B.Tech Minor Project — Artificial Intelligence & Data Science*