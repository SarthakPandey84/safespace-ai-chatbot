"""
dashboard/dashboard.py
SafeSpace AI — Research Analytics Dashboard

Reads from Supabase (PostgreSQL) when DATABASE_URL is set,
falls back to local SQLite for local dev.

Run: streamlit run dashboard/dashboard.py
"""

import os
import time
from pathlib import Path

import pandas as pd
import streamlit as st

# Load .env for local dev
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

DATABASE_URL   = os.getenv("DATABASE_URL")
USE_POSTGRES   = bool(DATABASE_URL)
AUTO_REFRESH_S = 10
SQLITE_PATH    = Path(__file__).parent.parent / "data" / "safespace.db"

EMOTION_COLOURS = {
    "anxious": "#f39c12", "sad": "#3498db", "angry": "#e74c3c",
    "hopeful": "#2ecc71", "lonely": "#9b59b6", "overwhelmed": "#e67e22",
    "confused": "#1abc9c", "numb": "#95a5a6", "grateful": "#27ae60",
    "fearful": "#c0392b", "ashamed": "#8e44ad", "frustrated": "#d35400",
    "neutral": "#7f8c8d",
}

# Columns match the actual chat_logs schema
QUERY = """
    SELECT
        id, session_id, timestamp,
        user_message_anon,
        ai_response,
        emotion,
        crisis_detected,
        pii_entities_found,
        response_latency_ms
    FROM chat_logs
    ORDER BY timestamp DESC
"""


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=AUTO_REFRESH_S)
def load_data() -> pd.DataFrame:
    """
    Loads all chat logs. TTL forces a fresh DB query every AUTO_REFRESH_S seconds.
    Opens and closes the connection immediately to avoid stale data.
    """
    try:
        if USE_POSTGRES:
            import psycopg2
            import psycopg2.extras
            conn = psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)
            with conn.cursor() as cur:
                cur.execute(QUERY)
                rows = cur.fetchall()
            conn.close()
            return pd.DataFrame([dict(r) for r in rows])

        else:
            if not SQLITE_PATH.exists():
                return pd.DataFrame()
            import sqlite3
            with sqlite3.connect(str(SQLITE_PATH), check_same_thread=False) as conn:
                return pd.read_sql_query(QUERY, conn, parse_dates=["timestamp"])

    except Exception as e:
        st.error(f"**Database error:** {e}")
        return pd.DataFrame()


# ── Page setup ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SafeSpace AI — Research Dashboard",
    page_icon="🌿",
    layout="wide",
)

st.title("🌿 SafeSpace AI — Research Dashboard")
db_label = "Supabase (PostgreSQL)" if USE_POSTGRES else f"SQLite @ `{SQLITE_PATH}`"
st.caption(f"Database: **{db_label}** · Auto-refreshes every {AUTO_REFRESH_S}s")

# Manual refresh
if st.button("🔄 Refresh now"):
    st.cache_data.clear()
    st.rerun()

# Auto-rerun timer
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()
if time.time() - st.session_state.last_refresh > AUTO_REFRESH_S:
    st.session_state.last_refresh = time.time()
    st.cache_data.clear()
    st.rerun()


# ── Load & validate data ──────────────────────────────────────────────────────

df = load_data()

if df.empty:
    if USE_POSTGRES:
        st.warning(
            "**Supabase is empty or unreachable.**\n\n"
            "Checklist:\n"
            "1. `DATABASE_URL` is set correctly (in `.env` locally, in Render env vars on production)\n"
            "2. The `chat_logs` table exists — it is created automatically when FastAPI starts\n"
            "3. At least one message has been sent through the chatbot\n\n"
            f"Current URL starts with: `{DATABASE_URL[:40]}...`"
        )
    else:
        st.warning(
            f"**No local data found.** Expected SQLite at `{SQLITE_PATH}`.\n\n"
            "Start the FastAPI backend and send a message first."
        )
    st.stop()

# Normalise types
df["timestamp"]       = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
df["crisis_detected"] = df["crisis_detected"].astype(bool)
df["emotion"]         = df["emotion"].fillna("neutral").str.lower()


# ── Sidebar filters ───────────────────────────────────────────────────────────

st.sidebar.header("Filters")
emotions    = sorted(df["emotion"].unique().tolist())
selected    = st.sidebar.multiselect("Emotion", emotions, default=emotions)
crisis_only = st.sidebar.checkbox("Crisis messages only", value=False)

if selected:
    df = df[df["emotion"].isin(selected)]
if crisis_only:
    df = df[df["crisis_detected"]]

if df.empty:
    st.info("No data matches the current filters.")
    st.stop()


# ── KPIs ──────────────────────────────────────────────────────────────────────

top_emotion  = df["emotion"].mode()[0] if len(df) else "—"
avg_latency  = df["response_latency_ms"].mean() if "response_latency_ms" in df else 0
pii_count    = df["pii_entities_found"].notna().sum()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Messages",    len(df))
k2.metric("Unique Sessions",   df["session_id"].nunique())
k3.metric("Crisis Detections", int(df["crisis_detected"].sum()))
k4.metric("Top Emotion",       top_emotion.capitalize())
k5.metric("Avg Latency (ms)",  f"{avg_latency:.0f}" if avg_latency else "—")

st.divider()


# ── Charts ────────────────────────────────────────────────────────────────────

import plotly.express as px

c1, c2 = st.columns(2)

with c1:
    st.subheader("Emotion Distribution")
    ec = df["emotion"].value_counts().reset_index()
    ec.columns = ["emotion", "count"]
    fig = px.pie(
        ec, names="emotion", values="count",
        color="emotion", color_discrete_map=EMOTION_COLOURS, hole=0.4,
    )
    fig.update_traces(textinfo="label+percent")
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("Emotion Over Time")
    tf = df.copy()
    tf["hour"] = tf["timestamp"].dt.floor("h")
    td = tf.groupby(["hour", "emotion"]).size().reset_index(name="count")
    fig2 = px.line(
        td, x="hour", y="count", color="emotion",
        color_discrete_map=EMOTION_COLOURS, markers=True,
    )
    st.plotly_chart(fig2, use_container_width=True)

c3, c4 = st.columns(2)

with c3:
    st.subheader("Messages per Session")
    sd = (
        df.groupby("session_id").size()
        .reset_index(name="messages")
        .sort_values("messages", ascending=False)
        .head(20)
    )
    # Truncate long session IDs for display
    sd["session_short"] = sd["session_id"].str[:8] + "..."
    fig3 = px.bar(
        sd, x="session_short", y="messages",
        color="messages", color_continuous_scale="Teal",
        labels={"session_short": "Session"},
    )
    st.plotly_chart(fig3, use_container_width=True)

with c4:
    st.subheader("Crisis Timeline")
    cd = df[df["crisis_detected"]].copy()
    if cd.empty:
        st.success("No crisis messages in current filter. ✅")
    else:
        cd["date"] = cd["timestamp"].dt.date
        fig4 = px.bar(
            cd.groupby("date").size().reset_index(name="count"),
            x="date", y="count",
            color_discrete_sequence=["#e74c3c"],
            labels={"count": "Crisis Events"},
        )
        st.plotly_chart(fig4, use_container_width=True)


# ── Raw logs table ────────────────────────────────────────────────────────────

st.divider()
st.subheader("📋 Anonymised Chat Logs")

display_cols = ["timestamp", "session_id", "emotion", "crisis_detected",
                "user_message_anon", "ai_response", "response_latency_ms"]
display_cols = [c for c in display_cols if c in df.columns]

st.dataframe(
    df[display_cols].head(200),
    use_container_width=True,
    hide_index=True,
)
st.caption(
    "⚠️ All user messages shown here have been anonymised by Microsoft Presidio before storage. "
    "No real names, phone numbers, emails, or Aadhaar numbers are stored."
)