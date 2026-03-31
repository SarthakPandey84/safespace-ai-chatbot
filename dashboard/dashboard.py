"""
dashboard/dashboard.py
SafeSpace AI — Research Analytics Dashboard
Run: streamlit run dashboard/dashboard.py
"""

import os
import sqlite3
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# ── Constants ─────────────────────────────────────────────────────────────────

# IMPORTANT: This path is resolved relative to the project root, not this file.
# If you run streamlit from a different directory, set DB_PATH to an absolute path.
DB_PATH = Path(__file__).parent.parent / "data" / "safespace.db"

EMOTION_COLOURS = {
    "anxious":     "#f39c12",
    "sad":         "#3498db",
    "angry":       "#e74c3c",
    "hopeful":     "#2ecc71",
    "lonely":      "#9b59b6",
    "overwhelmed": "#e67e22",
    "confused":    "#1abc9c",
    "numb":        "#95a5a6",
    "grateful":    "#27ae60",
    "fearful":     "#c0392b",
    "ashamed":     "#8e44ad",
    "frustrated":  "#d35400",
    "neutral":     "#7f8c8d",
}

AUTO_REFRESH_SECONDS = 10   # dashboard polls the DB every N seconds


# ── Data Loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=AUTO_REFRESH_SECONDS)   # KEY FIX: TTL forces re-query every N seconds
def load_data() -> pd.DataFrame:
    """
    Read all chat logs from SQLite.

    FIX NOTES:
    - check_same_thread=False is required when SQLite is opened outside the
      thread that created the connection (Streamlit reruns in a fresh thread).
    - We use a context manager so the connection is closed immediately, which
      prevents stale data from a cached connection object.
    - ttl=AUTO_REFRESH_SECONDS on @st.cache_data forces Streamlit to re-run
      this function periodically so the dashboard shows new rows.
    """
    if not DB_PATH.exists():
        return pd.DataFrame()

    try:
        with sqlite3.connect(str(DB_PATH), check_same_thread=False) as conn:
            df = pd.read_sql_query(
                """
                SELECT
                    id,
                    session_id,
                    timestamp,
                    user_message_anon   AS user_message,
                    ai_response,
                    emotion,
                    crisis_detected
                FROM chat_logs
                ORDER BY timestamp DESC
                """,
                conn,
                parse_dates=["timestamp"],
            )
        return df
    except Exception as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame()


# ── Page Setup ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SafeSpace AI — Research Dashboard",
    page_icon="🌿",
    layout="wide",
)

st.title("🌿 SafeSpace AI — Research Dashboard")
st.caption(
    f"Showing anonymised data only. Auto-refreshes every {AUTO_REFRESH_SECONDS}s. "
    f"Database: `{DB_PATH}`"
)

# Manual refresh button + auto-rerun timer
col_refresh, col_status = st.columns([1, 4])
with col_refresh:
    if st.button("🔄 Refresh now"):
        st.cache_data.clear()
        st.rerun()

# Schedule an automatic rerun so the dashboard stays live
# (Streamlit will rerun the script after the sleep)
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()

if time.time() - st.session_state.last_refresh > AUTO_REFRESH_SECONDS:
    st.session_state.last_refresh = time.time()
    st.cache_data.clear()
    st.rerun()

# ── Load Data ─────────────────────────────────────────────────────────────────

df = load_data()

if df.empty:
    st.warning(
        "No data found. Make sure:\n"
        "1. The FastAPI backend is running (`uvicorn backend.main:app --reload --port 8000`)\n"
        "2. You have sent at least one message through the chatbot\n"
        f"3. The database file exists at: `{DB_PATH}`"
    )
    st.stop()

# ── Sidebar Filters ───────────────────────────────────────────────────────────

st.sidebar.header("Filters")

all_emotions = sorted(df["emotion"].dropna().unique().tolist())
selected_emotions = st.sidebar.multiselect(
    "Emotion filter",
    options=all_emotions,
    default=all_emotions,
)

show_crisis = st.sidebar.checkbox("Crisis messages only", value=False)

if selected_emotions:
    df = df[df["emotion"].isin(selected_emotions)]

if show_crisis:
    df = df[df["crisis_detected"] == 1]

# ── KPI Row ───────────────────────────────────────────────────────────────────

total_msgs      = len(df)
unique_sessions = df["session_id"].nunique()
crisis_count    = int(df["crisis_detected"].sum()) if "crisis_detected" in df.columns else 0
top_emotion     = df["emotion"].mode()[0] if not df["emotion"].isna().all() else "—"

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Messages",    total_msgs)
k2.metric("Unique Sessions",   unique_sessions)
k3.metric("Crisis Detections", crisis_count)
k4.metric("Most Common Emotion", top_emotion.capitalize())

st.divider()

# ── Charts ────────────────────────────────────────────────────────────────────

row1_left, row1_right = st.columns(2)

with row1_left:
    st.subheader("Emotion Distribution")
    emotion_counts = (
        df["emotion"]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "emotion", "emotion": "count", "count": "count"})
    )
    # pandas ≥2 value_counts() returns df with original col name
    if "emotion" in emotion_counts.columns and emotion_counts.columns.tolist() == ["emotion", "count"]:
        pass
    else:
        emotion_counts.columns = ["emotion", "count"]

    fig_pie = px.pie(
        emotion_counts,
        names="emotion",
        values="count",
        color="emotion",
        color_discrete_map=EMOTION_COLOURS,
        hole=0.4,
    )
    fig_pie.update_traces(textinfo="label+percent")
    st.plotly_chart(fig_pie, use_container_width=True)


with row1_right:
    st.subheader("Emotion Over Time")
    if "timestamp" in df.columns and not df["timestamp"].isna().all():
        df_time = df.copy()
        df_time["hour"] = df_time["timestamp"].dt.floor("h")
        time_emotion = (
            df_time.groupby(["hour", "emotion"])
            .size()
            .reset_index(name="count")
        )
        fig_line = px.line(
            time_emotion,
            x="hour",
            y="count",
            color="emotion",
            color_discrete_map=EMOTION_COLOURS,
            markers=True,
        )
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("Timestamp data not available.")


row2_left, row2_right = st.columns(2)

with row2_left:
    st.subheader("Messages per Session")
    session_counts = (
        df.groupby("session_id")
        .size()
        .reset_index(name="messages")
        .sort_values("messages", ascending=False)
        .head(20)
    )
    fig_bar = px.bar(
        session_counts,
        x="session_id",
        y="messages",
        labels={"session_id": "Session ID (truncated)", "messages": "Message Count"},
        color="messages",
        color_continuous_scale="Teal",
    )
    fig_bar.update_xaxes(tickangle=45, tickfont_size=9)
    st.plotly_chart(fig_bar, use_container_width=True)


with row2_right:
    st.subheader("Crisis Detection Timeline")
    if "crisis_detected" in df.columns:
        crisis_df = df[df["crisis_detected"] == 1].copy()
        if crisis_df.empty:
            st.success("No crisis messages detected in selected filters.")
        else:
            crisis_df["date"] = crisis_df["timestamp"].dt.date
            crisis_by_day = crisis_df.groupby("date").size().reset_index(name="count")
            fig_crisis = px.bar(
                crisis_by_day,
                x="date",
                y="count",
                color_discrete_sequence=["#e74c3c"],
                labels={"date": "Date", "count": "Crisis Events"},
            )
            st.plotly_chart(fig_crisis, use_container_width=True)
    else:
        st.info("crisis_detected column not found in database.")


# ── Raw Logs Table ────────────────────────────────────────────────────────────

st.divider()
st.subheader("📋 Anonymised Chat Logs")
st.dataframe(
    df[["timestamp", "session_id", "emotion", "crisis_detected", "user_message", "ai_response"]]
    .head(200),
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "⚠️  All user messages shown here have been anonymised by Microsoft Presidio. "
    "No real names, phone numbers, or emails are stored."
)