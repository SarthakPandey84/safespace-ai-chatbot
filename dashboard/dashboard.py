import sys
import os
import logging
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.database import (
    get_all_chat_logs,
    get_all_session_metrics,
    get_emotion_distribution,
    get_daily_activity,
    get_summary_stats,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title            = "SafeSpace — Research Dashboard",
    page_icon             = "🌿",
    layout                = "wide",
    initial_sidebar_state = "expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,600;1,400&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        background-color: #eef6f2 !important;
        font-family: 'DM Sans', sans-serif;
        color: #0a0604 !important;
    }

    .stApp, .block-container {
        background-color: #eef6f2 !important;
    }

    p, span, div, li, label, caption, small, strong, em, a {
        color: #0a0604 !important;
    }

    h1 {
        font-family: 'Lora', serif !important;
        color: #0a0604 !important;
        font-weight: 800 !important;
        font-size: 2rem !important;
    }
    h2 {
        font-family: 'Lora', serif !important;
        color: #0a0604 !important;
        font-weight: 700 !important;
        font-size: 1.5rem !important;
    }
    h3 {
        font-family: 'DM Sans', sans-serif !important;
        color: #0a0604 !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
    }

    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #eef6f2 0%, #e6f2ee 100%);
        border: 1.5px solid #b0d0c4;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        box-shadow: 0 2px 8px rgba(44,36,32,0.10);
    }

    [data-testid="metric-container"] label,
    [data-testid="stMetricLabel"] {
        color: #0a0604 !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
    }

    [data-testid="stMetricValue"] {
        color: #0a0604 !important;
        font-weight: 800 !important;
        font-size: 2rem !important;
    }

    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] span,
    [data-testid="stMarkdownContainer"] li {
        color: #0a0604 !important;
        font-size: 1rem !important;
    }

    .stCaption, [data-testid="stCaptionContainer"] p {
        color: #2e2018 !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #231f1c 0%, #1a1614 100%);
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] small {
        color: #f0e8e0 !important;
    }

    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] strong {
        color: #ffffff !important;
    }

    .privacy-notice {
        background: rgba(26, 80, 120, 0.12);
        border: 1.5px solid rgba(26, 80, 120, 0.3);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        font-size: 0.875rem;
        color: #0a3050 !important;
        font-weight: 600;
        margin-bottom: 1rem;
    }

    .privacy-notice strong {
        color: #0a3050 !important;
    }

    .section-divider {
        border: none;
        border-top: 2px solid #a8c8bc;
        margin: 1.5rem 0;
    }

    [data-testid="stDataFrame"] {
        border-radius: 8px;
        overflow: hidden;
    }

    [data-testid="stDataFrame"] td,
    [data-testid="stDataFrame"] th {
        color: #0a0604 !important;
        font-weight: 500 !important;
    }

    .stSelectbox label,
[data-testid="stDownloadButton"] button {
        background-color: #2d7a3a !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.25rem !important;
        font-size: 0.9rem !important;
    }

    [data-testid="stDownloadButton"] button:hover {
        background-color: #1a5e28 !important;
        color: #ffffff !important;
    }
    .stDateInput label,
    .stCheckbox label {
        color: #f0e8e0 !important;
        font-weight: 600 !important;
    }

    button[kind="secondary"] {
        color: #0a0604 !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=60)
def load_summary_stats() -> dict:
    return get_summary_stats()

@st.cache_data(ttl=60)
def load_emotion_distribution() -> pd.DataFrame:
    data = get_emotion_distribution()
    if not data:
        return pd.DataFrame(columns=["emotion", "count"])
    return pd.DataFrame(data)

@st.cache_data(ttl=60)
def load_daily_activity() -> pd.DataFrame:
    data = get_daily_activity()
    if not data:
        return pd.DataFrame(columns=["date", "message_count"])
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data(ttl=60)
def load_chat_logs(limit: int = 200) -> pd.DataFrame:
    data = get_all_chat_logs(limit=limit)
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if "timestamp" in df.columns:
        df["timestamp"]   = pd.to_datetime(df["timestamp"])
        df["hour_of_day"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.day_name()
    return df

@st.cache_data(ttl=60)
def load_session_metrics() -> pd.DataFrame:
    data = get_all_session_metrics()
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if "session_start" in df.columns:
        df["session_start"] = pd.to_datetime(df["session_start"])
        df["last_active"]   = pd.to_datetime(df["last_active"])
        df["duration_min"]  = (
            df["last_active"] - df["session_start"]
        ).dt.total_seconds() / 60
        df["duration_min"]  = df["duration_min"].round(1)
    return df


CHART_COLORS = {
    "primary":    "#7c9e82",
    "accent":     "#c9956a",
    "muted":      "#a89588",
    "background": "#f0f7f4",
    "grid":       "#b8d4c8",
    "text":       "#0a0604",
}

EMOTION_COLORS = {
    "anxious":     "#c9956a",
    "sad":         "#7a9ec0",
    "angry":       "#c07a7a",
    "hopeful":     "#7ac08a",
    "lonely":      "#9a7ac0",
    "overwhelmed": "#c07a9a",
    "confused":    "#c0b47a",
    "numb":        "#a0a8b0",
    "grateful":    "#c0b47a",
    "fearful":     "#8aa0b8",
    "ashamed":     "#b89080",
    "frustrated":  "#c09060",
    "neutral":     "#a89588",
}

PLOTLY_LAYOUT_DEFAULTS = dict(
    paper_bgcolor = "rgba(0,0,0,0)",
    plot_bgcolor  = CHART_COLORS["background"],
    font          = dict(family="DM Sans", color="#0a0604", size=13),
    margin        = dict(t=40, b=40, l=20, r=20),
    hoverlabel    = dict(bgcolor="white", font_size=13, font_color="#0a0604"),
    legend        = dict(font=dict(color="#0a0604", size=12)),
)


with st.sidebar:
    st.markdown("## 🌿 SafeSpace")
    st.markdown("**Research Analytics Dashboard**")
    st.markdown("---")

    st.markdown("""
    <div class="privacy-notice">
        🔒 <strong>Privacy Notice</strong><br/>
        All data displayed here is anonymized.
        No personally identifiable information (PII) is stored or shown.
        Session IDs are random UUIDs. Use this data for research purposes only.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Controls")

    auto_refresh = st.checkbox(
        "Auto-refresh (60s)",
        value=False,
        help="Automatically reload data from the database every 60 seconds."
    )

    if st.button("🔄 Refresh Data Now", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("### 🔍 Filter")

    emotion_options = [
        "All", "anxious", "sad", "angry", "hopeful", "lonely",
        "overwhelmed", "confused", "numb", "grateful",
        "fearful", "ashamed", "frustrated", "neutral"
    ]
    selected_emotion = st.selectbox(
        "Filter by emotion",
        options = emotion_options,
        index   = 0,
    )

    st.markdown("### 📅 Date Range")
    date_end   = datetime.today().date()
    date_start = date_end - timedelta(days=30)
    start_date = st.date_input("From", value=date_start)
    end_date   = st.date_input("To",   value=date_end)

    st.markdown("---")
    st.caption("SafeSpace AI — B.Tech Final Year Project")
    st.caption("Built with FastAPI · Groq · Presidio · Streamlit")

if auto_refresh:
    import time as time_module
    time_module.sleep(60)
    st.cache_data.clear()
    st.rerun()


st.markdown("# SafeSpace AI — Research Dashboard")
st.markdown(
    "Anonymized behavioral analytics from the SafeSpace empathetic chatbot. "
    "All personally identifiable information has been removed before storage. "
    "Data is intended for academic mental health and HCI research only."
)

st.markdown('<hr class="section-divider"/>', unsafe_allow_html=True)

stats       = load_summary_stats()
df_emotions = load_emotion_distribution()
df_activity = load_daily_activity()
df_logs     = load_chat_logs(limit=500)
df_sessions = load_session_metrics()

has_data = bool(stats.get("total_messages", 0))

if not has_data:
    st.info(
        "📭 **No data yet.** \n\n"
        "The database is empty. Start the SafeSpace chatbot backend and send a few "
        "messages to populate the research dashboard.\n\n"
        "```bash\n"
        "# Terminal 1 — Start the backend:\n"
        "uvicorn backend.main:app --reload --port 8000\n\n"
        "# Terminal 2 — Open the chat:\n"
        "# Navigate to http://localhost:8000\n"
        "```",
        icon="🌿"
    )
    st.stop()


st.markdown("## 📊 Overview")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        label = "💬 Total Sessions",
        value = f"{stats.get('total_sessions', 0):,}",
        help  = "Number of unique anonymous conversation sessions."
    )
with col2:
    st.metric(
        label = "📨 Total Messages",
        value = f"{stats.get('total_messages', 0):,}",
        help  = "Total number of user messages processed."
    )
with col3:
    avg_turns = stats.get('avg_turns_per_session', 0) or 0
    st.metric(
        label = "🔄 Avg Turns / Session",
        value = f"{avg_turns:.1f}",
        help  = "Average number of back-and-forth message pairs per session."
    )
with col4:
    avg_latency = stats.get('avg_latency_ms', 0) or 0
    st.metric(
        label = "⚡ Avg Response Time",
        value = f"{avg_latency:.0f} ms",
        help  = "Average time for the AI to generate a response (milliseconds)."
    )
with col5:
    pii_count = stats.get('total_pii_scrubbed', 0) or 0
    st.metric(
        label = "🔒 PII Instances Scrubbed",
        value = f"{pii_count:,}",
        help  = "Total personal data instances anonymized across all sessions."
    )

st.markdown('<hr class="section-divider"/>', unsafe_allow_html=True)


st.markdown("## 💭 Emotional State Analysis")
st.caption(
    "The AI detects the primary emotional tone of each user message using structured "
    "prompt engineering. This section visualizes the distribution of emotional states "
    "across all recorded sessions — a core behavioral metric for mental health research."
)

if df_emotions.empty:
    st.warning("No emotion data available yet.")
else:
    col_pie, col_bar = st.columns([1, 1.4])

    with col_pie:
        st.markdown("### Emotion Distribution")
        emotion_color_list = [
            EMOTION_COLORS.get(e, CHART_COLORS["muted"])
            for e in df_emotions["emotion"]
        ]
        fig_donut = go.Figure(data=[go.Pie(
            labels        = df_emotions["emotion"].str.capitalize(),
            values        = df_emotions["count"],
            hole          = 0.5,
            marker        = dict(colors=emotion_color_list, line=dict(color="#ffffff", width=2)),
            textinfo      = "label+percent",
            hovertemplate = "<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
        )])
        fig_donut.update_layout(
            **PLOTLY_LAYOUT_DEFAULTS,
            showlegend  = False,
            height      = 340,
            annotations = [dict(
                text      = f"<b>{df_emotions['count'].sum()}</b><br>messages",
                x=0.5, y=0.5, font_size=14, showarrow=False,
                font=dict(color=CHART_COLORS["text"])
            )]
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_bar:
        st.markdown("### Emotion Frequency Ranking")
        df_emotions_sorted = df_emotions.sort_values("count", ascending=True)
        bar_colors = [
            EMOTION_COLORS.get(e, CHART_COLORS["muted"])
            for e in df_emotions_sorted["emotion"]
        ]
        fig_bar = go.Figure(go.Bar(
            x             = df_emotions_sorted["count"],
            y             = df_emotions_sorted["emotion"].str.capitalize(),
            orientation   = "h",
            marker        = dict(color=bar_colors, line=dict(color="rgba(0,0,0,0)")),
            text          = df_emotions_sorted["count"],
            textposition  = "outside",
            hovertemplate = "<b>%{y}</b>: %{x} messages<extra></extra>",
        ))
        fig_bar.update_layout(
            **PLOTLY_LAYOUT_DEFAULTS,
            height = 340,
            xaxis  = dict(showgrid=True, gridcolor=CHART_COLORS["grid"], title="Number of Messages", tickfont=dict(color="#0a0604", size=12), title_font=dict(color="#0a0604", size=13)),
            yaxis  = dict(showgrid=False, tickfont=dict(color="#0a0604", size=12), title_font=dict(color="#0a0604", size=13)),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

st.markdown('<hr class="section-divider"/>', unsafe_allow_html=True)


st.markdown("## 📅 Activity & Temporal Patterns")
st.caption(
    "When are users seeking support? Temporal patterns reveal peak usage times, "
    "which can inform staffing decisions for human counselors and system scaling."
)

col_timeline, col_hourly = st.columns([1.6, 1])

with col_timeline:
    st.markdown("### Daily Message Volume")
    if df_activity.empty:
        st.info("No activity data available.")
    else:
        df_activity_filtered = df_activity[
            (df_activity["date"].dt.date >= start_date) &
            (df_activity["date"].dt.date <= end_date)
        ]
        fig_timeline = go.Figure()
        fig_timeline.add_trace(go.Scatter(
            x             = df_activity_filtered["date"],
            y             = df_activity_filtered["message_count"],
            mode          = "lines+markers",
            name          = "Messages",
            line          = dict(color=CHART_COLORS["primary"], width=2.5, shape="spline"),
            fill          = "tozeroy",
            fillcolor     = "rgba(124, 158, 130, 0.15)",
            marker        = dict(size=6, color=CHART_COLORS["primary"]),
            hovertemplate = "<b>%{x|%b %d, %Y}</b><br>Messages: %{y}<extra></extra>",
        ))
        fig_timeline.update_layout(
            **PLOTLY_LAYOUT_DEFAULTS,
            height = 300,
            xaxis  = dict(showgrid=True, gridcolor=CHART_COLORS["grid"], title="Date", tickfont=dict(color="#0a0604", size=12), title_font=dict(color="#0a0604", size=13)),
            yaxis  = dict(showgrid=True, gridcolor=CHART_COLORS["grid"], title="Message Count", rangemode="tozero", tickfont=dict(color="#0a0604", size=12), title_font=dict(color="#0a0604", size=13)),
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

with col_hourly:
    st.markdown("### Activity by Hour of Day")
    if df_logs.empty or "hour_of_day" not in df_logs.columns:
        st.info("Insufficient data for hourly analysis.")
    else:
        hourly_counts = df_logs.groupby("hour_of_day").size().reset_index(name="count")
        all_hours     = pd.DataFrame({"hour_of_day": range(24)})
        hourly_counts = all_hours.merge(hourly_counts, on="hour_of_day", how="left").fillna(0)
        fig_hourly = go.Figure(go.Bar(
            x             = hourly_counts["hour_of_day"],
            y             = hourly_counts["count"],
            marker        = dict(
                color      = hourly_counts["count"],
                colorscale = [[0, "#f0ebe2"], [0.5, CHART_COLORS["primary"]], [1, "#3d5c42"]],
                showscale  = False,
            ),
            hovertemplate = "Hour %{x}:00 — %{y} messages<extra></extra>",
        ))
        fig_hourly.update_layout(
            **PLOTLY_LAYOUT_DEFAULTS,
            height = 300,
            xaxis  = dict(
                title    = "Hour of Day (24h)",
                tickvals = list(range(0, 24, 3)),
                ticktext = [f"{h:02d}:00" for h in range(0, 24, 3)],
                showgrid = False,
            ),
            yaxis = dict(showgrid=True, gridcolor=CHART_COLORS["grid"], title="Messages", tickfont=dict(color="#0a0604", size=12), title_font=dict(color="#0a0604", size=13)),
        )
        st.plotly_chart(fig_hourly, use_container_width=True)

st.markdown('<hr class="section-divider"/>', unsafe_allow_html=True)


st.markdown("## 🗂️ Session Behaviour Analysis")
st.caption(
    "Session-level metrics reveal how users engage with the platform over time. "
    "Longer sessions and higher turn counts may indicate deeper therapeutic engagement."
)

col_turns, col_pii, col_duration = st.columns(3)

with col_turns:
    st.markdown("### Turns per Session")
    if df_sessions.empty or "total_turns" not in df_sessions.columns:
        st.info("No session data yet.")
    else:
        fig_turns = px.histogram(
            df_sessions,
            x                       = "total_turns",
            nbins                   = 15,
            color_discrete_sequence = [CHART_COLORS["primary"]],
            labels                  = {"total_turns": "Number of Turns"},
            template                = "simple_white",
        )
        fig_turns.update_traces(marker_line_color="white", marker_line_width=1.5)
        fig_turns.update_layout(
            **PLOTLY_LAYOUT_DEFAULTS,
            height     = 280,
            xaxis      = dict(title="Turns per Session", showgrid=False, tickfont=dict(color="#0a0604", size=12), title_font=dict(color="#0a0604", size=13)),
            yaxis      = dict(title="Session Count", showgrid=True, gridcolor=CHART_COLORS["grid"], tickfont=dict(color="#0a0604", size=12), title_font=dict(color="#0a0604", size=13)),
            showlegend = False,
        )
        st.plotly_chart(fig_turns, use_container_width=True)

with col_pii:
    st.markdown("### PII Detections per Session")
    if df_sessions.empty or "total_pii_detected" not in df_sessions.columns:
        st.info("No PII data yet.")
    else:
        def classify_pii(count):
            if count == 0:   return "None (0)"
            elif count <= 2: return "Low (1-2)"
            elif count <= 5: return "Medium (3-5)"
            else:            return "High (6+)"

        df_sessions["pii_risk"] = df_sessions["total_pii_detected"].apply(classify_pii)
        pii_dist = df_sessions["pii_risk"].value_counts().reset_index()
        pii_dist.columns = ["Risk Level", "Sessions"]
        order    = ["None (0)", "Low (1-2)", "Medium (3-5)", "High (6+)"]
        pii_dist["Risk Level"] = pd.Categorical(pii_dist["Risk Level"], categories=order, ordered=True)
        pii_dist = pii_dist.sort_values("Risk Level")
        pii_colors = ["#7ac08a", "#c0c07a", "#c09060", "#c07a7a"]

        fig_pii = go.Figure(go.Bar(
            x             = pii_dist["Risk Level"],
            y             = pii_dist["Sessions"],
            marker_color  = pii_colors[:len(pii_dist)],
            marker_line   = dict(color="white", width=1.5),
            hovertemplate = "<b>%{x}</b>: %{y} sessions<extra></extra>",
        ))
        fig_pii.update_layout(
            **PLOTLY_LAYOUT_DEFAULTS,
            height     = 280,
            xaxis      = dict(title="PII Risk Category", showgrid=False, tickfont=dict(color="#0a0604", size=12), title_font=dict(color="#0a0604", size=13)),
            yaxis      = dict(title="Sessions", showgrid=True, gridcolor=CHART_COLORS["grid"], tickfont=dict(color="#0a0604", size=12), title_font=dict(color="#0a0604", size=13)),
            showlegend = False,
        )
        st.plotly_chart(fig_pii, use_container_width=True)

with col_duration:
    st.markdown("### Session Duration (minutes)")
    if df_sessions.empty or "duration_min" not in df_sessions.columns:
        st.info("No duration data yet.")
    else:
        df_dur  = df_sessions[df_sessions["duration_min"] >= 0]
        fig_dur = px.box(
            df_dur,
            y                       = "duration_min",
            color_discrete_sequence = [CHART_COLORS["accent"]],
            template                = "simple_white",
            points                  = "all",
            labels                  = {"duration_min": "Duration (min)"},
        )
        fig_dur.update_layout(
            **PLOTLY_LAYOUT_DEFAULTS,
            height = 280,
            yaxis  = dict(title="Duration (minutes)", showgrid=True, gridcolor=CHART_COLORS["grid"], tickfont=dict(color="#0a0604", size=12), title_font=dict(color="#0a0604", size=13)),
            xaxis  = dict(showticklabels=False, tickfont=dict(color="#0a0604", size=12), title_font=dict(color="#0a0604", size=13)),
        )
        st.plotly_chart(fig_dur, use_container_width=True)

st.markdown('<hr class="section-divider"/>', unsafe_allow_html=True)


st.markdown("## ⚡ System Performance Analysis")
st.caption(
    "AI response latency impacts user experience, especially for users in emotional "
    "distress who may interpret slow responses as disengagement."
)

if not df_logs.empty and "response_latency_ms" in df_logs.columns:
    df_latency = df_logs.dropna(subset=["response_latency_ms"]).copy()

    if not df_latency.empty:
        col_lat_box, col_lat_time = st.columns([1, 2])

        with col_lat_box:
            st.markdown("### Latency Distribution")
            p50 = df_latency["response_latency_ms"].quantile(0.50)
            p95 = df_latency["response_latency_ms"].quantile(0.95)
            p99 = df_latency["response_latency_ms"].quantile(0.99)
            st.metric("Median (P50)",   f"{p50:.0f} ms")
            st.metric("95th Pct (P95)", f"{p95:.0f} ms", help="95% of responses are faster than this.")
            st.metric("99th Pct (P99)", f"{p99:.0f} ms", help="The slowest 1% of responses take this long.")

        with col_lat_time:
            st.markdown("### Latency Over Time")
            fig_lat = go.Figure()
            fig_lat.add_trace(go.Scatter(
                x             = df_latency["timestamp"],
                y             = df_latency["response_latency_ms"],
                mode          = "markers",
                name          = "Response Time",
                marker        = dict(
                    color      = df_latency["response_latency_ms"],
                    colorscale = [[0, CHART_COLORS["primary"]], [1, CHART_COLORS["accent"]]],
                    size=6, opacity=0.7, showscale=True,
                    colorbar=dict(title="ms", thickness=12),
                ),
                hovertemplate = "Time: %{x}<br>Latency: %{y}ms<extra></extra>",
            ))
            if len(df_latency) >= 5:
                df_latency_sorted = df_latency.sort_values("timestamp")
                rolling_avg       = df_latency_sorted["response_latency_ms"].rolling(window=5, min_periods=1).mean()
                fig_lat.add_trace(go.Scatter(
                    x             = df_latency_sorted["timestamp"],
                    y             = rolling_avg,
                    mode          = "lines",
                    name          = "5-point Moving Avg",
                    line          = dict(color=CHART_COLORS["accent"], width=2, dash="dash"),
                    hovertemplate = "Avg: %{y:.0f}ms<extra></extra>",
                ))
            fig_lat.update_layout(
                **PLOTLY_LAYOUT_DEFAULTS,
                height = 260,
                xaxis  = dict(title="Time", showgrid=True, gridcolor=CHART_COLORS["grid"], tickfont=dict(color="#0a0604", size=12), title_font=dict(color="#0a0604", size=13)),
                yaxis  = dict(title="Latency (ms)", showgrid=True, gridcolor=CHART_COLORS["grid"], tickfont=dict(color="#0a0604", size=12), title_font=dict(color="#0a0604", size=13)),
            )
            st.plotly_chart(fig_lat, use_container_width=True)

st.markdown('<hr class="section-divider"/>', unsafe_allow_html=True)


st.markdown("## 🗃️ Anonymized Message Log")
st.caption(
    "Raw (anonymized) message records for qualitative research review. "
    "Personal details have been replaced with entity-type placeholders "
    "(e.g., `<PERSON>`, `<PHONE_NUMBER>`) by the Presidio anonymization pipeline."
)

st.markdown("""
<div class="privacy-notice">
    ⚠️ <strong>Researcher Reminder:</strong> The data below contains anonymized
    messages. Even though PII has been removed, treat this data with the same
    ethical care as personal data — do not share screenshots or export files
    without appropriate institutional oversight.
</div>
""", unsafe_allow_html=True)

if df_logs.empty:
    st.info("No message logs available.")
else:
    df_display = df_logs.copy()

    if selected_emotion != "All":
        df_display = df_display[df_display["detected_emotion"] == selected_emotion]

    if "timestamp" in df_display.columns:
        df_display = df_display[
            (df_display["timestamp"].dt.date >= start_date) &
            (df_display["timestamp"].dt.date <= end_date)
        ]

    display_columns = {
        "timestamp":              "Timestamp",
        "session_id":             "Session ID",
        "user_message_anonymized":"User Message (Anonymized)",
        "ai_response":            "AI Response",
        "detected_emotion":       "Emotion",
        "pii_entities_found":     "PII Entities Found",
        "response_latency_ms":    "Latency (ms)",
    }

    available_cols = [c for c in display_columns.keys() if c in df_display.columns]
    df_display     = df_display[available_cols].rename(columns=display_columns)

    if "Session ID" in df_display.columns:
        df_display["Session ID"] = df_display["Session ID"].apply(
            lambda x: f"{str(x)[:8]}..." if pd.notna(x) else x
        )

    st.markdown(f"**Showing {len(df_display):,} records** (filtered from {len(df_logs):,} total)")

    st.dataframe(
        df_display,
        use_container_width = True,
        height              = 400,
        column_config       = {
            "Timestamp": st.column_config.DatetimeColumn("Timestamp", format="DD/MM/YY HH:mm"),
            "Latency (ms)": st.column_config.NumberColumn("Latency (ms)", format="%d ms"),
            "User Message (Anonymized)": st.column_config.TextColumn("User Message (Anonymized)", width="large"),
            "AI Response": st.column_config.TextColumn("AI Response", width="large"),
        },
        hide_index = True,
    )

    csv_data = df_display.to_csv(index=False).encode("utf-8")
    st.download_button(
        label     = "⬇️ Export filtered data as CSV",
        data      = csv_data,
        file_name = f"safespace_research_data_{datetime.today().strftime('%Y%m%d')}.csv",
        mime      = "text/csv",
        help      = "Downloads the currently filtered and displayed data as a CSV file."
    )

st.markdown('<hr class="section-divider"/>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#a89588; font-size:0.8rem; padding: 1rem 0;">
    SafeSpace AI · B.Tech Final Year Project (AI & Data Science) · Research Dashboard v1.0<br/>
    Built with <strong>FastAPI</strong> · <strong>Groq (Llama 3.3 70B)</strong> ·
    <strong>Microsoft Presidio</strong> · <strong>Streamlit</strong> · <strong>SQLite</strong><br/>
    <em>All data is anonymized. Privacy by Design.</em>
</div>
""", unsafe_allow_html=True)