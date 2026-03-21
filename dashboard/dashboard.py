# ==============================================================================
# FILE: dashboard/dashboard.py
# PROJECT: SafeSpace AI - Empathetic Privacy-First Chatbot
# PURPOSE: A Streamlit-based research visualization dashboard that reads
#          anonymized behavioral data from the SQLite database and renders
#          interactive charts for academic analysis.
#
# HOW TO RUN (from the project root directory):
#   streamlit run dashboard/dashboard.py
#
# ARCHITECTURAL POSITION:
#   This dashboard is a COMPLETELY SEPARATE PROCESS from the FastAPI backend.
#   It reads directly from the SQLite database file and has NO dependency on
#   main.py, ai_engine.py, or privacy_engine.py. This decoupling means:
#     1. The dashboard works even if the chatbot server is offline.
#     2. A researcher can analyze data without running the full application.
#     3. The dashboard cannot accidentally trigger AI calls or modify user data.
#
#   [SafeSpace Backend] ──writes──► [SQLite DB] ◄──reads── [This Dashboard]
#
# PRIVACY GUARANTEE AT THIS LAYER:
#   This file ONLY calls database functions that expose anonymized columns.
#   The `user_message_raw` column is never queried anywhere in this file.
#   Even if a researcher has full access to this dashboard, they cannot
#   retrieve any PII — the architectural guarantee holds at every layer.
#
# ACADEMIC NOTE — Why Streamlit for the Research Dashboard?
#   Streamlit was chosen over alternatives (Dash, Grafana, custom React) for
#   three reasons that are directly relevant to an academic context:
#     1. SPEED: A full interactive dashboard in ~200 lines of pure Python.
#     2. REPRODUCIBILITY: Researchers can read and modify the dashboard logic
#        without frontend development skills — critical for academic reuse.
#     3. PYTHON-NATIVE: Data flows directly from Pandas DataFrames into charts,
#        with no serialization/deserialization layer between analysis and display.
# ==============================================================================

import sys
import os
import logging
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------------------------------------------------------
# PATH SETUP
# ACADEMIC NOTE — Why sys.path manipulation?
#   Streamlit runs dashboard.py as a standalone script from the dashboard/
#   directory. This means Python's module search path doesn't include the
#   project root, so `from backend.database import ...` would fail with a
#   ModuleNotFoundError.
#   We add the project root to sys.path explicitly to make the backend
#   package importable. This is the correct approach for a monorepo
#   project structure where multiple entry points share common modules.
# ------------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Now we can import from the backend package.
from backend.database import (
    get_all_chat_logs,
    get_all_session_metrics,
    get_emotion_distribution,
    get_daily_activity,
    get_summary_stats,
)

# ------------------------------------------------------------------------------
# LOGGING
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==============================================================================
# SECTION 1: STREAMLIT PAGE CONFIGURATION
# ==============================================================================
# st.set_page_config() MUST be the first Streamlit call in the script.
# ACADEMIC NOTE: This is a Streamlit architectural constraint — the page
# config sets browser-level properties (title, favicon, layout) that must
# be defined before any rendering occurs, similar to HTML's <head> section.

st.set_page_config(
    page_title     = "SafeSpace — Research Dashboard",
    page_icon      = "🌿",
    layout         = "wide",           # Use full browser width for data density
    initial_sidebar_state = "expanded",
)


# ==============================================================================
# SECTION 2: CUSTOM CSS THEMING
# ==============================================================================
# Streamlit supports injecting custom CSS via st.markdown with unsafe_allow_html.
# We use this to align the dashboard's visual language with the chat frontend:
# warm tones, clean typography, and a professional research aesthetic.

st.markdown("""
<style>
    /* Import the same font family as the chat frontend for visual consistency */
    @import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,600;1,400&family=DM+Sans:wght@300;400;500&display=swap');

    /* Global font override */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* Metric card styling — warm academic feel */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #f7f3ee 0%, #f0ebe2 100%);
        border: 1px solid #e0d8d0;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        box-shadow: 0 2px 8px rgba(44,36,32,0.06);
    }

    /* Dashboard title font */
    h1 { font-family: 'Lora', serif !important; color: #2c2420 !important; }
    h2 { font-family: 'Lora', serif !important; color: #3d3530 !important; }
    h3 { font-family: 'DM Sans', sans-serif !important; color: #4a3f39 !important; }

    /* Sidebar background */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #231f1c 0%, #1a1614 100%);
    }

    /* Privacy notice box */
    .privacy-notice {
        background: rgba(58, 107, 139, 0.08);
        border: 1px solid rgba(58, 107, 139, 0.2);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        font-size: 0.85rem;
        color: #4a6b7c;
        margin-bottom: 1rem;
    }

    /* Section divider */
    .section-divider {
        border: none;
        border-top: 1px solid #e0d8d0;
        margin: 1.5rem 0;
    }

    /* Data table styling */
    [data-testid="stDataFrame"] {
        border-radius: 8px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# SECTION 3: DATA LOADING WITH CACHING
# ==============================================================================

# ACADEMIC NOTE — Streamlit Caching (@st.cache_data):
#   Streamlit re-runs the ENTIRE script on every user interaction (a slider
#   move, a button click, etc.). Without caching, every interaction would
#   re-query the database — wasteful and slow for large datasets.
#
#   @st.cache_data caches the function's return value using its arguments
#   as the cache key. The `ttl` (Time To Live) parameter defines how long
#   the cache is valid before the next call hits the database again.
#
#   ttl=60 means: return cached data for up to 60 seconds, then refresh.
#   This is a trade-off between dashboard freshness and database load.
#   A 60-second delay is acceptable for a research dashboard (vs. a
#   real-time operational dashboard which would need ttl=5 or no cache).

@st.cache_data(ttl=60)
def load_summary_stats() -> dict:
    """Load high-level summary statistics. Cached for 60 seconds."""
    return get_summary_stats()

@st.cache_data(ttl=60)
def load_emotion_distribution() -> pd.DataFrame:
    """Load emotion distribution data as a Pandas DataFrame."""
    data = get_emotion_distribution()
    if not data:
        return pd.DataFrame(columns=["emotion", "count"])
    return pd.DataFrame(data)

@st.cache_data(ttl=60)
def load_daily_activity() -> pd.DataFrame:
    """Load daily message activity as a Pandas DataFrame."""
    data = get_daily_activity()
    if not data:
        return pd.DataFrame(columns=["date", "message_count"])
    df = pd.DataFrame(data)
    # Convert date strings to datetime objects for proper time-series plotting.
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data(ttl=60)
def load_chat_logs(limit: int = 200) -> pd.DataFrame:
    """Load anonymized chat logs as a Pandas DataFrame."""
    data = get_all_chat_logs(limit=limit)
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    # Parse timestamp column for time-based analysis.
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        # Extract hour of day for activity pattern analysis.
        df["hour_of_day"] = df["timestamp"].dt.hour
        # Extract day of week for weekly pattern analysis.
        df["day_of_week"] = df["timestamp"].dt.day_name()
    return df

@st.cache_data(ttl=60)
def load_session_metrics() -> pd.DataFrame:
    """Load per-session aggregated metrics as a Pandas DataFrame."""
    data = get_all_session_metrics()
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if "session_start" in df.columns:
        df["session_start"] = pd.to_datetime(df["session_start"])
        df["last_active"]   = pd.to_datetime(df["last_active"])
        # Calculate session duration in minutes.
        df["duration_min"]  = (
            df["last_active"] - df["session_start"]
        ).dt.total_seconds() / 60
        df["duration_min"]  = df["duration_min"].round(1)
    return df


# ==============================================================================
# SECTION 4: PLOTLY CHART THEME
# ==============================================================================
# Centralized chart configuration for visual consistency across all charts.
# ACADEMIC NOTE: Using a consistent color palette across all visualizations
# is a data visualization best practice (from Tufte's principles of graphical
# excellence). It prevents the "chart junk" problem where inconsistent styling
# distracts from the data itself.

CHART_COLORS = {
    "primary":    "#7c9e82",   # Sage green — main data series
    "accent":     "#c9956a",   # Warm amber — highlights
    "muted":      "#a89588",   # Muted brown — secondary series
    "background": "#f7f3ee",   # Warm parchment — chart backgrounds
    "grid":       "#e0d8d0",   # Soft warm grid lines
    "text":       "#4a3f39",   # Warm dark text
}

# Emotion-to-color mapping for consistent visual encoding across charts.
# ACADEMIC NOTE — Pre-attentive attributes in data visualization:
#   Color is a "pre-attentive attribute" — the brain processes it before
#   conscious attention. Consistently mapping 'sad' to blue and 'anxious'
#   to orange across ALL charts means viewers can cross-reference charts
#   without reading legends on every single one. This reduces cognitive load.
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
    paper_bgcolor = "rgba(0,0,0,0)",   # Transparent — inherits Streamlit bg
    plot_bgcolor  = CHART_COLORS["background"],
    font          = dict(family="DM Sans", color=CHART_COLORS["text"]),
    margin        = dict(t=40, b=40, l=20, r=20),
    hoverlabel    = dict(bgcolor="white", font_size=13),
)


# ==============================================================================
# SECTION 5: SIDEBAR
# ==============================================================================

with st.sidebar:
    st.markdown("## 🌿 SafeSpace")
    st.markdown("**Research Analytics Dashboard**")
    st.markdown("---")

    # Privacy Reminder — displayed prominently in the sidebar.
    # ACADEMIC NOTE: Displaying a privacy reminder to dashboard users (researchers)
    # serves an ethical purpose: it reinforces that the data they are viewing
    # is anonymized and should be treated as research data, not personal records.
    st.markdown("""
    <div class="privacy-notice">
        🔒 <strong>Privacy Notice</strong><br/>
        All data displayed here is anonymized.
        No personally identifiable information (PII) is stored or shown.
        Session IDs are random UUIDs. Use this data for research purposes only.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Controls")

    # Auto-refresh toggle.
    auto_refresh = st.checkbox(
        "Auto-refresh (60s)",
        value=False,
        help="Automatically reload data from the database every 60 seconds."
    )

    # Manual refresh button.
    if st.button("🔄 Refresh Data Now", use_container_width=True):
        # Clear all cached data to force a fresh database read on next access.
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")

    # Emotion filter for the data table.
    st.markdown("### 🔍 Filter")
    emotion_options = [
        "All", "anxious", "sad", "angry", "hopeful", "lonely",
        "overwhelmed", "confused", "numb", "grateful",
        "fearful", "ashamed", "frustrated", "neutral"
    ]
    selected_emotion = st.selectbox(
        "Filter by emotion",
        options   = emotion_options,
        index     = 0,
        help      = "Filter the data table below to show only messages with this detected emotion."
    )

    # Date range filter.
    st.markdown("### 📅 Date Range")
    date_end   = datetime.today().date()
    date_start = date_end - timedelta(days=30)
    start_date = st.date_input("From", value=date_start)
    end_date   = st.date_input("To",   value=date_end)

    st.markdown("---")
    st.caption("SafeSpace AI — B.Tech Final Year Project")
    st.caption("Built with FastAPI · Gemini · Presidio · Streamlit")

# Auto-refresh implementation.
# ACADEMIC NOTE: st.rerun() re-executes the entire Streamlit script from the top,
# simulating a page refresh. Combined with the cache TTL of 60 seconds, this
# implements a polling-based auto-refresh mechanism without WebSockets.
if auto_refresh:
    import time as time_module
    time_module.sleep(60)
    st.cache_data.clear()
    st.rerun()


# ==============================================================================
# SECTION 6: MAIN DASHBOARD LAYOUT
# ==============================================================================

# --- Page Title & Description ---
st.markdown("# SafeSpace AI — Research Dashboard")
st.markdown(
    "Anonymized behavioral analytics from the SafeSpace empathetic chatbot. "
    "All personally identifiable information has been removed before storage. "
    "Data is intended for academic mental health and HCI research only."
)

st.markdown('<hr class="section-divider"/>', unsafe_allow_html=True)

# Load all data.
stats         = load_summary_stats()
df_emotions   = load_emotion_distribution()
df_activity   = load_daily_activity()
df_logs       = load_chat_logs(limit=500)
df_sessions   = load_session_metrics()

# Check if there is any data to display at all.
has_data = bool(stats.get("total_messages", 0))

if not has_data:
    # --- Empty State ---
    # Show a helpful onboarding message instead of empty charts.
    # ACADEMIC NOTE: Empty state design is a UI principle stating that
    # a dashboard with no data should guide the user toward generating
    # data, not show broken/empty charts. This is especially important
    # during a live demo — the evaluator should understand how to proceed.
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
    st.stop()  # Halt rendering — no point rendering empty charts.


# ==============================================================================
# SECTION 7: SUMMARY METRIC CARDS
# ==============================================================================
# ACADEMIC NOTE — Dashboard Information Hierarchy:
#   We follow the "inverted pyramid" journalism principle for dashboard layout:
#   most important/aggregated information at the TOP (metric cards),
#   detailed breakdowns in the MIDDLE (charts),
#   raw data at the BOTTOM (data table).
#   This lets a viewer get the key insights with a 5-second glance at the top,
#   and drill into details if they want more.

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
        help  = "Total personal data instances anonymized across all sessions. "
                "This represents the privacy protection delivered by the system."
    )

st.markdown('<hr class="section-divider"/>', unsafe_allow_html=True)


# ==============================================================================
# SECTION 8: EMOTION ANALYSIS CHARTS
# ==============================================================================

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

    # --- Donut Chart: Emotion Distribution ---
    with col_pie:
        st.markdown("### Emotion Distribution")

        # Map each emotion to its designated color for visual consistency.
        emotion_color_list = [
            EMOTION_COLORS.get(e, CHART_COLORS["muted"])
            for e in df_emotions["emotion"]
        ]

        fig_donut = go.Figure(data=[go.Pie(
            labels    = df_emotions["emotion"].str.capitalize(),
            values    = df_emotions["count"],
            hole      = 0.5,           # Creates a donut chart (hole in center)
            marker    = dict(colors=emotion_color_list,
                             line=dict(color="#ffffff", width=2)),
            textinfo  = "label+percent",
            hovertemplate = "<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
        )])

        fig_donut.update_layout(
            **PLOTLY_LAYOUT_DEFAULTS,
            showlegend = False,
            height     = 340,
            # Add annotation in the center hole of the donut.
            annotations=[dict(
                text      = f"<b>{df_emotions['count'].sum()}</b><br>messages",
                x         = 0.5, y = 0.5,
                font_size = 14,
                showarrow = False,
                font      = dict(color=CHART_COLORS["text"])
            )]
        )
        st.plotly_chart(fig_donut, use_container_width=True)

        # ACADEMIC NOTE — Donut vs Pie Chart:
        #   Donut charts reduce the "wedge area" visual bias that affects pie charts
        #   (humans overestimate the area of wider-angle wedges). The center hole
        #   forces the eye to compare arc lengths instead of areas, which is
        #   a more perceptually accurate comparison. This is a data visualization
        #   research finding from Cleveland & McGill (1984).

    # --- Horizontal Bar Chart: Ranked Emotions ---
    with col_bar:
        st.markdown("### Emotion Frequency Ranking")

        df_emotions_sorted = df_emotions.sort_values("count", ascending=True)
        bar_colors = [
            EMOTION_COLORS.get(e, CHART_COLORS["muted"])
            for e in df_emotions_sorted["emotion"]
        ]

        fig_bar = go.Figure(go.Bar(
            x           = df_emotions_sorted["count"],
            y           = df_emotions_sorted["emotion"].str.capitalize(),
            orientation = "h",         # Horizontal bar — better for labeled categories
            marker      = dict(color=bar_colors, line=dict(color="rgba(0,0,0,0)")),
            text        = df_emotions_sorted["count"],
            textposition= "outside",
            hovertemplate = "<b>%{y}</b>: %{x} messages<extra></extra>",
        ))

        fig_bar.update_layout(
            **PLOTLY_LAYOUT_DEFAULTS,
            height    = 340,
            xaxis     = dict(
                showgrid=True, gridcolor=CHART_COLORS["grid"],
                title="Number of Messages"
            ),
            yaxis     = dict(showgrid=False),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

st.markdown('<hr class="section-divider"/>', unsafe_allow_html=True)


# ==============================================================================
# SECTION 9: TEMPORAL ACTIVITY ANALYSIS
# ==============================================================================

st.markdown("## 📅 Activity & Temporal Patterns")
st.caption(
    "When are users seeking support? Temporal patterns reveal peak usage times, "
    "which can inform staffing decisions for human counselors and system scaling."
)

col_timeline, col_hourly = st.columns([1.6, 1])

# --- Daily Activity Timeline ---
with col_timeline:
    st.markdown("### Daily Message Volume")

    if df_activity.empty:
        st.info("No activity data available.")
    else:
        # Apply date filter from the sidebar.
        df_activity_filtered = df_activity[
            (df_activity["date"].dt.date >= start_date) &
            (df_activity["date"].dt.date <= end_date)
        ]

        fig_timeline = go.Figure()

        # Area chart with gradient fill for visual depth.
        fig_timeline.add_trace(go.Scatter(
            x          = df_activity_filtered["date"],
            y          = df_activity_filtered["message_count"],
            mode       = "lines+markers",
            name       = "Messages",
            line       = dict(color=CHART_COLORS["primary"], width=2.5, shape="spline"),
            fill       = "tozeroy",
            fillcolor  = "rgba(124, 158, 130, 0.15)",
            marker     = dict(size=6, color=CHART_COLORS["primary"]),
            hovertemplate = "<b>%{x|%b %d, %Y}</b><br>Messages: %{y}<extra></extra>",
        ))

        fig_timeline.update_layout(
            **PLOTLY_LAYOUT_DEFAULTS,
            height = 300,
            xaxis  = dict(
                showgrid=True, gridcolor=CHART_COLORS["grid"],
                title="Date"
            ),
            yaxis  = dict(
                showgrid=True, gridcolor=CHART_COLORS["grid"],
                title="Message Count", rangemode="tozero"
            ),
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

# --- Hourly Activity Heatmap (Hour of Day) ---
with col_hourly:
    st.markdown("### Activity by Hour of Day")

    if df_logs.empty or "hour_of_day" not in df_logs.columns:
        st.info("Insufficient data for hourly analysis.")
    else:
        hourly_counts = (
            df_logs.groupby("hour_of_day")
            .size()
            .reset_index(name="count")
        )

        # Fill missing hours with 0 (no activity ≠ missing data — it's a valid 0).
        all_hours = pd.DataFrame({"hour_of_day": range(24)})
        hourly_counts = all_hours.merge(hourly_counts, on="hour_of_day", how="left").fillna(0)

        fig_hourly = go.Figure(go.Bar(
            x = hourly_counts["hour_of_day"],
            y = hourly_counts["count"],
            marker = dict(
                color    = hourly_counts["count"],
                colorscale = [[0, "#f0ebe2"], [0.5, CHART_COLORS["primary"]], [1, "#3d5c42"]],
                showscale= False,
            ),
            hovertemplate = "Hour %{x}:00 — %{y} messages<extra></extra>",
        ))

        fig_hourly.update_layout(
            **PLOTLY_LAYOUT_DEFAULTS,
            height = 300,
            xaxis  = dict(
                title="Hour of Day (24h)",
                tickvals=list(range(0, 24, 3)),
                ticktext=[f"{h:02d}:00" for h in range(0, 24, 3)],
                showgrid=False,
            ),
            yaxis  = dict(
                showgrid=True, gridcolor=CHART_COLORS["grid"],
                title="Messages"
            ),
        )
        st.plotly_chart(fig_hourly, use_container_width=True)

st.markdown('<hr class="section-divider"/>', unsafe_allow_html=True)


# ==============================================================================
# SECTION 10: SESSION BEHAVIOUR ANALYSIS
# ==============================================================================

st.markdown("## 🗂️ Session Behaviour Analysis")
st.caption(
    "Session-level metrics reveal how users engage with the platform over time. "
    "Longer sessions and higher turn counts may indicate deeper therapeutic engagement."
)

col_turns, col_pii, col_duration = st.columns(3)

# --- Session Turn Count Distribution ---
with col_turns:
    st.markdown("### Turns per Session")

    if df_sessions.empty or "total_turns" not in df_sessions.columns:
        st.info("No session data yet.")
    else:
        fig_turns = px.histogram(
            df_sessions,
            x         = "total_turns",
            nbins     = 15,
            color_discrete_sequence = [CHART_COLORS["primary"]],
            labels    = {"total_turns": "Number of Turns"},
            template  = "simple_white",
        )
        fig_turns.update_traces(marker_line_color="white", marker_line_width=1.5)
        fig_turns.update_layout(
            **PLOTLY_LAYOUT_DEFAULTS,
            height = 280,
            xaxis  = dict(title="Turns per Session", showgrid=False),
            yaxis  = dict(title="Session Count", showgrid=True, gridcolor=CHART_COLORS["grid"]),
            showlegend = False,
        )
        st.plotly_chart(fig_turns, use_container_width=True)

        # ACADEMIC NOTE — Histogram bin choice:
        #   nbins=15 is a reasonable starting point. For small datasets, fewer
        #   bins prevent over-fragmentation (Sturges' rule: k ≈ log2(n) + 1).
        #   For large datasets, more bins reveal distribution shape.
        #   In production, we'd implement dynamic bin count based on dataset size.

# --- PII Detection Rate per Session ---
with col_pii:
    st.markdown("### PII Detections per Session")

    if df_sessions.empty or "total_pii_detected" not in df_sessions.columns:
        st.info("No PII data yet.")
    else:
        # Categorize sessions by PII risk level for the chart.
        def classify_pii(count):
            if count == 0:   return "None (0)"
            elif count <= 2: return "Low (1-2)"
            elif count <= 5: return "Medium (3-5)"
            else:            return "High (6+)"

        df_sessions["pii_risk"] = df_sessions["total_pii_detected"].apply(classify_pii)
        pii_dist = df_sessions["pii_risk"].value_counts().reset_index()
        pii_dist.columns = ["Risk Level", "Sessions"]

        # Ensure consistent ordering.
        order = ["None (0)", "Low (1-2)", "Medium (3-5)", "High (6+)"]
        pii_dist["Risk Level"] = pd.Categorical(pii_dist["Risk Level"], categories=order, ordered=True)
        pii_dist = pii_dist.sort_values("Risk Level")

        pii_colors = ["#7ac08a", "#c0c07a", "#c09060", "#c07a7a"]

        fig_pii = go.Figure(go.Bar(
            x           = pii_dist["Risk Level"],
            y           = pii_dist["Sessions"],
            marker_color= pii_colors[:len(pii_dist)],
            marker_line = dict(color="white", width=1.5),
            hovertemplate = "<b>%{x}</b>: %{y} sessions<extra></extra>",
        ))
        fig_pii.update_layout(
            **PLOTLY_LAYOUT_DEFAULTS,
            height = 280,
            xaxis  = dict(title="PII Risk Category", showgrid=False),
            yaxis  = dict(title="Sessions", showgrid=True, gridcolor=CHART_COLORS["grid"]),
            showlegend = False,
        )
        st.plotly_chart(fig_pii, use_container_width=True)

# --- Session Duration Distribution ---
with col_duration:
    st.markdown("### Session Duration (minutes)")

    if df_sessions.empty or "duration_min" not in df_sessions.columns:
        st.info("No duration data yet.")
    else:
        # Filter out sessions shorter than 0 minutes (clock skew edge case).
        df_dur = df_sessions[df_sessions["duration_min"] >= 0]

        fig_dur = px.box(
            df_dur,
            y         = "duration_min",
            color_discrete_sequence = [CHART_COLORS["accent"]],
            template  = "simple_white",
            points    = "all",       # Show individual data points for small datasets
            labels    = {"duration_min": "Duration (min)"},
        )
        # ACADEMIC NOTE — Box Plot vs Histogram for Duration:
        #   Box plots are better for small-to-medium datasets (< 1000 rows) because
        #   they show the median, IQR, and outliers simultaneously — more information
        #   density than a histogram for the same chart space.
        #   The `points="all"` argument overlays raw data points, crucial for
        #   academic transparency (reviewers can see if N is small).

        fig_dur.update_layout(
            **PLOTLY_LAYOUT_DEFAULTS,
            height = 280,
            yaxis  = dict(title="Duration (minutes)", showgrid=True, gridcolor=CHART_COLORS["grid"]),
            xaxis  = dict(showticklabels=False),
        )
        st.plotly_chart(fig_dur, use_container_width=True)

st.markdown('<hr class="section-divider"/>', unsafe_allow_html=True)


# ==============================================================================
# SECTION 11: RESPONSE LATENCY ANALYSIS
# ==============================================================================

st.markdown("## ⚡ System Performance Analysis")
st.caption(
    "AI response latency impacts user experience, especially for users in emotional "
    "distress who may interpret slow responses as disengagement. This section "
    "tracks system performance over time."
)

if not df_logs.empty and "response_latency_ms" in df_logs.columns:
    df_latency = df_logs.dropna(subset=["response_latency_ms"]).copy()

    if not df_latency.empty:
        col_lat_box, col_lat_time = st.columns([1, 2])

        with col_lat_box:
            st.markdown("### Latency Distribution")

            p50  = df_latency["response_latency_ms"].quantile(0.50)
            p95  = df_latency["response_latency_ms"].quantile(0.95)
            p99  = df_latency["response_latency_ms"].quantile(0.99)

            st.metric("Median (P50)",  f"{p50:.0f} ms")
            st.metric("95th Pct (P95)", f"{p95:.0f} ms",
                      help="95% of responses are faster than this.")
            st.metric("99th Pct (P99)", f"{p99:.0f} ms",
                      help="The slowest 1% of responses take this long.")

            # ACADEMIC NOTE — Percentile latency vs. average latency:
            #   Average (mean) latency is a misleading metric because a few
            #   very slow responses can inflate it significantly, obscuring
            #   the typical user experience. P95 and P99 (percentile) latencies
            #   are the industry-standard metrics for measuring "tail latency"
            #   — the worst-case experience that a real percentage of users
            #   actually encounters. This is a strong point to raise in your viva.

        with col_lat_time:
            st.markdown("### Latency Over Time")

            fig_lat = go.Figure()
            fig_lat.add_trace(go.Scatter(
                x          = df_latency["timestamp"],
                y          = df_latency["response_latency_ms"],
                mode       = "markers",
                name       = "Response Time",
                marker     = dict(
                    color  = df_latency["response_latency_ms"],
                    colorscale = [[0, CHART_COLORS["primary"]], [1, CHART_COLORS["accent"]]],
                    size   = 6,
                    opacity= 0.7,
                    showscale = True,
                    colorbar  = dict(title="ms", thickness=12),
                ),
                hovertemplate = "Time: %{x}<br>Latency: %{y}ms<extra></extra>",
            ))

            # Add a rolling average trend line.
            if len(df_latency) >= 5:
                df_latency_sorted = df_latency.sort_values("timestamp")
                rolling_avg = df_latency_sorted["response_latency_ms"].rolling(window=5, min_periods=1).mean()
                fig_lat.add_trace(go.Scatter(
                    x    = df_latency_sorted["timestamp"],
                    y    = rolling_avg,
                    mode = "lines",
                    name = "5-point Moving Avg",
                    line = dict(color=CHART_COLORS["accent"], width=2, dash="dash"),
                    hovertemplate = "Avg: %{y:.0f}ms<extra></extra>",
                ))

            fig_lat.update_layout(
                **PLOTLY_LAYOUT_DEFAULTS,
                height = 260,
                xaxis  = dict(title="Time", showgrid=True, gridcolor=CHART_COLORS["grid"]),
                yaxis  = dict(title="Latency (ms)", showgrid=True, gridcolor=CHART_COLORS["grid"]),
            )
            st.plotly_chart(fig_lat, use_container_width=True)

st.markdown('<hr class="section-divider"/>', unsafe_allow_html=True)


# ==============================================================================
# SECTION 12: ANONYMIZED DATA TABLE
# ==============================================================================

st.markdown("## 🗃️ Anonymized Message Log")
st.caption(
    "Raw (anonymized) message records for qualitative research review. "
    "Personal details have been replaced with entity-type placeholders "
    "(e.g., `<PERSON>`, `<PHONE_NUMBER>`) by the Presidio anonymization pipeline."
)

# Privacy reminder directly above the data table.
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
    # --- Apply filters from the sidebar ---
    df_display = df_logs.copy()

    # Apply emotion filter.
    if selected_emotion != "All":
        df_display = df_display[df_display["detected_emotion"] == selected_emotion]

    # Apply date filter.
    if "timestamp" in df_display.columns:
        df_display = df_display[
            (df_display["timestamp"].dt.date >= start_date) &
            (df_display["timestamp"].dt.date <= end_date)
        ]

    # Select and rename only the columns relevant for research review.
    # user_message_raw is NOT included here — this enforces the privacy
    # guarantee at the presentation layer as well as the query layer.
    display_columns = {
        "timestamp":              "Timestamp",
        "session_id":             "Session ID",
        "user_message_anonymized":"User Message (Anonymized)",
        "ai_response":            "AI Response",
        "detected_emotion":       "Emotion",
        "pii_entities_found":     "PII Entities Found",
        "response_latency_ms":    "Latency (ms)",
    }

    # Only keep columns that actually exist in the DataFrame.
    available_cols  = [c for c in display_columns.keys() if c in df_display.columns]
    df_display      = df_display[available_cols].rename(columns=display_columns)

    # Truncate session_id for readability in the table.
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
            "Timestamp": st.column_config.DatetimeColumn(
                "Timestamp", format="DD/MM/YY HH:mm"
            ),
            "Latency (ms)": st.column_config.NumberColumn(
                "Latency (ms)", format="%d ms"
            ),
            "User Message (Anonymized)": st.column_config.TextColumn(
                "User Message (Anonymized)", width="large"
            ),
            "AI Response": st.column_config.TextColumn(
                "AI Response", width="large"
            ),
        },
        hide_index = True,
    )

    # Export to CSV button.
    # ACADEMIC NOTE: Providing a CSV export is essential for a research dashboard.
    # Researchers need to import data into R, SPSS, or other statistical tools
    # for formal analysis beyond what the dashboard visualizes. Streamlit's
    # st.download_button handles this natively with no backend involvement —
    # the CSV is generated client-side from the DataFrame in memory.
    csv_data = df_display.to_csv(index=False).encode("utf-8")
    st.download_button(
        label     = "⬇️ Export filtered data as CSV",
        data      = csv_data,
        file_name = f"safespace_research_data_{datetime.today().strftime('%Y%m%d')}.csv",
        mime      = "text/csv",
        help      = "Downloads the currently filtered and displayed data as a CSV file."
    )

# ==============================================================================
# SECTION 13: DASHBOARD FOOTER
# ==============================================================================

st.markdown('<hr class="section-divider"/>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#a89588; font-size:0.8rem; padding: 1rem 0;">
    SafeSpace AI · B.Tech Final Year Project (AI & Data Science) · Research Dashboard v1.0<br/>
    Built with <strong>FastAPI</strong> · <strong>Google Gemini</strong> ·
    <strong>Microsoft Presidio</strong> · <strong>Streamlit</strong> · <strong>SQLite</strong><br/>
    <em>All data is anonymized. Privacy by Design.</em>
</div>
""", unsafe_allow_html=True)
