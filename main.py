"""Ukraine Air Raid Alert Analytics Dashboard — main application module."""

from __future__ import annotations

import datetime as dt
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

DATA_URL: str = (
    "https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset"
    "/refs/heads/main/datasets/official_data_en.csv"
)
LOCAL_TZ: str = "Europe/Kyiv"
WAVE_GAP_MINUTES: int = 45
PAGE_SIZE: int = 50

# ---------------------------------------------------------------------------
# colour palette
# ---------------------------------------------------------------------------

COLOURS: dict[str, str] = {
    "primary": "#facc15",
    "secondary": "#38bdf8",
    "accent": "#f472b6",
    "danger": "#ef4444",
    "bg_card": "rgba(30, 30, 46, 0.65)",
    "text": "#e2e8f0",
}

# ---------------------------------------------------------------------------
# data helpers
# ---------------------------------------------------------------------------


@st.cache_data(ttl=0, show_spinner="Downloading dataset …")
def fetch_dataset() -> pd.DataFrame:
    """Download the CSV from GitHub and return a raw dataframe."""
    df = pd.read_csv(DATA_URL)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full preprocessing pipeline on a raw dataframe copy."""
    df = df.copy()

    # 1. timezone localisation
    df["started_at"] = (
        pd.to_datetime(df["started_at"], utc=True).dt.tz_convert(LOCAL_TZ)
    )
    df["finished_at"] = (
        pd.to_datetime(df["finished_at"], utc=True).dt.tz_convert(LOCAL_TZ)
    )

    # 2. duration
    df["duration_minutes"] = (
        (df["finished_at"] - df["started_at"]).dt.total_seconds() / 60.0
    )

    # 3. temporal features
    df["hour"] = df["started_at"].dt.hour
    df["day_of_week"] = df["started_at"].dt.day_name()
    df["month"] = df["started_at"].dt.to_period("M").astype(str)
    df["date"] = df["started_at"].dt.date

    # 4. wave clustering
    df = df.sort_values("started_at").reset_index(drop=True)
    time_deltas = df["started_at"].diff()
    df["wave_id"] = (
        time_deltas.gt(pd.Timedelta(minutes=WAVE_GAP_MINUTES)).cumsum() + 1
    )

    return df


# ---------------------------------------------------------------------------
# sidebar filters — returns filtered *oblast-level* view + full filtered df
# ---------------------------------------------------------------------------


def build_sidebar(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Render sidebar controls and return (oblast_df, full_filtered_df)."""
    st.sidebar.markdown("## 🇺🇦 Filters")

    # date range
    min_date = df["date"].min()
    max_date = df["date"].max()
    date_range = st.sidebar.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_d, end_d = date_range
    else:
        start_d, end_d = min_date, max_date

    full_filtered = df[(df["date"] >= start_d) & (df["date"] <= end_d)]

    # oblast multi-select (use only oblast-level records for list)
    oblast_options = sorted(
        df.loc[df["level"] == "oblast", "oblast"].dropna().unique()
    )
    selected_oblasts: list[str] = st.sidebar.multiselect(
        "Oblasts",
        options=oblast_options,
        default=oblast_options,
    )

    # oblast-level slice for tab-1 kpis
    oblast_df = full_filtered[
        (full_filtered["level"] == "oblast")
        & (full_filtered["oblast"].isin(selected_oblasts))
    ]

    # full filtered (all levels) but within selected oblasts
    full_filtered = full_filtered[
        full_filtered["oblast"].isin(selected_oblasts)
    ]

    return oblast_df, full_filtered


# ---------------------------------------------------------------------------
# kpi card helper
# ---------------------------------------------------------------------------


def kpi_card(label: str, value: str, delta: Optional[str] = None) -> None:
    """Render a styled metric card via st.metric."""
    st.metric(label=label, value=value, delta=delta)


# ---------------------------------------------------------------------------
# tab 1 — volumetric & baseline metrics
# ---------------------------------------------------------------------------


def tab_volumetric(oblast_df: pd.DataFrame, full_df: pd.DataFrame) -> None:
    """Render high-level KPI cards and paginated data table."""

    st.markdown("### Key Performance Indicators *(oblast-level only)*")
    st.caption(
        "Metrics below are computed from **oblast-level** records only to "
        "prevent double-counting across administrative hierarchies."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Total Alerts (N)", f"{len(oblast_df):,}")
    with c2:
        mean_dur = oblast_df["duration_minutes"].mean()
        kpi_card("Mean Duration (min)", f"{mean_dur:,.1f}" if pd.notna(mean_dur) else "—")
    with c3:
        total_hrs = oblast_df["duration_minutes"].sum() / 60.0
        kpi_card("Downtime Index (hrs)", f"{total_hrs:,.0f}")
    with c4:
        unique_waves = oblast_df["wave_id"].nunique()
        kpi_card("Unique Waves", f"{unique_waves:,}")

    # --- alert volume by oblast (bar chart) ---
    st.markdown("---")
    st.markdown("### Alert Volume by Oblast")

    vol = (
        oblast_df.groupby("oblast", as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("count", ascending=True)
    )
    fig_vol = px.bar(
        vol,
        y="oblast",
        x="count",
        orientation="h",
        color="count",
        color_continuous_scale=["#0e4429", "#006d32", "#26a641", "#39d353"],
        labels={"oblast": "", "count": "Alerts"},
    )
    fig_vol.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=max(400, len(vol) * 22),
        margin=dict(l=0, r=0, t=10, b=10),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_vol, use_container_width=True)

    # --- monthly trend line ---
    st.markdown("### Monthly Alert Trend")
    monthly = (
        oblast_df.groupby("month", as_index=False)
        .size()
        .rename(columns={"size": "alerts"})
        .sort_values("month")
    )
    fig_trend = px.area(
        monthly,
        x="month",
        y="alerts",
        color_discrete_sequence=[COLOURS["secondary"]],
        labels={"month": "Month", "alerts": "Alerts"},
    )
    fig_trend.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=340,
        margin=dict(l=0, r=0, t=10, b=10),
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # --- paginated data table ---
    st.markdown("---")
    st.markdown("### Processed Alert Log")
    display_cols = [
        "oblast",
        "raion",
        "hromada",
        "level",
        "started_at",
        "finished_at",
        "duration_minutes",
        "wave_id",
    ]
    available = [c for c in display_cols if c in full_df.columns]
    total_rows = len(full_df)
    total_pages = max(1, (total_rows + PAGE_SIZE - 1) // PAGE_SIZE)
    page = st.number_input(
        "Page", min_value=1, max_value=total_pages, value=1, step=1
    )
    start_idx = (page - 1) * PAGE_SIZE
    end_idx = start_idx + PAGE_SIZE
    st.caption(f"Showing rows {start_idx + 1}–{min(end_idx, total_rows)} of {total_rows}")
    st.dataframe(
        full_df.iloc[start_idx:end_idx][available].reset_index(drop=True),
        use_container_width=True,
        height=460,
    )


# ---------------------------------------------------------------------------
# tab 2 — temporal kinetics & seasonality
# ---------------------------------------------------------------------------

DOW_ORDER: list[str] = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


def tab_temporal(oblast_df: pd.DataFrame) -> None:
    """Render diurnal, day-of-week, and seasonality charts."""

    # --- diurnal (hourly) distribution ---
    st.markdown("### Diurnal Distribution (24 h)")
    hourly = (
        oblast_df.groupby("hour", as_index=False)
        .size()
        .rename(columns={"size": "alerts"})
    )
    # ensure all 24 hours present
    all_hours = pd.DataFrame({"hour": range(24)})
    hourly = all_hours.merge(hourly, on="hour", how="left").fillna(0)

    fig_h = px.bar(
        hourly,
        x="hour",
        y="alerts",
        color="alerts",
        color_continuous_scale=["#1e1e2e", "#facc15", "#ef4444"],
        labels={"hour": "Hour of Day (Kyiv)", "alerts": "Alerts"},
    )
    fig_h.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=360,
        margin=dict(l=0, r=0, t=10, b=10),
        coloraxis_showscale=False,
        xaxis=dict(dtick=1),
    )
    st.plotly_chart(fig_h, use_container_width=True)

    # --- day of week ---
    st.markdown("### Day-of-Week Variation")
    dow = (
        oblast_df.groupby("day_of_week", as_index=False)
        .agg(alerts=("day_of_week", "size"), avg_duration=("duration_minutes", "mean"))
    )
    dow["day_of_week"] = pd.Categorical(
        dow["day_of_week"], categories=DOW_ORDER, ordered=True
    )
    dow = dow.sort_values("day_of_week")

    fig_dow = go.Figure()
    fig_dow.add_trace(
        go.Bar(
            x=dow["day_of_week"],
            y=dow["alerts"],
            name="Alerts",
            marker_color=COLOURS["secondary"],
            yaxis="y",
        )
    )
    fig_dow.add_trace(
        go.Scatter(
            x=dow["day_of_week"],
            y=dow["avg_duration"],
            name="Avg Duration (min)",
            mode="lines+markers",
            marker_color=COLOURS["accent"],
            yaxis="y2",
        )
    )
    fig_dow.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=380,
        margin=dict(l=0, r=40, t=10, b=10),
        yaxis=dict(title="Alerts"),
        yaxis2=dict(title="Avg Duration (min)", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(fig_dow, use_container_width=True)

    # --- heatmap: hour x day-of-week ---
    st.markdown("### Hour × Day-of-Week Heatmap")
    heatmap_data = (
        oblast_df.groupby(["day_of_week", "hour"], as_index=False)
        .size()
        .rename(columns={"size": "alerts"})
    )
    heatmap_pivot = heatmap_data.pivot(
        index="day_of_week", columns="hour", values="alerts"
    ).reindex(DOW_ORDER).fillna(0)

    fig_hm = px.imshow(
        heatmap_pivot,
        color_continuous_scale=["#0d1117", "#facc15", "#ef4444"],
        labels=dict(x="Hour", y="Day", color="Alerts"),
        aspect="auto",
    )
    fig_hm.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=340,
        margin=dict(l=0, r=0, t=10, b=10),
    )
    st.plotly_chart(fig_hm, use_container_width=True)

    # --- stl placeholder ---
    st.markdown("### Time Series Decomposition (STL)")
    st.info(
        "📈 **Placeholder** — Weekly / monthly STL decomposition will appear "
        "here once `statsmodels.tsa.seasonal.STL` is wired into the pipeline. "
        "The grouped weekly alert counts are pre-computed and ready for decomposition.",
        icon="🔬",
    )
    weekly = (
        oblast_df.set_index("started_at")
        .resample("W")
        .size()
        .rename("alerts")
        .reset_index()
    )
    fig_w = px.line(
        weekly,
        x="started_at",
        y="alerts",
        color_discrete_sequence=[COLOURS["primary"]],
        labels={"started_at": "Week", "alerts": "Alerts"},
    )
    fig_w.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=300,
        margin=dict(l=0, r=0, t=10, b=10),
    )
    st.plotly_chart(fig_w, use_container_width=True)


# ---------------------------------------------------------------------------
# tab 3 — lead-time & velocity vectors
# ---------------------------------------------------------------------------


def tab_velocity(df: pd.DataFrame) -> None:
    """Render wave propagation analysis with lead-time lag vectors."""

    st.markdown("### Wave Propagation Analyzer")
    st.caption(
        "Select a tactical wave to see how the alert cascaded across oblasts. "
        "Lead-time is measured from the earliest alert in the wave."
    )

    # wave selector
    wave_ids = sorted(df["wave_id"].dropna().unique())
    if not wave_ids:
        st.warning("No waves found in the selected data range.")
        return

    # show a summary to help the user pick a wave
    wave_summary = (
        df.groupby("wave_id")
        .agg(
            start=("started_at", "min"),
            oblasts=("oblast", "nunique"),
            alerts=("wave_id", "size"),
        )
        .reset_index()
        .sort_values("start", ascending=False)
    )

    # only show waves with >= 2 oblasts for interesting analysis
    interesting = wave_summary[wave_summary["oblasts"] >= 2]

    col_sel, col_info = st.columns([1, 2])
    with col_sel:
        default_options = interesting["wave_id"].tolist()[:200]
        if not default_options:
            default_options = wave_ids[:200]
        selected_wave: int = st.selectbox(
            "Wave ID",
            options=default_options,
            format_func=lambda w: f"Wave {w}",
        )

    wave_data = df[df["wave_id"] == selected_wave].copy()
    wave_start = wave_data["started_at"].min()

    with col_info:
        st.markdown(
            f"**Wave {selected_wave}** — started "
            f"`{wave_start.strftime('%Y-%m-%d %H:%M')}` Kyiv · "
            f"**{wave_data['oblast'].nunique()}** oblasts · "
            f"**{len(wave_data)}** alerts"
        )

    # compute lead-time lag per oblast
    wave_data["lead_time_lag_mins"] = (
        (wave_data["started_at"] - wave_start).dt.total_seconds() / 60.0
    )

    # earliest alert per oblast within this wave
    oblast_lead = (
        wave_data.groupby("oblast", as_index=False)
        .agg(
            first_alert=("started_at", "min"),
            lag_mins=("lead_time_lag_mins", "min"),
            alerts_in_wave=("wave_id", "size"),
        )
        .sort_values("lag_mins")
    )

    # --- horizontal timeline bar ---
    st.markdown("#### Propagation Sequence")
    fig_prop = px.bar(
        oblast_lead,
        y="oblast",
        x="lag_mins",
        orientation="h",
        color="lag_mins",
        color_continuous_scale=["#22c55e", "#facc15", "#ef4444"],
        hover_data=["first_alert", "alerts_in_wave"],
        labels={
            "oblast": "",
            "lag_mins": "Lead-Time Lag (min)",
            "first_alert": "First Alert",
            "alerts_in_wave": "Alerts",
        },
    )
    fig_prop.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=max(360, len(oblast_lead) * 26),
        margin=dict(l=0, r=0, t=10, b=10),
        coloraxis_colorbar=dict(title="Lag (min)"),
    )
    st.plotly_chart(fig_prop, use_container_width=True)

    # --- detail table ---
    st.markdown("#### Detailed Wave Breakdown")
    display = wave_data[
        ["oblast", "raion", "hromada", "level", "started_at", "duration_minutes", "lead_time_lag_mins"]
    ].sort_values("lead_time_lag_mins")
    st.dataframe(display.reset_index(drop=True), use_container_width=True, height=400)


# ---------------------------------------------------------------------------
# custom css for dark theme polish
# ---------------------------------------------------------------------------


def inject_css() -> None:
    """Inject custom CSS to enhance visual theme."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="st-"] {
            font-family: 'Inter', sans-serif;
        }

        /* metric cards */
        [data-testid="stMetric"] {
            background: rgba(30, 30, 46, 0.65);
            border: 1px solid rgba(250, 204, 21, 0.18);
            border-radius: 12px;
            padding: 16px 20px;
            backdrop-filter: blur(8px);
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.82rem;
            letter-spacing: 0.03em;
            text-transform: uppercase;
            opacity: 0.75;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.65rem;
            font-weight: 700;
            color: #facc15;
        }

        /* tabs */
        button[data-baseweb="tab"] {
            font-weight: 600;
            font-size: 0.95rem;
            letter-spacing: 0.02em;
        }

        /* dataframe */
        [data-testid="stDataFrame"] {
            border-radius: 10px;
            overflow: hidden;
        }

        /* sidebar */
        section[data-testid="stSidebar"] {
            background: rgba(15, 15, 25, 0.85);
            backdrop-filter: blur(12px);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    """Application entry-point."""
    st.set_page_config(
        page_title="Ukraine Air Raid Alert Analytics",
        page_icon="🇺🇦",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_css()

    # header
    st.markdown(
        """
        # 🇺🇦 Ukraine Air Raid Alert Analytics
        *Real-time geospatial-temporal analysis of air raid alert patterns across Ukraine*
        """,
    )

    # load & preprocess
    with st.spinner("Loading & preprocessing data …"):
        raw = fetch_dataset()
        df = preprocess(raw)

    # sidebar filters
    oblast_df, full_df = build_sidebar(df)

    # tabs
    tab1, tab2, tab3 = st.tabs([
        "📊 Volumetric Metrics",
        "⏳ Temporal Kinetics",
        "🚀 Lead-Time & Velocity",
    ])

    with tab1:
        tab_volumetric(oblast_df, full_df)

    with tab2:
        tab_temporal(oblast_df)

    with tab3:
        tab_velocity(full_df)

    # footer
    st.markdown("---")
    st.caption(
        f"Data sourced from [ukrainian-air-raid-sirens-dataset]"
        f"(https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset) · "
        f"Last refresh: {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )


if __name__ == "__main__":
    main()
