"""Ukraine Air Raid Alert Analytics Dashboard — main application module."""

from __future__ import annotations

import datetime as dt
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from statsmodels.tsa.seasonal import STL

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
    st.sidebar.markdown("## :material/filter_alt: Filters")

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
    default_oblasts = [o for o in oblast_options if o not in ("Donetska oblast", "Luhanska oblast")]
    selected_oblasts: list[str] = st.sidebar.multiselect(
        "Oblasts",
        options=oblast_options,
        default=default_oblasts,
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

    # --- stl decomposition ---
    st.markdown("### Time Series Decomposition (STL)")

    stl_period = st.radio(
        "Aggregation period",
        options=["Weekly", "Monthly"],
        horizontal=True,
        key="stl_period",
    )
    rule = "W" if stl_period == "Weekly" else "MS"

    ts = (
        oblast_df.set_index("started_at")
        .resample(rule)
        .size()
        .rename("alerts")
    )
    ts.index = ts.index.tz_localize(None)  # stl needs tz-naive index
    ts = ts.asfreq(rule, fill_value=0)  # fill gaps to keep regular freq

    # stl needs at least 2 full cycles; seasonal period = ~52 weeks or 12 months
    seasonal_period = 52 if rule == "W" else 12
    if len(ts) >= 2 * seasonal_period + 1:
        result = STL(ts, period=seasonal_period, robust=True).fit()

        fig_stl = make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            subplot_titles=("Observed", "Trend", "Seasonal", "Residual"),
        )
        fig_stl.add_trace(
            go.Scatter(x=ts.index, y=result.observed, mode="lines",
                       line=dict(color=COLOURS["primary"], width=1.2), showlegend=False),
            row=1, col=1,
        )
        fig_stl.add_trace(
            go.Scatter(x=ts.index, y=result.trend, mode="lines",
                       line=dict(color=COLOURS["secondary"], width=2), showlegend=False),
            row=2, col=1,
        )
        fig_stl.add_trace(
            go.Scatter(x=ts.index, y=result.seasonal, mode="lines",
                       line=dict(color=COLOURS["accent"], width=1.2), showlegend=False),
            row=3, col=1,
        )
        fig_stl.add_trace(
            go.Scatter(x=ts.index, y=result.resid, mode="markers",
                       marker=dict(color=COLOURS["danger"], size=3, opacity=0.6), showlegend=False),
            row=4, col=1,
        )
        fig_stl.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=720,
            margin=dict(l=0, r=0, t=30, b=10),
        )
        st.plotly_chart(fig_stl, use_container_width=True)
    else:
        st.warning(
            f"Not enough data for STL decomposition at {stl_period.lower()} granularity "
            f"(need ≥ {2 * seasonal_period + 1} periods, have {len(ts)})."
        )
        # fallback: simple time series line
        fig_w = px.line(
            ts.reset_index(),
            x="started_at",
            y="alerts",
            color_discrete_sequence=[COLOURS["primary"]],
            labels={"started_at": stl_period, "alerts": "Alerts"},
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
# tab 3 — lead-time & velocity vectors (averaged across all waves)
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Computing wave propagation statistics …")
def _compute_wave_stats(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Compute per-wave lead-time lags and aggregate across all multi-oblast waves."""
    # only consider oblast-level alerts for cleaner inter-region analysis
    odf = df[df["level"] == "oblast"].copy()

    # per-wave: earliest alert = wave_start
    wave_starts = odf.groupby("wave_id")["started_at"].min().rename("wave_start")
    odf = odf.merge(wave_starts, on="wave_id")
    odf["lead_time_lag_mins"] = (
        (odf["started_at"] - odf["wave_start"]).dt.total_seconds() / 60.0
    )

    # earliest alert per oblast per wave
    per_wave_oblast = (
        odf.groupby(["wave_id", "oblast"], as_index=False)
        .agg(lag_mins=("lead_time_lag_mins", "min"))
    )

    # filter to waves that hit >= 2 oblasts (meaningful propagation)
    wave_oblast_counts = per_wave_oblast.groupby("wave_id")["oblast"].nunique()
    multi_waves = wave_oblast_counts[wave_oblast_counts >= 2].index
    per_wave_oblast = per_wave_oblast[per_wave_oblast["wave_id"].isin(multi_waves)]
    n_waves = int(len(multi_waves))

    # aggregate: mean, median, p25, p75 lag per oblast across all waves
    avg_lead = (
        per_wave_oblast.groupby("oblast", as_index=False)
        .agg(
            mean_lag=("lag_mins", "mean"),
            median_lag=("lag_mins", "median"),
            p25_lag=("lag_mins", lambda x: x.quantile(0.25)),
            p75_lag=("lag_mins", lambda x: x.quantile(0.75)),
            wave_appearances=("wave_id", "nunique"),
        )
        .sort_values("mean_lag")
    )

    # fraction of waves where each oblast was the *first* to trigger
    first_per_wave = per_wave_oblast.loc[
        per_wave_oblast.groupby("wave_id")["lag_mins"].idxmin()
    ]
    first_counts = (
        first_per_wave.groupby("oblast", as_index=False)
        .size()
        .rename(columns={"size": "times_first"})
    )
    avg_lead = avg_lead.merge(first_counts, on="oblast", how="left")
    avg_lead["times_first"] = avg_lead["times_first"].fillna(0).astype(int)
    avg_lead["pct_first"] = (avg_lead["times_first"] / n_waves * 100).round(1)

    return avg_lead, per_wave_oblast, n_waves


def tab_velocity(df: pd.DataFrame) -> None:
    """Render averaged wave propagation analysis across all waves."""

    st.markdown("### Average Wave Propagation Analysis")
    st.caption(
        "Aggregated lead-time statistics across all multi-oblast waves. "
        "Shows how quickly each oblast typically receives an alert "
        "relative to the first-triggered region in each wave."
    )

    if df.empty:
        st.warning("No data in the selected range.")
        return

    avg_lead, per_wave_oblast, n_waves = _compute_wave_stats(df)

    # --- kpi row ---
    k1, k2, k3 = st.columns(3)
    with k1:
        kpi_card("Multi-Oblast Waves", f"{n_waves:,}")
    with k2:
        kpi_card("Oblasts Covered", f"{avg_lead['oblast'].nunique()}")
    with k3:
        overall_mean = avg_lead["mean_lag"].mean()
        kpi_card("Avg Cascade Spread (min)", f"{overall_mean:.1f}")

    st.markdown("---")

    # --- average lead-time bar chart with iqr error bars ---
    st.markdown("#### Mean Lead-Time Lag by Oblast")
    st.caption(
        "Bars show the average minutes after the first alert in a wave. "
        "Error bars show the interquartile range (P25–P75)."
    )

    sorted_lead = avg_lead.sort_values("mean_lag", ascending=True)
    fig_avg = go.Figure()
    fig_avg.add_trace(
        go.Bar(
            y=sorted_lead["oblast"],
            x=sorted_lead["mean_lag"],
            orientation="h",
            marker=dict(
                color=sorted_lead["mean_lag"],
                colorscale=["#22c55e", "#facc15", "#ef4444"],
                showscale=True,
                colorbar=dict(title="Lag (min)"),
            ),
            error_x=dict(
                type="data",
                symmetric=False,
                array=(sorted_lead["p75_lag"] - sorted_lead["mean_lag"]).clip(lower=0).tolist(),
                arrayminus=(sorted_lead["mean_lag"] - sorted_lead["p25_lag"]).clip(lower=0).tolist(),
                color="rgba(255,255,255,0.35)",
                thickness=1.2,
            ),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Mean lag: %{x:.1f} min<br>"
                "<extra></extra>"
            ),
        )
    )
    fig_avg.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=max(420, len(sorted_lead) * 26),
        margin=dict(l=0, r=0, t=10, b=10),
        xaxis_title="Mean Lead-Time Lag (min)",
    )
    st.plotly_chart(fig_avg, use_container_width=True)

    # --- first-to-trigger frequency ---
    st.markdown("#### First-to-Trigger Frequency")
    st.caption(
        "How often each oblast was the first region to sound an alert in a wave."
    )
    first_sorted = avg_lead.sort_values("times_first", ascending=True)
    first_sorted = first_sorted[first_sorted["times_first"] > 0]

    fig_first = px.bar(
        first_sorted,
        y="oblast",
        x="times_first",
        orientation="h",
        color="pct_first",
        color_continuous_scale=["#1e1e2e", COLOURS["secondary"], COLOURS["primary"]],
        hover_data=["pct_first"],
        labels={
            "oblast": "",
            "times_first": "Times First",
            "pct_first": "% of Waves",
        },
    )
    fig_first.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=max(360, len(first_sorted) * 26),
        margin=dict(l=0, r=0, t=10, b=10),
        coloraxis_colorbar=dict(title="%"),
    )
    st.plotly_chart(fig_first, use_container_width=True)

    # --- lead-time distribution violin per top oblasts ---
    st.markdown("#### Lead-Time Distribution (Top 12 Oblasts by Frequency)")
    top_oblasts = (
        avg_lead.nlargest(12, "wave_appearances")["oblast"].tolist()
    )
    violin_data = per_wave_oblast[per_wave_oblast["oblast"].isin(top_oblasts)]

    if not violin_data.empty:
        fig_vio = px.box(
            violin_data,
            x="oblast",
            y="lag_mins",
            color="oblast",
            color_discrete_sequence=px.colors.qualitative.Set2,
            labels={"oblast": "", "lag_mins": "Lead-Time Lag (min)"},
        )
        fig_vio.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=440,
            margin=dict(l=0, r=0, t=10, b=10),
            showlegend=False,
            xaxis_tickangle=-40,
        )
        st.plotly_chart(fig_vio, use_container_width=True)

    # --- summary table ---
    st.markdown("#### Summary Statistics")
    display_cols = [
        "oblast", "mean_lag", "median_lag", "p25_lag", "p75_lag",
        "wave_appearances", "times_first", "pct_first",
    ]
    display_df = avg_lead[display_cols].rename(columns={
        "mean_lag": "Mean Lag (min)",
        "median_lag": "Median Lag (min)",
        "p25_lag": "P25 (min)",
        "p75_lag": "P75 (min)",
        "wave_appearances": "Waves Active",
        "times_first": "Times First",
        "pct_first": "% First",
    })
    st.dataframe(
        display_df.reset_index(drop=True),
        use_container_width=True,
        height=460,
    )


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
        page_icon=":material/public:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_css()

    # header
    st.markdown(
        """
        # :material/public: Ukraine Air Raid Alert Analytics
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
        ":material/bar_chart: Volumetric Metrics",
        ":material/hourglass_empty: Temporal Kinetics",
        ":material/rocket_launch: Lead-Time & Velocity",
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
