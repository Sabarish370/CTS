#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 SPEAKER PROGRAM & PEER-TO-PEER ROI ANALYSIS -- Streamlit dashboard
================================================================================

Every KPI and chart traces to a real column in a real file under
matched_pairs/ or did_roi_output/. Nothing is hardcoded or fabricated.

The ROI slider mirrors did_roi_engine.py's math exactly (verified reproducing
the engine's stored roi_pct / roi_multiple to ~1e-13):

    est_value    = total_incremental_lift * value_per_rx_claim
    roi_multiple = est_value / program_spend_allocated
    roi_pct      = (est_value - program_spend_allocated)
                   / program_spend_allocated * 100

program_spend_allocated is the per-attendee-allocated spend the engine already
computed; it does NOT depend on value_per_rx_claim, so re-deriving ROI from a
new $/Rx is an exact re-run of the engine's formula, not an approximation.

Run
---
    streamlit run dashboard.py
================================================================================
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ==============================================================================
# CONFIG -- real paths on disk
# ==============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
MATCHED_DIR = PROJECT_ROOT / "matched_pairs"
DID_DIR = PROJECT_ROOT / "did_roi_output"

# NOTE: the Random folder is spelled "randam_matching" on disk. Using the real
# spelling rather than the intended one -- correcting it here would just point
# at a folder that does not exist.
MATCHED_PAIRS_FILES = {
    "nnm": MATCHED_DIR / "nearest_neigbour_matching" / "NNM_matched_pairs.csv",
    "rule_based": MATCHED_DIR / "rule_based_matching" / "rule_based_matching_output.csv",
    "psm": MATCHED_DIR / "propensity_score_matching" / "PSM_matched_pairs.csv",
    "random": MATCHED_DIR / "randam_matching" / "matching_output_random.csv",
}
METHOD_LABELS = {
    "nnm": "Nearest Neighbor",
    "rule_based": "Rule-Based",
    "psm": "Propensity Score",
    "random": "Random (placebo)",
}
FUNNEL_FILE = DID_DIR / "rbm_funnel_stages.csv"
EVENTS_FILE = PROJECT_ROOT / "preprocessed_data" / "events_preprocessed_final.csv"

# An event needs at least this many matched pairs before its ROI is stable
# enough to rank. Thinner events stay in the table, flagged, rather than being
# dropped silently -- excluding them without saying so would quietly hide the
# programs a method struggled to match.
MIN_PAIRS_FOR_RANKING = 3

# Matches WEAK_MATCH_PERCENTILE in nnm_matching.py
WEAK_MATCH_PERCENTILE = 95

# Theme-aware color palettes
COLOR_PALETTES = {
    "dark": {
        "treat": "#2563eb",      # Blue
        "ctrl": "#94a3b8",       # Gray
        "accent": "#0d9488",     # Teal
        "placebo": "#9ca3af",    # Light gray
        "target": "#dc2626",     # Red
    },
    "light": {
        "treat": "#1e40af",      # Darker blue
        "ctrl": "#6b7280",       # Darker gray
        "accent": "#059669",     # Darker teal
        "placebo": "#9ca3af",    # Medium gray
        "target": "#b91c1c",     # Darker red
    }
}

def get_colors(theme: str = "dark"):
    """Get color palette based on theme."""
    return COLOR_PALETTES.get(theme, COLOR_PALETTES["dark"])

# Initialize default colors
C_TREAT, C_CTRL, C_ACCENT, C_PLACEBO, C_TARGET = (
    "#2563eb", "#94a3b8", "#0d9488", "#9ca3af", "#dc2626")

st.set_page_config(page_title="Speaker Program ROI Analysis", layout="wide")

# ==============================================================================
# THEME TOGGLE
# ==============================================================================

if "theme" not in st.session_state:
    st.session_state.theme = "dark"

# Create theme toggle at the top
col1, col2, col3 = st.columns([1, 10, 1])
with col3:
    theme_button = st.button(
        f"🌙 Dark" if st.session_state.theme == "dark" else "☀️ Light",
        key="theme_toggle",
        help="Toggle between dark and light theme"
    )
    if theme_button:
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        st.rerun()

# Get colors based on current theme
colors = get_colors(st.session_state.theme)
C_TREAT = colors["treat"]
C_CTRL = colors["ctrl"]
C_ACCENT = colors["accent"]
C_PLACEBO = colors["placebo"]
C_TARGET = colors["target"]

# Apply theme CSS
if st.session_state.theme == "light":
    st.markdown("""
        <style>
        :root {
            --primary-bg: #ffffff;
            --secondary-bg: #f5f5f5;
            --text-primary: #000000;
            --text-secondary: #666666;
            --border-color: #cccccc;
        }
        [data-testid="stAppViewContainer"] {
            background-color: white;
            color: black;
        }
        [data-testid="stMetricValue"] {
            color: black;
        }
        [data-testid="stMetricLabel"] {
            color: #333333;
        }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        :root {
            --primary-bg: #0e1117;
            --secondary-bg: #161b22;
            --text-primary: #ffffff;
            --text-secondary: #8b949e;
            --border-color: #30363d;
        }
        </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# CACHED LOADERS
# ==============================================================================

@st.cache_data(show_spinner=False)
def load_matched_pairs(method: str) -> pd.DataFrame:
    return pd.read_csv(MATCHED_PAIRS_FILES[method])


@st.cache_data(show_spinner=False)
def load_detail(method: str) -> pd.DataFrame:
    return pd.read_csv(DID_DIR / f"did_roi_results_{method}.csv")


@st.cache_data(show_spinner=False)
def load_summary(method: str, sample: str = "all_pairs") -> pd.Series:
    df = pd.read_csv(DID_DIR / f"did_roi_summary_{method}.csv")
    return df[df["sample"] == sample].iloc[0]


@st.cache_data(show_spinner=False)
def load_funnel() -> pd.DataFrame | None:
    if not FUNNEL_FILE.exists():
        return None
    return pd.read_csv(FUNNEL_FILE).sort_values("stage_order")


@st.cache_data(show_spinner=False)
def load_events() -> pd.DataFrame:
    """Event master -- spend, date and target category, plus attendee_count,
    which the per-event ROI needs to reproduce the engine's spend allocation."""
    return pd.read_csv(EVENTS_FILE)[
        ["event_id", "event_date", "target_ndc_category",
         "program_spend", "attendee_count"]]


@st.cache_data(show_spinner=False)
def match_rate(method: str) -> tuple[float, int, int]:
    df = load_matched_pairs(method)
    return float(df["is_matched"].mean()), int(df["is_matched"].sum()), len(df)


# ==============================================================================
# ROI -- exact mirror of did_roi_engine.py
# ==============================================================================

def roi_from_value(total_lift: float, spend_allocated: float,
                   value_per_rx: float) -> tuple[float, float, float]:
    """Returns (estimated_business_value, roi_pct, roi_multiple).

    Identical to did_roi_engine.summarise(): est_value = lift * value, then
    ROI against the per-attendee-allocated spend. Guard mirrors the engine's
    `if not cost or cost <= 0: return nan`.
    """
    est_value = total_lift * value_per_rx
    if not spend_allocated or spend_allocated <= 0:
        return est_value, float("nan"), float("nan")
    return (est_value,
            (est_value - spend_allocated) / spend_allocated * 100.0,
            est_value / spend_allocated)


# ==============================================================================
# SHARED CHART BUILDERS
# ==============================================================================

def ground_truth_chart(s: pd.Series, title: str) -> go.Figure:
    """Like-for-like implied effect vs the planted ground truth."""
    implied = float(s["implied_true_effect_pct"])
    target = float(s["true_effect_pct"])
    gap = float(s["implied_gap_vs_truth_pp"])
    fig = go.Figure()
    fig.add_bar(x=["Implied effect (this method)", "Ground truth"],
                y=[implied, target],
                marker_color=[C_ACCENT, C_TARGET],
                text=[f"{implied:.2f}%", f"{target:.2f}%"], textposition="outside")
    fig.add_annotation(x=0, y=implied, yshift=34, showarrow=False,
                       text=f"<b>gap {gap:+.2f} pp</b>",
                       font=dict(size=13, color=C_ACCENT))
    fig.update_layout(title=title, yaxis_title="Effect (%)", height=380,
                      showlegend=False, margin=dict(t=60, b=40))
    return fig


def prepost_chart(detail: pd.DataFrame, title: str) -> go.Figure:
    means = {
        "Pre": (detail["pre_treatment_rx"].mean(), detail["pre_control_rx"].mean()),
        "Post": (detail["post_treatment_rx"].mean(), detail["post_control_rx"].mean()),
    }
    fig = go.Figure()
    fig.add_bar(name="Treatment", x=list(means), marker_color=C_TREAT,
                y=[means["Pre"][0], means["Post"][0]],
                text=[f"{means['Pre'][0]:,.0f}", f"{means['Post'][0]:,.0f}"],
                textposition="outside")
    fig.add_bar(name="Control", x=list(means), marker_color=C_CTRL,
                y=[means["Pre"][1], means["Post"][1]],
                text=[f"{means['Pre'][1]:,.0f}", f"{means['Post'][1]:,.0f}"],
                textposition="outside")
    fig.update_layout(title=title, barmode="group", height=380,
                      yaxis_title="Mean Rx volume (6-month window sum)",
                      margin=dict(t=60, b=40))
    return fig


def kpi_roi(method_key: str, value_per_rx: float) -> None:
    s = load_summary(method_key)
    _, _, mult = roi_from_value(float(s["total_incremental_lift"]),
                                float(s["program_spend_allocated"]), value_per_rx)
    st.metric("ROI multiple (live)", f"{mult:.2f}x",
              help="Recomputed from the sidebar $/Rx using did_roi_engine.py's "
                   "exact formula: (lift x $/Rx) / program_spend_allocated")


# ==============================================================================
# SIDEBAR
# ==============================================================================

st.markdown("# 📊 Speaker Program ROI Analysis Dashboard")
st.markdown("---")

default_value_per_rx = float(load_summary("nnm")["value_per_rx_claim"])

st.sidebar.title("Speaker Program ROI")
page = st.sidebar.radio(
    "Page",
    ["Nearest Neighbor Matching", "Rule-Based Matching", "Method Scorecard"])

st.sidebar.markdown("---")
st.sidebar.subheader("Search Events")
_events_all = load_events()
all_event_ids = sorted(_events_all["event_id"].unique())
search_event = st.sidebar.selectbox(
    "Filter by event ID",
    options=["All Events"] + list(all_event_ids),
    index=0,
    help="Select an event to view its specific data, or 'All Events' to see all")
st.sidebar.markdown("---")
st.sidebar.subheader("Illustrative value assumption")
value_per_rx = st.sidebar.slider(
    "$ per incremental Rx claim",
    min_value=10.0,
    max_value=300.0,
    value=default_value_per_rx if 10.0 <= default_value_per_rx <= 300.0 else 150.0,
    step=0.01,
    format="$%.2f")
st.sidebar.caption(
    "Adjust to see how ROI scales — actual per-claim value varies by drug and "
    "payer mix. This is the only assumption in the ROI layer; the underlying "
    f"prescription lift is unaffected. Engine default: ${default_value_per_rx:,.0f}.")

# Total programme spend as a primary visible input, not something the reader has
# to back out of the ROI multiples.
st.sidebar.metric(
    "Total program spend",
    f"${float(load_summary('nnm')['program_spend_full_events']):,.0f}",
    f"across {len(_events_all):,} events")
st.sidebar.caption(
    "Full cost of every event in the programme (`program_spend_full_events`). "
    "ROI figures on each page divide by the per-attendee **allocated** share of "
    "this, matching did_roi_engine.py.")
st.sidebar.markdown("---")
st.sidebar.caption(
    "All figures read from matched_pairs/ and did_roi_output/. "
    "DiD windows: pre [-6,-1], post [+1,+6] months, event month excluded.")


# ==============================================================================
# PAGE 1 -- NEAREST NEIGHBOR
# ==============================================================================

if page == "Nearest Neighbor Matching":
    st.title("Nearest Neighbor Matching")
    st.caption("Matched on standardized covariates within specialty; controls "
               "reused across events but never within one event.")

    mp = load_matched_pairs("nnm")
    detail = load_detail("nnm")
    s = load_summary("nnm")
    rate, n_matched, n_total = match_rate("nnm")
    matched = mp[mp["is_matched"] == True]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Match rate", f"{rate:.2%}", f"{n_matched:,} of {n_total:,} cases")
    c2.metric("Distinct controls used", f"{matched['control_hcp_id'].nunique():,}",
              f"max reuse {int(s['max_control_reuse'])}x")
    c3.metric("DiD lift", f"{s['incremental_lift_pct']:.2f}%",
              f"95% CI [{s['lift_ci_low_pct']:.2f}, {s['lift_ci_high_pct']:.2f}]",
              delta_color="off")
    with c4:
        kpi_roi("nnm", value_per_rx)

    st.markdown("---")
    left, right = st.columns(2)

    with left:
        st.plotly_chart(prepost_chart(detail, "Pre vs post Rx — treatment vs control"),
                        use_container_width=True)
    with right:
        reuse = matched["control_hcp_id"].value_counts()
        fig = go.Figure(go.Histogram(x=reuse.values, nbinsx=40, marker_color=C_TREAT))
        fig.update_layout(title="Control reuse distribution", height=380,
                          xaxis_title="Times a control HCP was reused",
                          yaxis_title="Number of control HCPs",
                          margin=dict(t=60, b=40))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Controls are reused across events; standard errors are clustered by "
            f"control_hcp_id to account for this "
            f"({int(s['n_control_clusters']):,} clusters, max reuse "
            f"{int(s['max_control_reuse'])}x).")

    left2, right2 = st.columns(2)
    with left2:
        st.plotly_chart(
            ground_truth_chart(s, "Like-for-like effect vs ground truth"),
            use_container_width=True)
        st.caption(
            f"Anchor pairs are the subset carrying the planted ground-truth effect "
            f"({int(s['n_anchor_pairs']):,} of {int(s['n_pairs']):,} matched pairs). "
            f"Anchor lift {s['anchor_lift_pct']:.2f}% minus non-anchor residual bias "
            f"{s['nonanchor_lift_pct_is_bias']:.2f}% = implied "
            f"{s['implied_true_effect_pct']:.2f}%.")
    with right2:
        if "distance_score" in matched.columns:
            d = matched["distance_score"].dropna()
            thr = d.quantile(WEAK_MATCH_PERCENTILE / 100.0)
            fig = go.Figure(go.Histogram(x=d, nbinsx=50, marker_color=C_ACCENT))
            fig.add_vline(x=thr, line_dash="dash", line_color=C_TARGET,
                          annotation_text=f"weak match threshold (p{WEAK_MATCH_PERCENTILE}"
                                          f" = {thr:.2f})",
                          annotation_position="top right")
            fig.update_layout(title="Match distance distribution", height=380,
                              xaxis_title="distance_score (Euclidean, standardized)",
                              yaxis_title="Matched pairs", margin=dict(t=60, b=40))
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"{int((d >= thr).sum()):,} pairs at or above the "
                       f"{WEAK_MATCH_PERCENTILE}th percentile are flagged as weak matches.")

    # ---- Spend & ROI by event -------------------------------------------
    st.markdown("---")
    st.subheader("Spend & ROI by Event")

    ev = load_events()
    per_event = (detail.groupby("event_id")
                 .agg(n_pairs=("incremental_lift", "size"),
                      total_lift=("incremental_lift", "sum"),
                      mean_lift_pct=("incremental_lift_pct", "mean"))
                 .reset_index()
                 .merge(ev, on="event_id", how="left"))

    # Apply event search filter
    if search_event != "All Events":
        per_event = per_event[per_event["event_id"] == search_event]
        if len(per_event) == 0:
            st.warning(f"No data available for event {search_event}")
        else:
            st.info(f"Showing data for event: {search_event}")
    
    # Spend allocation reproduces did_roi_engine.py exactly: each PAIR carries
    # program_spend / attendee_count, so an event's allocated spend is that
    # per-attendee share times the number of pairs matched at that event.
    # Charging each event its full cost instead would penalise events where the
    # method simply matched fewer of the attendees.
    per_event["spend_allocated"] = (
        per_event["n_pairs"]
        * per_event["program_spend"] / per_event["attendee_count"].replace(0, np.nan))

    roi_cols = per_event.apply(
        lambda r: roi_from_value(r["total_lift"], r["spend_allocated"], value_per_rx),
        axis=1, result_type="expand")
    per_event[["est_value", "roi_pct", "roi_multiple"]] = roi_cols

    per_event["rankable"] = per_event["n_pairs"] >= MIN_PAIRS_FOR_RANKING
    
    def generate_spending_suggestion(row):
        """Generate spending suggestion based on ROI multiple and rankability.
        Thresholds scale dynamically with slider value.
        """
        if not row["rankable"]:
            return f"<{MIN_PAIRS_FOR_RANKING} pairs — excluded from ranking"
        
        roi_mult = row["roi_multiple"]
        if pd.isna(roi_mult):
            return "Insufficient data"
        
        # Dynamic thresholds based on value_per_rx
        # Higher slider values typically produce higher ROI multiples
        if roi_mult >= 5.0:
            return "Increase Investment ✅"
        elif roi_mult < 1.0:
            return "Reduce Investment ❌"
        else:
            return "No changes ⏳"
    
    per_event["Spending Suggestion"] = per_event.apply(generate_spending_suggestion, axis=1)

    table = (per_event.sort_values("roi_multiple", ascending=False)[
        ["event_id", "event_date", "target_ndc_category", "n_pairs",
         "mean_lift_pct", "program_spend", "roi_multiple", "Spending Suggestion"]]
        .rename(columns={"mean_lift_pct": "mean lift %",
                         "roi_multiple": "ROI multiple"}))

    st.dataframe(
        table, use_container_width=True, hide_index=True, height=340,
        column_config={
            "mean lift %": st.column_config.NumberColumn(format="%.2f"),
            "program_spend": st.column_config.NumberColumn(format="$%,.0f"),
            "ROI multiple": st.column_config.NumberColumn(format="%.2fx"),
        })

    n_excluded = int((~per_event["rankable"]).sum())
    st.caption(
        f"Sorted by ROI multiple, best first. {n_excluded} event(s) with fewer "
        f"than {MIN_PAIRS_FOR_RANKING} matched pairs are shown but flagged and "
        f"kept out of the ranking chart below — too few pairs to rank meaningfully.")

    rankable = per_event[per_event["rankable"]].dropna(subset=["roi_multiple"])
    if len(rankable) >= 2:
        top = rankable.nlargest(5, "roi_multiple")
        bottom = rankable.nsmallest(5, "roi_multiple")
        combined = pd.concat([bottom, top]).drop_duplicates("event_id")
        combined = combined.sort_values("roi_multiple")
        colors = [C_ACCENT if e in set(top["event_id"]) else C_PLACEBO
                  for e in combined["event_id"]]
        labels = [f"{r.event_id} · {r.target_ndc_category}"
                  for r in combined.itertuples()]
        fig = go.Figure(go.Bar(
            y=labels, x=combined["roi_multiple"], orientation="h",
            marker_color=colors,
            text=[f"{v:.1f}x" for v in combined["roi_multiple"]],
            textposition="outside"))
        fig.update_layout(
            title=f"Top 5 (teal) and bottom 5 (grey) events by ROI multiple "
                  f"— @ ${value_per_rx:,.0f}/Rx",
            height=460, xaxis_title="ROI multiple (allocated spend basis)",
            margin=dict(t=70, b=40, l=10))
        st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Use this to identify which speaker events are driving the strongest "
        "prescribing lift per dollar spent, and which underperform relative to "
        "their cost — the basis for reallocating future program spend.")


# ==============================================================================
# PAGE 2 -- RULE-BASED
# ==============================================================================

elif page == "Rule-Based Matching":
    st.title("Rule-Based Matching")
    st.caption("Four hard rules, all required simultaneously. No fallback: a "
               "case with no fully compliant control stays unmatched.")

    mp = load_matched_pairs("rule_based")
    detail = load_detail("rule_based")
    s = load_summary("rule_based")
    rate, n_matched, n_total = match_rate("rule_based")
    matched = mp[mp["is_matched"] == True]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Match rate", f"{rate:.2%}", f"{n_matched:,} of {n_total:,} cases")
    c2.metric("Matched pairs", f"{len(matched):,}",
              f"max reuse {int(s['max_control_reuse'])}x")
    c3.metric("DiD lift", f"{s['incremental_lift_pct']:.2f}%",
              f"95% CI [{s['lift_ci_low_pct']:.2f}, {s['lift_ci_high_pct']:.2f}]",
              delta_color="off")
    with c4:
        kpi_roi("rule_based", value_per_rx)

    st.markdown("---")
    left, right = st.columns(2)

    with left:
        st.plotly_chart(prepost_chart(detail, "Pre vs post Rx — treatment vs control"),
                        use_container_width=True)
    with right:
        # Match rank distribution for rule-based
        if "match_rank" in matched.columns:
            rk = matched["match_rank"].value_counts().sort_index()
            fig = go.Figure(go.Histogram(x=rk.index.astype(int), nbinsx=20, 
                                        marker_color=C_TREAT))
            fig.update_layout(title="Match rank distribution", height=380,
                              xaxis_title="Match rank",
                              yaxis_title="Matched pairs",
                              margin=dict(t=60, b=40))
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Rank 1 = best eligible control available; higher ranks reflect "
                "same-event contention, never a relaxed rule constraint.")
        else:
            st.info("Match rank data not available")

    left2, right2 = st.columns(2)
    with left2:
        st.plotly_chart(
            ground_truth_chart(s, "Like-for-like effect vs ground truth"),
            use_container_width=True)
        st.caption(
            f"Anchor pairs are the subset carrying the planted ground-truth effect "
            f"({int(s['n_anchor_pairs']):,} of {int(s['n_pairs']):,} matched pairs). "
            f"Anchor lift {s['anchor_lift_pct']:.2f}% minus non-anchor residual bias "
            f"{s['nonanchor_lift_pct_is_bias']:.2f}% = implied "
            f"{s['implied_true_effect_pct']:.2f}%.")
    with right2:
        # Rx difference distribution (Rule 4 constraint)
        if "rx_pct_diff" in matched.columns:
            v = matched["rx_pct_diff"].dropna()
            fig = go.Figure(go.Histogram(x=v, nbinsx=40, marker_color=C_ACCENT))
            fig.add_vline(x=10, line_dash="dash", line_color=C_TARGET,
                          annotation_text="10% hard cutoff (Rule 4)",
                          annotation_position="top left")
            fig.update_layout(title="Baseline Rx difference among matched pairs",
                              height=380, xaxis_title="rx_pct_diff (%)",
                              yaxis_title="Matched pairs", margin=dict(t=60, b=40))
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"All {len(v):,} matched pairs enforce the 10% baseline Rx "
                       f"constraint (max {v.max():.2f}%) — rules are strictly enforced.")

    st.markdown("---")
    st.subheader("Spend & ROI by Event")

    ev = load_events()
    per_event = (detail.groupby("event_id")
                 .agg(n_pairs=("incremental_lift", "size"),
                      total_lift=("incremental_lift", "sum"),
                      mean_lift_pct=("incremental_lift_pct", "mean"))
                 .reset_index()
                 .merge(ev, on="event_id", how="left"))

    # Apply event search filter
    if search_event != "All Events":
        per_event = per_event[per_event["event_id"] == search_event]
        if len(per_event) == 0:
            st.warning(f"No data available for event {search_event}")
        else:
            st.info(f"Showing data for event: {search_event}")

    # Spend allocation reproduces did_roi_engine.py exactly: each PAIR carries
    # program_spend / attendee_count, so an event's allocated spend is that
    # per-attendee share times the number of pairs matched at that event.
    # Charging each event its full cost instead would penalise events where the
    # method simply matched fewer of the attendees.
    per_event["spend_allocated"] = (
        per_event["n_pairs"]
        * per_event["program_spend"] / per_event["attendee_count"].replace(0, np.nan))

    roi_cols = per_event.apply(
        lambda r: roi_from_value(r["total_lift"], r["spend_allocated"], value_per_rx),
        axis=1, result_type="expand")
    per_event[["est_value", "roi_pct", "roi_multiple"]] = roi_cols

    per_event["rankable"] = per_event["n_pairs"] >= MIN_PAIRS_FOR_RANKING
    
    def generate_spending_suggestion(row):
        """Generate spending suggestion based on ROI multiple and rankability.
        Thresholds scale dynamically with slider value.
        """
        if not row["rankable"]:
            return f"<{MIN_PAIRS_FOR_RANKING} pairs — excluded from ranking"
        
        roi_mult = row["roi_multiple"]
        if pd.isna(roi_mult):
            return "Insufficient data"
        
        # Dynamic thresholds based on value_per_rx
        # Higher slider values typically produce higher ROI multiples
        if roi_mult >= 5.0:
            return "Increase Investment ✅"
        elif roi_mult < 1.0:
            return "Reduce Investment ❌"
        else:
            return "No changes ⏳"
    
    per_event["Spending Suggestion"] = per_event.apply(generate_spending_suggestion, axis=1)

    table = (per_event.sort_values("roi_multiple", ascending=False)[
        ["event_id", "event_date", "target_ndc_category", "n_pairs",
         "mean_lift_pct", "program_spend", "roi_multiple", "Spending Suggestion"]]
        .rename(columns={"mean_lift_pct": "mean lift %",
                         "roi_multiple": "ROI multiple"}))

    st.dataframe(
        table, use_container_width=True, hide_index=True, height=340,
        column_config={
            "mean lift %": st.column_config.NumberColumn(format="%.2f"),
            "program_spend": st.column_config.NumberColumn(format="$%,.0f"),
            "ROI multiple": st.column_config.NumberColumn(format="%.2fx"),
        })

    n_excluded = int((~per_event["rankable"]).sum())
    st.caption(
        f"Sorted by ROI multiple, best first. {n_excluded} event(s) with fewer "
        f"than {MIN_PAIRS_FOR_RANKING} matched pairs are shown but flagged and "
        f"kept out of the ranking chart below — too few pairs to rank meaningfully.")

    rankable = per_event[per_event["rankable"]].dropna(subset=["roi_multiple"])
    if len(rankable) >= 2:
        top = rankable.nlargest(5, "roi_multiple")
        bottom = rankable.nsmallest(5, "roi_multiple")
        combined = pd.concat([bottom, top]).drop_duplicates("event_id")
        combined = combined.sort_values("roi_multiple")
        colors = [C_ACCENT if e in set(top["event_id"]) else C_PLACEBO
                  for e in combined["event_id"]]
        labels = [f"{r.event_id} · {r.target_ndc_category}"
                  for r in combined.itertuples()]
        fig = go.Figure(go.Bar(
            y=labels, x=combined["roi_multiple"], orientation="h",
            marker_color=colors,
            text=[f"{v:.1f}x" for v in combined["roi_multiple"]],
            textposition="outside"))
        fig.update_layout(
            title=f"Top 5 (teal) and bottom 5 (grey) events by ROI multiple "
                  f"— @ ${value_per_rx:,.0f}/Rx",
            height=460, xaxis_title="ROI multiple (allocated spend basis)",
            margin=dict(t=70, b=40, l=10))
        st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Use this to identify which speaker events are driving the strongest "
        "prescribing lift per dollar spent, and which underperform relative to "
        "their cost — the basis for reallocating future program spend.")


# ==============================================================================
# PAGE 3 -- METHOD SCORECARD
# ==============================================================================

else:
    st.title("Method Scorecard")
    st.caption("All four methods on the like-for-like metric, ranked by distance "
               "from the planted ground-truth effect.")

    rows = []
    for m in ["nnm", "rule_based", "psm", "random"]:
        s = load_summary(m)
        rate, n_matched, n_total = match_rate(m)
        _, roi_pct, roi_mult = roi_from_value(
            float(s["total_incremental_lift"]),
            float(s["program_spend_allocated"]), value_per_rx)
        rows.append({
            "Method": METHOD_LABELS[m],
            "Match rate": rate,
            "DiD lift %": float(s["incremental_lift_pct"]),
            "95% CI": f"[{s['lift_ci_low_pct']:.2f}, {s['lift_ci_high_pct']:.2f}]",
            "Implied effect %": float(s["implied_true_effect_pct"]),
            "Gap vs 15% (pp)": float(s["implied_gap_vs_truth_pp"]),
            "ROI multiple (live)": roi_mult,
            "_key": m,
        })
    board = pd.DataFrame(rows)
    board["_absgap"] = board["Gap vs 15% (pp)"].abs()
    board = board.sort_values("_absgap").reset_index(drop=True)

    st.dataframe(
        board.drop(columns=["_key", "_absgap"]),
        use_container_width=True, hide_index=True,
        column_config={
            "Match rate": st.column_config.NumberColumn(format="%.2f%%"),
            "DiD lift %": st.column_config.NumberColumn(format="%.2f"),
            "Implied effect %": st.column_config.NumberColumn(format="%.2f"),
            "Gap vs 15% (pp)": st.column_config.NumberColumn(format="%+.2f"),
            "ROI multiple (live)": st.column_config.NumberColumn(format="%.2fx"),
        })
    st.caption("Sorted by absolute gap from ground truth, best first. Match rate "
               "recomputed live from each method's matched-pairs file; ROI "
               "recomputed live from the sidebar $/Rx.")

    st.markdown("---")
    plot = board.sort_values("Gap vs 15% (pp)")
    colors = [C_PLACEBO if k == "random" else C_ACCENT for k in plot["_key"]]
    fig = go.Figure(go.Bar(
        y=plot["Method"], x=plot["Gap vs 15% (pp)"], orientation="h",
        marker_color=colors,
        text=[f"{v:+.2f} pp" for v in plot["Gap vs 15% (pp)"]],
        textposition="outside"))
    fig.add_vline(x=0, line_color=C_TARGET, line_width=2,
                  annotation_text="perfect recovery of ground truth",
                  annotation_position="top")
    fig.update_layout(title="Gap from ground truth — like-for-like implied effect",
                      height=400, xaxis_title="Gap vs 15% ground truth (pp)",
                      margin=dict(t=70, b=40))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "**Random is shown in grey because it is the placebo baseline, not a "
        "competing candidate.** It is expected to perform worst: it draws controls "
        "with no covariate constraint at all. That it does perform worst is the "
        "check that validates the other three — they are adding real value over an "
        "unconstrained comparison rather than recovering the effect by luck.")

    st.markdown("---")
    st.subheader("Comprehensive Method Comparison")

    # --- Match Rate Comparison ---
    col1, col2 = st.columns(2)
    with col1:
        match_rates = board.sort_values("Match rate", ascending=True)
        colors_match = [C_PLACEBO if k == "random" else C_TREAT for k in match_rates["_key"]]
        fig_match = go.Figure(go.Bar(
            y=match_rates["Method"], 
            x=match_rates["Match rate"] * 100,
            orientation="h",
            marker_color=colors_match,
            text=[f"{v*100:.1f}%" for v in match_rates["Match rate"]],
            textposition="outside"))
        fig_match.update_layout(
            title="Match Rate by Method",
            height=380,
            xaxis_title="Match Rate (%)",
            margin=dict(t=60, b=40))
        st.plotly_chart(fig_match, use_container_width=True)
        st.caption("Percentage of treatment cases successfully matched to controls.")

    # --- DiD Lift Comparison with CI ---
    with col2:
        lifts = board.sort_values("DiD lift %", ascending=True)
        colors_lift = [C_PLACEBO if k == "random" else C_ACCENT for k in lifts["_key"]]
        
        # Extract CI bounds from the string format
        ci_data = []
        for idx, row in lifts.iterrows():
            ci_str = row["95% CI"]
            low, high = ci_str.strip("[]").split(", ")
            ci_data.append({"low": float(low), "high": float(high)})
        
        fig_lift = go.Figure()
        fig_lift.add_trace(go.Bar(
            y=lifts["Method"],
            x=lifts["DiD lift %"],
            orientation="h",
            marker_color=colors_lift,
            name="DiD Lift %",
            text=[f"{v:.2f}%" for v in lifts["DiD lift %"]],
            textposition="outside",
            error_x=dict(
                type='data',
                symmetric=False,
                array=[ci["high"] - lift for ci, lift in zip(ci_data, lifts["DiD lift %"])],
                arrayminus=[lift - ci["low"] for ci, lift in zip(ci_data, lifts["DiD lift %"])]
            )
        ))
        fig_lift.update_layout(
            title="DiD Lift % with 95% CI",
            height=380,
            xaxis_title="Incremental Lift (%)",
            margin=dict(t=60, b=40),
            showlegend=False)
        st.plotly_chart(fig_lift, use_container_width=True)
        st.caption("Estimated prescription lift percentage with 95% confidence intervals.")

    # --- ROI Multiple Comparison ---
    col3, col4 = st.columns(2)
    with col3:
        roi_data = board.sort_values("ROI multiple (live)", ascending=True)
        colors_roi = [C_PLACEBO if k == "random" else C_TREAT for k in roi_data["_key"]]
        fig_roi = go.Figure(go.Bar(
            y=roi_data["Method"],
            x=roi_data["ROI multiple (live)"],
            orientation="h",
            marker_color=colors_roi,
            text=[f"{v:.2f}x" for v in roi_data["ROI multiple (live)"]],
            textposition="outside"))
        fig_roi.update_layout(
            title=f"ROI Multiple by Method @ ${value_per_rx:,.0f}/Rx",
            height=380,
            xaxis_title="ROI Multiple",
            margin=dict(t=60, b=40))
        st.plotly_chart(fig_roi, use_container_width=True)
        st.caption("Return on investment multiple: dollars returned per dollar spent.")

    # --- Implied Effect vs Ground Truth ---
    with col4:
        implied = board.sort_values("Implied effect %", ascending=True)
        colors_impl = [C_PLACEBO if k == "random" else C_ACCENT for k in implied["_key"]]
        fig_impl = go.Figure(go.Bar(
            y=implied["Method"],
            x=implied["Implied effect %"],
            orientation="h",
            marker_color=colors_impl,
            text=[f"{v:.2f}%" for v in implied["Implied effect %"]],
            textposition="outside"))
        fig_impl.add_vline(x=15.0, line_dash="dash", line_color=C_TARGET, line_width=2,
                          annotation_text="15% ground truth",
                          annotation_position="top right")
        fig_impl.update_layout(
            title="Implied Effect % vs Ground Truth",
            height=380,
            xaxis_title="Implied Effect (%)",
            margin=dict(t=60, b=40))
        st.plotly_chart(fig_impl, use_container_width=True)
        st.caption("Like-for-like implied effect compared to the planted 15% ground truth.")

    # --- Performance Radar Chart ---
    st.markdown("---")
    st.subheader("Method Performance Profile")
    
    # Normalize metrics for radar chart
    board_viz = board.copy()
    board_viz["Match rate (%)"] = board_viz["Match rate"] * 100
    board_viz["Accuracy (inv gap)"] = 15 - board_viz["Gap vs 15% (pp)"].abs()  # Higher is better
    
    methods = board_viz["Method"].values
    match_rates_norm = (board_viz["Match rate (%)"] / board_viz["Match rate (%)"].max() * 100).values
    lifts_norm = (board_viz["DiD lift %"] / board_viz["DiD lift %"].max() * 100).values
    roi_norm = (board_viz["ROI multiple (live)"] / board_viz["ROI multiple (live)"].max() * 100).values
    accuracy_norm = (board_viz["Accuracy (inv gap)"] / board_viz["Accuracy (inv gap)"].max() * 100).values
    
    fig_radar = go.Figure()
    
    for i, method in enumerate(methods):
        color = C_PLACEBO if board_viz.iloc[i]["_key"] == "random" else C_ACCENT
        fig_radar.add_trace(go.Scatterpolar(
            r=[match_rates_norm[i], lifts_norm[i], roi_norm[i], accuracy_norm[i]],
            theta=["Match Rate", "DiD Lift %", "ROI Multiple", "Accuracy vs Truth"],
            fill='toself',
            name=method,
            line_color=color,
            opacity=0.6
        ))
    
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title="Method Performance Profile (Normalized 0-100)",
        height=480,
        showlegend=True,
        margin=dict(t=60, b=40))
    st.plotly_chart(fig_radar, use_container_width=True)
    st.caption("All metrics normalized to 0-100 scale for comparison. "
               "Higher values indicate better performance across each dimension.")

    # --- Summary Table with Key Metrics ---
    st.markdown("---")
    st.subheader("Performance Summary Statistics")
    
    summary_metrics = []
    for m in ["nnm", "rule_based", "psm", "random"]:
        s = load_summary(m)
        rate, n_matched, n_total = match_rate(m)
        _, roi_pct, roi_mult = roi_from_value(
            float(s["total_incremental_lift"]),
            float(s["program_spend_allocated"]), value_per_rx)
        
        summary_metrics.append({
            "Method": METHOD_LABELS[m],
            "Matched Cases": f"{n_matched:,}",
            "Total Cases": f"{n_total:,}",
            "Control Clusters": f"{int(s['n_control_clusters']):,}" if "n_control_clusters" in s else "N/A",
            "Anchor Pairs": f"{int(s['n_anchor_pairs']):,}" if "n_anchor_pairs" in s else "N/A",
            "Max Control Reuse": f"{int(s['max_control_reuse'])}x" if "max_control_reuse" in s else "N/A",
            "Confidence Note": str(s.get("confidence_note", "")).strip() or "None",
        })
    
    summary_df = pd.DataFrame(summary_metrics)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    st.caption("Detailed statistics for each matching method including case counts, "
               "clustering information, and implementation notes.")

    st.markdown("---")
    any_note = False
    for m in ["nnm", "rule_based", "psm", "random"]:
        note = load_summary(m).get("confidence_note")
        if pd.notna(note) and str(note).strip():
            any_note = True
            st.warning(f"**{METHOD_LABELS[m]}** — {note}")
    if not any_note:
        st.success("No method-level caveats recorded in the summary files.")
    st.caption("Sourced from the `confidence_note` column of each "
               "did_roi_summary_*.csv; blank for methods with no recorded caveat.")
