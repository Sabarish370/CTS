#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 DiD + ROI ENGINE -- Speaker Program & Peer-to-Peer ROI Analysis
================================================================================

Runs once per matching method against that method's matched-pairs file.

WHAT IT DOES, IN PLAIN TERMS
  Each matched-pairs file says "this doctor went to the program, this similar
  doctor didn't". This script looks up what actually happened to prescriptions
  after the program for both, subtracts what would have happened anyway (the
  control's own change), and reports the difference as a prescription COUNT.
  The dollar figure is a separate, deliberately thin multiplication on top.

TWO-LAYER DESIGN (Section 7)
  Layer 1 (expensive, one-time)  : incremental_lift -- a pure Rx count.
  Layer 2 (cheap, recomputable)  : x VALUE_PER_RX_CLAIM -> dollars, ROI.
  The dollar layer is never baked into the DiD, so a dashboard slider can
  re-derive every currency figure without re-running the engine.

WINDOWS (Section 4 -- taken as given, not re-derived)
  pre  = event month -6 .. -1
  post = event month +1 .. +6
  the event month itself is excluded from both (washout: the effect ramps up
  during that month, so it is neither a clean before nor a clean after).

Run
---
    python did_roi_engine.py
================================================================================
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

# ==============================================================================
# CONFIG
# ==============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
MATCHED_PAIRS_DIR = PROJECT_ROOT / "matched_pairs"
PREPROCESSED_DIR = PROJECT_ROOT / "preprocessed_data"
GENERATED_DIR = PROJECT_ROOT / "generated_data"
OUTPUT_DIR = PROJECT_ROOT / "did_roi_output"

RX_FILE = PREPROCESSED_DIR / "rx_claims_monthly_preprocessed (1).csv"
EVENTS_PREPROCESSED = PREPROCESSED_DIR / "events_preprocessed_final.csv"
EVENTS_RAW = GENERATED_DIR / "events.csv"
GROUND_TRUTH_FILE = GENERATED_DIR / "ground_truth_config.json"

# ---- the four methods, in the order they get reported ------------------------
METHODS = {
    "nnm": MATCHED_PAIRS_DIR / "nearest_neigbour_matching" / "NNM_matched_pairs.csv",
    "rule_based": MATCHED_PAIRS_DIR / "rule_based_matching" / "rule_based_matching_output.csv",
    "random": MATCHED_PAIRS_DIR / "randam_matching" / "matching_output_random.csv",
    # Repointed to the rebuilt PSM output. The previous file
    # (PSM_final_matching_output_FIXED.csv) sourced pre_period_rx_baseline from
    # the static baseline_rx_volume_monthly field, so it disagreed with NNM and
    # Rule-Based on the same (hcp, event); this engine treats that column as one
    # consistent fact across all four methods. Old file retired to
    # _RETIRED_PSM_final_matching_output_FIXED.csv.
    "psm": MATCHED_PAIRS_DIR / "propensity_score_matching" / "PSM_matched_pairs.csv",
}

# The 12 columns every method shares. Read ONLY these -- each method adds its
# own extras (rule_based 20, psm 15, nnm 14) and ignoring them is what keeps
# the four runs genuinely identical.
STANDARD_COLUMNS = [
    "treatment_hcp_id", "control_hcp_id", "method", "event_id", "event_date",
    "target_ndc_category", "specialty", "region", "pre_period_rx_baseline",
    "control_pre_period_rx_baseline", "is_matched", "match_rank",
]

# ---- windows (Section 4) -----------------------------------------------------
PRE_WINDOW = (-6, -1)      # inclusive, relative to event month
POST_WINDOW = (1, 6)       # inclusive; month 0 excluded == washout

# ---- ROI layer (Section 7) ---------------------------------------------------
VALUE_PER_RX_CLAIM = 150   # Assumed $ value per additional monthly prescription claim.
                           # NOT derived from the dataset -- no such field exists in
                           # ground_truth_config.json, hcp.csv, or events.csv.
                           # Chosen as a reasonable placeholder; must stay adjustable.

# ---- outliers (Section 8) ----------------------------------------------------
OUTLIER_NPIS = [1533568948, 1072969494, 1857093070]

# ---- per-method confidence flags (Section 10) --------------------------------
def build_confidence_note(method: str, n_pairs: int, n_treatment_total: int) -> str:
    """Per-method caveat, computed from THIS run instead of hardcoded.

    The previous version was a frozen string asserting PSM "drops 34% of
    treatment HCPs" and was "directionally informative, not a primary
    estimate". Both claims were written against the older PSM build whose
    pre_period_rx_baseline came from the static baseline_rx_volume_monthly
    field. After that bug was fixed PSM ranks #2 of 4 on the like-for-like
    metric, so an unconditional "not a primary estimate" now editorialises
    against its own result. State the drop rate as a fact and let the rank
    speak for itself.

    NOTE -- STALE STATISTIC REMOVED: the old note also quoted a
    propensity-score / Rx-similarity correlation of 0.166. That figure was
    measured against the pre-fix baseline and is NOT valid for the current
    output. It is deliberately not carried forward. If the team wants that
    diagnostic back, it must be RECOMPUTED against the corrected
    pre_period_rx_baseline (the one produced by NNM's compute_all_baselines)
    -- do not reuse the old number.
    """
    if method != "psm":
        return ""
    if not n_treatment_total:
        return "Common-support drop rate unavailable (no treatment cases)."
    drop_rate = (n_treatment_total - n_pairs) / n_treatment_total
    return (f"Common-support drop rate is {drop_rate:.2%} "
            f"({n_pairs}/{n_treatment_total} matched) -- a known small-sample "
            f"PSM limitation, not a code defect.")


def banner(t: str) -> None:
    print("\n" + "=" * 78); print(t); print("=" * 78)


def step(t: str) -> None:
    print("\n" + "-" * 78); print(t); print("-" * 78)


# ==============================================================================
# SHARED INPUTS
# ==============================================================================

def load_rx_panel():
    """Dense (hcp_id, ndc_category) x month matrix for O(1) window sums.

    Returns (matrix, row_lookup, month_to_idx). Re-filtering the 126k-row rx
    table per matched pair would be ~16k scans across the four methods; one
    pivot up front makes every window sum a numpy slice.
    """
    step("Loading Rx panel")
    rx = pd.read_csv(RX_FILE)

    months = (rx[["month", "month_date"]].drop_duplicates()
              .assign(month_date=lambda d: pd.to_datetime(d["month_date"]))
              .sort_values("month_date")["month"].tolist())
    month_to_idx = {m: i for i, m in enumerate(months)}
    rx["mi"] = rx["month"].map(month_to_idx)

    pivot = (rx.pivot_table(index=["hcp_id", "ndc_category"], columns="mi",
                            values="rx_volume", aggfunc="sum")
             .reindex(columns=range(len(months))))
    matrix = pivot.to_numpy(dtype=float)
    row_lookup = {k: i for i, k in enumerate(pivot.index)}

    n_gaps = int(np.isnan(matrix).sum())
    print(f"    rows                     : {len(rx):,d}")
    print(f"    panel                    : {rx.hcp_id.nunique():,d} HCPs x "
          f"{rx.ndc_category.nunique()} categories x {len(months)} months "
          f"({months[0]} .. {months[-1]})")
    print(f"    (hcp, category) series   : {len(row_lookup):,d}")
    print(f"    missing months INSIDE an existing series: {n_gaps}  "
          f"-> panel is {'COMPLETE (no gap-filling needed)' if n_gaps == 0 else 'GAPPY'}")
    return matrix, row_lookup, month_to_idx


def load_events_spend():
    """program_spend per event, preferring the preprocessed file (Section 2)."""
    step("Loading program_spend")
    note = ""
    if EVENTS_PREPROCESSED.exists():
        ev = pd.read_csv(EVENTS_PREPROCESSED)
        if "program_spend" in ev.columns:
            src = "events_preprocessed_final.csv"
        else:
            ev, src = pd.read_csv(EVENTS_RAW), "generated_data/events.csv"
            note = ("program_spend was MISSING from events_preprocessed_final.csv; "
                    "fell back to generated_data/events.csv")
            print(f"    !! {note}")
    else:
        ev, src = pd.read_csv(EVENTS_RAW), "generated_data/events.csv"
        note = "events_preprocessed_final.csv not found; used generated_data/events.csv"
        print(f"    !! {note}")

    spend = ev.set_index("event_id")["program_spend"].astype(float)
    attendees = (ev.set_index("event_id")["attendee_count"].astype(float)
                 if "attendee_count" in ev.columns else None)
    n_bad = int((spend.isna() | (spend <= 0)).sum())
    print(f"    source                   : {src}")
    print(f"    events                   : {len(spend):,d}")
    print(f"    program_spend null or <=0: {n_bad}  "
          f"(these events cannot carry an ROI denominator)")
    print(f"    total spend              : ${spend.sum():,.0f}")
    if attendees is not None:
        print(f"    attendee_count present   : yes -> per-attendee spend allocation available")
    return spend, attendees, src, note


def load_ground_truth():
    """true_effect_pct + the anchor (hcp, event) set, for COMPARISON ONLY.

    Section 9 bars the ground truth from being an input to any calculation.
    It is not one: every DiD, CI and ROI figure above is computed without it.
    It is used only afterwards, to SPLIT already-computed results into two
    buckets for interpretation -- which is a comparison step, not an input.

    Why the anchor set matters: the generator applied the +15% lift to each
    treatment HCP's ANCHOR event only (their earliest attendance), for that
    event's target category, from that date on. But the matched-pairs files
    carry one row per (hcp, event) and each HCP attended ~8 events. So roughly
    7 in 8 rows describe a program that never had a baked-in effect at all.
    Comparing an all-rows aggregate against 15% is therefore apples to
    oranges, and the split below is what makes the number interpretable.
    """
    gt = json.loads(GROUND_TRUTH_FILE.read_text(encoding="utf-8"))
    units = pd.DataFrame(gt["treatment_units"])
    anchors = set(zip(units["hcp_id"].astype("int64"), units["event_id"]))
    return float(gt["metadata"]["true_effect_pct"]), anchors


# ==============================================================================
# PER-PAIR DiD  (Section 5 -- pure prescription counts, no dollars)
# ==============================================================================

def compute_pair_level_did(pairs: pd.DataFrame, matrix, row_lookup, month_to_idx):
    """pre/post Rx SUMS for treatment and control, then the additive DiD.

    Both windows are summed over the target_ndc_category ONLY -- the drug the
    event actually promoted, not the HCP's total book of business. Summing all
    categories would bury a real product-level effect under unrelated volume.

    Missing-data policy (Section 12): the panel has no gaps inside a series, so
    the only way a pair can fail is if an HCP never prescribes the target
    category at all (no (hcp, category) series exists). Those pairs are
    EXCLUDED and counted explicitly rather than silently treated as zero --
    a real zero and "this doctor doesn't work in this therapeutic area" are
    different facts, and averaging the second one in as 0 would drag the
    estimate toward zero for a purely structural reason.
    """
    pre_lo, pre_hi = PRE_WINDOW
    post_lo, post_hi = POST_WINDOW
    n_months = matrix.shape[1]

    ev_month_idx = pairs["event_date"].str.slice(0, 7).map(month_to_idx).to_numpy()

    out = {k: np.full(len(pairs), np.nan) for k in
           ("pre_treatment_rx", "post_treatment_rx", "pre_control_rx", "post_control_rx")}
    coverage_fail = np.zeros(len(pairs), dtype=bool)
    window_fail = np.zeros(len(pairs), dtype=bool)

    t_ids = pairs["treatment_hcp_id"].to_numpy()
    c_ids = pairs["control_hcp_id"].to_numpy()
    cats = pairs["target_ndc_category"].to_numpy()

    for i in range(len(pairs)):
        e = ev_month_idx[i]
        if not np.isfinite(e):
            window_fail[i] = True
            continue
        e = int(e)
        a, b = e + pre_lo, e + pre_hi          # inclusive
        c, d = e + post_lo, e + post_hi        # inclusive
        if a < 0 or d >= n_months:
            window_fail[i] = True
            continue

        t_row = row_lookup.get((t_ids[i], cats[i]))
        c_row = row_lookup.get((c_ids[i], cats[i]))
        if t_row is None or c_row is None:
            coverage_fail[i] = True
            continue

        out["pre_treatment_rx"][i] = matrix[t_row, a:b + 1].sum()
        out["post_treatment_rx"][i] = matrix[t_row, c:d + 1].sum()
        out["pre_control_rx"][i] = matrix[c_row, a:b + 1].sum()
        out["post_control_rx"][i] = matrix[c_row, c:d + 1].sum()

    res = pairs.copy()
    for k, v in out.items():
        res[k] = v

    # --- Section 5, verbatim: a prescription count, no dollars anywhere near it
    res["treatment_rx_change"] = res["post_treatment_rx"] - res["pre_treatment_rx"]
    res["control_rx_change"] = res["post_control_rx"] - res["pre_control_rx"]
    res["incremental_lift"] = res["treatment_rx_change"] - res["control_rx_change"]
    res["incremental_lift_pct"] = np.where(
        res["pre_treatment_rx"] > 0,
        res["incremental_lift"] / res["pre_treatment_rx"] * 100.0, np.nan)

    res["_coverage_fail"] = coverage_fail
    res["_window_fail"] = window_fail
    res["_usable"] = res["incremental_lift"].notna() & (res["pre_treatment_rx"] > 0)
    return res


# ==============================================================================
# CLUSTERED INFERENCE  (Section 6)
# ==============================================================================

def clustered_mean_inference(y: np.ndarray, groups: np.ndarray):
    """Mean of y with standard errors clustered on `groups` (control_hcp_id).

    Some control doctors are reused across many treatment doctors (up to 82x
    for NNM). Treating each row as independent would make the engine far more
    confident than the data warrants, because those rows share one control's
    idiosyncratic Rx path. Clustering pools all rows sharing a control_hcp_id
    into a single unit of independent information.

    Implemented as an intercept-only OLS with cov_type='cluster' -- the
    coefficient IS the mean, and statsmodels applies the standard CR1
    finite-sample correction and t(G-1) reference distribution.
    """
    X = np.ones((len(y), 1))
    fit = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": groups})
    ci = fit.conf_int(alpha=0.05)[0]
    return {
        "mean": float(fit.params[0]),
        "se": float(fit.bse[0]),
        "t": float(fit.tvalues[0]),
        "p": float(fit.pvalues[0]),
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
        "n_clusters": int(pd.Series(groups).nunique()),
    }


# ==============================================================================
# AGGREGATION + ROI  (Sections 5, 7, 8, 9)
# ==============================================================================

def summarise(res: pd.DataFrame, label: str, method: str, spend: pd.Series,
              attendees, true_effect_pct: float, anchors: set,
              n_treatment_total: int) -> dict:
    """One summary row: DiD aggregate, clustered CI, then the ROI layer."""
    d = res[res["_usable"]]
    if len(d) == 0:
        return {"method": method, "sample": label, "n_pairs": 0,
                "confidence_note": build_confidence_note(method, 0, n_treatment_total)}

    total_lift = float(d["incremental_lift"].sum())
    total_pre_t = float(d["pre_treatment_rx"].sum())
    total_post_t = float(d["post_treatment_rx"].sum())
    total_pre_c = float(d["pre_control_rx"].sum())
    total_post_c = float(d["post_control_rx"].sum())

    # Volume-weighted aggregate -- the figure directly comparable to the 15%
    # ground truth (Section 5).
    lift_pct = total_lift / total_pre_t * 100.0

    inf = clustered_mean_inference(d["incremental_lift"].to_numpy(),
                                   d["control_hcp_id"].to_numpy())
    # Convert the CI on the per-pair MEAN lift into the same % units, holding
    # mean pre-period volume fixed (a scale change, not a new estimate).
    mean_pre_t = total_pre_t / len(d)
    to_pct = 100.0 / mean_pre_t

    # ---- ROI layer -- thin, standalone, recomputable (Section 7) ------------
    est_value = total_lift * VALUE_PER_RX_CLAIM

    # Spend allocation. PRIMARY = per-attendee share, because an event's cost is
    # shared across everyone who attended it; charging the whole program to only
    # the pairs a method happened to match would punish a selective method for
    # its selectivity rather than for its economics. events.csv carries
    # attendee_count precisely so this allocation is possible.
    ev_ids = d["event_id"]
    if attendees is not None:
        per_att = (ev_ids.map(spend) / ev_ids.map(attendees).replace(0, np.nan))
        spend_allocated = float(per_att.sum())
    else:
        spend_allocated = float(ev_ids.map(spend).sum())
    # SECONDARY = full cost of every distinct event touched by the matched set.
    spend_full = float(spend.reindex(ev_ids.unique()).sum())

    def roi(value, cost):
        if not cost or not np.isfinite(cost) or cost <= 0:
            return np.nan, np.nan
        return (value - cost) / cost * 100.0, value / cost

    roi_pct, roi_mult = roi(est_value, spend_allocated)
    roi_pct_full, roi_mult_full = roi(est_value, spend_full)

    # ---- diagnostics that explain a surprising headline number --------------
    # Additive DiD subtracts the control's ABSOLUTE change. If the control's
    # baseline volume differs from the treatment's, that absolute change is the
    # wrong size for the counterfactual and the estimate skews. The ratio DiD
    # is immune to that, so a gap between the two is a direct read on how much
    # baseline imbalance is distorting the headline figure.
    ratio_did_pct = ((total_post_t / total_pre_t) / (total_post_c / total_pre_c) - 1) * 100.0
    imbalance_pct = (total_pre_c - total_pre_t) / total_pre_t * 100.0

    # ---- ground-truth-aware decomposition (comparison layer only) ----------
    # anchor rows      = the ~1-in-8 cases that genuinely carry the +15%
    # non-anchor rows  = cases with NO baked-in effect, so whatever lift they
    #                    show is this method's residual confounding bias
    # implied effect   = anchor - non_anchor, i.e. the signal with that method's
    #                    own measured bias netted out. Closest to 15% wins.
    is_anchor = np.array([(t, e) in anchors for t, e in
                          zip(d["treatment_hcp_id"].astype("int64"), d["event_id"])])
    a_rows, n_rows = d[is_anchor], d[~is_anchor]

    def agg_pct(x):
        if len(x) == 0 or x["pre_treatment_rx"].sum() <= 0:
            return np.nan
        return x["incremental_lift"].sum() / x["pre_treatment_rx"].sum() * 100.0

    anchor_pct, nonanchor_pct = agg_pct(a_rows), agg_pct(n_rows)
    implied = (anchor_pct - nonanchor_pct
               if np.isfinite(anchor_pct) and np.isfinite(nonanchor_pct) else np.nan)

    return {
        "method": method,
        "sample": label,
        "n_pairs": len(d),
        "n_events": int(ev_ids.nunique()),
        "n_control_clusters": inf["n_clusters"],
        "max_control_reuse": int(d["control_hcp_id"].value_counts().max()),

        "total_pre_treatment_rx": total_pre_t,
        "total_post_treatment_rx": total_post_t,
        "total_pre_control_rx": total_pre_c,
        "total_post_control_rx": total_post_c,

        "total_incremental_lift": total_lift,
        "incremental_lift_pct": lift_pct,
        "mean_lift_per_pair": inf["mean"],
        "se_clustered": inf["se"],
        "t_stat": inf["t"],
        "p_value": inf["p"],
        "significant_05": bool(inf["p"] < 0.05),
        "lift_ci_low_pct": inf["ci_low"] * to_pct,
        "lift_ci_high_pct": inf["ci_high"] * to_pct,

        "value_per_rx_claim": VALUE_PER_RX_CLAIM,
        "estimated_business_value": est_value,
        "program_spend_allocated": spend_allocated,
        "roi_pct": roi_pct,
        "roi_multiple": roi_mult,
        "program_spend_full_events": spend_full,
        "roi_pct_full_spend": roi_pct_full,
        "roi_multiple_full_spend": roi_mult_full,

        "true_effect_pct": true_effect_pct,
        "gap_vs_truth_pp": lift_pct - true_effect_pct,

        "ratio_did_pct_diagnostic": ratio_did_pct,
        "baseline_imbalance_pct_diagnostic": imbalance_pct,

        # ---- comparison layer (uses ground truth; never feeds the above) ----
        "n_anchor_pairs": int(len(a_rows)),
        "n_nonanchor_pairs": int(len(n_rows)),
        "anchor_lift_pct": anchor_pct,
        "nonanchor_lift_pct_is_bias": nonanchor_pct,
        "implied_true_effect_pct": implied,
        "implied_gap_vs_truth_pp": (implied - true_effect_pct
                                    if np.isfinite(implied) else np.nan),

        "confidence_note": build_confidence_note(method, len(d), n_treatment_total),
    }


# ==============================================================================
# PER-METHOD RUN
# ==============================================================================

def run_method(method: str, path: Path, matrix, row_lookup, month_to_idx,
               spend, attendees, true_effect_pct, anchors, spend_note: str):
    banner(f"METHOD: {method}")
    raw = pd.read_csv(path)
    print(f"  file      : {path.relative_to(PROJECT_ROOT)}")
    print(f"  rows      : {len(raw):,d}   columns: {len(raw.columns)} "
          f"(reading only the {len(STANDARD_COLUMNS)} standard ones)")

    missing = [c for c in STANDARD_COLUMNS if c not in raw.columns]
    if missing:
        raise AssertionError(f"{method}: missing standard columns {missing}")

    df = raw[STANDARD_COLUMNS].copy()
    is_m = df["is_matched"]
    if is_m.dtype != bool:
        is_m = is_m.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
    df = df[is_m].copy()
    df["control_hcp_id"] = df["control_hcp_id"].astype("int64")
    print(f"  matched   : {len(df):,d}  ({len(df) / len(raw):.2%} of rows)")

    res = compute_pair_level_did(df, matrix, row_lookup, month_to_idx)

    n_cov = int(res["_coverage_fail"].sum())
    n_win = int(res["_window_fail"].sum())
    n_use = int(res["_usable"].sum())
    print(f"  usable    : {n_use:,d}")
    print(f"    dropped, no rx series for target category : {n_cov:,d}")
    print(f"    dropped, window falls outside panel       : {n_win:,d}")
    if n_use + n_cov + n_win != len(res):
        print(f"    !! unexplained shortfall: "
              f"{len(res) - n_use - n_cov - n_win:,d} rows")

    # Denominator for the common-support drop rate: every treatment case the
    # method was ASKED to match, i.e. all rows in its file, not just the ones
    # it succeeded on.
    n_treatment_total = len(raw)

    rows = [summarise(res, "all_pairs", method, spend, attendees,
                      true_effect_pct, anchors, n_treatment_total)]
    mask = (~res["treatment_hcp_id"].isin(OUTLIER_NPIS)
            & ~res["control_hcp_id"].isin(OUTLIER_NPIS))
    n_out = int((~mask).sum())
    rows.append(summarise(res[mask], "excl_outliers", method, spend,
                          attendees, true_effect_pct, anchors, n_treatment_total))
    print(f"  outlier rows removed for the 'excl_outliers' view: {n_out:,d}")

    summary = pd.DataFrame(rows)
    summary["spend_source_note"] = spend_note or "program_spend from events_preprocessed_final.csv"
    summary["missing_data_policy"] = (
        "Rx panel complete (no gaps within a series). Pairs whose target_ndc_category "
        "has no series for the treatment or control HCP are EXCLUDED and counted, "
        "never imputed as 0.")
    summary["window_definition"] = (
        f"pre=[{PRE_WINDOW[0]},{PRE_WINDOW[1]}], post=[{POST_WINDOW[0]},{POST_WINDOW[1]}] "
        f"months relative to event month; event month excluded (washout)")

    detail = res.loc[res["_usable"], [
        "treatment_hcp_id", "control_hcp_id", "event_id",
        "pre_treatment_rx", "post_treatment_rx", "pre_control_rx",
        "post_control_rx", "incremental_lift", "incremental_lift_pct"]].copy()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dpath = OUTPUT_DIR / f"did_roi_results_{method}.csv"
    spath = OUTPUT_DIR / f"did_roi_summary_{method}.csv"
    detail.to_csv(dpath, index=False)
    summary.to_csv(spath, index=False)

    a = rows[0]
    print(f"\n  RESULT (all pairs)")
    print(f"    total incremental lift : {a['total_incremental_lift']:>12,.0f} Rx")
    print(f"    incremental_lift_pct   : {a['incremental_lift_pct']:>12.2f}%   "
          f"[95% CI {a['lift_ci_low_pct']:.2f}% .. {a['lift_ci_high_pct']:.2f}%, "
          f"clustered on {a['n_control_clusters']} controls]")
    print(f"    vs true {true_effect_pct:.1f}%          : "
          f"{a['gap_vs_truth_pp']:>+12.2f} pp")
    print(f"    estimated value        : ${a['estimated_business_value']:>11,.0f}  "
          f"(@ ${VALUE_PER_RX_CLAIM}/Rx)")
    print(f"    allocated spend        : ${a['program_spend_allocated']:>11,.0f}  "
          f"-> ROI {a['roi_pct']:.1f}%  ({a['roi_multiple']:.2f}x)")
    print(f"    diagnostics            : ratio-DiD {a['ratio_did_pct_diagnostic']:.2f}%, "
          f"control baseline {a['baseline_imbalance_pct_diagnostic']:+.1f}% vs treatment")
    print(f"    anchor decomposition   : anchor {a['anchor_lift_pct']:.2f}% "
          f"({a['n_anchor_pairs']:,d} cases)  |  non-anchor "
          f"{a['nonanchor_lift_pct_is_bias']:.2f}% = residual bias "
          f"({a['n_nonanchor_pairs']:,d} cases)")
    print(f"    implied true effect    : {a['implied_true_effect_pct']:>12.2f}%   "
          f"({a['implied_gap_vs_truth_pp']:+.2f} pp vs {true_effect_pct:.1f}%)")
    print(f"    written                : {dpath.name}, {spath.name}")
    return summary, detail, res


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> int:
    banner("DiD + ROI ENGINE  --  4 methods, 8 output files")
    print(f"  output folder      : {OUTPUT_DIR}")
    print(f"  VALUE_PER_RX_CLAIM : ${VALUE_PER_RX_CLAIM}  (placeholder assumption, "
          f"not derived from data)")
    print(f"  windows            : pre {PRE_WINDOW}, post {POST_WINDOW}, "
          f"event month excluded")
    print(f"  outlier NPIs       : {OUTLIER_NPIS}")

    matrix, row_lookup, month_to_idx = load_rx_panel()
    spend, attendees, spend_src, spend_note = load_events_spend()
    true_effect_pct, anchors = load_ground_truth()
    print(f"\n    ground truth true_effect_pct = {true_effect_pct}  "
          f"(comparison only, never an input)")
    print(f"    anchor (hcp,event) cases carrying the baked-in effect: "
          f"{len(anchors):,d}")

    summaries, details = {}, {}
    for method, path in METHODS.items():
        s, d, _ = run_method(method, path, matrix, row_lookup, month_to_idx,
                             spend, attendees, true_effect_pct, anchors, spend_note)
        summaries[method] = s
        details[method] = d

    # ---- Section 12 sanity checks + cross-method comparison -----------------
    banner("SANITY CHECKS  (Section 12)")
    allrows = pd.concat(summaries.values(), ignore_index=True)
    a = allrows[allrows["sample"] == "all_pairs"].set_index("method")

    print("\n  lift plausible vs pre-period volume? "
          "(|lift| > total pre-period baseline would signal a bug)")
    for m in METHODS:
        r = a.loc[m]
        ratio = abs(r["total_incremental_lift"]) / r["total_pre_treatment_rx"]
        print(f"    {m:<11s} lift={r['total_incremental_lift']:>11,.0f}  "
              f"pre={r['total_pre_treatment_rx']:>12,.0f}  "
              f"ratio={ratio:6.3f}  [{'OK' if ratio < 1 else 'IMPLAUSIBLE'}]")

    print("\n  row-count reconciliation (no pair silently vanished)")
    for m, path in METHODS.items():
        raw = pd.read_csv(path)
        im = raw["is_matched"]
        n_m = int(im.sum()) if im.dtype == bool else int(
            im.astype(str).str.lower().eq("true").sum())
        n_d = len(details[m])
        print(f"    {m:<11s} matched={n_m:>5,d}  in detail file={n_d:>5,d}  "
              f"accounted-for gap={n_m - n_d:>3,d}")

    print("\n  HEADLINE (all matched rows) vs ground truth "
          f"{true_effect_pct:.1f}%   [* = placebo]")
    print(f"    {'method':<12s} {'lift %':>9s} {'95% CI':>22s} {'gap':>9s} "
          f"{'ROI %':>10s}")
    for m in METHODS:
        r = a.loc[m]
        star = " *" if m == "random" else "  "
        print(f"    {m:<12s}{star}{r['incremental_lift_pct']:>7.2f} "
              f"  [{r['lift_ci_low_pct']:>7.2f},{r['lift_ci_high_pct']:>7.2f}] "
              f"{r['gap_vs_truth_pp']:>+8.2f} {r['roi_pct']:>9.1f}")

    print("\n  !! The headline above is NOT directly comparable to 15%. Only "
          f"{len(anchors):,d} of the ~4,320\n     (hcp,event) rows are anchor "
          "cases carrying the baked-in effect; the rest dilute it.\n     The "
          "decomposition below is the like-for-like comparison.")
    print(f"\n    {'method':<12s} {'anchor%':>9s} {'bias%':>9s} "
          f"{'implied':>9s} {'gap':>9s}   verdict")
    ranked = sorted(METHODS, key=lambda m: abs(a.loc[m]["implied_gap_vs_truth_pp"]))
    for m in METHODS:
        r = a.loc[m]
        rank = ranked.index(m) + 1
        print(f"    {m:<12s}{r['anchor_lift_pct']:>9.2f} "
              f"{r['nonanchor_lift_pct_is_bias']:>9.2f} "
              f"{r['implied_true_effect_pct']:>9.2f} "
              f"{r['implied_gap_vs_truth_pp']:>+9.2f}   #{rank} closest to truth")
    print("\n    anchor%  = lift on cases that really do carry the +15%")
    print("    bias%    = lift on cases with NO true effect -> pure residual "
          "confounding")
    print("    implied  = anchor - bias, i.e. signal with that method's own bias "
          "netted out")

    banner("OUTPUT FILES")
    for f in sorted(OUTPUT_DIR.glob("did_roi_*.csv")):
        print(f"  {f.name:<38s} {f.stat().st_size:>9,d} B")
    return 0


if __name__ == "__main__":
    sys.exit(main())
