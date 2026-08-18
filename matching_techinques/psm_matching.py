#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 PROPENSITY SCORE MATCHING (PSM) -- Speaker Program & Peer-to-Peer ROI Analysis
================================================================================

Fourth leg of the 4-way matching comparison (NNM / Rule-Based / Random / PSM).

THE BUG THIS REBUILD FIXES
--------------------------
The previous psm_matching.py populated pre_period_rx_baseline from the STATIC
`baseline_rx_volume_monthly` column in the HCP master file. That field is a
single all-purpose number per HCP -- it is not event-specific, not restricted
to the event's target_ndc_category, and not a 6-month window ending before the
event. NNM and Rule-Based both compute a real windowed mean from
rx_claims_monthly, so PSM's column disagreed with theirs: only 24 of 4,320
rows matched (correlation 0.99 -- close enough to look right at a glance,
which is exactly what made it dangerous).

One shared DiD/ROI function consumes all four methods' outputs and treats
pre_period_rx_baseline as one consistent fact. It has to mean the same thing
in every file.

THE FIX
-------
Import NNM's own functions instead of reimplementing any of them -- the same
pattern rbm_matching.py already uses. compute_all_baselines() produces a
{(event_id, hcp_id): baseline} dict that is the SINGLE source of every
baseline in this script, for treatment rows and control candidates alike.
`baseline_rx_volume_monthly` is never read.

Run
---
    python psm_matching.py
================================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Make the sibling nnm_matching module importable no matter which directory the
# script is launched from. Python only auto-adds the script's own folder to
# sys.path when it is invoked directly, so `python matching_techinques/
# psm_matching.py` works but other invocation styles would not.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ---- reuse the existing pipeline's logic verbatim -----------------------------
import nnm_matching as nnm
from nnm_matching import (
    banner,
    build_attended_event_dates,
    build_control_exclusions,
    build_rx_lookup,
    build_treatment_cohort,
    compute_all_baselines,
    is_temporally_contaminated,
    load_data,
    step,
)

# ==============================================================================
# CONFIG
# ==============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
# Same project convention as NNM and RBM: matched_pairs/<method>_matching/.
# Imported from nnm_matching rather than re-derived, so a future layout move is
# one edit instead of three that can drift apart.
OUTPUT_DIR = nnm.MATCHED_PAIRS_DIR / "propensity_score_matching"
OUTPUT_FILE = OUTPUT_DIR / "PSM_matched_pairs.csv"
OUTPUT_XLSX = OUTPUT_DIR / "PSM_matched_pairs.xlsx"

METHOD_NAME = "psm"
CALIPER = 0.05                       # max |propensity difference| for a match
WINDOW_MONTHS = nnm.WINDOW_MONTHS    # 6 -- identical to NNM/RBM, by import

STANDARD_COLUMNS = [
    "treatment_hcp_id", "control_hcp_id", "method", "event_id", "event_date",
    "target_ndc_category", "specialty", "region", "pre_period_rx_baseline",
    "control_pre_period_rx_baseline", "is_matched", "match_rank",
]
PSM_COLUMNS = ["propensity_score", "control_propensity_score", "propensity_distance"]
OUTPUT_COLUMNS = STANDARD_COLUMNS + PSM_COLUMNS


# ==============================================================================
# CONTROL POOL  (per event x specialty x region group)
# ==============================================================================

def build_group_control_pool(hcp_by_id: pd.DataFrame, specialty: str, region: str,
                             event_id: str, treatment_ids: set, exclusions: dict,
                             event_attendee_ids: dict, baseline_lookup: dict,
                             attended_dates_lookup: dict,
                             window_start_ord: int, event_date_ord: int) -> list:
    """Eligible controls for one (event, specialty, region) group.

    Exclusions, in order:
      1. globally speaker-flagged / role='speaker' HCPs -- the union computed
         by NNM's build_control_exclusions(), not a hand-rolled speaker set.
      2. this event's own attendees (including the treatment HCPs themselves).
      3. temporally contaminated -- the candidate attended some OTHER program
         inside this event's own 6-month pre-period window, so their baseline
         already reflects program exposure. NNM's check, by import.
      4. NEW: no computable baseline for this event/window. The old script
         never checked this, so a candidate with a missing baseline could be
         fed into the logistic fit and matched on a NaN feature.
    """
    pool = hcp_by_id[(hcp_by_id["specialty"] == specialty)
                     & (hcp_by_id["region"] == region)]["hcp_id"].tolist()

    blocked = exclusions["globally_excluded"]
    same_event = event_attendee_ids.get(event_id, frozenset())

    eligible = []
    for cid in pool:
        if cid in blocked or cid in same_event or cid in treatment_ids:
            continue
        b = baseline_lookup.get((event_id, cid))
        if b is None or pd.isna(b):
            continue
        if is_temporally_contaminated(cid, window_start_ord, event_date_ord,
                                      attended_dates_lookup):
            continue
        eligible.append(cid)
    return eligible


# ==============================================================================
# PSM PER GROUP
# ==============================================================================

def match_one_group(group: pd.DataFrame, controls: list, hcp_by_id: pd.DataFrame,
                    baseline_lookup: dict, event_id: str) -> list[dict]:
    """Fit propensity on this group, then greedy 1:1 NN within the caliper.

    Features: one-hot specialty + region, plus the event-specific
    pre_period_rx_baseline (the corrected one, from NNM's lookup).

    Greedy assignment takes the globally closest pair first, so the tightest
    propensity matches claim their control before looser ones compete for it.
    Controls are consumed without replacement WITHIN this event.
    """
    rows = []
    t_ids = group["treatment_hcp_id"].tolist()

    def unmatched(t_id, t_base):
        r = group[group["treatment_hcp_id"] == t_id].iloc[0]
        return {
            "treatment_hcp_id": t_id, "control_hcp_id": "", "method": METHOD_NAME,
            "event_id": event_id, "event_date": r["event_date"],
            "target_ndc_category": r["target_ndc_category"],
            "specialty": r["specialty"], "region": r["region"],
            "pre_period_rx_baseline": t_base,
            "control_pre_period_rx_baseline": np.nan,
            "is_matched": False, "match_rank": pd.NA,
            "propensity_score": np.nan, "control_propensity_score": np.nan,
            "propensity_distance": np.nan,
        }

    # A logistic fit needs both classes present. An empty (or fully excluded)
    # control pool leaves a single class -- ship those rows unmatched rather
    # than letting LogisticRegression raise.
    if len(controls) == 0:
        return [unmatched(t, baseline_lookup.get((event_id, t), np.nan)) for t in t_ids]

    frame = pd.DataFrame({
        "hcp_id": t_ids + controls,
        "treated": [1] * len(t_ids) + [0] * len(controls),
    })
    frame["pre_period_rx_baseline"] = [
        baseline_lookup.get((event_id, h), np.nan) for h in frame["hcp_id"]]
    frame = frame.merge(hcp_by_id[["hcp_id", "specialty", "region"]],
                        on="hcp_id", how="left")

    # Any treatment row without a computable baseline cannot be modelled.
    bad = frame[(frame["treated"] == 1) & (frame["pre_period_rx_baseline"].isna())]
    if len(bad):
        rows.extend(unmatched(t, np.nan) for t in bad["hcp_id"])
        frame = frame[~frame["hcp_id"].isin(set(bad["hcp_id"]))]
        t_ids = [t for t in t_ids if t not in set(bad["hcp_id"])]
    if len(t_ids) == 0 or frame["treated"].nunique() < 2:
        rows.extend(unmatched(t, baseline_lookup.get((event_id, t), np.nan))
                    for t in t_ids)
        return rows

    X = pd.get_dummies(frame[["specialty", "region"]],
                       columns=["specialty", "region"]).astype(float)
    X["pre_period_rx_baseline"] = frame["pre_period_rx_baseline"].to_numpy()
    y = frame["treated"].to_numpy()

    model = LogisticRegression(max_iter=1000)
    model.fit(X.to_numpy(), y)
    frame["propensity"] = model.predict_proba(X.to_numpy())[:, 1]

    t_rows = frame[frame["treated"] == 1].set_index("hcp_id")
    c_rows = frame[frame["treated"] == 0].set_index("hcp_id")

    # All within-caliper (treatment, control) pairs, closest first.
    pairs = []
    for t in t_ids:
        ps_t = t_rows.at[t, "propensity"]
        for c in c_rows.index:
            dist = abs(ps_t - c_rows.at[c, "propensity"])
            if dist <= CALIPER:
                pairs.append((dist, t, c))
    pairs.sort(key=lambda p: (p[0], str(p[1]), str(p[2])))

    used_t, used_c, assigned = set(), set(), {}
    for dist, t, c in pairs:
        if t in used_t or c in used_c:
            continue
        used_t.add(t); used_c.add(c)
        assigned[t] = (c, dist)

    for t in t_ids:
        t_base = baseline_lookup.get((event_id, t), np.nan)
        r = group[group["treatment_hcp_id"] == t].iloc[0]
        if t not in assigned:
            rows.append(unmatched(t, t_base))
            continue
        c, dist = assigned[t]
        rows.append({
            "treatment_hcp_id": t, "control_hcp_id": c, "method": METHOD_NAME,
            "event_id": event_id, "event_date": r["event_date"],
            "target_ndc_category": r["target_ndc_category"],
            "specialty": r["specialty"], "region": r["region"],
            "pre_period_rx_baseline": t_base,
            "control_pre_period_rx_baseline": baseline_lookup.get((event_id, c), np.nan),
            "is_matched": True, "match_rank": 1,
            "propensity_score": float(t_rows.at[t, "propensity"]),
            "control_propensity_score": float(c_rows.at[c, "propensity"]),
            "propensity_distance": float(dist),
        })
    return rows


# ==============================================================================
# SELF-VALIDATION  (asserts, not warnings)
# ==============================================================================

def validate(output: pd.DataFrame, exclusions: dict, baseline_lookup: dict) -> None:
    banner("SELF-VALIDATION  (asserts -- failures raise, they do not warn)")

    cols = list(output.columns[:12])
    assert cols == STANDARD_COLUMNS, (
        f"first 12 columns are not the standard schema in order.\n"
        f"  expected: {STANDARD_COLUMNS}\n  got     : {cols}")
    print("    [PASS] first 12 columns are the standard schema, in order")

    matched = output[output["is_matched"] == True]
    unmatched = output[output["is_matched"] == False]
    bad_un = (unmatched["control_hcp_id"].astype(str).str.strip() != "").sum()
    bad_m = (matched["control_hcp_id"].astype(str).str.strip() == "").sum()
    assert bad_un == 0, f"{bad_un} unmatched rows carry a control_hcp_id"
    assert bad_m == 0, f"{bad_m} matched rows have an empty control_hcp_id"
    print(f"    [PASS] control_hcp_id present iff matched "
          f"({len(matched):,d} matched / {len(unmatched):,d} unmatched)")

    used = set(matched["control_hcp_id"].astype(str))
    leaked = used & {str(x) for x in exclusions["globally_excluded"]}
    assert not leaked, f"{len(leaked)} speakers used as controls: {sorted(leaked)[:5]}"
    print(f"    [PASS] zero speakers among {len(used):,d} distinct controls used")

    if len(matched):
        worst = matched["propensity_distance"].max()
        assert worst <= CALIPER + 1e-12, (
            f"propensity_distance exceeds caliper {CALIPER}: max={worst}")
        print(f"    [PASS] every matched propensity_distance <= {CALIPER} "
              f"(max={worst:.6f})")

    # ---- THE KEY CHECK: does the baseline equal NNM's computation exactly? --
    have = output[output["pre_period_rx_baseline"].notna()]
    recomputed = np.array([baseline_lookup.get((e, t), np.nan) for e, t in
                           zip(have["event_id"], have["treatment_hcp_id"])])
    diff = np.abs(have["pre_period_rx_baseline"].to_numpy() - recomputed)
    worst = float(np.nanmax(diff)) if len(diff) else 0.0
    assert worst < 1e-9, (
        f"pre_period_rx_baseline does NOT match NNM's computation: "
        f"max abs diff = {worst}")
    print(f"    [PASS] pre_period_rx_baseline matches NNM's computation exactly "
          f"({len(have):,d} rows, max abs diff = {worst:.2e})")


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> int:
    banner("PROPENSITY SCORE MATCHING -- Speaker Program ROI Analysis")
    print(f"  caliper            : {CALIPER}")
    print(f"  pre-period window  : {WINDOW_MONTHS} months (imported from NNM)")
    print(f"  baseline source    : NNM compute_all_baselines() -- the static "
          f"baseline_rx_volume_monthly field is never read")

    # ---- shared inputs, via NNM's own loaders -------------------------------
    data = load_data()
    rx_matrix, row_lookup, month_to_idx, _ = build_rx_lookup(data["rx"])

    events_idx = data["events"].copy()
    events_idx["event_idx"] = events_idx["event_month"].map(month_to_idx)
    if events_idx["event_idx"].isna().any():
        raise AssertionError("some events fall outside the rx panel")

    cohort = build_treatment_cohort(data, month_to_idx)
    exclusions = build_control_exclusions(data)
    attended_dates_lookup = build_attended_event_dates(data["attendance"], data["events"])
    baseline_lookup = compute_all_baselines(data, events_idx, rx_matrix, row_lookup)["lookup"]

    # ---- attach each treatment case's own baseline from the SAME dict -------
    cohort["pre_period_rx_baseline"] = [
        baseline_lookup.get((e, t), np.nan)
        for e, t in zip(cohort["event_id"], cohort["treatment_hcp_id"])]

    # Window bounds as ordinal ints, precomputed once (rbm_matching.py pattern)
    # so the per-candidate contamination check is an integer bisect rather than
    # a date re-parse inside the loop.
    ev_ts = pd.to_datetime(cohort["event_date"])
    cohort["event_date_ord"] = ev_ts.map(lambda d: d.toordinal())
    cohort["window_start_ord"] = (
        ev_ts - pd.DateOffset(months=WINDOW_MONTHS)).map(lambda d: d.toordinal())
    cohort["event_date"] = ev_ts.dt.strftime("%Y-%m-%d")

    hcp_by_id = data["hcp"][["hcp_id", "specialty", "region"]].drop_duplicates("hcp_id")
    event_attendee_ids = {
        k: frozenset(v) for k, v in
        data["attendance"][data["attendance"]["role"] == "attendee"]
        .groupby("event_id")["hcp_id"].apply(set).to_dict().items()}
    treatment_ids_by_event = (cohort.groupby("event_id")["treatment_hcp_id"]
                              .apply(set).to_dict())

    step("Matching, one (event_id, specialty, region) group at a time")
    all_rows, n_groups, n_empty_pool = [], 0, 0
    for (event_id, specialty, region), group in cohort.groupby(
            ["event_id", "specialty", "region"], sort=True):
        n_groups += 1
        controls = build_group_control_pool(
            hcp_by_id, specialty, region, event_id,
            treatment_ids_by_event.get(event_id, set()), exclusions,
            event_attendee_ids, baseline_lookup, attended_dates_lookup,
            int(group["window_start_ord"].iloc[0]), int(group["event_date_ord"].iloc[0]))
        if not controls:
            n_empty_pool += 1
        all_rows.extend(match_one_group(group, controls, hcp_by_id,
                                        baseline_lookup, event_id))

    output = pd.DataFrame(all_rows)[OUTPUT_COLUMNS]
    output = output.sort_values(["event_id", "treatment_hcp_id"],
                                kind="stable").reset_index(drop=True)

    print(f"    groups processed                : {n_groups:,d}")
    print(f"    groups with an empty control pool: {n_empty_pool:,d}")
    print(f"    rows produced                   : {len(output):,d} "
          f"(treatment cases: {len(cohort):,d})")
    assert len(output) == len(cohort), (
        f"row count drift: {len(output)} rows vs {len(cohort)} treatment cases")

    validate(output, exclusions, baseline_lookup)

    banner("EXPORT")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_FILE, index=False)
    try:
        output.to_excel(OUTPUT_XLSX, index=False)
        xlsx_note = str(OUTPUT_XLSX)
    except Exception as exc:                       # openpyxl missing, etc.
        xlsx_note = f"SKIPPED ({type(exc).__name__}: {exc})"
    n_m = int(output["is_matched"].sum())
    print(f"    csv          : {OUTPUT_FILE}")
    print(f"    xlsx         : {xlsx_note}")
    print(f"    rows         : {len(output):,d}")
    print(f"    matched      : {n_m:,d}")
    print(f"    unmatched    : {len(output) - n_m:,d}")
    print(f"    match rate   : {100 * n_m / len(output):.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
