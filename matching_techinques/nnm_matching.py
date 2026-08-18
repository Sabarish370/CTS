#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 NEAREST NEIGHBOR MATCHING -- Speaker Program & Peer-to-Peer ROI Analysis
================================================================================

Builds one leg of a 4-way matching comparison (Random / Rule-Based / Nearest
Neighbor / PSM, one per teammate). This script owns the Nearest Neighbor leg
only: for every treatment case (an HCP attending a speaker program), it finds
the closest-covariate control HCP from an eligible non-speaker pool and emits
a shared-schema CSV so all four methods can be compared apples-to-apples.

No Rx-lift / DiD / ROI math happens here -- that is a downstream concern for
whichever script consumes NNM_matched_pairs.csv alongside the other 3 methods'
outputs.

Pipeline (function order mirrors the task steps 1-9)
-----------------------------------------------------
  1. load_data                        -- read the 4 preprocessed CSVs
  2. build_treatment_cohort           -- one row per (attendee hcp_id, event_id)
  3. build_control_exclusions         -- global speaker exclusion set
  4. build_rx_lookup                  -- dense (hcp_id, ndc_category) x month
                                          matrix for O(1) windowed-mean lookups
  5. compute_pre_period_baselines     -- vectorized, batched by event (not by
                                          row) so ~240k pair-lookups run as
                                          ~200 numpy slices instead of an apply
  6. build_static_features            -- GLOBAL z-score + one-hot, fit once
                                          across the full HCP population so
                                          every treatment case's tiny pool is
                                          scored on a comparable, stable scale
  7. build_candidate_pool             -- Step 4's hierarchical hard filter
  8. rank_candidates_by_distance      -- Step 5's sklearn NearestNeighbors
  9. assign_matches_greedy            -- Step 6's global greedy 1:1 assignment
 10. assemble_output                  -- Step 7's exact schema/column order
 11. run_qa_report                    -- Step 8's validation prints + asserts
 12. export_results                   -- Step 9's CSV write + row-count check

Run
---
    python nnm_matching.py
================================================================================
"""

from __future__ import annotations

import bisect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# CONFIG -- the only place file paths / tunable parameters live
# ==============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent


def find_project_root(start: Path) -> Path:
    """Walk up from this file until the folder holding the project's data dirs.

    The scripts have already moved twice (project root -> matching_techinques/),
    and every move silently repointed INPUT_DIR/MATCHED_PAIRS_DIR at folders
    that did not exist. Anchoring on a marker directory instead of a fixed
    number of .parent hops means the next reorganisation costs nothing: the
    script finds the root wherever it now sits.
    """
    for candidate in (start, *start.parents):
        if (candidate / "preprocessed_data").is_dir():
            return candidate
    # Nothing found (fresh clone with no data yet) -- assume the conventional
    # layout of scripts living one level below the root.
    return start.parent


PROJECT_ROOT = find_project_root(SCRIPT_DIR)
INPUT_DIR = PROJECT_ROOT / "preprocessed_data"

HCP_FILE = INPUT_DIR / "hcp-final.csv"
ATTENDANCE_FILE = INPUT_DIR / "attendance-final.csv"
EVENTS_FILE = INPUT_DIR / "events_preprocessed_final.csv"
RX_FILE = INPUT_DIR / "rx_claims_monthly_preprocessed (1).csv"

# PROJECT LAYOUT CONVENTION (all paths anchored on PROJECT_ROOT, never on
# SCRIPT_DIR, so moving the scripts between folders cannot break them)
#     <root>/generated_data/                    <- generate_data.py output
#     <root>/preprocessed_data/                 <- shared INPUT for all 4 methods
#     <root>/Preprocessing_tasks/               <- scripts that build the above
#     <root>/matching_techinques/               <- the 4 matching scripts
#     <root>/matched_pairs/nearest_neigbour_matching/  <- this script
#     <root>/matched_pairs/rule_based_matching/        <- rbm_matching.py
#     <root>/matched_pairs/randam_matching/            <- teammate (Random)
#     <root>/matched_pairs/propensity_score_matching/  <- teammate (PSM)
# Folder names match what is actually on disk, including the "neigbour",
# "randam" and "techinques" spellings -- renaming them here would silently
# write to a NEW empty folder and orphan the existing outputs. Rename on disk
# first if those are ever cleaned up, then update these constants.
#
# Outputs deliberately do NOT land in preprocessed_data/ -- that folder is the
# shared INPUT contract for all four methods, and writing results into it makes
# a teammate's input folder grow method artifacts they never asked for.
MATCHED_PAIRS_DIR = PROJECT_ROOT / "matched_pairs"
OUTPUT_DIR = MATCHED_PAIRS_DIR / "nearest_neigbour_matching"
OUTPUT_FILE = OUTPUT_DIR / "NNM_matched_pairs.csv"

WINDOW_MONTHS = 6          # pre-period width: calendar months strictly before
                            # the event's month (the event month itself is a
                            # partial-exposure washout, so it is excluded)
MIN_POOL_SIZE = 5          # below this, relax specialty+region -> specialty-only
WEAK_MATCH_PERCENTILE = 95  # distance_score values at/above this pct are flagged

METHOD_NAME = "nearest_neighbor"

STATIC_CONTINUOUS_COLS = [
    "patient_volume_monthly", "baseline_rx_volume_monthly", "years_in_practice",
]
STATIC_CATEGORICAL_COLS = ["sub_specialty", "practice_setting"]

OUTPUT_COLUMNS = [
    "treatment_hcp_id", "control_hcp_id", "method", "event_id", "event_date",
    "target_ndc_category", "specialty", "region", "pre_period_rx_baseline",
    "control_pre_period_rx_baseline", "is_matched", "match_rank",
    "distance_score", "pool_relaxed",
]


def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def step(title: str) -> None:
    print("\n" + "-" * 78)
    print(title)
    print("-" * 78)


# ==============================================================================
# STEP 0 (load) -- read inputs, defend against the known CRLF quirk
# ==============================================================================

def load_data() -> dict[str, pd.DataFrame]:
    step("LOAD  --  reading the 4 preprocessed CSVs")

    hcp = pd.read_csv(HCP_FILE)
    attendance = pd.read_csv(ATTENDANCE_FILE)
    events = pd.read_csv(EVENTS_FILE)
    rx = pd.read_csv(RX_FILE)

    # hcp-final.csv ships with CRLF line endings. pandas' default C parser
    # already strips the \r during line splitting, but we verify that no
    # stray \r survived into any string column (e.g. from a prior naive
    # str.split('\n') upstream) rather than just trusting it silently.
    stray_cr_found = False
    for name, df in [("hcp", hcp), ("attendance", attendance), ("events", events)]:
        for col in df.select_dtypes(include="object").columns:
            n_bad = df[col].astype(str).str.contains("\r").sum()
            if n_bad:
                stray_cr_found = True
                print(f"    WARNING: {n_bad} stray CR characters in "
                      f"{name}.{col} -- stripping")
                df[col] = df[col].astype(str).str.replace("\r", "", regex=False)
    print(f"    CRLF check: "
          f"{'stray CR found and stripped' if stray_cr_found else 'clean (pandas default parser handled it)'}")

    print(f"    hcp-final.csv                 : {len(hcp):>7,d} rows")
    print(f"    attendance-final.csv          : {len(attendance):>7,d} rows")
    print(f"    events_preprocessed_final.csv : {len(events):>7,d} rows")
    print(f"    rx_claims_monthly.csv         : {len(rx):>7,d} rows")

    return {"hcp": hcp, "attendance": attendance, "events": events, "rx": rx}


# ==============================================================================
# STEP 4 (helper, built early) -- dense rx lookup matrix
# ==============================================================================

def build_rx_lookup(rx: pd.DataFrame):
    """Dense (hcp_id, ndc_category) x month_idx matrix + a month index map.

    Every downstream pre-period-baseline lookup becomes an O(1) dict hit plus
    a numpy slice, instead of re-filtering the 126k-row rx table per pair --
    this is what keeps ~240k (hcp, event) baseline computations fast.
    """
    step("Building dense rx lookup matrix (hcp_id x ndc_category x month)")

    month_map = (rx[["month", "month_date"]].drop_duplicates()
                 .assign(month_date=lambda d: pd.to_datetime(d["month_date"]))
                 .sort_values("month_date"))
    months_sorted = month_map["month"].tolist()
    month_to_idx = {m: i for i, m in enumerate(months_sorted)}
    n_months = len(months_sorted)

    rx = rx.copy()
    rx["month_idx"] = rx["month"].map(month_to_idx)

    pivot = rx.pivot_table(index=["hcp_id", "ndc_category"], columns="month_idx",
                           values="rx_volume", aggfunc="mean")
    pivot = pivot.reindex(columns=range(n_months))
    matrix = pivot.to_numpy(dtype=float)
    row_lookup = {key: i for i, key in enumerate(pivot.index)}

    print(f"    months in panel   : {n_months}  ({months_sorted[0]} .. {months_sorted[-1]})")
    print(f"    (hcp_id, category) series: {len(row_lookup):,d}")
    print(f"    matrix shape      : {matrix.shape}   nulls inside existing "
          f"series: {np.isnan(matrix).sum():,d}")
    return matrix, row_lookup, month_to_idx, n_months


def compute_pre_period_baselines(pairs: pd.DataFrame, rx_matrix: np.ndarray,
                                 row_lookup: dict, window_months: int) -> np.ndarray:
    """Mean rx_volume over the `window_months` calendar months strictly BEFORE
    each row's event month.

    `pairs` needs columns hcp_id, ndc_category, event_idx (any row index is
    fine, order-preserving). Grouped by event_idx -- not iterated row by row --
    so the ~200 distinct event windows each resolve as one vectorized numpy
    slice over their whole batch of candidates, rather than ~240,000 individual
    per-pair lookups.

    A row's result is NaN (i.e. "zero rx_claims rows in that window", per the
    business rule) when either: the HCP never prescribes that ndc_category at
    all, or every month in the window is missing for that HCP+category. We use
    nanmean over whatever window months ARE present rather than requiring the
    full window, since the rule is "zero rows", not "incomplete window".
    """
    result = np.full(len(pairs), np.nan)
    hcp_arr = pairs["hcp_id"].to_numpy()
    cat_arr = pairs["ndc_category"].to_numpy()
    idx_arr = pairs["event_idx"].to_numpy()
    pos_arr = np.arange(len(pairs))

    for event_idx in np.unique(idx_arr):
        start, end = event_idx - window_months, event_idx - 1  # inclusive
        if start < 0 or end < 0:
            continue  # would require pre-panel history that doesn't exist
        mask = idx_arr == event_idx
        rows = np.array([row_lookup.get((h, c), -1)
                         for h, c in zip(hcp_arr[mask], cat_arr[mask])])
        present = rows >= 0
        if not present.any():
            continue
        window_vals = rx_matrix[rows[present], start:end + 1]
        with np.errstate(invalid="ignore"):
            has_data = ~np.all(np.isnan(window_vals), axis=1)
            means = np.nanmean(window_vals, axis=1)
        out_positions = pos_arr[mask][present][has_data]
        result[out_positions] = means[has_data]
    return result


# ==============================================================================
# STEP 1 -- treatment cohort
# ==============================================================================

def build_treatment_cohort(data: dict, month_to_idx: dict) -> pd.DataFrame:
    """One row per (attendee hcp_id, event_id). specialty/region are the
    TREATMENT HCP's own attributes (from hcp-final.csv), not the event's
    specialty_focus/region -- those can legitimately differ, since an HCP may
    attend a portfolio-wide event outside their own specialty focus.
    """
    step("STEP 1  --  Building treatment cohort")

    att, hcp, events = data["attendance"], data["hcp"], data["events"]

    cohort = att.loc[att["role"] == "attendee",
                     ["hcp_id", "event_id"]].drop_duplicates()
    n_raw = len(cohort)

    cohort = cohort.merge(
        hcp[["hcp_id", "specialty", "region"]], on="hcp_id", how="left")
    cohort = cohort.merge(
        events[["event_id", "event_date", "target_ndc_category", "event_month"]],
        on="event_id", how="left")
    cohort = cohort.rename(columns={"hcp_id": "treatment_hcp_id"})
    cohort["event_idx"] = cohort["event_month"].map(month_to_idx)

    n_missing_join = cohort[["specialty", "region", "target_ndc_category"]].isna().any(axis=1).sum()
    print(f"    attendee rows (raw)         : {n_raw:,d}")
    print(f"    unique treatment HCPs       : {cohort['treatment_hcp_id'].nunique():,d}")
    print(f"    treatment cases (rows)      : {len(cohort):,d}")
    print(f"    rows with a failed hcp/event join: {n_missing_join}")
    if n_missing_join:
        raise AssertionError(
            f"{n_missing_join} treatment rows failed to join to hcp or events -- "
            "check for orphaned hcp_id/event_id references before proceeding")

    return cohort.reset_index(drop=True)


# ==============================================================================
# STEP 2 -- control exclusions
# ==============================================================================

def build_control_exclusions(data: dict) -> dict:
    """Global (event-independent) HALF of control eligibility: an HCP can
    NEVER be used as a control anywhere if EITHER of these holds -- both
    checks are unioned rather than relying on one, because the two source
    systems (the HCP master flag and the attendance log) can disagree, and a
    false negative here would let an actual speaker slip into a control pool
    undetected.

    This is only half of eligibility. The other half -- whether a candidate's
    OWN recent program attendance contaminates them for a SPECIFIC treatment
    case's pre-period window -- cannot be precomputed as a single global set,
    because it depends on which treatment case is asking (different event
    dates imply different windows). See build_attended_event_dates() below
    and its use in build_candidate_pool() (Step 4).
    """
    step("STEP 2  --  Building control exclusion set (speaker-status union)")

    hcp, att = data["hcp"], data["attendance"]

    speaker_flagged_ids = set(hcp.loc[hcp["speaker_eligible_flag"], "hcp_id"])
    speaker_role_ids = set(att.loc[att["role"] == "speaker", "hcp_id"])
    globally_excluded = speaker_flagged_ids | speaker_role_ids

    eligible_universe = set(hcp["hcp_id"]) - globally_excluded

    print(f"    speaker_eligible_flag=True HCPs   : {len(speaker_flagged_ids):,d}")
    print(f"    role='speaker' HCPs (attendance)  : {len(speaker_role_ids):,d}")
    print(f"    union (never-eligible as control)  : {len(globally_excluded):,d}")
    print(f"    flag-only (missed by role check)   : "
          f"{len(speaker_flagged_ids - speaker_role_ids):,d}")
    print(f"    role-only (missed by flag check)   : "
          f"{len(speaker_role_ids - speaker_flagged_ids):,d}")
    print(f"    eligible control universe          : {len(eligible_universe):,d} "
          f"of {len(hcp):,d} total HCPs")

    return {
        "speaker_flagged_ids": speaker_flagged_ids,
        "speaker_role_ids": speaker_role_ids,
        "globally_excluded": globally_excluded,
        "eligible_universe": eligible_universe,
    }


def build_attended_event_dates(attendance: pd.DataFrame, events: pd.DataFrame) -> dict:
    """hcp_id -> sorted array of ordinal-day event_dates for every event that
    HCP attended as role='attendee', across ALL events (not just one).

    This powers the temporal-contamination check: a candidate control who
    themselves attended ANY other event within a treatment case's 6-month
    pre-period window was, during that window, a recently-exposed HCP rather
    than an unexposed comparator -- their own Rx behavior in that window may
    already be reacting to their own program attendance, which would bias the
    treatment-vs-control gap toward zero. This was previously undetected: the
    only exclusions were role-based (speaker status) and same-event, with no
    check on a candidate's OTHER attendance history at all.

    Built once as a global per-HCP lookup (not per treatment case) precisely
    so the per-pairing check in build_candidate_pool is a cheap bisect over an
    already-sorted array, rather than re-scanning attendance-final.csv for
    every (treatment, candidate) combination.

    Uses the authoritative event_date from events_preprocessed_final.csv
    (joined via event_id), not the attendance log's own attendance_date --
    the two should normally agree, but event_date is the field the task
    defines the contamination window against.
    """
    att = attendance.loc[attendance["role"] == "attendee", ["hcp_id", "event_id"]]
    att = att.merge(events[["event_id", "event_date"]], on="event_id", how="left")
    ordinals = pd.to_datetime(att["event_date"]).map(lambda d: d.toordinal())
    lookup = (pd.DataFrame({"hcp_id": att["hcp_id"], "ord": ordinals})
              .groupby("hcp_id")["ord"]
              .apply(lambda s: np.sort(s.to_numpy())).to_dict())

    n_events_per_hcp = pd.Series({k: len(v) for k, v in lookup.items()})
    print(f"    HCPs with >=1 own attendee-role event: {len(lookup):,d}")
    print(f"    own-events-attended per HCP: min={n_events_per_hcp.min()}  "
          f"mean={n_events_per_hcp.mean():.2f}  max={n_events_per_hcp.max()}")
    return lookup


def is_temporally_contaminated(hcp_id, window_start_ord: int, window_end_ord_excl: int,
                               attended_dates_lookup: dict) -> bool:
    """True if `hcp_id` attended (as role='attendee') any event whose own
    event_date falls in [window_start_ord, window_end_ord_excl) -- i.e. inside
    the specific treatment case's pre-period window being evaluated.

    O(log k) via bisect over that HCP's own sorted attended-dates array
    (k = events that one HCP attended, typically single digits), rather than
    a linear scan -- called up to ~4,320 x (pool size) times, so this is the
    hot path the task's performance note calls out.
    """
    dates = attended_dates_lookup.get(hcp_id)
    if dates is None or len(dates) == 0:
        return False
    lo = bisect.bisect_left(dates, window_start_ord)
    hi = bisect.bisect_left(dates, window_end_ord_excl)
    return hi > lo


# ==============================================================================
# STEP 3 -- pre-period baselines for the whole population, batched by event
# ==============================================================================

def compute_all_baselines(data: dict, events_idx: pd.DataFrame, rx_matrix, row_lookup) -> dict:
    """Baseline for EVERY (hcp_id, event) combo, computed once and reused.

    Rather than recomputing a candidate's baseline separately for each
    treatment case that might consider them, we compute it once per
    (event, hcp) pair -- since the window and target category are fixed per
    event, every treatment case tied to that event (and every candidate ever
    considered for it) shares the same underlying lookup. This turns what
    would be an O(treatment_cases x pool_size) recomputation into a single
    O(hcp_population x events) batch pass.
    """
    step("STEP 3  --  Computing pre-period Rx baselines (batched by event)")

    hcp_ids = data["hcp"]["hcp_id"].to_numpy()
    ev = events_idx[["event_id", "event_idx", "target_ndc_category"]].drop_duplicates()

    # cross join: every HCP x every event
    pairs = (pd.DataFrame({"hcp_id": hcp_ids}).assign(_k=1)
             .merge(ev.assign(_k=1), on="_k").drop(columns="_k"))
    pairs = pairs.rename(columns={"target_ndc_category": "ndc_category"})

    pairs["pre_period_rx_baseline"] = compute_pre_period_baselines(
        pairs, rx_matrix, row_lookup, WINDOW_MONTHS)

    n_pairs = len(pairs)
    n_valid = pairs["pre_period_rx_baseline"].notna().sum()
    print(f"    (hcp, event) pairs evaluated : {n_pairs:,d}")
    print(f"    pairs with a computable baseline: {n_valid:,d}  ({n_valid / n_pairs:.1%})")
    print(f"    pairs with zero rx rows in window (dropped): {n_pairs - n_valid:,d}")

    baseline_lookup = {
        (row.event_id, row.hcp_id): row.pre_period_rx_baseline
        for row in pairs.itertuples()
    }
    valid_vals = pairs["pre_period_rx_baseline"].dropna()
    print(f"    baseline distribution (all valid pairs): "
          f"min={valid_vals.min():,.1f}  max={valid_vals.max():,.1f}  "
          f"mean={valid_vals.mean():,.1f}  median={valid_vals.median():,.1f}")

    return {"lookup": baseline_lookup, "all_pairs": pairs}


# ==============================================================================
# GLOBAL feature scaling (fit once, reused by every treatment case's pool)
# ==============================================================================

def build_static_features(hcp: pd.DataFrame) -> pd.DataFrame:
    """Z-score the 3 continuous covariates and one-hot the 2 categorical ones,
    fit on the FULL HCP population (not per treatment-case pool).

    Fitting per-pool would make z-scores incomparable across treatment cases
    (a pool of 6 candidates and a pool of 90 would each get their own mean/std,
    so a "distance of 0.5" would mean different things in different pools) --
    exactly the instability the task calls out. One global fit gives every
    treatment case's pool the same yardstick.
    """
    step("Building GLOBAL standardized feature table (static covariates)")

    scaler = StandardScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(hcp[STATIC_CONTINUOUS_COLS]),
        columns=[f"{c}_z" for c in STATIC_CONTINUOUS_COLS],
        index=hcp["hcp_id"],
    )
    dummies = pd.get_dummies(
        hcp.set_index("hcp_id")[STATIC_CATEGORICAL_COLS],
        columns=STATIC_CATEGORICAL_COLS,
    ).astype(float)

    features = scaled.join(dummies)
    print(f"    continuous (z-scored): {STATIC_CONTINUOUS_COLS}")
    print(f"    categorical (one-hot): {STATIC_CATEGORICAL_COLS} "
          f"-> {dummies.shape[1]} dummy columns")
    print(f"    feature table shape  : {features.shape}  "
          f"(fit on all {len(hcp):,d} HCPs)")
    return features


def fit_global_baseline_scaler(all_pairs: pd.DataFrame) -> tuple[float, float]:
    """Global mean/std of pre_period_rx_baseline across every computed (hcp,
    event) pair -- the same "fit globally" rationale as build_static_features,
    applied to the one covariate that is event-specific rather than static.
    """
    vals = all_pairs["pre_period_rx_baseline"].dropna().to_numpy()
    mean, std = float(vals.mean()), float(vals.std())
    if std == 0:
        std = 1.0
    print(f"    global pre_period_rx_baseline scaler: mean={mean:,.2f}  std={std:,.2f}")
    return mean, std


# ==============================================================================
# STEP 4 -- hierarchical hard filter (per treatment case)
# ==============================================================================

def build_candidate_pool(treatment_row, hcp_by_id: pd.DataFrame,
                         exclusions: dict, event_attendee_ids: dict,
                         baseline_lookup: dict, attended_dates_lookup: dict) -> tuple[list, bool]:
    """Candidate pool for one treatment case.

    Exclusions applied, in order:
      1. globally speaker-flagged / speaker-role HCPs (Step 2a) -- can never
         be a control anywhere.
      2. the treatment HCP itself.
      3. anyone who attended THIS SPECIFIC event as an attendee (co-attendees,
         including the treatment HCP again) -- they were themselves exposed to
         this exact program, so they cannot serve as an unexposed comparator
         for it. Note this does NOT disqualify them as a control for a
         DIFFERENT event -- only same-event self-exposure is excluded, per the
         shared spec all 4 matching methods follow.
      4. candidates with no computable pre-period baseline for this event's
         (window, target category) -- i.e. they don't prescribe the drug being
         promoted, so no valid pre-period comparison exists (Step 3).
      5. TEMPORAL CONTAMINATION (Step 2b, applied per-pairing): candidates who
         themselves attended ANY OTHER event, as an attendee, whose event_date
         falls inside THIS treatment case's own pre-period window
         [event_date - window_months, event_date). A candidate can be a valid
         control for one treatment case and contaminated for another,
         depending on how their own attendance history lines up against each
         case's specific window -- which is exactly why this can't be a
         single global exclusion set like #1, and has to be evaluated fresh
         per (treatment, candidate) pairing.
    Then the hierarchical hard filter: specialty+region first; if the
    resulting pool is smaller than MIN_POOL_SIZE, relax to specialty-only
    (region dropped, specialty always kept) and flag pool_relaxed=True.
    """
    event_id = treatment_row.event_id
    same_event_ids = event_attendee_ids.get(event_id, set())
    hard_excluded = exclusions["globally_excluded"] | same_event_ids | {treatment_row.treatment_hcp_id}

    window_start_ord = treatment_row.window_start_ord
    window_end_ord = treatment_row.event_date_ord  # exclusive upper bound

    def eligible_with_baseline(pool_df: pd.DataFrame) -> list:
        ids = [h for h in pool_df["hcp_id"] if h not in hard_excluded]
        ids = [h for h in ids if not pd.isna(baseline_lookup.get((event_id, h)))]
        ids = [h for h in ids if not is_temporally_contaminated(
            h, window_start_ord, window_end_ord, attended_dates_lookup)]
        return ids

    specialty_region_pool = hcp_by_id[
        (hcp_by_id["specialty"] == treatment_row.specialty) &
        (hcp_by_id["region"] == treatment_row.region)]
    pool = eligible_with_baseline(specialty_region_pool)
    pool_relaxed = False

    if len(pool) < MIN_POOL_SIZE:
        specialty_only_pool = hcp_by_id[hcp_by_id["specialty"] == treatment_row.specialty]
        relaxed_pool = eligible_with_baseline(specialty_only_pool)
        # Only adopt the relaxed pool if it's actually bigger -- relaxing
        # region can't shrink an already-region-matched pool, but guards
        # against a degenerate no-op relax still being labeled "relaxed".
        if len(relaxed_pool) > len(pool):
            pool = relaxed_pool
            pool_relaxed = True

    return sorted(pool), pool_relaxed  # sorted for deterministic NN tie-breaking


# ==============================================================================
# STEP 5 -- nearest neighbor ranking (per treatment case)
# ==============================================================================

def rank_candidates_by_distance(treatment_row, candidate_ids: list,
                                static_features: pd.DataFrame,
                                baseline_lookup: dict, baseline_mean: float,
                                baseline_std: float) -> list[tuple]:
    """Full ascending-distance ranking of `candidate_ids` against the
    treatment HCP, using sklearn NearestNeighbors (Euclidean) over:
      pre_period_rx_baseline (z, event-specific) + patient_volume_monthly (z)
      + baseline_rx_volume_monthly (z) + years_in_practice (z)
      + one-hot sub_specialty + one-hot practice_setting (all globally fit).

    Returns the FULL ranked list (not just the top-1) so Step 6's greedy
    assignment can fall back to the 2nd, 3rd, ... nearest candidate when a
    higher-priority treatment case has already claimed the nearest one.
    """
    if not candidate_ids:
        return []

    event_id = treatment_row.event_id

    def vector_for(hcp_id):
        raw_baseline = baseline_lookup[(event_id, hcp_id)]
        z_baseline = (raw_baseline - baseline_mean) / baseline_std
        return np.concatenate(([z_baseline], static_features.loc[hcp_id].to_numpy()))

    treatment_vec = vector_for(treatment_row.treatment_hcp_id).reshape(1, -1)
    candidate_matrix = np.vstack([vector_for(c) for c in candidate_ids])

    nn = NearestNeighbors(n_neighbors=len(candidate_ids), metric="euclidean")
    nn.fit(candidate_matrix)
    distances, indices = nn.kneighbors(treatment_vec)

    return [(candidate_ids[i], float(d))
            for d, i in zip(distances[0], indices[0])]


# ==============================================================================
# STEP 6 -- global greedy one-to-one assignment without replacement
# ==============================================================================

def assign_matches_greedy(cohort: pd.DataFrame, ranked_lists: dict) -> pd.DataFrame:
    """Sorts ALL treatment cases by their own best (nearest) candidate distance
    ascending, then assigns greedily in that order so the globally strongest
    matches lock in their first choice before weaker treatment cases compete
    for what's left.

    Control "used" tracking is scoped to (event_id, control_hcp_id), NOT to
    control_hcp_id alone. This is a deliberate methodological choice, not an
    oversight:

      * WITHOUT replacement WITHIN the same event_id -- two treatment HCPs who
        attended the same speaker program should not be handed the same
        control HCP. They are being compared side-by-side against the same
        event, and letting one control stand in for two different attendees
        of that event would understate how thin the true comparison pool is
        for that program.

      * WITH replacement ACROSS different event_ids -- a control is not a
        physical resource that gets "consumed"; each (treatment, control)
        pairing computes its OWN fresh pre_period_rx_baseline over that
        specific event's 6-month pre-window (Step 3). A control who is a good
        covariate match at the time of Event A in March 2023 is being asked
        an entirely independent question -- "were you a reasonable stand-in
        at THIS point in time?" -- when reused for Event B in November 2024.
        Treating every event_id as drawing from its own copy of the control
        pool is what makes 4,320 treatment cases matchable against a ~960-HCP
        pool at all; the earlier global-uniqueness version instead exhausted
        the pool after ~960 assignments regardless of match quality, because
        it conflated "used for one comparison" with "used up everywhere."

    A single control CAN still end up anchoring many treatment cases across
    many different events -- see the "controls reused across N events" QA
    check in Step 8, which exists specifically to flag whether that reuse
    is getting so wide (e.g. one HCP standing in for 40 different events)
    that it stops looking like an independent comparison.
    """
    step("STEP 6  --  Greedy one-to-one assignment (no replacement PER EVENT)")

    best_distance = {}
    for idx, ranked in ranked_lists.items():
        best_distance[idx] = ranked[0][1] if ranked else np.inf

    priority_order = sorted(
        cohort.index,
        key=lambda i: (best_distance.get(i, np.inf), cohort.at[i, "treatment_hcp_id"]),
    )

    used_controls_by_event: dict = {}  # event_id -> set of control_hcp_id already claimed
    assignment = {}  # idx -> dict of result fields

    for idx in priority_order:
        event_id = cohort.at[idx, "event_id"]
        used_this_event = used_controls_by_event.setdefault(event_id, set())

        ranked = ranked_lists.get(idx, [])
        if not ranked:
            # No candidate pool at all (empty even after relax) -- distinct
            # from "pool existed but every member got claimed first".
            reason = ("no_treatment_baseline" if cohort.at[idx, "no_treatment_baseline_flag"]
                      else "empty_pool")
            assignment[idx] = {"control_hcp_id": pd.NA, "is_matched": False,
                               "match_rank": pd.NA, "distance_score": np.nan,
                               "_unmatched_reason": reason}
            continue

        chosen = None
        for rank, (control_id, dist) in enumerate(ranked, start=1):
            if control_id not in used_this_event:
                chosen = (control_id, dist, rank)
                break

        if chosen is None:
            assignment[idx] = {"control_hcp_id": pd.NA, "is_matched": False,
                               "match_rank": pd.NA, "distance_score": np.nan,
                               "_unmatched_reason": "pool_exhausted"}
        else:
            control_id, dist, rank = chosen
            used_this_event.add(control_id)
            assignment[idx] = {"control_hcp_id": control_id, "is_matched": True,
                               "match_rank": rank, "distance_score": dist,
                               "_unmatched_reason": None}

    result = pd.DataFrame.from_dict(assignment, orient="index")
    n_matched = result["is_matched"].sum()
    n_unique_controls = result.loc[result["is_matched"], "control_hcp_id"].nunique()
    n_control_event_pairs = sum(len(s) for s in used_controls_by_event.values())
    print(f"    treatment cases processed        : {len(cohort):,d}")
    print(f"    matched                          : {n_matched:,d}")
    print(f"    unmatched                        : {len(cohort) - n_matched:,d}")
    print(f"    distinct controls used (any event): {n_unique_controls:,d}")
    print(f"    (control, event) claims made      : {n_control_event_pairs:,d}  "
          f"(this is what 'used without replacement' now counts against, "
          f"not the {n_unique_controls:,d} distinct-control figure above)")
    return result


# ==============================================================================
# STEP 7 -- assemble final output, exact schema/column order
# ==============================================================================

def assemble_output(cohort: pd.DataFrame, assignment: pd.DataFrame,
                    pool_relaxed_map: dict, baseline_lookup: dict) -> pd.DataFrame:
    step("STEP 7  --  Assembling final output (exact shared schema)")

    out = cohort.join(assignment)
    out["pool_relaxed"] = out.index.map(lambda i: pool_relaxed_map.get(i, False))

    out["control_pre_period_rx_baseline"] = [
        baseline_lookup.get((row.event_id, row.control_hcp_id), np.nan)
        if pd.notna(row.control_hcp_id) else np.nan
        for row in out.itertuples()
    ]

    # NOTE on `method`: the task text has one line implying `method` should be
    # left blank for unmatched rows, but Step 7's schema section explicitly
    # states method="nearest_neighbor" "for every row" -- since Step 7 defines
    # the authoritative final contract other teams' outputs must line up
    # against, we follow that literal instruction: `method` records which
    # ALGORITHM was used to attempt the match, independent of whether it
    # succeeded, so it is populated on every row including unmatched ones.
    out["method"] = METHOD_NAME

    out["event_date"] = pd.to_datetime(out["event_date"]).dt.strftime("%Y-%m-%d")
    out["control_hcp_id"] = out["control_hcp_id"].astype("Int64")
    out["match_rank"] = out["match_rank"].astype("Int64")
    out["is_matched"] = out["is_matched"].astype(bool)
    out["pool_relaxed"] = out["pool_relaxed"].astype(bool)

    final = out[OUTPUT_COLUMNS].copy()

    n_dupes = final.duplicated(subset=["treatment_hcp_id", "event_id"]).sum()
    print(f"    output rows              : {len(final):,d}")
    print(f"    duplicate treatment cases: {n_dupes}  (must be 0)")
    if n_dupes:
        raise AssertionError(f"{n_dupes} duplicate (treatment_hcp_id, event_id) "
                             "rows in output -- one row per treatment case required")

    return final, out  # `out` (with internal _unmatched_reason) kept for QA


# ==============================================================================
# STEP 8 -- QA / validation report (console only, never written to the CSV)
# ==============================================================================

def run_qa_report(final: pd.DataFrame, out_internal: pd.DataFrame,
                  exclusions: dict, attended_dates_lookup: dict,
                  previous_output) -> None:
    banner("STEP 8  --  VALIDATION / QA REPORT")

    n_total = len(final)
    n_matched = int(final["is_matched"].sum())
    n_unmatched = n_total - n_matched

    # ---- (1) counts -- reported exactly, no rounding up to a nicer number ---
    print(f"\n(1) Match rate")
    print(f"    total treatment cases : {n_total:,d}")
    print(f"    matched                : {n_matched:,d}  ({n_matched / n_total:.4%})")
    print(f"    unmatched               : {n_unmatched:,d}  ({n_unmatched / n_total:.4%})")

    # ---- (2) before/after comparison against the pre-fix run ----------------
    print(f"\n(2) Before/after comparison (temporal-contamination fix)")
    if previous_output is None:
        print("    no previous-run snapshot available -- before/after diff skipped")
    else:
        prev_matched = int(previous_output["is_matched"].sum())
        prev_total = len(previous_output)
        print(f"    BEFORE this fix : {prev_matched:,d} / {prev_total:,d} matched "
              f"({prev_matched / prev_total:.4%})")
        print(f"    AFTER  this fix : {n_matched:,d} / {n_total:,d} matched "
              f"({n_matched / n_total:.4%})")
        prev_keyed = previous_output[["treatment_hcp_id", "event_id", "is_matched"]].rename(
            columns={"is_matched": "was_matched"})
        cur_keyed = final[["treatment_hcp_id", "event_id", "is_matched"]]
        merged = prev_keyed.merge(cur_keyed, on=["treatment_hcp_id", "event_id"], how="inner")
        if len(merged) != n_total:
            print(f"    NOTE: previous snapshot join matched {len(merged):,d} of "
                  f"{n_total:,d} current rows -- comparing on overlap only")
        flipped_to_unmatched = merged[merged["was_matched"] & ~merged["is_matched"]]
        flipped_to_matched = merged[~merged["was_matched"] & merged["is_matched"]]
        print(f"    cases flipped True -> False (lost solely to the contamination "
              f"filter): {len(flipped_to_unmatched):,d}")
        print(f"    cases flipped False -> True (unexpected -- pool can only shrink "
              f"from this fix): {len(flipped_to_matched):,d}")
        if len(flipped_to_matched):
            print(f"    WARNING: {len(flipped_to_matched)} cases newly matched after "
                  f"a fix that only ADDS exclusions -- investigate before trusting "
                  f"this run (possible non-determinism or an unrelated code change)")

    # ---- (3) TEMPORAL CONTAMINATION -- independent re-audit, hard assertion -
    # Recomputed directly from `attended_dates_lookup` (the raw fact table)
    # against each matched row's ACTUAL assigned control + window, entirely
    # bypassing build_candidate_pool's own bookkeeping -- this checks the
    # OUTPUT, not whether the filter code merely believes it worked. Mirrors
    # exactly how this contamination bug was originally found.
    print(f"\n(3) Temporal contamination check (independent re-audit, hard assertion)")
    matched_rows = out_internal[out_internal["is_matched"]]
    contaminated_mask = [
        is_temporally_contaminated(row.control_hcp_id, row.window_start_ord,
                                   row.event_date_ord, attended_dates_lookup)
        for row in matched_rows.itertuples()
    ]
    n_contaminated = int(sum(contaminated_mask))
    print(f"    matched rows re-checked : {len(matched_rows):,d}")
    print(f"    contaminated (control's own attendance falls inside the "
          f"assigned pre-period window): {n_contaminated:,d}")
    if n_contaminated:
        bad = matched_rows.loc[contaminated_mask,
                               ["treatment_hcp_id", "control_hcp_id", "event_id"]]
        raise AssertionError(
            f"{n_contaminated} matched rows have a temporally contaminated "
            f"control -- the Step 4 filter is leaking. First few:\n"
            f"{bad.head(5).to_string(index=False)}")
    print("    [PASS] zero matched controls have contaminating attendance "
          "inside their assigned pre-period window")

    # ---- (4) zero speaker leakage into control_hcp_id (hard assert) -----
    print(f"\n(4) Speaker-leakage check (hard assertion)")
    used_controls = set(final.loc[final["is_matched"], "control_hcp_id"].dropna().astype(int))
    leaked = used_controls & exclusions["globally_excluded"]
    print(f"    distinct controls used : {len(used_controls):,d}")
    print(f"    of which speaker-flagged or speaker-role: {len(leaked):,d}")
    if leaked:
        raise AssertionError(
            f"{len(leaked)} speaker-flagged/speaker-role HCPs appear as "
            f"control_hcp_id: {sorted(leaked)[:10]}{'...' if len(leaked) > 10 else ''}")
    print("    [PASS] zero speakers found among assigned controls")

    # ---- (5) specialty match rate (hard assert, ~100% expected) ---------
    # Performed in main(), before this function is called, since it needs a
    # fresh hcp_id -> specialty lookup that isn't otherwise threaded through
    # here. Printed there; noted again for report continuity.
    print(f"\n(5) Specialty match rate -- see hard-assertion check printed above "
          f"run_qa_report (hierarchical specialty filter integrity)")

    # ---- (6) pool_relaxed breakdown --------------------------------------
    print(f"\n(6) pool_relaxed breakdown (thin-pool cells needing region relax)")
    n_relaxed = int(final["pool_relaxed"].sum())
    print(f"    rows with pool_relaxed=True : {n_relaxed:,d}  ({n_relaxed / n_total:.1%})")
    breakdown = (final.assign(region=out_internal["region"].to_numpy())
                 .groupby(["specialty", "region"])["pool_relaxed"]
                 .agg(["sum", "count"]))
    breakdown["pct"] = breakdown["sum"] / breakdown["count"]
    breakdown = breakdown[breakdown["sum"] > 0].sort_values("sum", ascending=False)
    if len(breakdown):
        print(breakdown.rename(columns={"sum": "n_relaxed", "count": "n_cases"}).to_string())
    else:
        print("    no specialty x region cell required relaxing")

    # ---- (7) distance_score distribution + weak-match flagging -----------
    print(f"\n(7) distance_score distribution (matched pairs only)")
    d = final.loc[final["is_matched"], "distance_score"]
    print(f"    min={d.min():.4f}  mean={d.mean():.4f}  median={d.median():.4f}  max={d.max():.4f}")
    threshold = np.percentile(d, WEAK_MATCH_PERCENTILE)
    weak = final[(final["is_matched"]) & (final["distance_score"] >= threshold)]
    print(f"    weak-match threshold (>= p{WEAK_MATCH_PERCENTILE}) = {threshold:.4f}")
    print(f"    flagged weak matches: {len(weak):,d}  ({len(weak) / n_matched:.1%} of matched)")
    if len(weak):
        cols = ["treatment_hcp_id", "control_hcp_id", "specialty", "region", "distance_score"]
        print(weak[cols].sort_values("distance_score", ascending=False).head(10).to_string(index=False))

    # ---- (8) unmatched breakdown by reason --------------------------------
    print(f"\n(8) Unmatched treatment cases by reason")
    reasons = out_internal.loc[~final["is_matched"], "_unmatched_reason"]
    bucketed = reasons.map(
        lambda r: "no_rx_history" if r == "no_treatment_baseline" else "pool_exhausted")
    counts = bucketed.value_counts()
    for reason in ["no_rx_history", "pool_exhausted"]:
        n = int(counts.get(reason, 0))
        print(f"    {reason:<16s} : {n:>5,d}  ({n / n_total:.1%} of all treatment cases)")

    # ---- (9) cross-event control reuse --------------------------------------
    # Step 6 now allows the SAME control to anchor treatment cases tied to
    # DIFFERENT events (each gets its own fresh pre-period baseline, so reuse
    # across events is methodologically fine -- see the Step 6 docstring).
    # But reuse can still get too wide: a control who ends up standing in for
    # 40 different events is a much weaker, less independent comparator than
    # one reused across 2 -- this check surfaces that so it can be judged,
    # not just assumed benign because the assertions above all pass.
    print(f"\n(9) Cross-event control reuse (controls CAN legitimately serve "
          f"multiple events; this quantifies how much)")
    reuse = (final.loc[final["is_matched"], ["control_hcp_id", "event_id"]]
             .drop_duplicates()
             .groupby("control_hcp_id")["event_id"].nunique())
    print(f"    distinct controls used         : {len(reuse):,d}")
    print(f"    events reused across -- min={reuse.min()}  mean={reuse.mean():.2f}  "
          f"median={reuse.median():.1f}  max={reuse.max()}")
    print(f"    controls used for exactly 1 event: {(reuse == 1).sum():,d}  "
          f"({(reuse == 1).mean():.1%} of controls used)")
    heavy_threshold = max(int(np.percentile(reuse, 95)), 1)
    heavy = reuse[reuse >= heavy_threshold].sort_values(ascending=False)
    print(f"    heaviest-reused controls (>= p95 = {heavy_threshold} events):")
    if len(heavy):
        print("    " + heavy.head(10).to_string().replace("\n", "\n    "))
    else:
        print("    none -- reuse is evenly spread")


def load_previous_output_snapshot():
    """Best-effort load of a PRIOR run's output CSV, purely to power the
    honest before/after comparison in Step 8 -- never used for computation.
    Must be called BEFORE Step 9 overwrites OUTPUT_FILE, or this would just
    read back the run currently in progress.

    Tries the current OUTPUT_FILE first, then the two legacy locations this
    file has lived in as the project folder was reorganised (project root,
    then preprocessed_data/). Returns None, not an error, if none exists --
    a missing snapshot degrades the QA report by one section, it isn't fatal.
    """
    for path in (OUTPUT_FILE,
                 MATCHED_PAIRS_DIR / "nnm_matching" / "NNM_matched_pairs.csv",
                 PROJECT_ROOT / "NNM_matched_pairs.csv",
                 INPUT_DIR / "NNM_matched_pairs.csv"):
        if path.exists():
            print(f"    previous-run snapshot for before/after comparison: {path}")
            return pd.read_csv(path)
    print("    no previous-run snapshot found at either candidate path -- "
          "before/after comparison will be skipped")
    return None


# ==============================================================================
# STEP 9 -- export
# ==============================================================================

def export_results(final: pd.DataFrame, n_expected: int) -> None:
    banner("STEP 9  --  EXPORT")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final.to_csv(OUTPUT_FILE, index=False, na_rep="")
    print(f"    written to: {OUTPUT_FILE}")
    print(f"    rows written: {len(final):,d}")
    print(f"    expected (= treatment/attendee rows from Step 1): {n_expected:,d}")
    if len(final) != n_expected:
        raise AssertionError(
            f"Row count mismatch: wrote {len(final):,d} rows but Step 1 built "
            f"{n_expected:,d} treatment cases -- rows were dropped somewhere")
    print("    [PASS] output row count matches treatment cohort size exactly")

    # ---- control_hcp_id raw-text dtype spot check ----------------------------
    # Read the file back as PLAIN TEXT, not through pandas, since pd.read_csv()
    # without an explicit dtype hint infers float64 for any numeric column that
    # contains blanks -- that read-back inference is what makes a perfectly
    # clean on-disk column (nullable Int64 -> clean int-or-empty text via
    # to_csv) LOOK like a ".0"-suffixed float column when casually reloaded.
    # Checking the raw bytes on disk is the only way to tell a real write-time
    # bug apart from that read-side artifact.
    print(f"\n    control_hcp_id raw-text spot check (first 5 data rows, "
          f"read as bytes -- not through pandas):")
    with open(OUTPUT_FILE, "r", encoding="utf-8") as fh:
        header_cols = fh.readline().rstrip("\n").split(",")
        control_idx = header_cols.index("control_hcp_id")
        suspect = []
        for i in range(5):
            line = fh.readline()
            if not line:
                break
            field = line.rstrip("\n").split(",")[control_idx]
            clean = field == "" or field.isdigit()
            print(f"      row {i + 1}: control_hcp_id = {field!r:<16s} "
                  f"[{'OK' if clean else 'SUSPECT'}]")
            if not clean:
                suspect.append(field)
    if suspect:
        raise AssertionError(
            f"control_hcp_id contains non-integer text on disk: {suspect} -- "
            "the Int64 cast before to_csv() is not producing clean output")
    print("    [PASS] control_hcp_id is clean integer text (or a truly empty "
          "cell) on disk -- no '.0' suffixes, no scientific notation")


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> int:
    banner("NEAREST NEIGHBOR MATCHING -- Speaker Program ROI Analysis")

    data = load_data()

    step("Loading previous-run snapshot for the before/after comparison "
        "(must happen before Step 9 overwrites the output file)")
    previous_output = load_previous_output_snapshot()

    rx_matrix, row_lookup, month_to_idx, n_months = build_rx_lookup(data["rx"])

    events_idx = data["events"].copy()
    events_idx["event_idx"] = events_idx["event_month"].map(month_to_idx)
    if events_idx["event_idx"].isna().any():
        raise AssertionError("Some events' event_month falls outside the rx_claims "
                             "panel -- cannot compute a pre-period baseline for them")

    cohort = build_treatment_cohort(data, month_to_idx)
    n_treatment_cases = len(cohort)

    exclusions = build_control_exclusions(data)
    attended_dates_lookup = build_attended_event_dates(data["attendance"], data["events"])

    baselines = compute_all_baselines(data, events_idx, rx_matrix, row_lookup)
    baseline_lookup = baselines["lookup"]

    static_features = build_static_features(data["hcp"])
    baseline_mean, baseline_std = fit_global_baseline_scaler(baselines["all_pairs"])

    hcp_by_id = data["hcp"][["hcp_id", "specialty", "region"]]

    event_attendee_ids: dict = (
        data["attendance"][data["attendance"]["role"] == "attendee"]
        .groupby("event_id")["hcp_id"].apply(set).to_dict())

    # ---- own baseline for every treatment case (drives "no_treatment_baseline") ----
    cohort["own_pre_period_rx_baseline"] = [
        baseline_lookup.get((row.event_id, row.treatment_hcp_id), np.nan)
        for row in cohort.itertuples()
    ]
    cohort["no_treatment_baseline_flag"] = cohort["own_pre_period_rx_baseline"].isna()
    n_no_baseline = int(cohort["no_treatment_baseline_flag"].sum())

    # Precompute each treatment case's own pre-period window bounds ONCE,
    # vectorized, as ordinal-day integers -- so the temporal-contamination
    # check inside the per-row loop below is a plain int comparison via
    # bisect, not a per-row datetime re-parse (the loop already runs ~4,320
    # times; date parsing inside it would dominate the runtime for no reason).
    event_date_ts = pd.to_datetime(cohort["event_date"])
    cohort["event_date_ord"] = event_date_ts.map(lambda d: d.toordinal())
    cohort["window_start_ord"] = (
        event_date_ts - pd.DateOffset(months=WINDOW_MONTHS)).map(lambda d: d.toordinal())

    step("STEP 4/5  --  Candidate pools + nearest-neighbor ranking per treatment case")
    print(f"    treatment cases with no computable own baseline "
          f"(auto-unmatched): {n_no_baseline:,d}")

    ranked_lists: dict = {}
    pool_relaxed_map: dict = {}
    for row in cohort.itertuples():
        idx = row.Index
        if row.no_treatment_baseline_flag:
            ranked_lists[idx] = []
            pool_relaxed_map[idx] = False
            continue
        pool, relaxed = build_candidate_pool(
            row, hcp_by_id, exclusions, event_attendee_ids, baseline_lookup,
            attended_dates_lookup)
        pool_relaxed_map[idx] = relaxed
        ranked_lists[idx] = rank_candidates_by_distance(
            row, pool, static_features, baseline_lookup, baseline_mean, baseline_std)

    n_empty_pool = sum(1 for i, r in ranked_lists.items()
                       if not r and not cohort.at[i, "no_treatment_baseline_flag"])
    print(f"    treatment cases with an empty candidate pool (even after relax): "
          f"{n_empty_pool:,d}")
    print(f"    treatment cases with >=1 candidate            : "
          f"{sum(1 for r in ranked_lists.values() if r):,d}")

    # rename own baseline column to match Step 7 schema before assignment/output
    cohort["pre_period_rx_baseline"] = cohort["own_pre_period_rx_baseline"]

    assignment = assign_matches_greedy(cohort, ranked_lists)
    final, out_internal = assemble_output(cohort, assignment, pool_relaxed_map, baseline_lookup)

    # (3) specialty match-rate hard assertion, done here where both treatment
    # and control specialty are easy to align by hcp_id lookup.
    hcp_specialty = data["hcp"].set_index("hcp_id")["specialty"]
    matched_final = final[final["is_matched"]]
    control_specialty = matched_final["control_hcp_id"].astype(int).map(hcp_specialty)
    specialty_match_rate = (control_specialty.to_numpy() == matched_final["specialty"].to_numpy()).mean()
    print(f"\n[Specialty match-rate check] {specialty_match_rate:.2%} of matched pairs "
          f"have control.specialty == treatment.specialty")
    if specialty_match_rate < 0.999:
        raise AssertionError(
            f"Specialty match rate {specialty_match_rate:.2%} is below the ~100% "
            "expected from a hard specialty filter -- hierarchical filter is leaking")
    print("    [PASS] specialty hard-filter integrity confirmed")

    run_qa_report(final, out_internal, exclusions, attended_dates_lookup, previous_output)

    export_results(final, n_treatment_cases)

    banner("SUMMARY")
    print(f"    treatment cases   : {n_treatment_cases:,d}")
    print(f"    matched           : {int(final['is_matched'].sum()):,d}")
    print(f"    output file       : {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
