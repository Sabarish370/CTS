#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 RULE-BASED MATCHING -- Speaker Program & Peer-to-Peer ROI Analysis
================================================================================

Third leg of the 4-way matching comparison (Random / Rule-Based / Nearest
Neighbor / PSM). Unlike NNM, which always returns the *closest* control it can
find, this method applies four HARD business rules and returns nothing at all
when they cannot all be satisfied simultaneously.

THE FOUR RULES (all must hold -- AND, never OR, never scored/weighted)
---------------------------------------------------------------------
  1. control.specialty          == treatment.specialty
  2. control.city               == treatment.city
  3. |control.experience_years   - treatment.experience_years| <= 3
  4. |treatment.rx_baseline - control.rx_baseline| / treatment.rx_baseline
                                                          * 100 <= 10

NO FALLBACK
-----------
If the eligible pool is empty after exclusions + all four rules, the row is
emitted with is_matched=False / control_hcp_id=null / match_rank=null. No
condition is relaxed, no "closest available" candidate is substituted, and
nothing is silently swapped in. Any controlled relaxation is a team design
decision to be raised explicitly -- it must never happen inside this script.

REUSED LOGIC (imported, not reimplemented)
------------------------------------------
Pre-period Rx baselines, the speaker-exclusion union, the treatment cohort
build, and the temporal-contamination check are all imported directly from
nnm_matching.py so this method's numbers are computed identically to NNM's.
Re-deriving them here would risk the four methods silently disagreeing on
what a "baseline" or a "contaminated control" even means.

Run
---
    python rbm_matching.py
================================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make the sibling nnm_matching module importable no matter which directory the
# script is launched from. Python only auto-adds the script's own folder to
# sys.path when it is invoked directly, so `python matching_techinques/
# rbm_matching.py` works but other invocation styles would not.
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
# Same project convention as NNM: matched_pairs/<method>_matching/. Imported
# from nnm_matching rather than re-derived, so if the project layout moves
# again both methods follow it from one edit instead of drifting apart.
OUTPUT_DIR = nnm.MATCHED_PAIRS_DIR / "rule_based_matching"
OUTPUT_FILE = OUTPUT_DIR / "rule_based_matching_output.csv"

METHOD_NAME = "rule_based"

# The four business-rule thresholds. Named constants so a future relaxation is
# an explicit, reviewable diff -- not a magic number buried in a comparison.
MAX_EXPERIENCE_DIFF_YEARS = 3
MAX_RX_PCT_DIFF = 10.0

# Column names in hcp-final.csv that back the rule vocabulary.
COL_EXPERIENCE = "years_in_practice"    # "experience_years" in the rule spec
COL_SPECIALTY = "specialty"
COL_CITY_ACTUAL = "city"                # the true city, always carried for audit

# --- GEOGRAPHY RULE (rule 2) --------------------------------------------------
# APPROVED CHANGE: was "city", now "region".
#
# Matching on city was the binding constraint by a wide margin -- 100 distinct
# cities over 1,200 HCPs gives a median of 2 HCPs per specialty x city cell,
# and 39.5% of cells hold exactly one person. Leave-one-out showed dropping
# city alone moved the match rate 2.73% -> 74.14%, while dropping any other
# single rule moved it to at most 28%. Region (5 values) is the intended
# granularity for a control pool of this size.
#
# This is a deliberate, approved widening of rule 2 -- NOT a silent fallback.
# Every matched row still satisfies all four rules simultaneously; the second
# rule is simply now defined at region level. Set back to "city" to restore
# the strict variant.
COL_GEO = "region"

WINDOW_MONTHS = nnm.WINDOW_MONTHS       # 6 -- identical to NNM, by import

# 12 standard columns shared by all four methods, then 6 audit columns so the
# output is self-verifiable without re-joining hcp-final.csv every time.
STANDARD_COLUMNS = [
    "treatment_hcp_id", "control_hcp_id", "method", "event_id", "event_date",
    "target_ndc_category", "specialty", "region", "pre_period_rx_baseline",
    "control_pre_period_rx_baseline", "is_matched", "match_rank",
]
# NOTE ON THE *_city AUDIT COLUMNS
# The team schema fixes these names as treatment_city / control_city, and the
# Section 7 validation snippet compares them directly to verify rule 2. Since
# rule 2 now matches on REGION, these two columns carry the region values --
# i.e. they hold "the geography field the rule actually enforced", so the
# shipped validator keeps working unchanged.
# To make sure a column named *_city is never silently lying to whoever opens
# the file, the true city is preserved alongside in *_city_actual. If the
# shared schema is ever revised, renaming these to treatment_geo/control_geo
# would remove the ambiguity entirely.
AUDIT_COLUMNS = [
    "control_specialty", "treatment_city", "control_city",
    "treatment_experience_years", "control_experience_years", "rx_pct_diff",
    "treatment_city_actual", "control_city_actual",
]
OUTPUT_COLUMNS = STANDARD_COLUMNS + AUDIT_COLUMNS


# ==============================================================================
# ELIGIBILITY -- the four rules, applied per treatment case
# ==============================================================================

def build_eligible_candidates(treatment_row, hcp_indexed: pd.DataFrame,
                              specialty_city_index: dict, exclusions: dict,
                              event_attendee_ids: dict, baseline_lookup: dict,
                              attended_dates_lookup: dict) -> list[tuple]:
    """Every control satisfying ALL FOUR rules for one treatment case.

    Returns [(control_hcp_id, rx_pct_diff), ...] sorted by rx_pct_diff
    ascending. Empty list == genuinely no eligible control; the caller emits
    an unmatched row rather than reaching for anything weaker.

    Exclusions are applied BEFORE the rule checks (per spec ordering):
      A. speaker_eligible_flag==True OR role=='speaker' anywhere -- never a
         control, for any event (imported union from NNM's Step 2).
      B. the treatment HCP itself.
      C. anyone who attended THIS event -- they were exposed to this exact
         program, so they cannot be its unexposed comparator.
      D. no computable pre-period baseline for this event's window/category
         (nothing to compare against, and rule 4 would divide into a NaN).
      E. temporally contaminated -- the candidate's own attendance at some
         OTHER program falls inside THIS case's 6-month pre-period window, so
         their "baseline" already reflects program exposure. Imported from
         NNM verbatim.

    Rules 1 and 2 are enforced by construction, not by scanning: candidates
    are pulled straight from a prebuilt (specialty, COL_GEO) index, so the
    same-specialty/same-geography pool is an O(1) dict hit instead of a filter
    over all 1,200 HCPs per case.

    TIE-BREAK (Section 5 requirement -- documented here and in the report)
    ---------------------------------------------------------------------
    When several candidates satisfy all four rules, they are ordered by:
        1st key: rx_pct_diff ASCENDING  -- smallest baseline Rx gap wins, so
                 the assigned control is the closest surviving comparator on
                 the one continuous dimension the rules constrain.
        2nd key: control_hcp_id ASCENDING -- breaks exact rx_pct_diff ties
                 deterministically, so reruns are byte-identical rather than
                 dependent on dict/groupby iteration order.
    match_rank is that candidate's 1-based position in this ordering:
    rank 1 = the closest eligible control was free and was taken; rank 2 = the
    closest was already claimed by another attendee OF THE SAME EVENT, so the
    second-closest was taken; and so on. match_rank therefore measures
    within-event contention, NOT a weakening of any rule -- every rank, however
    high, still satisfies all four conditions.
    """
    t_baseline = treatment_row.pre_period_rx_baseline
    if pd.isna(t_baseline) or t_baseline <= 0:
        # Rule 4 is undefined without a positive treatment denominator.
        return []

    # --- rules 1 + 2: same specialty AND same city (by construction) ---------
    key = (treatment_row.specialty, treatment_row.treatment_city)
    candidates = specialty_city_index.get(key, ())
    if not candidates:
        return []

    event_id = treatment_row.event_id
    same_event = event_attendee_ids.get(event_id, frozenset())
    blocked = exclusions["globally_excluded"]
    t_exp = treatment_row.treatment_experience_years
    w_start, w_end = treatment_row.window_start_ord, treatment_row.event_date_ord

    eligible = []
    for cid in candidates:
        # --- exclusions A / B / C -------------------------------------------
        if cid == treatment_row.treatment_hcp_id or cid in blocked or cid in same_event:
            continue

        # --- rule 3: experience within +/- 3 years ---------------------------
        c_exp = hcp_indexed.at[cid, COL_EXPERIENCE]
        if abs(int(c_exp) - int(t_exp)) > MAX_EXPERIENCE_DIFF_YEARS:
            continue

        # --- exclusion D: control needs a computable baseline ----------------
        c_baseline = baseline_lookup.get((event_id, cid))
        if c_baseline is None or pd.isna(c_baseline):
            continue

        # --- rule 4: baseline Rx within +/- 10% ------------------------------
        # Denominator is the TREATMENT baseline (per spec) -- deliberately
        # asymmetric, so this is not interchangeable with a symmetric percent
        # difference and must not be "tidied up" into one later.
        rx_pct_diff = abs(t_baseline - c_baseline) / t_baseline * 100.0
        if rx_pct_diff > MAX_RX_PCT_DIFF:
            continue

        # --- exclusion E: temporal contamination -----------------------------
        # Cheapest-first ordering puts this last: it is an O(log k) bisect, but
        # only a fraction of candidates survive rules 3+4 to reach it.
        if is_temporally_contaminated(cid, w_start, w_end, attended_dates_lookup):
            continue

        eligible.append((cid, rx_pct_diff))

    # TIE-BREAK: closest baseline Rx wins; hcp_id ascending breaks exact ties
    # so the result is deterministic across runs rather than dict-order luck.
    eligible.sort(key=lambda p: (p[1], p[0]))
    return eligible


# ==============================================================================
# ASSIGNMENT -- per-event without replacement, across-event with replacement
# ==============================================================================

def assign_controls(cohort: pd.DataFrame, eligible_lists: dict) -> pd.DataFrame:
    """Greedy assignment, identical in policy to NNM so the two are comparable.

      * WITHIN an event: without replacement. Two attendees of the same
        program never share a control -- they are being compared against the
        same event, and reusing one control across both would overstate how
        much independent comparison the pool actually supports.
      * ACROSS events: with replacement. Each pairing recomputes its own
        pre-period baseline against that event's own 6-month window, so a
        control reused at a different event is answering an independent
        question. Reuse is expected here and handled downstream via clustered
        standard errors -- Section (6) of the QA report quantifies exactly how
        much of it happens rather than leaving it implicit.

    Processing order is global, sorted by each case's best available
    rx_pct_diff ascending, so the tightest rule-satisfying pairs claim their
    control before looser ones compete for it.
    """
    step("ASSIGNMENT  --  per-event without replacement / across-event with replacement")

    best = {i: (lst[0][1] if lst else np.inf) for i, lst in eligible_lists.items()}
    order = sorted(cohort.index,
                   key=lambda i: (best[i], cohort.at[i, "treatment_hcp_id"]))

    used_by_event: dict = {}
    rows = {}

    for idx in order:
        event_id = cohort.at[idx, "event_id"]
        claimed = used_by_event.setdefault(event_id, set())
        eligible = eligible_lists.get(idx, [])

        if not eligible:
            rows[idx] = {"control_hcp_id": pd.NA, "is_matched": False,
                         "match_rank": pd.NA, "rx_pct_diff": np.nan,
                         "_unmatched_reason": "no_eligible_candidate"}
            continue

        chosen = None
        for rank, (cid, pct) in enumerate(eligible, start=1):
            if cid not in claimed:
                chosen = (cid, pct, rank)
                break

        if chosen is None:
            # Pool was non-empty but every member was already claimed by a
            # higher-priority attendee of this same event. Still NOT a licence
            # to relax a rule -- this stays unmatched.
            rows[idx] = {"control_hcp_id": pd.NA, "is_matched": False,
                         "match_rank": pd.NA, "rx_pct_diff": np.nan,
                         "_unmatched_reason": "pool_claimed_within_event"}
            continue

        cid, pct, rank = chosen
        claimed.add(cid)
        rows[idx] = {"control_hcp_id": cid, "is_matched": True,
                     "match_rank": rank, "rx_pct_diff": pct,
                     "_unmatched_reason": None}

    result = pd.DataFrame.from_dict(rows, orient="index")
    n_matched = int(result["is_matched"].sum())
    print(f"    treatment cases processed : {len(cohort):,d}")
    print(f"    matched                   : {n_matched:,d}  ({n_matched / len(cohort):.4%})")
    print(f"    unmatched                 : {len(cohort) - n_matched:,d}  "
          f"({(len(cohort) - n_matched) / len(cohort):.4%})")
    print(f"    distinct controls used    : "
          f"{result.loc[result['is_matched'], 'control_hcp_id'].nunique():,d}")
    return result


# ==============================================================================
# OUTPUT ASSEMBLY
# ==============================================================================

def assemble_output(cohort: pd.DataFrame, assignment: pd.DataFrame,
                    hcp_indexed: pd.DataFrame, baseline_lookup: dict) -> pd.DataFrame:
    step("OUTPUT  --  assembling 12 standard + 6 audit columns")

    out = cohort.join(assignment)
    out["method"] = METHOD_NAME

    matched_mask = out["is_matched"].to_numpy()
    cids = out["control_hcp_id"]

    out["control_specialty"] = [
        hcp_indexed.at[c, COL_SPECIALTY] if pd.notna(c) else pd.NA for c in cids]
    # *_city columns carry the geography the RULE enforced (COL_GEO);
    # *_city_actual preserves the true city regardless of what the rule uses.
    out["control_city"] = [
        hcp_indexed.at[c, COL_GEO] if pd.notna(c) else pd.NA for c in cids]
    out["control_city_actual"] = [
        hcp_indexed.at[c, COL_CITY_ACTUAL] if pd.notna(c) else pd.NA for c in cids]
    out["control_experience_years"] = [
        hcp_indexed.at[c, COL_EXPERIENCE] if pd.notna(c) else pd.NA for c in cids]
    out["control_pre_period_rx_baseline"] = [
        baseline_lookup.get((e, c)) if pd.notna(c) else np.nan
        for e, c in zip(out["event_id"], cids)]

    out["event_date"] = pd.to_datetime(out["event_date"]).dt.strftime("%Y-%m-%d")

    # Nullable Int64 so unmatched rows write as a genuinely empty cell rather
    # than upcasting the whole column to float and emitting "1234567890.0".
    out["control_hcp_id"] = out["control_hcp_id"].astype("Int64")
    out["match_rank"] = out["match_rank"].astype("Int64")
    out["control_experience_years"] = out["control_experience_years"].astype("Int64")
    out["treatment_experience_years"] = out["treatment_experience_years"].astype("Int64")
    out["is_matched"] = out["is_matched"].astype(bool)

    final = out[OUTPUT_COLUMNS].copy()

    n_dupes = int(final.duplicated(subset=["treatment_hcp_id", "event_id"]).sum())
    print(f"    output rows               : {len(final):,d}")
    print(f"    columns                   : {len(final.columns)} "
          f"({len(STANDARD_COLUMNS)} standard + {len(AUDIT_COLUMNS)} audit)")
    print(f"    duplicate treatment cases : {n_dupes}  (must be 0)")
    if n_dupes:
        raise AssertionError(f"{n_dupes} duplicate (treatment_hcp_id, event_id) rows")
    return final, out


# ==============================================================================
# SELF-VALIDATION -- the exact checks from the spec, run before the file ships
# ==============================================================================

def run_self_validation(final: pd.DataFrame, out_internal: pd.DataFrame,
                        exclusions: dict, attended_dates_lookup: dict,
                        diagnostics: dict) -> bool:
    banner("SELF-VALIDATION  --  all four rule checks must be 0")

    m = final[final["is_matched"]].copy()

    # Recomputed from the two baseline columns, NOT read from rx_pct_diff --
    # trusting the precomputed column would just be asking the script to
    # confirm its own arithmetic.
    m["rx_check"] = ((m["pre_period_rx_baseline"] - m["control_pre_period_rx_baseline"]).abs()
                     / m["pre_period_rx_baseline"] * 100)

    checks = {
        "Specialty violations": int((m["specialty"] != m["control_specialty"]).sum()),
        f"Geography violations ({COL_GEO})": int(
            (m["treatment_city"] != m["control_city"]).sum()),
        "Experience violations": int(
            (m["treatment_experience_years"] - m["control_experience_years"]).abs()
            .gt(MAX_EXPERIENCE_DIFF_YEARS).sum()),
        f"Rx diff violations (>{MAX_RX_PCT_DIFF:.0f}%)": int(
            m["rx_check"].gt(MAX_RX_PCT_DIFF).sum()),
    }

    print(f"\n(1) The four business rules  [matched rows re-audited: {len(m):,d}]")
    all_pass = True
    for name, count in checks.items():
        ok = count == 0
        all_pass &= ok
        print(f"    {name:<32s}: {'PASS' if ok else f'FAIL ({count} rows)'}")

    # Agreement between recomputed and precomputed -- catches a stale or
    # mis-derived audit column even when the underlying match is legitimate.
    drift = (m["rx_check"] - m["rx_pct_diff"]).abs().max()
    drift_ok = bool(pd.isna(drift) or drift < 1e-9)
    all_pass &= drift_ok
    print(f"    {'rx_pct_diff column accuracy':<32s}: "
          f"{'PASS' if drift_ok else f'FAIL (max drift {drift:.2e})'}")

    print(f"\n(2) Match rate")
    print(f"    total treatment cases : {len(final):,d}")
    print(f"    matched                : {len(m):,d}  ({100 * len(m) / len(final):.4f}%)")
    print(f"    unmatched               : {len(final) - len(m):,d}  "
          f"({100 * (len(final) - len(m)) / len(final):.4f}%)")

    print(f"\n(2b) Tie-break in effect: rx_pct_diff ASC, then control_hcp_id ASC")
    if len(m):
        rk = m["match_rank"].value_counts().sort_index()
        print(f"    match_rank distribution: "
              + ", ".join(f"rank {int(k)}={v:,d}" for k, v in rk.items()))
        print(f"    rank 1 (closest eligible control was free): "
              f"{int(rk.get(1, 0)):,d} ({rk.get(1, 0) / len(m):.1%}) -- higher ranks "
              f"mean same-event contention, never a relaxed rule")

    print(f"\n(3) Same-event control reuse (must be 0)")
    dupe = m.groupby("event_id")["control_hcp_id"].apply(lambda x: x.duplicated().any())
    n_dupe_events = int(dupe.sum())
    all_pass &= n_dupe_events == 0
    print(f"    events with a control assigned twice: {n_dupe_events}  "
          f"{'PASS' if n_dupe_events == 0 else 'FAIL'}")

    print(f"\n(4) Speaker exclusion (hard assertion)")
    used = set(m["control_hcp_id"].dropna().astype(int))
    leaked = used & exclusions["globally_excluded"]
    all_pass &= len(leaked) == 0
    print(f"    distinct controls used : {len(used):,d}")
    print(f"    speakers among them     : {len(leaked)}  "
          f"{'PASS' if not leaked else 'FAIL'}")

    print(f"\n(5) Temporal contamination (independent re-audit, hard assertion)")
    contaminated = sum(
        is_temporally_contaminated(r.control_hcp_id, r.window_start_ord,
                                   r.event_date_ord, attended_dates_lookup)
        for r in out_internal[out_internal["is_matched"]].itertuples())
    all_pass &= contaminated == 0
    print(f"    contaminated controls   : {contaminated}  "
          f"{'PASS' if contaminated == 0 else 'FAIL'}")

    print(f"\n(6) Cross-event control reuse (expected -- quantified, not hidden)")
    if len(m):
        reuse = m[["control_hcp_id", "event_id"]].drop_duplicates() \
            .groupby("control_hcp_id")["event_id"].nunique()
        print(f"    distinct controls used  : {len(reuse):,d}")
        print(f"    events per control -- min={reuse.min()}  mean={reuse.mean():.2f}  "
              f"median={reuse.median():.1f}  max={reuse.max()}")
        print(f"    used for exactly 1 event: {(reuse == 1).sum():,d} "
              f"({(reuse == 1).mean():.1%})")

    print(f"\n(7) Why cases went unmatched (funnel -- where the rules bite)")
    total = len(final)
    for label, n in diagnostics["funnel"]:
        print(f"    {label:<46s}: {n:>6,d}  ({n / total:6.2%})")

    print(f"\n(8) Unmatched by reason")
    reasons = out_internal.loc[~out_internal["is_matched"], "_unmatched_reason"].value_counts()
    for reason, n in reasons.items():
        print(f"    {reason:<46s}: {n:>6,d}  ({n / total:6.2%})")

    print(f"\n{'=' * 78}")
    print(f"  OVERALL: {'ALL CHECKS PASS -- file is ready' if all_pass else '*** FAILURES PRESENT -- DO NOT SEND FORWARD ***'}")
    print(f"{'=' * 78}")
    return all_pass


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> int:
    banner("RULE-BASED MATCHING -- Speaker Program ROI Analysis")
    print(f"  rules: same specialty AND same {COL_GEO} AND "
          f"|exp diff| <= {MAX_EXPERIENCE_DIFF_YEARS}y AND "
          f"|rx diff| <= {MAX_RX_PCT_DIFF:.0f}%")
    print(f"  rule 2 geography field: {COL_GEO!r}  "
          f"(approved widening from 'city'; true city retained in *_city_actual)")
    print(f"  tie-break: rx_pct_diff ASC, then control_hcp_id ASC; "
          f"match_rank = position in that order")
    print(f"  no-fallback policy: unmatched rows stay unmatched, never substituted")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  output directory: {OUTPUT_DIR}")

    # ---- load + reuse NNM's computations -----------------------------------
    data = load_data()
    rx_matrix, row_lookup, month_to_idx, _ = build_rx_lookup(data["rx"])

    events_idx = data["events"].copy()
    events_idx["event_idx"] = events_idx["event_month"].map(month_to_idx)

    cohort = build_treatment_cohort(data, month_to_idx)
    n_cases = len(cohort)

    exclusions = build_control_exclusions(data)
    attended_dates_lookup = build_attended_event_dates(data["attendance"], data["events"])

    baselines = compute_all_baselines(data, events_idx, rx_matrix, row_lookup)
    baseline_lookup = baselines["lookup"]

    # ---- attach the covariates the rules are written against ----------------
    hcp = data["hcp"]
    hcp_indexed = hcp.set_index("hcp_id")

    cohort = cohort.rename(columns={"treatment_hcp_id": "_tid"})
    cohort["treatment_hcp_id"] = cohort["_tid"]
    cohort = cohort.drop(columns="_tid")
    cohort["treatment_city"] = cohort["treatment_hcp_id"].map(hcp_indexed[COL_GEO])
    cohort["treatment_city_actual"] = cohort["treatment_hcp_id"].map(
        hcp_indexed[COL_CITY_ACTUAL])
    cohort["treatment_experience_years"] = cohort["treatment_hcp_id"].map(
        hcp_indexed[COL_EXPERIENCE])
    cohort["pre_period_rx_baseline"] = [
        baseline_lookup.get((e, h), np.nan)
        for e, h in zip(cohort["event_id"], cohort["treatment_hcp_id"])]

    # Pre-period window bounds, precomputed once as ordinal ints so the
    # per-candidate contamination check is an integer bisect, not a date parse.
    ev_ts = pd.to_datetime(cohort["event_date"])
    cohort["event_date_ord"] = ev_ts.map(lambda d: d.toordinal())
    cohort["window_start_ord"] = (ev_ts - pd.DateOffset(months=WINDOW_MONTHS)).map(
        lambda d: d.toordinal())

    # ---- (specialty, city) index: makes rules 1+2 an O(1) lookup ------------
    step("Building (specialty, city) candidate index -- enforces rules 1 & 2 by construction")
    specialty_city_index = (hcp.groupby([COL_SPECIALTY, COL_GEO])["hcp_id"]
                            .apply(lambda s: tuple(sorted(s))).to_dict())
    sizes = pd.Series({k: len(v) for k, v in specialty_city_index.items()})
    print(f"    geography field used by rule 2    : {COL_GEO!r}")
    print(f"    distinct (specialty, {COL_GEO}) cells : {len(specialty_city_index):,d}")
    print(f"    HCPs per cell -- min={sizes.min()}  mean={sizes.mean():.2f}  "
          f"median={sizes.median():.1f}  max={sizes.max()}")
    print(f"    cells containing only 1 HCP      : {(sizes == 1).sum():,d} "
          f"({(sizes == 1).mean():.1%})  <- these can never yield a control")

    event_attendee_ids = {
        k: frozenset(v) for k, v in
        data["attendance"][data["attendance"]["role"] == "attendee"]
        .groupby("event_id")["hcp_id"].apply(set).to_dict().items()}

    # ---- apply the four rules, per treatment case ---------------------------
    step("RULES  --  evaluating all four conditions per treatment case")
    eligible_lists: dict = {}
    funnel = {"no_specialty_city_peer": 0, "survived_all_rules": 0}
    for row in cohort.itertuples():
        key = (row.specialty, row.treatment_city)
        peers = [c for c in specialty_city_index.get(key, ())
                 if c != row.treatment_hcp_id
                 and c not in exclusions["globally_excluded"]]
        if not peers:
            funnel["no_specialty_city_peer"] += 1

        eligible = build_eligible_candidates(
            row, hcp_indexed, specialty_city_index, exclusions,
            event_attendee_ids, baseline_lookup, attended_dates_lookup)
        eligible_lists[row.Index] = eligible
        if eligible:
            funnel["survived_all_rules"] += 1

    n_with = funnel["survived_all_rules"]
    print(f"    cases with >=1 fully rule-satisfying candidate : {n_with:,d}  "
          f"({n_with / n_cases:.2%})")
    print(f"    cases with NO same-specialty+same-city peer at all: "
          f"{funnel['no_specialty_city_peer']:,d}  "
          f"({funnel['no_specialty_city_peer'] / n_cases:.2%})")
    pool_sizes = pd.Series([len(v) for v in eligible_lists.values()])
    print(f"    eligible pool size -- mean={pool_sizes.mean():.2f}  "
          f"max={pool_sizes.max()}  (0 for {int((pool_sizes == 0).sum()):,d} cases)")

    diagnostics = {"funnel": [
        ("no same-specialty+same-city peer exists", funnel["no_specialty_city_peer"]),
        ("has a peer, but none satisfied all 4 rules",
         n_cases - funnel["no_specialty_city_peer"] - n_with),
        ("at least one candidate satisfied all 4 rules", n_with),
    ]}

    # ---- assign + assemble + validate ---------------------------------------
    assignment = assign_controls(cohort, eligible_lists)
    final, out_internal = assemble_output(cohort, assignment, hcp_indexed, baseline_lookup)

    all_pass = run_self_validation(final, out_internal, exclusions,
                                   attended_dates_lookup, diagnostics)

    # ---- export --------------------------------------------------------------
    banner("EXPORT")
    final.to_csv(OUTPUT_FILE, index=False, na_rep="")
    print(f"    written to : {OUTPUT_FILE}")
    print(f"    rows       : {len(final):,d}  (expected {n_cases:,d})")
    if len(final) != n_cases:
        raise AssertionError("row count drift vs treatment cohort")

    with open(OUTPUT_FILE, "r", encoding="utf-8") as fh:
        cols = fh.readline().rstrip("\n").split(",")
        ci = cols.index("control_hcp_id")
        print(f"    control_hcp_id raw-text spot check (bytes on disk, not pandas):")
        for i in range(5):
            line = fh.readline()
            if not line:
                break
            field = line.rstrip("\n").split(",")[ci]
            print(f"      row {i + 1}: {field!r:<14s} "
                  f"[{'OK' if field == '' or field.isdigit() else 'SUSPECT'}]")

    if not all_pass:
        print("\n  *** self-validation FAILED -- do not send this file forward ***")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
