#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Precompute the Rule-Based funnel stage counts into a real data artifact.

WHY THIS EXISTS
    The dashboard's centrepiece funnel needs four stage counts:
        1. total treatment cases
        2. cases with >=1 same-specialty + same-region eligible peer
        3. cases where >=1 peer satisfied ALL FOUR rules
        4. cases actually matched (after within-event contention)

    Stages 1 and 4 are readable straight from rule_based_matching_output.csv.
    Stages 2 and 3 are NOT: unmatched rows carry no control columns at all, so
    the file cannot say whether a case failed for want of a peer, or had peers
    that all failed a rule. Recovering that needs the HCP master, the speaker
    exclusions, and the per-(event, hcp) baseline lookup.

    Rather than hardcode those numbers into the dashboard (which would make the
    chart untraceable to any file), this script recomputes them by calling
    rbm_matching's OWN build_eligible_candidates() -- the same function that
    produced the shipped matches -- and writes the result to a CSV the
    dashboard reads like any other data source.

Run once after any rule change:
    python build_funnel_artifact.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "matching_techinques"))

import nnm_matching as nnm                      # noqa: E402
import rbm_matching as rbm                      # noqa: E402
from nnm_matching import (                      # noqa: E402
    build_attended_event_dates,
    build_control_exclusions,
    build_rx_lookup,
    build_treatment_cohort,
    compute_all_baselines,
    load_data,
)

OUTPUT_FILE = PROJECT_ROOT / "did_roi_output" / "rbm_funnel_stages.csv"


def main() -> int:
    data = load_data()
    rx_matrix, row_lookup, month_to_idx, _ = build_rx_lookup(data["rx"])

    events_idx = data["events"].copy()
    events_idx["event_idx"] = events_idx["event_month"].map(month_to_idx)

    cohort = build_treatment_cohort(data, month_to_idx)
    exclusions = build_control_exclusions(data)
    attended = build_attended_event_dates(data["attendance"], data["events"])
    baseline_lookup = compute_all_baselines(
        data, events_idx, rx_matrix, row_lookup)["lookup"]

    hcp = data["hcp"]
    hcp_indexed = hcp.set_index("hcp_id")

    # Mirror rbm_matching's own field derivation exactly.
    cohort["treatment_city"] = cohort["treatment_hcp_id"].map(hcp_indexed[rbm.COL_GEO])
    cohort["treatment_experience_years"] = cohort["treatment_hcp_id"].map(
        hcp_indexed[rbm.COL_EXPERIENCE])
    cohort["pre_period_rx_baseline"] = [
        baseline_lookup.get((e, h), np.nan)
        for e, h in zip(cohort["event_id"], cohort["treatment_hcp_id"])]

    ev_ts = pd.to_datetime(cohort["event_date"])
    cohort["event_date_ord"] = ev_ts.map(lambda d: d.toordinal())
    cohort["window_start_ord"] = (
        ev_ts - pd.DateOffset(months=rbm.WINDOW_MONTHS)).map(lambda d: d.toordinal())

    specialty_geo_index = (hcp.groupby([rbm.COL_SPECIALTY, rbm.COL_GEO])["hcp_id"]
                           .apply(lambda s: tuple(sorted(s))).to_dict())
    event_attendee_ids = {
        k: frozenset(v) for k, v in
        data["attendance"][data["attendance"]["role"] == "attendee"]
        .groupby("event_id")["hcp_id"].apply(set).to_dict().items()}

    n_total = len(cohort)
    n_has_peer = 0
    n_survived = 0

    for row in cohort.itertuples():
        peers = [c for c in specialty_geo_index.get((row.specialty, row.treatment_city), ())
                 if c != row.treatment_hcp_id
                 and c not in exclusions["globally_excluded"]
                 and c not in event_attendee_ids.get(row.event_id, frozenset())]
        if peers:
            n_has_peer += 1
        if rbm.build_eligible_candidates(
                row, hcp_indexed, specialty_geo_index, exclusions,
                event_attendee_ids, baseline_lookup, attended):
            n_survived += 1

    rb_out = pd.read_csv(
        nnm.MATCHED_PAIRS_DIR / "rule_based_matching" / "rule_based_matching_output.csv")
    n_matched = int(rb_out["is_matched"].sum())

    stages = pd.DataFrame([
        {"stage_order": 1, "stage": "Total treatment cases",
         "n_cases": n_total,
         "detail": "one row per (treatment HCP, event) in the matched-pairs file"},
        {"stage_order": 2, "stage": "Has >=1 same-specialty + same-region peer",
         "n_cases": n_has_peer,
         "detail": "after speaker exclusion, self-exclusion and same-event exclusion"},
        {"stage_order": 3, "stage": "Peer satisfied ALL 4 rules",
         "n_cases": n_survived,
         "detail": "specialty AND region AND |exp diff|<=3y AND |Rx diff|<=10%, "
                   "plus temporal-contamination and computable-baseline filters"},
        {"stage_order": 4, "stage": "Actually matched",
         "n_cases": n_matched,
         "detail": "after within-event without-replacement contention"},
    ])
    stages["pct_of_total"] = stages["n_cases"] / n_total * 100.0
    stages["geo_rule_column"] = rbm.COL_GEO
    stages["max_experience_diff_years"] = rbm.MAX_EXPERIENCE_DIFF_YEARS
    stages["max_rx_pct_diff"] = rbm.MAX_RX_PCT_DIFF

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    stages.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 70)
    print("RULE-BASED FUNNEL STAGES")
    print("=" * 70)
    for r in stages.itertuples():
        print(f"  {r.stage:<45s} {r.n_cases:>6,d}  ({r.pct_of_total:5.2f}%)")
    print(f"\n  written -> {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
