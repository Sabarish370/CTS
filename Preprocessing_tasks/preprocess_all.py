#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 PREPROCESSING -- all four raw files -> the pipeline's canonical inputs
================================================================================

Merge of hcp_preprocessed.py, attendance_preprocessed.py, events_preprocessed.py
and rx_claims_monthly_preprocess.py. The cleaning logic of each is preserved
exactly; only the environment (Colab -> local) and the OUTPUT FILENAMES changed.

THE FILENAME BUG THIS FIXES
    The four originals wrote hcp_preprocessed.csv, attendance-final-1.csv,
    events_preprocessed_final.csv and rx_claims_monthly_preprocessed.csv.
    nnm_matching.py -- the verified source of truth for the matching pipeline --
    requires these exact names (confirmed against its constants, lines 83-86):

        hcp-final.csv
        attendance-final.csv
        events_preprocessed_final.csv
        rx_claims_monthly_preprocessed (1).csv      <- parenthetical is real

    Three of the four disagreed, so a fresh run needed a manual rename before
    matching could find its inputs. They now write the canonical names directly.

TWO ENVIRONMENT ADAPTATIONS (flagged, not silent)
    1. RAW FORMAT. The originals called pd.read_excel() on attendance and events
       because the Colab-uploaded files were XLSX despite a .csv extension. The
       files now in generated_data/ are genuine plain-text CSV (verified by
       magic bytes), on which read_excel raises. read_tabular() below sniffs the
       first four bytes and dispatches accordingly, so BOTH shapes work and the
       original intent is preserved rather than standardised away.
    2. EVENT DATE PARSING. events_preprocessed.py calls .dt.year on event_date
       without ever calling to_datetime -- it got datetime for free from
       read_excel. Reading real CSV yields strings, so an explicit
       pd.to_datetime() is required for identical behaviour. Same for the
       program_spend values restored out of the DQ log, which arrive as text
       from a CSV read and as numbers from an Excel read.

Run
---
    python preprocess_all.py
================================================================================
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


# ==============================================================================
# PATHS -- same find_project_root() pattern nnm_matching.py uses
# ==============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent


def find_project_root(start: Path) -> Path:
    """Walk up until the folder holding the project's data dirs."""
    for candidate in (start, *start.parents):
        if (candidate / "generated_data").is_dir():
            return candidate
    return start.parent


PROJECT_ROOT = find_project_root(SCRIPT_DIR)
RAW_DIR = PROJECT_ROOT / "generated_data"
OUT_DIR = PROJECT_ROOT / "preprocessed_data"

# Raw filenames as they ACTUALLY exist in generated_data/. The originals
# referenced events-2.csv and attendance-2.csv -- stale Colab upload artifacts
# that do not exist here.
RAW_HCP = "hcp.csv"
RAW_ATTENDANCE = "attendance.csv"
RAW_EVENTS = "events.csv"
RAW_RX = "rx_claims_monthly.csv"
RAW_DQ_LOG = "data_quality_injection_log.csv"
RAW_GROUND_TRUTH = "ground_truth_config.json"

# Canonical output names required by nnm_matching.py.
OUT_HCP = "hcp-final.csv"
OUT_ATTENDANCE = "attendance-final.csv"
OUT_EVENTS = "events_preprocessed_final.csv"
OUT_RX = "rx_claims_monthly_preprocessed (1).csv"


def banner(t: str) -> None:
    print("\n" + "=" * 70); print(t); print("=" * 70)


def read_tabular(path: Path, **kwargs) -> pd.DataFrame:
    """Read a table whose extension may lie about its format.

    The originals hard-coded read_excel for attendance/events because those
    uploads were XLSX named .csv. Sniffing the zip magic number handles both
    that case and genuine CSV, so the merged script survives either.
    """
    with open(path, "rb") as fh:
        magic = fh.read(4)
    if magic == b"PK\x03\x04":
        return pd.read_excel(path, **kwargs)
    return pd.read_csv(path, **kwargs)


# ==============================================================================
# 1. HCP
# ==============================================================================

def preprocess_hcp(raw_dir: Path, out_dir: Path) -> pd.DataFrame:
    banner("HCP PREPROCESSING")
    hcp = pd.read_csv(raw_dir / RAW_HCP)
    print("Shape:", hcp.shape)
    print("\nColumns:")
    print(hcp.columns.tolist())

    text_columns = [
        "first_name", "last_name", "specialty", "sub_specialty", "region",
        "state", "city", "practice_setting", "npi_taxonomy_code",
    ]
    for col in text_columns:
        if col in hcp.columns:
            hcp[col] = hcp[col].astype("string").str.strip()
    print("Text cleaning completed.")

    hcp["first_name"] = hcp["first_name"].str.title()
    hcp["last_name"] = hcp["last_name"].str.title()
    hcp["name"] = (
        hcp["first_name"].fillna("") + " " + hcp["last_name"].fillna("")
    ).str.strip()
    print("Name column created.")

    hcp["specialty"] = hcp["specialty"].astype("string").str.strip().str.lower()
    hcp["sub_specialty"] = hcp["sub_specialty"].astype("string").str.strip().str.lower()
    print("Specialty cleaning completed.")
    print("\nSpecialty values:")
    print(hcp["specialty"].value_counts(dropna=False))
    print("\nMissing sub-specialty:", hcp["sub_specialty"].isna().sum())

    hcp["region"] = hcp["region"].astype("string").str.strip().str.lower()
    region_mapping = {
        "northeast": "Northeast", "n.e.": "Northeast", "ne": "Northeast",
        "midwest": "Midwest", "mid-west": "Midwest", "mid west": "Midwest",
        "mw": "Midwest",
        "southeast": "Southeast", "s.e.": "Southeast", "se": "Southeast",
        "southwest": "Southwest", "south west": "Southwest", "sw": "Southwest",
        "west": "West", "western": "West", "west coast": "West", "w": "West",
    }
    hcp["region"] = hcp["region"].map(region_mapping)
    print("Region cleaning completed.")
    print("\nRegion values:")
    print(hcp["region"].value_counts(dropna=False))

    print("Missing sub-specialty values:", hcp["sub_specialty"].isna().sum())

    # Fill each row's missing sub_specialty with its specialty group's mode.
    # Uses a groupby-transform rather than groupby().apply() on a function
    # that reads the grouping column: pandas 2.2+ deprecated (and pandas 3.0
    # removed) passing the grouping column through to apply()'s func, which
    # silently dropped "specialty" from the result and broke every column
    # selection downstream. transform() never has that ambiguity.
    def group_mode(s: pd.Series):
        modes = s.mode()
        return modes.iloc[0] if len(modes) > 0 else pd.NA

    group_modes = hcp.groupby("specialty")["sub_specialty"].transform(group_mode)
    hcp["sub_specialty"] = hcp["sub_specialty"].fillna(group_modes)
    print("Missing sub-specialty after filling:", hcp["sub_specialty"].isna().sum())

    print("Duplicate HCP IDs:", hcp["hcp_id"].duplicated().sum())
    hcp = hcp.drop_duplicates(subset=["hcp_id"], keep="first").copy()
    print("Shape after removing duplicates:", hcp.shape)

    numeric_columns = ["hcp_id", "years_in_practice", "patient_volume_monthly",
                       "baseline_rx_volume_monthly"]
    for col in numeric_columns:
        if col in hcp.columns:
            hcp[col] = pd.to_numeric(hcp[col], errors="coerce")
    print("Numeric columns cleaned.")

    hcp["speaker_eligible_flag"] = (
        hcp["speaker_eligible_flag"].astype(str).str.strip().str.lower()
        .map({"true": True, "false": False, "1": True, "0": False,
              "yes": True, "no": False})
    )
    print("Speaker eligibility cleaned.")
    print("\nSpeaker eligible values:")
    print(hcp["speaker_eligible_flag"].value_counts(dropna=False))

    hcp = hcp.drop(columns=["first_name", "last_name"], errors="ignore")

    final_columns = [
        "hcp_id", "name", "specialty", "sub_specialty", "region", "state",
        "city", "years_in_practice", "practice_setting",
        "patient_volume_monthly", "baseline_rx_volume_monthly",
        "speaker_eligible_flag", "npi_taxonomy_code",
    ]
    hcp = hcp[final_columns]

    print("\n--- FINAL HCP VALIDATION ---")
    print("Shape:", hcp.shape)
    print("Unique HCP IDs:", hcp["hcp_id"].nunique())
    print("Duplicate HCP IDs:", hcp["hcp_id"].duplicated().sum())
    print("Missing values:\n", hcp.isnull().sum())

    out = out_dir / OUT_HCP
    hcp.to_csv(out, index=False)
    print("File saved successfully:", out)
    return hcp


# ==============================================================================
# 2. EVENTS
# ==============================================================================

def preprocess_events(raw_dir: Path, out_dir: Path) -> pd.DataFrame:
    banner("EVENTS PREPROCESSING")
    path = raw_dir / RAW_EVENTS
    with open(path, "rb") as f:
        print("First 20 bytes:", f.read(20))

    events = read_tabular(path)
    print("\nEvents loaded successfully!")
    print("Shape:", events.shape)
    print("\nColumns:")
    print(events.columns.tolist())

    dq_log = read_tabular(raw_dir / RAW_DQ_LOG)
    print("\nQuality log loaded successfully!")
    print("Shape:", dq_log.shape)
    print("Target files in quality log:")
    print(dq_log["target_file"].value_counts())

    event_log = dq_log[
        dq_log["target_file"].astype(str).str.lower() == "events.csv"].copy()
    print("Number of Events quality issues:", len(event_log))
    print("\nIssue types:")
    print(event_log["issue_type"].value_counts())

    # Row-by-row restoration of the values the injector overwrote.
    for _, row in event_log.iterrows():
        event_id = str(row["row_key"]).replace("event_id=", "")
        column = row["column"]
        original_value = row["original_value"]
        mask = events["event_id"].astype(str) == event_id
        if mask.any():
            events.loc[mask, column] = original_value
    print("Original values restored successfully.")

    # ENVIRONMENT ADAPTATION: original_value arrives numeric from an Excel read
    # but as text from a CSV read. Coerce so the downstream <0 comparisons and
    # the saved dtype match the original behaviour.
    events["program_spend"] = pd.to_numeric(events["program_spend"], errors="coerce")

    # ENVIRONMENT ADAPTATION: the original relied on read_excel to hand back a
    # datetime; .dt below needs one explicitly when the source is real CSV.
    events["event_date"] = pd.to_datetime(events["event_date"], errors="coerce")

    print("\nData types before cleaning:\n", events.dtypes)
    print("\nMissing values in Events dataset:\n", events.isnull().sum())
    print("Duplicate event IDs:", events["event_id"].duplicated().sum())
    print("Missing/invalid event dates:", events["event_date"].isna().sum())
    print("Negative program spend:", (events["program_spend"] < 0).sum())
    print("Negative capacity:", (events["max_capacity"] < 0).sum())
    print("Negative attendee count:", (events["attendee_count"] < 0).sum())
    print("Attendees greater than capacity:",
          (events["attendee_count"] > events["max_capacity"]).sum())

    events["event_year"] = events["event_date"].dt.year
    events["event_month"] = events["event_date"].dt.to_period("M").astype(str)
    print("Date features created successfully.")

    events["attendance_rate"] = events["attendee_count"] / events["max_capacity"]
    print("Attendance rate created successfully.")

    print("\n--- FINAL EVENTS VALIDATION ---")
    print("1. Number of rows:", len(events))
    print("2. Duplicate event IDs:", events["event_id"].duplicated().sum())
    print("3. Missing event dates:", events["event_date"].isna().sum())
    print("4. Missing program spend:", events["program_spend"].isna().sum())
    print("5. Negative program spend:", (events["program_spend"] < 0).sum())

    check_ids = ["EVT-0034", "EVT-0174", "EVT-0194", "EVT-0053"]
    print("\n9. Corrected event values:")
    print(events[events["event_id"].astype(str).isin(check_ids)][
        ["event_id", "program_spend"]])

    out = out_dir / OUT_EVENTS
    events.to_csv(out, index=False)
    print("File saved successfully!", out)
    print("Rows:", events.shape[0], "Columns:", events.shape[1])
    return events


# ==============================================================================
# 3. ATTENDANCE
# ==============================================================================

def preprocess_attendance(raw_dir: Path, out_dir: Path) -> pd.DataFrame:
    banner("ATTENDANCE PREPROCESSING")
    df = read_tabular(raw_dir / RAW_ATTENDANCE)
    print("Number of rows    :", df.shape[0])
    print("Number of columns :", df.shape[1])
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nDATA TYPES\n", df.dtypes)
    print("\nMISSING VALUES BEFORE PREPROCESSING\n", df.isnull().sum())
    print("\nCompletely duplicated rows:", df.duplicated().sum())
    print("Duplicate attendance IDs:", df["attendance_id"].duplicated().sum())

    # CHECK 1 of 2 -- the ATT-DUP-prefixed rows are INJECTED duplicate records,
    # a deliberate data-quality defect. Deliberately kept separate from the
    # generic drop_duplicates() below: two different problems, two checks.
    att_dup_mask = df["attendance_id"].astype(str).str.startswith("ATT-DUP-")
    att_dup_records = df[att_dup_mask].copy()
    print("\nNumber of ATT-DUP records:", len(att_dup_records))

    df_clean = df[~att_dup_mask].copy()
    print("Rows before:", len(df))
    print("Rows after :", len(df_clean))
    print("Rows removed:", len(df) - len(df_clean))

    # CHECK 2 of 2 -- ordinary accidental duplicate rows.
    before_duplicates = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    print("Rows before duplicate removal:", before_duplicates)
    print("Rows after duplicate removal :", len(df_clean))
    print("Duplicates removed            :", before_duplicates - len(df_clean))

    df_clean["attendance_date"] = pd.to_datetime(
        df_clean["attendance_date"], errors="coerce")
    print("Invalid dates:", df_clean["attendance_date"].isnull().sum())

    print("\nENGAGEMENT SCORE\n", df_clean["engagement_score"].describe())
    print("\nMissing engagement scores:", df_clean["engagement_score"].isnull().sum())
    invalid_engagement = df_clean[(df_clean["engagement_score"] < 0)
                                  | (df_clean["engagement_score"] > 10)]
    print("Invalid engagement scores:", len(invalid_engagement))

    print("\nUnique HCPs:", df_clean["hcp_id"].nunique())
    print("Missing HCP IDs:", df_clean["hcp_id"].isnull().sum())
    print("Unique Events:", df_clean["event_id"].nunique())
    print("Missing Event IDs:", df_clean["event_id"].isnull().sum())
    print("\nROLE DISTRIBUTION\n", df_clean["role"].value_counts(dropna=False))
    print("\nFINAL MISSING VALUE CHECK\n", df_clean.isnull().sum())

    df_clean = df_clean.reset_index(drop=True)

    out = out_dir / OUT_ATTENDANCE
    df_clean.to_csv(out, index=False)
    print("\nPREPROCESSING COMPLETED")
    print("Output file:", out)
    print("Final rows:", df_clean.shape[0])
    print("Final columns:", df_clean.shape[1])
    return df_clean


# ==============================================================================
# 4. RX CLAIMS MONTHLY
# ==============================================================================

def preprocess_rx(raw_dir: Path, out_dir: Path) -> pd.DataFrame:
    banner("RX CLAIMS MONTHLY PREPROCESSING")

    # ground_truth_config.json is read ONLY to print metadata and to report how
    # many rows fall outside the configured panel. It NEVER filters, adjusts or
    # transforms a single row -- keeping the ground truth strictly read-only is
    # what lets it stay a valid scoring key downstream.
    with open(raw_dir / RAW_GROUND_TRUTH, "r") as f:
        config = json.load(f)
    metadata = config["metadata"]
    print("Ground Truth Configuration")
    print("-" * 50)
    print("True effect:", metadata["true_effect_pct"], "%")
    print("Analysis window:", metadata["analysis_window_months"], "months")
    print("Rx panel:", metadata["rx_panel_months"])
    print("Event window:", metadata["event_window"])
    print("Treatment units:", metadata["n_treatment_units"])
    print("Washout rule:")
    print(metadata["washout_rule"])

    rx = pd.read_csv(raw_dir / RAW_RX)
    print("\nShape:", rx.shape)
    print("Columns:", rx.columns.tolist())

    dq_log = read_tabular(raw_dir / RAW_DQ_LOG)
    print("DQ log shape:", dq_log.shape)

    rx_log = dq_log[
        dq_log["target_file"].astype(str).str.lower() == "rx_claims_monthly.csv"].copy()
    print("RX quality-log records:", len(rx_log))
    print("\nIssue types:")
    print(rx_log["issue_type"].value_counts())

    print("\nInitial RX shape:", rx.shape)
    print("Missing values:\n", rx.isna().sum())
    print("Duplicate complete rows:", rx.duplicated().sum())
    print("Duplicate HCP-Month-NDC rows:",
          rx.duplicated(["hcp_id", "month", "ndc_category"]).sum())
    print("Month range:", rx["month"].min(), "to", rx["month"].max())
    print("Unique HCPs:", rx["hcp_id"].nunique())
    print("NDC categories:", rx["ndc_category"].unique())

    rx["hcp_id"] = rx["hcp_id"].astype(str).str.strip()
    rx["month_date"] = pd.to_datetime(
        rx["month"].astype(str).str.strip(), format="%Y-%m", errors="coerce")
    print("Invalid month values:", rx["month_date"].isna().sum())

    def make_row_key(row):
        return (f"hcp_id={row['hcp_id']}"
                f"|month={row['month']}"
                f"|ndc_category={row['ndc_category']}")

    rx["row_key"] = rx.apply(make_row_key, axis=1)

    missing_log = rx_log[rx_log["issue_type"] == "missing_rx_volume"].copy()
    missing_corrections = dict(zip(
        missing_log["row_key"],
        pd.to_numeric(missing_log["original_value"], errors="coerce")))
    print("Missing values in quality log:", len(missing_corrections))

    rx["rx_volume_restored_from_log"] = False
    for key, original_value in missing_corrections.items():
        mask = rx["row_key"] == key
        if mask.any():
            rx.loc[mask, "rx_volume"] = original_value
            rx.loc[mask, "rx_volume_restored_from_log"] = True
    print("Rows restored:", rx["rx_volume_restored_from_log"].sum())
    print("Remaining missing rx_volume:", rx["rx_volume"].isna().sum())

    rx["rx_volume"] = rx["rx_volume"].astype(str).str.replace(",", "", regex=False)
    rx["rx_volume"] = pd.to_numeric(rx["rx_volume"], errors="coerce")
    print("rx_volume dtype:", rx["rx_volume"].dtype)
    print("Non-numeric / remaining missing:", rx["rx_volume"].isna().sum())

    rx["rx_trend_baseline"] = pd.to_numeric(rx["rx_trend_baseline"], errors="coerce")

    print("Negative RX rows:", len(rx[rx["rx_volume"] < 0]))
    print("Duplicate HCP-month-NDC rows:", int(rx.duplicated(
        subset=["hcp_id", "month", "ndc_category"], keep=False).sum()))

    # Injected outliers are FLAGGED, never removed -- they are genuine extreme
    # values in the simulation, and dropping them would silently change the
    # distribution every downstream method sees.
    outlier_log = rx_log[rx_log["issue_type"] == "outlier_rx_volume"].copy()
    outlier_keys = set(outlier_log["row_key"])
    rx["is_injected_outlier"] = rx["row_key"].isin(outlier_keys)
    print("Injected outliers flagged:", rx["is_injected_outlier"].sum())

    rx["year"] = rx["month_date"].dt.year
    rx["month_number"] = rx["month_date"].dt.month
    rx["time_index"] = (
        (rx["month_date"].dt.year - rx["month_date"].dt.year.min()) * 12
        + rx["month_date"].dt.month)

    print("\nNDC category counts:")
    print(rx["ndc_category"].value_counts().sort_index())

    # Read-only panel check -- reports, does not filter.
    panel_start = pd.Period(metadata["rx_panel_months"][0], freq="M")
    panel_end = pd.Period(metadata["rx_panel_months"][1], freq="M")
    rx_period = rx["month_date"].dt.to_period("M")
    outside_panel = rx[(rx_period < panel_start) | (rx_period > panel_end)]
    print("Rows outside configured RX panel:", len(outside_panel),
          "(reported only -- no rows removed)")

    print("\n--- FINAL RX VALIDATION ---")
    print("Rows:", len(rx))
    print("Columns:", len(rx.columns))
    print("Unique HCPs:", rx["hcp_id"].nunique())
    print("Unique months:", rx["month_date"].nunique())
    print("NDC categories:", rx["ndc_category"].nunique())
    print("Month range:", rx["month_date"].min(), "to", rx["month_date"].max())
    print("Negative RX volume:", (rx["rx_volume"] < 0).sum())
    print("Values restored from DQ log:", rx["rx_volume_restored_from_log"].sum())

    rx = rx.drop(columns=["row_key"])

    final_columns = [
        "hcp_id", "month", "month_date", "year", "month_number", "time_index",
        "ndc_category", "rx_volume", "rx_trend_baseline",
        "is_injected_outlier", "rx_volume_restored_from_log",
    ]
    rx_preprocessed = rx[final_columns].copy()
    print(rx_preprocessed.columns.tolist())

    out = out_dir / OUT_RX
    rx_preprocessed.to_csv(out, index=False)
    print("Saved:", out)
    print("Shape:", rx_preprocessed.shape)
    return rx_preprocessed


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> int:
    banner("PREPROCESSING -- ALL FOUR RAW FILES")
    print(f"  project root : {PROJECT_ROOT}")
    print(f"  raw dir      : {RAW_DIR}")
    print(f"  out dir      : {OUT_DIR}")

    missing = [f for f in (RAW_HCP, RAW_ATTENDANCE, RAW_EVENTS, RAW_RX,
                           RAW_DQ_LOG, RAW_GROUND_TRUTH)
               if not (RAW_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(f"missing raw inputs in {RAW_DIR}: {missing}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # The four are independent -- none reads another's output. hcp and events
    # first, then attendance and rx, purely for readability.
    hcp = preprocess_hcp(RAW_DIR, OUT_DIR)
    events = preprocess_events(RAW_DIR, OUT_DIR)
    attendance = preprocess_attendance(RAW_DIR, OUT_DIR)
    rx = preprocess_rx(RAW_DIR, OUT_DIR)

    banner("SUMMARY -- canonical outputs")
    for name, df in ((OUT_HCP, hcp), (OUT_EVENTS, events),
                     (OUT_ATTENDANCE, attendance), (OUT_RX, rx)):
        p = OUT_DIR / name
        print(f"  {name:<42s} rows={len(df):>7,d}  cols={df.shape[1]:>2d}  "
              f"exists={p.exists()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
