import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# 1. SET FILE PATHS
# ============================================================

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

hcp_file = INPUT_DIR / "hcp-final.csv"
attendance_file = INPUT_DIR / "attendance-final.csv"
events_file = INPUT_DIR / "events_preprocessed_final.csv"
rx_file = INPUT_DIR / "rx_claims_monthly_preprocessed (1).csv"

# Output CSV -- folder name keeps the "randam" spelling that is actually on
# disk and that did_roi_engine.py reads from; renaming it here would write to
# a new empty folder and orphan the DiD/ROI input.
OUTPUT_DIR = PROJECT_ROOT / "matched_pairs" / "randam_matching"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

output_file = OUTPUT_DIR / "matching_output_random.csv"

# ============================================================
# 2. LOAD DATASETS
# ============================================================

print("\nLoading datasets...")

hcp = pd.read_csv(
    hcp_file,
    dtype={"hcp_id": str}
)

attendance = pd.read_csv(
    attendance_file,
    dtype={
        "hcp_id": str,
        "event_id": str
    }
)

events = pd.read_csv(
    events_file,
    dtype={"event_id": str}
)

rx = pd.read_csv(
    rx_file,
    dtype={"hcp_id": str}
)

print("Datasets loaded successfully.")

# ============================================================
# 3. CLEAN HCP IDs
# ============================================================

hcp["hcp_id"] = hcp["hcp_id"].astype(str).str.strip()

attendance["hcp_id"] = (
    attendance["hcp_id"]
    .astype(str)
    .str.strip()
)

attendance["event_id"] = (
    attendance["event_id"]
    .astype(str)
    .str.strip()
)

events["event_id"] = (
    events["event_id"]
    .astype(str)
    .str.strip()
)

rx["hcp_id"] = (
    rx["hcp_id"]
    .astype(str)
    .str.strip()
)

# ============================================================
# 4. CONVERT DATES
# ============================================================

attendance["attendance_date"] = pd.to_datetime(
    attendance["attendance_date"],
    errors="coerce"
)

events["event_date"] = pd.to_datetime(
    events["event_date"],
    errors="coerce"
)

rx["month_date"] = pd.to_datetime(
    rx["month_date"],
    errors="coerce"
)

# Convert Rx month to first day of month
rx["month_start"] = (
    rx["month_date"]
    .dt.to_period("M")
    .dt.to_timestamp()
)

# Convert Rx volume to numeric
rx["rx_volume"] = pd.to_numeric(
    rx["rx_volume"],
    errors="coerce"
)

# ============================================================
# 5. CREATE TREATMENT CASES
#    One row = one unique attendee-event combination
# ============================================================

attendees = attendance[
    attendance["role"]
    .fillna("")
    .str.lower()
    .eq("attendee")
].copy()

treatment = attendees.drop_duplicates(
    ["hcp_id", "event_id"]
).copy()

print("\nUnique treatment cases:", len(treatment))

# ============================================================
# 6. ADD EVENT INFORMATION
# ============================================================

event_info = events[
    [
        "event_id",
        "event_date",
        "target_ndc_category"
    ]
].drop_duplicates("event_id")

treatment = treatment.merge(
    event_info,
    on="event_id",
    how="left"
)

# ============================================================
# 7. ADD TREATMENT HCP INFORMATION
# ============================================================

hcp_info = hcp[
    [
        "hcp_id",
        "specialty",
        "region"
    ]
].drop_duplicates("hcp_id")

treatment = treatment.merge(
    hcp_info,
    on="hcp_id",
    how="left"
)

# ============================================================
# 8. IDENTIFY SPEAKERS
#
# A speaker cannot be used as a control if:
#
# 1. speaker_eligible_flag == True
# OR
# 2. they have role == speaker in attendance
# ============================================================

speaker_from_flag = set(
    hcp.loc[
        hcp["speaker_eligible_flag"]
        .fillna(False)
        .astype(bool),
        "hcp_id"
    ]
)

speaker_from_attendance = set(
    attendance.loc[
        attendance["role"]
        .fillna("")
        .str.lower()
        .eq("speaker"),
        "hcp_id"
    ]
)

speaker_ids = (
    speaker_from_flag |
    speaker_from_attendance
)

print(
    "Speaker HCPs excluded:",
    len(speaker_ids)
)

# ============================================================
# 9. INITIAL CONTROL POOL
# ============================================================

all_hcps = set(
    hcp["hcp_id"].astype(str)
)

control_pool = (
    all_hcps - speaker_ids
)

print(
    "Initial eligible control HCPs:",
    len(control_pool)
)

# ============================================================
# 10. PREPARE RX DATA
# ============================================================

rx_data = rx[
    [
        "hcp_id",
        "month_start",
        "ndc_category",
        "rx_volume"
    ]
].copy()

rx_data["hcp_id"] = (
    rx_data["hcp_id"]
    .astype(str)
    .str.strip()
)

rx_data["ndc_category"] = (
    rx_data["ndc_category"]
    .astype(str)
    .str.strip()
)

# Remove invalid rows
rx_data = rx_data[
    rx_data["month_start"].notna()
].copy()

# ============================================================
# 11. PRE-COMPUTE RX BASELINES
#
# This makes the matching significantly faster than repeatedly
# filtering the complete Rx dataframe for every candidate.
# ============================================================

print("\nPreparing Rx lookup...")

rx_lookup = {}

for (hcp_id, ndc_category), group in rx_data.groupby(
    ["hcp_id", "ndc_category"]
):

    group = group.sort_values("month_start")

    # If there are duplicate HCP/month/category records,
    # aggregate them first.
    monthly = (
        group
        .groupby("month_start", as_index=False)["rx_volume"]
        .mean()
    )

    rx_lookup[
        (str(hcp_id), str(ndc_category))
    ] = monthly

print(
    "Rx lookup prepared for",
    len(rx_lookup),
    "HCP/category combinations."
)

# ============================================================
# 12. FUNCTION TO CALCULATE PRE-PERIOD BASELINE
# ============================================================

def get_rx_baseline(
    hcp_id,
    ndc_category,
    pre_start,
    pre_end
):

    key = (
        str(hcp_id),
        str(ndc_category)
    )

    if key not in rx_lookup:
        return np.nan

    data = rx_lookup[key]

    values = data.loc[
        (data["month_start"] >= pre_start) &
        (data["month_start"] <= pre_end),
        "rx_volume"
    ]

    if values.empty:
        return np.nan

    return values.mean()


# ============================================================
# 13. STORE ATTENDEE PROGRAM EXPOSURES
#
# Used to prevent selecting a control who attended another
# program within the previous 6 months.
# ============================================================

attendee_exposure = attendees[
    [
        "hcp_id",
        "event_id"
    ]
].merge(
    events[
        [
            "event_id",
            "event_date"
        ]
    ],
    on="event_id",
    how="left"
)

attendee_exposure["hcp_id"] = (
    attendee_exposure["hcp_id"]
    .astype(str)
)

# ============================================================
# 14. STORE CURRENT EVENT ATTENDEES
# ============================================================

current_event_attendees_lookup = (
    attendees
    .groupby("event_id")["hcp_id"]
    .apply(
        lambda x: set(x.astype(str))
    )
    .to_dict()
)

# ============================================================
# 15. RANDOM MATCHING
# ============================================================

print("\nStarting random matching...")

random_generator = np.random.default_rng(42)

results = []

# Track controls already used for each event
used_controls_by_event = {}

total_cases = len(treatment)

for case_number, (_, treatment_row) in enumerate(
    treatment.iterrows(),
    start=1
):

    treatment_hcp = str(
        treatment_row["hcp_id"]
    )

    event_id = str(
        treatment_row["event_id"]
    )

    event_date = treatment_row["event_date"]

    target_ndc = str(
        treatment_row["target_ndc_category"]
    )

    # --------------------------------------------------------
    # Check for invalid event date
    # --------------------------------------------------------

    if pd.isna(event_date):

        results.append({

            "treatment_hcp_id": treatment_hcp,

            "control_hcp_id": "",

            "method": "random",

            "event_id": event_id,

            "event_date": "",

            "target_ndc_category": target_ndc,

            "specialty": treatment_row["specialty"],

            "region": treatment_row["region"],

            "pre_period_rx_baseline": np.nan,

            "control_pre_period_rx_baseline": np.nan,

            "is_matched": False,

            "match_rank": pd.NA
        })

        continue

    # --------------------------------------------------------
    # PRE-PERIOD
    #
    # 6 calendar months strictly before the event month.
    # Must match NNM/RBM (WINDOW_MONTHS = 6) -- one shared DiD/ROI
    # function consumes all four methods' outputs, so
    # pre_period_rx_baseline has to mean the same thing everywhere.
    # --------------------------------------------------------

    event_month = (
        event_date
        .to_period("M")
        .to_timestamp()
    )

    pre_start = (
        event_month -
        pd.DateOffset(months=6)
    )

    pre_end = (
        event_month -
        pd.DateOffset(months=1)
    )

    # --------------------------------------------------------
    # TREATMENT BASELINE
    # --------------------------------------------------------

    treatment_baseline = get_rx_baseline(
        treatment_hcp,
        target_ndc,
        pre_start,
        pre_end
    )

    # --------------------------------------------------------
    # START CONTROL CANDIDATES
    # --------------------------------------------------------

    candidates = set(control_pool)

    # Do not match HCP to themselves
    candidates.discard(treatment_hcp)

    # --------------------------------------------------------
    # REMOVE RECENTLY TREATED HCPs
    #
    # No attendee exposure during:
    #
    # event_date - 6 months <= exposure < event_date
    # --------------------------------------------------------

    six_month_start = (
        event_date -
        pd.DateOffset(months=6)
    )

    recent_attendees = attendee_exposure[
        (
            attendee_exposure["event_date"]
            >= six_month_start
        )
        &
        (
            attendee_exposure["event_date"]
            < event_date
        )
    ]

    recently_treated = set(
        recent_attendees["hcp_id"]
        .astype(str)
    )

    candidates -= recently_treated

    # --------------------------------------------------------
    # REMOVE CURRENT EVENT ATTENDEES
    # --------------------------------------------------------

    current_event_attendees = (
        current_event_attendees_lookup
        .get(event_id, set())
    )

    candidates -= current_event_attendees

    # --------------------------------------------------------
    # KEEP ONLY HCPs WITH RX HISTORY FOR TARGET CATEGORY
    # IN THE 6-MONTH PRE-PERIOD
    # --------------------------------------------------------

    valid_rx_controls = set()

    for candidate in candidates:

        baseline = get_rx_baseline(
            candidate,
            target_ndc,
            pre_start,
            pre_end
        )

        if not pd.isna(baseline):

            valid_rx_controls.add(candidate)

    candidates = valid_rx_controls

    # --------------------------------------------------------
    # PREVENT CONTROL REUSE WITHIN SAME EVENT
    # --------------------------------------------------------

    already_used = (
        used_controls_by_event
        .get(event_id, set())
    )

    candidates -= already_used

    # --------------------------------------------------------
    # NO VALID CONTROL
    # --------------------------------------------------------

    if len(candidates) == 0:

        results.append({

            "treatment_hcp_id":
                treatment_hcp,

            "control_hcp_id":
                "",

            "method":
                "random",

            "event_id":
                event_id,

            "event_date":
                event_date.strftime("%Y-%m-%d"),

            "target_ndc_category":
                target_ndc,

            "specialty":
                treatment_row["specialty"],

            "region":
                treatment_row["region"],

            "pre_period_rx_baseline":
                treatment_baseline,

            "control_pre_period_rx_baseline":
                np.nan,

            "is_matched":
                False,

            "match_rank":
                pd.NA
        })

    else:

        # ----------------------------------------------------
        # PURE RANDOM SELECTION
        # ----------------------------------------------------

        candidate_list = list(candidates)

        control_hcp = random_generator.choice(
            candidate_list
        )

        # ----------------------------------------------------
        # CONTROL BASELINE
        # ----------------------------------------------------

        control_baseline = get_rx_baseline(
            control_hcp,
            target_ndc,
            pre_start,
            pre_end
        )

        # ----------------------------------------------------
        # SAVE USED CONTROL
        # ----------------------------------------------------

        if event_id not in used_controls_by_event:

            used_controls_by_event[event_id] = set()

        used_controls_by_event[
            event_id
        ].add(control_hcp)

        # ----------------------------------------------------
        # SAVE MATCH
        # ----------------------------------------------------

        results.append({

            "treatment_hcp_id":
                treatment_hcp,

            "control_hcp_id":
                control_hcp,

            "method":
                "random",

            "event_id":
                event_id,

            "event_date":
                event_date.strftime("%Y-%m-%d"),

            "target_ndc_category":
                target_ndc,

            "specialty":
                treatment_row["specialty"],

            "region":
                treatment_row["region"],

            "pre_period_rx_baseline":
                treatment_baseline,

            "control_pre_period_rx_baseline":
                control_baseline,

            "is_matched":
                True,

            "match_rank":
                1
        })

    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if case_number % 100 == 0:

        print(
            f"Processed {case_number:,} "
            f"of {total_cases:,} treatment cases..."
        )


# ============================================================
# 16. CREATE FINAL DATAFRAME
# ============================================================

print("\nCreating final output...")

output = pd.DataFrame(results)

# ============================================================
# 17. EXACT REQUIRED COLUMN ORDER
# ============================================================

required_columns = [

    "treatment_hcp_id",

    "control_hcp_id",

    "method",

    "event_id",

    "event_date",

    "target_ndc_category",

    "specialty",

    "region",

    "pre_period_rx_baseline",

    "control_pre_period_rx_baseline",

    "is_matched",

    "match_rank"
]

output = output[
    required_columns
]

# ============================================================
# 18. SAVE CSV
# ============================================================

print("\nSaving CSV...")

output.to_csv(
    output_file,
    index=False
)

# ============================================================
# 19. VALIDATION
# ============================================================

matched = output[
    output["is_matched"] == True
].copy()

unmatched = output[
    output["is_matched"] == False
].copy()

print("\n============================================")
print("       RANDOM MATCHING COMPLETED")
print("============================================")

print(
    "Treatment cases:",
    len(output)
)

print(
    "Matched:",
    len(matched)
)

print(
    "Unmatched:",
    len(unmatched)
)

if len(output) > 0:

    match_rate = (
        output["is_matched"]
        .mean() * 100
    )

    print(
        "Match rate:",
        round(match_rate, 2),
        "%"
    )

# ============================================================
# 20. VALIDATION RESULTS
# ============================================================

print("\n============================================")
print("              VALIDATION")
print("============================================")

# Missing control IDs
missing_control_id = matched[
    "control_hcp_id"
].isna().sum()

print(
    "Matched rows with missing control ID:",
    missing_control_id
)

# Empty control IDs
empty_control_id = (
    matched["control_hcp_id"]
    .astype(str)
    .str.strip()
    .eq("")
    .sum()
)

print(
    "Matched rows with empty control ID:",
    empty_control_id
)

# Missing control baseline
missing_control_baseline = (
    matched[
        "control_pre_period_rx_baseline"
    ]
    .isna()
    .sum()
)

print(
    "Matched rows with missing control baseline:",
    missing_control_baseline
)

# Self matches
self_matches = (
    matched["treatment_hcp_id"]
    ==
    matched["control_hcp_id"]
).sum()

print(
    "Self matches:",
    self_matches
)

# Duplicate treatment cases
duplicate_treatment_cases = (
    output
    .duplicated(
        [
            "treatment_hcp_id",
            "event_id"
        ]
    )
    .sum()
)

print(
    "Duplicate treatment cases:",
    duplicate_treatment_cases
)

# Duplicate controls within event
duplicate_controls_same_event = (
    matched
    .duplicated(
        [
            "event_id",
            "control_hcp_id"
        ]
    )
    .sum()
)

print(
    "Duplicate controls within same event:",
    duplicate_controls_same_event
)

# ============================================================
# 21. FINAL FILE INFORMATION
# ============================================================

print("\n============================================")
print("             FILE SAVED SUCCESSFULLY")
print("============================================")

print(
    "CSV file name:",
    "matching_output_random.csv"
)

print(
    "Full file location:"
)

print(
    output_file
)

print(
    "\nNumber of rows in CSV:",
    len(output)
)

print(
    "Number of columns:",
    len(output.columns)
)

print("\nDone!")