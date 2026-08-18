#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 SPEAKER PROGRAM & PEER-TO-PEER ROI ANALYSIS  --  SYNTHETIC DATA GENERATOR
================================================================================

Produces a realistic, deliberately-messy synthetic pharma commercial dataset for
a Speaker Program ROI hackathon, with a KNOWN ground-truth causal effect baked in
so that competing matching methods (PSM, exact/stratified, nearest-neighbour on
covariates, random) can be scored against truth.

Design contract
---------------
  * TRUE EFFECT      : +15% lift on the event's target_ndc_category only,
                       for treatment HCPs only, starting at their event_date.
  * CONFOUNDING      : HCPs who attend speaker programs are selected on a latent
                       engagement/growth propensity that is correlated with
                       observable covariates. Their Rx *trend* is steeper even
                       absent any event, so a naive DiD against random controls
                       overstates the effect. Matching on covariates is what
                       claws it back toward 15%.
  * SPEAKERS         : speaker_eligible_flag=True HCPs skew high-volume and
                       senior. Actual speakers are held OUT of the treatment
                       pool and must be excluded from the control pool by all
                       matching methods (they are the most extreme confounders).
  * MESSINESS        : injected only into non-protected fields, after clean
                       generation, and logged row-by-row.

Reproducibility
---------------
  * One SEED (42) feeds a SeedSequence; every generation step draws from a named
    child stream, so adding/removing draws inside one step cannot shift another.
  * GENERATION_DATE is a fixed constant -- nothing derives from "today".
  * CSVs written with an explicit "\\n" line terminator so SHA-256 digests are
    stable across platforms.

Run
---
    python generate_data.py

Outputs (6 files) are written to /mnt/user-data/outputs on POSIX, otherwise to
./generated_data next to this script -- the upstream source that the team's
preprocessing step turns into ./preprocessed_data. Override with the
HACKATHON_OUTPUT_DIR env var.

================================================================================
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import platform
import random
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

# ==============================================================================
# SECTION 0 -- CONSTANTS  (everything tunable lives here)
# ==============================================================================

SEED = 42

# Fixed extract date. NEVER use datetime.date.today(): reruns must be identical.
GENERATION_DATE = dt.date(2025, 11, 15)

# ---- scale -------------------------------------------------------------------
N_HCPS = 1200
N_EVENTS = 200
N_TREATMENT_HCPS = 540          # unique role=attendee HCPs (spec: 500-600)
MIN_UNATTENDED_PER_SPECIALTY = 35   # hard-reserved control pool (spec: >=30)

# ---- timeline ----------------------------------------------------------------
# 21-month event window (spec: 18-24 months).
EVENT_WINDOW_START = dt.date(2023, 7, 1)
EVENT_WINDOW_END = dt.date(2025, 3, 31)

# Rx panel: one full month of slack beyond the required >=6 before earliest event
# (2023-07 -> 2023-01) and >=6 after latest event (2025-03 -> 2025-09).
RX_START_MONTH = "2022-12"
RX_END_MONTH = "2025-10"

# ---- causal design -----------------------------------------------------------
TRUE_EFFECT_PCT = 15.0          # the number every matching method is chasing
DID_WINDOW_MONTHS = 6           # 6 pre / 6 post around the anchor event month
# The event month itself is a washout: partially exposed, excluded from pre&post.

# Strength of the trend confounder. Raise -> naive DiD drifts further above 15%.
TREND_CONFOUND_LOADING = 0.0185
TREND_BASE_MONTHLY = 0.0016
TREND_IDIOSYNCRATIC_SD = 0.0020
TREND_CLIP = (-0.0220, 0.0350)

# How much of the engagement propensity is NOT explained by observable
# covariates. This single number sets the difficulty of the whole exercise:
# it is the share of the confounding that NO matching method can remove, so it
# must stay small enough that a well-specified model lands near the true 15%.
PROPENSITY_UNOBSERVED_SD = 0.26
# Sharpness of attendee selection on that propensity. Higher = more bias but
# thinner common support for PSM.
SELECTION_SHARPNESS = 2.10

# Gamma-Poisson dispersion on monthly Rx counts. Higher shape = tighter noise,
# so the effect a team recovers sits closer to the 15% design parameter.
RX_DISPERSION_SHAPE = 100.0

# ---- taxonomy ----------------------------------------------------------------
SPECIALTIES = [
    "oncology", "cardiology", "endocrinology", "primary care",
    "neurology", "pulmonology", "rheumatology",
]

# Sums to N_HCPS. Every specialty is far above the ~80 minimum.
SPECIALTY_COUNTS = {
    "primary care": 260,
    "cardiology": 190,
    "oncology": 175,
    "endocrinology": 165,
    "neurology": 150,
    "pulmonology": 140,
    "rheumatology": 120,
}

# Two portfolio-wide products every specialty writes. These are the categories
# that make cross-specialty attendance coherent (a neurologist can attend a
# shared-product program and still have Rx rows in that category).
SHARED_NDC_CATEGORIES = ["CV_ANTICOAG_ORAL", "METABOLIC_GLP1_INJ"]

# One specialty-anchored product per specialty.
SPECIALTY_PRIMARY_NDC = {
    "oncology": "ONC_TARGETED_ORAL",
    "cardiology": "CARD_HF_ARNI",
    "endocrinology": "ENDO_BASAL_INSULIN",
    "primary care": "PC_ANTIHYPERTENSIVE",
    "neurology": "NEURO_MIGRAINE_CGRP",
    "pulmonology": "PULM_ICS_LABA_INH",
    "rheumatology": "RHEUM_JAK_ORAL",
}

# Market-level monthly drift per category (a COMMON shock -- DiD differences it
# out, which is exactly why it is safe to include).
NDC_MARKET_DRIFT = {
    "CV_ANTICOAG_ORAL": 0.0020,
    "METABOLIC_GLP1_INJ": 0.0084,     # GLP-1 class boom
    "ONC_TARGETED_ORAL": 0.0041,
    "CARD_HF_ARNI": 0.0033,
    "ENDO_BASAL_INSULIN": -0.0021,    # eroding class
    "PC_ANTIHYPERTENSIVE": 0.0006,
    "NEURO_MIGRAINE_CGRP": 0.0062,
    "PULM_ICS_LABA_INH": 0.0014,
    "RHEUM_JAK_ORAL": 0.0027,
}

# Real NUCC provider taxonomy codes.
TAXONOMY_CODES = {
    "oncology": "207RX0202X",        # Internal Medicine / Medical Oncology
    "cardiology": "207RC0000X",      # Internal Medicine / Cardiovascular Disease
    "endocrinology": "207RE0101X",   # Endocrinology, Diabetes & Metabolism
    "primary care": "207Q00000X",    # Family Medicine
    "neurology": "2084N0400X",       # Psychiatry & Neurology / Neurology
    "pulmonology": "207RP1001X",     # Internal Medicine / Pulmonary Disease
    "rheumatology": "207RR0500X",    # Internal Medicine / Rheumatology
}

SUB_SPECIALTIES = {
    "oncology": ["breast", "thoracic", "gi malignancies", "hematologic",
                 "genitourinary", "melanoma/skin"],
    "cardiology": ["heart failure", "electrophysiology", "interventional",
                   "preventive", "structural heart"],
    "endocrinology": ["diabetes", "thyroid", "obesity medicine",
                      "bone & mineral", "reproductive endocrinology"],
    "primary care": ["family medicine", "internal medicine", "geriatrics",
                     "adolescent medicine", "urgent care"],
    "neurology": ["multiple sclerosis", "headache", "movement disorders",
                  "epilepsy", "neuromuscular", "stroke"],
    "pulmonology": ["asthma/allergy", "copd", "interstitial lung disease",
                    "sleep medicine", "critical care"],
    "rheumatology": ["rheumatoid arthritis", "lupus", "psoriatic arthritis",
                     "vasculitis", "osteoarthritis"],
}

PRACTICE_SETTINGS = ["hospital", "private", "academic"]

# Setting mix varies by specialty -- one of several covariates that keep
# baseline_rx_volume_monthly from being collinear with specialty.
SETTING_PROBS = {
    "oncology":      {"academic": 0.30, "hospital": 0.42, "private": 0.28},
    "cardiology":    {"academic": 0.16, "hospital": 0.40, "private": 0.44},
    "endocrinology": {"academic": 0.18, "hospital": 0.30, "private": 0.52},
    "primary care":  {"academic": 0.07, "hospital": 0.28, "private": 0.65},
    "neurology":     {"academic": 0.24, "hospital": 0.36, "private": 0.40},
    "pulmonology":   {"academic": 0.17, "hospital": 0.40, "private": 0.43},
    "rheumatology":  {"academic": 0.16, "hospital": 0.27, "private": 0.57},
}

REGIONS = ["Northeast", "Southeast", "Midwest", "Southwest", "West"]
REGION_PROBS = [0.21, 0.24, 0.21, 0.13, 0.21]

# Region -> state -> representative cities. Keeps geography internally consistent
# (Faker's city() is not state-aware).
GEOGRAPHY = {
    "Northeast": {
        "NY": ["New York", "Buffalo", "Rochester", "Albany", "Syracuse"],
        "MA": ["Boston", "Worcester", "Springfield", "Cambridge"],
        "PA": ["Philadelphia", "Pittsburgh", "Allentown", "Hershey"],
        "NJ": ["Newark", "Jersey City", "Princeton", "Camden"],
        "CT": ["Hartford", "New Haven", "Stamford"],
        "ME": ["Portland", "Bangor"],
    },
    "Southeast": {
        "FL": ["Miami", "Orlando", "Tampa", "Jacksonville", "Gainesville"],
        "GA": ["Atlanta", "Savannah", "Augusta", "Macon"],
        "NC": ["Charlotte", "Raleigh", "Durham", "Winston-Salem"],
        "TN": ["Nashville", "Memphis", "Knoxville", "Chattanooga"],
        "VA": ["Richmond", "Norfolk", "Charlottesville", "Arlington"],
        "AL": ["Birmingham", "Mobile", "Huntsville"],
        "SC": ["Charleston", "Columbia", "Greenville"],
    },
    "Midwest": {
        "IL": ["Chicago", "Springfield", "Peoria", "Rockford"],
        "OH": ["Cleveland", "Columbus", "Cincinnati", "Toledo"],
        "MI": ["Detroit", "Ann Arbor", "Grand Rapids", "Lansing"],
        "MN": ["Minneapolis", "St. Paul", "Rochester"],
        "WI": ["Milwaukee", "Madison", "Green Bay"],
        "MO": ["St. Louis", "Kansas City", "Columbia"],
        "IN": ["Indianapolis", "Fort Wayne", "South Bend"],
    },
    "Southwest": {
        "TX": ["Houston", "Dallas", "Austin", "San Antonio", "Fort Worth"],
        "AZ": ["Phoenix", "Tucson", "Scottsdale", "Mesa"],
        "NM": ["Albuquerque", "Santa Fe"],
        "OK": ["Oklahoma City", "Tulsa"],
        "NV": ["Las Vegas", "Reno"],
    },
    "West": {
        "CA": ["Los Angeles", "San Francisco", "San Diego", "Sacramento",
               "San Jose", "Palo Alto"],
        "WA": ["Seattle", "Spokane", "Tacoma"],
        "OR": ["Portland", "Eugene"],
        "CO": ["Denver", "Boulder", "Colorado Springs"],
        "UT": ["Salt Lake City", "Provo"],
    },
}

# Regional prescribing intensity (log-scale). A genuine, non-specialty covariate
# that PSM can pick up on.
REGION_RX_EFFECT = {
    "Northeast": 0.10, "Southeast": 0.02, "Midwest": -0.04,
    "Southwest": 0.07, "West": -0.09,
}
SETTING_RX_EFFECT = {"hospital": 0.06, "private": 0.00, "academic": 0.14}

# rx_volume is TRx (total prescriptions incl. refills), not NRx. The scale is
# set so the top few percent of writer-months genuinely reach four digits --
# which is what makes the injected "1,240"-style comma corruption a real parsing
# hazard rather than a no-op. Every ratio in the study is scale-invariant.
RX_SCALE = 2.8

# Specialty anchors for patient volume / Rx volume (deliberately overlapping).
SPECIALTY_BASELINE = {
    #                 patients/mo   rx/mo (pre-RX_SCALE)
    "primary care":   (330.0,       196.0),
    "cardiology":     (205.0,       168.0),
    "oncology":       (128.0,       121.0),
    "endocrinology":  (188.0,       182.0),
    "neurology":      (162.0,       139.0),
    "pulmonology":    (198.0,       152.0),
    "rheumatology":   (176.0,       165.0),
}
SPECIALTY_BASELINE = {k: (p, r * RX_SCALE) for k, (p, r) in SPECIALTY_BASELINE.items()}

SETTING_PATIENT_MULT = {"hospital": 1.10, "private": 1.00, "academic": 0.82}

# ---- events ------------------------------------------------------------------
VENUE_TYPES = ["in-person", "virtual"]
VENUE_PROBS = [0.70, 0.30]
SPEAKER_ELIGIBLE_SHARE = 0.20    # top 20% within each specialty
MAX_EVENTS_PER_SPEAKER = 3
PROB_EVENT_TARGETS_SPECIALTY_DRUG = 0.65   # else one of the shared products
# Relative pull of an out-of-specialty program (only ever a shared-product one).
CROSS_SPECIALTY_ATTENDANCE_WEIGHT = 0.20

EVENT_TOPIC_WORDS = {
    "ONC_TARGETED_ORAL": "Targeted Oral Therapy in Advanced Disease",
    "CARD_HF_ARNI": "Guideline-Directed Therapy in Heart Failure",
    "ENDO_BASAL_INSULIN": "Basal Insulin Optimization",
    "PC_ANTIHYPERTENSIVE": "Resistant Hypertension in Primary Care",
    "NEURO_MIGRAINE_CGRP": "CGRP Pathway Inhibition in Migraine",
    "PULM_ICS_LABA_INH": "Inhaled Maintenance Therapy in COPD & Asthma",
    "RHEUM_JAK_ORAL": "Oral JAK Inhibition in Inflammatory Arthritis",
    "CV_ANTICOAG_ORAL": "Anticoagulation Across Comorbid Populations",
    "METABOLIC_GLP1_INJ": "Incretin-Based Therapy: Cardiometabolic Evidence",
}
EVENT_FORMAT_WORDS = [
    "Speaker Program", "Peer-to-Peer Exchange", "Regional Dinner Program",
    "Advisory Roundtable", "Clinical Update Series", "Case-Based Forum",
]

# ---- data-quality injection rates -------------------------------------------
RATE_HCP_MISSING_SUBSPEC = 0.03
RATE_HCP_REGION_VARIANT = 0.02
RATE_HCP_DUPLICATE_ROW = 0.01
RATE_HCP_TAXONOMY_FORMAT = 0.02

RATE_ATT_DUPLICATE = 0.04
RATE_ATT_MISSING_ENGAGEMENT = 0.02
RATE_ATT_DATE_DRIFT = 0.01

RATE_RX_MISSING = 0.02
RATE_RX_COMMA_STRING = 0.01
N_RX_OUTLIERS = 13

RATE_EVT_SPEND_BAD = 0.02

# Messy region spellings -> canonical. This IS the clean lookup table the
# hackathon teams are expected to rebuild; it is printed to console and every
# individual substitution is recorded in the injection log.
REGION_ALIAS_LOOKUP = {
    "NE": "Northeast", "north east": "Northeast", "N.E.": "Northeast",
    "NORTHEAST": "Northeast",
    "SE": "Southeast", "south east": "Southeast", "S.E.": "Southeast",
    "SOUTHEAST": "Southeast",
    "MW": "Midwest", "mid west": "Midwest", "Mid-West": "Midwest",
    "MIDWEST": "Midwest",
    "SW": "Southwest", "south west": "Southwest", "S.W.": "Southwest",
    "SOUTHWEST": "Southwest",
    "W": "West", "west coast": "West", "WEST": "West", "Western": "West",
}
REGION_VARIANTS = {
    "Northeast": ["NE", "north east", "N.E.", "NORTHEAST"],
    "Southeast": ["SE", "south east", "S.E.", "SOUTHEAST"],
    "Midwest": ["MW", "mid west", "Mid-West", "MIDWEST"],
    "Southwest": ["SW", "south west", "S.W.", "SOUTHWEST"],
    "West": ["W", "west coast", "WEST", "Western"],
}

# ---- output ------------------------------------------------------------------
CANONICAL_OUTPUT_DIR = "/mnt/user-data/outputs"
# Local project layout: raw synthetic output lands in ./generated_data, which is
# the upstream source the team's preprocessing step reads to produce
# ./preprocessed_data. Renaming this would orphan that hand-off.
LOCAL_OUTPUT_DIRNAME = "generated_data"
LINE_TERMINATOR = "\n"          # explicit -> platform-independent SHA-256

FILE_HCP = "hcp.csv"
FILE_EVENTS = "events.csv"
FILE_ATTENDANCE = "attendance.csv"
FILE_RX = "rx_claims_monthly.csv"
FILE_GROUND_TRUTH = "ground_truth_config.json"
FILE_INJECTION_LOG = "data_quality_injection_log.csv"
OUTPUT_FILES = [FILE_HCP, FILE_EVENTS, FILE_ATTENDANCE, FILE_RX,
                FILE_GROUND_TRUTH, FILE_INJECTION_LOG]


# ==============================================================================
# SECTION 1 -- REPRODUCIBILITY PLUMBING
# ==============================================================================

def rng_for(stream_name: str) -> np.random.Generator:
    """Named child RNG stream derived from the single SEED.

    Using one entropy pool per generation step means a change in how many draws
    step A makes cannot shift the numbers step B produces. All streams still
    trace back to SEED alone. zlib.crc32 is used instead of hash() because
    Python's string hashing is salted per process.
    """
    tag = zlib.crc32(stream_name.encode("utf-8"))
    return np.random.default_rng(np.random.SeedSequence([SEED, tag]))


# Faker and stdlib random are seeded once, globally, from the same SEED.
Faker.seed(SEED)
random.seed(SEED)
FAKE = Faker("en_US")


def resolve_output_dir() -> Path:
    """/mnt/user-data/outputs on POSIX; ./generated_data locally. Env var wins."""
    override = os.environ.get("HACKATHON_OUTPUT_DIR", "").strip()
    if override:
        path = Path(override)
    elif os.name == "posix":
        path = Path(CANONICAL_OUTPUT_DIR)
    else:
        path = Path(__file__).resolve().parent / LOCAL_OUTPUT_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return path


OUTPUT_DIR = resolve_output_dir()


# ==============================================================================
# SECTION 2 -- SMALL HELPERS
# ==============================================================================

def banner(title: str, char: str = "=") -> None:
    print("\n" + char * 78)
    print(title)
    print(char * 78)


def step(title: str) -> None:
    print("\n" + "-" * 78)
    print(title)
    print("-" * 78)


def describe_numeric(df: pd.DataFrame, cols, indent: str = "    ") -> None:
    """min / max / mean / median sanity line per numeric column."""
    for col in cols:
        series = pd.to_numeric(df[col], errors="coerce")
        print(
            f"{indent}{col:<32s} min={series.min():>10,.2f}  "
            f"max={series.max():>12,.2f}  mean={series.mean():>10,.2f}  "
            f"median={series.median():>10,.2f}"
        )


def report_nulls(df: pd.DataFrame, name: str, indent: str = "    ") -> None:
    nulls = df.isna().sum()
    nulls = nulls[nulls > 0]
    if nulls.empty:
        print(f"{indent}nulls: none in any of {df.shape[1]} columns  [{name}]")
    else:
        print(f"{indent}nulls [{name}]:")
        for col, n in nulls.items():
            print(f"{indent}  {col:<30s} {n:>6,d}  ({n / len(df):.2%})")


def month_range(start_ym: str, end_ym: str) -> list[str]:
    """Inclusive list of 'YYYY-MM' strings."""
    sy, sm = (int(x) for x in start_ym.split("-"))
    ey, em = (int(x) for x in end_ym.split("-"))
    out, y, m = [], sy, sm
    while (y, m) <= (ey, em):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def date_to_month(d: dt.date) -> str:
    return f"{d.year:04d}-{d.month:02d}"


def shift_month(ym: str, k: int) -> str:
    y, m = (int(x) for x in ym.split("-"))
    idx = y * 12 + (m - 1) + k
    return f"{idx // 12:04d}-{idx % 12 + 1:02d}"


def npi_check_digit(nine_digit_body: str) -> str:
    """Real NPI check digit: Luhn over '80840' + 9-digit identifier."""
    payload = "80840" + nine_digit_body
    total = 0
    for i, ch in enumerate(reversed(payload)):
        d = int(ch)
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return str((10 - total % 10) % 10)


def standardize(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return (x - x.mean()) / (sd if sd > 0 else 1.0)


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """Deterministic CSV write (explicit newline, empty string for NA)."""
    df.to_csv(path, index=False, lineterminator=LINE_TERMINATOR, na_rep="")


class InjectionLog:
    """Row-level ledger of every field this script deliberately dirtied."""

    def __init__(self) -> None:
        self._rows: list[dict] = []

    def add(self, target_file: str, issue_type: str, row_key: str, column: str,
            original_value, new_value, notes: str = "") -> None:
        self._rows.append({
            "log_id": f"DQ-{len(self._rows) + 1:06d}",
            "target_file": target_file,
            "issue_type": issue_type,
            "row_key": row_key,
            "column": column,
            "original_value": "" if original_value is None
                              else ("<NULL>" if (isinstance(original_value, float)
                                                 and math.isnan(original_value))
                                    else str(original_value)),
            "new_value": "" if new_value is None
                         else ("<NULL>" if (isinstance(new_value, float)
                                            and math.isnan(new_value))
                               else str(new_value)),
            "notes": notes,
        })

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows, columns=[
            "log_id", "target_file", "issue_type", "row_key", "column",
            "original_value", "new_value", "notes",
        ])

    def __len__(self) -> int:
        return len(self._rows)


# ==============================================================================
# SECTION 3 -- CLEAN GENERATION
# ==============================================================================

def generate_hcp_master() -> pd.DataFrame:
    """1,200 HCPs with genuinely multivariate, non-collinear covariates.

    baseline_rx_volume_monthly is driven by patient volume, tenure (concave),
    region and practice setting, plus a large idiosyncratic term -- so specialty
    shifts the distribution but nowhere near determines it. That overlap is what
    gives propensity-score matching something real to model.

    speaker_eligible_flag is assigned as the top SPEAKER_ELIGIBLE_SHARE *within
    each specialty*, ranked on an index of Rx volume + tenure + academic setting.
    The skew is intentional: it is precisely why speakers are toxic controls and
    must be filtered out of the control pool by every matching method.
    """
    step("STEP 1/5  Generating hcp.csv")
    rng = rng_for("hcp")

    rows = []
    for specialty in SPECIALTIES:                    # fixed order -> stable
        n = SPECIALTY_COUNTS[specialty]
        pat_anchor, rx_anchor = SPECIALTY_BASELINE[specialty]
        setting_probs = SETTING_PROBS[specialty]
        settings = rng.choice(PRACTICE_SETTINGS, size=n,
                              p=[setting_probs[s] for s in PRACTICE_SETTINGS])
        regions = rng.choice(REGIONS, size=n, p=REGION_PROBS)

        # Tenure: gamma, right-skewed, 1-42 years.
        years = np.clip(np.round(rng.gamma(3.2, 4.2, size=n)) + 1, 1, 42).astype(int)

        for i in range(n):
            setting = str(settings[i])
            region = str(regions[i])
            yrs = int(years[i])

            state = str(rng.choice(sorted(GEOGRAPHY[region].keys())))
            city = str(rng.choice(GEOGRAPHY[region][state]))

            # Patient volume: specialty anchor x setting x tenure ramp x noise.
            tenure_ramp = 0.72 + 0.36 * (1.0 - math.exp(-yrs / 7.0))
            patient_vol = (
                pat_anchor
                * SETTING_PATIENT_MULT[setting]
                * tenure_ramp
                * math.exp(float(rng.normal(0.0, 0.26)))
            )
            patient_vol = int(np.clip(round(patient_vol), 25, 900))

            # Rx volume: multivariate, heavy idiosyncratic component.
            log_rx = (
                math.log(rx_anchor)
                + 0.55 * math.log(patient_vol / pat_anchor)
                + 0.0180 * (yrs - 14)
                - 0.00042 * (yrs - 14) ** 2
                + REGION_RX_EFFECT[region]
                + SETTING_RX_EFFECT[setting]
                + float(rng.normal(0.0, 0.32))
            )
            baseline_rx = int(np.clip(round(math.exp(log_rx)),
                                      12 * RX_SCALE, 1400 * RX_SCALE))

            rows.append({
                "specialty": specialty,
                "sub_specialty": str(rng.choice(SUB_SPECIALTIES[specialty])),
                "region": region,
                "state": state,
                "city": city,
                "years_in_practice": yrs,
                "practice_setting": setting,
                "patient_volume_monthly": patient_vol,
                "baseline_rx_volume_monthly": baseline_rx,
                "npi_taxonomy_code": TAXONOMY_CODES[specialty],
            })

    hcp = pd.DataFrame(rows)

    # ---- NPI-style ids: Luhn-valid, prefixed so they all start with 1 --------
    bodies = rng.choice(np.arange(100_000_000, 200_000_000),
                        size=len(hcp) * 2, replace=False)[: len(hcp)]
    hcp["hcp_id"] = [f"{b}{npi_check_digit(str(b))}" for b in bodies]

    # ---- names (Faker, globally seeded) -------------------------------------
    hcp["first_name"] = [FAKE.first_name() for _ in range(len(hcp))]
    hcp["last_name"] = [FAKE.last_name() for _ in range(len(hcp))]

    # ---- speaker eligibility: top share WITHIN specialty ---------------------
    elig_index = (
        1.15 * standardize(np.log(hcp["baseline_rx_volume_monthly"].to_numpy()))
        + 0.85 * standardize(hcp["years_in_practice"].to_numpy().astype(float))
        + 0.35 * (hcp["practice_setting"] == "academic").to_numpy()
        + 0.20 * (hcp["practice_setting"] == "hospital").to_numpy()
        + rng.normal(0.0, 0.45, size=len(hcp))
    )
    hcp["_elig_index"] = elig_index
    hcp["speaker_eligible_flag"] = False
    for _specialty, grp in hcp.groupby("specialty", sort=True):
        k = int(round(len(grp) * SPEAKER_ELIGIBLE_SHARE))
        top = grp["_elig_index"].nlargest(k).index
        hcp.loc[top, "speaker_eligible_flag"] = True

    # ---- latent engagement/growth propensity (NOT written to disk) ----------
    # This is the confounder: reps target growing, high-volume, accessible HCPs.
    # It is mostly a function of observables (so matching can partially undo it)
    # plus an unobserved residual (so matching cannot fully undo it).
    # NOTE: speaker_eligible_flag is deliberately NOT a term here. Speakers are
    # barred from the control pool, so any confounding routed through
    # eligibility would be unmatchable by construction -- an unfair, invisible
    # floor under every method's error. Eligibility still correlates with
    # propensity through Rx volume and tenure, which ARE matchable.
    prop_raw = (
        0.55 * standardize(np.log(hcp["baseline_rx_volume_monthly"].to_numpy()))
        + 0.25 * standardize(np.log(hcp["patient_volume_monthly"].to_numpy()))
        - 0.20 * standardize(hcp["years_in_practice"].to_numpy().astype(float))
        + 0.30 * hcp["practice_setting"].isin(["academic", "hospital"]).to_numpy()
        + hcp["region"].map(REGION_RX_EFFECT).to_numpy() * 1.4
        + rng.normal(0.0, PROPENSITY_UNOBSERVED_SD, size=len(hcp))
    )
    hcp["_propensity"] = standardize(prop_raw)

    # Individual Rx growth trend, loaded on the same propensity -> the parallel
    # trends violation that matching is supposed to repair.
    hcp["_trend"] = np.clip(
        TREND_BASE_MONTHLY
        + TREND_CONFOUND_LOADING * hcp["_propensity"].to_numpy()
        + rng.normal(0.0, TREND_IDIOSYNCRATIC_SD, size=len(hcp)),
        TREND_CLIP[0], TREND_CLIP[1],
    )

    # Per-HCP category mix (Dirichlet -> real heterogeneity in product split).
    mix = rng.dirichlet([9.0, 6.0, 5.0], size=len(hcp))
    hcp["_mix_primary"] = mix[:, 0]
    hcp["_mix_shared0"] = mix[:, 1]
    hcp["_mix_shared1"] = mix[:, 2]

    hcp = hcp.sort_values("hcp_id", kind="stable").reset_index(drop=True)
    hcp = hcp[[
        "hcp_id", "first_name", "last_name", "specialty", "sub_specialty",
        "region", "state", "city", "years_in_practice", "practice_setting",
        "patient_volume_monthly", "baseline_rx_volume_monthly",
        "speaker_eligible_flag", "npi_taxonomy_code",
        "_elig_index", "_propensity", "_trend",
        "_mix_primary", "_mix_shared0", "_mix_shared1",
    ]]

    # ---- sanity ------------------------------------------------------------
    print(f"    rows: {len(hcp):,d}   unique hcp_id: {hcp['hcp_id'].nunique():,d}")
    report_nulls(hcp, "hcp")
    print("    specialty distribution:")
    for specialty, n in hcp["specialty"].value_counts().sort_index().items():
        n_elig = int(hcp.loc[hcp["specialty"] == specialty,
                             "speaker_eligible_flag"].sum())
        print(f"      {specialty:<16s} {n:>5,d}   speaker_eligible={n_elig:>4,d}")
    describe_numeric(hcp, ["years_in_practice", "patient_volume_monthly",
                           "baseline_rx_volume_monthly"])

    elig = hcp[hcp["speaker_eligible_flag"]]
    non = hcp[~hcp["speaker_eligible_flag"]]
    print(f"    speaker_eligible=True : {len(elig):,d} "
          f"({len(elig) / len(hcp):.1%})")
    print(f"      mean baseline_rx  eligible={elig['baseline_rx_volume_monthly'].mean():,.1f}"
          f"  vs non-eligible={non['baseline_rx_volume_monthly'].mean():,.1f}"
          f"   (intended skew, must be preserved)")
    print(f"      mean years        eligible={elig['years_in_practice'].mean():,.1f}"
          f"  vs non-eligible={non['years_in_practice'].mean():,.1f}")

    # Within-specialty Rx spread -> proof of non-collinearity with specialty.
    within = hcp.groupby("specialty")["baseline_rx_volume_monthly"]
    overall_sd = hcp["baseline_rx_volume_monthly"].std()
    within_sd = math.sqrt((within.var().mul(within.size() - 1).sum())
                          / (len(hcp) - len(SPECIALTIES)))
    print(f"    baseline_rx SD: overall={overall_sd:,.1f}  within-specialty="
          f"{within_sd:,.1f}  -> {within_sd / overall_sd:.0%} of variance is "
          f"NOT specialty (PSM has real signal)")

    # Solvability: how much of the latent growth trend (the confounder that
    # breaks parallel trends) is recoverable from the covariates teams can see?
    # 1 - R^2 is the irreducible error floor no matching method can get under.
    design = np.column_stack([
        np.ones(len(hcp)),
        standardize(np.log(hcp["baseline_rx_volume_monthly"].to_numpy())),
        standardize(np.log(hcp["patient_volume_monthly"].to_numpy())),
        standardize(hcp["years_in_practice"].to_numpy().astype(float)),
        pd.get_dummies(hcp["practice_setting"], drop_first=True).to_numpy(float),
        pd.get_dummies(hcp["region"], drop_first=True).to_numpy(float),
        pd.get_dummies(hcp["specialty"], drop_first=True).to_numpy(float),
    ])
    target = hcp["_trend"].to_numpy()
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    resid = target - design @ coef
    r2 = 1.0 - resid.var() / target.var()
    print(f"    latent trend confounder: {r2:.1%} explained by observable "
          f"covariates\n      -> ~{1 - r2:.0%} is unobserved, the error floor "
          f"no matching method can beat")
    return hcp


def generate_events(hcp: pd.DataFrame) -> pd.DataFrame:
    """200 speaker programs across a 21-month window.

    Every speaker_hcp_id is drawn from the speaker_eligible_flag=True pool of the
    event's own specialty_focus, capped at MAX_EVENTS_PER_SPEAKER events each.
    attendee_count is a placeholder here -- it is written back after attendance
    is generated, because downstream ROI needs program_spend / attendee_count.
    """
    step("STEP 2/5  Generating events.csv")
    rng = rng_for("events")

    # Events per specialty, proportional to the speaker-eligible pool.
    elig_by_spec = (hcp[hcp["speaker_eligible_flag"]]
                    .groupby("specialty").size().reindex(SPECIALTIES).fillna(0))
    raw = elig_by_spec / elig_by_spec.sum() * N_EVENTS
    counts = {s: int(math.floor(raw[s])) for s in SPECIALTIES}
    remainder = N_EVENTS - sum(counts.values())
    for s in (raw - pd.Series(counts)).sort_values(ascending=False).index[:remainder]:
        counts[s] += 1

    window_days = (EVENT_WINDOW_END - EVENT_WINDOW_START).days
    speaker_usage: dict[str, int] = {}
    rows = []
    event_no = 0

    for specialty in SPECIALTIES:
        pool = hcp[(hcp["specialty"] == specialty) &
                   (hcp["speaker_eligible_flag"])].copy()
        pool = pool.sort_values("hcp_id", kind="stable")
        for _ in range(counts[specialty]):
            event_no += 1
            # Speaker: prefer the least-used eligible speakers in this specialty.
            avail = pool[pool["hcp_id"].map(
                lambda h: speaker_usage.get(h, 0) < MAX_EVENTS_PER_SPEAKER)]
            weights = np.array([1.0 / (1.0 + 2.5 * speaker_usage.get(h, 0))
                                for h in avail["hcp_id"]])
            weights = weights / weights.sum()
            spk_idx = rng.choice(len(avail), p=weights)
            speaker = avail.iloc[int(spk_idx)]
            speaker_usage[speaker["hcp_id"]] = speaker_usage.get(speaker["hcp_id"], 0) + 1

            # Target product: specialty-anchored or portfolio-wide.
            if rng.random() < PROB_EVENT_TARGETS_SPECIALTY_DRUG:
                target = SPECIALTY_PRIMARY_NDC[specialty]
            else:
                target = str(rng.choice(SHARED_NDC_CATEGORIES))

            # Date: uniform over the window with mild seasonality (fewer in
            # late December / mid summer).
            for _ in range(24):
                offset = int(rng.integers(0, window_days + 1))
                cand = EVENT_WINDOW_START + dt.timedelta(days=offset)
                season = 0.35 if cand.month == 12 and cand.day > 15 else (
                    0.55 if cand.month in (7, 8) else 1.0)
                if rng.random() < season:
                    break
            event_date = cand

            venue = str(rng.choice(VENUE_TYPES, p=VENUE_PROBS))
            if venue == "in-person":
                capacity = int(rng.integers(25, 66))
                spend = float(np.exp(rng.normal(math.log(19500.0), 0.48)))
            else:
                capacity = int(rng.integers(45, 141))
                spend = float(np.exp(rng.normal(math.log(6200.0), 0.52)))

            rows.append({
                "event_id": f"EVT-{event_no:04d}",
                "event_name": (
                    f"{EVENT_TOPIC_WORDS[target]} - "
                    f"{speaker['city']} {str(rng.choice(EVENT_FORMAT_WORDS))}"
                ),
                "event_date": event_date,
                "speaker_hcp_id": speaker["hcp_id"],
                "specialty_focus": specialty,
                "region": speaker["region"],
                "program_spend": int(round(spend, -1)),
                "target_ndc_category": target,
                "venue_type": venue,
                "max_capacity": capacity,
                "attendee_count": 0,          # written back after attendance
            })

    events = pd.DataFrame(rows).sort_values("event_date", kind="stable")
    events["event_id"] = [f"EVT-{i + 1:04d}" for i in range(len(events))]
    events = events.reset_index(drop=True)

    print(f"    rows: {len(events):,d}   unique event_id: "
          f"{events['event_id'].nunique():,d}")
    report_nulls(events, "events")
    print(f"    event_date span: {events['event_date'].min()} .. "
          f"{events['event_date'].max()}  "
          f"({(events['event_date'].max() - events['event_date'].min()).days / 30.44:.1f} months)")
    print(f"    distinct speakers used: {events['speaker_hcp_id'].nunique():,d} "
          f"(max events per speaker = {events['speaker_hcp_id'].value_counts().max()})")
    print(f"    all speakers speaker_eligible=True: "
          f"{events['speaker_hcp_id'].isin(hcp.loc[hcp['speaker_eligible_flag'], 'hcp_id']).all()}")
    print("    target_ndc_category mix:")
    for cat, n in events["target_ndc_category"].value_counts().sort_index().items():
        print(f"      {cat:<24s} {n:>4,d}")
    print("    venue_type mix: " + ", ".join(
        f"{v}={n}" for v, n in events["venue_type"].value_counts().sort_index().items()))
    describe_numeric(events, ["program_spend", "max_capacity"])
    return events


def generate_attendance(hcp: pd.DataFrame,
                        events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """~4,500 attendance rows; returns (attendance, events_with_counts, anchors).

    ------------------------------------------------------------------------
    PRIMARY-EVENT ANCHOR RULE  (the single treatment definition for this study)
    ------------------------------------------------------------------------
    An HCP with role='attendee' is a TREATMENT unit. HCPs attend more than one
    program, so the study needs one unambiguous index date per unit:

        primary event = the event of the HCP's EARLIEST attendance_date;
                        ties broken by the lowest event_id (lexicographic).

    Everything downstream -- pre/post windows, target_ndc_category filtering,
    the +15% lift, and ground_truth_config.json -- keys off that single anchor.
    The rule is order-independent and idempotent, so it survives the duplicate
    rows injected later: de-duplicating on (hcp_id, event_id) reproduces exactly
    the same anchor.

    Pool discipline:
      * Actual speakers are EXCLUDED from the attendee pool entirely, so no HCP
        is simultaneously treatment and speaker (no ambiguous units).
      * MIN_UNATTENDED_PER_SPECIALTY HCPs per specialty are reserved at random
        BEFORE selection and never receive any attendance row -- a clean,
        covariate-representative control pool for all four matching methods.
    """
    step("STEP 3/5  Generating attendance.csv")
    rng = rng_for("attendance")

    speaker_ids = set(events["speaker_hcp_id"].unique())

    # ---- reserve a guaranteed never-attended control pool per specialty ------
    reserved: set[str] = set()
    for specialty in SPECIALTIES:
        cand = hcp[(hcp["specialty"] == specialty) &
                   (~hcp["hcp_id"].isin(speaker_ids))]["hcp_id"].to_numpy()
        pick = rng.choice(cand, size=MIN_UNATTENDED_PER_SPECIALTY, replace=False)
        reserved.update(pick.tolist())

    # ---- select treatment HCPs on the latent propensity ---------------------
    # exp-weighted sampling -> treated skew high-propensity, untreated skew low.
    eligible = hcp[(~hcp["hcp_id"].isin(speaker_ids)) &
                   (~hcp["hcp_id"].isin(reserved))].copy()
    w = np.exp(SELECTION_SHARPNESS * eligible["_propensity"].to_numpy())
    w = w / w.sum()
    treat_ids = rng.choice(eligible["hcp_id"].to_numpy(),
                           size=N_TREATMENT_HCPS, replace=False, p=w)
    treat_ids = set(treat_ids.tolist())

    treated = hcp[hcp["hcp_id"].isin(treat_ids)].copy()

    # ---- per-HCP attendance quota (3-segment mixture, tied to propensity) ---
    # Highly engaged physicians attend many programs; most attend a handful.
    u = rng.random(len(treated)) + 0.45 * standardize(
        treated["_propensity"].to_numpy())
    seg = np.where(u < np.quantile(u, 0.48), 0,
                   np.where(u < np.quantile(u, 0.85), 1, 2))
    quota = np.where(
        seg == 0, 1 + rng.poisson(1.8, len(treated)),
        np.where(seg == 1, 5 + rng.poisson(4.0, len(treated)),
                 13 + rng.poisson(9.0, len(treated))))
    treated = treated.assign(_quota=quota)

    # ---- event eligibility ---------------------------------------------------
    # Same-specialty events always; other-specialty events only when the program
    # targets a portfolio-wide product the HCP actually prescribes.
    ev = events.reset_index(drop=True)
    ev_specialty = ev["specialty_focus"].to_numpy()
    ev_region = ev["region"].to_numpy()
    ev_shared = ev["target_ndc_category"].isin(SHARED_NDC_CATEGORIES).to_numpy()
    ev_ids = ev["event_id"].to_numpy()
    headroom = ev["max_capacity"].to_numpy().astype(int).copy()

    assignments: dict[str, list[int]] = {}
    order = rng.permutation(len(treated))
    for pos in order:
        row = treated.iloc[int(pos)]
        ok = ((ev_specialty == row["specialty"]) | ev_shared) & (headroom > 0)
        if not ok.any():
            continue
        idx = np.flatnonzero(ok)
        weight = np.where(ev_specialty[idx] == row["specialty"], 1.0,
                          CROSS_SPECIALTY_ATTENDANCE_WEIGHT)
        weight = weight * np.where(ev_region[idx] == row["region"], 3.0, 1.0)
        weight = weight * (0.55 + headroom[idx] / headroom.max())
        weight = weight / weight.sum()
        take = int(min(row["_quota"], len(idx)))
        chosen = rng.choice(idx, size=take, replace=False, p=weight)
        headroom[chosen] -= 1
        assignments[row["hcp_id"]] = sorted(int(c) for c in chosen)

    # Repair pass: guarantee every event has at least a handful of attendees so
    # program_spend / attendee_count never divides by zero downstream.
    MIN_ATTENDEES = 4
    per_event: dict[int, list[str]] = {i: [] for i in range(len(ev))}
    for h, idxs in assignments.items():
        for i in idxs:
            per_event[i].append(h)
    treated_by_spec = {s: treated[treated["specialty"] == s]["hcp_id"].to_numpy()
                       for s in SPECIALTIES}
    for i in range(len(ev)):
        while len(per_event[i]) < MIN_ATTENDEES:
            pool = (treated["hcp_id"].to_numpy() if ev_shared[i]
                    else treated_by_spec[ev_specialty[i]])
            pool = np.array([h for h in pool if h not in per_event[i]])
            if len(pool) == 0:
                break
            pick = str(rng.choice(pool))
            per_event[i].append(pick)
            assignments.setdefault(pick, []).append(i)

    # ---- materialise attendance rows ----------------------------------------
    prop_by_id = dict(zip(hcp["hcp_id"], hcp["_propensity"]))
    ev_date = ev["event_date"].to_numpy()
    records = []

    for hcp_id in sorted(assignments):
        base = 5.6 + 1.25 * prop_by_id[hcp_id]
        for i in sorted(set(assignments[hcp_id])):
            records.append({
                "hcp_id": hcp_id,
                "event_id": ev_ids[i],
                "attendance_date": ev_date[i],
                "role": "attendee",
                "engagement_score": round(float(np.clip(
                    base + rng.normal(0.0, 1.5), 1.0, 10.0)), 1),
            })

    # Speakers: one row at their own event, role='speaker'.
    for i in range(len(ev)):
        records.append({
            "hcp_id": ev["speaker_hcp_id"].iloc[i],
            "event_id": ev_ids[i],
            "attendance_date": ev_date[i],
            "role": "speaker",
            "engagement_score": round(float(np.clip(
                rng.normal(9.1, 0.55), 1.0, 10.0)), 1),
        })

    attendance = pd.DataFrame(records)
    attendance = attendance.sort_values(
        ["attendance_date", "event_id", "role", "hcp_id"], kind="stable"
    ).reset_index(drop=True)
    attendance.insert(0, "attendance_id",
                      [f"ATT-{i + 1:06d}" for i in range(len(attendance))])

    # ---- write attendee_count back into events -------------------------------
    counts = (attendance[attendance["role"] == "attendee"]
              .groupby("event_id")["hcp_id"].nunique())
    events_out = events.copy()
    events_out["attendee_count"] = (events_out["event_id"].map(counts)
                                    .fillna(0).astype(int))

    # ---- anchor table (the treatment definition) -----------------------------
    anchors = compute_primary_event_anchors(attendance)

    # ---- sanity --------------------------------------------------------------
    n_attendee_rows = int((attendance["role"] == "attendee").sum())
    n_speaker_rows = int((attendance["role"] == "speaker").sum())
    treat_actual = set(anchors["hcp_id"])
    print(f"    rows: {len(attendance):,d}  "
          f"(attendee={n_attendee_rows:,d}, speaker={n_speaker_rows:,d})")
    report_nulls(attendance, "attendance")
    print(f"    unique treatment HCPs (role=attendee): {len(treat_actual):,d}")
    print(f"    treatment x speaker overlap: "
          f"{len(treat_actual & speaker_ids)}  (must be 0)")
    print(f"    events per treatment HCP: "
          f"min={attendance[attendance['role'] == 'attendee'].groupby('hcp_id').size().min()}, "
          f"max={attendance[attendance['role'] == 'attendee'].groupby('hcp_id').size().max()}, "
          f"mean={n_attendee_rows / len(treat_actual):.2f}")
    describe_numeric(attendance, ["engagement_score"])
    describe_numeric(events_out, ["attendee_count"])

    attended = set(attendance["hcp_id"])
    print("    control pool (non-treatment, non-speaker, zero attendance rows):")
    for specialty in SPECIALTIES:
        pool = hcp[(hcp["specialty"] == specialty) &
                   (~hcp["hcp_id"].isin(attended))]
        flag = "OK " if len(pool) >= 30 else "LOW"
        print(f"      {specialty:<16s} {len(pool):>5,d}  [{flag}] "
              f"(>=30 required)")
    print(f"      TOTAL unattended: {len(hcp) - len(attended):,d}")
    print("    treatment selection check (confounding is intentional):")
    t = hcp[hcp["hcp_id"].isin(treat_actual)]
    c = hcp[(~hcp["hcp_id"].isin(attended))]
    print(f"      mean baseline_rx   treatment={t['baseline_rx_volume_monthly'].mean():,.1f}"
          f"  control-pool={c['baseline_rx_volume_monthly'].mean():,.1f}")
    print(f"      mean latent trend  treatment={t['_trend'].mean() * 100:.3f}%/mo"
          f"  control-pool={c['_trend'].mean() * 100:.3f}%/mo"
          f"   -> parallel-trends violation by design")
    return attendance, events_out, anchors


def compute_primary_event_anchors(attendance: pd.DataFrame) -> pd.DataFrame:
    """Apply the PRIMARY-EVENT ANCHOR RULE (see generate_attendance docstring).

    Earliest attendance_date wins; ties broken by lowest event_id. Duplicate
    (hcp_id, event_id) rows are collapsed first, so this is idempotent under the
    duplicate-record injection.
    """
    att = attendance[attendance["role"] == "attendee"].copy()
    att = att.drop_duplicates(subset=["hcp_id", "event_id"], keep="first")
    att = att.sort_values(["hcp_id", "attendance_date", "event_id"], kind="stable")
    anchors = att.groupby("hcp_id", as_index=False).first()
    return anchors[["hcp_id", "event_id", "attendance_date"]].rename(
        columns={"attendance_date": "anchor_date"})


def generate_rx_claims(hcp: pd.DataFrame, events: pd.DataFrame,
                       anchors: pd.DataFrame) -> pd.DataFrame:
    """Continuous monthly Rx panel: every HCP x 3 ndc_categories x 35 months.

    Each HCP prescribes their specialty-anchored product plus both portfolio-wide
    products, so any event's target_ndc_category always has real rows to filter
    to (drug-specific DiD, not a flat total).

    rx_trend_baseline is the deterministic counterfactual expectation (level x
    individual trend x market drift x seasonality, NO lift, NO noise).
    rx_volume is the realised count: gamma-Poisson noise around that baseline,
    multiplied by the treatment lift where -- and only where -- it applies.

    BAKED-IN EFFECT
    ---------------
        multiplier = 1.00                       months before the event month
                   = 1 + 0.15 * frac_remaining  the event month itself (ramp,
                                                so the lift literally starts on
                                                event_date; this month is a
                                                washout for analysis)
                   = 1.15                       every month after
    applied ONLY to (treatment HCP) x (their anchor event's target_ndc_category).
    All other categories for the same HCP, and every row for non-treatment HCPs,
    get trend and noise only.
    """
    step("STEP 4/5  Generating rx_claims_monthly.csv")
    rng = rng_for("rx_claims")

    months = month_range(RX_START_MONTH, RX_END_MONTH)
    month_idx = {m: i for i, m in enumerate(months)}
    n_months = len(months)

    ev_map = events.set_index("event_id")
    anchor_map = {}
    for _, r in anchors.iterrows():
        e = ev_map.loc[r["event_id"]]
        anchor_map[r["hcp_id"]] = (e["target_ndc_category"], e["event_date"])

    # Common seasonality (a shared shock -> differenced out by DiD).
    seasonal = 1.0 + 0.032 * np.sin(2 * np.pi * (np.arange(n_months) + 2.0) / 12.0)

    blocks = []
    hcp_indexed = hcp.set_index("hcp_id")

    for hcp_id, row in hcp_indexed.iterrows():
        primary = SPECIALTY_PRIMARY_NDC[row["specialty"]]
        cats = [primary, SHARED_NDC_CATEGORIES[0], SHARED_NDC_CATEGORIES[1]]
        shares = [row["_mix_primary"], row["_mix_shared0"], row["_mix_shared1"]]
        base_total = float(row["baseline_rx_volume_monthly"])
        trend = float(row["_trend"])
        growth = (1.0 + trend) ** np.arange(n_months)

        tgt_cat, tgt_date = anchor_map.get(hcp_id, (None, None))
        if tgt_cat is not None:
            ev_month = date_to_month(tgt_date)
            ev_pos = month_idx[ev_month]
            days_in_month = (
                (dt.date(tgt_date.year + (tgt_date.month == 12),
                         tgt_date.month % 12 + 1, 1) - dt.timedelta(days=1)).day)
            frac_remaining = (days_in_month - tgt_date.day + 1) / days_in_month

        for cat, share in zip(cats, shares):
            # Floor keeps every series comfortably away from zero, so post/pre
            # ratios stay well defined for every HCP.
            level = max(10.0 * RX_SCALE, base_total * float(share))
            drift = (1.0 + NDC_MARKET_DRIFT[cat]) ** np.arange(n_months)
            baseline = level * growth * drift * seasonal

            lift = np.ones(n_months)
            if tgt_cat is not None and cat == tgt_cat:
                lift[ev_pos + 1:] = 1.0 + TRUE_EFFECT_PCT / 100.0
                lift[ev_pos] = 1.0 + (TRUE_EFFECT_PCT / 100.0) * frac_remaining

            mu = baseline * lift
            # Gamma-Poisson: persistent practice-level dispersion + count noise.
            disp = rng.gamma(RX_DISPERSION_SHAPE, 1.0 / RX_DISPERSION_SHAPE,
                             size=n_months)
            volume = rng.poisson(np.maximum(mu * disp, 0.05))

            blocks.append(pd.DataFrame({
                "hcp_id": hcp_id,
                "month": months,
                "ndc_category": cat,
                "rx_volume": volume.astype(np.int64),
                "rx_trend_baseline": np.round(baseline, 2),
            }))

    rx = pd.concat(blocks, ignore_index=True)
    rx = rx.sort_values(["hcp_id", "ndc_category", "month"],
                        kind="stable").reset_index(drop=True)

    print(f"    rows: {len(rx):,d}   "
          f"({rx['hcp_id'].nunique():,d} HCPs x "
          f"{rx.groupby('hcp_id')['ndc_category'].nunique().max()} categories x "
          f"{n_months} months)")
    print("      note: the ~28,000-row target is a floor. Continuous history for "
          "every HCP x 3\n            categories across a 21-month event window "
          "plus 6 months of padding on\n            each side mathematically "
          "requires ~126k rows; the floor is exceeded 4.5x.")
    report_nulls(rx, "rx_claims_monthly")
    print(f"    month span: {months[0]} .. {months[-1]}  ({n_months} months)")
    print(f"      earliest event {events['event_date'].min()} -> "
          f"{month_idx[date_to_month(events['event_date'].min())]} months of "
          f"pre-history (>=6 required)")
    print(f"      latest   event {events['event_date'].max()} -> "
          f"{n_months - 1 - month_idx[date_to_month(events['event_date'].max())]}"
          f" months of post-history (>=6 required)")
    print(f"    distinct ndc_category values: {rx['ndc_category'].nunique()}")
    describe_numeric(rx, ["rx_volume", "rx_trend_baseline"])

    # Continuity check: every (hcp, category) series must be complete.
    per_series = rx.groupby(["hcp_id", "ndc_category"]).size()
    print(f"    panel continuity: every (hcp_id, ndc_category) series has "
          f"{per_series.min()}..{per_series.max()} months "
          f"({'COMPLETE' if per_series.min() == per_series.max() == n_months else 'GAPS'})")

    # Verify the lift actually landed.
    tgt_rows = []
    for hcp_id, (cat, d) in anchor_map.items():
        tgt_rows.append((hcp_id, cat, date_to_month(d)))
    tgt = pd.DataFrame(tgt_rows, columns=["hcp_id", "ndc_category", "ev_month"])
    chk = rx.merge(tgt, on=["hcp_id", "ndc_category"], how="inner")
    chk["rel"] = chk["month"].map(month_idx) - chk["ev_month"].map(month_idx)
    post = chk[(chk["rel"] >= 1) & (chk["rel"] <= DID_WINDOW_MONTHS)]
    pre = chk[(chk["rel"] <= -1) & (chk["rel"] >= -DID_WINDOW_MONTHS)]
    ratio_obs = post["rx_volume"].sum() / pre["rx_volume"].sum()
    ratio_cf = post["rx_trend_baseline"].sum() / pre["rx_trend_baseline"].sum()
    print(f"    baked-in effect check (treatment x target category, +/-6mo):")
    print(f"      observed post/pre = {ratio_obs:.4f}   counterfactual "
          f"post/pre = {ratio_cf:.4f}   implied lift = "
          f"{(ratio_obs / ratio_cf - 1) * 100:.2f}%  (target "
          f"{TRUE_EFFECT_PCT:.1f}%)")
    return rx


def build_ground_truth(hcp: pd.DataFrame, events: pd.DataFrame,
                       anchors: pd.DataFrame) -> dict:
    """Ground truth built ONLY from clean in-memory frames.

    Never re-derived from anything the injection step has touched -- this is
    protected field #4 and the scoring key for every matching method.
    """
    step("STEP 5/5  Building ground_truth_config.json (from CLEAN data)")

    ev = events.set_index("event_id")
    h = hcp.set_index("hcp_id")

    units = []
    for hcp_id in sorted(anchors["hcp_id"]):
        eid = anchors.loc[anchors["hcp_id"] == hcp_id, "event_id"].iloc[0]
        e = ev.loc[eid]
        units.append({
            "hcp_id": str(hcp_id),
            "event_id": str(eid),
            "event_date": e["event_date"].isoformat(),
            "target_ndc_category": str(e["target_ndc_category"]),
            "true_effect_pct": TRUE_EFFECT_PCT,
            "specialty": str(h.loc[hcp_id, "specialty"]),
            "region": str(h.loc[hcp_id, "region"]),
        })

    gt = {
        "metadata": {
            "seed": SEED,
            "generation_date": GENERATION_DATE.isoformat(),
            "true_effect_pct": TRUE_EFFECT_PCT,
            "effect_definition": (
                "Multiplicative +15% lift applied to rx_volume for the treatment "
                "HCP's anchor-event target_ndc_category only, starting on "
                "event_date. No lift on any other ndc_category, and none for "
                "non-treatment HCPs."
            ),
            "treatment_definition": (
                "PRIMARY-EVENT ANCHOR RULE: an HCP with role='attendee' is a "
                "treatment unit; their primary event is the event of their "
                "EARLIEST attendance_date, ties broken by lowest event_id. "
                "De-duplicate attendance on (hcp_id, event_id) first."
            ),
            "analysis_window_months": DID_WINDOW_MONTHS,
            "washout_rule": (
                "The event month itself is partially exposed (lift ramps from "
                "event_date) and should be excluded from both pre and post "
                "windows. Pre = months [-6,-1], post = months [+1,+6] relative "
                "to the event month."
            ),
            "control_pool_rule": (
                "Controls must exclude speakers. Filter out any hcp_id with "
                "speaker_eligible_flag=True OR any attendance row with "
                "role='speaker'; speaker_eligible=True HCPs who never spoke are "
                "still valid controls if they have zero attendance rows."
            ),
            "n_treatment_units": len(units),
            "rx_panel_months": [RX_START_MONTH, RX_END_MONTH],
            "event_window": [EVENT_WINDOW_START.isoformat(),
                             EVENT_WINDOW_END.isoformat()],
        },
        "treatment_units": units,
    }

    print(f"    treatment units: {len(units):,d}")
    print(f"    all true_effect_pct == {TRUE_EFFECT_PCT}: "
          f"{all(u['true_effect_pct'] == TRUE_EFFECT_PCT for u in units)}")
    print(f"    unique hcp_id in ground truth: "
          f"{len({u['hcp_id'] for u in units}):,d} (must equal unit count)")
    by_spec = pd.Series([u["specialty"] for u in units]).value_counts().sort_index()
    print("    treatment units by specialty:")
    for s, n in by_spec.items():
        print(f"      {s:<16s} {n:>5,d}")
    by_cat = pd.Series([u["target_ndc_category"] for u in units]).value_counts().sort_index()
    print("    treatment units by target_ndc_category:")
    for c, n in by_cat.items():
        print(f"      {c:<24s} {n:>5,d}")
    return gt


def canonical_json_bytes(obj: dict) -> bytes:
    """Byte-stable serialization used for both the file and the tamper check."""
    return (json.dumps(obj, indent=2, ensure_ascii=False,
                       sort_keys=False) + "\n").encode("utf-8")


# ==============================================================================
# SECTION 4 -- DATA-QUALITY INJECTION  (non-protected fields only)
# ==============================================================================

def inject_data_quality_issues(hcp: pd.DataFrame, events: pd.DataFrame,
                               attendance: pd.DataFrame, rx: pd.DataFrame,
                               anchors: pd.DataFrame):
    """Dirty the copies, log every change, never touch a protected field.

    Protected (see module docstring / spec):
      1. rx_volume for treatment HCPs x target category x months >= event month
      2. event_date                            3. target_ndc_category anywhere
      4. ground_truth_config.json              5. treatment assignment integrity
      6. speaker_eligible_flag / role='speaker'
      7. hcp_id primary keys
    """
    banner("DATA-QUALITY INJECTION  (post-generation, non-protected fields only)")
    log = InjectionLog()

    hcp_d = hcp.copy()
    events_d = events.copy()
    att_d = attendance.copy()
    rx_d = rx.copy()

    # ---- protected rx zone ---------------------------------------------------
    ev_map = events.set_index("event_id")
    protected_keys: set[tuple] = set()
    guard_keys: set[tuple] = set()   # protected + the 6-month pre window
    months = month_range(RX_START_MONTH, RX_END_MONTH)
    midx = {m: i for i, m in enumerate(months)}

    for _, a in anchors.iterrows():
        e = ev_map.loc[a["event_id"]]
        cat = e["target_ndc_category"]
        pos = midx[date_to_month(e["event_date"])]
        for i in range(pos, len(months)):                    # event month onward
            protected_keys.add((a["hcp_id"], months[i], cat))
        for i in range(max(0, pos - DID_WINDOW_MONTHS), len(months)):
            guard_keys.add((a["hcp_id"], months[i], cat))    # + pre window

    print(f"  protected rx cells (treatment x target cat x post): "
          f"{len(protected_keys):,d}")
    print(f"  guarded rx cells (protected + 6mo pre window, kept clean): "
          f"{len(guard_keys):,d}")

    hcp_d = _inject_hcp(hcp_d, log)
    _inject_events(events_d, log)
    att_d = _inject_attendance(att_d, anchors, log)
    rx_d = _inject_rx(rx_d, guard_keys, log)

    log_df = log.to_frame()
    print(f"\n  total logged modifications: {len(log_df):,d}")
    print("  by file / issue_type:")
    summary = (log_df.groupby(["target_file", "issue_type"])
               .size().reset_index(name="n"))
    for _, r in summary.iterrows():
        print(f"    {r['target_file']:<32s} {r['issue_type']:<28s} {r['n']:>6,d}")
    return hcp_d, events_d, att_d, rx_d, log_df


def _inject_hcp(hcp_d: pd.DataFrame, log: InjectionLog) -> pd.DataFrame:
    rng = rng_for("inject_hcp")
    n = len(hcp_d)
    print("\n  hcp.csv")

    # (1) ~3% missing sub_specialty
    idx = rng.choice(n, size=int(round(RATE_HCP_MISSING_SUBSPEC * n)), replace=False)
    for i in idx:
        log.add(FILE_HCP, "missing_sub_specialty",
                f"hcp_id={hcp_d.at[i, 'hcp_id']}", "sub_specialty",
                hcp_d.at[i, "sub_specialty"], None, "set to NULL")
    hcp_d.loc[idx, "sub_specialty"] = np.nan
    print(f"    missing sub_specialty        : {len(idx):>5,d}  "
          f"({len(idx) / n:.2%})")

    # (2) ~2% inconsistent region naming (canonical map kept in
    #     REGION_ALIAS_LOOKUP and printed to console)
    idx = rng.choice(n, size=int(round(RATE_HCP_REGION_VARIANT * n)), replace=False)
    for i in idx:
        orig = hcp_d.at[i, "region"]
        variant = str(rng.choice(REGION_VARIANTS[orig]))
        log.add(FILE_HCP, "region_naming_variant",
                f"hcp_id={hcp_d.at[i, 'hcp_id']}", "region", orig, variant,
                f"canonical='{orig}'; see REGION_ALIAS_LOOKUP")
        hcp_d.at[i, "region"] = variant
    print(f"    region naming variants       : {len(idx):>5,d}  "
          f"({len(idx) / n:.2%})")

    # (3) ~2% inconsistent npi_taxonomy_code formatting (dashes inserted)
    idx = rng.choice(n, size=int(round(RATE_HCP_TAXONOMY_FORMAT * n)), replace=False)
    for i in idx:
        orig = hcp_d.at[i, "npi_taxonomy_code"]
        variant = f"{orig[:4]}-{orig[4:9]}-{orig[9:]}"
        log.add(FILE_HCP, "taxonomy_code_format",
                f"hcp_id={hcp_d.at[i, 'hcp_id']}", "npi_taxonomy_code",
                orig, variant, "dashes inserted; strip non-alphanumerics to join")
        hcp_d.at[i, "npi_taxonomy_code"] = variant
    print(f"    taxonomy format variants     : {len(idx):>5,d}  "
          f"({len(idx) / n:.2%})")

    # (4) ~1% duplicate rows, same hcp_id, whitespace/case name variants.
    #     speaker_eligible_flag is copied verbatim, so the speaker filter stays
    #     unambiguous (protected field #6).
    idx = rng.choice(n, size=int(round(RATE_HCP_DUPLICATE_ROW * n)), replace=False)
    dupes = []
    for i in idx:
        row = hcp_d.loc[i].copy()
        style = int(rng.integers(0, 3))
        if style == 0:
            row["first_name"] = f"  {row['first_name'].upper()} "
            row["last_name"] = f" {row['last_name']}  "
        elif style == 1:
            row["first_name"] = row["first_name"].lower()
            row["last_name"] = row["last_name"].upper()
        else:
            row["first_name"] = f"{row['first_name']} "
            row["last_name"] = f"{row['last_name'].lower()}  "
        log.add(FILE_HCP, "duplicate_row_name_variant",
                f"hcp_id={row['hcp_id']}", "first_name|last_name",
                f"{hcp_d.at[i, 'first_name']}|{hcp_d.at[i, 'last_name']}",
                f"{row['first_name']}|{row['last_name']}",
                "extra row appended with same hcp_id; dedupe on hcp_id")
        dupes.append(row)
    if dupes:
        # Stable sort on hcp_id parks each duplicate next to its original, which
        # is what a master file merged from two source systems actually looks
        # like. speaker_eligible_flag is copied verbatim on both rows.
        hcp_d = pd.concat([hcp_d, pd.DataFrame(dupes)], ignore_index=True)
        hcp_d = hcp_d.sort_values("hcp_id", kind="stable").reset_index(drop=True)
    print(f"    duplicate rows appended      : {len(dupes):>5,d}  "
          f"({len(dupes) / n:.2%})   -> file now {len(hcp_d):,d} rows")
    return hcp_d


def _inject_events(events_d: pd.DataFrame, log: InjectionLog) -> None:
    """~2% program_spend missing or 0. event_date / target_ndc_category untouched."""
    rng = rng_for("inject_events")
    n = len(events_d)
    print("\n  events.csv")

    events_d["program_spend"] = events_d["program_spend"].astype(object)
    idx = rng.choice(n, size=int(round(RATE_EVT_SPEND_BAD * n)), replace=False)
    n_null = n_zero = 0
    for i in idx:
        orig = events_d.at[i, "program_spend"]
        if rng.random() < 0.5:
            events_d.at[i, "program_spend"] = np.nan
            n_null += 1
            log.add(FILE_EVENTS, "program_spend_missing",
                    f"event_id={events_d.at[i, 'event_id']}", "program_spend",
                    orig, None, "set to NULL")
        else:
            events_d.at[i, "program_spend"] = 0
            n_zero += 1
            log.add(FILE_EVENTS, "program_spend_zero",
                    f"event_id={events_d.at[i, 'event_id']}", "program_spend",
                    orig, 0, "set to 0 (invalid for ROI denominator)")
    print(f"    program_spend NULL           : {n_null:>5,d}")
    print(f"    program_spend 0              : {n_zero:>5,d}  "
          f"(combined {len(idx) / n:.2%})")


def _inject_attendance(att_d: pd.DataFrame, anchors: pd.DataFrame,
                       log: InjectionLog) -> pd.DataFrame:
    """Duplicates, missing engagement_score, small attendance_date drift.

    Both the duplicates and the date drift are engineered so that re-running the
    PRIMARY-EVENT ANCHOR RULE on the dirty file returns the identical anchor for
    every treatment HCP (protected field #5):
      * duplicates carry the SAME attendance_date as the row they copy;
      * drift only touches rows that are not an HCP's anchor and sit at least
        3 days after it, so a -2 day shift can never overtake the anchor.
    """
    rng = rng_for("inject_attendance")
    n = len(att_d)
    print("\n  attendance.csv")

    anchor_pairs = set(zip(anchors["hcp_id"], anchors["event_id"]))
    anchor_date = dict(zip(anchors["hcp_id"], anchors["anchor_date"]))

    # (1) ~4% duplicate attendance records (same hcp_id + event_id)
    k = int(round(RATE_ATT_DUPLICATE * n))
    idx = rng.choice(n, size=k, replace=False)
    dupes = []
    for j, i in enumerate(idx):
        row = att_d.loc[i].copy()
        row["attendance_id"] = f"ATT-DUP-{j + 1:05d}"
        # engagement_score may differ slightly (typical of merged source systems)
        if rng.random() < 0.4 and not pd.isna(row["engagement_score"]):
            row["engagement_score"] = round(float(np.clip(
                row["engagement_score"] + rng.normal(0, 0.3), 1.0, 10.0)), 1)
        log.add(FILE_ATTENDANCE, "duplicate_attendance_record",
                f"attendance_id={row['attendance_id']}", "<row>",
                f"source attendance_id={att_d.at[i, 'attendance_id']}",
                f"hcp_id={row['hcp_id']}|event_id={row['event_id']}",
                "same hcp_id+event_id, same attendance_date; "
                "dedupe on (hcp_id,event_id)")
        dupes.append(row)
    att_d = pd.concat([att_d, pd.DataFrame(dupes)], ignore_index=True)
    print(f"    duplicate records added      : {len(dupes):>5,d}  "
          f"({len(dupes) / n:.2%})   -> file now {len(att_d):,d} rows")

    # (2) ~2% missing engagement_score
    att_d["engagement_score"] = att_d["engagement_score"].astype(object)
    idx = rng.choice(len(att_d), size=int(round(RATE_ATT_MISSING_ENGAGEMENT * n)),
                     replace=False)
    for i in idx:
        log.add(FILE_ATTENDANCE, "missing_engagement_score",
                f"attendance_id={att_d.at[i, 'attendance_id']}",
                "engagement_score", att_d.at[i, "engagement_score"], None,
                "set to NULL")
        att_d.at[i, "engagement_score"] = np.nan
    print(f"    missing engagement_score     : {len(idx):>5,d}  "
          f"({len(idx) / n:.2%})")

    # (3) ~1% attendance_date drift, anchor-safe by construction
    eligible = []
    for i in range(len(att_d)):
        h, e = att_d.at[i, "hcp_id"], att_d.at[i, "event_id"]
        if att_d.at[i, "role"] == "speaker":
            eligible.append(i)                     # speakers are never treatment
            continue
        if (h, e) in anchor_pairs:
            continue                               # never drift the anchor row
        d = att_d.at[i, "attendance_date"]
        if h in anchor_date and (d - anchor_date[h]).days >= 3:
            eligible.append(i)
    eligible = np.array(eligible)
    idx = rng.choice(eligible, size=int(round(RATE_ATT_DATE_DRIFT * n)),
                     replace=False)
    for i in idx:
        orig = att_d.at[i, "attendance_date"]
        delta = int(rng.choice([-2, -1, 1, 2]))
        new = orig + dt.timedelta(days=delta)
        log.add(FILE_ATTENDANCE, "attendance_date_drift",
                f"attendance_id={att_d.at[i, 'attendance_id']}",
                "attendance_date", orig.isoformat(), new.isoformat(),
                f"{delta:+d} days vs event_date; non-anchor row, "
                "primary-event assignment unchanged")
        att_d.at[i, "attendance_date"] = new
    print(f"    attendance_date drift        : {len(idx):>5,d}  "
          f"({len(idx) / n:.2%})")

    att_d = att_d.sort_values(["attendance_date", "event_id", "role", "hcp_id"],
                              kind="stable").reset_index(drop=True)
    return att_d


def _inject_rx(rx_d: pd.DataFrame, guard_keys: set, log: InjectionLog) -> pd.DataFrame:
    """Nulls, comma-formatted strings, and genuine outliers -- all outside the
    guarded zone (treatment HCP x target category x [-6 months, end])."""
    rng = rng_for("inject_rx")
    n = len(rx_d)
    print("\n  rx_claims_monthly.csv")

    keys = list(zip(rx_d["hcp_id"], rx_d["month"], rx_d["ndc_category"]))
    is_guarded = np.array([k in guard_keys for k in keys])
    free_idx = np.flatnonzero(~is_guarded)
    print(f"    eligible (non-guarded) rows  : {len(free_idx):,d} of {n:,d}")

    rx_d["rx_volume"] = rx_d["rx_volume"].astype(object)
    rx_d["is_injected_outlier"] = False

    # A comma only appears once a value reaches four digits, so this injection
    # is drawn from the >=1000 tail -- otherwise f"{80:,d}" == "80" and the
    # "corruption" round-trips through the CSV as a perfectly good integer.
    # It is capped well short of exhausting that tail, so teams cannot simply
    # equate "large value" with "string value".
    vol = pd.to_numeric(rx_d["rx_volume"], errors="coerce").to_numpy()
    comma_pool = np.array([i for i in free_idx if vol[i] >= 1000])
    n_comma = int(min(round(RATE_RX_COMMA_STRING * n), 0.65 * len(comma_pool)))
    comma_idx = rng.choice(comma_pool, size=n_comma, replace=False)

    n_missing = int(round(RATE_RX_MISSING * n))
    rest = np.setdiff1d(free_idx, comma_idx, assume_unique=False)
    picks = rng.choice(rest, size=n_missing + N_RX_OUTLIERS, replace=False)
    miss_idx = picks[:n_missing]
    out_idx = picks[n_missing:]

    # (1) ~2% missing rx_volume
    for i in miss_idx:
        log.add(FILE_RX, "missing_rx_volume",
                f"hcp_id={rx_d.at[i, 'hcp_id']}|month={rx_d.at[i, 'month']}"
                f"|ndc_category={rx_d.at[i, 'ndc_category']}",
                "rx_volume", rx_d.at[i, "rx_volume"], None,
                "set to NULL; outside every treatment target-category window")
    rx_d.loc[miss_idx, "rx_volume"] = np.nan
    print(f"    missing rx_volume            : {len(miss_idx):>5,d}  "
          f"({len(miss_idx) / n:.2%})")

    # (2) ~1% comma-formatted string
    for i in comma_idx:
        orig = int(rx_d.at[i, "rx_volume"])
        new = f"{orig:,d}"
        log.add(FILE_RX, "rx_volume_comma_string",
                f"hcp_id={rx_d.at[i, 'hcp_id']}|month={rx_d.at[i, 'month']}"
                f"|ndc_category={rx_d.at[i, 'ndc_category']}",
                "rx_volume", orig, new,
                "stored as thousands-separated text; strip ',' then cast")
        rx_d.at[i, "rx_volume"] = new
    print(f"    comma-formatted rx_volume    : {len(comma_idx):>5,d}  "
          f"({len(comma_idx) / n:.2%})   drawn from {len(comma_pool):,d} rows "
          f">=1,000 (every one carries a real comma)")

    # (3) genuine outliers, 5-10x, flagged in a HIDDEN column.
    #     is_injected_outlier is internal validation only and is dropped before
    #     rx_claims_monthly.csv is written -- the injection log is the record.
    for i in out_idx:
        orig = int(rx_d.at[i, "rx_volume"])
        mult = float(rng.uniform(5.0, 10.0))
        new = int(round(orig * mult))
        log.add(FILE_RX, "outlier_rx_volume",
                f"hcp_id={rx_d.at[i, 'hcp_id']}|month={rx_d.at[i, 'month']}"
                f"|ndc_category={rx_d.at[i, 'ndc_category']}",
                "rx_volume", orig, new,
                f"{mult:.1f}x genuine outlier; flagged internally via "
                "is_injected_outlier (column not exported)")
        rx_d.at[i, "rx_volume"] = new
        rx_d.at[i, "is_injected_outlier"] = True
    print(f"    genuine outliers (5-10x)     : {len(out_idx):>5,d}  "
          f"[hidden flag column, not exported]")
    return rx_d


# ==============================================================================
# SECTION 5 -- PROTECTED-FIELD ASSERTIONS
# ==============================================================================

def run_protected_field_assertions(clean: dict, dirty: dict, anchors: pd.DataFrame,
                                   gt_clean: dict, gt_hash_before: str) -> bool:
    """(a)-(e) from the spec. Prints PASS/FAIL per check; returns overall pass."""
    banner("PROTECTED-FIELD ASSERTIONS")
    results: list[tuple[str, bool, str]] = []

    hcp_c, ev_c, att_c, rx_c = clean["hcp"], clean["events"], clean["attendance"], clean["rx"]
    hcp_d, ev_d, att_d, rx_d = dirty["hcp"], dirty["events"], dirty["attendance"], dirty["rx"]

    months = month_range(RX_START_MONTH, RX_END_MONTH)
    midx = {m: i for i, m in enumerate(months)}
    ev_map = ev_c.set_index("event_id")

    # ---- (a) treatment post-period target-category rx_volume unaltered -------
    prot = []
    for _, a in anchors.iterrows():
        e = ev_map.loc[a["event_id"]]
        pos = midx[date_to_month(e["event_date"])]
        for m in months[pos:]:
            prot.append((a["hcp_id"], m, e["target_ndc_category"]))
    prot_df = pd.DataFrame(prot, columns=["hcp_id", "month", "ndc_category"])

    c_slice = prot_df.merge(rx_c, on=["hcp_id", "month", "ndc_category"], how="left")
    d_slice = prot_df.merge(rx_d, on=["hcp_id", "month", "ndc_category"], how="left")
    n_missing_rows = int(c_slice["rx_volume"].isna().sum() + d_slice["rx_volume"].isna().sum())
    same_type = d_slice["rx_volume"].map(lambda v: isinstance(v, (int, np.integer))).all()
    values_equal = bool((pd.to_numeric(c_slice["rx_volume"], errors="coerce").fillna(-1)
                         == pd.to_numeric(d_slice["rx_volume"], errors="coerce").fillna(-2)).all())
    outlier_leak = int(d_slice.get("is_injected_outlier",
                                   pd.Series(False, index=d_slice.index)).sum())
    ok_a = values_equal and same_type and n_missing_rows == 0 and outlier_leak == 0
    results.append((
        "(a) treatment post-period target-category rx_volume is pristine",
        ok_a,
        f"{len(prot_df):,d} protected cells compared; values_equal={values_equal}, "
        f"all_still_int={same_type}, nulls={n_missing_rows}, outliers_leaked={outlier_leak}",
    ))

    # ---- (b) event_date and target_ndc_category unaltered --------------------
    dates_ok = bool((ev_c["event_date"].astype(str).to_numpy()
                     == ev_d["event_date"].astype(str).to_numpy()).all())
    cat_ok = bool((ev_c["target_ndc_category"].to_numpy()
                   == ev_d["target_ndc_category"].to_numpy()).all())
    ids_ok = bool((ev_c["event_id"].to_numpy() == ev_d["event_id"].to_numpy()).all())
    rx_cat_ok = bool(
        rx_c.groupby(["hcp_id", "month"])["ndc_category"].apply(
            lambda s: tuple(sorted(s))).equals(
        rx_d.groupby(["hcp_id", "month"])["ndc_category"].apply(
            lambda s: tuple(sorted(s))))
    )
    rx_month_ok = bool((rx_c["month"].to_numpy() == rx_d["month"].to_numpy()).all())
    ok_b = dates_ok and cat_ok and ids_ok and rx_cat_ok and rx_month_ok
    results.append((
        "(b) event_date + target_ndc_category unaltered (events & rx_claims)",
        ok_b,
        f"event_date={dates_ok}, events.target_ndc_category={cat_ok}, "
        f"event_id={ids_ok}, rx.ndc_category={rx_cat_ok}, rx.month={rx_month_ok}",
    ))

    # ---- (c) no duplicate / ambiguous treatment assignment -------------------
    anchors_dirty = compute_primary_event_anchors(att_d)
    same_units = set(anchors["hcp_id"]) == set(anchors_dirty["hcp_id"])
    merged = anchors.merge(anchors_dirty, on="hcp_id", suffixes=("_clean", "_dirty"))
    same_event = bool((merged["event_id_clean"] == merged["event_id_dirty"]).all())
    one_per_hcp = bool(anchors_dirty["hcp_id"].is_unique)
    speaker_ids = set(ev_d["speaker_hcp_id"])
    no_overlap = len(set(anchors_dirty["hcp_id"]) & speaker_ids) == 0
    dup_pairs = int(att_d.duplicated(subset=["hcp_id", "event_id"]).sum())
    dedup_pairs = int(att_d.drop_duplicates(subset=["hcp_id", "event_id"])
                      .duplicated(subset=["hcp_id", "event_id"]).sum())
    ok_c = same_units and same_event and one_per_hcp and no_overlap and dedup_pairs == 0
    results.append((
        "(c) treatment assignment integrity (one unambiguous anchor per HCP)",
        ok_c,
        f"{len(anchors_dirty):,d} units; identical_set={same_units}, "
        f"identical_anchor_event={same_event}, unique={one_per_hcp}, "
        f"speaker_overlap=0 -> {no_overlap}; "
        f"{dup_pairs:,d} duplicate rows collapse cleanly ({dedup_pairs} left)",
    ))

    # ---- (d) ground truth is an untouched clean copy -------------------------
    gt_hash_after = hashlib.sha256(canonical_json_bytes(gt_clean)).hexdigest()
    gt_path = OUTPUT_DIR / FILE_GROUND_TRUTH
    file_match = None
    if gt_path.exists():
        file_match = json.loads(gt_path.read_text(encoding="utf-8")) == gt_clean
    # Independent rebuild from the clean frames must reproduce it exactly.
    rebuilt = _rebuild_ground_truth_quiet(hcp_c, ev_c, anchors)
    rebuild_match = hashlib.sha256(canonical_json_bytes(rebuilt)).hexdigest() == gt_hash_after
    ok_d = (gt_hash_before == gt_hash_after) and rebuild_match
    results.append((
        "(d) ground_truth_config.json matches the clean generation exactly",
        ok_d,
        f"in-memory hash unchanged across injection={gt_hash_before == gt_hash_after}, "
        f"independent rebuild from clean frames matches={rebuild_match}"
        + (f", on-disk file matches={file_match}" if file_match is not None else
           ", on-disk file written after this check"),
    ))

    # ---- (e) speaker consistency across all three files ----------------------
    flag_by_id = hcp_d.groupby("hcp_id")["speaker_eligible_flag"]
    flags_consistent = bool(flag_by_id.nunique().max() == 1)
    spk = sorted(set(ev_d["speaker_hcp_id"]))
    flag_lookup = hcp_d.drop_duplicates("hcp_id").set_index("hcp_id")["speaker_eligible_flag"]
    all_present = all(s in flag_lookup.index for s in spk)
    all_flagged = all(bool(flag_lookup.get(s, False)) for s in spk)
    spk_rows = att_d[att_d["role"] == "speaker"]
    pairs_needed = set(zip(ev_d["speaker_hcp_id"], ev_d["event_id"]))
    pairs_have = set(zip(spk_rows["hcp_id"], spk_rows["event_id"]))
    all_have_row = pairs_needed.issubset(pairs_have)
    roles_clean = set(att_d["role"].unique()) == {"attendee", "speaker"}
    # The speaker-exclusion filter every matching method will run:
    excl = set(hcp_d.loc[hcp_d["speaker_eligible_flag"], "hcp_id"]) | \
        set(att_d.loc[att_d["role"] == "speaker", "hcp_id"])
    filter_catches_all = set(spk).issubset(excl)
    ok_e = (flags_consistent and all_present and all_flagged and all_have_row
            and roles_clean and filter_catches_all)
    results.append((
        "(e) speaker identifiability (events <-> hcp flag <-> role=speaker row)",
        ok_e,
        f"{len(spk):,d} distinct speakers; flag_consistent_per_hcp_id={flags_consistent}, "
        f"all_in_hcp={all_present}, all_flag_true={all_flagged}, "
        f"all_have_role_speaker_row={all_have_row}, role_vocab_clean={roles_clean}, "
        f"exclusion_filter_catches_all={filter_catches_all}",
    ))

    # ---- bonus: hcp_id primary keys preserved (protected field #7) ----------
    ids_clean, ids_dirty = set(hcp_c["hcp_id"]), set(hcp_d["hcp_id"])
    join_ok = (ids_clean == ids_dirty
               and set(rx_d["hcp_id"]).issubset(ids_dirty)
               and set(att_d["hcp_id"]).issubset(ids_dirty)
               and set(ev_d["speaker_hcp_id"]).issubset(ids_dirty)
               and hcp_d["hcp_id"].notna().all())
    results.append((
        "(+) hcp_id join keys intact across all files",
        join_ok,
        f"{len(ids_dirty):,d} distinct hcp_id (clean set identical={ids_clean == ids_dirty}); "
        f"every rx / attendance / events reference resolves",
    ))

    width = max(len(name) for name, _, _ in results)
    all_pass = True
    for name, ok, detail in results:
        all_pass &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}]  {name.ljust(width)}")
        print(f"          {detail}")
    print(f"\n  OVERALL: {'ALL ASSERTIONS PASSED' if all_pass else '*** FAILURES PRESENT ***'}")
    return bool(all_pass)


def _rebuild_ground_truth_quiet(hcp: pd.DataFrame, events: pd.DataFrame,
                                anchors: pd.DataFrame) -> dict:
    """build_ground_truth with the console output suppressed (used by check d)."""
    buf, sys.stdout = sys.stdout, open(os.devnull, "w", encoding="utf-8")
    try:
        return build_ground_truth(hcp, events, anchors)
    finally:
        sys.stdout.close()
        sys.stdout = buf


# ==============================================================================
# SECTION 6 -- NAIVE DiD SMOKE TEST (on CLEAN, pre-injection data)
# ==============================================================================

def run_naive_did_smoke_test(hcp: pd.DataFrame, events: pd.DataFrame,
                             attendance: pd.DataFrame, rx: pd.DataFrame,
                             anchors: pd.DataFrame) -> float:
    """Treatment vs. RANDOM controls, no matching, target-category filtered.

    This is the deliberate negative control: controls are drawn at random from
    the non-treatment, non-speaker pool and handed an index date and target
    category sampled from the treatment distribution. Because attendance was
    selected on a latent growth propensity, the control group's counterfactual
    trend is flatter -- so the naive estimate should land NOTICEABLY ABOVE the
    true 15%. That gap is the bias the four matching methods have to close.

    Windows: pre = event month -6..-1, post = +1..+6 (event month = washout).
    """
    banner("NAIVE DiD SMOKE TEST  (CLEAN pre-injection data, no matching)")
    rng = rng_for("smoke_test")

    months = month_range(RX_START_MONTH, RX_END_MONTH)
    midx = {m: i for i, m in enumerate(months)}
    ev_map = events.set_index("event_id")

    # ---- treatment frame -----------------------------------------------------
    t = anchors.copy()
    t["target_ndc_category"] = t["event_id"].map(ev_map["target_ndc_category"])
    t["ev_pos"] = t["event_id"].map(
        lambda e: midx[date_to_month(ev_map.loc[e, "event_date"])])
    t = t[["hcp_id", "target_ndc_category", "ev_pos"]]

    # ---- control frame: random draw from non-treatment, non-speaker HCPs -----
    # Each control is handed a pseudo index date and target category copied from
    # a randomly chosen treatment unit. The draw is restricted to treatment units
    # whose target category the control actually prescribes (their specialty's
    # anchored product plus the two portfolio-wide products) -- otherwise a
    # cardiologist assigned ONC_TARGETED_ORAL would silently drop out of the
    # comparison and quietly bias the control group. No covariate balancing of
    # any kind happens here: that is the whole point of the negative control.
    attended = set(attendance["hcp_id"])
    pool = hcp.loc[~hcp["hcp_id"].isin(attended),
                   ["hcp_id", "specialty"]].reset_index(drop=True)
    t_cat = t["target_ndc_category"].to_numpy()
    t_pos = t["ev_pos"].to_numpy()
    c_rows = []
    for specialty in SPECIALTIES:
        members = pool.loc[pool["specialty"] == specialty, "hcp_id"].to_numpy()
        if len(members) == 0:
            continue
        allowed = {SPECIALTY_PRIMARY_NDC[specialty], *SHARED_NDC_CATEGORIES}
        elig = np.flatnonzero(np.isin(t_cat, list(allowed)))
        draw = rng.choice(elig, size=len(members), replace=True)
        c_rows.append(pd.DataFrame({
            "hcp_id": members,
            "target_ndc_category": t_cat[draw],
            "ev_pos": t_pos[draw],
        }))
    c = pd.concat(c_rows, ignore_index=True).sort_values(
        "hcp_id", kind="stable").reset_index(drop=True)

    def group_means(frame: pd.DataFrame) -> tuple[float, float, pd.DataFrame]:
        m = rx.merge(frame.rename(columns={"target_ndc_category": "ndc_category"}),
                     on=["hcp_id", "ndc_category"], how="inner")
        m["rel"] = m["month"].map(midx) - m["ev_pos"]
        pre = m[(m["rel"] <= -1) & (m["rel"] >= -DID_WINDOW_MONTHS)]
        post = m[(m["rel"] >= 1) & (m["rel"] <= DID_WINDOW_MONTHS)]
        per = (pre.groupby("hcp_id")["rx_volume"].mean().rename("pre")
               .to_frame()
               .join(post.groupby("hcp_id")["rx_volume"].mean().rename("post"),
                     how="inner"))
        return float(pre["rx_volume"].mean()), float(post["rx_volume"].mean()), per

    t_pre, t_post, t_per = group_means(t)
    c_pre, c_post, c_per = group_means(c)

    ratio_t = t_post / t_pre
    ratio_c = c_post / c_pre
    naive_pct = (ratio_t / ratio_c - 1.0) * 100.0

    # Per-HCP ratio version (less sensitive to level differences).
    t_per_ratio = (t_per["post"] / t_per["pre"]).mean()
    c_per_ratio = (c_per["post"] / c_per["pre"]).mean()
    naive_pct_hcp = (t_per_ratio / c_per_ratio - 1.0) * 100.0

    # Additive DiD, for reference.
    additive = (t_post - t_pre) - (c_post - c_pre)

    print(f"  treatment units : {len(t_per):,d}   "
          f"control units (random, non-speaker): {len(c_per):,d}")
    print(f"  window          : pre = event month -{DID_WINDOW_MONTHS}..-1, "
          f"post = +1..+{DID_WINDOW_MONTHS}  (event month = washout)")
    print(f"  filtered to each unit's target_ndc_category only")
    print()
    print(f"  treatment  mean pre = {t_pre:8.2f}   mean post = {t_post:8.2f}   "
          f"ratio = {ratio_t:.4f}")
    print(f"  control    mean pre = {c_pre:8.2f}   mean post = {c_post:8.2f}   "
          f"ratio = {ratio_c:.4f}")
    print()
    print(f"  NAIVE DiD (ratio of ratios, group aggregate) : {naive_pct:6.2f}%")
    print(f"  NAIVE DiD (mean of per-HCP post/pre ratios)  : {naive_pct_hcp:6.2f}%")
    print(f"  NAIVE DiD (additive, Rx/month)               : {additive:6.2f}")
    print(f"  TRUE EFFECT                                  : {TRUE_EFFECT_PCT:6.2f}%")
    print(f"  CONFOUNDING BIAS                             : "
          f"{naive_pct - TRUE_EFFECT_PCT:+6.2f} pp")
    print()
    verdict = "PASS" if naive_pct > TRUE_EFFECT_PCT + 3.0 else "FAIL"
    print(f"  [{verdict}] naive estimate is "
          f"{'noticeably above' if verdict == 'PASS' else 'NOT sufficiently above'}"
          f" the true {TRUE_EFFECT_PCT:.0f}% -- "
          f"{'real structure exists for matching to correct' if verdict == 'PASS' else 'confounding too weak'}")
    return naive_pct


# ==============================================================================
# SECTION 7 -- WRITE + HASH
# ==============================================================================

def prepare_for_write(hcp_d, events_d, att_d, rx_d):
    """Drop internal-only columns and normalise types for a deterministic CSV."""
    hcp_out = hcp_d.drop(columns=[c for c in hcp_d.columns if c.startswith("_")])
    hcp_out = hcp_out.copy()
    hcp_out["speaker_eligible_flag"] = hcp_out["speaker_eligible_flag"].astype(bool)

    events_out = events_d.copy()
    events_out["event_date"] = events_out["event_date"].map(lambda d: d.isoformat())

    att_out = att_d.copy()
    att_out["attendance_date"] = att_out["attendance_date"].map(lambda d: d.isoformat())

    # is_injected_outlier is internal validation only -> never exported.
    rx_out = rx_d.drop(columns=["is_injected_outlier"], errors="ignore").copy()
    return hcp_out, events_out, att_out, rx_out


def write_all_outputs(hcp_out, events_out, att_out, rx_out, gt, log_df) -> None:
    banner("WRITING OUTPUT FILES")
    print(f"  output directory: {OUTPUT_DIR}")
    if os.name != "posix":
        print(f"  (spec path /mnt/user-data/outputs is POSIX-only; on "
              f"{platform.system()} this script writes to ./{LOCAL_OUTPUT_DIRNAME}"
              f" -- override with HACKATHON_OUTPUT_DIR)")

    write_csv(hcp_out, OUTPUT_DIR / FILE_HCP)
    write_csv(events_out, OUTPUT_DIR / FILE_EVENTS)
    write_csv(att_out, OUTPUT_DIR / FILE_ATTENDANCE)
    write_csv(rx_out, OUTPUT_DIR / FILE_RX)
    write_csv(log_df, OUTPUT_DIR / FILE_INJECTION_LOG)
    with open(OUTPUT_DIR / FILE_GROUND_TRUTH, "wb") as fh:
        fh.write(canonical_json_bytes(gt))

    frames = {FILE_HCP: hcp_out, FILE_EVENTS: events_out,
              FILE_ATTENDANCE: att_out, FILE_RX: rx_out,
              FILE_INJECTION_LOG: log_df}
    print()
    for name, df in frames.items():
        path = OUTPUT_DIR / name
        n_null = int(df.isna().sum().sum())
        print(f"  {name:<34s} {len(df):>8,d} rows x {df.shape[1]:>2d} cols   "
              f"{path.stat().st_size / 1e6:>7.2f} MB   nulls={n_null:,d}")
    gt_path = OUTPUT_DIR / FILE_GROUND_TRUTH
    print(f"  {FILE_GROUND_TRUTH:<34s} {len(gt['treatment_units']):>8,d} units"
          f"           {gt_path.stat().st_size / 1e6:>7.2f} MB")


def print_sha256_hashes() -> dict:
    banner("SHA-256 VERIFICATION HASHES")
    hashes = {}
    for name in OUTPUT_FILES:
        path = OUTPUT_DIR / name
        digest = sha256_of_file(path)
        hashes[name] = digest
        print(f"  {digest}  {name}")
    combined = hashlib.sha256(
        "".join(f"{n}:{hashes[n]}\n" for n in OUTPUT_FILES).encode("utf-8")
    ).hexdigest()
    print(f"\n  {combined}  <MANIFEST: sha256 of the 6 hashes above>")
    print("\n  Rerunning this script must reproduce every digest byte-for-byte.")
    return hashes


# ==============================================================================
# SECTION 8 -- MAIN
# ==============================================================================

def main() -> int:
    banner("SPEAKER PROGRAM & PEER-TO-PEER ROI  --  SYNTHETIC DATA GENERATION")
    print(f"  SEED                : {SEED}")
    print(f"  GENERATION_DATE     : {GENERATION_DATE.isoformat()}  (fixed constant)")
    print(f"  python / numpy      : {platform.python_version()} / {np.__version__}")
    print(f"  pandas              : {pd.__version__}")
    print(f"  output directory    : {OUTPUT_DIR}")
    print(f"  true effect         : +{TRUE_EFFECT_PCT:.0f}% on target_ndc_category only")

    # ---- 1. clean generation -------------------------------------------------
    banner("PHASE 1  --  CLEAN GENERATION")
    hcp = generate_hcp_master()
    events = generate_events(hcp)
    attendance, events, anchors = generate_attendance(hcp, events)
    rx = generate_rx_claims(hcp, events, anchors)
    gt_clean = build_ground_truth(hcp, events, anchors)
    gt_hash_before = hashlib.sha256(canonical_json_bytes(gt_clean)).hexdigest()
    print(f"    canonical ground-truth hash (pre-injection): {gt_hash_before[:32]}...")

    clean = {"hcp": hcp, "events": events, "attendance": attendance, "rx": rx}

    # ---- 2. inject messiness -------------------------------------------------
    hcp_d, events_d, att_d, rx_d, log_df = inject_data_quality_issues(
        hcp, events, attendance, rx, anchors)
    dirty = {"hcp": hcp_d, "events": events_d, "attendance": att_d, "rx": rx_d}

    print("\n  region alias -> canonical lookup table (the clean reference):")
    for alias, canon in REGION_ALIAS_LOOKUP.items():
        print(f"    {alias:<14s} -> {canon}")

    # ---- 3. protected-field assertions --------------------------------------
    all_pass = run_protected_field_assertions(
        clean, dirty, anchors, gt_clean, gt_hash_before)

    # ---- 4. write everything (ground truth comes from the CLEAN copy) --------
    hcp_out, events_out, att_out, rx_out = prepare_for_write(
        hcp_d, events_d, att_d, rx_d)
    write_all_outputs(hcp_out, events_out, att_out, rx_out, gt_clean, log_df)

    # Re-verify the on-disk ground truth against the in-memory clean object.
    on_disk = json.loads((OUTPUT_DIR / FILE_GROUND_TRUTH).read_text(encoding="utf-8"))
    gt_file_ok = on_disk == gt_clean
    print(f"\n  [{'PASS' if gt_file_ok else 'FAIL'}]  on-disk "
          f"ground_truth_config.json == clean in-memory object")
    all_pass &= gt_file_ok

    # ---- 5. naive DiD smoke test on CLEAN data ------------------------------
    naive = run_naive_did_smoke_test(hcp, events, attendance, rx, anchors)

    # ---- 6. hashes -----------------------------------------------------------
    print_sha256_hashes()

    banner("SUMMARY")
    print(f"  files written           : {len(OUTPUT_FILES)} -> {OUTPUT_DIR}")
    print(f"  hcp rows                : {len(hcp_out):,d}  "
          f"({hcp_out['hcp_id'].nunique():,d} unique hcp_id)")
    print(f"  events rows             : {len(events_out):,d}")
    print(f"  attendance rows         : {len(att_out):,d}")
    print(f"  rx_claims rows          : {len(rx_out):,d}")
    print(f"  injection log rows      : {len(log_df):,d}")
    print(f"  treatment units         : {len(anchors):,d}")
    print(f"  distinct speakers       : {events_out['speaker_hcp_id'].nunique():,d}")
    print(f"  never-attended controls : "
          f"{len(set(hcp['hcp_id']) - set(attendance['hcp_id'])):,d}")
    print(f"  protected assertions    : "
          f"{'ALL PASS' if all_pass else '*** FAILURE ***'}")
    print(f"  naive DiD (confounded)  : {naive:.2f}%  vs true "
          f"{TRUE_EFFECT_PCT:.0f}%  ({naive - TRUE_EFFECT_PCT:+.2f} pp of bias "
          f"for matching to remove)")
    print()
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
