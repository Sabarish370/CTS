#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 PIPELINE ORCHESTRATOR -- Speaker Program & Peer-to-Peer ROI Analysis
================================================================================

Runs the three stages in order via subprocess:

    1. preprocess   Preprocessing_tasks/preprocess_all.py
    2. match        4 matching scripts, SEQUENTIALLY
    3. did_roi      did_roi_engine.py  (processes all 4 methods in ONE run)

SCOPE
    generate_data.py is deliberately OUT of scope -- it is a one-time,
    seed-locked step and re-running it would invalidate every downstream
    artifact. generated_data/ground_truth_config.json is likewise never read
    or written here; the only interaction is a read-only existence check
    before the did_roi stage, because did_roi_engine.py depends on it.

WHY EXIT CODES ARE HANDLED PER-SCRIPT
    A read-only audit of the six scripts found their failure signalling is NOT
    uniform. Treating returncode==0 as "success" everywhere would let a
    silently-broken stage through. Each script therefore gets the handling its
    actual control flow justifies -- see EXIT_POLICY on each entry in
    SCRIPT_SPECS, and post_validate_random() for the one script whose
    returncode carries no information at all.

Run
---
    python pipeline.py --stage preprocess
    python pipeline.py --stage all
    python pipeline.py --stage match --methods nnm psm --force
================================================================================
"""

from __future__ import annotations

import argparse
import csv
import glob as globmod
import logging
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# ==============================================================================
# CONFIG -- paths verbatim. The misspellings ("neigbour", "randam",
# "techinques") are load-bearing: did_roi_engine.py and dashboard.py read those
# exact strings. Do not "fix" them here.
# ==============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
LOG_DIR = PROJECT_ROOT / "pipeline_logs"

PREPROCESSED_DIR = PROJECT_ROOT / "preprocessed_data"
MATCHED_PAIRS_DIR = PROJECT_ROOT / "matched_pairs"
DID_ROI_DIR = PROJECT_ROOT / "did_roi_output"
GROUND_TRUTH_FILE = PROJECT_ROOT / "generated_data" / "ground_truth_config.json"

# The rx output name contains a space and literal parens. Kept as an exact
# string AND matched via glob, because the parens make it easy to typo.
PREPROCESS_OUTPUTS = [
    PREPROCESSED_DIR / "hcp-final.csv",
    PREPROCESSED_DIR / "attendance-final.csv",
    PREPROCESSED_DIR / "events_preprocessed_final.csv",
    PREPROCESSED_DIR / "rx_claims_monthly_preprocessed (1).csv",
]
RX_OUTPUT_GLOB = "rx_claims_monthly_preprocessed (1).csv"

# The 12-column contract every matching output must satisfy in its FIRST 12
# positions, in this order. Extra trailing columns are expected and fine:
# nnm has 14, rbm 20, psm 15, random exactly 12.
STANDARD_COLUMNS = [
    "treatment_hcp_id", "control_hcp_id", "method", "event_id", "event_date",
    "target_ndc_category", "specialty", "region", "pre_period_rx_baseline",
    "control_pre_period_rx_baseline", "is_matched", "match_rank",
]

# Never add -O: it strips psm_matching.py's 7 asserts, which are that script's
# ONLY failure signal.
PYTHON = sys.executable

# Generous ceiling so a genuine hang eventually surfaces, while comfortably
# clearing random_matching's ~390s. Must stay well above 8 minutes.
SUBPROCESS_TIMEOUT_SEC = 1200

# random_matching.py reuses controls across events by design. This is a
# sanity ceiling for "implausibly often", not a correctness rule -- observed
# max reuse in practice is ~23.
MAX_PLAUSIBLE_CONTROL_REUSE = 50


@dataclass
class ScriptSpec:
    key: str
    label: str
    script: Path
    outputs: list[Path]
    expected_sec: int
    exit_policy: str          # see EXIT POLICY notes below
    note: str = ""


# EXIT POLICY values and what they mean:
#   "reliable"      returncode fully reflects validation outcome. Trust it.
#   "crash_only"    raises on hard failures, but soft check failures only print
#                   text. Scan stdout for FAIL/IMPLAUSIBLE -> WARNING.
#   "assert_based"  validation via assert statements; reliable unless -O.
#   "unreliable"    returncode carries NO information. Must post-validate.
MATCHING_SPECS: dict[str, ScriptSpec] = {
    "nnm": ScriptSpec(
        key="nnm", label="Nearest Neighbor",
        script=PROJECT_ROOT / "matching_techinques" / "nnm_matching.py",
        outputs=[MATCHED_PAIRS_DIR / "nearest_neigbour_matching" / "NNM_matched_pairs.csv"],
        expected_sec=75, exit_policy="crash_only",
        note="raises on hard failures; prints PASS/FAIL for protected-field checks "
             "without affecting exit code"),
    "rule_based": ScriptSpec(
        key="rule_based", label="Rule-Based",
        script=PROJECT_ROOT / "matching_techinques" / "rbm_matching.py",
        outputs=[MATCHED_PAIRS_DIR / "rule_based_matching" / "rule_based_matching_output.csv"],
        expected_sec=75, exit_policy="reliable",
        note="run_self_validation() feeds directly into return 1 / return 0"),
    "psm": ScriptSpec(
        key="psm", label="Propensity Score",
        script=PROJECT_ROOT / "matching_techinques" / "psm_matching.py",
        outputs=[MATCHED_PAIRS_DIR / "propensity_score_matching" / "PSM_matched_pairs.csv"],
        expected_sec=25, exit_policy="assert_based",
        note="7 asserts in validate(); .xlsx sibling is optional -- its write is "
             "wrapped in try/except and prints SKIPPED without failing"),
    "random": ScriptSpec(
        key="random", label="Random (placebo)",
        script=PROJECT_ROOT / "matching_techinques" / "random_matching.py",
        outputs=[MATCHED_PAIRS_DIR / "randam_matching" / "matching_output_random.csv"],
        expected_sec=390, exit_policy="unreliable",
        note="bare script, no __main__ guard, zero raises/asserts -- validation "
             "only PRINTS. returncode is meaningless; post_validate_random() is "
             "the real gate"),
}
METHOD_ORDER = ["nnm", "rule_based", "psm", "random"]   # random last: longest

PREPROCESS_SPEC = ScriptSpec(
    key="preprocess", label="Preprocessing",
    script=PROJECT_ROOT / "Preprocessing_tasks" / "preprocess_all.py",
    outputs=PREPROCESS_OUTPUTS, expected_sec=45, exit_policy="crash_only",
    note="raises FileNotFoundError only if raw inputs are missing; a 0 exit "
         "proves it ran, not that outputs are logically correct")

DID_ROI_SPEC = ScriptSpec(
    key="did_roi", label="DiD / ROI",
    script=PROJECT_ROOT / "did_roi_engine.py",
    outputs=[DID_ROI_DIR / f"did_roi_{kind}_{m}.csv"
             for kind in ("results", "summary") for m in METHOD_ORDER],
    expected_sec=50, exit_policy="crash_only",
    note="processes all 4 methods internally in ONE run; prints [IMPLAUSIBLE] "
         "sanity lines without affecting exit code")


@dataclass
class StageResult:
    name: str
    ok: bool
    duration_sec: float = 0.0
    returncode: int | None = None
    message: str = ""
    warnings: list[str] = field(default_factory=list)
    match_rate: str | None = None
    skipped: bool = False


log = logging.getLogger("pipeline")


# ==============================================================================
# LOGGING
# ==============================================================================

def setup_logging(timestamp: str) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"pipeline_run_{timestamp}.log"
    log.setLevel(logging.DEBUG)
    log.handlers.clear()

    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    ch.setLevel(logging.INFO)
    log.addHandler(fh)
    log.addHandler(ch)
    return log_path


def banner(text: str) -> None:
    log.info("=" * 74)
    log.info(text)
    log.info("=" * 74)


# ==============================================================================
# SUBPROCESS RUNNER
# ==============================================================================

def run_script(spec: ScriptSpec) -> tuple[int, str, float]:
    """Invoke a script with zero CLI args. cwd is irrelevant -- every script
    self-locates via __file__ / find_project_root(), so we do not set it.
    Never adds -O (that would strip psm_matching.py's asserts).
    """
    if not spec.script.exists():
        raise FileNotFoundError(f"script not found: {spec.script}")

    log.info(f"  -> {PYTHON} {spec.script.relative_to(PROJECT_ROOT)}")
    log.info(f"     expected runtime ~{spec.expected_sec}s  "
             f"(exit policy: {spec.exit_policy})")
    if spec.expected_sec >= 300:
        log.info(f"     NOTE: this is the longest stage (~{spec.expected_sec // 60}m"
                 f"{spec.expected_sec % 60:02d}s). It is NOT hung -- please wait.")

    t0 = time.perf_counter()
    proc = subprocess.run(
        [PYTHON, str(spec.script)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=SUBPROCESS_TIMEOUT_SEC)
    elapsed = time.perf_counter() - t0

    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    log.debug(f"----- stdout/stderr for {spec.script.name} -----\n{combined}")
    log.info(f"     finished in {elapsed:.1f}s  (returncode={proc.returncode})")
    if elapsed > spec.expected_sec * 3 and spec.expected_sec > 0:
        log.warning(f"     {spec.label} took {elapsed:.0f}s vs ~{spec.expected_sec}s "
                    f"expected -- unusually slow")
    return proc.returncode, combined, elapsed


def scan_stdout_for_soft_failures(spec: ScriptSpec, output: str) -> list[str]:
    """For 'crash_only' scripts: a 0 exit does not prove the internal checks
    passed. Surface FAIL / IMPLAUSIBLE lines as WARNINGs, never as failures --
    the spec is explicit that these do not fail the pipeline.
    """
    warnings: list[str] = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if "FAIL" in stripped or "IMPLAUSIBLE" in stripped:
            warnings.append(f"[{spec.key}] {stripped[:180]}")
    for w in warnings:
        log.warning(f"     soft-check warning: {w}")
    return warnings


MATCH_RATE_PATTERNS = [
    re.compile(r"match\s*rate\s*:?\s*([\d.]+)\s*%", re.I),          # psm, random
    re.compile(r"matched\s*:\s*[\d,]+\s*\(\s*([\d.]+)\s*%\s*\)"),   # nnm, rbm
]


def parse_match_rate(output: str) -> str | None:
    for pat in MATCH_RATE_PATTERNS:
        m = pat.search(output)
        if m:
            return f"{float(m.group(1)):.2f}%"
    return None


# ==============================================================================
# OUTPUT VERIFICATION
# ==============================================================================

def verify_preprocess_outputs() -> tuple[bool, str]:
    """All 4 preprocessing outputs must exist before ANY matching starts --
    the matching scripts read these files, so a partial preprocessing run would
    silently feed them stale or missing inputs.

    The rx file is located by glob as well as by exact path, because its name
    carries a space and literal parentheses that are easy to mistype.
    """
    missing = []
    for path in PREPROCESS_OUTPUTS:
        if path.name == RX_OUTPUT_GLOB:
            hits = globmod.glob(str(PREPROCESSED_DIR / RX_OUTPUT_GLOB))
            if not hits:
                missing.append(path.name)
            else:
                log.info(f"     rx output located via glob: {Path(hits[0]).name} "
                         f"({Path(hits[0]).stat().st_size:,} B)")
        elif not path.exists():
            missing.append(path.name)
        else:
            log.info(f"     found {path.name} ({path.stat().st_size:,} B)")
    if missing:
        return False, f"missing preprocessing outputs: {missing}"
    return True, "all 4 preprocessing outputs present"


def validate_schema(path: Path, key: str) -> tuple[bool, str]:
    """First 12 header fields must equal STANDARD_COLUMNS in order.
    Extra trailing columns are expected and allowed."""
    if not path.exists():
        return False, f"{path.name} does not exist"
    with open(path, "r", encoding="utf-8", newline="") as fh:
        header = next(csv.reader(fh), [])
    if len(header) < 12:
        return False, f"{path.name} has only {len(header)} columns, need >=12"
    first12 = header[:12]
    if first12 != STANDARD_COLUMNS:
        diffs = [f"pos {i}: got {g!r} want {w!r}"
                 for i, (g, w) in enumerate(zip(first12, STANDARD_COLUMNS)) if g != w]
        return False, f"{path.name} schema mismatch -> {'; '.join(diffs)}"
    return True, (f"{path.name}: first 12 columns match contract "
                  f"({len(header)} total, {len(header) - 12} extra)")


def post_validate_random(path: Path) -> tuple[bool, list[str]]:
    """The real gate for random_matching.py.

    That script has no __main__ guard, no raises and no asserts -- its
    validation section only PRINTS. It will exit 0 having produced garbage.
    These checks are therefore the only thing standing between a broken
    Random arm and the DiD stage, and a failure here IS a stage failure.

    Uses the csv module rather than pandas: fewer moving parts, and it reads
    exactly what is on disk without dtype inference smoothing over problems.
    """
    problems: list[str] = []
    if not path.exists():
        return False, [f"output missing: {path}"]

    n_rows = 0
    n_matched = 0
    n_self_match = 0
    control_counts: dict[str, int] = {}

    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            n_rows += 1
            t = (row.get("treatment_hcp_id") or "").strip()
            c = (row.get("control_hcp_id") or "").strip()
            matched = (row.get("is_matched") or "").strip().lower() in ("true", "1", "yes")
            if matched:
                n_matched += 1
                if c:
                    control_counts[c] = control_counts.get(c, 0) + 1
            if t and c and t == c:
                n_self_match += 1

    log.info(f"     rows={n_rows:,}  matched={n_matched:,}  "
             f"self_matches={n_self_match}  distinct_controls={len(control_counts):,}")

    if n_rows == 0:
        problems.append("row count is 0")
    if n_matched == 0:
        problems.append("no rows have is_matched=True")
    if n_self_match > 0:
        problems.append(f"{n_self_match} self-matched rows "
                        f"(treatment_hcp_id == control_hcp_id)")
    if control_counts:
        worst_id, worst_n = max(control_counts.items(), key=lambda kv: kv[1])
        log.info(f"     max control reuse: {worst_n} rows (control_hcp_id={worst_id})")
        if worst_n > MAX_PLAUSIBLE_CONTROL_REUSE:
            problems.append(
                f"control_hcp_id {worst_id} reused {worst_n} times "
                f"(> {MAX_PLAUSIBLE_CONTROL_REUSE} threshold) -- implausible")

    return (not problems), problems


def outputs_present(spec: ScriptSpec) -> bool:
    return all(p.exists() for p in spec.outputs)


# ==============================================================================
# STAGES
# ==============================================================================

def stage_preprocess(force: bool) -> StageResult:
    banner("STAGE 1/3 -- PREPROCESSING")
    spec = PREPROCESS_SPEC

    if not force and outputs_present(spec):
        log.info("  all 4 outputs already present; skipping (use --force to rerun)")
        ok, msg = verify_preprocess_outputs()
        return StageResult("preprocess", ok, message=msg, skipped=True)

    try:
        rc, out, elapsed = run_script(spec)
    except subprocess.TimeoutExpired:
        return StageResult("preprocess", False,
                           message=f"timed out after {SUBPROCESS_TIMEOUT_SEC}s")
    except FileNotFoundError as exc:
        return StageResult("preprocess", False, message=str(exc))

    if rc != 0:
        tail = "\n".join(out.strip().splitlines()[-12:])
        log.error(f"  preprocessing FAILED (returncode={rc})")
        log.error(f"  last lines:\n{tail}")
        return StageResult("preprocess", False, elapsed, rc,
                           f"returncode {rc} -- see log for traceback")

    warnings = scan_stdout_for_soft_failures(spec, out)

    # returncode 0 proves it ran, not that the outputs are correct -- verify.
    log.info("  verifying the 4 expected output files exist...")
    ok, msg = verify_preprocess_outputs()
    if not ok:
        log.error(f"  {msg}")
        return StageResult("preprocess", False, elapsed, rc, msg, warnings)

    log.info(f"  PASS -- {msg}")
    return StageResult("preprocess", True, elapsed, rc, msg, warnings)


def stage_match(methods: list[str], force: bool) -> list[StageResult]:
    banner("STAGE 2/3 -- MATCHING (sequential, 4 methods)")

    # Hard ordering gate: matching reads the files preprocessing writes.
    ok, msg = verify_preprocess_outputs()
    if not ok:
        log.error(f"  cannot start matching -- {msg}")
        log.error("  run: python pipeline.py --stage preprocess")
        return [StageResult("match:precondition", False, message=msg)]
    log.info(f"  precondition OK -- {msg}")

    results: list[StageResult] = []
    for key in [m for m in METHOD_ORDER if m in methods]:
        spec = MATCHING_SPECS[key]
        log.info("")
        log.info(f"--- {spec.label} ({key}) ---")
        log.info(f"    policy: {spec.note}")

        if not force and outputs_present(spec):
            log.info("    output already present; skipping (use --force to rerun)")
            results.append(StageResult(f"match:{key}", True, message="skipped",
                                       skipped=True))
            continue

        try:
            rc, out, elapsed = run_script(spec)
        except subprocess.TimeoutExpired:
            results.append(StageResult(f"match:{key}", False,
                                       message=f"timed out after {SUBPROCESS_TIMEOUT_SEC}s"))
            continue
        except FileNotFoundError as exc:
            results.append(StageResult(f"match:{key}", False, message=str(exc)))
            continue

        warnings: list[str] = []
        rate = parse_match_rate(out)
        if rate:
            log.info(f"     parsed match rate: {rate}")

        # --- per-script exit handling ------------------------------------
        if spec.exit_policy == "reliable":
            if rc != 0:
                log.error(f"    {spec.label} FAILED -- returncode {rc} is "
                          f"authoritative for this script")
                results.append(StageResult(f"match:{key}", False, elapsed, rc,
                                           f"returncode {rc}", warnings, rate))
                continue

        elif spec.exit_policy in ("crash_only", "assert_based"):
            if rc != 0:
                tail = "\n".join(out.strip().splitlines()[-12:])
                log.error(f"    {spec.label} FAILED (returncode={rc})\n{tail}")
                results.append(StageResult(f"match:{key}", False, elapsed, rc,
                                           f"returncode {rc}", warnings, rate))
                continue
            if spec.exit_policy == "crash_only":
                warnings += scan_stdout_for_soft_failures(spec, out)

        elif spec.exit_policy == "unreliable":
            # returncode tells us nothing here -- validate the artifact.
            log.info("    returncode is NOT a reliability signal for this script; "
                     "running post_validate_random()")
            good, problems = post_validate_random(spec.outputs[0])
            if not good:
                for p in problems:
                    log.error(f"    post-validation FAILED: {p}")
                results.append(StageResult(
                    f"match:{key}", False, elapsed, rc,
                    "post_validate_random failed: " + "; ".join(problems),
                    warnings, rate))
                continue
            log.info("    post_validate_random() PASSED")

        # --- output must exist (psm's .xlsx is optional, .csv is not) -----
        missing = [p.name for p in spec.outputs if not p.exists()]
        if missing:
            log.error(f"    expected output missing: {missing}")
            results.append(StageResult(f"match:{key}", False, elapsed, rc,
                                       f"missing output {missing}", warnings, rate))
            continue
        if key == "psm":
            xlsx = spec.outputs[0].with_suffix(".xlsx")
            log.info(f"    optional .xlsx sibling: "
                     f"{'present' if xlsx.exists() else 'absent (not an error)'}")

        log.info(f"    PASS -- {spec.outputs[0].name} written")
        results.append(StageResult(f"match:{key}", True, elapsed, rc, "ok",
                                   warnings, rate))
    return results


def stage_validate(methods: list[str]) -> StageResult:
    banner("STAGE 2.5/3 -- SCHEMA VALIDATION (12-column contract)")
    problems: list[str] = []
    warnings: list[str] = []

    for key in [m for m in METHOD_ORDER if m in methods]:
        spec = MATCHING_SPECS[key]
        ok, msg = validate_schema(spec.outputs[0], key)
        (log.info if ok else log.error)(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
        if not ok:
            problems.append(msg)

    # Random gets its content re-checked here too, since --stage validate may
    # be run standalone without having just executed the matching stage.
    if "random" in methods:
        log.info("  re-running post_validate_random() on the existing artifact...")
        good, probs = post_validate_random(MATCHING_SPECS["random"].outputs[0])
        if good:
            log.info("  [PASS] random output content checks")
        else:
            for p in probs:
                log.error(f"  [FAIL] random content: {p}")
            problems.extend(probs)

    if problems:
        return StageResult("validate", False, message="; ".join(problems),
                           warnings=warnings)
    return StageResult("validate", True, message="all schemas match the "
                       "12-column contract", warnings=warnings)


def stage_did_roi(force: bool) -> StageResult:
    banner("STAGE 3/3 -- DiD / ROI")
    spec = DID_ROI_SPEC

    # Direct dependency -- fail clearly here rather than letting the subprocess
    # crash with a cryptic traceback. Read-only existence check ONLY.
    if not GROUND_TRUTH_FILE.exists():
        msg = (f"required dependency missing: "
               f"{GROUND_TRUTH_FILE.relative_to(PROJECT_ROOT)}. "
               f"did_roi_engine.py reads this file for the ground-truth "
               f"comparison. It is produced by generate_data.py, which this "
               f"pipeline deliberately does not run.")
        log.error(f"  {msg}")
        return StageResult("did_roi", False, message=msg)
    log.info(f"  dependency present: "
             f"{GROUND_TRUTH_FILE.relative_to(PROJECT_ROOT)} "
             f"({GROUND_TRUTH_FILE.stat().st_size:,} B, read-only check)")

    # All 4 matching outputs must exist -- the engine reads a hardcoded
    # METHODS dict and will raise if any is absent.
    missing = [s.outputs[0].name for s in MATCHING_SPECS.values()
               if not s.outputs[0].exists()]
    if missing:
        msg = f"matching outputs missing, cannot run DiD/ROI: {missing}"
        log.error(f"  {msg}")
        return StageResult("did_roi", False, message=msg)

    if not force and outputs_present(spec):
        log.info("  did_roi outputs already present; skipping (use --force)")
        return StageResult("did_roi", True, message="skipped", skipped=True)

    log.info("  invoking did_roi_engine.py ONCE -- it loops all 4 methods "
             "internally (hardcoded METHODS dict, no per-method CLI)")
    try:
        rc, out, elapsed = run_script(spec)
    except subprocess.TimeoutExpired:
        return StageResult("did_roi", False,
                           message=f"timed out after {SUBPROCESS_TIMEOUT_SEC}s")
    except FileNotFoundError as exc:
        return StageResult("did_roi", False, message=str(exc))

    if rc != 0:
        tail = "\n".join(out.strip().splitlines()[-15:])
        log.error(f"  DiD/ROI FAILED (returncode={rc})\n{tail}")
        return StageResult("did_roi", False, elapsed, rc, f"returncode {rc}")

    warnings = scan_stdout_for_soft_failures(spec, out)

    missing_out = [p.name for p in spec.outputs if not p.exists()]
    if missing_out:
        msg = f"expected DiD/ROI outputs missing: {missing_out}"
        log.error(f"  {msg}")
        return StageResult("did_roi", False, elapsed, rc, msg, warnings)

    log.info(f"  PASS -- {len(spec.outputs)} output files written")
    return StageResult("did_roi", True, elapsed, rc,
                       f"{len(spec.outputs)} files written", warnings)


# ==============================================================================
# REPORT
# ==============================================================================

def write_report(timestamp: str, stage_arg: str, methods: list[str],
                 results: list[StageResult], log_path: Path,
                 total_sec: float) -> Path:
    path = LOG_DIR / f"pipeline_report_{timestamp}.md"
    overall = all(r.ok for r in results) if results else False
    all_warnings = [w for r in results for w in r.warnings]

    lines = [
        f"# Pipeline Run Report — {timestamp}",
        "",
        f"- **Overall:** {'PASS' if overall else 'FAIL'}",
        f"- **Stage requested:** `{stage_arg}`",
        f"- **Methods:** {', '.join(methods)}",
        f"- **Total runtime:** {total_sec:.1f}s",
        f"- **Log:** `{log_path.name}`",
        "",
        "## Stage results",
        "",
        "| stage | result | duration | returncode | detail |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        status = "SKIPPED" if r.skipped else ("PASS" if r.ok else "**FAIL**")
        dur = "—" if r.skipped else f"{r.duration_sec:.1f}s"
        rc = "—" if r.returncode is None else str(r.returncode)
        lines.append(f"| `{r.name}` | {status} | {dur} | {rc} | {r.message or '—'} |")

    lines += ["", "## Match rates (parsed from stdout)", ""]
    rates = [r for r in results if r.match_rate]
    if rates:
        lines += ["| method | match rate |", "|---|---|"]
        lines += [f"| `{r.name.replace('match:', '')}` | {r.match_rate} |" for r in rates]
    else:
        lines.append("_No match rates parsed (matching stage not run, or skipped)._")

    lines += ["", "## Warnings", ""]
    if all_warnings:
        lines.append("These come from scanning stdout for `FAIL` / `IMPLAUSIBLE` on "
                     "scripts whose exit code does not reflect soft-check outcomes. "
                     "They do **not** fail the pipeline.")
        lines.append("")
        lines += [f"- `{w}`" for w in all_warnings]
    else:
        lines.append("_None._")

    lines += [
        "", "## Notes", "",
        "- `generate_data.py` is out of scope (one-time, seed-locked).",
        "- `ground_truth_config.json` is only checked for existence, never read "
        "or written by this pipeline.",
        "- Stages run strictly sequentially; the 4 matching scripts are not "
        "parallelised, by design.",
        "- `random_matching.py`'s exit code is not a reliability signal — its "
        "artifact is validated by `post_validate_random()` instead.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Orchestrate preprocessing -> matching -> DiD/ROI.")
    parser.add_argument("--stage", default="all",
                        choices=["preprocess", "match", "validate", "did_roi", "all"])
    parser.add_argument("--methods", nargs="+", default=METHOD_ORDER,
                        choices=METHOD_ORDER,
                        help="matching methods to run (default: all 4)")
    parser.add_argument("--force", action="store_true",
                        help="rerun stages even if their outputs already exist")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = setup_logging(timestamp)

    banner("SPEAKER PROGRAM ROI -- PIPELINE")
    log.info(f"  project root : {PROJECT_ROOT}")
    log.info(f"  interpreter  : {PYTHON}")
    log.info(f"  stage        : {args.stage}")
    log.info(f"  methods      : {', '.join(args.methods)}")
    log.info(f"  force        : {args.force}")
    log.info(f"  log file     : {log_path}")
    log.info(f"  timeout      : {SUBPROCESS_TIMEOUT_SEC}s per subprocess "
             f"(must exceed random_matching's ~390s)")

    t0 = time.perf_counter()
    results: list[StageResult] = []

    if args.stage in ("preprocess", "all"):
        results.append(stage_preprocess(args.force))
        if not results[-1].ok and args.stage == "all":
            log.error("Preprocessing failed -- halting before matching, because "
                      "matching reads the files preprocessing writes.")
            total = time.perf_counter() - t0
            rp = write_report(timestamp, args.stage, args.methods, results,
                              log_path, total)
            log.info(f"report: {rp}")
            return 1

    if args.stage in ("match", "all"):
        results.extend(stage_match(args.methods, args.force))
        if any(not r.ok for r in results) and args.stage == "all":
            log.error("A matching method failed -- halting before DiD/ROI.")
            total = time.perf_counter() - t0
            rp = write_report(timestamp, args.stage, args.methods, results,
                              log_path, total)
            log.info(f"report: {rp}")
            return 1

    if args.stage in ("validate", "all"):
        results.append(stage_validate(args.methods))
        if not results[-1].ok and args.stage == "all":
            log.error("Schema validation failed -- halting before DiD/ROI, "
                      "because did_roi_engine.py requires the 12-column contract.")
            total = time.perf_counter() - t0
            rp = write_report(timestamp, args.stage, args.methods, results,
                              log_path, total)
            log.info(f"report: {rp}")
            return 1

    if args.stage in ("did_roi", "all"):
        results.append(stage_did_roi(args.force))

    total = time.perf_counter() - t0
    overall = all(r.ok for r in results) if results else False

    banner("PIPELINE SUMMARY")
    for r in results:
        status = "SKIPPED" if r.skipped else ("PASS" if r.ok else "FAIL")
        dur = "" if r.skipped else f"  {r.duration_sec:6.1f}s"
        log.info(f"  {status:<8s} {r.name:<22s}{dur}  {r.message}")
    n_warn = sum(len(r.warnings) for r in results)
    log.info(f"  total runtime: {total:.1f}s   warnings: {n_warn}")
    log.info(f"  OVERALL: {'PASS' if overall else 'FAIL'}")

    report_path = write_report(timestamp, args.stage, args.methods, results,
                               log_path, total)
    log.info(f"  log    : {log_path}")
    log.info(f"  report : {report_path}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
