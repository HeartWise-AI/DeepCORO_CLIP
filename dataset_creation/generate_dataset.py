#!/usr/bin/env python3
"""
Dataset Generation Script

This script consolidates the dataset generation logic from the notebook,
including merging predictions with stent metadata, applying filters,
generating reports, and sampling data.

Usage:
    python generate_dataset.py --config path/to/config.yaml
    python generate_dataset.py --input-csv path/to/input.csv --output-dir path/to/output
"""

import sys
import os
import re
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from tqdm import tqdm
import yaml
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

tqdm.pandas()

# ──────────────────────────────────────────────────────────────────────────
# Constants and Mappings
# ──────────────────────────────────────────────────────────────────────────

MAIN_STRUCTURE_MAP = {
    0: "Left Coronary",
    1: "Right Coronary", 
    2: "Other", 3: "Graft", 4: "Catheter", 5: "Femoral",
    6: "LV", 7: "TAVR", 8: "Aorta", 9: "Radial",
    10: "TEE probe", 11: "Pigtail"
}

DOMINANCE_MAP = {
    0: "right_dominant", 
    1: "left_dominant"
}

LABELS_TO_VESSEL_NAMES = {
    "left_main_stenosis": "the Left Main Coronary Artery (LMCA)",
    "prox_lad_stenosis": "the proximal LAD",
    "mid_lad_stenosis": "the mid LAD",
    "dist_lad_stenosis": "the distal LAD",
    "D1_stenosis": "D1 branch",
    "D2_stenosis": "D2 branch",
    "lcx_stenosis": "the proximal LCX",
    "dist_lcx_stenosis": "the distal LCX",
    "om1_stenosis": "OM1",
    "om2_stenosis": "OM2",
    "prox_rca_stenosis": "the proximal RCA",
    "mid_rca_stenosis": "the mid RCA",
    "dist_rca_stenosis": "the distal RCA",
    "pda_stenosis": "the PDA",
    "posterolateral_stenosis": "the posterolateral branch",
    "bx_stenosis": "Ramus",
    "lvp_stenosis": "left posterolateral branch",
    "lima_or_svg_stenosis": "the LIMA or SVG graft",
}

RCA_VESSELS = ["prox_rca_stenosis", "mid_rca_stenosis", "dist_rca_stenosis"]
NON_RCA_VESSELS = [
    "left_main_stenosis", "prox_lad_stenosis", "mid_lad_stenosis", "dist_lad_stenosis",
    "D1_stenosis", "D2_stenosis", "lcx_stenosis", "dist_lcx_stenosis",
    "om1_stenosis", "om2_stenosis", "bx_stenosis", "lvp_stenosis"
]
RIGHT_DOMINANCE_DEPENDENT_VESSELS = ["pda_stenosis", "posterolateral_stenosis"]
LEFT_DOMINANCE_DEPENDENT_VESSELS = ["pda_stenosis"]

# ──────────────────────────────────────────────────────────────────────────
# Formatting Functions
# ──────────────────────────────────────────────────────────────────────────

def format_stenosis_value(p: float) -> str:
    """Format stenosis percentage into descriptive text."""
    if p == 0:
        return "no significant stenosis"
    if p < 50:
        return f"mild stenosis (~{p}%)"
    if p < 70:
        return f"moderate stenosis (~{p}%)"
    if p < 90:
        return f"severe stenosis (~{p}%)"
    return f"critical stenosis (~{p}%)"


def format_calcification_value(c: str) -> str:
    """Format calcification description into standardized text."""
    txt = c.lower()
    if "no calcification" in txt or "pas de calcification" in txt:
        return "no calcifications"
    if any(k in txt for k in ("minimes", "mild")):
        return "minimal calcifications"
    if any(k in txt for k in ("modérées", "moderate")):
        return "moderate calcifications"
    if any(k in txt for k in ("importantes", "severe")):
        return "severe calcifications"
    return f"calcifications present ({c})"


def format_ifr_value(v: float) -> str:
    """Format IFR value into descriptive text."""
    return f"IFR {'normal' if v > 0.89 else 'abnormal'} (~{v:.2f})"


# ──────────────────────────────────────────────────────────────────────────
# Report Generation Functions
# ──────────────────────────────────────────────────────────────────────────

def create_report(row: pd.Series, coronary_specific_report: bool = True) -> str:
    """
    Create a detailed medical report based on coronary vessel analysis.
    
    Args:
        row: Pandas Series containing patient data
        coronary_specific_report: If True, create side-specific report
    
    Returns:
        Formatted medical report string
    """
    dom_raw = str(row.get("dominance_name", "")).replace("_", " ")
    dom_lower = dom_raw.lower()
    has_graft = ("pontage" in str(row.get("Conclusion", "")).lower()) or (row.get("bypass_graft", 0) == 1)

    # Override dominance when anatomy proves left-dominant:
    # lvp_stenosis > 0 means the LVP was evaluated and diseased, which only
    # exists in left-dominant (or co-dominant) hearts.
    lvp_val = pd.to_numeric(row.get("lvp_stenosis", 0), errors="coerce") or 0
    if lvp_val > 0:
        dom_lower = "left dominant"
        dom_raw = "left dominant"

    # Extend RCA/non-RCA sets based on dominance.
    # In left-dominant hearts LVP is the posterior vessel (LVP ≡ PDA), so
    # PDA is excluded — LVP already covers it in NON_RCA_VESSELS.
    # In right-dominant hearts PDA is the posterior vessel and LVP does not
    # exist anatomically, so LVP is excluded from non-RCA.
    if "right" in dom_lower:
        rca_extended = RCA_VESSELS + RIGHT_DOMINANCE_DEPENDENT_VESSELS
        non_rca_extended = [v for v in NON_RCA_VESSELS if v != "lvp_stenosis"]
    else:  # left- or co-dominant
        rca_extended = RCA_VESSELS
        non_rca_extended = NON_RCA_VESSELS  # LVP already included, no PDA needed

    # Choose which vessel list to describe
    if coronary_specific_report:
        side = row.get("main_structure_name")
        if side == "Right Coronary":
            local_order = rca_extended[:]
        elif side == "Left Coronary":
            local_order = non_rca_extended[:]
        else:
            return "Invalid main_structure_name for coronary-specific report"
    else:
        local_order = list(LABELS_TO_VESSEL_NAMES.keys())

    vessel_dict = LABELS_TO_VESSEL_NAMES.copy()
    if "left" in dom_lower:
        vessel_dict["pda_stenosis"] = "the LEFT PDA"
        vessel_dict["posterolateral_stenosis"] = "the LEFT posterolateral branch"

    if not has_graft and "lima_or_svg_stenosis" in local_order:
        local_order.remove("lima_or_svg_stenosis")

    # Assemble sentence(s)
    lines = []
    collateral_info = []  # Store collateral information separately
    
    for stenosis_lbl in local_order:
        prefix = stenosis_lbl.replace("_stenosis", "")
        vname = vessel_dict[stenosis_lbl]

        desc = []
        
        # Check for CTO first (overrides regular stenosis)
        cto_col = prefix + "_cto"
        is_cto = row.get(cto_col, 0)
        if pd.notna(is_cto) and is_cto == 1:
            desc.append("is 100% blocked and is a CTO")
        else:
            # Regular stenosis assessment
            st = row.get(stenosis_lbl, -1)
            if pd.notna(st) and st != -1:
                # Check for stent presence and restenosis
                stent_col = prefix + "_stent"
                has_stent = row.get(stent_col, 0)
                
                if pd.notna(has_stent) and has_stent > 0:
                    # Stent is present, check for restenosis
                    if float(st) <= 10:  # Very low stenosis in stented vessel
                        desc.append("no restenosis in stent")
                    else:
                        desc.append(f"in-stent restenosis ({format_stenosis_value(float(st))})")
                else:
                    # No stent, regular stenosis assessment
                    desc.append(format_stenosis_value(float(st)))

        # Calcification assessment
        calc = row.get(prefix + "_calcif", "-1")
        if isinstance(calc, str) and calc.strip() != "-1":
            desc.append(format_calcification_value(calc))

        # IFR assessment
        ifr = row.get(prefix + "_IFRHYPEREMIE", -1)
        if pd.notna(ifr) and ifr != -1:
            desc.append(format_ifr_value(float(ifr)))

        # Bifurcation lesion assessment
        bifurcation_col = prefix + "_bifurcation"
        bifurcation_value = row.get(bifurcation_col, None)
        if pd.notna(bifurcation_value) and bifurcation_value not in [0, 0.0, "", "0", "0.0", "Pas de lésion de bifurcation", "nan"]:
            if isinstance(bifurcation_value, str) and bifurcation_value.strip():
                # Skip if it's the French "no bifurcation lesion" text
                bifurcation_text = bifurcation_value.strip()
                if bifurcation_text.lower() != "pas de lésion de bifurcation":
                    desc.append(f"bifurcation lesion (Medina {bifurcation_text})")
            elif isinstance(bifurcation_value, (int, float)) and bifurcation_value != 0:
                desc.append(f"bifurcation lesion (Medina {bifurcation_value})")

        # Add vessel description if there are findings
        if desc:
            if len(desc) == 1:
                combined = desc[0]
            else:
                combined = ", ".join(desc[:-1]) + ", and " + desc[-1]
            lines.append(f"{vname} has {combined}.")
    
    # Check for collateral circulation - both receiving and giving
    if coronary_specific_report:
        # Only check collaterals for vessels that are part of the current coronary system
        vessels_to_check = local_order
    else:
        # For comprehensive reports, check all vessels
        vessels_to_check = list(LABELS_TO_VESSEL_NAMES.keys())
    
    # First pass: Find vessels that receive collaterals
    receiving_collaterals = []
    for stenosis_lbl in vessels_to_check:
        prefix = stenosis_lbl.replace("_stenosis", "")
        vessel_name = LABELS_TO_VESSEL_NAMES[stenosis_lbl]
        
        # Check for collateral column (this vessel receives collaterals)
        collateral_col = prefix + "_collateral"
        collateral_value = row.get(collateral_col, None)
        
        if pd.notna(collateral_value) and collateral_value not in [0, 0.0, "", "0", "0.0", "nan"]:
            # Handle both string vessel names and numeric codes
            if isinstance(collateral_value, str) and collateral_value.strip():
                donor_vessel = collateral_value.strip()
                # Skip if donor vessel is "nan" (case-insensitive)
                if donor_vessel.lower() != "nan":
                    receiving_collaterals.append(f"{vessel_name} receives collaterals from the {donor_vessel}.")
            elif isinstance(collateral_value, (int, float)) and collateral_value != 0:
                # If it's a numeric code, you might want to map it to vessel names
                receiving_collaterals.append(f"{vessel_name} receives collateral circulation (code: {collateral_value}).")
    
    # Second pass: Find vessels that give collaterals (check all vessels, not just current scope)
    giving_collaterals = []
    all_vessels = list(LABELS_TO_VESSEL_NAMES.keys())
    
    for current_vessel in vessels_to_check:  # Only report for vessels in current scope
        current_vessel_name = LABELS_TO_VESSEL_NAMES[current_vessel]
        
        # Check if this vessel is mentioned as a donor in any collateral column
        for other_vessel in all_vessels:
            other_prefix = other_vessel.replace("_stenosis", "")
            collateral_col = other_prefix + "_collateral"
            collateral_value = row.get(collateral_col, None)
            
            if pd.notna(collateral_value) and isinstance(collateral_value, str):
                donor_vessel = collateral_value.strip().lower()
                # Check if current vessel is the donor (flexible matching)
                current_vessel_variants = [
                    current_vessel_name.lower(),
                    current_vessel_name.lower().replace("the ", ""),
                    "rca" if "rca" in current_vessel_name.lower() else "",
                    "lad" if "lad" in current_vessel_name.lower() else "",
                    "lcx" if "lcx" in current_vessel_name.lower() else ""
                ]
                
                if any(variant and variant in donor_vessel for variant in current_vessel_variants if variant):
                    target_vessel_name = LABELS_TO_VESSEL_NAMES[other_vessel]
                    giving_collaterals.append(f"{current_vessel_name} gives collaterals to {target_vessel_name}.")
    
    # Add collateral information to the report
    collateral_info.extend(receiving_collaterals)
    collateral_info.extend(giving_collaterals)
    lines.extend(collateral_info)

    if dom_raw.strip():
        lines.append(f"The coronary circulation is {dom_raw}.")

    return "\n".join(lines) if lines else "No significant findings or additional data available."


# ──────────────────────────────────────────────────────────────────────────
# Data Processing Functions
# ──────────────────────────────────────────────────────────────────────────

# GT pcidone columns grouped by coronary side
_LCA_PCIDONE_COLS = [
    "left_main_pcidone", "prox_lad_pcidone", "mid_lad_pcidone", "dist_lad_pcidone",
    "D1_pcidone", "D2_pcidone", "prox_lcx_pcidone", "mid_lcx_pcidone", "dist_lcx_pcidone",
    "om1_pcidone", "om2_pcidone", "bx_pcidone", "lvp_pcidone",
]
_RCA_PCIDONE_COLS = [
    "prox_rca_pcidone", "mid_rca_pcidone", "dist_rca_pcidone",
    "pda_pcidone", "posterolateral_pcidone", "right_marginal_pcidone",
]


def _study_has_pci_on_side(df: pd.DataFrame) -> pd.Series:
    """Return a boolean Series: True when the GT *_pcidone columns confirm
    that a PCI was actually performed on the video's coronary side.

    If no *_pcidone columns are present, falls back to True (trust the
    stent_presence_class classifier as before).
    """
    lca_cols = [c for c in _LCA_PCIDONE_COLS if c in df.columns]
    rca_cols = [c for c in _RCA_PCIDONE_COLS if c in df.columns]

    if not lca_cols and not rca_cols:
        # No GT available — fall back to stent_presence_class
        return pd.Series(True, index=df.index)

    # Per-row: does the study have any pcidone > 0 for this video's side?
    lca_any = (df[lca_cols].apply(pd.to_numeric, errors="coerce").fillna(0) > 0).any(axis=1) if lca_cols else pd.Series(False, index=df.index)
    rca_any = (df[rca_cols].apply(pd.to_numeric, errors="coerce").fillna(0) > 0).any(axis=1) if rca_cols else pd.Series(False, index=df.index)

    is_left = df["main_structure_name"] == "Left Coronary"
    is_right = df["main_structure_name"] == "Right Coronary"

    # True when the GT says PCI happened on this video's artery side
    return (is_left & lca_any) | (is_right & rca_any) | (~is_left & ~is_right)


def assign_procedure_status(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign procedure status based on PCI (stent) presence and timing.

    This function creates three mutually-exclusive status categories:
    - "PCI": Current procedure has stent placement
    - "POST_PCI": Current procedure is after a previous PCI in the same study/artery
    - "diagnostic": Diagnostic procedure with no previous PCI

    Two key rules:
    1. stent_presence_class=1 → always PCI (never diagnostic), even if the
       structure classifier assigned the wrong coronary side.
    2. The PCI *cascade* (POST_PCI for later videos) only triggers when GT
       *_pcidone confirms PCI on that coronary side.  If pcidone=0 on the
       labelled side, the stent=1 video is skipped (PCI status) but does NOT
       contaminate neighbouring videos on that side — they stay diagnostic.

    Args:
        df: Input DataFrame with stent_presence_class and related columns

    Returns:
        DataFrame with added 'status' column
    """
    logger.info("Assigning procedure status based on PCI timing...")

    df_copy = df.copy()

    # ── 1. Ensure the column exists up front ────────────────────────────────────────
    df_copy["status"] = "unknown"          # will be overwritten below

    # ── 2a. stent=1 → always PCI for the video itself ──────────────────────────────
    df_copy["has_stent"] = df_copy["stent_presence_class"].eq(1)

    # ── 2b. Cascade only when GT pcidone confirms PCI on this side ──────────────────
    gt_pci_on_side = _study_has_pci_on_side(df_copy)
    df_copy["is_pci_cascade"] = df_copy["has_stent"] & gt_pci_on_side

    skipped = (df_copy["has_stent"] & ~gt_pci_on_side).sum()
    if skipped:
        logger.info("Stent=1 but pcidone=0 on labelled side: %d videos "
                     "(marked PCI but no cascade)", skipped)

    # cumulative "has confirmed PCI already been seen *earlier* in this study+artery?"
    group_cols = ["StudyInstanceUID", "main_structure_name"]
    df_copy["pci_seen_before"] = (
        df_copy
        .groupby(group_cols, sort=False)["is_pci_cascade"]
        .transform(lambda g: g.cumsum().shift(fill_value=0))
        .astype(bool)
    )

    # ── 3. Build the three mutually-exclusive conditions ───────────────────────────
    cond_pci        = df_copy["has_stent"]              # stent=1 → always PCI
    cond_post_pci   = (~cond_pci
                       & df_copy["pci_seen_before"]
                       & df_copy["contrast_agent_class"].eq(1))

    cond_diagnostic = ~cond_pci & ~df_copy["pci_seen_before"]

    # ── 4. Final assignment (vectorised) ───────────────────────────────────────────
    df_copy.loc[cond_pci,        "status"] = "PCI"
    df_copy.loc[cond_post_pci,   "status"] = "POST_PCI"
    df_copy.loc[cond_diagnostic, "status"] = "diagnostic"

    # ── 5. (Optional) tidy-up helper columns ───────────────────────────────────────
    df_copy.drop(columns=["has_stent", "is_pci_cascade", "pci_seen_before"], inplace=True)

    # Log status distribution
    status_counts = df_copy["status"].value_counts()
    logger.info(f"Status distribution: {status_counts.to_dict()}")

    return df_copy


def apply_hard_filters(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply hard filters to the dataset based on configuration.
    
    Args:
        df: Input DataFrame
        config: Configuration dictionary with filter parameters
    
    Returns:
        Filtered DataFrame
    """
    logger.info("Applying hard filters...")
    
    # Get filter criteria from config
    status_filter = config.get('filters', {}).get('status', 'diagnostic')
    main_structures = config.get('filters', {}).get('main_structures', ['Left Coronary', 'Right Coronary'])
    contrast_agent_class = config.get('filters', {}).get('contrast_agent_class', 1)
    
    # Handle both single status and list of statuses
    if isinstance(status_filter, str):
        # Single status - convert to list for uniform handling
        status_list = [status_filter]
    elif isinstance(status_filter, list):
        # Multiple statuses
        status_list = status_filter
    else:
        # Fallback to default
        logger.warning(f"Invalid status filter type: {type(status_filter)}. Using default 'diagnostic'")
        status_list = ['diagnostic']
    
    # Exclude congenital heart procedure studies (not coronary angiograms)
    congenital_mask = pd.Series(False, index=df.index)
    if "series_description" in df.columns:
        congenital_mask = df["series_description"].str.contains(
            "CONGENITAL", case=False, na=False
        )
        n_congenital = congenital_mask.sum()
        if n_congenital > 0:
            logger.info(
                "Excluding %d videos from %d congenital-procedure studies",
                n_congenital,
                df.loc[congenital_mask, "StudyInstanceUID"].nunique(),
            )

    # Exclude studies where every stenosis value is -1 or NaN (no coronary data)
    stenosis_cols = [c for c in df.columns if c.endswith("_stenosis")]
    no_stenosis_mask = pd.Series(False, index=df.index)
    if stenosis_cols:
        stenosis_vals = df[stenosis_cols].apply(pd.to_numeric, errors="coerce")
        no_stenosis_mask = ((stenosis_vals == -1) | stenosis_vals.isna()).all(axis=1)
        n_no_stenosis = no_stenosis_mask.sum()
        if n_no_stenosis > 0:
            logger.info(
                "Excluding %d videos with all stenosis = -1/NaN",
                n_no_stenosis,
            )

    # Apply filters
    filtered_df = df.loc[
        (df["status"].isin(status_list)) &
        (df["main_structure_name"].isin(main_structures)) &
        (df["contrast_agent_class"] == contrast_agent_class) &
        ~congenital_mask &
        ~no_stenosis_mask
    ].copy()

    logger.info(f"Dataset filtered from {len(df)} to {len(filtered_df)} rows")
    logger.info(f"Status filter applied: {status_list}")
    return filtered_df


def generate_reports(df: pd.DataFrame, coronary_specific: bool = True) -> pd.DataFrame:
    """
    Generate medical reports for all rows in the DataFrame.
    
    Args:
        df: Input DataFrame
        coronary_specific: Whether to generate coronary-specific reports
    
    Returns:
        DataFrame with added Report column
    """
    logger.info("Generating medical reports...")
    
    df_copy = df.copy()
    df_copy["Report"] = df_copy.progress_apply(
        lambda r: create_report(r, coronary_specific_report=coronary_specific), 
        axis=1
    )
    
    logger.info(f"Generated reports for {len(df_copy)} records")
    return df_copy


def sample_by_status(df: pd.DataFrame, n: int = 9, label_col: str = "status") -> Dict[str, pd.DataFrame]:
    """
    Sample data by status/label column.
    
    Args:
        df: Input DataFrame  
        n: Number of samples per status
        label_col: Column to group by
    
    Returns:
        Dictionary mapping status to sampled DataFrame
    """
    logger.info(f"Sampling {n} records per {label_col}...")
    
    return {
        lbl: sub.sample(n=min(n, len(sub)), random_state=42, replace=False)
        for lbl, sub in df.groupby(label_col)
    }


# ──────────────────────────────────────────────────────────────────────────
# Main Processing Functions
# ──────────────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_data(data_path: str) -> pd.DataFrame:
    """Load data from CSV or Parquet file."""
    logger.info(f"Loading data from {data_path}")
    
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    elif data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path}. Supported formats: .csv, .parquet")
    
    logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
    return df


def _extract_acq_time_from_filename(fn: str) -> Optional[float]:
    """Extract DICOM acquisition datetime (YYYYMMDDHHMMSS) from the SOP Instance UID in the filename.

    Many DICOM SOP UIDs embed the acquisition datetime as YYYYMMDDHHMMSS.
    This is more reliable than series_time (which can be a transfer/storage
    timestamp with corrupted hour values).

    Returns the full 14-digit datetime as a float so that procedures spanning
    midnight sort correctly (the date portion increments across midnight).
    """
    fn = str(fn)
    basename = fn.rsplit("/", 1)[-1]
    # The SOP UID is after the first '_' in StudyUID_SOPInstanceUID.dcm.avi
    parts = basename.split("_", 1)
    if len(parts) < 2:
        return None
    sop = parts[1]
    m = re.search(r"(20[12]\d[01]\d[0-3]\d\d{6})", sop)
    if m:
        return float(m.group(1))
    return None


def process_dataset(
    input_path: str,
    output_dir: str,
    config: Dict[str, Any]
) -> None:
    """
    Main dataset processing pipeline.
    
    Args:
        input_path: Path to input data file
        output_dir: Directory to save processed data
        config: Processing configuration
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_data(input_path)
    
    # Apply mapping if needed
    if 'apply_mappings' in config and config['apply_mappings']:
        if 'main_structure_class' in df.columns:
            df['main_structure_name'] = df['main_structure_class'].map(MAIN_STRUCTURE_MAP)
        if 'dominance_class' in df.columns:
            df['dominance_name'] = df['dominance_class'].map(DOMINANCE_MAP)
        
        # Sort by StudyInstanceUID and acquisition time for proper temporal ordering.
        # The most reliable time source is the DICOM acquisition timestamp embedded
        # in the SOP Instance UID portion of the filename (YYYYMMDDHHMMSS pattern).
        # series_time can be a transfer/storage time with corrupted values that break
        # sort order within a study, so we only use it as a fallback.
        if 'StudyInstanceUID' in df.columns and 'FileName' in df.columns:
            df['_acq_time'] = df['FileName'].apply(_extract_acq_time_from_filename)
            acq_coverage = df['_acq_time'].notna().sum()
            logger.info("Extracted acquisition time from filenames: %d / %d (%.1f%%)",
                        acq_coverage, len(df), 100.0 * acq_coverage / len(df))

            # Fallback to series_time / SeriesTime for videos without filename timestamps
            if 'series_time' in df.columns:
                fallback = pd.to_numeric(df['series_time'], errors='coerce')
                fallback = fallback.where(fallback > 0)  # -1 means missing
            elif 'SeriesTime' in df.columns:
                fallback = pd.to_numeric(df['SeriesTime'], errors='coerce')
            else:
                fallback = pd.Series(np.nan, index=df.index)

            df['_sort_time'] = df['_acq_time'].fillna(fallback)
            df = df.sort_values(['StudyInstanceUID', '_sort_time'], na_position='last')
            df.drop(columns=['_acq_time', '_sort_time'], inplace=True)
            logger.info("Sorted data by StudyInstanceUID and acquisition time")
        elif 'StudyInstanceUID' in df.columns:
            logger.warning("No FileName column — cannot extract acquisition time")
    
    # Assign procedure status based on PCI timing (if enabled and required columns exist)
    if config.get('assign_status', True):
        if 'stent_presence_class' in df.columns and 'StudyInstanceUID' in df.columns:
            df = assign_procedure_status(df)
        else:
            logger.warning("Cannot assign procedure status: missing required columns (stent_presence_class, StudyInstanceUID)")
    
    # Apply filters
    df_filtered = apply_hard_filters(df, config)
    
    # Generate reports
    coronary_specific = config.get('report_settings', {}).get('coronary_specific', True)
    df_with_reports = generate_reports(df_filtered, coronary_specific)
    
    # Save main processed dataset
    main_output_path = output_path / "processed_dataset.csv"
    df_with_reports.to_csv(main_output_path, index=False)
    logger.info(f"Saved processed dataset to {main_output_path}")
    
    # Generate samples if requested
    if config.get('sampling', {}).get('enabled', False):
        n_samples = config['sampling'].get('n_per_group', 9)
        label_col = config['sampling'].get('label_column', 'status')
        
        samples = sample_by_status(df_with_reports, n=n_samples, label_col=label_col)
        
        # Save samples
        samples_dir = output_path / "samples"
        samples_dir.mkdir(exist_ok=True)
        
        for label, sample_df in samples.items():
            sample_path = samples_dir / f"sample_{label}.csv"
            sample_df.to_csv(sample_path, index=False)
            logger.info(f"Saved {len(sample_df)} {label} samples to {sample_path}")
    
    # Save configuration used
    config_output_path = output_path / "processing_config.yaml"
    with open(config_output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info("Dataset processing completed successfully!")


# ──────────────────────────────────────────────────────────────────────────
# CLI Interface
# ──────────────────────────────────────────────────────────────────────────

def create_default_config() -> Dict[str, Any]:
    """Create default configuration."""
    return {
        'filters': {
            'status': 'diagnostic',  # Options: 'diagnostic', 'PCI', 'POST_PCI' or ['diagnostic', 'POST_PCI']
            'main_structures': ['Left Coronary', 'Right Coronary'],
            'contrast_agent_class': 1
        },
        'report_settings': {
            'coronary_specific': True
        },
        'sampling': {
            'enabled': True,
            'n_per_group': 9,
            'label_column': 'status'
        },
        'apply_mappings': True,
        'assign_status': True  # Whether to auto-assign status based on PCI timing
    }


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Generate medical dataset with coronary vessel reports"
    )
    parser.add_argument(
        '--input-csv', 
        required=True,
        help='Path to input CSV or Parquet file'
    )
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Directory to save processed dataset'
    )
    parser.add_argument(
        '--config',
        help='Path to configuration YAML file (optional)'
    )
    parser.add_argument(
        '--create-default-config',
        action='store_true',
        help='Create a default configuration file and exit'
    )
    
    args = parser.parse_args()
    
    if args.create_default_config:
        config = create_default_config()
        with open('default_config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print("Created default_config.yaml")
        return
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        logger.info("No config provided, using default configuration")
        config = create_default_config()
    
    # Process dataset
    process_dataset(args.input_csv, args.output_dir, config)


if __name__ == "__main__":
    main() 