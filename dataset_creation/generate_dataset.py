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

# Angiographic view classifications
ANGIOGRAPHIC_VIEWS = [
    {"label": 0, "description": "RAO Cranial", "primary_range": (-60, -15), "secondary_range": (15, 50)},
    {"label": 1, "description": "AP Cranial", "primary_range": (-15, 15), "secondary_range": (15, 50)},
    {"label": 2, "description": "LAO Cranial", "primary_range": (15, 60), "secondary_range": (15, 50)},
    {"label": 3, "description": "RAO Straight", "primary_range": (-60, -15), "secondary_range": (-15, 15)},
    {"label": 4, "description": "AP", "primary_range": (-15, 15), "secondary_range": (-15, 15)},
    {"label": 5, "description": "LAO Straight", "primary_range": (15, 60), "secondary_range": (-15, 15)},
    {"label": 6, "description": "RAO Caudal", "primary_range": (-60, -15), "secondary_range": (-50, -15)},
    {"label": 7, "description": "AP Caudal", "primary_range": (-15, 15), "secondary_range": (-50, -15)},
    {"label": 8, "description": "LAO Caudal", "primary_range": (15, 60), "secondary_range": (-50, -15)},
    {"label": 9, "description": "LAO Lateral", "primary_range": (-110, -70), "secondary_range": (-15, 15)},
    {"label": 10, "description": "RAO Lateral", "primary_range": (70, 110), "secondary_range": (-15, 15)},
    {"label": 11, "description": "Other", "primary_range": None, "secondary_range": None}
]

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
LEFT_DOMINANCE_DEPENDENT_VESSELS = ["pda_stenosis", "lvp_stenosis"]

# ──────────────────────────────────────────────────────────────────────────
# Angiographic View Functions
# ──────────────────────────────────────────────────────────────────────────

def classify_angiographic_view(primary_angle: float, secondary_angle: float) -> tuple:
    """
    Classify angiographic view based on primary and secondary angles.
    
    Args:
        primary_angle: Primary angle (LAO/RAO) in degrees
        secondary_angle: Secondary angle (Cranial/Caudal) in degrees
    
    Returns:
        Tuple of (view_label, view_description)
    """
    for view in ANGIOGRAPHIC_VIEWS[:-1]:  # Check all views except "Other"
        primary_range = view["primary_range"]
        secondary_range = view["secondary_range"]
        
        if (primary_range[0] <= primary_angle <= primary_range[1] and
            secondary_range[0] <= secondary_angle <= secondary_range[1]):
            return view["label"], view["description"]
    
    # If no match found, return "Other"
    return 11, "Other"


def add_angiographic_view_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add angiographic view classification to DataFrame.
    
    Args:
        df: DataFrame with primary_angle and secondary_angle columns
    
    Returns:
        DataFrame with added angiographic_view_label and angiographic_view_description columns
    """
    if 'primary_angle' in df.columns and 'secondary_angle' in df.columns:
        df['angiographic_view_label'], df['angiographic_view_description'] = zip(*df.apply(
            lambda row: classify_angiographic_view(row['primary_angle'], row['secondary_angle'])
            if pd.notna(row['primary_angle']) and pd.notna(row['secondary_angle'])
            else (11, "Other"),
            axis=1
        ))
    else:
        logger.warning("primary_angle or secondary_angle columns not found. Skipping angiographic view classification.")
        df['angiographic_view_label'] = 11
        df['angiographic_view_description'] = "Other"
    
    return df


def format_main_structure_description(main_structure_class: int) -> str:
    """
    Convert main_structure_class to descriptive string.
    
    Args:
        main_structure_class: Integer class value
    
    Returns:
        Descriptive string for the structure
    """
    structure_name = MAIN_STRUCTURE_MAP.get(main_structure_class, "Unknown")
    
    if structure_name == "Left Coronary":
        return "This is a left coronary artery"
    elif structure_name == "Right Coronary":
        return "This is a right coronary artery"
    elif structure_name == "Graft":
        return "This is a graft vessel"
    elif structure_name == "Catheter":
        return "This is a catheter"
    elif structure_name == "LV":
        return "This is a left ventricle view"
    elif structure_name == "Aorta":
        return "This is an aortic view"
    else:
        return f"This is a {structure_name.lower()} view"


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
    
    # Start with angiographic view description if available
    report_lines = []
    
    # Add angiographic view description
    view_desc = row.get("angiographic_view_description", "")
    if view_desc and view_desc != "Other":
        report_lines.append(f"The view is {view_desc}.")
    elif view_desc == "Other":
        report_lines.append(f"") 
        #Skip non-conventional views
    
    # Add main structure description
    main_structure_desc = row.get("main_structure_description", "")
    if main_structure_desc:
        report_lines.append(main_structure_desc + ".")

    # Extend RCA/non-RCA sets based on dominance
    if "right" in dom_lower:
        rca_extended = RCA_VESSELS + RIGHT_DOMINANCE_DEPENDENT_VESSELS
        non_rca_extended = NON_RCA_VESSELS
    else:  # left- or co-dominant
        rca_extended = RCA_VESSELS
        non_rca_extended = NON_RCA_VESSELS + LEFT_DOMINANCE_DEPENDENT_VESSELS

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

    # Assemble sentence(s) for vessel findings
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
    
    # Combine initial report lines (view and structure) with vessel findings
    all_lines = report_lines + lines
    return "\n".join(all_lines) if all_lines else "No significant findings or additional data available."


# ──────────────────────────────────────────────────────────────────────────
# Data Processing Functions
# ──────────────────────────────────────────────────────────────────────────

def assign_procedure_status(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign procedure status based on PCI (stent) presence and timing.
    
    This function creates three mutually-exclusive status categories:
    - "PCI": Current procedure has stent placement
    - "POST_PCI": Current procedure is after a previous PCI in the same study/artery
    - "diagnostic": Diagnostic procedure with no previous PCI
    
    Args:
        df: Input DataFrame with stent_presence_class and related columns
    
    Returns:
        DataFrame with added 'status' column
    """
    logger.info("Assigning procedure status based on PCI timing...")
    
    df_copy = df.copy()
    
    # ── 1. Ensure the column exists up front ────────────────────────────────────────
    df_copy["status"] = "unknown"          # will be overwritten below

    # ── 2. Convenience flags ────────────────────────────────────────────────────────
    df_copy["is_pci"] = df_copy["stent_presence_class"].eq(1)

    # cumulative "has PCI already been seen *earlier* in this study AND artery?"
    group_cols = ["StudyInstanceUID", "main_structure_name"]
    df_copy["pci_seen_before"] = (
        df_copy
        .groupby(group_cols, sort=False)["is_pci"]
        .cumsum()            # running total …
        .shift(fill_value=0) # … but *before* the current row
        .astype(bool)
    )

    # ── 3. Build the three mutually-exclusive conditions ───────────────────────────
    cond_pci        = df_copy["is_pci"]
    cond_post_pci   = (~cond_pci
                       & df_copy["pci_seen_before"]
                       & df_copy["contrast_agent_class"].eq(1))

    cond_diagnostic = ~cond_pci & ~df_copy["pci_seen_before"]

    # ── 4. Final assignment (vectorised) ───────────────────────────────────────────
    df_copy.loc[cond_pci,        "status"] = "PCI"
    df_copy.loc[cond_post_pci,   "status"] = "POST_PCI"
    df_copy.loc[cond_diagnostic, "status"] = "diagnostic"

    # ── 5. (Optional) tidy-up helper columns ───────────────────────────────────────
    df_copy.drop(columns=["is_pci", "pci_seen_before"], inplace=True)

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
    initial_len = len(df)
    
    # Drop rows where 'External_Exam' is True
    if 'External_Exam' in df.columns:
        df = df[df["External_Exam"] != True]
        logger.info(f"Dropped External_Exam: {initial_len} -> {len(df)} rows")
    
    # Drop rows where 'bypass_graft' is 1
    if 'bypass_graft' in df.columns:
        df = df[df["bypass_graft"] != 1]
        logger.info(f"Dropped bypass_graft: {len(df)} rows remaining")
    
    # Filter stenosis columns - keep rows where at least one stenosis is not NaN or -1.0
    stenosis_columns = [
        "prox_rca_stenosis", "mid_rca_stenosis", "dist_rca_stenosis",
        "left_main_stenosis", "prox_lad_stenosis", "mid_lad_stenosis", "dist_lad_stenosis",
        "D1_stenosis", "D2_stenosis", "prox_lcx_stenosis", "dist_lcx_stenosis",
        "om1_stenosis", "om2_stenosis", "bx_stenosis", "lvp_stenosis",
        "pda_stenosis", "posterolateral_stenosis"
    ]
    
    # Check which stenosis columns exist in the dataframe
    existing_stenosis_cols = [col for col in stenosis_columns if col in df.columns]
    
    if existing_stenosis_cols:
        # Create a boolean DataFrame indicating if each cell is NaN or -1.0
        is_na_or_minus_one = df[existing_stenosis_cols].isna() | (df[existing_stenosis_cols] == -1.0)
        
        # Create a mask for rows where NOT ALL stenosis columns are NaN or -1.0
        mask = ~is_na_or_minus_one.all(axis=1)
        
        # Filter the DataFrame
        df = df[mask]
        logger.info(f"Filtered stenosis columns: {len(df)} rows remaining")
    
    # Get filter criteria from config
    status_filter = config.get('filters', {}).get('status', 'diagnostic')
    main_structures = config.get('filters', {}).get('main_structures', ['Left Coronary', 'Right Coronary'])
    contrast_agent_class = config.get('filters', {}).get('contrast_agent_class', 1)
    
    # Handle both single status and list of statuses
    if isinstance(status_filter, str):
        status_list = [status_filter]
    elif isinstance(status_filter, list):
        status_list = status_filter
    else:
        logger.warning(f"Invalid status filter type: {type(status_filter)}. Using default 'diagnostic'")
        status_list = ['diagnostic']
    
    # Apply additional filters
    if 'status' in df.columns:
        df = df[df["status"].isin(status_list)]
        logger.info(f"Status filter applied: {status_list}")
    
    if 'main_structure_name' in df.columns:
        df = df[df["main_structure_name"].isin(main_structures)]
        logger.info(f"Main structure filter applied: {main_structures}")
    
    if 'contrast_agent_class' in df.columns:
        df = df[df["contrast_agent_class"] == contrast_agent_class]
        logger.info(f"Contrast agent filter applied: {contrast_agent_class}")
    
    logger.info(f"Dataset filtered from {initial_len} to {len(df)} rows")
    return df.copy()


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


def assign_patient_splits(df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.1, 
                         test_ratio: float = 0.2, random_state: int = 42, 
                         patient_column: str = 'CathReport_MRN') -> pd.DataFrame:
    """
    Assigns patients to train/val/test splits based on patient IDs.
    
    Args:
        df: Input DataFrame
        train_ratio: Ratio of patients for training set
        val_ratio: Ratio of patients for validation set
        test_ratio: Ratio of patients for test set (computed if not provided)
        random_state: Random seed for reproducibility
        patient_column: Column name containing patient IDs
    
    Returns:
        DataFrame with 'Split' column added
    """
    # Check if patient column exists
    if patient_column not in df.columns:
        # Fallback to StudyInstanceUID if available
        if 'StudyInstanceUID' in df.columns:
            patient_column = 'StudyInstanceUID'
            logger.info(f"Using StudyInstanceUID for patient splits (CathReport_MRN not found)")
        else:
            logger.warning("No patient identifier column found. Using random split.")
            # Add a dummy Split column with random assignment
            n = len(df)
            train_size = int(train_ratio * n)
            val_size = int(val_ratio * n)
            
            indices = np.arange(n)
            np.random.seed(random_state)
            np.random.shuffle(indices)
            
            df_copy = df.copy()
            df_copy['Split'] = 'test'
            df_copy.iloc[indices[:train_size], df_copy.columns.get_loc('Split')] = 'train'
            df_copy.iloc[indices[train_size:train_size+val_size], df_copy.columns.get_loc('Split')] = 'val'
            
            return df_copy
    
    # Get unique patients
    unique_patients = df[patient_column].drop_duplicates()
    n_patients = len(unique_patients)
    
    # Calculate split sizes
    train_size = int(train_ratio * n_patients)
    val_size = int(val_ratio * n_patients)
    
    # Sample patients for each split
    np.random.seed(random_state)
    patient_indices = np.arange(n_patients)
    np.random.shuffle(patient_indices)
    
    train_patients = unique_patients.iloc[patient_indices[:train_size]]
    val_patients = unique_patients.iloc[patient_indices[train_size:train_size+val_size]]
    test_patients = unique_patients.iloc[patient_indices[train_size+val_size:]]
    
    # Create copy and assign splits
    df_copy = df.copy()
    df_copy['Split'] = 'test'  # Default to test
    df_copy.loc[df_copy[patient_column].isin(train_patients), 'Split'] = 'train'
    df_copy.loc[df_copy[patient_column].isin(val_patients), 'Split'] = 'val'
    
    # Log split statistics
    split_counts = df_copy['Split'].value_counts()
    logger.info(f"Split distribution: {split_counts.to_dict()}")
    
    if patient_column in df.columns:
        patient_split_counts = df_copy.groupby('Split')[patient_column].nunique()
        logger.info(f"Unique patients per split: {patient_split_counts.to_dict()}")
    
    return df_copy


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
            # Add main structure description
            df['main_structure_description'] = df['main_structure_class'].apply(format_main_structure_description)
        if 'dominance_class' in df.columns:
            df['dominance_name'] = df['dominance_class'].map(DOMINANCE_MAP)
        
        # Add angiographic view classification
        df = add_angiographic_view_column(df)
        
        # Create bypass_graft column if Conclusion column exists
        if 'Conclusion' in df.columns:
            df['bypass_graft'] = df['Conclusion'].str.contains('pontage', case=False, na=False).astype(int)
            logger.info("Created bypass_graft column from Conclusion")
        
        # Sort by StudyInstanceUID and SeriesTime for proper temporal ordering
        if 'StudyInstanceUID' in df.columns and 'SeriesTime' in df.columns:
            df = df.sort_values(['StudyInstanceUID', 'SeriesTime'])
            logger.info("Sorted data by StudyInstanceUID and SeriesTime for temporal ordering")
    
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
    
    # Apply patient-based train/val/test split if configured
    if config.get('train_test_split', {}).get('enabled', False):
        train_ratio = config['train_test_split'].get('train_ratio', 0.7)
        val_ratio = config['train_test_split'].get('val_ratio', 0.1)
        test_ratio = config['train_test_split'].get('test_ratio', 0.2)
        random_state = config['train_test_split'].get('random_state', 42)
        patient_column = config['train_test_split'].get('patient_column', 'CathReport_MRN')
        separator = config.get('output_settings', {}).get('separator', ',')
        
        # Add patient-based splits
        df_with_splits = assign_patient_splits(
            df_with_reports, 
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_state=random_state,
            patient_column=patient_column
        )
        
        # Save single file with Split column
        main_output_path = output_path / "dataset_with_splits.csv"
        df_with_splits.to_csv(main_output_path, sep=separator, index=False)
        logger.info(f"Saved dataset with splits to {main_output_path}")
        
        # Also save separate files for each split if requested
        if config['train_test_split'].get('save_separate_files', False):
            for split_name in ['train', 'val', 'test']:
                split_df = df_with_splits[df_with_splits['Split'] == split_name]
                split_output_path = output_path / f"{split_name}_dataset.csv"
                split_df.to_csv(split_output_path, sep=separator, index=False)
                logger.info(f"Saved {split_name} dataset to {split_output_path} ({len(split_df)} rows)")
        
        # Update df_with_reports for downstream processing
        df_with_reports = df_with_splits
    else:
        # Save main processed dataset
        separator = config.get('output_settings', {}).get('separator', ',')
        main_output_path = output_path / "processed_dataset.csv"
        df_with_reports.to_csv(main_output_path, sep=separator, index=False)
        logger.info(f"Saved processed dataset to {main_output_path}")
    
    # Generate samples if requested
    if config.get('sampling', {}).get('enabled', False):
        n_samples = config['sampling'].get('n_per_group', 9)
        label_col = config['sampling'].get('label_column', 'status')
        
        # If splits are enabled, only sample from train set
        if 'Split' in df_with_reports.columns:
            sample_source = df_with_reports[df_with_reports['Split'] == 'train']
            logger.info(f"Sampling from train split only ({len(sample_source)} rows)")
        else:
            sample_source = df_with_reports
        
        samples = sample_by_status(sample_source, n=n_samples, label_col=label_col)
        
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
            'status': ['diagnostic', 'POST_PCI'],  # Options: 'diagnostic', 'PCI', 'POST_PCI' or list
            'main_structures': ['Left Coronary', 'Right Coronary'],
            'contrast_agent_class': 1
        },
        'report_settings': {
            'coronary_specific': True
        },
        'train_test_split': {
            'enabled': True,
            'train_ratio': 0.7,
            'val_ratio': 0.1,
            'test_ratio': 0.2,
            'random_state': 42,
            'patient_column': 'CathReport_MRN',
            'save_separate_files': False
        },
        'sampling': {
            'enabled': True,
            'n_per_group': 9,
            'label_column': 'status'
        },
        'output_settings': {
            'separator': 'α',
            'include_index': False
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