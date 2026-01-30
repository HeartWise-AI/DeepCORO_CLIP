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
from typing import Dict, Any, Optional, List, Tuple
import re
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

SEGMENT_NAME_MAP = {
    "left_main": "Left Main",
    "prox_lad": "Proximal LAD",
    "mid_lad": "Mid LAD",
    "dist_lad": "Distal LAD",
    "d1": "D1",
    "d2": "D2",
    "dist_lcx": "Distal LCx",
    "prox_lcx": "Proximal LCx",
    "lcx": "Proximal LCx",
    "om1": "OM1",
    "om2": "OM2",
    "ramus": "Ramus",
    "bx": "Ramus",
    "pda": "PDA",
    "posterolateral": "Posterolateral",
    "prox_rca": "Proximal RCA",
    "mid_rca": "Mid RCA",
    "dist_rca": "Distal RCA",
    "lvp": "LVP branch",
}

BIN_TO_SEVERITY = {
    "<30": "mild",
    "30-49": "mild",
    "50-69": "moderate",
    "70-89": "severe",
    ">=90": "critical",
    "100": "critical",
    "cto": "critical",
}

BIN_TO_PERCENT = {
    "<30": "<30%",
    "30-49": "30-49%",
    "50-69": "50-69%",
    "70-89": "70-89%",
    ">=90": ">=90%",
    "100": "100%",
    "cto": "100%",
}

SEVERITY_PHRASES = {
    "normal": "normal segment",
    "mild": "mild stenosis",
    "moderate": "moderate stenosis",
    "severe": "severe stenosis",
    "critical": "critical stenosis",
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


def parse_tags_field(tags_str: Any) -> Dict[str, str]:
    """Parse the pipe-delimited tags column into a dictionary."""
    tags: Dict[str, str] = {}
    if not isinstance(tags_str, str):
        return tags
    for token in tags_str.split("|"):
        if ":" not in token:
            continue
        key, value = token.split(":", 1)
        key = key.strip().lower()
        value = value.strip()
        if key:
            tags[key] = value
    return tags


def format_segment_name(segment: Optional[str]) -> str:
    """Return a human-readable segment name."""
    if not isinstance(segment, str) or not segment.strip():
        return "Unknown segment"
    normalized = segment.strip().lower()
    return SEGMENT_NAME_MAP.get(normalized, segment.replace("_", " ").strip().title())


def detect_calcification_level(prompt_text: str) -> Optional[str]:
    """Extract calcification severity descriptor from the prompt text."""
    text = (prompt_text or "").lower()
    if "calcification" not in text and "calcifications" not in text:
        return None
    if "severe" in text:
        return "severe calcification"
    if "moderate" in text:
        return "moderate calcification"
    if "minimal" in text or "mild" in text:
        return "mild calcification"
    return "calcification"


def _infer_severity_label(row: pd.Series, tags: Dict[str, str]) -> str:
    """Infer normalized severity label from disease_severity and bin."""
    severity = str(row.get("disease_severity") or "").strip().lower()
    if severity in {"critical", "cto"}:
        return "critical"
    if severity:
        return severity
    bin_label = str(row.get("bin") or tags.get("bin") or "").strip().lower()
    return BIN_TO_SEVERITY.get(bin_label, "normal")


def build_canonical_siglip_prompt(row: pd.Series) -> Tuple[str, Tuple[str, ...]]:
    """
    Build a canonical prompt text and its deduplication key
    based on segment, severity, lesion attributes, and collateral info.
    """
    tags = parse_tags_field(row.get("tags"))
    prompt_text = str(row.get("prompt_text") or "")
    category = str(row.get("category") or "").strip().lower()
    base_segment = row.get("segment") or tags.get("segment") or ""
    segment_code = str(base_segment).strip().lower()
    segment_name = format_segment_name(segment_code or prompt_text.split(";")[0])

    prompt_lower = prompt_text.lower()
    severity = _infer_severity_label(row, tags)
    bin_label = str(row.get("bin") or tags.get("bin") or "").strip().lower()
    percent_text = BIN_TO_PERCENT.get(bin_label, "")

    has_cto = "cto" in prompt_lower or category == "cto"
    has_thrombus = "thrombus" in prompt_lower or category == "thrombus"
    has_bifurcation = "bifurcation" in prompt_lower or category == "medina"
    has_calcification = "calcification" in prompt_lower or category == "calcification"
    calc_level = detect_calcification_level(prompt_text) if has_calcification else None
    has_stent = "stent" in prompt_lower or category == "in_stent"
    has_restenosis = "restenosis" in prompt_lower or category == "in_stent"

    has_stenosis = (
        "stenosis" in prompt_lower
        or category in {"stenosis", "in_stent", "medina"}
        or has_restenosis
    )

    medina_type = tags.get("medina")
    extras_codes: List[str] = []
    extras_phrases: List[str] = []

    if has_bifurcation:
        if medina_type:
            extras_codes.append(f"bifurcation:{medina_type}")
            extras_phrases.append(f"Medina {medina_type} bifurcation lesion")
        else:
            extras_codes.append("bifurcation")
            extras_phrases.append("a bifurcation lesion")
    if calc_level:
        extras_codes.append(f"calc:{calc_level}")
        extras_phrases.append(calc_level)
    if has_thrombus:
        extras_codes.append("thrombus")
        extras_phrases.append("thrombus")
    if has_stent and not has_restenosis:
        extras_codes.append("stent")
        extras_phrases.append("a stent present")

    ifr_phrase = None
    ifr_match = re.search(r"(ABNORMAL|NORMAL)?\s*IFR\s*([0-9.]+)", prompt_text, flags=re.IGNORECASE)
    if ifr_match:
        status = ifr_match.group(1).strip().lower() if ifr_match.group(1) else ""
        value = ifr_match.group(2)
        if status:
            ifr_phrase = f"{status.lower()} IFR ~{value}"
        else:
            ifr_phrase = f"IFR ~{value}"
        extras_codes.append("ifr")
        extras_phrases.append(ifr_phrase)

    base_code: str
    base_phrase: str
    severity_title = severity.capitalize() if severity else ""
    normal_phrase = SEVERITY_PHRASES.get("normal", "normal segment")

    if has_cto:
        base_code = "cto"
        base_phrase = "CTO (100% occlusion)"
    elif has_thrombus and not has_stenosis:
        base_code = "thrombus"
        base_phrase = "thrombus present"
    elif has_stenosis and severity != "normal":
        base_code = f"stenosis:{severity}"
        severity_word = severity_title.lower() if severity_title else "moderate"
        base_phrase = f"{severity_word} stenosis"
        if percent_text:
            base_phrase += f" ({percent_text})"
        if has_restenosis:
            base_phrase += " with in-stent restenosis"
    else:
        base_code = "normal"
        base_phrase = normal_phrase

    if base_code == "thrombus" and "thrombus" in extras_codes:
        filtered_pairs = [
            (code, phrase)
            for code, phrase in zip(extras_codes, extras_phrases)
            if code != "thrombus"
        ]
        extras_codes = [code for code, _ in filtered_pairs]
        extras_phrases = [phrase for _, phrase in filtered_pairs]

    # Append extras (calcification, medina, IFR, etc.)
    if extras_phrases:
        extras_clause = " and ".join(extras_phrases)
        if " with " in base_phrase:
            base_phrase = f"{base_phrase} and {extras_clause}"
        else:
            base_phrase = f"{base_phrase} with {extras_clause}"

    collateral_phrases: List[str] = []
    receives = {
        match.group(2).strip().rstrip(".")
        for match in re.finditer(r"(receives collaterals from)\s+([^.;]+)", prompt_text, flags=re.IGNORECASE)
    }
    gives = {
        match.group(2).strip().rstrip(".")
        for match in re.finditer(r"(gives collaterals to)\s+([^.;]+)", prompt_text, flags=re.IGNORECASE)
    }
    for src in sorted(receives):
        collateral_phrases.append(f"{segment_name} receives collaterals from {src}")
    for dest in sorted(gives):
        collateral_phrases.append(f"{segment_name} gives collaterals to {dest}")

    clauses = [f"{segment_name}; {base_phrase}."]
    for clause in collateral_phrases:
        clauses.append(f"{clause}.")
    canonical_text = " ".join(clauses)

    canonical_key = (
        segment_code or segment_name.lower(),
        base_code,
        tuple(sorted(extras_codes)),
    )
    return canonical_text, canonical_key


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
        .transform(lambda x: x.cumsum().shift(fill_value=0))
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
    
    # Apply filters
    filtered_df = df.loc[
        (df["status"].isin(status_list)) &
        (df["main_structure_name"].isin(main_structures)) &
        (df["contrast_agent_class"] == contrast_agent_class)
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


def canonicalize_siglip_texts(
    texts_path: Path,
    videos_path: Optional[Path] = None,
    output_texts_path: Optional[Path] = None,
    output_videos_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Deduplicate SigLIP prompts so that each segment/severity combination
    has a single canonical description.
    """
    texts_path = Path(texts_path)
    videos_path = Path(videos_path) if videos_path else None
    output_texts_path = Path(output_texts_path or texts_path)
    output_videos_path = Path(output_videos_path or videos_path) if videos_path else None

    logger.info(f"Loading SigLIP texts from {texts_path}")
    texts_df = pd.read_csv(texts_path)
    texts_df["__prompt_len"] = texts_df["prompt_text"].astype(str).str.len()
    texts_df = texts_df.sort_values("__prompt_len", ascending=False)

    key_to_id: Dict[Tuple[str, str, Tuple[str, ...]], str] = {}
    id_remap: Dict[str, str] = {}
    canonical_rows: List[Dict[str, Any]] = []

    for _, row in texts_df.iterrows():
        text_id = row["text_id"]
        canonical_text, canonical_key = build_canonical_siglip_prompt(row)
        if canonical_key in key_to_id:
            id_remap[text_id] = key_to_id[canonical_key]
            continue
        key_to_id[canonical_key] = text_id
        id_remap[text_id] = text_id
        updated_row = row.copy()
        updated_row["prompt_text"] = canonical_text
        canonical_rows.append(updated_row)

    canonical_df = pd.DataFrame(canonical_rows)
    drop_cols = [col for col in canonical_df.columns if col.startswith("__")]
    if drop_cols:
        canonical_df = canonical_df.drop(columns=drop_cols)
    canonical_df.sort_values("text_id", inplace=True)
    canonical_df.to_csv(output_texts_path, index=False)
    logger.info(
        "Canonicalized SigLIP texts: %d -> %d unique prompts",
        len(texts_df),
        len(canonical_df),
    )

    stats = {
        "original_texts": len(texts_df),
        "canonical_texts": len(canonical_df),
        "dropped_texts": len(texts_df) - len(canonical_df),
    }

    if videos_path and videos_path.exists():
        logger.info(f"Updating SigLIP videos at {videos_path}")
        videos_df = pd.read_csv(videos_path)

        def remap_ids(text_ids: Any) -> str:
            if not isinstance(text_ids, str) or not text_ids.strip():
                return text_ids
            new_ids: List[str] = []
            for tid in text_ids.split("|"):
                tid = tid.strip()
                if not tid:
                    continue
                canonical_id = id_remap.get(tid, tid)
                if canonical_id not in new_ids:
                    new_ids.append(canonical_id)
            return "|".join(new_ids)

        videos_df["positive_text_ids"] = videos_df["positive_text_ids"].apply(remap_ids)
        videos_df.to_csv(output_videos_path or videos_path, index=False)
        logger.info("Updated SigLIP videos with canonical text IDs")
        stats["videos_updated"] = True
    else:
        stats["videos_updated"] = False

    return stats


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
        help='Path to input CSV or Parquet file'
    )
    parser.add_argument(
        '--output-dir',
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
    parser.add_argument(
        '--canonicalize-siglip',
        action='store_true',
        help='Canonicalize SigLIP texts/videos and exit'
    )
    parser.add_argument(
        '--siglip-output-dir',
        help='Directory containing SigLIP texts.csv/videos.csv'
    )
    parser.add_argument(
        '--siglip-texts',
        help='Path to SigLIP texts.csv (overrides --siglip-output-dir)'
    )
    parser.add_argument(
        '--siglip-videos',
        help='Path to SigLIP videos.csv (overrides --siglip-output-dir)'
    )
    
    args = parser.parse_args()
    
    if args.create_default_config:
        config = create_default_config()
        with open('default_config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print("Created default_config.yaml")
        return

    if args.canonicalize_siglip:
        base_dir = Path(args.siglip_output_dir or "output_dataset/siglip_generated")
        texts_path = Path(args.siglip_texts) if args.siglip_texts else base_dir / "texts.csv"
        videos_path = Path(args.siglip_videos) if args.siglip_videos else base_dir / "videos.csv"
        stats = canonicalize_siglip_texts(texts_path, videos_path)
        logger.info("Canonicalization summary: %s", stats)
        return

    if not args.input_csv or not args.output_dir:
        parser.error("--input-csv and --output-dir are required unless --canonicalize-siglip is specified")

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
