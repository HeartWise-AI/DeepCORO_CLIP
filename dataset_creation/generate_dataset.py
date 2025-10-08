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
from collections import defaultdict
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

PROMPT_WEIGHTS = {
    "global_summary": 0.5,
    "abnormal_focus": 1.0,
    "lesion_atomic": 0.6,
    "negative_coverage": 0.6,
}


SEVERITY_BINS: List[Tuple[float, float, str, str]] = [
    (0.0, 1.0, "none", ""),
    (1.0, 30.0, "minimal", "<30%"),
    (30.0, 50.0, "mild", "30-49%"),
    (50.0, 70.0, "moderate", "50-69%"),
    (70.0, 90.0, "severe", "70-89%"),
    (90.0, float("inf"), "critical", "≥90%"),
]


SEVERITY_RANK = {
    "minimal": 0,
    "mild": 1,
    "moderate": 2,
    "severe": 3,
    "critical": 4,
}


SEVERITY_DESCRIPTOR = {
    "minimal": "minimal",
    "mild": "mild",
    "moderate": "moderate",
    "severe": "severe",
    "critical": "critical/occlusion-range",
}


TOKEN_NAME_MAP = {
    "prox": "Proximal",
    "mid": "Mid",
    "dist": "Distal",
    "left": "Left",
    "main": "main",
    "right": "Right",
    "lad": "LAD",
    "lcx": "LCx",
    "rca": "RCA",
    "pda": "PDA",
    "bx": "Ramus",
    "lvp": "LVP branch",
    "posterolateral": "Posterolateral branch",
    "diagonal": "Diagonal branch",
    "ramus": "Ramus",
    "marg": "Marginal",
    "marginal": "Marginal",
    "septal": "Septal",
    "s": "Septal",
    "rv": "RV",
    "rvg": "RVG",
    "om": "OM",
    "lima": "LIMA",
    "svg": "SVG",
}


NEGATIVE_TERRITORIES = {
    "Left main": {"left_main", "leftmain"},
    "LAD territory": {"lad", "diagonal", "d1", "d2", "d3"},
    "LCx territory": {"lcx", "om", "marg", "lvp", "ramus", "bx"},
    "RCA territory": {"rca", "rvg", "pda", "posterolateral", "right_marginal"},
}

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
# Prompt Helper Functions
# ──────────────────────────────────────────────────────────────────────────

def categorize_stenosis_value(value: Optional[float]) -> Optional[Dict[str, Any]]:
    if value is None or pd.isna(value):
        return None
    value = float(value)
    for lower, upper, label, range_text in SEVERITY_BINS:
        if label == "critical":
            if value >= lower:
                return {
                    "severity": label,
                    "range_text": range_text,
                    "rank": SEVERITY_RANK.get(label, -1),
                }
        elif lower <= value < upper:
            if label == "none":
                return None
            return {
                "severity": label,
                "range_text": range_text,
                "rank": SEVERITY_RANK.get(label, -1),
            }
    return None


def humanize_vessel_label(label: str) -> str:
    base = label.replace("_stenosis", "").replace("leftmain", "left_main")
    tokens = base.split("_")
    words: List[str] = []
    for token in tokens:
        if not token:
            continue
        lower = token.lower()
        if lower in TOKEN_NAME_MAP:
            words.append(TOKEN_NAME_MAP[lower])
            continue
        if lower.startswith("om") and lower[2:].isdigit():
            words.append(lower.upper())
            continue
        if lower.startswith("d") and lower[1:].isdigit():
            words.append(lower.upper())
            continue
        if lower.startswith("rv") and lower[2:].isdigit():
            words.append(lower.upper())
            continue
        if lower == "bx":
            words.append("Ramus")
            continue
        if lower == "lvp":
            words.append("LVP branch")
            continue
        words.append(token.capitalize())
    cleaned = " ".join(words).strip()
    if not cleaned:
        return label
    return cleaned[0].upper() + cleaned[1:]


def _infer_territory(label: str, dominance_name: str) -> Optional[str]:
    label_base = label.replace("_stenosis", "").replace("leftmain", "left_main").lower()
    dominance_lower = str(dominance_name or "").lower()

    if "pda" in label_base:
        if "left" in dominance_lower:
            return "LCx territory"
        return "RCA territory"
    if "posterolateral" in label_base:
        if "left" in dominance_lower:
            return "LCx territory"
        return "RCA territory"

    for territory, tokens in NEGATIVE_TERRITORIES.items():
        for token in tokens:
            if token in label_base:
                return territory
    return None


def _format_name_value_pairs(lesions: List[Dict[str, Any]]) -> str:
    parts = [f"{lesion['name']} (≈{int(round(lesion['value']))}%)" for lesion in lesions]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + f" and {parts[-1]}"


def compute_study_context(study_df: pd.DataFrame) -> Dict[str, Any]:
    context: Dict[str, Any] = {}
    context["StudyInstanceUID"] = study_df["StudyInstanceUID"].iloc[0] if "StudyInstanceUID" in study_df.columns else None
    if "SeriesInstanceUID" in study_df.columns:
        series_ids = study_df["SeriesInstanceUID"].dropna().unique().tolist()
        context["SeriesInstanceUIDs"] = series_ids if series_ids else [None]
    else:
        context["SeriesInstanceUIDs"] = [None]

    if "dominance_name" in study_df.columns:
        dominance_series = study_df["dominance_name"].dropna()
        context["dominance_name"] = dominance_series.iloc[0] if not dominance_series.empty else ""
    else:
        context["dominance_name"] = ""

    report_series = study_df.get("Report")
    if report_series is not None:
        unique_reports = []
        seen = set()
        for report in report_series.dropna().tolist():
            normalized = report.strip()
            if normalized and normalized not in seen:
                unique_reports.append(normalized)
                seen.add(normalized)
        context["reports"] = unique_reports
    else:
        context["reports"] = []

    stenosis_cols = [col for col in study_df.columns if col.endswith("_stenosis")]
    stenosis_values: Dict[str, float] = {}
    lesions: List[Dict[str, Any]] = []

    for col in stenosis_cols:
        numeric = pd.to_numeric(study_df[col], errors='coerce').replace(-1, np.nan)
        if numeric.empty:
            continue
        value = numeric.max(skipna=True)
        if pd.isna(value):
            stenosis_values[col] = np.nan
            continue
        value = float(value)
        stenosis_values[col] = value
        meta = categorize_stenosis_value(value)
        if meta is None:
            continue
        lesions.append({
            "column": col,
            "name": humanize_vessel_label(col),
            "value": value,
            "range_text": meta["range_text"],
            "severity": meta["severity"],
            "severity_rank": meta["rank"],
            "descriptor": SEVERITY_DESCRIPTOR.get(meta["severity"], meta["severity"]),
        })

    lesions.sort(key=lambda item: (item["severity_rank"], item["value"]), reverse=True)
    context["stenosis_values"] = stenosis_values
    context["lesions"] = lesions
    return context


def build_global_prompt(context: Dict[str, Any]) -> Optional[str]:
    reports = context.get("reports", [])
    if reports:
        return "\n\n".join(reports)

    lesions = [lesion for lesion in context.get("lesions", []) if lesion["severity_rank"] >= 1]
    if lesions:
        summary_parts = [f"{lesion['name']} {int(round(lesion['value']))}% stenosis" for lesion in lesions]
        return "; ".join(summary_parts) + "; other segments not specified."

    return "No significant stenosis documented; other segments not specified."


def build_abnormal_prompt(context: Dict[str, Any]) -> str:
    lesions = context.get("lesions", [])
    severe_or_worse = [lesion for lesion in lesions if lesion["severity_rank"] >= 3]
    phrases: List[str] = []

    if severe_or_worse:
        severe_sorted = sorted(severe_or_worse, key=lambda item: item["value"], reverse=True)
        for idx, lesion in enumerate(severe_sorted):
            severity_word = "Critical" if lesion["severity"] == "critical" else "Severe"
            prefix = severity_word if idx == 0 else f"Additional {severity_word.lower()}"
            phrases.append(f"{prefix} {lesion['name']} (≈{int(round(lesion['value']))}%)")

        moderate = [lesion for lesion in lesions if lesion["severity_rank"] == 2]
        if moderate:
            phrases.append(f"Additional moderate disease in {_format_name_value_pairs(moderate)}")

        mild = [lesion for lesion in lesions if lesion["severity_rank"] == 1]
        if mild:
            phrases.append(f"Additional mild disease in {_format_name_value_pairs(mild)}")

        result = "; ".join(phrases)
        if result and not result.endswith('.'):
            result += '.'
        return result

    return "No ≥70% stenosis identified."


def build_atomic_prompts(context: Dict[str, Any], max_atomic_prompts: Optional[int] = None) -> List[str]:
    lesions = [lesion for lesion in context.get("lesions", []) if lesion["severity_rank"] >= 1]
    if not lesions:
        return []

    atomic_prompts: List[str] = []
    for lesion in lesions:
        prompt = f"{lesion['name']}; {lesion['range_text']} stenosis ({lesion['descriptor']})."
        atomic_prompts.append(prompt)
        if max_atomic_prompts is not None and len(atomic_prompts) >= max_atomic_prompts:
            break
    return atomic_prompts


def build_negative_prompt(context: Dict[str, Any]) -> Optional[str]:
    dominance = context.get("dominance_name", "")
    territory_values: Dict[str, List[float]] = defaultdict(list)

    for label, value in context.get("stenosis_values", {}).items():
        territory = _infer_territory(label, dominance)
        if territory is None or pd.isna(value):
            continue
        territory_values[territory].append(float(value))

    statements: List[str] = []
    for territory in ["Left main", "LAD territory", "LCx territory", "RCA territory"]:
        values = territory_values.get(territory, [])
        if not values:
            statements.append(f"{territory}: not specified.")
            continue
        max_value = max(values)
        if max_value >= 50:
            continue
        threshold_text = "≤30%" if max_value <= 30 else "≤49%"
        statements.append(f"{territory}: no ≥50% stenosis identified ({threshold_text}).")

    if not statements:
        return None

    return " ".join(statements)


def generate_siglip_prompt_rows(df: pd.DataFrame, max_atomic_prompts: Optional[int] = None) -> pd.DataFrame:
    if "StudyInstanceUID" not in df.columns:
        raise ValueError("StudyInstanceUID column is required to generate prompts")

    prompt_rows: List[Dict[str, Any]] = []

    for study_uid, study_df in df.groupby("StudyInstanceUID"):
        context = compute_study_context(study_df)
        series_ids = context.get("SeriesInstanceUIDs", [None])
        if not series_ids:
            series_ids = [None]

        prompts: List[Tuple[str, str, float]] = []

        global_prompt = build_global_prompt(context)
        if global_prompt:
            prompts.append((global_prompt, "global_summary", PROMPT_WEIGHTS["global_summary"]))

        abnormal_prompt = build_abnormal_prompt(context)
        if abnormal_prompt:
            prompts.append((abnormal_prompt, "abnormal_focus", PROMPT_WEIGHTS["abnormal_focus"]))

        atomic_prompts = build_atomic_prompts(context, max_atomic_prompts=max_atomic_prompts)
        for atomic_text in atomic_prompts:
            prompts.append((atomic_text, "lesion_atomic", PROMPT_WEIGHTS["lesion_atomic"]))

        negative_prompt = build_negative_prompt(context)
        if negative_prompt:
            prompts.append((negative_prompt, "negative_coverage", PROMPT_WEIGHTS["negative_coverage"]))

        for series_id in series_ids:
            for text, prompt_type, weight in prompts:
                prompt_rows.append({
                    "StudyInstanceUID": study_uid,
                    "SeriesInstanceUID": series_id,
                    "prompt_text": text,
                    "prompt_type": prompt_type,
                    "prompt_weight": weight,
                })

    if not prompt_rows:
        return pd.DataFrame(columns=["StudyInstanceUID", "SeriesInstanceUID", "prompt_text", "prompt_type", "prompt_weight"])

    return pd.DataFrame(prompt_rows)


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
        
        # ──────────────────────────────────────────────────────────────────────────
        # FILTER EXCESSIVE NORMAL REPORTS (per vessel system)
        # ──────────────────────────────────────────────────────────────────────────
        normal_report_ratio = config.get('filters', {}).get('normal_report_ratio', 0.05)  # Default to 5%
        logger.info(f"Using normal report ratio: {normal_report_ratio*100:.1f}%")
        
        # Identify normal reports based on main_structure_name
        if 'main_structure_name' in df.columns:
            # Define vessels for each coronary side
            rca_vessels_cols = ["prox_rca_stenosis", "mid_rca_stenosis", "dist_rca_stenosis",
                                "pda_stenosis", "posterolateral_stenosis"]
            lca_vessels_cols = ["left_main_stenosis", "prox_lad_stenosis", "mid_lad_stenosis", 
                               "dist_lad_stenosis", "D1_stenosis", "D2_stenosis", 
                               "prox_lcx_stenosis", "dist_lcx_stenosis", "om1_stenosis", 
                               "om2_stenosis", "bx_stenosis", "lvp_stenosis"]
            
            # Check which columns exist
            existing_rca_cols = [col for col in rca_vessels_cols if col in df.columns]
            existing_lca_cols = [col for col in lca_vessels_cols if col in df.columns]
            
            # Process RCA reports
            indices_to_remove = pd.Index([])
            
            if existing_rca_cols:
                rca_mask = df['main_structure_name'] == 'Right Coronary'
                rca_df = df[rca_mask].copy()
                
                # Check if all RCA vessels have stenosis = 0 (accounting for NaN and -1)
                rca_stenosis_vals = rca_df[existing_rca_cols].fillna(-1)
                is_normal_rca = ((rca_stenosis_vals == 0) | (rca_stenosis_vals == -1)).all(axis=1) & \
                               (rca_stenosis_vals == 0).any(axis=1)  # At least one 0, rest 0 or -1
                
                normal_rca_indices = rca_df[is_normal_rca].index
                abnormal_rca_indices = rca_df[~is_normal_rca].index
                
                logger.info(f"RCA: Found {len(normal_rca_indices)} normal reports out of {len(rca_df)} ({len(normal_rca_indices)/len(rca_df)*100:.1f}%)")
                
                # Calculate target for RCA (5% of RCA reports)
                target_normal_rca = int(len(rca_df) * normal_report_ratio)
                
                if len(normal_rca_indices) > target_normal_rca:
                    logger.info(f"  Reducing RCA normal reports from {len(normal_rca_indices)} to {target_normal_rca} ({normal_report_ratio*100:.0f}% of RCA)")
                    
                    # Randomly select which normal RCA reports to remove
                    np.random.seed(42)
                    rca_to_remove = np.random.choice(normal_rca_indices, 
                                                    size=len(normal_rca_indices) - target_normal_rca, 
                                                    replace=False)
                    indices_to_remove = indices_to_remove.union(pd.Index(rca_to_remove))
            
            # Process LCA reports
            if existing_lca_cols:
                lca_mask = df['main_structure_name'] == 'Left Coronary'
                lca_df = df[lca_mask].copy()
                
                # Check if all LCA vessels have stenosis = 0 (accounting for NaN and -1)
                lca_stenosis_vals = lca_df[existing_lca_cols].fillna(-1)
                is_normal_lca = ((lca_stenosis_vals == 0) | (lca_stenosis_vals == -1)).all(axis=1) & \
                               (lca_stenosis_vals == 0).any(axis=1)  # At least one 0, rest 0 or -1
                
                normal_lca_indices = lca_df[is_normal_lca].index
                abnormal_lca_indices = lca_df[~is_normal_lca].index
                
                logger.info(f"LCA: Found {len(normal_lca_indices)} normal reports out of {len(lca_df)} ({len(normal_lca_indices)/len(lca_df)*100:.1f}%)")
                
                # Calculate target for LCA (5% of LCA reports)
                target_normal_lca = int(len(lca_df) * normal_report_ratio)
                
                if len(normal_lca_indices) > target_normal_lca:
                    logger.info(f"  Reducing LCA normal reports from {len(normal_lca_indices)} to {target_normal_lca} ({normal_report_ratio*100:.0f}% of LCA)")
                    
                    # Randomly select which normal LCA reports to remove
                    np.random.seed(43)  # Different seed for LCA
                    lca_to_remove = np.random.choice(normal_lca_indices, 
                                                    size=len(normal_lca_indices) - target_normal_lca, 
                                                    replace=False)
                    indices_to_remove = indices_to_remove.union(pd.Index(lca_to_remove))
            
            # Remove the selected indices
            if len(indices_to_remove) > 0:
                df = df.drop(indices_to_remove)
                logger.info(f"Removed {len(indices_to_remove)} normal reports")
                logger.info(f"After normal report filtering: {len(df)} rows remaining")
                
                # Log final distribution
                if existing_rca_cols:
                    rca_remaining = df[df['main_structure_name'] == 'Right Coronary']
                    rca_stenosis_vals = rca_remaining[existing_rca_cols].fillna(-1)
                    is_normal_rca_final = ((rca_stenosis_vals == 0) | (rca_stenosis_vals == -1)).all(axis=1) & \
                                         (rca_stenosis_vals == 0).any(axis=1)
                    logger.info(f"  Final RCA: {is_normal_rca_final.sum()} normal / {len(rca_remaining)} total ({is_normal_rca_final.sum()/len(rca_remaining)*100:.1f}%)")
                
                if existing_lca_cols:
                    lca_remaining = df[df['main_structure_name'] == 'Left Coronary']
                    lca_stenosis_vals = lca_remaining[existing_lca_cols].fillna(-1)
                    is_normal_lca_final = ((lca_stenosis_vals == 0) | (lca_stenosis_vals == -1)).all(axis=1) & \
                                         (lca_stenosis_vals == 0).any(axis=1)
                    logger.info(f"  Final LCA: {is_normal_lca_final.sum()} normal / {len(lca_remaining)} total ({is_normal_lca_final.sum()/len(lca_remaining)*100:.1f}%)")
        else:
            logger.warning("main_structure_name column not found. Skipping normal report filtering.")
    
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
        separator = config.get('output_settings', {}).get('separator', 'α')
        
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

    # Generate SIGLIP-ready prompt dataset if requested
    siglip_cfg = config.get('siglip_prompts', {})
    if siglip_cfg.get('enabled', False):
        prompt_df = generate_siglip_prompt_rows(
            df_with_reports,
            max_atomic_prompts=siglip_cfg.get('max_atomic_prompts')
        )

        prompt_output_name = siglip_cfg.get('output_filename', 'siglip_prompts_sample.parquet')
        prompt_output_path = output_path / prompt_output_name
        prompt_df.to_parquet(prompt_output_path, index=False)
        logger.info(f"Saved SIGLIP prompt dataset to {prompt_output_path} ({len(prompt_df)} rows)")

        sample_rows = siglip_cfg.get('print_rows', 5)
        if sample_rows and len(prompt_df) > 0:
            sample_preview = prompt_df.head(int(sample_rows))
            print("SIGLIP prompt sample:")
            print(sample_preview)

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
            'contrast_agent_class': 1,
            'normal_report_ratio': 0.05  # Limit normal reports to 5% per vessel type (0.05 = 5%, 0.1 = 10%, etc.)
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
        'siglip_prompts': {
            'enabled': False,
            'max_atomic_prompts': None,
            'output_filename': 'siglip_prompts_sample.parquet',
            'print_rows': 5
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
        '--normal-report-ratio',
        type=float,
        help='Percentage of normal reports to keep (0.05 = 5%%, 0.1 = 10%%, etc.)'
    )
    
    args = parser.parse_args()
    
    if args.create_default_config:
        config = create_default_config()
        with open('default_config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print("Created default_config.yaml")
        return
    
    # Validate required arguments for normal operation
    if not args.input_csv or not args.output_dir:
        parser.error("--input-csv and --output-dir are required when not creating default config")
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        logger.info("No config provided, using default configuration")
        config = create_default_config()
    
    # Override normal report ratio if provided via command line
    if args.normal_report_ratio is not None:
        if 'filters' not in config:
            config['filters'] = {}
        config['filters']['normal_report_ratio'] = args.normal_report_ratio
        logger.info(f"Overriding normal report ratio to {args.normal_report_ratio*100:.1f}% via command line")
    
    # Process dataset
    process_dataset(args.input_csv, args.output_dir, config)


if __name__ == "__main__":
    main() 
