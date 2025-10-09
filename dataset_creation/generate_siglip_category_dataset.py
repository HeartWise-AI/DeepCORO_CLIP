#!/usr/bin/env python3
"""
SigLIP Multi-Prompt Dataset Generation

Generates two files for SigLIP training with proper prompt_type separation:
1. videos.csv - Video metadata with Split and FileName
2. texts.csv - Multi-style prompts (lesion_atomic, abnormal_focus, negative_coverage, global_summary)

Implements correct SigLIP paradigm:
- prompt_type = text structure/training family
- tags = canonical medical metadata (category|segment|bin|tree|stent|calc|medina)
- soft_weight = loss weighting by prompt_type
"""

import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Set
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import numpy as np
from tqdm import tqdm
import yaml
import logging
import matplotlib.pyplot as plt

# Import from existing scripts
from generate_dataset import (
    load_data,
    apply_hard_filters,
    assign_procedure_status,
    assign_patient_splits,
    add_angiographic_view_column,
    format_main_structure_description,
    MAIN_STRUCTURE_MAP,
    DOMINANCE_MAP,
    RCA_VESSELS,
    NON_RCA_VESSELS,
    RIGHT_DOMINANCE_DEPENDENT_VESSELS,
    LEFT_DOMINANCE_DEPENDENT_VESSELS
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────
# Vessel name mappings
# ──────────────────────────────────────────────────────────────────────────

VESSEL_DISPLAY_NAMES = {
    "left_main": "Left Main",
    "prox_lad": "Proximal LAD",
    "mid_lad": "Mid LAD",
    "dist_lad": "Distal LAD",
    "D1": "D1",
    "D2": "D2",
    "lcx": "Proximal LCx",
    "prox_lcx": "Proximal LCx",
    "dist_lcx": "Distal LCx",
    "om1": "OM1",
    "om2": "OM2",
    "prox_rca": "Proximal RCA",
    "mid_rca": "Mid RCA",
    "dist_rca": "Distal RCA",
    "pda": "PDA",
    "posterolateral": "Posterolateral",
    "bx": "Ramus",
    "lvp": "LVP branch"
}

# Prompt type weights for SigLIP loss (by text structure/style)
PROMPT_TYPE_WEIGHTS = {
    "lesion_atomic": 1.0,      # One finding per sentence
    "abnormal_focus": 1.0,     # Only ≥70% with clinical details
    "negative_coverage": 0.6,  # Territory-level negatives
    "global_summary": 0.5      # Study-level roll-up
}

# Canonical segment names (lowercase, no spaces)
CANONICAL_SEGMENTS = {
    "left_main_stenosis": "left_main",
    "prox_lad_stenosis": "prox_lad",
    "mid_lad_stenosis": "mid_lad",
    "dist_lad_stenosis": "dist_lad",
    "D1_stenosis": "d1",
    "D2_stenosis": "d2",
    "lcx_stenosis": "prox_lcx",
    "dist_lcx_stenosis": "dist_lcx",
    "om1_stenosis": "om1",
    "om2_stenosis": "om2",
    "prox_rca_stenosis": "prox_rca",
    "mid_rca_stenosis": "mid_rca",
    "dist_rca_stenosis": "dist_rca",
    "pda_stenosis": "pda",
    "posterolateral_stenosis": "posterolateral",
    "bx_stenosis": "ramus",
    "lvp_stenosis": "lvp"
}

# ──────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────

def get_vessel_name(vessel_col: str) -> str:
    """Convert column name to display name."""
    base = vessel_col.replace("_stenosis", "")
    return VESSEL_DISPLAY_NAMES.get(base, base.replace("_", " ").title())


def get_canonical_segment(vessel_col: str) -> str:
    """Get canonical lowercase segment name."""
    return CANONICAL_SEGMENTS.get(vessel_col, vessel_col.replace("_stenosis", "").lower())


def get_tree(main_structure: str) -> str:
    """Get coronary tree (left|right)."""
    if main_structure == "Left Coronary":
        return "left"
    elif main_structure == "Right Coronary":
        return "right"
    return "unknown"


def parse_tags(tag_str: str) -> Dict[str, str]:
    """Parse pipe-separated canonical tag string into a dictionary."""
    tags: Dict[str, str] = {}
    if not isinstance(tag_str, str):
        return tags

    for kv in tag_str.split("|"):
        if ":" not in kv:
            continue
        key, value = kv.split(":", 1)
        tags[key] = value

    return tags


def extract_tree_from_tags(tag_str: str) -> Optional[str]:
    """Return left/right tree hint from canonical tag string."""
    if not isinstance(tag_str, str):
        return None
    for kv in tag_str.split("|"):
        if kv.startswith("tree:"):
            tree = kv.split(":", 1)[1].strip().lower()
            if tree in {"left", "right"}:
                return tree
    return None


def build_canonical_tags(**kwargs) -> str:
    """
    Build canonical tag string from key-value pairs.

    Args:
        category: Medical category (stenosis, in_stent, calcification, cto, medina, thrombus)
        segment: Canonical segment name (prox_lad, mid_rca, etc.)
        bin: Stenosis bin (<30, 30-49, 50-69, 70-89, >=90)
        tree: Coronary tree (left, right)
        stent: Stent present (y, n)
        calc: Calcification level (minimal, moderate, severe)
        medina: Medina code (e.g., "1.1.0")
        thrombus: Thrombus present (y, n)
        territory: Territory name for negative coverage

    Returns:
        Canonical tag string, e.g., "category:stenosis|segment:mid_lad|bin:70-89|tree:left"
    """
    tags = []
    order = ['category', 'segment', 'territory', 'bin', 'tree', 'stent', 'calc', 'medina', 'thrombus']

    for key in order:
        if key in kwargs and kwargs[key] is not None:
            value = str(kwargs[key]).lower().replace(" ", "_")
            tags.append(f"{key}:{value}")

    return "|".join(tags)


def discretize_stenosis(value: float) -> str:
    """Discretize stenosis into bins."""
    if value < 30:
        return "<30"
    elif value < 50:
        return "30-49"
    elif value < 70:
        return "50-69"
    elif value < 90:
        return "70-89"
    else:
        return ">=90"


def _collect_prompts_for_study(study_id: str, study_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Generate raw prompt records for a single study."""
    raw_records: List[Dict[str, Any]] = []

    left_df = study_df[study_df["main_structure_name"] == "Left Coronary"]
    right_df = study_df[study_df["main_structure_name"] == "Right Coronary"]

    if len(left_df) > 0:
        global_prompt = generate_global_summary_prompts(left_df, "Left Coronary")
        if global_prompt:
            raw_records.append({
                "study_id": study_id,
                "video_id": None,
                "prompt_type": global_prompt["prompt_type"],
                "prompt_text": global_prompt["prompt_text"],
                "tags": global_prompt["tags"],
                "soft_weight": global_prompt["soft_weight"],
                "main_structure": "Left Coronary"
            })

    if len(right_df) > 0:
        global_prompt = generate_global_summary_prompts(right_df, "Right Coronary")
        if global_prompt:
            raw_records.append({
                "study_id": study_id,
                "video_id": None,
                "prompt_type": global_prompt["prompt_type"],
                "prompt_text": global_prompt["prompt_text"],
                "tags": global_prompt["tags"],
                "soft_weight": global_prompt["soft_weight"],
                "main_structure": "Right Coronary"
            })

    for _, row in study_df.iterrows():
        main_structure = row.get("main_structure_name", "")
        dominance = row.get("dominance_name", "")
        video_id = row.get("video_id")

        relevant_vessels = get_relevant_vessels(main_structure, dominance)
        if not relevant_vessels:
            continue

        prompts: List[Dict[str, Any]] = []
        prompts.extend(generate_lesion_atomic_prompts(row, relevant_vessels, main_structure))
        prompts.extend(generate_abnormal_focus_prompts(row, relevant_vessels, main_structure))
        prompts.extend(generate_negative_coverage_prompts(row, relevant_vessels, main_structure))

        if not prompts:
            fallback_prompt = create_system_normal_prompt(main_structure)
            if fallback_prompt is not None:
                prompts.append(fallback_prompt)

        for prompt in prompts:
            raw_records.append({
                "study_id": study_id,
                "video_id": video_id,
                "prompt_type": prompt["prompt_type"],
                "prompt_text": prompt["prompt_text"],
                "tags": prompt["tags"],
                "soft_weight": prompt["soft_weight"],
                "main_structure": main_structure
            })

    return raw_records


def _process_study_parallel(args: Tuple[int, str, List[str], List[Tuple[Any, ...]]]) -> Tuple[int, List[Dict[str, Any]]]:
    """Worker helper to rebuild DataFrame and collect prompts for a study."""
    order_idx, study_id, columns, data = args
    study_df = pd.DataFrame.from_records(data, columns=columns)
    records = _collect_prompts_for_study(study_id, study_df)
    return order_idx, records


def normalize_calcification(calc_str: str) -> Optional[str]:
    """Normalize calcification to none|minimal|moderate|severe."""
    if not calc_str or pd.isna(calc_str):
        return None

    calc_lower = str(calc_str).lower()

    if "no calcif" in calc_lower or "pas de calcif" in calc_lower:
        return "none"
    elif "minime" in calc_lower or "mild" in calc_lower:
        return "minimal"
    elif "modérée" in calc_lower or "moderate" in calc_lower:
        return "moderate"
    elif "importante" in calc_lower or "severe" in calc_lower:
        return "severe"

    return None


def get_relevant_vessels(main_structure: str, dominance: str) -> List[str]:
    """Get vessel columns relevant to the video's main structure."""
    dominance_lower = str(dominance).lower()

    if main_structure == "Right Coronary":
        vessels = RCA_VESSELS.copy()
        if "right" in dominance_lower:
            vessels.extend(RIGHT_DOMINANCE_DEPENDENT_VESSELS)
        return vessels

    elif main_structure == "Left Coronary":
        vessels = NON_RCA_VESSELS.copy()
        if "left" in dominance_lower:
            vessels.extend(LEFT_DOMINANCE_DEPENDENT_VESSELS)
        return vessels

    return []


# ──────────────────────────────────────────────────────────────────────────
# Vessel details extraction
# ──────────────────────────────────────────────────────────────────────────

def get_vessel_details(row: pd.Series, vessel_col: str) -> Dict[str, Any]:
    """Extract all medical details for a vessel."""
    prefix = vessel_col.replace("_stenosis", "")
    details = {
        "vessel_col": vessel_col,
        "segment": get_canonical_segment(vessel_col),
        "display_name": get_vessel_name(vessel_col),
        "stenosis": None,
        "is_cto": False,
        "has_stent": False,
        "in_stent_restenosis": False,
        "calcification": None,
        "ifr": None,
        "medina": None,
        "thrombus": False
    }

    # Check CTO
    cto_col = prefix + "_cto"
    if cto_col in row.index:
        is_cto = row[cto_col]
        if pd.notna(is_cto) and is_cto == 1:
            details["is_cto"] = True
            details["stenosis"] = 100.0
            return details

    # Stenosis value
    if vessel_col in row.index:
        st = row[vessel_col]
        if pd.notna(st) and st >= 0:
            details["stenosis"] = float(st)

            # Stent check
            stent_col = prefix + "_stent"
            if stent_col in row.index:
                has_stent = row[stent_col]
                if pd.notna(has_stent) and has_stent > 0:
                    details["has_stent"] = True
                    if details["stenosis"] > 10:
                        details["in_stent_restenosis"] = True

    # Calcification
    calc_col = prefix + "_calcif"
    if calc_col in row.index:
        calc = row[calc_col]
        calc_norm = normalize_calcification(calc)
        if calc_norm and calc_norm != "none":
            details["calcification"] = calc_norm

    # IFR
    ifr_col = prefix + "_IFRHYPEREMIE"
    if ifr_col in row.index:
        ifr = row[ifr_col]
        if pd.notna(ifr) and ifr != -1:
            details["ifr"] = float(ifr)

    # Medina bifurcation
    bif_col = prefix + "_bifurcation"
    if bif_col in row.index:
        bif = row[bif_col]
        if pd.notna(bif) and bif not in [0, 0.0, "", "0", "0.0", "nan"]:
            if isinstance(bif, str):
                bif_str = bif.strip()
                if bif_str.lower() not in ["", "pas de lésion de bifurcation", "nan"]:
                    details["medina"] = bif_str
            elif isinstance(bif, (int, float)) and bif != 0:
                details["medina"] = str(bif)

    # Thrombus
    thrombus_col = prefix + "_thrombus"
    if thrombus_col in row.index:
        has_thrombus = row[thrombus_col]
        if pd.notna(has_thrombus) and has_thrombus == 1:
            details["thrombus"] = True

    return details


# ──────────────────────────────────────────────────────────────────────────
# Multi-prompt text generation (lesion_atomic, abnormal_focus, negative_coverage, global_summary)
# ──────────────────────────────────────────────────────────────────────────

def generate_lesion_atomic_prompts(row: pd.Series, relevant_vessels: List[str], main_structure: str) -> List[Dict[str, Any]]:
    """
    Generate lesion_atomic prompts: one finding per sentence.
    Format: "Mid LAD; 50-69% stenosis (moderate)."
    """
    prompts = []
    tree = get_tree(main_structure)

    for vessel_col in relevant_vessels:
        details = get_vessel_details(row, vessel_col)

        if details["stenosis"] is None:
            continue

        stenosis_value = float(details["stenosis"])
        is_zero_stenosis = abs(stenosis_value) < 1e-3
        vessel_name = details["display_name"]
        segment = details["segment"]

        # Explicit normal case (no disease and no hardware)
        if (
            is_zero_stenosis
            and not details["has_stent"]
            and not details["is_cto"]
        ):
            normal_text = f"{vessel_name}; no angiographic stenosis (normal)."
            normal_tags = build_canonical_tags(
                category="normal",
                segment=segment,
                tree=tree
            )
            prompts.append({
                "prompt_text": normal_text,
                "prompt_type": "lesion_atomic",
                "tags": normal_tags,
                "soft_weight": PROMPT_TYPE_WEIGHTS["lesion_atomic"]
            })
            continue

        bin_label = discretize_stenosis(stenosis_value)

        # Build prompt text
        if details["is_cto"]:
            text = f"{vessel_name}; chronic total occlusion (CTO)."
            category = "cto"
            tags = build_canonical_tags(category="cto", segment=segment, tree=tree)

        elif details["in_stent_restenosis"]:
            text = f"{vessel_name}; {bin_label}% in-stent restenosis"
            severity = "minimal" if bin_label == "<30" else "mild" if bin_label == "30-49" else "moderate" if bin_label == "50-69" else "severe" if bin_label == "70-89" else "critical"
            text += f" ({severity})."
            category = "in_stent"
            tags = build_canonical_tags(category="in_stent", segment=segment, bin=bin_label, tree=tree, stent="y")

        else:
            text = f"{vessel_name}; {bin_label}% stenosis"
            severity = "minimal" if bin_label == "<30" else "mild" if bin_label == "30-49" else "moderate" if bin_label == "50-69" else "severe" if bin_label == "70-89" else "critical"
            text += f" ({severity})."
            category = "stenosis"
            tags = build_canonical_tags(
                category="stenosis",
                segment=segment,
                bin=bin_label,
                tree=tree,
                stent="y" if details["has_stent"] else None
            )

        prompts.append({
            "prompt_text": text,
            "prompt_type": "lesion_atomic",
            "tags": tags,
            "soft_weight": PROMPT_TYPE_WEIGHTS["lesion_atomic"]
        })

        # Add calcification as separate atomic prompt if present
        if details["calcification"]:
            calc_text = f"{vessel_name}; {details['calcification']} calcification."
            calc_tags = build_canonical_tags(category="calcification", segment=segment, calc=details["calcification"], tree=tree)
            prompts.append({
                "prompt_text": calc_text,
                "prompt_type": "lesion_atomic",
                "tags": calc_tags,
                "soft_weight": PROMPT_TYPE_WEIGHTS["lesion_atomic"]
            })

        # Add Medina as separate atomic prompt if present
        if details["medina"]:
            medina_text = f"{vessel_name}; bifurcation lesion (Medina {details['medina']})."
            medina_tags = build_canonical_tags(category="medina", segment=segment, medina=details["medina"], tree=tree)
            prompts.append({
                "prompt_text": medina_text,
                "prompt_type": "lesion_atomic",
                "tags": medina_tags,
                "soft_weight": PROMPT_TYPE_WEIGHTS["lesion_atomic"]
            })

        # Add thrombus as separate atomic prompt if present
        if details["thrombus"]:
            thrombus_text = f"{vessel_name}; thrombus present."
            thrombus_tags = build_canonical_tags(category="thrombus", segment=segment, thrombus="y", tree=tree)
            prompts.append({
                "prompt_text": thrombus_text,
                "prompt_type": "lesion_atomic",
                "tags": thrombus_tags,
                "soft_weight": PROMPT_TYPE_WEIGHTS["lesion_atomic"]
            })

    return prompts


def generate_abnormal_focus_prompts(row: pd.Series, relevant_vessels: List[str], main_structure: str) -> List[Dict[str, Any]]:
    """
    Generate abnormal_focus prompts: only ≥70% stenosis with clinical details.
    Format: "Critical OM1 (≈90%) [in-stent restenosis, moderate calcification]."
    """
    prompts = []
    tree = get_tree(main_structure)

    for vessel_col in relevant_vessels:
        details = get_vessel_details(row, vessel_col)

        # Only include ≥70% or CTO
        if not details["is_cto"] and (details["stenosis"] is None or details["stenosis"] < 70):
            continue

        vessel_name = details["display_name"]
        segment = details["segment"]

        if details["is_cto"]:
            text = f"CTO in {vessel_name}"
            tags = build_canonical_tags(category="cto", segment=segment, tree=tree)

        else:
            bin_label = discretize_stenosis(details["stenosis"])
            severity = "Severe" if details["stenosis"] < 90 else "Critical"
            text = f"{severity} {vessel_name} (≈{int(details['stenosis'])}%)"

            # Add clinical details in brackets
            extras = []
            if details["in_stent_restenosis"]:
                extras.append("in-stent restenosis")
            elif details["has_stent"]:
                extras.append("stented")

            if details["calcification"] and details["calcification"] != "none":
                extras.append(f"{details['calcification']} calcification")

            if details["ifr"] is not None and details["ifr"] <= 0.89:
                extras.append(f"IFR {details['ifr']:.2f}")

            if details["medina"]:
                extras.append(f"Medina {details['medina']}")

            if details["thrombus"]:
                extras.append("thrombus")

            if extras:
                text += " [" + ", ".join(extras) + "]"

            text += "."

            category = "in_stent" if details["in_stent_restenosis"] else "stenosis"
            tags = build_canonical_tags(
                category=category,
                segment=segment,
                bin=bin_label,
                tree=tree,
                stent="y" if details["has_stent"] else None,
                calc=details["calcification"] if details["calcification"] else None,
                medina=details["medina"] if details["medina"] else None,
                thrombus="y" if details["thrombus"] else None
            )

        prompts.append({
            "prompt_text": text,
            "prompt_type": "abnormal_focus",
            "tags": tags,
            "soft_weight": PROMPT_TYPE_WEIGHTS["abnormal_focus"]
        })

    return prompts


def generate_negative_coverage_prompts(row: pd.Series, relevant_vessels: List[str], main_structure: str) -> List[Dict[str, Any]]:
    """
    Generate negative_coverage prompts: territory-level negatives.
    Format: "RCA territory: all lesions <30%." or "Left coronary: other territories <30%."
    """
    prompts = []
    tree = get_tree(main_structure)

    # Group vessels by territory
    if main_structure == "Right Coronary":
        territories = {"rca_territory": relevant_vessels}
    elif main_structure == "Left Coronary":
        lad_vessels = [v for v in relevant_vessels if "lad" in v.lower() or v.startswith("D")]
        lcx_vessels = [v for v in relevant_vessels if "lcx" in v.lower() or "om" in v.lower() or "bx" in v or "lvp" in v]
        lm_vessels = [v for v in relevant_vessels if "left_main" in v]

        territories = {}
        if lm_vessels:
            territories["left_main"] = lm_vessels
        if lad_vessels:
            territories["lad_territory"] = lad_vessels
        if lcx_vessels:
            territories["lcx_territory"] = lcx_vessels
    else:
        return prompts

    # Analyze each territory
    for territory_key, vessels in territories.items():
        max_stenosis = -1
        has_data = False

        for vessel_col in vessels:
            if vessel_col in row.index:
                stenosis = row[vessel_col]
                if pd.notna(stenosis) and stenosis >= 0:
                    has_data = True
                    max_stenosis = max(max_stenosis, stenosis)

        # Generate negative statement if max stenosis < 30%
        if has_data:
            territory_display = territory_key.replace("_", " ").title()

            if abs(max_stenosis) < 1e-3:
                text = f"{territory_display}: no angiographic stenosis detected."
                tags = build_canonical_tags(
                    category="normal",
                    territory=territory_key,
                    bin="0",
                    tree=tree
                )
            elif max_stenosis < 30:
                text = f"{territory_display}: all lesions <30%."
                tags = build_canonical_tags(
                    category="stenosis",
                    territory=territory_key,
                    bin="<30",
                    tree=tree
                )
            else:
                continue

            prompts.append({
                "prompt_text": text,
                "prompt_type": "negative_coverage",
                "tags": tags,
                "soft_weight": PROMPT_TYPE_WEIGHTS["negative_coverage"]
            })

    return prompts


def create_system_normal_prompt(main_structure: str) -> Optional[Dict[str, Any]]:
    """Create a fallback prompt stating that the coronary system is angiographically normal."""
    tree = get_tree(main_structure)
    if tree not in {"left", "right"}:
        return None

    system = "Left Coronary" if tree == "left" else "Right Coronary"
    text = f"{system}: no angiographic stenosis."
    tags = build_canonical_tags(category="normal", tree=tree)
    return {
        "prompt_text": text,
        "prompt_type": "negative_coverage",
        "tags": tags,
        "soft_weight": PROMPT_TYPE_WEIGHTS["negative_coverage"],
    }


def generate_global_summary_prompts(study_df: pd.DataFrame, main_structure: str) -> Optional[Dict[str, Any]]:
    """
    Generate global_summary prompt: study-level compact roll-up.
    Format: "Left system: Mid LAD 50-69% stenosis; OM1 ≥90% ISR; other segments not specified."
    """
    tree = get_tree(main_structure)
    system = "Right system" if tree == "right" else "Left system" if tree == "left" else "Unknown system"

    # Collect all significant findings across all videos in study
    findings = []

    for _, row in study_df.iterrows():
        relevant_vessels = get_relevant_vessels(row.get("main_structure_name", ""), row.get("dominance_name", ""))

        for vessel_col in relevant_vessels:
            details = get_vessel_details(row, vessel_col)

            if details["stenosis"] is None or details["stenosis"] < 30:
                continue

            vessel_name = details["display_name"]
            bin_label = discretize_stenosis(details["stenosis"])

            if details["is_cto"]:
                findings.append((details["stenosis"], f"{vessel_name} CTO"))
            elif details["in_stent_restenosis"]:
                findings.append((details["stenosis"], f"{vessel_name} {bin_label}% ISR"))
            else:
                findings.append((details["stenosis"], f"{vessel_name} {bin_label}% stenosis"))

    if not findings:
        text = f"{system}: No significant stenosis documented; segments not specified."
    else:
        findings.sort(key=lambda x: x[0], reverse=True)
        unique_descriptions: List[str] = []
        seen_descriptions: set[str] = set()
        for _, desc in findings:
            if desc in seen_descriptions:
                continue
            seen_descriptions.add(desc)
            unique_descriptions.append(desc)
            if len(unique_descriptions) >= 5:
                break

        if not unique_descriptions:
            text = f"{system}: No significant stenosis documented; segments not specified."
        else:
            text = (
                f"{system}: "
                + "; ".join(unique_descriptions)
                + "; other segments not specified."
            )

    tags = build_canonical_tags(category="summary", tree=tree)

    return {
        "prompt_text": text,
        "prompt_type": "global_summary",
        "tags": tags,
        "soft_weight": PROMPT_TYPE_WEIGHTS["global_summary"]
    }


# ──────────────────────────────────────────────────────────────────────────
# Main generation functions
# ──────────────────────────────────────────────────────────────────────────

def generate_multi_prompt_texts(df: pd.DataFrame, parallel_workers: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate multi-style text prompts for SigLIP training.

    Returns
    -------
    texts_df: pd.DataFrame
        Unique prompts with assigned text_id.
    raw_prompts_df: pd.DataFrame
        Non-deduplicated prompts with `video_id` (when applicable) and resolved `text_id`.
    """
    raw_records: List[Dict[str, Any]] = []

    grouped = list(df.groupby("StudyInstanceUID", sort=False))

    if parallel_workers and parallel_workers > 1 and len(grouped) > 1:
        logger.info(
            f"Parallel text generation enabled with {parallel_workers} workers"
        )
        futures = {}
        results: Dict[int, List[Dict[str, Any]]] = {}
        with ProcessPoolExecutor(max_workers=parallel_workers) as executor:
            for order_idx, (study_id, study_df) in enumerate(grouped):
                columns = list(study_df.columns)
                data = [tuple(row) for row in study_df.itertuples(index=False, name=None)]
                future = executor.submit(
                    _process_study_parallel,
                    (order_idx, study_id, columns, data)
                )
                futures[future] = order_idx

            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing studies", leave=False):
                order_idx, records = future.result()
                results[order_idx] = records

        for order_idx in range(len(grouped)):
            raw_records.extend(results.get(order_idx, []))
    else:
        for study_id, study_df in tqdm(grouped, desc="Generating multi-prompt texts"):
            raw_records.extend(_collect_prompts_for_study(study_id, study_df))

    raw_prompts_df = pd.DataFrame(raw_records, columns=[
        "study_id",
        "video_id",
        "prompt_type",
        "prompt_text",
        "tags",
        "soft_weight",
        "main_structure"
    ])

    if raw_prompts_df.empty:
        logger.warning("No text prompts generated!")
        empty_cols = ["text_id", "prompt_type", "prompt_text", "tags", "soft_weight"]
        return pd.DataFrame(columns=empty_cols), raw_prompts_df

    # Deduplicate unique prompts
    raw_prompts_df["soft_weight"] = raw_prompts_df["soft_weight"].astype(float)
    raw_prompts_df["dedup_key"] = (
        raw_prompts_df["prompt_type"].astype(str)
        + "||" + raw_prompts_df["prompt_text"].astype(str)
        + "||" + raw_prompts_df["tags"].astype(str)
    )

    texts_df = (
        raw_prompts_df
        .drop_duplicates(subset=["dedup_key"], keep="first")
        .reset_index(drop=True)
    )
    texts_df["text_id"] = [f"T{i:06d}" for i in range(len(texts_df))]

    # Attach final text_ids back to raw prompts
    raw_prompts_df = raw_prompts_df.merge(
        texts_df[["dedup_key", "text_id"]],
        on="dedup_key",
        how="left"
    )

    # Prepare final outputs
    texts_out = texts_df[[
        "text_id",
        "prompt_type",
        "prompt_text",
        "tags",
        "soft_weight"
    ]].copy()

    raw_prompts_out = raw_prompts_df[[
        "study_id",
        "video_id",
        "prompt_type",
        "prompt_text",
        "tags",
        "soft_weight",
        "main_structure",
        "text_id"
    ]].copy()

    return texts_out, raw_prompts_out


def generate_videos_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Generate the normalized videos.csv inventory."""
    videos = []

    for idx, row in df.iterrows():
        video_id = row.get("video_id")
        if not isinstance(video_id, str) or not video_id:
            video_id = f"V{idx:06d}"
        study_id = row.get("StudyInstanceUID", f"S{idx:06d}")
        main_structure = row.get("main_structure_name", "")
        view_name = row.get("angiographic_view_description", "Other")
        filename = row.get("FileName", "")
        split = row.get("Split", "train")

        videos.append({
            "video_id": video_id,
            "study_id": study_id,
            "main_structure": main_structure,
            "view_name": view_name,
            "FileName": filename,
            "Split": split,
            "split": split,
            "status": row.get("status", "")
        })

    return pd.DataFrame(videos)


def build_edges(
    raw_prompts_df: pd.DataFrame,
    videos_df: pd.DataFrame,
    texts_df: pd.DataFrame,
    cap_per_video: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Construct video→text positive supervision edges and a debug roll-up."""

    if raw_prompts_df is None or raw_prompts_df.empty:
        empty_edges = pd.DataFrame(columns=["video_id", "text_id", "weight"])
        videos_debug = videos_df.copy()
        videos_debug["positive_text_ids"] = ""
        stats = {
            "total_videos": len(videos_df),
            "videos_with_positives": 0,
            "positive_ratio": 0.0,
            "mean_pos_per_positive_video": 0.0,
            "median_pos_per_positive_video": 0.0,
            "max_pos_per_video": 0,
            "total_edges": 0,
            "prompt_type_counts": {},
            "category_counts": {}
        }
        return empty_edges, videos_debug, stats

    fallback_records = (
        raw_prompts_df[raw_prompts_df["video_id"].notna()]
        .copy()
    )
    fallback_records["tag_dict"] = fallback_records["tags"].apply(parse_tags)
    fallback_records["tree"] = fallback_records["tag_dict"].apply(lambda d: d.get("tree"))

    def fallback_priority(row: pd.Series) -> int:
        prompt_type = row.get("prompt_type")
        tag_dict = row.get("tag_dict") or {}
        category = tag_dict.get("category")
        bin_val = tag_dict.get("bin")
        text_lower = str(row.get("prompt_text", "")).lower()

        if prompt_type == "lesion_atomic":
            if category == "stenosis" and bin_val == "<30":
                return 0  # mild disease
            if category == "normal":
                return 1  # angiographically normal
        if prompt_type == "negative_coverage" and "no angiographic stenosis" in text_lower:
            return 2  # system-level normal
        return 99

    fallback_records = fallback_records[
        fallback_records["tree"].isin(["left", "right"])
    ]
    fallback_records["fallback_priority"] = fallback_records.apply(fallback_priority, axis=1)
    fallback_records = fallback_records[fallback_records["fallback_priority"] < 99]

    fallback_map: Dict[str, pd.Series] = {}
    if not fallback_records.empty:
        fallback_sorted = fallback_records.sort_values(
            by=["fallback_priority", "prompt_type", "text_id"]
        )
        for vid, group in fallback_sorted.groupby("video_id"):
            fallback_map[vid] = group.iloc[0]

    candidate = raw_prompts_df[
        raw_prompts_df["prompt_type"].isin(["lesion_atomic", "abnormal_focus"])
    ].copy()
    candidate = candidate[
        candidate["tags"].notna()
        & candidate["text_id"].notna()
        & candidate["video_id"].notna()
    ]

    candidate["tag_dict"] = candidate["tags"].apply(parse_tags)
    candidate["tree"] = candidate["tag_dict"].apply(lambda d: d.get("tree"))
    candidate["category"] = candidate["tag_dict"].apply(lambda d: d.get("category"))
    candidate["segment"] = candidate["tag_dict"].apply(lambda d: d.get("segment"))
    candidate["bin"] = candidate["tag_dict"].apply(lambda d: d.get("bin"))
    candidate = candidate[candidate["tree"].isin(["left", "right"])]

    # Keep at most one record per video/text pair so each video retains its own prompts
    candidate = candidate.drop_duplicates(subset=["video_id", "text_id"])

    cat_priority_map = {"in_stent": 0, "cto": 0, "stenosis": 1, "medina": 2, "calcification": 3, "thrombus": 3}
    bin_priority_map = {"<30": 5, "30-49": 4, "50-69": 3, "70-89": 2, ">=90": 1}
    prompt_priority_map = {"abnormal_focus": 0, "lesion_atomic": 1}

    candidate["category_priority"] = candidate["category"].map(cat_priority_map).fillna(4).astype(int)
    candidate["bin_priority"] = candidate["bin"].map(bin_priority_map).fillna(3).astype(int)
    candidate["prompt_priority"] = candidate["prompt_type"].map(prompt_priority_map).fillna(1).astype(int)

    videos_ext = videos_df.copy()
    videos_ext["tree"] = videos_ext["main_structure"].apply(get_tree)
    videos_ext = videos_ext[videos_ext["tree"].isin(["left", "right"])]
    video_tree_map = videos_ext.set_index("video_id")["tree"].to_dict()

    edges_records: List[Dict[str, Any]] = []
    assignment_records: List[Dict[str, Any]] = []
    counts: Dict[str, int] = defaultdict(int)

    candidate_sorted = candidate.sort_values(
        by=["category_priority", "prompt_priority", "bin_priority", "text_id"],
        ascending=[True, True, True, True]
    )

    if cap_per_video is not None:
        candidate_sorted = (
            candidate_sorted.groupby("video_id", group_keys=False)
            .head(cap_per_video)
        )

    for row in candidate_sorted.itertuples(index=False):
        vid = row.video_id
        video_tree = video_tree_map.get(vid)
        if not video_tree or row.tree != video_tree:
            continue

        weight = float(getattr(row, "soft_weight", 1.0))
        edges_records.append({
            "video_id": vid,
            "text_id": row.text_id,
            "weight": weight
        })
        assignment_records.append({
            "video_id": vid,
            "text_id": row.text_id,
            "prompt_type": row.prompt_type,
            "category": row.category,
            "weight": weight
        })
        counts[vid] += 1

    for vid in videos_ext["video_id"]:
        if counts.get(vid, 0) > 0:
            continue
        fallback = fallback_map.get(vid)
        if fallback is None:
            continue
        video_tree = video_tree_map.get(vid)
        if video_tree != fallback.get("tree"):
            continue
        fallback_text_id = fallback.get("text_id")
        if not isinstance(fallback_text_id, str):
            continue
        weight = float(fallback.get("soft_weight", 1.0))
        edges_records.append({
            "video_id": vid,
            "text_id": fallback_text_id,
            "weight": weight
        })
        tag_dict = fallback.get("tag_dict") or {}
        assignment_records.append({
            "video_id": vid,
            "text_id": fallback_text_id,
            "prompt_type": fallback.get("prompt_type"),
            "category": tag_dict.get("category"),
            "weight": weight
        })
        counts[vid] = 1

    edges_df = pd.DataFrame(edges_records)
    if not edges_df.empty:
        edges_df = edges_df.drop_duplicates(subset=["video_id", "text_id"])
    else:
        edges_df = pd.DataFrame(columns=["video_id", "text_id", "weight"])

    videos_debug = videos_df.copy()
    if not edges_df.empty:
        video_tree_map = videos_df.set_index("video_id")["main_structure"].map(get_tree).to_dict()
        text_tree_map = texts_df.set_index("text_id")["tags"].map(extract_tree_from_tags).to_dict()

        edges_df["video_tree"] = edges_df["video_id"].map(video_tree_map)
        edges_df["text_tree"] = edges_df["text_id"].map(text_tree_map)

        before_filter = len(edges_df)
        edges_df = edges_df[
            edges_df["video_tree"].notna()
            & edges_df["text_tree"].notna()
            & (edges_df["video_tree"] == edges_df["text_tree"])
        ].copy()
        dropped = before_filter - len(edges_df)
        if dropped > 0:
            logger.warning(
                "Dropped %d edges due to coronary tree mismatch between video and text",
                dropped,
            )

        edges_df = edges_df.drop(columns=["video_tree", "text_tree"])

        pos_map = (
            edges_df.groupby("video_id")["text_id"]
            .apply(lambda vals: "|".join(sorted(set(vals))))
            .reset_index(name="positive_text_ids")
        )
        videos_debug = videos_debug.merge(pos_map, on="video_id", how="left")
        videos_debug["positive_text_ids"] = videos_debug["positive_text_ids"].fillna("")
    else:
        videos_debug["positive_text_ids"] = ""

    stats: Dict[str, Any] = {
        "total_videos": len(videos_df),
        "videos_with_positives": edges_df["video_id"].nunique(),
        "total_edges": int(len(edges_df)),
        "prompt_type_counts": {},
        "category_counts": {}
    }

    if stats["videos_with_positives"] > 0:
        counts_series = edges_df.groupby("video_id")["text_id"].count()
        stats["positive_ratio"] = stats["videos_with_positives"] / stats["total_videos"]
        stats["mean_pos_per_positive_video"] = float(counts_series.mean())
        stats["median_pos_per_positive_video"] = float(counts_series.median())
        stats["max_pos_per_video"] = int(counts_series.max())
    else:
        stats.update({
            "positive_ratio": 0.0,
            "mean_pos_per_positive_video": 0.0,
            "median_pos_per_positive_video": 0.0,
            "max_pos_per_video": 0
        })

    if assignment_records:
        assignment_df = pd.DataFrame(assignment_records)
        stats["prompt_type_counts"] = assignment_df["prompt_type"].value_counts().to_dict()
        stats["category_counts"] = assignment_df["category"].value_counts().to_dict()
    else:
        stats["prompt_type_counts"] = {}
        stats["category_counts"] = {}

    return edges_df[["video_id", "text_id", "weight"]], videos_debug, stats


def write_debug_outputs(
    output_path: Path,
    config: Dict[str, Any],
    videos_df: pd.DataFrame,
    texts_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    raw_prompts_df: pd.DataFrame,
) -> None:
    """Persist optional debugging artifacts for quick sanity checks."""
    debug_cfg = config.get("debug_outputs", {})
    if not debug_cfg.get("enabled", True):
        return

    sample_size = int(debug_cfg.get("video_sample_size", 25))
    summary_rows = int(debug_cfg.get("summary_limit", 1000))

    try:
        counts = edges_df.groupby("video_id")["text_id"].size()
        counts_summary = counts.describe().to_frame(name="edges_per_video")
        counts_summary.loc["total_edges"] = len(edges_df)
        counts_summary.loc["unique_videos"] = counts.index.nunique()
        counts_summary.to_csv(output_path / "debug_edges_per_video_summary.csv")
    except Exception as exc:
        logger.warning("Failed to write edges per video summary: %s", exc)

    try:
        merged_edges = edges_df.merge(
            texts_df[["text_id", "prompt_type", "prompt_text", "tags"]],
            on="text_id",
            how="left",
        ).merge(
            videos_df[["video_id", "Split", "main_structure"]],
            on="video_id",
            how="left",
        )
        sample = (
            merged_edges.sort_values(["Split", "video_id"])
            .groupby("Split", group_keys=False)
            .head(sample_size)
        )
        sample.to_csv(output_path / "debug_video_text_samples.csv", index=False)
    except Exception as exc:
        logger.warning("Failed to write SigLIP debug samples: %s", exc)

    try:
        prompt_counts = (
            raw_prompts_df["prompt_type"]
            .value_counts()
            .rename_axis("prompt_type")
            .reset_index(name="count")
        )
        prompt_counts.to_csv(output_path / "debug_prompt_type_counts.csv", index=False)

        tag_counter: Counter[str] = Counter()
        for tags in raw_prompts_df["tags"].dropna():
            tag_counter.update(str(tags).split("|"))
        tag_series = pd.DataFrame(
            tag_counter.most_common(summary_rows), columns=["tag", "count"]
        )
        tag_series.to_csv(output_path / "debug_tag_frequency.csv", index=False)
    except Exception as exc:
        logger.warning("Failed to write prompt/tag debug summaries: %s", exc)
def process_siglip_dataset(
    input_path: str,
    output_dir: str,
    config: Dict[str, Any]
) -> None:
    """Main processing pipeline for SigLIP category dataset."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data with alpha separator
    logger.info(f"Loading data from {input_path}")
    if input_path.endswith('.csv'):
        df = pd.read_csv(input_path, sep='α', engine='python', on_bad_lines='skip')
    elif input_path.endswith('.parquet'):
        df = pd.read_parquet(input_path)
    else:
        raise ValueError(f"Unsupported file format: {input_path}")

    # Infer main structure class if missing but object_value available (legacy datasets)
    if 'main_structure_class' not in df.columns and 'object_value' in df.columns:
        logger.warning(
            "main_structure_class column missing; inferring from object_value using legacy mapping"
        )
        inferred = df['object_value'].map({5: 0, 9: 1})
        df['main_structure_class'] = inferred.fillna(2).astype(int)

    # Normalise stent presence column (allow alternate naming)
    if 'stent_presence_class' not in df.columns:
        stent_candidate = None
        for candidate in ['stent_presence', 'stent_presence_flag', 'stent_present']:
            if candidate in df.columns:
                stent_candidate = candidate
                break
        if stent_candidate is not None:
            logger.info(f"Creating stent_presence_class from {stent_candidate}")
            stent_series = pd.to_numeric(df[stent_candidate], errors='coerce').fillna(0.0)
            df['stent_presence_class'] = stent_series.astype(float).clip(lower=0).round().astype(int)
        else:
            logger.warning("No stent presence column found; PCI status will default to diagnostic")

    # Apply mappings
    if config.get('apply_mappings', True):
        if 'main_structure_class' in df.columns:
            df['main_structure_name'] = df['main_structure_class'].map(MAIN_STRUCTURE_MAP)
            df['main_structure_description'] = df['main_structure_class'].apply(format_main_structure_description)
        if 'dominance_class' in df.columns:
            df['dominance_name'] = df['dominance_class'].map(DOMINANCE_MAP)

        df = add_angiographic_view_column(df)

        if 'main_structure_name' in df.columns:
            counts_before = df['main_structure_name'].value_counts(dropna=False).sort_values(ascending=False)
            logger.info("Initial main_structure distribution:\n%s", counts_before)
            plot_path_before = output_path / 'main_structure_distribution_before.png'
            plt.figure(figsize=(10, 5))
            counts_before.plot(kind='bar')
            plt.title('Main Structure Distribution (Before Filtering)')
            plt.ylabel('Count')
            plt.xlabel('main_structure_name')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(plot_path_before, dpi=150)
            plt.close()

        if 'Conclusion' in df.columns:
            df['bypass_graft'] = df['Conclusion'].str.contains('pontage', case=False, na=False).astype(int)

        if 'StudyInstanceUID' in df.columns and 'SeriesTime' in df.columns:
            df = df.sort_values(['StudyInstanceUID', 'SeriesTime'])

    # Assign procedure status
    if config.get('assign_status', True):
        if 'status' in df.columns:
            logger.info("Status column already present; using existing values")
        elif 'stent_presence_class' in df.columns and 'StudyInstanceUID' in df.columns:
            df = assign_procedure_status(df)
        else:
            logger.warning(
                "Cannot assign procedure status: missing stent_presence_class or StudyInstanceUID."
                " Defaulting status to 'diagnostic'."
            )
            df['status'] = 'diagnostic'

    # Apply filters
    df_filtered = apply_hard_filters(df, config)

    # Apply patient-based splits
    if config.get('train_test_split', {}).get('enabled', False):
        train_ratio = config['train_test_split'].get('train_ratio', 0.7)
        val_ratio = config['train_test_split'].get('val_ratio', 0.1)
        test_ratio = config['train_test_split'].get('test_ratio', 0.2)
        random_state = config['train_test_split'].get('random_state', 42)
        patient_column = config['train_test_split'].get('patient_column', 'CathReport_MRN')

        df_filtered = assign_patient_splits(
            df_filtered,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_state=random_state,
            patient_column=patient_column
        )

    df_filtered = df_filtered.reset_index(drop=True)
    df_filtered["video_id"] = [f"V{i:06d}" for i in range(len(df_filtered))]

    # Keep only left/right coronary systems
    initial_count = len(df_filtered)
    if 'main_structure_class' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['main_structure_class'].isin([0, 1])]
    elif 'main_structure_name' in df_filtered.columns:
        mask_lr = df_filtered['main_structure_name'].isin(['Left Coronary', 'Right Coronary'])
        df_filtered = df_filtered[mask_lr]
    elif 'main_structure' in df_filtered.columns:
        mask_lr = df_filtered['main_structure'].isin(['Left Coronary', 'Right Coronary'])
        df_filtered = df_filtered[mask_lr]
    if len(df_filtered) != initial_count:
        logger.info(
            "Filtered to left/right coronary videos: %d -> %d",
            initial_count,
            len(df_filtered)
        )

    if 'main_structure_name' in df_filtered.columns:
        counts_after = df_filtered['main_structure_name'].value_counts(dropna=False).sort_values(ascending=False)
        logger.info("Filtered main_structure distribution:\n%s", counts_after)
        plot_path_after = output_path / 'main_structure_distribution_after.png'
        plt.figure(figsize=(6, 5))
        counts_after.plot(kind='bar', color='orange')
        plt.title('Main Structure Distribution (After Filtering)')
        plt.ylabel('Count')
        plt.xlabel('main_structure_name')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(plot_path_after, dpi=150)
        plt.close()

    # Optional sampling per split for debugging purposes
    sample_cfg = config.get('sample_limits', {}) or {}
    if sample_cfg:
        split_col = 'Split'
        if split_col in df_filtered.columns:
            rng_seed = sample_cfg.get('random_state', 421561)
            frames = []
            split_lower = df_filtered[split_col].astype(str).str.lower()
            unique_splits = split_lower.unique()
            for split in unique_splits:
                subset = df_filtered[split_lower == split]
                size_key = f"{split}_size"
                size = sample_cfg.get(size_key)
                if size is None:
                    frames.append(subset)
                    continue
                try:
                    size = int(size) if size is not None else None
                except ValueError:
                    size = None
                if size is None or size >= len(subset):
                    frames.append(subset)
                elif size > 0:
                    frames.append(subset.sample(n=size, random_state=rng_seed))
                else:
                    # skip this split entirely if size <= 0
                    continue
            if frames:
                df_filtered = pd.concat(frames, ignore_index=True)
                logger.info(
                    "Applied sample_limits; resulting dataset has %d rows", len(df_filtered)
                )

    # Generate videos.csv
    logger.info("Generating videos.csv...")
    videos_df = generate_videos_csv(df_filtered)
    videos_path = output_path / "videos.csv"
    videos_df.to_csv(videos_path, index=False)
    logger.info(f"Saved videos.csv to {videos_path} ({len(videos_df)} videos)")

    # Generate texts.csv
    logger.info("Generating multi-style text prompts...")
    parallel_workers = max(1, int(config.get('parallel_workers', 1)))

    texts_df, raw_prompts_df = generate_multi_prompt_texts(
        df_filtered,
        parallel_workers=parallel_workers
    )
    texts_path = output_path / "texts.csv"
    texts_df.to_csv(texts_path, index=False)
    logger.info(f"Saved texts.csv to {texts_path} ({len(texts_df)} unique prompts)")

    # Build edges_pos.csv and optional debug positives
    logger.info("Building video-text edges...")
    cap_per_video = config.get('cap_per_video')
    edges_df, videos_debug_df, edge_stats = build_edges(
        raw_prompts_df,
        videos_df,
        texts_df,
        cap_per_video=cap_per_video,
    )

    edges_path = output_path / "edges_pos.csv"
    edges_df.to_csv(edges_path, index=False)
    logger.info(f"Saved edges_pos.csv to {edges_path} ({len(edges_df)} edges)")

    videos_debug_path = output_path / "videos_with_debug_pos.csv"
    videos_debug_df.to_csv(videos_debug_path, index=False)
    logger.info(f"Saved debug positives to {videos_debug_path}")

    write_debug_outputs(
        output_path=output_path,
        config=config,
        videos_df=videos_df,
        texts_df=texts_df,
        edges_df=edges_df,
        raw_prompts_df=raw_prompts_df,
    )

    # Print statistics
    print("\n" + "="*60)
    print("SigLIP DATASET GENERATION SUMMARY")
    print("="*60)
    print(f"\nVideos: {len(videos_df)}")
    print(f"Unique studies: {videos_df['study_id'].nunique()}")
    print(f"Unique text prompts: {len(texts_df)}")

    if 'split' in videos_df.columns:
        print("\nVideo split distribution:")
        print(videos_df['split'].value_counts())

    if 'main_structure' in videos_df.columns:
        print("\nVideos by coronary system:")
        print(videos_df['main_structure'].value_counts())

    print("\nPrompt type distribution:")
    print(texts_df['prompt_type'].value_counts())

    print("\nPrompt type weights:")
    for ptype in texts_df['prompt_type'].unique():
        weight = texts_df[texts_df['prompt_type'] == ptype]['soft_weight'].iloc[0]
        count = len(texts_df[texts_df['prompt_type'] == ptype])
        print(f"  {ptype}: {weight} ({count} prompts)")

    if edge_stats.get("videos_with_positives", 0) > 0:
        print("\nSupervision coverage:")
        ratio = edge_stats['positive_ratio'] * 100
        print(
            f"  Videos with positives: {edge_stats['videos_with_positives']} "
            f"({ratio:.1f}%)"
        )
        print(
            f"  Positives per positive video: mean={edge_stats['mean_pos_per_positive_video']:.2f}, "
            f"median={edge_stats['median_pos_per_positive_video']:.1f}, "
            f"max={edge_stats['max_pos_per_video']}"
        )
        print(f"  Total edges: {edge_stats['total_edges']}")

        if edge_stats.get("prompt_type_counts"):
            print("\nEdge prompt type counts:")
            for key, val in edge_stats["prompt_type_counts"].items():
                print(f"  {key}: {val}")

        if edge_stats.get("category_counts"):
            print("\nEdge category counts:")
            for key, val in edge_stats["category_counts"].items():
                print(f"  {key}: {val}")
    else:
        print("\nSupervision coverage: No positive edges generated.")

    # Sample prompts
    print("\n" + "="*60)
    print("SAMPLE TEXT PROMPTS (20 random)")
    print("="*60)
    sample = texts_df.sample(min(20, len(texts_df)), random_state=42)
    for _, row in sample.iterrows():
        print(f"\n[{row['prompt_type'].upper()}] weight={row['soft_weight']}")
        print(f"  {row['prompt_text']}")
        print(f"  tags: {row['tags']}")

    # Save config
    config_path = output_path / "siglip_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    logger.info("\nSigLIP category dataset generation completed!")


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def create_default_config() -> Dict[str, Any]:
    """Create default configuration."""
    return {
        'filters': {
            'status': ['diagnostic', 'POST_PCI'],
            'main_structures': ['Left Coronary', 'Right Coronary'],
            'contrast_agent_class': 1,
            'normal_report_ratio': 0.05
        },
        'train_test_split': {
            'enabled': True,
            'train_ratio': 0.7,
            'val_ratio': 0.1,
            'test_ratio': 0.2,
            'random_state': 42,
            'patient_column': 'CathReport_MRN'
        },
        'apply_mappings': True,
        'assign_status': True,
        'cap_per_video': None,
        'parallel_workers': 1,
        'debug_outputs': {
            'enabled': True,
            'video_sample_size': 25,
            'summary_limit': 1000
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate SigLIP category-based dataset (videos.csv + texts.csv)"
    )
    parser.add_argument(
        '--input-path',
        required=True,
        help='Path to input CSV or Parquet file'
    )
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Directory to save videos.csv and texts.csv'
    )
    parser.add_argument(
        '--config',
        help='Path to configuration YAML file (optional)'
    )

    args = parser.parse_args()

    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = create_default_config()

    # Process dataset
    process_siglip_dataset(args.input_path, args.output_dir, config)


if __name__ == "__main__":
    main()
