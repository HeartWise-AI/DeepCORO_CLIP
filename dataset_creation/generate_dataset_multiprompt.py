#!/usr/bin/env python3
"""
Enhanced Dataset Generation Script with Multi-Prompt Support for SigLIP Training

This script extends the original generate_dataset.py to generate multiple prompts per video
following the SigLIP-inspired approach with different prompt types and weights.
Now includes ALL medical details: CTO, stents, calcification, IFR, bifurcations, etc.

IMPORTANT: Each video is restricted to either RCA vessels OR Non-RCA vessels based on main_structure_name.

Prompt Types Generated:
1. Global summary (weight: 0.5) - Complete study summary with all details
2. Abnormal focus (weight: 1.0) - Critical findings (≥70% stenosis) with medical details
3. Atomic per-lesion (weight: 0.6) - Individual lesion descriptions with full medical context
4. Negative coverage (weight: 0.6) - Normal territory descriptions
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

# Import from the original generate_dataset
from generate_dataset import (
    load_data,
    apply_hard_filters,
    generate_reports,
    sample_by_status,
    assign_procedure_status,
    assign_patient_splits,
    add_angiographic_view_column,
    format_main_structure_description,
    format_stenosis_value,
    format_calcification_value,
    format_ifr_value,
    MAIN_STRUCTURE_MAP,
    DOMINANCE_MAP,
    LABELS_TO_VESSEL_NAMES,
    RCA_VESSELS,
    NON_RCA_VESSELS,
    RIGHT_DOMINANCE_DEPENDENT_VESSELS,
    LEFT_DOMINANCE_DEPENDENT_VESSELS
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

tqdm.pandas()

# Vessel groupings for territory classification
LCX_VESSELS = ["lcx_stenosis", "dist_lcx_stenosis", "om1_stenosis", "om2_stenosis", "bx_stenosis", "lvp_stenosis"]
LAD_VESSELS = ["prox_lad_stenosis", "mid_lad_stenosis", "dist_lad_stenosis", "D1_stenosis", "D2_stenosis"]

# Prompt type weights
PROMPT_WEIGHTS = {
    "global_summary": 0.5,
    "abnormal_focus": 1.0,
    "lesion_atomic": 0.6,
    "negative_coverage": 0.6
}


def get_relevant_vessels(row: pd.Series, vessel_separation: bool = True) -> List[str]:
    """Get the list of vessels relevant to this video.

    Args:
        row: DataFrame row with video data
        vessel_separation: If True, separate RCA from Non-RCA vessels based on main_structure_name.
                         If False, include all vessels for full video aggregation.

    Returns:
        List of vessel column names to process
    """
    dominance = str(row.get("dominance_name", "")).lower()

    if not vessel_separation:
        # Full video aggregation - include ALL vessels regardless of main_structure
        vessels = RCA_VESSELS.copy() + NON_RCA_VESSELS.copy()
        # Add dominance-dependent vessels
        if "right" in dominance:
            vessels.extend(RIGHT_DOMINANCE_DEPENDENT_VESSELS)
        if "left" in dominance:
            vessels.extend(LEFT_DOMINANCE_DEPENDENT_VESSELS)
        return vessels

    # Original behavior - separate by main_structure
    main_structure = row.get("main_structure_name", "")

    if main_structure == "Right Coronary":
        # RCA view: only RCA vessels
        vessels = RCA_VESSELS.copy()
        # Add dominance-dependent vessels if right dominant
        if "right" in dominance:
            vessels.extend(RIGHT_DOMINANCE_DEPENDENT_VESSELS)
        return vessels

    elif main_structure == "Left Coronary":
        # Left coronary view: LAD, LCx, Left Main
        vessels = NON_RCA_VESSELS.copy()
        # Add dominance-dependent vessels if left dominant
        if "left" in dominance:
            vessels.extend(LEFT_DOMINANCE_DEPENDENT_VESSELS)
        return vessels

    else:
        # Unknown structure - return empty list
        logger.warning(f"Unknown main_structure_name: {main_structure}")
        return []


def get_vessel_display_name(vessel_col: str, row: pd.Series = None) -> str:
    """Get human-readable vessel name from column name, considering dominance."""
    vessel_map = {
        "left_main_stenosis": "Left Main",
        "prox_lad_stenosis": "Proximal LAD",
        "mid_lad_stenosis": "Mid LAD",
        "dist_lad_stenosis": "Distal LAD",
        "D1_stenosis": "D1",
        "D2_stenosis": "D2",
        "lcx_stenosis": "Proximal LCx",
        "dist_lcx_stenosis": "Distal LCx",
        "om1_stenosis": "OM1",
        "om2_stenosis": "OM2",
        "prox_rca_stenosis": "Proximal RCA",
        "mid_rca_stenosis": "Mid RCA",
        "dist_rca_stenosis": "Distal RCA",
        "pda_stenosis": "PDA",
        "posterolateral_stenosis": "Posterolateral",
        "bx_stenosis": "Ramus",
        "lvp_stenosis": "LVP branch"
    }

    # Adjust for dominance if row is provided
    if row is not None:
        dominance = str(row.get("dominance_name", "")).lower()
        if "left" in dominance:
            if vessel_col == "pda_stenosis":
                return "Left PDA"
            elif vessel_col == "posterolateral_stenosis":
                return "Left Posterolateral"

    return vessel_map.get(vessel_col, vessel_col.replace("_stenosis", "").replace("_", " ").title())


def get_vessel_details(row: pd.Series, vessel_col: str) -> Dict[str, Any]:
    """Extract all medical details for a vessel including CTO, stent, calcification, IFR."""
    prefix = vessel_col.replace("_stenosis", "")
    details = {
        "stenosis": None,
        "is_cto": False,
        "has_stent": False,
        "in_stent_restenosis": False,
        "calcification": None,
        "ifr": None,
        "bifurcation": None
    }

    # Check for CTO (100% blocked)
    cto_col = prefix + "_cto"
    if cto_col in row.index:
        is_cto = row[cto_col]
        if pd.notna(is_cto) and is_cto == 1:
            details["is_cto"] = True
            details["stenosis"] = 100.0
            return details  # CTO overrides other stenosis values

    # Regular stenosis value
    if vessel_col in row.index:
        st = row[vessel_col]
        if pd.notna(st) and st >= 0:
            details["stenosis"] = float(st)

            # Check for stent and restenosis
            stent_col = prefix + "_stent"
            if stent_col in row.index:
                has_stent = row[stent_col]
                if pd.notna(has_stent) and has_stent > 0:
                    details["has_stent"] = True
                    if details["stenosis"] > 10:  # In-stent restenosis
                        details["in_stent_restenosis"] = True

    # Calcification
    calc_col = prefix + "_calcif"
    if calc_col in row.index:
        calc = row[calc_col]
        if isinstance(calc, str) and calc.strip() not in ["-1", ""]:
            details["calcification"] = format_calcification_value(calc)

    # IFR assessment
    ifr_col = prefix + "_IFRHYPEREMIE"
    if ifr_col in row.index:
        ifr = row[ifr_col]
        if pd.notna(ifr) and ifr != -1:
            details["ifr"] = float(ifr)

    # Bifurcation lesion
    bifurcation_col = prefix + "_bifurcation"
    if bifurcation_col in row.index:
        bif = row[bifurcation_col]
        if pd.notna(bif) and bif not in [0, 0.0, "", "0", "0.0", "Pas de lésion de bifurcation", "nan"]:
            if isinstance(bif, str) and bif.strip():
                if bif.strip().lower() != "pas de lésion de bifurcation":
                    details["bifurcation"] = bif.strip()
            elif isinstance(bif, (int, float)) and bif != 0:
                details["bifurcation"] = str(bif)

    return details


def generate_global_summary(row: pd.Series, vessel_separation: bool = True) -> str:
    """Generate comprehensive global summary prompt (Type 1) - restricted to relevant vessels only."""
    findings = []

    # Get only the relevant vessels for this view
    relevant_vessels = get_relevant_vessels(row, vessel_separation)

    for vessel_col in relevant_vessels:
        if vessel_col in row.index:
            details = get_vessel_details(row, vessel_col)
            if details["stenosis"] is not None and details["stenosis"] > 0:
                vessel_name = get_vessel_display_name(vessel_col, row)

                # Basic description
                if details["is_cto"]:
                    desc = f"{vessel_name} 100% (CTO)"
                elif details["in_stent_restenosis"]:
                    desc = f"{vessel_name} {int(details['stenosis'])}% (in-stent restenosis)"
                else:
                    desc = f"{vessel_name} {int(details['stenosis'])}% stenosis"

                findings.append((details["stenosis"], desc))

    if not findings:
        main_structure = row.get("main_structure_name", "")
        if main_structure == "Right Coronary":
            return "RCA: No significant stenosis documented; segments not specified."
        elif main_structure == "Left Coronary":
            return "Left coronary: No significant stenosis documented; segments not specified."
        else:
            return "No significant stenosis documented; segments not specified."

    # Sort by severity
    findings.sort(key=lambda x: x[0], reverse=True)

    # Format as semicolon-separated list
    parts = [desc for _, desc in findings]
    return "; ".join(parts) + "; other segments not specified."


def generate_abnormal_focus(row: pd.Series, vessel_separation: bool = True) -> str:
    """Generate abnormal-focused summary (Type 2) - restricted to relevant vessels only."""
    critical_findings = []
    moderate_findings = []
    mild_findings = []

    # Get only the relevant vessels for this view
    relevant_vessels = get_relevant_vessels(row, vessel_separation)

    for vessel_col in relevant_vessels:
        if vessel_col in row.index:
            details = get_vessel_details(row, vessel_col)
            if details["stenosis"] is not None:
                vessel_name = get_vessel_display_name(vessel_col, row)
                value = details["stenosis"]

                if value >= 70 or details["is_cto"]:
                    # Include full medical details for critical findings
                    if details["is_cto"]:
                        desc = f"CTO in {vessel_name}"
                    else:
                        severity = "Critical" if value >= 90 else "Severe"
                        desc = f"{severity} {vessel_name} (≈{int(value)}%)"

                    # Add medical details
                    extras = []
                    if details["in_stent_restenosis"]:
                        extras.append("in-stent restenosis")
                    elif details["has_stent"]:
                        extras.append("stented")

                    if details["calcification"] and "no calcif" not in details["calcification"]:
                        extras.append(details["calcification"])

                    if details["ifr"] is not None and details["ifr"] <= 0.89:
                        extras.append(f"IFR {details['ifr']:.2f}")

                    if details["bifurcation"]:
                        extras.append(f"Medina {details['bifurcation']}")

                    if extras:
                        desc += " [" + ", ".join(extras) + "]"

                    critical_findings.append((value, desc))

                elif value >= 50:
                    moderate_findings.append((vessel_name, value))
                elif value >= 30:
                    mild_findings.append((vessel_name, value))

    if not critical_findings:
        main_structure = row.get("main_structure_name", "")
        if main_structure == "Right Coronary":
            return "RCA: No ≥70% stenosis identified."
        elif main_structure == "Left Coronary":
            return "Left coronary: No ≥70% stenosis identified."
        else:
            return "No ≥70% stenosis identified."

    # Sort critical by severity
    critical_findings.sort(key=lambda x: x[0], reverse=True)

    parts = []
    for i, (_, desc) in enumerate(critical_findings):
        if i == 0:
            parts.append(desc)
        else:
            # Make subsequent findings start with "Additional"
            if desc.startswith("CTO"):
                parts.append(f"Additional {desc}")
            elif desc.startswith("Critical") or desc.startswith("Severe"):
                parts.append(f"Additional {desc[0].lower() + desc[1:]}")
            else:
                parts.append(desc)

    # Add moderate/mild if present (without details)
    if moderate_findings:
        mod_desc = ", ".join([f"{v[0]} (≈{int(v[1])}%)" for v in moderate_findings])
        parts.append(f"Additional moderate disease in {mod_desc}")

    if mild_findings:
        mild_desc = ", ".join([f"{v[0]} (≈{int(v[1])}%)" for v in mild_findings])
        parts.append(f"Additional mild disease in {mild_desc}")

    return "; ".join(parts) + "."


def generate_atomic_lesions(row: pd.Series, vessel_separation: bool = True) -> List[str]:
    """Generate atomic per-lesion captions (Type 3) - restricted to relevant vessels only."""
    atomic_prompts = []

    # Get only the relevant vessels for this view
    relevant_vessels = get_relevant_vessels(row, vessel_separation)

    for vessel_col in relevant_vessels:
        if vessel_col in row.index:
            details = get_vessel_details(row, vessel_col)

            # Include lesions ≥30% or special cases (CTO, stented vessels)
            if (details["stenosis"] is not None and details["stenosis"] >= 30) or \
               details["is_cto"] or details["has_stent"]:

                vessel_name = get_vessel_display_name(vessel_col, row)

                # Build atomic description
                if details["is_cto"]:
                    prompt = f"{vessel_name}; 100% occlusion (CTO)"
                elif details["stenosis"] is not None:
                    value = details["stenosis"]

                    # Determine severity category
                    if value < 30:
                        if details["has_stent"]:
                            prompt = f"{vessel_name}; stented vessel, no restenosis"
                        else:
                            prompt = f"{vessel_name}; <30% stenosis (minimal)"
                    elif value < 50:
                        if details["in_stent_restenosis"]:
                            prompt = f"{vessel_name}; 30-49% in-stent restenosis (mild)"
                        else:
                            prompt = f"{vessel_name}; 30-49% stenosis (mild)"
                    elif value < 70:
                        if details["in_stent_restenosis"]:
                            prompt = f"{vessel_name}; 50-69% in-stent restenosis (moderate)"
                        else:
                            prompt = f"{vessel_name}; 50-69% stenosis (moderate)"
                    elif value < 90:
                        if details["in_stent_restenosis"]:
                            prompt = f"{vessel_name}; 70-89% in-stent restenosis (severe)"
                        else:
                            prompt = f"{vessel_name}; 70-89% stenosis (severe)"
                    else:
                        if details["in_stent_restenosis"]:
                            prompt = f"{vessel_name}; ≥90% in-stent restenosis (critical)"
                        else:
                            prompt = f"{vessel_name}; ≥90% stenosis (critical/occlusion-range)"

                    # Add medical details
                    extras = []
                    if details["calcification"] and "no calcif" not in details["calcification"]:
                        extras.append(details["calcification"])

                    if details["ifr"] is not None:
                        if details["ifr"] <= 0.89:
                            extras.append(f"IFR abnormal ({details['ifr']:.2f})")
                        else:
                            extras.append(f"IFR normal ({details['ifr']:.2f})")

                    if details["bifurcation"]:
                        extras.append(f"bifurcation (Medina {details['bifurcation']})")

                    if extras:
                        prompt += "; " + "; ".join(extras)
                else:
                    continue  # Skip if no valid data

                prompt += "."
                atomic_prompts.append(prompt)

    return atomic_prompts


def generate_negative_coverage(row: pd.Series, vessel_separation: bool = True) -> str:
    """Generate negative/normal coverage prompt (Type 4) - restricted to relevant vessels."""
    main_structure = row.get("main_structure_name", "")
    dominance = str(row.get("dominance_name", "")).lower()

    territories = {}

    if not vessel_separation:
        # Full video aggregation - report on all territories
        territories["RCA territory"] = RCA_VESSELS.copy()
        territories["Left main"] = ["left_main_stenosis"]
        territories["LAD territory"] = LAD_VESSELS
        territories["LCx territory"] = LCX_VESSELS.copy()

        # Add dominance-dependent vessels
        if "right" in dominance:
            territories["RCA territory"].extend(RIGHT_DOMINANCE_DEPENDENT_VESSELS)
        if "left" in dominance:
            territories["LCx territory"].extend(LEFT_DOMINANCE_DEPENDENT_VESSELS)

    elif main_structure == "Right Coronary":
        # For RCA videos, only report on RCA territory
        territories["RCA territory"] = RCA_VESSELS.copy()
        if "right" in dominance:
            territories["RCA territory"].extend(RIGHT_DOMINANCE_DEPENDENT_VESSELS)

    elif main_structure == "Left Coronary":
        # For Left coronary videos, report on Left Main, LAD, and LCx territories
        territories["Left main"] = ["left_main_stenosis"]
        territories["LAD territory"] = LAD_VESSELS
        territories["LCx territory"] = LCX_VESSELS.copy()
        if "left" in dominance:
            territories["LCx territory"].extend(LEFT_DOMINANCE_DEPENDENT_VESSELS)

    else:
        return ""  # Unknown structure

    negative_statements = []

    for territory_name, vessels in territories.items():
        # Analyze territory
        max_stenosis = -1
        has_data = False
        has_calcification = False
        all_normal_or_unknown = True

        for vessel in vessels:
            if vessel in row.index:
                details = get_vessel_details(row, vessel)

                if details["stenosis"] is not None and details["stenosis"] >= 0:
                    has_data = True
                    max_stenosis = max(max_stenosis, details["stenosis"])
                    if details["stenosis"] >= 30:
                        all_normal_or_unknown = False

                # Check for significant calcification
                if details["calcification"] and "no calcif" not in details["calcification"]:
                    has_calcification = True

        # Generate statement for this territory
        if not has_data:
            negative_statements.append(f"{territory_name}: not specified.")
        elif max_stenosis < 50:
            if max_stenosis <= 30:
                statement = f"{territory_name}: all lesions ≤30%"
            else:
                statement = f"{territory_name}: all lesions <50%"

            # Add calcification note if no significant disease but calcification present
            if has_calcification and all_normal_or_unknown:
                statement += " (calcifications noted)"

            negative_statements.append(statement + ".")

    return " ".join(negative_statements) if negative_statements else ""


def generate_multiprompt_dataset(df: pd.DataFrame, vessel_separation: bool = True) -> pd.DataFrame:
    """
    Generate multiple prompts per video following SigLIP approach with full medical details.

    Args:
        df: Input DataFrame with video data
        vessel_separation: If True, separate RCA from Non-RCA vessels based on main_structure_name.
                         If False, include all vessels for full video aggregation.

    Returns DataFrame with columns:
    - StudyInstanceUID
    - SeriesInstanceUID
    - FileName
    - prompt_text
    - prompt_type
    - prompt_weight
    - main_structure
    """
    all_prompts = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating multi-prompts"):
        study_id = row.get("StudyInstanceUID", f"study_{idx}")
        series_id = row.get("SeriesInstanceUID", f"series_{idx}")
        filename = row.get("FileName", "")
        main_structure = row.get("main_structure_name", "")

        # Skip if no valid main structure
        if main_structure not in ["Right Coronary", "Left Coronary"]:
            logger.warning(f"Skipping row {idx} with invalid main_structure_name: {main_structure}")
            continue

        # 1. Global summary
        global_summary = generate_global_summary(row, vessel_separation)
        if global_summary:
            all_prompts.append({
                "StudyInstanceUID": study_id,
                "SeriesInstanceUID": series_id,
                "FileName": filename,
                "prompt_text": global_summary,
                "prompt_type": "global_summary",
                "prompt_weight": PROMPT_WEIGHTS["global_summary"],
                "main_structure": main_structure
            })

        # 2. Abnormal focus
        abnormal_focus = generate_abnormal_focus(row, vessel_separation)
        if abnormal_focus:
            all_prompts.append({
                "StudyInstanceUID": study_id,
                "SeriesInstanceUID": series_id,
                "FileName": filename,
                "prompt_text": abnormal_focus,
                "prompt_type": "abnormal_focus",
                "prompt_weight": PROMPT_WEIGHTS["abnormal_focus"],
                "main_structure": main_structure
            })

        # 3. Atomic lesions
        atomic_lesions = generate_atomic_lesions(row, vessel_separation)
        for atomic_prompt in atomic_lesions:
            all_prompts.append({
                "StudyInstanceUID": study_id,
                "SeriesInstanceUID": series_id,
                "FileName": filename,
                "prompt_text": atomic_prompt,
                "prompt_type": "lesion_atomic",
                "prompt_weight": PROMPT_WEIGHTS["lesion_atomic"],
                "main_structure": main_structure
            })

        # 4. Negative coverage
        negative_coverage = generate_negative_coverage(row, vessel_separation)
        if negative_coverage:
            all_prompts.append({
                "StudyInstanceUID": study_id,
                "SeriesInstanceUID": series_id,
                "FileName": filename,
                "prompt_text": negative_coverage,
                "prompt_type": "negative_coverage",
                "prompt_weight": PROMPT_WEIGHTS["negative_coverage"],
                "main_structure": main_structure
            })

    return pd.DataFrame(all_prompts)


def process_dataset_multiprompt(
    input_path: str,
    output_dir: str,
    config: Dict[str, Any]
) -> None:
    """
    Main dataset processing pipeline with multi-prompt generation.
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_data(input_path)

    # Apply mappings if needed
    if config.get('apply_mappings', True):
        if 'main_structure_class' in df.columns:
            df['main_structure_name'] = df['main_structure_class'].map(MAIN_STRUCTURE_MAP)
            df['main_structure_description'] = df['main_structure_class'].apply(format_main_structure_description)
        if 'dominance_class' in df.columns:
            df['dominance_name'] = df['dominance_class'].map(DOMINANCE_MAP)

        df = add_angiographic_view_column(df)

        if 'Conclusion' in df.columns:
            df['bypass_graft'] = df['Conclusion'].str.contains('pontage', case=False, na=False).astype(int)

        if 'StudyInstanceUID' in df.columns and 'SeriesTime' in df.columns:
            df = df.sort_values(['StudyInstanceUID', 'SeriesTime'])

    # Assign procedure status
    if config.get('assign_status', True):
        if 'stent_presence_class' in df.columns and 'StudyInstanceUID' in df.columns:
            df = assign_procedure_status(df)

    # Apply filters
    df_filtered = apply_hard_filters(df, config)

    # Generate multiple prompts per video
    vessel_separation = config.get('vessel_separation', True)
    if vessel_separation:
        logger.info("Generating multiple prompts per video with vessel separation (RCA vs Non-RCA)...")
    else:
        logger.info("Generating multiple prompts per video with full vessel aggregation (all vessels)...")
    prompt_df = generate_multiprompt_dataset(df_filtered, vessel_separation)

    # Save the multi-prompt dataset
    output_file = output_path / "multiprompt_dataset.parquet"
    prompt_df.to_parquet(output_file, index=False)
    logger.info(f"Saved multi-prompt dataset to {output_file}")

    # Print sample
    print("\nSample of generated prompts:")
    print(prompt_df.head(10))

    # Print statistics
    print("\nPrompt statistics:")
    print(f"Total prompts generated: {len(prompt_df)}")
    print(f"Unique studies: {prompt_df['StudyInstanceUID'].nunique()}")
    print(f"Unique series: {prompt_df['SeriesInstanceUID'].nunique()}")

    if 'main_structure' in prompt_df.columns:
        print("\nPrompts by coronary system:")
        print(prompt_df['main_structure'].value_counts())

    print("\nPrompt type distribution:")
    print(prompt_df['prompt_type'].value_counts())
    print("\nPrompt weight distribution:")
    print(prompt_df.groupby('prompt_type')['prompt_weight'].first())

    # Save configuration
    config_output_path = output_path / "multiprompt_config.yaml"
    with open(config_output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    logger.info("Multi-prompt dataset generation with proper vessel separation completed successfully!")


def create_default_config() -> Dict[str, Any]:
    """Create default configuration for multi-prompt generation."""
    return {
        'filters': {
            'status': ['diagnostic', 'POST_PCI'],
            'main_structures': ['Left Coronary', 'Right Coronary'],
            'contrast_agent_class': 1,
            'normal_report_ratio': 0.05
        },
        'apply_mappings': True,
        'assign_status': True,
        'vessel_separation': True  # Set to False for full video aggregation (both L/R vessels)
    }


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Generate multi-prompt medical dataset for SigLIP training with proper RCA/Non-RCA separation"
    )
    parser.add_argument(
        '--input-path',
        required=True,
        help='Path to input Parquet file'
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
        '--no-vessel-separation',
        action='store_true',
        help='Generate prompts with full vessel aggregation (both L/R vessels together)'
    )

    args = parser.parse_args()

    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = create_default_config()

    # Override vessel_separation if specified on command line
    if args.no_vessel_separation:
        config['vessel_separation'] = False

    # Process dataset
    process_dataset_multiprompt(args.input_path, args.output_dir, config)


if __name__ == "__main__":
    main()