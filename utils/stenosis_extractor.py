#!/usr/bin/env python3
"""
Stenosis Extractor - NLP-based extraction of stenosis percentages from reports.

This module extracts stenosis information per artery from generated and target reports
to enable artery-specific loss weighting in captioning tasks.
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class StenosisInfo:
    """Information about stenosis in a specific artery."""
    artery: str
    percentage: float
    severity: str  # 'none', 'mild', 'moderate', 'severe', 'critical'
    has_calcification: bool = False
    has_bifurcation: bool = False
    is_cto: bool = False
    is_restenosis: bool = False


# Standardized artery name mappings
ARTERY_NAME_MAPPINGS = {
    # Left coronary system
    'lmca': 'left_main',
    'left main coronary artery': 'left_main',
    'left main': 'left_main',

    'proximal lad': 'prox_lad',
    'prox lad': 'prox_lad',
    'mid lad': 'mid_lad',
    'distal lad': 'dist_lad',
    'dist lad': 'dist_lad',

    'd1 branch': 'd1',
    'd1': 'd1',
    'd2 branch': 'd2',
    'd2': 'd2',

    'proximal lcx': 'prox_lcx',
    'prox lcx': 'prox_lcx',
    'distal lcx': 'dist_lcx',
    'dist lcx': 'dist_lcx',

    'om1': 'om1',
    'om2': 'om2',
    'ramus': 'ramus',

    'left posterolateral branch': 'lvp',
    'lvp': 'lvp',

    # Right coronary system
    'proximal rca': 'prox_rca',
    'prox rca': 'prox_rca',
    'mid rca': 'mid_rca',
    'distal rca': 'dist_rca',
    'dist rca': 'dist_rca',

    'pda': 'pda',
    'the pda': 'pda',
    'left pda': 'pda',  # Dominance-dependent

    'posterolateral branch': 'posterolateral',
    'the posterolateral branch': 'posterolateral',
}


# Severity classification
def classify_severity(percentage: float) -> str:
    """Classify stenosis percentage into severity category."""
    if percentage == 0 or percentage < 1:
        return 'none'
    elif percentage < 50:
        return 'mild'
    elif percentage < 70:
        return 'moderate'
    elif percentage < 90:
        return 'severe'
    else:
        return 'critical'


def _normalize_numeric_tokens(text: str) -> str:
    """Compress whitespace around decimal points and percent signs."""
    # Collapse patterns like "80. 0" -> "80.0"
    text = re.sub(r"(\d+)\s*\.\s*(\d+)", r"\1.\2", text)
    # Remove stray spaces before the percent sign: "80.0 %" -> "80.0%"
    text = re.sub(r"(\d)\s*%", r"\1%", text)
    return text


def normalize_artery_name(artery_text: str) -> Optional[str]:
    """
    Normalize artery name to standardized format.

    Args:
        artery_text: Raw artery name from report

    Returns:
        Standardized artery name or None if not recognized
    """
    artery_lower = artery_text.strip().lower()
    artery_lower = re.sub(r"[()]", " ", artery_lower)
    artery_lower = re.sub(r"\s+", " ", artery_lower).strip()

    # Try exact match first
    if artery_lower in ARTERY_NAME_MAPPINGS:
        return ARTERY_NAME_MAPPINGS[artery_lower]

    # Try partial matching
    for key, value in ARTERY_NAME_MAPPINGS.items():
        if key in artery_lower or artery_lower in key:
            return value

    return None


def extract_stenosis_from_report(report: str) -> Dict[str, StenosisInfo]:
    """
    Extract stenosis information from a medical report.

    The report format from generate_dataset.py is:
    "the <artery> has <severity> stenosis (~<percentage>%), [and <modifiers>]."

    Examples:
        - "the mid LAD has severe stenosis (~70.0%), moderate calcifications, and bifurcation lesion."
        - "the proximal RCA has no significant stenosis."
        - "the mid RCA has critical stenosis (~100.0%), and minimal calcifications."

    Args:
        report: Full medical report text

    Returns:
        Dictionary mapping standardized artery names to StenosisInfo objects
    """
    stenosis_dict: Dict[str, StenosisInfo] = {}

    # Pattern 1: Stenosis with percentage
    # Matches: "the mid LAD has severe stenosis (~70.0%)"
    pattern_with_pct = re.compile(
        r'(?:the\s+)?([a-zA-Z0-9\s()_-]+?)\s+has\s+(?:(mild|moderate|severe|critical)\s+)?stenosis\s*\(\s*~?\s*(\d+(?:\s*\.\s*\d+)?)\s*%\s*\)',
        re.IGNORECASE
    )

    # Pattern 2: "no significant stenosis"
    # Matches: "the proximal LAD has no significant stenosis"
    pattern_no_stenosis = re.compile(
        r'(?:the\s+)?([a-zA-Z0-9\s()_-]+?)\s+has\s+no\s+significant\s+stenosis',
        re.IGNORECASE
    )

    # Pattern 3: Restenosis
    # Matches: "the mid LAD has in-stent restenosis (mild stenosis (~40.0%))"
    pattern_restenosis = re.compile(
        r'(?:the\s+)?([a-zA-Z0-9\s()_-]+?)\s+has\s+(?:in-stent\s+)?restenosis\s+\(.*?stenosis\s+\(\s*~?\s*(\d+(?:\s*\.\s*\d+)?)\s*%\s*\)',
        re.IGNORECASE
    )

    # Pattern 4: CTO (Chronic Total Occlusion)
    # Matches: "the mid RCA is 100% blocked and is a CTO"
    pattern_cto = re.compile(
        r'(?:the\s+)?([a-zA-Z0-9\s()_-]+?)\s+is\s+100%\s+blocked\s+and\s+is\s+a\s+CTO',
        re.IGNORECASE
    )

    # Normalize numeric tokens once for reliable pattern matching
    normalized_report = _normalize_numeric_tokens(report)

    # Split into candidate sentences using newlines and sentence punctuation
    raw_segments = re.split(r'[\n\r]+', normalized_report)
    candidate_sentences: List[str] = []
    for segment in raw_segments:
        segment = segment.strip()
        if not segment:
            continue
        # Further split on period/semicolon when followed by whitespace; keeps decimals intact
        splits = re.split(r'(?<=[.;])\s+', segment)
        for item in splits:
            item = item.strip()
            if item:
                candidate_sentences.append(item)

    # Process each candidate sentence of the report
    for line in candidate_sentences:
        normalized_line = _normalize_numeric_tokens(line)

        # Check for CTO first (100% stenosis)
        cto_match = pattern_cto.search(normalized_line)
        if cto_match:
            artery_text = cto_match.group(1).strip()
            artery_name = normalize_artery_name(artery_text)
            if artery_name:
                stenosis_dict[artery_name] = StenosisInfo(
                    artery=artery_name,
                    percentage=100.0,
                    severity='critical',
                    is_cto=True
                )
                continue

        # Check for restenosis
        restenosis_match = pattern_restenosis.search(normalized_line)
        if restenosis_match:
            artery_text = restenosis_match.group(1).strip()
            percentage = float(restenosis_match.group(2).replace(' ', ''))
            artery_name = normalize_artery_name(artery_text)
            if artery_name:
                stenosis_dict[artery_name] = StenosisInfo(
                    artery=artery_name,
                    percentage=percentage,
                    severity=classify_severity(percentage),
                    is_restenosis=True,
                    has_calcification='calcification' in line.lower(),
                    has_bifurcation='bifurcation' in line.lower()
                )
                continue

        # Check for stenosis with percentage
        pct_match = pattern_with_pct.search(normalized_line)
        if pct_match:
            artery_text = pct_match.group(1).strip()
            severity_text = pct_match.group(2)
            percentage = float(pct_match.group(3).replace(' ', ''))
            artery_name = normalize_artery_name(artery_text)

            if artery_name:
                stenosis_dict[artery_name] = StenosisInfo(
                    artery=artery_name,
                    percentage=percentage,
                    severity=severity_text.lower() if severity_text else classify_severity(percentage),
                    has_calcification='calcification' in line.lower(),
                    has_bifurcation='bifurcation' in line.lower()
                )
                continue

        # Check for "no significant stenosis"
        no_stenosis_match = pattern_no_stenosis.search(normalized_line)
        if no_stenosis_match:
            artery_text = no_stenosis_match.group(1).strip()
            artery_name = normalize_artery_name(artery_text)
            if artery_name:
                stenosis_dict[artery_name] = StenosisInfo(
                    artery=artery_name,
                    percentage=0.0,
                    severity='none'
                )

    return stenosis_dict


def compare_stenosis_reports(
    generated_report: str,
    target_report: str
) -> Tuple[Dict[str, StenosisInfo], Dict[str, StenosisInfo], Dict[str, float]]:
    """
    Compare stenosis information between generated and target reports.

    Args:
        generated_report: Model-generated report text
        target_report: Ground truth report text

    Returns:
        Tuple of:
        - Generated stenosis dict
        - Target stenosis dict
        - Per-artery errors dict (target_pct - generated_pct)
    """
    generated = extract_stenosis_from_report(generated_report)
    target = extract_stenosis_from_report(target_report)

    # Compute per-artery errors
    errors = {}
    all_arteries = set(generated.keys()) | set(target.keys())

    for artery in all_arteries:
        gen_pct = generated.get(artery, StenosisInfo(artery, 0.0, 'none')).percentage
        tgt_pct = target.get(artery, StenosisInfo(artery, 0.0, 'none')).percentage
        errors[artery] = tgt_pct - gen_pct

    return generated, target, errors


def compute_stenosis_metrics(
    generated_report: str,
    target_report: str
) -> Dict[str, float]:
    """
    Compute stenosis-specific metrics between generated and target reports.

    Metrics:
    - mae: Mean Absolute Error across all arteries
    - rmse: Root Mean Squared Error
    - critical_recall: Recall for ≥70% stenoses
    - critical_precision: Precision for ≥70% stenoses
    - severity_accuracy: Fraction of arteries with correct severity class

    Args:
        generated_report: Model-generated report
        target_report: Ground truth report

    Returns:
        Dictionary of metric name -> value
    """
    gen, tgt, errors = compare_stenosis_reports(generated_report, target_report)

    if not errors:
        return {
            'mae': 0.0,
            'rmse': 0.0,
            'critical_recall': 0.0,
            'critical_precision': 0.0,
            'severity_accuracy': 0.0
        }

    # MAE and RMSE
    error_values = list(errors.values())
    mae = np.mean(np.abs(error_values))
    rmse = np.sqrt(np.mean(np.square(error_values)))

    # Critical stenosis metrics (≥70%)
    critical_threshold = 70.0

    tgt_critical_arteries = {a for a, info in tgt.items() if info.percentage >= critical_threshold}
    gen_critical_arteries = {a for a, info in gen.items() if info.percentage >= critical_threshold}

    if len(tgt_critical_arteries) > 0:
        true_positives = len(tgt_critical_arteries & gen_critical_arteries)
        critical_recall = true_positives / len(tgt_critical_arteries)
    else:
        critical_recall = 1.0  # No critical stenoses to find

    if len(gen_critical_arteries) > 0:
        true_positives = len(tgt_critical_arteries & gen_critical_arteries)
        critical_precision = true_positives / len(gen_critical_arteries)
    else:
        critical_precision = 1.0 if len(tgt_critical_arteries) == 0 else 0.0

    # Severity class accuracy
    all_arteries = set(gen.keys()) | set(tgt.keys())
    severity_correct = 0
    for artery in all_arteries:
        gen_severity = gen.get(artery, StenosisInfo(artery, 0.0, 'none')).severity
        tgt_severity = tgt.get(artery, StenosisInfo(artery, 0.0, 'none')).severity
        if gen_severity == tgt_severity:
            severity_correct += 1

    severity_accuracy = severity_correct / len(all_arteries) if all_arteries else 0.0

    return {
        'mae': mae,
        'rmse': rmse,
        'critical_recall': critical_recall,
        'critical_precision': critical_precision,
        'severity_accuracy': severity_accuracy
    }


def get_stenosis_feature_vector(report: str, artery_order: Optional[List[str]] = None) -> np.ndarray:
    """
    Convert report to fixed-length feature vector of stenosis percentages.

    This can be used as input to auxiliary prediction heads or for computing MSE loss.

    Args:
        report: Medical report text
        artery_order: List of artery names in desired order. If None, uses default order.

    Returns:
        Numpy array of shape (num_arteries,) with stenosis percentages

    Note:
        If report is malformed/unparseable, returns all zeros (no stenosis detected).
        This is appropriate for early training when generated text is garbage.
    """
    if artery_order is None:
        # Default order matching typical vessel sequence
        artery_order = [
            'left_main', 'prox_lad', 'mid_lad', 'dist_lad', 'd1', 'd2',
            'prox_lcx', 'dist_lcx', 'om1', 'om2', 'ramus', 'lvp',
            'prox_rca', 'mid_rca', 'dist_rca', 'pda', 'posterolateral'
        ]

    try:
        stenosis_dict = extract_stenosis_from_report(report)
    except Exception:
        # If extraction completely fails (malformed text), return all zeros
        return np.zeros(len(artery_order), dtype=np.float32)

    feature_vector = np.zeros(len(artery_order), dtype=np.float32)
    for i, artery in enumerate(artery_order):
        if artery in stenosis_dict:
            feature_vector[i] = stenosis_dict[artery].percentage
        # else: remains 0.0 (unknown/not mentioned)

    return feature_vector


# ──────────────────────────────────────────────────────────────────────────
# Testing and Validation Functions
# ──────────────────────────────────────────────────────────────────────────

def test_extractor_with_samples():
    """Test the extractor with sample reports from the dataset."""

    # Sample reports from dataset
    test_reports = [
        # Severe stenosis case
        """the Left Main Coronary Artery (LMCA) has no significant stenosis.
the proximal LAD has no significant stenosis.
the mid LAD has severe stenosis (~70.0%), moderate calcifications, and bifurcation lesion (Medina Bifurcation 1.1.0).
the distal LAD has severe stenosis (~80.0%), and minimal calcifications.
D1 branch has severe stenosis (~80.0%), and minimal calcifications.
D2 branch has no significant stenosis.
the proximal LCX has no significant stenosis.
the distal LCX has no significant stenosis.
OM1 has no significant stenosis.
OM2 has severe stenosis (~70.0%), and minimal calcifications.
Ramus has no significant stenosis.
left posterolateral branch has no significant stenosis.
The coronary circulation is right dominant.""",

        # Critical stenosis with CTO
        """the proximal RCA has critical stenosis (~95.0%), and minimal calcifications.
the mid RCA has critical stenosis (~100.0%), and minimal calcifications.
the distal RCA has no significant stenosis.
the PDA has no significant stenosis.
the posterolateral branch has no significant stenosis.
The coronary circulation is right dominant.""",

        # Normal report
        """the Left Main Coronary Artery (LMCA) has no significant stenosis.
the proximal LAD has no significant stenosis.
the mid LAD has no significant stenosis.
the distal LAD has no significant stenosis.
D1 branch has no significant stenosis.
D2 branch has no significant stenosis.
the proximal LCX has no significant stenosis.
the distal LCX has no significant stenosis.
OM1 has no significant stenosis.
OM2 has no significant stenosis.
Ramus has no significant stenosis.
left posterolateral branch has no significant stenosis.
The coronary circulation is left dominant."""
    ]

    print("="*80)
    print("STENOSIS EXTRACTOR TEST")
    print("="*80)

    for i, report in enumerate(test_reports, 1):
        print(f"\nTest Report {i}:")
        print("-"*80)

        stenoses = extract_stenosis_from_report(report)

        # Show only arteries with stenosis > 0
        significant_stenoses = {k: v for k, v in stenoses.items() if v.percentage > 0}

        if significant_stenoses:
            print(f"Found {len(significant_stenoses)} arteries with stenosis:")
            for artery, info in sorted(significant_stenoses.items(),
                                      key=lambda x: x[1].percentage, reverse=True):
                flags = []
                if info.has_calcification:
                    flags.append("calcif")
                if info.has_bifurcation:
                    flags.append("bifurc")
                if info.is_cto:
                    flags.append("CTO")

                flag_str = f" [{', '.join(flags)}]" if flags else ""
                print(f"  {artery:15s}: {info.percentage:5.1f}% ({info.severity}){flag_str}")
        else:
            print("No significant stenoses found (all normal)")

    # Test comparison functionality
    print("\n" + "="*80)
    print("COMPARISON TEST")
    print("="*80)

    target = test_reports[0]  # Severe stenoses
    generated = """the Left Main Coronary Artery (LMCA) has no significant stenosis.
the proximal LAD has no significant stenosis.
the mid LAD has mild stenosis (~20.0%).
the distal LAD has no significant stenosis.
D1 branch has no significant stenosis.
D2 branch has no significant stenosis.
the proximal LCX has no significant stenosis."""

    metrics = compute_stenosis_metrics(generated, target)

    print("\nTarget: Multiple severe stenoses (70-80%)")
    print("Generated: Mostly normal with one mild (~20%)")
    print("\nMetrics:")
    for metric, value in metrics.items():
        print(f"  {metric:20s}: {value:.4f}")


if __name__ == "__main__":
    test_extractor_with_samples()
