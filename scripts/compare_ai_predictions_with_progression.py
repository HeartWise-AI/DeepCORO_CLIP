#!/usr/bin/env python3
"""
Compare AI Stenosis Predictions with Ground Truth Progression

Matches inference predictions to study pairs and compares AI-predicted
stenosis changes between "changed" and "unchanged" groups.

Usage:
    python scripts/compare_ai_predictions_with_progression.py
"""

from __future__ import annotations

import ast
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Stenosis columns in inference file
VESSELS = [
    "prox_rca", "mid_rca", "dist_rca", "pda", "posterolateral",
    "left_main", "prox_lad", "mid_lad", "dist_lad",
    "D1", "D2", "prox_lcx", "mid_lcx", "dist_lcx",
    "om1", "om2", "bx", "lvp"
]


def extract_video_paths(video_name_str: str) -> List[str]:
    """Extract individual video paths from the inference video_name column."""
    if pd.isna(video_name_str):
        return []
    try:
        # Parse the list string
        videos = ast.literal_eval(video_name_str)
        # Filter out PAD entries
        return [v for v in videos if v != 'PAD' and isinstance(v, str)]
    except:
        return []


def extract_study_uid_from_path(video_path: str) -> Optional[str]:
    """Extract StudyInstanceUID from video path."""
    # Pattern: .../2.16.124.113611.1.118.1.1.XXXXXX_...
    # or: .../2.16.124.113611.1.118.1.1.XXXXXX/...
    match = re.search(r'(2\.16\.124\.113611\.1\.118\.1\.1\.\d+)', video_path)
    if match:
        return match.group(1)
    return None


def build_study_predictions_map(
    inference_df: pd.DataFrame
) -> Dict[str, Dict[str, float]]:
    """
    Build a mapping from StudyInstanceUID to AI predictions.

    Returns:
        {study_uid: {vessel: pred_value, ...}, ...}
    """
    study_preds = {}

    for _, row in inference_df.iterrows():
        video_paths = extract_video_paths(row['video_name'])
        if not video_paths:
            continue

        # Get study UID from first video
        study_uid = extract_study_uid_from_path(video_paths[0])
        if not study_uid:
            continue

        # Extract predictions for each vessel
        preds = {}
        for vessel in VESSELS:
            pred_col = f"{vessel}_stenosis_pred"
            if pred_col in row and pd.notna(row[pred_col]):
                preds[vessel] = float(row[pred_col])

        if preds:
            study_preds[study_uid] = preds

    return study_preds


def compute_ai_change_metrics(
    current_preds: Dict[str, float],
    prior_preds: Dict[str, float],
    changed_vessels_json: str,
) -> Dict:
    """
    Compute AI-based change metrics between two studies.

    Returns dict with:
        - ai_max_delta: Maximum AI-predicted change across all vessels
        - ai_mean_delta: Mean AI-predicted change across all vessels
        - ai_changed_vessel_delta: Mean AI delta for vessels that actually changed
        - ai_crosses_50: Number of vessels AI predicts crossing 50%
    """
    results = {
        'ai_max_delta': None,
        'ai_mean_delta': None,
        'ai_changed_vessel_delta': None,
        'ai_num_progressed': 0,
        'ai_crosses_50': 0,
    }

    # Compute deltas for all vessels
    deltas = []
    for vessel in VESSELS:
        if vessel in current_preds and vessel in prior_preds:
            delta = current_preds[vessel] - prior_preds[vessel]
            deltas.append((vessel, delta))

            # Check if AI predicts crossing 50%
            if prior_preds[vessel] < 50 and current_preds[vessel] >= 50:
                results['ai_crosses_50'] += 1

            # Count progressions (>20% increase)
            if delta >= 20:
                results['ai_num_progressed'] += 1

    if deltas:
        all_deltas = [d for _, d in deltas]
        results['ai_max_delta'] = max(all_deltas)
        results['ai_mean_delta'] = np.mean(all_deltas)

    # Get AI delta specifically for vessels that actually changed
    if changed_vessels_json and changed_vessels_json != '[]':
        try:
            changed = json.loads(changed_vessels_json)
            changed_vessel_names = [v.get('vessel') for v in changed]

            changed_deltas = []
            for vessel, delta in deltas:
                if vessel in changed_vessel_names:
                    changed_deltas.append(delta)

            if changed_deltas:
                results['ai_changed_vessel_delta'] = np.mean(changed_deltas)
        except:
            pass

    return results


def main():
    # Paths
    inference_path = Path("/media/data1/ravram/stenosis70/inference_predictions_best_epoch_-1.csv")
    scores_path = Path("/volume/DeepCORO_CLIP/output_dataset/stenosis_progression/stenosis_change_scores_test.csv")
    output_path = Path("/volume/DeepCORO_CLIP/output_dataset/stenosis_progression/ai_predictions_comparison.csv")

    # Load data
    logger.info("Loading inference predictions...")
    inference_df = pd.read_csv(inference_path)
    logger.info(f"Loaded {len(inference_df)} inference rows")

    logger.info("Loading change scores...")
    scores_df = pd.read_csv(scores_path)
    logger.info(f"Loaded {len(scores_df)} studies with change scores")

    # Build study -> predictions mapping
    logger.info("Building study predictions map...")
    study_preds = build_study_predictions_map(inference_df)
    logger.info(f"Mapped predictions for {len(study_preds)} studies")

    # Match studies to our change classification
    results = []
    matched_count = 0
    both_matched = 0

    for _, row in scores_df.iterrows():
        study_uid = row['StudyInstanceUID']
        prior_uid = row['prior_study_uid']

        current_preds = study_preds.get(study_uid, {})
        prior_preds = study_preds.get(prior_uid, {}) if pd.notna(prior_uid) else {}

        has_current = len(current_preds) > 0
        has_prior = len(prior_preds) > 0

        if has_current:
            matched_count += 1
        if has_current and has_prior:
            both_matched += 1

        # Compute AI change metrics
        ai_metrics = {}
        if has_current and has_prior:
            ai_metrics = compute_ai_change_metrics(
                current_preds, prior_preds, row.get('changed_vessels', '[]')
            )

        # Store all current predictions
        current_pred_cols = {}
        for vessel in VESSELS:
            current_pred_cols[f'current_{vessel}_pred'] = current_preds.get(vessel)

        # Store all prior predictions
        prior_pred_cols = {}
        for vessel in VESSELS:
            prior_pred_cols[f'prior_{vessel}_pred'] = prior_preds.get(vessel)

        results.append({
            'StudyInstanceUID': study_uid,
            'prior_study_uid': prior_uid,
            'change_status': row['change_status'],
            'severity_level': row['severity_level'],
            'has_current_pred': has_current,
            'has_prior_pred': has_prior,
            **ai_metrics,
            **current_pred_cols,
            **prior_pred_cols,
        })

    results_df = pd.DataFrame(results)

    # Save full results
    results_df.to_csv(output_path, index=False)
    logger.info(f"Saved comparison results to {output_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("AI PREDICTIONS vs GROUND TRUTH CHANGE COMPARISON")
    print("=" * 70)

    print(f"\nMatching Statistics:")
    print(f"  Total studies: {len(scores_df)}")
    print(f"  Studies with current AI predictions: {matched_count}")
    print(f"  Studies with BOTH current + prior AI predictions: {both_matched}")

    # Filter to studies with both predictions
    paired = results_df[results_df['has_current_pred'] & results_df['has_prior_pred']]

    if len(paired) == 0:
        print("\nNo study pairs found with AI predictions for both current and prior studies.")
        return

    print(f"\n{'='*70}")
    print("COMPARISON: Changed vs Unchanged (studies with both AI predictions)")
    print("=" * 70)

    for status in ['changed', 'unchanged']:
        subset = paired[paired['change_status'] == status]
        n = len(subset)

        if n == 0:
            continue

        print(f"\n{status.upper()} Studies (n={n}):")

        # AI max delta
        valid_max = subset['ai_max_delta'].dropna()
        if len(valid_max) > 0:
            print(f"  AI Max Delta (stenosis %):")
            print(f"    Mean:   {valid_max.mean():.1f}%")
            print(f"    Median: {valid_max.median():.1f}%")
            print(f"    Std:    {valid_max.std():.1f}%")
            print(f"    IQR:    [{valid_max.quantile(0.25):.1f}%, {valid_max.quantile(0.75):.1f}%]")

        # AI mean delta
        valid_mean = subset['ai_mean_delta'].dropna()
        if len(valid_mean) > 0:
            print(f"  AI Mean Delta (all vessels):")
            print(f"    Mean:   {valid_mean.mean():.1f}%")
            print(f"    Median: {valid_mean.median():.1f}%")

        # AI progression count
        valid_prog = subset['ai_num_progressed'].dropna()
        if len(valid_prog) > 0:
            print(f"  AI-Predicted Progressions (>=20% increase):")
            print(f"    Mean:   {valid_prog.mean():.2f} vessels")

        # AI 50% crossing
        valid_cross = subset['ai_crosses_50'].dropna()
        if len(valid_cross) > 0:
            has_crossing = (valid_cross > 0).sum()
            print(f"  AI-Predicted 50% Crossings:")
            print(f"    Studies with crossing: {has_crossing} ({100*has_crossing/n:.1f}%)")
            print(f"    Mean crossings/study:  {valid_cross.mean():.2f}")

    # Statistical test
    print(f"\n{'='*70}")
    print("STATISTICAL COMPARISON")
    print("=" * 70)

    changed = paired[paired['change_status'] == 'changed']['ai_max_delta'].dropna()
    unchanged = paired[paired['change_status'] == 'unchanged']['ai_max_delta'].dropna()

    if len(changed) > 0 and len(unchanged) > 0:
        from scipy import stats

        # Mann-Whitney U test (non-parametric)
        u_stat, p_value = stats.mannwhitneyu(changed, unchanged, alternative='greater')

        print(f"\nMann-Whitney U Test (AI Max Delta: changed > unchanged):")
        print(f"  U-statistic: {u_stat:.1f}")
        print(f"  p-value: {p_value:.4f}")

        if p_value < 0.05:
            print(f"  Result: SIGNIFICANT (p < 0.05)")
            print(f"  The AI predicts significantly larger stenosis changes in 'changed' studies")
        else:
            print(f"  Result: Not significant at p < 0.05")

    # Vessel-level comparison
    print(f"\n{'='*70}")
    print("PER-VESSEL AI DELTA COMPARISON (changed vs unchanged)")
    print("=" * 70)
    print(f"\n{'Vessel':<15} | {'Changed Mean':<12} | {'Unchanged Mean':<14} | {'Diff':<8}")
    print("-" * 60)

    for vessel in VESSELS:
        current_col = f'current_{vessel}_pred'
        prior_col = f'prior_{vessel}_pred'

        if current_col in paired.columns and prior_col in paired.columns:
            # Compute delta
            delta_col = paired[current_col].astype(float) - paired[prior_col].astype(float)

            changed_delta = delta_col[paired['change_status'] == 'changed'].dropna()
            unchanged_delta = delta_col[paired['change_status'] == 'unchanged'].dropna()

            if len(changed_delta) > 5 and len(unchanged_delta) > 5:
                c_mean = changed_delta.mean()
                u_mean = unchanged_delta.mean()
                diff = c_mean - u_mean

                print(f"{vessel:<15} | {c_mean:>10.1f}% | {u_mean:>12.1f}% | {diff:>+6.1f}%")


if __name__ == "__main__":
    main()
