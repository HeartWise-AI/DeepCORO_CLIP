#!/usr/bin/env python3
"""
Compute Clinical Change Scores for Stenosis Progression Studies

Uses CLIP visual similarity combined with ground truth stenosis data
to compute clinically meaningful change scores.

Filters for TEST split studies with quantitative comparison only.

Usage:
    python scripts/compute_study_change_scores.py \
        --checkpoint /path/to/clip_checkpoint.pt \
        --output-dir output_dataset/stenosis_progression
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.video_encoder import VideoEncoder
from utils.video import load_video
from utils.change_scoring import (
    compute_change_score,
    compute_clip_similarity,
    ChangeScore,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# CLIP MODEL LOADING
# =============================================================================

def load_clip_encoder(
    checkpoint_path: str,
    device: torch.device,
) -> VideoEncoder:
    """Load CLIP video encoder from checkpoint."""
    logger.info(f"Loading CLIP encoder from {checkpoint_path}")

    # Initialize encoder with same params as checkpoint
    # This checkpoint was trained with aggregator_depth=1
    video_encoder = VideoEncoder(
        backbone='mvit',
        num_frames=16,
        pretrained=False,  # Will load from checkpoint
        freeze_ratio=0.87,
        dropout=0.158,
        num_heads=8,
        aggregator_depth=1,  # Checkpoint has 1 block
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    video_encoder.load_state_dict(checkpoint["video_encoder"])

    video_encoder = video_encoder.to(device).float()
    video_encoder.eval()

    logger.info("CLIP encoder loaded successfully")
    return video_encoder


# =============================================================================
# VIDEO EMBEDDING EXTRACTION
# =============================================================================

def extract_video_embedding(
    video_path: str,
    encoder: VideoEncoder,
    device: torch.device,
    num_frames: int = 16,
) -> Optional[torch.Tensor]:
    """Extract CLIP embedding from a single video."""
    if not os.path.exists(video_path):
        return None

    try:
        # Load video
        video = load_video(
            video_path,
            n_frames=num_frames,
            resize=224,
            normalize=True,
            stride=2,
        )

        # Convert to tensor [1, 1, F, H, W, C] -> model expects this format
        video_tensor = torch.from_numpy(video).float()
        video_tensor = video_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and num_videos dims
        video_tensor = video_tensor.to(device)

        # Extract embedding
        with torch.no_grad():
            output = encoder(video_tensor)
            if isinstance(output, dict):
                embedding = output.get('video_embeds', output.get('pooled_output'))
            else:
                embedding = output

        return embedding.squeeze().cpu()

    except Exception as e:
        logger.warning(f"Error extracting embedding from {video_path}: {e}")
        return None


def extract_study_embedding(
    video_paths: List[str],
    encoder: VideoEncoder,
    device: torch.device,
    structure_mapping: Dict[str, Dict],
    target_structure: str,
) -> Optional[torch.Tensor]:
    """
    Extract mean-pooled embedding from study videos matching target structure.

    Args:
        video_paths: List of video file paths for the study
        encoder: CLIP video encoder
        device: torch device
        structure_mapping: Mapping of video path -> structure info
        target_structure: "Left Coronary" or "Right Coronary"

    Returns:
        Mean-pooled embedding tensor [512] or None
    """
    embeddings = []

    for video_path in video_paths:
        # Check if video matches target structure
        if video_path in structure_mapping:
            video_structure = structure_mapping[video_path].get("structure_name", "")
            if video_structure != target_structure:
                continue

        # Extract embedding
        embedding = extract_video_embedding(video_path, encoder, device)
        if embedding is not None:
            embeddings.append(embedding)

    if not embeddings:
        return None

    # Mean pool across videos
    stacked = torch.stack(embeddings)
    mean_embedding = stacked.mean(dim=0)
    return mean_embedding


# =============================================================================
# DATA LOADING
# =============================================================================

def load_video_structure_mapping(csv_path: Path) -> Dict[str, Dict]:
    """Load mapping of video filename -> structure info."""
    logger.info("Loading video structure mapping...")

    df = pd.read_csv(
        csv_path,
        sep='α',
        engine='python',
        usecols=['FileName', 'main_structure_class', 'main_structure_name']
    )

    mapping = {}
    for _, row in df.iterrows():
        filename = row['FileName']
        if pd.notna(filename):
            mapping[filename] = {
                "structure_class": row['main_structure_class'],
                "structure_name": row['main_structure_name'] if pd.notna(row['main_structure_name']) else "Unknown"
            }

    logger.info(f"Loaded structure info for {len(mapping)} videos")
    return mapping


def load_classification_data(
    csv_path: Path,
    original_data_path: Path,
    split: str = "test",
) -> pd.DataFrame:
    """
    Load classification data filtered for specific split with quantitative comparison.

    Args:
        csv_path: Path to stenosis_change_classification.csv
        original_data_path: Path to original dataset with Split column
        split: Which split to use ("train", "val", "test")

    Returns:
        Filtered DataFrame
    """
    logger.info(f"Loading classification data for {split} split...")

    # Load classification data
    df = pd.read_csv(csv_path)

    # Load original data for Split info
    orig_df = pd.read_csv(
        original_data_path,
        sep='α',
        engine='python',
        usecols=['StudyInstanceUID', 'Split']
    )

    # Get unique study splits
    study_splits = orig_df.groupby('StudyInstanceUID')['Split'].first().reset_index()

    # Merge
    df = df.merge(study_splits, on='StudyInstanceUID', how='left')

    # Filter for target split and quantitative comparison
    filtered = df[
        (df['Split'] == split) &
        (df['change_reason'].str.contains('quantitative', na=False))
    ].copy()

    logger.info(f"Filtered to {len(filtered)} {split} studies with quantitative comparison")
    logger.info(f"  Changed: {(filtered['change_status'] == 'changed').sum()}")
    logger.info(f"  Unchanged: {(filtered['change_status'] == 'unchanged').sum()}")

    return filtered


# =============================================================================
# VESSEL TO STRUCTURE MAPPING
# =============================================================================

LEFT_CORONARY_VESSELS = {
    "prox_lad", "mid_lad", "dist_lad",
    "prox_lcx", "mid_lcx", "dist_lcx",
    "left_main", "D1", "D2", "D3",
    "om1", "om2", "om3", "S1", "bx", "lvp",
}

RIGHT_CORONARY_VESSELS = {
    "prox_rca", "mid_rca", "dist_rca",
    "pda", "posterolateral", "right_marginal",
}


def get_primary_changed_structure(changed_vessels_json: str) -> str:
    """Determine which coronary structure has the most changes."""
    if not changed_vessels_json or changed_vessels_json == '[]':
        return "Left Coronary"  # Default

    try:
        vessels = json.loads(changed_vessels_json)
    except:
        return "Left Coronary"

    left_count = sum(1 for v in vessels if v.get('vessel', '') in LEFT_CORONARY_VESSELS)
    right_count = sum(1 for v in vessels if v.get('vessel', '') in RIGHT_CORONARY_VESSELS)

    return "Right Coronary" if right_count > left_count else "Left Coronary"


# =============================================================================
# MAIN SCORING FUNCTION
# =============================================================================

def compute_all_scores(
    df: pd.DataFrame,
    encoder: VideoEncoder,
    device: torch.device,
    structure_mapping: Dict[str, Dict],
    clip_weight: float = 0.4,
) -> pd.DataFrame:
    """
    Compute change scores for all study pairs in DataFrame.

    Args:
        df: DataFrame with study pairs
        encoder: CLIP video encoder
        device: torch device
        structure_mapping: Video -> structure mapping
        clip_weight: Weight for CLIP component

    Returns:
        DataFrame with added score columns
    """
    results = []

    # Build lookup for prior studies
    study_lookup = df.set_index('StudyInstanceUID').to_dict('index')

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Computing scores"):
        study_uid = row['StudyInstanceUID']
        prior_uid = row['prior_study_uid']

        # Initialize scores
        clip_similarity = None
        clip_change_score = None

        # Get video paths
        current_videos = row['FileName'].split('|') if row['FileName'] else []

        # Get prior study videos
        prior_videos = []
        if prior_uid and prior_uid in study_lookup:
            prior_row = study_lookup[prior_uid]
            prior_videos = prior_row.get('FileName', '').split('|') if prior_row.get('FileName') else []

        # Determine target structure based on changed vessels
        target_structure = get_primary_changed_structure(row['changed_vessels'])

        # Extract embeddings if we have videos for both studies
        if current_videos and prior_videos:
            current_embedding = extract_study_embedding(
                current_videos, encoder, device, structure_mapping, target_structure
            )
            prior_embedding = extract_study_embedding(
                prior_videos, encoder, device, structure_mapping, target_structure
            )

            if current_embedding is not None and prior_embedding is not None:
                clip_similarity = compute_clip_similarity(prior_embedding, current_embedding)
                clip_change_score = (1.0 - clip_similarity) * 10.0

        # Compute full change score
        change_score = compute_change_score(
            row['changed_vessels'],
            clip_similarity=clip_similarity,
            clip_weight=clip_weight,
        )

        results.append({
            'StudyInstanceUID': study_uid,
            'clip_similarity': clip_similarity,
            'clip_change_score': clip_change_score,
            'stenosis_change_score': change_score.stenosis_change_score,
            'combined_change_score': change_score.combined_change_score,
            'severity_level': change_score.severity_level,
            'severity_description': change_score.severity_description,
            'pci_warranting': change_score.pci_warranting,
            'num_pci_warranting': change_score.num_pci_warranting,
            'pci_warranting_vessels': '|'.join(change_score.pci_warranting_vessels),
            'max_delta': change_score.max_delta,
            'num_progressed_vessels': change_score.num_progressed_vessels,
        })

    # Create results DataFrame and merge with original
    results_df = pd.DataFrame(results)
    merged = df.merge(results_df, on='StudyInstanceUID', how='left')

    return merged


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compute clinical change scores for stenosis progression studies"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/media/data1/models/DeepCoro_CLIP/8av1xygm_20250605-083820_best_single_video/checkpoints/best_model_epoch_9.pt"),
        help="Path to CLIP checkpoint",
    )
    parser.add_argument(
        "--classification-csv",
        type=Path,
        default=Path("output_dataset/stenosis_progression/stenosis_change_classification.csv"),
        help="Path to classification CSV",
    )
    parser.add_argument(
        "--original-data",
        type=Path,
        default=Path("/media/data1/ravram/DeepCORO_CLIP_ENCODER/datasets/reports_with_alpha_separator_with_Calcifc_Stenosis_IFR_20250601_RCA_LCA_merged_with_left_dominance_dependent_vessels.csv"),
        help="Path to original dataset with Split column",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("output_dataset/stenosis_progression"),
        help="Output directory",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which data split to process",
    )
    parser.add_argument(
        "--clip-weight",
        type=float,
        default=0.4,
        help="Weight for CLIP component in combined score (0-1)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load CLIP encoder
    encoder = load_clip_encoder(str(args.checkpoint), device)

    # Load video structure mapping
    structure_mapping = load_video_structure_mapping(args.original_data)

    # Load and filter classification data
    df = load_classification_data(
        args.classification_csv,
        args.original_data,
        split=args.split,
    )

    # Compute scores
    logger.info("Computing change scores...")
    scored_df = compute_all_scores(
        df, encoder, device, structure_mapping, args.clip_weight
    )

    # Save output
    output_path = args.output_dir / f"stenosis_change_scores_{args.split}.csv"
    scored_df.to_csv(output_path, index=False)
    logger.info(f"Saved scores to {output_path}")

    # Print summary statistics
    logger.info("\n" + "=" * 60)
    logger.info("SCORING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total studies scored: {len(scored_df)}")

    # CLIP similarity stats
    valid_clip = scored_df['clip_similarity'].dropna()
    if len(valid_clip) > 0:
        logger.info(f"\nCLIP Similarity:")
        logger.info(f"  Mean: {valid_clip.mean():.3f}")
        logger.info(f"  Std:  {valid_clip.std():.3f}")
        logger.info(f"  Min:  {valid_clip.min():.3f}")
        logger.info(f"  Max:  {valid_clip.max():.3f}")
        logger.info(f"  Valid: {len(valid_clip)}/{len(scored_df)}")

    # Severity distribution
    logger.info(f"\nSeverity Level Distribution:")
    for level in range(6):
        count = (scored_df['severity_level'] == level).sum()
        pct = 100 * count / len(scored_df)
        logger.info(f"  Level {level}: {count} ({pct:.1f}%)")

    # PCI-warranting
    pci_count = scored_df['pci_warranting'].sum()
    logger.info(f"\nPCI-Warranting Progressions: {pci_count} ({100*pci_count/len(scored_df):.1f}%)")

    # By change status
    logger.info(f"\nBy Ground Truth Change Status:")
    for status in ['changed', 'unchanged']:
        subset = scored_df[scored_df['change_status'] == status]
        if len(subset) > 0:
            mean_score = subset['combined_change_score'].mean()
            mean_severity = subset['severity_level'].mean()
            logger.info(f"  {status}: n={len(subset)}, mean_score={mean_score:.2f}, mean_severity={mean_severity:.2f}")


if __name__ == "__main__":
    main()
