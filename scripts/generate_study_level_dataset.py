#!/usr/bin/env python3
"""
Generate Study-Level SigLIP Dataset

This script converts segment-level SigLIP data into study-level data where:
- Each study has ONE full report (reconstructed from all segment texts)
- Each video aligns to its study's full report
- Weighting applies at the study level based on max severity

The output format is compatible with the original base_config.yaml approach
where target_label=Report and each video aligns with a full report text.

Usage:
    python scripts/generate_study_level_dataset.py \
        --texts output_dataset/siglip_generated/texts.csv \
        --videos output_dataset/siglip_generated/videos.csv \
        --output-dir output_dataset/siglip_study_level
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Severity ordering for determining study-level severity
SEVERITY_ORDER = {
    'normal': 0,
    'mild': 1,
    'moderate': 2,
    'severe': 3,
    'critical': 4,
}

# Segment ordering for report reconstruction (anatomical order)
SEGMENT_ORDER = [
    'left_main',
    'prox_lad', 'mid_lad', 'dist_lad',
    'd1', 'd2',
    'prox_lcx', 'lcx', 'dist_lcx',
    'om1', 'om2', 'om3',
    'ramus', 'bx',
    'lvp',
    'prox_rca', 'mid_rca', 'dist_rca',
    'pda', 'posterolateral',
]


def parse_tree_from_tags(tags: str) -> Optional[str]:
    """Extract tree (left/right) from tags string."""
    if not isinstance(tags, str):
        return None
    for token in tags.split('|'):
        if token.startswith('tree:'):
            return token.split(':')[1].strip()
    return None


def get_severity_rank(severity: str) -> int:
    """Get numeric rank for severity comparison."""
    return SEVERITY_ORDER.get(str(severity).lower().strip(), 0)


def get_segment_rank(segment: str) -> int:
    """Get ordering rank for segment."""
    seg_lower = str(segment).lower().strip()
    try:
        return SEGMENT_ORDER.index(seg_lower)
    except ValueError:
        return 999  # Unknown segments go to end


def build_study_report(
    text_ids: List[str],
    texts_df: pd.DataFrame,
    text_id_to_row: Dict[str, pd.Series],
    deduplicate_segments: bool = True,
) -> Tuple[str, str, str, float]:
    """
    Build a full study report from segment text IDs.

    Args:
        text_ids: List of segment text IDs for this study
        texts_df: DataFrame with all texts
        text_id_to_row: Mapping from text_id to row
        deduplicate_segments: If True, keep only the most severe finding per segment

    Returns:
        Tuple of (report_text, max_severity, study_type, soft_weight)
    """
    if not text_ids:
        return "No findings available.", "normal", "normal", 1.0

    # Collect segment info
    segment_findings: Dict[str, List[dict]] = defaultdict(list)
    max_severity_rank = 0
    max_severity = "normal"

    for tid in text_ids:
        if tid not in text_id_to_row:
            continue
        row = text_id_to_row[tid]

        prompt_text = str(row.get('prompt_text', '')).strip()
        segment = str(row.get('segment', '')).strip().lower()
        severity = str(row.get('disease_severity', 'normal')).strip().lower()
        tree = parse_tree_from_tags(row.get('tags', ''))

        if not prompt_text:
            continue

        finding = {
            'text_id': tid,
            'text': prompt_text,
            'segment': segment,
            'severity': severity,
            'severity_rank': get_severity_rank(severity),
            'tree': tree,
            'rank': get_segment_rank(segment),
        }
        segment_findings[segment].append(finding)

        sev_rank = get_severity_rank(severity)
        if sev_rank > max_severity_rank:
            max_severity_rank = sev_rank
            max_severity = severity

    # Deduplicate: for each segment, keep the most severe finding
    final_segments = []
    if deduplicate_segments:
        for segment, findings in segment_findings.items():
            # Sort by severity (descending), then by text length (prefer more detailed)
            findings.sort(key=lambda x: (-x['severity_rank'], -len(x['text'])))
            best = findings[0]
            final_segments.append(best)
    else:
        # Keep all findings
        for findings in segment_findings.values():
            final_segments.extend(findings)

    # Sort segments by anatomical order
    final_segments.sort(key=lambda x: x['rank'])

    # Build report text - group by tree (left/right) for cleaner reports
    left_segments = [s for s in final_segments if s['tree'] == 'left']
    right_segments = [s for s in final_segments if s['tree'] == 'right']
    other_segments = [s for s in final_segments if s['tree'] not in ('left', 'right')]

    report_parts = []

    if left_segments:
        left_texts = [s['text'] for s in left_segments]
        report_parts.append("LEFT CORONARY SYSTEM: " + " ".join(left_texts))

    if right_segments:
        right_texts = [s['text'] for s in right_segments]
        report_parts.append("RIGHT CORONARY SYSTEM: " + " ".join(right_texts))

    if other_segments:
        other_texts = [s['text'] for s in other_segments]
        report_parts.append(" ".join(other_texts))

    report_text = "\n".join(report_parts) if report_parts else "No significant findings."

    # Determine study type based on max severity
    study_type = max_severity

    # Calculate study-level soft weight
    # Weight based on clinical importance at study level
    if max_severity in ('severe', 'critical'):
        soft_weight = 3.0  # High priority for severe cases
    elif max_severity == 'moderate':
        soft_weight = 2.0  # Medium priority
    elif max_severity == 'mild':
        soft_weight = 1.5  # Slight upweight
    else:
        soft_weight = 1.0  # Normal baseline

    return report_text, max_severity, study_type, soft_weight


def generate_study_level_dataset(
    texts_path: Path,
    videos_path: Path,
    output_dir: Path,
    include_simple_format: bool = True,
) -> Dict[str, int]:
    """
    Generate study-level dataset from segment-level SigLIP data.

    Args:
        texts_path: Path to segment-level texts.csv
        videos_path: Path to videos.csv with positive_text_ids
        output_dir: Output directory for study-level data
        include_simple_format: Also generate simple CSV compatible with base_config.yaml

    Returns:
        Statistics dictionary
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info(f"Loading texts from {texts_path}")
    texts_df = pd.read_csv(texts_path)
    text_id_to_row = {row['text_id']: row for _, row in texts_df.iterrows()}

    logger.info(f"Loading videos from {videos_path}")
    videos_df = pd.read_csv(videos_path)

    logger.info(f"Loaded {len(texts_df)} texts, {len(videos_df)} videos")

    # Group videos by study
    study_to_videos = defaultdict(list)
    study_to_text_ids = defaultdict(set)

    for _, row in videos_df.iterrows():
        study_id = row['study_id']
        video_id = row['video_id']

        study_to_videos[study_id].append(row)

        # Collect all text IDs for this study
        text_ids_str = str(row.get('positive_text_ids', ''))
        if text_ids_str and text_ids_str != 'nan':
            for tid in text_ids_str.split('|'):
                tid = tid.strip()
                if tid:
                    study_to_text_ids[study_id].add(tid)

    logger.info(f"Found {len(study_to_videos)} unique studies")

    # Build study-level reports
    study_reports = {}
    for study_id, text_ids in study_to_text_ids.items():
        report_text, max_severity, study_type, soft_weight = build_study_report(
            list(text_ids), texts_df, text_id_to_row
        )
        study_reports[study_id] = {
            'report_text': report_text,
            'max_severity': max_severity,
            'study_type': study_type,
            'soft_weight': soft_weight,
            'num_segments': len(text_ids),
        }

    # Create study-level texts.csv
    study_texts_rows = []
    study_id_to_text_id = {}

    for i, (study_id, report) in enumerate(study_reports.items()):
        text_id = f"S{i:06d}"
        study_id_to_text_id[study_id] = text_id

        study_texts_rows.append({
            'text_id': text_id,
            'study_id': study_id,
            'prompt_type': 'study_report',
            'prompt_text': report['report_text'],
            'soft_weight': report['soft_weight'],
            'disease_severity': report['max_severity'],
            'study_type': report['study_type'],
            'num_segments': report['num_segments'],
            'prompt_bucket': 'abnormal' if report['max_severity'] != 'normal' else 'normal',
        })

    study_texts_df = pd.DataFrame(study_texts_rows)
    study_texts_path = output_dir / 'texts.csv'
    study_texts_df.to_csv(study_texts_path, index=False)
    logger.info(f"Saved {len(study_texts_df)} study-level texts to {study_texts_path}")

    # Create study-level videos.csv
    study_videos_rows = []
    for study_id, video_rows in study_to_videos.items():
        text_id = study_id_to_text_id.get(study_id)
        if not text_id:
            continue

        for video_row in video_rows:
            study_videos_rows.append({
                'video_id': video_row['video_id'],
                'study_id': study_id,
                'main_structure': video_row.get('main_structure', ''),
                'view_name': video_row.get('view_name', ''),
                'FileName': video_row['FileName'],
                'Split': video_row.get('Split', video_row.get('split', '')),
                'split': video_row.get('split', video_row.get('Split', '')),
                'status': video_row.get('status', ''),
                'study_type': study_reports[study_id]['study_type'],
                'positive_text_ids': text_id,  # Single study-level text ID
            })

    study_videos_df = pd.DataFrame(study_videos_rows)
    study_videos_path = output_dir / 'videos.csv'
    study_videos_df.to_csv(study_videos_path, index=False)
    logger.info(f"Saved {len(study_videos_df)} videos to {study_videos_path}")

    # Also create simple format compatible with base_config.yaml
    if include_simple_format:
        # This format has FileName, Report, StudyInstanceUID columns
        simple_rows = []
        for study_id, video_rows in study_to_videos.items():
            report = study_reports.get(study_id, {})
            for video_row in video_rows:
                simple_rows.append({
                    'FileName': video_row['FileName'],
                    'Report': report.get('report_text', ''),
                    'StudyInstanceUID': study_id,
                    'Split': video_row.get('Split', video_row.get('split', '')),
                    'main_structure_name': video_row.get('main_structure', ''),
                    'study_type': report.get('study_type', 'normal'),
                    'soft_weight': report.get('soft_weight', 1.0),
                    'disease_severity': report.get('max_severity', 'normal'),
                })

        simple_df = pd.DataFrame(simple_rows)
        simple_path = output_dir / 'dataset_with_reports.csv'
        simple_df.to_csv(simple_path, index=False)
        logger.info(f"Saved simple format to {simple_path}")

    # Print severity distribution at study level
    severity_counts = study_texts_df['disease_severity'].value_counts()
    logger.info(f"\nStudy-level severity distribution:\n{severity_counts}")

    # Statistics
    stats = {
        'num_studies': len(study_reports),
        'num_videos': len(study_videos_df),
        'num_segment_texts': len(texts_df),
        'severity_distribution': severity_counts.to_dict(),
    }

    return stats


def main():
    parser = argparse.ArgumentParser(description="Generate study-level SigLIP dataset")
    parser.add_argument(
        '--texts',
        type=Path,
        default=Path('output_dataset/siglip_generated/texts.csv'),
        help='Path to segment-level texts.csv'
    )
    parser.add_argument(
        '--videos',
        type=Path,
        default=Path('output_dataset/siglip_generated/videos.csv'),
        help='Path to videos.csv'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('output_dataset/siglip_study_level'),
        help='Output directory for study-level data'
    )
    parser.add_argument(
        '--no-simple-format',
        action='store_true',
        help='Skip generating simple format compatible with base_config.yaml'
    )

    args = parser.parse_args()

    stats = generate_study_level_dataset(
        texts_path=args.texts,
        videos_path=args.videos,
        output_dir=args.output_dir,
        include_simple_format=not args.no_simple_format,
    )

    logger.info(f"\nGeneration complete. Stats: {stats}")


if __name__ == '__main__':
    main()
