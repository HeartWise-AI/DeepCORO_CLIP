#!/usr/bin/env python3
"""
Calculate recall@k and hit@k for different disease severity levels.
Maps validation results with text metadata to compute severity-specific retrieval metrics.
"""

import argparse
from collections import defaultdict

import pandas as pd

SEVERITY_ORDER = ['normal', 'mild', 'moderate', 'severe', 'critical', 'cto']
SEVERITY_SET = set(SEVERITY_ORDER)


def parse_text_ids(text_id_str):
    """Parse pipe-separated text IDs into a list."""
    if pd.isna(text_id_str):
        return []
    return [tid.strip() for tid in str(text_id_str).split('|') if tid and tid.strip()]


def parse_index_ids(index_str):
    """Parse pipe-separated numeric indices into a list of ints."""
    if pd.isna(index_str):
        return []
    indices: list[int] = []
    for token in str(index_str).split('|'):
        token = token.strip()
        if not token:
            continue
        try:
            indices.append(int(token))
        except ValueError:
            continue
    return indices


def recall_and_hit_at_k(predicted_ids, ground_truth_ids, k=5):
    """Return (recall@k, hit@k) using unique predictions."""
    gt_set = {int(idx) for idx in ground_truth_ids if idx is not None}
    if not gt_set:
        return None, None

    seen = set()
    unique_preds: list[int] = []
    for pid in predicted_ids:
        if pid in seen:
            continue
        seen.add(pid)
        unique_preds.append(pid)
        if len(unique_preds) >= k:
            break

    if not unique_preds:
        return 0.0, 0.0

    limit = min(k, len(unique_preds))
    prefix = unique_preds[:limit]
    hits = len([pred for pred in prefix if pred in gt_set])
    recall = hits / len(gt_set) if gt_set else 0.0
    hit = 1.0 if hits > 0 else 0.0
    return recall, hit


def get_severity_from_text_ids(text_ids, text_metadata):
    """
    Get the most severe severity level from a list of text IDs.

    Severity hierarchy: normal < mild < moderate < severe < critical < cto
    """
    severity_scores = {s: i for i, s in enumerate(SEVERITY_ORDER)}

    max_severity = 'normal'
    max_score = 0

    for tid in text_ids:
        if tid in text_metadata.index:
            severity = text_metadata.loc[tid, 'disease_severity']
            if pd.notna(severity):
                severity = str(severity).strip().lower()
                if severity in severity_scores:
                    score = severity_scores[severity]
                    if score > max_score:
                        max_score = score
                        max_severity = severity

    return max_severity


def main():
    parser = argparse.ArgumentParser(
        description='Calculate recall@5 for different disease severity levels'
    )
    parser.add_argument(
        '--val_results',
        type=str,
        default='outputs/DeepCORO_clip/dev_deep_coro_clip_single_video/2s1hs1n1_20251012-193155/val_epoch0.csv',
        help='Path to validation results CSV'
    )
    parser.add_argument(
        '--videos_csv',
        type=str,
        default='output_dataset/siglip_generated/videos.csv',
        help='Path to videos metadata CSV'
    )
    parser.add_argument(
        '--texts_csv',
        type=str,
        default='output_dataset/siglip_generated/texts.csv',
        help='Path to texts metadata CSV'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=5,
        help='Top-k for recall calculation (default: 5)'
    )

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"Calculating Recall@{args.k} and Hit@{args.k} by Disease Severity")
    print(f"{'='*80}\n")

    # Load data
    print("Loading data...")
    val_results = pd.read_csv(args.val_results)
    videos_meta = pd.read_csv(args.videos_csv)
    texts_meta = pd.read_csv(args.texts_csv)

    # Index texts by text_id for fast lookup
    if 'text_id' not in texts_meta.columns:
        raise ValueError("texts.csv must contain a 'text_id' column.")
    texts_meta_indexed = texts_meta.set_index('text_id')

    print(f"  - Validation results: {len(val_results)} videos")
    print(f"  - Videos metadata: {len(videos_meta)} videos")
    print(f"  - Texts metadata: {len(texts_meta)} texts")
    print()

    # Process each validation result
    severity_results = defaultdict(lambda: {'total': 0, 'recall_sum': 0.0, 'hit_sum': 0.0})

    print("Processing validation results...")
    for idx, row in val_results.iterrows():
        # Ground-truth indices live in the shared retrieval index space
        if 'ground_truth_indices' not in row or pd.isna(row['ground_truth_indices']):
            continue

        ground_truth_indices = parse_index_ids(row['ground_truth_indices'])
        if not ground_truth_indices:
            continue

        # Get predicted indices (top-k) - also numeric
        predicted_indices = []
        for i in range(1, args.k + 1):
            pred_idx_col = f'predicted_idx_{i}'
            if pred_idx_col in row and pd.notna(row[pred_idx_col]):
                try:
                    predicted_indices.append(int(row[pred_idx_col]))
                except (TypeError, ValueError):
                    continue

        if not predicted_indices:
            continue

        # Get severity from the severity_label column if available
        if 'severity_label' in row and pd.notna(row['severity_label']):
            severity = str(row['severity_label']).strip().lower()
        else:
            # Fallback: Convert ground truth indices to text_ids to determine severity
            ground_truth_text_ids = parse_text_ids(row.get('ground_truth_pos_ids', ''))
            severity = get_severity_from_text_ids(ground_truth_text_ids, texts_meta_indexed)

        if not severity:
            severity = 'unknown'
        severity = severity.lower()
        if severity not in SEVERITY_SET:
            severity = 'unknown'

        # Calculate recall@k and hit@k using numeric indices
        recall_val, hit_val = recall_and_hit_at_k(predicted_indices, ground_truth_indices, k=args.k)
        if recall_val is None:
            continue

        # Update stats
        severity_results[severity]['total'] += 1
        severity_results[severity]['recall_sum'] += recall_val
        severity_results[severity]['hit_sum'] += hit_val if hit_val is not None else 0.0

    # Calculate and display results
    print(f"\n{'='*80}")
    print(f"Results: Recall@{args.k} by Disease Severity")
    print(f"{'='*80}\n")

    severity_order = SEVERITY_ORDER
    overall_total = 0
    overall_recall_sum = 0.0
    overall_hit_sum = 0.0

    header_recall = f"Recall@{args.k}"
    header_hit = f"Hit@{args.k}"
    print(f"{'Severity':<15} {'Count':<10} {header_recall:<15} {header_hit:<15}")
    print(f"{'-'*60}")

    for severity in severity_order:
        if severity in severity_results:
            stats = severity_results[severity]
            total = stats['total']
            recall_avg = stats['recall_sum'] / total if total > 0 else 0.0
            hit_avg = stats['hit_sum'] / total if total > 0 else 0.0

            overall_total += total
            overall_recall_sum += stats['recall_sum']
            overall_hit_sum += stats['hit_sum']

            print(f"{severity:<15} {total:<10} {recall_avg:.4f} ({recall_avg*100:.2f}%) {hit_avg:.4f} ({hit_avg*100:.2f}%)")

    # Overall recall
    print(f"{'-'*60}")
    overall_recall = overall_recall_sum / overall_total if overall_total > 0 else 0.0
    overall_hit = overall_hit_sum / overall_total if overall_total > 0 else 0.0
    print(f"{'OVERALL':<15} {overall_total:<10} {overall_recall:.4f} ({overall_recall*100:.2f}%) {overall_hit:.4f} ({overall_hit*100:.2f}%)")
    print(f"\n{'='*80}\n")

    # Additional breakdown by severity groups
    print("\nGrouped Results:")
    print(f"{'='*80}")

    # Normal vs Abnormal
    normal_stats = severity_results.get('normal', {'total': 0, 'recall_sum': 0.0, 'hit_sum': 0.0})
    abnormal_keys = ['mild', 'moderate', 'severe', 'critical', 'cto']
    abnormal_total = sum(severity_results[s]['total'] for s in abnormal_keys if s in severity_results)
    abnormal_recall_sum = sum(severity_results[s]['recall_sum'] for s in abnormal_keys if s in severity_results)
    abnormal_hit_sum = sum(severity_results[s]['hit_sum'] for s in abnormal_keys if s in severity_results)

    print(f"\nNormal vs Abnormal:")
    print(f"{'-'*60}")
    normal_recall = normal_stats['recall_sum'] / normal_stats['total'] if normal_stats['total'] > 0 else 0.0
    normal_hit = normal_stats['hit_sum'] / normal_stats['total'] if normal_stats['total'] > 0 else 0.0
    abnormal_recall = abnormal_recall_sum / abnormal_total if abnormal_total > 0 else 0.0
    abnormal_hit = abnormal_hit_sum / abnormal_total if abnormal_total > 0 else 0.0
    print(f"{'Normal':<15} {normal_stats['total']:<10} {normal_recall:.4f} ({normal_recall*100:.2f}%) {normal_hit:.4f} ({normal_hit*100:.2f}%)")
    print(f"{'Abnormal':<15} {abnormal_total:<10} {abnormal_recall:.4f} ({abnormal_recall*100:.2f}%) {abnormal_hit:.4f} ({abnormal_hit*100:.2f}%)")

    # Mild/Moderate vs Severe/Critical/CTO
    mild_keys = ['mild', 'moderate']
    severe_keys = ['severe', 'critical', 'cto']
    mild_mod_total = sum(severity_results[s]['total'] for s in mild_keys if s in severity_results)
    mild_mod_recall_sum = sum(severity_results[s]['recall_sum'] for s in mild_keys if s in severity_results)
    mild_mod_hit_sum = sum(severity_results[s]['hit_sum'] for s in mild_keys if s in severity_results)
    severe_total = sum(severity_results[s]['total'] for s in severe_keys if s in severity_results)
    severe_recall_sum = sum(severity_results[s]['recall_sum'] for s in severe_keys if s in severity_results)
    severe_hit_sum = sum(severity_results[s]['hit_sum'] for s in severe_keys if s in severity_results)

    print(f"\nMild/Moderate vs Severe/Critical/CTO:")
    print(f"{'-'*60}")
    mild_mod_recall = mild_mod_recall_sum / mild_mod_total if mild_mod_total > 0 else 0.0
    mild_mod_hit = mild_mod_hit_sum / mild_mod_total if mild_mod_total > 0 else 0.0
    severe_recall = severe_recall_sum / severe_total if severe_total > 0 else 0.0
    severe_hit = severe_hit_sum / severe_total if severe_total > 0 else 0.0
    print(f"{'Mild/Moderate':<15} {mild_mod_total:<10} {mild_mod_recall:.4f} ({mild_mod_recall*100:.2f}%) {mild_mod_hit:.4f} ({mild_mod_hit*100:.2f}%)")
    print(f"{'Severe+':<15} {severe_total:<10} {severe_recall:.4f} ({severe_recall*100:.2f}%) {severe_hit:.4f} ({severe_hit*100:.2f}%)")

    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()
