#!/usr/bin/env python3
"""
Show practical examples of ground truth vs predicted texts.
Displays what the model predicted vs what was expected.
"""

import pandas as pd
import argparse
from pathlib import Path


def format_text_list(texts, max_length=80):
    """Format a list of texts for display."""
    formatted = []
    for i, text in enumerate(texts, 1):
        if len(text) > max_length:
            text = text[:max_length] + "..."
        formatted.append(f"  {i}. {text}")
    return "\n".join(formatted)


def show_example(row, texts_df, idx_to_text_id, example_num):
    """Show one example with GT and predictions."""

    print(f"\n{'='*100}")
    print(f"EXAMPLE {example_num}")
    print(f"{'='*100}")

    # Basic info
    filename = row['FileName']
    filename_short = filename.split('/')[-1] if '/' in filename else filename
    print(f"\nVideo: {filename_short}")

    if 'severity_label' in row:
        severity = row['severity_label']
        print(f"Severity: {severity}")

    # Ground truth
    print(f"\n{'-'*100}")
    print("GROUND TRUTH (What the video shows):")
    print(f"{'-'*100}")

    if 'ground_truth_pos_texts' in row and pd.notna(row['ground_truth_pos_texts']):
        gt_texts = str(row['ground_truth_pos_texts']).split('\n')
        gt_texts = [t.strip() for t in gt_texts if t.strip()]
        print(format_text_list(gt_texts[:5]))  # Show first 5
        if len(gt_texts) > 5:
            print(f"  ... and {len(gt_texts) - 5} more")

    # Predictions
    print(f"\n{'-'*100}")
    print("TOP-5 PREDICTIONS (What the model retrieved):")
    print(f"{'-'*100}")

    predictions = []
    for i in range(1, 6):
        pred_idx_col = f'predicted_idx_{i}'
        sim_col = f'sim_{i}'

        if pred_idx_col in row and pd.notna(row[pred_idx_col]):
            pred_idx = int(row[pred_idx_col])
            sim_score = row[sim_col] if sim_col in row else None

            # Get the text
            if pred_idx in idx_to_text_id:
                text_id = idx_to_text_id[pred_idx]
                if text_id in texts_df.index:
                    text = texts_df.loc[text_id, 'prompt_text']
                    severity_pred = texts_df.loc[text_id, 'disease_severity']

                    sim_str = f"(sim: {sim_score:.3f})" if sim_score is not None else ""
                    predictions.append(f"  {i}. [{severity_pred.upper()}] {text[:80]}... {sim_str}")

    if predictions:
        print("\n".join(predictions))

    # Match analysis
    print(f"\n{'-'*100}")
    print("MATCH ANALYSIS:")
    print(f"{'-'*100}")

    # Check if any prediction matches ground truth
    if 'ground_truth_pos_ids' in row and pd.notna(row['ground_truth_pos_ids']):
        gt_indices = set([int(x) for x in str(row['ground_truth_pos_ids']).split('|')])

        matched = []
        for i in range(1, 6):
            pred_idx_col = f'predicted_idx_{i}'
            if pred_idx_col in row and pd.notna(row[pred_idx_col]):
                pred_idx = int(row[pred_idx_col])
                if pred_idx in gt_indices:
                    matched.append(i)

        if matched:
            print(f"✓ MATCH FOUND at position(s): {matched}")
            print(f"  Recall@1: {'YES ✓' if 1 in matched else 'NO ✗'}")
            print(f"  Recall@5: YES ✓")
        else:
            print(f"✗ NO MATCH in top-5 predictions")
            print(f"  Recall@1: NO ✗")
            print(f"  Recall@5: NO ✗")


def main():
    parser = argparse.ArgumentParser(
        description='Show practical examples of GT vs predictions'
    )
    parser.add_argument(
        '--val_results',
        type=str,
        default='outputs/DeepCORO_clip/dev_deep_coro_clip_single_video/2s1hs1n1_20251012-193155/val_epoch0.csv',
        help='Path to validation results CSV'
    )
    parser.add_argument(
        '--texts_csv',
        type=str,
        default='output_dataset/siglip_generated/texts.csv',
        help='Path to texts metadata CSV'
    )
    parser.add_argument(
        '--num_examples',
        type=int,
        default=5,
        help='Number of examples to show'
    )
    parser.add_argument(
        '--severity',
        type=str,
        choices=['normal', 'mild', 'severe', 'all'],
        default='all',
        help='Filter by severity'
    )
    parser.add_argument(
        '--show_matches',
        action='store_true',
        help='Only show examples with matches (Recall@5 = 1)'
    )
    parser.add_argument(
        '--show_misses',
        action='store_true',
        help='Only show examples without matches (Recall@5 = 0)'
    )

    args = parser.parse_args()

    print(f"\n{'='*100}")
    print("GROUND TRUTH vs PREDICTIONS - PRACTICAL EXAMPLES")
    print(f"{'='*100}")

    # Load data
    print("\nLoading data...")
    val_results = pd.read_csv(args.val_results)
    texts_df = pd.read_csv(args.texts_csv)

    # Create index to text_id mapping
    idx_to_text_id = dict(enumerate(texts_df['text_id'].tolist()))
    texts_df.set_index('text_id', inplace=True)

    print(f"  - Loaded {len(val_results)} validation results")
    print(f"  - Loaded {len(texts_df)} texts")

    # Filter by severity if requested
    if args.severity != 'all':
        if 'severity_label' in val_results.columns:
            val_results = val_results[
                val_results['severity_label'].str.lower() == args.severity.lower()
            ]
            print(f"  - Filtered to {len(val_results)} {args.severity.upper()} cases")

    # Filter by match status if requested
    if args.show_matches or args.show_misses:
        matches = []
        for idx, row in val_results.iterrows():
            if 'ground_truth_pos_ids' in row and pd.notna(row['ground_truth_pos_ids']):
                gt_indices = set([int(x) for x in str(row['ground_truth_pos_ids']).split('|')])

                is_match = False
                for i in range(1, 6):
                    pred_idx_col = f'predicted_idx_{i}'
                    if pred_idx_col in row and pd.notna(row[pred_idx_col]):
                        pred_idx = int(row[pred_idx_col])
                        if pred_idx in gt_indices:
                            is_match = True
                            break

                matches.append(is_match)
            else:
                matches.append(False)

        val_results['has_match'] = matches

        if args.show_matches:
            val_results = val_results[val_results['has_match']]
            print(f"  - Showing only MATCHES: {len(val_results)} examples")
        elif args.show_misses:
            val_results = val_results[~val_results['has_match']]
            print(f"  - Showing only MISSES: {len(val_results)} examples")

    # Sample examples
    if len(val_results) == 0:
        print("\nNo examples found matching the criteria!")
        return

    num_examples = min(args.num_examples, len(val_results))
    sample_indices = val_results.sample(n=num_examples, random_state=42).index

    # Show examples
    for i, idx in enumerate(sample_indices, 1):
        row = val_results.loc[idx]
        show_example(row, texts_df, idx_to_text_id, i)

    print(f"\n{'='*100}\n")
    print(f"Shown {num_examples} examples out of {len(val_results)} available")
    print(f"\nTo see more examples, use:")
    print(f"  --num_examples N              Show N examples")
    print(f"  --severity [normal|mild|severe]  Filter by severity")
    print(f"  --show_matches                Only show correct predictions")
    print(f"  --show_misses                 Only show incorrect predictions")
    print()


if __name__ == '__main__':
    main()
