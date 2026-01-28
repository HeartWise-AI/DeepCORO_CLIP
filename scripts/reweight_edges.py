#!/usr/bin/env python
"""
Utility script to add inverse-frequency weights to SigLIP edges manifests.

Example:
    python scripts/reweight_edges.py \
        datasets/siglip_multiprompt_output/edges_pos.csv \
        datasets/siglip_multiprompt_output/texts.csv \
        datasets/siglip_multiprompt_output/edges_weighted.csv
"""

import argparse
from typing import Dict

import numpy as np
import pandas as pd


def effective_num_weight(count: int, beta: float = 0.999) -> float:
    """Compute the effective number of samples weight."""
    if count <= 0:
        return 1.0
    return (1.0 - beta) / (1.0 - beta ** count)


def compute_weights(edges_df: pd.DataFrame, prompt_col: str, beta: float) -> Dict[str, float]:
    counts = edges_df[prompt_col].fillna("unknown").value_counts()
    return {label: effective_num_weight(int(count), beta) for label, count in counts.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Reweight SigLIP edges by inverse prompt prevalence.")
    parser.add_argument("edges_csv", type=str, help="Path to edges.csv file.")
    parser.add_argument("texts_csv", type=str, help="Path to texts.csv file.")
    parser.add_argument("output_csv", type=str, help="Path to write weighted edges CSV.")
    parser.add_argument("--text-id-column", default="text_id", help="Text ID column name.")
    parser.add_argument("--prompt-type-column", default="prompt_type", help="Prompt type column name.")
    parser.add_argument("--beta", type=float, default=0.999, help="Smoothing factor for effective number weighting.")
    parser.add_argument(
        "--cap-percentile",
        type=float,
        default=99.0,
        help="Optional percentile to cap extremely large weights (set <=0 to disable).",
    )
    args = parser.parse_args()

    edges_df = pd.read_csv(args.edges_csv)
    texts_df = pd.read_csv(args.texts_csv)

    merge_cols = [args.text_id_column, args.prompt_type_column]
    available_cols = [col for col in merge_cols if col in texts_df.columns]
    if len(available_cols) != 2:
        raise ValueError(
            f"texts.csv must contain columns {merge_cols}, found {available_cols}."
        )

    texts_df = texts_df[[args.text_id_column, args.prompt_type_column]].drop_duplicates()
    merged = edges_df.merge(texts_df, on=args.text_id_column, how="left")

    weight_map = compute_weights(merged, args.prompt_type_column, args.beta)
    merged["weight"] = merged[args.prompt_type_column].fillna("unknown").map(weight_map).astype(float)

    if args.cap_percentile and args.cap_percentile > 0:
        cap_value = float(np.percentile(merged["weight"], args.cap_percentile))
        merged["weight"] = merged["weight"].clip(upper=cap_value)

    merged.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
