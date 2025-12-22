#!/usr/bin/env python3
"""
Evaluate semantic alignment between retrieved texts and ground-truth annotations.

For each video in a validation CSV (val_epoch*.csv) we look at the top-K predicted
text indices and compare their metadata (segment, tree, severity, etc.) against
the ground-truth positive texts defined in output_dataset/siglip_generated/videos.csv.

Example:
    python scripts/analyze_retrieval_semantics.py \
        --val_csv outputs/.../val_epoch1.csv \
        --texts_csv output_dataset/siglip_generated/texts.csv \
        --videos_csv output_dataset/siglip_generated/videos.csv \
        --top_k 5
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ATTRIBUTES = ("segment", "tree", "disease_severity", "category_binary")


def _normalize_string(value: object) -> Optional[str]:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text in {"", "nan", "none"}:
        return None
    return text


def _category_binary(meta_row: pd.Series) -> Optional[str]:
    """Map fine-grained categories to coarse Normal vs Stenosis/Other label."""
    candidate_fields = (
        meta_row.get("category"),
        meta_row.get("prompt_bucket"),
        meta_row.get("disease_severity"),
    )
    for field in candidate_fields:
        normalized = _normalize_string(field)
        if normalized is None:
            continue
        if normalized == "normal":
            return "normal"
        # Treat any non-normal label as abnormal/stenosis bucket
        return "stenosis_or_other"
    return None


def extract_attribute(meta_row: pd.Series, attr: str) -> Optional[str]:
    if attr == "category_binary":
        return _category_binary(meta_row)
    return _normalize_string(meta_row.get(attr))


def load_text_metadata(texts_csv: Path) -> Tuple[pd.DataFrame, Dict[int, str], Dict[str, int]]:
    texts_df = pd.read_csv(texts_csv)
    if "text_id" not in texts_df.columns:
        raise ValueError(f"'text_id' column missing from {texts_csv}")

    idx_to_text_id: Dict[int, str] = dict(enumerate(texts_df["text_id"].astype(str).tolist()))
    text_id_to_idx: Dict[str, int] = {tid: idx for idx, tid in idx_to_text_id.items()}
    texts_df = texts_df.set_index("text_id")
    return texts_df, idx_to_text_id, text_id_to_idx


def build_video_to_gt(
    videos_csv: Path, text_id_to_idx: Dict[str, int]
) -> Dict[str, List[int]]:
    videos_df = pd.read_csv(videos_csv)
    if "FileName" not in videos_df or "positive_text_ids" not in videos_df:
        raise ValueError(
            f"'FileName' or 'positive_text_ids' missing from {videos_csv}. "
            "Cannot construct ground-truth mapping."
        )

    video_to_gt: Dict[str, List[int]] = {}
    for _, row in videos_df.iterrows():
        file_path = row["FileName"]
        pos_field = row.get("positive_text_ids", "")
        if pd.isna(pos_field):
            continue
        indices: List[int] = []
        for raw_id in str(pos_field).split("|"):
            text_id = raw_id.strip()
            if not text_id:
                continue
            idx = text_id_to_idx.get(text_id)
            if idx is not None:
                indices.append(idx)
        if indices:
            video_to_gt[file_path] = indices
    return video_to_gt


def compute_attribute_sets(
    gt_indices: Iterable[int],
    idx_to_text_id: Dict[int, str],
    texts_df: pd.DataFrame,
) -> Dict[str, set]:
    attr_values: Dict[str, set] = {attr: set() for attr in ATTRIBUTES}
    for idx in gt_indices:
        text_id = idx_to_text_id.get(idx)
        if text_id is None or text_id not in texts_df.index:
            continue
        meta = texts_df.loc[text_id]
        for attr in ATTRIBUTES:
            value = extract_attribute(meta, attr)
            if value is not None:
                attr_values[attr].add(value)
    return attr_values


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_semantics(
    val_csv: Path,
    texts_csv: Path,
    videos_csv: Path,
    top_k: int = 5,
) -> Dict[str, Dict[int, Dict[str, int]]]:
    texts_df, idx_to_text_id, text_id_to_idx = load_text_metadata(texts_csv)
    video_to_gt_indices = build_video_to_gt(videos_csv, text_id_to_idx)
    val_df = pd.read_csv(val_csv)

    stats: Dict[str, Dict[int, Dict[str, int]]] = {
        attr: {rank: {"match": 0, "total": 0} for rank in range(1, top_k + 1)}
        for attr in ATTRIBUTES
    }

    for _, row in val_df.iterrows():
        filename = row["FileName"]
        gt_indices = video_to_gt_indices.get(filename)
        if not gt_indices:
            continue

        gt_attr_values = compute_attribute_sets(gt_indices, idx_to_text_id, texts_df)

        for rank in range(1, top_k + 1):
            pred_col = f"predicted_idx_{rank}"
            pred_idx = row.get(pred_col)
            if pd.isna(pred_idx):
                continue

            try:
                pred_idx = int(pred_idx)
            except (TypeError, ValueError):
                continue

            text_id = idx_to_text_id.get(pred_idx)
            if text_id is None or text_id not in texts_df.index:
                continue
            meta = texts_df.loc[text_id]

            for attr in ATTRIBUTES:
                if not gt_attr_values[attr]:
                    continue
                pred_value = extract_attribute(meta, attr)
                if pred_value is None:
                    continue
                stats[attr][rank]["total"] += 1
                if pred_value in gt_attr_values[attr]:
                    stats[attr][rank]["match"] += 1

    return stats


def print_summary(stats: Dict[str, Dict[int, Dict[str, int]]]) -> None:
    print("\nSemantic alignment of top-K predictions (per rank):")
    for attr, per_rank in stats.items():
        print(f"\nAttribute: {attr}")
        header = f"{'Rank':>4}  {'Match':>8}  {'Total':>8}  {'Pct':>7}"
        print(header)
        print("-" * len(header))
        for rank, counters in per_rank.items():
            total = counters["total"]
            match = counters["match"]
            pct = (match / total * 100) if total else 0.0
            print(f"{rank:>4}  {match:>8}  {total:>8}  {pct:6.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze how often top-K retrieved texts share semantic attributes with GT."
    )
    parser.add_argument("--val_csv", type=Path, required=True, help="Path to val_epoch*.csv")
    parser.add_argument(
        "--texts_csv",
        type=Path,
        default=Path("output_dataset/siglip_generated/texts.csv"),
        help="Path to SigLIP texts metadata CSV.",
    )
    parser.add_argument(
        "--videos_csv",
        type=Path,
        default=Path("output_dataset/siglip_generated/videos.csv"),
        help="Path to videos.csv containing positive_text_ids.",
    )
    parser.add_argument(
        "--top_k", type=int, default=5, help="How many predictions per sample to evaluate."
    )
    parser.add_argument(
        "--save_json",
        type=Path,
        default=None,
        help="Optional path to dump the raw statistics as JSON.",
    )
    args = parser.parse_args()

    stats = analyze_semantics(
        val_csv=args.val_csv,
        texts_csv=args.texts_csv,
        videos_csv=args.videos_csv,
        top_k=args.top_k,
    )
    print_summary(stats)

    if args.save_json:
        import json

        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"\nSaved raw stats to {args.save_json}")


if __name__ == "__main__":
    main()
