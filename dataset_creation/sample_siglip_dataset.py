#!/usr/bin/env python3
"""Utility to subsample existing SigLIP manifests for quick experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Subsample SigLIP manifests (videos/texts/edges) for quick experiments."
    )
    parser.add_argument(
        "--source-dir",
        required=True,
        help="Directory containing full videos.csv/texts.csv/edges.csv manifests",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the sampled manifests will be written",
    )
    parser.add_argument(
        "--train",
        type=int,
        default=1400,
        help="Number of train videos to keep (default: 1400)",
    )
    parser.add_argument(
        "--val",
        type=int,
        default=200,
        help="Number of validation videos to keep (default: 200)",
    )
    parser.add_argument(
        "--test",
        type=int,
        default=400,
        help="Number of test videos to keep (default: 400)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    parser.add_argument(
        "--skip-debug",
        action="store_true",
        help="Do not write videos_with_debug_pos.csv to the output",
    )
    return parser.parse_args()


def _load_manifest(directory: Path, name: str) -> pd.DataFrame:
    path = directory / name
    if not path.exists():
        raise FileNotFoundError(f"Expected manifest '{path}' not found")
    return pd.read_csv(path)


def _sample_split(videos: pd.DataFrame, split: str, target_size: int, random_state: int) -> pd.DataFrame:
    if target_size <= 0:
        return videos.iloc[0:0]
    subset = videos[videos["Split"].astype(str).str.lower() == split]
    if subset.empty:
        return subset
    if target_size >= len(subset):
        return subset
    return subset.sample(n=target_size, random_state=random_state)


def _write_config(output_dir: Path, sizes: Dict[str, int], random_state: int) -> None:
    config_path = output_dir / "siglip_config.yaml"
    lines = [
        "apply_mappings: true",
        "assign_status: true",
        "cap_per_video: null",
        "filters:",
        "  contrast_agent_class: 1",
        "  main_structures:",
        "  - Left Coronary",
        "  - Right Coronary",
        "  normal_report_ratio: 0.05",
        "parallel_workers: 1",
        "train_test_split:",
        "  enabled: true",
        "  patient_column: CathReport_MRN",
        f"  random_state: {random_state}",
        "  test_ratio: 0.2",
        "  train_ratio: 0.7",
        "  val_ratio: 0.1",
        "sample_limits:",
        f"  random_state: {random_state}",
        f"  train_size: {sizes['train']}",
        f"  val_size: {sizes['val']}",
        f"  test_size: {sizes['test']}",
    ]
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = _load_manifest(source_dir, "videos.csv")
    edges = _load_manifest(source_dir, "edges.csv")
    texts = _load_manifest(source_dir, "texts.csv")

    videos_debug = None
    if not args.skip_debug:
        try:
            videos_debug = _load_manifest(source_dir, "videos_with_debug_pos.csv")
        except FileNotFoundError:
            videos_debug = None

    # Keep only videos that have supervision edges
    supervised_ids = set(edges["video_id"].unique())
    videos = videos[videos["video_id"].isin(supervised_ids)].copy()

    samples = []
    sizes = {"train": args.train, "val": args.val, "test": args.test}
    for split, size in sizes.items():
        sampled = _sample_split(videos, split, size, args.random_state)
        if not sampled.empty:
            samples.append(sampled)
    if not samples:
        raise RuntimeError("No samples collected; check split names and sizes")

    sampled_videos = pd.concat(samples, ignore_index=True)
    sampled_video_ids = set(sampled_videos["video_id"])
    sampled_edges = edges[edges["video_id"].isin(sampled_video_ids)].copy()
    sampled_text_ids = set(sampled_edges["text_id"].unique())
    sampled_texts = texts[texts["text_id"].isin(sampled_text_ids)].copy()

    sampled_videos.to_csv(output_dir / "videos.csv", index=False)
    sampled_edges.to_csv(output_dir / "edges.csv", index=False)
    sampled_texts.to_csv(output_dir / "texts.csv", index=False)

    if videos_debug is not None:
        sampled_debug = videos_debug[videos_debug["video_id"].isin(sampled_video_ids)].copy()
        sampled_debug.to_csv(output_dir / "videos_with_debug_pos.csv", index=False)

    _write_config(output_dir, sizes, args.random_state)

    print("Sampling summary")
    print("================")
    print(f"Videos: {len(sampled_videos)}")
    print(sampled_videos["Split"].value_counts())
    print(f"Edges: {len(sampled_edges)}")
    print(f"Texts: {len(sampled_texts)}")
    if videos_debug is not None:
        print(f"Debug positives: {len(sampled_debug)}")


if __name__ == "__main__":
    main()
