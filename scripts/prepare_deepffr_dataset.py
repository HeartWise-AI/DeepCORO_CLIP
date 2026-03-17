#!/usr/bin/env python3
"""Prepare DeepFFR training dataset from the parquet file.

Uses FFR Hyperemia values, creates binary FFR columns
(FFR <= 0.80 = 1 abnormal, > 0.80 = 0 normal), uses existing splits from parquet.
Saves with alpha separator.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Config
SOURCE_PARQUET = "/media/data1/datasets/DeepCoro/2b_CathReport_HEMO_MHI_MERGED_2017-2024_VIDEO_LEVEL_with_status_and_split.parquet"
OUTPUT_CSV = "/volume/DeepCORO_CLIP/data/deepffr_training.csv"
SEP = "α"
FFR_THRESHOLD = 0.80
SEED = 42

# 17 vessels (same as IFR, excluding those with no FFR data)
VESSELS = [
    "prox_rca", "mid_rca", "dist_rca", "pda", "posterolateral",
    "left_main",
    "prox_lad", "mid_lad", "dist_lad", "D1", "D2",
    "prox_lcx", "mid_lcx", "dist_lcx", "om1", "om2", "bx",
]


def main():
    print(f"Reading source parquet: {SOURCE_PARQUET}")
    # Read only needed columns
    ffr_cols = [f"{v}_FFRHYPEREMIE" for v in VESSELS]
    keep_source = ["FileName", "StudyInstanceUID", "Split", "view_class"] + ffr_cols
    df = pd.read_parquet(SOURCE_PARQUET, columns=keep_source)
    print(f"  Total rows: {len(df)}, unique studies: {df['StudyInstanceUID'].nunique()}")

    # Convert FFR values: -1 -> NaN (sentinel for missing), 0 -> NaN (likely missing too)
    for col in ffr_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] <= 0, col] = np.nan

    # Filter to studies with at least one valid FFR measurement
    has_valid_ffr = df[ffr_cols].notna().any(axis=1)
    studies_with_ffr = df.loc[has_valid_ffr, "StudyInstanceUID"].unique()
    df_ffr = df[df["StudyInstanceUID"].isin(studies_with_ffr)].copy()
    print(f"  Studies with valid FFR: {len(studies_with_ffr)}")
    print(f"  Rows after filtering: {len(df_ffr)}")

    # Create binary FFR columns: <= 0.80 = 1 (abnormal/ischemic), > 0.80 = 0 (normal)
    for vessel in VESSELS:
        ffr_col = f"{vessel}_FFRHYPEREMIE"
        binary_col = f"{vessel}_FFRHYPEREMIE_binary"
        df_ffr[binary_col] = np.nan
        valid_mask = df_ffr[ffr_col].notna()
        df_ffr.loc[valid_mask, binary_col] = (df_ffr.loc[valid_mask, ffr_col] <= FFR_THRESHOLD).astype(float)

    # Print FFR data availability per vessel
    print(f"\n  FFR data availability (threshold: FFR <= {FFR_THRESHOLD} = abnormal):")
    for vessel in VESSELS:
        binary_col = f"{vessel}_FFRHYPEREMIE_binary"
        n_valid = df_ffr.groupby("StudyInstanceUID")[binary_col].apply(lambda x: x.notna().any()).sum()
        n_abnormal = df_ffr.groupby("StudyInstanceUID")[binary_col].apply(lambda x: (x == 1).any()).sum()
        if n_valid > 0:
            print(f"    {vessel}: {n_valid} studies ({n_abnormal} abnormal, {n_valid - n_abnormal} normal)")

    # Use existing splits from parquet
    print("\n  Split distribution:")
    for split in ["train", "val", "test"]:
        n_rows = (df_ffr["Split"] == split).sum()
        n_stud = df_ffr.loc[df_ffr["Split"] == split, "StudyInstanceUID"].nunique()
        print(f"    {split}: {n_stud} studies, {n_rows} rows")

    # Drop rows with NaN FileName
    df_ffr = df_ffr.dropna(subset=["FileName"])
    print(f"  Rows after dropping NaN FileName: {len(df_ffr)}")

    # Clean view_class
    df_ffr["view_class"] = df_ffr["view_class"].replace({"None": np.nan, "": np.nan})
    n_with_view = df_ffr["view_class"].notna().sum()
    print(f"\n  Rows with view_class: {n_with_view}/{len(df_ffr)}")

    # Keep only relevant columns
    keep_cols = ["FileName", "StudyInstanceUID", "Split", "view_class"]
    for vessel in VESSELS:
        keep_cols.append(f"{vessel}_FFRHYPEREMIE_binary")
    keep_cols = [c for c in keep_cols if c in df_ffr.columns]
    df_out = df_ffr[keep_cols].copy()

    # Save with alpha separator
    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_CSV, sep=SEP, index=False)
    print(f"\nSaved to: {OUTPUT_CSV}")
    print(f"  Shape: {df_out.shape}")
    print(f"  Columns: {list(df_out.columns)}")


if __name__ == "__main__":
    main()
