#!/usr/bin/env python3
"""Prepare DeepIFR training dataset from the main DeepCORO CSV.

Filters to studies with valid IFR measurements, creates binary IFR columns
(IFR <= 0.89 = 1 abnormal, > 0.89 = 0 normal), fresh patient-level splits,
and caps train to 100 studies. Saves with alpha separator.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Config
SOURCE_CSV = "/media/data1/datasets/DeepCoro_CLIP/CTO_THROMBUS_STENOSIS_70_CALCIF_inference_with_binary.csv"
VIEW_PARQUET = "/media/data1/datasets/DeepCoro/2b_CathReport_HEMO_MHI_MERGED_2017-2024_VIDEO_LEVEL_with_status_and_split.parquet"
OUTPUT_CSV = "/volume/DeepCORO_CLIP/data/deepifr_training_100studies.csv"
SEP = "α"
IFR_THRESHOLD = 0.89
TRAIN_LIMIT = 100
SEED = 42

# 17 vessels with IFR data (excluding lvp which has 0)
VESSELS = [
    "prox_rca", "mid_rca", "dist_rca", "pda", "posterolateral",
    "left_main",
    "prox_lad", "mid_lad", "dist_lad", "D1", "D2",
    "prox_lcx", "mid_lcx", "dist_lcx", "om1", "om2", "bx",
]


def main():
    print(f"Reading source CSV: {SOURCE_CSV}")
    df = pd.read_csv(SOURCE_CSV, sep=SEP, engine="python")
    print(f"  Total rows: {len(df)}, unique studies: {df['StudyInstanceUID'].nunique()}")

    # IFR columns for the 17 vessels
    ifr_cols = [f"{v}_IFRHYPER" for v in VESSELS]
    missing_cols = [c for c in ifr_cols if c not in df.columns]
    if missing_cols:
        print(f"  WARNING: Missing IFR columns: {missing_cols}")
        for c in missing_cols:
            df[c] = np.nan

    # Convert IFR values: -1 and 0 -> NaN (invalid/missing measurements)
    for col in ifr_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] <= 0, col] = np.nan

    # Filter to studies with at least one valid IFR measurement
    has_valid_ifr = df[ifr_cols].notna().any(axis=1)
    studies_with_ifr = df.loc[has_valid_ifr, "StudyInstanceUID"].unique()
    df_ifr = df[df["StudyInstanceUID"].isin(studies_with_ifr)].copy()
    print(f"  Studies with valid IFR: {len(studies_with_ifr)}")
    print(f"  Rows after filtering: {len(df_ifr)}")

    # Create binary IFR columns: <= 0.89 = 1 (abnormal/ischemic), > 0.89 = 0 (normal)
    # NaN where IFR is missing
    for vessel in VESSELS:
        ifr_col = f"{vessel}_IFRHYPER"
        binary_col = f"{vessel}_IFRHYPER_binary"
        df_ifr[binary_col] = np.nan
        valid_mask = df_ifr[ifr_col].notna()
        df_ifr.loc[valid_mask, binary_col] = (df_ifr.loc[valid_mask, ifr_col] <= IFR_THRESHOLD).astype(float)

    # Print IFR data availability per vessel
    print(f"\n  IFR data availability (threshold: IFR <= {IFR_THRESHOLD} = abnormal):")
    for vessel in VESSELS:
        binary_col = f"{vessel}_IFRHYPER_binary"
        n_valid = df_ifr.groupby("StudyInstanceUID")[binary_col].apply(lambda x: x.notna().any()).sum()
        n_abnormal = df_ifr.groupby("StudyInstanceUID")[binary_col].apply(lambda x: (x == 1).any()).sum()
        if n_valid > 0:
            print(f"    {vessel}: {n_valid} studies ({n_abnormal} abnormal, {n_valid - n_abnormal} normal)")

    # Create fresh patient-level splits (80/10/10)
    rng = np.random.RandomState(SEED)
    study_ids = df_ifr["StudyInstanceUID"].unique().copy()
    rng.shuffle(study_ids)

    n = len(study_ids)
    n_val = int(n * 0.10)
    n_test = int(n * 0.10)

    val_studies = set(study_ids[:n_val])
    test_studies = set(study_ids[n_val:n_val + n_test])
    train_studies_all = list(study_ids[n_val + n_test:])

    # Cap train to TRAIN_LIMIT studies, move excess to test
    if len(train_studies_all) > TRAIN_LIMIT:
        rng.shuffle(train_studies_all)
        excess = train_studies_all[TRAIN_LIMIT:]
        train_studies_all = train_studies_all[:TRAIN_LIMIT]
        test_studies.update(excess)
        print(f"\n  Capped train from {n - n_val - n_test} to {TRAIN_LIMIT} studies ({len(excess)} moved to test)")

    train_studies = set(train_studies_all)

    # Assign fresh splits
    df_ifr["Split"] = "test"
    df_ifr.loc[df_ifr["StudyInstanceUID"].isin(train_studies), "Split"] = "train"
    df_ifr.loc[df_ifr["StudyInstanceUID"].isin(val_studies), "Split"] = "val"

    # Final split counts
    print("\n  Final splits:")
    for split in ["train", "val", "test"]:
        n_rows = (df_ifr["Split"] == split).sum()
        n_stud = df_ifr.loc[df_ifr["Split"] == split, "StudyInstanceUID"].nunique()
        print(f"    {split}: {n_stud} studies, {n_rows} rows")

    # Merge view_class from parquet file (overrides any existing view_class from source CSV)
    print(f"\n  Merging view_class from: {VIEW_PARQUET}")
    if "view_class" in df_ifr.columns:
        df_ifr = df_ifr.drop(columns=["view_class"])
    df_view = pd.read_parquet(VIEW_PARQUET, columns=["FileName", "view_class"])
    df_view = df_view.dropna(subset=["FileName"])
    df_view = df_view.drop_duplicates(subset="FileName", keep="first")
    df_view["view_class"] = df_view["view_class"].replace({"None": np.nan, "": np.nan})
    df_ifr = df_ifr.merge(df_view, on="FileName", how="left")
    n_with_view = df_ifr["view_class"].notna().sum()
    print(f"  Rows with view_class: {n_with_view}/{len(df_ifr)}")
    print(f"  View class distribution:")
    print(df_ifr["view_class"].value_counts().to_string(header=False))

    # Keep only relevant columns: identifiers + view + binary IFR columns
    keep_cols = ["FileName", "StudyInstanceUID", "Split", "view_class"]
    for vessel in VESSELS:
        keep_cols.append(f"{vessel}_IFRHYPER_binary")
    keep_cols = [c for c in keep_cols if c in df_ifr.columns]
    df_out = df_ifr[keep_cols].copy()

    # Save with alpha separator
    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_CSV, sep=SEP, index=False)
    print(f"\nSaved to: {OUTPUT_CSV}")
    print(f"  Shape: {df_out.shape}")
    print(f"  Columns: {list(df_out.columns)}")


if __name__ == "__main__":
    main()
