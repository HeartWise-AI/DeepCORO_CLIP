#!/usr/bin/env python3
"""
Regenerate Dataset with Bug Fixes

Applies three fixes to the dataset pipeline:
1. SeriesTime sort bug: Use series_time (int64, full coverage) instead of
   SeriesTime (float64, ~86% NaN) for temporal ordering.
2. Dominance override: If lvp_stenosis > 0, force left-dominant regardless of label.
3. LVP/PDA mutual exclusivity: In left-dominant hearts exclude PDA (LVP ≡ PDA);
   in right-dominant hearts exclude LVP.

Outputs:
  - Updated parquet with corrected status assignments
  - Diagnostic-only inference CSV with regenerated reports (same splits)
"""

import sys
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

tqdm.pandas()

# Add project root so we can import generate_dataset
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataset_creation.generate_dataset import (
    assign_procedure_status,
    create_report,
    _extract_acq_time_from_filename,
    MAIN_STRUCTURE_MAP,
    DOMINANCE_MAP,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
INPUT_PARQUET = Path(
    "/media/data1/datasets/DeepCoro/"
    "2b_CathReport_HEMO_MHI_MERGED_2017-2024_VIDEO_LEVEL_with_status_and_split.parquet"
)
OUTPUT_DIR = Path("/volume/DeepCORO_CLIP/output_dataset/regenerated")
OUTPUT_PARQUET = OUTPUT_DIR / "2b_CathReport_HEMO_MHI_MERGED_2017-2024_VIDEO_LEVEL_with_status_and_split.parquet"
OUTPUT_INFERENCE_CSV = OUTPUT_DIR / "CTO_THROMBUS_STENOSIS_70_CALCIF_inference.csv"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load ──────────────────────────────────────────────────────────────
    logger.info("Loading parquet from %s", INPUT_PARQUET)
    df = pd.read_parquet(INPUT_PARQUET)
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    # Preserve original status for comparison
    old_status = df["status"].copy() if "status" in df.columns else None

    # ── 2. Apply mappings ────────────────────────────────────────────────────
    if "main_structure_class" in df.columns:
        df["main_structure_name"] = df["main_structure_class"].map(MAIN_STRUCTURE_MAP)
    if "dominance_class" in df.columns:
        df["dominance_name"] = df["dominance_class"].map(DOMINANCE_MAP)

    # ── 3. Sort by acquisition time from filename (FIX #1) ──────────────────
    # The DICOM acquisition timestamp embedded in SOP Instance UIDs is the most
    # reliable time source.  series_time can be a transfer/storage timestamp
    # with corrupted hour values that break sort order within a study.
    logger.info("Extracting acquisition time from filenames...")
    df["_acq_time"] = df["FileName"].apply(_extract_acq_time_from_filename)
    acq_coverage = df["_acq_time"].notna().sum()
    logger.info("Filename acq_time: %d / %d (%.1f%%)",
                acq_coverage, len(df), 100.0 * acq_coverage / len(df))

    # Fallback to series_time for videos without filename timestamps
    if "series_time" in df.columns:
        fallback = pd.to_numeric(df["series_time"], errors="coerce")
        fallback = fallback.where(fallback > 0)
    elif "SeriesTime" in df.columns:
        fallback = pd.to_numeric(df["SeriesTime"], errors="coerce")
    else:
        fallback = pd.Series(np.nan, index=df.index)

    df["_sort_time"] = df["_acq_time"].fillna(fallback)
    df = df.sort_values(["StudyInstanceUID", "_sort_time"], na_position="last")
    df.drop(columns=["_acq_time", "_sort_time"], inplace=True)
    logger.info("Sorted by StudyInstanceUID + acquisition time")

    # ── 4. Re-assign status ──────────────────────────────────────────────────
    df = assign_procedure_status(df)

    if old_status is not None:
        changed = (df["status"].values != old_status.values).sum()
        logger.info("Status changes vs original: %d / %d (%.2f%%)",
                     changed, len(df), 100.0 * changed / len(df))

    # ── 5. Save updated parquet ──────────────────────────────────────────────
    logger.info("Saving updated parquet to %s", OUTPUT_PARQUET)
    df.to_parquet(OUTPUT_PARQUET, index=False)
    logger.info("Parquet saved (%d rows)", len(df))

    # ── 6. Filter for inference (diagnostic + Left/Right Coronary + contrast) ─
    mask = (
        (df["status"] == "diagnostic")
        & (df["main_structure_name"].isin(["Left Coronary", "Right Coronary"]))
        & (df["contrast_agent_class"] == 1)
    )
    df_diag = df.loc[mask].copy()
    logger.info("Diagnostic + Left/Right + contrast=1: %d rows", len(df_diag))

    # ── 7. Map Split: test → inference (to match existing convention) ────────
    df_diag["Split"] = df_diag["Split"].replace({"test": "inference"})
    logger.info("Split distribution:\n%s", df_diag["Split"].value_counts())

    # ── 8. Generate reports (FIX #2 + #3: dominance override + LVP/PDA) ─────
    logger.info("Generating reports for %d diagnostic videos...", len(df_diag))
    df_diag["Report"] = df_diag.progress_apply(
        lambda r: create_report(r, coronary_specific_report=True), axis=1
    )

    # ── 9. Save inference CSV (alpha-separated to match existing format) ─────
    logger.info("Saving inference CSV to %s", OUTPUT_INFERENCE_CSV)
    df_diag.to_csv(OUTPUT_INFERENCE_CSV, index=False, sep="α")
    logger.info("Inference CSV saved (%d rows)", len(df_diag))

    # ── 10. Summary ──────────────────────────────────────────────────────────
    logger.info("\n=== SUMMARY ===")
    logger.info("Parquet: %s (%d rows)", OUTPUT_PARQUET, len(df))
    logger.info("  Status: %s", df["status"].value_counts().to_dict())
    logger.info("Inference: %s (%d rows)", OUTPUT_INFERENCE_CSV, len(df_diag))
    logger.info("  Split: %s", df_diag["Split"].value_counts().to_dict())

    # Check lvp override stats
    lvp_vals = pd.to_numeric(df_diag.get("lvp_stenosis", pd.Series(dtype=float)), errors="coerce").fillna(0)
    lvp_overrides = (lvp_vals > 0).sum()
    logger.info("  Dominance overrides (lvp>0 → left): %d videos", lvp_overrides)


if __name__ == "__main__":
    main()
