#!/usr/bin/env python3
"""
Regenerate Dataset with Bug Fixes

Applies the following fixes to the dataset pipeline:
1. Acquisition-time sort: Use DICOM timestamps from SOP UIDs in filenames
   (YYYYMMDDHHMMSS) for correct temporal ordering, including procedures
   that span midnight.  Falls back to series_time / SeriesTime.
2. PCI cascade gating: stent_presence_class=1 → always PCI, but the
   POST_PCI cascade only triggers when GT *_pcidone confirms PCI on the
   labelled coronary side.
3. Dominance override: If lvp_stenosis > 0, force left-dominant regardless
   of the dominance label.
4. LVP/PDA mutual exclusivity: In left-dominant hearts exclude PDA
   (LVP ≡ PDA); in right-dominant hearts exclude LVP.
5. Congenital exclusion: Filter out congenital-procedure studies and
   studies with all stenosis = -1/NaN from inference.

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
    sort_by_study_and_time,
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

    # ── 3. Sort by acquisition time (shared helper) ──────────────────────────
    df = sort_by_study_and_time(df)

    # ── 4. Re-assign status ──────────────────────────────────────────────────
    df = assign_procedure_status(df)

    if old_status is not None:
        # Use index-aligned comparison (not .values) so row reordering is safe
        changed = (df["status"] != old_status.reindex(df.index)).sum()
        logger.info("Status changes vs original: %d / %d (%.2f%%)",
                     changed, len(df), 100.0 * changed / len(df))

    # ── 5. Save updated parquet ──────────────────────────────────────────────
    logger.info("Saving updated parquet to %s", OUTPUT_PARQUET)
    df.to_parquet(OUTPUT_PARQUET, index=False)
    logger.info("Parquet saved (%d rows)", len(df))

    # ── 6. Filter for inference (diagnostic + Left/Right Coronary + contrast) ─
    #      Also exclude congenital-procedure studies and studies with no stenosis data.
    congenital_mask = df["series_description"].str.contains(
        "CONGENITAL", case=False, na=False
    )
    stenosis_cols = [c for c in df.columns if c.endswith("_stenosis")]
    stenosis_vals = df[stenosis_cols].apply(pd.to_numeric, errors="coerce")
    no_stenosis_mask = ((stenosis_vals == -1) | stenosis_vals.isna()).all(axis=1)

    n_cong = congenital_mask.sum()
    n_nosten = no_stenosis_mask.sum()
    if n_cong:
        logger.info("Excluding %d congenital-procedure videos (%d studies)",
                     n_cong, df.loc[congenital_mask, "StudyInstanceUID"].nunique())
    if n_nosten:
        logger.info("Excluding %d videos with all stenosis = -1/NaN", n_nosten)

    mask = (
        (df["status"] == "diagnostic")
        & (df["main_structure_name"].isin(["Left Coronary", "Right Coronary"]))
        & (df["contrast_agent_class"] == 1)
        & ~congenital_mask
        & ~no_stenosis_mask
    )
    df_diag = df.loc[mask].copy()
    logger.info("Diagnostic + Left/Right + contrast=1 (excl. congenital): %d rows", len(df_diag))

    # ── 7. Map Split: test → inference (to match existing convention) ────────
    df_diag["Split"] = df_diag["Split"].replace({"test": "inference"})
    logger.info("Split distribution:\n%s", df_diag["Split"].value_counts())

    # ── 8. Generate reports (dominance override + LVP/PDA) ───────────────────
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
