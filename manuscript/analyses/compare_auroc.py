#!/usr/bin/env python3
"""
Compare AUROC between two model inference runs:
  - zcb8cu0l (4,808 exams)
  - Sarra (4,828 exams)

Ground truth is loaded from the inference CSV dataset and merged by StudyInstanceUID.
AUROCs are computed on the intersection of exams for fair comparison.
"""

import pandas as pd
import numpy as np
import ast
import os
from sklearn.metrics import roc_auc_score

# ============================================================
# 1. Load predictions
# ============================================================
ZCB_PATH = (
    "/volume/DeepCORO_CLIP/outputs/DeepCORO_video_linear_probing/"
    "DeepCORO_stenosis_weighted_distal/20260223-162254_no_wandb/"
    "predictions/inference_predictions_epoch_-1.csv"
)
SARRA_PATH = (
    "/volume/DeepCORO_CLIP/outputs/DeepCORO_video_linear_probing/"
    "DeepCORO_video_linear_probing_multiview_improved_cls_token/"
    "20260223-163608_no_wandb/predictions/inference_predictions_epoch_-1.csv"
)
GT_PATH = "/media/data1/datasets/DeepCoro_CLIP/CTO_THROMBUS_STENOSIS_70_CALCIF_inference_with_binary.csv"

print("Loading zcb8cu0l predictions...")
zcb_df = pd.read_csv(ZCB_PATH)
print(f"  -> {len(zcb_df)} rows")

print("Loading Sarra predictions...")
sarra_df = pd.read_csv(SARRA_PATH)
print(f"  -> {len(sarra_df)} rows")

# ============================================================
# 2. Extract StudyInstanceUID from video_name
# ============================================================
def extract_study_uid(video_name_str):
    """Parse the list-of-paths string and extract the StudyInstanceUID
    from the first non-PAD path."""
    paths = ast.literal_eval(video_name_str)
    for p in paths:
        if p != "PAD":
            fname = os.path.basename(p)
            return fname.split("_")[0]
    return None

print("\nExtracting StudyInstanceUIDs...")
zcb_df["StudyInstanceUID"] = zcb_df["video_name"].apply(extract_study_uid)
sarra_df["StudyInstanceUID"] = sarra_df["video_name"].apply(extract_study_uid)

print(f"  zcb8cu0l unique studies: {zcb_df['StudyInstanceUID'].nunique()}")
print(f"  Sarra unique studies:    {sarra_df['StudyInstanceUID'].nunique()}")

# ============================================================
# 3. Load ground truth from the dataset CSV
# ============================================================
# Identify which columns we need from GT
segments = [
    "prox_rca", "mid_rca", "dist_rca", "pda", "posterolateral",
    "left_main", "prox_lad", "mid_lad", "dist_lad",
    "D1", "D2", "prox_lcx", "mid_lcx", "dist_lcx",
    "om1", "om2", "bx", "lvp",
]

gt_cols_needed = ["StudyInstanceUID", "Split"]
for seg in segments:
    gt_cols_needed.append(f"{seg}_stenosis_binary")
    gt_cols_needed.append(f"{seg}_calcif_binary")
    gt_cols_needed.append(f"{seg}_cto")
    gt_cols_needed.append(f"{seg}_thrombus")

print("\nLoading ground truth dataset...")
gt_all = pd.read_csv(GT_PATH, sep="\u03b1", engine="python", usecols=gt_cols_needed)
print(f"  -> {len(gt_all)} rows total")

# Keep only inference split and deduplicate by StudyInstanceUID
gt_inf = gt_all[gt_all["Split"] == "inference"].copy()
print(f"  -> {len(gt_inf)} inference rows")

# Ground truth is the same across all videos of a study, so just take the first per study
gt_study = gt_inf.groupby("StudyInstanceUID").first().reset_index()
gt_study.drop(columns=["Split"], inplace=True)
print(f"  -> {gt_study['StudyInstanceUID'].nunique()} unique inference studies with GT")

# ============================================================
# 4. Find common exams (intersection)
# ============================================================
common_uids = (
    set(zcb_df["StudyInstanceUID"]) &
    set(sarra_df["StudyInstanceUID"]) &
    set(gt_study["StudyInstanceUID"])
)
print(f"\nCommon exams across zcb8cu0l, Sarra, and GT: {len(common_uids)}")

# Filter and merge
zcb_common = zcb_df[zcb_df["StudyInstanceUID"].isin(common_uids)].copy()
sarra_common = sarra_df[sarra_df["StudyInstanceUID"].isin(common_uids)].copy()
gt_common = gt_study[gt_study["StudyInstanceUID"].isin(common_uids)].copy()

# Sort all by StudyInstanceUID for alignment
zcb_common = zcb_common.sort_values("StudyInstanceUID").reset_index(drop=True)
sarra_common = sarra_common.sort_values("StudyInstanceUID").reset_index(drop=True)
gt_common = gt_common.sort_values("StudyInstanceUID").reset_index(drop=True)

# Verify alignment
assert list(zcb_common["StudyInstanceUID"]) == list(gt_common["StudyInstanceUID"]), \
    "StudyInstanceUID mismatch between zcb8cu0l and GT"
assert list(sarra_common["StudyInstanceUID"]) == list(gt_common["StudyInstanceUID"]), \
    "StudyInstanceUID mismatch between Sarra and GT"

print(f"  zcb8cu0l common: {len(zcb_common)}")
print(f"  Sarra common:    {len(sarra_common)}")
print(f"  GT common:       {len(gt_common)}")

# ============================================================
# 5. Compute AUROCs
# ============================================================
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def compute_auroc(y_true, y_pred_logit):
    """Compute AUROC. Returns NaN if only one class present or all NaN."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred_logit))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred_logit[mask]
    if len(y_true_clean) < 2:
        return np.nan
    unique = np.unique(y_true_clean)
    if len(unique) < 2:
        return np.nan
    y_prob = sigmoid(y_pred_clean)
    return roc_auc_score(y_true_clean, y_prob)

# --- Stenosis Binary heads ---
print("\n" + "=" * 80)
print("STENOSIS BINARY AUROC COMPARISON (Common exams, n={})".format(len(gt_common)))
print("=" * 80)
print(f"{'Segment':<30} {'zcb8cu0l':>10} {'Sarra':>10} {'Diff':>10}  {'GT pos%':>8}")
print("-" * 80)

stenosis_results = []
for seg in segments:
    head = f"{seg}_stenosis_binary"
    pred_col = f"{head}_pred"
    gt_col = head  # in GT dataset

    y_true = gt_common[gt_col].values.astype(float)
    zcb_pred = zcb_common[pred_col].values.astype(float)
    sarra_pred = sarra_common[pred_col].values.astype(float)

    zcb_auroc = compute_auroc(y_true, zcb_pred)
    sarra_auroc = compute_auroc(y_true, sarra_pred)

    if np.isnan(zcb_auroc) or np.isnan(sarra_auroc):
        diff = np.nan
        diff_str = "N/A"
    else:
        diff = zcb_auroc - sarra_auroc
        diff_str = f"{diff:+.4f}"

    pos_rate = np.nanmean(y_true) * 100

    print(f"{seg:<30} {zcb_auroc:>10.4f} {sarra_auroc:>10.4f} {diff_str:>10}  {pos_rate:>7.1f}%")

    stenosis_results.append({
        "head_type": "stenosis_binary",
        "segment": seg,
        "zcb8cu0l_auroc": zcb_auroc,
        "sarra_auroc": sarra_auroc,
        "difference": diff,
        "gt_positive_pct": pos_rate,
        "n_samples": int(np.sum(~np.isnan(y_true))),
    })

# Compute mean across valid heads
valid_zcb = [r["zcb8cu0l_auroc"] for r in stenosis_results if not np.isnan(r["zcb8cu0l_auroc"])]
valid_sarra = [r["sarra_auroc"] for r in stenosis_results if not np.isnan(r["sarra_auroc"])]
mean_zcb = np.mean(valid_zcb) if valid_zcb else np.nan
mean_sarra = np.mean(valid_sarra) if valid_sarra else np.nan
mean_diff = mean_zcb - mean_sarra if not (np.isnan(mean_zcb) or np.isnan(mean_sarra)) else np.nan

print("-" * 80)
print(f"{'MEAN (stenosis_binary)':<30} {mean_zcb:>10.4f} {mean_sarra:>10.4f} {mean_diff:>+10.4f}")
print(f"  ({len(valid_zcb)}/{len(segments)} heads with valid AUROC)")

# --- Calcif, CTO, Thrombus heads ---
other_head_types = [
    ("calcif_binary", "_calcif_binary"),
    ("cto", "_cto"),
    ("thrombus", "_thrombus"),
]

all_results = list(stenosis_results)

for head_label, suffix in other_head_types:
    print(f"\n{'=' * 80}")
    print(f"{head_label.upper()} AUROC COMPARISON (Common exams, n={len(gt_common)})")
    print("=" * 80)
    print(f"{'Segment':<30} {'zcb8cu0l':>10} {'Sarra':>10} {'Diff':>10}  {'GT pos%':>8}")
    print("-" * 80)

    head_results = []
    for seg in segments:
        head = f"{seg}{suffix}"
        pred_col = f"{head}_pred"
        gt_col = head

        if gt_col not in gt_common.columns:
            continue

        y_true = gt_common[gt_col].values.astype(float)
        zcb_pred = zcb_common[pred_col].values.astype(float)
        sarra_pred = sarra_common[pred_col].values.astype(float)

        zcb_auroc = compute_auroc(y_true, zcb_pred)
        sarra_auroc = compute_auroc(y_true, sarra_pred)

        if np.isnan(zcb_auroc) or np.isnan(sarra_auroc):
            diff = np.nan
            diff_str = "N/A"
        else:
            diff = zcb_auroc - sarra_auroc
            diff_str = f"{diff:+.4f}"

        pos_rate = np.nanmean(y_true) * 100

        print(f"{seg:<30} {zcb_auroc:>10.4f} {sarra_auroc:>10.4f} {diff_str:>10}  {pos_rate:>7.1f}%")

        head_results.append({
            "head_type": head_label,
            "segment": seg,
            "zcb8cu0l_auroc": zcb_auroc,
            "sarra_auroc": sarra_auroc,
            "difference": diff,
            "gt_positive_pct": pos_rate,
            "n_samples": int(np.sum(~np.isnan(y_true))),
        })

    valid_zcb_h = [r["zcb8cu0l_auroc"] for r in head_results if not np.isnan(r["zcb8cu0l_auroc"])]
    valid_sarra_h = [r["sarra_auroc"] for r in head_results if not np.isnan(r["sarra_auroc"])]
    mean_zcb_h = np.mean(valid_zcb_h) if valid_zcb_h else np.nan
    mean_sarra_h = np.mean(valid_sarra_h) if valid_sarra_h else np.nan
    mean_diff_h = mean_zcb_h - mean_sarra_h if not (np.isnan(mean_zcb_h) or np.isnan(mean_sarra_h)) else np.nan

    print("-" * 80)
    print(f"{'MEAN (' + head_label + ')':<30} {mean_zcb_h:>10.4f} {mean_sarra_h:>10.4f} {mean_diff_h:>+10.4f}")
    print(f"  ({len(valid_zcb_h)}/{len(segments)} heads with valid AUROC)")

    all_results.extend(head_results)

# ============================================================
# 6. Overall summary
# ============================================================
print(f"\n{'=' * 80}")
print("OVERALL SUMMARY")
print("=" * 80)

for ht in ["stenosis_binary", "calcif_binary", "cto", "thrombus"]:
    subset = [r for r in all_results if r["head_type"] == ht]
    v_zcb = [r["zcb8cu0l_auroc"] for r in subset if not np.isnan(r["zcb8cu0l_auroc"])]
    v_sar = [r["sarra_auroc"] for r in subset if not np.isnan(r["sarra_auroc"])]
    m_zcb = np.mean(v_zcb) if v_zcb else np.nan
    m_sar = np.mean(v_sar) if v_sar else np.nan
    m_diff = m_zcb - m_sar if not (np.isnan(m_zcb) or np.isnan(m_sar)) else np.nan
    winner = "zcb8cu0l" if m_diff > 0 else "Sarra" if m_diff < 0 else "Tie"
    print(f"  {ht:<20}: zcb8cu0l={m_zcb:.4f}  Sarra={m_sar:.4f}  diff={m_diff:+.4f}  ({len(v_zcb)}/{len(subset)} heads)  Winner: {winner}")

# ============================================================
# 7. Save results to CSV
# ============================================================
out_path = "/volume/DeepCORO_CLIP/outputs/auroc_comparison_zcb8cu0l_vs_sarra.csv"
results_df = pd.DataFrame(all_results)
results_df.to_csv(out_path, index=False)
print(f"\nResults saved to: {out_path}")
print(f"Total rows: {len(results_df)}")
