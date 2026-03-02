"""
Final script with FULL GLOBAL MICRO METRICS + RCA/LCA breakdown:
- Produces all_metrics CSV (per segment)
- Produces diagnostic CSV per head
- Computes GLOBAL MICRO metrics for ALL metrics
- NEW: Computes RCA/LCA-specific micro metrics by pathology
- Writes full all_metrics table, diagnostic table, COMPLETE global metrics, and RCA/LCA breakdown into text report
- Uses percentile bootstrap (local RNG) for CI (default n_bootstrap=1000)
"""
import os
from datetime import datetime
import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    mean_absolute_error
)

# -------------------------
# CONFIG - edit here
# -------------------------
csv_path = "/volume/deepcoro_clip_newwbranch/models_evaluation/preds_with_true_filled_20251020_233850.csv"
output_dir = "/volume/deepcoro_clip_newwbranch/models_evaluation/code+resultas_avecglobalmetrics"
n_bootstrap_default = 1000
confidence_level = 0.95

# -------------------------
# Utility: bootstrap percentile CI (local RNG)
# -------------------------
def bootstrap_ci_with_flag(yt, ys, stat_fn, n_boot=n_bootstrap_default, seed=42, conf_level=confidence_level):
    """
    Returns (stat, lo, hi, ci_available)
    - If not enough valid bootstrap samples, returns (stat0, stat0, stat0, False)
    """
    yt = np.asarray(yt)
    ys = np.asarray(ys)
    if len(yt) == 0:
        return np.nan, np.nan, np.nan, False
    try:
        rng = np.random.default_rng(seed)
    except Exception:
        rng = np.random.RandomState(seed)
    stats = []
    n = len(yt)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            val = stat_fn(yt[idx], ys[idx])
            if val is None:
                continue
            if isinstance(val, float) and np.isnan(val):
                continue
            stats.append(float(val))
        except Exception:
            continue
    # point estimate
    try:
        stat0 = stat_fn(yt, ys)
    except Exception:
        stat0 = np.nan
    # decide availability
    if len(stats) < max(10, int(0.05 * n_boot)):
        return stat0, stat0, stat0, False
    alpha = 1 - conf_level
    lo = np.percentile(stats, 100 * alpha / 2)
    hi = np.percentile(stats, 100 * (1 - alpha / 2))
    return stat0, lo, hi, True

# -------------------------
# stat functions
# -------------------------
def stat_auc(yt, ys):
    if len(np.unique(yt)) <= 1:
        return np.nan
    try:
        return float(roc_auc_score(yt, ys))
    except Exception:
        return np.nan

def stat_auprc(yt, ys):
    if len(np.unique(yt)) <= 1:
        return np.nan
    try:
        return float(average_precision_score(yt, ys))
    except Exception:
        return np.nan

def stat_mae(yt, ys):
    try:
        return float(mean_absolute_error(yt, ys))
    except Exception:
        return np.nan

def stat_pearson(yt, ys):
    if len(np.unique(yt)) <= 1 or len(np.unique(ys)) <= 1:
        return np.nan
    try:
        r, _ = pearsonr(yt, ys)
        return float(r) if not np.isnan(r) else np.nan
    except Exception:
        return np.nan

def youden_info(yt, ys):
    """Return (threshold, sens_at_thresh, spec_at_thresh)."""
    try:
        if len(np.unique(yt)) <= 1:
            return np.nan, np.nan, np.nan
        fpr, tpr, thresh = roc_curve(yt, ys)
        spec = 1 - fpr
        youden = tpr - fpr
        idx = np.nanargmax(youden)
        return float(thresh[idx]), float(tpr[idx]), float(spec[idx])
    except Exception:
        return np.nan, np.nan, np.nan

# -------------------------
# map head -> segment and coronary
# -------------------------
def map_head_to_segment(head):
    h = head.lower()
    if 'left_main' in h:
        return 'Left Main', 'LCA'
    if 'prox_lad' in h:
        return 'Proximal LAD', 'LCA'
    if 'mid_lad' in h:
        return 'Mid LAD', 'LCA'
    if 'dist_lad' in h:
        return 'Distal LAD', 'LCA'
    if 'd1' in h:
        return 'D1', 'LCA'
    if 'd2' in h:
        return 'D2', 'LCA'
    if 'prox_lcx' in h:
        return 'Proximal LCX', 'LCA'
    if 'mid_lcx' in h:
        return 'Mid LCX', 'LCA'
    if 'dist_lcx' in h:
        return 'Distal LCX', 'LCA'
    if 'lvp' in h:
        return 'LVP', 'LCA'
    if 'om1' in h:
        return 'OM1', 'LCA'
    if 'om2' in h:
        return 'OM2', 'LCA'
    if 'bx' in h:
        return 'BX', 'LCA'
    if 'prox_rca' in h:
        return 'Proximal RCA', 'RCA'
    if 'mid_rca' in h:
        return 'Mid RCA', 'RCA'
    if 'dist_rca' in h:
        return 'Distal RCA', 'RCA'
    if 'pda' in h:
        return 'PDA', 'RCA'
    if 'posterolateral' in h:
        return 'Posterolateral', 'RCA'
    return 'Other', 'Unknown'

segments_order = [
    'Left Main', 'Proximal LAD', 'Mid LAD', 'Distal LAD', 'D1', 'D2',
    'Proximal LCX', 'Mid LCX', 'Distal LCX', 'LVP', 'OM1', 'OM2',
    'Proximal RCA', 'Mid RCA', 'Distal RCA', 'PDA', 'Posterolateral'
]

# -------------------------
# Load dataframe and identify pred columns
# -------------------------
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()
pred_cols = [c for c in df.columns if c.endswith('_pred')]

# classify stenosis regression vs binary
stenosis_cols = [c for c in pred_cols if 'stenosis' in c and 'binary' not in c]
binary_cols = [c for c in pred_cols if c not in stenosis_cols]

# -------------------------
# Build consolidated table (rows per Segment + Pathology)
# -------------------------
rows_all = []
base_seed = 42

for seg in segments_order:
    for pathology in ['stenosis', 'stenosis_binary', 'calcif_binary', 'cto', 'thrombus']:
        heads = []
        for col in pred_cols:
            base = col[:-5]
            seg_name, _ = map_head_to_segment(base)
            if seg_name != seg:
                continue
            if pathology == 'stenosis':
                if 'stenosis' in base and 'binary' not in base:
                    heads.append(base)
            elif pathology == 'stenosis_binary':
                if 'stenosis_binary' in base:
                    heads.append(base)
            elif pathology == 'thrombus':
                if 'thrombus' in base:
                    heads.append(base)
            else:
                if pathology.replace('_binary','') in base or pathology in base:
                    heads.append(base)
        if not heads:
            continue

        y_true_all = []
        y_score_all = []
        n_abnormal = 0
        n_total = 0
        for base in heads:
            tcol = base + '_true'
            pcol = base + '_pred'
            if tcol not in df.columns or pcol not in df.columns:
                continue
            mask = ~df[tcol].isna() & ~df[pcol].isna()
            if mask.sum() == 0:
                continue
            yt = df.loc[mask, tcol].values
            ys = df.loc[mask, pcol].values
            y_true_all.extend(list(yt))
            y_score_all.extend(list(ys))
            n_total += len(yt)
            n_abnormal += int((np.array(yt) > 0).sum())

        if n_total == 0:
            continue

        yt = np.array(y_true_all)
        ys = np.array(y_score_all)
        prevalence = (n_abnormal / n_total) * 100.0

        # init
        auc_val = auprc_val = sens_val = spec_val = youden_th = np.nan
        auc_lo = auc_hi = auprc_lo = auprc_hi = sens_lo = sens_hi = spec_lo = spec_hi = np.nan
        mae_val = mae_lo = mae_hi = pear_val = pear_lo = pear_hi = np.nan
        auc_ci_ok = auprc_ci_ok = sens_ci_ok = spec_ci_ok = mae_ci_ok = pear_ci_ok = False

        if pathology == 'stenosis':
            mae_val, mae_lo, mae_hi, mae_ci_ok = bootstrap_ci_with_flag(yt, ys, stat_mae, n_boot=n_bootstrap_default, seed=base_seed+1)
            pear_val, pear_lo, pear_hi, pear_ci_ok = bootstrap_ci_with_flag(yt, ys, stat_pearson, n_boot=n_bootstrap_default, seed=base_seed+2)
        else:
            auc_val, auc_lo, auc_hi, auc_ci_ok = bootstrap_ci_with_flag(yt, ys, stat_auc, n_boot=n_bootstrap_default, seed=base_seed+10)
            auprc_val, auprc_lo, auprc_hi, auprc_ci_ok = bootstrap_ci_with_flag(yt, ys, stat_auprc, n_boot=n_bootstrap_default, seed=base_seed+11)
            youden_th_point, sens_point, spec_point = youden_info(yt, ys)
            youden_th = youden_th_point
            sens_val, sens_lo, sens_hi, sens_ci_ok = bootstrap_ci_with_flag(yt, ys, lambda a,b: youden_info(a,b)[1], n_boot=n_bootstrap_default, seed=base_seed+12)
            spec_val, spec_lo, spec_hi, spec_ci_ok = bootstrap_ci_with_flag(yt, ys, lambda a,b: youden_info(a,b)[2], n_boot=n_bootstrap_default, seed=base_seed+13)

        row = {
            'Segment': seg,
            'Pathology': pathology.upper(),
            'Heads_pooled_count': len(heads),
            'n_abnormal': n_abnormal,
            'n_total': n_total,
            'Prevalence_pct': round(prevalence, 3),

            'AUC': round(auc_val, 4) if not np.isnan(auc_val) else None,
            'AUC_CI_lo': round(auc_lo, 4) if not np.isnan(auc_lo) else None,
            'AUC_CI_hi': round(auc_hi, 4) if not np.isnan(auc_hi) else None,
            'AUC_CI_available': bool(auc_ci_ok),

            'AUPRC': round(auprc_val, 4) if not np.isnan(auprc_val) else None,
            'AUPRC_CI_lo': round(auprc_lo, 4) if not np.isnan(auprc_lo) else None,
            'AUPRC_CI_hi': round(auprc_hi, 4) if not np.isnan(auprc_hi) else None,
            'AUPRC_CI_available': bool(auprc_ci_ok),

            'Youden_threshold': round(youden_th, 4) if not np.isnan(youden_th) else None,
            'Sens_Youden': round(sens_val, 4) if not np.isnan(sens_val) else None,
            'Sens_Youden_CI_lo': round(sens_lo, 4) if not np.isnan(sens_lo) else None,
            'Sens_Youden_CI_hi': round(sens_hi, 4) if not np.isnan(sens_hi) else None,
            'Sens_Youden_CI_available': bool(sens_ci_ok),
            'Spec_Youden': round(spec_val, 4) if not np.isnan(spec_val) else None,
            'Spec_Youden_CI_lo': round(spec_lo, 4) if not np.isnan(spec_lo) else None,
            'Spec_Youden_CI_hi': round(spec_hi, 4) if not np.isnan(spec_hi) else None,
            'Spec_Youden_CI_available': bool(spec_ci_ok),

            'MAE': round(mae_val, 4) if not np.isnan(mae_val) else None,
            'MAE_CI_lo': round(mae_lo, 4) if not np.isnan(mae_lo) else None,
            'MAE_CI_hi': round(mae_hi, 4) if not np.isnan(mae_hi) else None,
            'MAE_CI_available': bool(mae_ci_ok),
            'Pearson_r': round(pear_val, 4) if not np.isnan(pear_val) else None,
            'Pearson_r_CI_lo': round(pear_lo, 4) if not np.isnan(pear_lo) else None,
            'Pearson_r_CI_hi': round(pear_hi, 4) if not np.isnan(pear_hi) else None,
            'Pearson_CI_available': bool(pear_ci_ok),
        }
        rows_all.append(row)

# -------------------------
# Save consolidated CSV and diagnostic CSV
# -------------------------
all_df = pd.DataFrame(rows_all)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(output_dir, exist_ok=True)
all_csv = os.path.join(output_dir, f"all_metrics_{ts}.csv")
all_df.to_csv(all_csv, index=False)

# Diagnostic per head
diag_rows = []
for col in pred_cols:
    base = col[:-5]
    tcol = base + '_true'
    pcol = base + '_pred'
    if tcol not in df.columns or pcol not in df.columns:
        continue
    mask = ~df[tcol].isna() & ~df[pcol].isna()
    if mask.sum() == 0:
        continue
    yt = df.loc[mask, tcol].values
    ys = df.loc[mask, pcol].values
    n_total = len(yt)
    n_abnormal = int((np.array(yt) > 0).sum())
    prevalence = (n_abnormal / n_total)*100 if n_total>0 else np.nan
    try:
        mae_pt = mean_absolute_error(yt, ys)
    except Exception:
        mae_pt = np.nan
    try:
        if len(np.unique(yt)) > 1 and len(np.unique(ys)) > 1:
            pear_pt = pearsonr(yt, ys)[0]
        else:
            pear_pt = np.nan
    except Exception:
        pear_pt = np.nan
    try:
        auc_pt = roc_auc_score(yt, ys) if len(np.unique(yt))>1 else np.nan
    except Exception:
        auc_pt = np.nan
    try:
        auprc_pt = average_precision_score(yt, ys) if len(np.unique(yt))>1 else np.nan
    except Exception:
        auprc_pt = np.nan
    th, sens_pt, spec_pt = youden_info(yt, ys)
    diag_rows.append({
        'head': base, 'n_total': n_total, 'n_abnormal': n_abnormal, 'prevalence_pct': round(prevalence,3),
        'MAE': mae_pt, 'Pearson': pear_pt, 'AUC': auc_pt, 'AUPRC': auprc_pt,
        'Youden_threshold': th, 'Sens_Youden': sens_pt, 'Spec_Youden': spec_pt
    })
diag_df = pd.DataFrame(diag_rows)
diag_csv = os.path.join(output_dir, f"diagnostic_per_head_{ts}.csv")
diag_df.to_csv(diag_csv, index=False)

# -------------------------
# GLOBAL MICRO METRICS - ALL METRICS (Binary + Regression)
# -------------------------
report_lines = []
report_lines.append("="*120)
report_lines.append("ALL METRICS REPORT")
report_lines.append("="*120)
report_lines.append(f"Source file: {csv_path}")
report_lines.append(f"Generated: {datetime.now().isoformat()}")
report_lines.append("")

# ====================
# GLOBAL BINARY METRICS (MICRO POOLED)
# ====================
y_true_global_binary = []
y_score_global_binary = []
for col in binary_cols:
    base = col[:-5]
    tcol = base + '_true'
    if tcol not in df.columns:
        continue
    mask = ~df[tcol].isna() & ~df[col].isna()
    if mask.sum() == 0:
        continue
    y_true_global_binary.extend(list(df.loc[mask, tcol].values))
    y_score_global_binary.extend(list(df.loc[mask, col].values))

report_lines.append("### GLOBAL MICRO-POOLED BINARY METRICS (all binary tasks pooled):")
report_lines.append("")
if len(y_true_global_binary) > 0:
    yt_g_bin = np.array(y_true_global_binary)
    ys_g_bin = np.array(y_score_global_binary)
    n_total_bin = len(yt_g_bin)
    n_abnormal_bin = int((yt_g_bin > 0).sum())
    prevalence_bin = (n_abnormal_bin / n_total_bin) * 100.0
    
    # AUC
    auc_g, auc_lo_g, auc_hi_g, auc_ci_ok_g = bootstrap_ci_with_flag(
        yt_g_bin, ys_g_bin, stat_auc, n_boot=n_bootstrap_default, seed=999
    )
    # AUPRC
    auprc_g, auprc_lo_g, auprc_hi_g, auprc_ci_ok_g = bootstrap_ci_with_flag(
        yt_g_bin, ys_g_bin, stat_auprc, n_boot=n_bootstrap_default, seed=1000
    )
    # Youden + Sens + Spec
    youden_th_g, sens_g_point, spec_g_point = youden_info(yt_g_bin, ys_g_bin)
    sens_g, sens_lo_g, sens_hi_g, sens_ci_ok_g = bootstrap_ci_with_flag(
        yt_g_bin, ys_g_bin, lambda a,b: youden_info(a,b)[1], n_boot=n_bootstrap_default, seed=1001
    )
    spec_g, spec_lo_g, spec_hi_g, spec_ci_ok_g = bootstrap_ci_with_flag(
        yt_g_bin, ys_g_bin, lambda a,b: youden_info(a,b)[2], n_boot=n_bootstrap_default, seed=1002
    )
    
    report_lines.append(f"Total pooled examples: {n_total_bin}")
    report_lines.append(f"Total abnormal: {n_abnormal_bin}")
    report_lines.append(f"Prevalence: {prevalence_bin:.3f}%")
    report_lines.append("")
    report_lines.append(f"AUROC:  {auc_g:.4f}  [{auc_lo_g:.4f} - {auc_hi_g:.4f}]  CI_available: {auc_ci_ok_g}")
    report_lines.append(f"AUPRC:  {auprc_g:.4f}  [{auprc_lo_g:.4f} - {auprc_hi_g:.4f}]  CI_available: {auprc_ci_ok_g}")
    report_lines.append(f"Youden threshold: {youden_th_g:.4f}")
    report_lines.append(f"Sensitivity@Youden: {sens_g:.4f}  [{sens_lo_g:.4f} - {sens_hi_g:.4f}]  CI_available: {sens_ci_ok_g}")
    report_lines.append(f"Specificity@Youden: {spec_g:.4f}  [{spec_lo_g:.4f} - {spec_hi_g:.4f}]  CI_available: {spec_ci_ok_g}")
else:
    report_lines.append("No pooled binary examples found for global binary metrics.")

report_lines.append("")
report_lines.append("-"*120)
report_lines.append("")

# ====================
# GLOBAL REGRESSION METRICS (MICRO POOLED) - for stenosis
# ====================
y_true_global_reg = []
y_score_global_reg = []
for col in stenosis_cols:
    base = col[:-5]
    tcol = base + '_true'
    if tcol not in df.columns:
        continue
    mask = ~df[tcol].isna() & ~df[col].isna()
    if mask.sum() == 0:
        continue
    y_true_global_reg.extend(list(df.loc[mask, tcol].values))
    y_score_global_reg.extend(list(df.loc[mask, col].values))

report_lines.append("### GLOBAL MICRO-POOLED REGRESSION METRICS (all stenosis regression tasks pooled):")
report_lines.append("")
if len(y_true_global_reg) > 0:
    yt_g_reg = np.array(y_true_global_reg)
    ys_g_reg = np.array(y_score_global_reg)
    n_total_reg = len(yt_g_reg)
    n_abnormal_reg = int((yt_g_reg > 0).sum())
    prevalence_reg = (n_abnormal_reg / n_total_reg) * 100.0
    
    # MAE
    mae_g, mae_lo_g, mae_hi_g, mae_ci_ok_g = bootstrap_ci_with_flag(
        yt_g_reg, ys_g_reg, stat_mae, n_boot=n_bootstrap_default, seed=2000
    )
    # Pearson
    pear_g, pear_lo_g, pear_hi_g, pear_ci_ok_g = bootstrap_ci_with_flag(
        yt_g_reg, ys_g_reg, stat_pearson, n_boot=n_bootstrap_default, seed=2001
    )
    
    report_lines.append(f"Total pooled examples: {n_total_reg}")
    report_lines.append(f"Total abnormal (>0): {n_abnormal_reg}")
    report_lines.append(f"Prevalence: {prevalence_reg:.3f}%")
    report_lines.append("")
    report_lines.append(f"MAE:  {mae_g:.4f}  [{mae_lo_g:.4f} - {mae_hi_g:.4f}]  CI_available: {mae_ci_ok_g}")
    report_lines.append(f"Pearson r:  {pear_g:.4f}  [{pear_lo_g:.4f} - {pear_hi_g:.4f}]  CI_available: {pear_ci_ok_g}")
else:
    report_lines.append("No pooled regression (stenosis) examples found for global regression metrics.")

report_lines.append("")
report_lines.append("="*120)
report_lines.append("")

# ========================================
# NEW: RCA/LCA BREAKDOWN BY PATHOLOGY
# ========================================
report_lines.append("### MICRO-POOLED METRICS BY CORONARY ARTERY (RCA/LCA) AND PATHOLOGY:")
report_lines.append("")

for coronary in ['RCA', 'LCA']:
    coronary_name = "Right Coronary Artery (RCA)" if coronary == 'RCA' else "Left Coronary Artery (LCA)"
    report_lines.append("="*120)
    report_lines.append(f"{coronary_name}")
    report_lines.append("="*120)
    report_lines.append("")
    
    for pathology in ['stenosis', 'stenosis_binary', 'calcif_binary', 'cto', 'thrombus']:
        # Collect all data for this coronary + pathology
        y_true_coronary = []
        y_score_coronary = []
        
        for col in pred_cols:
            base = col[:-5]
            seg_name, coronary_type = map_head_to_segment(base)
            
            if coronary_type != coronary:
                continue
            
            # Filter by pathology
            if pathology == 'stenosis':
                if 'stenosis' not in base or 'binary' in base:
                    continue
            elif pathology == 'stenosis_binary':
                if 'stenosis_binary' not in base:
                    continue
            elif pathology == 'thrombus':
                if 'thrombus' not in base:
                    continue
            elif pathology == 'calcif_binary':
                if 'calcif' not in base:
                    continue
            elif pathology == 'cto':
                if 'cto' not in base:
                    continue
            
            tcol = base + '_true'
            pcol = base + '_pred'
            if tcol not in df.columns or pcol not in df.columns:
                continue
            
            mask = ~df[tcol].isna() & ~df[pcol].isna()
            if mask.sum() == 0:
                continue
            
            y_true_coronary.extend(list(df.loc[mask, tcol].values))
            y_score_coronary.extend(list(df.loc[mask, pcol].values))
        
        if len(y_true_coronary) == 0:
            continue
        
        yt_cor = np.array(y_true_coronary)
        ys_cor = np.array(y_score_coronary)
        n_total_cor = len(yt_cor)
        n_abnormal_cor = int((yt_cor > 0).sum())
        prevalence_cor = (n_abnormal_cor / n_total_cor) * 100.0
        
        report_lines.append(f"├── {pathology.upper().replace('_', ' ')}")
        report_lines.append(f"│   ├── Total examples: {n_total_cor}")
        report_lines.append(f"│   ├── Abnormal: {n_abnormal_cor}")
        report_lines.append(f"│   ├── Prevalence: {prevalence_cor:.3f}%")
        
        if pathology == 'stenosis':
            # Regression metrics
            mae_cor, mae_lo_cor, mae_hi_cor, mae_ci_ok_cor = bootstrap_ci_with_flag(
                yt_cor, ys_cor, stat_mae, n_boot=n_bootstrap_default, seed=3000+hash(coronary+pathology)%1000
            )
            pear_cor, pear_lo_cor, pear_hi_cor, pear_ci_ok_cor = bootstrap_ci_with_flag(
                yt_cor, ys_cor, stat_pearson, n_boot=n_bootstrap_default, seed=4000+hash(coronary+pathology)%1000
            )
            report_lines.append(f"│   ├── MAE: {mae_cor:.4f} [{mae_lo_cor:.4f} - {mae_hi_cor:.4f}] (CI: {mae_ci_ok_cor})")
            report_lines.append(f"│   └── Pearson r: {pear_cor:.4f} [{pear_lo_cor:.4f} - {pear_hi_cor:.4f}] (CI: {pear_ci_ok_cor})")
        else:
            # Binary classification metrics
            auc_cor, auc_lo_cor, auc_hi_cor, auc_ci_ok_cor = bootstrap_ci_with_flag(
                yt_cor, ys_cor, stat_auc, n_boot=n_bootstrap_default, seed=5000+hash(coronary+pathology)%1000
            )
            auprc_cor, auprc_lo_cor, auprc_hi_cor, auprc_ci_ok_cor = bootstrap_ci_with_flag(
                yt_cor, ys_cor, stat_auprc, n_boot=n_bootstrap_default, seed=6000+hash(coronary+pathology)%1000
            )
            youden_th_cor, sens_cor_point, spec_cor_point = youden_info(yt_cor, ys_cor)
            sens_cor, sens_lo_cor, sens_hi_cor, sens_ci_ok_cor = bootstrap_ci_with_flag(
                yt_cor, ys_cor, lambda a,b: youden_info(a,b)[1], n_boot=n_bootstrap_default, seed=7000+hash(coronary+pathology)%1000
            )
            spec_cor, spec_lo_cor, spec_hi_cor, spec_ci_ok_cor = bootstrap_ci_with_flag(
                yt_cor, ys_cor, lambda a,b: youden_info(a,b)[2], n_boot=n_bootstrap_default, seed=8000+hash(coronary+pathology)%1000
            )
            report_lines.append(f"│   ├── AUC: {auc_cor:.4f} [{auc_lo_cor:.4f} - {auc_hi_cor:.4f}] (CI: {auc_ci_ok_cor})")
            report_lines.append(f"│   ├── AUPRC: {auprc_cor:.4f} [{auprc_lo_cor:.4f} - {auprc_hi_cor:.4f}] (CI: {auprc_ci_ok_cor})")
            report_lines.append(f"│   ├── Youden threshold: {youden_th_cor:.4f}")
            report_lines.append(f"│   ├── Sensitivity@Youden: {sens_cor:.4f} [{sens_lo_cor:.4f} - {sens_hi_cor:.4f}] (CI: {sens_ci_ok_cor})")
            report_lines.append(f"│   └── Specificity@Youden: {spec_cor:.4f} [{spec_lo_cor:.4f} - {spec_hi_cor:.4f}] (CI: {spec_ci_ok_cor})")
        
        report_lines.append("")
    
    report_lines.append("")

report_lines.append("="*120)
report_lines.append("")

# -------------------------
# Text report: include FULL tables (all_metrics + diagnostic)
# -------------------------
report_lines.append("### FULL consolidated table (all_metrics - per Segment + Pathology):")
report_lines.append("")
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
report_lines.append(all_df.to_string(index=False))
report_lines.append("")
report_lines.append("="*120)
report_lines.append("")
report_lines.append("### Diagnostic per head (diagnostic_per_head):")
report_lines.append("")
report_lines.append(diag_df.to_string(index=False))
report_lines.append("")

report_txt = os.path.join(output_dir, f"all_metrics_report_{ts}.txt")
with open(report_txt, 'w', encoding='utf-8') as f:
    f.write("\n".join(report_lines))

# -------------------------
# Print summary
# -------------------------
print("="*80)
print("COMPUTATION COMPLETE")
print("="*80)
print(f"Saved consolidated metrics CSV: {all_csv}")
print(f"Saved diagnostic per head CSV: {diag_csv}")
print(f"Saved text report with FULL GLOBAL MICRO METRICS + RCA/LCA: {report_txt}")
print(f"Rows (Segment+Pathology): {len(all_df)}")
print("="*80)