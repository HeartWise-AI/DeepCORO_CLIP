# compute_all_metrics_with_text_table.py
"""
Final script:
- Produces all_metrics CSV
- Produces diagnostic CSV per head
- Writes full all_metrics table and diagnostic table into a text report (all_metrics_report_<ts>.txt)
- Uses percentile bootstrap (local RNG) for CI (default n_bootstrap=1000)
- Computes AUC, AUPRC, Youden threshold, Sens/Spec at Youden, MAE, Pearson, ICs, prevalence, n_abnormal/n_total
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
csv_path = "/volume/Deep-Coro-final-branch/mes_scripts/ancienresultat3/merged_predictions_with_truth.csv"
output_dir = "/volume/Deep-Coro-final-branch/mes_scripts/resultats4"
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
    if len(np.unique(yt)) <= 1:
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
# map head -> segment
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
        pear_pt = pearsonr(yt, ys)[0]
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
# Text report: include FULL tables (all_metrics + diagnostic) + global summary
# -------------------------
report_lines = []
report_lines.append("="*120)
report_lines.append("ALL METRICS REPORT")
report_lines.append("="*120)
report_lines.append(f"Source file: {csv_path}")
report_lines.append(f"Generated: {datetime.now().isoformat()}")
report_lines.append("")
report_lines.append("### FULL consolidated table (all_metrics):")
# Insert full table as a string
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
report_lines.append(all_df.to_string(index=False))
report_lines.append("")
report_lines.append("### Diagnostic per head (diagnostic_per_head):")
report_lines.append(diag_df.to_string(index=False))
report_lines.append("")

# Global pooled binary metrics
y_true_global = []
y_score_global = []
for col in binary_cols:
    base = col[:-5]
    tcol = base + '_true'
    if tcol not in df.columns:
        continue
    mask = ~df[tcol].isna() & ~df[col].isna()
    if mask.sum() == 0:
        continue
    y_true_global.extend(list(df.loc[mask, tcol].values))
    y_score_global.extend(list(df.loc[mask, col].values))

report_lines.append("### GLOBAL POOLED BINARY METRICS (micro):")
if len(y_true_global) > 0:
    yt_g = np.array(y_true_global)
    ys_g = np.array(y_score_global)
    auc_g, auc_lo_g, auc_hi_g, auc_ci_ok_g = bootstrap_ci_with_flag(yt_g, ys_g, stat_auc, n_boot=n_bootstrap_default, seed=999)
    auprc_g, auprc_lo_g, auprc_hi_g, auprc_ci_ok_g = bootstrap_ci_with_flag(yt_g, ys_g, stat_auprc, n_boot=n_bootstrap_default, seed=1000)
    report_lines.append(f"Micro AUROC (pool all binary): {auc_g:.4f}  CI: ({auc_lo_g:.4f} - {auc_hi_g:.4f})  CI_available: {auc_ci_ok_g}")
    report_lines.append(f"Micro AUPRC (pool all binary): {auprc_g:.4f} CI: ({auprc_lo_g:.4f} - {auprc_hi_g:.4f})  CI_available: {auprc_ci_ok_g}")
    report_lines.append(f"Total pooled examples (binary cols): {len(yt_g)}")
else:
    report_lines.append("No pooled binary examples found for global metrics.")

report_txt = os.path.join(output_dir, f"all_metrics_report_{ts}.txt")
with open(report_txt, 'w', encoding='utf-8') as f:
    f.write("\n".join(report_lines))

# -------------------------
# Print summary
# -------------------------
print("Saved consolidated metrics CSV:", all_csv)
print("Saved diagnostic per head CSV:", diag_csv)
print("Saved text report with full tables:", report_txt)
print("Rows (Segment+Pathology):", len(all_df))
