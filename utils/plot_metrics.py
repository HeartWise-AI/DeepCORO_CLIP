import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import os
import re # For sorting epoch numbers
from tqdm import tqdm # Ensure tqdm is installed

# Initialize tqdm for pandas
tqdm.pandas()

# --- Core Helper Functions ---
def mode(lst):
    """Return the most common element in lst (tie => any of top)."""
    if not lst or all(pd.isna(x) for x in lst):
        return None
    filtered_lst = [item for item in lst if pd.notna(item)]
    if not filtered_lst:
        return None
    return Counter(filtered_lst).most_common(1)[0][0]

def is_valid(x):
    """Return True if x is not NaN and not -1 or '-1'."""
    if pd.isna(x):
        return False
    if str(x) == "-1" or str(x) == "-1.0": # Handles -1 and -1.0
        return False
    return True

def build_val_text_index_map(df_dataset, key_col="val_text_index"):
    """Build a dict: val_text_index -> list of row(s)."""
    index_map = {}
    if key_col not in df_dataset.columns:
        print(f"Error: Key column '{key_col}' not found in df_dataset for build_val_text_index_map.")
        return index_map
    for _, row in df_dataset.iterrows():
        val_idx = row[key_col]
        if pd.isna(val_idx):
            continue
        val_idx = int(val_idx)
        if val_idx not in index_map:
            index_map[val_idx] = []
        index_map[val_idx].append(row)
    return index_map

# --- Aggregation and Metrics Functions ---
def aggregate_predictions_for_epoch(
    val_text_map,
    predictions_df,
    topk=5,
    vessel_labels=None
):
    """Uses pre-built val_text_map for O(1) lookups."""
    if vessel_labels is None:
        vessel_labels = [
            "leftmain_stenosis", "lad_stenosis", "mid_lad_stenosis", "dist_lad_stenosis",
            "diagonal_stenosis", "D2_stenosis", "D3_stenosis", "lcx_stenosis",
            "dist_lcx_stenosis", "lvp_stenosis", "marg_d_stenosis", "om1_stenosis",
            "om2_stenosis", "om3_stenosis", "prox_rca_stenosis", "mid_rca_stenosis",
            "dist_rca_stenosis", "RVG1_stenosis", "RVG2_stenosis", "pda_stenosis",
            "posterolateral_stenosis", "bx_stenosis", "lima_or_svg_stenosis",
        ]

    aggregated_rows = []
    if 'ground_truth_idx' not in predictions_df.columns:
        print("Error: 'ground_truth_idx' not found in predictions_df for aggregation.")
        return pd.DataFrame()

    for _, row in predictions_df.iterrows():
        gt_idx = row["ground_truth_idx"]
        if pd.isna(gt_idx): continue
        gt_idx = int(gt_idx)
        if gt_idx not in val_text_map:
            continue

        gt_data_list = val_text_map[gt_idx]
        if not gt_data_list: continue
        gt_row_series = gt_data_list[0] # Use the first GT entry

        predicted_rows_data = []
        for k_ in range(1, topk + 1):
            pred_col_name = f"predicted_idx_{k_}"
            if pred_col_name in row and pd.notna(row[pred_col_name]):
                pred_idx = int(row[pred_col_name])
                if pred_idx in val_text_map and val_text_map[pred_idx]:
                    predicted_rows_data.append(val_text_map[pred_idx][0])
        
        current_agg_dict = {}
        # Determine FileName for the aggregated row
        if "FileName" in row and pd.notna(row["FileName"]):
            current_agg_dict["FileName"] = row["FileName"]
        elif "FileName" in gt_row_series and pd.notna(gt_row_series["FileName"]): # Fallback to GT FileName
             current_agg_dict["FileName"] = gt_row_series["FileName"]
        elif "VideoFileNames" in row and pd.notna(row["VideoFileNames"]) and isinstance(row["VideoFileNames"], str): # Fallback to VideoFileNames
            current_agg_dict["FileName"] = row["VideoFileNames"].split(';')[0]
        else:
            current_agg_dict["FileName"] = f"unknown_gt_{gt_idx}" # Ultimate fallback

        current_agg_dict["ground_truth_idx"] = gt_idx

        for vessel_label_name in vessel_labels:
            vessel_prefix = vessel_label_name.replace("_stenosis", "")
            # Stenosis
            pred_sten_values = [pr_data[vessel_label_name] for pr_data in predicted_rows_data if vessel_label_name in pr_data and is_valid(pr_data[vessel_label_name])]
            current_agg_dict[f"predicted_{vessel_label_name}"] = np.nanmean(pred_sten_values) if pred_sten_values else np.nan
            gt_sten_val = gt_row_series.get(vessel_label_name)
            current_agg_dict[vessel_label_name] = gt_sten_val if is_valid(gt_sten_val) else np.nan

            # IFR
            ifr_col_name = vessel_prefix + "_IFRHYPEREMIE"
            if ifr_col_name in gt_row_series.index: # Check if GT expects IFR for this vessel
                pred_ifr_values = [pr_data[ifr_col_name] for pr_data in predicted_rows_data if ifr_col_name in pr_data and is_valid(pr_data[ifr_col_name])]
                current_agg_dict[f"predicted_{ifr_col_name}"] = np.nanmean(pred_ifr_values) if pred_ifr_values else np.nan
                gt_ifr_val = gt_row_series.get(ifr_col_name)
                current_agg_dict[ifr_col_name] = gt_ifr_val if is_valid(gt_ifr_val) else np.nan
            else:
                current_agg_dict[f"predicted_{ifr_col_name}"] = np.nan
                current_agg_dict[ifr_col_name] = np.nan

            # Calcification
            calcif_col_name = vessel_prefix + "_calcif"
            if calcif_col_name in gt_row_series.index: # Check if GT expects Calcif
                pred_calcif_values = [pr_data[calcif_col_name] for pr_data in predicted_rows_data if calcif_col_name in pr_data and is_valid(pr_data[calcif_col_name])]
                current_agg_dict[f"predicted_{calcif_col_name}"] = mode(pred_calcif_values) # mode handles empty list
                gt_calcif_val = gt_row_series.get(calcif_col_name)
                current_agg_dict[calcif_col_name] = gt_calcif_val if is_valid(gt_calcif_val) else None
            else:
                current_agg_dict[f"predicted_{calcif_col_name}"] = None
                current_agg_dict[calcif_col_name] = None
        
        # Coronary Dominance - Study Level
        if "coronary_dominance" in gt_row_series.index:
            pred_cd_values = [pr_data["coronary_dominance"] for pr_data in predicted_rows_data if "coronary_dominance" in pr_data and is_valid(pr_data["coronary_dominance"])]
            current_agg_dict["predicted_coronary_dominance"] = mode(pred_cd_values)
            gt_cd_val = gt_row_series.get("coronary_dominance")
            current_agg_dict["coronary_dominance"] = gt_cd_val if is_valid(gt_cd_val) else None
        else:
            current_agg_dict["predicted_coronary_dominance"] = None
            current_agg_dict["coronary_dominance"] = None
            
        aggregated_rows.append(current_agg_dict)
    return pd.DataFrame(aggregated_rows)

def compute_metrics(agg_df, vessel_labels):
    """Computes metrics for each vessel segment."""
    metrics = {
        "stenosis": {"mae": {}, "corr": {}},
        "ifr": {"mae": {}, "corr": {}},
        "calcif": {"accuracy": {}}
    }
    if agg_df.empty: return metrics

    for vessel in vessel_labels:
        gt_sten_col, pred_sten_col = vessel, f"predicted_{vessel}"
        if gt_sten_col in agg_df.columns and pred_sten_col in agg_df.columns:
            valid_stenosis = agg_df[[gt_sten_col, pred_sten_col]].dropna()
            if not valid_stenosis.empty:
                metrics["stenosis"]["mae"][vessel] = np.mean(np.abs(valid_stenosis[gt_sten_col] - valid_stenosis[pred_sten_col]))
                metrics["stenosis"]["corr"][vessel] = valid_stenosis[gt_sten_col].corr(valid_stenosis[pred_sten_col]) if len(valid_stenosis) > 1 and valid_stenosis[gt_sten_col].nunique() > 1 and valid_stenosis[pred_sten_col].nunique() > 1 else np.nan
            else:
                metrics["stenosis"]["mae"][vessel], metrics["stenosis"]["corr"][vessel] = np.nan, np.nan
        else:
            metrics["stenosis"]["mae"][vessel], metrics["stenosis"]["corr"][vessel] = np.nan, np.nan

        prefix = vessel.replace("_stenosis", "")
        gt_ifr_col, pred_ifr_col = f"{prefix}_IFRHYPEREMIE", f"predicted_{prefix}_IFRHYPEREMIE"
        if gt_ifr_col in agg_df.columns and pred_ifr_col in agg_df.columns:
            valid_ifr = agg_df[[gt_ifr_col, pred_ifr_col]].dropna()
            if not valid_ifr.empty:
                metrics["ifr"]["mae"][vessel] = np.mean(np.abs(valid_ifr[gt_ifr_col] - valid_ifr[pred_ifr_col]))
                metrics["ifr"]["corr"][vessel] = valid_ifr[gt_ifr_col].corr(valid_ifr[pred_ifr_col]) if len(valid_ifr) > 1 and valid_ifr[gt_ifr_col].nunique() > 1 and valid_ifr[pred_ifr_col].nunique() > 1 else np.nan
            else:
                metrics["ifr"]["mae"][vessel], metrics["ifr"]["corr"][vessel] = np.nan, np.nan
        else:
            metrics["ifr"]["mae"][vessel], metrics["ifr"]["corr"][vessel] = np.nan, np.nan
            
        gt_calcif_col, pred_calcif_col = f"{prefix}_calcif", f"predicted_{prefix}_calcif"
        if gt_calcif_col in agg_df.columns and pred_calcif_col in agg_df.columns:
            valid_calcif = agg_df[[gt_calcif_col, pred_calcif_col]].dropna()
            metrics["calcif"]["accuracy"][vessel] = np.mean(valid_calcif[gt_calcif_col] == valid_calcif[pred_calcif_col]) if not valid_calcif.empty else np.nan
        else:
            metrics["calcif"]["accuracy"][vessel] = np.nan

    gt_cd_col, pred_cd_col = "coronary_dominance", "predicted_coronary_dominance"
    if gt_cd_col in agg_df.columns and pred_cd_col in agg_df.columns:
        valid_cd = agg_df[[gt_cd_col, pred_cd_col]].dropna()
        metrics["coronary_dominance"] = {"accuracy": np.mean(valid_cd[gt_cd_col] == valid_cd[pred_cd_col])} if not valid_cd.empty else {"accuracy": np.nan}
    else:
        metrics["coronary_dominance"] = {"accuracy": np.nan}
    return metrics

# --- Plotting Functions ---
def plot_epoch_metrics_line_charts(results_dict):
    """Create line plots of average metrics across ALL vessel labels."""
    if not results_dict:
        print("No results to plot for overall epoch metrics.")
        return

    def get_epoch_num_from_key(key_str):
        match = re.search(r'epoch(\d+)', key_str)
        return int(match.group(1)) if match else -1

    sorted_epoch_keys = sorted(results_dict.keys(), key=get_epoch_num_from_key)
    if not sorted_epoch_keys or (len(sorted_epoch_keys) > 0 and get_epoch_num_from_key(sorted_epoch_keys[0]) == -1 and any(get_epoch_num_from_key(k) != -1 for k in sorted_epoch_keys)) :
        print("Warning: Could not sort all epochs numerically, using alphanumeric sort for overall metrics.")
        sorted_epoch_keys = sorted(results_dict.keys())

    plot_data = {
        "stenosis_mae": [], "ifr_mae": [],
        "stenosis_corr": [], "ifr_corr": [],
        "calcif_accuracy": [], "dominance_acc": [],
        "x_labels": [key.replace(".csv", "") for key in sorted_epoch_keys]
    }

    for ep_key in sorted_epoch_keys:
        metrics = results_dict[ep_key]["metrics"]
        for metric_category_key, expected_sub_metrics in [
            ("stenosis", ["mae", "corr"]), 
            ("ifr", ["mae", "corr"]), 
            ("calcif", ["accuracy"])
        ]:
            for sub_metric_key_name in expected_sub_metrics:
                plot_data_key = f"{metric_category_key}_{sub_metric_key_name}"
                metric_values_dict = metrics.get(metric_category_key, {}).get(sub_metric_key_name, {})
                if isinstance(metric_values_dict, dict): # Ensure it's a dict of vessel-specific values
                    metric_values = metric_values_dict.values()
                else: # Should not happen if metrics structure is correct
                    metric_values = [] 
                
                valid_metric_values = [v for v in metric_values if pd.notna(v)]
                average_value = np.nanmean(valid_metric_values) if valid_metric_values else np.nan
                
                # Initialize list in plot_data if it's the first epoch for this metric
                if plot_data_key not in plot_data: plot_data[plot_data_key] = []
                plot_data[plot_data_key].append(average_value)

        plot_data["dominance_acc"].append(metrics.get("coronary_dominance", {}).get("accuracy", np.nan))
    
    num_expected_points = len(plot_data["x_labels"])
    for key, value_list in plot_data.items():
        if key != "x_labels":
            if len(value_list) < num_expected_points:
                plot_data[key].extend([np.nan] * (num_expected_points - len(value_list)))
            elif len(value_list) > num_expected_points:
                 plot_data[key] = value_list[:num_expected_points]

    if not any(any(pd.notna(val) for val in data_list) for key, data_list in plot_data.items() if key != "x_labels"):
        print("No valid overall metric data available to plot across epochs.")
        return

    fig_main, axes_main = plt.subplots(nrows=3, ncols=1, figsize=(14, 18), sharex=True) 
    fig_main.suptitle("Overall Model Performance Metrics (All Vessels) Across Epochs", fontsize=18, y=0.98)
    # ... (Plotting code remains the same as v4) ...
    axes_main[0].plot(plot_data["x_labels"], plot_data["stenosis_mae"], marker='o', label="Stenosis MAE (Avg)", color="dodgerblue", linestyle='-')
    axes_main[0].set_title("Average Stenosis MAE", fontsize=14); axes_main[0].set_ylabel("Mean Absolute Error", fontsize=12); axes_main[0].legend(fontsize=10); axes_main[0].grid(True, linestyle=':', alpha=0.6)

    axes_main[1].plot(plot_data["x_labels"], plot_data["stenosis_corr"], marker='X', label="Stenosis Correlation (Avg)", color="forestgreen", linestyle='-')
    axes_main[1].set_title("Average Stenosis Correlation", fontsize=14); axes_main[1].set_ylabel("Avg Pearson Correlation", fontsize=12)
    valid_sten_corr = [v for v in plot_data["stenosis_corr"] if pd.notna(v)]; axes_main[1].set_ylim(min(0, np.min(valid_sten_corr))-0.1 if valid_sten_corr else 0, max(1, np.max(valid_sten_corr))+0.1 if valid_sten_corr else 1); axes_main[1].legend(fontsize=10); axes_main[1].grid(True, linestyle=':', alpha=0.6)

    axes_main[2].plot(plot_data["x_labels"], plot_data["calcif_accuracy"], marker='s', label="Calcification Accuracy (Avg)", color="darkorange", linestyle='-')
    axes_main[2].plot(plot_data["x_labels"], plot_data["dominance_acc"], marker='D', label="Coronary Dominance Accuracy", color="purple", linestyle='--')
    axes_main[2].set_title("Average Accuracy Scores", fontsize=14); axes_main[2].set_ylabel("Accuracy", fontsize=12)
    valid_acc_data = plot_data.get("calcif_accuracy", []) + plot_data.get("dominance_acc", []) # Ensure keys exist
    valid_acc = [v for v in valid_acc_data if pd.notna(v)]; axes_main[2].set_ylim(min(0, np.min(valid_acc))-0.1 if valid_acc else 0, max(1, np.max(valid_acc))+0.1 if valid_acc else 1); axes_main[2].set_xlabel("Epoch", fontsize=12); axes_main[2].legend(fontsize=10); axes_main[2].grid(True, linestyle=':', alpha=0.6)
    plt.setp(axes_main[2].get_xticklabels(), rotation=30, ha="right", fontsize=10); plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()

    if "ifr_mae" in plot_data and any(pd.notna(v) for v in plot_data["ifr_mae"]):
        fig_ifr_mae, ax_ifr_mae = plt.subplots(figsize=(14, 6))
        ax_ifr_mae.plot(plot_data["x_labels"], plot_data["ifr_mae"], marker='o', label="IFR MAE (Avg)", color="crimson", linestyle='-')
        ax_ifr_mae.set_title("Average IFR MAE across Epochs", fontsize=14); ax_ifr_mae.set_ylabel("Mean Absolute Error", fontsize=12); ax_ifr_mae.set_xlabel("Epoch", fontsize=12); ax_ifr_mae.legend(fontsize=10); ax_ifr_mae.grid(True, linestyle=':', alpha=0.6); plt.setp(ax_ifr_mae.get_xticklabels(), rotation=30, ha="right", fontsize=10); plt.tight_layout(); plt.show()
    else: print("No overall IFR MAE data to plot or all values are NaN.")

    if "ifr_corr" in plot_data and any(pd.notna(v) for v in plot_data["ifr_corr"]):
        fig_ifr_corr, ax_ifr_corr = plt.subplots(figsize=(14, 6))
        ax_ifr_corr.plot(plot_data["x_labels"], plot_data["ifr_corr"], marker='X', label="IFR Correlation (Avg)", color="teal", linestyle='-')
        ax_ifr_corr.set_title("Average IFR Correlation across Epochs", fontsize=14); ax_ifr_corr.set_ylabel("Avg Pearson Correlation", fontsize=12)
        valid_ifr_corr = [v for v in plot_data["ifr_corr"] if pd.notna(v)]; ax_ifr_corr.set_ylim(min(0, np.min(valid_ifr_corr))-0.1 if valid_ifr_corr else 0, max(1, np.max(valid_ifr_corr))+0.1 if valid_ifr_corr else 1); ax_ifr_corr.set_xlabel("Epoch", fontsize=12); ax_ifr_corr.legend(fontsize=10); ax_ifr_corr.grid(True, linestyle=':', alpha=0.6); plt.setp(ax_ifr_corr.get_xticklabels(), rotation=30, ha="right", fontsize=10); plt.tight_layout(); plt.show()
    else: print("No overall IFR Correlation data to plot or all values are NaN.")


# --- NEW: System-Specific Plotting Function ---
def plot_system_specific_metrics_line_charts(results_dict, system_name, system_vessel_labels):
    """
    Create line plots of average metrics across multiple epochs for a specific coronary artery system.
    Args:
        results_dict (dict): The main results dictionary from epoch evaluations.
        system_name (str): Name of the system (e.g., "LCA", "RCA") for plot titles.
        system_vessel_labels (list): List of vessel label strings belonging to this system.
    """
    if not results_dict:
        print(f"No results to plot for {system_name} system metrics.")
        return
    if not system_vessel_labels:
        print(f"No vessel labels provided for {system_name} system. Cannot plot metrics.")
        return

    def get_epoch_num_from_key(key_str):
        match = re.search(r'epoch(\d+)', key_str)
        return int(match.group(1)) if match else -1

    sorted_epoch_keys = sorted(results_dict.keys(), key=get_epoch_num_from_key)
    if not sorted_epoch_keys or (len(sorted_epoch_keys) > 0 and get_epoch_num_from_key(sorted_epoch_keys[0]) == -1 and any(get_epoch_num_from_key(k) != -1 for k in sorted_epoch_keys)):
        print(f"Warning: Could not sort all epochs numerically for {system_name} plots, using alphanumeric sort.")
        sorted_epoch_keys = sorted(results_dict.keys())

    plot_data_system = {
        "stenosis_mae": [], "ifr_mae": [],
        "stenosis_corr": [], "ifr_corr": [],
        "calcif_accuracy": [],
        "x_labels": [key.replace(".csv", "") for key in sorted_epoch_keys]
    }

    for ep_key in sorted_epoch_keys:
        metrics = results_dict[ep_key]["metrics"]
        
        # Filter metrics for the current system's vessels
        system_metrics_sten_mae = [metrics.get("stenosis", {}).get("mae", {}).get(v_lbl) for v_lbl in system_vessel_labels if pd.notna(metrics.get("stenosis", {}).get("mae", {}).get(v_lbl))]
        system_metrics_sten_corr = [metrics.get("stenosis", {}).get("corr", {}).get(v_lbl) for v_lbl in system_vessel_labels if pd.notna(metrics.get("stenosis", {}).get("corr", {}).get(v_lbl))]
        
        system_metrics_ifr_mae = [metrics.get("ifr", {}).get("mae", {}).get(v_lbl) for v_lbl in system_vessel_labels if pd.notna(metrics.get("ifr", {}).get("mae", {}).get(v_lbl))]
        system_metrics_ifr_corr = [metrics.get("ifr", {}).get("corr", {}).get(v_lbl) for v_lbl in system_vessel_labels if pd.notna(metrics.get("ifr", {}).get("corr", {}).get(v_lbl))]

        system_metrics_calcif_acc = [metrics.get("calcif", {}).get("accuracy", {}).get(v_lbl) for v_lbl in system_vessel_labels if pd.notna(metrics.get("calcif", {}).get("accuracy", {}).get(v_lbl))]

        plot_data_system["stenosis_mae"].append(np.nanmean(system_metrics_sten_mae) if system_metrics_sten_mae else np.nan)
        plot_data_system["stenosis_corr"].append(np.nanmean(system_metrics_sten_corr) if system_metrics_sten_corr else np.nan)
        plot_data_system["ifr_mae"].append(np.nanmean(system_metrics_ifr_mae) if system_metrics_ifr_mae else np.nan)
        plot_data_system["ifr_corr"].append(np.nanmean(system_metrics_ifr_corr) if system_metrics_ifr_corr else np.nan)
        plot_data_system["calcif_accuracy"].append(np.nanmean(system_metrics_calcif_acc) if system_metrics_calcif_acc else np.nan)

    if not any(any(pd.notna(val) for val in data_list) for key, data_list in plot_data_system.items() if key != "x_labels"):
        print(f"No valid metric data available to plot for {system_name} system across epochs.")
        return

    # Create plots for the system
    num_rows_plot = 3 # Stenosis MAE/Corr, IFR MAE/Corr, Calcif Accuracy
    fig_system, axes_system = plt.subplots(nrows=num_rows_plot, ncols=1, figsize=(14, num_rows_plot * 5), sharex=True)
    fig_system.suptitle(f"{system_name} System: Model Performance Metrics Across Epochs", fontsize=18, y=0.99)

    # Stenosis MAE for system
    axes_system[0].plot(plot_data_system["x_labels"], plot_data_system["stenosis_mae"], marker='o', label=f"{system_name} Stenosis MAE (Avg)", color="dodgerblue")
    axes_system[0].set_title(f"Average {system_name} Stenosis MAE", fontsize=14)
    axes_system[0].set_ylabel("Mean Absolute Error", fontsize=12)
    axes_system[0].legend(fontsize=10); axes_system[0].grid(True, linestyle=':', alpha=0.6)

    # Stenosis Correlation for system
    axes_system[1].plot(plot_data_system["x_labels"], plot_data_system["stenosis_corr"], marker='X', label=f"{system_name} Stenosis Corr (Avg)", color="forestgreen")
    axes_system[1].set_title(f"Average {system_name} Stenosis Correlation", fontsize=14)
    axes_system[1].set_ylabel("Avg Pearson Correlation", fontsize=12)
    valid_corr = [v for v in plot_data_system["stenosis_corr"] if pd.notna(v)]
    axes_system[1].set_ylim(min(0, np.min(valid_corr) - 0.1) if valid_corr else 0, max(1, np.max(valid_corr) + 0.1) if valid_corr else 1)
    axes_system[1].legend(fontsize=10); axes_system[1].grid(True, linestyle=':', alpha=0.6)
    
    # Calcification Accuracy for system
    axes_system[2].plot(plot_data_system["x_labels"], plot_data_system["calcif_accuracy"], marker='s', label=f"{system_name} Calcif. Accuracy (Avg)", color="darkorange")
    axes_system[2].set_title(f"Average {system_name} Calcification Accuracy", fontsize=14)
    axes_system[2].set_ylabel("Accuracy", fontsize=12)
    valid_calcif_acc = [v for v in plot_data_system["calcif_accuracy"] if pd.notna(v)]
    axes_system[2].set_ylim(min(0, np.min(valid_calcif_acc) - 0.1) if valid_calcif_acc else 0, max(1, np.max(valid_calcif_acc) + 0.1) if valid_calcif_acc else 1)
    axes_system[2].set_xlabel("Epoch", fontsize=12)
    axes_system[2].legend(fontsize=10); axes_system[2].grid(True, linestyle=':', alpha=0.6)

    plt.setp(axes_system[-1].get_xticklabels(), rotation=30, ha="right", fontsize=10)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()

    # Separate plots for IFR metrics if data exists
    if any(pd.notna(v) for v in plot_data_system["ifr_mae"]):
        fig_ifr_mae_sys, ax_ifr_mae_sys = plt.subplots(figsize=(14, 6))
        ax_ifr_mae_sys.plot(plot_data_system["x_labels"], plot_data_system["ifr_mae"], marker='o', label=f"{system_name} IFR MAE (Avg)", color="crimson")
        ax_ifr_mae_sys.set_title(f"Average {system_name} IFR MAE", fontsize=14); ax_ifr_mae_sys.set_ylabel("Mean Absolute Error", fontsize=12); ax_ifr_mae_sys.set_xlabel("Epoch", fontsize=12); ax_ifr_mae_sys.legend(fontsize=10); ax_ifr_mae_sys.grid(True, linestyle=':', alpha=0.6); plt.setp(ax_ifr_mae_sys.get_xticklabels(), rotation=30, ha="right", fontsize=10); plt.tight_layout(); plt.show()
    else:
        print(f"No IFR MAE data to plot for {system_name} system or all values are NaN.")

    if any(pd.notna(v) for v in plot_data_system["ifr_corr"]):
        fig_ifr_corr_sys, ax_ifr_corr_sys = plt.subplots(figsize=(14, 6))
        ax_ifr_corr_sys.plot(plot_data_system["x_labels"], plot_data_system["ifr_corr"], marker='X', label=f"{system_name} IFR Corr (Avg)", color="teal")
        ax_ifr_corr_sys.set_title(f"Average {system_name} IFR Correlation", fontsize=14); ax_ifr_corr_sys.set_ylabel("Avg Pearson Correlation", fontsize=12)
        valid_ifr_c = [v for v in plot_data_system["ifr_corr"] if pd.notna(v)]; ax_ifr_corr_sys.set_ylim(min(0, np.min(valid_ifr_c) - 0.1) if valid_ifr_c else 0, max(1, np.max(valid_ifr_c) + 0.1) if valid_ifr_c else 1); ax_ifr_corr_sys.set_xlabel("Epoch", fontsize=12); ax_ifr_corr_sys.legend(fontsize=10); ax_ifr_corr_sys.grid(True, linestyle=':', alpha=0.6); plt.setp(ax_ifr_corr_sys.get_xticklabels(), rotation=30, ha="right", fontsize=10); plt.tight_layout(); plt.show()
    else:
        print(f"No IFR Correlation data to plot for {system_name} system or all values are NaN.")


def display_stenosis_predictions_for_file(agg_df_single_epoch, file_name_to_display, vessel_labels_list=None):
    """Displays stenosis predictions for a specific file."""
    if vessel_labels_list is None: 
        vessel_labels_list = [
            "leftmain_stenosis", "lad_stenosis", "mid_lad_stenosis", "dist_lad_stenosis",
            "diagonal_stenosis", "D2_stenosis", "D3_stenosis", "lcx_stenosis",
            "dist_lcx_stenosis", "lvp_stenosis", "marg_d_stenosis", "om1_stenosis",
            "om2_stenosis", "om3_stenosis", "prox_rca_stenosis", "mid_rca_stenosis",
            "dist_rca_stenosis", "RVG1_stenosis", "RVG2_stenosis", "pda_stenosis",
            "posterolateral_stenosis", "bx_stenosis", "lima_or_svg_stenosis",
        ]
    if "FileName" not in agg_df_single_epoch.columns: print(f"Error: 'FileName' column not found."); return
    file_data_row_series = agg_df_single_epoch[agg_df_single_epoch["FileName"] == file_name_to_display]
    if file_data_row_series.empty: print(f"File '{file_name_to_display}' not found."); return
    row_data = file_data_row_series.iloc[0]; print(f"\n--- Stenosis Predictions for File: {file_name_to_display} ---")
    output_lines_display, max_key_len_display = [], 0
    for gt_vessel_key_display in vessel_labels_list:
        pred_vessel_key_display = f"predicted_{gt_vessel_key_display}"
        pred_val, gt_val = row_data.get(pred_vessel_key_display), row_data.get(gt_vessel_key_display)
        pred_val_str, gt_val_str = f"{pred_val:.1f}" if pd.notna(pred_val) else "N/A", f"{gt_val:.1f}" if pd.notna(gt_val) else "N/A"
        output_lines_display.append((pred_vessel_key_display, pred_val_str)); max_key_len_display = max(max_key_len_display, len(pred_vessel_key_display))
        output_lines_display.append((gt_vessel_key_display, gt_val_str)); max_key_len_display = max(max_key_len_display, len(gt_vessel_key_display))
    for key_str_disp, val_str_disp in output_lines_display: print(f"{key_str_disp:<{max_key_len_display + 4}} {val_str_disp}")

def plot_stenosis_predictions_for_file(agg_df_single_epoch, file_name_to_plot, vessel_labels_list=None, epoch_name_for_title=""):
    """Plots stenosis predictions for a specific file."""
    if vessel_labels_list is None:
        vessel_labels_list = [
            "leftmain_stenosis", "lad_stenosis", "mid_lad_stenosis", "dist_lad_stenosis",
            "diagonal_stenosis", "D2_stenosis", "D3_stenosis", "lcx_stenosis",
            "dist_lcx_stenosis", "lvp_stenosis", "marg_d_stenosis", "om1_stenosis",
            "om2_stenosis", "om3_stenosis", "prox_rca_stenosis", "mid_rca_stenosis",
            "dist_rca_stenosis", "RVG1_stenosis", "RVG2_stenosis", "pda_stenosis",
            "posterolateral_stenosis", "bx_stenosis", "lima_or_svg_stenosis",
        ]
    if "FileName" not in agg_df_single_epoch.columns: print(f"Error: 'FileName' column not found for plotting."); return
    file_data_row_series_plot = agg_df_single_epoch[agg_df_single_epoch["FileName"] == file_name_to_plot]
    if file_data_row_series_plot.empty: print(f"File '{file_name_to_plot}' not found for plotting."); return
    row_data_plot = file_data_row_series_plot.iloc[0]; plot_labels_names, gt_values_for_plot, pred_values_for_plot = [], [], []
    for gt_vessel_key_plot in vessel_labels_list:
        pred_vessel_key_plot = f"predicted_{gt_vessel_key_plot}"; gt_value_plot, pred_value_plot = row_data_plot.get(gt_vessel_key_plot), row_data_plot.get(pred_vessel_key_plot)
        if pd.notna(gt_value_plot) or pd.notna(pred_value_plot):
            plot_labels_names.append(gt_vessel_key_plot.replace("_stenosis", "").replace("_", " ").title()); gt_values_for_plot.append(gt_value_plot if pd.notna(gt_value_plot) else 0); pred_values_for_plot.append(pred_value_plot if pd.notna(pred_value_plot) else 0)
    if not plot_labels_names: print(f"No valid stenosis data to plot for file '{file_name_to_plot}'."); return
    x_indices, bar_width = np.arange(len(plot_labels_names)), 0.35; fig_single_file, ax_single_file = plt.subplots(figsize=(max(16, len(plot_labels_names) * 0.8), 7))
    bars_gt = ax_single_file.bar(x_indices - bar_width/2, gt_values_for_plot, bar_width, label='Ground Truth (%)', color='cornflowerblue'); bars_pred = ax_single_file.bar(x_indices + bar_width/2, pred_values_for_plot, bar_width, label='Predicted (%)', color='lightcoral')
    ax_single_file.set_ylabel('Stenosis Value (%)', fontsize=12); title_str_plot = f'Stenosis Comparison: {file_name_to_plot}'
    if epoch_name_for_title: title_str_plot += f' (Epoch: {epoch_name_for_title})'
    ax_single_file.set_title(title_str_plot, fontsize=15); ax_single_file.set_xticks(x_indices); ax_single_file.set_xticklabels(plot_labels_names, rotation=40, ha="right", fontsize=10); ax_single_file.legend(fontsize=11)
    all_vals_for_ylim = [v for v in gt_values_for_plot + pred_values_for_plot if pd.notna(v)] + [0, 100]; ax_single_file.set_ylim(0, np.nanmax(all_vals_for_ylim) * 1.15 if all_vals_for_ylim else 110); ax_single_file.grid(axis='y', linestyle='--', alpha=0.7)
    def add_labels_to_bars(bars):
        for bar in bars:
            height = bar.get_height()
            if pd.notna(height) and height != 0: ax_single_file.annotate(f'{height:.0f}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    add_labels_to_bars(bars_gt); add_labels_to_bars(bars_pred); fig_single_file.tight_layout(); plt.show()

def display_row_at_index(df, index, columns_to_display=None):
    """
    Display a specific row from a DataFrame at the given index.
    
    Args:
        df (pd.DataFrame): The DataFrame to display from
        index (int): The index of the row to display
        columns_to_display (list, optional): List of column names to display. If None, displays all columns.
    """
    if index < 0 or index >= len(df):
        print(f"Error: Index {index} is out of range. DataFrame has {len(df)} rows.")
        return
    
    # Get the row at the specified index
    row = df.iloc[index]
    
    # If specific columns are requested, filter them
    if columns_to_display:
        # Verify all requested columns exist
        missing_cols = [col for col in columns_to_display if col not in df.columns]
        if missing_cols:
            print(f"Warning: Columns not found in DataFrame: {missing_cols}")
            columns_to_display = [col for col in columns_to_display if col in df.columns]
        row = row[columns_to_display]
    
    # Convert to DataFrame for better display
    row_df = pd.DataFrame([row])
    
    # Display with all columns
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(f"\nRow at index {index}:")
    print(row_df.to_string(index=False))
    
    return row_df

def display_row_values(df, iloc, columns_to_display=None):
    """
    Display values for a specific row or rows in the DataFrame.
    
    Args:
        df (pd.DataFrame): The DataFrame to display from
        iloc (int or list): The index or indices of the row(s) to display
        columns_to_display (list, optional): List of column names to display. If None, displays all columns.
    """
    try:
        # Handle both single index and multiple indices
        if isinstance(iloc, (int, np.integer)):
            rows = df.iloc[[iloc]]
        else:
            rows = df.iloc[iloc]
            
        # If specific columns are requested, filter them
        if columns_to_display:
            # Verify all requested columns exist
            missing_cols = [col for col in columns_to_display if col not in df.columns]
            if missing_cols:
                print(f"Warning: Columns not found in DataFrame: {missing_cols}")
                columns_to_display = [col for col in columns_to_display if col in df.columns]
            rows = rows[columns_to_display]
        
        # Display with all columns
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(f"\nRow(s) at index {iloc}:")
        print(rows.to_string())
        
        return rows
        
    except Exception as e:
        print(f"Error displaying row values: {e}")
        return None

# --- Data Loading and Preparation Function ---
def load_and_prepare_main_dataset(
    dataset_csv_path, 
    predictions_base_dir,
    reference_epoch_basename_for_mapping,
    key_col_for_gt_map="val_text_index"
    ):
    """Loads and preprocesses the main GT dataset, creating the val_text_map."""
    print(f"\n--- Loading and Preparing Main Dataset ---")
    if not os.path.exists(dataset_csv_path): print(f"FATAL: GT dataset CSV not found: {dataset_csv_path}"); return None, None
    ref_epoch_full_path = os.path.join(predictions_base_dir, reference_epoch_basename_for_mapping)
    if not os.path.exists(ref_epoch_full_path): print(f"FATAL: Reference epoch CSV ('{reference_epoch_basename_for_mapping}') not found in '{predictions_base_dir}'"); return None, None

    try: df_dataset_main = pd.read_csv(dataset_csv_path, sep='α', encoding='utf-8', engine='python'); print(f"Loaded main dataset ({len(df_dataset_main)} rows).")
    except Exception as e: print(f"FATAL: Could not load main dataset. Error: {e}"); return None, None
    try: df_ref_epoch = pd.read_csv(ref_epoch_full_path); print(f"Loaded reference epoch ('{reference_epoch_basename_for_mapping}').")
    except Exception as e: print(f"FATAL: Could not load reference epoch CSV. Error: {e}"); return None, None

    file_to_idx_map = {}
    if 'FileName' in df_ref_epoch.columns and 'ground_truth_idx' in df_ref_epoch.columns:
        file_to_idx_map = dict(zip(df_ref_epoch['FileName'].astype(str), df_ref_epoch['ground_truth_idx']))
    elif 'VideoFileNames' in df_ref_epoch.columns and 'ground_truth_idx' in df_ref_epoch.columns:
        print("Warning: 'FileName' not in ref epoch. Using 'FirstVideo' from 'VideoFileNames'.")
        df_ref_epoch['FirstVideo'] = df_ref_epoch['VideoFileNames'].astype(str).str.split(';').str[0]
        file_to_idx_map = dict(zip(df_ref_epoch['FirstVideo'].astype(str), df_ref_epoch['ground_truth_idx']))
    else: print(f"FATAL: Cannot create mapping for '{key_col_for_gt_map}'. Required cols missing in ref epoch."); return None, None
    if not file_to_idx_map: print(f"Warning: filename_to_idx map for '{key_col_for_gt_map}' is empty.")

    if 'FileName' not in df_dataset_main.columns: print(f"FATAL: 'FileName' col missing in main dataset."); return None, None
    df_dataset_main[key_col_for_gt_map] = df_dataset_main['FileName'].astype(str).map(file_to_idx_map)
    print(f"Rows with '{key_col_for_gt_map}' after initial map: {df_dataset_main[key_col_for_gt_map].notna().sum()}")

    if 'StudyInstanceUID' in df_dataset_main.columns:
        print(f"Propagating '{key_col_for_gt_map}' by 'StudyInstanceUID'...")
        def propagate_idx_group(group):
            valid_idx = group[key_col_for_gt_map].dropna().unique()
            if len(valid_idx) > 0: group[key_col_for_gt_map] = valid_idx[0]
            return group
        df_dataset_main = df_dataset_main.groupby('StudyInstanceUID', group_keys=False).progress_apply(propagate_idx_group)
        print(f"Rows with '{key_col_for_gt_map}' after propagation: {df_dataset_main[key_col_for_gt_map].notna().sum()}")

    df_processed = df_dataset_main.dropna(subset=[key_col_for_gt_map])
    if df_processed.empty: print(f"FATAL: Main dataset empty after '{key_col_for_gt_map}' processing."); return None, None
    df_processed[key_col_for_gt_map] = df_processed[key_col_for_gt_map].astype(int)
    print(f"Final processed GT dataset size: {len(df_processed)} rows.")

    val_text_map = build_val_text_index_map(df_processed, key_col=key_col_for_gt_map)
    if not val_text_map: print(f"Warning: Global val_text_map ('{key_col_for_gt_map}') is empty.")
    return df_processed, val_text_map

# --- Epoch Evaluation Function ---
def run_evaluation_on_epochs(
    val_text_map_global, 
    epoch_csv_paths_to_evaluate,
    topk_predictions=5,
    custom_vessel_labels=None
    ):
    """Evaluates a list of epoch CSVs using a pre-built val_text_map."""
    print(f"\n--- Running Evaluation on {len(epoch_csv_paths_to_evaluate)} Epochs ---")
    vessel_labels_for_eval = custom_vessel_labels if custom_vessel_labels else [
        "leftmain_stenosis", "lad_stenosis", "mid_lad_stenosis", "dist_lad_stenosis", "diagonal_stenosis", 
        "D2_stenosis", "D3_stenosis", "lcx_stenosis", "dist_lcx_stenosis", "lvp_stenosis", "marg_d_stenosis", 
        "om1_stenosis", "om2_stenosis", "om3_stenosis", "prox_rca_stenosis", "mid_rca_stenosis", "dist_rca_stenosis", 
        "RVG1_stenosis", "RVG2_stenosis", "pda_stenosis", "posterolateral_stenosis", "bx_stenosis", "lima_or_svg_stenosis"
    ]
    if not val_text_map_global: print("Warning: val_text_map_global is empty. Results may be empty.")

    evaluated_results = {}
    for epoch_path in epoch_csv_paths_to_evaluate:
        epoch_name = os.path.basename(epoch_path)
        print(f"--- Evaluating {epoch_name} ---")
        try:
            preds_df = pd.read_csv(epoch_path)
            if 'ground_truth_idx' not in preds_df.columns: print(f"Warning: 'ground_truth_idx' missing in {epoch_name}. Skip."); continue
        except Exception as e: print(f"Warning: Cannot read {epoch_name} ({e}). Skip."); continue

        agg_df = aggregate_predictions_for_epoch(val_text_map_global, preds_df, topk_predictions, vessel_labels_for_eval)
        metrics = compute_metrics(agg_df, vessel_labels_for_eval) if not agg_df.empty else {
            "stenosis": {"mae": {lbl: np.nan for lbl in vessel_labels_for_eval}, "corr": {lbl: np.nan for lbl in vessel_labels_for_eval}},
            "ifr": {"mae": {lbl: np.nan for lbl in vessel_labels_for_eval}, "corr": {lbl: np.nan for lbl in vessel_labels_for_eval}},
            "calcif": {"accuracy": {lbl: np.nan for lbl in vessel_labels_for_eval}},
            "coronary_dominance": {"accuracy": np.nan}
        }
        if agg_df.empty: print(f"Warning: Aggregated DataFrame empty for {epoch_name}.")
        evaluated_results[epoch_name] = {"agg_df": agg_df, "metrics": metrics, "vessel_labels_used": vessel_labels_for_eval}
    return evaluated_results

# --- Main Orchestration Function ---
def run_full_evaluation_orchestrator(
    dataset_csv_path, 
    predictions_base_dir,
    reference_epoch_basename_for_mapping,
    specific_epoch_to_save_agg_df_basename=None,
    output_dir_for_saved_data=None,
    topk_predictions=5,
    custom_vessel_labels=None,
    key_col_for_gt_map_name="val_text_index"
    ):
    """Main orchestrator: loads data, evaluates all epochs, plots, saves specified agg_df."""
    print("--- Starting Full Evaluation Orchestration ---")
    if output_dir_for_saved_data is None: output_dir_for_saved_data = predictions_base_dir
    elif not os.path.isdir(output_dir_for_saved_data):
        try: os.makedirs(output_dir_for_saved_data); print(f"Created output dir: {output_dir_for_saved_data}")
        except Exception as e: print(f"FATAL: Cannot create output dir '{output_dir_for_saved_data}'. Error: {e}"); return None, None, None
    
    df_processed, val_text_map = load_and_prepare_main_dataset(dataset_csv_path, predictions_base_dir, reference_epoch_basename_for_mapping, key_col_for_gt_map_name)
    if df_processed is None or val_text_map is None: print("FATAL: Failed to load/prepare main dataset."); return None, None, None

    all_epoch_paths = [os.path.join(predictions_base_dir, f) for f in os.listdir(predictions_base_dir) if f.startswith("val_epoch") and f.endswith(".csv")]
    def get_epoch_num(filepath):
        try: return int(re.search(r'epoch(\d+)', os.path.basename(filepath)).group(1))
        except: return -1 
    all_epoch_paths.sort(key=get_epoch_num)
    if not all_epoch_paths: print(f"FATAL: No 'val_epoch*.csv' files in '{predictions_base_dir}'."); return df_processed, val_text_map, None
    print(f"Found {len(all_epoch_paths)} epochs for initial evaluation: {[os.path.basename(p) for p in all_epoch_paths]}")

    all_results = run_evaluation_on_epochs(val_text_map, all_epoch_paths, topk_predictions, custom_vessel_labels)
    if not all_results: print("Warning: No results from initial epoch evaluations.")

    if all_results: plot_epoch_metrics_line_charts(all_results)
    else: print("No results from initial evaluation to plot.")

    if specific_epoch_to_save_agg_df_basename and all_results and specific_epoch_to_save_agg_df_basename in all_results:
        agg_df_save = all_results[specific_epoch_to_save_agg_df_basename]["agg_df"]
        if not agg_df_save.empty:
            parent_folder = os.path.basename(os.path.normpath(predictions_base_dir))
            filename_save = f"{parent_folder}_{specific_epoch_to_save_agg_df_basename.replace('.csv','')}_agg.csv"
            save_path = os.path.join(output_dir_for_saved_data, filename_save)
            try: agg_df_save.to_csv(save_path, index=False); print(f"Saved agg_df for '{specific_epoch_to_save_agg_df_basename}' to: {save_path}")
            except Exception as e: print(f"Error saving agg_df: {e}")
        else: print(f"Agg_df for '{specific_epoch_to_save_agg_df_basename}' is empty. Not saving.")
    
    print("\n--- Full Evaluation Orchestration Finished ---")
    return df_processed, val_text_map, all_results

# --- Example of calling the main function (if script is run directly) ---
if __name__ == '__main__':
    print("Running plot_metrics.py as a standalone script...")
    MAIN_DATASET_CSV = "data/reports/reports_with_alpha_separator_with_Calcifc_Stenosis_IFR_20250507_STUDYLEVEL.csv"
    PREDICTIONS_CSV_DIR = "outputs/DeepCORO_clip/dev_deep_coro_clip_single_video/k0ohoagn_20250518-173722/outputs/DeepCORO_clip/dev_deep_coro_clip_single_video/k0ohoagn_20250518-173942"
    REFERENCE_EPOCH_BASENAME = "val_epoch15.csv" 
    EPOCH_TO_SAVE_AGG_DF = "val_epoch15.csv" 
    OUTPUT_SAVE_DIR = "analysis_outputs_script" 
    CUSTOM_LABELS = None 

    if OUTPUT_SAVE_DIR and not os.path.exists(OUTPUT_SAVE_DIR):
        os.makedirs(OUTPUT_SAVE_DIR); print(f"Created example output directory: {OUTPUT_SAVE_DIR}")

    processed_gt_data, val_text_map_data, initial_results = run_full_evaluation_orchestrator(
        dataset_csv_path=MAIN_DATASET_CSV,
        predictions_base_dir=PREDICTIONS_CSV_DIR,
        reference_epoch_basename_for_mapping=REFERENCE_EPOCH_BASENAME,
        specific_epoch_to_save_agg_df_basename=EPOCH_TO_SAVE_AGG_DF,
        output_dir_for_saved_data=OUTPUT_SAVE_DIR,
        custom_vessel_labels=CUSTOM_LABELS
    )

    if initial_results and processed_gt_data is not None and val_text_map_data is not None:
        print("\nExample: Standalone script finished. Data is ready for further operations.")
        # Example: Plot LCA specific metrics if results exist
        lca_specific_labels = [
            "leftmain_stenosis", "lad_stenosis", "mid_lad_stenosis", "dist_lad_stenosis",
            "diagonal_stenosis", "D2_stenosis", "D3_stenosis", "lcx_stenosis",
            "dist_lcx_stenosis", "om1_stenosis", "om2_stenosis", "om3_stenosis", "bx_stenosis"
        ]
        # Check if any of these labels are actually in the processed_gt_data columns
        lca_labels_in_data = [lbl for lbl in lca_specific_labels if lbl in processed_gt_data.columns]
        if lca_labels_in_data:
            print("\nPlotting LCA-specific metrics...")
            plot_system_specific_metrics_line_charts(initial_results, "LCA", lca_labels_in_data)
        else:
            print("\nNo defined LCA labels found in the processed dataset columns. Skipping LCA plot.")

