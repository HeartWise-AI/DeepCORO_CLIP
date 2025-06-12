import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re  # For sorting epoch numbers

# Import constants and helper functions from vessel_constants (to avoid circular imports)
from .vessel_constants import (
    LEFT_CORONARY_DOMINANCE_VESSELS,
    RIGHT_CORONARY_DOMINANCE_VESSELS,
    RCA_VESSELS,
    NON_RCA_VESSELS,
    mode
)

# Add new imports for multi-epoch analysis
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
import os
import glob

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
    if (not sorted_epoch_keys or 
        (len(sorted_epoch_keys) > 0 and get_epoch_num_from_key(sorted_epoch_keys[0]) == -1 
         and any(get_epoch_num_from_key(k) != -1 for k in sorted_epoch_keys))):
        print("Warning: Could not sort all epochs numerically, using alphanumeric sort for overall metrics.")
        sorted_epoch_keys = sorted(results_dict.keys())

    plot_data = {
        "stenosis_mae": [], "ifr_mae": [],
        "stenosis_corr": [], "ifr_corr": [],
        "calcif_accuracy": [], "string_accuracy": [],
        "retrieval_top1": [], "retrieval_top5": [],
        "x_labels": [key.replace(".csv", "") for key in sorted_epoch_keys]
    }

    for ep_key in sorted_epoch_keys:
        result_data = results_dict[ep_key]
        
        # Handle both old and new result formats
        if "metrics" in result_data:
            # Old format
            metrics = result_data["metrics"]
            agg_df = result_data.get("agg_df", pd.DataFrame())
        else:
            # New format (tuple of aggregated_df, metrics)
            if isinstance(result_data, tuple) and len(result_data) == 2:
                agg_df, metrics = result_data
            else:
                print(f"Warning: Unexpected result format for {ep_key}")
                continue
        
        # Process stenosis metrics
        for metric_category_key, expected_sub_metrics in [
            ("stenosis", ["mae", "corr"]), 
            ("ifr", ["mae", "corr"]), 
            ("calcif", ["accuracy"])
        ]:
            for sub_metric_key_name in expected_sub_metrics:
                plot_data_key = f"{metric_category_key}_{sub_metric_key_name}"
                metric_values_dict = metrics.get(metric_category_key, {}).get(sub_metric_key_name, {})
                if isinstance(metric_values_dict, dict):
                    metric_values = metric_values_dict.values()
                else:
                    metric_values = []
                
                valid_metric_values = [v for v in metric_values if pd.notna(v)]
                average_value = np.nanmean(valid_metric_values) if valid_metric_values else np.nan
                
                if plot_data_key not in plot_data:
                    plot_data[plot_data_key] = []
                plot_data[plot_data_key].append(average_value)

        # Add string accuracy metrics (calcification)
        string_accuracy_dict = metrics.get("string_accuracy", {})
        if string_accuracy_dict:
            string_accuracy_values = [v for v in string_accuracy_dict.values() if pd.notna(v)]
            plot_data["string_accuracy"].append(np.nanmean(string_accuracy_values) if string_accuracy_values else np.nan)
        else:
            plot_data["string_accuracy"].append(np.nan)
        
        # Add retrieval metrics if available in aggregated DataFrame
        if not agg_df.empty and 'top1_match' in agg_df.columns:
            plot_data["retrieval_top1"].append(agg_df['top1_match'].mean())
            plot_data["retrieval_top5"].append(agg_df['top5_match'].mean() if 'top5_match' in agg_df.columns else np.nan)
        else:
            plot_data["retrieval_top1"].append(np.nan)
            plot_data["retrieval_top5"].append(np.nan)
    
    num_expected_points = len(plot_data["x_labels"])
    for key, value_list in plot_data.items():
        if key != "x_labels":
            if len(value_list) < num_expected_points:
                plot_data[key].extend([np.nan] * (num_expected_points - len(value_list)))
            elif len(value_list) > num_expected_points:
                plot_data[key] = value_list[:num_expected_points]

    if not any(
        any(pd.notna(val) for val in data_list) 
        for _key, data_list in plot_data.items() 
        if _key != "x_labels"
    ):
        print("No valid overall metric data available to plot across epochs.")
        return

    # Create main plot with 4 subplots to include retrieval metrics
    fig_main, axes_main = plt.subplots(nrows=4, ncols=1, figsize=(14, 20), sharex=True)
    fig_main.suptitle("Overall Model Performance Metrics (All Vessels) Across Epochs", fontsize=18, y=0.98)
    
    # Stenosis MAE
    axes_main[0].plot(plot_data["x_labels"], plot_data["stenosis_mae"], marker='o', 
                      label="Stenosis MAE (Avg)", color="dodgerblue", linestyle='-')
    axes_main[0].set_title("Average Stenosis MAE", fontsize=14)
    axes_main[0].set_ylabel("Mean Absolute Error", fontsize=12)
    axes_main[0].legend(fontsize=10)
    axes_main[0].grid(True, linestyle=':', alpha=0.6)

    # Stenosis Correlation
    axes_main[1].plot(plot_data["x_labels"], plot_data["stenosis_corr"], marker='X', 
                      label="Stenosis Correlation (Avg)", color="forestgreen", linestyle='-')
    axes_main[1].set_title("Average Stenosis Correlation", fontsize=14)
    axes_main[1].set_ylabel("Avg Pearson Correlation", fontsize=12)
    valid_sten_corr = [v for v in plot_data["stenosis_corr"] if pd.notna(v)]
    if valid_sten_corr:
        axes_main[1].set_ylim(min(0, np.min(valid_sten_corr)) - 0.1, max(1, np.max(valid_sten_corr)) + 0.1)
    axes_main[1].legend(fontsize=10)
    axes_main[1].grid(True, linestyle=':', alpha=0.6)

    # Accuracies
    axes_main[2].plot(plot_data["x_labels"], plot_data["calcif_accuracy"], marker='s', 
                      label="Calcification Accuracy (Avg)", color="darkorange", linestyle='-')
    axes_main[2].plot(plot_data["x_labels"], plot_data["string_accuracy"], marker='D', 
                      label="String Accuracy (Calcif-related)", color="purple", linestyle='--')
    axes_main[2].set_title("Average Accuracy Scores", fontsize=14)
    axes_main[2].set_ylabel("Accuracy", fontsize=12)
    valid_acc_data = plot_data["calcif_accuracy"] + plot_data["string_accuracy"]
    valid_acc = [v for v in valid_acc_data if pd.notna(v)]
    if valid_acc:
        axes_main[2].set_ylim(min(0, np.min(valid_acc)) - 0.1, max(1, np.max(valid_acc)) + 0.1)
    axes_main[2].legend(fontsize=10)
    axes_main[2].grid(True, linestyle=':', alpha=0.6)
    
    # Retrieval Performance
    axes_main[3].plot(plot_data["x_labels"], plot_data["retrieval_top1"], marker='o', 
                      label="Top-1 Retrieval Accuracy", color="red", linestyle='-')
    axes_main[3].plot(plot_data["x_labels"], plot_data["retrieval_top5"], marker='s', 
                      label="Top-5 Retrieval Accuracy", color="orange", linestyle='-')
    axes_main[3].set_title("Retrieval Performance", fontsize=14)
    axes_main[3].set_ylabel("Accuracy", fontsize=12)
    axes_main[3].set_xlabel("Epoch", fontsize=12)
    valid_retrieval = [v for v in plot_data["retrieval_top1"] + plot_data["retrieval_top5"] if pd.notna(v)]
    if valid_retrieval:
        axes_main[3].set_ylim(0, max(1, np.max(valid_retrieval)) + 0.1)
    axes_main[3].legend(fontsize=10)
    axes_main[3].grid(True, linestyle=':', alpha=0.6)
    
    plt.setp(axes_main[3].get_xticklabels(), rotation=30, ha="right", fontsize=10)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Plot IFR metrics separately if available
    if "ifr_mae" in plot_data and any(pd.notna(v) for v in plot_data["ifr_mae"]):
        fig_ifr_mae, ax_ifr_mae = plt.subplots(figsize=(14, 6))
        ax_ifr_mae.plot(plot_data["x_labels"], plot_data["ifr_mae"], marker='o', 
                        label="IFR MAE (Avg)", color="crimson", linestyle='-')
        ax_ifr_mae.set_title("Average IFR MAE across Epochs", fontsize=14)
        ax_ifr_mae.set_ylabel("Mean Absolute Error", fontsize=12)
        ax_ifr_mae.set_xlabel("Epoch", fontsize=12)
        ax_ifr_mae.legend(fontsize=10)
        ax_ifr_mae.grid(True, linestyle=':', alpha=0.6)
        plt.setp(ax_ifr_mae.get_xticklabels(), rotation=30, ha="right", fontsize=10)
        plt.tight_layout()
        plt.show()
    else:
        print("No overall IFR MAE data to plot or all values are NaN.")

    if "ifr_corr" in plot_data and any(pd.notna(v) for v in plot_data["ifr_corr"]):
        fig_ifr_corr, ax_ifr_corr = plt.subplots(figsize=(14, 6))
        ax_ifr_corr.plot(plot_data["x_labels"], plot_data["ifr_corr"], marker='X', 
                         label="IFR Correlation (Avg)", color="teal", linestyle='-')
        ax_ifr_corr.set_title("Average IFR Correlation across Epochs", fontsize=14)
        ax_ifr_corr.set_ylabel("Avg Pearson Correlation", fontsize=12)
        valid_ifr_corr = [v for v in plot_data["ifr_corr"] if pd.notna(v)]
        if valid_ifr_corr:
            ax_ifr_corr.set_ylim(min(0, np.min(valid_ifr_corr)) - 0.1, max(1, np.max(valid_ifr_corr)) + 0.1)
        ax_ifr_corr.set_xlabel("Epoch", fontsize=12)
        ax_ifr_corr.legend(fontsize=10)
        ax_ifr_corr.grid(True, linestyle=':', alpha=0.6)
        plt.setp(ax_ifr_corr.get_xticklabels(), rotation=30, ha="right", fontsize=10)
        plt.tight_layout()
        plt.show()
    else:
        print("No overall IFR Correlation data to plot or all values are NaN.")

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
    if (not sorted_epoch_keys or 
        (len(sorted_epoch_keys) > 0 and get_epoch_num_from_key(sorted_epoch_keys[0]) == -1 
         and any(get_epoch_num_from_key(k) != -1 for k in sorted_epoch_keys))):
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
        
        # Filter metrics for system vessels
        system_metrics_sten_mae = [
            metrics.get("stenosis", {}).get("mae", {}).get(v_lbl) 
            for v_lbl in system_vessel_labels 
            if pd.notna(metrics.get("stenosis", {}).get("mae", {}).get(v_lbl))
        ]
        system_metrics_sten_corr = [
            metrics.get("stenosis", {}).get("corr", {}).get(v_lbl) 
            for v_lbl in system_vessel_labels 
            if pd.notna(metrics.get("stenosis", {}).get("corr", {}).get(v_lbl))
        ]
        system_metrics_ifr_mae = [
            metrics.get("ifr", {}).get("mae", {}).get(v_lbl) 
            for v_lbl in system_vessel_labels 
            if pd.notna(metrics.get("ifr", {}).get("mae", {}).get(v_lbl))
        ]
        system_metrics_ifr_corr = [
            metrics.get("ifr", {}).get("corr", {}).get(v_lbl) 
            for v_lbl in system_vessel_labels 
            if pd.notna(metrics.get("ifr", {}).get("corr", {}).get(v_lbl))
        ]
        system_metrics_calcif_acc = [
            metrics.get("calcif", {}).get("accuracy", {}).get(v_lbl) 
            for v_lbl in system_vessel_labels 
            if pd.notna(metrics.get("calcif", {}).get("accuracy", {}).get(v_lbl))
        ]

        # Calculate averages for the system
        plot_data_system["stenosis_mae"].append(
            np.nanmean(system_metrics_sten_mae) if system_metrics_sten_mae else np.nan
        )
        plot_data_system["stenosis_corr"].append(
            np.nanmean(system_metrics_sten_corr) if system_metrics_sten_corr else np.nan
        )
        plot_data_system["ifr_mae"].append(
            np.nanmean(system_metrics_ifr_mae) if system_metrics_ifr_mae else np.nan
        )
        plot_data_system["ifr_corr"].append(
            np.nanmean(system_metrics_ifr_corr) if system_metrics_ifr_corr else np.nan
        )
        plot_data_system["calcif_accuracy"].append(
            np.nanmean(system_metrics_calcif_acc) if system_metrics_calcif_acc else np.nan
        )

    # Check if there's any valid data to plot
    if not any(
        any(pd.notna(val) for val in data_list) 
        for _key, data_list in plot_data_system.items() 
        if _key != "x_labels"
    ):
        print(f"No valid {system_name} metric data available to plot across epochs.")
        return

    # Create the main plot with stenosis metrics
    fig_sys, axes_sys = plt.subplots(nrows=2, ncols=1, figsize=(14, 12), sharex=True)
    fig_sys.suptitle(f"{system_name} System Performance Metrics Across Epochs", fontsize=18, y=0.98)
    
    axes_sys[0].plot(plot_data_system["x_labels"], plot_data_system["stenosis_mae"], marker='o', 
                     label=f"{system_name} Stenosis MAE", color="dodgerblue", linestyle='-')
    axes_sys[0].set_title(f"{system_name} System Stenosis MAE", fontsize=14)
    axes_sys[0].set_ylabel("Mean Absolute Error", fontsize=12)
    axes_sys[0].legend(fontsize=10)
    axes_sys[0].grid(True, linestyle=':', alpha=0.6)

    axes_sys[1].plot(plot_data_system["x_labels"], plot_data_system["stenosis_corr"], marker='X', 
                     label=f"{system_name} Stenosis Correlation", color="forestgreen", linestyle='-')
    axes_sys[1].plot(plot_data_system["x_labels"], plot_data_system["calcif_accuracy"], marker='s', 
                     label=f"{system_name} Calcification Accuracy", color="darkorange", linestyle='--')
    axes_sys[1].set_title(f"{system_name} System Stenosis Correlation & Calcification Accuracy", fontsize=14)
    axes_sys[1].set_ylabel("Correlation / Accuracy", fontsize=12)
    axes_sys[1].set_xlabel("Epoch", fontsize=12)
    
    # Set appropriate y-limits
    valid_corr_acc = [v for v in plot_data_system["stenosis_corr"] + plot_data_system["calcif_accuracy"] if pd.notna(v)]
    if valid_corr_acc:
        axes_sys[1].set_ylim(min(0, np.min(valid_corr_acc)) - 0.1, max(1, np.max(valid_corr_acc)) + 0.1)
    
    axes_sys[1].legend(fontsize=10)
    axes_sys[1].grid(True, linestyle=':', alpha=0.6)
    
    plt.setp(axes_sys[1].get_xticklabels(), rotation=30, ha="right", fontsize=10)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Plot IFR metrics if available
    if any(pd.notna(v) for v in plot_data_system["ifr_mae"]):
        fig_ifr_mae_sys, ax_ifr_mae_sys = plt.subplots(figsize=(14, 6))
        ax_ifr_mae_sys.plot(plot_data_system["x_labels"], plot_data_system["ifr_mae"], marker='o', 
                            label=f"{system_name} IFR MAE", color="crimson")
        ax_ifr_mae_sys.set_title(f"{system_name} System IFR MAE", fontsize=14)
        ax_ifr_mae_sys.set_ylabel("Mean Absolute Error", fontsize=12)
        ax_ifr_mae_sys.set_xlabel("Epoch", fontsize=12)
        ax_ifr_mae_sys.legend(fontsize=10)
        ax_ifr_mae_sys.grid(True, linestyle=':', alpha=0.6)
        plt.setp(ax_ifr_mae_sys.get_xticklabels(), rotation=30, ha="right", fontsize=10)
        plt.tight_layout()
        plt.show()
    else:
        print(f"No IFR MAE data to plot for {system_name} system or all values are NaN.")

    if any(pd.notna(v) for v in plot_data_system["ifr_corr"]):
        fig_ifr_corr_sys, ax_ifr_corr_sys = plt.subplots(figsize=(14, 6))
        ax_ifr_corr_sys.plot(plot_data_system["x_labels"], plot_data_system["ifr_corr"], marker='X', 
                             label=f"{system_name} IFR Corr (Avg)", color="teal")
        ax_ifr_corr_sys.set_title(f"Average {system_name} IFR Correlation", fontsize=14)
        ax_ifr_corr_sys.set_ylabel("Avg Pearson Correlation", fontsize=12)
        valid_ifr_c = [v for v in plot_data_system["ifr_corr"] if pd.notna(v)]
        if valid_ifr_c:
            ax_ifr_corr_sys.set_ylim(min(0, np.min(valid_ifr_c) - 0.1), max(1, np.max(valid_ifr_c) + 0.1))
        ax_ifr_corr_sys.set_xlabel("Epoch", fontsize=12)
        ax_ifr_corr_sys.legend(fontsize=10)
        ax_ifr_corr_sys.grid(True, linestyle=':', alpha=0.6)
        plt.setp(ax_ifr_corr_sys.get_xticklabels(), rotation=30, ha="right", fontsize=10)
        plt.tight_layout()
        plt.show()
    else:
        print(f"No IFR Correlation data to plot for {system_name} system or all values are NaN.")

def plot_coronary_artery_specific_metrics_line_charts(results_dict, coronary_artery):
    """
    Create line plots of average metrics across multiple epochs for left or right coronary area.
    
    Args:
        results_dict (dict): The main results dictionary from epoch evaluations.
        coronary_artery (str): Either "left" or "right" to specify which coronary area to plot.
    """
    if not results_dict:
        print(f"No results to plot for {coronary_artery} coronary area metrics.")
        return
    
    # Define vessel labels based on coronary area
    if coronary_artery.lower() == "left":
        area_vessel_labels = LEFT_CORONARY_DOMINANCE_VESSELS
        area_name = "Left Coronary Area"
    elif coronary_artery.lower() == "right":
        area_vessel_labels = RIGHT_CORONARY_DOMINANCE_VESSELS
        area_name = "Right Coronary Area"
    else:
        print(f"Invalid coronary area '{coronary_artery}'. Must be 'left' or 'right'.")
        return
    
    if not area_vessel_labels:
        print(f"No vessel labels defined for {area_name}. Cannot plot metrics.")
        return

    def get_epoch_num_from_key(key_str):
        match = re.search(r'epoch(\d+)', key_str)
        return int(match.group(1)) if match else -1

    sorted_epoch_keys = sorted(results_dict.keys(), key=get_epoch_num_from_key)
    if (not sorted_epoch_keys or 
        (len(sorted_epoch_keys) > 0 and get_epoch_num_from_key(sorted_epoch_keys[0]) == -1 
         and any(get_epoch_num_from_key(k) != -1 for k in sorted_epoch_keys))):
        print(f"Warning: Could not sort all epochs numerically for {area_name} plots, using alphanumeric sort.")
        sorted_epoch_keys = sorted(results_dict.keys())

    plot_data_area = {
        "stenosis_mae": [], "ifr_mae": [],
        "stenosis_corr": [], "ifr_corr": [],
        "calcif_accuracy": [],
        "x_labels": [key.replace(".csv", "") for key in sorted_epoch_keys]
    }

    for ep_key in sorted_epoch_keys:
        metrics = results_dict[ep_key]["metrics"]
        
        # Filter metrics for area vessels
        area_metrics_sten_mae = [
            metrics.get("stenosis", {}).get("mae", {}).get(v_lbl) 
            for v_lbl in area_vessel_labels 
            if pd.notna(metrics.get("stenosis", {}).get("mae", {}).get(v_lbl))
        ]
        area_metrics_sten_corr = [
            metrics.get("stenosis", {}).get("corr", {}).get(v_lbl) 
            for v_lbl in area_vessel_labels 
            if pd.notna(metrics.get("stenosis", {}).get("corr", {}).get(v_lbl))
        ]
        area_metrics_ifr_mae = [
            metrics.get("ifr", {}).get("mae", {}).get(v_lbl) 
            for v_lbl in area_vessel_labels 
            if pd.notna(metrics.get("ifr", {}).get("mae", {}).get(v_lbl))
        ]
        area_metrics_ifr_corr = [
            metrics.get("ifr", {}).get("corr", {}).get(v_lbl) 
            for v_lbl in area_vessel_labels 
            if pd.notna(metrics.get("ifr", {}).get("corr", {}).get(v_lbl))
        ]
        area_metrics_calcif_acc = [
            metrics.get("calcif", {}).get("accuracy", {}).get(v_lbl) 
            for v_lbl in area_vessel_labels 
            if pd.notna(metrics.get("calcif", {}).get("accuracy", {}).get(v_lbl))
        ]

        # Calculate averages for the area
        plot_data_area["stenosis_mae"].append(
            np.nanmean(area_metrics_sten_mae) if area_metrics_sten_mae else np.nan
        )
        plot_data_area["stenosis_corr"].append(
            np.nanmean(area_metrics_sten_corr) if area_metrics_sten_corr else np.nan
        )
        plot_data_area["ifr_mae"].append(
            np.nanmean(area_metrics_ifr_mae) if area_metrics_ifr_mae else np.nan
        )
        plot_data_area["ifr_corr"].append(
            np.nanmean(area_metrics_ifr_corr) if area_metrics_ifr_corr else np.nan
        )
        plot_data_area["calcif_accuracy"].append(
            np.nanmean(area_metrics_calcif_acc) if area_metrics_calcif_acc else np.nan
        )

    # Check if there's any valid data to plot
    if not any(
        any(pd.notna(val) for val in data_list) 
        for _key, data_list in plot_data_area.items() 
        if _key != "x_labels"
    ):
        print(f"No valid {area_name} metric data available to plot across epochs.")
        return

    # Create the main plot with stenosis metrics
    fig_area, axes_area = plt.subplots(nrows=2, ncols=1, figsize=(14, 12), sharex=True)
    fig_area.suptitle(f"{area_name} Performance Metrics Across Epochs", fontsize=18, y=0.98)
    
    axes_area[0].plot(plot_data_area["x_labels"], plot_data_area["stenosis_mae"], marker='o', 
                      label=f"{area_name} Stenosis MAE", color="dodgerblue", linestyle='-')
    axes_area[0].set_title(f"{area_name} Stenosis MAE", fontsize=14)
    axes_area[0].set_ylabel("Mean Absolute Error", fontsize=12)
    axes_area[0].legend(fontsize=10)
    axes_area[0].grid(True, linestyle=':', alpha=0.6)

    axes_area[1].plot(plot_data_area["x_labels"], plot_data_area["stenosis_corr"], marker='X', 
                      label=f"{area_name} Stenosis Correlation", color="forestgreen", linestyle='-')
    axes_area[1].plot(plot_data_area["x_labels"], plot_data_area["calcif_accuracy"], marker='s', 
                      label=f"{area_name} Calcification Accuracy", color="darkorange", linestyle='--')
    axes_area[1].set_title(f"{area_name} Stenosis Correlation & Calcification Accuracy", fontsize=14)
    axes_area[1].set_ylabel("Correlation / Accuracy", fontsize=12)
    axes_area[1].set_xlabel("Epoch", fontsize=12)
    
    # Set appropriate y-limits
    valid_corr_acc = [v for v in plot_data_area["stenosis_corr"] + plot_data_area["calcif_accuracy"] if pd.notna(v)]
    if valid_corr_acc:
        axes_area[1].set_ylim(min(0, np.min(valid_corr_acc)) - 0.1, max(1, np.max(valid_corr_acc)) + 0.1)
    
    axes_area[1].legend(fontsize=10)
    axes_area[1].grid(True, linestyle=':', alpha=0.6)
    
    plt.setp(axes_area[1].get_xticklabels(), rotation=30, ha="right", fontsize=10)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Plot IFR metrics if available
    if any(pd.notna(v) for v in plot_data_area["ifr_mae"]):
        fig_ifr_mae_area, ax_ifr_mae_area = plt.subplots(figsize=(14, 6))
        ax_ifr_mae_area.plot(plot_data_area["x_labels"], plot_data_area["ifr_mae"], marker='o', 
                           label=f"{area_name} IFR MAE", color="crimson")
        ax_ifr_mae_area.set_title(f"{area_name} IFR MAE", fontsize=14)
        ax_ifr_mae_area.set_ylabel("Mean Absolute Error", fontsize=12)
        ax_ifr_mae_area.set_xlabel("Epoch", fontsize=12)
        ax_ifr_mae_area.legend(fontsize=10)
        ax_ifr_mae_area.grid(True, linestyle=':', alpha=0.6)
        plt.setp(ax_ifr_mae_area.get_xticklabels(), rotation=30, ha="right", fontsize=10)
        plt.tight_layout()
        plt.show()
    else:
        print(f"No IFR MAE data to plot for {area_name} or all values are NaN.")

    if any(pd.notna(v) for v in plot_data_area["ifr_corr"]):
        fig_ifr_corr_area, ax_ifr_corr_area = plt.subplots(figsize=(14, 6))
        ax_ifr_corr_area.plot(plot_data_area["x_labels"], plot_data_area["ifr_corr"], marker='X', 
                            label=f"{area_name} IFR Correlation", color="teal")
        ax_ifr_corr_area.set_title(f"{area_name} IFR Correlation", fontsize=14)
        ax_ifr_corr_area.set_ylabel("Avg Pearson Correlation", fontsize=12)
        valid_ifr_c = [v for v in plot_data_area["ifr_corr"] if pd.notna(v)]
        if valid_ifr_c:
            ax_ifr_corr_area.set_ylim(min(0, np.min(valid_ifr_c) - 0.1), max(1, np.max(valid_ifr_c) + 0.1))
        ax_ifr_corr_area.set_xlabel("Epoch", fontsize=12)
        ax_ifr_corr_area.legend(fontsize=10)
        ax_ifr_corr_area.grid(True, linestyle=':', alpha=0.6)
        plt.setp(ax_ifr_corr_area.get_xticklabels(), rotation=30, ha="right", fontsize=10)
        plt.tight_layout()
        plt.show()
    else:
        print(f"No IFR Correlation data to plot for {area_name} or all values are NaN.")

def plot_ground_truth_comparison_results(aggregated_df, title_suffix="", study_level_df=None):
    """
    Plot results from ground truth comparison analysis showing retrieval performance
    and structure-based metrics.
    
    Args:
        aggregated_df: DataFrame from aggregate_predictions_for_epoch with new format
        title_suffix: Additional text for plot titles
        study_level_df: Optional study-level DataFrame for additional analysis
    """
    if aggregated_df.empty:
        print("Empty aggregated DataFrame provided for plotting.")
        return
    
    print(f"Plotting ground truth comparison results for {len(aggregated_df)} predictions...")
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Ground Truth Comparison Analysis {title_suffix}', fontsize=16, fontweight='bold')
    
    # 1. Retrieval Performance by Structure
    if 'main_structure_name' in aggregated_df.columns and 'top1_match' in aggregated_df.columns:
        structure_performance = aggregated_df.groupby('main_structure_name').agg({
            'top1_match': 'mean',
            'top3_match': 'mean',
            'top5_match': 'mean'
        }).reset_index()
        
        x_pos = np.arange(len(structure_performance))
        width = 0.25
        
        axes[0,0].bar(x_pos - width, structure_performance['top1_match'], width, 
                      label='Top-1', color='red', alpha=0.7)
        axes[0,0].bar(x_pos, structure_performance['top3_match'], width, 
                      label='Top-3', color='orange', alpha=0.7)
        axes[0,0].bar(x_pos + width, structure_performance['top5_match'], width, 
                      label='Top-5', color='yellow', alpha=0.7)
        
        axes[0,0].set_title('Retrieval Accuracy by Coronary Structure', fontweight='bold')
        axes[0,0].set_xlabel('Coronary Structure')
        axes[0,0].set_ylabel('Accuracy')
        axes[0,0].set_xticks(x_pos)
        axes[0,0].set_xticklabels(structure_performance['main_structure_name'])
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].set_ylim(0, 1)
        
        # Add value labels on bars
        for i, (top1, top3, top5) in enumerate(zip(structure_performance['top1_match'], 
                                                  structure_performance['top3_match'],
                                                  structure_performance['top5_match'])):
            axes[0,0].text(i - width, top1 + 0.02, f'{top1:.3f}', ha='center', va='bottom', fontsize=9)
            axes[0,0].text(i, top3 + 0.02, f'{top3:.3f}', ha='center', va='bottom', fontsize=9)
            axes[0,0].text(i + width, top5 + 0.02, f'{top5:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Distribution of Rank of Ground Truth
    if 'rank_of_gt' in aggregated_df.columns:
        valid_ranks = aggregated_df[aggregated_df['rank_of_gt'] > 0]['rank_of_gt']
        if len(valid_ranks) > 0:
            axes[0,1].hist(valid_ranks, bins=range(1, 7), alpha=0.7, color='skyblue', edgecolor='black')
            axes[0,1].set_title('Distribution of Ground Truth Rank', fontweight='bold')
            axes[0,1].set_xlabel('Rank of Ground Truth')
            axes[0,1].set_ylabel('Frequency')
            axes[0,1].grid(True, alpha=0.3)
            
            # Add percentage labels
            total_found = len(valid_ranks)
            for i in range(1, 6):
                count = sum(valid_ranks == i)
                percentage = count / total_found * 100
                if count > 0:
                    axes[0,1].text(i, count + 0.5, f'{percentage:.1f}%', ha='center', va='bottom')
        else:
            axes[0,1].text(0.5, 0.5, 'No valid ranks found', ha='center', va='center', 
                          transform=axes[0,1].transAxes, fontsize=12)
            axes[0,1].set_title('Distribution of Ground Truth Rank', fontweight='bold')
    
    # 3. Structure-level Stenosis Performance (if available)
    structure_cols = [col for col in aggregated_df.columns 
                      if col.startswith(('gt_left_coronary_mean', 'gt_right_coronary_mean', 
                                       'pred_left_coronary_mean', 'pred_right_coronary_mean'))]
    
    if structure_cols:
        # Calculate MAE by structure
        mae_by_structure = {}
        for structure in ['Left Coronary', 'Right Coronary']:
            structure_data = aggregated_df[aggregated_df['main_structure_name'] == structure]
            if len(structure_data) > 0:
                gt_col = f'gt_{structure.lower().replace(" ", "_")}_mean_stenosis'
                pred_col = f'pred_{structure.lower().replace(" ", "_")}_mean_stenosis'
                
                if gt_col in structure_data.columns and pred_col in structure_data.columns:
                    valid_mask = structure_data[gt_col].notna() & structure_data[pred_col].notna()
                    if valid_mask.sum() > 0:
                        gt_vals = structure_data[gt_col][valid_mask]
                        pred_vals = structure_data[pred_col][valid_mask]
                        mae = mean_absolute_error(gt_vals, pred_vals)
                        mae_by_structure[structure] = mae
        
        if mae_by_structure:
            structures = list(mae_by_structure.keys())
            maes = list(mae_by_structure.values())
            
            axes[1,0].bar(structures, maes, color=['lightcoral', 'lightblue'], alpha=0.7)
            axes[1,0].set_title('Mean Absolute Error by Coronary Structure', fontweight='bold')
            axes[1,0].set_xlabel('Coronary Structure')
            axes[1,0].set_ylabel('MAE (%)')
            axes[1,0].grid(True, alpha=0.3)
            
            # Add value labels
            for i, mae in enumerate(maes):
                axes[1,0].text(i, mae + max(maes) * 0.02, f'{mae:.2f}', ha='center', va='bottom')
    else:
        axes[1,0].text(0.5, 0.5, 'No structure-level stenosis data', ha='center', va='center', 
                      transform=axes[1,0].transAxes, fontsize=12)
        axes[1,0].set_title('Mean Absolute Error by Coronary Structure', fontweight='bold')
    
    # 4. Dominance Distribution
    if 'dominance_name' in aggregated_df.columns:
        dominance_counts = aggregated_df['dominance_name'].value_counts()
        
        # Create pie chart
        colors = ['lightgreen', 'lightpink', 'lightyellow'][:len(dominance_counts)]
        wedges, texts, autotexts = axes[1,1].pie(dominance_counts.values, 
                                                labels=dominance_counts.index, 
                                                autopct='%1.1f%%',
                                                colors=colors,
                                                startangle=90)
        axes[1,1].set_title('Distribution by Coronary Dominance', fontweight='bold')
        
        # Add count information
        for i, (dominance, count) in enumerate(dominance_counts.items()):
            percentage = count / len(aggregated_df) * 100
            print(f"   {dominance}: {count} cases ({percentage:.1f}%)")
    else:
        axes[1,1].text(0.5, 0.5, 'No dominance data', ha='center', va='center', 
                      transform=axes[1,1].transAxes, fontsize=12)
        axes[1,1].set_title('Distribution by Coronary Dominance', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\n📊 GROUND TRUTH COMPARISON SUMMARY:")
    if 'top1_match' in aggregated_df.columns:
        print(f"   🎯 Overall Top-1 Accuracy: {aggregated_df['top1_match'].mean():.3f}")
        print(f"   🎯 Overall Top-3 Accuracy: {aggregated_df['top3_match'].mean():.3f}")
        print(f"   🎯 Overall Top-5 Accuracy: {aggregated_df['top5_match'].mean():.3f}")
    
    if 'rank_of_gt' in aggregated_df.columns:
        found_gt = aggregated_df['rank_of_gt'] > 0
        print(f"   📍 Ground truth found in top-5: {found_gt.sum()}/{len(aggregated_df)} ({found_gt.mean():.3f})")
        if found_gt.sum() > 0:
            avg_rank = aggregated_df[found_gt]['rank_of_gt'].mean()
            print(f"   📍 Average rank when found: {avg_rank:.2f}")
    
    if 'main_structure_name' in aggregated_df.columns:
        structure_counts = aggregated_df['main_structure_name'].value_counts()
        print(f"   🫀 Structure distribution:")
        for structure, count in structure_counts.items():
            print(f"      {structure}: {count} cases ({count/len(aggregated_df)*100:.1f}%)")

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
    if "FileName" not in agg_df_single_epoch.columns:
        print(f"Error: 'FileName' column not found.")
        return
    
    file_data_row_series = agg_df_single_epoch[agg_df_single_epoch["FileName"] == file_name_to_display]
    if file_data_row_series.empty:
        print(f"File '{file_name_to_display}' not found.")
        return
    
    row_data = file_data_row_series.iloc[0]
    print(f"\n--- Stenosis Predictions for File: {file_name_to_display} ---")
    output_lines_display = []
    max_key_len_display = 0
    
    for gt_vessel_key_display in vessel_labels_list:
        pred_vessel_key_display = f"predicted_{gt_vessel_key_display}"
        pred_val = row_data.get(pred_vessel_key_display)
        gt_val = row_data.get(gt_vessel_key_display)
        
        pred_val_str = f"{pred_val:.1f}" if pd.notna(pred_val) else "N/A"
        gt_val_str = f"{gt_val:.1f}" if pd.notna(gt_val) else "N/A"
        
        output_lines_display.append((pred_vessel_key_display, pred_val_str))
        max_key_len_display = max(max_key_len_display, len(pred_vessel_key_display))
        
        output_lines_display.append((gt_vessel_key_display, gt_val_str))
        max_key_len_display = max(max_key_len_display, len(gt_vessel_key_display))
    
    for key_str_disp, val_str_disp in output_lines_display:
        print(f"{key_str_disp:<{max_key_len_display + 4}} {val_str_disp}")

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
    if "FileName" not in agg_df_single_epoch.columns:
        print(f"Error: 'FileName' column not found for plotting.")
        return
    
    file_data_row_series_plot = agg_df_single_epoch[agg_df_single_epoch["FileName"] == file_name_to_plot]
    if file_data_row_series_plot.empty:
        print(f"File '{file_name_to_plot}' not found for plotting.")
        return
    
    row_data_plot = file_data_row_series_plot.iloc[0]
    plot_labels_names = []
    gt_values_for_plot = []
    pred_values_for_plot = []
    
    for gt_vessel_key_plot in vessel_labels_list:
        pred_vessel_key_plot = f"predicted_{gt_vessel_key_plot}"
        gt_value_plot = row_data_plot.get(gt_vessel_key_plot)
        pred_value_plot = row_data_plot.get(pred_vessel_key_plot)
        
        if pd.notna(gt_value_plot) or pd.notna(pred_value_plot):
            plot_labels_names.append(
                gt_vessel_key_plot.replace("_stenosis", "").replace("_", " ").title()
            )
            gt_values_for_plot.append(gt_value_plot if pd.notna(gt_value_plot) else 0)
            pred_values_for_plot.append(pred_value_plot if pd.notna(pred_value_plot) else 0)
    
    if not plot_labels_names:
        print(f"No valid stenosis data to plot for file '{file_name_to_plot}'.")
        return
    
    x_indices = np.arange(len(plot_labels_names))
    bar_width = 0.35
    fig_single_file, ax_single_file = plt.subplots(figsize=(max(16, len(plot_labels_names) * 0.8), 7))
    
    bars_gt = ax_single_file.bar(x_indices - bar_width / 2, gt_values_for_plot, bar_width,
                                 label='Ground Truth (%)', color='cornflowerblue')
    bars_pred = ax_single_file.bar(x_indices + bar_width / 2, pred_values_for_plot, bar_width,
                                   label='Predicted (%)', color='lightcoral')
    
    ax_single_file.set_ylabel('Stenosis Value (%)', fontsize=12)
    title_str_plot = f'Stenosis Comparison: {file_name_to_plot}'
    if epoch_name_for_title:
        title_str_plot += f' (Epoch: {epoch_name_for_title})'
    ax_single_file.set_title(title_str_plot, fontsize=15)
    ax_single_file.set_xticks(x_indices)
    ax_single_file.set_xticklabels(plot_labels_names, rotation=40, ha="right", fontsize=10)
    ax_single_file.legend(fontsize=11)
    
    all_vals_for_ylim = [v for v in gt_values_for_plot + pred_values_for_plot if pd.notna(v)] + [0, 100]
    ax_single_file.set_ylim(0, np.nanmax(all_vals_for_ylim) * 1.15 if all_vals_for_ylim else 110)
    ax_single_file.grid(axis='y', linestyle='--', alpha=0.7)
    
    def add_labels_to_bars(bars):
        for bar in bars:
            height = bar.get_height()
            if pd.notna(height) and height != 0:
                ax_single_file.annotate(f'{height:.0f}',
                                        xy=(bar.get_x() + bar.get_width() / 2, height),
                                        xytext=(0, 3),
                                        textcoords="offset points",
                                        ha='center', va='bottom', fontsize=8)
    
    add_labels_to_bars(bars_gt)
    add_labels_to_bars(bars_pred)
    
    fig_single_file.tight_layout()
    plt.show()

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
    
    row = df.iloc[index]
    if columns_to_display:
        missing_cols = [col for col in columns_to_display if col not in df.columns]
        if missing_cols:
            print(f"Warning: Columns not found in DataFrame: {missing_cols}")
            columns_to_display = [col for col in columns_to_display if col in df.columns]
        row = row[columns_to_display]
    
    row_df = pd.DataFrame([row])
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
        if isinstance(iloc, (int, np.integer)):
            rows = df.iloc[[iloc]]
        else:
            rows = df.iloc[iloc]
        
        if columns_to_display:
            missing_cols = [col for col in columns_to_display if col not in df.columns]
            if missing_cols:
                print(f"Warning: Columns not found in DataFrame: {missing_cols}")
                columns_to_display = [col for col in columns_to_display if col in df.columns]
            rows = rows[columns_to_display]
        
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(f"\nRow(s) at index {iloc}:")
        print(rows.to_string())
        return rows
        
    except Exception as e:
        print(f"Error displaying row values: {e}")
        return None

# --- Multi-Epoch Analysis Functions ---

def calculate_epoch_metrics_from_predictions(epoch_file_path, df_dataset, val_text_map=None):
    """
    Calculate metrics for a single epoch from prediction file and ground truth dataset.
    
    Args:
        epoch_file_path (str): Path to the epoch prediction CSV file
        df_dataset (pd.DataFrame): Ground truth dataset with vessel stenosis values
        val_text_map (dict): Optional text mapping (if needed for text-based predictions)
    
    Returns:
        dict: Metrics for this epoch including MAE, correlation, accuracy
    """
    try:
        # Load epoch predictions
        df_predictions = pd.read_csv(epoch_file_path)
        print(f"   Loading {os.path.basename(epoch_file_path)}: {len(df_predictions)} predictions")
        
        # Extract epoch number from filename
        epoch_match = re.search(r'epoch(\d+)', epoch_file_path)
        epoch_num = int(epoch_match.group(1)) if epoch_match else -1
        
        # Merge predictions with ground truth dataset
        merged_df = pd.merge(df_predictions, df_dataset, on='FileName', how='inner', suffixes=('_pred', '_gt'))
        
        if len(merged_df) == 0:
            print(f"   ⚠️ No merged data for {os.path.basename(epoch_file_path)}")
            return None
        
        print(f"   Merged data: {len(merged_df)} samples")
        
        # For retrieval-based predictions, we need to aggregate predictions
        # Using the top prediction (predicted_idx_1) for now
        merged_df['predicted_stenosis'] = merged_df.apply(
            lambda row: get_aggregated_stenosis_from_predictions(row, df_dataset), axis=1
        )
        
        # Define vessel groups
        left_coronary_vessels = [
            'left_main_stenosis', 'prox_lad_stenosis', 'mid_lad_stenosis', 
            'dist_lad_stenosis', 'D1_stenosis', 'D2_stenosis'
        ]
        
        right_coronary_vessels = [
            'prox_rca_stenosis', 'mid_rca_stenosis', 'dist_rca_stenosis'
        ]
        
        # Initialize metrics structure
        epoch_metrics = {
            'epoch': epoch_num,
            'filename': os.path.basename(epoch_file_path),
            'total_samples': len(merged_df),
            'metrics': {
                'stenosis': {'mae': {}, 'corr': {}},
                'ifr': {'mae': {}, 'corr': {}},
                'calcif': {'accuracy': {}},
                'string_accuracy': {}
            }
        }
        
        # Calculate stenosis metrics for each vessel
        all_vessels = left_coronary_vessels + right_coronary_vessels
        
        for vessel in all_vessels:
            if vessel in merged_df.columns:
                # Get valid predictions and ground truth
                valid_mask = merged_df[vessel].notna() & merged_df['predicted_stenosis'].notna()
                
                if valid_mask.sum() > 5:  # Need at least 5 samples
                    gt_values = merged_df[vessel][valid_mask]
                    pred_values = merged_df['predicted_stenosis'][valid_mask] 
                    
                    # Calculate MAE
                    mae = mean_absolute_error(gt_values, pred_values)
                    epoch_metrics['metrics']['stenosis']['mae'][vessel] = mae
                    
                    # Calculate correlation
                    if len(set(gt_values)) > 1 and len(set(pred_values)) > 1:
                        try:
                            corr, p_value = pearsonr(gt_values, pred_values)
                            if not np.isnan(corr):
                                epoch_metrics['metrics']['stenosis']['corr'][vessel] = corr
                        except:
                            pass  # Skip if correlation calculation fails
        
        # Calculate IFR metrics if available
        ifr_columns = [col for col in merged_df.columns if 'ifr' in col.lower() and 'predicted' not in col.lower()]
        for ifr_col in ifr_columns:
            pred_ifr_col = f'predicted_{ifr_col}'
            if pred_ifr_col in merged_df.columns or 'predicted_stenosis' in merged_df.columns:
                valid_mask = merged_df[ifr_col].notna() & merged_df['predicted_stenosis'].notna()
                
                if valid_mask.sum() > 5:
                    gt_ifr = merged_df[ifr_col][valid_mask]
                    pred_ifr = merged_df['predicted_stenosis'][valid_mask]  # Use stenosis as proxy
                    
                    mae_ifr = mean_absolute_error(gt_ifr, pred_ifr)
                    epoch_metrics['metrics']['ifr']['mae'][ifr_col] = mae_ifr
                    
                    if len(set(gt_ifr)) > 1 and len(set(pred_ifr)) > 1:
                        try:
                            corr_ifr, _ = pearsonr(gt_ifr, pred_ifr)
                            if not np.isnan(corr_ifr):
                                epoch_metrics['metrics']['ifr']['corr'][ifr_col] = corr_ifr
                        except:
                            pass
        
        # Calculate calcification accuracy if available
        calcif_columns = [col for col in merged_df.columns if 'calcif' in col.lower() and 'predicted' not in col.lower()]
        for calcif_col in calcif_columns:
            # For calcification, we'll calculate accuracy based on thresholding
            if calcif_col in merged_df.columns:
                valid_mask = merged_df[calcif_col].notna()
                if valid_mask.sum() > 5:
                    gt_calcif = merged_df[calcif_col][valid_mask] > 0  # Binary classification
                    pred_calcif = merged_df['predicted_stenosis'][valid_mask] > 50  # Threshold for prediction
                    
                    accuracy = (gt_calcif == pred_calcif).mean()
                    epoch_metrics['metrics']['calcif']['accuracy'][calcif_col] = accuracy
        
        return epoch_metrics
        
    except Exception as e:
        print(f"   ❌ Error processing {os.path.basename(epoch_file_path)}: {e}")
        return None

def get_aggregated_stenosis_from_predictions(row, df_dataset):
    """
    Extract stenosis value from prediction indices using ground truth dataset.
    Uses top 3 predictions and averages their stenosis values.
    """
    try:
        predicted_indices = []
        similarities = []
        
        # Get top 3 predictions
        for i in range(1, 4):  # predicted_idx_1, predicted_idx_2, predicted_idx_3
            idx_col = f'predicted_idx_{i}'
            sim_col = f'sim_{i}'
            
            if idx_col in row and sim_col in row:
                if pd.notna(row[idx_col]) and pd.notna(row[sim_col]):
                    predicted_indices.append(int(row[idx_col]))
                    similarities.append(float(row[sim_col]))
        
        if not predicted_indices:
            return np.nan
        
        # Get stenosis values for predicted indices
        stenosis_values = []
        weights = []
        
        for idx, sim in zip(predicted_indices, similarities):
            if idx < len(df_dataset):
                # Get a representative stenosis value (could be from multiple vessels)
                vessels_to_check = ['prox_lad_stenosis', 'mid_lad_stenosis', 'left_main_stenosis', 'prox_rca_stenosis']
                vessel_stenosis = []
                
                for vessel in vessels_to_check:
                    if vessel in df_dataset.columns:
                        stenosis_val = df_dataset.iloc[idx][vessel]
                        if pd.notna(stenosis_val):
                            vessel_stenosis.append(stenosis_val)
                
                if vessel_stenosis:
                    avg_stenosis = np.mean(vessel_stenosis)
                    stenosis_values.append(avg_stenosis)
                    weights.append(sim)
        
        if stenosis_values:
            # Weighted average based on similarity scores
            if len(weights) == len(stenosis_values):
                return np.average(stenosis_values, weights=weights)
            else:
                return np.mean(stenosis_values)
        
        return np.nan
        
    except Exception as e:
        return np.nan

def analyze_multi_epoch_performance(prediction_dir, df_dataset, epochs_to_analyze=None):
    """
    Analyze performance across multiple epochs.
    
    Args:
        prediction_dir (str): Directory containing epoch prediction files
        df_dataset (pd.DataFrame): Ground truth dataset
        epochs_to_analyze (list): List of epoch numbers to analyze (None for all)
    
    Returns:
        dict: Multi-epoch analysis results
    """
    print(f"\n🔍 MULTI-EPOCH PERFORMANCE ANALYSIS")
    print(f"📁 Prediction directory: {prediction_dir}")
    
    # Find all epoch files
    epoch_files = glob.glob(os.path.join(prediction_dir, "val_epoch*.csv"))
    epoch_files.sort()
    
    if not epoch_files:
        print(f"❌ No epoch files found in {prediction_dir}")
        return None
    
    print(f"📊 Found {len(epoch_files)} epoch files")
    
    # Filter epochs if specified
    if epochs_to_analyze:
        filtered_files = []
        for epoch_file in epoch_files:
            epoch_match = re.search(r'epoch(\d+)', epoch_file)
            if epoch_match and int(epoch_match.group(1)) in epochs_to_analyze:
                filtered_files.append(epoch_file)
        epoch_files = filtered_files
    
    print(f"📈 Analyzing {len(epoch_files)} epochs...")
    
    # Process each epoch
    epoch_results = {}
    
    for epoch_file in epoch_files:
        epoch_metrics = calculate_epoch_metrics_from_predictions(epoch_file, df_dataset)
        if epoch_metrics:
            epoch_key = f"val_epoch{epoch_metrics['epoch']}.csv"
            epoch_results[epoch_key] = epoch_metrics
    
    if not epoch_results:
        print("❌ No valid epoch results calculated")
        return None
    
    print(f"✅ Successfully processed {len(epoch_results)} epochs")
    return epoch_results

def plot_multi_epoch_metrics_comprehensive(epoch_results, title_suffix=""):
    """
    Create comprehensive line plots for multi-epoch performance analysis.
    
    Args:
        epoch_results (dict): Results from analyze_multi_epoch_performance()
        title_suffix (str): Additional text for plot titles
    """
    if not epoch_results:
        print("❌ No epoch results to plot")
        return
    
    # Sort epochs numerically
    def get_epoch_num_from_key(key_str):
        match = re.search(r'epoch(\d+)', key_str)
        return int(match.group(1)) if match else -1

    sorted_epoch_keys = sorted(epoch_results.keys(), key=get_epoch_num_from_key)
    epochs = [get_epoch_num_from_key(key) for key in sorted_epoch_keys]
    
    # Extract metrics data
    overall_stenosis_mae = []
    overall_stenosis_corr = []
    left_coronary_mae = []
    right_coronary_mae = []
    left_coronary_corr = []
    right_coronary_corr = []
    
    left_vessels = ['left_main_stenosis', 'prox_lad_stenosis', 'mid_lad_stenosis', 'dist_lad_stenosis', 'D1_stenosis', 'D2_stenosis']
    right_vessels = ['prox_rca_stenosis', 'mid_rca_stenosis', 'dist_rca_stenosis']
    
    for epoch_key in sorted_epoch_keys:
        metrics = epoch_results[epoch_key]['metrics']
        stenosis_mae = metrics['stenosis']['mae']
        stenosis_corr = metrics['stenosis']['corr']
        
        # Overall metrics
        all_mae_values = [v for v in stenosis_mae.values() if pd.notna(v)]
        all_corr_values = [v for v in stenosis_corr.values() if pd.notna(v)]
        
        overall_stenosis_mae.append(np.mean(all_mae_values) if all_mae_values else np.nan)
        overall_stenosis_corr.append(np.mean(all_corr_values) if all_corr_values else np.nan)
        
        # Left coronary metrics
        left_mae_values = [stenosis_mae.get(v, np.nan) for v in left_vessels if v in stenosis_mae]
        left_corr_values = [stenosis_corr.get(v, np.nan) for v in left_vessels if v in stenosis_corr]
        
        left_coronary_mae.append(np.nanmean(left_mae_values) if left_mae_values else np.nan)
        left_coronary_corr.append(np.nanmean(left_corr_values) if left_corr_values else np.nan)
        
        # Right coronary metrics
        right_mae_values = [stenosis_mae.get(v, np.nan) for v in right_vessels if v in stenosis_mae]
        right_corr_values = [stenosis_corr.get(v, np.nan) for v in right_vessels if v in stenosis_corr]
        
        right_coronary_mae.append(np.nanmean(right_mae_values) if right_mae_values else np.nan)
        right_coronary_corr.append(np.nanmean(right_corr_values) if right_corr_values else np.nan)
    
    # Create comprehensive plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Multi-Epoch Performance Analysis {title_suffix}', fontsize=16, fontweight='bold')
    
    # MAE Plot
    axes[0,0].plot(epochs, overall_stenosis_mae, 'o-', label='Overall MAE', color='red', linewidth=2)
    axes[0,0].plot(epochs, left_coronary_mae, 's-', label='Left Coronary MAE', color='lightcoral', linewidth=2)
    axes[0,0].plot(epochs, right_coronary_mae, '^-', label='Right Coronary MAE', color='lightblue', linewidth=2)
    axes[0,0].set_title('Stenosis Mean Absolute Error (MAE)', fontweight='bold')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('MAE (%)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Correlation Plot
    axes[0,1].plot(epochs, overall_stenosis_corr, 'o-', label='Overall Correlation', color='green', linewidth=2)
    axes[0,1].plot(epochs, left_coronary_corr, 's-', label='Left Coronary Corr', color='lightgreen', linewidth=2)
    axes[0,1].plot(epochs, right_coronary_corr, '^-', label='Right Coronary Corr', color='cyan', linewidth=2)
    axes[0,1].set_title('Stenosis Pearson Correlation', fontweight='bold')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Correlation')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Individual vessel MAE (top 5 vessels)
    vessel_mae_data = {}
    for epoch_key in sorted_epoch_keys:
        stenosis_mae = epoch_results[epoch_key]['metrics']['stenosis']['mae']
        for vessel, mae in stenosis_mae.items():
            if vessel not in vessel_mae_data:
                vessel_mae_data[vessel] = []
            vessel_mae_data[vessel].append(mae)
    
    # Plot top 5 vessels with most data
    vessel_counts = {v: len([x for x in maes if pd.notna(x)]) for v, maes in vessel_mae_data.items()}
    top_vessels = sorted(vessel_counts.keys(), key=lambda x: vessel_counts[x], reverse=True)[:5]
    
    colors = ['purple', 'orange', 'brown', 'pink', 'gray']
    for i, vessel in enumerate(top_vessels):
        if vessel in vessel_mae_data:
            vessel_epochs = epochs[:len(vessel_mae_data[vessel])]
            axes[1,0].plot(vessel_epochs, vessel_mae_data[vessel], 
                          'o-', label=vessel.replace('_stenosis', ''), 
                          color=colors[i % len(colors)], linewidth=1.5)
    
    axes[1,0].set_title('Individual Vessel MAE', fontweight='bold')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('MAE (%)')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Performance improvement plot
    if len(overall_stenosis_mae) > 1 and not np.isnan(overall_stenosis_mae[0]):
        mae_improvement = [(overall_stenosis_mae[0] - mae) / overall_stenosis_mae[0] * 100 
                          for mae in overall_stenosis_mae]
        axes[1,1].plot(epochs, mae_improvement, 'o-', label='MAE Improvement', color='purple', linewidth=2)
        axes[1,1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1,1].set_title('Performance Improvement (%)', fontweight='bold')
        axes[1,1].set_xlabel('Epoch')
        axes[1,1].set_ylabel('Improvement over Epoch 0 (%)')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"\n📊 MULTI-EPOCH SUMMARY:")
    print(f"   🎯 Epochs analyzed: {len(epochs)}")
    
    valid_mae = [x for x in overall_stenosis_mae if pd.notna(x)]
    valid_corr = [x for x in overall_stenosis_corr if pd.notna(x)]
    
    if valid_mae:
        best_mae_idx = np.nanargmin(overall_stenosis_mae)
        print(f"   📈 Best MAE: {overall_stenosis_mae[best_mae_idx]:.3f} (Epoch {epochs[best_mae_idx]})")
    
    if valid_corr:
        best_corr_idx = np.nanargmax(overall_stenosis_corr)
        print(f"   📈 Best Correlation: {overall_stenosis_corr[best_corr_idx]:.3f} (Epoch {epochs[best_corr_idx]})")
    
    if len(valid_mae) > 1:
        final_improvement = (valid_mae[0] - valid_mae[-1]) / valid_mae[0] * 100
        print(f"   🚀 Total MAE improvement: {final_improvement:.2f}%")