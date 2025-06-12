import os
import re
import glob
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

# THIRD‑PARTY LIBS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error

# LOCAL IMPORTS
try:
    from utils.vessel_constants import (
        RCA_VESSELS,
        NON_RCA_VESSELS,
        RIGHT_DOMINANCE_DEPENDENT_VESSELS,
        LEFT_DOMINANCE_DEPENDENT_VESSELS,
        LEFT_CORONARY_DOMINANCE_VESSELS,
        RIGHT_CORONARY_DOMINANCE_VESSELS,
        mode,
    )
except ImportError:
    # Fallback for when running as script
    try:
        from vessel_constants import (
            RCA_VESSELS,
            NON_RCA_VESSELS,
            RIGHT_DOMINANCE_DEPENDENT_VESSELS,
            LEFT_DOMINANCE_DEPENDENT_VESSELS,
            LEFT_CORONARY_DOMINANCE_VESSELS,
            RIGHT_CORONARY_DOMINANCE_VESSELS,
            mode,
        )
    except ImportError:
        print("⚠️ Warning: vessel_constants not available, using fallback definitions")
        # Fallback definitions
        RCA_VESSELS = ["prox_rca_stenosis", "mid_rca_stenosis", "dist_rca_stenosis"]
        NON_RCA_VESSELS = ["left_main_stenosis", "prox_lad_stenosis", "mid_lad_stenosis", "dist_lad_stenosis", "prox_lcx_stenosis", "dist_lcx_stenosis"]
        RIGHT_DOMINANCE_DEPENDENT_VESSELS = ["pda_stenosis", "posterolateral_stenosis"]
        LEFT_DOMINANCE_DEPENDENT_VESSELS = ["lvp_stenosis"]
        LEFT_CORONARY_DOMINANCE_VESSELS = []
        RIGHT_CORONARY_DOMINANCE_VESSELS = []
        
        def mode(lst):
            if not lst:
                return None
            return max(set(lst), key=lst.count)

# TRY loading plotting helpers lazily to avoid circular imports during unit tests
def _import_plotting_functions():
    """Lazy import plotting helpers. Returns (plot_epoch, plot_system, available_flag)."""
    try:
        from utils.plot_metrics import (
            plot_epoch_metrics_line_charts,
            plot_system_specific_metrics_line_charts,
        )

        return plot_epoch_metrics_line_charts, plot_system_specific_metrics_line_charts, True
    except ImportError as exc:  # pragma: no‑cover
        print(
            f"⚠️  Warning: plotting utilities not available – charts will be skipped ({exc})"
        )
        return None, None, False


plot_epoch_metrics_line_charts, plot_system_specific_metrics_line_charts, PLOTTING_AVAILABLE = (
    _import_plotting_functions()
)

# Configure tqdm for pandas
tqdm.pandas()

# ----------------------------------------------------------------------------------
# CONSTANTS
BATCH_SIZE: int = 1_000  # batch numeric ops to reduce memory footprint
CHUNK_SIZE: int = 500

DEFAULT_VESSEL_LABELS: List[str] = [
    "left_main_stenosis",
    "prox_lad_stenosis",
    "mid_lad_stenosis",
    "dist_lad_stenosis",
    "D1_stenosis",
    "D2_stenosis",
    "D3_stenosis",
    "prox_lcx_stenosis",
    "dist_lcx_stenosis",
    "lvp_stenosis",
    "om1_stenosis",
    "om2_stenosis",
    "om3_stenosis",
    "prox_rca_stenosis",
    "mid_rca_stenosis",
    "dist_rca_stenosis",
    "RVG1_stenosis",
    "RVG2_stenosis",
    "pda_stenosis",
    "posterolateral_stenosis",
    "bx_stenosis",
    "lima_or_svg_stenosis",
]

__all__ = [  # exported symbols
    "aggregate_study_level_data",
    "build_val_text_index_map",
    "aggregate_predictions_for_epoch",
    "compute_metrics",
    "load_report_and_epoch_data",
    "create_optimize_analysis_csv",
    "run_ground_truth_comparison_analysis",
    "create_study_level_metrics",
    # Optimized multi-epoch functions
    "load_report_data_once",
    "create_persistent_val_text_map", 
    "process_single_epoch_optimized",
    "run_multi_epoch_analysis_optimized",
]

# ----------------------------------------------------------------------------------
# HELPER FUNCTIONS
def is_valid(x, *, is_ifrhyper: bool = False) -> bool:
    """Return *True* if ``x`` is a valid numeric reading (not sentinel)."""
    if pd.isna(x):
        return False
    if str(x) in {"-1", "-1.0"}:
        return False
    if is_ifrhyper and str(x) in {"0", "0.0"}:
        return False
    return True


def get_vessels_for_dominance(
    dominance_name: Union[str, int, float, None]
) -> Tuple[List[str], List[str]]:
    """Return vessel lists adjusted for coronary dominance."""

    if dominance_name is None or pd.isna(dominance_name):
        print("Warning: dominance_name is None/NaN, defaulting to right dominant")
        dominance_str = "right_dominant"
    else:
        # numeric encodings 0 = right, 1 = left
        if dominance_name in {0, 0.0, "0", "0.0"}:
            dominance_str = "right_dominant"
        elif dominance_name in {1, 1.0, "1", "1.0"}:
            dominance_str = "left_dominant"
        else:
            dominance_str = str(dominance_name).lower()

    if "right" in dominance_str:
        rca_extended = RCA_VESSELS + RIGHT_DOMINANCE_DEPENDENT_VESSELS
        non_rca_extended = NON_RCA_VESSELS
    else:  # left or co‑dominant
        rca_extended = RCA_VESSELS
        non_rca_extended = NON_RCA_VESSELS + LEFT_DOMINANCE_DEPENDENT_VESSELS

    return rca_extended, non_rca_extended


# ----------------------------------------------------------------------------------
# CORE DATAFRAME AGGREGATION UTILITIES
def aggregate_study_level_data(
    df: pd.DataFrame,
    study_col: str = "StudyInstanceUID",
    dominance_col: str = "dominance_name",
    main_structure_col: str = "main_structure_name",
) -> pd.DataFrame:
    """Aggregate all video‑level rows into one row per study.

    Notes
    -----
    * Uses vectorised `groupby` operations for performance.
    * Handles alternative column spellings gracefully.
    """

    print("Starting OPTIMISED study‑level aggregation …")
    print(f"Input DataFrame shape: {df.shape}")

    if study_col not in df.columns:
        print(f"⚠️  '{study_col}' not in DataFrame – returning original frame.")
        return df

    # Resolve alternative column names ------------------------------------------------
    dominance_candidates = [dominance_col, "dominance_class", "coronary_dominance"]
    dominance_col = next((c for c in dominance_candidates if c in df.columns), dominance_col)

    main_structure_candidates = [main_structure_col, "main_structure_class"]
    main_structure_col = next(
        (c for c in main_structure_candidates if c in df.columns), main_structure_col
    )

    print(f"Using columns: {study_col}, {dominance_col}, {main_structure_col}")

    # Determine vessel columns present in df ------------------------------------------
    all_vessel_labels = DEFAULT_VESSEL_LABELS
    vessel_cols_present = [c for c in all_vessel_labels if c in df.columns]

    print(f"Found {len(vessel_cols_present)} vessel stenosis columns")

    # build meta / non‑vessel columns list
    meta_cols = {study_col, dominance_col, main_structure_col}
    vessel_related_cols: set = set()

    vessel_prefixes = {v.replace("_stenosis", "") for v in vessel_cols_present}
    for col in df.columns:
        for prefix in vessel_prefixes:
            if prefix in col and (
                col.endswith("_IFRHYPER") or col.endswith("_calcif") or col.endswith("_stenosis")
            ):
                vessel_related_cols.add(col)

    non_vessel_cols = [c for c in df.columns if c not in vessel_related_cols and c not in meta_cols]

    # Vectorised aggregation ----------------------------------------------------------
    num_studies = df[study_col].nunique()
    print(f"Aggregating {num_studies} studies …")

    agg_dict: Dict[str, Union[str, callable]] = {}
    for col in non_vessel_cols:
        if df[col].dtype == "object":
            agg_dict[col] = lambda x: x.mode().iloc[0] if not x.mode().empty else None
        else:
            agg_dict[col] = "mean"

    non_vessel_agg = (
        df.groupby(study_col)[non_vessel_cols].agg(agg_dict) if non_vessel_cols else pd.DataFrame()
    )
    dominance_agg = df.groupby(study_col)[dominance_col].first() if dominance_col in df.columns else pd.Series()

    # Vessel‑level aggregation --------------------------------------------------------
    vessel_results: List[Dict[str, Union[str, float]]] = []

    for study_uid, study_df in tqdm(df.groupby(study_col), total=num_studies, desc="Studies"):
        study_result: Dict[str, Union[str, float]] = {study_col: study_uid}

        dominance_val = study_df[dominance_col].dropna().iloc[0] if dominance_col in study_df.columns else None
        rca_vessels, non_rca_vessels = get_vessels_for_dominance(dominance_val)

        # aggregate stenosis
        if vessel_cols_present:
            stenosis_means = study_df[vessel_cols_present].mean()
            study_result.update(stenosis_means.to_dict())

        # IFR aggregation
        for vessel in vessel_cols_present:
            ifr_col = vessel.replace("_stenosis", "_IFRHYPER")
            if ifr_col in study_df.columns:
                valid_vals = study_df[ifr_col].dropna()
                valid_vals = valid_vals[valid_vals > 0.01]  # exclude 0, ‑1 sentinels
                study_result[ifr_col] = valid_vals.mean() if not valid_vals.empty else np.nan

        # Calcification aggregation – categorical (use mode)
        for vessel in vessel_cols_present:
            calcif_col = vessel.replace("_stenosis", "_calcif")
            if calcif_col in study_df.columns:
                vals = study_df[calcif_col].dropna()
                study_result[calcif_col] = mode(vals.tolist()) if not vals.empty else None

        vessel_results.append(study_result)

    vessel_df = pd.DataFrame(vessel_results)

    if not non_vessel_agg.empty:
        vessel_df = vessel_df.merge(non_vessel_agg, left_on=study_col, right_index=True, how="left")

    if not dominance_agg.empty:
        vessel_df[dominance_col] = vessel_df[study_col].map(dominance_agg)

    print(f"Aggregation complete. Output shape: {vessel_df.shape}")
    return vessel_df


# ----------------------------------------------------------------------------------
def build_val_text_index_map(
    df_dataset: pd.DataFrame,
    key_col: str = "val_text_index",
    *,
    study_level_aggregation: bool = False,
) -> Dict[int, List[pd.Series]]:
    """Create mapping *val_text_index → list[Series]* with optional study aggregation."""

    print(
        f"Building val_text_index_map (study_level_aggregation={study_level_aggregation}) – input shape {df_dataset.shape}"
    )

    if study_level_aggregation:
        df_dataset = aggregate_study_level_data(df_dataset)

    if key_col not in df_dataset.columns:
        raise KeyError(f"'{key_col}' not found in DataFrame columns")

    valid_df = df_dataset.dropna(subset=[key_col]).copy()
    valid_df[key_col] = valid_df[key_col].astype(int)

    index_map: Dict[int, List[pd.Series]] = {
        val_idx: [row for _, row in grp.iterrows()]
        for val_idx, grp in valid_df.groupby(key_col)
    }

    print(f"Index map built with {len(index_map)} keys")
    return index_map


def aggregate_predictions_for_epoch(
    val_text_map,
    predictions_df,
    topk=5,
    vessel_labels=None,
    study_level_aggregation=False,
    filename_to_study_map=None
):
    """
    UPDATED: Compare ground_truth_idx with predicted_idx_1 to predicted_idx_K (K=5)
    and aggregate based on main_structure_name and dominance.
    """

    if vessel_labels is None:
        vessel_labels = [
            "left_main_stenosis", "prox_lad_stenosis", "mid_lad_stenosis", "dist_lad_stenosis",
            "D1_stenosis", "D2_stenosis", "D3_stenosis", "prox_lcx_stenosis",
            "dist_lcx_stenosis", "lvp_stenosis", "om1_stenosis", "om2_stenosis", "om3_stenosis", 
            "prox_rca_stenosis", "mid_rca_stenosis", "dist_rca_stenosis", "RVG1_stenosis", 
            "RVG2_stenosis", "pda_stenosis", "posterolateral_stenosis", "bx_stenosis", "lima_or_svg_stenosis"
    ]

    if 'ground_truth_idx' not in predictions_df.columns:
        print("Error: 'ground_truth_idx' not found in predictions_df for aggregation.")
        return pd.DataFrame()

    print(f"Processing predictions with study_level_aggregation={study_level_aggregation}")
    print(f"Input predictions shape: {predictions_df.shape}")
    
    # DEBUG: Check val_text_map coverage
    unique_gt_indices_in_predictions = predictions_df['ground_truth_idx'].dropna().nunique()
    gt_indices_in_val_text_map = len(val_text_map)
    print(f"   🔍 DEBUG: Unique ground_truth_idx in predictions: {unique_gt_indices_in_predictions}")
    print(f"   🔍 DEBUG: Entries in val_text_map: {gt_indices_in_val_text_map}")
    
    # Check how many prediction gt_idx values are actually in val_text_map
    predictions_gt_set = set(predictions_df['ground_truth_idx'].dropna().astype(int))
    val_text_map_keys = set(val_text_map.keys())
    common_gt_indices = predictions_gt_set & val_text_map_keys
    missing_in_val_text_map = predictions_gt_set - val_text_map_keys
    
    print(f"   🔍 DEBUG: GT indices in both predictions and val_text_map: {len(common_gt_indices)}")
    print(f"   🔍 DEBUG: GT indices in predictions but NOT in val_text_map: {len(missing_in_val_text_map)}")
    
    if missing_in_val_text_map:
        print(f"   ⚠️ WARNING: {len(missing_in_val_text_map)} ground_truth_idx values from predictions not found in val_text_map!")
        print(f"   🔍 DEBUG: First 5 missing GT indices: {list(missing_in_val_text_map)[:5]}")
    
    # Filter valid predictions
    valid_predictions = predictions_df.dropna(subset=['ground_truth_idx']).copy()
    valid_predictions['ground_truth_idx'] = valid_predictions['ground_truth_idx'].astype(int)
    print(f"   🔍 DEBUG: Valid predictions shape: {valid_predictions.shape}")
    valid_predictions.to_csv('valid_predictions.csv', index=False)
    
    # Pre-compute prediction columns
    pred_cols = [f"predicted_idx_{k}" for k in range(1, topk + 1)]
    existing_pred_cols = [col for col in pred_cols if col in valid_predictions.columns]
    
    if not existing_pred_cols:
        print("Error: No prediction columns found in the DataFrame")
        return pd.DataFrame()
    print(f"Found prediction columns: {existing_pred_cols}")
    
    # Save val_text_map for debugging
    val_text_map_path = 'val_text_map.pkl'
    with open(val_text_map_path, 'wb') as f:
        pickle.dump(val_text_map, f)
    print(f"Saved val_text_map to {val_text_map_path}")
    
    # Filter predictions that exist in val_text_map
    valid_gt_mask = valid_predictions['ground_truth_idx'].isin(val_text_map.keys())
    valid_predictions = valid_predictions[valid_gt_mask]
    print(f"Processing {len(valid_predictions)} valid predictions...")
    
    # FIXED: Create comprehensive study mapping to ensure we don't lose any StudyInstanceUIDs
    # Build a mapping from StudyInstanceUID to all its ground truth indices
    study_to_gt_indices = {}
    gt_to_study_uids = {}  # Changed: GT index can map to multiple StudyInstanceUIDs
    
    for gt_idx, study_data_list in val_text_map.items():
        if study_data_list:
            for study_data in study_data_list:
                study_uid = getattr(study_data, 'StudyInstanceUID', None)
                if study_uid:
                    # Build study_to_gt_indices mapping
                    if study_uid not in study_to_gt_indices:
                        study_to_gt_indices[study_uid] = []
                    if gt_idx not in study_to_gt_indices[study_uid]:
                        study_to_gt_indices[study_uid].append(gt_idx)
                    
                    # Build gt_to_study_uids mapping (one-to-many)
                    if gt_idx not in gt_to_study_uids:
                        gt_to_study_uids[gt_idx] = []
                    if study_uid not in gt_to_study_uids[gt_idx]:
                        gt_to_study_uids[gt_idx].append(study_uid)
    
    print(f"   🔍 DEBUG: Total unique StudyInstanceUIDs in val_text_map: {len(study_to_gt_indices)}")
    
    # DEBUG: Analyze val_text_map structure
    gt_indices_with_multiple_studies = 0
    gt_indices_with_multiple_study_uids = 0
    sample_gt_idx_analysis = list(val_text_map.keys())[0]
    
    for gt_idx, study_data_list in list(val_text_map.items())[:5]:  # Check first 5
        if len(study_data_list) > 1:
            gt_indices_with_multiple_studies += 1
            unique_study_uids = set()
            for study_data in study_data_list:
                study_uid = getattr(study_data, 'StudyInstanceUID', None)
                if study_uid:
                    unique_study_uids.add(study_uid)
            if len(unique_study_uids) > 1:
                gt_indices_with_multiple_study_uids += 1
                if gt_idx == sample_gt_idx_analysis:
                    print(f"   🔍 DEBUG: GT index {gt_idx} has {len(study_data_list)} studies with {len(unique_study_uids)} unique StudyInstanceUIDs:")
                    for i, uid in enumerate(list(unique_study_uids)[:3]):
                        print(f"      - StudyInstanceUID {i+1}: {uid}")
    
    print(f"   🔍 DEBUG: GT indices with multiple studies: {gt_indices_with_multiple_studies} / {len(val_text_map)}")
    print(f"   🔍 DEBUG: GT indices with multiple StudyInstanceUIDs: {gt_indices_with_multiple_study_uids} / {len(val_text_map)}")
    
    # Build a mapping from StudyInstanceUID to its predictions
    study_to_predictions = {}
    missing_study_mappings = []
    
    for _, pred_row in valid_predictions.iterrows():
        gt_idx = int(pred_row['ground_truth_idx'])
        study_uids = gt_to_study_uids.get(gt_idx, [])
        if study_uids:
            # A GT index can map to multiple StudyInstanceUIDs, add prediction to all of them
            for study_uid in study_uids:
                if study_uid not in study_to_predictions:
                    study_to_predictions[study_uid] = []
                study_to_predictions[study_uid].append(pred_row)
        else:
            missing_study_mappings.append(gt_idx)
    
    print(f"   🔍 DEBUG: StudyInstanceUIDs with valid predictions: {len(study_to_predictions)}")
    print(f"   🔍 DEBUG: Ground truth indices missing study mapping: {len(missing_study_mappings)}")
    
    if missing_study_mappings:
        print(f"   🔍 DEBUG: First 10 missing GT indices: {missing_study_mappings[:10]}")
    
    # Check which studies from val_text_map don't have predictions
    studies_without_predictions = set(study_to_gt_indices.keys()) - set(study_to_predictions.keys())
    print(f"   🔍 DEBUG: Studies in val_text_map but without predictions: {len(studies_without_predictions)}")
    
    if len(studies_without_predictions) > 0:
        print(f"   🔍 DEBUG: First 5 studies without predictions: {list(studies_without_predictions)[:5]}")
        
        # Check if these studies have ground truth indices that are not in the predictions
        sample_study = list(studies_without_predictions)[0]
        sample_gt_indices = study_to_gt_indices[sample_study]
        predictions_gt_indices = set(valid_predictions['ground_truth_idx'].astype(int))
        missing_from_predictions = set(sample_gt_indices) - predictions_gt_indices
        print(f"   🔍 DEBUG: Sample study '{sample_study}' has GT indices {sample_gt_indices}")
        print(f"   🔍 DEBUG: GT indices missing from predictions: {list(missing_from_predictions)[:5]}")
        
        # Check if ALL ground truth indices for studies without predictions are missing
        all_gt_indices_for_missing_studies = []
        for study in list(studies_without_predictions)[:10]:  # Check first 10
            all_gt_indices_for_missing_studies.extend(study_to_gt_indices[study])
        
        missing_gt_count = len(set(all_gt_indices_for_missing_studies) - predictions_gt_indices)
        print(f"   🔍 DEBUG: Of {len(set(all_gt_indices_for_missing_studies))} GT indices from missing studies, {missing_gt_count} are not in predictions")
        
        # DEBUGGING THE MAPPING ISSUE: Check if GT indices from missing studies map to different StudyInstanceUIDs
        if len(sample_gt_indices) > 0:
            sample_gt_idx = sample_gt_indices[0]
            mapped_study_uids = gt_to_study_uids.get(sample_gt_idx, [])
            print(f"   🔍 DEBUG: GT index {sample_gt_idx} from study '{sample_study}' maps to: {mapped_study_uids}")
            
            # Check if this GT index appears in predictions and what study it's associated with
            sample_predictions = valid_predictions[valid_predictions['ground_truth_idx'] == sample_gt_idx]
            if len(sample_predictions) > 0:
                print(f"   🔍 DEBUG: GT index {sample_gt_idx} appears in {len(sample_predictions)} predictions")
                # Try to find what study this prediction thinks it belongs to
                for _, pred_row in sample_predictions.head(2).iterrows():
                    filename = pred_row.get('FileName', 'NO_FILENAME')
                    print(f"   🔍 DEBUG: Prediction filename: {filename}")
            else:
                print(f"   🔍 DEBUG: GT index {sample_gt_idx} does NOT appear in predictions (contradiction!)")
        
        # Check for potential duplication in the new gt_to_study_uids mapping
        total_gt_to_study_mappings = sum(len(study_uids) for study_uids in gt_to_study_uids.values())
        print(f"   🔍 DEBUG: Total GT-to-Study mappings: {total_gt_to_study_mappings} (should be >= {len(gt_to_study_uids)} GT indices)")
        
        # Find GT indices that map to multiple StudyInstanceUIDs
        multi_study_gt_indices = {gt_idx: study_uids for gt_idx, study_uids in gt_to_study_uids.items() if len(study_uids) > 1}
        print(f"   🔍 DEBUG: GT indices mapping to multiple studies: {len(multi_study_gt_indices)} / {len(gt_to_study_uids)}")
        
        if len(multi_study_gt_indices) > 0:
            sample_multi_gt_idx = list(multi_study_gt_indices.keys())[0]
            sample_multi_study_uids = multi_study_gt_indices[sample_multi_gt_idx]
            print(f"   🔍 DEBUG: Sample GT index {sample_multi_gt_idx} maps to studies: {sample_multi_study_uids[:3]}")
            
            # Check if predictions for this GT index are now distributed to all these studies
            predictions_for_gt = valid_predictions[valid_predictions['ground_truth_idx'] == sample_multi_gt_idx]
            print(f"   🔍 DEBUG: GT index {sample_multi_gt_idx} has {len(predictions_for_gt)} predictions")
            studies_that_got_these_predictions = set()
            for study_uid in sample_multi_study_uids:
                if study_uid in study_to_predictions:
                    studies_that_got_these_predictions.add(study_uid)
            print(f"   🔍 DEBUG: Of {len(sample_multi_study_uids)} studies, {len(studies_that_got_these_predictions)} got predictions for this GT index")
    
    aggregated_rows = []
    
    if study_level_aggregation:
        # FIXED: Process at study level - ensure we capture ALL studies with both left and right coronary data
        print("Processing with study-level aggregation...")
        
        for study_uid in tqdm(study_to_gt_indices.keys(), desc="Processing studies"):
            gt_indices_for_study = study_to_gt_indices[study_uid]
            predictions_for_study = study_to_predictions.get(study_uid, [])
            
            # Collect all ground truth data for this study (both left and right coronary)
            left_gt_data = []
            right_gt_data = []
            
            for gt_idx in gt_indices_for_study:
                if gt_idx in val_text_map:
                    gt_data_list = val_text_map[gt_idx]
                    for gt_row in gt_data_list:
                        main_structure = getattr(gt_row, 'main_structure_name', None)
                        if main_structure == 'Left Coronary':
                            left_gt_data.append((gt_idx, gt_row))
                        elif main_structure == 'Right Coronary':
                            right_gt_data.append((gt_idx, gt_row))
            
            # Process this study if we have predictions for it
            if predictions_for_study:
                # Create one row per main structure type that has ground truth data
                for structure_name, gt_data_pairs in [('Left Coronary', left_gt_data), ('Right Coronary', right_gt_data)]:
                    if not gt_data_pairs:
                        continue
                    
                    # For study-level, we may have multiple GT indices for the same structure
                    # Take the first one as representative for this analysis
                    gt_idx, gt_row = gt_data_pairs[0]
                    dominance_name = getattr(gt_row, 'dominance_name', None)
                    
                    # Determine vessels to include based on structure and dominance
                    if structure_name == 'Left Coronary':
                        rca_vessels, non_rca_vessels = get_vessels_for_dominance(dominance_name)
                        target_vessels = non_rca_vessels
                    else:  # Right Coronary
                        rca_vessels, non_rca_vessels = get_vessels_for_dominance(dominance_name)
                        target_vessels = rca_vessels
                    
                    # Filter target vessels to only include those in our vessel_labels
                    target_vessels = [v for v in target_vessels if v in vessel_labels]
                    
                    # Initialize row data
                    row_data = {
                        'ground_truth_idx': gt_idx,
                        'main_structure_name': structure_name,
                        'dominance_name': dominance_name,
                        'target_vessel_count': len(target_vessels),
                        'StudyInstanceUID': study_uid,
                        'all_gt_indices_for_study': ';'.join(map(str, sorted(gt_indices_for_study))),
                        'predictions_count_for_study': len(predictions_for_study)
                    }
                    
                    # Get ground truth stenosis values for target vessels (aggregate across all GT data for this structure)
                    gt_stenosis_values = {}
                    for vessel in target_vessels:
                        vessel_values = []
                        for gt_idx_pair, gt_data_row in gt_data_pairs:
                            if hasattr(gt_data_row, vessel) and is_valid(getattr(gt_data_row, vessel), is_ifrhyper=False):
                                vessel_values.append(getattr(gt_data_row, vessel))
                        if vessel_values:
                            aggregated_gt_value = np.mean(vessel_values)
                            gt_stenosis_values[vessel] = aggregated_gt_value
                            row_data[f'gt_{vessel}'] = aggregated_gt_value
                        else:
                            row_data[f'gt_{vessel}'] = np.nan
                    
                    # Process predictions for this study and structure
                    pred_stenosis_values = {vessel: [] for vessel in target_vessels}
                    retrieval_metrics = {
                        'top1_match': False,
                        'top3_match': False,
                        'top5_match': False,
                        'rank_of_gt': -1,
                        'valid_predictions': 0
                    }
                    
                    # Look for predictions that match any of the ground truth indices for this structure
                    structure_gt_indices = [pair[0] for pair in gt_data_pairs]
                    
                    for pred_row in predictions_for_study:
                        # Add filename if available
                        filename_col = None
                        for col in pred_row.index:
                            if 'FileName' in col or 'filename' in col.lower():
                                filename_col = col
                                break
                        if filename_col and 'FileName' not in row_data:
                            row_data['FileName'] = pred_row[filename_col]
                        
                        # Check each predicted index
                        for rank, pred_col in enumerate(existing_pred_cols, 1):
                            if pd.notna(pred_row[pred_col]):
                                try:
                                    pred_idx = int(pred_row[pred_col])
                                    retrieval_metrics['valid_predictions'] += 1
                                    
                                    # Check if this prediction matches any ground truth for this structure
                                    if pred_idx in structure_gt_indices:
                                        if rank == 1:
                                            retrieval_metrics['top1_match'] = True
                                        if rank <= 3:
                                            retrieval_metrics['top3_match'] = True
                                        if rank <= 5:
                                            retrieval_metrics['top5_match'] = True
                                        if retrieval_metrics['rank_of_gt'] == -1:
                                            retrieval_metrics['rank_of_gt'] = rank
                                    
                                    # Get stenosis values from this predicted index
                                    if pred_idx in val_text_map:
                                        pred_data_list = val_text_map[pred_idx]
                                        if pred_data_list:
                                            pred_data = pred_data_list[0]
                                            
                                            # Extract stenosis values for target vessels
                                            for vessel in target_vessels:
                                                if hasattr(pred_data, vessel) and is_valid(getattr(pred_data, vessel), is_ifrhyper=False):
                                                    pred_stenosis_values[vessel].append(getattr(pred_data, vessel))
                                
                                except (ValueError, TypeError):
                                    continue
                    
                    # Add retrieval metrics to row data
                    row_data.update(retrieval_metrics)
                    
                    # Aggregate predicted stenosis values (mean across all predictions)
                    for vessel in target_vessels:
                        if pred_stenosis_values[vessel]:
                            row_data[f'pred_{vessel}'] = np.mean(pred_stenosis_values[vessel])
                        else:
                            row_data[f'pred_{vessel}'] = np.nan
                    
                    # Calculate structure-level aggregated metrics
                    valid_gt_values = [v for v in gt_stenosis_values.values() if pd.notna(v)]
                    valid_pred_values = [row_data[f'pred_{vessel}'] for vessel in target_vessels 
                                       if pd.notna(row_data.get(f'pred_{vessel}', np.nan))]
                    
                    row_data[f'gt_{structure_name.lower().replace(" ", "_")}_mean_stenosis'] = np.mean(valid_gt_values) if valid_gt_values else np.nan
                    row_data[f'pred_{structure_name.lower().replace(" ", "_")}_mean_stenosis'] = np.mean(valid_pred_values) if valid_pred_values else np.nan
                    
                    aggregated_rows.append(row_data)
    
    else:
        # Individual-level processing (original logic)
        print("Processing with individual-level aggregation...")
        
        for _, pred_row in tqdm(valid_predictions.iterrows(), total=len(valid_predictions), desc="Processing predictions"):
            gt_idx = int(pred_row['ground_truth_idx'])
            
            # Get ground truth data
            if gt_idx not in val_text_map:
                continue
                
            gt_data_list = val_text_map[gt_idx]
            if not gt_data_list:
                continue
                
            # FIXED: Handle multiple studies per GT index - process each study separately
            for gt_row in gt_data_list:
                # Get main structure and dominance from ground truth
                main_structure = getattr(gt_row, 'main_structure_name', None)
                dominance_name = getattr(gt_row, 'dominance_name', None)
                
                if main_structure not in ['Left Coronary', 'Right Coronary']:
                    # DEBUG: Track what main_structure values are being filtered out
                    # print(f"Warning: Unknown main_structure_name '{main_structure}' for gt_idx {gt_idx}")
                    continue
                
                # Determine vessels to include based on structure and dominance
                if main_structure == 'Left Coronary':
                    # Left coronary vessels + dominance-dependent vessels
                    rca_vessels, non_rca_vessels = get_vessels_for_dominance(dominance_name)
                    target_vessels = non_rca_vessels  # This includes left vessels + dominance-dependent vessels for left dominant
                else:  # Right Coronary
                    # Right coronary vessels + dominance-dependent vessels
                    rca_vessels, non_rca_vessels = get_vessels_for_dominance(dominance_name)
                    target_vessels = rca_vessels  # This includes right vessels + dominance-dependent vessels for right dominant
                
                # Filter target vessels to only include those in our vessel_labels
                target_vessels = [v for v in target_vessels if v in vessel_labels]
                
                # Initialize row data
                row_data = {
                    'ground_truth_idx': gt_idx,
                    'main_structure_name': main_structure,
                    'dominance_name': dominance_name,
                    'target_vessel_count': len(target_vessels)
                }
                
                # Add StudyInstanceUID from ground truth data
                study_uid = getattr(gt_row, 'StudyInstanceUID', None)
                if study_uid:
                    row_data['StudyInstanceUID'] = study_uid
                
                # Add FileName if available
                filename_col = None
                for col in pred_row.index:
                    if 'FileName' in col or 'filename' in col.lower():
                        filename_col = col
                        break
                if filename_col:
                    row_data['FileName'] = pred_row[filename_col]
                
                # Get ground truth stenosis values for target vessels
                gt_stenosis_values = {}
                for vessel in target_vessels:
                    if hasattr(gt_row, vessel) and is_valid(getattr(gt_row, vessel), is_ifrhyper=False):
                        gt_stenosis_values[vessel] = getattr(gt_row, vessel)
                        row_data[f'gt_{vessel}'] = getattr(gt_row, vessel)
                    else:
                        row_data[f'gt_{vessel}'] = np.nan
                
                # Process predicted indices and compare with ground truth
                pred_stenosis_values = {vessel: [] for vessel in target_vessels}
                retrieval_metrics = {
                    'top1_match': False,
                    'top3_match': False,
                    'top5_match': False,
                    'rank_of_gt': -1,
                    'valid_predictions': 0
                }
                
                # Check each predicted index
                for rank, pred_col in enumerate(existing_pred_cols, 1):
                    if pd.notna(pred_row[pred_col]):
                        try:
                            pred_idx = int(pred_row[pred_col])
                            retrieval_metrics['valid_predictions'] += 1
                            
                            # Check if this prediction matches ground truth
                            if pred_idx == gt_idx:
                                if rank == 1:
                                    retrieval_metrics['top1_match'] = True
                                if rank <= 3:
                                    retrieval_metrics['top3_match'] = True
                                if rank <= 5:
                                    retrieval_metrics['top5_match'] = True
                                if retrieval_metrics['rank_of_gt'] == -1:
                                    retrieval_metrics['rank_of_gt'] = rank
                            
                            # Get stenosis values from this predicted index
                            if pred_idx in val_text_map:
                                pred_data_list = val_text_map[pred_idx]
                                # FIXED: When using predictions, use the first study for consistency
                                # (or we could aggregate across all studies with same pred_idx)
                                if pred_data_list:
                                    pred_data = pred_data_list[0]
                                    
                                    # Extract stenosis values for target vessels
                                    for vessel in target_vessels:
                                        if hasattr(pred_data, vessel) and is_valid(getattr(pred_data, vessel), is_ifrhyper=False):
                                            pred_stenosis_values[vessel].append(getattr(pred_data, vessel))
                        
                        except (ValueError, TypeError):
                            continue
                
                # Add retrieval metrics to row data
                row_data.update(retrieval_metrics)
                
                # Aggregate predicted stenosis values (mean across all predictions)
                for vessel in target_vessels:
                    if pred_stenosis_values[vessel]:
                        row_data[f'pred_{vessel}'] = np.mean(pred_stenosis_values[vessel])
                    else:
                        row_data[f'pred_{vessel}'] = np.nan
                
                # Calculate structure-level aggregated metrics
                valid_gt_values = [v for v in gt_stenosis_values.values() if pd.notna(v)]
                valid_pred_values = [row_data[f'pred_{vessel}'] for vessel in target_vessels 
                                   if pd.notna(row_data.get(f'pred_{vessel}', np.nan))]
                
                row_data[f'gt_{main_structure.lower().replace(" ", "_")}_mean_stenosis'] = np.mean(valid_gt_values) if valid_gt_values else np.nan
                row_data[f'pred_{main_structure.lower().replace(" ", "_")}_mean_stenosis'] = np.mean(valid_pred_values) if valid_pred_values else np.nan
                
                aggregated_rows.append(row_data)
    
    # Create final DataFrame
    if aggregated_rows:
        result_df = pd.DataFrame(aggregated_rows)
        print(f"Successfully aggregated {len(result_df)} rows")
        
        # DEBUG: Check final unique study count
        if 'StudyInstanceUID' in result_df.columns:
            final_unique_studies = result_df['StudyInstanceUID'].nunique()
            print(f"   🔍 DEBUG: Final unique StudyInstanceUIDs in result: {final_unique_studies}")
            
            if study_level_aggregation:
                total_studies_in_val_text_map = len(study_to_gt_indices)
                studies_with_predictions = len(study_to_predictions)
                print(f"   🔍 DEBUG: Total studies in val_text_map: {total_studies_in_val_text_map}")
                print(f"   🔍 DEBUG: Studies with predictions: {studies_with_predictions}")
                print(f"   🔍 DEBUG: Coverage: {studies_with_predictions/total_studies_in_val_text_map*100:.1f}% of studies have predictions")
        
        # Print retrieval performance summary
        if len(result_df) > 0:
            top1_accuracy = result_df['top1_match'].mean()
            top3_accuracy = result_df['top3_match'].mean()
            top5_accuracy = result_df['top5_match'].mean()
            avg_rank = result_df[result_df['rank_of_gt'] > 0]['rank_of_gt'].mean()
            
            print(f"\n=== RETRIEVAL PERFORMANCE SUMMARY ===")
            print(f"Top-1 Accuracy: {top1_accuracy:.3f}")
            print(f"Top-3 Accuracy: {top3_accuracy:.3f}")
            print(f"Top-5 Accuracy: {top5_accuracy:.3f}")
            print(f"Average Rank (when found): {avg_rank:.2f}")
            print(f"Left Coronary cases: {sum(result_df['main_structure_name'] == 'Left Coronary')}")
            print(f"Right Coronary cases: {sum(result_df['main_structure_name'] == 'Right Coronary')}")
            print("=====================================\n")
        
        return result_df
    else:
        print("Warning: No rows were aggregated")
        return pd.DataFrame()

def compute_metrics(agg_df, vessel_labels):
    """
    Compute comprehensive metrics for stenosis, calcification, and IFR predictions
    
    Args:
        agg_df: Aggregated dataframe with ground truth and predicted values
        vessel_labels: List of vessel label names to compute metrics for
        
    Returns:
        dict: Comprehensive metrics including MAE, correlation, and accuracy
    """
    print("Computing comprehensive metrics...")
    
    if agg_df.empty:
        print("Warning: Empty aggregated dataframe provided")
        return {
            "stenosis": {"mae": {}, "corr": {}},
            "calcif": {"accuracy": {}},
            "ifr": {"mae": {}, "corr": {}},
            "string_accuracy": {}
        }
    
    metrics = {
        "stenosis": {"mae": {}, "corr": {}},
        "calcif": {"accuracy": {}}, 
        "ifr": {"mae": {}, "corr": {}},
        "string_accuracy": {}
    }
    
    total_vessels_processed = 0
    
    # 1. Stenosis metrics (MAE and correlation)
    print("   📊 Computing stenosis metrics...")
    for vessel in tqdm(vessel_labels, desc="Processing vessels", leave=False):
        # FIXED: Use the correct column naming convention from aggregated dataframe
        gt_col = f'gt_{vessel}'
        pred_col = f'pred_{vessel}'
        
        if gt_col in agg_df.columns and pred_col in agg_df.columns:
            # Get valid data points (both GT and prediction available)
            valid_mask = agg_df[gt_col].notna() & agg_df[pred_col].notna()
            
            if valid_mask.sum() > 5:  # Need at least 5 valid points for meaningful metrics
                gt_values = agg_df[gt_col][valid_mask]
                pred_values = agg_df[pred_col][valid_mask]
                
                try:
                    # Calculate MAE
                    mae = mean_absolute_error(gt_values, pred_values)
                    metrics["stenosis"]["mae"][vessel] = mae
                    
                    # Calculate correlation (only if we have variance in both GT and predictions)
                    if len(set(gt_values)) > 1 and len(set(pred_values)) > 1:
                        corr, p_value = pearsonr(gt_values, pred_values)
                        if not np.isnan(corr):
                            metrics["stenosis"]["corr"][vessel] = corr
                    
                    total_vessels_processed += 1
                    
                except Exception as e:
                    print(f"   ⚠️ Error computing metrics for {vessel}: {e}")
    
    # 2. IFR metrics (if available)
    print("   💛 Computing IFR metrics...")
    ifr_columns = [col for col in agg_df.columns if 'ifr' in col.lower() and not col.startswith('pred_')]
    
    for ifr_col in ifr_columns:
        pred_ifr_col = f'pred_{ifr_col.replace("gt_", "")}'
        
        if pred_ifr_col in agg_df.columns:
            # Initial mask for non-null values
            valid_mask = agg_df[ifr_col].notna() & agg_df[pred_ifr_col].notna()
            
            if valid_mask.sum() > 5:
                # Filter out missing IFR values (-1.0 and 0.0)
                gt_ifr_raw = agg_df[ifr_col][valid_mask]
                pred_ifr_raw = agg_df[pred_ifr_col][valid_mask]
                
                # Remove -1.0 and 0.0 values (missing data indicators)
                valid_ifr_mask = (gt_ifr_raw != -1.0) & (gt_ifr_raw != 0.0) & (pred_ifr_raw != -1.0) & (pred_ifr_raw != 0.0)
                
                if valid_ifr_mask.sum() > 5:  # Need at least 5 valid IFR measurements
                    gt_ifr = gt_ifr_raw[valid_ifr_mask]
                    pred_ifr = pred_ifr_raw[valid_ifr_mask]
                    
                    try:
                        # MAE
                        mae_ifr = mean_absolute_error(gt_ifr, pred_ifr)
                        metrics["ifr"]["mae"][ifr_col] = mae_ifr
                        
                        # Correlation
                        if len(set(gt_ifr)) > 1 and len(set(pred_ifr)) > 1:
                            corr_ifr, _ = pearsonr(gt_ifr, pred_ifr)
                            if not np.isnan(corr_ifr):
                                metrics["ifr"]["corr"][ifr_col] = corr_ifr
                        
                        print(f"   ✅ IFR metrics for {ifr_col}: MAE={mae_ifr:.3f} (n={valid_ifr_mask.sum()} valid measurements, excluded {valid_mask.sum()-valid_ifr_mask.sum()} missing values)")
                    
                    except Exception as e:
                        print(f"   ⚠️ Error computing IFR metrics for {ifr_col}: {e}")
                else:
                    print(f"   ⚠️ Insufficient valid IFR data for {ifr_col} after removing missing values: {valid_ifr_mask.sum()} valid (excluded {valid_mask.sum()-valid_ifr_mask.sum()} missing)")
            else:
                print(f"   ⚠️ Insufficient data for {ifr_col}: only {valid_mask.sum()} non-null values")
    
    # 3. Calcification accuracy (binary classification)
    print("   🦴 Computing calcification accuracy...")
    calcif_columns = [col for col in agg_df.columns if 'calcif' in col.lower() and not col.startswith('pred_')]
    
    for calcif_col in calcif_columns:
        pred_calcif_col = f'pred_{calcif_col.replace("gt_", "")}'
        
        if pred_calcif_col in agg_df.columns:
            valid_mask = agg_df[calcif_col].notna() & agg_df[pred_calcif_col].notna()
            
            if valid_mask.sum() > 5:
                try:
                    # Map French calcification categories to numeric values
                    def map_calcification_value(val):
                        if pd.isna(val):
                            return None
                        val_str = str(val).strip().lower()
                        
                        # Map to 0 (no calcification)
                        if val_str in ['-1', 'pas de calcification']:
                            return 0
                        # Map to 1 (minimal calcifications)  
                        elif 'minimes' in val_str:
                            return 1
                        # Map to 2 (moderate calcifications)
                        elif 'modérées' in val_str or 'moderees' in val_str:
                            return 2
                        # Map to 3 (important/severe calcifications)
                        elif 'importantes' in val_str or 'bourgeon calcaire' in val_str:
                            return 3
                        else:
                            # Try to convert to numeric directly (for already numeric values)
                            try:
                                return float(val)
                            except:
                                return None
                    
                    # Apply mapping to both ground truth and predictions
                    gt_calcif_mapped = agg_df[calcif_col][valid_mask].apply(map_calcification_value)
                    pred_calcif_mapped = agg_df[pred_calcif_col][valid_mask].apply(map_calcification_value)
                    
                    # Remove any None values that couldn't be mapped
                    final_valid_mask = gt_calcif_mapped.notna() & pred_calcif_mapped.notna()
                    
                    if final_valid_mask.sum() > 5:  # Need at least 5 valid values
                        gt_calcif_final = gt_calcif_mapped[final_valid_mask]
                        pred_calcif_final = pred_calcif_mapped[final_valid_mask]
                        
                        # Calculate accuracy (exact match between categories)
                        accuracy = (gt_calcif_final == pred_calcif_final).mean()
                        metrics["calcif"]["accuracy"][calcif_col] = accuracy
                        
                        # Calculate category-specific accuracy
                        category_breakdown = {}
                        for category in [0, 1, 2, 3]:
                            category_mask = gt_calcif_final == category
                            if category_mask.sum() > 0:
                                category_accuracy = (gt_calcif_final[category_mask] == pred_calcif_final[category_mask]).mean()
                                category_breakdown[category] = {
                                    'accuracy': category_accuracy,
                                    'count': category_mask.sum()
                                }
                        
                        # Store category breakdown in metrics
                        if "category_breakdown" not in metrics["calcif"]:
                            metrics["calcif"]["category_breakdown"] = {}
                        metrics["calcif"]["category_breakdown"][calcif_col] = category_breakdown
                        
                        print(f"   ✅ Calcification accuracy for {calcif_col}: {accuracy:.3f} (n={final_valid_mask.sum()})")
                        
                        # Print category-specific accuracies
                        category_names = {0: "No calcif", 1: "Minimal", 2: "Moderate", 3: "Severe"}
                        for cat, info in category_breakdown.items():
                            print(f"      📊 {category_names[cat]} (cat {cat}): {info['accuracy']:.3f} (n={info['count']})")
                    else:
                        print(f"   ⚠️ Insufficient valid calcification data for {calcif_col} after mapping: {final_valid_mask.sum()} values")
                
                except Exception as e:
                    print(f"   ⚠️ Error computing calcification accuracy for {calcif_col}: {e}")
    
    # Print overall calcification category summary
    if "category_breakdown" in metrics["calcif"] and metrics["calcif"]["category_breakdown"]:
        print(f"\n   📊 CALCIFICATION CATEGORY SUMMARY:")
        category_names = {0: "No calcification", 1: "Minimal calcifications", 2: "Moderate calcifications", 3: "Severe calcifications"}
        
        # Aggregate across all vessels
        overall_category_stats = {}
        for category in [0, 1, 2, 3]:
            total_correct = 0
            total_count = 0
            
            for vessel_breakdown in metrics["calcif"]["category_breakdown"].values():
                if category in vessel_breakdown:
                    vessel_info = vessel_breakdown[category]
                    total_correct += vessel_info['accuracy'] * vessel_info['count']
                    total_count += vessel_info['count']
            
            if total_count > 0:
                overall_accuracy = total_correct / total_count
                overall_category_stats[category] = {
                    'accuracy': overall_accuracy,
                    'total_count': total_count
                }
        
        # Print overall category performance
        for category, stats in overall_category_stats.items():
            print(f"   🎯 {category_names[category]}: {stats['accuracy']:.3f} overall accuracy (n={stats['total_count']} total)")
    
    # 4. String-based accuracy (for text predictions if any)
    string_columns = [col for col in agg_df.columns if 'pred_string' in col.lower()]
    for string_col in string_columns:
        base_col = string_col.replace('pred_', '').replace('_string', '')
        gt_string_col = f'gt_{base_col}_string' if f'gt_{base_col}_string' in agg_df.columns else f'gt_{base_col}'
        
        if gt_string_col in agg_df.columns:
            valid_mask = agg_df[gt_string_col].notna() & agg_df[string_col].notna()
            
            if valid_mask.sum() > 5:
                try:
                    gt_strings = agg_df[gt_string_col][valid_mask]
                    pred_strings = agg_df[string_col][valid_mask]
                    
                    accuracy = (gt_strings == pred_strings).mean()
                    metrics["string_accuracy"][string_col] = accuracy
                
                except Exception as e:
                    print(f"   ⚠️ Error computing string accuracy for {string_col}: {e}")
    
    # Summary
    stenosis_mae_count = len(metrics["stenosis"]["mae"])
    stenosis_corr_count = len(metrics["stenosis"]["corr"])
    ifr_mae_count = len(metrics["ifr"]["mae"])
    calcif_acc_count = len(metrics["calcif"]["accuracy"])
    
    print(f"   ✅ Metrics computed: {stenosis_mae_count} stenosis MAE, {stenosis_corr_count} stenosis correlations")
    print(f"   ✅ Additional: {ifr_mae_count} IFR MAE, {calcif_acc_count} calcification accuracies")
    
    return metrics

def load_report_and_epoch_data(
    report_csv_path,
    epoch_csv_path,
):
    """
    NEW: Load report data and epoch predictions, merge on FileName to create proper val_text_map
    for ground_truth_idx comparison with predicted_idx_1 to predicted_idx_K.
    
    Args:
        report_csv_path: Path to the report CSV file
        epoch_csv_path: Path to the epoch prediction CSV file (e.g., val_epoch_0.csv)
        val_unique_texts_path: Path to val_unique_texts.csv (auto-detected if None)
        
    Returns:
        tuple: (df_report, df_epoch, val_text_map) or (None, None, None) on failure
    """
    print(f"\n--- Loading Report and Epoch Data for Ground Truth Comparison ---")
    
    # Input validation
    if not report_csv_path or not os.path.exists(report_csv_path):
        print(f"   ✗ Report file not found: {report_csv_path}")
        return None, None, None
        
    if not epoch_csv_path or not os.path.exists(epoch_csv_path):
        print(f"   ✗ Epoch file not found: {epoch_csv_path}")
        return None, None, None
    
    # 1. Load report data
    print(f"1. Loading report from: {report_csv_path}")
    df_report = None
    
    # Try different separators and encodings for report
    load_attempts = [
        {'sep': 'α', 'encoding': 'utf-8', 'engine': 'python'},
        {'sep': ',', 'encoding': 'utf-8'},
        {'sep': ';', 'encoding': 'utf-8'},
        {'sep': '\t', 'encoding': 'utf-8'},
        {'sep': 'α', 'encoding': 'latin-1', 'engine': 'python'},
        {'sep': ',', 'encoding': 'latin-1'}
    ]
    
    for i, params in enumerate(load_attempts):
        try:
            df_report = pd.read_csv(report_csv_path, on_bad_lines='skip', **params)
            print(f"   ✓ Loaded report with {params['sep']} separator: {len(df_report)} rows")
            break
        except Exception as e:
            if i == len(load_attempts) - 1:  # Last attempt
                print(f"   ✗ Failed to load report with all attempted separators")
                print(f"   Last error: {e}")
                return None, None, None
            continue
    
    if df_report is None or len(df_report) == 0:
        print("   ✗ Report is empty after loading")
        return None, None, None
    
    # 2. Load epoch predictions
    print(f"2. Loading epoch predictions from: {epoch_csv_path}")
    try:
        df_epoch = pd.read_csv(epoch_csv_path)
        print(f"   ✓ Loaded epoch predictions: {len(df_epoch)} rows")
    except Exception as e:
        print(f"   ✗ Failed to load epoch predictions: {e}")
        return None, None, None
    
    
    # 4. Validate essential columns
    # Check for FileName column in both datasets
    report_filename_col = None
    epoch_filename_col = None
    
    filename_col_candidates = ['FileName', 'filename', 'file_name', 'File_Name', 'filepath', 'FilePath']
    
    for col in df_report.columns:
        for candidate in filename_col_candidates:
            if candidate.lower() in col.lower():
                report_filename_col = col
                break
        if report_filename_col:
            break
    
    for col in df_epoch.columns:
        for candidate in filename_col_candidates:
            if candidate.lower() in col.lower():
                epoch_filename_col = col
                break
        if epoch_filename_col:
            break
    
    if not report_filename_col:
        print(f"   ✗ FileName column not found in report. Available: {list(df_report.columns[:10])}")
        return None, None, None
    if not epoch_filename_col:
        print(f"   ✗ FileName column not found in epoch predictions. Available: {list(df_epoch.columns[:10])}")
        return None, None, None
    
    print(f"   ✓ Found FileName columns - Report: {report_filename_col}, Epoch: {epoch_filename_col}")
    
    # Check for ground_truth_idx in epoch predictions
    if 'ground_truth_idx' not in df_epoch.columns:
        print(f"   ✗ ground_truth_idx column not found in epoch predictions")
        return None, None, None
    
    # Check for main_structure_name and dominance_name in report
    required_report_cols = ['main_structure_name', 'dominance_name']
    missing_report_cols = [col for col in required_report_cols if col not in df_report.columns]
    if missing_report_cols:
        print(f"   ⚠️ Missing columns in report (will try alternatives): {missing_report_cols}")
    
    # 5. Merge epoch predictions with report on FileName
    print(f"4. Merging epoch predictions with report on FileName...")
    try:
        # Enhanced debugging for merge process
        print(f"   🔍 DEBUG: Report data shape: {df_report.shape}")
        print(f"   🔍 DEBUG: Epoch predictions shape: {df_epoch.shape}")
        
        # Check unique counts before merge
        unique_report_files = df_report[report_filename_col].nunique()
        unique_epoch_files = df_epoch[epoch_filename_col].nunique()
        unique_report_studies = df_report[df_report['Split'] == 'val']['StudyInstanceUID'].nunique() if 'StudyInstanceUID' in df_report.columns else 0
        print(f"   📊 DEBUG: Unique filenames in report: {unique_report_files}")
        print(f"   📊 DEBUG: Unique filenames in epoch: {unique_epoch_files}")
        print(f"   📊 DEBUG: Unique StudyInstanceUIDs in report: {unique_report_studies}")
        
        # Check sample filenames for debugging
        sample_report_files = df_report[report_filename_col].head(3).tolist()
        sample_epoch_files = df_epoch[epoch_filename_col].head(3).tolist()
        print(f"   🔍 DEBUG: Sample report filenames: {sample_report_files}")
        print(f"   🔍 DEBUG: Sample epoch filenames: {sample_epoch_files}")
        
        # Check for exact filename matches
        common_files = set(df_report[report_filename_col]) & set(df_epoch[epoch_filename_col])
        print(f"   🎯 DEBUG: Common filenames found: {len(common_files)}")
        
        if len(common_files) < unique_epoch_files:
            print(f"   ⚠️ WARNING: Only {len(common_files)}/{unique_epoch_files} epoch filenames found in report")
            
            # Show some mismatched filenames for debugging
            epoch_files_set = set(df_epoch[epoch_filename_col])
            report_files_set = set(df_report[report_filename_col])
            
            missing_in_report = epoch_files_set - report_files_set
            missing_in_epoch = report_files_set - epoch_files_set
            
            if missing_in_report:
                print(f"   🔍 DEBUG: First 5 epoch filenames not in report: {list(missing_in_report)[:5]}")
            if missing_in_epoch:
                print(f"   🔍 DEBUG: First 5 report filenames not in epoch: {list(missing_in_epoch)[:5]}")
        
        # Merge epoch with report to get ground_truth_idx mapping
        merged_df = pd.merge(
            df_epoch[[epoch_filename_col, 'ground_truth_idx']], 
            df_report, 
            left_on=epoch_filename_col, 
            right_on=report_filename_col, 
            how='inner'
        )
        
        print(f"   ✓ Merged {len(merged_df)} rows (from {len(df_epoch)} epoch predictions and {len(df_report)} report entries)")
        
        # Enhanced post-merge debugging
        if 'StudyInstanceUID' in merged_df.columns:
            unique_merged_studies = merged_df['StudyInstanceUID'].nunique()
            print(f"   📊 DEBUG: Unique StudyInstanceUIDs in merged data: {unique_merged_studies}")
            
            if unique_merged_studies < unique_report_studies:
                print(f"   ⚠️ WARNING: Lost {unique_report_studies - unique_merged_studies} unique studies in merge!")
                
                # Find which studies were lost
                report_studies = set(df_report['StudyInstanceUID'].dropna())
                merged_studies = set(merged_df['StudyInstanceUID'].dropna())
                lost_studies = report_studies - merged_studies
                
                if lost_studies:
                    print(f"   🔍 DEBUG: First 5 lost StudyInstanceUIDs: {list(lost_studies)[:5]}")
                    
                    # Check if lost studies have corresponding filenames in epoch
                    sample_lost_study = list(lost_studies)[0]
                    lost_study_files = df_report[df_report['StudyInstanceUID'] == sample_lost_study][report_filename_col].tolist()
                    print(f"   🔍 DEBUG: Sample lost study {sample_lost_study} has files: {lost_study_files}")
                    
                    files_in_epoch = [f for f in lost_study_files if f in set(df_epoch[epoch_filename_col])]
                    print(f"   🔍 DEBUG: Of those, {len(files_in_epoch)} are in epoch predictions")
        
        if len(merged_df) == 0:
            print("   ✗ No successful merges between epoch and report data")
            return None, None, None
        
    except Exception as e:
        print(f"   ✗ Error merging epoch and report data: {e}")
        return None, None, None
    
    # 6. Create val_text_map: ground_truth_idx -> report data
    print(f"5. Creating val_text_map from merged data...")
    val_text_map = {}
    
    try:
        for _, row in merged_df.iterrows():
            gt_idx = int(row['ground_truth_idx'])
            
            # FIXED: Accumulate instead of overwrite when multiple studies share same GT index
            row_data = row.drop([epoch_filename_col]).copy()  # Keep report filename column
            if gt_idx not in val_text_map:
                val_text_map[gt_idx] = []
            val_text_map[gt_idx].append(row_data)  # Accumulate all studies with this GT index
        
        print(f"   ✓ Created val_text_map with {len(val_text_map)} ground truth indices")
        
        # Enhanced debugging for the fix
        total_study_entries = sum(len(study_list) for study_list in val_text_map.values())
        duplicate_gt_indices = sum(1 for study_list in val_text_map.values() if len(study_list) > 1)
        
        print(f"   📊 Total study entries in val_text_map: {total_study_entries}")
        print(f"   📊 GT indices with multiple studies: {duplicate_gt_indices}")
        
        # Verify we captured all studies
        studies_in_val_text_map = set()
        for gt_idx, study_list in val_text_map.items():
            for study_data in study_list:
                study_uid = getattr(study_data, 'StudyInstanceUID', None)
                if study_uid:
                    studies_in_val_text_map.add(study_uid)
        
        all_studies_in_merged = set(merged_df['StudyInstanceUID'])
        missing_studies = all_studies_in_merged - studies_in_val_text_map
        
        print(f"   🎯 Studies in merged_df: {len(all_studies_in_merged)}")
        print(f"   🎯 Studies captured in val_text_map: {len(studies_in_val_text_map)}")
        print(f"   🎯 Missing studies after fix: {len(missing_studies)}")
        
        if len(missing_studies) == 0:
            print(f"   ✅ SUCCESS: All {len(all_studies_in_merged)} studies captured!")
        else:
            print(f"   ⚠️ WARNING: Still missing {len(missing_studies)} studies")
        
        # Print sample mappings for verification
        sample_indices = list(val_text_map.keys())[:3]
        for gt_idx in sample_indices:
            study_list = val_text_map[gt_idx]
            study_count = len(study_list)
            if study_count > 1:
                study_uids = [getattr(study_data, 'StudyInstanceUID', 'Unknown') for study_data in study_list]
                print(f"   Sample - GT Index {gt_idx}: {study_count} studies - {study_uids}")
            else:
                study_data = study_list[0]
                main_structure = getattr(study_data, 'main_structure_name', 'Unknown')
                dominance = getattr(study_data, 'dominance_name', 'Unknown')
                print(f"   Sample - GT Index {gt_idx}: {main_structure}, dominance: {dominance}")
        
    except Exception as e:
        print(f"   ✗ Error creating val_text_map: {e}")
        return None, None, None
    
    print(f"\n✓ Successfully loaded and prepared data:")
    print(f"  - Report entries: {len(df_report)}")
    print(f"  - Epoch predictions: {len(df_epoch)}")
    print(f"  - Val text mappings: {len(val_text_map)}")
    
    return df_report, df_epoch, val_text_map

# --- Study-Level Optimize Analysis Export Function ---
def create_optimize_analysis_csv(
    val_text_map,
    predictions_df,
    predictions_base_dir,
    epoch_name,
    vessel_labels=None,
    output_dir=None,
    study_level_aggregation=True
):
    """
    FIXED: Creates a CSV for optimize analysis with proper main structure detection.
    - When study_level_aggregation=True: One row per study/exam with aggregated data
    - When study_level_aggregation=False: One row per ground truth index
    """
    print(f"\n--- Creating Optimize Analysis CSV for {epoch_name} (study_level_aggregation={study_level_aggregation}) ---")
    
    if vessel_labels is None:
        vessel_labels = [
            "left_main_stenosis", "prox_lad_stenosis", "mid_lad_stenosis", "dist_lad_stenosis",
            "D1_stenosis", "D2_stenosis", "D3_stenosis", "lcx_stenosis", "dist_lcx_stenosis",
            "lvp_stenosis", "marg_d_stenosis", "om1_stenosis", "om2_stenosis", "om3_stenosis",
            "prox_rca_stenosis", "mid_rca_stenosis", "dist_rca_stenosis", "RVG1_stenosis",
            "RVG2_stenosis", "pda_stenosis", "posterolateral_stenosis", "bx_stenosis", "lima_or_svg_stenosis"
        ]
    
    lca_vessels = [
        "left_main_stenosis", "prox_lad_stenosis", "mid_lad_stenosis", "dist_lad_stenosis",
        "D1_stenosis", "D2_stenosis", "D3_stenosis", "lcx_stenosis", "dist_lcx_stenosis",
        "lvp_stenosis", "marg_d_stenosis", "om1_stenosis", "om2_stenosis", "om3_stenosis", "bx_stenosis"
    ]
    rca_vessels = [
        "prox_rca_stenosis", "mid_rca_stenosis", "dist_rca_stenosis", "RVG1_stenosis",
        "RVG2_stenosis", "pda_stenosis", "posterolateral_stenosis", "lima_or_svg_stenosis"
    ]
    
    def determine_main_structure_from_gt(gt_idx, val_text_map):
        """
        FIXED: Determine main coronary structure from ground truth data, not predictions.
        Returns: 'left', 'right', or 'unknown'
        """
        if gt_idx not in val_text_map or not val_text_map[gt_idx]:
            return 'unknown'
        
        study_data = val_text_map[gt_idx][0]  # Get first study data row
        
        # Method 1: Check for explicit main_structure columns
        if hasattr(study_data, 'main_structure_name'):
            main_struct = getattr(study_data, 'main_structure_name')
            if main_struct == 'Left Coronary':
                return 'left'
            elif main_struct == 'Right Coronary':
                return 'right'
        
        if hasattr(study_data, 'main_structure_class'):
            main_class = getattr(study_data, 'main_structure_class')
            if main_class == 0:
                return 'left'
            elif main_class == 1:
                return 'right'
        
        # Method 2: Infer from vessel stenosis values
        left_vessel_count = 0
        right_vessel_count = 0
        
        for vessel in lca_vessels:
            if hasattr(study_data, vessel) and pd.notna(getattr(study_data, vessel)):
                stenosis_val = getattr(study_data, vessel)
                if stenosis_val > 0:  # Has stenosis in this left vessel
                    left_vessel_count += 1
        
        for vessel in rca_vessels:
            if hasattr(study_data, vessel) and pd.notna(getattr(study_data, vessel)):
                stenosis_val = getattr(study_data, vessel)
                if stenosis_val > 0:  # Has stenosis in this right vessel
                    right_vessel_count += 1
        
        # Classify based on which side has more stenotic vessels
        if left_vessel_count > right_vessel_count:
            return 'left'
        elif right_vessel_count > left_vessel_count:
            return 'right'
        elif left_vessel_count > 0 and right_vessel_count > 0:
            return 'both'  # Has both sides
        
        return 'unknown'
    
    optimize_rows = []
    
    if study_level_aggregation:
        # Study-level aggregation: one row per study
        study_ids = set()
        study_to_gt_idx_map = {}
        
        for gt_idx, study_data_list in val_text_map.items():
            if study_data_list:
                study_data = study_data_list[0]
                # Try different ways to get study ID
                study_id = None
                if hasattr(study_data, 'StudyInstanceUID'):
                    study_id = getattr(study_data, 'StudyInstanceUID')
                elif hasattr(study_data, 'name') and study_data.name is not None:
                    study_id = study_data.name
                elif hasattr(study_data, 'index'):
                    study_id = f"Study_{getattr(study_data, 'index')}"
                else:
                    study_id = f"Study_{gt_idx}"
                
                if study_id is not None:
                    study_ids.add(study_id)
                    if study_id not in study_to_gt_idx_map:
                        study_to_gt_idx_map[study_id] = []
                    study_to_gt_idx_map[study_id].append(gt_idx)
        
        print(f"Found {len(study_ids)} unique studies")
        
        left_count = 0
        right_count = 0
        both_count = 0
        
        for study_id in study_ids:
            gt_indices = study_to_gt_idx_map[study_id]
            gt_data_combined = {}
            
            # Combine ground truth
            for vessel in vessel_labels:
                vessel_values = []
                for gt_idx in gt_indices:
                    if gt_idx in val_text_map:
                        study_data_list = val_text_map[gt_idx]
                        for study_data in study_data_list:
                            if hasattr(study_data, vessel) and pd.notna(getattr(study_data, vessel)):
                                vessel_values.append(getattr(study_data, vessel))
                if vessel_values:
                    gt_data_combined[f"gt_{vessel}"] = np.mean(vessel_values)
                else:
                    gt_data_combined[f"gt_{vessel}"] = np.nan
            
            # Add IFR data
            for vessel in vessel_labels:
                vessel_prefix = vessel.replace("_stenosis", "")
                ifr_col = f"{vessel_prefix}_IFRHYPER"
                ifr_values = []
                for gt_idx in gt_indices:
                    if gt_idx in val_text_map:
                        study_data_list = val_text_map[gt_idx]
                        for study_data in study_data_list:
                            if hasattr(study_data, ifr_col) and pd.notna(getattr(study_data, ifr_col)):
                                ifr_val = getattr(study_data, ifr_col)
                                if ifr_val > 0.01:
                                    ifr_values.append(ifr_val)
                if ifr_values:
                    gt_data_combined[f"gt_{ifr_col}"] = np.mean(ifr_values)
                else:
                    gt_data_combined[f"gt_{ifr_col}"] = np.nan
            
            # FIXED: Classify videos by main structure using ground truth
            study_predictions = predictions_df[predictions_df['ground_truth_idx'].isin(gt_indices)]
            if study_predictions.empty:
                continue
            
            left_coronary_preds = []
            right_coronary_preds = []
            
            # Classify each prediction based on its ground truth
            for _, pred_row in study_predictions.iterrows():
                pred_gt_idx = pred_row['ground_truth_idx']
                main_structure = determine_main_structure_from_gt(pred_gt_idx, val_text_map)
                
                if main_structure == 'left':
                    left_coronary_preds.append(pred_row)
                elif main_structure == 'right':
                    right_coronary_preds.append(pred_row)
                # Skip 'unknown' or 'both' for now
            
            # Convert back to DataFrames
            left_coronary_preds = pd.DataFrame(left_coronary_preds) if left_coronary_preds else pd.DataFrame()
            right_coronary_preds = pd.DataFrame(right_coronary_preds) if right_coronary_preds else pd.DataFrame()
            
            # Count for summary
            if len(left_coronary_preds) > 0 and len(right_coronary_preds) > 0:
                both_count += 1
            elif len(left_coronary_preds) > 0:
                left_count += 1
            elif len(right_coronary_preds) > 0:
                right_count += 1
            
            pred_data = {}
            
            # Process LCA predictions using top-k predictions columns
            pred_cols = [f"predicted_idx_{k}" for k in range(1, 6)]  # predicted_idx_1 to predicted_idx_5
            existing_pred_cols = [col for col in pred_cols if col in left_coronary_preds.columns]
            
            for vessel in lca_vessels:
                vessel_pred_values = []
                for _, pred_row in left_coronary_preds.iterrows():
                    for pred_col in existing_pred_cols:
                        if pd.notna(pred_row[pred_col]):
                            pred_idx = int(pred_row[pred_col])
                            if pred_idx in val_text_map:
                                pred_data_list = val_text_map[pred_idx]
                                if pred_data_list:
                                    pred_study_data = pred_data_list[0]
                                    if hasattr(pred_study_data, vessel) and pd.notna(getattr(pred_study_data, vessel)):
                                        vessel_pred_values.append(getattr(pred_study_data, vessel))
                
                pred_data[f"pred_left_{vessel}"] = np.mean(vessel_pred_values) if vessel_pred_values else np.nan
            
            # Process RCA predictions 
            for vessel in rca_vessels:
                vessel_pred_values = []
                for _, pred_row in right_coronary_preds.iterrows():
                    for pred_col in existing_pred_cols:
                        if pd.notna(pred_row[pred_col]):
                            pred_idx = int(pred_row[pred_col])
                            if pred_idx in val_text_map:
                                pred_data_list = val_text_map[pred_idx]
                                if pred_data_list:
                                    pred_study_data = pred_data_list[0]
                                    if hasattr(pred_study_data, vessel) and pd.notna(getattr(pred_study_data, vessel)):
                                        vessel_pred_values.append(getattr(pred_study_data, vessel))
                
                pred_data[f"pred_right_{vessel}"] = np.mean(vessel_pred_values) if vessel_pred_values else np.nan
            
            row_data = {
                'StudyInstanceUID': study_id,
                'ground_truth_indices': ';'.join(map(str, sorted(gt_indices))),
                'num_left_videos': len(left_coronary_preds),
                'num_right_videos': len(right_coronary_preds),
                'total_videos': len(study_predictions)
            }
            row_data.update(gt_data_combined)
            row_data.update(pred_data)
            
            optimize_rows.append(row_data)
        
        # Print summary
        print(f"\nOptimize Analysis Summary:")
        print(f"- Total studies: {len(study_ids)}")
        print(f"- Studies with left coronary videos: {left_count}")
        print(f"- Studies with right coronary videos: {right_count}")
        print(f"- Studies with both left and right videos: {both_count}")
    
    else:
        # Individual-level processing (similar fixes apply)
        print(f"Found {len(val_text_map)} ground truth indices")
        for gt_idx, study_data_list in val_text_map.items():
            if not study_data_list:
                continue
            study_data = study_data_list[0]
            study_id = getattr(study_data, 'StudyInstanceUID', f'Unknown_Study_{gt_idx}')
            
            gt_data = {}
            for vessel in vessel_labels:
                if hasattr(study_data, vessel) and pd.notna(getattr(study_data, vessel)):
                    gt_data[f"gt_{vessel}"] = getattr(study_data, vessel)
                else:
                    gt_data[f"gt_{vessel}"] = np.nan
            
            # Add IFR data  
            for vessel in vessel_labels:
                vessel_prefix = vessel.replace("_stenosis", "")
                ifr_col = f"{vessel_prefix}_IFRHYPER"
                if hasattr(study_data, ifr_col) and pd.notna(getattr(study_data, ifr_col)):
                    ifr_val = getattr(study_data, ifr_col)
                    if ifr_val > 0.01:
                        gt_data[f"gt_{ifr_col}"] = ifr_val
                    else:
                        gt_data[f"gt_{ifr_col}"] = np.nan
                else:
                    gt_data[f"gt_{ifr_col}"] = np.nan
            
            gt_predictions = predictions_df[predictions_df['ground_truth_idx'] == gt_idx]
            if gt_predictions.empty:
                continue
            
            # FIXED: Use ground truth-based structure classification
            main_structure = determine_main_structure_from_gt(gt_idx, val_text_map)
            
            if main_structure == 'left':
                left_coronary_preds = gt_predictions
                right_coronary_preds = pd.DataFrame()
            elif main_structure == 'right':
                left_coronary_preds = pd.DataFrame()
                right_coronary_preds = gt_predictions
            else:
                left_coronary_preds = pd.DataFrame()
                right_coronary_preds = pd.DataFrame()
            
            pred_data = {}
            pred_cols = [f"predicted_idx_{k}" for k in range(1, 6)]
            existing_pred_cols = [col for col in pred_cols if col in gt_predictions.columns]
            
            # Process LCA predictions
            for vessel in lca_vessels:
                vessel_pred_values = []
                for _, pred_row in left_coronary_preds.iterrows():
                    for pred_col in existing_pred_cols:
                        if pd.notna(pred_row[pred_col]):
                            pred_idx = int(pred_row[pred_col])
                            if pred_idx in val_text_map:
                                pred_data_list = val_text_map[pred_idx]
                                if pred_data_list:
                                    pred_study_data = pred_data_list[0]
                                    if hasattr(pred_study_data, vessel) and pd.notna(getattr(pred_study_data, vessel)):
                                        vessel_pred_values.append(getattr(pred_study_data, vessel))
                
                pred_data[f"pred_left_{vessel}"] = np.mean(vessel_pred_values) if vessel_pred_values else np.nan
            
            # Process RCA predictions
            for vessel in rca_vessels:
                vessel_pred_values = []
                for _, pred_row in right_coronary_preds.iterrows():
                    for pred_col in existing_pred_cols:
                        if pd.notna(pred_row[pred_col]):
                            pred_idx = int(pred_row[pred_col])
                            if pred_idx in val_text_map:
                                pred_data_list = val_text_map[pred_idx]
                                if pred_data_list:
                                    pred_study_data = pred_data_list[0]
                                    if hasattr(pred_study_data, vessel) and pd.notna(getattr(pred_study_data, vessel)):
                                        vessel_pred_values.append(getattr(pred_study_data, vessel))
                
                pred_data[f"pred_right_{vessel}"] = np.mean(vessel_pred_values) if vessel_pred_values else np.nan
            
            row_data = {
                'StudyInstanceUID': study_id,
                'ground_truth_idx': gt_idx,
                'main_structure': main_structure,
                'num_left_videos': len(left_coronary_preds),
                'num_right_videos': len(right_coronary_preds),
                'total_videos': len(gt_predictions)
            }
            row_data.update(gt_data)
            row_data.update(pred_data)
            
            optimize_rows.append(row_data)
    
    # Save to CSV
    if optimize_rows:
        optimize_df = pd.DataFrame(optimize_rows)
        
        if output_dir is None:
            output_dir = predictions_base_dir
        
        level_suffix = "study_level" if study_level_aggregation else "individual_level"
        output_filename = f"optimize_analysis_{epoch_name.replace('.csv', '')}_{level_suffix}.csv"
        output_path = os.path.join(output_dir, output_filename)
        
        try:
            optimize_df.to_csv(output_path, index=False)
            print(f"Saved optimize analysis to: {output_path}")
            return True
        except Exception as e:
            print(f"Error saving optimize analysis: {e}")
            return False
    else:
        print("No data to save for optimize analysis")
        return False


def run_ground_truth_comparison_analysis(
    report_csv_path,
    epoch_csv_path,
    topk=5,
    vessel_labels=None,
    output_dir=None
):
    """
    NEW: Main function to run ground truth comparison analysis with study-level metrics.
    Loads report and epoch data, compares ground_truth_idx with predicted_idx_1 to predicted_idx_K,
    aggregates based on main_structure_name and dominance, and creates study-level metrics.
    
    Args:
        report_csv_path: Path to the report CSV file
        epoch_csv_path: Path to the epoch prediction CSV file
        topk: Number of top predictions to consider (default: 5)
        vessel_labels: List of vessel labels to analyze (optional)
        output_dir: Directory to save results (optional)
    
    Returns:
        tuple: (aggregated_df, metrics, study_level_df) or (None, None, None) on failure
    """
    print(f"\n🚀 === GROUND TRUTH COMPARISON ANALYSIS WITH STUDY METRICS ===")
    print(f"📋 Configuration:")
    print(f"   📄 Report: {os.path.basename(report_csv_path)}")
    print(f"   📊 Epoch: {os.path.basename(epoch_csv_path)}")
    print(f"   🔢 Top-K: {topk}")
    
    if vessel_labels is None:
        vessel_labels = DEFAULT_VESSEL_LABELS
    
    try:
        # Initialize variables
        study_aggregated_df = pd.DataFrame()
        
        # Step 1: Load and prepare data
        df_report, df_epoch, val_text_map = load_report_and_epoch_data(
            report_csv_path, epoch_csv_path
        )
        
        if df_report is None or df_epoch is None or val_text_map is None:
            print("💥 FATAL: Data loading failed")
            return None, None, None
        
        # Step 2: Aggregate predictions (individual level first, then study level)
        print(f"\n📊 Running individual prediction aggregation...")
        aggregated_df = aggregate_predictions_for_epoch(
            val_text_map=val_text_map,
            predictions_df=df_epoch,
            topk=topk,
            vessel_labels=vessel_labels,
            study_level_aggregation=False,  # Individual predictions first
            filename_to_study_map=None
        )
        
        print(f"\n📊 Running study-level aggregation to capture all studies...")
        study_aggregated_df = aggregate_predictions_for_epoch(
            val_text_map=val_text_map,
            predictions_df=df_epoch,
            topk=topk,
            vessel_labels=vessel_labels,
            study_level_aggregation=True,  # Study-level to capture all studies
            filename_to_study_map=None
        )
        
        if aggregated_df.empty:
            print("💥 FATAL: Individual aggregation failed - empty results")
            return None, None, None
        
        if study_aggregated_df.empty:
            print("💥 FATAL: Study-level aggregation failed - empty results")
            # Fall back to creating study-level metrics from individual predictions
            print("🏥 Falling back to creating study-level metrics from individual predictions...")
            study_level_df = create_study_level_metrics(aggregated_df, vessel_labels)
        else:
            print(f"✅ Study-level aggregation successful: {len(study_aggregated_df)} rows")
            study_level_df = study_aggregated_df
        
        if study_level_df.empty:
            print("⚠️ WARNING: Study-level aggregation resulted in empty dataframe")
        
        # Step 4: Compute metrics (from individual predictions)
        print(f"\n📈 Computing metrics...")
        metrics = compute_metrics(aggregated_df, vessel_labels)
        
        # Step 5: Save results if output directory specified
        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            # Save individual predictions
            epoch_name = os.path.basename(epoch_csv_path).replace('.csv', '')
            agg_output_path = os.path.join(output_dir, f"individual_predictions_{epoch_name}.csv")
            aggregated_df.to_csv(agg_output_path, index=False)
            print(f"   ✅ Saved individual predictions: {agg_output_path}")
            
            # Save study-level metrics
            study_output_path = os.path.join(output_dir, f"study_level_metrics_{epoch_name}.csv")
            study_level_df.to_csv(study_output_path, index=False)
            print(f"   ✅ Saved study-level metrics: {study_output_path}")
            
            # If we have study-aggregated data, save it separately too
            if not study_aggregated_df.empty:
                study_agg_output_path = os.path.join(output_dir, f"study_aggregated_predictions_{epoch_name}.csv")
                study_aggregated_df.to_csv(study_agg_output_path, index=False)
                print(f"   ✅ Saved study-aggregated predictions: {study_agg_output_path}")
            
            # Save metrics summary
            metrics_output_path = os.path.join(output_dir, f"metrics_summary_{epoch_name}.txt")
            with open(metrics_output_path, 'w') as f:
                f.write(f"Ground Truth Comparison Metrics - {epoch_name}\n")
                f.write("=" * 50 + "\n\n")
                
                f.write("STENOSIS METRICS:\n")
                valid_mae = [v for v in metrics['stenosis']['mae'].values() if not pd.isna(v)]
                if valid_mae:
                    f.write(f"Average MAE: {np.mean(valid_mae):.3f} ± {np.std(valid_mae):.3f}\n")
                    f.write(f"Best MAE: {np.min(valid_mae):.3f}\n")
                    f.write(f"Worst MAE: {np.max(valid_mae):.3f}\n\n")
                
                valid_corr = [v for v in metrics['stenosis']['corr'].values() if not pd.isna(v)]
                if valid_corr:
                    f.write(f"Average Correlation: {np.mean(valid_corr):.3f} ± {np.std(valid_corr):.3f}\n")
                    f.write(f"Best Correlation: {np.max(valid_corr):.3f}\n\n")
                
                # Study-level summary
                f.write("STUDY-LEVEL SUMMARY:\n")
                f.write(f"Total studies: {len(study_level_df)}\n")
                if not study_level_df.empty:
                    # Use structure counts from the study_level_df instead
                    if 'left_coronary_predictions' in study_level_df.columns and 'right_coronary_predictions' in study_level_df.columns:
                        left_only_studies = sum((study_level_df['left_coronary_predictions'] > 0) & (study_level_df['right_coronary_predictions'] == 0))
                        right_only_studies = sum((study_level_df['left_coronary_predictions'] == 0) & (study_level_df['right_coronary_predictions'] > 0))
                        both_structures_studies = sum(study_level_df['has_both_structures']) if 'has_both_structures' in study_level_df.columns else 0
                        f.write(f"Studies with left coronary only: {left_only_studies}\n")
                        f.write(f"Studies with right coronary only: {right_only_studies}\n")
                        f.write(f"Studies with both structures: {both_structures_studies}\n")
            
            print(f"   ✅ Saved metrics summary: {metrics_output_path}")
        
        # Step 6: Print summary
        print(f"\n✅ === ANALYSIS COMPLETED ===")
        print(f"   📊 Individual predictions: {len(aggregated_df):,}")
        print(f"   🏥 Study-level metrics: {len(study_level_df):,}")
        
        if not study_aggregated_df.empty:
            print(f"   📊 Study-aggregated data: {len(study_aggregated_df):,}")
            unique_studies_captured = study_aggregated_df['StudyInstanceUID'].nunique() if 'StudyInstanceUID' in study_aggregated_df.columns else 0
            print(f"   🏥 Unique studies captured in study-level: {unique_studies_captured:,}")
        
        print(f"   🎯 Left Coronary cases (individual): {sum(aggregated_df['main_structure_name'] == 'Left Coronary')}")
        print(f"   🎯 Right Coronary cases (individual): {sum(aggregated_df['main_structure_name'] == 'Right Coronary')}")
        
        if not study_aggregated_df.empty:
            print(f"   🎯 Left Coronary cases (study-level): {sum(study_aggregated_df['main_structure_name'] == 'Left Coronary')}")
            print(f"   🎯 Right Coronary cases (study-level): {sum(study_aggregated_df['main_structure_name'] == 'Right Coronary')}")
        
        if len(aggregated_df) > 0:
            top1_acc = aggregated_df['top1_match'].mean()
            top5_acc = aggregated_df['top5_match'].mean()
            print(f"   🏆 Retrieval performance (individual): Top-1: {top1_acc:.3f}, Top-5: {top5_acc:.3f}")
        
        if not study_aggregated_df.empty and len(study_aggregated_df) > 0:
            study_top1_acc = study_aggregated_df['top1_match'].mean()
            study_top5_acc = study_aggregated_df['top5_match'].mean()
            print(f"   🏆 Retrieval performance (study-level): Top-1: {study_top1_acc:.3f}, Top-5: {study_top5_acc:.3f}")
        
        return aggregated_df, metrics, study_level_df
        
    except Exception as e:
        print(f"💥 FATAL ERROR in analysis: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def create_study_level_metrics(aggregated_df, vessel_labels):
    """
    FIXED: Create study-level metrics by aggregating individual predictions.
    Groups by StudyInstanceUID only to ensure we capture ALL studies.
    Combines both left and right coronary data within each study to avoid duplicates.
    
    Args:
        aggregated_df: DataFrame with individual predictions
        vessel_labels: List of vessel labels to aggregate
    
    Returns:
        pd.DataFrame: Study-level aggregated metrics
    """
    print(f"🏥 Creating FIXED study-level metrics from {len(aggregated_df)} individual predictions...")
    
    if aggregated_df.empty:
        print("   ❌ Empty input dataframe")
        return pd.DataFrame()
    
    # DEBUG: Check unique studies in input
    if 'StudyInstanceUID' in aggregated_df.columns:
        unique_input_studies = aggregated_df['StudyInstanceUID'].nunique()
        print(f"   🔍 DEBUG: Unique StudyInstanceUIDs in aggregated_df: {unique_input_studies}")
        
        # Check for missing dominance values
        missing_dominance = aggregated_df['dominance_name'].isna().sum() if 'dominance_name' in aggregated_df.columns else 0
        print(f"   🔍 DEBUG: Rows with missing dominance_name: {missing_dominance}")
        
        # Check dominance value distribution
        if 'dominance_name' in aggregated_df.columns:
            dominance_counts = aggregated_df['dominance_name'].value_counts(dropna=False)
            print(f"   🔍 DEBUG: Dominance distribution: {dominance_counts.to_dict()}")
    
    # Find stenosis columns (both ground truth and predicted)
    stenosis_gt_cols = [col for col in aggregated_df.columns if col.startswith('gt_') and col.endswith('_stenosis')]
    stenosis_pred_cols = [col for col in aggregated_df.columns if col.startswith('pred_') and col.endswith('_stenosis')]
    
    print(f"   📊 Found {len(stenosis_gt_cols)} ground truth stenosis columns")
    print(f"   📊 Found {len(stenosis_pred_cols)} predicted stenosis columns")
    
    if not stenosis_gt_cols or not stenosis_pred_cols:
        print("   ❌ No stenosis columns found for aggregation")
        print(f"   🔍 Available stenosis GT cols: {stenosis_gt_cols}")
        print(f"   🔍 Available stenosis pred cols: {stenosis_pred_cols}")
        return pd.DataFrame()
    
    # FIXED: Group by StudyInstanceUID ONLY to ensure we capture all studies
    if 'StudyInstanceUID' not in aggregated_df.columns:
        print("   ❌ StudyInstanceUID column not found")
        return pd.DataFrame()
    
    print(f"   🔗 FIXED: Grouping by StudyInstanceUID ONLY (to capture all 4881 studies)")
    
    study_level_rows = []
    
    for study_id, group_df in aggregated_df.groupby('StudyInstanceUID'):
        # Get dominance (use first non-null value, or 'Unknown')
        dominance_values = group_df['dominance_name'].dropna() if 'dominance_name' in group_df.columns else []
        dominance = dominance_values.iloc[0] if len(dominance_values) > 0 else 'Unknown'
        
        # Count left vs right coronary predictions within this study
        left_coronary_count = sum(group_df['main_structure_name'] == 'Left Coronary') if 'main_structure_name' in group_df.columns else 0
        right_coronary_count = sum(group_df['main_structure_name'] == 'Right Coronary') if 'main_structure_name' in group_df.columns else 0
        
        # Initialize study-level row
        study_row = {
            'StudyInstanceUID': study_id,
            'dominance_name': dominance,
            'prediction_count': len(group_df),
            'left_coronary_predictions': left_coronary_count,
            'right_coronary_predictions': right_coronary_count,
            'has_both_structures': left_coronary_count > 0 and right_coronary_count > 0,
            'avg_top1_match': group_df['top1_match'].mean() if 'top1_match' in group_df.columns else np.nan,
            'avg_top5_match': group_df['top5_match'].mean() if 'top5_match' in group_df.columns else np.nan,
        }
        
        # Get vessel lists based on dominance (from vessel_constants)
        try:
            # Use already imported constants
            rca_vessels = [vessel for vessel in RCA_VESSELS if f'gt_{vessel}' in stenosis_gt_cols]
            non_rca_vessels = [vessel for vessel in NON_RCA_VESSELS if f'gt_{vessel}' in stenosis_gt_cols]
            
            # Add dominance-dependent vessels
            if dominance == 'right_dominant':
                dominance_vessels = [vessel for vessel in RIGHT_DOMINANCE_DEPENDENT_VESSELS if f'gt_{vessel}' in stenosis_gt_cols]
            else:  # left_dominant or Unknown
                dominance_vessels = [vessel for vessel in LEFT_DOMINANCE_DEPENDENT_VESSELS if f'gt_{vessel}' in stenosis_gt_cols]
            
            # Combine RCA with dominance-dependent vessels
            rca_extended_vessels = rca_vessels + dominance_vessels
            
        except NameError:
            # Fallback if vessel_constants not available
            print("   ⚠️ vessel_constants not available, using basic vessel grouping")
            all_vessels = [col.replace('gt_', '').replace('_stenosis', '') for col in stenosis_gt_cols]
            rca_extended_vessels = [v for v in all_vessels if 'rca' in v.lower() or 'rvg' in v.lower() or 'pda' in v.lower()]
            non_rca_vessels = [v for v in all_vessels if v not in rca_extended_vessels]
        
        # Aggregate stenosis values for each vessel type
        for vessel_type, vessel_list in [
            ('rca', rca_extended_vessels),
            ('non_rca', non_rca_vessels)
        ]:
            gt_values = []
            pred_values = []
            
            for vessel in vessel_list:
                gt_col = f'gt_{vessel}_stenosis'
                pred_col = f'pred_{vessel}_stenosis'
                
                if gt_col in group_df.columns:
                    gt_vals = group_df[gt_col].dropna()
                    if len(gt_vals) > 0:
                        gt_values.extend(gt_vals.tolist())
                
                if pred_col in group_df.columns:
                    pred_vals = group_df[pred_col].dropna()
                    if len(pred_vals) > 0:
                        pred_values.extend(pred_vals.tolist())
            
            # Calculate averages for this vessel type
            if gt_values:
                study_row[f'avg_gt_{vessel_type}_stenosis_overall'] = np.mean(gt_values)
                study_row[f'count_gt_{vessel_type}_stenosis_overall'] = len(gt_values)
            else:
                study_row[f'avg_gt_{vessel_type}_stenosis_overall'] = np.nan
                study_row[f'count_gt_{vessel_type}_stenosis_overall'] = 0
            
            if pred_values:
                study_row[f'avg_pred_{vessel_type}_stenosis_overall'] = np.mean(pred_values)
                study_row[f'count_pred_{vessel_type}_stenosis_overall'] = len(pred_values)
            else:
                study_row[f'avg_pred_{vessel_type}_stenosis_overall'] = np.nan
                study_row[f'count_pred_{vessel_type}_stenosis_overall'] = 0
            
            # Calculate MAE for this vessel type (if we have matching GT and pred values)
            if gt_values and pred_values:
                # For MAE, we need to match GT and predicted values properly
                vessel_mae_values = []
                for vessel in vessel_list:
                    gt_col = f'gt_{vessel}_stenosis'
                    pred_col = f'pred_{vessel}_stenosis'
                    
                    if gt_col in group_df.columns and pred_col in group_df.columns:
                        for _, row in group_df.iterrows():
                            gt_val = row.get(gt_col)
                            pred_val = row.get(pred_col)
                            if pd.notna(gt_val) and pd.notna(pred_val):
                                vessel_mae_values.append(abs(gt_val - pred_val))
                
                study_row[f'mae_{vessel_type}_stenosis_overall'] = np.mean(vessel_mae_values) if vessel_mae_values else np.nan
            else:
                study_row[f'mae_{vessel_type}_stenosis_overall'] = np.nan
        
        # Calculate overall averages across ALL vessels (both RCA and non-RCA)
        all_gt_values = []
        all_pred_values = []
        all_mae_values = []
        
        for gt_col in stenosis_gt_cols:
            gt_vals = group_df[gt_col].dropna()
            if len(gt_vals) > 0:
                all_gt_values.extend(gt_vals.tolist())
            
            pred_col = f'pred_{gt_col.replace("gt_", "")}'
            if pred_col in group_df.columns:
                pred_vals = group_df[pred_col].dropna()
                if len(pred_vals) > 0:
                    all_pred_values.extend(pred_vals.tolist())
                
                # Calculate MAE for each prediction
                for _, row in group_df.iterrows():
                    gt_val = row.get(gt_col)
                    pred_val = row.get(pred_col)
                    if pd.notna(gt_val) and pd.notna(pred_val):
                        all_mae_values.append(abs(gt_val - pred_val))
        
        # Overall averages
        if all_gt_values:
            study_row['avg_gt_stenosis_overall'] = np.mean(all_gt_values)
            study_row['count_gt_stenosis_overall'] = len(all_gt_values)
        else:
            study_row['avg_gt_stenosis_overall'] = np.nan
            study_row['count_gt_stenosis_overall'] = 0
        
        if all_pred_values:
            study_row['avg_pred_stenosis_overall'] = np.mean(all_pred_values)
            study_row['count_pred_stenosis_overall'] = len(all_pred_values)
        else:
            study_row['avg_pred_stenosis_overall'] = np.nan
            study_row['count_pred_stenosis_overall'] = 0
        
        # Overall MAE
        if all_mae_values:
            study_row['mae_stenosis_overall'] = np.mean(all_mae_values)
        else:
            study_row['mae_stenosis_overall'] = np.nan
        
        study_level_rows.append(study_row)
    
    study_level_df = pd.DataFrame(study_level_rows)
    
    print(f"   ✅ FIXED: Created study-level metrics: {len(study_level_df)} unique studies")
    
    # VERIFY we have all expected studies
    if len(study_level_df) != unique_input_studies:
        print(f"   ⚠️ WARNING: Expected {unique_input_studies} studies but got {len(study_level_df)}")
    else:
        print(f"   🎯 SUCCESS: Captured all {unique_input_studies} expected studies!")
    
    if not study_level_df.empty:
        print(f"   📊 Columns: {len(study_level_df.columns)}")
        
        # Summary by structure combinations
        if 'has_both_structures' in study_level_df.columns:
            both_count = sum(study_level_df['has_both_structures'])
            left_only = sum((study_level_df['left_coronary_predictions'] > 0) & (study_level_df['right_coronary_predictions'] == 0))
            right_only = sum((study_level_df['left_coronary_predictions'] == 0) & (study_level_df['right_coronary_predictions'] > 0))
            
            print(f"      🫀 Studies with both left & right: {both_count}")
            print(f"      🫀 Studies with left only: {left_only}")
            print(f"      🫀 Studies with right only: {right_only}")
        
        # Summary by dominance
        if 'dominance_name' in study_level_df.columns:
            dominance_counts = study_level_df['dominance_name'].value_counts()
            for dominance, count in dominance_counts.items():
                print(f"      🔗 {dominance}: {count} studies")
    
    return study_level_df

def plot_results_if_available(all_results):
    """
    Plot results if plotting functions are available, otherwise skip gracefully.
    """
    if not PLOTTING_AVAILABLE:
        print("   📊 Plotting skipped - plot_metrics module not available")
        return False
    
    if not all_results:
        print("   📊 No results to plot")
        return False
    
    try:
        print("   📊 Creating epoch metrics plots...")
        plot_epoch_metrics_line_charts(all_results)
        print("   ✅ Epoch metrics plots created successfully")
        return True
    except Exception as e:
        print(f"   ⚠️ Error creating plots: {e}")
        return False

# Add these optimized functions for multi-epoch processing after the existing functions

def load_report_data_once(report_csv_path):
    """
    Load report data once and optimize it for repeated use across multiple epochs.
    
    Returns:
        pd.DataFrame: Optimized report dataframe or None on failure
    """
    print(f"🚀 Loading report data once for multi-epoch analysis...")
    print(f"📄 Report: {report_csv_path}")
    
    if not os.path.exists(report_csv_path):
        print(f"   ❌ Report file not found: {report_csv_path}")
        return None
    
    # Try different separators and encodings
    load_attempts = [
        {'sep': 'α', 'encoding': 'utf-8', 'engine': 'python'},
        {'sep': ',', 'encoding': 'utf-8'},
        {'sep': ';', 'encoding': 'utf-8'},
        {'sep': '\t', 'encoding': 'utf-8'},
        {'sep': 'α', 'encoding': 'latin-1', 'engine': 'python'},
        {'sep': ',', 'encoding': 'latin-1'}
    ]
    
    df_report = None
    for i, params in enumerate(load_attempts):
        try:
            df_report = pd.read_csv(report_csv_path, on_bad_lines='skip', **params)
            print(f"   ✅ Loaded report with {params['sep']} separator: {len(df_report):,} rows")
            break
        except Exception as e:
            if i == len(load_attempts) - 1:
                print(f"   ❌ Failed to load report with all attempted separators")
                return None
            continue
    
    if df_report is None or len(df_report) == 0:
        print("   ❌ Report is empty after loading")
        return None
    
    # Optimize report for faster lookups
    print("   🔧 Optimizing report data for faster lookups...")
    
    # Find filename column
    filename_col = None
    filename_candidates = ['FileName', 'filename', 'file_name', 'File_Name', 'filepath', 'FilePath']
    for col in df_report.columns:
        for candidate in filename_candidates:
            if candidate.lower() in col.lower():
                filename_col = col
                break
        if filename_col:
            break
    
    if not filename_col:
        print(f"   ❌ FileName column not found in report")
        return None
    
    # Create filename index for faster lookups
    df_report.set_index(filename_col, inplace=True, drop=False)
    
    print(f"   ✅ Report optimized: {len(df_report):,} rows indexed by filename")
    return df_report

def create_persistent_val_text_map(df_report, filename_col='FileName'):
    """
    Create a persistent val_text_map that can be reused across epochs.
    
    Returns:
        dict: Persistent val_text_map for reuse
    """
    print("🗺️ Creating persistent val_text_map for multi-epoch reuse...")
    
    if df_report is None or len(df_report) == 0:
        print("   ❌ Invalid report data")
        return None
    
    # Use the index (filename) to create the mapping
    persistent_map = {}
    
    # Group by filename and create mapping
    print("   🔧 Building filename to report data mapping...")
    for filename, group in df_report.groupby(filename_col):
        # Store all rows for this filename (handle duplicates)
        persistent_map[filename] = [row for _, row in group.iterrows()]
    
    print(f"   ✅ Created persistent mapping: {len(persistent_map):,} unique filenames")
    return persistent_map

def process_single_epoch_optimized(
    epoch_csv_path,
    persistent_report_map,
    topk=5,
    vessel_labels=None
):
    """
    Process a single epoch using pre-loaded report data for maximum speed.
    
    Returns:
        tuple: (aggregated_df, metrics, study_level_df) or (None, None, None) on failure
    """
    epoch_name = os.path.basename(epoch_csv_path).replace('.csv', '')
    print(f"⚡ Processing {epoch_name} with optimized pipeline...")
    
    # Load epoch predictions
    try:
        df_epoch = pd.read_csv(epoch_csv_path)
        print(f"   📊 Loaded epoch predictions: {len(df_epoch):,} rows")
    except Exception as e:
        print(f"   ❌ Failed to load epoch predictions: {e}")
        return None, None, None
    
    if df_epoch.empty:
        print("   ❌ Empty epoch predictions")
        return None, None, None
    
    # Find filename column in epoch data
    epoch_filename_col = None
    filename_candidates = ['FileName', 'filename', 'file_name', 'File_Name', 'filepath', 'FilePath']
    for col in df_epoch.columns:
        for candidate in filename_candidates:
            if candidate.lower() in col.lower():
                epoch_filename_col = col
                break
        if epoch_filename_col:
            break
    
    if not epoch_filename_col:
        print(f"   ❌ FileName column not found in epoch predictions")
        return None, None, None
    
    # Fast merge using persistent map
    print("   🚀 Fast merging with persistent report data...")
    merged_rows = []
    
    for _, epoch_row in df_epoch.iterrows():
        filename = epoch_row[epoch_filename_col]
        if filename in persistent_report_map:
            # Get report data for this filename
            report_data_list = persistent_report_map[filename]
            for report_row in report_data_list:
                # Combine epoch and report data
                merged_row = epoch_row.to_dict()
                merged_row.update(report_row.to_dict() if hasattr(report_row, 'to_dict') else report_row)
                merged_rows.append(merged_row)
    
    if not merged_rows:
        print("   ❌ No successful merges")
        return None, None, None
    
    merged_df = pd.DataFrame(merged_rows)
    print(f"   ✅ Fast merge completed: {len(merged_df):,} rows")
    
    # Create val_text_map for this epoch
    print("   🗺️ Creating val_text_map...")
    val_text_map = {}
    
    for _, row in merged_df.iterrows():
        if 'ground_truth_idx' in row and pd.notna(row['ground_truth_idx']):
            gt_idx = int(row['ground_truth_idx'])
            if gt_idx not in val_text_map:
                val_text_map[gt_idx] = []
            val_text_map[gt_idx].append(row)
    
    print(f"   ✅ Created val_text_map: {len(val_text_map):,} ground truth indices")
    
    # Process aggregations
    if vessel_labels is None:
        vessel_labels = DEFAULT_VESSEL_LABELS
    
    print("   📊 Running individual prediction aggregation...")
    aggregated_df = aggregate_predictions_for_epoch(
        val_text_map=val_text_map,
        predictions_df=df_epoch,
        topk=topk,
        vessel_labels=vessel_labels,
        study_level_aggregation=False
    )
    
    if aggregated_df.empty:
        print("   ❌ Individual aggregation failed")
        return None, None, None
    
    print("   🏥 Creating study-level metrics...")
    study_level_df = create_study_level_metrics(aggregated_df, vessel_labels)
    
    print("   📈 Computing metrics...")
    metrics = compute_metrics(aggregated_df, vessel_labels)
    
    print(f"   ✅ {epoch_name} completed: {len(aggregated_df):,} individual, {len(study_level_df):,} studies")
    
    return aggregated_df, metrics, study_level_df

def run_multi_epoch_analysis_optimized(
    report_csv_path,
    epoch_files,
    topk=5,
    vessel_labels=None,
    output_dir=None,
    max_epochs=None
):
    """
    Optimized multi-epoch analysis that loads report data once and reuses it.
    
    Args:
        report_csv_path: Path to report CSV
        epoch_files: List of epoch CSV file paths
        topk: Number of top predictions to consider
        vessel_labels: List of vessel labels
        output_dir: Output directory for results
        max_epochs: Maximum number of epochs to process (for testing)
        
    Returns:
        tuple: (all_epoch_results, all_epoch_metrics)
    """
    print("🚀 === OPTIMIZED MULTI-EPOCH ANALYSIS ===")
    print(f"📊 Processing {len(epoch_files)} epochs with optimized pipeline")
    
    if max_epochs:
        epoch_files = epoch_files[:max_epochs]
        print(f"🔢 Limited to first {max_epochs} epochs for testing")
    
    # Step 1: Load report data once
    print("\n1️⃣ Loading report data once...")
    persistent_report = load_report_data_once(report_csv_path)
    if persistent_report is None:
        print("❌ Failed to load report data")
        return {}, {}
    
    # Step 2: Create persistent mapping
    print("\n2️⃣ Creating persistent filename mapping...")
    filename_col = persistent_report.columns[0]  # First column should be filename
    for col in persistent_report.columns:
        if 'filename' in col.lower() or 'filepath' in col.lower():
            filename_col = col
            break
    
    persistent_map = create_persistent_val_text_map(persistent_report, filename_col)
    if persistent_map is None:
        print("❌ Failed to create persistent mapping")
        return {}, {}
    
    # Step 3: Process epochs efficiently
    print(f"\n3️⃣ Processing {len(epoch_files)} epochs...")
    all_epoch_results = {}
    all_epoch_metrics = {}
    
    start_time = pd.Timestamp.now()
    
    for i, epoch_file in enumerate(epoch_files):
        epoch_name = os.path.basename(epoch_file).replace('.csv', '')
        
        epoch_start = pd.Timestamp.now()
        print(f"\n📊 [{i+1}/{len(epoch_files)}] Processing {epoch_name}...")
        
        try:
            aggregated_df, metrics, study_level_df = process_single_epoch_optimized(
                epoch_csv_path=epoch_file,
                persistent_report_map=persistent_map,
                topk=topk,
                vessel_labels=vessel_labels
            )
            
            if aggregated_df is not None and metrics is not None:
                all_epoch_results[epoch_name] = aggregated_df
                all_epoch_metrics[epoch_name] = metrics
                
                # Quick summary
                if len(aggregated_df) > 0:
                    top1_acc = aggregated_df['top1_match'].mean()
                    top5_acc = aggregated_df['top5_match'].mean()
                    left_count = sum(aggregated_df['main_structure_name'] == 'Left Coronary')
                    right_count = sum(aggregated_df['main_structure_name'] == 'Right Coronary')
                    
                    epoch_duration = pd.Timestamp.now() - epoch_start
                    print(f"   ✅ {epoch_name}: Top-1={top1_acc:.3f}, Top-5={top5_acc:.3f}")
                    print(f"   🫀 Left={left_count}, Right={right_count}")
                    print(f"   ⏱️ Duration: {epoch_duration.total_seconds():.1f}s")
                
                # Save results if output directory specified
                if output_dir:
                    epoch_output_dir = os.path.join(output_dir, epoch_name)
                    os.makedirs(epoch_output_dir, exist_ok=True)
                    
                    # Save key results
                    agg_path = os.path.join(epoch_output_dir, f"individual_predictions_{epoch_name}.csv")
                    aggregated_df.to_csv(agg_path, index=False)
                    
                    if study_level_df is not None and not study_level_df.empty:
                        study_path = os.path.join(epoch_output_dir, f"study_level_metrics_{epoch_name}.csv")
                        study_level_df.to_csv(study_path, index=False)
            else:
                print(f"   ❌ Failed to process {epoch_name}")
                
        except Exception as e:
            print(f"   💥 Error processing {epoch_name}: {e}")
            continue
    
    total_duration = pd.Timestamp.now() - start_time
    print(f"\n🎉 OPTIMIZED MULTI-EPOCH ANALYSIS COMPLETED!")
    print(f"   📊 Successfully processed: {len(all_epoch_results)}/{len(epoch_files)} epochs")
    print(f"   ⏱️ Total duration: {total_duration.total_seconds():.1f}s")
    print(f"   ⚡ Average per epoch: {total_duration.total_seconds()/max(1, len(all_epoch_results)):.1f}s")
    
    return all_epoch_results, all_epoch_metrics

if __name__ == '__main__':
    print("Running data_aggregation.py as a standalone script...")
    
    # Example: New Ground Truth Comparison Analysis with FIXED study-level aggregation
    REPORT_CSV_PATH = "data/reports/reports_with_alpha_separator_with_Calcifc_Stenosis_IFR_20250601_RCA_LCA_merged_with_left_dominance_dependent_vessels.csv"
    EPOCH_CSV_PATH = "outputs/DeepCORO_clip/dev_deep_coro_clip_single_video/8av1xygm_20250605-083820_best_single_video/val_epoch0.csv"
    OUTPUT_DIR = "ground_truth_comparison_results"
    
    print("\n🔧 === FIXED GROUND TRUTH COMPARISON ANALYSIS ===")
    print("🎯 This version fixes:")
    print("   - Duplicate StudyInstanceUIDs in study-level metrics")
    print("   - Missing studies in merge process")
    print("   - Proper aggregation of both left and right coronary data per study")
    
    aggregated_df, metrics, study_level_df = run_ground_truth_comparison_analysis(
        report_csv_path=REPORT_CSV_PATH,
        epoch_csv_path=EPOCH_CSV_PATH,
        topk=5,
        output_dir=OUTPUT_DIR
    )
    
    if aggregated_df is not None and metrics is not None and study_level_df is not None:
        print("\n✅ FIXED ground truth comparison analysis completed successfully!")
        
        # Show duplicate check for StudyInstanceUID
        if 'StudyInstanceUID' in study_level_df.columns:
            duplicate_studies = study_level_df['StudyInstanceUID'].duplicated().sum()
            unique_studies_count = study_level_df['StudyInstanceUID'].nunique()
            total_rows = len(study_level_df)
            
            print(f"\n🔍 === DUPLICATE CHECK RESULTS ===")
            print(f"   📊 Total study-level rows: {total_rows}")
            print(f"   📊 Unique StudyInstanceUIDs: {unique_studies_count}")
            print(f"   📊 Duplicate StudyInstanceUIDs: {duplicate_studies}")
            
            if duplicate_studies == 0:
                print("   ✅ SUCCESS: No duplicate StudyInstanceUIDs found!")
            else:
                print("   ❌ ISSUE: Still have duplicate StudyInstanceUIDs")
        
        # Show structure combination summary
        if 'has_both_structures' in study_level_df.columns:
            both_structures = sum(study_level_df['has_both_structures'])
            left_only = sum((study_level_df['left_coronary_predictions'] > 0) & (study_level_df['right_coronary_predictions'] == 0))
            right_only = sum((study_level_df['left_coronary_predictions'] == 0) & (study_level_df['right_coronary_predictions'] > 0))
            
            print(f"\n🫀 === STRUCTURE COMBINATION SUMMARY ===")
            print(f"   📊 Studies with both left & right coronary: {both_structures}")
            print(f"   📊 Studies with left coronary only: {left_only}")
            print(f"   📊 Studies with right coronary only: {right_only}")
            print(f"   📊 Total studies: {both_structures + left_only + right_only}")
        
        # Show some sample results for verification
        print(f"\n📋 Sample individual results (first 3 rows):")
        sample_cols = ['ground_truth_idx', 'main_structure_name', 'dominance_name', 
                      'top1_match', 'top5_match', 'target_vessel_count']
        available_cols = [col for col in sample_cols if col in aggregated_df.columns]
        print(aggregated_df[available_cols].head(3).to_string(index=False))
        
        print(f"\n🏥 Sample study-level results (first 3 rows):")
        study_cols = ['StudyInstanceUID', 'dominance_name', 'left_coronary_predictions', 'right_coronary_predictions',
                     'has_both_structures', 'avg_gt_stenosis_overall', 'avg_pred_stenosis_overall']
        available_study_cols = [col for col in study_cols if col in study_level_df.columns]
        print(study_level_df[available_study_cols].head(3).to_string(index=False))
        
        # Calculate final metrics summary
        if len(aggregated_df) > 0:
            top1_acc = aggregated_df['top1_match'].mean()
            top5_acc = aggregated_df['top5_match'].mean()
            print(f"\n🏆 === PERFORMANCE SUMMARY ===")
            print(f"   📊 Individual predictions: {len(aggregated_df):,}")
            print(f"   🏥 Unique studies: {len(study_level_df):,}")
            print(f"   🎯 Retrieval performance: Top-1: {top1_acc:.3f}, Top-5: {top5_acc:.3f}")
            print(f"   🎯 Left Coronary cases: {sum(aggregated_df['main_structure_name'] == 'Left Coronary')}")
            print(f"   🎯 Right Coronary cases: {sum(aggregated_df['main_structure_name'] == 'Right Coronary')}")
            
            # Summary from study-level data
            if not study_level_df.empty and 'left_coronary_predictions' in study_level_df.columns:
                left_only = sum((study_level_df['left_coronary_predictions'] > 0) & (study_level_df['right_coronary_predictions'] == 0))
                right_only = sum((study_level_df['left_coronary_predictions'] == 0) & (study_level_df['right_coronary_predictions'] > 0))
                both_structures = sum(study_level_df['has_both_structures']) if 'has_both_structures' in study_level_df.columns else 0
                print(f"   🏥 Studies with left only: {left_only}")
                print(f"   🏥 Studies with right only: {right_only}")
                print(f"   🏥 Studies with both structures: {both_structures}")
        
        # Try to plot results if plotting is available
        try:
            from utils.plot_metrics import plot_ground_truth_comparison_results
            plot_ground_truth_comparison_results(aggregated_df, "val_epoch0_FIXED")
            print("✅ Plotting completed successfully!")
        except ImportError:
            print("⚠️ Plotting not available - continuing without plots")
    else:
        print("❌ FIXED ground truth comparison analysis failed")
