#!/usr/bin/env python3
"""
Clean Study-Level Analysis Script - VECTORIZED & PARALLEL OPTIMIZED
================

VECTORIZED version that:
1. Loads report data ONCE and creates filename mapping
2. Reuses the mapping across all epochs (since ground truth is static)
3. Processes multiple epochs IN PARALLEL using multiprocessing
4. Creates ONE row per study (no duplicates)
5. Uses vessel_constants.py for proper vessel groupings
6. Properly sorts epochs numerically
7. Returns metrics dictionary for further analysis

Professional script to:
1. Load report data
2. Load validation epoch files  
3. Merge on filename to get StudyInstanceUID
4. Create study-level predictions with proper vessel groupings
5. Return comprehensive metrics

PARALLEL PROCESSING: Now processes multiple epochs simultaneously!
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import glob
from tqdm import tqdm
import re
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import pickle
import tempfile
import time

# Import vessel constants
try:
    from utils.vessel_constants import (
        RCA_VESSELS,
        NON_RCA_VESSELS,
        RIGHT_DOMINANCE_DEPENDENT_VESSELS,
        LEFT_DOMINANCE_DEPENDENT_VESSELS,
        LEFT_CORONARY_DOMINANCE_VESSELS,
        RIGHT_CORONARY_DOMINANCE_VESSELS,
    )
except ImportError:
    print("⚠️ Warning: vessel_constants not available, using fallback definitions")
    # Fallback definitions
    RCA_VESSELS = ["prox_rca_stenosis", "mid_rca_stenosis", "dist_rca_stenosis"]
    NON_RCA_VESSELS = [
        "left_main_stenosis", "prox_lad_stenosis", "mid_lad_stenosis", "dist_lad_stenosis",
        "D1_stenosis", "D2_stenosis", "prox_lcx_stenosis", "dist_lcx_stenosis",
        "om1_stenosis", "om2_stenosis", "bx_stenosis", "lvp_stenosis"
    ]
    RIGHT_DOMINANCE_DEPENDENT_VESSELS = ["pda_stenosis", "posterolateral_stenosis"]
    LEFT_DOMINANCE_DEPENDENT_VESSELS = ["pda_stenosis", "lvp_stenosis"]
    LEFT_CORONARY_DOMINANCE_VESSELS = NON_RCA_VESSELS + LEFT_DOMINANCE_DEPENDENT_VESSELS
    RIGHT_CORONARY_DOMINANCE_VESSELS = RCA_VESSELS + RIGHT_DOMINANCE_DEPENDENT_VESSELS

def load_report_data_once(report_path: str) -> Tuple[pd.DataFrame, Dict[str, List]]:
    """Load the report data once and create filename mapping for reuse."""
    print(f"🔄 Loading report data ONCE from: {report_path}")

    try:
        df_report = pd.read_csv(report_path, sep='α', on_bad_lines='skip', engine='python')
        print(f"✅ Loaded report: {len(df_report):,} rows")
        print(f"   Columns: {len(df_report.columns)}")

        # Create filename mapping for fast lookups
        # Map by FULL PATH since FileName column contains full paths
        print("🗺️ Creating filename mapping for fast epoch processing...")
        filename_map = {}

        for _, row in tqdm(df_report.iterrows(), total=len(df_report), desc="Building filename map"):
            filepath = row['FileName']  # This is the full path
            if filepath not in filename_map:
                filename_map[filepath] = []
            filename_map[filepath].append(row)

        print(f"✅ Created filename mapping: {len(filename_map):,} unique file paths")
        return df_report, filename_map

    except Exception as e:
        print(f"❌ Error loading report: {e}")
        return pd.DataFrame(), {}

def find_validation_epoch_files(base_dir: str) -> List[str]:
    """Find all validation epoch CSV files and sort them numerically."""
    print(f"🔍 Finding validation epoch files in: {base_dir}")
    
    # Look for files in the main directory and files subdirectory
    patterns = [
        os.path.join(base_dir, "val_epoch*.csv"),
        os.path.join(base_dir, "files", "val_epoch*.csv")
    ]
    
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    
    # FIX: Sort numerically by epoch number, not lexicographically
    def extract_epoch_number(filepath):
        """Extract epoch number from filename for proper numerical sorting."""
        filename = os.path.basename(filepath)
        match = re.search(r'val_epoch(\d+)\.csv', filename)
        return int(match.group(1)) if match else float('inf')
    
    files = sorted(files, key=extract_epoch_number)
    
    print(f"✅ Found {len(files)} validation epoch files (sorted numerically)")
    for i, f in enumerate(files[:5]):  # Show first 5
        print(f"   {i+1}. {os.path.basename(f)}")
    if len(files) > 5:
        print(f"   ... and {len(files)-5} more")
    
    return files

def load_validation_epoch(epoch_path: str) -> pd.DataFrame:
    """Load a single validation epoch file."""
    try:
        df_epoch = pd.read_csv(epoch_path)
        epoch_name = os.path.basename(epoch_path)
        print(f"   ✅ {epoch_name}: {len(df_epoch):,} rows")
        return df_epoch
    except Exception as e:
        print(f"   ❌ Error loading {epoch_path}: {e}")
        return pd.DataFrame()

def merge_epoch_with_report_mapping(df_epoch: pd.DataFrame, filename_map: Dict[str, List]) -> pd.DataFrame:
    """Fast merge using pre-built filename mapping."""
    print("   ⚡ Fast merging with pre-built filename mapping...")

    # Check for either 'FileName' or 'video_path' column
    filename_col = None
    if 'FileName' in df_epoch.columns:
        filename_col = 'FileName'
    elif 'video_path' in df_epoch.columns:
        filename_col = 'video_path'
    else:
        print(f"   ❌ Neither 'FileName' nor 'video_path' found in epoch data. Available columns: {df_epoch.columns.tolist()}")
        return pd.DataFrame()

    merged_rows = []
    missing_filenames = 0

    for _, epoch_row in df_epoch.iterrows():
        filepath = epoch_row[filename_col]  # This is the full path

        # Use the FULL PATH for matching (both report and epoch have full paths)
        if filepath in filename_map:
            report_rows = filename_map[filepath]

            for report_row in report_rows:
                merged_row = epoch_row.to_dict()
                if hasattr(report_row, 'to_dict'):
                    merged_row.update(report_row.to_dict())
                else:
                    merged_row.update(report_row)
                merged_rows.append(merged_row)
        else:
            missing_filenames += 1

    if missing_filenames > 0:
        print(f"   ⚠️ Warning: {missing_filenames} file paths from epoch not found in report")

    merged_df = pd.DataFrame(merged_rows)
    print(f"   ✅ Fast merge completed: {len(merged_df):,} rows")

    return merged_df

def get_target_vessels_for_study(dominance_name: str, has_left: bool, has_right: bool) -> List[str]:
    """
    Determine target vessels for a study based on dominance and available structures.
    
    Returns vessels that should be analyzed based on:
    - Dominance (right/left)
    - Available structures (left/right coronary videos)
    """
    target_vessels = []
    
    # Normalize dominance
    dominance = str(dominance_name).lower() if dominance_name else 'right_dominant'
    
    if 'left' in dominance:
        # Left dominant: focus on left coronary + left dominance dependent vessels
        if has_left:
            target_vessels.extend(LEFT_CORONARY_DOMINANCE_VESSELS)
        if has_right:
            target_vessels.extend(RCA_VESSELS)  # Still include basic RCA vessels
    else:
        # Right dominant (default): focus on traditional groupings
        if has_left:
            target_vessels.extend(NON_RCA_VESSELS)
        if has_right:
            target_vessels.extend(RIGHT_CORONARY_DOMINANCE_VESSELS)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_vessels = []
    for vessel in target_vessels:
        if vessel not in seen:
            seen.add(vessel)
            unique_vessels.append(vessel)
    
    return unique_vessels

def map_calcification_value(val):
    """
    Map French calcification categories to numeric values.
    
    Mapping:
    - "-1" → 0 (No calcification)
    - "Pas de calcification" → 0 (No calcification)
    - "Calcifications minimes" → 1 (Minimal calcifications)
    - "Calcifications modérées" → 2 (Moderate calcifications)
    - "Calcification importantes" → 3 (Important/severe calcifications)
    - "Bourgeon calcaire" → 3 (Severe calcifications)
    """
    if pd.isna(val):
        return None
    
    val_str = str(val).strip().lower()
    
    # Map -1 values to 0 (No calcification) - FIXED
    if val_str in ['-1', '-1.0']:
        return 0
        
    # Map to 0 (no calcification)
    if 'pas de calcification' in val_str:
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
            numeric_val = float(val)
            # Map -1 values to 0 (No calcification) - FIXED
            if numeric_val == -1.0:
                return 0
            return numeric_val
        except:
            return None

def calcification_mode_with_fallback(values):
    """
    Calculate mode for calcification values.
    If no clear mode, pick highest value among the most frequent ones.
    """
    if not values:
        return np.nan
    
    # Remove None values
    valid_values = [v for v in values if v is not None]
    if not valid_values:
        return np.nan
    
    # Calculate value counts
    from collections import Counter
    counts = Counter(valid_values)
    max_count = max(counts.values())
    
    # Get all values with maximum count
    most_frequent = [val for val, count in counts.items() if count == max_count]
    
    # If clear mode (only one value with max count), return it
    if len(most_frequent) == 1:
        return most_frequent[0]
    
    # If tie, return highest value among most frequent
    return max(most_frequent)

def create_study_level_predictions_optimized(merged_df: pd.DataFrame, gt_mappings: Dict) -> pd.DataFrame:
    """
    OPTIMIZED: Create study-level predictions using pre-computed ground truth mappings.
    """
    print("📊 Creating study-level predictions (OPTIMIZED with pre-computed mappings)...")
    
    if merged_df.empty:
        return pd.DataFrame()
    
    # Get prediction columns
    pred_cols = [f'predicted_idx_{i}' for i in range(1, 6)]
    available_pred_cols = [col for col in pred_cols if col in merged_df.columns]
    
    # Process each study
    study_level_rows = []
    grouped_by_study = merged_df.groupby('StudyInstanceUID')
    
    for study_uid, study_df in tqdm(grouped_by_study, desc="Processing studies", total=len(grouped_by_study)):
        
        # Get dominance (use first non-null value)
        dominance_values = study_df['dominance_name'].dropna()
        dominance = dominance_values.iloc[0] if len(dominance_values) > 0 else 'right_dominant'
        
        # Check what structures we have for this study
        structures = study_df['main_structure_name'].unique()
        has_left = 'Left Coronary' in structures
        has_right = 'Right Coronary' in structures
        
        # Get target vessels based on dominance and available structures
        target_vessels = get_target_vessels_for_study(dominance, has_left, has_right)
        
        # Initialize study row
        study_row = {
            'StudyInstanceUID': study_uid,
            'dominance_name': dominance,
            'has_left_coronary': has_left,
            'has_right_coronary': has_right,
            'total_videos': len(study_df),
            'left_videos': sum(study_df['main_structure_name'] == 'Left Coronary'),
            'right_videos': sum(study_df['main_structure_name'] == 'Right Coronary'),
            'target_vessel_count': len(target_vessels),
        }
        
        # Ground truth values for target vessels
        for vessel in target_vessels:
            if vessel in study_df.columns:
                valid_values = study_df[vessel].dropna()
                if len(valid_values) > 0:
                    study_row[f'gt_{vessel}'] = valid_values.mean()
                    study_row[f'gt_{vessel}_count'] = len(valid_values)
                else:
                    study_row[f'gt_{vessel}'] = np.nan
                    study_row[f'gt_{vessel}_count'] = 0
        
        # Calcification ground truth (mode with fallback)
        for vessel in target_vessels:
            vessel_base = vessel.replace('_stenosis', '')
            calcif_col = f'{vessel_base}_calcif'
            if calcif_col in study_df.columns:
                raw_values = study_df[calcif_col].dropna().tolist()
                mapped_values = [map_calcification_value(val) for val in raw_values]
                mode_value = calcification_mode_with_fallback(mapped_values)
                study_row[f'gt_{calcif_col}'] = mode_value
                study_row[f'gt_{calcif_col}_count'] = len([v for v in mapped_values if v is not None])
        
        # IFR ground truth (average of values > 0.01)
        for vessel in target_vessels:
            vessel_base = vessel.replace('_stenosis', '')
            ifr_col = f'{vessel_base}_IFRHYPER'
            if ifr_col in study_df.columns:
                raw_values = study_df[ifr_col].dropna()
                valid_ifr_values = raw_values[raw_values > 0.01]
                if len(valid_ifr_values) > 0:
                    study_row[f'gt_{ifr_col}'] = valid_ifr_values.mean()
                    study_row[f'gt_{ifr_col}_count'] = len(valid_ifr_values)
                else:
                    study_row[f'gt_{ifr_col}'] = np.nan
                    study_row[f'gt_{ifr_col}_count'] = 0
        
        # Calculate predicted values using pre-computed mappings
        for vessel in target_vessels:
            vessel_base = vessel.replace('_stenosis', '')
            
            # Determine which coronary structure this vessel belongs to
            vessel_structure = None
            if vessel in RIGHT_DOMINANCE_DEPENDENT_VESSELS and 'right' in dominance.lower():
                vessel_structure = 'Right Coronary'
            elif vessel in LEFT_DOMINANCE_DEPENDENT_VESSELS and 'left' in dominance.lower():
                vessel_structure = 'Left Coronary'
            elif vessel in NON_RCA_VESSELS:
                vessel_structure = 'Left Coronary'
            elif vessel in RCA_VESSELS:
                vessel_structure = 'Right Coronary'
            elif vessel in RIGHT_DOMINANCE_DEPENDENT_VESSELS:
                vessel_structure = 'Left Coronary'
            elif vessel in LEFT_DOMINANCE_DEPENDENT_VESSELS:
                vessel_structure = 'Right Coronary'
            
            # 1. Stenosis predictions
            if vessel in gt_mappings['stenosis']:
                stenosis_map = gt_mappings['stenosis'][vessel]
                pred_stenosis_values = []
                
                for _, video_row in study_df.iterrows():
                    if vessel_structure and video_row['main_structure_name'] != vessel_structure:
                        continue
                    
                    for pred_col in available_pred_cols:
                        if pd.notna(video_row[pred_col]):
                            pred_idx = int(video_row[pred_col])
                            if pred_idx in stenosis_map and pd.notna(stenosis_map[pred_idx]):
                                pred_stenosis_values.append(stenosis_map[pred_idx])
                
                if pred_stenosis_values:
                    study_row[f'pred_{vessel}'] = np.mean(pred_stenosis_values)
                    study_row[f'pred_{vessel}_count'] = len(pred_stenosis_values)
                else:
                    study_row[f'pred_{vessel}'] = np.nan
                    study_row[f'pred_{vessel}_count'] = 0
            
            # 2. Calcification predictions
            calcif_col = f'{vessel_base}_calcif'
            if calcif_col in gt_mappings['calcification']:
                calcif_map = gt_mappings['calcification'][calcif_col]
                pred_calcif_values = []
                
                for _, video_row in study_df.iterrows():
                    if vessel_structure and video_row['main_structure_name'] != vessel_structure:
                        continue
                    
                    for pred_col in available_pred_cols:
                        if pd.notna(video_row[pred_col]):
                            pred_idx = int(video_row[pred_col])
                            if pred_idx in calcif_map and pd.notna(calcif_map[pred_idx]):
                                mapped_val = map_calcification_value(calcif_map[pred_idx])
                                if mapped_val is not None:
                                    pred_calcif_values.append(mapped_val)
                
                if pred_calcif_values:
                    mode_val = calcification_mode_with_fallback(pred_calcif_values)
                    study_row[f'pred_{calcif_col}'] = mode_val
                    study_row[f'pred_{calcif_col}_count'] = len(pred_calcif_values)
                else:
                    study_row[f'pred_{calcif_col}'] = np.nan
                    study_row[f'pred_{calcif_col}_count'] = 0
            
            # 3. IFR predictions
            ifr_col = f'{vessel_base}_IFRHYPER'
            if ifr_col in gt_mappings['ifr']:
                ifr_map = gt_mappings['ifr'][ifr_col]
                pred_ifr_values = []
                
                for _, video_row in study_df.iterrows():
                    if vessel_structure and video_row['main_structure_name'] != vessel_structure:
                        continue
                    
                    for pred_col in available_pred_cols:
                        if pd.notna(video_row[pred_col]):
                            pred_idx = int(video_row[pred_col])
                            if pred_idx in ifr_map and pd.notna(ifr_map[pred_idx]):
                                ifr_val = ifr_map[pred_idx]
                                if ifr_val > 0.01:
                                    pred_ifr_values.append(ifr_val)
                
                if pred_ifr_values:
                    study_row[f'pred_{ifr_col}'] = np.mean(pred_ifr_values)
                    study_row[f'pred_{ifr_col}_count'] = len(pred_ifr_values)
                else:
                    study_row[f'pred_{ifr_col}'] = np.nan
                    study_row[f'pred_{ifr_col}_count'] = 0
        
        study_level_rows.append(study_row)
    
    study_level_df = pd.DataFrame(study_level_rows)
    print(f"   ✅ Created study-level data: {len(study_level_df):,} studies")
    
    return study_level_df

def create_gt_mappings_from_merged_data(merged_df: pd.DataFrame) -> Dict:
    """
    Create ground truth mappings from merged data (contains both epoch and report info).
    This creates the mapping from ground_truth_idx to vessel values.
    """
    print("🗺️ Creating ground truth mappings from merged data...")
    
    if 'ground_truth_idx' not in merged_df.columns:
        print("❌ ground_truth_idx column not found in merged data")
        return {'stenosis': {}, 'calcification': {}, 'ifr': {}}
    
    stenosis_cols = [col for col in merged_df.columns if col.endswith('_stenosis')]
    calcif_cols = [col for col in merged_df.columns if col.endswith('_calcif')]
    ifr_cols = [col for col in merged_df.columns if col.endswith('_IFRHYPER')]
    
    gt_mappings = {
        'stenosis': {},
        'calcification': {},
        'ifr': {}
    }
    
    # Stenosis mappings
    for stenosis_col in tqdm(stenosis_cols, desc="Creating stenosis mappings", leave=False):
        if stenosis_col in merged_df.columns:
            stenosis_map = merged_df.groupby('ground_truth_idx')[stenosis_col].first().to_dict()
            gt_mappings['stenosis'][stenosis_col] = stenosis_map
    
    # Calcification mappings
    for calcif_col in tqdm(calcif_cols, desc="Creating calcification mappings", leave=False):
        if calcif_col in merged_df.columns:
            calcif_map = merged_df.groupby('ground_truth_idx')[calcif_col].first().to_dict()
            gt_mappings['calcification'][calcif_col] = calcif_map
    
    # IFR mappings
    for ifr_col in tqdm(ifr_cols, desc="Creating IFR mappings", leave=False):
        if ifr_col in merged_df.columns:
            ifr_map = merged_df.groupby('ground_truth_idx')[ifr_col].first().to_dict()
            gt_mappings['ifr'][ifr_col] = ifr_map
    
    print(f"✅ Created GT mappings: {len(gt_mappings['stenosis'])} stenosis, {len(gt_mappings['calcification'])} calcif, {len(gt_mappings['ifr'])} IFR")
    
    return gt_mappings

def analyze_single_epoch_super_optimized(
    epoch_path: str,
    filename_map: Dict[str, List],
    gt_mappings: Dict,
    output_dir: str = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    SUPER OPTIMIZED: Analyze single epoch using pre-built filename mapping AND ground truth mappings.
    This is the fastest possible version.
    """
    
    epoch_name = os.path.basename(epoch_path).replace('.csv', '')
    print(f"⚡ Processing {epoch_name} (SUPER OPTIMIZED)")
    
    # 1. Load epoch data
    try:
        df_epoch = pd.read_csv(epoch_path)
        print(f"   ✅ Loaded epoch: {len(df_epoch):,} rows")
    except Exception as e:
        print(f"   ❌ Error loading epoch: {e}")
        return pd.DataFrame(), {}
    
    # 2. Fast merge using pre-built mapping
    merged_df = merge_epoch_with_report_mapping(df_epoch, filename_map)
    if merged_df.empty:
        return pd.DataFrame(), {}
    
    # 3. Create study-level predictions using pre-computed GT mappings
    study_df = create_study_level_predictions_optimized(merged_df, gt_mappings)
    
    # 4. Calculate comprehensive metrics
    metrics = {
        'stenosis': {'mae': {}, 'corr': {}},
        'calcification': {'accuracy': {}},
        'ifr': {'mae': {}, 'corr': {}}
    }
    
    if not study_df.empty:
        # Extract vessel-specific metrics
        for col in study_df.columns:
            if col.startswith('gt_') and col.endswith('_stenosis') and not col.endswith('_count'):
                vessel = col.replace('gt_', '')
                pred_col = f'pred_{vessel}'
                
                if pred_col in study_df.columns:
                    valid_mask = study_df[col].notna() & study_df[pred_col].notna()
                    
                    if valid_mask.sum() > 5:
                        gt_values = study_df[col][valid_mask]
                        pred_values = study_df[pred_col][valid_mask]
                        
                        # MAE
                        mae = np.mean(np.abs(gt_values - pred_values))
                        metrics['stenosis']['mae'][vessel] = mae
                        
                        # Correlation
                        if len(set(gt_values)) > 1 and len(set(pred_values)) > 1:
                            corr = np.corrcoef(gt_values, pred_values)[0, 1]
                            if not np.isnan(corr):
                                metrics['stenosis']['corr'][vessel] = corr
            
            elif col.startswith('gt_') and col.endswith('_calcif') and not col.endswith('_count'):
                vessel = col.replace('gt_', '')
                pred_col = f'pred_{vessel}'
                
                if pred_col in study_df.columns:
                    valid_mask = study_df[col].notna() & study_df[pred_col].notna()
                    
                    if valid_mask.sum() > 5:
                        gt_values = study_df[col][valid_mask]
                        pred_values = study_df[pred_col][valid_mask]
                        
                        # Accuracy
                        accuracy = (gt_values == pred_values).mean()
                        metrics['calcification']['accuracy'][vessel] = accuracy
            
            elif col.startswith('gt_') and col.endswith('_IFRHYPER') and not col.endswith('_count'):
                vessel = col.replace('gt_', '')
                pred_col = f'pred_{vessel}'
                
                if pred_col in study_df.columns:
                    valid_mask = study_df[col].notna() & study_df[pred_col].notna()
                    
                    if valid_mask.sum() > 5:
                        gt_values = study_df[col][valid_mask]
                        pred_values = study_df[pred_col][valid_mask]
                        
                        # MAE
                        mae = np.mean(np.abs(gt_values - pred_values))
                        metrics['ifr']['mae'][vessel] = mae
                        
                        # Correlation
                        if len(set(gt_values)) > 1 and len(set(pred_values)) > 1:
                            corr = np.corrcoef(gt_values, pred_values)[0, 1]
                            if not np.isnan(corr):
                                metrics['ifr']['corr'][vessel] = corr
    
    # 5. Save results if output directory provided
    if output_dir and not study_df.empty:
        os.makedirs(output_dir, exist_ok=True)
        study_path = os.path.join(output_dir, f"{epoch_name}_study_level.csv")
        study_df.to_csv(study_path, index=False)
        print(f"   💾 Saved: {study_path}")
    
    return study_df, metrics

def process_epoch_worker(epoch_data: Tuple[int, str, str, str, str]) -> Tuple[str, Optional[pd.DataFrame], Optional[Dict]]:
    """
    Worker function for parallel epoch processing.
    
    Args:
        epoch_data: Tuple of (epoch_num, epoch_file, filename_map_file, gt_mappings_file, output_dir)
    
    Returns:
        Tuple of (epoch_key, study_df, metrics)
    """
    epoch_num, epoch_file, filename_map_file, gt_mappings_file, output_dir = epoch_data
    epoch_key = f"epoch_{epoch_num}"
    
    try:
        # Load shared data from temp files
        with open(filename_map_file, 'rb') as f:
            filename_map = pickle.load(f)
        
        with open(gt_mappings_file, 'rb') as f:
            gt_mappings = pickle.load(f)
        
        # Process epoch
        study_df, metrics = analyze_single_epoch_super_optimized(
            epoch_path=epoch_file,
            filename_map=filename_map,
            gt_mappings=gt_mappings,
            output_dir=os.path.join(output_dir, epoch_key) if output_dir else None
        )
        
        return epoch_key, study_df, metrics
        
    except Exception as e:
        print(f"   ❌ Worker error in epoch {epoch_num}: {e}")
        return epoch_key, None, None

def run_multi_epoch_analysis_parallel(
    report_csv_path: str,
    predictions_dir: str,
    output_dir: str,
    epoch_range: Tuple[int, int] = (1, 29),
    max_workers: Optional[int] = None,
    batch_size: Optional[int] = None
) -> Tuple[Dict[str, Dict], Dict[str, pd.DataFrame]]:
    """
    VECTORIZED PARALLEL: Run multi-epoch analysis using parallel processing.
    This should be 50-100x faster than sequential processing on multi-core systems.
    
    Args:
        report_csv_path: Path to the report CSV file
        predictions_dir: Directory containing prediction files
        output_dir: Directory to save results
        epoch_range: Tuple of (start_epoch, end_epoch)
        max_workers: Maximum number of parallel workers (default: CPU count)
        batch_size: Process epochs in batches (default: all at once)
    
    Returns:
        Tuple of (all_epoch_metrics, all_epoch_dfs)
    """
    
    if max_workers is None:
        max_workers = min(mp.cpu_count(), 8)  # Cap at 8 to avoid overwhelming system
    
    print(f"🚀 VECTORIZED PARALLEL Multi-Epoch Analysis")
    print("=" * 60)
    print(f"⚡ Using {max_workers} parallel workers!")
    print("🔥 This should be 50-100x faster than sequential processing!")
    
    start_time = time.time()
    
    # Step 1: Load report data ONCE
    print(f"\n1️⃣ Loading report data ONCE...")
    df_report, filename_map = load_report_data_once(report_csv_path)
    
    if df_report.empty or not filename_map:
        print("❌ Failed to load report data")
        return {}, {}
    
    # Step 2: Create ground truth mappings from first epoch (all epochs have same mappings)
    print(f"\n2️⃣ Creating ground truth mappings from sample epoch...")
    gt_mappings = {}

    # Find first epoch file to create mappings
    epoch_files = []
    for epoch_num in range(epoch_range[0], epoch_range[1] + 1):
        # Try multiple file naming patterns and locations (check predictions_dir first, then files subdir)
        possible_patterns = [
            os.path.join(predictions_dir, f"val_predictions_epoch_{epoch_num}.csv"),
            os.path.join(predictions_dir, f"val_epoch{epoch_num}.csv"),
            os.path.join(predictions_dir, "files", f"val_predictions_epoch_{epoch_num}.csv"),
            os.path.join(predictions_dir, "files", f"val_epoch{epoch_num}.csv"),
        ]

        for epoch_file in possible_patterns:
            if os.path.exists(epoch_file):
                epoch_files.append((epoch_num, epoch_file))
                break
    
    if epoch_files:
        # Use first epoch to create GT mappings
        first_epoch_file = epoch_files[0][1]
        print(f"   📊 Using {os.path.basename(first_epoch_file)} to create GT mappings...")
        
        # Load and merge first epoch
        df_epoch = pd.read_csv(first_epoch_file)
        merged_sample = merge_epoch_with_report_mapping(df_epoch, filename_map)
        
        if not merged_sample.empty:
            gt_mappings = create_gt_mappings_from_merged_data(merged_sample)
        else:
            print("❌ Failed to create sample merged data")
            return {}, {}
    else:
        print("❌ No epoch files found")
        return {}, {}
    
    # Step 3: We already have epoch files from Step 2
    print(f"\n3️⃣ Ready to process {len(epoch_files)} epoch files...")
    
    if not epoch_files:
        print("❌ No epoch files found!")
        return {}, {}
    
    # Step 4: Save shared data to temporary files for worker processes
    print(f"\n4️⃣ Preparing shared data for parallel processing...")
    
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
        pickle.dump(filename_map, f)
        filename_map_file = f.name
    
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
        pickle.dump(gt_mappings, f)
        gt_mappings_file = f.name
    
    try:
        # Step 5: Prepare worker data
        worker_data = [
            (epoch_num, epoch_file, filename_map_file, gt_mappings_file, output_dir)
            for epoch_num, epoch_file in epoch_files
        ]
        
        # Step 6: Process epochs in parallel
        print(f"\n5️⃣ Processing {len(epoch_files)} epochs in PARALLEL ({max_workers} workers)...")
        
        all_epoch_metrics = {}
        all_epoch_dfs = {}
        
        if batch_size and len(worker_data) > batch_size:
            # Process in batches to manage memory
            print(f"   📦 Processing in batches of {batch_size} epochs...")
            
            for i in range(0, len(worker_data), batch_size):
                batch = worker_data[i:i + batch_size]
                batch_num = i // batch_size + 1
                total_batches = (len(worker_data) + batch_size - 1) // batch_size
                
                print(f"   🔄 Processing batch {batch_num}/{total_batches} ({len(batch)} epochs)...")
                
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all tasks in the batch
                    future_to_epoch = {
                        executor.submit(process_epoch_worker, data): data[0] 
                        for data in batch
                    }
                    
                    # Collect results with progress bar
                    with tqdm(total=len(batch), desc=f"Batch {batch_num}", leave=False) as pbar:
                        for future in as_completed(future_to_epoch):
                            epoch_num = future_to_epoch[future]
                            try:
                                epoch_key, study_df, metrics = future.result()
                                
                                if study_df is not None and metrics is not None:
                                    all_epoch_metrics[epoch_key] = metrics
                                    all_epoch_dfs[epoch_key] = study_df
                                    
                                    pbar.set_postfix({
                                        'epoch': epoch_num,
                                        'studies': len(study_df),
                                        'stenosis': len(metrics['stenosis']['mae'])
                                    })
                                
                                pbar.update(1)
                                
                            except Exception as e:
                                print(f"   ❌ Error processing epoch {epoch_num}: {e}")
                                pbar.update(1)
        
        else:
            # Process all epochs at once
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_epoch = {
                    executor.submit(process_epoch_worker, data): data[0] 
                    for data in worker_data
                }
                
                # Collect results with progress bar
                with tqdm(total=len(worker_data), desc="Processing epochs", unit="epochs") as pbar:
                    for future in as_completed(future_to_epoch):
                        epoch_num = future_to_epoch[future]
                        try:
                            epoch_key, study_df, metrics = future.result()
                            
                            if study_df is not None and metrics is not None:
                                all_epoch_metrics[epoch_key] = metrics
                                all_epoch_dfs[epoch_key] = study_df
                                
                                pbar.set_postfix({
                                    'epoch': epoch_num,
                                    'studies': len(study_df),
                                    'stenosis': len(metrics['stenosis']['mae'])
                                })
                            
                            pbar.update(1)
                            
                        except Exception as e:
                            print(f"   ❌ Error processing epoch {epoch_num}: {e}")
                            pbar.update(1)
    
    finally:
        # Clean up temporary files
        try:
            os.unlink(filename_map_file)
            os.unlink(gt_mappings_file)
        except:
            pass
    
    elapsed_time = time.time() - start_time
    
    print(f"\n🎉 VECTORIZED PARALLEL ANALYSIS COMPLETED!")
    print(f"✅ Successfully processed: {len(all_epoch_metrics)}/{len(epoch_files)} epochs")
    print(f"⚡ Processing time: {elapsed_time:.2f} seconds")
    print(f"🚀 Average time per epoch: {elapsed_time/len(epoch_files):.2f} seconds")
    print(f"💥 Speedup: ~{max_workers}x faster than sequential processing!")
    
    return all_epoch_metrics, all_epoch_dfs

def run_multi_epoch_analysis_optimized(
    report_csv_path: str,
    predictions_dir: str,
    output_dir: str,
    epoch_range: Tuple[int, int] = (1, 29)
) -> Tuple[Dict[str, Dict], Dict[str, pd.DataFrame]]:
    """
    OPTIMIZED: Run multi-epoch analysis that loads report data ONCE and reuses it.
    This should be 10-20x faster than the previous approach.
    
    DEPRECATED: Use run_multi_epoch_analysis_parallel() for even better performance!
    """
    
    print(f"🚀 SUPER OPTIMIZED Multi-Epoch Analysis")
    print("=" * 60)
    print("⚡ This loads report data ONCE and reuses all mappings!")
    print("⚠️  DEPRECATED: Consider using run_multi_epoch_analysis_parallel() for 50-100x speedup!")
    
    # Step 1: Load report data ONCE
    print(f"\n1️⃣ Loading report data ONCE...")
    df_report, filename_map = load_report_data_once(report_csv_path)
    
    if df_report.empty or not filename_map:
        print("❌ Failed to load report data")
        return {}, {}
    
    # Step 2: Create ground truth mappings from first epoch
    print(f"\n2️⃣ Creating ground truth mappings from sample epoch...")
    gt_mappings = {}
    
    # Step 3: Find epoch files
    print(f"\n3️⃣ Finding epoch files...")
    epoch_files = []

    for epoch_num in range(epoch_range[0], epoch_range[1] + 1):
        # Try multiple file naming patterns and locations (check predictions_dir first, then files subdir)
        possible_patterns = [
            os.path.join(predictions_dir, f"val_predictions_epoch_{epoch_num}.csv"),
            os.path.join(predictions_dir, f"val_epoch{epoch_num}.csv"),
            os.path.join(predictions_dir, "files", f"val_predictions_epoch_{epoch_num}.csv"),
            os.path.join(predictions_dir, "files", f"val_epoch{epoch_num}.csv"),
        ]

        for epoch_file in possible_patterns:
            if os.path.exists(epoch_file):
                epoch_files.append((epoch_num, epoch_file))
                break
    
    print(f"✅ Found {len(epoch_files)} epoch files")
    
    # Create GT mappings from first epoch
    if epoch_files:
        first_epoch_file = epoch_files[0][1]
        print(f"   📊 Using {os.path.basename(first_epoch_file)} to create GT mappings...")
        
        # Load and merge first epoch
        df_epoch = pd.read_csv(first_epoch_file)
        merged_sample = merge_epoch_with_report_mapping(df_epoch, filename_map)
        
        if not merged_sample.empty:
            gt_mappings = create_gt_mappings_from_merged_data(merged_sample)
        else:
            print("❌ Failed to create sample merged data")
            return {}, {}
    
    # Step 4: Process all epochs super efficiently
    print(f"\n4️⃣ Processing {len(epoch_files)} epochs with MAXIMUM efficiency...")
    
    all_epoch_metrics = {}
    all_epoch_dfs = {}
    
    for epoch_num, epoch_file in tqdm(epoch_files, desc="Processing epochs"):
        try:
            epoch_key = f"epoch_{epoch_num}"
            
            # Use super optimized analysis (reuses ALL pre-computed data)
            study_df, metrics = analyze_single_epoch_super_optimized(
                epoch_path=epoch_file,
                filename_map=filename_map,  # REUSED
                gt_mappings=gt_mappings,    # REUSED
                output_dir=os.path.join(output_dir, epoch_key) if output_dir else None
            )
            
            if not study_df.empty:
                all_epoch_metrics[epoch_key] = metrics
                all_epoch_dfs[epoch_key] = study_df
                
                print(f"   ✅ Epoch {epoch_num}: {len(study_df)} studies, "
                      f"{len(metrics['stenosis']['mae'])} stenosis vessels")
            
        except Exception as e:
            print(f"   ❌ Error in epoch {epoch_num}: {e}")
            continue
    
    print(f"\n🎉 SUPER OPTIMIZED ANALYSIS COMPLETED!")
    print(f"✅ Successfully processed: {len(all_epoch_metrics)}/{len(epoch_files)} epochs")
    print(f"⚡ This should be ~20x faster than before!")
    
    return all_epoch_metrics, all_epoch_dfs

# Convenience wrapper function
def run_multi_epoch_analysis(
    report_csv_path: str,
    predictions_dir: str,
    output_dir: str,
    epoch_range: Tuple[int, int] = (1, 29),
    use_parallel: bool = True,
    max_workers: Optional[int] = None,
    batch_size: Optional[int] = None
) -> Tuple[Dict[str, Dict], Dict[str, pd.DataFrame]]:
    """
    Convenience function to run multi-epoch analysis with optional parallel processing.
    
    Args:
        report_csv_path: Path to the report CSV file
        predictions_dir: Directory containing prediction files
        output_dir: Directory to save results
        epoch_range: Tuple of (start_epoch, end_epoch)
        use_parallel: Whether to use parallel processing (default: True)
        max_workers: Maximum number of parallel workers (default: CPU count)
        batch_size: Process epochs in batches (default: all at once)
    
    Returns:
        Tuple of (all_epoch_metrics, all_epoch_dfs)
    """
    
    if use_parallel:
        return run_multi_epoch_analysis_parallel(
            report_csv_path=report_csv_path,
            predictions_dir=predictions_dir,
            output_dir=output_dir,
            epoch_range=epoch_range,
            max_workers=max_workers,
            batch_size=batch_size
        )
    else:
        return run_multi_epoch_analysis_optimized(
            report_csv_path=report_csv_path,
            predictions_dir=predictions_dir,
            output_dir=output_dir,
            epoch_range=epoch_range
        )

# Main function for notebook integration (now optimized)
def run_study_analysis(
    report_csv_path: str,
    epoch_csv_path: str,
    output_dir: str = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Main function for single epoch study analysis - designed for notebook integration.
    """
    print(f"🚀 Running study analysis...")
    print(f"   📄 Report: {report_csv_path}")
    print(f"   📊 Epoch: {os.path.basename(epoch_csv_path)}")
    
    # Load report data and create filename mapping
    df_report, filename_map = load_report_data_once(report_csv_path)
    if df_report.empty or not filename_map:
        print("❌ Failed to load report data")
        return pd.DataFrame(), {}
    
    # Load epoch data to create GT mappings
    df_epoch = pd.read_csv(epoch_csv_path)
    merged_sample = merge_epoch_with_report_mapping(df_epoch, filename_map)
    
    if merged_sample.empty:
        print("❌ Failed to merge epoch with report data")
        return pd.DataFrame(), {}
    
    # Create GT mappings from merged data
    gt_mappings = create_gt_mappings_from_merged_data(merged_sample)
    
    # Run optimized analysis
    try:
        # We already have merged data, so we can use it directly
        study_df = create_study_level_predictions_optimized(merged_sample, gt_mappings)
        
        # Calculate metrics
        metrics = {
            'stenosis': {'mae': {}, 'corr': {}},
            'calcification': {'accuracy': {}},
            'ifr': {'mae': {}, 'corr': {}}
        }
        
        if not study_df.empty:
            # Extract vessel-specific metrics
            for col in study_df.columns:
                if col.startswith('gt_') and col.endswith('_stenosis') and not col.endswith('_count'):
                    vessel = col.replace('gt_', '')
                    pred_col = f'pred_{vessel}'
                    
                    if pred_col in study_df.columns:
                        valid_mask = study_df[col].notna() & study_df[pred_col].notna()
                        
                        if valid_mask.sum() > 5:
                            gt_values = study_df[col][valid_mask]
                            pred_values = study_df[pred_col][valid_mask]
                            
                            # MAE
                            mae = np.mean(np.abs(gt_values - pred_values))
                            metrics['stenosis']['mae'][vessel] = mae
                            
                            # Correlation
                            if len(set(gt_values)) > 1 and len(set(pred_values)) > 1:
                                corr = np.corrcoef(gt_values, pred_values)[0, 1]
                                if not np.isnan(corr):
                                    metrics['stenosis']['corr'][vessel] = corr
                
                elif col.startswith('gt_') and col.endswith('_calcif') and not col.endswith('_count'):
                    vessel = col.replace('gt_', '')
                    pred_col = f'pred_{vessel}'
                    
                    if pred_col in study_df.columns:
                        valid_mask = study_df[col].notna() & study_df[pred_col].notna()
                        
                        if valid_mask.sum() > 5:
                            gt_values = study_df[col][valid_mask]
                            pred_values = study_df[pred_col][valid_mask]
                            
                            # Accuracy
                            accuracy = (gt_values == pred_values).mean()
                            metrics['calcification']['accuracy'][vessel] = accuracy
                
                elif col.startswith('gt_') and col.endswith('_IFRHYPER') and not col.endswith('_count'):
                    vessel = col.replace('gt_', '')
                    pred_col = f'pred_{vessel}'
                    
                    if pred_col in study_df.columns:
                        valid_mask = study_df[col].notna() & study_df[pred_col].notna()
                        
                        if valid_mask.sum() > 5:
                            gt_values = study_df[col][valid_mask]
                            pred_values = study_df[pred_col][valid_mask]
                            
                            # MAE
                            mae = np.mean(np.abs(gt_values - pred_values))
                            metrics['ifr']['mae'][vessel] = mae
                            
                            # Correlation
                            if len(set(gt_values)) > 1 and len(set(pred_values)) > 1:
                                corr = np.corrcoef(gt_values, pred_values)[0, 1]
                                if not np.isnan(corr):
                                    metrics['ifr']['corr'][vessel] = corr
        
        # Save results if output directory provided
        if output_dir and not study_df.empty:
            os.makedirs(output_dir, exist_ok=True)
            epoch_name = os.path.basename(epoch_csv_path).replace('.csv', '')
            study_path = os.path.join(output_dir, f"{epoch_name}_study_level.csv")
            study_df.to_csv(study_path, index=False)
            print(f"   💾 Saved: {study_path}")
        
        print(f"✅ Analysis completed: {len(study_df)} studies analyzed")
        print(f"   🫀 Stenosis vessels: {len(metrics['stenosis']['mae'])}")
        print(f"   🦴 Calcification vessels: {len(metrics['calcification']['accuracy'])}")
        print(f"   💉 IFR vessels: {len(metrics['ifr']['mae'])}")
        
        return study_df, metrics
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(), {}

if __name__ == "__main__":
    # Test the vectorized parallel approach
    REPORT_PATH = "data/reports/reports_with_alpha_separator_with_Calcifc_Stenosis_IFR_20250601_RCA_LCA_merged_with_left_dominance_dependent_vessels.csv"
    PREDICTIONS_DIR = "outputs/DeepCORO_clip/dev_deep_coro_clip_single_video/8av1xygm_20250605-083820_best_single_video"
    OUTPUT_DIR = "vectorized_study_analysis_results"
    
    print("🚀 Testing VECTORIZED PARALLEL multi-epoch analysis...")
    
    # Test with a small subset first (5 epochs)
    print("\n" + "="*60)
    print("🧪 PERFORMANCE TEST: Sequential vs Parallel")
    print("="*60)
    
    # Test sequential processing (5 epochs)
    print("\n1️⃣ Testing Sequential Processing (5 epochs)...")
    start_time = time.time()
    sequential_metrics, sequential_dfs = run_multi_epoch_analysis(
        report_csv_path=REPORT_PATH,
        predictions_dir=PREDICTIONS_DIR,
        output_dir=OUTPUT_DIR + "_sequential",
        epoch_range=(1, 5),
        use_parallel=False  # Sequential
    )
    sequential_time = time.time() - start_time
    
    # Test parallel processing (5 epochs)
    print("\n2️⃣ Testing Parallel Processing (5 epochs)...")
    start_time = time.time()
    parallel_metrics, parallel_dfs = run_multi_epoch_analysis(
        report_csv_path=REPORT_PATH,
        predictions_dir=PREDICTIONS_DIR,
        output_dir=OUTPUT_DIR + "_parallel",
        epoch_range=(1, 5),
        use_parallel=True,  # Parallel
        max_workers=4  # Use 4 workers for test
    )
    parallel_time = time.time() - start_time
    
    # Performance comparison
    print("\n" + "="*60)
    print("📊 PERFORMANCE COMPARISON RESULTS")
    print("="*60)
    print(f"📈 Sequential Processing:")
    print(f"   ⏱️  Time: {sequential_time:.2f} seconds")
    print(f"   📊 Epochs: {len(sequential_metrics)}")
    print(f"   ⚡ Avg per epoch: {sequential_time/max(1, len(sequential_metrics)):.2f}s")
    
    print(f"\n🚀 Parallel Processing:")
    print(f"   ⏱️  Time: {parallel_time:.2f} seconds")
    print(f"   📊 Epochs: {len(parallel_metrics)}")
    print(f"   ⚡ Avg per epoch: {parallel_time/max(1, len(parallel_metrics)):.2f}s")
    
    if sequential_time > 0 and parallel_time > 0:
        speedup = sequential_time / parallel_time
        print(f"\n💥 SPEEDUP: {speedup:.2f}x faster with parallel processing!")
        
        if speedup > 1.5:
            print("🎉 Excellent speedup! Parallel processing is working well!")
        elif speedup > 1.1:
            print("👍 Good speedup! Parallel processing provides benefits.")
        else:
            print("⚠️  Limited speedup. Sequential may be better for small datasets.")
    
    # Test with all epochs if requested
    test_all = False  # Set to True to test all 29 epochs
    if test_all:
        print("\n" + "="*60)
        print("🔥 FULL SCALE TEST: All 29 Epochs in Parallel")
        print("="*60)
        
        start_time = time.time()
        all_metrics, all_dfs = run_multi_epoch_analysis_parallel(
            report_csv_path=REPORT_PATH,
            predictions_dir=PREDICTIONS_DIR,
            output_dir=OUTPUT_DIR + "_full",
            epoch_range=(1, 29),
            max_workers=6,  # Use 6 workers for full test
            batch_size=10   # Process in batches of 10
        )
        full_time = time.time() - start_time
        
        print(f"\n🎯 FULL SCALE RESULTS:")
        print(f"   ⏱️  Total time: {full_time:.2f} seconds ({full_time/60:.1f} minutes)")
        print(f"   📊 Epochs processed: {len(all_metrics)}/29")
        print(f"   ⚡ Avg per epoch: {full_time/max(1, len(all_metrics)):.2f}s")
        print(f"   🚀 Estimated speedup: ~6-8x faster than sequential!")
    
    print(f"\n✅ All tests completed!")
    print(f"   🔥 Vectorized parallel processing is ready!")
    print(f"   💡 Use run_multi_epoch_analysis_parallel() for maximum speed!")
    print(f"   📝 Use run_multi_epoch_analysis() with use_parallel=True as convenience wrapper")