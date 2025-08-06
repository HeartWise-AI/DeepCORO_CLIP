#!/usr/bin/env python3
"""
Test script for enhanced dataset generation with normal report filtering.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_creation.generate_dataset import apply_hard_filters, create_report

def create_test_dataframe():
    """Create a test DataFrame with mix of normal and abnormal stenosis values."""
    n_rows = 100
    
    # Create base data
    data = {
        'StudyInstanceUID': [f'study_{i}' for i in range(n_rows)],
        'SeriesTime': range(n_rows),
        'main_structure_class': np.random.choice([0, 1], n_rows),  # 0: Left, 1: Right
        'main_structure_name': ['Left Coronary' if x == 0 else 'Right Coronary' 
                                for x in np.random.choice([0, 1], n_rows)],
        'dominance_class': np.random.choice([0, 1], n_rows),
        'dominance_name': ['right_dominant' if x == 0 else 'left_dominant' 
                          for x in np.random.choice([0, 1], n_rows)],
        'stent_presence_class': np.zeros(n_rows),  # All diagnostic
        'contrast_agent_class': np.ones(n_rows),
        'External_Exam': np.zeros(n_rows, dtype=bool),
        'bypass_graft': np.zeros(n_rows),
        'status': ['diagnostic'] * n_rows,
    }
    
    # Create stenosis columns with 60% normal (all 0s) and 40% with some stenosis
    stenosis_columns = [
        "prox_rca_stenosis", "mid_rca_stenosis", "dist_rca_stenosis",
        "left_main_stenosis", "prox_lad_stenosis", "mid_lad_stenosis", "dist_lad_stenosis",
        "D1_stenosis", "D2_stenosis", "prox_lcx_stenosis", "dist_lcx_stenosis",
        "om1_stenosis", "om2_stenosis", "bx_stenosis", "lvp_stenosis",
        "pda_stenosis", "posterolateral_stenosis"
    ]
    
    # Initialize all stenosis columns
    for col in stenosis_columns:
        data[col] = np.zeros(n_rows)
    
    # For 40% of rows, add some stenosis values
    abnormal_rows = np.random.choice(n_rows, size=int(n_rows * 0.4), replace=False)
    
    for idx in abnormal_rows:
        # Randomly select which vessels have stenosis
        n_stenotic_vessels = np.random.randint(1, 6)
        
        # Choose vessels based on structure
        if data['main_structure_name'][idx] == 'Right Coronary':
            vessel_pool = ["prox_rca_stenosis", "mid_rca_stenosis", "dist_rca_stenosis",
                          "pda_stenosis", "posterolateral_stenosis"]
        else:
            vessel_pool = ["left_main_stenosis", "prox_lad_stenosis", "mid_lad_stenosis", 
                          "dist_lad_stenosis", "D1_stenosis", "D2_stenosis",
                          "prox_lcx_stenosis", "dist_lcx_stenosis", "om1_stenosis", "om2_stenosis"]
        
        stenotic_vessels = np.random.choice(vessel_pool, 
                                           size=min(n_stenotic_vessels, len(vessel_pool)), 
                                           replace=False)
        
        for vessel in stenotic_vessels:
            # Assign random stenosis value (10-100)
            data[vessel][idx] = np.random.choice([20, 30, 40, 50, 60, 70, 80, 90, 100])
    
    df = pd.DataFrame(data)
    return df

def test_normal_report_filtering():
    """Test that normal report filtering works correctly."""
    print("Testing Normal Report Filtering")
    print("=" * 50)
    
    # Create test data
    df = create_test_dataframe()
    print(f"Created test DataFrame with {len(df)} rows")
    
    # Count initial normal reports
    stenosis_columns = [col for col in df.columns if col.endswith('_stenosis')]
    is_normal = (df[stenosis_columns] == 0).all(axis=1)
    initial_normal_count = is_normal.sum()
    initial_abnormal_count = (~is_normal).sum()
    
    print(f"Initial distribution:")
    print(f"  - Normal reports: {initial_normal_count} ({initial_normal_count/len(df)*100:.1f}%)")
    print(f"  - Abnormal reports: {initial_abnormal_count} ({initial_abnormal_count/len(df)*100:.1f}%)")
    
    # Apply filters with normal report ratio
    config = {
        'filters': {
            'status': 'diagnostic',
            'main_structures': ['Left Coronary', 'Right Coronary'],
            'contrast_agent_class': 1,
            'normal_report_ratio': 0.05  # Limit to 5% per vessel type
        }
    }
    
    print(f"\nApplying filters with normal_report_ratio = {config['filters']['normal_report_ratio']} per vessel type")
    df_filtered = apply_hard_filters(df, config)
    
    # Count normal reports after filtering by vessel type
    print(f"\nFinal distribution:")
    print(f"  - Total rows: {len(df_filtered)}")
    
    # Check RCA normal reports
    rca_mask = df_filtered['main_structure_name'] == 'Right Coronary'
    rca_filtered = df_filtered[rca_mask]
    if len(rca_filtered) > 0:
        rca_vessels = ["prox_rca_stenosis", "mid_rca_stenosis", "dist_rca_stenosis",
                      "pda_stenosis", "posterolateral_stenosis"]
        existing_rca = [col for col in rca_vessels if col in df_filtered.columns]
        rca_stenosis = rca_filtered[existing_rca].fillna(-1)
        is_normal_rca = ((rca_stenosis == 0) | (rca_stenosis == -1)).all(axis=1) & \
                       (rca_stenosis == 0).any(axis=1)
        rca_normal = is_normal_rca.sum()
        print(f"  - RCA: {rca_normal} normal / {len(rca_filtered)} total ({rca_normal/len(rca_filtered)*100:.1f}%)")
    
    # Check LCA normal reports
    lca_mask = df_filtered['main_structure_name'] == 'Left Coronary'
    lca_filtered = df_filtered[lca_mask]
    if len(lca_filtered) > 0:
        lca_vessels = ["left_main_stenosis", "prox_lad_stenosis", "mid_lad_stenosis", 
                      "dist_lad_stenosis", "D1_stenosis", "D2_stenosis",
                      "prox_lcx_stenosis", "dist_lcx_stenosis", "om1_stenosis", "om2_stenosis"]
        existing_lca = [col for col in lca_vessels if col in df_filtered.columns]
        lca_stenosis = lca_filtered[existing_lca].fillna(-1)
        is_normal_lca = ((lca_stenosis == 0) | (lca_stenosis == -1)).all(axis=1) & \
                       (lca_stenosis == 0).any(axis=1)
        lca_normal = is_normal_lca.sum()
        print(f"  - LCA: {lca_normal} normal / {len(lca_filtered)} total ({lca_normal/len(lca_filtered)*100:.1f}%)")
    
    target_ratio = config['filters']['normal_report_ratio']
    print(f"\nValidation:")
    if len(rca_filtered) > 0 and rca_normal/len(rca_filtered) <= target_ratio + 0.01:
        print(f"✓ RCA normal ratio ({rca_normal/len(rca_filtered):.2f}) is within target ({target_ratio})")
    if len(lca_filtered) > 0 and lca_normal/len(lca_filtered) <= target_ratio + 0.01:
        print(f"✓ LCA normal ratio ({lca_normal/len(lca_filtered):.2f}) is within target ({target_ratio})")
    
    # Test report generation for a few samples
    print("\n" + "=" * 50)
    print("Sample Reports:")
    print("=" * 50)
    
    # Generate overall normal check for samples
    all_stenosis_cols = [col for col in df_filtered.columns if col.endswith('_stenosis')]
    overall_stenosis = df_filtered[all_stenosis_cols].fillna(-1)
    is_overall_normal = ((overall_stenosis == 0) | (overall_stenosis == -1)).all(axis=1) & \
                       (overall_stenosis == 0).any(axis=1)
    
    # Show a normal report
    normal_samples = df_filtered[is_overall_normal].head(1)
    if len(normal_samples) > 0:
        print("\nNormal Report Example:")
        report = create_report(normal_samples.iloc[0], coronary_specific_report=True)
        print(report)
    
    # Show an abnormal report
    abnormal_samples = df_filtered[~is_overall_normal].head(1)
    if len(abnormal_samples) > 0:
        print("\nAbnormal Report Example:")
        report = create_report(abnormal_samples.iloc[0], coronary_specific_report=True)
        print(report)
    
    return df_filtered

def test_edge_cases():
    """Test edge cases for the filtering logic."""
    print("\n" + "=" * 50)
    print("Testing Edge Cases")
    print("=" * 50)
    
    # Test 1: All reports are normal
    print("\nTest 1: All normal reports")
    df = create_test_dataframe()
    stenosis_columns = [col for col in df.columns if col.endswith('_stenosis')]
    for col in stenosis_columns:
        df[col] = 0  # Set all to normal
    
    config = {'filters': {'normal_report_ratio': 0.05}}
    df_filtered = apply_hard_filters(df, config)
    
    # Count by vessel type
    rca_count = len(df_filtered[df_filtered['main_structure_name'] == 'Right Coronary'])
    lca_count = len(df_filtered[df_filtered['main_structure_name'] == 'Left Coronary'])
    print(f"  RCA: {rca_count} rows remaining")
    print(f"  LCA: {lca_count} rows remaining")
    print(f"  Total: {len(df_filtered)} rows")
    
    # Test 2: No normal reports
    print("\nTest 2: No normal reports")
    df = create_test_dataframe()
    for col in stenosis_columns:
        df[col] = np.random.choice([30, 50, 70], len(df))  # All abnormal
    
    df_filtered = apply_hard_filters(df, config)
    is_normal = (df_filtered[stenosis_columns] == 0).all(axis=1)
    print(f"  Normal reports after filtering: {is_normal.sum()} (should be 0)")
    
    # Test 3: Mixed with NaN values
    print("\nTest 3: Mixed with NaN values")
    df = create_test_dataframe()
    # Set some columns to NaN
    for col in stenosis_columns[:5]:
        df.loc[df.index[:20], col] = np.nan
    
    df_filtered = apply_hard_filters(df, config)
    print(f"  Rows after filtering: {len(df_filtered)}")

if __name__ == "__main__":
    print("Enhanced Dataset Generation Test")
    print("=" * 50)
    
    # Run main test
    df_filtered = test_normal_report_filtering()
    
    # Run edge case tests
    test_edge_cases()
    
    print("\n" + "=" * 50)
    print("All tests completed!")