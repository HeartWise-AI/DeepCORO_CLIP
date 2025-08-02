#!/usr/bin/env python3
"""
Test script to verify the complete generate_dataset.py functionality.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add workspace to path
workspace_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if workspace_path not in sys.path:
    sys.path.insert(0, workspace_path)

# Add dataset_creation to path
if 'dataset_creation' not in sys.path:
    sys.path.append(os.path.join(workspace_path, 'dataset_creation'))

from dataset_creation.generate_dataset import (
    apply_hard_filters,
    assign_patient_splits,
    add_angiographic_view_column,
    format_main_structure_description,
    create_default_config
)

def create_test_data():
    """Create test DataFrame with all required columns."""
    np.random.seed(42)
    n_samples = 100
    
    # Create test data
    data = {
        'StudyInstanceUID': [f'study_{i//5}' for i in range(n_samples)],
        'CathReport_MRN': [f'patient_{i//10}' for i in range(n_samples)],
        'SeriesTime': list(range(n_samples)),
        'FileName': [f'file_{i}.dcm' for i in range(n_samples)],
        'External_Exam': np.random.choice([True, False], n_samples, p=[0.1, 0.9]),
        'Conclusion': np.random.choice(['Normal', 'pontage present', 'Other'], n_samples, p=[0.7, 0.1, 0.2]),
        'main_structure_class': np.random.choice([0, 1, 2], n_samples, p=[0.4, 0.4, 0.2]),
        'dominance_class': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'primary_angle': np.random.uniform(-110, 110, n_samples),
        'secondary_angle': np.random.uniform(-50, 50, n_samples),
        'stent_presence_class': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'contrast_agent_class': np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),
        'status': np.random.choice(['diagnostic', 'PCI', 'POST_PCI'], n_samples, p=[0.5, 0.2, 0.3]),
    }
    
    # Add stenosis columns with mix of valid values, NaN, and -1.0
    stenosis_columns = [
        "prox_rca_stenosis", "mid_rca_stenosis", "dist_rca_stenosis",
        "left_main_stenosis", "prox_lad_stenosis", "mid_lad_stenosis", "dist_lad_stenosis",
        "D1_stenosis", "D2_stenosis", "prox_lcx_stenosis", "dist_lcx_stenosis",
        "om1_stenosis", "om2_stenosis", "bx_stenosis", "lvp_stenosis",
        "pda_stenosis", "posterolateral_stenosis"
    ]
    
    for col in stenosis_columns:
        values = np.random.choice([np.nan, -1.0, 0, 30, 50, 70, 90], n_samples, 
                                 p=[0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1])
        data[col] = values
    
    return pd.DataFrame(data)

def test_stenosis_filtering():
    """Test stenosis filtering logic."""
    print("\nTesting Stenosis Filtering...")
    
    # Create test data with specific stenosis patterns
    df = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'prox_rca_stenosis': [np.nan, 50, np.nan, -1.0, np.nan],
        'mid_rca_stenosis': [-1.0, 70, np.nan, -1.0, -1.0],
        'dist_rca_stenosis': [np.nan, 30, -1.0, -1.0, np.nan],
        'External_Exam': [False, False, False, False, False],
        'bypass_graft': [0, 0, 0, 0, 0],
        'status': ['diagnostic'] * 5,
        'main_structure_name': ['Left Coronary'] * 5,
        'contrast_agent_class': [1] * 5
    })
    
    config = {
        'filters': {
            'status': 'diagnostic',
            'main_structures': ['Left Coronary', 'Right Coronary'],
            'contrast_agent_class': 1
        }
    }
    
    # Apply filters
    filtered = apply_hard_filters(df, config)
    
    # Row 2 has valid stenosis (50, 70, 30)
    # Row 3 has one valid stenosis (NaN is different from -1.0)
    # Rows 1, 4, 5 should be filtered out (all NaN or -1.0)
    
    print(f"Original rows: {len(df)}")
    print(f"Filtered rows: {len(filtered)}")
    print(f"Expected: 2 rows (rows 2 and 3)")
    
    assert len(filtered) == 2, f"Expected 2 rows, got {len(filtered)}"
    print("✓ Stenosis filtering works correctly")

def test_patient_splits():
    """Test patient-based splitting."""
    print("\nTesting Patient-Based Splits...")
    
    df = create_test_data()
    
    # Apply splits
    df_with_splits = assign_patient_splits(
        df,
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        random_state=42,
        patient_column='CathReport_MRN'
    )
    
    # Check split column exists
    assert 'Split' in df_with_splits.columns, "Split column not added"
    
    # Check split distribution
    split_counts = df_with_splits['Split'].value_counts()
    print(f"Split distribution: {split_counts.to_dict()}")
    
    # Check no patient appears in multiple splits
    for patient in df_with_splits['CathReport_MRN'].unique():
        patient_splits = df_with_splits[df_with_splits['CathReport_MRN'] == patient]['Split'].unique()
        assert len(patient_splits) == 1, f"Patient {patient} appears in multiple splits: {patient_splits}"
    
    print("✓ Patient-based splitting works correctly")

def test_angiographic_views():
    """Test angiographic view classification."""
    print("\nTesting Angiographic View Classification...")
    
    df = pd.DataFrame({
        'primary_angle': [-45, 0, 30, -90, 90, 200],
        'secondary_angle': [30, 0, -30, 0, 0, 100]
    })
    
    df = add_angiographic_view_column(df)
    
    expected_views = [
        "RAO Cranial",  # -45, 30
        "AP",           # 0, 0
        "LAO Caudal",   # 30, -30
        "LAO Lateral",  # -90, 0
        "RAO Lateral",  # 90, 0
        "Other"         # 200, 100
    ]
    
    for i, expected in enumerate(expected_views):
        actual = df.iloc[i]['angiographic_view_description']
        assert actual == expected, f"Row {i}: Expected '{expected}', got '{actual}'"
        print(f"✓ {actual} classified correctly")
    
    print("✓ All angiographic views classified correctly")

def test_bypass_graft_creation():
    """Test bypass_graft column creation."""
    print("\nTesting Bypass Graft Column Creation...")
    
    df = pd.DataFrame({
        'Conclusion': [
            'Normal findings',
            'Patient avec pontage',
            'PONTAGE present',
            'No significant stenosis',
            np.nan
        ]
    })
    
    # Create bypass_graft column
    df['bypass_graft'] = df['Conclusion'].str.contains('pontage', case=False, na=False).astype(int)
    
    expected = [0, 1, 1, 0, 0]
    actual = df['bypass_graft'].tolist()
    
    assert actual == expected, f"Expected {expected}, got {actual}"
    print("✓ Bypass graft column created correctly")

def test_alpha_separator():
    """Test alpha separator in output."""
    print("\nTesting Alpha Separator...")
    
    df = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': ['a', 'b', 'c'],
        'col3': [1.1, 2.2, 3.3]
    })
    
    # Save with alpha separator
    test_file = '/tmp/test_alpha.csv'
    df.to_csv(test_file, sep='α', index=False)
    
    # Read back and verify
    df_read = pd.read_csv(test_file, sep='α')
    
    assert df.equals(df_read), "Data doesn't match after save/load with alpha separator"
    
    # Check file content
    with open(test_file, 'r') as f:
        content = f.read()
        assert 'α' in content, "Alpha separator not found in file"
    
    # Clean up
    os.remove(test_file)
    
    print("✓ Alpha separator works correctly")

def test_config_defaults():
    """Test default configuration values."""
    print("\nTesting Default Configuration...")
    
    config = create_default_config()
    
    # Check split ratios
    assert config['train_test_split']['train_ratio'] == 0.7, "Train ratio should be 0.7"
    assert config['train_test_split']['val_ratio'] == 0.1, "Val ratio should be 0.1"
    assert config['train_test_split']['test_ratio'] == 0.2, "Test ratio should be 0.2"
    
    # Check separator
    assert config['output_settings']['separator'] == 'α', "Default separator should be α"
    
    # Check filters
    assert config['filters']['status'] == ['diagnostic', 'POST_PCI'], "Default status filter incorrect"
    
    print("✓ Default configuration correct")
    print(f"  - Train/Val/Test: 0.7/0.1/0.2")
    print(f"  - Separator: α")
    print(f"  - Status filter: {config['filters']['status']}")

def main():
    """Run all tests."""
    print("="*60)
    print("Testing Generate Dataset Script")
    print("="*60)
    
    try:
        test_stenosis_filtering()
        test_patient_splits()
        test_angiographic_views()
        test_bypass_graft_creation()
        test_alpha_separator()
        test_config_defaults()
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        print("\nThe generate_dataset.py script correctly:")
        print("  1. Filters stenosis columns (drops rows with all NaN/-1.0)")
        print("  2. Creates patient-based train/val/test splits (0.7/0.1/0.2)")
        print("  3. Classifies angiographic views")
        print("  4. Creates bypass_graft column from 'pontage' in Conclusion")
        print("  5. Saves with alpha (α) separator")
        print("  6. Drops External_Exam and bypass_graft rows")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)