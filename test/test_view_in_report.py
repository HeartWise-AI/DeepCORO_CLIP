#!/usr/bin/env python3
"""
Test that angiographic view descriptions appear in reports.
"""

import os
import sys
import pandas as pd
import numpy as np

# Add workspace to path
workspace_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if workspace_path not in sys.path:
    sys.path.insert(0, workspace_path)

# Add dataset_creation to path
if 'dataset_creation' not in sys.path:
    sys.path.append(os.path.join(workspace_path, 'dataset_creation'))

from dataset_creation.generate_dataset import (
    create_report,
    add_angiographic_view_column,
    format_main_structure_description,
    MAIN_STRUCTURE_MAP,
    DOMINANCE_MAP
)

def test_report_with_view():
    """Test that reports include view descriptions."""
    
    # Create a sample row with all necessary data
    row_data = {
        # Angiographic view data
        'primary_angle': -45,  # RAO Cranial
        'secondary_angle': 30,
        
        # Main structure data
        'main_structure_class': 0,  # Left Coronary
        'main_structure_name': 'Left Coronary',
        
        # Dominance
        'dominance_name': 'right_dominant',
        
        # Some stenosis values
        'prox_lad_stenosis': 70,
        'mid_lad_stenosis': 50,
        'dist_lad_stenosis': 30,
        'prox_rca_stenosis': 0,
        'mid_rca_stenosis': 0,
        
        # Other columns
        'Conclusion': 'Normal findings',
        'bypass_graft': 0
    }
    
    # Create DataFrame and add view classification
    df = pd.DataFrame([row_data])
    df = add_angiographic_view_column(df)
    
    # Add main structure description
    df['main_structure_description'] = df['main_structure_class'].apply(format_main_structure_description)
    
    # Print the view and structure info
    print("=" * 60)
    print("Row Data Before Report Generation:")
    print("=" * 60)
    print(f"Primary angle: {df.iloc[0]['primary_angle']}°")
    print(f"Secondary angle: {df.iloc[0]['secondary_angle']}°")
    print(f"Angiographic view label: {df.iloc[0]['angiographic_view_label']}")
    print(f"Angiographic view description: {df.iloc[0]['angiographic_view_description']}")
    print(f"Main structure name: {df.iloc[0]['main_structure_name']}")
    print(f"Main structure description: {df.iloc[0]['main_structure_description']}")
    
    # Generate report
    row = df.iloc[0]
    report = create_report(row, coronary_specific_report=True)
    
    print("\n" + "=" * 60)
    print("Generated Report:")
    print("=" * 60)
    print(report)
    
    print("\n" + "=" * 60)
    print("Checking Report Content:")
    print("=" * 60)
    
    # Check if view description is in report
    if "The view is" in report:
        print("✓ View description found in report")
        # Extract the view line
        for line in report.split('\n'):
            if "The view is" in line:
                print(f"  View line: {line}")
    else:
        print("✗ View description NOT found in report")
    
    # Check if structure description is in report
    if "This is a" in report:
        print("✓ Structure description found in report")
        # Extract the structure line
        for line in report.split('\n'):
            if "This is a" in line:
                print(f"  Structure line: {line}")
    else:
        print("✗ Structure description NOT found in report")
    
    # Check stenosis descriptions
    if "stenosis" in report.lower():
        print("✓ Stenosis descriptions found in report")
    
    return report

def test_multiple_views():
    """Test different view classifications."""
    print("\n" + "=" * 60)
    print("Testing Multiple View Types:")
    print("=" * 60)
    
    test_cases = [
        {'primary_angle': -45, 'secondary_angle': 30, 'expected': 'RAO Cranial'},
        {'primary_angle': 0, 'secondary_angle': 0, 'expected': 'AP'},
        {'primary_angle': 30, 'secondary_angle': -30, 'expected': 'LAO Caudal'},
        {'primary_angle': -90, 'secondary_angle': 0, 'expected': 'LAO Lateral'},
        {'primary_angle': 90, 'secondary_angle': 0, 'expected': 'RAO Lateral'},
    ]
    
    for i, test in enumerate(test_cases, 1):
        row_data = {
            'primary_angle': test['primary_angle'],
            'secondary_angle': test['secondary_angle'],
            'main_structure_class': 0,
            'main_structure_name': 'Left Coronary',
            'dominance_name': 'right_dominant',
            'prox_lad_stenosis': 50,
            'bypass_graft': 0
        }
        
        df = pd.DataFrame([row_data])
        df = add_angiographic_view_column(df)
        df['main_structure_description'] = df['main_structure_class'].apply(format_main_structure_description)
        
        row = df.iloc[0]
        report = create_report(row, coronary_specific_report=True)
        
        print(f"\nTest {i}: Primary={test['primary_angle']}°, Secondary={test['secondary_angle']}°")
        print(f"Expected view: {test['expected']}")
        print(f"View in data: {row['angiographic_view_description']}")
        
        # Get first two lines of report
        lines = report.split('\n')[:2]
        print(f"First lines of report:")
        for line in lines:
            print(f"  {line}")
        
        if f"The view is {test['expected']}" in report:
            print(f"✓ View '{test['expected']}' found in report")
        else:
            print(f"✗ View '{test['expected']}' NOT found in report")

if __name__ == "__main__":
    print("Testing View Descriptions in Reports")
    print("=" * 60)
    
    # Test single report
    report = test_report_with_view()
    
    # Test multiple views
    test_multiple_views()
    
    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)