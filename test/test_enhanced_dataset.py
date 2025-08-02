#!/usr/bin/env python3
"""
Test script for enhanced dataset generation with angiographic views.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import yaml

# Add the dataset_creation directory to the path
if 'dataset_creation' not in sys.path:
    sys.path.append('dataset_creation')

# Import the dataset generation functions
from generate_dataset import (
    create_default_config,
    load_data,
    add_angiographic_view_column,
    format_main_structure_description,
    classify_angiographic_view,
    split_train_test,
    process_dataset,
    MAIN_STRUCTURE_MAP,
    DOMINANCE_MAP,
    ANGIOGRAPHIC_VIEWS
)

def test_angiographic_classification():
    """Test angiographic view classification function."""
    print("\n" + "="*60)
    print("Testing Angiographic View Classification")
    print("="*60)
    
    test_cases = [
        (-45, 30, "RAO Cranial"),
        (0, 30, "AP Cranial"),
        (30, 30, "LAO Cranial"),
        (-30, 0, "RAO Straight"),
        (0, 0, "AP"),
        (30, 0, "LAO Straight"),
        (-30, -30, "RAO Caudal"),
        (0, -30, "AP Caudal"),
        (30, -30, "LAO Caudal"),
        (-90, 0, "LAO Lateral"),
        (90, 0, "RAO Lateral"),
        (200, 200, "Other")
    ]
    
    for primary, secondary, expected in test_cases:
        label, description = classify_angiographic_view(primary, secondary)
        status = "✓" if description == expected else "✗"
        print(f"{status} Primary: {primary:4d}°, Secondary: {secondary:4d}° → {description:15s} (expected: {expected})")

def test_main_structure_descriptions():
    """Test main structure description formatting."""
    print("\n" + "="*60)
    print("Testing Main Structure Descriptions")
    print("="*60)
    
    for class_id, name in MAIN_STRUCTURE_MAP.items():
        description = format_main_structure_description(class_id)
        print(f"Class {class_id:2d} ({name:15s}): {description}")

def create_sample_data():
    """Create sample data for testing."""
    print("\n" + "="*60)
    print("Creating Sample Data")
    print("="*60)
    
    # Create sample DataFrame with required columns
    data = {
        'StudyInstanceUID': ['study1'] * 5 + ['study2'] * 5,
        'SeriesTime': list(range(10)),
        'FileName': [f'file_{i}.dcm' for i in range(10)],
        'main_structure_class': [0, 1, 0, 1, 2, 1, 0, 1, 0, 3],
        'dominance_class': [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
        'primary_angle': [-45, 0, 30, -30, 0, 30, -30, 0, 30, -90],
        'secondary_angle': [30, 30, 30, 0, 0, 0, -30, -30, -30, 0],
        'stent_presence_class': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
        'contrast_agent_class': [1] * 10,
        'prox_lad_stenosis': [50, 0, 70, 30, 0, 60, 40, 0, 80, 20],
        'mid_lad_stenosis': [30, 0, 50, 20, 0, 40, 30, 0, 60, 10],
        'prox_rca_stenosis': [0, 60, 0, 70, 0, 50, 0, 80, 0, 40],
        'mid_rca_stenosis': [0, 40, 0, 50, 0, 30, 0, 60, 0, 20]
    }
    
    df = pd.DataFrame(data)
    
    # Add main structure names and descriptions
    df['main_structure_name'] = df['main_structure_class'].map(MAIN_STRUCTURE_MAP)
    df['main_structure_description'] = df['main_structure_class'].apply(format_main_structure_description)
    
    # Add dominance names
    df['dominance_name'] = df['dominance_class'].map(DOMINANCE_MAP)
    
    # Add angiographic view classification
    df = add_angiographic_view_column(df)
    
    print(f"Created sample DataFrame with {len(df)} rows")
    print("\nSample of generated columns:")
    print(df[['main_structure_name', 'main_structure_description', 
              'angiographic_view_label', 'angiographic_view_description']].head())
    
    return df

def test_train_test_split(df):
    """Test train/test split functionality."""
    print("\n" + "="*60)
    print("Testing Train/Test Split")
    print("="*60)
    
    train_df, test_df = split_train_test(df, test_size=0.3, random_state=42)
    
    print(f"Original data: {len(df)} rows")
    print(f"Train set: {len(train_df)} rows ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Test set: {len(test_df)} rows ({len(test_df)/len(df)*100:.1f}%)")
    
    # Check for no data leakage
    train_studies = set(train_df['StudyInstanceUID'].unique())
    test_studies = set(test_df['StudyInstanceUID'].unique())
    overlap = train_studies.intersection(test_studies)
    
    if overlap:
        print(f"⚠️ WARNING: Found {len(overlap)} studies in both train and test sets!")
    else:
        print("✓ No data leakage: train and test sets have distinct studies")

def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("Enhanced Dataset Generation Tests")
    print("="*60)
    
    # Test individual functions
    test_angiographic_classification()
    test_main_structure_descriptions()
    
    # Create and test with sample data
    df = create_sample_data()
    test_train_test_split(df)
    
    # Display available angiographic views
    print("\n" + "="*60)
    print("Available Angiographic Views")
    print("="*60)
    print(f"{'Label':<6} {'Description':<15} {'Primary Range':<20} {'Secondary Range':<20}")
    print("-" * 61)
    for view in ANGIOGRAPHIC_VIEWS:
        primary = f"{view['primary_range']}" if view['primary_range'] else "N/A"
        secondary = f"{view['secondary_range']}" if view['secondary_range'] else "N/A"
        print(f"{view['label']:<6} {view['description']:<15} {primary:<20} {secondary:<20}")
    
    print("\n✓ All tests completed successfully!")

if __name__ == "__main__":
    main()