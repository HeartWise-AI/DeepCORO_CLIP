#!/usr/bin/env python3
"""
Test angiographic view classification with actual data angles.
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
    classify_angiographic_view,
    ANGIOGRAPHIC_VIEWS
)

def test_with_actual_angles():
    """Test classification with actual angle values from the data."""
    
    print("Current Angiographic View Ranges:")
    print("=" * 80)
    print(f"{'View':<15} {'Primary Range':<25} {'Secondary Range':<25}")
    print("-" * 80)
    for view in ANGIOGRAPHIC_VIEWS:
        primary = str(view['primary_range']) if view['primary_range'] else "None"
        secondary = str(view['secondary_range']) if view['secondary_range'] else "None"
        print(f"{view['description']:<15} {primary:<25} {secondary:<25}")
    
    print("\n" + "=" * 80)
    print("Testing with Actual Data Angles:")
    print("=" * 80)
    
    # Test angles from actual data
    test_angles = [
        (-3.0, -3.0),    # Common in data
        (-3.0, 40.0),    # Common combination
        (-12.0, -3.0),   # Another combination
        (-12.0, 40.0),   
        (40.0, -3.0),
        (40.0, 40.0),
        (-3.0, 0.0),
        (0.0, -3.0),
        (40.0, 0.0),
    ]
    
    print(f"{'Primary':<10} {'Secondary':<10} {'Classification':<20} {'In Range?':<10}")
    print("-" * 60)
    
    for primary, secondary in test_angles:
        label, description = classify_angiographic_view(primary, secondary)
        
        # Check which range it falls into
        in_range = "No"
        for view in ANGIOGRAPHIC_VIEWS[:-1]:  # Exclude "Other"
            p_range = view['primary_range']
            s_range = view['secondary_range']
            if (p_range[0] <= primary <= p_range[1] and 
                s_range[0] <= secondary <= s_range[1]):
                in_range = "Yes"
                break
        
        print(f"{primary:<10.1f} {secondary:<10.1f} {description:<20} {in_range:<10}")
    
    print("\n" + "=" * 80)
    print("Analysis:")
    print("=" * 80)
    print("Your actual data angles don't match the standard angiographic view ranges!")
    print("\nActual data has:")
    print("  Primary: -3.0, -12.0, 40.0")
    print("  Secondary: Similar limited values")
    print("\nBut classification expects:")
    print("  RAO: -60° to -15° (your -3.0, -12.0 are too small)")
    print("  LAO: 15° to 60° (your 40.0 fits here)")
    print("  AP: -15° to 15° (your -3.0 fits here)")
    
    print("\n" + "=" * 80)
    print("Suggested Fixes:")
    print("=" * 80)
    print("Option 1: Check if the angle columns are correct in your data")
    print("Option 2: Adjust the classification ranges to match your data")
    print("Option 3: Use a different classification scheme")
    print("Option 4: Map the limited values to view types directly")

def suggest_new_classification():
    """Suggest a new classification based on actual data patterns."""
    
    print("\n" + "=" * 80)
    print("Suggested New Classification for Your Data:")
    print("=" * 80)
    
    # Based on the actual values, create simpler classification
    print("""
    If your data has these specific discrete values, consider:
    
    Primary = -3.0:  Could be AP (near 0°)
    Primary = -12.0: Could be slight RAO
    Primary = 40.0:  Could be LAO
    
    Secondary = -3.0:  Straight (near 0°)
    Secondary = 40.0:  Cranial
    
    Simplified mapping:
    (-3, 40)   -> AP Cranial
    (-12, 40)  -> RAO Cranial  
    (40, 40)   -> LAO Cranial
    (-3, -3)   -> AP
    (-12, -3)  -> RAO Straight
    (40, -3)   -> LAO Straight
    """)

if __name__ == "__main__":
    test_with_actual_angles()
    suggest_new_classification()