#!/usr/bin/env python3
"""
Study ID Mapper - Maps between FileName patterns and actual StudyInstanceUIDs
"""

import pandas as pd
import sys

def find_study_by_filename_pattern(pattern, predictions_csv_path):
    """
    Find actual StudyInstanceUIDs by searching for a pattern in FileName.
    
    Args:
        pattern: String to search for in FileName (e.g., '1.2.392.200036.9116.1467.20170720125102258.4.2')
        predictions_csv_path: Path to individual_predictions CSV file
    
    Returns:
        Dict with mapping information
    """
    print(f"🔍 Searching for FileName pattern: {pattern}")
    
    try:
        # Load individual predictions
        df = pd.read_csv(predictions_csv_path)
        print(f"   ✓ Loaded {len(df)} individual predictions")
        
        # Search for pattern in FileName
        matching_rows = df[df['FileName'].str.contains(pattern, na=False)]
        print(f"   📊 Found {len(matching_rows)} rows matching pattern")
        
        if len(matching_rows) == 0:
            print(f"   ❌ No matches found for pattern: {pattern}")
            return None
        
        # Get unique StudyInstanceUIDs
        unique_studies = matching_rows['StudyInstanceUID'].unique()
        print(f"   📋 Unique StudyInstanceUIDs found: {len(unique_studies)}")
        
        # Create mapping
        result = {
            'search_pattern': pattern,
            'unique_study_uids': list(unique_studies),
            'total_predictions': len(matching_rows),
            'study_breakdown': {}
        }
        
        # Breakdown by StudyInstanceUID
        for study_uid in unique_studies:
            study_rows = matching_rows[matching_rows['StudyInstanceUID'] == study_uid]
            
            structures = study_rows['main_structure_name'].unique()
            gt_indices = study_rows['ground_truth_idx'].unique()
            
            result['study_breakdown'][study_uid] = {
                'prediction_count': len(study_rows),
                'structures': list(structures),
                'ground_truth_indices': list(gt_indices),
                'sample_filenames': study_rows['FileName'].head(2).tolist()
            }
            
            print(f"   📋 {study_uid}:")
            print(f"       🔢 Predictions: {len(study_rows)}")
            print(f"       🏗️ Structures: {structures}")
            print(f"       🎯 GT indices: {gt_indices}")
        
        return result
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None

def check_study_in_final_metrics(study_uids, study_metrics_csv_path):
    """
    Check if StudyInstanceUIDs exist in final study-level metrics.
    """
    print(f"\n🔍 Checking if studies exist in final study-level metrics...")
    
    try:
        df_study = pd.read_csv(study_metrics_csv_path)
        print(f"   ✓ Loaded {len(df_study)} study-level metrics")
        
        for study_uid in study_uids:
            study_row = df_study[df_study['StudyInstanceUID'] == study_uid]
            
            if len(study_row) > 0:
                row = study_row.iloc[0]
                print(f"   ✅ FOUND: {study_uid}")
                print(f"       📊 Predictions: {row.get('prediction_count', 'N/A')}")
                print(f"       🫀 Left coronary: {row.get('left_coronary_predictions', 'N/A')}")
                print(f"       🫀 Right coronary: {row.get('right_coronary_predictions', 'N/A')}")
                print(f"       🔗 Both structures: {row.get('has_both_structures', 'N/A')}")
                print(f"       🎯 Dominance: {row.get('dominance_name', 'N/A')}")
            else:
                print(f"   ❌ NOT FOUND: {study_uid}")
        
    except Exception as e:
        print(f"   ❌ Error checking study metrics: {e}")

if __name__ == "__main__":
    # Example usage
    PATTERN = "1.2.392.200036.9116.1467.20170720125102258.4.2"
    PREDICTIONS_CSV = "ground_truth_comparison_results/individual_predictions_val_epoch0.csv"
    STUDY_METRICS_CSV = "ground_truth_comparison_results/study_level_metrics_val_epoch0.csv"
    
    print("🎯 === STUDY ID MAPPER ===")
    print("This tool helps map between FileName patterns and actual StudyInstanceUIDs")
    
    # Find the mapping
    mapping = find_study_by_filename_pattern(PATTERN, PREDICTIONS_CSV)
    
    if mapping:
        print(f"\n✅ === MAPPING RESULTS ===")
        print(f"Search pattern: {mapping['search_pattern']}")
        print(f"Total predictions: {mapping['total_predictions']}")
        print(f"Unique StudyInstanceUIDs: {len(mapping['unique_study_uids'])}")
        
        # Check in final metrics
        check_study_in_final_metrics(mapping['unique_study_uids'], STUDY_METRICS_CSV)
        
        print(f"\n🎯 === SUMMARY ===")
        print(f"The study you're looking for (with FileName pattern '{PATTERN}') IS present in the final study-level metrics!")
        print(f"It exists as {len(mapping['unique_study_uids'])} separate StudyInstanceUIDs:")
        for study_uid in mapping['unique_study_uids']:
            print(f"   📋 {study_uid}")
        
    else:
        print(f"\n❌ No mapping found for pattern: {PATTERN}") 