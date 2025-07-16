import ast
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path
import hashlib
import json
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data():
    """Load all required datasets with error handling."""
    try:
        df_inference = pd.read_csv('post_pci_inference_2017-2024.csv')
        # df_inference = pd.read_csv('diagnostic_inference_2017-2024.csv')
        df_ground_truth = pd.read_csv('reports/processed_dataset.csv')
        
        logger.info(f"Loaded datasets: inference({len(df_inference)}), ground_truth({len(df_ground_truth)})")
        return df_inference, df_ground_truth
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def get_pred_columns(df):
    """Extract columns containing '_pre' in their names."""
    columns_oi = [col for col in df.columns.tolist() if '_pred' in col]
    logger.info(f"Found {len(columns_oi)} _pred columns: {columns_oi}")
    return columns_oi

def create_filename_to_study_mapping(df_ground_truth):
    """Create a mapping from filename to StudyInstanceUID for faster lookups."""
    mapping = df_ground_truth.set_index('FileName')['StudyInstanceUID'].to_dict()
    logger.info(f"Created filename to StudyInstanceUID mapping with {len(mapping)} entries")
    return mapping

def process_data_optimized(df_inference, df_ground_truth, columns_oi):
    """Process data using vectorized operations where possible."""
    # Create a copy of the ground truth dataframe
    df_ground_truth_updated = df_ground_truth.copy()
    
    logger.info(f"Processing data - Original shape: {df_ground_truth.shape}")
    logger.info(f"Processing data - Updated shape: {df_ground_truth_updated.shape}")
    
    # Check if any columns were lost during copy
    original_cols = set(df_ground_truth.columns)
    updated_cols = set(df_ground_truth_updated.columns)
    lost_cols = original_cols - updated_cols
    if lost_cols:
        logger.warning(f"Columns lost during copy: {lost_cols}")
    
    # Create filename to StudyInstanceUID mapping
    filename_to_study = create_filename_to_study_mapping(df_ground_truth)
    
    # Track statistics
    processed_count = 0
    not_found_count = 0
    error_count = 0
    
    # Process each row in post_pci dataframe
    for idx, row in tqdm(df_inference.iterrows(), total=len(df_inference), desc="Processing rows"):
        try:
            # Parse video names safely
            videos = ast.literal_eval(row.video_name)
            if not videos:
                logger.warning(f"Empty video list at row {idx}")
                continue
                
            # Get study UID using the first video
            first_video = videos[0]
            study_uid = filename_to_study.get(first_video)
            
            if study_uid is None:
                not_found_count += 1
                if not_found_count <= 5:  # Log first 5 missing entries
                    logger.warning(f"StudyInstanceUID not found for video: {first_video}")
                continue
            
            # Find matching rows in ground truth
            mask = df_ground_truth_updated.StudyInstanceUID == study_uid
            
            if mask.any():
                # Update all pre-PCI columns at once using vectorized operation
                df_ground_truth_updated.loc[mask, columns_oi] = row[columns_oi].values
                processed_count += 1
            else:
                not_found_count += 1
                logger.warning(f"No matching StudyInstanceUID found: {study_uid}")
                
        except (ValueError, SyntaxError) as e:
            error_count += 1
            logger.error(f"Error parsing video_name at row {idx}: {e}")
        except Exception as e:
            error_count += 1
            logger.error(f"Unexpected error at row {idx}: {e}")
    
    # Log final statistics
    logger.info(f"Processing complete:")
    logger.info(f"  - Successfully processed: {processed_count}")
    logger.info(f"  - Not found: {not_found_count}")
    logger.info(f"  - Errors: {error_count}")
    logger.info(f"  - Final shape: {df_ground_truth_updated.shape}")
    
    # Check if any columns were lost during processing
    final_cols = set(df_ground_truth_updated.columns)
    lost_during_processing = original_cols - final_cols
    if lost_during_processing:
        logger.warning(f"Columns lost during processing: {lost_during_processing}")
    
    return df_ground_truth_updated

def save_results(df_updated, output_path):
    """Save the updated dataframe with error handling."""
    try:
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert all columns to string to avoid parquet type issues
        logger.info("Converting data types for parquet compatibility...")
        df_to_save = df_updated.copy()
        
        # Convert all columns to string to avoid mixed type issues
        for col in df_to_save.columns:
            df_to_save[col] = df_to_save[col].astype(str)
        
        logger.info(f"Attempting to save as parquet: {output_path}")
        df_to_save.to_parquet(output_path)
        logger.info(f"Updated dataset saved to '{output_path}'")
        
        # Print summary statistics
        logger.info(f"Final dataset shape: {df_to_save.shape}")
        logger.info(f"Columns with pre-PCI data: {[col for col in df_to_save.columns if '_pre' in col]}")
        
    except Exception as e:
        logger.error(f"Error saving as parquet: {e}")
        logger.info("Attempting to save as CSV instead...")
        
        try:
            # Fallback to CSV
            csv_path = output_path.replace('.parquet', '.csv')
            df_updated.to_csv(csv_path, index=False)
            logger.info(f"Dataset saved as CSV: {csv_path}")
            
            # Print summary statistics
            logger.info(f"Final dataset shape: {df_updated.shape}")
            logger.info(f"Columns with pre-PCI data: {[col for col in df_updated.columns if '_pre' in col]}")
            
        except Exception as csv_error:
            logger.error(f"Error saving as CSV: {csv_error}")
            raise

def anonymize_with_tracking(df_original, mapping_filepath='reports/anonymization_mapping_post_pci.json'):
    """Anonymize dataset by hashing sensitive values while creating detailed tracking mapping."""
    
    logger.info("Starting anonymization process...")
    logger.info(f"Original dataset shape: {df_original.shape}")
    logger.info(f"Original columns: {len(df_original.columns)}")
    
    # Create copy for anonymization
    df_anon = df_original.copy()
    
    # High priority columns to anonymize (not remove)
    high_priority_anonymize = [
        'CathReport_MRN', 'Patient_ID', 'IDPATIENT', 'PatientName', 
        'PatientBirthDate', 'PatientBirthTime',
        'InstitutionName', 'InstitutionAddress', 'StationName', 
        'DeviceSerialNumber', 'HOSPITAL', 'SALLE', 'NSEJOUR',
        'DICOMPath', 'FileName', 'StudyInstanceUID', 'SeriesInstanceUID',
        'SOPInstanceUID', 'StudyID', 'NameOfPhysiciansReadingStudy',
        'PerformingPhysicianName', 'OperatorsName'
    ]
    
    logger.info(f"Columns to anonymize: {len(high_priority_anonymize)}")
    logger.info(f"Anonymization columns: {high_priority_anonymize}")
    
    # Initialize detailed tracking
    anonymization_log = {
        'created_date': datetime.now().isoformat(),
        'total_records': len(df_original),
        'total_columns_original': len(df_original.columns),
        'columns_anonymized': [],
        'columns_kept': [],
        'value_mappings': {},
        'study_mappings': {},
        'patient_mappings': {},
        'column_anonymization_details': {}
    }
    
    logger.info("Processing columns for anonymization...")
    
    # Since we're only adding predictions and not dropping columns, 
    # all original columns should exist in df_anon
    logger.info(f"Original columns: {len(df_original.columns)}")
    logger.info(f"Anonymized dataframe columns: {len(df_anon.columns)}")
    
    # Track each column's anonymization
    for col in df_original.columns:
        if col in high_priority_anonymize:
            # Column will be anonymized (should always exist in df_anon since we don't drop columns)
            logger.info(f"Anonymizing column: {col}")
            
            anonymization_log['columns_anonymized'].append({
                'column_name': col,
                'action': 'anonymized',
                'reason': 'high_priority_identifier'
            })
            
            # Store original data type and sample values
            anonymization_log['column_anonymization_details'][col] = {
                'action': 'anonymized',
                'reason': 'high_priority_identifier',
                'original_data_type': str(df_original[col].dtype),
                'original_unique_values': df_original[col].nunique(),
                'original_sample_values': df_original[col].dropna().head(3).tolist(),
                'value_mappings': {}
            }
            
            logger.info(f"  - Original data type: {df_original[col].dtype}")
            logger.info(f"  - Unique values: {df_original[col].nunique()}")
            logger.info(f"  - Sample values: {df_original[col].dropna().head(3).tolist()}")
            
            # Create value mappings for this column
            unique_values = df_original[col].dropna().unique()
            logger.info(f"  - Creating mappings for {len(unique_values)} unique values...")
            
            for value in unique_values:
                if pd.notna(value) and str(value).strip() != '':
                    # Create anonymized value based on column type
                    if col in ['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID']:
                        anonymized_value = f"{col.split('_')[0]}_{hashlib.md5(str(value).encode()).hexdigest()[:16]}"
                    elif col in ['Patient_ID', 'IDPATIENT']:
                        anonymized_value = f"PATIENT_{hashlib.md5(str(value).encode()).hexdigest()[:16]}"
                    elif col in ['CathReport_MRN']:
                        anonymized_value = f"MRN_{hashlib.md5(str(value).encode()).hexdigest()[:16]}"
                    elif col in ['PatientName']:
                        anonymized_value = f"NAME_{hashlib.md5(str(value).encode()).hexdigest()[:16]}"
                    elif col in ['InstitutionName', 'HOSPITAL']:
                        anonymized_value = f"HOSP_{hashlib.md5(str(value).encode()).hexdigest()[:16]}"
                    elif col in ['PatientBirthDate']:
                        # Keep year only for birth dates
                        try:
                            date_obj = pd.to_datetime(value)
                            anonymized_value = f"YEAR_{date_obj.year}"
                        except:
                            anonymized_value = f"DATE_{hashlib.md5(str(value).encode()).hexdigest()[:16]}"
                    else:
                        # Generic anonymization for other columns
                        anonymized_value = f"{col.upper()}_{hashlib.md5(str(value).encode()).hexdigest()[:16]}"
                    
                    # Store the mapping
                    anonymization_log['column_anonymization_details'][col]['value_mappings'][anonymized_value] = str(value)
                    anonymization_log['value_mappings'][f"{col}_{anonymized_value}"] = str(value)
            
            logger.info(f"  - Created {len(anonymization_log['column_anonymization_details'][col]['value_mappings'])} value mappings")
            
            # Apply anonymization to the column
            logger.info(f"  - Applying anonymization to column {col}...")
            df_anon[col] = df_anon[col].apply(lambda x: anonymize_value(x, col, anonymization_log['column_anonymization_details'][col]['value_mappings']))
            logger.info(f"  - Column {col} anonymization complete")
            
        else:
            # Column was kept unchanged (all non-anonymized columns)
            logger.debug(f"Keeping column unchanged: {col}")
            anonymization_log['columns_kept'].append({
                'column_name': col,
                'action': 'kept',
                'reason': 'safe_for_research'
            })
    
    logger.info("Creating study mappings...")
    
    # Create detailed study mappings
    study_count = 0
    for idx, study_uid in enumerate(df_original['StudyInstanceUID'].unique()):
        if pd.notna(study_uid):
            study_count += 1
            if study_count % 1000 == 0:  # Log progress every 1000 studies
                logger.info(f"  - Processed {study_count} studies...")
                
            # Get anonymized study ID
            anonymized_study_id = anonymize_value(study_uid, 'StudyInstanceUID', 
                                                anonymization_log['column_anonymization_details']['StudyInstanceUID']['value_mappings'])
            
            # Get all original values for this study
            study_data = df_original[df_original['StudyInstanceUID'] == study_uid]
            original_values = {}
            
            for col in high_priority_anonymize:
                if col in df_original.columns:
                    original_values[col] = study_data[col].iloc[0] if len(study_data) > 0 else None
            
            anonymization_log['study_mappings'][anonymized_study_id] = {
                'original_study_uid': study_uid,
                'original_values': original_values,
                'anonymized_study_id': anonymized_study_id
            }
    
    logger.info(f"Created {len(anonymization_log['study_mappings'])} study mappings")
    
    # Create patient mappings if Patient_ID exists
    if 'Patient_ID' in df_original.columns:
        logger.info("Creating patient mappings...")
        patient_count = 0
        for patient_id in df_original['Patient_ID'].unique():
            if pd.notna(patient_id):
                patient_count += 1
                if patient_count % 1000 == 0:  # Log progress every 1000 patients
                    logger.info(f"  - Processed {patient_count} patients...")
                    
                anonymized_patient_id = anonymize_value(patient_id, 'Patient_ID',
                                                      anonymization_log['column_anonymization_details']['Patient_ID']['value_mappings'])
                anonymization_log['patient_mappings'][anonymized_patient_id] = {
                    'original_patient_id': patient_id,
                    'anonymized_patient_id': anonymized_patient_id
                }
        
        logger.info(f"Created {len(anonymization_log['patient_mappings'])} patient mappings")
    else:
        logger.info("No Patient_ID column found, skipping patient mappings")
    
    # Save detailed mapping
    logger.info(f"Saving anonymization mapping to: {mapping_filepath}")
    save_mapping(anonymization_log, mapping_filepath)
    
    # Log anonymization summary
    logger.info(f"Anonymization Summary:")
    logger.info(f"  - Original columns: {len(df_original.columns)}")
    logger.info(f"  - Anonymized columns: {len(anonymization_log['columns_anonymized'])}")
    logger.info(f"  - Columns kept: {len(anonymization_log['columns_kept'])}")
    logger.info(f"  - Studies mapped: {len(anonymization_log['study_mappings'])}")
    logger.info(f"  - Value mappings created: {len(anonymization_log['value_mappings'])}")
    logger.info(f"  - Final dataset shape: {df_anon.shape}")
    
    logger.info("Anonymization process completed successfully!")
    
    return df_anon, anonymization_log

def anonymize_value(value, column_name, value_mappings):
    """Anonymize a single value based on column type and existing mappings."""
    
    if pd.isna(value) or str(value).strip() == '':
        return value
    
    # Check if we already have a mapping for this value
    for anonymized_val, original_val in value_mappings.items():
        if str(original_val) == str(value):
            return anonymized_val
    
    # Create new mapping if not found
    if column_name in ['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID']:
        return f"{column_name.split('_')[0]}_{hashlib.md5(str(value).encode()).hexdigest()[:16]}"
    elif column_name in ['Patient_ID', 'IDPATIENT']:
        return f"PATIENT_{hashlib.md5(str(value).encode()).hexdigest()[:16]}"
    elif column_name in ['CathReport_MRN']:
        return f"MRN_{hashlib.md5(str(value).encode()).hexdigest()[:16]}"
    elif column_name in ['PatientName']:
        return f"NAME_{hashlib.md5(str(value).encode()).hexdigest()[:16]}"
    elif column_name in ['InstitutionName', 'HOSPITAL']:
        return f"HOSP_{hashlib.md5(str(value).encode()).hexdigest()[:16]}"
    elif column_name in ['PatientBirthDate']:
        # Keep year only for birth dates
        try:
            date_obj = pd.to_datetime(value)
            return f"YEAR_{date_obj.year}"
        except:
            return f"DATE_{hashlib.md5(str(value).encode()).hexdigest()[:16]}"
    else:
        return f"{column_name.upper()}_{hashlib.md5(str(value).encode()).hexdigest()[:16]}"

def check_data_types(df, stage_name="dataset"):
    """Check and log data types for debugging."""
    logger.info(f"Data type check for {stage_name}:")
    logger.info(f"  - Shape: {df.shape}")
    logger.info(f"  - Columns: {len(df.columns)}")
    
    # Check for mixed data types
    mixed_type_columns = []
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check if object column has mixed types
            sample_values = df[col].dropna().head(10)
            if len(sample_values) > 0:
                types = set(type(val).__name__ for val in sample_values)
                if len(types) > 1:
                    mixed_type_columns.append(col)
                    logger.warning(f"    - Column '{col}' has mixed types: {types}")
    
    if mixed_type_columns:
        logger.warning(f"  - Found {len(mixed_type_columns)} columns with mixed data types")
    else:
        logger.info("  - No mixed data types detected")
    
    return mixed_type_columns

def recover_original_value(anonymized_value, column_name, mapping_filepath='reports/anonymization_mapping_post_pci.json'):
    """Recover original value for a specific anonymized value."""
    
    with open(mapping_filepath, 'r') as f:
        mapping = json.load(f)
    
    if column_name in mapping['column_anonymization_details']:
        value_mappings = mapping['column_anonymization_details'][column_name]['value_mappings']
        if anonymized_value in value_mappings:
            return value_mappings[anonymized_value]
    
    return None

def recover_original_info(anonymized_study_id, mapping_filepath='reports/anonymization_mapping_post_pci.json'):
    """Recover original information for a specific study ID with detailed column info."""
    
    with open(mapping_filepath, 'r') as f:
        mapping = json.load(f)
    
    if anonymized_study_id in mapping['study_mappings']:
        study_info = mapping['study_mappings'][anonymized_study_id]
        
        print(f"Recovered information for {anonymized_study_id}:")
        print(f"  - Original Study UID: {study_info['original_study_uid']}")
        print(f"  - Original values:")
        
        for col, value in study_info['original_values'].items():
            print(f"    - {col}: {value}")
        
        return study_info
    else:
        print(f"No mapping found for {anonymized_study_id}")
        return None

def verify_anonymization_integrity(df_original, df_anonymized, mapping):
    """Verify that anonymization was done correctly with detailed reporting."""
    
    print("Anonymization Integrity Check:")
    print("=" * 50)
    
    # Check column anonymization
    anonymized_columns = [item['column_name'] for item in mapping['columns_anonymized'] if item['action'] == 'anonymized']
    print(f"Columns anonymized: {len(anonymized_columns)}")
    for col in anonymized_columns[:5]:  # Show first 5
        print(f"  - {col}")
    if len(anonymized_columns) > 5:
        print(f"  ... and {len(anonymized_columns) - 5} more")
    
    # Check that anonymized columns are still present but with different values
    missing_columns = [col for col in anonymized_columns if col not in df_anonymized.columns]
    if missing_columns:
        print(f"❌ ERROR: {len(missing_columns)} anonymized columns missing from anonymized dataset")
        for col in missing_columns:
            print(f"  - {col}")
    else:
        print("✅ All anonymized columns present in anonymized dataset")
    
    # Check value changes
    print(f"\nValue Anonymization Check:")
    for col in anonymized_columns[:3]:  # Check first 3 columns
        if col in df_original.columns and col in df_anonymized.columns:
            original_unique = df_original[col].nunique()
            anonymized_unique = df_anonymized[col].nunique()
            print(f"  - {col}: {original_unique} unique values → {anonymized_unique} unique values")
    
    # Check study mappings
    original_studies = set(df_original['StudyInstanceUID'].unique())
    mapped_studies = set(mapping['study_mappings'].keys())
    
    print(f"\nStudy Mapping:")
    print(f"  - Original studies: {len(original_studies)}")
    print(f"  - Mapped studies: {len(mapped_studies)}")
    print(f"  - Mapping coverage: {len(mapped_studies) / len(original_studies) * 100:.1f}%")
    
    # Check for missing mappings
    missing_studies = original_studies - set([mapping['study_mappings'][k]['original_study_uid'] for k in mapping['study_mappings']])
    if missing_studies:
        print(f"⚠️  Missing mappings for {len(missing_studies)} studies")
    else:
        print("✅ All studies properly mapped")
    
    # Dataset size comparison
    print(f"\nDataset Comparison:")
    print(f"  - Original shape: {df_original.shape}")
    print(f"  - Anonymized shape: {df_anonymized.shape}")
    print(f"  - Columns preserved: {df_original.shape[1] == df_anonymized.shape[1]}")
    
    return len(missing_columns) == 0 and len(missing_studies) == 0

def save_mapping(mapping, filepath):
    """Save the mapping to a secure location."""
    with open(filepath, 'w') as f:
        json.dump(mapping, f, indent=2, default=str)
    print(f"Mapping saved to: {filepath}")

def get_anonymization_report(mapping_filepath='reports/anonymization_mapping_post_pci.json'):
    """Generate a detailed anonymization report."""
    
    with open(mapping_filepath, 'r') as f:
        mapping = json.load(f)
    
    print("Anonymization Report:")
    print("=" * 50)
    print(f"Created: {mapping['created_date']}")
    print(f"Total records: {mapping['total_records']}")
    print(f"Original columns: {mapping['total_columns_original']}")
    
    print(f"\nColumns Anonymized ({len(mapping['columns_anonymized'])}):")
    for item in mapping['columns_anonymized']:
        print(f"  - {item['column_name']}: {item['action']} ({item['reason']})")
    
    print(f"\nColumns Kept ({len(mapping['columns_kept'])}):")
    for item in mapping['columns_kept'][:10]:  # Show first 10
        print(f"  - {item['column_name']}")
    if len(mapping['columns_kept']) > 10:
        print(f"  ... and {len(mapping['columns_kept']) - 10} more")
    
    print(f"\nStudy Mappings: {len(mapping['study_mappings'])}")
    print(f"Patient Mappings: {len(mapping['patient_mappings'])}")
    print(f"Value Mappings: {len(mapping['value_mappings'])}")

def main():
    """Main function to orchestrate the data processing."""
    try:
        # Load data
        df_inference, df_ground_truth = load_data()
        
        # Check data types before processing
        check_data_types(df_ground_truth, "ground truth")
        
        # Get pre-PCI columns
        columns_oi = get_pred_columns(df_inference)
        
        if not columns_oi:
            logger.warning("No pre-PCI columns found!")
            return
        
        # Process data
        df_ground_truth_updated = process_data_optimized(df_inference, df_ground_truth, columns_oi)
        
        # Check data types after processing
        check_data_types(df_ground_truth_updated, "processed data")
        
        # Anonymize with detailed tracking
        df_anonymized, anonymization_log = anonymize_with_tracking(
            df_ground_truth_updated, 
            mapping_filepath='reports/anonymization_mapping_post_pci.json'
        )
        
        # Check data types after anonymization
        check_data_types(df_anonymized, "anonymized data")
        
        # Generate anonymization report
        get_anonymization_report('reports/anonymization_mapping_post_pci.json')
        
        # Verify integrity
        verify_anonymization_integrity(df_ground_truth_updated, df_anonymized, anonymization_log)
        
        # Show sample of results
        logger.info("Sample of anonymized data:")
        print(df_anonymized.head())
        
        # Save anonymized results
        output_path = 'reports/anonymized_post_pci_dataset_2017-2024.parquet'
        save_results(df_anonymized, output_path)
        
        logger.info("✅ Anonymization complete with detailed tracking mapping saved")
        
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        raise

if __name__ == "__main__":
    main()
    



