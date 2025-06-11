# Dataset Generation Notebook Usage Examples
# Copy these code cells into your Jupyter notebook

"""
Cell 1: Setup and Imports
"""
# Import required libraries
import sys
import os
from pathlib import Path
import pandas as pd
import yaml

# Add the dataset_creation directory to the path (if running from parent directory)
if 'dataset_creation' not in sys.path:
    sys.path.append('dataset_creation')

# Import the dataset generation functions
from generate_dataset import (
    create_default_config,
    load_data,
    apply_hard_filters,
    generate_reports,
    sample_by_status,
    process_dataset,
    MAIN_STRUCTURE_MAP,
    DOMINANCE_MAP
)

"""
Cell 2: Option A - Complete Processing in One Step
"""
# Define your input and output paths
input_csv_path = "/path/to/your/input.csv"  # Replace with your actual path
output_directory = "/path/to/your/output"   # Replace with your actual path

# Create configuration
config = {
    'filters': {
        'status': 'diagnostic',
        'main_structures': ['Left Coronary', 'Right Coronary'],
        'contrast_agent_class': 1
    },
    'report_settings': {
        'coronary_specific': True
    },
    'sampling': {
        'enabled': True,
        'n_per_group': 9,
        'label_column': 'status'
    },
    'apply_mappings': True
}

# Process the dataset
process_dataset(input_csv_path, output_directory, config)

"""
Cell 3: Option B - Step-by-Step Processing (More Control)
"""
# Step 1: Load your data
input_file = "/path/to/your/data.csv"  # Replace with your file path
df = load_data(input_file)
print(f"Loaded {len(df)} records")
df.head()

"""
Cell 4: Apply Mappings (if needed)
"""
# Apply mappings for categorical variables
if 'main_structure_class' in df.columns:
    df['main_structure_name'] = df['main_structure_class'].map(MAIN_STRUCTURE_MAP)
    
if 'dominance_class' in df.columns:
    df['dominance_name'] = df['dominance_class'].map(DOMINANCE_MAP)

# Sort by StudyInstanceUID and SeriesTime for proper temporal ordering
if 'StudyInstanceUID' in df.columns and 'SeriesTime' in df.columns:
    df = df.sort_values(['StudyInstanceUID', 'SeriesTime'])
    print("Sorted data by StudyInstanceUID and SeriesTime")

print("Applied mappings")
if 'main_structure_class' in df.columns:
    print("\nMain structure mapping:")
    print(df[['main_structure_class', 'main_structure_name']].value_counts())

"""
Cell 5: Apply Filters
"""
# Create or load configuration
config = create_default_config()  # or use your custom config

# Apply filters
df_filtered = apply_hard_filters(df, config)
print(f"Dataset filtered from {len(df)} to {len(df_filtered)} records")

# Show filter results
print(f"\nFilter summary:")
print(f"Status: {df_filtered['status'].unique()}")
print(f"Main structures: {df_filtered['main_structure_name'].unique()}")
print(f"Contrast agent class: {df_filtered['contrast_agent_class'].unique()}")

"""
Cell 6: Generate Reports
"""
# Generate medical reports
df_with_reports = generate_reports(df_filtered, coronary_specific=True)
print(f"Generated reports for {len(df_with_reports)} records")

# Display a sample report
if len(df_with_reports) > 0:
    print("\nSample Report:")
    print(f"File: {df_with_reports['FileName'].iloc[0]}")
    print(f"Structure: {df_with_reports['main_structure_name'].iloc[0]}")
    print(f"Report:\n{df_with_reports['Report'].iloc[0]}")
    
    # Show CTO, collateral, and bifurcation columns if they exist
    sample_row = df_with_reports.iloc[0]
    cto_cols = [col for col in df_with_reports.columns if col.endswith('_cto')]
    collateral_cols = [col for col in df_with_reports.columns if col.endswith('_collateral')]
    bifurcation_cols = [col for col in df_with_reports.columns if col.endswith('_bifurcation')]
    
    if cto_cols:
        print(f"\nCTO findings:")
        for col in cto_cols:
            if pd.notna(sample_row[col]) and sample_row[col] == 1:
                print(f"  {col}: {sample_row[col]} (CTO present)")
    
    if bifurcation_cols:
        print(f"\nBifurcation lesions:")
        for col in bifurcation_cols:
            if pd.notna(sample_row[col]) and sample_row[col] not in [0, 0.0, "", "0"]:
                print(f"  {col}: {sample_row[col]} (Medina classification)")
    
    if collateral_cols:
        print(f"\nCollateral circulation:")
        for col in collateral_cols:
            if pd.notna(sample_row[col]) and sample_row[col] not in [0, 0.0, "", "0"]:
                print(f"  {col}: {sample_row[col]}")

"""
Cell 7: Create Samples and Save Results
"""
# Create balanced samples
samples = sample_by_status(df_with_reports, n=9, label_col='status')

for label, sample_df in samples.items():
    print(f"\n{label}: {len(sample_df)} samples")
    print(sample_df[['FileName', 'main_structure_name', 'status']].head(3))

# Save results
output_dir = Path("/path/to/your/output")  # Replace with your path
output_dir.mkdir(parents=True, exist_ok=True)

# Save main dataset
main_output_path = output_dir / "processed_dataset.csv"
df_with_reports.to_csv(main_output_path, index=False)
print(f"\nSaved main dataset to {main_output_path}")

# Save samples
samples_dir = output_dir / "samples"
samples_dir.mkdir(exist_ok=True)

for label, sample_df in samples.items():
    sample_path = samples_dir / f"sample_{label}.csv"
    sample_df.to_csv(sample_path, index=False)
    print(f"Saved {len(sample_df)} {label} samples to {sample_path}")

"""
Cell 8: Using Shell Commands (Alternative Method)
"""
# If you prefer to use the script as a command-line tool from within the notebook:

# First, create a configuration file
custom_config = {
    'filters': {
        'status': 'diagnostic',
        'main_structures': ['Left Coronary', 'Right Coronary'],
        'contrast_agent_class': 1
    },
    'report_settings': {
        'coronary_specific': True
    },
    'sampling': {
        'enabled': True,
        'n_per_group': 15,
        'label_column': 'status'
    },
    'apply_mappings': True
}

# Save configuration
config_path = "custom_config.yaml"
with open(config_path, 'w') as f:
    yaml.dump(custom_config, f, default_flow_style=False)

# Run the script using shell commands
input_csv = "/path/to/your/input.csv"
output_dir = "/path/to/your/output"

# Note: Use ! to run shell commands in Jupyter
# !source ../.venv/bin/activate && cd dataset_creation && python generate_dataset.py \
#     --input-csv {input_csv} \
#     --output-dir {output_dir} \
#     --config {config_path}

"""
Cell 9: Exploratory Analysis After Processing
"""
# Load and explore the processed dataset
processed_df = pd.read_csv("/path/to/your/output/processed_dataset.csv")

print(f"Processed dataset shape: {processed_df.shape}")
print(f"\nColumns: {list(processed_df.columns)}")
print(f"\nStatus distribution:")
print(processed_df['status'].value_counts())
print(f"\nMain structure distribution:")
print(processed_df['main_structure_name'].value_counts())

# View sample reports
print("\nSample Reports:")
for i in range(min(3, len(processed_df))):
    print(f"\n--- Report {i+1} ---")
    print(f"File: {processed_df.iloc[i]['FileName']}")
    print(f"Structure: {processed_df.iloc[i]['main_structure_name']}")
    print(f"Report: {processed_df.iloc[i]['Report']}")

"""
Cell 10: Custom Configuration Examples
"""
# Example 1: Process only Left Coronary with more samples
left_coronary_config = {
    'filters': {
        'status': 'diagnostic',
        'main_structures': ['Left Coronary'],  # Only left coronary
        'contrast_agent_class': 1
    },
    'report_settings': {
        'coronary_specific': True
    },
    'sampling': {
        'enabled': True,
        'n_per_group': 20,  # More samples
        'label_column': 'status'
    },
    'apply_mappings': True
}

# Example 2: Process all structures with comprehensive reports
comprehensive_config = {
    'filters': {
        'status': 'diagnostic',
        'main_structures': ['Left Coronary', 'Right Coronary', 'Other'],
        'contrast_agent_class': 1
    },
    'report_settings': {
        'coronary_specific': False  # Comprehensive reports
    },
    'sampling': {
        'enabled': False  # No sampling, keep all data
    },
    'apply_mappings': True
}

# Use any of these configurations with process_dataset()
# process_dataset(input_csv_path, output_directory, left_coronary_config)

"""
Quick Usage Summary for Notebook:

1. Copy the import cell (Cell 1) to your notebook
2. Choose either:
   - Option A (Cell 2): One-step processing
   - Option B (Cells 3-7): Step-by-step with more control
3. Customize the configuration as needed
4. Replace file paths with your actual paths
5. Run the cells in order

Key points:
- Always activate your virtual environment first: source .venv/bin/activate
- Update file paths to match your data location
- Adjust configuration parameters as needed
- Check the output directory for results
""" 