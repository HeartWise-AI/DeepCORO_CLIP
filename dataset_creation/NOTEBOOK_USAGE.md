# Running Dataset Generation in Jupyter Notebook

This guide shows you how to use the `generate_dataset.py` script within a Jupyter notebook environment.

## Quick Start

### Method 1: Simple One-Step Processing

```python
# Cell 1: Setup
import sys
sys.path.append('dataset_creation')
from generate_dataset import process_dataset, create_default_config

# Cell 2: Process your data
input_csv = "/path/to/your/input.csv"      # Update this path
output_dir = "/path/to/your/output"        # Update this path
config = create_default_config()

process_dataset(input_csv, output_dir, config)
```

### Method 2: Step-by-Step with More Control

```python
# Cell 1: Imports
import sys
import pandas as pd
sys.path.append('dataset_creation')
from generate_dataset import *

# Cell 2: Load and explore data
df = load_data("/path/to/your/input.csv")
print(f"Loaded {len(df)} records")
df.head()

# Cell 3: Apply filters
config = create_default_config()
df_filtered = apply_hard_filters(df, config)
print(f"After filtering: {len(df_filtered)} records")

# Cell 4: Generate reports
df_with_reports = generate_reports(df_filtered)
print("Sample report:")
print(df_with_reports['Report'].iloc[0])

# Cell 5: Save results
df_with_reports.to_csv("/path/to/output/processed_dataset.csv", index=False)
```

## Complete Example

Here's a complete notebook example you can copy and modify:

```python
# ============================================================================
# Cell 1: Setup and Imports
# ============================================================================
import sys
import os
from pathlib import Path
import pandas as pd
import yaml

# Add dataset_creation to path
sys.path.append('dataset_creation')

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

# ============================================================================
# Cell 2: Configuration
# ============================================================================
# Update these paths for your data
INPUT_CSV = "/volume/data/your_dataset.csv"  # Your input file
OUTPUT_DIR = "/volume/processed_data"        # Where to save results

# Custom configuration
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

# ============================================================================
# Cell 3: Process Dataset (Option 1 - Simple)
# ============================================================================
# This does everything in one step
process_dataset(INPUT_CSV, OUTPUT_DIR, config)

# ============================================================================
# Cell 4: Manual Processing (Option 2 - Step by Step)
# ============================================================================
# Load data
df = load_data(INPUT_CSV)
print(f"Original dataset: {len(df)} records")

# Apply mappings if needed
if 'main_structure_class' in df.columns:
    df['main_structure_name'] = df['main_structure_class'].map(MAIN_STRUCTURE_MAP)
if 'dominance_class' in df.columns:
    df['dominance_name'] = df['dominance_class'].map(DOMINANCE_MAP)

# Sort by StudyInstanceUID and SeriesTime for proper temporal ordering
if 'StudyInstanceUID' in df.columns and 'SeriesTime' in df.columns:
    df = df.sort_values(['StudyInstanceUID', 'SeriesTime'])

# Apply filters
df_filtered = apply_hard_filters(df, config)
print(f"After filters: {len(df_filtered)} records")

# Generate reports
df_with_reports = generate_reports(df_filtered, coronary_specific=True)
print(f"Generated {len(df_with_reports)} reports")

# ============================================================================
# Cell 5: Explore Results
# ============================================================================
# View distribution
print("Status distribution:")
print(df_with_reports['status'].value_counts())

print("\nStructure distribution:")
print(df_with_reports['main_structure_name'].value_counts())

# View sample reports
print("\nSample Reports:")
for i in range(min(3, len(df_with_reports))):
    print(f"\n--- Sample {i+1} ---")
    print(f"File: {df_with_reports.iloc[i]['FileName']}")
    print(f"Structure: {df_with_reports.iloc[i]['main_structure_name']}")
    print(f"Report: {df_with_reports.iloc[i]['Report'][:200]}...")

# ============================================================================
# Cell 6: Create Samples and Save
# ============================================================================
# Create balanced samples
samples = sample_by_status(df_with_reports, n=9, label_col='status')

# Save main dataset
output_path = Path(OUTPUT_DIR)
output_path.mkdir(parents=True, exist_ok=True)

main_file = output_path / "processed_dataset.csv"
df_with_reports.to_csv(main_file, index=False)
print(f"Saved main dataset: {main_file}")

# Save samples
samples_dir = output_path / "samples"
samples_dir.mkdir(exist_ok=True)

for label, sample_df in samples.items():
    sample_file = samples_dir / f"sample_{label}.csv"
    sample_df.to_csv(sample_file, index=False)
    print(f"Saved {len(sample_df)} {label} samples: {sample_file}")
```

## Configuration Options

You can customize the processing by modifying the config dictionary:

```python
# Example configurations for different use cases

# Configuration 1: Only Left Coronary, more samples
left_only_config = {
    'filters': {
        'status': 'diagnostic',
        'main_structures': ['Left Coronary'],  # Only left side
        'contrast_agent_class': 1
    },
    'sampling': {
        'enabled': True,
        'n_per_group': 20,  # More samples
        'label_column': 'status'
    },
    'apply_mappings': True
}

# Configuration 2: All data, no sampling
comprehensive_config = {
    'filters': {
        'status': 'diagnostic',
        'main_structures': ['Left Coronary', 'Right Coronary'],
        'contrast_agent_class': 1
    },
    'report_settings': {
        'coronary_specific': False  # Full reports
    },
    'sampling': {
        'enabled': False  # Keep all data
    },
    'apply_mappings': True
}

# Configuration 3: PCI procedures only
pci_config = {
    'filters': {
        'status': 'PCI',  # Only PCI procedures
        'main_structures': ['Left Coronary', 'Right Coronary'],
        'contrast_agent_class': 1
    },
    'sampling': {
        'enabled': True,
        'n_per_group': 15,
        'label_column': 'status'
    },
    'apply_mappings': True,
    'assign_status': True
}

# Configuration 4: Post-PCI follow-up procedures
post_pci_config = {
    'filters': {
        'status': 'POST_PCI',  # Only post-PCI procedures
        'main_structures': ['Left Coronary', 'Right Coronary'],
        'contrast_agent_class': 1
    },
    'sampling': {
        'enabled': True,
        'n_per_group': 10,
        'label_column': 'status'
    },
    'apply_mappings': True,
    'assign_status': True
}
```

## Using Shell Commands

If you prefer to run the script as a command-line tool:

```python
# Cell 1: Create configuration file
config = create_default_config()
with open('my_config.yaml', 'w') as f:
    yaml.dump(config, f)

# Cell 2: Run script via shell
input_file = "/path/to/input.csv"
output_dir = "/path/to/output"

!cd dataset_creation && python generate_dataset.py \
    --input-csv {input_file} \
    --output-dir {output_dir} \
    --config ../my_config.yaml
```

## Common Issues and Solutions

### 1. Import Errors
```python
# If you get import errors, make sure the path is correct:
import sys
sys.path.append('dataset_creation')  # Adjust path as needed
```

### 2. Missing Dependencies
```python
# Install missing packages:
!pip install pandas numpy tqdm PyYAML
```

### 3. File Path Issues
```python
# Use absolute paths or check your current directory:
import os
print("Current directory:", os.getcwd())
print("Files in current directory:", os.listdir('.'))
```

### 4. Virtual Environment
```python
# If using virtual environment, activate it first:
!source .venv/bin/activate && python -c "import pandas; print('OK')"
```

## Output Structure

After running the script, you'll get:

```
output_directory/
├── processed_dataset.csv          # Main processed dataset with reports
├── processing_config.yaml         # Configuration used
└── samples/                       # Balanced samples (if enabled)
    ├── sample_diagnostic.csv
    └── sample_therapeutic.csv
```

## Next Steps

1. **Update file paths** in the examples above to match your data
2. **Choose a processing method** (simple one-step or detailed step-by-step)
3. **Customize the configuration** for your specific needs
4. **Run the cells** in order
5. **Check the output directory** for results

The processed dataset will include:
- Original data columns
- Generated medical reports in the `Report` column
- Mapped categorical variables
- Filtered records based on your criteria 