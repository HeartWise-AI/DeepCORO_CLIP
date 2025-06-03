# Dataset Creation

This directory contains scripts and utilities for generating medical datasets from coronary angiography data.

## Files

- `generate_dataset.py` - Main script for dataset generation
- `config_template.yaml` - Configuration template with all available options
- `README.md` - This documentation file

## Overview

The `generate_dataset.py` script consolidates the dataset generation logic from the Jupyter notebook into a reusable, configurable Python script. It performs the following operations:

1. **Data Loading** - Loads data from CSV or Parquet files
2. **Mapping** - Maps categorical variables to human-readable names
3. **Status Assignment** - Automatically assigns procedure status based on PCI timing
4. **Filtering** - Applies configurable filters to the dataset
5. **Report Generation** - Creates detailed medical reports for coronary vessel analysis
6. **Sampling** - Creates balanced samples for training/testing
7. **Output** - Saves processed datasets and configuration files

## Installation

Make sure you have the required dependencies installed:

```bash
# Activate the virtual environment
source .venv/bin/activate

# Install required packages (if not already installed via pyproject.toml)
pip install pandas numpy tqdm PyYAML
```

## Usage

### Basic Usage

```bash
# Navigate to the dataset_creation directory
cd dataset_creation

# Run with minimal arguments (uses default configuration)
python generate_dataset.py --input-csv /path/to/input.csv --output-dir /path/to/output

# Run with custom configuration
python generate_dataset.py --input-csv /path/to/input.csv --output-dir /path/to/output --config config.yaml
```

### Create Default Configuration

```bash
# Generate a default configuration file
python generate_dataset.py --create-default-config
```

This creates a `default_config.yaml` file that you can customize.

### Example Usage

```bash
# Example with real paths
python generate_dataset.py \
    --input-csv /volume/data/merged_predictions.csv \
    --output-dir /volume/processed_datasets/coronary_dataset_v1 \
    --config config_template.yaml
```

## Configuration

The script uses YAML configuration files to control processing. See `config_template.yaml` for all available options.

### Key Configuration Sections

#### Filters
```yaml
filters:
  status: "diagnostic"
  main_structures:
    - "Left Coronary"
    - "Right Coronary"
  contrast_agent_class: 1
```

#### Report Settings
```yaml
report_settings:
  coronary_specific: true  # Generate side-specific reports
```

#### Sampling
```yaml
sampling:
  enabled: true
  n_per_group: 9
  label_column: "status"
```

## Input Data Format

The script expects input data with the following key columns:

### Required Columns
- `main_structure_class` - Main coronary structure class (0-11, will be mapped to names)
- `contrast_agent_class` - Contrast agent classification (0/1)
- `FileName` - Path to associated video/image file
- `StudyInstanceUID` - Unique identifier for each study
- `SeriesTime` - Time of series acquisition (for temporal ordering)

### Optional Columns for Status Assignment
- `stent_presence_class` - Stent presence classification (0/1) for PCI status assignment
- `dominance_class` - Coronary dominance class (0/1, will be mapped to names)

### Optional Columns (for report generation)
- Stenosis columns: `leftmain_stenosis`, `prox_lad_stenosis`, etc.
- Calcification columns: `leftmain_calcif`, `prox_lad_calcif`, etc.
- IFR columns: `leftmain_IFRHYPEREMIE`, `prox_lad_IFRHYPEREMIE`, etc.
- CTO columns: `leftmain_cto`, `prox_lad_cto`, etc. (1 = CTO present, 0 = no CTO)
- Collateral columns: `leftmain_collateral`, `prox_lad_collateral`, etc. (vessel name or code)
- Bifurcation columns: `leftmain_bifurcation`, `prox_lad_bifurcation`, etc. (Medina classification)
- `dominance_name` - Coronary dominance information
- `Conclusion` - Clinical conclusion text

## Output Structure

The script creates the following output structure:

```
output_dir/
├── processed_dataset.csv          # Main processed dataset
├── processing_config.yaml         # Configuration used for processing
└── samples/                       # Sample datasets (if sampling enabled)
    ├── sample_diagnostic.csv
    └── sample_therapeutic.csv
```

## Generated Reports

The script generates detailed medical reports for each record, including:

- **Stenosis Assessment** - Degree of vessel narrowing (mild, moderate, severe, critical)
- **CTO Assessment** - Chronic Total Occlusion identification (100% blocked vessels)
- **Calcification Description** - Presence and severity of calcifications
- **IFR Values** - Instant wave-free ratio measurements
- **Bifurcation Lesions** - Medina classification for bifurcation involvement
- **Collateral Circulation** - Identification of vessels providing collateral flow
- **Coronary Dominance** - Left or right dominance patterns

### Example Report
```
the Left Main Coronary Artery (LMCA) has moderate stenosis (~55%), and bifurcation lesion (Medina 1.1.0).
the proximal LAD has mild stenosis (~20.0%), and bifurcation lesion (Medina 1.1.0).
the mid LAD is 100% blocked and is a CTO, and moderate calcifications.
the distal LAD has no significant stenosis.
D1 branch has no significant stenosis.
the distal RCA gives collaterals to the LAD.
The coronary circulation is right dominant.
```

## Procedure Status Assignment

The script automatically assigns procedure status based on PCI (Percutaneous Coronary Intervention) timing:

- **diagnostic**: Pure diagnostic procedures with no stent placement
- **PCI**: Current procedure involves stent placement
- **POST_PCI**: Follow-up procedure after previous PCI in the same study/artery

This classification is based on:
- `stent_presence_class` column (1 = stent present, 0 = no stent)
- `StudyInstanceUID` for grouping procedures within the same study
- `main_structure_name` for tracking procedures within the same coronary artery
- Temporal ordering within studies

## Medical Context

This script is designed for processing coronary angiography data with focus on:

- **Diagnostic vs Interventional** procedures (PCI vs non-PCI)
- **Left vs Right coronary** systems
- **Contrast-enhanced** procedures only
- **Stenosis quantification** across vessel segments
- **Calcification assessment** 
- **Functional assessment** via IFR
- **Temporal relationships** between procedures in the same study

## Vessel Mapping

The script includes comprehensive mapping of coronary vessel segments:

### Left Coronary System
- Left Main (LMCA)
- Left Anterior Descending (LAD) - proximal, mid, distal
- Diagonal branches (D1, D2)
- Left Circumflex (LCX) - proximal, distal
- Obtuse Marginal branches (OM1, OM2)

### Right Coronary System
- Right Coronary Artery (RCA) - proximal, mid, distal
- Posterior Descending Artery (PDA)
- Posterolateral branches

### Dominance-Dependent Vessels
The script handles coronary dominance patterns:
- **Right Dominant**: PDA and posterolateral from RCA
- **Left Dominant**: PDA and posterolateral from LCX

## Error Handling

The script includes comprehensive error handling for:
- Missing input files
- Invalid configuration files
- Missing required columns
- Data type mismatches
- Output directory creation failures

## Logging

The script provides detailed logging at INFO level, including:
- Data loading progress
- Filter application results
- Report generation progress
- Sampling statistics
- Output file locations

## Performance Considerations

- Large datasets are processed with progress bars via `tqdm`
- Memory usage is optimized through selective column processing
- Sampling reduces output size for balanced datasets
- Configuration caching avoids repeated YAML parsing

## Extending the Script

To add new functionality:

1. **New Filters**: Add filter logic in `apply_hard_filters()`
2. **New Report Elements**: Extend `create_report()` function
3. **New Output Formats**: Add format options in `process_dataset()`
4. **New Vessel Types**: Update vessel mapping dictionaries

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   pip install pandas numpy tqdm PyYAML
   ```

2. **File Not Found Errors**
   - Check input file paths
   - Ensure output directory is writable

3. **Configuration Errors**
   - Validate YAML syntax
   - Check configuration keys match template

4. **Memory Issues with Large Files**
   - Process in chunks for very large datasets
   - Reduce sampling size if needed

### Debug Mode

For debugging, modify the logging level:
```python
logging.basicConfig(level=logging.DEBUG)
```

## Integration with Main Project

This script integrates with the main DeepCORO_CLIP project:

- Uses same data formats as training pipelines
- Compatible with video processing utilities
- Follows project coding standards
- Uses project dependency management (pyproject.toml) 