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

## No-PHI Input Schema

The released pipeline operates on de-identified, structured per-study tables.
The repository does **not** ship any clinical reports, raw DICOMs, PHI fields,
or model checkpoints. The schema below is what `generate_dataset.py` expects and
is exactly what a downstream user needs to reproduce the dataset construction
step. None of the listed fields contain protected health information once
DICOMs have been de-identified per the institutional pipeline; identifying
fields such as `PatientName`, `PatientBirthDate`, accession numbers, or MRNs
are not consumed by any code path in this directory.

### Required identifier columns (de-identified)
| Column | Type | Description |
| --- | --- | --- |
| `StudyInstanceUID` | string | De-identified pseudonymous study identifier. |
| `SeriesInstanceUID` | string | De-identified pseudonymous series identifier. |
| `FileName` | string | Path to the de-identified video/image file (`*.avi` or `*.mp4`). |
| `SeriesTime` | numeric | Acquisition time, used only for intra-study temporal ordering. |

### Required acquisition/classification columns
| Column | Type | Description |
| --- | --- | --- |
| `main_structure_class` | int 0-11 | Acquisition target (Left/Right Coronary, Graft, Catheter, etc.). Mapped via `MAIN_STRUCTURE_MAP`. |
| `contrast_agent_class` | int {0, 1} | Whether contrast injection was detected (1) or not (0). |
| `stent_presence_class` | int {0, 1} | Whether a stent is visible in the clip (1) or not (0). Drives the `diagnostic` / `PCI` / `POST_PCI` status assignment. |
| `dominance_class` | int {0, 1} | Coronary dominance (right=0, left=1). Mapped via `DOMINANCE_MAP`. |

### Structured per-vessel label columns
For each named vessel segment (`left_main`, `prox_lad`, `mid_lad`, `dist_lad`,
`D1`, `D2`, `prox_lcx`, `mid_lcx`, `dist_lcx`, `om1`, `om2`, `prox_rca`,
`mid_rca`, `dist_rca`, `pda`, `posterolateral`, `bx`, `lvp`, `lima_or_svg`),
the following per-vessel numeric or short-string fields are read by the report
generator. Each field is optional; missing values are skipped.

| Suffix | Description | Units / domain |
| --- | --- | --- |
| `*_stenosis` | Lumen narrowing percentage | 0 - 100 |
| `*_calcif` | Calcification severity | 0 = none, 1 = mild, 2 = moderate, 3 = severe |
| `*_cto` | Chronic total occlusion flag | 0 / 1 |
| `*_IFRHYPEREMIE` | Hyperemic instant wave-free ratio | 0.0 - 1.0 (or null) |
| `*_collateral` | Recipient vessel of collateral flow | short vessel code or null |
| `*_bifurcation` | Medina bifurcation classification | "1.1.0", "1.0.1", etc. |

### Optional free-text columns
| Column | Description |
| --- | --- |
| `Conclusion` | Optional procedural conclusion text (used by some downstream training pipelines, ignored by `generate_dataset.py`). |
| `Indications` | Optional indication free text (also ignored by `generate_dataset.py`). |

No additional clinical history, demographics, MRN, accession number, or
operator-identifying field is consumed by the released script.

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

## Upstream LLM Extraction of Structured Labels from Reports

The per-vessel structured columns above (`*_stenosis`, `*_calcif`, `*_cto`,
`*_IFRHYPEREMIE`, `*_collateral`, `*_bifurcation`) were extracted at the
originating institution from de-identified clinical procedure reports. The
extraction step is upstream of this directory and is not required to reproduce
training: any pipeline that populates the structured columns above is
sufficient.

The following template is a reproducible LLM extraction prompt. It is published
to allow third parties to reconstruct an equivalent labeling pipeline on their
own report corpus. It is **not** asserted to be byte-for-byte identical to the
original internal prompt used to generate the released MHI training labels;
that internal prompt was not retained as a version-controlled artifact at the
time of label generation. The template is functionally complete and was
specifically rewritten to operate on de-identified text only.

### Reproducible prompt template

```
SYSTEM: You are a cardiology assistant that extracts structured findings from
a de-identified coronary angiography report. Return only valid JSON matching
the schema below. If a vessel is not mentioned, set its `stenosis_percent`,
`calcification`, `cto`, `IFR_hyperemic`, `collateral_recipient`, and
`bifurcation_medina` fields to null. Do not invent findings. Do not include
any patient-identifying text in the output. Do not include free-text
commentary outside the JSON object.

USER: Extract findings for the following vessels: left_main, prox_lad,
mid_lad, dist_lad, D1, D2, prox_lcx, mid_lcx, dist_lcx, om1, om2, prox_rca,
mid_rca, dist_rca, pda, posterolateral, bx (ramus), lvp, lima_or_svg.

Each vessel object follows this schema:
{
  "stenosis_percent":     integer in [0, 100] | null,
  "calcification":        0 | 1 | 2 | 3 | null,
  "cto":                  true | false | null,
  "IFR_hyperemic":        number in [0.0, 1.0] | null,
  "collateral_recipient": short vessel code | null,
  "bifurcation_medina":   "X.Y.Z" string | null
}

Also extract:
{
  "coronary_dominance": "right" | "left" | "co-dominant" | null
}

The report text is delimited by <<< and >>>. Treat it as the sole source of
truth; do not use any external clinical knowledge to fill in missing
findings.

REPORT:
<<<
{de-identified report text}
>>>
```

Implementation notes:

* The original extraction was performed with a constrained-decoding wrapper
  that rejected outputs that failed schema validation and re-queried up to
  three times. Any modern instruction-tuned model with structured output
  support (JSON-mode or grammar-constrained decoding) can be substituted.
* Reports were de-identified upstream of this prompt; the prompt assumes no
  PHI is present and does not include any de-identification instructions.
* Outputs were flattened by the dataset-build step into the per-vessel
  columns documented above. A reference flattening helper is provided in
  `notebook_usage_example.py`.

## Validation Procedure

The released documentation describes a forward-looking validation protocol
rather than the historical internal validation summary, because the source
record for the originally reported 99/100 spot-check was not retained in a
form that could be released with the code.

### Recommended validation protocol

1. **Sampling.** Draw a stratified random sample of 100 de-identified reports
   from the corpus to be labeled. Stratify by (a) vessel-territory positivity
   for >=70% stenosis (RCA, LAD, LCx, branch / distal) and (b)
   procedural complexity (single-vessel, multi-vessel). Within each stratum
   sample proportionally to base rate, with a minimum of 5 reports per
   stratum.
2. **Reviewer setup.** Recruit at least one interventional or non-interventional
   cardiologist or trained reviewer who did not author the prompt template.
   Provide the reviewer with the de-identified report text and the
   extraction output side by side.
3. **Adjudication.** For each vessel x finding cell, the reviewer marks
   `agree`, `disagree`, or `ambiguous`. For numeric fields,
   `agree` is defined as within +/-10 percentage points (stenosis) or
   +/-0.05 (IFR) of the report text; categorical and binary fields require
   exact agreement.
4. **Reported metrics.** Report (a) per-field exact agreement rate, (b)
   per-field Cohen's kappa, (c) the dominant categories of disagreement,
   and (d) the fraction of records for which the entire structured output
   would have changed a downstream training label.
5. **Release artifacts.** Publish the anonymized agreement table and a
   short error analysis. Do not release the raw report text or the
   per-record reviewer judgments.

Until this protocol is executed against the new prompt template above, no
historical 99/100 validation result is claimed in either the repository
documentation or the manuscript.