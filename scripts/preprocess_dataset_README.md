# preprocess_dataset — input spec

Preprocesses a coronary angiography dataset for the Orion/validation pipeline. Used in `scripts/external_validation.py`.

## Required columns

| Column              | Description                    | Renamed to        |
|---------------------|--------------------------------|-------------------|
| `ss_patient_id`     | Patient identifier             | `Patient_ID`      |
| `ss_event_cath_id`  | Cath study / event identifier  | `StudyInstanceUID`|

These two must be present for the renames to apply. All other columns are optional; only those present in your CSV are processed.

## Study-Level Granularity

The validation input is **study-level**, not purely patient-level.

- A single patient (`ss_patient_id`) can have **multiple** `ss_event_cath_id` values.
- Each `ss_event_cath_id` represents one study / cath event and is handled independently downstream.
- If a patient has multiple studies, the ground truth may differ between studies.
- Within a single study, all DICOMs belonging to that study should share the same ground truth.
- In practice, this means you may have multiple rows with the same `ss_patient_id`, but different `ss_event_cath_id` values and different `DICOMPath` values.
- If you expand one study into multiple DICOM rows, duplicate the same study-level labels across all rows from that study.

## Optional columns

### Stenosis (regression)

Any column whose name ends with `_stenosis` is treated as a stenosis percentage (0–100). The preprocessor adds a derived column `{name}_binary` (1 if value > 70, else 0).

Standard set (see `utils/external_validation_constants.REGRESSION_COLUMNS`):

- `prox_rca_stenosis`, `mid_rca_stenosis`, `dist_rca_stenosis`, `pda_stenosis`, `posterolateral_stenosis`
- `left_main_stenosis`, `prox_lad_stenosis`, `mid_lad_stenosis`, `dist_lad_stenosis`, `D1_stenosis`, `D2_stenosis`
- `prox_lcx_stenosis`, `mid_lcx_stenosis`, `dist_lcx_stenosis`, `om1_stenosis`, `om2_stenosis`
- `bx_stenosis`, `lvp_stenosis`

**Format:** numeric 0–100 (integer or float).

### Binary — calcification (`*_calcif_binary`)

**Format:** one of `none`, `mild`, `moderate`, `severe`.  
Mapped to 0 (none) or 1 (mild/moderate/severe).

Standard columns: 18 segments per category (same as stenosis, including `bx` and `lvp`). See `BINARY_COLUMNS` in `utils/external_validation_constants.py`.

### Binary — CTO (`*_cto`) and thrombus (`*_thrombus`)

**Format:** boolean `True` / `False`.  
Mapped to 1 / 0. If your CSV has strings `"True"`/`"False"` or 0/1, convert to bool before calling `preprocess_dataset` (or the mapping will not apply).

## Other behavior

- **Empty columns** are detected and dropped (with a message printed).
- Column order in the CSV does not matter.

## CSV template

Use `preprocess_dataset_template.csv` as a header + one example row. Fill or delete columns as needed; the preprocessor only uses columns that exist in the dataframe.
