#!/usr/bin/env python3
"""
Merge contrast injection metadata and XA radiation metrics with the Cath report.

The injection (`resutls_inj.csv`) and XA (`results_xa.csv`) files contain rows
identified by `accession_number`. Each CSV column is comma-separated, but any
given column can contain pipe-separated values that must be expanded so that
every accession number occupies its own row. The expanded tables are merged,
summarised, and finally joined onto the study-level Cath report parquet file so
that each study instance is guaranteed to carry contrast metadata.

Example:
    python scripts/merge_contrast_with_reports.py \\
        --injection-csv data/resutls_inj.csv \\
        --xa-csv data/results_xa.csv \\
        --report-parquet /mediadata1/.../2b_CathReport_HEMO_MHI_MERGED_2017-2024_STUDY_LEVEL.parquet \\
        --output-parquet data/report_with_contrast.parquet \\
        --output-csv data/report_with_contrast.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pandas as pd

COLUMN_DELIMITER_CANDIDATES: Tuple[str, ...] = (",", "|", ";", "\t", "α")
VALUE_DELIMITER = "|"
ACCESSION_COLUMN_CANDIDATES: Tuple[str, ...] = (
    "accession_number",
    "AccessionNumber",
    "ACC_NUM",
    "Num Accession",
    "Num_Accession",
    "num_accession",
    "NUM_ACCESSION",
    "StudyInstanceUID",
    "studyinstanceuid",
    "STUDYINSTANCE",
    "studyinstance",
)
VOLUME_PATTERN = re.compile(r"([-+]?\d+(?:[.,]\d+)?)")


def detect_delimiter(file_path: Path, fallback: str = ",") -> str:
    """Infer the column delimiter by scanning the first non-empty header line."""
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            counts = [(delim, stripped.count(delim)) for delim in COLUMN_DELIMITER_CANDIDATES]
            counts.sort(key=lambda item: item[1], reverse=True)
            if counts and counts[0][1] > 0:
                return counts[0][0]
            break
    return fallback


def read_flat_file(
    path: Path, column_delimiter: Optional[str] = None, value_delimiter: str = VALUE_DELIMITER
) -> pd.DataFrame:
    """Load a CSV-like file and expand any pipe-delimited values."""
    delimiter = column_delimiter or detect_delimiter(path)
    df = pd.read_csv(
        path,
        sep=delimiter,
        dtype=str,
        keep_default_na=False,
        engine="python",
    )

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
        df[col] = df[col].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "NONE": pd.NA})

    expanded = expand_pipe_delimited_values(df, value_delimiter=value_delimiter)
    return expanded.reset_index(drop=True)


def expand_pipe_delimited_values(df: pd.DataFrame, value_delimiter: str = VALUE_DELIMITER) -> pd.DataFrame:
    """Explode any column that contains pipe-delimited values."""
    if df.empty:
        return df.copy()

    object_columns = df.select_dtypes(include="object").columns.tolist()
    if not object_columns:
        return df.copy()

    contains_delimiter = any(
        df[col].astype(str).str.contains(value_delimiter, regex=False).any()
        for col in object_columns
    )
    if not contains_delimiter:
        return df.copy()

    columns = df.columns.tolist()
    expanded_rows: List[dict] = []

    for row_values in df.itertuples(index=False, name=None):
        row_dict = dict(zip(columns, row_values))
        per_column_values = {}
        has_multi_value = False

        for col, value in row_dict.items():
            if isinstance(value, str) and value_delimiter in value:
                parts = [part.strip() for part in value.split(value_delimiter)]
                cleaned = [part if part else pd.NA for part in parts]
                if not cleaned:
                    cleaned = [pd.NA]
                per_column_values[col] = cleaned
                if len(cleaned) > 1:
                    has_multi_value = True
            else:
                per_column_values[col] = [value]

        if not has_multi_value:
            expanded_rows.append(row_dict)
            continue

        max_len = max(len(values) for values in per_column_values.values())
        for col, values in per_column_values.items():
            if len(values) == 1 and max_len > 1:
                per_column_values[col] = values * max_len
            elif len(values) < max_len:
                filler = pd.NA if len(values) > 1 else values[0]
                per_column_values[col] = values + [filler] * (max_len - len(values))

        for idx in range(max_len):
            expanded_rows.append({col: per_column_values[col][idx] for col in columns})

    return pd.DataFrame(expanded_rows, columns=columns)


def normalize_accession(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip().str.upper()
    normalized = normalized.replace({"": pd.NA, "NAN": pd.NA, "NONE": pd.NA})
    return normalized


def _combine_unique(series: pd.Series) -> Optional[str]:
    values = {str(value).strip() for value in series if isinstance(value, str) and value.strip()}
    return "|".join(sorted(values)) if values else None


def summarize_injection_data(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["accession_number"])

    if "accession_number" not in df.columns:
        raise ValueError("Injection CSV does not contain 'accession_number'.")

    df = df.copy()
    df["accession_number"] = normalize_accession(df["accession_number"])
    df = df.dropna(subset=["accession_number"])

    if "dose" in df.columns:
        df["inj_contrast_ml"] = df["dose"].apply(_extract_volume_ml)
    else:
        df["inj_contrast_ml"] = pd.NA

    if "date_heure_injection" in df.columns:
        df["inj_date_time"] = pd.to_datetime(df["date_heure_injection"], errors="coerce")
    else:
        df["inj_date_time"] = pd.NaT

    grouped = df.groupby("accession_number", dropna=False)
    summary = grouped.agg(
        inj_event_count=("accession_number", "size"),
        inj_total_contrast_ml=("inj_contrast_ml", "sum"),
        inj_min_contrast_ml=("inj_contrast_ml", "min"),
        inj_max_contrast_ml=("inj_contrast_ml", "max"),
        inj_first_injection_time=("inj_date_time", "min"),
        inj_last_injection_time=("inj_date_time", "max"),
    ).reset_index()

    for source_col, target_col in [
        ("type_produit", "inj_products"),
        ("code_produit", "inj_product_codes"),
        ("desc_produit", "inj_product_descriptions"),
    ]:
        if source_col in df.columns:
            summary = summary.merge(
                grouped[source_col].apply(_combine_unique).rename(target_col),
                on="accession_number",
                how="left",
            )

    return summary


def _extract_volume_ml(value: object) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace(",", ".")
    match = VOLUME_PATTERN.search(text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def load_xa_data(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["accession_number"])
    if "accession_number" not in df.columns:
        raise ValueError("XA CSV does not contain 'accession_number'.")

    df = df.copy()
    df["accession_number"] = normalize_accession(df["accession_number"])
    df = df.dropna(subset=["accession_number"])

    renamed = df.rename(
        columns={col: (col if col == "accession_number" else f"xa_{col}") for col in df.columns}
    )
    value_columns = [col for col in renamed.columns if col.startswith("xa_")]
    if value_columns:
        renamed = renamed.sort_values(by=value_columns)
    deduped = renamed.groupby("accession_number", dropna=False)
    return deduped.first().reset_index()


def locate_accession_column(columns: Sequence[str]) -> Optional[str]:
    normalized_map = {col.lower().replace(" ", "_"): col for col in columns}
    for candidate in ACCESSION_COLUMN_CANDIDATES:
        key = candidate.lower().replace(" ", "_")
        if key in normalized_map:
            return normalized_map[key]

    for col in columns:
        if "accession" in col.lower():
            return col
    for col in columns:
        if "studyinstance" in col.lower():
            return col
    return None


def load_report_dataframe(report_path: Path) -> Tuple[pd.DataFrame, str]:
    report_df = pd.read_parquet(report_path)
    accession_column = locate_accession_column(report_df.columns)
    if not accession_column:
        raise ValueError(
            "Could not find an accession or StudyInstance column in the report parquet file."
        )

    report_df = report_df.copy()
    report_df["accession_number"] = normalize_accession(report_df[accession_column])
    return report_df, accession_column


def merge_datasets(
    report_df: pd.DataFrame, injection_df: pd.DataFrame, xa_df: pd.DataFrame
) -> pd.DataFrame:
    combined = pd.merge(
        injection_df,
        xa_df,
        on="accession_number",
        how="outer",
        sort=True,
    )
    merged = report_df.merge(combined, on="accession_number", how="left", sort=False)

    indicator_columns = [
        column
        for column in merged.columns
        if column.startswith("inj_") and column not in {"inj_event_count"}
    ]
    if not indicator_columns:
        indicator_columns = [column for column in merged.columns if column.startswith("xa_")]

    if indicator_columns:
        merged["contrast_data_available"] = merged[indicator_columns].notna().any(axis=1)
    else:
        merged["contrast_data_available"] = False
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--injection-csv",
        type=Path,
        default=Path("data/resutls_inj.csv"),
        help="Path to the injection CSV (default: data/resutls_inj.csv).",
    )
    parser.add_argument(
        "--xa-csv",
        type=Path,
        default=Path("data/results_xa.csv"),
        help="Path to the XA CSV (default: data/results_xa.csv).",
    )
    parser.add_argument(
        "--report-parquet",
        type=Path,
        default=Path(
            "/mediadata1/datasets/DeepCoro/2b_CathReport_HEMO_MHI_MERGED_2017-2024_STUDY_LEVEL.parquet"
        ),
        help="Cath report parquet file to enrich.",
    )
    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=Path("data/report_with_contrast.parquet"),
        help="Path where the merged parquet file will be written.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional path to also write the merged dataset as CSV.",
    )
    parser.add_argument(
        "--missing-output",
        type=Path,
        default=None,
        help="Optional CSV path listing studies without contrast metadata.",
    )
    parser.add_argument(
        "--injection-delimiter",
        type=str,
        default=None,
        help="Explicit column delimiter for the injection CSV (auto-detected by default).",
    )
    parser.add_argument(
        "--xa-delimiter",
        type=str,
        default=None,
        help="Explicit column delimiter for the XA CSV (auto-detected by default).",
    )
    parser.add_argument(
        "--value-delimiter",
        type=str,
        default=VALUE_DELIMITER,
        help="Delimiter used to expand multi-value cells (default: '|').",
    )
    return parser.parse_args()


def save_dataframe(df: pd.DataFrame, path: Optional[Path], fmt: str) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        df.to_parquet(path, index=False)
    elif fmt == "csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported format '{fmt}'")


def main() -> None:
    args = parse_args()

    injection_df = read_flat_file(args.injection_csv, column_delimiter=args.injection_delimiter, value_delimiter=args.value_delimiter)
    injection_summary = summarize_injection_data(injection_df)
    xa_df_raw = read_flat_file(args.xa_csv, column_delimiter=args.xa_delimiter, value_delimiter=args.value_delimiter)
    xa_df = load_xa_data(xa_df_raw)
    report_df, accession_column = load_report_dataframe(args.report_parquet)
    merged_df = merge_datasets(report_df, injection_summary, xa_df)

    total_studies = len(report_df)
    studies_with_contrast = int(merged_df["contrast_data_available"].sum())
    print(
        f"Merged dataset contains {total_studies} studies; "
        f"{studies_with_contrast} have associated contrast metadata."
    )
    if injection_summary.empty:
        print("Warning: no injection data was found after expansion.")
    else:
        print(f"Injection table covers {len(injection_summary)} accession numbers.")
    if xa_df.empty:
        print("Warning: no XA metadata rows were found.")

    save_dataframe(merged_df, args.output_parquet, fmt="parquet")
    save_dataframe(merged_df, args.output_csv, fmt="csv")

    if args.missing_output:
        missing = merged_df.loc[~merged_df["contrast_data_available"], ["accession_number"]]
        save_dataframe(missing, args.missing_output, fmt="csv")

    print(f"Accession numbers were extracted from '{accession_column}' in the Cath report.")
    print(f"Merged parquet saved to: {args.output_parquet}")
    if args.output_csv:
        print(f"Merged CSV saved to: {args.output_csv}")
    if args.missing_output:
        print(f"Missing-contrast report saved to: {args.missing_output}")


if __name__ == "__main__":
    main()
