from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


def read_csv_with_fallback(
    csv_path: str | Path,
    *,
    expected_columns: Optional[Iterable[str]] = None,
    seps: tuple[Optional[str], ...] = ("α", ",", "\t", None),
) -> pd.DataFrame:
    """
    Load a CSV file trying a list of candidate separators until the expected columns appear.

    Args:
        csv_path: Path to the CSV.
        expected_columns: Optional iterable of column names that must exist.
        seps: Candidate separators to try, ending with None (pandas default autodetect).

    Returns:
        pandas.DataFrame containing the parsed CSV.
    """
    path = Path(csv_path).expanduser()
    expected = {col for col in (expected_columns or []) if col}
    last_df: pd.DataFrame | None = None
    last_exc: Exception | None = None

    for sep in seps:
        try:
            if sep is None:
                df = pd.read_csv(path)
            else:
                df = pd.read_csv(path, sep=sep, engine="python")
        except Exception as exc:
            last_exc = exc
            continue

        last_df = df
        if expected:
            if expected.issubset(df.columns):
                return df
        else:
            # No expected columns provided: stop when we clearly split into multiple columns.
            if df.shape[1] > 1:
                return df

    if expected and last_df is not None:
        missing = expected - set(last_df.columns)
        raise KeyError(
            f"Missing expected columns {sorted(missing)} in '{path}'. "
            f"Tried separators {seps}."
        )

    if last_df is None and last_exc is not None:
        raise last_exc

    return last_df if last_df is not None else pd.DataFrame()
