"""Non-mutating checks for the ETTh1 time-series data."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

EXPECTED_COLUMNS: tuple[str, ...] = (
    "date",
    "HUFL",
    "HULL",
    "MUFL",
    "MULL",
    "LUFL",
    "LULL",
    "OT",
)
TIMESTAMP_COLUMN = "date"
EXPECTED_INTERVAL = pd.Timedelta(1, unit="h")
SENSOR_COLUMNS: tuple[str, ...] = tuple(c for c in EXPECTED_COLUMNS if c != TIMESTAMP_COLUMN)


def validate_dataset(
    frame: pd.DataFrame,
    *,
    expected_columns: Sequence[str] = EXPECTED_COLUMNS,
    timestamp_column: str = TIMESTAMP_COLUMN,
    expected_interval: pd.Timedelta = EXPECTED_INTERVAL,
) -> list[str]:
    """Return every detected validation issue without changing ``frame``.

    Issues are human-readable and stable enough for reports/tests. No data is
    imputed, sorted, coerced in place, or otherwise repaired.
    """
    issues: list[str] = []
    missing_columns = [name for name in expected_columns if name not in frame.columns]
    if missing_columns:
        issues.append(f"Missing required columns: {', '.join(missing_columns)}")

    if timestamp_column not in frame.columns:
        issues.append(f"Timestamp column '{timestamp_column}' is missing; timestamp checks skipped.")
        timestamps = None
    else:
        timestamps = pd.to_datetime(frame[timestamp_column], errors="coerce")
        bad_timestamps = int(timestamps.isna().sum())
        if bad_timestamps:
            issues.append(
                f"Timestamp column '{timestamp_column}' has {bad_timestamps} missing or unparseable value(s)."
            )

    if timestamps is not None:
        if not timestamps.isna().any() and not timestamps.is_monotonic_increasing:
            issues.append(f"Timestamps in '{timestamp_column}' are not in chronological order.")

        valid_timestamps = timestamps.dropna()
        duplicate_count = int(valid_timestamps.duplicated().sum())
        if duplicate_count:
            issues.append(f"Found {duplicate_count} duplicate timestamp row(s).")

        deltas = timestamps.diff().dropna()
        if not deltas.empty:
            irregular = deltas[deltas != expected_interval]
            if not irregular.empty:
                counts = {str(interval): int(count) for interval, count in irregular.value_counts().items()}
                issues.append(
                    f"Expected {expected_interval} timestamp spacing; found irregular interval(s): {counts}."
                )

    for column in frame.columns:
        missing_count = int(frame[column].isna().sum())
        if missing_count:
            issues.append(f"Column '{column}' has {missing_count} missing value(s).")

    for column in expected_columns:
        if column == timestamp_column or column not in frame.columns:
            continue
        series = frame[column]
        if not is_numeric_dtype(series.dtype):
            issues.append(f"Sensor column '{column}' must be numeric; found dtype '{series.dtype}'.")
            continue
        nonmissing_values = series.dropna().to_numpy(dtype=float)
        nonfinite_count = int((~np.isfinite(nonmissing_values)).sum())
        if nonfinite_count:
            issues.append(f"Sensor column '{column}' has {nonfinite_count} non-finite value(s).")

    return issues
