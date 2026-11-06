"""Chronological train/validation/test splitting for time-series frames."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ChronologicalSplit:
    """Copies of the clean chronological segments, retaining source row indices."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def chronological_split(
    frame: pd.DataFrame,
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    timestamp_column: str = "date",
) -> ChronologicalSplit:
    """Split an ordered frame into clean contiguous periods without mutating it.

    Fractions are converted to boundaries using floor for the train and
    validation endpoints; all remaining rows go to test. Original index labels
    are retained so event metadata can identify zero-based source CSV rows.
    """
    fractions = (train_fraction, validation_fraction, test_fraction)
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in fractions):
        raise ValueError("Split fractions must be numeric values.")
    if any(value <= 0 or value >= 1 for value in fractions):
        raise ValueError("Each split fraction must be greater than 0 and less than 1.")
    if not abs(sum(fractions) - 1.0) <= 1e-9:
        raise ValueError("Train, validation, and test fractions must sum to 1.0.")
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("A non-empty pandas DataFrame is required.")
    if timestamp_column not in frame.columns:
        raise ValueError(f"Timestamp column '{timestamp_column}' is missing.")

    timestamps = pd.to_datetime(frame[timestamp_column], errors="coerce")
    if timestamps.isna().any():
        raise ValueError(f"Timestamp column '{timestamp_column}' contains missing or unparseable values.")
    if not timestamps.is_monotonic_increasing:
        raise ValueError("Input must already be in chronological order; it will not be sorted.")
    if timestamps.duplicated().any():
        raise ValueError("Input contains duplicate timestamps; it will not be repaired.")

    row_count = len(frame)
    train_end = int(row_count * train_fraction)
    validation_end = int(row_count * (train_fraction + validation_fraction))
    if train_end < 1 or validation_end - train_end < 1 or row_count - validation_end < 1:
        raise ValueError("The dataset is too small for the requested three non-empty splits.")

    return ChronologicalSplit(
        train=frame.iloc[:train_end].copy(deep=True),
        validation=frame.iloc[train_end:validation_end].copy(deep=True),
        test=frame.iloc[validation_end:].copy(deep=True),
    )
