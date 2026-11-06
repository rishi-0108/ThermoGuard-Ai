"""Read CSV files for analysis without modifying the source file."""

from pathlib import Path

import pandas as pd


class DatasetLoadError(RuntimeError):
    """Raised when a dataset cannot be read as a CSV file."""


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a CSV into memory without writing or repairing its source.

    Timestamp parsing and data checks are deliberately handled separately by
    :func:`thermoguard.data.validation.validate_dataset`.
    """
    source = Path(path)
    if not source.is_file():
        raise DatasetLoadError(f"Dataset file does not exist: {source}")

    try:
        return pd.read_csv(source)
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise DatasetLoadError(f"Could not read dataset CSV at {source}: {exc}") from exc
