"""Generate a JSON profile for a local ETTh1 CSV."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from thermoguard.data.loader import DatasetLoadError, load_dataset
from thermoguard.data.validation import (
    EXPECTED_COLUMNS,
    EXPECTED_INTERVAL,
    TIMESTAMP_COLUMN,
    validate_dataset,
)

SOURCE_URL = "https://github.com/zhouhaoyi/ETDataset"
LICENSE_NOTE = (
    "The ETDataset repository currently states CC BY-ND 4.0. The repository-level license does not clearly enumerate "
    "which assets it covers, so its applicability to the CSV assets remains uncertain. This project does not claim "
    "redistribution rights; do not redistribute the source CSV or derived data without clarification."
)
INDIA_TIMEZONE = timezone(timedelta(hours=5, minutes=30))


def _safe_number(value: Any) -> int | float | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return int(number) if number.is_integer() else number


def sha256_file(path: str | Path) -> str:
    """Calculate SHA-256 over the exact on-disk bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_profile(path: str | Path, *, access_date: str | None = None) -> dict[str, Any]:
    """Build a data-dependent profile without changing the source CSV."""
    source = Path(path)
    frame = load_dataset(source)
    issues = validate_dataset(frame)

    if TIMESTAMP_COLUMN in frame.columns:
        timestamps = pd.to_datetime(frame[TIMESTAMP_COLUMN], errors="coerce")
        parsed_timestamps = timestamps.dropna()
        start = parsed_timestamps.min().isoformat(sep=" ") if not parsed_timestamps.empty else None
        end = parsed_timestamps.max().isoformat(sep=" ") if not parsed_timestamps.empty else None
        interval_counts = {
            str(interval): int(count)
            for interval, count in timestamps.diff().dropna().value_counts().sort_index().items()
        }
        duplicate_count = int(parsed_timestamps.duplicated().sum())
    else:
        timestamps = None
        start = end = None
        interval_counts = {}
        duplicate_count = 0

    missing_counts = {str(column): int(count) for column, count in frame.isna().sum().items()}
    numeric_summary: dict[str, dict[str, int | float | None]] = {}
    for column in frame.columns:
        if column == TIMESTAMP_COLUMN or not pd.api.types.is_numeric_dtype(frame[column].dtype):
            continue
        values = frame[column].to_numpy(dtype=float, na_value=np.nan)
        finite_values = values[np.isfinite(values)]
        if finite_values.size:
            summary = {
                "count": int(finite_values.size),
                "mean": _safe_number(np.mean(finite_values)),
                "std": _safe_number(np.std(finite_values, ddof=1)) if finite_values.size > 1 else None,
                "min": _safe_number(np.min(finite_values)),
                "25%": _safe_number(np.percentile(finite_values, 25)),
                "50%": _safe_number(np.percentile(finite_values, 50)),
                "75%": _safe_number(np.percentile(finite_values, 75)),
                "max": _safe_number(np.max(finite_values)),
            }
        else:
            summary = {key: None for key in ("count", "mean", "std", "min", "25%", "50%", "75%", "max")}
        numeric_summary[str(column)] = summary

    return {
        "dataset": {
            "source_url": SOURCE_URL,
            "source_file_url": f"{SOURCE_URL}/blob/main/ETT-small/ETTh1.csv",
            "filename": source.name,
            "access_date": access_date or datetime.now(INDIA_TIMEZONE).date().isoformat(),
            "license_note": LICENSE_NOTE,
        },
        "validation": {
            "is_valid": not issues,
            "issues": issues,
            "expected_columns": list(EXPECTED_COLUMNS),
            "expected_interval": str(EXPECTED_INTERVAL),
        },
        "profile": {
            "row_count": len(frame),
            "columns": [str(column) for column in frame.columns],
            "start_timestamp": start,
            "end_timestamp": end,
            "timestamp_interval_counts": interval_counts,
            "missing_value_counts": missing_counts,
            "duplicate_timestamp_rows": duplicate_count,
            "numeric_summary_finite_values": numeric_summary,
            "sha256": sha256_file(source),
            "file_size_bytes": int(source.stat().st_size),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate and profile a local ETTh1 CSV.")
    parser.add_argument("--input", type=Path, default=Path("data/raw/ETTh1.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports/results/ETTh1_profile.json"))
    args = parser.parse_args(argv)

    try:
        profile = create_profile(args.input)
    except DatasetLoadError as exc:
        parser.error(str(exc))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profile, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(profile, indent=2, allow_nan=False))
    return 0 if profile["validation"]["is_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
