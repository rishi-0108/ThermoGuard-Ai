"""Offline tests for loading and validation; no network access is used."""

import pandas as pd
import pytest

from thermoguard.data.loader import load_dataset
from thermoguard.data.validation import validate_dataset


@pytest.fixture
def valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3, freq="h").astype(str),
            "HUFL": [1.0, 2.0, 3.0],
            "HULL": [1.1, 2.1, 3.1],
            "MUFL": [1.2, 2.2, 3.2],
            "MULL": [1.3, 2.3, 3.3],
            "LUFL": [1.4, 2.4, 3.4],
            "LULL": [1.5, 2.5, 3.5],
            "OT": [20.0, 21.0, 22.0],
        }
    )


def test_valid_data_has_no_issues(valid_frame: pd.DataFrame) -> None:
    assert validate_dataset(valid_frame) == []


def test_loader_reads_temporary_csv_without_network(tmp_path, valid_frame: pd.DataFrame) -> None:
    path = tmp_path / "fixture.csv"
    valid_frame.to_csv(path, index=False)
    loaded = load_dataset(path)
    assert loaded.shape == valid_frame.shape
    assert path.exists()


def test_missing_column_is_reported(valid_frame: pd.DataFrame) -> None:
    issues = validate_dataset(valid_frame.drop(columns=["OT"]))
    assert any("Missing required columns: OT" in issue for issue in issues)


def test_malformed_timestamp_is_reported(valid_frame: pd.DataFrame) -> None:
    valid_frame.loc[1, "date"] = "not-a-timestamp"
    issues = validate_dataset(valid_frame)
    assert any("unparseable" in issue for issue in issues)


def test_duplicate_timestamp_is_reported(valid_frame: pd.DataFrame) -> None:
    valid_frame.loc[1, "date"] = valid_frame.loc[0, "date"]
    issues = validate_dataset(valid_frame)
    assert any("duplicate timestamp" in issue for issue in issues)


def test_irregular_interval_is_reported(valid_frame: pd.DataFrame) -> None:
    valid_frame.loc[2, "date"] = "2024-01-01 04:00:00"
    issues = validate_dataset(valid_frame)
    assert any("irregular interval" in issue for issue in issues)


def test_missing_value_is_reported(valid_frame: pd.DataFrame) -> None:
    valid_frame.loc[1, "OT"] = float("nan")
    issues = validate_dataset(valid_frame)
    assert any("Column 'OT' has 1 missing" in issue for issue in issues)


def test_nonfinite_numeric_value_is_reported(valid_frame: pd.DataFrame) -> None:
    valid_frame.loc[1, "OT"] = float("inf")
    issues = validate_dataset(valid_frame)
    assert any("OT' has 1 non-finite" in issue for issue in issues)


def test_nonnumeric_sensor_type_is_reported(valid_frame: pd.DataFrame) -> None:
    valid_frame["OT"] = valid_frame["OT"].astype(object)
    valid_frame.loc[1, "OT"] = "warm"
    issues = validate_dataset(valid_frame)
    assert any("Sensor column 'OT' must be numeric" in issue for issue in issues)
