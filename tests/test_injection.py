"""Offline tests for chronological splitting and seeded synthetic injection."""

import numpy as np
import pandas as pd
import pytest

from thermoguard.data.injection import inject_anomaly
from thermoguard.data.splitting import chronological_split


@pytest.fixture
def sample_frame() -> pd.DataFrame:
    row = np.arange(40, dtype=float)
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=len(row), freq="h").astype(str),
            "HUFL": row + 1,
            "HULL": row + 2,
            "MUFL": row + 3,
            "MULL": row + 4,
            "LUFL": row + 5,
            "LULL": row + 6,
            "OT": 20 + np.sin(row / 2),
        }
    )


@pytest.fixture
def split(sample_frame: pd.DataFrame):
    return chronological_split(
        sample_frame,
        train_fraction=0.5,
        validation_fraction=0.25,
        test_fraction=0.25,
    )


def test_chronological_split_is_contiguous_and_preserves_source_indices(sample_frame: pd.DataFrame) -> None:
    split = chronological_split(
        sample_frame,
        train_fraction=0.5,
        validation_fraction=0.25,
        test_fraction=0.25,
    )
    assert (len(split.train), len(split.validation), len(split.test)) == (20, 10, 10)
    assert split.train.index.tolist() == list(range(20))
    assert split.validation.index.tolist() == list(range(20, 30))
    assert split.test.index.tolist() == list(range(30, 40))
    assert split.train.date.iloc[-1] < split.validation.date.iloc[0]
    assert split.validation.date.iloc[-1] < split.test.date.iloc[0]
    pd.testing.assert_frame_equal(sample_frame, pd.concat([split.train, split.validation, split.test]))


@pytest.mark.parametrize(
    ("anomaly_type", "parameters"),
    [
        ("gradual_drift", {"magnitude": 4.0}),
        ("sudden_spike", {"magnitude": -9.0}),
        ("variance_change", {"variance_multiplier": 2.5}),
    ],
)
def test_injection_labels_inclusive_boundaries_and_keeps_original(
    anomaly_type: str, parameters: dict, split
) -> None:
    source = split.validation.copy(deep=True)
    result = inject_anomaly(
        split.validation,
        anomaly_type=anomaly_type,
        segment="validation",
        start_position=2,
        duration=5,
        seed=314,
        reference_values=split.train["OT"],
        **parameters,
    )
    assert result.data["anomaly_label"].tolist() == [0, 0, 1, 1, 1, 1, 1, 0, 0, 0]
    assert result.data["OT"].equals(source["OT"])
    assert result.data["OT_injected"].iloc[2:7].notna().all()
    assert np.isfinite(result.data["OT_injected"].to_numpy()).all()
    assert result.event.start_row_index == int(source.index[2])
    assert result.event.end_row_index == int(source.index[6])
    assert result.event.start_timestamp == source.date.iloc[2]
    assert result.event.end_timestamp == source.date.iloc[6]
    assert result.event.anomaly_type == anomaly_type
    assert result.event.affected_feature == "OT"
    assert result.event.seed == 314
    pd.testing.assert_frame_equal(split.validation, source)


def test_gradual_drift_is_linear_and_ends_at_configured_magnitude(split) -> None:
    result = inject_anomaly(
        split.validation,
        anomaly_type="gradual_drift",
        segment="validation",
        start_position=1,
        duration=4,
        magnitude=8.0,
        seed=7,
    )
    offsets = result.data["OT_injected"].iloc[1:5].to_numpy() - split.validation["OT"].iloc[1:5].to_numpy()
    np.testing.assert_allclose(offsets, [2.0, 4.0, 6.0, 8.0])


def test_variance_change_increases_event_variance(split) -> None:
    result = inject_anomaly(
        split.validation,
        anomaly_type="variance_change",
        segment="validation",
        start_position=1,
        duration=6,
        variance_multiplier=3.0,
        reference_values=split.train["OT"],
        seed=88,
    )
    clean_variance = split.validation["OT"].iloc[1:7].var(ddof=1)
    injected_variance = result.data["OT_injected"].iloc[1:7].var(ddof=1)
    assert injected_variance > clean_variance


def test_fixed_seed_repeats_identically_and_different_seeds_change_noise(split) -> None:
    kwargs = {
        "anomaly_type": "variance_change",
        "segment": "test",
        "start_position": 1,
        "duration": 8,
        "variance_multiplier": 2.0,
        "reference_values": split.train["OT"],
    }
    first = inject_anomaly(split.test, seed=23, **kwargs)
    repeat = inject_anomaly(split.test, seed=23, **kwargs)
    other_seed = inject_anomaly(split.test, seed=24, **kwargs)
    pd.testing.assert_series_equal(first.data["OT_injected"], repeat.data["OT_injected"])
    assert not first.data["OT_injected"].equals(other_seed.data["OT_injected"])
    assert first.event.seed != other_seed.event.seed


def test_clean_training_period_is_not_modified_by_heldout_injection(split) -> None:
    clean_train = split.train.copy(deep=True)
    clean_validation = split.validation.copy(deep=True)
    inject_anomaly(
        split.validation,
        anomaly_type="gradual_drift",
        segment="validation",
        start_position=0,
        duration=3,
        magnitude=2.0,
        seed=55,
    )
    pd.testing.assert_frame_equal(split.train, clean_train)
    pd.testing.assert_frame_equal(split.validation, clean_validation)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"segment": "train", "start_position": 0, "duration": 2, "magnitude": 1.0},
        {"segment": "test", "start_position": -1, "duration": 2, "magnitude": 1.0},
        {"segment": "test", "start_position": 9, "duration": 2, "magnitude": 1.0},
        {"segment": "test", "start_position": 0, "duration": 0, "magnitude": 1.0},
        {"segment": "test", "start_position": 0, "duration": 2, "magnitude": float("nan")},
    ],
)
def test_invalid_spike_parameters_are_rejected(split, kwargs: dict) -> None:
    with pytest.raises(ValueError):
        inject_anomaly(
            split.test,
            anomaly_type="sudden_spike",
            seed=1,
            **kwargs,
        )


def test_invalid_variance_parameters_are_rejected(split) -> None:
    with pytest.raises(ValueError, match="greater than 1"):
        inject_anomaly(
            split.test,
            anomaly_type="variance_change",
            segment="test",
            start_position=0,
            duration=5,
            seed=1,
            variance_multiplier=1.0,
            reference_values=split.train["OT"],
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [("start_position", 1.5), ("seed", "not-an-integer")],
)
def test_wrong_injection_parameter_types_raise_type_error(split, field: str, value) -> None:
    kwargs = {
        "anomaly_type": "sudden_spike",
        "segment": "test",
        "start_position": 0,
        "duration": 2,
        "magnitude": 1.0,
        "seed": 1,
    }
    kwargs[field] = value
    with pytest.raises(TypeError):
        inject_anomaly(split.test, **kwargs)
    with pytest.raises(ValueError, match="reference_values"):
        inject_anomaly(
            split.test,
            anomaly_type="variance_change",
            segment="test",
            start_position=0,
            duration=5,
            seed=1,
            variance_multiplier=2.0,
        )


def test_variance_event_cannot_cross_segment_boundary(split) -> None:
    with pytest.raises(ValueError, match="fit inside"):
        inject_anomaly(
            split.test,
            anomaly_type="sudden_spike",
            segment="test",
            start_position=8,
            duration=3,
            magnitude=2.0,
            seed=4,
        )
