"""Unit tests for the statistical control-limit baseline."""

import numpy as np
import pandas as pd
import pytest

from thermoguard.detectors.control_limits import ControlLimitError, StatisticalControlLimits


def training_series() -> pd.Series:
    return pd.Series([8.0, 10.0, 12.0], name="OT", index=[4, 5, 6])


def test_normal_observations_are_not_flagged_and_indices_align() -> None:
    detector = StatisticalControlLimits(k=1.0).fit(training_series())
    observations = pd.Series([9.0, 10.0, 11.0], name="OT", index=[20, 21, 22])
    prediction = detector.predict(observations)
    assert prediction.tolist() == [0, 0, 0]
    assert prediction.index.tolist() == observations.index.tolist()
    assert prediction.name == "control_limit_prediction"


def test_high_and_low_outliers_are_flagged() -> None:
    detector = StatisticalControlLimits(k=1.0).fit(training_series())
    observations = pd.Series([7.0, 13.0], name="OT", index=[50, 51])
    assert detector.predict(observations).tolist() == [1, 1]


def test_limit_boundaries_are_inclusive_and_remain_normal() -> None:
    detector = StatisticalControlLimits(k=1.0).fit(training_series())
    observations = pd.Series([8.0, 12.0], name="OT")
    assert detector.lower_limit_ == 8.0
    assert detector.upper_limit_ == 12.0
    assert detector.predict(observations).tolist() == [0, 0]


def test_score_returns_signed_standardized_deviation_and_is_deterministic() -> None:
    detector = StatisticalControlLimits(k=2.0).fit(training_series())
    observations = pd.Series([8.0, 12.0], name="OT", index=[2, 8])
    first = detector.score(observations)
    second = detector.score(observations)
    pd.testing.assert_series_equal(first, second)
    assert first.index.tolist() == [2, 8]
    np.testing.assert_allclose(first.to_numpy(), [-1.0, 1.0])


@pytest.mark.parametrize("k", [0, -1, float("nan"), float("inf")])
def test_invalid_k_is_rejected(k: float) -> None:
    with pytest.raises(ControlLimitError, match="k"):
        StatisticalControlLimits(k=k)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (pd.Series([], dtype=float, name="OT"), "empty"),
        (pd.Series([1.0], name="OT"), "At least two"),
        (pd.Series([1.0, 1.0], name="OT"), "zero variance"),
        (pd.Series([1.0, float("nan")], name="OT"), "missing"),
        (pd.Series([1.0, float("inf")], name="OT"), "non-finite"),
        (pd.Series(["1", "2"], name="OT"), "numeric"),
    ],
)
def test_invalid_training_values_are_rejected(values: pd.Series, message: str) -> None:
    with pytest.raises(ControlLimitError, match=message):
        StatisticalControlLimits().fit(values)


def test_training_rejects_labels_and_injected_values() -> None:
    with pytest.raises(ControlLimitError, match="clean"):
        StatisticalControlLimits().fit(
            pd.DataFrame({"OT": [1.0, 2.0], "OT_injected": [1.0, 2.0]})
        )
    with pytest.raises(ControlLimitError, match="label"):
        StatisticalControlLimits().fit(pd.Series([0, 1], name="anomaly_label"))


def test_prediction_rejects_invalid_or_label_input() -> None:
    detector = StatisticalControlLimits().fit(training_series())
    with pytest.raises(ControlLimitError, match="not fitted"):
        StatisticalControlLimits().predict(pd.Series([1.0], name="OT"))
    with pytest.raises(ControlLimitError, match="Ground-truth"):
        detector.predict(pd.DataFrame({"anomaly_label": [0, 1]}), value_column="anomaly_label")
    with pytest.raises(ControlLimitError, match="non-finite"):
        detector.predict(pd.Series([float("inf")], name="OT"))


def test_prediction_can_use_injected_column_without_reading_labels() -> None:
    detector = StatisticalControlLimits(k=1.0).fit(training_series())
    observations = pd.DataFrame(
        {"OT": [10.0], "OT_injected": [20.0], "anomaly_label": [1]}, index=[100]
    )
    prediction = detector.predict(observations, value_column="OT_injected")
    assert prediction.index.tolist() == [100]
    assert prediction.tolist() == [1]


def test_validation_and_test_values_do_not_change_fit_statistics() -> None:
    train = training_series()
    validation = pd.Series([10.0, 12.0], name="OT")
    test = pd.Series([11.0, 13.0], name="OT")
    first = StatisticalControlLimits(k=2.5).fit(train)
    validation[:] = 1_000_000.0
    test[:] = -1_000_000.0
    second = StatisticalControlLimits(k=2.5).fit(train)
    assert (first.mean_, first.std_, first.lower_limit_, first.upper_limit_) == (
        second.mean_, second.std_, second.lower_limit_, second.upper_limit_
    )
    assert (first.mean_, first.std_) == (10.0, 2.0)
