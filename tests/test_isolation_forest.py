"""Offline tests for the Isolation Forest detector."""

import numpy as np
import pandas as pd
import pytest

from thermoguard.detectors.isolation_forest import IsolationForestError, TemperatureIsolationForest


def make_detector(**kwargs: object) -> TemperatureIsolationForest:
    kwargs.setdefault("n_estimators", 100)
    kwargs.setdefault("contamination", 0.05)
    return TemperatureIsolationForest(**kwargs)


def normal_training() -> pd.Series:
    return pd.Series(np.sin(np.arange(300) / 12.0), name="OT")


def test_clear_outliers_are_detected_and_normal_values_are_not() -> None:
    detector = make_detector(contamination=0.10).fit(normal_training())
    values = pd.Series([0.0, 0.2, -0.1, 120.0], name="OT", index=[9, 8, 7, 6])
    predicted = detector.predict(values)
    assert predicted.iloc[:3].tolist() == [0, 0, 0]
    assert predicted.iloc[3:].tolist() == [1]


def test_prediction_length_alignment_and_score_direction() -> None:
    detector = make_detector().fit(normal_training())
    observations = pd.DataFrame({"OT_injected": [0.0, 20.0]}, index=[101, 104])
    predictions = detector.predict(observations, value_column="OT_injected")
    scores = detector.score(observations, value_column="OT_injected")
    assert len(predictions) == len(observations)
    assert predictions.index.tolist() == [101, 104]
    assert scores.index.tolist() == [101, 104]
    assert predictions.name == "isolation_forest_prediction"
    assert predictions.tolist() == [0, 1]
    assert scores.iloc[1] > scores.iloc[0]


def test_prediction_rejects_missing_column_and_empty_input() -> None:
    detector = make_detector().fit(normal_training())
    with pytest.raises(IsolationForestError, match="missing value column"):
        detector.predict(pd.DataFrame({"wrong": [1.0]}), value_column="OT_injected")
    with pytest.raises(IsolationForestError, match="empty"):
        detector.predict(pd.Series([], dtype=float, name="OT"))


def test_fixed_seed_produces_deterministic_predictions_and_scores() -> None:
    data = normal_training()
    observations = pd.Series(np.r_[data.to_numpy(), 10.0, -10.0], name="OT")
    first = make_detector(random_state=23).fit(data)
    second = make_detector(random_state=23).fit(data)
    pd.testing.assert_series_equal(first.predict(observations), second.predict(observations))
    pd.testing.assert_series_equal(first.score(observations), second.score(observations))


def test_fit_state_depends_only_on_clean_training_and_uses_one_feature() -> None:
    training = normal_training()
    detector = make_detector().fit(training)
    assert detector.n_samples_ == len(training)
    assert detector.model_.n_features_in_ == 1
    validation = pd.DataFrame({"OT_injected": [9999.0], "anomaly_label": [1]})
    test = pd.DataFrame({"OT_injected": [-9999.0], "anomaly_label": [1]})
    # Held-out frames are not accepted by fit and cannot alter the fitted estimator.
    state_before = [tree.tree_.threshold.copy() for tree in detector.model_.estimators_]
    with pytest.raises(IsolationForestError, match="clean"):
        detector.fit(validation)
    with pytest.raises(IsolationForestError, match="clean"):
        detector.fit(test)
    state_after = [tree.tree_.threshold.copy() for tree in detector.model_.estimators_]
    for before, after in zip(state_before, state_after, strict=True):
        np.testing.assert_array_equal(before, after)


def test_labels_are_never_accepted_as_fit_or_prediction_feature() -> None:
    with pytest.raises(IsolationForestError, match="not a label"):
        make_detector(fit_feature="anomaly_label")
    with pytest.raises(IsolationForestError, match="no label"):
        make_detector().fit(pd.DataFrame({"OT": [1.0, 2.0], "anomaly_label": [0, 1]}))
    detector = make_detector().fit(normal_training())
    with pytest.raises(IsolationForestError, match="Ground-truth"):
        detector.predict(pd.DataFrame({"anomaly_label": [0, 1]}), value_column="anomaly_label")
    with pytest.raises(IsolationForestError, match="Ground-truth"):
        detector.predict(pd.Series([0, 1], name="anomaly_label"))


@pytest.mark.parametrize("data", [pd.DataFrame({"bad": [1.0, 2.0]}), pd.Series([], dtype=float, name="OT")])
def test_missing_column_and_empty_training_are_rejected(data: pd.DataFrame | pd.Series) -> None:
    with pytest.raises(IsolationForestError):
        make_detector().fit(data)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_training_and_prediction_values_are_rejected(bad: float) -> None:
    with pytest.raises(IsolationForestError, match="missing|non-finite"):
        make_detector().fit(pd.Series([1.0, 2.0, bad], name="OT"))
    detector = make_detector().fit(normal_training())
    with pytest.raises(IsolationForestError, match="missing|non-finite"):
        detector.predict(pd.Series([bad], name="OT"))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"contamination": 0}, {"contamination": 0.6}, {"contamination": "invalid"},
        {"n_estimators": 0}, {"random_state": 1.5}, {"max_features": 0},
        {"max_samples": 0}, {"n_jobs": 0},
    ],
)
def test_invalid_configuration_is_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(IsolationForestError):
        make_detector(**kwargs)


def test_sklearn_minus_one_and_one_map_to_binary_anomaly_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    detector = make_detector().fit(normal_training())
    monkeypatch.setattr(detector.model_, "predict", lambda _: np.array([1, -1, 1, -1]))
    output = detector.predict(pd.Series([0.0, 1.0, 2.0, 3.0], name="OT"))
    assert output.tolist() == [0, 1, 0, 1]
