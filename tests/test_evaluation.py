"""Offline evaluation metrics and end-to-end comparison tests."""

import json
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest
import yaml

from thermoguard.data.injection import inject_anomaly
from thermoguard.evaluation.comparison import main
from thermoguard.evaluation.metrics import EvaluationError, calculate_metrics


def test_confusion_counts_and_metrics() -> None:
    result = calculate_metrics([1, 1, 0, 0, 1], [1, 0, 1, 0, 0])
    assert result["true_positives"] == 1
    assert result["false_positives"] == 1
    assert result["true_negatives"] == 1
    assert result["false_negatives"] == 2
    assert result["precision"] == 0.5
    assert result["recall"] == pytest.approx(1 / 3)
    assert result["f1"] == pytest.approx(0.4)
    assert result["accuracy"] == pytest.approx(0.4)
    assert result["specificity"] == 0.5
    assert result["false_positive_rate"] == 0.5
    assert result["n_true_anomaly_points"] == 3
    assert result["n_predicted_anomaly_points"] == 2


def test_zero_denominators_are_explicitly_zero() -> None:
    result = calculate_metrics([0, 0], [0, 0])
    assert result["precision"] == result["recall"] == result["f1"] == 0
    assert result["specificity"] == 1
    assert result["false_positive_rate"] == 0
    no_negative_truth = calculate_metrics([1, 1], [1, 0])
    assert no_negative_truth["specificity"] == 0
    assert no_negative_truth["false_positive_rate"] == 0


@pytest.mark.parametrize(
    ("labels", "predictions", "message"),
    [
        ([0, 1], [0], "length mismatch"),
        ([0, 2], [0, 1], "binary"),
        ([0, None], [0, 1], "missing"),
        ([], [], "empty"),
    ],
)
def test_invalid_metric_inputs_are_rejected(labels, predictions, message: str) -> None:
    with pytest.raises(EvaluationError, match=message):
        calculate_metrics(labels, predictions)


def _make_project(tmp_path):
    raw_path = tmp_path / "ETTh1.csv"
    dates = pd.date_range("2020-01-01", periods=100, freq="h")
    frame = pd.DataFrame({
        "date": dates,
        "HUFL": np.sin(np.arange(100) / 4),
        "HULL": np.cos(np.arange(100) / 5),
        "MUFL": np.sin(np.arange(100) / 6),
        "MULL": np.cos(np.arange(100) / 7),
        "LUFL": np.sin(np.arange(100) / 8),
        "LULL": np.cos(np.arange(100) / 9),
        "OT": 20 + np.sin(np.arange(100) / 3),
    })
    frame.to_csv(raw_path, index=False)
    validation = frame.iloc[60:80].copy()
    test = frame.iloc[80:].copy()
    injected = []
    events = []
    scenarios_cfg = [
        ("validation_fixture", "validation", validation, 4, 3.0),
        ("test_fixture", "test", test, 5, -4.0),
    ]
    for name, segment, segment_frame, position, magnitude in scenarios_cfg:
        injection = inject_anomaly(
            segment_frame, anomaly_type="sudden_spike", segment=segment,
            start_position=position, duration=2, magnitude=magnitude, seed=42,
        )
        result = injection.data.copy()
        result["scenario_name"] = name
        result["segment"] = segment
        injected.append(result)
        events.append(asdict(injection.event) | {"scenario_name": name})
    scenarios_path = tmp_path / "scenarios.csv"
    pd.concat(injected).to_csv(scenarios_path, index_label="source_row_index")
    events_path = tmp_path / "events.json"
    events_path.write_text(json.dumps({"events": events}), encoding="utf-8")
    experiments_path = tmp_path / "experiments.yaml"
    experiments_path.write_text(yaml.safe_dump({
        "split": {"train_fraction": 0.6, "validation_fraction": 0.2, "test_fraction": 0.2},
        "scenarios": [
            {"name": name, "segment": segment, "anomaly_type": "sudden_spike"}
            for name, segment, *_ in scenarios_cfg
        ],
    }), encoding="utf-8")
    detectors_path = tmp_path / "detectors.yaml"
    detectors_path.write_text(yaml.safe_dump({"detectors": {
        "statistical_control_limits": {"k": 2.0, "fit_feature": "OT"},
        "isolation_forest": {
            "random_state": 42, "contamination": 0.05, "n_estimators": 50,
            "max_samples": "auto", "max_features": 1.0, "bootstrap": False,
            "n_jobs": 1, "fit_feature": "OT",
        },
    }}), encoding="utf-8")
    return raw_path, scenarios_path, events_path, experiments_path, detectors_path


def test_evaluates_both_detectors_scenarios_writes_artifacts_and_keeps_labels_isolated(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from thermoguard.detectors.control_limits import StatisticalControlLimits
    from thermoguard.detectors.isolation_forest import TemperatureIsolationForest

    paths = _make_project(tmp_path)
    seen_predictions: list[str] = []
    seen_training: list[str] = []
    control_fit = StatisticalControlLimits.fit
    forest_fit = TemperatureIsolationForest.fit
    control_predict = StatisticalControlLimits.predict
    forest_predict = TemperatureIsolationForest.predict

    def control_fit_spy(self, observations):
        assert isinstance(observations, pd.Series)
        assert observations.name == "OT"
        seen_training.append(observations.name)
        return control_fit(self, observations)

    def forest_fit_spy(self, observations):
        assert isinstance(observations, pd.Series)
        assert observations.name == "OT"
        seen_training.append(observations.name)
        return forest_fit(self, observations)

    def control_spy(self, observations, *, value_column=None):
        assert isinstance(observations, pd.Series)
        assert observations.name == "OT_injected"
        seen_predictions.append(observations.name)
        return control_predict(self, observations, value_column=value_column)

    def forest_spy(self, observations, *, value_column=None):
        assert isinstance(observations, pd.Series)
        assert observations.name == "OT_injected"
        seen_predictions.append(observations.name)
        return forest_predict(self, observations, value_column=value_column)

    monkeypatch.setattr(StatisticalControlLimits, "fit", control_fit_spy)
    monkeypatch.setattr(TemperatureIsolationForest, "fit", forest_fit_spy)
    monkeypatch.setattr(StatisticalControlLimits, "predict", control_spy)
    monkeypatch.setattr(TemperatureIsolationForest, "predict", forest_spy)
    output_dir = tmp_path / "outputs"
    sentinel = tmp_path / "control_limits_validation_gradual_drift.csv"
    sentinel.write_text("preserve me", encoding="utf-8")
    raw, scenarios, events, experiments, detectors = paths
    assert main([
        "--raw", str(raw), "--scenarios", str(scenarios), "--events", str(events),
        "--experiments-config", str(experiments), "--detector-config", str(detectors),
        "--output-dir", str(output_dir),
    ]) == 0
    metrics_path = output_dir / "scenario_metrics.csv"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "evaluation_report.md"
    assert metrics_path.is_file() and summary_path.is_file() and report_path.is_file()
    metrics = pd.read_csv(metrics_path)
    assert len(metrics) == 4
    assert set(metrics["detector"]) == {"statistical_control_limits", "isolation_forest"}
    assert set(zip(metrics["scenario_name"], metrics["segment"])) == {
        ("validation_fixture", "validation"), ("test_fixture", "test")
    }
    assert "anomaly_label" not in metrics.columns
    assert {"precision", "recall", "f1", "false_positive_rate"}.issubset(metrics.columns)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["labels_used_for"].startswith("Scoring/reporting only")
    assert len(summary["scenario_identifiers"]) == 2
    assert "Methodology" in report_path.read_text(encoding="utf-8")
    assert sentinel.read_text(encoding="utf-8") == "preserve me"
    assert len(seen_predictions) == 4
    assert seen_training == ["OT"] * 4


def test_deterministic_metrics_excluding_timestamp(tmp_path) -> None:
    from thermoguard.evaluation.comparison import _evaluate

    raw, scenarios, events, experiments, detectors = _make_project(tmp_path)
    kwargs = {
        "raw_path": raw, "scenarios_path": scenarios, "events_path": events,
        "experiments_config_path": experiments, "detector_config_path": detectors,
    }
    first_metrics, first_summary = _evaluate(**kwargs)
    second_metrics, second_summary = _evaluate(**kwargs)
    pd.testing.assert_frame_equal(first_metrics, second_metrics)
    assert first_summary["dataset"] == second_summary["dataset"]
    assert first_summary["scenario_identifiers"] == second_summary["scenario_identifiers"]
    assert first_summary["evaluation_timestamp_utc"]
