"""Offline tests for dashboard selection and prediction preparation."""

import numpy as np
import pandas as pd
import pytest

from thermoguard.dashboard.service import (
    DashboardDataError,
    build_scenario_predictions,
    describe_sampling_interval,
    format_date_range,
    load_dashboard_sources,
    load_optional_metrics,
    lookup_evaluation_metrics,
    resolve_event_duration,
    validate_selection,
)


@pytest.fixture
def dashboard_case():
    dates = pd.date_range("2022-01-01", periods=100, freq="h")
    raw = pd.DataFrame({
        "date": dates,
        "HUFL": np.sin(np.arange(100) / 4), "HULL": np.cos(np.arange(100) / 5),
        "MUFL": np.sin(np.arange(100) / 6), "MULL": np.cos(np.arange(100) / 7),
        "LUFL": np.sin(np.arange(100) / 8), "LULL": np.cos(np.arange(100) / 9),
        "OT": 20 + np.sin(np.arange(100) / 3),
    })
    selected = raw.iloc[60:80].copy()
    selected["source_row_index"] = selected.index
    selected["OT_injected"] = selected["OT"].to_numpy(copy=True)
    selected.loc[selected.index[4:6], "OT_injected"] += 8.0
    selected["anomaly_label"] = 0
    selected.loc[selected.index[4:6], "anomaly_label"] = 1
    selected["scenario_name"] = "validation_demo"
    selected["segment"] = "validation"
    scenario_config = {
        "name": "validation_demo", "anomaly_type": "sudden_spike", "segment": "validation",
        "start_position": 4, "duration": 2, "magnitude": 8.0,
    }
    split_config = {"train_fraction": 0.6, "validation_fraction": 0.2, "test_fraction": 0.2}
    detector_config = {
        "statistical_control_limits": {"k": 2.0, "fit_feature": "OT", "prediction_feature": "OT_injected"},
        "isolation_forest": {
            "random_state": 3, "contamination": 0.05, "n_estimators": 40,
            "max_samples": "auto", "max_features": 1.0, "bootstrap": False,
            "n_jobs": 1, "fit_feature": "OT", "prediction_feature": "OT_injected",
        },
    }
    return raw, selected, scenario_config, split_config, detector_config


def test_scenario_and_detector_selection_validation(dashboard_case) -> None:
    _, _, scenario, _, _ = dashboard_case
    assert validate_selection("validation_demo", "isolation_forest", [scenario]) == scenario
    with pytest.raises(DashboardDataError, match="not configured"):
        validate_selection("unknown", "isolation_forest", [scenario])
    with pytest.raises(DashboardDataError, match="not supported"):
        validate_selection("validation_demo", "unknown", [scenario])


def test_overview_formatters_show_full_range_and_human_sampling_interval(dashboard_case) -> None:
    raw, _, _, _, _ = dashboard_case
    assert describe_sampling_interval(raw) == "1 hour"
    assert format_date_range(raw["date"]) == "Jan 01 2022 00:00 → Jan 05 2022 03:00"


def test_event_duration_uses_config_or_metadata_and_never_invents_value() -> None:
    assert resolve_event_duration({"duration": 72}) == "72 rows"
    assert resolve_event_duration({}, {"parameters": {"duration_rows": 4}}) == "4 rows"
    assert resolve_event_duration({}, {"start_row_index": 10, "end_row_index": 12}) == "3 rows"
    assert resolve_event_duration({}) == "Not available"


def test_chart_marks_only_detector_predictions_and_preserves_time_order(dashboard_case) -> None:
    from thermoguard.dashboard.app import _make_chart

    raw, scenarios, scenario, split, configs = dashboard_case
    chart_data, _ = build_scenario_predictions(
        raw, scenarios, scenario_name=scenario["name"], detector_name="statistical_control_limits",
        scenario_config=scenario, split_config=split,
        detector_config=configs["statistical_control_limits"],
    )
    chart = _make_chart(chart_data, scenario["name"], "Control limits")
    assert len(chart.data) == 3
    assert list(chart.data[0].x) == chart_data["date"].tolist()
    predicted_trace = chart.data[2]
    assert len(predicted_trace.x) == int(chart_data["prediction"].sum())
    assert "anomaly_label" not in chart_data.columns


@pytest.mark.parametrize("detector_name", ["statistical_control_limits", "isolation_forest"])
def test_predictions_align_and_ground_truth_is_excluded(dashboard_case, detector_name: str) -> None:
    raw, scenarios, scenario, split, configs = dashboard_case
    chart, summary = build_scenario_predictions(
        raw, scenarios, scenario_name=scenario["name"], detector_name=detector_name,
        scenario_config=scenario, split_config=split, detector_config=configs[detector_name],
    )
    assert len(chart) == len(scenarios)
    assert chart.index.equals(scenarios.index)
    assert chart["date"].tolist() == pd.to_datetime(scenarios["date"]).tolist()
    assert "anomaly_label" not in chart.columns
    assert "prediction" in chart.columns and "anomaly_score" in chart.columns
    assert summary["training_rows"] == 60
    assert summary["prediction_rows"] == len(scenarios)


def test_prediction_api_receives_only_ot_injected(dashboard_case, monkeypatch: pytest.MonkeyPatch) -> None:
    from thermoguard.detectors.control_limits import StatisticalControlLimits

    raw, scenarios, scenario, split, configs = dashboard_case
    original = StatisticalControlLimits.predict
    received = []

    def spy(self, values, *, value_column=None):
        assert isinstance(values, pd.Series)
        assert values.name == "OT_injected"
        received.append(values.name)
        return original(self, values, value_column=value_column)

    monkeypatch.setattr(StatisticalControlLimits, "predict", spy)
    build_scenario_predictions(
        raw, scenarios, scenario_name=scenario["name"], detector_name="statistical_control_limits",
        scenario_config=scenario, split_config=split,
        detector_config=configs["statistical_control_limits"],
    )
    assert received == ["OT_injected"]


def test_malformed_scenario_and_split_mismatch_are_rejected(dashboard_case) -> None:
    raw, scenarios, scenario, split, configs = dashboard_case
    with pytest.raises(DashboardDataError, match="missing columns"):
        build_scenario_predictions(
            raw, scenarios.drop(columns="OT_injected"), scenario_name=scenario["name"],
            detector_name="statistical_control_limits", scenario_config=scenario,
            split_config=split, detector_config=configs["statistical_control_limits"],
        )
    malformed = scenarios.copy()
    malformed["date"] = "not-a-date"
    with pytest.raises(DashboardDataError, match="malformed timestamps"):
        build_scenario_predictions(
            raw, malformed, scenario_name=scenario["name"], detector_name="statistical_control_limits",
            scenario_config=scenario, split_config=split,
            detector_config=configs["statistical_control_limits"],
        )
    misaligned = scenarios.copy()
    misaligned.iloc[0, misaligned.columns.get_loc("date")] = pd.Timestamp("2022-01-03 13:00:00")
    with pytest.raises(DashboardDataError, match="timestamps do not align"):
        build_scenario_predictions(
            raw, misaligned, scenario_name=scenario["name"], detector_name="statistical_control_limits",
            scenario_config=scenario, split_config=split,
            detector_config=configs["statistical_control_limits"],
        )
    wrong_segment = scenario | {"segment": "test"}
    with pytest.raises(DashboardDataError, match="split assignment"):
        build_scenario_predictions(
            raw, scenarios, scenario_name=scenario["name"], detector_name="statistical_control_limits",
            scenario_config=wrong_segment, split_config=split,
            detector_config=configs["statistical_control_limits"],
        )


def test_source_loader_reports_missing_scenario_columns(tmp_path) -> None:
    raw_path = tmp_path / "raw.csv"
    scenario_path = tmp_path / "scenarios.csv"
    pd.DataFrame({"date": ["2020-01-01"], "OT": [1.0]}).to_csv(raw_path, index=False)
    pd.DataFrame({"scenario_name": ["s"]}).to_csv(scenario_path, index=False)
    with pytest.raises(DashboardDataError, match="missing required columns"):
        load_dashboard_sources(raw_path, scenario_path)
    with pytest.raises(DashboardDataError, match="does not exist"):
        load_dashboard_sources(tmp_path / "missing.csv", scenario_path)


def test_optional_evaluation_artifact_and_exact_metric_lookup(tmp_path) -> None:
    assert load_optional_metrics(tmp_path / "absent.csv") is None
    assert lookup_evaluation_metrics(None, scenario_name="s", detector_name="d") is None
    table = pd.DataFrame([
        {"scenario_name": "s", "detector": "d", "segment": "test", "f1": 0.75},
        {"scenario_name": "s", "detector": "other", "segment": "test", "f1": 0.1},
    ])
    assert lookup_evaluation_metrics(table, scenario_name="s", detector_name="d")["f1"] == 0.75
    assert lookup_evaluation_metrics(table, scenario_name="missing", detector_name="d") is None
    with pytest.raises(DashboardDataError, match="missing columns"):
        lookup_evaluation_metrics(pd.DataFrame({"f1": [1.0]}), scenario_name="s", detector_name="d")
