"""Data preparation helpers shared by the Streamlit dashboard and offline tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from thermoguard.data.loader import DatasetLoadError, load_dataset
from thermoguard.data.splitting import chronological_split
from thermoguard.data.validation import validate_dataset
from thermoguard.detectors.control_limits import StatisticalControlLimits
from thermoguard.detectors.isolation_forest import TemperatureIsolationForest

DETECTOR_LABELS = {
    "statistical_control_limits": "Statistical control limits",
    "isolation_forest": "Isolation Forest",
}
SCENARIO_REQUIRED_COLUMNS = {
    "source_row_index", "date", "OT", "OT_injected", "anomaly_label", "scenario_name", "segment"
}
EVALUATION_REQUIRED_COLUMNS = {"scenario_name", "detector", "segment"}


class DashboardDataError(ValueError):
    """Raised when dashboard sources or selection state cannot be used safely."""


def format_date_range(values: pd.Series) -> str:
    """Format a complete date-time range compactly for the overview section."""
    parsed = pd.to_datetime(values, errors="coerce", format="mixed").dropna()
    if parsed.empty:
        return "Unavailable"
    return f"{parsed.min():%b %d %Y %H:%M} → {parsed.max():%b %d %Y %H:%M}"


def describe_sampling_interval(raw: pd.DataFrame) -> str:
    """Report the modal timestamp spacing in a short human-readable form."""
    if "date" not in raw.columns:
        return "Unavailable"
    timestamps = pd.to_datetime(raw["date"], errors="coerce", format="mixed")
    deltas = timestamps.diff().dropna()
    if deltas.empty:
        return "Unavailable"
    modes = deltas.mode()
    if modes.empty:
        return "Variable"
    interval = pd.Timedelta(modes.iloc[0])
    interval_seconds = interval.total_seconds()
    if interval_seconds == 3600:
        return "1 hour"
    if interval_seconds == 60:
        return "1 minute"
    if interval_seconds == 86400:
        return "1 day"
    return str(interval)


def resolve_event_duration(
    scenario_config: dict[str, Any], event_metadata: dict[str, Any] | None = None
) -> str:
    """Resolve duration from config, then event metadata, or show unavailable."""
    configured = scenario_config.get("duration")
    if isinstance(configured, int) and not isinstance(configured, bool) and configured > 0:
        return f"{configured} rows"
    event = event_metadata or {}
    parameters = event.get("parameters", {})
    recorded = parameters.get("duration_rows")
    if isinstance(recorded, int) and not isinstance(recorded, bool) and recorded > 0:
        return f"{recorded} rows"
    start = event.get("start_row_index")
    end = event.get("end_row_index")
    if isinstance(start, int) and isinstance(end, int) and end >= start:
        return f"{end - start + 1} rows"
    return "Not available"


def validate_selection(
    scenario_name: str,
    detector_name: str,
    configured_scenarios: list[dict[str, Any]],
) -> dict[str, Any]:
    """Resolve an exact configured scenario and supported detector selection."""
    by_name = {item.get("name"): item for item in configured_scenarios}
    if scenario_name not in by_name:
        raise DashboardDataError(f"Scenario '{scenario_name}' is not configured.")
    if detector_name not in DETECTOR_LABELS:
        raise DashboardDataError(f"Detector '{detector_name}' is not supported.")
    return by_name[scenario_name]


def load_dashboard_sources(
    raw_path: str | Path,
    scenarios_path: str | Path,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Read source and scenario artifacts without modifying them or repairing data."""
    try:
        raw = load_dataset(raw_path)
    except DatasetLoadError as exc:
        raise DashboardDataError(str(exc)) from exc
    quality_issues = validate_dataset(raw)
    try:
        scenarios = pd.read_csv(scenarios_path)
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise DashboardDataError(f"Could not read synthetic scenario artifact '{scenarios_path}': {exc}") from exc
    missing = SCENARIO_REQUIRED_COLUMNS.difference(scenarios.columns)
    if missing:
        raise DashboardDataError(
            "Scenario artifact is missing required columns: " + ", ".join(sorted(missing))
        )
    if scenarios.empty:
        raise DashboardDataError("Scenario artifact contains no rows.")
    return raw, quality_issues, scenarios


def build_scenario_predictions(
    raw: pd.DataFrame,
    scenario_rows: pd.DataFrame,
    *,
    scenario_name: str,
    detector_name: str,
    scenario_config: dict[str, Any],
    split_config: dict[str, Any],
    detector_config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit on clean train OT and predict held-out injected OT; labels never enter APIs.

    The returned chart frame deliberately omits ``anomaly_label`` and contains
    timestamp-aligned original/injected readings and model outputs only.
    """
    selected_config = validate_selection(scenario_name, detector_name, [scenario_config])
    if scenario_rows.empty:
        raise DashboardDataError("Scenario artifact contains no rows.")
    missing = SCENARIO_REQUIRED_COLUMNS.difference(scenario_rows.columns)
    if missing:
        raise DashboardDataError("Scenario rows are missing columns: " + ", ".join(sorted(missing)))
    selected = scenario_rows.loc[scenario_rows["scenario_name"] == scenario_name].copy()
    if selected.empty:
        raise DashboardDataError(f"Scenario '{scenario_name}' is missing from the scenario artifact.")
    segments = selected["segment"].drop_duplicates().tolist()
    if len(segments) != 1 or segments[0] != selected_config.get("segment"):
        raise DashboardDataError(f"Scenario '{scenario_name}' has an invalid or inconsistent split assignment.")
    segment = segments[0]
    if segment not in ("validation", "test"):
        raise DashboardDataError("Scenario predictions must use a held-out validation or test segment.")
    if selected["source_row_index"].isna().any() or selected["source_row_index"].duplicated().any():
        raise DashboardDataError("Scenario source_row_index values must be present and unique.")
    timestamps = pd.to_datetime(selected["date"], errors="coerce", format="mixed")
    if timestamps.isna().any():
        raise DashboardDataError("Scenario artifact contains malformed timestamps.")
    if not timestamps.is_monotonic_increasing:
        raise DashboardDataError("Scenario rows are not chronological; the dashboard will not reorder them.")

    split = chronological_split(
        raw,
        train_fraction=split_config["train_fraction"],
        validation_fraction=split_config["validation_fraction"],
        test_fraction=split_config["test_fraction"],
    )
    legal_indices = set(getattr(split, segment).index.tolist())
    source_indices = selected["source_row_index"].astype(int)
    if not set(source_indices.tolist()).issubset(legal_indices):
        raise DashboardDataError(f"Scenario '{scenario_name}' includes rows outside its {segment} split.")
    expected_rows = raw.iloc[source_indices.to_numpy()]
    expected_timestamps = pd.to_datetime(expected_rows["date"], errors="coerce", format="mixed")
    if not timestamps.reset_index(drop=True).equals(expected_timestamps.reset_index(drop=True)):
        raise DashboardDataError("Scenario timestamps do not align with their source row indices.")
    if not np.array_equal(
        selected["OT"].to_numpy(dtype=float), expected_rows["OT"].to_numpy(dtype=float)
    ):
        raise DashboardDataError("Scenario original OT values do not align with their source row indices.")

    train = split.train["OT"]
    # Do not select, transform, or pass anomaly_label to either detector.
    prediction_values = selected["OT_injected"].copy()
    if detector_name == "statistical_control_limits":
        if detector_config.get("fit_feature", "OT") != "OT":
            raise DashboardDataError("Control-limit dashboard fitting is restricted to clean OT.")
        detector = StatisticalControlLimits(
            k=detector_config["k"], feature_column="OT"
        ).fit(train)
        predictions = detector.predict(prediction_values)
        scores = detector.score(prediction_values)
        settings: dict[str, Any] = {
            "fit_feature": "OT",
            "prediction_feature": "OT_injected",
            "k": detector.k,
            "training_mean": detector.mean_,
            "training_sample_std": detector.std_,
            "lower_limit": detector.lower_limit_,
            "upper_limit": detector.upper_limit_,
        }
        score_name = "control_limit_score_z"
    elif detector_name == "isolation_forest":
        if detector_config.get("fit_feature", "OT") != "OT":
            raise DashboardDataError("Isolation Forest dashboard fitting is restricted to clean OT.")
        if detector_config.get("prediction_feature", "OT_injected") != "OT_injected":
            raise DashboardDataError("Isolation Forest dashboard prediction is restricted to OT_injected.")
        detector = TemperatureIsolationForest(
            random_state=detector_config.get("random_state", 20261023),
            contamination=detector_config.get("contamination", 0.01),
            n_estimators=detector_config.get("n_estimators", 200),
            max_samples=detector_config.get("max_samples", "auto"),
            max_features=detector_config.get("max_features", 1.0),
            bootstrap=detector_config.get("bootstrap", False),
            n_jobs=detector_config.get("n_jobs", 1),
            fit_feature="OT",
        ).fit(train)
        predictions = detector.predict(prediction_values)
        scores = detector.score(prediction_values)
        settings = {
            "fit_feature": "OT",
            "prediction_feature": "OT_injected",
            "random_state": detector.random_state,
            "contamination": detector.contamination,
            "n_estimators": detector.n_estimators,
            "max_samples": detector.max_samples,
            "max_features": detector.max_features,
            "bootstrap": detector.bootstrap,
            "n_jobs": detector.n_jobs,
        }
        score_name = "isolation_forest_anomaly_score"
    else:
        raise DashboardDataError(f"Detector '{detector_name}' is not supported.")

    if not predictions.index.equals(selected.index) or not scores.index.equals(selected.index):
        raise DashboardDataError("Detector output does not align with scenario rows.")
    chart_frame = pd.DataFrame(
        {
            "date": timestamps.to_numpy(),
            "OT": selected["OT"].to_numpy(),
            "OT_injected": selected["OT_injected"].to_numpy(),
            "prediction": predictions.to_numpy(),
            "anomaly_score": scores.to_numpy(),
        },
        index=selected.index,
    )
    if "anomaly_label" in chart_frame.columns:
        raise DashboardDataError("Ground-truth labels must not enter prediction chart data.")
    return chart_frame, {
        "scenario_name": scenario_name,
        "segment": segment,
        "training_rows": len(train),
        "prediction_rows": len(chart_frame),
        "predicted_anomalies": int(predictions.sum()),
        "anomaly_percentage": float(predictions.mean() * 100),
        "settings": settings,
        "score_name": score_name,
    }


def lookup_evaluation_metrics(
    metrics: pd.DataFrame | None,
    *,
    scenario_name: str,
    detector_name: str,
) -> pd.Series | None:
    """Return exactly one scenario/detector metric row, or None if artifact absent."""
    if metrics is None:
        return None
    missing = EVALUATION_REQUIRED_COLUMNS.difference(metrics.columns)
    if missing:
        raise DashboardDataError("Evaluation metrics artifact is missing columns: " + ", ".join(sorted(missing)))
    selected = metrics.loc[
        (metrics["scenario_name"] == scenario_name) & (metrics["detector"] == detector_name)
    ]
    if selected.empty:
        return None
    if len(selected) != 1:
        raise DashboardDataError(
            f"Evaluation metrics contain {len(selected)} rows for '{scenario_name}' / '{detector_name}'."
        )
    return selected.iloc[0]


def load_optional_metrics(path: str | Path) -> pd.DataFrame | None:
    """Load existing evaluation output; a missing optional artifact is not fatal."""
    source = Path(path)
    if not source.is_file():
        return None
    try:
        return pd.read_csv(source)
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise DashboardDataError(f"Could not read evaluation metrics artifact '{source}': {exc}") from exc
