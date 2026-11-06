"""Two-sided statistical control limits for a single temperature series."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from thermoguard.data.loader import load_dataset
from thermoguard.data.splitting import chronological_split
from thermoguard.data.validation import validate_dataset


class ControlLimitError(ValueError):
    """Raised when data or detector state cannot support a valid prediction."""


class StatisticalControlLimits:
    """Mean ± k sample-standard-deviation detector for one numeric feature.

    Fit with clean training data only. Prediction flags values strictly below
    or above the fitted limits; observations exactly on a limit are normal.
    """

    def __init__(self, k: float = 3.0, feature_column: str = "OT") -> None:
        if isinstance(k, bool) or not isinstance(k, (int, float)) or not math.isfinite(float(k)):
            raise ControlLimitError("k must be a finite positive number.")
        if float(k) <= 0:
            raise ControlLimitError("k must be greater than zero.")
        if not isinstance(feature_column, str) or not feature_column:
            raise ControlLimitError("feature_column must be a non-empty string.")
        if feature_column == "anomaly_label" or feature_column.endswith("_injected"):
            raise ControlLimitError("Fit feature must be a clean source measurement, not a label or injected value.")
        self.k = float(k)
        self.feature_column = feature_column
        self.mean_: float | None = None
        self.std_: float | None = None
        self.lower_limit_: float | None = None
        self.upper_limit_: float | None = None
        self.n_samples_: int | None = None

    def fit(self, training_data: pd.DataFrame | pd.Series) -> StatisticalControlLimits:
        """Estimate limits from only ``feature_column`` in clean training data."""
        if isinstance(training_data, pd.DataFrame):
            reserved = {"anomaly_label", f"{self.feature_column}_injected"}
            present = reserved.intersection(training_data.columns)
            if present:
                raise ControlLimitError(
                    "Training data must be clean and contain no ground-truth label or injected feature; "
                    f"found: {', '.join(sorted(present))}."
                )
            if self.feature_column not in training_data.columns:
                raise ControlLimitError(f"Training data is missing feature '{self.feature_column}'.")
            values = training_data[self.feature_column]
        elif isinstance(training_data, pd.Series):
            if training_data.name == "anomaly_label" or (
                training_data.name is not None and training_data.name.endswith("_injected")
            ):
                raise ControlLimitError("Training data must not be an anomaly label or injected feature.")
            if training_data.name not in (None, self.feature_column):
                raise ControlLimitError(
                    f"Training Series must represent '{self.feature_column}', got '{training_data.name}'."
                )
            values = training_data
        else:
            raise ControlLimitError("training_data must be a pandas DataFrame or Series.")

        numeric = self._validated_values(values, context="training")
        if numeric.size < 2:
            raise ControlLimitError("At least two clean training observations are required.")
        mean = float(np.mean(numeric))
        standard_deviation = float(np.std(numeric, ddof=1))
        if not math.isfinite(mean) or not math.isfinite(standard_deviation):
            raise ControlLimitError("Training mean and standard deviation must be finite.")
        if standard_deviation == 0:
            raise ControlLimitError(
                "Clean training feature has zero variance; control limits would collapse to one value."
            )

        self.mean_ = mean
        self.std_ = standard_deviation
        self.lower_limit_ = mean - self.k * standard_deviation
        self.upper_limit_ = mean + self.k * standard_deviation
        self.n_samples_ = int(numeric.size)
        return self

    @staticmethod
    def _validated_values(values: pd.Series, *, context: str) -> np.ndarray:
        if not pd.api.types.is_numeric_dtype(values.dtype):
            raise ControlLimitError(f"{context.capitalize()} values must be numeric; got dtype '{values.dtype}'.")
        if values.empty:
            raise ControlLimitError(f"{context.capitalize()} data is empty.")
        if values.isna().any():
            raise ControlLimitError(f"{context.capitalize()} data contains missing values.")
        numeric = values.to_numpy(dtype=float, copy=True)
        if not np.isfinite(numeric).all():
            raise ControlLimitError(f"{context.capitalize()} data contains non-finite values.")
        return numeric

    def _require_fitted(self) -> tuple[float, float, float, float]:
        if any(
            value is None
            for value in (self.mean_, self.std_, self.lower_limit_, self.upper_limit_)
        ):
            raise ControlLimitError("Detector is not fitted; call fit() with clean training data first.")
        return self.mean_, self.std_, self.lower_limit_, self.upper_limit_

    def _select_prediction_values(
        self,
        observations: pd.DataFrame | pd.Series,
        value_column: str | None,
    ) -> pd.Series:
        if isinstance(observations, pd.DataFrame):
            column = value_column or self.feature_column
            if column == "anomaly_label":
                raise ControlLimitError("Ground-truth labels cannot be used as detector inputs.")
            if column not in observations.columns:
                raise ControlLimitError(f"Prediction data is missing value column '{column}'.")
            values = observations[column]
        elif isinstance(observations, pd.Series):
            if observations.name == "anomaly_label":
                raise ControlLimitError("Ground-truth labels cannot be used as detector inputs.")
            values = observations
        else:
            raise ControlLimitError("observations must be a pandas DataFrame or Series.")
        self._validated_values(values, context="prediction")
        return values

    def score(
        self,
        observations: pd.DataFrame | pd.Series,
        *,
        value_column: str | None = None,
    ) -> pd.Series:
        """Return signed standardized deviations, aligned to input row indices."""
        mean, std, _, _ = self._require_fitted()
        values = self._select_prediction_values(observations, value_column)
        scores = (values.to_numpy(dtype=float, copy=True) - mean) / std
        return pd.Series(scores, index=values.index, name="control_limit_score_z")

    def predict(
        self,
        observations: pd.DataFrame | pd.Series,
        *,
        value_column: str | None = None,
    ) -> pd.Series:
        """Return aligned uint8 flags: 1 outside fitted limits, else 0."""
        _, _, lower, upper = self._require_fitted()
        values = self._select_prediction_values(observations, value_column)
        array = values.to_numpy(dtype=float, copy=True)
        predictions = ((array < lower) | (array > upper)).astype(np.uint8)
        return pd.Series(predictions, index=values.index, name="control_limit_prediction")


def _run_prediction_command(
    *,
    raw_path: Path,
    scenario_path: Path,
    experiment_config_path: Path,
    detector_config_path: Path,
    scenario_name: str,
    output_path: Path,
) -> dict[str, Any]:
    experiments = yaml.safe_load(experiment_config_path.read_text(encoding="utf-8"))
    detector_root = yaml.safe_load(detector_config_path.read_text(encoding="utf-8"))
    split_config = experiments["split"]
    detector_config = detector_root["detectors"]["statistical_control_limits"]
    fit_feature = detector_config.get("fit_feature", "OT")
    prediction_feature = detector_config.get("prediction_feature", "OT_injected")
    k = detector_config["k"]

    raw_frame = load_dataset(raw_path)
    source_issues = validate_dataset(raw_frame)
    if source_issues:
        raise ControlLimitError("Raw source validation failed: " + " | ".join(source_issues))
    chronological = chronological_split(
        raw_frame,
        train_fraction=split_config["train_fraction"],
        validation_fraction=split_config["validation_fraction"],
        test_fraction=split_config["test_fraction"],
    )
    if fit_feature != "OT":
        raise ControlLimitError("This baseline is configured to fit only the clean OT feature.")
    detector = StatisticalControlLimits(k=k, feature_column=fit_feature)
    detector.fit(chronological.train[fit_feature])

    required_columns = ["source_row_index", "date", "OT", prediction_feature, "scenario_name", "segment"]
    scenario_data = pd.read_csv(scenario_path, usecols=required_columns)
    selected = scenario_data.loc[scenario_data["scenario_name"] == scenario_name].copy()
    if selected.empty:
        raise ControlLimitError(f"Scenario '{scenario_name}' was not found in {scenario_path}.")
    segments = selected["segment"].drop_duplicates().tolist()
    if len(segments) != 1 or segments[0] not in ("validation", "test"):
        raise ControlLimitError("Selected scenario must belong to one held-out validation or test segment.")
    if selected["source_row_index"].duplicated().any():
        raise ControlLimitError("Selected scenario has duplicate source_row_index values.")
    permitted_indices = set(getattr(chronological, segments[0]).index.tolist())
    scenario_indices = set(selected["source_row_index"].astype(int).tolist())
    if not scenario_indices.issubset(permitted_indices):
        raise ControlLimitError("Selected scenario contains source rows outside its assigned held-out segment.")

    selected = selected.set_index("source_row_index", drop=False, verify_integrity=True)
    # Labels are intentionally not loaded by `usecols`; only the sensor value
    # and its timestamp/source index are passed through the prediction API.
    scores = detector.score(selected, value_column=prediction_feature)
    predictions = detector.predict(selected, value_column=prediction_feature)
    output = selected[["source_row_index", "date", "OT", prediction_feature, "scenario_name", "segment"]].copy()
    output["control_limit_score_z"] = scores
    output["control_limit_prediction"] = predictions

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    summary = {
        "scenario_name": scenario_name,
        "segment": segments[0],
        "prediction_rows": len(output),
        "predicted_anomalies": int(predictions.sum()),
        "prediction_rate": float(predictions.mean()),
        "fit_feature": fit_feature,
        "fit_rows": detector.n_samples_,
        "fit_mean": detector.mean_,
        "fit_sample_std": detector.std_,
        "k": detector.k,
        "lower_limit": detector.lower_limit_,
        "upper_limit": detector.upper_limit_,
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "output_file": str(output_path),
        "labels_used": False,
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fit mean ± k standard-deviation limits on clean training OT and predict a held-out scenario."
    )
    parser.add_argument("--raw", type=Path, default=Path("data/raw/ETTh1.csv"))
    parser.add_argument(
        "--scenarios", type=Path, default=Path("data/processed/ETTh1_synthetic_scenarios.csv")
    )
    parser.add_argument("--experiments-config", type=Path, default=Path("configs/experiments.yaml"))
    parser.add_argument("--detector-config", type=Path, default=Path("configs/detectors.yaml"))
    parser.add_argument("--scenario", type=str, default=None)
    parser.add_argument("--output", type=Path, default=Path("reports/results/control_limits_predictions.csv"))
    args = parser.parse_args(argv)

    try:
        detector_root = yaml.safe_load(args.detector_config.read_text(encoding="utf-8"))
        default_scenario = detector_root["detectors"]["statistical_control_limits"].get(
            "default_scenario", "validation_gradual_drift"
        )
        scenario_name = args.scenario or default_scenario
        summary = _run_prediction_command(
            raw_path=args.raw,
            scenario_path=args.scenarios,
            experiment_config_path=args.experiments_config,
            detector_config_path=args.detector_config,
            scenario_name=scenario_name,
            output_path=args.output,
        )
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2))
    print(f"Prediction summary: {args.output.with_suffix('.summary.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
