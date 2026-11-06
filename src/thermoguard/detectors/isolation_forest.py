"""Univariate Isolation Forest detector and held-out scenario command."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import IsolationForest

from thermoguard.data.loader import load_dataset
from thermoguard.data.splitting import chronological_split
from thermoguard.data.validation import validate_dataset


class IsolationForestError(ValueError):
    """Raised when detector inputs or configuration are invalid."""


class TemperatureIsolationForest:
    """Univariate Isolation Forest fit on clean OT and applied to scenario values.

    ``anomaly_score`` is the negated sklearn ``score_samples`` value, so larger
    values indicate observations that are more anomalous.
    """

    def __init__(
        self,
        *,
        random_state: int = 20261023,
        contamination: float | str = 0.01,
        n_estimators: int = 200,
        max_samples: float | str = "auto",
        max_features: float = 1.0,
        bootstrap: bool = False,
        n_jobs: int | None = 1,
        fit_feature: str = "OT",
    ) -> None:
        if isinstance(random_state, bool) or not isinstance(random_state, int):
            raise IsolationForestError("random_state must be an integer.")
        if contamination != "auto":
            if isinstance(contamination, bool) or not isinstance(contamination, (int, float)):
                raise IsolationForestError("contamination must be 'auto' or a number in (0, 0.5].")
            if not math.isfinite(float(contamination)) or not 0 < float(contamination) <= 0.5:
                raise IsolationForestError("contamination must be 'auto' or a number in (0, 0.5].")
        if isinstance(n_estimators, bool) or not isinstance(n_estimators, int) or n_estimators < 1:
            raise IsolationForestError("n_estimators must be a positive integer.")
        if isinstance(max_samples, str):
            if max_samples != "auto":
                raise IsolationForestError("max_samples must be 'auto', an integer, or a fraction in (0, 1].")
        elif isinstance(max_samples, bool) or not isinstance(max_samples, (int, float)):
            raise IsolationForestError("max_samples must be 'auto', an integer, or a fraction in (0, 1].")
        elif isinstance(max_samples, int):
            if max_samples < 1:
                raise IsolationForestError("Integer max_samples must be positive.")
        elif not math.isfinite(float(max_samples)) or not 0 < float(max_samples) <= 1:
            raise IsolationForestError("Fractional max_samples must be in (0, 1].")
        if isinstance(max_features, bool) or not isinstance(max_features, (int, float)):
            raise IsolationForestError("max_features must be a number in (0, 1].")
        if not math.isfinite(float(max_features)) or not 0 < float(max_features) <= 1:
            raise IsolationForestError("max_features must be in (0, 1].")
        if not isinstance(bootstrap, bool):
            raise IsolationForestError("bootstrap must be a boolean.")
        if n_jobs is not None and (isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or n_jobs == 0):
            raise IsolationForestError("n_jobs must be a non-zero integer or null.")
        if not isinstance(fit_feature, str) or not fit_feature:
            raise IsolationForestError("fit_feature must be a non-empty string.")
        if fit_feature == "anomaly_label" or fit_feature.endswith("_injected"):
            raise IsolationForestError("fit_feature must be a clean source measurement, not a label or injected value.")

        self.random_state = random_state
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.fit_feature = fit_feature
        self.model_: IsolationForest | None = None
        self.n_samples_: int | None = None

    def fit(self, training_data: pd.DataFrame | pd.Series) -> TemperatureIsolationForest:
        """Fit on clean training observations only; labels/injected fields are rejected."""
        if isinstance(training_data, pd.DataFrame):
            if "anomaly_label" in training_data.columns or f"{self.fit_feature}_injected" in training_data.columns:
                raise IsolationForestError("Training data must be clean and contain no label or injected feature.")
            if self.fit_feature not in training_data.columns:
                raise IsolationForestError(f"Training data is missing feature '{self.fit_feature}'.")
            values = training_data[self.fit_feature]
        elif isinstance(training_data, pd.Series):
            if training_data.name == "anomaly_label" or (
                training_data.name is not None and training_data.name.endswith("_injected")
            ):
                raise IsolationForestError("Training data must be a clean source measurement, not a label or injected value.")
            if training_data.name not in (None, self.fit_feature):
                raise IsolationForestError(f"Training Series must represent '{self.fit_feature}'.")
            values = training_data
        else:
            raise IsolationForestError("training_data must be a pandas DataFrame or Series.")
        array = self._values(values, "training")
        if array.size < 2:
            raise IsolationForestError("At least two clean training observations are required.")
        if isinstance(self.max_samples, int) and self.max_samples > array.size:
            raise IsolationForestError("max_samples cannot exceed the number of training observations.")
        try:
            model = IsolationForest(
                n_estimators=self.n_estimators,
                max_samples=self.max_samples,
                contamination=self.contamination,
                max_features=self.max_features,
                bootstrap=self.bootstrap,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            )
            model.fit(array.reshape(-1, 1))
        except (TypeError, ValueError) as exc:
            raise IsolationForestError(f"Could not fit Isolation Forest: {exc}") from exc
        self.model_ = model
        self.n_samples_ = int(array.size)
        return self

    @staticmethod
    def _values(values: pd.Series, context: str) -> np.ndarray:
        if not pd.api.types.is_numeric_dtype(values.dtype):
            raise IsolationForestError(f"{context.capitalize()} values must be numeric; got dtype '{values.dtype}'.")
        if values.empty:
            raise IsolationForestError(f"{context.capitalize()} data is empty.")
        if values.isna().any():
            raise IsolationForestError(f"{context.capitalize()} data contains missing values.")
        array = values.to_numpy(dtype=float, copy=True)
        if not np.isfinite(array).all():
            raise IsolationForestError(f"{context.capitalize()} data contains non-finite values.")
        return array

    def _prediction_values(
        self, observations: pd.DataFrame | pd.Series, value_column: str | None
    ) -> tuple[pd.Series, np.ndarray]:
        if self.model_ is None:
            raise IsolationForestError("Detector is not fitted; call fit() with clean training data first.")
        if isinstance(observations, pd.DataFrame):
            column = value_column or self.fit_feature
            if column == "anomaly_label":
                raise IsolationForestError("Ground-truth labels cannot be used as detector inputs.")
            if column not in observations.columns:
                raise IsolationForestError(f"Prediction data is missing value column '{column}'.")
            values = observations[column]
        elif isinstance(observations, pd.Series):
            if observations.name == "anomaly_label":
                raise IsolationForestError("Ground-truth labels cannot be used as detector inputs.")
            values = observations
        else:
            raise IsolationForestError("observations must be a pandas DataFrame or Series.")
        return values, self._values(values, "prediction").reshape(-1, 1)

    def score(self, observations: pd.DataFrame | pd.Series, *, value_column: str | None = None) -> pd.Series:
        """Return aligned anomaly scores; larger means more anomalous."""
        values, matrix = self._prediction_values(observations, value_column)
        scores = -self.model_.score_samples(matrix)
        if not np.isfinite(scores).all():
            raise IsolationForestError("Model produced non-finite anomaly scores.")
        return pd.Series(scores, index=values.index, name="isolation_forest_anomaly_score")

    def predict(self, observations: pd.DataFrame | pd.Series, *, value_column: str | None = None) -> pd.Series:
        """Return aligned uint8 anomaly flags (1=outlier, 0=inlier)."""
        values, matrix = self._prediction_values(observations, value_column)
        sklearn_labels = self.model_.predict(matrix)
        if not np.isin(sklearn_labels, (-1, 1)).all():
            raise IsolationForestError("Isolation Forest returned unexpected prediction labels.")
        flags = (sklearn_labels == -1).astype(np.uint8)
        return pd.Series(flags, index=values.index, name="isolation_forest_prediction")


def _run_prediction_command(
    *, raw_path: Path, scenario_path: Path, experiment_config_path: Path,
    detector_config_path: Path, scenario_name: str, output_path: Path,
) -> dict[str, Any]:
    experiments = yaml.safe_load(experiment_config_path.read_text(encoding="utf-8"))
    detector_root = yaml.safe_load(detector_config_path.read_text(encoding="utf-8"))
    split_config = experiments["split"]
    config = detector_root["detectors"]["isolation_forest"]
    fit_feature = config.get("fit_feature", "OT")
    prediction_feature = config.get("prediction_feature", "OT_injected")
    if fit_feature != "OT" or prediction_feature != "OT_injected":
        raise IsolationForestError("This command requires clean OT fitting and OT_injected held-out prediction.")

    raw_frame = load_dataset(raw_path)
    source_issues = validate_dataset(raw_frame)
    if source_issues:
        raise IsolationForestError("Raw source validation failed: " + " | ".join(source_issues))
    chronological = chronological_split(
        raw_frame,
        train_fraction=split_config["train_fraction"],
        validation_fraction=split_config["validation_fraction"],
        test_fraction=split_config["test_fraction"],
    )
    detector = TemperatureIsolationForest(
        random_state=config.get("random_state", 20261023),
        contamination=config.get("contamination", 0.01),
        n_estimators=config.get("n_estimators", 200),
        max_samples=config.get("max_samples", "auto"),
        max_features=config.get("max_features", 1.0),
        bootstrap=config.get("bootstrap", False),
        n_jobs=config.get("n_jobs", 1),
        fit_feature=fit_feature,
    ).fit(chronological.train[fit_feature])

    required = ["source_row_index", "date", "OT", prediction_feature, "scenario_name", "segment"]
    scenarios = pd.read_csv(scenario_path, usecols=required)
    selected = scenarios.loc[scenarios["scenario_name"] == scenario_name].copy()
    if selected.empty:
        raise IsolationForestError(f"Scenario '{scenario_name}' was not found in {scenario_path}.")
    segments = selected["segment"].drop_duplicates().tolist()
    if len(segments) != 1 or segments[0] not in ("validation", "test"):
        raise IsolationForestError("Selected scenario must belong to one held-out validation or test segment.")
    if selected["source_row_index"].duplicated().any():
        raise IsolationForestError("Selected scenario has duplicate source_row_index values.")
    allowed = set(getattr(chronological, segments[0]).index.tolist())
    if not set(selected["source_row_index"].astype(int)).issubset(allowed):
        raise IsolationForestError("Scenario contains source rows outside its assigned held-out segment.")
    selected = selected.set_index("source_row_index", drop=False, verify_integrity=True)
    predictions = detector.predict(selected, value_column=prediction_feature)
    scores = detector.score(selected, value_column=prediction_feature)
    output = selected[required].copy()
    output["isolation_forest_anomaly_score"] = scores
    output["isolation_forest_prediction"] = predictions
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    summary = {
        "scenario_name": scenario_name,
        "segment": segments[0],
        "training_rows": detector.n_samples_,
        "prediction_rows": len(output),
        "predicted_anomalies": int(predictions.sum()),
        "prediction_rate": float(predictions.mean()),
        "fit_feature": fit_feature,
        "prediction_feature": prediction_feature,
        "random_state": detector.random_state,
        "contamination": detector.contamination,
        "n_estimators": detector.n_estimators,
        "max_samples": detector.max_samples,
        "max_features": detector.max_features,
        "bootstrap": detector.bootstrap,
        "n_jobs": detector.n_jobs,
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "output_file": str(output_path),
        "labels_used": False,
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fit Isolation Forest on clean training OT and predict a held-out synthetic scenario."
    )
    parser.add_argument("--raw", type=Path, default=Path("data/raw/ETTh1.csv"))
    parser.add_argument("--scenarios", type=Path, default=Path("data/processed/ETTh1_synthetic_scenarios.csv"))
    parser.add_argument("--experiments-config", type=Path, default=Path("configs/experiments.yaml"))
    parser.add_argument("--detector-config", type=Path, default=Path("configs/detectors.yaml"))
    parser.add_argument("--scenario", type=str, default=None)
    parser.add_argument("--output", type=Path, default=Path("reports/results/isolation_forest_predictions.csv"))
    args = parser.parse_args(argv)
    try:
        root = yaml.safe_load(args.detector_config.read_text(encoding="utf-8"))
        default_name = root["detectors"]["isolation_forest"].get("default_scenario", "validation_gradual_drift")
        summary = _run_prediction_command(
            raw_path=args.raw, scenario_path=args.scenarios,
            experiment_config_path=args.experiments_config,
            detector_config_path=args.detector_config,
            scenario_name=args.scenario or default_name, output_path=args.output,
        )
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2))
    print(f"Prediction summary: {args.output.with_suffix('.summary.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
