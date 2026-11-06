"""Reproducible pointwise comparison of the project's two baseline detectors."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from thermoguard.data.loader import load_dataset
from thermoguard.data.splitting import chronological_split
from thermoguard.data.validation import validate_dataset
from thermoguard.detectors.control_limits import StatisticalControlLimits
from thermoguard.detectors.isolation_forest import TemperatureIsolationForest
from thermoguard.evaluation.metrics import METRIC_DEFINITIONS, EvaluationError, calculate_metrics


class ComparisonError(ValueError):
    """Raised when scenario artifacts do not match the configured held-out splits."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evaluate(
    *, raw_path: Path, scenarios_path: Path, events_path: Path,
    experiments_config_path: Path, detector_config_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    experiments = yaml.safe_load(experiments_config_path.read_text(encoding="utf-8"))
    detector_root = yaml.safe_load(detector_config_path.read_text(encoding="utf-8"))
    split_config = experiments["split"]
    detector_configs = detector_root["detectors"]

    raw = load_dataset(raw_path)
    issues = validate_dataset(raw)
    if issues:
        raise ComparisonError("Raw dataset validation failed: " + " | ".join(issues))
    split = chronological_split(
        raw,
        train_fraction=split_config["train_fraction"],
        validation_fraction=split_config["validation_fraction"],
        test_fraction=split_config["test_fraction"],
    )

    # Labels are read for scoring but remain separate from the detector inputs below.
    usecols = ["source_row_index", "date", "OT", "OT_injected", "anomaly_label", "scenario_name", "segment"]
    scenarios = pd.read_csv(scenarios_path, usecols=usecols)
    events_document = json.loads(events_path.read_text(encoding="utf-8"))
    event_by_name = {event["scenario_name"]: event for event in events_document["events"]}
    if not event_by_name:
        raise ComparisonError("Scenario event metadata contains no scenarios.")
    configured_names = [scenario["name"] for scenario in experiments["scenarios"]]
    if set(event_by_name) != set(configured_names):
        raise ComparisonError("Event metadata scenario names do not match experiments.yaml.")

    records: list[dict[str, Any]] = []
    for scenario_name in configured_names:
        selected = scenarios.loc[scenarios["scenario_name"] == scenario_name].copy()
        if selected.empty:
            raise ComparisonError(f"Scenario '{scenario_name}' is missing from {scenarios_path}.")
        segments = selected["segment"].drop_duplicates().tolist()
        event_segment = event_by_name[scenario_name]["segment"]
        if len(segments) != 1 or segments[0] != event_segment or segments[0] not in ("validation", "test"):
            raise ComparisonError(f"Scenario '{scenario_name}' has inconsistent held-out segment metadata.")
        if selected["source_row_index"].duplicated().any():
            raise ComparisonError(f"Scenario '{scenario_name}' has duplicate source rows.")
        split_indices = set(getattr(split, segments[0]).index.tolist())
        scenario_indices = set(selected["source_row_index"].astype(int).tolist())
        if not scenario_indices or not scenario_indices.issubset(split_indices):
            raise ComparisonError(f"Scenario '{scenario_name}' contains rows outside its {segments[0]} split.")

        # Keep the scoring target independent and pass only the sensor Series to each detector.
        y_true = selected["anomaly_label"].copy()
        detector_input = selected["OT_injected"].copy()
        calculate_metrics(y_true, y_true)  # validates binary, nonempty labels before model execution
        control_cfg = detector_configs["statistical_control_limits"]
        control = StatisticalControlLimits(
            k=control_cfg["k"], feature_column=control_cfg.get("fit_feature", "OT")
        ).fit(split.train[control_cfg.get("fit_feature", "OT")])
        forest_cfg = detector_configs["isolation_forest"]
        forest = TemperatureIsolationForest(
            random_state=forest_cfg.get("random_state", 20261023),
            contamination=forest_cfg.get("contamination", 0.01),
            n_estimators=forest_cfg.get("n_estimators", 200),
            max_samples=forest_cfg.get("max_samples", "auto"),
            max_features=forest_cfg.get("max_features", 1.0),
            bootstrap=forest_cfg.get("bootstrap", False),
            n_jobs=forest_cfg.get("n_jobs", 1),
            fit_feature=forest_cfg.get("fit_feature", "OT"),
        ).fit(split.train[forest_cfg.get("fit_feature", "OT")])
        # Prediction functions receive only OT_injected, never y_true or a label-bearing frame.
        predictions_by_detector = {
            "statistical_control_limits": control.predict(detector_input),
            "isolation_forest": forest.predict(detector_input),
        }
        for detector_name, predicted in predictions_by_detector.items():
            if not predicted.index.equals(y_true.index):
                raise ComparisonError(f"{detector_name} predictions are not aligned for '{scenario_name}'.")
            metrics = calculate_metrics(y_true, predicted)
            records.append({
                "scenario_name": scenario_name,
                "anomaly_type": event_by_name[scenario_name]["anomaly_type"],
                "segment": segments[0],
                "detector": detector_name,
                "training_rows": len(split.train),
                "prediction_rows": len(selected),
                **metrics,
            })

    result_frame = pd.DataFrame.from_records(records)
    config_snapshot = {
        "split": split_config,
        "detectors": detector_configs,
        "scenarios": experiments["scenarios"],
    }
    summary = {
        "configuration": config_snapshot,
        "dataset": {
            "filename": raw_path.name,
            "sha256": _sha256(raw_path),
            "row_count": len(raw),
            "start_timestamp": str(pd.to_datetime(raw["date"]).iloc[0]),
            "end_timestamp": str(pd.to_datetime(raw["date"]).iloc[-1]),
        },
        "scenario_metadata": events_document,
        "evaluation_timestamp_utc": datetime.now(UTC).isoformat(),
        "scenario_identifiers": [
            {"scenario_name": row["name"], "segment": row["segment"], "anomaly_type": row["anomaly_type"]}
            for row in experiments["scenarios"]
        ],
        "metric_definitions": METRIC_DEFINITIONS,
        "zero_denominator_convention": "Precision, recall, F1, specificity, and false-positive rate are 0 when their denominator is zero.",
        "labels_used_for": "Scoring/reporting only; never detector fit or prediction inputs.",
        "detector_fit_protocol": "Each detector is fit on the clean chronological training OT series only.",
        "prediction_protocol": "Each detector predicts the scenario OT_injected Series in its held-out validation or test segment.",
        "synthetic_data_notice": "Synthetic injected anomalies are not naturally occurring confirmed faults and do not establish real-world fault performance.",
    }
    return result_frame, summary


def _render_report(metrics: pd.DataFrame, summary: dict[str, Any]) -> str:
    display = metrics[["scenario_name", "anomaly_type", "segment", "detector", "true_positives", "false_positives", "true_negatives", "false_negatives", "precision", "recall", "f1", "accuracy", "specificity", "false_positive_rate", "n_true_anomaly_points", "n_predicted_anomaly_points"]].copy()
    for column in ("precision", "recall", "f1", "accuracy", "specificity", "false_positive_rate"):
        display[column] = display[column].map(lambda value: f"{value:.4f}")
    table = "| " + " | ".join(display.columns) + " |\n"
    table += "| " + " | ".join("---" for _ in display.columns) + " |\n"
    for row in display.itertuples(index=False, name=None):
        table += "| " + " | ".join(str(value) for value in row) + " |\n"
    scenarios = "\n".join(
        f"- `{item['scenario_name']}` — {item['anomaly_type']} in **{item['segment']}**"
        for item in summary["scenario_identifiers"]
    )
    return f"""# ThermoGuard AI comparative detector evaluation

## Methodology

Both existing detectors are fit independently on the same clean chronological training segment, using original `OT` only. Each predicts the held-out scenario's `OT_injected` readings. Ground-truth `anomaly_label` is kept outside the detector APIs and used only for point-level metric calculation. Parameters come from `configs/detectors.yaml`; no test-label tuning or threshold selection is performed. Validation and test results remain identified separately.

ETTh1 fingerprint: `{summary['dataset']['sha256']}` ({summary['dataset']['row_count']} rows; `{summary['dataset']['start_timestamp']}` to `{summary['dataset']['end_timestamp']}`). Training rows per scenario: {metrics['training_rows'].iloc[0]}.

## Dataset and scenarios

{scenarios}

Ground-truth points use 1 for injected event rows and 0 otherwise. These are synthetic injected anomalies, not naturally occurring confirmed faults.

## Results

{table}

## Interpretation

These are pointwise results on the configured synthetic scenarios. The test scenarios are held-out from detector fitting and parameter selection. Results describe behavior on these specific injections; they are not evidence of performance on real industrial faults. The univariate detectors do not use timestamps or operating context.

## Limitations

- Pointwise metrics do not measure event detection delay or event-level matching.
- The existing scenarios cover a limited set of synthetic drift, spike, and variance patterns.
- Isolation Forest `contamination` and control-limit `k` are configured assumptions, not values tuned here.
- Synthetic anomaly results do not establish naturally occurring fault performance.

## Metric conventions

{summary['zero_denominator_convention']} See the machine-readable summary for metric definitions.

## Reproduction

Run from the project root in PowerShell:

```powershell
.\\.venv\\Scripts\\python.exe -m thermoguard.data.injection --input data/raw/ETTh1.csv --config configs/experiments.yaml --output data/processed/ETTh1_synthetic_scenarios.csv
.\\.venv\\Scripts\\python.exe -m thermoguard.evaluation.comparison --raw data/raw/ETTh1.csv --scenarios data/processed/ETTh1_synthetic_scenarios.csv --events data/processed/ETTh1_synthetic_scenarios.events.json --experiments-config configs/experiments.yaml --detector-config configs/detectors.yaml --output-dir reports/results/comparative_evaluation
.\\.venv\\Scripts\\python.exe -m pytest
```

Evaluation timestamp (UTC): `{summary['evaluation_timestamp_utc']}`.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare configured detectors on held-out synthetic scenarios.")
    parser.add_argument("--raw", type=Path, default=Path("data/raw/ETTh1.csv"))
    parser.add_argument("--scenarios", type=Path, default=Path("data/processed/ETTh1_synthetic_scenarios.csv"))
    parser.add_argument("--events", type=Path, default=Path("data/processed/ETTh1_synthetic_scenarios.events.json"))
    parser.add_argument("--experiments-config", type=Path, default=Path("configs/experiments.yaml"))
    parser.add_argument("--detector-config", type=Path, default=Path("configs/detectors.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/results/comparative_evaluation"))
    args = parser.parse_args(argv)
    try:
        metrics, summary = _evaluate(
            raw_path=args.raw,
            scenarios_path=args.scenarios,
            events_path=args.events,
            experiments_config_path=args.experiments_config,
            detector_config_path=args.detector_config,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = args.output_dir / "scenario_metrics.csv"
        summary_path = args.output_dir / "summary.json"
        report_path = args.output_dir / "evaluation_report.md"
        metrics.to_csv(metrics_path, index=False)
        summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        report_path.write_text(_render_report(metrics, summary), encoding="utf-8")
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError, EvaluationError) as exc:
        parser.error(str(exc))
    print(f"Metrics: {metrics_path}")
    print(f"Summary: {summary_path}")
    print(f"Report: {report_path}")
    print(metrics.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
