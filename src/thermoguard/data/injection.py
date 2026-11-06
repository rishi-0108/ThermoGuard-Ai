"""Seeded synthetic anomaly injection for held-out time-series segments.

The original feature values are retained. Injected values and labels are
written to additional columns on a returned copy; this module never writes the
raw dataset.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml

from thermoguard.data.loader import load_dataset
from thermoguard.data.splitting import ChronologicalSplit, chronological_split
from thermoguard.data.validation import validate_dataset

AnomalyType = Literal["gradual_drift", "sudden_spike", "variance_change"]
HeldOutSegment = Literal["validation", "test"]
TIMESTAMP_COLUMN = "date"


@dataclass(frozen=True)
class InjectionEvent:
    """Ground-truth metadata for one inclusive synthetic anomaly interval."""

    anomaly_type: str
    segment: str
    start_row_index: int
    end_row_index: int
    start_timestamp: str
    end_timestamp: str
    parameters: dict[str, Any]
    affected_feature: str
    seed: int


@dataclass(frozen=True)
class InjectionResult:
    """Injected copy and point-level ground-truth labels for one event."""

    data: pd.DataFrame
    event: InjectionEvent


def inject_anomaly(
    segment_frame: pd.DataFrame,
    *,
    anomaly_type: AnomalyType,
    segment: HeldOutSegment,
    start_position: int,
    duration: int,
    seed: int,
    magnitude: float | None = None,
    variance_multiplier: float | None = None,
    reference_values: pd.Series | np.ndarray | list[float] | None = None,
    affected_feature: str = "OT",
    timestamp_column: str = TIMESTAMP_COLUMN,
) -> InjectionResult:
    """Inject one event into a copy of validation or test data.

    ``start_position`` is zero-based within the selected segment. Event end
    positions are inclusive. The frame's existing index is retained and is
    stored in event metadata; for ETTh1's original dataframe this is the
    zero-based data-row number (header excluded).
    """
    if segment not in ("validation", "test"):
        raise ValueError("Anomalies may only be injected into validation or test segments.")
    if anomaly_type not in ("gradual_drift", "sudden_spike", "variance_change"):
        raise ValueError(f"Unsupported anomaly type: {anomaly_type!r}.")
    if not isinstance(segment_frame, pd.DataFrame) or segment_frame.empty:
        raise ValueError("A non-empty held-out segment DataFrame is required.")
    if affected_feature not in segment_frame.columns:
        raise ValueError(f"Affected feature '{affected_feature}' is missing from the segment.")
    if timestamp_column not in segment_frame.columns:
        raise ValueError(f"Timestamp column '{timestamp_column}' is missing from the segment.")
    if not isinstance(start_position, int) or isinstance(start_position, bool):
        raise TypeError("start_position must be an integer segment offset.")
    if not isinstance(duration, int) or isinstance(duration, bool) or duration <= 0:
        raise ValueError("duration must be a positive integer number of rows.")
    if start_position < 0 or start_position + duration > len(segment_frame):
        raise ValueError("The full event must fit inside its assigned held-out segment.")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer.")

    index_values = segment_frame.index
    if not index_values.is_unique or not all(isinstance(value, (int, np.integer)) for value in index_values):
        raise ValueError("Segment index must contain unique integer source-row indices.")
    timestamps = pd.to_datetime(segment_frame[timestamp_column], errors="coerce")
    if timestamps.isna().any() or not timestamps.is_monotonic_increasing:
        raise ValueError("Segment timestamps must be parseable and chronologically ordered.")

    original = segment_frame[affected_feature]
    if not pd.api.types.is_numeric_dtype(original.dtype):
        raise ValueError(f"Affected feature '{affected_feature}' must be numeric.")
    original_values = original.to_numpy(dtype=float, copy=True)
    if not np.isfinite(original_values).all():
        raise ValueError(f"Affected feature '{affected_feature}' contains non-finite values.")

    event_stop = start_position + duration
    event_slice = slice(start_position, event_stop)
    clean_event_values = original_values[event_slice].copy()
    injected_event_values = clean_event_values.copy()
    parameters: dict[str, Any] = {
        "start_position": start_position,
        "duration_rows": duration,
    }

    if anomaly_type in ("gradual_drift", "sudden_spike"):
        if magnitude is None or not math.isfinite(float(magnitude)) or float(magnitude) == 0:
            raise ValueError(f"{anomaly_type} requires a finite, non-zero magnitude.")
        magnitude = float(magnitude)
        parameters["magnitude"] = magnitude
        if anomaly_type == "gradual_drift":
            ramp = np.linspace(magnitude / duration, magnitude, duration, dtype=float)
            injected_event_values += ramp
            parameters["ramp_start_offset"] = float(ramp[0])
            parameters["ramp_end_offset"] = float(ramp[-1])
        else:
            injected_event_values += magnitude
    else:
        if variance_multiplier is None or not math.isfinite(float(variance_multiplier)):
            raise ValueError("variance_change requires a finite variance_multiplier greater than 1.")
        variance_multiplier = float(variance_multiplier)
        if variance_multiplier <= 1:
            raise ValueError("variance_multiplier must be greater than 1 to increase variability.")
        if duration < 3:
            raise ValueError("variance_change requires duration of at least 3 rows.")
        if reference_values is None:
            raise ValueError("variance_change requires clean training reference_values.")
        reference = np.asarray(reference_values, dtype=float)
        reference = reference[np.isfinite(reference)]
        if reference.size < 2:
            raise ValueError("variance_change needs at least two finite clean training reference values.")
        reference_std = float(np.std(reference, ddof=1))
        if not math.isfinite(reference_std) or reference_std <= 0:
            raise ValueError("variance_change needs non-zero variability in clean training reference_values.")

        # Create seeded noise with zero mean and no sample covariance with the
        # event's original centered values. Its scale then adds the requested
        # training-reference variance component without shifting event mean.
        rng = np.random.default_rng(seed)
        noise = rng.normal(size=duration)
        noise -= noise.mean()
        centered_event = clean_event_values - clean_event_values.mean()
        centered_norm_sq = float(np.dot(centered_event, centered_event))
        if centered_norm_sq > np.finfo(float).eps:
            noise -= (float(np.dot(noise, centered_event)) / centered_norm_sq) * centered_event
            noise -= noise.mean()
        noise_std = float(np.std(noise, ddof=1))
        if not math.isfinite(noise_std) or noise_std <= np.finfo(float).eps:
            raise ValueError("Could not construct variance-change noise for this event duration.")
        target_noise_std = reference_std * math.sqrt(variance_multiplier - 1.0)
        injected_event_values += noise * (target_noise_std / noise_std)
        parameters.update(
            {
                "variance_multiplier": variance_multiplier,
                "reference_std": reference_std,
                "added_noise_std": target_noise_std,
            }
        )

    if not np.isfinite(injected_event_values).all():
        raise ValueError("Injection produced non-finite values; no result was returned.")

    result_frame = segment_frame.copy(deep=True)
    injected_values = original_values.copy()
    injected_values[event_slice] = injected_event_values
    result_frame[f"{affected_feature}_injected"] = injected_values
    result_frame["anomaly_label"] = np.zeros(len(result_frame), dtype=np.uint8)
    result_frame.iloc[start_position:event_stop, result_frame.columns.get_loc("anomaly_label")] = 1

    start_index = int(index_values[start_position])
    end_index = int(index_values[event_stop - 1])
    event = InjectionEvent(
        anomaly_type=anomaly_type,
        segment=segment,
        start_row_index=start_index,
        end_row_index=end_index,
        start_timestamp=timestamps.iloc[start_position].isoformat(sep=" "),
        end_timestamp=timestamps.iloc[event_stop - 1].isoformat(sep=" "),
        parameters=parameters,
        affected_feature=affected_feature,
        seed=seed,
    )
    return InjectionResult(data=result_frame, event=event)


def _run_config(config_path: Path, input_path: Path, output_path: Path) -> tuple[Path, Path]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("split"), dict):
        raise TypeError("Injection config must define a split mapping.")
    split_config = config["split"]
    frame = load_dataset(input_path)
    issues = validate_dataset(frame)
    if issues:
        raise ValueError("Source dataset validation failed: " + " | ".join(issues))

    split: ChronologicalSplit = chronological_split(
        frame,
        train_fraction=split_config["train_fraction"],
        validation_fraction=split_config["validation_fraction"],
        test_fraction=split_config["test_fraction"],
    )
    root_seed = config.get("seed")
    if not isinstance(root_seed, int) or isinstance(root_seed, bool):
        raise TypeError("Injection config must define an integer seed.")

    outputs: list[pd.DataFrame] = []
    events: list[dict[str, Any]] = []
    scenarios = config.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("Injection config must define at least one scenario.")
    for scenario in scenarios:
        if not isinstance(scenario, dict):
            raise TypeError("Each scenario must be a mapping.")
        name = scenario.get("name")
        segment_name = scenario.get("segment")
        if not isinstance(name, str) or not name:
            raise ValueError("Each scenario must have a non-empty name.")
        if segment_name not in ("validation", "test"):
            raise ValueError(f"Scenario '{name}' must target validation or test.")
        segment_frame = getattr(split, segment_name)
        scenario_seed = scenario.get("seed", root_seed)
        kwargs: dict[str, Any] = {
            "anomaly_type": scenario.get("anomaly_type"),
            "segment": segment_name,
            "start_position": scenario.get("start_position"),
            "duration": scenario.get("duration"),
            "seed": scenario_seed,
            "affected_feature": scenario.get("affected_feature", "OT"),
        }
        for field in ("magnitude", "variance_multiplier"):
            if field in scenario:
                kwargs[field] = scenario[field]
        if kwargs["anomaly_type"] == "variance_change":
            kwargs["reference_values"] = split.train[kwargs["affected_feature"]]

        injection = inject_anomaly(segment_frame, **kwargs)
        output = injection.data.copy()
        output["scenario_name"] = name
        output["segment"] = segment_name
        outputs.append(output)
        events.append(asdict(injection.event) | {"scenario_name": name})

    combined = pd.concat(outputs, axis=0, ignore_index=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index_label="source_row_index")
    event_path = output_path.with_suffix(".events.json")
    event_path.write_text(
        json.dumps(
            {
                "source_filename": input_path.name,
                "source_sha256": _sha256(input_path),
                "split": split_config,
                "root_seed": root_seed,
                "events": events,
                "label_convention": "anomaly_label=1 for inclusive event rows; 0 otherwise",
                "synthetic_only_notice": "These are synthetic injected anomalies, not naturally occurring confirmed faults.",
            },
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_path, event_path


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate configured synthetic ETTh1 scenarios.")
    parser.add_argument("--input", type=Path, default=Path("data/raw/ETTh1.csv"))
    parser.add_argument("--config", type=Path, default=Path("configs/experiments.yaml"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/ETTh1_synthetic_scenarios.csv"))
    args = parser.parse_args(argv)
    try:
        data_path, event_path = _run_config(args.config, args.input, args.output)
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    print(f"Synthetic scenario data: {data_path}")
    print(f"Ground-truth event metadata: {event_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
