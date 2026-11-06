"""Streamlit dashboard for ThermoGuard AI's configured ETTh1 scenarios."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

from thermoguard.dashboard.service import (
    DETECTOR_LABELS,
    DashboardDataError,
    build_scenario_predictions,
    describe_sampling_interval,
    format_date_range,
    load_dashboard_sources,
    load_optional_metrics,
    lookup_evaluation_metrics,
    resolve_event_duration,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_PATH = PROJECT_ROOT / "data/raw/ETTh1.csv"
SCENARIO_PATH = PROJECT_ROOT / "data/processed/ETTh1_synthetic_scenarios.csv"
EVENTS_PATH = PROJECT_ROOT / "data/processed/ETTh1_synthetic_scenarios.events.json"
EXPERIMENTS_PATH = PROJECT_ROOT / "configs/experiments.yaml"
DETECTORS_PATH = PROJECT_ROOT / "configs/detectors.yaml"
METRICS_PATH = PROJECT_ROOT / "reports/results/comparative_evaluation/scenario_metrics.csv"


def _load_yaml(path: Path) -> dict:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise DashboardDataError(f"Could not read configuration '{path}': {exc}") from exc
    if not isinstance(loaded, dict):
        raise DashboardDataError(f"Configuration '{path}' must contain a YAML mapping.")
    return loaded


def _format_metric(value: object) -> str:
    if pd.isna(value):
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _make_chart(frame: pd.DataFrame, scenario_name: str, detector_label: str) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=frame["date"], y=frame["OT"], mode="lines", name="Original OT",
        line={"color": "#46d7b5", "width": 2.2},
        hovertemplate="%{x}<br>Original OT: %{y:.2f}<extra></extra>",
    ))
    figure.add_trace(go.Scatter(
        x=frame["date"], y=frame["OT_injected"], mode="lines", name="Scenario OT",
        line={"color": "#ffb000", "width": 2.2, "dash": "dash"},
        hovertemplate="%{x}<br>Scenario OT: %{y:.2f}<extra></extra>",
    ))
    flagged = frame.loc[frame["prediction"] == 1]
    figure.add_trace(go.Scatter(
        x=flagged["date"], y=flagged["OT_injected"], mode="markers", name=f"{detector_label} prediction",
        marker={"color": "#ff4d6d", "size": 10, "symbol": "diamond", "line": {"color": "white", "width": 1}},
        customdata=flagged["anomaly_score"],
        hovertemplate="%{x}<br>Predicted anomaly<br>Scenario OT: %{y:.2f}<br>Score: %{customdata:.4f}<extra></extra>",
    ))
    figure.update_layout(
        title=f"{scenario_name} · temperature readings and predicted anomalies",
        xaxis_title="Timestamp",
        yaxis_title="Oil temperature (OT)",
        height=430,
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        margin={"l": 20, "r": 20, "t": 90, "b": 20},
    )
    return figure


def main() -> None:
    st.set_page_config(page_title="ThermoGuard AI", page_icon="🌡️", layout="wide")
    st.title("ThermoGuard AI — Temperature Anomaly Detection")
    st.markdown(
        "Explore unusual transformer-temperature readings in configured ETT scenarios. "
        "Detectors are fit on clean training data and applied to held-out scenario readings."
    )
    st.warning(
        "Synthetic-data notice: injected anomalies are controlled examples for evaluation. "
        "They are not confirmed real-world faults and do not establish real industrial performance."
    )

    try:
        experiments = _load_yaml(EXPERIMENTS_PATH)
        detector_root = _load_yaml(DETECTORS_PATH)
        raw, quality_issues, scenarios = load_dashboard_sources(RAW_PATH, SCENARIO_PATH)
    except DashboardDataError as exc:
        st.error(f"Dashboard data is unavailable: {exc}")
        st.info(
            "Check that `data/raw/ETTh1.csv` and the generated scenario CSV exist. "
            "See `docs/data_sources.md` and `docs/dashboard.md` for local setup commands."
        )
        st.stop()

    configured_scenarios = experiments.get("scenarios", [])
    if not configured_scenarios:
        st.error("No scenarios are configured in configs/experiments.yaml.")
        st.stop()
    scenario_names = [item["name"] for item in configured_scenarios]
    left, right = st.columns(2)
    with left:
        scenario_name = st.selectbox("Synthetic scenario", scenario_names)
    with right:
        detector_name = st.selectbox(
            "Detector", list(DETECTOR_LABELS), format_func=lambda name: DETECTOR_LABELS[name]
        )
    selected_config = next(item for item in configured_scenarios if item["name"] == scenario_name)
    detector_config = detector_root.get("detectors", {}).get(detector_name)
    if not isinstance(detector_config, dict):
        st.error(f"No detector configuration found for `{detector_name}`.")
        st.stop()
    try:
        prediction_frame, prediction_summary = build_scenario_predictions(
            raw,
            scenarios,
            scenario_name=scenario_name,
            detector_name=detector_name,
            scenario_config=selected_config,
            split_config=experiments["split"],
            detector_config=detector_config,
        )
    except (DashboardDataError, ValueError, KeyError, TypeError) as exc:
        st.error(f"Could not prepare predictions: {exc}")
        st.info("Regenerate the configured scenario artifact and check the project configuration.")
        st.stop()

    st.subheader("Dataset overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{len(raw):,}")
    col2.metric("Measured interval", describe_sampling_interval(raw))
    col3.metric("Temperature-related signals", "7")
    st.markdown(f"**Date range:** {format_date_range(raw['date'])}")
    with st.expander("Show sensor columns (7)"):
        st.code("HUFL · HULL · MUFL · MULL · LUFL · LULL · OT", language=None)
    st.caption("ETTh1 source: Zhou et al., ETT dataset repository. Local source data is read-only in this app.")

    st.subheader("Selected scenario predictions")
    stat1, stat2, stat3, stat4 = st.columns(4)
    stat1.metric("Observations analyzed", f"{prediction_summary['prediction_rows']:,}")
    stat2.metric("Predicted anomalies", f"{prediction_summary['predicted_anomalies']:,}")
    stat3.metric("Predicted anomaly rate", f"{prediction_summary['anomaly_percentage']:.2f}%")
    stat4.metric("Segment", prediction_summary["segment"].title())
    st.caption(
        f"Fit rows: {prediction_summary['training_rows']:,} clean training OT observations. "
        "Predictions use OT_injected; synthetic labels are not detector inputs."
    )
    st.plotly_chart(
        _make_chart(prediction_frame, scenario_name, DETECTOR_LABELS[detector_name]),
        width="stretch",
    )
    with st.expander("Selected detector settings"):
        st.json(prediction_summary["settings"])

    st.subheader("Evaluation-only metrics")
    st.caption(
        "Metrics compare detector predictions with synthetic ground-truth labels for scoring only. "
        "Validation and test scenarios are distinguished; these are not confirmed faults."
    )
    try:
        evaluation_metrics = load_optional_metrics(METRICS_PATH)
        metric_row = lookup_evaluation_metrics(
            evaluation_metrics, scenario_name=scenario_name, detector_name=detector_name
        )
    except DashboardDataError as exc:
        st.warning(f"Evaluation metrics are unavailable: {exc}")
        metric_row = None
    if metric_row is None:
        st.info(
            "No matching evaluation row is available. Run the comparative evaluation command "
            "in `docs/evaluation.md` to generate it."
        )
    else:
        st.write(f"Scenario split: **{metric_row['segment']}** · Scenario: **{scenario_name}**")
        if int(metric_row["n_true_anomaly_points"]) > 0 and prediction_summary["predicted_anomalies"] == 0:
            st.info(
                f"The evaluation artifact contains {int(metric_row['n_true_anomaly_points'])} "
                "synthetic anomaly points. Under the current settings, this detector did not "
                "flag the injected points in this scenario."
            )
        metric_columns = st.columns(6)
        metric_keys = ["precision", "recall", "f1", "accuracy", "specificity", "false_positive_rate"]
        for column, key in zip(metric_columns, metric_keys, strict=True):
            column.metric(key.replace("_", " ").title(), _format_metric(metric_row.get(key)))
        st.caption(
            f"TP {int(metric_row['true_positives'])} · FP {int(metric_row['false_positives'])} · "
            f"TN {int(metric_row['true_negatives'])} · FN {int(metric_row['false_negatives'])} · "
            f"Synthetic anomaly points {int(metric_row['n_true_anomaly_points'])} · "
            f"Predicted anomaly points {int(metric_row['n_predicted_anomaly_points'])}"
        )

    st.subheader("Configured scenario overview")
    group_sizes = scenarios.groupby("scenario_name", sort=False).size().to_dict()
    try:
        event_document = json.loads(EVENTS_PATH.read_text(encoding="utf-8"))
        events_by_name = {event["scenario_name"]: event for event in event_document.get("events", [])}
    except (OSError, json.JSONDecodeError, TypeError):
        events_by_name = {}
    overview = pd.DataFrame([
        {
            "Scenario": item["name"],
            "Synthetic pattern": item["anomaly_type"],
            "Held-out segment": item["segment"],
            "Configured event duration": resolve_event_duration(item, events_by_name.get(item["name"])),
            "Scenario rows": group_sizes.get(item["name"], 0),
        }
        for item in configured_scenarios
    ])
    st.dataframe(overview, hide_index=True, width="stretch")
    st.caption("Scenario patterns and event durations describe synthetic injections, not confirmed fault events.")

    with st.expander("How the detectors work and their limitations"):
        st.markdown(
            "**Statistical control limits** flag values outside the clean training mean ± k sample "
            "standard deviations. This global limit does not model seasonality or operating regimes.\n\n"
            "**Isolation Forest** isolates unusual values through randomized tree partitions. Its "
            "contamination setting controls a decision threshold; it is not an observed fault rate. "
            "The current detector is univariate and does not model time order.\n\n"
            "Both methods fit only clean training OT and predict held-out OT_injected values. Synthetic "
            "results do not establish performance on naturally occurring industrial faults."
        )

    st.subheader("Data quality")
    if quality_issues:
        st.error(f"Dataset validation found {len(quality_issues)} issue(s).")
        st.dataframe(pd.DataFrame({"Validation issue": quality_issues}), hide_index=True, width="stretch")
    else:
        st.success("Dataset validation passed: required fields, chronological hourly timestamps, numeric sensors, and finite non-missing values.")
    st.caption(f"Source file: `{RAW_PATH.relative_to(PROJECT_ROOT)}` · Validation is non-mutating.")


if __name__ == "__main__":
    main()
