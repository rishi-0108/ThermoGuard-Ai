# ThermoGuard AI Streamlit dashboard

## Launch

From `D:\Code\ThermoSense-AI` in Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m streamlit run src/thermoguard/dashboard/app.py
```

The dashboard uses the project's `.venv` and does not need a browser automation
package.

## Views and workflow

Choose one configured synthetic scenario and one of the two existing detectors.
The page then:

1. Shows the ETTh1 date range, row count, measured interval, and temperature
   signal columns.
2. Fits the selected detector on clean training `OT` only, then predicts the
   held-out scenario's `OT_injected` values.
3. Charts original and scenario temperature readings with detector-predicted
   anomaly markers, preserving timestamp order.
4. Summarizes prediction count/rate and the configured detector settings.
5. Shows matching metrics from the existing comparative evaluation artifact in
   an explicitly labeled **Evaluation-only metrics** section. These metrics use
   synthetic ground truth for scoring; labels are never passed to detector fit
   or prediction APIs.
6. Lists configured scenario type, held-out split, duration, and scenario row
   count, followed by detector limitations and data-quality results.

The prominent synthetic-data notice applies throughout: injected anomalies
are not confirmed real-world faults, and these results do not establish
performance on naturally occurring industrial faults.

## Local artifacts

Required inputs:

- `data/raw/ETTh1.csv` — locally acquired original source file.
- `data/processed/ETTh1_synthetic_scenarios.csv` — generated scenarios and
  labels. The dashboard reads labels only as part of the data artifact; the
  prediction service selects only `OT_injected` before calling a detector.
- `configs/experiments.yaml` and `configs/detectors.yaml`.

Optional evaluation artifact:

- `reports/results/comparative_evaluation/scenario_metrics.csv`.

If it is absent, the dashboard continues to work and explains how to generate
the evaluation. Missing required inputs, malformed scenario artifacts, and
invalid selections show actionable error messages. The app does not download,
rewrite, or regenerate data, and it does not offer model retraining or
parameter-tuning controls. It fits a fresh selected detector on clean training
observations during a Streamlit rerun; held-out data never participates in fit.

Use [the data-source guide](data_sources.md) for source/license notes and
[the evaluation guide](evaluation.md) for scenario/evaluation reproduction
commands.
