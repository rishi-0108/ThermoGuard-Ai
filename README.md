# ThermoGuard AI — Temperature Anomaly Detection

ThermoGuard AI is a college Applied AI project for identifying unusual
transformer temperature readings in the ETT hourly time series. It compares a
statistical control-limit baseline with Isolation Forest, evaluates both on
controlled held-out scenarios, and presents the results in an interactive
Streamlit dashboard.

## Objectives and features

- Validate and profile local ETTh1 measurements without changing the source.
- Preserve chronological train, validation, and test boundaries.
- Create reproducible synthetic gradual drift, spike, and variance-change
  scenarios with point-level labels and event metadata.
- Compare statistical control limits and Isolation Forest with pointwise
  classification metrics.
- Provide an interactive time-series chart, detector predictions, evaluation
  metrics, and data-quality status.
- Keep experiment settings, commands, tests, and reports reproducible.

**Synthetic-data notice:** injected anomalies are controlled examples, not
confirmed real-world faults. Their results do not prove detector performance
on naturally occurring industrial failures.

## Dataset and licensing caveat

The initial dataset is ETTh1 from the [official ETDataset repository](https://github.com/zhouhaoyi/ETDataset),
file `ETT-small/ETTh1.csv` (hourly; `OT` is oil temperature). The repository
states CC BY-ND 4.0, but the repository-level license does not clearly identify
whether or how it applies to CSV assets. Applicability remains uncertain; this
project does not claim redistribution rights. Obtain the dataset locally and do
not commit or redistribute the source file or generated derivatives. See
[data-source notes](docs/data_sources.md) before acquisition or release.

## Environment setup

The project targets Python 3.13.7. In Windows PowerShell from
`D:\Code\ThermoSense-AI`:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Use `\.venv\Scripts\python.exe` for the commands below so they run with the
project environment.

## Full local workflow

Keep the unchanged source at `data/raw/ETTh1.csv`. These commands use the
configured paths; scenario, prediction, profile, and evaluation outputs are
data-dependent and ignored by Git.

```powershell
.\.venv\Scripts\python.exe -m thermoguard.data.profile --input data/raw/ETTh1.csv --output reports/results/ETTh1_profile.json
.\.venv\Scripts\python.exe -m thermoguard.data.injection --input data/raw/ETTh1.csv --config configs/experiments.yaml --output data/processed/ETTh1_synthetic_scenarios.csv
.\.venv\Scripts\python.exe -m thermoguard.detectors.control_limits --raw data/raw/ETTh1.csv --scenarios data/processed/ETTh1_synthetic_scenarios.csv --experiments-config configs/experiments.yaml --detector-config configs/detectors.yaml --scenario validation_gradual_drift --output reports/results/control_limits_validation_gradual_drift.csv
.\.venv\Scripts\python.exe -m thermoguard.detectors.isolation_forest --raw data/raw/ETTh1.csv --scenarios data/processed/ETTh1_synthetic_scenarios.csv --experiments-config configs/experiments.yaml --detector-config configs/detectors.yaml --scenario validation_gradual_drift --output reports/results/isolation_forest_validation_gradual_drift.csv
.\.venv\Scripts\python.exe -m thermoguard.evaluation.comparison --raw data/raw/ETTh1.csv --scenarios data/processed/ETTh1_synthetic_scenarios.csv --events data/processed/ETTh1_synthetic_scenarios.events.json --experiments-config configs/experiments.yaml --detector-config configs/detectors.yaml --output-dir reports/results/comparative_evaluation
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\ruff.exe check src tests
```

## Dashboard demonstration

Launch from the project root with:

```powershell
.\.venv\Scripts\python.exe -m streamlit run src/thermoguard/dashboard/app.py
```

Select a scenario and detector, review training/prediction counts and the chart,
then open the evaluation-only metric section. Scenario labels are used only
for offline metric scoring, never for detector fit or prediction. For a guided
sequence and known limitations, see the [dashboard guide](docs/dashboard.md)
and [submission checklist](docs/submission_checklist.md).
