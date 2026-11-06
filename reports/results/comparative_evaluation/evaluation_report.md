# ThermoGuard AI comparative detector evaluation

## Methodology

Both existing detectors are fit independently on the same clean chronological training segment, using original `OT` only. Each predicts the held-out scenario's `OT_injected` readings. Ground-truth `anomaly_label` is kept outside the detector APIs and used only for point-level metric calculation. Parameters come from `configs/detectors.yaml`; no test-label tuning or threshold selection is performed. Validation and test results remain identified separately.

ETTh1 fingerprint: `f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066` (17420 rows; `2016-07-01 00:00:00` to `2018-06-26 19:00:00`). Training rows per scenario: 12194.

## Dataset and scenarios

- `validation_gradual_drift` — gradual_drift in **validation**
- `test_positive_spike` — sudden_spike in **test**
- `test_negative_spike` — sudden_spike in **test**
- `test_variance_increase` — variance_change in **test**

Ground-truth points use 1 for injected event rows and 0 otherwise. These are synthetic injected anomalies, not naturally occurring confirmed faults.

## Results

| scenario_name | anomaly_type | segment | detector | true_positives | false_positives | true_negatives | false_negatives | precision | recall | f1 | accuracy | specificity | false_positive_rate | n_true_anomaly_points | n_predicted_anomaly_points |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| validation_gradual_drift | gradual_drift | validation | statistical_control_limits | 0 | 0 | 2541 | 72 | 0.0000 | 0.0000 | 0.0000 | 0.9724 | 1.0000 | 0.0000 | 72 | 0 |
| validation_gradual_drift | gradual_drift | validation | isolation_forest | 0 | 308 | 2233 | 72 | 0.0000 | 0.0000 | 0.0000 | 0.8546 | 0.8788 | 0.1212 | 72 | 308 |
| test_positive_spike | sudden_spike | test | statistical_control_limits | 0 | 0 | 2610 | 3 | 0.0000 | 0.0000 | 0.0000 | 0.9989 | 1.0000 | 0.0000 | 3 | 0 |
| test_positive_spike | sudden_spike | test | isolation_forest | 0 | 19 | 2591 | 3 | 0.0000 | 0.0000 | 0.0000 | 0.9916 | 0.9927 | 0.0073 | 3 | 19 |
| test_negative_spike | sudden_spike | test | statistical_control_limits | 0 | 0 | 2611 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.9992 | 1.0000 | 0.0000 | 2 | 0 |
| test_negative_spike | sudden_spike | test | isolation_forest | 2 | 19 | 2592 | 0 | 0.0952 | 1.0000 | 0.1739 | 0.9927 | 0.9927 | 0.0073 | 2 | 21 |
| test_variance_increase | variance_change | test | statistical_control_limits | 2 | 0 | 2565 | 46 | 1.0000 | 0.0417 | 0.0800 | 0.9824 | 1.0000 | 0.0000 | 48 | 2 |
| test_variance_increase | variance_change | test | isolation_forest | 7 | 19 | 2546 | 41 | 0.2692 | 0.1458 | 0.1892 | 0.9770 | 0.9926 | 0.0074 | 48 | 26 |


## Interpretation

These are pointwise results on the configured synthetic scenarios. The test scenarios are held-out from detector fitting and parameter selection. Results describe behavior on these specific injections; they are not evidence of performance on real industrial faults. The univariate detectors do not use timestamps or operating context.

## Limitations

- Pointwise metrics do not measure event detection delay or event-level matching.
- The existing scenarios cover a limited set of synthetic drift, spike, and variance patterns.
- Isolation Forest `contamination` and control-limit `k` are configured assumptions, not values tuned here.
- Synthetic anomaly results do not establish naturally occurring fault performance.

## Metric conventions

Precision, recall, F1, specificity, and false-positive rate are 0 when their denominator is zero. See the machine-readable summary for metric definitions.

## Reproduction

Run from the project root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m thermoguard.data.injection --input data/raw/ETTh1.csv --config configs/experiments.yaml --output data/processed/ETTh1_synthetic_scenarios.csv
.\.venv\Scripts\python.exe -m thermoguard.evaluation.comparison --raw data/raw/ETTh1.csv --scenarios data/processed/ETTh1_synthetic_scenarios.csv --events data/processed/ETTh1_synthetic_scenarios.events.json --experiments-config configs/experiments.yaml --detector-config configs/detectors.yaml --output-dir reports/results/comparative_evaluation
.\.venv\Scripts\python.exe -m pytest
```

Evaluation timestamp (UTC): `2026-10-22T18:51:53.188064+00:00`.
