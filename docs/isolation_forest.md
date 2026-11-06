# Isolation Forest baseline

`TemperatureIsolationForest` uses scikit-learn's Isolation Forest to identify
observations that are easier to isolate in randomly partitioned trees. This
Chunk 6 implementation is univariate: it fits only the clean training segment's
original `OT` readings, as a single numeric feature. It predicts held-out
scenario readings from `OT_injected`. It does not load or accept
`anomaly_label` as a model feature.

The sklearn `predict` convention is `-1` for outliers and `1` for inliers; this
module maps those values to `1` and `0`, respectively. The reported anomaly
score is `-score_samples`, so a larger value means a more anomalous observation.
Predictions and scores preserve the input row index and order.

## Leakage controls and limitations

The command validates and chronologically splits the unchanged raw dataset,
then fits solely on the clean training `OT` series. It reads only selected
scenario columns and excludes ground-truth labels. A requested scenario must
belong wholly to validation or test and its source indices must be inside that
segment. No held-out data or labels tune the model. The prediction artifact
contains no ground-truth columns or performance metrics.

Isolation Forest is sensitive to distribution changes and does not model time
order, seasonality, or operating context in this univariate setup.
`contamination` controls the model's decision threshold (the expected fraction
of training observations treated as outliers); it is not evidence about the
true fault rate. Select it from domain assumptions or a later validation-only
procedure, never from test labels. Synthetic injected anomalies are controlled
test cases and do not establish performance on naturally occurring faults.

## Configuration

`configs/detectors.yaml` defines `random_state` (reproducibility),
`contamination` (`auto` or a proportion in `(0, 0.5]`), `n_estimators`,
`max_samples` (`auto`, an integer, or a fraction), `max_features` (fraction),
`bootstrap`, and `n_jobs`. `fit_feature` must remain clean `OT` and
`prediction_feature` must remain `OT_injected` for this command. The default
scenario is `validation_gradual_drift`.

## Run

From the project root in PowerShell (first regenerate the scenario artifact if
needed):

```powershell
.\.venv\Scripts\python.exe -m thermoguard.data.injection --input data/raw/ETTh1.csv --config configs/experiments.yaml --output data/processed/ETTh1_synthetic_scenarios.csv
.\.venv\Scripts\python.exe -m thermoguard.detectors.isolation_forest --raw data/raw/ETTh1.csv --scenarios data/processed/ETTh1_synthetic_scenarios.csv --experiments-config configs/experiments.yaml --detector-config configs/detectors.yaml --scenario validation_gradual_drift --output reports/results/isolation_forest_validation_gradual_drift.csv
.\.venv\Scripts\python.exe -m pytest
```

The command writes a prediction CSV and same-stem `.summary.json` under the
ignored `reports/results/` directory. The summary reports counts and model
configuration only, without comparing predictions to ground truth.
