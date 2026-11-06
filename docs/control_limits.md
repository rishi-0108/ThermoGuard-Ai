# Statistical control-limit baseline

`StatisticalControlLimits` is the first baseline detector. Its CLI loads and validates the unchanged ETTh1 source, creates the configured chronological split, and fits only the clean training segment's original `OT` values. Fit input containing `anomaly_label` or `OT_injected` is rejected. Prediction uses the configured held-out scenario's `OT_injected` values; its CSV reader explicitly selects columns without loading `anomaly_label`.

For clean training observations (x_1, \ldots, x_n), the fitted mean is

\[
\bar{x} = \frac{1}{n}\sum_{i=1}^n x_i
\]

and the fitted sample standard deviation uses `ddof=1`. With configured (k > 0), limits are

\[
L = \bar{x} - k s, \qquad U = \bar{x} + k s.
\]

Prediction is `1` if a value is strictly below (L) or strictly above (U), and `0` otherwise. Values exactly on a limit are normal. The optional score is the signed standardized deviation ((x - \bar{x}) / s). Predictions and scores retain the input Series/DataFrame index.

Empty, too-small, nonnumeric, missing, non-finite, and zero-variance training inputs are rejected with a clear error. Zero variance is not converted into a zero-width detector. Prediction inputs receive the same numeric and finite-value checks. This is a simple global limit: it assumes a reasonably stable process distribution and does not model seasonality, autocorrelation, changing operating regimes, or event context. Its control limits are not a physical safety specification.

`k` is set in `configs/detectors.yaml`. Choose it from domain assumptions or, in a later evaluation stage, validation data. Do not tune it on test labels. The prediction command does not load labels or calculate detection metrics.

## Reproduce a held-out prediction

From the project root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m thermoguard.data.injection --input data/raw/ETTh1.csv --config configs/experiments.yaml --output data/processed/ETTh1_synthetic_scenarios.csv
.\.venv\Scripts\python.exe -m thermoguard.detectors.control_limits --raw data/raw/ETTh1.csv --scenarios data/processed/ETTh1_synthetic_scenarios.csv --experiments-config configs/experiments.yaml --detector-config configs/detectors.yaml --scenario validation_gradual_drift --output reports/results/control_limits_validation_gradual_drift.csv
.\.venv\Scripts\python.exe -m pytest
```

The prediction command writes a data-dependent CSV and a `.summary.json` sidecar under the ignored `reports/results/` directory. The CSV contains the scenario's source row index, timestamp, original `OT`, injected value, scenario/segment names, signed score, and binary prediction; it omits ground-truth labels. Both outputs are reproducible for the same dataset, split configuration, scenario configuration, and `k`.
