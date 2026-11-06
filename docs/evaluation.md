# Comparative detector evaluation

The evaluation runner compares the existing statistical control-limit and
Isolation Forest baselines on every scenario listed in
`configs/experiments.yaml`. It preserves the chronological split: each model is
fit separately on the clean training segment's original `OT` readings and
predicts the scenario's held-out `OT_injected` readings. Validation scenarios
and test scenarios keep their segment identifiers in all outputs.

`anomaly_label` is read only for scoring after a detector's prediction has been
produced. It is never passed to `fit` or `predict`, and no test labels are used
to choose `k`, `contamination`, or another parameter. The configured values
remain assumptions for this comparison; this chunk performs no tuning.

## Metrics

The tidy CSV contains confusion counts, precision, recall, F1, accuracy,
specificity, false-positive rate, true anomaly-point count, and predicted
anomaly-point count for each scenario and detector. If precision, recall, F1,
specificity, or false-positive rate has a zero denominator, the result is
defined as `0`. Accuracy is defined for every nonempty scenario.

Metrics are pointwise. They do not capture detection delay, event-level
matching, or operational cost. The scenarios are controlled synthetic
injections and do not establish either detector's performance on naturally
occurring industrial faults.

## Outputs and reproducibility

From the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m thermoguard.data.injection --input data/raw/ETTh1.csv --config configs/experiments.yaml --output data/processed/ETTh1_synthetic_scenarios.csv
.\.venv\Scripts\python.exe -m thermoguard.evaluation.comparison --raw data/raw/ETTh1.csv --scenarios data/processed/ETTh1_synthetic_scenarios.csv --events data/processed/ETTh1_synthetic_scenarios.events.json --experiments-config configs/experiments.yaml --detector-config configs/detectors.yaml --output-dir reports/results/comparative_evaluation
.\.venv\Scripts\python.exe -m pytest
```

The command writes `scenario_metrics.csv`, `summary.json`, and
`evaluation_report.md` below the ignored `reports/results/comparative_evaluation/`
directory. The JSON includes the UTC run timestamp, raw dataset SHA-256,
configuration, event metadata, split/scenario identifiers, and metric
definitions. The report carries methodology, results, interpretation,
limitations, and reproduction commands. These filenames are separate from the
individual detector prediction artifacts.
