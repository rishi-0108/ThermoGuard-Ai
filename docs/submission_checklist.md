# ThermoGuard AI submission checklist

## Before a demo

- [ ] Use the project root `D:\Code\ThermoSense-AI` and the project's `.venv`.
- [ ] Confirm the locally acquired `data/raw/ETTh1.csv` is present and unchanged.
- [ ] Review `configs/experiments.yaml` and `configs/detectors.yaml` so the
  audience knows which split, scenarios, and detector assumptions are active.
- [ ] Confirm the generated scenario CSV and `.events.json` are present under
  `data/processed/`.
- [ ] Confirm comparative metrics exist at
  `reports/results/comparative_evaluation/scenario_metrics.csv`.
- [ ] Start the dashboard with the command at the end of this document.
- [ ] Keep the notice visible: anomalies are synthetic examples, not confirmed
  real faults.

## Recommended dashboard demonstration

1. Start on `validation_gradual_drift` with statistical control limits. Point
   out the original OT series, injected scenario readings, and model prediction
   markers on the chronological chart.
2. Show training rows, analyzed rows, predicted anomaly count/rate, and settings.
   Emphasize that the model fits clean training OT and predicts the held-out
   `OT_injected` signal.
3. Open **Evaluation-only metrics**. Explain that validation/test designation
   is retained and synthetic labels are consulted only to score predictions.
4. Switch to Isolation Forest and compare its predictions with the same
   scenario. Then show a test spike or variance-change scenario to illustrate
   that behavior depends on the anomaly pattern.
5. Review the scenario overview and data-quality section. Close by stating the
   limitations and the synthetic-data caveat.

## Detectors and metrics

- **Statistical control limits** flag values outside training mean ± `k` sample
  standard deviations. This is a simple global univariate limit; it does not
  account for seasonality, serial dependence, or process regimes.
- **Isolation Forest** uses randomized tree partitions to isolate unusual
  values. Its configured `contamination` affects the decision threshold; it is
  not an observed fault prevalence.
- **Precision** is the share of predicted anomaly points that are labeled
  synthetic anomalies; **recall** is the share of synthetic anomaly points
  detected; **F1** is their harmonic mean. **Accuracy** is the share of all
  points classified correctly. **Specificity** is the share of normal points
  correctly left unflagged; **false-positive rate** is the share of normal
  points incorrectly flagged. Undefined rate denominators are reported as zero.

These are pointwise metrics. They do not report event-level detection delay or
operational cost. No test-label parameter tuning is performed.

## Known limitations and honest interpretation

- The models use only one measurement (`OT`) and do not model time context,
  seasonality, or other sensor relationships.
- Synthetic drift, spikes, and variance changes are selected examples. They do
  not establish performance on naturally occurring confirmed industrial
  faults, fault prevalence, or deployment safety.
- The current evaluation showed poor recall for several configured cases and
  false positives from Isolation Forest. Interpret scores scenario by scenario;
  do not present a single aggregate as general fault-detection capability.
- The ETTh1 repository displays CC BY-ND 4.0, but its application to CSV assets
  remains uncertain. Do not redistribute the raw CSV or derivatives without
  rights-holder clarification.
- Dashboard rendering should be demonstrated interactively on the target
  machine; the automated smoke checks do not replace a human visual review.

## Key project files and artifacts

- Data pipeline: `src/thermoguard/data/{loader,validation,splitting,injection,profile}.py`
- Detectors: `src/thermoguard/detectors/{control_limits,isolation_forest}.py`
- Evaluation: `src/thermoguard/evaluation/{metrics,comparison}.py`
- Dashboard: `src/thermoguard/dashboard/{app,service}.py`
- Configuration: `configs/{dataset,experiments,detectors}.yaml`
- Tests: `tests/test_dataset_validation.py`, `tests/test_injection.py`,
  `tests/test_control_limits.py`, `tests/test_isolation_forest.py`,
  `tests/test_evaluation.py`, `tests/test_dashboard.py`
- Documentation: `README.md`, `docs/data_sources.md`, `docs/anomaly_injection.md`,
  `docs/control_limits.md`, `docs/isolation_forest.md`, `docs/evaluation.md`,
  `docs/dashboard.md`, `docs/reproducibility.md`
- Local generated artifacts (ignored by Git): `reports/results/ETTh1_profile.json`,
  detector prediction CSVs and summaries, and
  `reports/results/comparative_evaluation/{scenario_metrics.csv,summary.json,evaluation_report.md}`.

## Final verification before submission

- [ ] Run the full workflow in `README.md` using the local dataset.
- [ ] Run `.\.venv\Scripts\python.exe -m pytest` and confirm all tests pass.
- [ ] Run `.\.venv\Scripts\ruff.exe check src tests` and confirm no findings.
- [ ] Verify `data/raw/ETTh1.csv` SHA-256 matches the recorded dataset fingerprint.
- [ ] Confirm detector prediction artifacts omit `anomaly_label`.
- [ ] Confirm evaluation rows identify each scenario's validation/test segment.
- [ ] Visually review the dashboard at laptop width and check chart hover,
  selection behavior, metric split labels, and warning text.
- [ ] Check the upstream dataset license and obtain any needed rights
  clarification before distributing data or derivatives.

## Dashboard launch command

From `D:\Code\ThermoSense-AI` in Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m streamlit run src/thermoguard/dashboard/app.py
```
