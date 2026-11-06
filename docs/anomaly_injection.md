# Synthetic anomaly generation

## Split policy

Injection always follows a chronological split. `configs/experiments.yaml` currently assigns 70% of rows to clean training, 15% to validation, and the remaining 15% to test. Boundaries use floor for the train and validation endpoints, with all remaining rows assigned to test. Original source-row indices are retained. The split helper rejects unordered, duplicate, missing, or unparseable timestamps; it does not sort or repair input.

The training segment stays clean and is never passed as the target of an injection. Validation and test scenarios are generated independently from their respective clean segment copies. Variance-change scale is derived from the clean training `OT` values.

## Injection and labels

All scenarios affect `OT` and write a separate `OT_injected` value column. The original `OT` column and other source fields remain unchanged in the returned copy. `anomaly_label` is a point-level binary ground-truth label: `1` for every row in the inclusive event interval and `0` elsewhere. Labels and event metadata must be excluded from detector input features in later chunks.

- **Gradual drift:** add a linear ramp over `duration` rows. The signed `magnitude` is the final offset; the first affected row receives one `magnitude / duration` step.
- **Sudden spike:** add the signed `magnitude` to each row in the configured short event interval.
- **Variance change:** add seeded, zero-mean Gaussian noise in the event interval. Noise is constructed orthogonal to the original event deviations so it increases sample variance without changing the event mean. Its standard deviation is `clean_training_OT_sample_std * sqrt(variance_multiplier - 1)`. Thus, the multiplier controls added variance relative to clean training variability; finite sample event variance need not equal exactly that multiplier times its own original variance.

Each event record includes anomaly type, segment, inclusive start/end source-row indices, timestamps, feature, seed, start offset, duration, and type-specific parameters. A scenario's `start_position` is a zero-based offset within its assigned segment. Event `start_row_index`/`end_row_index` are zero-based data row indices in the original CSV (header excluded).

These are **synthetic injected anomalies**, not naturally occurring confirmed faults. They are designed for controlled method development and do not establish real-world fault prevalence, physical plausibility, or detector performance on naturally occurring failures. Results depend on the chosen magnitudes, durations, seed, and placement. Scenarios are emitted separately from clean segment copies rather than compounded on top of one another.

## Reproduce configured scenarios

From the project root in PowerShell, with the project environment installed:

```powershell
.\.venv\Scripts\python.exe -m thermoguard.data.injection --input data/raw/ETTh1.csv --config configs/experiments.yaml --output data/processed/ETTh1_synthetic_scenarios.csv
.\.venv\Scripts\python.exe -m pytest
```

The command writes the generated, data-dependent CSV to `data/processed/` and a sibling `.events.json` file with event metadata and the source checksum. Both outputs are ignored by Git. The source CSV is only read and is never overwritten. To reproduce with a different scenario, edit the configuration's seed, segment, start offset, duration, magnitude, or variance multiplier. Keep the configuration with any published metrics; do not distribute the source dataset or generated derivatives.
