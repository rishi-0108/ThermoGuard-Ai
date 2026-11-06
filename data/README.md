# Local data and validation

## ETTh1

The expected local path is `data/raw/ETTh1.csv`. Obtain `ETT-small/ETTh1.csv` from the [official ETDataset repository](https://github.com/zhouhaoyi/ETDataset). Check the repository's current README and `LICENSE` before acquisition. The repository currently states Creative Commons Attribution-NoDerivatives 4.0 (CC BY-ND 4.0), but its repository-level license does not clearly enumerate which assets it covers; applicability to the CSV asset remains uncertain. This project makes no claim about redistribution rights. Keep the original CSV unchanged and do not commit or redistribute source or derived data.

## Validate and profile

From the project root in PowerShell, with the project virtual environment installed:

```powershell
.\.venv\Scripts\python.exe -m thermoguard.data.profile --input data/raw/ETTh1.csv --output reports/results/ETTh1_profile.json
.\.venv\Scripts\python.exe -m pytest
```

The profile command validates the required columns, timestamps, chronological order, duplicates, hourly spacing, missing values, numeric sensor types, and finite sensor values. It reports detected problems and returns a nonzero exit code for an invalid dataset; it never sorts, imputes, coerces, or overwrites the input CSV. The JSON profile includes the source, access date, validation issues, measurements, and SHA-256 checksum.

To reproduce the configured synthetic scenarios after validation, see [the injection methodology and command](../docs/anomaly_injection.md). The command writes only under `data/processed/` and does not modify the raw CSV.

The statistical control-limit baseline and its train-only fit, held-out prediction, configuration, and command are documented in [control_limits.md](../docs/control_limits.md).

The `data/raw/`, `data/interim/`, `data/processed/`, and `reports/results/` paths are excluded from Git. The latter may contain data-dependent statistics and checksums.
