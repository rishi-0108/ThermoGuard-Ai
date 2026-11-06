# Dataset source and access

## ETT-small / ETTh1

- Repository: <https://github.com/zhouhaoyi/ETDataset>
- Dataset folder: `ETT-small/`
- Files currently listed in that folder: `ETTh1.csv`, `ETTh2.csv`, `ETTm1.csv`, and `ETTm2.csv`.
- Initial candidate file: `ETTh1.csv`
- Format: CSV; columns: `date`, `HUFL`, `HULL`, `MUFL`, `MULL`, `LUFL`, `LULL`, `OT`.
- `OT` is oil temperature; the other six measurement columns describe high, middle, and low useful/useless load.
- Timestamp resolution: hourly. The upstream README describes `h` variants as hourly; the published `ETTh1.csv` header and sample timestamps (`00:00`, `01:00`, `02:00`) confirm one-hour steps.
- Covered period: July 2016 through July 2018, as stated by upstream README.
- Requested academic citation: Zhou et al., “Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting,” AAAI 2021. See the upstream README for the full BibTeX entry.
- Data collection attribution in upstream README: the authors and Beijing Guowang Fuda Science & Technology Development Company.
- License shown in the repository's current `LICENSE`: Creative Commons Attribution-NoDerivatives 4.0 International (CC BY-ND 4.0), <https://creativecommons.org/licenses/by-nd/4.0/>.
- License handling: do not commit or redistribute original files or anomaly-injected derivatives. This project provides acquisition instructions and transformation code only. The repository-level license does not clearly enumerate which repository assets it covers; its apparent application to the CSV data should be treated as an uncertainty. Confirm intended use with the rights holder if that distinction becomes material, and seek permission if distribution requirements change.
- Repository checked: 2026-10-22. Recheck the upstream license, file path, and terms before each release.

## Acquiring the file

1. Visit the upstream repository and review its current `LICENSE` and README.
2. Download `ETT-small/ETTh1.csv` from the upstream `main` branch.
3. Save the file as `data/raw/ETTh1.csv` locally. The `data/raw/` directory is ignored by Git.
4. Do not upload the data file or derived dataset to this repository. Record the retrieval date and, when experiments are implemented, record a SHA-256 digest in local run metadata.

No dataset is included in this repository at setup stage.
