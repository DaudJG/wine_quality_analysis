# Wine quality analysis

A personal learning project exploring red-wine measurements and quality ratings with Python. It contains a statistical-analysis notebook and a Dash dashboard for exploring observations.

## What to inspect

- `wine_quality_analysis.ipynb`: distributions, correlations, hypothesis tests, regression diagnostics and binary classification.
- `wine_quality_utils.py`: plotting and modelling helpers.
- `app.py`: feature selectors, histograms, scatter plots and sample records. The dashboard does not serve a prediction model.

## Setup on Windows or Linux

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Git. These commands work in PowerShell or a Linux shell:

```text
git clone https://github.com/DaudJG/wine_quality_analysis.git
cd wine_quality_analysis
uv sync --locked
uv run --locked python app.py
```

Open http://127.0.0.1:8050. Select two measurements on the first tab; the second tab shows sample data and another scatter plot.

The setup uses Python 3.12.12 and uv 0.11.29. `pyproject.toml` declares dependencies and `uv.lock` records their resolved versions. `uv sync --locked` recreates the ignored `.venv` and refuses an inconsistent lockfile. Conda is not required. Installation needs internet access; the bundled dataset allows the analysis to run offline afterward.

Open the notebook in a notebook editor using `.venv` as its Python interpreter, or execute every cell with:

```text
uv run --locked python verify.py --notebook
```

This starts a fresh kernel with the locked interpreter and writes an executed copy to ignored `verification-output/`. Add `--write-notebook` only when intentionally refreshing the checked-in outputs.

## Data and interpretation

The [UCI Wine Quality dataset](https://archive.ics.uci.edu/dataset/186/wine+quality) is attributed to Paulo Cortez and colleagues. Keep its source attribution when reusing it. The included red-wine CSV has 1,599 records and 12 columns; its SHA-256 is `d6a0d9bd24806944818795f22500c46cb6424cbff517aacda36595d3ed9b2daa`.

The notebook removes 240 exact duplicate rows as an analytical choice, leaving 1,359. The dashboard displays all source records. Classification uses 815 training, 272 validation and 272 test rows; scaling is fitted on training rows and the decision threshold is selected on validation rows.

Earlier exploration informed feature choices. Therefore, these splits are not independent external validation. The quality rating is ordinal, associations are not causal, and the notebook does not support production performance or winemaking recommendations.

## Verification

`verify.py` checks source dimensions, train-only preprocessing, finite model predictions, positive WLS weights, all 121 valid dashboard feature pairs, cleared/invalid selections and HTTP callback responses. `--notebook` runs all 103 code cells and checks numerical results, split separation and rendered charts. The verification workflow runs on Windows and Ubuntu. Small floating-point differences between platforms are expected; macOS is unverified.
