# Wine quality analysis

A personal learning project using Python to explore red-wine measurements and quality ratings. It includes a statistical-analysis notebook and a Dash dashboard for exploring the data.

## What is implemented

- Distribution plots, outlier checks and correlation analysis.
- Train/test preprocessing and statistical modelling helpers in `wine_quality_utils.py`.
- A dashboard with feature selectors, histograms, scatter plots and a sample of the data.

The dashboard displays observations. It does not serve a prediction model. Associations in this dataset are not causal evidence or advice to consumers and winemakers.

## Run locally

Use Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/DaudJG/wine_quality_analysis.git
cd wine_quality_analysis
uv sync --locked --python 3.12
uv run python app.py
```

Open `http://127.0.0.1:8050`. Select two features to change the histogram and scatter plot. The second tab shows sample records and an additional scatter plot.

`wine_quality_analysis.ipynb` contains the analysis. Open it with a notebook editor using this repository's `.venv` Python interpreter. `environment.yml` is the earlier Conda setup; `pyproject.toml` and `uv.lock` define the maintained environment.

## Data and limitations

The supplied `winequality-red.csv` contains physicochemical measurements and quality ratings from the [UCI Wine Quality dataset](https://archive.ics.uci.edu/dataset/186/wine+quality). Attribute the source dataset when reusing it. This project is for learning and has not been validated for production decisions.

Notebook results depend on the split, model and evaluation settings. The dashboard does not imply deployed model performance or business impact.

## Verification

On 8 September 2026, the maintained environment was checked with all 103 notebook code cells run sequentially, the dashboard's three HTTP endpoints, two plot selections and the modelling helper's held-out prediction path. Two diagnostic-plot method typos were corrected. These are execution checks, not independent validation of predictive performance or scientific conclusions. Saved notebook outputs have not been replaced.
