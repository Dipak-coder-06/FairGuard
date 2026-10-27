# FairGuard AI

FairGuard is a Streamlit workbench for assessing classification performance, group fairness, explainability, data drift, privacy proxies, and model governance using the UCI Adult Income dataset.

## Run locally

Python 3.10–3.12 is recommended. From the project directory:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
python -m pip install -r requirements.txt
streamlit run app.py
```

The first run trains the built-in models and stores them under `artifacts/`. The included `data/adult_income.csv` is used when present; otherwise the loader downloads the Adult dataset through OpenML.

## Verify the pipeline

Run the integration script from any working directory:

```bash
python full_test.py
```

It trains both built-in models and exercises the fairness, explainability, drift, privacy, governance, and pre-built Model C flows. The script may take a few minutes on the first run. `smoke_test.py` provides a shorter check of the main pipeline.

## Project layout

- `app.py` — Streamlit dashboard
- `src/` — data loading, training, fairness, explainability, drift, privacy, and model card modules
- `data/adult_income.csv` — local dataset
- `input/` — pre-trained models available for Model C comparison
- `artifacts/` — generated models, reports, and MLflow tracking data

Uploaded Model C files must be `.joblib` or `.pkl` files containing a `(model, preprocessor)` tuple compatible with the project feature columns.
