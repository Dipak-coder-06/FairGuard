import os
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "adult_income.csv")
SENSITIVE_COLS = ["sex", "race", "age_bin"]
CAT_COLS = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "native-country", "sex", "race",
]
NUM_COLS = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]


def _bin_age(age: pd.Series) -> pd.Series:
    return pd.cut(age, bins=[0, 30, 50, 100], labels=["<30", "30-50", ">50"])


def load(n_sample: int = 20000, random_state: int = 42):
    if os.getenv("FAIRGUARD_SAMPLE"):
        n_sample = min(n_sample, 1000)
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
        raw = fetch_openml("adult", version=2, as_frame=True)
        df = raw.frame.copy()
        df["income"] = (raw.target.astype(str).str.strip().str.replace(".", "", regex=False)
                        .isin([">=50K", ">50K"])).astype(int)
        df.dropna(inplace=True)
        df.to_csv(DATA_PATH, index=False)

    df.dropna(inplace=True)
    df = df.sample(min(n_sample, len(df)), random_state=random_state).reset_index(drop=True)

    if "income" not in df.columns:
        if "class" in df.columns:
            df["income"] = df["class"].astype(str).str.strip().str.replace(".", "", regex=False).isin([">=50K", ">50K"]).astype(int)
        else:
            raise ValueError("No income/class column found")

    df["age_bin"] = _bin_age(df["age"].astype(float))
    df["sex"] = df["sex"].astype(str).str.strip()
    df["race"] = df["race"].astype(str).str.strip()

    y = df["income"].astype(int)
    sens = df[SENSITIVE_COLS].copy()

    # Sensitive columns remain available in sens for auditing only.
    drop_cols = ["income", "age_bin", "sex", "race"]
    if "class" in df.columns:
        drop_cols.append("class")
    X = df.drop(columns=drop_cols, errors="ignore")

    return X, y, sens


def profile(X: pd.DataFrame, y: pd.Series, sens: pd.DataFrame) -> dict:
    out = {}
    out["class_balance"] = y.value_counts(normalize=True).to_dict()
    out["n_rows"] = len(X)
    out["n_features"] = X.shape[1]
    for col in sens.columns:
        grp = pd.DataFrame({"y": y, "s": sens[col]})
        out[f"group_counts_{col}"] = grp["s"].value_counts().to_dict()
        out[f"group_base_rates_{col}"] = grp.groupby("s", observed=True)["y"].mean().to_dict()
    return out
