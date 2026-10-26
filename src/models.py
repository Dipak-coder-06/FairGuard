import os
import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from fairlearn.reductions import ExponentiatedGradient, DemographicParity

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
MLRUNS_DIR = os.path.join(ARTIFACT_DIR, "mlruns")

CAT_COLS = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "native-country", "sex", "race",
]
NUM_COLS = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]


def _build_preprocessor(X: pd.DataFrame):
    cat_present = [c for c in CAT_COLS if c in X.columns]
    num_present = [c for c in NUM_COLS if c in X.columns]
    transformers = []
    if cat_present:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_present))
    if num_present:
        transformers.append(("num", StandardScaler(), num_present))
    return ColumnTransformer(transformers, remainder="drop")


def _get_feature_names(preprocessor, X):
    names = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "cat":
            names.extend(trans.get_feature_names_out(cols).tolist())
        else:
            names.extend(cols)
    return names


def train_all(X_tr: pd.DataFrame, y_tr: pd.Series, s_tr: pd.DataFrame) -> dict:
    os.makedirs(MLRUNS_DIR, exist_ok=True)
    mlflow.set_tracking_uri(f"file:{MLRUNS_DIR}")
    mlflow.set_experiment("fairguard")

    prep_A = _build_preprocessor(X_tr)
    X_tr_t = prep_A.fit_transform(X_tr)

    prep_B = _build_preprocessor(X_tr)
    X_tr_tb = prep_B.fit_transform(X_tr)

    sensitive_col = s_tr["sex"].astype(str)

    model_A = HistGradientBoostingClassifier(max_iter=100, random_state=42)
    model_A.fit(X_tr_t, y_tr)

    base_lr = LogisticRegression(max_iter=1000, solver="lbfgs", C=0.5)
    eg = ExponentiatedGradient(base_lr, constraints=DemographicParity(), max_iter=30, nu=0.05)
    eg.fit(X_tr_tb, y_tr, sensitive_features=sensitive_col)

    with mlflow.start_run(run_name="Model_A_Unmitigated"):
        mlflow.log_param("model_type", "HistGradientBoosting")
        mlflow.log_param("mitigated", False)

    with mlflow.start_run(run_name="Model_B_FairnessMitigated"):
        mlflow.log_param("model_type", "ExponentiatedGradient_LR")
        mlflow.log_param("mitigated", True)
        mlflow.log_param("constraint", "DemographicParity")

    joblib.dump((model_A, prep_A), os.path.join(ARTIFACT_DIR, "model_A.joblib"))
    joblib.dump((eg, prep_B), os.path.join(ARTIFACT_DIR, "model_B.joblib"))

    return {"A": (model_A, prep_A), "B": (eg, prep_B)}


def predict(model_tuple, X: pd.DataFrame) -> tuple:
    model, prep = model_tuple
    Xt = prep.transform(X)
    preds = model.predict(Xt)
    try:
        proba = model.predict_proba(Xt)[:, 1]
    except AttributeError:
        try:
            pmf = model._pmf_predict(Xt)
            if hasattr(pmf, "iloc"):
                proba = pmf.iloc[:, 1].values
            else:
                proba = np.array(pmf)[:, 1]
        except Exception:
            proba = np.array(preds, dtype=float)
    return np.array(preds), np.array(proba)


def metrics(y_true, preds, proba) -> dict:
    try:
        auc = roc_auc_score(y_true, proba)
    except Exception:
        auc = float("nan")
    return {
        "accuracy": round(accuracy_score(y_true, preds), 4),
        "auc": round(float(auc), 4),
        "f1": round(f1_score(y_true, preds, zero_division=0), 4),
    }
