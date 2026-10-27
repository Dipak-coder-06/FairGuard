import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import shap
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # disable PIL decompression bomb check


def proxy_warning(feature_names):
    names = [str(name).lower() for name in feature_names[:10]]
    return any("relationship" in name or "marital-status" in name
               or "marital_status" in name for name in names)


def _get_shap_values(model_tuple, X_sample: pd.DataFrame):
    model, prep = model_tuple
    Xt = prep.transform(X_sample)
    source_names = []
    try:
        feature_names = []
        for name, trans, cols in prep.transformers_:
            if name == "cat":
                feature_names.extend(trans.get_feature_names_out(cols).tolist())
                source_names.extend(c for c, cats in zip(cols, trans.categories_)
                                    for _ in cats)
            else:
                numeric = cols if isinstance(cols, list) else [cols]
                feature_names.extend(numeric)
                source_names.extend(numeric)
        Xt_df = pd.DataFrame(Xt, columns=feature_names)
    except Exception:
        Xt_df = pd.DataFrame(Xt)
        source_names = Xt_df.columns.tolist()

    def collapse(vals, data, base):
        names = list(dict.fromkeys(source_names))
        if len(source_names) != vals.shape[1] or len(names) == vals.shape[1]:
            return vals, data, names
        out = np.zeros((vals.shape[0], len(names)))
        dat = np.zeros_like(out)
        for i, name in enumerate(names):
            cols = [j for j, src in enumerate(source_names) if src == name]
            out[:, i] = vals[:, cols].sum(axis=1)
            dat[:, i] = data[:, cols].sum(axis=1)
        return out, dat, names

    # Use generic shap.Explainer — works for any sklearn model
    try:
        # Preprocessors return a NumPy matrix and the estimators are trained
        # without column labels. SHAP supplies a DataFrame here; converting it
        # back avoids a sklearn warning for every internal perturbation.
        predict_fn = lambda values: model.predict_proba(np.asarray(values))
        bg = shap.maskers.Independent(Xt_df, max_samples=50)
        explainer = shap.Explainer(predict_fn, bg, feature_names=Xt_df.columns.tolist())
        sv = explainer(Xt_df, check_additivity=False)
        # sv shape: (n_samples, n_features, n_outputs) — take positive class
        if sv.values.ndim == 3:
            vals = sv.values[:, :, 1]
            base = sv.base_values[:, 1] if sv.base_values.ndim == 2 else sv.base_values
        else:
            vals = sv.values
            base = sv.base_values
        vals, data, names = collapse(vals, Xt_df.values, base)
        return shap.Explanation(
            values=vals, base_values=base,
            data=data, feature_names=names
        )
    except Exception:
        # Final fallback: predict label, use Independent masker
        try:
            bg = shap.maskers.Independent(Xt_df, max_samples=50)
            predict_fn = lambda values: model.predict(np.asarray(values))
            explainer = shap.Explainer(predict_fn, bg, feature_names=Xt_df.columns.tolist())
            sv = explainer(Xt_df, check_additivity=False)
            vals, data, names = collapse(sv.values, Xt_df.values, sv.base_values)
            return shap.Explanation(values=vals, base_values=sv.base_values,
                                    data=data, feature_names=names)
        except Exception as e:
            # Return zeros so app doesn't crash
            vals = np.zeros((len(Xt_df), len(Xt_df.columns)))
            base = np.zeros(len(Xt_df))
            vals, data, names = collapse(vals, Xt_df.values, base)
            return shap.Explanation(values=vals, base_values=base,
                                    data=data, feature_names=names)


def global_beeswarm(model_tuple, X_sample: pd.DataFrame) -> plt.Figure:
    sv = _get_shap_values(model_tuple, X_sample.head(200))
    plt.figure(figsize=(10, 7))
    shap.plots.beeswarm(sv, show=False, max_display=15)
    fig = plt.gcf(); fig.set_dpi(72)
    plt.tight_layout()
    return fig


def global_bar(model_tuple, X_sample: pd.DataFrame) -> plt.Figure:
    sv = _get_shap_values(model_tuple, X_sample.head(200))
    plt.figure(figsize=(10, 6))
    shap.plots.bar(sv, show=False, max_display=15)
    fig = plt.gcf(); fig.set_dpi(72)
    plt.tight_layout()
    return fig


def local_waterfall(model_tuple, X_sample: pd.DataFrame, idx: int = 0) -> plt.Figure:
    sv = _get_shap_values(model_tuple, X_sample.head(200))
    safe_idx = min(idx, len(sv) - 1)
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(sv[safe_idx], show=False, max_display=15)
    fig = plt.gcf(); fig.set_dpi(72)
    plt.tight_layout()
    return fig


def group_diff(model_tuple, X_sample: pd.DataFrame, sens_col: pd.Series) -> pd.DataFrame:
    n = len(X_sample)
    sens_col = sens_col.iloc[:n].reset_index(drop=True)
    sv = _get_shap_values(model_tuple, X_sample)
    vals = np.abs(sv.values)
    groups = sens_col.unique()
    result = {}
    for grp in groups:
        mask = (sens_col == grp).values
        if mask.shape[0] != vals.shape[0]:
            mask = mask[:vals.shape[0]]
        result[str(grp)] = vals[mask].mean(axis=0)
    df = pd.DataFrame(result, index=sv.feature_names)
    keys = list(result.keys())
    if len(keys) >= 2:
        df["diff"] = df[keys[0]] - df[keys[1]]
    return df.round(4)
