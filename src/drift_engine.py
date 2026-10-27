import os
import json
import numpy as np
import pandas as pd
try:  # Evidently 0.4.x
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
except ImportError:  # Evidently 0.7.x moved these public APIs
    from evidently import Report
    from evidently.presets import DataDriftPreset

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")


def drift(ref_df: pd.DataFrame, cur_df: pd.DataFrame) -> dict:
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    ref = ref_df.copy()
    cur = cur_df.copy()
    for c in ref.columns:
        if ref[c].dtype == "object" or str(ref[c].dtype) == "category":
            ref[c] = ref[c].astype(str)
            cur[c] = cur[c].astype(str)

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=ref, current_data=cur)
    html_path = os.path.join(ARTIFACT_DIR, "drift.html")
    if hasattr(snapshot, "save_html"):
        snapshot.save_html(html_path)
        result_dict = snapshot.dict()
    else:  # Evidently 0.4.x returns the report object from run().
        report.save_html(html_path)
        result_dict = report.as_dict()
    metrics_data = result_dict.get("metrics", [])

    share_drifted = 0.0
    per_feature = {}

    for m in metrics_data:
        mname = m.get("metric", "")
        mresult = m.get("result", {})
        # Evidently 0.7 stores results as named v2 metrics.
        metric_name = m.get("metric_name", "")
        value = m.get("value", {})
        if "DriftedColumnsCount" in metric_name and isinstance(value, dict):
            share_drifted = float(value.get("share", 0.0))
        elif metric_name.startswith("ValueDrift(column="):
            feat = metric_name.split("column=", 1)[1].split(",", 1)[0].rstrip(")")
            score = float(value) if np.isscalar(value) else 0.0
            per_feature[feat] = {"stattest": "value drift", "score": round(score, 4),
                                 "drifted": bool(score < 0.05)}
        if "DatasetDriftMetric" in mname or "share_of_drifted_columns" in str(mresult):
            share_drifted = mresult.get("share_of_drifted_columns", 0.0)
        if "DataDriftTable" in mname or "drift_by_columns" in str(mresult):
            drift_cols = mresult.get("drift_by_columns", {})
            for feat, fdata in drift_cols.items():
                per_feature[feat] = {
                    "stattest": fdata.get("stattest_name", ""),
                    "score": round(float(fdata.get("drift_score", 0.0)), 4),
                    "drifted": fdata.get("drift_detected", False),
                }

    if not per_feature:
        for m in metrics_data:
            mresult = m.get("result", {})
            if isinstance(mresult, dict):
                for k, v in mresult.items():
                    if isinstance(v, dict) and "drift_detected" in v:
                        per_feature[k] = {
                            "stattest": v.get("stattest_name", ""),
                            "score": round(float(v.get("drift_score", 0.0)), 4),
                            "drifted": v.get("drift_detected", False),
                        }

    return {
        "share_drifted": share_drifted,
        "html_path": html_path,
        "per_feature": per_feature,
    }


def make_shifted(df: pd.DataFrame) -> pd.DataFrame:
    shifted = df.copy()
    if "age" in shifted.columns:
        shifted["age"] = (shifted["age"].astype(float) + 5).clip(upper=90)
    if "hours-per-week" in shifted.columns:
        shifted["hours-per-week"] = (shifted["hours-per-week"].astype(float) + 3).clip(upper=99)
    return shifted
