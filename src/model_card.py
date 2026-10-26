import os
from datetime import datetime

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")


def generate(metrics_A: dict, metrics_B: dict,
             fairness_A: dict, fairness_B: dict,
             privacy: dict, drift: dict) -> str:

    def badge(m, v):
        from src.fairness_engine import status
        return status(m, v)

    now = datetime.now().strftime("%Y-%m-%d")

    dpd_A = fairness_A.get("demographic_parity_diff", "N/A")
    dpd_B = fairness_B.get("demographic_parity_diff", "N/A")
    eod_A = fairness_A.get("equalized_odds_diff", "N/A")
    eod_B = fairness_B.get("equalized_odds_diff", "N/A")
    di_A = fairness_A.get("disparate_impact", "N/A")
    di_B = fairness_B.get("disparate_impact", "N/A")
    erd_A = fairness_A.get("error_rate_disparity", "N/A")
    erd_B = fairness_B.get("error_rate_disparity", "N/A")

    md = f"""# FairGuard AI — Model Card
*Generated: {now} | Standard: Mitchell et al. (2019) Model Cards*

---

## 1. Model Details

| | Model A (Unmitigated) | Model B (Fairness-Aware) |
|---|---|---|
| **Type** | HistGradientBoostingClassifier | ExponentiatedGradient + LogisticRegression |
| **Constraint** | None | Demographic Parity |
| **Framework** | scikit-learn | Fairlearn |
| **MLflow Run** | Model_A_Unmitigated | Model_B_FairnessMitigated |

---

## 2. Intended Use

**Primary Use:** Income bracket classification for demonstrating responsible AI evaluation techniques.

**Out-of-scope Use:** Real hiring, lending, or credit decisions without additional human review, bias auditing, and regulatory compliance checks. Model outputs MUST NOT be used to draw conclusions about protected groups.

---

## 3. Factors

**Sensitive Attributes Evaluated:** `sex`, `race`, `age_bin`

**Intersectional Groups:** `sex × race` combinations evaluated separately.

---

## 4. Performance Metrics

| Metric | Model A | Model B |
|--------|---------|---------|
| Accuracy | {metrics_A.get('accuracy','N/A')} | {metrics_B.get('accuracy','N/A')} |
| AUC-ROC | {metrics_A.get('auc','N/A')} | {metrics_B.get('auc','N/A')} |
| F1 Score | {metrics_A.get('f1','N/A')} | {metrics_B.get('f1','N/A')} |

---

## 5. Evaluation Data

- **Dataset:** UCI Adult Census Income (fetch_openml, version 2)
- **Split:** 80% train / 20% test (stratified by income)
- **Sample:** Up to 20,000 rows after NaN removal
- **Features:** 14 original features; age binned to `<30 / 30-50 / >50` for group analysis

---

## 6. Fairness & Privacy Findings

### Fairness Summary

| Metric | Model A | Status A | Model B | Status B |
|--------|---------|----------|---------|----------|
| Demographic Parity Diff | {dpd_A} | {badge('demographic_parity_diff', dpd_A) if isinstance(dpd_A, float) else 'N/A'} | {dpd_B} | {badge('demographic_parity_diff', dpd_B) if isinstance(dpd_B, float) else 'N/A'} |
| Equalized Odds Diff | {eod_A} | {badge('equalized_odds_diff', eod_A) if isinstance(eod_A, float) else 'N/A'} | {eod_B} | {badge('equalized_odds_diff', eod_B) if isinstance(eod_B, float) else 'N/A'} |
| Disparate Impact | {di_A} | {badge('disparate_impact', di_A) if isinstance(di_A, float) else 'N/A'} | {di_B} | {badge('disparate_impact', di_B) if isinstance(di_B, float) else 'N/A'} |
| Error Rate Disparity | {erd_A} | {badge('error_rate_disparity', erd_A) if isinstance(erd_A, float) else 'N/A'} | {erd_B} | {badge('error_rate_disparity', erd_B) if isinstance(erd_B, float) else 'N/A'} |

### Privacy Risk (Model A)

| Indicator | Value |
|-----------|-------|
| Min k-Anonymity (k) | {privacy.get('k_anon', {}).get('min_k', 'N/A')} |
| Rows with k < 5 | {privacy.get('k_anon', {}).get('pct_rows_k_lt_5', 'N/A')}% |
| MIA Train/Test AUC Gap | {privacy.get('mia', {}).get('auc_gap', 'N/A')} |
| Membership Inference Risk | **{privacy.get('mia', {}).get('risk', 'N/A')}** |

### Data Drift

| Indicator | Value |
|-----------|-------|
| Share of Drifted Features | {drift.get('share_drifted', 'N/A')} |

---

## 7. Fairness–Accuracy Trade-off Analysis

Model B applies the **Exponential Gradient** reduction with a **Demographic Parity** constraint, which directly optimises for equal selection rates across sensitive groups. This typically results in a measurable drop in raw accuracy and AUC compared to the unconstrained Model A, because the model is constrained away from patterns that, while predictive, correlate with protected attributes.

**Why Demographic Parity and Equalized Odds can conflict:** Demographic Parity requires equal *selection rates* (P(ŷ=1 | group)), while Equalized Odds requires equal *true positive rates and false positive rates* across groups. When the *base rates* (P(y=1 | group)) differ between groups — as they do in the UCI Adult dataset across sex and race — it is **mathematically impossible** to satisfy both constraints simultaneously (Chouldechova, 2017; Kleinberg et al., 2016). Enforcing one necessarily violates the other. FairGuard AI reports both metrics to expose this tension explicitly and leave the trade-off decision to human stakeholders.

---

## 8. Caveats

- UCI Adult data reflects 1994 US Census income patterns; distributions may not generalise.
- Fairness metrics are sensitive to how groups are defined and which sensitive attribute is chosen as the primary axis.
- SHAP values on Model B (mitigated) may differ significantly from Model A because the underlying model family changed.
- k-Anonymity is a weak privacy guarantee; re-identification risk may still exist.

---

## 9. Mitigations

- **Pre-processing:** Fairlearn sample reweighing (if enabled).
- **In-processing:** Exponential Gradient with Demographic Parity (Model B).
- **Post-processing:** Threshold calibration per group (recommended next step).
- **Monitoring:** Evidently AI drift reports on production data.
- **Human Oversight:** Sign-off checklist enforced before deployment.

---

## 10. Sign-Off Checklist

- [ ] Fairness audit reviewed by responsible AI lead
- [ ] All metrics above threshold reviewed and documented
- [ ] Drift report reviewed; no significant distributional shift confirmed
- [ ] Privacy risk assessed; k-anonymity deemed acceptable
- [ ] Model Card approved by data governance team
- [ ] Legal / regulatory review completed
- [ ] Deployment approved

---
*FairGuard AI · Responsible AI Assessment Workbench*
"""

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    md_path = os.path.join(ARTIFACT_DIR, "model_card.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    try:
        import markdown as md_lib
        html_body = md_lib.markdown(md, extensions=["tables"])
        html = f"<html><head><meta charset='utf-8'><style>body{{font-family:sans-serif;max-width:900px;margin:auto;padding:2em}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccc;padding:.5em}}th{{background:#f0f0f0}}</style></head><body>{html_body}</body></html>"
        html_path = os.path.join(ARTIFACT_DIR, "model_card.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
    except Exception:
        pass

    return md
