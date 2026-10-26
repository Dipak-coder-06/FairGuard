# FairGuard AI — Model Card
*Generated: 2026-10-26 | Standard: Mitchell et al. (2019) Model Cards*

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
| Accuracy | 0.861 | 0.8333 |
| AUC-ROC | 0.9223 | 0.7317 |
| F1 Score | 0.6982 | 0.607 |

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
| Demographic Parity Diff | 0.1851 | Warning | 0.0153 | Pass |
| Equalized Odds Diff | 0.11 | Warning | 0.3986 | Fail |
| Disparate Impact | 0.3299 | Fail | 0.9173 | Pass |
| Error Rate Disparity | 0.1132 | Warning | 0.1066 | Warning |

### Privacy Risk (Model A)

| Indicator | Value |
|-----------|-------|
| Min k-Anonymity (k) | 1 |
| Rows with k < 5 | 59.15% |
| MIA Train/Test AUC Gap | 0.0231 |
| Membership Inference Risk | **Low** |

### Data Drift

| Indicator | Value |
|-----------|-------|
| Share of Drifted Features | 0.14285714285714285 |

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
