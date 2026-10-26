# FairGuard AI — Input Models

Pre-trained models for comparison testing. Each `.joblib` file contains a `(model, preprocessor)` tuple trained on **UCI Adult Income** data (80/20 split, 20k rows).

| File | Algorithm | Notes |
|------|-----------|-------|
| `random_forest.joblib` | RandomForestClassifier (100 trees, depth 8) | Strong baseline, may show high bias |
| `logistic_regression.joblib` | LogisticRegression (L2, C=1.0) | Linear, interpretable, SHAP-friendly |
| `decision_tree.joblib` | DecisionTreeClassifier (depth 6) | Fully transparent, low complexity |
| `gradient_boosting.joblib` | GradientBoostingClassifier (80 trees, depth 4) | High accuracy unconstrained |
| `linear_svc_calibrated.joblib` | LinearSVC + Platt scaling | Fast linear classifier with probability output |

## How to Use

1. Open **http://localhost:8501**
2. In the **sidebar**, select any model from the **"Input Models"** dropdown
3. It loads as **Model C** and appears in:
   - 📈 Model Comparison tab
   - ⚖️ Fairness Audit tab  
   - 📋 Governance tab (full fairness scorecard)

## Re-generate Models

```bash
venv\Scripts\python.exe generate_input_models.py
```

## Format for Custom Uploads

Your own model must be saved as a `(model, preprocessor)` tuple:

```python
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

prep = ColumnTransformer(...)  # same feature schema as Adult dataset
model = YourClassifier()
model.fit(prep.fit_transform(X_train), y_train)

joblib.dump((model, prep), "my_model.joblib")
```
Then upload `my_model.joblib` via the **"Upload Your Own Model"** file uploader in the sidebar.
