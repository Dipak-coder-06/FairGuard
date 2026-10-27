"""
Deep integration test â€” covers every tab's code path including Model C upload flow.
Run: venv\Scripts\python.exe debug_all.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi": 72, "savefig.dpi": 72})

import joblib, traceback
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src import data_loader, models, fairness_engine, xai_engine, drift_engine, privacy_engine, model_card

PASS = []; FAIL = []

def check(name, fn):
    try:
        result = fn()
        PASS.append(name)
        print(f"  [PASS] {name}" + (f"  -> {result}" if result is not None else ""))
    except Exception as e:
        FAIL.append(name)
        print(f"  [FAIL] {name}")
        traceback.print_exc()

print("=== SETUP ===")
X, y, sens = data_loader.load()
X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(X, y, sens, test_size=0.2, random_state=42, stratify=y)
trained = models.train_all(X_tr, y_tr, s_tr)
results = {}
for name, m in trained.items():
    p, pr = models.predict(m, X_te)
    results[name] = {"model": m, "preds": p, "proba": pr, "metrics": models.metrics(y_te, p, pr)}
print(f"  Models trained: {list(results.keys())}")

# Load all input models as Model C candidates
input_dir = os.path.join(ROOT, "input")
input_models = {f: joblib.load(os.path.join(input_dir, f))
                for f in sorted(os.listdir(input_dir)) if f.endswith(".joblib")}

SAMPLE_N = 200
sample   = X_te.head(SAMPLE_N).reset_index(drop=True)
sens_s   = s_te["sex"].reset_index(drop=True).iloc[:SAMPLE_N]

print("\n=== TAB 1 â€” DATA EXPLORER ===")
profile = data_loader.profile(X, y, sens)
check("profile keys",    lambda: list(profile.keys()))
check("class_balance",   lambda: profile["class_balance"])
check("group counts sex",lambda: profile["group_counts_sex"])
check("group rates race",lambda: profile["group_base_rates_race"])
check("bar chart sex",   lambda: (
    plt.subplots(figsize=(8,3))[0].clear() or "ok"
))
plt.close("all")

print("\n=== TAB 2 â€” MODEL COMPARISON ===")
for attr in ["sex","race","age_bin"]:
    for nm in ["A","B"]:
        check(f"audit model={nm} attr={attr}", lambda nm=nm, attr=attr: (
            fairness_engine.audit(y_te, results[nm]["preds"], s_te[attr])["demographic_parity_diff"]
        ))

check("status DPD pass",  lambda: fairness_engine.status("demographic_parity_diff", 0.05))
check("status DPD warn",  lambda: fairness_engine.status("demographic_parity_diff", 0.15))
check("status DPD fail",  lambda: fairness_engine.status("demographic_parity_diff", 0.25))
check("status DI pass",   lambda: fairness_engine.status("disparate_impact", 0.85))
check("status DI fail",   lambda: fairness_engine.status("disparate_impact", 0.60))

print("\n=== TAB 3 â€” FAIRNESS AUDIT ===")
for attr in ["sex","race","age_bin"]:
    check(f"by_group {attr}",        lambda a=attr: fairness_engine.audit(y_te, results["A"]["preds"], s_te[a])["by_group"].shape)
    check(f"confusion_by_group {attr}", lambda a=attr: (
        fairness_engine.confusion_by_group(y_te, results["A"]["preds"], s_te[a]),
        plt.close("all"), "ok"
    )[-1])
check("intersectional sexÃ—race",     lambda: fairness_engine.intersectional(y_te, results["A"]["preds"], s_te[["sex","race"]]).shape)

print("\n=== TAB 4 â€” EXPLAINABILITY ===")
check("beeswarm model A", lambda: (
    xai_engine.global_beeswarm(results["A"]["model"], sample),
    plt.close("all"), f"{SAMPLE_N} rows"
)[-1])
check("bar model A",      lambda: (
    xai_engine.global_bar(results["A"]["model"], sample),
    plt.close("all"), "ok"
)[-1])
check("waterfall idx=0",  lambda: (
    xai_engine.local_waterfall(results["A"]["model"], sample, 0),
    plt.close("all"), "ok"
)[-1])
check("waterfall idx=99", lambda: (
    xai_engine.local_waterfall(results["A"]["model"], sample, 99),
    plt.close("all"), "ok"
)[-1])
check("group_diff",       lambda: xai_engine.group_diff(results["A"]["model"], sample, sens_s).shape)

# Model B (ExponentiatedGradient) â€” use base predictor
model_B_obj = results["B"]["model"][0]
has_predictors = hasattr(model_B_obj, "predictors_")
preds_list = list(model_B_obj.predictors_) if has_predictors else []
if has_predictors and len(preds_list) > 0:
    b_tuple = (preds_list[0], results["B"]["model"][1])
    check("beeswarm model B (base LR)", lambda: (
        xai_engine.global_beeswarm(b_tuple, sample),
        plt.close("all"), "ok"
    )[-1])

print("\n=== TAB 5 â€” DRIFT & PRIVACY ===")
check("drift report", lambda: drift_engine.drift(X_tr.copy(), drift_engine.make_shifted(X_te.copy()))["share_drifted"])
check("drift html exists", lambda: os.path.exists("artifacts/drift.html"))

combined = pd.concat([X_te.reset_index(drop=True), s_te.reset_index(drop=True)], axis=1)
check("k_anonymity",  lambda: privacy_engine.k_anonymity(combined))
check("mia model A",  lambda: privacy_engine.mia_proxy(results["A"]["model"], X_tr, y_tr, X_te, y_te)["risk"])
check("mia model B",  lambda: privacy_engine.mia_proxy(results["B"]["model"], X_tr, y_tr, X_te, y_te)["risk"])

print("\n=== TAB 6 â€” GOVERNANCE ===")
dr  = drift_engine.drift(X_tr.copy(), drift_engine.make_shifted(X_te.copy()))
ka  = privacy_engine.k_anonymity(combined)
mia = privacy_engine.mia_proxy(results["A"]["model"], X_tr, y_tr, X_te, y_te)
fa_A = fairness_engine.audit(y_te, results["A"]["preds"], s_te["sex"])
fa_B = fairness_engine.audit(y_te, results["B"]["preds"], s_te["sex"])
check("model_card generate", lambda: (
    model_card.generate(results["A"]["metrics"], results["B"]["metrics"],
                        fa_A, fa_B, {"k_anon": ka, "mia": mia}, dr),
    os.path.exists("artifacts/model_card.md")
)[-1])
check("model_card.md size", lambda: f"{os.path.getsize('artifacts/model_card.md')} bytes")

print("\n=== MODEL C â€” ALL INPUT MODELS ===")
for fname, obj in input_models.items():
    m_c, prep_c = obj
    check(f"[C] predict {fname}", lambda m=m_c, p=prep_c: models.predict((m,p), X_te)[0].shape)
    check(f"[C] metrics {fname}", lambda m=m_c, p=prep_c: models.metrics(y_te, *models.predict((m,p), X_te)))
    check(f"[C] fairness sex {fname}", lambda m=m_c, p=prep_c: (
        fairness_engine.audit(y_te, models.predict((m,p), X_te)[0], s_te["sex"])["demographic_parity_diff"]
    ))
    check(f"[C] fairness race {fname}", lambda m=m_c, p=prep_c: (
        fairness_engine.audit(y_te, models.predict((m,p), X_te)[0], s_te["race"])["demographic_parity_diff"]
    ))
    check(f"[C] fairness age_bin {fname}", lambda m=m_c, p=prep_c: (
        fairness_engine.audit(y_te, models.predict((m,p), X_te)[0], s_te["age_bin"])["demographic_parity_diff"]
    ))
    check(f"[C] intersectional {fname}", lambda m=m_c, p=prep_c: (
        fairness_engine.intersectional(y_te, models.predict((m,p), X_te)[0], s_te[["sex","race"]]).shape
    ))
    check(f"[C] confusion mats {fname}", lambda m=m_c, p=prep_c: (
        fairness_engine.confusion_by_group(y_te, models.predict((m,p), X_te)[0], s_te["sex"]),
        plt.close("all"), "ok"
    )[-1])
    check(f"[C] beeswarm {fname}", lambda m=m_c, p=prep_c: (
        xai_engine.global_beeswarm((m,p), sample),
        plt.close("all"), "ok"
    )[-1])
    check(f"[C] waterfall {fname}", lambda m=m_c, p=prep_c: (
        xai_engine.local_waterfall((m,p), sample, 0),
        plt.close("all"), "ok"
    )[-1])
    check(f"[C] group_diff {fname}", lambda m=m_c, p=prep_c: (
        xai_engine.group_diff((m,p), sample, sens_s).shape
    ))
    check(f"[C] mia {fname}", lambda m=m_c, p=prep_c: (
        privacy_engine.mia_proxy((m,p), X_tr, y_tr, X_te, y_te)["risk"]
    ))
    plt.close("all")

print(f"\n{'='*50}")
print(f"PASSED: {len(PASS)} / {len(PASS)+len(FAIL)}")
if FAIL:
    print(f"FAILED: {len(FAIL)}")
    for f in FAIL: print(f"  - {f}")
else:
    print("ALL TESTS PASSED")

