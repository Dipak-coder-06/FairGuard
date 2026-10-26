import warnings
warnings.filterwarnings("ignore")
import os, sys, joblib
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
sys.path.insert(0, r"c:\Users\dhira\Documents\Dipak_Project")
os.chdir(r"c:\Users\dhira\Documents\Dipak_Project")

from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src import data_loader, models, fairness_engine, xai_engine, drift_engine, privacy_engine, model_card

PASS = []
FAIL = []

def chk(name, fn):
    try:
        r = fn()
        PASS.append(name)
        print(f"  [PASS] {name}" + (f"  => {r}" if r is not None else ""))
    except Exception as e:
        FAIL.append(name)
        import traceback
        print(f"  [FAIL] {name}: {e}")
        traceback.print_exc()

print("=== SETUP ===")
X, y, sens = data_loader.load()
Xtr, Xte, ytr, yte, str_, ste = train_test_split(X, y, sens, test_size=0.2, random_state=42, stratify=y)
trained = models.train_all(Xtr, ytr, str_)
res = {}
for n, m in trained.items():
    p, pr = models.predict(m, Xte)
    res[n] = {"model": m, "preds": p, "proba": pr, "metrics": models.metrics(yte, p, pr)}
print(f"  Models trained: {list(res.keys())}")

SAMPLE_N = 200
sample  = Xte.head(SAMPLE_N).reset_index(drop=True)
sens_s  = ste["sex"].reset_index(drop=True).iloc[:SAMPLE_N]

# ── TAB 1 ────────────────────────────────────────────────────────────────────
print("\n=== TAB 1 - DATA EXPLORER ===")
profile = data_loader.profile(X, y, sens)
chk("profile keys",       lambda: sorted(profile.keys()))
chk("class_balance",      lambda: profile["class_balance"])
chk("group_counts_sex",   lambda: profile["group_counts_sex"])
chk("group_rates_race",   lambda: profile["group_base_rates_race"])
chk("group_rates_age_bin",lambda: profile["group_base_rates_age_bin"])

# ── TAB 2 ────────────────────────────────────────────────────────────────────
print("\n=== TAB 2 - MODEL COMPARISON ===")
for attr in ["sex", "race", "age_bin"]:
    for nm in ["A", "B"]:
        chk(f"audit model={nm} attr={attr}",
            lambda nm=nm, attr=attr: fairness_engine.audit(yte, res[nm]["preds"], ste[attr])["demographic_parity_diff"])
chk("status DPD pass",  lambda: fairness_engine.status("demographic_parity_diff", 0.05))
chk("status DPD warn",  lambda: fairness_engine.status("demographic_parity_diff", 0.15))
chk("status DPD fail",  lambda: fairness_engine.status("demographic_parity_diff", 0.25))
chk("status DI pass",   lambda: fairness_engine.status("disparate_impact", 0.85))
chk("status DI fail",   lambda: fairness_engine.status("disparate_impact", 0.60))
chk("metrics model A",  lambda: res["A"]["metrics"])
chk("metrics model B",  lambda: res["B"]["metrics"])

# ── TAB 3 ────────────────────────────────────────────────────────────────────
print("\n=== TAB 3 - FAIRNESS AUDIT ===")
for attr in ["sex", "race", "age_bin"]:
    chk(f"by_group {attr}",
        lambda a=attr: fairness_engine.audit(yte, res["A"]["preds"], ste[a])["by_group"].shape)
    chk(f"confusion_by_group {attr}",
        lambda a=attr: (fairness_engine.confusion_by_group(yte, res["A"]["preds"], ste[a]), plt.close("all"), "ok")[-1])
chk("intersectional sex x race",
    lambda: fairness_engine.intersectional(yte, res["A"]["preds"], ste[["sex", "race"]]).shape)

# ── TAB 4 ────────────────────────────────────────────────────────────────────
print("\n=== TAB 4 - EXPLAINABILITY ===")
chk("beeswarm model A",
    lambda: (xai_engine.global_beeswarm(res["A"]["model"], sample), plt.close("all"), "ok")[-1])
chk("bar model A",
    lambda: (xai_engine.global_bar(res["A"]["model"], sample), plt.close("all"), "ok")[-1])
chk("waterfall idx=0",
    lambda: (xai_engine.local_waterfall(res["A"]["model"], sample, 0), plt.close("all"), "ok")[-1])
chk("waterfall idx=99",
    lambda: (xai_engine.local_waterfall(res["A"]["model"], sample, 99), plt.close("all"), "ok")[-1])
chk("group_diff A",
    lambda: xai_engine.group_diff(res["A"]["model"], sample, sens_s).shape)

model_B_obj = res["B"]["model"][0]
if hasattr(model_B_obj, "predictors_"):
    preds_list = list(model_B_obj.predictors_)
    if preds_list:
        b_tup = (preds_list[0], res["B"]["model"][1])
        chk("beeswarm model B",
            lambda: (xai_engine.global_beeswarm(b_tup, sample), plt.close("all"), "ok")[-1])

# ── TAB 5 ────────────────────────────────────────────────────────────────────
print("\n=== TAB 5 - DRIFT & PRIVACY ===")
chk("drift report",
    lambda: drift_engine.drift(Xtr.copy(), drift_engine.make_shifted(Xte.copy()))["share_drifted"])
chk("drift html",
    lambda: os.path.exists("artifacts/drift.html"))

combined = pd.concat([Xte.reset_index(drop=True), ste.reset_index(drop=True)], axis=1)
chk("k_anonymity",  lambda: privacy_engine.k_anonymity(combined))
chk("mia model A",  lambda: privacy_engine.mia_proxy(res["A"]["model"], Xtr, ytr, Xte, yte)["risk"])
chk("mia model B",  lambda: privacy_engine.mia_proxy(res["B"]["model"], Xtr, ytr, Xte, yte)["risk"])

# ── TAB 6 ────────────────────────────────────────────────────────────────────
print("\n=== TAB 6 - GOVERNANCE ===")
dr  = drift_engine.drift(Xtr.copy(), drift_engine.make_shifted(Xte.copy()))
ka  = privacy_engine.k_anonymity(combined)
mia = privacy_engine.mia_proxy(res["A"]["model"], Xtr, ytr, Xte, yte)
fa_A = fairness_engine.audit(yte, res["A"]["preds"], ste["sex"])
fa_B = fairness_engine.audit(yte, res["B"]["preds"], ste["sex"])
chk("model_card generate",
    lambda: (model_card.generate(res["A"]["metrics"], res["B"]["metrics"],
                                 fa_A, fa_B, {"k_anon": ka, "mia": mia}, dr),
             os.path.exists("artifacts/model_card.md"))[-1])
chk("model_card.md size",
    lambda: f"{os.path.getsize('artifacts/model_card.md')} bytes")

# ── MODEL C — all input models ───────────────────────────────────────────────
print("\n=== MODEL C - INPUT MODELS ===")
input_dir = r"c:\Users\dhira\Documents\Dipak_Project\input"
for fname in sorted(os.listdir(input_dir)):
    if not fname.endswith(".joblib"):
        continue
    try:
        obj = joblib.load(os.path.join(input_dir, fname))
        m_c, prep_c = obj
        chk(f"[C] predict {fname}",
            lambda m=m_c, p=prep_c: models.predict((m, p), Xte)[0].shape)
        chk(f"[C] metrics {fname}",
            lambda m=m_c, p=prep_c: models.metrics(yte, *models.predict((m, p), Xte)))
        for attr in ["sex", "race", "age_bin"]:
            chk(f"[C] fairness {attr} {fname}",
                lambda m=m_c, p=prep_c, a=attr: fairness_engine.audit(
                    yte, models.predict((m, p), Xte)[0], ste[a])["demographic_parity_diff"])
        chk(f"[C] intersectional {fname}",
            lambda m=m_c, p=prep_c: fairness_engine.intersectional(
                yte, models.predict((m, p), Xte)[0], ste[["sex", "race"]]).shape)
        chk(f"[C] confusion {fname}",
            lambda m=m_c, p=prep_c: (fairness_engine.confusion_by_group(
                yte, models.predict((m, p), Xte)[0], ste["sex"]), plt.close("all"), "ok")[-1])
        chk(f"[C] beeswarm {fname}",
            lambda m=m_c, p=prep_c: (xai_engine.global_beeswarm((m, p), sample), plt.close("all"), "ok")[-1])
        chk(f"[C] waterfall {fname}",
            lambda m=m_c, p=prep_c: (xai_engine.local_waterfall((m, p), sample, 0), plt.close("all"), "ok")[-1])
        chk(f"[C] group_diff {fname}",
            lambda m=m_c, p=prep_c: xai_engine.group_diff((m, p), sample, sens_s).shape)
        chk(f"[C] mia {fname}",
            lambda m=m_c, p=prep_c: privacy_engine.mia_proxy((m, p), Xtr, ytr, Xte, yte)["risk"])
    except Exception as e:
        FAIL.append(f"load {fname}")
        print(f"  [FAIL] load {fname}: {e}")
    plt.close("all")

# ── APP SYNTAX ───────────────────────────────────────────────────────────────
print("\n=== APP SYNTAX ===")
import ast
chk("app.py syntax", lambda: ast.parse(open("app.py", encoding="utf-8").read()) or "OK")

# ── SUMMARY ──────────────────────────────────────────────────────────────────
total = len(PASS) + len(FAIL)
print(f"\n{'='*55}")
print(f"  PASSED : {len(PASS)} / {total}")
if FAIL:
    print(f"  FAILED : {len(FAIL)}")
    for f in FAIL:
        print(f"    - {f}")
else:
    print("  ALL TESTS PASSED")
print("="*55)
