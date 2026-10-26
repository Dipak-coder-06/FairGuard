import os, sys
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
sys.path.insert(0, r"c:\Users\dhira\Documents\Dipak_Project")
os.chdir(r"c:\Users\dhira\Documents\Dipak_Project")

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from sklearn.model_selection import train_test_split
from src import data_loader, models, fairness_engine, xai_engine, drift_engine, privacy_engine, model_card
import joblib, os

print("=== DATA ===")
X, y, sens = data_loader.load()
X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(X, y, sens, test_size=0.2, random_state=42, stratify=y)
print(f"X_tr={X_tr.shape} X_te={X_te.shape}")

print("=== MODELS ===")
trained = models.train_all(X_tr, y_tr, s_tr)
results = {}
for name, m in trained.items():
    p, pr = models.predict(m, X_te)
    met   = models.metrics(y_te, p, pr)
    fa    = fairness_engine.audit(y_te, p, s_te["sex"])
    print(f"  Model {name}: acc={met['accuracy']} auc={met['auc']} DPD={fa['demographic_parity_diff']} EOD={fa['equalized_odds_diff']} DI={fa['disparate_impact']}")
    results[name] = {"model": m, "preds": p, "proba": pr, "metrics": met, "fairness": fa}

print("=== FAIRNESS (race) ===")
fa_race = fairness_engine.audit(y_te, results["A"]["preds"], s_te["race"])
print(f"  DPD={fa_race['demographic_parity_diff']} status={fairness_engine.status('demographic_parity_diff', fa_race['demographic_parity_diff'])}")

print("=== FAIRNESS (age_bin) ===")
fa_age = fairness_engine.audit(y_te, results["A"]["preds"], s_te["age_bin"])
print(f"  DPD={fa_age['demographic_parity_diff']}")

print("=== INTERSECTIONAL ===")
inter = fairness_engine.intersectional(y_te, results["A"]["preds"], s_te[["sex","race"]])
print(f"  Groups: {list(inter.index[:3])}")

print("=== CONFUSION BY GROUP ===")
fig = fairness_engine.confusion_by_group(y_te, results["A"]["preds"], s_te["sex"])
print(f"  Figure OK: {fig}")
import matplotlib.pyplot as plt; plt.close("all")

print("=== XAI (beeswarm, bar, waterfall, group_diff) ===")
sample = X_te.head(300).reset_index(drop=True)
sens_s = s_te["sex"].reset_index(drop=True).iloc[:300]
fig_bee = xai_engine.global_beeswarm(results["A"]["model"], sample)
print(f"  beeswarm OK: {fig_bee.get_size_inches()}")
fig_bar = xai_engine.global_bar(results["A"]["model"], sample)
print(f"  bar OK: {fig_bar.get_size_inches()}")
fig_wf  = xai_engine.local_waterfall(results["A"]["model"], sample, idx=0)
print(f"  waterfall OK: {fig_wf.get_size_inches()}")
gd = xai_engine.group_diff(results["A"]["model"], sample, sens_s)
print(f"  group_diff OK: {gd.shape}")
plt.close("all")

print("=== DRIFT ===")
dr = drift_engine.drift(X_tr.copy(), drift_engine.make_shifted(X_te.copy()))
print(f"  share_drifted={dr['share_drifted']} features={len(dr['per_feature'])}")

print("=== PRIVACY ===")
import pandas as pd
combined = pd.concat([X_te.reset_index(drop=True), s_te.reset_index(drop=True)], axis=1)
ka  = privacy_engine.k_anonymity(combined)
mia = privacy_engine.mia_proxy(results["A"]["model"], X_tr, y_tr, X_te, y_te)
print(f"  k_anon min_k={ka['min_k']} pct_lt5={ka['pct_rows_k_lt_5']}%")
print(f"  MIA gap={mia['auc_gap']} risk={mia['risk']}")

print("=== MODEL CARD ===")
mc = model_card.generate(
    results["A"]["metrics"], results["B"]["metrics"],
    results["A"]["fairness"], results["B"]["fairness"],
    {"k_anon": ka, "mia": mia}, dr
)
assert os.path.exists("artifacts/model_card.md")
print(f"  model_card.md OK ({os.path.getsize('artifacts/model_card.md')} bytes)")

print("=== INPUT MODELS ===")
for f in sorted(os.listdir("input")):
    if not f.endswith(".joblib"): continue
    obj = joblib.load(os.path.join("input", f))
    assert isinstance(obj, tuple) and len(obj)==2, f"Bad format: {f}"
    m, prep = obj
    p2, pr2 = models.predict((m, prep), X_te)
    met2 = models.metrics(y_te, p2, pr2)
    fa2  = fairness_engine.audit(y_te, p2, s_te["sex"])
    print(f"  {f:40s} acc={met2['accuracy']} DPD={fa2['demographic_parity_diff']}")

print("\n=== ALL TESTS PASSED ===")
