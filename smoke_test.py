import sys, os
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from sklearn.model_selection import train_test_split
from src import data_loader, models, fairness_engine, privacy_engine, drift_engine, model_card

print("Imports OK")
X, y, sens = data_loader.load()
print(f"Data: {X.shape}")

X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(X, y, sens, test_size=0.2, random_state=42, stratify=y)
trained = models.train_all(X_tr, y_tr, s_tr)
print("Models trained")

results = {}
for name, m in trained.items():
    p, pr = models.predict(m, X_te)
    met = models.metrics(y_te, p, pr)
    fa = fairness_engine.audit(y_te, p, s_te["sex"])
    print(f"Model {name}: acc={met['accuracy']} DPD={fa['demographic_parity_diff']}")
    results[name] = {"model": m, "preds": p, "proba": pr, "metrics": met, "fairness": fa}

ka = privacy_engine.k_anonymity(X_te)
mia = privacy_engine.mia_proxy(trained["A"], X_tr, y_tr, X_te, y_te)
print(f"Privacy k={ka['min_k']} MIA={mia['risk']}")

ref = X_tr.copy()
cur = drift_engine.make_shifted(X_te.copy())
dr = drift_engine.drift(ref, cur)
print(f"Drift share={dr['share_drifted']}")

pA, prA = results["A"]["preds"], results["A"]["proba"]
pB, prB = results["B"]["preds"], results["B"]["proba"]
fA = results["A"]["fairness"]
fB = results["B"]["fairness"]

mc = model_card.generate(
    results["A"]["metrics"], results["B"]["metrics"],
    fA, fB,
    {"k_anon": ka, "mia": mia},
    dr,
)
assert os.path.exists("artifacts/model_card.md"), "model_card.md missing!"
print("model_card.md OK")
print("SMOKE TEST PASSED")
