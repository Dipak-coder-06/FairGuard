from sklearn.model_selection import train_test_split
from src import data_loader, models, fairness_engine


if __name__ == "__main__":
    X, y, sens = data_loader.load()
    Xtr, Xte, ytr, yte, str_, ste = train_test_split(
        X, y, sens, test_size=.2, random_state=42, stratify=y)
    for name, model in models.train_all(Xtr, ytr, str_).items():
        pred, _ = models.predict(model, Xte)
        print(name, fairness_engine.audit(yte, pred, ste["sex"]))
