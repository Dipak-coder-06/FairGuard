from sklearn.model_selection import train_test_split
from src import data_loader, models


if __name__ == "__main__":
    X, y, sens = data_loader.load()
    Xtr, Xte, ytr, yte, str_, _ = train_test_split(
        X, y, sens, test_size=.2, random_state=42, stratify=y)
    for name, model in models.train_all(Xtr, ytr, str_).items():
        pred, proba = models.predict(model, Xte)
        print(name, models.metrics(yte, pred, proba))
