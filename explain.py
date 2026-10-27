from sklearn.model_selection import train_test_split
from src import data_loader, models, xai_engine


if __name__ == "__main__":
    X, y, sens = data_loader.load()
    Xtr, Xte, ytr, _, str_, _ = train_test_split(
        X, y, sens, test_size=.2, random_state=42, stratify=y)
    model = models.train_all(Xtr, ytr, str_)["A"]
    values = xai_engine._get_shap_values(model, Xte.head(5))
    print(f"shap={values.values.shape} features={values.feature_names}")
