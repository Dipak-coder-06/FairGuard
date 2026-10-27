from sklearn.model_selection import train_test_split
from src import data_loader, drift_engine


if __name__ == "__main__":
    X, _, _ = data_loader.load()
    ref, cur = train_test_split(X, test_size=.2, random_state=42)
    print(drift_engine.drift(ref, drift_engine.make_shifted(cur)))
