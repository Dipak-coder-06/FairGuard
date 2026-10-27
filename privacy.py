import pandas as pd
from sklearn.model_selection import train_test_split
from src import data_loader, privacy_engine


if __name__ == "__main__":
    X, _, sens = data_loader.load()
    _, test, _, _, _, sens_test = train_test_split(
        X, sens, sens, test_size=.2, random_state=42)
    print(privacy_engine.k_anonymity(
        pd.concat([test.reset_index(drop=True), sens_test.reset_index(drop=True)], axis=1)))
