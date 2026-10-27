from src.data_loader import load


if __name__ == "__main__":
    X, y, sens = load()
    print(f"data={X.shape} target={y.shape} sensitive={sens.columns.tolist()}")
