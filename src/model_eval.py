import pandas as pd
import pickle
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import yaml

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def main():
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()  # Convert to Series
    model = load_pickle("models/model.pkl")

    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }

    os.makedirs("metrics", exist_ok=True)
    pd.DataFrame([metrics]).to_csv("metrics/metrics.csv", index=False)
    print("Evaluation complete. Metrics saved.")

if __name__ == "__main__":
    main()
