import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
import os
import yaml

def load_data(data_path):
    X_train = pd.read_csv(os.path.join(data_path, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(data_path, 'y_train.csv'))
    return X_train, y_train.squeeze()  # Convert y_train to Series

solver= yaml.safe_load(open("params.yaml"))["model_build"]["solver"]
max_iter= yaml.safe_load(open("params.yaml"))["model_build"]["max_iter"]

def train_model(X_train, y_train):
    model = LogisticRegression(solver=solver, max_iter=max_iter)
    model.fit(X_train, y_train)
    return model

def save_model(model, model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)  # Ensure directory exists
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

def main():
    data_path = "data/processed"
    model_path = "models/model.pkl"

    X_train, y_train = load_data(data_path)
    model = train_model(X_train, y_train)
    save_model(model, model_path)
    print(f"Model saved at {model_path}")

if __name__ == "__main__":
    main()
