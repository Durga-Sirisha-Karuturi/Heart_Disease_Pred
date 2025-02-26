import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import yaml

def load_data(data_path):
    return pd.read_csv(data_path)

test_size = 0.20
def split_data(data):
    X = data.drop('target', axis=1)
    y = data['target']
    return train_test_split(X, y, test_size = test_size, random_state=42)

def preprocess_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def save_data(X_train, X_test, y_train, y_test, output_path):
    os.makedirs(output_path, exist_ok=True)
    pd.DataFrame(X_train).to_csv(os.path.join(output_path, 'X_train.csv'), index=False)
    pd.DataFrame(X_test).to_csv(os.path.join(output_path, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_path, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_path, 'y_test.csv'), index=False)

def main():
    data_path = "data/raw/data.csv"
    output_path = "data/processed"
    
    data = load_data(data_path)
    X_train, X_test, y_train, y_test = split_data(data)
    X_train_scaled, X_test_scaled, scaler = preprocess_data(X_train, X_test)
    save_data(X_train_scaled, X_test_scaled, y_train, y_test, output_path)

if __name__ == "__main__":
    main()