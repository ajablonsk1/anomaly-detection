import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_data(file_paths):
    dfs = []
    for fp in file_paths:
        df = pd.read_csv(fp, on_bad_lines="skip", low_memory=False)
        df.columns = df.columns.str.strip()
        dfs.append(df)
        print(
            f"  - {fp.split('/')[-1]}: {len(df)} records, label: {df['Label'].unique()[0]}"
        )

    data = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal records loaded: {len(data)}")
    print(f"Attack classes: {data['Label'].unique()}")
    return data


def preprocess_data(data):
    data = data.drop(
        [
            "Unnamed: 0",
            "Flow ID",
            "Source IP",
            "Destination IP",
            "Timestamp",
            "Source Port",
            "Destination Port",
        ],
        axis=1,
        errors="ignore",
    )

    for col in data.columns:
        if col != "Label" and data[col].dtype == "object":
            data = data.drop(col, axis=1)

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna()

    return data


def prepare_binary_labels(y):
    return y.apply(lambda x: 0 if x == "BENIGN" else 1)


def prepare_multiclass_labels(y):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return y_encoded, le


def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled
