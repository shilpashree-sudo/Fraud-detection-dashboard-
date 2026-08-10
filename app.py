import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from flask import Flask
app = Flask(_name_)
import xgboost as xgb

MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "fraud_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")

def generate_synthetic_data(n_samples: int = 10000) -> pd.DataFrame:
    np.random.seed(42)
    # PCA features V1-V28
    features = {f"V{i}": np.random.normal(0, 1, n_samples) for i in range(1, 29)}
    # Add Amount and Class (99.5% legit, 0.5% fraud)
    features["Amount"] = np.random.exponential(scale=100.0, size=n_samples)
    features["Class"] = np.random.choice([0, 1], size=n_samples, p=[0.995, 0.005])
    return pd.DataFrame(features)

def train_and_save_model():
    print("[1/4] Generating dataset...")
    df = generate_synthetic_data()

    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Preprocessing
    scaler = StandardScaler()
    X["Amount"] = scaler.fit_transform(X[["Amount"]])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Balance classes with SMOTE
    print("[2/4] Applying SMOTE oversampling...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Train Model
    print("[3/4] Training XGBoost Classifier...")
    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        eval_metric="logloss"
    )
    model.fit(X_train_res, y_train_res)

    # Save artifacts
    print("[4/4] Saving model artifacts...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print("Model training complete! Artifacts saved to ./models/")

if __name__ == "__main__":
    train_and_save_model()@app.route("/")
def home():
    return "Fraud Detection Dashboard is running!"

