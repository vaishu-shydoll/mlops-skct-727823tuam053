# SUBHASHINI | Roll No: 727823tuam053
"""
Pipeline Stage 2 — Model Training
Air Quality Index Prediction | MLOps Assignment
Student : SUBHASHINI | Roll No: 727823tuam053
"""

import numpy as np
import pandas as pd
import os, time, json
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

ROLL_NUMBER  = "727823tuam053"
STUDENT_NAME = "SUBHASHINI"
RANDOM_SEED  = 42

print(f"ROLL_NUMBER={ROLL_NUMBER} | TIMESTAMP={datetime.utcnow().isoformat()}Z")
print(f"[Stage 2 - train_pipeline.py] Student: {STUDENT_NAME} | Roll: {ROLL_NUMBER}")

def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def main():
    print("[train] Loading preprocessed data from outputs/...")
    X_train = pd.read_csv("outputs/X_train.csv")
    X_test  = pd.read_csv("outputs/X_test.csv")
    y_train = pd.read_csv("outputs/y_train.csv").squeeze()
    y_test  = pd.read_csv("outputs/y_test.csv").squeeze()

    print(f"[train] X_train: {X_train.shape}, X_test: {X_test.shape}")

    model = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05,
        max_depth=5, subsample=0.8, random_state=RANDOM_SEED)

    print("[train] Training GradientBoostingRegressor (best model)...")
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = round(time.time() - t0, 4)

    y_pred = model.predict(X_test)
    metrics = {
        "MAE":   round(mean_absolute_error(y_test, y_pred), 4),
        "RMSE":  round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
        "R2":    round(r2_score(y_test, y_pred), 4),
        "MAPE":  round(mape(y_test.values, y_pred), 4),
        "training_time_seconds": train_time
    }
    print(f"[train] Metrics: {metrics}")

    os.makedirs("outputs", exist_ok=True)
    joblib.dump(model, "outputs/best_model.pkl")
    with open("outputs/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[train] Model saved to outputs/best_model.pkl")
    print(f"[train] Stage 2 COMPLETE | Roll: {ROLL_NUMBER} | "
          f"Time: {datetime.utcnow().isoformat()}Z")

if __name__ == "__main__":
    main()
