# SUBHASHINI | Roll No: 727823tuam053
"""
Pipeline Stage 3 — Model Evaluation
Air Quality Index Prediction | MLOps Assignment
Student : SUBHASHINI | Roll No: 727823tuam053
"""

import numpy as np
import pandas as pd
import os, json
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

ROLL_NUMBER  = "727823tuam053"
STUDENT_NAME = "SUBHASHINI"

print(f"ROLL_NUMBER={ROLL_NUMBER} | TIMESTAMP={datetime.utcnow().isoformat()}Z")
print(f"[Stage 3 - evaluate.py] Student: {STUDENT_NAME} | Roll: {ROLL_NUMBER}")

def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def smape(y_true, y_pred):
    d = (np.abs(y_true) + np.abs(y_pred)) / 2
    m = d != 0
    return np.mean(np.abs(y_true[m] - y_pred[m]) / d[m]) * 100

def main():
    print("[evaluate] Loading model and test data...")
    model  = joblib.load("outputs/best_model.pkl")
    X_test = pd.read_csv("outputs/X_test.csv")
    y_test = pd.read_csv("outputs/y_test.csv").squeeze()

    y_pred = model.predict(X_test)

    report = {
        "student_name": STUDENT_NAME,
        "roll_number":  ROLL_NUMBER,
        "dataset":      "AirQualityIndexPrediction",
        "timestamp":    datetime.utcnow().isoformat() + "Z",
        "MAE":    round(mean_absolute_error(y_test, y_pred), 4),
        "RMSE":   round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
        "R2":     round(r2_score(y_test, y_pred), 4),
        "MAPE":   round(mape(y_test.values, y_pred), 4),
        "SMAPE":  round(smape(y_test.values, y_pred), 4),
        "n_test_samples": len(y_test),
        "target_met_R2_gt_0.65": bool(r2_score(y_test, y_pred) > 0.65),
        "target_met_MAPE_lt_20": bool(mape(y_test.values, y_pred) < 20.0),
    }

    print("\n[evaluate] ===== FINAL EVALUATION REPORT =====")
    for k, v in report.items():
        print(f"  {k:35s}: {v}")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[evaluate] Report saved to outputs/evaluation_report.json")
    print(f"[evaluate] Stage 3 COMPLETE | Roll: {ROLL_NUMBER} | "
          f"Time: {datetime.utcnow().isoformat()}Z")

if __name__ == "__main__":
    main()
