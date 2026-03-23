# SUBHASHINI | Roll No: 727823tuam053
"""
Pipeline Stage 1 — Data Preparation
Air Quality Index Prediction | MLOps Assignment
Student : SUBHASHINI | Roll No: 727823tuam053
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

ROLL_NUMBER  = "727823tuam053"
STUDENT_NAME = "SUBHASHINI"
RANDOM_SEED  = 42
np.random.seed(RANDOM_SEED)

print(f"ROLL_NUMBER={ROLL_NUMBER} | TIMESTAMP={datetime.utcnow().isoformat()}Z")
print(f"[Stage 1 - data_prep.py] Student: {STUDENT_NAME} | Roll: {ROLL_NUMBER}")

def generate_aqi_dataset(n=2000, seed=42):
    np.random.seed(seed)
    PM25      = np.random.gamma(shape=3.0, scale=20, size=n)
    PM10      = PM25 * np.random.uniform(1.4, 2.0, n)
    NO2       = np.random.gamma(shape=2.5, scale=18, size=n)
    SO2       = np.random.gamma(shape=1.5, scale=10, size=n)
    CO        = np.random.gamma(shape=2.0, scale=0.8, size=n)
    O3        = np.random.normal(loc=50, scale=20, size=n).clip(0)
    Temp      = np.random.normal(loc=28, scale=8, size=n)
    Humidity  = np.random.uniform(30, 90, size=n)
    WindSpeed = np.random.exponential(scale=5, size=n)
    AQI = (0.35*PM25 + 0.20*PM10 + 0.15*NO2 + 0.10*SO2 +
           15*CO + 0.08*O3 - 0.5*WindSpeed + 0.3*Humidity +
           np.random.normal(0, 8, n)).clip(0, 500)
    return pd.DataFrame({
        "PM2.5": PM25, "PM10": PM10, "NO2": NO2, "SO2": SO2,
        "CO": CO, "O3": O3, "Temperature": Temp,
        "Humidity": Humidity, "WindSpeed": WindSpeed, "AQI": AQI
    })

def main():
    print(f"[data_prep] Loading and preprocessing AQI dataset...")
    df = generate_aqi_dataset()
    print(f"[data_prep] Dataset shape: {df.shape}")
    print(f"[data_prep] Missing values: {df.isnull().sum().sum()}")
    print(f"[data_prep] AQI stats — mean={df['AQI'].mean():.2f}, std={df['AQI'].std():.2f}")

    # Remove outliers using IQR
    Q1, Q3 = df["AQI"].quantile(0.25), df["AQI"].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df["AQI"] >= Q1 - 1.5*IQR) & (df["AQI"] <= Q3 + 1.5*IQR)]
    print(f"[data_prep] After outlier removal: {df.shape}")

    X = df.drop("AQI", axis=1)
    y = df["AQI"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED)

    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_s  = pd.DataFrame(scaler.transform(X_test),      columns=X.columns)

    os.makedirs("outputs", exist_ok=True)
    X_train_s.to_csv("outputs/X_train.csv", index=False)
    X_test_s.to_csv("outputs/X_test.csv",   index=False)
    y_train.to_csv("outputs/y_train.csv",   index=False)
    y_test.to_csv("outputs/y_test.csv",     index=False)
    joblib.dump(scaler, "outputs/scaler.pkl")

    print(f"[data_prep] Train size: {X_train_s.shape}, Test size: {X_test_s.shape}")
    print(f"[data_prep] Artifacts saved to outputs/")
    print(f"[data_prep] Stage 1 COMPLETE | Roll: {ROLL_NUMBER} | "
          f"Time: {datetime.utcnow().isoformat()}Z")

if __name__ == "__main__":
    main()
