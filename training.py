"""
SUBHASHINI | Roll No: 727823tuam053 | Dataset: Air Quality Index Prediction
MLOps Individual Assignment - Component A: MLflow Experiment Tracking
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import joblib
from datetime import datetime
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               ExtraTreesRegressor, AdaBoostRegressor)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# ─── Constants ───────────────────────────────────────────────────────────────
STUDENT_NAME  = "SUBHASHINI"
ROLL_NUMBER   = "727823tuam053"
DATASET_NAME  = "AirQualityIndexPrediction"
EXPERIMENT    = f"SKCT_{ROLL_NUMBER}_{DATASET_NAME}"
RANDOM_SEED   = 42
np.random.seed(RANDOM_SEED)

# ─── 1. Synthetic AQI Dataset ─────────────────────────────────────────────────
def generate_aqi_dataset():
    """
    Generates a realistic AQI dataset with features:
    PM2.5, PM10, NO2, SO2, CO, O3, Temperature, Humidity, WindSpeed
    """
    n = 2000
    np.random.seed(RANDOM_SEED)

    PM25       = np.random.gamma(shape=3.0, scale=20, size=n)
    PM10       = PM25 * np.random.uniform(1.4, 2.0, n)
    NO2        = np.random.gamma(shape=2.5, scale=18, size=n)
    SO2        = np.random.gamma(shape=1.5, scale=10, size=n)
    CO         = np.random.gamma(shape=2.0, scale=0.8, size=n)
    O3         = np.random.normal(loc=50, scale=20, size=n).clip(0)
    Temp       = np.random.normal(loc=28, scale=8, size=n)
    Humidity   = np.random.uniform(30, 90, size=n)
    WindSpeed  = np.random.exponential(scale=5, size=n)

    AQI = (0.35 * PM25 + 0.20 * PM10 + 0.15 * NO2 + 0.10 * SO2
           + 15  * CO   + 0.08 * O3
           - 0.5 * WindSpeed + 0.3 * Humidity
           + np.random.normal(0, 8, n))
    AQI = AQI.clip(0, 500)

    df = pd.DataFrame({
        "PM2.5": PM25, "PM10": PM10, "NO2": NO2, "SO2": SO2,
        "CO": CO, "O3": O3, "Temperature": Temp,
        "Humidity": Humidity, "WindSpeed": WindSpeed, "AQI": AQI
    })
    return df

# ─── 2. EDA ───────────────────────────────────────────────────────────────────
def run_eda(df, out_dir="notebooks"):
    os.makedirs(out_dir, exist_ok=True)

    # Plot 1 – AQI Distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df["AQI"], kde=True, color="steelblue", ax=ax)
    ax.set_title(f"AQI Distribution | {STUDENT_NAME} | {ROLL_NUMBER}")
    ax.set_xlabel("AQI Value"); ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/plot1_aqi_distribution.png", dpi=100)
    plt.close()

    # Plot 2 – Correlation Heatmap
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title(f"Feature Correlation Heatmap | {STUDENT_NAME} | {ROLL_NUMBER}")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/plot2_correlation_heatmap.png", dpi=100)
    plt.close()

    # Plot 3 – PM2.5 vs AQI scatter
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(df["PM2.5"], df["AQI"], c=df["NO2"], cmap="plasma",
                    alpha=0.5, s=10)
    plt.colorbar(sc, ax=ax, label="NO2")
    ax.set_xlabel("PM2.5"); ax.set_ylabel("AQI")
    ax.set_title(f"PM2.5 vs AQI (coloured by NO2) | {STUDENT_NAME}")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/plot3_pm25_vs_aqi.png", dpi=100)
    plt.close()

    print(f"[EDA] 3 plots saved to '{out_dir}/'")
    print(df.describe().to_string())

# ─── 3. Metric helpers ────────────────────────────────────────────────────────
def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask  = denom != 0
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100

def model_size_mb(model, path="/tmp/_tmp_model.pkl"):
    joblib.dump(model, path)
    size = os.path.getsize(path) / (1024 * 1024)
    os.remove(path)
    return round(size, 4)

# ─── 4. Single MLflow run ─────────────────────────────────────────────────────
def run_experiment(run_name, model, params, X_train, X_test, y_train, y_test, n_features):
    with mlflow.start_run(run_name=run_name):
        # Tags
        mlflow.set_tag("student_name", STUDENT_NAME)
        mlflow.set_tag("roll_number",  ROLL_NUMBER)
        mlflow.set_tag("dataset",      DATASET_NAME)

        # Log hyperparameters
        mlflow.log_params(params)
        mlflow.log_param("random_seed", RANDOM_SEED)
        mlflow.log_param("n_features",  n_features)

        # Train
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = round(time.time() - t0, 4)

        # Predict & Metrics
        y_pred = model.predict(X_test)
        mae_val   = round(mean_absolute_error(y_test, y_pred), 4)
        rmse_val  = round(np.sqrt(mean_squared_error(y_test, y_pred)), 4)
        r2_val    = round(r2_score(y_test, y_pred), 4)
        mape_val  = round(mape(y_test.values, y_pred), 4)
        smape_val = round(smape(y_test.values, y_pred), 4)
        size_mb   = model_size_mb(model)

        mlflow.log_metric("MAE",                    mae_val)
        mlflow.log_metric("RMSE",                   rmse_val)
        mlflow.log_metric("R2",                     r2_val)
        mlflow.log_metric("MAPE",                   mape_val)
        mlflow.log_metric("SMAPE",                  smape_val)
        mlflow.log_metric("training_time_seconds",  train_time)
        mlflow.log_metric("model_size_mb",          size_mb)

        # Save model artifact
        mlflow.sklearn.log_model(model, "model")

        run_id = mlflow.active_run().info.run_id
        print(f"  ✓ {run_name:45s} | R²={r2_val:6.4f} | MAE={mae_val:7.3f} "
              f"| MAPE={mape_val:6.2f}% | t={train_time}s | run_id={run_id[:8]}…")
        return run_id, r2_val, mae_val

# ─── 5. Main ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print(f"  MLOps Assignment — Component A")
    print(f"  Student  : {STUDENT_NAME}")
    print(f"  Roll No  : {ROLL_NUMBER}")
    print(f"  Dataset  : Air Quality Index Prediction")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Dataset
    df = generate_aqi_dataset()
    print(f"\n[DATA] Shape: {df.shape} | AQI range: {df['AQI'].min():.1f}–{df['AQI'].max():.1f}")

    # EDA
    run_eda(df, out_dir="notebooks")

    # Prepare
    X = df.drop("AQI", axis=1)
    y = df["AQI"]
    n_features = X.shape[1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_s  = pd.DataFrame(scaler.transform(X_test),      columns=X.columns)

    # MLflow setup
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment(EXPERIMENT)
    print(f"\n[MLflow] Experiment: {EXPERIMENT}")
    print("-" * 70)

    experiments = [
        # (run_name, model, params, scaled?)
        ("LinearRegression_baseline",
         LinearRegression(), {"fit_intercept": True, "algorithm": "LinearRegression"}, True),

        ("Ridge_alpha1",
         Ridge(alpha=1.0, random_state=RANDOM_SEED),
         {"alpha": 1.0, "algorithm": "Ridge"}, True),

        ("Ridge_alpha10",
         Ridge(alpha=10.0, random_state=RANDOM_SEED),
         {"alpha": 10.0, "algorithm": "Ridge"}, True),

        ("Lasso_alpha0.5",
         Lasso(alpha=0.5, random_state=RANDOM_SEED, max_iter=3000),
         {"alpha": 0.5, "algorithm": "Lasso"}, True),

        ("ElasticNet_a0.5_l10.5",
         ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=RANDOM_SEED, max_iter=3000),
         {"alpha": 0.5, "l1_ratio": 0.5, "algorithm": "ElasticNet"}, True),

        ("DecisionTree_depth5",
         DecisionTreeRegressor(max_depth=5, random_state=RANDOM_SEED),
         {"max_depth": 5, "algorithm": "DecisionTreeRegressor"}, False),

        ("DecisionTree_depth10",
         DecisionTreeRegressor(max_depth=10, random_state=RANDOM_SEED),
         {"max_depth": 10, "algorithm": "DecisionTreeRegressor"}, False),

        ("RandomForest_n100",
         RandomForestRegressor(n_estimators=100, max_depth=10, random_state=RANDOM_SEED),
         {"n_estimators": 100, "max_depth": 10, "algorithm": "RandomForestRegressor"}, False),

        ("RandomForest_n200_d15",
         RandomForestRegressor(n_estimators=200, max_depth=15, random_state=RANDOM_SEED),
         {"n_estimators": 200, "max_depth": 15, "algorithm": "RandomForestRegressor"}, False),

        ("GradientBoosting_lr0.1",
         GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, max_depth=4, random_state=RANDOM_SEED),
         {"n_estimators": 150, "learning_rate": 0.1, "max_depth": 4, "algorithm": "GradientBoostingRegressor"}, False),

        ("GradientBoosting_lr0.05",
         GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=RANDOM_SEED),
         {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 5, "algorithm": "GradientBoostingRegressor"}, False),

        ("ExtraTrees_n150",
         ExtraTreesRegressor(n_estimators=150, max_depth=12, random_state=RANDOM_SEED),
         {"n_estimators": 150, "max_depth": 12, "algorithm": "ExtraTreesRegressor"}, False),

        ("KNN_k5",
         KNeighborsRegressor(n_neighbors=5),
         {"n_neighbors": 5, "algorithm": "KNeighborsRegressor"}, True),

        ("SVR_rbf_C10",
         SVR(kernel="rbf", C=10, epsilon=0.5),
         {"kernel": "rbf", "C": 10, "epsilon": 0.5, "algorithm": "SVR"}, True),
    ]

    results = []
    best_r2, best_run_id, best_model_name = -999, None, None

    for run_name, model, params, use_scaled in experiments:
        Xtr = X_train_s if use_scaled else X_train
        Xte = X_test_s  if use_scaled else X_test
        rid, r2, mae = run_experiment(run_name, model, params, Xtr, Xte, y_train, y_test, n_features)
        results.append({"Run": run_name, "R²": r2, "MAE": mae})
        if r2 > best_r2:
            best_r2, best_run_id, best_model_name = r2, rid, run_name
            best_model_obj = model

    # Save best model separately
    print("\n[Best Model]")
    print(f"  Name   : {best_model_name}")
    print(f"  R²     : {best_r2}")
    print(f"  Run ID : {best_run_id}")

    with mlflow.start_run(run_id=best_run_id):
        mlflow.set_tag("best_model", "true")

    # Summary table
    print("\n" + "=" * 70)
    print("  EXPERIMENT SUMMARY TABLE")
    print("=" * 70)
    res_df = pd.DataFrame(results).sort_values("R²", ascending=False)
    print(res_df.to_string(index=False))
    print("=" * 70)
    print(f"\n[DONE] {len(results)} experiments logged to MLflow experiment '{EXPERIMENT}'")

if __name__ == "__main__":
    main()
