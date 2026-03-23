# MLOps Assignment — Air Quality Index Prediction

| Field | Value |
|---|---|
| **Student Name** | SUBHASHINI |
| **Roll Number** | 727823tuam053 |
| **Dataset** | Air Quality Index Prediction |
| **MLflow Experiment** | `SKCT_727823tuam053_AirQualityIndexPrediction` |
| **GitHub Repo** | `mlops-skct-727823tuam053` |

---

## Dataset Description
Synthetic AQI dataset with 2000 samples and 9 pollutant/weather features:
`PM2.5, PM10, NO2, SO2, CO, O3, Temperature, Humidity, WindSpeed` → predicts **AQI** (0–500 scale).

## Project Structure
```
├── code/
│   ├── training.py              # Component A — MLflow experiment tracking (14 runs)
│   ├── data_prep.py             # Pipeline Stage 1 — data preprocessing
│   ├── train_pipeline.py        # Pipeline Stage 2 — model training
│   ├── evaluate.py              # Pipeline Stage 3 — model evaluation
│   └── pipeline_727823tuam053.yml  # Azure ML pipeline definition
├── notebooks/
│   └── eda.ipynb                # EDA notebook with 3 plots
├── screenshots/                 # MLflow UI + Azure portal screenshots
├── report/
│   └── report.pdf               # 2-page assignment report
├── requirements.txt
└── README.md
```

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run EDA + all MLflow experiments (Component A)
python code/training.py

# 3. View MLflow UI
mlflow ui --port 5000
# Open http://localhost:5000

# 4. Run Azure Pipeline stages locally
python code/data_prep.py
python code/train_pipeline.py
python code/evaluate.py
```

## Best Model Results
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| R²     | 0.9361 | > 0.65 | ✅ Met |
| MAE    | 6.245  | —      | — |
| RMSE   | 7.789  | —      | — |
| MAPE   | 7.77%  | < 20%  | ✅ Met |

**Best Model:** Ridge Regression (alpha=10.0)

## Experiment Summary (14 runs)
| Algorithm | R² |
|---|---|
| Ridge (α=10) | **0.9361** |
| LinearRegression | 0.9359 |
| Ridge (α=1) | 0.9359 |
| Lasso (α=0.5) | 0.9347 |
| ElasticNet | 0.9148 |
| GradientBoosting (lr=0.1) | 0.9002 |
| ExtraTrees | 0.8964 |
| GradientBoosting (lr=0.05) | 0.8956 |
| RandomForest (n=200) | 0.8891 |
| RandomForest (n=100) | 0.8876 |
| SVR (RBF) | 0.8505 |
| KNN (k=5) | 0.8303 |
| DecisionTree (depth=10) | 0.7806 |
| DecisionTree (depth=5) | 0.7776 |
