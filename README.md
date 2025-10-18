# AnomAlert — Advanced Anomaly Detection System for Network Intrusion

**AnomAlert** is a self-contained, end-to-end project that demonstrates unsupervised anomaly detection for network intrusion using Isolation Forest and an Autoencoder (MLP) plus time-series forecasting with Prophet / ARIMA. The project includes data ingestion & preprocessing, model training, model saving/loading, clustering (DBSCAN) utilities, and a lightweight demo UI using **Gradio** (fast and simple — no Streamlit / MLflow required).

---

# Key features

* Synthetic + optional KDD dataset loader
* Preprocessing pipeline: categorical encoding, scaling, timestamp indexing
* Unsupervised detectors:

  * Isolation Forest
  * Autoencoder (MLPRegressor used as a reconstruction-based autoencoder)
* Dimensionality reduction (PCA) and DBSCAN clustering for grouping/identifying noise
* Time-series forecasting of aggregated anomaly scores via Prophet (ARIMA fallback)
* Simple, real-time detection function for single-sample scoring
* Gradio demo UI for quick local testing and visualization
* Model saving/loading with `joblib` (no MLflow dependency)
* Clear modular structure under `src/` for easy extension

---

# Repo structure

```
anomaly-detection-system/
├── README.md
├── config.yaml
├── requirements.txt
├── run_all.sh
├── app_gradio.py
├── train_models.py
├── save_models.py
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── features.py
│   ├── models.py
│   ├── forecast.py
│   └── detect.py
├── data/
│   └── raw/                # place KDDTrain+.csv & KDDTest+.csv here if available
├── models/                 # created by scripts; contains .joblib / artifacts
└── notebooks/
```

---

# Quick start (local)

### 1. Clone or copy the project

If you haven't already:

```bash
git clone https://github.com/rohitmannur007/AnomAlert.git
cd AnomAlert
```

### 2. (Optional) Create & activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> Note: `prophet` may take longer to install on some macOS setups. If installation fails, you can set `forecast.method: arima` in `config.yaml` to use ARIMA as a fallback.

### 4. Train models (will use synthetic data if no KDD CSVs present)

```bash
python train_models.py
```

This script:

* loads raw (or synthetic) data,
* preprocesses it,
* trains Isolation Forest and Autoencoder,
* saves models & preprocessors to `models/`.

### 5. Launch the demo (Gradio)

```bash
python app_gradio.py
```

Open `http://localhost:7860` in your browser to access the UI.

### 6. All-in-one (train + launch)

Make `run_all.sh` executable and run it:

```bash
chmod +x run_all.sh
./run_all.sh
```

---

# Configuration (`config.yaml`)

Example:

```yaml
data:
  raw_path: "./data/raw"
  processed_path: "./data/processed"

models:
  path: "./models"

forecast:
  method: "prophet"   # "prophet" or "arima"
```

Adjust paths or forecasting method as needed.

---

# Data

* Put your real dataset CSVs in `data/raw/` named:

  * `KDDTrain+.csv`
  * `KDDTest+.csv`
* If those files are not present, the project generates synthetic network-like data for demonstration.

**Important:** If you use a custom dataset, update `src/data.py` to map column names and select appropriate features.

---

# Important files explained

* `src/data.py` — data loading and preprocessing (encoding + scaling + timestamping)
* `src/features.py` — PCA, DBSCAN utilities
* `src/models.py` — training functions for Isolation Forest and Autoencoder; model save/load helpers
* `src/forecast.py` — Prophet & ARIMA wrappers for forecasting aggregated anomaly scores
* `src/detect.py` — single-sample preparation + detection function that combines model outputs
* `train_models.py` — top-level training script to produce models in `models/`
* `app_gradio.py` — Gradio demo UI (inspect data, run single-sample detection, show aggregated time-series)
* `run_all.sh` — convenience script to install deps (if needed), train, and launch the app

---

# How detection scoring works

* **Isolation Forest** returns `decision_function` (higher = normal). We invert it to get an anomaly score (`-decision_function`) so larger = more anomalous.
* **Autoencoder** uses reconstruction MSE per sample as anomaly score.
* **Combined score** is a simple mean of available model scores (you may replace with a weighted sum or more complex fusion logic).
* Aggregated time-series score is computed by averaging per-sample scores and then resampling by minute for plotting/forecasting.

---

# Forecasting

* Use Prophet (recommended) for forecasting aggregated anomaly scores. If Prophet isn't available, configure ARIMA in `config.yaml`.
* In `src/forecast.py` you can call `train_prophet(ts_df)` or `train_arima(ts_series)` to save forecasting models.

---

# Model persistence & deployment notes

* Models and preprocessors saved to `models/` using `joblib` (and Prophet's `.save()`).
* For production you can:

  * Containerize the app with Docker (add a Dockerfile and bind ports).
  * Deploy to a VM/container on AWS/GCP/Azure.
  * Add a lightweight API endpoint (FastAPI or Flask) that loads models once and serves detection requests (useful for real-time streaming ingestion).

---

# Testing & evaluation

* Evaluate unsupervised models using:

  * ROC-AUC (if you have labels): compare anomaly score vs true label.
  * Precision/Recall at top-k anomalies to gauge practical alerting performance.
* Use `sklearn.metrics` for evaluation after predictions.

---

# Troubleshooting & tips

* **"ModuleNotFoundError" for `src`** — run scripts from project root so Python's working directory includes `src`, or add `sys.path.append(os.getcwd())` early in a runner script (not recommended long-term).
* **Prophet install fails on macOS** — ensure `C++` toolchain & `cmdstanpy` dependencies are installed. Alternatively switch to ARIMA in `config.yaml`.
* **Port already in use (7860)** — change port in `app_gradio.py` `demo.launch(server_port=XXXX)` or set `share=False` if using public Gradio links.
* **Large datasets** — train on a subset locally, or use a cloud VM with more memory/CPU. Consider batching and incremental retraining.

---

# Development & extension ideas

* Add streaming ingestion (Kafka) and sliding-window real-time scoring.
* Add prescriptive actions (e.g., auto-block an IP or raise an alert) with a mock connector or cloud security API.
* Add MLflow or a model registry for versioning (optional; architecture designed to avoid it by default).
* Move heavy training to a scheduled job (CI/CD or Airflow / Databricks / AWS SageMaker pipelines).
* Add unit tests & CI workflow (GitHub Actions) to run static checks + basic smoke tests.

---

# License & Contribution

* Add a license of your choice (MIT recommended for open-source demos).
* Contributing: open issues, add PRs for enhancements, and include tests for new modules.

---

# Example Git push (one-shot)

If you want to push this repository to your GitHub `AnomAlert` repo (replace path with your project path):

```bash
cd ~/Desktop/untitled\ folder  # or your project path
git init
git add .
git commit -m "Initial commit: AnomAlert — anomaly detection system"
git branch -M main
git remote add origin https://github.com/rohitmannur007/AnomAlert.git
git push -u origin main
```

---

# Contact / Support

If you run into environment-specific issues (macOS M1/M2, missing build tools, or Prophet install problems), include:

* Python version (e.g., `python --version`)
* OS and architecture (e.g., macOS 13.4 on arm64)
* Exact error trace — and I’ll help you debug.

---

## Final note

This project is designed to be modular and educational: you can replace the synthetic data loader with your KDD dataset, swap the MLP autoencoder for a deep autoencoder (TensorFlow/PyTorch), or add a production API and CI pipeline later. If you want, I can now produce:

* a Dockerfile to containerize the app, or
* a GitHub Actions workflow that trains and validates models on push.

Which would you like next?
