import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_preprocessing_tools(model_dir="./models"):
    encoders_path = os.path.join(model_dir, "encoders.joblib")
    scaler_path = os.path.join(model_dir, "scaler.joblib")
    encoders = joblib.load(encoders_path) if os.path.exists(encoders_path) else {}
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    return encoders, scaler

def prepare_sample(sample_df, encoders, scaler):
    df = sample_df.copy()
    # apply encoders
    for c, le in (encoders.items() if encoders else []):
        if c in df.columns:
            df[c] = le.transform(df[c].astype(str))
    # numeric scaling
    if scaler is not None:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        df[num_cols] = scaler.transform(df[num_cols])
    return df.values

def detect_anomaly(sample_df, models, encoders, scaler):
    """
    sample_df: single-row DataFrame of features
    models: {'iso': ..., 'ae': ...}
    Returns combined score + per-model scores.
    """
    X = prepare_sample(sample_df, encoders, scaler)
    results = {}
    if models.get('iso') is not None:
        iso_score = -models['iso'].decision_function(X)  # larger = more anomalous
        results['iso_score'] = float(iso_score[0])
    else:
        results['iso_score'] = None

    if models.get('ae') is not None:
        X_pred = models['ae'].predict(X)
        mse = np.mean((X - X_pred)**2, axis=1)
        results['ae_score'] = float(mse[0])
    else:
        results['ae_score'] = None

    # Combined score: normalized sum (simple)
    scores = []
    for k in ['iso_score', 'ae_score']:
        if results[k] is not None:
            scores.append(results[k])
    if len(scores) == 0:
        results['combined'] = None
    else:
        results['combined'] = float(np.mean(scores))
    return results
