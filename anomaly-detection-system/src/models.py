import os
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import roc_auc_score, mean_squared_error

MODEL_DIR_DEFAULT = "./models"

def train_isolation_forest(X, model_dir=MODEL_DIR_DEFAULT, **kwargs):
    iso = IsolationForest(n_estimators=200, contamination='auto', random_state=42, **kwargs)
    iso.fit(X)
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(iso, os.path.join(model_dir, "isolation_forest.joblib"))
    return iso

def train_autoencoder(X, model_dir=MODEL_DIR_DEFAULT, hidden_layers=(64,32,64), max_iter=300):
    """
    Trains an MLPRegressor to reconstruct input features (simple autoencoder analog).
    """
    ae = MLPRegressor(hidden_layer_sizes=hidden_layers, max_iter=max_iter, random_state=42)
    # Fit X -> X
    ae.fit(X, X)
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(ae, os.path.join(model_dir, "autoencoder.joblib"))
    return ae

def load_models(model_dir=MODEL_DIR_DEFAULT):
    iso_path = os.path.join(model_dir, "isolation_forest.joblib")
    ae_path = os.path.join(model_dir, "autoencoder.joblib")
    models = {}
    if os.path.exists(iso_path):
        models['iso'] = joblib.load(iso_path)
    else:
        models['iso'] = None
    if os.path.exists(ae_path):
        models['ae'] = joblib.load(ae_path)
    else:
        models['ae'] = None
    return models

def anomaly_score_autoencoder(ae_model, X):
    """
    Reconstruction error per sample
    """
    X_pred = ae_model.predict(X)
    mse = np.mean((X - X_pred)**2, axis=1)
    return mse

def anomaly_score_isoforest(iso_model, X):
    # IsolationForest.decision_function -> higher = more normal. We invert to get anomaly score.
    scores = -iso_model.decision_function(X)  # larger = more anomalous
    return scores
