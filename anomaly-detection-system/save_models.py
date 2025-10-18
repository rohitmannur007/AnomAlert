#!/usr/bin/env python3
import os
import joblib
from pathlib import Path
from train_models import main as train_all

# If models exist, skip training
MODEL_DIR = Path("./models")
iso_path = MODEL_DIR / "isolation_forest.joblib"
ae_path = MODEL_DIR / "autoencoder.joblib"

if not (iso_path.exists() and ae_path.exists()):
    print("Models not found. Training now...")
    train_all()
else:
    print("Models already exist. Skipping training.")
