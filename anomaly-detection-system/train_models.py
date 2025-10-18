import os
import yaml
from src.data import load_raw, preprocess, ensure_dirs
from src.models import train_isolation_forest, train_autoencoder
import joblib
import pandas as pd

def main():
    # load config
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    ensure_dirs(cfg)

    df = load_raw(cfg)
    df_proc, encoders, scaler = preprocess(df, cfg)

    # features for training: drop labels and non-numeric (we already encoded)
    X = df_proc.select_dtypes(include=['number']).drop(columns=['label_binary'], errors='ignore').fillna(0).values

    # train models
    print("Training Isolation Forest...")
    iso = train_isolation_forest(X, model_dir=cfg['models']['path'])
    print("Training Autoencoder (MLP)...")
    ae = train_autoencoder(X, model_dir=cfg['models']['path'])

    print("Saving preprocessing artifacts...")
    # encoders and scaler already saved in data.preprocess
    print("Done.")

if __name__ == "__main__":
    main()
