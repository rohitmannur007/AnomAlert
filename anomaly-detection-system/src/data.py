import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

def ensure_dirs(cfg):
    os.makedirs(cfg['data']['raw_path'], exist_ok=True)
    os.makedirs(cfg['data']['processed_path'], exist_ok=True)
    os.makedirs(cfg['models']['path'], exist_ok=True)

def load_raw(cfg):
    """
    Load KDDTrain+ / KDDTest+ if present. If not present, generate synthetic dataset.
    Returns concatenated dataframe (train+test).
    """
    raw_dir = cfg['data']['raw_path']
    train_file = os.path.join(raw_dir, "KDDTrain+.csv")
    test_file = os.path.join(raw_dir, "KDDTest+.csv")

    if os.path.exists(train_file) and os.path.exists(test_file):
        train = pd.read_csv(train_file, header=None)
        test = pd.read_csv(test_file, header=None)
        df = pd.concat([train, test], ignore_index=True)
        # if real KDD file, user should adjust columns externally
    else:
        # Generate synthetic dataset similar to network features
        n = 2000
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            'duration': rng.integers(0, 1000, size=n),
            'protocol_type': rng.choice(['tcp','udp','icmp'], size=n),
            'service': rng.choice(['http','ftp','ssh','dns','smtp'], size=n),
            'flag': rng.choice(['SF','S0','REJ','RSTR'], size=n),
            'src_bytes': rng.integers(0, 100000, size=n),
            'dst_bytes': rng.integers(0, 100000, size=n),
            'wrong_fragment': rng.integers(0,5,size=n),
            'urgent': rng.integers(0,3,size=n),
            # simplified label: ~95% normal, 5% anomaly
            'label': np.where(rng.random(n) < 0.95, 'normal', 'anomaly')
        })
    return df

def preprocess(df, cfg, save_encoders=True):
    """
    - encode categorical (LabelEncoder saved to models/)
    - scale numeric features (StandardScaler saved to models/)
    - create timestamp index for time-series
    """
    model_dir = cfg['models']['path']
    df = df.copy()

    # Ensure label present
    if 'label' not in df.columns:
        df['label'] = 'normal'

    # categorical
    cat_cols = ['protocol_type', 'service', 'flag']
    encoders = {}
    for c in cat_cols:
        if c in df.columns:
            le = LabelEncoder()
            df[c] = le.fit_transform(df[c].astype(str))
            encoders[c] = le

    # numeric columns: pick numeric dtype columns only
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # drop label if numeric by chance
    num_cols = [c for c in num_cols if c not in ['label']]
    scaler = StandardScaler()
    if len(num_cols) > 0:
        df[num_cols] = scaler.fit_transform(df[num_cols])

    # binary label: 0 normal, 1 anomaly
    df['label_binary'] = df['label'].apply(lambda x: 0 if str(x).lower() == 'normal' else 1)

    # timestamp index
    df['timestamp'] = pd.date_range(start='2023-01-01', periods=len(df), freq='T')
    df.set_index('timestamp', inplace=True)

    # save encoders and scaler
    if save_encoders:
        joblib.dump(encoders, os.path.join(model_dir, 'encoders.joblib'))
        joblib.dump(scaler, os.path.join(model_dir, 'scaler.joblib'))

    return df, encoders, scaler
