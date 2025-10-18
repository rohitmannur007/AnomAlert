import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

def reduce_dimensionality(X, n_components=20):
    pca = PCA(n_components=min(n_components, X.shape[1]))
    Xp = pca.fit_transform(X)
    return Xp, pca

def run_dbscan(X, eps=0.5, min_samples=5):
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)
    # label -1 is noise (potential anomalies)
    return labels, db
