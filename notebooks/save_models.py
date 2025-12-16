import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# CONFIG
EMBEDDINGS_PATH = "embeddings/image_embeddings.npy"
CSV_PATH = "data/sampled_balanced_10k.csv"
MODEL_DIR = "models"

PCA_COMPONENTS = 128
N_CLUSTERS = 17
RANDOM_STATE = 42

# LOAD
X = np.load(EMBEDDINGS_PATH)

# FIT
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)

kmeans = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=RANDOM_STATE,
    n_init=10
)
kmeans.fit(X_pca)

# SAVE
import os
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
joblib.dump(pca, f"{MODEL_DIR}/pca.pkl")
joblib.dump(kmeans, f"{MODEL_DIR}/kmeans.pkl")

print("Models saved.")
