import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# =========================
# CONFIG
# =========================
EMBEDDINGS_PATH = "embeddings/image_embeddings.npy"
CSV_PATH = "data/sampled_balanced_10k.csv"
OUTPUT_CSV = "data/clustered_fashion_10k.csv"

PCA_COMPONENTS = 128     # strong tradeoff: speed vs information
N_CLUSTERS = 17          # based on hypertuning k
RANDOM_STATE = 42

# =========================
# LOAD DATA
# =========================
embeddings = np.load(EMBEDDINGS_PATH)
df = pd.read_csv(CSV_PATH)

assert len(embeddings) == len(df), "Embeddings and CSV row count mismatch"

print("Embeddings shape:", embeddings.shape)

# =========================
# STANDARDIZE
# =========================
scaler = StandardScaler()
embeddings_scaled = scaler.fit_transform(embeddings)

# =========================
# PCA
# =========================
pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)
embeddings_pca = pca.fit_transform(embeddings_scaled)

explained_var = np.sum(pca.explained_variance_ratio_)
print(f"PCA explained variance: {explained_var:.2%}")

# =========================
# K-MEANS
# =========================
kmeans = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=RANDOM_STATE,
    n_init=10
)

cluster_labels = kmeans.fit_predict(embeddings_pca)

df["cluster"] = cluster_labels

# =========================
# EVALUATION
# =========================
sil_score = silhouette_score(embeddings_pca, cluster_labels)
print(f"Silhouette score: {sil_score:.4f}")

# =========================
# SAVE
# =========================
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False)

print("Clustered CSV saved to:", OUTPUT_CSV)
print("Cluster distribution:")
print(df["cluster"].value_counts().sort_index())
