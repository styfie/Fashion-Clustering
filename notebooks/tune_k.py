import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# =========================
# CONFIG
# =========================
EMBEDDINGS_PATH = "embeddings/image_embeddings.npy"

K_RANGE = range(5, 21)  
PCA_COMPONENTS = 128
RANDOM_STATE = 42

# =========================
# LOAD EMBEDDINGS
# =========================
embeddings = np.load(EMBEDDINGS_PATH)
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

print(
    f"PCA variance retained: "
    f"{np.sum(pca.explained_variance_ratio_):.2%}"
)

# =========================
# TUNE K
# =========================
results = []

for k in K_RANGE:
    print(f"Testing K = {k}")

    kmeans = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=10
    )

    labels = kmeans.fit_predict(embeddings_pca)

    sil = silhouette_score(embeddings_pca, labels)
    inertia = kmeans.inertia_

    results.append({
        "k": k,
        "silhouette": sil,
        "inertia": inertia
    })

# =========================
# RESULTS
# =========================
results_df = pd.DataFrame(results)
print("\nK tuning results:")
print(results_df)

best_k = results_df.loc[
    results_df["silhouette"].idxmax(), "k"
]

print(f"\nBest K by silhouette score: {best_k}")
