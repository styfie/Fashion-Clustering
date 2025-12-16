import os
import random
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# =========================
# CONFIG
# =========================
CSV_PATH = "data/clustered_fashion_10k.csv"
OUTPUT_DIR = "outputs/cluster_grids"

IMAGES_PER_CLUSTER = 16     # 4x4 grid
GRID_SIZE = 4
IMAGE_SIZE = 224
RANDOM_STATE = 42

random.seed(RANDOM_STATE)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(CSV_PATH)

assert "cluster" in df.columns, "Cluster column not found"

clusters = sorted(df["cluster"].unique())
print(f"Total clusters: {len(clusters)}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# VISUALIZE EACH CLUSTER
# =========================
for cluster_id in clusters:
    cluster_df = df[df["cluster"] == cluster_id]

    if len(cluster_df) < IMAGES_PER_CLUSTER:
        print(f"Skipping cluster {cluster_id} (only {len(cluster_df)} images)")
        continue

    sampled = cluster_df.sample(
        n=IMAGES_PER_CLUSTER,
        random_state=RANDOM_STATE
    )

    fig, axes = plt.subplots(GRID_SIZE, GRID_SIZE, figsize=(8, 8))
    fig.suptitle(
        f"Cluster {cluster_id} | {len(cluster_df)} images",
        fontsize=14
    )

    for ax, (_, row) in zip(axes.flatten(), sampled.iterrows()):
        try:
            img = Image.open(row["image_path"]).convert("RGB")
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
            ax.imshow(img)
            ax.axis("off")
        except Exception:
            ax.axis("off")

    # Hide any unused axes
    for ax in axes.flatten()[len(sampled):]:
        ax.axis("off")

    output_path = os.path.join(OUTPUT_DIR, f"cluster_{cluster_id}.png")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Saved: {output_path}")
