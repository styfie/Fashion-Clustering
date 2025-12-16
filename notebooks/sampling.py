import os
import pandas as pd
import numpy as np

# =========================
# CONFIG
# =========================
CSV_PATH = "data/styles.csv"
IMAGE_DIR = "data/images"
OUTPUT_CSV = "data/sampled_balanced_10k.csv"

TARGET_SIZE = 10_000
MIN_SAMPLES_PER_CATEGORY = 200
RANDOM_STATE = 42

# =========================
# LOAD METADATA (ROBUST)
# =========================
df = pd.read_csv(
    CSV_PATH,
    on_bad_lines="skip",
    encoding="utf-8"
)

# =========================
# BUILD IMAGE PATHS
# =========================
df["image_path"] = df["id"].astype(str) + ".jpg"
df["image_path"] = df["image_path"].apply(
    lambda x: os.path.join(IMAGE_DIR, x)
)

# =========================
# FILTER VALID IMAGES
# =========================
df = df[df["image_path"].apply(os.path.exists)]
print(f"Valid images: {len(df)}")

# =========================
# OPTIONAL: APPAREL ONLY
# =========================
df = df[df["masterCategory"] == "Apparel"]

# =========================
# DATASET OVERVIEW
# =========================
print(f"Article types: {df['articleType'].nunique()}")
print(df["articleType"].value_counts().head())

# =========================
# FILTER RARE CATEGORIES
# =========================
article_counts = df["articleType"].value_counts()
valid_types = article_counts[
    article_counts >= MIN_SAMPLES_PER_CATEGORY
].index

df_filtered = df[df["articleType"].isin(valid_types)]

num_categories = df_filtered["articleType"].nunique()
min_category_size = df_filtered["articleType"].value_counts().min()

print(f"Categories kept: {num_categories}")
print(f"Smallest category size: {min_category_size}")

# =========================
# COMPUTE SAFE SAMPLE SIZE
# =========================
samples_per_category = min(
    TARGET_SIZE // num_categories,
    min_category_size
)

print(f"Samples per category: {samples_per_category}")

# =========================
# BALANCED SAMPLING
# =========================
df_balanced = (
    df_filtered
    .groupby("articleType", group_keys=False)
    .apply(lambda x: x.sample(
        n=samples_per_category,
        random_state=RANDOM_STATE
    ))
)

# =========================
# TOP-UP TO EXACT 10K
# =========================
current_size = len(df_balanced)
remaining = TARGET_SIZE - current_size

if remaining > 0:
    extra = (
        df_filtered
        .drop(df_balanced.index)
        .sample(n=remaining, random_state=RANDOM_STATE)
    )
    df_balanced = pd.concat([df_balanced, extra])

# =========================
# FINAL CHECKS
# =========================
print(f"Final sample size: {len(df_balanced)}")
print("\nFinal distribution (top 10):")
print(df_balanced["articleType"].value_counts().head(10))

# =========================
# SAVE RESULT
# =========================
os.makedirs("data", exist_ok=True)
df_balanced.to_csv(OUTPUT_CSV, index=False)

print(f"\nSaved sampled dataset to: {OUTPUT_CSV}")
