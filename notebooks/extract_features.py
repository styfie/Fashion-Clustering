import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models

# =========================
# CONFIG
# =========================
CSV_PATH = "data/sampled_balanced_10k.csv"
OUTPUT_PATH = "embeddings/image_embeddings.npy"

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 0        # Windows-safe
RANDOM_STATE = 42

# =========================
# DEVICE
# =========================
device = torch.device("cpu")  # keep CPU for stability
print("Using device:", device)

# =========================
# LOAD METADATA
# =========================
df = pd.read_csv(CSV_PATH)
image_paths = df["image_path"].tolist()

print(f"Images to process: {len(image_paths)}")

# =========================
# DATASET
# =========================
class FashionDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
        except Exception:
            # fallback for corrupted images
            image = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)

        return image

# =========================
# TRANSFORMS
# =========================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

dataset = FashionDataset(image_paths, transform)
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

# =========================
# LOAD RESNET50
# =========================
model = models.resnet50(pretrained=True)
model.fc = torch.nn.Identity()  # remove classifier
model.eval()
model.to(device)

# =========================
# FEATURE EXTRACTION
# =========================
all_embeddings = []

with torch.no_grad():
    for batch in tqdm(dataloader, desc="Extracting embeddings"):
        batch = batch.to(device)
        features = model(batch)          # (B, 2048)
        all_embeddings.append(features.cpu().numpy())

embeddings = np.vstack(all_embeddings)

# =========================
# SAVE
# =========================
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
np.save(OUTPUT_PATH, embeddings)

print("Embeddings saved to:", OUTPUT_PATH)
print("Embeddings shape:", embeddings.shape)
