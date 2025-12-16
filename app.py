import streamlit as st
import numpy as np
import pandas as pd
import joblib
from PIL import Image

import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights

# =========================
# CONFIG
# =========================
CSV_PATH = "data/clustered_fashion_demo.csv"
MODEL_DIR = "models"
IMAGE_SIZE = 224
TOP_K = 12

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(CSV_PATH)

# =========================
# LOAD MODELS
# =========================
scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
pca = joblib.load(f"{MODEL_DIR}/pca.pkl")
kmeans = joblib.load(f"{MODEL_DIR}/kmeans.pkl")

# =========================
# LOAD CNN
# =========================
device = torch.device("cpu")

cnn = resnet50(weights=ResNet50_Weights.DEFAULT)
cnn.fc = torch.nn.Identity()
cnn.eval()
cnn.to(device)

# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(
    page_title="FaSHioN — Editorial Visual Discovery",
    layout="wide"
)

# ---------- Premium Editorial CSS ----------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #fdfdfc;
    }

    /* Typography */
    h1, h2, h3 {
        font-family: 'Playfair Display', 'Georgia', serif;
        letter-spacing: 1.5px;
    }

    p, span, div {
        font-family: 'Helvetica Neue', 'Arial', sans-serif;
    }

    /* Hero title */
    .hero-title {
        font-size: 56px;
        font-weight: 500;
        text-align: center;
        margin-top: 40px;
        margin-bottom: 10px;
    }

    .hero-subtitle {
        text-align: center;
        color: #7a7a7a;
        font-size: 15px;
        letter-spacing: 1px;
        margin-bottom: 60px;
    }

    /* Section labels */
    .section-label {
        text-transform: uppercase;
        letter-spacing: 2px;
        font-size: 12px;
        color: #9a9a9a;
        margin-bottom: 10px;
    }

    /* Image styling */
    img {
        border-radius: 6px;
    }

    /* Remove Streamlit chrome */
    footer, header {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- Hero ----------
st.markdown(
    "<div class='hero-title'>FaSHioN</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='hero-subtitle'>EDITORIAL VISUAL DISCOVERY OF FASHION</div>",
    unsafe_allow_html=True
)

# ---------- Upload Section ----------
st.markdown("<div class='section-label'>Upload</div>", unsafe_allow_html=True)
st.markdown(
    "<span style='color:#6f6f6f;'>Select a fashion item to explore visually similar pieces curated by form and texture.</span>",
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload fashion image",
    type=["jpg", "png", "jpeg"],
    label_visibility="collapsed"
)

if uploaded_file:
    st.write("")
    st.write("")

    # ---------- Layout ----------
    left_col, right_col = st.columns([1, 2.3], gap="large")

    # =========================
    # LEFT — FEATURE ITEM
    # =========================
    with left_col:
        st.markdown("<div class='section-label'>Selected Item</div>", unsafe_allow_html=True)
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, width=340)

    # =========================
    # FEATURE EXTRACTION
    # =========================
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = cnn(img_tensor).cpu().numpy()

    embedding_scaled = scaler.transform(embedding)
    embedding_pca = pca.transform(embedding_scaled)
    cluster = kmeans.predict(embedding_pca)[0]

    # =========================
    # RIGHT — CURATED RESULTS
    # =========================
    with right_col:
        st.markdown("<div class='section-label'>Curated Selection</div>", unsafe_allow_html=True)
        st.markdown(
            f"<span style='color:#7a7a7a;'>Visually aligned pieces · Editorial Cluster {cluster}</span>",
            unsafe_allow_html=True
        )

        st.write("")
        st.write("")

        cluster_df = df[df["cluster"] == cluster].sample(TOP_K)

        cols = st.columns(4)
        for i, (_, row) in enumerate(cluster_df.iterrows()):
            img = Image.open(row["image_path"]).convert("RGB")
            cols[i % 4].image(img, width=170)

    st.write("")
    st.write("")

    # ---------- Editorial Footer ----------
    st.markdown(
        "<div style='text-align:center; color:#9a9a9a; font-size:12px; letter-spacing:1px;'>"
        "An editorial exploration of fashion through visual intelligence"
        "</div>",
        unsafe_allow_html=True
    )
