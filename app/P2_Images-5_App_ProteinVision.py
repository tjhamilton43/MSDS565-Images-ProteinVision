"""MSDS 565 Project 2 — Protein Source Recognition.

Run from repository root:
    streamlit run app/P2_Images-5_App_ProteinVision.py

The professor-provided image_explainers.py must remain in this app/ folder.
Notebook 3 must first have generated models/P2_N3_*.keras and label_mapping.json.
"""
from pathlib import Path
import json

import keras
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError
import streamlit as st
import image_explainers as xai

REPO = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO / "models"
TARGET_SIZE = (224, 224)
MODELS = {
    "MobileNetV2 (small)": ("mobilenetv2", keras.applications.MobileNetV2),
    "ResNet50 (medium)": ("resnet50", keras.applications.ResNet50),
    "EfficientNetV2M (large)": ("efficientnetv2m", keras.applications.EfficientNetV2M),
}

st.set_page_config(page_title="Protein Source Recognition", page_icon="🥗", layout="wide")
st.title("Protein Source Recognition")
st.caption("MSDS 565 · Five protein classes · Three frozen ImageNet feature extractors + trained linear probes")


def prep_uploaded_image(original: Image.Image) -> np.ndarray:
    """Match Notebook 2: RGB, ImageOps.pad, bilinear, black, uint8 0–255."""
    rgb = original.convert("RGB")
    padded = ImageOps.pad(
        rgb, TARGET_SIZE, method=Image.Resampling.BILINEAR, color=(0, 0, 0)
    )
    return np.asarray(padded, dtype=np.uint8)


@st.cache_resource(show_spinner="Loading trained models…")
def load_models():
    mapping_file = MODELS_DIR / "label_mapping.json"
    if not mapping_file.exists():
        raise FileNotFoundError("Missing models/label_mapping.json — run Notebook 3 first.")
    mapping = json.loads(mapping_file.read_text())
    class_names = [name for name, idx in sorted(mapping.items(), key=lambda pair: pair[1])]
    loaded = {}
    for display_name, (key, constructor) in MODELS.items():
        probe_path = MODELS_DIR / f"P2_N3_{key}.keras"
        if not probe_path.exists():
            raise FileNotFoundError(f"Missing {probe_path.name} — run Notebook 3 first.")
        backbone = constructor(
            weights="imagenet", include_top=False, pooling="avg", input_shape=(*TARGET_SIZE, 3)
        )
        backbone.trainable = False
        probe = keras.models.load_model(probe_path, compile=False)
        loaded[display_name] = keras.Sequential([
            keras.Input((*TARGET_SIZE, 3)), backbone, probe
        ])
    return class_names, loaded


st.write(
    "Upload a photo and compare how the three pretrained models classify its main protein source. "
    "The explanations help show what each one focused on."
)
image_file = st.file_uploader("Upload a food photograph", type=["jpg", "jpeg", "png", "webp"])

if image_file is None:
    st.info("Upload an image to see all three predictions and explanation maps.")
    st.stop()

try:
    image = Image.open(image_file)
    image.load()
    processed = prep_uploaded_image(image)
except (OSError, ValueError, UnidentifiedImageError) as exc:
    st.error(f"Could not read this image: {exc}")
    st.stop()

st.image(image, caption="Uploaded photograph", width=340)
st.caption("Preprocessing: RGB → 224 × 224 with black aspect-ratio padding; raw 0–255 pixels.")

try:
    class_names, trained = load_models()
except Exception as exc:
    st.error(str(exc))
    st.info("Run Notebook 3 first and keep the saved .keras probes in the models/ folder.")
    st.stop()

st.warning(
    "This is a five-class demo, not a calorie estimator. It always assigns one of the "
    "known protein classes, even if the uploaded photo contains a different food. "
    "Probabilities are not a reliable out-of-distribution detector."
)

for display_name, model in trained.items():
    st.divider()
    st.subheader(display_name)
    try:
        # The professor's helper handles this backbone's preprocessing internally.
        # Do NOT preprocess twice.
        probabilities = xai.predict_proba(model, processed)
        top_idx = int(np.argmax(probabilities))
        st.write(f"**Prediction: {class_names[top_idx].title()}** ({probabilities[top_idx]:.1%})")
        ordered = pd.DataFrame({
            "Protein": class_names,
            "Probability": np.asarray(probabilities, dtype=float),
        }).sort_values("Probability", ascending=False)
        st.bar_chart(ordered.set_index("Protein"), horizontal=True, height=200)

        # Three different explanation methods for the predicted class.
        raw = xai.saliency(model, processed, class_index=top_idx)
        smooth = xai.saliency(model, processed, smooth=32, class_index=top_idx)
        cam = xai.grad_cam(model, processed, class_index=top_idx)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(xai.overlay(processed, raw), caption="Raw saliency", use_container_width=True)
        with col2:
            st.image(xai.overlay(processed, smooth), caption="SmoothGrad (32)", use_container_width=True)
        with col3:
            st.image(xai.overlay(processed, cam), caption="Grad-CAM", use_container_width=True)
    except Exception as exc:
        st.error(f"Couldn't process {display_name}: {exc}")

st.divider()
st.caption(
    "Live demo ideas: test one clear protein photo, one tricky look-alike (e.g., chicken vs tofu), "
    "and one food outside the five categories. Compare confidence and explanation maps."
)
