import os
import sys
from pathlib import Path

# Streamlit puts app/ on sys.path, not the project root, so src would not import.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import numpy as np
from src.inference.predict import predict_from_file
from src.training.train import train
from src.utils.io import resolve_path


st.set_page_config(page_title="Plant Disease Detection", layout="centered")
st.title("Multi-Spectral Plant Disease Detection")

st.write("Upload an RGB image or a 4-band .npy file (RGB+NIR).")
file = st.file_uploader("Image or .npy", type=["png", "jpg", "jpeg", "npy"])

if file is not None:
    is_npy = file.name.lower().endswith(".npy")
    if is_npy:
        arr = np.load(file)
        st.write(f"Loaded array shape: {arr.shape}")
        file.seek(0)
    else:
        st.image(file)
        file.seek(0)

    model_path = resolve_path("data/processed/model.pt")
    if not os.path.exists(model_path):
        with st.spinner("No model found. Training one now, this can take a while..."):
            train()

    try:
        pred, class_name = predict_from_file(file, is_npy=is_npy)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    if class_name:
        st.success(f"Prediction class: {pred} ({class_name})")
    else:
        st.success(f"Prediction class: {pred}")

    st.caption(f"Model path: {model_path}")
