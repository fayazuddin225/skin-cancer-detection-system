# streamlit_app.py
"""Streamlit version of the skin‑cancer detection app.
It loads the same models (outlier_model.pkl, hybrid_model.pkl, InceptionV3) and provides a simple UI for uploading an image and showing the prediction.
"""

import streamlit as st
import numpy as np
import joblib
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

# ---------------------------------------------------
# Custom CSS for a premium look (dark mode, gradient, glassmorphism)
# ---------------------------------------------------
def _add_custom_css():
    st.markdown(
        """
        <style>
        /* Gradient background */
        .stApp {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            color: #f0f0f0;
        }
        /* Glassmorphism card */
        .card {
            background: rgba(255, 255, 255, 0.12);
            border-radius: 12px;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.18);
            padding: 1.5rem;
            margin-top: 1rem;
        }
        /* Buttons */
        .stButton > button {
            background: linear-gradient(90deg, #ff7e5f, #feb47b);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1.2rem;
            font-weight: 600;
            transition: transform 0.2s;
        }
        .stButton > button:hover {
            transform: scale(1.05);
        }
        /* File uploader */
        .stFileUploader > div > div > input {
            color: #fff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Call the CSS injection once
_add_custom_css()

from tensorflow.keras.preprocessing.image import img_to_array

# -------------------------------------------------------------------
# Load models (executed once when the script runs)
# -------------------------------------------------------------------
@st.cache_resource
def load_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    outlier_path = os.path.join(base_dir, "outlier_model.pkl")
    hybrid_path = os.path.join(base_dir, "hybrid_model.pkl")
    # Load outlier and hybrid models
    outlier_model = joblib.load(outlier_path) if os.path.exists(outlier_path) else None
    hybrid_model = joblib.load(hybrid_path) if os.path.exists(hybrid_path) else None
    # Load InceptionV3 (pre‑trained on ImageNet, without top layers)
    inception_model = InceptionV3(weights="imagenet", include_top=False, pooling="avg")
    return outlier_model, hybrid_model, inception_model

outlier_model, hybrid_model, inception_model = load_models()

# -------------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------------
st.set_page_config(page_title="Skin Cancer Detection AI", layout="centered")
st.title("Skin Cancer Detection AI")
st.markdown("Advanced AI‑powered analysis for early detection of **Benign** and **Malignant** skin lesions.")

uploaded_file = st.file_uploader("Upload an image (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show preview
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # -------------------------------------------------------------------
    # Preprocess for outlier detection (64x64)
    # -------------------------------------------------------------------
    img_outlier = image.resize((64, 64)).convert("RGB")
    img_array_outlier = np.array(img_outlier).flatten().reshape(1, -1) / 255.0

    # -------------------------------------------------------------------
    # Outlier check
    # -------------------------------------------------------------------
    if outlier_model is not None:
        is_inlier = outlier_model.predict(img_array_outlier)[0]
        if is_inlier == -1:
            st.error("Invalid image detected. Please upload a skin lesion image.")
            st.stop()
    else:
        st.warning("Outlier model not loaded – skipping outlier check.")

    # -------------------------------------------------------------------
    # Hybrid model prediction (InceptionV3 features + SVM)
    # -------------------------------------------------------------------
    if hybrid_model is not None and inception_model is not None:
        img_inception = image.resize((299, 299)).convert("RGB")
        x = img_to_array(img_inception)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = inception_model.predict(x, verbose=0).flatten().reshape(1, -1)
        pred = hybrid_model.predict(features)[0]
        prob = hybrid_model.predict_proba(features)[0]
        confidence = float(max(prob)) * 100
        label = "Malignant" if pred == 1 else "Benign"
        st.markdown(
    f"""
    <div class='card'>
        <h3>Result: {label}</h3>
        <p>Confidence: {confidence:.2f}%</p>
    </div>
    """,
    unsafe_allow_html=True,
)
    else:
        st.error("Models not loaded – cannot perform prediction.")
else:
    st.info("Please upload an image to get a prediction.")

st.markdown("---")
st.caption("Note: This AI tool is for educational purposes only and should not replace professional medical advice.")
