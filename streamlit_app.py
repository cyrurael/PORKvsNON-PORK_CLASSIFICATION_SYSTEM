# ============================================================
# 🥩 PORK vs NON-PORK CLASSIFIER — PREDICTION ONLY
# Model: pork_classifier.keras
# Binary Classes: PORK | NON-PORK
# ============================================================

import os
import time
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# -------------------------- CONFIGURATION --------------------------

MODEL_PATH = "pork_classifier.keras"
IMAGE_SIZE = (224, 224)
THRESHOLD = 0.50  # 0.5 = balanced threshold

# -------------------------- PAGE SETUP --------------------------

st.set_page_config(
    page_title="Pork vs Non-Pork Classifier",
    page_icon="🥩",
    layout="wide"
)

# -------------------------- CUSTOM STYLES --------------------------

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
.main-title {font-size: 2.2rem; font-weight: 800; text-align:center; color:#1a1a1a;}
.sub-title {text-align:center; color:#666; margin-bottom:2rem; font-size:1.1rem;}
.result-pork {
    background: linear-gradient(135deg, #e5f5e9, #c8e6d0);
    color:#146c35; padding:1.5rem; border-radius:12px;
    text-align:center; font-size:2rem; font-weight:800;
    border:2px solid #a8d5b5;
}
.result-nonpork {
    background: linear-gradient(135deg, #e6f1ff, #cfe2ff);
    color:#084298; padding:1.5rem; border-radius:12px;
    text-align:center; font-size:2rem; font-weight:800;
    border:2px solid #b6d4fe;
}
</style>
""", unsafe_allow_html=True)

# -------------------------- LOAD MODEL --------------------------

@st.cache_resource(show_spinner="🔄 Loading model... Please wait.")
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model

# -------------------------- PREDICTION FUNCTION --------------------------

def predict_image(model, pil_image):
    start_time = time.time()
    
    # Preprocess image
    image = pil_image.convert("RGB").resize(IMAGE_SIZE)
    img_array = tf.keras.utils.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    non_pork_prob = float(model.predict(img_array, verbose=0)[0][0])
    pork_prob = 1.0 - non_pork_prob
    
    if non_pork_prob >= THRESHOLD:
        label = "NON-PORK"
        confidence = non_pork_prob
    else:
        label = "PORK"
        confidence = pork_prob
    
    inference_time = (time.time() - start_time) * 1000
    
    return {
        "label": label,
        "confidence": confidence,
        "pork_prob": pork_prob,
        "non_pork_prob": non_pork_prob,
        "inference_ms": inference_time
    }

# -------------------------- MAIN UI --------------------------

st.markdown('<div class="main-title">🥩 PORK vs NON-PORK CLASSIFICATION</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Upload an image and the model will classify it as PORK or NON-PORK</div>', unsafe_allow_html=True)

# Load model
model = load_model()

if model is None:
    st.error(f"❌ Model file not found!")
    st.warning(f"Please make sure `{MODEL_PATH}` is in the same folder as your `app.py` file.")
    st.stop()
else:
    st.success(f"✅ Model loaded successfully from `{MODEL_PATH}`")

# -------------------------- UPLOAD & PREDICT --------------------------

st.subheader("🔍 Upload an Image")
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png", "bmp"],
    key="image_upload"
)

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="📷 Uploaded Image", use_column_width=True)
    
    with col2:
        if st.button("🔍 Classify Image", type="primary", use_container_width=True):
            with st.spinner("Analyzing..."):
                result = predict_image(model, image)
            
            # Show result
            if result["label"] == "PORK":
                st.markdown(
                    '<div class="result-pork">🐷 PORK DETECTED</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="result-nonpork">🥩 NON-PORK</div>',
                    unsafe_allow_html=True
                )
            
            st.metric("Confidence", f"{result['confidence'] * 100:.2f}%")
            st.progress(float(result["confidence"]))
            
            st.subheader("📊 Probabilities")
            st.info(f"🐷 **PORK:** {result['pork_prob'] * 100:.2f}%")
            st.info(f"🥩 **NON-PORK:** {result['non_pork_prob'] * 100:.2f}%")
            st.caption(f"⚡ Inference time: {result['inference_ms']:.1f} ms")

# -------------------------- FOOTER --------------------------

st.divider()
st.caption("Model: MobileNetV2 · Binary Classification · Built with Streamlit & TensorFlow")
