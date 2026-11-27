# ============================================================
# 🥩 PORK vs NON-PORK CLASSIFIER — WITH EVALUATION
# Model: pork_classifier.keras
# Binary Classes: PORK | NON-PORK
# ============================================================

import os
import time
import numpy as np
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    ConfusionMatrixDisplay
)

# -------------------------- CONFIGURATION --------------------------

MODEL_PATH = "pork_classifier.keras"
IMAGE_SIZE = (224, 224)
THRESHOLD = 0.50
TEST_IMAGES_FOLDER = "test_images"

# -------------------------- PAGE SETUP --------------------------

st.set_page_config(
    page_title="Pork vs Non-Pork Classifier",
    page_icon="🥩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------- CUSTOM STYLES --------------------------

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
.main-title {font-size: 2rem; font-weight: 800; text-align:center; color:#1a1a1a;}
.sub-title {text-align:center; color:#666; margin-bottom:1.5rem; font-size:1rem;}
.result-pork {
    background: linear-gradient(135deg, #e5f5e9, #c8e6d0);
    color:#146c35; padding:1.2rem; border-radius:10px;
    text-align:center; font-size:1.6rem; font-weight:700;
    border:2px solid #a8d5b5;
}
.result-nonpork {
    background: linear-gradient(135deg, #e6f1ff, #cfe2ff);
    color:#084298; padding:1.2rem; border-radius:10px;
    text-align:center; font-size:1.6rem; font-weight:700;
    border:2px solid #b6d4fe;
}
</style>
""", unsafe_allow_html=True)

# -------------------------- SIDEBAR NAVIGATION --------------------------

with st.sidebar:
    st.title("🥩 Navigation")
    page = st.radio("Go to", [
        "🔍 Image Prediction",
        "📈 Performance Overview",
        "📋 Model Evaluation"
    ])
    st.divider()
    st.info(f"Model: `{MODEL_PATH}`")
    st.info(f"Threshold: {THRESHOLD}")

# -------------------------- LOAD MODEL --------------------------

@st.cache_resource(show_spinner="🔄 Loading model... Please wait.")
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

if model is None:
    st.error(f"❌ Model not found! Place `{MODEL_PATH}` in your project folder.")
    st.stop()

# -------------------------- PREDICTION HELPER --------------------------

def predict_image(model, pil_image):
    start = time.time()
    image = pil_image.convert("RGB").resize(IMAGE_SIZE)
    arr = tf.keras.utils.img_to_array(image)
    arr = np.expand_dims(arr, axis=0)
    non_pork_prob = float(model.predict(arr, verbose=0)[0][0])
    pork_prob = 1.0 - non_pork_prob
    label = "NON-PORK" if non_pork_prob >= THRESHOLD else "PORK"
    confidence = non_pork_prob if label == "NON-PORK" else pork_prob
    inference_ms = (time.time() - start) * 1000
    return {
        "label": label, "confidence": confidence,
        "pork_prob": pork_prob, "non_pork_prob": non_pork_prob,
        "inference_ms": inference_ms
    }

# -------------------------- LOAD TEST DATA FOR EVALUATION --------------------------

def load_test_dataset():
    if not os.path.exists(TEST_IMAGES_FOLDER):
        return None, None, None
    
    images = []
    labels = []
    paths = []
    
    pork_dir = os.path.join(TEST_IMAGES_FOLDER, "PORK")
    nonpork_dir = os.path.join(TEST_IMAGES_FOLDER, "NON-PORK")
    
    if os.path.exists(pork_dir):
        for fname in os.listdir(pork_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                fpath = os.path.join(pork_dir, fname)
                try:
                    img = Image.open(fpath).convert("RGB").resize(IMAGE_SIZE)
                    arr = tf.keras.utils.img_to_array(img)
                    images.append(arr)
                    labels.append(0)
                    paths.append(fpath)
                except Exception:
                    pass
    
    if os.path.exists(nonpork_dir):
        for fname in os.listdir(nonpork_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                fpath = os.path.join(nonpork_dir, fname)
                try:
                    img = Image.open(fpath).convert("RGB").resize(IMAGE_SIZE)
                    arr = tf.keras.utils.img_to_array(img)
                    images.append(arr)
                    labels.append(1)
                    paths.append(fpath)
                except Exception:
                    pass
    
    if len(images) == 0:
        return None, None, None
    
    return np.array(images), np.array(labels), paths

# -------------------------- PAGE 1: IMAGE PREDICTION --------------------------

if page == "🔍 Image Prediction":
    st.markdown('<div class="main-title">🐷 PORK vs NON-PORK CLASSIFICATION</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Upload an image to classify it as PORK or NON-PORK</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "📷 Upload an image",
        type=["jpg", "jpeg", "png", "bmp"]
    )
    
    if uploaded_file:
        col1, col2 = st.columns([1, 1])
        with col1:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            if st.button("🔍 Classify", type="primary", use_container_width=True):
                with st.spinner("Analyzing..."):
                    result = predict_image(model, image)
                
                if result["label"] == "PORK":
                    st.markdown('<div class="result-pork">🐷 PORK</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="result-nonpork">🥩 NON-PORK</div>', unsafe_allow_html=True)
                
                st.metric("Confidence", f"{result['confidence']*100:.2f}%")
                st.progress(float(result["confidence"]))
                st.subheader("📊 Probabilities")
                st.write(f"🐷 PORK: **{result['pork_prob']*100:.2f}%**")
                st.write(f"🥩 NON-PORK: **{result['non_pork_prob']*100:.2f}%**")
                st.caption(f"⚡ Inference time: {result['inference_ms']:.1f} ms")

# -------------------------- PAGE 2: PERFORMANCE OVERVIEW --------------------------

elif page == "📈 Performance Overview":
    st.subheader("📊 Model Performance Overview")
    
    # You can update these values with your actual model metrics
    st.markdown("### 📈 Summary Metrics")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", "94.50%")
    m2.metric("Precision", "94.00%")
    m3.metric("Recall", "95.20%")
    m4.metric("F1-Score", "94.60%")
    m5.metric("AUC-ROC", "0.9720")
    
    st.markdown("### 📖 Metric Explanation")
    with st.expander("What do these metrics mean?"):
        st.write("**Accuracy:** Overall percentage of correct predictions.")
        st.write("**Precision:** Of all predicted NON-PORK, how many actually are NON-PORK?")
        st.write("**Recall:** Of all actual NON-PORK, how many were correctly identified?")
        st.write("**F1-Score:** Harmonic mean of Precision and Recall — balances both.")
        st.write("**AUC-ROC:** Model's ability to distinguish classes (1.0 = perfect).")

# -------------------------- PAGE 3: MODEL EVALUATION --------------------------

elif page == "📋 Model Evaluation":
    st.subheader("📊 Model Evaluation on Test Dataset")
    
    X_test, y_true, paths = load_test_dataset()
    
    if X_test is None:
        st.info("ℹ️ Add test images to `test_images/PORK/` and `test_images/NON-PORK/` to run evaluation.")
        
        # SAMPLE EVALUATION DISPLAY
        y_true_sample = [0, 0, 0, 0, 1, 1, 1, 1]
        y_pred_sample = [0, 0, 1, 0, 1, 1, 0, 1]
        
        cm = confusion_matrix(y_true_sample, y_pred_sample, labels=[0, 1])
        report = classification_report(
            y_true_sample, y_pred_sample,
            target_names=["PORK", "NON-PORK"], output_dict=True
        )
        
        st.markdown("### 📌 Sample Confusion Matrix")
        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["PORK", "NON-PORK"]
        )
        disp.plot(ax=ax, cmap="Blues", values_format="d")
        ax.set_title("Confusion Matrix (Sample)")
        st.pyplot(fig)
        plt.close(fig)
        
        st.markdown("### 📌 Sample Classification Report")
        st.dataframe({
            "Class": ["PORK", "NON-PORK"],
            "Precision": [f"{report['PORK']['precision']:.4f}", f"{report['NON-PORK']['precision']:.4f}"],
            "Recall": [f"{report['PORK']['recall']:.4f}", f"{report['NON-PORK']['recall']:.4f}"],
            "F1-Score": [f"{report['PORK']['f1-score']:.4f}", f"{report['NON-PORK']['f1-score']:.4f}"],
            "Support": [report['PORK']['support'], report['NON-PORK']['support']]
        }, use_container_width=True)
        
    else:
        st.success(f"✅ Found {len(X_test)} test images!")
        
        if st.button("📊 Run Evaluation", type="primary"):
            with st.spinner("Evaluating model on test images..."):
                y_prob = model.predict(X_test, verbose=0).reshape(-1)
                y_pred = (y_prob >= THRESHOLD).astype(int)
                
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                auc = roc_auc_score(y_true, y_prob)
                cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                report = classification_report(
                    y_true, y_pred,
                    target_names=["PORK", "NON-PORK"], output_dict=True
                )
            
            st.session_state.eval_results = {
                "accuracy": accuracy, "precision": precision,
                "recall": recall, "f1": f1, "auc": auc,
                "cm": cm, "report": report
            }

    if "eval_results" in st.session_state:
        r = st.session_state.eval_results
        
        st.markdown("### 📊 Evaluation Metrics")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", f"{r['accuracy']*100:.2f}%")
        m2.metric("Precision", f"{r['precision']*100:.2f}%")
        m3.metric("Recall", f"{r['recall']*100:.2f}%")
        m4.metric("F1-Score", f"{r['f1']*100:.2f}%")
        m5.metric("AUC-ROC", f"{r['auc']:.4f}")
        
        st.markdown("### 📉 Confusion Matrix")
        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=r["cm"],
            display_labels=["PORK", "NON-PORK"]
        )
        disp.plot(ax=ax, cmap="Blues", values_format="d")
        ax.set_title("Validation Confusion Matrix")
        st.pyplot(fig)
        plt.close(fig)
        
        st.markdown("### 📋 Classification Report")
        st.dataframe({
            "Class": ["PORK", "NON-PORK"],
            "Precision": [f"{r['report']['PORK']['precision']:.4f}", f"{r['report']['NON-PORK']['precision']:.4f}"],
            "Recall": [f"{r['report']['PORK']['recall']:.4f}", f"{r['report']['NON-PORK']['recall']:.4f}"],
            "F1-Score": [f"{r['report']['PORK']['f1-score']:.4f}", f"{r['report']['NON-PORK']['f1-score']:.4f}"],
            "Support": [r['report']['PORK']['support'], r['report']['NON-PORK']['support']]
        }, use_container_width=True)

# -------------------------- FOOTER --------------------------

st.divider()
st.caption("🐷 PORK vs NON-PORK Classification System | MobileNetV2 | Streamlit")
