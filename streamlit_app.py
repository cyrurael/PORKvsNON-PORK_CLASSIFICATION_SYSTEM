# ============================================================
# 🥩 PORK vs NON-PORK CLASSIFIER — ALL-IN-ONE
# Auto-computes metrics from predictions — no manual input needed
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
    layout="wide"
)

# -------------------------- STYLES --------------------------

st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
.main-title {font-size: 2rem; font-weight: 800; text-align:center;}
.sub-title {text-align:center; color:#666; margin-bottom:1.5rem;}
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

# -------------------------- LOAD MODEL --------------------------

@st.cache_resource(show_spinner="🔄 Loading model...")
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

if model is None:
    st.error(f"❌ Model not found! Place `{MODEL_PATH}` in your project folder.")
    st.stop()

# -------------------------- PREDICTION FUNCTION --------------------------

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

# -------------------------- LOAD & EVALUATE TEST DATA --------------------------

@st.cache_data(show_spinner="📊 Evaluating model performance...")
def get_model_evaluation():
    """Load test images and compute all metrics automatically"""
    if not os.path.exists(TEST_IMAGES_FOLDER):
        return None
    
    X_test, y_true = [], []
    
    # Load PORK images → label 0
    pork_dir = os.path.join(TEST_IMAGES_FOLDER, "PORK")
    if os.path.exists(pork_dir):
        for fname in os.listdir(pork_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                try:
                    path = os.path.join(pork_dir, fname)
                    img = Image.open(path).convert("RGB").resize(IMAGE_SIZE)
                    X_test.append(tf.keras.utils.img_to_array(img))
                    y_true.append(0)
                except: pass
    
    # Load NON-PORK images → label 1
    nonpork_dir = os.path.join(TEST_IMAGES_FOLDER, "NON-PORK")
    if os.path.exists(nonpork_dir):
        for fname in os.listdir(nonpork_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                try:
                    path = os.path.join(nonpork_dir, fname)
                    img = Image.open(path).convert("RGB").resize(IMAGE_SIZE)
                    X_test.append(tf.keras.utils.img_to_array(img))
                    y_true.append(1)
                except: pass
    
    if len(X_test) < 4:  # Need minimum for meaningful metrics
        return None
    
    X_test = np.array(X_test)
    y_true = np.array(y_true)
    
    # Run predictions
    y_prob = model.predict(X_test, verbose=0).reshape(-1)
    y_pred = (y_prob >= THRESHOLD).astype(int)
    
    # Compute all metrics
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_prob),
        "cm": confusion_matrix(y_true, y_pred, labels=[0, 1]),
        "report": classification_report(
            y_true, y_pred, target_names=["PORK", "NON-PORK"], output_dict=True
        ),
        "total_samples": len(y_true),
        "pork_samples": int(sum(y_true == 0)),
        "nonpork_samples": int(sum(y_true == 1))
    }

# Auto-load evaluation data
eval_data = get_model_evaluation()

# -------------------------- MAIN APP — ALL IN ONE PAGE --------------------------

st.markdown('<h1 class="main-title">🥩 PORK vs NON-PORK CLASSIFICATION SYSTEM</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Upload an image to classify it — performance metrics shown below</p>', unsafe_allow_html=True)

# ─────────────── SECTION 1: IMAGE PREDICTION ───────────────

st.subheader("🔍 Image Classification")
uploaded_file = st.file_uploader("📷 Upload an image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file:
    col1, col2 = st.columns([1, 1])
    with col1:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        with st.spinner("Analyzing..."):
            result = predict_image(model, image)
        
        if result["label"] == "PORK":
            st.markdown('<div class="result-pork">🐷 PORK DETECTED</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-nonpork">🥩 NON-PORK</div>', unsafe_allow_html=True)
        
        st.metric("Confidence", f"{result['confidence']*100:.2f}%")
        st.progress(float(result["confidence"]))
        st.write(f"🐷 PORK Probability: **{result['pork_prob']*100:.2f}%**")
        st.write(f"🥩 NON-PORK Probability: **{result['non_pork_prob']*100:.2f}%**")
        st.caption(f"⚡ Inference time: {result['inference_ms']:.1f} ms")
else:
    st.info("👆 Upload an image above to get started")

st.divider()

# ─────────────── SECTION 2: PERFORMANCE OVERVIEW ───────────────

st.subheader("📈 Model Performance Overview")

if eval_data:
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accuracy", f"{eval_data['accuracy']*100:.2f}%")
    col2.metric("Precision", f"{eval_data['precision']*100:.2f}%")
    col3.metric("Recall", f"{eval_data['recall']*100:.2f}%")
    col4.metric("F1-Score", f"{eval_data['f1']*100:.2f}%")
    col5.metric("AUC-ROC", f"{eval_data['auc']:.4f}")
    
    st.caption(f"📊 Based on {eval_data['total_samples']} test images "
               f"({eval_data['pork_samples']} PORK + {eval_data['nonpork_samples']} NON-PORK)")
    
    with st.expander("📖 What these metrics mean"):
        st.write("**Accuracy:** Overall % of correct predictions")
        st.write("**Precision:** How many NON-PORK predictions are actually NON-PORK")
        st.write("**Recall:** How many actual NON-PORK were correctly found")
        st.write("**F1-Score:** Balanced average of Precision & Recall")
        st.write("**AUC-ROC:** Model's ability to distinguish classes (1.0 = perfect)")
else:
    st.warning("⚠️ Test dataset not found. Create folder `test_images/PORK/` and `test_images/NON-PORK/` with test images to auto-display performance metrics.")

st.divider()

# ─────────────── SECTION 3: MODEL EVALUATION ───────────────

st.subheader("📋 Model Evaluation")

if eval_data:
    col_cm, col_report = st.columns([1, 1])
    
    with col_cm:
        st.markdown("### Confusion Matrix")
        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=eval_data["cm"],
            display_labels=["PORK", "NON-PORK"]
        )
        disp.plot(ax=ax, cmap="Blues", values_format="d")
        ax.set_title("Prediction vs Actual Labels")
        st.pyplot(fig)
        plt.close(fig)
        
        st.info("""
        🟦 **Top-Left:** PORK correctly predicted as PORK  
        🟦 **Bottom-Right:** NON-PORK correctly predicted as NON-PORK  
        ⬜ **Off-diagonal:** Misclassifications
        """)
    
    with col_report:
        st.markdown("### Classification Report")
        report_df = [
            {
                "Class": "🐷 PORK",
                "Precision": f"{eval_data['report']['PORK']['precision']:.4f}",
                "Recall": f"{eval_data['report']['PORK']['recall']:.4f}",
                "F1-Score": f"{eval_data['report']['PORK']['f1-score']:.4f}",
                "Support": eval_data['report']['PORK']['support']
            },
            {
                "Class": "🥩 NON-PORK",
                "Precision": f"{eval_data['report']['NON-PORK']['precision']:.4f}",
                "Recall": f"{eval_data['report']['NON-PORK']['recall']:.4f}",
                "F1-Score": f"{eval_data['report']['NON-PORK']['f1-score']:.4f}",
                "Support": eval_data['report']['NON-PORK']['support']
            }
        ]
        st.dataframe(report_df, use_container_width=True, hide_index=True)
else:
    st.info("""
    ### 📁 To enable auto-evaluation:
    
    Create this folder structure in your project:
The app will **automatically compute and display all metrics** when it loads — no buttons to click!
""")

# -------------------------- FOOTER --------------------------

st.divider()
st.caption("🐷 PORK vs NON-PORK Classification System | MobileNetV2 | TensorFlow/Keras | Streamlit")

