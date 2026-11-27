import streamlit as st
from PIL import Image
import numpy as np
import os
import time

# ───────────────────────────
# FIX: Use lightweight tflite-runtime instead of full TensorFlow
# ───────────────────────────
try:
    from tflite_runtime.interpreter import Interpreter
except (ImportError, ModuleNotFoundError):
    try:
        from tensorflow.lite.python.interpreter import Interpreter
        st.warning("⚠️ Using TensorFlow — app may be slower to start")
    except Exception as e:
        st.error(f"❌ Could not load any interpreter: {e}")
        st.stop()

# ───────────────────────────
# Page Setup
# ───────────────────────────
st.set_page_config(page_title="PORK vs NON-PORK CLASSIFICATION", page_icon="🐷", layout="wide")
st.title("🐷 PORK vs NON-PORK CLASSIFICATION SYSTEM")

# ───────────────────────────
# File Paths (Safe for Streamlit Cloud)
# ───────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_unquant.tflite")
LABELS_PATH = os.path.join(BASE_DIR, "labels.txt")

# ───────────────────────────
# Load Model & Labels
# ───────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model not found at: {MODEL_PATH}")
        st.info("💡 Make sure 'model_unquant.tflite' is in your GitHub repo root folder")
        st.stop()
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

def load_labels():
    if not os.path.exists(LABELS_PATH):
        st.error(f"❌ Labels not found at: {LABELS_PATH}")
        st.stop()
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]

interpreter = load_model()
labels = load_labels()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']  # [1, height, width, channels]
_, HEIGHT, WIDTH, _ = input_shape

# ───────────────────────────
# Sidebar: Model Info & Evaluation Section
# ───────────────────────────
with st.sidebar:
    st.header("📊 Model Information")
    st.info(f"Input Size: {WIDTH}×{HEIGHT} px")
    st.info(f"Classes: {', '.join(labels)}")
    st.info(f"Model: Unquantized TFLite")
    
    st.divider()
    st.header("📈 Performance & Evaluation")
    
    # Pre-computed metrics — UPDATE THESE WITH YOUR ACTUAL TEST RESULTS!
    st.subheader("Overall Metrics")
    
    overall_accuracy = 94.2
    precision_scores = [93.5, 95.0]
    recall_scores    = [92.0, 96.0]
    f1_scores        = [92.7, 95.5]
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", f"{overall_accuracy:.1f}%")
        st.metric("Avg Precision", f"{np.mean(precision_scores):.1f}%")
    with col2:
        st.metric("Avg Recall", f"{np.mean(recall_scores):.1f}%")
        st.metric("Avg F1-Score", f"{np.mean(f1_scores):.1f}%")
    
    st.divider()
    st.subheader("Per-Class Performance")
    
    perf_data = {
        "Class": labels,
        "Precision (%)": [f"{p:.1f}" for p in precision_scores],
        "Recall (%)":    [f"{r:.1f}" for r in recall_scores],
        "F1-Score (%)":  [f"{f:.1f}" for f in f1_scores]
    }
    st.table(perf_data)
    
    st.divider()
    st.caption("ℹ️ Update these values with your actual test dataset results")

# ───────────────────────────
# Confusion Matrix — FIXED & SAFE VERSION
# ───────────────────────────
st.subheader("📉 Confusion Matrix")
st.caption("How well the model classifies each category")

# Make sure we have exactly 2 labels
if len(labels) >= 2:
    label_0 = labels[0]
    label_1 = labels[1]
else:
    label_0 = "Non-Pork"
    label_1 = "Pork"
    st.warning("⚠️ labels.txt has fewer than 2 labels — using defaults")

# ┌── REPLACE THESE WITH YOUR ACTUAL TEST RESULTS ──
confusion_matrix = np.array([
    [46, 3],   # Non-Pork: 46 correct, 3 misclassified as Pork
    [2, 49]    # Pork: 49 correct, 2 misclassified as Non-Pork
])

cm_normalized = confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True)

fig_cm = f"""
<style>
.cm-table {{ border-collapse: collapse; width: 100%; font-family: sans-serif; }}
.cm-table td, .cm-table th {{ border: 1px solid #ddd; padding: 12px; text-align: center; }}
.cm-table th {{ background-color: #f0f2f6; }}
</style>
<table class="cm-table">
  <tr>
    <th></th>
    <th colspan="2">🔮 Predicted</th>
  </tr>
  <tr>
    <th rowspan="2">✅ Actual</th>
    <th>{label_0}</th>
    <th>{label_1}</th>
  </tr>
  <tr>
    <td style="background-color: rgba(144, 238, 144, {cm_normalized[0,0]:.4f}); font-weight: bold;">
      {confusion_matrix[0,0]}<br><small>{cm_normalized[0,0]*100:.1f}%</small><br><span style="color:green">✓ Correct</span>
    </td>
    <td style="background-color: rgba(255, 160, 122, {cm_normalized[0,1]:.4f});">
      {confusion_matrix[0,1]}<br><small>{cm_normalized[0,1]*100:.1f}%</small><br><span style="color:red">✗ Error</span>
    </td>
  </tr>
  <tr>
    <th>{label_1}</th>
    <td style="background-color: rgba(255, 160, 122, {cm_normalized[1,0]:.4f});">
      {confusion_matrix[1,0]}<br><small>{cm_normalized[1,0]*100:.1f}%</small><br><span style="color:red">✗ Error</span>
    </td>
    <td style="background-color: rgba(144, 238, 144, {cm_normalized[1,1]:.4f}); font-weight: bold;">
      {confusion_matrix[1,1]}<br><small>{cm_normalized[1,1]*100:.1f}%</small><br><span style="color:green">✓ Correct</span>
    </td>
  </tr>
</table>
"""
st.markdown(fig_cm, unsafe_allow_html=True)

st.info("""
**Reading the matrix:**
- 🟢 **Diagonal** = Correct predictions
- 🟠 **Off-diagonal** = Misclassifications/Errors
- Higher diagonal values = Better performance
""")


# ───────────────────────────
# HELPER: Prediction Function (reusable)
# ───────────────────────────
def predict_image(image):
    """Run model prediction on a PIL Image"""
    start_time = time.time()
    
    img = image.resize((WIDTH, HEIGHT))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    
    inference_time = (time.time() - start_time) * 1000
    return predictions, inference_time

# ───────────────────────────
# 🆕 SECTION 1: Sample Test Image (Built-in Demo)
# ───────────────────────────
st.subheader("🧪 Demo: Sample Test Image")
st.info("💡 Upload a file named `sample_test.jpg` to your GitHub repo root to use this demo section")

SAMPLE_IMAGE_PATH = os.path.join(BASE_DIR, "sample_test.jpg")

if os.path.exists(SAMPLE_IMAGE_PATH):
    sample_img = Image.open(SAMPLE_IMAGE_PATH).convert("RGB")
    
    col_demo1, col_demo2 = st.columns([1, 1])
    
    with col_demo1:
        st.image(sample_img, caption="🖼️ Sample Test Image", use_column_width=True)
    
    with col_demo2:
        if st.button("🔄 Run Prediction on Sample Image", type="primary"):
            with st.spinner("Analyzing..."):
                predictions, inf_time = predict_image(sample_img)
                
                pred_idx = int(np.argmax(predictions))
                pred_label = labels[pred_idx]
                confidence_val = float(predictions[pred_idx]) * 100
                
                st.success(f"🎯 **RESULT: {pred_label.upper()}** ({confidence_val:.2f}% confidence)")
                st.caption(f"⚡ Inference time: {inf_time:.1f} ms")
                
                st.subheader("📊 Confidence Scores")
                chart_data = {"Class": labels, "Confidence (%)": [float(p)*100 for p in predictions]}
                st.bar_chart(chart_data, x="Class", y="Confidence (%)", color="#ff4b5c")
                
                for i, label in enumerate(labels):
                    conf = float(predictions[i]) * 100
                    st.write(f"**{label}**: {conf:.2f}%")
                    st.progress(float(predictions[i]))
else:
    st.warning("⚠️ Sample image not found — add `sample_test.jpg` to your GitHub repo to enable the demo!")

st.divider()

# ───────────────────────────
# 🆕 SECTION 2: Upload Your Own Image
# ───────────────────────────
st.subheader("🔍 Test With Your Own Image")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="📷 Uploaded Image", use_column_width=True)
    
    with col2:
        with st.spinner("Analyzing..."):
            predictions, inf_time = predict_image(image)
            
            pred_idx = int(np.argmax(predictions))
            pred_label = labels[pred_idx]
            confidence_val = float(predictions[pred_idx]) * 100
            
            st.success(f"🎯 **VERDICT: {pred_label.upper()}** ({confidence_val:.2f}% confidence)")
            st.caption(f"⚡ Inference time: {inf_time:.1f} ms")
            
            st.subheader("📊 Confidence Scores")
            chart_data = {"Class": labels, "Confidence (%)": [float(p)*100 for p in predictions]}
            st.bar_chart(chart_data, x="Class", y="Confidence (%)", color="#ff4b5c")
            
            for i, label in enumerate(labels):
                conf = float(predictions[i]) * 100
                st.write(f"**{label}**: {conf:.2f}%")
                st.progress(float(predictions[i]))
