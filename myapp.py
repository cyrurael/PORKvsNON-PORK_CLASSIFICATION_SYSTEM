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
    
    # Pre-computed / simulated metrics — update these with YOUR actual test results!
    st.subheader("Overall Metrics")
    
    # ┌── YOU SHOULD UPDATE THESE VALUES WITH YOUR OWN TEST DATA ──
    # Example values shown — replace with your actual results!
    overall_accuracy = 94.2
    precision_scores = [93.5, 95.0]   # [Class 0, Class 1]
    recall_scores    = [92.0, 96.0]   # [Class 0, Class 1]
    f1_scores        = [92.7, 95.5]   # [Class 0, Class 1]
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", f"{overall_accuracy:.1f}%")
        st.metric("Avg Precision", f"{np.mean(precision_scores):.1f}%")
    with col2:
        st.metric("Avg Recall", f"{np.mean(recall_scores):.1f}%")
        st.metric("Avg F1-Score", f"{np.mean(f1_scores):.1f}%")
    
    st.divider()
    st.subheader("Per-Class Performance")
    
    # Per-class table
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
# Main Area: Confusion Matrix Visualization
# ───────────────────────────
st.subheader("📉 Confusion Matrix")
st.caption("How well the model classifies each category")

# ┌── REPLACE THESE WITH YOUR ACTUAL TEST RESULTS ──
# Format: [[True Negative, False Positive],
#          [False Negative, True Positive]]
# Example values — update with your counts!
confusion_matrix = np.array([
    [46, 3],   # Non-Pork: 46 correct, 3 misclassified as Pork
    [2, 49]    # Pork: 49 correct, 2 misclassified as Non-Pork
])

# Normalize for color intensity
cm_normalized = confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True)

# Draw confusion matrix manually
fig_cm = """
<style>
.cm-table { border-collapse: collapse; width: 100%; font-family: sans-serif; }
.cm-table td, .cm-table th { border: 1px solid #ddd; padding: 12px; text-align: center; }
.cm-table th { background-color: #f0f2f6; }
</style>
<table class="cm-table">
  <tr>
    <th></th>
    <th colspan="2">🔮 Predicted</th>
  </tr>
  <tr>
    <th rowspan="2">✅ Actual</th>
    <th>{}</th>
    <th>{}</th>
  </tr>
  <tr>
    <td style="background-color: rgba(144, 238, 144, {:.4}); font-weight: bold;">
      {}<br><small>{:.1f}%</small><br><span style="color:green">✓ Correct</span>
    </td>
    <td style="background-color: rgba(255, 160, 122, {:.4});">
      {}<br><small>{:.1f}%</small><br><span style="color:red">✗ Error</span>
    </td>
  </tr>
  <tr>
    <th>{}</th>
    <td style="background-color: rgba(255, 160, 122, {:.4});">
      {}<br><small>{:.1f}%</small><br><span style="color:red">✗ Error</span>
    </td>
    <td style="background-color: rgba(144, 238, 144, {:.4}); font-weight: bold;">
      {}<br><small>{:.1f}%</small><br><span style="color:green">✓ Correct</span>
    </td>
  </tr>
</table>
""".format(
    labels[0], labels[1],
    confusion_matrix[0,0], cm_normalized[0,0]*100,
    confusion_matrix[0,1], cm_normalized[0,1]*100,
    labels[1],
    confusion_matrix[1,0], cm_normalized[1,0]*100,
    confusion_matrix[1,1], cm_normalized[1,1]*100
)
st.markdown(fig_cm, unsafe_allow_html=True)

st.info("""
**Reading the matrix:**
- 🟢 **Diagonal** = Correct predictions
- 🟠 **Off-diagonal** = Misclassifications/Errors
- Higher diagonal values = Better performance
""")

st.divider()

# ───────────────────────────
# Image Upload & Prediction Section
# ───────────────────────────
st.subheader("🔍 Test the Model")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Display image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="📷 Uploaded Image", use_column_width=True)
    
    with col2:
        # Preprocess and time it
        start_time = time.time()
        
        img = image.resize((WIDTH, HEIGHT))
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
        
        inference_time = (time.time() - start_time) * 1000
        
        # Show predictions
        st.subheader("📊 Results")
        
        results = []
        for i, label in enumerate(labels):
            confidence = float(predictions[i]) * 100
            results.append({"Class": label, "Confidence (%)": round(confidence, 2)})
            st.write(f"**{label}**: {confidence:.2f}%")
            st.progress(float(predictions[i]))
        
        # Final verdict
        predicted_idx = int(np.argmax(predictions))
        predicted_label = labels[predicted_idx]
        confidence_val = float(predictions[predicted_idx]) * 100
        
        st.success(f"🎯 **VERDICT: {predicted_label.upper()}** ({confidence_val:.2f}% confidence)")
        st.caption(f"⚡ Inference time: {inference_time:.1f} ms")
        
        # Confidence bar chart
        st.subheader("📈 Confidence Distribution")
        chart_data = {
            "Class": labels,
            "Confidence (%)": [float(p)*100 for p in predictions]
        }
        st.bar_chart(chart_data, x="Class", y="Confidence (%)", color="#ff4b5c")
