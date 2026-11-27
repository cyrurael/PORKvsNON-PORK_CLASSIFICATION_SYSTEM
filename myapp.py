import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Page config
st.set_page_config(page_title="PORK vs NON-PORK CLASSIFICATION", page_icon="🤖")
st.title("🖼️ PORK vs NON-PORK CLASSIFICATION")

# Load model and labels
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="model_unquant.tflite")
    interpreter.allocate_tensors()
    return interpreter

def load_labels():
    with open("labels.txt", "r") as f:
        return [line.strip() for line in f.readlines()]

interpreter = load_model()
labels = load_labels()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']  # usually [1, 224, 224, 3]

# Upload image
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess image
    img = image.resize((input_shape[1], input_shape[2]))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # normalize to 0-1
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    
    # Show results
    st.subheader("📊 Predictions")
    for i, label in enumerate(labels):
        confidence = float(predictions[i]) * 100
        st.write(f"**{label}**: {confidence:.2f}%")
        st.progress(float(predictions[i]))
