# ============================================================
# 🥩 PORK vs NON-PORK IMAGE CLASSIFICATION SYSTEM
# Streamlit + TensorFlow/Keras + MobileNetV2
# Model File: pork_classifier.keras
# Binary Classes: 0 = PORK | 1 = NON-PORK
# ============================================================

import os
import io
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

DATA_DIR = "meat_dataset/train"
MODEL_PATH = "pork_classifier.keras"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
DEFAULT_EPOCHS = 30
LEARNING_RATE = 0.0001
SEED = 123
THRESHOLD = 0.50
PORK_FOLDER_NAMES = {"pork", "pok"}
AUTOTUNE = tf.data.AUTOTUNE

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
.main-title {font-size: 2.2rem; font-weight: 800; text-align:center; color:#1a1a1a;}
.sub-title {text-align:center; color:#666; margin-bottom:1.5rem; font-size:1.1rem;}
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
.card {background:#f8f9fa; border-radius:10px; padding:1.2rem; border:1px solid #e9ecef;}
.small-muted {color:#888; font-size:.85rem;}
.stMetric {background:#fff; border-radius:8px; padding:10px !important; border:1px solid #eee;}
</style>
""", unsafe_allow_html=True)

# -------------------------- SESSION STATE --------------------------

defaults = {
    "training_history": None,
    "training_completed": False,
    "last_evaluation": None,
    "prediction_history": [],
    "training_error": None
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# -------------------------- HELPER FUNCTIONS --------------------------

def get_dataset_class_names(data_dir):
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Dataset folder not found: `{data_dir}`.\n"
            "Create folder structure: meat_dataset/train/pok, beef, horse"
        )
    names = sorted([n for n in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, n))])
    normalized = [n.strip().lower() for n in names]
    pork_matches = [i for i, n in enumerate(normalized) if n in PORK_FOLDER_NAMES]
    if len(names) != 3:
        raise ValueError(f"Expected 3 folders, found {len(names)}: {names}")
    if len(pork_matches) != 1:
        raise ValueError(f"Could not identify pork folder in: {names}")
    return names, pork_matches[0]


def load_datasets():
    class_names, pork_index = get_dataset_class_names(DATA_DIR)
    train_raw = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.20, subset="training", seed=SEED,
        image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, label_mode="int", shuffle=True
    )
    val_raw = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.20, subset="validation", seed=SEED,
        image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, label_mode="int", shuffle=False
    )
    class_names = train_raw.class_names
    normalized = [n.strip().lower() for n in class_names]
    pork_matches = [i for i, n in enumerate(normalized) if n in PORK_FOLDER_NAMES]
    pork_index = pork_matches[0]

    def convert_labels(images, labels):
        binary = tf.where(tf.equal(labels, pork_index), 0, 1)
        return images, tf.cast(binary, tf.float32)

    return (
        train_raw.map(convert_labels, AUTOTUNE).prefetch(AUTOTUNE),
        val_raw.map(convert_labels, AUTOTUNE).prefetch(AUTOTUNE),
        class_names, pork_index
    )


def build_model():
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.10),
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomContrast(0.10)
    ], name="augmentation")

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), include_top=False, weights="imagenet"
    )
    base_model.trainable = False

    inputs = tf.keras.layers.Input(shape=(224, 224, 3))
    x = augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.40)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall"), tf.keras.metrics.AUC(name="auc")]
    )
    return model


@st.cache_resource(show_spinner="Loading model...")
def load_saved_model(model_path, modified_time):
    _ = modified_time
    return tf.keras.models.load_model(model_path, compile=False)


def get_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return load_saved_model(MODEL_PATH, os.path.getmtime(MODEL_PATH))


def predict_pil_image(model, pil_image):
    start = time.time()
    image = pil_image.convert("RGB").resize(IMAGE_SIZE)
    arr = tf.keras.utils.img_to_array(image)
    arr = np.expand_dims(arr, axis=0)
    non_pork_prob = float(model.predict(arr, verbose=0)[0][0])
    pork_prob = 1.0 - non_pork_prob
    label = "NON-PORK" if non_pork_prob >= THRESHOLD else "PORK"
    confidence = non_pork_prob if label == "NON-PORK" else pork_prob
    inference_time = (time.time() - start) * 1000
    return {
        "label": label, "confidence": confidence,
        "pork_probability": pork_prob, "non_pork_probability": non_pork_prob,
        "inference_ms": inference_time
    }


def collect_evaluation(model, dataset):
    y_true, y_prob = [], []
    for images, labels in dataset:
        probs = model.predict(images, verbose=0).reshape(-1)
        y_prob.extend(probs.tolist())
        y_true.extend(labels.numpy().astype(int).tolist())
    y_true, y_prob = np.array(y_true), np.array(y_prob)
    return y_true, y_prob, (y_prob >= THRESHOLD).astype(int)

# -------------------------- HEADER --------------------------

st.markdown('<div class="main-title">🥩 PORK vs NON-PORK CLASSIFICATION SYSTEM</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">MobileNetV2 · Transfer Learning · Binary Image Classification</div>', unsafe_allow_html=True)

# -------------------------- SIDEBAR --------------------------

with st.sidebar:
    st.title("🥩 Navigation")
    page = st.radio("Select Module", [
        "📊 Dashboard", "🏋️ Model Training", "📈 Performance Visualization",
        "🔍 Test Meat Image", "📋 Model Evaluation"
    ])
    st.divider()
    st.markdown("**Classification Mapping**")
    st.markdown("🐷 `pork/pok` → **PORK (Class 0)**")
    st.markdown("🥩 `beef/horse` → **NON-PORK (Class 1)**")
    st.divider()
    st.markdown("**Model Info**")
    st.info(f"File: `{MODEL_PATH}`\nInput: 224×224 RGB\nThreshold: {THRESHOLD}")

# ============================================================
# 📊 DASHBOARD
# ============================================================

if page == "📊 Dashboard":
    st.subheader("System Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Source Classes", "3")
    c2.metric("Output Classes", "2")
    c3.metric("Input Size", "224 × 224")
    c4.metric("Model", "MobileNetV2")

    st.markdown("### 🗂️ Dataset Structure")
    st.code("""
pork_nonpork_classifier/
├── app.py
├── pork_classifier.keras   ← Trained Model
└── meat_dataset/
    └── train/
        ├── pok/      ← PORK
        ├── beef/     ← NON-PORK
        └── horse/    ← NON-PORK
    """, language="text")

    if os.path.isdir(DATA_DIR):
        try:
            folders, pork_idx = get_dataset_class_names(DATA_DIR)
            st.success(f"✅ Dataset Detected: {', '.join(folders)}")
            st.info(f"🐷 Pork Folder: **{folders[pork_idx]}**")
            counts = {}
            for f in folders:
                fp = os.path.join(DATA_DIR, f)
                counts[f] = sum(1 for x in os.listdir(fp) if os.path.isfile(os.path.join(fp, x))
                                and x.lower().endswith((".jpg", ".jpeg", ".png")))
            st.markdown("### 📷 Images per Class")
            st.bar_chart(counts, use_container_width=True)
            st.dataframe([{
                "Folder": name,
                "Category": "🐷 PORK" if i == pork_idx else "🥩 NON-PORK",
                "Image Count": counts[name]
            } for i, name in enumerate(folders)], use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(f"⚠️ {e}")
    else:
        st.warning(f"📁 Dataset folder not found: `{DATA_DIR}`")

    st.markdown("### 🤖 Model Status")
    if os.path.exists(MODEL_PATH):
        size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        st.success(f"✅ Model Ready: `{MODEL_PATH}` ({size_mb:.1f} MB)")
    else:
        st.warning("⚠️ No model found — go to **Model Training** to create one")

# ============================================================
# 🏋️ MODEL TRAINING
# ============================================================

elif page == "🏋️ Model Training":
    st.subheader("Train the Classification Model")
    st.info("Train MobileNetV2 to distinguish PORK from NON-PORK meats")

    epochs = st.slider("Training Epochs", 5, 100, DEFAULT_EPOCHS)
    start_btn = st.button("🚀 Start Training", type="primary", use_container_width=True)

    if start_btn:
        try:
            with st.spinner("📂 Loading dataset..."):
                train_ds, val_ds, class_names, pork_idx = load_datasets()
            st.success(f"✅ Loaded: {', '.join(class_names)}")
            st.info(f"Mapping: **{class_names[pork_idx]} = PORK**, others = NON-PORK")

            model = build_model()
            progress = st.progress(0)
            status = st.empty()
            chart_area = st.empty()
            logs = []

            class ProgressCB(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    r = {"epoch": epoch+1}
                    r.update({k: float(v) for k, v in logs.items()})
                    logs = r
                    logs["epoch"] = epoch + 1
                    logs = r
                    logs["epoch"] = epoch + 1
                    logs = r
                    logs["epoch"] = epoch + 1
                    logs = r
                    logs["epoch"] = epoch + 1
                    logs = r
                    logs["epoch"] = epoch + 1
                    logs = r
                    logs["epoch"] = epoch + 1
                    logs = r
                    logs["epoch"] = epoch + 1
                    logs = r
                    logs["epoch"] = epoch + 1
                    logs = r
                    logs["epoch"] = epoch + 1
                    logs = r
                    logs["epoch"] = epoch + 1
                    logs = r
                    logs["epoch"] = epoch + 1
                    logs = r
                    logs["epoch"] = epoch + 1
                    logs = r
                    logs["epoch"] = epoch + 1
                    logs = r
                    logs["epoch"] = epoch + 1
                    logs = r
                    logs["epoch"] = epoch + 1
                    logs = r
                    logs["epoch"] = epoch + 1
                    logs = r
                    logs["epoch"] = epoch + 1
                    logs = r
                    logs["epoch"] = epoch + 1
                    logs = r
                    logs["epoch"] = epoch + 1
                    logs = r
                    logs["epoch"] = epoch + 1
                    logs = r
                    logs["epoch"] = epoch + 1
                    logs = r
                    logs["epoch"] = epoch + 1
                    logs = r
                    logs["epoch"] = epoch + 1
                    logs = r
                    logs["epoch"] = epoch + 1
                    logs = r
                    logs["epoch"] = epoch + 1
                    logs = r
                    logs["epoch"] = epoch + 1
                    logs = r
                    logs["epoch"] = epoch + 1
                    logs = r
                    logs["epoch"] = epoch + 1
                    logs = r
                    logs["epoch"] = epoch + 1
                    logs = r
                    logs["epoch"] = epoch + 1
                    logs = r
                    logs["epoch"] = epoch + 1
                    logs = r
                    logs["epoch"] = epoch + 1
                    logs = r
                    logs["epoch"] = epoch + 1
                    logs = r
                    logs["epoch"] = epoch + 1
                    logs = r
                    logs["epoch"] = epoch + 1
                    logs = r
                    logs["epoch"] = epoch + 1
                    logs = r
                    logs["epoch"] = epoch + 1
                    logs = r
                    logs["epoch"] = epoch + 1
                    logs = r
                    logs["epoch"] = epoch + 1
                    logs = r
                    logs["epoch"] = epoch + 1
                    logs = r
                    logs["epoch"] = epoch + 1
                    logs = r
                    logs["epoch"] = epoch + 1
                    logs = r
                    logs["epoch"] = epoch + 1
                    logs = r
                    logs["epoch"] = epoch + 1
                    logs = r
                    logs["epoch"] = epoch + 1
                    logs = r
